"""
Example demonstrating asynchronous and concurrent operations with the API JSON loader.

This example shows how to use the async capabilities for improved performance
when dealing with multiple data sources and large datasets.
"""

import asyncio
import time
import json
import tempfile
import os
from typing import List
import pandas as pd

# Import the async-enabled components
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.application.use_cases.data_api_json_use_case import DataAPIJSONUseCase


class MockEmbeddingService:
    """Mock embedding service for demonstration."""
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings."""
        # Simulate some processing time
        time.sleep(0.1)
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockRepository:
    """Mock repository for demonstration."""
    
    async def table_exists(self, table_name: str) -> bool:
        await asyncio.sleep(0.05)  # Simulate database query
        return False
    
    async def create_table(self, **kwargs):
        await asyncio.sleep(0.1)  # Simulate table creation
        print(f"Created table: {kwargs.get('table_name', 'unknown')}")
    
    async def insert_data(self, table_name: str, df: pd.DataFrame, pk_columns: List[str]):
        await asyncio.sleep(0.05)  # Simulate data insertion
        print(f"Inserted {len(df)} rows into {table_name}")
        return len(df)


def create_sample_json_files() -> List[str]:
    """Create sample JSON files for testing."""
    files = []
    
    try:
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                data = [
                    {
                        "id": j + i * 100,
                        "title": f"Article {j} from Source {i}",
                        "content": f"This is the content of article {j} from data source {i}. " * 5,
                        "category": f"category_{i}",
                        "author": f"Author {j % 10}",
                        "published_date": f"2024-01-{(j % 28) + 1:02d}"
                    }
                    for j in range(1, 101)  # 100 articles per file
                ]
                json.dump(data, f)
                files.append(f.name)
                print(f"Created sample file: {f.name} with {len(data)} records")
        
        return files
    
    except Exception as e:
        # Clean up on error
        for file_path in files:
            try:
                os.unlink(file_path)
            except OSError:
                pass
        raise e


async def example_1_basic_async_loading():
    """Example 1: Basic async loading from a single source."""
    print("\n=== Example 1: Basic Async Loading ===")
    
    # Create a sample dataset
    sample_data = [
        {"id": i, "name": f"Item {i}", "value": i * 10}
        for i in range(1, 101)
    ]
    
    # Initialize the async loader
    loader = APIJSONStorageLoader()
    
    start_time = time.time()
    
    # Load data asynchronously
    df = await loader.load_json(sample_data)
    
    execution_time = time.time() - start_time
    
    print(f"Loaded {len(df)} rows in {execution_time:.3f} seconds")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample data:\n{df.head()}")


async def example_2_concurrent_file_loading():
    """Example 2: Concurrent loading from multiple JSON files."""
    print("\n=== Example 2: Concurrent File Loading ===")
    
    # Create sample files
    files = create_sample_json_files()
    
    try:
        loader = APIJSONStorageLoader()
        
        # Sequential loading (for comparison)
        print("Sequential loading:")
        start_time = time.time()
        sequential_dfs = []
        for file_path in files:
            df = await loader.load_json(file_path)
            sequential_dfs.append(df)
        sequential_time = time.time() - start_time
        sequential_total_rows = sum(len(df) for df in sequential_dfs)
        
        print(f"Sequential: {sequential_total_rows} rows in {sequential_time:.3f} seconds")
        
        # Concurrent loading
        print("\nConcurrent loading:")
        start_time = time.time()
        concurrent_df = await loader.load_json_concurrent(files, max_concurrent=3)
        concurrent_time = time.time() - start_time
        
        print(f"Concurrent: {len(concurrent_df)} rows in {concurrent_time:.3f} seconds")
        print(f"Speedup: {sequential_time / concurrent_time:.2f}x")
        
        # Verify data integrity
        categories = concurrent_df['category'].unique()
        print(f"Categories found: {sorted(categories)}")
        
    finally:
        # Clean up files
        for file_path in files:
            try:
                os.unlink(file_path)
            except OSError:
                pass


async def example_3_async_use_case_with_embeddings():
    """Example 3: Async use case with embedding generation."""
    print("\n=== Example 3: Async Use Case with Embeddings ===")
    
    # Create sample data
    articles_data = [
        {
            "id": i,
            "title": f"Breaking News Article {i}",
            "content": f"This is the detailed content of news article {i}. " * 20,
            "author": f"Reporter {i % 5}",
            "category": ["politics", "technology", "sports", "health", "business"][i % 5]
        }
        for i in range(1, 201)  # 200 articles
    ]
    
    # Setup components
    mock_repo = MockRepository()
    mock_embedding_service = MockEmbeddingService()
    loader = APIJSONStorageLoader()
    
    # Create use case
    use_case = DataAPIJSONUseCase(mock_repo, mock_embedding_service, loader)
    
    start_time = time.time()
    
    # Execute with async context manager for proper cleanup
    async with use_case:
        result = await use_case.execute(
            source=articles_data,
            table_name="news_articles",
            embed_columns_names=["title", "content"],
            embed_type="combined",
            create_table_if_not_exists=True
        )
    
    execution_time = time.time() - start_time
    
    print(f"Processed {result.rows_processed} rows in {execution_time:.3f} seconds")
    print(f"Operation successful: {result.success}")
    print(f"Table created: {result.table_created}")
    print(f"Warnings: {len(result.warnings)}")


async def example_4_concurrent_use_cases():
    """Example 4: Running multiple use cases concurrently."""
    print("\n=== Example 4: Concurrent Use Cases ===")
    
    async def process_dataset(dataset_id: int, num_items: int = 100):
        """Process a single dataset."""
        data = [
            {
                "id": i + dataset_id * 1000,
                "title": f"Dataset {dataset_id} Item {i}",
                "description": f"Description for item {i} in dataset {dataset_id}",
                "category": f"category_{dataset_id}",
                "value": i * (dataset_id + 1)
            }
            for i in range(1, num_items + 1)
        ]
        
        # Setup components for this use case
        mock_repo = MockRepository()
        mock_embedding_service = MockEmbeddingService()
        loader = APIJSONStorageLoader()
        
        use_case = DataAPIJSONUseCase(mock_repo, mock_embedding_service, loader)
        
        async with use_case:
            result = await use_case.execute(
                source=data,
                table_name=f"dataset_{dataset_id}",
                embed_columns_names=["title", "description"],
                embed_type="separated",
                create_table_if_not_exists=True
            )
        
        return result
    
    # Sequential processing (for comparison)
    print("Sequential processing:")
    start_time = time.time()
    sequential_results = []
    for i in range(5):
        result = await process_dataset(i)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    sequential_total_rows = sum(r.rows_processed for r in sequential_results)
    print(f"Sequential: {sequential_total_rows} rows in {sequential_time:.3f} seconds")
    
    # Concurrent processing
    print("\nConcurrent processing:")
    start_time = time.time()
    tasks = [process_dataset(i) for i in range(5, 10)]
    concurrent_results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start_time
    
    concurrent_total_rows = sum(r.rows_processed for r in concurrent_results)
    print(f"Concurrent: {concurrent_total_rows} rows in {concurrent_time:.3f} seconds")
    print(f"Speedup: {sequential_time / concurrent_time:.2f}x")
    
    # Verify all operations succeeded
    all_successful = all(r.success for r in concurrent_results)
    print(f"All operations successful: {all_successful}")


async def example_5_memory_optimized_large_dataset():
    """Example 5: Memory-optimized processing of large datasets."""
    print("\n=== Example 5: Memory-Optimized Large Dataset Processing ===")
    
    # Create a large dataset that will trigger chunked processing
    print("Creating large dataset...")
    large_data = [
        {
            "id": i,
            "title": f"Large Dataset Item {i}",
            "content": f"This is a long content field for item {i}. " * 50,  # ~2KB per item
            "metadata": {
                "created_at": f"2024-01-{(i % 28) + 1:02d}",
                "tags": [f"tag_{j}" for j in range(i % 5 + 1)],
                "nested_data": {
                    "level1": {"level2": {"value": i * 10}}
                }
            }
        }
        for i in range(1, 5001)  # 5000 items, ~10MB total
    ]
    
    print(f"Created dataset with {len(large_data)} items")
    
    # Setup components
    mock_repo = MockRepository()
    mock_embedding_service = MockEmbeddingService()
    loader = APIJSONStorageLoader()
    
    use_case = DataAPIJSONUseCase(mock_repo, mock_embedding_service, loader)
    
    start_time = time.time()
    
    async with use_case:
        result = await use_case.execute(
            source=large_data,
            table_name="large_dataset",
            embed_columns_names=["title"],  # Only embed title to reduce processing time
            embed_type="combined",
            create_table_if_not_exists=True,
            # Configuration for JSON processing
            flatten_nested=True,
            separator="_",
            max_depth=3
        )
    
    execution_time = time.time() - start_time
    
    print(f"Processed {result.rows_processed} rows in {execution_time:.3f} seconds")
    print(f"Average processing rate: {result.rows_processed / execution_time:.1f} rows/second")
    print(f"Operation successful: {result.success}")
    print(f"Warnings: {len(result.warnings)}")


async def main():
    """Run all examples."""
    print("Async and Concurrent Operations Examples")
    print("=" * 50)
    
    try:
        await example_1_basic_async_loading()
        await example_2_concurrent_file_loading()
        await example_3_async_use_case_with_embeddings()
        await example_4_concurrent_use_cases()
        await example_5_memory_optimized_large_dataset()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())