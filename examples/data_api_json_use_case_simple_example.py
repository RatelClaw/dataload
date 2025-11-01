"""
Simple example usage of DataAPIJSONUseCase for loading API/JSON data with embeddings.

This example demonstrates basic usage of the DataAPIJSONUseCase with mock implementations.
"""

import asyncio
import pandas as pd
from typing import List

# Import the use case
from dataload.application.use_cases.data_api_json_use_case import DataAPIJSONUseCase


# Simple mock implementations for demonstration
class SimpleMockRepository:
    """Simple mock repository for example purposes."""
    def __init__(self):
        self.tables = {}
        
    async def table_exists(self, table_name: str) -> bool:
        return table_name in self.tables
        
    async def create_table(self, table_name: str, df, pk_columns: List[str], embed_type: str, embed_columns_names: List[str]):
        self.tables[table_name] = df.copy()
        return {col: 'text' for col in df.columns}
        
    async def insert_data(self, table_name: str, df, pk_columns: List[str]):
        if table_name in self.tables:
            self.tables[table_name] = df.copy()


class SimpleMockEmbeddingProvider:
    """Simple mock embedding provider for example purposes."""
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Generate simple mock embeddings
        return [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(len(texts))]


class SimpleMockStorageLoader:
    """Simple mock storage loader for example purposes."""
    def __init__(self, **kwargs):
        pass
        
    def load_json(self, source, config=None):
        # Return simple mock data based on source
        if isinstance(source, list):
            # Raw JSON data
            return pd.DataFrame(source)
        else:
            # Default mock data
            return pd.DataFrame({
                "id": [1, 2, 3],
                "title": ["Post 1", "Post 2", "Post 3"],
                "content": ["Content 1", "Content 2", "Content 3"],
                "category": ["Tech", "Science", "Tech"]
            })
        
    def validate_config(self, config):
        return []


async def simple_example():
    """Simple example of using DataAPIJSONUseCase."""
    print("=== Simple DataAPIJSONUseCase Example ===")
    
    # Set up mock components
    repository = SimpleMockRepository()
    embedding_service = SimpleMockEmbeddingProvider()
    storage_loader = SimpleMockStorageLoader()
    
    # Create use case
    use_case = DataAPIJSONUseCase(
        repo=repository,
        embedding_service=embedding_service,
        storage_loader=storage_loader
    )
    
    try:
        # Example 1: Load raw JSON data with embeddings
        print("\n--- Example 1: Raw JSON Data ---")
        
        raw_data = [
            {"id": 1, "name": "Alice", "bio": "Software engineer with 5 years experience"},
            {"id": 2, "name": "Bob", "bio": "Data scientist specializing in ML"},
            {"id": 3, "name": "Carol", "bio": "Product manager with technical background"}
        ]
        
        result = await use_case.execute(
            source=raw_data,
            table_name="users",
            embed_columns_names=["name", "bio"],
            embed_type="combined",
            pk_columns=["id"]
        )
        
        print(f"✓ Success! Processed {result.rows_processed} rows")
        print(f"  - Table created: {result.table_created}")
        print(f"  - Execution time: {result.execution_time:.2f}s")
        
        # Show final data
        if "users" in repository.tables:
            final_data = repository.tables["users"]
            print(f"  - Final columns: {list(final_data.columns)}")
        
        # Example 2: Load with separated embeddings
        print("\n--- Example 2: Separated Embeddings ---")
        
        result2 = await use_case.execute(
            source="mock_api_endpoint",  # Will use default mock data
            table_name="posts",
            embed_columns_names=["title", "content"],
            embed_type="separated",
            pk_columns=["id"]
        )
        
        print(f"✓ Success! Processed {result2.rows_processed} rows")
        print(f"  - Table created: {result2.table_created}")
        
        # Show embedding columns
        if "posts" in repository.tables:
            final_data = repository.tables["posts"]
            embedding_cols = [col for col in final_data.columns if col.endswith('_enc')]
            print(f"  - Embedding columns: {embedding_cols}")
        
        # Example 3: No embeddings
        print("\n--- Example 3: No Embeddings ---")
        
        result3 = await use_case.execute(
            source=[{"id": 1, "product": "Laptop", "price": 999.99}],
            table_name="products",
            embed_columns_names=[],  # No embeddings
            pk_columns=["id"]
        )
        
        print(f"✓ Success! Processed {result3.rows_processed} rows (no embeddings)")
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def validation_examples():
    """Examples of validation scenarios."""
    print("\n=== Validation Examples ===")
    
    repository = SimpleMockRepository()
    embedding_service = SimpleMockEmbeddingProvider()
    storage_loader = SimpleMockStorageLoader()
    
    use_case = DataAPIJSONUseCase(
        repo=repository,
        embedding_service=embedding_service,
        storage_loader=storage_loader
    )
    
    # Example 1: Invalid table name
    try:
        await use_case.execute(
            source=[{"id": 1}],
            table_name="",  # Empty table name
            embed_columns_names=[]
        )
    except Exception as e:
        print(f"✓ Caught validation error: {type(e).__name__}: {e}")
    
    # Example 2: Invalid embed_type
    try:
        await use_case.execute(
            source=[{"id": 1, "text": "test"}],
            table_name="test_table",
            embed_columns_names=["text"],
            embed_type="invalid_type"
        )
    except Exception as e:
        print(f"✓ Caught validation error: {type(e).__name__}")


async def main():
    """Run all examples."""
    print("DataAPIJSONUseCase Simple Examples")
    print("=" * 50)
    
    await simple_example()
    await validation_examples()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: These examples use mock implementations for demonstration.")
    print("In production, you would configure:")
    print("- Actual PostgreSQL database repository")
    print("- Real embedding service (OpenAI, Gemini, etc.)")
    print("- APIJSONStorageLoader with real API credentials")


if __name__ == "__main__":
    asyncio.run(main())