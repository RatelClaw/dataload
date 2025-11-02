"""
Example usage of DataAPIJSONUseCase for loading API/JSON data with embeddings.

This example demonstrates various scenarios for using the DataAPIJSONUseCase
to load data from APIs, JSON files, and raw JSON data with embedding generation
and database operations.
"""

import asyncio
import json
import tempfile
import os
from typing import List

# Import the use case and required interfaces
from dataload.application.use_cases.data_api_json_use_case import DataAPIJSONUseCase
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader

# For this example, we'll create simple mock implementations
# In a real application, you would use actual implementations

class MockDataMoveRepository:
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

class MockEmbeddingProvider:
    """Simple mock embedding provider for example purposes."""
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

class MockAPIJSONStorageLoader:
    """Simple mock storage loader for example purposes."""
    def __init__(self, **kwargs):
        pass
        
    def load_json(self, source, config=None):
        import pandas as pd
        # Return simple mock data
        return pd.DataFrame({
            "id": [1, 2, 3],
            "title": ["Post 1", "Post 2", "Post 3"],
            "body": ["Content 1", "Content 2", "Content 3"]
        })
        
    def validate_config(self, config):
        return []


async def example_api_loading():
    """Example: Loading data from an API endpoint."""
    print("=== API Loading Example ===")
    
    # Set up components
    repository = MockDataMoveRepository()
    embedding_service = MockEmbeddingProvider()
    
    # Configure API loader with authentication
    api_loader = MockAPIJSONStorageLoader(
        base_url="https://jsonplaceholder.typicode.com",
        timeout=30,
        retry_attempts=3
    )
    
    # Create use case
    use_case = DataAPIJSONUseCase(
        repo=repository,
        embedding_service=embedding_service,
        storage_loader=api_loader
    )
    
    try:
        # Load data from API with embeddings
        result = await use_case.execute(
            source="/posts",  # API endpoint
            table_name="blog_posts",
            embed_columns_names=["title", "body"],
            embed_type="combined",
            create_table_if_not_exists=True,
            pk_columns=["id"]
        )
        
        print(f"✓ API loading completed successfully!")
        print(f"  - Rows processed: {result.rows_processed}")
        print(f"  - Table created: {result.table_created}")
        print(f"  - Execution time: {result.execution_time:.2f}s")
        print(f"  - Warnings: {len(result.warnings)}")
        
    except Exception as e:
        print(f"✗ API loading failed: {e}")


async def example_json_file_loading():
    """Example: Loading data from a JSON file."""
    print("\n=== JSON File Loading Example ===")
    
    # Create sample JSON data
    sample_data = [
        {
            "id": 1,
            "product_name": "Laptop Computer",
            "description": "High-performance laptop for professionals",
            "category": "Electronics",
            "price": 1299.99,
            "tags": ["laptop", "computer", "professional"]
        },
        {
            "id": 2,
            "product_name": "Wireless Headphones",
            "description": "Premium noise-canceling wireless headphones",
            "category": "Electronics", 
            "price": 299.99,
            "tags": ["headphones", "wireless", "audio"]
        },
        {
            "id": 3,
            "product_name": "Office Chair",
            "description": "Ergonomic office chair with lumbar support",
            "category": "Furniture",
            "price": 449.99,
            "tags": ["chair", "office", "ergonomic"]
        }
    ]
    
    # Write to temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f, indent=2)
        json_file_path = f.name
    
    try:
        # Set up components
        repository = MockDataMoveRepository()
        embedding_service = MockEmbeddingProvider()
        api_loader = MockAPIJSONStorageLoader()
        
        use_case = DataAPIJSONUseCase(
            repo=repository,
            embedding_service=embedding_service,
            storage_loader=api_loader
        )
        
        # Load JSON file with column mapping and embeddings
        result = await use_case.execute(
            source=json_file_path,
            table_name="products",
            embed_columns_names=["product_name", "description"],
            embed_type="separated",  # Create separate embedding columns
            pk_columns=["id"],
            column_name_mapping={
                "product_name": "name",
                "description": "desc"
            }
        )
        
        print(f"✓ JSON file loading completed successfully!")
        print(f"  - Rows processed: {result.rows_processed}")
        print(f"  - Table created: {result.table_created}")
        print(f"  - Operation type: {result.operation_type}")
        
        # Show the final data structure
        if repository.tables.get("products") is not None:
            final_data = repository.tables["products"]
            print(f"  - Final columns: {list(final_data.columns)}")
            print(f"  - Embedding columns: {[col for col in final_data.columns if col.endswith('_enc')]}")
        
    except Exception as e:
        print(f"✗ JSON file loading failed: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(json_file_path):
            os.unlink(json_file_path)


async def example_raw_json_loading():
    """Example: Loading raw JSON data with transformations."""
    print("\n=== Raw JSON Loading Example ===")
    
    # Raw JSON data (list of dictionaries)
    user_data = [
        {
            "user_id": 1,
            "first_name": "Alice",
            "last_name": "Johnson",
            "email": "alice.johnson@example.com",
            "department": "Engineering",
            "join_date": "2023-01-15",
            "skills": ["Python", "Machine Learning", "Data Analysis"]
        },
        {
            "user_id": 2,
            "first_name": "Bob",
            "last_name": "Smith", 
            "email": "bob.smith@example.com",
            "department": "Product",
            "join_date": "2023-02-20",
            "skills": ["Product Management", "Strategy", "Analytics"]
        },
        {
            "user_id": 3,
            "first_name": "Carol",
            "last_name": "Davis",
            "email": "carol.davis@example.com", 
            "department": "Design",
            "join_date": "2023-03-10",
            "skills": ["UI/UX Design", "Prototyping", "User Research"]
        }
    ]
    
    # Set up components
    repository = MockDataMoveRepository()
    embedding_service = MockEmbeddingProvider()
    api_loader = APIJSONStorageLoader()
    
    use_case = DataAPIJSONUseCase(
        repo=repository,
        embedding_service=embedding_service,
        storage_loader=api_loader
    )
    
    try:
        # Load raw JSON with transformations and embeddings
        result = await use_case.execute(
            source=user_data,
            table_name="employees",
            embed_columns_names=["full_name", "skills_text"],  # Use transformed column names
            embed_type="combined",
            pk_columns=["id"],
            column_name_mapping={
                "user_id": "id",
                "first_name": "fname",
                "last_name": "lname"
            },
            update_request_body_mapping={
                "full_name": "concat({fname}, ' ', {lname})",
                "skills_text": "join({skills}, ', ')"
            }
        )
        
        print(f"✓ Raw JSON loading completed successfully!")
        print(f"  - Rows processed: {result.rows_processed}")
        print(f"  - Table created: {result.table_created}")
        print(f"  - Execution time: {result.execution_time:.2f}s")
        
        # Show transformation results
        if repository.tables.get("employees") is not None:
            final_data = repository.tables["employees"]
            print(f"  - Transformed columns: {[col for col in final_data.columns if col in ['full_name', 'skills_text']]}")
        
    except Exception as e:
        print(f"✗ Raw JSON loading failed: {e}")


async def example_existing_table_update():
    """Example: Updating an existing table with new data."""
    print("\n=== Existing Table Update Example ===")
    
    # Set up components
    repository = MockDataMoveRepository()
    embedding_service = MockEmbeddingProvider()
    api_loader = APIJSONStorageLoader()
    
    use_case = DataAPIJSONUseCase(
        repo=repository,
        embedding_service=embedding_service,
        storage_loader=api_loader
    )
    
    try:
        # First, create initial data
        initial_data = [
            {"id": 1, "title": "Initial Post 1", "content": "Initial content 1"},
            {"id": 2, "title": "Initial Post 2", "content": "Initial content 2"}
        ]
        
        result1 = await use_case.execute(
            source=initial_data,
            table_name="blog_posts",
            embed_columns_names=["title", "content"],
            embed_type="combined",
            pk_columns=["id"],
            create_table_if_not_exists=True
        )
        
        print(f"✓ Initial data loaded: {result1.rows_processed} rows")
        
        # Now update with new data
        update_data = [
            {"id": 2, "title": "Updated Post 2", "content": "Updated content 2"},  # Update existing
            {"id": 3, "title": "New Post 3", "content": "New content 3"},        # Add new
            {"id": 4, "title": "New Post 4", "content": "New content 4"}         # Add new
        ]
        
        result2 = await use_case.execute(
            source=update_data,
            table_name="blog_posts",
            embed_columns_names=["title", "content"],
            embed_type="combined",
            pk_columns=["id"],
            create_table_if_not_exists=False  # Table already exists
        )
        
        print(f"✓ Table updated: {result2.rows_processed} rows processed")
        print(f"  - Table created: {result2.table_created}")
        print(f"  - Operation type: {result2.operation_type}")
        
        # Show final table state
        if repository.tables.get("blog_posts") is not None:
            final_data = repository.tables["blog_posts"]
            print(f"  - Final row count: {len(final_data)}")
        
    except Exception as e:
        print(f"✗ Table update failed: {e}")


async def example_error_handling():
    """Example: Demonstrating error handling scenarios."""
    print("\n=== Error Handling Examples ===")
    
    repository = MockDataMoveRepository()
    embedding_service = MockEmbeddingProvider()
    api_loader = APIJSONStorageLoader()
    
    use_case = DataAPIJSONUseCase(
        repo=repository,
        embedding_service=embedding_service,
        storage_loader=api_loader
    )
    
    # Example 1: Invalid parameters
    try:
        await use_case.execute(
            source={"test": "data"},
            table_name="",  # Empty table name
            embed_columns_names=["test"]
        )
    except Exception as e:
        print(f"✓ Caught validation error: {type(e).__name__}: {e}")
    
    # Example 2: Missing embedding columns
    try:
        await use_case.execute(
            source=[{"id": 1, "name": "test"}],
            table_name="test_table",
            embed_columns_names=["nonexistent_column"]
        )
    except Exception as e:
        print(f"✓ Caught embedding validation error: {type(e).__name__}")
    
    # Example 3: Invalid embed_type
    try:
        await use_case.execute(
            source=[{"id": 1, "name": "test"}],
            table_name="test_table",
            embed_type="invalid_type"
        )
    except Exception as e:
        print(f"✓ Caught embed_type validation error: {type(e).__name__}")


async def main():
    """Run all examples."""
    print("DataAPIJSONUseCase Examples")
    print("=" * 50)
    
    # Note: These examples use mock implementations for demonstration
    # In a real application, you would configure actual database and embedding services
    
    await example_json_file_loading()
    await example_raw_json_loading()
    await example_existing_table_update()
    await example_error_handling()
    
    # API loading example would require actual API access
    # await example_api_loading()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nNote: These examples use mock implementations.")
    print("In production, configure actual PostgreSQL and embedding services.")


if __name__ == "__main__":
    asyncio.run(main())