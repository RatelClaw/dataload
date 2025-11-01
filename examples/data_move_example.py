"""
Example usage of DataMoveUseCase for moving data from CSV to PostgreSQL.

This example demonstrates the main functionality of the DataMove use case
including table creation, data replacement, and validation scenarios.
"""

import asyncio
import pandas as pd
from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.interfaces.storage_loader import StorageLoaderInterface
from dataload.config import logger
from dataload.application.use_cases.data_move_use_case import ValidationError
from dataload.application.use_cases.data_move_use_case import DatabaseOperationError


class LocalCSVLoader(StorageLoaderInterface):
    """Simple local CSV loader for demonstration."""
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV from local file path."""
        return pd.read_csv(file_path)


async def example_new_table_creation():
    """Example: Create a new table from CSV data."""
    print("\n=== Example 1: New Table Creation ===")
    
    # Create sample CSV data
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'Sales']
    })
    
    # Save to CSV for demonstration
    csv_path = "sample_employees.csv"
    sample_data.to_csv(csv_path, index=False)
    print(f"Created sample CSV: {csv_path}")
    
    # # Initialize components (you would use real DB connection in practice)
    # db_connection = DBConnection()  # Configure with your DB settings
    # await db_connection.initialize()  # Initialize the connection pool
    # repository = PostgresDataMoveRepository(db_connection)
    # # Use auto-detection for storage loader
    # use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
    # 
    # # Execute data move for new table (local file)
    # result = await use_case.execute(
    #     csv_path=csv_path,
    #     table_name="employees",
    #     move_type="new_schema",
    #     primary_key_columns=["id"]
    # )
    # 
    # print(f"Result: {result.success}")
    # print(f"Rows processed: {result.rows_processed}")
    # print(f"Table created: {result.table_created}")
    # print(f"Operation type: {result.operation_type}")
    
    print("Note: Uncomment the code above and configure DB connection to run actual example")


async def example_s3_integration():
    """Example: Load data from S3 CSV file."""
    print("\n=== Example 1b: S3 Integration ===")
    
    # This example shows how to load from S3
    s3_uri = "s3://your-bucket/data/employees.csv"
    
    # Initialize components (mock for demonstration)
    # db_connection = DBConnection()  # Configure with your DB settings
    # repository = PostgresDataMoveRepository(db_connection)
    # 
    # # Use auto-detection - will automatically use S3Loader for s3:// URIs
    # use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
    # 
    # try:
    #     # Execute data move from S3
    #     result = await use_case.execute(
    #         csv_path=s3_uri,
    #         table_name="employees_from_s3",
    #         primary_key_columns=["id"]
    #     )
    #     
    #     print(f"S3 data load successful: {result.rows_processed} rows")
    #     print(f"Table created: {result.table_created}")
    # 
    # except DataMoveError as e:
    #     if e.context.get("error_type") == "s3_credentials_error":
    #         print("AWS credentials issue - check your AWS configuration")
    #     elif e.context.get("error_type") == "s3_resource_error":
    #         print("S3 bucket or key not found - check the S3 URI")
    #     else:
    #         print(f"S3 operation failed: {e}")
    
    print("Note: Configure AWS credentials and S3 access to use S3 integration")
    print("Required: pip install boto3")


async def example_dry_run_validation():
    """Example: Dry run to preview what would happen."""
    print("\n=== Example 2: Dry Run Validation ===")
    
    # This example shows how to use dry_run to validate without making changes
    csv_path = "sample_employees.csv"
    
    # Initialize components (mock for demonstration)
    # use_case = DataMoveUseCase(repository, storage_loader)
    # 
    # # Get preview of operation
    # preview = await use_case.get_operation_preview(
    #     csv_path=csv_path,
    #     table_name="employees"
    # )
    # 
    # print(f"Validation passed: {preview.validation_passed}")
    # print(f"Recommendations: {preview.recommendations}")
    # print(f"Warnings: {preview.warnings}")
    # 
    # # Or use execute with dry_run=True
    # result = await use_case.execute(
    #     csv_path=csv_path,
    #     table_name="employees",
    #     dry_run=True
    # )
    # 
    # print(f"Dry run result: {result.success}")
    # print(f"Would process: {result.rows_processed} rows")
    
    print("Note: This example shows the API - configure DB connection to run")


async def example_existing_schema_validation():
    """Example: Strict validation for existing table."""
    print("\n=== Example 3: Existing Schema Validation ===")
    
    # This example shows existing_schema mode (strict validation)
    csv_path = "sample_employees.csv"
    
    # Initialize components (you would use real DB connection in practice)
    db_connection = DBConnection()  # Configure with your DB settings
    await db_connection.initialize()  # Initialize the connection pool
    repository = PostgresDataMoveRepository(db_connection)
    
    # Use auto-detection for storage loader
    use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
    
    # use_case = DataMoveUseCase(repository, storage_loader)
    
    try:
        result = await use_case.execute(
            csv_path=csv_path,
            table_name="employees",
            move_type="existing_schema"  # Strict validation
        )
        print(f"Data replacement successful: {result.rows_processed} rows")
    
    except ValidationError as e:
        print(f"Validation failed: {e}")
        # Handle validation errors (schema mismatch, type conflicts, etc.)
    
    except DatabaseOperationError as e:
        print(f"Database operation failed: {e}")
        # Handle database errors
    
    print("Note: existing_schema requires exact column and type matching")


async def example_new_schema_flexibility():
    """Example: Flexible validation allowing schema changes."""
    print("\n=== Example 4: New Schema Flexibility ===")
    
    # This example shows new_schema mode (flexible validation)
    csv_path = "sample_employees.csv"
    
    # use_case = DataMoveUseCase(repository, storage_loader)
    # 
    # try:
    #     result = await use_case.execute(
    #         csv_path=csv_path,
    #         table_name="flexible_employees_table",
    #         move_type="new_schema"  # Flexible validation
    #     )
    #     print(f"Schema update successful: {result.schema_updated}")
    #     print(f"Data replacement successful: {result.rows_processed} rows")
    # 
    # except CaseSensitivityError as e:
    #     print(f"Case sensitivity conflict: {e}")
    #     # Handle case conflicts between CSV and DB columns
    # 
    # except ValidationError as e:
    #     print(f"Validation failed: {e}")
    
    print("Note: new_schema allows column additions/removals but prevents case conflicts")


async def example_error_handling():
    """Example: Comprehensive error handling."""
    print("\n=== Example 5: Error Handling ===")
    
    # use_case = DataMoveUseCase(repository, storage_loader)
    # 
    # try:
    #     result = await use_case.execute(
    #         csv_path="nonexistent.csv",
    #         table_name="test_table"
    #     )
    # 
    # except DataMoveError as e:
    #     print(f"DataMove error: {e}")
    #     print(f"Error context: {e.context}")
    #     
    #     # Handle specific error types
    #     if isinstance(e, ValidationError):
    #         print("This is a validation error")
    #     elif isinstance(e, DatabaseOperationError):
    #         print("This is a database operation error")
    #     elif isinstance(e, SchemaConflictError):
    #         print("This is a schema conflict error")
    #     elif isinstance(e, CaseSensitivityError):
    #         print("This is a case sensitivity error")
    # 
    # except Exception as e:
    #     print(f"Unexpected error: {e}")
    
    print("Note: DataMove provides detailed error context for debugging")


def example_configuration():
    """Example: How to configure DataMoveUseCase."""
    print("\n=== Example 6: Configuration ===")
    
    print("""
    # Method 1: Auto-detection (Recommended)
    # Automatically selects S3Loader or LocalLoader based on path
    from dataload.infrastructure.db.db_connection import DBConnection
    from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
    from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
    
    # 1. Configure database connection
    db_connection = DBConnection(
        host="localhost",
        port=5432,
        database="your_database",
        user="your_user",
        password="your_password"
    )
    
    # 2. Create repository
    repository = PostgresDataMoveRepository(db_connection)
    
    # 3. Create use case with auto-detection
    use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
    
    # 4. Execute with local file (automatically uses LocalLoader)
    result = await use_case.execute(
        csv_path="local_file.csv",
        table_name="your_table"
    )
    
    # 5. Execute with S3 file (automatically uses S3Loader)
    result = await use_case.execute(
        csv_path="s3://your-bucket/your-file.csv",
        table_name="your_table"
    )
    
    # Method 2: Explicit storage loader
    from dataload.infrastructure.storage.loaders import LocalLoader, S3Loader
    
    # For local files only
    local_loader = LocalLoader()
    use_case_local = DataMoveUseCase(
        repository=repository,
        storage_loader=local_loader
    )
    
    # For S3 files only
    s3_loader = S3Loader()
    use_case_s3 = DataMoveUseCase(
        repository=repository,
        storage_loader=s3_loader
    )
    
    # Method 3: Factory method for specific path
    storage_loader = DataMoveUseCase.create_storage_loader("s3://bucket/file.csv")
    use_case = DataMoveUseCase(
        repository=repository,
        storage_loader=storage_loader
    )
    """)


async def main():
    """Run all examples."""
    print("DataMove Use Case Examples")
    print("=" * 50)
    
    # await example_new_table_creation()
    # await example_s3_integration()
    # await example_dry_run_validation()
    await example_existing_schema_validation()
    # await example_new_schema_flexibility()
    # await example_error_handling()
    # example_configuration()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run these examples with a real database:")
    print("1. Configure your PostgreSQL connection in .env file")
    print("2. Uncomment the code in each example function")
    print("3. Install required dependencies: pip install pandas asyncpg")
    print("4. Run: python examples/data_move_example.py")


if __name__ == "__main__":
    asyncio.run(main())