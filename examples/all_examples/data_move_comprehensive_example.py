#!/usr/bin/env python3
"""
Comprehensive DataMove Use Case Examples

This script demonstrates all major features of the DataMove use case including:
- New table creation from CSV data
- Existing schema validation (strict mode)
- New schema flexibility (flexible mode)
- S3 and local file integration
- Error handling and troubleshooting
- Dry-run validation
- Performance optimization

The DataMove use case is designed for production-grade data migration from CSV files
to PostgreSQL databases without embedding generation overhead.
"""

import asyncio
import pandas as pd
import os
from typing import Optional

# DataMove imports
from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.storage.loaders import LocalLoader, S3Loader

# Exception imports for error handling
from dataload.domain.entities import (
    DataMoveError,
    ValidationError,
    DatabaseOperationError,
    SchemaConflictError,
    CaseSensitivityError,
)


def create_sample_data():
    """Create sample CSV files for demonstration."""
    
    # Sample 1: Employee data for new table creation
    employees_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com', 'diana@company.com', 'eve@company.com'],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR'],
        'salary': [95000, 75000, 68000, 102000, 85000],
        'hire_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2022-11-05', '2023-04-12'],
        'active': [True, True, False, True, True],
        'skills': [
            ['Python', 'PostgreSQL', 'Docker'],
            ['Sales', 'CRM', 'Negotiation'],
            ['Marketing', 'Analytics', 'SEO'],
            ['Python', 'Machine Learning', 'AWS'],
            ['HR', 'Recruiting', 'Training']
        ]
    })
    
    # Sample 2: Updated employee data with schema changes
    employees_updated = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson', 'Frank Miller'],
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com', 'diana@company.com', 'eve@company.com', 'frank@company.com'],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR', 'Finance'],
        'salary': [98000, 77000, 70000, 105000, 87000, 92000],
        'hire_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2022-11-05', '2023-04-12', '2023-05-01'],
        'active': [True, True, True, True, True, True],  # Charlie is now active
        # New columns added
        'phone': ['+1-555-0101', '+1-555-0102', '+1-555-0103', '+1-555-0104', '+1-555-0105', '+1-555-0106'],
        'manager_id': [None, 1, 1, None, 1, 4],
        # skills column removed to demonstrate column removal
    })
    
    # Sample 3: Data with case sensitivity conflicts
    employees_case_conflict = pd.DataFrame({
        'ID': [1, 2, 3],  # Case conflict with 'id'
        'Name': ['Alice', 'Bob', 'Charlie'],  # Case conflict with 'name'
        'EMAIL': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],  # Case conflict with 'email'
        'Department': ['Engineering', 'Sales', 'Marketing'],
        'new_field': ['value1', 'value2', 'value3']
    })
    
    # Sample 4: Vector data for advanced use cases
    vector_data = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'document': ['First document text', 'Second document text', 'Third document text', 'Fourth document text'],
        'category': ['tech', 'business', 'science', 'tech'],
        'embedding': [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6, 0.7, 0.8]
        ],
        'metadata': [
            {'author': 'John', 'tags': ['tech', 'ai']},
            {'author': 'Jane', 'tags': ['business', 'strategy']},
            {'author': 'Bob', 'tags': ['science', 'research']},
            {'author': 'Alice', 'tags': ['tech', 'ml']}
        ]
    })
    
    # Save sample files
    os.makedirs('sample_data', exist_ok=True)
    employees_data.to_csv('sample_data/employees.csv', index=False)
    employees_updated.to_csv('sample_data/employees_updated.csv', index=False)
    employees_case_conflict.to_csv('sample_data/employees_case_conflict.csv', index=False)
    vector_data.to_csv('sample_data/vector_documents.csv', index=False)
    
    print("✅ Created sample CSV files in 'sample_data/' directory")
    return {
        'employees': 'sample_data/employees.csv',
        'employees_updated': 'sample_data/employees_updated.csv',
        'employees_case_conflict': 'sample_data/employees_case_conflict.csv',
        'vector_documents': 'sample_data/vector_documents.csv'
    }


async def setup_datamove_use_case() -> DataMoveUseCase:
    """
    Set up DataMoveUseCase with database connection.
    
    Returns:
        Configured DataMoveUseCase instance
        
    Note:
        In production, configure your database connection properly.
        This example uses environment variables from .env file.
    """
    try:
        # Initialize database connection
        db_connection = DBConnection()
        await db_connection.initialize()
        
        # Create repository
        repository = PostgresDataMoveRepository(db_connection)
        
        # Create use case with auto-loader (automatically selects S3 or Local based on path)
        use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
        
        print("✅ DataMove use case initialized successfully")
        return use_case
        
    except Exception as e:
        print(f"❌ Failed to initialize DataMove use case: {e}")
        print("💡 Make sure your .env file is configured with database credentials")
        raise


async def example_1_new_table_creation(use_case: DataMoveUseCase, sample_files: dict):
    """
    Example 1: Create a new table from CSV data.
    
    This demonstrates the simplest use case where the target table doesn't exist
    and DataMove creates it automatically from the CSV schema.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: New Table Creation")
    print("="*60)
    
    try:
        # First, try to check if table exists and handle accordingly
        table_name = "dm_employees_new"
        
        # Try as new table first, if it fails because table exists, 
        # then use existing_schema mode
        try:
            result = await use_case.execute(
                csv_path=sample_files['employees'],
                table_name=table_name,
                primary_key_columns=["id"]  # Specify primary key for new table
            )
        except ValidationError as ve:
            if "move_type parameter is required" in str(ve):
                print("📋 Table already exists, using existing_schema mode for demonstration")
                result = await use_case.execute(
                    csv_path=sample_files['employees'],
                    table_name=table_name,
                    move_type="existing_schema",
                    primary_key_columns=["id"]
                )
            else:
                raise ve
        
        print(f"✅ Success: {result.success}")
        print(f"📊 Rows processed: {result.rows_processed}")
        print(f"⏱️  Execution time: {result.execution_time:.2f}s")
        print(f"🆕 Table created: {result.table_created}")
        print(f"🔄 Operation type: {result.operation_type}")
        
        if result.warnings:
            print("⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   - {warning}")
                
    except DataMoveError as e:
        print(f"❌ DataMove failed: {e}")
        print(f"🔍 Error context: {e.context}")


async def example_2_dry_run_validation(use_case: DataMoveUseCase, sample_files: dict):
    """
    Example 2: Dry-run validation to preview operations.
    
    This shows how to validate and preview what would happen without
    actually making any changes to the database.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Dry-Run Validation")
    print("="*60)
    
    try:
        # Get operation preview
        preview = await use_case.get_operation_preview(
            csv_path=sample_files['employees_updated'],
            table_name="dm_employees_new",  # Table from example 1
            move_type="new_schema"  # Required for existing table
        )
        
        print(f"✅ Validation passed: {preview.validation_passed}")
        print(f"📋 Schema analysis:")
        print(f"   - Table exists: {preview.schema_analysis.table_exists}")
        print(f"   - Columns added: {preview.schema_analysis.columns_added}")
        print(f"   - Columns removed: {preview.schema_analysis.columns_removed}")
        print(f"   - Schema update required: {preview.schema_analysis.requires_schema_update}")
        
        if preview.recommendations:
            print("💡 Recommendations:")
            for rec in preview.recommendations:
                print(f"   - {rec}")
                
        if preview.warnings:
            print("⚠️  Warnings:")
            for warning in preview.warnings:
                print(f"   - {warning}")
        
        # Also demonstrate dry_run parameter
        print("\n📋 Using dry_run parameter:")
        dry_result = await use_case.execute(
            csv_path=sample_files['employees_updated'],
            table_name="dm_employees_new",
            move_type="new_schema",  # Required for existing table
            dry_run=True
        )
        
        print(f"✅ Dry run success: {dry_result.success}")
        print(f"📊 Would process: {dry_result.rows_processed} rows")
        
    except DataMoveError as e:
        print(f"❌ Validation failed: {e}")


async def example_3_existing_schema_strict(use_case: DataMoveUseCase, sample_files: dict):
    """
    Example 3: Existing schema validation (strict mode).
    
    This demonstrates strict validation where CSV must exactly match
    the existing table schema (column names, types, nullability).
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Existing Schema Validation (Strict)")
    print("="*60)
    
    try:
        # This should succeed - same schema as original
        result = await use_case.execute(
            csv_path=sample_files['employees'],
            table_name="dm_employees_new",
            move_type="existing_schema"  # Strict validation
        )
        
        print(f"✅ Success: {result.success}")
        print(f"📊 Rows processed: {result.rows_processed}")
        print(f"🔄 Operation type: {result.operation_type}")
        
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
        print("💡 This is expected if CSV schema doesn't exactly match table schema")
        
    # Now try with updated data (should fail due to schema changes)
    print("\n🔍 Testing with schema changes (should fail):")
    try:
        result = await use_case.execute(
            csv_path=sample_files['employees_updated'],
            table_name="dm_employees_new",
            move_type="existing_schema"
        )
        
    except ValidationError as e:
        print(f"❌ Expected validation failure: {e}")
        print("💡 existing_schema mode requires exact schema match")


async def example_4_new_schema_flexible(use_case: DataMoveUseCase, sample_files: dict):
    """
    Example 4: New schema validation (flexible mode).
    
    This demonstrates flexible validation that allows column additions
    and removals while preventing case-sensitivity conflicts.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: New Schema Validation (Flexible)")
    print("="*60)
    
    try:
        # This should succeed - allows schema evolution
        result = await use_case.execute(
            csv_path=sample_files['employees_updated'],
            table_name="dm_employees_new",
            move_type="new_schema"  # Flexible validation
        )
        
        print(f"✅ Success: {result.success}")
        print(f"📊 Rows processed: {result.rows_processed}")
        print(f"🔄 Schema updated: {result.schema_updated}")
        print(f"🔄 Operation type: {result.operation_type}")
        
        # Show what changed
        report = result.validation_report
        if report.schema_analysis.columns_added:
            print(f"➕ Columns added: {report.schema_analysis.columns_added}")
        if report.schema_analysis.columns_removed:
            print(f"➖ Columns removed: {report.schema_analysis.columns_removed}")
            
    except DataMoveError as e:
        print(f"❌ Operation failed: {e}")


async def example_5_case_sensitivity_conflicts(use_case: DataMoveUseCase, sample_files: dict):
    """
    Example 5: Case sensitivity conflict detection.
    
    This demonstrates how DataMove detects and prevents case-sensitivity
    conflicts that could cause data corruption.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Case Sensitivity Conflict Detection")
    print("="*60)
    
    try:
        # This should fail due to case conflicts
        result = await use_case.execute(
            csv_path=sample_files['employees_case_conflict'],
            table_name="dm_employees_new",
            move_type="new_schema"
        )
        
    except CaseSensitivityError as e:
        print(f"❌ Expected case sensitivity error: {e}")
        print("💡 DataMove prevents case conflicts to avoid data corruption")
        
        # Show detailed conflict information
        if hasattr(e, 'context') and 'case_conflicts' in e.context:
            conflicts = e.context['case_conflicts']
            print(f"🔍 Found {len(conflicts)} case conflicts:")
            for conflict in conflicts:
                print(f"   - DB: '{conflict.db_column}' vs CSV: '{conflict.csv_column}'")


async def example_6_s3_integration(use_case: DataMoveUseCase):
    """
    Example 6: S3 integration for cloud-based CSV files.
    
    This demonstrates loading data from S3 buckets with automatic
    loader selection and comprehensive error handling.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: S3 Integration")
    print("="*60)
    
    # Note: This is a demonstration - you would need actual S3 access
    s3_uri = "s3://your-data-bucket/employees/employees.csv"
    
    print(f"📁 S3 URI: {s3_uri}")
    print("💡 DataMove automatically detects S3 URIs and uses S3Loader")
    
    try:
        # DataMove automatically uses S3Loader for s3:// URIs
        result = await use_case.execute(
            csv_path=s3_uri,
            table_name="dm_employees_s3",
            primary_key_columns=["id"]
        )
        
        print(f"✅ S3 data load successful: {result.rows_processed} rows")
        
    except DataMoveError as e:
        print(f"❌ S3 operation failed: {e}")
        
        # Handle specific S3 errors
        if e.context.get("error_type") == "s3_operation_failed":
            print("💡 Troubleshooting S3 issues:")
            print("   - Check AWS credentials (aws configure)")
            print("   - Verify S3 bucket permissions")
            print("   - Ensure bucket and key exist")
            print("   - Check network connectivity")
        
    print("📝 Note: Configure AWS credentials and S3 access to test S3 integration")


async def example_7_vector_data_handling(use_case: DataMoveUseCase, sample_files: dict):
    """
    Example 7: Vector data handling for ML/AI use cases.
    
    This demonstrates how DataMove handles vector columns and JSON metadata
    commonly used in machine learning and AI applications.
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Vector Data Handling")
    print("="*60)
    
    try:
        result = await use_case.execute(
            csv_path=sample_files['vector_documents'],
            table_name="dm_vector_documents",
            primary_key_columns=["id"]
        )
        
        print(f"✅ Vector data loaded successfully")
        print(f"📊 Rows processed: {result.rows_processed}")
        print(f"🔄 Operation type: {result.operation_type}")
        print("💡 DataMove automatically handles:")
        print("   - Vector columns (arrays of floats)")
        print("   - JSON metadata columns")
        print("   - Proper PostgreSQL type mapping")
        
    except DataMoveError as e:
        print(f"❌ Vector data operation failed: {e}")


async def example_8_performance_optimization(use_case: DataMoveUseCase, sample_files: dict):
    """
    Example 8: Performance optimization for large datasets.
    
    This demonstrates batch processing and performance monitoring
    features for handling large-scale data migrations.
    """
    print("\n" + "="*60)
    print("EXAMPLE 8: Performance Optimization")
    print("="*60)
    
    try:
        # Use custom batch size for performance tuning
        result = await use_case.execute(
            csv_path=sample_files['employees'],
            table_name="dm_employees_performance",
            primary_key_columns=["id"],
            batch_size=500  # Smaller batches for demonstration
        )
        
        print(f"✅ Performance-optimized load completed")
        print(f"📊 Rows processed: {result.rows_processed}")
        print(f"⏱️  Execution time: {result.execution_time:.2f}s")
        print(f"🚀 Throughput: {result.rows_processed / result.execution_time:.0f} rows/second")
        
        print("\n💡 Performance optimization tips:")
        print("   - Adjust batch_size based on your data size and memory")
        print("   - Use larger batches (1000-5000) for better performance")
        print("   - Monitor execution_time and throughput metrics")
        print("   - Consider connection pooling for multiple operations")
        
    except DataMoveError as e:
        print(f"❌ Performance test failed: {e}")


async def example_9_comprehensive_error_handling():
    """
    Example 9: Comprehensive error handling patterns.
    
    This demonstrates all the different types of errors that can occur
    and how to handle them appropriately in production code.
    """
    print("\n" + "="*60)
    print("EXAMPLE 9: Comprehensive Error Handling")
    print("="*60)
    
    # This example shows error handling patterns without actual database operations
    print("🔍 Error handling patterns for production use:")
    
    error_handling_code = '''
async def production_data_move_with_error_handling():
    """Production-ready error handling example."""
    
    try:
        result = await use_case.execute(
            csv_path="data.csv",
            table_name="target_table",
            move_type="existing_schema"
        )
        
        # Handle successful operation
        logger.info(f"Data move completed: {result.rows_processed} rows")
        return result
        
    except ValidationError as e:
        # Handle validation failures
        logger.error(f"Validation failed: {e}")
        
        if isinstance(e, SchemaConflictError):
            # Schema mismatch - suggest using new_schema mode
            logger.info("Consider using move_type='new_schema' for flexible validation")
            
        elif isinstance(e, CaseSensitivityError):
            # Case conflicts - suggest column renaming
            logger.info("Fix case conflicts in CSV column names")
            
        # Return error details for user feedback
        return {"error": "validation_failed", "details": e.context}
        
    except DatabaseOperationError as e:
        # Handle database connection/operation failures
        logger.error(f"Database operation failed: {e}")
        
        # Implement retry logic for transient failures
        if "connection" in str(e).lower():
            logger.info("Retrying database connection...")
            # Implement retry logic here
            
        return {"error": "database_failed", "details": e.context}
        
    except DataMoveError as e:
        # Handle other DataMove errors (file access, etc.)
        logger.error(f"DataMove operation failed: {e}")
        
        if e.context.get("error_type") == "file_not_found":
            logger.info("Check file path and permissions")
            
        elif e.context.get("error_type") == "s3_operation_failed":
            logger.info("Check AWS credentials and S3 permissions")
            
        return {"error": "operation_failed", "details": e.context}
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {"error": "unexpected_error", "message": str(e)}
'''
    
    print(error_handling_code)


async def cleanup_example_tables(use_case: DataMoveUseCase):
    """Clean up example tables created during demonstrations."""
    print("\n" + "="*60)
    print("CLEANUP: Removing Example Tables")
    print("="*60)
    
    tables_to_cleanup = [
        "dm_employees_new",
        "dm_employees_s3", 
        "dm_vector_documents",
        "dm_employees_performance"
    ]
    
    for table_name in tables_to_cleanup:
        try:
            # Note: This would require implementing a drop_table method
            # For now, just show what would be cleaned up
            print(f"🗑️  Would drop table: {table_name}")
            
        except Exception as e:
            print(f"⚠️  Could not drop {table_name}: {e}")
    
    print("💡 Note: Implement repository.drop_table() method for automatic cleanup")


async def main():
    """Run all DataMove examples with comprehensive error handling."""
    
    print("DataMove Use Case - Comprehensive Examples")
    print("=" * 60)
    print("🚀 Production-grade data migration from CSV to PostgreSQL")
    print("📚 Demonstrates all major features and error handling patterns")
    print()
    
    try:
        # Create sample data
        sample_files = create_sample_data()
        
        # Set up DataMove use case
        use_case = await setup_datamove_use_case()
        
        # Run all examples
        await example_1_new_table_creation(use_case, sample_files)
        await example_2_dry_run_validation(use_case, sample_files)
        await example_3_existing_schema_strict(use_case, sample_files)
        await example_4_new_schema_flexible(use_case, sample_files)
        await example_5_case_sensitivity_conflicts(use_case, sample_files)
        await example_6_s3_integration(use_case)
        await example_7_vector_data_handling(use_case, sample_files)
        await example_8_performance_optimization(use_case, sample_files)
        await example_9_comprehensive_error_handling()
        
        # Cleanup (optional)
        # await cleanup_example_tables(use_case)
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("💡 Check the console output above for detailed results")
        print("📖 See troubleshooting guide below for common issues")
        
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        print("💡 Make sure your database is configured and accessible")
        
    finally:
        # Close database connection
        try:
            if 'use_case' in locals() and hasattr(use_case.repository, 'db_connection'):
                await use_case.repository.db_connection.close()
                print("🔌 Database connection closed")
        except:
            pass


if __name__ == "__main__":
    print("🔧 Configuration required:")
    print("   1. Set up .env file with database credentials")
    print("   2. Ensure PostgreSQL is running and accessible")
    print("   3. Install required dependencies: pip install vector-dataloader")
    print()
    
    # Run examples
    asyncio.run(main())