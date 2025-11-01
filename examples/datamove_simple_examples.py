#!/usr/bin/env python3
"""
Simple DataMove Examples - Both Move Type Scenarios

This script demonstrates the two main validation modes of DataMove:
1. existing_schema (strict validation)
2. new_schema (flexible validation)

These examples show the core functionality without complex error handling
to make the concepts clear and easy to understand.
"""

import asyncio
import pandas as pd
import os
from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
from dataload.infrastructure.db.db_connection import DBConnection


def create_sample_files():
    """Create sample CSV files for the examples."""
    
    # Original employee data
    original_data = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com'],
        'department': ['Engineering', 'Sales', 'Marketing']
    })
    
    # Updated data with same schema (for existing_schema example)
    updated_same_schema = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince'],
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com', 'diana@company.com'],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering']
    })
    
    # Evolved data with schema changes (for new_schema example)
    evolved_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com', 'diana@company.com', 'eve@company.com'],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR'],
        # New columns added
        'salary': [95000, 75000, 68000, 102000, 85000],
        'hire_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2022-11-05', '2023-04-12']
        # Note: No 'department' column to show column removal handling
    })
    
    # Save files
    os.makedirs('simple_examples', exist_ok=True)
    original_data.to_csv('simple_examples/employees_original.csv', index=False)
    updated_same_schema.to_csv('simple_examples/employees_updated.csv', index=False)
    evolved_data.to_csv('simple_examples/employees_evolved.csv', index=False)
    
    print("📁 Created sample files:")
    print("   - simple_examples/employees_original.csv")
    print("   - simple_examples/employees_updated.csv") 
    print("   - simple_examples/employees_evolved.csv")


async def setup_datamove():
    """Set up DataMove use case with database connection."""
    
    # Initialize database connection
    db_connection = DBConnection()
    await db_connection.initialize()
    
    # Create repository
    repository = PostgresDataMoveRepository(db_connection)
    
    # Create use case with auto-loader
    use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
    
    return use_case, db_connection


async def example_1_create_new_table(use_case):
    """
    Example 1: Create a new table from CSV data.
    
    When the target table doesn't exist, DataMove automatically creates it
    from the CSV schema. No move_type parameter is needed.
    """
    print("\n" + "="*50)
    print("EXAMPLE 1: Create New Table")
    print("="*50)
    
    result = await use_case.execute(
        csv_path="simple_examples/employees_original.csv",
        table_name="dm_employees_simple",
        primary_key_columns=["id"]
    )
    
    print(f"✅ Success: {result.success}")
    print(f"📊 Rows processed: {result.rows_processed}")
    print(f"🆕 Table created: {result.table_created}")
    print(f"🔄 Operation type: {result.operation_type}")
    
    return result.success


async def example_2_existing_schema_strict(use_case):
    """
    Example 2: Existing Schema Validation (Strict Mode)
    
    This mode requires the CSV to exactly match the existing table schema:
    - Same column names (case-sensitive)
    - Same data types
    - Same nullable constraints
    
    Use this when you want to ensure data integrity and prevent schema drift.
    """
    print("\n" + "="*50)
    print("EXAMPLE 2: Existing Schema (Strict Validation)")
    print("="*50)
    
    print("📋 Mode: existing_schema")
    print("🔍 Validation: Strict - CSV must exactly match table schema")
    print("✅ Use when: Data integrity is critical, no schema changes allowed")
    
    try:
        result = await use_case.execute(
            csv_path="simple_examples/employees_updated.csv",
            table_name="dm_employees_simple",
            move_type="existing_schema"  # Strict validation
        )
        
        print(f"✅ Success: {result.success}")
        print(f"📊 Rows processed: {result.rows_processed}")
        print(f"🔄 Operation type: {result.operation_type}")
        print("💡 The CSV schema exactly matched the table schema")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("💡 This happens when CSV schema doesn't exactly match table schema")
        return False


async def example_3_new_schema_flexible(use_case):
    """
    Example 3: New Schema Validation (Flexible Mode)
    
    This mode allows schema evolution:
    - Column additions are allowed
    - Column removals are allowed  
    - Case-sensitivity conflicts are prevented
    - Data type changes are validated
    
    Use this when you need flexibility for evolving data structures.
    """
    print("\n" + "="*50)
    print("EXAMPLE 3: New Schema (Flexible Validation)")
    print("="*50)
    
    print("📋 Mode: new_schema")
    print("🔍 Validation: Flexible - allows column additions/removals")
    print("✅ Use when: Schema evolution is expected, flexibility needed")
    
    try:
        result = await use_case.execute(
            csv_path="simple_examples/employees_evolved.csv",
            table_name="dm_employees_simple",
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
        
        print("💡 Schema evolution was handled gracefully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("💡 This could happen due to case conflicts or other validation issues")
        return False


async def example_4_dry_run_preview(use_case):
    """
    Example 4: Dry Run - Preview Operations
    
    Use dry_run=True to validate and preview what would happen
    without actually making any changes to the database.
    """
    print("\n" + "="*50)
    print("EXAMPLE 4: Dry Run Preview")
    print("="*50)
    
    print("📋 Mode: dry_run=True")
    print("🔍 Purpose: Preview operations without making changes")
    print("✅ Use when: Testing validation, previewing schema changes")
    
    # Preview with existing_schema mode
    result_strict = await use_case.execute(
        csv_path="simple_examples/employees_evolved.csv",
        table_name="dm_employees_simple",
        move_type="existing_schema",
        dry_run=True
    )
    
    print(f"\n🔍 Strict validation preview:")
    print(f"   Validation passed: {result_strict.validation_report.validation_passed}")
    if not result_strict.validation_report.validation_passed:
        print(f"   Errors: {len(result_strict.validation_report.errors)}")
        for error in result_strict.validation_report.errors[:2]:  # Show first 2 errors
            print(f"     - {error}")
    
    # Preview with new_schema mode
    result_flexible = await use_case.execute(
        csv_path="simple_examples/employees_evolved.csv",
        table_name="dm_employees_simple",
        move_type="new_schema",
        dry_run=True
    )
    
    print(f"\n🔍 Flexible validation preview:")
    print(f"   Validation passed: {result_flexible.validation_report.validation_passed}")
    print(f"   Would process: {result_flexible.rows_processed} rows")
    
    analysis = result_flexible.validation_report.schema_analysis
    if analysis.columns_added:
        print(f"   Would add columns: {analysis.columns_added}")
    if analysis.columns_removed:
        print(f"   Would remove columns: {analysis.columns_removed}")
    
    print("💡 Dry run completed - no actual changes made")


async def example_5_comparison_summary():
    """
    Example 5: Mode Comparison Summary
    
    This example summarizes when to use each validation mode.
    """
    print("\n" + "="*50)
    print("EXAMPLE 5: Validation Mode Comparison")
    print("="*50)
    
    comparison_table = """
    ┌─────────────────────┬─────────────────────┬─────────────────────┐
    │ Scenario            │ existing_schema     │ new_schema          │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ New table creation  │ Not applicable      │ Not applicable      │
    │                     │ (use no move_type)  │ (use no move_type)  │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Exact schema match  │ ✅ Perfect fit      │ ✅ Also works       │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Column additions    │ ❌ Fails validation │ ✅ Handles gracefully│
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Column removals     │ ❌ Fails validation │ ✅ Handles gracefully│
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Case conflicts      │ ❌ Fails validation │ ❌ Prevents conflicts│
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Data integrity      │ ✅ Maximum safety   │ ✅ Good safety      │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Schema evolution    │ ❌ Not supported    │ ✅ Fully supported  │
    └─────────────────────┴─────────────────────┴─────────────────────┘
    """
    
    print(comparison_table)
    
    print("\n💡 Recommendations:")
    print("   🔒 Use existing_schema for:")
    print("      - Production data with stable schemas")
    print("      - Critical data where exact match is required")
    print("      - Compliance scenarios requiring strict validation")
    
    print("\n   🔄 Use new_schema for:")
    print("      - Evolving data structures")
    print("      - ETL pipelines with changing sources")
    print("      - Development and testing environments")
    
    print("\n   🆕 No move_type needed for:")
    print("      - Creating new tables from CSV")
    print("      - Initial data loads")


async def main():
    """Run all simple examples."""
    
    print("DataMove Simple Examples - Move Type Scenarios")
    print("=" * 60)
    print("🎯 Purpose: Demonstrate existing_schema vs new_schema validation")
    print("📚 Learn: When and how to use each validation mode")
    print()
    
    try:
        # Create sample files
        create_sample_files()
        
        # Set up DataMove
        print("\n🔧 Setting up DataMove...")
        use_case, db_connection = await setup_datamove()
        print("✅ DataMove initialized successfully")
        
        # Run examples
        success1 = await example_1_create_new_table(use_case)
        
        if success1:
            await example_2_existing_schema_strict(use_case)
            await example_3_new_schema_flexible(use_case)
            await example_4_dry_run_preview(use_case)
        
        await example_5_comparison_summary()
        
        print("\n" + "="*60)
        print("✅ All examples completed!")
        print("💡 Key takeaways:")
        print("   - existing_schema: Strict validation for stable schemas")
        print("   - new_schema: Flexible validation for evolving schemas")
        print("   - dry_run: Preview changes without making modifications")
        print("   - No move_type: Automatic for new table creation")
        
        # Close connection
        await db_connection.close()
        print("🔌 Database connection closed")
        
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        print("💡 Make sure your database is configured in .env file")
        print("📖 See troubleshooting guide for help")


if __name__ == "__main__":
    print("🔧 Prerequisites:")
    print("   1. Configure .env file with database credentials")
    print("   2. Ensure PostgreSQL is running")
    print("   3. Install: pip install vector-dataloader")
    print()
    
    # Run examples
    asyncio.run(main())