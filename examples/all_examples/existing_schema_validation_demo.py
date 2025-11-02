"""
Demonstration of existing_schema validation logic for DataMove use case.

This script shows how the existing_schema validation works with different scenarios:
- Successful validation with matching schema
- Validation failures with missing/extra columns
- Type mismatch detection
- Nullable constraint violations
- Case sensitivity issues
"""

import asyncio
import pandas as pd
from dataload.application.services.validation import ValidationService
from dataload.domain.entities import TableInfo, ColumnInfo


async def demo_existing_schema_validation():
    """Demonstrate existing_schema validation with various scenarios."""
    
    print("=== DataMove Existing Schema Validation Demo ===\n")
    
    # Create validation service
    validation_service = ValidationService()
    
    # Define a sample table schema
    table_info = TableInfo(
        name="users",
        columns={
            "id": ColumnInfo(name="id", data_type="integer", nullable=False),
            "name": ColumnInfo(name="name", data_type="text", nullable=True),
            "email": ColumnInfo(name="email", data_type="text", nullable=False),
            "age": ColumnInfo(name="age", data_type="integer", nullable=True),
        },
        primary_keys=["id"],
        constraints=[],
        indexes=[]
    )
    
    print("Target Table Schema:")
    print(f"Table: {table_info.name}")
    for col_name, col_info in table_info.columns.items():
        nullable_str = "NULL" if col_info.nullable else "NOT NULL"
        print(f"  - {col_name}: {col_info.data_type} {nullable_str}")
    print()
    
    # Scenario 1: Perfect match (should pass)
    print("Scenario 1: Perfect Schema Match")
    print("-" * 40)
    
    perfect_df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
        "age": [25, 30, 35]
    })
    
    result = await validation_service.validate_data_move(
        table_info=table_info,
        df=perfect_df,
        move_type="existing_schema"
    )
    
    print(f"Validation Passed: {result.validation_passed}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    else:
        print("✅ No validation errors!")
    print()
    
    # Scenario 2: Missing columns (should fail)
    print("Scenario 2: Missing Required Columns")
    print("-" * 40)
    
    missing_cols_df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
        # Missing: email, age
    })
    
    result = await validation_service.validate_data_move(
        table_info=table_info,
        df=missing_cols_df,
        move_type="existing_schema"
    )
    
    print(f"Validation Passed: {result.validation_passed}")
    print("Errors:")
    for error in result.errors:
        print(f"  - {error}")
    print()
    
    # Scenario 3: Extra columns (should fail)
    print("Scenario 3: Extra Columns in CSV")
    print("-" * 40)
    
    extra_cols_df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
        "age": [25, 30, 35],
        "extra_column": ["extra1", "extra2", "extra3"]  # Extra column
    })
    
    result = await validation_service.validate_data_move(
        table_info=table_info,
        df=extra_cols_df,
        move_type="existing_schema"
    )
    
    print(f"Validation Passed: {result.validation_passed}")
    print("Errors:")
    for error in result.errors:
        print(f"  - {error}")
    print()
    
    # Scenario 4: Type mismatches (should fail)
    print("Scenario 4: Data Type Mismatches")
    print("-" * 40)
    
    type_mismatch_df = pd.DataFrame({
        "id": ["not_an_integer", "also_not_int", "still_not_int"],  # Should be integer
        "name": ["Alice", "Bob", "Charlie"],
        "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
        "age": ["twenty_five", "thirty", "thirty_five"]  # Should be integer
    })
    
    result = await validation_service.validate_data_move(
        table_info=table_info,
        df=type_mismatch_df,
        move_type="existing_schema"
    )
    
    print(f"Validation Passed: {result.validation_passed}")
    print("Errors:")
    for error in result.errors:
        print(f"  - {error}")
    print()
    
    # Scenario 5: Nullable constraint violations (should fail)
    print("Scenario 5: Nullable Constraint Violations")
    print("-" * 40)
    
    null_violation_df = pd.DataFrame({
        "id": [1, None, 3],  # id is NOT NULL but has null
        "name": ["Alice", "Bob", None],  # name is nullable, so OK
        "email": ["alice@test.com", None, "charlie@test.com"],  # email is NOT NULL but has null
        "age": [25, 30, None]  # age is nullable, so OK
    })
    
    result = await validation_service.validate_data_move(
        table_info=table_info,
        df=null_violation_df,
        move_type="existing_schema"
    )
    
    print(f"Validation Passed: {result.validation_passed}")
    print("Errors:")
    for error in result.errors:
        print(f"  - {error}")
    print()
    
    # Scenario 6: Case sensitivity issues (should fail)
    print("Scenario 6: Case Sensitivity Issues")
    print("-" * 40)
    
    case_mismatch_df = pd.DataFrame({
        "ID": [1, 2, 3],  # Should be 'id'
        "Name": ["Alice", "Bob", "Charlie"],  # Should be 'name'
        "EMAIL": ["alice@test.com", "bob@test.com", "charlie@test.com"],  # Should be 'email'
        "Age": [25, 30, 35]  # Should be 'age'
    })
    
    result = await validation_service.validate_data_move(
        table_info=table_info,
        df=case_mismatch_df,
        move_type="existing_schema"
    )
    
    print(f"Validation Passed: {result.validation_passed}")
    print("Errors:")
    for error in result.errors:
        print(f"  - {error}")
    
    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")
    print()
    
    print("=== Demo Complete ===")
    print("\nKey Features of existing_schema validation:")
    print("✅ Strict column name matching (case-sensitive)")
    print("✅ Exact data type compatibility checking")
    print("✅ Nullable constraint validation")
    print("✅ Comprehensive error collection and reporting")
    print("✅ Actionable error messages with specific details")
    print("✅ Edge case handling (empty DataFrames, etc.)")


if __name__ == "__main__":
    asyncio.run(demo_existing_schema_validation())