"""
Example demonstrating new_schema flexible validation functionality.

This example shows how the new_schema validation mode allows for flexible
schema evolution while preventing case-sensitivity conflicts and maintaining
data integrity.
"""

import asyncio
import pandas as pd
from dataload.application.services.validation.validation_service import ValidationService
from dataload.domain.entities import (
    TableInfo, ColumnInfo, Constraint, IndexInfo
)


def create_sample_table_info():
    """Create a sample table structure for demonstration."""
    return TableInfo(
        name='users',
        columns={
            'id': ColumnInfo(name='id', data_type='integer', nullable=False),
            'username': ColumnInfo(name='username', data_type='text', nullable=False),
            'email': ColumnInfo(name='email', data_type='text', nullable=False),
            'created_at': ColumnInfo(name='created_at', data_type='timestamp', nullable=True),
            'last_login': ColumnInfo(name='last_login', data_type='timestamp', nullable=True),
        },
        primary_keys=['id'],
        constraints=[
            Constraint(name='pk_users', type='PRIMARY KEY', columns=['id']),
            Constraint(name='unique_username', type='UNIQUE', columns=['username']),
            Constraint(name='unique_email', type='UNIQUE', columns=['email'])
        ],
        indexes=[
            IndexInfo(name='idx_username', columns=['username'], index_type='btree', unique=True),
            IndexInfo(name='idx_email', columns=['email'], index_type='btree', unique=True)
        ]
    )


def create_compatible_csv_data():
    """Create CSV data that's compatible with new_schema mode."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'username': ['alice', 'bob', 'charlie', 'diana'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'diana@example.com'],
        'created_at': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        # New columns being added
        'first_name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'last_name': ['Smith', 'Johnson', 'Brown', 'Wilson'],
        'phone': ['+1234567890', '+0987654321', '+1122334455', '+5566778899'],
        'status': ['active', 'active', 'inactive', 'active']
        # Note: 'last_login' column is being removed (not present in CSV)
    })


def create_case_conflict_csv_data():
    """Create CSV data with case-sensitivity conflicts."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'Username': ['alice', 'bob', 'charlie'],  # Case conflict with 'username'
        'EMAIL': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],  # Case conflict with 'email'
        'created_at': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'new_field': ['value1', 'value2', 'value3']
    })


def create_problematic_csv_data():
    """Create CSV data with various validation issues."""
    return pd.DataFrame({
        'id': [1, 2, None, 4],  # Null in primary key
        'username': ['alice', 'bob', 'charlie', None],  # Null in non-nullable column
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'diana@example.com'],
        'created_at': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'new_column': ['value1', 'value2', 'value3', 'value4']
        # Note: Removing 'last_login' column
    })


async def demonstrate_successful_validation():
    """Demonstrate successful new_schema validation."""
    print("=== Successful New Schema Validation ===")
    
    validation_service = ValidationService()
    table_info = create_sample_table_info()
    csv_data = create_compatible_csv_data()
    
    result = await validation_service.validate_data_move(
        table_info=table_info,
        df=csv_data,
        move_type='new_schema'
    )
    
    print(f"Validation Passed: {result.validation_passed}")
    print(f"Schema Update Required: {result.schema_analysis.requires_schema_update}")
    print(f"Columns Added: {result.schema_analysis.columns_added}")
    print(f"Columns Removed: {result.schema_analysis.columns_removed}")
    
    print("\nWarnings:")
    for warning in result.warnings:
        print(f"  - {warning}")
    
    print("\nRecommendations:")
    for recommendation in result.recommendations:
        print(f"  - {recommendation}")
    
    print()


async def demonstrate_case_conflict_detection():
    """Demonstrate case-sensitivity conflict detection."""
    print("=== Case-Sensitivity Conflict Detection ===")
    
    validation_service = ValidationService()
    table_info = create_sample_table_info()
    csv_data = create_case_conflict_csv_data()
    
    result = await validation_service.validate_data_move(
        table_info=table_info,
        df=csv_data,
        move_type='new_schema'
    )
    
    print(f"Validation Passed: {result.validation_passed}")
    print(f"Case Conflicts Found: {len(result.case_conflicts)}")
    
    print("\nCase Conflicts:")
    for conflict in result.case_conflicts:
        print(f"  - Type: {conflict.conflict_type}")
        print(f"    DB Column: {conflict.db_column}")
        print(f"    CSV Column: {conflict.csv_column}")
    
    print("\nErrors:")
    for error in result.errors:
        print(f"  - {error}")
    
    print("\nRecommendations:")
    for recommendation in result.recommendations:
        print(f"  - {recommendation}")
    
    print()


async def demonstrate_constraint_validation():
    """Demonstrate constraint validation in new_schema mode."""
    print("=== Constraint Validation ===")
    
    validation_service = ValidationService()
    table_info = create_sample_table_info()
    csv_data = create_problematic_csv_data()
    
    result = await validation_service.validate_data_move(
        table_info=table_info,
        df=csv_data,
        move_type='new_schema'
    )
    
    print(f"Validation Passed: {result.validation_passed}")
    print(f"Constraint Violations: {len(result.constraint_violations)}")
    
    print("\nConstraint Violations:")
    for violation in result.constraint_violations:
        print(f"  - Constraint: {violation.constraint_name} ({violation.constraint_type})")
        print(f"    Column: {violation.column_name}")
        print(f"    Violation Type: {violation.violation_type}")
        print(f"    Affected Rows: {violation.affected_rows}")
    
    print("\nErrors:")
    for error in result.errors:
        print(f"  - {error}")
    
    print()


async def demonstrate_backward_compatibility_checks():
    """Demonstrate backward compatibility analysis."""
    print("=== Backward Compatibility Analysis ===")
    
    validation_service = ValidationService()
    table_info = create_sample_table_info()
    
    # CSV that removes indexed and constrained columns
    csv_data = pd.DataFrame({
        'id': [1, 2, 3],
        # Removing 'username' (has unique constraint and index)
        # Removing 'email' (has unique constraint and index)
        'created_at': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'new_field1': ['value1', 'value2', 'value3'],
        'new_field2': ['value1', 'value2', 'value3'],
        'new_field3': ['value1', 'value2', 'value3'],
    })
    
    result = await validation_service.validate_data_move(
        table_info=table_info,
        df=csv_data,
        move_type='new_schema'
    )
    
    print(f"Validation Passed: {result.validation_passed}")
    print(f"Columns Removed: {result.schema_analysis.columns_removed}")
    
    print("\nBackward Compatibility Warnings:")
    for warning in result.warnings:
        if 'constraint' in warning.lower() or 'index' in warning.lower():
            print(f"  - {warning}")
    
    print("\nRecommendations:")
    for recommendation in result.recommendations:
        if 'backup' in recommendation.lower() or 'impact' in recommendation.lower():
            print(f"  - {recommendation}")
    
    print()


async def main():
    """Run all demonstration examples."""
    print("New Schema Flexible Validation Examples")
    print("=" * 50)
    print()
    
    await demonstrate_successful_validation()
    await demonstrate_case_conflict_detection()
    await demonstrate_constraint_validation()
    await demonstrate_backward_compatibility_checks()
    
    print("Examples completed!")


if __name__ == '__main__':
    asyncio.run(main())