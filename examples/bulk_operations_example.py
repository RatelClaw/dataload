#!/usr/bin/env python3
"""
Example demonstrating bulk operations functionality in DataMove use case.

This example shows how to use the bulk insert and data replacement operations
with different data types including vectors and JSON data.
"""

import pandas as pd
import numpy as np
import asyncio
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
from dataload.infrastructure.db.db_connection import DBConnection


async def demonstrate_bulk_operations():
    """
    Demonstrate bulk operations with sample data.
    
    Note: This is a demonstration script. In a real scenario, you would:
    1. Set up a proper database connection with credentials
    2. Create the target table first
    3. Handle errors appropriately
    """
    
    print("DataMove Bulk Operations Example")
    print("=" * 40)
    
    # Create sample data with various types
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'score': [95.5, 87.2, 92.1, 88.7, 94.3],
        'active': [True, False, True, True, False],
        'metadata': [
            {'department': 'Engineering', 'level': 'Senior'},
            {'department': 'Marketing', 'level': 'Junior'},
            {'department': 'Engineering', 'level': 'Mid'},
            {'department': 'Sales', 'level': 'Senior'},
            {'department': 'HR', 'level': 'Manager'}
        ],
        'embedding': [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7]
        ]
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Columns: {list(sample_data.columns)}")
    print("\nSample data:")
    print(sample_data.head())
    
    # Demonstrate type conversion capabilities
    print("\n" + "=" * 40)
    print("Type Conversion Demonstration")
    print("=" * 40)
    
    # This would normally be done with a real database connection
    # For demonstration, we'll show the type conversion logic
    
    # Mock repository for demonstration
    mock_db = None  # In real usage: DBConnection(connection_string)
    repo = PostgresDataMoveRepository(mock_db)
    
    # Demonstrate batch size calculation
    optimal_batch_size = repo._calculate_optimal_batch_size(sample_data, target_memory_mb=50)
    print(f"Optimal batch size for this data: {optimal_batch_size}")
    
    # Demonstrate value conversion
    print("\nValue conversion examples:")
    test_values = [
        None,
        [1.0, 2.0, 3.0],  # Vector
        {'key': 'value'},  # JSON
        np.int64(42),      # Numpy integer
        "test_string"      # String
    ]
    
    for value in test_values:
        converted = repo._convert_value_for_postgres(value)
        print(f"  {value} ({type(value)}) -> {converted} ({type(converted)})")
    
    # Demonstrate COPY method feasibility check
    from dataload.domain.entities import TableInfo, ColumnInfo
    
    table_info = TableInfo(
        name='sample_table',
        columns={
            'id': ColumnInfo(name='id', data_type='integer', nullable=False),
            'name': ColumnInfo(name='name', data_type='text', nullable=True),
            'score': ColumnInfo(name='score', data_type='double precision', nullable=True),
            'active': ColumnInfo(name='active', data_type='boolean', nullable=True),
            'metadata': ColumnInfo(name='metadata', data_type='jsonb', nullable=True),
            'embedding': ColumnInfo(name='embedding', data_type='vector(3)', nullable=True),
        },
        primary_keys=['id'],
        constraints=[],
        indexes=[]
    )
    
    can_use_copy = repo._can_use_copy_method(sample_data, table_info)
    print(f"\nCan use PostgreSQL COPY method: {can_use_copy}")
    
    # Demonstrate memory validation
    try:
        repo._validate_memory_usage(sample_data, max_memory_mb=100)
        print("✓ Memory usage validation passed")
    except Exception as e:
        print(f"✗ Memory usage validation failed: {e}")
    
    print("\n" + "=" * 40)
    print("Bulk Operations Features")
    print("=" * 40)
    
    print("✓ Transaction-safe data replacement")
    print("✓ Configurable batch processing")
    print("✓ Automatic type conversion and validation")
    print("✓ Memory usage optimization")
    print("✓ PostgreSQL COPY optimization for simple data")
    print("✓ Batch INSERT for complex data types")
    print("✓ Conflict resolution strategies (error, ignore, update)")
    print("✓ Progress monitoring for large datasets")
    print("✓ Vector dimension validation")
    print("✓ JSON data handling")
    
    print("\n" + "=" * 40)
    print("Usage Example")
    print("=" * 40)
    
    usage_example = '''
# Example usage in a real application:

from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository

# Set up database connection
db_connection = DBConnection("postgresql://user:pass@localhost/db")
repo = PostgresDataMoveRepository(db_connection)

# Replace all data in existing table
rows_inserted = await repo.replace_table_data(
    table_name="my_table",
    df=my_dataframe,
    batch_size=1000
)

# Bulk insert with conflict handling
rows_inserted = await repo.bulk_insert_data(
    table_name="my_table",
    df=my_dataframe,
    batch_size=1000,
    on_conflict="ignore"  # or "error" or "update"
)

print(f"Successfully processed {rows_inserted} rows")
'''
    
    print(usage_example)


if __name__ == "__main__":
    asyncio.run(demonstrate_bulk_operations())