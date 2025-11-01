#!/usr/bin/env python3
"""
Example demonstrating vector column handling in DataMove use case.

This example shows how the DataMove use case properly detects, validates,
and handles vector columns during data migration operations.
"""

import pandas as pd
import numpy as np
import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository


def create_sample_data_with_vectors():
    """Create sample DataFrame with various vector formats."""
    
    # Sample data with different vector representations
    data = {
        'id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'embedding_3d': [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6], 
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2]
        ],
        'description_vector': [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([5.0, 6.0, 7.0, 8.0]),
            np.array([9.0, 10.0, 11.0, 12.0]),
            np.array([13.0, 14.0, 15.0, 16.0])
        ],
        'text_vectors': [
            '[0.1, 0.2]',
            '0.3, 0.4',
            '[0.5, 0.6]',
            '0.7, 0.8'
        ],
        'category': ['A', 'B', 'A', 'C']
    }
    
    return pd.DataFrame(data)


async def demonstrate_vector_handling():
    """Demonstrate vector column detection and handling."""
    
    print("=== Vector Column Handling Example ===\n")
    
    # Create sample data
    df = create_sample_data_with_vectors()
    print("Sample DataFrame with vector columns:")
    print(df)
    print(f"\nDataFrame info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print()
    
    # Initialize repository (without actual DB connection for demo)
    repo = PostgresDataMoveRepository(None)
    
    print("=== 1. Vector Type Detection ===")
    for col in df.columns:
        inferred_type = repo._infer_postgres_type(df[col])
        print(f"  Column '{col}': {inferred_type}")
    print()
    
    print("=== 2. Vector Dimension Extraction ===")
    test_types = ['vector(3)', 'vector(1536)', 'vector(4)', 'text', 'integer']
    for test_type in test_types:
        dimension = repo._extract_vector_dimension(test_type)
        print(f"  {test_type} -> dimension: {dimension}")
    print()
    
    print("=== 3. Vector String Parsing ===")
    test_vectors = [
        '[0.1, 0.2, 0.3]',
        '0.1,0.2,0.3',
        '[1.0, 2.0, 3.0, 4.0]',
        '1.0, 2.0, 3.0, 4.0'
    ]
    
    for vector_str in test_vectors:
        try:
            parsed = repo._parse_vector_string(vector_str)
            print(f"  '{vector_str}' -> {parsed} (dim: {len(parsed)})")
        except Exception as e:
            print(f"  '{vector_str}' -> ERROR: {e}")
    print()
    
    print("=== 4. Vector Validation ===")
    
    # Test valid vectors
    errors = repo._validate_dataframe_vector_column(df, 'embedding_3d', 3)
    print(f"  'embedding_3d' column (expected dim 3): {len(errors)} errors")
    
    # Test dimension mismatch
    errors = repo._validate_dataframe_vector_column(df, 'embedding_3d', 4)
    print(f"  'embedding_3d' column (expected dim 4): {len(errors)} errors")
    if errors:
        print(f"    First error: {errors[0]}")
    
    # Test text vector parsing
    errors = repo._validate_dataframe_vector_column(df, 'text_vectors', 2)
    print(f"  'text_vectors' column (expected dim 2): {len(errors)} errors")
    print()
    
    print("=== 5. Vector Data Conversion ===")
    vector_columns = [('embedding_3d', 3), ('description_vector', 4), ('text_vectors', 2)]
    
    print("Original vector data types:")
    for col, _ in vector_columns:
        if col in df.columns:
            sample_val = df[col].iloc[0]
            print(f"  {col}: {type(sample_val)} - {sample_val}")
    
    df_converted = repo._convert_vector_data(df, vector_columns)
    
    print("\nConverted vector data types:")
    for col, _ in vector_columns:
        if col in df_converted.columns:
            sample_val = df_converted[col].iloc[0]
            print(f"  {col}: {type(sample_val)} - {sample_val}")
    print()
    
    print("=== 6. Vector Index Recommendations ===")
    print("Based on vector dimensions, recommended index types:")
    for col, dim in vector_columns:
        if dim <= 2000:
            index_type = "HNSW (better for dimensions <= 2000)"
        else:
            index_type = "IVFFlat (required for dimensions > 2000)"
        print(f"  {col} (dim {dim}): {index_type}")
    print()
    
    print("=== Vector Handling Demo Complete ===")
    print("\nKey Features Demonstrated:")
    print("✓ Automatic vector column detection")
    print("✓ Vector dimension extraction and validation")
    print("✓ Multiple vector format support (list, numpy array, string)")
    print("✓ Vector data type conversion for PostgreSQL")
    print("✓ Dimension mismatch detection")
    print("✓ Appropriate index type selection")


if __name__ == "__main__":
    asyncio.run(demonstrate_vector_handling())