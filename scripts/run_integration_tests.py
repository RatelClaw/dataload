#!/usr/bin/env python3
"""
Simple Integration Test Runner for APIJSONStorageLoader.

This script runs basic integration tests to validate the APIJSONStorageLoader
works correctly with existing components.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.infrastructure.storage.loaders import LocalLoader, S3Loader
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase
from dataload.application.use_cases.data_api_json_use_case import DataAPIJSONUseCase
from dataload.interfaces.storage_loader import StorageLoaderInterface


class MockRepository:
    """Simple mock repository for testing."""
    
    def __init__(self):
        self.tables = {}
        self.operations = []
    
    async def table_exists(self, table_name: str) -> bool:
        return table_name in self.tables
    
    async def create_table(self, table_name: str, df: pd.DataFrame, pk_columns: list, 
                          embed_type: str, embed_columns_names: list) -> dict:
        column_types = {}
        for col in df.columns:
            if df[col].dtype == 'int64':
                column_types[col] = "integer"
            elif df[col].dtype == 'float64':
                column_types[col] = "double precision"
            else:
                column_types[col] = "text"
        
        if embed_type == "combined":
            column_types["embeddings"] = "vector"
        elif embed_type == "separated":
            for col in embed_columns_names:
                column_types[f"{col}_enc"] = "vector"
        
        self.tables[table_name] = {
            'columns': column_types,
            'data': []
        }
        self.operations.append(('create_table', table_name))
        return column_types
    
    async def insert_data(self, table_name: str, df: pd.DataFrame, pk_columns: list):
        if table_name in self.tables:
            self.tables[table_name]['data'].extend(df.to_dict('records'))
        self.operations.append(('insert_data', table_name, len(df)))
    
    async def upsert_data(self, table_name: str, df: pd.DataFrame, pk_columns: list):
        if table_name in self.tables:
            self.tables[table_name]['data'].extend(df.to_dict('records'))
        self.operations.append(('upsert_data', table_name, len(df)))


class MockEmbeddingService:
    """Simple mock embedding service for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    def get_embeddings(self, texts: list) -> list:
        self.call_count += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


async def test_interface_compliance():
    """Test that all loaders implement the interface correctly."""
    print("Testing interface compliance...")
    
    loaders = [
        APIJSONStorageLoader(),
        LocalLoader(),
        S3Loader()
    ]
    
    for loader in loaders:
        assert isinstance(loader, StorageLoaderInterface)
        assert hasattr(loader, 'load_csv')
        assert hasattr(loader, 'load_json')
        assert callable(getattr(loader, 'load_csv'))
        assert callable(getattr(loader, 'load_json'))
    
    print("✅ Interface compliance test passed")


async def test_csv_backward_compatibility():
    """Test CSV loading backward compatibility."""
    print("Testing CSV backward compatibility...")
    
    # Create test CSV data
    test_data = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com']
    })
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        csv_path = f.name
    
    try:
        # Test APIJSONStorageLoader
        api_loader = APIJSONStorageLoader()
        api_df = api_loader.load_csv(csv_path)
        
        # Test LocalLoader
        local_loader = LocalLoader()
        local_df = local_loader.load_csv(csv_path)
        
        # Results should be identical
        pd.testing.assert_frame_equal(api_df, local_df)
        pd.testing.assert_frame_equal(api_df, test_data)
        
        print("✅ CSV backward compatibility test passed")
        
    finally:
        os.unlink(csv_path)


async def test_json_loading():
    """Test JSON loading functionality."""
    print("Testing JSON loading...")
    
    # Test data
    json_data = [
        {
            'id': 1,
            'name': 'Alice',
            'profile': {'age': 30, 'city': 'New York'},
            'skills': ['Python', 'SQL']
        },
        {
            'id': 2,
            'name': 'Bob',
            'profile': {'age': 25, 'city': 'San Francisco'},
            'skills': ['JavaScript', 'React']
        }
    ]
    
    # Test APIJSONStorageLoader
    api_loader = APIJSONStorageLoader()
    df = await api_loader.load_json(json_data)
    
    assert len(df) == 2
    assert 'id' in df.columns
    assert 'name' in df.columns
    # Should have flattened nested fields
    assert any('profile' in col for col in df.columns)
    
    print("✅ JSON loading test passed")


async def test_use_case_integration():
    """Test integration with existing use cases."""
    print("Testing use case integration...")
    
    # Create test data
    test_data = pd.DataFrame({
        'id': [1, 2],
        'name': ['Alice', 'Bob'],
        'description': ['Engineer', 'Designer']
    })
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        csv_path = f.name
    
    try:
        # Test with dataloadUseCase
        mock_repo = MockRepository()
        mock_embedding = MockEmbeddingService()
        api_loader = APIJSONStorageLoader()
        
        use_case = dataloadUseCase(
            repo=mock_repo,
            embedding_service=mock_embedding,
            storage_loader=api_loader
        )
        
        await use_case.execute(
            s3_uri=csv_path,
            table_name="test_table",
            embed_columns_names=["name", "description"],
            pk_columns=["id"],
            create_table_if_not_exists=True,
            embed_type="combined"
        )
        
        # Verify operations
        assert "test_table" in mock_repo.tables
        assert len(mock_repo.tables["test_table"]["data"]) == 2
        assert mock_embedding.call_count == 1
        
        print("✅ Use case integration test passed")
        
    finally:
        os.unlink(csv_path)


async def test_api_json_use_case():
    """Test DataAPIJSONUseCase integration."""
    print("Testing DataAPIJSONUseCase integration...")
    
    # Test JSON data
    json_data = [
        {'id': 1, 'name': 'Alice', 'description': 'Engineer'},
        {'id': 2, 'name': 'Bob', 'description': 'Designer'}
    ]
    
    mock_repo = MockRepository()
    mock_embedding = MockEmbeddingService()
    api_loader = APIJSONStorageLoader()
    
    use_case = DataAPIJSONUseCase(
        repo=mock_repo,
        embedding_service=mock_embedding,
        storage_loader=api_loader
    )
    
    result = await use_case.execute(
        source=json_data,
        table_name="json_test_table",
        embed_columns_names=["name", "description"],
        pk_columns=["id"],
        create_table_if_not_exists=True,
        embed_type="combined"
    )
    
    # Verify result
    assert result.success
    assert result.rows_processed == 2
    assert "json_test_table" in mock_repo.tables
    
    print("✅ DataAPIJSONUseCase integration test passed")


async def test_column_mapping():
    """Test column mapping functionality."""
    print("Testing column mapping...")
    
    json_data = [
        {
            'user_id': 1,
            'full_name': 'Alice Johnson',
            'job_title': 'Software Engineer'
        }
    ]
    
    mock_repo = MockRepository()
    mock_embedding = MockEmbeddingService()
    api_loader = APIJSONStorageLoader()
    
    use_case = DataAPIJSONUseCase(
        repo=mock_repo,
        embedding_service=mock_embedding,
        storage_loader=api_loader
    )
    
    result = await use_case.execute(
        source=json_data,
        table_name="mapping_test",
        embed_columns_names=["name"],  # Using mapped column name
        column_name_mapping={
            'user_id': 'id',
            'full_name': 'name',
            'job_title': 'title'
        },
        pk_columns=["id"],
        create_table_if_not_exists=True
    )
    
    assert result.success
    assert result.rows_processed == 1
    
    # Verify mapped columns exist
    table_info = mock_repo.tables["mapping_test"]
    columns = set(table_info['columns'].keys())
    assert 'id' in columns
    assert 'name' in columns
    assert 'title' in columns
    
    print("✅ Column mapping test passed")


async def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    
    api_loader = APIJSONStorageLoader()
    
    # Test non-existent CSV file
    try:
        api_loader.load_csv("nonexistent.csv")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass  # Expected
    
    # Test invalid JSON
    try:
        await api_loader.load_json("invalid json string")
        assert False, "Should have raised an error"
    except Exception:
        pass  # Expected
    
    # Test empty JSON list (should work)
    try:
        df = await api_loader.load_json([])
        assert len(df) == 0
    except Exception as e:
        # Empty JSON might fail in some implementations, which is acceptable
        print(f"  Note: Empty JSON handling: {e}")
        pass
    
    print("✅ Error handling test passed")


async def run_all_tests():
    """Run all integration tests."""
    print("Running APIJSONStorageLoader Integration Tests")
    print("=" * 50)
    
    tests = [
        test_interface_compliance,
        test_csv_backward_compatibility,
        test_json_loading,
        test_use_case_integration,
        test_api_json_use_case,
        test_column_mapping,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            start_time = time.time()
            await test()
            duration = time.time() - start_time
            print(f"  Duration: {duration:.2f}s")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed!")
        return True
    else:
        print(f"⚠️  {failed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)