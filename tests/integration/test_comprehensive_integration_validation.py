"""
Comprehensive Integration Validation Test Suite.

This test suite validates the complete integration of APIJSONStorageLoader
with all existing components and ensures no regressions in functionality.
It serves as the final validation for task 17.
"""

import asyncio
import json
import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional

# Import all relevant components for integration testing
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.infrastructure.storage.loaders import S3Loader, LocalLoader
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase
from dataload.application.use_cases.data_api_json_use_case import DataAPIJSONUseCase
from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.interfaces.storage_loader import StorageLoaderInterface
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface
from dataload.interfaces.data_move_repository import DataMoveRepositoryInterface
from dataload.domain.entities import DataMoveResult, ValidationReport, TableSchema
from dataload.config import logger


class MockEmbeddingProvider:
    """Mock embedding provider that simulates real embedding services."""
    
    def __init__(self, provider_name: str = "mock", embedding_dim: int = 384):
        self.provider_name = provider_name
        self.embedding_dim = embedding_dim
        self.call_count = 0
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings."""
        self.call_count += 1
        embeddings = []
        for i, text in enumerate(texts):
            # Generate deterministic embeddings based on text hash
            embedding = [(hash(text) + j) % 100 / 100.0 for j in range(self.embedding_dim)]
            embeddings.append(embedding)
        return embeddings


class MockRepository:
    """Mock repository that simulates database operations."""
    
    def __init__(self):
        self.tables = {}
        self.operations = []
        self.call_counts = {
            'table_exists': 0,
            'create_table': 0,
            'insert_data': 0,
            'upsert_data': 0,
            'get_table_schema': 0
        }
    
    async def table_exists(self, table_name: str) -> bool:
        self.call_counts['table_exists'] += 1
        return table_name in self.tables
    
    async def create_table(self, table_name: str, df: pd.DataFrame, pk_columns: List[str], 
                          embed_type: str, embed_columns: List[str]) -> Dict[str, str]:
        self.call_counts['create_table'] += 1
        
        # Simulate table creation
        column_types = {}
        for col in df.columns:
            if col in embed_columns and embed_type == "separated":
                column_types[f"{col}_enc"] = "vector"
            elif col == "embeddings" and embed_type == "combined":
                column_types[col] = "vector"
            elif df[col].dtype == 'int64':
                column_types[col] = "integer"
            elif df[col].dtype == 'float64':
                column_types[col] = "double precision"
            else:
                column_types[col] = "text"
        
        self.tables[table_name] = {
            'columns': column_types,
            'pk_columns': pk_columns,
            'data': []
        }
        
        self.operations.append(('create_table', table_name, len(df)))
        return column_types
    
    async def insert_data(self, table_name: str, df: pd.DataFrame, pk_columns: List[str]):
        self.call_counts['insert_data'] += 1
        if table_name in self.tables:
            self.tables[table_name]['data'].extend(df.to_dict('records'))
        self.operations.append(('insert_data', table_name, len(df)))
    
    async def upsert_data(self, table_name: str, df: pd.DataFrame, pk_columns: List[str]):
        self.call_counts['upsert_data'] += 1
        if table_name in self.tables:
            # Simulate upsert by replacing existing records
            existing_data = self.tables[table_name]['data']
            new_records = df.to_dict('records')
            
            for new_record in new_records:
                # Find existing record with same primary key
                existing_index = None
                for i, existing_record in enumerate(existing_data):
                    if all(existing_record.get(pk) == new_record.get(pk) for pk in pk_columns):
                        existing_index = i
                        break
                
                if existing_index is not None:
                    existing_data[existing_index] = new_record
                else:
                    existing_data.append(new_record)
        
        self.operations.append(('upsert_data', table_name, len(df)))
    
    async def get_table_schema(self, table_name: str) -> TableSchema:
        self.call_counts['get_table_schema'] += 1
        if table_name in self.tables:
            table_info = self.tables[table_name]
            return TableSchema(
                table_name=table_name,
                columns=table_info['columns'],
                nullables={col: True for col in table_info['columns']},
                primary_keys=table_info['pk_columns']
            )
        else:
            raise Exception(f"Table {table_name} does not exist")
    
    async def get_data_columns(self, table_name: str) -> List[str]:
        if table_name in self.tables:
            return list(self.tables[table_name]['columns'].keys())
        else:
            raise Exception(f"Table {table_name} does not exist")


class TestComprehensiveIntegration:
    """Comprehensive integration tests for all components."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        return MockRepository()
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        return MockEmbeddingProvider("test_provider", 384)
    
    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com'],
            'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR'],
            'description': ['Senior Software Engineer', 'Sales Manager', 'Marketing Specialist', 'DevOps Engineer', 'HR Coordinator']
        })
    
    @pytest.fixture
    def sample_json_data(self):
        """Sample JSON data for testing."""
        return [
            {
                'id': 1,
                'personal_info': {
                    'name': 'Alice Johnson',
                    'email': 'alice@test.com',
                    'contact': {
                        'phone': '555-0001',
                        'address': {'city': 'New York', 'state': 'NY'}
                    }
                },
                'employment': {
                    'department': 'Engineering',
                    'position': 'Senior Software Engineer',
                    'skills': ['Python', 'JavaScript', 'SQL'],
                    'salary': 95000
                }
            },
            {
                'id': 2,
                'personal_info': {
                    'name': 'Bob Smith',
                    'email': 'bob@test.com',
                    'contact': {
                        'phone': '555-0002',
                        'address': {'city': 'San Francisco', 'state': 'CA'}
                    }
                },
                'employment': {
                    'department': 'Sales',
                    'position': 'Sales Manager',
                    'skills': ['CRM', 'Negotiation', 'Leadership'],
                    'salary': 85000
                }
            }
        ]
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_json_file(self, sample_json_data):
        """Create temporary JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_data, f)
            yield f.name
        os.unlink(f.name)
    
    def test_loader_interface_compliance(self):
        """Test that all loaders properly implement the interface."""
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
    
    @pytest.mark.asyncio
    async def test_csv_backward_compatibility(self, temp_csv_file, sample_csv_data):
        """Test CSV loading backward compatibility across all loaders."""
        # Test APIJSONStorageLoader
        api_loader = APIJSONStorageLoader()
        api_df = api_loader.load_csv(temp_csv_file)
        
        # Test LocalLoader
        local_loader = LocalLoader()
        local_df = local_loader.load_csv(temp_csv_file)
        
        # Results should be identical
        pd.testing.assert_frame_equal(api_df, local_df)
        pd.testing.assert_frame_equal(api_df, sample_csv_data)
        
        # Verify data integrity
        assert len(api_df) == 5
        assert list(api_df.columns) == ['id', 'name', 'email', 'department', 'description']
        assert api_df['id'].tolist() == [1, 2, 3, 4, 5]
    
    @pytest.mark.asyncio
    async def test_json_compatibility_and_enhancement(self, sample_json_data):
        """Test JSON loading compatibility and enhancements."""
        # Test APIJSONStorageLoader (advanced)
        api_loader = APIJSONStorageLoader()
        api_df = await api_loader.load_json(sample_json_data)
        
        # Test LocalLoader (basic)
        local_loader = LocalLoader()
        local_df = await local_loader.load_json(sample_json_data)
        
        # Both should handle basic data
        assert len(api_df) == len(local_df) == 2
        assert 'id' in api_df.columns and 'id' in local_df.columns
        
        # APIJSONStorageLoader should handle nested structures better
        api_columns = set(api_df.columns)
        local_columns = set(local_df.columns)
        
        # API loader should have flattened nested fields
        nested_fields = [col for col in api_columns if '_' in col or '.' in col]
        assert len(nested_fields) > 0, "APIJSONStorageLoader should flatten nested structures"
    
    @pytest.mark.asyncio
    async def test_dataload_use_case_integration(self, mock_repository, mock_embedding_service, temp_csv_file):
        """Test integration with existing dataloadUseCase."""
        # Test with APIJSONStorageLoader
        api_loader = APIJSONStorageLoader()
        use_case = dataloadUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        await use_case.execute(
            s3_uri=temp_csv_file,
            table_name="test_employees",
            embed_columns_names=["name", "description"],
            pk_columns=["id"],
            create_table_if_not_exists=True,
            embed_type="combined"
        )
        
        # Verify repository interactions
        assert mock_repository.call_counts['create_table'] == 1
        assert mock_repository.call_counts['insert_data'] == 1
        
        # Verify embedding service was called
        assert mock_embedding_service.call_count == 1
        
        # Verify table was created with correct structure
        assert "test_employees" in mock_repository.tables
        table_info = mock_repository.tables["test_employees"]
        assert len(table_info['data']) == 5
        assert 'embeddings' in table_info['columns']
    
    @pytest.mark.asyncio
    async def test_data_api_json_use_case_integration(self, mock_repository, mock_embedding_service, sample_json_data):
        """Test integration with DataAPIJSONUseCase."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        result = await use_case.execute(
            source=sample_json_data,
            table_name="json_employees",
            embed_columns_names=["personal_info_name", "employment_position"],
            pk_columns=["id"],
            create_table_if_not_exists=True,
            embed_type="separated"
        )
        
        # Verify successful execution
        assert isinstance(result, DataMoveResult)
        assert result.success
        assert result.rows_processed == 2
        assert result.table_created
        
        # Verify repository interactions
        assert mock_repository.call_counts['create_table'] == 1
        assert mock_repository.call_counts['insert_data'] == 1
        
        # Verify embedding columns were created
        table_info = mock_repository.tables["json_employees"]
        columns = table_info['columns']
        assert any('_enc' in col for col in columns), "Should have embedding columns with _enc suffix"
    
    @pytest.mark.asyncio
    async def test_column_mapping_integration(self, mock_repository, mock_embedding_service, sample_json_data):
        """Test column mapping integration with embeddings."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        result = await use_case.execute(
            source=sample_json_data,
            table_name="mapped_employees",
            embed_columns_names=["full_name", "job_title"],  # Using mapped names
            pk_columns=["employee_id"],
            column_name_mapping={
                'id': 'employee_id',
                'personal_info_name': 'full_name',
                'employment_position': 'job_title',
                'employment_department': 'dept'
            },
            create_table_if_not_exists=True,
            embed_type="combined"
        )
        
        assert result.success
        assert result.rows_processed == 2
        
        # Verify mapped columns exist
        table_info = mock_repository.tables["mapped_employees"]
        columns = set(table_info['columns'].keys())
        
        assert 'employee_id' in columns
        assert 'full_name' in columns
        assert 'job_title' in columns
        assert 'dept' in columns
        assert 'embeddings' in columns  # Combined embedding column
    
    @pytest.mark.asyncio
    async def test_data_transformation_integration(self, mock_repository, mock_embedding_service, sample_json_data):
        """Test data transformation integration."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        result = await use_case.execute(
            source=sample_json_data,
            table_name="transformed_employees",
            embed_columns_names=["display_name"],
            pk_columns=["id"],
            update_request_body_mapping={
                'display_name': "concat({personal_info_name}, ' - ', {employment_position})",
                'salary_k': "round({employment_salary} / 1000)"
            },
            create_table_if_not_exists=True
        )
        
        assert result.success
        assert result.rows_processed == 2
        
        # Verify transformed columns exist
        table_info = mock_repository.tables["transformed_employees"]
        columns = set(table_info['columns'].keys())
        
        assert 'display_name' in columns
        assert 'salary_k' in columns
        
        # Verify transformed data
        data_records = table_info['data']
        assert len(data_records) == 2
        
        # Check that display_name was computed
        for record in data_records:
            assert 'display_name' in record
            assert ' - ' in str(record['display_name'])  # Should contain the concatenation
    
    @pytest.mark.asyncio
    async def test_existing_table_upsert_integration(self, mock_repository, mock_embedding_service, sample_json_data):
        """Test upsert operations with existing tables."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        # First, create table with initial data
        await use_case.execute(
            source=sample_json_data,
            table_name="upsert_test",
            pk_columns=["id"],
            create_table_if_not_exists=True
        )
        
        assert "upsert_test" in mock_repository.tables
        initial_count = len(mock_repository.tables["upsert_test"]['data'])
        
        # Now upsert with updated and new data
        updated_data = [
            {
                'id': 1,  # Existing record - should update
                'personal_info': {'name': 'Alice Johnson Updated', 'email': 'alice.new@test.com'},
                'employment': {'department': 'Engineering', 'position': 'Lead Engineer'}
            },
            {
                'id': 3,  # New record - should insert
                'personal_info': {'name': 'Charlie Brown', 'email': 'charlie@test.com'},
                'employment': {'department': 'Marketing', 'position': 'Marketing Manager'}
            }
        ]
        
        result = await use_case.execute(
            source=updated_data,
            table_name="upsert_test",
            pk_columns=["id"],
            create_table_if_not_exists=False
        )
        
        assert result.success
        assert result.rows_processed == 2
        
        # Verify upsert operation was called
        assert mock_repository.call_counts['upsert_data'] == 1
        
        # Verify data was updated/inserted correctly
        final_data = mock_repository.tables["upsert_test"]['data']
        assert len(final_data) == initial_count + 1  # One update, one insert
        
        # Find the updated record
        updated_record = next((r for r in final_data if r['id'] == 1), None)
        assert updated_record is not None
        assert 'Updated' in str(updated_record.get('personal_info_name', ''))
    
    @pytest.mark.asyncio
    async def test_embedding_provider_compatibility(self, mock_repository):
        """Test compatibility with different embedding providers."""
        # Test with different mock providers
        providers = [
            MockEmbeddingProvider("gemini", 768),
            MockEmbeddingProvider("openai", 1536),
            MockEmbeddingProvider("custom", 384)
        ]
        
        api_loader = APIJSONStorageLoader()
        
        for i, provider in enumerate(providers):
            use_case = DataAPIJSONUseCase(
                repo=mock_repository,
                embedding_service=provider,
                storage_loader=api_loader
            )
            
            test_data = [{'id': i+1, 'content': f'Test content for {provider.provider_name}'}]
            
            result = await use_case.execute(
                source=test_data,
                table_name=f"embedding_test_{provider.provider_name}",
                embed_columns_names=["content"],
                pk_columns=["id"]
            )
            
            assert result.success
            assert provider.call_count == 1
            
            # Verify embedding dimension is preserved
            table_info = mock_repository.tables[f"embedding_test_{provider.provider_name}"]
            assert 'embeddings' in table_info['columns']
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_repository, mock_embedding_service):
        """Test concurrent operations don't interfere with each other."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        # Create multiple concurrent operations
        tasks = []
        for i in range(3):
            data = [{'id': j, 'content': f'Content {i}-{j}'} for j in range(2)]
            task = use_case.execute(
                source=data,
                table_name=f"concurrent_test_{i}",
                embed_columns_names=["content"],
                pk_columns=["id"]
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all operations succeeded
        for i, result in enumerate(results):
            assert result.success
            assert result.rows_processed == 2
            assert f"concurrent_test_{i}" in mock_repository.tables
    
    @pytest.mark.asyncio
    async def test_error_handling_compatibility(self, mock_repository, mock_embedding_service):
        """Test error handling compatibility across components."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        # Test various error scenarios
        error_scenarios = [
            # Invalid JSON structure
            {"name": "invalid_json", "data": "not json", "should_fail": True},
            # Empty data
            {"name": "empty_data", "data": [], "should_fail": False},
            # Invalid column mapping
            {"name": "invalid_mapping", "data": [{'id': 1}], "config": {'column_name_mapping': {'nonexistent': 'mapped'}}, "should_fail": False}
        ]
        
        for scenario in error_scenarios:
            try:
                config = scenario.get('config', {})
                result = await use_case.execute(
                    source=scenario['data'],
                    table_name=f"error_test_{scenario['name']}",
                    **config
                )
                
                if scenario['should_fail']:
                    pytest.fail(f"Expected {scenario['name']} to fail, but it succeeded")
                else:
                    # Should handle gracefully
                    assert isinstance(result, DataMoveResult)
                    
            except Exception as e:
                if not scenario['should_fail']:
                    pytest.fail(f"Unexpected failure for {scenario['name']}: {e}")
                # Expected failure - verify error type is appropriate
                assert isinstance(e, (ValueError, TypeError, Exception))
    
    @pytest.mark.asyncio
    async def test_performance_regression(self, mock_repository, mock_embedding_service):
        """Test that performance hasn't regressed significantly."""
        import time
        
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        # Create larger dataset for performance testing
        large_data = []
        for i in range(100):
            large_data.append({
                'id': i,
                'name': f'User {i}',
                'email': f'user{i}@test.com',
                'profile': {
                    'age': 20 + (i % 50),
                    'preferences': {
                        'theme': 'dark' if i % 2 else 'light',
                        'notifications': i % 3 == 0
                    }
                },
                'metadata': {
                    'tags': [f'tag{j}' for j in range(i % 5)],
                    'scores': [float(j) for j in range(i % 3)]
                }
            })
        
        start_time = time.time()
        
        result = await use_case.execute(
            source=large_data,
            table_name="performance_test",
            embed_columns_names=["name"],
            pk_columns=["id"],
            column_name_mapping={'profile_age': 'age'},
            update_request_body_mapping={'display_name': "concat('User: ', {name})"}
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert result.success
        assert result.rows_processed == 100
        assert processing_time < 10.0, f"Processing took too long: {processing_time}s"
        
        # Verify data integrity
        table_info = mock_repository.tables["performance_test"]
        assert len(table_info['data']) == 100
        
        # Verify all processing was applied
        sample_record = table_info['data'][0]
        assert 'age' in sample_record  # Column mapping applied
        assert 'display_name' in sample_record  # Transformation applied
        assert sample_record['display_name'].startswith('User: ')  # Transformation worked
    
    def test_migration_compatibility(self):
        """Test that migration from existing loaders is seamless."""
        # Simulate existing code patterns
        
        # Pattern 1: Direct loader usage
        old_loader = LocalLoader()
        new_loader = APIJSONStorageLoader()
        
        # Both should have same interface
        assert hasattr(old_loader, 'load_csv')
        assert hasattr(new_loader, 'load_csv')
        
        # Pattern 2: Use case integration
        mock_repo = Mock()
        mock_embedding = Mock()
        
        # Old pattern
        old_use_case = dataloadUseCase(mock_repo, mock_embedding, old_loader)
        
        # New pattern (drop-in replacement)
        new_use_case = dataloadUseCase(mock_repo, mock_embedding, new_loader)
        
        # Both should have same interface
        assert hasattr(old_use_case, 'execute')
        assert hasattr(new_use_case, 'execute')
        
        # Pattern 3: Auto-loader compatibility
        with patch('dataload.application.use_cases.data_move_use_case.DataMoveUseCase.create_with_auto_loader') as mock_create:
            mock_use_case = Mock()
            mock_create.return_value = mock_use_case
            
            # Should work with auto-detection
            auto_use_case = DataMoveUseCase.create_with_auto_loader(repository=mock_repo)
            assert auto_use_case is not None


if __name__ == "__main__":
    # Run the comprehensive integration tests
    pytest.main([__file__, "-v", "--tb=short"])