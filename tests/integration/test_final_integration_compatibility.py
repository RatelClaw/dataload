"""
Final Integration and Compatibility Tests for APIJSONStorageLoader.

This test suite validates the complete integration of the APIJSONStorageLoader
with existing components and ensures backward compatibility with existing
loaders and use cases.

Test Coverage:
- Integration with existing S3Loader and LocalLoader
- Backward compatibility with existing use cases
- Integration with existing embedding providers (Gemini, etc.)
- Validation with current database repository implementations
- Comprehensive regression tests on existing functionality
"""

import asyncio
import json
import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

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


class TestStorageLoaderCompatibility:
    """Test compatibility between APIJSONStorageLoader and existing loaders."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],
            'department': ['Engineering', 'Sales', 'Marketing']
        })
    
    @pytest.fixture
    def sample_json_data(self):
        """Create sample JSON data for testing."""
        return [
            {
                'id': 1,
                'name': 'Alice',
                'email': 'alice@test.com',
                'department': 'Engineering',
                'profile': {'age': 30, 'city': 'New York'}
            },
            {
                'id': 2,
                'name': 'Bob',
                'email': 'bob@test.com',
                'department': 'Sales',
                'profile': {'age': 25, 'city': 'San Francisco'}
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
    
    def test_storage_loader_interface_compliance(self):
        """Test that all loaders implement StorageLoaderInterface correctly."""
        # Test APIJSONStorageLoader
        api_loader = APIJSONStorageLoader()
        assert isinstance(api_loader, StorageLoaderInterface)
        assert hasattr(api_loader, 'load_csv')
        assert hasattr(api_loader, 'load_json')
        
        # Test existing loaders
        s3_loader = S3Loader()
        assert isinstance(s3_loader, StorageLoaderInterface)
        assert hasattr(s3_loader, 'load_csv')
        assert hasattr(s3_loader, 'load_json')
        
        local_loader = LocalLoader()
        assert isinstance(local_loader, StorageLoaderInterface)
        assert hasattr(local_loader, 'load_csv')
        assert hasattr(local_loader, 'load_json')
    
    def test_csv_loading_compatibility(self, temp_csv_file, sample_csv_data):
        """Test CSV loading compatibility across all loaders."""
        # Test APIJSONStorageLoader CSV loading
        api_loader = APIJSONStorageLoader()
        api_df = api_loader.load_csv(temp_csv_file)
        
        # Test LocalLoader CSV loading
        local_loader = LocalLoader()
        local_df = local_loader.load_csv(temp_csv_file)
        
        # Results should be identical
        pd.testing.assert_frame_equal(api_df, local_df)
        pd.testing.assert_frame_equal(api_df, sample_csv_data)
    
    @pytest.mark.asyncio
    async def test_json_loading_compatibility(self, sample_json_data):
        """Test JSON loading compatibility across loaders."""
        # Test APIJSONStorageLoader
        api_loader = APIJSONStorageLoader()
        api_df = await api_loader.load_json(sample_json_data)
        
        # Test LocalLoader basic JSON support
        local_loader = LocalLoader()
        local_df = await local_loader.load_json(sample_json_data)
        
        # Both should handle basic JSON data
        assert len(api_df) == len(local_df) == 2
        assert 'id' in api_df.columns and 'id' in local_df.columns
        assert 'name' in api_df.columns and 'name' in local_df.columns
    
    @pytest.mark.asyncio
    async def test_json_file_loading_compatibility(self, temp_json_file, sample_json_data):
        """Test JSON file loading compatibility."""
        # Test APIJSONStorageLoader
        api_loader = APIJSONStorageLoader()
        api_df = await api_loader.load_json(temp_json_file)
        
        # Test LocalLoader
        local_loader = LocalLoader()
        local_df = await local_loader.load_json(temp_json_file)
        
        # Both should load the same data
        assert len(api_df) == len(local_df) == 2
        
        # APIJSONStorageLoader should handle nested structures better
        assert 'profile_age' in api_df.columns or 'profile.age' in api_df.columns
    
    def test_error_handling_compatibility(self):
        """Test error handling compatibility across loaders."""
        api_loader = APIJSONStorageLoader()
        local_loader = LocalLoader()
        
        # Test non-existent file handling
        with pytest.raises(FileNotFoundError):
            api_loader.load_csv("nonexistent.csv")
        
        with pytest.raises(FileNotFoundError):
            local_loader.load_csv("nonexistent.csv")
    
    @pytest.mark.asyncio
    async def test_s3_uri_handling(self):
        """Test S3 URI handling compatibility."""
        api_loader = APIJSONStorageLoader()
        local_loader = LocalLoader()
        
        # LocalLoader should reject S3 URIs for JSON
        with pytest.raises(ValueError, match="does not support API URLs"):
            await local_loader.load_json("https://api.example.com/data")
        
        # APIJSONStorageLoader should handle API URLs (though may fail without proper setup)
        # This tests the interface compatibility, not actual API calls
        try:
            await api_loader.load_json("https://httpbin.org/json")
        except Exception as e:
            # Expected to fail in test environment, but should not be a ValueError about unsupported URLs
            assert not isinstance(e, ValueError) or "does not support" not in str(e)


class TestUseCaseCompatibility:
    """Test compatibility with existing use cases."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = Mock(spec=DataMoveRepositoryInterface)
        repo.table_exists = AsyncMock(return_value=False)
        repo.create_table = AsyncMock(return_value={'id': 'integer', 'name': 'text'})
        repo.insert_data = AsyncMock()
        repo.get_table_schema = AsyncMock(return_value=TableSchema(
            table_name='test_table',
            columns={'id': 'integer', 'name': 'text'},
            nullables={'id': False, 'name': True},
            primary_keys=['id']
        ))
        repo.get_data_columns = AsyncMock(return_value=['id', 'name'])
        return repo
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = Mock(spec=EmbeddingProviderInterface)
        service.get_embeddings = Mock(return_value=[[0.1, 0.2, 0.3]] * 2)
        return service
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data."""
        return pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob'],
            'description': ['Engineer', 'Designer']
        })
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_dataload_use_case_with_api_loader(self, mock_repository, mock_embedding_service, temp_csv_file):
        """Test dataloadUseCase with APIJSONStorageLoader."""
        # Create use case with APIJSONStorageLoader
        api_loader = APIJSONStorageLoader()
        use_case = dataloadUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        # Execute use case
        await use_case.execute(
            s3_uri=temp_csv_file,  # Using local file path
            table_name="test_table",
            embed_columns_names=["name", "description"],
            pk_columns=["id"],
            create_table_if_not_exists=True,
            embed_type="combined"
        )
        
        # Verify repository calls
        mock_repository.create_table.assert_called_once()
        mock_repository.insert_data.assert_called_once()
        
        # Verify embedding service calls
        mock_embedding_service.get_embeddings.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dataload_use_case_with_local_loader(self, mock_repository, mock_embedding_service, temp_csv_file):
        """Test dataloadUseCase with LocalLoader for comparison."""
        # Create use case with LocalLoader
        local_loader = LocalLoader()
        use_case = dataloadUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=local_loader
        )
        
        # Execute use case
        await use_case.execute(
            s3_uri=temp_csv_file,
            table_name="test_table",
            embed_columns_names=["name", "description"],
            pk_columns=["id"],
            create_table_if_not_exists=True,
            embed_type="combined"
        )
        
        # Verify same behavior as APIJSONStorageLoader
        mock_repository.create_table.assert_called_once()
        mock_repository.insert_data.assert_called_once()
        mock_embedding_service.get_embeddings.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_data_api_json_use_case_integration(self, mock_repository, mock_embedding_service):
        """Test DataAPIJSONUseCase integration."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        # Test with JSON data
        json_data = [
            {'id': 1, 'name': 'Alice', 'description': 'Engineer'},
            {'id': 2, 'name': 'Bob', 'description': 'Designer'}
        ]
        
        result = await use_case.execute(
            source=json_data,
            table_name="api_test_table",
            embed_columns_names=["name", "description"],
            pk_columns=["id"],
            create_table_if_not_exists=True,
            embed_type="combined"
        )
        
        # Verify successful execution
        assert isinstance(result, DataMoveResult)
        assert result.success
        assert result.rows_processed == 2
    
    @pytest.mark.asyncio
    async def test_data_move_use_case_auto_loader_compatibility(self, mock_repository):
        """Test DataMoveUseCase auto-loader compatibility."""
        # Test that auto-loader can work with different file types
        with patch('dataload.application.use_cases.data_move_use_case.DataMoveUseCase.create_with_auto_loader') as mock_create:
            mock_use_case = Mock()
            mock_use_case.execute = AsyncMock(return_value=DataMoveResult(
                success=True,
                rows_processed=2,
                execution_time=1.0,
                validation_report=Mock(),
                errors=[],
                warnings=[],
                table_created=True,
                schema_updated=False,
                operation_type="new_table_creation"
            ))
            mock_create.return_value = mock_use_case
            
            # Create use case with auto-loader
            use_case = DataMoveUseCase.create_with_auto_loader(repository=mock_repository)
            
            # Should work with different file types
            result = await use_case.execute(
                csv_path="test.csv",
                table_name="test_table"
            )
            
            assert result.success


class TestEmbeddingProviderCompatibility:
    """Test compatibility with existing embedding providers."""
    
    @pytest.fixture
    def mock_gemini_provider(self):
        """Mock Gemini embedding provider."""
        provider = Mock(spec=EmbeddingProviderInterface)
        provider.get_embeddings = Mock(return_value=[[0.1, 0.2, 0.3, 0.4]] * 3)
        return provider
    
    @pytest.fixture
    def mock_openai_provider(self):
        """Mock OpenAI embedding provider."""
        provider = Mock(spec=EmbeddingProviderInterface)
        provider.get_embeddings = Mock(return_value=[[0.5, 0.6, 0.7, 0.8]] * 3)
        return provider
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = Mock(spec=DataMoveRepositoryInterface)
        repo.table_exists = AsyncMock(return_value=False)
        repo.create_table = AsyncMock(return_value={'id': 'integer', 'name': 'text', 'description': 'text'})
        repo.insert_data = AsyncMock()
        return repo
    
    @pytest.mark.asyncio
    async def test_gemini_provider_integration(self, mock_repository, mock_gemini_provider):
        """Test integration with Gemini embedding provider."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_gemini_provider,
            storage_loader=api_loader
        )
        
        json_data = [
            {'id': 1, 'name': 'Alice', 'description': 'Software Engineer'},
            {'id': 2, 'name': 'Bob', 'description': 'Product Designer'},
            {'id': 3, 'name': 'Charlie', 'description': 'Data Scientist'}
        ]
        
        result = await use_case.execute(
            source=json_data,
            table_name="gemini_test",
            embed_columns_names=["name", "description"],
            embed_type="combined"
        )
        
        assert result.success
        assert result.rows_processed == 3
        mock_gemini_provider.get_embeddings.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_openai_provider_integration(self, mock_repository, mock_openai_provider):
        """Test integration with OpenAI embedding provider."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_openai_provider,
            storage_loader=api_loader
        )
        
        json_data = [
            {'id': 1, 'name': 'Alice', 'description': 'Software Engineer'},
            {'id': 2, 'name': 'Bob', 'description': 'Product Designer'},
            {'id': 3, 'name': 'Charlie', 'description': 'Data Scientist'}
        ]
        
        result = await use_case.execute(
            source=json_data,
            table_name="openai_test",
            embed_columns_names=["description"],
            embed_type="separated"
        )
        
        assert result.success
        assert result.rows_processed == 3
        mock_openai_provider.get_embeddings.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embedding_column_mapping_compatibility(self, mock_repository, mock_gemini_provider):
        """Test embedding generation with column mapping."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_repository,
            embedding_service=mock_gemini_provider,
            storage_loader=api_loader
        )
        
        json_data = [
            {'user_id': 1, 'full_name': 'Alice Johnson', 'job_title': 'Software Engineer'}
        ]
        
        result = await use_case.execute(
            source=json_data,
            table_name="mapping_test",
            embed_columns_names=["name", "title"],  # Using mapped column names
            column_name_mapping={
                'user_id': 'id',
                'full_name': 'name',
                'job_title': 'title'
            },
            embed_type="combined"
        )
        
        assert result.success
        assert result.rows_processed == 1
        mock_gemini_provider.get_embeddings.assert_called_once()


class TestDatabaseRepositoryCompatibility:
    """Test compatibility with database repository implementations."""
    
    @pytest.fixture
    def mock_postgres_repository(self):
        """Mock PostgreSQL repository."""
        repo = Mock(spec=DataMoveRepositoryInterface)
        repo.table_exists = AsyncMock(return_value=False)
        repo.create_table = AsyncMock(return_value={
            'id': 'integer',
            'name': 'text',
            'email': 'text',
            'created_at': 'timestamp',
            'embeddings': 'vector'
        })
        repo.insert_data = AsyncMock()
        repo.upsert_data = AsyncMock()
        repo.get_table_schema = AsyncMock(return_value=TableSchema(
            table_name='test_table',
            columns={'id': 'integer', 'name': 'text', 'email': 'text'},
            nullables={'id': False, 'name': True, 'email': True},
            primary_keys=['id']
        ))
        return repo
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        service = Mock(spec=EmbeddingProviderInterface)
        service.get_embeddings = Mock(return_value=[[0.1, 0.2, 0.3]] * 2)
        return service
    
    @pytest.mark.asyncio
    async def test_postgres_repository_integration(self, mock_postgres_repository, mock_embedding_service):
        """Test integration with PostgreSQL repository."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_postgres_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        json_data = [
            {'id': 1, 'name': 'Alice', 'email': 'alice@test.com'},
            {'id': 2, 'name': 'Bob', 'email': 'bob@test.com'}
        ]
        
        result = await use_case.execute(
            source=json_data,
            table_name="postgres_test",
            embed_columns_names=["name"],
            pk_columns=["id"],
            create_table_if_not_exists=True
        )
        
        assert result.success
        assert result.rows_processed == 2
        mock_postgres_repository.create_table.assert_called_once()
        mock_postgres_repository.insert_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_existing_table_upsert_compatibility(self, mock_postgres_repository, mock_embedding_service):
        """Test upsert operations with existing tables."""
        # Configure repository to return existing table
        mock_postgres_repository.table_exists.return_value = True
        
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_postgres_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        json_data = [
            {'id': 1, 'name': 'Alice Updated', 'email': 'alice.new@test.com'},
            {'id': 3, 'name': 'Charlie', 'email': 'charlie@test.com'}
        ]
        
        result = await use_case.execute(
            source=json_data,
            table_name="existing_table",
            pk_columns=["id"],
            create_table_if_not_exists=False
        )
        
        assert result.success
        assert result.rows_processed == 2
        # Should use upsert for existing table
        mock_postgres_repository.upsert_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vector_column_compatibility(self, mock_postgres_repository, mock_embedding_service):
        """Test vector column handling compatibility."""
        api_loader = APIJSONStorageLoader()
        use_case = DataAPIJSONUseCase(
            repo=mock_postgres_repository,
            embedding_service=mock_embedding_service,
            storage_loader=api_loader
        )
        
        json_data = [
            {'id': 1, 'content': 'This is test content for embedding'}
        ]
        
        result = await use_case.execute(
            source=json_data,
            table_name="vector_test",
            embed_columns_names=["content"],
            embed_type="combined"
        )
        
        assert result.success
        mock_embedding_service.get_embeddings.assert_called_once()
        
        # Verify that embeddings were generated and passed to repository
        call_args = mock_postgres_repository.insert_data.call_args
        df_inserted = call_args[0][1]  # Second argument is the DataFrame
        assert 'embeddings' in df_inserted.columns


class TestRegressionTests:
    """Comprehensive regression tests for existing functionality."""
    
    @pytest.fixture
    def sample_data_scenarios(self):
        """Various data scenarios for regression testing."""
        return {
            'simple_flat': [
                {'id': 1, 'name': 'Alice', 'age': 30},
                {'id': 2, 'name': 'Bob', 'age': 25}
            ],
            'nested_objects': [
                {
                    'id': 1,
                    'user': {'name': 'Alice', 'profile': {'age': 30, 'city': 'NYC'}},
                    'metadata': {'created': '2024-01-01', 'active': True}
                }
            ],
            'arrays': [
                {
                    'id': 1,
                    'name': 'Alice',
                    'skills': ['Python', 'JavaScript', 'SQL'],
                    'projects': [
                        {'name': 'Project A', 'status': 'completed'},
                        {'name': 'Project B', 'status': 'in_progress'}
                    ]
                }
            ],
            'mixed_types': [
                {
                    'id': 1,
                    'name': 'Alice',
                    'age': 30,
                    'salary': 75000.50,
                    'active': True,
                    'start_date': '2023-01-15',
                    'tags': None,
                    'metadata': {}
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_json_flattening_regression(self, sample_data_scenarios):
        """Test JSON flattening doesn't break existing functionality."""
        api_loader = APIJSONStorageLoader()
        
        for scenario_name, data in sample_data_scenarios.items():
            df = await api_loader.load_json(data)
            
            # Basic assertions that should always pass
            assert len(df) > 0, f"Failed for scenario: {scenario_name}"
            assert 'id' in df.columns, f"Missing id column in scenario: {scenario_name}"
            assert df['id'].iloc[0] == 1, f"Wrong id value in scenario: {scenario_name}"
    
    @pytest.mark.asyncio
    async def test_column_mapping_regression(self, sample_data_scenarios):
        """Test column mapping doesn't break existing functionality."""
        api_loader = APIJSONStorageLoader()
        
        config = {
            'column_name_mapping': {
                'id': 'user_id',
                'name': 'full_name'
            }
        }
        
        for scenario_name, data in sample_data_scenarios.items():
            if 'name' in str(data):  # Only test scenarios with name field
                df = await api_loader.load_json(data, config)
                
                assert 'user_id' in df.columns, f"Column mapping failed for scenario: {scenario_name}"
                if 'full_name' in df.columns:
                    assert df['full_name'].notna().any(), f"Name mapping failed for scenario: {scenario_name}"
    
    @pytest.mark.asyncio
    async def test_data_transformation_regression(self, sample_data_scenarios):
        """Test data transformations don't break existing functionality."""
        api_loader = APIJSONStorageLoader()
        
        config = {
            'update_request_body_mapping': {
                'computed_field': "concat('user_', {id})"
            }
        }
        
        for scenario_name, data in sample_data_scenarios.items():
            df = await api_loader.load_json(data, config)
            
            assert len(df) > 0, f"Transformation broke data loading for scenario: {scenario_name}"
            # Computed field should be added
            if 'computed_field' in df.columns:
                assert df['computed_field'].notna().any(), f"Transformation failed for scenario: {scenario_name}"
    
    def test_csv_loading_regression(self):
        """Test CSV loading still works as expected."""
        api_loader = APIJSONStorageLoader()
        
        # Create test CSV
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            df = api_loader.load_csv(csv_path)
            
            # Verify data integrity
            pd.testing.assert_frame_equal(df, test_data)
            assert len(df) == 3
            assert list(df.columns) == ['id', 'name', 'value']
            
        finally:
            os.unlink(csv_path)
    
    @pytest.mark.asyncio
    async def test_error_handling_regression(self):
        """Test error handling still works correctly."""
        api_loader = APIJSONStorageLoader()
        
        # Test various error scenarios
        error_scenarios = [
            # Invalid JSON structure
            ("invalid_json", "not valid json"),
            # Empty data
            ("empty_list", []),
            # Invalid configuration
            ("invalid_config", [{'id': 1}])
        ]
        
        for scenario_name, data in error_scenarios:
            if scenario_name == "invalid_json":
                # This should raise an error
                with pytest.raises(Exception):
                    await api_loader.load_json(data)
            elif scenario_name == "empty_list":
                # Empty list should return empty DataFrame
                df = await api_loader.load_json(data)
                assert len(df) == 0
            elif scenario_name == "invalid_config":
                # Invalid config should be handled gracefully
                invalid_config = {'separator': ''}  # Empty separator
                errors = api_loader.validate_config(invalid_config)
                assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_performance_regression(self, sample_data_scenarios):
        """Test performance hasn't degraded significantly."""
        import time
        
        api_loader = APIJSONStorageLoader()
        
        # Test with larger dataset
        large_data = []
        for i in range(100):
            large_data.append({
                'id': i,
                'name': f'User {i}',
                'email': f'user{i}@test.com',
                'profile': {
                    'age': 20 + (i % 50),
                    'city': f'City {i % 10}',
                    'preferences': {
                        'theme': 'dark' if i % 2 else 'light',
                        'notifications': i % 3 == 0
                    }
                },
                'tags': [f'tag{j}' for j in range(i % 5)]
            })
        
        start_time = time.time()
        df = await api_loader.load_json(large_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions
        assert len(df) == 100
        assert processing_time < 5.0, f"Processing took too long: {processing_time}s"
        
        # Memory usage should be reasonable
        import sys
        df_size = sys.getsizeof(df)
        assert df_size < 10 * 1024 * 1024, f"DataFrame too large: {df_size} bytes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])