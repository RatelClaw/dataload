"""
Unit tests for APIJSONStorageLoader.

This module contains unit tests for individual methods and components
of the APIJSONStorageLoader class.
"""

import pytest
import pandas as pd
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

from src.dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from src.dataload.domain.api_entities import APIResponse


class TestAPIJSONStorageLoaderUnit:
    """Unit tests for APIJSONStorageLoader methods."""
    
    @pytest.fixture
    def loader(self):
        """Create APIJSONStorageLoader instance for testing."""
        return APIJSONStorageLoader(
            base_url="https://api.example.com",
            api_token="test-token"
        )
    
    def test_initialization(self):
        """Test loader initialization with various parameters."""
        # Test with minimal parameters
        loader = APIJSONStorageLoader()
        assert loader.base_url is None
        assert loader.timeout == 30
        assert loader.retry_attempts == 3
        
        # Test with full parameters
        loader = APIJSONStorageLoader(
            base_url="https://api.test.com",
            api_token="token123",
            jwt_token="jwt123",
            timeout=60,
            retry_attempts=5,
            verify_ssl=False,
            default_headers={"User-Agent": "test"}
        )
        assert loader.base_url == "https://api.test.com"
        assert loader.timeout == 60
        assert loader.retry_attempts == 5
        assert loader.verify_ssl is False
        assert loader.default_headers == {"User-Agent": "test"}
    
    def test_process_config_valid(self, loader):
        """Test configuration processing with valid parameters."""
        config = {
            'flatten_nested': True,
            'separator': '__',
            'max_depth': 5,
            'column_name_mapping': {'old': 'new'},
            'page_size': 50
        }
        
        processed = loader._process_config(config)
        
        assert processed['flatten_nested'] is True
        assert processed['separator'] == '__'
        assert processed['max_depth'] == 5
        assert processed['column_name_mapping'] == {'old': 'new'}
        assert processed['page_size'] == 50
    
    def test_process_config_defaults(self, loader):
        """Test configuration processing with default values."""
        config = {}
        processed = loader._process_config(config)
        
        assert processed['flatten_nested'] is True
        assert processed['separator'] == '_'
        assert processed['max_depth'] is None
        assert processed['column_name_mapping'] == {}
        assert processed['page_size'] == 100
    
    def test_process_config_invalid(self, loader):
        """Test configuration processing with invalid parameters."""
        # Empty separator
        with pytest.raises(ValueError, match="Separator cannot be empty"):
            loader._process_config({'separator': ''})
        
        # Invalid max_depth
        with pytest.raises(ValueError, match="Max depth must be greater than 0"):
            loader._process_config({'max_depth': 0})
        
        # Invalid page_size
        with pytest.raises(ValueError, match="Page size must be greater than 0"):
            loader._process_config({'page_size': -1})
    
    def test_load_from_file_success(self, loader):
        """Test successful JSON file loading."""
        test_data = {"test": "value", "number": 42}
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            json_path = f.name
        
        try:
            result = loader._load_from_file(json_path)
            assert result == test_data
        finally:
            os.unlink(json_path)
    
    def test_load_from_file_not_found(self, loader):
        """Test JSON file loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            loader._load_from_file("non_existent.json")
    
    def test_load_from_file_invalid_json(self, loader):
        """Test JSON file loading with invalid JSON."""
        # Create file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content }")
            json_path = f.name
        
        try:
            with pytest.raises(Exception):  # Should raise JSONParsingError
                loader._load_from_file(json_path)
        finally:
            os.unlink(json_path)
    
    @patch('src.dataload.infrastructure.storage.api_json_loader.APIHandler')
    async def test_load_from_api_single_response(self, mock_handler_class, loader):
        """Test API loading with single response."""
        test_data = [{"id": 1, "name": "test"}]
        
        # Mock API response
        mock_response = APIResponse(
            data=test_data,
            status_code=200,
            headers={},
            response_time=0.1,
            url="https://api.example.com/data",
            method="GET"
        )
        
        # Mock handler
        mock_handler = AsyncMock()
        mock_handler.fetch_data.return_value = mock_response
        mock_handler.__aenter__.return_value = mock_handler
        
        # Replace handler in loader
        loader.api_handler = mock_handler
        
        config = {'method': 'GET', 'headers': {}, 'params': {}}
        result = await loader._load_from_api("https://api.example.com/data", config)
        
        assert result == test_data
        mock_handler.fetch_data.assert_called_once()
    
    @patch('src.dataload.infrastructure.storage.api_json_loader.APIHandler')
    async def test_load_from_api_paginated_response(self, mock_handler_class, loader):
        """Test API loading with paginated responses."""
        # Mock multiple page responses
        page1_data = [{"id": 1, "name": "item1"}]
        page2_data = [{"id": 2, "name": "item2"}]
        
        mock_response1 = APIResponse(
            data=page1_data, status_code=200, headers={}, response_time=0.1,
            url="https://api.example.com/data?page=1", method="GET"
        )
        mock_response2 = APIResponse(
            data=page2_data, status_code=200, headers={}, response_time=0.1,
            url="https://api.example.com/data?page=2", method="GET"
        )
        
        # Mock handler to return list of responses (paginated)
        mock_handler = AsyncMock()
        mock_handler.fetch_data.return_value = [mock_response1, mock_response2]
        mock_handler.__aenter__.return_value = mock_handler
        
        loader.api_handler = mock_handler
        
        config = {'pagination_enabled': True, 'page_size': 1}
        result = await loader._load_from_api("https://api.example.com/data", config)
        
        # Should combine data from both pages
        expected = page1_data + page2_data
        assert result == expected
    
    def test_apply_transformations(self, loader):
        """Test request body transformations."""
        df = pd.DataFrame([
            {"first_name": "John", "last_name": "Doe", "salary": 50000},
            {"first_name": "Jane", "last_name": "Smith", "salary": 60000}
        ])
        
        config = {
            'update_request_body_mapping': {
                'full_name': "concat({first_name}, ' ', {last_name})",
                'salary_k': "{salary} / 1000"
            },
            'fail_on_error': True,
            'preserve_original_data': True
        }
        
        result = loader._apply_transformations(df, config)
        
        assert result.success
        assert 'full_name' in result.transformed_dataframe.columns
        assert 'salary_k' in result.transformed_dataframe.columns
        assert len(result.applied_transformations) == 2
    
    def test_apply_column_mapping(self, loader):
        """Test column name mapping."""
        df = pd.DataFrame([
            {"old_name1": "value1", "old_name2": "value2"},
            {"old_name1": "value3", "old_name2": "value4"}
        ])
        
        config = {
            'column_name_mapping': {
                'old_name1': 'new_name1',
                'old_name2': 'new_name2'
            },
            'case_sensitive': True,
            'validation_mode': 'strict',
            'fail_on_error': True
        }
        
        result = loader._apply_column_mapping(df, config)
        
        assert result.success
        assert 'new_name1' in result.mapped_dataframe.columns
        assert 'new_name2' in result.mapped_dataframe.columns
        assert 'old_name1' not in result.mapped_dataframe.columns
        assert 'old_name2' not in result.mapped_dataframe.columns
        assert len(result.applied_mappings) == 2
    
    def test_validate_config_valid(self, loader):
        """Test configuration validation with valid config."""
        config = {
            'column_name_mapping': {'old': 'new'},
            'update_request_body_mapping': {'field': 'expression'},
            'pagination_enabled': True,
            'page_size': 100,
            'max_pages': 10
        }
        
        errors = loader.validate_config(config)
        assert len(errors) == 0
    
    def test_validate_config_invalid(self, loader):
        """Test configuration validation with invalid config."""
        config = {
            'column_name_mapping': 'not_a_dict',  # Should be dict
            'update_request_body_mapping': {'': 'expression'},  # Empty key
            'pagination_enabled': True,
            'page_size': 0,  # Invalid page size
            'max_pages': -1  # Invalid max pages
        }
        
        errors = loader.validate_config(config)
        assert len(errors) > 0
        
        # Check specific error types
        error_text = ' '.join(errors).lower()
        assert 'dictionary' in error_text
        assert 'page_size' in error_text or 'positive' in error_text
    
    @patch('src.dataload.infrastructure.storage.api_json_loader.JSONFlattener')
    def test_process_json_pipeline_success(self, mock_flattener_class, loader):
        """Test successful JSON processing pipeline."""
        # Mock flattening result
        mock_df = pd.DataFrame([{"id": 1, "name": "test"}])
        mock_flattening_result = Mock()
        mock_flattening_result.success = True
        mock_flattening_result.dataframe = mock_df
        mock_flattening_result.warnings = []
        
        # Mock flattener
        mock_flattener = Mock()
        mock_flattener.flatten_json.return_value = mock_flattening_result
        loader.json_flattener = mock_flattener
        
        json_data = [{"id": 1, "name": "test"}]
        config = {'flatten_nested': True}
        
        result = loader._process_json_pipeline(json_data, config)
        
        assert result.success
        assert result.dataframe is not None
        assert len(result.dataframe) == 1
        assert result.rows_loaded == 1
        assert result.columns_created == 2
    
    @patch('src.dataload.infrastructure.storage.api_json_loader.JSONFlattener')
    def test_process_json_pipeline_failure(self, mock_flattener_class, loader):
        """Test JSON processing pipeline with failure."""
        # Mock flattener to raise exception
        mock_flattener = Mock()
        mock_flattener.flatten_json.side_effect = Exception("Flattening failed")
        loader.json_flattener = mock_flattener
        
        json_data = [{"id": 1, "name": "test"}]
        config = {'flatten_nested': True}
        
        result = loader._process_json_pipeline(json_data, config)
        
        assert not result.success
        assert result.dataframe is None
        assert len(result.errors) > 0
        assert result.rows_loaded == 0
    
    def test_context_manager_sync(self, loader):
        """Test synchronous context manager."""
        with loader as ctx_loader:
            assert ctx_loader is loader
    
    async def test_context_manager_async(self, loader):
        """Test asynchronous context manager."""
        async with loader as ctx_loader:
            assert ctx_loader is loader
    
    def test_load_json_invalid_source_type(self, loader):
        """Test load_json with invalid source type."""
        with pytest.raises(ValueError, match="Unsupported source type"):
            loader.load_json(123)  # Integer not supported
        
        with pytest.raises(ValueError, match="Unsupported source type"):
            loader.load_json(['not', 'dict', 'list'])  # List of strings not supported