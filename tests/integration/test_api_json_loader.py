"""
Integration tests for APIJSONStorageLoader.

This module contains comprehensive integration tests for the APIJSONStorageLoader
class, testing complete loading workflows with various data sources and configurations.
"""

import pytest
import pandas as pd
import json
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from src.dataload.domain.api_entities import APIResponse, APIError, JSONParsingError


class TestAPIJSONStorageLoaderIntegration:
    """Integration tests for APIJSONStorageLoader complete workflows."""
    
    @pytest.fixture
    def sample_json_data(self):
        """Sample JSON data for testing."""
        return [
            {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "profile": {
                    "age": 30,
                    "city": "New York",
                    "preferences": {
                        "theme": "dark",
                        "notifications": True
                    }
                },
                "tags": ["developer", "python"]
            },
            {
                "id": 2,
                "name": "Jane Smith",
                "email": "jane@example.com",
                "profile": {
                    "age": 25,
                    "city": "San Francisco",
                    "preferences": {
                        "theme": "light",
                        "notifications": False
                    }
                },
                "tags": ["designer", "ui/ux"]
            }
        ]
    
    @pytest.fixture
    def loader(self):
        """Create APIJSONStorageLoader instance for testing."""
        return APIJSONStorageLoader(
            base_url="https://api.example.com",
            api_token="test-token",
            timeout=30,
            retry_attempts=2
        )
    
    def test_load_csv_basic(self, loader):
        """Test basic CSV loading functionality."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,email\n")
            f.write("1,John Doe,john@example.com\n")
            f.write("2,Jane Smith,jane@example.com\n")
            csv_path = f.name
        
        try:
            # Load CSV
            df = loader.load_csv(csv_path)
            
            # Verify results
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ['id', 'name', 'email']
            assert df.iloc[0]['name'] == 'John Doe'
            assert df.iloc[1]['name'] == 'Jane Smith'
            
        finally:
            os.unlink(csv_path)
    
    def test_load_csv_file_not_found(self, loader):
        """Test CSV loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            loader.load_csv("non_existent_file.csv")
    
    def test_load_json_from_dict(self, loader, sample_json_data):
        """Test loading JSON from dictionary object."""
        df = loader.load_json(sample_json_data)
        
        # Verify basic structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        
        # Check that nested data was flattened
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert 'email' in df.columns
        assert 'profile_age' in df.columns
        assert 'profile_city' in df.columns
        assert 'profile_preferences_theme' in df.columns
        
        # Verify data values
        assert df.iloc[0]['name'] == 'John Doe'
        assert df.iloc[0]['profile_age'] == 30
        assert df.iloc[1]['profile_preferences_theme'] == 'light'   
 
    def test_load_json_from_file(self, loader, sample_json_data):
        """Test loading JSON from local file."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_data, f)
            json_path = f.name
        
        try:
            df = loader.load_json(json_path)
            
            # Verify results
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert 'id' in df.columns
            assert 'name' in df.columns
            assert df.iloc[0]['name'] == 'John Doe'
            
        finally:
            os.unlink(json_path)
    
    def test_load_json_file_not_found(self, loader):
        """Test JSON loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            loader.load_json("non_existent_file.json")
    
    def test_load_json_invalid_file(self, loader):
        """Test JSON loading with invalid JSON file."""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            json_path = f.name
        
        try:
            with pytest.raises(JSONParsingError):
                loader.load_json(json_path)
        finally:
            os.unlink(json_path)
    
    @patch('src.dataload.infrastructure.storage.api_json_loader.APIHandler')
    async def test_load_json_from_api_success(self, mock_handler_class, loader, sample_json_data):
        """Test loading JSON from API endpoint."""
        # Mock API response
        mock_response = APIResponse(
            data=sample_json_data,
            status_code=200,
            headers={'content-type': 'application/json'},
            response_time=0.5,
            url="https://api.example.com/users",
            method="GET"
        )
        
        # Mock handler instance
        mock_handler = AsyncMock()
        mock_handler.fetch_data.return_value = mock_response
        mock_handler.__aenter__.return_value = mock_handler
        mock_handler.__aexit__.return_value = None
        
        # Replace the handler in the loader
        loader.api_handler = mock_handler
        
        # Test API loading
        df = loader.load_json("https://api.example.com/users")
        
        # Verify results
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'id' in df.columns
        assert 'name' in df.columns
        
        # Verify API handler was called correctly
        mock_handler.fetch_data.assert_called_once()
    
    def test_load_json_with_column_mapping(self, loader, sample_json_data):
        """Test JSON loading with column name mapping."""
        config = {
            'column_name_mapping': {
                'name': 'full_name',
                'email': 'email_address',
                'profile_age': 'age'
            }
        }
        
        df = loader.load_json(sample_json_data, config)
        
        # Verify column mapping was applied
        assert 'full_name' in df.columns
        assert 'email_address' in df.columns
        assert 'age' in df.columns
        assert 'name' not in df.columns  # Original column should be renamed
        assert 'email' not in df.columns
        assert 'profile_age' not in df.columns
        
        # Verify data integrity
        assert df.iloc[0]['full_name'] == 'John Doe'
        assert df.iloc[0]['email_address'] == 'john@example.com'
    
    def test_load_json_with_transformations(self, loader):
        """Test JSON loading with request body transformations."""
        simple_data = [
            {"first_name": "John", "last_name": "Doe", "salary": 50000},
            {"first_name": "Jane", "last_name": "Smith", "salary": 60000}
        ]
        
        config = {
            'update_request_body_mapping': {
                'full_name': "concat({first_name}, ' ', {last_name})",
                'salary_k': "round({salary} / 1000) * 1000"
            }
        }
        
        df = loader.load_json(simple_data, config)
        
        # Verify transformations were applied
        assert 'full_name' in df.columns
        assert 'salary_k' in df.columns
        
        # Verify transformation results
        assert df.iloc[0]['full_name'] == 'John Doe'
        assert df.iloc[1]['full_name'] == 'Jane Smith'
        assert df.iloc[0]['salary_k'] == 50000  # Should be rounded to nearest thousand
        assert df.iloc[1]['salary_k'] == 60000
    
    def test_load_json_with_custom_flattening(self, loader, sample_json_data):
        """Test JSON loading with custom flattening configuration."""
        config = {
            'separator': '__',
            'max_depth': 2,
            'handle_arrays': 'join'
        }
        
        df = loader.load_json(sample_json_data, config)
        
        # Verify custom separator was used
        assert 'profile__age' in df.columns
        assert 'profile__city' in df.columns
        
        # Verify arrays were joined instead of expanded
        assert 'tags' in df.columns
        assert isinstance(df.iloc[0]['tags'], str)  # Should be joined string
        assert 'developer, python' in df.iloc[0]['tags']
    
    def test_load_json_complex_workflow(self, loader):
        """Test complete workflow with flattening, transformations, and mapping."""
        complex_data = [
            {
                "user_id": 1,
                "personal_info": {
                    "first_name": "John",
                    "last_name": "Doe",
                    "contact": {
                        "email": "john@example.com",
                        "phone": "555-1234"
                    }
                },
                "employment": {
                    "position": "Developer",
                    "salary": 75000,
                    "start_date": "2020-01-15"
                }
            }
        ]
        
        config = {
            # First flatten with custom separator
            'separator': '_',
            'flatten_nested': True,
            
            # Then apply transformations
            'update_request_body_mapping': {
                'full_name': "concat({personal_info_first_name}, ' ', {personal_info_last_name})",
                'annual_salary_k': "round({employment_salary} / 1000)"
            },
            
            # Finally apply column mappings
            'column_name_mapping': {
                'user_id': 'id',
                'personal_info_contact_email': 'email',
                'employment_position': 'job_title',
                'full_name': 'name'
            }
        }
        
        df = loader.load_json(complex_data, config)
        
        # Verify complete pipeline worked
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert 'email' in df.columns
        assert 'job_title' in df.columns
        assert 'annual_salary_k' in df.columns
        
        # Verify data transformations
        assert df.iloc[0]['id'] == 1
        assert df.iloc[0]['name'] == 'John Doe'
        assert df.iloc[0]['email'] == 'john@example.com'
        assert df.iloc[0]['job_title'] == 'Developer'
        assert df.iloc[0]['annual_salary_k'] == 75  # 75000 / 1000 rounded
    
    def test_config_validation(self, loader):
        """Test configuration validation."""
        # Test valid config
        valid_config = {
            'column_name_mapping': {'old_name': 'new_name'},
            'separator': '_',
            'page_size': 100
        }
        errors = loader.validate_config(valid_config)
        assert len(errors) == 0
        
        # Test invalid config
        invalid_config = {
            'separator': '',  # Empty separator
            'page_size': -1,  # Invalid page size
            'column_name_mapping': 'not_a_dict'  # Wrong type
        }
        errors = loader.validate_config(invalid_config)
        assert len(errors) > 0
        assert any('separator' in error.lower() for error in errors)
        assert any('page_size' in error.lower() for error in errors)
        assert any('dictionary' in error.lower() for error in errors)
    
    def test_unsupported_source_type(self, loader):
        """Test loading with unsupported source type."""
        with pytest.raises(ValueError, match="Unsupported source type"):
            loader.load_json(12345)  # Integer is not supported
    
    def test_empty_json_data(self, loader):
        """Test loading empty JSON data."""
        # Empty list
        df = loader.load_json([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        
        # Empty dict
        df = loader.load_json({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_single_record_json(self, loader):
        """Test loading single JSON record."""
        single_record = {
            "id": 1,
            "name": "John Doe",
            "nested": {"value": 42}
        }
        
        df = loader.load_json(single_record)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert 'nested_value' in df.columns
        assert df.iloc[0]['nested_value'] == 42
    
    async def test_context_manager_usage(self, loader):
        """Test using loader as async context manager."""
        async with loader as ctx_loader:
            assert ctx_loader is loader
            
            # Test that we can use the loader normally
            simple_data = {"test": "value"}
            df = ctx_loader.load_json(simple_data)
            assert isinstance(df, pd.DataFrame)
    
    def test_error_handling_in_pipeline(self, loader):
        """Test error handling during processing pipeline."""
        # Test with malformed transformation expression
        config = {
            'update_request_body_mapping': {
                'invalid_field': 'invalid_expression_syntax('
            },
            'fail_on_error': True
        }
        
        simple_data = [{"test": "value"}]
        
        # Should raise error in strict mode
        with pytest.raises(Exception):  # Could be various transformation errors
            loader.load_json(simple_data, config)
        
        # Should not raise error in lenient mode
        config['fail_on_error'] = False
        df = loader.load_json(simple_data, config)
        assert isinstance(df, pd.DataFrame)  # Should still return a DataFrame