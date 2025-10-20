"""
Unit tests for the ColumnMapper class.

This module contains comprehensive tests for column mapping functionality,
including validation, conflict detection, and error handling scenarios.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.dataload.domain.column_mapper import ColumnMapper
from src.dataload.domain.api_entities import (
    ColumnMappingConfig,
    MappingResult,
    ColumnMappingError,
    ValidationMode
)


class TestColumnMapper:
    """Test suite for ColumnMapper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = ColumnMapper()
        
        # Sample DataFrame for testing
        self.sample_df = pd.DataFrame({
            'first_name': ['John', 'Jane'],
            'last_name': ['Doe', 'Smith'],
            'email_address': ['john@example.com', 'jane@example.com'],
            'phone_number': ['123-456-7890', '098-765-4321'],
            'created_at': ['2023-01-01', '2023-01-02']
        })
    
    def test_basic_column_mapping(self):
        """Test basic column name mapping functionality."""
        mapping = {
            'first_name': 'fname',
            'last_name': 'lname',
            'email_address': 'email'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        result = self.mapper.apply_mapping(self.sample_df, config)
        
        assert result.success
        assert len(result.applied_mappings) == 3
        assert 'fname' in result.mapped_dataframe.columns
        assert 'lname' in result.mapped_dataframe.columns
        assert 'email' in result.mapped_dataframe.columns
        assert 'first_name' not in result.mapped_dataframe.columns
        assert result.applied_mappings['first_name'] == 'fname'
        assert len(result.unmapped_columns) == 2  # phone_number, created_at
    
    def test_case_insensitive_mapping(self):
        """Test case-insensitive column mapping."""
        mapping = {
            'FIRST_NAME': 'fname',
            'Last_Name': 'lname'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=False
        )
        
        result = self.mapper.apply_mapping(self.sample_df, config)
        
        assert result.success
        assert len(result.applied_mappings) == 2
        assert 'fname' in result.mapped_dataframe.columns
        assert 'lname' in result.mapped_dataframe.columns
        assert result.applied_mappings['first_name'] == 'fname'
        assert result.applied_mappings['last_name'] == 'lname'
    
    def test_missing_source_column_strict_mode(self):
        """Test handling of missing source columns in strict mode."""
        mapping = {
            'first_name': 'fname',
            'nonexistent_column': 'new_name'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True,
            allow_missing_columns=False
        )
        
        with pytest.raises(ColumnMappingError) as exc_info:
            self.mapper.apply_mapping(self.sample_df, config)
        
        assert "Source columns not found" in str(exc_info.value)
        assert "nonexistent_column" in str(exc_info.value)
    
    def test_missing_source_column_lenient_mode(self):
        """Test handling of missing source columns in lenient mode."""
        mapping = {
            'first_name': 'fname',
            'nonexistent_column': 'new_name'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.LENIENT,
            case_sensitive=True
        )
        
        result = self.mapper.apply_mapping(self.sample_df, config)
        
        assert len(result.warnings) > 0
        assert any("Source columns not found" in warning for warning in result.warnings)
        assert len(result.applied_mappings) == 1  # Only first_name should be mapped
        assert result.applied_mappings['first_name'] == 'fname'
    
    def test_target_column_conflicts(self):
        """Test detection of target column conflicts."""
        mapping = {
            'first_name': 'last_name'  # Conflicts with existing column
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        with pytest.raises(ColumnMappingError) as exc_info:
            self.mapper.apply_mapping(self.sample_df, config)
        
        assert "Target column conflicts" in str(exc_info.value)
    
    def test_duplicate_target_names(self):
        """Test detection of duplicate target column names."""
        mapping = {
            'first_name': 'name',
            'last_name': 'name'  # Duplicate target
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        with pytest.raises(ColumnMappingError) as exc_info:
            self.mapper.apply_mapping(self.sample_df, config)
        
        assert "Duplicate target column name" in str(exc_info.value)
    
    def test_reserved_keyword_validation(self):
        """Test validation against reserved SQL keywords."""
        mapping = {
            'first_name': 'select',  # Reserved keyword
            'last_name': 'from'      # Reserved keyword
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        with pytest.raises(ColumnMappingError) as exc_info:
            self.mapper.apply_mapping(self.sample_df, config)
        
        assert "reserved SQL keyword" in str(exc_info.value)
    
    def test_invalid_column_identifier(self):
        """Test validation of column identifier format."""
        mapping = {
            'first_name': 'invalid name',  # Contains space
            'last_name': '123invalid'      # Starts with number
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        with pytest.raises(ColumnMappingError) as exc_info:
            self.mapper.apply_mapping(self.sample_df, config)
        
        assert "not a valid column identifier" in str(exc_info.value)
    
    def test_column_name_length_validation(self):
        """Test validation of column name length."""
        long_name = 'a' * 64  # Exceeds PostgreSQL limit of 63
        mapping = {
            'first_name': long_name
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        with pytest.raises(ColumnMappingError) as exc_info:
            self.mapper.apply_mapping(self.sample_df, config)
        
        assert "exceeds maximum column name length" in str(exc_info.value)
    
    def test_preserve_unmapped_columns_true(self):
        """Test preserving unmapped columns."""
        mapping = {
            'first_name': 'fname'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True,
            preserve_unmapped_columns=True
        )
        
        result = self.mapper.apply_mapping(self.sample_df, config)
        
        assert result.success
        assert 'fname' in result.mapped_dataframe.columns
        assert 'last_name' in result.mapped_dataframe.columns  # Preserved
        assert 'email_address' in result.mapped_dataframe.columns  # Preserved
        assert len(result.mapped_dataframe.columns) == 5  # All original columns preserved
    
    def test_preserve_unmapped_columns_false(self):
        """Test dropping unmapped columns."""
        mapping = {
            'first_name': 'fname'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True,
            preserve_unmapped_columns=False
        )
        
        result = self.mapper.apply_mapping(self.sample_df, config)
        
        assert result.success
        assert 'fname' in result.mapped_dataframe.columns
        assert 'last_name' not in result.mapped_dataframe.columns  # Dropped
        assert 'email_address' not in result.mapped_dataframe.columns  # Dropped
        assert len(result.mapped_dataframe.columns) == 1  # Only mapped column
        assert len(result.warnings) > 0  # Should warn about dropped columns
    
    def test_empty_mapping(self):
        """Test handling of empty mapping configuration."""
        config = ColumnMappingConfig(
            column_name_mapping={},
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        result = self.mapper.apply_mapping(self.sample_df, config)
        
        assert result.success
        assert len(result.applied_mappings) == 0
        assert len(result.unmapped_columns) == len(self.sample_df.columns)
        assert result.mapped_dataframe.equals(self.sample_df)
    
    def test_self_mapping_validation(self):
        """Test validation against self-mappings."""
        mapping = {
            'first_name': 'first_name'  # Self-mapping
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        with pytest.raises(ColumnMappingError) as exc_info:
            self.mapper.apply_mapping(self.sample_df, config)
        
        assert "Self-mappings detected" in str(exc_info.value)
    
    def test_empty_source_or_target_validation(self):
        """Test validation of empty source or target names."""
        mapping = {
            '': 'fname',  # Empty source
            'last_name': ''  # Empty target
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        with pytest.raises(ColumnMappingError) as exc_info:
            self.mapper.apply_mapping(self.sample_df, config)
        
        error_message = str(exc_info.value)
        assert "cannot be empty" in error_message
    
    def test_transformation_stats(self):
        """Test transformation statistics generation."""
        mapping = {
            'first_name': 'fname',
            'last_name': 'lname'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        result = self.mapper.apply_mapping(self.sample_df, config)
        
        stats = result.transformation_stats
        assert stats['total_columns_input'] == 5
        assert stats['total_columns_output'] == 5
        assert stats['mappings_applied'] == 2
        assert stats['columns_unmapped'] == 3
        assert stats['conflicts_detected'] == 0
        assert stats['warnings_generated'] == 0
    
    def test_validate_embedding_columns_case_sensitive(self):
        """Test validation of embedding columns with case sensitivity."""
        mapped_columns = ['fname', 'lname', 'email', 'phone']
        embed_columns = ['fname', 'email', 'nonexistent']
        
        valid, missing = self.mapper.validate_embedding_columns(
            embed_columns, mapped_columns, case_sensitive=True
        )
        
        assert valid == ['fname', 'email']
        assert missing == ['nonexistent']
    
    def test_validate_embedding_columns_case_insensitive(self):
        """Test validation of embedding columns without case sensitivity."""
        mapped_columns = ['fname', 'lname', 'email', 'phone']
        embed_columns = ['FNAME', 'Email', 'nonexistent']
        
        valid, missing = self.mapper.validate_embedding_columns(
            embed_columns, mapped_columns, case_sensitive=False
        )
        
        assert valid == ['fname', 'email']
        assert missing == ['nonexistent']
    
    def test_suggest_mappings(self):
        """Test column mapping suggestions based on similarity."""
        source_columns = ['first_name', 'last_name', 'email_addr']
        target_columns = ['fname', 'lname', 'email_address', 'phone']
        
        suggestions = self.mapper.suggest_mappings(
            source_columns, target_columns, similarity_threshold=0.5
        )
        
        assert 'first_name' in suggestions
        assert 'fname' in suggestions['first_name']
        assert 'last_name' in suggestions
        assert 'lname' in suggestions['last_name']
        assert 'email_addr' in suggestions
        assert 'email_address' in suggestions['email_addr']
    
    def test_similarity_calculation(self):
        """Test string similarity calculation."""
        # Test exact match
        similarity = self.mapper._calculate_similarity('test', 'test')
        assert similarity == 1.0
        
        # Test completely different strings
        similarity = self.mapper._calculate_similarity('abc', 'xyz')
        assert similarity < 0.5
        
        # Test similar strings
        similarity = self.mapper._calculate_similarity('first_name', 'fname')
        assert 0.3 < similarity < 0.8
        
        # Test empty strings
        similarity = self.mapper._calculate_similarity('', '')
        assert similarity == 1.0
        
        similarity = self.mapper._calculate_similarity('test', '')
        assert similarity == 0.0
    
    def test_case_insensitive_conflict_detection(self):
        """Test case-insensitive conflict detection."""
        df = pd.DataFrame({
            'Name': ['John'],
            'EMAIL': ['john@example.com']
        })
        
        mapping = {
            'Name': 'email'  # Conflicts with EMAIL (case-insensitive)
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=False
        )
        
        with pytest.raises(ColumnMappingError) as exc_info:
            self.mapper.apply_mapping(df, config)
        
        assert "conflicts with existing unmapped column" in str(exc_info.value)
    
    def test_ignore_validation_mode(self):
        """Test ignore validation mode behavior."""
        mapping = {
            'first_name': 'select',  # Reserved keyword
            'nonexistent': 'new_col'  # Missing source
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.IGNORE,
            case_sensitive=True
        )
        
        result = self.mapper.apply_mapping(self.sample_df, config)
        
        # Should complete without raising exceptions
        assert len(result.applied_mappings) == 1  # Only valid mapping applied
        assert result.applied_mappings['first_name'] == 'select'
        assert len(result.warnings) > 0  # Should still generate warnings
    
    @patch('src.dataload.domain.column_mapper.logger')
    def test_logging_behavior(self, mock_logger):
        """Test that appropriate logging occurs during mapping operations."""
        mapping = {
            'first_name': 'fname'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        self.mapper.apply_mapping(self.sample_df, config)
        
        # Verify logging calls were made
        mock_logger.info.assert_called()
        mock_logger.debug.assert_called()
    
    def test_dataframe_not_modified_in_place(self):
        """Test that the original DataFrame is not modified."""
        original_columns = self.sample_df.columns.tolist()
        
        mapping = {
            'first_name': 'fname'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        result = self.mapper.apply_mapping(self.sample_df, config)
        
        # Original DataFrame should be unchanged
        assert self.sample_df.columns.tolist() == original_columns
        assert 'fname' not in self.sample_df.columns
        
        # Result DataFrame should have the mapping applied
        assert 'fname' in result.mapped_dataframe.columns
        assert 'first_name' not in result.mapped_dataframe.columns


class TestColumnMapperEdgeCases:
    """Test edge cases and error conditions for ColumnMapper."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = ColumnMapper()
    
    def test_empty_dataframe(self):
        """Test mapping with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        mapping = {
            'col1': 'new_col1'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.LENIENT,
            case_sensitive=True
        )
        
        result = self.mapper.apply_mapping(empty_df, config)
        
        assert len(result.applied_mappings) == 0
        assert len(result.warnings) > 0  # Should warn about missing columns
    
    def test_single_column_dataframe(self):
        """Test mapping with single column DataFrame."""
        single_col_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        mapping = {
            'col1': 'new_col1'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        result = self.mapper.apply_mapping(single_col_df, config)
        
        assert result.success
        assert len(result.applied_mappings) == 1
        assert 'new_col1' in result.mapped_dataframe.columns
        assert 'col1' not in result.mapped_dataframe.columns
    
    def test_unicode_column_names(self):
        """Test mapping with Unicode column names."""
        unicode_df = pd.DataFrame({
            '名前': ['田中', '佐藤'],
            'メール': ['tanaka@example.com', 'sato@example.com']
        })
        
        mapping = {
            '名前': 'name',
            'メール': 'email'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        result = self.mapper.apply_mapping(unicode_df, config)
        
        assert result.success
        assert len(result.applied_mappings) == 2
        assert 'name' in result.mapped_dataframe.columns
        assert 'email' in result.mapped_dataframe.columns
    
    def test_special_characters_in_column_names(self):
        """Test handling of special characters in column names."""
        special_df = pd.DataFrame({
            'col-with-dashes': [1, 2],
            'col.with.dots': [3, 4],
            'col with spaces': [5, 6]
        })
        
        mapping = {
            'col-with-dashes': 'col_with_underscores',
            'col.with.dots': 'col_with_underscores2'
        }
        
        config = ColumnMappingConfig(
            column_name_mapping=mapping,
            update_request_body_mapping={},
            validation_mode=ValidationMode.STRICT,
            case_sensitive=True
        )
        
        result = self.mapper.apply_mapping(special_df, config)
        
        assert result.success
        assert len(result.applied_mappings) == 2
        assert 'col_with_underscores' in result.mapped_dataframe.columns
        assert 'col_with_underscores2' in result.mapped_dataframe.columns


if __name__ == '__main__':
    pytest.main([__file__])