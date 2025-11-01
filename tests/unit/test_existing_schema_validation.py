"""
Unit tests for existing_schema validation logic.

These tests verify the strict validation requirements for existing_schema mode,
including column name matching, type compatibility, nullable constraints,
and comprehensive error reporting.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock
from dataload.application.services.validation import ValidationService
from dataload.domain.entities import (
    TableInfo, ColumnInfo, ValidationReport, SchemaAnalysis,
    CaseConflict, TypeMismatch, ConstraintViolation, ValidationError
)


class TestExistingSchemaValidation:
    """Test cases for existing_schema validation logic."""
    
    @pytest.fixture
    def validation_service(self):
        """Create a ValidationService instance."""
        return ValidationService()
    
    @pytest.fixture
    def sample_table_info(self):
        """Create sample table info for testing."""
        return TableInfo(
            name="test_table",
            columns={
                "id": ColumnInfo(name="id", data_type="integer", nullable=False),
                "name": ColumnInfo(name="name", data_type="text", nullable=True),
                "email": ColumnInfo(name="email", data_type="text", nullable=False),
                "age": ColumnInfo(name="age", data_type="integer", nullable=True),
                "vector_col": ColumnInfo(name="vector_col", data_type="vector(3)", nullable=True)
            },
            primary_keys=["id"],
            constraints=[],
            indexes=[]
        )
    
    @pytest.fixture
    def matching_dataframe(self):
        """Create a DataFrame that exactly matches the table schema."""
        return pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
            "age": [25, 30, 35],
            "vector_col": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        })
    
    @pytest.mark.asyncio
    async def test_existing_schema_validation_success(
        self, validation_service, sample_table_info, matching_dataframe
    ):
        """Test successful existing_schema validation with matching schema."""
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=matching_dataframe,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is True
        assert len(result.errors) == 0
        assert result.schema_analysis.compatible is True
        assert len(result.schema_analysis.columns_added) == 0
        assert len(result.schema_analysis.columns_removed) == 0
        assert "Schema validation passed" in " ".join(result.recommendations)
    
    @pytest.mark.asyncio
    async def test_existing_schema_missing_columns_error(
        self, validation_service, sample_table_info
    ):
        """Test existing_schema validation fails when CSV is missing required columns."""
        # DataFrame missing 'email' and 'age' columns
        incomplete_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "vector_col": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        })
        
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=incomplete_df,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is False
        assert len(result.errors) > 0
        
        # Check for specific missing column errors
        missing_column_errors = [
            error for error in result.errors 
            if "Missing columns" in error
        ]
        assert len(missing_column_errors) > 0
        assert "email" in missing_column_errors[0]
        assert "age" in missing_column_errors[0]
        
        # Check schema analysis
        assert result.schema_analysis.compatible is False
        assert "email" in result.schema_analysis.columns_removed
        assert "age" in result.schema_analysis.columns_removed
    
    @pytest.mark.asyncio
    async def test_existing_schema_extra_columns_error(
        self, validation_service, sample_table_info
    ):
        """Test existing_schema validation fails when CSV has extra columns."""
        # DataFrame with extra columns
        extra_columns_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
            "age": [25, 30, 35],
            "vector_col": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            "extra_col1": ["extra1", "extra2", "extra3"],
            "extra_col2": [100, 200, 300]
        })
        
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=extra_columns_df,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is False
        assert len(result.errors) > 0
        
        # Check for specific extra column errors
        extra_column_errors = [
            error for error in result.errors 
            if "Extra columns" in error
        ]
        assert len(extra_column_errors) > 0
        assert "extra_col1" in extra_column_errors[0]
        assert "extra_col2" in extra_column_errors[0]
        
        # Check schema analysis
        assert result.schema_analysis.compatible is False
        assert "extra_col1" in result.schema_analysis.columns_added
        assert "extra_col2" in result.schema_analysis.columns_added
    
    @pytest.mark.asyncio
    async def test_existing_schema_case_sensitivity_error(
        self, validation_service, sample_table_info
    ):
        """Test existing_schema validation fails with case-sensitive column name mismatches."""
        # DataFrame with case mismatches
        case_mismatch_df = pd.DataFrame({
            "ID": [1, 2, 3],  # Should be 'id'
            "Name": ["Alice", "Bob", "Charlie"],  # Should be 'name'
            "EMAIL": ["alice@test.com", "bob@test.com", "charlie@test.com"],  # Should be 'email'
            "age": [25, 30, 35],
            "Vector_Col": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]  # Should be 'vector_col'
        })
        
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=case_mismatch_df,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is False
        assert len(result.errors) > 0
        
        # Should have both missing and extra column errors due to case mismatches
        missing_errors = [error for error in result.errors if "Missing columns" in error]
        extra_errors = [error for error in result.errors if "Extra columns" in error]
        
        assert len(missing_errors) > 0
        assert len(extra_errors) > 0
        
        # Check that case conflicts are detected
        case_conflict_errors = [error for error in result.errors if "Case mismatch" in error]
        assert len(case_conflict_errors) > 0
    
    @pytest.mark.asyncio
    async def test_existing_schema_nullable_constraint_violation(
        self, validation_service, sample_table_info
    ):
        """Test existing_schema validation fails when non-nullable columns have null values."""
        # DataFrame with null values in non-nullable columns
        null_violation_df = pd.DataFrame({
            "id": [1, None, 3],  # id is NOT NULL but has null value
            "name": ["Alice", "Bob", None],  # name is nullable, so this is OK
            "email": ["alice@test.com", None, "charlie@test.com"],  # email is NOT NULL but has null
            "age": [25, 30, None],  # age is nullable, so this is OK
            "vector_col": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        })
        
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=null_violation_df,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is False
        assert len(result.errors) > 0
        
        # Check for specific null constraint violation errors
        null_errors = [
            error for error in result.errors 
            if "NOT NULL" in error and "null values" in error
        ]
        assert len(null_errors) >= 2  # Should have errors for 'id' and 'email'
        
        # Verify specific columns are mentioned
        null_error_text = " ".join(null_errors)
        assert "id" in null_error_text
        assert "email" in null_error_text
    
    @pytest.mark.asyncio
    async def test_existing_schema_type_mismatch_error(
        self, validation_service, sample_table_info
    ):
        """Test existing_schema validation fails with incompatible data types."""
        # DataFrame with incompatible types
        type_mismatch_df = pd.DataFrame({
            "id": ["not_an_integer", "also_not_int", "still_not_int"],  # Should be integer
            "name": ["Alice", "Bob", "Charlie"],
            "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
            "age": ["twenty_five", "thirty", "thirty_five"],  # Should be integer
            "vector_col": ["not_a_vector", "also_not_vector", "still_not_vector"]  # Should be vector
        })
        
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=type_mismatch_df,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is False
        assert len(result.errors) > 0
        
        # Check for type mismatch errors
        type_errors = [
            error for error in result.errors 
            if "Type mismatch" in error or "mismatch" in error.lower()
        ]
        assert len(type_errors) > 0
        
        # Verify specific columns with type issues are mentioned
        type_error_text = " ".join(type_errors)
        assert "id" in type_error_text or any("id" in error for error in result.errors)
        assert "age" in type_error_text or any("age" in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_existing_schema_comprehensive_error_collection(
        self, validation_service, sample_table_info
    ):
        """Test that existing_schema validation collects ALL validation errors comprehensively."""
        # DataFrame with multiple types of errors
        problematic_df = pd.DataFrame({
            "ID": [None, "not_int", 3],  # Case mismatch + null in NOT NULL + type mismatch
            "name": ["Alice", "Bob", "Charlie"],
            "EMAIL": ["alice@test.com", None, "charlie@test.com"],  # Case mismatch + null in NOT NULL
            "extra_column": ["extra1", "extra2", "extra3"],  # Extra column
            # Missing: age, vector_col
        })
        
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=problematic_df,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is False
        assert len(result.errors) > 0
        
        # Should collect multiple types of errors
        error_text = " ".join(result.errors)
        
        # Check for missing columns
        assert any("Missing columns" in error for error in result.errors)
        
        # Check for extra columns
        assert any("Extra columns" in error for error in result.errors)
        
        # Check for case mismatches or null violations
        assert (
            any("Case mismatch" in error for error in result.errors) or
            any("NOT NULL" in error for error in result.errors)
        )
        
        # Verify recommendations are provided
        assert len(result.recommendations) > 0
        assert any("existing_schema" in rec for rec in result.recommendations)
    
    @pytest.mark.asyncio
    async def test_existing_schema_actionable_error_messages(
        self, validation_service, sample_table_info
    ):
        """Test that existing_schema validation provides actionable error messages."""
        # DataFrame with various issues
        problematic_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            # Missing: email, age, vector_col
            "extra_col": ["extra1", "extra2", "extra3"]
        })
        
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=problematic_df,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is False
        
        # Check that error messages are actionable and specific
        for error in result.errors:
            # Error messages should be specific and mention column names
            assert len(error) > 10  # Not just generic messages
            
            if "Missing columns" in error:
                # Should list specific missing columns
                assert "email" in error or "age" in error or "vector_col" in error
            
            if "Extra columns" in error:
                # Should list specific extra columns
                assert "extra_col" in error
        
        # Check that recommendations are actionable
        recommendations_text = " ".join(result.recommendations)
        assert (
            "match exactly" in recommendations_text or
            "case-sensitive" in recommendations_text or
            "new_schema" in recommendations_text
        )
    
    @pytest.mark.asyncio
    async def test_existing_schema_edge_case_empty_dataframe(
        self, validation_service, sample_table_info
    ):
        """Test existing_schema validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=empty_df,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is False
        assert len(result.errors) > 0
        
        # Should report missing columns
        missing_error = next(
            (error for error in result.errors if "Missing columns" in error), 
            None
        )
        assert missing_error is not None
    
    @pytest.mark.asyncio
    async def test_existing_schema_edge_case_single_column(
        self, validation_service
    ):
        """Test existing_schema validation with single column table and DataFrame."""
        single_col_table = TableInfo(
            name="single_col_table",
            columns={
                "id": ColumnInfo(name="id", data_type="integer", nullable=False)
            },
            primary_keys=["id"],
            constraints=[],
            indexes=[]
        )
        
        single_col_df = pd.DataFrame({"id": [1, 2, 3]})
        
        result = await validation_service.validate_data_move(
            table_info=single_col_table,
            df=single_col_df,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is True
        assert len(result.errors) == 0
        assert result.schema_analysis.compatible is True
    
    @pytest.mark.asyncio
    async def test_existing_schema_parameter_validation(
        self, validation_service, sample_table_info, matching_dataframe
    ):
        """Test parameter validation for existing_schema mode."""
        # Test missing move_type
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=matching_dataframe,
            move_type=None  # Should require move_type for existing tables
        )
        
        assert result.validation_passed is False
        assert any("move_type parameter is required" in error for error in result.errors)
        
        # Test invalid move_type
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=matching_dataframe,
            move_type="invalid_mode"
        )
        
        assert result.validation_passed is False
        assert any("Invalid move_type" in error for error in result.errors)
        
        # Test None DataFrame
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=None,
            move_type="existing_schema"
        )
        
        assert result.validation_passed is False
        assert any("DataFrame cannot be None" in error for error in result.errors)