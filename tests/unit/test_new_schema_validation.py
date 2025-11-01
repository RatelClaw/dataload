"""Tests for new_schema flexible validation functionality."""

import pytest
import pandas as pd
from dataload.application.services.validation.validation_service import ValidationService
from dataload.application.services.validation.schema_validator import SchemaValidator
from dataload.application.services.validation.case_sensitivity_validator import CaseSensitivityValidator
from dataload.domain.entities import (
    TableInfo, ColumnInfo, Constraint, IndexInfo, CaseConflict
)


@pytest.fixture
def validation_service():
    """Create a ValidationService instance for testing."""
    return ValidationService()


@pytest.fixture
def sample_table_info():
    """Create sample table info for testing."""
    return TableInfo(
        name='test_table',
        columns={
            'id': ColumnInfo(name='id', data_type='integer', nullable=False),
            'name': ColumnInfo(name='name', data_type='text', nullable=True),
            'email': ColumnInfo(name='email', data_type='text', nullable=False),
            'created_at': ColumnInfo(name='created_at', data_type='timestamp', nullable=True),
        },
        primary_keys=['id'],
        constraints=[
            Constraint(name='pk_test_table', type='PRIMARY KEY', columns=['id']),
            Constraint(name='unique_email', type='UNIQUE', columns=['email'])
        ],
        indexes=[
            IndexInfo(name='idx_email', columns=['email'], index_type='btree', unique=True)
        ]
    )


@pytest.fixture
def compatible_csv_data():
    """Create CSV data compatible with new_schema mode."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],
        'created_at': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'new_column': ['value1', 'value2', 'value3']  # New column
    })


@pytest.fixture
def case_conflict_csv_data():
    """Create CSV data with case-sensitivity conflicts."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'Name': ['Alice', 'Bob', 'Charlie'],  # Case conflict with 'name'
        'EMAIL': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],  # Case conflict with 'email'
        'new_column': ['value1', 'value2', 'value3']
    })


@pytest.fixture
def schema_evolution_csv_data():
    """Create CSV data that represents schema evolution."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        # 'email' column removed
        'phone': ['+1234567890', '+0987654321', '+1122334455'],  # New column
        'status': ['active', 'inactive', 'active'],  # New column
        'updated_at': ['2023-01-01', '2023-01-02', '2023-01-03']  # New column
    })


class TestNewSchemaValidation:
    """Test cases for new_schema flexible validation."""
    
    @pytest.mark.asyncio
    async def test_new_schema_allows_column_additions(self, validation_service, sample_table_info, compatible_csv_data):
        """Test that new_schema mode allows column additions."""
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=compatible_csv_data,
            move_type='new_schema'
        )
        
        assert result.validation_passed
        assert 'new_column' in result.schema_analysis.columns_added
        assert len(result.schema_analysis.columns_removed) == 0
        assert result.schema_analysis.requires_schema_update
    
    @pytest.mark.asyncio
    async def test_new_schema_allows_column_removals(self, validation_service, sample_table_info, schema_evolution_csv_data):
        """Test that new_schema mode allows column removals."""
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=schema_evolution_csv_data,
            move_type='new_schema'
        )
        
        # Should allow removal of non-critical columns
        assert 'email' in result.schema_analysis.columns_removed
        assert 'created_at' in result.schema_analysis.columns_removed
        assert len(result.schema_analysis.columns_added) > 0
        assert result.schema_analysis.requires_schema_update
    
    @pytest.mark.asyncio
    async def test_new_schema_detects_case_conflicts(self, validation_service, sample_table_info, case_conflict_csv_data):
        """Test that new_schema mode detects and prevents case-sensitivity conflicts."""
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=case_conflict_csv_data,
            move_type='new_schema'
        )
        
        assert not result.validation_passed
        assert len(result.case_conflicts) > 0
        
        # Check for specific case conflicts
        conflict_columns = [c.csv_column for c in result.case_conflicts]
        assert 'Name' in conflict_columns or 'EMAIL' in conflict_columns
        
        # Should have error messages about case conflicts
        case_error_found = any('case-sensitivity conflict' in error.lower() for error in result.errors)
        assert case_error_found
    
    @pytest.mark.asyncio
    async def test_new_schema_prevents_primary_key_removal(self, validation_service, sample_table_info):
        """Test that new_schema mode prevents removal of primary key columns."""
        # CSV data without the primary key column 'id'
        csv_without_pk = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],
        })
        
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=csv_without_pk,
            move_type='new_schema'
        )
        
        # Should fail validation due to primary key removal
        assert not result.validation_passed
        pk_error_found = any('primary key' in error.lower() for error in result.errors)
        assert pk_error_found
    
    @pytest.mark.asyncio
    async def test_new_schema_backward_compatibility_checks(self, validation_service, sample_table_info, schema_evolution_csv_data):
        """Test backward compatibility checks in new_schema mode."""
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=schema_evolution_csv_data,
            move_type='new_schema'
        )
        
        # Should have warnings about removed indexed/constrained columns
        email_removal_warning = any('email' in warning.lower() for warning in result.warnings)
        assert email_removal_warning
        
        # Should have recommendations about schema changes
        schema_recommendations = any('schema' in rec.lower() for rec in result.recommendations)
        assert schema_recommendations
    
    @pytest.mark.asyncio
    async def test_new_schema_constraint_validation(self, validation_service, sample_table_info):
        """Test constraint validation in new_schema mode."""
        # CSV data with null values in non-nullable column
        csv_with_nulls = pd.DataFrame({
            'id': [1, 2, None],  # Null in primary key
            'name': ['Alice', 'Bob', 'Charlie'],
            'email': [None, 'bob@test.com', 'charlie@test.com'],  # Null in non-nullable column
        })
        
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=csv_with_nulls,
            move_type='new_schema'
        )
        
        assert not result.validation_passed
        
        # Should have errors about null constraint violations
        null_error_found = any('not null' in error.lower() for error in result.errors)
        assert null_error_found
    
    @pytest.mark.asyncio
    async def test_new_schema_comprehensive_reporting(self, validation_service, sample_table_info, schema_evolution_csv_data):
        """Test comprehensive reporting in new_schema mode."""
        result = await validation_service.validate_data_move(
            table_info=sample_table_info,
            df=schema_evolution_csv_data,
            move_type='new_schema'
        )
        
        # Should have detailed schema analysis
        assert result.schema_analysis is not None
        assert len(result.schema_analysis.columns_added) > 0
        assert len(result.schema_analysis.columns_removed) > 0
        
        # Should have recommendations
        assert len(result.recommendations) > 0
        
        # Should have warnings about schema changes
        assert len(result.warnings) > 0
        
        # Should indicate schema update is required
        assert result.schema_analysis.requires_schema_update


class TestCaseSensitivityValidator:
    """Test cases for enhanced case sensitivity validation."""
    
    def test_detect_case_conflicts(self):
        """Test case conflict detection."""
        validator = CaseSensitivityValidator()
        
        db_columns = ['id', 'name', 'email']
        csv_columns = ['id', 'Name', 'EMAIL', 'new_col']
        
        conflicts = validator.detect_case_conflicts(db_columns, csv_columns)
        
        assert len(conflicts) == 2  # 'Name' and 'EMAIL' conflicts
        
        conflict_types = [c.conflict_type for c in conflicts]
        assert 'case_mismatch' in conflict_types
    
    def test_suggest_conflict_resolution(self):
        """Test conflict resolution suggestions."""
        validator = CaseSensitivityValidator()
        
        db_columns = ['id', 'name', 'email']
        csv_columns = ['id', 'Name', 'EMAIL']
        
        suggestions = validator.suggest_conflict_resolution(db_columns, csv_columns)
        
        assert 'rename_csv_columns' in suggestions
        assert 'general_recommendations' in suggestions
        assert len(suggestions['rename_csv_columns']) > 0
        assert len(suggestions['general_recommendations']) > 0
    
    def test_validate_column_naming_convention(self):
        """Test column naming convention validation."""
        validator = CaseSensitivityValidator()
        
        columns = ['id', 'User Name', 'EMAIL', 'user-id', 'select']
        
        issues = validator.validate_column_naming_convention(columns)
        
        assert 'warnings' in issues
        assert 'recommendations' in issues
        assert 'best_practices' in issues
        
        # Should detect various issues
        assert len(issues['warnings']) > 0
        assert len(issues['recommendations']) > 0
        assert len(issues['best_practices']) > 0
    
    def test_prevent_case_conflicts_in_new_schema(self):
        """Test comprehensive case conflict prevention."""
        validator = CaseSensitivityValidator()
        
        db_columns = ['id', 'name', 'email']
        csv_columns = ['id', 'Name', 'EMAIL', 'Name']  # Includes duplicate
        
        result = validator.prevent_case_conflicts_in_new_schema(db_columns, csv_columns)
        
        assert result['has_conflicts']
        assert not result['safe_to_proceed']
        assert result['prevention_required']
        assert result['risk_assessment'] in ['low', 'medium', 'high']
        assert len(result['resolution_strategies']) > 0


class TestSchemaValidator:
    """Test cases for enhanced schema validation."""
    
    @pytest.mark.asyncio
    async def test_validate_new_schema_with_modifications(self, sample_table_info):
        """Test new schema validation with column modifications."""
        validator = SchemaValidator()
        
        # CSV with type changes
        csv_data = pd.DataFrame({
            'id': ['1', '2', '3'],  # Changed from integer to text
            'name': ['Alice', 'Bob', 'Charlie'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],
            'new_column': [1, 2, 3]  # New integer column
        })
        
        result = await validator.validate_new_schema(sample_table_info, csv_data)
        
        assert result.table_exists
        assert len(result.columns_added) == 1
        assert 'new_column' in result.columns_added
        assert result.requires_schema_update
        
        # Should detect type modification for 'id' column
        id_modification = next((m for m in result.columns_modified if m.column_name == 'id'), None)
        assert id_modification is not None
        assert id_modification.csv_type == 'text'
        assert id_modification.db_type == 'integer'


if __name__ == '__main__':
    pytest.main([__file__])