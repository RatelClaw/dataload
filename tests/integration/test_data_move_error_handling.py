"""
Integration tests for DataMove error handling and rollback functionality.
"""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, Mock, patch
import asyncio

from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.domain.entities import (
    DataMoveError,
    ValidationError,
    DatabaseOperationError,
    DataMoveResult,
    ValidationReport,
    SchemaAnalysis
)


class MockStorageLoader:
    """Mock storage loader for testing."""
    
    def __init__(self, should_fail=False, fail_with=None):
        self.should_fail = should_fail
        self.fail_with = fail_with or Exception("Mock storage error")
    
    def load_csv(self, path):
        if self.should_fail:
            raise self.fail_with
        
        # Return a simple test DataFrame
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })


class MockRepository:
    """Mock repository for testing."""
    
    def __init__(self, table_exists=False, should_fail_on=None):
        self.table_exists_result = table_exists
        self.should_fail_on = should_fail_on or []
        self.operations_called = []
    
    async def table_exists(self, table_name):
        self.operations_called.append(f"table_exists({table_name})")
        if "table_exists" in self.should_fail_on:
            raise DatabaseOperationError("Mock table_exists failure")
        return self.table_exists_result
    
    async def get_table_info(self, table_name):
        self.operations_called.append(f"get_table_info({table_name})")
        if "get_table_info" in self.should_fail_on:
            raise DatabaseOperationError("Mock get_table_info failure")
        
        from dataload.domain.entities import TableInfo, ColumnInfo
        return TableInfo(
            name=table_name,
            columns={
                'id': ColumnInfo('id', 'integer', False),
                'name': ColumnInfo('name', 'text', True),
                'value': ColumnInfo('value', 'numeric', True)
            },
            primary_keys=['id'],
            constraints=[],
            indexes=[]
        )
    
    async def create_table_from_dataframe(self, table_name, df, primary_key_columns=None):
        self.operations_called.append(f"create_table_from_dataframe({table_name})")
        if "create_table" in self.should_fail_on:
            raise DatabaseOperationError("Mock create_table failure")
        return {'id': 'integer', 'name': 'text', 'value': 'numeric'}
    
    async def replace_table_data(self, table_name, df, batch_size=1000):
        self.operations_called.append(f"replace_table_data({table_name})")
        if "replace_data" in self.should_fail_on:
            raise DatabaseOperationError("Mock replace_data failure")
        return len(df)
    
    async def transaction(self):
        """Mock transaction context manager."""
        return MockTransaction()


class MockTransaction:
    """Mock transaction context manager."""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # Simulate rollback on exception
            pass
        return False


class MockValidationService:
    """Mock validation service for testing."""
    
    def __init__(self, should_fail=False, fail_with=None):
        self.should_fail = should_fail
        self.fail_with = fail_with or ValidationError("Mock validation failure")
    
    async def validate_data_move(self, table_info, df, move_type=None):
        if self.should_fail:
            raise self.fail_with
        
        # Return a successful validation report
        schema_analysis = SchemaAnalysis(
            table_exists=table_info is not None,
            columns_added=[],
            columns_removed=[],
            columns_modified=[],
            case_conflicts=[],
            constraint_violations=[],
            compatible=True,
            requires_schema_update=False
        )
        
        return ValidationReport(
            schema_analysis=schema_analysis,
            case_conflicts=[],
            type_mismatches=[],
            constraint_violations=[],
            recommendations=[],
            warnings=[],
            errors=[],
            validation_passed=True
        )


class TestDataMoveErrorHandling:
    """Test comprehensive error handling in DataMove use case."""
    
    @pytest.mark.asyncio
    async def test_csv_loading_error_handling(self):
        """Test error handling when CSV loading fails."""
        storage_loader = MockStorageLoader(
            should_fail=True, 
            fail_with=FileNotFoundError("CSV file not found")
        )
        repository = MockRepository()
        validation_service = MockValidationService()
        
        use_case = DataMoveUseCase(repository, storage_loader, validation_service)
        
        with pytest.raises(DataMoveError) as exc_info:
            await use_case.execute("nonexistent.csv", "test_table")
        
        error = exc_info.value
        assert "Failed to load CSV file" in str(error)
        assert error.context["csv_path"] == "nonexistent.csv"
        assert error.context["error_type"] == "file_not_found"
    
    @pytest.mark.asyncio
    async def test_database_connection_error_handling(self):
        """Test error handling when database operations fail."""
        storage_loader = MockStorageLoader()
        repository = MockRepository(should_fail_on=["table_exists"])
        validation_service = MockValidationService()
        
        use_case = DataMoveUseCase(repository, storage_loader, validation_service)
        
        with pytest.raises(DatabaseOperationError) as exc_info:
            await use_case.execute("test.csv", "test_table")
        
        error = exc_info.value
        assert "Mock table_exists failure" in str(error)
    
    @pytest.mark.asyncio
    async def test_validation_error_collection(self):
        """Test that validation errors are properly collected and reported."""
        storage_loader = MockStorageLoader()
        repository = MockRepository(table_exists=True)
        validation_service = MockValidationService(
            should_fail=True,
            fail_with=ValidationError("Schema mismatch detected")
        )
        
        use_case = DataMoveUseCase(repository, storage_loader, validation_service)
        
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute("test.csv", "test_table", move_type="existing_schema")
        
        error = exc_info.value
        assert "Schema mismatch detected" in str(error)
        assert error.context["table_name"] == "test_table"
        assert error.context["move_type"] == "existing_schema"
    
    @pytest.mark.asyncio
    async def test_parameter_validation_error_handling(self):
        """Test error handling for invalid parameters."""
        storage_loader = MockStorageLoader()
        repository = MockRepository()
        validation_service = MockValidationService()
        
        use_case = DataMoveUseCase(repository, storage_loader, validation_service)
        
        # Test empty CSV path
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute("", "test_table")
        
        error = exc_info.value
        assert "csv_path cannot be empty" in str(error)
        # The error should have comprehensive context after going through error handling
        assert hasattr(error, 'context')
        assert error.context is not None
        assert "csv_path" in error.context  # Should have operation context
        
        # Test empty table name
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute("test.csv", "")
        
        error = exc_info.value
        assert "table_name cannot be empty" in str(error)
        assert hasattr(error, 'context')
        assert "table_name" in error.context
        
        # Test invalid batch size
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute("test.csv", "test_table", batch_size=0)
        
        error = exc_info.value
        assert "batch_size must be greater than 0" in str(error)
        assert hasattr(error, 'context')
        assert "batch_size" in error.context
    
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_data_movement_failure(self):
        """Test that transactions are rolled back when data movement fails."""
        storage_loader = MockStorageLoader()
        repository = MockRepository(should_fail_on=["replace_data"])
        validation_service = MockValidationService()
        
        use_case = DataMoveUseCase(repository, storage_loader, validation_service)
        
        with pytest.raises(DatabaseOperationError) as exc_info:
            await use_case.execute("test.csv", "test_table")
        
        error = exc_info.value
        assert "Mock replace_data failure" in str(error)
        
        # Verify that table creation was attempted before the failure
        assert "create_table_from_dataframe(test_table)" in repository.operations_called
        assert "replace_table_data(test_table)" in repository.operations_called
    
    @pytest.mark.asyncio
    async def test_comprehensive_error_context(self):
        """Test that errors include comprehensive context information."""
        storage_loader = MockStorageLoader()
        repository = MockRepository(should_fail_on=["create_table"])
        validation_service = MockValidationService()
        
        use_case = DataMoveUseCase(repository, storage_loader, validation_service)
        
        with pytest.raises(DatabaseOperationError) as exc_info:
            await use_case.execute(
                "test.csv", 
                "test_table", 
                batch_size=500,
                primary_key_columns=["id"]
            )
        
        error = exc_info.value
        context = error.context
        
        # Verify comprehensive context is included
        assert context["csv_path"] == "test.csv"
        assert context["table_name"] == "test_table"
        assert context["batch_size"] == 500
        assert context["primary_key_columns"] == ["id"]
        assert "execution_time" in context
        assert "operation_stage" in context
    
    @pytest.mark.asyncio
    async def test_successful_operation_with_warnings(self):
        """Test successful operation that includes warnings."""
        storage_loader = MockStorageLoader()
        repository = MockRepository()
        
        # Create validation service that returns warnings
        validation_service = MockValidationService()
        
        # Mock the validation to return warnings
        async def mock_validate_with_warnings(table_info, df, move_type=None):
            schema_analysis = SchemaAnalysis(
                table_exists=False,
                columns_added=list(df.columns),
                columns_removed=[],
                columns_modified=[],
                case_conflicts=[],
                constraint_violations=[],
                compatible=True,
                requires_schema_update=False
            )
            
            return ValidationReport(
                schema_analysis=schema_analysis,
                case_conflicts=[],
                type_mismatches=[],
                constraint_violations=[],
                recommendations=["New table will be created"],
                warnings=["CSV file contains mixed data types"],
                errors=[],
                validation_passed=True
            )
        
        validation_service.validate_data_move = mock_validate_with_warnings
        
        use_case = DataMoveUseCase(repository, storage_loader, validation_service)
        
        result = await use_case.execute("test.csv", "test_table")
        
        assert result.success is True
        assert len(result.warnings) > 0
        assert "CSV file contains mixed data types" in result.warnings
        assert result.rows_processed == 3  # Mock DataFrame has 3 rows
        assert result.table_created is True
    
    @pytest.mark.asyncio
    async def test_dry_run_with_validation_errors(self):
        """Test dry run mode with validation errors."""
        storage_loader = MockStorageLoader()
        repository = MockRepository(table_exists=True)
        validation_service = MockValidationService(
            should_fail=True,
            fail_with=ValidationError("Validation failed in dry run")
        )
        
        use_case = DataMoveUseCase(repository, storage_loader, validation_service)
        
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute(
                "test.csv", 
                "test_table", 
                move_type="existing_schema",
                dry_run=True
            )
        
        error = exc_info.value
        assert "Validation failed in dry run" in str(error)
        
        # Verify no data operations were attempted in dry run
        assert "replace_table_data" not in str(repository.operations_called)
        assert "create_table_from_dataframe" not in str(repository.operations_called)


if __name__ == "__main__":
    pytest.main([__file__])