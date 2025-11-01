"""
Unit tests for DataMoveUseCase.

These tests verify the main orchestrator functionality including table existence
detection, validation routing, dry-run capabilities, and storage loader integration.
"""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.infrastructure.storage.loaders import LocalLoader, S3Loader
from dataload.domain.entities import (
    DataMoveResult,
    ValidationReport,
    SchemaAnalysis,
    TableInfo,
    ColumnInfo,
    ValidationError,
    DatabaseOperationError,
    DataMoveError,
    DBOperationError,
)


class TestDataMoveUseCase:
    """Test cases for DataMoveUseCase orchestrator."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock DataMoveRepositoryInterface."""
        repository = AsyncMock()
        return repository

    @pytest.fixture
    def mock_storage_loader(self):
        """Create a mock StorageLoaderInterface."""
        storage_loader = MagicMock()
        return storage_loader

    @pytest.fixture
    def mock_validation_service(self):
        """Create a mock ValidationService."""
        validation_service = AsyncMock()
        return validation_service

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })

    @pytest.fixture
    def sample_table_info(self):
        """Create a sample TableInfo for testing."""
        return TableInfo(
            name='test_table',
            columns={
                'id': ColumnInfo(name='id', data_type='integer', nullable=False),
                'name': ColumnInfo(name='name', data_type='text', nullable=True),
                'age': ColumnInfo(name='age', data_type='integer', nullable=True)
            },
            primary_keys=['id'],
            constraints=[],
            indexes=[]
        )

    @pytest.fixture
    def successful_validation_report(self):
        """Create a successful validation report."""
        return ValidationReport(
            schema_analysis=SchemaAnalysis(
                table_exists=True,
                columns_added=[],
                columns_removed=[],
                columns_modified=[],
                case_conflicts=[],
                constraint_violations=[],
                compatible=True,
                requires_schema_update=False
            ),
            case_conflicts=[],
            type_mismatches=[],
            constraint_violations=[],
            recommendations=[],
            warnings=[],
            errors=[],
            validation_passed=True
        )

    def test_init(self, mock_repository, mock_storage_loader):
        """Test DataMoveUseCase initialization."""
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader
        )
        
        assert use_case.repository == mock_repository
        assert use_case.storage_loader == mock_storage_loader
        assert use_case.validation_service is not None

    def test_init_with_validation_service(
        self, mock_repository, mock_storage_loader, mock_validation_service
    ):
        """Test DataMoveUseCase initialization with custom validation service."""
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader,
            validation_service=mock_validation_service
        )
        
        assert use_case.validation_service == mock_validation_service

    @pytest.mark.asyncio
    async def test_execute_new_table_dry_run(
        self, 
        mock_repository, 
        mock_storage_loader, 
        mock_validation_service,
        sample_dataframe,
        successful_validation_report
    ):
        """Test execute method for new table creation in dry-run mode."""
        # Setup mocks
        mock_storage_loader.load_csv.return_value = sample_dataframe
        mock_repository.table_exists.return_value = False
        mock_validation_service.validate_data_move.return_value = successful_validation_report
        
        # Create use case
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader,
            validation_service=mock_validation_service
        )
        
        # Execute dry run
        result = await use_case.execute(
            csv_path="test.csv",
            table_name="test_table",
            dry_run=True
        )
        
        # Verify result
        assert isinstance(result, DataMoveResult)
        assert result.success is True
        assert result.rows_processed == 3
        assert result.table_created is False  # Dry run doesn't create table
        assert result.operation_type == "new_table"
        
        # Verify mocks were called correctly
        mock_storage_loader.load_csv.assert_called_once_with("test.csv")
        mock_repository.table_exists.assert_called_once_with("test_table")
        mock_validation_service.validate_data_move.assert_called_once_with(
            table_info=None, df=sample_dataframe, move_type=None
        )
        
        # Verify no actual data operations in dry run
        mock_repository.create_table_from_dataframe.assert_not_called()
        mock_repository.replace_table_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_existing_table_existing_schema(
        self,
        mock_repository,
        mock_storage_loader,
        mock_validation_service,
        sample_dataframe,
        sample_table_info,
        successful_validation_report
    ):
        """Test execute method for existing table with existing_schema validation."""
        # Setup mocks
        mock_storage_loader.load_csv.return_value = sample_dataframe
        mock_repository.table_exists.return_value = True
        mock_repository.get_table_info.return_value = sample_table_info
        mock_repository.replace_table_data.return_value = 3
        mock_validation_service.validate_data_move.return_value = successful_validation_report
        
        # Create use case
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader,
            validation_service=mock_validation_service
        )
        
        # Execute
        result = await use_case.execute(
            csv_path="test.csv",
            table_name="test_table",
            move_type="existing_schema"
        )
        
        # Verify result
        assert result.success is True
        assert result.rows_processed == 3
        assert result.table_created is False
        assert result.operation_type == "existing_schema"
        
        # Verify validation was called with correct parameters
        mock_validation_service.validate_data_move.assert_called_once_with(
            table_info=sample_table_info, df=sample_dataframe, move_type="existing_schema"
        )
        
        # Verify data replacement was called
        mock_repository.replace_table_data.assert_called_once_with(
            table_name="test_table", df=sample_dataframe, batch_size=1000
        )

    @pytest.mark.asyncio
    async def test_execute_missing_move_type_for_existing_table(
        self,
        mock_repository,
        mock_storage_loader,
        mock_validation_service,
        sample_dataframe,
        sample_table_info
    ):
        """Test that ValidationError is raised when move_type is missing for existing table."""
        # Setup mocks
        mock_storage_loader.load_csv.return_value = sample_dataframe
        mock_repository.table_exists.return_value = True
        mock_repository.get_table_info.return_value = sample_table_info
        
        # Mock validation service to raise error for missing move_type
        mock_validation_service.validate_data_move.side_effect = ValidationError(
            "move_type parameter is required when target table exists"
        )
        
        # Create use case
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader,
            validation_service=mock_validation_service
        )
        
        # Execute and expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute(
                csv_path="test.csv",
                table_name="test_table"
                # move_type is missing
            )
        
        assert "move_type parameter is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_invalid_move_type(
        self,
        mock_repository,
        mock_storage_loader,
        mock_validation_service,
        sample_dataframe,
        sample_table_info
    ):
        """Test that ValidationError is raised for invalid move_type."""
        # Setup mocks
        mock_storage_loader.load_csv.return_value = sample_dataframe
        mock_repository.table_exists.return_value = True
        mock_repository.get_table_info.return_value = sample_table_info
        
        # Mock validation service to raise error for invalid move_type
        mock_validation_service.validate_data_move.side_effect = ValidationError(
            "Invalid move_type: 'invalid'. Must be 'existing_schema' or 'new_schema'."
        )
        
        # Create use case
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader,
            validation_service=mock_validation_service
        )
        
        # Execute and expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute(
                csv_path="test.csv",
                table_name="test_table",
                move_type="invalid"
            )
        
        assert "Invalid move_type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_csv_loading_failure(
        self,
        mock_repository,
        mock_storage_loader,
        mock_validation_service
    ):
        """Test handling of CSV loading failures."""
        # Setup mocks
        mock_storage_loader.load_csv.side_effect = Exception("File not found")
        
        # Create use case
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader,
            validation_service=mock_validation_service
        )
        
        # Execute and expect DataMoveError
        with pytest.raises(Exception) as exc_info:
            await use_case.execute(
                csv_path="nonexistent.csv",
                table_name="test_table"
            )
        
        assert "Failed to load CSV file" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_operation_preview(
        self,
        mock_repository,
        mock_storage_loader,
        mock_validation_service,
        sample_dataframe,
        successful_validation_report
    ):
        """Test get_operation_preview method."""
        # Setup mocks
        mock_storage_loader.load_csv.return_value = sample_dataframe
        mock_repository.table_exists.return_value = False
        mock_validation_service.validate_data_move.return_value = successful_validation_report
        
        # Create use case
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader,
            validation_service=mock_validation_service
        )
        
        # Get preview
        preview = await use_case.get_operation_preview(
            csv_path="test.csv",
            table_name="test_table"
        )
        
        # Verify preview
        assert isinstance(preview, ValidationReport)
        assert preview.validation_passed is True
        
        # Verify no actual data operations occurred
        mock_repository.create_table_from_dataframe.assert_not_called()
        mock_repository.replace_table_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_empty_dataframe(
        self,
        mock_repository,
        mock_storage_loader,
        mock_validation_service,
        successful_validation_report
    ):
        """Test execute method with empty DataFrame."""
        # Setup mocks
        empty_df = pd.DataFrame()
        mock_storage_loader.load_csv.return_value = empty_df
        mock_repository.table_exists.return_value = False
        mock_repository.create_table_from_dataframe.return_value = {}
        mock_validation_service.validate_data_move.return_value = successful_validation_report
        
        # Create use case
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader,
            validation_service=mock_validation_service
        )
        
        # Execute
        result = await use_case.execute(
            csv_path="empty.csv",
            table_name="test_table"
        )
        
        # Verify result
        assert result.success is True
        assert result.rows_processed == 0
        assert result.table_created is True
        assert any("CSV file is empty" in warning for warning in result.warnings)
        
        # Verify empty table was created
        mock_repository.create_table_from_dataframe.assert_called_once()


class TestStorageLoaderIntegration:
    """Test cases for storage loader integration and auto-detection."""

    def test_create_storage_loader_s3_path(self):
        """Test that S3Loader is created for S3 URIs."""
        s3_path = "s3://my-bucket/data/file.csv"
        loader = DataMoveUseCase.create_storage_loader(s3_path)
        assert isinstance(loader, S3Loader)

    def test_create_storage_loader_local_path(self):
        """Test that LocalLoader is created for local paths."""
        local_path = "/path/to/file.csv"
        loader = DataMoveUseCase.create_storage_loader(local_path)
        assert isinstance(loader, LocalLoader)

    def test_create_storage_loader_relative_path(self):
        """Test that LocalLoader is created for relative paths."""
        relative_path = "data/file.csv"
        loader = DataMoveUseCase.create_storage_loader(relative_path)
        assert isinstance(loader, LocalLoader)

    def test_create_storage_loader_empty_path(self):
        """Test that ValueError is raised for empty paths."""
        with pytest.raises(ValueError, match="CSV path cannot be empty"):
            DataMoveUseCase.create_storage_loader("")

    def test_create_storage_loader_whitespace_path(self):
        """Test that ValueError is raised for whitespace-only paths."""
        with pytest.raises(ValueError, match="CSV path cannot be empty"):
            DataMoveUseCase.create_storage_loader("   ")

    def test_create_with_auto_loader(self):
        """Test creating DataMoveUseCase with auto loader detection."""
        mock_repository = AsyncMock()
        use_case = DataMoveUseCase.create_with_auto_loader(repository=mock_repository)
        
        assert use_case.repository == mock_repository
        assert use_case.storage_loader is None  # Auto-detection mode
        assert use_case.validation_service is not None

    @pytest.mark.asyncio
    async def test_auto_loader_selection_s3(self):
        """Test that S3Loader is automatically selected for S3 paths."""
        mock_repository = AsyncMock()
        mock_repository.table_exists.return_value = False
        
        # Create use case with auto-detection
        use_case = DataMoveUseCase.create_with_auto_loader(repository=mock_repository)
        
        # Mock the storage loader creation
        with patch.object(DataMoveUseCase, 'create_storage_loader') as mock_create:
            mock_s3_loader = MagicMock()
            mock_s3_loader.load_csv.return_value = pd.DataFrame({'id': [1], 'name': ['test']})
            mock_create.return_value = mock_s3_loader
            
            # Mock validation service
            mock_validation = AsyncMock()
            mock_validation.validate_data_move.return_value = ValidationReport(
                validation_passed=True,
                schema_analysis=SchemaAnalysis(
                    table_exists=False,
                    columns_added=[],
                    columns_removed=[],
                    columns_modified=[],
                    case_conflicts=[],
                    constraint_violations=[],
                    compatible=True,
                    requires_schema_update=False
                ),
                case_conflicts=[],
                type_mismatches=[],
                constraint_violations=[],
                errors=[],
                warnings=[],
                recommendations=[]
            )
            use_case.validation_service = mock_validation
            
            # Execute with S3 path
            s3_path = "s3://bucket/file.csv"
            result = await use_case.execute(
                csv_path=s3_path,
                table_name="test_table",
                dry_run=True
            )
            
            # Verify S3Loader was created for S3 path
            mock_create.assert_called_once_with(s3_path)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_auto_loader_selection_local(self):
        """Test that LocalLoader is automatically selected for local paths."""
        mock_repository = AsyncMock()
        mock_repository.table_exists.return_value = False
        
        # Create use case with auto-detection
        use_case = DataMoveUseCase.create_with_auto_loader(repository=mock_repository)
        
        # Mock the storage loader creation
        with patch.object(DataMoveUseCase, 'create_storage_loader') as mock_create:
            mock_local_loader = MagicMock()
            mock_local_loader.load_csv.return_value = pd.DataFrame({'id': [1], 'name': ['test']})
            mock_create.return_value = mock_local_loader
            
            # Mock validation service
            mock_validation = AsyncMock()
            mock_validation.validate_data_move.return_value = ValidationReport(
                validation_passed=True,
                schema_analysis=SchemaAnalysis(
                    table_exists=False,
                    columns_added=[],
                    columns_removed=[],
                    columns_modified=[],
                    case_conflicts=[],
                    constraint_violations=[],
                    compatible=True,
                    requires_schema_update=False
                ),
                case_conflicts=[],
                type_mismatches=[],
                constraint_violations=[],
                errors=[],
                warnings=[],
                recommendations=[]
            )
            use_case.validation_service = mock_validation
            
            # Execute with local path
            local_path = "data/file.csv"
            result = await use_case.execute(
                csv_path=local_path,
                table_name="test_table",
                dry_run=True
            )
            
            # Verify LocalLoader was created for local path
            mock_create.assert_called_once_with(local_path)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_dboperation_error_handling(self):
        """Test handling of DBOperationError from storage loaders."""
        mock_repository = AsyncMock()
        
        # Create use case with explicit storage loader
        mock_storage_loader = MagicMock()
        mock_storage_loader.load_csv.side_effect = DBOperationError("S3 access denied")
        
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader
        )
        
        # Execute and expect DataMoveError
        with pytest.raises(DataMoveError) as exc_info:
            await use_case.execute(
                csv_path="s3://bucket/file.csv",
                table_name="test_table"
            )
        
        # Verify error context
        error = exc_info.value
        assert "Storage operation failed" in str(error)
        assert error.context["error_type"] == "s3_operation_failed"
        assert error.context["source_type"] == "S3"
        assert "Check S3 bucket permissions" in error.context["suggestion"]

    @pytest.mark.asyncio
    async def test_local_dboperation_error_handling(self):
        """Test handling of DBOperationError from local storage loader."""
        mock_repository = AsyncMock()
        
        # Create use case with explicit storage loader
        mock_storage_loader = MagicMock()
        mock_storage_loader.load_csv.side_effect = DBOperationError("File not found")
        
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader
        )
        
        # Execute and expect DataMoveError
        with pytest.raises(DataMoveError) as exc_info:
            await use_case.execute(
                csv_path="local/file.csv",
                table_name="test_table"
            )
        
        # Verify error context
        error = exc_info.value
        assert "Storage operation failed" in str(error)
        assert error.context["error_type"] == "local_operation_failed"
        assert error.context["source_type"] == "local"
        assert "Check file path, permissions" in error.context["suggestion"]

    @pytest.mark.asyncio
    async def test_invalid_s3_uri_error_handling(self):
        """Test handling of invalid S3 URI format."""
        mock_repository = AsyncMock()
        
        # Create use case with auto-detection
        use_case = DataMoveUseCase.create_with_auto_loader(repository=mock_repository)
        
        # Mock storage loader creation to raise ValueError
        with patch.object(DataMoveUseCase, 'create_storage_loader') as mock_create:
            mock_s3_loader = MagicMock()
            mock_s3_loader.load_csv.side_effect = ValueError("Invalid S3 URI format")
            mock_create.return_value = mock_s3_loader
            
            # Execute and expect DataMoveError
            with pytest.raises(DataMoveError) as exc_info:
                await use_case.execute(
                    csv_path="s3://invalid-uri",
                    table_name="test_table"
                )
            
            # Verify error context
            error = exc_info.value
            assert "Invalid path or path format" in str(error)
            assert error.context["error_type"] == "invalid_s3_uri"
            assert "Check S3 URI format" in error.context["suggestion"]

    @pytest.mark.asyncio
    async def test_missing_boto3_error_handling(self):
        """Test handling of missing boto3 dependency."""
        mock_repository = AsyncMock()
        
        # Create use case with auto-detection
        use_case = DataMoveUseCase.create_with_auto_loader(repository=mock_repository)
        
        # Mock storage loader creation to raise ImportError
        with patch.object(DataMoveUseCase, 'create_storage_loader') as mock_create:
            mock_s3_loader = MagicMock()
            mock_s3_loader.load_csv.side_effect = ImportError("No module named 'boto3'")
            mock_create.return_value = mock_s3_loader
            
            # Execute and expect DataMoveError
            with pytest.raises(DataMoveError) as exc_info:
                await use_case.execute(
                    csv_path="s3://bucket/file.csv",
                    table_name="test_table"
                )
            
            # Verify error context
            error = exc_info.value
            assert "S3 support not available" in str(error)
            assert error.context["error_type"] == "missing_s3_dependency"
            assert "Install boto3" in error.context["suggestion"]

    def test_explicit_storage_loader_usage(self):
        """Test that explicit storage loader is used when provided."""
        mock_repository = AsyncMock()
        mock_storage_loader = MagicMock()
        
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=mock_storage_loader
        )
        
        # Test that the provided loader is used
        loader = use_case._get_storage_loader("any_path")
        assert loader == mock_storage_loader

    def test_auto_detection_when_no_loader_provided(self):
        """Test that auto-detection works when no storage loader is provided."""
        mock_repository = AsyncMock()
        
        use_case = DataMoveUseCase(
            repository=mock_repository,
            storage_loader=None  # No explicit loader
        )
        
        # Test S3 path detection
        with patch.object(DataMoveUseCase, 'create_storage_loader') as mock_create:
            mock_create.return_value = MagicMock()
            
            loader = use_case._get_storage_loader("s3://bucket/file.csv")
            mock_create.assert_called_once_with("s3://bucket/file.csv")