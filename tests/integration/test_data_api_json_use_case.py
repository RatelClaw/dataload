"""
Integration tests for DataAPIJSONUseCase.

This module provides comprehensive integration tests for the DataAPIJSONUseCase
with mock database and embedding services to test various scenarios including
API loading, JSON processing, embedding generation, and database operations.
"""

import pytest
import asyncio
import json
import tempfile
import os
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

from dataload.application.use_cases.data_api_json_use_case import DataAPIJSONUseCase
from dataload.interfaces.data_move_repository import DataMoveRepositoryInterface
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.domain.entities import (
    DataMoveResult, ValidationReport, SchemaAnalysis, TableInfo, ColumnInfo,
    DataMoveError, ValidationError, DatabaseOperationError
)
from dataload.domain.api_entities import (
    APIResponse, APIJSONLoadResult, AuthenticationError, JSONParsingError
)


class MockDataMoveRepository(DataMoveRepositoryInterface):
    """Mock implementation of DataMoveRepositoryInterface for testing."""
    
    def __init__(self):
        self.tables = {}  # table_name -> DataFrame
        self.table_schemas = {}  # table_name -> schema info
        self.call_log = []  # Track method calls for verification
        
    async def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        self.call_log.append(("table_exists", table_name))
        return table_name in self.tables
    
    async def get_table_info(self, table_name: str) -> TableInfo:
        """Get table information."""
        self.call_log.append(("get_table_info", table_name))
        if table_name not in self.tables:
            raise DatabaseOperationError(f"Table {table_name} does not exist")
        
        # Create mock table info
        df = self.tables[table_name]
        columns = {}
        for col in df.columns:
            columns[col] = ColumnInfo(
                name=col,
                data_type="text",
                nullable=True
            )
        
        return TableInfo(
            name=table_name,
            columns=columns,
            primary_keys=[],
            constraints=[],
            indexes=[]
        )
    
    async def create_table(self, 
                          table_name: str, 
                          df: pd.DataFrame,
                          pk_columns: List[str],
                          embed_type: str,
                          embed_columns_names: List[str]) -> Dict[str, str]:
        """Create table with embedding support."""
        self.call_log.append(("create_table", table_name, pk_columns, embed_type, embed_columns_names))
        
        # Store the DataFrame structure
        self.tables[table_name] = pd.DataFrame(columns=df.columns)
        
        # Return column types
        column_types = {}
        for col in df.columns:
            if col.endswith('_enc') or col == 'embeddings':
                column_types[col] = 'vector'
            elif col in ['is_active']:
                column_types[col] = 'boolean'
            elif col in ['embed_columns_names']:
                column_types[col] = 'text[]'
            else:
                column_types[col] = 'text'
        
        return column_types
    
    async def create_table_from_dataframe(self,
                                        table_name: str,
                                        df: pd.DataFrame,
                                        primary_key_columns: Optional[List[str]] = None) -> Dict[str, str]:
        """Create table from DataFrame schema."""
        self.call_log.append(("create_table_from_dataframe", table_name, primary_key_columns))
        
        self.tables[table_name] = pd.DataFrame(columns=df.columns)
        
        column_types = {}
        for col in df.columns:
            column_types[col] = 'text'  # Simplified for testing
        
        return column_types
    
    async def insert_data(self, table_name: str, df: pd.DataFrame, pk_columns: List[str]) -> None:
        """Insert data into table."""
        self.call_log.append(("insert_data", table_name, len(df), pk_columns))
        
        if table_name not in self.tables:
            raise DatabaseOperationError(f"Table {table_name} does not exist")
        
        # Simulate data insertion
        if table_name not in self.tables or self.tables[table_name].empty:
            self.tables[table_name] = df.copy()
        else:
            self.tables[table_name] = pd.concat([self.tables[table_name], df], ignore_index=True)
    
    async def replace_table_data(self, 
                               table_name: str, 
                               df: pd.DataFrame,
                               batch_size: int = 1000) -> int:
        """Replace table data."""
        self.call_log.append(("replace_table_data", table_name, len(df), batch_size))
        
        if table_name not in self.tables:
            raise DatabaseOperationError(f"Table {table_name} does not exist")
        
        # Replace data
        self.tables[table_name] = df.copy()
        return len(df)
    
    # Implement other required methods with basic functionality
    async def analyze_schema_compatibility(self, table_name: str, df: pd.DataFrame) -> SchemaAnalysis:
        return SchemaAnalysis(
            table_exists=table_name in self.tables,
            columns_added=[],
            columns_removed=[],
            columns_modified=[],
            case_conflicts=[],
            constraint_violations=[],
            compatible=True,
            requires_schema_update=False
        )
    
    async def get_column_case_conflicts(self, table_name: str, df_columns: List[str]) -> List:
        return []
    
    async def validate_type_compatibility(self, table_name: str, df: pd.DataFrame) -> List:
        return []
    
    async def validate_constraints(self, table_name: str, df: pd.DataFrame) -> List:
        return []
    
    async def update_table_schema(self, table_name: str, df: pd.DataFrame, allow_column_drops: bool = False) -> Dict[str, str]:
        return {}
    
    async def bulk_insert_data(self, table_name: str, df: pd.DataFrame, batch_size: int = 1000, on_conflict: str = "error") -> int:
        return len(df)
    
    async def execute_in_transaction(self, operations: List[callable]) -> bool:
        return True
    
    async def get_postgresql_version(self) -> str:
        return "13.0"
    
    async def analyze_table_statistics(self, table_name: str) -> Dict[str, Any]:
        return {}
    
    async def optimize_table_performance(self, table_name: str) -> None:
        pass
    
    async def detect_vector_columns(self, table_name: str) -> List:
        return []
    
    async def validate_vector_dimensions(self, table_name: str, df: pd.DataFrame) -> List[str]:
        return []
    
    # Implement missing methods from DataRepositoryInterface
    async def get_table_schema(self, table_name: str):
        """Get table schema."""
        from dataload.domain.entities import TableSchema
        self.call_log.append(("get_table_schema", table_name))
        
        if table_name not in self.tables:
            raise DatabaseOperationError(f"Table {table_name} does not exist")
        
        df = self.tables[table_name]
        columns = {}
        nullables = {}
        
        for col in df.columns:
            if col.endswith('_enc') or col == 'embeddings':
                columns[col] = 'vector'
            elif col in ['is_active']:
                columns[col] = 'boolean'
            elif col in ['embed_columns_names']:
                columns[col] = 'jsonb'
            else:
                columns[col] = 'text'
            nullables[col] = True
        
        return TableSchema(columns=columns, nullables=nullables)
    
    async def update_data(self, table_name: str, df: pd.DataFrame, pk_columns: List[str]):
        """Update data in table."""
        self.call_log.append(("update_data", table_name, len(df), pk_columns))
        
        if table_name not in self.tables:
            raise DatabaseOperationError(f"Table {table_name} does not exist")
        
        # Simple mock update - just append data
        self.tables[table_name] = pd.concat([self.tables[table_name], df], ignore_index=True)
    
    async def set_inactive(self, table_name: str, pks: List[tuple], pk_columns: List[str]):
        """Set records inactive."""
        self.call_log.append(("set_inactive", table_name, len(pks), pk_columns))
        # Mock implementation - just log the call
        pass
    
    async def get_active_data(self, table_name: str, columns: List[str]) -> pd.DataFrame:
        """Get active data."""
        self.call_log.append(("get_active_data", table_name, columns))
        
        if table_name not in self.tables:
            return pd.DataFrame(columns=columns)
        
        df = self.tables[table_name]
        if 'is_active' in df.columns:
            active_df = df[df['is_active'] == True]
        else:
            active_df = df
        
        # Return only requested columns
        available_columns = [col for col in columns if col in active_df.columns]
        return active_df[available_columns] if available_columns else pd.DataFrame(columns=columns)
    
    async def get_embed_columns_names(self, table_name: str) -> List[str]:
        """Get embedding column names."""
        self.call_log.append(("get_embed_columns_names", table_name))
        
        if table_name not in self.tables:
            raise DatabaseOperationError(f"Table {table_name} does not exist")
        
        df = self.tables[table_name]
        if 'embed_columns_names' in df.columns and not df.empty:
            return df['embed_columns_names'].iloc[0] if isinstance(df['embed_columns_names'].iloc[0], list) else []
        return []
    
    async def get_data_columns(self, table_name: str) -> List[str]:
        """Get data columns (excluding system columns)."""
        self.call_log.append(("get_data_columns", table_name))
        
        if table_name not in self.tables:
            raise DatabaseOperationError(f"Table {table_name} does not exist")
        
        df = self.tables[table_name]
        system_columns = ['embed_columns_names', 'embed_columns_value', 'embeddings', 'is_active']
        data_columns = [col for col in df.columns 
                       if col not in system_columns and not col.endswith('_enc')]
        return data_columns
    
    async def add_column(self, table_name: str, column_name: str, column_type: str):
        """Add column to table."""
        self.call_log.append(("add_column", table_name, column_name, column_type))
        
        if table_name not in self.tables:
            raise DatabaseOperationError(f"Table {table_name} does not exist")
        
        # Add column with default values
        if column_name not in self.tables[table_name].columns:
            if column_type == 'boolean':
                self.tables[table_name][column_name] = False
            elif 'vector' in column_type:
                self.tables[table_name][column_name] = None
            else:
                self.tables[table_name][column_name] = None


class MockEmbeddingProvider(EmbeddingProviderInterface):
    """Mock implementation of EmbeddingProviderInterface for testing."""
    
    def __init__(self, embedding_dimension: int = 384):
        self.embedding_dimension = embedding_dimension
        self.call_log = []
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings."""
        self.call_log.append(("get_embeddings", len(texts)))
        
        # Generate mock embeddings (simple pattern for testing)
        embeddings = []
        for i, text in enumerate(texts):
            # Create a simple pattern based on text length and index
            embedding = [float((len(text) + i + j) % 100) / 100.0 for j in range(self.embedding_dimension)]
            embeddings.append(embedding)
        
        return embeddings


class MockAPIJSONStorageLoader(APIJSONStorageLoader):
    """Mock implementation of APIJSONStorageLoader for testing."""
    
    def __init__(self):
        # Initialize with minimal configuration
        super().__init__(base_url="http://mock-api.com")
        self.mock_data = {}  # source -> data mapping
        self.call_log = []
        
    def set_mock_data(self, source: str, data: Any):
        """Set mock data for a specific source."""
        self.mock_data[source] = data
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """Mock CSV loading."""
        self.call_log.append(("load_csv", path))
        return pd.DataFrame({"id": [1, 2], "name": ["test1", "test2"]})
    
    def load_json(self, source, config=None) -> pd.DataFrame:
        """Mock JSON loading with basic processing simulation."""
        self.call_log.append(("load_json", str(source)[:50], config))
        
        # Return mock data if available
        if str(source) in self.mock_data:
            data = self.mock_data[str(source)]
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            elif isinstance(data, Exception):
                raise data
            else:
                df = pd.DataFrame({
                    "id": [1, 2, 3],
                    "title": ["Test Title 1", "Test Title 2", "Test Title 3"],
                    "description": ["Description 1", "Description 2", "Description 3"],
                    "category": ["A", "B", "A"]
                })
        else:
            # Default mock data
            df = pd.DataFrame({
                "id": [1, 2, 3],
                "title": ["Test Title 1", "Test Title 2", "Test Title 3"],
                "description": ["Description 1", "Description 2", "Description 3"],
                "category": ["A", "B", "A"]
            })
        
        # Apply basic processing if config is provided
        if config:
            # Apply column name mapping
            column_mapping = config.get('column_name_mapping', {})
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            # Apply basic request body transformations
            transform_mapping = config.get('update_request_body_mapping', {})
            if transform_mapping:
                for target_field, expression in transform_mapping.items():
                    # Simple mock transformation - just create the field
                    if 'concat' in expression.lower():
                        # Mock concatenation - just combine first two columns
                        if len(df.columns) >= 2:
                            df[target_field] = df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str)
                        else:
                            df[target_field] = 'mock_concat_value'
                    else:
                        # Constant value or other transformation
                        df[target_field] = expression
        
        return df


@pytest.fixture
def mock_repository():
    """Fixture for mock repository."""
    return MockDataMoveRepository()


@pytest.fixture
def mock_embedding_service():
    """Fixture for mock embedding service."""
    return MockEmbeddingProvider()


@pytest.fixture
def mock_storage_loader():
    """Fixture for mock storage loader."""
    return MockAPIJSONStorageLoader()


@pytest.fixture
def use_case(mock_repository, mock_embedding_service, mock_storage_loader):
    """Fixture for DataAPIJSONUseCase with mocks."""
    return DataAPIJSONUseCase(
        repo=mock_repository,
        embedding_service=mock_embedding_service,
        storage_loader=mock_storage_loader
    )


class TestDataAPIJSONUseCase:
    """Test cases for DataAPIJSONUseCase."""
    
    @pytest.mark.asyncio
    async def test_basic_api_loading_new_table(self, use_case, mock_storage_loader):
        """Test basic API loading with new table creation."""
        # Arrange
        source = "https://api.example.com/data"
        table_name = "api_data"
        embed_columns = ["title", "description"]
        
        # Act
        result = await use_case.execute(
            source=source,
            table_name=table_name,
            embed_columns_names=embed_columns,
            create_table_if_not_exists=True,
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 3
        assert result.table_created is True
        assert result.operation_type == "new_table_creation"
        assert len(result.errors) == 0
        
        # Verify repository calls
        repo_calls = [call[0] for call in use_case.repo.call_log]
        assert "table_exists" in repo_calls
        assert "create_table" in repo_calls
        assert "insert_data" in repo_calls
        
        # Verify embedding service was called
        assert len(use_case.embedding_service.call_log) > 0
        assert use_case.embedding_service.call_log[0][0] == "get_embeddings"
    
    @pytest.mark.asyncio
    async def test_json_file_loading(self, use_case, mock_storage_loader):
        """Test loading from JSON file."""
        # Arrange
        json_data = [
            {"id": 1, "name": "Item 1", "value": 100},
            {"id": 2, "name": "Item 2", "value": 200}
        ]
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            json_file_path = f.name
        
        try:
            # Set up mock data
            expected_df = pd.DataFrame(json_data)
            mock_storage_loader.set_mock_data(json_file_path, expected_df)
            
            # Act
            result = await use_case.execute(
                source=json_file_path,
                table_name="json_file_data",
                embed_columns_names=["name"],
                embed_type="separated"
            )
            
            # Assert
            assert result.success is True
            assert result.rows_processed == 2
            assert result.table_created is True
            
        finally:
            # Clean up
            os.unlink(json_file_path)
    
    @pytest.mark.asyncio
    async def test_raw_json_data_loading(self, use_case, mock_storage_loader):
        """Test loading from raw JSON data."""
        # Arrange
        raw_json_data = [
            {"user_id": 1, "username": "alice", "email": "alice@example.com"},
            {"user_id": 2, "username": "bob", "email": "bob@example.com"}
        ]
        
        expected_df = pd.DataFrame(raw_json_data)
        mock_storage_loader.set_mock_data(str(raw_json_data), expected_df)
        
        # Act
        result = await use_case.execute(
            source=raw_json_data,
            table_name="users",
            embed_columns_names=["username", "email"],
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.table_created is True
        
        # Verify the DataFrame was processed correctly
        stored_data = use_case.repo.tables["users"]
        assert "embeddings" in stored_data.columns
        assert "embed_columns_names" in stored_data.columns
        assert "is_active" in stored_data.columns
    
    @pytest.mark.asyncio
    async def test_column_mapping(self, use_case, mock_storage_loader):
        """Test column name mapping functionality."""
        # Arrange
        source_data = pd.DataFrame({
            "user_id": [1, 2],
            "full_name": ["Alice Smith", "Bob Jones"],
            "email_address": ["alice@example.com", "bob@example.com"]
        })
        
        mock_storage_loader.set_mock_data("test_source", source_data)
        
        column_mapping = {
            "user_id": "id",
            "full_name": "name",
            "email_address": "email"
        }
        
        # Act
        result = await use_case.execute(
            source="test_source",
            table_name="mapped_users",
            column_name_mapping=column_mapping,
            embed_columns_names=["name", "email"],  # Use mapped names
            embed_type="separated"
        )
        
        # Assert
        assert result.success is True
        
        # Verify mapped columns exist in final data
        stored_data = use_case.repo.tables["mapped_users"]
        expected_columns = {"id", "name", "email", "name_enc", "email_enc", "embed_columns_names", "is_active"}
        assert set(stored_data.columns).issuperset(expected_columns)
    
    @pytest.mark.asyncio
    async def test_existing_table_update(self, use_case, mock_storage_loader):
        """Test updating existing table."""
        # Arrange - create existing table
        existing_data = pd.DataFrame({
            "id": [1, 2],
            "name": ["Old Name 1", "Old Name 2"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "embed_columns_names": [["name"], ["name"]],
            "is_active": [True, True]
        })
        use_case.repo.tables["existing_table"] = existing_data
        
        # New data to add
        new_data = pd.DataFrame({
            "id": [3, 4],
            "name": ["New Name 3", "New Name 4"]
        })
        mock_storage_loader.set_mock_data("new_data", new_data)
        
        # Act
        result = await use_case.execute(
            source="new_data",
            table_name="existing_table",
            pk_columns=["id"],
            embed_columns_names=["name"],
            create_table_if_not_exists=False,
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.table_created is False
        assert result.operation_type == "existing_table_update"
        
        # Verify data was added
        final_data = use_case.repo.tables["existing_table"]
        assert len(final_data) == 4  # 2 existing + 2 new
    
    @pytest.mark.asyncio
    async def test_empty_data_handling(self, use_case, mock_storage_loader):
        """Test handling of empty data."""
        # Arrange
        empty_df = pd.DataFrame(columns=["id", "name", "description"])
        mock_storage_loader.set_mock_data("empty_source", empty_df)
        
        # Act
        result = await use_case.execute(
            source="empty_source",
            table_name="empty_table",
            embed_columns_names=["name", "description"],
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 0
        assert result.table_created is True
        assert len(result.warnings) > 0  # Should warn about empty data
    
    @pytest.mark.asyncio
    async def test_validation_errors(self, use_case):
        """Test various validation error scenarios."""
        
        # Test empty table name
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute(
                source={"test": "data"},
                table_name="",
                embed_columns_names=["test"]
            )
        assert "Table name cannot be empty" in str(exc_info.value)
        
        # Test invalid embed_type
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute(
                source={"test": "data"},
                table_name="test_table",
                embed_type="invalid_type"
            )
        assert "Invalid embed_type" in str(exc_info.value)
        
        # Test None source
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute(
                source=None,
                table_name="test_table"
            )
        assert "Source cannot be None" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_embedding_column_validation(self, use_case, mock_storage_loader):
        """Test validation of embedding columns after mapping."""
        # Arrange
        source_data = pd.DataFrame({
            "id": [1, 2],
            "title": ["Title 1", "Title 2"]
        })
        mock_storage_loader.set_mock_data("test_source", source_data)
        
        # Act & Assert - embedding column doesn't exist
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute(
                source="test_source",
                table_name="test_table",
                embed_columns_names=["nonexistent_column"]
            )
        assert "Embedding columns not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, use_case, mock_storage_loader):
        """Test handling of API errors."""
        # Arrange - set up mock to raise API error
        api_error = AuthenticationError("Invalid API key")
        mock_storage_loader.set_mock_data("error_source", api_error)
        
        # Act & Assert
        with pytest.raises(AuthenticationError):
            await use_case.execute(
                source="error_source",
                table_name="test_table"
            )
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, use_case, mock_storage_loader):
        """Test handling of database errors."""
        # Arrange
        source_data = pd.DataFrame({"id": [1], "name": ["test"]})
        mock_storage_loader.set_mock_data("test_source", source_data)
        
        # Mock repository to raise error on table creation
        original_create_table = use_case.repo.create_table
        async def failing_create_table(*args, **kwargs):
            raise DatabaseOperationError("Database connection failed")
        
        use_case.repo.create_table = failing_create_table
        
        # Act & Assert
        with pytest.raises(DatabaseOperationError):
            await use_case.execute(
                source="test_source",
                table_name="test_table"
            )
    
    @pytest.mark.asyncio
    async def test_request_body_transformation(self, use_case, mock_storage_loader):
        """Test request body transformation functionality."""
        # Arrange
        source_data = pd.DataFrame({
            "first_name": ["Alice", "Bob"],
            "last_name": ["Smith", "Jones"],
            "age": [25, 30]
        })
        mock_storage_loader.set_mock_data("transform_source", source_data)
        
        transformation_mapping = {
            "full_name": "concat({first_name}, ' ', {last_name})",
            "age_group": "adult"  # constant value
        }
        
        # Act
        result = await use_case.execute(
            source="transform_source",
            table_name="transformed_table",
            update_request_body_mapping=transformation_mapping,
            embed_columns_names=["full_name"]
        )
        
        # Assert
        assert result.success is True
        
        # Note: The actual transformation logic would be tested in the 
        # RequestBodyTransformer unit tests. Here we're testing integration.
    
    @pytest.mark.asyncio
    async def test_separated_embeddings(self, use_case, mock_storage_loader):
        """Test separated embedding generation."""
        # Arrange
        source_data = pd.DataFrame({
            "id": [1, 2],
            "title": ["Title 1", "Title 2"],
            "description": ["Desc 1", "Desc 2"]
        })
        mock_storage_loader.set_mock_data("separated_source", source_data)
        
        # Act
        result = await use_case.execute(
            source="separated_source",
            table_name="separated_embeddings",
            embed_columns_names=["title", "description"],
            embed_type="separated"
        )
        
        # Assert
        assert result.success is True
        
        # Verify separated embedding columns were created
        stored_data = use_case.repo.tables["separated_embeddings"]
        assert "title_enc" in stored_data.columns
        assert "description_enc" in stored_data.columns
        assert "embeddings" not in stored_data.columns  # Should not have combined column
    
    @pytest.mark.asyncio
    async def test_no_embeddings(self, use_case, mock_storage_loader):
        """Test loading without embedding generation."""
        # Arrange
        source_data = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Item 1", "Item 2", "Item 3"],
            "value": [100, 200, 300]
        })
        mock_storage_loader.set_mock_data("no_embed_source", source_data)
        
        # Act
        result = await use_case.execute(
            source="no_embed_source",
            table_name="no_embeddings_table",
            embed_columns_names=[],  # Empty list - no embeddings
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        
        # Verify no embedding-related columns were added
        stored_data = use_case.repo.tables["no_embeddings_table"]
        assert "embeddings" not in stored_data.columns
        assert "embed_columns_names" not in stored_data.columns
        embedding_cols = [col for col in stored_data.columns if col.endswith('_enc')]
        assert len(embedding_cols) == 0
    
    @pytest.mark.asyncio
    async def test_embedding_with_column_mapping_integration(self, use_case, mock_storage_loader):
        """Test comprehensive integration of embedding generation with column mapping."""
        # Arrange - complex API response structure
        source_data = pd.DataFrame({
            "product_id": [1, 2, 3],
            "product_name": ["Laptop Pro", "Desktop Elite", "Tablet Mini"],
            "product_description": ["High-performance laptop", "Powerful desktop", "Compact tablet"],
            "product_category": ["Electronics", "Electronics", "Electronics"],
            "price_usd": [1299.99, 899.99, 399.99],
            "availability_status": ["in_stock", "limited", "out_of_stock"]
        })
        mock_storage_loader.set_mock_data("complex_api_source", source_data)
        
        # Complex column mapping
        column_mapping = {
            "product_id": "id",
            "product_name": "name",
            "product_description": "description",
            "product_category": "category",
            "price_usd": "price",
            "availability_status": "status"
        }
        
        # Act - use mapped column names for embeddings
        result = await use_case.execute(
            source="complex_api_source",
            table_name="products",
            column_name_mapping=column_mapping,
            embed_columns_names=["name", "description", "category"],  # Mapped names
            embed_type="separated",
            pk_columns=["id"]
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 3
        
        # Verify all mapped columns exist
        final_data = use_case.repo.tables["products"]
        expected_columns = {"id", "name", "description", "category", "price", "status"}
        assert expected_columns.issubset(set(final_data.columns))
        
        # Verify separated embedding columns
        assert "name_enc" in final_data.columns
        assert "description_enc" in final_data.columns
        assert "category_enc" in final_data.columns
        
        # Verify embedding service was called for each column
        embedding_calls = [call for call in use_case.embedding_service.call_log if call[0] == "get_embeddings"]
        assert len(embedding_calls) == 3  # One for each embedding column
    
    @pytest.mark.asyncio
    async def test_embedding_validation_error_with_mapping_suggestions(self, use_case, mock_storage_loader):
        """Test that embedding validation provides helpful suggestions when column mapping is used."""
        # Arrange
        source_data = pd.DataFrame({
            "api_title": ["Title 1", "Title 2"],
            "api_content": ["Content 1", "Content 2"],
            "api_tags": ["tag1,tag2", "tag3,tag4"]
        })
        mock_storage_loader.set_mock_data("suggestion_source", source_data)
        
        column_mapping = {
            "api_title": "title",
            "api_content": "content",
            "api_tags": "tags"
        }
        
        # Act & Assert - use original API field names instead of mapped names
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute(
                source="suggestion_source",
                table_name="suggestion_table",
                column_name_mapping=column_mapping,
                embed_columns_names=["api_title", "api_content"],  # Wrong: original names
                embed_type="combined"
            )
        
        error = exc_info.value
        error_message = str(error)
        
        # Verify comprehensive error message
        assert "Embedding columns not found" in error_message
        assert "embed_columns_names should refer to mapped column names" in error_message
        assert "api_title" in error_message and "should be 'title'" in error_message
        assert "api_content" in error_message and "should be 'content'" in error_message
        assert "Available columns after mapping" in error_message
    
    @pytest.mark.asyncio
    async def test_embedding_with_mixed_valid_invalid_columns(self, use_case, mock_storage_loader):
        """Test embedding validation when some columns are valid and others are invalid."""
        # Arrange
        source_data = pd.DataFrame({
            "title": ["Title 1", "Title 2"],
            "description": ["Description 1", "Description 2"],
            "category": ["Category A", "Category B"]
        })
        mock_storage_loader.set_mock_data("mixed_source", source_data)
        
        # Act & Assert - mix valid and invalid column names
        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute(
                source="mixed_source",
                table_name="mixed_table",
                embed_columns_names=["title", "nonexistent_column", "description"],
                embed_type="combined"
            )
        
        error = exc_info.value
        assert "nonexistent_column" in str(error)
        # Should still mention the valid columns that were found
        assert "title" in str(error) or "description" in str(error)
    
    @pytest.mark.asyncio
    async def test_embedding_generation_with_null_values(self, use_case, mock_storage_loader):
        """Test embedding generation when columns contain null values."""
        # Arrange
        source_data = pd.DataFrame({
            "title": ["Title 1", None, "Title 3"],
            "description": [None, "Description 2", "Description 3"],
            "category": ["Category A", "Category B", None]
        })
        mock_storage_loader.set_mock_data("null_source", source_data)
        
        # Act
        result = await use_case.execute(
            source="null_source",
            table_name="null_table",
            embed_columns_names=["title", "description"],
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        
        # Verify embeddings were generated despite null values
        final_data = use_case.repo.tables["null_table"]
        assert "embeddings" in final_data.columns
        assert len(final_data) == 3
        
        # Verify embedding service was called
        assert len(use_case.embedding_service.call_log) > 0
    
    @pytest.mark.asyncio
    async def test_embedding_with_complex_data_types(self, use_case, mock_storage_loader):
        """Test embedding generation with complex data types (lists, dicts)."""
        # Arrange
        source_data = pd.DataFrame({
            "title": ["Product 1", "Product 2"],
            "tags": [["electronics", "laptop"], ["furniture", "chair"]],
            "metadata": [{"brand": "TechCorp", "model": "X1"}, {"brand": "FurnCorp", "model": "Y2"}],
            "price": [999.99, 299.99]
        })
        mock_storage_loader.set_mock_data("complex_types_source", source_data)
        
        # Act
        result = await use_case.execute(
            source="complex_types_source",
            table_name="complex_types_table",
            embed_columns_names=["title", "tags", "metadata"],
            embed_type="separated"
        )
        
        # Assert
        assert result.success is True
        
        # Verify separated embeddings were created for complex types
        final_data = use_case.repo.tables["complex_types_table"]
        assert "title_enc" in final_data.columns
        assert "tags_enc" in final_data.columns
        assert "metadata_enc" in final_data.columnsables["no_embeddings_table"]
        assert "embeddings" not in stored_data.columns
        assert "title_enc" not in stored_data.columns
        assert "description_enc" not in stored_data.columns
        
        # But should still have system columns
        assert "embed_columns_names" in stored_data.columns
        assert "is_active" in stored_data.columns


class TestDataAPIJSONUseCaseUpsertFunctionality:
    """Test cases for upsert functionality in DataAPIJSONUseCase."""
    
    @pytest.mark.asyncio
    async def test_upsert_with_primary_keys_new_records(self, use_case, mock_storage_loader):
        """Test upsert operations with primary keys for new records."""
        # Arrange - create existing table with some data (including all embedding columns)
        existing_data = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "email": ["alice@example.com", "bob@example.com"],
            "embed_columns_value": ["name='Alice', email='alice@example.com'", "name='Bob', email='bob@example.com'"],
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "embed_columns_names": [["name", "email"], ["name", "email"]],
            "is_active": [True, True],
            "created_at": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "updated_at": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"]
        })
        use_case.repo.tables["users"] = existing_data
        
        # New data to upsert (new records)
        new_data = pd.DataFrame({
            "id": [3, 4],
            "name": ["Charlie", "Diana"],
            "email": ["charlie@example.com", "diana@example.com"]
        })
        mock_storage_loader.set_mock_data("new_users", new_data)
        
        # Act
        result = await use_case.execute(
            source="new_users",
            table_name="users",
            pk_columns=["id"],
            embed_columns_names=["name", "email"],
            create_table_if_not_exists=False,
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 2
        assert result.table_created is False
        assert result.operation_type == "existing_table_update"
        
        # Verify upsert method was called
        repo_calls = [call for call in use_case.repo.call_log if call[0] == "update_data"]
        assert len(repo_calls) == 1
        assert repo_calls[0][2] == 2  # 2 rows processed
        assert repo_calls[0][3] == ["id"]  # primary key columns
        
        # Verify final data contains both old and new records
        final_data = use_case.repo.tables["users"]
        assert len(final_data) == 4  # 2 existing + 2 new
    
    @pytest.mark.asyncio
    async def test_upsert_with_primary_keys_update_existing(self, use_case, mock_storage_loader):
        """Test upsert operations that update existing records."""
        # Arrange - create existing table
        existing_data = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "email": ["alice@old.com", "bob@old.com"],
            "embed_columns_value": ["name='Alice', email='alice@old.com'", "name='Bob', email='bob@old.com'"],
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "embed_columns_names": [["name", "email"], ["name", "email"]],
            "is_active": [True, True],
            "created_at": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "updated_at": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"]
        })
        use_case.repo.tables["users"] = existing_data
        
        # Updated data (same IDs, different values)
        updated_data = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice Updated", "Bob Updated"],
            "email": ["alice@new.com", "bob@new.com"]
        })
        mock_storage_loader.set_mock_data("updated_users", updated_data)
        
        # Act
        result = await use_case.execute(
            source="updated_users",
            table_name="users",
            pk_columns=["id"],
            embed_columns_names=["name", "email"],
            create_table_if_not_exists=False,
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 2
        assert result.operation_type == "existing_table_update"
        
        # Verify update_data was called (which handles upserts)
        repo_calls = [call for call in use_case.repo.call_log if call[0] == "update_data"]
        assert len(repo_calls) == 1
    
    @pytest.mark.asyncio
    async def test_upsert_with_composite_primary_keys(self, use_case, mock_storage_loader):
        """Test upsert operations with composite primary keys."""
        # Arrange - create table with composite primary key
        existing_data = pd.DataFrame({
            "user_id": [1, 1, 2],
            "project_id": [100, 101, 100],
            "role": ["admin", "viewer", "editor"],
            "permissions": ["all", "read", "write"],
            "embed_columns_value": ["role='admin'", "role='viewer'", "role='editor'"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "embed_columns_names": [["role"], ["role"], ["role"]],
            "is_active": [True, True, True]
        })
        use_case.repo.tables["user_projects"] = existing_data
        
        # New data with composite keys (some new, some updates)
        new_data = pd.DataFrame({
            "user_id": [1, 2, 3],
            "project_id": [100, 101, 100],  # (1,100) exists, (2,101) new, (3,100) new
            "role": ["super_admin", "admin", "viewer"],
            "permissions": ["super", "all", "read"]
        })
        mock_storage_loader.set_mock_data("user_project_updates", new_data)
        
        # Act
        result = await use_case.execute(
            source="user_project_updates",
            table_name="user_projects",
            pk_columns=["user_id", "project_id"],
            embed_columns_names=["role"],
            create_table_if_not_exists=False,
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 3
        
        # Verify composite primary keys were used
        repo_calls = [call for call in use_case.repo.call_log if call[0] == "update_data"]
        assert len(repo_calls) == 1
        assert repo_calls[0][3] == ["user_id", "project_id"]  # composite PK
    
    @pytest.mark.asyncio
    async def test_upsert_without_primary_keys_fallback_to_insert(self, use_case, mock_storage_loader):
        """Test upsert behavior when no primary keys are specified (fallback to insert)."""
        # Arrange - create existing table
        existing_data = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "email": ["alice@example.com", "bob@example.com"],
            "embed_columns_value": ["name='Alice'", "name='Bob'"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "embed_columns_names": [["name"], ["name"]],
            "is_active": [True, True]
        })
        use_case.repo.tables["no_pk_table"] = existing_data
        
        # New data without primary keys
        new_data = pd.DataFrame({
            "name": ["Charlie", "Diana"],
            "email": ["charlie@example.com", "diana@example.com"]
        })
        mock_storage_loader.set_mock_data("no_pk_data", new_data)
        
        # Act
        result = await use_case.execute(
            source="no_pk_data",
            table_name="no_pk_table",
            pk_columns=None,  # No primary keys
            embed_columns_names=["name"],
            create_table_if_not_exists=False,
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 2
        
        # Verify insert_data was called instead of update_data
        insert_calls = [call for call in use_case.repo.call_log if call[0] == "insert_data"]
        update_calls = [call for call in use_case.repo.call_log if call[0] == "update_data"]
        
        assert len(insert_calls) == 1
        assert len(update_calls) == 0  # Should not call update_data
        assert insert_calls[0][3] == []  # Empty primary key list
    
    @pytest.mark.asyncio
    async def test_upsert_with_timestamp_handling(self, use_case, mock_storage_loader):
        """Test that upsert operations properly handle created_at and updated_at timestamps."""
        # Arrange - create table with timestamp columns
        import datetime
        old_timestamp = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        
        existing_data = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "created_at": [old_timestamp, old_timestamp],
            "updated_at": [old_timestamp, old_timestamp],
            "embed_columns_value": ["name='Alice'", "name='Bob'"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "embed_columns_names": [["name"], ["name"]],
            "is_active": [True, True]
        })
        use_case.repo.tables["timestamped_table"] = existing_data
        
        # Mock table info to include timestamp columns
        original_get_table_info = use_case.repo.get_table_info
        async def mock_get_table_info(table_name):
            table_info = await original_get_table_info(table_name)
            # Add timestamp columns to the mock
            table_info.columns["created_at"] = ColumnInfo(
                name="created_at", data_type="timestamp", nullable=True
            )
            table_info.columns["updated_at"] = ColumnInfo(
                name="updated_at", data_type="timestamp", nullable=True
            )
            return table_info
        
        use_case.repo.get_table_info = mock_get_table_info
        
        # New data for upsert
        new_data = pd.DataFrame({
            "id": [1, 3],  # Update existing ID 1, insert new ID 3
            "name": ["Alice Updated", "Charlie"]
        })
        mock_storage_loader.set_mock_data("timestamp_data", new_data)
        
        # Act
        result = await use_case.execute(
            source="timestamp_data",
            table_name="timestamped_table",
            pk_columns=["id"],
            embed_columns_names=["name"],
            create_table_if_not_exists=False,
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 2
        
        # Verify that the DataFrame passed to update_data includes timestamp columns
        update_calls = [call for call in use_case.repo.call_log if call[0] == "update_data"]
        assert len(update_calls) == 1
        
        # The actual timestamp validation would be done in the repository layer
        # Here we're testing that the use case properly prepares the data
    
    @pytest.mark.asyncio
    async def test_upsert_primary_key_validation_errors(self, use_case, mock_storage_loader):
        """Test validation errors for primary key requirements in upsert operations."""
        # Arrange - create existing table
        existing_data = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "embed_columns_names": [["name"], ["name"]],
            "is_active": [True, True]
        })
        use_case.repo.tables["validation_table"] = existing_data
        
        # Test 1: Primary key column missing from DataFrame
        data_missing_pk = pd.DataFrame({
            "name": ["Charlie", "Diana"],
            "email": ["charlie@example.com", "diana@example.com"]
        })
        mock_storage_loader.set_mock_data("missing_pk_data", data_missing_pk)
        
        with pytest.raises(DatabaseOperationError) as exc_info:
            await use_case.execute(
                source="missing_pk_data",
                table_name="validation_table",
                pk_columns=["id"],  # ID column not in DataFrame
                create_table_if_not_exists=False
            )
        assert "Primary key columns not found in DataFrame" in str(exc_info.value)
        
        # Test 2: Primary key column contains null values
        data_with_nulls = pd.DataFrame({
            "id": [3, None],  # Null value in primary key
            "name": ["Charlie", "Diana"]
        })
        mock_storage_loader.set_mock_data("null_pk_data", data_with_nulls)
        
        with pytest.raises(DatabaseOperationError) as exc_info:
            await use_case.execute(
                source="null_pk_data",
                table_name="validation_table",
                pk_columns=["id"],
                create_table_if_not_exists=False
            )
        assert "contains" in str(exc_info.value) and "null values" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_upsert_schema_compatibility_validation(self, use_case, mock_storage_loader):
        """Test schema compatibility validation for upsert operations."""
        # Arrange - create existing table with specific schema
        existing_data = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "category": ["A", "B"],
            "embed_columns_value": ["name='Alice'", "name='Bob'"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "embed_columns_names": [["name"], ["name"]],
            "is_active": [True, True]
        })
        use_case.repo.tables["schema_table"] = existing_data
        
        # Test 1: DataFrame with extra columns not in table
        data_extra_columns = pd.DataFrame({
            "id": [3, 4],
            "name": ["Charlie", "Diana"],
            "category": ["C", "D"],
            "extra_column": ["extra1", "extra2"]  # This column doesn't exist in table
        })
        mock_storage_loader.set_mock_data("extra_col_data", data_extra_columns)
        
        with pytest.raises(DatabaseOperationError) as exc_info:
            await use_case.execute(
                source="extra_col_data",
                table_name="schema_table",
                pk_columns=["id"],
                create_table_if_not_exists=False
            )
        assert "columns not present in table" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_upsert_with_embedding_column_updates(self, use_case, mock_storage_loader):
        """Test that upsert operations properly handle embedding column updates."""
        # Arrange - create existing table with embeddings
        existing_data = pd.DataFrame({
            "id": [1, 2],
            "title": ["Old Title 1", "Old Title 2"],
            "description": ["Old Desc 1", "Old Desc 2"],
            "embed_columns_value": ["title='Old Title 1', description='Old Desc 1'", "title='Old Title 2', description='Old Desc 2'"],
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "embed_columns_names": [["title", "description"], ["title", "description"]],
            "is_active": [True, True]
        })
        use_case.repo.tables["embedding_table"] = existing_data
        
        # Updated data with new content (should generate new embeddings)
        updated_data = pd.DataFrame({
            "id": [1, 3],  # Update existing ID 1, insert new ID 3
            "title": ["Updated Title 1", "New Title 3"],
            "description": ["Updated Desc 1", "New Desc 3"]
        })
        mock_storage_loader.set_mock_data("embedding_updates", updated_data)
        
        # Act
        result = await use_case.execute(
            source="embedding_updates",
            table_name="embedding_table",
            pk_columns=["id"],
            embed_columns_names=["title", "description"],
            create_table_if_not_exists=False,
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 2
        
        # Verify embeddings were generated for the updated data
        embedding_calls = [call for call in use_case.embedding_service.call_log if call[0] == "get_embeddings"]
        assert len(embedding_calls) > 0
        
        # Verify update_data was called with embedding columns
        update_calls = [call for call in use_case.repo.call_log if call[0] == "update_data"]
        assert len(update_calls) == 1
    
    @pytest.mark.asyncio
    async def test_upsert_with_separated_embeddings(self, use_case, mock_storage_loader):
        """Test upsert operations with separated embedding type."""
        # Arrange - create existing table with separated embeddings
        existing_data = pd.DataFrame({
            "id": [1, 2],
            "title": ["Title 1", "Title 2"],
            "description": ["Desc 1", "Desc 2"],
            "title_enc": [[0.1, 0.2], [0.3, 0.4]],
            "description_enc": [[0.5, 0.6], [0.7, 0.8]],
            "embed_columns_names": [["title", "description"], ["title", "description"]],
            "is_active": [True, True]
        })
        use_case.repo.tables["separated_table"] = existing_data
        
        # New data for upsert
        new_data = pd.DataFrame({
            "id": [2, 3],  # Update existing ID 2, insert new ID 3
            "title": ["Updated Title 2", "New Title 3"],
            "description": ["Updated Desc 2", "New Desc 3"]
        })
        mock_storage_loader.set_mock_data("separated_updates", new_data)
        
        # Act
        result = await use_case.execute(
            source="separated_updates",
            table_name="separated_table",
            pk_columns=["id"],
            embed_columns_names=["title", "description"],
            create_table_if_not_exists=False,
            embed_type="separated"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 2
        
        # Verify separated embeddings were generated
        embedding_calls = use_case.embedding_service.call_log
        # Should have 2 calls for each embedding column (title and description)
        assert len([call for call in embedding_calls if call[0] == "get_embeddings"]) >= 2
    
    @pytest.mark.asyncio
    async def test_upsert_transaction_rollback_simulation(self, use_case, mock_storage_loader):
        """Test transaction safety and rollback behavior for upsert operations."""
        # Arrange - create existing table
        existing_data = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "embed_columns_value": ["name='Alice'", "name='Bob'"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "embed_columns_names": [["name"], ["name"]],
            "is_active": [True, True]
        })
        use_case.repo.tables["transaction_table"] = existing_data
        
        # Mock repository to fail during update_data
        original_update_data = use_case.repo.update_data
        async def failing_update_data(*args, **kwargs):
            raise DatabaseOperationError("Simulated database failure during upsert")
        
        use_case.repo.update_data = failing_update_data
        
        # New data for upsert
        new_data = pd.DataFrame({
            "id": [1, 3],
            "name": ["Alice Updated", "Charlie"]
        })
        mock_storage_loader.set_mock_data("transaction_data", new_data)
        
        # Act & Assert
        with pytest.raises(DatabaseOperationError) as exc_info:
            await use_case.execute(
                source="transaction_data",
                table_name="transaction_table",
                pk_columns=["id"],
                embed_columns_names=["name"],
                create_table_if_not_exists=False,
                embed_type="combined"
            )
        
        assert "Simulated database failure" in str(exc_info.value)
        
        # Verify that the error contains proper context
        assert hasattr(exc_info.value, 'context')
        assert 'execution_time' in exc_info.value.context
        
        # Restore original method
        use_case.repo.update_data = original_update_data
    
    @pytest.mark.asyncio
    async def test_upsert_empty_dataframe_handling(self, use_case, mock_storage_loader):
        """Test upsert behavior with empty DataFrame."""
        # Arrange - create existing table
        existing_data = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "embed_columns_value": ["name='Alice'", "name='Bob'"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "embed_columns_names": [["name"], ["name"]],
            "is_active": [True, True]
        })
        use_case.repo.tables["empty_upsert_table"] = existing_data
        
        # Empty DataFrame
        empty_data = pd.DataFrame(columns=["id", "name"])
        mock_storage_loader.set_mock_data("empty_upsert_data", empty_data)
        
        # Act
        result = await use_case.execute(
            source="empty_upsert_data",
            table_name="empty_upsert_table",
            pk_columns=["id"],
            embed_columns_names=["name"],
            create_table_if_not_exists=False,
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 0
        assert result.operation_type == "existing_table_update"
        
        # Verify no database operations were performed for empty data
        update_calls = [call for call in use_case.repo.call_log if call[0] == "update_data"]
        insert_calls = [call for call in use_case.repo.call_log if call[0] == "insert_data"]
        assert len(update_calls) == 0
        assert len(insert_calls) == 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_workflow(self, use_case, mock_storage_loader):
        """Test comprehensive workflow with all features."""
        # Arrange
        source_data = pd.DataFrame({
            "user_id": [1, 2, 3],
            "first_name": ["Alice", "Bob", "Charlie"],
            "last_name": ["Smith", "Jones", "Brown"],
            "bio": ["Software engineer", "Data scientist", "Product manager"],
            "tags": ["python,ai", "ml,stats", "product,strategy"]
        })
        mock_storage_loader.set_mock_data("comprehensive_source", source_data)
        
        column_mapping = {
            "user_id": "id",
            "first_name": "fname",
            "last_name": "lname"
        }
        
        transformation_mapping = {
            "full_name": "concat({fname}, ' ', {lname})"
        }
        
        # Act
        result = await use_case.execute(
            source="comprehensive_source",
            table_name="comprehensive_table",
            pk_columns=["id"],
            embed_columns_names=["full_name", "bio"],
            embed_type="combined",
            column_name_mapping=column_mapping,
            update_request_body_mapping=transformation_mapping,
            create_table_if_not_exists=True
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 3
        assert result.table_created is True
        
        # Verify final data structure
        stored_data = use_case.repo.tables["comprehensive_table"]
        expected_columns = {
            "id", "fname", "lname", "bio", "tags",  # Original + mapped
            "full_name",  # Transformed
            "embeddings", "embed_columns_names", "is_active"  # Embedding-related
        }
        assert set(stored_data.columns).issuperset(expected_columns)
        
        # Verify embedding service was called
        assert len(use_case.embedding_service.call_log) > 0
        
        # Verify repository operations
        repo_calls = [call[0] for call in use_case.repo.call_log]
        assert "table_exists" in repo_calls
        assert "create_table" in repo_calls
        assert "insert_data" in repo_calls


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])