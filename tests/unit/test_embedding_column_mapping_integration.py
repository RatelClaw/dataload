"""
Unit tests for embedding column validation with column mapping integration.

This module tests the integration between column mapping and embedding generation,
ensuring that embed_columns_names refers to mapped column names and that proper
validation and error messages are provided.
"""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

from src.dataload.application.use_cases.data_api_json_use_case import DataAPIJSONUseCase
from src.dataload.domain.column_mapper import ColumnMapper
from src.dataload.domain.api_entities import ColumnMappingConfig, ValidationMode
from src.dataload.domain.entities import ValidationError, DataMoveError
from src.dataload.interfaces.data_move_repository import DataMoveRepositoryInterface
from src.dataload.interfaces.embedding_provider import EmbeddingProviderInterface
from src.dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader


class MockRepository:
    """Mock repository for testing."""
    
    def __init__(self):
        self.tables = {}
        self.call_log = []
    
    async def table_exists(self, table_name: str) -> bool:
        self.call_log.append(("table_exists", table_name))
        return table_name in self.tables
    
    async def create_table(self, table_name: str, df: pd.DataFrame, pk_columns, embed_type, embed_columns_names):
        self.call_log.append(("create_table", table_name, pk_columns, embed_type, embed_columns_names))
        self.tables[table_name] = df.copy()
        return {col: "text" for col in df.columns}
    
    async def insert_data(self, table_name: str, df: pd.DataFrame, pk_columns):
        self.call_log.append(("insert_data", table_name, len(df), pk_columns))
        if table_name in self.tables:
            self.tables[table_name] = pd.concat([self.tables[table_name], df], ignore_index=True)
        else:
            self.tables[table_name] = df.copy()


class MockEmbeddingService:
    """Mock embedding service for testing."""
    
    def __init__(self):
        self.call_log = []
    
    def get_embeddings(self, texts):
        self.call_log.append(("get_embeddings", len(texts)))
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockStorageLoader:
    """Mock storage loader for testing."""
    
    def __init__(self):
        self.mock_data = {}
        self.call_log = []
    
    def set_mock_data(self, source, data):
        self.mock_data[str(source)] = data
    
    def load_json(self, source, config=None):
        self.call_log.append(("load_json", str(source), config))
        
        if str(source) in self.mock_data:
            df = self.mock_data[str(source)].copy()
        else:
            df = pd.DataFrame({
                "api_field_1": ["value1", "value2"],
                "api_field_2": ["desc1", "desc2"],
                "api_field_3": ["cat1", "cat2"]
            })
        
        # Apply column mapping if provided
        if config and config.get('column_name_mapping'):
            df = df.rename(columns=config['column_name_mapping'])
        
        return df
    
    def validate_config(self, config):
        return []


class TestEmbeddingColumnMappingIntegration:
    """Test embedding column validation with column mapping."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = MockRepository()
        self.embedding_service = MockEmbeddingService()
        self.storage_loader = MockStorageLoader()
        self.use_case = DataAPIJSONUseCase(
            repo=self.repository,
            embedding_service=self.embedding_service,
            storage_loader=self.storage_loader
        )
        self.column_mapper = ColumnMapper()
    
    @pytest.mark.asyncio
    async def test_embedding_with_mapped_column_names_combined(self):
        """Test embedding generation with mapped column names - combined type."""
        # Arrange
        source_data = pd.DataFrame({
            "api_title": ["Title 1", "Title 2"],
            "api_description": ["Description 1", "Description 2"],
            "api_category": ["Category A", "Category B"]
        })
        self.storage_loader.set_mock_data("test_source", source_data)
        
        column_mapping = {
            "api_title": "title",
            "api_description": "description",
            "api_category": "category"
        }
        
        # Act - use mapped column names for embedding
        result = await self.use_case.execute(
            source="test_source",
            table_name="mapped_table",
            column_name_mapping=column_mapping,
            embed_columns_names=["title", "description"],  # Mapped names
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        assert result.rows_processed == 2
        
        # Verify the final table has correct columns
        final_data = self.repository.tables["mapped_table"]
        assert "title" in final_data.columns
        assert "description" in final_data.columns
        assert "category" in final_data.columns
        assert "embeddings" in final_data.columns
        assert "embed_columns_names" in final_data.columns
        
        # Verify embedding service was called
        assert len(self.embedding_service.call_log) > 0
    
    @pytest.mark.asyncio
    async def test_embedding_with_mapped_column_names_separated(self):
        """Test embedding generation with mapped column names - separated type."""
        # Arrange
        source_data = pd.DataFrame({
            "user_name": ["Alice", "Bob"],
            "user_email": ["alice@example.com", "bob@example.com"],
            "user_bio": ["Bio 1", "Bio 2"]
        })
        self.storage_loader.set_mock_data("user_source", source_data)
        
        column_mapping = {
            "user_name": "name",
            "user_email": "email",
            "user_bio": "bio"
        }
        
        # Act - use mapped column names for embedding
        result = await self.use_case.execute(
            source="user_source",
            table_name="users_table",
            column_name_mapping=column_mapping,
            embed_columns_names=["name", "bio"],  # Mapped names
            embed_type="separated"
        )
        
        # Assert
        assert result.success is True
        
        # Verify separated embedding columns were created
        final_data = self.repository.tables["users_table"]
        assert "name" in final_data.columns
        assert "email" in final_data.columns
        assert "bio" in final_data.columns
        assert "name_enc" in final_data.columns
        assert "bio_enc" in final_data.columns
        assert "embeddings" not in final_data.columns  # Should not have combined column
    
    @pytest.mark.asyncio
    async def test_embedding_validation_error_original_field_names(self):
        """Test validation error when using original API field names instead of mapped names."""
        # Arrange
        source_data = pd.DataFrame({
            "original_title": ["Title 1", "Title 2"],
            "original_desc": ["Description 1", "Description 2"]
        })
        self.storage_loader.set_mock_data("error_source", source_data)
        
        column_mapping = {
            "original_title": "title",
            "original_desc": "description"
        }
        
        # Act & Assert - use original field names (should fail)
        with pytest.raises(ValidationError) as exc_info:
            await self.use_case.execute(
                source="error_source",
                table_name="error_table",
                column_name_mapping=column_mapping,
                embed_columns_names=["original_title", "original_desc"],  # Original names (wrong)
                embed_type="combined"
            )
        
        error = exc_info.value
        error_str = str(error)
        assert "Embedding columns not found" in error_str
        assert "embed_columns_names should refer to mapped column names" in error_str
        assert "original_title" in error_str
        assert "should be 'title'" in error_str
    
    @pytest.mark.asyncio
    async def test_embedding_validation_error_nonexistent_columns(self):
        """Test validation error for completely nonexistent columns."""
        # Arrange
        source_data = pd.DataFrame({
            "field1": ["Value 1", "Value 2"],
            "field2": ["Value 3", "Value 4"]
        })
        self.storage_loader.set_mock_data("nonexistent_source", source_data)
        
        # Act & Assert - use nonexistent column names
        with pytest.raises(ValidationError) as exc_info:
            await self.use_case.execute(
                source="nonexistent_source",
                table_name="nonexistent_table",
                embed_columns_names=["nonexistent_column"],
                embed_type="combined"
            )
        
        error = exc_info.value
        assert "Embedding columns not found" in str(error)
        assert "nonexistent_column" in str(error)
    
    @pytest.mark.asyncio
    async def test_embedding_validation_with_suggestions(self):
        """Test that validation provides helpful suggestions for similar column names."""
        # Arrange
        source_data = pd.DataFrame({
            "title": ["Title 1", "Title 2"],
            "description": ["Description 1", "Description 2"],
            "category": ["Category A", "Category B"]
        })
        self.storage_loader.set_mock_data("suggestion_source", source_data)
        
        # Act & Assert - use slightly wrong column name
        with pytest.raises(ValidationError) as exc_info:
            await self.use_case.execute(
                source="suggestion_source",
                table_name="suggestion_table",
                embed_columns_names=["titel"],  # Typo in "title"
                embed_type="combined"
            )
        
        error = exc_info.value
        assert "Embedding columns not found" in str(error)
        # The error should include suggestions for similar columns
        assert "title" in str(error).lower()
    
    def test_column_mapper_validate_embedding_columns_with_mapping(self):
        """Test ColumnMapper's embedding validation with mapping."""
        # Arrange
        original_columns = ["api_name", "api_description", "api_category"]
        column_mapping = {
            "api_name": "name",
            "api_description": "description"
            # api_category not mapped (should remain as-is)
        }
        
        # Test case 1: Valid mapped column names
        embed_columns = ["name", "description"]
        result = self.column_mapper.validate_embedding_columns_with_mapping(
            embed_columns, original_columns, column_mapping
        )
        
        assert result["valid_columns"] == ["name", "description"]
        assert result["missing_columns"] == []
        assert len(result["suggestions"]) == 0
        
        # Test case 2: Using original field names (should suggest mapped names)
        embed_columns = ["api_name", "api_description"]
        result = self.column_mapper.validate_embedding_columns_with_mapping(
            embed_columns, original_columns, column_mapping
        )
        
        assert result["valid_columns"] == []
        assert result["missing_columns"] == ["api_name", "api_description"]
        assert "api_name" in result["suggestions"]
        assert result["suggestions"]["api_name"]["suggestion"] == "name"
        assert result["suggestions"]["api_name"]["type"] == "original_name_used"
    
    def test_column_mapper_validate_embedding_columns_case_insensitive(self):
        """Test case-insensitive embedding column validation."""
        # Arrange
        original_columns = ["API_NAME", "api_description"]
        column_mapping = {
            "API_NAME": "name",
            "api_description": "description"
        }
        
        # Test with case variations
        embed_columns = ["NAME", "Description"]
        result = self.column_mapper.validate_embedding_columns_with_mapping(
            embed_columns, original_columns, column_mapping, case_sensitive=False
        )
        
        assert "name" in result["valid_columns"]
        assert "description" in result["valid_columns"]
        assert len(result["warnings"]) > 0  # Should warn about case mismatches
    
    @pytest.mark.asyncio
    async def test_embedding_with_partial_mapping(self):
        """Test embedding when only some columns are mapped."""
        # Arrange
        source_data = pd.DataFrame({
            "api_title": ["Title 1", "Title 2"],
            "description": ["Description 1", "Description 2"],  # Not mapped
            "api_category": ["Category A", "Category B"]
        })
        self.storage_loader.set_mock_data("partial_source", source_data)
        
        column_mapping = {
            "api_title": "title",
            "api_category": "category"
            # description not mapped - should remain as-is
        }
        
        # Act - use mix of mapped and unmapped column names
        result = await self.use_case.execute(
            source="partial_source",
            table_name="partial_table",
            column_name_mapping=column_mapping,
            embed_columns_names=["title", "description"],  # Mixed: mapped + unmapped
            embed_type="combined"
        )
        
        # Assert
        assert result.success is True
        
        # Verify both mapped and unmapped columns exist
        final_data = self.repository.tables["partial_table"]
        assert "title" in final_data.columns  # Mapped
        assert "description" in final_data.columns  # Unmapped (original name)
        assert "category" in final_data.columns  # Mapped
    
    @pytest.mark.asyncio
    async def test_embedding_with_empty_column_mapping(self):
        """Test embedding when no column mapping is provided."""
        # Arrange
        source_data = pd.DataFrame({
            "title": ["Title 1", "Title 2"],
            "description": ["Description 1", "Description 2"]
        })
        self.storage_loader.set_mock_data("no_mapping_source", source_data)
        
        # Act - no column mapping
        result = await self.use_case.execute(
            source="no_mapping_source",
            table_name="no_mapping_table",
            embed_columns_names=["title", "description"],  # Original names
            embed_type="separated"
        )
        
        # Assert
        assert result.success is True
        
        # Verify original column names are preserved
        final_data = self.repository.tables["no_mapping_table"]
        assert "title" in final_data.columns
        assert "description" in final_data.columns
        assert "title_enc" in final_data.columns
        assert "description_enc" in final_data.columns
    
    @pytest.mark.asyncio
    async def test_embedding_error_handling_during_generation(self):
        """Test error handling when embedding generation fails."""
        # Arrange
        source_data = pd.DataFrame({
            "title": ["Title 1", "Title 2"],
            "description": ["Description 1", "Description 2"]
        })
        self.storage_loader.set_mock_data("error_gen_source", source_data)
        
        # Mock embedding service to raise an error
        def failing_get_embeddings(texts):
            raise Exception("Embedding service unavailable")
        
        self.embedding_service.get_embeddings = failing_get_embeddings
        
        # Act & Assert
        with pytest.raises(DataMoveError) as exc_info:
            await self.use_case.execute(
                source="error_gen_source",
                table_name="error_gen_table",
                embed_columns_names=["title", "description"],
                embed_type="combined"
            )
        
        error = exc_info.value
        assert "Embedding generation failed" in str(error)
        assert "Embedding service unavailable" in str(error)
    
    def test_format_embed_value_with_mapped_columns(self):
        """Test _format_embed_value method with various data types."""
        # Arrange
        row = pd.Series({
            "title": "Test Title",
            "description": "Test Description",
            "tags": ["tag1", "tag2", "tag3"],
            "metadata": {"key": "value"},
            "score": 95.5,
            "empty_field": None,
            "zero_value": 0
        })
        
        embed_columns = ["title", "description", "tags", "score", "empty_field", "zero_value"]
        
        # Act
        result = self.use_case._format_embed_value(row, embed_columns)
        
        # Assert
        assert "title='Test Title'" in result
        assert "description='Test Description'" in result
        assert "tags='tag1, tag2, tag3'" in result
        assert "score='95.5'" in result
        assert "empty_field=''" in result
        assert "zero_value='0'" in result
    
    def test_format_embed_value_missing_column_error(self):
        """Test _format_embed_value with missing column."""
        # Arrange
        row = pd.Series({"title": "Test Title"})
        embed_columns = ["title", "missing_column"]
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            self.use_case._format_embed_value(row, embed_columns)
        
        error = exc_info.value
        assert "Column 'missing_column' not found" in str(error)
    
    def test_suggest_similar_columns(self):
        """Test _suggest_similar_columns method."""
        # Arrange
        missing_columns = ["titel", "descripion", "nonexistent"]
        available_columns = {"title", "description", "category", "tags"}
        
        # Act
        suggestions = self.use_case._suggest_similar_columns(missing_columns, available_columns)
        
        # Assert
        assert "titel" in suggestions
        assert "title" in suggestions["titel"]
        assert "descripion" in suggestions
        assert "description" in suggestions["descripion"]
        # "nonexistent" might not have good suggestions
    
    def test_calculate_similarity(self):
        """Test _calculate_similarity method."""
        # Test exact match
        assert self.use_case._calculate_similarity("test", "test") == 1.0
        
        # Test completely different
        similarity = self.use_case._calculate_similarity("abc", "xyz")
        assert similarity < 0.5
        
        # Test similar strings
        similarity = self.use_case._calculate_similarity("title", "titel")
        assert similarity > 0.5
        
        # Test empty strings
        assert self.use_case._calculate_similarity("", "") == 1.0
        assert self.use_case._calculate_similarity("test", "") == 0.0


if __name__ == '__main__':
    pytest.main([__file__])