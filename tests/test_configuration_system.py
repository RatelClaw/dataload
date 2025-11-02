"""
Tests for the new configuration system for embedding providers and vector stores.
"""

import pytest
from unittest.mock import Mock, patch
from dataload.config.embedding_config import (
    GeminiEmbeddingConfig,
    BedrockEmbeddingConfig,
    SentenceTransformersConfig,
    OpenAIEmbeddingConfig,
    VectorStoreConfig,
    create_embedding_config,
    create_vector_store_config,
    EmbeddingConfigDefaults,
    VectorStoreConfigDefaults
)


class TestEmbeddingConfigs:
    """Test embedding configuration classes."""
    
    def test_gemini_config_defaults(self):
        """Test Gemini configuration with defaults."""
        config = create_embedding_config("gemini")
        assert config.model == "text-embedding-004"
        assert config.dimension == 768
        assert config.task_type == "SEMANTIC_SIMILARITY"
    
    def test_gemini_config_custom(self):
        """Test Gemini configuration with custom values."""
        custom_config = {
            "model": "text-embedding-004",
            "dimension": 1024,
            "task_type": "RETRIEVAL_DOCUMENT"
        }
        config = create_embedding_config("gemini", custom_config)
        assert config.model == "text-embedding-004"
        assert config.dimension == 1024
        assert config.task_type == "RETRIEVAL_DOCUMENT"
    
    def test_gemini_config_partial(self):
        """Test Gemini configuration with partial custom values."""
        custom_config = {"dimension": 1024}
        config = create_embedding_config("gemini", custom_config)
        assert config.model == "text-embedding-004"  # Default
        assert config.dimension == 1024  # Custom
        assert config.task_type == "SEMANTIC_SIMILARITY"  # Default
    
    def test_bedrock_config_defaults(self):
        """Test Bedrock configuration with defaults."""
        config = create_embedding_config("bedrock")
        assert config.model_id == "amazon.titan-embed-text-v2:0"
        assert config.dimension == 1024
        assert config.content_type == "application/json"
    
    def test_sentence_transformers_config_defaults(self):
        """Test SentenceTransformers configuration with defaults."""
        config = create_embedding_config("sentence_transformers")
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.dimension == 384
        assert config.device == "auto"
        assert config.normalize_embeddings == True
    
    def test_openai_config_defaults(self):
        """Test OpenAI configuration with defaults."""
        config = create_embedding_config("openai")
        assert config.model == "text-embedding-3-small"
        assert config.dimension == 1536
    
    def test_invalid_provider(self):
        """Test error handling for invalid provider."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_config("invalid_provider")


class TestVectorStoreConfigs:
    """Test vector store configuration classes."""
    
    def test_postgres_config_defaults(self):
        """Test PostgreSQL configuration with defaults."""
        config = create_vector_store_config("postgres")
        assert config.dimension == 1024
        assert config.index_type == "ivfflat"
        assert config.distance_metric == "cosine"
        assert config.ivfflat_lists == 100
    
    def test_chroma_config_defaults(self):
        """Test ChromaDB configuration with defaults."""
        config = create_vector_store_config("chroma")
        assert config.dimension == 1024
        assert config.distance_metric == "cosine"
        assert config.persist_directory == "./chroma_db"
    
    def test_faiss_config_defaults(self):
        """Test FAISS configuration with defaults."""
        config = create_vector_store_config("faiss")
        assert config.dimension == 1024
        assert config.distance_metric == "euclidean"
        assert config.faiss_index_type == "IndexFlatL2"
    
    def test_vector_store_config_custom(self):
        """Test vector store configuration with custom values."""
        custom_config = {
            "dimension": 768,
            "index_type": "hnsw",
            "distance_metric": "dot_product"
        }
        config = create_vector_store_config("postgres", custom_config)
        assert config.dimension == 768
        assert config.index_type == "hnsw"
        assert config.distance_metric == "dot_product"
    
    def test_invalid_vector_store_type(self):
        """Test error handling for invalid vector store type."""
        with pytest.raises(ValueError, match="Unknown vector store type"):
            create_vector_store_config("invalid_store")


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_gemini_config_validation_invalid_dimension(self):
        """Test Gemini configuration validation for invalid dimension."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            GeminiEmbeddingConfig(dimension=-1)
    
    def test_gemini_config_validation_invalid_model(self):
        """Test Gemini configuration validation for invalid model."""
        with pytest.raises(ValueError, match="Model must be one of"):
            GeminiEmbeddingConfig(model="invalid-model")
    
    def test_bedrock_config_validation_invalid_model(self):
        """Test Bedrock configuration validation for invalid model."""
        with pytest.raises(ValueError, match="Model ID must be one of"):
            BedrockEmbeddingConfig(model_id="invalid-model")
    
    def test_sentence_transformers_config_validation_invalid_device(self):
        """Test SentenceTransformers configuration validation for invalid device."""
        with pytest.raises(ValueError, match="Device must be"):
            SentenceTransformersConfig(device="invalid-device")
    
    def test_vector_store_config_validation_invalid_dimension(self):
        """Test vector store configuration validation for invalid dimension."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            VectorStoreConfig(dimension=-1)
    
    def test_vector_store_config_validation_invalid_distance_metric(self):
        """Test vector store configuration validation for invalid distance metric."""
        with pytest.raises(ValueError, match="Distance metric must be"):
            VectorStoreConfig(distance_metric="invalid-metric")


class TestEmbeddingProviderIntegration:
    """Test embedding providers with configuration system."""
    
    @patch('dataload.application.services.embedding.gemini_provider.genai')
    def test_gemini_provider_with_config(self, mock_genai):
        """Test Gemini provider initialization with custom config."""
        from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
        
        # Mock the genai module
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        
        custom_config = {
            "model": "text-embedding-004",
            "dimension": 768,
            "api_key": "test-key"
        }
        
        provider = GeminiEmbeddingProvider(custom_config)
        
        assert provider.config.model == "text-embedding-004"
        assert provider.config.dimension == 768
        assert provider.get_dimension() == 768
        mock_genai.Client.assert_called_once_with(api_key="test-key")
    
    @patch('dataload.application.services.embedding.sentence_transformers_provider.SentenceTransformer')
    def test_sentence_transformers_provider_with_config(self, mock_st):
        """Test SentenceTransformers provider initialization with custom config."""
        from dataload.application.services.embedding.sentence_transformers_provider import SentenceTransformersProvider
        
        # Mock the SentenceTransformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        
        custom_config = {
            "model_name": "all-mpnet-base-v2",
            "device": "cpu",
            "normalize_embeddings": True
        }
        
        provider = SentenceTransformersProvider(custom_config)
        
        assert provider.config.model_name == "all-mpnet-base-v2"
        assert provider.config.device == "cpu"
        assert provider.config.normalize_embeddings == True
        assert provider.get_dimension() == 768
        mock_st.assert_called_once_with("all-mpnet-base-v2", device="cpu")


class TestVectorStoreIntegration:
    """Test vector stores with configuration system."""
    
    @patch('dataload.infrastructure.vector_stores.chroma_store.chromadb')
    def test_chroma_store_with_config(self, mock_chromadb):
        """Test ChromaDB store initialization with custom config."""
        from dataload.infrastructure.vector_stores.chroma_store import ChromaVectorStore
        
        # Mock chromadb
        mock_client = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.list_collections.return_value = []
        
        custom_config = {
            "dimension": 768,
            "distance_metric": "cosine",
            "persist_directory": "./custom_chroma"
        }
        
        store = ChromaVectorStore(config=custom_config)
        
        assert store.config.dimension == 768
        assert store.config.distance_metric == "cosine"
        assert store.config.persist_directory == "./custom_chroma"
    
    def test_faiss_store_with_config(self):
        """Test FAISS store initialization with custom config."""
        from dataload.infrastructure.vector_stores.faiss_store import FaissVectorStore
        
        custom_config = {
            "dimension": 512,
            "distance_metric": "dot_product",
            "faiss_index_type": "IndexFlatIP"
        }
        
        # Create a temporary directory for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FaissVectorStore(persistence_path=temp_dir, config=custom_config)
            
            assert store.config.dimension == 512
            assert store.config.distance_metric == "dot_product"
            assert store.config.faiss_index_type == "IndexFlatIP"


if __name__ == "__main__":
    pytest.main([__file__])