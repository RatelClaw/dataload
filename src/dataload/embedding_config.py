"""
Configuration classes for embedding providers and vector stores.
Provides a robust, production-grade configuration system with defaults and validation.
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os


@dataclass
class BaseEmbeddingConfig(ABC):
    """Base configuration class for all embedding providers."""
    
    dimension: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    @abstractmethod
    def validate(self) -> None:
        """Validate the configuration parameters."""
        pass
    
    def merge_with_defaults(self, defaults: Dict[str, Any]) -> 'BaseEmbeddingConfig':
        """Merge current config with defaults, keeping non-None values."""
        for key, default_value in defaults.items():
            if hasattr(self, key) and getattr(self, key) is None:
                setattr(self, key, default_value)
        return self


@dataclass
class GeminiEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for Gemini embedding provider."""
    
    model: Optional[str] = None
    api_key: Optional[str] = None
    task_type: Optional[str] = None
    dimension: Optional[int] = None
    
    def validate(self) -> None:
        """Validate Gemini configuration."""
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        
        valid_models = ["text-embedding-004", "text-embedding-005"]
        if self.model and self.model not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")


@dataclass
class BedrockEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for Bedrock embedding provider."""
    
    model_id: Optional[str] = None
    region: Optional[str] = None
    dimension: Optional[int] = None
    content_type: Optional[str] = None
    
    def validate(self) -> None:
        """Validate Bedrock configuration."""
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        
        valid_models = [
            "amazon.titan-embed-text-v1",
            "amazon.titan-embed-text-v2:0",
            "cohere.embed-english-v3",
            "cohere.embed-multilingual-v3"
        ]
        if self.model_id and self.model_id not in valid_models:
            raise ValueError(f"Model ID must be one of {valid_models}")


@dataclass
class SentenceTransformersConfig(BaseEmbeddingConfig):
    """Configuration for SentenceTransformers embedding provider."""
    
    model_name: Optional[str] = None
    dimension: Optional[int] = None
    device: Optional[str] = None
    normalize_embeddings: Optional[bool] = None
    
    def validate(self) -> None:
        """Validate SentenceTransformers configuration."""
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        
        if self.device and self.device not in ["cpu", "cuda", "auto"]:
            raise ValueError("Device must be 'cpu', 'cuda', or 'auto'")


@dataclass
class OpenAIEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for OpenAI embedding provider."""
    
    model: Optional[str] = None
    api_key: Optional[str] = None
    dimension: Optional[int] = None
    
    def validate(self) -> None:
        """Validate OpenAI configuration."""
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        
        valid_models = [
            "text-embedding-3-small",
            "text-embedding-3-large", 
            "text-embedding-ada-002"
        ]
        if self.model and self.model not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores and data repositories."""
    
    dimension: Optional[int] = None
    index_type: Optional[str] = None
    distance_metric: Optional[str] = None
    
    # PostgreSQL specific
    ivfflat_lists: Optional[int] = None
    hnsw_m: Optional[int] = None
    hnsw_ef_construction: Optional[int] = None
    
    # ChromaDB specific
    persist_directory: Optional[str] = None
    
    # FAISS specific
    faiss_index_type: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate vector store configuration."""
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        
        if self.distance_metric and self.distance_metric not in ["cosine", "euclidean", "dot_product"]:
            raise ValueError("Distance metric must be 'cosine', 'euclidean', or 'dot_product'")
        
        if self.index_type and self.index_type not in ["ivfflat", "hnsw"]:
            raise ValueError("Index type must be 'ivfflat' or 'hnsw'")


class EmbeddingConfigDefaults:
    """Default configurations for all embedding providers."""
    
    GEMINI = {
        "model": "text-embedding-004",
        "dimension": 768,
        "task_type": "SEMANTIC_SIMILARITY",
        "api_key": os.getenv("GOOGLE_API_KEY")
    }
    
    BEDROCK = {
        "model_id": "amazon.titan-embed-text-v2:0",
        "region": os.getenv("AWS_REGION", "us-east-1"),
        "dimension": 1024,
        "content_type": "application/json"
    }
    
    SENTENCE_TRANSFORMERS = {
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "device": "auto",
        "normalize_embeddings": True
    }
    
    OPENAI = {
        "model": "text-embedding-3-small",
        "dimension": 1536,
        "api_key": os.getenv("OPENAI_API_KEY")
    }


class VectorStoreConfigDefaults:
    """Default configurations for vector stores."""
    
    POSTGRES = {
        "dimension": 1024,
        "index_type": "ivfflat",
        "distance_metric": "cosine",
        "ivfflat_lists": 100,
        "hnsw_m": 16,
        "hnsw_ef_construction": 64
    }
    
    CHROMA = {
        "dimension": 1024,
        "distance_metric": "cosine",
        "persist_directory": "./chroma_db"
    }
    
    FAISS = {
        "dimension": 1024,
        "distance_metric": "euclidean",
        "faiss_index_type": "IndexFlatL2"
    }


def create_embedding_config(
    provider: str, 
    config: Optional[Dict[str, Any]] = None
) -> BaseEmbeddingConfig:
    """
    Factory function to create embedding configuration with defaults.
    
    Args:
        provider: Name of the embedding provider
        config: Optional configuration dictionary
        
    Returns:
        Configured embedding config object
    """
    config = config or {}
    
    if provider.lower() == "gemini":
        defaults = EmbeddingConfigDefaults.GEMINI
        config_obj = GeminiEmbeddingConfig(**config)
        return config_obj.merge_with_defaults(defaults)
    
    elif provider.lower() == "bedrock":
        defaults = EmbeddingConfigDefaults.BEDROCK
        config_obj = BedrockEmbeddingConfig(**config)
        return config_obj.merge_with_defaults(defaults)
    
    elif provider.lower() == "sentence_transformers":
        defaults = EmbeddingConfigDefaults.SENTENCE_TRANSFORMERS
        config_obj = SentenceTransformersConfig(**config)
        return config_obj.merge_with_defaults(defaults)
    
    elif provider.lower() == "openai":
        defaults = EmbeddingConfigDefaults.OPENAI
        config_obj = OpenAIEmbeddingConfig(**config)
        return config_obj.merge_with_defaults(defaults)
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def create_vector_store_config(
    store_type: str,
    config: Optional[Dict[str, Any]] = None
) -> VectorStoreConfig:
    """
    Factory function to create vector store configuration with defaults.
    
    Args:
        store_type: Type of vector store (postgres, chroma, faiss)
        config: Optional configuration dictionary
        
    Returns:
        Configured vector store config object
    """
    config = config or {}
    
    if store_type.lower() == "postgres":
        defaults = VectorStoreConfigDefaults.POSTGRES
    elif store_type.lower() == "chroma":
        defaults = VectorStoreConfigDefaults.CHROMA
    elif store_type.lower() == "faiss":
        defaults = VectorStoreConfigDefaults.FAISS
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
    
    # Merge config with defaults
    merged_config = {**defaults, **config}
    return VectorStoreConfig(**merged_config)