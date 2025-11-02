from typing import List, Optional, Dict, Any
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface
from dataload.config import logger
from dataload.domain.entities import EmbeddingError
from dataload.embedding_config import SentenceTransformersConfig, create_embedding_config


class SentenceTransformersProvider(EmbeddingProviderInterface):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbeddingError(
                "The 'sentence-transformers' extra is required to use SentenceTransformersProvider. "
                "Install with: pip install vector-dataloader[sentence-transformers]"
            )
        
        # Initialize configuration with defaults
        self.config: SentenceTransformersConfig = create_embedding_config("sentence_transformers", config)
        
        # Initialize model with configuration
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device
        )
        
        # Update dimension from model if not provided in config
        if self.config.dimension is None:
            self.config.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Initialized SentenceTransformers provider with model: {self.config.model_name}, dimension: {self.config.dimension}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize_embeddings
            ).tolist()
            
            logger.info(f"Generated {len(embeddings)} embeddings with SentenceTransformers using model {self.config.model_name}")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise EmbeddingError(f"Embedding failed: {e}")
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.dimension
