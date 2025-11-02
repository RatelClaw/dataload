import os
from typing import List, Optional, Dict, Any
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface
from dataload.config import logger
from dataload.domain.entities import EmbeddingError
from dataload.embedding_config import OpenAIEmbeddingConfig, create_embedding_config


class OpenAIEmbeddingProvider(EmbeddingProviderInterface):
    """Embedding provider using OpenAI API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # --- LAZY IMPORT openai HERE ---
        try:
            from openai import OpenAI  # Import moved here!
        except ImportError:
            raise EmbeddingError(
                "The 'openai' extra is required to use OpenAIEmbeddingProvider. "
                "Install with: pip install vector-dataloader[openai]"
            )
        # -----------------------------

        # Initialize configuration with defaults
        self.config: OpenAIEmbeddingConfig = create_embedding_config("openai", config)
        
        # Use config values or fallback to environment
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY is not set in environment variables or config")

        # Use the lazily imported class
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI provider with model: {self.config.model}, dimension: {self.config.dimension}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI.
        """
        # This method is fine because it relies on self.client created in __init__
        try:
            # Prepare request parameters
            request_params = {
                "model": self.config.model,
                "input": texts
            }
            
            # Add dimensions parameter if specified and supported by model
            if self.config.dimension and self.config.model in ["text-embedding-3-small", "text-embedding-3-large"]:
                request_params["dimensions"] = self.config.dimension
            
            response = self.client.embeddings.create(**request_params)
            results = [item.embedding for item in response.data]

            # Validate dimension if configured
            if results and self.config.dimension:
                actual_dim = len(results[0])
                if actual_dim != self.config.dimension:
                    logger.warning(
                        f"OpenAI model {self.config.model} returned dimension {actual_dim}. Expected {self.config.dimension}."
                    )

            logger.info(f"Generated {len(results)} embeddings with OpenAI using model {self.config.model}")
            return results

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise EmbeddingError(f"OpenAI embedding failed: {e}")
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.dimension
