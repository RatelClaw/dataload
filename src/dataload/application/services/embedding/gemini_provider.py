import os
from typing import List, Optional, Dict, Any
import numpy as np

from dataload.interfaces.embedding_provider import EmbeddingProviderInterface
from dataload.config import logger
from dataload.domain.entities import EmbeddingError
from dataload.embedding_config import GeminiEmbeddingConfig, create_embedding_config


class GeminiEmbeddingProvider(EmbeddingProviderInterface):
    """Embedding provider using Google Gemini (Generative AI)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # --- LAZY IMPORT google-genai HERE ---
        try:
            from google import genai
            from google.genai import (
                types,
            )  # Keep this import here for client instantiation
        except ImportError:
            raise EmbeddingError(
                "The 'gemini' extra is required to use GeminiEmbeddingProvider. "
                "Install with: pip install vector-dataloader[gemini]"
            )
        # ------------------------------------

        # Initialize configuration with defaults
        self.config: GeminiEmbeddingConfig = create_embedding_config("gemini", config)
        
        # Use config values or fallback to environment
        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EmbeddingError("GOOGLE_API_KEY is not set in environment variables or config")
        
        self.client = genai.Client(api_key=api_key)
        logger.info(f"Initialized Gemini provider with model: {self.config.model}, dimension: {self.config.dimension}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Gemini.
        """
        try:
            from google.genai import types
        except ImportError:
            # Should not happen, but a safe guard
            raise EmbeddingError("Gemini dependencies failed to load.")

        try:
            resp = self.client.models.embed_content(
                model=self.config.model,
                contents=texts,
                config=types.EmbedContentConfig(task_type=self.config.task_type),
            )

            embeddings = [e.values for e in resp.embeddings]

            # Validate dimension if configured
            if embeddings and self.config.dimension:
                actual_dim = len(embeddings[0])
                if actual_dim != self.config.dimension:
                    logger.warning(
                        f"Model {self.config.model} returned dimension {actual_dim}. Expected {self.config.dimension}."
                    )

            logger.info(f"Generated {len(embeddings)} embeddings with Gemini using model {self.config.model}")
            return embeddings

        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise EmbeddingError(f"Gemini embedding failed: {e}")
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.dimension
