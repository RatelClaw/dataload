"""
Mock Embedding Provider for Examples

This module provides a simple mock embedding provider that can be used
for testing and examples when you don't have access to real embedding services.
"""

import hashlib
import math
from typing import List
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface


class MockEmbeddingProvider(EmbeddingProviderInterface):
    """
    Mock embedding provider that generates deterministic embeddings.
    
    This provider creates embeddings based on text hashing, which means:
    - Same text always produces the same embedding
    - Different texts produce different embeddings
    - Embeddings have consistent dimensionality
    - No external API calls required
    
    Useful for:
    - Testing and development
    - Examples and demonstrations
    - When real embedding services are not available
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the mock embedding provider.
        
        Args:
            embedding_dim: Dimension of the generated embeddings (default: 384)
        """
        self.embedding_dim = embedding_dim
        self.call_count = 0
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate mock embeddings for a list of texts.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors (lists of floats)
        """
        self.call_count += 1
        embeddings = []
        
        for text in texts:
            # Generate deterministic embedding based on text hash
            embedding = self._generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding vector for text.
        
        Uses SHA-256 hash to create deterministic, distributed values.
        
        Args:
            text: Input text string
            
        Returns:
            List of float values representing the embedding
        """
        # Create hash of the text
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        # Convert hash to numbers and normalize
        embedding = []
        for i in range(self.embedding_dim):
            # Use different parts of the hash for each dimension
            hash_segment = text_hash[(i * 2) % len(text_hash):(i * 2 + 8) % len(text_hash)]
            if len(hash_segment) < 8:
                hash_segment = (text_hash + text_hash)[i * 2:i * 2 + 8]
            
            # Convert hex to int and normalize to [-1, 1]
            value = int(hash_segment, 16) / (16**8 / 2) - 1
            
            # Add some variation based on position
            value += math.sin(i * 0.1) * 0.1
            
            # Clamp to [-1, 1] range
            value = max(-1, min(1, value))
            
            embedding.append(value)
        
        return embedding


class SimpleEmbeddingProvider(EmbeddingProviderInterface):
    """
    Very simple embedding provider for basic testing.
    
    Creates embeddings based on text length and character frequencies.
    Less sophisticated than MockEmbeddingProvider but even simpler.
    """
    
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize the simple embedding provider.
        
        Args:
            embedding_dim: Dimension of the generated embeddings (default: 128)
        """
        self.embedding_dim = embedding_dim
        self.call_count = 0
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate simple embeddings based on text characteristics."""
        self.call_count += 1
        embeddings = []
        
        for text in texts:
            embedding = self._simple_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def _simple_embedding(self, text: str) -> List[float]:
        """Generate simple embedding based on text characteristics."""
        text = text.lower()
        embedding = []
        
        for i in range(self.embedding_dim):
            if i < len(text):
                # Use character ASCII value
                value = ord(text[i]) / 128.0 - 1.0
            elif i == len(text):
                # Text length feature
                value = len(text) / 100.0 - 1.0
            elif i == len(text) + 1:
                # Vowel ratio
                vowels = sum(1 for c in text if c in 'aeiou')
                value = (vowels / max(1, len(text))) * 2 - 1
            elif i == len(text) + 2:
                # Average character value
                avg_char = sum(ord(c) for c in text) / max(1, len(text))
                value = avg_char / 128.0 - 1.0
            else:
                # Random-ish values based on text hash
                value = (hash(text + str(i)) % 1000) / 500.0 - 1.0
            
            # Clamp to [-1, 1]
            value = max(-1, min(1, value))
            embedding.append(value)
        
        return embedding


# For backward compatibility
GeminiEmbeddingProvider = MockEmbeddingProvider