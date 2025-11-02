from abc import ABC, abstractmethod
from typing import List


class EmbeddingProviderInterface(ABC):
    """Abstract interface for all Embedding Providers."""

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.
        NOTE: Must be synchronous or use a batching/throttling wrapper in implementation.
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Returns the dimension of the embeddings produced by this provider.
        """
        pass
