"""
Examples demonstrating the new configuration system for embedding providers and vector stores.
This shows how to use custom configurations or rely on defaults.
"""

from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
from dataload.application.services.embedding.bedrock_provider import BedrockEmbeddingProvider
from dataload.application.services.embedding.sentence_transformers_provider import SentenceTransformersProvider
from dataload.application.services.embedding.openai_provider import OpenAIEmbeddingProvider
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from dataload.infrastructure.vector_stores.faiss_store import FaissVectorStore
from dataload.infrastructure.db.db_connection import DBConnection


def example_embedding_providers():
    """Examples of using embedding providers with custom configurations."""
    
    print("=== Embedding Provider Configuration Examples ===\n")
    
    # 1. Using default configuration (no config passed)
    print("1. Default Gemini Provider:")
    gemini_default = GeminiEmbeddingProvider()
    print(f"   Model: {gemini_default.config.model}")
    print(f"   Dimension: {gemini_default.config.dimension}")
    print()
    
    # 2. Using custom configuration
    print("2. Custom Gemini Provider:")
    gemini_custom = GeminiEmbeddingProvider({
        "model": "text-embedding-004",
        "dimension": 768,
        "task_type": "RETRIEVAL_DOCUMENT"
    })
    print(f"   Model: {gemini_custom.config.model}")
    print(f"   Dimension: {gemini_custom.config.dimension}")
    print(f"   Task Type: {gemini_custom.config.task_type}")
    print()
    
    # 3. Partial configuration (missing values use defaults)
    print("3. Partial Bedrock Provider:")
    bedrock_partial = BedrockEmbeddingProvider({
        "dimension": 1536  # Only specify dimension, other values use defaults
    })
    print(f"   Model ID: {bedrock_partial.config.model_id}")
    print(f"   Dimension: {bedrock_partial.config.dimension}")
    print(f"   Region: {bedrock_partial.config.region}")
    print()
    
    # 4. SentenceTransformers with custom model
    print("4. Custom SentenceTransformers Provider:")
    st_custom = SentenceTransformersProvider({
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "device": "cpu",
        "normalize_embeddings": True
    })
    print(f"   Model: {st_custom.config.model_name}")
    print(f"   Dimension: {st_custom.config.dimension}")
    print(f"   Device: {st_custom.config.device}")
    print()
    
    # 5. OpenAI with custom dimensions
    print("5. Custom OpenAI Provider:")
    openai_custom = OpenAIEmbeddingProvider({
        "model": "text-embedding-3-large",
        "dimension": 3072
    })
    print(f"   Model: {openai_custom.config.model}")
    print(f"   Dimension: {openai_custom.config.dimension}")
    print()


def example_vector_stores():
    """Examples of using vector stores with custom configurations."""
    
    print("=== Vector Store Configuration Examples ===\n")
    
    # 1. PostgreSQL with custom configuration
    print("1. Custom PostgreSQL Repository:")
    # Note: You would need a real DBConnection for this to work
    # db_conn = DBConnection(...)
    # postgres_custom = PostgresDataRepository(db_conn, {
    #     "dimension": 1536,
    #     "index_type": "hnsw",
    #     "distance_metric": "cosine",
    #     "hnsw_m": 32,
    #     "hnsw_ef_construction": 128
    # })
    print("   Configuration: dimension=1536, index_type=hnsw, distance_metric=cosine")
    print()
    
    # 2. ChromaDB with custom configuration
    print("2. Custom ChromaDB Store:")
    chroma_custom = ChromaVectorStore(
        mode="persistent",
        config={
            "dimension": 768,
            "distance_metric": "cosine",
            "persist_directory": "./custom_chroma_db"
        }
    )
    print(f"   Dimension: {chroma_custom.config.dimension}")
    print(f"   Distance Metric: {chroma_custom.config.distance_metric}")
    print(f"   Persist Directory: {chroma_custom.config.persist_directory}")
    print()
    
    # 3. FAISS with custom configuration
    print("3. Custom FAISS Store:")
    faiss_custom = FaissVectorStore(
        config={
            "dimension": 1024,
            "distance_metric": "dot_product",
            "faiss_index_type": "IndexFlatIP"
        }
    )
    print(f"   Dimension: {faiss_custom.config.dimension}")
    print(f"   Distance Metric: {faiss_custom.config.distance_metric}")
    print(f"   Index Type: {faiss_custom.config.faiss_index_type}")
    print()


def example_complete_workflow():
    """Example showing a complete workflow with custom configurations."""
    
    print("=== Complete Workflow Example ===\n")
    
    # 1. Create embedding provider with custom config
    embedding_provider = SentenceTransformersProvider({
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "normalize_embeddings": True
    })
    
    # 2. Create vector store with matching dimension
    vector_store = ChromaVectorStore(
        mode="in-memory",
        config={
            "dimension": 384,  # Match embedding dimension
            "distance_metric": "cosine"
        }
    )
    
    print(f"Embedding Provider Dimension: {embedding_provider.get_dimension()}")
    print(f"Vector Store Dimension: {vector_store.config.dimension}")
    print("✓ Dimensions match - ready for data processing!")
    print()


def example_migration_scenarios():
    """Examples showing how to migrate from old to new configuration system."""
    
    print("=== Migration Examples ===\n")
    
    print("OLD WAY:")
    print("embedding = SentenceTransformersProvider('all-MiniLM-L6-v2')")
    print()
    
    print("NEW WAY (backward compatible):")
    print("embedding = SentenceTransformersProvider()  # Uses defaults")
    print("# OR")
    print("embedding = SentenceTransformersProvider({")
    print("    'model_name': 'all-MiniLM-L6-v2',")
    print("    'dimension': 384")
    print("})")
    print()
    
    print("ADVANCED CONFIGURATION:")
    print("embedding = SentenceTransformersProvider({")
    print("    'model_name': 'sentence-transformers/all-mpnet-base-v2',")
    print("    'dimension': 768,")
    print("    'device': 'cuda',")
    print("    'normalize_embeddings': True")
    print("})")
    print()


if __name__ == "__main__":
    example_embedding_providers()
    example_vector_stores()
    example_complete_workflow()
    example_migration_scenarios()