"""
Simple demonstration of the new configuration system.
This example shows how to use the new configuration features without complex setup.
"""

import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_embedding_configurations():
    """Demonstrate embedding provider configurations."""
    print("🚀 Embedding Provider Configuration Demo\n")
    
    # Import embedding providers
    try:
        from dataload.application.services.embedding.sentence_transformers_provider import SentenceTransformersProvider
        
        print("1. Default SentenceTransformers Configuration:")
        print("   embedding = SentenceTransformersProvider()")
        try:
            embedding_default = SentenceTransformersProvider()
            print(f"   ✓ Model: {embedding_default.config.model_name}")
            print(f"   ✓ Dimension: {embedding_default.config.dimension}")
            print(f"   ✓ Device: {embedding_default.config.device}")
        except Exception as e:
            print(f"   ⚠️  Note: {e}")
        print()
        
        print("2. Custom SentenceTransformers Configuration:")
        print("   embedding = SentenceTransformersProvider({")
        print("       'model_name': 'all-MiniLM-L6-v2',")
        print("       'dimension': 384,")
        print("       'device': 'cpu'")
        print("   })")
        try:
            embedding_custom = SentenceTransformersProvider({
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384,
                "device": "cpu",
                "normalize_embeddings": True
            })
            print(f"   ✓ Model: {embedding_custom.config.model_name}")
            print(f"   ✓ Dimension: {embedding_custom.config.dimension}")
            print(f"   ✓ Device: {embedding_custom.config.device}")
            print(f"   ✓ Normalize: {embedding_custom.config.normalize_embeddings}")
        except Exception as e:
            print(f"   ⚠️  Note: {e}")
        print()
        
    except ImportError as e:
        print(f"⚠️  SentenceTransformers not available: {e}")
        print("   Install with: pip install sentence-transformers")
        print()
    
    # Demonstrate other providers (without actually initializing them)
    print("3. Other Provider Configurations (examples):")
    print()
    
    print("   Gemini Provider:")
    print("   embedding = GeminiEmbeddingProvider({")
    print("       'model': 'text-embedding-004',")
    print("       'dimension': 768,")
    print("       'task_type': 'SEMANTIC_SIMILARITY'")
    print("   })")
    print()
    
    print("   OpenAI Provider:")
    print("   embedding = OpenAIEmbeddingProvider({")
    print("       'model': 'text-embedding-3-small',")
    print("       'dimension': 1536")
    print("   })")
    print()
    
    print("   Bedrock Provider:")
    print("   embedding = BedrockEmbeddingProvider({")
    print("       'model_id': 'amazon.titan-embed-text-v2:0',")
    print("       'dimension': 1024,")
    print("       'region': 'us-east-1'")
    print("   })")
    print()


def demo_vector_store_configurations():
    """Demonstrate vector store configurations."""
    print("🗄️  Vector Store Configuration Demo\n")
    
    try:
        from dataload.infrastructure.vector_stores.chroma_store import ChromaVectorStore
        
        print("1. Default ChromaDB Configuration:")
        print("   store = ChromaVectorStore()")
        try:
            store_default = ChromaVectorStore(mode="in-memory")
            print(f"   ✓ Dimension: {store_default.config.dimension}")
            print(f"   ✓ Distance Metric: {store_default.config.distance_metric}")
            print(f"   ✓ Mode: in-memory")
        except Exception as e:
            print(f"   ⚠️  Note: {e}")
        print()
        
        print("2. Custom ChromaDB Configuration:")
        print("   store = ChromaVectorStore(config={")
        print("       'dimension': 768,")
        print("       'distance_metric': 'cosine',")
        print("       'persist_directory': './my_chroma_db'")
        print("   })")
        try:
            store_custom = ChromaVectorStore(
                mode="in-memory",
                config={
                    "dimension": 768,
                    "distance_metric": "cosine",
                    "persist_directory": "./my_chroma_db"
                }
            )
            print(f"   ✓ Dimension: {store_custom.config.dimension}")
            print(f"   ✓ Distance Metric: {store_custom.config.distance_metric}")
            print(f"   ✓ Persist Directory: {store_custom.config.persist_directory}")
        except Exception as e:
            print(f"   ⚠️  Note: {e}")
        print()
        
    except ImportError as e:
        print(f"⚠️  ChromaDB not available: {e}")
        print("   Install with: pip install chromadb")
        print()
    
    # Demonstrate other stores (without actually initializing them)
    print("3. Other Store Configurations (examples):")
    print()
    
    print("   PostgreSQL Repository:")
    print("   repo = PostgresDataRepository(db_conn, {")
    print("       'dimension': 1024,")
    print("       'index_type': 'ivfflat',")
    print("       'distance_metric': 'cosine',")
    print("       'ivfflat_lists': 100")
    print("   })")
    print()
    
    print("   FAISS Store:")
    print("   store = FaissVectorStore(config={")
    print("       'dimension': 512,")
    print("       'distance_metric': 'euclidean',")
    print("       'faiss_index_type': 'IndexFlatL2'")
    print("   })")
    print()


def demo_configuration_validation():
    """Demonstrate configuration validation."""
    print("✅ Configuration Validation Demo\n")
    
    try:
        from dataload.embedding_config import GeminiEmbeddingConfig, VectorStoreConfig
        
        print("1. Valid Configuration:")
        try:
            config = GeminiEmbeddingConfig(
                model="text-embedding-004",
                dimension=768,
                task_type="SEMANTIC_SIMILARITY"
            )
            print(f"   ✓ Gemini config created successfully")
            print(f"   ✓ Model: {config.model}")
            print(f"   ✓ Dimension: {config.dimension}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()
        
        print("2. Invalid Configuration (negative dimension):")
        try:
            config = GeminiEmbeddingConfig(dimension=-1)
            print(f"   ❌ This should not succeed!")
        except ValueError as e:
            print(f"   ✓ Validation caught error: {e}")
        print()
        
        print("3. Invalid Configuration (invalid model):")
        try:
            config = GeminiEmbeddingConfig(model="invalid-model")
            print(f"   ❌ This should not succeed!")
        except ValueError as e:
            print(f"   ✓ Validation caught error: {e}")
        print()
        
        print("4. Vector Store Validation:")
        try:
            config = VectorStoreConfig(
                dimension=1024,
                distance_metric="cosine",
                index_type="ivfflat"
            )
            print(f"   ✓ Vector store config created successfully")
            print(f"   ✓ Dimension: {config.dimension}")
            print(f"   ✓ Distance Metric: {config.distance_metric}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()
        
    except ImportError as e:
        print(f"⚠️  Configuration module not available: {e}")
        print()


def demo_migration_examples():
    """Show migration from old to new system."""
    print("🔄 Migration Examples\n")
    
    print("OLD WAY (before configuration system):")
    print("   embedding = SentenceTransformersProvider('all-MiniLM-L6-v2')")
    print("   repo = PostgresDataRepository(db_conn)")
    print("   # Limited customization, hardcoded values")
    print()
    
    print("NEW WAY (with configuration system):")
    print("   # Use defaults")
    print("   embedding = SentenceTransformersProvider()")
    print("   repo = PostgresDataRepository(db_conn)")
    print()
    
    print("   # Custom configuration")
    print("   embedding = SentenceTransformersProvider({")
    print("       'model_name': 'all-MiniLM-L6-v2',")
    print("       'dimension': 384,")
    print("       'device': 'cpu',")
    print("       'normalize_embeddings': True")
    print("   })")
    print()
    
    print("   repo = PostgresDataRepository(db_conn, {")
    print("       'dimension': 384,  # Match embedding dimension")
    print("       'index_type': 'hnsw',")
    print("       'distance_metric': 'cosine'")
    print("   })")
    print()
    
    print("BENEFITS:")
    print("   ✓ Flexible configuration")
    print("   ✓ Validation prevents errors")
    print("   ✓ Consistent dimensions across components")
    print("   ✓ Production-ready defaults")
    print("   ✓ Backward compatibility")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 DataLoad Configuration System Demo")
    print("=" * 60)
    print()
    
    demo_embedding_configurations()
    demo_vector_store_configurations()
    demo_configuration_validation()
    demo_migration_examples()
    
    print("=" * 60)
    print("✨ Demo completed! Check the documentation for more details.")
    print("📚 See: docs/configuration_system_guide.md")
    print("🔧 Examples: examples/configuration_examples.py")
    print("🧪 Tests: tests/test_configuration_system.py")
    print("=" * 60)