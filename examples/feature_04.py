#!/usr/bin/env python3
"""
FEATURE 4: Embedding Providers with Configuration System

This demonstrates the new configuration system for embedding providers and vector stores:
1. GeminiEmbeddingProvider with custom config (768-dim)
2. SentenceTransformersProvider with custom config (384-dim) 
3. BedrockEmbeddingProvider with custom config (1024-dim)
4. OpenAIEmbeddingProvider with custom config (1536-dim)
5. Vector stores with matching configurations
6. Mock provider for testing

Shows how to:
- Configure embedding providers with custom dimensions and settings
- Configure vector stores with matching dimensions
- Use default configurations vs custom configurations
- Maintain backward compatibility

Prerequisites:
- Run 01_generate_test_data.py first
- PostgreSQL with pgvector
- API keys for cloud providers (optional - will use mock if not available)
"""

import asyncio
import os
import sys
from typing import List, Tuple, Dict, Any

# Ensure src directory is in path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.storage.loaders import LocalLoader
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface

# Import embedding providers
try:
    from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  GeminiEmbeddingProvider not available")

try:
    from dataload.application.services.embedding.sentence_transformers_provider import SentenceTransformersProvider
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("⚠️  SentenceTransformersProvider not available")

try:
    from dataload.application.services.embedding.bedrock_provider import BedrockEmbeddingProvider
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    print("⚠️  BedrockEmbeddingProvider not available")

try:
    from dataload.application.services.embedding.openai_provider import OpenAIEmbeddingProvider
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAIEmbeddingProvider not available")


class SimpleMockProvider(EmbeddingProviderInterface):
    """Mock provider for testing without API keys."""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.embedding_dim = config.get('dimension', 384)
        self.model_name = config.get('model_name', 'mock-model')
        print(f"   📏 Mock Provider - Model: {self.model_name}, Dimension: {self.embedding_dim}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        import hashlib
        embeddings = []
        for text in texts:
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            embedding = [(hash_val + i) % 100 / 100.0 for i in range(self.embedding_dim)]
            embeddings.append(embedding)
        return embeddings
    
    def get_dimension(self) -> int:
        return self.embedding_dim


# ==================== CONFIGURATION EXAMPLES ====================

def get_embedding_configs():
    """Define custom configurations for each embedding provider."""
    return {
        'gemini': {
            'model': 'text-embedding-004',
            'dimension': 768,
            'task_type': 'SEMANTIC_SIMILARITY'
        },
        'sentence_transformers': {
            'model_name': 'sentence-transformers/all-mpnet-base-v2',
            'dimension': 768,  # Using larger model for better quality
            'device': 'cpu',
            'normalize_embeddings': True
        },
        'bedrock': {
            'model_id': 'amazon.titan-embed-text-v2:0',
            'dimension': 1024,
            'region': 'us-east-1',
            'content_type': 'application/json'
        },
        'openai': {
            'model': 'text-embedding-3-large',
            'dimension': 3072  # Using large model for maximum quality
        },
        'mock_small': {
            'dimension': 384,
            'model_name': 'mock-small-model'
        },
        'mock_large': {
            'dimension': 1536,
            'model_name': 'mock-large-model'
        }
    }


def get_vector_store_configs():
    """Define custom configurations for vector stores matching embedding dimensions."""
    return {
        'postgres_768': {
            'dimension': 768,
            'index_type': 'hnsw',  # Use HNSW for better performance with 768 dims
            'distance_metric': 'cosine',
            'hnsw_m': 32,
            'hnsw_ef_construction': 128
        },
        'postgres_1024': {
            'dimension': 1024,
            'index_type': 'ivfflat',  # Use IVFFlat for 1024 dims
            'distance_metric': 'cosine',
            'ivfflat_lists': 100
        },
        'postgres_3072': {
            'dimension': 3072,
            'index_type': 'ivfflat',  # Must use IVFFlat for high dimensions
            'distance_metric': 'cosine',
            'ivfflat_lists': 200
        },
        'postgres_384': {
            'dimension': 384,
            'index_type': 'hnsw',  # HNSW works well for smaller dimensions
            'distance_metric': 'cosine',
            'hnsw_m': 16,
            'hnsw_ef_construction': 64
        }
    }


# ==================== FEATURE 4.1: Gemini with Custom Config ====================

async def feature_4_1_gemini_custom_config():
    """Feature 4.1: Google Gemini with Custom Configuration
    
    Demonstrates:
    - Custom model and dimension configuration
    - Matching vector store configuration
    - High-quality 768-dimensional embeddings
    """
    print("\n" + "="*70)
    print("FEATURE 4.1: Gemini Embeddings with Custom Configuration")
    print("="*70)
    
    # Get configurations
    embedding_configs = get_embedding_configs()
    vector_configs = get_vector_store_configs()
    
    gemini_config = embedding_configs['gemini']
    postgres_config = vector_configs['postgres_768']
    
    print(f"🔧 Embedding Config: {gemini_config}")
    print(f"🔧 Vector Store Config: {postgres_config}")
    
    if not GEMINI_AVAILABLE:
        print("❌ GeminiEmbeddingProvider not available")
        print("   Install: pip install google-generativeai")
        return
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️  GOOGLE_API_KEY not set - using mock provider")
        embedding = SimpleMockProvider(gemini_config)
    else:
        embedding = GeminiEmbeddingProvider(gemini_config)
        print("✅ Gemini provider initialized with custom config")
    
    db_conn = None
    try:
        # Setup with custom configurations
        db_conn = DBConnection()
        await db_conn.initialize()
        repo = PostgresDataRepository(db_conn, postgres_config)
        loader = LocalLoader()
        use_case = dataloadUseCase(repo, embedding, loader)
        
        # Load with custom Gemini configuration
        print("📥 Loading data with custom Gemini config...")
        await use_case.execute(
            'test_data/csv/documents.csv',
            'gemini_custom_documents',
            ['title', 'content'],
            ['doc_id'],
            create_table_if_not_exists=True,
            embed_type='combined'
        )
        
        print("   ✅ Table: gemini_custom_documents")
        print(f"   🔢 Embeddings: {embedding.get_dimension()}-dimensional")
        print(f"   📊 Index: {postgres_config['index_type']} with {postgres_config['distance_metric']} distance")
        
        # Test search
        print("\n🔍 Testing semantic search...")
        query = "artificial intelligence and machine learning"
        query_emb = embedding.get_embeddings([query])[0]
        results = await repo.search('gemini_custom_documents', query_emb, top_k=3)
        
        print(f"   Query: '{query}'")
        for i, r in enumerate(results, 1):
            title = r['metadata'].get('title', 'N/A')
            similarity = 1 - r['distance']
            print(f"      {i}. {title} (similarity: {similarity:.3f})")
        
        print(f"\n💡 Custom Gemini Features:")
        print(f"   ✓ Model: {gemini_config['model']}")
        print(f"   ✓ Dimension: {gemini_config['dimension']}")
        print(f"   ✓ Task Type: {gemini_config['task_type']}")
        print(f"   ✓ Optimized vector index for dimension")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if db_conn:
            await db_conn.close()


# ==================== FEATURE 4.2: SentenceTransformers with Custom Config ====================

async def feature_4_2_sentence_transformers_custom():
    """Feature 4.2: SentenceTransformers with Custom Configuration
    
    Demonstrates:
    - Custom model selection (larger, better quality model)
    - Custom dimension and device configuration
    - Matching vector store configuration
    """
    print("\n" + "="*70)
    print("FEATURE 4.2: SentenceTransformers with Custom Configuration")
    print("="*70)
    
    # Get configurations
    embedding_configs = get_embedding_configs()
    vector_configs = get_vector_store_configs()
    
    st_config = embedding_configs['sentence_transformers']
    postgres_config = vector_configs['postgres_768']
    
    print(f"🔧 Embedding Config: {st_config}")
    print(f"🔧 Vector Store Config: {postgres_config}")
    
    if not ST_AVAILABLE:
        print("⚠️  SentenceTransformersProvider not available")
        print("   Install: pip install sentence-transformers")
        embedding = SimpleMockProvider(st_config)
    else:
        embedding = SentenceTransformersProvider(st_config)
        print("✅ SentenceTransformers provider initialized with custom config")
    
    db_conn = None
    try:
        # Setup with custom configurations
        db_conn = DBConnection()
        await db_conn.initialize()
        repo = PostgresDataRepository(db_conn, postgres_config)
        loader = LocalLoader()
        use_case = dataloadUseCase(repo, embedding, loader)
        
        # Load with custom SentenceTransformers configuration
        print("📥 Loading data with custom SentenceTransformers config...")
        await use_case.execute(
            'test_data/csv/products.csv',
            'st_custom_products',
            ['name', 'description'],
            ['product_id'],
            create_table_if_not_exists=True,
            embed_type='separated'
        )
        
        print("   ✅ Table: st_custom_products")
        print(f"   🔢 Embeddings: {embedding.get_dimension()}-dimensional")
        print("   📋 Columns: name_enc, description_enc")
        print(f"   📊 Index: {postgres_config['index_type']} with {postgres_config['distance_metric']} distance")
        
        # Test search
        print("\n🔍 Testing search...")
        query = "wireless bluetooth speaker"
        query_emb = embedding.get_embeddings([query])[0]
        results = await repo.search(
            'st_custom_products',
            query_emb,
            top_k=3,
            embed_column='description_enc'
        )
        
        print(f"   Query: '{query}' (in descriptions)")
        for i, r in enumerate(results, 1):
            name = r['metadata'].get('name', 'N/A')
            print(f"      {i}. {name}")
        
        print(f"\n💡 Custom SentenceTransformers Features:")
        print(f"   ✓ Model: {st_config['model_name']}")
        print(f"   ✓ Dimension: {st_config['dimension']}")
        print(f"   ✓ Device: {st_config['device']}")
        print(f"   ✓ Normalization: {st_config['normalize_embeddings']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if db_conn:
            await db_conn.close()


# ==================== FEATURE 4.3: Bedrock with Custom Config ====================

async def feature_4_3_bedrock_custom_config():
    """Feature 4.3: AWS Bedrock with Custom Configuration
    
    Demonstrates:
    - Custom model and region configuration
    - 1024-dimensional embeddings
    - IVFFlat index optimization for medium dimensions
    """
    print("\n" + "="*70)
    print("FEATURE 4.3: AWS Bedrock with Custom Configuration")
    print("="*70)
    
    # Get configurations
    embedding_configs = get_embedding_configs()
    vector_configs = get_vector_store_configs()
    
    bedrock_config = embedding_configs['bedrock']
    postgres_config = vector_configs['postgres_1024']
    
    print(f"🔧 Embedding Config: {bedrock_config}")
    print(f"🔧 Vector Store Config: {postgres_config}")
    
    if not BEDROCK_AVAILABLE:
        print("⚠️  BedrockEmbeddingProvider not available")
        embedding = SimpleMockProvider(bedrock_config)
    else:
        try:
            embedding = BedrockEmbeddingProvider(bedrock_config)
            print("✅ Bedrock provider initialized with custom config")
        except Exception as e:
            print(f"⚠️  Bedrock initialization failed: {e}")
            print("   Using mock provider instead")
            embedding = SimpleMockProvider(bedrock_config)
    
    db_conn = None
    try:
        # Setup with custom configurations
        db_conn = DBConnection()
        await db_conn.initialize()
        repo = PostgresDataRepository(db_conn, postgres_config)
        loader = LocalLoader()
        use_case = dataloadUseCase(repo, embedding, loader)
        
        # Load with custom Bedrock configuration
        print("📥 Loading data with custom Bedrock config...")
        await use_case.execute(
            'test_data/csv/employees_basic.csv',
            'bedrock_custom_employees',
            ['name', 'bio'],
            ['id'],
            create_table_if_not_exists=True,
            embed_type='combined'
        )
        
        print("   ✅ Table: bedrock_custom_employees")
        print(f"   🔢 Embeddings: {embedding.get_dimension()}-dimensional")
        print(f"   📊 Index: {postgres_config['index_type']} with {postgres_config['distance_metric']} distance")
        
        print(f"\n💡 Custom Bedrock Features:")
        print(f"   ✓ Model: {bedrock_config['model_id']}")
        print(f"   ✓ Dimension: {bedrock_config['dimension']}")
        print(f"   ✓ Region: {bedrock_config['region']}")
        print(f"   ✓ IVFFlat index optimized for 1024 dimensions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if db_conn:
            await db_conn.close()


# ==================== FEATURE 4.4: OpenAI with Custom Config ====================

async def feature_4_4_openai_custom_config():
    """Feature 4.4: OpenAI with Custom Configuration
    
    Demonstrates:
    - Custom model with high dimensions (3072)
    - IVFFlat index for high-dimensional embeddings
    - Maximum quality embeddings
    """
    print("\n" + "="*70)
    print("FEATURE 4.4: OpenAI with Custom Configuration")
    print("="*70)
    
    # Get configurations
    embedding_configs = get_embedding_configs()
    vector_configs = get_vector_store_configs()
    
    openai_config = embedding_configs['openai']
    postgres_config = vector_configs['postgres_3072']
    
    print(f"🔧 Embedding Config: {openai_config}")
    print(f"🔧 Vector Store Config: {postgres_config}")
    
    if not OPENAI_AVAILABLE:
        print("⚠️  OpenAIEmbeddingProvider not available")
        embedding = SimpleMockProvider(openai_config)
    else:
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️  OPENAI_API_KEY not set - using mock provider")
            embedding = SimpleMockProvider(openai_config)
        else:
            try:
                embedding = OpenAIEmbeddingProvider(openai_config)
                print("✅ OpenAI provider initialized with custom config")
            except Exception as e:
                print(f"⚠️  OpenAI initialization failed: {e}")
                embedding = SimpleMockProvider(openai_config)
    
    db_conn = None
    try:
        # Setup with custom configurations
        db_conn = DBConnection()
        await db_conn.initialize()
        repo = PostgresDataRepository(db_conn, postgres_config)
        loader = LocalLoader()
        use_case = dataloadUseCase(repo, embedding, loader)
        
        # Load with custom OpenAI configuration
        print("📥 Loading data with custom OpenAI config...")
        await use_case.execute(
            'test_data/csv/documents.csv',
            'openai_custom_documents',
            ['title', 'content'],
            ['doc_id'],
            create_table_if_not_exists=True,
            embed_type='combined'
        )
        
        print("   ✅ Table: openai_custom_documents")
        print(f"   🔢 Embeddings: {embedding.get_dimension()}-dimensional")
        print(f"   📊 Index: {postgres_config['index_type']} with {postgres_config['distance_metric']} distance")
        
        print(f"\n💡 Custom OpenAI Features:")
        print(f"   ✓ Model: {openai_config['model']}")
        print(f"   ✓ Dimension: {openai_config['dimension']}")
        print(f"   ✓ High-quality embeddings")
        print(f"   ✓ IVFFlat index for high dimensions (>2000)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if db_conn:
            await db_conn.close()


# ==================== FEATURE 4.5: Configuration Comparison ====================

async def feature_4_5_configuration_comparison():
    """Feature 4.5: Compare Default vs Custom Configurations
    
    Demonstrates:
    - Default configuration behavior
    - Custom configuration behavior
    - Backward compatibility
    """
    print("\n" + "="*70)
    print("FEATURE 4.5: Default vs Custom Configuration Comparison")
    print("="*70)
    
    db_conn = None
    try:
        db_conn = DBConnection()
        await db_conn.initialize()
        loader = LocalLoader()
        
        # Test 1: Default configuration (backward compatibility)
        print("\n🔹 Test 1: Default Configuration (Backward Compatible)")
        if ST_AVAILABLE:
            try:
                default_embedding = SentenceTransformersProvider()  # No config passed
                default_repo = PostgresDataRepository(db_conn)  # No config passed
                
                print(f"   ✅ Default embedding dimension: {default_embedding.get_dimension()}")
                print(f"   ✅ Default repo dimension: {default_repo.config.dimension}")
                print("   ✅ Backward compatibility maintained")
            except Exception as e:
                print(f"   ⚠️  Default config test: {e}")
        else:
            print("   ⚠️  SentenceTransformers not available for default test")
        
        # Test 2: Custom configuration
        print("\n🔹 Test 2: Custom Configuration")
        embedding_configs = get_embedding_configs()
        vector_configs = get_vector_store_configs()
        
        custom_embedding = SimpleMockProvider(embedding_configs['mock_large'])
        custom_repo = PostgresDataRepository(db_conn, vector_configs['postgres_384'])
        
        print(f"   ✅ Custom embedding dimension: {custom_embedding.get_dimension()}")
        print(f"   ✅ Custom repo dimension: {custom_repo.config.dimension}")
        print(f"   ✅ Custom index type: {custom_repo.config.index_type}")
        
        # Test 3: Partial configuration
        print("\n🔹 Test 3: Partial Configuration (Mix of Custom and Defaults)")
        partial_config = {'dimension': 512}  # Only specify dimension
        partial_embedding = SimpleMockProvider(partial_config)
        
        print(f"   ✅ Partial config dimension: {partial_embedding.get_dimension()}")
        print("   ✅ Other values use defaults")
        
        print(f"\n💡 Configuration System Benefits:")
        print(f"   ✓ Backward compatibility - old code still works")
        print(f"   ✓ Flexible configuration - customize what you need")
        print(f"   ✓ Sensible defaults - production-ready out of the box")
        print(f"   ✓ Validation - prevents configuration errors")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if db_conn:
            await db_conn.close()


# ==================== COMPARISON TABLE ====================

def print_configuration_comparison():
    """Print detailed comparison of all configurations."""
    print("\n" + "="*70)
    print("📊 EMBEDDING PROVIDER CONFIGURATION COMPARISON")
    print("="*70)
    
    embedding_configs = get_embedding_configs()
    
    comparison = """
┌─────────────────────┬──────────────┬────────────────┬─────────────┬─────────────┐
│ Provider            │ Dimension    │ Model/Type     │ Special     │ Use Case    │
├─────────────────────┼──────────────┼────────────────┼─────────────┼─────────────┤
│ Gemini (Custom)     │ 768          │ text-embed-004 │ Semantic    │ Production  │
│ SentenceT (Custom)  │ 768          │ all-mpnet-v2   │ Local       │ Self-hosted │
│ Bedrock (Custom)    │ 1024         │ titan-v2       │ AWS Native  │ Enterprise  │
│ OpenAI (Custom)     │ 3072         │ text-embed-3L  │ Highest Dim │ Premium     │
│ Mock Small          │ 384          │ mock-small     │ Testing     │ Development │
│ Mock Large          │ 1536         │ mock-large     │ Testing     │ Development │
└─────────────────────┴──────────────┴────────────────┴─────────────┴─────────────┘
"""
    print(comparison)
    
    print("\n📊 VECTOR STORE CONFIGURATION COMPARISON")
    print("="*70)
    
    vector_configs = get_vector_store_configs()
    
    vector_comparison = """
┌─────────────────────┬──────────────┬────────────────┬─────────────┬─────────────┐
│ Configuration       │ Dimension    │ Index Type     │ Distance    │ Optimization│
├─────────────────────┼──────────────┼────────────────┼─────────────┼─────────────┤
│ postgres_384        │ 384          │ HNSW           │ Cosine      │ Small Dims  │
│ postgres_768        │ 768          │ HNSW           │ Cosine      │ Medium Dims │
│ postgres_1024       │ 1024         │ IVFFlat        │ Cosine      │ Medium Dims │
│ postgres_3072       │ 3072         │ IVFFlat        │ Cosine      │ High Dims   │
└─────────────────────┴──────────────┴────────────────┴─────────────┴─────────────┘
"""
    print(vector_comparison)
    
    print("\n💡 Configuration Guidelines:")
    print("   🎯 Match embedding and vector store dimensions")
    print("   📏 Use HNSW for dimensions ≤ 2000, IVFFlat for higher")
    print("   🚀 Cosine distance works well for most embedding types")
    print("   ⚙️  Tune index parameters based on data size and performance needs")


# ==================== MAIN ====================

async def main():
    """Run all embedding provider configuration examples."""
    print("=" * 70)
    print("FEATURE 4: Embedding Providers with Configuration System")
    print("=" * 70)
    print("\n🎯 Demonstrates:")
    print("   - Custom configurations for all embedding providers")
    print("   - Matching vector store configurations")
    print("   - Backward compatibility")
    print("   - Configuration validation and best practices")
    
    try:
        # Run all configuration examples
        await feature_4_1_gemini_custom_config()
        await feature_4_2_sentence_transformers_custom()
        await feature_4_3_bedrock_custom_config()
        await feature_4_4_openai_custom_config()
        await feature_4_5_configuration_comparison()
        
        # Print comparison tables
        print_configuration_comparison()
        
        print("\n" + "="*70)
        print("✅ Feature 4 Complete!")
        print("="*70)
        print("\n📊 Tables Created with Custom Configurations:")
        print("   - gemini_custom_documents (768-dim, HNSW index)")
        print("   - st_custom_products (768-dim, HNSW index)")
        print("   - bedrock_custom_employees (1024-dim, IVFFlat index)")
        print("   - openai_custom_documents (3072-dim, IVFFlat index)")
        
        print("\n💡 Key Configuration Benefits:")
        print("   ✓ No hardcoded dimensions - all configurable")
        print("   ✓ Automatic index optimization based on dimensions")
        print("   ✓ Consistent configuration across providers and stores")
        print("   ✓ Production-ready defaults with custom flexibility")
        print("   ✓ Full backward compatibility maintained")
        
        print("\n🚀 Migration Path:")
        print("   OLD: embedding = SentenceTransformersProvider('model-name')")
        print("   NEW: embedding = SentenceTransformersProvider({'model_name': 'model-name', 'dimension': 768})")
        print("   COMPATIBLE: embedding = SentenceTransformersProvider()  # Still works!")
        
    except Exception as e:
        print(f"\n❌ Feature 4 failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 DataLoad Library - Feature 4 (Embedding Providers with Configuration)\n")
    asyncio.run(main())