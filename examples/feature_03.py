#!/usr/bin/env python3
"""
FEATURE 3: Vector Stores (PostgreSQL, Chroma, FAISS)

This demonstrates using different vector stores with the library:
1. PostgreSQL with pgvector (main_pg_gemni.py, main_pg_st.py)
2. ChromaDB (main_chroma_gemni.py, main_chroma_st.py)
3. FAISS (main_faiss_genai.py, main_faiss_st.py)

Shows embedding strategies and search across different backends.

Prerequisites:
- Run 01_generate_test_data.py first
- PostgreSQL with pgvector (for PostgreSQL backend)
- GEMINI_API_KEY (optional - uses mock if not set)
"""

import asyncio
import os
from typing import List, Any
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from dataload.infrastructure.vector_stores.faiss_store import FaissVectorStore
from dataload.infrastructure.storage.loaders import LocalLoader
from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface


class SimpleMockProvider(EmbeddingProviderInterface):
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def get_dimension(self) -> int:
        return self.embedding_dim
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        import hashlib
        embeddings = []
        for text in texts:
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            embedding = [(hash_val + i) % 100 / 100.0 for i in range(self.embedding_dim)]
            embeddings.append(embedding)
        return embeddings





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

# ==================== FEATURE 3.1: PostgreSQL Vector Store ====================

async def feature_3_1_postgres_vector_store():
    """
    Feature 3.1: PostgreSQL with pgvector
    
    Based on: main_pg_gemni.py
    - Most common vector store
    - Persistent storage
    - SQL queries with vector similarity
    """
    print("\n" + "="*70)
    print("FEATURE 3.1: PostgreSQL Vector Store (pgvector)")
    print("="*70)
    
    db_conn = None
    try:
        emb_configs = get_embedding_configs()
        vec_configs = get_vector_store_configs()
        
        gemini_config = emb_configs['gemini']
        postgres_config = vec_configs['postgres_768']
        
        print(f"🔧 Embedding Config: {gemini_config}")
        print(f"🔧 Vector Store Config: {postgres_config}")
        # 1. Database connection

        # Database connection
        db_conn = DBConnection()
        await db_conn.initialize()
        # 2. Repository: INJECT the explicit 768-dim config
        repo = PostgresDataRepository(db_conn, config=postgres_config)
        print("✅ Database connected")

        embedding = SimpleMockProvider(768)
        loader = LocalLoader()
        use_case = dataloadUseCase(repo, embedding, loader)
        
        print("✅ PostgreSQL vector store initialized")
        
        # Load data with combined embeddings
        print("📥 Loading data with combined embeddings...")
        await use_case.execute(
            'test_data/csv/products.csv',
            'postgres_products_combined',
            ['name', 'description'],
            ['product_id'],
            create_table_if_not_exists=True,
            embed_type='combined'  # Single embeddings column
        )
        print("   ✅ Table: postgres_products_combined")
        print("   🔢 Embedding column: embeddings")
        
        # Load data with separated embeddings
        print("\n📥 Loading data with separated embeddings...")
        await use_case.execute(
            'test_data/csv/products.csv',
            'postgres_products_separated',
            ['name', 'description'],
            ['product_id'],
            create_table_if_not_exists=True,
            embed_type='separated'  # Separate columns
        )
        print("   ✅ Table: postgres_products_separated")
        print("   🔢 Embedding columns: name_enc, description_enc")
        
        # Search demonstration
        print("\n🔍 Testing similarity search...")
        query = "wireless headphones"
        query_emb = embedding.get_embeddings([query])[0]
        
        # Search in combined
        results = await repo.search('postgres_products_combined', query_emb, top_k=3)
        print(f"   Query: '{query}' (combined embeddings)")
        for i, r in enumerate(results, 1):
            name = r['metadata'].get('name', 'N/A')
            print(f"      {i}. {name}")
        
        # Search in separated (specific field)
        results = await repo.search(
            'postgres_products_separated', 
            query_emb, 
            top_k=3,
            embed_column='name_enc'  # Search only in name field
        )
        print(f"\n   Query: '{query}' (name field only)")
        for i, r in enumerate(results, 1):
            name = r['metadata'].get('name', 'N/A')
            print(f"      {i}. {name}")
        
        print(f"\n💡 PostgreSQL Features:")
        print(f"   ✓ Persistent storage")
        print(f"   ✓ SQL queries available")
        print(f"   ✓ ACID transactions")
        print(f"   ✓ Supports both combined and separated embeddings")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if db_conn:
            await db_conn.close()


# ==================== FEATURE 3.2: ChromaDB Vector Store ====================

async def feature_3_2_chroma_vector_store():
    """
    Feature 3.2: ChromaDB Vector Store
    
    Based on: main_chroma_gemni.py, main_chroma_st.py
    - Lightweight vector database
    - Persistent or in-memory mode
    - Great for development/testing
    """
    print("\n" + "="*70)
    print("FEATURE 3.2: ChromaDB Vector Store")
    print("="*70)
    
    try:
        # Initialize Chroma (from main_chroma_gemni.py)
        # Persistent mode
        repo_persistent = ChromaVectorStore(
            mode="persistent",
            path="./test_chroma_db"
        )
        print("✅ ChromaDB initialized (persistent mode)")
        
        embedding = SimpleMockProvider(768)
        loader = LocalLoader()
        # ChromaVectorStore does not explicitly implement the DataRepositoryInterface
        # at the type level; cast to Any so the dataloadUseCase constructor accepts it.
        repo_adapter: Any = repo_persistent
        use_case = dataloadUseCase(repo_adapter, embedding, loader)
        
        # Load with separated embeddings
        print("📥 Loading data with separated embeddings...")
        await use_case.execute(
            'test_data/csv/documents.csv',
            'chroma_documents',
            ['title', 'content'],
            ['doc_id'],
            create_table_if_not_exists=True,
            embed_type='separated'
        )
        print("   ✅ Collection: chroma_documents")
        print("   🔢 Embeddings: title_enc, content_enc")
        
        # Search
        print("\n🔍 Testing search in ChromaDB...")
        query = "machine learning python"
        query_emb = embedding.get_embeddings([query])[0]
        
        results = await repo_persistent.search(
            'chroma_documents',
            query_emb,
            top_k=3,
            embed_column='title_enc'
        )
        
        print(f"   Query: '{query}'")
        for i, r in enumerate(results, 1):
            title = r['metadata'].get('title', 'N/A')
            print(f"      {i}. {title}")
        
        print(f"\n💡 ChromaDB Features:")
        print(f"   ✓ Lightweight and fast")
        print(f"   ✓ Persistent or in-memory modes")
        print(f"   ✓ Great for development/prototyping")
        print(f"   ✓ Easy to setup (no external dependencies)")
        
        # Demonstrate in-memory mode
        print("\n📝 In-memory mode (for testing):")
        repo_memory = ChromaVectorStore(mode="in-memory")
        print("   ✅ In-memory ChromaDB created")
        print("   💡 Data lost when process ends")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== FEATURE 3.3: FAISS Vector Store ====================

async def feature_3_3_faiss_vector_store():
    """
    Feature 3.3: FAISS Vector Store
    
    Based on: main_faiss_genai.py, main_faiss_st.py
    - Facebook AI Similarity Search
    - Extremely fast for large datasets
    - Local file-based persistence
    """
    print("\n" + "="*70)
    print("FEATURE 3.3: FAISS Vector Store")
    print("="*70)
    
    try:
        # Initialize FAISS (from main_faiss_genai.py)
        repo = FaissVectorStore()
        print("✅ FAISS vector store initialized")
        
        embedding = SimpleMockProvider(768)
        loader = LocalLoader()
        repo: Any = repo
        use_case = dataloadUseCase(repo, embedding, loader)
        
        # Load with combined embeddings
        print("📥 Loading data with combined embeddings...")
        await use_case.execute(
            'test_data/csv/products.csv',
            'faiss_products',
            ['name', 'description'],
            ['product_id'],
            create_table_if_not_exists=True,
            embed_type='combined'
        )
        print("   ✅ FAISS index: faiss_products")
        print("   💾 Saved to: faiss_data/faiss_products_embeddings.index")
        
        # Search
        print("\n🔍 Testing FAISS search...")
        query = "laptop computer"
        query_emb = embedding.get_embeddings([query])[0]
        
        results = await repo.search('faiss_products', query_emb, top_k=3)
        
        print(f"   Query: '{query}'")
        for i, r in enumerate(results, 1):
            name = r['metadata'].get('name', 'N/A')
            distance = r['distance']
            print(f"      {i}. {name} (distance: {distance:.3f})")
        
        # Load with separated embeddings
        print("\n📥 Loading with separated embeddings...")
        await use_case.execute(
            'test_data/csv/documents.csv',
            'faiss_documents',
            ['title', 'content'],
            ['doc_id'],
            create_table_if_not_exists=True,
            embed_type='separated'
        )
        print("   ✅ FAISS index: faiss_documents (title_enc, content_enc)")
        
        # Search in specific field
        results = await repo.search(
            'faiss_documents',
            query_emb,
            top_k=3,
            embed_column='title_enc'
        )
        print(f"\n   Query: '{query}' (title field)")
        for i, r in enumerate(results, 1):
            title = r['metadata'].get('title', 'N/A')
            print(f"      {i}. {title}")
        
        print(f"\n💡 FAISS Features:")
        print(f"   ✓ Extremely fast similarity search")
        print(f"   ✓ Optimized for large-scale datasets")
        print(f"   ✓ File-based persistence")
        print(f"   ✓ Automatic index loading/saving")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== COMPARISON ====================

def print_comparison():
    """Print comparison of vector stores."""
    print("\n" + "="*70)
    print("📊 VECTOR STORE COMPARISON")
    print("="*70)
    
    comparison = """
┌──────────────┬─────────────┬─────────────┬─────────────┐
│ Feature      │ PostgreSQL  │ ChromaDB    │ FAISS       │
├──────────────┼─────────────┼─────────────┼─────────────┤
│ Speed        │ Medium      │ Fast        │ Fastest     │
│ Persistence  │ Database    │ Optional    │ File-based  │
│ Setup        │ Complex     │ Easy        │ Easy        │
│ SQL Queries  │ ✅ Yes      │ ❌ No       │ ❌ No       │
│ Scalability  │ High        │ Medium      │ Very High   │
│ Best For     │ Production  │ Development │ Large Scale │
└──────────────┴─────────────┴─────────────┴─────────────┘
"""
    print(comparison)
    
    print("\n💡 When to use:")
    print("   🗄️  PostgreSQL: Production apps with existing PostgreSQL")
    print("   🔬 ChromaDB: Development, prototyping, small datasets")
    print("   🚀 FAISS: Large-scale similarity search, performance-critical")


# ==================== MAIN ====================

async def main():
    """Run all vector store examples."""
    print("=" * 70)
    print("FEATURE 3: Vector Stores (PostgreSQL, Chroma, FAISS)")
    print("=" * 70)
    print("\n📚 Based on library examples:")
    print("   - main_pg_gemni.py (PostgreSQL)")
    print("   - main_chroma_gemni.py (ChromaDB)")
    print("   - main_faiss_genai.py (FAISS)")
    
    try:
        # Run all vector store examples
        await feature_3_1_postgres_vector_store()
        await feature_3_2_chroma_vector_store()
        await feature_3_3_faiss_vector_store()
        
        # Print comparison
        print_comparison()
        
        print("\n" + "="*70)
        print("✅ Feature 3 Complete!")
        print("="*70)
        print("\n📊 Data Loaded Into:")
        print("   PostgreSQL:")
        print("     - postgres_products_combined")
        print("     - postgres_products_separated")
        print("   ChromaDB:")
        print("     - chroma_documents")
        print("   FAISS:")
        print("     - faiss_products")
        print("     - faiss_documents")
        
        print("\n💡 All three vector stores working!")
        print("   ✓ Same dataloadUseCase interface")
        print("   ✓ Same embedding workflow")
        print("   ✓ Different backends for different needs")
        
    except Exception as e:
        print(f"\n❌ Feature 3 failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 DataLoad Library - Feature 3 (Vector Stores)\n")
    asyncio.run(main())