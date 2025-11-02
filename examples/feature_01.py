#!/usr/bin/env python3
"""
FEATURE 1: CSV to PostgreSQL with Embeddings - REFACRORED

This script demonstrates loading CSV files into PostgreSQL with various
embedding strategies using the DataLoad library, ensuring dimension consistency
and clean configuration passing.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any, Tuple
# Assuming these imports are correct based on the original code structure
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.storage.loaders import LocalLoader
from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface


# ----------------------------------------------------------------------
# Custom/Mock Providers
# ----------------------------------------------------------------------

class SimpleMockProvider(EmbeddingProviderInterface):
    """Simple mock embedding provider for testing without API keys."""
    
    # Refactored to accept and store config/dimension explicitly
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_dim = config['dimension']
    
    def get_dimension(self) -> int:
        return self.embedding_dim

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic mock embeddings."""
        return [[float(i % 100) / 100 for i in range(self.embedding_dim)] 
                for _ in texts]



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

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

async def setup_core_components() -> Tuple[DBConnection, dataloadUseCase, EmbeddingProviderInterface]:
    """
    Initializes the core 768-dimension components (Repo and Embedder).
    """
    print("🔧 Setting up core 768-dim components...")
    
    emb_configs = get_embedding_configs()
    vec_configs = get_vector_store_configs()
    
    gemini_config = emb_configs['gemini']
    postgres_config = vec_configs['postgres_768']
    
    print(f"🔧 Embedding Config: {gemini_config}")
    print(f"🔧 Vector Store Config: {postgres_config}")
       # 1. Database connection
    db_conn = DBConnection()
    await db_conn.initialize()
    print("✅ Database connected.")
    
    # 2. Repository: INJECT the explicit 768-dim config
    repo = PostgresDataRepository(db_conn, config=postgres_config)
    
    # 3. Embedding provider: INJECT the explicit 768-dim config
    if not os.getenv('GEMINI_API_KEY'):
        embedding = SimpleMockProvider(config=gemini_config) # Use mock with 768-dim config
        print(f"✅ Using Mock Embedding Provider (Dimension: {embedding.get_dimension()})")
    else:
        embedding = GeminiEmbeddingProvider(config=gemini_config)
        print(f"✅ Using Gemini Embedding Provider (Dimension: {embedding.get_dimension()})")
        
    # 4. Storage loader
    loader = LocalLoader()
    
    # 5. Use case
    use_case = dataloadUseCase(repo, embedding, loader)
    
    return db_conn, use_case, embedding


# ----------------------------------------------------------------------
# Feature Functions
# ----------------------------------------------------------------------

# Feature 1.1 (Uncommented and Integrated)
async def feature_1_1_simple_csv_loading(use_case: dataloadUseCase):
    """
    Feature 1.1: Simple CSV Loading (No Embeddings)
    """
    print("\n" + "="*70)
    print("FEATURE 1.1: Simple CSV Loading (No Embeddings)")
    print("="*70)
    
    try:
        await use_case.execute(
            s3_uri='test_data/csv/employees_basic.csv',
            table_name='employees_no_embeddings',
            embed_columns_names=[],  # No embeddings
            pk_columns=['id'],
            create_table_if_not_exists=True,
            # Explicitly set embed_type=None or remove if execute handles it
        )
        print(f"✅ Success! Table: **employees_no_embeddings** (No embedding columns created)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Feature 1.2 (No change needed, as setup is now correct)
async def feature_1_2_combined_embeddings(use_case: dataloadUseCase):
    """
    Feature 1.2: CSV with Embeddings (Combined Mode). Uses core 768-dim setup.
    """
    print("\n" + "="*70)
    print("FEATURE 1.2: CSV with Embeddings (Combined Mode)")
    print("="*70)
    
    try:
        await use_case.execute(
            s3_uri='test_data/csv/employees_basic.csv',
            table_name='employees_combined_emb',
            embed_columns_names=['name', 'bio'], 
            pk_columns=['id'],
            create_table_if_not_exists=True,
            embed_type='combined' 
        )
        
        print(f"✅ Success! Table: **employees_combined_emb** (Combined name + bio embeddings)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Feature 1.3 (No change needed, as setup is now correct)
async def feature_1_3_separated_embeddings(use_case: dataloadUseCase):
    """
    Feature 1.3: CSV with Embeddings (Separated Mode). Uses core 768-dim setup.
    """
    print("\n" + "="*70)
    print("FEATURE 1.3: CSV with Embeddings (Separated Mode)")
    print("="*70)
    
    try:
        await use_case.execute(
            s3_uri='test_data/csv/products.csv',
            table_name='products_separated_emb',
            embed_columns_names=['name', 'description'],
            pk_columns=['product_id'],
            create_table_if_not_exists=True,
            embed_type='separated' 
        )
        
        print(f"✅ Success! Table: **products_separated_emb** (Separate name_enc, description_enc columns)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# Feature 1.4 (Refactored for isolated 384-dim mock)
async def feature_1_4_mock_embeddings(db_conn: DBConnection):
    """
    Feature 1.4: CSV with Mock Embeddings (No API Key).
    Uses dedicated 384-dimension components for testing isolation.
    """
    print("\n" + "="*70)
    print("FEATURE 1.4: CSV with Mock Embeddings (Dedicated 384-dim)")
    print("="*70)
    
    try:
        emb_configs = get_embedding_configs()
        vec_configs = get_vector_store_configs()

        # Dedicated 384-dim setup
        mock_config = emb_configs['mock_384']
        pg_config_384 = vec_configs['postgres_384']
        
        # Initialize isolated components
        mock_embedding = SimpleMockProvider(config=mock_config)
        mock_repo = PostgresDataRepository(db_conn, config=pg_config_384)
        loader = LocalLoader()
        
        use_case = dataloadUseCase(mock_repo, mock_embedding, loader)
        
        print(f"  - Using isolated dimension: {mock_embedding.get_dimension()}")
        
        await use_case.execute(
            s3_uri='test_data/csv/documents.csv',
            table_name='documents_mock_emb_384', # Changed table name to reflect dim
            embed_columns_names=['title', 'content'],
            pk_columns=['doc_id'],
            create_table_if_not_exists=True,
            embed_type='combined'
        )
        
        print(f"✅ Success! Table: **documents_mock_emb_384** (Mock embeddings created)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ----------------------------------------------------------------------
# Demonstration
# ----------------------------------------------------------------------

async def demonstrate_search(db_conn, embedding: EmbeddingProviderInterface):
    """Demonstrate vector similarity search."""
    print("\n" + "="*70)
    print("🔍 BONUS: Vector Similarity Search Demo")
    print("="*70)
    
    try:
        # Search in combined embeddings table (employees_combined_emb)
        query = "software engineer python"
        # Get embeddings from the core provider
        query_embedding = embedding.get_embeddings([query])[0]
        
        
        repo = PostgresDataRepository(db_conn)
        # For combined mode (uses 'embeddings' column)
        results = await repo.search("test_table_com_pg_gemini_st", query_embedding, top_k=5)
        print("Combined mode retrieval results:")
        for result in results:
            print(
                f"ID: {result['id']}, Document: {result['document']}, Distance: {result['distance']}, Metadata: {result['metadata']}"
            )

            
        # NOTE: This requires the PostgresDataRepository to have a search method
        # and for the employees_combined_emb table to be created first (Feature 1.2)
        # results = await repo.search(
        #     table_name='employees_combined_emb',
        #     query_embedding=query_embedding,
        #     top_k=3
        # )
        
        # ... (rest of the display logic remains the same)
        print(f"🔎 Query: '{query}'")
        print(f"📋 Top 3 results:")
        for i, result in enumerate(results, 1):
            name = result['metadata'].get('name', 'N/A')
            bio = result['metadata'].get('bio', 'N/A')[:60] + '...'
            similarity = 1 - result['distance']
            print(f"\n  {i}. {name}")
            print(f"    Bio: {bio}")
            print(f"    Similarity: {similarity:.3f}")
        
    except Exception as e:
        print(f"⚠️ Search demo skipped (check if repo.search() exists and tables are populated): {e}")


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

async def main():
    """Run all Feature 1 examples."""
    # ... (Intro printing)
    db_conn = None
    core_use_case = None
    core_embedding = None
    
    try:
        # 1. Setup Core Components (768-dim)
        db_conn, core_use_case, core_embedding = await setup_core_components()
        
        # 2. Run features using the core 768-dim components
        await feature_1_1_simple_csv_loading(core_use_case)
        await feature_1_2_combined_embeddings(core_use_case)
        await feature_1_3_separated_embeddings(core_use_case)
        
        # 3. Run feature 1.4 using its isolated 384-dim components
        await feature_1_4_mock_embeddings(db_conn)
        
        # 4. Demo search (uses core 768-dim components and table from 1.2)
        await demonstrate_search(db_conn, core_embedding)
        
        print("\n" + "="*70)
        print("✅ Feature 1 Complete!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Feature 1 failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if db_conn:
            await db_conn.close()
            print("\n🔌 Database connection closed")


if __name__ == "__main__":
    print("🚀 DataLoad Library - Feature 1 Examples")
    print("Run 01_generate_test_data.py first to create test data\n")
    asyncio.run(main())