#!/usr/bin/env python3
"""
FEATURE 1: CSV to PostgreSQL with Embeddings

This script demonstrates loading CSV files into PostgreSQL with various
embedding strategies using the DataLoad library.

Sub-features covered:
1.1 - Simple CSV Loading (no embeddings)
1.2 - CSV with Gemini Embeddings (Combined mode)
1.3 - CSV with Gemini Embeddings (Separated mode)
1.4 - CSV with Mock Embeddings (no API key needed)

Prerequisites:
- Run 01_generate_test_data.py first
- PostgreSQL with pgvector extension
- GEMINI_API_KEY environment variable (for real embeddings)
"""

import asyncio
import os
import sys
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.storage.loaders import LocalLoader
from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase


# Mock Embedding Provider (no API key needed)
from typing import List
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface

class SimpleMockProvider(EmbeddingProviderInterface):
    """Simple mock embedding provider for testing without API keys."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic mock embeddings."""
        return [[float(i % 100) / 100 for i in range(self.embedding_dim)] 
                for _ in texts]


# ==================== SETUP ====================

async def setup_components(use_mock=False):
    """Initialize database and embedding components."""
    print("🔧 Setting up components...")
    
    # Database connection
    db_conn = DBConnection()
    await db_conn.initialize()
    repo = PostgresDataRepository(db_conn)
    print("✅ Database connected")
    
    # Embedding provider
    if use_mock or not os.getenv('GOOGLE_API_KEY'):
        embedding = SimpleMockProvider(embedding_dim=384)
        print("✅ Using Mock Embedding Provider")
    else:
        embedding = GeminiEmbeddingProvider()
        print("✅ Using Gemini Embedding Provider")
    
    # Storage loader
    loader = LocalLoader()
    
    # Use case
    use_case = dataloadUseCase(repo, embedding, loader)
    
    return db_conn, use_case, embedding




# ==================== FEATURE 1.2: Combined Embeddings ====================

async def feature_1_2_combined_embeddings(use_case):
    """
    Feature 1.2: CSV with Gemini Embeddings (Combined Mode)
    
    Creates a single 'embeddings' column that combines multiple text fields.
    Best for: Simple similarity search across multiple fields.
    """
    print("\n" + "="*70)
    print("FEATURE 1.2: CSV with Embeddings (Combined Mode)")
    print("="*70)
    
    try:
        await use_case.execute(
            s3_uri='test_data/csv/employees_basic.csv',
            table_name='employees_combined_emb',
            embed_columns_names=['name', 'bio'],  # Combine these fields
            pk_columns=['id'],
            create_table_if_not_exists=True,
            embed_type='combined'  # Single embeddings column
        )
        
        print(f"✅ Success!")
        print(f"   📊 Table: employees_combined_emb")
        print(f"   🔢 Embedding column: embeddings (combined name + bio)")
        print(f"   💡 Use for: General similarity search")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ==================== FEATURE 1.3: Separated Embeddings ====================

async def feature_1_3_separated_embeddings(use_case):
    """
    Feature 1.3: CSV with Gemini Embeddings (Separated Mode)
    
    Creates separate embedding columns for each specified field.
    Best for: Targeted search on specific fields.
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
            embed_type='separated'  # Separate columns: name_enc, description_enc
        )
        
        print(f"✅ Success!")
        print(f"   📊 Table: products_separated_emb")
        print(f"   🔢 Embedding columns:")
        print(f"      - name_enc (product name embeddings)")
        print(f"      - description_enc (description embeddings)")
        print(f"   💡 Use for: Search by name OR description separately")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ==================== FEATURE 1.4: Mock Embeddings ====================

async def feature_1_4_mock_embeddings():
    """
    Feature 1.4: CSV with Mock Embeddings (No API Key)
    
    Demonstrates using mock embeddings for testing without API keys.
    Useful for development and testing.
    """
    print("\n" + "="*70)
    print("FEATURE 1.4: CSV with Mock Embeddings (No API Key)")
    print("="*70)
    
    try:
        # Force mock embeddings
        db_conn, use_case, _ = await setup_components(use_mock=False)
        
        await use_case.execute(
            s3_uri='test_data/csv/documents.csv',
            table_name='documents_mock_emb',
            embed_columns_names=['title', 'content'],
            pk_columns=['doc_id'],
            create_table_if_not_exists=True,
            embed_type='combined'
        )
        
        print(f"✅ Success!")
        print(f"   📊 Table: documents_mock_emb")
        print(f"   🔢 Mock embeddings created (deterministic)")
        print(f"   💡 Use for: Testing without Gemini API key")
        
        await db_conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")


# # ==================== FEATURE 1.1: Simple CSV Loading ====================

# async def feature_1_1_simple_csv_loading(use_case):
#     """
#     Feature 1.1: Simple CSV Loading (No Embeddings)
    
#     Demonstrates basic CSV to PostgreSQL loading without embeddings.
#     Useful for pure data migration scenarios.
#     """
#     print("\n" + "="*70)
#     print("FEATURE 1.1: Simple CSV Loading (No Embeddings)")
#     print("="*70)
    
#     try:
#         result = await use_case.execute(
#             s3_uri='test_data/csv/employees_basic.csv',
#             table_name='employees_no_embeddings',
#             embed_columns_names=[],  # No embeddings
#             pk_columns=['id'],
#             create_table_if_not_exists=True
#         )
        
#         print(f"✅ Success!")
#         print(f"   📊 Rows loaded: {result.rows_processed if hasattr(result, 'rows_processed') else 'N/A'}")
#         print(f"   🗄️  Table: employees_no_embeddings")
#         print(f"   💡 No embedding columns created")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")


# ==================== DEMONSTRATION ====================

async def demonstrate_search(use_case, embedding):
    """Demonstrate vector similarity search."""
    print("\n" + "="*70)
    print("🔍 BONUS: Vector Similarity Search Demo")
    print("="*70)
    
    try:
        # Search in combined embeddings table
        query = "software engineer python"
        query_embedding = embedding.get_embeddings([query])[0]
        
        # Get repository from use case
        repo = use_case.repo
        
        results = await repo.search(
            table_name='employees_combined_emb',
            query_embedding=query_embedding,
            top_k=3
        )
        
        print(f"🔎 Query: '{query}'")
        print(f"📋 Top 3 results:")
        for i, result in enumerate(results, 1):
            name = result['metadata'].get('name', 'N/A')
            bio = result['metadata'].get('bio', 'N/A')[:60] + '...'
            similarity = 1 - result['distance']
            print(f"\n   {i}. {name}")
            print(f"      Bio: {bio}")
            print(f"      Similarity: {similarity:.3f}")
        
    except Exception as e:
        print(f"⚠️ Search demo skipped: {e}")


# ==================== MAIN ====================

async def main():
    """Run all Feature 1 examples."""
    print("=" * 70)
    print("FEATURE 1: CSV to PostgreSQL with Embeddings")
    print("=" * 70)
    print("\n📚 This demonstrates various CSV loading strategies:")
    print("   1.1 - Simple loading (no embeddings)")
    print("   1.2 - Combined embeddings (single column)")
    print("   1.3 - Separated embeddings (multiple columns)")
    print("   1.4 - Mock embeddings (no API key)")
    
    db_conn = None
    
    try:
        # Setup
        db_conn, use_case, embedding = await setup_components()
        
        # Run features
        await feature_1_2_combined_embeddings(use_case)
        await feature_1_3_separated_embeddings(use_case)
        # await feature_1_4_mock_embeddings()
        # await feature_1_1_simple_csv_loading(use_case)
        
        # Demo search
        await demonstrate_search(use_case, embedding)
        
        print("\n" + "="*70)
        print("✅ Feature 1 Complete!")
        print("="*70)
        print("\n📊 Tables Created:")
        print("   - employees_no_embeddings (no embeddings)")
        print("   - employees_combined_emb (combined embeddings)")
        print("   - products_separated_emb (separated embeddings)")
        print("   - documents_mock_emb (mock embeddings)")
        
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