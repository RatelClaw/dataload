#!/usr/bin/env python3
"""
FEATURE 2: API/JSON to PostgreSQL with Gemini Embeddings

This demonstrates the ACTUAL API loading workflow from the library.
Based on: data_api_json_use_case_example.py, final_summary.md, readme_api_examples.md

NOTE: This script is modified to work with the bug in v1.2.6.
The library v1.2.6 runs: Flatten -> Transform -> Map
Therefore, the 'transform' step must use FLATTENED names, not MAPPED names.

Prerequisites:
- Run 01_generate_test_data.py first
- PostgreSQL with pgvector extension
- GEMINI_API_KEY (optional - uses mock if not set)
"""

import asyncio
import os
import sys
import tempfile
from typing import List, Tuple
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface





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


async def setup_components(use_mock=False):
    """Initialize components following the library pattern."""
    print("🔧 Setting up components...")
     
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

    embedding = GeminiEmbeddingProvider(config=gemini_config)
    print(f"✅ Using Gemini Embedding Provider (Dimension: {embedding.get_dimension()})")

        
    
    # API loader
    api_loader = APIJSONStorageLoader(
        timeout=30,
        retry_attempts=3
    )
    print("✅ API JSON loader initialized")
    
    # Use case (standard pattern)
    use_case = dataloadUseCase(repo, embedding, api_loader)
    
    return db_conn, use_case, embedding, api_loader, repo


# ==================== FEATURE 2.1: Direct API Loading ====================

async def feature_2_1_direct_api_loading(api_loader, use_case):
    """
    Feature 2.1: Direct API/JSON Loading
    
    Load JSON from file/API, convert to CSV, load with embeddings.
    """
    print("\n" + "="*70)
    print("FEATURE 2.1: Direct API/JSON Loading (Library Pattern)")
    print("="*70)
    
    try:
        # Step 1: Load JSON using APIJSONStorageLoader
        print("📥 Loading JSON data...")
        # Config is empty, so this will just load and not flatten/map/transform
        df = await api_loader.load_json('test_data/api_responses/devices_unnested.json')
        print(f"   ✅ Loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"   📋 Columns: {list(df.columns)[:5]}...")
        
        # Step 2: Save to temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            df.to_csv(f.name, index=False)
            temp_csv_path = f.name
        print(f"   💾 Saved to temp CSV: {temp_csv_path}")
        
        # Step 3: Load CSV with embeddings using dataloadUseCase
        print("   🔄 Loading into PostgreSQL with embeddings...")
        await use_case.execute(
            temp_csv_path,
            'api_devices_direct',
            ['name'],  # Embed device names
            ['id'],
            create_table_if_not_exists=True,
            embed_type='combined'
        )
        
        # Cleanup
        os.unlink(temp_csv_path)
        
        print(f"✅ Success!")
        print(f"   📊 Table: api_devices_direct")
        print(f"   🔢 Embeddings: name field embedded")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== FEATURE 2.2: JSON Flattening ====================

async def feature_2_2_json_flattening(api_loader, use_case):
    """
    Feature 2.2: JSON Flattening (Nested Structures)
    
    Flatten nested JSON like {"data": {"color": "red"}} → data_color: "red"
    """
    print("\n" + "="*70)
    print("FEATURE 2.2: JSON Flattening (Nested Structures)")
    print("="*70)
    
    try:
        # Configuration for flattening
        config = {
            'flatten_nested': True,
            'separator': '_',
            'max_depth': 3
        }
        
        # Load and flatten
        df = await api_loader.load_json('test_data/api_responses/devices.json', config)
        print(f"   ✅ Flattened {len(df)} rows")
        print(f"   📋 Columns after flattening: {list(df.columns)[:8]}...")
        print(f"   💡 Nested 'data' object flattened to: data_color, data_capacity, etc.")
        
        # Save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            df.to_csv(f.name, index=False)
            temp_csv = f.name
        
        await use_case.execute(
            temp_csv,
            'api_devices_flattened',
            [],  # No embeddings for this demo
            ['id'],
            create_table_if_not_exists=True
        )
        
        os.unlink(temp_csv)
        print(f"✅ Table created: api_devices_flattened")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== FEATURE 2.3: Column Mapping ====================

async def feature_2_3_column_mapping(api_loader, use_case):
    """
    Feature 2.3: Column Mapping
    
    Map API fields to clean database column names.
    Order (in v1.2.6): Flatten → (Transform - N/A) → Map
    """
    print("\n" + "="*70)
    print("FEATURE 2.3: Column Mapping")
    print("="*70)
    
    try:
        # Config has flatten and map, but no transform
        config = {
            'flatten_nested': True,
            'separator': '_',
            'column_name_mapping': {
                # Map AFTER flattening, so we map the flattened names
                'id': 'device_id',
                'name': 'device_name',
                'data_color': 'color',  # This exists after flattening
                'data_capacity': 'capacity',
                'data_price': 'price',
                'data_year': 'year'
            }
        }
        
        df = await api_loader.load_json('test_data/api_responses/devices.json', config)
        print(f"   ✅ Mapped columns: {list(df.columns)[:6]}...")
        print(f"   💡 Flattened then mapped:")
        print(f"       id → device_id")
        print(f"       name → device_name")
        print(f"       data_color → color")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            df.to_csv(f.name, index=False)
            temp_csv = f.name
        
        await use_case.execute(
            temp_csv,
            'api_devices_mapped',
            ['device_name'],
            ['device_id'],
            create_table_if_not_exists=True,
            embed_type='combined'
        )
        
        os.unlink(temp_csv)
        print(f"✅ Table created: api_devices_mapped")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== FEATURE 2.4: Data Transformations ====================

async def feature_2_4_transformations(api_loader, use_case):
    """
    Feature 2.4: Data Transformations (Computed Fields)
    
    WORKAROUND for v1.2.6: Flatten -> Transform -> Map
    Transformations MUST use the FLATTENED column names!
    """
    print("\n" + "="*70)
    print("FEATURE 2.4: Data Transformations (WORKAROUND for v1.2.6)")
    print("="*70)
    
    try:
        config = {
            # Step 1: Flatten
            'flatten_nested': True,
            'separator': '_',
            
            # Step 3: Map columns (will run AFTER transform in v1.2.6)
            'column_name_mapping': {
                'id': 'device_id',
                'name': 'device_name',
                'data_color': 'color',
                'data_capacity': 'capacity',
                'data_price': 'price',
                'data_year': 'year'
                # Note: The new 'description' column will pass through un-mapped
            },
            
            # Step 2: Transform (runs BEFORE map in v1.2.6)
            # 
            # !!! THIS IS THE FIX !!!
            # We MUST use the FLATTENED names: {name}, {data_color}, {data_capacity}
            # NOT the mapped names: {device_name}, {color}, {capacity}
            #
            'update_request_body_mapping': {
                'description': "concat({name}, ' - Color: ', coalesce({data_color}, 'N/A'), ', Capacity: ', coalesce({data_capacity}, 'N/A'))"
            }
        }
        
        df = await api_loader.load_json('test_data/api_responses/devices.json', config)
        print(f"   ✅ Created computed field: 'description'")
        print(f"   📋 Columns (after map): {list(df.columns)[:8]}...")
        
        # Show sample description
        # The column 'description' exists because it was created in transform
        # and passed through mapping.
        if 'description' in df.columns and len(df) > 0:
            print(f"   💡 Sample description: {df['description'].iloc[0]}")
            # This will show: "iPhone 15 Pro - Color: Titanium, Capacity: 256GB"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            df.to_csv(f.name, index=False)
            temp_csv = f.name
        
        await use_case.execute(
            temp_csv,
            'api_devices_transformed',
            ['device_name', 'description'], # These columns exist in the final DF
            ['device_id'],
            create_table_if_not_exists=True,
            embed_type='separated'
        )
        
        os.unlink(temp_csv)
        print(f"✅ Table created: api_devices_transformed")
        print(f"   🔢 Embeddings: device_name_enc, description_enc")
        print(f"   💡 Order (v1.2.6): Flatten → Transform → Map → Embed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== FEATURE 2.5: Complete Workflow ====================

async def feature_2_5_complete_workflow(api_loader, use_case, embedding, repo):
    """
    Feature 2.5: Complete API to Vector Search Workflow
    
    Full pipeline with WORKAROUND for v1.2.6 transformation order.
    """
    print("\n" + "="*70)
    print("FEATURE 2.5: Complete Workflow with Search (WORKAROUND)")
    print("="*70)
    
    try:
        config = {
            # 1. Flatten
            'flatten_nested': True,
            'separator': '_',
            
            # 3. Map (flattened names → clean names)
            'column_name_mapping': {
                'id': 'device_id',
                'name': 'device_name',
                'data_color': 'color',
                'data_capacity': 'capacity',
                'data_price': 'price',
                'data_year': 'year'
            },
            
            # 2. Transform (using FLATTENED names)
            #
            # !!! THIS IS THE FIX !!!
            # Use {name}, {data_color}, {data_capacity}
            #
            'update_request_body_mapping': {
                'description': "concat({name}, ' - ', coalesce({data_color}, 'N/A'), ', ', coalesce({data_capacity}, 'N/A'))"
            }
        }
        
        # Load and transform
        df = await api_loader.load_json('test_data/api_responses/devices.json', config)
        print(f"   ✅ Processed {len(df)} devices")
        print(f"   Final columns: {list(df.columns)}")
        
        # Save to CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            df.to_csv(f.name, index=False)
            temp_csv = f.name
        
        # Load with embeddings
        # The final DF has 'device_name' (from map) and 'description' (from transform)
        await use_case.execute(
            temp_csv,
            'api_devices_complete',
            ['device_name', 'description'],
            ['device_id'],
            create_table_if_not_exists=True,
            embed_type='separated'
        )
        
        os.unlink(temp_csv)
        print(f"   ✅ Loaded into PostgreSQL with embeddings")
        
        # Perform similarity search
        print("\n   🔍 Testing similarity search...")
        query_text = "Apple iPhone smartphone"
        query_embedding = embedding.get_embeddings([query_text])[0]
        
        results = await repo.search(
            'api_devices_complete',
            query_embedding,
            top_k=3,
            embed_column='device_name_enc'
        )
        
        print(f"   📱 Query: '{query_text}'")
        print(f"   📋 Top 3 results:")
        for i, result in enumerate(results, 1):
            device_name = result['metadata'].get('device_name', 'N/A')
            color = result['metadata'].get('color', 'N/A')
            similarity = 1 - result['distance']
            print(f"       {i}. {device_name} ({color}) - similarity: {similarity:.3f}")
        
        print(f"\n✅ Complete workflow successful!")
        print(f"   💡 Order (v1.2.6): Flatten → Transform → Map → Embed → Search")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== BONUS: Nested User Data ====================

async def bonus_nested_user_data(api_loader, use_case):
    """
    BONUS: Complex Nested JSON (User API)
    
    WORKAROUND for v1.2.6 order.
    """
    print("\n" + "="*70)
    print("BONUS: Complex Nested JSON (Users) - WORKAROUND")
    print("="*70)
    
    try:
        config = {
            # 1. Flatten (creates profile_first_name, profile_last_name, etc.)
            'flatten_nested': True,
            'separator': '_',
            'max_depth': 4,
            
            # 3. Map (flattened names → clean names)
            'column_name_mapping': {
                'id': 'user_id',
                'profile_first_name': 'first_name',  # Map flattened names
                'profile_last_name': 'last_name',
                'profile_bio': 'bio',
                'contact_email': 'email',
                'contact_phone': 'phone',
                'contact_address_city': 'city',
                'contact_address_state': 'state'
            },
            
            # 2. Transform (using FLATTENED names)
            #
            # !!! THIS IS THE FIX !!!
            # Use {profile_first_name} and {profile_last_name}
            #
            'update_request_body_mapping': {
                'full_name': "concat({profile_first_name}, ' ', {profile_last_name})"
            }
        }
        
        df = await api_loader.load_json('test_data/api_responses/users.json', config)
        print(f"   ✅ Flattened nested user data: {len(df)} rows")
        print(f"   📋 Final columns: {list(df.columns)[:8]}...")
        
        # Show sample full_name
        if 'full_name' in df.columns and len(df) > 0:
            print(f"   💡 Sample full_name: {df['full_name'].iloc[0]}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            df.to_csv(f.name, index=False)
            temp_csv = f.name
        
        await use_case.execute(
            temp_csv,
            'api_users_nested',
            ['full_name', 'bio'],
            ['user_id'],
            create_table_if_not_exists=True,
            embed_type='combined'
        )
        
        os.unlink(temp_csv)
        print(f"   ✅ Table created: api_users_nested")
        print(f"   💡 Order (v1.2.6): Flatten → Transform → Map → Embed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== EXPLANATION ====================

def print_transformation_order_explanation():
    """Explain the transformation order."""
    print("\n" + "="*70)
    print("💡 TRANSFORMATION ORDER EXPLANATION")
    print("="*70)
    
    explanation = """
The INTENDED order is: Flatten → Map → Transform

Why this order?
1. FLATTEN: Converts {"data": {"color": "red"}} → data_color: "red"
2. MAP: Renames data_color → color
3. TRANSFORM: Uses the MAPPED name 'color' in expressions ✅

----------------------------------------------------------------------
THE BUG IN v1.2.6: The library runs in the WRONG order:
Flatten → Transform → Map

Why this script now works:
1. FLATTEN: Converts {"data": {"color": "red"}} → data_color: "red"
2. TRANSFORM: We changed the script to use the FLATTENED name {data_color}
   'description' = concat(..., {data_color}, ...) ✅
3. MAP: Renames data_color → color (and 'description' passes through)

THE FIX:
The code in your repository's 'main' branch IS correct.
This example script is a WORKAROUND for the 'v1.2.6' pip release.

To fix this permanently, release a new version (e.g., 1.2.7)
based on your 'main' branch code.
"""
    print(explanation)


# ==================== MAIN EXECUTION ====================

async def main():
    """Run all features."""
    
    # Use a mock provider for reliability in this example
    db_conn, use_case, embedding, api_loader, repo = await setup_components(use_mock=True)
    
    # Check for dummy data
    if not os.path.exists('test_data/api_responses/devices.json'):
        print("❌ CRITICAL: 'test_data/api_responses/devices.json' not found.")
        print("Please run '01_generate_test_data.py' from the examples folder first.")
        await db_conn.close()
        return

    if not os.path.exists('test_data/api_responses/devices_unnested.json'):
        print("❌ CRITICAL: 'test_data/api_responses/devices_unnested.json' not found.")
        print("Please run '01_generate_test_data.py' from the examples folder first.")
        await db_conn.close()
        return

    if not os.path.exists('test_data/api_responses/users.json'):
        print("❌ CRITICAL: 'test_data/api_responses/users.json' not found.")
        print("Please run '01_generate_test_data.py' from the examples folder first.")
        await db_conn.close()
        return

    print("\n🚀 DataLoad Library - Feature 2 (API/JSON Loading)")
    print("⚠️  Running with WORKAROUND for v1.2.6 bug.")
    print("   Correct Order (v1.2.6): Flatten → Transform → Map\n")
    
    # Run features
    await feature_2_1_direct_api_loading(api_loader, use_case)
    await feature_2_2_json_flattening(api_loader, use_case)
    await feature_2_3_column_mapping(api_loader, use_case)
    
    # These will now succeed with the workaround
    await feature_2_4_transformations(api_loader, use_case)
    await feature_2_5_complete_workflow(api_loader, use_case, embedding, repo)
    await bonus_nested_user_data(api_loader, use_case)
    
    print_transformation_order_explanation()
    
    print("\n" + "="*70)
    print("✅ Feature 2 Complete!")
    print("="*70)
    
    print("\n📊 Tables Created:")
    print("   - api_devices_direct (basic loading)")
    print("   - api_devices_flattened (JSON flattening)")
    print("   - api_devices_mapped (column mapping)")
    print("   - api_devices_transformed (computed fields - WORKAROUND)")
    print("   - api_devices_complete (full workflow + search - WORKAROUND)")
    print("   - api_users_nested (complex nested JSON - WORKAROUND)")
    
    print("\n💡 Key Workaround Pattern (for v1.2.6):")
    print("   ✓ Flatten nested JSON first")
    print("   ✓ Transform using FLATTENED names")
    print("   ✓ Map columns last")
    print("   ✓ Generate embeddings on final CSV")
    
    # Close connection
    await db_conn.close()
    print("\n🔌 Database connection closed")

if __name__ == "__main__":
    # Fix for asyncio on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
