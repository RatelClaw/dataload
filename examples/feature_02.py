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
from typing import List
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface


# Mock Provider for testing without API key
class SimpleMockProvider(EmbeddingProviderInterface):
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        import hashlib
        embeddings = []
        for text in texts:
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            embedding = [(hash_val + i) % 100 / 100.0 for i in range(self.embedding_dim)]
            embeddings.append(embedding)
        return embeddings


async def setup_components(use_mock=False):
    """Initialize components following the library pattern."""
    print("🔧 Setting up components...")
    
    # Database connection
    db_conn = DBConnection()
    await db_conn.initialize()
    repo = PostgresDataRepository(db_conn)
    print("✅ Database connected")
    
    # Embedding provider
    if use_mock or not os.getenv('GEMINI_API_KEY'):
        embedding = SimpleMockProvider(embedding_dim=768)
        print("✅ Using Mock Embedding Provider (768-dim, Gemini-compatible)")
    else:
        embedding = GeminiEmbeddingProvider()
        print("✅ Using Gemini Embedding Provider")
    
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

# #!/usr/bin/env python3
# """
# FEATURE 2: API/JSON to PostgreSQL with Gemini Embeddings

# This demonstrates the ACTUAL API loading workflow from the library.
# Based on: data_api_json_use_case_example.py, final_summary.md, readme_api_examples.md

# The correct order is:
# 1. Flatten nested JSON
# 2. Apply column mapping
# 3. Apply transformations (AFTER mapping, so mapped names are available)
# 4. Generate embeddings

# Prerequisites:
# - Run 01_generate_test_data.py first
# - PostgreSQL with pgvector extension
# - GEMINI_API_KEY (optional - uses mock if not set)
# """

# import asyncio
# import os
# import sys
# import tempfile
# from typing import List
# from dataload.infrastructure.db.db_connection import DBConnection
# from dataload.infrastructure.db.data_repository import PostgresDataRepository
# from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
# from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
# from dataload.application.use_cases.data_loader_use_case import dataloadUseCase
# from dataload.interfaces.embedding_provider import EmbeddingProviderInterface


# # Mock Provider for testing without API key
# class SimpleMockProvider(EmbeddingProviderInterface):
#     def __init__(self, embedding_dim: int = 768):
#         self.embedding_dim = embedding_dim
    
#     def get_embeddings(self, texts: List[str]) -> List[List[float]]:
#         import hashlib
#         embeddings = []
#         for text in texts:
#             hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
#             embedding = [(hash_val + i) % 100 / 100.0 for i in range(self.embedding_dim)]
#             embeddings.append(embedding)
#         return embeddings


# async def setup_components(use_mock=False):
#     """Initialize components following the library pattern."""
#     print("🔧 Setting up components...")
    
#     # Database connection
#     db_conn = DBConnection()
#     await db_conn.initialize()
#     repo = PostgresDataRepository(db_conn)
#     print("✅ Database connected")
    
#     # Embedding provider
#     if use_mock or not os.getenv('GEMINI_API_KEY'):
#         embedding = SimpleMockProvider(embedding_dim=768)
#         print("✅ Using Mock Embedding Provider (768-dim, Gemini-compatible)")
#     else:
#         embedding = GeminiEmbeddingProvider()
#         print("✅ Using Gemini Embedding Provider")
    
#     # API loader
#     api_loader = APIJSONStorageLoader(
#         timeout=30,
#         retry_attempts=3
#     )
#     print("✅ API JSON loader initialized")
    
#     # Use case (standard pattern)
#     use_case = dataloadUseCase(repo, embedding, api_loader)
    
#     return db_conn, use_case, embedding, api_loader, repo


# # ==================== FEATURE 2.1: Direct API Loading ====================

# async def feature_2_1_direct_api_loading(api_loader, use_case):
#     """
#     Feature 2.1: Direct API/JSON Loading
    
#     Load JSON from file/API, convert to CSV, load with embeddings.
#     """
#     print("\n" + "="*70)
#     print("FEATURE 2.1: Direct API/JSON Loading (Library Pattern)")
#     print("="*70)
    
#     try:
#         # Step 1: Load JSON using APIJSONStorageLoader
#         print("📥 Loading JSON data...")
#         df = await api_loader.load_json('test_data/api_responses/devices.json')
#         print(f"   ✅ Loaded {len(df)} rows with {len(df.columns)} columns")
#         print(f"   📋 Columns: {list(df.columns)[:5]}...")
        
#         # Step 2: Save to temporary CSV
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
#             df.to_csv(f.name, index=False)
#             temp_csv_path = f.name
#         print(f"   💾 Saved to temp CSV: {temp_csv_path}")
        
#         # Step 3: Load CSV with embeddings using dataloadUseCase
#         print("   🔄 Loading into PostgreSQL with embeddings...")
#         await use_case.execute(
#             temp_csv_path,
#             'api_devices_direct',
#             ['name'],  # Embed device names
#             ['id'],
#             create_table_if_not_exists=True,
#             embed_type='combined'
#         )
        
#         # Cleanup
#         os.unlink(temp_csv_path)
        
#         print(f"✅ Success!")
#         print(f"   📊 Table: api_devices_direct")
#         print(f"   🔢 Embeddings: name field embedded")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")


# # ==================== FEATURE 2.2: JSON Flattening ====================

# async def feature_2_2_json_flattening(api_loader, use_case):
#     """
#     Feature 2.2: JSON Flattening (Nested Structures)
    
#     Flatten nested JSON like {"data": {"color": "red"}} → data_color: "red"
#     """
#     print("\n" + "="*70)
#     print("FEATURE 2.2: JSON Flattening (Nested Structures)")
#     print("="*70)
    
#     try:
#         # Configuration for flattening
#         config = {
#             'flatten_nested': True,
#             'separator': '_',
#             'max_depth': 3
#         }
        
#         # Load and flatten
#         df = await api_loader.load_json('test_data/api_responses/devices.json', config)
#         print(f"   ✅ Flattened {len(df)} rows")
#         print(f"   📋 Columns after flattening: {list(df.columns)[:8]}...")
#         print(f"   💡 Nested 'data' object flattened to: data_color, data_capacity, etc.")
        
#         # Save and load
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
#             df.to_csv(f.name, index=False)
#             temp_csv = f.name
        
#         await use_case.execute(
#             temp_csv,
#             'api_devices_flattened',
#             [],  # No embeddings for this demo
#             ['id'],
#             create_table_if_not_exists=True
#         )
        
#         os.unlink(temp_csv)
#         print(f"✅ Table created: api_devices_flattened")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")


# # ==================== FEATURE 2.3: Column Mapping ====================

# async def feature_2_3_column_mapping(api_loader, use_case):
#     """
#     Feature 2.3: Column Mapping
    
#     Map API fields to clean database column names.
#     Order: Flatten → Map columns
#     """
#     print("\n" + "="*70)
#     print("FEATURE 2.3: Column Mapping")
#     print("="*70)
    
#     try:
#         # CORRECT: First flatten, then map
#         config = {
#             'flatten_nested': True,
#             'separator': '_',
#             'column_name_mapping': {
#                 # Map AFTER flattening, so we map the flattened names
#                 'id': 'device_id',
#                 'name': 'device_name',
#                 'data_color': 'color',  # This exists after flattening
#                 'data_capacity': 'capacity',
#                 'data_price': 'price',
#                 'data_year': 'year'
#             }
#         }
        
#         df = await api_loader.load_json('test_data/api_responses/devices.json', config)
#         print(f"   ✅ Mapped columns: {list(df.columns)[:6]}...")
#         print(f"   💡 Flattened then mapped:")
#         print(f"      id → device_id")
#         print(f"      name → device_name")
#         print(f"      data_color → color")
        
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
#             df.to_csv(f.name, index=False)
#             temp_csv = f.name
        
#         await use_case.execute(
#             temp_csv,
#             'api_devices_mapped',
#             ['device_name'],
#             ['device_id'],
#             create_table_if_not_exists=True,
#             embed_type='combined'
#         )
        
#         os.unlink(temp_csv)
#         print(f"✅ Table created: api_devices_mapped")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")


# # ==================== FEATURE 2.4: Data Transformations ====================

# async def feature_2_4_transformations(api_loader, use_case):
#     """
#     Feature 2.4: Data Transformations (Computed Fields)
    
#     CRITICAL ORDER: Flatten → Map → Transform
#     Transformations use the MAPPED column names!
    
#     From data_api_json_use_case_example.py and final_summary.md
#     """
#     print("\n" + "="*70)
#     print("FEATURE 2.4: Data Transformations (CORRECT ORDER)")
#     print("="*70)
    
#     try:
#         # CORRECT: Flatten → Map → Transform (using mapped names)
#         config = {
#             # Step 1: Flatten
#             'flatten_nested': True,
#             'separator': '_',
            
#             # Step 2: Map columns
#             'column_name_mapping': {
#                 'id': 'device_id',
#                 'name': 'device_name',
#                 'data_color': 'color',
#                 'data_capacity': 'capacity',
#                 'data_price': 'price'
#             },
            
#             # Step 3: Transform using MAPPED names
#             'update_request_body_mapping': {
#                 # Use mapped column names: device_name, color, capacity
#                 'description': "concat({device_name}, ' - Color: ', coalesce({color}, 'N/A'), ', Capacity: ', coalesce({capacity}, 'N/A'))"
#             }
#         }
        
#         df = await api_loader.load_json('test_data/api_responses/devices.json', config)
#         print(f"   ✅ Created computed field: 'description'")
#         print(f"   📋 Columns: {list(df.columns)[:8]}...")
        
#         # Show sample description
#         if 'description' in df.columns and len(df) > 0:
#             print(f"   💡 Sample description: {df['description'].iloc[0]}")
        
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
#             df.to_csv(f.name, index=False)
#             temp_csv = f.name
        
#         await use_case.execute(
#             temp_csv,
#             'api_devices_transformed',
#             ['device_name', 'description'],
#             ['device_id'],
#             create_table_if_not_exists=True,
#             embed_type='separated'
#         )
        
#         os.unlink(temp_csv)
#         print(f"✅ Table created: api_devices_transformed")
#         print(f"   🔢 Embeddings: device_name_enc, description_enc")
#         print(f"   💡 Order: Flatten → Map → Transform → Embed")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()


# # ==================== FEATURE 2.5: Complete Workflow ====================

# async def feature_2_5_complete_workflow(api_loader, use_case, embedding, repo):
#     """
#     Feature 2.5: Complete API to Vector Search Workflow
    
#     Full pipeline with CORRECT transformation order.
#     Based on: final_summary.md and comprehensive_api_to_vector_example.py
#     """
#     print("\n" + "="*70)
#     print("FEATURE 2.5: Complete Workflow with Search")
#     print("="*70)
    
#     try:
#         # Complete configuration with CORRECT order
#         config = {
#             # 1. Flatten
#             'flatten_nested': True,
#             'separator': '_',
            
#             # 2. Map (flattened names → clean names)
#             'column_name_mapping': {
#                 'id': 'device_id',
#                 'name': 'device_name',
#                 'data_color': 'color',
#                 'data_capacity': 'capacity',
#                 'data_price': 'price',
#                 'data_year': 'year'
#             },
            
#             # 3. Transform (using MAPPED names)
#             'update_request_body_mapping': {
#                 'description': "concat({device_name}, ' - ', coalesce({color}, 'N/A'), ', ', coalesce({capacity}, 'N/A'))"
#             }
#         }
        
#         # Load and transform
#         df = await api_loader.load_json('test_data/api_responses/devices.json', config)
#         print(f"   ✅ Processed {len(df)} devices")
        
#         # Save to CSV
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
#             df.to_csv(f.name, index=False)
#             temp_csv = f.name
        
#         # Load with embeddings
#         await use_case.execute(
#             temp_csv,
#             'api_devices_complete',
#             ['device_name', 'description'],
#             ['device_id'],
#             create_table_if_not_exists=True,
#             embed_type='separated'
#         )
        
#         os.unlink(temp_csv)
#         print(f"   ✅ Loaded into PostgreSQL with embeddings")
        
#         # Perform similarity search
#         print("\n   🔍 Testing similarity search...")
#         query_text = "Apple iPhone smartphone"
#         query_embedding = embedding.get_embeddings([query_text])[0]
        
#         results = await repo.search(
#             'api_devices_complete',
#             query_embedding,
#             top_k=3,
#             embed_column='device_name_enc'
#         )
        
#         print(f"   📱 Query: '{query_text}'")
#         print(f"   📋 Top 3 results:")
#         for i, result in enumerate(results, 1):
#             device_name = result['metadata'].get('device_name', 'N/A')
#             color = result['metadata'].get('color', 'N/A')
#             similarity = 1 - result['distance']
#             print(f"      {i}. {device_name} ({color}) - similarity: {similarity:.3f}")
        
#         print(f"\n✅ Complete workflow successful!")
#         print(f"   💡 Order: Flatten → Map → Transform → Embed → Search")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()


# # ==================== BONUS: Nested User Data ====================

# async def bonus_nested_user_data(api_loader, use_case):
#     """
#     BONUS: Complex Nested JSON (User API)
    
#     CORRECT ORDER for nested data.
#     """
#     print("\n" + "="*70)
#     print("BONUS: Complex Nested JSON (Users) - CORRECT ORDER")
#     print("="*70)
    
#     try:
#         config = {
#             # 1. Flatten (creates profile_first_name, profile_last_name, etc.)
#             'flatten_nested': True,
#             'separator': '_',
#             'max_depth': 4,
            
#             # 2. Map (flattened names → clean names)
#             'column_name_mapping': {
#                 'id': 'user_id',
#                 'profile_first_name': 'first_name',  # Map flattened names
#                 'profile_last_name': 'last_name',
#                 'profile_bio': 'bio',
#                 'contact_email': 'email',
#                 'contact_phone': 'phone',
#                 'contact_address_city': 'city',
#                 'contact_address_state': 'state'
#             },
            
#             # 3. Transform (using MAPPED names)
#             'update_request_body_mapping': {
#                 'full_name': "concat({first_name}, ' ', {last_name})"  # Use mapped names
#             }
#         }
        
#         df = await api_loader.load_json('test_data/api_responses/users.json', config)
#         print(f"   ✅ Flattened nested user data: {len(df)} rows")
#         print(f"   📋 Final columns: {list(df.columns)[:8]}...")
        
#         # Show sample full_name
#         if 'full_name' in df.columns and len(df) > 0:
#             print(f"   💡 Sample full_name: {df['full_name'].iloc[0]}")
        
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
#             df.to_csv(f.name, index=False)
#             temp_csv = f.name
        
#         await use_case.execute(
#             temp_csv,
#             'api_users_nested',
#             ['full_name', 'bio'],
#             ['user_id'],
#             create_table_if_not_exists=True,
#             embed_type='combined'
#         )
        
#         os.unlink(temp_csv)
#         print(f"   ✅ Table created: api_users_nested")
#         print(f"   💡 Order: Flatten → Map → Transform → Embed")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()


# # ==================== EXPLANATION ====================

# def print_transformation_order_explanation():
#     """Explain the correct transformation order."""
#     print("\n" + "="*70)
#     print("💡 TRANSFORMATION ORDER EXPLANATION")
#     print("="*70)
    
#     explanation = """
# The CORRECT order is: Flatten → Map → Transform

# Why this order?
# 1. FLATTEN: Converts {"data": {"color": "red"}} → data_color: "red"
# 2. MAP: Renames data_color → color
# 3. TRANSFORM: Uses the MAPPED name 'color' in expressions

# Example:
#   Original JSON: {"data": {"color": "red"}}
  
#   After Flatten: data_color = "red"
#   After Map: color = "red" (data_color renamed)
#   After Transform: description = concat(..., {color}, ...) ✅ WORKS!

# WRONG Order (Transform before Map):
#   After Flatten: data_color = "red"
#   After Transform: Tries to use {color} ❌ DOESN'T EXIST YET!
#   After Map: color = "red" (too late!)

# From the library code (api_json_loader.py):
#   1. df = self._flatten_json(data)
#   2. df = self._apply_column_mapping(df)
#   3. df = self._apply_transformations(df)
  
# This is why transformations use MAPPED column names!
# """
#     print(explanation)


# # ==================== MAIN ====================

# async def main():
#     """Run all Feature 2 examples."""
#     print("=" * 70)
#     print("FEATURE 2: API/JSON to PostgreSQL (CORRECT ORDER)")
#     print("=" * 70)
#     print("\n📚 Based on: data_api_json_use_case_example.py, final_summary.md")
#     print("\n⚠️  CRITICAL: Transformations use MAPPED column names!")
#     print("   Order: Flatten → Map → Transform")
    
#     db_conn = None
    
#     try:
#         # Setup (use mock by default)
#         db_conn, use_case, embedding, api_loader, repo = await setup_components(use_mock=True)
        
#         # Run all features
#         await feature_2_1_direct_api_loading(api_loader, use_case)
#         await feature_2_2_json_flattening(api_loader, use_case)
#         await feature_2_3_column_mapping(api_loader, use_case)
#         await feature_2_4_transformations(api_loader, use_case)
#         await feature_2_5_complete_workflow(api_loader, use_case, embedding, repo)
        
#         # Bonus
#         await bonus_nested_user_data(api_loader, use_case)
        
#         # Explanation
#         print_transformation_order_explanation()
        
#         print("\n" + "="*70)
#         print("✅ Feature 2 Complete!")
#         print("="*70)
#         print("\n📊 Tables Created:")
#         print("   - api_devices_direct (basic loading)")
#         print("   - api_devices_flattened (JSON flattening)")
#         print("   - api_devices_mapped (column mapping)")
#         print("   - api_devices_transformed (computed fields - CORRECT ORDER)")
#         print("   - api_devices_complete (full workflow + search)")
#         print("   - api_users_nested (complex nested JSON)")
        
#         print("\n💡 Key Library Pattern:")
#         print("   ✓ Flatten nested JSON first")
#         print("   ✓ Map flattened names to clean names")
#         print("   ✓ Transform using MAPPED names")
#         print("   ✓ Generate embeddings last")
        
#     except Exception as e:
#         print(f"\n❌ Feature 2 failed: {e}")
#         import traceback
#         traceback.print_exc()
        
#     finally:
#         if db_conn:
#             await db_conn.close()
#             print("\n🔌 Database connection closed")


# if __name__ == "__main__":
#     print("🚀 DataLoad Library - Feature 2 (API/JSON Loading)")
#     print("Correct transformation order: Flatten → Map → Transform\n")
    
#     asyncio.run(main())


# # #!/usr/bin/env python3
# # """
# # FEATURE 2: API/JSON to PostgreSQL with Gemini Embeddings

# # This demonstrates the ACTUAL API loading workflow from the library:
# # 1. Load JSON from API/file using APIJSONStorageLoader
# # 2. Process and transform JSON data
# # 3. Convert to CSV temporarily (library pattern)
# # 4. Load into PostgreSQL with embeddings using dataloadUseCase

# # Based on: simple_api_example.py, working_api_example.py, api_to_postgres_gemini_example.py

# # Prerequisites:
# # - Run 01_generate_test_data.py first
# # - PostgreSQL with pgvector extension
# # - GEMINI_API_KEY (optional - uses mock if not set)
# # """

# # import asyncio
# # import os
# # import sys
# # import tempfile
# # from typing import List
# # from dataload.infrastructure.db.db_connection import DBConnection
# # from dataload.infrastructure.db.data_repository import PostgresDataRepository
# # from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
# # from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
# # from dataload.application.use_cases.data_loader_use_case import dataloadUseCase
# # from dataload.interfaces.embedding_provider import EmbeddingProviderInterface


# # # Mock Provider for testing without API key
# # class SimpleMockProvider(EmbeddingProviderInterface):
# #     def __init__(self, embedding_dim: int = 768):  # Gemini uses 768
# #         self.embedding_dim = embedding_dim
    
# #     def get_embeddings(self, texts: List[str]) -> List[List[float]]:
# #         import hashlib
# #         embeddings = []
# #         for text in texts:
# #             hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
# #             embedding = [(hash_val + i) % 100 / 100.0 for i in range(self.embedding_dim)]
# #             embeddings.append(embedding)
# #         return embeddings


# # async def setup_components(use_mock=False):
# #     """Initialize components following the library pattern."""
# #     print("🔧 Setting up components...")
    
# #     # Database connection (from main_pg_gemni.py pattern)
# #     db_conn = DBConnection()
# #     await db_conn.initialize()
# #     repo = PostgresDataRepository(db_conn)
# #     print("✅ Database connected")
    
# #     # Embedding provider
# #     if use_mock or not os.getenv('GEMINI_API_KEY'):
# #         embedding = SimpleMockProvider(embedding_dim=768)
# #         print("✅ Using Mock Embedding Provider (768-dim, Gemini-compatible)")
# #     else:
# #         embedding = GeminiEmbeddingProvider()
# #         print("✅ Using Gemini Embedding Provider")
    
# #     # API loader (from simple_api_example.py)
# #     api_loader = APIJSONStorageLoader(
# #         timeout=30,
# #         retry_attempts=3
# #     )
# #     print("✅ API JSON loader initialized")
    
# #     # Use case (standard pattern from main_pg_gemni.py)
# #     use_case = dataloadUseCase(repo, embedding, api_loader)
    
# #     return db_conn, use_case, embedding, api_loader, repo


# # # ==================== FEATURE 2.1: Direct API Loading ====================

# # async def feature_2_1_direct_api_loading(api_loader, use_case):
# #     """
# #     Feature 2.1: Direct API/JSON Loading
    
# #     Load JSON from file/API, convert to CSV, load with embeddings.
# #     This is the ACTUAL workflow from simple_api_example.py
# #     """
# #     print("\n" + "="*70)
# #     print("FEATURE 2.1: Direct API/JSON Loading (Library Pattern)")
# #     print("="*70)
    
# #     try:
# #         # Step 1: Load JSON using APIJSONStorageLoader
# #         print("📥 Loading JSON data...")
# #         df = await api_loader.load_json('test_data/api_responses/devices.json')
# #         print(f"   ✅ Loaded {len(df)} rows with {len(df.columns)} columns")
# #         print(f"   📋 Columns: {list(df.columns)[:5]}...")
        
# #         # Step 2: Save to temporary CSV (library pattern from working_api_example.py)
# #         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
# #             df.to_csv(f.name, index=False)
# #             temp_csv_path = f.name
# #         print(f"   💾 Saved to temp CSV: {temp_csv_path}")
        
# #         # Step 3: Load CSV with embeddings using dataloadUseCase
# #         print("   🔄 Loading into PostgreSQL with embeddings...")
# #         await use_case.execute(
# #             temp_csv_path,
# #             'api_devices_direct',
# #             ['name'],  # Embed device names
# #             ['id'],
# #             create_table_if_not_exists=True,
# #             embed_type='combined'
# #         )
        
# #         # Cleanup
# #         os.unlink(temp_csv_path)
        
# #         print(f"✅ Success!")
# #         print(f"   📊 Table: api_devices_direct")
# #         print(f"   🔢 Embeddings: name field embedded")
# #         print(f"   💡 Pattern: JSON → DataFrame → CSV → PostgreSQL with embeddings")
        
# #     except Exception as e:
# #         print(f"❌ Error: {e}")
# #         import traceback
# #         traceback.print_exc()


# # # ==================== FEATURE 2.2: JSON Flattening ====================

# # async def feature_2_2_json_flattening(api_loader, use_case):
# #     """
# #     Feature 2.2: JSON Flattening (Nested Structures)
    
# #     Flatten nested JSON like {"data": {"color": "red"}} → data_color: "red"
# #     Based on simple_api_example.py configuration
# #     """
# #     print("\n" + "="*70)
# #     print("FEATURE 2.2: JSON Flattening (Nested Structures)")
# #     print("="*70)
    
# #     try:
# #         # Configuration for flattening (from simple_api_example.py)
# #         config = {
# #             'flatten_nested': True,
# #             'separator': '_',
# #             'max_depth': 3
# #         }
        
# #         # Load and flatten
# #         df = await api_loader.load_json('test_data/api_responses/devices.json', config)
# #         print(f"   ✅ Flattened {len(df)} rows")
# #         print(f"   📋 Columns after flattening: {list(df.columns)}")
# #         print(f"   💡 Nested 'data' object flattened to: data_color, data_capacity, etc.")
        
# #         # Save and load
# #         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
# #             df.to_csv(f.name, index=False)
# #             temp_csv = f.name
        
# #         await use_case.execute(
# #             temp_csv,
# #             'api_devices_flattened',
# #             [],  # No embeddings for this demo
# #             ['id'],
# #             create_table_if_not_exists=True
# #         )
        
# #         os.unlink(temp_csv)
# #         print(f"✅ Table created: api_devices_flattened")
        
# #     except Exception as e:
# #         print(f"❌ Error: {e}")


# # # ==================== FEATURE 2.3: Column Mapping ====================

# # async def feature_2_3_column_mapping(api_loader, use_case):
# #     """
# #     Feature 2.3: Column Mapping
    
# #     Map API fields to clean database column names.
# #     From api_to_postgres_gemini_example.py
# #     """
# #     print("\n" + "="*70)
# #     print("FEATURE 2.3: Column Mapping")
# #     print("="*70)
    
# #     try:
# #         # Configuration from simple_api_example.py
# #         config = {
# #             'flatten_nested': True,
# #             'separator': '_',
# #             'column_name_mapping': {
# #                 'id': 'device_id',
# #                 'name': 'device_name',
# #                 'data_color': 'color',
# #                 'data_capacity': 'capacity',
# #                 'data_price': 'price',
# #                 'data_year': 'year'
# #             }
# #         }
        
# #         df = await api_loader.load_json('test_data/api_responses/devices.json', config)
# #         print(f"   ✅ Mapped columns: {list(df.columns)}")
# #         print(f"   💡 API fields mapped to clean names:")
# #         print(f"      id → device_id")
# #         print(f"      name → device_name")
# #         print(f"      data_color → color")
        
# #         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
# #             df.to_csv(f.name, index=False)
# #             temp_csv = f.name
        
# #         await use_case.execute(
# #             temp_csv,
# #             'api_devices_mapped',
# #             ['device_name'],
# #             ['device_id'],
# #             create_table_if_not_exists=True,
# #             embed_type='combined'
# #         )
        
# #         os.unlink(temp_csv)
# #         print(f"✅ Table created: api_devices_mapped")
        
# #     except Exception as e:
# #         print(f"❌ Error: {e}")


# # # ==================== FEATURE 2.4: Data Transformations ====================

# # async def feature_2_4_transformations(api_loader, use_case):
# #     """
# #     Feature 2.4: Data Transformations (Computed Fields)
    
# #     Create computed fields using SQL-like expressions.
# #     From simple_api_example.py pattern
# #     """
# #     print("\n" + "="*70)
# #     print("FEATURE 2.4: Data Transformations")
# #     print("="*70)
    
# #     try:
# #         # Full configuration from simple_api_example.py
# #         config = {
# #             'flatten_nested': True,
# #             'separator': '_',
# #             'column_name_mapping': {
# #                 'id': 'device_id',
# #                 'name': 'device_name',
# #                 'data_color': 'color',
# #                 'data_capacity': 'capacity',
# #                 'data_price': 'price'
# #             },
# #             'update_request_body_mapping': {
# #                 # Create description field (from simple_api_example.py)
# #                 'description': "concat({device_name}, ' - Color: ', coalesce({color}, 'N/A'), ', Capacity: ', coalesce({capacity}, 'N/A'))"
# #             }
# #         }
        
# #         df = await api_loader.load_json('test_data/api_responses/devices.json', config)
# #         print(f"   ✅ Created computed field: 'description'")
# #         print(f"   📋 Columns: {list(df.columns)}")
        
# #         # Show sample description
# #         if 'description' in df.columns and len(df) > 0:
# #             print(f"   💡 Sample: {df['description'].iloc[0]}")
        
# #         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
# #             df.to_csv(f.name, index=False)
# #             temp_csv = f.name
        
# #         await use_case.execute(
# #             temp_csv,
# #             'api_devices_transformed',
# #             ['device_name', 'description'],
# #             ['device_id'],
# #             create_table_if_not_exists=True,
# #             embed_type='separated'
# #         )
        
# #         os.unlink(temp_csv)
# #         print(f"✅ Table created: api_devices_transformed")
# #         print(f"   🔢 Embeddings: device_name_enc, description_enc")
        
# #     except Exception as e:
# #         print(f"❌ Error: {e}")


# # # ==================== FEATURE 2.5: Complete Workflow ====================

# # async def feature_2_5_complete_workflow(api_loader, use_case, embedding, repo):
# #     """
# #     Feature 2.5: Complete API to Vector Search Workflow
    
# #     Full pipeline from working_api_example.py and api_to_postgres_gemini_example.py:
# #     1. Load from API
# #     2. Transform data
# #     3. Generate embeddings
# #     4. Store in PostgreSQL
# #     5. Perform similarity search
# #     """
# #     print("\n" + "="*70)
# #     print("FEATURE 2.5: Complete Workflow with Search")
# #     print("="*70)
    
# #     try:
# #         # Complete configuration
# #         config = {
# #             'flatten_nested': True,
# #             'separator': '_',
# #             'column_name_mapping': {
# #                 'id': 'device_id',
# #                 'name': 'device_name',
# #                 'data_color': 'color',
# #                 'data_capacity': 'capacity',
# #                 'data_price': 'price',
# #                 'data_year': 'year'
# #             },
# #             'update_request_body_mapping': {
# #                 'description': "concat({device_name}, ' - ', coalesce({color}, 'N/A'), ', ', coalesce({capacity}, 'N/A'))"
# #             }
# #         }
        
# #         # Load and transform
# #         df = await api_loader.load_json('test_data/api_responses/devices.json', config)
# #         print(f"   ✅ Processed {len(df)} devices")
        
# #         # Save to CSV
# #         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
# #             df.to_csv(f.name, index=False)
# #             temp_csv = f.name
        
# #         # Load with embeddings
# #         await use_case.execute(
# #             temp_csv,
# #             'api_devices_complete',
# #             ['device_name', 'description'],
# #             ['device_id'],
# #             create_table_if_not_exists=True,
# #             embed_type='separated'
# #         )
        
# #         os.unlink(temp_csv)
# #         print(f"   ✅ Loaded into PostgreSQL with embeddings")
        
# #         # Perform similarity search (from api_to_postgres_gemini_example.py)
# #         print("\n   🔍 Testing similarity search...")
# #         query_text = "Apple iPhone smartphone"
# #         query_embedding = embedding.get_embeddings([query_text])[0]
        
# #         results = await repo.search(
# #             'api_devices_complete',
# #             query_embedding,
# #             top_k=3,
# #             embed_column='device_name_enc'
# #         )
        
# #         print(f"   📱 Query: '{query_text}'")
# #         print(f"   📋 Top 3 results:")
# #         for i, result in enumerate(results, 1):
# #             device_name = result['metadata'].get('device_name', 'N/A')
# #             color = result['metadata'].get('color', 'N/A')
# #             similarity = 1 - result['distance']
# #             print(f"      {i}. {device_name} ({color}) - similarity: {similarity:.3f}")
        
# #         print(f"\n✅ Complete workflow successful!")
# #         print(f"   💡 Pattern: API → Transform → Embed → Store → Search")
        
# #     except Exception as e:
# #         print(f"❌ Error: {e}")
# #         import traceback
# #         traceback.print_exc()


# # # ==================== BONUS: Nested User Data ====================

# # async def bonus_nested_user_data(api_loader, use_case):
# #     """
# #     BONUS: Complex Nested JSON (User API)
    
# #     Handle deeply nested structures with multiple levels.
# #     Based on comprehensive_api_to_vector_example.py
# #     """
# #     print("\n" + "="*70)
# #     print("BONUS: Complex Nested JSON (Users)")
# #     print("="*70)
    
# #     try:
# #         config = {
# #             'flatten_nested': True,
# #             'separator': '_',
# #             'max_depth': 4,
# #             'column_name_mapping': {
# #                 'id': 'user_id',
# #                 'profile_first_name': 'first_name',
# #                 'profile_last_name': 'last_name',
# #                 'profile_bio': 'bio',
# #                 'contact_email': 'email',
# #                 'contact_phone': 'phone',
# #                 'contact_address_city': 'city',
# #                 'contact_address_state': 'state'
# #             },
# #             'update_request_body_mapping': {
# #                 'full_name': "concat({first_name}, ' ', {last_name})"
# #             }
# #         }
        
# #         df = await api_loader.load_json('test_data/api_responses/users.json', config)
# #         print(f"   ✅ Flattened nested user data: {len(df)} rows")
# #         print(f"   📋 Flattened fields: {list(df.columns)}")
        
# #         with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
# #             df.to_csv(f.name, index=False)
# #             temp_csv = f.name
        
# #         await use_case.execute(
# #             temp_csv,
# #             'api_users_nested',
# #             ['full_name', 'bio'],
# #             ['user_id'],
# #             create_table_if_not_exists=True,
# #             embed_type='combined'
# #         )
        
# #         os.unlink(temp_csv)
# #         print(f"   ✅ Table created: api_users_nested")
        
# #     except Exception as e:
# #         print(f"❌ Error: {e}")


# # # ==================== MAIN ====================

# # async def main():
# #     """Run all Feature 2 examples."""
# #     print("=" * 70)
# #     print("FEATURE 2: API/JSON to PostgreSQL (ACTUAL Library Pattern)")
# #     print("=" * 70)
# #     print("\n📚 This demonstrates the REAL API loading workflow:")
# #     print("   1. Load JSON using APIJSONStorageLoader")
# #     print("   2. Transform and flatten data")
# #     print("   3. Save to temporary CSV")
# #     print("   4. Load CSV with embeddings using dataloadUseCase")
# #     print("   5. Perform vector similarity search")
# #     print("\n💡 Based on: simple_api_example.py, working_api_example.py")
    
# #     db_conn = None
    
# #     try:
# #         # Setup (use mock by default, works without API key)
# #         db_conn, use_case, embedding, api_loader, repo = await setup_components(use_mock=False)
        
# #         # Run all features
# #         await feature_2_1_direct_api_loading(api_loader, use_case)
# #         await feature_2_2_json_flattening(api_loader, use_case)
# #         await feature_2_3_column_mapping(api_loader, use_case)
# #         await feature_2_4_transformations(api_loader, use_case)
# #         await feature_2_5_complete_workflow(api_loader, use_case, embedding, repo)
        
# #         # Bonus
# #         await bonus_nested_user_data(api_loader, use_case)
        
# #         print("\n" + "="*70)
# #         print("✅ Feature 2 Complete!")
# #         print("="*70)
# #         print("\n📊 Tables Created:")
# #         print("   - api_devices_direct (basic loading)")
# #         print("   - api_devices_flattened (JSON flattening)")
# #         print("   - api_devices_mapped (column mapping)")
# #         print("   - api_devices_transformed (computed fields)")
# #         print("   - api_devices_complete (full workflow + search)")
# #         print("   - api_users_nested (complex nested JSON)")
        
# #         print("\n💡 Key Library Patterns:")
# #         print("   ✓ APIJSONStorageLoader for JSON loading")
# #         print("   ✓ DataFrame → CSV → dataloadUseCase workflow")
# #         print("   ✓ Nested JSON flattening with config")
# #         print("   ✓ Column mapping for clean schemas")
# #         print("   ✓ Computed fields with SQL-like expressions")
# #         print("   ✓ Vector search with PostgreSQL pgvector")
        
# #     except Exception as e:
# #         print(f"\n❌ Feature 2 failed: {e}")
# #         import traceback
# #         traceback.print_exc()
        
# #     finally:
# #         if db_conn:
# #             await db_conn.close()
# #             print("\n🔌 Database connection closed")


# # if __name__ == "__main__":
# #     print("🚀 DataLoad Library - Feature 2 (API/JSON Loading)")
# #     print("Based on actual library examples from the repository\n")
    
# #     asyncio.run(main())