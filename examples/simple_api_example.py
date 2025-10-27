#!/usr/bin/env python3
"""
Simple API Example using restful-api.dev

This example demonstrates loading data from the specific API endpoint you mentioned:
https://api.restful-api.dev/objects

It shows:
1. Loading data from the API
2. Processing nested JSON structures
3. Generating embeddings with Gemini
4. Storing in PostgreSQL with vector search

This is a simplified version of the comprehensive example.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.application.use_cases.data_api_json_use_case import DataAPIJSONUseCase
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
from examples.mock_embedding_provider import MockEmbeddingProvider
from dataload.config import logger


async def simple_api_to_vector_example():
    """
    Simple example: Load device data from API and store with embeddings.
    
    This function demonstrates the complete workflow:
    1. Connect to PostgreSQL database
    2. Initialize Gemini embedding provider
    3. Load data from restful-api.dev API
    4. Process and flatten nested JSON
    5. Generate embeddings for device names
    6. Store in PostgreSQL with vector columns
    """
    
    print("🚀 Simple API to Vector Example")
    print("=" * 50)
    
    # Step 1: Setup database connection (using environment variables)
    print("📊 Setting up database connection...")
    # DBConnection uses these environment variables:
    # LOCAL_POSTGRES_HOST, LOCAL_POSTGRES_PORT, LOCAL_POSTGRES_DB, 
    # LOCAL_POSTGRES_USER, LOCAL_POSTGRES_PASSWORD
    db_connection = DBConnection()
    
    try:
        await db_connection.initialize()
        print("✅ Database connected")
        
        # Step 2: Setup embedding provider
        print("🤖 Setting up embedding provider...")
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if gemini_api_key:
            # Use real Gemini embedding provider
            try:
                embedding_service = GeminiEmbeddingProvider()
                print("✅ Gemini embedding provider initialized")
            except Exception as e:
                print(f"⚠️  Gemini provider failed ({e}), using mock provider")
                embedding_service = MockEmbeddingProvider(embedding_dim=384)
        else:
            # Use mock embedding provider for demonstration
            embedding_service = MockEmbeddingProvider(embedding_dim=384)
            print("✅ Using mock embedding provider (set GEMINI_API_KEY for real embeddings)")
        
        # Step 3: Setup API loader and use case
        print("🔧 Setting up API loader...")
        api_loader = APIJSONStorageLoader(
            timeout=30,
            retry_attempts=3
        )
        
        repository = PostgresDataMoveRepository(db_connection)
        
        use_case = DataAPIJSONUseCase(
            repo=repository,
            embedding_service=embedding_service,
            storage_loader=api_loader
        )
        print("✅ Components initialized")
        
        # Step 4: Load data from API
        print("🌐 Loading data from API...")
        api_url = "https://api.restful-api.dev/objects"
        
        # Configuration for processing the API response
        config = {
            # Flatten nested JSON (the 'data' field contains nested info)
            'flatten_nested': True,
            'separator': '_',
            
            # Create a comprehensive description using original field names (before mapping)
            'update_request_body_mapping': {
                'description': "concat({name}, ' - Color: ', coalesce({data_color}, 'N/A'), ', Capacity: ', coalesce({data_capacity}, 'N/A'))"
            },
            
            # Map API fields to cleaner database column names (applied after transformations)
            'column_name_mapping': {
                'id': 'device_id',
                'name': 'device_name',
                'data_color': 'color',
                'data_capacity': 'capacity',
                'data_capacity_gb': 'capacity_gb',  # This field exists in the data
                'data_price': 'price',
                'data_generation': 'generation',
                'data_year': 'year'
            }
        }
        
        # Execute the API to vector workflow
        result = await use_case.execute(
            source=api_url,
            table_name="devices_simple",
            embed_columns_names=["device_name", "description"],  # Generate embeddings for these fields
            pk_columns=["device_id"],
            create_table_if_not_exists=True,
            embed_type="separated",  # Create separate embedding columns
            **config
        )
        
        # Step 5: Display results
        if result.success:
            print("🎉 SUCCESS!")
            print(f"   📊 Devices loaded: {result.rows_processed}")
            print(f"   ⏱️  Processing time: {result.execution_time:.2f} seconds")
            print(f"   🆕 Table created: {result.table_created}")
            
            if result.warnings:
                print(f"   ⚠️  Warnings: {len(result.warnings)}")
            
            print("\n📋 What was created:")
            print("   - Table: devices_simple")
            print("   - Columns: device_id, device_name, color, capacity, price, etc.")
            print("   - Embeddings: device_name_enc, description_enc")
            print("   - You can now perform vector similarity searches!")
            
            print("\n🔍 Example queries you can run:")
            print("   -- Find similar devices by name")
            print("   SELECT device_name, color, price")
            print("   FROM devices_simple")
            print("   ORDER BY device_name_enc <-> (SELECT device_name_enc FROM devices_simple WHERE device_name LIKE '%iPhone%' LIMIT 1)")
            print("   LIMIT 5;")
            
        else:
            print("❌ FAILED!")
            for error in result.errors:
                print(f"   Error: {error}")
    
    except Exception as e:
        print(f"❌ Exception: {e}")
        raise
    
    finally:
        # Cleanup
        await db_connection.close()
        print("🔌 Database connection closed")


async def test_api_loading_only():
    """
    Test just the API loading part without database operations.
    
    This is useful for testing the API connection and data processing
    without needing a full database setup.
    """
    
    print("\n🧪 Testing API Loading Only")
    print("=" * 30)
    
    try:
        # Just test the API loading
        api_loader = APIJSONStorageLoader()
        
        print("🌐 Loading data from API...")
        df = await api_loader.load_json("https://api.restful-api.dev/objects")
        
        print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show first few rows
        print("\n📊 Sample data:")
        print(df.head(3).to_string())
        
        # Test with configuration
        print("\n🔧 Testing with configuration...")
        config = {
            'flatten_nested': True,
            'column_name_mapping': {
                'id': 'device_id',
                'name': 'device_name'
            }
        }
        
        df_configured = await api_loader.load_json("https://api.restful-api.dev/objects", config)
        print(f"✅ Configured loading: {len(df_configured)} rows, {len(df_configured.columns)} columns")
        print(f"📋 New columns: {list(df_configured.columns)}")
        
    except Exception as e:
        print(f"❌ API loading failed: {e}")


async def main():
    """Main function to run the example."""
    
    # Check if we have database configuration
    db_host = os.getenv('DB_HOST', 'localhost')
    
    print("💡 This example can run in two modes:")
    print("   1. API-only mode: Just loads and processes API data")
    print("   2. Full mode: Loads data, generates embeddings, stores in PostgreSQL")
    print()
    
    # Always run API-only test first
    await test_api_loading_only()
    
    # Try to run full example if database seems configured
    try:
        print("\n🔄 Attempting full database example...")
        await simple_api_to_vector_example()
    except Exception as e:
        print(f"\n⚠️  Full example failed (this is normal if database not configured): {e}")
        print("\n💡 To run the full example with database:")
        print("   1. Install PostgreSQL with pgvector extension")
        print("   2. Configure database settings (DB_HOST, DB_PORT, etc.)")
        print("   3. Run: python examples/setup_environment.py")
        print("   4. Optionally set GEMINI_API_KEY for real embeddings")


if __name__ == "__main__":
    print("📱 Device Data API to Vector Store Example")
    print("Using API: https://api.restful-api.dev/objects")
    print("=" * 60)
    
    asyncio.run(main())