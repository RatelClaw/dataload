#!/usr/bin/env python3
"""
Working API Example - Uses Existing Infrastructure

This example uses the exact same pattern as main_pg_gemni.py but loads API data.
It converts API data to CSV format to work with the existing dataloadUseCase.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase


async def main():
    """
    Load API data using the exact same pattern as main_pg_gemni.py
    """
    
    print("🚀 Working API Example - Using Existing Infrastructure")
    print("=" * 60)
    
    # Step 1: Initialize database connection (same as main_pg_gemni.py)
    print("📊 Initializing database connection...")
    db_conn = DBConnection()
    await db_conn.initialize()
    
    # Step 2: Initialize repository (same as main_pg_gemni.py)
    repo = PostgresDataRepository(db_conn)
    
    # Step 3: Initialize embedding provider (same as main_pg_gemni.py)
    print("🤖 Initializing embedding provider...")
    try:
        embedding = GeminiEmbeddingProvider()
        print("✅ Gemini embedding provider initialized")
    except Exception as e:
        print(f"⚠️  Gemini provider failed ({e}), this example needs real embeddings")
        print("   Please set GEMINI_API_KEY environment variable")
        await db_conn.close()
        return
    
    # Step 4: Initialize API loader
    print("🌐 Initializing API loader...")
    api_loader = APIJSONStorageLoader()
    
    # Step 5: Create use case (same as main_pg_gemni.py)
    use_case = dataloadUseCase(repo, embedding, api_loader)
    
    try:
        # Step 6: Load API data and convert to CSV format
        print("📱 Loading device data from API...")
        api_url = "https://api.restful-api.dev/objects"
        
        # Load and process API data
        config = {
            'flatten_nested': True,
            'separator': '_',
            'column_name_mapping': {
                'id': 'device_id',
                'name': 'device_name',
                'data_color': 'color',
                'data_capacity': 'capacity',
                'data_price': 'price'
            },
            'update_request_body_mapping': {
                'description': "concat({device_name}, ' - ', coalesce({color}, 'N/A'))"
            }
        }
        
        df = await api_loader.load_json(api_url, config)
        print(f"✅ Loaded {len(df)} devices from API")
        
        # Step 7: Save to temporary CSV (to work with existing dataloadUseCase)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_csv_path = f.name
        
        print(f"💾 Saved to temporary CSV: {temp_csv_path}")
        
        # Step 8: Use existing dataloadUseCase (same pattern as main_pg_gemni.py)
        print("🔄 Loading data with embeddings...")
        
        await use_case.execute(
            temp_csv_path,                    # CSV file path
            'api_devices_working',            # Table name
            ['device_name', 'description'],   # Columns to embed
            ['device_id'],                    # Primary key columns
            create_table_if_not_exists=True,
            embed_type='separated'            # Create separate embedding columns
        )
        
        print("✅ Data loaded successfully!")
        
        # Step 9: Test similarity search (same as main_pg_gemni.py)
        print("\n🔍 Testing similarity search...")
        
        query_text = "Apple iPhone smartphone"
        query_embedding = embedding.get_embeddings([query_text])[0]
        
        # Search using device_name embeddings
        results = await repo.search(
            'api_devices_working', 
            query_embedding, 
            top_k=5, 
            embed_column='device_name_enc'
        )
        
        print("📱 Devices similar to 'Apple iPhone smartphone':")
        for i, result in enumerate(results, 1):
            device_name = result['metadata'].get('device_name', 'N/A')
            color = result['metadata'].get('color', 'N/A')
            capacity = result['metadata'].get('capacity', 'N/A')
            similarity = 1 - result['distance']
            
            print(f"  {i}. {device_name}")
            print(f"     Color: {color}, Capacity: {capacity}")
            print(f"     Similarity: {similarity:.3f}")
            print()
        
        # Clean up temporary file
        os.unlink(temp_csv_path)
        print(f"🗑️  Cleaned up temporary file")
        
        print("🎉 SUCCESS! API data loaded and searchable!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Close database connection
        await db_conn.close()
        print("🔌 Database connection closed")


if __name__ == "__main__":
    print("📱 Working API to PostgreSQL Example")
    print("Uses the exact same pattern as main_pg_gemni.py")
    print("API: https://api.restful-api.dev/objects")
    print()
    
    # Check for Gemini API key
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  GEMINI_API_KEY not set!")
        print("   This example requires real Gemini embeddings")
        print("   Set it with: export GEMINI_API_KEY='your-api-key-here'")
        print()
    
    asyncio.run(main())