#!/usr/bin/env python3
"""
Simple Working Example - No API Key Required

This example works immediately without requiring GEMINI_API_KEY.
It uses the existing infrastructure with a simple embedding provider.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase


class SimpleEmbeddingProvider(EmbeddingProviderInterface):
    """Simple embedding provider that works without API keys."""
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate simple embeddings based on text characteristics."""
        embeddings = []
        for text in texts:
            # Create a simple 384-dimensional embedding
            embedding = []
            text_lower = text.lower()
            
            for i in range(384):
                if i < len(text_lower):
                    # Use character values
                    value = (ord(text_lower[i]) - 97) / 25.0  # Normalize a-z to 0-1
                else:
                    # Use text hash for remaining dimensions
                    value = ((hash(text) + i) % 1000) / 1000.0
                
                # Convert to [-1, 1] range
                value = (value * 2) - 1
                embedding.append(value)
            
            embeddings.append(embedding)
        
        return embeddings


async def main():
    """
    Load API data using existing infrastructure with simple embeddings.
    """
    
    print("🚀 Simple Working API Example")
    print("=" * 40)
    print("✅ No API keys required!")
    print("✅ Uses existing infrastructure")
    print("✅ Works immediately")
    print()
    
    # Step 1: Initialize database connection
    print("📊 Connecting to database...")
    db_conn = DBConnection()
    await db_conn.initialize()
    
    # Step 2: Initialize repository
    repo = PostgresDataRepository(db_conn)
    
    # Step 3: Initialize simple embedding provider (no API key needed)
    print("🤖 Initializing simple embedding provider...")
    embedding = SimpleEmbeddingProvider()
    
    # Step 4: Initialize API loader
    print("🌐 Initializing API loader...")
    api_loader = APIJSONStorageLoader()
    
    # Step 5: Create use case
    use_case = dataloadUseCase(repo, embedding, api_loader)
    
    try:
        # Step 6: Load API data
        print("📱 Loading device data from API...")
        api_url = "https://api.restful-api.dev/objects"
        
        # Simple configuration
        config = {
            'flatten_nested': True,
            'separator': '_',
            'column_name_mapping': {
                'id': 'device_id',
                'name': 'device_name'
            }
        }
        
        df = await api_loader.load_json(api_url, config)
        print(f"✅ Loaded {len(df)} devices")
        print(f"📋 Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
        
        # Step 7: Save to CSV for existing use case
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_csv_path = f.name
        
        # Step 8: Load with embeddings using existing pattern
        print("🔄 Creating table with embeddings...")
        
        await use_case.execute(
            temp_csv_path,
            'simple_devices',
            ['device_name'],  # Just embed device names
            ['device_id'],
            create_table_if_not_exists=True,
            embed_type='combined'  # Single embeddings column
        )
        
        print("✅ SUCCESS! Data loaded with embeddings!")
        
        # Step 9: Test search
        print("\n🔍 Testing search...")
        
        query_text = "iPhone"
        query_embedding = embedding.get_embeddings([query_text])[0]
        
        results = await repo.search('simple_devices', query_embedding, top_k=3)
        
        print("📱 Top 3 results for 'iPhone':")
        for i, result in enumerate(results, 1):
            device_name = result['metadata'].get('device_name', 'N/A')
            similarity = 1 - result['distance']
            print(f"  {i}. {device_name} (similarity: {similarity:.3f})")
        
        # Clean up
        os.unlink(temp_csv_path)
        
        print("\n🎉 Complete! Your API data is now searchable!")
        print("📊 Table created: simple_devices")
        print("🔍 You can now perform vector similarity searches!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await db_conn.close()
        print("\n🔌 Database connection closed")


if __name__ == "__main__":
    print("📱 Simple API to Vector Store Example")
    print("API: https://api.restful-api.dev/objects")
    print("Database: PostgreSQL with pgvector")
    print("=" * 50)
    
    asyncio.run(main())