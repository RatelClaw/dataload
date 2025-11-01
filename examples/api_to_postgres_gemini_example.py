#!/usr/bin/env python3
"""
API to PostgreSQL with Gemini Embeddings Example

This example uses the existing codebase infrastructure exactly as shown in main_pg_gemni.py
but loads data from APIs instead of CSV files.

It demonstrates:
1. Loading data from https://api.restful-api.dev/objects
2. Using the existing GeminiEmbeddingProvider
3. Using the existing PostgresDataRepository
4. Using the existing dataloadUseCase with APIJSONStorageLoader
5. Performing vector similarity searches
"""

import asyncio
import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataload.infrastructure.db.db_connection import DBConnection
from dataload.infrastructure.db.data_repository import PostgresDataRepository
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase
from dataload.config import logger


async def load_api_data_to_postgres():
    """
    Load API data to PostgreSQL using existing infrastructure.
    
    This function replicates the pattern from main_pg_gemni.py but loads
    data from APIs instead of CSV files.
    """
    
    print("🚀 API to PostgreSQL with Gemini Embeddings")
    print("=" * 60)
    
    # Step 1: Initialize database connection (same as main_pg_gemni.py)
    print("📊 Initializing database connection...")
    db_conn = DBConnection()
    await db_conn.initialize()
    
    # Step 2: Initialize repository (same as main_pg_gemni.py)
    repo = PostgresDataRepository(db_conn)
    
    # Step 3: Initialize Gemini embedding provider (same as main_pg_gemni.py)
    print("🤖 Initializing Gemini embedding provider...")
    embedding = GeminiEmbeddingProvider()
    
    # Step 4: Initialize API loader (instead of LocalLoader)
    print("🌐 Initializing API JSON loader...")
    api_loader = APIJSONStorageLoader()
    
    # Step 5: Create use case (same pattern as main_pg_gemni.py)
    use_case = dataloadUseCase(repo, embedding, api_loader)
    
    try:
        # Step 6: Load data from API and convert to CSV format for existing use case
        print("📱 Loading device data from API...")
        
        # Load data from your specified API
        api_url = "https://api.restful-api.dev/objects"
        
        # Configure JSON processing
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
                'description': "concat({device_name}, ' - ', coalesce({color}, 'N/A'), ' ', coalesce({capacity}, 'N/A'))"
            }
        }
        
        # Load and process API data
        df = await api_loader.load_json(api_url, config)
        print(f"✅ Loaded {len(df)} devices from API")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Step 7: Save to temporary CSV file (to work with existing dataloadUseCase)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_csv_path = f.name
        
        print(f"💾 Saved API data to temporary CSV: {temp_csv_path}")
        
        # Step 8: Use existing dataloadUseCase to load CSV with embeddings
        print("🔄 Loading data with embeddings using existing use case...")
        
        await use_case.execute(
            temp_csv_path,                          # CSV file path
            'api_devices_gemini',                   # Table name
            ['device_name', 'description'],         # Columns to embed
            ['device_id'],                          # Primary key columns
            create_table_if_not_exists=True,
            embed_type='separated'                  # Create separate embedding columns
        )
        
        print("✅ Data loaded successfully with Gemini embeddings!")
        
        # Step 9: Demonstrate vector similarity search (same as main_pg_gemni.py)
        print("\n🔍 Performing similarity search...")
        
        # Search for devices similar to "iPhone"
        query_text = "Apple iPhone smartphone"
        query_embedding = embedding.get_embeddings([query_text])[0]
        
        # Search using device_name embeddings
        results = await repo.search(
            'api_devices_gemini', 
            query_embedding, 
            top_k=5, 
            embed_column='device_name_enc'
        )
        
        print("📱 Devices similar to 'Apple iPhone smartphone':")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['metadata'].get('device_name', 'N/A')}")
            print(f"     Color: {result['metadata'].get('color', 'N/A')}")
            print(f"     Capacity: {result['metadata'].get('capacity', 'N/A')}")
            print(f"     Similarity: {1 - result['distance']:.3f}")
            print()
        
        # Search using description embeddings
        query_text2 = "wireless headphones audio"
        query_embedding2 = embedding.get_embeddings([query_text2])[0]
        
        results2 = await repo.search(
            'api_devices_gemini',
            query_embedding2,
            top_k=3,
            embed_column='description_enc'
        )
        
        print("🎧 Devices similar to 'wireless headphones audio':")
        for i, result in enumerate(results2, 1):
            print(f"  {i}. {result['metadata'].get('device_name', 'N/A')}")
            print(f"     Description: {result['metadata'].get('description', 'N/A')}")
            print(f"     Similarity: {1 - result['distance']:.3f}")
            print()
        
        # Clean up temporary file
        os.unlink(temp_csv_path)
        print(f"🗑️  Cleaned up temporary file: {temp_csv_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    
    finally:
        # Close database connection
        await db_conn.close()
        print("🔌 Database connection closed")


async def load_blog_posts_example():
    """
    Load blog posts from JSONPlaceholder API.
    
    This demonstrates loading different types of API data.
    """
    
    print("\n" + "=" * 60)
    print("📝 Loading Blog Posts from JSONPlaceholder API")
    print("=" * 60)
    
    # Initialize components
    db_conn = DBConnection()
    await db_conn.initialize()
    repo = PostgresDataRepository(db_conn)
    embedding = GeminiEmbeddingProvider()
    api_loader = APIJSONStorageLoader()
    use_case = dataloadUseCase(repo, embedding, api_loader)
    
    try:
        # Load blog posts
        api_url = "https://jsonplaceholder.typicode.com/posts"
        
        config = {
            'column_name_mapping': {
                'id': 'post_id',
                'userId': 'user_id',
                'title': 'post_title',
                'body': 'post_content'
            },
            'update_request_body_mapping': {
                'full_text': "concat({post_title}, ' - ', {post_content})"
            }
        }
        
        # Load and process
        df = await api_loader.load_json(api_url, config)
        print(f"✅ Loaded {len(df)} blog posts")
        
        # Save to temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_csv_path = f.name
        
        # Load with embeddings
        await use_case.execute(
            temp_csv_path,
            'api_blog_posts_gemini',
            ['post_title', 'full_text'],
            ['post_id'],
            create_table_if_not_exists=True,
            embed_type='combined'  # Single embeddings column
        )
        
        print("✅ Blog posts loaded with embeddings!")
        
        # Search for posts about technology
        query_text = "technology programming software"
        query_embedding = embedding.get_embeddings([query_text])[0]
        
        results = await repo.search(
            'api_blog_posts_gemini',
            query_embedding,
            top_k=3
        )
        
        print("💻 Posts similar to 'technology programming software':")
        for i, result in enumerate(results, 1):
            title = result['metadata'].get('post_title', 'N/A')
            content = result['metadata'].get('post_content', 'N/A')[:100] + "..."
            print(f"  {i}. {title}")
            print(f"     {content}")
            print(f"     Similarity: {1 - result['distance']:.3f}")
            print()
        
        # Clean up
        os.unlink(temp_csv_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    
    finally:
        await db_conn.close()


async def main():
    """Main function to run all examples."""
    
    # Check for Gemini API key
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  GEMINI_API_KEY environment variable not set!")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        print("   Set it with: export GEMINI_API_KEY='your-api-key-here'")
        print("\n   This example requires real Gemini embeddings.")
        return
    
    try:
        # Run device data example
        await load_api_data_to_postgres()
        
        # Run blog posts example
        await load_blog_posts_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\n📊 Database tables created:")
        print("   - api_devices_gemini (with device_name_enc, description_enc)")
        print("   - api_blog_posts_gemini (with embeddings column)")
        print("\n🔍 You can now perform vector similarity searches on this data!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("\n💡 Make sure you have:")
        print("   1. PostgreSQL running with pgvector extension")
        print("   2. GEMINI_API_KEY environment variable set")
        print("   3. Database connection configured")


if __name__ == "__main__":
    print("🌐 API to PostgreSQL with Gemini Embeddings Example")
    print("Using existing codebase infrastructure")
    print("API: https://api.restful-api.dev/objects")
    print("=" * 60)
    
    asyncio.run(main())