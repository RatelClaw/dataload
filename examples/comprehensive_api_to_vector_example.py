#!/usr/bin/env python3
"""
Comprehensive API to Vector Store Example

This example demonstrates how to:
1. Load data from multiple free APIs
2. Process and transform the data
3. Generate embeddings using Gemini
4. Store data in PostgreSQL with vector embeddings

APIs used:
- https://api.restful-api.dev/objects (Device data)
- https://jsonplaceholder.typicode.com/posts (Sample posts)
- https://jsonplaceholder.typicode.com/users (User data)
- https://httpbin.org/json (Test JSON data)

Requirements:
- PostgreSQL with pgvector extension
- Google Gemini API key
- Environment variables configured
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

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


class ComprehensiveAPIVectorExample:
    """
    Comprehensive example class for loading API data into vector store.
    
    This class demonstrates:
    - Multiple API data sources
    - Data transformation and mapping
    - Embedding generation with Gemini
    - PostgreSQL vector storage
    - Error handling and logging
    """
    
    def __init__(self):
        """Initialize the example with database and embedding service."""
        self.db_connection = None
        self.repository = None
        self.embedding_service = None
        self.api_loader = None
        self.use_case = None
    
    async def setup_services(self):
        """
        Set up all required services: database, embeddings, and API loader.
        
        This method initializes:
        - PostgreSQL database connection with pgvector
        - Gemini embedding provider
        - APIJSONStorageLoader for API data loading
        - DataAPIJSONUseCase for orchestrating the workflow
        """
        logger.info("Setting up services...")
        
        # 1. Initialize database connection
        # Make sure you have these environment variables set:
        # LOCAL_POSTGRES_HOST, LOCAL_POSTGRES_PORT, LOCAL_POSTGRES_DB,
        # LOCAL_POSTGRES_USER, LOCAL_POSTGRES_PASSWORD
        self.db_connection = DBConnection()
        
        # Initialize the connection pool
        await self.db_connection.initialize()
        logger.info("✅ Database connection initialized")
        
        # 2. Create PostgreSQL repository for vector operations
        self.repository = PostgresDataMoveRepository(self.db_connection)
        logger.info("✅ PostgreSQL repository created")
        
        # 3. Initialize Gemini embedding provider
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if gemini_api_key:
            try:
                # Use real Gemini embedding provider
                self.embedding_service = GeminiEmbeddingProvider()
                logger.info("✅ Gemini embedding provider initialized")
            except Exception as e:
                logger.warning(f"Gemini provider failed ({e}), using mock provider")
                self.embedding_service = MockEmbeddingProvider(embedding_dim=768)
                logger.info("✅ Mock embedding provider initialized (fallback)")
        else:
            # Use mock embedding provider for demonstration
            self.embedding_service = MockEmbeddingProvider(embedding_dim=768)
            logger.info("✅ Mock embedding provider initialized (set GEMINI_API_KEY for real embeddings)")
        
        # 4. Initialize API JSON loader with configuration
        self.api_loader = APIJSONStorageLoader(
            timeout=30,           # 30 second timeout for API calls
            retry_attempts=3,     # Retry failed requests 3 times
            verify_ssl=True       # Verify SSL certificates
        )
        logger.info("✅ API JSON loader initialized")
        
        # 5. Create the main use case for API to vector workflow
        self.use_case = DataAPIJSONUseCase(
            repo=self.repository,
            embedding_service=self.embedding_service,
            storage_loader=self.api_loader
        )
        logger.info("✅ DataAPIJSONUseCase initialized")
        
        logger.info("🎉 All services set up successfully!")
    
    async def load_device_data(self):
        """
        Load device data from restful-api.dev and store with embeddings.
        
        This example demonstrates:
        - Loading from the API endpoint you provided
        - Handling nested JSON structures (data field)
        - Column mapping for cleaner database schema
        - Generating embeddings from device names and descriptions
        """
        logger.info("=" * 60)
        logger.info("EXAMPLE 1: Loading Device Data from restful-api.dev")
        logger.info("=" * 60)
        
        # API endpoint with device data
        api_url = "https://api.restful-api.dev/objects"
        
        # Configuration for processing the API data
        config = {
            # Flatten nested JSON structures (like the 'data' field)
            'flatten_nested': True,
            'separator': '_',  # Use underscore to separate nested field names
            
            # Column mapping to create cleaner database column names
            'column_name_mapping': {
                'id': 'device_id',
                'name': 'device_name',
                'data_color': 'color',
                'data_capacity': 'capacity',
                'data_capacity GB': 'capacity_gb',  # Handle space in field name
                'data_price': 'price',
                'data_generation': 'generation',
                'data_year': 'year',
                'data_CPU model': 'cpu_model',
                'data_Hard disk size': 'hard_disk_size',
                'data_Strap Colour': 'strap_color',
                'data_Case Size': 'case_size',
                'data_Color': 'color_alt',  # Alternative color field
                'data_Description': 'description',
                'data_Capacity': 'capacity_alt',  # Alternative capacity field
                'data_Screen size': 'screen_size',
                'data_Generation': 'generation_alt',
                'data_Price': 'price_alt'
            },
            
            # Data transformations to create computed fields
            'update_request_body_mapping': {
                # Create a comprehensive description for embedding
                'full_description': "concat({device_name}, ' - ', coalesce({description}, {color}, {capacity}, 'Device'))",
                # Create a price category
                'price_category': "case when {price} > 1000 then 'Premium' when {price} > 500 then 'Mid-range' else 'Budget' end"
            }
        }
        
        try:
            # Execute the API to vector workflow
            result = await self.use_case.execute(
                source=api_url,
                table_name="devices",
                embed_columns_names=["device_name", "full_description"],  # Generate embeddings for these fields
                pk_columns=["device_id"],
                create_table_if_not_exists=True,
                embed_type="separated",  # Create separate embedding columns for each field
                **config
            )
            
            # Display results
            if result.success:
                logger.info(f"✅ Device data loaded successfully!")
                logger.info(f"   📊 Rows processed: {result.rows_processed}")
                logger.info(f"   ⏱️  Execution time: {result.execution_time:.2f}s")
                logger.info(f"   🆕 Table created: {result.table_created}")
                logger.info(f"   🔄 Schema updated: {result.schema_updated}")
                
                if result.warnings:
                    logger.info(f"   ⚠️  Warnings: {len(result.warnings)}")
                    for warning in result.warnings[:3]:  # Show first 3 warnings
                        logger.info(f"      - {warning}")
            else:
                logger.error(f"❌ Device data loading failed!")
                for error in result.errors:
                    logger.error(f"   Error: {error}")
        
        except Exception as e:
            logger.error(f"❌ Exception during device data loading: {e}")
            raise
    
    async def load_blog_posts_data(self):
        """
        Load blog posts from JSONPlaceholder and store with embeddings.
        
        This example demonstrates:
        - Loading from a different API structure
        - Simple flat JSON data
        - Text embedding for content search
        - Combined embedding type
        """
        logger.info("=" * 60)
        logger.info("EXAMPLE 2: Loading Blog Posts from JSONPlaceholder")
        logger.info("=" * 60)
        
        # JSONPlaceholder posts API
        api_url = "https://jsonplaceholder.typicode.com/posts"
        
        # Configuration for blog posts
        config = {
            # Column mapping for cleaner names
            'column_name_mapping': {
                'id': 'post_id',
                'userId': 'user_id',
                'title': 'post_title',
                'body': 'post_content'
            },
            
            # Create computed fields
            'update_request_body_mapping': {
                # Combine title and content for comprehensive embedding
                'full_text': "concat({post_title}, ' - ', {post_content})",
                # Create a content length category
                'content_length': "case when length({post_content}) > 200 then 'Long' when length({post_content}) > 100 then 'Medium' else 'Short' end"
            }
        }
        
        try:
            result = await self.use_case.execute(
                source=api_url,
                table_name="blog_posts",
                embed_columns_names=["full_text"],  # Single combined embedding
                pk_columns=["post_id"],
                create_table_if_not_exists=True,
                embed_type="combined",  # Single embeddings column
                **config
            )
            
            if result.success:
                logger.info(f"✅ Blog posts loaded successfully!")
                logger.info(f"   📊 Rows processed: {result.rows_processed}")
                logger.info(f"   ⏱️  Execution time: {result.execution_time:.2f}s")
            else:
                logger.error(f"❌ Blog posts loading failed!")
                for error in result.errors:
                    logger.error(f"   Error: {error}")
        
        except Exception as e:
            logger.error(f"❌ Exception during blog posts loading: {e}")
            raise
    
    async def load_user_data(self):
        """
        Load user data from JSONPlaceholder with complex nested structures.
        
        This example demonstrates:
        - Complex nested JSON with address and company info
        - Multiple levels of nesting
        - Geographic data handling
        - Professional information embedding
        """
        logger.info("=" * 60)
        logger.info("EXAMPLE 3: Loading User Data with Complex Nesting")
        logger.info("=" * 60)
        
        api_url = "https://jsonplaceholder.typicode.com/users"
        
        config = {
            # Handle complex nesting
            'flatten_nested': True,
            'separator': '_',
            'max_depth': 3,  # Limit nesting depth to avoid too many columns
            
            # Comprehensive column mapping
            'column_name_mapping': {
                'id': 'user_id',
                'name': 'full_name',
                'username': 'username',
                'email': 'email',
                'phone': 'phone',
                'website': 'website',
                'address_street': 'street',
                'address_suite': 'suite',
                'address_city': 'city',
                'address_zipcode': 'zipcode',
                'address_geo_lat': 'latitude',
                'address_geo_lng': 'longitude',
                'company_name': 'company_name',
                'company_catchPhrase': 'company_slogan',
                'company_bs': 'company_business'
            },
            
            # Create meaningful computed fields
            'update_request_body_mapping': {
                # Professional profile for embedding
                'professional_profile': "concat({full_name}, ' works at ', {company_name}, ' - ', {company_slogan})",
                # Location description
                'location_info': "concat({city}, ', ', {street}, ' ', {suite})",
                # Contact summary
                'contact_summary': "concat('Email: ', {email}, ', Phone: ', {phone}, ', Website: ', coalesce({website}, 'N/A'))"
            }
        }
        
        try:
            result = await self.use_case.execute(
                source=api_url,
                table_name="users",
                embed_columns_names=["professional_profile", "contact_summary"],
                pk_columns=["user_id"],
                create_table_if_not_exists=True,
                embed_type="separated",
                **config
            )
            
            if result.success:
                logger.info(f"✅ User data loaded successfully!")
                logger.info(f"   📊 Rows processed: {result.rows_processed}")
                logger.info(f"   ⏱️  Execution time: {result.execution_time:.2f}s")
            else:
                logger.error(f"❌ User data loading failed!")
        
        except Exception as e:
            logger.error(f"❌ Exception during user data loading: {e}")
            raise
    
    async def load_test_json_data(self):
        """
        Load test JSON data from httpbin.org.
        
        This example demonstrates:
        - Simple API with test data
        - Handling different JSON structures
        - Error handling for API failures
        """
        logger.info("=" * 60)
        logger.info("EXAMPLE 4: Loading Test JSON Data")
        logger.info("=" * 60)
        
        api_url = "https://httpbin.org/json"
        
        try:
            result = await self.use_case.execute(
                source=api_url,
                table_name="test_data",
                embed_columns_names=[],  # No embeddings for test data
                pk_columns=[],  # No primary key for test data
                create_table_if_not_exists=True
            )
            
            if result.success:
                logger.info(f"✅ Test data loaded successfully!")
                logger.info(f"   📊 Rows processed: {result.rows_processed}")
            else:
                logger.error(f"❌ Test data loading failed!")
        
        except Exception as e:
            logger.info(f"ℹ️  Test data loading failed (expected): {e}")
            # This is expected as httpbin.org/json returns a single object, not an array
    
    async def demonstrate_local_json_processing(self):
        """
        Demonstrate processing local JSON data with the same workflow.
        
        This example shows:
        - Processing in-memory JSON data
        - Custom data structures
        - Advanced transformations
        """
        logger.info("=" * 60)
        logger.info("EXAMPLE 5: Processing Local JSON Data")
        logger.info("=" * 60)
        
        # Sample local JSON data (could be from file or generated)
        local_data = [
            {
                "product_id": "P001",
                "product_info": {
                    "name": "Wireless Headphones",
                    "category": "Electronics",
                    "specifications": {
                        "battery_life": "30 hours",
                        "connectivity": "Bluetooth 5.0",
                        "weight": "250g"
                    }
                },
                "pricing": {
                    "retail_price": 199.99,
                    "discount": 15,
                    "currency": "USD"
                },
                "reviews": [
                    {"rating": 5, "comment": "Excellent sound quality"},
                    {"rating": 4, "comment": "Good battery life"}
                ]
            },
            {
                "product_id": "P002",
                "product_info": {
                    "name": "Smart Watch",
                    "category": "Wearables",
                    "specifications": {
                        "battery_life": "7 days",
                        "connectivity": "WiFi, Bluetooth",
                        "weight": "45g"
                    }
                },
                "pricing": {
                    "retail_price": 299.99,
                    "discount": 20,
                    "currency": "USD"
                },
                "reviews": [
                    {"rating": 5, "comment": "Great fitness tracking"},
                    {"rating": 5, "comment": "Stylish design"}
                ]
            }
        ]
        
        config = {
            'flatten_nested': True,
            'separator': '_',
            'handle_arrays': 'join',  # Join array elements instead of expanding
            
            'column_name_mapping': {
                'product_id': 'id',
                'product_info_name': 'name',
                'product_info_category': 'category',
                'product_info_specifications_battery_life': 'battery_life',
                'product_info_specifications_connectivity': 'connectivity',
                'product_info_specifications_weight': 'weight',
                'pricing_retail_price': 'price',
                'pricing_discount': 'discount_percent',
                'pricing_currency': 'currency'
            },
            
            'update_request_body_mapping': {
                'product_description': "concat({name}, ' - ', {category}, ' with ', {battery_life}, ' battery')",
                'discounted_price': "round({price} * (100 - {discount_percent}) / 100, 2)",
                'connectivity_summary': "concat('Connectivity: ', {connectivity}, ', Weight: ', {weight})"
            }
        }
        
        try:
            result = await self.use_case.execute(
                source=local_data,
                table_name="products",
                embed_columns_names=["product_description", "connectivity_summary"],
                pk_columns=["id"],
                create_table_if_not_exists=True,
                embed_type="combined",
                **config
            )
            
            if result.success:
                logger.info(f"✅ Local JSON data processed successfully!")
                logger.info(f"   📊 Rows processed: {result.rows_processed}")
            else:
                logger.error(f"❌ Local JSON processing failed!")
        
        except Exception as e:
            logger.error(f"❌ Exception during local JSON processing: {e}")
            raise
    
    async def demonstrate_concurrent_loading(self):
        """
        Demonstrate loading from multiple APIs concurrently.
        
        This example shows:
        - Concurrent API calls for better performance
        - Handling multiple data sources simultaneously
        - Error handling in concurrent scenarios
        """
        logger.info("=" * 60)
        logger.info("EXAMPLE 6: Concurrent API Loading")
        logger.info("=" * 60)
        
        # Multiple API endpoints to load concurrently
        api_sources = [
            "https://jsonplaceholder.typicode.com/albums",
            "https://jsonplaceholder.typicode.com/photos?_limit=10",  # Limit photos for demo
            "https://jsonplaceholder.typicode.com/todos?_limit=20"    # Limit todos for demo
        ]
        
        try:
            # Use the concurrent loading feature of APIJSONStorageLoader
            df = await self.api_loader.load_json_concurrent(
                sources=api_sources,
                config={'flatten_nested': True},
                max_concurrent=3
            )
            
            logger.info(f"✅ Concurrent loading completed!")
            logger.info(f"   📊 Total rows loaded: {len(df)}")
            logger.info(f"   📋 Columns: {list(df.columns)}")
            
            # Store the combined data
            result = await self.use_case.execute(
                source=df.to_dict('records'),  # Convert DataFrame back to records
                table_name="combined_data",
                embed_columns_names=[],  # No embeddings for this demo
                pk_columns=[],
                create_table_if_not_exists=True
            )
            
            if result.success:
                logger.info(f"   💾 Combined data stored successfully!")
        
        except Exception as e:
            logger.error(f"❌ Concurrent loading failed: {e}")
    
    async def demonstrate_error_scenarios(self):
        """
        Demonstrate various error handling scenarios.
        
        This example shows:
        - Invalid API endpoints
        - Network timeouts
        - Invalid JSON responses
        - Graceful error handling
        """
        logger.info("=" * 60)
        logger.info("EXAMPLE 7: Error Handling Scenarios")
        logger.info("=" * 60)
        
        error_scenarios = [
            {
                "name": "Invalid URL",
                "url": "https://invalid-api-endpoint-that-does-not-exist.com/data",
                "expected": "Connection error"
            },
            {
                "name": "Valid domain, invalid endpoint",
                "url": "https://jsonplaceholder.typicode.com/nonexistent",
                "expected": "404 Not Found"
            },
            {
                "name": "Timeout scenario",
                "url": "https://httpbin.org/delay/10",  # 10 second delay
                "expected": "Timeout error"
            }
        ]
        
        for scenario in error_scenarios:
            logger.info(f"Testing: {scenario['name']}")
            try:
                # Set a short timeout for demonstration
                short_timeout_loader = APIJSONStorageLoader(timeout=5)
                df = await short_timeout_loader.load_json(scenario['url'])
                logger.info(f"   ⚠️  Unexpected success for {scenario['name']}")
            except Exception as e:
                logger.info(f"   ✅ Expected error caught: {type(e).__name__}: {str(e)[:100]}...")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.db_connection:
            await self.db_connection.close()
            logger.info("🔌 Database connection closed")
    
    async def run_all_examples(self):
        """
        Run all examples in sequence.
        
        This method demonstrates the complete workflow from API to vector store.
        """
        try:
            # Setup all services
            await self.setup_services()
            
            # Run all examples
            await self.load_device_data()
            await self.load_blog_posts_data()
            await self.load_user_data()
            await self.load_test_json_data()
            await self.demonstrate_local_json_processing()
            await self.demonstrate_concurrent_loading()
            await self.demonstrate_error_scenarios()
            
            logger.info("=" * 60)
            logger.info("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("Check your PostgreSQL database for the following tables:")
            logger.info("  - devices (with device_name_enc and full_description_enc embeddings)")
            logger.info("  - blog_posts (with embeddings column)")
            logger.info("  - users (with professional_profile_enc and contact_summary_enc embeddings)")
            logger.info("  - products (with embeddings column)")
            logger.info("  - combined_data (no embeddings)")
            logger.info("=" * 60)
        
        except Exception as e:
            logger.error(f"❌ Example execution failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """
    Main function to run the comprehensive API to vector example.
    
    Before running this example, make sure you have:
    1. PostgreSQL with pgvector extension installed
    2. Environment variables set:
       - GEMINI_API_KEY: Your Google Gemini API key
       - DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD: Database connection details
    3. Required Python packages installed
    """
    
    # Check if database is configured (Gemini API key is optional - will use mock if not available)
    logger.info("🔧 Checking configuration...")
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        logger.info("✅ GEMINI_API_KEY found - will use enhanced embeddings")
    else:
        logger.info("ℹ️  GEMINI_API_KEY not found - will use mock embeddings for demonstration")
    
    # Create and run the example
    example = ComprehensiveAPIVectorExample()
    await example.run_all_examples()


if __name__ == "__main__":
    """
    Entry point for the comprehensive API to vector example.
    
    Usage:
        python examples/comprehensive_api_to_vector_example.py
    
    Environment Setup:
        export GEMINI_API_KEY="your-gemini-api-key"
        export DB_HOST="localhost"
        export DB_PORT="5432"
        export DB_NAME="vector_db"
        export DB_USER="postgres"
        export DB_PASSWORD="your-password"
    """
    
    print("🚀 Starting Comprehensive API to Vector Store Example")
    print("=" * 60)
    print("This example demonstrates:")
    print("  ✅ Loading data from multiple free APIs")
    print("  ✅ Processing complex nested JSON structures")
    print("  ✅ Generating embeddings with Google Gemini")
    print("  ✅ Storing data in PostgreSQL with vector search")
    print("  ✅ Error handling and concurrent operations")
    print("=" * 60)
    
    # Run the async main function
    asyncio.run(main())