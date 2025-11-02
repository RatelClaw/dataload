"""
Example demonstrating APIHandler usage for external API communication.

This example shows how to use the APIHandler class to make authenticated
requests to external APIs with retry logic and pagination support.
"""

import asyncio
import json
from dataload.domain import APIHandler, PaginationConfig, PaginationType, AuthType


async def basic_api_example():
    """Basic API request example."""
    print("=== Basic API Request Example ===")
    
    # Create APIHandler with API key authentication
    handler = APIHandler(
        base_url="https://jsonplaceholder.typicode.com",
        timeout=30,
        retry_attempts=3
    )
    
    try:
        async with handler:
            # Make a simple GET request
            response = await handler.get("/posts/1")
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {response.response_time:.2f}s")
            print(f"Data: {json.dumps(response.data, indent=2)}")
            
    except Exception as e:
        print(f"Error: {e}")


async def authenticated_api_example():
    """Example with API key authentication."""
    print("\n=== Authenticated API Request Example ===")
    
    # Create APIHandler with API key authentication
    handler = APIHandler(
        base_url="https://api.example.com",
        api_token="your-api-key-here",
        timeout=30,
        retry_attempts=3
    )
    
    # Demonstrate header preparation
    headers = handler._prepare_headers()
    print(f"Authentication Headers: {headers}")
    
    # Note: This would make an actual request if the API existed
    print("Would make authenticated request to /users endpoint")


async def jwt_authentication_example():
    """Example with JWT token authentication."""
    print("\n=== JWT Authentication Example ===")
    
    handler = APIHandler(
        base_url="https://api.example.com",
        jwt_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        timeout=30
    )
    
    headers = handler._prepare_headers()
    print(f"JWT Headers: {headers}")
    print(f"Auth Type: {handler.auth_config.auth_type}")


async def basic_auth_example():
    """Example with basic authentication."""
    print("\n=== Basic Authentication Example ===")
    
    handler = APIHandler(
        base_url="https://api.example.com",
        username="testuser",
        password="testpass",
        timeout=30
    )
    
    headers = handler._prepare_headers()
    print(f"Basic Auth Headers: {headers}")
    print(f"Auth Type: {handler.auth_config.auth_type}")


async def pagination_example():
    """Example with pagination configuration."""
    print("\n=== Pagination Configuration Example ===")
    
    handler = APIHandler(
        base_url="https://jsonplaceholder.typicode.com",
        timeout=30
    )
    
    # Configure pagination
    pagination_config = PaginationConfig(
        enabled=True,
        pagination_type=PaginationType.PAGE_SIZE,
        page_param="page",
        size_param="limit",
        page_size=10,
        max_pages=3
    )
    
    print(f"Pagination Config: {pagination_config}")
    
    try:
        async with handler:
            # This would fetch multiple pages if the API supported pagination
            response = await handler.get(
                "/posts",
                params={"_limit": 5},  # JSONPlaceholder uses _limit
                pagination_config=pagination_config
            )
            
            if isinstance(response, list):
                print(f"Fetched {len(response)} pages")
                for i, page_response in enumerate(response):
                    print(f"Page {i+1}: {len(page_response.data)} items")
            else:
                print(f"Single response: {len(response.data)} items")
                
    except Exception as e:
        print(f"Error: {e}")


async def error_handling_example():
    """Example demonstrating error handling."""
    print("\n=== Error Handling Example ===")
    
    handler = APIHandler(
        base_url="https://httpstat.us",  # Service for testing HTTP status codes
        timeout=10,
        retry_attempts=2
    )
    
    try:
        async with handler:
            # Test different error scenarios
            test_cases = [
                ("/200", "Success case"),
                ("/404", "Not found error"),
                ("/500", "Server error"),
                ("/timeout", "Timeout error (if supported)")
            ]
            
            for endpoint, description in test_cases:
                try:
                    print(f"\nTesting {description}...")
                    response = await handler.get(endpoint)
                    print(f"✓ Success: {response.status_code}")
                except Exception as e:
                    print(f"✗ Error: {type(e).__name__}: {e}")
                    
    except Exception as e:
        print(f"Handler error: {e}")


async def custom_headers_example():
    """Example with custom headers."""
    print("\n=== Custom Headers Example ===")
    
    handler = APIHandler(
        base_url="https://jsonplaceholder.typicode.com",
        default_headers={
            "User-Agent": "DataLoader-APIHandler/1.0",
            "Accept": "application/json"
        }
    )
    
    try:
        async with handler:
            # Add additional headers for this request
            response = await handler.get(
                "/posts/1",
                headers={
                    "X-Custom-Header": "custom-value",
                    "X-Request-ID": "12345"
                }
            )
            
            print(f"Response: {response.status_code}")
            print(f"URL: {response.url}")
            
    except Exception as e:
        print(f"Error: {e}")


async def post_request_example():
    """Example of POST request with JSON data."""
    print("\n=== POST Request Example ===")
    
    handler = APIHandler(
        base_url="https://jsonplaceholder.typicode.com",
        timeout=30
    )
    
    try:
        async with handler:
            # Create a new post
            new_post = {
                "title": "Test Post",
                "body": "This is a test post created via APIHandler",
                "userId": 1
            }
            
            response = await handler.post("/posts", json_data=new_post)
            
            print(f"Created post: {response.status_code}")
            print(f"Response: {json.dumps(response.data, indent=2)}")
            
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all examples."""
    print("APIHandler Examples")
    print("=" * 50)
    
    await basic_api_example()
    await authenticated_api_example()
    await jwt_authentication_example()
    await basic_auth_example()
    await pagination_example()
    await error_handling_example()
    await custom_headers_example()
    await post_request_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())