#!/usr/bin/env python3
"""
Example demonstrating pagination support in APIHandler.

This example shows how to use different pagination patterns:
1. Page/Size pagination
2. Offset/Limit pagination  
3. Cursor-based pagination
"""

import asyncio
import json
from dataload.domain.api_handler import APIHandler
from dataload.domain.api_entities import PaginationConfig, PaginationType


async def example_page_size_pagination():
    """Example of page/size pagination pattern."""
    print("=== Page/Size Pagination Example ===")
    
    handler = APIHandler(
        base_url="https://jsonplaceholder.typicode.com",
        timeout=10
    )
    
    pagination_config = PaginationConfig(
        enabled=True,
        pagination_type=PaginationType.PAGE_SIZE,
        page_param="page",
        size_param="limit",
        page_size=5,
        max_pages=2  # Limit to 2 pages for demo
    )
    
    try:
        async with handler:
            responses = await handler.fetch_data(
                '/posts',
                pagination_config=pagination_config
            )
            
            print(f"Fetched {len(responses)} pages")
            for i, response in enumerate(responses):
                print(f"Page {i+1}: {len(response.data)} items")
                if response.pagination_info:
                    print(f"  Pagination info: {response.pagination_info}")
                    
    except Exception as e:
        print(f"Error: {e}")


async def example_offset_limit_pagination():
    """Example of offset/limit pagination pattern."""
    print("\n=== Offset/Limit Pagination Example ===")
    
    handler = APIHandler(
        base_url="https://jsonplaceholder.typicode.com",
        timeout=10
    )
    
    pagination_config = PaginationConfig(
        enabled=True,
        pagination_type=PaginationType.OFFSET_LIMIT,
        offset_param="start",
        limit_param="limit", 
        page_size=10,
        max_pages=2
    )
    
    try:
        async with handler:
            responses = await handler.fetch_data(
                '/posts',
                pagination_config=pagination_config
            )
            
            print(f"Fetched {len(responses)} pages")
            for i, response in enumerate(responses):
                print(f"Page {i+1}: {len(response.data)} items")
                
    except Exception as e:
        print(f"Error: {e}")


async def example_cursor_pagination():
    """Example of cursor-based pagination (simulated)."""
    print("\n=== Cursor-Based Pagination Example ===")
    
    # Note: This is a simulated example since jsonplaceholder doesn't support cursor pagination
    # In a real scenario, you would use an API that supports cursor-based pagination
    
    handler = APIHandler(
        base_url="https://api.example.com",  # Placeholder URL
        api_token="your-api-key",
        timeout=10
    )
    
    pagination_config = PaginationConfig(
        enabled=True,
        pagination_type=PaginationType.CURSOR,
        cursor_param="cursor",
        size_param="limit",
        page_size=20
    )
    
    print("Cursor pagination configuration created:")
    print(f"  Pagination type: {pagination_config.pagination_type}")
    print(f"  Cursor parameter: {pagination_config.cursor_param}")
    print(f"  Page size: {pagination_config.page_size}")
    
    # In a real implementation, you would use:
    # try:
    #     async with handler:
    #         responses = await handler.fetch_data(
    #             '/your-endpoint',
    #             pagination_config=pagination_config
    #         )
    #         
    #         print(f"Fetched {len(responses)} pages")
    #         for i, response in enumerate(responses):
    #             print(f"Page {i+1}: {len(response.data)} items")
    #             if response.pagination_info and response.pagination_info.get('next_cursor'):
    #                 print(f"  Next cursor: {response.pagination_info['next_cursor']}")
    #                 
    # except Exception as e:
    #     print(f"Error: {e}")


async def example_memory_efficient_pagination():
    """Example of memory-efficient pagination for large datasets."""
    print("\n=== Memory-Efficient Pagination Example ===")
    
    handler = APIHandler(
        base_url="https://jsonplaceholder.typicode.com",
        timeout=10
    )
    
    pagination_config = PaginationConfig(
        enabled=True,
        pagination_type=PaginationType.PAGE_SIZE,
        page_param="page",
        size_param="limit",
        page_size=25,  # Larger page size for efficiency
        max_pages=4    # Process multiple pages
    )
    
    try:
        async with handler:
            responses = await handler.fetch_data(
                '/posts',
                pagination_config=pagination_config
            )
            
            print(f"Processed {len(responses)} pages efficiently")
            
            # Process each page individually to manage memory
            total_items = 0
            for i, response in enumerate(responses):
                page_items = len(response.data)
                total_items += page_items
                print(f"Page {i+1}: {page_items} items (Total so far: {total_items})")
                
                # In a real scenario, you might process and discard each page
                # to keep memory usage low:
                # process_page_data(response.data)
                # del response  # Free memory
                
            print(f"Total items processed: {total_items}")
            
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all pagination examples."""
    print("APIHandler Pagination Examples")
    print("=" * 50)
    
    await example_page_size_pagination()
    await example_offset_limit_pagination()
    await example_cursor_pagination()
    await example_memory_efficient_pagination()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())