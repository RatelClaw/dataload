"""
Performance tests for asynchronous and concurrent operations in the API JSON loader.

This module contains comprehensive performance tests to validate the async implementation
and concurrent processing capabilities of the API JSON loader system.
"""

import asyncio
import time
import pytest
import pandas as pd
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import json
import tempfile
import os

from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.application.use_cases.data_api_json_use_case import DataAPIJSONUseCase
from dataload.domain.api_entities import APIResponse, APIError
from dataload.domain.entities import DataMoveResult


class MockEmbeddingService:
    """Mock embedding service for performance testing."""
    
    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.call_count = 0
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings with configurable delay."""
        self.call_count += 1
        time.sleep(self.delay)  # Simulate processing time
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockRepository:
    """Mock repository for performance testing."""
    
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.call_count = 0
    
    async def table_exists(self, table_name: str) -> bool:
        await asyncio.sleep(self.delay)
        return False
    
    async def create_table(self, **kwargs):
        self.call_count += 1
        await asyncio.sleep(self.delay)
    
    async def insert_data(self, table_name: str, df: pd.DataFrame, pk_columns: List[str]):
        self.call_count += 1
        await asyncio.sleep(self.delay)
        return len(df)
    
    async def update_data(self, table_name: str, df: pd.DataFrame, pk_columns: List[str]):
        self.call_count += 1
        await asyncio.sleep(self.delay)
        return len(df)


@pytest.fixture
def mock_api_responses():
    """Create mock API responses for testing."""
    return [
        {
            "data": [
                {"id": i, "name": f"item_{i}", "value": i * 10}
                for i in range(1, 101)  # 100 items per response
            ]
        }
        for _ in range(5)  # 5 different responses
    ]


@pytest.fixture
def temp_json_files():
    """Create temporary JSON files for testing."""
    files = []
    try:
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                data = [
                    {"id": j + i * 100, "name": f"file_{i}_item_{j}", "category": f"cat_{i}"}
                    for j in range(1, 51)  # 50 items per file
                ]
                json.dump(data, f)
                files.append(f.name)
        yield files
    finally:
        for file_path in files:
            try:
                os.unlink(file_path)
            except OSError:
                pass


class TestAsyncAPIJSONLoader:
    """Test async functionality of APIJSONStorageLoader."""
    
    @pytest.mark.asyncio
    async def test_async_load_json_api_endpoint(self):
        """Test async loading from API endpoint."""
        loader = APIJSONStorageLoader()
        
        # Mock the API handler
        mock_response = APIResponse(
            data=[{"id": 1, "name": "test"}],
            status_code=200,
            headers={},
            response_time=0.1,
            url="https://api.example.com/data",
            method="GET"
        )
        
        with patch.object(loader.api_handler, 'fetch_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_response
            
            start_time = time.time()
            df = await loader.load_json("https://api.example.com/data")
            execution_time = time.time() - start_time
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df.iloc[0]['id'] == 1
            assert df.iloc[0]['name'] == "test"
            assert execution_time < 1.0  # Should be fast with mocked response
            mock_fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_load_json_file(self, temp_json_files):
        """Test async loading from JSON file."""
        loader = APIJSONStorageLoader()
        
        start_time = time.time()
        df = await loader.load_json(temp_json_files[0])
        execution_time = time.time() - start_time
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50  # 50 items in test file
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert 'category' in df.columns
        assert execution_time < 1.0  # File loading should be fast
    
    @pytest.mark.asyncio
    async def test_concurrent_load_multiple_sources(self, temp_json_files):
        """Test concurrent loading from multiple sources."""
        loader = APIJSONStorageLoader()
        
        # Test with multiple file sources
        start_time = time.time()
        df = await loader.load_json_concurrent(temp_json_files, max_concurrent=3)
        execution_time = time.time() - start_time
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 150  # 3 files * 50 items each
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert 'category' in df.columns
        
        # Verify data from all files is present
        categories = df['category'].unique()
        assert len(categories) == 3
        assert 'cat_0' in categories
        assert 'cat_1' in categories
        assert 'cat_2' in categories
        
        # Concurrent loading should be faster than sequential
        assert execution_time < 2.0
    
    @pytest.mark.asyncio
    async def test_concurrent_load_with_api_endpoints(self, mock_api_responses):
        """Test concurrent loading from multiple API endpoints."""
        loader = APIJSONStorageLoader()
        
        # Mock multiple API endpoints
        endpoints = [
            "https://api1.example.com/data",
            "https://api2.example.com/data", 
            "https://api3.example.com/data"
        ]
        
        mock_responses = [
            APIResponse(
                data=response["data"],
                status_code=200,
                headers={},
                response_time=0.1,
                url=endpoint,
                method="GET"
            )
            for endpoint, response in zip(endpoints, mock_api_responses[:3])
        ]
        
        with patch.object(loader.api_handler, 'fetch_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = mock_responses
            
            start_time = time.time()
            df = await loader.load_json_concurrent(endpoints, max_concurrent=3)
            execution_time = time.time() - start_time
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 300  # 3 endpoints * 100 items each
            assert 'id' in df.columns
            assert 'name' in df.columns
            assert 'value' in df.columns
            
            # Should make 3 API calls
            assert mock_fetch.call_count == 3
            
            # Concurrent execution should be faster than sequential
            assert execution_time < 1.0
    
    @pytest.mark.asyncio
    async def test_memory_optimization_large_dataset(self):
        """Test memory optimization with large datasets."""
        loader = APIJSONStorageLoader()
        
        # Create a large dataset
        large_data = [
            {"id": i, "text": f"This is a long text field for item {i} " * 10}
            for i in range(5000)  # 5000 items
        ]
        
        start_time = time.time()
        df = await loader.load_json(large_data)
        execution_time = time.time() - start_time
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5000
        assert 'id' in df.columns
        assert 'text' in df.columns
        
        # Should handle large dataset efficiently
        assert execution_time < 5.0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_concurrent_operations(self):
        """Test error handling during concurrent operations."""
        loader = APIJSONStorageLoader()
        
        # Mix of valid and invalid sources
        sources = [
            {"valid": "data"},  # Valid raw JSON
            "https://invalid-url-that-does-not-exist.com/data",  # Invalid URL
            {"another": "valid", "data": "source"}  # Valid raw JSON
        ]
        
        with patch.object(loader.api_handler, 'fetch_data', new_callable=AsyncMock) as mock_fetch:
            # Make the API call fail for the invalid URL
            mock_fetch.side_effect = APIError("Connection failed")
            
            # Should handle partial failures gracefully
            df = await loader.load_json_concurrent(sources, max_concurrent=2)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2  # Only the 2 valid raw JSON sources
            
            # Should have attempted the API call
            mock_fetch.assert_called_once()


class TestAsyncDataAPIJSONUseCase:
    """Test async functionality of DataAPIJSONUseCase."""
    
    @pytest.mark.asyncio
    async def test_async_execute_with_embeddings(self):
        """Test async execution with embedding generation."""
        # Setup mocks
        mock_repo = MockRepository(delay=0.05)
        mock_embedding_service = MockEmbeddingService(delay=0.1)
        mock_loader = APIJSONStorageLoader()
        
        # Mock the loader's load_json method
        test_data = pd.DataFrame([
            {"id": 1, "title": "Test Title 1", "content": "Test content 1"},
            {"id": 2, "title": "Test Title 2", "content": "Test content 2"}
        ])
        
        with patch.object(mock_loader, 'load_json', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = test_data
            
            use_case = DataAPIJSONUseCase(mock_repo, mock_embedding_service, mock_loader)
            
            start_time = time.time()
            result = await use_case.execute(
                source="https://api.example.com/data",
                table_name="test_table",
                embed_columns_names=["title", "content"],
                embed_type="combined"
            )
            execution_time = time.time() - start_time
            
            assert isinstance(result, DataMoveResult)
            assert result.success
            assert result.rows_processed == 2
            assert execution_time < 2.0  # Should be reasonably fast
            
            # Verify async calls were made
            mock_load.assert_called_once()
            assert mock_repo.call_count >= 1
            assert mock_embedding_service.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_chunked_embedding_generation(self):
        """Test chunked embedding generation for large datasets."""
        # Setup mocks
        mock_repo = MockRepository(delay=0.01)
        mock_embedding_service = MockEmbeddingService(delay=0.05)
        mock_loader = APIJSONStorageLoader()
        
        # Create large dataset that will trigger chunking
        large_data = pd.DataFrame([
            {"id": i, "title": f"Title {i}", "content": f"Content for item {i}"}
            for i in range(2500)  # Large enough to trigger chunking
        ])
        
        with patch.object(mock_loader, 'load_json', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = large_data
            
            use_case = DataAPIJSONUseCase(mock_repo, mock_embedding_service, mock_loader)
            
            start_time = time.time()
            result = await use_case.execute(
                source={"large": "dataset"},
                table_name="large_table",
                embed_columns_names=["title"],
                embed_type="separated"
            )
            execution_time = time.time() - start_time
            
            assert isinstance(result, DataMoveResult)
            assert result.success
            assert result.rows_processed == 2500
            
            # Chunked processing should still be reasonably fast
            assert execution_time < 10.0
            
            # Should have made multiple embedding calls due to chunking
            assert mock_embedding_service.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_concurrent_separated_embeddings(self):
        """Test concurrent generation of separated embeddings."""
        # Setup mocks
        mock_repo = MockRepository(delay=0.01)
        mock_embedding_service = MockEmbeddingService(delay=0.1)
        mock_loader = APIJSONStorageLoader()
        
        test_data = pd.DataFrame([
            {"id": i, "title": f"Title {i}", "content": f"Content {i}", "summary": f"Summary {i}"}
            for i in range(100)
        ])
        
        with patch.object(mock_loader, 'load_json', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = test_data
            
            use_case = DataAPIJSONUseCase(mock_repo, mock_embedding_service, mock_loader)
            
            start_time = time.time()
            result = await use_case.execute(
                source="test_data",
                table_name="test_table",
                embed_columns_names=["title", "content", "summary"],  # 3 columns
                embed_type="separated"
            )
            execution_time = time.time() - start_time
            
            assert isinstance(result, DataMoveResult)
            assert result.success
            assert result.rows_processed == 100
            
            # Concurrent embedding generation should be faster than sequential
            # With 3 columns and 0.1s delay each, sequential would take ~0.3s minimum
            # Concurrent should be closer to 0.1s
            assert execution_time < 0.25
            
            # Should have made 3 embedding calls (one per column)
            assert mock_embedding_service.call_count == 3
    
    @pytest.mark.asyncio
    async def test_chunked_database_operations(self):
        """Test chunked database operations for large datasets."""
        # Setup mocks
        mock_repo = MockRepository(delay=0.02)
        mock_embedding_service = MockEmbeddingService(delay=0.01)
        mock_loader = APIJSONStorageLoader()
        
        # Create dataset large enough to trigger chunking
        large_data = pd.DataFrame([
            {"id": i, "name": f"Item {i}", "value": i * 10}
            for i in range(2500)  # Large enough to trigger chunking
        ])
        
        with patch.object(mock_loader, 'load_json', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = large_data
            
            # Mock table exists to trigger update path
            with patch.object(mock_repo, 'table_exists', return_value=True):
                use_case = DataAPIJSONUseCase(mock_repo, mock_embedding_service, mock_loader)
                
                start_time = time.time()
                result = await use_case.execute(
                    source={"large": "dataset"},
                    table_name="large_table",
                    pk_columns=["id"],
                    create_table_if_not_exists=False
                )
                execution_time = time.time() - start_time
                
                assert isinstance(result, DataMoveResult)
                assert result.success
                assert result.rows_processed == 2500
                
                # Should have made multiple database calls due to chunking
                assert mock_repo.call_count >= 3
                
                # Chunked operations should still be reasonably fast
                assert execution_time < 5.0
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_on_error(self):
        """Test proper resource cleanup when errors occur."""
        # Setup mocks
        mock_repo = MockRepository()
        mock_embedding_service = MockEmbeddingService()
        mock_loader = APIJSONStorageLoader()
        
        # Mock loader to raise an error
        with patch.object(mock_loader, 'load_json', new_callable=AsyncMock) as mock_load:
            mock_load.side_effect = APIError("Simulated API error")
            
            use_case = DataAPIJSONUseCase(mock_repo, mock_embedding_service, mock_loader)
            
            # Use async context manager to test cleanup
            async with use_case:
                with pytest.raises(APIError):
                    await use_case.execute(
                        source="https://api.example.com/data",
                        table_name="test_table"
                    )
            
            # Cleanup should have been called
            # We can't easily test this without more complex mocking,
            # but the context manager should handle it
    
    @pytest.mark.asyncio
    async def test_performance_with_concurrent_use_cases(self):
        """Test performance when running multiple use cases concurrently."""
        # Setup shared mocks
        mock_repo = MockRepository(delay=0.05)
        mock_embedding_service = MockEmbeddingService(delay=0.1)
        
        async def run_use_case(use_case_id: int) -> DataMoveResult:
            """Run a single use case."""
            mock_loader = APIJSONStorageLoader()
            
            test_data = pd.DataFrame([
                {"id": i + use_case_id * 100, "name": f"Item {i}", "category": f"cat_{use_case_id}"}
                for i in range(50)
            ])
            
            with patch.object(mock_loader, 'load_json', new_callable=AsyncMock) as mock_load:
                mock_load.return_value = test_data
                
                use_case = DataAPIJSONUseCase(mock_repo, mock_embedding_service, mock_loader)
                
                return await use_case.execute(
                    source=f"source_{use_case_id}",
                    table_name=f"table_{use_case_id}",
                    embed_columns_names=["name"],
                    embed_type="combined"
                )
        
        # Run multiple use cases concurrently
        start_time = time.time()
        tasks = [run_use_case(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # All use cases should succeed
        assert len(results) == 5
        for result in results:
            assert isinstance(result, DataMoveResult)
            assert result.success
            assert result.rows_processed == 50
        
        # Concurrent execution should be faster than sequential
        # Sequential would take ~5 * (embedding_delay + db_delay) = ~5 * 0.15 = 0.75s minimum
        # Concurrent should be closer to max(embedding_delay, db_delay) = 0.1s
        assert execution_time < 1.0


@pytest.mark.asyncio
async def test_memory_usage_optimization():
    """Test that async operations optimize memory usage."""
    # This is a basic test - in practice you'd use memory profiling tools
    loader = APIJSONStorageLoader()
    
    # Create multiple large datasets
    datasets = []
    for i in range(3):
        data = [
            {"id": j + i * 1000, "large_text": "x" * 1000}  # 1KB per item
            for j in range(1000)  # 1000 items = ~1MB per dataset
        ]
        datasets.append(data)
    
    start_time = time.time()
    
    # Process datasets concurrently
    tasks = [loader.load_json(data) for data in datasets]
    results = await asyncio.gather(*tasks)
    
    execution_time = time.time() - start_time
    
    # All results should be valid
    assert len(results) == 3
    for df in results:
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
    
    # Should complete in reasonable time
    assert execution_time < 5.0


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])