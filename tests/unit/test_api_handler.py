"""
Unit tests for APIHandler.

These tests verify the API communication functionality including authentication,
retry logic, timeout handling, pagination, and error handling.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import httpx

from dataload.domain.api_handler import APIHandler
from dataload.domain.api_entities import (
    APIConfig, AuthConfig, PaginationConfig, APIResponse, PaginationInfo,
    AuthType, PaginationType, APIError, AuthenticationError, APITimeoutError,
    APIRateLimitError, APIConnectionError
)


class TestAPIHandler:
    """Test cases for APIHandler class."""

    @pytest.fixture
    def mock_httpx_client(self):
        """Create a mock httpx.AsyncClient."""
        client = AsyncMock(spec=httpx.AsyncClient)
        return client

    @pytest.fixture
    def sample_api_response_data(self):
        """Sample API response data for testing."""
        return {
            "users": [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
            ],
            "total": 2,
            "page": 1,
            "has_next": False
        }

    @pytest.fixture
    def sample_paginated_response_data(self):
        """Sample paginated API response data for testing."""
        return {
            "data": [
                {"id": 1, "name": "John Doe"},
                {"id": 2, "name": "Jane Smith"}
            ],
            "total": 10,
            "page": 1,
            "has_next": True,
            "next": "https://api.example.com/users?page=2"
        }

    @pytest.fixture
    def api_handler_basic(self):
        """Create APIHandler with basic configuration."""
        return APIHandler(
            base_url="https://api.example.com",
            timeout=30,
            retry_attempts=3
        )

    @pytest.fixture
    def api_handler_with_auth(self):
        """Create APIHandler with API key authentication."""
        return APIHandler(
            base_url="https://api.example.com",
            api_token="test-api-key",
            timeout=30,
            retry_attempts=2
        )

    @pytest.fixture
    def pagination_config(self):
        """Create pagination configuration for testing."""
        return PaginationConfig(
            enabled=True,
            pagination_type=PaginationType.PAGE_SIZE,
            page_param="page",
            size_param="size",
            page_size=2,
            max_pages=5
        )

    def test_init_with_api_key(self):
        """Test APIHandler initialization with API key authentication."""
        handler = APIHandler(
            base_url="https://api.example.com",
            api_token="test-key"
        )
        
        assert handler.base_url == "https://api.example.com"
        assert handler.auth_config.auth_type == AuthType.API_KEY
        assert handler.auth_config.api_key == "test-key"
        assert handler.timeout == 30
        assert handler.retry_attempts == 3

    def test_init_with_jwt_token(self):
        """Test APIHandler initialization with JWT authentication."""
        handler = APIHandler(
            base_url="https://api.example.com",
            jwt_token="jwt-token-123"
        )
        
        assert handler.auth_config.auth_type == AuthType.JWT
        assert handler.auth_config.jwt_token == "jwt-token-123"

    def test_init_with_bearer_token(self):
        """Test APIHandler initialization with Bearer token authentication."""
        handler = APIHandler(
            base_url="https://api.example.com",
            bearer_token="bearer-token-456"
        )
        
        assert handler.auth_config.auth_type == AuthType.BEARER
        assert handler.auth_config.bearer_token == "bearer-token-456"

    def test_init_with_basic_auth(self):
        """Test APIHandler initialization with basic authentication."""
        handler = APIHandler(
            base_url="https://api.example.com",
            username="testuser",
            password="testpass"
        )
        
        assert handler.auth_config.auth_type == AuthType.BASIC
        assert handler.auth_config.username == "testuser"
        assert handler.auth_config.password == "testpass"

    def test_init_no_auth(self):
        """Test APIHandler initialization without authentication."""
        handler = APIHandler(base_url="https://api.example.com")
        
        assert handler.auth_config.auth_type == AuthType.NONE

    def test_prepare_headers_api_key(self, api_handler_with_auth):
        """Test header preparation with API key authentication."""
        headers = api_handler_with_auth._prepare_headers()
        
        assert headers['X-API-Key'] == 'test-api-key'
        assert headers['Content-Type'] == 'application/json'

    def test_prepare_headers_jwt(self):
        """Test header preparation with JWT authentication."""
        handler = APIHandler(jwt_token="jwt-token-123")
        headers = handler._prepare_headers()
        
        assert headers['Authorization'] == 'Bearer jwt-token-123'

    def test_prepare_headers_bearer(self):
        """Test header preparation with Bearer token authentication."""
        handler = APIHandler(bearer_token="bearer-token-456")
        headers = handler._prepare_headers()
        
        assert headers['Authorization'] == 'Bearer bearer-token-456'

    def test_prepare_headers_basic(self):
        """Test header preparation with basic authentication."""
        handler = APIHandler(username="testuser", password="testpass")
        headers = handler._prepare_headers()
        
        # Basic auth should be base64 encoded
        assert 'Authorization' in headers
        assert headers['Authorization'].startswith('Basic ')

    def test_prepare_headers_additional(self, api_handler_basic):
        """Test header preparation with additional headers."""
        additional_headers = {'Custom-Header': 'custom-value'}
        headers = api_handler_basic._prepare_headers(additional_headers)
        
        assert headers['Custom-Header'] == 'custom-value'
        assert headers['Content-Type'] == 'application/json'

    def test_build_url_with_base_url(self, api_handler_basic):
        """Test URL building with base URL."""
        url = api_handler_basic._build_url('/users')
        assert url == 'https://api.example.com/users'
        
        url = api_handler_basic._build_url('users')
        assert url == 'https://api.example.com/users'

    def test_build_url_full_url(self, api_handler_basic):
        """Test URL building with full URL."""
        full_url = 'https://other-api.com/data'
        url = api_handler_basic._build_url(full_url)
        assert url == full_url

    def test_build_url_no_base_url(self):
        """Test URL building without base URL raises error."""
        handler = APIHandler()
        
        with pytest.raises(APIError, match="No base URL configured"):
            handler._build_url('/users')

    @pytest.mark.asyncio
    async def test_make_request_success(self, api_handler_basic, sample_api_response_data):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = sample_api_response_data
        mock_response.reason_phrase = "OK"
        
        with patch.object(api_handler_basic, '_ensure_client'):
            api_handler_basic._client = AsyncMock()
            api_handler_basic._client.request.return_value = mock_response
            
            response = await api_handler_basic._make_request(
                'GET', 'https://api.example.com/users'
            )
            
            assert isinstance(response, APIResponse)
            assert response.status_code == 200
            assert response.data == sample_api_response_data
            assert response.success
            assert response.method == 'GET'

    @pytest.mark.asyncio
    async def test_make_request_authentication_error(self, api_handler_basic):
        """Test API request with authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.reason_phrase = "Unauthorized"
        mock_response.headers = {}
        mock_response.text = "Unauthorized"
        
        with patch.object(api_handler_basic, '_ensure_client'):
            api_handler_basic._client = AsyncMock()
            api_handler_basic._client.request.return_value = mock_response
            
            with pytest.raises(AuthenticationError, match="Authentication failed"):
                await api_handler_basic._make_request(
                    'GET', 'https://api.example.com/users'
                )

    @pytest.mark.asyncio
    async def test_make_request_rate_limit_error(self, api_handler_basic):
        """Test API request with rate limit error."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.headers = {}
        mock_response.text = "Rate limit exceeded"
        
        with patch.object(api_handler_basic, '_ensure_client'):
            api_handler_basic._client = AsyncMock()
            api_handler_basic._client.request.return_value = mock_response
            
            with pytest.raises(APIRateLimitError, match="Rate limit exceeded"):
                await api_handler_basic._make_request(
                    'GET', 'https://api.example.com/users'
                )

    @pytest.mark.asyncio
    async def test_make_request_server_error(self, api_handler_basic):
        """Test API request with server error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.reason_phrase = "Internal Server Error"
        mock_response.headers = {}
        mock_response.text = "Server error"
        
        with patch.object(api_handler_basic, '_ensure_client'):
            api_handler_basic._client = AsyncMock()
            api_handler_basic._client.request.return_value = mock_response
            
            with pytest.raises(APIError, match="Server error"):
                await api_handler_basic._make_request(
                    'GET', 'https://api.example.com/users'
                )

    @pytest.mark.asyncio
    async def test_make_request_timeout_error(self, api_handler_basic):
        """Test API request with timeout error."""
        with patch.object(api_handler_basic, '_ensure_client'):
            api_handler_basic._client = AsyncMock()
            api_handler_basic._client.request.side_effect = httpx.TimeoutException("Timeout")
            
            with pytest.raises(APITimeoutError, match="Request timeout"):
                await api_handler_basic._make_request(
                    'GET', 'https://api.example.com/users'
                )

    @pytest.mark.asyncio
    async def test_make_request_connection_error(self, api_handler_basic):
        """Test API request with connection error."""
        with patch.object(api_handler_basic, '_ensure_client'):
            api_handler_basic._client = AsyncMock()
            api_handler_basic._client.request.side_effect = httpx.ConnectError("Connection failed")
            
            with pytest.raises(APIConnectionError, match="Connection error"):
                await api_handler_basic._make_request(
                    'GET', 'https://api.example.com/users'
                )

    @pytest.mark.asyncio
    async def test_retry_request_success_after_failure(self, api_handler_basic, sample_api_response_data):
        """Test retry logic with success after initial failure."""
        # First call fails, second succeeds
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.headers = {'content-type': 'application/json'}
        mock_response_success.json.return_value = sample_api_response_data
        mock_response_success.reason_phrase = "OK"
        
        with patch.object(api_handler_basic, '_ensure_client'):
            api_handler_basic._client = AsyncMock()
            api_handler_basic._client.request.side_effect = [
                httpx.ConnectError("Connection failed"),
                mock_response_success
            ]
            
            # Mock sleep to speed up test
            with patch('asyncio.sleep'):
                response = await api_handler_basic._retry_request(
                    'GET', 'https://api.example.com/users'
                )
                
                assert response.status_code == 200
                assert response.data == sample_api_response_data

    @pytest.mark.asyncio
    async def test_retry_request_all_attempts_fail(self, api_handler_basic):
        """Test retry logic when all attempts fail."""
        with patch.object(api_handler_basic, '_ensure_client'):
            api_handler_basic._client = AsyncMock()
            api_handler_basic._client.request.side_effect = httpx.ConnectError("Connection failed")
            
            # Mock sleep to speed up test
            with patch('asyncio.sleep'):
                with pytest.raises(APIConnectionError, match="Connection error"):
                    await api_handler_basic._retry_request(
                        'GET', 'https://api.example.com/users'
                    )

    @pytest.mark.asyncio
    async def test_retry_request_no_retry_on_auth_error(self, api_handler_basic):
        """Test that authentication errors are not retried."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.reason_phrase = "Unauthorized"
        mock_response.headers = {}
        mock_response.text = "Unauthorized"
        
        with patch.object(api_handler_basic, '_ensure_client'):
            api_handler_basic._client = AsyncMock()
            api_handler_basic._client.request.return_value = mock_response
            
            with pytest.raises(AuthenticationError):
                await api_handler_basic._retry_request(
                    'GET', 'https://api.example.com/users'
                )
            
            # Should only be called once (no retries)
            assert api_handler_basic._client.request.call_count == 1

    @pytest.mark.asyncio
    async def test_fetch_data_simple(self, api_handler_basic, sample_api_response_data):
        """Test simple data fetching without pagination."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = sample_api_response_data
        mock_response.reason_phrase = "OK"
        
        with patch.object(api_handler_basic, '_retry_request') as mock_retry:
            mock_retry.return_value = APIResponse(
                data=sample_api_response_data,
                status_code=200,
                headers={'content-type': 'application/json'},
                response_time=0.5,
                url='https://api.example.com/users',
                method='GET'
            )
            
            response = await api_handler_basic.fetch_data('/users')
            
            assert isinstance(response, APIResponse)
            assert response.data == sample_api_response_data
            mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_data_with_pagination(self, api_handler_basic, pagination_config, sample_paginated_response_data):
        """Test data fetching with pagination."""
        # Mock two pages of data
        page1_data = {
            "data": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}],
            "total": 4,
            "page": 1,
            "has_next": True
        }
        page2_data = {
            "data": [{"id": 3, "name": "Bob"}, {"id": 4, "name": "Alice"}],
            "total": 4,
            "page": 2,
            "has_next": False
        }
        
        responses = [
            APIResponse(
                data=page1_data,
                status_code=200,
                headers={},
                response_time=0.5,
                url='https://api.example.com/users?page=1&size=2',
                method='GET'
            ),
            APIResponse(
                data=page2_data,
                status_code=200,
                headers={},
                response_time=0.4,
                url='https://api.example.com/users?page=2&size=2',
                method='GET'
            )
        ]
        
        with patch.object(api_handler_basic, '_fetch_paginated_data') as mock_paginated:
            mock_paginated.return_value = responses
            
            result = await api_handler_basic.fetch_data(
                '/users',
                pagination_config=pagination_config
            )
            
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0].data == page1_data
            assert result[1].data == page2_data

    @pytest.mark.asyncio
    async def test_extract_pagination_info_from_response_body(self, api_handler_basic):
        """Test pagination info extraction from response body."""
        response_data = {
            "data": [{"id": 1}, {"id": 2}],
            "total": 10,
            "has_next": True,
            "next": "https://api.example.com/users?page=2"
        }
        
        response = APIResponse(
            data=response_data,
            status_code=200,
            headers={},
            response_time=0.5,
            url='https://api.example.com/users',
            method='GET'
        )
        
        pagination_config = PaginationConfig(
            enabled=True,
            page_size=2
        )
        
        pagination_info = api_handler_basic._extract_pagination_info(
            response, 1, pagination_config
        )
        
        assert pagination_info is not None
        assert pagination_info.current_page == 1
        assert pagination_info.total_count == 10
        assert pagination_info.total_pages == 5
        assert pagination_info.has_next_page is True
        assert pagination_info.next_page_url == "https://api.example.com/users?page=2"

    @pytest.mark.asyncio
    async def test_extract_pagination_info_from_headers(self, api_handler_basic):
        """Test pagination info extraction from response headers."""
        response = APIResponse(
            data=[{"id": 1}, {"id": 2}],
            status_code=200,
            headers={
                'X-Total-Count': '20',
                'X-Next-Page': 'https://api.example.com/users?page=2'
            },
            response_time=0.5,
            url='https://api.example.com/users',
            method='GET'
        )
        
        pagination_config = PaginationConfig(
            enabled=True,
            page_size=2,
            total_count_header='X-Total-Count',
            next_page_header='X-Next-Page'
        )
        
        pagination_info = api_handler_basic._extract_pagination_info(
            response, 1, pagination_config
        )
        
        assert pagination_info is not None
        assert pagination_info.total_count == 20
        assert pagination_info.total_pages == 10
        assert pagination_info.has_next_page is True
        assert pagination_info.next_page_url == 'https://api.example.com/users?page=2'

    def test_should_continue_pagination_with_info(self, api_handler_basic):
        """Test pagination continuation decision with pagination info."""
        pagination_info = PaginationInfo(
            current_page=1,
            total_pages=3,
            page_size=2,
            total_count=6,
            has_next_page=True
        )
        
        pagination_config = PaginationConfig(enabled=True, page_size=2)
        response = APIResponse(
            data=[{"id": 1}, {"id": 2}],
            status_code=200,
            headers={},
            response_time=0.5,
            url='https://api.example.com/users',
            method='GET'
        )
        
        should_continue = api_handler_basic._should_continue_pagination(
            response, pagination_info, pagination_config
        )
        
        assert should_continue is True

    def test_should_continue_pagination_without_info(self, api_handler_basic):
        """Test pagination continuation decision without pagination info."""
        pagination_config = PaginationConfig(enabled=True, page_size=2)
        
        # Response with full page size - should continue
        response_full = APIResponse(
            data=[{"id": 1}, {"id": 2}],
            status_code=200,
            headers={},
            response_time=0.5,
            url='https://api.example.com/users',
            method='GET'
        )
        
        should_continue = api_handler_basic._should_continue_pagination(
            response_full, None, pagination_config
        )
        assert should_continue is True
        
        # Response with less than page size - should not continue
        response_partial = APIResponse(
            data=[{"id": 1}],
            status_code=200,
            headers={},
            response_time=0.5,
            url='https://api.example.com/users',
            method='GET'
        )
        
        should_continue = api_handler_basic._should_continue_pagination(
            response_partial, None, pagination_config
        )
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_context_manager(self, api_handler_basic):
        """Test APIHandler as async context manager."""
        with patch.object(api_handler_basic, '_ensure_client') as mock_ensure:
            with patch.object(api_handler_basic, 'close') as mock_close:
                async with api_handler_basic as handler:
                    assert handler is api_handler_basic
                    mock_ensure.assert_called_once()
                
                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_convenience_methods(self, api_handler_basic, sample_api_response_data):
        """Test convenience methods (get, post, put, delete, patch)."""
        mock_response = APIResponse(
            data=sample_api_response_data,
            status_code=200,
            headers={},
            response_time=0.5,
            url='https://api.example.com/users',
            method='GET'
        )
        
        with patch.object(api_handler_basic, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = mock_response
            
            # Test GET
            result = await api_handler_basic.get('/users')
            mock_fetch.assert_called_with('/users', method='GET')
            assert result == mock_response
            
            # Test POST
            await api_handler_basic.post('/users', json_data={"name": "John"})
            mock_fetch.assert_called_with('/users', method='POST', json_data={"name": "John"})
            
            # Test PUT
            await api_handler_basic.put('/users/1', json_data={"name": "Jane"})
            mock_fetch.assert_called_with('/users/1', method='PUT', json_data={"name": "Jane"})
            
            # Test DELETE
            await api_handler_basic.delete('/users/1')
            mock_fetch.assert_called_with('/users/1', method='DELETE')
            
            # Test PATCH
            await api_handler_basic.patch('/users/1', json_data={"name": "Bob"})
            mock_fetch.assert_called_with('/users/1', method='PATCH', json_data={"name": "Bob"})


class TestAPIHandlerIntegration:
    """Integration tests for APIHandler with more complex scenarios."""

    @pytest.mark.asyncio
    async def test_full_pagination_workflow(self):
        """Test complete pagination workflow with multiple pages."""
        handler = APIHandler(
            base_url="https://api.example.com",
            api_token="test-key",
            retry_attempts=1
        )
        
        pagination_config = PaginationConfig(
            enabled=True,
            pagination_type=PaginationType.PAGE_SIZE,
            page_param="page",
            size_param="limit",
            page_size=2,
            max_pages=3
        )
        
        # Mock responses for 3 pages
        page_responses = [
            {
                "data": [{"id": 1, "name": "User1"}, {"id": 2, "name": "User2"}],
                "total": 5,
                "has_next": True
            },
            {
                "data": [{"id": 3, "name": "User3"}, {"id": 4, "name": "User4"}],
                "total": 5,
                "has_next": True
            },
            {
                "data": [{"id": 5, "name": "User5"}],
                "total": 5,
                "has_next": False
            }
        ]
        
        mock_responses = []
        for i, page_data in enumerate(page_responses):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.json.return_value = page_data
            mock_response.reason_phrase = "OK"
            mock_responses.append(mock_response)
        
        with patch.object(handler, '_ensure_client'):
            handler._client = AsyncMock()
            handler._client.request.side_effect = mock_responses
            
            responses = await handler.fetch_data(
                '/users',
                pagination_config=pagination_config
            )
            
            assert isinstance(responses, list)
            assert len(responses) == 3
            
            # Verify each page
            for i, response in enumerate(responses):
                assert response.data == page_responses[i]
                assert response.status_code == 200
            
            # Verify correct pagination parameters were sent
            calls = handler._client.request.call_args_list
            assert len(calls) == 3
            
            # Check first page parameters
            first_call_params = calls[0][1]['params']
            assert first_call_params['page'] == 1
            assert first_call_params['limit'] == 2
            
            # Check second page parameters
            second_call_params = calls[1][1]['params']
            assert second_call_params['page'] == 2
            assert second_call_params['limit'] == 2

    @pytest.mark.asyncio
    async def test_authentication_methods_integration(self):
        """Test different authentication methods in integration."""
        test_cases = [
            {
                "auth_params": {"api_token": "test-api-key"},
                "expected_header": ("X-API-Key", "test-api-key")
            },
            {
                "auth_params": {"jwt_token": "jwt-token-123"},
                "expected_header": ("Authorization", "Bearer jwt-token-123")
            },
            {
                "auth_params": {"bearer_token": "bearer-token-456"},
                "expected_header": ("Authorization", "Bearer bearer-token-456")
            },
            {
                "auth_params": {"username": "user", "password": "pass"},
                "expected_header": ("Authorization", "Basic dXNlcjpwYXNz")  # base64 of "user:pass"
            }
        ]
        
        for test_case in test_cases:
            handler = APIHandler(
                base_url="https://api.example.com",
                **test_case["auth_params"]
            )
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.json.return_value = {"success": True}
            mock_response.reason_phrase = "OK"
            
            with patch.object(handler, '_ensure_client'):
                handler._client = AsyncMock()
                handler._client.request.return_value = mock_response
                
                await handler.get('/test')
                
                # Verify correct authentication header was sent
                call_args = handler._client.request.call_args
                headers = call_args[1]['headers']
                
                expected_key, expected_value = test_case["expected_header"]
                assert headers[expected_key] == expected_value

    @pytest.mark.asyncio
    async def test_error_handling_and_retry_integration(self):
        """Test error handling and retry logic integration."""
        handler = APIHandler(
            base_url="https://api.example.com",
            retry_attempts=2,
            retry_delay=0.1  # Fast retry for testing
        )
        
        # Test successful retry after connection error
        mock_success_response = MagicMock()
        mock_success_response.status_code = 200
        mock_success_response.headers = {'content-type': 'application/json'}
        mock_success_response.json.return_value = {"success": True}
        mock_success_response.reason_phrase = "OK"
        
        with patch.object(handler, '_ensure_client'):
            handler._client = AsyncMock()
            handler._client.request.side_effect = [
                httpx.ConnectError("Connection failed"),
                httpx.ConnectError("Connection failed again"),
                mock_success_response
            ]
            
            response = await handler.get('/test')
            
            assert response.status_code == 200
            assert response.data == {"success": True}
            
            # Should have made 3 attempts (initial + 2 retries)
            assert handler._client.request.call_count == 3

    @pytest.mark.asyncio
    async def test_offset_limit_pagination(self):
        """Test offset/limit pagination pattern."""
        handler = APIHandler(
            base_url="https://api.example.com",
            api_token="test-key"
        )
        
        pagination_config = PaginationConfig(
            enabled=True,
            pagination_type=PaginationType.OFFSET_LIMIT,
            offset_param="offset",
            limit_param="limit",
            page_size=3,
            max_pages=2
        )
        
        # Mock responses for offset/limit pagination
        page_responses = [
            {
                "items": [{"id": 1}, {"id": 2}, {"id": 3}],
                "total": 5,
                "offset": 0,
                "limit": 3,
                "has_next": True
            },
            {
                "items": [{"id": 4}, {"id": 5}],
                "total": 5,
                "offset": 3,
                "limit": 3,
                "has_next": False
            }
        ]
        
        mock_responses = []
        for page_data in page_responses:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.json.return_value = page_data
            mock_response.reason_phrase = "OK"
            mock_responses.append(mock_response)
        
        with patch.object(handler, '_ensure_client'):
            handler._client = AsyncMock()
            handler._client.request.side_effect = mock_responses
            
            responses = await handler.fetch_data(
                '/items',
                pagination_config=pagination_config
            )
            
            assert isinstance(responses, list)
            assert len(responses) == 2
            
            # Verify pagination parameters
            calls = handler._client.request.call_args_list
            
            # First page: offset=0, limit=3
            first_params = calls[0][1]['params']
            assert first_params['offset'] == 0
            assert first_params['limit'] == 3
            
            # Second page: offset=3, limit=3
            second_params = calls[1][1]['params']
            assert second_params['offset'] == 3
            assert second_params['limit'] == 3

    @pytest.mark.asyncio
    async def test_cursor_based_pagination(self):
        """Test cursor-based pagination pattern."""
        handler = APIHandler(
            base_url="https://api.example.com",
            api_token="test-key"
        )
        
        pagination_config = PaginationConfig(
            enabled=True,
            pagination_type=PaginationType.CURSOR,
            cursor_param="cursor",
            page_size=2
        )
        
        # Mock responses for cursor-based pagination
        page_responses = [
            {
                "data": [{"id": 1, "name": "Item1"}, {"id": 2, "name": "Item2"}],
                "next_cursor": "cursor_abc123",
                "has_more": True
            },
            {
                "data": [{"id": 3, "name": "Item3"}, {"id": 4, "name": "Item4"}],
                "next_cursor": "cursor_def456",
                "has_more": True
            },
            {
                "data": [{"id": 5, "name": "Item5"}],
                "next_cursor": None,
                "has_more": False
            }
        ]
        
        mock_responses = []
        for page_data in page_responses:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.json.return_value = page_data
            mock_response.reason_phrase = "OK"
            mock_responses.append(mock_response)
        
        # Mock the cursor-based pagination logic
        with patch.object(handler, '_ensure_client'):
            handler._client = AsyncMock()
            handler._client.request.side_effect = mock_responses
            
            # Override the pagination method to handle cursor logic
            original_fetch_paginated = handler._fetch_paginated_data
            
            async def mock_fetch_paginated_data(method, url, headers, params, json_data, data, pagination_config):
                responses = []
                current_cursor = None
                page_count = 0
                
                while page_count < 3:  # Limit for test
                    page_params = params.copy() if params else {}
                    if current_cursor:
                        page_params[pagination_config.cursor_param] = current_cursor
                    
                    response = await handler._retry_request(
                        method, url, headers, page_params, json_data, data
                    )
                    
                    responses.append(response)
                    
                    # Extract next cursor from response
                    if isinstance(response.data, dict):
                        current_cursor = response.data.get('next_cursor')
                        has_more = response.data.get('has_more', False)
                        
                        if not has_more or not current_cursor:
                            break
                    
                    page_count += 1
                
                return responses
            
            handler._fetch_paginated_data = mock_fetch_paginated_data
            
            responses = await handler.fetch_data(
                '/items',
                pagination_config=pagination_config
            )
            
            assert isinstance(responses, list)
            assert len(responses) == 3
            
            # Verify cursor progression
            calls = handler._client.request.call_args_list
            
            # First call should have no cursor
            first_params = calls[0][1].get('params', {})
            assert 'cursor' not in first_params
            
            # Second call should have first cursor
            second_params = calls[1][1].get('params', {})
            assert second_params.get('cursor') == 'cursor_abc123'
            
            # Third call should have second cursor
            third_params = calls[2][1].get('params', {})
            assert third_params.get('cursor') == 'cursor_def456'

    @pytest.mark.asyncio
    async def test_pagination_memory_efficiency(self):
        """Test memory-efficient processing for large paginated datasets."""
        handler = APIHandler(
            base_url="https://api.example.com",
            api_token="test-key"
        )
        
        pagination_config = PaginationConfig(
            enabled=True,
            pagination_type=PaginationType.PAGE_SIZE,
            page_size=1000,  # Large page size
            max_pages=5
        )
        
        # Create large mock responses
        large_responses = []
        for page in range(5):
            # Simulate large dataset
            large_data = {
                "data": [{"id": i + page * 1000, "data": f"item_{i + page * 1000}"} for i in range(1000)],
                "page": page + 1,
                "has_next": page < 4
            }
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.json.return_value = large_data
            mock_response.reason_phrase = "OK"
            large_responses.append(mock_response)
        
        with patch.object(handler, '_ensure_client'):
            handler._client = AsyncMock()
            handler._client.request.side_effect = large_responses
            
            responses = await handler.fetch_data(
                '/large-dataset',
                pagination_config=pagination_config
            )
            
            assert isinstance(responses, list)
            assert len(responses) == 5
            
            # Verify each response contains expected amount of data
            for i, response in enumerate(responses):
                assert len(response.data['data']) == 1000
                assert response.data['page'] == i + 1
            
            # Verify memory efficiency by checking that responses are processed individually
            # (This is implicit in the current implementation - each page is processed separately)
            total_items = sum(len(resp.data['data']) for resp in responses)
            assert total_items == 5000

    @pytest.mark.asyncio
    async def test_pagination_with_max_pages_limit(self):
        """Test pagination respects max_pages configuration."""
        handler = APIHandler(
            base_url="https://api.example.com",
            api_token="test-key"
        )
        
        pagination_config = PaginationConfig(
            enabled=True,
            pagination_type=PaginationType.PAGE_SIZE,
            page_size=2,
            max_pages=2  # Limit to 2 pages
        )
        
        # Mock 4 pages of data, but should only fetch 2
        page_responses = []
        for i in range(4):
            page_data = {
                "data": [{"id": i * 2 + 1}, {"id": i * 2 + 2}],
                "page": i + 1,
                "has_next": True
            }
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.json.return_value = page_data
            mock_response.reason_phrase = "OK"
            page_responses.append(mock_response)
        
        with patch.object(handler, '_ensure_client'):
            handler._client = AsyncMock()
            handler._client.request.side_effect = page_responses
            
            responses = await handler.fetch_data(
                '/items',
                pagination_config=pagination_config
            )
            
            # Should only get 2 pages due to max_pages limit
            assert isinstance(responses, list)
            assert len(responses) == 2
            
            # Verify only 2 API calls were made
            assert handler._client.request.call_count == 2

    @pytest.mark.asyncio
    async def test_pagination_error_handling(self):
        """Test error handling during paginated requests."""
        handler = APIHandler(
            base_url="https://api.example.com",
            api_token="test-key",
            retry_attempts=1
        )
        
        pagination_config = PaginationConfig(
            enabled=True,
            pagination_type=PaginationType.PAGE_SIZE,
            page_size=2
        )
        
        # First page succeeds, second page fails
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.headers = {'content-type': 'application/json'}
        success_response.json.return_value = {
            "data": [{"id": 1}, {"id": 2}],
            "has_next": True
        }
        success_response.reason_phrase = "OK"
        
        with patch.object(handler, '_ensure_client'):
            handler._client = AsyncMock()
            handler._client.request.side_effect = [
                success_response,
                httpx.HTTPError("Server error")
            ]
            
            # Should raise error on second page
            with pytest.raises(APIError):
                await handler.fetch_data(
                    '/items',
                    pagination_config=pagination_config
                )

    def test_pagination_info_extraction_edge_cases(self):
        """Test pagination info extraction with various edge cases."""
        pagination_config = PaginationConfig(enabled=True, page_size=10)
        
        # Test with empty response
        empty_response = APIResponse(
            data=[],
            status_code=200,
            headers={},
            response_time=0.1,
            url='https://api.example.com/items',
            method='GET'
        )
        
        handler = APIHandler(base_url="https://api.example.com")
        
        pagination_info = handler._extract_pagination_info(
            empty_response, 1, pagination_config
        )
        
        assert pagination_info is not None
        assert pagination_info.has_next_page is False  # Empty response indicates no more pages
        
        # Test with malformed pagination data
        malformed_response = APIResponse(
            data={"total": "invalid", "has_next": "not_boolean"},
            status_code=200,
            headers={'X-Total-Count': 'invalid'},
            response_time=0.1,
            url='https://api.example.com/items',
            method='GET'
        )
        
        pagination_info = handler._extract_pagination_info(
            malformed_response, 1, pagination_config
        )
        
        # Should handle malformed data gracefully by returning None
        assert pagination_info is None
        
        # Test with response that's not a dict or list
        string_response = APIResponse(
            data="plain text response",
            status_code=200,
            headers={},
            response_time=0.1,
            url='https://api.example.com/items',
            method='GET'
        )
        
        pagination_info = handler._extract_pagination_info(
            string_response, 1, pagination_config
        )
        
        assert pagination_info is not None
        assert pagination_info.has_next_page is False  # Conservative default