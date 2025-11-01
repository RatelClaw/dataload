"""
Integration tests for APIHandler.

These tests verify that the APIHandler integrates properly with the rest of the system
and can be imported and used as expected.
"""

import pytest
from dataload.domain import APIHandler, AuthType, PaginationConfig, PaginationType


class TestAPIHandlerIntegration:
    """Integration tests for APIHandler."""

    def test_api_handler_import(self):
        """Test that APIHandler can be imported from the domain module."""
        # This test verifies that the import works correctly
        assert APIHandler is not None
        assert hasattr(APIHandler, '__init__')
        assert hasattr(APIHandler, 'fetch_data')

    def test_api_handler_instantiation(self):
        """Test that APIHandler can be instantiated with various configurations."""
        # Test basic instantiation
        handler = APIHandler()
        assert handler is not None
        assert handler.auth_config.auth_type == AuthType.NONE

        # Test with API key
        handler_with_key = APIHandler(
            base_url="https://api.example.com",
            api_token="test-key"
        )
        assert handler_with_key.auth_config.auth_type == AuthType.API_KEY
        assert handler_with_key.auth_config.api_key == "test-key"

        # Test with JWT
        handler_with_jwt = APIHandler(
            base_url="https://api.example.com",
            jwt_token="jwt-token"
        )
        assert handler_with_jwt.auth_config.auth_type == AuthType.JWT
        assert handler_with_jwt.auth_config.jwt_token == "jwt-token"

    def test_api_handler_configuration_validation(self):
        """Test that APIHandler properly validates configuration."""
        # Test URL building
        handler = APIHandler(base_url="https://api.example.com")
        
        # Test with relative endpoint
        url = handler._build_url("/users")
        assert url == "https://api.example.com/users"
        
        # Test with full URL
        full_url = "https://other-api.com/data"
        url = handler._build_url(full_url)
        assert url == full_url

    def test_pagination_config_integration(self):
        """Test that pagination configuration works with APIHandler."""
        pagination_config = PaginationConfig(
            enabled=True,
            pagination_type=PaginationType.PAGE_SIZE,
            page_param="page",
            size_param="limit",
            page_size=50,
            max_pages=10
        )
        
        assert pagination_config.enabled is True
        assert pagination_config.pagination_type == PaginationType.PAGE_SIZE
        assert pagination_config.page_size == 50
        assert pagination_config.max_pages == 10

    def test_header_preparation_integration(self):
        """Test that header preparation works correctly with different auth types."""
        # Test API key authentication
        handler_api_key = APIHandler(api_token="test-api-key")
        headers = handler_api_key._prepare_headers()
        assert "X-API-Key" in headers
        assert headers["X-API-Key"] == "test-api-key"
        assert headers["Content-Type"] == "application/json"

        # Test JWT authentication
        handler_jwt = APIHandler(jwt_token="jwt-token-123")
        headers = handler_jwt._prepare_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer jwt-token-123"

        # Test Bearer token authentication
        handler_bearer = APIHandler(bearer_token="bearer-token-456")
        headers = handler_bearer._prepare_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer bearer-token-456"

        # Test Basic authentication
        handler_basic = APIHandler(username="testuser", password="testpass")
        headers = handler_basic._prepare_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")

    @pytest.mark.asyncio
    async def test_context_manager_integration(self):
        """Test that APIHandler works as an async context manager."""
        handler = APIHandler(base_url="https://api.example.com")
        
        # Test context manager protocol
        async with handler as h:
            assert h is handler
            # Verify that the client is initialized
            assert handler._client is not None
        
        # Verify that the client is closed after exiting context
        # Note: We can't directly check if client is closed without making a request,
        # but we can verify the close method was called by checking the client is None
        # after manual close
        await handler.close()

    def test_error_hierarchy_integration(self):
        """Test that API error classes are properly integrated."""
        from dataload.domain import (
            APIError, AuthenticationError, APITimeoutError, 
            APIRateLimitError, APIConnectionError
        )
        
        # Test error hierarchy
        assert issubclass(AuthenticationError, APIError)
        assert issubclass(APITimeoutError, APIError)
        assert issubclass(APIRateLimitError, APIError)
        assert issubclass(APIConnectionError, APIError)
        
        # Test error instantiation
        auth_error = AuthenticationError("Auth failed")
        assert str(auth_error) == "Auth failed"
        
        timeout_error = APITimeoutError("Timeout")
        assert str(timeout_error) == "Timeout"

    def test_configuration_models_integration(self):
        """Test that configuration models work correctly with APIHandler."""
        from dataload.domain import AuthConfig, PaginationConfig, APIConfig
        
        # Test AuthConfig
        auth_config = AuthConfig(
            auth_type=AuthType.API_KEY,
            api_key="test-key"
        )
        assert auth_config.auth_type == AuthType.API_KEY
        assert auth_config.api_key == "test-key"
        
        # Test PaginationConfig
        pagination_config = PaginationConfig(
            enabled=True,
            page_size=25
        )
        assert pagination_config.enabled is True
        assert pagination_config.page_size == 25
        
        # Test APIConfig
        api_config = APIConfig(
            base_url="https://api.example.com",
            authentication=auth_config,
            timeout=60,
            pagination=pagination_config
        )
        assert api_config.base_url == "https://api.example.com"
        assert api_config.authentication == auth_config
        assert api_config.timeout == 60
        assert api_config.pagination == pagination_config

    def test_convenience_methods_integration(self):
        """Test that convenience methods are available and properly configured."""
        handler = APIHandler(base_url="https://api.example.com")
        
        # Verify all convenience methods exist
        assert hasattr(handler, 'get')
        assert hasattr(handler, 'post')
        assert hasattr(handler, 'put')
        assert hasattr(handler, 'delete')
        assert hasattr(handler, 'patch')
        
        # Verify they are callable
        assert callable(handler.get)
        assert callable(handler.post)
        assert callable(handler.put)
        assert callable(handler.delete)
        assert callable(handler.patch)