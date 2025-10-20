"""
APIHandler for external API communication.

This module provides the APIHandler class that manages HTTP requests to external APIs
with authentication, retry logic, timeout handling, and pagination support.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import urljoin, urlparse
import base64
import logging

try:
    import httpx
except ImportError:
    raise ImportError(
        "httpx is required for API functionality. "
        "Please install it with: pip install httpx"
    )

from .api_entities import (
    APIConfig, AuthConfig, PaginationConfig, APIResponse, PaginationInfo,
    AuthType, PaginationType, APIError, AuthenticationError, APITimeoutError,
    APIRateLimitError, APIConnectionError
)

logger = logging.getLogger(__name__)


class APIHandler:
    """
    Handles API requests with authentication, retry mechanisms, and pagination support.
    
    This class provides a comprehensive solution for making HTTP requests to external APIs
    with proper error handling, authentication, and retry logic with exponential backoff.
    """
    
    def __init__(self, 
                 base_url: Optional[str] = None,
                 api_token: Optional[str] = None,
                 jwt_token: Optional[str] = None,
                 bearer_token: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 timeout: int = 30,
                 retry_attempts: int = 3,
                 retry_delay: float = 1.0,
                 retry_backoff_factor: float = 2.0,
                 verify_ssl: bool = True,
                 default_headers: Optional[Dict[str, str]] = None):
        """
        Initialize the APIHandler with configuration parameters.
        
        Args:
            base_url: Base URL for API requests
            api_token: API key for authentication
            jwt_token: JWT token for authentication
            bearer_token: Bearer token for authentication
            username: Username for basic authentication
            password: Password for basic authentication
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Initial delay between retries in seconds
            retry_backoff_factor: Multiplier for exponential backoff
            verify_ssl: Whether to verify SSL certificates
            default_headers: Default headers to include in all requests
        """
        self.base_url = base_url
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.verify_ssl = verify_ssl
        self.default_headers = default_headers or {}
        
        # Determine authentication type and create auth config
        self.auth_config = self._create_auth_config(
            api_token, jwt_token, bearer_token, username, password
        )
        
        # Create HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"APIHandler initialized with base_url={base_url}, auth_type={self.auth_config.auth_type}")
    
    def _create_auth_config(self, 
                           api_token: Optional[str],
                           jwt_token: Optional[str], 
                           bearer_token: Optional[str],
                           username: Optional[str],
                           password: Optional[str]) -> AuthConfig:
        """Create authentication configuration based on provided credentials."""
        if api_token:
            return AuthConfig(auth_type=AuthType.API_KEY, api_key=api_token)
        elif jwt_token:
            return AuthConfig(auth_type=AuthType.JWT, jwt_token=jwt_token)
        elif bearer_token:
            return AuthConfig(auth_type=AuthType.BEARER, bearer_token=bearer_token)
        elif username and password:
            return AuthConfig(auth_type=AuthType.BASIC, username=username, password=password)
        else:
            return AuthConfig(auth_type=AuthType.NONE)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                verify=self.verify_ssl,
                headers=self.default_headers
            )
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _prepare_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Prepare headers with authentication and additional headers.
        
        Args:
            additional_headers: Additional headers to include
            
        Returns:
            Dict containing all headers for the request
        """
        headers = self.default_headers.copy()
        
        # Add authentication headers
        if self.auth_config.auth_type == AuthType.API_KEY:
            headers['X-API-Key'] = self.auth_config.api_key
        elif self.auth_config.auth_type == AuthType.JWT:
            headers['Authorization'] = f'Bearer {self.auth_config.jwt_token}'
        elif self.auth_config.auth_type == AuthType.BEARER:
            headers['Authorization'] = f'Bearer {self.auth_config.bearer_token}'
        elif self.auth_config.auth_type == AuthType.BASIC:
            credentials = base64.b64encode(
                f'{self.auth_config.username}:{self.auth_config.password}'.encode()
            ).decode()
            headers['Authorization'] = f'Basic {credentials}'
        
        # Add custom auth headers if provided
        if self.auth_config.headers:
            headers.update(self.auth_config.headers)
        
        # Add additional headers
        if additional_headers:
            headers.update(additional_headers)
        
        # Ensure content-type for JSON requests
        if 'Content-Type' not in headers:
            headers['Content-Type'] = 'application/json'
        
        return headers
    
    def _build_url(self, endpoint: str) -> str:
        """
        Build full URL from base URL and endpoint.
        
        Args:
            endpoint: API endpoint path or full URL
            
        Returns:
            Complete URL for the request
        """
        if endpoint.startswith(('http://', 'https://')):
            return endpoint
        
        if self.base_url:
            return urljoin(self.base_url.rstrip('/') + '/', endpoint.lstrip('/'))
        
        raise APIError(f"No base URL configured and endpoint is not a full URL: {endpoint}")
    
    async def _make_request(self,
                           method: str,
                           url: str,
                           headers: Optional[Dict[str, str]] = None,
                           params: Optional[Dict[str, Any]] = None,
                           json_data: Optional[Dict[str, Any]] = None,
                           data: Optional[Union[str, bytes]] = None) -> APIResponse:
        """
        Make a single HTTP request with error handling.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            url: Complete URL for the request
            headers: Request headers
            params: Query parameters
            json_data: JSON data for request body
            data: Raw data for request body
            
        Returns:
            APIResponse object containing response data and metadata
            
        Raises:
            APIError: For various API-related errors
        """
        await self._ensure_client()
        
        start_time = time.time()
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            response = await self._client.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                content=data
            )
            
            response_time = time.time() - start_time
            
            # Parse response data
            try:
                if response.headers.get('content-type', '').startswith('application/json'):
                    response_data = response.json()
                else:
                    response_data = response.text
            except Exception as e:
                logger.warning(f"Failed to parse response as JSON: {e}")
                response_data = response.text
            
            api_response = APIResponse(
                data=response_data,
                status_code=response.status_code,
                headers=dict(response.headers),
                response_time=response_time,
                url=url,
                method=method
            )
            
            # Handle HTTP error status codes
            if not api_response.success:
                if response.status_code == 401:
                    raise AuthenticationError(f"Authentication failed: {response.status_code} {response.reason_phrase}")
                elif response.status_code == 429:
                    raise APIRateLimitError(f"Rate limit exceeded: {response.status_code} {response.reason_phrase}")
                elif response.status_code >= 500:
                    raise APIError(f"Server error: {response.status_code} {response.reason_phrase}")
                else:
                    raise APIError(f"HTTP error: {response.status_code} {response.reason_phrase}")
            
            logger.debug(f"Request completed successfully in {response_time:.2f}s")
            return api_response
            
        except (AuthenticationError, APIRateLimitError, APIError):
            # Re-raise API-specific errors without wrapping
            raise
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timeout after {self.timeout}s: {e}")
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Connection error: {e}")
        except httpx.HTTPError as e:
            raise APIError(f"HTTP error: {e}")
        except Exception as e:
            raise APIError(f"Unexpected error during request: {e}")
    
    async def _retry_request(self,
                            method: str,
                            url: str,
                            headers: Optional[Dict[str, str]] = None,
                            params: Optional[Dict[str, Any]] = None,
                            json_data: Optional[Dict[str, Any]] = None,
                            data: Optional[Union[str, bytes]] = None) -> APIResponse:
        """
        Make HTTP request with retry logic and exponential backoff.
        
        Args:
            method: HTTP method
            url: Complete URL
            headers: Request headers
            params: Query parameters
            json_data: JSON request body
            data: Raw request body
            
        Returns:
            APIResponse object
            
        Raises:
            APIError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.retry_attempts + 1):
            try:
                return await self._make_request(method, url, headers, params, json_data, data)
                
            except (APITimeoutError, APIConnectionError, APIError) as e:
                last_exception = e
                
                # Don't retry authentication errors or rate limit errors
                if isinstance(e, (AuthenticationError, APIRateLimitError)):
                    raise e
                
                if attempt < self.retry_attempts:
                    delay = self.retry_delay * (self.retry_backoff_factor ** attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_attempts + 1}): {e}. "
                                 f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed for {method} {url}")
                    raise e
        
        # This should never be reached, but just in case
        raise last_exception or APIError("Request failed after all retry attempts")
    
    async def fetch_data(self,
                        endpoint: str,
                        method: str = "GET",
                        headers: Optional[Dict[str, str]] = None,
                        params: Optional[Dict[str, Any]] = None,
                        json_data: Optional[Dict[str, Any]] = None,
                        data: Optional[Union[str, bytes]] = None,
                        pagination_config: Optional[PaginationConfig] = None) -> Union[APIResponse, List[APIResponse]]:
        """
        Fetch data from an API endpoint with optional pagination.
        
        Args:
            endpoint: API endpoint path or full URL
            method: HTTP method (default: GET)
            headers: Additional headers for the request
            params: Query parameters
            json_data: JSON data for request body
            data: Raw data for request body
            pagination_config: Configuration for pagination handling
            
        Returns:
            APIResponse object or list of APIResponse objects if paginated
            
        Raises:
            APIError: For various API-related errors
        """
        url = self._build_url(endpoint)
        request_headers = self._prepare_headers(headers)
        
        logger.info(f"Fetching data from {url} with method {method}")
        
        # Handle non-paginated requests
        if not pagination_config or not pagination_config.enabled:
            return await self._retry_request(method, url, request_headers, params, json_data, data)
        
        # Handle paginated requests
        return await self._fetch_paginated_data(
            method, url, request_headers, params, json_data, data, pagination_config
        )
    
    async def _fetch_paginated_data(self,
                                   method: str,
                                   url: str,
                                   headers: Dict[str, str],
                                   params: Optional[Dict[str, Any]],
                                   json_data: Optional[Dict[str, Any]],
                                   data: Optional[Union[str, bytes]],
                                   pagination_config: PaginationConfig) -> List[APIResponse]:
        """
        Fetch all pages of data from a paginated API endpoint.
        
        Args:
            method: HTTP method
            url: Complete URL
            headers: Request headers
            params: Query parameters
            json_data: JSON request body
            data: Raw request body
            pagination_config: Pagination configuration
            
        Returns:
            List of APIResponse objects, one for each page
        """
        responses = []
        current_page = 1
        current_cursor = None
        params = params or {}
        
        logger.info(f"Starting paginated data fetch with config: {pagination_config}")
        
        while True:
            # Prepare pagination parameters
            page_params = params.copy()
            
            if pagination_config.pagination_type == PaginationType.PAGE_SIZE:
                page_params[pagination_config.page_param] = current_page
                page_params[pagination_config.size_param] = pagination_config.page_size
            elif pagination_config.pagination_type == PaginationType.OFFSET_LIMIT:
                offset = (current_page - 1) * pagination_config.page_size
                page_params[pagination_config.offset_param] = offset
                page_params[pagination_config.limit_param] = pagination_config.page_size
            elif pagination_config.pagination_type == PaginationType.CURSOR:
                if current_cursor:
                    page_params[pagination_config.cursor_param] = current_cursor
                # For cursor-based pagination, we might still want to include page size
                if pagination_config.size_param:
                    page_params[pagination_config.size_param] = pagination_config.page_size
            
            # Make request for current page
            try:
                response = await self._retry_request(
                    method, url, headers, page_params, json_data, data
                )
                
                # Add pagination info to response
                pagination_info = self._extract_pagination_info(
                    response, current_page, pagination_config
                )
                response.pagination_info = pagination_info.__dict__ if pagination_info else None
                
                responses.append(response)
                
                logger.debug(f"Fetched page {current_page}, got {len(response.data) if isinstance(response.data, list) else 1} items")
                
                # For cursor-based pagination, extract the next cursor
                if pagination_config.pagination_type == PaginationType.CURSOR:
                    if pagination_info and pagination_info.next_cursor:
                        current_cursor = pagination_info.next_cursor
                    elif isinstance(response.data, dict):
                        # Try to extract cursor from common response patterns
                        current_cursor = (
                            response.data.get('next_cursor') or
                            response.data.get('cursor') or
                            response.data.get('next') or
                            response.data.get('nextCursor')
                        )
                    else:
                        current_cursor = None
                
                # Check if we should continue pagination
                if not self._should_continue_pagination(response, pagination_info, pagination_config):
                    break
                
                current_page += 1
                
                # Check max pages limit
                if pagination_config.max_pages and current_page > pagination_config.max_pages:
                    logger.warning(f"Reached max pages limit: {pagination_config.max_pages}")
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching page {current_page}: {e}")
                raise e
        
        logger.info(f"Completed paginated fetch: {len(responses)} pages, {current_page} total pages processed")
        return responses
    
    def _extract_pagination_info(self,
                                response: APIResponse,
                                current_page: int,
                                pagination_config: PaginationConfig) -> Optional[PaginationInfo]:
        """
        Extract pagination information from API response.
        
        Args:
            response: API response object
            current_page: Current page number
            pagination_config: Pagination configuration
            
        Returns:
            PaginationInfo object or None if pagination info cannot be extracted
        """
        try:
            # Try to extract pagination info from headers
            total_count = None
            total_pages = None
            has_next_page = False
            next_page_url = None
            next_cursor = None
            
            if pagination_config.total_count_header:
                total_count_str = response.headers.get(pagination_config.total_count_header)
                if total_count_str:
                    total_count = int(total_count_str)
                    total_pages = (total_count + pagination_config.page_size - 1) // pagination_config.page_size
            
            if pagination_config.next_page_header:
                next_page_url = response.headers.get(pagination_config.next_page_header)
                has_next_page = bool(next_page_url)
            
            # Try to extract pagination info from response body
            if isinstance(response.data, dict):
                # Common pagination patterns in response body
                if 'total' in response.data:
                    total_count = response.data['total']
                    total_pages = (total_count + pagination_config.page_size - 1) // pagination_config.page_size
                
                if 'has_next' in response.data:
                    has_next_page = response.data['has_next']
                
                if 'next' in response.data:
                    next_page_url = response.data['next']
                    if not has_next_page:  # Only set has_next_page if not already set
                        has_next_page = bool(response.data['next'])
                
                # Extract cursor information for cursor-based pagination
                if pagination_config.pagination_type == PaginationType.CURSOR:
                    next_cursor = (
                        response.data.get('next_cursor') or
                        response.data.get('cursor') or
                        response.data.get('nextCursor')
                    )
                    # For cursor-based pagination, has_next is often indicated by presence of cursor
                    if 'has_more' in response.data:
                        has_next_page = response.data['has_more']
                    elif next_cursor:
                        has_next_page = True
                
                # Check if current page has fewer items than page size (indicates last page)
                if 'data' in response.data and isinstance(response.data['data'], list):
                    if len(response.data['data']) < pagination_config.page_size:
                        has_next_page = False
            
            # If response is a list, check if it's smaller than page size
            elif isinstance(response.data, list):
                if len(response.data) < pagination_config.page_size:
                    has_next_page = False
                else:
                    has_next_page = True  # Assume there might be more pages
            
            return PaginationInfo(
                current_page=current_page,
                total_pages=total_pages,
                page_size=pagination_config.page_size,
                total_count=total_count,
                has_next_page=has_next_page,
                next_page_url=next_page_url,
                next_cursor=next_cursor
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract pagination info: {e}")
            return None
    
    def _should_continue_pagination(self,
                                  response: APIResponse,
                                  pagination_info: Optional[PaginationInfo],
                                  pagination_config: PaginationConfig) -> bool:
        """
        Determine if pagination should continue based on response and pagination info.
        
        Args:
            response: Current API response
            pagination_info: Extracted pagination information
            pagination_config: Pagination configuration
            
        Returns:
            True if pagination should continue, False otherwise
        """
        # If we have explicit pagination info, use it
        if pagination_info:
            return pagination_info.has_next_page
        
        # Fallback: check if response data suggests more pages
        if isinstance(response.data, list):
            # If we got fewer items than page size, assume no more pages
            return len(response.data) >= pagination_config.page_size
        elif isinstance(response.data, dict):
            # Check common pagination indicators in response
            if 'data' in response.data and isinstance(response.data['data'], list):
                return len(response.data['data']) >= pagination_config.page_size
        
        # Conservative default: don't continue if we can't determine
        return False
    
    async def get(self, endpoint: str, **kwargs) -> Union[APIResponse, List[APIResponse]]:
        """Make a GET request."""
        return await self.fetch_data(endpoint, method="GET", **kwargs)
    
    async def post(self, endpoint: str, **kwargs) -> Union[APIResponse, List[APIResponse]]:
        """Make a POST request."""
        return await self.fetch_data(endpoint, method="POST", **kwargs)
    
    async def put(self, endpoint: str, **kwargs) -> Union[APIResponse, List[APIResponse]]:
        """Make a PUT request."""
        return await self.fetch_data(endpoint, method="PUT", **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> Union[APIResponse, List[APIResponse]]:
        """Make a DELETE request."""
        return await self.fetch_data(endpoint, method="DELETE", **kwargs)
    
    async def patch(self, endpoint: str, **kwargs) -> Union[APIResponse, List[APIResponse]]:
        """Make a PATCH request."""
        return await self.fetch_data(endpoint, method="PATCH", **kwargs)