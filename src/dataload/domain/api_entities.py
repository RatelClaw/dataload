"""
Domain entities for API/JSON operations.

This module contains all the domain entities, configuration models, and exceptions
specifically related to API data loading and JSON processing operations.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from .entities import DataMoveError, ValidationError


# API-specific exceptions
class APIError(DataMoveError):
    """Base exception for API-related errors."""
    pass


class AuthenticationError(APIError):
    """Authentication failures."""
    pass


class APITimeoutError(APIError):
    """API request timeout errors."""
    pass


class APIRateLimitError(APIError):
    """API rate limit exceeded."""
    pass


class APIConnectionError(APIError):
    """API connection failures."""
    pass


class JSONParsingError(DataMoveError):
    """JSON parsing and validation errors."""
    pass


class ColumnMappingError(ValidationError):
    """Column mapping configuration errors."""
    pass


class RequestTransformationError(DataMoveError):
    """Request body transformation errors."""
    pass


# Enums for configuration options
class AuthType(Enum):
    """Supported authentication types."""
    API_KEY = "api_key"
    JWT = "jwt"
    BEARER = "bearer"
    BASIC = "basic"
    NONE = "none"


class PaginationType(Enum):
    """Supported pagination patterns."""
    PAGE_SIZE = "page_size"
    OFFSET_LIMIT = "offset_limit"
    CURSOR = "cursor"
    NONE = "none"


class ArrayHandlingStrategy(Enum):
    """Strategies for handling arrays in JSON."""
    EXPAND = "expand"  # Create separate rows for each array element
    JOIN = "join"      # Join array elements into a single string
    IGNORE = "ignore"  # Skip array fields


class NullHandlingStrategy(Enum):
    """Strategies for handling null values."""
    KEEP = "keep"      # Keep null values as NaN
    DROP = "drop"      # Drop rows with null values
    FILL = "fill"      # Fill with default values


class DuplicateKeyStrategy(Enum):
    """Strategies for handling duplicate keys after flattening."""
    SUFFIX = "suffix"      # Add numeric suffix
    ERROR = "error"        # Raise an error
    OVERWRITE = "overwrite"  # Overwrite with last value


class ValidationMode(Enum):
    """Validation modes for column mapping."""
    STRICT = "strict"    # Fail on any validation error
    LENIENT = "lenient"  # Warn on validation errors but continue
    IGNORE = "ignore"    # Ignore validation errors


# API Configuration Models
@dataclass
class AuthConfig:
    """Authentication configuration for API requests."""
    
    auth_type: AuthType
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    bearer_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Validate authentication configuration."""
        if self.auth_type == AuthType.API_KEY and not self.api_key:
            raise ValidationError("API key is required when auth_type is 'api_key'")
        elif self.auth_type == AuthType.JWT and not self.jwt_token:
            raise ValidationError("JWT token is required when auth_type is 'jwt'")
        elif self.auth_type == AuthType.BEARER and not self.bearer_token:
            raise ValidationError("Bearer token is required when auth_type is 'bearer'")
        elif self.auth_type == AuthType.BASIC and (not self.username or not self.password):
            raise ValidationError("Username and password are required when auth_type is 'basic'")


@dataclass
class PaginationConfig:
    """Pagination configuration for API requests."""
    
    enabled: bool = False
    pagination_type: PaginationType = PaginationType.PAGE_SIZE
    page_param: str = "page"
    size_param: str = "size"
    offset_param: str = "offset"
    limit_param: str = "limit"
    cursor_param: str = "cursor"
    max_pages: Optional[int] = None
    page_size: int = 100
    total_count_header: Optional[str] = None
    next_page_header: Optional[str] = None
    
    def __post_init__(self):
        """Validate pagination configuration."""
        if self.enabled and self.page_size <= 0:
            raise ValidationError("Page size must be greater than 0")
        if self.max_pages is not None and self.max_pages <= 0:
            raise ValidationError("Max pages must be greater than 0")


@dataclass
class APIConfig:
    """Configuration for API requests."""
    
    base_url: str
    authentication: AuthConfig
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    pagination: Optional[PaginationConfig] = None
    default_headers: Optional[Dict[str, str]] = None
    verify_ssl: bool = True
    
    def __post_init__(self):
        """Validate API configuration."""
        if not self.base_url:
            raise ValidationError("Base URL is required")
        if self.timeout <= 0:
            raise ValidationError("Timeout must be greater than 0")
        if self.retry_attempts < 0:
            raise ValidationError("Retry attempts must be non-negative")
        if self.retry_delay < 0:
            raise ValidationError("Retry delay must be non-negative")


# JSON Processing Models
@dataclass
class JSONProcessingConfig:
    """Configuration for JSON processing operations."""
    
    flatten_nested: bool = True
    separator: str = "_"
    max_depth: Optional[int] = None
    handle_arrays: ArrayHandlingStrategy = ArrayHandlingStrategy.EXPAND
    null_handling: NullHandlingStrategy = NullHandlingStrategy.KEEP
    duplicate_key_strategy: DuplicateKeyStrategy = DuplicateKeyStrategy.SUFFIX
    preserve_original_keys: bool = False
    normalize_column_names: bool = True
    
    def __post_init__(self):
        """Validate JSON processing configuration."""
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValidationError("Max depth must be greater than 0")
        if not self.separator:
            raise ValidationError("Separator cannot be empty")


@dataclass
class FlatteningResult:
    """Result of JSON flattening operation."""
    
    dataframe: pd.DataFrame
    original_structure: Dict[str, Any]
    flattened_columns: List[str]
    conflicts_resolved: List[str]
    warnings: List[str]
    processing_stats: Dict[str, Any]
    
    @property
    def success(self) -> bool:
        """Check if flattening was successful."""
        return not self.dataframe.empty
    
    @property
    def column_count(self) -> int:
        """Get the number of columns after flattening."""
        return len(self.flattened_columns)
    
    @property
    def row_count(self) -> int:
        """Get the number of rows after flattening."""
        return len(self.dataframe)


# Column Mapping Models
@dataclass
class ColumnMappingConfig:
    """Configuration for column mapping operations."""
    
    column_name_mapping: Dict[str, str]
    update_request_body_mapping: Dict[str, str]
    validation_mode: ValidationMode = ValidationMode.STRICT
    case_sensitive: bool = True
    allow_missing_columns: bool = False
    preserve_unmapped_columns: bool = True
    
    def __post_init__(self):
        """Validate column mapping configuration."""
        if not isinstance(self.column_name_mapping, dict):
            raise ValidationError("Column name mapping must be a dictionary")
        if not isinstance(self.update_request_body_mapping, dict):
            raise ValidationError("Update request body mapping must be a dictionary")


@dataclass
class MappingResult:
    """Result of column mapping operation."""
    
    mapped_dataframe: pd.DataFrame
    applied_mappings: Dict[str, str]
    unmapped_columns: List[str]
    conflicts: List[str]
    warnings: List[str]
    transformation_stats: Dict[str, Any]
    
    @property
    def success(self) -> bool:
        """Check if mapping was successful."""
        return len(self.conflicts) == 0
    
    @property
    def mapping_coverage(self) -> float:
        """Calculate the percentage of columns that were mapped."""
        total_columns = len(self.applied_mappings) + len(self.unmapped_columns)
        if total_columns == 0:
            return 0.0
        return len(self.applied_mappings) / total_columns * 100


# Request Transformation Models
@dataclass
class TransformationRule:
    """A single transformation rule for request body data."""
    
    target_field: str
    source_expression: str
    transformation_type: str  # 'copy', 'concat', 'compute', 'constant'
    default_value: Optional[Any] = None
    required: bool = True
    
    def __post_init__(self):
        """Validate transformation rule."""
        if not self.target_field:
            raise ValidationError("Target field cannot be empty")
        if not self.source_expression and self.transformation_type != 'constant':
            raise ValidationError("Source expression is required for non-constant transformations")


@dataclass
class RequestTransformationConfig:
    """Configuration for request body transformations."""
    
    transformation_rules: List[TransformationRule]
    execution_order: List[str]  # Order of field names to execute transformations
    fail_on_error: bool = True
    preserve_original_data: bool = True
    
    def __post_init__(self):
        """Validate transformation configuration."""
        rule_names = {rule.target_field for rule in self.transformation_rules}
        
        # Validate execution order
        for field_name in self.execution_order:
            if field_name not in rule_names:
                raise ValidationError(f"Field '{field_name}' in execution order not found in transformation rules")
        
        # Check for duplicate target fields
        target_fields = [rule.target_field for rule in self.transformation_rules]
        if len(target_fields) != len(set(target_fields)):
            raise ValidationError("Duplicate target fields found in transformation rules")


@dataclass
class TransformationResult:
    """Result of request body transformation operation."""
    
    transformed_dataframe: pd.DataFrame
    applied_transformations: List[str]
    failed_transformations: List[str]
    warnings: List[str]
    execution_stats: Dict[str, Any]
    
    @property
    def success(self) -> bool:
        """Check if transformation was successful."""
        return len(self.failed_transformations) == 0
    
    @property
    def transformation_count(self) -> int:
        """Get the number of successful transformations."""
        return len(self.applied_transformations)


# API Response Models
@dataclass
class APIResponse:
    """Represents an API response with metadata."""
    
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    status_code: int
    headers: Dict[str, str]
    response_time: float
    url: str
    method: str
    pagination_info: Optional[Dict[str, Any]] = None
    
    @property
    def success(self) -> bool:
        """Check if the API response was successful."""
        return 200 <= self.status_code < 300
    
    @property
    def is_paginated(self) -> bool:
        """Check if the response contains pagination information."""
        return self.pagination_info is not None


@dataclass
class PaginationInfo:
    """Information about pagination state."""
    
    current_page: int
    total_pages: Optional[int]
    page_size: int
    total_count: Optional[int]
    has_next_page: bool
    next_page_url: Optional[str] = None
    next_cursor: Optional[str] = None
    
    @property
    def progress_percentage(self) -> Optional[float]:
        """Calculate pagination progress as percentage."""
        if self.total_pages is None:
            return None
        return (self.current_page / self.total_pages) * 100


# Comprehensive operation result models
@dataclass
class APIJSONLoadResult:
    """Comprehensive result of an API/JSON loading operation."""
    
    success: bool
    dataframe: Optional[pd.DataFrame]
    rows_loaded: int
    columns_created: int
    execution_time: float
    api_responses: List[APIResponse]
    flattening_result: Optional[FlatteningResult]
    mapping_result: Optional[MappingResult]
    transformation_result: Optional[TransformationResult]
    errors: List[Exception]
    warnings: List[str]
    operation_metadata: Dict[str, Any]
    
    @property
    def total_api_calls(self) -> int:
        """Get the total number of API calls made."""
        return len(self.api_responses)
    
    @property
    def average_response_time(self) -> float:
        """Calculate average API response time."""
        if not self.api_responses:
            return 0.0
        return sum(resp.response_time for resp in self.api_responses) / len(self.api_responses)
    
    @property
    def has_errors(self) -> bool:
        """Check if there were any errors during the operation."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there were any warnings during the operation."""
        return len(self.warnings) > 0