"""
Domain layer for the dataload package.

This module contains all domain entities, value objects, and domain-specific
exceptions used throughout the application.
"""

# Import existing entities
from .entities import (
    # Base exceptions
    DBOperationError,
    DataValidationError,
    EmbeddingError,
    DataMoveError,
    ValidationError,
    SchemaConflictError,
    CaseSensitivityError,
    DataTypeError,
    DatabaseOperationError,
    
    # Data models
    TableSchema,
    ColumnInfo,
    Constraint,
    IndexInfo,
    TableInfo,
    CaseConflict,
    TypeMismatch,
    ConstraintViolation,
    ConversionSuggestion,
    SchemaAnalysis,
    ValidationReport,
    DataMoveResult,
)

# Import API handler
from .api_handler import APIHandler

# Import API/JSON entities
from .api_entities import (
    # API-specific exceptions
    APIError,
    AuthenticationError,
    APITimeoutError,
    APIRateLimitError,
    APIConnectionError,
    JSONParsingError,
    ColumnMappingError,
    RequestTransformationError,
    
    # Enums
    AuthType,
    PaginationType,
    ArrayHandlingStrategy,
    NullHandlingStrategy,
    DuplicateKeyStrategy,
    ValidationMode,
    
    # Configuration models
    AuthConfig,
    PaginationConfig,
    APIConfig,
    JSONProcessingConfig,
    ColumnMappingConfig,
    RequestTransformationConfig,
    
    # Result models
    FlatteningResult,
    MappingResult,
    TransformationResult,
    APIResponse,
    PaginationInfo,
    APIJSONLoadResult,
    
    # Transformation models
    TransformationRule,
)

__all__ = [
    # API handler
    "APIHandler",
    
    # Base exceptions
    "DBOperationError",
    "DataValidationError", 
    "EmbeddingError",
    "DataMoveError",
    "ValidationError",
    "SchemaConflictError",
    "CaseSensitivityError",
    "DataTypeError",
    "DatabaseOperationError",
    
    # API-specific exceptions
    "APIError",
    "AuthenticationError",
    "APITimeoutError",
    "APIRateLimitError",
    "APIConnectionError",
    "JSONParsingError",
    "ColumnMappingError",
    "RequestTransformationError",
    
    # Data models
    "TableSchema",
    "ColumnInfo",
    "Constraint",
    "IndexInfo",
    "TableInfo",
    "CaseConflict",
    "TypeMismatch",
    "ConstraintViolation",
    "ConversionSuggestion",
    "SchemaAnalysis",
    "ValidationReport",
    "DataMoveResult",
    
    # Enums
    "AuthType",
    "PaginationType",
    "ArrayHandlingStrategy",
    "NullHandlingStrategy",
    "DuplicateKeyStrategy",
    "ValidationMode",
    
    # Configuration models
    "AuthConfig",
    "PaginationConfig",
    "APIConfig",
    "JSONProcessingConfig",
    "ColumnMappingConfig",
    "RequestTransformationConfig",
    
    # Result models
    "FlatteningResult",
    "MappingResult",
    "TransformationResult",
    "APIResponse",
    "PaginationInfo",
    "APIJSONLoadResult",
    
    # Transformation models
    "TransformationRule",
]