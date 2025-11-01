from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum


class TableSchema(BaseModel):
    """Represents the schema of a data table."""

    columns: Dict[str, str]
    nullables: Dict[str, bool]


# Existing exceptions
class DBOperationError(Exception):
    """Raised for errors during database/vector store operations."""

    pass


class DataValidationError(Exception):
    """Raised when input data (e.g., CSV) fails validation."""

    pass


class EmbeddingError(Exception):
    """Raised when an embedding provider fails to generate vectors."""

    pass


# DataMove-specific exceptions
class DataMoveError(Exception):
    """Base exception for DataMove operations."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ValidationError(DataMoveError):
    """Schema or data validation failures."""
    pass


class SchemaConflictError(ValidationError):
    """Schema compatibility issues."""
    pass


class CaseSensitivityError(ValidationError):
    """Case-sensitive column name conflicts."""
    pass


class DataTypeError(ValidationError):
    """Data type conversion or compatibility issues."""
    pass


class DatabaseOperationError(DataMoveError):
    """Database connection or operation failures."""
    pass


# DataMove data models
@dataclass
class ColumnInfo:
    """Information about a database column."""
    
    name: str
    data_type: str
    nullable: bool
    default_value: Optional[Any] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None


@dataclass
class Constraint:
    """Database constraint information."""
    
    name: str
    type: str  # 'PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK'
    columns: List[str]
    referenced_table: Optional[str] = None
    referenced_columns: Optional[List[str]] = None


@dataclass
class IndexInfo:
    """Database index information."""
    
    name: str
    columns: List[str]
    index_type: str  # 'btree', 'hash', 'ivfflat', 'hnsw'
    unique: bool = False


@dataclass
class TableInfo:
    """Comprehensive table information."""
    
    name: str
    columns: Dict[str, ColumnInfo]
    primary_keys: List[str]
    constraints: List[Constraint]
    indexes: List[IndexInfo]


@dataclass
class CaseConflict:
    """Case-sensitivity conflict between column names."""
    
    db_column: str
    csv_column: str
    conflict_type: str  # 'case_mismatch', 'duplicate_insensitive'


@dataclass
class TypeMismatch:
    """Data type mismatch between database and CSV."""
    
    column_name: str
    db_type: str
    csv_type: str
    compatible: bool
    conversion_required: bool
    sample_values: Optional[List[Any]] = None


@dataclass
class ConstraintViolation:
    """Constraint violation information."""
    
    constraint_name: str
    constraint_type: str
    column_name: str
    violation_type: str
    affected_rows: Optional[int] = None
    sample_violations: Optional[List[Any]] = None


@dataclass
class ConversionSuggestion:
    """Suggestion for resolving type conversion issues."""
    
    column_name: str
    from_type: str
    to_type: str
    conversion_function: str
    risk_level: str  # 'low', 'medium', 'high'
    description: str


@dataclass
class SchemaAnalysis:
    """Analysis of schema compatibility between database and CSV."""
    
    table_exists: bool
    columns_added: List[str]
    columns_removed: List[str]
    columns_modified: List[TypeMismatch]
    case_conflicts: List[CaseConflict]
    constraint_violations: List[ConstraintViolation]
    compatible: bool
    requires_schema_update: bool


@dataclass
class ValidationReport:
    """Comprehensive validation report for data movement operations."""
    
    schema_analysis: SchemaAnalysis
    case_conflicts: List[CaseConflict]
    type_mismatches: List[TypeMismatch]
    constraint_violations: List[ConstraintViolation]
    recommendations: List[str]
    warnings: List[str]
    errors: List[str]
    validation_passed: bool


@dataclass
class DataMoveResult:
    """Result of a data movement operation."""
    
    success: bool
    rows_processed: int
    execution_time: float
    validation_report: ValidationReport
    errors: List[DataMoveError]
    warnings: List[str]
    table_created: bool = False
    schema_updated: bool = False
    operation_type: Optional[str] = None  # 'new_table', 'existing_schema', 'new_schema'
