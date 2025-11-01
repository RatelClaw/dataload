# DataMove Use Case - API Documentation

This document provides comprehensive API documentation for the DataMove use case, including all classes, methods, parameters, and usage patterns.

## Table of Contents

1. [Overview](#overview)
2. [DataMoveUseCase Class](#datamoveusecase-class)
3. [Data Models](#data-models)
4. [Exception Classes](#exception-classes)
5. [Repository Interface](#repository-interface)
6. [Storage Loaders](#storage-loaders)
7. [Validation Service](#validation-service)
8. [Usage Patterns](#usage-patterns)
9. [Configuration](#configuration)

## Overview

The DataMove use case provides a production-grade solution for moving data from CSV files (local or S3) to PostgreSQL databases. It offers comprehensive validation, error handling, and performance optimization without embedding generation overhead.

### Key Features

- **Automatic Storage Detection**: Automatically selects S3 or local file loader based on path
- **Flexible Validation**: Supports both strict (`existing_schema`) and flexible (`new_schema`) validation modes
- **Transaction Safety**: Automatic rollback on failures
- **Performance Optimization**: Configurable batch processing and memory management
- **Comprehensive Error Handling**: Detailed error context and recovery suggestions
- **Dry-Run Support**: Preview operations without making changes

## DataMoveUseCase Class

### Class Definition

```python
class DataMoveUseCase:
    """
    Main orchestrator for data movement operations from CSV to PostgreSQL.
    
    This use case handles the complete workflow of moving data from CSV files
    (local or S3) to PostgreSQL tables with comprehensive validation, schema
    management, and error handling.
    """
```

### Constructor

```python
def __init__(
    self,
    repository: DataMoveRepositoryInterface,
    storage_loader: Optional[StorageLoaderInterface] = None,
    validation_service: Optional[ValidationService] = None,
):
    """
    Initialize the DataMove use case.
    
    Args:
        repository: DataMoveRepositoryInterface implementation for database operations
        storage_loader: Optional StorageLoaderInterface for loading CSV data.
                      If None, will auto-select based on file path (S3 vs local)
        validation_service: Optional ValidationService instance (creates default if None)
    """
```

### Factory Methods

#### create_with_auto_loader

```python
@classmethod
def create_with_auto_loader(
    cls,
    repository: DataMoveRepositoryInterface,
    validation_service: Optional[ValidationService] = None,
) -> "DataMoveUseCase":
    """
    Create DataMoveUseCase with automatic storage loader selection.
    
    This convenience method creates a DataMoveUseCase that will automatically
    select the appropriate storage loader (S3 or Local) based on the CSV path
    provided to the execute() method.
    
    Args:
        repository: DataMoveRepositoryInterface implementation for database operations
        validation_service: Optional ValidationService instance (creates default if None)
        
    Returns:
        DataMoveUseCase: Instance configured for automatic loader selection
        
    Example:
        >>> from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
        >>> from dataload.infrastructure.db.db_connection import DBConnection
        >>> 
        >>> db_connection = DBConnection()
        >>> await db_connection.initialize()
        >>> repository = PostgresDataMoveRepository(db_connection)
        >>> use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
    """
```

#### create_storage_loader

```python
@staticmethod
def create_storage_loader(csv_path: str) -> StorageLoaderInterface:
    """
    Create appropriate storage loader based on the CSV path.
    
    This factory method automatically selects between S3Loader and LocalLoader
    based on the path format:
    - Paths starting with "s3://" use S3Loader
    - All other paths use LocalLoader
    
    Args:
        csv_path: Path to CSV file (local path or S3 URI like s3://bucket/key)
        
    Returns:
        StorageLoaderInterface: Appropriate loader for the path type
        
    Raises:
        ValueError: If path format is not supported
        
    Example:
        >>> # Automatic S3 loader selection
        >>> s3_loader = DataMoveUseCase.create_storage_loader("s3://bucket/file.csv")
        >>> 
        >>> # Automatic local loader selection
        >>> local_loader = DataMoveUseCase.create_storage_loader("/path/to/file.csv")
    """
```

### Main Methods

#### execute

```python
async def execute(
    self,
    csv_path: str,
    table_name: str,
    move_type: Optional[str] = None,
    dry_run: bool = False,
    batch_size: int = 1000,
    primary_key_columns: Optional[List[str]] = None,
) -> DataMoveResult:
    """
    Execute the data movement operation with comprehensive error handling and rollback.
    
    This method orchestrates the complete data movement workflow with automatic
    transaction management, detailed error reporting, and rollback capabilities:
    1. Validate input parameters
    2. Load CSV data with error handling
    3. Detect table existence with connection retry
    4. Route to appropriate validation strategy
    5. Perform data movement with transaction safety (unless dry_run=True)
    
    Args:
        csv_path: Path to CSV file (local path or S3 URI)
                 Examples:
                 - Local: "data/employees.csv", "/absolute/path/file.csv"
                 - S3: "s3://bucket/path/file.csv"
        
        table_name: Name of the target PostgreSQL table
                   Must be a valid PostgreSQL identifier
        
        move_type: Type of move operation, required when target table exists
                  - "existing_schema": Strict validation, exact schema match required
                  - "new_schema": Flexible validation, allows column additions/removals
                  - None: Only valid for new table creation
        
        dry_run: If True, perform validation only without actual data changes
                Useful for previewing operations and testing validation
        
        batch_size: Number of rows to process in each batch for bulk operations
                   Default: 1000. Adjust based on memory and performance needs
                   Range: 100-10000 (recommended)
        
        primary_key_columns: List of columns to use as primary key for new tables
                           Only used when creating new tables
                           Example: ["id"] or ["user_id", "timestamp"]
    
    Returns:
        DataMoveResult: Comprehensive result of the operation containing:
        - success: Boolean indicating operation success
        - rows_processed: Number of rows successfully processed
        - execution_time: Total execution time in seconds
        - validation_report: Detailed validation results
        - errors: List of any errors encountered
        - warnings: List of warnings generated
        - table_created: Boolean indicating if new table was created
        - schema_updated: Boolean indicating if schema was modified
        - operation_type: Type of operation performed
    
    Raises:
        ValidationError: If validation fails or invalid parameters provided
                        Contains detailed context about validation failures
        
        DatabaseOperationError: If database operations fail
                              Contains connection and operation details
        
        DataMoveError: For other data movement related errors
                      Contains comprehensive error context
    
    Examples:
        >>> # Create new table from CSV
        >>> result = await use_case.execute(
        ...     csv_path="employees.csv",
        ...     table_name="employees",
        ...     primary_key_columns=["id"]
        ... )
        
        >>> # Update existing table with strict validation
        >>> result = await use_case.execute(
        ...     csv_path="employees_updated.csv",
        ...     table_name="employees",
        ...     move_type="existing_schema"
        ... )
        
        >>> # Flexible schema update
        >>> result = await use_case.execute(
        ...     csv_path="employees_v2.csv",
        ...     table_name="employees",
        ...     move_type="new_schema"
        ... )
        
        >>> # Dry run validation
        >>> result = await use_case.execute(
        ...     csv_path="test_data.csv",
        ...     table_name="test_table",
        ...     dry_run=True
        ... )
        
        >>> # S3 integration
        >>> result = await use_case.execute(
        ...     csv_path="s3://data-bucket/employees.csv",
        ...     table_name="employees_s3"
        ... )
    """
```

#### get_operation_preview

```python
async def get_operation_preview(
    self,
    csv_path: str,
    table_name: str,
    move_type: Optional[str] = None,
) -> ValidationReport:
    """
    Get a preview of what the data movement operation would do without executing it.
    
    This is equivalent to running execute() with dry_run=True but only returns
    the validation report for easier integration.
    
    Args:
        csv_path: Path to CSV file
        table_name: Name of target table
        move_type: Type of move operation
        
    Returns:
        ValidationReport: Preview of the operation containing:
        - schema_analysis: Analysis of schema compatibility
        - case_conflicts: Any case-sensitivity conflicts found
        - type_mismatches: Data type compatibility issues
        - constraint_violations: Constraint validation failures
        - recommendations: Suggested actions
        - warnings: Non-blocking issues
        - errors: Validation errors
        - validation_passed: Overall validation result
        
    Raises:
        ValidationError: If validation fails
        DataMoveError: If preview generation fails
        
    Example:
        >>> preview = await use_case.get_operation_preview(
        ...     csv_path="data.csv",
        ...     table_name="target_table"
        ... )
        >>> 
        >>> if preview.validation_passed:
        ...     print("✅ Validation passed")
        ...     print(f"Would process {len(df)} rows")
        ... else:
        ...     print("❌ Validation failed:")
        ...     for error in preview.errors:
        ...         print(f"  - {error}")
    """
```

## Data Models

### DataMoveResult

```python
@dataclass
class DataMoveResult:
    """Result of a data movement operation."""
    
    success: bool                           # Operation success status
    rows_processed: int                     # Number of rows processed
    execution_time: float                   # Total execution time in seconds
    validation_report: ValidationReport     # Detailed validation results
    errors: List[DataMoveError]            # List of errors encountered
    warnings: List[str]                    # List of warnings generated
    table_created: bool = False            # Whether new table was created
    schema_updated: bool = False           # Whether schema was modified
    operation_type: Optional[str] = None   # Type of operation performed
```

### ValidationReport

```python
@dataclass
class ValidationReport:
    """Comprehensive validation report for data movement operations."""
    
    schema_analysis: SchemaAnalysis         # Schema compatibility analysis
    case_conflicts: List[CaseConflict]     # Case-sensitivity conflicts
    type_mismatches: List[TypeMismatch]    # Data type mismatches
    constraint_violations: List[ConstraintViolation]  # Constraint violations
    recommendations: List[str]              # Recommended actions
    warnings: List[str]                    # Non-blocking warnings
    errors: List[str]                      # Validation errors
    validation_passed: bool                # Overall validation result
```

### SchemaAnalysis

```python
@dataclass
class SchemaAnalysis:
    """Analysis of schema compatibility between database and CSV."""
    
    table_exists: bool                     # Whether target table exists
    columns_added: List[str]               # Columns present in CSV but not DB
    columns_removed: List[str]             # Columns present in DB but not CSV
    columns_modified: List[TypeMismatch]   # Columns with type changes
    case_conflicts: List[CaseConflict]     # Case-sensitivity conflicts
    constraint_violations: List[ConstraintViolation]  # Constraint issues
    compatible: bool                       # Overall compatibility
    requires_schema_update: bool           # Whether schema update is needed
```

### TableInfo

```python
@dataclass
class TableInfo:
    """Comprehensive table information."""
    
    name: str                              # Table name
    columns: Dict[str, ColumnInfo]         # Column definitions
    primary_keys: List[str]                # Primary key columns
    constraints: List[Constraint]          # Table constraints
    indexes: List[IndexInfo]               # Table indexes
```

### ColumnInfo

```python
@dataclass
class ColumnInfo:
    """Information about a database column."""
    
    name: str                              # Column name
    data_type: str                         # PostgreSQL data type
    nullable: bool                         # Whether column allows NULL
    default_value: Optional[Any] = None    # Default value if any
    max_length: Optional[int] = None       # Maximum length for text types
    precision: Optional[int] = None        # Precision for numeric types
    scale: Optional[int] = None            # Scale for numeric types
```

### CaseConflict

```python
@dataclass
class CaseConflict:
    """Case-sensitivity conflict between column names."""
    
    db_column: str                         # Database column name
    csv_column: str                        # CSV column name
    conflict_type: str                     # Type of conflict
    # Conflict types: 'case_mismatch', 'duplicate_insensitive'
```

### TypeMismatch

```python
@dataclass
class TypeMismatch:
    """Data type mismatch between database and CSV."""
    
    column_name: str                       # Column with type mismatch
    db_type: str                          # Database column type
    csv_type: str                         # CSV/pandas inferred type
    compatible: bool                      # Whether types are compatible
    conversion_required: bool             # Whether conversion is needed
    sample_values: Optional[List[Any]] = None  # Sample values for analysis
```

### ConstraintViolation

```python
@dataclass
class ConstraintViolation:
    """Constraint violation information."""
    
    constraint_name: str                   # Name of violated constraint
    constraint_type: str                   # Type of constraint
    column_name: str                       # Column involved in violation
    violation_type: str                    # Type of violation
    affected_rows: Optional[int] = None    # Number of affected rows
    sample_violations: Optional[List[Any]] = None  # Sample violating values
```

## Exception Classes

### Exception Hierarchy

```python
DataMoveError (Base exception)
├── ValidationError (Schema/data validation failures)
│   ├── SchemaConflictError (Schema compatibility issues)
│   ├── CaseSensitivityError (Case-sensitive column conflicts)
│   └── DataTypeError (Data type conversion issues)
└── DatabaseOperationError (Database connection/operation failures)
```

### DataMoveError

```python
class DataMoveError(Exception):
    """Base exception for DataMove operations."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize DataMove error with context.
        
        Args:
            message: Human-readable error message
            context: Dictionary containing error context and debugging information
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
```

**Common Context Keys:**
- `error_type`: Specific error category
- `file_path`: Path to CSV file
- `table_name`: Target table name
- `operation_stage`: Where the error occurred
- `original_error`: Original exception message
- `suggestion`: Recommended solution

### ValidationError

```python
class ValidationError(DataMoveError):
    """Schema or data validation failures."""
    pass
```

**Common Scenarios:**
- Missing required columns
- Extra columns in CSV
- Data type mismatches
- Constraint violations
- Invalid move_type parameter

### SchemaConflictError

```python
class SchemaConflictError(ValidationError):
    """Schema compatibility issues."""
    pass
```

**Common Scenarios:**
- Column type incompatibilities
- Nullable constraint mismatches
- Primary key conflicts

### CaseSensitivityError

```python
class CaseSensitivityError(ValidationError):
    """Case-sensitive column name conflicts."""
    pass
```

**Common Scenarios:**
- CSV has "Name" but DB has "name"
- Multiple columns with case-only differences

### DatabaseOperationError

```python
class DatabaseOperationError(DataMoveError):
    """Database connection or operation failures."""
    pass
```

**Common Scenarios:**
- Connection timeouts
- Authentication failures
- Transaction rollback failures
- SQL execution errors

## Repository Interface

### DataMoveRepositoryInterface

```python
class DataMoveRepositoryInterface(ABC):
    """Abstract interface for database operations specific to data movement."""
    
    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database."""
        pass
    
    @abstractmethod
    async def get_table_info(self, table_name: str) -> TableInfo:
        """Get comprehensive table information."""
        pass
    
    @abstractmethod
    async def create_table_from_dataframe(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        primary_key_columns: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Create table from DataFrame schema."""
        pass
    
    @abstractmethod
    async def replace_table_data(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> int:
        """Replace all data in table with DataFrame data."""
        pass
    
    @abstractmethod
    async def analyze_schema_compatibility(
        self, 
        table_name: str, 
        df: pd.DataFrame
    ) -> SchemaAnalysis:
        """Analyze schema compatibility between table and DataFrame."""
        pass
    
    @abstractmethod
    async def get_column_case_conflicts(
        self, 
        table_name: str, 
        df_columns: List[str]
    ) -> List[CaseConflict]:
        """Detect case-sensitivity conflicts in column names."""
        pass
```

## Storage Loaders

### StorageLoaderInterface

```python
class StorageLoaderInterface(ABC):
    """Abstract interface for loading CSV data from various sources."""
    
    @abstractmethod
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV data from the specified path."""
        pass
```

### LocalLoader

```python
class LocalLoader(StorageLoaderInterface):
    """Loader for local CSV files."""
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV from local file system.
        
        Args:
            file_path: Path to local CSV file
            
        Returns:
            DataFrame containing CSV data
            
        Raises:
            DBOperationError: If file loading fails
        """
```

### S3Loader

```python
class S3Loader(StorageLoaderInterface):
    """Loader for CSV files stored in AWS S3."""
    
    def load_csv(self, s3_uri: str) -> pd.DataFrame:
        """
        Load CSV from S3 bucket.
        
        Args:
            s3_uri: S3 URI in format s3://bucket/key
            
        Returns:
            DataFrame containing CSV data
            
        Raises:
            DBOperationError: If S3 loading fails
        """
```

## Validation Service

### ValidationService

```python
class ValidationService:
    """Centralized validation logic for different move scenarios."""
    
    async def validate_data_move(
        self,
        table_info: Optional[TableInfo],
        df: pd.DataFrame,
        move_type: Optional[str]
    ) -> ValidationReport:
        """
        Validate data movement operation.
        
        Args:
            table_info: Target table information (None for new tables)
            df: DataFrame containing CSV data
            move_type: Type of move operation
            
        Returns:
            ValidationReport: Comprehensive validation results
        """
```

## Usage Patterns

### Basic Usage Pattern

```python
from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
from dataload.infrastructure.db.db_connection import DBConnection

async def basic_data_move():
    # 1. Set up database connection
    db_connection = DBConnection()
    await db_connection.initialize()
    
    # 2. Create repository
    repository = PostgresDataMoveRepository(db_connection)
    
    # 3. Create use case with auto-loader
    use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
    
    # 4. Execute data move
    result = await use_case.execute(
        csv_path="data.csv",
        table_name="target_table",
        primary_key_columns=["id"]
    )
    
    # 5. Handle result
    if result.success:
        print(f"✅ Success: {result.rows_processed} rows processed")
    else:
        print(f"❌ Failed: {result.errors}")
    
    # 6. Close connection
    await db_connection.close()
```

### Error Handling Pattern

```python
async def robust_data_move():
    try:
        result = await use_case.execute(
            csv_path="data.csv",
            table_name="target_table",
            move_type="existing_schema"
        )
        
        return {"success": True, "rows": result.rows_processed}
        
    except ValidationError as e:
        # Handle validation failures
        if isinstance(e, CaseSensitivityError):
            return {"error": "case_conflicts", "details": e.context}
        elif isinstance(e, SchemaConflictError):
            return {"error": "schema_mismatch", "suggestion": "try new_schema mode"}
        else:
            return {"error": "validation_failed", "details": e.context}
            
    except DatabaseOperationError as e:
        # Handle database failures
        return {"error": "database_failed", "details": e.context}
        
    except DataMoveError as e:
        # Handle other DataMove errors
        return {"error": "operation_failed", "details": e.context}
```

### Dry-Run Validation Pattern

```python
async def validate_before_execute():
    # 1. Get operation preview
    preview = await use_case.get_operation_preview(
        csv_path="data.csv",
        table_name="target_table"
    )
    
    # 2. Check validation results
    if not preview.validation_passed:
        print("❌ Validation failed:")
        for error in preview.errors:
            print(f"  - {error}")
        return False
    
    # 3. Show what will happen
    print("✅ Validation passed")
    if preview.schema_analysis.columns_added:
        print(f"➕ Will add columns: {preview.schema_analysis.columns_added}")
    if preview.schema_analysis.columns_removed:
        print(f"➖ Will remove columns: {preview.schema_analysis.columns_removed}")
    
    # 4. Execute if validation passed
    result = await use_case.execute(
        csv_path="data.csv",
        table_name="target_table",
        move_type="new_schema"
    )
    
    return result.success
```

### S3 Integration Pattern

```python
async def s3_data_move():
    # DataMove automatically detects S3 URIs
    s3_uri = "s3://data-bucket/path/to/file.csv"
    
    try:
        result = await use_case.execute(
            csv_path=s3_uri,
            table_name="s3_data_table"
        )
        
        print(f"✅ S3 data loaded: {result.rows_processed} rows")
        
    except DataMoveError as e:
        if e.context.get("error_type") == "s3_operation_failed":
            print("💡 Check AWS credentials and S3 permissions")
        raise
```

## Configuration

### Environment Variables

```bash
# Database Configuration
LOCAL_POSTGRES_HOST=localhost
LOCAL_POSTGRES_PORT=5432
LOCAL_POSTGRES_DB=your_database
LOCAL_POSTGRES_USER=your_username
LOCAL_POSTGRES_PASSWORD=your_password

# AWS Configuration (for S3)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

### Database Connection Configuration

```python
from dataload.infrastructure.db.db_connection import DBConnection

# Method 1: Using environment variables
db_connection = DBConnection()

# Method 2: Direct configuration
db_connection = DBConnection(
    host="localhost",
    port=5432,
    database="your_db",
    user="your_user",
    password="your_password"
)
```

### Performance Tuning

```python
# Adjust batch size based on data characteristics
batch_sizes = {
    "small_data": 500,      # < 10MB CSV files
    "medium_data": 1000,    # 10MB - 100MB CSV files  
    "large_data": 2000,     # 100MB - 1GB CSV files
    "huge_data": 5000,      # > 1GB CSV files
}

# Use appropriate batch size
result = await use_case.execute(
    csv_path="large_file.csv",
    table_name="target_table",
    batch_size=batch_sizes["large_data"]
)
```

---

*This API documentation covers all public interfaces and usage patterns for the DataMove use case. For implementation details and examples, see the comprehensive example scripts and troubleshooting guide.*