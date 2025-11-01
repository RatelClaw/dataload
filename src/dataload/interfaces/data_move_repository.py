from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataload.infrastructure.db.data_repository import DataRepositoryInterface
from dataload.domain.entities import (
    TableInfo,
    SchemaAnalysis,
    CaseConflict,
    TypeMismatch,
    ConstraintViolation,
)


class DataMoveRepositoryInterface(DataRepositoryInterface):
    """
    Abstract interface for data movement operations extending existing repository patterns.
    
    This interface extends DataRepositoryInterface to add specific methods for table analysis,
    schema validation, and data operations required by the DataMove use case while maintaining
    compatibility with existing repository design patterns.
    """

    # Table Analysis Methods
    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
            
        Raises:
            DatabaseOperationError: If database connection or query fails
        """
        pass

    @abstractmethod
    async def get_table_info(self, table_name: str) -> TableInfo:
        """
        Get comprehensive information about a table including columns, constraints, and indexes.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            TableInfo object containing complete table metadata
            
        Raises:
            DatabaseOperationError: If table doesn't exist or query fails
        """
        pass

    # Schema Analysis and Validation Methods
    @abstractmethod
    async def analyze_schema_compatibility(
        self, table_name: str, df: pd.DataFrame
    ) -> SchemaAnalysis:
        """
        Analyze compatibility between existing table schema and DataFrame schema.
        
        Args:
            table_name: Name of the existing table
            df: DataFrame containing the data to be moved
            
        Returns:
            SchemaAnalysis object with detailed compatibility information
            
        Raises:
            DatabaseOperationError: If table analysis fails
        """
        pass

    @abstractmethod
    async def get_column_case_conflicts(
        self, table_name: str, df_columns: List[str]
    ) -> List[CaseConflict]:
        """
        Detect case-sensitivity conflicts between database columns and DataFrame columns.
        
        Args:
            table_name: Name of the table to check against
            df_columns: List of DataFrame column names
            
        Returns:
            List of CaseConflict objects describing any conflicts found
            
        Raises:
            DatabaseOperationError: If table analysis fails
        """
        pass

    @abstractmethod
    async def validate_type_compatibility(
        self, table_name: str, df: pd.DataFrame
    ) -> List[TypeMismatch]:
        """
        Validate data type compatibility between table and DataFrame.
        
        Args:
            table_name: Name of the table to validate against
            df: DataFrame containing the data to validate
            
        Returns:
            List of TypeMismatch objects for incompatible types
            
        Raises:
            DatabaseOperationError: If validation fails
        """
        pass

    @abstractmethod
    async def validate_constraints(
        self, table_name: str, df: pd.DataFrame
    ) -> List[ConstraintViolation]:
        """
        Validate that DataFrame data doesn't violate table constraints.
        
        Args:
            table_name: Name of the table with constraints
            df: DataFrame containing the data to validate
            
        Returns:
            List of ConstraintViolation objects for any violations found
            
        Raises:
            DatabaseOperationError: If constraint validation fails
        """
        pass

    # Data Movement Methods
    @abstractmethod
    async def create_table_from_dataframe(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        primary_key_columns: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Create a new table based on DataFrame schema without embedding-specific columns.
        
        This method creates a table optimized for plain data storage without the
        embedding-related columns (embeddings, embed_columns_names, etc.) that are
        added by the existing create_table method.
        
        Args:
            table_name: Name of the table to create
            df: DataFrame defining the schema and data types
            primary_key_columns: Optional list of columns to use as primary key
            
        Returns:
            Dictionary mapping column names to PostgreSQL data types
            
        Raises:
            DatabaseOperationError: If table creation fails
            ValidationError: If DataFrame schema is invalid
        """
        pass

    @abstractmethod
    async def replace_table_data(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> int:
        """
        Replace all data in an existing table with DataFrame data.
        
        This operation truncates the existing table and inserts new data in a
        transaction-safe manner with rollback capability on failure.
        
        Args:
            table_name: Name of the table to replace data in
            df: DataFrame containing the new data
            batch_size: Number of rows to process in each batch
            
        Returns:
            Number of rows successfully inserted
            
        Raises:
            DatabaseOperationError: If data replacement fails
            ValidationError: If data doesn't match table schema
        """
        pass

    @abstractmethod
    async def update_table_schema(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        allow_column_drops: bool = False
    ) -> Dict[str, str]:
        """
        Update table schema to match DataFrame schema for new_schema mode.
        
        This method adds new columns and optionally drops columns that are not
        present in the DataFrame, while preserving existing data where possible.
        
        Args:
            table_name: Name of the table to update
            df: DataFrame with the target schema
            allow_column_drops: Whether to drop columns not in DataFrame
            
        Returns:
            Dictionary of schema changes made (column_name -> action)
            
        Raises:
            DatabaseOperationError: If schema update fails
            SchemaConflictError: If incompatible changes are required
        """
        pass

    # Bulk Operations for Performance
    @abstractmethod
    async def bulk_insert_data(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        batch_size: int = 1000,
        on_conflict: str = "error"
    ) -> int:
        """
        Perform optimized bulk insert of DataFrame data.
        
        Uses PostgreSQL-specific bulk insert optimizations like COPY or batch inserts
        for maximum performance with large datasets.
        
        Args:
            table_name: Name of the table to insert into
            df: DataFrame containing the data to insert
            batch_size: Number of rows to process in each batch
            on_conflict: How to handle conflicts ('error', 'ignore', 'update')
            
        Returns:
            Number of rows successfully inserted
            
        Raises:
            DatabaseOperationError: If bulk insert fails
            ValidationError: If data validation fails
        """
        pass

    # Transaction Management
    @abstractmethod
    async def execute_in_transaction(self, operations: List[callable]) -> bool:
        """
        Execute multiple operations within a single transaction.
        
        All operations succeed or all are rolled back on any failure.
        
        Args:
            operations: List of async callable operations to execute
            
        Returns:
            True if all operations succeeded, False otherwise
            
        Raises:
            DatabaseOperationError: If transaction management fails
        """
        pass

    # PostgreSQL-Specific Operations
    @abstractmethod
    async def get_postgresql_version(self) -> str:
        """
        Get PostgreSQL server version for compatibility checks.
        
        Returns:
            PostgreSQL version string
            
        Raises:
            DatabaseOperationError: If version query fails
        """
        pass

    @abstractmethod
    async def analyze_table_statistics(self, table_name: str) -> Dict[str, any]:
        """
        Get PostgreSQL table statistics for performance optimization.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            Dictionary containing table statistics (row count, size, etc.)
            
        Raises:
            DatabaseOperationError: If statistics query fails
        """
        pass

    @abstractmethod
    async def optimize_table_performance(self, table_name: str) -> None:
        """
        Run PostgreSQL-specific optimizations on the table.
        
        This may include VACUUM, ANALYZE, or index optimization operations.
        
        Args:
            table_name: Name of the table to optimize
            
        Raises:
            DatabaseOperationError: If optimization fails
        """
        pass

    # Vector Column Handling (for compatibility with existing vector operations)
    @abstractmethod
    async def detect_vector_columns(self, table_name: str) -> List[Tuple[str, int]]:
        """
        Detect vector columns and their dimensions in the table.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            List of tuples (column_name, dimension) for vector columns
            
        Raises:
            DatabaseOperationError: If vector detection fails
        """
        pass

    @abstractmethod
    async def validate_vector_dimensions(
        self, 
        table_name: str, 
        df: pd.DataFrame
    ) -> List[str]:
        """
        Validate that vector columns in DataFrame match table vector dimensions.
        
        Args:
            table_name: Name of the table with vector columns
            df: DataFrame containing vector data to validate
            
        Returns:
            List of error messages for dimension mismatches
            
        Raises:
            DatabaseOperationError: If validation fails
        """
        pass