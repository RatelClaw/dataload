import pandas as pd
import numpy as np
import json
import time
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from asyncpg.exceptions import PostgresError
from contextlib import asynccontextmanager

from dataload.interfaces.data_move_repository import DataMoveRepositoryInterface
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.domain.entities import (
    TableInfo,
    ColumnInfo,
    Constraint,
    IndexInfo,
    SchemaAnalysis,
    CaseConflict,
    TypeMismatch,
    ConstraintViolation,
    DatabaseOperationError,
    ValidationError,
    SchemaConflictError,
    DataTypeError,
)
from dataload.config import logger
from tenacity import retry, stop_after_attempt, wait_fixed


class PostgresDataMoveRepository(DataMoveRepositoryInterface):
    """
    PostgreSQL-specific implementation of data movement operations.
    
    This repository provides optimized PostgreSQL operations for the DataMove use case,
    including table analysis, schema validation, and bulk data operations without
    the embedding-specific overhead of the main DataRepository.
    """

    def __init__(self, db_connection: DBConnection):
        """
        Initialize the repository with a database connection.
        
        Args:
            db_connection: DBConnection instance for database operations
        """
        self.db = db_connection

    # Table Analysis Methods
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
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
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = $1
        )
        """
        try:
            async with self.db.get_connection() as conn:
                result = await conn.fetchval(query, table_name)
                return bool(result)
        except PostgresError as e:
            logger.error(f"Error checking table existence for {table_name}: {e}")
            raise DatabaseOperationError(f"Failed to check table existence: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
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
        if not await self.table_exists(table_name):
            raise DatabaseOperationError(f"Table {table_name} does not exist")

        try:
            async with self.db.get_connection() as conn:
                # Get column information
                columns_query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position
                """
                column_rows = await conn.fetch(columns_query, table_name)
                
                # Get primary key information
                pk_query = """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = $1 AND tc.constraint_type = 'PRIMARY KEY'
                ORDER BY kcu.ordinal_position
                """
                pk_rows = await conn.fetch(pk_query, table_name)
                
                # Get constraint information
                constraints_query = """
                SELECT 
                    tc.constraint_name,
                    tc.constraint_type,
                    kcu.column_name,
                    ccu.table_name AS referenced_table,
                    ccu.column_name AS referenced_column
                FROM information_schema.table_constraints tc
                LEFT JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                LEFT JOIN information_schema.constraint_column_usage ccu 
                    ON tc.constraint_name = ccu.constraint_name
                WHERE tc.table_name = $1
                """
                constraint_rows = await conn.fetch(constraints_query, table_name)
                
                # Get index information
                indexes_query = """
                SELECT 
                    i.relname AS index_name,
                    a.attname AS column_name,
                    am.amname AS index_type,
                    ix.indisunique AS is_unique
                FROM pg_class t
                JOIN pg_index ix ON t.oid = ix.indrelid
                JOIN pg_class i ON i.oid = ix.indexrelid
                JOIN pg_am am ON i.relam = am.oid
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
                WHERE t.relname = $1 AND t.relkind = 'r'
                """
                index_rows = await conn.fetch(indexes_query, table_name)

            # Process column information
            columns = {}
            for row in column_rows:
                columns[row['column_name']] = ColumnInfo(
                    name=row['column_name'],
                    data_type=row['data_type'],
                    nullable=row['is_nullable'] == 'YES',
                    default_value=row['column_default'],
                    max_length=row['character_maximum_length'],
                    precision=row['numeric_precision'],
                    scale=row['numeric_scale']
                )

            # Process primary keys
            primary_keys = [row['column_name'] for row in pk_rows]

            # Process constraints
            constraints = []
            constraint_dict = {}
            for row in constraint_rows:
                constraint_name = row['constraint_name']
                if constraint_name not in constraint_dict:
                    constraint_dict[constraint_name] = {
                        'name': constraint_name,
                        'type': row['constraint_type'],
                        'columns': [],
                        'referenced_table': row['referenced_table'],
                        'referenced_columns': []
                    }
                
                if row['column_name']:
                    constraint_dict[constraint_name]['columns'].append(row['column_name'])
                if row['referenced_column']:
                    constraint_dict[constraint_name]['referenced_columns'].append(row['referenced_column'])

            for constraint_data in constraint_dict.values():
                constraints.append(Constraint(
                    name=constraint_data['name'],
                    type=constraint_data['type'],
                    columns=constraint_data['columns'],
                    referenced_table=constraint_data['referenced_table'],
                    referenced_columns=constraint_data['referenced_columns'] or None
                ))

            # Process indexes
            indexes = []
            index_dict = {}
            for row in index_rows:
                index_name = row['index_name']
                if index_name not in index_dict:
                    index_dict[index_name] = {
                        'name': index_name,
                        'columns': [],
                        'index_type': row['index_type'],
                        'unique': row['is_unique']
                    }
                index_dict[index_name]['columns'].append(row['column_name'])

            for index_data in index_dict.values():
                indexes.append(IndexInfo(
                    name=index_data['name'],
                    columns=index_data['columns'],
                    index_type=index_data['index_type'],
                    unique=index_data['unique']
                ))

            return TableInfo(
                name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                constraints=constraints,
                indexes=indexes
            )

        except PostgresError as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            raise DatabaseOperationError(f"Failed to get table information: {e}")

    # Schema Analysis and Validation Methods
    
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
        try:
            table_info = await self.get_table_info(table_name)
            df_columns = set(df.columns)
            db_columns = set(table_info.columns.keys())
            
            # Analyze column differences
            columns_added = list(df_columns - db_columns)
            columns_removed = list(db_columns - df_columns)
            
            # Analyze type mismatches for common columns
            columns_modified = []
            common_columns = df_columns & db_columns
            
            for col in common_columns:
                db_type = table_info.columns[col].data_type
                df_type = self._infer_postgres_type(df[col])
                
                if not self._are_types_compatible(db_type, df_type):
                    columns_modified.append(TypeMismatch(
                        column_name=col,
                        db_type=db_type,
                        csv_type=df_type,
                        compatible=False,
                        conversion_required=True,
                        sample_values=df[col].dropna().head(3).tolist()
                    ))
            
            # Validate vector dimensions for vector columns
            vector_dimension_errors = await self.validate_vector_dimensions(table_name, df)
            if vector_dimension_errors:
                # Add vector dimension errors as type mismatches
                for error in vector_dimension_errors:
                    columns_modified.append(TypeMismatch(
                        column_name="vector_validation",
                        db_type="vector",
                        csv_type="vector",
                        compatible=False,
                        conversion_required=False,
                        sample_values=[error]
                    ))
            
            # Check case conflicts
            case_conflicts = await self.get_column_case_conflicts(table_name, list(df_columns))
            
            # Check constraint violations
            constraint_violations = await self.validate_constraints(table_name, df)
            
            # Determine overall compatibility
            compatible = (
                len(columns_modified) == 0 and 
                len(case_conflicts) == 0 and 
                len(constraint_violations) == 0
            )
            
            requires_schema_update = len(columns_added) > 0 or len(columns_removed) > 0
            
            return SchemaAnalysis(
                table_exists=True,
                columns_added=columns_added,
                columns_removed=columns_removed,
                columns_modified=columns_modified,
                case_conflicts=case_conflicts,
                constraint_violations=constraint_violations,
                compatible=compatible,
                requires_schema_update=requires_schema_update
            )
            
        except Exception as e:
            logger.error(f"Error analyzing schema compatibility: {e}")
            raise DatabaseOperationError(f"Schema analysis failed: {e}")

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
        try:
            table_info = await self.get_table_info(table_name)
            db_columns = list(table_info.columns.keys())
            
            conflicts = []
            
            # Create case-insensitive mapping of database columns
            db_lower_map = {col.lower(): col for col in db_columns}
            
            # Check for case conflicts
            for df_col in df_columns:
                df_col_lower = df_col.lower()
                
                # If there's a case-insensitive match but not exact match
                if df_col_lower in db_lower_map and df_col != db_lower_map[df_col_lower]:
                    conflicts.append(CaseConflict(
                        db_column=db_lower_map[df_col_lower],
                        csv_column=df_col,
                        conflict_type='case_mismatch'
                    ))
            
            # Check for duplicate case-insensitive columns in DataFrame
            df_lower_counts = {}
            for df_col in df_columns:
                df_col_lower = df_col.lower()
                if df_col_lower not in df_lower_counts:
                    df_lower_counts[df_col_lower] = []
                df_lower_counts[df_col_lower].append(df_col)
            
            for lower_col, cols in df_lower_counts.items():
                if len(cols) > 1:
                    for col in cols:
                        conflicts.append(CaseConflict(
                            db_column='',
                            csv_column=col,
                            conflict_type='duplicate_insensitive'
                        ))
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error detecting case conflicts: {e}")
            raise DatabaseOperationError(f"Case conflict detection failed: {e}")

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
        try:
            table_info = await self.get_table_info(table_name)
            mismatches = []
            
            for col in df.columns:
                if col in table_info.columns:
                    db_type = table_info.columns[col].data_type
                    df_type = self._infer_postgres_type(df[col])
                    
                    compatible = self._are_types_compatible(db_type, df_type)
                    conversion_required = not compatible and self._can_convert_type(db_type, df_type)
                    
                    if not compatible:
                        mismatches.append(TypeMismatch(
                            column_name=col,
                            db_type=db_type,
                            csv_type=df_type,
                            compatible=compatible,
                            conversion_required=conversion_required,
                            sample_values=df[col].dropna().head(3).tolist()
                        ))
            
            return mismatches
            
        except Exception as e:
            logger.error(f"Error validating type compatibility: {e}")
            raise DatabaseOperationError(f"Type validation failed: {e}")

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
        try:
            table_info = await self.get_table_info(table_name)
            violations = []
            
            # Check primary key constraints
            for constraint in table_info.constraints:
                if constraint.type == 'PRIMARY KEY':
                    for col in constraint.columns:
                        if col in df.columns:
                            # Check for null values in primary key columns
                            null_count = df[col].isnull().sum()
                            if null_count > 0:
                                violations.append(ConstraintViolation(
                                    constraint_name=constraint.name,
                                    constraint_type='PRIMARY KEY',
                                    column_name=col,
                                    violation_type='null_in_primary_key',
                                    affected_rows=null_count
                                ))
                            
                            # Check for duplicate values in primary key columns
                            duplicate_count = df[col].duplicated().sum()
                            if duplicate_count > 0:
                                violations.append(ConstraintViolation(
                                    constraint_name=constraint.name,
                                    constraint_type='PRIMARY KEY',
                                    column_name=col,
                                    violation_type='duplicate_primary_key',
                                    affected_rows=duplicate_count,
                                    sample_violations=df[df[col].duplicated()][col].head(3).tolist()
                                ))
                
                # Check NOT NULL constraints
                elif constraint.type == 'NOT NULL':
                    for col in constraint.columns:
                        if col in df.columns:
                            null_count = df[col].isnull().sum()
                            if null_count > 0:
                                violations.append(ConstraintViolation(
                                    constraint_name=constraint.name,
                                    constraint_type='NOT NULL',
                                    column_name=col,
                                    violation_type='null_value',
                                    affected_rows=null_count
                                ))
            
            # Check column nullability from table info
            for col_name, col_info in table_info.columns.items():
                if col_name in df.columns and not col_info.nullable:
                    null_count = df[col_name].isnull().sum()
                    if null_count > 0:
                        violations.append(ConstraintViolation(
                            constraint_name=f"{col_name}_not_null",
                            constraint_type='NOT NULL',
                            column_name=col_name,
                            violation_type='null_value',
                            affected_rows=null_count
                        ))
            
            return violations
            
        except Exception as e:
            logger.error(f"Error validating constraints: {e}")
            raise DatabaseOperationError(f"Constraint validation failed: {e}")

    # Data Movement Methods - Placeholder implementations for now
    
    async def create_table_from_dataframe(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        primary_key_columns: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Create a new table based on DataFrame schema without embedding-specific columns.
        
        This method creates a table optimized for plain data storage without the
        embedding-related columns that are added by the existing create_table method.
        Includes proper vector column detection and indexing.
        
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
        if df.empty:
            raise ValidationError("Cannot create table from empty DataFrame")
        
        if primary_key_columns:
            missing_pk_cols = [col for col in primary_key_columns if col not in df.columns]
            if missing_pk_cols:
                raise ValidationError(f"Primary key columns not found in DataFrame: {missing_pk_cols}")
            
            # Check for null values in primary key columns
            for col in primary_key_columns:
                if df[col].isnull().any():
                    raise ValidationError(f"Primary key column '{col}' contains null values")
        
        try:
            column_types = {}
            columns = []
            vector_columns = []
            
            # Generate column definitions and detect vector columns
            for col in df.columns:
                pg_type = self._infer_postgres_type(df[col])
                not_null = " NOT NULL" if primary_key_columns and col in primary_key_columns else ""
                columns.append(f'"{col}" {pg_type}{not_null}')
                column_types[col] = pg_type
                
                # Track vector columns for indexing
                if pg_type.startswith('vector('):
                    dimension = self._extract_vector_dimension(pg_type)
                    if dimension:
                        vector_columns.append((col, dimension))
                        logger.info(f"Detected vector column '{col}' with dimension {dimension}")
            
            # Add primary key constraint if specified
            if primary_key_columns:
                quoted_pk_columns = [f'"{col}"' for col in primary_key_columns]
                columns.append(f"PRIMARY KEY ({', '.join(quoted_pk_columns)})")
            
            # Create table and vector indexes
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
            
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    await conn.execute(query)
                    
                    # Create vector indexes if any vector columns were detected
                    if vector_columns:
                        logger.info(f"Creating vector indexes for {len(vector_columns)} vector columns")
                        await self._create_vector_indexes_in_transaction(
                            conn, table_name, vector_columns
                        )
            
            logger.info(
                f"Created table {table_name} with {len(df.columns)} columns "
                f"({len(vector_columns)} vector columns)"
            )
            return column_types
            
        except PostgresError as e:
            logger.error(f"Error creating table {table_name}: {e}")
            raise DatabaseOperationError(f"Failed to create table: {e}")

    async def _create_vector_indexes_in_transaction(
        self, 
        conn, 
        table_name: str, 
        vector_columns: List[Tuple[str, int]]
    ) -> None:
        """
        Create vector indexes within an existing transaction.
        
        Args:
            conn: Database connection with active transaction
            table_name: Name of the table
            vector_columns: List of (column_name, dimension) tuples
        """
        for column_name, dimension in vector_columns:
            # Choose index type based on dimension
            if dimension > 2000:
                # Use IVFFlat for high dimensions
                lists_param = max(10, min(1000, dimension // 10))
                index_query = f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_{column_name}_ivfflat
                    ON {table_name} 
                    USING ivfflat ("{column_name}" vector_cosine_ops) 
                    WITH (lists = {lists_param})
                """
                index_type = "ivfflat"
            else:
                # Use HNSW for lower dimensions (better performance)
                index_query = f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_{column_name}_hnsw
                    ON {table_name} 
                    USING hnsw ("{column_name}" vector_cosine_ops) 
                    WITH (m = 16, ef_construction = 64)
                """
                index_type = "hnsw"
            
            await conn.execute(index_query)
            logger.debug(f"Created {index_type} index for vector column {column_name} (dim={dimension})")

    async def replace_table_data(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> int:
        """
        Replace all data in an existing table with DataFrame data.
        
        This operation truncates the existing table and inserts new data in a
        transaction-safe manner with rollback capability on failure. Includes
        performance optimization and memory management for large datasets.
        
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
        start_time = time.time()
        
        if df.empty:
            logger.warning(f"Replacing table {table_name} with empty DataFrame")
            # Still truncate the table even if DataFrame is empty
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    await conn.execute(f"TRUNCATE TABLE {table_name}")
            return 0
        
        try:
            # Validate memory usage for large datasets
            self._validate_memory_usage(df, max_memory_mb=1000)
            
            # Validate table exists
            if not await self.table_exists(table_name):
                raise ValidationError(f"Table {table_name} does not exist")
            
            # Get table schema for validation
            table_info = await self.get_table_info(table_name)
            
            # Validate DataFrame columns match table columns
            df_columns = set(df.columns)
            db_columns = set(table_info.columns.keys())
            
            missing_cols = db_columns - df_columns
            extra_cols = df_columns - db_columns
            
            if missing_cols:
                raise ValidationError(f"DataFrame missing required columns: {missing_cols}")
            if extra_cols:
                raise ValidationError(f"DataFrame has extra columns not in table: {extra_cols}")
            
            # Detect vector columns in the table
            vector_columns = await self.detect_vector_columns(table_name)
            
            # Convert DataFrame to match PostgreSQL types
            logger.info(f"Converting data types for {len(df)} rows in table {table_name}")
            df_converted = self._convert_dataframe_types(df, table_info)
            
            # Apply vector-specific conversions if needed
            if vector_columns:
                logger.info(f"Converting {len(vector_columns)} vector columns")
                df_converted = self._convert_vector_data(df_converted, vector_columns)
            
            # Optimize batch size based on data characteristics
            optimal_batch_size = self._calculate_optimal_batch_size(df_converted)
            final_batch_size = min(batch_size, optimal_batch_size)
            
            logger.info(f"Using batch size {final_batch_size} for data replacement")
            
            rows_inserted = 0
            
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    # Truncate existing data
                    await conn.execute(f"TRUNCATE TABLE {table_name}")
                    logger.info(f"Truncated existing data in table {table_name}")
                    
                    # Insert new data in batches with progress monitoring
                    if not df_converted.empty:
                        rows_inserted = await self._bulk_insert_data(
                            conn, table_name, df_converted, final_batch_size
                        )
            
            elapsed_time = time.time() - start_time
            rows_per_second = rows_inserted / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(
                f"Replaced data in table {table_name}: {rows_inserted:,} rows inserted "
                f"in {elapsed_time:.2f}s ({rows_per_second:.0f} rows/sec)"
            )
            return rows_inserted
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error replacing table data after {elapsed_time:.2f}s: {e}")
            raise DatabaseOperationError(f"Failed to replace table data: {e}")

    # Enhanced Transaction Management with Comprehensive Error Handling
    
    @asynccontextmanager
    async def transaction(self):
        """
        Provide a transaction context manager with comprehensive error handling and rollback.
        
        This context manager ensures that all operations within the transaction are
        either committed together or rolled back on any failure, with detailed
        error reporting and connection management.
        
        Yields:
            Database connection with active transaction
            
        Raises:
            DatabaseOperationError: If transaction management fails
        """
        try:
            # Get database connection using the async context manager
            async with self.db.get_connection() as connection:
                logger.debug("Acquired database connection for transaction")
                
                # Start transaction
                async with connection.transaction():
                    logger.debug("Transaction started successfully")
                    
                    try:
                        yield connection
                        logger.debug("Transaction committed successfully")
                        
                    except Exception as operation_error:
                        # The transaction context manager will automatically rollback
                        logger.warning(f"Operation failed, transaction will be rolled back: {operation_error}")
                        
                        # Re-raise the original operation error
                        if isinstance(operation_error, (DatabaseOperationError, ValidationError)):
                            raise operation_error
                        else:
                            raise DatabaseOperationError(
                                f"Transaction operation failed: {operation_error}",
                                {
                                    "original_error": str(operation_error),
                                    "error_type": "transaction_operation_failed",
                                    "exception_type": type(operation_error).__name__
                                }
                            ) from operation_error
                    
        except DatabaseOperationError:
            # Re-raise database operation errors as-is
            raise
            
        except ConnectionError as e:
            error_msg = "Failed to establish database connection for transaction"
            logger.error(f"{error_msg}: {e}")
            raise DatabaseOperationError(error_msg, {
                "error_type": "connection_failed",
                "original_error": str(e)
            }) from e
            
        except Exception as e:
            error_msg = f"Unexpected error in transaction management: {e}"
            logger.error(error_msg, exc_info=True)
            raise DatabaseOperationError(error_msg, {
                "error_type": "transaction_management_failed",
                "original_error": str(e),
                "exception_type": type(e).__name__
            }) from e
            if connection:
                try:
                    await connection.close()
                    logger.debug("Database connection closed")
                except Exception as close_error:
                    logger.warning(f"Error closing database connection: {close_error}")

    # Note: This method is not currently used and has been removed to avoid
    # async context manager confusion. The @retry decorator on individual methods
    # provides sufficient retry logic for database operations.

    async def execute_in_transaction(self, operations: List[callable]) -> bool:
        """
        Execute multiple operations within a single transaction with comprehensive error handling.
        
        All operations succeed or all are rolled back on any failure. This method
        provides detailed error reporting and handles various failure scenarios.
        
        Args:
            operations: List of async callable operations to execute
            
        Returns:
            True if all operations succeeded, False otherwise
            
        Raises:
            DatabaseOperationError: If transaction management fails
            ValidationError: If any operation validation fails
        """
        if not operations:
            logger.warning("No operations provided to execute_in_transaction")
            return True
        
        operation_results = []
        failed_operation_index = None
        
        try:
            async with self.transaction() as conn:
                logger.info(f"Executing {len(operations)} operations in transaction")
                
                for i, operation in enumerate(operations):
                    try:
                        logger.debug(f"Executing operation {i + 1}/{len(operations)}")
                        result = await operation(conn)
                        operation_results.append(result)
                        
                    except Exception as op_error:
                        failed_operation_index = i
                        logger.error(f"Operation {i + 1} failed: {op_error}")
                        
                        # Enhance error with operation context
                        if isinstance(op_error, (DatabaseOperationError, ValidationError)):
                            if not hasattr(op_error, 'context') or not op_error.context:
                                op_error.context = {}
                            op_error.context.update({
                                "failed_operation_index": i,
                                "total_operations": len(operations),
                                "completed_operations": len(operation_results)
                            })
                            raise op_error
                        else:
                            raise DatabaseOperationError(
                                f"Operation {i + 1} failed: {op_error}",
                                {
                                    "failed_operation_index": i,
                                    "total_operations": len(operations),
                                    "completed_operations": len(operation_results),
                                    "original_error": str(op_error),
                                    "exception_type": type(op_error).__name__
                                }
                            ) from op_error
                
                logger.info(f"All {len(operations)} operations completed successfully")
                return True
                
        except (DatabaseOperationError, ValidationError):
            # Re-raise known errors (transaction already rolled back)
            logger.error(
                f"Transaction failed at operation {failed_operation_index + 1 if failed_operation_index is not None else 'unknown'}"
            )
            raise
            
        except Exception as e:
            error_msg = f"Unexpected error during transaction execution: {e}"
            logger.error(error_msg, exc_info=True)
            raise DatabaseOperationError(error_msg, {
                "error_type": "transaction_execution_failed",
                "failed_operation_index": failed_operation_index,
                "total_operations": len(operations),
                "completed_operations": len(operation_results),
                "original_error": str(e),
                "exception_type": type(e).__name__
            }) from e

    async def execute_with_connection_retry(
        self, operation: callable, max_retries: int = 3, retry_delay: float = 1.0
    ) -> Any:
        """
        Execute a database operation with connection retry logic.
        
        This method handles transient connection failures by retrying the operation
        with exponential backoff and comprehensive error reporting.
        
        Args:
            operation: Async callable that takes a connection parameter
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            
        Returns:
            Result of the operation
            
        Raises:
            DatabaseOperationError: If operation fails after all retries
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                async with self.db.get_connection() as conn:
                    result = await operation(conn)
                    logger.debug(f"Operation completed successfully on attempt {attempt + 1}")
                    return result
                    
            except ConnectionError as e:
                last_error = e
                logger.warning(f"Connection failed on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying operation in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                    
            except (DatabaseOperationError, ValidationError) as e:
                # Don't retry for validation or known database errors
                logger.error(f"Operation failed with non-retryable error: {e}")
                raise e
                
            except Exception as e:
                last_error = e
                logger.warning(f"Operation failed on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
        
        # All retry attempts failed
        error_msg = f"Operation failed after {max_retries} attempts"
        logger.error(f"{error_msg}. Last error: {last_error}")
        raise DatabaseOperationError(error_msg, {
            "error_type": "operation_retry_exhausted",
            "max_retries": max_retries,
            "last_error": str(last_error),
            "exception_type": type(last_error).__name__ if last_error else "unknown"
        }) from last_error

    # Helper Methods for Type Inference and Conversion
    
    def _infer_postgres_type(self, series: pd.Series) -> str:
        """
        Infer PostgreSQL data type from pandas Series.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            PostgreSQL data type string
        """
        dtype = str(series.dtype)
        
        # Handle vector/array types
        if series.apply(lambda x: isinstance(x, (list, np.ndarray))).any():
            # Check if it's a vector type by examining first non-null value
            first_array = series.dropna().iloc[0] if not series.dropna().empty else None
            if first_array is not None and isinstance(first_array, (list, np.ndarray)):
                try:
                    # Try to determine if it's a vector
                    arr = np.array(first_array)
                    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                        return f"vector({len(arr)})"
                except:
                    pass
            return "jsonb"
        
        # Handle JSON/dict types
        if series.apply(lambda x: isinstance(x, dict)).any():
            return "jsonb"
        
        # Standard type mapping
        type_mapping = {
            "object": "text",
            "float64": "double precision",
            "float32": "double precision", 
            "int64": "bigint",
            "int32": "integer",
            "bool": "boolean",
            "datetime64": "timestamp",
            "timedelta64": "interval",
        }
        
        # For object columns, try to infer more specific types
        if dtype == "object" and not series.empty:
            non_null = series.dropna()
            if len(non_null) > 0:
                # Check if numeric
                try:
                    if non_null.apply(lambda x: str(x).replace(".", "").replace("-", "").isdigit()).all():
                        max_val = non_null.astype(float).max()
                        return "bigint" if max_val > 2**31 - 1 else "integer"
                except:
                    pass
        
        return type_mapping.get(dtype, "text")
    
    def _are_types_compatible(self, db_type: str, df_type: str) -> bool:
        """
        Check if database type and DataFrame type are compatible.
        
        Args:
            db_type: PostgreSQL data type
            df_type: Inferred DataFrame type
            
        Returns:
            True if types are compatible
        """
        # Exact match
        if db_type == df_type:
            return True
        
        # Compatible numeric types
        numeric_types = {"integer", "bigint", "double precision", "numeric", "real"}
        if db_type in numeric_types and df_type in numeric_types:
            return True
        
        # Text compatibility
        text_types = {"text", "varchar", "character varying", "char"}
        if db_type in text_types and df_type in text_types:
            return True
        
        # Vector type compatibility
        if db_type.startswith("vector(") and df_type.startswith("vector("):
            return db_type == df_type  # Dimensions must match exactly
        
        return False
    
    def _can_convert_type(self, db_type: str, df_type: str) -> bool:
        """
        Check if DataFrame type can be converted to database type.
        
        Args:
            db_type: PostgreSQL data type
            df_type: Inferred DataFrame type
            
        Returns:
            True if conversion is possible
        """
        # Most types can be converted to text
        if db_type in {"text", "varchar", "character varying"}:
            return True
        
        # Numeric conversions
        if db_type in {"integer", "bigint"} and df_type in {"double precision", "text"}:
            return True
        
        if db_type == "double precision" and df_type in {"integer", "bigint", "text"}:
            return True
        
        # Boolean conversions
        if db_type == "boolean" and df_type == "text":
            return True
        
        return False
    
    def _convert_dataframe_types(self, df: pd.DataFrame, table_info: TableInfo) -> pd.DataFrame:
        """
        Convert DataFrame column types to match PostgreSQL table schema.
        
        Performs efficient type conversion with validation and error reporting.
        Handles large datasets by processing columns individually and providing
        detailed error information for debugging.
        
        Args:
            df: DataFrame to convert
            table_info: Target table schema information
            
        Returns:
            DataFrame with converted types
            
        Raises:
            DataTypeError: If type conversion fails with detailed error information
        """
        df_converted = df.copy()
        conversion_errors = []
        
        for col in df.columns:
            if col in table_info.columns:
                col_info = table_info.columns[col]
                pg_type = col_info.data_type
                
                try:
                    if pg_type in ("integer", "bigint"):
                        # Handle integer conversion with validation
                        numeric_series = pd.to_numeric(df_converted[col], errors="coerce")
                        
                        # Check for conversion failures
                        failed_conversions = df_converted[col][numeric_series.isna() & df_converted[col].notna()]
                        if not failed_conversions.empty:
                            conversion_errors.append(
                                f"Column '{col}': Failed to convert {len(failed_conversions)} values to integer. "
                                f"Sample values: {failed_conversions.head(3).tolist()}"
                            )
                        
                        if col_info.nullable:
                            df_converted[col] = numeric_series.apply(
                                lambda x: int(x) if pd.notnull(x) else None
                            )
                        else:
                            # Fill NaN with 0 for non-nullable columns
                            df_converted[col] = numeric_series.fillna(0).astype(int)
                    
                    elif pg_type in ("double precision", "real", "numeric"):
                        # Handle float conversion with validation
                        numeric_series = pd.to_numeric(df_converted[col], errors="coerce")
                        
                        # Check for conversion failures
                        failed_conversions = df_converted[col][numeric_series.isna() & df_converted[col].notna()]
                        if not failed_conversions.empty:
                            conversion_errors.append(
                                f"Column '{col}': Failed to convert {len(failed_conversions)} values to float. "
                                f"Sample values: {failed_conversions.head(3).tolist()}"
                            )
                        
                        if col_info.nullable:
                            df_converted[col] = numeric_series.apply(
                                lambda x: float(x) if pd.notnull(x) else None
                            )
                        else:
                            df_converted[col] = numeric_series.fillna(0.0).astype(float)
                    
                    elif pg_type == "boolean":
                        # Handle boolean conversion with validation
                        def convert_to_bool(x):
                            if x is None:
                                return None if col_info.nullable else False
                            
                            # Handle pandas NA and numpy scalar NaN
                            try:
                                if pd.isna(x):
                                    return None if col_info.nullable else False
                            except (TypeError, ValueError):
                                # pd.isna() can fail on complex types, continue processing
                                pass
                            
                            if isinstance(x, bool):
                                return x
                            if isinstance(x, str):
                                x_lower = x.lower().strip()
                                if x_lower in ('true', 't', '1', 'yes', 'y'):
                                    return True
                                elif x_lower in ('false', 'f', '0', 'no', 'n'):
                                    return False
                                else:
                                    raise ValueError(f"Cannot convert '{x}' to boolean")
                            if isinstance(x, (int, float)):
                                return bool(x)
                            raise ValueError(f"Cannot convert {type(x)} to boolean")
                        
                        try:
                            df_converted[col] = df_converted[col].apply(convert_to_bool)
                        except Exception as e:
                            conversion_errors.append(f"Column '{col}': Boolean conversion failed - {e}")
                    
                    elif pg_type in ("text", "varchar", "char"):
                        # Handle text conversion
                        df_converted[col] = df_converted[col].apply(
                            lambda x: str(x) if pd.notnull(x) else (None if col_info.nullable else "")
                        )
                    
                    elif pg_type.startswith("vector("):
                        # Handle vector conversion with dimension validation
                        expected_dim = int(pg_type[7:-1])  # Extract dimension from vector(n)
                        
                        def convert_to_vector(x):
                            if x is None:
                                return None if col_info.nullable else [0.0] * expected_dim
                            
                            # Handle pandas NA and numpy scalar NaN
                            try:
                                if pd.isna(x):
                                    return None if col_info.nullable else [0.0] * expected_dim
                            except (TypeError, ValueError):
                                # pd.isna() can fail on complex types like arrays, continue processing
                                pass
                            
                            if isinstance(x, str):
                                try:
                                    # Try to parse as JSON array
                                    x = json.loads(x)
                                except:
                                    # Try to parse as comma-separated values
                                    x = [float(v.strip()) for v in x.split(',')]
                            
                            if isinstance(x, (list, np.ndarray)):
                                vector = [float(v) for v in x]
                                if len(vector) != expected_dim:
                                    raise ValueError(
                                        f"Vector dimension mismatch: expected {expected_dim}, got {len(vector)}"
                                    )
                                return vector
                            
                            raise ValueError(f"Cannot convert {type(x)} to vector")
                        
                        try:
                            df_converted[col] = df_converted[col].apply(convert_to_vector)
                        except Exception as e:
                            conversion_errors.append(f"Column '{col}': Vector conversion failed - {e}")
                    
                    elif pg_type == "jsonb":
                        # Handle JSONB conversion
                        def convert_to_jsonb(x):
                            if x is None:
                                return None if col_info.nullable else "{}"
                            
                            # Handle pandas NA and numpy scalar NaN
                            try:
                                if pd.isna(x):
                                    return None if col_info.nullable else "{}"
                            except (TypeError, ValueError):
                                # pd.isna() can fail on complex types like arrays, continue processing
                                pass
                            
                            if isinstance(x, str):
                                try:
                                    # Validate JSON string
                                    json.loads(x)
                                    return x
                                except:
                                    # If not valid JSON, treat as string value
                                    return json.dumps(x)
                            
                            if isinstance(x, (dict, list)):
                                return json.dumps(x)
                            
                            if isinstance(x, np.ndarray):
                                return json.dumps(x.tolist())
                            
                            # For other types, convert to JSON string
                            return json.dumps(x)
                        
                        df_converted[col] = df_converted[col].apply(convert_to_jsonb)
                    
                    elif pg_type.startswith("timestamp"):
                        # Handle timestamp conversion
                        try:
                            df_converted[col] = pd.to_datetime(df_converted[col], errors="coerce")
                        except Exception as e:
                            conversion_errors.append(f"Column '{col}': Timestamp conversion failed - {e}")
                    
                    else:
                        # For unknown types, convert to string
                        logger.warning(f"Unknown PostgreSQL type '{pg_type}' for column '{col}', converting to string")
                        df_converted[col] = df_converted[col].apply(
                            lambda x: str(x) if pd.notnull(x) else None
                        )
                
                except Exception as e:
                    conversion_errors.append(f"Column '{col}': Unexpected conversion error - {e}")
        
        # Raise error if any conversions failed
        if conversion_errors:
            error_message = "Data type conversion failed:\n" + "\n".join(conversion_errors)
            logger.error(error_message)
            raise DataTypeError(error_message)
        
        return df_converted
    
    async def _bulk_insert_data(
        self, conn, table_name: str, df: pd.DataFrame, batch_size: int
    ) -> int:
        """
        Perform bulk insert of DataFrame data using batched operations.
        
        This is an internal method used by replace_table_data for transaction-safe
        bulk insertion with proper data type conversion and error handling.
        
        Args:
            conn: Database connection (with active transaction)
            table_name: Name of the table to insert into
            df: DataFrame containing the data (already type-converted)
            batch_size: Number of rows per batch
            
        Returns:
            Number of rows inserted
            
        Raises:
            DatabaseOperationError: If insertion fails
        """
        if df.empty:
            return 0
        
        pandas_columns = list(df.columns)
        sql_columns = [f'"{col}"' for col in pandas_columns]
        
        query = f"INSERT INTO {table_name} ({', '.join(sql_columns)}) VALUES ({', '.join(f'${i+1}' for i in range(len(sql_columns)))})"
        
        rows_inserted = 0
        
        try:
            # Process in batches for memory efficiency
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]
                
                # Convert batch to properly formatted values
                values = []
                for row in batch_df.itertuples(index=False, name=None):
                    converted_row = tuple(
                        self._convert_value_for_postgres(val) for val in row
                    )
                    values.append(converted_row)
                
                # Execute batch insert
                await conn.executemany(query, values)
                rows_inserted += len(values)
                
                # Log progress for large datasets
                if len(df) > 10000 and (start_idx + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Inserted {rows_inserted}/{len(df)} rows into {table_name}")
        
        except Exception as e:
            logger.error(f"Error in bulk insert batch operation: {e}")
            raise DatabaseOperationError(f"Batch insert failed: {e}")
        
        return rows_inserted

    # Placeholder methods for interface compliance - these will be implemented in later tasks
    
    async def update_table_schema(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        allow_column_drops: bool = False
    ) -> Dict[str, str]:
        """Placeholder - will be implemented in later tasks."""
        raise NotImplementedError("Will be implemented in task 10")

    async def bulk_insert_data(
        self, 
        table_name: str, 
        df: pd.DataFrame,
        batch_size: int = 1000,
        on_conflict: str = "error"
    ) -> int:
        """
        Perform bulk insert operations using PostgreSQL COPY or batch inserts.
        
        This method provides optimized bulk data insertion with configurable batch sizes
        and conflict resolution strategies. Uses PostgreSQL COPY for maximum performance
        when possible, falling back to batch inserts for complex data types.
        
        Args:
            table_name: Name of the table to insert data into
            df: DataFrame containing the data to insert
            batch_size: Number of rows to process in each batch (default: 1000)
            on_conflict: How to handle conflicts - "error", "ignore", or "update"
            
        Returns:
            Number of rows successfully inserted
            
        Raises:
            DatabaseOperationError: If bulk insert operation fails
            ValidationError: If data validation fails
        """
        if df.empty:
            logger.info(f"No data to insert into table {table_name}")
            return 0
        
        try:
            # Validate table exists
            if not await self.table_exists(table_name):
                raise ValidationError(f"Table {table_name} does not exist")
            
            # Get table schema for validation and type conversion
            table_info = await self.get_table_info(table_name)
            
            # Validate DataFrame columns exist in table
            df_columns = set(df.columns)
            db_columns = set(table_info.columns.keys())
            
            missing_cols = df_columns - db_columns
            if missing_cols:
                raise ValidationError(f"DataFrame contains columns not in table: {missing_cols}")
            
            # Convert DataFrame to match PostgreSQL types
            df_converted = self._convert_dataframe_types(df, table_info)
            
            # Validate constraints before insertion
            constraint_violations = await self.validate_constraints(table_name, df_converted)
            if constraint_violations:
                violation_summary = "; ".join([
                    f"{v.constraint_type} violation in {v.column_name}: {v.violation_type}"
                    for v in constraint_violations[:3]
                ])
                raise ValidationError(f"Constraint violations found: {violation_summary}")
            
            rows_inserted = 0
            
            # Choose insertion method based on data complexity and conflict handling
            if on_conflict == "error" and self._can_use_copy_method(df_converted, table_info):
                # Use PostgreSQL COPY for maximum performance
                rows_inserted = await self._bulk_insert_with_copy(
                    table_name, df_converted, batch_size
                )
            else:
                # Use batch INSERT statements for complex scenarios
                rows_inserted = await self._bulk_insert_with_batches(
                    table_name, df_converted, batch_size, on_conflict
                )
            
            logger.info(f"Bulk inserted {rows_inserted} rows into table {table_name}")
            return rows_inserted
            
        except Exception as e:
            logger.error(f"Error in bulk insert operation: {e}")
            raise DatabaseOperationError(f"Bulk insert failed: {e}")

    async def _bulk_insert_with_copy(
        self, 
        table_name: str, 
        df: pd.DataFrame, 
        batch_size: int
    ) -> int:
        """
        Use PostgreSQL COPY command for high-performance bulk insert.
        
        Args:
            table_name: Name of the table to insert into
            df: DataFrame containing the data
            batch_size: Number of rows per batch
            
        Returns:
            Number of rows inserted
        """
        import io
        
        rows_inserted = 0
        pandas_columns = list(df.columns)
        sql_columns = [f'"{col}"' for col in pandas_columns]
        
        async with self.db.get_connection() as conn:
            async with conn.transaction():
                # Process in batches to manage memory usage
                for start_idx in range(0, len(df), batch_size):
                    end_idx = min(start_idx + batch_size, len(df))
                    batch_df = df.iloc[start_idx:end_idx]
                    
                    # Convert DataFrame to CSV format in memory
                    csv_buffer = io.StringIO()
                    batch_df.to_csv(
                        csv_buffer, 
                        index=False, 
                        header=False, 
                        na_rep='\\N',  # PostgreSQL NULL representation
                        sep='\t'  # Use tab separator for COPY
                    )
                    csv_buffer.seek(0)
                    
                    # Use COPY command for bulk insert
                    copy_sql = f"COPY {table_name} ({', '.join(sql_columns)}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N')"
                    
                    await conn.copy_from_table(
                        table_name,
                        source=csv_buffer,
                        columns=pandas_columns,
                        format='csv',
                        delimiter='\t',
                        null='\\N'
                    )
                    
                    rows_inserted += len(batch_df)
        
        return rows_inserted

    async def _bulk_insert_with_batches(
        self, 
        table_name: str, 
        df: pd.DataFrame, 
        batch_size: int,
        on_conflict: str
    ) -> int:
        """
        Use batch INSERT statements for bulk insert with conflict handling.
        
        Args:
            table_name: Name of the table to insert into
            df: DataFrame containing the data
            batch_size: Number of rows per batch
            on_conflict: How to handle conflicts - "error", "ignore", or "update"
            
        Returns:
            Number of rows inserted
        """
        rows_inserted = 0
        pandas_columns = list(df.columns)
        sql_columns = [f'"{col}"' for col in pandas_columns]
        
        # Build base INSERT query
        placeholders = ', '.join(f'${i+1}' for i in range(len(sql_columns)))
        base_query = f"INSERT INTO {table_name} ({', '.join(sql_columns)}) VALUES ({placeholders})"
        
        # Add conflict resolution clause
        if on_conflict == "ignore":
            query = f"{base_query} ON CONFLICT DO NOTHING"
        elif on_conflict == "update":
            # For update, we need to know the primary key columns
            table_info = await self.get_table_info(table_name)
            if table_info.primary_keys:
                update_clauses = [
                    f'"{col}" = EXCLUDED."{col}"' 
                    for col in pandas_columns 
                    if col not in table_info.primary_keys
                ]
                if update_clauses:
                    pk_columns = ', '.join(f'"{pk}"' for pk in table_info.primary_keys)
                    update_set = ', '.join(update_clauses)
                    query = f"{base_query} ON CONFLICT ({pk_columns}) DO UPDATE SET {update_set}"
                else:
                    query = f"{base_query} ON CONFLICT DO NOTHING"
            else:
                logger.warning(f"No primary key found for table {table_name}, using DO NOTHING for conflicts")
                query = f"{base_query} ON CONFLICT DO NOTHING"
        else:  # on_conflict == "error"
            query = base_query
        
        async with self.db.get_connection() as conn:
            async with conn.transaction():
                # Process in batches
                for start_idx in range(0, len(df), batch_size):
                    end_idx = min(start_idx + batch_size, len(df))
                    batch_df = df.iloc[start_idx:end_idx]
                    
                    # Convert batch to list of tuples
                    values = [
                        tuple(self._convert_value_for_postgres(val) for val in row)
                        for row in batch_df.itertuples(index=False, name=None)
                    ]
                    
                    # Execute batch insert
                    await conn.executemany(query, values)
                    rows_inserted += len(values)
        
        return rows_inserted

    def _can_use_copy_method(self, df: pd.DataFrame, table_info: TableInfo) -> bool:
        """
        Determine if COPY method can be used for bulk insert.
        
        COPY method is faster but has limitations with complex data types
        and doesn't support conflict resolution.
        
        Args:
            df: DataFrame to analyze
            table_info: Table schema information
            
        Returns:
            True if COPY method can be used, False otherwise
        """
        # Check for complex data types that might not work well with COPY
        for col in df.columns:
            if col in table_info.columns:
                col_type = table_info.columns[col].data_type
                
                # Vector and JSONB types might need special handling
                if col_type.startswith('vector(') or col_type == 'jsonb':
                    # Check if column contains complex nested structures
                    sample_values = df[col].dropna().head(5)
                    for val in sample_values:
                        if isinstance(val, (dict, list)) and len(str(val)) > 1000:
                            return False  # Complex nested data, use batch method
        
        return True

    def _convert_value_for_postgres(self, value):
        """
        Convert a single value to PostgreSQL-compatible format.
        
        Args:
            value: Value to convert
            
        Returns:
            PostgreSQL-compatible value
        """
        # Handle None and scalar NaN values
        if value is None:
            return None
        
        # Handle pandas NA and numpy scalar NaN
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            # pd.isna() can fail on complex types like arrays, continue processing
            pass
        
        if isinstance(value, (list, np.ndarray)):
            # Convert arrays to PostgreSQL array format
            if isinstance(value, np.ndarray):
                value = value.tolist()
            return value
        
        if isinstance(value, dict):
            # Convert dict to JSON string
            return json.dumps(value)
        
        if isinstance(value, (np.integer, np.floating)):
            # Convert numpy types to Python types
            return value.item()
        
        return value

    async def get_postgresql_version(self) -> str:
        """Get PostgreSQL server version for compatibility checks."""
        try:
            async with self.db.get_connection() as conn:
                version = await conn.fetchval("SELECT version()")
                return version
        except PostgresError as e:
            logger.error(f"Error getting PostgreSQL version: {e}")
            raise DatabaseOperationError(f"Failed to get PostgreSQL version: {e}")

    async def analyze_table_statistics(self, table_name: str) -> Dict[str, any]:
        """Get PostgreSQL table statistics for performance optimization."""
        try:
            async with self.db.get_connection() as conn:
                # Get basic table statistics
                stats_query = """
                SELECT 
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    most_common_vals,
                    most_common_freqs,
                    histogram_bounds
                FROM pg_stats 
                WHERE tablename = $1
                """
                stats = await conn.fetch(stats_query, table_name)
                
                # Get table size information
                size_query = """
                SELECT 
                    pg_size_pretty(pg_total_relation_size($1)) as total_size,
                    pg_size_pretty(pg_relation_size($1)) as table_size,
                    (SELECT count(*) FROM """ + table_name + """) as row_count
                """
                size_info = await conn.fetchrow(size_query, table_name)
                
                return {
                    'statistics': [dict(row) for row in stats],
                    'size_info': dict(size_info) if size_info else {},
                    'table_name': table_name
                }
        except PostgresError as e:
            logger.error(f"Error getting table statistics: {e}")
            raise DatabaseOperationError(f"Failed to get table statistics: {e}")

    async def optimize_table_performance(self, table_name: str) -> None:
        """Run PostgreSQL-specific optimizations on the table."""
        try:
            async with self.db.get_connection() as conn:
                # Run VACUUM ANALYZE for statistics update and cleanup
                await conn.execute(f"VACUUM ANALYZE {table_name}")
                logger.info(f"Optimized table {table_name} with VACUUM ANALYZE")
        except PostgresError as e:
            logger.error(f"Error optimizing table {table_name}: {e}")
            raise DatabaseOperationError(f"Failed to optimize table: {e}")

    async def detect_vector_columns(self, table_name: str) -> List[Tuple[str, int]]:
        """Detect vector columns and their dimensions in the table."""
        try:
            table_info = await self.get_table_info(table_name)
            vector_columns = []
            
            for col_name, col_info in table_info.columns.items():
                if col_info.data_type.startswith('vector('):
                    # Extract dimension from vector(n) type
                    dimension_str = col_info.data_type[7:-1]  # Remove 'vector(' and ')'
                    try:
                        dimension = int(dimension_str)
                        vector_columns.append((col_name, dimension))
                    except ValueError:
                        logger.warning(f"Could not parse vector dimension for column {col_name}")
            
            return vector_columns
        except Exception as e:
            logger.error(f"Error detecting vector columns: {e}")
            raise DatabaseOperationError(f"Failed to detect vector columns: {e}")

    async def validate_vector_dimensions(
        self, 
        table_name: str, 
        df: pd.DataFrame
    ) -> List[str]:
        """Validate that vector columns in DataFrame match table vector dimensions."""
        try:
            vector_columns = await self.detect_vector_columns(table_name)
            errors = []
            
            for col_name, expected_dim in vector_columns:
                if col_name in df.columns:
                    # Check vector dimensions in DataFrame
                    sample_vectors = df[col_name].dropna().head(10)
                    for idx, vector in enumerate(sample_vectors):
                        if isinstance(vector, (list, np.ndarray)):
                            actual_dim = len(vector)
                            if actual_dim != expected_dim:
                                errors.append(
                                    f"Column '{col_name}' row {idx}: expected dimension {expected_dim}, "
                                    f"got {actual_dim}"
                                )
                        else:
                            errors.append(
                                f"Column '{col_name}' row {idx}: expected vector, got {type(vector)}"
                            )
            
            return errors
        except Exception as e:
            logger.error(f"Error validating vector dimensions: {e}")
            raise DatabaseOperationError(f"Failed to validate vector dimensions: {e}")

    # Methods from DataRepositoryInterface that need to be implemented for compatibility
    
    async def get_table_schema(self, table_name: str):
        """Get table schema in the format expected by existing code."""
        table_info = await self.get_table_info(table_name)
        columns = {name: col.data_type for name, col in table_info.columns.items()}
        nullables = {name: col.nullable for name, col in table_info.columns.items()}
        
        # Import here to avoid circular imports
        from dataload.domain.entities import TableSchema
        return TableSchema(columns=columns, nullables=nullables)

    async def insert_data(self, table_name: str, df: pd.DataFrame, pk_columns: List[str]):
        """Insert data using existing interface - delegates to bulk insert."""
        return await self._bulk_insert_data(None, table_name, df, 1000)

    async def update_data(self, table_name: str, df: pd.DataFrame, pk_columns: List[str]):
        """Update data using existing interface - not implemented for DataMove."""
        raise NotImplementedError("Update operations not supported in DataMove use case")

    async def set_inactive(self, table_name: str, pks: List[tuple], pk_columns: List[str]):
        """Set inactive using existing interface - not implemented for DataMove."""
        raise NotImplementedError("Set inactive operations not supported in DataMove use case")

    async def get_active_data(self, table_name: str, columns: List[str]) -> pd.DataFrame:
        """Get active data using existing interface - returns all data for DataMove."""
        sql_columns = [f'"{col}"' for col in columns]
        query = f"SELECT {', '.join(sql_columns)} FROM {table_name}"
        
        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query)
        
        return pd.DataFrame(rows, columns=columns)

    async def get_embed_columns_names(self, table_name: str) -> List[str]:
        """Get embed column names - not applicable for DataMove."""
        return []

    async def get_data_columns(self, table_name: str) -> List[str]:
        """Get data columns excluding system columns."""
        table_info = await self.get_table_info(table_name)
        # Return all columns for DataMove (no system columns to exclude)
        return list(table_info.columns.keys())

    async def create_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        pk_columns: List[str],
        embed_type: str = "combined",
        embed_columns_names: List[str] = [],
    ) -> Dict[str, str]:
        """Create table using existing interface - delegates to create_table_from_dataframe."""
        return await self.create_table_from_dataframe(table_name, df, pk_columns)

    async def add_column(self, table_name: str, column_name: str, column_type: str):
        """Add column to existing table."""
        query = f'ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS "{column_name}" {column_type}'
        try:
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    await conn.execute(query)
        except PostgresError as e:
            logger.error(f"Error adding column {column_name} to {table_name}: {e}")
            raise DatabaseOperationError(f"Failed to add column: {e}")

    # Performance and Memory Management Helper Methods
    
    def _calculate_optimal_batch_size(self, df: pd.DataFrame, target_memory_mb: int = 100) -> int:
        """
        Calculate optimal batch size based on DataFrame memory usage.
        
        Args:
            df: DataFrame to analyze
            target_memory_mb: Target memory usage per batch in MB
            
        Returns:
            Optimal batch size for processing
        """
        if df.empty:
            return 1000
        
        # Calculate memory usage per row
        memory_per_row = df.memory_usage(deep=True).sum() / len(df)
        target_memory_bytes = target_memory_mb * 1024 * 1024
        
        # Calculate optimal batch size
        optimal_batch_size = max(1, int(target_memory_bytes / memory_per_row))
        
        # Cap at reasonable limits
        return min(max(optimal_batch_size, 100), 10000)

    async def _monitor_bulk_operation_progress(
        self, 
        operation_name: str, 
        total_rows: int, 
        processed_rows: int, 
        start_time: float
    ) -> None:
        """
        Monitor and log progress of bulk operations.
        
        Args:
            operation_name: Name of the operation being monitored
            total_rows: Total number of rows to process
            processed_rows: Number of rows processed so far
            start_time: Start time of the operation
        """
        if total_rows > 1000 and processed_rows % 5000 == 0:
            elapsed_time = time.time() - start_time
            progress_percent = (processed_rows / total_rows) * 100
            rows_per_second = processed_rows / elapsed_time if elapsed_time > 0 else 0
            
            estimated_total_time = (total_rows / rows_per_second) if rows_per_second > 0 else 0
            estimated_remaining_time = estimated_total_time - elapsed_time
            
            logger.info(
                f"{operation_name} progress: {processed_rows:,}/{total_rows:,} rows "
                f"({progress_percent:.1f}%) - {rows_per_second:.0f} rows/sec - "
                f"ETA: {estimated_remaining_time:.0f}s"
            )

    def _validate_memory_usage(self, df: pd.DataFrame, max_memory_mb: int = 500) -> None:
        """
        Validate that DataFrame memory usage is within acceptable limits.
        
        Args:
            df: DataFrame to validate
            max_memory_mb: Maximum allowed memory usage in MB
            
        Raises:
            ValidationError: If memory usage exceeds limits
        """
        memory_usage_bytes = df.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)
        
        if memory_usage_mb > max_memory_mb:
            raise ValidationError(
                f"DataFrame memory usage ({memory_usage_mb:.1f} MB) exceeds limit ({max_memory_mb} MB). "
                f"Consider processing in smaller chunks or increasing memory limits."
            )

    async def _optimize_bulk_insert_strategy(
        self, 
        table_name: str, 
        df: pd.DataFrame, 
        batch_size: int
    ) -> Tuple[str, int]:
        """
        Determine the optimal bulk insert strategy and batch size.
        
        Args:
            table_name: Name of the target table
            df: DataFrame containing the data
            batch_size: Requested batch size
            
        Returns:
            Tuple of (strategy_name, optimized_batch_size)
        """
        # Analyze data characteristics
        has_complex_types = any(
            col_type.startswith('vector(') or col_type == 'jsonb'
            for col_type in [self._infer_postgres_type(df[col]) for col in df.columns]
        )
        
        # Get table statistics if available
        try:
            table_stats = await self.analyze_table_statistics(table_name)
            table_size = table_stats.get('size_info', {}).get('row_count', 0)
        except:
            table_size = 0
        
        # Determine strategy
        if has_complex_types or table_size > 100000:
            strategy = "batch_insert"
            # Use smaller batches for complex data
            optimized_batch_size = min(batch_size, 500)
        else:
            strategy = "copy_insert"
            # Use larger batches for simple data
            optimized_batch_size = min(batch_size * 2, 5000)
        
        # Adjust based on DataFrame size
        if len(df) < 1000:
            optimized_batch_size = len(df)
        
        return strategy, optimized_batch_size

    # Vector Column Handling Implementation
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
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
        try:
            # Query to detect vector columns and extract dimensions
            query = """
            SELECT 
                column_name,
                data_type,
                udt_name
            FROM information_schema.columns
            WHERE table_name = $1 
            AND (data_type = 'USER-DEFINED' AND udt_name = 'vector')
            """
            
            async with self.db.get_connection() as conn:
                rows = await conn.fetch(query, table_name)
                
                vector_columns = []
                
                for row in rows:
                    column_name = row['column_name']
                    
                    # Get the actual vector type definition to extract dimension
                    type_query = """
                    SELECT 
                        pg_get_expr(adbin, adrelid) as column_default,
                        format_type(atttypid, atttypmod) as formatted_type
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    WHERE c.relname = $1 AND a.attname = $2 AND NOT a.attisdropped
                    """
                    
                    type_row = await conn.fetchrow(type_query, table_name, column_name)
                    
                    if type_row and type_row['formatted_type']:
                        formatted_type = type_row['formatted_type']
                        # Extract dimension from vector(n) format
                        dimension = self._extract_vector_dimension(formatted_type)
                        if dimension:
                            vector_columns.append((column_name, dimension))
                            logger.debug(f"Detected vector column {column_name} with dimension {dimension}")
                
                return vector_columns
                
        except PostgresError as e:
            logger.error(f"Error detecting vector columns in {table_name}: {e}")
            raise DatabaseOperationError(f"Failed to detect vector columns: {e}")

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
        try:
            vector_columns = await self.detect_vector_columns(table_name)
            errors = []
            
            for column_name, expected_dimension in vector_columns:
                if column_name in df.columns:
                    # Validate vector dimensions in DataFrame
                    validation_errors = self._validate_dataframe_vector_column(
                        df, column_name, expected_dimension
                    )
                    errors.extend(validation_errors)
            
            return errors
            
        except Exception as e:
            logger.error(f"Error validating vector dimensions: {e}")
            raise DatabaseOperationError(f"Vector dimension validation failed: {e}")

    def _extract_vector_dimension(self, formatted_type: str) -> Optional[int]:
        """
        Extract dimension from PostgreSQL vector type string.
        
        Args:
            formatted_type: PostgreSQL formatted type string (e.g., 'vector(1536)')
            
        Returns:
            Vector dimension as integer, or None if not a vector type
        """
        import re
        
        # Match vector(n) pattern
        match = re.match(r'vector\((\d+)\)', formatted_type)
        if match:
            return int(match.group(1))
        
        # Handle case where dimension might not be specified
        if formatted_type == 'vector':
            logger.warning("Vector column found without explicit dimension")
            return None
        
        return None

    def _validate_dataframe_vector_column(
        self, 
        df: pd.DataFrame, 
        column_name: str, 
        expected_dimension: int
    ) -> List[str]:
        """
        Validate vector data in a DataFrame column.
        
        Args:
            df: DataFrame containing the vector column
            column_name: Name of the vector column
            expected_dimension: Expected vector dimension
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            # Check if column exists
            if column_name not in df.columns:
                return [f"Vector column '{column_name}' not found in DataFrame"]
            
            vector_series = df[column_name].dropna()
            
            if len(vector_series) == 0:
                return [f"Vector column '{column_name}' contains no non-null values"]
            
            # Validate each vector in the column
            for idx, vector_value in vector_series.items():
                try:
                    # Handle different vector formats
                    if isinstance(vector_value, str):
                        # Parse string representation of vector
                        vector_array = self._parse_vector_string(vector_value)
                    elif isinstance(vector_value, (list, np.ndarray)):
                        vector_array = np.array(vector_value, dtype=float)
                    else:
                        errors.append(
                            f"Row {idx}: Invalid vector format in column '{column_name}'. "
                            f"Expected list, array, or string, got {type(vector_value)}"
                        )
                        continue
                    
                    # Validate dimension
                    actual_dimension = len(vector_array)
                    if actual_dimension != expected_dimension:
                        errors.append(
                            f"Row {idx}: Vector dimension mismatch in column '{column_name}'. "
                            f"Expected {expected_dimension}, got {actual_dimension}"
                        )
                    
                    # Validate that all elements are numeric
                    if not np.issubdtype(vector_array.dtype, np.number):
                        errors.append(
                            f"Row {idx}: Vector contains non-numeric values in column '{column_name}'"
                        )
                    
                    # Check for invalid values (NaN, inf)
                    if np.any(np.isnan(vector_array)) or np.any(np.isinf(vector_array)):
                        errors.append(
                            f"Row {idx}: Vector contains NaN or infinite values in column '{column_name}'"
                        )
                        
                except Exception as e:
                    errors.append(
                        f"Row {idx}: Error validating vector in column '{column_name}': {e}"
                    )
            
            # Limit error reporting to avoid overwhelming output
            if len(errors) > 10:
                errors = errors[:10] + [f"... and {len(errors) - 10} more validation errors"]
            
            return errors
            
        except Exception as e:
            return [f"Error validating vector column '{column_name}': {e}"]

    def _parse_vector_string(self, vector_str: str) -> np.ndarray:
        """
        Parse a string representation of a vector into a numpy array.
        
        Args:
            vector_str: String representation of vector (e.g., '[1.0, 2.0, 3.0]' or '1.0,2.0,3.0')
            
        Returns:
            Numpy array of float values
            
        Raises:
            ValueError: If string cannot be parsed as vector
        """
        try:
            # Remove brackets and split by comma
            cleaned = vector_str.strip('[](){} ')
            if not cleaned:
                raise ValueError("Empty vector string")
            
            # Split by comma and convert to float
            values = [float(x.strip()) for x in cleaned.split(',')]
            return np.array(values, dtype=float)
            
        except Exception as e:
            raise ValueError(f"Cannot parse vector string '{vector_str}': {e}")



    async def create_vector_indexes(
        self, 
        table_name: str, 
        vector_columns: List[Tuple[str, int]],
        index_type: str = "ivfflat"
    ) -> None:
        """
        Create appropriate indexes for vector columns.
        
        Args:
            table_name: Name of the table
            vector_columns: List of (column_name, dimension) tuples
            index_type: Type of index to create ('ivfflat' or 'hnsw')
            
        Raises:
            DatabaseOperationError: If index creation fails
        """
        try:
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    for column_name, dimension in vector_columns:
                        # Choose index parameters based on dimension and type
                        if index_type == "ivfflat":
                            # IVFFlat is better for high dimensions (>2000)
                            lists_param = max(10, min(1000, dimension // 10))
                            index_query = f"""
                                CREATE INDEX IF NOT EXISTS idx_{table_name}_{column_name}_ivfflat
                                ON {table_name} 
                                USING ivfflat ("{column_name}" vector_cosine_ops) 
                                WITH (lists = {lists_param})
                            """
                        elif index_type == "hnsw":
                            # HNSW is limited to 2000 dimensions but faster for lower dimensions
                            if dimension > 2000:
                                logger.warning(
                                    f"HNSW index not supported for dimension {dimension} > 2000, "
                                    f"using IVFFlat instead"
                                )
                                lists_param = max(10, min(1000, dimension // 10))
                                index_query = f"""
                                    CREATE INDEX IF NOT EXISTS idx_{table_name}_{column_name}_ivfflat
                                    ON {table_name} 
                                    USING ivfflat ("{column_name}" vector_cosine_ops) 
                                    WITH (lists = {lists_param})
                                """
                            else:
                                index_query = f"""
                                    CREATE INDEX IF NOT EXISTS idx_{table_name}_{column_name}_hnsw
                                    ON {table_name} 
                                    USING hnsw ("{column_name}" vector_cosine_ops) 
                                    WITH (m = 16, ef_construction = 64)
                                """
                        else:
                            raise ValueError(f"Unsupported index type: {index_type}")
                        
                        await conn.execute(index_query)
                        logger.info(f"Created {index_type} index for vector column {column_name}")
                        
        except PostgresError as e:
            logger.error(f"Error creating vector indexes: {e}")
            raise DatabaseOperationError(f"Failed to create vector indexes: {e}")

    def _convert_vector_data(
        self, 
        df: pd.DataFrame, 
        vector_columns: List[Tuple[str, int]]
    ) -> pd.DataFrame:
        """
        Convert vector data in DataFrame to PostgreSQL-compatible format.
        
        Args:
            df: DataFrame containing vector data
            vector_columns: List of (column_name, dimension) tuples
            
        Returns:
            DataFrame with properly formatted vector data
        """
        df_converted = df.copy()
        
        for column_name, expected_dimension in vector_columns:
            if column_name in df_converted.columns:
                logger.debug(f"Converting vector column {column_name}")
                
                def convert_vector_value(value):
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        return None
                    
                    try:
                        if isinstance(value, str):
                            vector_array = self._parse_vector_string(value)
                        elif isinstance(value, (list, np.ndarray)):
                            vector_array = np.array(value, dtype=float)
                        else:
                            raise ValueError(f"Unsupported vector format: {type(value)}")
                        
                        # Validate dimension
                        if len(vector_array) != expected_dimension:
                            raise ValueError(
                                f"Dimension mismatch: expected {expected_dimension}, "
                                f"got {len(vector_array)}"
                            )
                        
                        # Convert to list of floats for PostgreSQL
                        return vector_array.tolist()
                        
                    except Exception as e:
                        logger.error(f"Error converting vector value: {e}")
                        raise DataTypeError(f"Vector conversion failed: {e}")
                
                df_converted[column_name] = df_converted[column_name].apply(convert_vector_value)
        
        return df_converted

    def _infer_postgres_type(self, series: pd.Series) -> str:
        """
        Infer PostgreSQL data type from pandas Series, with vector support.
        
        Args:
            series: Pandas series to analyze
            
        Returns:
            PostgreSQL type string
        """
        # Check if this might be a vector column first
        non_null_series = series.dropna()
        
        if len(non_null_series) > 0:
            # Check if all non-null values are list-like (potential vectors)
            sample_values = non_null_series.head(10)
            vector_like_count = 0
            vector_dimensions = set()
            
            for value in sample_values:
                if isinstance(value, (list, np.ndarray)):
                    vector_like_count += 1
                    vector_dimensions.add(len(value))
                elif isinstance(value, str):
                    try:
                        parsed_vector = self._parse_vector_string(value)
                        vector_like_count += 1
                        vector_dimensions.add(len(parsed_vector))
                    except ValueError:
                        pass
            
            # If most values are vector-like and have consistent dimensions
            if vector_like_count >= len(sample_values) * 0.8 and len(vector_dimensions) == 1:
                dimension = list(vector_dimensions)[0]
                logger.info(f"Detected vector column with dimension {dimension}")
                return f"vector({dimension})"
        
        # Standard type mapping from existing repository
        pd_to_pg = {
            "object": "text",
            "float64": "double precision",
            "float32": "double precision", 
            "int64": "bigint",
            "int32": "integer",
            "bool": "boolean",
            "datetime64": "timestamp",
            "timedelta64": "interval",
        }
        
        dtype = str(series.dtype)
        
        # Check if column contains lists/dicts/arrays (potential jsonb)
        if series.apply(lambda x: isinstance(x, (list, dict, np.ndarray))).any():
            return "jsonb"
        
        # For object columns, do additional analysis
        non_null = series.dropna()
        if dtype == "object" and len(non_null) > 0:
            try:
                # Check if numeric-like strings
                if (
                    non_null.apply(
                        lambda x: isinstance(x, (str, float))
                        and str(x).replace(".", "").isdigit()
                    ).all()
                    and not non_null.apply(
                        lambda x: isinstance(x, str) and len(x) > 20
                    ).any()
                ):
                    max_val = non_null.astype(float).max()
                    return "bigint" if max_val > 2**31 - 1 else "integer"
                else:
                    return "text"
            except (ValueError, TypeError):
                return "text"
        
        return pd_to_pg.get(dtype, "text")

    def _are_types_compatible(self, db_type: str, df_type: str) -> bool:
        """
        Check if database type and DataFrame inferred type are compatible.
        
        Args:
            db_type: PostgreSQL data type from database
            df_type: Inferred PostgreSQL type from DataFrame
            
        Returns:
            True if types are compatible, False otherwise
        """
        # Exact match
        if db_type == df_type:
            return True
        
        # Compatible numeric types
        numeric_types = {'integer', 'bigint', 'double precision', 'real', 'numeric'}
        if db_type in numeric_types and df_type in numeric_types:
            return True
        
        # Text types are generally compatible
        text_types = {'text', 'varchar', 'character varying'}
        if db_type in text_types and df_type in text_types:
            return True
        
        # Vector types - check dimensions
        if db_type.startswith('vector(') and df_type.startswith('vector('):
            db_dim = self._extract_vector_dimension(db_type)
            df_dim = self._extract_vector_dimension(df_type)
            return db_dim == df_dim
        
        # JSONB compatibility
        if db_type == 'jsonb' and df_type in ['jsonb', 'json']:
            return True
        
        return False

    def _can_convert_type(self, db_type: str, df_type: str) -> bool:
        """
        Check if DataFrame type can be converted to database type.
        
        Args:
            db_type: PostgreSQL data type from database
            df_type: Inferred PostgreSQL type from DataFrame
            
        Returns:
            True if conversion is possible, False otherwise
        """
        # If types are already compatible, no conversion needed
        if self._are_types_compatible(db_type, df_type):
            return True
        
        # Text can usually be converted to other types
        if df_type == 'text':
            return True
        
        # Numeric conversions
        if db_type in {'integer', 'bigint'} and df_type in {'double precision', 'real'}:
            return True
        
        if db_type in {'double precision', 'real'} and df_type in {'integer', 'bigint'}:
            return True
        
        # Vector dimension mismatches cannot be easily converted
        if db_type.startswith('vector(') and df_type.startswith('vector('):
            return False
        
        return False

    def _convert_dataframe_types(self, df: pd.DataFrame, table_info: TableInfo) -> pd.DataFrame:
        """
        Convert DataFrame column types to match PostgreSQL table schema.
        
        Args:
            df: DataFrame to convert
            table_info: Table schema information
            
        Returns:
            DataFrame with converted types
        """
        df_converted = df.copy()
        
        for col_name, col_info in table_info.columns.items():
            if col_name in df_converted.columns:
                pg_type = col_info.data_type
                nullable = col_info.nullable
                
                try:
                    if pg_type in ('integer', 'bigint'):
                        df_converted[col_name] = pd.to_numeric(df_converted[col_name], errors='coerce')
                        if not nullable:
                            df_converted[col_name] = df_converted[col_name].fillna(0)
                        df_converted[col_name] = df_converted[col_name].apply(
                            lambda x: int(x) if pd.notnull(x) else None
                        )
                    
                    elif pg_type in ('double precision', 'real', 'numeric'):
                        df_converted[col_name] = pd.to_numeric(df_converted[col_name], errors='coerce')
                        if not nullable:
                            df_converted[col_name] = df_converted[col_name].fillna(0.0)
                        df_converted[col_name] = df_converted[col_name].apply(
                            lambda x: float(x) if pd.notnull(x) else None
                        )
                    
                    elif pg_type == 'boolean':
                        df_converted[col_name] = df_converted[col_name].apply(
                            lambda x: bool(x) if pd.notnull(x) else None
                        )
                    
                    elif pg_type == 'timestamp':
                        df_converted[col_name] = pd.to_datetime(df_converted[col_name], errors='coerce')
                        df_converted[col_name] = df_converted[col_name].apply(
                            lambda x: x if pd.notnull(x) else None
                        )
                    
                    elif pg_type == 'jsonb':
                        df_converted[col_name] = df_converted[col_name].apply(
                            lambda x: (
                                json.dumps(x.tolist() if isinstance(x, np.ndarray) else x)
                                if isinstance(x, (list, dict, np.ndarray))
                                else (json.dumps(x) if pd.notnull(x) else None)
                            )
                        )
                    
                    elif pg_type.startswith('vector('):
                        # Vector conversion is handled separately in _convert_vector_data
                        pass
                    
                    else:  # Default to text
                        df_converted[col_name] = df_converted[col_name].apply(
                            lambda x: str(x) if pd.notnull(x) else None
                        )
                
                except Exception as e:
                    logger.warning(f"Error converting column {col_name} to {pg_type}: {e}")
                    # Fall back to string conversion
                    df_converted[col_name] = df_converted[col_name].apply(
                        lambda x: str(x) if pd.notnull(x) else None
                    )
        
        return df_converted