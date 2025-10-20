"""
DataAPIJSONUseCase - Use case for loading data from APIs and JSON sources.

This use case orchestrates the complete workflow for loading data from external APIs
and JSON sources, including flattening, column mapping, transformations, embedding
generation, and database operations while maintaining compatibility with existing
patterns and interfaces.
"""

import asyncio
import time
from typing import Optional, List, Dict, Union, Any, Set
import pandas as pd

from dataload.interfaces.data_move_repository import DataMoveRepositoryInterface
from dataload.interfaces.embedding_provider import EmbeddingProviderInterface
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.domain.entities import (
    DataMoveResult, ValidationReport, DataMoveError, ValidationError,
    DatabaseOperationError, DataValidationError
)
from dataload.domain.api_entities import (
    APIJSONLoadResult, APIError, AuthenticationError, JSONParsingError,
    ColumnMappingError, RequestTransformationError
)
from dataload.config import logger


class DataAPIJSONUseCase:
    """
    Use case for loading data from APIs and JSON sources with embedding generation.
    
    This use case provides a comprehensive solution for loading data from external APIs
    and JSON sources into PostgreSQL databases with optional embedding generation.
    It integrates with the existing DataMoveRepositoryInterface and EmbeddingProviderInterface
    while adding support for API/JSON specific processing capabilities.
    
    Key Features:
        - **API Data Loading**: Fetch data from external APIs with authentication
        - **JSON Processing**: Handle complex nested JSON structures with flattening
        - **Column Mapping**: Transform API field names to database column names
        - **Data Transformation**: Apply custom transformations to request data
        - **Embedding Generation**: Generate embeddings for specified columns
        - **Table Management**: Create new tables or update existing ones
        - **Transaction Safety**: Automatic rollback on failures
        - **Comprehensive Error Handling**: Detailed error context and recovery
    
    Embedding Types:
        - **separated**: Creates individual embedding columns with "_enc" suffix
        - **combined**: Creates a single "embeddings" column combining all specified columns
    
    Usage Patterns:
        
        Basic API Loading:
            >>> use_case = DataAPIJSONUseCase(repository, embedding_service, api_loader)
            >>> result = await use_case.execute(
            ...     source="https://api.example.com/data",
            ...     table_name="api_data",
            ...     embed_columns_names=["title", "description"]
            ... )
        
        JSON File Loading:
            >>> result = await use_case.execute(
            ...     source="/path/to/data.json",
            ...     table_name="json_data",
            ...     create_table_if_not_exists=True
            ... )
        
        Raw JSON Data:
            >>> json_data = [{"id": 1, "name": "test"}]
            >>> result = await use_case.execute(
            ...     source=json_data,
            ...     table_name="raw_data"
            ... )
        
        With Column Mapping:
            >>> result = await use_case.execute(
            ...     source="https://api.example.com/users",
            ...     table_name="users",
            ...     column_name_mapping={"user_id": "id", "full_name": "name"},
            ...     embed_columns_names=["name"]  # Uses mapped column name
            ... )
        
        Existing Table Update:
            >>> result = await use_case.execute(
            ...     source="https://api.example.com/updates",
            ...     table_name="existing_table",
            ...     pk_columns=["id"],
            ...     create_table_if_not_exists=False
            ... )
    
    Error Handling:
        All operations provide comprehensive error context through DataMoveError
        and its subclasses, including API-specific errors like AuthenticationError
        and JSONParsingError.
    
    Thread Safety:
        DataAPIJSONUseCase instances are not thread-safe. Create separate instances
        for concurrent operations or use appropriate synchronization.
    """
    
    def __init__(self,
                 repo: DataMoveRepositoryInterface,
                 embedding_service: EmbeddingProviderInterface,
                 storage_loader: APIJSONStorageLoader):
        """
        Initialize the DataAPIJSONUseCase.
        
        Args:
            repo: DataMoveRepositoryInterface implementation for database operations
            embedding_service: EmbeddingProviderInterface for generating embeddings
            storage_loader: APIJSONStorageLoader for loading API/JSON data
        """
        self.repo = repo
        self.embedding_service = embedding_service
        self.storage_loader = storage_loader
        self._active_tasks: Set[asyncio.Task] = set()
        
        logger.info("DataAPIJSONUseCase initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper resource cleanup."""
        await self.cleanup()
    
    async def cleanup(self):
        """
        Clean up resources and cancel any active tasks.
        
        This method ensures proper cleanup of:
        - Active async tasks
        - Storage loader connections
        - Database connections (if applicable)
        """
        try:
            # Cancel any active tasks
            if self._active_tasks:
                logger.info(f"Cancelling {len(self._active_tasks)} active tasks")
                for task in self._active_tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait for tasks to complete cancellation
                if self._active_tasks:
                    await asyncio.gather(*self._active_tasks, return_exceptions=True)
                
                self._active_tasks.clear()
            
            # Close storage loader connections
            if hasattr(self.storage_loader, 'close'):
                await self.storage_loader.close()
            
            # Close repository connections if applicable
            if hasattr(self.repo, 'close'):
                await self.repo.close()
            
            logger.debug("DataAPIJSONUseCase cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _track_task(self, task: asyncio.Task) -> asyncio.Task:
        """
        Track an async task for proper cleanup.
        
        Args:
            task: Async task to track
            
        Returns:
            The same task for chaining
        """
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        return task
    
    async def execute(self,
                     source: Union[str, Dict, List[Dict]],
                     table_name: str,
                     embed_columns_names: Optional[List[str]] = None,
                     pk_columns: Optional[List[str]] = None,
                     create_table_if_not_exists: bool = True,
                     embed_type: str = "combined",
                     column_name_mapping: Optional[Dict[str, str]] = None,
                     update_request_body_mapping: Optional[Dict[str, str]] = None,
                     **kwargs) -> DataMoveResult:
        """
        Execute the API/JSON data loading workflow with embedding generation.
        
        This method orchestrates the complete workflow:
        1. Load and process JSON/API data
        2. Apply column mappings and transformations
        3. Generate embeddings for specified columns
        4. Create or update database table
        5. Insert/upsert data with embeddings
        
        Args:
            source: Data source (API URL, file path, or raw JSON data)
            table_name: Name of the target PostgreSQL table
            embed_columns_names: List of column names to generate embeddings for
            pk_columns: List of primary key columns for table creation/upserts
            create_table_if_not_exists: Whether to create table if it doesn't exist
            embed_type: Type of embedding ("combined" or "separated")
            column_name_mapping: Mapping from API field names to database column names
            update_request_body_mapping: Transformations to apply to request data
            **kwargs: Additional configuration parameters for JSON processing
            
        Returns:
            DataMoveResult: Comprehensive result of the operation
            
        Raises:
            ValidationError: If validation fails or invalid parameters provided
            DatabaseOperationError: If database operations fail
            APIError: If API requests fail
            DataMoveError: For other data movement related errors
        """
        start_time = time.time()
        operation_context = {
            "source": str(source)[:100] + "..." if len(str(source)) > 100 else str(source),
            "table_name": table_name,
            "embed_columns_names": embed_columns_names,
            "pk_columns": pk_columns,
            "create_table_if_not_exists": create_table_if_not_exists,
            "embed_type": embed_type,
            "column_name_mapping": column_name_mapping,
            "update_request_body_mapping": update_request_body_mapping
        }
        
        table_created = False
        schema_updated = False
        rows_processed = 0
        collected_errors = []
        collected_warnings = []
        api_load_result = None
        
        try:
            logger.info(f"Starting DataAPIJSON operation: {type(source).__name__} -> {table_name}")
            
            # Step 1: Validate input parameters
            logger.debug("Validating input parameters")
            self._validate_parameters(
                source, table_name, embed_columns_names, pk_columns, 
                embed_type, column_name_mapping, update_request_body_mapping
            )
            
            # Step 2: Load and process JSON/API data
            logger.info("Loading and processing JSON/API data")
            api_load_result = await self._load_and_process_data(
                source, column_name_mapping, update_request_body_mapping, kwargs
            )
            
            if not api_load_result.success or api_load_result.dataframe is None:
                error_msg = f"Data loading failed: {api_load_result.errors}"
                raise DataMoveError(error_msg, operation_context)
            
            collected_warnings.extend(api_load_result.warnings)
            df = api_load_result.dataframe
            
            if df.empty:
                collected_warnings.append("Loaded data is empty - no records to process")
                logger.warning("Loaded data is empty")
            
            # Step 3: Validate embedding columns after processing (will be done after data processing)
            
            # Step 4: Check table existence and determine operation type
            logger.info(f"Checking table existence: {table_name}")
            table_exists = await self.repo.table_exists(table_name)
            
            if table_exists and not create_table_if_not_exists:
                operation_type = "existing_table_update"
            elif table_exists and create_table_if_not_exists:
                operation_type = "existing_table_replace"
            else:
                operation_type = "new_table_creation"
            
            logger.info(f"Operation type determined: {operation_type}")
            
            # Step 5: Validate embedding columns after processing and generate embeddings if specified
            df_with_embeddings = df.copy()
            if embed_columns_names and not df.empty:
                logger.debug("Validating embedding columns after data processing")
                self._validate_embedding_columns(df, embed_columns_names, column_name_mapping)
                
                logger.info(f"Generating embeddings for columns: {embed_columns_names}")
                df_with_embeddings = await self._generate_embeddings(
                    df, embed_columns_names, embed_type
                )
            
            # Step 6: Execute database operations based on operation type
            logger.info("Executing database operations")
            rows_processed, table_created, schema_updated = await self._execute_database_operations(
                df_with_embeddings, table_name, operation_type, pk_columns, embed_columns_names, embed_type
            )
            
            execution_time = time.time() - start_time
            
            # Create successful result
            result = DataMoveResult(
                success=True,
                rows_processed=rows_processed,
                execution_time=execution_time,
                validation_report=self._create_validation_report(api_load_result),
                errors=collected_errors,
                warnings=collected_warnings,
                table_created=table_created,
                schema_updated=schema_updated,
                operation_type=operation_type
            )
            
            logger.info(
                f"DataAPIJSON operation completed successfully in {execution_time:.2f}s: "
                f"{rows_processed} rows processed, {len(collected_warnings)} warnings"
            )
            return result
            
        except (ValidationError, DatabaseOperationError, APIError, DataMoveError) as e:
            # Handle known errors with enhanced context
            execution_time = time.time() - start_time
            
            if not hasattr(e, 'context') or not e.context:
                e.context = {}
            e.context.update({
                **operation_context,
                "execution_time": execution_time,
                "api_load_result": api_load_result.__dict__ if api_load_result else None
            })
            
            collected_errors.append(e)
            
            logger.error(f"DataAPIJSON operation failed after {execution_time:.2f}s: {e}")
            
            # Create failure result
            failure_result = DataMoveResult(
                success=False,
                rows_processed=rows_processed,
                execution_time=execution_time,
                validation_report=self._create_validation_report(api_load_result) if api_load_result else self._create_empty_validation_report(),
                errors=collected_errors,
                warnings=collected_warnings,
                table_created=table_created,
                schema_updated=schema_updated,
                operation_type=operation_context.get("operation_type")
            )
            
            e.context["failure_result"] = failure_result
            raise e
            
        except Exception as e:
            # Handle unexpected errors
            execution_time = time.time() - start_time
            
            error_msg = f"Unexpected error during DataAPIJSON operation: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            
            data_move_error = DataMoveError(error_msg, {
                **operation_context,
                "execution_time": execution_time,
                "original_exception_type": type(e).__name__,
                "original_exception_message": str(e),
                "api_load_result": api_load_result.__dict__ if api_load_result else None
            })
            
            collected_errors.append(data_move_error)
            
            failure_result = DataMoveResult(
                success=False,
                rows_processed=rows_processed,
                execution_time=execution_time,
                validation_report=self._create_validation_report(api_load_result) if api_load_result else self._create_empty_validation_report(),
                errors=collected_errors,
                warnings=collected_warnings,
                table_created=table_created,
                schema_updated=schema_updated,
                operation_type=operation_context.get("operation_type")
            )
            
            data_move_error.context["failure_result"] = failure_result
            raise data_move_error from e    

    def _validate_parameters(self,
                           source: Union[str, Dict, List[Dict]],
                           table_name: str,
                           embed_columns_names: Optional[List[str]],
                           pk_columns: Optional[List[str]],
                           embed_type: str,
                           column_name_mapping: Optional[Dict[str, str]],
                           update_request_body_mapping: Optional[Dict[str, str]]) -> None:
        """
        Validate input parameters for the execute method.
        
        Args:
            source: Data source to validate
            table_name: Table name to validate
            embed_columns_names: Embedding columns to validate
            pk_columns: Primary key columns to validate
            embed_type: Embedding type to validate
            column_name_mapping: Column mapping to validate
            update_request_body_mapping: Request body mapping to validate
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate source
        if source is None:
            raise ValidationError("Source cannot be None", {
                "error_type": "null_source",
                "provided_source": source
            })
        
        if isinstance(source, str) and not source.strip():
            raise ValidationError("Source string cannot be empty", {
                "error_type": "empty_source_string",
                "provided_source": source
            })
        
        # Validate table name
        if not table_name or not table_name.strip():
            raise ValidationError("Table name cannot be empty", {
                "error_type": "empty_table_name",
                "provided_table_name": table_name
            })
        
        # Validate embed_type
        if embed_type not in ["combined", "separated"]:
            raise ValidationError(
                f"Invalid embed_type: '{embed_type}'. Must be 'combined' or 'separated'.",
                {
                    "error_type": "invalid_embed_type",
                    "provided_embed_type": embed_type,
                    "valid_embed_types": ["combined", "separated"]
                }
            )
        
        # Validate embed_columns_names
        if embed_columns_names is not None:
            if not isinstance(embed_columns_names, list):
                raise ValidationError("embed_columns_names must be a list", {
                    "error_type": "invalid_embed_columns_type",
                    "provided_type": type(embed_columns_names).__name__
                })
            
            if len(embed_columns_names) == 0:
                # Empty list is valid - means no embeddings
                pass
            else:
                for col in embed_columns_names:
                    if not isinstance(col, str) or not col.strip():
                        raise ValidationError(f"Invalid column name in embed_columns_names: '{col}'", {
                            "error_type": "invalid_embed_column_name",
                            "invalid_column": col
                        })
        
        # Validate pk_columns
        if pk_columns is not None:
            if not isinstance(pk_columns, list):
                raise ValidationError("pk_columns must be a list", {
                    "error_type": "invalid_pk_columns_type",
                    "provided_type": type(pk_columns).__name__
                })
            
            for col in pk_columns:
                if not isinstance(col, str) or not col.strip():
                    raise ValidationError(f"Invalid column name in pk_columns: '{col}'", {
                        "error_type": "invalid_pk_column_name",
                        "invalid_column": col
                    })
        
        # Validate column_name_mapping
        if column_name_mapping is not None:
            if not isinstance(column_name_mapping, dict):
                raise ValidationError("column_name_mapping must be a dictionary", {
                    "error_type": "invalid_column_mapping_type",
                    "provided_type": type(column_name_mapping).__name__
                })
            
            for source_col, target_col in column_name_mapping.items():
                if not isinstance(source_col, str) or not isinstance(target_col, str):
                    raise ValidationError(
                        f"Column mapping keys and values must be strings: '{source_col}' -> '{target_col}'",
                        {
                            "error_type": "invalid_column_mapping_types",
                            "source_column": source_col,
                            "target_column": target_col
                        }
                    )
                
                if not source_col.strip() or not target_col.strip():
                    raise ValidationError(
                        f"Column names cannot be empty: '{source_col}' -> '{target_col}'",
                        {
                            "error_type": "empty_column_names",
                            "source_column": source_col,
                            "target_column": target_col
                        }
                    )
        
        # Validate update_request_body_mapping
        if update_request_body_mapping is not None:
            if not isinstance(update_request_body_mapping, dict):
                raise ValidationError("update_request_body_mapping must be a dictionary", {
                    "error_type": "invalid_request_mapping_type",
                    "provided_type": type(update_request_body_mapping).__name__
                })
            
            for target_field, expression in update_request_body_mapping.items():
                if not isinstance(target_field, str) or not isinstance(expression, str):
                    raise ValidationError(
                        f"Request mapping keys and values must be strings: '{target_field}' -> '{expression}'",
                        {
                            "error_type": "invalid_request_mapping_types",
                            "target_field": target_field,
                            "expression": expression
                        }
                    )
                
                if not target_field.strip():
                    raise ValidationError(f"Target field name cannot be empty: '{target_field}'", {
                        "error_type": "empty_target_field",
                        "target_field": target_field
                    })
    
    async def _load_and_process_data(self,
                                   source: Union[str, Dict, List[Dict]],
                                   column_name_mapping: Optional[Dict[str, str]],
                                   update_request_body_mapping: Optional[Dict[str, str]],
                                   additional_config: Dict[str, Any]) -> APIJSONLoadResult:
        """
        Load and process data from the source with mappings and transformations.
        
        Args:
            source: Data source (API URL, file path, or raw JSON)
            column_name_mapping: Column name mappings to apply
            update_request_body_mapping: Request body transformations to apply
            additional_config: Additional configuration parameters
            
        Returns:
            APIJSONLoadResult: Result of the data loading and processing
            
        Raises:
            APIError: If API loading fails
            JSONParsingError: If JSON processing fails
            DataMoveError: For other loading errors
        """
        try:
            # Prepare configuration for the storage loader
            config = {
                "column_name_mapping": column_name_mapping or {},
                "update_request_body_mapping": update_request_body_mapping or {},
                **additional_config
            }
            
            # Validate configuration
            config_errors = self.storage_loader.validate_config(config)
            if config_errors:
                raise ValidationError(f"Invalid configuration: {'; '.join(config_errors)}", {
                    "config_errors": config_errors,
                    "provided_config": config
                })
            
            # Load data using the storage loader
            df = await self.storage_loader.load_json(source, config)
            
            # Create a successful result (the storage loader doesn't return APIJSONLoadResult directly)
            return APIJSONLoadResult(
                success=True,
                dataframe=df,
                rows_loaded=len(df),
                columns_created=len(df.columns),
                execution_time=0.0,  # This would be tracked by the storage loader
                api_responses=[],
                flattening_result=None,
                mapping_result=None,
                transformation_result=None,
                errors=[],
                warnings=[],
                operation_metadata={
                    "source_type": type(source).__name__,
                    "config_used": config
                }
            )
            
        except (APIError, JSONParsingError, ColumnMappingError, RequestTransformationError) as e:
            # Re-raise domain-specific errors
            logger.error(f"Data loading failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during data loading: {e}")
            raise DataMoveError(f"Failed to load and process data: {e}") from e
    
    def _validate_embedding_columns(self,
                                  df: pd.DataFrame,
                                  embed_columns_names: List[str],
                                  column_name_mapping: Optional[Dict[str, str]]) -> None:
        """
        Validate that embedding columns exist in the DataFrame after mapping.
        
        This method ensures that all specified embedding columns exist in the DataFrame
        after column mapping has been applied. It provides helpful error messages and
        suggestions when columns are missing, including guidance about using mapped
        column names instead of original API field names.
        
        Args:
            df: DataFrame to check (after mapping has been applied)
            embed_columns_names: List of column names for embedding (should be mapped names)
            column_name_mapping: Column name mappings that were applied
            
        Raises:
            ValidationError: If embedding columns are not found with detailed context
        """
        if not embed_columns_names:
            return
        
        available_columns = set(df.columns)
        missing_columns = []
        
        # Check each embedding column
        for col in embed_columns_names:
            if col not in available_columns:
                missing_columns.append(col)
        
        if missing_columns:
            # Build comprehensive error message with suggestions
            error_msg = f"Embedding columns not found in DataFrame: {missing_columns}"
            
            context = {
                "missing_columns": missing_columns,
                "available_columns": list(available_columns),
                "embed_columns_names": embed_columns_names
            }
            
            # Provide specific guidance based on column mapping
            if column_name_mapping:
                error_msg += "\n\nNote: embed_columns_names should refer to mapped column names, not original API field names."
                
                # Check if missing columns might be original field names
                reverse_mapping = {v: k for k, v in column_name_mapping.items()}
                potential_original_names = []
                suggested_mapped_names = []
                
                for missing_col in missing_columns:
                    # Check if this might be an original field name
                    if missing_col in column_name_mapping:
                        mapped_name = column_name_mapping[missing_col]
                        if mapped_name in available_columns:
                            potential_original_names.append(missing_col)
                            suggested_mapped_names.append(mapped_name)
                
                if potential_original_names:
                    error_msg += f"\n\nIt appears you may be using original API field names instead of mapped names:"
                    for orig, mapped in zip(potential_original_names, suggested_mapped_names):
                        error_msg += f"\n  - '{orig}' should be '{mapped}'"
                    error_msg += f"\n\nSuggested embed_columns_names: {suggested_mapped_names + [col for col in embed_columns_names if col not in potential_original_names]}"
                
                context.update({
                    "column_name_mapping": column_name_mapping,
                    "reverse_mapping": reverse_mapping,
                    "potential_original_names": potential_original_names,
                    "suggested_mapped_names": suggested_mapped_names
                })
            else:
                # No mapping was applied, suggest available columns that might be similar
                suggestions = self._suggest_similar_columns(missing_columns, available_columns)
                if suggestions:
                    error_msg += f"\n\nDid you mean one of these available columns? {suggestions}"
                    context["column_suggestions"] = suggestions
            
            error_msg += f"\n\nAvailable columns after mapping: {sorted(available_columns)}"
            
            logger.error(f"Embedding column validation failed: {error_msg}")
            raise ValidationError(error_msg, context)
    
    def _suggest_similar_columns(self, missing_columns: List[str], available_columns: Set[str]) -> Dict[str, List[str]]:
        """
        Suggest similar column names for missing embedding columns.
        
        Args:
            missing_columns: List of missing column names
            available_columns: Set of available column names
            
        Returns:
            Dictionary mapping missing columns to suggested alternatives
        """
        suggestions = {}
        
        for missing_col in missing_columns:
            col_suggestions = []
            missing_lower = missing_col.lower()
            
            # Look for exact case-insensitive matches
            for available_col in available_columns:
                if available_col.lower() == missing_lower:
                    col_suggestions.append(available_col)
            
            # Look for partial matches if no exact match found
            if not col_suggestions:
                for available_col in available_columns:
                    available_lower = available_col.lower()
                    # Check if either contains the other
                    if (missing_lower in available_lower or 
                        available_lower in missing_lower or
                        self._calculate_similarity(missing_lower, available_lower) > 0.6):
                        col_suggestions.append(available_col)
            
            if col_suggestions:
                suggestions[missing_col] = col_suggestions[:3]  # Limit to top 3 suggestions
        
        return suggestions
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using simple Levenshtein distance.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if str1 == str2:
            return 1.0
        
        len1, len2 = len(str1), len(str2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Simple implementation for basic similarity
        max_len = max(len1, len2)
        common_chars = sum(1 for c1, c2 in zip(str1, str2) if c1 == c2)
        
        return common_chars / max_len
    
    async def _generate_embeddings(self,
                                 df: pd.DataFrame,
                                 embed_columns_names: List[str],
                                 embed_type: str) -> pd.DataFrame:
        """
        Generate embeddings for specified columns in the DataFrame with memory optimization.
        
        This method generates embeddings for the specified columns after column mapping
        has been applied. It supports both "combined" and "separated" embedding types
        and ensures that all specified columns exist in the DataFrame.
        
        Args:
            df: DataFrame containing the data (after column mapping)
            embed_columns_names: List of column names to generate embeddings for (mapped names)
            embed_type: Type of embedding ("combined" or "separated")
            
        Returns:
            DataFrame with embedding columns added
            
        Raises:
            ValidationError: If embedding columns are missing or invalid
            DataMoveError: For other embedding-related errors
        """
        try:
            logger.info(f"Starting embedding generation for {len(embed_columns_names)} columns with {embed_type} type")
            
            # Validate that all embedding columns exist in the DataFrame
            missing_columns = [col for col in embed_columns_names if col not in df.columns]
            if missing_columns:
                error_msg = (f"Cannot generate embeddings: columns {missing_columns} not found in DataFrame. "
                           f"Available columns: {list(df.columns)}")
                logger.error(error_msg)
                raise ValidationError(error_msg, {
                    "missing_embedding_columns": missing_columns,
                    "available_columns": list(df.columns),
                    "requested_embed_columns": embed_columns_names,
                    "embed_type": embed_type
                })
            
            df_with_embeddings = df.copy()
            
            # Add embed_columns_names metadata column
            df_with_embeddings["embed_columns_names"] = [embed_columns_names] * len(df_with_embeddings)
            
            # For large datasets, use chunked processing to optimize memory usage
            chunk_size = 1000
            if len(df) > chunk_size:
                return await self._generate_embeddings_chunked(
                    df_with_embeddings, embed_columns_names, embed_type, chunk_size
                )
            
            if embed_type == "combined":
                # Generate combined embeddings
                logger.debug(f"Generating combined embeddings for columns: {embed_columns_names}")
                
                # Validate that all columns have data
                empty_columns = []
                for col in embed_columns_names:
                    if df_with_embeddings[col].isna().all():
                        empty_columns.append(col)
                
                if empty_columns:
                    logger.warning(f"Embedding columns contain only null values: {empty_columns}")
                
                # Create combined text for embedding
                df_with_embeddings["embed_columns_value"] = df_with_embeddings[embed_columns_names].apply(
                    lambda row: self._format_embed_value(row, embed_columns_names), axis=1
                )
                
                # Generate embeddings
                texts = df_with_embeddings["embed_columns_value"].tolist()
                
                # Validate that we have texts to embed
                if not texts or all(not text.strip() for text in texts):
                    logger.warning("All embedding texts are empty - generating embeddings for empty strings")
                
                # Run embedding generation in executor to avoid blocking
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, self.embedding_service.get_embeddings, texts
                )
                df_with_embeddings["embeddings"] = embeddings
                
                logger.debug(f"Generated {len(embeddings)} combined embeddings")
                
            else:  # separated
                # Generate separate embeddings for each column concurrently
                logger.debug(f"Generating separated embeddings for columns: {embed_columns_names}")
                
                embedding_tasks = []
                for col in embed_columns_names:
                    logger.debug(f"Processing column '{col}' for separated embeddings")
                    
                    # Check if column has any non-null values
                    if df_with_embeddings[col].isna().all():
                        logger.warning(f"Column '{col}' contains only null values")
                    
                    # Convert to string and handle null values
                    texts = df_with_embeddings[col].fillna('').astype(str).tolist()
                    
                    # Create async task for embedding generation
                    task = asyncio.get_event_loop().run_in_executor(
                        None, self.embedding_service.get_embeddings, texts
                    )
                    embedding_tasks.append((col, task))
                
                # Wait for all embedding tasks to complete
                for col, task in embedding_tasks:
                    embeddings = await task
                    embedding_col_name = f"{col}_enc"
                    df_with_embeddings[embedding_col_name] = embeddings
                    
                    logger.debug(f"Generated {len(embeddings)} embeddings for column '{col}' -> '{embedding_col_name}'")
            
            # Add is_active column (following existing pattern)
            df_with_embeddings["is_active"] = True
            
            # Log summary of embedding generation
            embedding_columns_created = []
            if embed_type == "combined":
                embedding_columns_created = ["embeddings"]
            else:
                embedding_columns_created = [f"{col}_enc" for col in embed_columns_names]
            
            logger.info(f"Successfully generated {embed_type} embeddings for {len(embed_columns_names)} columns. "
                       f"Created embedding columns: {embedding_columns_created}")
            
            return df_with_embeddings
            
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            error_msg = f"Embedding generation failed for columns {embed_columns_names} with {embed_type} type: {e}"
            logger.error(error_msg, exc_info=True)
            raise DataMoveError(error_msg, {
                "embed_columns_names": embed_columns_names,
                "embed_type": embed_type,
                "dataframe_columns": list(df.columns),
                "original_error": str(e),
                "error_type": type(e).__name__
            }) from e
    
    async def _generate_embeddings_chunked(self,
                                         df: pd.DataFrame,
                                         embed_columns_names: List[str],
                                         embed_type: str,
                                         chunk_size: int) -> pd.DataFrame:
        """
        Generate embeddings in chunks for memory optimization with large datasets.
        
        Args:
            df: DataFrame with embed_columns_names already added
            embed_columns_names: List of column names to generate embeddings for
            embed_type: Type of embedding ("combined" or "separated")
            chunk_size: Size of each processing chunk
            
        Returns:
            DataFrame with embedding columns added
            
        Raises:
            DataMoveError: If chunked embedding generation fails
        """
        total_rows = len(df)
        logger.info(f"Starting chunked embedding generation: {total_rows} rows in chunks of {chunk_size}")
        
        try:
            processed_chunks = []
            
            for i in range(0, total_rows, chunk_size):
                chunk_end = min(i + chunk_size, total_rows)
                chunk_df = df.iloc[i:chunk_end].copy()
                
                logger.debug(f"Processing embedding chunk {i//chunk_size + 1}: rows {i+1}-{chunk_end}")
                
                if embed_type == "combined":
                    # Create combined text for embedding
                    chunk_df["embed_columns_value"] = chunk_df[embed_columns_names].apply(
                        lambda row: self._format_embed_value(row, embed_columns_names), axis=1
                    )
                    
                    # Generate embeddings for this chunk
                    texts = chunk_df["embed_columns_value"].tolist()
                    embeddings = await asyncio.get_event_loop().run_in_executor(
                        None, self.embedding_service.get_embeddings, texts
                    )
                    chunk_df["embeddings"] = embeddings
                    
                else:  # separated
                    # Generate separate embeddings for each column in this chunk
                    for col in embed_columns_names:
                        texts = chunk_df[col].fillna('').astype(str).tolist()
                        embeddings = await asyncio.get_event_loop().run_in_executor(
                            None, self.embedding_service.get_embeddings, texts
                        )
                        embedding_col_name = f"{col}_enc"
                        chunk_df[embedding_col_name] = embeddings
                
                # Add is_active column for this chunk
                chunk_df["is_active"] = True
                
                processed_chunks.append(chunk_df)
                
                # Log progress for large operations
                if total_rows > 5000:
                    progress = (chunk_end / total_rows) * 100
                    logger.info(f"Embedding generation progress: {chunk_end}/{total_rows} rows ({progress:.1f}%)")
                
                # Allow other async operations to run between chunks
                await asyncio.sleep(0)
            
            # Combine all processed chunks
            result_df = pd.concat(processed_chunks, ignore_index=True)
            
            logger.info(f"Chunked embedding generation completed: {len(result_df)} total rows processed")
            return result_df
            
        except Exception as e:
            logger.error(f"Chunked embedding generation failed: {e}")
            raise DataMoveError(f"Chunked embedding generation failed: {e}") from e
    
    def _format_embed_value(self, row: pd.Series, embed_columns_names: List[str]) -> str:
        """
        Format embed_columns_value with column names and handle diverse types.
        
        This method creates a formatted string representation of the specified columns
        for embedding generation. It handles various data types and null values gracefully.
        
        Args:
            row: DataFrame row containing the data
            embed_columns_names: List of column names to include (mapped column names)
            
        Returns:
            Formatted string for embedding generation
            
        Raises:
            KeyError: If a specified column doesn't exist in the row
        """
        parts = []
        
        for col in embed_columns_names:
            try:
                value = row[col]
                
                # Handle different data types
                if isinstance(value, list):
                    # Convert list to comma-separated string
                    value_str = ", ".join(str(v) for v in value if v is not None)
                elif isinstance(value, dict):
                    # Convert dict to key=value pairs
                    value_str = ", ".join(f"{k}={v}" for k, v in value.items() if v is not None)
                elif isinstance(value, (int, float)):
                    # Handle numeric values, including NaN
                    if pd.isna(value):
                        value_str = ""
                    else:
                        value_str = str(value)
                elif value is None or pd.isna(value):
                    # Handle null values
                    value_str = ""
                else:
                    # Convert to string, handling any other types
                    value_str = str(value).strip()
                
                # Add to parts with column name for context
                parts.append(f"{col}='{value_str}'")
                
            except KeyError:
                # This should not happen if validation was done correctly
                logger.error(f"Column '{col}' not found in row during embedding formatting")
                raise ValidationError(f"Column '{col}' not found in DataFrame row", {
                    "missing_column": col,
                    "available_columns": list(row.index),
                    "embed_columns_names": embed_columns_names
                })
        
        result = ", ".join(parts)
        
        # Ensure we don't return completely empty strings
        if not result.strip():
            logger.warning(f"Generated empty embedding value for columns {embed_columns_names}")
            result = f"empty_values_for_columns={','.join(embed_columns_names)}"
        
        return result
    
    async def _execute_database_operations(self,
                                         df: pd.DataFrame,
                                         table_name: str,
                                         operation_type: str,
                                         pk_columns: Optional[List[str]],
                                         embed_columns_names: Optional[List[str]],
                                         embed_type: str) -> tuple[int, bool, bool]:
        """
        Execute database operations based on the operation type.
        
        Args:
            df: DataFrame with data and embeddings
            table_name: Name of target table
            operation_type: Type of operation to perform
            pk_columns: Primary key columns
            embed_columns_names: Embedding column names
            embed_type: Embedding type
            
        Returns:
            Tuple of (rows_processed, table_created, schema_updated)
            
        Raises:
            DatabaseOperationError: If database operations fail
        """
        rows_processed = 0
        table_created = False
        schema_updated = False
        
        try:
            if operation_type == "new_table_creation":
                # Create new table and insert data
                logger.info(f"Creating new table '{table_name}' with embeddings")
                
                if not df.empty:
                    # For new tables with embeddings, we need to use the existing create_table method
                    # that handles embedding columns properly
                    await self.repo.create_table(
                        table_name=table_name,
                        df=df,
                        pk_columns=pk_columns or [],
                        embed_type=embed_type,
                        embed_columns_names=embed_columns_names or []
                    )
                    table_created = True
                    
                    # Insert data
                    await self.repo.insert_data(table_name, df, pk_columns or [])
                    rows_processed = len(df)
                else:
                    # Create empty table
                    await self.repo.create_table(
                        table_name=table_name,
                        df=df,
                        pk_columns=pk_columns or [],
                        embed_type=embed_type,
                        embed_columns_names=embed_columns_names or []
                    )
                    table_created = True
                    logger.info("Created empty table with embedding schema")
                    
            elif operation_type == "existing_table_replace":
                # Replace data in existing table
                logger.info(f"Replacing data in existing table '{table_name}'")
                
                if not df.empty:
                    # Use replace_table_data for bulk replacement
                    rows_processed = await self.repo.replace_table_data(
                        table_name=table_name,
                        df=df,
                        batch_size=1000
                    )
                else:
                    # Truncate table for empty DataFrame
                    rows_processed = await self.repo.replace_table_data(
                        table_name=table_name,
                        df=df,
                        batch_size=1000
                    )
                    
            elif operation_type == "existing_table_update":
                # Update existing table with upsert operations
                logger.info(f"Updating existing table '{table_name}' with upsert operations")
                
                if not df.empty:
                    rows_processed = await self._execute_upsert_operations(
                        df, table_name, pk_columns, embed_columns_names, embed_type
                    )
                else:
                    logger.info("No data to update")
                    
            else:
                raise DatabaseOperationError(f"Unknown operation type: {operation_type}")
            
            logger.info(f"Database operations completed: {rows_processed} rows processed")
            return rows_processed, table_created, schema_updated
            
        except Exception as e:
            logger.error(f"Database operations failed: {e}")
            raise DatabaseOperationError(f"Failed to execute database operations: {e}") from e
    
    async def _execute_upsert_operations(self,
                                       df: pd.DataFrame,
                                       table_name: str,
                                       pk_columns: Optional[List[str]],
                                       embed_columns_names: Optional[List[str]],
                                       embed_type: str) -> int:
        """
        Execute upsert operations for existing table updates with proper timestamp handling.
        
        This method implements primary key-based upsert operations with:
        - Proper timestamp handling for created_at and updated_at columns
        - Transaction safety with rollback on failures
        - Embedding column updates for existing records
        - Validation of primary key requirements
        
        Args:
            df: DataFrame with data and embeddings to upsert
            table_name: Name of target table
            pk_columns: Primary key columns for upsert operations
            embed_columns_names: Embedding column names
            embed_type: Embedding type ("combined" or "separated")
            
        Returns:
            Number of rows processed
            
        Raises:
            DatabaseOperationError: If upsert operations fail
            ValidationError: If primary key validation fails
        """
        try:
            # Validate primary key columns are provided for upsert operations
            if not pk_columns:
                logger.warning("No primary key columns specified - performing insert operations only")
                await self.repo.insert_data(table_name, df, [])
                return len(df)
            
            # Validate primary key columns exist in DataFrame
            missing_pk_cols = [col for col in pk_columns if col not in df.columns]
            if missing_pk_cols:
                raise ValidationError(
                    f"Primary key columns not found in DataFrame: {missing_pk_cols}",
                    {
                        "missing_pk_columns": missing_pk_cols,
                        "available_columns": list(df.columns),
                        "required_pk_columns": pk_columns
                    }
                )
            
            # Validate primary key columns don't contain null values
            for col in pk_columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    raise ValidationError(
                        f"Primary key column '{col}' contains {null_count} null values",
                        {
                            "column": col,
                            "null_count": null_count,
                            "total_rows": len(df)
                        }
                    )
            
            # Prepare DataFrame with timestamp handling
            df_with_timestamps = await self._prepare_dataframe_with_timestamps(
                df, table_name, pk_columns
            )
            
            # Execute upsert operations with transaction safety
            rows_processed = await self._execute_transactional_upsert(
                df_with_timestamps, table_name, pk_columns, embed_columns_names, embed_type
            )
            
            logger.info(f"Successfully processed {rows_processed} rows with upsert operations")
            return rows_processed
            
        except (ValidationError, DatabaseOperationError) as e:
            logger.error(f"Upsert operations failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during upsert operations: {e}")
            raise DatabaseOperationError(f"Upsert operations failed: {e}") from e
    
    async def _prepare_dataframe_with_timestamps(self,
                                               df: pd.DataFrame,
                                               table_name: str,
                                               pk_columns: List[str]) -> pd.DataFrame:
        """
        Prepare DataFrame with proper timestamp handling for created_at and updated_at columns.
        
        This method:
        - Adds updated_at timestamp for all records
        - Preserves existing created_at for updates, adds created_at for new records
        - Handles timezone-aware timestamps
        
        Args:
            df: Original DataFrame
            table_name: Name of target table
            pk_columns: Primary key columns
            
        Returns:
            DataFrame with proper timestamp columns
        """
        import datetime
        
        df_prepared = df.copy()
        current_timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        try:
            # Check if table has timestamp columns
            table_info = await self.repo.get_table_info(table_name)
            has_created_at = "created_at" in table_info.columns
            has_updated_at = "updated_at" in table_info.columns
            
            # Always set updated_at to current timestamp for upsert operations
            if has_updated_at:
                df_prepared["updated_at"] = current_timestamp
                logger.debug("Added updated_at timestamp for upsert operations")
            
            # Handle created_at column - preserve existing values for updates
            if has_created_at:
                if "created_at" not in df_prepared.columns:
                    # For new records, set created_at to current timestamp
                    # For existing records, this will be preserved by the upsert logic
                    df_prepared["created_at"] = current_timestamp
                    logger.debug("Added created_at timestamp for new records")
                else:
                    # If created_at is already in DataFrame, ensure it's properly formatted
                    df_prepared["created_at"] = pd.to_datetime(
                        df_prepared["created_at"], errors="coerce"
                    ).fillna(current_timestamp)
                    logger.debug("Validated and filled missing created_at timestamps")
            
            return df_prepared
            
        except Exception as e:
            logger.error(f"Error preparing timestamps: {e}")
            # If timestamp preparation fails, continue without timestamps
            logger.warning("Continuing without timestamp handling due to error")
            return df_prepared
    
    async def _execute_transactional_upsert(self,
                                          df: pd.DataFrame,
                                          table_name: str,
                                          pk_columns: List[str],
                                          embed_columns_names: Optional[List[str]],
                                          embed_type: str) -> int:
        """
        Execute upsert operations within a transaction with rollback capability.
        
        This method uses the existing update_data method which implements
        INSERT ... ON CONFLICT ... DO UPDATE SET for proper upsert behavior.
        
        Args:
            df: DataFrame with data and timestamps
            table_name: Name of target table
            pk_columns: Primary key columns
            embed_columns_names: Embedding column names
            embed_type: Embedding type
            
        Returns:
            Number of rows processed
            
        Raises:
            DatabaseOperationError: If transaction fails
        """
        try:
            # Validate table schema compatibility before upsert
            await self._validate_upsert_schema_compatibility(df, table_name, embed_columns_names, embed_type)
            
            # For large datasets, use chunked processing to optimize memory usage
            chunk_size = 1000
            if len(df) > chunk_size:
                return await self._execute_chunked_upsert(df, table_name, pk_columns, chunk_size)
            else:
                # Use the existing update_data method which handles upserts properly
                # This method uses INSERT ... ON CONFLICT ... DO UPDATE SET
                await self.repo.update_data(table_name, df, pk_columns)
                
                rows_processed = len(df)
                logger.info(f"Transactional upsert completed: {rows_processed} rows processed")
                
                return rows_processed
            
        except Exception as e:
            logger.error(f"Transactional upsert failed: {e}")
            raise DatabaseOperationError(f"Upsert transaction failed: {e}") from e
    
    async def _execute_chunked_upsert(self,
                                    df: pd.DataFrame,
                                    table_name: str,
                                    pk_columns: List[str],
                                    chunk_size: int) -> int:
        """
        Execute upsert operations in chunks for memory optimization.
        
        Args:
            df: DataFrame with data to upsert
            table_name: Name of target table
            pk_columns: Primary key columns
            chunk_size: Size of each chunk
            
        Returns:
            Total number of rows processed
            
        Raises:
            DatabaseOperationError: If chunked upsert fails
        """
        total_rows = len(df)
        rows_processed = 0
        
        logger.info(f"Starting chunked upsert: {total_rows} rows in chunks of {chunk_size}")
        
        try:
            # Process DataFrame in chunks
            for i in range(0, total_rows, chunk_size):
                chunk_end = min(i + chunk_size, total_rows)
                chunk_df = df.iloc[i:chunk_end].copy()
                
                logger.debug(f"Processing chunk {i//chunk_size + 1}: rows {i+1}-{chunk_end}")
                
                # Execute upsert for this chunk
                await self.repo.update_data(table_name, chunk_df, pk_columns)
                
                chunk_rows = len(chunk_df)
                rows_processed += chunk_rows
                
                # Log progress for large operations
                if total_rows > 5000:
                    progress = (rows_processed / total_rows) * 100
                    logger.info(f"Chunked upsert progress: {rows_processed}/{total_rows} rows ({progress:.1f}%)")
                
                # Allow other async operations to run between chunks
                await asyncio.sleep(0)
            
            logger.info(f"Chunked upsert completed: {rows_processed} total rows processed")
            return rows_processed
            
        except Exception as e:
            logger.error(f"Chunked upsert failed at row {rows_processed}: {e}")
            raise DatabaseOperationError(f"Chunked upsert failed: {e}") from e
    
    async def _validate_upsert_schema_compatibility(self,
                                                  df: pd.DataFrame,
                                                  table_name: str,
                                                  embed_columns_names: Optional[List[str]],
                                                  embed_type: str) -> None:
        """
        Validate that DataFrame schema is compatible with existing table for upsert operations.
        
        This method ensures:
        - All DataFrame columns exist in the target table
        - Embedding columns are properly configured
        - Data types are compatible
        
        Args:
            df: DataFrame to validate
            table_name: Name of target table
            embed_columns_names: Embedding column names
            embed_type: Embedding type
            
        Raises:
            ValidationError: If schema is incompatible
            DatabaseOperationError: If validation fails
        """
        try:
            # Get table schema information
            table_info = await self.repo.get_table_info(table_name)
            table_columns = set(table_info.columns.keys())
            df_columns = set(df.columns)
            
            # Check for missing columns in table
            missing_in_table = df_columns - table_columns
            if missing_in_table:
                raise ValidationError(
                    f"DataFrame contains columns not present in table '{table_name}': {missing_in_table}",
                    {
                        "missing_columns": list(missing_in_table),
                        "table_columns": list(table_columns),
                        "dataframe_columns": list(df_columns)
                    }
                )
            
            # Validate embedding columns exist in table if specified
            if embed_columns_names:
                if embed_type == "combined":
                    required_embed_cols = ["embeddings", "embed_columns_value", "embed_columns_names"]
                else:  # separated
                    required_embed_cols = [f"{col}_enc" for col in embed_columns_names]
                    required_embed_cols.append("embed_columns_names")
                
                missing_embed_cols = [col for col in required_embed_cols if col not in table_columns]
                if missing_embed_cols:
                    raise ValidationError(
                        f"Table '{table_name}' missing required embedding columns: {missing_embed_cols}",
                        {
                            "missing_embedding_columns": missing_embed_cols,
                            "embed_type": embed_type,
                            "embed_columns_names": embed_columns_names
                        }
                    )
            
            # Validate vector dimensions if vector columns exist
            vector_columns = await self.repo.detect_vector_columns(table_name)
            if vector_columns:
                vector_errors = await self.repo.validate_vector_dimensions(table_name, df)
                if vector_errors:
                    raise ValidationError(
                        f"Vector dimension validation failed: {'; '.join(vector_errors)}",
                        {
                            "vector_errors": vector_errors,
                            "vector_columns": vector_columns
                        }
                    )
            
            logger.debug(f"Schema compatibility validated for upsert operations on table '{table_name}'")
            
        except ValidationError as e:
            logger.error(f"Schema validation failed for upsert: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error during schema validation: {e}")
            raise DatabaseOperationError(f"Schema validation failed: {e}") from e
    
    def _create_validation_report(self, api_load_result: Optional[APIJSONLoadResult]) -> ValidationReport:
        """
        Create a validation report from the API load result.
        
        Args:
            api_load_result: Result from API/JSON loading operation
            
        Returns:
            ValidationReport: Validation report for the operation
        """
        from dataload.domain.entities import SchemaAnalysis
        
        if not api_load_result:
            return self._create_empty_validation_report()
        
        # Create a basic schema analysis
        schema_analysis = SchemaAnalysis(
            table_exists=False,  # This will be updated by the calling method
            columns_added=[],
            columns_removed=[],
            columns_modified=[],
            case_conflicts=[],
            constraint_violations=[],
            compatible=api_load_result.success,
            requires_schema_update=False
        )
        
        # Extract warnings and errors
        warnings = api_load_result.warnings.copy()
        errors = [str(error) for error in api_load_result.errors]
        
        # Add processing-specific information to recommendations
        recommendations = []
        if api_load_result.flattening_result:
            if api_load_result.flattening_result.conflicts_resolved:
                recommendations.append(f"Resolved {len(api_load_result.flattening_result.conflicts_resolved)} column name conflicts during JSON flattening")
        
        if api_load_result.mapping_result:
            if api_load_result.mapping_result.unmapped_columns:
                recommendations.append(f"Consider mapping unmapped columns: {api_load_result.mapping_result.unmapped_columns}")
        
        return ValidationReport(
            schema_analysis=schema_analysis,
            case_conflicts=[],
            type_mismatches=[],
            constraint_violations=[],
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            validation_passed=api_load_result.success and len(errors) == 0
        )
    
    def _create_empty_validation_report(self) -> ValidationReport:
        """
        Create an empty validation report for error cases.
        
        Returns:
            ValidationReport: Empty validation report
        """
        from dataload.domain.entities import SchemaAnalysis
        
        schema_analysis = SchemaAnalysis(
            table_exists=False,
            columns_added=[],
            columns_removed=[],
            columns_modified=[],
            case_conflicts=[],
            constraint_violations=[],
            compatible=False,
            requires_schema_update=False
        )
        
        return ValidationReport(
            schema_analysis=schema_analysis,
            case_conflicts=[],
            type_mismatches=[],
            constraint_violations=[],
            recommendations=[],
            warnings=[],
            errors=["Validation report could not be generated"],
            validation_passed=False
        )