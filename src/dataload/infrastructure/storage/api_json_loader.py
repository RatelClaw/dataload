"""
APIJSONStorageLoader implementation for loading data from APIs and JSON sources.

This module provides the main APIJSONStorageLoader class that implements the
StorageLoaderInterface for API and JSON data sources with comprehensive
processing capabilities including flattening, mapping, and transformations.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import logging

from dataload.interfaces.storage_loader import StorageLoaderInterface
from dataload.domain.api_handler import APIHandler
from dataload.domain.json_flattener import JSONFlattener
from dataload.domain.column_mapper import ColumnMapper
from dataload.domain.request_body_transformer import RequestBodyTransformer
from dataload.domain.api_entities import (
    APIConfig, AuthConfig, PaginationConfig, JSONProcessingConfig,
    ColumnMappingConfig, RequestTransformationConfig, APIJSONLoadResult,
    APIError, AuthenticationError, JSONParsingError, ColumnMappingError,
    RequestTransformationError, AuthType, ValidationMode, TransformationResult, MappingResult
)
from dataload.domain.entities import DBOperationError, DataMoveError
from dataload.config import logger


class APIJSONStorageLoader(StorageLoaderInterface):
    """
    Loads data from APIs and JSON sources with advanced processing capabilities.
    
    This class integrates APIHandler, JSONFlattener, ColumnMapper, and 
    RequestBodyTransformer to provide comprehensive API/JSON data loading
    functionality while maintaining compatibility with the StorageLoaderInterface.
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
                 verify_ssl: bool = True,
                 default_headers: Optional[Dict[str, str]] = None):
        """
        Initialize APIJSONStorageLoader with configuration parameters.
        
        Args:
            base_url: Base URL for API requests
            api_token: API key for authentication
            jwt_token: JWT token for authentication  
            bearer_token: Bearer token for authentication
            username: Username for basic authentication
            password: Password for basic authentication
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            verify_ssl: Whether to verify SSL certificates
            default_headers: Default headers to include in all requests
        """
        self.base_url = base_url
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.verify_ssl = verify_ssl
        self.default_headers = default_headers or {}
        
        # Initialize domain components
        self.api_handler = APIHandler(
            base_url=base_url,
            api_token=api_token,
            jwt_token=jwt_token,
            bearer_token=bearer_token,
            username=username,
            password=password,
            timeout=timeout,
            retry_attempts=retry_attempts,
            verify_ssl=verify_ssl,
            default_headers=default_headers
        )
        
        self.json_flattener = JSONFlattener()
        self.column_mapper = ColumnMapper()
        self.request_transformer = RequestBodyTransformer()
        
        logger.info(f"APIJSONStorageLoader initialized with base_url={base_url}")
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Load a CSV file and return a pandas DataFrame.
        
        This method provides backward compatibility with existing CSV loading functionality.
        For CSV files, it delegates to pandas.read_csv with basic error handling.
        
        Args:
            path: Path to the CSV file to load
            
        Returns:
            pd.DataFrame: DataFrame containing the loaded CSV data
            
        Raises:
            FileNotFoundError: If the CSV file cannot be found
            DBOperationError: If the CSV file cannot be parsed
        """
        logger.info(f"Loading CSV file: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        try:
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded CSV: {path}, rows={len(df)}, columns={len(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file {path}: {e}")
            raise DBOperationError(f"Failed to load CSV file: {e}")
    
    async def load_json(self, source: Union[str, dict], config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Load JSON data from API, file, or raw data and return a pandas DataFrame.
        
        This method supports multiple JSON data sources:
        - API endpoints (URLs starting with http/https)
        - Local JSON files (file paths)
        - Raw JSON data (dict objects)
        
        Args:
            source: The JSON data source (URL, file path, or dict)
            config: Configuration parameters for JSON processing
                
        Returns:
            pd.DataFrame: DataFrame containing the processed JSON data
            
        Raises:
            ValueError: If the source format is invalid
            APIError: If API request fails
            JSONParsingError: If JSON cannot be parsed
            FileNotFoundError: If JSON file cannot be found
            DBOperationError: For other processing errors
        """
        logger.info(f"Loading JSON data from source type: {type(source).__name__}")
        
        try:
            # Validate and process configuration
            processed_config = self._process_config(config or {})
            
            # Determine source type and load data
            if isinstance(source, (dict, list)):
                # Raw JSON data (dict or list of dicts)
                json_data = source
                logger.debug(f"Processing raw JSON {type(source).__name__} data")
            elif isinstance(source, str):
                if source.startswith(('http://', 'https://')):
                    # API endpoint
                    json_data = await self._load_from_api(source, processed_config)
                    logger.debug(f"Loaded data from API endpoint: {source}")
                else:
                    # Local JSON file - run in executor to avoid blocking
                    json_data = await asyncio.get_event_loop().run_in_executor(
                        None, self._load_from_file, source
                    )
                    logger.debug(f"Loaded data from JSON file: {source}")
            else:
                raise ValueError(f"Unsupported source type: {type(source)}. Must be dict, list, URL string, or file path string.")
            
            # Process the JSON data through the pipeline - run in executor for CPU-intensive work
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._process_json_pipeline, json_data, processed_config
            )
            
            if result.success and result.dataframe is not None:
                logger.info(f"Successfully processed JSON data: {len(result.dataframe)} rows, {len(result.dataframe.columns)} columns")
                return result.dataframe
            else:
                error_msg = f"JSON processing failed: {result.errors}"
                logger.error(error_msg)
                raise DBOperationError(error_msg)
            
        except (APIError, JSONParsingError, ColumnMappingError, RequestTransformationError):
            # Re-raise domain-specific errors
            raise
        except FileNotFoundError:
            # Re-raise file not found errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading JSON data: {e}")
            raise DBOperationError(f"Failed to load JSON data: {e}")
    
    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate the configuration parameters.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Processed configuration dictionary with validated parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        processed = {
            # JSON processing configuration
            'flatten_nested': config.get('flatten_nested', True),
            'separator': config.get('separator', '_'),
            'max_depth': config.get('max_depth'),
            'handle_arrays': config.get('handle_arrays', 'expand'),
            'null_handling': config.get('null_handling', 'keep'),
            
            # Column mapping configuration
            'column_name_mapping': config.get('column_name_mapping', {}),
            'update_request_body_mapping': config.get('update_request_body_mapping', {}),
            'case_sensitive': config.get('case_sensitive', True),
            'validation_mode': config.get('validation_mode', 'strict'),
            
            # API configuration
            'pagination_enabled': config.get('pagination_enabled', False),
            'page_size': config.get('page_size', 100),
            'max_pages': config.get('max_pages'),
            'headers': config.get('headers', {}),
            'params': config.get('params', {}),
            'method': config.get('method', 'GET'),
            
            # Processing options
            'preserve_original_data': config.get('preserve_original_data', True),
            'fail_on_error': config.get('fail_on_error', True)
        }
        
        # Validate critical parameters
        if processed['separator'] == '':
            raise ValueError("Separator cannot be empty")
        
        if processed['max_depth'] is not None and processed['max_depth'] <= 0:
            raise ValueError("Max depth must be greater than 0")
        
        if processed['page_size'] <= 0:
            raise ValueError("Page size must be greater than 0")
        
        return processed    

    async def _load_from_api(self, url: str, config: Dict[str, Any]) -> Union[Dict, List[Dict]]:
        """
        Load JSON data from an API endpoint.
        
        Args:
            url: API endpoint URL
            config: Configuration parameters
            
        Returns:
            JSON data from the API response
            
        Raises:
            APIError: If API request fails
        """
        try:
            # Prepare pagination configuration if enabled
            pagination_config = None
            if config.get('pagination_enabled', False):
                from dataload.domain.api_entities import PaginationConfig, PaginationType
                pagination_config = PaginationConfig(
                    enabled=True,
                    pagination_type=PaginationType.PAGE_SIZE,
                    page_size=config.get('page_size', 100),
                    max_pages=config.get('max_pages')
                )
            
            # Make API request using the APIHandler
            async with self.api_handler as handler:
                response = await handler.fetch_data(
                    endpoint=url,
                    method=config.get('method', 'GET'),
                    headers=config.get('headers', {}),
                    params=config.get('params', {}),
                    pagination_config=pagination_config
                )
            
            # Handle paginated responses
            if isinstance(response, list):
                # Multiple pages - combine all data
                combined_data = []
                for page_response in response:
                    if isinstance(page_response.data, list):
                        combined_data.extend(page_response.data)
                    elif isinstance(page_response.data, dict):
                        # If response is a dict, look for common data keys
                        if 'data' in page_response.data and isinstance(page_response.data['data'], list):
                            combined_data.extend(page_response.data['data'])
                        elif 'results' in page_response.data and isinstance(page_response.data['results'], list):
                            combined_data.extend(page_response.data['results'])
                        elif 'items' in page_response.data and isinstance(page_response.data['items'], list):
                            combined_data.extend(page_response.data['items'])
                        else:
                            combined_data.append(page_response.data)
                return combined_data
            else:
                # Single response
                return response.data
                
        except Exception as e:
            if isinstance(e, (APIError, AuthenticationError)):
                raise
            raise APIError(f"Failed to load data from API: {e}")
    
    def _load_from_file(self, file_path: str) -> Union[Dict, List[Dict]]:
        """
        Load JSON data from a local file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            JSON data from the file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            JSONParsingError: If JSON parsing fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            return json_data
        except json.JSONDecodeError as e:
            raise JSONParsingError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            raise JSONParsingError(f"Failed to read JSON file {file_path}: {e}")
    
    def _process_json_pipeline(self, json_data: Union[Dict, List[Dict]], config: Dict[str, Any]) -> 'APIJSONLoadResult':
        """
        Process JSON data through the complete pipeline.
        
        Args:
            json_data: Raw JSON data to process
            config: Configuration parameters
            
        Returns:
            APIJSONLoadResult with processed DataFrame and metadata
            
        Raises:
            JSONParsingError: If JSON processing fails
            ColumnMappingError: If column mapping fails
            RequestTransformationError: If data transformation fails
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Step 1: Flatten JSON structure
            logger.debug("Starting JSON flattening")
            json_config = JSONProcessingConfig(
                flatten_nested=config.get('flatten_nested', True),
                separator=config.get('separator', '_'),
                max_depth=config.get('max_depth'),
                handle_arrays=getattr(__import__('dataload.domain.api_entities', fromlist=['ArrayHandlingStrategy']).ArrayHandlingStrategy, 
                                    config.get('handle_arrays', 'expand').upper()),
                null_handling=getattr(__import__('dataload.domain.api_entities', fromlist=['NullHandlingStrategy']).NullHandlingStrategy,
                                    config.get('null_handling', 'keep').upper())
            )
            
            self.json_flattener.config = json_config
            flattening_result = self.json_flattener.flatten_json(json_data)
            
            if not flattening_result.success:
                raise JSONParsingError("JSON flattening failed")
            
            warnings.extend(flattening_result.warnings)
            df = flattening_result.dataframe
            
            # Step 2: Apply request body transformations (if configured)
            transformation_result = None
            if config.get('update_request_body_mapping'):
                logger.debug("Applying request body transformations")
                transformation_result = self._apply_transformations(df, config)
                df = transformation_result.transformed_dataframe
                warnings.extend(transformation_result.warnings)
            
            # Step 3: Apply column mappings (if configured)
            mapping_result = None
            if config.get('column_name_mapping'):
                logger.debug("Applying column mappings")
                mapping_result = self._apply_column_mapping(df, config)
                df = mapping_result.mapped_dataframe
                warnings.extend(mapping_result.warnings)
            
            execution_time = time.time() - start_time
            
            # Create comprehensive result
            result = APIJSONLoadResult(
                success=True,
                dataframe=df,
                rows_loaded=len(df),
                columns_created=len(df.columns),
                execution_time=execution_time,
                api_responses=[],  # Will be populated by calling method if needed
                flattening_result=flattening_result,
                mapping_result=mapping_result,
                transformation_result=transformation_result,
                errors=errors,
                warnings=warnings,
                operation_metadata={
                    'source_type': 'json_processing',
                    'config_used': config,
                    'processing_steps': ['flatten', 'transform', 'map'] if transformation_result and mapping_result 
                                      else ['flatten', 'map'] if mapping_result 
                                      else ['flatten', 'transform'] if transformation_result 
                                      else ['flatten']
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            errors.append(e)
            
            # Return failed result
            return APIJSONLoadResult(
                success=False,
                dataframe=None,
                rows_loaded=0,
                columns_created=0,
                execution_time=execution_time,
                api_responses=[],
                flattening_result=None,
                mapping_result=None,
                transformation_result=None,
                errors=errors,
                warnings=warnings,
                operation_metadata={'source_type': 'json_processing', 'config_used': config}
            )
    
    def _apply_transformations(self, df: pd.DataFrame, config: Dict[str, Any]) -> 'TransformationResult':
        """
        Apply request body transformations to the DataFrame.
        
        Args:
            df: DataFrame to transform
            config: Configuration containing transformation mappings
            
        Returns:
            TransformationResult with transformed DataFrame and metadata
        """
        from dataload.domain.api_entities import RequestTransformationConfig, TransformationRule
        
        # Convert update_request_body_mapping to transformation rules
        transformation_rules = []
        update_mapping = config.get('update_request_body_mapping', {})
        
        for target_field, source_expression in update_mapping.items():
            # Determine transformation type based on expression
            if source_expression.startswith('concat('):
                transformation_type = 'concat'
            elif any(op in source_expression for op in ['+', '-', '*', '/', 'round(', 'upper(', 'lower(']):
                transformation_type = 'compute'
            elif '{' in source_expression and '}' in source_expression:
                transformation_type = 'copy'
            else:
                transformation_type = 'constant'
            
            rule = TransformationRule(
                target_field=target_field,
                source_expression=source_expression,
                transformation_type=transformation_type,
                required=not config.get('fail_on_error', True)
            )
            transformation_rules.append(rule)
        
        # Create transformation configuration
        transformation_config = RequestTransformationConfig(
            transformation_rules=transformation_rules,
            execution_order=[rule.target_field for rule in transformation_rules],
            fail_on_error=config.get('fail_on_error', True),
            preserve_original_data=config.get('preserve_original_data', True)
        )
        
        # Apply transformations
        return self.request_transformer.transform_data(df, transformation_config)
    
    def _apply_column_mapping(self, df: pd.DataFrame, config: Dict[str, Any]) -> 'MappingResult':
        """
        Apply column name mappings to the DataFrame.
        
        Args:
            df: DataFrame to map
            config: Configuration containing column mappings
            
        Returns:
            MappingResult with mapped DataFrame and metadata
        """
        from dataload.domain.api_entities import ColumnMappingConfig, ValidationMode
        
        # Create column mapping configuration
        validation_mode_str = config.get('validation_mode', 'strict')
        validation_mode = ValidationMode.STRICT if validation_mode_str == 'strict' else \
                         ValidationMode.LENIENT if validation_mode_str == 'lenient' else \
                         ValidationMode.IGNORE
        
        mapping_config = ColumnMappingConfig(
            column_name_mapping=config.get('column_name_mapping', {}),
            update_request_body_mapping=config.get('update_request_body_mapping', {}),
            validation_mode=validation_mode,
            case_sensitive=config.get('case_sensitive', True),
            allow_missing_columns=not config.get('fail_on_error', True),
            preserve_unmapped_columns=True
        )
        
        # Apply column mappings
        return self.column_mapper.apply_mapping(df, mapping_config)
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration parameters and return any validation errors.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            # Validate basic configuration
            self._process_config(config)
        except ValueError as e:
            errors.append(str(e))
        
        # Validate column mappings
        column_mapping = config.get('column_name_mapping', {})
        if column_mapping:
            if not isinstance(column_mapping, dict):
                errors.append("column_name_mapping must be a dictionary")
            else:
                for source, target in column_mapping.items():
                    if not isinstance(source, str) or not isinstance(target, str):
                        errors.append(f"Column mapping keys and values must be strings: {source} -> {target}")
                    if not source.strip() or not target.strip():
                        errors.append(f"Column names cannot be empty: '{source}' -> '{target}'")
        
        # Validate transformation mappings
        transform_mapping = config.get('update_request_body_mapping', {})
        if transform_mapping:
            if not isinstance(transform_mapping, dict):
                errors.append("update_request_body_mapping must be a dictionary")
            else:
                for target, expression in transform_mapping.items():
                    if not isinstance(target, str) or not isinstance(expression, str):
                        errors.append(f"Transformation mapping keys and values must be strings: {target} -> {expression}")
                    if not target.strip():
                        errors.append(f"Target field name cannot be empty: '{target}'")
        
        # Validate API configuration
        if config.get('pagination_enabled', False):
            page_size = config.get('page_size', 100)
            if not isinstance(page_size, int) or page_size <= 0:
                errors.append("page_size must be a positive integer")
            
            max_pages = config.get('max_pages')
            if max_pages is not None and (not isinstance(max_pages, int) or max_pages <= 0):
                errors.append("max_pages must be a positive integer or None")
        
        return errors
    
    async def load_json_concurrent(self, 
                                  sources: List[Union[str, dict]], 
                                  config: Optional[Dict] = None,
                                  max_concurrent: int = 5) -> pd.DataFrame:
        """
        Load JSON data from multiple sources concurrently and combine results.
        
        This method enables concurrent processing of multiple API endpoints or data sources
        for improved performance when dealing with multiple data sources.
        
        Args:
            sources: List of JSON data sources (URLs, file paths, or dict objects)
            config: Configuration parameters for JSON processing
            max_concurrent: Maximum number of concurrent operations (default: 5)
                
        Returns:
            pd.DataFrame: Combined DataFrame containing data from all sources
            
        Raises:
            ValueError: If sources list is empty or contains invalid sources
            APIError: If any API request fails
            JSONParsingError: If JSON cannot be parsed
            DBOperationError: For other processing errors
        """
        if not sources:
            raise ValueError("Sources list cannot be empty")
        
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")
        
        logger.info(f"Loading JSON data from {len(sources)} sources concurrently (max_concurrent={max_concurrent})")
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def load_single_source(source: Union[str, dict]) -> pd.DataFrame:
            """Load data from a single source with semaphore control."""
            async with semaphore:
                try:
                    return await self.load_json(source, config)
                except Exception as e:
                    logger.error(f"Failed to load from source {source}: {e}")
                    # Re-raise with source context
                    raise APIError(f"Failed to load from source {source}: {e}") from e
        
        try:
            # Execute all loads concurrently
            tasks = [load_single_source(source) for source in sources]
            dataframes = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            successful_dfs = []
            errors = []
            
            for i, result in enumerate(dataframes):
                if isinstance(result, Exception):
                    errors.append(f"Source {i} ({sources[i]}): {result}")
                elif isinstance(result, pd.DataFrame):
                    if not result.empty:
                        successful_dfs.append(result)
                    else:
                        logger.warning(f"Source {i} ({sources[i]}) returned empty DataFrame")
                else:
                    errors.append(f"Source {i} ({sources[i]}): Unexpected result type {type(result)}")
            
            # Check if we have any successful results
            if not successful_dfs:
                error_msg = f"All {len(sources)} sources failed to load data"
                if errors:
                    error_msg += f": {'; '.join(errors)}"
                logger.error(error_msg)
                raise DBOperationError(error_msg)
            
            # Log any partial failures
            if errors:
                logger.warning(f"Some sources failed to load ({len(errors)}/{len(sources)}): {'; '.join(errors)}")
            
            # Combine all successful DataFrames
            if len(successful_dfs) == 1:
                combined_df = successful_dfs[0]
            else:
                # Ensure all DataFrames have compatible schemas before concatenating
                combined_df = await self._combine_dataframes_safely(successful_dfs)
            
            logger.info(f"Successfully loaded and combined data from {len(successful_dfs)}/{len(sources)} sources: "
                       f"{len(combined_df)} total rows, {len(combined_df.columns)} columns")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Concurrent loading failed: {e}")
            raise DBOperationError(f"Failed to load data concurrently: {e}") from e
    
    async def _combine_dataframes_safely(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Safely combine multiple DataFrames with schema compatibility checks.
        
        Args:
            dataframes: List of DataFrames to combine
            
        Returns:
            Combined DataFrame
            
        Raises:
            DBOperationError: If DataFrames have incompatible schemas
        """
        if not dataframes:
            return pd.DataFrame()
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        try:
            # Check schema compatibility
            base_columns = set(dataframes[0].columns)
            schema_conflicts = []
            
            for i, df in enumerate(dataframes[1:], 1):
                df_columns = set(df.columns)
                if df_columns != base_columns:
                    missing_in_base = df_columns - base_columns
                    missing_in_current = base_columns - df_columns
                    
                    if missing_in_base or missing_in_current:
                        schema_conflicts.append({
                            'dataframe_index': i,
                            'missing_in_base': list(missing_in_base),
                            'missing_in_current': list(missing_in_current)
                        })
            
            if schema_conflicts:
                # Try to align schemas by adding missing columns with null values
                logger.warning(f"Schema conflicts detected, attempting to align schemas: {schema_conflicts}")
                
                # Get union of all columns
                all_columns = set()
                for df in dataframes:
                    all_columns.update(df.columns)
                
                # Align all DataFrames to have the same columns
                aligned_dfs = []
                for df in dataframes:
                    aligned_df = df.copy()
                    for col in all_columns:
                        if col not in aligned_df.columns:
                            aligned_df[col] = None
                    # Reorder columns consistently
                    aligned_df = aligned_df[sorted(all_columns)]
                    aligned_dfs.append(aligned_df)
                
                dataframes = aligned_dfs
                logger.info(f"Successfully aligned {len(dataframes)} DataFrames with {len(all_columns)} columns")
            
            # Combine DataFrames
            combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
            
            # Log combination statistics
            total_rows = sum(len(df) for df in dataframes)
            logger.debug(f"Combined {len(dataframes)} DataFrames: {total_rows} total rows -> {len(combined_df)} final rows")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Failed to combine DataFrames: {e}")
            raise DBOperationError(f"Schema alignment failed: {e}") from e

    async def close(self):
        """Close any open connections and clean up resources."""
        if hasattr(self.api_handler, 'close'):
            await self.api_handler.close()
        logger.debug("APIJSONStorageLoader closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Note: This is synchronous, so we can't call async close() here
        # Users should use async context manager or call close() explicitly
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()