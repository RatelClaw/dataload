"""
JSONFlattener for processing nested JSON structures.

This module provides functionality to flatten complex nested JSON structures
into tabular format suitable for database storage and processing.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import json
import logging
from collections import defaultdict

from .api_entities import (
    JSONProcessingConfig,
    FlatteningResult,
    JSONParsingError,
    ArrayHandlingStrategy,
    NullHandlingStrategy,
    DuplicateKeyStrategy
)

logger = logging.getLogger(__name__)


class JSONFlattener:
    """
    Flattens complex nested JSON structures into tabular format.
    
    Handles nested objects, arrays, and mixed data types with configurable
    processing options and conflict resolution strategies.
    """
    
    def __init__(self, config: Optional[JSONProcessingConfig] = None):
        """
        Initialize JSONFlattener with configuration.
        
        Args:
            config: JSON processing configuration. If None, uses default settings.
        """
        self.config = config or JSONProcessingConfig()
        self._column_counter = defaultdict(int)
        self._warnings = []
        self._conflicts_resolved = []
        
    def flatten_json(
        self,
        json_data: Union[Dict[str, Any], List[Dict[str, Any]], str],
        separator: Optional[str] = None,
        max_depth: Optional[int] = None
    ) -> FlatteningResult:
        """
        Flatten JSON data into a pandas DataFrame.
        
        Args:
            json_data: JSON data to flatten (dict, list of dicts, or JSON string)
            separator: Custom separator for nested keys (overrides config)
            max_depth: Maximum depth for flattening (overrides config)
            
        Returns:
            FlatteningResult containing the flattened DataFrame and metadata
            
        Raises:
            JSONParsingError: If JSON parsing or processing fails
        """
        try:
            # Reset state for new operation
            self._column_counter.clear()
            self._warnings.clear()
            self._conflicts_resolved.clear()
            
            # Use provided parameters or fall back to config
            sep = separator if separator is not None else self.config.separator
            depth = max_depth if max_depth is not None else self.config.max_depth
            
            # Parse JSON string if needed
            if isinstance(json_data, str):
                try:
                    json_data = json.loads(json_data)
                except json.JSONDecodeError as e:
                    raise JSONParsingError(f"Invalid JSON string: {e}")
            
            # Ensure we have a list of records
            if isinstance(json_data, dict):
                records = [json_data]
                original_structure = json_data
            elif isinstance(json_data, list):
                records = json_data
                original_structure = {"records": json_data}
            else:
                raise JSONParsingError(f"Unsupported JSON data type: {type(json_data)}")
            
            if not records:
                return self._create_empty_result(original_structure)
            
            # Handle empty dict case
            if len(records) == 1 and not records[0]:
                return self._create_empty_result(original_structure)
            
            # Flatten each record
            flattened_records = []
            for i, record in enumerate(records):
                try:
                    flattened_record = self._flatten_record(record, sep, depth)
                    flattened_records.append(flattened_record)
                except Exception as e:
                    logger.warning(f"Failed to flatten record {i}: {e}")
                    self._warnings.append(f"Skipped record {i}: {e}")
                    continue
            
            if not flattened_records:
                return self._create_empty_result(original_structure)
            
            # Create DataFrame
            df = pd.DataFrame(flattened_records)
            
            # Handle null values according to strategy
            df = self._handle_null_values(df)
            
            # Normalize column names if requested
            if self.config.normalize_column_names:
                df = self._normalize_column_names(df, sep)
            
            # Get final column list
            flattened_columns = list(df.columns)
            
            # Create processing stats
            processing_stats = {
                "total_records": len(records),
                "processed_records": len(flattened_records),
                "skipped_records": len(records) - len(flattened_records),
                "total_columns": len(flattened_columns),
                "conflicts_resolved": len(self._conflicts_resolved),
                "warnings_count": len(self._warnings)
            }
            
            return FlatteningResult(
                dataframe=df,
                original_structure=original_structure,
                flattened_columns=flattened_columns,
                conflicts_resolved=self._conflicts_resolved.copy(),
                warnings=self._warnings.copy(),
                processing_stats=processing_stats
            )
            
        except Exception as e:
            if isinstance(e, JSONParsingError):
                raise
            raise JSONParsingError(f"Failed to flatten JSON: {e}")
    
    def _flatten_record(
        self,
        record: Dict[str, Any],
        separator: str,
        max_depth: Optional[int],
        current_depth: int = 0,
        parent_key: str = ""
    ) -> Dict[str, Any]:
        """
        Recursively flatten a single JSON record.
        
        Args:
            record: The record to flatten
            separator: Separator for nested keys
            max_depth: Maximum depth for flattening
            current_depth: Current recursion depth
            parent_key: Parent key prefix
            
        Returns:
            Flattened record as a dictionary
        """
        flattened = {}
        
        for key, value in record.items():
            # Create the new key
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            
            if isinstance(value, dict) and self.config.flatten_nested:
                # Check depth limit before recursing
                if max_depth is not None and current_depth >= max_depth:
                    # Convert to string if we've reached max depth
                    flattened[new_key] = self._serialize_value(value)
                else:
                    # Recursively flatten nested objects
                    nested_flattened = self._flatten_record(
                        value, separator, max_depth, current_depth + 1, new_key
                    )
                    flattened.update(nested_flattened)
                
            elif isinstance(value, list):
                # Check depth limit for arrays too
                if max_depth is not None and current_depth >= max_depth:
                    flattened[new_key] = self._serialize_value(value)
                else:
                    # Handle arrays according to strategy
                    flattened.update(
                        self._handle_array(new_key, value, separator, max_depth, current_depth)
                    )
                
            else:
                # Handle primitive values (including when flatten_nested is False for dicts)
                if isinstance(value, dict) and not self.config.flatten_nested:
                    flattened[new_key] = self._serialize_value(value)
                else:
                    flattened[new_key] = value
        
        return flattened
    
    def _handle_array(
        self,
        key: str,
        array: List[Any],
        separator: str,
        max_depth: Optional[int],
        current_depth: int
    ) -> Dict[str, Any]:
        """
        Handle array values according to the configured strategy.
        
        Args:
            key: The key for this array
            array: The array to process
            separator: Separator for nested keys
            max_depth: Maximum depth for flattening
            current_depth: Current recursion depth
            
        Returns:
            Dictionary with processed array data
        """
        if self.config.handle_arrays == ArrayHandlingStrategy.IGNORE:
            return {}
        
        if self.config.handle_arrays == ArrayHandlingStrategy.JOIN:
            # Join array elements into a single string
            joined_value = ", ".join(str(item) for item in array)
            return {key: joined_value}
        
        if self.config.handle_arrays == ArrayHandlingStrategy.EXPAND:
            # Expand array into separate columns or rows
            result = {}
            
            for i, item in enumerate(array):
                item_key = f"{key}{separator}{i}"
                
                if isinstance(item, dict) and self.config.flatten_nested:
                    # Flatten nested objects in array
                    nested_flattened = self._flatten_record(
                        item, separator, max_depth, current_depth + 1, item_key
                    )
                    result.update(nested_flattened)
                else:
                    result[item_key] = item
            
            return result
        
        # Default: convert to string
        return {key: str(array)}
    
    def _handle_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle null values according to the configured strategy.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with null values handled
        """
        if self.config.null_handling == NullHandlingStrategy.DROP:
            # Drop rows with any null values
            original_count = len(df)
            df = df.dropna()
            dropped_count = original_count - len(df)
            if dropped_count > 0:
                self._warnings.append(f"Dropped {dropped_count} rows with null values")
                
        elif self.config.null_handling == NullHandlingStrategy.FILL:
            # Fill null values with appropriate defaults based on inferred type
            for column in df.columns:
                # Check if column contains numeric data (excluding nulls)
                non_null_values = df[column].dropna()
                if len(non_null_values) > 0:
                    # Try to determine if it's numeric
                    try:
                        pd.to_numeric(non_null_values.iloc[0])
                        # If numeric, fill with 0
                        df[column] = df[column].fillna(0)
                    except (ValueError, TypeError):
                        # If not numeric, fill with empty string
                        df[column] = df[column].fillna('')
                else:
                    # If all values are null, fill with empty string
                    df[column] = df[column].fillna('')
        
        # KEEP strategy: do nothing, preserve null values
        return df
    
    def _normalize_column_names(self, df: pd.DataFrame, separator: str = "_") -> pd.DataFrame:
        """
        Normalize column names and handle duplicates.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with normalized column names
        """
        new_columns = []
        column_counts = defaultdict(int)
        
        for col in df.columns:
            # Basic normalization: lowercase, replace spaces and hyphens with underscores
            normalized = str(col).lower().replace(' ', '_').replace('-', '_')
            
            # Preserve the separator that was used for flattening
            allowed_chars = ['_', separator] if separator != '_' else ['_']
            normalized = ''.join(c for c in normalized if c.isalnum() or c in allowed_chars)
            
            # Ensure it doesn't start with a number
            if normalized and normalized[0].isdigit():
                normalized = f"col_{normalized}"
            
            # Handle empty names
            if not normalized:
                normalized = f"col_{len(new_columns)}"
            
            # Handle duplicates
            if normalized in column_counts:
                if self.config.duplicate_key_strategy == DuplicateKeyStrategy.ERROR:
                    raise JSONParsingError(f"Duplicate column name after normalization: {normalized}")
                elif self.config.duplicate_key_strategy == DuplicateKeyStrategy.SUFFIX:
                    column_counts[normalized] += 1
                    original_normalized = normalized
                    normalized = f"{normalized}_{column_counts[normalized]}"
                    self._conflicts_resolved.append(f"Renamed duplicate column '{original_normalized}' to '{normalized}'")
                # OVERWRITE strategy: keep the last occurrence (do nothing)
            
            column_counts[normalized] += 1
            new_columns.append(normalized)
        
        df.columns = new_columns
        return df
    
    def _serialize_value(self, value: Any) -> str:
        """
        Serialize a complex value to string.
        
        Args:
            value: Value to serialize
            
        Returns:
            String representation of the value
        """
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(value)
        return str(value)
    
    def _create_empty_result(self, original_structure: Dict[str, Any]) -> FlatteningResult:
        """
        Create an empty FlatteningResult for cases with no data.
        
        Args:
            original_structure: The original JSON structure
            
        Returns:
            Empty FlatteningResult
        """
        return FlatteningResult(
            dataframe=pd.DataFrame(),
            original_structure=original_structure,
            flattened_columns=[],
            conflicts_resolved=[],
            warnings=["No data to flatten"],
            processing_stats={
                "total_records": 0,
                "processed_records": 0,
                "skipped_records": 0,
                "total_columns": 0,
                "conflicts_resolved": 0,
                "warnings_count": 1
            }
        )
    
    def validate_json_structure(self, json_data: Union[Dict, List, str]) -> Tuple[bool, List[str]]:
        """
        Validate JSON structure before flattening.
        
        Args:
            json_data: JSON data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Parse JSON string if needed
            if isinstance(json_data, str):
                try:
                    json_data = json.loads(json_data)
                except json.JSONDecodeError as e:
                    return False, [f"Invalid JSON: {e}"]
            
            # Check if it's a supported structure
            if not isinstance(json_data, (dict, list)):
                issues.append(f"Unsupported JSON type: {type(json_data)}")
                return False, issues
            
            # If it's a list, check that all elements are dictionaries
            if isinstance(json_data, list):
                if not json_data:
                    issues.append("Empty array provided")
                    return False, issues
                
                for i, item in enumerate(json_data):
                    if not isinstance(item, dict):
                        issues.append(f"Array element {i} is not an object: {type(item)}")
            
            # Check for extremely deep nesting
            max_depth = self._calculate_max_depth(json_data)
            if max_depth > 20:  # Arbitrary limit
                issues.append(f"JSON structure is very deeply nested (depth: {max_depth})")
                # Still return True but with warning
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Validation error: {e}"]
    
    def _calculate_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """
        Calculate the maximum depth of a nested structure.
        
        Args:
            obj: Object to analyze
            current_depth: Current depth level
            
        Returns:
            Maximum depth found
        """
        if not isinstance(obj, (dict, list)):
            return current_depth
        
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(
                self._calculate_max_depth(value, current_depth + 1)
                for value in obj.values()
            )
        
        if isinstance(obj, list):
            if not obj:
                return current_depth
            return max(
                self._calculate_max_depth(item, current_depth + 1)
                for item in obj
            )
        
        return current_depth