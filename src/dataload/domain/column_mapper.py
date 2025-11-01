"""
Column mapping functionality for transforming field names.

This module provides the ColumnMapper class for applying column name mappings
with validation, conflict detection, and comprehensive error handling.
"""

import re
from typing import Dict, List, Set, Optional, Tuple, Any
import pandas as pd
import logging

from .api_entities import (
    ColumnMappingConfig, 
    MappingResult, 
    ColumnMappingError,
    ValidationMode
)

logger = logging.getLogger(__name__)


class ColumnMapper:
    """
    Handles column name mapping and validation for API/JSON data processing.
    
    This class applies column name transformations while detecting conflicts,
    validating mappings, and providing detailed error reporting.
    """
    
    def __init__(self):
        """Initialize the ColumnMapper."""
        self._reserved_keywords = {
            'select', 'from', 'where', 'insert', 'update', 'delete', 'create', 'drop',
            'alter', 'table', 'index', 'view', 'database', 'schema', 'primary', 'key',
            'foreign', 'references', 'constraint', 'unique', 'not', 'null', 'default',
            'check', 'order', 'by', 'group', 'having', 'union', 'join', 'inner',
            'outer', 'left', 'right', 'on', 'as', 'distinct', 'count', 'sum', 'avg',
            'min', 'max', 'and', 'or', 'in', 'exists', 'between', 'like', 'is'
        }
    
    def apply_mapping(
        self, 
        df: pd.DataFrame, 
        config: ColumnMappingConfig
    ) -> MappingResult:
        """
        Apply column name mappings to a DataFrame with validation and conflict detection.
        
        Args:
            df: The DataFrame to apply mappings to
            config: Configuration containing mapping rules and validation settings
            
        Returns:
            MappingResult containing the mapped DataFrame and operation details
            
        Raises:
            ColumnMappingError: When validation fails in strict mode
        """
        logger.info(f"Applying column mapping to DataFrame with {len(df.columns)} columns")
        
        # Initialize result tracking
        applied_mappings: Dict[str, str] = {}
        unmapped_columns: List[str] = []
        conflicts: List[str] = []
        warnings: List[str] = []
        
        # Validate configuration
        self._validate_config(config, df.columns.tolist())
        
        # Create a copy of the DataFrame to avoid modifying the original
        mapped_df = df.copy()
        
        try:
            # Step 1: Validate source columns exist
            missing_sources = self._check_missing_source_columns(
                df.columns.tolist(), 
                config.column_name_mapping,
                config.case_sensitive
            )
            
            if missing_sources:
                message = f"Source columns not found: {missing_sources}"
                if config.validation_mode == ValidationMode.STRICT:
                    conflicts.append(message)
                    if not config.allow_missing_columns:
                        raise ColumnMappingError(message)
                else:
                    warnings.append(message)
            
            # Step 2: Check for target column conflicts
            target_conflicts = self._check_target_conflicts(
                df.columns.tolist(),
                config.column_name_mapping,
                config.case_sensitive
            )
            
            if target_conflicts:
                conflict_msg = f"Target column conflicts detected: {target_conflicts}"
                conflicts.append(conflict_msg)
                if config.validation_mode == ValidationMode.STRICT:
                    raise ColumnMappingError(conflict_msg)
                else:
                    warnings.append(conflict_msg)
            
            # Step 3: Validate target column names
            invalid_targets = self._validate_target_column_names(config.column_name_mapping)
            
            if invalid_targets:
                invalid_msg = f"Invalid target column names: {invalid_targets}"
                conflicts.append(invalid_msg)
                if config.validation_mode == ValidationMode.STRICT:
                    raise ColumnMappingError(invalid_msg)
                else:
                    warnings.append(invalid_msg)
            
            # Step 4: Apply mappings
            successful_mappings = self._apply_column_mappings(
                mapped_df,
                config.column_name_mapping,
                config.case_sensitive
            )
            
            applied_mappings.update(successful_mappings)
            
            # Step 5: Identify unmapped columns
            mapped_sources = set(successful_mappings.keys())
            if config.case_sensitive:
                unmapped_columns = [col for col in df.columns if col not in mapped_sources]
            else:
                mapped_sources_lower = {col.lower() for col in mapped_sources}
                unmapped_columns = [
                    col for col in df.columns 
                    if col.lower() not in mapped_sources_lower
                ]
            
            # Step 6: Handle unmapped columns based on configuration
            if not config.preserve_unmapped_columns:
                columns_to_drop = [col for col in unmapped_columns if col in mapped_df.columns]
                if columns_to_drop:
                    mapped_df = mapped_df.drop(columns=columns_to_drop)
                    warnings.append(f"Dropped unmapped columns: {columns_to_drop}")
            
            # Generate transformation statistics
            transformation_stats = {
                'total_columns_input': len(df.columns),
                'total_columns_output': len(mapped_df.columns),
                'mappings_applied': len(applied_mappings),
                'columns_unmapped': len(unmapped_columns),
                'conflicts_detected': len(conflicts),
                'warnings_generated': len(warnings)
            }
            
            logger.info(f"Column mapping completed. Applied {len(applied_mappings)} mappings, "
                       f"{len(conflicts)} conflicts, {len(warnings)} warnings")
            
            return MappingResult(
                mapped_dataframe=mapped_df,
                applied_mappings=applied_mappings,
                unmapped_columns=unmapped_columns,
                conflicts=conflicts,
                warnings=warnings,
                transformation_stats=transformation_stats
            )
            
        except Exception as e:
            logger.error(f"Error during column mapping: {str(e)}")
            if isinstance(e, ColumnMappingError):
                raise
            else:
                raise ColumnMappingError(f"Unexpected error during column mapping: {str(e)}")
    
    def _validate_config(self, config: ColumnMappingConfig, df_columns: List[str]) -> None:
        """Validate the column mapping configuration."""
        if not config.column_name_mapping:
            return
        
        # Check for empty source or target names
        for source, target in config.column_name_mapping.items():
            if not source or not source.strip():
                raise ColumnMappingError("Source column name cannot be empty")
            if not target or not target.strip():
                raise ColumnMappingError("Target column name cannot be empty")
        
        # Check for self-mappings
        self_mappings = [
            source for source, target in config.column_name_mapping.items()
            if source == target
        ]
        if self_mappings:
            raise ColumnMappingError(f"Self-mappings detected: {self_mappings}")
    
    def _check_missing_source_columns(
        self, 
        df_columns: List[str], 
        mapping: Dict[str, str],
        case_sensitive: bool
    ) -> List[str]:
        """Check for source columns that don't exist in the DataFrame."""
        if case_sensitive:
            df_columns_set = set(df_columns)
            return [source for source in mapping.keys() if source not in df_columns_set]
        else:
            df_columns_lower = {col.lower() for col in df_columns}
            return [
                source for source in mapping.keys() 
                if source.lower() not in df_columns_lower
            ]
    
    def _check_target_conflicts(
        self, 
        df_columns: List[str], 
        mapping: Dict[str, str],
        case_sensitive: bool
    ) -> List[str]:
        """Check for conflicts where target names already exist as unmapped columns."""
        conflicts = []
        
        # Get columns that are not being mapped (sources)
        if case_sensitive:
            unmapped_columns = set(df_columns) - set(mapping.keys())
            target_names = set(mapping.values())
        else:
            mapped_sources_lower = {source.lower() for source in mapping.keys()}
            unmapped_columns = {
                col for col in df_columns 
                if col.lower() not in mapped_sources_lower
            }
            target_names = {target.lower() for target in mapping.values()}
        
        # Check for conflicts
        for target in mapping.values():
            target_check = target.lower() if not case_sensitive else target
            
            if case_sensitive:
                if target in unmapped_columns:
                    conflicts.append(f"Target '{target}' conflicts with existing unmapped column")
            else:
                conflicting_cols = [
                    col for col in unmapped_columns 
                    if col.lower() == target_check
                ]
                if conflicting_cols:
                    conflicts.append(
                        f"Target '{target}' conflicts with existing unmapped column(s): {conflicting_cols}"
                    )
        
        # Check for duplicate target names
        target_list = list(mapping.values())
        if case_sensitive:
            seen_targets = set()
            for target in target_list:
                if target in seen_targets:
                    conflicts.append(f"Duplicate target column name: '{target}'")
                seen_targets.add(target)
        else:
            seen_targets_lower = set()
            for target in target_list:
                target_lower = target.lower()
                if target_lower in seen_targets_lower:
                    conflicts.append(f"Duplicate target column name (case-insensitive): '{target}'")
                seen_targets_lower.add(target_lower)
        
        return conflicts
    
    def _validate_target_column_names(self, mapping: Dict[str, str]) -> List[str]:
        """Validate target column names against naming conventions and reserved words."""
        invalid_targets = []
        
        for source, target in mapping.items():
            # Check for reserved SQL keywords
            if target.lower() in self._reserved_keywords:
                invalid_targets.append(f"'{target}' is a reserved SQL keyword")
            
            # Check for valid identifier format (letters, numbers, underscores, no spaces)
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', target):
                invalid_targets.append(f"'{target}' is not a valid column identifier")
            
            # Check for length (PostgreSQL limit is 63 characters)
            if len(target) > 63:
                invalid_targets.append(f"'{target}' exceeds maximum column name length (63 characters)")
        
        return invalid_targets
    
    def _apply_column_mappings(
        self, 
        df: pd.DataFrame, 
        mapping: Dict[str, str],
        case_sensitive: bool
    ) -> Dict[str, str]:
        """Apply the actual column mappings to the DataFrame."""
        successful_mappings = {}
        
        for source, target in mapping.items():
            # Find the actual column name in the DataFrame
            actual_source = self._find_column_name(df.columns.tolist(), source, case_sensitive)
            
            if actual_source is not None:
                # Rename the column
                df.rename(columns={actual_source: target}, inplace=True)
                successful_mappings[actual_source] = target
                logger.debug(f"Mapped column '{actual_source}' -> '{target}'")
        
        return successful_mappings
    
    def _find_column_name(
        self, 
        df_columns: List[str], 
        target_column: str, 
        case_sensitive: bool
    ) -> Optional[str]:
        """Find the actual column name in the DataFrame, handling case sensitivity."""
        if case_sensitive:
            return target_column if target_column in df_columns else None
        else:
            target_lower = target_column.lower()
            for col in df_columns:
                if col.lower() == target_lower:
                    return col
            return None
    
    def validate_embedding_columns(
        self, 
        embed_columns: List[str], 
        mapped_columns: List[str],
        case_sensitive: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Validate that embedding columns exist after mapping.
        
        Args:
            embed_columns: List of column names to embed
            mapped_columns: List of column names after mapping
            case_sensitive: Whether to perform case-sensitive matching
            
        Returns:
            Tuple of (valid_columns, missing_columns)
        """
        valid_columns = []
        missing_columns = []
        
        if case_sensitive:
            mapped_set = set(mapped_columns)
            for col in embed_columns:
                if col in mapped_set:
                    valid_columns.append(col)
                else:
                    missing_columns.append(col)
        else:
            mapped_lower = {col.lower(): col for col in mapped_columns}
            for col in embed_columns:
                col_lower = col.lower()
                if col_lower in mapped_lower:
                    valid_columns.append(mapped_lower[col_lower])
                else:
                    missing_columns.append(col)
        
        return valid_columns, missing_columns
    
    def validate_embedding_columns_with_mapping(
        self,
        embed_columns: List[str],
        original_columns: List[str],
        column_mapping: Dict[str, str],
        case_sensitive: bool = True
    ) -> Dict[str, Any]:
        """
        Validate embedding columns considering column mapping transformations.
        
        This method provides comprehensive validation for embedding columns when
        column mapping is involved, including suggestions for corrections.
        
        Args:
            embed_columns: List of embedding column names to validate
            original_columns: List of original column names before mapping
            column_mapping: Dictionary of column name mappings (original -> mapped)
            case_sensitive: Whether to perform case-sensitive validation
            
        Returns:
            Dictionary containing validation results and suggestions
        """
        result = {
            "valid_columns": [],
            "missing_columns": [],
            "suggestions": {},
            "warnings": [],
            "errors": []
        }
        
        # Create mapped columns list
        mapped_columns = []
        for col in original_columns:
            if col in column_mapping:
                mapped_columns.append(column_mapping[col])
            else:
                mapped_columns.append(col)
        
        # Validate each embedding column
        for embed_col in embed_columns:
            if case_sensitive:
                if embed_col in mapped_columns:
                    result["valid_columns"].append(embed_col)
                else:
                    result["missing_columns"].append(embed_col)
                    # Check if it might be an original column name
                    if embed_col in column_mapping:
                        mapped_name = column_mapping[embed_col]
                        result["suggestions"][embed_col] = {
                            "type": "original_name_used",
                            "suggestion": mapped_name,
                            "message": f"Use mapped name '{mapped_name}' instead of original name '{embed_col}'"
                        }
                    else:
                        # Look for similar columns
                        similar_cols = self._find_similar_columns(embed_col, mapped_columns, case_sensitive)
                        if similar_cols:
                            result["suggestions"][embed_col] = {
                                "type": "similar_names",
                                "suggestion": similar_cols[0],
                                "alternatives": similar_cols,
                                "message": f"Did you mean '{similar_cols[0]}'? Other options: {similar_cols[1:]}"
                            }
            else:
                # Case-insensitive validation
                embed_col_lower = embed_col.lower()
                mapped_lower = {col.lower(): col for col in mapped_columns}
                
                if embed_col_lower in mapped_lower:
                    actual_col = mapped_lower[embed_col_lower]
                    result["valid_columns"].append(actual_col)
                    if embed_col != actual_col:
                        result["warnings"].append(f"Case mismatch: using '{actual_col}' for '{embed_col}'")
                else:
                    result["missing_columns"].append(embed_col)
                    # Check original names case-insensitively
                    original_lower = {col.lower(): col for col in column_mapping.keys()}
                    if embed_col_lower in original_lower:
                        original_col = original_lower[embed_col_lower]
                        mapped_name = column_mapping[original_col]
                        result["suggestions"][embed_col] = {
                            "type": "original_name_used",
                            "suggestion": mapped_name,
                            "message": f"Use mapped name '{mapped_name}' instead of original name '{embed_col}'"
                        }
        
        return result
    
    def _find_similar_columns(self, target_col: str, available_columns: List[str], case_sensitive: bool) -> List[str]:
        """
        Find columns similar to the target column.
        
        Args:
            target_col: Column name to find matches for
            available_columns: List of available column names
            case_sensitive: Whether to use case-sensitive matching
            
        Returns:
            List of similar column names, sorted by similarity
        """
        similarities = []
        target_lower = target_col.lower() if not case_sensitive else target_col
        
        for col in available_columns:
            col_compare = col.lower() if not case_sensitive else col
            similarity = self._calculate_similarity(target_lower, col_compare)
            if similarity > 0.5:  # Only include reasonably similar columns
                similarities.append((col, similarity))
        
        # Sort by similarity (descending) and return column names
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [col for col, _ in similarities[:3]]  # Return top 3 matches
    
    def suggest_mappings(
        self, 
        source_columns: List[str], 
        target_columns: List[str],
        similarity_threshold: float = 0.8
    ) -> Dict[str, List[str]]:
        """
        Suggest potential column mappings based on name similarity.
        
        Args:
            source_columns: List of source column names
            target_columns: List of target column names  
            similarity_threshold: Minimum similarity score for suggestions
            
        Returns:
            Dictionary mapping source columns to lists of suggested target columns
        """
        suggestions = {}
        
        for source in source_columns:
            source_suggestions = []
            
            for target in target_columns:
                similarity = self._calculate_similarity(source, target)
                if similarity >= similarity_threshold:
                    source_suggestions.append((target, similarity))
            
            # Sort by similarity score (descending)
            source_suggestions.sort(key=lambda x: x[1], reverse=True)
            suggestions[source] = [target for target, _ in source_suggestions]
        
        return suggestions
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Levenshtein distance."""
        # Simple implementation - could be enhanced with more sophisticated algorithms
        str1_lower = str1.lower()
        str2_lower = str2.lower()
        
        if str1_lower == str2_lower:
            return 1.0
        
        # Calculate Levenshtein distance
        len1, len2 = len(str1_lower), len(str2_lower)
        
        if len1 == 0:
            return 0.0 if len2 > 0 else 1.0
        if len2 == 0:
            return 0.0
        
        # Create matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize first row and column
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j
        
        # Fill matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if str1_lower[i-1] == str2_lower[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )
        
        # Calculate similarity as 1 - (distance / max_length)
        distance = matrix[len1][len2]
        max_length = max(len1, len2)
        similarity = 1.0 - (distance / max_length)
        
        return similarity