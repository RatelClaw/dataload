"""Schema validation for DataMove operations."""

from typing import Set, List, Any
import pandas as pd
from dataload.domain.entities import (
    TableInfo, SchemaAnalysis, TypeMismatch, CaseConflict, ConstraintViolation
)


class SchemaValidator:
    """
    Validates schema compatibility between database tables and CSV data.
    Provides different validation strategies for existing_schema and new_schema modes.
    """
    
    async def validate_existing_schema(
        self, 
        table_info: TableInfo, 
        df: pd.DataFrame
    ) -> SchemaAnalysis:
        """
        Perform strict schema validation for existing_schema move type.
        Requires exact column name and type matching.
        
        Args:
            table_info: Information about the existing database table
            df: DataFrame containing CSV data
            
        Returns:
            SchemaAnalysis: Analysis of schema compatibility
        """
        db_columns = set(table_info.columns.keys())
        csv_columns = set(df.columns)
        
        # For existing_schema, we require exact matches
        columns_added = list(csv_columns - db_columns)
        columns_removed = list(db_columns - csv_columns)
        
        # Schema is compatible only if columns match exactly
        compatible = len(columns_added) == 0 and len(columns_removed) == 0
        
        return SchemaAnalysis(
            table_exists=True,
            columns_added=columns_added,
            columns_removed=columns_removed,
            columns_modified=[],  # Type mismatches handled by DataTypeValidator
            case_conflicts=[],    # Case conflicts handled by CaseSensitivityValidator
            constraint_violations=[],  # Constraint validation handled separately
            compatible=compatible,
            requires_schema_update=False  # existing_schema never updates schema
        )
    
    async def validate_new_schema(
        self, 
        table_info: TableInfo, 
        df: pd.DataFrame
    ) -> SchemaAnalysis:
        """
        Perform flexible schema validation for new_schema move type.
        Allows column additions and removals with enhanced analysis.
        
        Args:
            table_info: Information about the existing database table
            df: DataFrame containing CSV data
            
        Returns:
            SchemaAnalysis: Analysis of schema compatibility and required changes
        """
        db_columns = set(table_info.columns.keys())
        csv_columns = set(df.columns)
        
        # For new_schema, we allow differences but track them
        columns_added = list(csv_columns - db_columns)
        columns_removed = list(db_columns - csv_columns)
        
        # Analyze column modifications (type changes for existing columns)
        columns_modified = self._analyze_column_modifications(table_info, df)
        
        # Schema is compatible for new_schema mode unless there are critical issues
        # (case conflicts and constraint violations are checked separately)
        compatible = self._assess_new_schema_compatibility(
            table_info, columns_added, columns_removed, columns_modified
        )
        
        requires_schema_update = (
            len(columns_added) > 0 or 
            len(columns_removed) > 0 or 
            len(columns_modified) > 0
        )
        
        return SchemaAnalysis(
            table_exists=True,
            columns_added=columns_added,
            columns_removed=columns_removed,
            columns_modified=columns_modified,  # Enhanced with actual type modifications
            case_conflicts=[],    # Case conflicts handled by CaseSensitivityValidator
            constraint_violations=[],  # Constraint validation handled separately
            compatible=compatible,
            requires_schema_update=requires_schema_update
        )
    
    def _analyze_column_modifications(
        self, 
        table_info: TableInfo, 
        df: pd.DataFrame
    ) -> List['TypeMismatch']:
        """
        Analyze modifications to existing columns (type changes).
        
        Args:
            table_info: Current table information
            df: DataFrame with new data
            
        Returns:
            List of TypeMismatch objects for columns that need modification
        """
        from dataload.domain.entities import TypeMismatch
        
        modifications = []
        
        # Check columns that exist in both database and CSV
        common_columns = set(table_info.columns.keys()) & set(df.columns)
        
        for col_name in common_columns:
            db_column = table_info.columns[col_name]
            csv_type = self._infer_csv_column_type(df[col_name])
            
            # Check if types are different and require modification
            if not self._are_types_equivalent(db_column.data_type, csv_type):
                modifications.append(TypeMismatch(
                    column_name=col_name,
                    db_type=db_column.data_type,
                    csv_type=csv_type,
                    compatible=self._is_type_modification_safe(db_column.data_type, csv_type),
                    conversion_required=True,
                    sample_values=self._get_sample_values(df[col_name])
                ))
        
        return modifications
    
    def _assess_new_schema_compatibility(
        self, 
        table_info: TableInfo, 
        columns_added: List[str], 
        columns_removed: List[str], 
        columns_modified: List['TypeMismatch']
    ) -> bool:
        """
        Assess overall compatibility for new_schema mode.
        
        Args:
            table_info: Table information
            columns_added: List of columns being added
            columns_removed: List of columns being removed
            columns_modified: List of column modifications
            
        Returns:
            True if schema changes are compatible, False otherwise
        """
        # Check for removal of critical columns (primary keys, etc.)
        for pk_col in table_info.primary_keys:
            if pk_col in columns_removed:
                return False  # Cannot remove primary key columns
        
        # Check for unsafe type modifications
        for modification in columns_modified:
            if not modification.compatible:
                return False  # Unsafe type modification
        
        # Check for removal of too many columns (potential data loss)
        if len(columns_removed) > len(table_info.columns) * 0.8:
            return False  # Removing more than 80% of columns is likely an error
        
        return True
    
    def _infer_csv_column_type(self, series: pd.Series) -> str:
        """
        Infer PostgreSQL-compatible type for a pandas Series.
        Simplified version for schema validation.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            Inferred PostgreSQL type name
        """
        if series.empty or series.isna().all():
            return 'text'
        
        dtype_str = str(series.dtype)
        
        # Basic type mapping
        type_mapping = {
            'object': 'text',
            'string': 'text',
            'int64': 'bigint',
            'int32': 'integer',
            'int16': 'smallint',
            'float64': 'double precision',
            'float32': 'real',
            'bool': 'boolean',
            'datetime64[ns]': 'timestamp'
        }
        
        return type_mapping.get(dtype_str, 'text')
    
    def _are_types_equivalent(self, db_type: str, csv_type: str) -> bool:
        """
        Check if database type and CSV type are equivalent.
        
        Args:
            db_type: Database column type
            csv_type: Inferred CSV type
            
        Returns:
            True if types are equivalent, False otherwise
        """
        # Normalize types for comparison
        db_normalized = self._normalize_type(db_type)
        csv_normalized = self._normalize_type(csv_type)
        
        return db_normalized == csv_normalized
    
    def _normalize_type(self, type_name: str) -> str:
        """Normalize PostgreSQL type names for comparison."""
        type_aliases = {
            'varchar': 'text',
            'character varying': 'text',
            'character': 'text',
            'char': 'text',
            'int4': 'integer',
            'int2': 'smallint',
            'int8': 'bigint',
            'float4': 'real',
            'float8': 'double precision',
            'bool': 'boolean'
        }
        
        return type_aliases.get(type_name.lower(), type_name.lower())
    
    def _is_type_modification_safe(self, db_type: str, csv_type: str) -> bool:
        """
        Check if modifying from db_type to csv_type is safe.
        
        Args:
            db_type: Current database type
            csv_type: Target CSV type
            
        Returns:
            True if modification is safe, False otherwise
        """
        # Text types are generally safe to modify to
        if csv_type == 'text':
            return True
        
        # Numeric type promotions are generally safe
        numeric_hierarchy = ['smallint', 'integer', 'bigint', 'real', 'double precision']
        if db_type in numeric_hierarchy and csv_type in numeric_hierarchy:
            db_idx = numeric_hierarchy.index(db_type)
            csv_idx = numeric_hierarchy.index(csv_type)
            return csv_idx >= db_idx  # Can promote to larger type
        
        # Same types are safe
        return self._are_types_equivalent(db_type, csv_type)
    
    def _get_sample_values(self, series: pd.Series, n: int = 3) -> List[Any]:
        """Get sample values from a series."""
        non_null = series.dropna()
        if non_null.empty:
            return []
        return non_null.head(n).tolist()
    
    def _analyze_column_differences(
        self, 
        db_columns: Set[str], 
        csv_columns: Set[str]
    ) -> tuple[List[str], List[str]]:
        """
        Analyze differences between database and CSV columns.
        
        Args:
            db_columns: Set of database column names
            csv_columns: Set of CSV column names
            
        Returns:
            Tuple of (columns_added, columns_removed)
        """
        columns_added = list(csv_columns - db_columns)
        columns_removed = list(db_columns - csv_columns)
        
        return columns_added, columns_removed