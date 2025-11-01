"""Data type validation and compatibility checking for PostgreSQL."""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from dataload.domain.entities import (
    TableInfo, ColumnInfo, TypeMismatch, ConversionSuggestion
)


class DataTypeValidator:
    """
    Validates data type compatibility between CSV data and PostgreSQL database columns.
    Provides type conversion suggestions and compatibility analysis.
    """
    
    def __init__(self):
        # PostgreSQL type compatibility matrix
        self.pg_type_hierarchy = {
            'text': ['varchar', 'char', 'character varying', 'character'],
            'integer': ['smallint', 'int2', 'int4'],
            'bigint': ['integer', 'smallint', 'int8', 'int4', 'int2'],
            'numeric': ['decimal', 'real', 'double precision', 'float4', 'float8'],
            'boolean': ['bool'],
            'timestamp': ['timestamptz', 'timestamp without time zone', 'timestamp with time zone'],
            'date': [],
            'jsonb': ['json'],
            'text[]': ['varchar[]', 'character varying[]'],
            'integer[]': ['int4[]', 'int2[]'],
            'vector': []  # pgvector extension type
        }
        
        # Pandas to PostgreSQL type mapping
        self.pandas_to_pg_mapping = {
            'object': 'text',
            'string': 'text', 
            'int64': 'bigint',
            'int32': 'integer',
            'int16': 'smallint',
            'float64': 'double precision',
            'float32': 'real',
            'bool': 'boolean',
            'datetime64[ns]': 'timestamp',
            'timedelta64[ns]': 'interval',
            'category': 'text'
        }
    
    def validate_type_compatibility(
        self, 
        table_info: TableInfo, 
        df: pd.DataFrame
    ) -> List[TypeMismatch]:
        """
        Validate type compatibility between database columns and CSV data.
        
        Args:
            table_info: Information about the database table
            df: DataFrame containing CSV data
            
        Returns:
            List of TypeMismatch objects for incompatible types
        """
        type_mismatches = []
        
        # Only check columns that exist in both database and CSV
        common_columns = set(table_info.columns.keys()) & set(df.columns)
        
        for col_name in common_columns:
            db_column = table_info.columns[col_name]
            csv_type = self._infer_csv_column_type(df[col_name])
            
            compatibility = self._check_type_compatibility(
                db_column.data_type, 
                csv_type, 
                df[col_name]
            )
            
            if not compatibility['compatible']:
                type_mismatches.append(TypeMismatch(
                    column_name=col_name,
                    db_type=db_column.data_type,
                    csv_type=csv_type,
                    compatible=compatibility['compatible'],
                    conversion_required=compatibility['conversion_required'],
                    sample_values=self._get_sample_values(df[col_name])
                ))
        
        return type_mismatches
    
    def suggest_type_conversions(
        self, 
        mismatches: List[TypeMismatch]
    ) -> List[ConversionSuggestion]:
        """
        Suggest type conversions for resolving type mismatches.
        
        Args:
            mismatches: List of type mismatches to resolve
            
        Returns:
            List of conversion suggestions
        """
        suggestions = []
        
        for mismatch in mismatches:
            suggestion = self._generate_conversion_suggestion(mismatch)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _infer_csv_column_type(self, series: pd.Series) -> str:
        """
        Infer the PostgreSQL-compatible type for a pandas Series.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            Inferred PostgreSQL type name
        """
        # Handle empty or all-null series
        if series.empty or series.isna().all():
            return 'text'
        
        # Get the pandas dtype
        dtype_str = str(series.dtype)
        
        # Check for vector types first (arrays of numbers with consistent length)
        if self._is_vector_column(series):
            return 'vector'
        
        # Check for array/list types
        if self._is_array_column(series):
            element_type = self._infer_array_element_type(series)
            return f"{element_type}[]"
        
        # Handle object dtype with special cases
        if dtype_str == 'object':
            return self._infer_object_type(series)
        
        # Use direct mapping for known types
        return self.pandas_to_pg_mapping.get(dtype_str, 'text')
    
    def _is_array_column(self, series: pd.Series) -> bool:
        """Check if a series contains array/list data."""
        non_null = series.dropna()
        if non_null.empty:
            return False
        
        # Check if most values are lists
        list_count = non_null.apply(lambda x: isinstance(x, (list, tuple))).sum()
        return list_count > len(non_null) * 0.8
    
    def _is_vector_column(self, series: pd.Series) -> bool:
        """Check if a series contains vector data (consistent-length numeric arrays)."""
        non_null = series.dropna()
        if non_null.empty:
            return False
        
        # Check if values are numeric arrays with consistent length
        try:
            first_valid = None
            for val in non_null:
                if isinstance(val, (list, tuple, np.ndarray)):
                    if all(isinstance(x, (int, float, np.number)) for x in val):
                        if first_valid is None:
                            first_valid = len(val)
                        elif len(val) != first_valid:
                            return False
                    else:
                        return False
                else:
                    return False
            
            return first_valid is not None and first_valid > 0
        except:
            return False
    
    def _infer_array_element_type(self, series: pd.Series) -> str:
        """Infer the element type for array columns."""
        non_null = series.dropna()
        if non_null.empty:
            return 'text'
        
        # Sample some array elements
        sample_elements = []
        for val in non_null.head(100):
            if isinstance(val, (list, tuple)):
                sample_elements.extend(val[:5])  # Take first 5 elements from each array
        
        if not sample_elements:
            return 'text'
        
        # Create a temporary series to infer type
        element_series = pd.Series(sample_elements)
        return self._infer_csv_column_type(element_series)
    
    def _infer_object_type(self, series: pd.Series) -> str:
        """Infer PostgreSQL type for object dtype series."""
        non_null = series.dropna()
        if non_null.empty:
            return 'text'
        
        # Sample some values for analysis
        sample = non_null.head(1000)
        
        # Check if all values can be converted to numeric
        try:
            numeric_series = pd.to_numeric(sample, errors='coerce')
            if not numeric_series.isna().any():
                # All values are numeric
                if (numeric_series % 1 == 0).all():
                    # All integers
                    max_val = numeric_series.max()
                    if max_val <= 2147483647:  # int4 max
                        return 'integer'
                    else:
                        return 'bigint'
                else:
                    return 'double precision'
        except:
            pass
        
        # Check if values look like booleans
        unique_values = set(str(v).lower() for v in sample.unique())
        if unique_values.issubset({'true', 'false', 't', 'f', '1', '0', 'yes', 'no'}):
            return 'boolean'
        
        # Check if values look like dates
        try:
            pd.to_datetime(sample.head(10), errors='raise')
            return 'timestamp'
        except:
            pass
        
        # Default to text
        return 'text'
    
    def _check_type_compatibility(
        self, 
        db_type: str, 
        csv_type: str, 
        series: pd.Series
    ) -> Dict[str, bool]:
        """
        Check if CSV type is compatible with database type.
        
        Args:
            db_type: PostgreSQL database column type
            csv_type: Inferred CSV column type
            series: The actual data series for additional validation
            
        Returns:
            Dictionary with compatibility information
        """
        # Normalize type names
        db_type_normalized = self._normalize_pg_type(db_type)
        csv_type_normalized = self._normalize_pg_type(csv_type)
        
        # Special handling for vector types
        if db_type.startswith('vector') and csv_type == 'vector':
            # Check if dimensions match by examining the data
            if self._validate_vector_dimensions(db_type, series):
                return {'compatible': True, 'conversion_required': False}
            else:
                return {'compatible': False, 'conversion_required': False}
        
        # Exact match
        if db_type_normalized == csv_type_normalized:
            return {'compatible': True, 'conversion_required': False}
        
        # Check type hierarchy compatibility
        if self._is_type_compatible(db_type_normalized, csv_type_normalized):
            return {'compatible': True, 'conversion_required': True}
        
        # Check if conversion is possible
        if self._is_conversion_possible(db_type_normalized, csv_type_normalized, series):
            return {'compatible': True, 'conversion_required': True}
        
        # Incompatible
        return {'compatible': False, 'conversion_required': False}
    
    def _normalize_pg_type(self, pg_type: str) -> str:
        """Normalize PostgreSQL type names."""
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
            'bool': 'boolean',
            'timestamptz': 'timestamp',
            'timestamp without time zone': 'timestamp',
            'timestamp with time zone': 'timestamp'
        }
        
        return type_aliases.get(pg_type.lower(), pg_type.lower())
    
    def _is_type_compatible(self, db_type: str, csv_type: str) -> bool:
        """Check if types are compatible based on PostgreSQL type hierarchy."""
        # Text types are generally compatible
        if db_type == 'text':
            return True
        
        # Numeric type compatibility - allow downcasting if safe
        numeric_hierarchy = ['smallint', 'integer', 'bigint', 'real', 'double precision', 'numeric']
        if db_type in numeric_hierarchy and csv_type in numeric_hierarchy:
            # Special case: bigint to integer is compatible if values fit
            if db_type == 'integer' and csv_type == 'bigint':
                return True  # Will be checked during conversion
            
            db_idx = numeric_hierarchy.index(db_type)
            csv_idx = numeric_hierarchy.index(csv_type)
            return db_idx >= csv_idx  # Can promote to larger type
        
        # Array type compatibility
        if db_type.endswith('[]') and csv_type.endswith('[]'):
            db_element = db_type[:-2]
            csv_element = csv_type[:-2]
            return self._is_type_compatible(db_element, csv_element)
        
        return False
    
    def _is_conversion_possible(
        self, 
        db_type: str, 
        csv_type: str, 
        series: pd.Series
    ) -> bool:
        """Check if conversion from CSV type to DB type is possible."""
        # Text to other types - check if data can be converted
        if csv_type == 'text':
            if db_type in ['integer', 'bigint', 'smallint']:
                return self._can_convert_to_integer(series)
            elif db_type in ['real', 'double precision', 'numeric']:
                return self._can_convert_to_numeric(series)
            elif db_type == 'boolean':
                return self._can_convert_to_boolean(series)
            elif db_type == 'timestamp':
                return self._can_convert_to_timestamp(series)
        
        # Numeric to text is always possible
        if db_type == 'text':
            return True
        
        return False
    
    def _can_convert_to_integer(self, series: pd.Series) -> bool:
        """Check if series can be converted to integer."""
        try:
            sample = series.dropna().head(100)
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            return False
    
    def _can_convert_to_numeric(self, series: pd.Series) -> bool:
        """Check if series can be converted to numeric."""
        try:
            sample = series.dropna().head(100)
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            return False
    
    def _can_convert_to_boolean(self, series: pd.Series) -> bool:
        """Check if series can be converted to boolean."""
        sample = series.dropna().head(100)
        unique_values = set(str(v).lower() for v in sample.unique())
        boolean_values = {'true', 'false', 't', 'f', '1', '0', 'yes', 'no'}
        return unique_values.issubset(boolean_values)
    
    def _can_convert_to_timestamp(self, series: pd.Series) -> bool:
        """Check if series can be converted to timestamp."""
        try:
            sample = series.dropna().head(10)
            # Try common date formats first to avoid warnings
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    pd.to_datetime(sample, format=fmt, errors='raise')
                    return True
                except:
                    continue
            
            # Fall back to automatic parsing
            pd.to_datetime(sample, errors='raise')
            return True
        except:
            return False
    
    def _get_sample_values(self, series: pd.Series, n: int = 3) -> List[Any]:
        """Get sample values from a series for error reporting."""
        non_null = series.dropna()
        if non_null.empty:
            return []
        
        return non_null.head(n).tolist()
    
    def _generate_conversion_suggestion(
        self, 
        mismatch: TypeMismatch
    ) -> Optional[ConversionSuggestion]:
        """Generate a conversion suggestion for a type mismatch."""
        conversion_map = {
            ('text', 'integer'): ('pd.to_numeric', 'medium', 'Convert text to integer'),
            ('text', 'bigint'): ('pd.to_numeric', 'medium', 'Convert text to bigint'),
            ('text', 'double precision'): ('pd.to_numeric', 'low', 'Convert text to numeric'),
            ('text', 'boolean'): ('custom_bool_convert', 'medium', 'Convert text to boolean'),
            ('text', 'timestamp'): ('pd.to_datetime', 'high', 'Convert text to timestamp'),
            ('integer', 'text'): ('str', 'low', 'Convert integer to text'),
            ('bigint', 'integer'): ('int', 'medium', 'Downcast bigint to integer'),
        }
        
        key = (mismatch.csv_type, mismatch.db_type)
        if key in conversion_map:
            func, risk, desc = conversion_map[key]
            return ConversionSuggestion(
                column_name=mismatch.column_name,
                from_type=mismatch.csv_type,
                to_type=mismatch.db_type,
                conversion_function=func,
                risk_level=risk,
                description=desc
            )
        
        return None
    
    def _validate_vector_dimensions(self, db_type: str, series: pd.Series) -> bool:
        """
        Validate that vector data in series matches the expected dimensions.
        
        Args:
            db_type: Database vector type (e.g., 'vector(3)')
            series: Series containing vector data
            
        Returns:
            True if dimensions match, False otherwise
        """
        # Extract dimension from db_type (e.g., 'vector(3)' -> 3)
        import re
        match = re.search(r'vector\((\d+)\)', db_type)
        if not match:
            return True  # If no dimension specified, assume compatible
        
        expected_dim = int(match.group(1))
        
        # Check dimensions in the series data
        non_null = series.dropna()
        if non_null.empty:
            return True
        
        for val in non_null.head(10):  # Check first 10 values
            if isinstance(val, (list, tuple)):
                if len(val) != expected_dim:
                    return False
        
        return True