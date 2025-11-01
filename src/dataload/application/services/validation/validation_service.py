"""Main validation service that orchestrates different validation strategies."""

from typing import Optional, List, Dict
import pandas as pd
from dataload.domain.entities import (
    TableInfo, ValidationReport, SchemaAnalysis, ValidationError
)
from .schema_validator import SchemaValidator
from .case_sensitivity_validator import CaseSensitivityValidator
from .data_type_validator import DataTypeValidator


class ValidationService:
    """
    Central validation service that coordinates different validation strategies
    based on the move_type parameter.
    """
    
    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.case_sensitivity_validator = CaseSensitivityValidator()
        self.data_type_validator = DataTypeValidator()
    
    async def validate_data_move(
        self,
        table_info: Optional[TableInfo],
        df: pd.DataFrame,
        move_type: Optional[str] = None
    ) -> ValidationReport:
        """
        Validate data movement with comprehensive error collection and reporting.
        
        This method performs thorough validation based on table existence and move_type,
        collecting all validation errors, warnings, and recommendations to provide
        a complete picture of the validation results.
        
        Args:
            table_info: Information about the target table (None if table doesn't exist)
            df: DataFrame containing the CSV data to be moved
            move_type: Type of move operation ('existing_schema', 'new_schema', or None)
            
        Returns:
            ValidationReport: Comprehensive validation results with collected errors
            
        Raises:
            ValidationError: If validation parameters are invalid (not for validation failures)
        """
        collected_errors = []
        collected_warnings = []
        
        try:
            # Validate input parameters first
            parameter_errors = self._validate_input_parameters(table_info, df, move_type)
            if parameter_errors:
                collected_errors.extend(parameter_errors)
                
                # Return early for parameter validation failures
                return self._create_error_report(
                    errors=collected_errors,
                    warnings=collected_warnings,
                    message="Parameter validation failed"
                )
            
            # If table doesn't exist, validate for new table creation
            if table_info is None:
                return await self._validate_new_table_creation(df)
            
            # Perform validation based on move_type with comprehensive error collection
            if move_type == 'existing_schema':
                return await self._validate_existing_schema_comprehensive(table_info, df)
            else:  # new_schema
                return await self._validate_new_schema_comprehensive(table_info, df)
                
        except Exception as e:
            # Catch any unexpected validation errors and include them in the report
            collected_errors.append(f"Unexpected validation error: {e}")
            return self._create_error_report(
                errors=collected_errors,
                warnings=collected_warnings,
                message=f"Validation failed due to unexpected error: {e}"
            )

    def _validate_input_parameters(
        self, 
        table_info: Optional[TableInfo], 
        df: pd.DataFrame, 
        move_type: Optional[str]
    ) -> List[str]:
        """
        Validate input parameters and return list of parameter errors.
        
        Args:
            table_info: Table information
            df: DataFrame to validate
            move_type: Move type parameter
            
        Returns:
            List of parameter validation error messages
        """
        errors = []
        
        # Validate DataFrame
        if df is None:
            errors.append("DataFrame cannot be None")
        elif not isinstance(df, pd.DataFrame):
            errors.append("Data must be a pandas DataFrame")
        elif len(df.columns) == 0:
            if table_info is not None and move_type == 'existing_schema':
                # For existing_schema, empty DataFrame means missing all columns
                missing_columns = list(table_info.columns.keys())
                errors.append(f"Missing columns in CSV that exist in database: {', '.join(missing_columns)}")
            else:
                errors.append("DataFrame must have at least one column")
        
        # Validate move_type for existing tables
        if table_info is not None:
            if move_type is None:
                errors.append(
                    "move_type parameter is required when target table exists. "
                    "Use 'existing_schema' for strict validation or 'new_schema' for flexible validation."
                )
            elif move_type not in ['existing_schema', 'new_schema']:
                errors.append(
                    f"Invalid move_type: '{move_type}'. Must be 'existing_schema' or 'new_schema'."
                )
        
        # Validate table_info structure if provided
        if table_info is not None:
            if not hasattr(table_info, 'columns') or not table_info.columns:
                errors.append("Table information is incomplete: missing column information")
            if not hasattr(table_info, 'name') or not table_info.name:
                errors.append("Table information is incomplete: missing table name")
        
        return errors

    async def _validate_new_table_creation(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate DataFrame for new table creation with comprehensive checks.
        
        Args:
            df: DataFrame to validate for new table creation
            
        Returns:
            ValidationReport with validation results
        """
        errors = []
        warnings = []
        recommendations = []
        
        # Check for duplicate column names
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            errors.append(f"Duplicate column names found: {duplicate_columns}")
        
        # Check for empty column names
        empty_columns = [col for col in df.columns if not col or str(col).strip() == '']
        if empty_columns:
            errors.append("Empty or whitespace-only column names found")
        
        # Check for problematic column names
        problematic_columns = []
        for col in df.columns:
            col_str = str(col)
            if col_str.startswith(' ') or col_str.endswith(' '):
                problematic_columns.append(f"'{col}' (leading/trailing spaces)")
            elif any(char in col_str for char in ['"', "'", ';', '\n', '\r']):
                problematic_columns.append(f"'{col}' (contains special characters)")
        
        if problematic_columns:
            warnings.extend([f"Problematic column name: {col}" for col in problematic_columns])
            recommendations.append("Consider cleaning column names before table creation")
        
        # Check data types and potential issues
        type_warnings = []
        for col in df.columns:
            series = df[col]
            
            # Check for mixed types
            if series.dtype == 'object' and not series.empty:
                unique_types = set(type(val).__name__ for val in series.dropna().head(100))
                if len(unique_types) > 1:
                    type_warnings.append(f"Column '{col}' has mixed data types: {unique_types}")
            
            # Check for very large text values
            if series.dtype == 'object':
                max_length = series.astype(str).str.len().max()
                if max_length > 10000:
                    warnings.append(f"Column '{col}' has very long text values (max: {max_length} chars)")
        
        warnings.extend(type_warnings)
        
        # Add general recommendations
        recommendations.extend([
            f"New table will be created with {len(df.columns)} columns",
            "All CSV columns will be preserved in the new table",
            "Consider adding primary key constraints if needed"
        ])
        
        if not df.empty:
            recommendations.append(f"Table will be populated with {len(df)} rows")
        
        # Create schema analysis for new table
        schema_analysis = SchemaAnalysis(
            table_exists=False,
            columns_added=list(df.columns),
            columns_removed=[],
            columns_modified=[],
            case_conflicts=[],
            constraint_violations=[],
            compatible=len(errors) == 0,
            requires_schema_update=False
        )
        
        return ValidationReport(
            schema_analysis=schema_analysis,
            case_conflicts=[],
            type_mismatches=[],
            constraint_violations=[],
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            validation_passed=len(errors) == 0
        )
    
    async def _validate_existing_schema_comprehensive(
        self, 
        table_info: TableInfo, 
        df: pd.DataFrame
    ) -> ValidationReport:
        """
        Perform comprehensive strict validation for existing_schema move type.
        
        This method collects all possible validation errors and warnings to provide
        a complete picture of schema compatibility issues.
        """
        collected_errors = []
        collected_warnings = []
        recommendations = []
        
        try:
            # Perform schema validation with error collection
            schema_result = await self.schema_validator.validate_existing_schema(table_info, df)
            
            # Collect schema mismatch errors
            if schema_result.columns_added:
                collected_errors.append(
                    f"Extra columns in CSV not present in database: {', '.join(schema_result.columns_added)}"
                )
            
            if schema_result.columns_removed:
                collected_errors.append(
                    f"Missing columns in CSV that exist in database: {', '.join(schema_result.columns_removed)}"
                )
            
            # Perform data type validation with error collection
            try:
                type_mismatches = self.data_type_validator.validate_type_compatibility(table_info, df)
                
                for mismatch in type_mismatches:
                    if not mismatch.compatible:
                        error_msg = (
                            f"Type mismatch in column '{mismatch.column_name}': "
                            f"expected {mismatch.db_type}, got {mismatch.csv_type}"
                        )
                        if mismatch.sample_values:
                            error_msg += f" (sample values: {mismatch.sample_values[:3]})"
                        collected_errors.append(error_msg)
                    elif mismatch.conversion_required:
                        collected_warnings.append(
                            f"Type conversion may be required for column '{mismatch.column_name}': "
                            f"{mismatch.csv_type} -> {mismatch.db_type}"
                        )
                        
            except Exception as type_validation_error:
                collected_errors.append(f"Type validation failed: {type_validation_error}")
                type_mismatches = []
            
            # Check for case sensitivity issues (even in existing_schema mode)
            try:
                case_conflicts = self.case_sensitivity_validator.detect_case_conflicts(
                    list(table_info.columns.keys()),
                    list(df.columns)
                )
                
                for conflict in case_conflicts:
                    if conflict.conflict_type == 'case_mismatch':
                        collected_errors.append(
                            f"Case mismatch: database column '{conflict.db_column}' "
                            f"vs CSV column '{conflict.csv_column}'"
                        )
                    elif conflict.conflict_type == 'duplicate_insensitive':
                        collected_errors.append(
                            f"Duplicate case-insensitive column in CSV: '{conflict.csv_column}'"
                        )
                        
            except Exception as case_validation_error:
                collected_warnings.append(f"Case sensitivity validation failed: {case_validation_error}")
                case_conflicts = []
            
            # Validate data constraints if possible
            constraint_violations = []
            try:
                # Check for null values in non-nullable columns
                for col_name, col_info in table_info.columns.items():
                    if col_name in df.columns and not col_info.nullable:
                        null_count = df[col_name].isnull().sum()
                        if null_count > 0:
                            collected_errors.append(
                                f"Column '{col_name}' is NOT NULL in database but has {null_count} null values in CSV"
                            )
                            
            except Exception as constraint_validation_error:
                collected_warnings.append(f"Constraint validation failed: {constraint_validation_error}")
            
            # Generate comprehensive recommendations
            if collected_errors:
                recommendations.extend([
                    "For existing_schema validation, CSV must exactly match database schema",
                    "Ensure all column names match exactly (case-sensitive)",
                    "Verify all data types are compatible with database schema",
                    "Check that non-nullable columns don't have null values",
                    "Consider using 'new_schema' move_type for more flexibility"
                ])
            else:
                recommendations.extend([
                    "Schema validation passed - CSV is compatible with existing table",
                    f"Ready to replace data in table '{table_info.name}'"
                ])
            
            # Add specific recommendations based on issues found
            if any("Type mismatch" in error for error in collected_errors):
                recommendations.append("Review and convert data types in CSV to match database schema")
            
            if any("Missing columns" in error for error in collected_errors):
                recommendations.append("Add missing columns to CSV or remove them from database schema")
            
            if any("Extra columns" in error for error in collected_errors):
                recommendations.append("Remove extra columns from CSV or add them to database schema")
            
            validation_passed = len(collected_errors) == 0
            
            return ValidationReport(
                schema_analysis=schema_result,
                case_conflicts=case_conflicts,
                type_mismatches=type_mismatches,
                constraint_violations=constraint_violations,
                recommendations=recommendations,
                warnings=collected_warnings,
                errors=collected_errors,
                validation_passed=validation_passed
            )
            
        except Exception as e:
            # Handle any unexpected errors during validation
            collected_errors.append(f"Validation process failed: {e}")
            return self._create_error_report(
                errors=collected_errors,
                warnings=collected_warnings,
                message=f"Existing schema validation failed: {e}"
            )
    
    async def _validate_new_schema_comprehensive(
        self, 
        table_info: TableInfo, 
        df: pd.DataFrame
    ) -> ValidationReport:
        """
        Perform comprehensive flexible validation for new_schema move type.
        
        This method allows column additions/removals but prevents case-sensitivity
        conflicts and collects all validation issues for comprehensive reporting.
        Implements schema evolution with backward compatibility checks.
        """
        collected_errors = []
        collected_warnings = []
        recommendations = []
        
        try:
            # Perform flexible schema validation with error collection
            schema_result = await self.schema_validator.validate_new_schema(table_info, df)
            
            # Enhanced case sensitivity conflict detection (critical in new_schema mode)
            try:
                case_conflicts = self.case_sensitivity_validator.detect_case_conflicts(
                    list(table_info.columns.keys()),
                    list(df.columns)
                )
                
                for conflict in case_conflicts:
                    if conflict.conflict_type == 'case_mismatch':
                        collected_errors.append(
                            f"Case-sensitivity conflict: database column '{conflict.db_column}' "
                            f"conflicts with CSV column '{conflict.csv_column}'. "
                            f"This would cause data corruption in PostgreSQL."
                        )
                    elif conflict.conflict_type == 'duplicate_insensitive':
                        collected_errors.append(
                            f"Duplicate case-insensitive column in CSV: '{conflict.csv_column}'. "
                            f"PostgreSQL treats these as the same column."
                        )
                        
            except Exception as case_validation_error:
                collected_warnings.append(f"Case sensitivity validation failed: {case_validation_error}")
                case_conflicts = []
            
            # Enhanced backward compatibility checks
            backward_compatibility_issues = self._check_backward_compatibility(table_info, df, schema_result)
            collected_warnings.extend(backward_compatibility_issues['warnings'])
            collected_errors.extend(backward_compatibility_issues['errors'])
            recommendations.extend(backward_compatibility_issues['recommendations'])
            
            # Perform data type validation for existing columns with error collection
            try:
                type_mismatches = self.data_type_validator.validate_type_compatibility(table_info, df)
                
                for mismatch in type_mismatches:
                    if not mismatch.compatible and not mismatch.conversion_required:
                        error_msg = (
                            f"Incompatible type in column '{mismatch.column_name}': "
                            f"cannot convert {mismatch.csv_type} to {mismatch.db_type}"
                        )
                        if mismatch.sample_values:
                            error_msg += f" (sample values: {mismatch.sample_values[:3]})"
                        collected_errors.append(error_msg)
                    elif mismatch.conversion_required:
                        collected_warnings.append(
                            f"Type conversion required for column '{mismatch.column_name}': "
                            f"{mismatch.csv_type} -> {mismatch.db_type}"
                        )
                        
            except Exception as type_validation_error:
                collected_warnings.append(f"Type validation failed: {type_validation_error}")
                type_mismatches = []
            
            # Enhanced constraint validation for existing columns
            constraint_violations = []
            try:
                constraint_violations = self._validate_constraints_new_schema(table_info, df)
                
                for violation in constraint_violations:
                    if violation.violation_type == 'null_constraint':
                        collected_errors.append(
                            f"Column '{violation.column_name}' is NOT NULL in database but has "
                            f"{violation.affected_rows} null values in CSV"
                        )
                    elif violation.violation_type == 'primary_key_constraint':
                        collected_errors.append(
                            f"Primary key column '{violation.column_name}' has duplicate or null values in CSV"
                        )
                    elif violation.violation_type == 'unique_constraint':
                        collected_warnings.append(
                            f"Unique constraint on '{violation.column_name}' may be violated by CSV data"
                        )
                            
            except Exception as constraint_validation_error:
                collected_warnings.append(f"Constraint validation failed: {constraint_validation_error}")
            
            # Enhanced schema change reporting and validation
            schema_change_analysis = self._analyze_schema_changes(table_info, df, schema_result)
            collected_warnings.extend(schema_change_analysis['warnings'])
            recommendations.extend(schema_change_analysis['recommendations'])
            
            # Generate comprehensive recommendations based on validation results
            if collected_errors:
                if case_conflicts:
                    recommendations.extend([
                        "Rename conflicting columns to avoid case-sensitivity issues",
                        "Use consistent casing for column names (recommend lowercase with underscores)",
                        "Consider using a column mapping strategy to resolve conflicts"
                    ])
                
                if any("Incompatible type" in error for error in collected_errors):
                    recommendations.extend([
                        "Review and fix incompatible data types in CSV",
                        "Consider data transformation before import",
                        "Verify data quality and format consistency"
                    ])
                
                if any("NOT NULL" in error for error in collected_errors):
                    recommendations.extend([
                        "Remove null values from non-nullable columns",
                        "Consider providing default values for missing data",
                        "Review data completeness requirements"
                    ])
                    
            else:
                recommendations.extend([
                    "Schema validation passed for new_schema mode",
                    f"Ready to update table '{table_info.name}' with flexible schema changes"
                ])
            
            # Add specific recommendations for schema evolution
            if schema_result.requires_schema_update:
                recommendations.extend([
                    "Schema evolution detected - review all changes carefully",
                    "Test schema changes in a development environment first",
                    "Document schema changes for team awareness",
                    "Consider versioning your database schema"
                ])
            
            validation_passed = len(collected_errors) == 0
            
            return ValidationReport(
                schema_analysis=schema_result,
                case_conflicts=case_conflicts,
                type_mismatches=type_mismatches,
                constraint_violations=constraint_violations,
                recommendations=recommendations,
                warnings=collected_warnings,
                errors=collected_errors,
                validation_passed=validation_passed
            )
            
        except Exception as e:
            # Handle any unexpected errors during validation
            collected_errors.append(f"Validation process failed: {e}")
            return self._create_error_report(
                errors=collected_errors,
                warnings=collected_warnings,
                message=f"New schema validation failed: {e}"
            )
    
    def _check_backward_compatibility(
        self, 
        table_info: TableInfo, 
        df: pd.DataFrame, 
        schema_result: 'SchemaAnalysis'
    ) -> Dict[str, List[str]]:
        """
        Check for backward compatibility issues in schema evolution.
        
        Args:
            table_info: Current table information
            df: DataFrame with new data
            schema_result: Schema analysis result
            
        Returns:
            Dictionary with warnings, errors, and recommendations
        """
        warnings = []
        errors = []
        recommendations = []
        
        # Check for removal of primary key columns
        for pk_col in table_info.primary_keys:
            if pk_col in schema_result.columns_removed:
                errors.append(
                    f"Primary key column '{pk_col}' is being removed. "
                    f"This breaks backward compatibility and may cause application failures."
                )
                recommendations.append(
                    f"Consider keeping primary key column '{pk_col}' or provide migration strategy"
                )
        
        # Check for removal of columns that have constraints
        for constraint in table_info.constraints:
            for col in constraint.columns:
                if col in schema_result.columns_removed:
                    if constraint.type in ['UNIQUE', 'FOREIGN KEY']:
                        warnings.append(
                            f"Column '{col}' with {constraint.type} constraint is being removed"
                        )
                        recommendations.append(
                            f"Review impact of removing constrained column '{col}'"
                        )
        
        # Check for removal of indexed columns
        for index in table_info.indexes:
            for col in index.columns:
                if col in schema_result.columns_removed:
                    warnings.append(
                        f"Indexed column '{col}' (index: {index.name}) is being removed"
                    )
                    recommendations.append(
                        f"Consider performance impact of removing indexed column '{col}'"
                    )
        
        # Check for large-scale column removal (potential data loss)
        if len(schema_result.columns_removed) > len(table_info.columns) * 0.5:
            warnings.append(
                f"Removing {len(schema_result.columns_removed)} columns "
                f"({len(schema_result.columns_removed)/len(table_info.columns)*100:.1f}% of table)"
            )
            recommendations.extend([
                "Large-scale column removal detected - verify this is intentional",
                "Consider backing up data before proceeding",
                "Review if this should be a new table instead of schema evolution"
            ])
        
        return {
            'warnings': warnings,
            'errors': errors,
            'recommendations': recommendations
        }
    
    def _validate_constraints_new_schema(
        self, 
        table_info: TableInfo, 
        df: pd.DataFrame
    ) -> List['ConstraintViolation']:
        """
        Validate constraints for new_schema mode with enhanced checking.
        
        Args:
            table_info: Table information with constraints
            df: DataFrame to validate
            
        Returns:
            List of constraint violations found
        """
        from dataload.domain.entities import ConstraintViolation
        
        violations = []
        
        # Check NOT NULL constraints for existing columns
        for col_name, col_info in table_info.columns.items():
            if col_name in df.columns and not col_info.nullable:
                null_count = df[col_name].isnull().sum()
                if null_count > 0:
                    violations.append(ConstraintViolation(
                        constraint_name=f"{col_name}_not_null",
                        constraint_type="NOT NULL",
                        column_name=col_name,
                        violation_type="null_constraint",
                        affected_rows=null_count
                    ))
        
        # Check PRIMARY KEY constraints
        for pk_col in table_info.primary_keys:
            if pk_col in df.columns:
                # Check for nulls in primary key
                null_count = df[pk_col].isnull().sum()
                if null_count > 0:
                    violations.append(ConstraintViolation(
                        constraint_name=f"pk_{pk_col}",
                        constraint_type="PRIMARY KEY",
                        column_name=pk_col,
                        violation_type="primary_key_constraint",
                        affected_rows=null_count
                    ))
                
                # Check for duplicates in primary key
                duplicate_count = df[pk_col].duplicated().sum()
                if duplicate_count > 0:
                    violations.append(ConstraintViolation(
                        constraint_name=f"pk_{pk_col}",
                        constraint_type="PRIMARY KEY",
                        column_name=pk_col,
                        violation_type="primary_key_constraint",
                        affected_rows=duplicate_count
                    ))
        
        # Check UNIQUE constraints
        for constraint in table_info.constraints:
            if constraint.type == 'UNIQUE':
                constraint_columns = [col for col in constraint.columns if col in df.columns]
                if constraint_columns:
                    # Check for duplicates in unique constraint columns
                    if len(constraint_columns) == 1:
                        col = constraint_columns[0]
                        duplicate_count = df[col].duplicated().sum()
                        if duplicate_count > 0:
                            violations.append(ConstraintViolation(
                                constraint_name=constraint.name,
                                constraint_type="UNIQUE",
                                column_name=col,
                                violation_type="unique_constraint",
                                affected_rows=duplicate_count
                            ))
                    else:
                        # Multi-column unique constraint
                        duplicate_count = df[constraint_columns].duplicated().sum()
                        if duplicate_count > 0:
                            violations.append(ConstraintViolation(
                                constraint_name=constraint.name,
                                constraint_type="UNIQUE",
                                column_name=", ".join(constraint_columns),
                                violation_type="unique_constraint",
                                affected_rows=duplicate_count
                            ))
        
        return violations
    
    def _analyze_schema_changes(
        self, 
        table_info: TableInfo, 
        df: pd.DataFrame, 
        schema_result: 'SchemaAnalysis'
    ) -> Dict[str, List[str]]:
        """
        Analyze and report on schema changes for new_schema mode.
        
        Args:
            table_info: Current table information
            df: DataFrame with new data
            schema_result: Schema analysis result
            
        Returns:
            Dictionary with warnings and recommendations
        """
        warnings = []
        recommendations = []
        
        # Detailed reporting on column additions
        if schema_result.columns_added:
            warnings.append(
                f"New columns will be added to table: {', '.join(schema_result.columns_added)}"
            )
            
            # Analyze new column types
            new_column_types = {}
            for col in schema_result.columns_added:
                if col in df.columns:
                    inferred_type = self.data_type_validator._infer_csv_column_type(df[col])
                    new_column_types[col] = inferred_type
            
            if new_column_types:
                type_summary = ", ".join([f"{col}({typ})" for col, typ in new_column_types.items()])
                warnings.append(f"New column types: {type_summary}")
            
            recommendations.extend([
                "Review new columns and their inferred data types",
                "Consider adding appropriate constraints to new columns",
                "Verify that new columns don't conflict with application expectations"
            ])
        
        # Detailed reporting on column removals
        if schema_result.columns_removed:
            warnings.append(
                f"Existing columns will be removed from table: {', '.join(schema_result.columns_removed)}"
            )
            
            # Analyze removed column types and constraints
            removed_column_info = []
            for col in schema_result.columns_removed:
                if col in table_info.columns:
                    col_info = table_info.columns[col]
                    removed_column_info.append(f"{col}({col_info.data_type})")
            
            if removed_column_info:
                warnings.append(f"Removed column types: {', '.join(removed_column_info)}")
            
            recommendations.extend([
                "Ensure removed columns don't contain important data",
                "Consider exporting data from removed columns before proceeding",
                "Verify that applications can handle missing columns"
            ])
        
        # Schema evolution impact analysis
        total_columns = len(table_info.columns)
        change_percentage = (len(schema_result.columns_added) + len(schema_result.columns_removed)) / total_columns * 100
        
        if change_percentage > 25:
            warnings.append(
                f"Significant schema change detected: {change_percentage:.1f}% of columns affected"
            )
            recommendations.extend([
                "Consider if this should be a new table instead of schema evolution",
                "Plan for comprehensive testing of schema changes",
                "Review impact on dependent systems and applications"
            ])
        
        # Performance impact warnings
        if len(schema_result.columns_added) > 10:
            warnings.append("Adding many columns may impact query performance")
            recommendations.append("Consider performance testing after schema changes")
        
        # Data migration complexity warnings
        if schema_result.columns_added and schema_result.columns_removed:
            warnings.append("Both adding and removing columns - complex schema evolution")
            recommendations.extend([
                "Consider breaking this into multiple migration steps",
                "Test the complete migration process thoroughly",
                "Document the schema evolution for future reference"
            ])
        
        return {
            'warnings': warnings,
            'recommendations': recommendations
        }

    def _create_error_report(
        self, 
        errors: List[str], 
        warnings: List[str], 
        message: str
    ) -> ValidationReport:
        """
        Create a validation report for error cases.
        
        Args:
            errors: List of error messages
            warnings: List of warning messages  
            message: Main error message
            
        Returns:
            ValidationReport indicating validation failure
        """
        from dataload.domain.entities import SchemaAnalysis
        
        # Create a minimal schema analysis for error cases
        error_schema_analysis = SchemaAnalysis(
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
            schema_analysis=error_schema_analysis,
            case_conflicts=[],
            type_mismatches=[],
            constraint_violations=[],
            recommendations=[
                "Fix validation errors before proceeding",
                "Review input parameters and data format"
            ],
            warnings=warnings,
            errors=errors,
            validation_passed=False
        )