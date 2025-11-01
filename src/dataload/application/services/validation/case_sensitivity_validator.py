"""Case sensitivity validation for DataMove operations."""

from typing import List, Dict, Set, Any
from dataload.domain.entities import CaseConflict


class CaseSensitivityValidator:
    """
    Validates and detects case-sensitivity conflicts between database columns and CSV columns.
    Critical for new_schema mode where case conflicts can cause data corruption.
    """
    
    def detect_case_conflicts(
        self, 
        db_columns: List[str], 
        csv_columns: List[str]
    ) -> List[CaseConflict]:
        """
        Detect case-sensitivity conflicts between database and CSV columns.
        
        Args:
            db_columns: List of database column names
            csv_columns: List of CSV column names
            
        Returns:
            List of CaseConflict objects describing conflicts found
        """
        conflicts = []
        
        # Create case-insensitive mappings for detection
        db_lower_map = self._create_case_insensitive_map(db_columns)
        csv_lower_map = self._create_case_insensitive_map(csv_columns)
        
        # Check for conflicts between DB and CSV columns
        conflicts.extend(self._find_cross_conflicts(db_lower_map, csv_lower_map))
        
        # Check for internal CSV conflicts (multiple CSV columns with same case-insensitive name)
        conflicts.extend(self._find_internal_csv_conflicts(csv_lower_map))
        
        return conflicts
    
    def _create_case_insensitive_map(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Create a mapping from lowercase column names to original column names.
        
        Args:
            columns: List of column names
            
        Returns:
            Dictionary mapping lowercase names to list of original names
        """
        case_map = {}
        for col in columns:
            lower_col = col.lower()
            if lower_col not in case_map:
                case_map[lower_col] = []
            case_map[lower_col].append(col)
        
        return case_map
    
    def _find_cross_conflicts(
        self, 
        db_lower_map: Dict[str, List[str]], 
        csv_lower_map: Dict[str, List[str]]
    ) -> List[CaseConflict]:
        """
        Find conflicts between database and CSV columns.
        
        Args:
            db_lower_map: Case-insensitive mapping of database columns
            csv_lower_map: Case-insensitive mapping of CSV columns
            
        Returns:
            List of conflicts between database and CSV columns
        """
        conflicts = []
        
        for lower_name in db_lower_map:
            if lower_name in csv_lower_map:
                db_originals = db_lower_map[lower_name]
                csv_originals = csv_lower_map[lower_name]
                
                # Check if any combinations have different cases
                for db_col in db_originals:
                    for csv_col in csv_originals:
                        if db_col != csv_col:  # Different case
                            conflicts.append(CaseConflict(
                                db_column=db_col,
                                csv_column=csv_col,
                                conflict_type="case_mismatch"
                            ))
        
        return conflicts
    
    def _find_internal_csv_conflicts(
        self, 
        csv_lower_map: Dict[str, List[str]]
    ) -> List[CaseConflict]:
        """
        Find internal conflicts within CSV columns (multiple columns with same case-insensitive name).
        
        Args:
            csv_lower_map: Case-insensitive mapping of CSV columns
            
        Returns:
            List of internal CSV conflicts
        """
        conflicts = []
        
        for lower_name, originals in csv_lower_map.items():
            if len(originals) > 1:
                # Multiple CSV columns with same case-insensitive name
                for i, col1 in enumerate(originals):
                    for col2 in originals[i+1:]:
                        conflicts.append(CaseConflict(
                            db_column="",  # No DB column involved
                            csv_column=f"{col1}, {col2}",
                            conflict_type="duplicate_insensitive"
                        ))
        
        return conflicts
    
    def has_case_conflicts(
        self, 
        db_columns: List[str], 
        csv_columns: List[str]
    ) -> bool:
        """
        Quick check if there are any case-sensitivity conflicts.
        
        Args:
            db_columns: List of database column names
            csv_columns: List of CSV column names
            
        Returns:
            True if conflicts exist, False otherwise
        """
        conflicts = self.detect_case_conflicts(db_columns, csv_columns)
        return len(conflicts) > 0
    
    def get_safe_column_mapping(
        self, 
        db_columns: List[str], 
        csv_columns: List[str]
    ) -> Dict[str, str]:
        """
        Generate a safe column mapping that avoids case conflicts.
        
        Args:
            db_columns: List of database column names
            csv_columns: List of CSV column names
            
        Returns:
            Dictionary mapping CSV column names to safe database column names
        """
        mapping = {}
        conflicts = self.detect_case_conflicts(db_columns, csv_columns)
        
        # Start with direct mapping for non-conflicting columns
        for csv_col in csv_columns:
            mapping[csv_col] = csv_col
        
        # Resolve conflicts by suggesting alternative names
        for conflict in conflicts:
            if conflict.conflict_type == "case_mismatch":
                # Suggest using the database column name
                mapping[conflict.csv_column] = conflict.db_column
        
        return mapping
    
    def suggest_conflict_resolution(
        self, 
        db_columns: List[str], 
        csv_columns: List[str]
    ) -> Dict[str, List[str]]:
        """
        Suggest resolutions for case-sensitivity conflicts.
        
        Args:
            db_columns: List of database column names
            csv_columns: List of CSV column names
            
        Returns:
            Dictionary with conflict resolution suggestions
        """
        conflicts = self.detect_case_conflicts(db_columns, csv_columns)
        suggestions = {
            'rename_csv_columns': [],
            'rename_db_columns': [],
            'use_column_mapping': [],
            'general_recommendations': []
        }
        
        for conflict in conflicts:
            if conflict.conflict_type == "case_mismatch":
                suggestions['rename_csv_columns'].append(
                    f"Rename CSV column '{conflict.csv_column}' to '{conflict.db_column}'"
                )
                suggestions['use_column_mapping'].append(
                    f"Map '{conflict.csv_column}' -> '{conflict.db_column}'"
                )
            elif conflict.conflict_type == "duplicate_insensitive":
                csv_cols = conflict.csv_column.split(", ")
                suggestions['rename_csv_columns'].append(
                    f"Rename one of the duplicate CSV columns: {conflict.csv_column}"
                )
        
        # Add general recommendations
        if conflicts:
            suggestions['general_recommendations'].extend([
                "Use consistent casing for all column names (recommend lowercase with underscores)",
                "Avoid column names that differ only in case",
                "Consider using a naming convention standard for your team",
                "Test column name changes in a development environment first"
            ])
        
        return suggestions
    
    def validate_column_naming_convention(
        self, 
        columns: List[str]
    ) -> Dict[str, List[str]]:
        """
        Validate column names against PostgreSQL best practices.
        
        Args:
            columns: List of column names to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        issues = {
            'warnings': [],
            'recommendations': [],
            'best_practices': []
        }
        
        for col in columns:
            # Check for mixed case
            if col != col.lower() and col != col.upper():
                issues['warnings'].append(
                    f"Column '{col}' uses mixed case - may cause confusion"
                )
            
            # Check for spaces
            if ' ' in col:
                issues['warnings'].append(
                    f"Column '{col}' contains spaces - requires quoting in SQL"
                )
            
            # Check for special characters
            special_chars = set(col) - set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
            if special_chars:
                issues['warnings'].append(
                    f"Column '{col}' contains special characters: {special_chars}"
                )
            
            # Check for reserved words (basic check)
            reserved_words = {
                'user', 'order', 'group', 'table', 'index', 'select', 'insert', 
                'update', 'delete', 'create', 'drop', 'alter', 'where', 'from'
            }
            if col.lower() in reserved_words:
                issues['warnings'].append(
                    f"Column '{col}' is a PostgreSQL reserved word"
                )
        
        # Add best practice recommendations
        if any(' ' in col for col in columns):
            issues['recommendations'].append("Replace spaces with underscores in column names")
        
        if any(col != col.lower() for col in columns):
            issues['recommendations'].append("Use lowercase column names for consistency")
        
        issues['best_practices'].extend([
            "Use snake_case for column names (e.g., 'user_id', 'created_at')",
            "Keep column names descriptive but concise",
            "Avoid abbreviations that might be unclear",
            "Use consistent naming patterns across your schema"
        ])
        
        return issues
    
    def prevent_case_conflicts_in_new_schema(
        self, 
        db_columns: List[str], 
        csv_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive case conflict prevention for new_schema mode.
        
        Args:
            db_columns: List of database column names
            csv_columns: List of CSV column names
            
        Returns:
            Dictionary with conflict analysis and prevention strategies
        """
        conflicts = self.detect_case_conflicts(db_columns, csv_columns)
        
        result = {
            'has_conflicts': len(conflicts) > 0,
            'conflicts': conflicts,
            'conflict_count': len(conflicts),
            'prevention_required': False,
            'safe_to_proceed': True,
            'resolution_strategies': [],
            'risk_assessment': 'low'
        }
        
        if conflicts:
            result['prevention_required'] = True
            result['safe_to_proceed'] = False
            
            # Assess risk level
            case_mismatch_count = sum(1 for c in conflicts if c.conflict_type == 'case_mismatch')
            duplicate_count = sum(1 for c in conflicts if c.conflict_type == 'duplicate_insensitive')
            
            if duplicate_count > 0:
                result['risk_assessment'] = 'high'
                result['resolution_strategies'].append(
                    "CRITICAL: Resolve duplicate case-insensitive columns before proceeding"
                )
            elif case_mismatch_count > 3:
                result['risk_assessment'] = 'medium'
                result['resolution_strategies'].append(
                    "Multiple case conflicts detected - systematic renaming recommended"
                )
            else:
                result['risk_assessment'] = 'medium'
            
            # Add specific resolution strategies
            result['resolution_strategies'].extend([
                "Use column mapping to resolve conflicts without changing source data",
                "Standardize on lowercase column names with underscores",
                "Implement pre-processing to normalize column names",
                "Consider using a data transformation pipeline"
            ])
        
        return result