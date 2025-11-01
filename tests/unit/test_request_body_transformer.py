"""
Unit tests for RequestBodyTransformer functionality.

This module contains comprehensive tests for the RequestBodyTransformer class,
covering all transformation types, error handling, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import patch, MagicMock

from src.dataload.domain.request_body_transformer import RequestBodyTransformer
from src.dataload.domain.api_entities import (
    TransformationRule,
    RequestTransformationConfig,
    TransformationResult,
    RequestTransformationError
)
from src.dataload.domain.entities import ValidationError


class TestRequestBodyTransformer:
    """Test cases for RequestBodyTransformer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = RequestBodyTransformer()
        
        # Sample DataFrame for testing
        self.sample_df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'age': [30, 25, 35],
            'salary': [50000.0, 60000.0, 75000.0],
            'department': ['Engineering', 'Marketing', 'Sales'],
            'active': [True, True, False]
        })
    
    def test_constant_transformation(self):
        """Test constant value transformation."""
        rule = TransformationRule(
            target_field='created_by',
            source_expression='',
            transformation_type='constant',
            default_value='system'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['created_by'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        assert result.success
        assert 'created_by' in result.transformed_dataframe.columns
        assert all(result.transformed_dataframe['created_by'] == 'system')
        assert len(result.applied_transformations) == 1
        assert 'created_by' in result.applied_transformations
    
    def test_copy_transformation(self):
        """Test copying values from existing column."""
        rule = TransformationRule(
            target_field='employee_name',
            source_expression='name',
            transformation_type='copy'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['employee_name'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        assert result.success
        assert 'employee_name' in result.transformed_dataframe.columns
        pd.testing.assert_series_equal(
            result.transformed_dataframe['employee_name'],
            self.sample_df['name'],
            check_names=False
        )
    
    def test_copy_transformation_missing_column_required(self):
        """Test copy transformation with missing required column."""
        rule = TransformationRule(
            target_field='missing_field',
            source_expression='nonexistent_column',
            transformation_type='copy',
            required=True
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['missing_field'],
            fail_on_error=True
        )
        
        with pytest.raises(RequestTransformationError):
            self.transformer.transform_data(self.sample_df, config)
    
    def test_copy_transformation_missing_column_optional(self):
        """Test copy transformation with missing optional column."""
        rule = TransformationRule(
            target_field='optional_field',
            source_expression='nonexistent_column',
            transformation_type='copy',
            required=False,
            default_value='N/A'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['optional_field'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        assert result.success
        assert 'optional_field' in result.transformed_dataframe.columns
        assert all(result.transformed_dataframe['optional_field'] == 'N/A')
    
    def test_concat_transformation(self):
        """Test concatenation transformation."""
        rule = TransformationRule(
            target_field='full_info',
            source_expression="concat({name}, ' (', str({age}), ' years old)')",
            transformation_type='concat'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['full_info'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        assert result.success
        assert 'full_info' in result.transformed_dataframe.columns
        
        expected_values = [
            'John Doe (30 years old)',
            'Jane Smith (25 years old)',
            'Bob Johnson (35 years old)'
        ]
        
        for i, expected in enumerate(expected_values):
            assert expected in str(result.transformed_dataframe['full_info'].iloc[i])
    
    def test_compute_transformation_arithmetic(self):
        """Test computation transformation with arithmetic operations."""
        rule = TransformationRule(
            target_field='annual_bonus',
            source_expression='{salary} * 0.1',
            transformation_type='compute'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['annual_bonus'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        assert result.success
        assert 'annual_bonus' in result.transformed_dataframe.columns
        
        expected_bonus = self.sample_df['salary'] * 0.1
        pd.testing.assert_series_equal(
            result.transformed_dataframe['annual_bonus'],
            expected_bonus,
            check_names=False
        )
    
    def test_compute_transformation_conditional(self):
        """Test computation transformation with conditional logic."""
        rule = TransformationRule(
            target_field='status',
            source_expression="'Active' if {active} else 'Inactive'",
            transformation_type='compute'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['status'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        assert result.success
        assert 'status' in result.transformed_dataframe.columns
        
        expected_status = ['Active', 'Active', 'Inactive']
        assert result.transformed_dataframe['status'].tolist() == expected_status
    
    def test_multiple_transformations_with_dependencies(self):
        """Test multiple transformations with proper dependency order."""
        rules = [
            TransformationRule(
                target_field='base_salary',
                source_expression='salary',
                transformation_type='copy'
            ),
            TransformationRule(
                target_field='bonus',
                source_expression='{base_salary} * 0.15',
                transformation_type='compute'
            ),
            TransformationRule(
                target_field='total_compensation',
                source_expression='{base_salary} + {bonus}',
                transformation_type='compute'
            )
        ]
        
        config = RequestTransformationConfig(
            transformation_rules=rules,
            execution_order=['base_salary', 'bonus', 'total_compensation'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        assert result.success
        assert len(result.applied_transformations) == 3
        
        # Verify calculations
        expected_bonus = self.sample_df['salary'] * 0.15
        expected_total = self.sample_df['salary'] + expected_bonus
        
        pd.testing.assert_series_equal(
            result.transformed_dataframe['bonus'],
            expected_bonus,
            check_names=False
        )
        pd.testing.assert_series_equal(
            result.transformed_dataframe['total_compensation'],
            expected_total,
            check_names=False
        )
    
    def test_execution_order_validation(self):
        """Test validation of execution order."""
        rules = [
            TransformationRule(
                target_field='field_a',
                source_expression='{field_b}',
                transformation_type='copy'
            ),
            TransformationRule(
                target_field='field_b',
                source_expression='constant_value',
                transformation_type='constant',
                default_value='test'
            )
        ]
        
        # Wrong order - field_a depends on field_b but comes first
        wrong_order = ['field_a', 'field_b']
        
        # Should raise error due to unsatisfied dependency
        with pytest.raises(RequestTransformationError, match="unsatisfied dependencies"):
            self.transformer.validate_transformation_order(rules, wrong_order)
        
        # Correct order should work
        correct_order = ['field_b', 'field_a']
        validated_order = self.transformer.validate_transformation_order(rules, correct_order)
        assert validated_order == correct_order
    
    def test_suggest_execution_order(self):
        """Test automatic suggestion of execution order."""
        rules = [
            TransformationRule(
                target_field='field_c',
                source_expression='{field_a} + {field_b}',
                transformation_type='compute'
            ),
            TransformationRule(
                target_field='field_a',
                source_expression='salary',
                transformation_type='copy'
            ),
            TransformationRule(
                target_field='field_b',
                source_expression='age',
                transformation_type='copy'
            )
        ]
        
        suggested_order = self.transformer.suggest_execution_order(rules)
        
        # field_c should come after field_a and field_b
        assert suggested_order.index('field_c') > suggested_order.index('field_a')
        assert suggested_order.index('field_c') > suggested_order.index('field_b')
        assert len(suggested_order) == 3
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        rules = [
            TransformationRule(
                target_field='field_a',
                source_expression='{field_b}',
                transformation_type='copy'
            ),
            TransformationRule(
                target_field='field_b',
                source_expression='{field_a}',
                transformation_type='copy'
            )
        ]
        
        with pytest.raises(RequestTransformationError, match="Circular dependencies"):
            self.transformer.suggest_execution_order(rules)
    
    def test_invalid_transformation_type(self):
        """Test handling of invalid transformation type."""
        rule = TransformationRule(
            target_field='invalid_field',
            source_expression='name',
            transformation_type='invalid_type'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['invalid_field'],
            fail_on_error=True
        )
        
        with pytest.raises(RequestTransformationError, match="Invalid transformation type"):
            self.transformer.transform_data(self.sample_df, config)
    
    def test_expression_safety_validation(self):
        """Test validation of potentially dangerous expressions."""
        dangerous_expressions = [
            '__import__("os").system("rm -rf /")',
            'exec("print(\'dangerous\')")',
            'eval("1+1")',
            'open("/etc/passwd").read()'
        ]
        
        for expr in dangerous_expressions:
            rule = TransformationRule(
                target_field='dangerous_field',
                source_expression=expr,
                transformation_type='compute'
            )
            
            config = RequestTransformationConfig(
                transformation_rules=[rule],
                execution_order=['dangerous_field'],
                fail_on_error=True
            )
            
            with pytest.raises(RequestTransformationError):
                self.transformer.transform_data(self.sample_df, config)
    
    def test_fail_on_error_false(self):
        """Test behavior when fail_on_error is False."""
        rules = [
            TransformationRule(
                target_field='valid_field',
                source_expression='name',
                transformation_type='copy'
            ),
            TransformationRule(
                target_field='invalid_field',
                source_expression='nonexistent_column',
                transformation_type='copy',
                required=True
            )
        ]
        
        config = RequestTransformationConfig(
            transformation_rules=rules,
            execution_order=['valid_field', 'invalid_field'],
            fail_on_error=False
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        # Should not raise error, but should have failures and warnings
        assert not result.success  # Has failed transformations
        assert len(result.applied_transformations) == 1  # Only valid_field succeeded
        assert len(result.failed_transformations) == 1   # invalid_field failed
        assert len(result.warnings) > 0
    
    def test_preserve_original_data_false(self):
        """Test behavior when preserve_original_data is False."""
        rule = TransformationRule(
            target_field='new_field',
            source_expression='test_value',
            transformation_type='constant',
            default_value='test'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['new_field'],
            preserve_original_data=False
        )
        
        original_df = self.sample_df.copy()
        result = self.transformer.transform_data(self.sample_df, config)
        
        # Original DataFrame should be modified
        assert 'new_field' in self.sample_df.columns
        assert result.transformed_dataframe is self.sample_df
    
    def test_transformation_preview(self):
        """Test transformation preview functionality."""
        rule = TransformationRule(
            target_field='preview_field',
            source_expression='{name} - {department}',
            transformation_type='concat'
        )
        
        preview = self.transformer.get_transformation_preview(
            self.sample_df, 
            rule, 
            sample_size=2
        )
        
        assert preview['success'] is True
        assert preview['rule']['target_field'] == 'preview_field'
        assert len(preview['sample_input']) == 2
        assert len(preview['sample_output']) == 2
        assert preview['error'] is None
    
    def test_transformation_preview_with_error(self):
        """Test transformation preview with invalid rule."""
        rule = TransformationRule(
            target_field='error_field',
            source_expression='{nonexistent_column}',
            transformation_type='copy',
            required=True
        )
        
        preview = self.transformer.get_transformation_preview(
            self.sample_df, 
            rule, 
            sample_size=2
        )
        
        assert preview['success'] is False
        assert preview['error'] is not None
        assert preview['sample_output'] is None
    
    def test_column_reference_extraction(self):
        """Test extraction of column references from expressions."""
        expressions_and_expected = [
            ('{name} + {age}', ['name', 'age']),
            ('concat({first_name}, " ", {last_name})', ['first_name', 'last_name']),
            ('{salary} * 1.1 + {bonus}', ['salary', 'bonus']),
            ('constant_value', []),
            ('{single_column}', ['single_column'])
        ]
        
        for expression, expected_columns in expressions_and_expected:
            extracted = self.transformer._extract_column_references(expression)
            assert set(extracted) == set(expected_columns)
    
    def test_transformation_statistics(self):
        """Test that transformation statistics are properly calculated."""
        rules = [
            TransformationRule(
                target_field='field1',
                source_expression='name',
                transformation_type='copy'
            ),
            TransformationRule(
                target_field='field2',
                source_expression='test',
                transformation_type='constant',
                default_value='test'
            )
        ]
        
        config = RequestTransformationConfig(
            transformation_rules=rules,
            execution_order=['field1', 'field2'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        stats = result.execution_stats
        assert stats['total_rules'] == 2
        assert stats['successful_transformations'] == 2
        assert stats['failed_transformations'] == 0
        assert stats['rows_processed'] == len(self.sample_df)
        assert stats['columns_before'] == len(self.sample_df.columns)
        assert stats['columns_after'] == len(self.sample_df.columns) + 2
        assert 'execution_time_seconds' in stats
    
    def test_empty_transformation_rules(self):
        """Test handling of empty transformation rules."""
        config = RequestTransformationConfig(
            transformation_rules=[],
            execution_order=[],
            fail_on_error=True
        )
        
        with pytest.raises(RequestTransformationError, match="No transformation rules provided"):
            self.transformer.transform_data(self.sample_df, config)
    
    def test_missing_default_value_for_constant(self):
        """Test validation of constant transformation without default value."""
        rule = TransformationRule(
            target_field='constant_field',
            source_expression='',
            transformation_type='constant'
            # Missing default_value
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['constant_field'],
            fail_on_error=True
        )
        
        with pytest.raises(RequestTransformationError, match="Default value is required"):
            self.transformer.transform_data(self.sample_df, config)
    
    def test_datetime_transformations(self):
        """Test transformations involving datetime functions."""
        rule = TransformationRule(
            target_field='created_at',
            source_expression='now()',
            transformation_type='compute'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['created_at'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        assert result.success
        assert 'created_at' in result.transformed_dataframe.columns
        
        # Check that all values are datetime objects
        for value in result.transformed_dataframe['created_at']:
            assert isinstance(value, datetime)
    
    def test_string_manipulation_functions(self):
        """Test string manipulation functions in transformations."""
        rule = TransformationRule(
            target_field='upper_name',
            source_expression='upper({name})',
            transformation_type='compute'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['upper_name'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        assert result.success
        assert 'upper_name' in result.transformed_dataframe.columns
        
        expected_upper = self.sample_df['name'].str.upper()
        pd.testing.assert_series_equal(
            result.transformed_dataframe['upper_name'],
            expected_upper,
            check_names=False
        )
    
    def test_mathematical_functions(self):
        """Test mathematical functions in transformations."""
        rule = TransformationRule(
            target_field='rounded_salary',
            source_expression='round({salary} / 1000) * 1000',
            transformation_type='compute'
        )
        
        config = RequestTransformationConfig(
            transformation_rules=[rule],
            execution_order=['rounded_salary'],
            fail_on_error=True
        )
        
        result = self.transformer.transform_data(self.sample_df, config)
        
        assert result.success
        assert 'rounded_salary' in result.transformed_dataframe.columns
        
        # Verify rounding logic
        expected_rounded = (self.sample_df['salary'] / 1000).round() * 1000
        pd.testing.assert_series_equal(
            result.transformed_dataframe['rounded_salary'],
            expected_rounded,
            check_names=False
        )


class TestTransformationRuleValidation:
    """Test cases for transformation rule validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = RequestBodyTransformer()
        self.sample_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
    
    def test_valid_transformation_rule(self):
        """Test validation of valid transformation rule."""
        rule = TransformationRule(
            target_field='valid_field',
            source_expression='col1',
            transformation_type='copy'
        )
        
        # Should not raise any exception
        self.transformer._validate_transformation_rule(rule, self.sample_df)
    
    def test_invalid_transformation_type(self):
        """Test validation with invalid transformation type."""
        rule = TransformationRule(
            target_field='invalid_field',
            source_expression='col1',
            transformation_type='invalid'
        )
        
        with pytest.raises(RequestTransformationError, match="Invalid transformation type"):
            self.transformer._validate_transformation_rule(rule, self.sample_df)
    
    def test_missing_source_for_copy(self):
        """Test validation of copy rule without source expression."""
        # This should fail at the dataclass level due to __post_init__ validation
        with pytest.raises(ValidationError, match="Source expression is required"):
            TransformationRule(
                target_field='copy_field',
                source_expression='',
                transformation_type='copy'
            )
    
    def test_nonexistent_source_column_required(self):
        """Test validation with nonexistent required source column."""
        rule = TransformationRule(
            target_field='copy_field',
            source_expression='nonexistent_col',
            transformation_type='copy',
            required=True
        )
        
        with pytest.raises(RequestTransformationError, match="Source column .* not found"):
            self.transformer._validate_transformation_rule(rule, self.sample_df)
    
    def test_nonexistent_source_column_optional(self):
        """Test validation with nonexistent optional source column."""
        rule = TransformationRule(
            target_field='copy_field',
            source_expression='nonexistent_col',
            transformation_type='copy',
            required=False
        )
        
        # Should not raise exception for optional columns
        self.transformer._validate_transformation_rule(rule, self.sample_df)
    
    def test_invalid_expression_syntax(self):
        """Test validation with invalid expression syntax."""
        rule = TransformationRule(
            target_field='compute_field',
            source_expression='{col1} + (',  # Invalid syntax
            transformation_type='compute'
        )
        
        with pytest.raises(RequestTransformationError, match="Invalid expression syntax"):
            self.transformer._validate_transformation_rule(rule, self.sample_df)
    
    def test_expression_with_nonexistent_columns(self):
        """Test validation of expression referencing nonexistent columns."""
        rule = TransformationRule(
            target_field='compute_field',
            source_expression='{col1} + {nonexistent_col}',
            transformation_type='compute'
        )
        
        with pytest.raises(RequestTransformationError, match="Expression references non-existent columns"):
            # This will fail during validation
            config = RequestTransformationConfig(
                transformation_rules=[rule],
                execution_order=['compute_field'],
                fail_on_error=True
            )
            self.transformer.transform_data(self.sample_df, config)


if __name__ == '__main__':
    pytest.main([__file__])