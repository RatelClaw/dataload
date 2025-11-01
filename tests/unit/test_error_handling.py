"""
Unit tests for comprehensive error handling functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock

from dataload.application.services.error_handling import (
    ErrorCollector,
    ErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    with_error_handling,
    create_enhanced_error
)
from dataload.domain.entities import (
    DataMoveError,
    ValidationError,
    DatabaseOperationError
)


class TestErrorCollector:
    """Test error collection functionality."""
    
    def test_error_collector_initialization(self):
        """Test error collector initializes correctly."""
        collector = ErrorCollector()
        assert len(collector.errors) == 0
        assert len(collector.warnings) == 0
        assert collector.operation_context is None
    
    def test_set_operation_context(self):
        """Test setting operation context."""
        collector = ErrorCollector()
        collector.set_operation_context("test_op", "test_stage", param1="value1")
        
        assert collector.operation_context.operation == "test_op"
        assert collector.operation_context.stage == "test_stage"
        assert collector.operation_context.parameters["param1"] == "value1"
    
    def test_add_error(self):
        """Test adding errors to collector."""
        collector = ErrorCollector()
        collector.set_operation_context("test_op", "test_stage")
        
        test_error = ValueError("Test error")
        collector.add_error(
            test_error,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION
        )
        
        assert len(collector.errors) == 1
        collected_error = collector.errors[0]
        assert collected_error.error == test_error
        assert collected_error.severity == ErrorSeverity.HIGH
        assert collected_error.category == ErrorCategory.VALIDATION
    
    def test_add_warning(self):
        """Test adding warnings to collector."""
        collector = ErrorCollector()
        collector.add_warning("Test warning")
        
        assert len(collector.warnings) == 1
        assert collector.warnings[0] == "Test warning"
    
    def test_has_critical_errors(self):
        """Test checking for critical errors."""
        collector = ErrorCollector()
        collector.set_operation_context("test_op", "test_stage")
        
        # Add non-critical error
        collector.add_error(ValueError("Test"), severity=ErrorSeverity.MEDIUM)
        assert not collector.has_critical_errors()
        
        # Add critical error
        collector.add_error(ValueError("Critical"), severity=ErrorSeverity.CRITICAL)
        assert collector.has_critical_errors()
    
    def test_get_errors_by_category(self):
        """Test filtering errors by category."""
        collector = ErrorCollector()
        collector.set_operation_context("test_op", "test_stage")
        
        collector.add_error(ValueError("Val error"), category=ErrorCategory.VALIDATION)
        collector.add_error(ConnectionError("DB error"), category=ErrorCategory.DATABASE)
        collector.add_error(FileNotFoundError("File error"), category=ErrorCategory.FILE_IO)
        
        validation_errors = collector.get_errors_by_category(ErrorCategory.VALIDATION)
        assert len(validation_errors) == 1
        assert isinstance(validation_errors[0].error, ValueError)
        
        db_errors = collector.get_errors_by_category(ErrorCategory.DATABASE)
        assert len(db_errors) == 1
        assert isinstance(db_errors[0].error, ConnectionError)


class TestErrorHandler:
    """Test error handler functionality."""
    
    @pytest.mark.asyncio
    async def test_handle_operation_success(self):
        """Test successful operation handling."""
        handler = ErrorHandler()
        
        async def successful_operation():
            return "success"
        
        async with handler.handle_operation("test_op", "test_stage"):
            result = await successful_operation()
            assert result == "success"
    
    @pytest.mark.asyncio
    async def test_handle_operation_with_known_error(self):
        """Test handling known errors."""
        handler = ErrorHandler()
        
        async def failing_operation():
            raise ValidationError("Test validation error")
        
        with pytest.raises(ValidationError):
            async with handler.handle_operation("test_op", "test_stage"):
                await failing_operation()
        
        assert len(handler.collector.errors) == 1
        assert isinstance(handler.collector.errors[0].error, ValidationError)
    
    @pytest.mark.asyncio
    async def test_handle_operation_with_unexpected_error(self):
        """Test handling unexpected errors."""
        handler = ErrorHandler()
        
        async def failing_operation():
            raise RuntimeError("Unexpected error")
        
        with pytest.raises(DataMoveError):
            async with handler.handle_operation("test_op", "test_stage"):
                await failing_operation()
        
        assert len(handler.collector.errors) == 1
        assert isinstance(handler.collector.errors[0].error, RuntimeError)
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """Test retry logic with eventual success."""
        handler = ErrorHandler()
        
        call_count = 0
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = await handler.execute_with_retry(
            flaky_operation,
            max_retries=3,
            retry_delay=0.01  # Fast retry for testing
        )
        
        assert result == "success"
        assert call_count == 3
        assert len(handler.collector.warnings) == 2  # Two retry warnings
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self):
        """Test retry logic when all attempts fail."""
        handler = ErrorHandler()
        
        async def always_failing_operation():
            raise ConnectionError("Always fails")
        
        with pytest.raises(ConnectionError):
            await handler.execute_with_retry(
                always_failing_operation,
                max_retries=2,
                retry_delay=0.01
            )
        
        assert len(handler.collector.errors) == 1
        assert len(handler.collector.warnings) == 2  # Two retry warnings
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        handler = ErrorHandler()
        
        call_count = 0
        async def operation_with_validation_error():
            nonlocal call_count
            call_count += 1
            raise ValidationError("Validation failed")
        
        with pytest.raises(ValidationError):
            await handler.execute_with_retry(
                operation_with_validation_error,
                max_retries=3,
                retry_delay=0.01
            )
        
        assert call_count == 1  # Should not retry validation errors
        assert len(handler.collector.errors) == 1


class TestConvenienceFunctions:
    """Test convenience functions for error handling."""
    
    @pytest.mark.asyncio
    async def test_with_error_handling_success(self):
        """Test with_error_handling convenience function."""
        async def successful_operation():
            return "success"
        
        result = await with_error_handling(
            successful_operation,
            "test_operation",
            "test_stage"
        )
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_with_error_handling_failure(self):
        """Test with_error_handling with failure."""
        async def failing_operation():
            raise ValueError("Test error")
        
        with pytest.raises(DataMoveError):
            await with_error_handling(
                failing_operation,
                "test_operation",
                "test_stage"
            )
    
    def test_create_enhanced_error(self):
        """Test creating enhanced errors."""
        original_error = ValueError("Original error")
        
        enhanced_error = create_enhanced_error(
            original_error,
            "test_operation",
            "test_stage",
            additional_param="test_value"
        )
        
        assert isinstance(enhanced_error, DataMoveError)
        assert "test_operation" in str(enhanced_error)
        assert enhanced_error.context["operation"] == "test_operation"
        assert enhanced_error.context["stage"] == "test_stage"
        assert enhanced_error.context["additional_param"] == "test_value"
    
    def test_create_enhanced_error_from_datamove_error(self):
        """Test enhancing existing DataMoveError."""
        original_error = DataMoveError("Original error", {"existing_key": "existing_value"})
        
        enhanced_error = create_enhanced_error(
            original_error,
            "test_operation",
            "test_stage",
            new_key="new_value"
        )
        
        assert enhanced_error is original_error  # Should be the same object
        assert enhanced_error.context["existing_key"] == "existing_value"
        assert enhanced_error.context["operation"] == "test_operation"
        assert enhanced_error.context["new_key"] == "new_value"


if __name__ == "__main__":
    pytest.main([__file__])