"""
Comprehensive error handling utilities for DataMove operations.

This module provides centralized error handling patterns, error collection,
context management, and rollback utilities for robust data movement operations.
"""

import asyncio
import traceback
from typing import List, Dict, Any, Optional, Callable, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

from dataload.domain.entities import (
    DataMoveError,
    ValidationError,
    DatabaseOperationError,
    SchemaConflictError,
    CaseSensitivityError,
    DataTypeError
)
from dataload.config import logger


class ErrorSeverity(Enum):
    """Error severity levels for categorizing errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better organization."""
    VALIDATION = "validation"
    DATABASE = "database"
    FILE_IO = "file_io"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    DATA_INTEGRITY = "data_integrity"
    PERFORMANCE = "performance"
    UNEXPECTED = "unexpected"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    operation: str
    stage: str
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def add_context(self, key: str, value: Any) -> None:
        """Add additional context information."""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary."""
        return {
            "operation": self.operation,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "stack_trace": self.stack_trace
        }


@dataclass
class CollectedError:
    """Represents a collected error with full context."""
    error: Exception
    context: ErrorContext
    severity: ErrorSeverity
    category: ErrorCategory
    recoverable: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collected error to dictionary."""
        return {
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
            "context": self.context.to_dict()
        }


class ErrorCollector:
    """Collects and manages errors during operations."""
    
    def __init__(self):
        self.errors: List[CollectedError] = []
        self.warnings: List[str] = []
        self.operation_context: Optional[ErrorContext] = None
    
    def set_operation_context(self, operation: str, stage: str, **kwargs) -> None:
        """Set the current operation context."""
        self.operation_context = ErrorContext(
            operation=operation,
            stage=stage,
            parameters=kwargs
        )
    
    def add_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNEXPECTED,
        recoverable: bool = False,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an error to the collection."""
        context = self.operation_context or ErrorContext("unknown", "unknown")
        
        if additional_context:
            for key, value in additional_context.items():
                context.add_context(key, value)
        
        # Add stack trace for debugging
        context.stack_trace = traceback.format_exc()
        
        collected_error = CollectedError(
            error=error,
            context=context,
            severity=severity,
            category=category,
            recoverable=recoverable
        )
        
        self.errors.append(collected_error)
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error in {context.operation}: {error}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error in {context.operation}: {error}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error in {context.operation}: {error}")
        else:
            logger.info(f"Low severity error in {context.operation}: {error}")
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)
    
    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return any(error.severity == ErrorSeverity.CRITICAL for error in self.errors)
    
    def has_high_severity_errors(self) -> bool:
        """Check if there are any high severity errors."""
        return any(error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH] for error in self.errors)
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[CollectedError]:
        """Get errors filtered by category."""
        return [error for error in self.errors if error.category == category]
    
    def get_recoverable_errors(self) -> List[CollectedError]:
        """Get errors that might be recoverable."""
        return [error for error in self.errors if error.recoverable]
    
    def clear(self) -> None:
        """Clear all collected errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
    
    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary of collected errors."""
        return {
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "critical_errors": len([e for e in self.errors if e.severity == ErrorSeverity.CRITICAL]),
            "high_severity_errors": len([e for e in self.errors if e.severity == ErrorSeverity.HIGH]),
            "recoverable_errors": len(self.get_recoverable_errors()),
            "errors_by_category": {
                category.value: len(self.get_errors_by_category(category))
                for category in ErrorCategory
            }
        }


class ErrorHandler:
    """Comprehensive error handling with retry logic and rollback capabilities."""
    
    def __init__(self, collector: Optional[ErrorCollector] = None):
        self.collector = collector or ErrorCollector()
    
    @asynccontextmanager
    async def handle_operation(
        self,
        operation_name: str,
        stage: str,
        rollback_func: Optional[Callable] = None,
        **context_kwargs
    ):
        """
        Context manager for handling operations with comprehensive error management.
        
        Args:
            operation_name: Name of the operation being performed
            stage: Current stage of the operation
            rollback_func: Optional function to call for rollback on failure
            **context_kwargs: Additional context parameters
        """
        self.collector.set_operation_context(operation_name, stage, **context_kwargs)
        
        try:
            yield self.collector
            
        except (ValidationError, DatabaseOperationError, DataMoveError) as known_error:
            # Handle known errors with appropriate categorization
            category = self._categorize_known_error(known_error)
            severity = self._determine_severity(known_error)
            
            self.collector.add_error(
                known_error,
                severity=severity,
                category=category,
                recoverable=self._is_recoverable_error(known_error)
            )
            
            # Attempt rollback if provided
            if rollback_func:
                await self._safe_rollback(rollback_func, operation_name)
            
            raise known_error
            
        except Exception as unexpected_error:
            # Handle unexpected errors
            self.collector.add_error(
                unexpected_error,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.UNEXPECTED,
                recoverable=False
            )
            
            # Attempt rollback if provided
            if rollback_func:
                await self._safe_rollback(rollback_func, operation_name)
            
            # Wrap unexpected errors in DataMoveError
            wrapped_error = DataMoveError(
                f"Unexpected error in {operation_name}: {unexpected_error}",
                context=context_kwargs
            )
            raise wrapped_error from unexpected_error
    
    async def execute_with_retry(
        self,
        operation: Callable,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_multiplier: float = 1.5,
        retryable_exceptions: tuple = (ConnectionError, TimeoutError)
    ) -> Any:
        """
        Execute an operation with retry logic and error collection.
        
        Args:
            operation: Async callable to execute
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries
            backoff_multiplier: Multiplier for exponential backoff
            retryable_exceptions: Tuple of exception types that should trigger retry
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: The last exception if all retries fail
        """
        last_error = None
        current_delay = retry_delay
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                result = await operation()
                
                if attempt > 0:
                    logger.info(f"Operation succeeded on attempt {attempt + 1}")
                
                return result
                
            except retryable_exceptions as e:
                last_error = e
                
                if attempt < max_retries:
                    self.collector.add_warning(
                        f"Retryable error on attempt {attempt + 1}: {e}. "
                        f"Retrying in {current_delay:.1f} seconds..."
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_multiplier
                else:
                    self.collector.add_error(
                        e,
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.NETWORK if isinstance(e, ConnectionError) else ErrorCategory.PERFORMANCE,
                        additional_context={"retry_attempts": attempt + 1}
                    )
                    
            except Exception as e:
                # Non-retryable error
                self.collector.add_error(
                    e,
                    severity=ErrorSeverity.HIGH,
                    category=self._categorize_exception(e),
                    additional_context={"retry_attempts": attempt + 1}
                )
                raise e
        
        # All retries exhausted
        raise last_error
    
    async def _safe_rollback(self, rollback_func: Callable, operation_name: str) -> None:
        """Safely execute rollback function with error handling."""
        try:
            logger.warning(f"Attempting rollback for operation: {operation_name}")
            
            if asyncio.iscoroutinefunction(rollback_func):
                await rollback_func()
            else:
                rollback_func()
                
            logger.info(f"Rollback completed successfully for operation: {operation_name}")
            
        except Exception as rollback_error:
            self.collector.add_error(
                rollback_error,
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.DATABASE,
                additional_context={"rollback_operation": operation_name}
            )
            
            logger.error(f"Rollback failed for operation {operation_name}: {rollback_error}")
    
    def _categorize_known_error(self, error: Exception) -> ErrorCategory:
        """Categorize known error types."""
        if isinstance(error, ValidationError):
            return ErrorCategory.VALIDATION
        elif isinstance(error, DatabaseOperationError):
            return ErrorCategory.DATABASE
        elif isinstance(error, (SchemaConflictError, CaseSensitivityError)):
            return ErrorCategory.DATA_INTEGRITY
        elif isinstance(error, DataTypeError):
            return ErrorCategory.VALIDATION
        else:
            return ErrorCategory.UNEXPECTED
    
    def _categorize_exception(self, error: Exception) -> ErrorCategory:
        """Categorize general exception types."""
        if isinstance(error, (FileNotFoundError, PermissionError)):
            return ErrorCategory.FILE_IO
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, MemoryError):
            return ErrorCategory.PERFORMANCE
        else:
            return ErrorCategory.UNEXPECTED
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type."""
        if isinstance(error, (SchemaConflictError, CaseSensitivityError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, ValidationError):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, DatabaseOperationError):
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error might be recoverable."""
        # Connection errors and timeouts are often recoverable
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        
        # Some validation errors might be recoverable with data fixes
        if isinstance(error, ValidationError):
            return True
        
        # Schema conflicts might be recoverable with schema changes
        if isinstance(error, SchemaConflictError):
            return True
        
        return False


# Convenience functions for common error handling patterns

async def with_error_handling(
    operation: Callable,
    operation_name: str,
    stage: str,
    rollback_func: Optional[Callable] = None,
    **context_kwargs
) -> Any:
    """
    Execute an operation with comprehensive error handling.
    
    Args:
        operation: Async callable to execute
        operation_name: Name of the operation
        stage: Current stage of the operation
        rollback_func: Optional rollback function
        **context_kwargs: Additional context
        
    Returns:
        Result of the operation
    """
    handler = ErrorHandler()
    
    async with handler.handle_operation(operation_name, stage, rollback_func, **context_kwargs):
        return await operation()


def create_enhanced_error(
    base_error: Exception,
    operation: str,
    stage: str,
    **additional_context
) -> DataMoveError:
    """
    Create an enhanced DataMoveError with comprehensive context.
    
    Args:
        base_error: Original exception
        operation: Operation name
        stage: Operation stage
        **additional_context: Additional context information
        
    Returns:
        Enhanced DataMoveError
    """
    context = {
        "operation": operation,
        "stage": stage,
        "original_error_type": type(base_error).__name__,
        "original_error_message": str(base_error),
        **additional_context
    }
    
    if isinstance(base_error, DataMoveError):
        # Enhance existing DataMoveError
        if hasattr(base_error, 'context') and base_error.context:
            context.update(base_error.context)
        base_error.context = context
        return base_error
    else:
        # Create new DataMoveError
        return DataMoveError(
            f"Error in {operation} at stage {stage}: {base_error}",
            context=context
        )