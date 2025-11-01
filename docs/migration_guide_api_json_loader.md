# Migration Guide: Adopting APIJSONStorageLoader

This guide helps you migrate from existing loaders (S3Loader, LocalLoader) to the new APIJSONStorageLoader, or integrate APIJSONStorageLoader into your existing workflows.

## Table of Contents

1. [Overview](#overview)
2. [Backward Compatibility](#backward-compatibility)
3. [Migration Scenarios](#migration-scenarios)
4. [Step-by-Step Migration](#step-by-step-migration)
5. [Configuration Changes](#configuration-changes)
6. [Common Issues and Solutions](#common-issues-and-solutions)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

## Overview

The APIJSONStorageLoader extends the existing data loading infrastructure to support:

- **API endpoints** with authentication
- **JSON data processing** with nested structure flattening
- **Column mapping** and data transformations
- **Full backward compatibility** with existing CSV loading

### Key Benefits

- **Unified Interface**: Single loader for CSV, JSON files, and API endpoints
- **Advanced Processing**: Nested JSON flattening, column mapping, data transformations
- **Seamless Integration**: Drop-in replacement for existing loaders
- **Enhanced Features**: Authentication, pagination, retry logic for APIs

## Backward Compatibility

✅ **Fully Compatible**: APIJSONStorageLoader maintains 100% backward compatibility with existing functionality.

### What Stays the Same

- **CSV Loading**: `load_csv()` method works identically
- **Interface Compliance**: Implements `StorageLoaderInterface`
- **Use Case Integration**: Works with existing `dataloadUseCase` and `DataMoveUseCase`
- **Error Handling**: Same error types and patterns
- **Database Integration**: Compatible with all existing repositories

### What's Enhanced

- **JSON Support**: New `load_json()` method with advanced processing
- **API Support**: Direct API endpoint loading with authentication
- **Data Processing**: Column mapping, transformations, nested structure handling

## Migration Scenarios

### Scenario 1: No Changes Required (Existing CSV Workflows)

If you're only using CSV files, **no migration is needed**. Your existing code will work unchanged.

```python
# Before (LocalLoader)
from dataload.infrastructure.storage.loaders import LocalLoader
loader = LocalLoader()
df = loader.load_csv("data.csv")

# After (APIJSONStorageLoader) - Same code works!
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
loader = APIJSONStorageLoader()
df = loader.load_csv("data.csv")  # Identical behavior
```

### Scenario 2: Adding JSON Support to Existing Workflows

Enhance existing workflows with JSON capabilities without breaking changes.

```python
# Existing use case - no changes needed
use_case = dataloadUseCase(
    repo=repository,
    embedding_service=embedding_service,
    storage_loader=APIJSONStorageLoader()  # Drop-in replacement
)

# Still works for CSV
await use_case.execute(
    s3_uri="data.csv",
    table_name="my_table",
    embed_columns_names=["description"],
    pk_columns=["id"]
)
```

### Scenario 3: Migrating from Basic JSON to Advanced Processing

Upgrade from basic JSON handling to advanced processing capabilities.

```python
# Before: Basic JSON with LocalLoader
loader = LocalLoader()
df = await loader.load_json("data.json")  # Basic flattening

# After: Advanced JSON processing
loader = APIJSONStorageLoader()
df = await loader.load_json("data.json", {
    'flatten_nested': True,
    'separator': '_',
    'column_name_mapping': {'user_id': 'id', 'full_name': 'name'},
    'update_request_body_mapping': {'display_name': "concat({first_name}, ' ', {last_name})"}
})
```

### Scenario 4: Adding API Data Sources

Extend existing data pipelines to include API sources.

```python
# Existing CSV pipeline
use_case = DataAPIJSONUseCase(
    repo=repository,
    embedding_service=embedding_service,
    storage_loader=APIJSONStorageLoader()
)

# Add API data source to same pipeline
await use_case.execute(
    source="https://api.example.com/users",  # New: API endpoint
    table_name="users",
    embed_columns_names=["bio"],
    column_name_mapping={'user_id': 'id'}
)
```

## Step-by-Step Migration

### Step 1: Update Imports

Replace existing loader imports with APIJSONStorageLoader:

```python
# Before
from dataload.infrastructure.storage.loaders import LocalLoader, S3Loader

# After
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
```

### Step 2: Update Loader Instantiation

Replace loader instances:

```python
# Before
local_loader = LocalLoader()
s3_loader = S3Loader()

# After
api_loader = APIJSONStorageLoader()
# Single loader handles all sources (local files, S3, APIs)
```

### Step 3: Update Use Case Configuration

Update use case instantiation:

```python
# Before
use_case = dataloadUseCase(
    repo=repository,
    embedding_service=embedding_service,
    storage_loader=LocalLoader()
)

# After
use_case = dataloadUseCase(
    repo=repository,
    embedding_service=embedding_service,
    storage_loader=APIJSONStorageLoader()
)
```

### Step 4: Enhance with New Features (Optional)

Add new capabilities as needed:

```python
# Enhanced loader with API authentication
api_loader = APIJSONStorageLoader(
    api_token="your-api-key",
    timeout=60,
    retry_attempts=3
)

# Enhanced use case for API/JSON data
api_use_case = DataAPIJSONUseCase(
    repo=repository,
    embedding_service=embedding_service,
    storage_loader=api_loader
)
```

### Step 5: Test and Validate

Run your existing tests to ensure compatibility:

```python
# Your existing tests should pass unchanged
def test_existing_csv_workflow():
    loader = APIJSONStorageLoader()  # Drop-in replacement
    df = loader.load_csv("test_data.csv")
    assert len(df) > 0
    assert 'id' in df.columns
```

## Configuration Changes

### Environment Variables

No changes required for existing environment variables. New optional variables for API features:

```bash
# Existing variables (unchanged)
DATABASE_URL=postgresql://user:pass@localhost/db
EMBEDDING_SERVICE_API_KEY=your-embedding-key

# New optional variables for API features
DEFAULT_API_TIMEOUT=30
DEFAULT_RETRY_ATTEMPTS=3
DEFAULT_API_BASE_URL=https://api.example.com
```

### Configuration Files

Existing configuration files work unchanged. New optional configuration for enhanced features:

```python
# config.py - existing configuration unchanged
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mydb"
}

# New optional API configuration
API_CONFIG = {
    "timeout": 30,
    "retry_attempts": 3,
    "verify_ssl": True
}
```

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem**: `ImportError` when importing APIJSONStorageLoader

**Solution**: Ensure you have the latest version installed:

```bash
pip install --upgrade vector-dataloader
```

### Issue 2: Async/Await Compatibility

**Problem**: Existing synchronous code with new async `load_json` method

**Solution**: Use async context or run in executor:

```python
# Option 1: Convert to async
async def load_data():
    loader = APIJSONStorageLoader()
    df = await loader.load_json(data)
    return df

# Option 2: Run in executor for sync code
import asyncio
loader = APIJSONStorageLoader()
df = asyncio.run(loader.load_json(data))
```

### Issue 3: Column Name Changes

**Problem**: JSON flattening creates different column names than expected

**Solution**: Use column mapping to maintain expected names:

```python
config = {
    'column_name_mapping': {
        'nested_field_name': 'expected_name',
        'api_field': 'db_column'
    }
}
df = await loader.load_json(data, config)
```

### Issue 4: Performance Differences

**Problem**: Processing seems slower with nested JSON

**Solution**: Optimize configuration for your use case:

```python
# For simple data, disable unnecessary processing
config = {
    'flatten_nested': False,  # If data is already flat
    'max_depth': 2,          # Limit nesting depth
    'handle_arrays': 'join'   # Join arrays instead of expanding
}
```

### Issue 5: Authentication Issues

**Problem**: API authentication failures

**Solution**: Configure authentication properly:

```python
# API key authentication
loader = APIJSONStorageLoader(api_token="your-key")

# JWT authentication
loader = APIJSONStorageLoader(jwt_token="your-jwt")

# Custom headers
loader = APIJSONStorageLoader(default_headers={
    "Authorization": "Bearer your-token",
    "X-API-Key": "your-key"
})
```

## Best Practices

### 1. Gradual Migration

Migrate incrementally to minimize risk:

```python
# Phase 1: Replace loader, keep existing functionality
loader = APIJSONStorageLoader()
# Test all existing CSV workflows

# Phase 2: Add JSON file support
# Test JSON file loading

# Phase 3: Add API endpoints
# Test API data loading

# Phase 4: Add advanced features
# Test column mapping, transformations
```

### 2. Configuration Management

Centralize configuration for consistency:

```python
# config.py
DEFAULT_JSON_CONFIG = {
    'flatten_nested': True,
    'separator': '_',
    'handle_arrays': 'expand',
    'fail_on_error': False
}

# usage.py
loader = APIJSONStorageLoader()
df = await loader.load_json(data, DEFAULT_JSON_CONFIG)
```

### 3. Error Handling

Maintain robust error handling:

```python
from dataload.domain.entities import DataMoveError
from dataload.domain.api_entities import APIError, JSONParsingError

try:
    df = await loader.load_json(source, config)
except APIError as e:
    logger.error(f"API error: {e}")
    # Handle API-specific errors
except JSONParsingError as e:
    logger.error(f"JSON parsing error: {e}")
    # Handle JSON-specific errors
except DataMoveError as e:
    logger.error(f"Data move error: {e}")
    # Handle general data errors
```

### 4. Testing Strategy

Maintain comprehensive testing:

```python
def test_backward_compatibility():
    """Test that existing functionality still works."""
    loader = APIJSONStorageLoader()

    # Test CSV loading (should be identical)
    df_csv = loader.load_csv("test.csv")
    assert len(df_csv) > 0

    # Test with existing use cases
    use_case = dataloadUseCase(repo, embedding_service, loader)
    # Should work without changes

def test_new_features():
    """Test new JSON/API features."""
    loader = APIJSONStorageLoader()

    # Test JSON loading
    df_json = await loader.load_json(json_data)
    assert len(df_json) > 0

    # Test with configuration
    df_configured = await loader.load_json(json_data, config)
    assert 'mapped_column' in df_configured.columns
```

### 5. Performance Optimization

Optimize for your specific use cases:

```python
# For large datasets
config = {
    'max_depth': 3,           # Limit nesting depth
    'handle_arrays': 'join',  # Avoid array expansion
    'preserve_original_data': False  # Save memory
}

# For high-frequency API calls
loader = APIJSONStorageLoader(
    timeout=10,        # Shorter timeout
    retry_attempts=2,  # Fewer retries
    verify_ssl=False   # If in trusted environment
)
```

## Examples

### Example 1: Simple Migration

```python
# Before: Using LocalLoader
from dataload.infrastructure.storage.loaders import LocalLoader
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase

loader = LocalLoader()
use_case = dataloadUseCase(repo, embedding_service, loader)

await use_case.execute(
    s3_uri="employees.csv",
    table_name="employees",
    embed_columns_names=["description"],
    pk_columns=["id"]
)

# After: Using APIJSONStorageLoader (identical functionality)
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.application.use_cases.data_loader_use_case import dataloadUseCase

loader = APIJSONStorageLoader()  # Drop-in replacement
use_case = dataloadUseCase(repo, embedding_service, loader)

await use_case.execute(
    s3_uri="employees.csv",  # Same CSV file
    table_name="employees",
    embed_columns_names=["description"],
    pk_columns=["id"]
)
```

### Example 2: Adding JSON Support

```python
# Enhanced workflow with JSON support
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader
from dataload.application.use_cases.data_api_json_use_case import DataAPIJSONUseCase

loader = APIJSONStorageLoader()
use_case = DataAPIJSONUseCase(repo, embedding_service, loader)

# Load from JSON file (new capability)
await use_case.execute(
    source="employees.json",
    table_name="employees",
    embed_columns_names=["description"],
    pk_columns=["id"]
)

# Load from API endpoint (new capability)
await use_case.execute(
    source="https://api.company.com/employees",
    table_name="employees",
    embed_columns_names=["description"],
    pk_columns=["id"]
)
```

### Example 3: Advanced Configuration

```python
# Advanced configuration with all features
loader = APIJSONStorageLoader(
    api_token="your-api-key",
    timeout=60,
    retry_attempts=3
)

use_case = DataAPIJSONUseCase(repo, embedding_service, loader)

await use_case.execute(
    source="https://api.company.com/employees",
    table_name="employees",
    embed_columns_names=["full_name", "bio"],  # Using mapped names
    pk_columns=["employee_id"],
    column_name_mapping={
        'id': 'employee_id',
        'name': 'full_name',
        'profile_bio': 'bio'
    },
    update_request_body_mapping={
        'display_name': "concat({first_name}, ' ', {last_name})",
        'years_experience': "round(({current_year} - {start_year}))"
    }
)
```

### Example 4: Gradual Feature Adoption

```python
# Phase 1: Basic migration (no functional changes)
class DataPipeline:
    def __init__(self):
        # Replace loader only
        self.loader = APIJSONStorageLoader()  # Was: LocalLoader()
        self.use_case = dataloadUseCase(repo, embedding_service, self.loader)

    async def load_csv_data(self, file_path):
        # Existing method - no changes
        return await self.use_case.execute(
            s3_uri=file_path,
            table_name="data",
            embed_columns_names=["content"],
            pk_columns=["id"]
        )

# Phase 2: Add JSON support
class EnhancedDataPipeline(DataPipeline):
    def __init__(self):
        super().__init__()
        # Add JSON-specific use case
        self.json_use_case = DataAPIJSONUseCase(repo, embedding_service, self.loader)

    async def load_json_data(self, source, config=None):
        # New method for JSON data
        return await self.json_use_case.execute(
            source=source,
            table_name="json_data",
            embed_columns_names=["content"],
            pk_columns=["id"],
            **config or {}
        )

# Phase 3: Full feature utilization
class AdvancedDataPipeline(EnhancedDataPipeline):
    async def load_api_data(self, endpoint, mapping_config):
        # Advanced API loading with full configuration
        return await self.json_use_case.execute(
            source=endpoint,
            table_name="api_data",
            embed_columns_names=mapping_config.get('embed_columns', []),
            pk_columns=mapping_config.get('pk_columns', ['id']),
            column_name_mapping=mapping_config.get('column_mapping', {}),
            update_request_body_mapping=mapping_config.get('transformations', {})
        )
```

## Conclusion

The APIJSONStorageLoader provides a seamless migration path with full backward compatibility. You can:

1. **Start immediately** with zero code changes for existing CSV workflows
2. **Add features gradually** as needed (JSON files, API endpoints, advanced processing)
3. **Maintain existing patterns** and error handling
4. **Enhance capabilities** with new features like column mapping and data transformations

The migration is designed to be **risk-free** and **incremental**, allowing you to adopt new features at your own pace while maintaining all existing functionality.

For questions or issues during migration, refer to the troubleshooting guide or create an issue in the project repository.
