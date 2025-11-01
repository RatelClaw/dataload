# DataMove Use Case - Quick Start Guide

Get up and running with DataMove in minutes! This guide covers the essential steps to start migrating data from CSV files to PostgreSQL databases.

## What is DataMove?

DataMove is a production-grade data migration tool that moves data from CSV files (local or S3) to PostgreSQL databases. Unlike other tools in the vector-dataloader library, DataMove focuses purely on data migration without embedding generation.

### Key Benefits

✅ **Simple Setup** - Works with existing PostgreSQL databases  
✅ **Flexible Validation** - Handles schema changes gracefully  
✅ **Production Ready** - Comprehensive error handling and rollback  
✅ **S3 Integration** - Seamless cloud file support  
✅ **Performance Optimized** - Batch processing and memory management  

## Prerequisites

### 1. Install vector-dataloader

```bash
pip install vector-dataloader
```

### 2. PostgreSQL Database

Ensure you have a PostgreSQL database running and accessible.

### 3. Environment Configuration

Create a `.env` file in your project root:

```env
LOCAL_POSTGRES_HOST=localhost
LOCAL_POSTGRES_PORT=5432
LOCAL_POSTGRES_DB=your_database
LOCAL_POSTGRES_USER=your_username
LOCAL_POSTGRES_PASSWORD=your_password
```

## 5-Minute Quick Start

### Step 1: Prepare Your CSV Data

Create a sample CSV file (`employees.csv`):

```csv
id,name,email,department,salary,active
1,Alice Johnson,alice@company.com,Engineering,95000,true
2,Bob Smith,bob@company.com,Sales,75000,true
3,Charlie Brown,charlie@company.com,Marketing,68000,false
```

### Step 2: Basic Data Migration

```python
import asyncio
from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
from dataload.infrastructure.db.db_connection import DBConnection

async def quick_start_example():
    # 1. Set up database connection
    db_connection = DBConnection()
    await db_connection.initialize()
    
    # 2. Create repository and use case
    repository = PostgresDataMoveRepository(db_connection)
    use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
    
    # 3. Move data to new table
    result = await use_case.execute(
        csv_path="employees.csv",
        table_name="employees",
        primary_key_columns=["id"]
    )
    
    # 4. Check results
    if result.success:
        print(f"✅ Success! Processed {result.rows_processed} rows")
        print(f"⏱️ Completed in {result.execution_time:.2f} seconds")
    else:
        print(f"❌ Failed: {result.errors}")
    
    # 5. Close connection
    await db_connection.close()

# Run the example
asyncio.run(quick_start_example())
```

### Step 3: Verify Your Data

Connect to your PostgreSQL database and verify:

```sql
SELECT * FROM employees;
```

You should see your CSV data in the table!

## Common Use Cases

### Use Case 1: Create New Table

When the target table doesn't exist, DataMove creates it automatically:

```python
result = await use_case.execute(
    csv_path="new_data.csv",
    table_name="new_table",
    primary_key_columns=["id"]  # Specify primary key
)
```

### Use Case 2: Update Existing Table (Strict)

For exact schema matching with existing tables:

```python
result = await use_case.execute(
    csv_path="updated_data.csv",
    table_name="existing_table",
    move_type="existing_schema"  # Strict validation
)
```

### Use Case 3: Flexible Schema Updates

Allow column additions/removals:

```python
result = await use_case.execute(
    csv_path="evolved_data.csv",
    table_name="existing_table",
    move_type="new_schema"  # Flexible validation
)
```

### Use Case 4: Preview Changes (Dry Run)

Test without making changes:

```python
result = await use_case.execute(
    csv_path="test_data.csv",
    table_name="target_table",
    dry_run=True  # No actual changes
)

print(f"Would process {result.rows_processed} rows")
```

### Use Case 5: S3 Integration

Load from cloud storage:

```python
result = await use_case.execute(
    csv_path="s3://your-bucket/data.csv",  # S3 URI
    table_name="cloud_data"
)
```

## Error Handling

DataMove provides detailed error information:

```python
try:
    result = await use_case.execute(
        csv_path="data.csv",
        table_name="target_table"
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Context: {e.context}")
except DatabaseOperationError as e:
    print(f"Database error: {e}")
except DataMoveError as e:
    print(f"DataMove error: {e}")
```

## Configuration Options

### Batch Size Tuning

Adjust for performance:

```python
result = await use_case.execute(
    csv_path="large_file.csv",
    table_name="target_table",
    batch_size=2000  # Larger batches for big files
)
```

### Custom Storage Loaders

Explicit loader selection:

```python
from dataload.infrastructure.storage.loaders import LocalLoader, S3Loader

# For local files only
local_loader = LocalLoader()
use_case = DataMoveUseCase(repository=repository, storage_loader=local_loader)

# For S3 files only
s3_loader = S3Loader()
use_case = DataMoveUseCase(repository=repository, storage_loader=s3_loader)
```

## Validation Modes Explained

### `move_type=None` (Default for New Tables)

- **When to use**: Target table doesn't exist
- **Behavior**: Creates new table from CSV schema
- **Validation**: Basic data type validation only

```python
# Creates new table automatically
result = await use_case.execute(
    csv_path="data.csv",
    table_name="new_table"
)
```

### `move_type="existing_schema"` (Strict Mode)

- **When to use**: Exact schema match required
- **Behavior**: Validates CSV exactly matches table schema
- **Validation**: Column names, types, and nullability must match exactly

```python
# Requires exact match
result = await use_case.execute(
    csv_path="data.csv",
    table_name="existing_table",
    move_type="existing_schema"
)
```

### `move_type="new_schema"` (Flexible Mode)

- **When to use**: Schema evolution allowed
- **Behavior**: Allows column additions/removals
- **Validation**: Prevents case-sensitivity conflicts

```python
# Allows schema changes
result = await use_case.execute(
    csv_path="evolved_data.csv",
    table_name="existing_table",
    move_type="new_schema"
)
```

## Best Practices

### 1. Always Use Dry Run First

```python
# Preview changes
preview = await use_case.get_operation_preview(
    csv_path="data.csv",
    table_name="target_table"
)

if preview.validation_passed:
    # Execute actual operation
    result = await use_case.execute(...)
```

### 2. Handle Errors Gracefully

```python
async def safe_data_move(csv_path, table_name):
    try:
        result = await use_case.execute(csv_path=csv_path, table_name=table_name)
        return {"success": True, "rows": result.rows_processed}
    except CaseSensitivityError:
        return {"error": "Fix column name case conflicts"}
    except SchemaConflictError:
        return {"error": "Try move_type='new_schema' for flexibility"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}
```

### 3. Monitor Performance

```python
result = await use_case.execute(...)

print(f"Processed {result.rows_processed} rows")
print(f"Time: {result.execution_time:.2f}s")
print(f"Throughput: {result.rows_processed/result.execution_time:.0f} rows/sec")
```

### 4. Use Appropriate Batch Sizes

```python
# File size based batch sizing
file_size_mb = os.path.getsize("data.csv") / 1024 / 1024

if file_size_mb < 10:
    batch_size = 500
elif file_size_mb < 100:
    batch_size = 1000
else:
    batch_size = 2000

result = await use_case.execute(..., batch_size=batch_size)
```

## Troubleshooting Quick Fixes

### Problem: "Table already exists" error

**Solution**: Use appropriate `move_type`:

```python
# For existing tables, specify move_type
result = await use_case.execute(
    csv_path="data.csv",
    table_name="existing_table",
    move_type="existing_schema"  # or "new_schema"
)
```

### Problem: "Column name case conflicts"

**Solution**: Standardize CSV column names:

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.columns = df.columns.str.lower()  # Convert to lowercase
df.to_csv("fixed_data.csv", index=False)
```

### Problem: "Database connection failed"

**Solution**: Check your `.env` file and database status:

```bash
# Test database connection
psql -h localhost -U your_user -d your_db -c "SELECT 1;"
```

### Problem: "File not found"

**Solution**: Use absolute paths:

```python
import os
csv_path = os.path.abspath("data.csv")
```

### Problem: "S3 access denied"

**Solution**: Configure AWS credentials:

```bash
aws configure
# Enter your Access Key ID and Secret Access Key
```

## Next Steps

### 1. Explore Advanced Features

- **Vector Data Handling**: For ML/AI use cases with embedding columns
- **JSON Metadata**: Support for complex nested data structures
- **Performance Optimization**: Advanced batch processing and memory management

### 2. Production Deployment

- **Connection Pooling**: For high-throughput scenarios
- **Monitoring**: Integration with logging and metrics systems
- **Error Recovery**: Automated retry and recovery strategies

### 3. Integration Patterns

- **ETL Pipelines**: Integrate with Apache Airflow or similar tools
- **Event-Driven**: Trigger on file uploads or schedule-based processing
- **API Integration**: Wrap in REST API for web application integration

## Getting Help

### Documentation

- **API Documentation**: Complete method and parameter reference
- **Troubleshooting Guide**: Common issues and solutions
- **Example Scripts**: Comprehensive usage examples

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share use cases

### Support

For production support and consulting, contact the maintainers through the project repository.

---

**Congratulations!** 🎉 You're now ready to use DataMove for production-grade data migration. Start with the basic examples above and gradually explore more advanced features as your needs grow.