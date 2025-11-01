# DataMove Use Case - Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using the DataMove use case for migrating data from CSV files to PostgreSQL databases.

## Table of Contents

1. [Setup and Configuration Issues](#setup-and-configuration-issues)
2. [File Access and Loading Issues](#file-access-and-loading-issues)
3. [Database Connection Issues](#database-connection-issues)
4. [Validation and Schema Issues](#validation-and-schema-issues)
5. [Performance and Memory Issues](#performance-and-memory-issues)
6. [S3 Integration Issues](#s3-integration-issues)
7. [Data Type and Conversion Issues](#data-type-and-conversion-issues)
8. [Error Reference](#error-reference)

## Setup and Configuration Issues

### Issue: "Module not found" or Import Errors

**Symptoms:**
```
ImportError: No module named 'dataload'
ModuleNotFoundError: No module named 'dataload.application.use_cases.data_move_use_case'
```

**Solutions:**
1. **Install the package:**
   ```bash
   pip install vector-dataloader
   ```

2. **Verify installation:**
   ```bash
   pip show vector-dataloader
   ```

3. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   ```

4. **For development installations:**
   ```bash
   pip install -e .
   ```

### Issue: Database Configuration Missing

**Symptoms:**
```
DataMoveError: Database connection failed
KeyError: 'LOCAL_POSTGRES_HOST'
```

**Solutions:**
1. **Create .env file in project root:**
   ```env
   LOCAL_POSTGRES_HOST=localhost
   LOCAL_POSTGRES_PORT=5432
   LOCAL_POSTGRES_DB=your_database
   LOCAL_POSTGRES_USER=your_username
   LOCAL_POSTGRES_PASSWORD=your_password
   ```

2. **Verify environment variables:**
   ```python
   import os
   print(os.getenv('LOCAL_POSTGRES_HOST'))
   ```

3. **Test database connection:**
   ```python
   from dataload.infrastructure.db.db_connection import DBConnection
   
   async def test_connection():
       db = DBConnection()
       await db.initialize()
       print("✅ Database connection successful")
       await db.close()
   ```

## File Access and Loading Issues

### Issue: CSV File Not Found

**Symptoms:**
```
DataMoveError: CSV file not found: /path/to/file.csv
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**
1. **Check file path:**
   ```python
   import os
   print(os.path.exists('your_file.csv'))
   print(os.path.abspath('your_file.csv'))
   ```

2. **Use absolute paths:**
   ```python
   csv_path = os.path.abspath('data/employees.csv')
   ```

3. **Verify file permissions:**
   ```bash
   ls -la your_file.csv
   ```

### Issue: CSV Parsing Errors

**Symptoms:**
```
pandas.errors.ParserError: Error tokenizing data
UnicodeDecodeError: 'utf-8' codec can't decode
```

**Solutions:**
1. **Check CSV format:**
   ```python
   import pandas as pd
   
   # Try different encodings
   df = pd.read_csv('file.csv', encoding='utf-8')
   # or
   df = pd.read_csv('file.csv', encoding='latin-1')
   ```

2. **Handle malformed CSV:**
   ```python
   df = pd.read_csv('file.csv', error_bad_lines=False, warn_bad_lines=True)
   ```

3. **Check for BOM (Byte Order Mark):**
   ```python
   df = pd.read_csv('file.csv', encoding='utf-8-sig')
   ```

### Issue: Empty CSV File

**Symptoms:**
```
Warning: CSV file is empty - no data to move
```

**Solutions:**
1. **Verify file content:**
   ```bash
   head -5 your_file.csv
   wc -l your_file.csv
   ```

2. **Check for headers only:**
   ```python
   df = pd.read_csv('file.csv')
   print(f"Shape: {df.shape}")
   print(f"Columns: {df.columns.tolist()}")
   ```

## Database Connection Issues

### Issue: Connection Timeout or Refused

**Symptoms:**
```
DatabaseOperationError: Database connection failed
psycopg2.OperationalError: could not connect to server
```

**Solutions:**
1. **Check PostgreSQL service:**
   ```bash
   # Linux/Mac
   sudo systemctl status postgresql
   
   # Windows
   net start postgresql-x64-13
   ```

2. **Verify connection parameters:**
   ```python
   import asyncpg
   
   async def test_direct_connection():
       conn = await asyncpg.connect(
           host='localhost',
           port=5432,
           database='your_db',
           user='your_user',
           password='your_password'
       )
       await conn.close()
   ```

3. **Check firewall and network:**
   ```bash
   telnet localhost 5432
   ```

### Issue: Authentication Failed

**Symptoms:**
```
psycopg2.OperationalError: FATAL: password authentication failed
psycopg2.OperationalError: FATAL: role "user" does not exist
```

**Solutions:**
1. **Verify credentials:**
   ```sql
   -- Connect as postgres superuser
   \du  -- List users
   ```

2. **Create user if needed:**
   ```sql
   CREATE USER your_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE your_db TO your_user;
   ```

3. **Check pg_hba.conf:**
   ```
   # Add line for local connections
   local   all             your_user                               md5
   ```

### Issue: Database Does Not Exist

**Symptoms:**
```
psycopg2.OperationalError: FATAL: database "your_db" does not exist
```

**Solutions:**
1. **Create database:**
   ```sql
   CREATE DATABASE your_db;
   ```

2. **List existing databases:**
   ```sql
   \l  -- In psql
   ```

## Validation and Schema Issues

### Issue: Schema Validation Failures

**Symptoms:**
```
ValidationError: Validation failed with 3 error(s)
SchemaConflictError: Column 'name' type mismatch: expected text, got object
```

**Solutions:**
1. **Use dry-run to preview issues:**
   ```python
   preview = await use_case.get_operation_preview(
       csv_path='file.csv',
       table_name='target_table'
   )
   print(preview.errors)
   print(preview.recommendations)
   ```

2. **Check column types:**
   ```python
   df = pd.read_csv('file.csv')
   print(df.dtypes)
   print(df.info())
   ```

3. **Use appropriate move_type:**
   ```python
   # For exact schema match
   move_type='existing_schema'
   
   # For flexible schema evolution
   move_type='new_schema'
   ```

### Issue: Case Sensitivity Conflicts

**Symptoms:**
```
CaseSensitivityError: Case sensitivity conflict detected
DB Column: 'name' vs CSV Column: 'Name'
```

**Solutions:**
1. **Standardize column names in CSV:**
   ```python
   df = pd.read_csv('file.csv')
   df.columns = df.columns.str.lower()  # Convert to lowercase
   df.to_csv('fixed_file.csv', index=False)
   ```

2. **Check existing table schema:**
   ```sql
   SELECT column_name, data_type 
   FROM information_schema.columns 
   WHERE table_name = 'your_table';
   ```

3. **Use column mapping (if implemented):**
   ```python
   # Future feature - column name mapping
   column_mapping = {'Name': 'name', 'EMAIL': 'email'}
   ```

### Issue: Missing Required Columns

**Symptoms:**
```
ValidationError: Missing required columns: ['id', 'email']
```

**Solutions:**
1. **Add missing columns to CSV:**
   ```python
   df = pd.read_csv('file.csv')
   df['id'] = range(1, len(df) + 1)  # Add auto-incrementing ID
   df['email'] = df['name'].str.lower() + '@company.com'  # Generate emails
   ```

2. **Use new_schema mode for flexibility:**
   ```python
   result = await use_case.execute(
       csv_path='file.csv',
       table_name='target_table',
       move_type='new_schema'  # Allows missing columns
   )
   ```

### Issue: Extra Columns in CSV

**Symptoms:**
```
ValidationError: Extra columns found in CSV: ['extra_col1', 'extra_col2']
```

**Solutions:**
1. **Remove extra columns:**
   ```python
   df = pd.read_csv('file.csv')
   required_columns = ['id', 'name', 'email']  # Define required columns
   df = df[required_columns]
   df.to_csv('cleaned_file.csv', index=False)
   ```

2. **Use new_schema mode:**
   ```python
   # new_schema mode allows extra columns
   move_type='new_schema'
   ```

## Performance and Memory Issues

### Issue: Out of Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate array
DataMoveError: Memory usage validation failed
```

**Solutions:**
1. **Use smaller batch sizes:**
   ```python
   result = await use_case.execute(
       csv_path='large_file.csv',
       table_name='target_table',
       batch_size=500  # Reduce from default 1000
   )
   ```

2. **Process file in chunks:**
   ```python
   # For very large files, split into smaller files first
   chunk_size = 10000
   for i, chunk in enumerate(pd.read_csv('large_file.csv', chunksize=chunk_size)):
       chunk.to_csv(f'chunk_{i}.csv', index=False)
   ```

3. **Monitor memory usage:**
   ```python
   import psutil
   
   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

### Issue: Slow Performance

**Symptoms:**
- Operations taking longer than expected
- High CPU or memory usage

**Solutions:**
1. **Optimize batch size:**
   ```python
   # Test different batch sizes
   batch_sizes = [500, 1000, 2000, 5000]
   for batch_size in batch_sizes:
       start_time = time.time()
       # Run operation
       execution_time = time.time() - start_time
       print(f"Batch size {batch_size}: {execution_time:.2f}s")
   ```

2. **Use PostgreSQL COPY for simple data:**
   ```python
   # DataMove automatically uses COPY when possible
   # Ensure data types are simple (no JSON/arrays for best performance)
   ```

3. **Monitor database performance:**
   ```sql
   -- Check active connections
   SELECT * FROM pg_stat_activity WHERE state = 'active';
   
   -- Check table locks
   SELECT * FROM pg_locks WHERE granted = false;
   ```

## S3 Integration Issues

### Issue: AWS Credentials Not Found

**Symptoms:**
```
DataMoveError: S3 operation failed
NoCredentialsError: Unable to locate credentials
```

**Solutions:**
1. **Configure AWS credentials:**
   ```bash
   aws configure
   # Enter your Access Key ID, Secret Access Key, Region
   ```

2. **Use environment variables:**
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```

3. **Use IAM roles (for EC2):**
   ```python
   # IAM roles are automatically detected on EC2 instances
   ```

### Issue: S3 Bucket Access Denied

**Symptoms:**
```
DataMoveError: S3 operation failed
ClientError: An error occurred (AccessDenied) when calling the GetObject operation
```

**Solutions:**
1. **Check bucket permissions:**
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "s3:GetObject",
                   "s3:ListBucket"
               ],
               "Resource": [
                   "arn:aws:s3:::your-bucket",
                   "arn:aws:s3:::your-bucket/*"
               ]
           }
       ]
   }
   ```

2. **Verify bucket and key exist:**
   ```bash
   aws s3 ls s3://your-bucket/
   aws s3 ls s3://your-bucket/path/to/file.csv
   ```

3. **Test S3 access:**
   ```python
   import boto3
   
   s3 = boto3.client('s3')
   response = s3.list_objects_v2(Bucket='your-bucket', Prefix='path/')
   print(response)
   ```

### Issue: Invalid S3 URI Format

**Symptoms:**
```
DataMoveError: Invalid S3 URI format
ValueError: Invalid S3 path format
```

**Solutions:**
1. **Use correct S3 URI format:**
   ```python
   # Correct format
   s3_uri = "s3://bucket-name/path/to/file.csv"
   
   # Incorrect formats
   # "https://s3.amazonaws.com/bucket/file.csv"  # Wrong
   # "bucket/file.csv"  # Wrong
   # "s3://bucket"  # Missing file path
   ```

2. **Validate S3 URI:**
   ```python
   import re
   
   def validate_s3_uri(uri):
       pattern = r'^s3://[a-z0-9.-]+/.+'
       return bool(re.match(pattern, uri))
   ```

## Data Type and Conversion Issues

### Issue: Vector Data Type Errors

**Symptoms:**
```
DataTypeError: Cannot convert vector data
psycopg2.errors.InvalidTextRepresentation: invalid input syntax for type vector
```

**Solutions:**
1. **Ensure pgvector extension:**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Check vector format in CSV:**
   ```python
   # Vectors should be arrays of numbers
   df = pd.DataFrame({
       'id': [1, 2],
       'embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Correct format
   })
   ```

3. **Validate vector dimensions:**
   ```python
   # All vectors should have same dimensions
   embeddings = df['embedding'].tolist()
   dimensions = [len(emb) for emb in embeddings]
   assert len(set(dimensions)) == 1, "All vectors must have same dimensions"
   ```

### Issue: JSON Data Type Errors

**Symptoms:**
```
DataTypeError: Cannot convert JSON data
psycopg2.errors.InvalidTextRepresentation: invalid input syntax for type json
```

**Solutions:**
1. **Ensure valid JSON format:**
   ```python
   import json
   
   # Validate JSON data
   for idx, row in df.iterrows():
       try:
           json.loads(row['json_column'])
       except json.JSONDecodeError as e:
           print(f"Invalid JSON at row {idx}: {e}")
   ```

2. **Convert Python objects to JSON strings:**
   ```python
   df['metadata'] = df['metadata'].apply(json.dumps)
   ```

### Issue: Date/Time Conversion Errors

**Symptoms:**
```
DataTypeError: Cannot convert datetime data
ValueError: time data '2023-13-45' does not match format
```

**Solutions:**
1. **Standardize date formats:**
   ```python
   df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
   df['date_column'] = df['date_column'].dt.strftime('%Y-%m-%d')
   ```

2. **Handle invalid dates:**
   ```python
   # Find invalid dates
   invalid_dates = df[df['date_column'].isna()]
   print(f"Found {len(invalid_dates)} invalid dates")
   ```

## Error Reference

### DataMoveError Hierarchy

```
DataMoveError (Base exception)
├── ValidationError (Schema/data validation failures)
│   ├── SchemaConflictError (Schema compatibility issues)
│   ├── CaseSensitivityError (Case-sensitive column conflicts)
│   └── DataTypeError (Data type conversion issues)
└── DatabaseOperationError (Database connection/operation failures)
```

### Common Error Contexts

Each DataMoveError includes a `context` dictionary with detailed information:

```python
try:
    result = await use_case.execute(...)
except DataMoveError as e:
    print(f"Error: {e}")
    print(f"Context: {e.context}")
    
    # Common context keys:
    # - error_type: Specific error category
    # - file_path: Path to CSV file
    # - table_name: Target table name
    # - operation_stage: Where the error occurred
    # - original_error: Original exception message
    # - suggestion: Recommended solution
```

### Error Type Categories

| Error Type | Description | Common Causes |
|------------|-------------|---------------|
| `file_not_found` | CSV file doesn't exist | Wrong path, file moved |
| `permission_denied` | File access denied | File permissions, locked file |
| `s3_operation_failed` | S3 access failed | Credentials, permissions, network |
| `invalid_s3_uri` | Malformed S3 URI | Wrong URI format |
| `database_connection_failed` | Can't connect to DB | Service down, wrong credentials |
| `validation_failed` | Schema validation failed | Column mismatch, type conflicts |
| `case_sensitivity_conflict` | Column name case conflicts | Mixed case column names |
| `constraint_violation` | Data violates constraints | NULL in NOT NULL column |

## Getting Help

### Enable Debug Logging

```python
import logging
from dataload.config import logger

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)
```

### Collect Diagnostic Information

```python
async def collect_diagnostics():
    """Collect system information for troubleshooting."""
    
    import sys
    import pandas as pd
    import asyncpg
    
    print("=== DataMove Diagnostics ===")
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"AsyncPG version: {asyncpg.__version__}")
    
    # Test database connection
    try:
        db = DBConnection()
        await db.initialize()
        print("✅ Database connection: OK")
        await db.close()
    except Exception as e:
        print(f"❌ Database connection: {e}")
    
    # Test file access
    test_file = "test.csv"
    try:
        pd.DataFrame({'test': [1, 2, 3]}).to_csv(test_file, index=False)
        df = pd.read_csv(test_file)
        os.remove(test_file)
        print("✅ File operations: OK")
    except Exception as e:
        print(f"❌ File operations: {e}")
```

### Report Issues

When reporting issues, include:

1. **Error message and full traceback**
2. **DataMove version:** `pip show vector-dataloader`
3. **Python version:** `python --version`
4. **Operating system**
5. **Sample CSV data (anonymized)**
6. **Database schema (if relevant)**
7. **Configuration (without credentials)**

### Community Resources

- **GitHub Issues:** Report bugs and feature requests
- **Documentation:** Check latest documentation for updates
- **Examples:** Review example scripts for best practices

---

*This troubleshooting guide covers the most common issues. For additional help, check the project documentation or open an issue on GitHub.*