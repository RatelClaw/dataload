# API to Vector Store Examples

This directory contains comprehensive examples showing how to load data from APIs and store it in PostgreSQL with vector embeddings using the APIJSONStorageLoader.

## 📁 Files Overview

### Main Examples

- **`comprehensive_api_to_vector_example.py`** - Complete example with multiple APIs, advanced processing, and full vector store integration
- **`simple_api_example.py`** - Simple example using the restful-api.dev endpoint you specified
- **`api_json_loader_example.py`** - Fixed version showing basic APIJSONStorageLoader usage (now with proper async/await)

### Setup and Utilities

- **`setup_environment.py`** - Environment setup and validation script
- **`README_API_Examples.md`** - This documentation file

## 🚀 Quick Start

### 1. Environment Setup

First, run the setup script to check your environment:

```bash
python examples/setup_environment.py
```

This will check for:

- Python 3.8+
- Required packages
- PostgreSQL with pgvector
- Gemini API configuration

### 2. Set Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Database Configuration
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=vector_db
export DB_USER=postgres
export DB_PASSWORD=your_password

# Gemini API Key (required for embeddings)
export GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run Examples

#### Simple Example (Your API Endpoint)

```bash
python examples/simple_api_example.py
```

This loads data from `https://api.restful-api.dev/objects` and demonstrates:

- API data loading
- Nested JSON processing
- Embedding generation
- PostgreSQL storage

#### Comprehensive Example (Multiple APIs)

```bash
python examples/comprehensive_api_to_vector_example.py
```

This demonstrates:

- Multiple free APIs
- Complex data transformations
- Concurrent API loading
- Error handling scenarios

#### Basic APIJSONStorageLoader Usage

```bash
python examples/api_json_loader_example.py
```

This shows basic loader functionality without database operations.

## 📊 API Endpoints Used

### Primary Example (Your Request)

- **URL**: `https://api.restful-api.dev/objects`
- **Data**: Device information (phones, tablets, etc.)
- **Structure**: Nested JSON with `data` field containing specifications

### Additional Free APIs (Comprehensive Example)

- **JSONPlaceholder Posts**: `https://jsonplaceholder.typicode.com/posts`
- **JSONPlaceholder Users**: `https://jsonplaceholder.typicode.com/users`
- **JSONPlaceholder Albums**: `https://jsonplaceholder.typicode.com/albums`
- **HTTPBin Test**: `https://httpbin.org/json`

## 🔧 Configuration Options

### JSON Processing

```python
config = {
    'flatten_nested': True,        # Flatten nested objects
    'separator': '_',              # Field separator for flattened names
    'max_depth': 3,               # Maximum nesting depth
    'handle_arrays': 'expand'      # How to handle arrays: 'expand', 'join', 'first'
}
```

### Column Mapping

```python
config = {
    'column_name_mapping': {
        'api_field_name': 'db_column_name',
        'nested_field': 'clean_name'
    }
}
```

### Data Transformations

```python
config = {
    'update_request_body_mapping': {
        'computed_field': "concat({field1}, ' - ', {field2})",
        'price_category': "case when {price} > 1000 then 'Premium' else 'Budget' end"
    }
}
```

## 🎯 Embedding Types

### Separated Embeddings

Creates individual embedding columns for each specified field:

```python
embed_type="separated"
embed_columns_names=["name", "description"]
# Creates: name_enc, description_enc columns
```

### Combined Embeddings

Creates a single embedding column combining all specified fields:

```python
embed_type="combined"
embed_columns_names=["name", "description"]
# Creates: embeddings column
```

## 📋 Database Schema Examples

### Device Data (from restful-api.dev)

```sql
CREATE TABLE devices_simple (
    device_id INTEGER PRIMARY KEY,
    device_name TEXT,
    color TEXT,
    capacity TEXT,
    price NUMERIC,
    description TEXT,
    device_name_enc VECTOR(768),    -- Gemini embedding
    description_enc VECTOR(768)     -- Gemini embedding
);
```

### Vector Similarity Queries

```sql
-- Find devices similar to iPhone
SELECT device_name, color, price
FROM devices_simple
ORDER BY device_name_enc <-> (
    SELECT device_name_enc
    FROM devices_simple
    WHERE device_name LIKE '%iPhone%'
    LIMIT 1
)
LIMIT 5;

-- Semantic search in descriptions
SELECT device_name, description
FROM devices_simple
ORDER BY description_enc <-> (
    SELECT embedding
    FROM generate_embedding('wireless headphones')
)
LIMIT 10;
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Async/Await Error

```
TypeError: object of type 'coroutine' has no len()
```

**Solution**: Use `await` with `load_json()`:

```python
# Wrong
df = loader.load_json(data)

# Correct
df = await loader.load_json(data)
```

#### 2. Missing Environment Variables

```
ValueError: GEMINI_API_KEY environment variable is required
```

**Solution**: Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

#### 3. Database Connection Error

```
asyncpg.exceptions.InvalidCatalogNameError: database "vector_db" does not exist
```

**Solution**: Create the database:

```bash
createdb vector_db
psql vector_db -c "CREATE EXTENSION vector;"
```

#### 4. API Timeout

```
APIError: Failed to load data from API: timeout
```

**Solution**: Increase timeout or check network:

```python
api_loader = APIJSONStorageLoader(timeout=60)
```

### Getting Help

1. **Run setup script**: `python examples/setup_environment.py`
2. **Check logs**: Look for detailed error messages in console output
3. **Test components individually**: Use the simple examples first
4. **Verify API access**: Test API endpoints in browser first

## 📚 Advanced Usage

### Concurrent API Loading

```python
# Load from multiple APIs simultaneously
sources = [
    "https://api1.example.com/data",
    "https://api2.example.com/data",
    "https://api3.example.com/data"
]

df = await api_loader.load_json_concurrent(
    sources=sources,
    max_concurrent=3
)
```

### Custom Authentication

```python
# API with authentication
api_loader = APIJSONStorageLoader(
    api_token="your-api-key",
    default_headers={
        "Authorization": "Bearer your-token",
        "X-API-Key": "your-key"
    }
)
```

### Error Handling

```python
try:
    result = await use_case.execute(
        source=api_url,
        table_name="my_table",
        embed_columns_names=["text_field"]
    )

    if result.success:
        print(f"Success: {result.rows_processed} rows")
    else:
        print(f"Failed: {result.errors}")

except APIError as e:
    print(f"API Error: {e}")
except DatabaseOperationError as e:
    print(f"Database Error: {e}")
```

## 🎉 Next Steps

After running these examples successfully:

1. **Explore Vector Search**: Try similarity queries on your embedded data
2. **Add More APIs**: Integrate additional data sources
3. **Customize Processing**: Modify transformations for your use case
4. **Scale Up**: Use concurrent loading for multiple APIs
5. **Build Applications**: Create search interfaces using the vector data

## 📖 Additional Resources

- [PostgreSQL pgvector Documentation](https://github.com/pgvector/pgvector)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [APIJSONStorageLoader Source Code](../src/dataload/infrastructure/storage/api_json_loader.py)
- [Migration Guide](../docs/migration_guide_api_json_loader.md)
