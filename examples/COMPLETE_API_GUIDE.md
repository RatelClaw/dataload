# Complete API to Vector Store Guide

## 🎯 Overview

This guide shows you how to load data from APIs (including the specific endpoint you requested: `https://api.restful-api.dev/objects`) and store it in PostgreSQL with vector embeddings for semantic search.

## 📊 Your API Endpoint Results

When you run the examples with your specified API endpoint, here's what you get:

### API Response Structure

```json
[
  {
    "id": "1",
    "name": "Google Pixel 6 Pro",
    "data": {
      "color": "Cloudy White",
      "capacity": "128 GB"
    }
  },
  {
    "id": "2",
    "name": "Apple iPhone 12 Mini, 256GB, Blue",
    "data": null
  }
  // ... more devices
]
```

### Processed Data (13 devices loaded)

- **Rows**: 13 device records
- **Columns**: 19 columns after flattening nested JSON
- **Key Fields**: device_id, device_name, color, capacity, price, generation, etc.

## 🚀 Quick Start (Works Immediately)

### 1. Test API Loading (No Setup Required)

```bash
python examples/simple_api_example.py
```

This will:

- ✅ Load data from `https://api.restful-api.dev/objects`
- ✅ Process nested JSON structures
- ✅ Show you the data structure
- ✅ Work without any database or API key setup

### 2. Test Basic JSON Processing

```bash
python examples/api_json_loader_example.py
```

This demonstrates:

- ✅ JSON flattening
- ✅ Column mapping
- ✅ Data transformations
- ✅ CSV compatibility

## 🔧 Full Setup (For Database + Embeddings)

### Prerequisites

1. **PostgreSQL with pgvector extension**
2. **Python packages**: `pip install asyncpg pandas aiohttp`
3. **Optional**: Google Gemini API key for real embeddings

### Environment Variables

```bash
# Database (required for full example)
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=vector_db
export DB_USER=postgres
export DB_PASSWORD=your_password

# Embeddings (optional - uses mock if not provided)
export GEMINI_API_KEY=your_gemini_api_key
```

### Setup Script

```bash
python examples/setup_environment.py
```

## 📱 Device Data Example (Your API)

Here's exactly what happens with your API endpoint:

### Input Data

```json
{
  "id": "1",
  "name": "Google Pixel 6 Pro",
  "data": {
    "color": "Cloudy White",
    "capacity": "128 GB"
  }
}
```

### Processing Steps

1. **Flatten nested JSON**: `data.color` → `data_color`
2. **Column mapping**: `id` → `device_id`, `name` → `device_name`
3. **Create descriptions**: Combine name + specs for embeddings
4. **Generate embeddings**: Convert text to vectors for similarity search
5. **Store in PostgreSQL**: With vector columns for search

### Final Database Schema

```sql
CREATE TABLE devices_simple (
    device_id INTEGER PRIMARY KEY,
    device_name TEXT,
    color TEXT,
    capacity TEXT,
    price NUMERIC,
    description TEXT,
    device_name_enc VECTOR(384),    -- Embedding for device name
    description_enc VECTOR(384)     -- Embedding for description
);
```

### Sample Queries

```sql
-- Find devices similar to "iPhone"
SELECT device_name, color, price
FROM devices_simple
ORDER BY device_name_enc <-> (
    SELECT device_name_enc
    FROM devices_simple
    WHERE device_name LIKE '%iPhone%'
    LIMIT 1
)
LIMIT 5;

-- Semantic search for "wireless headphones"
SELECT device_name, description
FROM devices_simple
ORDER BY description_enc <-> embedding_for('wireless headphones')
LIMIT 10;
```

## 🌐 Additional Free APIs

The comprehensive example includes these free APIs:

### 1. Blog Posts

- **URL**: `https://jsonplaceholder.typicode.com/posts`
- **Data**: Blog posts with titles and content
- **Embeddings**: Full text for content search

### 2. User Profiles

- **URL**: `https://jsonplaceholder.typicode.com/users`
- **Data**: User info with nested address/company data
- **Embeddings**: Professional profiles and contact info

### 3. Albums & Photos

- **URL**: `https://jsonplaceholder.typicode.com/albums`
- **Data**: Album metadata
- **Use Case**: Concurrent loading demonstration

## 🔄 Processing Configuration

### JSON Flattening

```python
config = {
    'flatten_nested': True,        # Convert nested objects to flat columns
    'separator': '_',              # Use underscore for nested field names
    'max_depth': 3,               # Limit nesting depth
    'handle_arrays': 'expand'      # How to handle arrays
}
```

### Column Mapping

```python
config = {
    'column_name_mapping': {
        'id': 'device_id',                    # Clean up field names
        'name': 'device_name',
        'data_color': 'color',
        'data_capacity': 'capacity',
        'data_price': 'price'
    }
}
```

### Data Transformations

```python
config = {
    'update_request_body_mapping': {
        # Create comprehensive description for embeddings
        'description': "concat({device_name}, ' - Color: ', coalesce({color}, 'N/A'), ', Capacity: ', coalesce({capacity}, 'N/A'))",

        # Create price categories
        'price_category': "case when {price} > 1000 then 'Premium' when {price} > 500 then 'Mid-range' else 'Budget' end"
    }
}
```

## 🎯 Embedding Types

### Separated Embeddings (Recommended)

```python
embed_type="separated"
embed_columns_names=["device_name", "description"]
```

Creates:

- `device_name_enc` column with embeddings for device names
- `description_enc` column with embeddings for descriptions

### Combined Embeddings

```python
embed_type="combined"
embed_columns_names=["device_name", "description"]
```

Creates:

- Single `embeddings` column combining both fields

## 🔍 Search Examples

### Exact Match

```sql
SELECT * FROM devices_simple WHERE device_name LIKE '%iPhone%';
```

### Semantic Similarity

```sql
-- Find devices similar to a specific device
SELECT d1.device_name, d1.color, d1.price,
       d1.device_name_enc <-> d2.device_name_enc as similarity
FROM devices_simple d1, devices_simple d2
WHERE d2.device_name = 'Apple iPhone 12 Pro Max'
ORDER BY similarity
LIMIT 5;
```

### Content Search

```sql
-- Search by description similarity
SELECT device_name, description
FROM devices_simple
ORDER BY description_enc <-> (
    SELECT description_enc
    FROM devices_simple
    WHERE description LIKE '%wireless%'
    LIMIT 1
)
LIMIT 10;
```

## 🛠️ Troubleshooting

### Issue: Async Error

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

### Issue: Database Connection

```
asyncpg.exceptions.InvalidCatalogNameError
```

**Solution**: Create database and extension:

```bash
createdb vector_db
psql vector_db -c "CREATE EXTENSION vector;"
```

### Issue: Missing Packages

```
ModuleNotFoundError: No module named 'asyncpg'
```

**Solution**: Install required packages:

```bash
pip install asyncpg pandas aiohttp python-dotenv
```

## 📈 Performance Tips

### Concurrent Loading

```python
# Load multiple APIs simultaneously
sources = [
    "https://api1.example.com/data",
    "https://api2.example.com/data"
]

df = await api_loader.load_json_concurrent(
    sources=sources,
    max_concurrent=3
)
```

### Batch Processing

```python
# Process large datasets in chunks
for chunk in chunked_data:
    result = await use_case.execute(
        source=chunk,
        table_name="large_dataset",
        embed_columns_names=["text_field"]
    )
```

### Optimize Embeddings

```python
# Use separated embeddings for different search types
embed_columns_names=["title", "content", "tags"]
embed_type="separated"

# This creates: title_enc, content_enc, tags_enc
# Allows targeted searches on specific fields
```

## 🎉 Next Steps

1. **Start Simple**: Run `python examples/simple_api_example.py`
2. **Add Database**: Set up PostgreSQL and run full examples
3. **Customize**: Modify configurations for your specific APIs
4. **Scale Up**: Add more APIs and concurrent processing
5. **Build Apps**: Create search interfaces using the vector data

## 📚 File Reference

- **`simple_api_example.py`** - Your specific API endpoint example
- **`comprehensive_api_to_vector_example.py`** - Multiple APIs with full features
- **`api_json_loader_example.py`** - Basic loader functionality (fixed async)
- **`setup_environment.py`** - Environment validation and setup
- **`mock_embedding_provider.py`** - Mock embeddings for testing
- **`README_API_Examples.md`** - Detailed documentation

## 🔗 Resources

- [PostgreSQL pgvector](https://github.com/pgvector/pgvector)
- [Google Gemini API](https://ai.google.dev/docs)
- [JSONPlaceholder API](https://jsonplaceholder.typicode.com/)
- [RESTful API Test Service](https://restful-api.dev/)

---

**Ready to start?** Run `python examples/simple_api_example.py` to see your API data in action! 🚀
