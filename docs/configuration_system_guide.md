# Configuration System Guide

This guide explains the new configuration system for embedding providers and vector stores, which provides flexible, production-grade configuration management with sensible defaults.

## Overview

The new configuration system allows you to:

- Use default configurations without any setup
- Customize specific parameters while keeping defaults for others
- Validate configurations to prevent runtime errors
- Maintain backward compatibility with existing code

## Embedding Providers

### Before (Old Way)

```python
# Limited customization
embedding = SentenceTransformersProvider("all-MiniLM-L6-v2")
embedding = GeminiEmbeddingProvider()
```

### After (New Way)

```python
# Use defaults
embedding = SentenceTransformersProvider()

# Custom configuration
embedding = SentenceTransformersProvider({
    "model_name": "all-mpnet-base-v2",
    "dimension": 768,
    "device": "cuda",
    "normalize_embeddings": True
})

# Partial configuration (missing values use defaults)
embedding = GeminiEmbeddingProvider({
    "dimension": 1024  # Other values use defaults
})
```

## Available Configurations

### Gemini Embedding Provider

```python
config = {
    "model": "text-embedding-004",           # Default: "text-embedding-004"
    "dimension": 768,                        # Default: 768
    "task_type": "SEMANTIC_SIMILARITY",      # Default: "SEMANTIC_SIMILARITY"
    "api_key": "your-api-key"               # Default: from GOOGLE_API_KEY env var
}
embedding = GeminiEmbeddingProvider(config)
```

### Bedrock Embedding Provider

```python
config = {
    "model_id": "amazon.titan-embed-text-v2:0",  # Default: "amazon.titan-embed-text-v2:0"
    "region": "us-east-1",                       # Default: from AWS_REGION env var
    "dimension": 1024,                           # Default: 1024
    "content_type": "application/json"           # Default: "application/json"
}
embedding = BedrockEmbeddingProvider(config)
```

### SentenceTransformers Provider

```python
config = {
    "model_name": "all-MiniLM-L6-v2",       # Default: "all-MiniLM-L6-v2"
    "dimension": 384,                        # Default: 384 (auto-detected from model)
    "device": "auto",                        # Default: "auto" (cpu/cuda/auto)
    "normalize_embeddings": True             # Default: True
}
embedding = SentenceTransformersProvider(config)
```

### OpenAI Embedding Provider

```python
config = {
    "model": "text-embedding-3-small",      # Default: "text-embedding-3-small"
    "dimension": 1536,                      # Default: 1536
    "api_key": "your-api-key"              # Default: from OPENAI_API_KEY env var
}
embedding = OpenAIEmbeddingProvider(config)
```

## Vector Stores

### PostgreSQL Data Repository

```python
config = {
    "dimension": 1024,                      # Default: 1024
    "index_type": "ivfflat",               # Default: "ivfflat" (or "hnsw")
    "distance_metric": "cosine",           # Default: "cosine"
    "ivfflat_lists": 100,                  # Default: 100
    "hnsw_m": 16,                          # Default: 16
    "hnsw_ef_construction": 64             # Default: 64
}
repo = PostgresDataRepository(db_conn, config)
```

### ChromaDB Vector Store

```python
config = {
    "dimension": 1024,                      # Default: 1024
    "distance_metric": "cosine",           # Default: "cosine"
    "persist_directory": "./chroma_db"     # Default: "./chroma_db"
}
store = ChromaVectorStore(mode="persistent", config=config)
```

### FAISS Vector Store

```python
config = {
    "dimension": 1024,                      # Default: 1024
    "distance_metric": "euclidean",        # Default: "euclidean"
    "faiss_index_type": "IndexFlatL2"      # Default: "IndexFlatL2"
}
store = FaissVectorStore(config=config)
```

## Complete Workflow Example

```python
from dataload.application.services.embedding.sentence_transformers_provider import SentenceTransformersProvider
from dataload.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from dataload.infrastructure.db.data_repository import PostgresDataRepository

# 1. Create embedding provider with custom config
embedding_config = {
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "dimension": 768,
    "device": "cuda",
    "normalize_embeddings": True
}
embedding_provider = SentenceTransformersProvider(embedding_config)

# 2. Create vector store with matching dimension
vector_store_config = {
    "dimension": 768,  # Match embedding dimension
    "distance_metric": "cosine",
    "persist_directory": "./my_vector_db"
}
vector_store = ChromaVectorStore(mode="persistent", config=vector_store_config)

# 3. Create PostgreSQL repository with matching dimension
postgres_config = {
    "dimension": 768,  # Match embedding dimension
    "index_type": "hnsw",  # Use HNSW for better performance with lower dimensions
    "distance_metric": "cosine"
}
postgres_repo = PostgresDataRepository(db_conn, postgres_config)

# Now all components use consistent 768-dimensional embeddings
print(f"Embedding dimension: {embedding_provider.get_dimension()}")
print(f"Vector store dimension: {vector_store.config.dimension}")
print(f"PostgreSQL dimension: {postgres_repo.config.dimension}")
```

## Migration Guide

### Step 1: Update Embedding Provider Initialization

```python
# OLD
embedding = SentenceTransformersProvider("all-MiniLM-L6-v2")

# NEW (backward compatible)
embedding = SentenceTransformersProvider()  # Uses default model

# NEW (with custom config)
embedding = SentenceTransformersProvider({
    "model_name": "all-MiniLM-L6-v2",
    "dimension": 384
})
```

### Step 2: Update Vector Store Initialization

```python
# OLD
repo = PostgresDataRepository(db_conn)

# NEW (backward compatible)
repo = PostgresDataRepository(db_conn)  # Uses defaults

# NEW (with custom config)
repo = PostgresDataRepository(db_conn, {
    "dimension": 768,
    "index_type": "hnsw"
})
```

### Step 3: Ensure Dimension Consistency

```python
# Make sure all components use the same dimension
EMBEDDING_DIM = 768

embedding_provider = SentenceTransformersProvider({
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "dimension": EMBEDDING_DIM
})

vector_store = ChromaVectorStore(config={
    "dimension": EMBEDDING_DIM
})

postgres_repo = PostgresDataRepository(db_conn, {
    "dimension": EMBEDDING_DIM
})
```

## Configuration Validation

The system automatically validates configurations:

```python
# This will raise a ValueError
try:
    config = GeminiEmbeddingConfig(dimension=-1)  # Invalid dimension
except ValueError as e:
    print(f"Configuration error: {e}")

# This will raise a ValueError
try:
    config = VectorStoreConfig(distance_metric="invalid")  # Invalid metric
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Environment Variables

The system respects environment variables for sensitive data:

```bash
# Set in your environment
export GOOGLE_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export AWS_REGION="us-west-2"
```

```python
# These will use environment variables automatically
gemini = GeminiEmbeddingProvider()  # Uses GOOGLE_API_KEY
openai = OpenAIEmbeddingProvider()  # Uses OPENAI_API_KEY
bedrock = BedrockEmbeddingProvider()  # Uses AWS_REGION
```

## Best Practices

1. **Dimension Consistency**: Always ensure embedding providers and vector stores use the same dimension.

2. **Environment Variables**: Use environment variables for API keys and sensitive configuration.

3. **Validation**: Let the system validate your configurations to catch errors early.

4. **Defaults**: Rely on sensible defaults when possible to reduce configuration complexity.

5. **Documentation**: Document your custom configurations for team members.

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure all components use the same embedding dimension.

   ```python
   # Check dimensions match
   assert embedding_provider.get_dimension() == vector_store.config.dimension
   ```

2. **Invalid Configuration**: The system will raise `ValueError` for invalid configurations.

   ```python
   # Handle configuration errors
   try:
       provider = GeminiEmbeddingProvider(config)
   except ValueError as e:
       print(f"Invalid configuration: {e}")
   ```

3. **Missing API Keys**: Ensure environment variables are set for API-based providers.
   ```bash
   # Check environment variables
   echo $GOOGLE_API_KEY
   echo $OPENAI_API_KEY
   ```

### Getting Help

- Check the configuration validation error messages for specific issues
- Refer to the examples in `examples/configuration_examples.py`
- Run the tests in `tests/test_configuration_system.py` to verify your setup
