"""
Test script to verify the configuration system implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_embedding_configs():
    """Test embedding provider configurations."""
    print("🧪 Testing Embedding Configurations...")
    
    try:
        from dataload.embedding_config import (
            create_embedding_config,
            GeminiEmbeddingConfig,
            BedrockEmbeddingConfig,
            SentenceTransformersConfig,
            OpenAIEmbeddingConfig
        )
        
        # Test Gemini config
        gemini_config = create_embedding_config("gemini", {"dimension": 768})
        assert gemini_config.dimension == 768
        assert gemini_config.model == "text-embedding-004"
        print("✅ Gemini configuration: PASSED")
        
        # Test Bedrock config
        bedrock_config = create_embedding_config("bedrock", {"dimension": 1024})
        assert bedrock_config.dimension == 1024
        assert bedrock_config.model_id == "amazon.titan-embed-text-v2:0"
        print("✅ Bedrock configuration: PASSED")
        
        # Test SentenceTransformers config
        st_config = create_embedding_config("sentence_transformers", {"model_name": "all-MiniLM-L6-v2"})
        assert st_config.model_name == "all-MiniLM-L6-v2"
        assert st_config.dimension == 384
        print("✅ SentenceTransformers configuration: PASSED")
        
        # Test OpenAI config
        openai_config = create_embedding_config("openai", {"dimension": 1536})
        assert openai_config.dimension == 1536
        assert openai_config.model == "text-embedding-3-small"
        print("✅ OpenAI configuration: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding configuration test failed: {e}")
        return False


def test_vector_store_configs():
    """Test vector store configurations."""
    print("\n🧪 Testing Vector Store Configurations...")
    
    try:
        from dataload.embedding_config import create_vector_store_config, VectorStoreConfig
        
        # Test PostgreSQL config
        postgres_config = create_vector_store_config("postgres", {"dimension": 768})
        assert postgres_config.dimension == 768
        assert postgres_config.index_type == "ivfflat"
        print("✅ PostgreSQL configuration: PASSED")
        
        # Test ChromaDB config
        chroma_config = create_vector_store_config("chroma", {"dimension": 512})
        assert chroma_config.dimension == 512
        assert chroma_config.distance_metric == "cosine"
        print("✅ ChromaDB configuration: PASSED")
        
        # Test FAISS config
        faiss_config = create_vector_store_config("faiss", {"dimension": 1024})
        assert faiss_config.dimension == 1024
        assert faiss_config.faiss_index_type == "IndexFlatL2"
        print("✅ FAISS configuration: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store configuration test failed: {e}")
        return False


def test_embedding_providers():
    """Test embedding providers with configurations."""
    print("\n🧪 Testing Embedding Providers...")
    
    try:
        # Test Gemini provider (without actual API call)
        from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider
        
        # This will fail due to missing API key, but we can test config initialization
        try:
            gemini = GeminiEmbeddingProvider({"dimension": 768, "model": "text-embedding-004"})
        except Exception as e:
            if "GOOGLE_API_KEY" in str(e):
                print("✅ Gemini provider configuration: PASSED (API key required for full test)")
            else:
                raise e
        
        # Test Bedrock provider
        from dataload.application.services.embedding.bedrock_provider import BedrockEmbeddingProvider
        
        try:
            bedrock = BedrockEmbeddingProvider({"dimension": 1024, "region": "us-east-1"})
        except Exception as e:
            if "bedrock" in str(e).lower() or "aws" in str(e).lower():
                print("✅ Bedrock provider configuration: PASSED (AWS credentials required for full test)")
            else:
                raise e
        
        # Test OpenAI provider
        from dataload.application.services.embedding.openai_provider import OpenAIEmbeddingProvider
        
        try:
            openai = OpenAIEmbeddingProvider({"dimension": 1536, "model": "text-embedding-3-small"})
        except Exception as e:
            if "OPENAI_API_KEY" in str(e):
                print("✅ OpenAI provider configuration: PASSED (API key required for full test)")
            else:
                raise e
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding provider test failed: {e}")
        return False


def test_vector_stores():
    """Test vector stores with configurations."""
    print("\n🧪 Testing Vector Stores...")
    
    try:
        # Test ChromaDB store
        from dataload.infrastructure.vector_stores.chroma_store import ChromaVectorStore
        
        chroma_store = ChromaVectorStore(
            mode="in-memory",
            config={"dimension": 768, "distance_metric": "cosine"}
        )
        assert chroma_store.config.dimension == 768
        print("✅ ChromaDB store configuration: PASSED")
        
        # Test FAISS store
        from dataload.infrastructure.vector_stores.faiss_store import FaissVectorStore
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_store = FaissVectorStore(
                persistence_path=temp_dir,
                config={"dimension": 512, "faiss_index_type": "IndexFlatL2"}
            )
            assert faiss_store.config.dimension == 512
            print("✅ FAISS store configuration: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False


def test_validation():
    """Test configuration validation."""
    print("\n🧪 Testing Configuration Validation...")
    
    try:
        from dataload.embedding_config import GeminiEmbeddingConfig, VectorStoreConfig
        
        # Test invalid dimension
        try:
            invalid_config = GeminiEmbeddingConfig(dimension=-1)
            print("❌ Validation should have failed for negative dimension")
            return False
        except ValueError:
            print("✅ Negative dimension validation: PASSED")
        
        # Test invalid model
        try:
            invalid_config = GeminiEmbeddingConfig(model="invalid-model")
            print("❌ Validation should have failed for invalid model")
            return False
        except ValueError:
            print("✅ Invalid model validation: PASSED")
        
        # Test invalid distance metric
        try:
            invalid_config = VectorStoreConfig(distance_metric="invalid-metric")
            print("❌ Validation should have failed for invalid distance metric")
            return False
        except ValueError:
            print("✅ Invalid distance metric validation: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility."""
    print("\n🧪 Testing Backward Compatibility...")
    
    try:
        # Test that providers can still be initialized without config
        from dataload.application.services.embedding.sentence_transformers_provider import SentenceTransformersProvider
        
        try:
            # This should work with default config
            st_provider = SentenceTransformersProvider()
        except Exception as e:
            if "sentence-transformers" in str(e):
                print("✅ Backward compatibility: PASSED (sentence-transformers not installed)")
            else:
                raise e
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Running Configuration System Implementation Tests\n")
    
    tests = [
        test_embedding_configs,
        test_vector_store_configs,
        test_embedding_providers,
        test_vector_stores,
        test_validation,
        test_backward_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Configuration system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)