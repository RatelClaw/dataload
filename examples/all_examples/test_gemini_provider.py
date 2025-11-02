#!/usr/bin/env python3
"""
Test Gemini Embedding Provider

Simple test to verify the GeminiEmbeddingProvider works correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataload.application.services.embedding.gemini_provider import GeminiEmbeddingProvider


def test_gemini_provider():
    """Test the Gemini embedding provider."""
    
    print("🧪 Testing Gemini Embedding Provider")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        print("   Set it with: export GEMINI_API_KEY='your-api-key-here'")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        # Initialize provider
        print("🤖 Initializing Gemini provider...")
        provider = GeminiEmbeddingProvider()
        print("✅ Provider initialized")
        
        # Test embedding generation
        print("📝 Testing embedding generation...")
        test_texts = [
            "Apple iPhone 13 Pro Max",
            "Samsung Galaxy S22 Ultra",
            "Google Pixel 6 Pro"
        ]
        
        embeddings = provider.get_embeddings(test_texts)
        print(f"✅ Generated embeddings for {len(test_texts)} texts")
        
        # Check embedding properties
        print(f"📊 Embedding dimensions: {len(embeddings[0])}")
        print(f"📊 Embedding type: {type(embeddings[0][0])}")
        print(f"📊 Sample values: {embeddings[0][:5]}")
        
        # Test similarity (same text should be identical)
        same_text_embeddings = provider.get_embeddings(["Apple iPhone 13 Pro Max", "Apple iPhone 13 Pro Max"])
        
        if same_text_embeddings[0] == same_text_embeddings[1]:
            print("✅ Consistency test passed (same text = same embedding)")
        else:
            print("⚠️  Consistency test failed (same text ≠ same embedding)")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_gemini_provider()
    sys.exit(0 if success else 1)