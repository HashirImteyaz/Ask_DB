#!/usr/bin/env python3
"""Test LlamaIndex import"""

try:
    from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    print("✅ LlamaIndex core imports successful!")
    
    # Test basic functionality
    docs = [Document(text="Test document")]
    index = VectorStoreIndex.from_documents(docs)
    print("✅ VectorStoreIndex creation successful!")
    
    retriever = index.as_retriever()
    print("✅ Retriever creation successful!")
    
    print("\n🎉 LlamaIndex is fully functional!")
    
except ImportError as e:
    print(f"❌ LlamaIndex import failed: {e}")
    print("\nTrying alternative import paths...")
    
    try:
        from llama_index import Settings, VectorStoreIndex, Document
        print("✅ Alternative import paths work!")
    except ImportError as e2:
        print(f"❌ Alternative imports also failed: {e2}")
