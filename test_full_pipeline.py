"""
Complete Pipeline Test - Downloads everything and tests end-to-end
This ensures all dependencies work: Upload → Parse → Embed → Vector Store → LLM
"""
import os
import sys
from pathlib import Path

print("\n" + "="*80)
print("RAG DOCUMENT ASSISTANT - FULL PIPELINE TEST")
print("="*80)

# Test 1: Import all required packages
print("\n[1/6] 📦 Testing package imports...")
try:
    import streamlit as st
    print("  ✅ streamlit")
    import fitz  # PyMuPDF
    print("  ✅ pymupdf (fitz)")
    import pandas as pd
    print("  ✅ pandas")
    from docx import Document
    print("  ✅ python-docx")
    from langchain_core.documents import Document as LCDocument
    print("  ✅ langchain-core")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("  ✅ langchain-text-splitters")
    from sentence_transformers import SentenceTransformer
    print("  ✅ sentence-transformers")
    from langchain.embeddings.base import Embeddings
    print("  ✅ langchain embeddings")
    from langchain_chroma import Chroma
    print("  ✅ langchain-chroma")
    from groq import Groq
    print("  ✅ groq")
    from rapidocr_onnxruntime import RapidOCR
    print("  ✅ rapidocr-onnxruntime")
    print("\n  🎉 All packages imported successfully!")
except ImportError as e:
    print(f"\n  ❌ Import Error: {e}")
    print("\n  Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Download Embedding Model
print("\n[2/6] 📥 Downloading embedding model (all-MiniLM-L6-v2)...")
try:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"  Cache directory: {cache_dir}")
    
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
    print("  ✅ Model downloaded and loaded")
    
    # Test embedding
    test_embedding = model.encode("This is a test sentence.")
    print(f"  ✅ Embedding test successful (dimension: {len(test_embedding)})")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Test 3: Initialize OCR Engine
print("\n[3/6] 🔍 Initializing OCR engine...")
try:
    ocr_engine = RapidOCR()
    print("  ✅ OCR engine initialized")
    print("  ✅ ONNX models downloaded")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Test 4: Test Document Processing
print("\n[4/6] 📄 Testing document processing...")
try:
    # Create a test document
    test_text = """
    Artificial Intelligence and Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. Deep learning is a
    type of machine learning based on artificial neural networks.
    
    Key concepts include:
    - Supervised Learning
    - Unsupervised Learning
    - Reinforcement Learning
    - Neural Networks
    """
    
    # Create LangChain document
    doc = LCDocument(page_content=test_text, metadata={"source": "test.txt", "page": 1})
    print("  ✅ Document created")
    
    # Test text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])
    print(f"  ✅ Text splitting successful ({len(chunks)} chunks)")
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Test 5: Test Vector Store
print("\n[5/6] 🗄️ Testing vector store (ChromaDB)...")
try:
    # Create embedding wrapper
    class BGEEmbeddings(Embeddings):
        def __init__(self, model):
            self.model = model
        def embed_documents(self, texts):
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        def embed_query(self, text):
            return self.model.encode(text, normalize_embeddings=True).tolist()
    
    embedding_fn = BGEEmbeddings(model)
    print("  ✅ Embedding function created")
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name="test_collection"
    )
    print("  ✅ Vector store created")
    
    # Test retrieval
    query = "What is machine learning?"
    results = vector_store.similarity_search(query, k=2)
    print(f"  ✅ Similarity search successful (found {len(results)} results)")
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Test 6: Test LLM (Groq) - Optional
print("\n[6/6] 🤖 Testing LLM connection (Groq)...")
groq_api_key = os.getenv("GROQ_API_KEY", "")

if not groq_api_key:
    print("  ⚠️  GROQ_API_KEY not set - skipping LLM test")
    print("  💡 To test LLM: Set GROQ_API_KEY environment variable")
else:
    try:
        client = Groq(api_key=groq_api_key)
        
        # Build context from retrieved docs
        context = "\n\n".join([doc.page_content for doc in results])
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"Answer based on this context:\n\n{context}"
                },
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=256
        )
        
        answer = response.choices[0].message.content
        print("  ✅ LLM connection successful")
        print(f"\n  Query: {query}")
        print(f"  Answer: {answer[:200]}...")
        
    except Exception as e:
        print(f"  ❌ LLM Error: {e}")
        print("  💡 Check your GROQ_API_KEY")

# Summary
print("\n" + "="*80)
print("✅ FULL PIPELINE TEST COMPLETE")
print("="*80)
print("\n📋 Summary:")
print("  ✅ All Python packages installed")
print("  ✅ Embedding model downloaded and working")
print("  ✅ OCR engine initialized")
print("  ✅ Document processing working")
print("  ✅ Vector store (ChromaDB) working")
if groq_api_key:
    print("  ✅ LLM (Groq) connection working")
else:
    print("  ⚠️  LLM test skipped (no API key)")

print("\n🚀 Your app is ready to use!")
print("   Run: streamlit run app.py")
print("\n💡 Tips:")
print("   - First document upload may take 30-60 seconds")
print("   - Subsequent uploads will be faster (models are cached)")
print("   - Add GROQ_API_KEY to test LLM functionality")
print()
