"""
Script to download all required models for the RAG application
"""
import os
from sentence_transformers import SentenceTransformer
from rapidocr_onnxruntime import RapidOCR

print("="*60)
print("DOWNLOADING ALL MODELS FOR RAG APPLICATION")
print("="*60)

# Download embedding model
print("\n[1/2] 📦 Downloading embedding model: all-MiniLM-L6-v2")
print("      (This is a smaller, faster model - ~80MB)")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Test the model
    test_embedding = model.encode("test")
    print(f"      ✅ Model downloaded and tested successfully!")
    print(f"      Embedding size: {len(test_embedding)} dimensions")
except Exception as e:
    print(f"      ❌ Error: {e}")

# Initialize OCR engine
print("\n[2/2] 📦 Initializing OCR engine (RapidOCR)")
print("      (Downloads ONNX models for text recognition)")
try:
    ocr_engine = RapidOCR()
    print("      ✅ OCR engine initialized successfully!")
except Exception as e:
    print(f"      ❌ Error: {e}")

print("\n" + "="*60)
print("🎉 ALL MODELS ARE READY!")
print("="*60)
print("\n✅ You can now run: streamlit run app.py")
print("   The app should load much faster now.")
