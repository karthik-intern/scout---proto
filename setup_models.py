"""
One-time setup: Download all models before running the app
Run this once: python setup_models.py
"""
import os
from sentence_transformers import SentenceTransformer
from rapidocr_onnxruntime import RapidOCR

print("\n" + "="*70)
print("SETTING UP RAG DOCUMENT ASSISTANT - ONE-TIME SETUP")
print("="*70)

# Set cache directory
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")
os.makedirs(cache_dir, exist_ok=True)

print(f"\n📁 Cache directory: {cache_dir}")

# Download embedding model
print("\n[1/2] 📦 Downloading embedding model: all-MiniLM-L6-v2")
print("      Size: ~80MB (this may take 1-2 minutes)")
print("      Downloading...", end="", flush=True)

try:
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
    # Test the model
    test_text = "This is a test sentence."
    embedding = model.encode(test_text)
    print(" DONE! ✅")
    print(f"      Model loaded successfully")
    print(f"      Embedding dimensions: {len(embedding)}")
    print(f"      Location: {cache_dir}")
except Exception as e:
    print(f"\n      ❌ Error: {e}")
    exit(1)

# Initialize OCR engine
print("\n[2/2] 📦 Initializing OCR engine (RapidOCR)")
print("      Downloading ONNX models...", end="", flush=True)

try:
    ocr_engine = RapidOCR()
    print(" DONE! ✅")
    print("      OCR engine ready")
except Exception as e:
    print(f"\n      ❌ Error: {e}")
    exit(1)

print("\n" + "="*70)
print("✅ SETUP COMPLETE! All models are cached and ready.")
print("="*70)
print("\n🚀 You can now run your app:")
print("   python -m streamlit run app.py")
print("\n💡 Next time you upload documents, it will be MUCH faster!")
print("   (Models are now cached locally)\n")
