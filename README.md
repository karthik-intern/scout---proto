# 📄 RAG Document Assistant

An AI-powered document assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from your documents.

## ✨ Features

- **Multi-format Support**: PDF, DOCX, XLSX, CSV, TXT, images (PNG, JPG)
- **OCR Capabilities**: Extracts text from scanned documents and images
- **Semantic Search**: Uses vector embeddings for intelligent document retrieval
- **Chat Interface**: Conversational Q&A with context from your documents
- **Source Citations**: Shows which documents/pages answers came from

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

### 3. Open in Browser

The app will open at: http://localhost:8501

## 📝 How to Use

1. **Upload Documents**: Drag and drop your files (supports multiple files)
2. **Process Documents**: Click "Process Documents" button
3. **Enter API Key**: Add your Groq API key in the sidebar
4. **Ask Questions**: Type questions about your documents in the chat

## ⚡ Performance Tips

### First Run is Slow?

The first time you process documents, the app will:
- Download the embedding model (~80MB) - **happens once**
- Download OCR models - **happens once**
- These are cached locally for future use

### Speed Improvements Made:

1. **Smaller Embedding Model**: Using `all-MiniLM-L6-v2` (faster than BGE)
2. **Caching**: Models are cached using `@st.cache_resource`
3. **Optimized Chunking**: Larger chunks (1000 tokens) for faster processing

### Expected Processing Times:

- **First document upload**: 1-2 minutes (downloading models)
- **Subsequent uploads**: 10-30 seconds (depending on document size)
- **Query response**: 2-5 seconds

## 🔑 API Key Setup

### Option 1: Environment Variable (Recommended for deployment)

```bash
export GROQ_API_KEY="your_api_key_here"
```

### Option 2: Streamlit Secrets (For Streamlit Cloud)

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_api_key_here"
```

### Option 3: Manual Entry

Enter your API key in the sidebar when running the app.

Get your Groq API key: https://console.groq.com

## 🏗️ Architecture

```
Documents → Parser → Text Chunker → Embeddings → ChromaDB
                                                      ↓
User Question → Embedding → Vector Search → Groq LLM → Answer
```

### Tech Stack:

- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.1)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **OCR**: RapidOCR
- **Document Processing**: LangChain, PyMuPDF, python-docx, pandas

## 📦 Dependencies

- `streamlit` - Web interface
- `langchain` - Document processing framework
- `chromadb` - Vector database
- `sentence-transformers` - Text embeddings
- `pymupdf` - PDF parsing
- `rapidocr-onnxruntime` - OCR for images/scanned PDFs
- `groq` - LLM API client
- `pandas`, `openpyxl` - Excel/CSV processing
- `python-docx` - Word document processing

## 🐛 Troubleshooting

### "Building vector store" takes too long

- **First time**: This is normal (downloading models)
- **Every time**: Check your internet connection or restart the app

### OCR not working

Make sure you have the correct version:
```bash
pip install rapidocr-onnxruntime<=1.2.3
```

### Out of memory errors

- Process fewer documents at once
- Use smaller chunk sizes in the code

## 📚 Project Structure

```
proct/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── download_models.py     # Helper to pre-download models
├── setup_models.py        # One-time setup script
└── README.md             # This file
```

## 🚢 Deployment

### Deploy to Streamlit Cloud:

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repository
4. Add `GROQ_API_KEY` in Secrets
5. Deploy!

### Environment Variables for Production:

```bash
GROQ_API_KEY=your_key_here
```

## 📄 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Feel free to submit issues or pull requests.

## 💡 Future Improvements

- [ ] Support for more LLM providers (OpenAI, Anthropic)
- [ ] Multi-language support
- [ ] Document preview with highlighting
- [ ] Export chat history
- [ ] Batch processing of folders
- [ ] Custom chunking strategies
- [ ] Advanced filters (date range, file type)

---

**Built with ❤️ using LangChain, Streamlit, and Groq**
