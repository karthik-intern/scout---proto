import os
import tempfile
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from groq import Groq

# ---------- Page Config ----------
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG Document Assistant")
st.markdown("Upload documents, ask questions, get AI-powered answers from your files.")


# ---------- Embedding Model ----------
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def load_embedding_model():
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")
    os.makedirs(cache_dir, exist_ok=True)
    return SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)


class BGEEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()


# ---------- Document Parsers ----------
def parse_pdf(file_path):
    """Parse PDF - extract text from each page"""
    documents = []
    pdf = fitz.open(file_path)
    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        text = page.get_text().strip()
        if text:
            documents.append(
                LCDocument(
                    page_content=text,
                    metadata={"source": os.path.basename(file_path), "page": page_num + 1}
                )
            )
    return documents


def parse_docx(file_path):
    """Parse Word document"""
    doc = Document(file_path)
    text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    return [LCDocument(page_content=text, metadata={"source": os.path.basename(file_path), "page": 1})]


def parse_excel(file_path):
    """Parse Excel file - each sheet becomes a document"""
    documents = []
    excel = pd.ExcelFile(file_path)
    for sheet in excel.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        text = df.astype(str).to_string(index=False)
        documents.append(
            LCDocument(page_content=text, metadata={"source": os.path.basename(file_path), "sheet": sheet})
        )
    return documents


def parse_csv(file_path):
    """Parse CSV file"""
    df = pd.read_csv(file_path)
    text = df.astype(str).to_string(index=False)
    return [LCDocument(page_content=text, metadata={"source": os.path.basename(file_path), "page": 1})]


def parse_txt(file_path):
    """Parse text file"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [LCDocument(page_content=text, metadata={"source": os.path.basename(file_path), "page": 1})]


# ---------- Get API Key from Secrets ----------
groq_api_key = ""
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = os.getenv("GROQ_API_KEY", "")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("**Supported formats:** PDF, DOCX, XLSX, CSV, TXT")
    st.markdown("---")
    if groq_api_key:
        st.success("✅ API Key configured")
    else:
        st.error("❌ GROQ_API_KEY not found in secrets")
    st.markdown("---")
    st.markdown("Built with LangChain + ChromaDB + Groq")


# ---------- Session State ----------
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------- File Upload ----------
st.header("1️⃣ Upload Documents")

uploaded_files = st.file_uploader(
    "Drop your files here",
    type=["pdf", "docx", "xlsx", "csv", "txt"],
    accept_multiple_files=True
)

if uploaded_files and st.button("🔄 Process Documents", type="primary"):
    # Load embedding model (cached after first load)
    with st.spinner("Loading embedding model..."):
        embedding_model = load_embedding_model()
    
    all_documents = []
    progress = st.progress(0)
    status = st.status("Processing documents...", expanded=True)

    for i, uploaded_file in enumerate(uploaded_files):
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            if ext == ".pdf":
                docs = parse_pdf(tmp_path)
            elif ext == ".docx":
                docs = parse_docx(tmp_path)
            elif ext == ".xlsx":
                docs = parse_excel(tmp_path)
            elif ext == ".csv":
                docs = parse_csv(tmp_path)
            elif ext == ".txt":
                docs = parse_txt(tmp_path)
            else:
                docs = []
                status.write(f"⚠️ Unsupported: {uploaded_file.name}")

            all_documents.extend(docs)
            status.write(f"✅ Parsed: {uploaded_file.name} ({len(docs)} chunk(s))")

        except Exception as e:
            status.write(f"❌ Error with {uploaded_file.name}: {e}")

        finally:
            try:
                os.remove(tmp_path)
            except (PermissionError, FileNotFoundError):
                pass

        progress.progress((i + 1) / len(uploaded_files))

    if all_documents:
        # Chunk documents
        status.write("🧠 Chunking documents...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunked_documents = []
        for doc in all_documents:
            if len(doc.page_content.strip()) > 50:
                chunks = text_splitter.create_documents(
                    [doc.page_content],
                    metadatas=[doc.metadata]
                )
                chunked_documents.extend(chunks)
            else:
                chunked_documents.append(doc)

        # Build vector store
        status.write("📦 Building vector store...")
        embedding_fn = BGEEmbeddings(embedding_model)
        vector_store = Chroma.from_documents(
            documents=chunked_documents,
            embedding=embedding_fn,
            collection_name="rag_docs"
        )

        st.session_state.documents = all_documents
        st.session_state.vector_store = vector_store

        status.update(label="✅ Processing complete!", state="complete")
        st.success(f"Processed {len(all_documents)} document(s) into {len(chunked_documents)} chunks.")


# ---------- Document Preview ----------
if st.session_state.documents:
    st.header("2️⃣ Document Preview")
    with st.expander(f"📋 View parsed documents ({len(st.session_state.documents)} total)", expanded=False):
        for i, doc in enumerate(st.session_state.documents):
            st.markdown(f"**Document {i+1}** — `{doc.metadata.get('source', 'unknown')}`")
            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
            st.markdown("---")


# ---------- Chat / Q&A ----------
if st.session_state.vector_store:
    st.header("3️⃣ Ask Questions")

    if not groq_api_key:
        st.warning("⚠️ Enter your Groq API key in the sidebar to enable Q&A.")
    else:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_question = st.chat_input("Ask a question about your documents...")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve relevant chunks
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    )
                    relevant_docs = retriever.invoke(user_question)

                    # Build context
                    context = "\n\n---\n\n".join([
                        f"[Source: {doc.metadata.get('source', 'unknown')}, "
                        f"Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                        for doc in relevant_docs
                    ])

                    # Query Groq
                    client = Groq(api_key=groq_api_key)
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful assistant that answers questions based on the provided document context. "
                                    "Use only the information from the context below. If the answer is not in the context, "
                                    "say so clearly. Cite the source document and page when possible.\n\n"
                                    f"Context:\n{context}"
                                )
                            },
                            {"role": "user", "content": user_question}
                        ],
                        temperature=0.3,
                        max_tokens=1024
                    )

                    answer = response.choices[0].message.content
                    st.markdown(answer)

                    # Show sources
                    with st.expander("📚 Sources"):
                        for doc in relevant_docs:
                            st.markdown(
                                f"- **{doc.metadata.get('source', 'unknown')}** "
                                f"(Page {doc.metadata.get('page', 'N/A')})"
                            )

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
