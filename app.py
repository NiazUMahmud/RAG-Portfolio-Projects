"""
RAG Document Q&A — Streamlit App
=================================
Chat with your PDFs using a full Retrieval-Augmented Generation pipeline:

    PDF upload → PyMuPDF text extraction → recursive chunking
    → SentenceTransformer embeddings → ChromaDB vector store
    → similarity search → Groq LLM answer with cited sources

Run with:
    streamlit run app.py
"""

import os
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv

import chromadb
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
PERSIST_DIR = APP_DIR / "vector_store"
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

# The notebook's .env uses `Groq_API_KEY`; support both spellings.
load_dotenv(APP_DIR / ".env")
load_dotenv(APP_DIR / "data" / "notebook" / ".env")


def get_groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_KEY") or ""


# ---------------------------------------------------------------------------
# Cached resources (loaded once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Connecting to vector store...")
def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine",
                  "description": "PDF document embeddings for RAG"},
    )


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------
def extract_pdf_documents(file_bytes: bytes, source_name: str) -> list[Document]:
    """Extract one Document per page from a PDF's raw bytes."""
    docs = []
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_num, page in enumerate(pdf):
            text = page.get_text().strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": source_name,
                              "page": page_num + 1,
                              "total_pages": pdf.page_count},
                ))
    return docs


def chunk_documents(docs: list[Document], chunk_size: int,
                    chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def ingest_chunks(chunks: list[Document]) -> int:
    """Embed chunks and add them to the ChromaDB collection."""
    if not chunks:
        return 0
    model = get_embedding_model()
    collection = get_collection()

    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True,
                              show_progress_bar=False)
    start = collection.count()
    collection.add(
        ids=[f"chunk_{start + i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# Retrieval + generation
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
    """Return the top-k most similar chunks with cosine similarity scores."""
    model = get_embedding_model()
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
    )
    return [
        {"content": doc, "metadata": meta, "similarity": 1 - dist}
        for doc, meta, dist in zip(results["documents"][0],
                                   results["metadatas"][0],
                                   results["distances"][0])
    ]


def generate_answer(query: str, context_docs: list[dict[str, Any]],
                    api_key: str, model_name: str) -> str:
    context = "\n\n---\n\n".join(
        f"[{d['metadata'].get('source', 'unknown')} — page "
        f"{d['metadata'].get('page', '?')}]\n{d['content']}"
        for d in context_docs
    )
    llm = ChatGroq(api_key=api_key, model=model_name, temperature=0.1)
    prompt = f"""You are a helpful assistant that answers questions using only the provided context.

Context from documents:
{context}

Question: {query}

Answer based on the context above. If the context does not contain the answer, say so clearly. Be concise and cite the source document/page when relevant."""
    return llm.invoke(prompt).content


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")
st.title("📚 RAG Document Q&A")
st.caption("Upload PDFs, build a vector index, and chat with your documents — "
           "powered by ChromaDB, SentenceTransformers, and Groq.")

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = get_groq_api_key()
    if not api_key:
        api_key = st.text_input("Groq API key", type="password",
                                help="Get a free key at console.groq.com")
    else:
        st.success("Groq API key loaded from .env")

    model_name = st.selectbox("LLM model", GROQ_MODELS)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 4)

    st.divider()
    st.header("📥 Document ingestion")
    chunk_size = st.slider("Chunk size (chars)", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 100, step=50)

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf",
                                      accept_multiple_files=True)
    if uploaded_files and st.button("Ingest uploaded PDFs", type="primary",
                                    use_container_width=True):
        with st.status("Ingesting documents...", expanded=True) as status:
            total_chunks = 0
            for file in uploaded_files:
                st.write(f"Extracting **{file.name}**...")
                docs = extract_pdf_documents(file.getvalue(), file.name)
                chunks = chunk_documents(docs, chunk_size, chunk_overlap)
                total_chunks += ingest_chunks(chunks)
                st.write(f"→ {len(docs)} pages, {len(chunks)} chunks indexed")
            status.update(label=f"Done — {total_chunks} chunks added",
                          state="complete")

    st.divider()
    collection = get_collection()
    st.metric("Chunks in vector store", collection.count())
    if st.button("🗑️ Clear vector store", use_container_width=True):
        chromadb.PersistentClient(path=str(PERSIST_DIR)) \
                .delete_collection(COLLECTION_NAME)
        get_collection.clear()
        st.session_state.messages = []
        st.rerun()

# --- Chat area ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for i, src in enumerate(msg.get("sources", []), 1):
            with st.expander(
                f"Source {i}: {src['metadata'].get('source', 'unknown')} "
                f"(page {src['metadata'].get('page', '?')}) — "
                f"similarity {src['similarity']:.2f}"
            ):
                st.text(src["content"][:1500])

if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if get_collection().count() == 0:
            answer = ("The vector store is empty — upload and ingest at least "
                      "one PDF from the sidebar first.")
            st.warning(answer)
            sources = []
        elif not api_key:
            answer = "Please provide a Groq API key in the sidebar."
            st.warning(answer)
            sources = []
        else:
            with st.spinner("Retrieving relevant chunks..."):
                sources = retrieve(query, top_k)
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(query, sources, api_key, model_name)
                    st.markdown(answer)
                except Exception as e:
                    answer = f"LLM call failed: {e}"
                    st.error(answer)
                    sources = []
            for i, src in enumerate(sources, 1):
                with st.expander(
                    f"Source {i}: {src['metadata'].get('source', 'unknown')} "
                    f"(page {src['metadata'].get('page', '?')}) — "
                    f"similarity {src['similarity']:.2f}"
                ):
                    st.text(src["content"][:1500])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
