# LinkedIn Post — RAG Document Q&A App

🚀 I just built and shipped a RAG (Retrieval-Augmented Generation) app that lets you chat with your PDFs!

Ever wished you could just *ask* a 100-page document a question instead of scrolling through it? That's exactly what this app does.

📄 Upload any PDF → ask questions in plain English → get answers grounded in the actual document, with page-level citations and similarity scores for every source.

Under the hood, here's the full pipeline I implemented:

🔹 Document ingestion — PyMuPDF extracts text page by page
🔹 Chunking — LangChain's RecursiveCharacterTextSplitter breaks text into overlapping chunks so context isn't lost at boundaries
🔹 Embeddings — SentenceTransformers (all-MiniLM-L6-v2) converts each chunk into a 384-dimensional vector
🔹 Vector store — ChromaDB persists the embeddings and runs cosine-similarity search
🔹 Generation — Groq's blazing-fast Llama 3.3 70B answers using only the retrieved context
🔹 UI — Streamlit chat interface with tunable chunk size, overlap, and top-k retrieval

My favorite part: the app shows you *why* it gave each answer — every response comes with the exact source chunks, page numbers, and similarity scores. No black box.

Biggest lessons from building this:
✅ Chunking strategy matters more than you'd think — chunk size and overlap directly change retrieval quality
✅ Cosine vs. L2 distance in your vector store is not a detail you can ignore
✅ Grounding the LLM with "answer only from the context" dramatically reduces hallucinations

Tested it on the "Attention Is All You Need" paper — asked it to explain multi-head attention, and it answered with the exact formula, citing page 5. 🎯

Tech stack: Python · LangChain · ChromaDB · SentenceTransformers · Groq · Streamlit · PyMuPDF

The code is part of my RAG portfolio series — happy to share it or talk about the architecture. What documents would you want to chat with?

#RAG #GenAI #LLM #LangChain #Python #Streamlit #MachineLearning #NLP #VectorDatabase #AIEngineering #DataScience #Groq #ChromaDB #BuildInPublic
