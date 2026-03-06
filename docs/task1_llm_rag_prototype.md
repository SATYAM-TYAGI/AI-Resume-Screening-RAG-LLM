## Task 1 — LLM-Powered AI Prototype (Resume screening assistant)

### What’s implemented
- **LLM**: **OpenAI** when `OPENAI_API_KEY` is set; else **Ollama** (e.g. `llama3.1`) or no-LLM
- **RAG**:
  - **Embeddings**: **OpenAI** when key set; else **local hashing embeddings** (pure Python)
  - **Vector DB / vector store**: **SQLite-based persistent vector store** (prototype-scale)
  - **Retrieval**: cosine similarity (dot-product on normalized vectors), top‑k
- **Chunking strategy**:
  - Normalize whitespace
  - Heuristic **section splitting** (e.g., Skills / Experience / Education)
  - Paragraph packing into bounded chunks with overlap (see `src/rag/chunking.py`)
- **Prompt engineering**:
  - JSON schema for screening output
  - “Use only provided CONTEXT” constraints
  - Citation requirements (`chunk_id`)
- **Basic UI**: Streamlit (`app/streamlit_app.py`)

### How to run
See `README.md` (PowerShell quickstart).

### Where to look in code
- **UI**: `app/streamlit_app.py`
- **Ingestion**: `src/rag/loaders.py` + `src/rag/chunking.py`
- **Embeddings**: `src/rag/embeddings.py`
- **Vector store**: `src/rag/vectorstore.py`
- **RAG orchestration**: `src/rag/rag.py`
- **Prompts**: `src/rag/prompts.py`

