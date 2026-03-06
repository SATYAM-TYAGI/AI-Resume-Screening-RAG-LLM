# AI Resume Screener (RAG + LLM) — Prototype

This is a lightweight **LLM-powered resume screening assistant** with **RAG (embeddings + vector DB)**, a **chunking strategy**, **prompt guardrails**, and a **Streamlit UI**.

## Features
- **Ingest resumes**: PDF / DOCX / TXT
- **Chunking strategy**: section-aware + bounded chunks with overlap
- **Vector DB**: local persistent **SQLite-based vector store** (prototype-friendly)
- **Embeddings**: **OpenAI** when `OPENAI_API_KEY` is set, else **local hashing embeddings**
- **LLM**: **OpenAI** (gpt-4o-mini) when key is set; else **Ollama** (e.g. llama3.1) if `USE_OLLAMA=1`
- **RAG chat**: source-grounded answers with citations over indexed resumes
- **Resume screening**: generate a structured fit assessment vs a job description
- **Guardrails**:
  - **Source-grounded answering** 
  - **Confidence gating** via retrieval score thresholds
- **Advanced capability (Task 3)**: **multi-document reasoning** (compare multiple resumes with evidence)

## Design choices & rationale
- **Hybrid stack**: OpenAI is used when you set `OPENAI_API_KEY` (better embeddings + LLM); otherwise local hashing embeddings + Ollama or no-LLM mode keep the app runnable without any API.
- **SQLite vector store**: simple, file-based storage that is:
  - easy to reason about,
  - sufficient for a small corpus of resumes,
  - closer to how a real vector DB would be used (ids, metadata, similarity search).
- **Hashing embeddings**: a lightweight alternative to large embedding models:
  - pure Python, fast enough, and robust to Python 3.13,
  - still supports cosine-style retrieval for RAG.
- **OpenAI vs Ollama**: when an API key is set, OpenAI is used for both embeddings and chat; otherwise Ollama (or no-LLM) keeps the app working locally and free.
- **Guardrails in the RAG layer**:
  - retrieval confidence thresholds reduce hallucinated answers,
  - prompts explicitly enforce “answer only from context” + citations,
  - resume screening is done per-document to avoid cross-candidate leakage.

## Quickstart (Windows PowerShell)

```powershell
cd "C:\Users\Admin\Documents\AI Krisent\AI Resume Screener"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
copy .env.example .env
# Edit .env: set OPENAI_API_KEY for OpenAI, or USE_OLLAMA=1 for local Ollama
streamlit run .\app\streamlit_app.py
```

### Notes
- With **OpenAI**: set `OPENAI_API_KEY` in `.env`; the app uses OpenAI for embeddings and chat.
- Without a key: set `USE_OLLAMA=1` for local Ollama, or leave both off for **no-LLM** (evidence-only) mode.
- All app data is stored locally under `.data/`.

### Libraries, APIs, and services used
- **Python libraries**:
  - `streamlit`: web UI.
  - `python-dotenv`: load `.env` configuration.
  - `pypdf`, `python-docx`: parse PDF / DOCX resumes.
  - `pydantic`: data models (`ScreeningResult`, citations, etc.).
  - `requests`: HTTP client for calling Ollama.
- **APIs / services**:
  - **OpenAI** (optional): embeddings + chat when `OPENAI_API_KEY` is set.
  - **Ollama** (optional): local LLM when no OpenAI key and `USE_OLLAMA=1`.
  - Local filesystem for the SQLite vector store and uploaded resumes.

## Repository layout
- `app/streamlit_app.py`: UI
- `src/rag/`: ingestion, chunking, embeddings, retrieval, prompting
- `docs/`: writeups for Tasks 1–4

## Task mapping
- **Task 1 (LLM + RAG prototype)**: implemented via `src/rag/*` and the Streamlit UI for ingesting resumes, chunking, indexing, and querying with an LLM.
- **Task 2 (hallucination & quality control)**: guardrails in prompts and `RAGService` (confidence thresholds, source-grounded answers, scoped retrieval).
- **Task 3 (advanced capability)**: **multi-document reasoning** in the “Compare” tab, where multiple resumes are evaluated against the same JD with evidence-backed ranking.
- **Task 4 (enterprise architecture)**: high-level internal assistant architecture documented in `docs/task4_enterprise_ai_architecture.md`.

