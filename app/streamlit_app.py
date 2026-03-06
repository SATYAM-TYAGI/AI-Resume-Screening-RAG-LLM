import os
import sys
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

# Ensure project root (with `src/`) is on PYTHONPATH even when Streamlit
# changes the working directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.rag.llm import LLMConfig, OpenAILLM, OllamaLLM
from src.rag.rag import RAGService, RAGConfig
from src.rag.schemas import ScreeningResult


def _get_data_dir() -> Path:
    data_dir = os.getenv("DATA_DIR", ".data")
    return Path(data_dir).resolve()


def _init_services() -> RAGService:
    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    rag_cfg = RAGConfig(
        persist_dir=str(data_dir / "vectors"),
        collection_name="resumes",
        top_k=6,
        min_relevance_score=0.02,
    )

    llm: Optional[object] = None
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if api_key:
        llm = OpenAILLM(LLMConfig(api_key=api_key, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
    elif os.getenv("USE_OLLAMA", "0") == "1":
        llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3.1"))
    return RAGService(rag_cfg, llm=llm, data_dir=data_dir)


def _render_sidebar(rag: RAGService) -> None:
    st.sidebar.header("Index")
    st.sidebar.caption("Upload resumes.")

    uploaded = st.sidebar.file_uploader(
        "Upload resume files (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if st.sidebar.button("Ingest uploaded files", disabled=not uploaded):
        with st.spinner("Ingesting and indexing..."):
            results = rag.ingest_uploaded_files(uploaded)
        st.sidebar.success(f"Indexed {len(results)} file(s).")

    st.sidebar.divider()
    st.sidebar.subheader("Candidate resumes")
    docs = rag.list_documents()
    if not docs:
        st.sidebar.info("No resumes indexed yet.")
    else:
        for d in docs:
            st.sidebar.write(f"- {d}")

    if docs and st.sidebar.button("Clear Resumes"):
        rag.reset_index()
        st.sidebar.success("Resumes cleared.")


def _screen_tab(rag: RAGService) -> None:
    st.subheader("Resume screening")
    st.caption("Enter a job description, then select which resumes to evaluate.")

    job_desc = st.text_area("Job description", height=180, placeholder="Paste the job description here...")
    doc_names = rag.list_documents()

    selected = st.multiselect("Resumes to screen", options=doc_names, default=doc_names[:1] if doc_names else [])
    if st.button("Run screening", disabled=not (job_desc.strip() and selected)):
        with st.spinner("Running screening (RAG-grounded)..."):
            results: List[ScreeningResult] = rag.screen_resumes(job_desc=job_desc, resume_doc_names=selected)
        st.success("Done.")

        for r in results:
            st.markdown(f"### {r.resume_name}")
            st.write(f"**Overall fit**: {r.overall_fit}  |  **Confidence**: {r.confidence:.2f}")
            st.write(r.summary)
            st.markdown("**Strengths**")
            for s in r.strengths:
                st.write(f"- {s}")
            st.markdown("**Gaps / risks**")
            for g in r.gaps:
                st.write(f"- {g}")
            st.markdown("**Evidence (citations)**")
            for c in r.citations:
                st.write(f"- `{c.source}` (chunk {c.chunk_id}): {c.quote}")
            st.divider()


def _compare_tab(rag: RAGService) -> None:
    st.subheader("Compare multiple resumes (advanced: multi-document reasoning)")
    st.caption("Compares selected resumes against the same job description with evidence for each claim.")

    job_desc = st.text_area("Job description", height=160, key="compare_jd")
    doc_names = rag.list_documents()
    selected = st.multiselect("Resumes to compare", options=doc_names, default=doc_names[:2] if len(doc_names) >= 2 else doc_names)

    if st.button("Compare", disabled=not (job_desc.strip() and len(selected) >= 2)):
        with st.spinner("Comparing resumes..."):
            report = rag.compare_resumes(job_desc=job_desc, resume_doc_names=selected)
        st.success("Done.")
        st.markdown(report)


def _chat_tab(rag: RAGService) -> None:
    st.subheader("RAG chat (source-grounded)")
    st.caption("Ask questions about the indexed resumes. The assistant will cite sources or refuse.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question (e.g., 'Which candidate has Kubernetes experience?')")
    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag.answer_question(user_q)
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})


def _inject_streamlit_secrets() -> None:
    """Copy Streamlit Cloud secrets into os.environ so RAG/LLM code sees them."""
    try:
        for key, value in st.secrets.items():
            if key in os.environ:
                continue
            if isinstance(value, dict):
                # e.g. [openai] api_key = "sk-..." -> OPENAI_API_KEY
                for sub_key, sub_value in value.items():
                    if sub_value is not None:
                        env_name = f"{key.upper()}_{sub_key.upper()}" if key else sub_key.upper()
                        os.environ[env_name] = str(sub_value)
            elif value is not None:
                os.environ[key] = str(value)
    except Exception:
        pass


def main() -> None:
    load_dotenv()
    _inject_streamlit_secrets()

    st.set_page_config(page_title="AI Resume Screener", layout="wide")
    st.title("AI Resume Screener (RAG + LLM)")

    rag = _init_services()
    _render_sidebar(rag)

    tab_screen, tab_compare, tab_chat = st.tabs(["Screen", "Compare", "Chat"])
    with tab_screen:
        _screen_tab(rag)
    with tab_compare:
        _compare_tab(rag)
    with tab_chat:
        _chat_tab(rag)


if __name__ == "__main__":
    main()

