from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .chunking import chunk_text
from .embeddings import EmbeddingConfig, make_embedder
from .loaders import load_document
from .prompts import ANSWER_SYSTEM, COMPARE_SYSTEM, SCREEN_SYSTEM
from .schemas import Citation, RetrievedChunk, ScreeningResult
from .vectorstore import SQLiteVectorStore, VectorStoreConfig


@dataclass(frozen=True)
class RAGConfig:
    persist_dir: str
    collection_name: str = "resumes"
    top_k: int = 6
    min_relevance_score: float = 0.02


class RAGService:
    def __init__(self, cfg: RAGConfig, *, llm=None, data_dir: Path):
        self._cfg = cfg
        self._llm = llm  # optional
        self._data_dir = data_dir

        embed_cfg = EmbeddingConfig(
            api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        )
        self._embedder = make_embedder(embed_cfg)

        vs_cfg = VectorStoreConfig(persist_dir=cfg.persist_dir, collection_name=cfg.collection_name)
        self._vs = SQLiteVectorStore(vs_cfg)

        self._uploads_dir = self._data_dir / "uploads"
        self._uploads_dir.mkdir(parents=True, exist_ok=True)

    def reset_index(self) -> None:
        self._vs.reset()

    def list_documents(self) -> List[str]:
        return self._vs.list_doc_names()

    def ingest_uploaded_files(self, uploaded_files) -> List[str]:
        saved_paths: List[Path] = []
        for uf in uploaded_files:
            out_path = self._uploads_dir / uf.name
            out_path.write_bytes(uf.getbuffer())
            saved_paths.append(out_path)

        indexed: List[str] = []
        for p in saved_paths:
            self.ingest_file(p)
            indexed.append(p.name)
        return indexed

    def ingest_file(self, path: Path) -> None:
        doc = load_document(path)
        chunks = chunk_text(doc.text, doc_name=doc.doc_name)

        texts = [c.text for c in chunks]
        vecs = self._embedder.embed(texts)
        ids = [c.chunk_id for c in chunks]
        metas = []
        for c in chunks:
            m = dict(c.meta)
            m.update({"doc_name": doc.doc_name, "source": doc.doc_name, **doc.meta})
            metas.append(m)

        self._vs.upsert(ids=ids, embeddings=vecs, documents=texts, metadatas=metas)

    def _retrieve(self, query: str, *, where: Optional[Dict] = None) -> List[RetrievedChunk]:
        qv = self._embedder.embed([query])[0]
        res = self._vs.query(query_embedding=qv, top_k=self._cfg.top_k, where=where)
        out: List[RetrievedChunk] = []
        for _id, doc, meta, score in res:
            out.append(
                RetrievedChunk(
                    doc_name=meta.get("doc_name", "unknown"),
                    chunk_id=_id,
                    text=doc,
                    score=score,
                    page=meta.get("page"),
                )
            )
        return out

    def _guardrail_context(self, chunks: List[RetrievedChunk]) -> Tuple[bool, str]:
        """
        Guardrail 1: Confidence gating based on retrieval score.
        """
        if not chunks:
            return False, "No relevant context retrieved."
        best = max(c.score for c in chunks)
        if best < self._cfg.min_relevance_score:
            return False, f"Low retrieval confidence (best score={best:.2f} < {self._cfg.min_relevance_score:.2f})."
        return True, ""

    def answer_question(self, question: str) -> str:
        chunks = self._retrieve(question)
        if not chunks:
            return (
                "I don't have enough evidence in the indexed resumes to answer that.\n\n"
                "Reason: No resume content was retrieved (index may be empty or query had no match)."
            )

        ok, reason = self._guardrail_context(chunks)
        context = "\n\n".join([f"[{c.chunk_id} | score={c.score:.2f}]\n{c.text}" for c in chunks])
        low_confidence_note = "" if ok else f"Note: Retrieval confidence was low ({reason}). Answer from the context below as best you can and mention if evidence is limited.\n\n"

        if not self._llm:
            # no-LLM fallback: return top chunks only
            top = chunks[:3]
            bullets = "\n".join([f"- `{c.chunk_id}`: {c.text[:220].strip()}..." for c in top])
            return (
                "LLM is not configured (set `OPENAI_API_KEY`). Here are the most relevant excerpts:\n\n"
                f"{bullets}"
            )

        user = f"{low_confidence_note}QUESTION:\n{question}\n\nCONTEXT:\n{context}"
        try:
            answer = self._llm.complete_markdown(system=ANSWER_SYSTEM, user=user)
            return answer.strip()
        except Exception as e:
            # Graceful degradation when LLM quota/errors occur.
            top = chunks[:3]
            bullets = "\n".join([f"- `{c.chunk_id}`: {c.text[:220].strip()}..." for c in top])
            return (
                "LLM is currently unavailable (for example, due to rate limits or exhausted quota).\n\n"
                "Here are the most relevant excerpts from the resumes instead:\n\n"
                f"{bullets}"
            )

    def screen_resumes(self, *, job_desc: str, resume_doc_names: Sequence[str]) -> List[ScreeningResult]:
        results: List[ScreeningResult] = []
        for doc_name in resume_doc_names:
            where = {"doc_name": doc_name}
            chunks = self._retrieve(job_desc, where=where)
            ok, reason = self._guardrail_context(chunks)

            # Always provide something; if low confidence, degrade gracefully.
            if not self._llm or not ok:
                results.append(self._screen_fallback(doc_name, job_desc, chunks, reason))
                continue

            context = "\n\n".join([f"[{c.chunk_id} | score={c.score:.2f}]\n{c.text}" for c in chunks])
            user = f"JOB DESCRIPTION:\n{job_desc}\n\nCONTEXT (single resume: {doc_name}):\n{context}"
            try:
                data = self._llm.complete_json(system=SCREEN_SYSTEM, user=user)
                sr = self._parse_screening_json(doc_name, data, chunks)
                results.append(sr)
            except Exception as e:
                err_name = e.__class__.__name__
                if err_name == "ConnectionError":
                    fallback_reason = (
                        "LLM connection failed. "
                        "If using OpenAI: check internet and firewall. "
                        "If using Ollama: start it with 'ollama serve' and ensure the model is pulled (e.g. ollama pull llama3.1)."
                    )
                else:
                    fallback_reason = f"LLM error: {err_name}"
                results.append(self._screen_fallback(doc_name, job_desc, chunks, fallback_reason))
        return results

    def _parse_screening_json(self, doc_name: str, data: Dict, chunks: List[RetrievedChunk]) -> ScreeningResult:
        # If model forgets citations, patch them with top chunks.
        citations = []
        for c in data.get("citations", []) or []:
            if isinstance(c, dict) and c.get("chunk_id") and c.get("quote"):
                citations.append(Citation(source=doc_name, chunk_id=str(c["chunk_id"]), quote=str(c["quote"])))
        if not citations:
            top = chunks[:3]
            citations = [
                Citation(source=doc_name, chunk_id=t.chunk_id, quote=t.text[:240].strip()) for t in top
            ]

        return ScreeningResult(
            resume_name=doc_name,
            overall_fit=data.get("overall_fit", "Unclear"),
            confidence=float(data.get("confidence", 0.4)),
            summary=str(data.get("summary", "")),
            strengths=[str(x) for x in (data.get("strengths") or [])][:8],
            gaps=[str(x) for x in (data.get("gaps") or [])][:8],
            citations=citations[:6],
        )

    def _screen_fallback(
        self,
        doc_name: str,
        job_desc: str,
        chunks: List[RetrievedChunk],
        reason: str,
    ) -> ScreeningResult:
        # Very simple heuristic: average of top scores to approximate confidence.
        top = chunks[:5]
        conf = (sum(c.score for c in top) / len(top)) if top else 0.0
        overall = "Unclear" if conf < 0.35 else ("Moderate" if conf < 0.55 else "Strong")

        citations = [Citation(source=doc_name, chunk_id=c.chunk_id, quote=c.text[:240].strip()) for c in top[:3]]
        summary = (
            "LLM is not configured or retrieval confidence is low. "
            "Showing best-effort, evidence-first excerpts.\n"
            f"Reason: {reason}"
        )
        strengths = ["See cited excerpts for evidence; configure LLM for richer synthesis."]
        gaps = ["Insufficient evidence to fully assess without stronger retrieval / LLM synthesis."]
        return ScreeningResult(
            resume_name=doc_name,
            overall_fit=overall,
            confidence=max(0.0, min(1.0, conf)),
            summary=summary,
            strengths=strengths,
            gaps=gaps,
            citations=citations,
        )

    def compare_resumes(self, *, job_desc: str, resume_doc_names: Sequence[str]) -> str:
        # Retrieve context per resume independently to avoid cross-contamination.
        per_resume_blocks: List[str] = []
        for doc_name in resume_doc_names:
            chunks = self._retrieve(job_desc, where={"doc_name": doc_name})
            ok, reason = self._guardrail_context(chunks)
            if not ok:
                per_resume_blocks.append(f"## {doc_name}\nNo strong evidence retrieved. Reason: {reason}\n")
                continue
            context = "\n\n".join([f"[{c.chunk_id} | score={c.score:.2f}]\n{c.text}" for c in chunks])
            per_resume_blocks.append(f"## {doc_name}\n{context}\n")

        combined_context = "\n\n".join(per_resume_blocks)

        if not self._llm:
            return (
                "LLM is not configured (set `OPENAI_API_KEY`).\n\n"
                "Top evidence per resume:\n\n"
                + "\n\n".join(per_resume_blocks[:6])
            )

        user = f"JOB DESCRIPTION:\n{job_desc}\n\nCONTEXT (grouped by resume):\n{combined_context}\n\n" \
               "Return a markdown report with:\n" \
               "- a comparison table\n" \
               "- a ranked recommendation\n" \
               "- bullet evidence per candidate with chunk_id citations.\n"
        try:
            return self._llm.complete_markdown(system=COMPARE_SYSTEM, user=user)
        except Exception as e:
            return (
                "LLM is currently unavailable (for example, due to rate limits or exhausted quota).\n\n"
                "Here is the raw evidence per resume instead:\n\n"
                + "\n\n".join(per_resume_blocks[:6])
            )

