from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docx import Document as DocxDocument
from pypdf import PdfReader


@dataclass(frozen=True)
class LoadedDoc:
    doc_name: str
    text: str
    meta: Dict


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    parts: List[str] = []
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())
    return "\n".join(parts).strip()


def _read_pdf(path: Path) -> Tuple[str, Dict]:
    """
    Returns (text, meta). Meta includes per-page offsets if needed later.
    """
    reader = PdfReader(str(path))
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            pages.append(f"[PAGE {i+1}]\n{t.strip()}")
        else:
            pages.append(f"[PAGE {i+1}]\n")
    text = "\n\n".join(pages).strip()
    return text, {"num_pages": len(reader.pages)}


def load_document(path: Path) -> LoadedDoc:
    suffix = path.suffix.lower()
    doc_name = path.name
    if suffix == ".txt":
        return LoadedDoc(doc_name=doc_name, text=_read_txt(path), meta={"type": "txt"})
    if suffix == ".docx":
        return LoadedDoc(doc_name=doc_name, text=_read_docx(path), meta={"type": "docx"})
    if suffix == ".pdf":
        text, meta = _read_pdf(path)
        meta = {"type": "pdf", **meta}
        return LoadedDoc(doc_name=doc_name, text=text, meta=meta)
    raise ValueError(f"Unsupported file type: {suffix}")

