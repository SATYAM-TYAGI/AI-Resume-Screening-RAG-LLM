from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    meta: Dict


_SECTION_RE = re.compile(
    r"^\s*(summary|experience|work experience|projects|skills|education|certifications|certificates|"
    r"publications|achievements|languages|interests)\s*$",
    re.IGNORECASE,
)


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Heuristic section splitter based on common resume headers.
    Returns list of (section_title, section_text).
    """
    lines = [ln.strip() for ln in text.split("\n")]
    sections: List[Tuple[str, List[str]]] = []
    current_title = "BODY"
    current: List[str] = []

    for ln in lines:
        if not ln:
            current.append("")
            continue
        if _SECTION_RE.match(ln):
            # flush prior
            if current:
                sections.append((current_title, current))
            current_title = ln.strip()
            current = []
            continue
        current.append(ln)

    if current:
        sections.append((current_title, current))

    return [(title, "\n".join(body).strip()) for title, body in sections if "\n".join(body).strip()]


def chunk_text(
    text: str,
    *,
    doc_name: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[Chunk]:
    """
    Simple bounded chunker:
    - normalize
    - split by sections, then paragraphs
    - pack into chunks up to max_chars with overlap
    """
    text = normalize_text(text)
    sections = split_into_sections(text)
    if not sections:
        sections = [("BODY", text)]

    chunks: List[Chunk] = []
    seq = 0

    for section_title, section_text in sections:
        paras = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        buf = ""
        for p in paras:
            candidate = (buf + "\n\n" + p).strip() if buf else p
            if len(candidate) <= max_chars:
                buf = candidate
                continue

            # flush buf
            if buf:
                chunk_id = f"{doc_name}:{seq:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=buf,
                        meta={"doc_name": doc_name, "section": section_title, "seq": seq},
                    )
                )
                seq += 1
                # start new with overlap tail from flushed buf
                tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
                buf = (tail + "\n\n" + p).strip()
            else:
                # paragraph itself too big; hard-split
                start = 0
                while start < len(p):
                    piece = p[start : start + max_chars]
                    chunk_id = f"{doc_name}:{seq:04d}"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=piece,
                            meta={"doc_name": doc_name, "section": section_title, "seq": seq},
                        )
                    )
                    seq += 1
                    start += max_chars - overlap_chars if max_chars > overlap_chars else max_chars
                buf = ""

        if buf:
            chunk_id = f"{doc_name}:{seq:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=buf,
                    meta={"doc_name": doc_name, "section": section_title, "seq": seq},
                )
            )
            seq += 1

    return chunks

