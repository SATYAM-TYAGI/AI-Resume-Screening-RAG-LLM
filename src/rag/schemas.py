from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    chunk_id: str
    quote: str


OverallFit = Literal["Strong", "Moderate", "Weak", "Unclear"]


class ScreeningResult(BaseModel):
    resume_name: str
    overall_fit: OverallFit
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    strengths: List[str]
    gaps: List[str]
    citations: List[Citation]


class RetrievedChunk(BaseModel):
    doc_name: str
    chunk_id: str
    text: str
    score: float
    page: Optional[int] = None

