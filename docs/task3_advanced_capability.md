## Task 3 — Rapid Iteration Challenge (Advanced capability)

### Capability shown: **Multi-document reasoning**
Implemented in the **Compare** tab and `RAGService.compare_resumes()` (`src/rag/rag.py`).

**What it does**
- Takes **one job description** and **2+ resumes**
- Retrieves evidence **per resume independently**
- Produces a **comparative report** (table + ranking) with **chunk_id citations**

### Why this capability
Resume screening is rarely single-document: recruiters compare multiple candidates, trade off strengths, and want justifications.

### Trade-offs
- **Pros**:
  - More aligned with real workflows (shortlists, ranking)
  - Encourages evidence-first decisions (citations per candidate)
- **Cons**:
  - Higher token usage (multiple contexts)
  - More chances of confusion if prompts aren’t strict (mitigated by grouping context by resume)

### Limitations
- Prototype vector store is SQLite + brute-force similarity; not intended for very large corpora.
- PDF extraction quality can vary; garbage in → garbage out.
- “Fairness” is not guaranteed: comparisons can reflect bias in the resumes or the model.

