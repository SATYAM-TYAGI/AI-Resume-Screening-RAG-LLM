## Task 2 — Hallucination & Quality Control

### Why hallucinations happen in *this* system
- **Retrieval miss / low recall**: the relevant resume text isn’t retrieved (bad chunking, weak embedding match, too-small top‑k), so the LLM “fills the gap.”
- **Noisy extraction**: PDF text extraction can drop tables, columns, or bullets, producing incomplete context.
- **Cross-document contamination**: if multiple resumes are retrieved together without scoping, the LLM may attribute one candidate’s skill to another.
- **Prompt under-specification**: if “use only context” is weak, the model will use general knowledge (e.g., “a DevOps engineer likely knows Kubernetes”).
- **Overconfident decoding**: higher temperature or unconstrained output can increase fluent but wrong text.

### Guardrails implemented (2+)
#### Guardrail 1 — **Confidence threshold gating**
- Implemented in `RAGService._guardrail_context()` (`src/rag/rag.py`)
- If best retrieval score is below `min_relevance_score`, the assistant **refuses** instead of guessing.

#### Guardrail 2 — **Source-grounded prompting + citation requirement**
- Implemented in prompts in `src/rag/prompts.py`
- The assistant is instructed to:
  - answer **only from CONTEXT**
  - refuse when evidence is missing
  - include **citations** (chunk ids) for factual claims

#### Guardrail 3 (extra) — **Document-scoped retrieval for screening**
- Screening uses `where={"doc_name": <resume>}` so evidence is pulled from **only that candidate** (reduces cross-resume leakage).

### Example: improved responses
#### Example A — Before guardrails (typical failure)
User: “Does Candidate A have Kubernetes experience?”

Bad (hallucinated):  
“Yes, Candidate A has strong Kubernetes experience and has deployed clusters in production.”

#### Example A — After guardrails (this prototype)
If retrieval confidence is low, the system responds:
“I don't have enough evidence in the indexed resumes to answer that.

Reason: Low retrieval confidence (best score=0.18 < 0.25).”

If evidence exists, the answer must include citations:
“Candidate A mentions Kubernetes in their skills/experience.  
Evidence: [`CandidateA.pdf:0003`], [`CandidateA.pdf:0004`].”

#### Example B — Cross-resume contamination avoided
User: “Summarize Candidate B’s AWS experience.”

Bad (no scoping): mixes Candidate A’s AWS content into Candidate B.

After guardrail: screening retrieval is scoped per candidate (`doc_name` filter), so Candidate B’s answer can only cite Candidate B chunks; otherwise it will say “not supported.”

