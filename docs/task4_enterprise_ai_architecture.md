## Task 4 — Enterprise AI System Architecture (Internal assistant)

### High-level architecture diagram (Mermaid)

```mermaid
flowchart LR
  subgraph Sources[Enterprise Data Sources]
    A1[Confluence / Wiki]
    A2[Google Drive / SharePoint]
    A3[Jira / Ticketing]
    A4[Git repos]
    A5[Databases / Data warehouse]
  end

  subgraph Ingestion[Data Ingestion & Processing]
    B1[Connectors + permissions]
    B2[ETL: clean + normalize]
    B3[Chunking + metadata]
    B4[PII/Secrets redaction]
    B5[Embeddings service]
  end

  subgraph Storage[Storage Layer]
    C1[(Raw doc store\nS3/Blob + versioning)]
    C2[(Metadata store\nPostgres)]
    C3[(Vector DB\nPinecone / Weaviate / pgvector)]
  end

  subgraph Orchestration[LLM Orchestration Layer]
    D1[API Gateway + AuthN/Z]
    D2[Policy engine\n(allowed tools, data, scopes)]
    D3[Retriever (RAG)]
    D4[Prompt builder + templates]
    D5[LLM Router\n(model tiering)]
    D6[Tool calling\n(search, tickets, DB, actions)]
  end

  subgraph Observability[Monitoring & Evaluation]
    E1[Logging + tracing]
    E2[Cost metrics + quotas]
    E3[Quality evals\n(golden sets)]
    E4[Safety monitors\n(PII leaks, jailbreaks)]
    E5[Feedback loop\n(user ratings)]
  end

  Sources --> B1 --> B2 --> B3 --> B4 --> B5
  B2 --> C1
  B3 --> C2
  B5 --> C3

  User[Employees\nWeb/Chat UI] --> D1 --> D2 --> D3
  D3 --> C3
  D3 --> C1
  D3 --> D4 --> D5 --> LLM[LLM Providers\n(OpenAI/Anthropic/Local)]
  D5 --> D6

  D1 --> Observability
  D5 --> Observability
  D6 --> Observability
  Ingestion --> Observability
```

### Explanation (required components)
#### Data ingestion
- **Connectors** pull content from approved systems (wiki, drives, Jira, Git, DBs) using service accounts.
- Enforce **ACLs at ingestion** (store doc-level permissions and group membership), and again at retrieval time.
- Normalize to a canonical document format and preserve **version history**.

#### Vector DB choice
Typical options:
- **pgvector (Postgres)**: simple ops, strong governance; good for moderate scale.
- **Pinecone/Weaviate**: managed scaling, hybrid search, filtering; higher cost but faster time-to-prod.

Criteria:
- ACL-aware filtering, metadata filtering, latency SLOs, hybrid BM25+vector support, operational burden.

#### LLM orchestration
- Central “orchestrator” performs:
  - query understanding / routing
  - retrieval (RAG)
  - prompt construction with policy constraints
  - tool calling with allowlisted tools and audited actions
- Use **strict prompting + schemas** for structured outputs.

#### Cost control
- **Model tiering** (cheap model for intent + retrieval, stronger model for final synthesis)
- **Caching** (embedding cache, retrieval cache, response cache for repeated queries)
- **Budgets/quotas** per team/user, rate limits, and “max tokens per response”
- **Batching** embeddings and retrieval when processing multiple docs

#### Monitoring & evaluation
- **Tracing**: capture prompt, retrieved doc IDs, tool calls, latency.
- **Quality**: offline eval sets (“golden Q/A”) + online monitors (refusal rate, citation coverage).
- **Safety**: jailbreak detection, PII leakage detectors, policy violations.
- **Drift**: monitor embedding/LLM changes; re-cknowledge content freshness and re-index jobs.

