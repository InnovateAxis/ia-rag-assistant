# InnovateAxis P7 — Tenant-Isolated RAG Assistant

## Architecture decisions

The decisions a client's security team will want to read, in order.

| ADR | Decision |
|---|---|
| [ADR-0001](docs/adr/0001-pgvector-under-rls.md) | Embeddings live in PostgreSQL under the same row-level security policies as the source documents, not in a dedicated vector database. The database enforces tenant isolation, not application code. |
| [ADR-0002](docs/adr/0002-hybrid-retrieval-justification.md) | Hybrid retrieval (full-text + vector), justified by a measured keyword baseline against this project's own corpus and golden set. |

## Layout

| Path | Purpose |
|---|---|
| `src/api/` | FastAPI routes, SSE streaming |
| `src/auth/` | JWT verification, tenant context binding |
| `src/db/` | connection pool, session context, RLS helpers |
| `src/ingest/` | document intake, chunking, embedding, upsert |
| `src/chunking/` | section-aware splitter |
| `src/retrieval/` | vector, fulltext, fusion, rerank |
| `src/orchestrate/` | query rewrite -> retrieve -> rerank -> generate |
| `src/generate/` | grounded answering, citation spans, refusal |
| `src/llm/` | provider interface, cost accounting |
| `src/telemetry/` | langfuse traces, structured logs |
| `migrations/` | SQL only. RLS policies live here and nowhere else. |
| `tests/isolation/` | the suite that must never be skipped |
| `evals/` | golden.jsonl — 75 question/answer pairs |
| `docs/adr/` | architecture decision records |
