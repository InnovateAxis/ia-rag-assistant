# InnovateAxis P7 — Tenant-Isolated RAG Assistant

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
