# Step 2.6 — index tuning and ingestion throughput, measured on this project

Produced by `scripts/measure_index_tuning_2.6.py`. Raw numbers: `evals/index_tuning_2.6_results.json`. Real PostgreSQL 16 + pgvector (`pgvector/pgvector:pg16`), started by testcontainers — never a mock.

## Seed

`document_chunks` seeded to **250,000** rows via `src.db.seed_scale_cli` (synthetic uniform-random vectors, `embed_model = 'synthetic-2.6-seed'`), SKEWED per CARRYFORWARD F46: `ten_scale_small` holds **111** rows (the 111 real-corpus chunk count), `ten_scale_large` holds the remaining **249,889**. An evenly split seed would hide the effect below.

## Parameter table

`hit@5` is marked not measurable, for two independent reasons stated in full in this script's module docstring and in CARRYFORWARD F45/F46: (1) no live embedding model in this environment, so 250k synthetic vectors carry no semantics; (2) 2.2 already measured `hit@5` at 100% on the real 111-document corpus (`evals/chunking_impact_2.2.md`), so the metric is saturated and cannot discriminate `ef_search` values regardless. p95 latency and ingestion throughput are genuine measurements, taken on this run, at this scale.

| m | ef_search | ef_construction | hit@5 | p95 (ms) | mean (ms) | index build (s) |
|---|---|---|---|---|---|---|
| 16 | 40 | 64 | not measurable on synthetic vectors — see note above | 12.4 | 8.5 | 1420.3 |
| 16 | 60 | 64 | not measurable on synthetic vectors — see note above | 10.2 | 7.8 | 1420.3 |
| 16 | 100 | 64 | not measurable on synthetic vectors — see note above | 15.8 | 11.8 | 1420.3 |
| 32 | 60 | 64 | not measurable on synthetic vectors — see note above | 36.4 | 60.2 | 4694.4 |

**The runbook's illustrative grid (78.1–80.4% hit@5) is from a different corpus and is not this project's number — not restated here as measured.**

**Mean above p95 on the m=32 ef_search=60 row** — a real measured artifact, not an error: `min_ms`/`max_ms` in the raw JSON show a single large outlier per flagged row (e.g. one query far slower than the other 299 — plausible on a memory-constrained host under this container's concurrent index-build load) pulling the mean above the 95th percentile, which by construction excludes that one slowest sample. Reported as measured rather than smoothed away.

## Ingestion throughput

**1,476 chunks/sec** (250,000 chunks in 169.4s), measured as batched `INSERT` (`asyncpg.executemany`, batch size 2000) into `document_chunks` via `src.db.seed_scale_cli.seed_scale`. This measures the storage layer's raw write throughput, not `src.ingest.pipeline.ingest`'s per-document transaction (2.1) — that path is not rebuilt here (CARRYFORWARD F44) and its per-document chunk count is far smaller than what a throughput number at this scale needs to be useful.

## F3 — the real deliverable of this step

CARRYFORWARD F3: RLS is applied as a post-index filter, so a tenant holding a small share of a large table can get zero rows back from the vector index. Measured here at 250k scale with `ten_scale_small` holding 111 of 250,000 rows, index `m=16, ef_construction=64` (this step's chosen configuration). Forcing the planner onto the vector index took two things, not one: `set enable_seqscan = off` alone was measured to be insufficient — the planner used migrations/0001's `(tenant_id, document_id)` btree index instead (CARRYFORWARD B2's predicted shape), an equality lookup that is cheap for a 111-row tenant regardless of the vector index. That btree index was dropped for this probe only — after the m=16 latency rows above (measured with the full, realistic index set) and restored before the m=32 row below — so the HNSW index was the only one available. One `order by embedding <=> $1 limit 5` query per `ef_search` value below, same tenant and query vector throughout so only `ef_search` varies; `plan_uses_hnsw_index` (raw results) confirms the HNSW index actually ran each time rather than assuming the GUC setting worked:

| ef_search | scan node | index used | rows returned (of 5 requested) |
|---|---|---|---|
| 40 | Index Scan | document_chunks_embedding_hnsw_sweep | 0 |
| 60 | Index Scan | document_chunks_embedding_hnsw_sweep | 0 |
| 100 | Index Scan | document_chunks_embedding_hnsw_sweep | 0 |

**Recall did NOT recover as `ef_search` rose** — rows returned stayed flat across the whole sweep, with the HNSW index confirmed in use at every `ef_search` value (`plan_uses_hnsw_index` true throughout), confirming CARRYFORWARD F3 and ADR-0001's claim at this project's own 250k/111-row scale.

Reported as measured, not assumed — the table above stands whether it confirms or refutes F3's original claim, per this step's brief.

