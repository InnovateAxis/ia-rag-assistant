# Step 2.6 — index tuning and ingestion throughput, measured on this project

Produced by `scripts/measure_index_tuning_2.6.py`. Raw numbers: `evals/index_tuning_2.6_results.json`. Real PostgreSQL 16 + pgvector (`pgvector/pgvector:pg16`), started by testcontainers — never a mock.

## Seed

`document_chunks` seeded to **250,000** rows via `src.db.seed_scale_cli` (synthetic uniform-random vectors, `embed_model = 'synthetic-2.6-seed'`), SKEWED per CARRYFORWARD F46: `ten_scale_small` holds **111** rows (the 111 real-corpus chunk count), `ten_scale_large` holds the remaining **249,889**. An evenly split seed would hide the effect below.

## Cold start

The latency of the first query issued against each freshly built index, reported on its own — never folded into any mean or median below, and no cause attributed beyond what is shown here:

| m | ef_construction | ef_search | first query (ms) | plan (scan / index) | indexes present |
|---|---|---|---|---|---|
| 16 | 64 | 40 | 104.3 | Index Scan / document_chunks_embedding_hnsw_sweep | document_chunks_content_tsv_idx, document_chunks_document_id_chunk_index_key, document_chunks_embedding_hnsw_sweep, document_chunks_pkey, document_chunks_tenant_id_document_id_idx |
| 32 | 64 | 60 | 75.4 | Index Scan / document_chunks_embedding_hnsw_sweep | document_chunks_content_tsv_idx, document_chunks_document_id_chunk_index_key, document_chunks_embedding_hnsw_sweep, document_chunks_pkey, document_chunks_tenant_id_document_id_idx |

## Parameter table

`hit@5` is marked not measurable, for two independent reasons stated in full in this script's module docstring and in CARRYFORWARD F45/F46: (1) no live embedding model in this environment, so 250k synthetic vectors carry no semantics; (2) 2.2 already measured `hit@5` at 100% on the real 111-document corpus (`evals/chunking_impact_2.2.md`), so the metric is saturated and cannot discriminate `ef_search` values regardless.

Every row below is measured twice, because a latency number is only attributable to `m`/`ef_search` if the plan that served it is known: **natural** is whatever plan the planner picks with the full, unmodified index set (nothing forced); **forced** constrains the planner onto the HNSW index by the same method `_f3_probe` uses (`enable_seqscan = off` plus dropping the competing `(tenant_id, document_id)` btree index for the duration of the forced measurements only, restored immediately after). Both report the plan and the index set actually present, rather than assuming either. `n` is the sample count kept AFTER discarding the first 30 samples of that configuration as warm-up.

### Natural plan (full index set, nothing forced)

| m | ef_search | ef_construction | hit@5 | plan (scan / index) | n | median (ms) | p95 (ms) | mean (ms) | min (ms) | max (ms) | index build (s) | indexes present |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 40 | 64 | not measurable — see note above | Index Scan / document_chunks_embedding_hnsw_sweep | 300 | 6.6 | 9.8 | 6.8 | 4.3 | 12.7 | 1304.7 | document_chunks_content_tsv_idx, document_chunks_document_id_chunk_index_key, document_chunks_embedding_hnsw_sweep, document_chunks_pkey, document_chunks_tenant_id_document_id_idx |
| 16 | 60 | 64 | not measurable — see note above | Index Scan / document_chunks_embedding_hnsw_sweep | 300 | 7.3 | 9.9 | 7.5 | 5.0 | 15.3 | 1304.7 | document_chunks_content_tsv_idx, document_chunks_document_id_chunk_index_key, document_chunks_embedding_hnsw_sweep, document_chunks_pkey, document_chunks_tenant_id_document_id_idx |
| 16 | 100 | 64 | not measurable — see note above | Index Scan / document_chunks_embedding_hnsw_sweep | 300 | 10.8 | 15.4 | 11.3 | 8.1 | 40.4 | 1304.7 | document_chunks_content_tsv_idx, document_chunks_document_id_chunk_index_key, document_chunks_embedding_hnsw_sweep, document_chunks_pkey, document_chunks_tenant_id_document_id_idx |
| 32 | 60 | 64 | not measurable — see note above | Index Scan / document_chunks_embedding_hnsw_sweep | 300 | 15.2 | 25.3 | 16.3 | 10.8 | 57.0 | 4649.4 | document_chunks_content_tsv_idx, document_chunks_document_id_chunk_index_key, document_chunks_embedding_hnsw_sweep, document_chunks_pkey, document_chunks_tenant_id_document_id_idx |

### Forced onto the HNSW index

| m | ef_search | ef_construction | hit@5 | plan (scan / index) | n | median (ms) | p95 (ms) | mean (ms) | min (ms) | max (ms) | indexes present |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 40 | 64 | not measurable — see note above | Index Scan / document_chunks_embedding_hnsw_sweep | 300 | 5.9 | 9.0 | 6.2 | 4.0 | 14.3 | document_chunks_content_tsv_idx, document_chunks_document_id_chunk_index_key, document_chunks_embedding_hnsw_sweep, document_chunks_pkey |
| 16 | 60 | 64 | not measurable — see note above | Index Scan / document_chunks_embedding_hnsw_sweep | 300 | 7.0 | 9.2 | 7.3 | 5.3 | 25.5 | document_chunks_content_tsv_idx, document_chunks_document_id_chunk_index_key, document_chunks_embedding_hnsw_sweep, document_chunks_pkey |
| 16 | 100 | 64 | not measurable — see note above | Index Scan / document_chunks_embedding_hnsw_sweep | 300 | 11.0 | 14.6 | 11.3 | 8.0 | 33.2 | document_chunks_content_tsv_idx, document_chunks_document_id_chunk_index_key, document_chunks_embedding_hnsw_sweep, document_chunks_pkey |
| 32 | 60 | 64 | not measurable — see note above | Index Scan / document_chunks_embedding_hnsw_sweep | 300 | 13.7 | 18.1 | 14.1 | 9.9 | 37.0 | document_chunks_content_tsv_idx, document_chunks_document_id_chunk_index_key, document_chunks_embedding_hnsw_sweep, document_chunks_pkey |

**The runbook's illustrative grid (78.1–80.4% hit@5) is from a different corpus and is not this project's number — not restated here as measured.**

No row's mean exceeded its p95 this run — see the raw JSON for the full min/max range of every row.

## Ingestion throughput

**1,609 chunks/sec** (250,000 chunks in 155.4s), measured as batched `INSERT` (`asyncpg.executemany`, batch size 2000) into `document_chunks` via `src.db.seed_scale_cli.seed_scale`. This measures the storage layer's raw write throughput, not `src.ingest.pipeline.ingest`'s per-document transaction (2.1) — that path is not rebuilt here (CARRYFORWARD F44) and its per-document chunk count is far smaller than what a throughput number at this scale needs to be useful.

## F3 — the real deliverable of this step

CARRYFORWARD F3: RLS is applied as a post-index filter, so a tenant holding a small share of a large table can get zero rows back from the vector index. Measured here at 250k scale with `ten_scale_small` holding 111 of 250,000 rows, index `m=16, ef_construction=64` (this step's chosen configuration). Forcing the planner onto the vector index took two things, not one: `set enable_seqscan = off` alone was measured to be insufficient — the planner used migrations/0001's `(tenant_id, document_id)` btree index instead (CARRYFORWARD B2's predicted shape), an equality lookup that is cheap for a 111-row tenant regardless of the vector index. That btree index was dropped for this probe only and restored immediately after — the parameter table above measures this same drop/restore independently, per row, as its own "forced" column set. One `order by embedding <=> $1 limit 5` query per `ef_search` value below, same tenant and query vector throughout so only `ef_search` varies; `plan_uses_hnsw_index` (raw results) confirms the HNSW index actually ran each time rather than assuming the GUC setting worked:

| ef_search | scan node | index used | rows returned (of 5 requested) |
|---|---|---|---|
| 40 | Index Scan | document_chunks_embedding_hnsw_sweep | 0 |
| 60 | Index Scan | document_chunks_embedding_hnsw_sweep | 0 |
| 100 | Index Scan | document_chunks_embedding_hnsw_sweep | 0 |

**Recall did NOT recover as `ef_search` rose** — rows returned stayed flat across the whole sweep, with the HNSW index confirmed in use at every `ef_search` value (`plan_uses_hnsw_index` true throughout), confirming CARRYFORWARD F3 and ADR-0001's claim at this project's own 250k/111-row scale.

Reported as measured, not assumed — the table above stands whether it confirms or refutes F3's original claim, per this step's brief.

