-- 0003_embedding_v2.sql — step 2.4, the re-embed procedure's schema.
--
-- Adds the second embedding column plus the two operational tables the
-- no-downtime re-embed procedure needs: per-tenant backfill progress
-- (resumable) and a per-tenant cutover flag (instant rollback). Only the
-- schema lands here; the backfill loop and the flag flip are application
-- code in src/ingest/backfill.py, run in a tenant's own RLS session exactly
-- like every other write in this project (Invariant 1) — this file writes
-- no tenant predicate and neither does that code.
--
-- Dimension note (carry-forward F40): there is no second embedding model on
-- this project today. settings.embed_dim is 1536 and document_chunks.embedding
-- is vector(1536). embedding_v2 below is deliberately the SAME dimension,
-- matching settings.embed_dim as it stands — not a new number invented for
-- this file. If a future model with a different dimension is ever adopted,
-- this column's type must be changed to match settings.embed_dim at that
-- time; do not pick a number here that settings does not already have.
--
-- Re-embed/re-chunk sequencing note (carry-forward F40): step 2.2
-- (commit 90d1884, immediately before this one) changed chunking rules.
-- traycer-agents/ia-backend.md's non-negotiable is "never re-embed and
-- change chunking in the same release." That governs when this machinery is
-- RUN, not when it is built — this migration adds schema and no re-embed is
-- executed here. Whoever runs the backfill for real must do it in a release
-- that does not also carry a chunking change, or the eventual hit@5
-- before/after comparison (deferred to 2.6) is confounded by two variables
-- moving at once.

alter table document_chunks
  add column embedding_v2   vector(1536),
  add column embed_model_v2 text,
  add column embed_dim_v2   int;

-- Per-tenant backfill progress. RLS-governed exactly like document_chunks
-- (user-decided, carry-forward F40a) rather than an admin-only operational
-- table: a tenant's own session sees only its own progress row, the same
-- tenancy pattern as everything else in this database, and the cutover flag
-- below has to be readable on the query path anyway.
create table backfill_progress (
  tenant_id      text primary key,
  last_chunk_id  uuid,
  chunks_done    int         not null default 0,
  status         text        not null default 'in_progress'
                 check (status in ('in_progress', 'complete')),
  updated_at     timestamptz not null default now()
);

alter table backfill_progress enable row level security;
alter table backfill_progress force  row level security;

create policy backfill_progress_tenant_isolation on backfill_progress
  for all
  using      (tenant_id = current_setting('request.jwt.claims', true)::json->>'tenant_id')
  with check (tenant_id = current_setting('request.jwt.claims', true)::json->>'tenant_id');

revoke all on backfill_progress from public;
grant select, insert, update, delete on backfill_progress to ia_rag_service;

-- Per-tenant cutover flag. Same policy shape as backfill_progress and
-- document_chunks — same reasoning, not split treatment between the two new
-- tables (carry-forward F40a rejects splitting it).
create table embedding_cutover (
  tenant_id    text primary key,
  active_model text        not null default 'v1'
               check (active_model in ('v1', 'v2')),
  flipped_at   timestamptz,
  updated_at   timestamptz not null default now()
);

alter table embedding_cutover enable row level security;
alter table embedding_cutover force  row level security;

create policy embedding_cutover_tenant_isolation on embedding_cutover
  for all
  using      (tenant_id = current_setting('request.jwt.claims', true)::json->>'tenant_id')
  with check (tenant_id = current_setting('request.jwt.claims', true)::json->>'tenant_id');

revoke all on embedding_cutover from public;
grant select, insert, update, delete on embedding_cutover to ia_rag_service;
