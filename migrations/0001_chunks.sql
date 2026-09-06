-- 0001_chunks.sql — the pgvector chunk store.
--
-- Designed jointly with the P3 backend engineer: these rows live in the same
-- database as the documents they index and inherit the same tenancy model.
-- The isolation guarantee itself is in 0002_chunks_rls.sql; this file only
-- creates the table it protects.

create extension if not exists vector;

create table document_chunks (
  id             uuid primary key default gen_random_uuid(),
  tenant_id      text        not null,
  document_id    uuid        not null references documents(id) on delete cascade,
  chunk_index    int         not null,
  content        text        not null,
  content_tsv    tsvector    generated always as (to_tsvector('english', content)) stored,
  embedding      vector(1536) not null,
  embed_model    text        not null,   -- pinned, per row
  embed_dim      int         not null,
  section_path   text[],                 -- ["Rate Sheet","Zone 3","Surcharges"]
  page_from      int,
  page_to        int,
  token_count    int         not null,
  created_at     timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create index on document_chunks using hnsw (embedding vector_cosine_ops);
create index on document_chunks using gin  (content_tsv);
create index on document_chunks (tenant_id, document_id);
