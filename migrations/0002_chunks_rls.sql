-- 0002_chunks_rls.sql — the guarantee itself.
--
-- This file is the whole tenant-isolation story for document_chunks. It is
-- deliberately plain SQL, in this repository, readable end to end by a client's
-- security reviewer without running anything. No ORM, no generator, no
-- abstraction layer: the policy below is the only thing standing between two
-- customers' documents, and it should be possible to audit it by reading it.
--
-- The application NEVER writes a tenant predicate of its own. Every query is
-- filtered here, by the database, whether or not the caller remembered to.

alter table document_chunks enable row level security;
alter table document_chunks force  row level security;   -- applies to the owner too

create policy chunks_tenant_isolation on document_chunks
  for all
  using      (tenant_id = current_setting('request.jwt.claims', true)::json->>'tenant_id')
  with check (tenant_id = current_setting('request.jwt.claims', true)::json->>'tenant_id');

-- Fails closed by construction. With no claim set, current_setting(..., true)
-- returns NULL, the comparison is NULL rather than true, and the row is not
-- returned. A dropped claim yields zero rows, never another tenant's rows.
--
-- USING governs which rows are visible to select/update/delete; WITH CHECK
-- governs the rows insert/update may write. Both are required: without
-- WITH CHECK a caller could write a row stamped with another tenant's id.

-- The service role used by this application is NOT a superuser and
-- does NOT hold BYPASSRLS. Verified by a test, because a superuser
-- connection silently ignores every policy above.
revoke all on document_chunks from public;
grant select, insert, update, delete on document_chunks to ia_rag_service;
