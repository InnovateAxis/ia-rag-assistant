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

-- Fails closed by construction, by one of two paths. If the claim was never
-- set, current_setting(..., true) returns NULL, the comparison is NULL rather
-- than true, and no row is returned. If the claim was set but is not valid
-- JSON -- which includes the empty string, and note that set_config(..., NULL)
-- stores an empty string rather than a SQL NULL -- the ::json cast raises
-- instead. A claim that is valid JSON but carries no usable tenant_id (key
-- absent, null, or empty) returns no rows by the first path.
--
-- So a dropped claim either returns zero rows or errors, depending on how it
-- was dropped. It never returns another tenant's rows. Callers and tests must
-- expect both outcomes: clear the claim by leaving it unset or by RESET, not
-- by assigning '' or NULL, or a test asserting "no rows" will see an
-- exception instead.
--
-- USING governs which rows are visible to select/update/delete; WITH CHECK
-- governs the rows insert/update may write. Both are required: without
-- WITH CHECK a caller could write a row stamped with another tenant's id.

-- The service role used by this application is NOT a superuser and
-- does NOT hold BYPASSRLS. Verified by a test, because a superuser
-- connection silently ignores every policy above.
revoke all on document_chunks from public;
grant select, insert, update, delete on document_chunks to ia_rag_service;
