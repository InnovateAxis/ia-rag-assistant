"""Characterizes carry-forward F4 / B1: foreign-key checks run outside RLS.

This is not a test that the isolation guarantee holds — tests/isolation/
already owns that, is frozen at six tests (MIN_ISOLATION_TESTS = 6), and
this file must not perturb it: it lives outside that directory and carries
no @pytest.mark.isolation.

What this documents, reproducibly: RLS's WITH CHECK on `document_chunks`
only asserts that the row being written has `tenant_id = current tenant`.
It says nothing about which document `document_id` points at, and the
foreign key on that column only asserts the id exists — not who owns it.
So a session can legitimately write a chunk stamped with its own tenant_id
that nonetheless references another tenant's document. The isolation suite
still reports six green, because every one of those six tests asserts
something about *reading* `document_chunks` under RLS, and this row is
correctly tenant-stamped for reads.

This is exactly what carry-forward F4 means by "RLS protects the VALUE of
tenant_id, not its correctness" and what ADR-0001 cites as "demonstrated on
this project, not theorised" (carry-forward B1). `src/ingest/pipeline.py`
never constructs a row this way — it always derives `tenant_id` from the
verified session context, never from a document lookup — so this test
exercises the raw database mechanism the pipeline is deliberately built to
never expose, not the pipeline itself.
"""

from __future__ import annotations

import uuid

from src.db import session as db

EMBED_DIM = 1536


async def test_fk_permits_a_chunk_referencing_another_tenants_document(pool, doc_ids):
    chunk_id = uuid.uuid4()
    # chunk_index is a value tests/ingest/test_pipeline.py's fake chunkers
    # never use (0 and 1 only), so this mis-stamped row never collides with
    # `document_chunks (document_id, chunk_index)`'s unique constraint - it
    # would otherwise be invisible to (and so undeletable by) an acme
    # session's RLS-scoped `delete ... where document_id = $1`, the same way
    # it is invisible to every read in this file's own assertions below.
    async with db.session("ten_globex") as s:
        await s.execute(
            "insert into document_chunks "
            "(id, tenant_id, document_id, chunk_index, content, embedding, "
            " embed_model, embed_dim, token_count) "
            "values ($1, $2, $3, 999, 'mis-stamped chunk', $4, "
            "        'characterization-test-only', $5, 1)",
            chunk_id,
            "ten_globex",
            doc_ids["ten_acme"],  # <- another tenant's document
            [0.0] * EMBED_DIM,
            EMBED_DIM,
        )
        row = await s.fetchrow(
            "select tenant_id, document_id from document_chunks where id = $1",
            chunk_id,
        )

    assert row is not None, (
        "expected the mis-stamped insert to succeed — if it now fails, "
        "something (a new constraint, a trigger) has started enforcing "
        "tenant/document consistency, and this characterization is stale "
        "and should be revisited rather than deleted outright"
    )
    assert row["tenant_id"] == "ten_globex"
    assert row["document_id"] == doc_ids["ten_acme"]
