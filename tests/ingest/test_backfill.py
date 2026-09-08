"""Step 2.4 — the re-embed procedure. Against real PostgreSQL (this
directory's `pool`/`doc_ids` fixtures), never against an empty table
(Invariant 5): every test here seeds real chunks for at least two tenants
first, through the same `ingest()` (2.1) this project already ships, then
exercises the backfill/cutover machinery against them.

Covers 2.4's three Done-when items only. Step 5 of the runbook procedure
(hit@5 before/after cutover) is deferred to 2.6 (carry-forward F40) and is
deliberately not touched here — there is no corpus in `document_chunks` at
this point in the runbook to measure against.
"""

from __future__ import annotations

import uuid

import pytest

from src.config import settings
from src.db import session as db
from src.ingest.backfill import (
    BackfillBatchResult,
    BackfillNotCompleteError,
    backfill_tenant,
    cutover_status,
    fetch_active_embedding_metadata,
    flip_to_v2,
    rollback_to_v1,
    run_backfill,
)
from src.ingest.pipeline import Chunk, EmbeddedChunk, ingest
from src.llm.embeddings import ModelMismatchError

EMBED_DIM = 1536
ACME = "ten_acme"
GLOBEX = "ten_globex"
MERIDIAN = "ten_meridian"


async def _reset_backfill_state(tenant_id: str) -> None:
    """`backfill_progress`/`embedding_cutover` are keyed by tenant_id and
    persist across tests in this session-scoped container — unlike chunks,
    which `ingest()`'s delete+insert already resets per test. Deletes here
    carry no `tenant_id` predicate: RLS already scopes each session to at
    most one row per table."""
    async with db.session(tenant_id) as s:
        await s.execute("delete from backfill_progress")
        await s.execute("delete from embedding_cutover")


@pytest.fixture(autouse=True)
async def _clean_backfill_state(pool):
    for tenant in (ACME, GLOBEX, MERIDIAN):
        await _reset_backfill_state(tenant)
    yield


def _n_chunks(n: int) -> list[Chunk]:
    return [Chunk(index=i, content=f"part {i}", token_count=2) for i in range(n)]


async def _fake_embed_v1(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Stamps `settings.embed_model`/`settings.embed_dim` — the same values
    a real 2.1/2.3 ingest would stamp for whatever is currently configured
    as "the" embedding model. Using anything else here would make every v1
    row look stale to `refuse_on_model_mismatch`, which compares against
    `settings` by default, not against whatever an earlier write happened
    to use."""
    return [
        EmbeddedChunk(
            chunk=c,
            embedding=[0.0] * EMBED_DIM,
            embed_model=settings.embed_model,
            embed_dim=settings.embed_dim,
        )
        for c in chunks
    ]


async def _run_backfill_to_completion(
    tenant_id: str, *, batch_size: int = 10
) -> BackfillBatchResult:
    """`backfill_tenant` processes one batch per call and only reports
    `done=True` on the call that finds nothing left — a single call with a
    generous `batch_size` still needs an extra call to observe completion.
    Tests that only care about the end state (not the batching shape) use
    this instead of re-deriving the loop each time."""
    result = await backfill_tenant(tenant_id, embed=_ZeroVectorProvider(), batch_size=batch_size)
    while not result.done:
        result = await backfill_tenant(
            tenant_id, embed=_ZeroVectorProvider(), batch_size=batch_size
        )
    return result


class _ZeroVectorProvider:
    """Fakes `BulkEmbeddingProvider`: texts in, deterministic zero vectors
    out, same order — no network call, matching this project's existing
    fake-embedder pattern (tests/ingest/test_pipeline.py, tests/llm/test_embeddings.py)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.0] * EMBED_DIM for _ in texts]


async def _seed_chunks(document_id: uuid.UUID, tenant_id: str, n: int) -> None:
    await ingest(document_id, tenant_id, chunk=lambda doc: _n_chunks(n), embed=_fake_embed_v1)


async def test_backfill_is_resumable(pool, doc_ids):
    """A crash at 60% resumes at 60%: batching into pieces smaller than the
    total and calling backfill_tenant repeatedly must reach exactly the same
    end state as one large batch, touching every chunk exactly once."""
    await _seed_chunks(doc_ids[ACME], ACME, n=5)

    seen_batches = []
    done = False
    while not done:
        result = await backfill_tenant(ACME, embed=_ZeroVectorProvider(), batch_size=2)
        seen_batches.append(result.processed)
        done = result.done

    # 5 chunks in batches of 2: two full batches, one partial, one empty
    # "done" batch — proving the loop advances and terminates rather than
    # looping forever or re-processing.
    assert seen_batches == [2, 2, 1, 0]

    async with db.session(ACME) as s:
        rows = await s.fetch(
            "select embed_model_v2, embed_dim_v2 from document_chunks "
            "where document_id = $1",
            doc_ids[ACME],
        )
        progress = await s.fetchrow("select status, chunks_done from backfill_progress")

    assert len(rows) == 5
    assert all(r["embed_model_v2"] == settings.embed_model for r in rows)
    assert all(r["embed_dim_v2"] == settings.embed_dim for r in rows)
    assert progress["status"] == "complete"
    assert progress["chunks_done"] == 5

    # Calling again after completion is a genuine no-op, not a re-scan.
    again = await backfill_tenant(ACME, embed=_ZeroVectorProvider(), batch_size=2)
    assert again.processed == 0
    assert again.done is True


async def test_backfill_is_tenant_scoped(pool, doc_ids):
    """Backfilling one tenant must not touch another tenant's chunks or
    progress row — the property Invariant 1 exists to guarantee, checked
    here at the application level rather than only at the RLS-policy level
    tests/isolation already covers."""
    await _seed_chunks(doc_ids[ACME], ACME, n=3)
    await _seed_chunks(doc_ids[GLOBEX], GLOBEX, n=3)

    result = await _run_backfill_to_completion(ACME)
    assert result.done is True

    async with db.session(ACME) as s:
        acme_rows = await s.fetch(
            "select embedding_v2 from document_chunks where document_id = $1",
            doc_ids[ACME],
        )
    assert all(r["embedding_v2"] is not None for r in acme_rows)

    async with db.session(GLOBEX) as s:
        globex_rows = await s.fetch(
            "select embedding_v2 from document_chunks where document_id = $1",
            doc_ids[GLOBEX],
        )
        globex_progress = await s.fetchrow("select status from backfill_progress")

    assert all(r["embedding_v2"] is None for r in globex_rows)
    assert globex_progress is None


async def test_run_backfill_drives_every_named_tenant_to_completion(pool, doc_ids):
    await _seed_chunks(doc_ids[ACME], ACME, n=4)
    await _seed_chunks(doc_ids[GLOBEX], GLOBEX, n=2)

    totals = await run_backfill(
        _ZeroVectorProvider(), tenant_ids=[ACME, GLOBEX], batch_size=3
    )

    assert totals == {ACME: 4, GLOBEX: 2}
    for tenant, doc_id in ((ACME, doc_ids[ACME]), (GLOBEX, doc_ids[GLOBEX])):
        async with db.session(tenant) as s:
            remaining = await s.fetchval(
                "select count(*) from document_chunks "
                "where document_id = $1 and embedding_v2 is null",
                doc_id,
            )
        assert remaining == 0


async def test_cutover_flag_flips_and_rolls_back_instantly(pool, doc_ids):
    await _seed_chunks(doc_ids[ACME], ACME, n=2)

    assert await cutover_status(ACME) == "v1"

    with pytest.raises(BackfillNotCompleteError):
        await flip_to_v2(ACME)
    assert await cutover_status(ACME) == "v1"  # refused flip changed nothing

    await _run_backfill_to_completion(ACME)
    await flip_to_v2(ACME)
    assert await cutover_status(ACME) == "v2"

    await rollback_to_v1(ACME)
    assert await cutover_status(ACME) == "v1"


async def test_cutover_flag_is_per_tenant_not_global(pool, doc_ids):
    await _seed_chunks(doc_ids[ACME], ACME, n=1)
    await _seed_chunks(doc_ids[GLOBEX], GLOBEX, n=1)
    await _run_backfill_to_completion(ACME)
    await _run_backfill_to_completion(GLOBEX)

    await flip_to_v2(ACME)

    assert await cutover_status(ACME) == "v2"
    assert await cutover_status(GLOBEX) == "v1"


async def test_query_refuses_when_a_retrieved_chunk_is_a_stale_embedding_model(pool, doc_ids):
    """The property 2.4's third Done-when item names: a query must never
    mix embedding models. Seeds one chunk whose embed_model_v2 is stale
    relative to settings, flips the tenant to v2, and asserts the query path
    (fetch_active_embedding_metadata) refuses via refuse_on_model_mismatch
    (2.3) rather than silently returning a cross-model result."""
    await _seed_chunks(doc_ids[ACME], ACME, n=1)
    await _run_backfill_to_completion(ACME)

    async with db.session(ACME) as s:
        await s.execute(
            "update document_chunks set embed_model_v2 = 'stale-embed-model' "
            "where document_id = $1",
            doc_ids[ACME],
        )
    await flip_to_v2(ACME)

    with pytest.raises(ModelMismatchError):
        await fetch_active_embedding_metadata(ACME, doc_ids[ACME])


async def test_query_succeeds_when_active_model_is_consistent(pool, doc_ids):
    await _seed_chunks(doc_ids[ACME], ACME, n=2)

    # Still on v1: reads the v1 columns, which ingest() stamped consistently.
    rows = await fetch_active_embedding_metadata(ACME, doc_ids[ACME])
    assert len(rows) == 2
    assert all(r["embed_model"] == settings.embed_model for r in rows)

    await _run_backfill_to_completion(ACME)
    await flip_to_v2(ACME)

    rows_v2 = await fetch_active_embedding_metadata(ACME, doc_ids[ACME])
    assert len(rows_v2) == 2
    assert all(r["embed_model"] == settings.embed_model for r in rows_v2)
