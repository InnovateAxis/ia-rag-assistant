"""src/ingest/backfill.py — Step 2.4, the re-embed procedure.

Builds exactly the three things 2.4's Done-when lists, against
`migrations/0003_embedding_v2.sql`'s schema:

1. A resumable, tenant-scoped backfill loop.
2. A per-tenant cutover flag with instant rollback.
3. A query path that never mixes embedding models, reusing
   `src.llm.embeddings.refuse_on_model_mismatch` (2.3) rather than writing a
   second mismatch check.

Step 5 of the runbook's procedure (compare hit@5 before/after cutover) is
explicitly deferred to 2.6 (carry-forward F40b) — no committed step
populates `document_chunks` from the corpus yet, so there is nothing to
measure against. Nothing here builds that comparison or seeds a corpus.

**Carry-forward F39 — the load-bearing constraint on every query below:**
the obvious reading of "backfill tenant by tenant, in batches, resumable"
writes `where tenant_id = $1` into the backfill statement and resumes on a
`(tenant_id, id)` cursor. Both are Invariant 1 violations — the first
tenant predicate in application SQL on this project — and both are
redundant besides: `db.session(tenant_id)` already scopes every statement
in the transaction through `document_chunks`'s RLS policy. Every query in
this module that touches `document_chunks`, `backfill_progress` or
`embedding_cutover` filters on `id` (the per-row primary key) or nothing at
all, never on `tenant_id`. The tenant argument each function takes selects
*which RLS session to open*, not a WHERE-clause value.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from src.config import settings
from src.db import session as db
from src.db.tenants import list_tenant_ids
from src.llm.embeddings import refuse_on_model_mismatch

DEFAULT_BATCH_SIZE = 500


class BulkEmbeddingProvider(Protocol):
    """Same shape as `src.llm.embeddings.EmbeddingProvider` — texts in,
    vectors out, same order. Not reused by import because that Protocol is
    typed against `2.3`'s batching internals; this one is the raw seam the
    backfill loop needs and nothing more."""

    async def __call__(self, texts: list[str]) -> list[list[float]]: ...


class BackfillNotCompleteError(Exception):
    """Raised by `flip_to_v2` when a tenant's backfill has not reached 100%.
    Flipping early would serve a mix of populated and NULL `embedding_v2`
    rows under the v2 flag — silently mixing embedding spaces, the exact
    failure the runbook procedure calls out as "the worst kind"."""

    def __init__(self, tenant_id: str) -> None:
        super().__init__(f"backfill not complete for tenant {tenant_id!r}")
        self.tenant_id = tenant_id


@dataclass(frozen=True)
class BackfillBatchResult:
    """One batch's worth of work. `done=True` means this tenant has no more
    `embedding_v2 is null` rows left — the caller (or `run_backfill`) stops
    looping and the tenant is ready for `flip_to_v2`."""

    processed: int
    last_id: UUID | None
    done: bool


# ---------------------------------------------------------------------------
# 1. Resumable, tenant-scoped backfill.
# ---------------------------------------------------------------------------

# No tenant_id anywhere: RLS (via db.session(tenant_id)) already restricts
# every one of these rows to the caller's own tenant. `$1` is the resume
# cursor (NULL on a fresh start, the last-seen id otherwise) and it is
# compared against `id` alone, never against a tenant-bearing tuple.
_SELECT_UNBACKFILLED_BATCH = """
select id, content
from document_chunks
where embedding_v2 is null
  and ($1::uuid is null or id > $1::uuid)
order by id
limit $2
"""

_UPDATE_CHUNK_EMBEDDING_V2 = """
update document_chunks
set embedding_v2 = $1, embed_model_v2 = $2, embed_dim_v2 = $3
where id = $4
"""

# backfill_progress's primary key IS tenant_id, but this is an upsert of the
# one row RLS already scopes to the caller — never a WHERE tenant_id = $1.
_UPSERT_PROGRESS = """
insert into backfill_progress (tenant_id, last_chunk_id, chunks_done, status, updated_at)
values ($1, $2, $3, 'in_progress', now())
on conflict (tenant_id) do update
set last_chunk_id = excluded.last_chunk_id,
    chunks_done   = backfill_progress.chunks_done + excluded.chunks_done,
    status        = 'in_progress',
    updated_at    = now()
"""

_MARK_PROGRESS_COMPLETE = """
insert into backfill_progress (tenant_id, last_chunk_id, chunks_done, status, updated_at)
values ($1, $2, 0, 'complete', now())
on conflict (tenant_id) do update
set status     = 'complete',
    updated_at = now()
"""

# No WHERE clause at all: RLS already leaves at most one row visible — the
# caller's own tenant's progress row, if one exists yet.
_SELECT_PROGRESS = "select last_chunk_id, status from backfill_progress"


async def backfill_tenant(
    tenant_id: str,
    *,
    embed: BulkEmbeddingProvider,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> BackfillBatchResult:
    """Process one batch of `tenant_id`'s un-backfilled chunks and record
    progress. Crash-safe by construction: progress is written in the same
    transaction as the embeddings it describes (one `db.session` block is
    one transaction), so a crash mid-batch leaves the ledger and the data it
    describes consistent with each other — never a completed write with no
    recorded progress, or vice versa. Calling this again with the same
    `tenant_id` resumes from `last_chunk_id`, never redoing prior batches
    and never revisiting a row across tenants.
    """
    async with db.session(tenant_id) as s:
        progress = await s.fetchrow(_SELECT_PROGRESS)
        cursor: UUID | None = progress["last_chunk_id"] if progress else None
        if progress is not None and progress["status"] == "complete":
            return BackfillBatchResult(processed=0, last_id=cursor, done=True)

        rows = await s.fetch(_SELECT_UNBACKFILLED_BATCH, cursor, batch_size)
        if not rows:
            await s.execute(_MARK_PROGRESS_COMPLETE, tenant_id, cursor)
            return BackfillBatchResult(processed=0, last_id=cursor, done=True)

        texts = [r["content"] for r in rows]
        vectors = await embed(texts)
        for row, vector in zip(rows, vectors, strict=True):
            await s.execute(
                _UPDATE_CHUNK_EMBEDDING_V2,
                vector,
                settings.embed_model,
                settings.embed_dim,
                row["id"],
            )

        new_cursor: UUID = rows[-1]["id"]
        await s.execute(_UPSERT_PROGRESS, tenant_id, new_cursor, len(rows))
        return BackfillBatchResult(processed=len(rows), last_id=new_cursor, done=False)


async def run_backfill(
    embed: BulkEmbeddingProvider,
    *,
    tenant_ids: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int]:
    """Drive every tenant's backfill to completion, tenant by tenant, one
    batch at a time. `tenant_ids` defaults to `src.db.tenants.list_tenant_ids`
    — the infra-role seam that enumerates tenants from outside any single
    tenant's RLS context (carry-forward F40). Passing an explicit list is
    what tests use, so a test never depends on that infra connection.

    Returns the count of chunks actually backfilled per tenant during this
    run — 0 for a tenant that was already complete.
    """
    if tenant_ids is None:
        tenant_ids = await list_tenant_ids()

    totals: dict[str, int] = {}
    for tenant_id in tenant_ids:
        total = 0
        done = False
        while not done:
            result = await backfill_tenant(tenant_id, embed=embed, batch_size=batch_size)
            total += result.processed
            done = result.done
        totals[tenant_id] = total
    return totals


# ---------------------------------------------------------------------------
# 2. Per-tenant cutover flag, instant rollback.
# ---------------------------------------------------------------------------

_SELECT_CUTOVER = "select active_model from embedding_cutover"

_UPSERT_CUTOVER = """
insert into embedding_cutover (tenant_id, active_model, flipped_at, updated_at)
values ($1, $2, now(), now())
on conflict (tenant_id) do update
set active_model = excluded.active_model,
    flipped_at   = now(),
    updated_at   = now()
"""


async def cutover_status(tenant_id: str) -> str:
    """`'v1'` or `'v2'`. Defaults to `'v1'` when no row exists yet — a
    tenant is on v1 until explicitly flipped, matching the migration's
    column default."""
    async with db.session(tenant_id) as s:
        row = await s.fetchrow(_SELECT_CUTOVER)
    return row["active_model"] if row is not None else "v1"


async def flip_to_v2(tenant_id: str) -> None:
    """Runbook procedure step 4: "When a tenant reaches 100%, flip that
    tenant to v2 via a feature flag. Per-tenant flip, not global." Refuses
    to flip a tenant whose backfill has not reached `'complete'` — see
    `BackfillNotCompleteError`.
    """
    async with db.session(tenant_id) as s:
        progress = await s.fetchrow(_SELECT_PROGRESS)
        if progress is None or progress["status"] != "complete":
            raise BackfillNotCompleteError(tenant_id)
        await s.execute(_UPSERT_CUTOVER, tenant_id, "v2")


async def rollback_to_v1(tenant_id: str) -> None:
    """Runbook procedure step 5: "flip that tenant back — the flag makes
    this instant." One row write, no data touched, no re-embed: this is the
    entire rollback. Never raises on an already-v1 tenant — rollback is
    idempotent."""
    async with db.session(tenant_id) as s:
        await s.execute(_UPSERT_CUTOVER, tenant_id, "v1")


# ---------------------------------------------------------------------------
# 3. A query that never mixes embedding models.
# ---------------------------------------------------------------------------

# Column aliases line up so refuse_on_model_mismatch (2.3) can read `id`,
# `embed_model`, `embed_dim` off either shape without caring which one it got.
_SELECT_ACTIVE_COLUMNS = {
    "v1": "id, embed_model as embed_model, embed_dim as embed_dim",
    "v2": "id, embed_model_v2 as embed_model, embed_dim_v2 as embed_dim",
}


async def fetch_active_embedding_metadata(tenant_id: str, document_id: UUID) -> list[object]:
    """The query-path half of "never query across two embedding spaces":
    reads whichever column pair (`embed_model`/`embed_dim` or
    `embed_model_v2`/`embed_dim_v2`) matches this tenant's current cutover
    flag, then runs the result through `refuse_on_model_mismatch` (2.3, not
    rebuilt here) before returning anything. `document_id` scopes it further,
    still with no `tenant_id` predicate — RLS already restricted the rows to
    this tenant.

    Not the vector-similarity query itself (3.1 builds that, using the
    already-real `refuse_on_model_mismatch` this function also calls) — this
    is the seam 2.4's third Done-when item needs to have a real query path
    to test against, so 3.1 wires this pattern in rather than re-deriving
    "which column pair" from scratch.
    """
    status = await cutover_status(tenant_id)
    columns = _SELECT_ACTIVE_COLUMNS[status]
    async with db.session(tenant_id) as s:
        rows = await s.fetch(
            f"select {columns} from document_chunks where document_id = $1",
            document_id,
        )
    refuse_on_model_mismatch(rows)
    return list(rows)


async def _cli_main() -> None:  # pragma: no cover - operator entry point, not exercised by tests
    """Not invoked by any step's Done-when or by CI. Documented here so a
    human operator running the real backfill later has a starting point,
    and so the F40 sequencing note (never re-embed and re-chunk in the same
    release) has somewhere concrete to be read before this runs for real."""
    raise NotImplementedError(
        "wire a real BulkEmbeddingProvider (see src.llm.embeddings) before running "
        "this for real; run_backfill()/flip_to_v2() are the entry points."
    )


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_cli_main())
