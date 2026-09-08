"""src/ingest/deletion.py — Step 2.5, the chunk-lifecycle write path.

Owns every statement in this project that REMOVES chunk rows, so there is
exactly one place the deletion rules are written down and exactly one place
to audit them:

1. `delete_chunks_for_document` — remove one document's chunks. Composable
   inside a caller's transaction; `src.ingest.pipeline.ingest` calls it for
   the delete half of its atomic replace, so the update path and the delete
   path are the same statement rather than two that can drift apart.
2. `delete_document_chunks` — the same thing with its own transaction, for
   a caller that is only deleting.
3. `purge_tenant` — tenant offboarding, application half.

**Carry-forward F39 — the load-bearing constraint on every statement below.**
Deletion here is tenant-scoped, and the obvious implementation of that
sentence writes

    delete from document_chunks where tenant_id = $1 and document_id = $2

which is Invariant 1 violated, and redundant besides: `db.session(tenant_id)`
has already set the request claim for the life of the transaction, so
`chunks_tenant_isolation` (migrations/0002_chunks_rls.sql) scopes every
statement in it. A delete cannot reach another tenant's row because the
policy's USING clause never makes that row visible to delete in the first
place. The predicates below are `document_id` — a per-row foreign key — or
nothing at all. The `tenant_id` argument each function takes selects *which
RLS session to open*, never a WHERE-clause value. Same rule and same shape
as `src/ingest/backfill.py`.

The lifecycle contract these functions implement, and the one limit it has
(carry-forward F5, the cross-tenant cascade), are written out in
`docs/design/deletion-contract.md`. Read that before changing anything here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from src.db import session as db


class _Connection(Protocol):
    """The one method these statements need from an asyncpg connection.

    Structural, not an `asyncpg.Connection` annotation: `src.ingest` is one
    of `pyproject.toml`'s `[tool.importlinter]` `source_modules` and may not
    import `asyncpg`, even for a type (carry-forward F35). `db.session`
    yields the real connection; this Protocol only describes what is used.
    """

    async def execute(self, query: str, *args: object) -> str: ...


@dataclass(frozen=True)
class PurgeCounts:
    """Rows removed from each table this repository owns, for one tenant."""

    chunks: int
    backfill_progress: int
    embedding_cutover: int

    @property
    def total(self) -> int:
        return self.chunks + self.backfill_progress + self.embedding_cutover


# One document's chunks. `document_id` alone — see the module docstring.
_DELETE_CHUNKS_FOR_DOCUMENT = "delete from document_chunks where document_id = $1"

# Tenant offboarding. No WHERE clause at all, deliberately: inside
# `db.session(tenant_id)` the policy has already restricted the visible rows
# to that tenant's, so "delete everything I can see" IS "delete everything
# belonging to this tenant". Writing `where tenant_id = $1` here would add
# nothing except a second, unenforced copy of the rule the policy owns.
_PURGE_CHUNKS = "delete from document_chunks"
_PURGE_BACKFILL_PROGRESS = "delete from backfill_progress"
_PURGE_EMBEDDING_CUTOVER = "delete from embedding_cutover"


def _rows_affected(status: str) -> int:
    """asyncpg returns the command tag, e.g. `"DELETE 12"`."""
    return int(status.rsplit(" ", 1)[-1])


async def delete_chunks_for_document(conn: _Connection, document_id: UUID) -> int:
    """Remove every chunk of `document_id` **inside the caller's already-open
    transaction and tenant session**, and return how many rows went.

    Takes a connection rather than opening its own session precisely so it
    can be the delete half of an atomic replace: `src.ingest.pipeline.ingest`
    calls this and then inserts the new chunks on the same connection, so a
    reader on any other connection sees the old set or the new set and never
    a mixture of the two. See `tests/ingest/test_deletion.py`.
    """
    return _rows_affected(await conn.execute(_DELETE_CHUNKS_FOR_DOCUMENT, document_id))


async def delete_document_chunks(document_id: UUID, tenant_id: str) -> int:
    """Remove every chunk of `document_id`, in `tenant_id`'s own RLS session.

    For a caller that is deleting and not replacing. A document belonging to
    another tenant is not an error and not a special case: the policy makes
    its chunks invisible to this session, the delete matches nothing, and the
    return value is 0. There is no code path here that could behave
    differently, because there is no tenant comparison here to get wrong.

    Note what this does NOT do: it does not delete the `documents` row.
    `documents` belongs to Pod P (carry-forward C7) and this repository has
    no migration for it and no business writing it. When P deletes a
    document, `document_chunks`'s `on delete cascade` removes these rows in
    P's own transaction without this function being called at all — that is
    the contract's first clause, and `docs/design/deletion-contract.md`
    records both what that buys and the one edge where it cuts the other way.
    """
    async with db.session(tenant_id) as s:
        return await delete_chunks_for_document(s, document_id)


async def purge_tenant(tenant_id: str) -> PurgeCounts:
    """Tenant offboarding, application half: remove every row this repository
    holds for `tenant_id`, in that tenant's own RLS session, in ONE
    transaction.

    Three tables, all of them RLS-governed with the same policy shape:
    `document_chunks` (0001/0002) plus `backfill_progress` and
    `embedding_cutover` (0003, carry-forward F40a). The contract's wording is
    "every chunk removed", and chunks are the reason offboarding matters —
    but the two operational tables are keyed by `tenant_id` and leaving an
    offboarded tenant's rows in them would leave that tenant's identifier in
    this database after we said it was gone.

    `documents` is not touched: Pod P owns that table and its own offboarding
    (carry-forward C7).

    The returned counts are what this session deleted. They are NOT the
    verification the contract asks for — a count taken inside the very RLS
    session that did the deleting cannot distinguish "the rows are gone" from
    "the rows were never visible to me". That verification has to be taken
    from outside the policy, and it lives in `src/db/offboard_cli.py`, which
    is infrastructure and reached as a `python -m` entry point.
    """
    async with db.session(tenant_id) as s:
        chunks = _rows_affected(await s.execute(_PURGE_CHUNKS))
        progress = _rows_affected(await s.execute(_PURGE_BACKFILL_PROGRESS))
        cutover = _rows_affected(await s.execute(_PURGE_EMBEDDING_CUTOVER))
    return PurgeCounts(
        chunks=chunks, backfill_progress=progress, embedding_cutover=cutover
    )
