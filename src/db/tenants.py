"""src/db/tenants.py — infrastructure-role tenant enumeration.

Carry-forward F40: the backfill orchestrator (src/ingest/backfill.py) needs
the list of tenants to backfill, and it needs that list from OUTSIDE any
single tenant's RLS session context — `db.session(tenant_id)` intentionally
scopes every query to one tenant, so there is no way to ask "which tenants
exist" from inside it.

This module is that seam, built the same way `src.db.migrate` and
`src.db.bootstrap_roles` already are: a direct `asyncpg.connect` against
`admin_dsn()`, used only for one-shot infrastructure/orchestration queries,
never as a second path application code reaches for at query time. It lives
in `src.db`, which is deliberately absent from `[tool.importlinter]`'s
`source_modules` — the same infrastructure exception `migrate.py` and
`bootstrap_roles.py` already rely on — so this file may import `asyncpg`
directly.

**No module under `src.api`, `src.auth`, `src.chunking`, `src.generate`,
`src.ingest`, `src.llm`, `src.orchestrate`, `src.retrieval` or
`src.telemetry` may import this file.** `src.ingest.backfill.run_backfill`
takes `tenant_ids` as a required argument precisely so it never needs to —
carry-forward F35: an earlier draft imported `list_tenant_ids` straight into
`src.ingest.backfill`, which is byte-for-byte the
`source_module -> ... -> asyncpg` chain `forbid-direct-asyncpg-access`
exists to break, caught before it shipped. `src/db/backfill_cli.py` is the
one place this module and `run_backfill` are wired together, because that
module is not one of the nine `source_modules` packages either.

This is NOT a `BYPASSRLS` grant (Invariant 4 forbids that) and it is NOT a
second application-side connection pool: `list_tenant_ids` opens one
short-lived connection, reads, and closes it — there is no pool, no
long-lived state, and `src.db.session.pool` is untouched. It is also not
exposed on any query or API path; only `src.db.backfill_cli`'s
`python -m` entry point (an operator-run procedure, not a request handler)
calls it.
"""

from __future__ import annotations

import asyncpg

from src.db.bootstrap_roles import admin_dsn

_LIST_TENANTS = "select distinct tenant_id from document_chunks order by tenant_id"


async def list_tenant_ids(dsn: str | None = None) -> list[str]:
    """Enumerate every tenant `document_chunks` currently holds rows for.

    Connects directly as the admin role (`admin_dsn()`, the same DSN
    `migrate.py`/`bootstrap_roles.py` use for schema setup) rather than
    through `src.db.session`, precisely because a tenant-scoped session
    could never see other tenants' rows to enumerate them in the first
    place — that is RLS working as designed, not an obstacle to route
    around at the data-plane level.
    """
    conn = await asyncpg.connect(dsn or admin_dsn())
    try:
        rows = await conn.fetch(_LIST_TENANTS)
        return [r["tenant_id"] for r in rows]
    finally:
        await conn.close()
