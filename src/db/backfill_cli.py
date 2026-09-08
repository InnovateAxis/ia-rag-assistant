"""src/db/backfill_cli.py — operator entry point for step 2.4's re-embed
backfill, invoked as `python -m src.db.backfill_cli`, the same pattern
`src.db.migrate` and `src.db.bootstrap_roles` already use for infra
operations that need something an application module may not have.

This is deliberately the ONLY place `src.db.tenants.list_tenant_ids`
(infra-role: direct `asyncpg`, admin DSN, outside any tenant's RLS context)
and `src.ingest.backfill.run_backfill` (application-role: tenant-scoped via
`db.session`, one RLS session per tenant) are wired together. Neither module
imports the other:

* `src.ingest.backfill.run_backfill` takes `tenant_ids` as a required
  argument and never discovers them itself.
* `src.db.tenants` has no idea backfill exists.

This module is not one of `pyproject.toml`'s `[tool.importlinter]`
`source_modules` (`src.api`, `src.auth`, `src.chunking`, `src.generate`,
`src.ingest`, `src.llm`, `src.orchestrate`, `src.retrieval`,
`src.telemetry`), so it may import both freely — the same reason
`src/db/migrate.py` may import `asyncpg` directly while application code may
not. Carry-forward F35: this is the shape that keeps the
`forbid-direct-asyncpg-access` contract meaningful rather than merely
passing — see that module's own docstring on why an earlier version of this
seam (importing `list_tenant_ids` straight into `src.ingest.backfill`) would
have been exactly the chain the contract exists to break.

Not invoked by any step's Done-when or by CI — carry-forward F40: no
re-embed runs on this project yet, and running one in the same release as
2.2's chunking change would confound the hit@5 comparison 2.6 owns. Provided
so a human operator has a real `python -m` seam when that day comes, and so
this wiring is demonstrated rather than left implied.
"""

from __future__ import annotations

import asyncio

from src.db.tenants import list_tenant_ids
from src.ingest.backfill import BulkEmbeddingProvider, run_backfill


async def run(embed: BulkEmbeddingProvider) -> dict[str, int]:
    """Enumerate every tenant `document_chunks` holds rows for, then drive
    each one's backfill to completion. Returns `run_backfill`'s per-tenant
    totals."""
    tenant_ids = await list_tenant_ids()
    return await run_backfill(embed, tenant_ids=tenant_ids)


async def main() -> None:  # pragma: no cover - operator entry point
    raise NotImplementedError(
        "wire a real BulkEmbeddingProvider (see src.llm.embeddings) before running "
        "this for real, then call run(embed) with it."
    )


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
