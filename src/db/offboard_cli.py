"""src/db/offboard_cli.py — step 2.5's tenant-offboarding entry point,
invoked as `python -m src.db.offboard_cli <tenant_id>`, the same pattern
`src.db.migrate`, `src.db.bootstrap_roles` and `src.db.backfill_cli`
already use for operations that need something an application module may
not have.

Offboarding is two halves and they are deliberately not the same kind of
operation:

* **Removal is application-side and inside RLS.** `src.ingest.deletion.
  purge_tenant` runs `delete from document_chunks` — no WHERE clause — in
  the offboarded tenant's own `db.session`, so the policy scopes it. Nothing
  here can reach another tenant's rows because nothing here compares
  `tenant_id` to anything.
* **Verification is infrastructure-side and outside RLS.** The contract says
  offboarding is "verified by a count query, and the verification is
  logged". A count taken inside the tenant's own session is not a
  verification: it returns 0 whether the rows are gone or merely invisible,
  which is the one distinction the whole exercise turns on. So the count
  below runs on an admin connection that RLS does not filter — and because
  a count that silently became RLS-scoped would pass forever while proving
  nothing, `_assert_sees_past_rls` refuses to report a number at all unless
  the connecting role genuinely bypasses the policy.

That count is the ONE tenant predicate in this project's SQL, and it is not
application SQL: this module is not one of `pyproject.toml`'s
`[tool.importlinter]` `source_modules` (`src.api`, `src.auth`,
`src.chunking`, `src.generate`, `src.ingest`, `src.llm`, `src.orchestrate`,
`src.retrieval`, `src.telemetry`), and **no module under any of those nine
packages may import it** — the same seam rule `src/db/tenants.py` and
`src/db/backfill_cli.py` carry (carry-forward F35/F40). It is reachable only
from an operator's `python -m`, never from a request path. Invariant 1
governs the query path, where a hand-written tenant filter is a promise
about code discipline standing in for the policy; here the point is to look
at what the policy would hide, which is exactly why it cannot live there.

Invariant 4 is untouched: no second application pool (removal goes through
`src.db.session`'s one pool), no `BYPASSRLS` grant to `ia_rag_service`, and
the verification is a single short-lived connection — opened, read, closed.

**The two connections are configured separately and MUST NOT be the same
one.** The pool comes from `DATABASE_URL` and has to be the service role,
because the purge statements carry no tenant predicate and the policy is the
only thing scoping them. The verification comes from
`IA_RAG_VERIFICATION_DSN` and has to be a role the policy does not apply to,
or its count is zero for every tenant regardless. An earlier version of this
module resolved the verification through `bootstrap_roles.admin_dsn`, which
also reads `DATABASE_URL` — so the two were necessarily the same connection,
and following this module's own advice to point that variable at the admin
role ran the no-WHERE purge on a bypassing connection, deleting every
tenant's chunks and exiting 0 with a truthful zero for the offboarded
tenant. `verification_dsn()` now refuses to fall back to `DATABASE_URL` or
to equal it, and `src.ingest.deletion.purge_tenant` independently refuses to
run at all on a connection that bypasses RLS. Either check alone would have
stopped it; both are here because the failure is silent and total.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass

import asyncpg

from src.db import session as db
from src.ingest.deletion import PurgeCounts, purge_tenant

logger = logging.getLogger(__name__)

# The verification connection gets its OWN environment variable, and there is
# deliberately no fallback to DATABASE_URL.
#
# This is not tidiness. The purge runs on `src.db.session`'s pool, which reads
# DATABASE_URL and MUST be the service role, because the purge statements have
# no tenant predicate and RLS is the only thing scoping them. The verification
# must be the opposite: a role RLS does not apply to, or its count reads zero
# for every tenant whether or not the purge worked. `bootstrap_roles.admin_dsn`
# also reads DATABASE_URL, so resolving the verification through it made the
# two connections the same one: pointed at the service role the verification
# refused, and pointed at the admin role — which an earlier version of this
# module actually instructed the operator to do — the no-WHERE purge ran on a
# bypassing connection and deleted every tenant's chunks while reporting the
# offboarded tenant's remaining count as a truthful zero.
#
# Two different requirements on one variable cannot both be satisfied, so they
# get two variables, and this one refuses to fall back or to alias.
VERIFICATION_DSN_ENV = "IA_RAG_VERIFICATION_DSN"

# The deliberate tenant predicate. See the module docstring; these run on an
# admin connection outside RLS, which is the only place a count of "rows this
# tenant still has" can mean anything.
_COUNT_CHUNKS = "select count(*) from document_chunks where tenant_id = $1"
_COUNT_BACKFILL_PROGRESS = "select count(*) from backfill_progress where tenant_id = $1"
_COUNT_EMBEDDING_CUTOVER = "select count(*) from embedding_cutover where tenant_id = $1"

_BYPASSES_RLS = """
select rolsuper or rolbypassrls from pg_roles where rolname = current_user
"""


class VerificationNotTrustworthy(Exception):
    """Raised when the verification connection is itself subject to RLS, or
    when it has not been configured separately from the application pool.

    Such a connection reports zero remaining rows for every tenant, always,
    including one whose data was never deleted. Reporting that number would
    be worse than reporting nothing.
    """


def verification_dsn() -> str:
    """The DSN for the count, from `IA_RAG_VERIFICATION_DSN`.

    Refuses if it is unset, and refuses if it is the same string as
    `DATABASE_URL`. The second check is the one that matters: a verification
    connection equal to the application pool's is either RLS-scoped (the
    count means nothing) or RLS-bypassing (the purge means far too much).
    Neither is a configuration this entry point may run under, and the two
    being equal is precisely the mistake that is easy to make and invisible
    afterwards.
    """
    dsn = os.environ.get(VERIFICATION_DSN_ENV)
    if not dsn:
        raise VerificationNotTrustworthy(
            f"{VERIFICATION_DSN_ENV} is not set. Offboarding needs two distinct "
            "connections: DATABASE_URL (the service role, which RLS scopes, used "
            f"for the purge) and {VERIFICATION_DSN_ENV} (a role RLS does not "
            "apply to, used only to count what survived). There is no fallback "
            "here on purpose — see this module's docstring."
        )
    if dsn == os.environ.get("DATABASE_URL"):
        raise VerificationNotTrustworthy(
            f"{VERIFICATION_DSN_ENV} is identical to DATABASE_URL. One of the two "
            "roles must be wrong: the purge needs a connection row-level security "
            "applies to, and the count needs one it does not. If they are the same "
            "connection, either the count is meaningless or the purge is unscoped."
        )
    return dsn


class TenantNotEmpty(Exception):
    """Raised when rows for the tenant survive the purge. Offboarding has
    not happened; the operator must not be told that it has."""

    def __init__(self, tenant_id: str, remaining: TenantRowCounts) -> None:
        super().__init__(
            f"tenant {tenant_id!r} still has rows after purge: {remaining}"
        )
        self.tenant_id = tenant_id
        self.remaining = remaining


@dataclass(frozen=True)
class TenantRowCounts:
    """What is left for a tenant, seen from outside the policy."""

    chunks: int
    backfill_progress: int
    embedding_cutover: int

    @property
    def total(self) -> int:
        return self.chunks + self.backfill_progress + self.embedding_cutover


async def _assert_sees_past_rls(conn: asyncpg.Connection) -> None:
    if not await conn.fetchval(_BYPASSES_RLS):
        raise VerificationNotTrustworthy(
            f"the verification connection is role {await conn.fetchval('select current_user')!r}, "
            "which row-level security applies to. Every count it returns would be "
            "filtered by the same policy the count exists to check, so it would report "
            f"0 remaining rows for any tenant whether or not the purge worked. Point "
            f"{VERIFICATION_DSN_ENV} at a role that bypasses RLS. Do NOT point "
            "DATABASE_URL there instead: the purge runs on that pool and its "
            "statements carry no tenant predicate, so on a bypassing connection they "
            "would delete every tenant's rows."
        )


async def count_tenant_rows(tenant_id: str, dsn: str | None = None) -> TenantRowCounts:
    """Count every row this repository still holds for `tenant_id`, over a
    connection RLS does not filter — `IA_RAG_VERIFICATION_DSN` unless `dsn`
    is passed explicitly (tests do; the entry point does not).

    Refuses (`VerificationNotTrustworthy`) rather than returning a number if
    that DSN is unset, is the same as `DATABASE_URL`, or turns out to connect
    as a role the policy applies to — see the module docstring.
    """
    conn = await asyncpg.connect(dsn or verification_dsn())
    try:
        await _assert_sees_past_rls(conn)
        return TenantRowCounts(
            chunks=await conn.fetchval(_COUNT_CHUNKS, tenant_id),
            backfill_progress=await conn.fetchval(_COUNT_BACKFILL_PROGRESS, tenant_id),
            embedding_cutover=await conn.fetchval(_COUNT_EMBEDDING_CUTOVER, tenant_id),
        )
    finally:
        await conn.close()


async def offboard_tenant(
    tenant_id: str, *, verification_dsn: str | None = None
) -> tuple[PurgeCounts, TenantRowCounts]:
    """Purge `tenant_id` inside its own RLS session, then verify from outside
    it and log the verification. Raises `TenantNotEmpty` if anything survived.

    Returns what was deleted and what the verification found, so a caller
    that wants to record both has them.
    """
    deleted = await purge_tenant(tenant_id)
    remaining = await count_tenant_rows(tenant_id, verification_dsn)
    logger.info(
        "tenant offboarding verified: tenant_id=%s deleted_chunks=%d "
        "deleted_backfill_progress=%d deleted_embedding_cutover=%d "
        "remaining_chunks=%d remaining_backfill_progress=%d "
        "remaining_embedding_cutover=%d verified_outside_rls=true",
        tenant_id,
        deleted.chunks,
        deleted.backfill_progress,
        deleted.embedding_cutover,
        remaining.chunks,
        remaining.backfill_progress,
        remaining.embedding_cutover,
    )
    if remaining.total:
        raise TenantNotEmpty(tenant_id, remaining)
    return deleted, remaining


async def main(argv: list[str]) -> int:  # pragma: no cover - operator entry point
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(argv) != 1:
        print("usage: python -m src.db.offboard_cli <tenant_id>", file=sys.stderr)
        return 2
    await db.create_pool()
    try:
        await offboard_tenant(argv[0])
    finally:
        await db.close_pool()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
