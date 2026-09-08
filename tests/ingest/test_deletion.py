"""Step 2.5 — deletion, updates and orphan prevention.

Four properties, and one limit:

* Deleting a document removes its chunks **in the same transaction** — proved
  by rolling that transaction back and watching the chunks come back, not by
  reading `on delete cascade` off the migration.
* Replacing a document's chunks is atomic: a concurrent reader sees the old
  set or the new set, never both and never neither. A positive control in
  this file runs the same replacement across two transactions and shows the
  same sampler catching it, so "no violation observed" is a result rather
  than an artefact of sampling too slowly — the F17 reasoning applied to a
  probe instead of a suite: a check never shown to fail says nothing when it
  passes.
* Deletion is tenant-scoped by the policy, not by a predicate — deleting
  another tenant's document id is not an error and not a special case; it
  matches nothing.
* Tenant offboarding removes every row this repository holds, verified from
  OUTSIDE row-level security.

The limit is `test_cross_tenant_cascade_removes_another_tenants_chunk`
(carry-forward F5). `docs/design/deletion-contract.md` states both what the
cascade buys and the one edge where it cuts the other way.

None of these carry `@pytest.mark.isolation`: `tests/isolation/` is frozen at
six tests (`MIN_ISOLATION_TESTS = 6`) and this file must not perturb that
count, exactly like `tests/ingest/test_fk_bypasses_rls.py`.
"""

from __future__ import annotations

import asyncio
import uuid

import asyncpg
import pytest

from src.db import offboard_cli
from src.db import session as db
from src.db.offboard_cli import (
    VERIFICATION_DSN_ENV,
    TenantNotEmpty,
    VerificationNotTrustworthy,
    count_tenant_rows,
    offboard_tenant,
    verification_dsn,
)
from src.ingest.deletion import (
    PurgeCounts,
    UnscopedPurgeRefused,
    delete_document_chunks,
    purge_tenant,
)
from src.ingest.pipeline import Chunk, EmbeddedChunk, ingest

ACME = "ten_acme"
GLOBEX = "ten_globex"
EMBED_DIM = 1536

# Big enough that the writing transaction stays open for a measurable stretch,
# so the sampler below gets many chances to catch a half-applied replacement
# if one were possible. The positive control replaces the same number of
# chunks, so the two windows are the same size and comparing them is fair.
REPLACEMENT_CHUNKS = 100

OLD_CONTENTS = tuple(f"old clause {i}" for i in range(REPLACEMENT_CHUNKS))
NEW_CONTENTS = tuple(f"new clause {i}" for i in range(REPLACEMENT_CHUNKS))

_SELECT_CONTENTS = (
    "select content from document_chunks where document_id = $1 order by chunk_index"
)

_COUNT_CHUNKS_FOR_DOCUMENT = (
    "select count(*) from document_chunks where document_id = $1"
)

_INSERT_MIS_STAMPED_CHUNK = (
    "insert into document_chunks "
    "(id, tenant_id, document_id, chunk_index, content, embedding, "
    " embed_model, embed_dim, token_count) "
    "values ($1, $2, $3, 999, 'mis-stamped chunk', $4, "
    "        'characterization-test-only', $5, 1)"
)


class _Rollback(Exception):
    """Raised to abort a transaction on purpose. Never a real failure."""


def _chunker(contents):
    def chunk(doc):
        return [Chunk(index=i, content=c, token_count=1) for i, c in enumerate(contents)]

    return chunk


async def _fake_embed(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk=c,
            embedding=[0.0] * EMBED_DIM,
            embed_model="fake-embed-test-only",
            embed_dim=EMBED_DIM,
        )
        for c in chunks
    ]


async def _seed(tenant: str, document_id: uuid.UUID, contents) -> None:
    """Write `contents` as this tenant's chunks for `document_id`, through the
    real ingest path (2.1) rather than raw SQL."""
    await ingest(document_id, tenant, chunk=_chunker(contents), embed=_fake_embed)


async def _contents(tenant: str, document_id: uuid.UUID) -> tuple[str, ...]:
    async with db.session(tenant) as s:
        rows = await s.fetch(_SELECT_CONTENTS, document_id)
    return tuple(r["content"] for r in rows)


async def _admin_count_chunks(admin_dsn: str, sql: str, arg) -> int:
    """Count from outside RLS.

    Every use of this helper turns on the difference between "the row is
    gone" and "the row is invisible to the session that asked". A
    tenant-scoped count cannot tell those apart, which is the whole reason
    the offboarding verification is an infrastructure operation.
    """
    conn = await asyncpg.connect(admin_dsn)
    try:
        return await conn.fetchval(sql, arg)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# 1. Deleting a document removes its chunks in the same transaction.
# ---------------------------------------------------------------------------


async def test_deleting_a_document_cascades_inside_the_same_transaction(
    pool, fresh_doc_ids
):
    """The contract's first clause, proved as a transaction property rather
    than read off `migrations/0001_chunks.sql:13`.

    Inside one transaction: count the chunks, delete the parent `documents`
    row, count again — they are already gone, so the cascade belongs to this
    transaction and not to a later job. Then roll that transaction back and
    the chunks return. A cascade running in its own transaction, or a nightly
    cleanup sweep, could not come back.

    This is precisely what the two-store design cannot offer: there, the
    document delete and the embedding cleanup are two operations in two
    systems with no shared transaction, and a crash between them leaves a
    deleted document that is still retrievable and still quotable.
    """
    doc = fresh_doc_ids[ACME]
    await _seed(ACME, doc, ("clause one", "clause two"))

    observed: dict[str, int] = {}
    with pytest.raises(_Rollback):
        async with db.session(ACME) as s:
            observed["before"] = await s.fetchval(_COUNT_CHUNKS_FOR_DOCUMENT, doc)
            await s.execute("delete from documents where id = $1", doc)
            observed["after_delete"] = await s.fetchval(_COUNT_CHUNKS_FOR_DOCUMENT, doc)
            raise _Rollback

    assert observed == {"before": 2, "after_delete": 0}
    assert await _contents(ACME, doc) == ("clause one", "clause two"), (
        "the chunks did not come back after the rollback, so the cascade did "
        "not share the deleting transaction"
    )


async def test_committed_document_delete_leaves_no_chunk_behind(
    pool, fresh_doc_ids, admin_dsn
):
    """The same delete, committed: no orphaned embedding survives it. Checked
    from outside RLS as well, so "gone" means gone rather than hidden."""
    doc = fresh_doc_ids[ACME]
    await _seed(ACME, doc, ("clause one", "clause two"))

    async with db.session(ACME) as s:
        await s.execute("delete from documents where id = $1", doc)

    assert await _contents(ACME, doc) == ()
    assert await _admin_count_chunks(admin_dsn, _COUNT_CHUNKS_FOR_DOCUMENT, doc) == 0


# ---------------------------------------------------------------------------
# 2. The limit: carry-forward F5, the cross-tenant cascade.
# ---------------------------------------------------------------------------


async def test_cross_tenant_cascade_removes_another_tenants_chunk(
    pool, fresh_doc_ids, admin_dsn
):
    """Carry-forward F5, made reproducible from the repository instead of
    resting on the group-03 verifier's prose.

    `document_chunks.document_id` carries `on delete cascade`, and foreign
    keys are enforced outside row-level security. So a `documents` row
    deleted in acme's own RLS-scoped session takes with it **every** chunk
    referencing it — including one stamped `ten_globex`, a tenant whose
    session issued nothing, whose policy was never consulted, and whose row
    is simply gone.

    Both halves of this have to be stated, and neither may be dropped.

    * It is a real cross-tenant effect. It is destructive rather than
      disclosing — acme still cannot read globex's chunk, because the policy
      holds on every read — but acme's delete removed it.
    * It is conditional, and the condition is carry-forward F4: it needs a
      chunk stamped with one tenant's id while referencing another tenant's
      document. RLS enforces the VALUE of `tenant_id`, never its
      CORRECTNESS, so the database cannot rule such a row out. That is why
      step 2.1's ingest derives `tenant_id` from the verified session context
      and never from caller-supplied data or from the document row — the
      precondition is structurally hard rather than merely discouraged. The
      mis-stamped row below is planted by raw SQL precisely because no
      application path in this repository will produce one.

    See `docs/design/deletion-contract.md`, where this sits next to the
    runbook claim it qualifies.
    """
    acme_doc = fresh_doc_ids[ACME]
    await _seed(ACME, acme_doc, ("acme clause one", "acme clause two"))

    # F4's shape, planted the way tests/ingest/test_fk_bypasses_rls.py plants
    # it: globex's own session, globex's own tenant_id, acme's document_id.
    # chunk_index 999 keeps it clear of acme's real chunks under the
    # (document_id, chunk_index) unique constraint.
    globex_chunk = uuid.uuid4()
    async with db.session(GLOBEX) as s:
        await s.execute(
            _INSERT_MIS_STAMPED_CHUNK,
            globex_chunk,
            GLOBEX,
            acme_doc,
            [0.0] * EMBED_DIM,
            EMBED_DIM,
        )
        planted = await s.fetchrow(
            "select id from document_chunks where id = $1", globex_chunk
        )
    assert planted is not None, "precondition: globex can see its own chunk"

    # acme deletes its own document, in its own tenant session. ia_rag_service
    # is NOSUPERUSER and NOBYPASSRLS, so this delete is genuinely scoped by
    # `documents_tenant_isolation`: acme reaches for nothing here.
    async with db.session(ACME) as s:
        await s.execute("delete from documents where id = $1", acme_doc)

    async with db.session(GLOBEX) as s:
        survivor = await s.fetchrow(
            "select id from document_chunks where id = $1", globex_chunk
        )
    assert survivor is None, (
        "globex's chunk survived acme's delete — F5 no longer reproduces, "
        "which is a change worth understanding before this test is edited"
    )

    # Not merely invisible to globex: actually deleted. A tenant-scoped read
    # could not distinguish the two, so this one runs outside the policy.
    remaining = await _admin_count_chunks(
        admin_dsn, "select count(*) from document_chunks where id = $1", globex_chunk
    )
    assert remaining == 0, (
        "the row is hidden from globex but still present — that would be a "
        "different finding from F5 and this test should not be reporting it"
    )


# ---------------------------------------------------------------------------
# 3. Updating replaces atomically, with no window where both exist.
# ---------------------------------------------------------------------------


async def _sample_until(
    stop: asyncio.Event, started: asyncio.Event, tenant: str, document_id: uuid.UUID
) -> list[tuple[str, ...]]:
    """Read this document's chunk contents over and over, from a connection
    of its own, until `stop` is set. Signals `started` once the first sample
    is in, so the writer never begins before the old state has been observed.
    """
    samples: list[tuple[str, ...]] = []
    while True:
        samples.append(await _contents(tenant, document_id))
        started.set()
        if stop.is_set():
            return samples
        await asyncio.sleep(0)


async def test_replacing_a_documents_chunks_is_never_half_visible(pool, fresh_doc_ids):
    """The contract's second clause: "old chunks deleted, new chunks
    inserted, one transaction. Never leave both versions — the assistant will
    cite the superseded rate."

    A reader for the same tenant, on its own connection, samples the document
    continuously while the replacement runs. Every sample must be exactly the
    old set or exactly the new set. Not both (the superseded rate is still
    quotable), and not neither (the document silently answers nothing while
    its update is in flight).

    `test_a_two_transaction_replacement_is_caught_by_the_same_sampler` below
    is this test's positive control.
    """
    doc = fresh_doc_ids[ACME]
    await _seed(ACME, doc, OLD_CONTENTS)

    stop, started = asyncio.Event(), asyncio.Event()

    async def replace() -> None:
        await started.wait()
        try:
            await _seed(ACME, doc, NEW_CONTENTS)
        finally:
            stop.set()

    samples, _ = await asyncio.gather(
        _sample_until(stop, started, ACME, doc), replace()
    )

    unexpected = [s for s in samples if s not in (OLD_CONTENTS, NEW_CONTENTS)]
    assert not unexpected, (
        f"{len(unexpected)} of {len(samples)} samples saw a state that was "
        f"neither the old nor the new set; first was {unexpected[0][:4]}... "
        f"({len(unexpected[0])} chunks)"
    )
    assert OLD_CONTENTS in samples, "sampler never observed the pre-update state"
    assert await _contents(ACME, doc) == NEW_CONTENTS


async def test_a_two_transaction_replacement_is_caught_by_the_same_sampler(
    pool, fresh_doc_ids
):
    """Positive control for the test above.

    The same document, the same chunk count, the same sampler — but the
    delete and the insert are committed separately, which is what a
    replacement looks like when it is NOT one transaction (and is exactly the
    shape a two-store design is forced into). The sampler must catch it. If
    it does not, it is sampling too coarsely to have proved anything above,
    and both tests are worthless together rather than one of them silently
    passing on its own.
    """
    doc = fresh_doc_ids[ACME]
    await _seed(ACME, doc, OLD_CONTENTS)

    stop, started = asyncio.Event(), asyncio.Event()

    async def replace_in_two_transactions() -> None:
        await started.wait()
        try:
            await delete_document_chunks(doc, ACME)  # transaction one, commits
            await _seed(ACME, doc, NEW_CONTENTS)  # transaction two
        finally:
            stop.set()

    samples, _ = await asyncio.gather(
        _sample_until(stop, started, ACME, doc), replace_in_two_transactions()
    )

    caught = [s for s in samples if s not in (OLD_CONTENTS, NEW_CONTENTS)]
    assert caught, (
        "the sampler saw only clean states across a deliberately non-atomic "
        "replacement, so it is too coarse to support the atomicity claim in "
        f"the previous test ({len(samples)} samples taken)"
    )


# ---------------------------------------------------------------------------
# 4. Deletion is scoped by the policy, not by a predicate.
# ---------------------------------------------------------------------------


async def test_deleting_another_tenants_document_matches_nothing(
    pool, fresh_doc_ids, admin_dsn
):
    """`delete_document_chunks` writes no tenant predicate (Invariant 1,
    carry-forward F39). It does not need one: asked to delete a document that
    belongs to globex, acme's session matches zero rows, because the policy
    never made those rows visible to delete.

    There is no branch here that could be got wrong — no comparison, no
    ownership check, no error case. That is the entire argument for putting
    the rule in the database.
    """
    globex_doc = fresh_doc_ids[GLOBEX]
    await _seed(GLOBEX, globex_doc, ("globex clause one", "globex clause two"))

    deleted = await delete_document_chunks(globex_doc, ACME)

    assert deleted == 0
    assert await _contents(GLOBEX, globex_doc) == (
        "globex clause one",
        "globex clause two",
    )
    assert (
        await _admin_count_chunks(admin_dsn, _COUNT_CHUNKS_FOR_DOCUMENT, globex_doc) == 2
    )


# ---------------------------------------------------------------------------
# 5. Tenant offboarding, verified by a post-delete count.
# ---------------------------------------------------------------------------


async def _seed_operational_rows(tenant: str) -> None:
    """A backfill-progress and a cutover row (0003), so offboarding has all
    three of this repository's tenant-keyed tables to clear."""
    async with db.session(tenant) as s:
        await s.execute(
            "insert into backfill_progress (tenant_id) values ($1) "
            "on conflict (tenant_id) do nothing",
            tenant,
        )
        await s.execute(
            "insert into embedding_cutover (tenant_id) values ($1) "
            "on conflict (tenant_id) do nothing",
            tenant,
        )


async def test_offboarding_removes_every_row_and_the_count_proves_it(
    pool, fresh_doc_ids, admin_dsn
):
    """The contract's third clause: "every chunk removed; verified by a count
    query, and the verification is logged."

    The purge runs in the offboarded tenant's own RLS session with no WHERE
    clause at all; the verification runs on an admin connection outside the
    policy, because a count taken inside that session returns 0 whether the
    rows are gone or merely invisible. `offboard_tenant` does both and logs
    the result, and raises rather than reporting success if anything survives.
    """
    await _seed(ACME, fresh_doc_ids[ACME], ("acme one", "acme two"))
    await _seed(GLOBEX, fresh_doc_ids[GLOBEX], ("globex one", "globex two"))
    await _seed_operational_rows(ACME)
    await _seed_operational_rows(GLOBEX)

    globex_before = await count_tenant_rows(GLOBEX, admin_dsn)
    assert globex_before.chunks >= 2

    deleted, remaining = await offboard_tenant(ACME, verification_dsn=admin_dsn)

    assert deleted.chunks >= 2
    assert deleted.backfill_progress == 1
    assert deleted.embedding_cutover == 1
    assert remaining.total == 0
    assert (remaining.chunks, remaining.backfill_progress, remaining.embedding_cutover) == (0, 0, 0)

    globex_after = await count_tenant_rows(GLOBEX, admin_dsn)
    assert globex_after == globex_before, (
        "offboarding one tenant changed another tenant's row counts"
    )


async def test_offboarding_verification_refuses_a_connection_rls_applies_to(
    pool, fresh_doc_ids, service_dsn
):
    """Positive control for the verification itself.

    Pointed at `ia_rag_service` — the application role, which RLS filters and
    which has no tenant claim set on a bare connection — the count would
    report zero remaining rows for every tenant, forever, including a tenant
    whose data was never touched. It refuses instead. Without this guard, a
    misconfigured `DATABASE_URL` turns the whole offboarding verification
    into a constant.
    """
    await _seed(ACME, fresh_doc_ids[ACME], ("acme one", "acme two"))

    with pytest.raises(VerificationNotTrustworthy):
        await count_tenant_rows(ACME, service_dsn)

    # And the rows really are still there, as seen from a connection that can
    # see them: the refusal above is not describing an already-empty table.
    assert await _contents(ACME, fresh_doc_ids[ACME]) == ("acme one", "acme two")


async def test_offboarding_reports_failure_when_the_purge_silently_did_nothing(
    pool, fresh_doc_ids, admin_dsn, monkeypatch
):
    """`offboard_tenant` must never tell an operator a tenant is gone when it
    is not — which is the only reason to run a verification count at all.

    The purge is replaced with one that deletes nothing and reports zeroes,
    standing in for every way removal can fail to happen: a statement that
    matched no rows, a table this purge does not know about, a transaction
    that rolled back. The counting is real, over real surviving rows, and it
    has to turn that into a refusal rather than a clean exit.
    """
    await _seed(ACME, fresh_doc_ids[ACME], ("acme one", "acme two"))

    async def purge_nothing(tenant_id: str) -> PurgeCounts:
        return PurgeCounts(chunks=0, backfill_progress=0, embedding_cutover=0)

    monkeypatch.setattr(offboard_cli, "purge_tenant", purge_nothing)

    with pytest.raises(TenantNotEmpty) as caught:
        await offboard_tenant(ACME, verification_dsn=admin_dsn)

    assert caught.value.tenant_id == ACME
    assert caught.value.remaining.chunks >= 2


# ---------------------------------------------------------------------------
# 6. The purge's precondition, and the configuration that can remove it.
#
# Found by ia-verifier at 32a192c, which ran the shipped `python -m` entry
# point against a real database under both plausible values of DATABASE_URL
# and measured that one of them deleted every tenant's chunks and exited 0.
# These are the tests that would have caught it.
# ---------------------------------------------------------------------------


async def test_purge_refuses_to_run_on_a_connection_that_bypasses_rls(
    fresh_doc_ids, service_dsn, admin_dsn
):
    """`purge_tenant`'s statements carry no tenant predicate, so the policy is
    the only thing that makes them mean "this tenant". Bound to a pool that
    bypasses RLS they would mean "every tenant", and the offboarding would
    still report success, because the count for the tenant being removed is
    then truthfully zero.

    Note this test does NOT request the `pool` fixture: it binds the pool to
    the superuser DSN itself, which is the misconfiguration under test.
    """
    # Seed through a correctly-scoped pool first.
    await db.create_pool(service_dsn)
    try:
        await _seed(ACME, fresh_doc_ids[ACME], ("acme one", "acme two"))
        await _seed(GLOBEX, fresh_doc_ids[GLOBEX], ("globex one", "globex two"))
    finally:
        await db.close_pool()

    before = await count_tenant_rows(GLOBEX, admin_dsn)

    await db.create_pool(admin_dsn)  # the misconfiguration
    try:
        with pytest.raises(UnscopedPurgeRefused) as caught:
            await purge_tenant(ACME)
    finally:
        await db.close_pool()

    assert caught.value.tenant_id == ACME
    assert (
        await count_tenant_rows(GLOBEX, admin_dsn) == before
    ), "the refused purge still reached another tenant's rows"
    assert (await count_tenant_rows(ACME, admin_dsn)).chunks >= 2, (
        "the refused purge still deleted the target tenant's rows"
    )


def test_verification_dsn_refuses_to_fall_back_to_the_application_pool(monkeypatch):
    """The verification connection and the application pool must be two
    different roles, so they are two different variables with no fallback.
    Resolving the verification through anything that also reads DATABASE_URL
    makes them the same connection, which is either a meaningless count or an
    unscoped purge — there is no value of one variable that is correct for
    both jobs.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://ia_rag_service@localhost/x")
    monkeypatch.delenv(VERIFICATION_DSN_ENV, raising=False)

    with pytest.raises(VerificationNotTrustworthy, match="is not set"):
        verification_dsn()

    monkeypatch.setenv(VERIFICATION_DSN_ENV, "postgresql://ia_rag_service@localhost/x")
    with pytest.raises(VerificationNotTrustworthy, match="identical to DATABASE_URL"):
        verification_dsn()

    monkeypatch.setenv(VERIFICATION_DSN_ENV, "postgresql://postgres@localhost/x")
    assert verification_dsn() == "postgresql://postgres@localhost/x"
