"""The tenant isolation guarantee, asserted against real PostgreSQL.

Written before the retrieval code these tests will govern. They exist so that no
retrieval path can ever be merged without satisfying them, and a failure here is
never to be resolved by weakening an assertion.

Five of the six tests below are the runbook's own, reproduced exactly. The sixth
pins the policy *set* and is the single authorised addition (carry-forward F2).
"""

import pytest


@pytest.mark.isolation
async def test_similarity_search_cannot_cross_tenants(db, seed):
    """The core guarantee. Tenant A's embedding is a perfect match for
    tenant B's query, and must still be unreachable."""
    async with db.session(tenant="ten_globex") as s:
        rows = await s.fetch(
            "select id, tenant_id from document_chunks "
            "order by embedding <=> $1 limit 20",
            seed.acme_exact_embedding,          # identical vector, other tenant
        )
    assert rows == [], "cross-tenant vector leak"


@pytest.mark.isolation
async def test_direct_id_fetch_cannot_cross_tenants(db, seed):
    async with db.session(tenant="ten_globex") as s:
        row = await s.fetchrow(
            "select * from document_chunks where id = $1", seed.acme_chunk_id)
    assert row is None


@pytest.mark.isolation
async def test_missing_tenant_claim_returns_nothing(db):
    """A bug that drops the claim must fail CLOSED, never open."""
    async with db.session(tenant=None) as s:
        rows = await s.fetch("select id from document_chunks limit 10")
    assert rows == []


@pytest.mark.isolation
async def test_service_role_has_no_bypass(db):
    async with db.session(tenant="ten_acme") as s:
        r = await s.fetchrow(
            "select rolsuper, rolbypassrls from pg_roles where rolname = current_user")
    assert not r["rolsuper"] and not r["rolbypassrls"]


@pytest.mark.isolation
async def test_rls_is_forced_on_every_chunk_table(db):
    async with db.session(tenant="ten_acme") as s:
        rows = await s.fetch(
            "select relname, relrowsecurity, relforcerowsecurity from pg_class "
            "where relname in ('document_chunks','chunk_feedback')")
    for t in rows:
        assert t["relrowsecurity"] and t["relforcerowsecurity"], t["relname"]


# Postgres's own normalised rendering of the predicate in 0002_chunks_rls.sql.
# USING and WITH CHECK are identical there by design: USING governs which rows
# are visible, WITH CHECK which rows may be written, and both must be the tenant
# comparison or a caller could write a row stamped with another tenant's id.
TENANT_PREDICATE = (
    "(tenant_id = ((current_setting('request.jwt.claims'::text, true))"
    "::json ->> 'tenant_id'::text))"
)

# The exact policy set `0002_chunks_rls.sql` is allowed to produce, in
# pg_policies' own column order: (policyname, permissive, roles, cmd, qual,
# with_check). Changing this constant is a deliberate change to the isolation
# guarantee and should be reviewed as one.
#
# roles is ("public",) because 0002 declares the policy with no TO clause, which
# Postgres defaults to PUBLIC. pg_policies.roles is name[], and asyncpg hands it
# back as a list, which cannot live in a set of tuples — hence tuple() below.
EXPECTED_CHUNK_POLICIES = {
    (
        "chunks_tenant_isolation",
        "PERMISSIVE",
        ("public",),
        "ALL",
        TENANT_PREDICATE,
        TENANT_PREDICATE,
    )
}


@pytest.mark.isolation
async def test_policy_set_on_document_chunks_is_exactly_as_declared(db):
    """Pin the policy set, and the rule and reach of every policy in it.

    Four ways to break this table leave every other test in this file green,
    and this test is the only thing that catches any of them.

    ADDED: Postgres ORs permissive policies, so `create policy debug_readall
    on document_chunks using (true)` grants a full cross-tenant read while
    `chunks_tenant_isolation` remains present and correct.

    REMOVED: dropping `chunks_tenant_isolation` leaves five of the six tests
    here passing, because none of the others assert that any policy exists.

    MODIFIED: this is why `qual` and `with_check` are pinned and not just the
    policy's name. `alter policy chunks_tenant_isolation using (tenant_id =
    <claim> or current_setting('app.support_mode', true) = 'on')` keeps the
    name, the permissiveness and the command intact, so a test asserting only
    those three reports success — while any session that sets `app.support_mode`
    reads every tenant's rows. Verified as a live leak, not a hypothetical.

    ROLES: `alter policy chunks_tenant_isolation on document_chunks to
    <unused role>` keeps the name, permissiveness, command and both rules
    intact, so pinning those alone still reports success. No policy then
    applies to the service role, RLS denies every read, and the service can
    see nothing at all — while every other assertion here is trivially
    satisfied, because a table nobody can read leaks nothing. This one fails
    closed rather than open, which is why it is pinned here but is not the
    whole answer: only a positive control proves a tenant can still reach its
    own rows.

    The comparison is `==` rather than `>=` deliberately: a superset check
    catches none of the four.
    """
    async with db.session(tenant="ten_acme") as s:
        rows = await s.fetch(
            "select policyname, permissive, roles, cmd, qual, with_check "
            "from pg_policies "
            "where schemaname = 'public' and tablename = 'document_chunks'")
    actual = {
        (
            r["policyname"],
            r["permissive"],
            tuple(r["roles"]),
            r["cmd"],
            r["qual"],
            r["with_check"],
        )
        for r in rows
    }
    assert actual == EXPECTED_CHUNK_POLICIES, (
        "policy set on document_chunks changed.\n"
        f"  unexpected: {sorted(actual - EXPECTED_CHUNK_POLICIES)}\n"
        f"  missing:    {sorted(EXPECTED_CHUNK_POLICIES - actual)}"
    )
