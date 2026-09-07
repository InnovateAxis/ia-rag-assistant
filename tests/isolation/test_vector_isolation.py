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


# The exact policy set `0002_chunks_rls.sql` is allowed to produce, as
# (policyname, permissive, cmd). Changing this constant is a deliberate change
# to the isolation guarantee and should be reviewed as one.
EXPECTED_CHUNK_POLICIES = {("chunks_tenant_isolation", "PERMISSIVE", "ALL")}


@pytest.mark.isolation
async def test_policy_set_on_document_chunks_is_exactly_as_declared(db):
    """A second permissive policy silently opens everything.

    Postgres ORs permissive policies, so `create policy debug_readall on
    document_chunks using (true)` grants a full cross-tenant read while
    `chunks_tenant_isolation` remains present and correct — and every other
    test in this file still passes. Nothing else asserts the policy *set*,
    so the additive case is invisible without this test.

    Pinning the set, rather than asserting the expected policy is present,
    is the whole point: `>=` would not catch an addition.
    """
    async with db.session(tenant="ten_acme") as s:
        rows = await s.fetch(
            "select policyname, permissive, cmd from pg_policies "
            "where schemaname = 'public' and tablename = 'document_chunks'")
    actual = {(r["policyname"], r["permissive"], r["cmd"]) for r in rows}
    assert actual == EXPECTED_CHUNK_POLICIES, (
        f"policy set on document_chunks changed: "
        f"unexpected {actual - EXPECTED_CHUNK_POLICIES}, "
        f"missing {EXPECTED_CHUNK_POLICIES - actual}"
    )
