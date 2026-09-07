"""Fixtures for the tenant isolation suite.

Runs against real PostgreSQL 16 with pgvector, started by testcontainers from
the same `pgvector/pgvector:pg16` image 0.5's CI uses. Never a mock and never
sqlite: the guarantee under test is a PostgreSQL row-level security policy, and
a fake database would happily confirm a policy that does not exist.

Two properties of this file are load-bearing:

* **Both tenants are seeded for the whole session**, in the container fixture
  rather than in `seed`, so that no test can run against an empty table even if
  it never requests the seed fixture. An empty table passes every isolation test
  ever written (CLAUDE.md invariant 5).
* **The connection is `ia_rag_service`**, which is NOSUPERUSER and NOBYPASSRLS.
  Connecting as the owner or a superuser would bypass every policy in `0002`
  and make the entire suite vacuous.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import asyncpg
import pytest
from testcontainers.community.postgres import PostgresContainer

from src.db.bootstrap_roles import SERVICE_ROLE, service_password
from src.db.migrate import apply_migrations, ensure_service_role

# The realistic failure is not a deleted policy - it is a refactor that renames
# the marker, the suite collects zero tests, and CI stays green for two months.
#
# Six, not five: the runbook's five plus the policy-set test (carry-forward F2).
# Left at five, deleting any one test would still satisfy the guard, which is
# exactly the silent shrink this count exists to catch. Raise this whenever a
# test is added; never lower it to make a run pass.
MIN_ISOLATION_TESTS = 6

POSTGRES_IMAGE = "pgvector/pgvector:pg16"
EMBED_DIM = 1536
ACME = "ten_acme"
GLOBEX = "ten_globex"


def pytest_collection_modifyitems(config, items):
    """Guard against the isolation suite silently vanishing.

    A skipped isolation test is indistinguishable from a passing one
    in a CI summary. Refuse to run at all rather than run empty.
    """
    iso = [i for i in items if "isolation" in i.keywords]
    if len(iso) < MIN_ISOLATION_TESTS:
        pytest.exit(f"isolation suite has {len(iso)} tests, "
                    f"expected at least {MIN_ISOLATION_TESTS}", returncode=2)


def _vector(seed: int, lo: int, hi: int) -> list[float]:
    """A deterministic positive vector supported only on [lo, hi).

    Acme's vectors and Globex's vectors occupy disjoint halves of the space, so
    their cosine distance is exactly 1.0 while a repeat of Acme's own vector is
    exactly 0.0. That makes "the cross-tenant row would have been the top hit"
    a fact about the data rather than a hope about the random seed.
    """
    rng = random.Random(seed)
    v = [0.0] * EMBED_DIM
    for i in range(lo, hi):
        v[i] = rng.uniform(0.1, 1.0)
    return v


def _acme_vector(seed: int) -> list[float]:
    return _vector(seed, 0, EMBED_DIM // 2)


def _globex_vector(seed: int) -> list[float]:
    return _vector(seed, EMBED_DIM // 2, EMBED_DIM)


@dataclass
class Seed:
    """Handles onto the seeded rows, for tests that reference them by name."""

    acme_chunk_id: uuid.UUID
    acme_document_id: uuid.UUID
    globex_document_id: uuid.UUID
    acme_exact_embedding: list[float] = field(repr=False, default_factory=list)


async def _register_vector_codec(conn: asyncpg.Connection) -> None:
    """asyncpg has no built-in codec for pgvector's `vector` type."""
    await conn.set_type_codec(
        "vector",
        encoder=lambda v: "[" + ",".join(repr(float(x)) for x in v) + "]",
        decoder=lambda s: [float(x) for x in s.strip("[]").split(",")] if s else [],
        format="text",
    )


async def _connect(dsn: str) -> asyncpg.Connection:
    conn = await asyncpg.connect(dsn)
    await _register_vector_codec(conn)
    return conn


CREATE_DOCUMENTS = """
create table if not exists documents (
  id        uuid primary key default gen_random_uuid(),
  tenant_id text not null,
  title     text not null default 'seed document'
);
"""

INSERT_CHUNK = """
insert into document_chunks
  (id, tenant_id, document_id, chunk_index, content,
   embedding, embed_model, embed_dim, token_count)
values ($1, $2, $3, $4, $5, $6, 'text-embedding-3-large', 1536, 12)
"""


async def _provision(admin_dsn: str, service_dsn: str) -> Seed:
    """Bring the database to the state every isolation test assumes."""
    admin = await asyncpg.connect(admin_dsn)
    try:
        # `documents` first: 0001's foreign key needs it. It is a stand-in for
        # Pod P's table, which lives in another repository. This must never
        # become a migration in this repo - that would claim another pod's table.
        await admin.execute(CREATE_DOCUMENTS)
        await ensure_service_role(admin)
        await apply_migrations(admin)
        await admin.execute(f"grant select, insert on documents to {SERVICE_ROLE}")

        acme_doc = uuid.uuid4()
        globex_doc = uuid.uuid4()
        await admin.executemany(
            "insert into documents (id, tenant_id) values ($1, $2)",
            [(acme_doc, ACME), (globex_doc, GLOBEX)],
        )
    finally:
        await admin.close()

    acme_chunk_id = uuid.uuid4()
    acme_exact = _acme_vector(1)

    # Seeded through the service role with the tenant claim set, so the rows are
    # written the way the application will write them. This exercises the
    # policy's WITH CHECK on the way in; a policy that rejected legitimate
    # in-tenant writes would fail here rather than silently seed nothing.
    svc = await _connect(service_dsn)
    try:
        batches = (
            (
                ACME,
                acme_doc,
                [(acme_chunk_id, 0, "acme rate sheet zone 3", acme_exact)]
                + [
                    (uuid.uuid4(), i, f"acme chunk {i}", _acme_vector(10 + i))
                    for i in (1, 2)
                ],
            ),
            (
                GLOBEX,
                globex_doc,
                [
                    (uuid.uuid4(), i, f"globex chunk {i}", _globex_vector(100 + i))
                    for i in (0, 1, 2)
                ],
            ),
        )
        for tenant, doc_id, rows in batches:
            async with svc.transaction():
                await svc.execute(
                    "select set_config('request.jwt.claims', $1, true)",
                    json.dumps({"tenant_id": tenant}),
                )
                for chunk_id, idx, content, embedding in rows:
                    await svc.execute(
                        INSERT_CHUNK, chunk_id, tenant, doc_id, idx, content, embedding
                    )
    finally:
        await svc.close()

    # Invariant 5, enforced rather than assumed. If either tenant seeded zero
    # rows, every test below would pass while proving nothing at all.
    admin = await asyncpg.connect(admin_dsn)
    try:
        counts = {
            r["tenant_id"]: r["n"]
            for r in await admin.fetch(
                "select tenant_id, count(*) as n from document_chunks group by tenant_id"
            )
        }
    finally:
        await admin.close()
    if counts.get(ACME, 0) < 1 or counts.get(GLOBEX, 0) < 1:
        raise RuntimeError(
            "refusing to run the isolation suite against a table that is not "
            f"seeded for two tenants; got {counts}"
        )

    return Seed(
        acme_chunk_id=acme_chunk_id,
        acme_document_id=acme_doc,
        globex_document_id=globex_doc,
        acme_exact_embedding=acme_exact,
    )


class Database:
    """The only handle the tests get on the database.

    Mirrors the shape `src/db/session.py` will take at 0.6: a transaction per
    session with the tenant claim set `is_local`, so the claim cannot outlive
    the transaction that set it.
    """

    def __init__(self, dsn: str, seed: Seed) -> None:
        self.dsn = dsn
        self.seed = seed

    @asynccontextmanager
    async def session(self, tenant: str | None = None):
        conn = await _connect(self.dsn)
        try:
            async with conn.transaction():
                if tenant is not None:
                    # is_local=true scopes the claim to this transaction. Set it
                    # session-wide instead and a pooled connection carries one
                    # tenant's context into the next request. Phase 6 tests that.
                    await conn.execute(
                        "select set_config('request.jwt.claims', $1, true)",
                        json.dumps({"tenant_id": tenant}),
                    )
                # tenant is None: the setting is left UNSET, never '' or NULL.
                # Postgres coerces both of those to an empty string, and
                # ''::json raises - which would turn "fails closed" into
                # "raises" and hide the property under test (carry-forward C5).
                yield conn
        finally:
            await conn.close()


@pytest.fixture(scope="session")
def _postgres():
    """Real PostgreSQL with pgvector. Started once for the whole suite."""
    with PostgresContainer(
        POSTGRES_IMAGE,
        username="postgres",
        password="postgres",
        dbname="postgres",
        driver=None,
    ) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        admin_dsn = f"postgresql://postgres:postgres@{host}:{port}/postgres"
        service_dsn = (
            f"postgresql://{SERVICE_ROLE}:{service_password()}@{host}:{port}/postgres"
        )
        seed = asyncio.run(_provision(admin_dsn, service_dsn))
        yield service_dsn, seed


@pytest.fixture
def db(_postgres) -> Database:
    service_dsn, seed = _postgres
    return Database(service_dsn, seed)


@pytest.fixture
def seed(_postgres) -> Seed:
    return _postgres[1]
