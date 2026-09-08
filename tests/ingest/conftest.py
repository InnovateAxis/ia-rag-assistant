"""Fixtures for step 2.1's ingestion tests.

Own database, own container — deliberately not sharing tests/isolation's
fixture. Two reasons:

* Carry-forward F18: that fixture's `documents` stand-in has no RLS. Fine
  for the isolation suite, which never touches `documents` in an assertion,
  but this suite's "ingest of another tenant's document raises not-found"
  test needs `documents` to behave the way it will in production — under
  RLS, mirroring `document_chunks` — or the test would prove nothing.
* This suite's `documents` stand-in is provisioned with **three** tenants
  (carry-forward F21: the real corpus has three, not two), so ingest is
  exercised against the same shape it will see once 1.1's data reaches it.

Real PostgreSQL 16 with pgvector via testcontainers, same image and pattern
as tests/isolation/conftest.py, for the same reason: the guarantee under
test is a database policy, and a fake database would happily confirm one
that does not exist.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator

import asyncpg
import pytest
from testcontainers.community.postgres import PostgresContainer

import src.db.session as db
from src.db.bootstrap_roles import SERVICE_ROLE, service_password
from src.db.migrate import apply_migrations, ensure_service_role

POSTGRES_IMAGE = "pgvector/pgvector:pg16"
EMBED_DIM = 1536

ACME = "ten_acme"
GLOBEX = "ten_globex"
MERIDIAN = "ten_meridian"
TENANTS = (ACME, GLOBEX, MERIDIAN)

# A minimal stand-in for Pod P's `documents` table (carry-forward C7: never a
# migration in this repo). Unlike tests/isolation's stand-in, this one carries
# real RLS — see the module docstring on why that is load-bearing here.
_CREATE_DOCUMENTS = """
create table if not exists documents (
  id           uuid primary key default gen_random_uuid(),
  tenant_id    text not null,
  title        text not null default 'ingest test document',
  storage_key  text not null default 's3://bucket/ingest-test-key',
  content_type text not null default 'text/plain'
);
alter table documents enable row level security;
alter table documents force  row level security;
create policy documents_tenant_isolation on documents
  for all
  using      (tenant_id = current_setting('request.jwt.claims', true)::json->>'tenant_id')
  with check (tenant_id = current_setting('request.jwt.claims', true)::json->>'tenant_id');
"""


async def _provision(admin_dsn: str) -> dict[str, uuid.UUID]:
    """Bring the database to the state this suite assumes, and return the
    one seeded document id per tenant."""
    admin = await asyncpg.connect(admin_dsn)
    try:
        await admin.execute(_CREATE_DOCUMENTS)
        await ensure_service_role(admin)
        await apply_migrations(admin)
        # Ingest only ever reads documents. `delete` is granted alongside
        # `select` for step 2.5's cascade tests, which have to issue the
        # `delete from documents` from a role RLS actually applies to:
        # `ia_rag_service` is NOSUPERUSER/NOBYPASSRLS, so a delete in its
        # session is genuinely scoped by `documents_tenant_isolation` above,
        # while the same statement on this admin connection (a superuser)
        # would bypass every policy and prove nothing about which tenant
        # issued it. Still no `insert`/`update`: this suite's document rows
        # are always created admin-side, standing in for Pod P (C7).
        await admin.execute(f"grant select, delete on documents to {SERVICE_ROLE}")

        doc_ids = {tenant: uuid.uuid4() for tenant in TENANTS}
        await admin.executemany(
            "insert into documents (id, tenant_id) values ($1, $2)",
            [(doc_ids[t], t) for t in TENANTS],
        )
    finally:
        await admin.close()
    return doc_ids


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
        admin = f"postgresql://postgres:postgres@{host}:{port}/postgres"
        service_dsn = (
            f"postgresql://{SERVICE_ROLE}:{service_password()}@{host}:{port}/postgres"
        )
        doc_ids = asyncio.run(_provision(admin))
        yield service_dsn, doc_ids, admin


@pytest.fixture
def doc_ids(_postgres) -> dict[str, uuid.UUID]:
    return _postgres[1]


@pytest.fixture
def admin_dsn(_postgres) -> str:
    """The superuser DSN, for the two things a service-role session cannot
    honestly do in a test: create `documents` rows (Pod P's job, C7) and
    look at `document_chunks` from outside RLS to check that a row is gone
    rather than merely hidden.

    Never used to exercise application behaviour — a superuser connection
    ignores every policy in `0002` and would make any assertion about
    isolation vacuous.
    """
    return _postgres[2]


@pytest.fixture
async def fresh_doc_ids(admin_dsn) -> AsyncGenerator[dict[str, uuid.UUID], None]:
    """One brand-new `documents` row per tenant, created admin-side and
    dropped afterwards.

    Function-scoped and separate from `doc_ids`, which is seeded once for the
    whole session: step 2.5's tests delete document rows, and deleting the
    session-scoped seed would cascade its chunks away underneath every later
    test in this suite. Tests that destroy a document take one of these
    instead.
    """
    conn = await asyncpg.connect(admin_dsn)
    ids = {tenant: uuid.uuid4() for tenant in TENANTS}
    try:
        await conn.executemany(
            "insert into documents (id, tenant_id) values ($1, $2)",
            [(ids[t], t) for t in TENANTS],
        )
        yield ids
    finally:
        # Cascade clears any chunks the test left behind on these documents.
        await conn.execute(
            "delete from documents where id = any($1::uuid[])", list(ids.values())
        )
        await conn.close()


@pytest.fixture
def service_dsn(_postgres) -> str:
    """The `ia_rag_service` DSN — NOSUPERUSER, NOBYPASSRLS, the connection
    every application query in this project uses. Requested directly only by
    the test that checks the offboarding verification refuses to report a
    count over a connection RLS applies to."""
    return _postgres[0]


@pytest.fixture
async def pool(_postgres) -> AsyncGenerator[asyncpg.Pool, None]:
    """Bind `src.db.session.pool` for this test (carry-forward F29: 2.1 is
    the first step that calls `session()`, so this fixture is what actually
    does the binding).

    Function-scoped and created inside the test's own event loop rather than
    at session scope, deliberately: asyncpg pools are bound to the loop that
    creates them, and pytest-asyncio gives each test function its own loop
    (`asyncio_default_fixture_loop_scope = "function"`). A session-scoped
    pool would be created against a loop that is closed by the time later
    tests run. The container underneath (`_postgres`) is still session-scoped
    and only spun up once; only this thin pool binding is per-test.
    """
    service_dsn = _postgres[0]
    bound = await db.create_pool(service_dsn)
    try:
        yield bound
    finally:
        await db.close_pool()
