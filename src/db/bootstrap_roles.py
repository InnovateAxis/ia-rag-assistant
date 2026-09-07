"""Create the service role the application connects as.

Invoked by 0.5's CI as `python -m src.db.bootstrap_roles`, and by
`src.db.migrate` before any migration is applied — `0002_chunks_rls.sql`
grants to this role, so it has to exist first.

The role is created NOSUPERUSER and NOBYPASSRLS, and those two attributes are
re-asserted on every run even when the role already exists. A superuser or a
BYPASSRLS holder silently ignores every policy in `0002`, so a connection with
either attribute would make the whole isolation suite pass while guaranteeing
nothing. `test_service_role_has_no_bypass` asserts this from the other side.
"""

from __future__ import annotations

import asyncio
import os

import asyncpg

SERVICE_ROLE = "ia_rag_service"

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
DEFAULT_SERVICE_PASSWORD = "ia_rag_service_dev"


def service_password() -> str:
    return os.environ.get("IA_RAG_SERVICE_PASSWORD", DEFAULT_SERVICE_PASSWORD)


def admin_dsn() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


async def ensure_service_role(conn: asyncpg.Connection, password: str | None = None) -> None:
    """Create or correct `ia_rag_service`. Idempotent.

    Role attributes are enforced on every call rather than only at creation,
    so a role that was granted BYPASSRLS by hand during an incident is pulled
    back into line the next time migrations run.
    """
    password = password or service_password()

    exists = await conn.fetchval(
        "select true from pg_roles where rolname = $1", SERVICE_ROLE
    )
    if not exists:
        # Identifiers and literals are quoted by Postgres itself rather than by
        # string interpolation here; DDL cannot take bind parameters.
        await conn.execute(
            await conn.fetchval(
                "select format('create role %I login password %L', $1::text, $2::text)",
                SERVICE_ROLE,
                password,
            )
        )

    await conn.execute(
        await conn.fetchval(
            "select format('alter role %I nosuperuser nobypassrls "
            "noinherit login password %L', $1::text, $2::text)",
            SERVICE_ROLE,
            password,
        )
    )

    database = await conn.fetchval("select current_database()")
    await conn.execute(
        await conn.fetchval(
            "select format('grant connect on database %I to %I', $1::text, $2::text)",
            database,
            SERVICE_ROLE,
        )
    )
    await conn.execute(
        await conn.fetchval(
            "select format('grant usage on schema public to %I', $1::text)",
            SERVICE_ROLE,
        )
    )

    row = await conn.fetchrow(
        "select rolsuper, rolbypassrls from pg_roles where rolname = $1", SERVICE_ROLE
    )
    if row is None or row["rolsuper"] or row["rolbypassrls"]:
        raise RuntimeError(
            f"{SERVICE_ROLE} must be NOSUPERUSER and NOBYPASSRLS; "
            f"got rolsuper={row and row['rolsuper']}, "
            f"rolbypassrls={row and row['rolbypassrls']}"
        )


async def main() -> None:
    conn = await asyncpg.connect(admin_dsn())
    try:
        await ensure_service_role(conn)
    finally:
        await conn.close()
    print(f"{SERVICE_ROLE}: present, nosuperuser, nobypassrls")


if __name__ == "__main__":
    asyncio.run(main())
