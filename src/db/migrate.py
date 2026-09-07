"""Apply the SQL files in `migrations/`, in filename order.

Invoked as `python -m src.db.migrate` by 0.5's CI and by R.1. Plain SQL applied
in order, no ORM and no migration framework: `0002_chunks_rls.sql` is the whole
tenant-isolation story and a client's security reviewer has to be able to audit
it by reading the file, not by reasoning about a generator.

Two orderings are load-bearing here:

* `ia_rag_service` is created before any file is applied. `0002` ends with a
  grant to that role, so applying migrations first — as 0.5's CI reads top to
  bottom — aborts on a fresh database with `role "ia_rag_service" does not
  exist`. Ensuring the role here makes the CI order work as written.
* `documents` must already exist. `0001` declares
  `references documents(id)`, but that table belongs to Pod P and lives in a
  different repository; this repo deliberately does not create it. Its absence
  is reported as a named precondition below rather than as a foreign-key error
  from the middle of a migration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from src.db.bootstrap_roles import admin_dsn, ensure_service_role

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

# Bookkeeping owned by this module, not a migration. The runbook's SQL files stay
# byte-identical; nothing in migrations/ knows this table exists.
LEDGER_DDL = """
create table if not exists schema_migrations (
  filename   text primary key,
  applied_at timestamptz not null default now()
);
"""


class MissingPreconditionError(RuntimeError):
    """A table this repo does not own is absent from the target database."""


def migration_files(directory: Path | None = None) -> list[Path]:
    directory = directory or MIGRATIONS_DIR
    return sorted(directory.glob("*.sql"))


async def assert_documents_table_exists(conn: asyncpg.Connection) -> None:
    exists = await conn.fetchval("select to_regclass('public.documents') is not null")
    if not exists:
        raise MissingPreconditionError(
            "Table 'documents' does not exist. migrations/0001_chunks.sql declares "
            "document_id ... references documents(id), but 'documents' is owned by "
            "Pod P in a different repository and is deliberately not created here. "
            "Provision it in the target database before running migrations. The "
            "isolation suite creates a minimal stand-in in its own fixture."
        )


async def applied_migrations(conn: asyncpg.Connection) -> set[str]:
    await conn.execute(LEDGER_DDL)
    return {r["filename"] for r in await conn.fetch("select filename from schema_migrations")}


async def apply_migrations(conn: asyncpg.Connection, directory: Path | None = None) -> list[Path]:
    """Apply every migration not yet recorded in the ledger.

    Re-running is a no-op rather than an error. Without the ledger a second run
    aborts on `create table document_chunks`, and — because a failed run leaves
    whatever it had already applied in place — a partially applied database can
    never be brought forward by re-running.
    """
    already = await applied_migrations(conn)
    applied: list[Path] = []
    for path in migration_files(directory):
        if path.name in already:
            continue
        sql = path.read_text(encoding="utf-8")
        # One transaction per file, ledger row included: a migration and the
        # record of it land together or not at all, so the ledger can never
        # claim a migration that did not fully apply.
        async with conn.transaction():
            await conn.execute(sql)
            await conn.execute(
                "insert into schema_migrations (filename) values ($1)", path.name
            )
        applied.append(path)
    return applied


async def migrate(dsn: str | None = None, directory: Path | None = None) -> list[Path]:
    conn = await asyncpg.connect(dsn or admin_dsn())
    try:
        await ensure_service_role(conn)
        await assert_documents_table_exists(conn)
        return await apply_migrations(conn, directory)
    finally:
        await conn.close()


async def main() -> None:
    applied = await migrate()
    for path in applied:
        print(f"applied {path.name}")
    if not applied:
        print("nothing to apply; every migration is already recorded")


if __name__ == "__main__":
    asyncio.run(main())
