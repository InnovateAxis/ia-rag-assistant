import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
from asyncpg import Pool

# This module is the sole importer of asyncpg for application code.
# pyproject.toml's [tool.importlinter] contract forbids application modules from importing asyncpg
# (infrastructure modules like bootstrap_roles and migrate are allowed direct connections).
#
# `pool` is bound by `create_pool()`, not at import time. Carry-forward F29:
# nothing in the runbook creates it, so whichever step first needs a live
# connection binds THIS pool rather than improvising a second one — a second
# pool would be exactly the second retrieval path Invariant 1 and the
# import-linter contract both exist to prevent.
pool: Pool


async def _register_vector_codec(conn: asyncpg.Connection) -> None:
    """asyncpg has no built-in codec for pgvector's `vector` type.

    Registered per-connection via `create_pool`'s `init` callback, so every
    connection this pool ever hands out — to any query touching
    `document_chunks.embedding` — already has it.
    """
    await conn.set_type_codec(
        "vector",
        encoder=lambda v: "[" + ",".join(repr(float(x)) for x in v) + "]",
        decoder=lambda s: [float(x) for x in s.strip("[]").split(",")] if s else [],
        format="text",
    )


async def create_pool(dsn: str | None = None) -> Pool:
    """Bind the module-level `pool`. Call once, before the first `session()`.

    `dsn` defaults to `DATABASE_URL`, which `.env.example` documents as the
    SERVICE role's connection string — never the table owner, never a
    superuser (Invariant 4).
    """
    global pool
    pool = await asyncpg.create_pool(
        dsn or os.environ["DATABASE_URL"], init=_register_vector_codec
    )
    return pool


async def close_pool() -> None:
    """Release the module-level `pool`. Mirrors `create_pool`; test fixtures
    and any future application shutdown hook should call this rather than
    reaching into `pool` directly."""
    await pool.close()


@asynccontextmanager
async def session(tenant_id: str | None) -> AsyncGenerator:
    """Every query in this service goes through here. There is no other
    path to the pool for application code; import-linter forbids application modules from
    importing asyncpg directly.

    Args:
        tenant_id: The tenant identifier to set in the request context.
                   If None, the context is unset.

    Yields:
        An asyncpg connection with the tenant claim set in the transaction-local
        request.jwt.claims setting.
    """
    async with pool.acquire() as conn, conn.transaction():
        claims = json.dumps({"tenant_id": tenant_id}) if tenant_id else "{}"
        # set_config with is_local=true scopes it to this transaction,
        # so a pooled connection cannot leak one tenant's context into
        # the next request. This single boolean is load-bearing.
        await conn.execute(
            "select set_config('request.jwt.claims', $1, true)", claims)
        yield conn
