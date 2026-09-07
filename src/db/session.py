import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from asyncpg import Pool


# Pool is initialized elsewhere; this module enforces it is only accessed here.
# An import-lint rule forbids importing pool outside this module.
pool: Pool


@asynccontextmanager
async def session(tenant_id: str | None) -> AsyncGenerator:
    """Every query in this service goes through here. There is no other
    path to the pool, enforced by an import-lint rule.

    Args:
        tenant_id: The tenant identifier to set in the request context.
                   If None, the context is unset.

    Yields:
        An asyncpg connection with the tenant claim set in the transaction-local
        request.jwt.claims setting.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            claims = json.dumps({"tenant_id": tenant_id}) if tenant_id else "{}"
            # set_config with is_local=true scopes it to this transaction,
            # so a pooled connection cannot leak one tenant's context into
            # the next request. This single boolean is load-bearing.
            await conn.execute(
                "select set_config('request.jwt.claims', $1, true)", claims)
            yield conn
