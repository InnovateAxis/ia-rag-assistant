"""src/db/seed_scale_cli.py — step 2.6's scale seed, invoked as
`python -m src.db.seed_scale_cli`, the same infrastructure seam
`src.db.migrate`, `src.db.bootstrap_roles`, `src.db.tenants`,
`src.db.backfill_cli` and `src.db.offboard_cli` already use for operations an
application module may not perform.

Step 2.6 (CARRYFORWARD F46) needs `document_chunks` seeded to 250,000 rows
before its HNSW parameter sweep means anything, and — the instruction most
likely to be missed — SKEWED across tenants, never evenly split.
CARRYFORWARD F3 found that a tenant holding a small share of a large table
can get zero rows back from the vector index, because RLS is applied as a
post-index filter rather than an index condition. That is a planner-cost
effect driven by table size and tenant skew, not by what the embeddings
mean, so it reproduces on synthetic vectors — but ONLY if the seed is
skewed. An evenly split seed hides the exact effect group-14 (3.1) and
group-15 (3.6) were routed here to receive.

This module writes `tenant_id` as ordinary column DATA on every insert — the
same shape every insert in this project already has
(`src/ingest/pipeline.py`'s `_INSERT_CHUNK` sets the column the identical
way) — never as a `WHERE tenant_id = ...` predicate. Invariant 1 forbids the
latter, not the former; nothing here queries by tenant, only writes rows
stamped with one.

The embeddings are uniform-random vectors. CARRYFORWARD F45: there is no
live embedding model in this environment and no chunker runs either, so
`embed_model` is stamped `'synthetic-2.6-seed'` rather than a real model
name, to keep that fact visible in the data itself, not only in this
docstring. This is a deliberate departure from `src.ingest.pipeline.ingest`
(2.1)'s per-document transaction: this module seeds throwaway synthetic rows
for an index-tuning measurement at a scale (250k) no real corpus on this
project reaches, not real tenant documents, and 2.1-2.5's ingest/chunking/
embedding machinery is not rebuilt here (see this step's carry-forwards,
F44 in particular).

This module is not one of `pyproject.toml`'s `[tool.importlinter]`
`source_modules`, and it is reached only from this file's own `python -m`
entry point and from `scripts/measure_index_tuning_2.6.py` (also outside the
nine application packages) — verified by hand with grep (CARRYFORWARD F43:
the import-linter contract forbids `asyncpg`, not this module by name, so
reachability from application code has to be checked, not assumed to be
caught by the tool).
"""

from __future__ import annotations

import asyncio
import random
import struct
import time
import uuid
from dataclasses import dataclass

import asyncpg

from src.db.bootstrap_roles import admin_dsn

EMBED_DIM = 1536
EMBED_MODEL = "synthetic-2.6-seed"

# The skew CARRYFORWARD F46 asks for by name: one tenant holding a small
# share (111 — "the 111 real chunks are a natural choice") of a large table.
# Two tenants, not three: F3's own language is "a tenant holding a small
# share of a large table", and a third tenant adds nothing this measurement
# needs. Names are distinguishable from the real corpus's `ten_acme` /
# `ten_globex` / `ten_meridian` (CARRYFORWARD F21) on sight, so a reader of
# `document_chunks` can never mistake this synthetic seed for real tenant
# data.
SMALL_TENANT = "ten_scale_small"
LARGE_TENANT = "ten_scale_large"
SMALL_TENANT_CHUNKS = 111
TOTAL_CHUNKS = 250_000
LARGE_TENANT_CHUNKS = TOTAL_CHUNKS - SMALL_TENANT_CHUNKS

_INSERT_DOC = "insert into documents (id, tenant_id) values ($1, $2)"

_INSERT_CHUNK = """
insert into document_chunks
  (id, tenant_id, document_id, chunk_index, content, embedding,
   embed_model, embed_dim, token_count)
values ($1, $2, $3, 0, $4, $5, $6, $7, 12)
"""

BATCH_SIZE = 2000


def _encode_vector_binary(v: list[float]) -> bytes:
    """pgvector's own wire format (`vector_send` in pgvector's `vector.c`):
    uint16 dim, uint16 unused(0), then `dim` big-endian float4s — not
    asyncpg's guesswork, pgvector's documented binary layout."""
    dim = len(v)
    return struct.pack(f">HH{dim}f", dim, 0, *v)


def _decode_vector_binary(b: bytes) -> list[float]:
    (dim, _unused) = struct.unpack_from(">HH", b, 0)
    return list(struct.unpack_from(f">{dim}f", b, 4))


async def register_vector_codec(conn: asyncpg.Connection) -> None:
    """asyncpg has no built-in codec for pgvector's `vector` type.

    Deliberately NOT the text-format codec `src/db/session.py` and
    `tests/isolation/conftest.py` use. Measured on this project: with the
    text codec, `executemany` on a 1536-dim column degrades from ~8,000
    rows/sec to under 200 rows/sec as the batch grows — asyncpg's own docs
    note text-format custom codecs are slow, and at 250,000 rows the
    difference is the difference between this script finishing and being
    killed for the (unrelated) host memory pressure a multi-minute run
    invites. Round-trip correctness against pgvector's real binary format
    was verified before this replaced the text codec. This module writes no
    real tenant data and is never on a request path (see module docstring),
    so this divergence from the application codec is scoped entirely to
    this bulk-seed tool.
    """
    await conn.set_type_codec(
        "vector",
        encoder=_encode_vector_binary,
        decoder=_decode_vector_binary,
        format="binary",
    )


def _synthetic_vector(rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBED_DIM)]


@dataclass(frozen=True)
class SeedThroughput:
    """What got measured while seeding — the genuine half of this step's
    Done-when (CARRYFORWARD F46): synthetic vectors make `hit@5` noise, but
    how fast rows go in is real regardless of what they contain."""

    total_chunks: int
    elapsed_seconds: float

    @property
    def chunks_per_second(self) -> float:
        return self.total_chunks / self.elapsed_seconds if self.elapsed_seconds else 0.0


async def _seed_tenant(
    conn: asyncpg.Connection, tenant_id: str, count: int, *, seed: int, batch_size: int = BATCH_SIZE
) -> None:
    """One document per chunk (`chunk_index` always 0) — the same shape 2.2
    measured on the real corpus (CARRYFORWARD F36: no document in the
    111-document corpus splits into more than one chunk), extended to
    synthetic scale."""
    rng = random.Random(seed)
    written = 0
    while written < count:
        n = min(batch_size, count - written)
        doc_ids = [uuid.uuid4() for _ in range(n)]
        await conn.executemany(_INSERT_DOC, [(doc_id, tenant_id) for doc_id in doc_ids])
        chunk_rows = [
            (
                uuid.uuid4(),
                tenant_id,
                doc_id,
                f"synthetic seed chunk {tenant_id} {written + i}",
                _synthetic_vector(rng),
                EMBED_MODEL,
                EMBED_DIM,
            )
            for i, doc_id in enumerate(doc_ids)
        ]
        await conn.executemany(_INSERT_CHUNK, chunk_rows)
        written += n


async def seed_scale(conn: asyncpg.Connection) -> SeedThroughput:
    """Seed `document_chunks` to 250,000 rows, skewed: `SMALL_TENANT` holds
    `SMALL_TENANT_CHUNKS` (111), `LARGE_TENANT` holds the rest. Returns the
    genuine throughput of the bulk load. Caller must have already applied
    migrations and registered the vector codec on `conn`."""
    start = time.perf_counter()
    await _seed_tenant(conn, SMALL_TENANT, SMALL_TENANT_CHUNKS, seed=1)
    await _seed_tenant(conn, LARGE_TENANT, LARGE_TENANT_CHUNKS, seed=2)
    elapsed = time.perf_counter() - start
    return SeedThroughput(total_chunks=TOTAL_CHUNKS, elapsed_seconds=elapsed)


async def main() -> None:  # pragma: no cover - operator entry point
    conn = await asyncpg.connect(admin_dsn())
    try:
        await register_vector_codec(conn)
        throughput = await seed_scale(conn)
    finally:
        await conn.close()
    print(
        f"seeded {throughput.total_chunks} chunks in {throughput.elapsed_seconds:.1f}s "
        f"({throughput.chunks_per_second:.0f} chunks/sec)"
    )


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
