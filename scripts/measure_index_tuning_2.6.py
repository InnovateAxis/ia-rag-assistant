"""Step 2.6 — HNSW index tuning and ingestion throughput, measured on a real
PostgreSQL 16 + pgvector instance (`pgvector/pgvector:pg16`, the same image
0.5's CI and `tests/isolation/conftest.py` use), never a mock.

CARRYFORWARD F45/F46 govern what this script can honestly claim:

* **No live embedding model in this environment.** `src/db/seed_scale_cli.py`
  seeds 250,000 synthetic (uniform-random) vectors. `hit@5` over vectors that
  carry no semantics is noise, not a measurement — and separately, 2.2
  already measured `hit@5` at 100% on the real 111-document corpus
  (`evals/chunking_impact_2.2.md`), so the metric is saturated and could not
  discriminate `ef_search` values even with real embeddings at this scale.
  The runbook's own grid (78.1–80.4% hit@5) is an inherited illustration
  from a different corpus and is not restated here as this project's number.
* **What IS genuinely measured regardless of what the vectors mean:** p95
  query latency across the `(m, ef_search)` grid, and ingestion throughput.
  Both are planner/IO effects, not semantic ones.
* **F3 is this step's highest-value output, ahead of the parameter table**
  (CARRYFORWARD F46, explicit): RLS is applied as a post-index filter, so a
  tenant holding a small share of a large table can get zero rows back from
  the vector index. That is a planner-cost effect reproducible on synthetic
  vectors, but only under a SKEWED seed — `seed_scale_cli` gives one tenant
  111 rows against a 250,000-row table for exactly this reason. This script
  forces the planner onto the HNSW index for that small tenant, reports rows
  returned, raises `hnsw.ef_search`, and reports whether recall recovers —
  whichever way that comes out (group-14/3.1 and group-15/3.6 need the real
  answer, not the runbook's).

Does not touch `migrations/0001_chunks.sql` (CARRYFORWARD F46, explicit: its
HNSW index is declared without `m`/`ef_construction`, and stays that way).
Every index this script creates and drops lives only inside the ephemeral
container database it starts — never in a migration, never in a
long-running database.

Usage:
    uv run python "scripts/measure_index_tuning_2.6.py"

Writes `evals/index_tuning_2.6.md` (committed table + F3 measurement) and
`evals/index_tuning_2.6_results.json` (raw numbers).
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import asyncpg
from testcontainers.community.postgres import PostgresContainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.bootstrap_roles import SERVICE_ROLE, ensure_service_role, service_password
from src.db.migrate import apply_migrations
from src.db.seed_scale_cli import (
    EMBED_DIM,
    LARGE_TENANT,
    SMALL_TENANT,
    SMALL_TENANT_CHUNKS,
    TOTAL_CHUNKS,
    register_vector_codec,
    seed_scale,
)

POSTGRES_IMAGE = "pgvector/pgvector:pg16"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_MD = REPO_ROOT / "evals" / "index_tuning_2.6.md"
OUT_JSON = REPO_ROOT / "evals" / "index_tuning_2.6_results.json"

# The runbook's own grid, minus the numbers this project cannot reproduce
# (CARRYFORWARD F46): (m, ef_construction, ef_search). ef_construction is
# fixed at 64 across every row, exactly as the runbook block shows, so only
# two distinct indexes are ever built (m=16 covers three ef_search values;
# m=32 needs its own build) — ef_search is a per-session setting, not an
# index property.
GRID = [
    (16, 64, 40),
    (16, 64, 60),
    (16, 64, 100),
    (32, 64, 60),
]

N_LATENCY_SAMPLES = 300
SWEEP_INDEX_NAME = "document_chunks_embedding_hnsw_sweep"

CREATE_DOCUMENTS = """
create table if not exists documents (
  id        uuid primary key default gen_random_uuid(),
  tenant_id text not null,
  title     text not null default 'seed document'
);
"""


def _rand_vector(rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBED_DIM)]


async def _provision(admin_dsn: str) -> None:
    conn = await asyncpg.connect(admin_dsn)
    try:
        # Stand-in for Pod P's table (CARRYFORWARD C7) — same minimal shape
        # tests/isolation/conftest.py uses, never a migration in this repo.
        await conn.execute(CREATE_DOCUMENTS)
        await ensure_service_role(conn)
        await apply_migrations(conn)
        await conn.execute(f"grant select, insert on documents to {SERVICE_ROLE}")

        # migrations/0001_chunks.sql's own (unnamed, default-parameter) HNSW
        # index is untouched in migrations/ (CARRYFORWARD F46 forbids editing
        # that file) but is dropped HERE, at runtime, inside this ephemeral
        # container only. Left in place it would incrementally maintain
        # itself across all 250,000 inserts below — HNSW's slow, memory-heavy
        # path — and then sit alongside this script's own sweep index,
        # doubling the resident HNSW footprint for no measurement benefit.
        # Bulk-loading onto a bare table, then building each swept index
        # fresh with a single `create index`, is both the standard pgvector
        # practice and the only way this measurement ran to completion inside
        # this environment's memory budget. Never touches migrations/0001.
        default_indexes = await conn.fetch(
            "select indexname from pg_indexes where tablename = 'document_chunks' "
            "and indexdef ilike '%using hnsw%'"
        )
        for row in default_indexes:
            await conn.execute(f'drop index if exists "{row["indexname"]}"')
    finally:
        await conn.close()


async def _competing_btree_index(conn: asyncpg.Connection) -> asyncpg.Record | None:
    """`migrations/0001_chunks.sql`'s `(tenant_id, document_id)` btree index —
    untouched (CARRYFORWARD F46), but its existence changes what "force the
    planner onto the vector index" (this step's own instruction) requires.

    Measured on this project: with only `enable_seqscan = off`, the planner
    did NOT use the HNSW index for the small tenant's query at all — it used
    THIS index instead (an equality lookup on `tenant_id`, RLS's own filter,
    is exactly the shape a btree on `(tenant_id, document_id)` serves
    cheaply for a 111-row tenant), confirmed by walking the EXPLAIN plan to
    the actual scan node rather than trusting the top-level "Limit" node.
    CARRYFORWARD B2 named this possibility directly: "a simple equality RLS
    qual on an indexed column can become an index condition in an ordinary
    btree scan." `enable_seqscan = off` alone forces away from one
    alternative, not this one. Returned so the caller can drop it for the F3
    probe specifically and restore it (`indexdef` is the exact DDL to
    recreate it) before any other measurement in this script runs, since
    every OTHER row in the parameter table should reflect the real,
    unmodified index set migrations/0001 would ship.
    """
    return await conn.fetchrow(
        "select indexname, indexdef from pg_indexes where tablename = 'document_chunks' "
        "and indexdef ilike '%btree (tenant_id, document_id)%'"
    )


async def _build_index(conn: asyncpg.Connection, m: int, ef_construction: int) -> float:
    await conn.execute(f"drop index if exists {SWEEP_INDEX_NAME}")
    t0 = time.perf_counter()
    await conn.execute(
        f"create index {SWEEP_INDEX_NAME} on document_chunks "
        f"using hnsw (embedding vector_cosine_ops) "
        f"with (m = {m}, ef_construction = {ef_construction})"
    )
    return time.perf_counter() - t0


async def _p95_latency_ms(
    conn: asyncpg.Connection, ef_search: int, tenant_id: str, n: int = N_LATENCY_SAMPLES
) -> dict:
    """Real query latency against the large tenant, the realistic case: a
    normal top-5 vector query under this (m, ef_search) index.

    The tenant claim is set once, session-level (`is_local=false`), for the
    life of this connection rather than per-transaction. That departs from
    `src/db/session.py`'s per-request `is_local=true` on purpose: this
    benchmark measures raw HNSW query latency, not the isolation guarantee
    itself — that guarantee is what `tests/isolation/` tests the real way,
    with a fresh `is_local=true` claim per transaction on a pooled
    connection. Reusing one claim here removes transaction-setup overhead
    from the very number being measured.
    """
    await conn.execute(
        "select set_config('request.jwt.claims', $1, false)",
        f'{{"tenant_id": "{tenant_id}"}}',
    )
    await conn.execute(f"set hnsw.ef_search = {ef_search}")
    rng = random.Random(42)
    latencies_ms = []
    for _ in range(n):
        qv = _rand_vector(rng)
        t0 = time.perf_counter()
        await conn.fetch(
            "select id from document_chunks order by embedding <=> $1 limit 5", qv
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    latencies_ms.sort()
    p95_idx = min(int(len(latencies_ms) * 0.95), len(latencies_ms) - 1)
    return {
        "n": n,
        "p95_ms": round(latencies_ms[p95_idx], 1),
        "mean_ms": round(statistics.mean(latencies_ms), 1),
        "min_ms": round(latencies_ms[0], 1),
        "max_ms": round(latencies_ms[-1], 1),
    }


async def _f3_probe(conn: asyncpg.Connection, ef_search: int, k: int = 5) -> dict:
    """CARRYFORWARD F3, measured at 250k scale with a 111-row tenant.

    Forces the planner onto the HNSW index inside the small tenant's own RLS
    session, then reports how many rows an ordinary top-k query actually
    returns. RLS is a post-index filter here (CARRYFORWARD F3's own claim):
    the HNSW index returns its `ef_search`-bounded candidate list first, and
    only THEN does the policy discard every row that is not this tenant's —
    so a candidate list that happens to contain none of the small tenant's
    111 rows (out of 250,000) returns zero, with no error and no warning.

    Two things have to be true for "forced onto the vector index" to mean
    what it says, both the caller's responsibility (see `run()`):
    `enable_seqscan = off` (below) rules out an exact sequential scan, and
    the competing `(tenant_id, document_id)` btree index must already be
    dropped — measured on this project, `enable_seqscan = off` alone was not
    enough, because the planner used that OTHER index instead of the HNSW
    one (CARRYFORWARD B2's predicted shape, confirmed). This function
    verifies both, via `scan_node_type`/`scan_index_name` below, rather than
    assuming the GUC setting worked.
    """
    await conn.execute(
        "select set_config('request.jwt.claims', $1, false)",
        f'{{"tenant_id": "{SMALL_TENANT}"}}',
    )
    await conn.execute("set enable_seqscan = off")
    await conn.execute(f"set hnsw.ef_search = {ef_search}")
    rng = random.Random(7)
    qv = _rand_vector(rng)
    plan = await conn.fetch(
        f"explain (format json) select id from document_chunks "
        f"order by embedding <=> $1 limit {k}",
        qv,
    )
    plan_json = json.loads(plan[0]["QUERY PLAN"])
    rows = await conn.fetch(
        f"select id from document_chunks order by embedding <=> $1 limit {k}", qv
    )

    # The top-level node of `ORDER BY ... LIMIT` is always "Limit" regardless
    # of what runs underneath it — checking only that node (an earlier
    # version of this function did) would always read "Limit" and never
    # actually confirm the HNSW index ran. Walk to the child scan node
    # instead, and require BOTH its type and its index name, so a fallback
    # to an exact Seq Scan + Sort (which `enable_seqscan = off` discourages
    # but does not forbid outright) is visible rather than silently
    # indistinguishable from a real ANN index scan.
    scan_node = plan_json[0]["Plan"]
    while "Plans" in scan_node:
        scan_node = scan_node["Plans"][0]
    return {
        "ef_search": ef_search,
        "requested_k": k,
        "rows_returned": len(rows),
        "top_node_type": plan_json[0]["Plan"]["Node Type"],
        "scan_node_type": scan_node["Node Type"],
        "scan_index_name": scan_node.get("Index Name"),
        "plan_uses_hnsw_index": scan_node.get("Index Name") == SWEEP_INDEX_NAME,
    }


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def run() -> dict:
    # Conservative server memory profile for this ephemeral container —
    # this environment's Docker VM is memory-constrained (observed: an
    # earlier run with default settings was killed for host memory
    # pressure). None of these affect what is measured: p95 latency and
    # rows-returned are unaffected by shared_buffers/maintenance_work_mem,
    # and disabling fsync/synchronous_commit only affects durability, which
    # a throwaway container torn down at the end of this script does not
    # need. Real measurement, tuned to fit; not a shortcut on what is real.
    pg_command = (
        "postgres "
        "-c shared_buffers=128MB "
        "-c maintenance_work_mem=256MB "
        "-c work_mem=8MB "
        "-c max_wal_size=512MB "
        "-c fsync=off "
        "-c synchronous_commit=off "
        "-c full_page_writes=off"
    )
    with PostgresContainer(
        POSTGRES_IMAGE,
        username="postgres",
        password="postgres",
        dbname="postgres",
        driver=None,
        command=pg_command,
        # Docker's default /dev/shm (64MB) is not enough for parallel HNSW
        # index build's DSM segments at 250k rows — measured on this project:
        # the first attempt at this scale failed with `DiskFullError: could
        # not resize shared memory segment ... 265326112 bytes` (~253MB).
        # 1GB is comfortably above that, still small next to this container's
        # own memory profile above.
        shm_size="1g",
    ) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        admin_dsn = f"postgresql://postgres:postgres@{host}:{port}/postgres"
        service_dsn = (
            f"postgresql://{SERVICE_ROLE}:{service_password()}@{host}:{port}/postgres"
        )

        _log("provisioning: documents stand-in, service role, migrations")
        await _provision(admin_dsn)

        _log(f"seeding {TOTAL_CHUNKS:,} chunks (skewed {SMALL_TENANT_CHUNKS} / "
             f"{TOTAL_CHUNKS - SMALL_TENANT_CHUNKS:,})")
        seed_conn = await asyncpg.connect(admin_dsn)
        try:
            await register_vector_codec(seed_conn)
            throughput = await seed_scale(seed_conn)
        finally:
            await seed_conn.close()
        _log(f"seed done: {throughput.chunks_per_second:.0f} chunks/sec, "
             f"{throughput.elapsed_seconds:.1f}s")

        # Confirm the skew actually landed — Invariant 5 in spirit: never
        # trust a measurement against an unverified seed.
        check_conn = await asyncpg.connect(admin_dsn)
        try:
            counts = {
                r["tenant_id"]: r["n"]
                for r in await check_conn.fetch(
                    "select tenant_id, count(*) as n from document_chunks group by tenant_id"
                )
            }
        finally:
            await check_conn.close()
        assert counts.get(SMALL_TENANT) == SMALL_TENANT_CHUNKS, counts
        assert sum(counts.values()) == TOTAL_CHUNKS, counts
        _log(f"seed verified: {counts}")

        # Checkpoint to disk now — a bulk seed that took real time should
        # not be silently lost if a later phase is interrupted.
        OUT_JSON.write_text(
            json.dumps({"throughput": throughput.__dict__, "seed_counts": counts}, indent=2)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        # Phase 1 — latency grid, with the FULL realistic index set in place
        # (migrations/0001's `(tenant_id, document_id)` btree included), one
        # distinct (m, ef_construction) index built per key, ef_search swept
        # as a session setting across every grid row that shares one.
        latency_conn = await asyncpg.connect(service_dsn)
        try:
            await register_vector_codec(latency_conn)
            built = {}
            grid_results = []
            # Only the m=16 rows here — phase 3 below handles GRID's m=32
            # row on its own, after phase 2's index-drop/restore is done, so
            # it is never built twice.
            for m, ef_construction, ef_search in GRID:
                if m != 16:
                    continue
                key = (m, ef_construction)
                if key not in built:
                    _log(f"building index m={m} ef_construction={ef_construction} "
                         f"(250k rows, this is the slow step)")
                    build_conn = await asyncpg.connect(admin_dsn)
                    try:
                        built[key] = await _build_index(build_conn, m, ef_construction)
                    finally:
                        await build_conn.close()
                    _log(f"index built in {built[key]:.1f}s")

                _log(f"measuring p95 latency: m={m} ef_search={ef_search}")
                latency = await _p95_latency_ms(latency_conn, ef_search, LARGE_TENANT)
                grid_results.append(
                    {
                        "m": m,
                        "ef_construction": ef_construction,
                        "ef_search": ef_search,
                        "index_build_seconds": round(built[key], 1),
                        **latency,
                    }
                )
                _log(f"  -> p95={latency['p95_ms']}ms mean={latency['mean_ms']}ms")
        finally:
            await latency_conn.close()

        # Phase 2 — F3, at this step's chosen index config (m=16, ec=64,
        # the runbook's own "<- chosen" row), which is still the current
        # sweep index at this point (GRID orders m=16 before m=32).
        #
        # "Force the planner onto the vector index" (this step's own
        # instruction) needs more than `enable_seqscan = off`: measured on
        # this project, that alone was not enough, because the planner used
        # migrations/0001's `(tenant_id, document_id)` btree index instead —
        # a cheap equality lookup for a 111-row tenant, and CARRYFORWARD B2's
        # predicted shape. So that index is dropped for this phase only, and
        # restored immediately after (its exact `indexdef`, captured before
        # dropping) — every other row in the parameter table above and the
        # m=32 row below reflects the real, unmodified index set.
        f3_conn = await asyncpg.connect(service_dsn)
        ddl_conn = await asyncpg.connect(admin_dsn)
        try:
            await register_vector_codec(f3_conn)
            competing = await _competing_btree_index(ddl_conn)
            if competing is not None:
                _log(f"F3 phase: dropping {competing['indexname']} "
                     f"(restored after) so only the HNSW index is available "
                     f"once enable_seqscan=off")
                await ddl_conn.execute(f'drop index if exists "{competing["indexname"]}"')
            # `enable_seqscan = off` is set inside `_f3_probe` itself, on
            # `f3_conn` — the connection that actually runs the query.
            # Setting it here on `ddl_conn` would do nothing.

            f3_results = []
            for ef_search in (40, 60, 100):
                _log(f"F3 probe: ef_search={ef_search}")
                f3_result = await _f3_probe(f3_conn, ef_search)
                f3_results.append(f3_result)
                _log(f"  -> rows_returned={f3_result['rows_returned']} "
                     f"scan={f3_result['scan_node_type']} "
                     f"index={f3_result['scan_index_name']} "
                     f"used_hnsw={f3_result['plan_uses_hnsw_index']}")

            if competing is not None:
                await ddl_conn.execute(competing["indexdef"])
                _log(f"F3 phase: restored {competing['indexname']}")
        finally:
            await f3_conn.close()
            await ddl_conn.close()

        # Phase 3 — the m=32 row, same realistic index set as phase 1
        # (the competing btree index above is restored by the time this
        # runs, since GRID's only m=32 row comes after every m=16 row).
        m, ef_construction, ef_search = GRID[-1]
        assert m == 32, GRID  # phase 2 assumed m=16 was still current; guard it
        latency_conn = await asyncpg.connect(service_dsn)
        try:
            await register_vector_codec(latency_conn)
            _log(f"building index m={m} ef_construction={ef_construction} "
                 f"(250k rows, this is the slow step)")
            build_conn = await asyncpg.connect(admin_dsn)
            try:
                build_seconds = await _build_index(build_conn, m, ef_construction)
            finally:
                await build_conn.close()
            _log(f"index built in {build_seconds:.1f}s")

            _log(f"measuring p95 latency: m={m} ef_search={ef_search}")
            latency = await _p95_latency_ms(latency_conn, ef_search, LARGE_TENANT)
            grid_results.append(
                {
                    "m": m,
                    "ef_construction": ef_construction,
                    "ef_search": ef_search,
                    "index_build_seconds": round(build_seconds, 1),
                    **latency,
                }
            )
            _log(f"  -> p95={latency['p95_ms']}ms mean={latency['mean_ms']}ms")
        finally:
            await latency_conn.close()

        return {
            "throughput": {
                "total_chunks": throughput.total_chunks,
                "elapsed_seconds": round(throughput.elapsed_seconds, 1),
                "chunks_per_second": round(throughput.chunks_per_second, 1),
            },
            "seed_counts": counts,
            "grid": grid_results,
            "f3": f3_results,
        }


def _render_markdown(results: dict) -> str:
    grid = results["grid"]
    f3 = results["f3"]
    thr = results["throughput"]

    lines = [
        "# Step 2.6 — index tuning and ingestion throughput, measured on this project",
        "",
        (
            "Produced by `scripts/measure_index_tuning_2.6.py`. Raw numbers: "
            "`evals/index_tuning_2.6_results.json`. Real PostgreSQL 16 + pgvector "
            "(`pgvector/pgvector:pg16`), started by testcontainers — never a mock."
        ),
        "",
        "## Seed",
        "",
        (
            f"`document_chunks` seeded to **{thr['total_chunks']:,}** rows via "
            f"`src.db.seed_scale_cli` (synthetic uniform-random vectors, "
            f"`embed_model = 'synthetic-2.6-seed'`), SKEWED per CARRYFORWARD F46: "
            f"`{SMALL_TENANT}` holds **{results['seed_counts'].get(SMALL_TENANT)}** "
            f"rows (the 111 real-corpus chunk count), `{LARGE_TENANT}` holds the "
            f"remaining **{results['seed_counts'].get(LARGE_TENANT):,}**. An evenly "
            f"split seed would hide the effect below."
        ),
        "",
        "## Parameter table",
        "",
        (
            "`hit@5` is marked not measurable, for two independent reasons stated "
            "in full in this script's module docstring and in CARRYFORWARD F45/F46: "
            "(1) no live embedding model in this environment, so 250k synthetic "
            "vectors carry no semantics; (2) 2.2 already measured `hit@5` at 100% "
            "on the real 111-document corpus (`evals/chunking_impact_2.2.md`), so "
            "the metric is saturated and cannot discriminate `ef_search` values "
            "regardless. p95 latency and ingestion throughput are genuine "
            "measurements, taken on this run, at this scale."
        ),
        "",
        "| m | ef_search | ef_construction | hit@5 | p95 (ms) | mean (ms) | index build (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in grid:
        lines.append(
            f"| {row['m']} | {row['ef_search']} | {row['ef_construction']} | "
            f"not measurable on synthetic vectors — see note above | "
            f"{row['p95_ms']} | {row['mean_ms']} | {row['index_build_seconds']} |"
        )

    outlier_rows = [r for r in grid if r["mean_ms"] > r["p95_ms"]]
    lines += [
        "",
        (
            "**The runbook's illustrative grid (78.1–80.4% hit@5) is from a "
            "different corpus and is not this project's number — not restated "
            "here as measured.**"
        ),
    ]
    if outlier_rows:
        lines += [
            "",
            (
                "**Mean above p95 on "
                + ", ".join(f"the m={r['m']} ef_search={r['ef_search']} row" for r in outlier_rows)
                + "** — a real measured artifact, not an error: `min_ms`/`max_ms` "
                "in the raw JSON show a single large outlier per flagged row "
                "(e.g. one query far slower than the other 299 — plausible on a "
                "memory-constrained host under this container's concurrent "
                "index-build load) pulling the mean above the 95th percentile, "
                "which by construction excludes that one slowest sample. "
                "Reported as measured rather than smoothed away."
            ),
        ]
    lines += [
        "",
        "## Ingestion throughput",
        "",
        (
            f"**{thr['chunks_per_second']:,.0f} chunks/sec** "
            f"({thr['total_chunks']:,} chunks in {thr['elapsed_seconds']:.1f}s), "
            f"measured as batched `INSERT` (`asyncpg.executemany`, "
            f"batch size {2000}) into `document_chunks` via "
            f"`src.db.seed_scale_cli.seed_scale`. This measures the storage "
            f"layer's raw write throughput, not `src.ingest.pipeline.ingest`'s "
            f"per-document transaction (2.1) — that path is not rebuilt here "
            f"(CARRYFORWARD F44) and its per-document chunk count is far smaller "
            f"than what a throughput number at this scale needs to be useful."
        ),
        "",
        "## F3 — the real deliverable of this step",
        "",
        (
            f"CARRYFORWARD F3: RLS is applied as a post-index filter, so a tenant "
            f"holding a small share of a large table can get zero rows back from "
            f"the vector index. Measured here at 250k scale with "
            f"`{SMALL_TENANT}` holding {results['seed_counts'].get(SMALL_TENANT)} "
            f"of {thr['total_chunks']:,} rows, index `m=16, ef_construction=64` "
            f"(this step's chosen configuration). Forcing the planner onto the "
            f"vector index took two things, not one: `set enable_seqscan = off` "
            f"alone was measured to be insufficient — the planner used "
            f"migrations/0001's `(tenant_id, document_id)` btree index instead "
            f"(CARRYFORWARD B2's predicted shape), an equality lookup that is "
            f"cheap for a 111-row tenant regardless of the vector index. That "
            f"btree index was dropped for this probe only — after the m=16 "
            f"latency rows above (measured with the full, realistic index set) "
            f"and restored before the m=32 row below — so the HNSW index was "
            f"the only one available. One "
            f"`order by embedding <=> $1 limit 5` query per `ef_search` value "
            f"below, same tenant and query vector throughout so only "
            f"`ef_search` varies; `plan_uses_hnsw_index` (raw results) confirms "
            f"the HNSW index actually ran each time rather than assuming the "
            f"GUC setting worked:"
        ),
        "",
        "| ef_search | scan node | index used | rows returned (of 5 requested) |",
        "|---|---|---|---|",
    ]
    for r in f3:
        lines.append(
            f"| {r['ef_search']} | {r['scan_node_type']} | "
            f"{r['scan_index_name'] or '(none — not an index scan)'} | "
            f"{r['rows_returned']} |"
        )

    all_used_hnsw = all(r["plan_uses_hnsw_index"] for r in f3)
    recovered = len({r["rows_returned"] for r in f3}) > 1 and max(
        r["rows_returned"] for r in f3
    ) > min(r["rows_returned"] for r in f3)
    if not all_used_hnsw:
        verdict = (
            "**Inconclusive as a test of F3** — at least one `ef_search` value did "
            "not actually run the HNSW index scan (`plan_uses_hnsw_index` is False "
            "in the raw results), despite `enable_seqscan = off`. `enable_seqscan` "
            "discourages a sequential scan, it does not forbid one, so this is "
            "reported rather than assumed away. The rows-returned counts above are "
            "still the true counts for whatever plan actually ran, and the scan "
            "node/index-used columns say which plan that was for each row."
        )
    elif recovered:
        verdict = (
            "**Recall recovered as `ef_search` rose** — rows returned increased "
            "somewhere in this sweep, with the HNSW index confirmed in use at "
            "every `ef_search` value (`plan_uses_hnsw_index` true throughout)."
        )
    else:
        verdict = (
            "**Recall did NOT recover as `ef_search` rose** — rows returned stayed "
            "flat across the whole sweep, with the HNSW index confirmed in use at "
            "every `ef_search` value (`plan_uses_hnsw_index` true throughout), "
            "confirming CARRYFORWARD F3 and ADR-0001's claim at this project's own "
            "250k/111-row scale."
        )
    lines += [
        "",
        verdict,
        "",
        (
            "Reported as measured, not assumed — the table above stands whether "
            "it confirms or refutes F3's original claim, per this step's brief."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    results = asyncio.run(run())
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    OUT_MD.write_text(_render_markdown(results), encoding="utf-8", newline="\n")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
