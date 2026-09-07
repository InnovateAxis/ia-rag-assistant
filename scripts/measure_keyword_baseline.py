"""Step 1.4 — measure PostgreSQL full-text search alone, before embeddings exist.

Loads every corpus document into a throwaway table with a `tsvector` column,
then runs each answerable golden question (answerable_single and
answerable_multi — the only two categories with an `expected_chunks` list) as
a `plainto_tsquery` restricted to the asking tenant's own documents, and
checks whether an expected document is in the top 5 by `ts_rank`.

This measures retrieval only, at document granularity: no chunking (2.2), no
embeddings (2.3), no reranking (3.4) exist yet. Every document in this corpus
is short enough that "the right document" and "the right chunk" coincide for
this baseline, which is exactly what makes it possible to measure before
Phase 2 starts.

Deliberately NOT run against the project's real schema: `document_chunks`
(0.3) is RLS-governed and empty until ingestion (2.1) exists, and this script
must not become a second, undocumented retrieval path — it talks to its own
throwaway table in a scratch database, never to the tenant-isolated one.

Usage:
    docker run -d --name p7-baseline-fts -e POSTGRES_PASSWORD=baseline \\
        -e POSTGRES_DB=baseline -p 15544:5432 pgvector/pgvector:pg16
    uv run python scripts/measure_keyword_baseline.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.metrics import hit_rate_at_k, mrr

DSN = "postgresql://postgres:baseline@localhost:15544/baseline"
REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "evals" / "golden.jsonl"
TOP_K = 5


def _git_ls_corpus_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        line
        for line in out.splitlines()
        if line.startswith("corpus/") and line.endswith(".txt")
    ]


def _git_show(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _load_golden() -> list[dict]:
    lines = GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _chunk_to_relpath(chunk_id: str) -> str:
    """'acme/rate_sheets/rate_sheet_01#1' -> 'corpus/acme/rate_sheets/rate_sheet_01.txt'"""
    base = chunk_id.split("#", 1)[0]
    return f"corpus/{base}.txt"


async def main() -> None:
    files = _git_ls_corpus_files()
    if len(files) != 111:
        print(f"WARNING: expected 111 corpus documents, found {len(files)}", file=sys.stderr)

    golden = _load_golden()
    answerable = [q for q in golden if q["type"] in ("answerable_single", "answerable_multi")]

    conn = await asyncpg.connect(DSN)
    try:
        await conn.execute("drop table if exists documents_baseline")
        await conn.execute(
            """
            create table documents_baseline (
                id serial primary key,
                tenant text not null,
                relpath text not null,
                content text not null,
                ts tsvector generated always as (to_tsvector('english', content)) stored
            )
            """
        )

        rows = []
        for relpath in files:
            # corpus/<tenant>/<category>/<file> -> tenant slug is the first segment
            tenant_slug = relpath.split("/")[1]
            content = _git_show(relpath)
            rows.append((tenant_slug, relpath, content))

        await conn.executemany(
            "insert into documents_baseline (tenant, relpath, content) values ($1, $2, $3)",
            rows,
        )
        await conn.execute("create index on documents_baseline using gin (ts)")

        tenant_map = {"ten_acme": "acme", "ten_globex": "globex", "ten_meridian": "meridian"}

        per_category_hits: dict[str, list[bool]] = defaultdict(list)
        per_category_hit1: dict[str, list[bool]] = defaultdict(list)
        per_category_mrr: dict[str, list[float]] = defaultdict(list)
        details = []

        for q in answerable:
            tenant_slug = tenant_map[q["tenant"]]
            expected_relpaths = {_chunk_to_relpath(c) for c in q["expected_chunks"]}

            # plainto_tsquery ANDs every stemmed term, which makes a natural-
            # language question (with words like "allow" or "apply" that never
            # appear verbatim in the source) match nothing even when the
            # document plainly answers it. A keyword-only baseline should
            # measure term-overlap ranking (any of these words, ranked by how
            # many/how rare), not an all-terms-required filter — so the
            # question is stemmed with plainto_tsquery and its AND ('&') is
            # rewritten to OR ('|') before searching.
            and_query = await conn.fetchval(
                "select plainto_tsquery('english', $1)::text", q["question"]
            )
            or_query_text = and_query.replace(" & ", " | ") if and_query else ""
            if not or_query_text:
                results = []
            else:
                # No LIMIT: rank every matching document in this tenant's
                # (30-42 document) corpus so hit@1 / hit@5 / MRR can all be
                # read off the same ranked list, rather than re-querying per k.
                results = await conn.fetch(
                    """
                    select relpath, ts_rank(ts, to_tsquery('english', $1)) as rank
                    from documents_baseline
                    where tenant = $2 and ts @@ to_tsquery('english', $1)
                    order by rank desc, relpath asc
                    """,
                    or_query_text,
                    tenant_slug,
                )
            ranked_relpaths = [r["relpath"] for r in results]
            reciprocal_rank = mrr(ranked_relpaths, expected_relpaths)
            hit_at_1 = hit_rate_at_k(ranked_relpaths, expected_relpaths, k=1)
            hit_at_5 = hit_rate_at_k(ranked_relpaths, expected_relpaths, k=TOP_K)
            per_category_hits[q["type"]].append(hit_at_5)
            per_category_hit1[q["type"]].append(hit_at_1)
            per_category_mrr[q["type"]].append(reciprocal_rank)
            details.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "hit_at_1": hit_at_1,
                    "hit_at_5": hit_at_5,
                    "reciprocal_rank": round(reciprocal_rank, 3),
                    "expected": sorted(expected_relpaths),
                    "top_5": ranked_relpaths[:TOP_K],
                }
            )

        print("=== Step 1.4 — keyword-only (full-text) retrieval baseline ===\n")
        print(f"{'Category':22s} {'hit@1':>8s} {'hit@5':>8s} {'MRR':>8s}   n")
        overall_hit1, overall_hit5, overall_mrr = [], [], []
        for category in ("answerable_single", "answerable_multi"):
            hit1, hit5, rr = per_category_hit1[category], per_category_hits[category], per_category_mrr[category]
            overall_hit1.extend(hit1)
            overall_hit5.extend(hit5)
            overall_mrr.extend(rr)
            print(
                f"{category:22s} {100.0 * sum(hit1) / len(hit1):7.1f}% "
                f"{100.0 * sum(hit5) / len(hit5):7.1f}% "
                f"{sum(rr) / len(rr):8.3f}   {len(hit5)}"
            )
        print(
            f"{'Overall answerable':22s} {100.0 * sum(overall_hit1) / len(overall_hit1):7.1f}% "
            f"{100.0 * sum(overall_hit5) / len(overall_hit5):7.1f}% "
            f"{sum(overall_mrr) / len(overall_mrr):8.3f}   {len(overall_hit5)}\n"
        )

        weak = [d for d in details if d["reciprocal_rank"] < 1.0]
        print(f"Questions where the expected document was NOT ranked #1 ({len(weak)}):")
        for d in weak:
            print(
                f"  {d['id']} [{d['type']}] rr={d['reciprocal_rank']:.3f} "
                f"expected {d['expected']} top5 {d['top_5']}"
            )

        # Refusal categories (cross_tenant_trap, absent_but_plausible,
        # out_of_corpus) have no expected_chunks — a keyword search cannot be
        # "wrong" about them the way it can for an answerable question. What
        # it CAN do is return a confident-looking top-1 match from the
        # asker's own tenant that has nothing to do with the question, which
        # is precisely the shape of the failure the refusal orchestrator
        # (4.1) exists to catch — retrieval isolation stops it from ever
        # seeing another tenant's row, but nothing stops it from serving up
        # its OWN tenant's superficially similar document with confidence.
        trap_probes = []
        for q in golden:
            if q["type"] != "cross_tenant_trap":
                continue
            tenant_slug = tenant_map[q["tenant"]]
            and_query = await conn.fetchval(
                "select plainto_tsquery('english', $1)::text", q["question"]
            )
            or_query_text = and_query.replace(" & ", " | ") if and_query else ""
            top1 = None
            if or_query_text:
                row = await conn.fetchrow(
                    """
                    select relpath, ts_rank(ts, to_tsquery('english', $1)) as rank
                    from documents_baseline
                    where tenant = $2 and ts @@ to_tsquery('english', $1)
                    order by rank desc
                    limit 1
                    """,
                    or_query_text,
                    tenant_slug,
                )
                top1 = row["relpath"] if row else None
            trap_probes.append(
                {"id": q["id"], "question": q["question"], "own_tenant_top1": top1}
            )

        print(f"\nCross-tenant trap probes — top-1 match within the ASKER's own tenant ({len(trap_probes)}):")
        spurious = 0
        for t in trap_probes:
            if t["own_tenant_top1"] is not None:
                spurious += 1
            print(f"  {t['id']}: {t['question']!r} -> {t['own_tenant_top1']}")
        print(
            f"{spurious}/{len(trap_probes)} traps returned a same-tenant document with no "
            "relevance to the question asked — keyword search alone cannot refuse; it always "
            "ranks its best available match, however irrelevant. Refusal has to happen after "
            "retrieval, on a score threshold, not from an empty result set (see step 4.1)."
        )

        Path(REPO_ROOT / "evals" / "baseline_1.4_results.json").write_text(
            json.dumps(
                {
                    "top_k": TOP_K,
                    "per_category_hit_at_1": {
                        cat: sum(v) / len(v) for cat, v in per_category_hit1.items()
                    },
                    "per_category_hit_at_5": {
                        cat: sum(v) / len(v) for cat, v in per_category_hits.items()
                    },
                    "per_category_mrr": {
                        cat: sum(v) / len(v) for cat, v in per_category_mrr.items()
                    },
                    "overall_answerable_hit_at_1": sum(overall_hit1) / len(overall_hit1),
                    "overall_answerable_hit_at_5": sum(overall_hit5) / len(overall_hit5),
                    "overall_answerable_mrr": sum(overall_mrr) / len(overall_mrr),
                    "details": details,
                    "cross_tenant_trap_probes": trap_probes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("\nWrote evals/baseline_1.4_results.json")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
