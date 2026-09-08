"""Step 2.2 — measure the five chunking-strategy rows for real, on this corpus.

CARRYFORWARD F36 decided (user, explicitly): "measure honestly and commit
the real table" rather than ship the runbook's inherited 58.2%->81.4%
illustration, which is a different, larger corpus's numbers. This script is
this project's measurement, same disposition and same reasoning as
`scripts/measure_keyword_baseline.py` (1.4): reads the corpus from
`git show HEAD:<path>` rather than the working tree (so the numbers are
provably about a specific commit, not local edits), talks to its own
throwaway table rather than the real `document_chunks` schema (must not
become a second, undocumented retrieval path — Invariant 1), and measures
retrieval only, via PostgreSQL full-text search, exactly like 1.4 did before
chunking existed.

**Why keyword search again, not real embeddings.** 2.3's embedder
(`src/llm/embeddings.py`) calls a live embedding provider; there is no API
key configured in this environment, and this script must not fabricate
semantic-similarity numbers by faking one. Keyword full-text search over
each strategy's SEARCHED TEXT is a real, honestly-measurable proxy for what
the case-study table's rows 4 and 5 actually vary — "they change the
embedded text rather than the boundaries" — because the *searched text*
changes exactly the way the *embedded text* would. It is not a substitute
for a real embedding-based retrieval measurement, and the note this script
prints alongside the table says so.

**What each row searches, concretely** (see `src/chunking/rules.py` and
`src/llm/embeddings.py` for where each piece comes from):

  1. Fixed 512 chars, no structure   -> `chunk_fixed_chars_naive(text)[i].text`
  2. Fixed 512 tokens, sentence-safe -> `chunk_fixed_tokens_sentence_safe(text)[i].text`
  3. Section-aware, no path prefix   -> `chunk_structure_aware(text)[i].text`
  4. Section-aware + path prefix     -> `"{section_path joined} > \n\n{content}"`
                                         (`src/llm/embeddings.py::_embed_text`'s
                                         exact formula — rule 5)
  5. Section-aware + path + metadata -> row 4's text + the document's parsed
                                         front-matter (rule 6: title/date/
                                         version). No `document_chunks`
                                         column carries this (see
                                         `src/llm/embeddings.py`'s module
                                         docstring) — it is measured here
                                         only, never written to the real
                                         schema.

Usage (same container pattern as 1.4):
    docker run -d --name p7-chunking-fts -e POSTGRES_PASSWORD=baseline \\
        -e POSTGRES_DB=baseline -p 15545:5432 pgvector/pgvector:pg16
    uv run python scripts/measure_chunking_impact.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.metrics import hit_rate_at_k, mrr
from src.chunking.rules import (
    RawChunk,
    chunk_fixed_chars_naive,
    chunk_fixed_tokens_sentence_safe,
    chunk_structure_aware,
    golden_chunk_id,
    parse_structure,
)

DSN = "postgresql://postgres:baseline@localhost:15545/baseline"
REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "evals" / "golden.jsonl"
TOP_K = 5

STRATEGIES = (
    "Fixed 512 chars, no structure",
    "Fixed 512 tokens, sentence-safe",
    "Section-aware, no path prefix",
    "Section-aware + path prefix",
    "Section-aware + path + metadata",
)


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    tenant_slug: str
    texts: tuple[str, str, str, str, str]  # one per strategy, same order as STRATEGIES


def _git_ls_corpus_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(
        line for line in out.splitlines() if line.startswith("corpus/") and line.endswith(".txt")
    )


def _git_show(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _load_golden() -> list[dict]:
    lines = GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _embedded_text(chunk: RawChunk) -> str:
    """Row 4 — matches `src/llm/embeddings.py::_embed_text` exactly."""
    prefix = " > ".join(chunk.section_path)
    return f"{prefix}\n\n{chunk.text}" if prefix else chunk.text


def _metadata_suffix(text: str) -> str:
    """Row 5's addition over row 4: the parsed front-matter block (rule 6),
    rendered as plain text. Empty string if a document has no metadata."""
    parsed = parse_structure(text)
    if not parsed.metadata:
        return ""
    return " | ".join(f"{k}: {v}" for k, v in parsed.metadata.items())


def _index_document(relpath: str, text: str) -> list[IndexedChunk]:
    # corpus/<tenant>/<category>/<file>.txt
    _, tenant_slug, category, filename = relpath.split("/")
    stem = filename.removesuffix(".txt")

    naive = chunk_fixed_chars_naive(text)
    sentence_safe = chunk_fixed_tokens_sentence_safe(text)
    structure_aware = chunk_structure_aware(text)

    n = len(structure_aware)
    if not (len(naive) == n and len(sentence_safe) == n):
        # Would only happen on a document long enough for the strategies to
        # disagree (CARRYFORWARD F36: none of the 111 are). Fail loudly
        # rather than silently mis-align chunk indices across strategies.
        raise AssertionError(
            f"{relpath}: strategies disagree on chunk count "
            f"(naive={len(naive)}, sentence_safe={len(sentence_safe)}, "
            f"structure_aware={n}) — this corpus was expected to never split"
        )

    metadata_suffix = _metadata_suffix(text)
    indexed = []
    for i in range(n):
        row4 = _embedded_text(structure_aware[i])
        row5 = f"{row4}\n\n{metadata_suffix}" if metadata_suffix else row4
        indexed.append(
            IndexedChunk(
                chunk_id=golden_chunk_id(tenant_slug, category, stem, i),
                tenant_slug=tenant_slug,
                texts=(naive[i].text, sentence_safe[i].text, structure_aware[i].text, row4, row5),
            )
        )
    return indexed


async def _measure_strategy(
    conn: asyncpg.Connection,
    strategy_index: int,
    chunks: list[IndexedChunk],
    answerable: list[dict],
    tenant_map: dict[str, str],
) -> dict:
    await conn.execute("drop table if exists chunks_baseline")
    await conn.execute(
        """
        create table chunks_baseline (
            chunk_id text primary key,
            tenant text not null,
            content text not null,
            ts tsvector generated always as (to_tsvector('english', content)) stored
        )
        """
    )
    await conn.executemany(
        "insert into chunks_baseline (chunk_id, tenant, content) values ($1, $2, $3)",
        [(c.chunk_id, c.tenant_slug, c.texts[strategy_index]) for c in chunks],
    )
    await conn.execute("create index on chunks_baseline using gin (ts)")

    per_category_hit1: dict[str, list[bool]] = defaultdict(list)
    per_category_hit5: dict[str, list[bool]] = defaultdict(list)
    per_category_mrr: dict[str, list[float]] = defaultdict(list)

    for q in answerable:
        tenant_slug = tenant_map[q["tenant"]]
        expected = set(q["expected_chunks"])

        and_query = await conn.fetchval(
            "select plainto_tsquery('english', $1)::text", q["question"]
        )
        or_query_text = and_query.replace(" & ", " | ") if and_query else ""
        if not or_query_text:
            ranked_ids: list[str] = []
        else:
            rows = await conn.fetch(
                """
                select chunk_id, ts_rank(ts, to_tsquery('english', $1)) as rank
                from chunks_baseline
                where tenant = $2 and ts @@ to_tsquery('english', $1)
                order by rank desc, chunk_id asc
                """,
                or_query_text,
                tenant_slug,
            )
            ranked_ids = [r["chunk_id"] for r in rows]

        per_category_hit1[q["type"]].append(hit_rate_at_k(ranked_ids, expected, k=1))
        per_category_hit5[q["type"]].append(hit_rate_at_k(ranked_ids, expected, k=TOP_K))
        per_category_mrr[q["type"]].append(mrr(ranked_ids, expected))

    overall_hit1 = [v for vs in per_category_hit1.values() for v in vs]
    overall_hit5 = [v for vs in per_category_hit5.values() for v in vs]
    overall_mrr = [v for vs in per_category_mrr.values() for v in vs]

    return {
        "hit_at_1": sum(overall_hit1) / len(overall_hit1),
        "hit_at_5": sum(overall_hit5) / len(overall_hit5),
        "mrr": sum(overall_mrr) / len(overall_mrr),
        "n": len(overall_hit5),
        "per_category_hit_at_5": {k: sum(v) / len(v) for k, v in per_category_hit5.items()},
    }


async def main(commit: str) -> None:
    files = _git_ls_corpus_files()
    if len(files) != 111:
        print(f"WARNING: expected 111 corpus documents, found {len(files)}", file=sys.stderr)

    golden = _load_golden()
    answerable = [q for q in golden if q["type"] in ("answerable_single", "answerable_multi")]
    tenant_map = {"ten_acme": "acme", "ten_globex": "globex", "ten_meridian": "meridian"}

    all_chunks: list[IndexedChunk] = []
    total_docs_seen = 0
    docs_that_split = 0
    for relpath in files:
        text = _git_show(relpath)
        chunks = _index_document(relpath, text)
        total_docs_seen += 1
        if len(chunks) > 1:
            docs_that_split += 1
        all_chunks.extend(chunks)

    print(f"Indexed {total_docs_seen} documents -> {len(all_chunks)} chunks total.")
    print(f"Documents that produced more than one chunk: {docs_that_split}\n")

    conn = await asyncpg.connect(DSN)
    results = {}
    try:
        for i, name in enumerate(STRATEGIES):
            results[name] = await _measure_strategy(conn, i, all_chunks, answerable, tenant_map)
    finally:
        await conn.close()

    print("=== Step 2.2 — chunking strategy impact on this corpus ===\n")
    print(f"{'Strategy':34s} {'hit@1':>8s} {'hit@5':>8s} {'MRR':>8s}   n")
    for name in STRATEGIES:
        r = results[name]
        print(
            f"{name:34s} {100.0 * r['hit_at_1']:7.1f}% {100.0 * r['hit_at_5']:7.1f}% "
            f"{r['mrr']:8.3f}   {r['n']}"
        )

    Path(REPO_ROOT / "evals" / "chunking_impact_2.2_results.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "n_documents": total_docs_seen,
                "n_documents_that_split": docs_that_split,
                "n_chunks": len(all_chunks),
                "top_k": TOP_K,
                "strategies": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nWrote evals/chunking_impact_2.2_results.json")


if __name__ == "__main__":
    asyncio.run(main(_current_commit()))
