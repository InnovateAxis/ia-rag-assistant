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

**Corrected after user review.** An earlier version of this script (and of
`chunk_fixed_chars_naive` itself) applied rule 2's 200-token floor to row 1
as if it were universal, which made row 1 silently identical to rows 2-3 for
the wrong reason — character length and token count diverge on this corpus
(max document: 649 characters, 127 tokens), so a *character*-based cutoff at
512 engages even though no *token*-based cutoff at 512 does. `rules.py`'s
`chunk_fixed_chars_naive` is now genuinely naive (hard 512-character
windows, no floor), which means — unlike rows 2-5 — row 1 DOES split
documents on this corpus, at a different chunk count per document than
`evals/golden.jsonl`'s `#1`-only placeholder assumes (CARRYFORWARD F23).
Scoring row 1 against `expected_chunks` unmodified would silently score a
"miss" whenever the answer happens to fall in a later naive chunk than the
first — not because retrieval failed, but because the ground truth id was
wrong for this row. `_row1_expected_chunk_ids` below re-derives, per
question, which of THAT document's actual naive chunks contain the
question's `expected_answer_contains` text, and scores row 1 against that
instead. Rows 2-5 are untouched: every document still produces exactly one
chunk under the token-based and structure-aware strategies, so
`golden.jsonl`'s own `#1` id is still exactly right for them.

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
                                         (one row PER naive chunk — a
                                         document may produce several)
  2. Fixed 512 tokens, sentence-safe -> `chunk_fixed_tokens_sentence_safe(text)[0].text`
  3. Section-aware, no path prefix   -> `chunk_structure_aware(text)[0].text`
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
class SharedChunk:
    """One entry per document — rows 2-5's world, where every document is
    still exactly one chunk (CARRYFORWARD F36)."""

    chunk_id: str
    tenant_slug: str
    texts: tuple[str, str, str, str]  # row2, row3, row4, row5


@dataclass(frozen=True)
class NaiveChunk:
    """One entry per NAIVE chunk — row 1's world, where a document can be
    more than one row here."""

    chunk_id: str
    doc_id: str
    tenant_slug: str
    text: str


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


def _index_document(relpath: str, text: str) -> tuple[SharedChunk, list[NaiveChunk]]:
    # corpus/<tenant>/<category>/<file>.txt
    _, tenant_slug, category, filename = relpath.split("/")
    stem = filename.removesuffix(".txt")
    doc_id = f"{tenant_slug}/{category}/{stem}"

    sentence_safe = chunk_fixed_tokens_sentence_safe(text)
    structure_aware = chunk_structure_aware(text)
    if len(sentence_safe) != 1 or len(structure_aware) != 1:
        # Still a real invariant for the two floor-respecting strategies
        # (CARRYFORWARD F36) — unlike row 1, rows 2 and 3 are token-based
        # and every document here is under the 200-token floor.
        raise AssertionError(
            f"{relpath}: expected exactly one chunk from the token-based "
            f"strategies (sentence_safe={len(sentence_safe)}, "
            f"structure_aware={len(structure_aware)})"
        )

    row4_text = _embedded_text(structure_aware[0])
    metadata_suffix = _metadata_suffix(text)
    row5_text = f"{row4_text}\n\n{metadata_suffix}" if metadata_suffix else row4_text

    shared = SharedChunk(
        chunk_id=golden_chunk_id(tenant_slug, category, stem, 0),
        tenant_slug=tenant_slug,
        texts=(sentence_safe[0].text, structure_aware[0].text, row4_text, row5_text),
    )

    naive = chunk_fixed_chars_naive(text)
    naive_chunks = [
        NaiveChunk(
            chunk_id=golden_chunk_id(tenant_slug, category, stem, i),
            doc_id=doc_id,
            tenant_slug=tenant_slug,
            text=c.text,
        )
        for i, c in enumerate(naive)
    ]
    return shared, naive_chunks


def _row1_expected_chunk_ids(
    question: dict, naive_by_doc: dict[str, list[NaiveChunk]]
) -> set[str]:
    """Row 1 splits documents; `golden.jsonl`'s pinned `#1` id (F23's
    placeholder for "the document's only chunk") is not necessarily the
    naive chunk that actually contains the answer. Re-derive ground truth
    for THIS row by checking, per referenced document, which of its real
    naive chunks contain the question's `expected_answer_contains` text.
    An empty result is a legitimate outcome, not an error — it means the
    naive split cut through the answer (or across the boundary of the
    substring itself), which is a real thing a naive splitter can do and a
    real miss for this row, not a bug to paper over."""
    expected: set[str] = set()
    substrings = question.get("expected_answer_contains", [])
    referenced_docs = {cid.split("#", 1)[0] for cid in question.get("expected_chunks", [])}
    for doc_id in referenced_docs:
        for nc in naive_by_doc.get(doc_id, []):
            if any(sub in nc.text for sub in substrings):
                expected.add(nc.chunk_id)
    return expected


async def _run_search(
    conn: asyncpg.Connection, tenant_slug: str, question_text: str
) -> list[str]:
    and_query = await conn.fetchval("select plainto_tsquery('english', $1)::text", question_text)
    or_query_text = and_query.replace(" & ", " | ") if and_query else ""
    if not or_query_text:
        return []
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
    return [r["chunk_id"] for r in rows]


def _score(
    answerable: list[dict],
    tenant_map: dict[str, str],
    ranked_ids_by_qid: dict[str, list[str]],
    expected_ids_by_qid: dict[str, set[str]],
) -> dict:
    per_category_hit1: dict[str, list[bool]] = defaultdict(list)
    per_category_hit5: dict[str, list[bool]] = defaultdict(list)
    per_category_mrr: dict[str, list[float]] = defaultdict(list)

    for q in answerable:
        ranked = ranked_ids_by_qid[q["id"]]
        expected = expected_ids_by_qid[q["id"]]
        per_category_hit1[q["type"]].append(hit_rate_at_k(ranked, expected, k=1))
        per_category_hit5[q["type"]].append(hit_rate_at_k(ranked, expected, k=TOP_K))
        per_category_mrr[q["type"]].append(mrr(ranked, expected))

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


async def _measure_shared_strategy(
    conn: asyncpg.Connection,
    strategy_index: int,  # 0..3 for rows 2..5
    shared_chunks: list[SharedChunk],
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
        [(c.chunk_id, c.tenant_slug, c.texts[strategy_index]) for c in shared_chunks],
    )
    await conn.execute("create index on chunks_baseline using gin (ts)")

    ranked_by_qid = {}
    expected_by_qid = {}
    for q in answerable:
        tenant_slug = tenant_map[q["tenant"]]
        ranked_by_qid[q["id"]] = await _run_search(conn, tenant_slug, q["question"])
        expected_by_qid[q["id"]] = set(q["expected_chunks"])

    return _score(answerable, tenant_map, ranked_by_qid, expected_by_qid)


async def _measure_row1(
    conn: asyncpg.Connection,
    all_naive: list[NaiveChunk],
    naive_by_doc: dict[str, list[NaiveChunk]],
    answerable: list[dict],
    tenant_map: dict[str, str],
) -> tuple[dict, int]:
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
        [(c.chunk_id, c.tenant_slug, c.text) for c in all_naive],
    )
    await conn.execute("create index on chunks_baseline using gin (ts)")

    ranked_by_qid = {}
    expected_by_qid = {}
    n_undecidable = 0
    for q in answerable:
        tenant_slug = tenant_map[q["tenant"]]
        ranked_by_qid[q["id"]] = await _run_search(conn, tenant_slug, q["question"])
        expected = _row1_expected_chunk_ids(q, naive_by_doc)
        if not expected:
            n_undecidable += 1
        expected_by_qid[q["id"]] = expected

    return _score(answerable, tenant_map, ranked_by_qid, expected_by_qid), n_undecidable


async def main(commit: str) -> None:
    files = _git_ls_corpus_files()
    if len(files) != 111:
        print(f"WARNING: expected 111 corpus documents, found {len(files)}", file=sys.stderr)

    golden = _load_golden()
    answerable = [q for q in golden if q["type"] in ("answerable_single", "answerable_multi")]
    tenant_map = {"ten_acme": "acme", "ten_globex": "globex", "ten_meridian": "meridian"}

    shared_chunks: list[SharedChunk] = []
    all_naive: list[NaiveChunk] = []
    naive_by_doc: dict[str, list[NaiveChunk]] = {}
    docs_that_split_naively = 0

    for relpath in files:
        text = _git_show(relpath)
        shared, naive_chunks = _index_document(relpath, text)
        shared_chunks.append(shared)
        doc_id = naive_chunks[0].doc_id
        naive_by_doc[doc_id] = naive_chunks
        all_naive.extend(naive_chunks)
        if len(naive_chunks) > 1:
            docs_that_split_naively += 1

    print(f"Indexed {len(files)} documents.")
    print(f"Row 1 (naive 512-char): {len(all_naive)} chunks total; "
          f"{docs_that_split_naively} of {len(files)} documents split into more than one.")
    print(f"Rows 2-5 (token/structure-based): {len(shared_chunks)} chunks total "
          f"(one per document — none split, CARRYFORWARD F36).\n")

    conn = await asyncpg.connect(DSN)
    results: dict[str, dict] = {}
    row1_undecidable = 0
    try:
        results[STRATEGIES[0]], row1_undecidable = await _measure_row1(
            conn, all_naive, naive_by_doc, answerable, tenant_map
        )
        for i, name in enumerate(STRATEGIES[1:], start=0):
            results[name] = await _measure_shared_strategy(
                conn, i, shared_chunks, answerable, tenant_map
            )
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
    print(
        f"\nRow 1 questions where no naive chunk contained the expected "
        f"answer text at all: {row1_undecidable} of {len(answerable)}."
    )

    Path(REPO_ROOT / "evals" / "chunking_impact_2.2_results.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "n_documents": len(files),
                "n_documents_that_split_under_row1_naive": docs_that_split_naively,
                "n_row1_naive_chunks": len(all_naive),
                "n_shared_chunks_rows_2_to_5": len(shared_chunks),
                "row1_questions_with_no_derivable_ground_truth": row1_undecidable,
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
