"""Step 2.2 — proving carry-forward F23/F36 against the real corpus, rather
than asserting them in prose.

F23: `evals/golden.jsonl`'s `expected_chunks` ids
(`<tenant>/<category>/<file-stem>#1`) are a placeholder meaning "the
document's first chunk". F36 measured (at `6affe26`) that every one of the
111 corpus documents is short enough that no chunker respecting rule 2's
200-token floor can split it. This file re-derives that on the committed
corpus rather than trusting the earlier measurement, and asserts the id
convention actually matches what the real chunker produces.

Reads `corpus/` from the working tree (not `git show HEAD:`) — these tests
run in CI against a checked-out tree, the same way the real ingestion
pipeline eventually will; `scripts/measure_chunking_impact.py` reads from
`git show` instead because it is a one-off measurement run by hand against a
specific commit (same reasoning as `scripts/measure_keyword_baseline.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.chunking.rules import (
    TableBlock,
    chunk_structure_aware,
    golden_chunk_id,
    parse_structure,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_ROOT = REPO_ROOT / "corpus"
GOLDEN_PATH = REPO_ROOT / "evals" / "golden.jsonl"

_TENANT_SLUG = {"ten_acme": "acme", "ten_globex": "globex", "ten_meridian": "meridian"}


def _all_corpus_relpaths() -> list[str]:
    return sorted(
        str(p.relative_to(CORPUS_ROOT)).replace("\\", "/")
        for p in CORPUS_ROOT.rglob("*.txt")
    )


def _golden_referenced_doc_ids() -> set[str]:
    """Every distinct `<tenant>/<category>/<file-stem>` referenced anywhere
    in `expected_chunks`, ignoring the `#N` suffix."""
    ids: set[str] = set()
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        for chunk_id in q.get("expected_chunks", []):
            ids.add(chunk_id.split("#", 1)[0])
    return ids


def test_precondition_corpus_has_111_documents():
    """If this fails, every test below is checking the wrong corpus (same
    C6/F17 lesson: a check that can pass vacuously over the wrong or an
    empty input set proves nothing)."""
    assert len(_all_corpus_relpaths()) == 111


def test_precondition_golden_set_references_documents_that_exist():
    referenced = _golden_referenced_doc_ids()
    assert len(referenced) > 0
    all_paths = set(_all_corpus_relpaths())
    for doc_id in referenced:
        assert f"{doc_id}.txt" in all_paths, f"golden.jsonl references missing document {doc_id}"


def test_every_golden_referenced_document_chunks_to_exactly_one_chunk():
    """The F23 resolution: prove it, don't assert it. If any referenced
    document ever splits into more than one chunk, its golden `#1` id would
    silently stop meaning "the whole document" and this test is the one
    that must catch it before a chunker change ships."""
    referenced = _golden_referenced_doc_ids()
    assert len(referenced) >= 30, "expected roughly 33 distinct referenced documents (F23)"

    for doc_id in sorted(referenced):
        text = (CORPUS_ROOT / f"{doc_id}.txt").read_text(encoding="utf-8")
        chunks = chunk_structure_aware(text)
        assert len(chunks) == 1, (
            f"{doc_id} produced {len(chunks)} chunks — golden.jsonl's "
            f"'{doc_id}#1' placeholder no longer means 'the whole document'"
        )


def test_golden_chunk_ids_match_what_the_real_chunker_produces():
    """Not just "one chunk" — the exact id string golden.jsonl pins."""
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        for expected_id in q.get("expected_chunks", []):
            doc_id, _, suffix = expected_id.partition("#")
            tenant_slug, category, stem = doc_id.split("/")
            chunk_index = int(suffix) - 1
            assert golden_chunk_id(tenant_slug, category, stem, chunk_index) == expected_id


def test_every_corpus_document_chunks_to_exactly_one_chunk():
    """The full F36 claim, not just the golden-referenced subset: all 111,
    not only the ~33 golden.jsonl happens to reference."""
    for relpath in _all_corpus_relpaths():
        text = (CORPUS_ROOT / relpath).read_text(encoding="utf-8")
        chunks = chunk_structure_aware(text)
        assert len(chunks) == 1, f"{relpath} produced {len(chunks)} chunks, expected 1"


def test_no_table_is_ever_split_across_the_real_corpus():
    """Rule 3, over real data. Every rate-sheet-shaped table (`Zone A/B/C`
    blocks) is trivially inside its document's single chunk (see the test
    above) — the non-trivial part, and the part that would actually catch a
    regression, is confirming the parser finds a meaningful number of real
    tables in the first place (C6's lesson: a loop over zero rows asserts
    nothing and reports green)."""
    tables_found = 0
    for relpath in _all_corpus_relpaths():
        text = (CORPUS_ROOT / relpath).read_text(encoding="utf-8")
        parsed = parse_structure(text)
        doc_tables = [b for s in parsed.sections for b in s.blocks if isinstance(b, TableBlock)]
        tables_found += len(doc_tables)

        chunks = chunk_structure_aware(text)
        assert len(chunks) == 1
        for table in doc_tables:
            assert table.text in chunks[0].text, f"{relpath}: table not intact in its chunk"

    assert tables_found > 50, (
        f"expected the rate-sheet Zone A/B/C blocks to be detected as tables "
        f"across the corpus; found {tables_found}"
    )


@pytest.mark.parametrize("category_glob", ["*/rate_sheets/*.txt"])
def test_rate_sheets_with_zone_pricing_are_detected_as_containing_a_table(category_glob):
    """A tighter, non-vacuous positive control on top of the count guard
    above: specifically the zone-priced rate sheets (4 of every 5, per
    `scripts/generate_corpus.py`'s `sheet_types` cycle — every type except
    "Fuel Surcharge Policy") must parse with at least one table block."""
    zone_priced = [
        p for p in CORPUS_ROOT.glob(category_glob) if "Zone A" in p.read_text(encoding="utf-8")
    ]
    # 33 rate sheets total (12 + 11 + 10), 4 of every 5 `sheet_types` are
    # zone-priced (all but "Fuel Surcharge Policy") -> ~26; 25 observed.
    assert len(zone_priced) > 20

    for path in zone_priced:
        parsed = parse_structure(path.read_text(encoding="utf-8"))
        doc_tables = [b for s in parsed.sections for b in s.blocks if isinstance(b, TableBlock)]
        assert doc_tables, f"{path.name}: expected at least one table block"
