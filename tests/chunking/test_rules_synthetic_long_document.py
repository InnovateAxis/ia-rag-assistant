"""Step 2.2 — the multi-chunk path.

CARRYFORWARD F36: every real corpus document is under the 200-token floor,
so `chunk_structure_aware`/`chunk_fixed_tokens_sentence_safe`/
`chunk_fixed_chars_naive` all return exactly one chunk on real data — see
`tests/chunking/test_corpus_chunking.py`. That proves the strategies never
DISAGREE here; it does not prove any of them can actually split, respect a
table boundary, or produce a sentence-safe 15% overlap. This file is
synthetic, well above every threshold in `rules.py`, built specifically to
exercise rules 2-4 for real. It is not corpus data and does not enlarge the
corpus — the user's explicit instruction not to do either applies to
`corpus/`, not to a string constructed inside a test.
"""

from __future__ import annotations

from src.chunking.rules import (
    CEILING_TOKENS,
    FLOOR_TOKENS,
    _split_sentences,
    chunk_fixed_chars_naive,
    chunk_fixed_tokens_sentence_safe,
    chunk_structure_aware,
    count_tokens,
)

_FILLER_SENTENCE = (
    "This paragraph exists only to push the synthetic document well past "
    "the two hundred token floor and the nine hundred token ceiling so the "
    "chunking rules actually have to make a splitting decision on it."
)


def _long_section(heading: str, n_sentences: int) -> str:
    body = " ".join(f"{_FILLER_SENTENCE} Point {i}." for i in range(n_sentences))
    return f"{heading}\n{body}\n"


_TABLE = """RATE TABLE

Zone A (0-100 miles):
  LTL: $0.85/lb minimum $85
  Full: $1200 per load

Zone B (100-300 miles):
  LTL: $0.95/lb minimum $120
  Full: $1600 per load

Zone C (300-600 miles):
  LTL: $1.10/lb minimum $180
  Full: $2200 per load

Zone D (600-1000 miles):
  LTL: $1.25/lb minimum $220
  Full: $2800 per load
"""

SYNTHETIC_DOC = (
    "SYNTHETIC LONG POLICY DOCUMENT\n\n"
    "Effective Date: 2024-01-01\n"
    "Tenant: Test Co\n\n"
    + _long_section("OVERVIEW", 20)
    + "\n"
    + _long_section("SCOPE", 20)
    + "\n"
    + _TABLE
    + "\n"
    + _long_section("PROCEDURES", 20)
    + "\n"
    + _long_section("COMPLIANCE", 20)
)


def test_synthetic_document_is_actually_long_enough_to_force_a_split():
    """A precondition on the fixture itself — if this fails, the test below
    proves nothing, the same lesson carry-forward C6 records about an empty
    loop asserting nothing."""
    assert count_tokens(SYNTHETIC_DOC) > 3 * CEILING_TOKENS
    assert len(chunk_structure_aware(SYNTHETIC_DOC)) > 1


def test_structure_aware_splits_into_multiple_chunks_respecting_the_floor():
    chunks = chunk_structure_aware(SYNTHETIC_DOC)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count >= FLOOR_TOKENS, (
            f"chunk under the floor: {c.token_count} tokens"
        )


def test_structure_aware_never_splits_the_table():
    """The table must never be torn apart: no chunk may contain some of
    the four zone markers but not the rest. It is fine — expected, even —
    for the table to appear whole in two consecutive chunks when rule 4's
    overlap carries a chunk's tail into the next chunk's head; that is
    duplication, not a split."""
    chunks = chunk_structure_aware(SYNTHETIC_DOC)

    table_lines = [
        "Zone A (0-100 miles):",
        "Zone B (100-300 miles):",
        "Zone C (300-600 miles):",
        "Zone D (600-1000 miles):",
    ]
    whole = [c for c in chunks if all(line in c.text for line in table_lines)]
    partial = [
        c
        for c in chunks
        if c not in whole and any(line in c.text for line in table_lines)
    ]
    assert whole, "expected the four-zone table to appear whole in at least one chunk"
    assert not partial, (
        "found a chunk containing some but not all of the table's zone "
        "markers — the table was split"
    )


def test_naive_char_split_can_cut_through_the_table_where_structure_aware_cannot():
    """The point of comparison: row 1 (naive) is blind to structure, so on
    text long enough to force a split, it is free to land mid-table — which
    is exactly what rule 3 exists to prevent, and exactly what
    `chunk_structure_aware` was just shown not to do above."""
    naive_chunks = chunk_fixed_chars_naive(SYNTHETIC_DOC, size=300)

    table_lines = [
        "Zone A (0-100 miles):",
        "Zone B (100-300 miles):",
        "Zone C (300-600 miles):",
        "Zone D (600-1000 miles):",
    ]
    whole_in_one = any(all(line in c.text for line in table_lines) for c in naive_chunks)
    assert not whole_in_one, (
        "expected the naive 600-char splitter to cut through the table on "
        "this synthetic document; if it didn't, this test no longer "
        "demonstrates what rule 3 protects against and should be revisited"
    )


def test_fixed_tokens_sentence_safe_boundaries_never_land_mid_sentence():
    chunks = chunk_fixed_tokens_sentence_safe(SYNTHETIC_DOC, size=300)

    assert len(chunks) > 1
    for c in chunks:
        text = c.text.strip()
        assert text.endswith((".", "!", "?")), (
            f"chunk does not end on a sentence boundary: ...{text[-40:]!r}"
        )


def test_fixed_tokens_sentence_safe_overlap_is_roughly_fifteen_percent_at_a_sentence():
    chunks = chunk_fixed_tokens_sentence_safe(SYNTHETIC_DOC, size=300)
    assert len(chunks) > 1

    for i in range(len(chunks) - 1):
        prev_sentences = _split_sentences(chunks[i].text)
        next_first_sentence = _split_sentences(chunks[i + 1].text)[0]
        assert next_first_sentence in prev_sentences[-3:], (
            "expected the next chunk to open on one of the previous "
            "chunk's last few sentences (rule 4's overlap), at a sentence "
            "boundary rather than a mid-clause cut"
        )


def test_naive_char_split_ignores_sentence_boundaries():
    """Confirms row 1 really is the "no structure" baseline the case-study
    table names it as — if it also respected sentences, rows 1 and 2 would
    not be distinguishable strategies."""
    chunks = chunk_fixed_chars_naive(SYNTHETIC_DOC, size=300)
    assert len(chunks) > 1
    mid_word_cuts = sum(
        1
        for c in chunks[:-1]
        if c.text and c.text[-1].isalnum() and not c.text.rstrip().endswith((".", "!", "?"))
    )
    assert mid_word_cuts > 0, (
        "expected at least one naive chunk boundary to land mid-word/"
        "mid-sentence; if none did, this corpus no longer distinguishes "
        "the naive strategy from the sentence-safe one"
    )
