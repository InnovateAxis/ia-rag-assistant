# Step 2.2 — chunking strategy impact, measured on this corpus

Produced by `scripts/measure_chunking_impact.py` at commit `6affe268`. Raw
numbers: `evals/chunking_impact_2.2_results.json`.

| Strategy                          | hit@1 | hit@5 | MRR   | n  |
|------------------------------------|------:|------:|------:|---:|
| Fixed 512 chars, no structure      | 89.6% | 91.7% | 0.916 | 48 |
| Fixed 512 tokens, sentence-safe    | 89.6% | 91.7% | 0.916 | 48 |
| Section-aware, no path prefix      | 89.6% | 91.7% | 0.916 | 48 |
| Section-aware + path prefix        | 85.4% | 95.8% | 0.894 | 48 |
| Section-aware + path + metadata    | 91.7% | 100.0%| 0.943 | 48 |

## Why rows 1-3 are identical, not approximately close

**Measured, not assumed: 0 of the corpus's 111 documents produced more than
one chunk, under any of the three strategies.** Every document's total
token count is below `FLOOR_TOKENS + TARGET_MIN_TOKENS`
(`src/chunking/rules.py`), so all three strategies take the same
floor-driven "coalesce to one chunk" branch and return the exact same
single-chunk text — the same boundaries because there are nothing to draw
boundaries between (CARRYFORWARD F36; the largest document is 127 tokens).
The three rows' `hit@1`/`hit@5`/MRR figures above are bit-identical, not
merely close, which is the tell that this was actually measured rather than
assumed — see F36's own "watch for on verification" note.

Rule 3 (never split a table) and rule 4 (15% sentence-boundary overlap)
never fire on this corpus for the same reason — nothing here is long enough
to reach a second chunk. Both rules are exercised for real, on synthetic
text built specifically to reach them, in
`tests/chunking/test_rules_synthetic_long_document.py`; that test suite is
where "never split a table" and "15% overlap at a sentence" are actually
proven, not this table.

## Why rows 4 and 5 differ from 1-3, and from each other

Rows 1-3 search each chunk's raw `content` only — no title, no heading, no
metadata (the document title lives in `section_path`, not `content`; see
`src/chunking/rules.py`'s and `src/llm/embeddings.py`'s module docstrings
for why). Row 4 additionally searches the section-path breadcrumb (title,
and heading if the chunk has one) prepended exactly the way
`src/llm/embeddings.py::_embed_text` builds the text it sends for embedding
— rule 5. Row 5 additionally appends the document's parsed front-matter
(effective date, revision, document id — rule 6) as plain text; this
metadata has no `document_chunks` column to live in (see
`src/llm/embeddings.py`'s docstring), so this row's construction exists only
in this measurement script, never in production ingestion.

Row 4's hit@1 (85.4%) is genuinely *lower* than rows 1-3, not a measurement
error: PostgreSQL's `ts_rank` weights term frequency, and prepending a title
that repeats words already present in the body dilutes the rank-1 term
match for some questions while broadening recall enough to raise hit@5
(95.8%). Row 5 recovers and improves on both, because appending the
metadata block reconstructs a superset of what the original whole-document
text contained (title, dates, ids, body) — its numbers converge on
`evals/baseline_1.4_results.json`'s own `overall_answerable_*` figures
(91.7% / 100% / 0.943) for exactly that reason: row 5's searched text is,
in different order, close to the same information 1.4 searched over the
raw file. Coincidence in appearance, not in mechanism.

## Methodology, stated plainly

Every row is measured by the *same* mechanism 1.4 used: PostgreSQL
`plainto_tsquery`/`ts_rank` full-text search, term-OR rewritten the same
way 1.4's script rewrites it, scoped to the asking tenant, against a
throwaway table (never `document_chunks` — this script must not become a
second, undocumented retrieval path, same reasoning
`scripts/measure_keyword_baseline.py` states for itself). Only the
**searched text** varies by row. This is a real, measurable proxy for how
each strategy's embedded text would differ semantically, not a substitute
for measuring embeddings directly — there is no embedding provider API key
configured in this environment, and fabricating similarity scores from a
fake embedder would misrepresent them as real.

**Not being claimed here:** that chunking gave a bigger improvement than
any later retrieval change made on this project. That is the runbook's own
commentary about a different, larger corpus. On this corpus, with every
document short enough to never split, the honest finding is closer to the
opposite — the four strategies mostly differ by how much of the original
document's *identifying* text (title, metadata) ends up searchable at all,
not by chunk boundaries, because there are none to compare.
