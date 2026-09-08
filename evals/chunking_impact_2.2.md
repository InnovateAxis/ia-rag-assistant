# Step 2.2 — chunking strategy impact, measured on this corpus

Produced by `scripts/measure_chunking_impact.py` at commit `bccca8bd`. Raw
numbers: `evals/chunking_impact_2.2_results.json`.

| Strategy                          | hit@1 | hit@5 | MRR   | n  |
|------------------------------------|------:|------:|------:|---:|
| Fixed 512 chars, no structure       | 52.1% | 58.3% | 0.541 | 48 |
| Fixed 512 tokens, sentence-safe    | 89.6% | 91.7% | 0.916 | 48 |
| Section-aware, no path prefix      | 89.6% | 91.7% | 0.916 | 48 |
| Section-aware + path prefix        | 85.4% | 95.8% | 0.894 | 48 |
| Section-aware + path + metadata    | 91.7% | 100.0%| 0.943 | 48 |

**Corrected after user review — see git history for the earlier, wrong
version of this table and note.** The first pass of this measurement
applied rule 2's 200-token floor inside `chunk_fixed_chars_naive` (row 1)
and, after that was removed, still built row 1 from the same
title/metadata-stripped text the structural strategies use. Both changes
made row 1 silently identical to rows 2-3 for reasons that had nothing to
do with a real naive character splitter: character length and token count
diverge on this corpus (largest document: 649 raw characters, 127 tokens),
and the document body alone (without title/metadata) never exceeds 512
characters even though 57 of 111 raw files do. `chunk_fixed_chars_naive`
now slices the raw input text directly, with no structure parsing and no
floor guard at all — see its docstring in `src/chunking/rules.py` for the
full account of both mistakes and fixes. Confirmed by grep
(`grep -rn chunk_fixed_chars_naive src/`) that nothing outside that
function and this measurement script calls it; production ingestion
(`src/chunking/__init__.py`) uses `chunk_structure_aware` only and was not
touched.

## Row 1 is now genuinely naive, and genuinely different

**Measured: 57 of the 111 documents split into more than one 512-character
chunk under row 1 (168 chunks total); 0 split under any other strategy.**
Row 1 no longer measures the same thing as rows 2-3 — it is worse on every
metric here, not by construction, by what a blind character splitter
actually does to this corpus: it cuts mid-sentence, sometimes mid-word, and
for `answerable_multi` questions specifically its hit@5 is **0.0%** (see
`evals/chunking_impact_2.2_results.json`'s `per_category_hit_at_5`) — those
questions need facts a naive split has a good chance of separating into
different chunks. Nothing about this result was adjusted to fit an
expectation; row 1 was already worse before this correction (for the wrong,
uninteresting reason) and is worse after it (for the real one). Had it come
out better, that result would stand unchanged too.

**Scoring row 1 needed one more correction beyond the splitter itself.**
`evals/golden.jsonl`'s `expected_chunks` pins `<doc>#1` as a placeholder for
"the document's only chunk" (CARRYFORWARD F23) — true for rows 2-5, where
nothing splits, but not for row 1, where a document can have several real
chunks and the answer is not guaranteed to be in the first one. Scoring row
1 against the pinned `#1` id unmodified would have counted a correct
retrieval as a miss whenever the naive split happened to put the answer in
a later chunk — an artifact of the ground-truth id, not a retrieval
failure. `scripts/measure_chunking_impact.py::_row1_expected_chunk_ids`
re-derives, per question, which of that document's actual naive chunks
contain the `expected_answer_contains` text, and scores row 1 against that
instead. **19 of 48 answerable questions had no naive chunk containing the
expected answer text at all** — a real consequence of naive splitting
(the answer's own substring can itself be cut across a chunk boundary),
counted as a legitimate miss rather than excluded.

## Why rows 2-3 are still identical

**Measured, not assumed: 0 of the corpus's 111 documents produced more than
one chunk under either the sentence-safe token strategy or the
structure-aware strategy.** Both are token-based and every document's total
token count is below `FLOOR_TOKENS + TARGET_MIN_TOKENS`
(`src/chunking/rules.py`), so both take the same floor-driven "coalesce to
one chunk" branch and return the same single-chunk text (the largest
document is 127 tokens; CARRYFORWARD F36). This part of F36's "watch for on
verification" note holds exactly as stated — for the token-based rows.

Rule 3 (never split a table) and rule 4 (15% sentence-boundary overlap)
never fire on `chunk_structure_aware` on this corpus for the same reason —
nothing here is long enough to reach a second chunk under a floor-respecting
strategy. Both rules are exercised for real, on synthetic text built
specifically to reach them, in
`tests/chunking/test_rules_synthetic_long_document.py`.

## Why rows 4 and 5 differ from 2-3, and from each other

Rows 2-3 search each chunk's raw `content` only — no title, no heading, no
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

Row 4's hit@1 (85.4%) is genuinely *lower* than rows 2-3, not a measurement
error: PostgreSQL's `ts_rank` weights term frequency, and prepending a title
that repeats words already present in the body dilutes the rank-1 term
match for some questions while broadening recall enough to raise hit@5
(95.8%). Row 5 recovers and improves on both, because appending the
metadata block reconstructs a superset of what the original whole-document
text contained (title, dates, ids, body).

## Methodology, stated plainly

Every row is measured by the *same* mechanism 1.4 used: PostgreSQL
`plainto_tsquery`/`ts_rank` full-text search, term-OR rewritten the same
way 1.4's script rewrites it, scoped to the asking tenant, against a
throwaway table (never `document_chunks` — this script must not become a
second, undocumented retrieval path, same reasoning
`scripts/measure_keyword_baseline.py` states for itself). Only the
**searched text** (and, for row 1 only, the **ground-truth chunk id**, per
the correction above) varies by row. This is a real, measurable proxy for
how each strategy's embedded text would differ semantically, not a
substitute for measuring embeddings directly — there is no embedding
provider API key configured in this environment, and fabricating similarity
scores from a fake embedder would misrepresent them as real.

**hit@5 is not uniformly saturated across these rows** (row 1: 58.3%, row
4: 95.8%, row 5: 100.0%) — it discriminates on its own here, and hit@1/MRR
are reported alongside it regardless, because whether it happens to
saturate is a property of the measurement, not something to assume before
running it.

**Not being claimed here:** that chunking gave a bigger improvement than
any later retrieval change made on this project. That is the runbook's own
commentary about a different, larger corpus. The honest finding on this
corpus is narrower: a genuinely naive character splitter measurably loses
to every token- or structure-aware strategy here, and how much of the
original document's *identifying* text (title, metadata) ends up
searchable matters more than chunk boundaries do for the strategies that
never split.
