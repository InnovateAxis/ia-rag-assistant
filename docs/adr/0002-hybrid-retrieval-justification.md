# ADR-0002: Hybrid retrieval (full-text + vector), justified by a measured baseline

## Status

Accepted.

## Context

Step 1.4 asks a question that is answerable before any embedding exists:
before Phase 2 builds a vector index, how far does PostgreSQL full-text
search alone get on this project's actual golden set? The answer must be
evidence, not received wisdom — "hybrid retrieval is best practice" is not
a reason this project can defend to a client's security reviewer; a measured
number against the real 111-document corpus and the real 75-question golden
set (`evals/golden.jsonl`, step 1.2) is.

`scripts/measure_keyword_baseline.py` is the reproducible measurement. It
loads every corpus document (read from `git show HEAD:<path>`, per F22 —
never from the working tree) into a throwaway PostgreSQL table with a
generated `tsvector` column, and for each of the 48 answerable golden
questions (30 `answerable_single`, 18 `answerable_multi` — the only two
categories with an `expected_chunks` list) runs a keyword search scoped to
the asking tenant, ranked by `ts_rank`. It measures at **document**
granularity: chunking (2.2) does not exist yet, and every document in this
corpus is short enough that "the right document" and "the right chunk"
coincide for this baseline. Re-run once 2.2 lands to confirm the picture
holds at chunk granularity — a chunker that splits the detention SOP's rate
table away from its free-time sentence, for instance, could change this.

One methodology note that mattered: `plainto_tsquery` ANDs every stemmed
term in the query, which makes a natural-language question fail outright
whenever it contains a word — "allow", "apply", "charge" — that never
appears verbatim in the source document, even when the document plainly
answers the question. The measurement rewrites that AND into an OR before
searching, and ranks by `ts_rank` over the OR match. Reporting the raw
AND-query numbers instead would have overstated how badly keyword search
does, for a reason that has nothing to do with keyword search's real
ceiling — a stricter filter, not a weaker retrieval signal.

## Measured result

| Category | hit@1 | hit@5 | MRR | n |
|---|---|---|---|---|
| answerable_single | 90.0% | 100.0% | 0.931 | 30 |
| answerable_multi | 94.4% | 100.0% | 0.963 | 18 |
| **Overall answerable** | **91.7%** | **100.0%** | **0.943** | **48** |

Full detail, per question, is committed at `evals/baseline_1.4_results.json`
— every question's hit@1/hit@5/reciprocal-rank and its actual top-5, so this
ADR's summary table can be audited without re-running the script (which
needs a throwaway PostgreSQL container; see the script's docstring).

**hit@5 is saturated at 100%.** For this corpus specifically — 30-42
documents per tenant, deliberately built with the three cross-tenant
overlaps from 1.1 — a top-5 window is wide enough that keyword search alone
already surfaces the right document for every answerable question. That is
a real, measured property of *this* corpus, not evidence that keyword search
generalizes; the composition note in `evals/golden.jsonl`'s design (step 1.2)
did not shape questions to defeat keyword search, and it shows. The
informative signal here is rank quality (hit@1, MRR), not recall.

### Where keyword already wins

Every one of the 44 questions that landed at rank 1 did so on the strength
of an exact, rare-within-tenant term: a SOP or rate-sheet **id**
(`SOP-ACME-001`, `RS-GLOBEX-002`), a **percentage** stated to one decimal
(`15.5%`, `18.2%`, `16.8%` — 1.1's fuel-surcharge overlap, deliberately
different per tenant), a **dollar figure** (`$2850`, `$50 per 8-hour
period`), or a **shipment id** (`SHP-ACME-20240817-0002`). These are exactly
the categories the runbook names as keyword's strength — exact identifiers,
rare domain terms, numbers quoted verbatim — and the measurement confirms
it rather than assuming it.

### Where keyword struggles — and why hybrid alone will not fix it

The 4 questions that missed rank 1 (q008, q017, q027, q039 — one per tenant
plus a multi-hop question inheriting the same document) all point at the
same thing: **the five MSA amendment documents in each tenant's contracts
folder are byte-identical in body** (`OVERVIEW` / `SERVICE LEVELS` /
`LIABILITY` / `DISPUTE RESOLUTION`, generated from one template), differing
only by a `Document ID` field the natural-language question never names.
Keyword search still finds the right one (rank 2-4, always inside the
top 5) because a stray term ties it near the top, but it has no way to rank
it first among true near-duplicates — and **neither will an embedding
model**, because the documents are also semantically identical where it
matters. This is a corpus-modeling question for whoever owns 2.2/2.6 —
consolidate near-duplicate boilerplate at ingest, or surface the most recent
amendment by effective date, or accept that "which amendment" is not a
retrieval problem at all — not a gap hybrid retrieval in Phase 3 closes by
itself.

### The stronger argument for hybrid: what keyword cannot do by construction

The measurement above only covers questions with a right answer to find.
The same script also ran the 7 `cross_tenant_trap` questions — asking, in
each asker's own tenant scope, about a fact that belongs to a *different*
tenant — through the identical keyword search, restricted correctly to the
asker's own documents (the way RLS will restrict it for real). **All 7 of
7 still returned a top-1 match, confidently ranked, with no relevance to the
question asked:**

> "What is Globex Logistics's fuel surcharge?" (asked as `ten_acme`) →
> `corpus/acme/shipments/shipment_0004.txt` — not even Acme's own fuel
> policy, just a shipment record whose `Fuel Surcharge` line shared enough
> stemmed terms to rank first.

Keyword search has no notion of "nothing here answers this." RLS
(Invariant 1) guarantees the search never *sees* Globex's row when Acme
asks — that boundary holds regardless of what retrieval algorithm runs on
top of it — but it does nothing to stop the search from serving up Acme's
own unrelated content with a positive score. A retrieval layer, keyword or
vector or both, will always rank *something* if asked to; recognizing that
the something is not an answer is not a retrieval-quality problem to
optimize away; it is the reason **refusal happens in the orchestrator on a
measured score threshold, before generation (step 4.1), never inferred from
an empty result set.** This is the sharper justification for the work in
Phase 3-4: not "keyword recall is too low" — on this corpus it is not — but
"keyword has no calibrated confidence signal to refuse on, and no way to
recognize a wrong-tenant near-miss from a right-tenant answer." Vector
similarity, reranking, and an explicit score floor exist to supply that
signal, not to out-recall keyword search on documents keyword already
finds.

## Decision

Keep the full-text index (`document_chunks`'s existing GIN/tsvector path)
as a first-class retrieval path alongside vector search, fused at
retrieval time (3.3, RRF) rather than treating full-text as a fallback.
Concretely:

1. **Do not remove or downgrade full-text search once vector search
   exists.** On this corpus it already resolves 44 of 48 answerable
   questions to rank 1 by itself; a hybrid design that only consults it
   when vector search fails would throw that away.
2. **Do not expect hybrid retrieval to resolve the MSA near-duplicate
   case.** Route that to 2.2 (chunking/ingest) or 2.6 as a
   content-modeling decision, not a retrieval-tuning one.
3. **Refusal is an orchestration decision on a score threshold (4.1), not
   a retrieval-layer inference from empty or low-confidence results.**
   Both keyword and vector retrieval will confidently return *something*
   for a cross-tenant trap or an absent-but-plausible question; nothing
   short of the orchestrator's explicit refusal branch catches that.

## Consequences

- Any future step that touches full-text retrieval (3.2) must re-run
  `scripts/measure_keyword_baseline.py` — or its chunk-granularity
  successor once 2.2 lands — before changing the balance between keyword
  and vector weight in RRF (3.3); this ADR's numbers are the baseline that
  change is measured against, not received wisdom to defer to indefinitely.
- The MSA near-duplicate finding is carried forward for whoever owns 2.2:
  see `runbook/CARRYFORWARD.md`.
- The cross-tenant confident-wrong-match finding is carried forward for
  4.1's refusal orchestrator design: see `runbook/CARRYFORWARD.md`.
