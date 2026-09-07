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

**Corrected after ia-verifier caught the first version of this section
overstating its own mechanism** (`ts_rank` scores a document against the
*query's* stemmed terms — a value that appears only in the answer, never in
the question, cannot have driven the document's rank; six of the seven
example terms first written here were exactly that: answer values quoted
from `expected_answer_contains`, appearing in zero of the 75 questions).
What actually drove each rank-1 result, checked directly rather than
asserted:

- **8 of the 44** rank-1 questions (q010, q020, q030, q031, q034, q037, q043,
  q044) contain a literal exact identifier in the question itself — a
  shipment id (`SHP-ACME-20240817-0002`, `SHP-GLOBEX-20240817-0002`, ...) —
  and those 8 rank first because that string is unique within the tenant's
  corpus. This is the runbook's "exact identifiers" case, confirmed, but it
  is 8 questions, not 44.
- **5 of the 44** rank-1 questions (q031, q035, q037, q043, q045) quote a
  dollar figure lifted verbatim from the record they ask about — an
  arithmetic-check phrasing ("Shipment SHP-...-0002 was charged a $X fuel
  surcharge on an $820 base freight charge — does that match our policy?",
  or "what would the fuel surcharge be on a ... charge of $Y?") that
  restates a value from the document before asking about it: q031 quotes
  `$127.10` and `$820`, both verbatim in the rank-1
  `corpus/acme/shipments/shipment_0002.txt`; q037 quotes `$149.24` and
  `$820`, verbatim in `corpus/globex/shipments/shipment_0002.txt`; q043
  quotes `$137.76` and `$820`, verbatim in
  `corpus/meridian/shipments/shipment_0002.txt`; q035 quotes `$1600`,
  verbatim in `corpus/acme/rate_sheets/rate_sheet_02.txt`; q045 quotes
  `$1200`, verbatim in `corpus/meridian/rate_sheets/rate_sheet_03.txt`. By
  this section's own `ts_rank` reasoning — a value that appears in the
  question can drive the rank of the document it also appears in — these 5
  are figure-driven, not domain-vocabulary-driven.
- **These two groups overlap, not partition, the 44**: q031, q037 and q043
  each contain both a shipment id and a dollar figure — the "Shipment
  SHP-...-0002 was charged $X ... does that match our policy?" questions
  name the record twice, once by id and once by the figure it holds. So of
  the 8 identifier questions, 5 (q010, q020, q030, q034, q044) carry no
  figure and 3 (q031, q037, q043) carry both; of the 5 figure questions, 2
  (q035, q045) carry no identifier and the same 3 carry both.
- **The remaining 34** questions carry neither a literal identifier nor a
  quoted figure and rank first on ordinary domain vocabulary that is rare
  *within a 30-42 document tenant corpus* even though it is not a formal
  identifier: "detention", "hazmat", "kestrel haulage", "accessorial",
  "zone pricing", "claims processing", "temperature control", "customs
  documentation". Each of these terms is the corpus's own generator-cycle
  category name, so it appears in the 1-3 documents of that category and
  nowhere else in the tenant's other 27-39 documents — rare enough to rank
  decisively without being an identifier in the runbook's narrower sense.
  This is still "rare domain terms," the runbook's second predicted
  category — just not the same mechanism as an id or a number. (8 + 5 − 3
  overlap + 34 = 44.)
- **Percentages and SOP/rate-sheet/MSA ids**, as opposed to dollar figures,
  are still never asked for by name in any of the 44 — that half of the
  original claim holds; only the dollar-figure half was wrong. Precision
  matters here because this ADR's Consequences bind future RRF-weight
  decisions (3.2/3.3) to this section, not to a restated intuition.

### Where keyword struggles — and why hybrid alone will not fix it

The 4 questions that missed rank 1 (q008, q017, q027, q039 — one per tenant
plus a multi-hop question inheriting the same document) all point at the
same thing: each tenant's contracts folder holds several MSA amendment
documents generated from one template (`OVERVIEW` / `SERVICE LEVELS` /
`LIABILITY` / `DISPUTE RESOLUTION`, byte-identical across all of them) —
**Acme has 5, Globex has 4, Meridian has 3**, not five in every tenant as
an earlier version of this ADR and of `runbook/CARRYFORWARD.md` F24 both
said, caught by ia-verifier against the actual committed tree.

**The real mechanism is the title line, not an absence of any distinguishing
signal.** Each MSA amendment is titled either `Amendment` or `Master Service
Agreement` — for Acme, `01` and `07` are `Amendment`; `02`, `06` and `08` are
`Master Service Agreement`. Every one of the 4 missed questions asks about
"our master service agreement" — a phrase that matches the *title* of the
majority of amendments verbatim, while the document `evals/golden.jsonl`
happens to pin as the expected chunk (`msa_amendment_01`) is titled
`Amendment`. Keyword search is not failing to distinguish indistinguishable
documents here; it is correctly ranking the documents whose title the
question actually used ahead of the one the golden set arbitrarily pinned.
The four `LIABILITY`/`SERVICE LEVELS` body sections are genuinely identical
and carry no distinguishing content — that part of the original diagnosis
holds — but the title does distinguish these documents, just not toward the
pinned one. In every one of the 4 cases the top-ranked (wrong) document
contains the same expected answer phrase verbatim (`4 business hours`,
`24/7 hotline`, `Meridian Transport's operational guidelines`) — the answer
was available at rank 1 in all 4 cases, just attached to a different
Document ID than the one pinned. hit@1 = 91.7% therefore understates how
often a right answer was actually available at the top of the ranking.

This is still a corpus-modeling question for whoever owns 2.2/2.6, but the
shape of the fix is narrower than "consolidate indistinguishable documents":
either the golden set should accept any same-titled amendment as a correct
citation for a title-matching question (a golden-set fix, not a retrieval
fix), or ingest should tag amendments by effective date/version so "our
master service agreement" resolves to a specific current one rather than
whichever amendment's title happens to match best. Neither is a gap hybrid
retrieval in Phase 3 closes by itself, since an embedding model asked "what
does our master service agreement say" has the same title-driven ambiguity
to resolve.

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
