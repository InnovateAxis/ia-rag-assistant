# 0001 — Embeddings in PostgreSQL under RLS, not a vector database

- Status: Accepted
- Deciders: Principal Engineer, Senior AI Engineer, AppSec, Shaheer

## Context
The assistant answers over each tenant's private documents. The
question every client security team asks is how we guarantee tenant
A can never retrieve tenant B's content.

A dedicated vector database (Pinecone, Weaviate, Qdrant) would work,
with isolation implemented as a metadata filter on every query. That
places the guarantee in application code: one query path that forgets
the filter leaks another customer's documents, silently, with no
error and no log entry.

## Decision
Embeddings live in `document_chunks` in the same PostgreSQL database
as the source documents, governed by the same row-level security
policies. The guarantee is enforced by the database.

## Consequences
+ Isolation is structural. A forgotten filter returns zero rows
  rather than another tenant's data.
+ One datastore to secure, back up, audit and pay for.
+ Chunks are deleted transactionally with their parent document —
  no orphaned embeddings, which is a real and common leak in
  two-store designs.
+ A client's security reviewer can read the policy as SQL.
− RLS enforces the value of `tenant_id`, not its correctness. The
  policy compares each row's `tenant_id` against the session's
  verified claim and cannot be talked out of that comparison; it has
  no opinion on whether the value was right when the row was
  written. Foreign-key checks run outside RLS. This was demonstrated
  on this project, not theorised: a session in `ten_globex`'s
  context successfully inserted a chunk whose `document_id`
  referenced one of `ten_acme`'s documents. If a chunk is ever
  mis-stamped at ingest, the policy will faithfully serve acme's
  content to globex, and every isolation test still passes, because
  RLS is working exactly as designed. What this decision buys is
  "the database enforces the tenant column", not "the tenant column
  is right".

  The mitigation is to make mis-stamping structurally hard at the
  one point where it can occur: ingest must derive `tenant_id` from
  the verified session context and never from caller-supplied data.
  That is step 2.1's contract, and it is not built yet. Until it is,
  this is an open limit rather than a closed one.
− Failing closed means failing silently, and this design obliges us
  to compensate for that permanently. The first advantage above — a
  forgotten filter returns zero rows rather than another tenant's
  data — carries an exact cost: at the query boundary, a correctly
  working system that has nothing to say and a badly broken one are
  indistinguishable. Both return an empty result. The alternative
  architecture fails differently and more loudly, because a
  metadata-filter bug in a vector store tends to return the wrong
  rows, and wrong rows get noticed. Zero rows get attributed to the
  corpus.

  The obligation this places on us is structural rather than a task
  to be completed: this design requires a positive control — an
  assertion, maintained for as long as the design stands, that each
  tenant can still read its own data — because nothing about an
  empty result distinguishes a working policy from a broken one. A
  suite that only proves tenants cannot read each other's data
  cannot tell the difference either.
− pgvector's HNSW index is slower than a purpose-built store at very
  large scale. The working figure is roughly 5M chunks per database
  on our instance class. That number is an inherited planning
  assumption this project has not tested — the corpus at the time of
  writing is 111 documents, and no measurement supporting the figure
  is on record here. It is a capacity-planning starting point; the
  revisit trigger below, not the chunk count, is what carries the
  weight.
− The isolation guarantee costs vector recall, and it costs it
  before it costs latency. PostgreSQL applies an RLS policy as a
  filter over rows the index has already returned, not as a
  condition the index itself can use. An HNSW search therefore finds
  its nearest neighbours across the whole table, and the policy then
  discards every row belonging to another tenant. For a tenant
  holding a small share of a large table, the neighbours the index
  returns can be entirely other tenants' rows, leaving the query
  with nothing. Measured on this project: forcing the planner onto
  the vector index in that shape returned zero rows, and raising
  `hnsw.ef_search` did not recover them.

  This is the honest form of the scale limit stated above. The
  ceiling is not only that HNSW slows down somewhere near some
  number of chunks — it is that recall degrades for a small tenant
  in a large table well before any such ceiling is reached, and it
  degrades quietly. Today the `(tenant_id, document_id)` index
  carries recall and the corpus is far too small for this to bite.
  That will not hold as the table grows. Steps 3.1 and 3.6 own the
  fix; it is not solved today.
− Re-embedding requires a migration rather than a reindex API call.
− We forgo managed vector-store features (hybrid built in, managed
  reranking). We implement those ourselves; see ADR-0002.

## When we would revisit
A single tenant exceeding ~5M chunks, or p95 retrieval latency
exceeding 400ms at production volume.
