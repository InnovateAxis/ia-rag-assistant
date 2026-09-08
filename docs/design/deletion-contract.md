# The deletion contract — step 2.5

What happens to a chunk when the document behind it goes away, who guarantees
it, and the one place the guarantee cuts the other way.

Deletion is a data-protection requirement here, not a tidiness one. A deleted
contract whose embeddings survive is still retrievable and still quotable in
an answer, and the client finds out when the assistant cites an agreement they
terminated.

---

## The four clauses, and where each one lives

| Clause | Mechanism | Code | Status |
|---|---|---|---|
| Document deleted → chunks cascade-deleted in the same transaction | `document_chunks.document_id … on delete cascade` (`migrations/0001_chunks.sql:13`) — the database does it, inside whatever transaction issued the delete | none needed; `src/ingest/deletion.delete_document_chunks` covers the case where this service is asked to remove a document's chunks without the parent row going away | Built · `tests/ingest/test_deletion.py::test_deleting_a_document_cascades_inside_the_same_transaction` |
| Document updated → old chunks deleted, new inserted, one transaction | one `db.session()` block is one transaction; the delete and every insert are inside it | `src/ingest/pipeline.ingest`, calling `src/ingest/deletion.delete_chunks_for_document` on the same connection | Built · `test_replacing_a_documents_chunks_is_never_half_visible` |
| Tenant offboarded → every row removed, verified by a post-delete count, verification logged | purge inside the tenant's RLS session; count from an admin connection outside it | `src/ingest/deletion.purge_tenant` + `src/db/offboard_cli.offboard_tenant` | Built · `test_offboarding_removes_every_row_and_the_count_proves_it` |
| Document unshared → if P3 revokes access, chunks removed in the same request | — | — | **Not built.** It depends on a Pod P revocation integration that does not exist, and it is not in 2.5's Done-when. Nothing in this repository implements it and nothing should be read as if it does. |

## Invariant 1 in the deletion path

There is no tenant predicate in any of it. Every statement that removes a
chunk is one of these three:

```sql
delete from document_chunks where document_id = $1   -- one document
delete from document_chunks                          -- tenant offboarding
-- and the cascade, which the application never writes at all
```

The second one is the whole argument in one line. Inside
`db.session(tenant_id)`, `chunks_tenant_isolation` has already restricted the
visible rows to that tenant's, so *"delete everything I can see"* **is**
*"delete everything belonging to this tenant"*. Adding `where tenant_id = $1`
would not narrow it; it would only add a second, unenforced copy of a rule the
policy already owns — and a copy that can be forgotten is the failure mode
this architecture exists to remove (carry-forward F39).

The same reasoning covers the delete-by-`document_id` case, and it has a
consequence worth stating because it looks like a missing check: asking to
delete a document that belongs to another tenant is not an error and not a
special case. It matches zero rows, because the policy never made those rows
visible to delete. There is no ownership comparison in the code to get wrong
(`test_deleting_another_tenants_document_matches_nothing`).

## What the single datastore buys

> **The two-store design fails exactly here.** With embeddings in a separate
> vector database, deletion is two operations across two systems with no
> shared transaction. The failure — a deleted document that is still
> retrievable — is invisible until a client notices the assistant quoting a
> contract they terminated. Cascade delete in one database makes the whole
> class of bug impossible.

This is true, and it is the strongest claim this project makes about
deletion. It is worth being precise about *which* class of bug it eliminates:
**a document whose deletion succeeded while its embeddings survived.** That
bug cannot occur here, and the proof is not that the code is careful — it is
that there is no window in which the two states disagree. The test rolls the
deleting transaction back and watches the chunks come back with it; a cascade
running in its own transaction, or a nightly sweep, could not do that.

The update clause is the same property under load: a reader sampling a
document continuously while its chunks are replaced sees the old set or the
new set, never a mixture and never an empty document. The measurement, at
`REPLACEMENT_CHUNKS = 100`: 32 samples across the write, 0 of them dirty. The
same sampler over the same replacement split across two transactions caught 31
dirty states out of 33, which is what makes the first number a result rather
than an artefact of sampling too slowly.

## The limit: one tenant's delete can destroy another tenant's chunk

Carry-forward **F5**, demonstrated at group-03 and now reproducible from this
repository: `tests/ingest/test_deletion.py::test_cross_tenant_cascade_removes_another_tenants_chunk`.

Foreign-key enforcement runs **outside** row-level security. `on delete
cascade` therefore removes *every* chunk row referencing a deleted document,
without consulting any policy and without regard to which tenant each of those
rows is stamped for. So a `documents` row deleted in acme's own RLS-scoped
session takes with it a chunk stamped `ten_globex` — a tenant whose session
issued nothing and whose policy was never asked.

Both halves of this belong in the same paragraph, and neither may be dropped.

**It is real, and it is a cross-tenant effect.** It is destructive rather than
disclosing: acme still cannot *read* globex's chunk — the policy holds on
every read, and the isolation suite stays green throughout — but acme's delete
removed it. "No tenant can see another tenant's data" is not the same
statement as "no tenant can affect another tenant's data", and this project
guarantees the first.

**It is conditional, and the condition is carry-forward F4.** It requires a
chunk stamped with one tenant's id while referencing another tenant's
document. RLS enforces the *value* of `tenant_id`, never its *correctness*, so
the database cannot rule such a row out — which is exactly why step 2.1's
ingest derives `tenant_id` from the verified session context and never from
caller-supplied data or from the document row (`src/ingest/pipeline.ingest`).
The precondition is structurally hard rather than merely discouraged. The
mis-stamped row in the test is planted with raw SQL precisely because no
application path in this repository will produce one.

**This is not a refutation of the claim above.** The cascade eliminates
orphaned embeddings, and it does; F5 is a different edge of the same
mechanism — the same disregard for policy that makes the cleanup unmissable
also makes it unselective. Reading either one as cancelling the other gets the
architecture wrong in both directions.

**What was decided at 2.5, and what was not.** The decision was to
characterise F5 with a test and record it here as a limit, rather than leave a
demonstrated cross-tenant effect resting on prose in a project whose entire
argument is that the guarantee is testable. No change was made to the foreign
key, no tenant-consistency constraint or trigger was added, and reopening
ADR-0001 — which already concedes the F4 gap this sits inside — was considered
and not chosen. Anyone revisiting that should read carry-forward F41 first.

## Tenant offboarding: two halves, deliberately different

Removal is application-side and **inside** the policy:
`src/ingest/deletion.purge_tenant` runs the no-WHERE deletes above in the
offboarded tenant's own session, across the three tables this repository owns
— `document_chunks` (0001/0002) plus `backfill_progress` and
`embedding_cutover` (0003). The contract says "every chunk removed", and
chunks are why offboarding matters; the two operational tables are keyed by
`tenant_id`, and leaving an offboarded tenant's rows in them would leave that
tenant's identifier in this database after we said it was gone.

Verification is infrastructure-side and **outside** the policy:
`src/db/offboard_cli.count_tenant_rows`, on an admin connection, reached only
through `python -m src.db.offboard_cli <tenant_id>`. This is the part that is
easy to get wrong in a way that looks right — a count taken inside the
tenant's own session returns 0 whether the rows are gone or merely invisible,
which is the one distinction the entire exercise turns on. That count would
pass forever while proving nothing, so it is not the one we take.

Two consequences of that, both deliberate:

* **It is the one tenant predicate in this project's SQL**, and it is not
  application SQL. `src/db/offboard_cli.py` is not one of `pyproject.toml`'s
  `[tool.importlinter]` `source_modules`, and **no module under `src.api`,
  `src.auth`, `src.chunking`, `src.generate`, `src.ingest`, `src.llm`,
  `src.orchestrate`, `src.retrieval` or `src.telemetry` may import it** — the
  same seam rule `src/db/tenants.py` and `src/db/backfill_cli.py` already
  carry (carry-forwards F35, F40). Invariant 1 governs the query path, where a
  hand-written tenant filter substitutes a promise about code discipline for
  the policy. Here the point is to look at what the policy would hide, which
  is why it cannot live there.
* **The count refuses to report a number it cannot trust.** If the connecting
  role is subject to RLS, `count_tenant_rows` raises
  `VerificationNotTrustworthy` instead of returning the zero it would
  otherwise return for every tenant, always, including one whose data was
  never touched (`test_offboarding_verification_refuses_a_connection_rls_applies_to`).
  `offboard_tenant` raises `TenantNotEmpty` rather than reporting success when
  anything survives, and logs the counts either way.

Invariant 4 is untouched throughout: no second application pool (removal goes
through `src.db.session`'s one pool), no `BYPASSRLS` grant to
`ia_rag_service`, and the verification connection is the same short-lived
admin connection `migrate.py` already opens for schema work.

## What this repository does not own

The `documents` table belongs to Pod P (carry-forward C7). There is no
migration for it here and nothing in `src/` writes it. When P deletes a
document, the cascade removes our chunks in P's own transaction, with no call
into this code at all — which is the first clause of the contract, and the
reason it is the only one that needs no application code to be true.
