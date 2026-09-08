"""src/ingest/pipeline.py — Step 2.1, tenant-scoped ingestion.

Builds the transaction shape only. Carry-forward F34: at this point in the
runbook `src/chunking` and `src/llm` are both empty — chunking is 2.2 and
embedding is 2.3, both AI Engineer, one group away — so a real chunker or
embedder cannot be written here without inventing work that group owns.
`chunk` and `embed` are injected collaborators instead: 2.2 and 2.3 fill them
in, this module owns only the transaction they run inside.

Tenant context is established FIRST. Every subsequent statement in this
transaction — read, chunk write, embedding write — is inside the policy.
There is no window where it is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from src.db import session as db
from src.ingest.deletion import delete_chunks_for_document


class DocumentNotFound(Exception):
    """Raised when a document id does not exist, or belongs to another tenant.

    Both cases produce exactly this exception and nothing else. We
    deliberately cannot tell the difference, and neither can a caller
    probing for other tenants' document ids — see `ingest`'s read below.
    """

    def __init__(self, document_id: UUID) -> None:
        super().__init__(f"document not found: {document_id}")
        self.document_id = document_id


@dataclass(frozen=True)
class DocumentRecord:
    id: UUID
    title: str
    storage_key: str
    content_type: str


@dataclass(frozen=True)
class Chunk:
    """One unit of content a chunker produces. 2.2 owns the real chunker;
    this is the contract it must satisfy to plug into `ingest`."""

    index: int
    content: str
    section_path: list[str] | None = None
    page_from: int | None = None
    page_to: int | None = None
    token_count: int = 0


@dataclass(frozen=True)
class EmbeddedChunk:
    """One chunk plus the embedding 2.3's embedder attaches to it."""

    chunk: Chunk
    embedding: list[float]
    embed_model: str
    embed_dim: int


@dataclass(frozen=True)
class IngestResult:
    document_id: UUID
    chunk_count: int


class Chunker(Protocol):
    def __call__(self, doc: DocumentRecord) -> list[Chunk]: ...


class Embedder(Protocol):
    async def __call__(self, chunks: list[Chunk]) -> list[EmbeddedChunk]: ...


_INSERT_CHUNK = """
insert into document_chunks
  (tenant_id, document_id, chunk_index, content, embedding,
   embed_model, embed_dim, section_path, page_from, page_to, token_count)
values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
"""

_SELECT_DOCUMENT = (
    "select id, title, storage_key, content_type from documents where id = $1"
)


async def ingest(
    document_id: UUID,
    tenant_id: str,
    *,
    chunk: Chunker,
    embed: Embedder,
) -> IngestResult:
    """Tenant-scoped ingestion.

    `tenant_id` must be the caller's own verified session context (e.g. a
    validated JWT claim from 4.4), never a value re-derived from the
    document row or from anything else caller-controlled. Carry-forward F4:
    RLS enforces the VALUE written to `tenant_id`, not whether it is the
    RIGHT value, so a mis-stamped write is not something the database can
    catch after the fact — the defence has to be structural, here, at the
    one place chunks get written. `document_chunks.tenant_id` is always the
    `tenant_id` this function received, never anything read back off `doc`
    or off a chunk.

    `chunk` and `embed` are injected collaborators (carry-forward F34).
    """
    async with db.session(tenant_id) as s:
        row = await s.fetchrow(_SELECT_DOCUMENT, document_id)
        if row is None:
            # Either it does not exist, or it belongs to someone else. We
            # deliberately cannot tell the difference, and neither can a
            # caller probing for other tenants' document ids. No tenant
            # predicate is written here (Invariant 1) — the RLS policy on
            # `documents` already scoped this read to `tenant_id`.
            raise DocumentNotFound(document_id)

        doc = DocumentRecord(
            id=row["id"],
            title=row["title"],
            storage_key=row["storage_key"],
            content_type=row["content_type"],
        )

        chunks = chunk(doc)
        embedded = await embed(chunks)

        # Re-ingesting the same document replaces its chunks atomically:
        # delete and every insert below run inside the one transaction this
        # `async with` block opened, so a concurrent reader sees the old set
        # or the new set and never both (step 2.5's second Done-when item;
        # `tests/ingest/test_deletion.py` proves it against a reader that
        # samples throughout the write). No tenant predicate on the delete
        # either — RLS already scopes it to `tenant_id` for the life of this
        # transaction, exactly like every other statement here. The statement
        # itself lives in `src.ingest.deletion` (2.5), which owns every chunk
        # removal in this project; passing `s` keeps it in THIS transaction.
        await delete_chunks_for_document(s, document_id)

        for ec in embedded:
            await s.execute(
                _INSERT_CHUNK,
                tenant_id,
                document_id,
                ec.chunk.index,
                ec.chunk.content,
                ec.embedding,
                ec.embed_model,
                ec.embed_dim,
                ec.chunk.section_path,
                ec.chunk.page_from,
                ec.chunk.page_to,
                ec.chunk.token_count,
            )

    return IngestResult(document_id=document_id, chunk_count=len(embedded))
