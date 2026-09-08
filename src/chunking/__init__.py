"""src/chunking — Step 2.2, fills the `Chunker` protocol `src/ingest/pipeline.py`
(2.1) already defines: `Chunker.__call__(self, doc: DocumentRecord) -> list[Chunk]`.

`src/chunking/rules.py` does the actual structure parsing and splitting on
plain text; this module's only job is the adapter — load the document's
content, run it through a rule strategy, and shape the result into the
`Chunk` dataclass 2.1's `ingest()` already inserts.

**Storage is out of scope here, same disposition as carry-forward C7 ("the
`documents` table belongs to Pod P, never claim ownership of it here").**
This repo's own corpus lives as files under `corpus/<tenant>/<category>/`, so
`storage_key` is read as a path relative to the repository root. A real
deployed backend (S3, etc.) would replace `_load_document_text` behind this
same seam without touching the chunking rules or `Chunk` shape.
"""

from __future__ import annotations

from pathlib import Path

from src.chunking.rules import RawChunk, chunk_structure_aware
from src.ingest.pipeline import Chunk, DocumentRecord

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_document_text(storage_key: str) -> str:
    path = Path(storage_key)
    if not path.is_absolute():
        path = _REPO_ROOT / storage_key
    return path.read_text(encoding="utf-8")


def _to_chunk(index: int, raw: RawChunk) -> Chunk:
    return Chunk(
        index=index,
        content=raw.text,
        section_path=list(raw.section_path),
        token_count=raw.token_count,
    )


def section_aware_chunk(doc: DocumentRecord) -> list[Chunk]:
    """The `Chunker` implementation 2.1's `ingest()` is called with. Rule 1:
    structure first, never a blind character count — see `rules.py` for
    rules 2-4. Rule 5 (section-path prefix) is carried on each `Chunk` via
    `section_path`; the embedder (2.3) is what actually prepends it to the
    text it sends for embedding."""
    text = _load_document_text(doc.storage_key)
    raw_chunks = chunk_structure_aware(text)
    return [_to_chunk(i, rc) for i, rc in enumerate(raw_chunks)]


# `src.ingest.pipeline.Chunker` is a `Protocol` with `__call__`; a plain
# function of the right shape satisfies it structurally — pass
# `section_aware_chunk` directly as `ingest()`'s `chunk=` argument.
