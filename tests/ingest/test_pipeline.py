"""Step 2.1 — tenant-scoped ingestion.

Tests the transaction shape only: tenant context established first, the
document read, chunks and embeddings written inside one transaction, a
re-ingest replacing them atomically. Chunking (2.2) and embedding (2.3) do
not exist yet (carry-forward F34), so every test here injects a fake
chunker and a fake embedder — never a real one, and never anything living
under src/.
"""

from __future__ import annotations

import uuid

import asyncpg
import pytest

from src.db import session as db
from src.ingest.pipeline import Chunk, DocumentNotFound, EmbeddedChunk, ingest

EMBED_DIM = 1536


def fake_chunk(doc) -> list[Chunk]:
    return [
        Chunk(index=0, content=f"{doc.title} — part 1", token_count=3),
        Chunk(index=1, content=f"{doc.title} — part 2", token_count=3),
    ]


async def fake_embed(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk=c,
            embedding=[0.0] * EMBED_DIM,
            embed_model="fake-embed-test-only",
            embed_dim=EMBED_DIM,
        )
        for c in chunks
    ]


async def test_ingest_writes_chunks_and_embeddings_in_one_transaction(pool, doc_ids):
    acme = "ten_acme"
    result = await ingest(doc_ids[acme], acme, chunk=fake_chunk, embed=fake_embed)
    assert result.chunk_count == 2

    async with db.session(acme) as s:
        rows = await s.fetch(
            "select chunk_index, content, embed_model from document_chunks "
            "where document_id = $1 order by chunk_index",
            doc_ids[acme],
        )
    assert [r["chunk_index"] for r in rows] == [0, 1]
    assert all(r["embed_model"] == "fake-embed-test-only" for r in rows)


async def test_reingest_replaces_chunks_atomically(pool, doc_ids):
    acme = "ten_acme"
    await ingest(doc_ids[acme], acme, chunk=fake_chunk, embed=fake_embed)

    def fake_chunk_v2(doc) -> list[Chunk]:
        return [Chunk(index=0, content="replaced", token_count=1)]

    result = await ingest(doc_ids[acme], acme, chunk=fake_chunk_v2, embed=fake_embed)
    assert result.chunk_count == 1

    async with db.session(acme) as s:
        rows = await s.fetch(
            "select content from document_chunks where document_id = $1 "
            "order by chunk_index",
            doc_ids[acme],
        )
    assert [r["content"] for r in rows] == ["replaced"]


async def test_ingest_rolls_back_entirely_on_a_bad_embedding_mid_batch(pool, doc_ids):
    """Proves "one transaction", rather than reading it off the code: seed
    two good chunks, then fail a re-ingest partway through its own insert
    batch (after its delete already ran) with a wrong-dimension vector.
    Because delete and every insert share one transaction, the failure must
    roll all of it back — the two original chunks come back untouched."""
    acme = "ten_acme"
    await ingest(doc_ids[acme], acme, chunk=fake_chunk, embed=fake_embed)

    async def bad_embed(chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embedded = await fake_embed(chunks)
        good, bad = embedded[0], embedded[1]
        return [
            good,
            EmbeddedChunk(
                chunk=bad.chunk,
                embedding=[0.0] * 3,  # wrong dimension: document_chunks wants 1536
                embed_model=bad.embed_model,
                embed_dim=3,
            ),
        ]

    with pytest.raises(asyncpg.PostgresError):
        await ingest(doc_ids[acme], acme, chunk=fake_chunk, embed=bad_embed)

    async with db.session(acme) as s:
        rows = await s.fetch(
            "select chunk_index, content from document_chunks "
            "where document_id = $1 order by chunk_index",
            doc_ids[acme],
        )
    assert [r["chunk_index"] for r in rows] == [0, 1]
    assert all("part" in r["content"] for r in rows)


async def test_ingest_other_tenants_document_raises_not_found(pool, doc_ids):
    with pytest.raises(DocumentNotFound):
        await ingest(
            doc_ids["ten_globex"], "ten_acme", chunk=fake_chunk, embed=fake_embed
        )


async def test_ingest_unknown_document_raises_not_found(pool, doc_ids):
    with pytest.raises(DocumentNotFound):
        await ingest(uuid.uuid4(), "ten_acme", chunk=fake_chunk, embed=fake_embed)
