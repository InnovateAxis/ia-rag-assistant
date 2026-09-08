"""Step 2.3 — embedding batching/concurrency/cost, and the F37 model-mismatch
refusal guard. Pure unit tests: a fake `EmbeddingProvider`, no network call,
no database, no query path (F37: `src/retrieval/` stays empty until 3.1)."""

from __future__ import annotations

import asyncio

import pytest

from src.chunking.rules import count_tokens
from src.config import settings
from src.ingest.pipeline import Chunk
from src.llm.embeddings import (
    BoundedConcurrencyEmbedder,
    ModelMismatchError,
    refuse_on_model_mismatch,
)

DIM = 8


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(index=i, content=f"chunk body number {i}", section_path=["Doc", f"Section {i}"])
        for i in range(n)
    ]


class RecordingProvider:
    """Fake provider: returns a deterministic vector per text and records
    concurrency so tests can assert the semaphore is real, not decorative."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)  # force overlap so bounding is observable
        async with self._lock:
            self.in_flight -= 1
        self.calls.append(texts)
        return [[float(len(t) % 7)] * DIM for t in texts]


async def test_embedder_prepends_section_path_to_the_embedded_text():
    provider = RecordingProvider()
    embedder = BoundedConcurrencyEmbedder(provider, batch_size=10, max_concurrency=4)
    chunks = _chunks(2)

    result = await embedder(chunks)

    assert len(result) == 2
    sent = provider.calls[0]
    assert sent[0] == "Doc > Section 0\n\nchunk body number 0"
    assert sent[1] == "Doc > Section 1\n\nchunk body number 1"
    # Stored content stays the raw, unprefixed chunk text (2.1's insert
    # writes `ec.chunk.content` verbatim) — the prefix exists only in the
    # text sent for embedding, never in `Chunk.content` itself.
    assert result[0].chunk.content == "chunk body number 0"


async def test_embedder_stamps_configured_model_and_dim_on_every_row():
    provider = RecordingProvider()
    embedder = BoundedConcurrencyEmbedder(provider, model="test-model-v1", dim=DIM)

    result = await embedder(_chunks(3))

    assert all(ec.embed_model == "test-model-v1" for ec in result)
    assert all(ec.embed_dim == DIM for ec in result)
    assert all(len(ec.embedding) == DIM for ec in result)


async def test_embedder_batches_and_bounds_concurrency():
    provider = RecordingProvider()
    # 10 chunks, batch_size=2 -> 5 batches; max_concurrency=2 must never let
    # more than 2 batches be in flight against the provider at once.
    embedder = BoundedConcurrencyEmbedder(provider, batch_size=2, max_concurrency=2)

    result = await embedder(_chunks(10))

    assert len(result) == 10
    assert len(provider.calls) == 5
    assert all(len(batch) == 2 for batch in provider.calls)
    assert provider.max_in_flight <= 2
    assert provider.max_in_flight > 1  # proves batches actually overlapped


async def test_embedder_preserves_chunk_order_across_batches():
    provider = RecordingProvider()
    embedder = BoundedConcurrencyEmbedder(provider, batch_size=3, max_concurrency=4)
    chunks = _chunks(11)

    result = await embedder(chunks)

    assert [ec.chunk.index for ec in result] == list(range(11))


async def test_embedder_empty_input_returns_empty_and_costs_nothing():
    provider = RecordingProvider()
    embedder = BoundedConcurrencyEmbedder(provider)

    result = await embedder([])

    assert result == []
    assert provider.calls == []
    assert embedder.last_run_cost.tokens == 0
    assert embedder.last_run_cost.usd == 0.0


async def test_embedder_cost_accounting_matches_token_count_and_settings_price():
    provider = RecordingProvider()
    embedder = BoundedConcurrencyEmbedder(provider, batch_size=10)
    chunks = _chunks(4)

    await embedder(chunks)

    expected_tokens = sum(
        count_tokens(f"Doc > Section {i}\n\nchunk body number {i}") for i in range(4)
    )
    assert embedder.last_run_cost.tokens == expected_tokens
    expected_usd = (expected_tokens / 1000.0) * settings.embed_price_usd_per_1k_tokens
    assert embedder.last_run_cost.usd == pytest.approx(expected_usd)


async def test_embedder_cost_resets_between_runs():
    provider = RecordingProvider()
    embedder = BoundedConcurrencyEmbedder(provider)

    await embedder(_chunks(2))
    first_run_tokens = embedder.last_run_cost.tokens
    await embedder(_chunks(1))

    assert embedder.last_run_cost.tokens < first_run_tokens


# ---------------------------------------------------------------------------
# F37 — model-mismatch refusal
# ---------------------------------------------------------------------------


def _row(embed_model: str, embed_dim: int, chunk_id: str = "c1") -> dict:
    return {"id": chunk_id, "embed_model": embed_model, "embed_dim": embed_dim}


def test_refuse_on_model_mismatch_passes_when_every_row_matches_current():
    rows = [_row("text-embedding-3-large", 1536, "c1"), _row("text-embedding-3-large", 1536, "c2")]
    refuse_on_model_mismatch(rows, current_model="text-embedding-3-large", current_dim=1536)


def test_refuse_on_model_mismatch_raises_on_a_stale_model_name():
    rows = [_row("text-embedding-3-large", 1536), _row("text-embedding-ada-002", 1536)]
    with pytest.raises(ModelMismatchError) as exc_info:
        refuse_on_model_mismatch(rows, current_model="text-embedding-3-large", current_dim=1536)
    assert exc_info.value.offending == [("c1", "text-embedding-ada-002", 1536)]


def test_refuse_on_model_mismatch_raises_on_a_dimension_change_alone():
    rows = [_row("text-embedding-3-large", 3072)]
    with pytest.raises(ModelMismatchError):
        refuse_on_model_mismatch(rows, current_model="text-embedding-3-large", current_dim=1536)


def test_refuse_on_model_mismatch_names_every_offending_row_not_just_the_first():
    rows = [_row("old-model", 1536, "c1"), _row("text-embedding-3-large", 1536, "c2"), _row("old-model", 1536, "c3")]
    with pytest.raises(ModelMismatchError) as exc_info:
        refuse_on_model_mismatch(rows, current_model="text-embedding-3-large", current_dim=1536)
    assert {o[0] for o in exc_info.value.offending} == {"c1", "c3"}


def test_refuse_on_model_mismatch_empty_rows_is_not_an_error():
    refuse_on_model_mismatch([], current_model="text-embedding-3-large", current_dim=1536)


def test_refuse_on_model_mismatch_defaults_to_settings_when_current_not_passed():
    rows = [_row(settings.embed_model, settings.embed_dim)]
    refuse_on_model_mismatch(rows)  # must not raise
    with pytest.raises(ModelMismatchError):
        refuse_on_model_mismatch([_row("something-else", settings.embed_dim)])


def test_refuse_on_model_mismatch_accepts_asyncpg_record_shaped_mappings():
    """`asyncpg.Record` supports `Mapping`'s `[]` and `.get`, but not
    construction from kwargs the way a dict does — a plain dict is the
    closest fake without adding a test-only asyncpg dependency, and this
    test exists to pin that the function only ever uses Mapping's read
    interface (`[]`/`.get`), never anything dict-specific, so a real
    Record works unmodified at 3.1."""
    class FakeRecord:
        def __init__(self, d: dict) -> None:
            self._d = d

        def __getitem__(self, key: str) -> object:
            return self._d[key]

        def get(self, key: str, default: object = None) -> object:
            return self._d.get(key, default)

    rows = [FakeRecord({"chunk_id": "r1", "embed_model": "stale", "embed_dim": 1536})]
    with pytest.raises(ModelMismatchError) as exc_info:
        refuse_on_model_mismatch(rows, current_model="text-embedding-3-large", current_dim=1536)
    assert exc_info.value.offending[0][0] == "r1"
