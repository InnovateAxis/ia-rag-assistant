"""src/llm/embeddings.py — Step 2.3, fills the `Embedder` protocol
`src/ingest/pipeline.py` (2.1) already defines:
`Embedder.__call__(self, chunks: list[Chunk]) -> list[EmbeddedChunk]`.

**Deviation from group-11.md's illustrative `src/ingest/embed.py` block,
recorded here rather than silently:** that block both computes embeddings
AND writes them to `document_chunks` in its own `db.session()` block. 2.1
(carry-forward F34, "build the seam, defer the fillers") already built the
one transaction that reads the document and writes chunks + embeddings —
`embed`/`chunk` are injected as pure collaborators precisely so 2.2/2.3 do
not reopen a second transaction or a second write path. This module
therefore returns `EmbeddedChunk` objects; `src.ingest.pipeline.ingest()`
does the insert. Rebuilding the write here would be exactly what "do not
rebuild anything 2.1 shipped" forbids.

The section-path prefix (rule 5) IS applied here, matching the runbook
block's `f"{' > '.join(c.section_path)}\n\n{c.content}"` exactly — this is
the text actually sent for embedding; `Chunk.content` (and therefore the
`content` / `content_tsv` columns) stays the raw, unprefixed text.

**Rule 6 ("carry title, effective date and version as metadata") has no
landing place in `document_chunks`** — `migrations/0001_chunks.sql` has no
metadata column, and adding one is a schema change no group has authorised
here (same family as carry-forwards F34/F37: an acceptance criterion
referencing storage this project's schema does not have). This module
therefore embeds section-path-prefixed text only, matching the runbook's own
code exactly; the metadata-enriched variant used for the case-study table's
row 5 is a `scripts/measure_chunking_impact.py`-only construction, not
something wired into ingestion — see that script's docstring.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import structlog

from src.chunking.rules import count_tokens
from src.config import settings
from src.ingest.pipeline import Chunk, EmbeddedChunk

logger = structlog.get_logger(__name__)


class EmbeddingProvider(Protocol):
    """The raw call to an embedding model: texts in, vectors out, same
    order. Everything else in this module (batching, concurrency, cost) is
    provider-agnostic and built on top of this one seam, so tests inject a
    fake provider and never make a network call."""

    async def __call__(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddingProvider:
    """Real implementation: an OpenAI-compatible `/embeddings` endpoint over
    HTTP (`httpx` is already a pinned dependency; no `openai` package is).
    Not exercised by any test in this group — there is no live API key in
    this environment, and 2.2/2.3's Done-when asks for a real, unit-tested
    *function*, not a live network call. Constructed here so the seam is
    real rather than merely described, and swappable behind
    `EmbeddingProvider` without touching `BoundedConcurrencyEmbedder`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = settings.embed_model,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
        # Preserve the caller's ordering, matching an OpenAI-shaped
        # response's own `index` field rather than assuming list order.
        by_index = {row["index"]: row["embedding"] for row in data["data"]}
        return [by_index[i] for i in range(len(texts))]


@dataclass
class EmbeddingCost:
    """Accumulated cost for one `BoundedConcurrencyEmbedder` run (2.3's
    Done-when: "cost accounting"). `usd` uses `settings.embed_price_usd_per_1k_tokens`
    — see `src/config.py` for its provenance."""

    tokens: int = 0
    usd: float = 0.0

    def add(self, tokens: int) -> None:
        self.tokens += tokens
        self.usd += (tokens / 1000.0) * settings.embed_price_usd_per_1k_tokens


def _embed_text(chunk: Chunk) -> str:
    """Rule 5, exactly as group-11.md's illustrative block writes it."""
    prefix = " > ".join(chunk.section_path or [])
    return f"{prefix}\n\n{chunk.content}" if prefix else chunk.content


class BoundedConcurrencyEmbedder:
    """Satisfies `src.ingest.pipeline.Embedder`. Batches chunks
    (`settings.embed_batch_size`), runs batches through the injected
    `provider` with at most `settings.embed_max_concurrency` in flight at
    once (`asyncio.Semaphore`), and tracks token/cost accounting on
    `self.last_run_cost`.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        model: str | None = None,
        dim: int | None = None,
        max_concurrency: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._provider = provider
        self._model = model or settings.embed_model
        self._dim = dim or settings.embed_dim
        self._semaphore = asyncio.Semaphore(max_concurrency or settings.embed_max_concurrency)
        self._batch_size = batch_size or settings.embed_batch_size
        self.last_run_cost = EmbeddingCost()

    async def _embed_batch(self, batch: list[Chunk]) -> list[EmbeddedChunk]:
        texts = [_embed_text(c) for c in batch]
        async with self._semaphore:
            vectors = await self._provider(texts)
        for text in texts:
            self.last_run_cost.add(count_tokens(text))
        return [
            EmbeddedChunk(chunk=c, embedding=v, embed_model=self._model, embed_dim=self._dim)
            for c, v in zip(batch, vectors, strict=True)
        ]

    async def __call__(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        self.last_run_cost = EmbeddingCost()
        if not chunks:
            return []
        batches = [
            chunks[i : i + self._batch_size] for i in range(0, len(chunks), self._batch_size)
        ]
        results = await asyncio.gather(*(self._embed_batch(b) for b in batches))
        embedded = [ec for batch in results for ec in batch]
        logger.info(
            "embedding.batch_complete",
            n_chunks=len(embedded),
            n_batches=len(batches),
            tokens=self.last_run_cost.tokens,
            usd=round(self.last_run_cost.usd, 6),
        )
        return embedded


# ---------------------------------------------------------------------------
# CARRYFORWARD F37 — model-mismatch refusal. A real, unit-tested function;
# no query path (`src/retrieval/` is empty until 3.1). 3.1 wires this in
# rather than writing a second one.
# ---------------------------------------------------------------------------


class ModelMismatchError(Exception):
    """Raised when one or more retrieved chunks were embedded with a model
    or dimension other than the one currently configured. Cosine distance
    between two different embedding spaces is noise (see this module's and
    `migrations/0001_chunks.sql`'s comments on why `embed_model`/`embed_dim`
    are stamped per row) — serving such a chunk back as a search result
    would silently return garbage."""

    def __init__(self, offending: list[tuple[str, str, int]]) -> None:
        self.offending = offending
        summary = ", ".join(f"{cid} ({model}/{dim})" for cid, model, dim in offending)
        super().__init__(f"{len(offending)} chunk(s) embedded with a stale model: {summary}")


def refuse_on_model_mismatch(
    rows: Sequence[Mapping[str, object]],
    *,
    current_model: str | None = None,
    current_dim: int | None = None,
) -> None:
    """Raises `ModelMismatchError` if any row's `embed_model`/`embed_dim`
    differs from the current configuration. `rows` is any sequence of
    mappings carrying `id` (or `chunk_id`), `embed_model` and `embed_dim` —
    deliberately shaped to accept `asyncpg.Record` rows directly (a
    `Record` implements `Mapping`), so 3.1 can pass its retrieved rows
    straight through without adapting them.

    An empty `rows` sequence raises nothing — nothing was retrieved, so
    there is nothing to refuse on model grounds; a genuinely empty result is
    the refusal orchestrator's (4.1) concern, not this guard's.
    """
    model = current_model or settings.embed_model
    dim = current_dim or settings.embed_dim
    offending: list[tuple[str, str, int]] = []
    for r in rows:
        row_model = str(r["embed_model"])
        row_dim = cast(int, r["embed_dim"])
        if row_model != model or row_dim != dim:
            row_id = str(r.get("id") or r.get("chunk_id") or "?")
            offending.append((row_id, row_model, row_dim))
    if offending:
        raise ModelMismatchError(offending)
