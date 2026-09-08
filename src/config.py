"""src/config.py — shared application settings, read once from the environment.

Not owned by any single runbook step. `pydantic-settings` is already a pinned
dependency (pyproject.toml) and `.env.example` already documents EMBED_MODEL /
EMBED_DIM, but nothing before 2.2/2.3 needed to read them programmatically.
Same disposition as carry-forwards C7/C8/F20: buildable now, needed by the
runbook's own code (group-11.md's illustrative `src/ingest/embed.py` block
reads `settings.embed_model`), owned by nobody more specific than "whoever
needs it first".

This module talks to no database and imports no `asyncpg` — it is plain
environment parsing, so it needs no entry in `[tool.importlinter]`'s
`source_modules` list.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 1536

    # OpenAI's published list price for text-embedding-3-large: $0.13 per 1M
    # tokens (openai.com/api/pricing, checked when this was written). This is
    # a cited vendor price, not a measurement of anything on this project —
    # see CARRYFORWARD F27/F36 on the difference between the two. It exists
    # only so cost accounting (2.3's Done-when) has a number to multiply
    # token counts by, and it is trivially overridable via env if EMBED_MODEL
    # changes to a different provider.
    embed_price_usd_per_1k_tokens: float = 0.00013

    # Bounded concurrency for batch embedding (2.3's Done-when: "bounded
    # concurrency"). Not in .env.example because nothing before 2.3 needed it.
    embed_max_concurrency: int = 4
    embed_batch_size: int = 96


settings = Settings()
