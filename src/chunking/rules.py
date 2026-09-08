"""src/chunking/rules.py — Step 2.2, the section-aware chunking engine.

Pure text in, `RawChunk` out. No I/O, no database, no `src.db` import —
testable directly on strings, same reasoning `evals/metrics.py` gives for
being importable before retrieval exists.

Implements group-11.md's six chunking rules:

  1. Split on document STRUCTURE first — headings, table boundaries. Never a
     blind character count.
  2. Target 400-700 tokens. Below 200 there is not enough context to embed
     meaningfully; above 900 the embedding becomes an average of several
     topics.
  3. NEVER split a table. If a table exceeds budget, split by row groups and
     repeat the header row in each part.
  4. Overlap 15% at natural boundaries only — a sentence, never mid-clause.
  5. Prepend the section path to the embedded text (done by the embedder,
     `src/llm/embeddings.py` — this module only attaches `section_path` to
     each chunk for it to use).
  6. Carry document title, effective date and version as metadata (also the
     embedder's job — see its module docstring for why there is no database
     column for it).

CARRYFORWARD F36 (measured at `6affe26`): every one of the 111 corpus
documents is under the 200-token floor (max 127 tokens). Rule 2's floor is
project-wide, not a per-strategy toggle — a splitter that would leave a
remainder chunk below 200 tokens must not perform that split; it merges the
remainder back into its neighbour instead. Applied consistently, that floor
is what makes rows 1-3 of 2.2's case-study table measure identically on this
corpus: no document here is long enough for ANY floor-respecting strategy to
produce more than one chunk, not because the strategies behave the same in
general, but because none of them ever reaches a token count where they
would disagree. `tests/chunking/test_rules_synthetic_long_document.py`
exercises the disagreement directly, on text long enough to reach it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

# ---------------------------------------------------------------------------
# Rule 2's numbers
# ---------------------------------------------------------------------------

FLOOR_TOKENS = 200
TARGET_MIN_TOKENS = 400
TARGET_MAX_TOKENS = 700
CEILING_TOKENS = 900
OVERLAP_RATIO = 0.15

# `text-embedding-3-large` (the default in .env.example) is tokenized with
# cl100k_base, same as every other current OpenAI embedding model — using
# the encoding directly rather than `encoding_for_model` avoids a lookup
# that fails on a model name tiktoken has never heard of.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """The one place a token count is computed, so every rule (200/400-700/
    900/15%) is measured against the same tokenizer."""
    return len(_ENCODING.encode(text))


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace. Rule 4:
    overlap only at a sentence boundary, never mid-clause — this is the
    function that boundary is measured against."""
    parts = [p for p in _SENTENCE_END_RE.split(text.strip()) if p]
    return parts or [text.strip()]


# ---------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableBlock:
    """An indented record group — this corpus's shape for a table (see
    `scripts/generate_corpus.py`'s Zone A/B/C blocks: an intro line followed
    by indented fields). `header_line` is the intro line if there was one
    directly above the first indented row, so rule 3's "repeat the header
    row in each part" has something to repeat when a table is split."""

    header_line: str | None
    rows: tuple[str, ...]

    @property
    def text(self) -> str:
        lines = ([self.header_line] if self.header_line else []) + list(self.rows)
        return "\n".join(lines)


@dataclass(frozen=True)
class TextBlock:
    """One paragraph of ordinary prose or a bullet/numbered list, joined
    from its source lines with a single space (blank lines are the
    paragraph separator, not embedded newlines)."""

    text: str


Block = TableBlock | TextBlock


@dataclass(frozen=True)
class Section:
    """One heading and the blocks under it. `heading is None` for a
    document's preamble — content that appears before its first ALL-CAPS
    heading (e.g. the fuel surcharge policy's opening sentence)."""

    heading: str | None
    blocks: tuple[Block, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    metadata: dict[str, str]
    sections: tuple[Section, ...]


_METADATA_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /]{0,40}):\s+(\S.*)$")


def _is_heading(line: str) -> bool:
    """A heading is a line with no lowercase letters and at least one
    letter — "CALCULATION", "DETENTION TIME WINDOWS", "BASE FUEL SURCHARGE:
    15.5%" all qualify; "TERM: 24 months..." and "CARRIER: FedEx Logistics"
    do not, because their values are mixed-case — they are single data
    fields, not section titles with content under them, and are correctly
    left as body text.

    Never true for an indented line: an indented ALL-CAPS label inside a
    table row (this corpus has none, but a future one might) must not be
    read as a section boundary cutting a table in half.
    """
    stripped = line.strip()
    if not stripped or line[:1].isspace():
        return False
    return stripped == stripped.upper() and any(ch.isalpha() for ch in stripped)


def _is_table_row(line: str) -> bool:
    return bool(line) and line[:1].isspace() and line.strip() != ""


def parse_structure(text: str) -> ParsedDocument:
    """Rule 1: split on structure first. Returns the title, the leading
    metadata block (rule 6's raw material), and the body as a sequence of
    (heading, blocks) sections with table row-groups called out separately
    so a splitter can refuse to cut through one (rule 3)."""
    lines = text.split("\n")

    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    title = lines[i].strip() if i < len(lines) else ""
    i += 1

    while i < len(lines) and not lines[i].strip():
        i += 1

    metadata: dict[str, str] = {}
    while i < len(lines) and lines[i].strip():
        m = _METADATA_RE.match(lines[i])
        if not m:
            break
        metadata[m.group(1).strip()] = m.group(2).strip()
        i += 1

    sections: list[Section] = []
    heading: str | None = None
    blocks: list[Block] = []
    para_lines: list[str] = []
    table_rows: list[str] = []

    def flush_para() -> None:
        if para_lines:
            blocks.append(TextBlock(text=" ".join(para_lines)))
            para_lines.clear()

    def flush_table() -> None:
        nonlocal table_rows
        if table_rows:
            header = None
            if para_lines and para_lines[-1].rstrip().endswith(":"):
                header = para_lines.pop()
            flush_para()
            blocks.append(TableBlock(header_line=header, rows=tuple(table_rows)))
            table_rows = []

    def flush_section() -> None:
        flush_table()
        flush_para()
        if heading is not None or blocks:
            sections.append(Section(heading=heading, blocks=tuple(blocks)))
        blocks.clear()

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush_table()
            flush_para()
        elif _is_table_row(line):
            # Deliberately not flushing `para_lines` here: `flush_table`
            # needs the still-pending intro line (e.g. "Zone A (...):") to
            # pop as the table's header, before flushing whatever paragraph
            # text remains ahead of it.
            table_rows.append(line.strip())
        elif _is_heading(line):
            flush_section()
            heading = line.strip()
        else:
            flush_table()
            para_lines.append(line.strip())
        i += 1
    flush_section()

    return ParsedDocument(title=title, metadata=metadata, sections=tuple(sections))


# ---------------------------------------------------------------------------
# Splitting — three strategies, one shared floor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawChunk:
    """What this module produces: text plus the section path it came from.
    `src/chunking/__init__.py` turns these into the `Chunk` dataclass
    `src/ingest/pipeline.py` (2.1) already defines and expects."""

    text: str
    section_path: tuple[str, ...]
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_count", count_tokens(self.text))


def _whole_document_chunk(parsed: ParsedDocument) -> RawChunk:
    """One chunk covering the entire document, section path = the title
    only, since a single chunk's content is not any one section's alone."""
    parts = [s.text for s in parsed.sections if s.text]
    text = "\n\n".join(parts) if parts else ""
    return RawChunk(text=text, section_path=(parsed.title,))


def _document_total_tokens(parsed: ParsedDocument) -> int:
    return count_tokens(
        "\n\n".join(s.text for s in parsed.sections if s.text) or ""
    )


def chunk_structure_aware(text: str) -> list[RawChunk]:
    """Rows 3-5 of the case-study table: split at heading boundaries first,
    never through a table, respecting the floor/target/ceiling and 15%
    sentence-boundary overlap. On this corpus every document is under the
    200-token floor in total, so this always returns exactly one chunk
    (CARRYFORWARD F36) — the multi-chunk path is exercised in
    `tests/chunking/test_rules_synthetic_long_document.py`.
    """
    parsed = parse_structure(text)
    if _document_total_tokens(parsed) < FLOOR_TOKENS + TARGET_MIN_TOKENS:
        # Even a single split could leave a remainder below the floor
        # (rule 2). Coalesce to one chunk rather than risk it — see the
        # module docstring's CARRYFORWARD F36 note.
        return [_whole_document_chunk(parsed)]

    return _greedy_pack_sections(parsed)


def _greedy_pack_sections(parsed: ParsedDocument) -> list[RawChunk]:
    """Pack whole sections into a chunk until the target ceiling, splitting
    a single oversized section by its table/text blocks (never mid-table),
    and prepending a 15% sentence-boundary overlap from the end of the
    previous chunk to the start of the next."""
    chunks: list[RawChunk] = []
    pending: list[tuple[str | None, Block]] = []
    pending_tokens = 0

    def flush() -> None:
        nonlocal pending, pending_tokens
        if not pending:
            return
        text = "\n\n".join(b.text for _, b in pending)
        heading = next((h for h, _ in pending if h), None)
        path = (parsed.title, heading) if heading else (parsed.title,)
        if chunks and OVERLAP_RATIO > 0:
            prev_sentences = _split_sentences(chunks[-1].text)
            n_overlap = max(1, round(len(prev_sentences) * OVERLAP_RATIO))
            overlap = " ".join(prev_sentences[-n_overlap:])
            text = f"{overlap} {text}"
        chunks.append(RawChunk(text=text, section_path=path))
        pending = []
        pending_tokens = 0

    for section in parsed.sections:
        for block in section.blocks:
            block_tokens = count_tokens(block.text)
            if isinstance(block, TableBlock) and pending_tokens + block_tokens > CEILING_TOKENS:
                # Rule 3: never split a table. If it alone pushes past the
                # ceiling, it still goes in whole — flush what came before
                # it first so it starts its own chunk.
                flush()
            elif pending_tokens + block_tokens > TARGET_MAX_TOKENS and pending_tokens >= FLOOR_TOKENS:
                flush()
            pending.append((section.heading, block))
            pending_tokens += block_tokens
    flush()

    # A trailing chunk under the floor is merged back into the previous one
    # rather than shipped undersized (rule 2's floor, applied at the tail).
    if len(chunks) > 1 and chunks[-1].token_count < FLOOR_TOKENS:
        merged_text = chunks[-2].text + "\n\n" + chunks[-1].text
        merged_path = chunks[-2].section_path
        chunks[-2:] = [RawChunk(text=merged_text, section_path=merged_path)]

    return chunks


def chunk_fixed_tokens_sentence_safe(text: str, *, size: int = 512) -> list[RawChunk]:
    """Row 2: fixed-size token windows, but the boundary always falls after
    a sentence, never mid-clause (rule 4's overlap requirement implies a
    sentence-safe boundary in the first place — an overlap "at a sentence"
    that lands mid-sentence is not one). Ignores structure entirely — no
    heading awareness, no table awareness — which is exactly what makes it
    the comparison point for row 3."""
    parsed = parse_structure(text)
    whole = _whole_document_chunk(parsed)
    if whole.token_count < FLOOR_TOKENS + size:
        return [whole]
    return _pack_sentences(_split_sentences(whole.text), size, parsed.title)


def chunk_fixed_chars_naive(text: str, *, size: int = 512) -> list[RawChunk]:
    """Row 1: the naive baseline — genuinely naive. Blind `size`-character
    windows over the RAW input text, no sentence safety, no floor guard,
    and — this is the part that took two attempts to get right — no
    `parse_structure` call at all. A structurally-blind splitter that first
    calls the structure parser to strip the title and metadata block before
    slicing is not naive, it has just outsourced its structure-awareness to
    a function it claims not to use.

    **Two corrections after user review, both against the same underlying
    mistake: treating rule 2's 200-token floor, and then the title/metadata
    split, as though they applied to every row.** They do not — row 1's
    entire purpose in the case-study table is to BE the naive comparison
    point. The first fix removed the floor guard but still sliced
    `_whole_document_chunk(parse_structure(text)).text` (body only, title
    and metadata already stripped) — silently identical to rows 2-3 again,
    for a second, subtler reason: on this corpus the body alone never
    exceeds 512 characters (max 457) even though the full raw file does (max
    649, 57 of 111 documents over 512). Splitting the raw `text` argument
    directly is what makes this row actually naive AND actually engage.

    Nothing under `src/` outside this function calls it — confirmed by
    `grep -rn chunk_fixed_chars_naive src/`, which returns only this
    definition (plus this docstring's own mention of the command).
    """
    return [
        RawChunk(text=text[start : start + size], section_path=())
        for start in range(0, len(text), size)
    ] or [RawChunk(text=text, section_path=())]


def _pack_sentences(sentences: list[str], target_tokens: int, title: str) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        stoks = count_tokens(sentence)
        if current and current_tokens + stoks > target_tokens and current_tokens >= FLOOR_TOKENS:
            chunks.append(RawChunk(text=" ".join(current), section_path=(title,)))
            n_overlap = max(1, round(len(current) * OVERLAP_RATIO))
            current = current[-n_overlap:]
            current_tokens = count_tokens(" ".join(current))
        current.append(sentence)
        current_tokens += stoks
    if current:
        chunks.append(RawChunk(text=" ".join(current), section_path=(title,)))

    if len(chunks) > 1 and chunks[-1].token_count < FLOOR_TOKENS:
        merged = chunks[-2].text + " " + chunks[-1].text
        chunks[-2:] = [RawChunk(text=merged, section_path=chunks[-2].section_path)]
    return chunks


def golden_chunk_id(tenant_slug: str, category: str, file_stem: str, chunk_index: int) -> str:
    """`evals/golden.jsonl`'s id convention: 1-based chunk numbering after
    the `#`. `chunk_index` here is the 0-based index `Chunk.index` uses."""
    return f"{tenant_slug}/{category}/{file_stem}#{chunk_index + 1}"
