"""Eval metric definitions — what each number means, written down before any
retrieval or generation code exists to produce one.

Step 1.3 exists because a metric nobody defined in advance gets defined
after the fact, to match whatever the system already does. Every function
below states what it measures and, in its docstring, what it does NOT
measure — the second half is what stops a green dashboard from being read
as a stronger claim than the number supports.

None of this module talks to a database, an embedding model or an LLM. It
takes plain data in (ranked chunk ids, golden expected sets, judgments,
timings, costs) and returns a number. That is deliberate: metrics.py must
be importable and testable before src/retrieval or src/generate exist.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median

# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def hit_rate_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Iterable[str],
    k: int,
) -> bool:
    """Whether at least one expected chunk appears in the top k retrieved.

    Measures RETRIEVAL only. A hit here says the right passage was
    available to the generator; it says nothing about whether the
    generator actually used it, cited it correctly, or produced a
    faithful answer. Generation cannot fix a miss here, and a hit here
    cannot fix a bad generation either — the two are scored separately
    on purpose (see faithfulness, citation_accuracy).

    This function answers one question at a time. A per-category rate is
    the mean of this over the questions in that category — never blend
    categories with different `k` or different question types into one
    average (see the module docstring in golden.jsonl's companion doc:
    report per-category, never the overall number alone).
    """
    expected = set(expected_chunk_ids)
    top_k = set(retrieved_chunk_ids[:k])
    return bool(expected & top_k)


def mrr(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Iterable[str],
) -> float:
    """Reciprocal rank (1/rank) of the first expected chunk; 0.0 if absent.

    Distinguishes "found it at rank 1" (score 1.0) from "found it at rank
    9" (score ~0.11) from "never found it" (score 0.0) — hit_rate_at_k
    collapses all three of the first two into "hit". Still RETRIEVAL
    only: a chunk ranked first is not the same as a chunk the generator
    actually drew the answer from.

    Averaging this across questions gives Mean Reciprocal Rank, but do
    not average it across questions that have a different number of
    expected chunks — a multi-chunk question's "first" expected chunk is
    not comparable to a single-chunk question's only one. Report
    answerable_single and answerable_multi MRR separately.
    """
    expected = set(expected_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Iterable[str],
    k: int,
) -> float:
    """Fraction of expected chunks retrieved in the top k.

    For answerable_multi questions: hit_rate_at_k asks "did we get
    anything useful"; this asks "did we get all of it". A question whose
    answer needs three chunks and gets one back scores 1.0 on
    hit_rate_at_k and 0.33 here — the gap between the two numbers is the
    signal that generation is being asked to answer from a partial
    picture. Meaningless (returns 0.0 by convention, not an error) for a
    question with zero expected chunks; do not compute it for
    out_of_corpus, absent_but_plausible or cross_tenant_trap questions,
    which have none.
    """
    expected = set(expected_chunk_ids)
    if not expected:
        return 0.0
    top_k = set(retrieved_chunk_ids[:k])
    return len(expected & top_k) / len(expected)


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimJudgment:
    """One claim extracted from a generated answer, and whether a human or
    model judge found it supported by the retrieved chunks shown to the
    generator."""

    claim: str
    supported: bool


def faithfulness(judgments: Sequence[ClaimJudgment]) -> float:
    """Fraction of claims in an answer that are supported by a retrieved
    chunk — the anti-hallucination number.

    Does NOT measure whether the answer is correct, complete, or well
    retrieved. A faithful answer can still be faithfully wrong if every
    retrieved chunk was itself irrelevant (that failure belongs to
    hit_rate_at_k, not here), and a faithful answer can still omit facts
    the corpus actually contains (that belongs to recall_at_k). This
    number only asks: for what the answer *did* say, is there a receipt.

    The judgments are produced by a model, sampled and spot-checked by a
    human — this function does not do the judging, it only aggregates a
    judgment sequence someone else produced. An empty `judgments` sequence
    returns 0.0, matching "an answer that claims nothing is not faithful
    to anything," not 1.0 by vacuous truth — a refusal should never be
    scored through this function in the first place (see
    refusal_correctness).
    """
    if not judgments:
        return 0.0
    supported = sum(1 for j in judgments if j.supported)
    return supported / len(judgments)


@dataclass(frozen=True)
class CitationCheck:
    """One citation attached to a claim, and whether the cited chunk's
    text actually contains the claim it is attached to."""

    cited_chunk_id: str
    claim: str
    chunk_contains_claim: bool


def citation_accuracy(checks: Sequence[CitationCheck]) -> float:
    """Fraction of citations whose cited chunk actually contains the claim
    attached to it.

    Does NOT measure faithfulness. A claim can be true and supported by
    *some* retrieved chunk while being cited to the *wrong* one — that is
    a citation defect even though the claim itself is fine, and this
    function is the only one of the two that catches it. A client's
    security reviewer who clicks a citation and finds it does not say
    what the answer implies it says is exactly the failure this measures.
    An empty `checks` sequence returns 0.0, the same "nothing to point at
    is not accurate" convention as faithfulness.
    """
    if not checks:
        return 0.0
    correct = sum(1 for c in checks if c.chunk_contains_claim)
    return correct / len(checks)


@dataclass(frozen=True)
class RefusalResult:
    """The graded outcome for one golden question with a known expected
    behaviour ("answer" or "refuse") and what the system actually did."""

    question_id: str
    expected_behaviour: str  # "answer" | "refuse"
    system_refused: bool


@dataclass(frozen=True)
class RefusalCorrectness:
    """Both refusal directions, reported together on purpose — see
    refusal_correctness's docstring for why neither number alone is
    meaningful."""

    correct_refusal_rate: float  # of questions that SHOULD refuse, how many did
    false_refusal_rate: float  # of questions that SHOULD answer, how many wrongly refused
    n_should_refuse: int
    n_should_answer: int


def refusal_correctness(results: Sequence[RefusalResult]) -> RefusalCorrectness:
    """Of the questions that should be refused, how many were? And of the
    questions that should be answered, how many were wrongly refused?
    BOTH directions, always reported together.

    Does NOT measure whether an answered question was answered
    *correctly* — that is faithfulness and citation_accuracy's job. A
    system that refuses everything scores 1.0 on correct_refusal_rate and
    is useless; a system that never refuses scores 1.0 on
    (1 - false_refusal_rate) and fails cross_tenant_trap and
    out_of_corpus every time. Neither rate is a ceiling and a floor at
    the same time: `false_refusal_rate` is a ceiling — pushing it to zero
    by refusing more aggressively is a regression, not an improvement, and
    every other rate in this module is a floor (higher is better). This is
    the one metric in this file where "higher is better" does not hold in
    both fields it returns, which is exactly why the two fields are
    reported together and never collapsed into one blended accuracy
    number.

    Raises ValueError on an empty `results` sequence rather than
    returning a placeholder 0.0 for both fields — an empty result set
    with no should-refuse and no should-answer questions is not "perfect
    both ways," it is "nothing was measured," and those must not look the
    same on a dashboard.
    """
    if not results:
        raise ValueError("refusal_correctness: empty results — nothing was measured")

    should_refuse = [r for r in results if r.expected_behaviour == "refuse"]
    should_answer = [r for r in results if r.expected_behaviour == "answer"]

    correct_refusals = sum(1 for r in should_refuse if r.system_refused)
    false_refusals = sum(1 for r in should_answer if r.system_refused)

    correct_refusal_rate = (
        correct_refusals / len(should_refuse) if should_refuse else 0.0
    )
    false_refusal_rate = (
        false_refusals / len(should_answer) if should_answer else 0.0
    )

    return RefusalCorrectness(
        correct_refusal_rate=correct_refusal_rate,
        false_refusal_rate=false_refusal_rate,
        n_should_refuse=len(should_refuse),
        n_should_answer=len(should_answer),
    )


# ---------------------------------------------------------------------------
# Operational metrics
# ---------------------------------------------------------------------------


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile over an already-sorted sequence.

    Not interpolated. For small eval sets (75 questions, sometimes fewer
    once filtered per category) the difference between nearest-rank and
    linear interpolation is noise; nearest-rank is used because it always
    returns a value that was actually observed, which matters when this
    number gets quoted back to a client.
    """
    if not sorted_values:
        raise ValueError("_percentile: empty sequence")
    index = max(0, min(len(sorted_values) - 1, round(pct * (len(sorted_values) - 1))))
    return sorted_values[index]


@dataclass(frozen=True)
class StageLatency:
    """Wall-clock milliseconds for one pipeline stage on one query."""

    stage: str  # e.g. "embed", "vector_search", "fulltext_search", "rerank", "generate"
    ms: float


@dataclass(frozen=True)
class LatencyReport:
    p50_ms: float
    p95_ms: float
    by_stage_p50_ms: dict[str, float]
    by_stage_p95_ms: dict[str, float]


def latency_p50_p95(
    end_to_end_ms: Sequence[float],
    stage_latencies: Sequence[StageLatency] = (),
) -> LatencyReport:
    """End-to-end p50/p95 latency, and the same broken down per stage.

    Does NOT measure correctness at any stage — a fast wrong answer and a
    slow right answer both contribute only to this number, never to each
    other. The per-stage breakdown exists because "p95 is bad" is not
    actionable; "p95 is bad because rerank is bad" is. A stage missing
    from `stage_latencies` for some queries is silently excluded from that
    stage's percentile rather than treated as zero — a stage that ran on
    only some queries would otherwise report a falsely fast p95.
    """
    if not end_to_end_ms:
        raise ValueError("latency_p50_p95: no end-to-end samples")

    e2e_sorted = sorted(end_to_end_ms)
    by_stage: dict[str, list[float]] = {}
    for sample in stage_latencies:
        by_stage.setdefault(sample.stage, []).append(sample.ms)

    by_stage_p50 = {stage: _percentile(sorted(vals), 0.50) for stage, vals in by_stage.items()}
    by_stage_p95 = {stage: _percentile(sorted(vals), 0.95) for stage, vals in by_stage.items()}

    return LatencyReport(
        p50_ms=_percentile(e2e_sorted, 0.50),
        p95_ms=_percentile(e2e_sorted, 0.95),
        by_stage_p50_ms=by_stage_p50,
        by_stage_p95_ms=by_stage_p95,
    )


@dataclass(frozen=True)
class QueryCost:
    """USD cost of one query, broken down by the calls that produced it."""

    embedding_usd: float
    rerank_usd: float
    generation_usd: float

    @property
    def total_usd(self) -> float:
        return self.embedding_usd + self.rerank_usd + self.generation_usd


def usd_per_query(costs: Sequence[QueryCost]) -> float:
    """Mean total cost per query: embedding + rerank + generation.

    Does NOT measure cost per *correct* answer — a system that is cheap
    and wrong is indistinguishable from a system that is cheap and right
    by this number alone. Always report usd_per_query beside
    faithfulness and refusal_correctness on the same run, never in
    isolation; a cost figure with no accompanying quality figure invites
    optimizing the wrong thing.
    """
    if not costs:
        raise ValueError("usd_per_query: no cost samples")
    return sum(c.total_usd for c in costs) / len(costs)


def median_usd_per_query(costs: Sequence[QueryCost]) -> float:
    """Median total cost per query — less sensitive than the mean to the
    rare query that triggers a large rerank or retry.

    Report alongside, not instead of, usd_per_query: a mean far above the
    median means a small number of expensive queries are carrying the
    average, which the median alone would hide.
    """
    if not costs:
        raise ValueError("median_usd_per_query: no cost samples")
    return median(c.total_usd for c in costs)
