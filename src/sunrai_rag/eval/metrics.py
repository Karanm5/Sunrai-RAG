"""Retrieval and answer-quality metrics.

Definitions are stated explicitly because "Recall@k" means different things
in different papers, and a comparison is only meaningful if the reader knows
which convention produced the numbers.

Retrieval metrics treat relevance as binary: a retrieved item is relevant if
its id is in the gold set for that question.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold items appearing in the top k.

    With a single gold item this reduces to hit-rate (1.0 or 0.0). With
    several, partial credit is given. Returns 0.0 when the gold set is empty,
    since no question can be answered from nothing.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    found = gold_set.intersection(retrieved[:k])
    return len(found) / len(gold_set)


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of the top k that is relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not retrieved:
        return 0.0
    top = retrieved[:k]
    gold_set = set(gold)
    return sum(1 for item in top if item in gold_set) / len(top)


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """1 / rank of the first relevant item; 0.0 if none retrieved.

    Averaged over questions this gives MRR. Sensitive to the top of the
    ranking, which is what matters when only the top few chunks are passed
    to the generator.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for position, item in enumerate(retrieved, start=1):
        if item in gold_set:
            return 1.0 / position
    return 0.0


def dcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Discounted cumulative gain with binary relevance, log2(rank+1) discount."""
    if k <= 0:
        raise ValueError("k must be positive")
    gold_set = set(gold)
    total = 0.0
    for position, item in enumerate(retrieved[:k], start=1):
        if item in gold_set:
            total += 1.0 / math.log2(position + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """DCG normalised by the ideal ranking (all gold items placed first)."""
    if k <= 0:
        raise ValueError("k must be positive")
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, gold, k) / idcg


@dataclass
class RetrievalScores:
    """Aggregated retrieval metrics for one system on one question set."""

    n_questions: int
    recall: dict[int, float]
    precision: dict[int, float]
    ndcg: dict[int, float]
    mrr: float

    def to_row(self, primary_k: int) -> dict[str, float | int]:
        return {
            "n_questions": self.n_questions,
            f"recall@{primary_k}": round(self.recall.get(primary_k, 0.0), 4),
            f"precision@{primary_k}": round(self.precision.get(primary_k, 0.0), 4),
            f"ndcg@{primary_k}": round(self.ndcg.get(primary_k, 0.0), 4),
            "mrr": round(self.mrr, 4),
        }

    def to_dict(self) -> dict:
        return {
            "n_questions": self.n_questions,
            "recall": {str(k): round(v, 4) for k, v in self.recall.items()},
            "precision": {str(k): round(v, 4) for k, v in self.precision.items()},
            "ndcg": {str(k): round(v, 4) for k, v in self.ndcg.items()},
            "mrr": round(self.mrr, 4),
        }


def evaluate_retrieval(
    retrieved_per_question: Sequence[Sequence[str]],
    gold_per_question: Sequence[Sequence[str]],
    k_values: Sequence[int],
) -> RetrievalScores:
    """Aggregate retrieval metrics across a question set (macro-averaged).

    Macro-averaging (mean over questions, each weighted equally) is used so a
    handful of questions with many gold regions cannot dominate the headline.
    """
    if len(retrieved_per_question) != len(gold_per_question):
        raise ValueError("retrieved and gold lists must align")
    n = len(retrieved_per_question)
    if n == 0:
        return RetrievalScores(0, {k: 0.0 for k in k_values}, {k: 0.0 for k in k_values}, {k: 0.0 for k in k_values}, 0.0)

    recall = {
        k: mean(
            recall_at_k(r, g, k)
            for r, g in zip(retrieved_per_question, gold_per_question, strict=True)
        )
        for k in k_values
    }
    precision = {
        k: mean(
            precision_at_k(r, g, k)
            for r, g in zip(retrieved_per_question, gold_per_question, strict=True)
        )
        for k in k_values
    }
    ndcg = {
        k: mean(
            ndcg_at_k(r, g, k)
            for r, g in zip(retrieved_per_question, gold_per_question, strict=True)
        )
        for k in k_values
    }
    mrr = mean(
        reciprocal_rank(r, g)
        for r, g in zip(retrieved_per_question, gold_per_question, strict=True)
    )
    return RetrievalScores(n, recall, precision, ndcg, mrr)


# --------------------------------------------------------------------------
# Answer-quality metrics
# --------------------------------------------------------------------------


def citation_validity(cited_ids: Sequence[str], known_ids: set[str]) -> float:
    """Fraction of citations that point at items that actually exist.

    A cheap, judge-free integrity check: it catches a generator inventing
    chunk ids, which is a failure mode that faithfulness scoring can miss.
    """
    if not cited_ids:
        return 0.0
    return sum(1 for c in cited_ids if c in known_ids) / len(cited_ids)


def answer_has_support(answer_text: str, provenance_empty: bool) -> bool:
    """True when a non-refusal answer is backed by at least some evidence.

    An answer that declines to answer ("not stated in the provided context")
    is *correct* behaviour when retrieval found nothing, so it is not counted
    as unsupported.
    """
    refusal_markers = (
        "not stated",
        "not provided",
        "cannot answer",
        "no relevant",
        "insufficient",
        "not found in",
    )
    lowered = answer_text.lower().strip()
    is_refusal = any(marker in lowered for marker in refusal_markers)
    if is_refusal:
        return True
    return not provenance_empty


def aggregate_judge_scores(scores: Sequence[float]) -> dict[str, float]:
    """Mean and a normal-approximation 95% interval for judge scores.

    The interval is reported because a judged mean over ~60 questions carries
    real uncertainty; quoting it without a spread would overstate precision.
    """
    if not scores:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    n = len(scores)
    mu = mean(scores)
    if n < 2:
        return {"mean": round(mu, 4), "ci95_low": round(mu, 4), "ci95_high": round(mu, 4), "n": n}
    variance = sum((s - mu) ** 2 for s in scores) / (n - 1)
    stderr = math.sqrt(variance / n)
    margin = 1.96 * stderr
    return {
        "mean": round(mu, 4),
        "ci95_low": round(max(0.0, mu - margin), 4),
        "ci95_high": round(min(1.0, mu + margin), 4),
        "n": n,
    }
