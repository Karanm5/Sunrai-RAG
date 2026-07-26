"""Run the baseline-vs-enhanced comparison and write the results table.

The output is deliberately segmented. A single blended Recall number would
hide the finding that matters: the enhanced system should win where the
baseline is structurally blind (visual and multi-hop questions) and roughly
tie elsewhere. Reporting the segments is what turns a demo into evidence.

Integrity guard: `run_comparison` refuses to write headline results when the
LLM backend is the test stub, so a stub run can never be mistaken for a real
result.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import Provenance, QAItem
from .metrics import (
    RetrievalScores,
    aggregate_judge_scores,
    answer_has_support,
    citation_validity,
    evaluate_retrieval,
)

log = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
Assess an answer produced by a retrieval system.

Question: {question}
Evidence given to the system:
{evidence}
Answer produced: {answer}
Reference answer: {reference}

Score two things from 0.0 to 1.0:
- "faithfulness": is every claim in the answer supported by the evidence?
  (An answer that correctly says the evidence is insufficient scores 1.0.)
- "relevance": does the answer address the question?

Return ONLY valid JSON: {{"faithfulness": 0.0, "relevance": 0.0}}
"""


class StubResultsError(RuntimeError):
    """Raised when a stub-backend run tries to write headline results."""


@dataclass
class SystemResult:
    """One system's scores, overall and per query segment."""

    system_name: str
    overall: RetrievalScores
    by_segment: dict[str, RetrievalScores]
    judge: dict[str, Any]
    mean_latency_s: float
    citation_validity: float
    support_rate: float

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "overall": self.overall.to_dict(),
            "by_segment": {k: v.to_dict() for k, v in self.by_segment.items()},
            "judge": self.judge,
            "mean_latency_s": round(self.mean_latency_s, 4),
            "citation_validity": round(self.citation_validity, 4),
            "support_rate": round(self.support_rate, 4),
        }


def _provenance_ids(provenance: Provenance) -> list[str]:
    """Retrieved ids in rank order, interleaved across modalities.

    Uses `ranked_evidence_ids()` rather than `all_cited_ids()`: concatenating
    text-then-visual would push every visual hit past position k, making
    Recall@k structurally incapable of crediting visual retrieval. See the
    docstring on Provenance.ranked_evidence_ids for the full reasoning.

    Gold sets are region ids, so chunk ids are mapped back to their source
    regions by the caller before scoring.
    """
    return provenance.ranked_evidence_ids()


def _to_region_ids(item_ids: Sequence[str], chunk_to_regions: dict[str, list[str]]) -> list[str]:
    """Map retrieved ids into region-id space, preserving rank order.

    Chunk ids resolve to their source regions; region ids (from the visual
    retriever) pass through unchanged. Without this, the two retrievers would
    be scored in incompatible id spaces.
    """
    out: list[str] = []
    for item_id in item_ids:
        for region_id in chunk_to_regions.get(item_id, [item_id]):
            if region_id not in out:
                out.append(region_id)
    return out


def evaluate_system(
    system: Any,
    qa_items: Sequence[QAItem],
    chunk_to_regions: dict[str, list[str]],
    known_ids: set[str],
    k_values: Sequence[int],
    judge_llm: Any = None,
    judge_sample_size: int = 0,
    generate_answers: bool = True,
) -> SystemResult:
    """Score one system across the full question set."""
    retrieved_all: list[list[str]] = []
    gold_all: list[list[str]] = []
    per_segment: dict[str, tuple[list[list[str]], list[list[str]]]] = {}
    latencies: list[float] = []
    citation_scores: list[float] = []
    support_flags: list[bool] = []
    judge_faith: list[float] = []
    judge_rel: list[float] = []

    for index, item in enumerate(qa_items):
        if generate_answers and hasattr(system, "answer"):
            answer = system.answer(item.question)
            provenance = answer.provenance
            if answer.latency_s is not None:
                latencies.append(answer.latency_s)
            support_flags.append(
                answer_has_support(answer.answer_text, provenance.is_empty())
            )
            from ..rag.pipelines import extract_citations

            cited = extract_citations(answer.answer_text)
            if cited:
                citation_scores.append(citation_validity(cited, known_ids))
        else:
            provenance = system.retrieve(item.question)
            answer = None

        retrieved = _to_region_ids(_provenance_ids(provenance), chunk_to_regions)
        retrieved_all.append(retrieved)
        gold_all.append(item.gold_region_ids)

        segment = item.query_type.value
        per_segment.setdefault(segment, ([], []))
        per_segment[segment][0].append(retrieved)
        per_segment[segment][1].append(item.gold_region_ids)

        # Judge a bounded sample -- judging every question is costly and the
        # extra precision does not change the conclusion.
        if judge_llm is not None and answer is not None and index < judge_sample_size:
            evidence = "\n".join(
                f"[{c.item_id}]" for c in provenance.text_chunks + provenance.visual_regions
            )
            raw = judge_llm.complete(
                _JUDGE_PROMPT.format(
                    question=item.question,
                    evidence=evidence or "(none)",
                    answer=answer.answer_text,
                    reference=item.gold_answer or "(not available)",
                )
            )
            scores = _parse_judge(raw)
            if scores:
                judge_faith.append(scores[0])
                judge_rel.append(scores[1])

    return SystemResult(
        system_name=getattr(system, "name", "unknown"),
        overall=evaluate_retrieval(retrieved_all, gold_all, k_values),
        by_segment={
            name: evaluate_retrieval(r, g, k_values)
            for name, (r, g) in sorted(per_segment.items())
        },
        judge={
            "faithfulness": aggregate_judge_scores(judge_faith),
            "relevance": aggregate_judge_scores(judge_rel),
        },
        mean_latency_s=sum(latencies) / len(latencies) if latencies else 0.0,
        citation_validity=(
            sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
        ),
        support_rate=(
            sum(support_flags) / len(support_flags) if support_flags else 0.0
        ),
    )


def _parse_judge(raw: str) -> tuple[float, float] | None:
    try:
        text = (raw or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        payload = json.loads(text[start : end + 1])
        faith = float(payload.get("faithfulness", 0.0))
        rel = float(payload.get("relevance", 0.0))
        if not (0.0 <= faith <= 1.0 and 0.0 <= rel <= 1.0):
            return None
        return faith, rel
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def compute_deltas(
    baseline: SystemResult, enhanced: SystemResult, primary_k: int
) -> dict[str, Any]:
    """Enhanced minus baseline, overall and per segment.

    Deltas are the headline. Because constructed-QA circularity inflates both
    systems similarly, the gap between them survives that bias better than
    either absolute value does.
    """
    def gap(metric: str, b: RetrievalScores, e: RetrievalScores) -> float:
        if metric == "mrr":
            return round(e.mrr - b.mrr, 4)
        return round(e.recall.get(primary_k, 0.0) - b.recall.get(primary_k, 0.0), 4)

    segments = sorted(set(baseline.by_segment) | set(enhanced.by_segment))
    return {
        "overall": {
            f"recall@{primary_k}": gap("recall", baseline.overall, enhanced.overall),
            "mrr": gap("mrr", baseline.overall, enhanced.overall),
        },
        "by_segment": {
            segment: {
                f"recall@{primary_k}": gap(
                    "recall",
                    baseline.by_segment.get(segment, baseline.overall),
                    enhanced.by_segment.get(segment, enhanced.overall),
                ),
                "mrr": gap(
                    "mrr",
                    baseline.by_segment.get(segment, baseline.overall),
                    enhanced.by_segment.get(segment, enhanced.overall),
                ),
            }
            for segment in segments
        },
    }


def run_comparison(
    systems: dict[str, Any],
    qa_items: Sequence[QAItem],
    chunk_to_regions: dict[str, list[str]],
    known_ids: set[str],
    cfg,
    judge_llm: Any = None,
    allow_stub: bool = False,
) -> dict[str, Any]:
    """Evaluate every system and assemble the comparison payload."""
    if cfg.llm.backend == "stub" and not allow_stub:
        raise StubResultsError(
            "Refusing to produce headline results with the stub LLM backend. "
            "Set llm.backend to 'anthropic' or 'local' for reportable numbers, "
            "or pass allow_stub=True for a smoke test."
        )

    results: dict[str, SystemResult] = {}
    for label, system in systems.items():
        log.info("Evaluating %s", label)
        results[label] = evaluate_system(
            system,
            qa_items,
            chunk_to_regions,
            known_ids,
            k_values=cfg.eval.k_values,
            judge_llm=judge_llm if cfg.eval.judge_enabled else None,
            judge_sample_size=cfg.eval.judge_sample_size,
            generate_answers=hasattr(system, "answer"),
        )

    payload: dict[str, Any] = {
        "config": {
            "seed": cfg.seed,
            "llm_backend": cfg.llm.backend,
            "text_embed_model": cfg.models.text_embed_model,
            "clip_model": cfg.models.clip_model,
            "top_k": cfg.retrieval.top_k,
            "kg_extractor": cfg.kg.extractor,
        },
        "n_questions": len(qa_items),
        "systems": {label: result.to_dict() for label, result in results.items()},
    }

    if "baseline" in results and "enhanced" in results:
        payload["deltas"] = compute_deltas(
            results["baseline"], results["enhanced"], cfg.eval.primary_k
        )
    return payload


def write_results(payload: dict[str, Any], results_dir: str | Path, primary_k: int) -> Path:
    """Write results.json plus a flat comparison.csv for the report."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    rows: list[str] = [
        f"system,segment,n_questions,recall@{primary_k},precision@{primary_k},"
        f"ndcg@{primary_k},mrr"
    ]
    for label, system in payload["systems"].items():
        for segment_name, scores in [("overall", system["overall"])] + sorted(
            system["by_segment"].items()
        ):
            rows.append(
                ",".join([
                    label,
                    segment_name,
                    str(scores["n_questions"]),
                    str(scores["recall"].get(str(primary_k), 0.0)),
                    str(scores["precision"].get(str(primary_k), 0.0)),
                    str(scores["ndcg"].get(str(primary_k), 0.0)),
                    str(scores["mrr"]),
                ])
            )
    csv_path = results_dir / "comparison.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path


def format_summary(payload: dict[str, Any], primary_k: int) -> str:
    """Human-readable summary printed at the end of an evaluation run."""
    lines = [
        "",
        "=" * 68,
        f"RESULTS  ({payload['n_questions']} questions, "
        f"backend={payload['config']['llm_backend']}, seed={payload['config']['seed']})",
        "=" * 68,
        f"{'system':<26}{'segment':<20}{'R@'+str(primary_k):>8}{'MRR':>8}",
        "-" * 68,
    ]
    for label, system in payload["systems"].items():
        for segment_name, scores in [("overall", system["overall"])] + sorted(
            system["by_segment"].items()
        ):
            lines.append(
                f"{label:<26}{segment_name:<20}"
                f"{scores['recall'].get(str(primary_k), 0.0):>8.3f}"
                f"{scores['mrr']:>8.3f}"
            )
    if "deltas" in payload:
        lines += ["-" * 68, "DELTAS (enhanced - baseline)", "-" * 68]
        for segment, delta in payload["deltas"]["by_segment"].items():
            lines.append(
                f"{'':<26}{segment:<20}"
                f"{delta[f'recall@{primary_k}']:>+8.3f}{delta['mrr']:>+8.3f}"
            )
    lines.append("=" * 68)
    return "\n".join(lines)
