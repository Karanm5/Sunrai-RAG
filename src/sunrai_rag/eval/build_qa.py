"""Construct the evaluation question set.

PubLayNet ships no QA ground truth, so the question set has to be built. The
obvious approach, ask an LLM to write a question from a chunk, then check
retrieval finds that chunk, is circular: the question inherits the chunk's
vocabulary, so retrieval succeeds for the wrong reason and Recall is
inflated for every system.

Three safeguards, all reported in the write-up rather than buried:

1. **Paraphrase constraint.** The generator is instructed to avoid reusing
   distinctive terms from the source. `lexical_overlap` measures how well
   that held, and questions above a threshold are dropped.
2. **Segmentation by query type.** Questions are labelled by *where the
   answer lives*. The comparison is then read per segment, so the headline
   is "enhanced wins on visual questions" rather than a single blended
   number that circularity has inflated.
3. **Deltas over absolutes.** Circularity inflates both systems, so the gap
   between them remains informative even where absolute values do not.

Nothing here removes the limitation. It bounds it and makes it measurable.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

from ..index.bm25 import tokenize
from ..schemas import Chunk, QAItem, QueryType, Region

log = logging.getLogger(__name__)

_TEXT_QUESTION_PROMPT = """\
Write ONE question that this passage from a scientific paper answers.

Rules:
- The question must be answerable from the passage alone.
- Do NOT reuse the passage's distinctive words. Paraphrase heavily.
- Do not mention "the passage", "the text", or "this study".
- Return ONLY valid JSON: {{"question": "...", "answer": "..."}}

Passage:
{text}
"""

_VISUAL_QUESTION_PROMPT = """\
This is the caption of a figure or table in a scientific paper.

Write ONE question whose answer appears ONLY in that figure/table, for
example a reported numeric value, or which item performs best.

Rules:
- Do NOT reuse the caption's distinctive words. Paraphrase heavily.
- Do not mention "the figure", "the caption", or "the table" by number.
- Return ONLY valid JSON: {{"question": "...", "answer": "..."}}

Caption:
{text}
"""

_MULTIHOP_QUESTION_PROMPT = """\
Two related excerpts from the same scientific paper:

Excerpt A: {text_a}

Excerpt B: {text_b}

Write ONE question that requires BOTH excerpts to answer, neither alone
should be sufficient.

Rules:
- Do NOT reuse distinctive words from either excerpt. Paraphrase heavily.
- Return ONLY valid JSON: {{"question": "...", "answer": "..."}}
"""


def lexical_overlap(question: str, source_text: str) -> float:
    """Fraction of the question's content words that appear in the source.

    The circularity proxy. A question sharing most of its vocabulary with its
    source region is retrievable by lexical matching alone and tells us
    nothing about semantic retrieval quality.
    """
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return 0.0
    s_tokens = set(tokenize(source_text))
    return len(q_tokens & s_tokens) / len(q_tokens)


def _parse_qa_json(raw: str) -> tuple[str, str] | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    if not question or len(question) < 10:
        return None
    return question, answer


def build_qa_set(
    chunks: Sequence[Chunk],
    visual_regions: Sequence[Region],
    llm,
    n_per_type: int = 25,
    max_overlap: float = 0.6,
    seed: int = 42,
) -> list[QAItem]:
    """Build a segmented, overlap-filtered question set.

    Returns questions of three types. Every item records the gold region ids
    that genuinely contain its answer, that is the ground truth retrieval
    is scored against.
    """
    import random

    rng = random.Random(seed)
    items: list[QAItem] = []
    rejected = 0

    # --- text-answerable: answer lives in body text ---
    text_pool = [c for c in chunks if len(c.text) > 200]
    rng.shuffle(text_pool)
    for chunk in text_pool:
        if sum(1 for i in items if i.query_type is QueryType.TEXT_ANSWERABLE) >= n_per_type:
            break
        parsed = _parse_qa_json(llm.complete(_TEXT_QUESTION_PROMPT.format(text=chunk.text[:2000])))
        if parsed is None:
            continue
        question, answer = parsed
        if lexical_overlap(question, chunk.text) > max_overlap:
            rejected += 1
            continue
        items.append(QAItem(
            qa_id=f"text_{len(items):04d}", question=question,
            query_type=QueryType.TEXT_ANSWERABLE,
            gold_region_ids=list(chunk.source_region_ids),
            gold_answer=answer, doc_id=chunk.doc_id,
        ))

    # --- visual-requiring: answer lives in a figure/table region ---
    visual_pool = [r for r in visual_regions if r.text and len(r.text) > 20]
    rng.shuffle(visual_pool)
    for region in visual_pool:
        if sum(1 for i in items if i.query_type is QueryType.VISUAL_REQUIRING) >= n_per_type:
            break
        parsed = _parse_qa_json(llm.complete(_VISUAL_QUESTION_PROMPT.format(text=region.text)))
        if parsed is None:
            continue
        question, answer = parsed
        if lexical_overlap(question, region.text or "") > max_overlap:
            rejected += 1
            continue
        items.append(QAItem(
            qa_id=f"visual_{len(items):04d}", question=question,
            query_type=QueryType.VISUAL_REQUIRING,
            gold_region_ids=[region.region_id],
            gold_answer=answer, doc_id=region.doc_id,
        ))

    # --- multi-hop: answer needs two regions from the same document ---
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    multi_docs = [d for d, cs in by_doc.items() if len(cs) >= 2]
    rng.shuffle(multi_docs)
    for doc_id in multi_docs:
        if sum(1 for i in items if i.query_type is QueryType.MULTI_HOP) >= n_per_type:
            break
        a, b = rng.sample(by_doc[doc_id], 2)
        parsed = _parse_qa_json(llm.complete(
            _MULTIHOP_QUESTION_PROMPT.format(text_a=a.text[:1000], text_b=b.text[:1000])))
        if parsed is None:
            continue
        question, answer = parsed
        if lexical_overlap(question, a.text + " " + b.text) > max_overlap:
            rejected += 1
            continue
        items.append(QAItem(
            qa_id=f"multihop_{len(items):04d}", question=question,
            query_type=QueryType.MULTI_HOP,
            gold_region_ids=list(a.source_region_ids) + list(b.source_region_ids),
            gold_answer=answer, doc_id=doc_id,
        ))

    log.info("Built %d questions (%d rejected for lexical overlap > %.2f)",
             len(items), rejected, max_overlap)
    return items


def qa_set_stats(items: Sequence[QAItem]) -> dict:
    """Summarise the question set for the report's methodology section."""
    by_type: dict[str, int] = {}
    for item in items:
        by_type[item.query_type.value] = by_type.get(item.query_type.value, 0) + 1
    gold_counts = [len(i.gold_region_ids) for i in items]
    return {
        "total": len(items),
        "by_type": dict(sorted(by_type.items())),
        "mean_gold_regions": round(sum(gold_counts) / len(gold_counts), 2) if gold_counts else 0.0,
    }


def save_qa_set(items: Sequence[QAItem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([i.to_dict() for i in items], indent=2), encoding="utf-8")


def load_qa_set(path: str | Path) -> list[QAItem]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [QAItem.from_dict(d) for d in payload]
