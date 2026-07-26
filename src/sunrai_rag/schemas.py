"""Core data structures shared across the pipeline.

Every stage of the pipeline speaks in these types. Keeping them in one place
(rather than passing raw dicts) is what makes the ingestion -> retrieval ->
generation -> evaluation chain testable in isolation.

PubLayNet category ids are fixed by the dataset:
    1=text, 2=title, 3=list, 4=table, 5=figure
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RegionType(str, Enum):
    """The five PubLayNet layout classes."""

    TEXT = "text"
    TITLE = "title"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"


# PubLayNet's COCO category_id -> RegionType. Fixed by the dataset definition.
CATEGORY_ID_TO_TYPE: dict[int, RegionType] = {
    1: RegionType.TEXT,
    2: RegionType.TITLE,
    3: RegionType.LIST,
    4: RegionType.TABLE,
    5: RegionType.FIGURE,
}

# Region types whose content is words (-> OCR) vs. pixels (-> vision model).
TEXTUAL_TYPES: frozenset[RegionType] = frozenset(
    {RegionType.TEXT, RegionType.TITLE, RegionType.LIST}
)
VISUAL_TYPES: frozenset[RegionType] = frozenset({RegionType.FIGURE, RegionType.TABLE})


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel coordinates, COCO convention (x, y, w, h)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Convert to (left, top, right, bottom) integer pixels for cropping."""
        return (
            int(round(self.x)),
            int(round(self.y)),
            int(round(self.x + self.w)),
            int(round(self.y + self.h)),
        )

    def clipped(self, width: int, height: int) -> BBox:
        """Clip the box to lie inside a page of the given size."""
        x0 = min(max(self.x, 0.0), float(width))
        y0 = min(max(self.y, 0.0), float(height))
        x1 = min(max(self.x + self.w, 0.0), float(width))
        y1 = min(max(self.y + self.h, 0.0), float(height))
        return BBox(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))

    def is_degenerate(self, min_side: float = 2.0) -> bool:
        return self.w < min_side or self.h < min_side


@dataclass
class Region:
    """One annotated layout region on one page.

    `text` is populated by OCR for textual types; `image_path` points at the
    saved crop for visual types. A region is the atomic unit of provenance:
    every answer the system produces cites region ids.
    """

    region_id: str
    doc_id: str
    page_id: str
    region_type: RegionType
    bbox: BBox
    text: str | None = None
    image_path: str | None = None
    ocr_confidence: float | None = None

    @property
    def is_textual(self) -> bool:
        return self.region_type in TEXTUAL_TYPES

    @property
    def is_visual(self) -> bool:
        return self.region_type in VISUAL_TYPES

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["region_type"] = self.region_type.value
        return d


@dataclass
class Chunk:
    """A retrievable unit of text, traceable back to its source regions.

    Chunks are what the text retriever indexes. `source_region_ids` is the
    link that lets an answer cite the exact page regions it came from.
    """

    chunk_id: str
    doc_id: str
    page_id: str
    text: str
    source_region_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredItem:
    """A retrieved item with its score and which retriever produced it."""

    item_id: str
    score: float
    modality: str = "text"  # "text" | "image" | "graph" | "fused"
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Provenance:
    """The receipts for an answer: everything the system used to produce it.

    This is the explainability payload. An answer without a populated
    Provenance is treated as a failure by the test suite.
    """

    text_chunks: list[ScoredItem] = field(default_factory=list)
    visual_regions: list[ScoredItem] = field(default_factory=list)
    graph_entities: list[str] = field(default_factory=list)
    graph_paths: list[list[str]] = field(default_factory=list)
    retrieval_strategy: str = "unknown"

    def all_cited_ids(self) -> list[str]:
        """Every cited id. Order is text-then-visual; use
        `ranked_evidence_ids()` for anything rank-sensitive."""
        return [c.item_id for c in self.text_chunks] + [
            v.item_id for v in self.visual_regions
        ]

    def ranked_evidence_ids(self) -> list[str]:
        """Evidence ids ordered by rank, interleaved across modalities.

        Why this exists (it fixes a real measurement bug): the two retrievers
        return separate id spaces, so simply concatenating text-then-visual
        puts every visual hit *after* all k text hits. With top_k=5 the
        visual retriever's rank-1 result lands at position 6 and Recall@5 can
        never credit it -- the enhanced system looks blind to figures even
        when its visual retrieval is perfect.

        Round-robin by rank gives each modality fair access to the top of the
        list, which also reflects how the evidence is actually presented to
        the generator (both modalities appear together in the prompt).
        """
        text = sorted(
            self.text_chunks, key=lambda i: (i.rank if i.rank is not None else 10**6)
        )
        visual = sorted(
            self.visual_regions, key=lambda i: (i.rank if i.rank is not None else 10**6)
        )
        out: list[str] = []
        for position in range(max(len(text), len(visual))):
            if position < len(text):
                out.append(text[position].item_id)
            if position < len(visual):
                out.append(visual[position].item_id)
        seen: set[str] = set()
        return [i for i in out if not (i in seen or seen.add(i))]

    def is_empty(self) -> bool:
        return not self.text_chunks and not self.visual_regions

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_chunks": [c.to_dict() for c in self.text_chunks],
            "visual_regions": [v.to_dict() for v in self.visual_regions],
            "graph_entities": list(self.graph_entities),
            "graph_paths": [list(p) for p in self.graph_paths],
            "retrieval_strategy": self.retrieval_strategy,
        }


@dataclass
class Answer:
    """A generated answer plus the evidence that produced it."""

    query: str
    answer_text: str
    provenance: Provenance
    system_name: str = "unknown"
    latency_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer_text": self.answer_text,
            "provenance": self.provenance.to_dict(),
            "system_name": self.system_name,
            "latency_s": self.latency_s,
        }


class QueryType(str, Enum):
    """Query segmentation used for the honest baseline-vs-enhanced comparison.

    The central hypothesis: TEXT_ANSWERABLE queries should be a near-tie,
    while VISUAL_REQUIRING and MULTI_HOP queries are where the enhanced
    system should win. Reporting these separately is what makes the
    comparison informative rather than a single inflated number.
    """

    TEXT_ANSWERABLE = "text_answerable"
    VISUAL_REQUIRING = "visual_requiring"
    MULTI_HOP = "multi_hop"


@dataclass
class QAItem:
    """One evaluation question with its gold evidence.

    `gold_region_ids` is the ground truth for retrieval metrics: the regions
    that genuinely contain the answer.
    """

    qa_id: str
    question: str
    query_type: QueryType
    gold_region_ids: list[str] = field(default_factory=list)
    gold_answer: str | None = None
    doc_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["query_type"] = self.query_type.value
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QAItem:
        return QAItem(
            qa_id=d["qa_id"],
            question=d["question"],
            query_type=QueryType(d["query_type"]),
            gold_region_ids=list(d.get("gold_region_ids", [])),
            gold_answer=d.get("gold_answer"),
            doc_id=d.get("doc_id"),
        )
