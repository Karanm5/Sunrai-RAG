"""The two systems under comparison.

`BaselineRAG`  - text-only dense retrieval over OCR'd chunks.
`EnhancedRAG`  - the same text retrieval, fused with CLIP visual retrieval,
                 plus knowledge-graph expansion.

They share one interface (`answer(query) -> Answer`) so the evaluation
harness treats them identically. That symmetry is what makes the comparison
fair: same corpus, same generator, same top-k, same prompt template. The only
differences are the ones under test, visual evidence and structured
knowledge.

Anything else varying between the two would confound the result, which is
why the shared parts live in `_generate`.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..index.fusion import reciprocal_rank_fusion
from ..index.vector_store import VectorStore
from ..kg.graph import KnowledgeGraph
from ..schemas import Answer, Chunk, Provenance, Region, ScoredItem

_SYSTEM_PROMPT = (
    "You answer questions about scientific documents using only the evidence "
    "provided. Never use outside knowledge. If the evidence does not contain "
    "the answer, reply exactly: 'Not stated in the provided context.'"
)

_ANSWER_TEMPLATE = """\
Answer the question using ONLY the evidence below.

{evidence}

Question: {query}

Instructions:
- Answer in 1-3 sentences.
- Cite the evidence you used with its bracketed id, e.g. [{example_id}].
- If the evidence is insufficient, reply exactly: Not stated in the provided context.

Answer:"""

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


class RAGSystem(Protocol):
    name: str

    def answer(self, query: str) -> Answer: ...
    def retrieve(self, query: str) -> Provenance: ...


def _format_text_evidence(chunks: Sequence[Chunk], items: Sequence[ScoredItem]) -> list[str]:
    by_id = {c.chunk_id: c for c in chunks}
    lines: list[str] = []
    for item in items:
        chunk = by_id.get(item.item_id)
        if chunk is None:
            continue
        lines.append(f"[{chunk.chunk_id}] (text) {chunk.text}")
    return lines


def _format_visual_evidence(
    regions: Sequence[Region], items: Sequence[ScoredItem]
) -> list[str]:
    """Describe retrieved figures/tables for the generator.

    A caption is used when available; otherwise the region is named by type
    and page. Naming it at all matters: it lets the model say "Figure 2 on
    page 3 is the relevant evidence" instead of silently ignoring the image.
    """
    by_id = {r.region_id: r for r in regions}
    lines: list[str] = []
    for item in items:
        region = by_id.get(item.item_id)
        if region is None:
            continue
        descriptor = region.text or f"{region.region_type.value} region"
        lines.append(
            f"[{region.region_id}] ({region.region_type.value} on page "
            f"{region.page_id} of {region.doc_id}) {descriptor}"
        )
    return lines


def _generate(llm: Any, query: str, evidence_lines: Sequence[str]) -> str:
    """Shared generation step. Identical for both systems by design."""
    if not evidence_lines:
        return "Not stated in the provided context."
    example_id = _CITATION_RE.sub("", evidence_lines[0].split("]")[0].lstrip("[")) or "id"
    prompt = _ANSWER_TEMPLATE.format(
        evidence="\n\n".join(evidence_lines),
        query=query,
        example_id=example_id,
    )
    return llm.complete(prompt, system=_SYSTEM_PROMPT).strip()


def extract_citations(answer_text: str) -> list[str]:
    """Pull bracketed ids out of a generated answer."""
    return [m.strip() for m in _CITATION_RE.findall(answer_text) if m.strip()]


@dataclass
class BaselineRAG:
    """Text-only RAG: dense retrieval over OCR'd chunks, then generate.

    This is the control condition. It can only ever see words, so any
    question whose answer lives solely in a figure is structurally out of
    reach, which is precisely what the segmented evaluation measures.
    """

    chunks: list[Chunk]
    text_store: VectorStore
    text_embedder: Any
    llm: Any
    top_k: int = 5
    name: str = field(default="baseline_text_only", init=False)

    def retrieve(self, query: str) -> Provenance:
        query_vector = self.text_embedder.embed_texts([query])
        hits = self.text_store.search(query_vector, k=self.top_k)
        return Provenance(text_chunks=hits, retrieval_strategy="dense_text")

    def answer(self, query: str) -> Answer:
        started = time.perf_counter()
        provenance = self.retrieve(query)
        evidence = _format_text_evidence(self.chunks, provenance.text_chunks)
        text = _generate(self.llm, query, evidence)
        return Answer(
            query=query,
            answer_text=text,
            provenance=provenance,
            system_name=self.name,
            latency_s=time.perf_counter() - started,
        )


@dataclass
class EnhancedRAG:
    """Multimodal + knowledge-graph RAG.

    Three evidence sources, fused by rank:
      1. dense text retrieval (identical to the baseline)
      2. CLIP retrieval over figure/table crops, using the query text
         embedded into CLIP's shared space
      3. knowledge-graph expansion from entities named in the query

    The graph contributes regions that neither embedding retriever would
    surface, typically the region reporting a *result* when the query names
    a *method*. Those arrive as a third ranking rather than being appended,
    so fusion arbitrates rather than one source always winning.
    """

    chunks: list[Chunk]
    regions: list[Region]
    text_store: VectorStore
    text_embedder: Any
    llm: Any
    image_store: VectorStore | None = None
    image_embedder: Any = None
    # Dense index over text RECOVERED FROM visual regions by OCR. CLIP is
    # trained on natural photographs and performs at chance on cropped
    # scientific tables; the information in those regions is overwhelmingly
    # textual, so a text encoder is the right instrument for it. This is
    # still cross-modal, the content comes from a non-text modality and is
    # deliberately absent from the baseline's index.
    visual_text_store: VectorStore | None = None
    kg: KnowledgeGraph | None = None
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    graph_hops: int = 1
    max_graph_regions: int = 5
    # Text retrieval is the strongest single signal, so graph candidates are
    # down-weighted rather than fused as equals. Fusing at parity measurably
    # displaced correct text results.
    text_weight: float = 1.0
    graph_weight: float = 0.3
    name: str = field(default="enhanced_multimodal_kg", init=False)

    def _region_to_chunk_ids(self, region_ids: Sequence[str]) -> list[str]:
        """Map graph-derived regions onto the chunks that carry their text."""
        wanted = set(region_ids)
        return [
            c.chunk_id
            for c in self.chunks
            if any(rid in wanted for rid in c.source_region_ids)
        ]

    def retrieve(self, query: str) -> Provenance:
        # 1. Dense text retrieval over the same index the baseline uses.
        text_vector = self.text_embedder.embed_texts([query])
        text_hits = self.text_store.search(text_vector, k=self.candidate_k)

        # 2. Cross-modal retrieval over visual regions, by two routes.
        visual_hits: list[ScoredItem] = []

        # 2a. OCR'd text from tables and figures, embedded with the text
        # encoder. This is the route that actually works on scientific
        # documents.
        if self.visual_text_store is not None and len(self.visual_text_store):
            visual_hits.extend(
                self.visual_text_store.search(text_vector, k=self.top_k)
            )

        # 2b. CLIP over the raw crops. Retained because it can catch purely
        # pictorial figures that carry no readable text, but it is fused
        # after the OCR route rather than relied upon.
        if (
            self.image_store is not None
            and self.image_embedder is not None
            and len(self.image_store)
        ):
            clip_query = self.image_embedder.embed_texts([query])
            clip_hits = self.image_store.search(clip_query, k=self.top_k)
            if visual_hits:
                visual_hits = reciprocal_rank_fusion(
                    [visual_hits, clip_hits], k=self.rrf_k, top_n=self.top_k
                )
            else:
                visual_hits = clip_hits

        # 3. Graph expansion from entities mentioned in the query.
        graph_entities: list[str] = []
        graph_paths: list[list[str]] = []
        graph_hits: list[ScoredItem] = []
        if self.kg is not None:
            graph_entities = self.kg.link_query_entities(query)
            if graph_entities:
                graph_paths = self.kg.expansion_paths(
                    graph_entities, hops=self.graph_hops
                )
                region_ids = self.kg.regions_from_expansion(
                    graph_entities,
                    hops=self.graph_hops,
                    max_regions=self.max_graph_regions,
                )
                graph_chunk_ids = self._region_to_chunk_ids(region_ids)
                graph_hits = [
                    ScoredItem(cid, 1.0 / (i + 1), "graph", i + 1)
                    for i, cid in enumerate(graph_chunk_ids[: self.max_graph_regions])
                ]

        # Fuse the text rankings. Visual hits stay separate: they are a
        # different id space (regions, not chunks), and merging id spaces in
        # one ranking would corrupt the retrieval metrics.
        rankings = [text_hits]
        weights = [self.text_weight]
        if graph_hits:
            rankings.append(graph_hits)
            weights.append(self.graph_weight)
        fused = reciprocal_rank_fusion(
            rankings, k=self.rrf_k, weights=weights, top_n=self.top_k
        )

        return Provenance(
            text_chunks=fused,
            visual_regions=visual_hits,
            graph_entities=graph_entities,
            graph_paths=graph_paths,
            retrieval_strategy="rrf(dense_text+graph)+visual_text+clip",
        )

    def answer(self, query: str) -> Answer:
        started = time.perf_counter()
        provenance = self.retrieve(query)
        evidence = _format_text_evidence(self.chunks, provenance.text_chunks)
        evidence += _format_visual_evidence(self.regions, provenance.visual_regions)
        if self.kg is not None and provenance.graph_entities:
            reached = self.kg.expand(provenance.graph_entities, hops=self.graph_hops)
            subgraph = self.kg.describe_subgraph(reached)
            if subgraph and "no graph relations" not in subgraph:
                evidence.append(f"[knowledge_graph] Structured relations:\n{subgraph}")
        text = _generate(self.llm, query, evidence)
        return Answer(
            query=query,
            answer_text=text,
            provenance=provenance,
            system_name=self.name,
            latency_s=time.perf_counter() - started,
        )


@dataclass
class RandomRetriever:
    """Random retrieval floor.

    Included so the report can show how far above chance the systems sit. A
    dense retriever that barely beats this is broken, and without the floor
    that failure is invisible in an absolute Recall number.
    """

    chunk_ids: list[str]
    seed: int = 42
    top_k: int = 5
    name: str = field(default="random_floor", init=False)

    def retrieve(self, query: str) -> Provenance:
        # Seeded per query so results are reproducible but not identical
        # across questions.
        rng = np.random.default_rng(
            abs(hash((self.seed, query))) % (2**32)
        )
        k = min(self.top_k, len(self.chunk_ids))
        if k == 0:
            return Provenance(retrieval_strategy="random")
        picked = rng.choice(len(self.chunk_ids), size=k, replace=False)
        return Provenance(
            text_chunks=[
                ScoredItem(self.chunk_ids[int(i)], 0.0, "random", rank + 1)
                for rank, i in enumerate(picked)
            ],
            retrieval_strategy="random",
        )


@dataclass
class BM25Retriever:
    """Lexical retrieval floor, wrapped in the shared retriever interface.

    Reported alongside the dense systems so the write-up can answer the
    question an absolute Recall number cannot: does semantic retrieval
    actually beat plain word matching? If it does not, the embedding stack is
    not earning its cost, and without this row that failure stays invisible.
    """

    index: Any  # BM25Index
    top_k: int = 5
    name: str = field(default="bm25_lexical_floor", init=False)

    def retrieve(self, query: str) -> Provenance:
        return Provenance(
            text_chunks=self.index.search(query, k=self.top_k),
            retrieval_strategy="bm25_lexical",
        )
