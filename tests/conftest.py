"""Shared fixtures.

The synthetic corpus is built to exercise the project's central hypothesis:
some facts appear only in a figure region and never in the body text. A
text-only retriever therefore *cannot* reach them, while a multimodal or
graph-enhanced retriever can. Tests assert exactly that asymmetry.
"""

from __future__ import annotations

import pytest

from sunrai_rag.config import Config
from sunrai_rag.index.vector_store import VectorStore
from sunrai_rag.represent.embedders import HashingEmbedder
from sunrai_rag.schemas import BBox, Chunk, Region, RegionType


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64)


@pytest.fixture
def regions() -> list[Region]:
    """Three text regions and two visual regions across two documents.

    Note the deliberate gap: the accuracy *value* for the random forest
    appears only in the figure caption region, not in any body text.
    """
    return [
        Region(
            region_id="PMC001_00001#r000",
            doc_id="PMC001",
            page_id="00001",
            region_type=RegionType.TITLE,
            bbox=BBox(10, 10, 400, 40),
            text="Deep learning methods for cardiac risk prediction",
            ocr_confidence=0.95,
        ),
        Region(
            region_id="PMC001_00001#r001",
            doc_id="PMC001",
            page_id="00001",
            region_type=RegionType.TEXT,
            bbox=BBox(10, 60, 400, 200),
            text=(
                "We trained a random forest classifier and a convolutional "
                "neural network on the cardiac cohort. Performance was "
                "assessed using accuracy and AUC as shown in Figure 2."
            ),
            ocr_confidence=0.91,
        ),
        Region(
            region_id="PMC001_00001#r002",
            doc_id="PMC001",
            page_id="00001",
            region_type=RegionType.FIGURE,
            bbox=BBox(10, 280, 400, 300),
            # Caption-only text: the numeric result lives here alone.
            text="Figure 2: accuracy of 92.4% for the random forest model",
            image_path="/tmp/fig2.png",
        ),
        Region(
            region_id="PMC002_00003#r000",
            doc_id="PMC002",
            page_id="00003",
            region_type=RegionType.TEXT,
            bbox=BBox(20, 20, 380, 180),
            text=(
                "Support vector machines were applied to the imaging corpus. "
                "We report sensitivity and specificity in Table 1."
            ),
            ocr_confidence=0.88,
        ),
        Region(
            region_id="PMC002_00003#r001",
            doc_id="PMC002",
            page_id="00003",
            region_type=RegionType.TABLE,
            bbox=BBox(20, 220, 380, 250),
            text="Table 1: sensitivity 0.81, specificity 0.77",
            image_path="/tmp/table1.png",
        ),
    ]


@pytest.fixture
def chunks(regions) -> list[Chunk]:
    """One chunk per region carrying usable text."""
    return [
        Chunk(
            chunk_id=f"{r.region_id}#c0",
            doc_id=r.doc_id,
            page_id=r.page_id,
            text=r.text or "",
            source_region_ids=[r.region_id],
        )
        for r in regions
        if r.text
    ]


@pytest.fixture
def text_store(chunks, embedder) -> VectorStore:
    embeddings = embedder.embed_texts([c.text for c in chunks])
    return VectorStore([c.chunk_id for c in chunks], embeddings, modality="text")


@pytest.fixture
def config() -> Config:
    return Config()
