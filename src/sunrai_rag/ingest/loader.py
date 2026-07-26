"""Stream WebDataset shards, crop regions, and assemble a validated corpus.

The corpus is the single artefact every downstream stage consumes. It is
persisted as JSON so that ingestion runs once and the (expensive) OCR stage
never blocks iteration on retrieval or evaluation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..schemas import BBox, Chunk, Region, RegionType
from .ocr import OCRCache, OCRStats, TesseractEngine, indexable_regions, ocr_regions
from .regions import parse_page_annotation

log = logging.getLogger(__name__)


@dataclass
class Corpus:
    """Everything ingestion produces, with provenance intact."""

    regions: list[Region] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    ocr_stats: dict[str, Any] = field(default_factory=dict)
    page_count: int = 0

    @property
    def doc_ids(self) -> list[str]:
        return sorted({r.doc_id for r in self.regions})

    def region_by_id(self, region_id: str) -> Region | None:
        return next((r for r in self.regions if r.region_id == region_id), None)

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)

    def visual_regions(self) -> list[Region]:
        return [r for r in self.regions if r.is_visual]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "regions": [r.to_dict() for r in self.regions],
            "chunks": [c.to_dict() for c in self.chunks],
            "ocr_stats": self.ocr_stats,
            "page_count": self.page_count,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> Corpus:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        regions = [
            Region(
                region_id=r["region_id"],
                doc_id=r["doc_id"],
                page_id=r["page_id"],
                region_type=RegionType(r["region_type"]),
                bbox=BBox(**r["bbox"]),
                text=r.get("text"),
                image_path=r.get("image_path"),
                ocr_confidence=r.get("ocr_confidence"),
            )
            for r in payload["regions"]
        ]
        chunks = [Chunk(**c) for c in payload["chunks"]]
        return Corpus(
            regions=regions,
            chunks=chunks,
            ocr_stats=payload.get("ocr_stats", {}),
            page_count=payload.get("page_count", 0),
        )


class CorpusValidationError(ValueError):
    """Raised when the assembled corpus violates an invariant."""


def validate_corpus(corpus: Corpus, min_ocr_chars: int = 20) -> dict[str, Any]:
    """Assert the invariants every downstream stage relies on.

    This is the data-quality gate. It fails loudly rather than letting a
    malformed corpus surface later as an inexplicable retrieval result.
    """
    problems: list[str] = []

    if not corpus.regions:
        problems.append("corpus contains no regions")

    region_ids = [r.region_id for r in corpus.regions]
    if len(region_ids) != len(set(region_ids)):
        problems.append("duplicate region_id values")

    chunk_ids = [c.chunk_id for c in corpus.chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        problems.append("duplicate chunk_id values")

    valid_ids = set(region_ids)
    for chunk in corpus.chunks:
        dangling = [rid for rid in chunk.source_region_ids if rid not in valid_ids]
        if dangling:
            problems.append(f"chunk {chunk.chunk_id} cites unknown regions {dangling}")
            break  # one example is enough to fail the gate

    for region in corpus.regions:
        if region.bbox.is_degenerate():
            problems.append(f"degenerate bbox on {region.region_id}")
            break

    for chunk in corpus.chunks:
        if not chunk.text.strip():
            problems.append(f"empty chunk text on {chunk.chunk_id}")
            break

    if problems:
        raise CorpusValidationError("; ".join(problems))

    return {
        "regions": len(corpus.regions),
        "chunks": len(corpus.chunks),
        "documents": len(corpus.doc_ids),
        "pages": corpus.page_count,
        "visual_regions": len(corpus.visual_regions()),
        "ocr": corpus.ocr_stats,
    }


def build_chunks(regions: list[Region], cfg: Config) -> list[Chunk]:
    """Turn OCR'd regions into retrievable chunks.

    "region" strategy (default): one chunk per layout region. This keeps the
    chunk boundary aligned with the document's own visual structure, which
    makes provenance exact -- a cited chunk maps to one highlightable box on
    the page. That property is worth more here than marginal retrieval gains
    from arbitrary windowing.

    "window" strategy: concatenate a page's regions and slide a fixed window,
    provided for comparison.
    """
    usable = indexable_regions(regions, min_chars=cfg.ingest.min_ocr_chars)
    chunks: list[Chunk] = []

    if cfg.chunk.strategy == "region":
        for region in usable:
            text = (region.text or "").strip()
            if len(text) < cfg.chunk.min_chunk_chars:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{region.region_id}#c0",
                    doc_id=region.doc_id,
                    page_id=region.page_id,
                    text=text,
                    source_region_ids=[region.region_id],
                )
            )
        return chunks

    # Window strategy: group by page, preserving reading order (top then left).
    by_page: dict[tuple[str, str], list[Region]] = {}
    for region in usable:
        by_page.setdefault((region.doc_id, region.page_id), []).append(region)

    for (doc_id, page_id), page_regions in sorted(by_page.items()):
        page_regions.sort(key=lambda r: (r.bbox.y, r.bbox.x))
        spans: list[tuple[int, int, str]] = []
        cursor = 0
        parts: list[str] = []
        for region in page_regions:
            text = (region.text or "").strip()
            if not text:
                continue
            parts.append(text)
            spans.append((cursor, cursor + len(text), region.region_id))
            cursor += len(text) + 1
        page_text = " ".join(parts)

        step = cfg.chunk.window_chars - cfg.chunk.overlap_chars
        for i, start in enumerate(range(0, max(1, len(page_text)), step)):
            window = page_text[start : start + cfg.chunk.window_chars]
            if len(window.strip()) < cfg.chunk.min_chunk_chars:
                continue
            end = start + len(window)
            sources = [rid for s, e, rid in spans if s < end and e > start]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{page_id}#w{i:03d}",
                    doc_id=doc_id,
                    page_id=page_id,
                    text=window.strip(),
                    source_region_ids=sources,
                )
            )
    return chunks


def _shard_urls(cfg: Config) -> list[str]:
    return [
        cfg.ingest.dataset_url_template.format(i=i) for i in range(cfg.ingest.num_shards)
    ]


def iter_samples(cfg: Config) -> Iterator[tuple[str, Any, Any]]:
    """Yield (sample_key, PIL image, annotation payload) from the shards.

    Uses `datasets` in streaming mode so no full download is required. This
    is the only function in the package that touches the network.
    """
    from datasets import load_dataset  # lazy: keeps logic tests offline

    urls = _shard_urls(cfg)
    log.info("Streaming %d shard(s)", len(urls))
    ds = load_dataset(
        "webdataset", data_files={"train": urls}, split="train", streaming=True
    )

    for i, sample in enumerate(ds):
        if i >= cfg.ingest.max_pages:
            break
        key = sample.get("__key__") or f"sample_{i:06d}"
        image = sample.get("png") or sample.get("jpg") or sample.get("image")
        annotation = sample.get("json")
        if image is None or annotation is None:
            log.warning("Skipping sample %s: missing image or annotation", key)
            continue
        yield key, image, annotation


def crop_regions(image: Any, regions: list[Region], save_dir: Path | None) -> dict[str, Any]:
    """Crop each region from the page image; optionally persist visual crops.

    Visual crops are saved because the demo displays them as evidence and
    CLIP re-reads them during indexing.
    """
    crops: dict[str, Any] = {}
    for region in regions:
        left, top, right, bottom = region.bbox.to_xyxy()
        if right <= left or bottom <= top:
            continue
        crop = image.crop((left, top, right, bottom))
        crops[region.region_id] = crop
        if save_dir and region.is_visual:
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / f"{region.region_id.replace('#', '_')}.png"
            crop.save(out)
            region.image_path = str(out)
    return crops


def ingest(cfg: Config) -> Corpus:
    """Run the full ingestion pipeline: stream -> parse -> crop -> OCR -> chunk."""
    cfg.ensure_dirs()
    crop_dir = Path(cfg.paths.artifacts_dir) / "crops"
    cache = OCRCache(Path(cfg.paths.artifacts_dir) / "ocr_cache.json")
    engine = TesseractEngine(
        lang=cfg.ingest.ocr_lang,
        psm=cfg.ingest.ocr_psm,
        min_crop_height=cfg.ingest.ocr_min_crop_height,
    )

    all_regions: list[Region] = []
    total_stats = OCRStats()
    page_confidences: list[float] = []
    page_count = 0

    for key, image, annotation in iter_samples(cfg):
        width, height = image.size
        regions = parse_page_annotation(
            annotation,
            sample_key=key,
            page_width=width,
            page_height=height,
            min_side_px=cfg.ingest.min_region_side_px,
        )
        if not regions:
            continue
        crops = crop_regions(image, regions, crop_dir if cfg.ingest.save_crops else None)
        stats = ocr_regions(
            regions,
            crops,
            engine=engine,
            min_chars=cfg.ingest.min_ocr_chars,
            cache=cache,
        )
        total_stats.attempted += stats.attempted
        total_stats.empty += stats.empty
        total_stats.below_min_chars += stats.below_min_chars
        page_confidences.extend(
            r.ocr_confidence for r in regions if r.ocr_confidence is not None
        )
        all_regions.extend(regions)
        page_count += 1
        if page_count % 25 == 0:
            log.info("Ingested %d pages, %d regions", page_count, len(all_regions))

    if page_confidences:
        total_stats.mean_confidence = sum(page_confidences) / len(page_confidences)

    chunks = build_chunks(all_regions, cfg)
    corpus = Corpus(
        regions=all_regions,
        chunks=chunks,
        ocr_stats=total_stats.to_dict(),
        page_count=page_count,
    )
    summary = validate_corpus(corpus, min_ocr_chars=cfg.ingest.min_ocr_chars)
    log.info("Corpus validated: %s", summary)
    return corpus
