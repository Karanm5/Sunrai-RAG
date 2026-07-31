"""Verify the ingestion path with REAL OCR against known ground truth.

Everything here is the production code path, real Tesseract, real cropping,
real chunking, real validation. Only the pages are synthetic, and that is
deliberate: because we rendered them, we know exactly what the text should
be, so OCR quality is *measurable* rather than eyeballed.

This closes the largest untested gap in the build. The logic tests use a fake
OCR engine; this proves the real one works end to end.
"""

from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sunrai_rag.config import Config, set_global_seeds  # noqa: E402
from sunrai_rag.ingest.loader import (  # noqa: E402
    Corpus,
    build_chunks,
    crop_regions,
    validate_corpus,
)
from sunrai_rag.ingest.ocr import OCRCache, TesseractEngine, ocr_regions  # noqa: E402
from sunrai_rag.ingest.regions import parse_page_annotation  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pages"


def similarity(a: str, b: str) -> float:
    """Character-level similarity; 1.0 means OCR recovered the text exactly."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def main() -> int:
    cfg = Config()
    cfg.ingest.min_ocr_chars = 20
    set_global_seeds(cfg.seed)

    print("=" * 72)
    print("REAL OCR VERIFICATION  (production code path, real Tesseract)")
    print("=" * 72)

    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    engine = TesseractEngine(lang=cfg.ingest.ocr_lang, psm=cfg.ingest.ocr_psm)
    cache = OCRCache(None)  # no cache: we want genuine OCR work every run

    all_regions = []
    scores: list[float] = []
    total_attempted = total_empty = 0

    for entry in manifest:
        key = entry["key"]
        image = Image.open(FIXTURES / f"{key}.png")
        payload = json.loads((FIXTURES / f"{key}.json").read_text())

        regions = parse_page_annotation(
            payload, sample_key=key, page_width=image.size[0],
            page_height=image.size[1], min_side_px=cfg.ingest.min_region_side_px,
        )
        assert len(regions) == len(payload["annotations"]), (
            f"{key}: parsed {len(regions)} of {len(payload['annotations'])} annotations"
        )

        crops = crop_regions(image, regions, save_dir=None)
        stats = ocr_regions(
            regions, crops, engine=engine,
            min_chars=cfg.ingest.min_ocr_chars, cache=cache,
        )
        total_attempted += stats.attempted
        total_empty += stats.empty
        all_regions.extend(regions)

        # Compare recovered text against what we rendered.
        for region, truth in zip(regions, entry["truth"], strict=True):
            if not region.is_textual:
                continue
            score = similarity(region.text or "", truth["expected_text"])
            scores.append(score)

    print(f"\nPages processed:        {len(manifest)}")
    print(f"Regions parsed:         {len(all_regions)}")
    print(f"Textual regions OCR'd:  {total_attempted}")
    print(f"Empty OCR results:      {total_empty}")

    mean_sim = sum(scores) / len(scores) if scores else 0.0
    worst = min(scores) if scores else 0.0
    print("\nOCR fidelity vs ground truth:")
    print(f"  mean similarity:      {mean_sim:.3f}")
    print(f"  worst region:         {worst:.3f}")
    print(f"  regions above 0.90:   {sum(1 for s in scores if s > 0.90)}/{len(scores)}")

    confidences = [r.ocr_confidence for r in all_regions if r.ocr_confidence is not None]
    if confidences:
        print(f"  mean tesseract conf:  {sum(confidences)/len(confidences):.3f}")

    print("\nSample recovered text:")
    for region in all_regions:
        if region.is_textual and region.text:
            print(f'  [{region.region_type.value}] "{region.text[:88]}..."')
            break

    # Full downstream path on real OCR output.
    chunks = build_chunks(all_regions, cfg)
    corpus = Corpus(regions=all_regions, chunks=chunks, page_count=len(manifest))
    summary = validate_corpus(corpus, min_ocr_chars=cfg.ingest.min_ocr_chars)
    print(f"\nCorpus built and validated: {summary}")

    print("\n" + "=" * 72)
    print("ASSERTIONS")
    print("=" * 72)
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  {'PASS' if condition else 'FAIL'}  {label}{detail}")
        ok = ok and condition

    check("OCR recovers text at high fidelity", mean_sim > 0.90,
          f" (mean {mean_sim:.3f})")
    check("no textual region produced empty OCR", total_empty == 0)
    check("every annotation became a region",
          len(all_regions) == sum(len(json.loads((FIXTURES / f"{e['key']}.json").read_text())["annotations"]) for e in manifest))
    check("chunks were produced from real OCR text", len(chunks) > 0,
          f" ({len(chunks)} chunks)")
    check("all chunk provenance resolves", all(
        c.source_region_ids and all(
            rid in {r.region_id for r in all_regions} for rid in c.source_region_ids)
        for c in chunks))
    check("visual regions excluded from text chunks", not (
        {rid for c in chunks for rid in c.source_region_ids}
        & {r.region_id for r in all_regions if r.is_visual}))

    # The property the whole project depends on: figure captions hold values
    # that never appear in body text.
    body = " ".join(r.text or "" for r in all_regions if r.is_textual).lower()
    figure_only = [r for r in all_regions if r.is_visual]
    check("figures were detected as visual regions", len(figure_only) > 0,
          f" ({len(figure_only)})")
    check("accuracy values absent from body text", "80.5" not in body and "84.5" not in body)

    print("=" * 72)
    print("REAL OCR VERIFICATION PASSED" if ok else "VERIFICATION FAILED")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
