"""Parse PubLayNet layout annotations into typed Region objects.

Design note (important, and worth being able to defend at interview):
the WebDataset shards pair a page render (`.png`) with a layout annotation
(`.json`). The annotation gives bounding boxes and a category id -- it does
NOT contain the page text. That single fact drives the whole ingestion
design: text has to be recovered by OCR over the region crops (see ocr.py).

The parser is deliberately defensive. Published WebDataset conversions of
PubLayNet differ slightly in how they nest the annotation list, so we accept
several shapes rather than assuming one and crashing on shard 2.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ..schemas import CATEGORY_ID_TO_TYPE, BBox, Region, RegionType


class AnnotationParseError(ValueError):
    """Raised when an annotation cannot be interpreted at all."""


def _coerce_annotation_list(payload: Any) -> list[dict[str, Any]]:
    """Pull the list of region annotations out of whatever shape we were given.

    Accepted shapes, in order of preference:
      1. {"annotations": [...]}          - COCO-style page record
      2. [...]                            - bare list of annotations
      3. {"regions"|"objects"|"boxes": [...]} - common re-export variants

    Raw `str`/`bytes` are parsed first: whether the WebDataset loader decodes
    a `.json` member into a Python object or hands back the raw text depends
    on the loader and its version, and getting that wrong fails ingestion on
    the very first sample.
    """
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AnnotationParseError(
                f"Annotation payload was text but is not valid JSON: {exc}"
            ) from exc

    if isinstance(payload, dict):
        for key in ("annotations", "regions", "objects", "boxes"):
            value = payload.get(key)
            if isinstance(value, list):
                return [a for a in value if isinstance(a, dict)]
        raise AnnotationParseError(
            f"No annotation list found in mapping with keys {sorted(payload)}"
        )
    if isinstance(payload, list):
        return [a for a in payload if isinstance(a, dict)]
    raise AnnotationParseError(f"Unsupported annotation payload type: {type(payload)}")


def _extract_bbox(ann: dict[str, Any]) -> BBox | None:
    """Read a bbox in COCO (x, y, w, h) form, tolerating a few variants."""
    raw = ann.get("bbox")
    if raw is None:
        for key in ("box", "bounding_box", "bbox_xywh"):
            if key in ann:
                raw = ann[key]
                break
    if raw is None:
        return None
    if isinstance(raw, dict):
        try:
            return BBox(
                float(raw["x"]), float(raw["y"]), float(raw["w"]), float(raw["h"])
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            x, y, w, h = (float(v) for v in raw[:4])
        except (TypeError, ValueError):
            return None
        # Heuristic: if the last two values look like absolute corner
        # coordinates rather than a width/height, convert xyxy -> xywh.
        if w > x and h > y and (w - x) > 0 and (h - y) > 0 and ann.get("format") == "xyxy":
            return BBox(x, y, w - x, h - y)
        return BBox(x, y, w, h)
    return None


def _extract_region_type(ann: dict[str, Any]) -> RegionType | None:
    """Resolve the layout class from either a category id or a name."""
    cat_id = ann.get("category_id", ann.get("category", ann.get("label")))
    if isinstance(cat_id, bool):
        return None
    if isinstance(cat_id, (int, float)):
        return CATEGORY_ID_TO_TYPE.get(int(cat_id))
    if isinstance(cat_id, str):
        name = cat_id.strip().lower()
        try:
            return RegionType(name)
        except ValueError:
            if name.isdigit():
                return CATEGORY_ID_TO_TYPE.get(int(name))
    return None


def parse_page_annotation(
    payload: Any,
    sample_key: str,
    page_width: int,
    page_height: int,
    min_side_px: float = 8.0,
) -> list[Region]:
    """Convert one page's annotation payload into validated Regions.

    Regions are dropped (not raised on) when they are unusable: unknown
    category, missing bbox, or degenerate after clipping to the page. The
    caller logs the drop rate -- a silently high drop rate is a data-quality
    signal worth reporting, so we return counts via `parse_stats`.

    `sample_key` is the WebDataset key, e.g. "PMC4991227_00003", from which
    doc_id ("PMC4991227") and page ("00003") are derived.
    """
    doc_id, page_id = split_sample_key(sample_key)
    annotations = _coerce_annotation_list(payload)

    regions: list[Region] = []
    for idx, ann in enumerate(annotations):
        region_type = _extract_region_type(ann)
        if region_type is None:
            continue
        bbox = _extract_bbox(ann)
        if bbox is None:
            continue
        bbox = bbox.clipped(page_width, page_height)
        if bbox.is_degenerate(min_side=min_side_px):
            continue
        regions.append(
            Region(
                region_id=f"{sample_key}#r{idx:03d}",
                doc_id=doc_id,
                page_id=page_id,
                region_type=region_type,
                bbox=bbox,
            )
        )
    return regions


def parse_stats(payload: Any, regions: Iterable[Region]) -> dict[str, int]:
    """Report how many annotations survived parsing, for data-quality logging."""
    try:
        total = len(_coerce_annotation_list(payload))
    except AnnotationParseError:
        total = 0
    kept = len(list(regions))
    return {"annotations_total": total, "regions_kept": kept, "dropped": total - kept}


def split_sample_key(sample_key: str) -> tuple[str, str]:
    """Split "PMC4991227_00003" into ("PMC4991227", "00003").

    Falls back to (key, "0") for keys that don't carry a page suffix, so an
    unexpected naming scheme degrades rather than crashes.
    """
    key = sample_key.strip()
    if "_" in key:
        doc, _, page = key.rpartition("_")
        if doc:
            return doc, page
    return key, "0"
