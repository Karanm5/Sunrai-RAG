"""OCR over layout-region crops.

Because the dataset ships page images without text, OCR is a load-bearing
stage, not a convenience. Two consequences we handle explicitly:

1. OCR quality bounds the whole system's ceiling. We record per-region
   confidence and expose an aggregate empty-rate so the report can state the
   limitation with a number rather than a hand-wave.
2. OCR is the slowest ingestion step. Results are cached per region id so
   re-running ingestion after a code change costs nothing.

The engine is injected (`OCREngine` protocol), which lets the test suite run
the full ingestion path with a deterministic fake and no Tesseract install.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..schemas import Region

log = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
# Ligatures and hyphenation artefacts that OCR reliably produces on PMC renders.
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")


class OCREngine(Protocol):
    """Anything that can turn an image crop into (text, mean_confidence)."""

    def read(self, image: Any) -> tuple[str, float]: ...


def find_tesseract_binary(explicit_path: str | None = None) -> str | None:
    """Locate the tesseract executable across platforms.

    On Linux and macOS tesseract is normally on PATH and this is a no-op. On
    Windows the common installers do not always add it, so pytesseract fails
    with a confusing "tesseract is not installed" error even when it is. We
    check the standard install locations before giving up.

    Returns the path to use, or None if PATH already resolves it.
    """
    import shutil

    if explicit_path:
        return explicit_path

    found = shutil.which("tesseract")
    if found:
        return found

    candidates = [
        # Windows: default locations for the UB Mannheim and official builds
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        # macOS Homebrew, both architectures
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


@dataclass
class TesseractEngine:
    """Default engine: pytesseract over the region crop.

    `psm=6` ("assume a single uniform block of text") suits PubLayNet
    regions, which are already segmented into single blocks by the layout
    annotation -- letting Tesseract re-segment tends to hurt.

    `min_crop_height` upscales small crops before OCR. This is not a guess:
    measured on rendered fixtures (scripts/verify_ocr.py), resolution is the
    dominant OCR failure mode -- far more than blur or JPEG artefacts -- and
    upscaling recovers most of the loss:

        page quality      no upscale      upscale to 200px
        ~100 DPI          0.742 mean      0.990 mean   (worst 0.271 -> 0.962)
        ~80 DPI           0.189 mean      0.821 mean

    The gain saturates around 200px, so a larger target only costs time.
    Set to 0 to disable.
    """

    lang: str = "eng"
    psm: int = 6
    min_crop_height: int = 200
    tesseract_cmd: str | None = None
    _configured: bool = field(default=False, init=False, repr=False)

    def _ensure_binary(self) -> None:
        """Point pytesseract at the binary once, with a clear error if absent."""
        if self._configured:
            return
        import pytesseract

        path = find_tesseract_binary(self.tesseract_cmd)
        if path:
            pytesseract.pytesseract.tesseract_cmd = path
            log.info("Using tesseract at %s", path)
        else:
            raise RuntimeError(
                "Could not find the tesseract executable.\n"
                "  Windows: install from "
                "https://github.com/UB-Mannheim/tesseract/wiki, then either "
                "tick 'Add to PATH' during setup or set ingest.tesseract_cmd "
                "in your config to the full path of tesseract.exe\n"
                "  macOS:   brew install tesseract\n"
                "  Linux:   sudo apt-get install tesseract-ocr tesseract-ocr-eng"
            )
        self._configured = True

    @staticmethod
    def _resample_filter() -> Any:
        """LANCZOS filter if Pillow is present, otherwise None.

        Kept optional on purpose. This module advertises itself as runnable
        without the heavy image stack, and the test suite relies on that to
        run on a minimal dependency set in CI. Importing Pillow
        unconditionally here silently broke that contract once already.

        Also tolerates Pillow's move of the constants onto `Image.Resampling`.
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            return None
        namespace = getattr(PILImage, "Resampling", PILImage)
        return getattr(namespace, "LANCZOS", None)

    def _prepare(self, image: Any) -> Any:
        """Upscale crops that are too small for reliable OCR."""
        if not self.min_crop_height:
            return image
        height = getattr(image, "height", None)
        if not height or height >= self.min_crop_height:
            return image

        factor = self.min_crop_height / height
        size = (max(1, int(image.width * factor)), self.min_crop_height)
        resample = self._resample_filter()
        return image.resize(size, resample) if resample is not None else image.resize(size)

    def read(self, image: Any) -> tuple[str, float]:
        import pytesseract  # imported lazily so logic tests need no binary

        self._ensure_binary()
        image = self._prepare(image)
        config = f"--psm {self.psm}"
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=config, output_type=pytesseract.Output.DICT
        )
        words: list[str] = []
        confs: list[float] = []
        # strict=False on purpose: tesseract occasionally returns ragged
        # text/conf lists, and dropping the tail is preferable to raising.
        for word, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
            if not word or not word.strip():
                continue
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                continue
            if conf_val < 0:  # Tesseract uses -1 for non-text boxes
                continue
            words.append(word)
            confs.append(conf_val)
        text = " ".join(words)
        mean_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return text, mean_conf


def clean_ocr_text(raw: str) -> str:
    """Normalise the predictable OCR artefacts on scientific page renders.

    Deliberately conservative: fixes ligatures, joins hyphenated line breaks,
    and collapses whitespace. We do not attempt spelling correction, which
    would risk silently rewriting scientific terms and units.
    """
    if not raw:
        return ""
    text = raw
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


@dataclass
class OCRStats:
    """Aggregate OCR quality, reported in the ingestion log and the report."""

    attempted: int = 0
    empty: int = 0
    below_min_chars: int = 0
    mean_confidence: float = 0.0

    @property
    def empty_rate(self) -> float:
        return self.empty / self.attempted if self.attempted else 0.0

    @property
    def usable_rate(self) -> float:
        if not self.attempted:
            return 0.0
        return (self.attempted - self.empty - self.below_min_chars) / self.attempted

    def to_dict(self) -> dict[str, float | int]:
        return {
            "attempted": self.attempted,
            "empty": self.empty,
            "below_min_chars": self.below_min_chars,
            "empty_rate": round(self.empty_rate, 4),
            "usable_rate": round(self.usable_rate, 4),
            "mean_confidence": round(self.mean_confidence, 4),
        }


class OCRCache:
    """Disk cache keyed by region id. Makes re-ingestion effectively free."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._data: dict[str, list[Any]] = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, region_id: str) -> tuple[str, float] | None:
        hit = self._data.get(region_id)
        if isinstance(hit, list) and len(hit) == 2:
            return str(hit[0]), float(hit[1])
        return None

    def put(self, region_id: str, text: str, conf: float) -> None:
        self._data[region_id] = [text, conf]

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")


def ocr_regions(
    regions: Sequence[Region],
    crops: dict[str, Any],
    engine: OCREngine,
    min_chars: int = 20,
    cache: OCRCache | None = None,
) -> OCRStats:
    """OCR every textual region in place; return quality statistics.

    Regions whose OCR yields fewer than `min_chars` usable characters keep
    their text (for provenance) but are counted separately so the caller can
    exclude them from the index. Visual regions are skipped entirely.
    """
    stats = OCRStats()
    confidences: list[float] = []

    for region in regions:
        if not region.is_textual:
            continue
        crop = crops.get(region.region_id)
        if crop is None:
            continue

        stats.attempted += 1
        cached = cache.get(region.region_id) if cache else None
        if cached is not None:
            text, conf = cached
        else:
            raw, conf = engine.read(crop)
            text = clean_ocr_text(raw)
            if cache:
                cache.put(region.region_id, text, conf)

        region.text = text
        region.ocr_confidence = conf
        confidences.append(conf)

        if not text:
            stats.empty += 1
        elif len(text) < min_chars:
            stats.below_min_chars += 1

    if confidences:
        stats.mean_confidence = sum(confidences) / len(confidences)
    if cache:
        cache.flush()
    return stats


def indexable_regions(regions: Sequence[Region], min_chars: int = 20) -> list[Region]:
    """Textual regions with enough recovered text to be worth indexing."""
    return [
        r
        for r in regions
        if r.is_textual and r.text and len(r.text.strip()) >= min_chars
    ]
