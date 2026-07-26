"""Ingestion: annotation parsing, OCR cleaning, chunking, corpus validation."""
import pytest

from sunrai_rag.ingest.loader import Corpus, CorpusValidationError, build_chunks, validate_corpus
from sunrai_rag.ingest.ocr import (
    OCRCache,
    TesseractEngine,
    clean_ocr_text,
    indexable_regions,
    ocr_regions,
)
from sunrai_rag.ingest.regions import (
    AnnotationParseError,
    parse_page_annotation,
    parse_stats,
    split_sample_key,
)
from sunrai_rag.schemas import BBox, Chunk, Region, RegionType

# ---------- annotation parsing ----------

def test_parses_coco_style_page_record():
    payload = {"annotations": [
        {"bbox": [10, 20, 100, 50], "category_id": 1},
        {"bbox": [10, 90, 100, 50], "category_id": 5},
    ]}
    regions = parse_page_annotation(payload, "PMC123_00001", 500, 500)
    assert [r.region_type for r in regions] == [RegionType.TEXT, RegionType.FIGURE]
    assert regions[0].doc_id == "PMC123" and regions[0].page_id == "00001"


@pytest.mark.parametrize("payload", [
    [{"bbox": [1, 2, 30, 40], "category_id": 2}],
    {"regions": [{"bbox": [1, 2, 30, 40], "category_id": 2}]},
    {"objects": [{"bbox": [1, 2, 30, 40], "category_id": 2}]},
])
def test_accepts_known_schema_variants(payload):
    """Re-exports of PubLayNet nest annotations differently; all must work."""
    assert len(parse_page_annotation(payload, "PMC1_0", 200, 200)) == 1


def test_rejects_uninterpretable_payload():
    with pytest.raises(AnnotationParseError):
        parse_page_annotation("not an annotation", "PMC1_0", 100, 100)


def test_drops_unknown_category_and_missing_bbox():
    payload = {"annotations": [
        {"bbox": [1, 1, 50, 50], "category_id": 99},   # unknown class
        {"category_id": 1},                             # no bbox
        {"bbox": [1, 1, 50, 50], "category_id": 1},     # valid
    ]}
    assert len(parse_page_annotation(payload, "PMC1_0", 200, 200)) == 1


def test_drops_degenerate_regions_after_clipping():
    payload = {"annotations": [
        {"bbox": [0, 0, 1, 1], "category_id": 1},        # too small
        {"bbox": [990, 990, 100, 100], "category_id": 1},# off-page
        {"bbox": [0, 0, 100, 100], "category_id": 1},    # good
    ]}
    regions = parse_page_annotation(payload, "PMC1_0", 500, 500, min_side_px=8.0)
    assert len(regions) == 1


def test_crops_never_exceed_page_bounds():
    payload = {"annotations": [{"bbox": [400, 400, 500, 500], "category_id": 1}]}
    r = parse_page_annotation(payload, "PMC1_0", 500, 500)[0]
    left, top, right, bottom = r.bbox.to_xyxy()
    assert 0 <= left < right <= 500 and 0 <= top < bottom <= 500


def test_string_category_names_resolve():
    payload = {"annotations": [{"bbox": [1, 1, 50, 50], "category": "figure"}]}
    assert parse_page_annotation(payload, "P_0", 100, 100)[0].region_type is RegionType.FIGURE


def test_region_ids_unique_within_page():
    payload = {"annotations": [{"bbox": [1, 1, 50, 50], "category_id": 1}] * 5}
    regions = parse_page_annotation(payload, "PMC1_0", 200, 200)
    assert len({r.region_id for r in regions}) == 5


def test_parse_stats_reports_drops():
    payload = {"annotations": [
        {"bbox": [1, 1, 50, 50], "category_id": 1},
        {"bbox": [1, 1, 50, 50], "category_id": 99},
    ]}
    regions = parse_page_annotation(payload, "P_0", 100, 100)
    stats = parse_stats(payload, regions)
    assert stats == {"annotations_total": 2, "regions_kept": 1, "dropped": 1}


@pytest.mark.parametrize("key,expected", [
    ("PMC4991227_00003", ("PMC4991227", "00003")),
    ("PMC1_2_00007", ("PMC1_2", "00007")),
    ("noPageSuffix", ("noPageSuffix", "0")),
])
def test_sample_key_splitting(key, expected):
    assert split_sample_key(key) == expected


# ---------- OCR ----------

def test_ocr_cleaning_fixes_ligatures_hyphenation_whitespace():
    assert clean_ocr_text("classi\ufb01cation") == "classification"
    assert clean_ocr_text("signi-  ficant") == "significant"
    assert clean_ocr_text("  a   b  ") == "a b"
    assert clean_ocr_text("") == ""


def test_ocr_cleaning_preserves_scientific_tokens():
    """Must not 'correct' units or symbols."""
    assert clean_ocr_text("p < 0.05, R2 = 0.917") == "p < 0.05, R2 = 0.917"


class _FakeEngine:
    def __init__(self, mapping): self.mapping = mapping
    def read(self, image): return self.mapping.get(image, ("", 0.0))


def test_ocr_populates_text_and_skips_visual_regions():
    text_region = Region("r1", "d", "p", RegionType.TEXT, BBox(0, 0, 10, 10))
    figure_region = Region("r2", "d", "p", RegionType.FIGURE, BBox(0, 0, 10, 10))
    engine = _FakeEngine({"img1": ("Hello world of science", 0.9)})
    stats = ocr_regions([text_region, figure_region], {"r1": "img1", "r2": "img2"}, engine)
    assert text_region.text == "Hello world of science"
    assert figure_region.text is None          # visual regions untouched
    assert stats.attempted == 1                # figure not attempted


def test_ocr_stats_track_empty_and_short_results():
    regions = [Region(f"r{i}", "d", "p", RegionType.TEXT, BBox(0, 0, 9, 9)) for i in range(3)]
    engine = _FakeEngine({"a": ("", 0.0), "b": ("tiny", 0.5), "c": ("a" * 50, 0.9)})
    stats = ocr_regions(regions, {"r0": "a", "r1": "b", "r2": "c"}, engine, min_chars=20)
    assert stats.empty == 1 and stats.below_min_chars == 1
    assert stats.empty_rate == pytest.approx(1 / 3)
    assert stats.usable_rate == pytest.approx(1 / 3)


def test_ocr_cache_prevents_recompute(tmp_path):
    cache = OCRCache(tmp_path / "c.json")
    region = Region("r1", "d", "p", RegionType.TEXT, BBox(0, 0, 9, 9))
    engine = _FakeEngine({"img": ("cached text value here", 0.8)})
    ocr_regions([region], {"r1": "img"}, engine, cache=cache)

    class _Exploding:
        def read(self, image): raise AssertionError("should have used the cache")

    region2 = Region("r1", "d", "p", RegionType.TEXT, BBox(0, 0, 9, 9))
    ocr_regions([region2], {"r1": "img"}, _Exploding(), cache=OCRCache(tmp_path / "c.json"))
    assert region2.text == "cached text value here"


def test_indexable_regions_filters_short_text(regions):
    usable = indexable_regions(regions, min_chars=20)
    assert all(r.is_textual and len(r.text) >= 20 for r in usable)


# ---------- chunking ----------

def test_region_chunking_is_one_to_one(regions, config):
    config.chunk.strategy = "region"
    chunks = build_chunks(regions, config)
    assert all(len(c.source_region_ids) == 1 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_window_chunking_tracks_all_source_regions(config):
    config.chunk.strategy = "window"
    config.chunk.window_chars = 120
    config.chunk.overlap_chars = 20
    regions = [
        Region(f"PMC1_00001#r{i:03d}", "PMC1", "00001", RegionType.TEXT,
               BBox(0, i * 100, 300, 90), text=f"Sentence number {i} " + "padding " * 12)
        for i in range(4)
    ]
    chunks = build_chunks(regions, config)
    assert len(chunks) > 1
    assert all(c.source_region_ids for c in chunks)
    known = {r.region_id for r in regions}
    assert all(set(c.source_region_ids) <= known for c in chunks)


def test_chunking_excludes_visual_regions(regions, config):
    chunks = build_chunks(regions, config)
    visual_ids = {r.region_id for r in regions if r.is_visual}
    cited = {rid for c in chunks for rid in c.source_region_ids}
    assert not (cited & visual_ids)


# ---------- corpus validation ----------

def test_valid_corpus_passes_and_summarises(regions, chunks):
    summary = validate_corpus(Corpus(regions=regions, chunks=chunks, page_count=2))
    assert summary["regions"] == len(regions)
    assert summary["documents"] == 2
    assert summary["visual_regions"] == 2


def test_validation_rejects_dangling_chunk_citation(regions):
    bad = [Chunk("c1", "d", "p", "text here", ["does_not_exist"])]
    with pytest.raises(CorpusValidationError, match="unknown regions"):
        validate_corpus(Corpus(regions=regions, chunks=bad))


def test_validation_rejects_duplicate_ids(regions, chunks):
    with pytest.raises(CorpusValidationError, match="duplicate"):
        validate_corpus(Corpus(regions=regions + [regions[0]], chunks=chunks))


def test_validation_rejects_empty_corpus():
    with pytest.raises(CorpusValidationError, match="no regions"):
        validate_corpus(Corpus())


def test_corpus_roundtrip_preserves_everything(regions, chunks, tmp_path):
    original = Corpus(regions=regions, chunks=chunks, ocr_stats={"empty_rate": 0.1}, page_count=2)
    original.save(tmp_path / "corpus.json")
    loaded = Corpus.load(tmp_path / "corpus.json")
    assert len(loaded.regions) == len(regions)
    assert loaded.regions[0].region_type is regions[0].region_type
    assert loaded.regions[0].bbox == regions[0].bbox
    assert loaded.ocr_stats == {"empty_rate": 0.1}
    validate_corpus(loaded)


# ---------- annotation payload encodings ----------

@pytest.mark.parametrize("encode", [
    lambda d: d,                                   # already-decoded dict
    lambda d: __import__("json").dumps(d),         # raw JSON string
    lambda d: __import__("json").dumps(d).encode(),# raw JSON bytes
])
def test_accepts_decoded_and_raw_json_payloads(encode):
    """Whether the WebDataset loader decodes .json depends on its version;
    ingestion must not die on the first sample because of that."""
    payload = {"annotations": [{"bbox": [1, 1, 50, 50], "category_id": 1}]}
    assert len(parse_page_annotation(encode(payload), "PMC1_0", 200, 200)) == 1


def test_malformed_json_text_raises_clearly():
    with pytest.raises(AnnotationParseError, match="not valid JSON"):
        parse_page_annotation("{not json at all", "PMC1_0", 100, 100)


# ---------- OCR crop upscaling (measured optimisation) ----------

class _FakeImage:
    """Stands in for a PIL image so ingestion is testable without Pillow."""
    def __init__(self, w, h): self.width, self.height = w, h
    def resize(self, size, resample=None):
        return _FakeImage(size[0], size[1])


def test_small_crops_are_upscaled_before_ocr():
    """Resolution is the dominant OCR failure mode; small crops get upscaled."""
    engine = TesseractEngine(min_crop_height=200)
    assert engine._prepare(_FakeImage(400, 50)).height == 200


def test_upscaling_preserves_aspect_ratio():
    out = TesseractEngine(min_crop_height=200)._prepare(_FakeImage(400, 50))
    assert out.width == 1600  # 400 * (200/50)


def test_large_crops_are_not_upscaled():
    """Avoid wasting compute: the measured gain saturates at 200px."""
    original = _FakeImage(800, 400)
    assert TesseractEngine(min_crop_height=200)._prepare(original) is original


def test_upscaling_can_be_disabled():
    original = _FakeImage(400, 50)
    assert TesseractEngine(min_crop_height=0)._prepare(original) is original


def test_upscaling_works_without_pillow_installed(monkeypatch):
    """Regression guard: CI installs a minimal dependency set with no Pillow.

    An unconditional `from PIL import Image` inside the upscaling path broke
    CI once while passing locally, because Pillow happened to be installed on
    the dev machine.
    """
    import builtins

    real_import = builtins.__import__

    def no_pillow(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("Pillow is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pillow)

    engine = TesseractEngine(min_crop_height=200)
    assert engine._resample_filter() is None
    assert engine._prepare(_FakeImage(400, 50)).height == 200
