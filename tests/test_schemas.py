"""Schema invariants: bbox geometry, region typing, provenance integrity."""
from sunrai_rag.schemas import (
    CATEGORY_ID_TO_TYPE,
    TEXTUAL_TYPES,
    VISUAL_TYPES,
    BBox,
    Provenance,
    QAItem,
    QueryType,
    RegionType,
    ScoredItem,
)


def test_bbox_area_and_conversion():
    b = BBox(10, 20, 30, 40)
    assert b.area == 1200
    assert b.to_xyxy() == (10, 20, 40, 60)


def test_bbox_clipping_keeps_box_inside_page():
    b = BBox(-10, -10, 100, 100).clipped(50, 50)
    assert b.x == 0 and b.y == 0
    assert b.x + b.w <= 50 and b.y + b.h <= 50


def test_bbox_clipped_fully_outside_is_degenerate():
    assert BBox(200, 200, 50, 50).clipped(100, 100).is_degenerate()


def test_degenerate_detection():
    assert BBox(0, 0, 1, 50).is_degenerate(min_side=2.0)
    assert not BBox(0, 0, 50, 50).is_degenerate(min_side=2.0)


def test_publaynet_category_mapping_is_complete():
    assert set(CATEGORY_ID_TO_TYPE) == {1, 2, 3, 4, 5}
    assert CATEGORY_ID_TO_TYPE[1] is RegionType.TEXT
    assert CATEGORY_ID_TO_TYPE[5] is RegionType.FIGURE


def test_textual_and_visual_types_partition_all_types():
    assert TEXTUAL_TYPES.isdisjoint(VISUAL_TYPES)
    assert TEXTUAL_TYPES | VISUAL_TYPES == set(RegionType)


def test_region_modality_flags(regions):
    text_region = regions[1]
    figure_region = regions[2]
    assert text_region.is_textual and not text_region.is_visual
    assert figure_region.is_visual and not figure_region.is_textual


def test_provenance_empty_and_citation_collection():
    assert Provenance().is_empty()
    p = Provenance(
        text_chunks=[ScoredItem("c1", 0.9)],
        visual_regions=[ScoredItem("r1", 0.8, "image")],
    )
    assert not p.is_empty()
    assert set(p.all_cited_ids()) == {"c1", "r1"}


def test_qa_item_roundtrip():
    item = QAItem("q1", "What accuracy?", QueryType.VISUAL_REQUIRING, ["r1"], "92.4%")
    restored = QAItem.from_dict(item.to_dict())
    assert restored == item
    assert restored.query_type is QueryType.VISUAL_REQUIRING
