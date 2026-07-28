"""Retrieval: vector store, BM25, and RRF fusion properties."""
import importlib.util

import numpy as np
import pytest

from sunrai_rag.index.bm25 import BM25Index, tokenize
from sunrai_rag.index.fusion import dedupe_preserving_order, reciprocal_rank_fusion
from sunrai_rag.index.vector_store import VectorStore, l2_normalise
from sunrai_rag.schemas import ScoredItem

# ---------- normalisation ----------

def test_l2_normalise_produces_unit_rows():
    normed = l2_normalise(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)


def test_l2_normalise_handles_zero_vector_without_nan():
    normed = l2_normalise(np.array([[0.0, 0.0]]))
    assert not np.isnan(normed).any()


# ---------- vector store ----------

def test_search_returns_k_results_in_descending_order(text_store):
    hits = text_store.search(text_store.embeddings[0], k=3)
    assert len(hits) == 3
    assert [h.rank for h in hits] == [1, 2, 3]
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_self_retrieval_ranks_the_source_chunk_first(text_store):
    """The core sanity check: a chunk must retrieve itself."""
    for i, chunk_id in enumerate(text_store.ids):
        assert text_store.search(text_store.embeddings[i], k=1)[0].item_id == chunk_id


def test_semantically_related_query_beats_unrelated(chunks, embedder, text_store):
    q = embedder.embed_texts(["random forest classifier cardiac cohort"])
    top = text_store.search(q, k=1)[0].item_id
    assert "PMC001" in top


def test_search_is_deterministic_across_calls(text_store):
    q = np.random.default_rng(0).normal(size=text_store.dim)
    assert [h.item_id for h in text_store.search(q, k=4)] == \
           [h.item_id for h in text_store.search(q, k=4)]


def test_k_larger_than_corpus_is_clamped(text_store):
    assert len(text_store.search(text_store.embeddings[0], k=999)) == len(text_store)


def test_invalid_k_rejected(text_store):
    with pytest.raises(ValueError):
        text_store.search(text_store.embeddings[0], k=0)


def test_dimension_mismatch_rejected(text_store):
    with pytest.raises(ValueError, match="dim"):
        text_store.search(np.ones(text_store.dim + 5), k=1)


def test_misaligned_construction_rejected():
    with pytest.raises(ValueError, match="align"):
        VectorStore(["a", "b"], np.zeros((3, 4)))


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        VectorStore(["a", "a"], np.zeros((2, 4)))


def test_empty_store_returns_no_hits():
    assert VectorStore([], np.zeros((0, 8))).search(np.ones(8), k=3) == []


def test_store_roundtrip(text_store, tmp_path):
    text_store.save(tmp_path / "idx")
    loaded = VectorStore.load(tmp_path / "idx")
    assert loaded.ids == text_store.ids
    q = text_store.embeddings[1]
    assert [h.item_id for h in loaded.search(q, k=3)] == \
           [h.item_id for h in text_store.search(q, k=3)]


# ---------- BM25 ----------

def test_bm25_ranks_exact_term_match_first():
    idx = BM25Index(["d1", "d2", "d3"], [
        "random forest classifier for cardiac outcomes",
        "convolutional neural network for imaging",
        "support vector machine for tabular data",
    ])
    assert idx.search("random forest", k=1)[0].item_id == "d1"


def test_bm25_scores_zero_when_no_terms_overlap():
    idx = BM25Index(["d1"], ["completely unrelated content"])
    assert idx.search("zzzz qqqq", k=1)[0].score == 0.0


def test_bm25_idf_never_negative_for_ubiquitous_terms():
    """A term in every document must not produce a negative contribution."""
    idx = BM25Index([f"d{i}" for i in range(5)], ["common word here"] * 5)
    assert all(s.score >= 0 for s in idx.search("common", k=5))


def test_bm25_tokenizer_drops_stopwords_and_punctuation():
    assert tokenize("The accuracy of the model, was 92.4%!") == ["accuracy", "model", "92", "4"]


def test_bm25_deterministic_tiebreak():
    idx = BM25Index(["d1", "d2"], ["same text", "same text"])
    assert [h.item_id for h in idx.search("same", k=2)] == ["d1", "d2"]


# ---------- RRF ----------

def _ids(items): return [i.item_id for i in items]


def test_rrf_is_order_invariant():
    a = [ScoredItem("x", 9.0, "text", 1), ScoredItem("y", 8.0, "text", 2)]
    b = [ScoredItem("y", 0.5, "image", 1), ScoredItem("z", 0.4, "image", 2)]
    assert _ids(reciprocal_rank_fusion([a, b])) == _ids(reciprocal_rank_fusion([b, a]))


def test_rrf_rewards_agreement_across_rankers():
    a = [ScoredItem("x", 9.0, "text", 1), ScoredItem("y", 8.0, "text", 2)]
    b = [ScoredItem("y", 0.5, "image", 1), ScoredItem("z", 0.4, "image", 2)]
    assert _ids(reciprocal_rank_fusion([a, b]))[0] == "y"


def test_rrf_is_scale_free():
    small = [ScoredItem("x", 0.001, "text", 1), ScoredItem("y", 0.0009, "text", 2)]
    huge = [ScoredItem("x", 1e9, "text", 1), ScoredItem("y", 9e8, "text", 2)]
    other = [ScoredItem("y", 1.0, "image", 1)]
    assert _ids(reciprocal_rank_fusion([small, other])) == \
           _ids(reciprocal_rank_fusion([huge, other]))


def test_rrf_rank_improvement_never_lowers_score():
    other = [ScoredItem("a", 1.0, "text", 1)]
    low = reciprocal_rank_fusion([other, [ScoredItem("t", 1.0, "image", 5)]])
    high = reciprocal_rank_fusion([other, [ScoredItem("t", 1.0, "image", 1)]])
    score = lambda res: next(i.score for i in res if i.item_id == "t")
    assert score(high) > score(low)


def test_rrf_weights_shift_the_ranking():
    a = [ScoredItem("x", 1.0, "text", 1)]
    b = [ScoredItem("y", 1.0, "image", 1)]
    assert _ids(reciprocal_rank_fusion([a, b], weights=[10.0, 1.0]))[0] == "x"
    assert _ids(reciprocal_rank_fusion([a, b], weights=[1.0, 10.0]))[0] == "y"


def test_rrf_records_contributing_modalities():
    a = [ScoredItem("x", 1.0, "text", 1)]
    b = [ScoredItem("x", 1.0, "graph", 1)]
    assert "text" in reciprocal_rank_fusion([a, b])[0].modality
    assert "graph" in reciprocal_rank_fusion([a, b])[0].modality


def test_rrf_output_ranks_are_sequential():
    a = [ScoredItem(f"d{i}", 1.0, "text", i + 1) for i in range(5)]
    assert [i.rank for i in reciprocal_rank_fusion([a])] == [1, 2, 3, 4, 5]


def test_rrf_respects_top_n():
    a = [ScoredItem(f"d{i}", 1.0, "text", i + 1) for i in range(10)]
    assert len(reciprocal_rank_fusion([a], top_n=3)) == 3


def test_rrf_rejects_invalid_inputs():
    a = [ScoredItem("x", 1.0, "text", 1)]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([a], k=0)
    with pytest.raises(ValueError, match="align"):
        reciprocal_rank_fusion([a], weights=[1.0, 2.0])


def test_rrf_handles_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []


def test_dedupe_keeps_best_ranked_occurrence():
    items = [ScoredItem("a", 0.9, "text", 1), ScoredItem("a", 0.1, "image", 2),
             ScoredItem("b", 0.5, "text", 3)]
    out = dedupe_preserving_order(items)
    assert _ids(out) == ["a", "b"] and out[0].score == 0.9


# CLIP tests need torch, which is deliberately NOT in requirements-min.txt:
# the package advertises that its logic layers run without the heavy ML stack,
# and CI enforces that by installing only the minimal set. These tests are
# skipped there rather than failing, and run in the full-dependency job.
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed (optional dependency)",
)


# ---------- CLIP output-shape compatibility ----------

class _FakeTensor:
    """Minimal stand-in for a torch tensor."""
    def __init__(self, data): self.data = data
    def cpu(self): return self
    def numpy(self):
        import numpy as _np
        return _np.array(self.data, dtype=_np.float32)


class _Wrapper:
    """Mimics transformers returning a ModelOutput instead of a tensor."""
    def __init__(self, pooler_output=None, image_embeds=None):
        self.pooler_output = pooler_output
        if image_embeds is not None:
            self.image_embeds = image_embeds


def _clip_extract(result, model, projection):
    from sunrai_rag.represent.embedders import CLIPEmbedder
    return CLIPEmbedder._as_embedding(result, model, projection)


@requires_torch
def test_clip_passes_through_a_plain_tensor(monkeypatch):
    import torch
    t = torch.zeros(2, 512)
    assert _clip_extract(t, object(), "visual_projection") is t


@requires_torch
def test_clip_unwraps_image_embeds(monkeypatch):
    import torch
    embeds = torch.zeros(2, 512)
    out = _clip_extract(_Wrapper(image_embeds=embeds), object(), "visual_projection")
    assert out is embeds


@requires_torch
def test_clip_projects_raw_pooler_output():
    """768-dim pooler_output is pre-projection and must be projected."""
    import torch

    class _Model:
        def __init__(self):
            self.visual_projection = torch.nn.Linear(768, 512, bias=False)

    out = _clip_extract(_Wrapper(pooler_output=torch.zeros(2, 768)),
                        _Model(), "visual_projection")
    assert out.shape == (2, 512)


@requires_torch
def test_clip_leaves_already_projected_output_alone():
    """512-dim pooler_output is ALREADY projected.

    Projecting it again is a shape error at best; treating the two modalities
    inconsistently is a silent correctness bug at worst. Decided by measuring
    the dimension, not by assuming.
    """
    import torch

    class _Model:
        def __init__(self):
            self.visual_projection = torch.nn.Linear(768, 512, bias=False)

    pooled = torch.zeros(7, 512)
    out = _clip_extract(_Wrapper(pooler_output=pooled), _Model(), "visual_projection")
    assert out is pooled, "already-projected features must pass through untouched"


@requires_torch
def test_clip_refuses_to_guess_on_unexpected_dimension():
    import torch

    class _Model:
        def __init__(self):
            self.visual_projection = torch.nn.Linear(768, 512, bias=False)

    with pytest.raises(RuntimeError, match="matches neither"):
        _clip_extract(_Wrapper(pooler_output=torch.zeros(2, 999)),
                      _Model(), "visual_projection")


@requires_torch
def test_clip_falls_back_to_cls_token():
    import torch

    class _Out:
        def __init__(self):
            self.last_hidden_state = torch.zeros(2, 50, 768)
            self.pooler_output = None

    class _Model:
        def __init__(self):
            self.visual_projection = torch.nn.Linear(768, 512, bias=False)

    out = _clip_extract(_Out(), _Model(), "visual_projection")
    assert out.shape == (2, 512)


@requires_torch
def test_clip_raises_on_unrecognised_output():
    with pytest.raises(RuntimeError, match="Could not extract embeddings"):
        _clip_extract(object(), object(), "visual_projection")


@requires_torch
def test_clip_embeddings_are_detached_from_autograd():
    """Regression: the projection ran outside no_grad, so the result carried
    gradient tracking and .numpy() refused to convert it.

    Only bites when the model returns UNPROJECTED features, which is why it
    surfaced on the text path while images worked.
    """
    import numpy as np
    import torch

    from sunrai_rag.represent.embedders import CLIPEmbedder

    class _Processor:
        def __call__(self, text=None, return_tensors=None, padding=None, truncation=None):
            return {"input_ids": torch.ones(len(text), 4, dtype=torch.long),
                    "attention_mask": torch.ones(len(text), 4, dtype=torch.long)}

    class _Out:
        def __init__(self, pooled): self.pooler_output = pooled

    class _Model:
        device = "cpu"
        def __init__(self):
            self.text_projection = torch.nn.Linear(768, 512, bias=False)
        def get_text_features(self, **kwargs):
            return _Out(torch.zeros(kwargs["input_ids"].shape[0], 768))

    embedder = CLIPEmbedder()
    embedder._model = _Model()
    embedder._processor = _Processor()
    embedder._ensure_model = lambda: (embedder._model, embedder._processor)

    vectors = embedder.embed_texts(["a query"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (1, 512)
