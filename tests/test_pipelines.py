"""End-to-end pipeline behaviour, including the central hypothesis test."""
import numpy as np
import pytest

from sunrai_rag.index.vector_store import VectorStore
from sunrai_rag.kg.extract import RuleExtractor, extract_corpus
from sunrai_rag.kg.graph import build_kg
from sunrai_rag.rag.llm import ResponseCache, StubBackend
from sunrai_rag.rag.pipelines import (
    BaselineRAG,
    EnhancedRAG,
    RandomRetriever,
    extract_citations,
)
from sunrai_rag.schemas import Answer, Provenance, ScoredItem


@pytest.fixture
def llm():
    return StubBackend()


@pytest.fixture
def baseline(chunks, text_store, embedder, llm):
    return BaselineRAG(chunks, text_store, embedder, llm, top_k=3)


@pytest.fixture
def image_store(regions, embedder):
    """CLIP-style store over visual regions, keyed by region id."""
    visual = [r for r in regions if r.is_visual]
    embeddings = embedder.embed_texts([r.text or "" for r in visual])
    return VectorStore([r.region_id for r in visual], embeddings, modality="image")


@pytest.fixture
def kg(regions):
    return build_kg(extract_corpus(regions, RuleExtractor()))


@pytest.fixture
def enhanced(chunks, regions, text_store, embedder, llm, image_store, kg):
    return EnhancedRAG(
        chunks=chunks, regions=regions, text_store=text_store,
        text_embedder=embedder, llm=llm, image_store=image_store,
        image_embedder=embedder, kg=kg, top_k=3, candidate_k=10, graph_hops=2,
    )


# ---------- baseline ----------

def test_baseline_returns_answer_with_provenance(baseline):
    ans = baseline.answer("What model was used?")
    assert isinstance(ans, Answer)
    assert ans.answer_text and not ans.provenance.is_empty()
    assert ans.system_name == "baseline_text_only"
    assert ans.latency_s is not None and ans.latency_s >= 0


def test_baseline_respects_top_k(baseline):
    assert len(baseline.retrieve("random forest").text_chunks) == 3


def test_baseline_never_retrieves_visual_regions(baseline, regions):
    """Structural claim: the baseline's index contains chunks only."""
    got = {c.item_id for c in baseline.retrieve("figure showing accuracy").text_chunks}
    assert all("#c0" in g for g in got)


def test_baseline_provenance_ids_all_exist(baseline, chunks):
    known = {c.chunk_id for c in chunks}
    assert set(baseline.retrieve("accuracy").all_cited_ids()) <= known


def test_baseline_is_deterministic(baseline):
    a = [c.item_id for c in baseline.retrieve("neural network").text_chunks]
    b = [c.item_id for c in baseline.retrieve("neural network").text_chunks]
    assert a == b


# ---------- enhanced ----------

def test_enhanced_returns_visual_evidence(enhanced):
    prov = enhanced.retrieve("What does the figure show about accuracy?")
    assert prov.visual_regions, "enhanced system must surface visual evidence"


def test_enhanced_visual_ids_are_regions_not_chunks(enhanced, regions):
    visual_ids = {r.region_id for r in regions if r.is_visual}
    prov = enhanced.retrieve("figure accuracy")
    assert {v.item_id for v in prov.visual_regions} <= visual_ids


def test_enhanced_records_graph_entities_when_query_names_one(enhanced):
    prov = enhanced.retrieve("How well did the random forest perform?")
    assert prov.graph_entities, "query naming a graph entity should link it"


def test_enhanced_records_traceable_graph_paths(enhanced):
    prov = enhanced.retrieve("How well did the random forest perform?")
    assert prov.graph_paths, "graph contribution must be explainable"


def test_enhanced_degrades_gracefully_without_graph_match(enhanced):
    """A query naming nothing in the graph must still return text results."""
    prov = enhanced.retrieve("entirely unrelated question about weather")
    assert prov.graph_entities == []
    assert prov.text_chunks


def test_enhanced_reports_its_retrieval_strategy(enhanced):
    assert "clip" in enhanced.retrieve("accuracy").retrieval_strategy.lower()


def test_enhanced_works_with_components_disabled(chunks, regions, text_store, embedder, llm):
    """Ablation path: no image store, no KG -> behaves like the baseline."""
    ablated = EnhancedRAG(chunks, regions, text_store, embedder, llm,
                          image_store=None, image_embedder=None, kg=None, top_k=3)
    prov = ablated.retrieve("random forest")
    assert prov.text_chunks and not prov.visual_regions


def test_enhanced_provenance_ids_all_exist(enhanced, chunks, regions):
    known = {c.chunk_id for c in chunks} | {r.region_id for r in regions}
    assert set(enhanced.retrieve("accuracy figure").all_cited_ids()) <= known


def test_enhanced_is_deterministic(enhanced):
    a = [c.item_id for c in enhanced.retrieve("random forest accuracy").text_chunks]
    b = [c.item_id for c in enhanced.retrieve("random forest accuracy").text_chunks]
    assert a == b


# ---------- the central hypothesis ----------

def test_enhanced_reaches_visual_only_evidence_that_baseline_cannot(baseline, enhanced):
    """The project's core claim, as an executable assertion.

    The accuracy value (92.4%) exists ONLY in the figure region. The
    text-only baseline cannot return a visual region at all; the enhanced
    system must surface it.
    """
    query = "What accuracy did the random forest achieve?"
    figure_id = "PMC001_00001#r002"

    baseline_evidence = set(baseline.retrieve(query).all_cited_ids())
    enhanced_evidence = set(enhanced.retrieve(query).all_cited_ids())

    assert figure_id not in baseline_evidence, "baseline should not reach the figure"
    assert figure_id in enhanced_evidence, "enhanced must reach the figure"


def test_both_systems_share_identical_generation_path(baseline, enhanced, llm):
    """Fairness check: any difference must come from evidence, not prompting."""
    baseline.answer("What model was used?")
    enhanced.answer("What model was used?")
    assert len(llm.calls) == 2
    template = lambda p: p.split("Question:")[0].split("[")[0]
    assert "Answer the question using ONLY the evidence below" in llm.calls[0]
    assert "Answer the question using ONLY the evidence below" in llm.calls[1]


# ---------- generation contract ----------

def test_generator_refuses_when_no_evidence(chunks, embedder, llm):
    empty = BaselineRAG([], VectorStore([], np.zeros((0, 64))), embedder, llm, top_k=3)
    assert "Not stated" in empty.answer("anything at all?").answer_text


def test_citation_extraction_parses_bracketed_ids():
    assert extract_citations("The answer is X [chunk_1] and [chunk_2].") == \
           ["chunk_1", "chunk_2"]


def test_citation_extraction_handles_no_citations():
    assert extract_citations("No citations here.") == []


def test_generator_receives_visual_evidence_in_prompt(enhanced, llm):
    enhanced.answer("What does the figure show?")
    assert any("(figure on page" in c or "(table on page" in c for c in llm.calls)


def test_generator_receives_graph_evidence_in_prompt(enhanced, llm):
    enhanced.answer("How did the random forest perform?")
    assert any("knowledge_graph" in c for c in llm.calls)


# ---------- random floor ----------

def test_random_floor_returns_k_results(chunks):
    floor = RandomRetriever([c.chunk_id for c in chunks], top_k=3)
    assert len(floor.retrieve("any query").text_chunks) == 3


def test_random_floor_is_reproducible_per_query(chunks):
    ids = [c.chunk_id for c in chunks]
    a = RandomRetriever(ids, seed=7, top_k=3).retrieve("q")
    b = RandomRetriever(ids, seed=7, top_k=3).retrieve("q")
    assert [x.item_id for x in a.text_chunks] == [x.item_id for x in b.text_chunks]


def test_random_floor_varies_across_queries(chunks):
    floor = RandomRetriever([c.chunk_id for c in chunks], seed=7, top_k=2)
    results = {tuple(x.item_id for x in floor.retrieve(f"query {i}").text_chunks)
               for i in range(8)}
    assert len(results) > 1


def test_random_floor_never_duplicates(chunks):
    picks = [x.item_id for x in
             RandomRetriever([c.chunk_id for c in chunks], top_k=4).retrieve("q").text_chunks]
    assert len(picks) == len(set(picks))


# ---------- LLM cache ----------

def test_cache_returns_stored_response(tmp_path):
    cache = ResponseCache(tmp_path / "c.json")
    key = ResponseCache.make_key("anthropic", "m", "prompt", None)
    cache.put(key, "cached")
    assert cache.get(key) == "cached"


def test_cache_key_depends_on_model_and_system(tmp_path):
    k1 = ResponseCache.make_key("anthropic", "m1", "p", None)
    k2 = ResponseCache.make_key("anthropic", "m2", "p", None)
    k3 = ResponseCache.make_key("anthropic", "m1", "p", "system")
    assert len({k1, k2, k3}) == 3


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.json"
    key = ResponseCache.make_key("b", "m", "p", None)
    c1 = ResponseCache(path); c1.put(key, "v"); c1.flush()
    assert ResponseCache(path).get(key) == "v"


def test_cache_tracks_hit_rate(tmp_path):
    cache = ResponseCache(tmp_path / "c.json")
    key = ResponseCache.make_key("b", "m", "p", None)
    cache.get(key); cache.put(key, "v"); cache.get(key)
    assert cache.stats() == {"hits": 1, "misses": 1, "hit_rate": 0.5, "entries": 1}


def test_disabled_cache_stores_nothing(tmp_path):
    cache = ResponseCache(tmp_path / "c.json", enabled=False)
    key = ResponseCache.make_key("b", "m", "p", None)
    cache.put(key, "v")
    assert cache.get(key) is None


# ---------- regression: modality starvation in ranked evidence ----------

def test_ranked_evidence_interleaves_modalities():
    """Regression guard for a measurement bug found during development.

    Concatenating text-then-visual pushed every visual hit past position k,
    so Recall@k could never credit visual retrieval and the enhanced system
    appeared blind to figures even when its visual retrieval was perfect.
    """
    prov = Provenance(
        text_chunks=[ScoredItem(f"chunk{i}", 0.9, "text", i + 1) for i in range(5)],
        visual_regions=[ScoredItem("GOLD_FIGURE", 0.95, "image", 1)],
    )
    ranked = prov.ranked_evidence_ids()
    assert ranked.index("GOLD_FIGURE") < 5, "visual rank-1 must reach the top-5"
    assert ranked[0] == "chunk0" and ranked[1] == "GOLD_FIGURE"


def test_ranked_evidence_preserves_within_modality_order():
    prov = Provenance(
        text_chunks=[ScoredItem("t1", 0.9, "text", 1), ScoredItem("t2", 0.8, "text", 2)],
        visual_regions=[ScoredItem("v1", 0.9, "image", 1), ScoredItem("v2", 0.8, "image", 2)],
    )
    ranked = prov.ranked_evidence_ids()
    assert ranked.index("t1") < ranked.index("t2")
    assert ranked.index("v1") < ranked.index("v2")


def test_ranked_evidence_handles_single_modality():
    text_only = Provenance(text_chunks=[ScoredItem("t1", 0.9, "text", 1)])
    assert text_only.ranked_evidence_ids() == ["t1"]
    visual_only = Provenance(visual_regions=[ScoredItem("v1", 0.9, "image", 1)])
    assert visual_only.ranked_evidence_ids() == ["v1"]
    assert Provenance().ranked_evidence_ids() == []


def test_ranked_evidence_deduplicates():
    prov = Provenance(
        text_chunks=[ScoredItem("same", 0.9, "text", 1)],
        visual_regions=[ScoredItem("same", 0.8, "image", 1)],
    )
    assert prov.ranked_evidence_ids() == ["same"]


def test_enhanced_visual_hit_is_scored_within_top_k(enhanced):
    """The end-to-end consequence: a figure the system retrieves must be
    reachable by a top-k metric."""
    prov = enhanced.retrieve("What accuracy did the random forest achieve?")
    ranked = prov.ranked_evidence_ids()
    assert "PMC001_00001#r002" in ranked[:5]


# ---------- BM25 sanity floor ----------

def test_bm25_floor_conforms_to_retriever_interface(chunks):
    from sunrai_rag.index.bm25 import BM25Index
    from sunrai_rag.rag.pipelines import BM25Retriever

    floor = BM25Retriever(
        BM25Index([c.chunk_id for c in chunks], [c.text for c in chunks]), top_k=3
    )
    prov = floor.retrieve("random forest classifier")
    assert len(prov.text_chunks) == 3
    assert prov.retrieval_strategy == "bm25_lexical"
    assert floor.name == "bm25_lexical_floor"


def test_bm25_floor_finds_exact_lexical_match(chunks):
    from sunrai_rag.index.bm25 import BM25Index
    from sunrai_rag.rag.pipelines import BM25Retriever

    floor = BM25Retriever(
        BM25Index([c.chunk_id for c in chunks], [c.text for c in chunks]), top_k=1
    )
    top = floor.retrieve("convolutional neural network cardiac cohort").text_chunks[0]
    assert "PMC001" in top.item_id
