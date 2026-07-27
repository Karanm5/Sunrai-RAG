"""Knowledge graph: extraction validity, linking, expansion, persistence."""
import pytest

from sunrai_rag.kg.extract import (
    Entity,
    Extraction,
    LLMExtractor,
    Relation,
    RuleExtractor,
    extract_corpus,
)
from sunrai_rag.kg.graph import KnowledgeGraph, build_kg
from sunrai_rag.rag.llm import StubBackend
from sunrai_rag.schemas import BBox, Region, RegionType


def _region(text, rid="r1"):
    return Region(rid, "d1", "p1", RegionType.TEXT, BBox(0, 0, 100, 100), text=text)


# ---------- rule extraction ----------

def test_extracts_methods_and_metrics():
    r = _region("We used a random forest and report accuracy and AUC for the cohort.")
    result = RuleExtractor().extract(r)
    types = {e.entity_type for e in result.entities}
    assert "method" in types and "metric" in types


def test_extracts_metric_value_as_finding():
    r = _region("The model achieved an accuracy of 92.4% on the held-out test set.")
    findings = [e for e in RuleExtractor().extract(r).entities if e.entity_type == "finding"]
    assert findings and "92.4" in findings[0].name


def test_extracts_figure_references():
    r = _region("Results are summarised in Figure 2 and Table 1 of this paper section.")
    names = {e.name.lower() for e in RuleExtractor().extract(r).entities}
    assert any("figure 2" in n for n in names)


def test_links_method_to_metric_in_same_region():
    r = _region("A random forest was evaluated; we report accuracy for each fold here.")
    rels = RuleExtractor().extract(r).relations
    assert any(x.relation_type == "evaluated_by" for x in rels)


def test_short_text_yields_nothing():
    assert RuleExtractor().extract(_region("Too short.")).entities == []


def test_extraction_is_deterministic():
    r = _region("Random forest and CNN evaluated by accuracy and AUC in Figure 3 here.")
    a = RuleExtractor().extract(r)
    b = RuleExtractor().extract(r)
    assert [e.node_id for e in a.entities] == [e.node_id for e in b.entities]


def test_node_id_normalises_case_for_merging():
    e1 = Entity("Random Forest", "method", "r1")
    e2 = Entity("random forest", "method", "r2")
    assert e1.node_id == e2.node_id


def test_corpus_extraction_deduplicates_across_regions():
    regions = [_region("A random forest was used with accuracy reported here.", f"r{i}")
               for i in range(3)]
    result = extract_corpus(regions, RuleExtractor())
    assert len({e.node_id for e in result.entities}) == len(result.entities)


def test_corpus_extraction_respects_max_regions():
    regions = [_region(f"Random forest number {i} evaluated by accuracy here.", f"r{i}")
               for i in range(10)]
    result = extract_corpus(regions, RuleExtractor(), max_regions=2)
    assert {e.source_region_id for e in result.entities} <= {"r0", "r1"}


# ---------- LLM extraction ----------

def test_llm_extractor_parses_valid_json():
    stub = StubBackend(responses={"Extract a knowledge graph": (
        '{"entities": [{"name": "BERT", "type": "method"}, '
        '{"name": "F1", "type": "metric"}], '
        '"relations": [{"head": "BERT", "tail": "F1", "type": "evaluated_by"}]}')})
    result = LLMExtractor(llm=stub).extract(_region("x" * 60))
    assert len(result.entities) == 2 and len(result.relations) == 1


def test_llm_extractor_strips_markdown_fences():
    stub = StubBackend(responses={"Extract a knowledge graph":
        '```json\n{"entities": [{"name": "SVM", "type": "method"}], "relations": []}\n```'})
    assert len(LLMExtractor(llm=stub).extract(_region("x" * 60)).entities) == 1


def test_llm_extractor_drops_hallucinated_relations():
    """A relation citing an undeclared entity is the signature of a made-up edge."""
    stub = StubBackend(responses={"Extract a knowledge graph": (
        '{"entities": [{"name": "BERT", "type": "method"}], '
        '"relations": [{"head": "BERT", "tail": "GHOST", "type": "uses"}]}')})
    result = LLMExtractor(llm=stub).extract(_region("x" * 60))
    assert result.relations == []


def test_llm_extractor_rejects_invalid_entity_types():
    stub = StubBackend(responses={"Extract a knowledge graph":
        '{"entities": [{"name": "X", "type": "not_a_type"}], "relations": []}'})
    assert LLMExtractor(llm=stub).extract(_region("x" * 60)).entities == []


def test_llm_extractor_survives_unparseable_output():
    stub = StubBackend(responses={"Extract a knowledge graph": "I cannot help with that."})
    result = LLMExtractor(llm=stub).extract(_region("x" * 60))
    assert result.entities == [] and result.relations == []


# ---------- graph ----------

@pytest.fixture
def kg():
    extraction = Extraction(
        entities=[
            Entity("random forest", "method", "rA"),
            Entity("accuracy", "metric", "rA"),
            Entity("accuracy = 92.4%", "finding", "rB"),
            Entity("orphan method", "method", "rC"),
        ],
        relations=[
            Relation("method:random forest", "metric:accuracy", "evaluated_by", "rA"),
            Relation("metric:accuracy", "finding:accuracy = 92.4%", "has_value", "rB"),
        ],
    )
    return build_kg(extraction)


def test_graph_builds_expected_shape(kg):
    assert kg.n_nodes == 4 and kg.n_edges == 2
    assert kg.stats()["entity_types"]["method"] == 2


def test_edges_to_undeclared_nodes_are_dropped(kg):
    before = kg.n_edges
    kg.add_relation(Relation("method:random forest", "method:ghost", "uses", "rX"))
    assert kg.n_edges == before


def test_repeated_entity_accumulates_provenance(kg):
    kg.add_entity(Entity("random forest", "method", "rZ"))
    assert set(kg.regions_for_node("method:random forest")) == {"rA", "rZ"}


def test_entity_linking_finds_mentioned_nodes(kg):
    assert "method:random forest" in kg.link_query_entities("How did the random forest do?")


def test_entity_linking_avoids_substring_false_positives(kg):
    """'accuracy' must not match inside 'inaccuracy'."""
    assert "metric:accuracy" not in kg.link_query_entities("Discussion of inaccuracies here")


def test_entity_linking_returns_empty_for_unrelated_query(kg):
    assert kg.link_query_entities("completely unrelated weather question") == []


def test_one_hop_expansion_reaches_neighbours(kg):
    assert "metric:accuracy" in kg.expand(["method:random forest"], hops=1)


def test_two_hop_expansion_reaches_the_finding(kg):
    """The multi-hop case: method -> metric -> reported value."""
    one = kg.expand(["method:random forest"], hops=1)
    two = kg.expand(["method:random forest"], hops=2)
    assert "finding:accuracy = 92.4%" not in one
    assert "finding:accuracy = 92.4%" in two


def test_expansion_traverses_edges_in_both_directions(kg):
    assert "method:random forest" in kg.expand(["metric:accuracy"], hops=1)


def test_expansion_of_unknown_node_is_empty_not_an_error(kg):
    assert kg.expand(["method:does not exist"], hops=1) == set()


def test_isolated_node_expands_to_only_itself(kg):
    assert kg.expand(["method:orphan method"], hops=1) == {"method:orphan method"}


def test_zero_hops_returns_only_seeds(kg):
    assert kg.expand(["method:random forest"], hops=0) == {"method:random forest"}


def test_negative_hops_rejected(kg):
    with pytest.raises(ValueError):
        kg.expand(["method:random forest"], hops=-1)


def test_expansion_prefers_regions_plain_retrieval_would_miss(kg):
    """The KG's value is reaching rB from a query naming only the method."""
    regions = kg.regions_from_expansion(["method:random forest"], hops=2, max_regions=5)
    assert "rB" in regions


def test_expansion_paths_are_traceable_for_explainability(kg):
    paths = kg.expansion_paths(["method:random forest"], hops=2)
    assert paths and all(p[0] == "method:random forest" for p in paths)


def test_subgraph_description_is_prompt_ready(kg):
    text = kg.describe_subgraph(kg.expand(["method:random forest"], hops=2))
    assert "evaluated_by" in text and "-->" in text


def test_subgraph_description_handles_no_relations(kg):
    assert "no graph relations" in kg.describe_subgraph(["method:orphan method"])


def test_graph_roundtrip_preserves_structure_and_provenance(kg, tmp_path):
    kg.save(tmp_path / "kg.json")
    loaded = KnowledgeGraph.load(tmp_path / "kg.json")
    assert loaded.n_nodes == kg.n_nodes and loaded.n_edges == kg.n_edges
    assert loaded.regions_for_node("method:random forest") == \
           kg.regions_for_node("method:random forest")
    assert loaded.expand(["method:random forest"], hops=2) == \
           kg.expand(["method:random forest"], hops=2)


# ---------- batched extraction (rate-limit optimisation) ----------

def _batch_response(n):
    import json
    return json.dumps({"results": [
        {"index": i,
         "entities": [{"name": f"method{i}", "type": "method"},
                      {"name": "accuracy", "type": "metric"}],
         "relations": [{"head": f"method{i}", "tail": "accuracy",
                        "type": "evaluated_by"}]}
        for i in range(n)]})


def _regions(n):
    return [_region("x" * 80, f"r{i}") for i in range(n)]


def test_batching_reduces_api_calls():
    """One call per region burns the request budget on repeated boilerplate."""
    stub = StubBackend(responses={"Extract a knowledge graph from EACH": _batch_response(5)})
    extract_corpus(_regions(10), LLMExtractor(llm=stub, batch_size=5))
    assert len(stub.calls) == 2, f"expected 2 batched calls, got {len(stub.calls)}"


def test_batching_preserves_per_region_provenance():
    """Entities must still be attributed to the region they came from."""
    stub = StubBackend(responses={"Extract a knowledge graph from EACH": _batch_response(3)})
    result = extract_corpus(_regions(3), LLMExtractor(llm=stub, batch_size=3))
    sources = {e.source_region_id for e in result.entities}
    assert sources == {"r0", "r1", "r2"}


def test_unparseable_batch_falls_back_per_region():
    """A malformed batch costs accuracy on that batch, never the whole run."""
    stub = StubBackend(responses={"Extract a knowledge graph from EACH": "not json",
                                  "Extract a knowledge graph from this": '{"entities": [], "relations": []}'})
    extract_corpus(_regions(3), LLMExtractor(llm=stub, batch_size=3))
    assert len(stub.calls) == 4  # 1 failed batch + 3 individual retries


def test_omitted_index_is_retried_individually():
    """Models sometimes silently drop an excerpt; that region still gets tried."""
    import json
    partial = json.dumps({"results": [
        {"index": 0, "entities": [{"name": "svm", "type": "method"}], "relations": []}]})
    stub = StubBackend(responses={"Extract a knowledge graph from EACH": partial,
                                  "Extract a knowledge graph from this": '{"entities": [], "relations": []}'})
    extract_corpus(_regions(3), LLMExtractor(llm=stub, batch_size=3))
    assert len(stub.calls) == 3  # 1 batch + 2 retries for the omitted indices


def test_batch_ignores_out_of_range_indices():
    import json
    bad = json.dumps({"results": [
        {"index": 99, "entities": [{"name": "ghost", "type": "method"}], "relations": []}]})
    stub = StubBackend(responses={"Extract a knowledge graph from EACH": bad,
                                  "Extract a knowledge graph from this": '{"entities": [], "relations": []}'})
    result = extract_corpus(_regions(2), LLMExtractor(llm=stub, batch_size=2))
    assert all("ghost" not in e.name for e in result.entities)


def test_short_regions_excluded_from_batch():
    stub = StubBackend(responses={"Extract a knowledge graph from EACH": _batch_response(1)})
    regions = [_region("too short", "r0"), _region("x" * 80, "r1")]
    extract_corpus(regions, LLMExtractor(llm=stub, batch_size=5))
    assert "[0]" in stub.calls[0] and "[1]" not in stub.calls[0]
