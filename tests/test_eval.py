"""Evaluation: metrics, QA construction safeguards, and the comparison runner."""
import json

import pytest

from sunrai_rag.config import Config
from sunrai_rag.eval.build_qa import (
    build_qa_set,
    lexical_overlap,
    load_qa_set,
    qa_set_stats,
    save_qa_set,
)
from sunrai_rag.eval.metrics import (
    aggregate_judge_scores,
    answer_has_support,
    citation_validity,
    dcg_at_k,
    evaluate_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from sunrai_rag.eval.run_eval import (
    StubResultsError,
    _to_region_ids,
    compute_deltas,
    evaluate_system,
    format_summary,
    run_comparison,
    write_results,
)
from sunrai_rag.rag.llm import StubBackend
from sunrai_rag.schemas import Chunk, Provenance, QAItem, QueryType, ScoredItem

# ---------- metric correctness (hand-computed) ----------

def test_recall_at_k_hand_computed():
    assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert recall_at_k(["a", "b"], ["a", "z"], 2) == pytest.approx(0.5)


def test_precision_at_k_hand_computed():
    assert precision_at_k(["a", "b", "c", "d"], ["a", "c"], 4) == pytest.approx(0.5)


def test_reciprocal_rank_is_inverse_of_first_hit_position():
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_ndcg_hand_computed():
    # single gold at rank 3: DCG = 1/log2(4) = 0.5, IDCG = 1 -> 0.5
    assert ndcg_at_k(["a", "b", "c"], ["c"], 3) == pytest.approx(0.5)
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)


def test_ndcg_rewards_higher_placement():
    assert ndcg_at_k(["g", "x", "y"], ["g"], 3) > ndcg_at_k(["x", "y", "g"], ["g"], 3)


def test_dcg_accumulates_multiple_hits():
    assert dcg_at_k(["a", "b"], ["a", "b"], 2) > dcg_at_k(["a", "z"], ["a", "b"], 2)


def test_metrics_handle_empty_gold_and_empty_retrieval():
    assert recall_at_k(["a"], [], 1) == 0.0
    assert precision_at_k([], ["a"], 1) == 0.0
    assert ndcg_at_k([], ["a"], 1) == 0.0


@pytest.mark.parametrize("fn", [recall_at_k, precision_at_k, ndcg_at_k])
def test_metrics_reject_non_positive_k(fn):
    with pytest.raises(ValueError):
        fn(["a"], ["a"], 0)


def test_aggregate_retrieval_macro_averages():
    scores = evaluate_retrieval([["a"], ["z"]], [["a"], ["a"]], k_values=[1])
    assert scores.recall[1] == pytest.approx(0.5)
    assert scores.n_questions == 2


def test_aggregate_handles_empty_question_set():
    assert evaluate_retrieval([], [], k_values=[1, 5]).n_questions == 0


def test_aggregate_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        evaluate_retrieval([["a"]], [["a"], ["b"]], k_values=[1])


# ---------- answer-quality metrics ----------

def test_citation_validity_detects_invented_ids():
    assert citation_validity(["real", "fake"], {"real"}) == pytest.approx(0.5)
    assert citation_validity([], {"real"}) == 0.0


def test_refusal_counts_as_supported():
    """Correctly declining when evidence is absent is not a failure."""
    assert answer_has_support("Not stated in the provided context.", True)
    assert not answer_has_support("The accuracy was 92%.", True)
    assert answer_has_support("The accuracy was 92%.", False)


def test_judge_aggregation_reports_uncertainty():
    out = aggregate_judge_scores([0.8, 0.9, 0.7, 1.0, 0.6])
    assert out["ci95_low"] < out["mean"] < out["ci95_high"]
    assert out["n"] == 5


def test_judge_aggregation_handles_edge_cases():
    assert aggregate_judge_scores([])["n"] == 0
    single = aggregate_judge_scores([0.5])
    assert single["mean"] == single["ci95_low"] == 0.5


# ---------- QA construction safeguards ----------

def test_lexical_overlap_detects_circular_question():
    source = "The random forest classifier achieved high accuracy on the cohort"
    assert lexical_overlap("random forest classifier accuracy cohort", source) == 1.0


def test_lexical_overlap_low_for_genuine_paraphrase():
    source = "The random forest classifier achieved high accuracy on the cohort"
    assert lexical_overlap("Which tree ensemble performed best?", source) < 0.5


def test_lexical_overlap_handles_empty_question():
    assert lexical_overlap("", "some source text") == 0.0


def test_qa_builder_rejects_high_overlap_questions():
    """A verbatim-copy question must be filtered out as circular.

    Constructed so only the text branch can fire: one chunk (so multi-hop,
    which needs two per document, cannot trigger) and no visual regions.
    """
    long_text = ("The random forest classifier was trained on the cardiac cohort "
                 "and evaluated against a convolutional baseline using accuracy "
                 "and area under the curve across five stratified folds of data. ") * 2
    chunk = Chunk("c1", "PMC1", "00001", long_text, ["PMC1_00001#r000"])
    stub = StubBackend(responses={
        "Write ONE question": json.dumps({"question": long_text[:150], "answer": "x"})})
    assert build_qa_set([chunk], [], stub, n_per_type=3, max_overlap=0.6) == []


def test_qa_builder_keeps_paraphrase_on_the_same_source():
    """Control for the test above: same source, paraphrased question survives."""
    long_text = ("The random forest classifier was trained on the cardiac cohort "
                 "and evaluated against a convolutional baseline using accuracy "
                 "and area under the curve across five stratified folds of data. ") * 2
    chunk = Chunk("c1", "PMC1", "00001", long_text, ["PMC1_00001#r000"])
    stub = StubBackend(responses={
        "Write ONE question": '{"question": "Which tree ensemble performed best?",'
                              ' "answer": "x"}'})
    items = build_qa_set([chunk], [], stub, n_per_type=3, max_overlap=0.6)
    assert len(items) == 1
    assert items[0].gold_region_ids == ["PMC1_00001#r000"]


def test_qa_builder_accepts_paraphrased_questions(chunks, regions):
    stub = StubBackend(responses={
        "Write ONE question": '{"question": "Which ensemble approach worked best overall?",'
                              ' "answer": "unclear"}'})
    items = build_qa_set(chunks, [r for r in regions if r.is_visual],
                         stub, n_per_type=2, max_overlap=0.6)
    assert items and all(i.gold_region_ids for i in items)


def test_qa_builder_survives_unparseable_llm_output(chunks, regions):
    stub = StubBackend(responses={"Write ONE question": "sorry, no"})
    assert build_qa_set(chunks, regions, stub, n_per_type=2) == []


def test_qa_visual_items_point_at_visual_regions(chunks, regions):
    stub = StubBackend(responses={
        "Write ONE question": '{"question": "Which approach scored highest overall?",'
                              ' "answer": "x"}'})
    items = build_qa_set(chunks, [r for r in regions if r.is_visual],
                         stub, n_per_type=5, max_overlap=0.9)
    visual = [i for i in items if i.query_type is QueryType.VISUAL_REQUIRING]
    visual_ids = {r.region_id for r in regions if r.is_visual}
    assert visual and all(set(i.gold_region_ids) <= visual_ids for i in visual)


def test_qa_multihop_items_cite_multiple_regions(chunks, regions):
    stub = StubBackend(responses={
        "Write ONE question": '{"question": "Which approach scored highest overall?",'
                              ' "answer": "x"}'})
    items = build_qa_set(chunks, regions, stub, n_per_type=5, max_overlap=0.9)
    multihop = [i for i in items if i.query_type is QueryType.MULTI_HOP]
    assert multihop and all(len(i.gold_region_ids) >= 2 for i in multihop)


def test_qa_builder_is_reproducible(chunks, regions):
    stub = lambda: StubBackend(responses={
        "Write ONE question": '{"question": "Which ensemble approach worked best?",'
                              ' "answer": "x"}'})
    a = build_qa_set(chunks, regions, stub(), n_per_type=3, seed=7)
    b = build_qa_set(chunks, regions, stub(), n_per_type=3, seed=7)
    assert [i.question for i in a] == [i.question for i in b]
    assert [i.gold_region_ids for i in a] == [i.gold_region_ids for i in b]


def test_qa_set_roundtrip(tmp_path):
    items = [QAItem("q1", "A question here?", QueryType.VISUAL_REQUIRING, ["r1"], "ans")]
    save_qa_set(items, tmp_path / "qa.json")
    assert load_qa_set(tmp_path / "qa.json") == items


def test_qa_stats_summarise_segments():
    items = [
        QAItem("q1", "x?", QueryType.TEXT_ANSWERABLE, ["r1"]),
        QAItem("q2", "y?", QueryType.VISUAL_REQUIRING, ["r2"]),
        QAItem("q3", "z?", QueryType.MULTI_HOP, ["r3", "r4"]),
    ]
    stats = qa_set_stats(items)
    assert stats["total"] == 3 and len(stats["by_type"]) == 3
    assert stats["mean_gold_regions"] == 1.33  # rounded to 2dp for reporting


# ---------- id-space mapping ----------

def test_chunk_ids_map_to_source_regions():
    assert _to_region_ids(["c1"], {"c1": ["r1", "r2"]}) == ["r1", "r2"]


def test_region_ids_pass_through_unchanged():
    assert _to_region_ids(["r9"], {}) == ["r9"]


def test_mapping_preserves_rank_order_and_dedupes():
    mapping = {"c1": ["r1"], "c2": ["r1", "r2"]}
    assert _to_region_ids(["c1", "c2"], mapping) == ["r1", "r2"]


# ---------- comparison runner ----------

class _FakeSystem:
    def __init__(self, name, returns): self.name, self._returns = name, returns
    def retrieve(self, query):
        return Provenance(text_chunks=[ScoredItem(i, 1.0, "text", n + 1)
                                       for n, i in enumerate(self._returns)])


def test_evaluate_system_scores_retrieval_only_systems():
    qa = [QAItem("q1", "question?", QueryType.TEXT_ANSWERABLE, ["r1"])]
    result = evaluate_system(_FakeSystem("s", ["r1"]), qa, {}, {"r1"},
                             k_values=[1], generate_answers=False)
    assert result.overall.recall[1] == 1.0


def test_evaluate_system_segments_by_query_type():
    qa = [
        QAItem("q1", "a?", QueryType.TEXT_ANSWERABLE, ["r1"]),
        QAItem("q2", "b?", QueryType.VISUAL_REQUIRING, ["r9"]),
    ]
    result = evaluate_system(_FakeSystem("s", ["r1"]), qa, {}, {"r1"},
                             k_values=[1], generate_answers=False)
    assert result.by_segment["text_answerable"].recall[1] == 1.0
    assert result.by_segment["visual_requiring"].recall[1] == 0.0


def test_deltas_are_enhanced_minus_baseline():
    qa = [QAItem("q1", "a?", QueryType.VISUAL_REQUIRING, ["r_gold"])]
    weak = evaluate_system(_FakeSystem("b", ["r_wrong"]), qa, {}, set(),
                           k_values=[1], generate_answers=False)
    strong = evaluate_system(_FakeSystem("e", ["r_gold"]), qa, {}, set(),
                             k_values=[1], generate_answers=False)
    deltas = compute_deltas(weak, strong, primary_k=1)
    assert deltas["overall"]["recall@1"] == 1.0
    assert deltas["by_segment"]["visual_requiring"]["recall@1"] == 1.0


def test_deltas_can_be_negative_when_enhanced_loses():
    """An honest harness must be able to report the enhanced system losing."""
    qa = [QAItem("q1", "a?", QueryType.TEXT_ANSWERABLE, ["r_gold"])]
    strong = evaluate_system(_FakeSystem("b", ["r_gold"]), qa, {}, set(),
                             k_values=[1], generate_answers=False)
    weak = evaluate_system(_FakeSystem("e", ["r_wrong"]), qa, {}, set(),
                           k_values=[1], generate_answers=False)
    assert compute_deltas(strong, weak, primary_k=1)["overall"]["recall@1"] == -1.0


def test_run_comparison_refuses_stub_results_by_default():
    cfg = Config()
    cfg.llm.backend = "stub"
    qa = [QAItem("q1", "a?", QueryType.TEXT_ANSWERABLE, ["r1"])]
    with pytest.raises(StubResultsError, match="Refusing"):
        run_comparison({"baseline": _FakeSystem("b", ["r1"])}, qa, {}, {"r1"}, cfg)


def test_run_comparison_allows_explicit_stub_smoke_test():
    cfg = Config()
    cfg.llm.backend = "stub"
    cfg.eval.k_values = [1]
    cfg.eval.primary_k = 1
    qa = [QAItem("q1", "a?", QueryType.TEXT_ANSWERABLE, ["r1"])]
    payload = run_comparison({"baseline": _FakeSystem("b", ["r1"]),
                              "enhanced": _FakeSystem("e", ["r1"])},
                             qa, {}, {"r1"}, cfg, allow_stub=True)
    assert "deltas" in payload and payload["n_questions"] == 1


def test_results_written_as_json_and_csv(tmp_path):
    cfg = Config()
    cfg.llm.backend = "stub"
    cfg.eval.k_values = [1]
    cfg.eval.primary_k = 1
    qa = [QAItem("q1", "a?", QueryType.TEXT_ANSWERABLE, ["r1"])]
    payload = run_comparison({"baseline": _FakeSystem("b", ["r1"])},
                             qa, {}, {"r1"}, cfg, allow_stub=True)
    csv_path = write_results(payload, tmp_path, primary_k=1)
    assert csv_path.exists() and (tmp_path / "results.json").exists()
    header = csv_path.read_text().splitlines()[0]
    assert "recall@1" in header and "system" in header


def test_results_payload_records_provenance_of_the_run():
    cfg = Config()
    cfg.llm.backend = "stub"
    cfg.eval.k_values = [1]
    cfg.eval.primary_k = 1
    qa = [QAItem("q1", "a?", QueryType.TEXT_ANSWERABLE, ["r1"])]
    payload = run_comparison({"baseline": _FakeSystem("b", ["r1"])},
                             qa, {}, {"r1"}, cfg, allow_stub=True)
    assert payload["config"]["seed"] == cfg.seed
    assert payload["config"]["text_embed_model"] == cfg.models.text_embed_model


def test_summary_formats_without_error():
    cfg = Config()
    cfg.llm.backend = "stub"
    cfg.eval.k_values = [1]
    cfg.eval.primary_k = 1
    qa = [QAItem("q1", "a?", QueryType.TEXT_ANSWERABLE, ["r1"])]
    payload = run_comparison({"baseline": _FakeSystem("b", ["r1"]),
                              "enhanced": _FakeSystem("e", ["r1"])},
                             qa, {}, {"r1"}, cfg, allow_stub=True)
    text = format_summary(payload, primary_k=1)
    assert "RESULTS" in text and "DELTAS" in text
