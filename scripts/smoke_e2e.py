"""Offline end-to-end smoke test.

Builds a synthetic corpus in which some answers exist ONLY in figure
regions, then runs the full pipeline -- index, KG, QA construction,
evaluation -- with no network, no models and no API key.

Purpose: prove the wiring works and the segmented comparison behaves as
designed, before spending time on the real dataset. Numbers produced here
are NOT results: the embedder is a hashing stub and the corpus is synthetic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sunrai_rag.config import Config, set_global_seeds
from sunrai_rag.eval.run_eval import format_summary, run_comparison, write_results
from sunrai_rag.index.bm25 import BM25Index
from sunrai_rag.index.vector_store import VectorStore
from sunrai_rag.ingest.loader import Corpus, build_chunks, validate_corpus
from sunrai_rag.kg.extract import RuleExtractor, extract_corpus
from sunrai_rag.kg.graph import KnowledgeGraph
from sunrai_rag.rag.llm import StubBackend
from sunrai_rag.rag.pipelines import (
    BaselineRAG,
    BM25Retriever,
    EnhancedRAG,
    RandomRetriever,
)
from sunrai_rag.represent.embedders import HashingEmbedder
from sunrai_rag.schemas import BBox, QAItem, QueryType, Region, RegionType

METHODS = ["random forest", "convolutional neural network", "support vector machine",
           "gradient boosting", "logistic regression", "transformer"]

def build_synthetic_corpus(n_docs=6):
    regions = []
    for d in range(n_docs):
        doc, method = f"PMC{d:04d}", METHODS[d % len(METHODS)]
        regions.append(Region(
            f"{doc}_00001#r000", doc, "00001", RegionType.TITLE, BBox(10,10,400,40),
            text=f"Predictive modelling study number {d} on clinical outcomes", ocr_confidence=0.95))
        regions.append(Region(
            f"{doc}_00001#r001", doc, "00001", RegionType.TEXT, BBox(10,60,400,200),
            text=(f"In this study we trained a {method} on the clinical cohort of "
                  f"patients. Model performance was assessed using accuracy and AUC. "
                  f"The complete numerical results are presented in Figure {d+1}."),
            ocr_confidence=0.92))
        regions.append(Region(
            f"{doc}_00002#r000", doc, "00002", RegionType.TEXT, BBox(10,10,400,200),
            text=(f"Data were collected prospectively across three sites over twelve "
                  f"months. Preprocessing followed standard normalisation for study {d}."),
            ocr_confidence=0.90))
        # The value exists ONLY here -- no body text mentions it.
        regions.append(Region(
            f"{doc}_00001#r002", doc, "00001", RegionType.FIGURE, BBox(10,280,400,300),
            text=f"Figure {d+1}: accuracy of {80+d*2}.{d}% achieved by the {method}",
            image_path=None))
    return regions

def main():
    cfg = Config()
    cfg.llm.backend = "stub"; cfg.eval.k_values = [1,3,5]; cfg.eval.primary_k = 5
    cfg.eval.judge_enabled = False; cfg.retrieval.top_k = 5; cfg.retrieval.graph_hops = 2
    set_global_seeds(cfg.seed)

    print("=" * 68); print("OFFLINE END-TO-END SMOKE TEST"); print("=" * 68)

    regions = build_synthetic_corpus()
    chunks = build_chunks(regions, cfg)
    corpus = Corpus(regions=regions, chunks=chunks, page_count=12)
    print("1. corpus validated:", validate_corpus(corpus))

    embedder = HashingEmbedder(dim=128)
    text_store = VectorStore([c.chunk_id for c in chunks],
                             embedder.embed_texts([c.text for c in chunks]), "text")
    visual = [r for r in regions if r.is_visual]
    image_store = VectorStore([r.region_id for r in visual],
                              embedder.embed_texts([r.text or "" for r in visual]), "image")
    print(f"2. indices built: {len(text_store)} text, {len(image_store)} visual")

    kg = KnowledgeGraph.from_extraction(
        extract_corpus([r for r in regions if r.is_textual], RuleExtractor()))
    print("3. knowledge graph:", kg.stats())

    llm = StubBackend()
    baseline = BaselineRAG(chunks, text_store, embedder, llm, top_k=cfg.retrieval.top_k)
    enhanced = EnhancedRAG(chunks, regions, text_store, embedder, llm,
                           image_store=image_store, image_embedder=embedder, kg=kg,
                           top_k=cfg.retrieval.top_k, candidate_k=20, graph_hops=2)

    # Hand-built QA: gold evidence known exactly, so metrics are interpretable.
    qa = []
    for d in range(6):
        doc, method = f"PMC{d:04d}", METHODS[d % len(METHODS)]
        qa.append(QAItem(f"t{d}", f"What was collected across the three sites for study {d}?",
                         QueryType.TEXT_ANSWERABLE, [f"{doc}_00002#r000"]))
        # Answer lives ONLY in the figure region:
        qa.append(QAItem(f"v{d}", f"What score did the {method} reach?",
                         QueryType.VISUAL_REQUIRING, [f"{doc}_00001#r002"]))
    print(f"4. QA set: {len(qa)} questions across 2 segments")

    systems = {
        "baseline": baseline,
        "enhanced": enhanced,
        "bm25_floor": BM25Retriever(
            BM25Index([c.chunk_id for c in chunks], [c.text for c in chunks]), top_k=5),
        "random_floor": RandomRetriever(
            [c.chunk_id for c in chunks], seed=cfg.seed, top_k=5),
    }
    payload = run_comparison(
        systems, qa, {c.chunk_id: c.source_region_ids for c in chunks},
        {c.chunk_id for c in chunks} | {r.region_id for r in regions},
        cfg, allow_stub=True)
    print(format_summary(payload, cfg.eval.primary_k))

    out = Path("results/smoke")
    write_results(payload, out, cfg.eval.primary_k)
    print(f"\n5. results written to {out}/")

    # --- assertions on behaviour, not just execution ---
    d = payload["deltas"]["by_segment"]
    vis = d["visual_requiring"]["recall@5"]
    b_vis = payload["systems"]["baseline"]["by_segment"]["visual_requiring"]["recall"]["5"]
    e_vis = payload["systems"]["enhanced"]["by_segment"]["visual_requiring"]["recall"]["5"]
    rnd = payload["systems"]["random_floor"]["overall"]["recall"]["5"]
    e_all = payload["systems"]["enhanced"]["overall"]["recall"]["5"]

    print("\n" + "=" * 68); print("BEHAVIOURAL ASSERTIONS"); print("=" * 68)
    assert b_vis == 0.0, f"baseline must be blind to figure-only answers, got {b_vis}"
    print(f"  PASS  baseline recall on visual questions = {b_vis:.3f} (structurally blind)")
    assert e_vis > 0.0, f"enhanced must reach figure evidence, got {e_vis}"
    print(f"  PASS  enhanced recall on visual questions = {e_vis:.3f}")
    assert vis > 0, f"delta must favour enhanced on visual segment, got {vis}"
    print(f"  PASS  delta on visual segment = {vis:+.3f}")
    assert e_all > rnd, f"enhanced ({e_all}) must beat random floor ({rnd})"
    print(f"  PASS  enhanced {e_all:.3f} > random floor {rnd:.3f}")

    # determinism
    p2 = run_comparison(systems, qa, {c.chunk_id: c.source_region_ids for c in chunks},
                        set(), cfg, allow_stub=True)
    assert p2["systems"]["enhanced"]["overall"] == payload["systems"]["enhanced"]["overall"]
    print("  PASS  two runs produce identical results (deterministic)")
    bm25_all = payload["systems"]["bm25_floor"]["overall"]["recall"]["5"]
    assert e_all >= bm25_all, f"enhanced ({e_all}) should not trail BM25 ({bm25_all})"
    print(f"  PASS  enhanced {e_all:.3f} >= BM25 lexical floor {bm25_all:.3f}")
    print("=" * 68); print("SMOKE TEST PASSED"); print("=" * 68)

if __name__ == "__main__":
    main()
