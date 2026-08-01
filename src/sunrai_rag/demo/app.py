"""Streamlit demo: ask a question, compare both systems, inspect the evidence.

The demo is built around the comparison rather than around a single answer
box. Showing baseline and enhanced side by side, each with its provenance,
makes the project's claim inspectable by a non-technical reviewer, which
is the point of the explainability requirement.

Run:  streamlit run src/sunrai_rag/demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sunrai_rag.config import load_config, set_global_seeds  # noqa: E402
from sunrai_rag.index.vector_store import VectorStore  # noqa: E402
from sunrai_rag.ingest.loader import Corpus  # noqa: E402
from sunrai_rag.kg.graph import KnowledgeGraph  # noqa: E402
from sunrai_rag.rag.llm import build_llm  # noqa: E402
from sunrai_rag.rag.pipelines import BaselineRAG, EnhancedRAG  # noqa: E402
from sunrai_rag.represent.embedders import (  # noqa: E402
    build_image_embedder,
    build_text_embedder,
)

st.set_page_config(page_title="Multimodal RAG + KG", layout="wide")


@st.cache_resource(show_spinner="Loading pipeline…")
def load_systems(config_path: str):
    cfg = load_config(config_path if Path(config_path).exists() else None)
    set_global_seeds(cfg.seed)
    art = Path(cfg.paths.artifacts_dir)

    corpus_path = art / "corpus.json"
    if not corpus_path.exists():
        return None, None, None, cfg, "Corpus not found. Run: make ingest"

    corpus = Corpus.load(corpus_path)
    text_index = art / "text_index"
    if not text_index.exists():
        return None, None, corpus, cfg, "Text index not found. Run: make index"

    text_store = VectorStore.load(text_index)
    text_embedder = build_text_embedder(cfg)
    llm = build_llm(cfg, Path(cfg.paths.cache_dir) / "llm_cache.json")

    image_store = None
    image_embedder = None
    if (art / "image_index").exists():
        image_store = VectorStore.load(art / "image_index")
        image_embedder = build_image_embedder(cfg)

    # The OCR-text index over visual regions is the route that actually
    # works on scientific documents; omitting it here would demo the
    # CLIP-only path, which retrieves at chance on this content.
    visual_text_store = (
        VectorStore.load(art / "visual_text_index")
        if (art / "visual_text_index").exists()
        else None
    )

    kg = KnowledgeGraph.load(art / "kg.json") if (art / "kg.json").exists() else None

    baseline = BaselineRAG(
        chunks=corpus.chunks, text_store=text_store,
        text_embedder=text_embedder, llm=llm, top_k=cfg.retrieval.top_k,
    )
    enhanced = EnhancedRAG(
        chunks=corpus.chunks, regions=corpus.regions, text_store=text_store,
        text_embedder=text_embedder, llm=llm, image_store=image_store,
        image_embedder=image_embedder, visual_text_store=visual_text_store,
        kg=kg, top_k=cfg.retrieval.top_k,
        candidate_k=cfg.retrieval.candidate_k, rrf_k=cfg.retrieval.rrf_k,
        graph_hops=cfg.retrieval.graph_hops,
        max_graph_regions=cfg.retrieval.max_graph_regions,
        text_weight=cfg.retrieval.text_weight,
        graph_weight=cfg.retrieval.graph_weight,
    )
    return baseline, enhanced, corpus, cfg, None


def render_provenance(answer, corpus, show_images: bool = True) -> None:
    """Render the evidence behind an answer: text, figures, and graph paths."""
    prov = answer.provenance
    st.caption(f"Retrieval strategy: `{prov.retrieval_strategy}`")

    if prov.text_chunks:
        st.markdown("**Text evidence**")
        for item in prov.text_chunks:
            chunk = corpus.chunk_by_id(item.item_id)
            if chunk is None:
                continue
            with st.expander(f"{item.item_id}  ·  score {item.score:.3f}"):
                st.write(chunk.text)
                st.caption(f"{chunk.doc_id}, page {chunk.page_id}")

    if prov.visual_regions:
        st.markdown("**Visual evidence**")
        for item in prov.visual_regions:
            region = corpus.region_by_id(item.item_id)
            if region is None:
                continue
            st.caption(
                f"{item.item_id} · {region.region_type.value} · score {item.score:.3f}"
            )
            if show_images and region.image_path and Path(region.image_path).exists():
                st.image(region.image_path, width=380)
            elif region.text:
                st.info(region.text)

    if prov.graph_entities:
        st.markdown("**Knowledge graph**")
        st.write("Linked entities: " + ", ".join(f"`{e}`" for e in prov.graph_entities))
        for path in prov.graph_paths[:5]:
            st.caption(" → ".join(path))


def main() -> None:
    st.title("Multimodal RAG + Knowledge Graph over scientific documents")
    st.caption(
        "Baseline (text-only) vs Enhanced (text + figures + knowledge graph). "
        "Every answer shows the evidence it was built from."
    )

    config_path = st.sidebar.text_input("Config", "configs/default.yaml")
    baseline, enhanced, corpus, cfg, error = load_systems(config_path)

    if error:
        st.error(error)
        st.stop()

    with st.sidebar:
        st.subheader("Corpus")
        st.metric("Documents", len(corpus.doc_ids))
        st.metric("Regions", len(corpus.regions))
        st.metric("Text chunks", len(corpus.chunks))
        st.metric("Figures / tables", len(corpus.visual_regions()))
        if enhanced.visual_text_store is not None:
            st.caption(
                f"visual-text index: {len(enhanced.visual_text_store)} regions"
            )
        if corpus.ocr_stats:
            st.caption(f"OCR usable rate: {corpus.ocr_stats.get('usable_rate', 0):.1%}")
        st.divider()
        st.caption(f"LLM backend: `{cfg.llm.backend}`")
        st.caption(f"Top-k: {cfg.retrieval.top_k}")

        results_path = Path(cfg.paths.results_dir) / "results.json"
        if results_path.exists():
            st.divider()
            st.subheader("Latest results")
            payload = json.loads(results_path.read_text())
            deltas = payload.get("deltas", {}).get("by_segment", {})
            for segment, delta in deltas.items():
                key = f"recall@{cfg.eval.primary_k}"
                st.metric(segment, f"{delta.get(key, 0):+.3f}", help="enhanced − baseline")

    question = st.text_input(
        "Question",
        placeholder="e.g. Which method achieved the highest reported accuracy?",
    )
    mode = st.radio(
        "Mode", ["Compare both", "Enhanced only", "Baseline only"], horizontal=True
    )

    if not question:
        st.info("Enter a question to see retrieval and generation in action.")
        return

    if mode == "Compare both":
        left, right = st.columns(2)
        with left:
            st.subheader("Baseline, text only")
            with st.spinner("Retrieving…"):
                answer = baseline.answer(question)
            st.write(answer.answer_text)
            st.caption(f"{answer.latency_s:.2f}s")
            render_provenance(answer, corpus, show_images=False)
        with right:
            st.subheader("Enhanced, multimodal + KG")
            with st.spinner("Retrieving…"):
                answer = enhanced.answer(question)
            st.write(answer.answer_text)
            st.caption(f"{answer.latency_s:.2f}s")
            render_provenance(answer, corpus)
    else:
        system = enhanced if mode == "Enhanced only" else baseline
        with st.spinner("Retrieving…"):
            answer = system.answer(question)
        st.write(answer.answer_text)
        st.caption(f"{answer.latency_s:.2f}s")
        render_provenance(answer, corpus)


if __name__ == "__main__":
    main()
