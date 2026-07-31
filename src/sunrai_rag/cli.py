"""Command-line entry points for every pipeline stage.

    python -m sunrai_rag.cli ingest    --config configs/default.yaml
    python -m sunrai_rag.cli index     --config configs/default.yaml
    python -m sunrai_rag.cli kg        --config configs/default.yaml
    python -m sunrai_rag.cli build-qa  --config configs/default.yaml
    python -m sunrai_rag.cli evaluate  --config configs/default.yaml
    python -m sunrai_rag.cli ask       --config configs/default.yaml -q "..."

Each stage persists its artefact, so a stage can be re-run without repeating
the expensive ones. This is what makes iteration on retrieval cheap after a
single slow OCR pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import load_config, set_global_seeds
from .eval.build_qa import build_qa_set, load_qa_set, qa_set_stats, save_qa_set
from .eval.run_eval import (
    format_summary,
    run_comparison,
    verify_from_log,
    write_results,
    write_retrieval_log,
)
from .index.bm25 import BM25Index
from .index.vector_store import VectorStore
from .ingest.loader import Corpus, ingest
from .kg.extract import LLMExtractor, RuleExtractor, extract_corpus
from .kg.graph import KnowledgeGraph
from .rag.llm import build_llm
from .rag.pipelines import (
    BaselineRAG,
    BM25Retriever,
    EnhancedRAG,
    RandomRetriever,
)
from .represent.embedders import build_image_embedder, build_text_embedder

log = logging.getLogger("sunrai_rag")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _paths(cfg):
    art = Path(cfg.paths.artifacts_dir)
    return {
        "corpus": art / "corpus.json",
        "text_index": art / "text_index",
        "image_index": art / "image_index",
        "visual_text_index": art / "visual_text_index",
        "kg": art / "kg.json",
        "qa": art / "qa_set.json",
        "cache": Path(cfg.paths.cache_dir) / "llm_cache.json",
    }


def _require(path: Path, stage: str) -> None:
    if not path.exists():
        sys.exit(f"Missing {path}. Run `python -m sunrai_rag.cli {stage}` first.")


# ---------------------------------------------------------------- stages


def cmd_ingest(cfg) -> None:
    paths = _paths(cfg)
    corpus = ingest(cfg)
    corpus.save(paths["corpus"])
    log.info(
        "Ingested %d pages -> %d regions, %d chunks (OCR: %s)",
        corpus.page_count, len(corpus.regions), len(corpus.chunks), corpus.ocr_stats,
    )
    log.info("Saved corpus to %s", paths["corpus"])


def cmd_index(cfg) -> None:
    paths = _paths(cfg)
    _require(paths["corpus"], "ingest")
    corpus = Corpus.load(paths["corpus"])

    log.info("Embedding %d text chunks", len(corpus.chunks))
    text_embedder = build_text_embedder(cfg)
    text_vectors = text_embedder.embed_texts([c.text for c in corpus.chunks])
    VectorStore(
        [c.chunk_id for c in corpus.chunks], text_vectors, modality="text"
    ).save(paths["text_index"])

    # Dense index over OCR'd text from tables and figures. This is the
    # retrieval route that actually works on scientific documents; CLIP over
    # raw crops performs at chance on them.
    visual_with_text = [r for r in corpus.visual_regions() if (r.text or "").strip()]
    if visual_with_text:
        log.info("Embedding OCR text from %d visual regions", len(visual_with_text))
        visual_vectors = text_embedder.embed_texts(
            [r.text or "" for r in visual_with_text]
        )
        VectorStore(
            [r.region_id for r in visual_with_text], visual_vectors, modality="image"
        ).save(paths["visual_text_index"])
    else:
        log.warning("No OCR text on visual regions; skipping visual-text index.")

    visual = [r for r in corpus.visual_regions() if r.image_path]
    if visual:
        from PIL import Image

        log.info("Embedding %d visual regions with CLIP", len(visual))
        image_embedder = build_image_embedder(cfg)
        images = [Image.open(r.image_path) for r in visual]
        image_vectors = image_embedder.embed_images(images)
        VectorStore(
            [r.region_id for r in visual], image_vectors, modality="image"
        ).save(paths["image_index"])
    else:
        log.warning("No visual regions with saved crops; skipping image index.")
    log.info("Indices written to %s", cfg.paths.artifacts_dir)


def cmd_kg(cfg) -> None:
    paths = _paths(cfg)
    _require(paths["corpus"], "ingest")
    corpus = Corpus.load(paths["corpus"])

    if cfg.kg.extractor == "llm":
        extractor = LLMExtractor(
            llm=build_llm(cfg, paths["cache"]), batch_size=cfg.kg.batch_size
        )
    else:
        extractor = RuleExtractor()

    textual = [r for r in corpus.regions if r.is_textual and r.text]
    log.info("Extracting KG from %d regions using %s", len(textual), cfg.kg.extractor)
    extraction = extract_corpus(
        textual, extractor, max_regions=cfg.kg.max_regions_for_extraction
    )
    kg = KnowledgeGraph.from_extraction(extraction)
    kg.save(paths["kg"])
    log.info("Knowledge graph: %s", json.dumps(kg.stats()))


def _load_systems(cfg, corpus: Corpus):
    paths = _paths(cfg)
    _require(paths["text_index"], "index")
    text_store = VectorStore.load(paths["text_index"])
    text_embedder = build_text_embedder(cfg)
    llm = build_llm(cfg, paths["cache"])

    image_store = None
    image_embedder = None
    if paths["image_index"].exists():
        image_store = VectorStore.load(paths["image_index"])
        image_embedder = build_image_embedder(cfg)

    visual_text_store = (
        VectorStore.load(paths["visual_text_index"])
        if paths["visual_text_index"].exists()
        else None
    )

    kg = KnowledgeGraph.load(paths["kg"]) if paths["kg"].exists() else None

    baseline = BaselineRAG(
        chunks=corpus.chunks, text_store=text_store, text_embedder=text_embedder,
        llm=llm, top_k=cfg.retrieval.top_k,
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
    return baseline, enhanced, llm


def cmd_build_qa(cfg) -> None:
    paths = _paths(cfg)
    _require(paths["corpus"], "ingest")
    corpus = Corpus.load(paths["corpus"])
    llm = build_llm(cfg, paths["cache"])
    items = build_qa_set(
        corpus.chunks, corpus.visual_regions(), llm,
        n_per_type=cfg.eval.n_questions_per_type, seed=cfg.seed,
    )
    save_qa_set(items, paths["qa"])
    log.info("QA set: %s", json.dumps(qa_set_stats(items)))


def cmd_evaluate(cfg) -> None:
    paths = _paths(cfg)
    _require(paths["corpus"], "ingest")
    _require(paths["qa"], "build-qa")
    corpus = Corpus.load(paths["corpus"])
    qa_items = load_qa_set(paths["qa"])
    baseline, enhanced, llm = _load_systems(cfg, corpus)

    systems = {"baseline": baseline, "enhanced": enhanced}

    # Sanity floors. These exist to make a weak result visible: dense
    # retrieval that fails to beat BM25 has not earned the "semantic"
    # claim, and neither floor shows up in an absolute Recall number.
    if cfg.retrieval.use_bm25_floor:
        systems["bm25_floor"] = BM25Retriever(
            BM25Index(
                [c.chunk_id for c in corpus.chunks],
                [c.text for c in corpus.chunks],
            ),
            top_k=cfg.retrieval.top_k,
        )
    if cfg.eval.include_random_floor:
        systems["random_floor"] = RandomRetriever(
            [c.chunk_id for c in corpus.chunks], seed=cfg.seed, top_k=cfg.retrieval.top_k
        )

    chunk_to_regions = {c.chunk_id: c.source_region_ids for c in corpus.chunks}
    known_ids = {c.chunk_id for c in corpus.chunks} | {r.region_id for r in corpus.regions}

    payload = run_comparison(
        systems, qa_items, chunk_to_regions, known_ids, cfg,
        judge_llm=llm if cfg.eval.judge_enabled else None,
    )
    csv_path = write_results(payload, cfg.paths.results_dir, cfg.eval.primary_k)
    log_path = write_retrieval_log(payload.pop("_results"), cfg.paths.results_dir)
    print(format_summary(payload, cfg.eval.primary_k))
    log.info("Wrote %s", csv_path)
    log.info("Wrote %s (recheck with: sunrai-rag verify)", log_path)


def cmd_verify(cfg) -> None:
    """Recompute the headline metrics from the saved retrieval log."""
    path = Path(cfg.paths.results_dir) / "retrieval_log.json"
    if not path.exists():
        sys.exit(f"{path} not found. Run `sunrai-rag evaluate` first.")
    print(verify_from_log(path, cfg.eval.k_values, cfg.eval.primary_k))


def cmd_ask(cfg, question: str, system_name: str) -> None:
    paths = _paths(cfg)
    _require(paths["corpus"], "ingest")
    corpus = Corpus.load(paths["corpus"])
    baseline, enhanced, _ = _load_systems(cfg, corpus)
    system = baseline if system_name == "baseline" else enhanced

    answer = system.answer(question)
    print(f"\nQ: {question}\nSystem: {answer.system_name}\n\n{answer.answer_text}\n")
    print("Evidence:")
    for item in answer.provenance.text_chunks:
        print(f"  [text]  {item.item_id}  (score {item.score:.3f})")
    for item in answer.provenance.visual_regions:
        print(f"  [image] {item.item_id}  (score {item.score:.3f})")
    if answer.provenance.graph_entities:
        print(f"  [graph] entities: {', '.join(answer.provenance.graph_entities)}")
        for path in answer.provenance.graph_paths:
            print(f"          path: {' -> '.join(path)}")


# ---------------------------------------------------------------- main


def cmd_doctor(cfg) -> int:
    """Check the environment before a long run fails halfway through.

    Every problem hit during this project's bring-up -- a missing tesseract
    binary, an absent optional dependency, an unset API key, a retired model
    name -- announced itself only after minutes of work had already been
    spent. This checks all of them up front, on any platform, in seconds.

    Returns the number of blocking problems found.
    """
    import platform
    import shutil

    ok, warn, fail = "  OK  ", " WARN ", " FAIL "
    problems = 0

    print("=" * 62)
    print("ENVIRONMENT CHECK")
    print("=" * 62)
    print(f"{ok}python {platform.python_version()} on {platform.system()}")

    # --- required imports -------------------------------------------------
    required = {
        "numpy": "numpy", "yaml": "pyyaml", "networkx": "networkx",
        "PIL": "pillow", "datasets": "datasets", "requests": "requests",
    }
    optional = {
        "pytesseract": "pytesseract (needed for ingest)",
        "sentence_transformers": "sentence-transformers (needed for index)",
        "transformers": "transformers (needed for index)",
        "torch": "torch (needed for index)",
    }
    for module, package in required.items():
        try:
            __import__(module)
            print(f"{ok}{package}")
        except ImportError:
            print(f"{fail}{package} missing  ->  pip install {package}")
            problems += 1
    for module, label in optional.items():
        try:
            __import__(module)
            print(f"{ok}{label}")
        except ImportError:
            print(f"{warn}{label} missing")

    # --- tesseract binary -------------------------------------------------
    from .ingest.ocr import find_tesseract_binary

    path = find_tesseract_binary(cfg.ingest.tesseract_cmd or None)
    if path:
        print(f"{ok}tesseract at {path}")
    else:
        print(f"{fail}tesseract not found")
        print("        Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("        macOS:   brew install tesseract")
        print("        Linux:   sudo apt-get install tesseract-ocr tesseract-ocr-eng")
        problems += 1

    # --- generation backend ----------------------------------------------
    backend = cfg.llm.backend
    if backend in ("stub", "local"):
        print(f"{ok}llm backend '{backend}' needs no API key")
    else:
        env_var = cfg.llm.api_key_env or {
            "anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY", "openrouter": "OPENROUTER_API_KEY",
        }.get(backend, "")
        if env_var and os.environ.get(env_var):
            print(f"{ok}{env_var} is set")
            if backend == "groq":
                _check_groq_model(cfg, ok, fail)
        elif env_var:
            print(f"{fail}{env_var} is not set")
            print(f"        export {env_var}=your_key_here")
            problems += 1

    # --- disk -------------------------------------------------------------
    free_gb = shutil.disk_usage(".").free / 1e9
    if free_gb < 3:
        print(f"{warn}only {free_gb:.1f} GB free; models need ~2 GB")
    else:
        print(f"{ok}{free_gb:.1f} GB disk free")

    print("=" * 62)
    if problems:
        print(f"{problems} blocking problem(s). Fix these before running.")
    else:
        print("All good. Run:  sunrai-rag ingest --config <your config>")
    print("=" * 62)
    return problems


def _check_groq_model(cfg, ok: str, fail: str) -> None:
    """Confirm the configured model still exists; providers retire them."""
    try:
        import requests

        response = requests.get(
            f"{cfg.llm.base_url or 'https://api.groq.com/openai/v1'}/models",
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            timeout=15,
        )
        if response.status_code != 200:
            print(f"{fail}could not list models (HTTP {response.status_code})")
            return
        available = sorted(m["id"] for m in response.json().get("data", []))
        if cfg.llm.model in available:
            print(f"{ok}model '{cfg.llm.model}' is available")
        else:
            print(f"{fail}model '{cfg.llm.model}' is NOT available")
            print("        Pick one of these and set llm.model in your config:")
            for name in available[:12]:
                print(f"          {name}")
    except Exception as exc:  # network flake should not fail the whole check
        print(f"        (could not verify model: {exc})")



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sunrai_rag", description=__doc__)
    parser.add_argument("stage", choices=[
        "doctor", "ingest", "index", "kg", "build-qa", "evaluate", "verify",
        "ask", "all",
    ])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("-q", "--question", default=None)
    parser.add_argument("--system", default="enhanced", choices=["baseline", "enhanced"])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    cfg = load_config(args.config if Path(args.config).exists() else None)
    set_global_seeds(cfg.seed)
    cfg.ensure_dirs()
    log.info("Config loaded (seed=%d, llm=%s)", cfg.seed, cfg.llm.backend)

    if args.stage == "doctor":
        raise SystemExit(1 if cmd_doctor(cfg) else 0)
    if args.stage == "ingest":
        cmd_ingest(cfg)
    elif args.stage == "index":
        cmd_index(cfg)
    elif args.stage == "kg":
        cmd_kg(cfg)
    elif args.stage == "build-qa":
        cmd_build_qa(cfg)
    elif args.stage == "evaluate":
        cmd_evaluate(cfg)
    elif args.stage == "verify":
        cmd_verify(cfg)
    elif args.stage == "ask":
        if not args.question:
            sys.exit("ask requires -q/--question")
        cmd_ask(cfg, args.question, args.system)
    elif args.stage == "all":
        cmd_ingest(cfg)
        cmd_index(cfg)
        cmd_kg(cfg)
        cmd_build_qa(cfg)
        cmd_evaluate(cfg)


if __name__ == "__main__":
    main()
