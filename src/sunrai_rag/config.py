"""Configuration loading, validation, and global determinism control.

Every entry point takes `--config`. No magic numbers live in the code: if a
number affects a result, it is in the YAML and therefore appears in the
reproducibility trail.

`set_global_seeds()` is called once at the top of every script. Combined with
the on-disk LLM cache, this is what makes two runs of the evaluation produce
byte-identical results tables.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    results_dir: str = "results"
    cache_dir: str = "artifacts/llm_cache"


@dataclass
class IngestConfig:
    dataset_url_template: str = (
        "https://huggingface.co/datasets/lhoestq/small-publaynet-wds/"
        "resolve/main/publaynet-train-{i:06d}.tar"
    )
    num_shards: int = 1
    max_pages: int = 200
    min_region_side_px: float = 8.0
    min_ocr_chars: int = 20
    ocr_lang: str = "eng"
    ocr_psm: int = 6
    ocr_min_crop_height: int = 200  # upscale small crops; see TesseractEngine
    tesseract_cmd: str = ""  # full path to tesseract.exe; blank = auto-detect
    save_crops: bool = True


@dataclass
class ChunkConfig:
    strategy: str = "region"  # "region" | "window"
    window_chars: int = 900
    overlap_chars: int = 150
    min_chunk_chars: int = 40


@dataclass
class ModelConfig:
    text_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embed_revision: str | None = None
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_revision: str | None = None
    caption_model: str = "Salesforce/blip-image-captioning-base"
    enable_captions: bool = False
    embed_batch_size: int = 32
    device: str = "auto"  # "auto" | "cpu" | "cuda"


@dataclass
class LLMConfig:
    # "anthropic" | "groq" | "together" | "openrouter" | "ollama"
    # | "openai_compatible" | "local" | "stub"
    backend: str = "stub"
    model: str = "claude-sonnet-4-6"
    local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_tokens: int = 800
    temperature: float = 0.0
    cache_enabled: bool = True
    # OpenAI-compatible providers only. Left blank, the provider preset fills
    # these in; set them to override or to point at a self-hosted endpoint.
    base_url: str = ""
    api_key_env: str = ""
    timeout_s: int = 90
    max_retries: int = 5
    requests_per_minute: int = 30  # client-side throttle; 0 disables


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    use_bm25_floor: bool = True
    graph_hops: int = 1
    max_graph_regions: int = 5


@dataclass
class KGConfig:
    extractor: str = "rule"  # "llm" | "rule"
    entity_types: list[str] = field(
        default_factory=lambda: ["method", "metric", "dataset", "finding", "figure_ref"]
    )
    max_regions_for_extraction: int = 400
    batch_size: int = 5  # regions per LLM call; 1 disables batching


@dataclass
class EvalConfig:
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    primary_k: int = 5
    n_questions_per_type: int = 25
    judge_enabled: bool = True
    judge_sample_size: int = 60
    include_random_floor: bool = True
    # Retrieval metrics need no LLM at all. Turning generation off gives the
    # full baseline-vs-enhanced retrieval comparison in seconds and zero API
    # calls, which is the graded core; answer quality can be measured in a
    # separate, slower pass.
    generate_answers: bool = True


@dataclass
class Config:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    kg: KGConfig = field(default_factory=KGConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def ensure_dirs(self) -> None:
        for p in (
            self.paths.data_dir,
            self.paths.artifacts_dir,
            self.paths.results_dir,
            self.paths.cache_dir,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)


_SECTION_TYPES = {
    "paths": PathsConfig,
    "ingest": IngestConfig,
    "chunk": ChunkConfig,
    "models": ModelConfig,
    "llm": LLMConfig,
    "retrieval": RetrievalConfig,
    "kg": KGConfig,
    "eval": EvalConfig,
}


class ConfigError(ValueError):
    """Raised when a config file contains unknown or invalid entries."""


def _build_section(name: str, cls: type, raw: dict[str, Any]) -> Any:
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in config section '{name}': {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    return cls(**raw)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load and validate a YAML config.

    Unknown keys raise rather than being silently ignored -- a silently
    ignored typo in a config is a reproducibility bug waiting to happen.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Top level of the config file must be a mapping.")

    unknown_top = set(raw) - set(_SECTION_TYPES) - {"seed"}
    if unknown_top:
        raise ConfigError(f"Unknown top-level config section(s): {sorted(unknown_top)}")

    sections: dict[str, Any] = {}
    for name, cls in _SECTION_TYPES.items():
        section_raw = raw.get(name) or {}
        if not isinstance(section_raw, dict):
            raise ConfigError(f"Config section '{name}' must be a mapping.")
        sections[name] = _build_section(name, cls, section_raw)

    cfg = Config(seed=int(raw.get("seed", 42)), **sections)

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ConfigError(f"Unknown config override: {key}")
        setattr(cfg, key, value)

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Cross-field sanity checks that a type system alone won't catch."""
    if cfg.retrieval.top_k <= 0:
        raise ConfigError("retrieval.top_k must be positive.")
    if cfg.retrieval.candidate_k < cfg.retrieval.top_k:
        raise ConfigError(
            "retrieval.candidate_k must be >= retrieval.top_k "
            f"(got {cfg.retrieval.candidate_k} < {cfg.retrieval.top_k})."
        )
    if cfg.eval.primary_k not in cfg.eval.k_values:
        raise ConfigError(
            f"eval.primary_k ({cfg.eval.primary_k}) must appear in "
            f"eval.k_values ({cfg.eval.k_values})."
        )
    valid_backends = {
        "anthropic", "groq", "together", "openrouter", "ollama",
        "openai_compatible", "local", "stub",
    }
    if cfg.llm.backend not in valid_backends:
        raise ConfigError(
            f"Unknown llm.backend: {cfg.llm.backend!r}. "
            f"Valid: {sorted(valid_backends)}"
        )
    if cfg.llm.backend == "openai_compatible" and not (
        cfg.llm.base_url and cfg.llm.api_key_env
    ):
        raise ConfigError(
            "llm.backend='openai_compatible' requires llm.base_url and "
            "llm.api_key_env. Use a named preset (groq, together, ...) to "
            "have them filled in automatically."
        )
    if cfg.llm.requests_per_minute < 0:
        raise ConfigError("llm.requests_per_minute must be >= 0 (0 disables).")
    if cfg.kg.extractor not in {"llm", "rule"}:
        raise ConfigError(f"Unknown kg.extractor: {cfg.kg.extractor!r}")
    if cfg.chunk.strategy not in {"region", "window"}:
        raise ConfigError(f"Unknown chunk.strategy: {cfg.chunk.strategy!r}")
    if cfg.chunk.overlap_chars >= cfg.chunk.window_chars:
        raise ConfigError("chunk.overlap_chars must be < chunk.window_chars.")
    if cfg.llm.temperature != 0.0:
        # Not fatal, but it breaks the determinism guarantee we advertise.
        raise ConfigError(
            "llm.temperature must be 0.0 for reproducible results. "
            "Change it deliberately only if you also drop the determinism claim."
        )


def set_global_seeds(seed: int) -> None:
    """Seed every RNG that can affect a result.

    Torch is seeded only if installed, so the logic layers stay importable
    without the heavy ML stack (which is what lets the fast tests run).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:  # pragma: no cover - exercised only in the full environment
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
