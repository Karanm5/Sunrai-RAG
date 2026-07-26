"""Pluggable LLM backends with an on-disk response cache.

Three backends behind one interface:

* `AnthropicBackend` - default, highest answer quality.
* `LocalBackend`     - a small HF instruct model, so the pipeline runs with
                       no API key and no network. Lower quality; stated.
* `StubBackend`      - deterministic, dependency-free, used by the test suite
                       and CI so the full pipeline is exercised on every push.

The caching layer matters more than it looks. It makes the evaluation
*reproducible* (a re-run replays identical responses rather than resampling)
and cheap (a re-run costs nothing). Determinism in an LLM pipeline is
otherwise unattainable, so this is the mechanism behind the reproducibility
claim in the report.

Design note for the SUNRAI context: local-vs-API is a first-class switch, not
an afterthought, because a platform that must run "on local systems or in the
cloud" cannot hard-depend on a hosted model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)


class LLMBackend(Protocol):
    name: str

    def complete(self, prompt: str, system: str | None = None) -> str: ...


class ResponseCache:
    """Content-addressed prompt -> response cache.

    Key includes the backend name and model so switching models never serves
    a stale response from a different one.
    """

    def __init__(self, path: str | Path | None, enabled: bool = True):
        self.path = Path(path) if path else None
        self.enabled = enabled and self.path is not None
        self._data: dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        if self.enabled and self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log.warning("Could not read LLM cache at %s; starting empty", self.path)
                self._data = {}

    @staticmethod
    def make_key(backend: str, model: str, prompt: str, system: str | None) -> str:
        digest = hashlib.sha256()
        for part in (backend, model, system or "", prompt):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        value = self._data.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        self._data[key] = value

    def flush(self) -> None:
        if not self.enabled or not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "entries": len(self._data),
        }


@dataclass
class AnthropicBackend:
    """Anthropic Messages API backend."""

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 800
    temperature: float = 0.0
    cache: ResponseCache | None = None
    name: str = field(default="anthropic", init=False)

    def complete(self, prompt: str, system: str | None = None) -> str:
        key = ResponseCache.make_key(self.name, self.model, prompt, system)
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        import anthropic  # lazy import

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Either export it, or run with "
                "llm.backend=local (or =stub) to reproduce without an API key."
            )
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        if self.cache:
            self.cache.put(key, text)
            self.cache.flush()
        return text


@dataclass
class LocalBackend:
    """Small local instruct model via transformers. No API key, no network
    after the first model download. Answer quality is materially lower than
    the API backend -- the report states this rather than comparing across
    backends as though they were equivalent."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_tokens: int = 800
    cache: ResponseCache | None = None
    name: str = field(default="local", init=False)
    _pipe: object | None = field(default=None, init=False, repr=False)

    def _ensure_pipe(self):
        if self._pipe is None:
            from transformers import pipeline  # lazy import

            self._pipe = pipeline(
                "text-generation", model=self.model_name, device_map="auto"
            )
        return self._pipe

    def complete(self, prompt: str, system: str | None = None) -> str:
        key = ResponseCache.make_key(self.name, self.model_name, prompt, system)
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        pipe = self._ensure_pipe()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        out = pipe(  # type: ignore[operator]
            messages,
            max_new_tokens=self.max_tokens,
            do_sample=False,  # greedy -> deterministic
            return_full_text=False,
        )
        text = out[0]["generated_text"]
        if isinstance(text, list):  # some versions return message dicts
            text = text[-1].get("content", "")
        text = str(text).strip()
        if self.cache:
            self.cache.put(key, text)
            self.cache.flush()
        return text


@dataclass
class StubBackend:
    """Deterministic offline backend for tests and CI.

    Produces a structurally valid answer derived from the prompt so the full
    pipeline -- including citation parsing and provenance assembly -- is
    exercised without any model. It is not a quality signal and must never be
    used for reported results; `run_eval` refuses to write headline results
    when this backend is active.
    """

    name: str = field(default="stub", init=False)
    responses: dict[str, str] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls.append(prompt)
        for trigger, response in self.responses.items():
            if trigger in prompt:
                return response
        if "Return ONLY valid JSON" in prompt:
            return '{"entities": [], "relations": []}'
        if "faithfulness" in prompt.lower():
            return '{"faithfulness": 1.0, "relevance": 1.0}'
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return f"Stub answer derived from the provided context [{digest}]."


def build_llm(cfg, cache_path: str | Path | None = None) -> LLMBackend:
    """Construct the backend named in config."""
    cache = ResponseCache(cache_path, enabled=cfg.llm.cache_enabled)
    backend = cfg.llm.backend
    if backend == "anthropic":
        return AnthropicBackend(
            model=cfg.llm.model,
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
            cache=cache,
        )
    if backend == "local":
        return LocalBackend(
            model_name=cfg.llm.local_model, max_tokens=cfg.llm.max_tokens, cache=cache
        )
    if backend == "stub":
        return StubBackend()
    raise ValueError(f"Unknown llm.backend: {backend!r}")
