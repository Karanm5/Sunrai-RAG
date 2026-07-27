"""Embedding backends behind one protocol.

The protocol split is what makes the pipeline testable: `HashingEmbedder`
gives deterministic vectors with no model download, so the retrieval,
fusion and evaluation logic is exercised end to end in CI.

CLIP is the multimodal component. It embeds figure/table crops and query
text into a *shared* space, which is what allows a text query to retrieve an
image directly -- the capability the text-only baseline structurally lacks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class TextEmbedder(Protocol):
    dim: int

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class ImageEmbedder(Protocol):
    dim: int

    def embed_images(self, images: Sequence[Any]) -> np.ndarray: ...
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class SentenceTransformerEmbedder:
    """Dense text embeddings via sentence-transformers."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "auto"
    _model: Any = field(default=None, init=False, repr=False)
    dim: int = field(default=384, init=False)

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            device = None if self.device == "auto" else self.device
            self._model = SentenceTransformer(self.model_name, device=device)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # VectorStore normalises
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class CLIPEmbedder:
    """CLIP image+text embeddings in a shared space."""

    model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 16
    device: str = "auto"
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    dim: int = field(default=512, init=False)

    def _ensure_model(self):
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor  # lazy

            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model.eval()
            if self.device != "cpu" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self.dim = int(self._model.config.projection_dim)
        return self._model, self._processor

    @staticmethod
    def _as_embedding(result: Any, model: Any, projection_name: str) -> Any:
        """Coerce a CLIP feature call into a projected embedding tensor.

        `get_image_features` / `get_text_features` normally return a tensor,
        but some transformers versions return a ModelOutput wrapper instead.
        Unwrapping it naively to `pooler_output` would be a silent
        correctness bug: that is the *pre-projection* hidden state (768-dim
        for ViT-B/32), while the other modality would still be projected
        (512-dim). The two would no longer share a space, and cross-modal
        retrieval would quietly return meaningless results.

        So when we get a wrapper, we apply the projection ourselves.
        """
        import torch

        if torch.is_tensor(result):
            return result
        embeds = getattr(result, "image_embeds", None)
        if embeds is None:
            embeds = getattr(result, "text_embeds", None)
        if embeds is not None and torch.is_tensor(embeds):
            return embeds
        pooled = getattr(result, "pooler_output", None)
        if pooled is None:
            raise RuntimeError(
                f"Could not extract embeddings from CLIP output of type "
                f"{type(result)}. Expected a tensor, image_embeds/text_embeds, "
                "or pooler_output."
            )
        projection = getattr(model, projection_name)
        return projection(pooled)

    def embed_images(self, images: Sequence[Any]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch

        model, processor = self._ensure_model()
        out: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = [im.convert("RGB") for im in images[start : start + self.batch_size]]
            inputs = processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(model.device)
            with torch.no_grad():
                result = model.get_image_features(pixel_values=pixel_values)
            features = self._as_embedding(result, model, "visual_projection")
            out.append(features.cpu().numpy().astype(np.float32))
        return np.vstack(out)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch

        model, processor = self._ensure_model()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            )
            model_inputs = {
                k: v.to(model.device)
                for k, v in inputs.items()
                if k in ("input_ids", "attention_mask")
            }
            with torch.no_grad():
                result = model.get_text_features(**model_inputs)
            features = self._as_embedding(result, model, "text_projection")
            out.append(features.cpu().numpy().astype(np.float32))
        return np.vstack(out)


@dataclass
class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. No model, no network.

    Real lexical signal (shared words produce similar vectors), so retrieval
    tests assert meaningful behaviour rather than tautologies. Used by the
    test suite and CI only -- never for reported results.
    """

    dim: int = 64

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            token = "".join(ch for ch in token if ch.isalnum())
            if not token:
                continue
            slot = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[slot] += 1.0
        return vector

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])

    def embed_images(self, images: Sequence[Any]) -> np.ndarray:
        """Embed images by their repr, so image retrieval is deterministic
        in tests without a vision model."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(str(im)) for im in images])


def build_text_embedder(cfg) -> TextEmbedder:
    return SentenceTransformerEmbedder(
        model_name=cfg.models.text_embed_model,
        batch_size=cfg.models.embed_batch_size,
        device=cfg.models.device,
    )


def build_image_embedder(cfg) -> ImageEmbedder:
    return CLIPEmbedder(
        model_name=cfg.models.clip_model,
        batch_size=cfg.models.embed_batch_size,
        device=cfg.models.device,
    )
