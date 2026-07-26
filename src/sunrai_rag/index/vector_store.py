"""Dense vector index with exact search.

Exact (brute-force) cosine search is used deliberately. At this corpus scale
an approximate index (HNSW/IVF) buys nothing in latency but introduces
recall variance across builds, which would undermine the determinism claim
the evaluation rests on. The interface matches FAISS's `IndexFlatIP`
semantics, so swapping in FAISS for a larger corpus is a one-class change --
`FaissVectorStore` below does exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..schemas import ScoredItem


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise so inner product == cosine similarity.

    Zero rows are left as zeros rather than producing NaN; they simply never
    win a retrieval, which is the correct behaviour for an empty embedding.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return (matrix / safe).astype(np.float32)


class VectorStore:
    """In-memory exact-search index over L2-normalised embeddings."""

    def __init__(self, ids: list[str], embeddings: np.ndarray, modality: str = "text"):
        if len(ids) != len(embeddings):
            raise ValueError(
                f"ids ({len(ids)}) and embeddings ({len(embeddings)}) must align"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique")
        self.ids = list(ids)
        self.embeddings = l2_normalise(embeddings)
        self.modality = modality

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if len(self) else 0

    def search(self, query_vector: np.ndarray, k: int) -> list[ScoredItem]:
        """Return the top-k most similar items, highest score first.

        Ties are broken by index order, which makes the ranking a total order
        and therefore stable across runs -- important for reproducibility.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if len(self) == 0:
            return []

        query = l2_normalise(np.asarray(query_vector, dtype=np.float32))[0]
        if query.shape[0] != self.dim:
            raise ValueError(
                f"query dim {query.shape[0]} != index dim {self.dim}"
            )

        scores = self.embeddings @ query
        k = min(k, len(self))
        # argsort on (-score, index) gives deterministic tie-breaking.
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
        return [
            ScoredItem(
                item_id=self.ids[i],
                score=float(scores[i]),
                modality=self.modality,
                rank=rank + 1,
            )
            for rank, i in enumerate(order)
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "meta.json").write_text(
            json.dumps({"ids": self.ids, "modality": self.modality}),
            encoding="utf-8",
        )

    @staticmethod
    def load(directory: str | Path) -> VectorStore:
        directory = Path(directory)
        embeddings = np.load(directory / "embeddings.npy")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return VectorStore(meta["ids"], embeddings, modality=meta.get("modality", "text"))


class FaissVectorStore(VectorStore):
    """FAISS-backed drop-in for larger corpora.

    Uses IndexFlatIP, which is also exact -- so results match the numpy path.
    Present to show the scaling route without making FAISS a hard dependency
    of the reproduction.
    """

    def __init__(self, ids: list[str], embeddings: np.ndarray, modality: str = "text"):
        super().__init__(ids, embeddings, modality=modality)
        import faiss  # lazy import

        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(self.embeddings)

    def search(self, query_vector: np.ndarray, k: int) -> list[ScoredItem]:
        if k <= 0:
            raise ValueError("k must be positive")
        if len(self) == 0:
            return []
        query = l2_normalise(np.asarray(query_vector, dtype=np.float32))
        k = min(k, len(self))
        scores, indices = self._index.search(query, k)
        return [
            ScoredItem(
                item_id=self.ids[int(idx)],
                score=float(score),
                modality=self.modality,
                rank=rank + 1,
            )
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=True))
            if idx >= 0
        ]
