"""BM25 lexical retrieval, implemented directly.

Purpose: an honest sanity floor. If dense embeddings do not clearly beat
plain lexical matching, the "semantic retrieval" claim is unearned. Reporting
BM25 alongside the dense baseline is what turns a comparison into evidence.

Implemented in ~60 lines rather than pulled from a library so the scoring is
inspectable and pinned -- no silent behaviour change from a dependency bump.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..schemas import ScoredItem

_TOKEN = re.compile(r"[a-z0-9]+")

# Minimal stoplist: high-frequency function words that add noise to BM25 on
# scientific prose. Deliberately short -- aggressive stopping removes terms
# like "control" or "significant" that carry meaning in this domain.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those they we our""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation with a minimal stoplist."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """Okapi BM25 over a fixed document set."""

    def __init__(
        self,
        ids: list[str],
        texts: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if len(ids) != len(texts):
            raise ValueError("ids and texts must align")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique")

        self.ids = list(ids)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(t) for t in texts]
        self.doc_lens = [len(d) for d in self.doc_tokens]
        self.avg_doc_len = (
            sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        )
        self.term_freqs = [Counter(d) for d in self.doc_tokens]

        doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))
        self.doc_freq = doc_freq
        self.n_docs = len(self.ids)

    def __len__(self) -> int:
        return self.n_docs

    def _idf(self, term: str) -> float:
        """Robertson-Sparck Jones IDF with the standard +0.5 smoothing.

        Clamped at zero so terms appearing in most documents cannot push a
        score negative, which would make ranking non-monotonic.
        """
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return max(0.0, math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)))

    def score(self, query: str, doc_index: int) -> float:
        tokens = tokenize(query)
        if not tokens or self.avg_doc_len == 0:
            return 0.0
        tf = self.term_freqs[doc_index]
        doc_len = self.doc_lens[doc_index]
        total = 0.0
        for term in tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            numerator = freq * (self.k1 + 1.0)
            denominator = freq + self.k1 * (
                1.0 - self.b + self.b * doc_len / self.avg_doc_len
            )
            total += self._idf(term) * numerator / denominator
        return total

    def search(self, query: str, k: int) -> list[ScoredItem]:
        """Top-k by BM25, deterministic tie-breaking by document order."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self.n_docs == 0:
            return []
        scores = [self.score(query, i) for i in range(self.n_docs)]
        order = sorted(range(self.n_docs), key=lambda i: (-scores[i], i))[
            : min(k, self.n_docs)
        ]
        return [
            ScoredItem(
                item_id=self.ids[i],
                score=float(scores[i]),
                modality="lexical",
                rank=rank + 1,
            )
            for rank, i in enumerate(order)
        ]
