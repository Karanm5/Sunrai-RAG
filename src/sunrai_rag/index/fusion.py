"""Reciprocal Rank Fusion (RRF) for late multimodal fusion.

Why RRF rather than a weighted score sum: dense cosine scores, BM25 scores
and CLIP similarities live on incomparable scales, so summing them requires
per-modality normalisation and tuned weights -- both of which would need a
validation split this task does not have, and would risk tuning on the test
set. RRF uses only *rank*, so it is scale-free and has a single, robust
hyper-parameter.

    RRF(d) = sum over rankers r of  weight_r / (k + rank_r(d))

Cormack et al. (2009) introduced RRF and found k=60 robust; we keep that as
the default and expose it in config rather than hard-coding it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from ..schemas import ScoredItem


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredItem]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[ScoredItem]:
    """Fuse several ranked lists into one.

    Properties the test suite pins down:
      * order-invariant  - fusing [A, B] equals fusing [B, A]
      * rank-monotonic   - improving an item's rank never lowers its score
      * scale-free       - the input scores are ignored entirely

    Items are identified by `item_id`; an item appearing in several rankings
    accumulates contributions from each. The output `modality` is set to
    "fused", with the contributing modalities recorded in the order they were
    first seen.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights ({len(weights)}) must align with rankings ({len(rankings)})"
        )

    fused: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    modalities: dict[str, list[str]] = defaultdict(list)

    counter = 0
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, item in enumerate(ranking):
            # Trust the explicit rank when present; otherwise use list order.
            rank = item.rank if item.rank is not None else position + 1
            fused[item.item_id] += weight / (k + rank)
            if item.item_id not in first_seen:
                first_seen[item.item_id] = counter
                counter += 1
            if item.modality not in modalities[item.item_id]:
                modalities[item.item_id].append(item.modality)

    # Sort by fused score desc; break ties by first-seen order so the result
    # is a total order and therefore reproducible.
    ordered = sorted(
        fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]])
    )
    if top_n is not None:
        ordered = ordered[:top_n]

    return [
        ScoredItem(
            item_id=item_id,
            score=float(score),
            modality="fused:" + "+".join(modalities[item_id]),
            rank=rank + 1,
        )
        for rank, (item_id, score) in enumerate(ordered)
    ]


def dedupe_preserving_order(items: Iterable[ScoredItem]) -> list[ScoredItem]:
    """Drop repeated item_ids, keeping the first (best-ranked) occurrence."""
    seen: set[str] = set()
    out: list[ScoredItem] = []
    for item in items:
        if item.item_id in seen:
            continue
        seen.add(item.item_id)
        out.append(item)
    return out
