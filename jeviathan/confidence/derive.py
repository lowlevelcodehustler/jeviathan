"""Confidence derivation — TypeSafe's exact formula, generalized.

From docs.typesafe.ai/confidence: confidence collapses the shape of the
probability distribution into a single [0, 1] number. For n options their
demo uses (n * peak - 1) / (n - 1); all mass on one option -> 1.0, flat
spread -> 0.0. We use the same generalization for Choice and Score.
"""

from __future__ import annotations


def distribution_confidence(probs: dict[str, float]) -> float:
    if not probs:
        return 0.0
    n = len(probs)
    if n == 1:
        return 1.0
    peak = max(probs.values())
    value = (n * peak - 1.0) / (n - 1.0)
    return max(0.0, min(1.0, value))


def argmax(probs: dict[str, float]) -> str:
    """Deterministic argmax: first key with the maximum probability."""
    best_key: str | None = None
    best_val = -1.0
    for key, val in probs.items():
        if val > best_val:
            best_val = val
            best_key = key
    assert best_key is not None
    return best_key


def score_expectation(level_probs: dict[str, float], level_names: list[str]) -> float:
    """Continuous score as the expectation over ordered 0-based level indices.

    E.g. levels [calm, frustrated, very_frustrated] with distribution
    {calm: .1, frustrated: .6, very_frustrated: .3} -> 0*0.1 + 1*0.6 + 2*0.3 = 1.2
    (cf. TypeSafe's illustrative `score: 1.4`).
    """
    total = sum(level_probs.values()) or 1.0
    return sum(
        idx * level_probs.get(name, 0.0) / total
        for idx, name in enumerate(level_names)
    )
