"""Calibration — the "RLCD-lite" meld.

Jev's killer feature is calibrated confidence (probabilities optimized
against outcomes). We can't RL-train a 27B on a laptop, so we approximate:
fit Platt scaling parameters (a, b) per question type / id on a labeled eval
set, then transform raw model probabilities through sigmoid(a*logit(p)+b).

Artifact JSON shape:
{
  "version": 1,
  "method": "platt",
  "fitted_at": "...",
  "profile": "rtx5090",
  "model": "jevitan-qwen3.8-27b",
  "params": {"choice": {"a": .., "b": ..}, "<qid>": {"a": .., "b": ..}}
}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _logit(p: float) -> float:
    eps = 1e-6
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def fit_platt(raw_probs: list[float], labels: list[int]) -> tuple[float, float]:
    """Fit (a, b) minimizing NLL of sigmoid(a*logit(p)+b). numpy-only."""
    import numpy as np

    if not raw_probs:
        return 1.0, 0.0
    z = np.array([_logit(p) for p in raw_probs], dtype=np.float64)
    y = np.array(labels, dtype=np.float64)
    a, b = 1.0, 0.0
    lr = 0.05
    for step in range(3000):
        logits = a * z + b
        s = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        )
        grad_a = -float(np.mean((y - s) * z))
        grad_b = -float(np.mean(y - s))
        a -= lr * grad_a
        b -= lr * grad_b
        if step % 500 == 499:
            lr *= 0.5
    return float(a), float(b)


def expected_calibration_error(
    probs: list[float], labels: list[int], bins: int = 10
) -> float:
    """Standard 10-bin ECE over P(correct)."""
    if not probs:
        return 0.0
    edges = [i / bins for i in range(bins + 1)]
    total = len(probs)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idxs = [
            i for i, p in enumerate(probs) if (lo <= p < hi) or (hi == 1.0 and p == hi)
        ]
        if not idxs:
            continue
        conf = sum(probs[i] for i in idxs) / len(idxs)
        acc = sum(labels[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / total) * abs(conf - acc)
    return float(ece)


def load_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    """JSONL rows: {"state": ..., "questions": {...}, "labels": {qid: option|bool|int}}."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class Calibrator:
    """Applies fitted Platt parameters to raw distributions.

    Lookup order per question: specific qid params, then type-level params,
    then identity (a=1, b=0).
    """

    params: dict[str, tuple[float, float]]

    @classmethod
    def from_artifact(cls, path: str | Path) -> "Calibrator":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != 1 or data.get("method") != "platt":
            raise ValueError(f"unsupported calibration artifact: {path}")
        params = {k: (float(v["a"]), float(v["b"])) for k, v in data["params"].items()}
        return cls(params=params)

    def _ab(self, qtype: str, qid: str) -> tuple[float, float]:
        if qid in self.params:
            return self.params[qid]
        if qtype in self.params:
            return self.params[qtype]
        return 1.0, 0.0

    def apply_distribution(
        self, dist: dict[str, float], qtype: str, qid: str
    ) -> dict[str, float]:
        a, b = self._ab(qtype, qid)
        if abs(a - 1.0) < 1e-9 and abs(b) < 1e-9:
            return dist
        logits = {k: a * _logit(p) + b for k, p in dist.items()}
        m = max(logits.values())
        exps = {k: math.exp(v - m) for k, v in logits.items()}
        total = sum(exps.values()) or 1.0
        return {k: e / total for k, e in exps.items()}

    def apply_noul(self, p_true: float, qid: str) -> float:
        a, b = self._ab("noul", qid)
        z = a * _logit(p_true) + b
        return 1.0 / (1.0 + math.exp(-z))
