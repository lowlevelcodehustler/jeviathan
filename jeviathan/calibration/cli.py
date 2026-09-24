"""Calibration CLI — fit Platt parameters from a labeled eval set.

Usage:
    python -m jeviathan.calibration.cli fit \
        --data evals/sample_eval.jsonl \
        --profile rtx5090 \
        --out calibration/calib.json

Flow: run each eval row through the engine with calibration disabled,
collect raw P(correct) per question id, pool by type when a single qid has
fewer than MIN_SAMPLES rows, fit Platt (a,b), write artifact, and print
ECE before/after.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..calibration.calibrator import (
    expected_calibration_error,
    fit_platt,
    load_eval_rows,
)
from ..config import active_profile, load_profile
from ..engine.systemone_engine import SystemOneEngine
from ..schemas.typesafe import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

MIN_SAMPLES_PER_QID = 20


async def _collect(engine: SystemOneEngine, rows: list[dict]) -> dict[str, list[tuple[float, int]]]:
    """qid -> [(raw P(correct), label)] across all eval rows."""
    collected: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        req = SystemOneRequest(state=row["state"], questions=row["questions"])
        resp = await engine.system_one(req)
        labels = row.get("labels") or {}
        for qid, answer in resp.answers.items():
            if qid not in labels:
                continue
            label = labels[qid]
            if answer.type == "choice":
                p = answer.probabilities.get(str(label), 0.0)
                collected[qid].append((p, 1 if answer.choice == str(label) else 0))
            elif answer.type == "score":
                # label is the expected level index (int or numeric string)
                target = float(label)
                names = [str(i) for i in range(len(answer.probabilities))]
                p = sum(
                    prob * (1.0 - min(abs(idx - target), 2.0) / 2.0)
                    for idx, (name, prob) in enumerate(zip(names, answer.probabilities.values()))
                )
                collected[qid].append((p, 1 if abs(answer.score - target) <= 0.5 else 0))
            elif answer.type == "noul":
                want = bool(label)
                p = answer.noul if want else 1.0 - answer.noul
                collected[qid].append((p, 1 if (answer.noul >= 0.5) == want else 0))
    return dict(collected)


async def _run(args: argparse.Namespace) -> None:
    profile = load_profile(args.profile)
    # Force calibration off while collecting raw distributions.
    profile.calibration.enabled = False
    engine = SystemOneEngine(profile)

    rows = load_eval_rows(args.data)
    print(f"Collected {len(rows)} eval rows from {args.data}")
    collected = await _collect(engine, rows)

    # Question types per qid (from first row that contains it).
    qtype_by_qid: dict[str, str] = {}
    for row in rows:
        for qid, q in row["questions"].items():
            if isinstance(q, ChoiceQuestion):
                qtype_by_qid.setdefault(qid, "choice")
            elif isinstance(q, ScoreQuestion):
                qtype_by_qid.setdefault(qid, "score")
            else:
                qtype_by_qid.setdefault(qid, "noul")

    params: dict[str, dict[str, float]] = {}
    report_lines = []
    for qtype in ("choice", "score", "noul"):
        pooled_p, pooled_y = [], []
        for qid, pairs in collected.items():
            if qtype_by_qid.get(qid) != qtype:
                continue
            ps = [p for p, _ in pairs]
            ys = [y for _, y in pairs]
            pooled_p.extend(ps)
            pooled_y.extend(ys)
        a, b = fit_platt(pooled_p, pooled_y)
        params[qtype] = {"a": round(a, 6), "b": round(b, 6)}

        # Per-qid overrides where we have enough data.
        for qid, pairs in collected.items():
            if qtype_by_qid.get(qid) != qtype or len(pairs) < MIN_SAMPLES_PER_QID:
                continue
            qa, qb = fit_platt([p for p, _ in pairs], [y for _, y in pairs])
            params[qid] = {"a": round(qa, 6), "b": round(qb, 6)}

        ece_raw = expected_calibration_error(pooled_p, pooled_y)
        # ECE after applying type-level Platt.
        import math as _m

        def _apply(p: float) -> float:
            eps = 1e-6
            p = min(max(p, eps), 1 - eps)
            z = a * _m.log(p / (1 - p)) + b
            return 1.0 / (1.0 + _m.exp(-z))

        ece_cal = expected_calibration_error([_apply(p) for p in pooled_p], pooled_y)
        report_lines.append(
            f"{qtype:7s} n={len(pooled_p):5d}  a={a:+.4f} b={b:+.4f}   "
            f"ECE raw={ece_raw:.3f} -> calibrated={ece_cal:.3f}"
        )

    artifact = {
        "version": 1,
        "method": "platt",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile.name,
        "model": profile.backend.model,
        "params": params,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)

    print("\n".join(report_lines))
    print(f"\nWrote calibration artifact -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit_p = sub.add_parser("fit", help="Fit Platt parameters from a labeled eval set")
    fit_p.add_argument("--data", required=True, help="Path to JSONL eval file")
    fit_p.add_argument("--profile", default=None, help="Profile name (default: active)")
    fit_p.add_argument(
        "--out", default="calibration/calib.json", help="Artifact output path"
    )
    args = parser.parse_args()
    if args.cmd == "fit":
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
