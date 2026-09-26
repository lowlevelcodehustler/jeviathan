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
from pydantic import TypeAdapter

from ..schemas.typesafe import Answer, SystemOneRequest

MIN_SAMPLES_PER_QID = 20


def _row_pairs(
    answers: dict,
    labels: dict,
) -> tuple[dict[str, list[tuple[float, int]]], dict[str, str]]:
    """Extract (p_correct, label) pairs + question types from one validated response."""
    collected: dict[str, list[tuple[float, int]]] = defaultdict(list)
    qtypes: dict[str, str] = {}
    for qid, answer in answers.items():
        if qid not in labels:
            continue
        label = labels[qid]
        qtypes.setdefault(qid, answer.type)
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
    return dict(collected), qtypes


async def _collect(
    engine: SystemOneEngine,
    rows: list[dict],
    raw_out: str | None = None,
    resume: bool = False,
) -> tuple[dict[str, list[tuple[float, int]]], dict[str, str]]:
    """Run every eval row through the engine; return (pairs by qid, type by qid).

    With raw_out set, each row's full response is appended to a JSONL file so
    later refits can reuse it via --from-raw without another model pass.
    With resume set, rows whose state already appears in the raw file are
    skipped (crash recovery — the laptop GPU takes minutes per row).
    """
    collected: dict[str, list[tuple[float, int]]] = defaultdict(list)
    qtype_by_qid: dict[str, str] = {}
    skip_states: set[str] = set()
    if resume and raw_out and Path(raw_out).exists():
        with open(raw_out, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    skip_states.add(json.loads(line)["state"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(skip_states)} rows already in {raw_out}", flush=True)
    raw_fh = open(raw_out, "a", encoding="utf-8") if raw_out else None
    try:
        for i, row in enumerate(rows):
            if resume and row["state"] in skip_states:
                print(f"  row {i + 1}/{len(rows)} skipped (already collected)", flush=True)
                continue
            req = SystemOneRequest(state=row["state"], questions=row["questions"])
            resp = await engine.system_one(req)
            labels = row.get("labels") or {}
            pairs, qtypes = _row_pairs(resp.answers, labels)
            for qid, plist in pairs.items():
                collected[qid].extend(plist)
            for qid, qt in qtypes.items():
                qtype_by_qid.setdefault(qid, qt)
            if raw_fh:
                raw_fh.write(
                    json.dumps(
                        {
                            "state": row["state"],
                            "questions": row["questions"],
                            "labels": labels,
                            "answers": {k: v.model_dump() for k, v in resp.answers.items()},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw_fh.flush()
            print(f"  row {i + 1}/{len(rows)} done", flush=True)
    finally:
        if raw_fh:
            raw_fh.close()
    return dict(collected), qtype_by_qid


def _collect_from_raw(raw_path: str) -> tuple[dict[str, list[tuple[float, int]]], dict[str, str]]:
    """Rebuild pairs from a previously dumped raw file (no model calls)."""
    collected: dict[str, list[tuple[float, int]]] = defaultdict(list)
    qtype_by_qid: dict[str, str] = {}
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            answer_adapter = TypeAdapter(Answer)
            answers = {qid: answer_adapter.validate_python(a) for qid, a in row["answers"].items()}
            pairs, qtypes = _row_pairs(answers, row.get("labels") or {})
            for qid, plist in pairs.items():
                collected[qid].extend(plist)
            for qid, qt in qtypes.items():
                qtype_by_qid.setdefault(qid, qt)
    return dict(collected), qtype_by_qid


async def _run(args: argparse.Namespace) -> None:
    profile = load_profile(args.profile)
    # Force calibration off while collecting raw distributions.
    profile.calibration.enabled = False
    engine = SystemOneEngine(profile)

    if args.from_raw:
        print(f"Refitting from raw dump: {args.from_raw}")
        collected, qtype_by_qid = _collect_from_raw(args.from_raw)
    else:
        rows = load_eval_rows(args.data)
        print(f"Collected {len(rows)} eval rows from {args.data}")
        raw_out = args.raw_out or f"calibration/raw-{profile.name}.jsonl"
        collected, qtype_by_qid = await _collect(
            engine, rows, raw_out=raw_out, resume=args.resume
        )

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
    fit_p.add_argument("--data", help="Path to JSONL eval file")
    fit_p.add_argument(
        "--from-raw",
        help="Refit from a previously dumped raw responses file (no model calls)",
    )
    fit_p.add_argument(
        "--raw-out",
        default=None,
        help="Where to append raw responses (default: calibration/raw-<profile>.jsonl)",
    )
    fit_p.add_argument(
        "--resume",
        action="store_true",
        help="Skip eval rows already present in the raw-out file (crash recovery)",
    )
    fit_p.add_argument("--profile", default=None, help="Profile name (default: active)")
    fit_p.add_argument(
        "--out", default="calibration/calib.json", help="Artifact output path"
    )
    args = parser.parse_args()
    if args.cmd == "fit":
        if not args.from_raw and not args.data:
            fit_p.error("one of --data or --from-raw is required")
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
