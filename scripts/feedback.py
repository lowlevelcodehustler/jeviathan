#!/usr/bin/env python3
"""Jeviathan decision feedback loop — capture, review, retrain, promote.

The nightly feature that feeds decisions and their outcomes back into the
models Jeviathan works with (shim/PyTorch tier). Ollama/vLLM tiers keep
capturing decisions; `promote` applies to the shim because it is the only
tier that loads HF weights directly.

Workflow:
  1. capture   profile opt-in (feedback.enabled) logs every pass, or use
               `log` to record one manually
  2. review    feedback.py list / show <id> / review <id> --verdict ...
  3. train     feedback.py build-dataset && feedback.py train
  4. promote   feedback.py promote <run-id>   (then: manage.py restart shim)

Examples:
  python scripts/feedback.py status
  python scripts/feedback.py list --pending
  python scripts/feedback.py review d-20260930T142201-a1b2c3 \
      --verdict incorrect --correction-json '{"department": "returns"}'
  python scripts/feedback.py build-dataset
  python scripts/feedback.py train --epochs 3 --max-len 512
  python scripts/feedback.py promote <run-id>
  python scripts/feedback.py promote --revert

Training extras (same venv as the shim): pip install -r requirements-train.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # jeviathan package
sys.path.insert(0, str(Path(__file__).resolve().parent))  # manage.py

from jeviathan.feedback.dataset import build_sft_dataset, export_eval_rows  # noqa: E402
from jeviathan.feedback.store import DecisionLog, VERDICTS  # noqa: E402


# ------------------------------------------------------------------ helpers

def _write_pointer(filename: str, value: str | None) -> Path:
    """Write a one-line pointer file with exact bytes (no shell mangling), or
    delete it when value is None. Control characters are rejected — a mangled
    path here causes obscure downstream crashes."""
    p = REPO / filename
    if value is None:
        if p.exists():
            p.unlink()
        return p
    bad = [c for c in value if ord(c) < 0x20 and c != "\n"]
    if bad:
        raise SystemExit(f"refusing to write {filename}: control characters {bad!r}")
    # Bytes, not text mode: Windows text writes would turn the trailing \n
    # into CRLF and corrupt a path that downstream resolvers read verbatim.
    p.write_bytes((value.rstrip("\n") + "\n").encode("utf-8"))
    return p


def _resolve_base_dir(cli_value: str | None) -> Path:
    """Base weights dir for training: --base > manage.py's shim resolution."""
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    try:
        import manage

        return Path(manage.resolve_shim_model_dir(None)).resolve()
    except SystemExit as exc:
        raise SystemExit(
            f"no base model dir ({exc})\nPass --base <dir> to train against a "
            "specific weights folder."
        ) from None


def _find_run(run_id: str) -> Path | None:
    """Locate finetunes/<base>/<run-id>/ by exact or suffix match."""
    root = REPO / "finetunes"
    if not root.is_dir():
        return None
    matches = [d for d in root.glob("*/*") if d.is_dir() and (d.name == run_id or d.name.endswith(run_id))]
    if len(matches) > 1:
        raise SystemExit(f"run id {run_id!r} is ambiguous: {[m.name for m in matches]}")
    return matches[0] if matches else None


def _print_records(records: list[dict], limit: int, pending_only: bool) -> None:
    rows = [r for r in records if not (pending_only and not (r.get("feedback") or {}).get("verdict"))]
    for rec in rows[-limit:]:
        fb = rec.get("feedback") or {}
        verdict = fb.get("verdict", "pending")
        print(f"{rec['id']}  {rec['ts']}  {rec.get('model', '?'):<24s} {verdict:<10s} "
              f"q={len(rec.get('questions') or {})}")


# ---------------------------------------------------------------- commands

def cmd_status(args) -> int:
    log = DecisionLog()
    stats = log.stats()
    print(f"Jeviathan feedback status @ {REPO}")
    try:
        import manage

        base = manage.resolve_shim_model_dir(None)
    except SystemExit:
        base = "(unset)"
    print(f"  base model dir : {base}")
    adapter_file = REPO / ".jeviathan_adapter_dir"
    if adapter_file.is_file():
        print(f"  active adapter : {adapter_file.read_text(encoding='utf-8').strip()}")
    else:
        print("  active adapter : (none — serving bare base)")
    print(f"  decision log   : {log.path} ({stats['total']} records, "
          f"{stats['pending']} pending review)")
    for v in VERDICTS:
        if stats[v]:
            print(f"                   {v}: {stats[v]}")
    runs_root = REPO / "finetunes"
    if runs_root.is_dir():
        for base_dir in sorted(runs_root.iterdir()):
            if not base_dir.is_dir():
                continue
            for run in sorted(base_dir.iterdir(), reverse=True):
                report = run / "train_report.json"
                best = ""
                if report.is_file():
                    try:
                        r = json.loads(report.read_text(encoding="utf-8"))
                        last = (r.get("history") or [{}])[-1]
                        best = f"eval={last.get('eval_loss', '?')}"
                    except (json.JSONDecodeError, OSError):
                        pass
                print(f"  run            : finetunes/{base_dir.name}/{run.name} {best}")
    return 0


def cmd_list(args) -> int:
    records = DecisionLog().load_all()
    if not records:
        print("no decisions logged yet (enable feedback in your profile, or use `log`)")
        return 0
    _print_records(records, args.limit, args.pending)
    return 0


def cmd_show(args) -> int:
    rec = DecisionLog().get(args.id)
    if rec is None:
        raise SystemExit(f"no decision with id {args.id!r}")
    print(json.dumps(rec, indent=2, ensure_ascii=False))
    return 0


def cmd_log(args) -> int:
    """Record a decision manually (the engine logs automatically when the
    profile has feedback.enabled)."""
    state = args.state_file.read_text(encoding="utf-8") if args.state_file else args.state
    questions = json.loads(args.questions)
    answers = json.loads(args.answers)
    rid = DecisionLog().record(
        profile=args.profile, model=args.model, strategy=args.strategy,
        state=state, questions=questions, raw_output=None, answers=answers,
    )
    print(f"logged {rid} — review it with: feedback.py review {rid} --verdict ...")
    return 0


def cmd_review(args) -> int:
    log = DecisionLog()
    rec = log.get(args.id)
    if rec is None:
        raise SystemExit(f"no decision with id {args.id!r}")
    correction: dict = {}
    raw = args.correction_json or (
        args.correction_file.read_text(encoding="utf-8") if args.correction_file else None
    )
    if raw:
        correction = json.loads(raw)
        known = set((rec.get("questions") or {}).keys())
        unknown = [k for k in correction if k not in known]
        if unknown:
            raise SystemExit(f"correction keys {unknown} are not questions of this record "
                             f"(known: {sorted(known)})")
    log.attach_feedback(args.id, args.verdict, correction or None, args.note)
    print(f"recorded verdict={args.verdict}"
          + (f" corrections={list(correction)}" if correction else "")
          + (f" note={args.note!r}" if args.note else ""))
    print("next: feedback.py build-dataset && feedback.py train")
    return 0


def cmd_build_dataset(args) -> int:
    records = DecisionLog().load_all()
    train_items, eval_items, excluded = build_sft_dataset(
        records, eval_ratio=args.eval_ratio, seed=args.seed
    )
    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for item in train_items:
            fh.write(json.dumps({**item, "eval": False}, ensure_ascii=False) + "\n")
        for item in eval_items:
            fh.write(json.dumps({**item, "eval": True}, ensure_ascii=False) + "\n")
    print(f"wrote {len(train_items)} train + {len(eval_items)} eval items -> {out}")
    if excluded:
        print(f"excluded: {excluded}")
    if not (train_items or eval_items):
        print("nothing usable yet — review decisions with verdicts/corrections first")
        return 1
    return 0


def cmd_train(args) -> int:
    from jeviathan.feedback.trainer import qlora_train

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = REPO / args.dataset
    if not dataset_path.is_file():
        raise SystemExit(f"dataset not found: {dataset_path} — run `build-dataset` first")
    items = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_items = [it for it in items if not it.get("eval")]
    eval_items = [it for it in items if it.get("eval")]

    base_dir = _resolve_base_dir(args.base)
    base_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(base_dir).name or "base")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = REPO / "finetunes" / base_name / run_id

    qlora_train(
        base_dir, train_items, eval_items, out_dir,
        rank=args.rank, alpha=args.alpha, lr=args.lr, epochs=args.epochs,
        max_len=args.max_len, grad_accum=args.grad_accum, seed=args.seed,
        max_steps=args.max_steps,
    )
    print(f"\nrun id: {run_id}")
    print("promote with:  python scripts/feedback.py promote " + run_id)
    return 0


def cmd_promote(args) -> int:
    if args.revert:
        _write_pointer(".jeviathan_adapter_dir", None)
        print("adapter pointer cleared — the shim will serve the bare base after restart")
        print("restart with: python scripts/manage.py restart shim")
        return 0

    run = _find_run(args.run_id)
    if run is None:
        raise SystemExit(f"no finetunes run matching {args.run_id!r} (see `status`)")
    adapter_dir = run / "adapter"
    if not adapter_dir.is_dir():
        raise SystemExit(f"{run} has no adapter/ dir — retrain first")

    target = str(adapter_dir.resolve())
    _write_pointer(".jeviathan_adapter_dir", target)
    print(f"promoted {run.name}: .jeviathan_adapter_dir -> {target}")
    if not args.yes:
        print("restart the shim to serve it:")
        print("  python scripts/manage.py restart shim")
        print("(rollback anytime: feedback.py promote --revert)")
    return 0


def cmd_export_eval(args) -> int:
    records = DecisionLog().load_all()
    rows = export_eval_rows(records)
    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} labeled eval rows -> {out}")
    if rows:
        print("refit calibration on the promoted model:")
        print(f"  python scripts/manage.py fit --data {args.out} "
              f"--profile <your-profile> --out calibration/<name>.json")
    return 0


# -------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status", help="model dir, active adapter, log stats, runs")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("list", help="list logged decisions")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--pending", action="store_true", help="only unreviewed records")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="print one record in full")
    p.add_argument("id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("log", help="record a decision manually")
    p.add_argument("--state", default=None, help="state text (or --state-file)")
    p.add_argument("--state-file", type=Path, default=None)
    p.add_argument("--questions", required=True, help='JSON: {"qid": {question dict}}')
    p.add_argument("--answers", required=True, help='JSON: {"qid": {answer dict}}')
    p.add_argument("--profile", default="manual")
    p.add_argument("--model", default="manual")
    p.add_argument("--strategy", default="one_shot")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("review", help="attach a verdict/correction to a record")
    p.add_argument("id")
    p.add_argument("--verdict", required=True, choices=VERDICTS)
    p.add_argument("--correction-json", default=None,
                   help='JSON: {"qid": <option name | 0..1 | level | {"probabilities": {...}}>}')
    p.add_argument("--correction-file", type=Path, default=None)
    p.add_argument("--note", default=None)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("build-dataset", help="verified records -> SFT chat pairs")
    p.add_argument("--out", default="data/sft.jsonl")
    p.add_argument("--eval-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser("train", help="QLoRA fine-tune the base model (needs GPU + peft)")
    p.add_argument("--dataset", default="data/sft.jsonl")
    p.add_argument("--base", default=None, help="weights dir (default: shim's resolved dir)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=None, help="default: 2 x rank")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0, help="0 = run all epochs")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("promote", help="point the shim at a run's adapter (or --revert)")
    p.add_argument("run_id", nargs="?", default=None)
    p.add_argument("--revert", action="store_true", help="clear the active adapter")
    p.add_argument("--yes", action="store_true", help="suppress restart instructions")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("export-eval", help="verified records -> calibration-fit rows")
    p.add_argument("--out", default="evals/feedback_eval.jsonl")
    p.set_defaults(func=cmd_export_eval)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
