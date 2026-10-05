"""Build SFT chat pairs from logged decisions + reviewer feedback.

Prompt side: records are re-validated into a SystemOneRequest and compiled
with the SAME compiler the engine uses at inference (compile_request), so
training and serving see byte-identical prompts — no template drift.

Target side: the canonical JSON distribution object the SYSTEM_PROMPT contract
demands. Per question, in priority order:
  1. explicit reviewer correction (probabilities dict, or an answer value that
     becomes a one-hot over the expected options);
  2. verdict == "correct" -> the model's own answer distribution (reinforces
     verified behaviour; this is also the anti-forgetting anchor).

Usability rule (v1): a record trains only if EVERY question resolves to a
target above. Anything else is excluded and counted — we never teach the
model an answer nobody verified. Residual miscalibration from one-hot targets
is absorbed by the existing calibration layer: refit after `promote`
(`manage.py fit`, see docs/user-guide.md).
"""

from __future__ import annotations

import json
import random

from ..compiler.prompt_compiler import compile_request
from ..schemas.typesafe import SystemOneRequest


def expected_options(q: dict) -> list[str]:
    """Option/level names for a question dict (mirrors schemas.level_names())."""
    t = q.get("type")
    if t == "choice":
        return list((q.get("criteria") or {}).keys())
    if t == "score":
        levels = q.get("levels") or []
        return [(lv.get("name") or str(i)) for i, lv in enumerate(levels)]
    return ["true", "false"]  # noul


def _renormalize(dist: dict[str, float]) -> dict[str, float] | None:
    clamped = {k: max(0.0, min(1.0, float(v))) for k, v in dist.items()}
    total = sum(clamped.values())
    if total <= 0:
        return None
    return {k: round(v / total, 6) for k, v in clamped.items()}


def correction_to_distribution(qid: str, q: dict, value: object) -> dict[str, float] | None:
    """Reviewer correction -> distribution over the question's options.

    Accepted shapes per type:
      any:        {"probabilities": {...}}  (explicit; renormalized)
      choice:     "option_name" | {"choice": "option_name"}   -> one-hot
      noul:       0..1 number | "true"/"false" | {"noul": x}  -> {true, false}
      score:      level name or index | {"score": x}          -> one-hot level
    Returns None when the correction is unparseable for this question.
    """
    options = expected_options(q)
    if isinstance(value, dict):
        if "probabilities" in value and isinstance(value["probabilities"], dict):
            dist = {name: float(value["probabilities"].get(name, 0.0)) for name in options}
            return _renormalize(dist)
        if q.get("type") == "choice":
            value = value.get("choice", value)
        elif q.get("type") == "noul":
            value = value.get("noul", value)
        else:  # score
            value = value.get("score", value)

    t = q.get("type")
    if t == "choice":
        name = str(value)
        return {o: (1.0 if o == name else 0.0) for o in options} if name in options else None
    if t == "noul":
        if isinstance(value, str):
            if value not in ("true", "false"):
                return None
            p = 1.0 if value == "true" else 0.0
        elif isinstance(value, (int, float)):
            p = max(0.0, min(1.0, float(value)))
        else:
            return None
        return {"true": round(p, 6), "false": round(1.0 - p, 6)}
    # score
    if isinstance(value, str) and value in options:
        idx = options.index(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        idx = int(round(float(value)))
        if not 0 <= idx < len(options):
            return None
    else:
        return None
    return {o: (1.0 if i == idx else 0.0) for i, o in enumerate(options)}


def record_target(record: dict, qid: str) -> dict[str, float] | None:
    """Target distribution for one question of a reviewed record, or None."""
    fb = record.get("feedback") or {}
    verdict = fb.get("verdict")
    correction = (fb.get("correction") or {}).get(qid)

    if correction is not None:
        return correction_to_distribution(qid, record["questions"][qid], correction)

    if verdict == "correct":
        ans = (record.get("answers") or {}).get(qid)
        if not isinstance(ans, dict):
            return None
        if ans.get("type") == "noul":
            p = max(0.0, min(1.0, float(ans.get("noul", 0.5))))
            return {"true": round(p, 6), "false": round(1.0 - p, 6)}
        dist = _renormalize({k: v for k, v in (ans.get("probabilities") or {}).items()})
        if dist is None:
            return None
        # Re-key over the question's options so the target always matches the
        # contract shape exactly.
        return {name: dist.get(name, 0.0) for name in expected_options(record["questions"][qid])}

    return None  # unverified question


def build_sft_dataset(
    records: list[dict], eval_ratio: float = 0.15, seed: int = 42
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """-> (train_items, eval_items, excluded_counts).

    Each item: {"decision_id", "messages": [system, user, assistant]}. The
    assistant content is the exact JSON object SYSTEM_PROMPT demands.
    """
    usable: list[tuple[dict, dict[str, dict[str, float]]]] = []
    excluded = {"no_feedback": 0, "unverified_question": 0, "bad_correction": 0}

    for rec in records:
        fb = rec.get("feedback") or {}
        if not fb.get("verdict"):
            excluded["no_feedback"] += 1
            continue
        targets: dict[str, dict[str, float]] = {}
        dropped_reason: str | None = None
        for qid, q in (rec.get("questions") or {}).items():
            target = record_target(rec, qid)
            if target is None:
                has_corr = (fb.get("correction") or {}).get(qid) is not None
                dropped_reason = "bad_correction" if has_corr else "unverified_question"
                break
            targets[qid] = target
        if dropped_reason:
            excluded[dropped_reason] += 1
            continue
        usable.append((rec, targets))

    rng = random.Random(seed)
    rng.shuffle(usable)
    n_eval = int(round(len(usable) * eval_ratio)) if len(usable) >= 2 else 0
    n_eval = min(n_eval, max(0, len(usable) - 1))

    def _item(rec: dict, targets: dict[str, dict[str, float]]) -> dict:
        req = SystemOneRequest(state=rec["state"], questions=dict(rec["questions"]))
        system_prompt, user_message = compile_request(req)
        target_json = json.dumps(
            {"answers": {qid: {"probabilities": t} for qid, t in targets.items()}},
            ensure_ascii=False,
        )
        return {
            "decision_id": rec["id"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": target_json},
            ],
        }

    eval_items = [_item(rec, targets) for rec, targets in usable[:n_eval]]
    train_items = [_item(rec, targets) for rec, targets in usable[n_eval:]]
    return train_items, eval_items, excluded


def export_eval_rows(records: list[dict]) -> list[dict]:
    """Verified records -> calibration-fit rows {state, questions, labels}.

    Labels are the argmax of each question's target distribution (the same
    usability rule as build_sft_dataset), so `manage.py fit` can refit Platt
    parameters on the fine-tuned model without any new model passes.
    """
    rows: list[dict] = []
    for rec, targets in _verified_records(records):
        labels = {qid: max(t.items(), key=lambda kv: kv[1])[0] for qid, t in targets.items()}
        rows.append({"state": rec["state"], "questions": dict(rec["questions"]), "labels": labels})
    return rows


def _verified_records(records: list[dict]) -> list[tuple[dict, dict[str, dict[str, float]]]]:
    out = []
    for rec in records:
        fb = rec.get("feedback") or {}
        if not fb.get("verdict"):
            continue
        targets: dict[str, dict[str, float]] = {}
        ok = True
        for qid, q in (rec.get("questions") or {}).items():
            target = record_target(rec, qid)
            if target is None:
                ok = False
                break
            targets[qid] = target
        if ok:
            out.append((rec, targets))
    return out
