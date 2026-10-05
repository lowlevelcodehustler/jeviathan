"""Append-only decision log for the feedback loop.

Two files, both append-only (no line rewrites, safe under concurrent API +
CLI access):

  data/decisions.jsonl   one JSON line per System One pass (engine opt-in)
  data/feedback.jsonl    one JSON line per review: {id, verdict, correction, note}

The decisions file is never mutated; reviews attach by id and the latest
entry wins. Path override: $JEVIATHAN_DECISION_LOG (the sidecar lives next to
it). Records are small (~1-2 KB) so dev-scale logs stay trivially readable.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # jeviathan/feedback/store.py -> repo root
SCHEMA = 1
VERDICTS = ("correct", "incorrect", "partial")


def default_log_path() -> Path:
    env = os.environ.get("JEVIATHAN_DECISION_LOG")
    if env:
        return Path(env)
    return REPO_ROOT / "data" / "decisions.jsonl"


class DecisionLog:
    """Thread-safe append-only decision log + feedback sidecar."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_log_path()
        self.feedback_path = self.path.parent / "feedback.jsonl"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ write

    def record(
        self,
        *,
        profile: str,
        model: str,
        strategy: str,
        state: object,
        questions: dict[str, dict],
        raw_output: str | None,
        answers: dict[str, dict],
        usage: dict | None = None,
    ) -> str:
        """Append one decision pass. Returns the new record id."""
        now = time.time()
        rid = f"d-{time.strftime('%Y%m%dT%H%M%S', time.localtime(now))}-{secrets.token_hex(3)}"
        line = {
            "schema": SCHEMA,
            "id": rid,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
            + f".{int((now % 1) * 1e6):06d}",
            "profile": profile,
            "model": model,
            "strategy": strategy,
            "state": state,
            "questions": questions,
            "raw_output": raw_output,
            "answers": answers,
            "usage": usage,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # newline="" keeps LF line endings on Windows (byte-stable JSONL).
            with open(self.path, "a", encoding="utf-8", newline="") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        return rid

    def attach_feedback(
        self,
        decision_id: str,
        verdict: str,
        correction: dict | None = None,
        note: str | None = None,
    ) -> None:
        """Append a review for `decision_id` (last entry wins on read)."""
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
        now = time.time()
        line = {
            "id": decision_id,
            "verdict": verdict,
            "correction": correction or {},
            "note": note,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
            + f".{int((now % 1) * 1e6):06d}",
        }
        with self._lock:
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            # newline="" keeps LF line endings on Windows (byte-stable JSONL).
            with open(self.feedback_path, "a", encoding="utf-8", newline="") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------- read

    def load_all(self) -> list[dict]:
        """All decisions with the latest matching feedback merged in."""
        records = _read_jsonl(self.path)
        if not records:
            return []
        by_id: dict[str, list[dict]] = {}
        for fb in _read_jsonl(self.feedback_path):
            by_id.setdefault(fb.get("id"), []).append(fb)
        for rec in records:
            entries = by_id.get(rec["id"])
            if entries:
                # Latest ts wins; on a tie the later file entry wins (reviews
                # are append-only, so file order is the true sequence).
                latest = None
                for e in entries:
                    if latest is None or e.get("ts", "") >= latest.get("ts", ""):
                        latest = e
                rec["feedback"] = {
                    "verdict": latest.get("verdict"),
                    "correction": latest.get("correction") or {},
                    "note": latest.get("note"),
                    "ts": latest.get("ts"),
                }
        return records

    def get(self, decision_id: str) -> dict | None:
        for rec in self.load_all():
            if rec["id"] == decision_id:
                return rec
        return None

    def stats(self) -> dict:
        counts = {"total": 0, "pending": 0, **{v: 0 for v in VERDICTS}}
        for rec in self.load_all():
            counts["total"] += 1
            verdict = (rec.get("feedback") or {}).get("verdict")
            if verdict is None:
                counts["pending"] += 1
            else:
                counts[verdict] += 1
        return counts


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.is_file():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn write from a crash; skip, keep the rest
            if isinstance(obj, dict):
                out.append(obj)
    return out
