"""Decision feedback loop tests — no GPU required (store/dataset/CLI/hooks)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from jeviathan.config import FeedbackConfig, Profile  # noqa: E402
from jeviathan.engine.systemone_engine import SystemOneEngine  # noqa: E402
from jeviathan.feedback.dataset import (  # noqa: E402
    build_sft_dataset,
    correction_to_distribution,
    export_eval_rows,
)
from jeviathan.feedback.store import DecisionLog  # noqa: E402

import feedback as fb_cli  # noqa: E402
from tests.helpers import MockBackend, mock_profile, mock_text, sample_request  # noqa: E402


# ------------------------------------------------------------------ fixtures

def make_record(rid="d-test-1", verdict=None, correction=None) -> dict:
    return {
        "schema": 1,
        "id": rid,
        "ts": "2026-09-30T10:00:00.000000",
        "profile": "test",
        "model": "mock-model",
        "strategy": "one_shot",
        "state": "Customer ticket: 'My card was charged twice.'",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": None, "returns": None},
            },
            "urgency": {"type": "noul", "instructions": "Is this ticket urgent?"},
        },
        "raw_output": mock_text(),
        "answers": {
            "department": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.78,
                "probabilities": {"billing": 0.78, "returns": 0.22},
            },
            "urgency": {"type": "noul", "noul": 0.62},
        },
        "usage": None,
        "feedback": (
            {"verdict": verdict, "correction": correction or {}, "note": None}
            if verdict
            else None
        ),
    }


# -------------------------------------------------------------------- store

def test_store_roundtrip_and_last_review_wins(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    rid = log.record(
        profile="test", model="m", strategy="one_shot", state="s",
        questions={"q": {"type": "noul"}}, raw_output=None, answers={},
    )
    assert rid.startswith("d-")

    # Two reviews of the same id: the later one must win.
    log.attach_feedback(rid, "incorrect", correction={"q": 0.9})
    log.attach_feedback(rid, "correct")
    rec = log.get(rid)
    assert rec["feedback"]["verdict"] == "correct"

    stats = log.stats()
    assert stats["total"] == 1 and stats["correct"] == 1 and stats["pending"] == 0


def test_store_rejects_bad_verdict(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    rid = log.record(
        profile="t", model="m", strategy="one_shot", state="s",
        questions={}, raw_output=None, answers={},
    )
    with pytest.raises(ValueError):
        log.attach_feedback(rid, "maybe")


# ------------------------------------------------------------------ dataset

def test_dataset_usability_rules():
    records = [
        make_record("d-1", verdict="correct"),                                   # usable: reinforce
        make_record("d-2", verdict="incorrect",                                  # usable: both corrected
                    correction={"department": "returns", "urgency": 0.9}),
        make_record("d-3", verdict="partial", correction={"department": "returns"}),  # urgency unverified
        make_record("d-4"),                                                      # pending
    ]
    train, ev, excluded = build_sft_dataset(records, eval_ratio=0.0)
    assert len(train) == 2 and not ev
    assert excluded["unverified_question"] == 1 and excluded["no_feedback"] == 1

    by_id = {item["decision_id"]: item for item in train}
    # verdict=correct -> the model's own distributions are reinforced.
    target = json.loads(by_id["d-1"]["messages"][2]["content"])["answers"]
    assert target["department"]["probabilities"] == {"billing": 0.78, "returns": 0.22}
    assert target["urgency"]["probabilities"] == {"true": 0.62, "false": 0.38}
    # corrections -> one-hot / explicit values over the expected options.
    target = json.loads(by_id["d-2"]["messages"][2]["content"])["answers"]
    assert target["department"]["probabilities"] == {"billing": 0.0, "returns": 1.0}
    assert target["urgency"]["probabilities"] == {"true": 0.9, "false": 0.1}


def test_dataset_target_matches_contract_shape():
    train, _, _ = build_sft_dataset([make_record("d-1", verdict="correct")], eval_ratio=0.0)
    item = train[0]
    assert [m["role"] for m in item["messages"]] == ["system", "user", "assistant"]
    payload = json.loads(item["messages"][2]["content"])
    answers = payload["answers"]
    # Exact contract: every question id, each with a probabilities object whose
    # keys are exactly the expected options and which sums to 1.
    assert set(answers) == {"department", "urgency"}
    for qid, entry in answers.items():
        probs = entry["probabilities"]
        assert (set(probs) == {"billing", "returns"}) if qid == "department" else (set(probs) == {"true", "false"})
        assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_explicit_probability_corrections_clamped_and_renormalized():
    q = make_record()["questions"]["department"]
    dist = correction_to_distribution("department", q, {"probabilities": {"billing": 2.0, "returns": -1.0}})
    assert dist == {"billing": 1.0, "returns": 0.0}
    # Unknown option name -> unparseable -> None (record gets excluded).
    assert correction_to_distribution("department", q, "refunds") is None


def test_export_eval_rows_labels():
    rows = export_eval_rows([make_record("d-2", verdict="incorrect",
                                         correction={"department": "returns", "urgency": 0.9})])
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {"state", "questions", "labels"}
    assert row["labels"] == {"department": "returns", "urgency": "true"}


# ------------------------------------------------------- promote pointer file

def test_write_pointer_exact_bytes_and_revert(tmp_path, monkeypatch):
    monkeypatch.setattr(fb_cli, "REPO", tmp_path)
    p = fb_cli._write_pointer(".jeviathan_adapter_dir", str(tmp_path / "run1" / "adapter"))
    assert p.read_text(encoding="utf-8") == f"{tmp_path / 'run1' / 'adapter'}\n"
    raw = p.read_bytes()
    assert all(b >= 0x20 or b == 0x0A for b in raw), "no control bytes allowed"

    fb_cli._write_pointer(".jeviathan_adapter_dir", None)
    assert not p.exists()


def test_write_pointer_rejects_control_chars(tmp_path, monkeypatch):
    monkeypatch.setattr(fb_cli, "REPO", tmp_path)
    with pytest.raises(SystemExit):
        fb_cli._write_pointer(".jeviathan_adapter_dir", "bad\tpath")


# ------------------------------------------------------------- engine hook

async def test_engine_hook_records_when_enabled(tmp_path, monkeypatch):
    log_file = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("JEVIATHAN_DECISION_LOG", str(log_file))
    profile = mock_profile()
    profile.feedback = FeedbackConfig(enabled=True)
    engine = SystemOneEngine(profile, backend=MockBackend(mock_text()))

    resp = await engine.system_one(sample_request())
    assert "department" in resp.answers

    records = DecisionLog(log_file).load_all()
    assert len(records) == 1
    rec = records[0]
    assert set(rec["questions"]) == {"department", "urgency", "frustration"}
    assert rec["answers"]["department"]["choice"] == "billing"
    assert rec.get("feedback") is None


async def test_engine_hook_silent_when_disabled(tmp_path, monkeypatch):
    log_file = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("JEVIATHAN_DECISION_LOG", str(log_file))
    profile = mock_profile()  # feedback disabled by default
    engine = SystemOneEngine(profile, backend=MockBackend(mock_text()))

    await engine.system_one(sample_request())
    assert not log_file.exists(), "disabled profiles must not create the log"


async def test_engine_hook_never_breaks_decisions(tmp_path, monkeypatch):
    """A logging failure (e.g. unwritable dir) must not fail the decision."""
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")  # a FILE where mkdir(parents=True) lands
    monkeypatch.setenv("JEVIATHAN_DECISION_LOG", str(blocker / "sub.jsonl"))
    profile = mock_profile()
    profile.feedback = FeedbackConfig(enabled=True)
    engine = SystemOneEngine(profile, backend=MockBackend(mock_text()))

    resp = await engine.system_one(sample_request())
    assert "department" in resp.answers


# ------------------------------------------------------------------- config

def test_profile_feedback_block():
    from jeviathan.config import _build

    p = _build({"feedback": {"enabled": True}}, "t")
    assert p.feedback.enabled is True
    assert _build({}, "t").feedback.enabled is False
