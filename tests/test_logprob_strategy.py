"""v1.1 logprob strategy: dispatch, fallback-on-None, renormalization (no GPU)."""

from __future__ import annotations

import json
import math
import re

import pytest

from jeviathan.backends.base import CompletionResult, DecisionBackend
from jeviathan.compiler.prompt_compiler import (
    SCORING_SYSTEM_PROMPT,
    compile_scoring_user_message,
    expected_options,
)
from jeviathan.engine.systemone_engine import SystemOneEngine
from tests.helpers import mock_profile, mock_text, sample_request


class LogprobMockBackend(DecisionBackend):
    """score_prefixes returns per-question raw scores (or None = unsupported).

    normalize=True mimics the real backend (softmax over mean logprobs);
    normalize=False hands the engine unnormalized mass to exercise its
    defensive renormalization.
    """

    def __init__(
        self,
        scores_by_qid: dict[str, dict[str, float]] | None,
        normalize: bool = True,
    ) -> None:
        self.scores_by_qid = scores_by_qid
        self.normalize = normalize
        self.score_calls: list[tuple[str, str, list[str]]] = []
        self.complete_calls = 0

    async def score_prefixes(self, system_prompt, user_message, prefixes):
        m = re.search(r'"id":\s*"([^"]+)"', user_message)
        qid = m.group(1) if m else "?"
        self.score_calls.append((system_prompt, user_message, list(prefixes)))
        if self.scores_by_qid is None:
            return None
        raw = {p: float(self.scores_by_qid.get(qid, {}).get(p, 0.0)) for p in prefixes}
        if not self.normalize:
            return raw
        mx = max(raw.values())
        exps = {k: math.exp(v - mx) for k, v in raw.items()}
        total = sum(exps.values()) or 1.0
        return {k: e / total for k, e in exps.items()}

    async def complete(self, system_prompt, user_message):
        self.complete_calls += 1
        return CompletionResult(text=mock_text(), input_tokens=100, output_tokens=50)


def _logprob_profile():
    profile = mock_profile()
    profile.sampling.strategy = "logprob"
    return profile


async def test_logprob_strategy_dispatches_to_score_prefixes():
    scores = {
        "department": {"billing": -0.5, "returns": -2.0},
        "urgency": {"true": -0.2, "false": -1.2},
        "frustration": {"calm": -2.0, "frustrated": -0.5, "very_frustrated": -1.0},
    }
    backend = LogprobMockBackend(scores)
    engine = SystemOneEngine(_logprob_profile(), backend=backend)
    resp = await engine.system_one(sample_request())

    # One scoring pass per question; no one_shot completion needed.
    assert len(backend.score_calls) == 3
    assert backend.complete_calls == 0
    for sp, um, prefixes in backend.score_calls:
        assert sp == SCORING_SYSTEM_PROMPT
        assert um.endswith("ANSWER:")

    # Expected distributions (softmax over the mock's mean logprobs).
    def _soft(raw: dict[str, float]) -> dict[str, float]:
        mx = max(raw.values())
        exps = {k: math.exp(v - mx) for k, v in raw.items()}
        total = sum(exps.values()) or 1.0
        return {k: e / total for k, e in exps.items()}

    dept_p = _soft(scores["department"])
    urg_p = _soft(scores["urgency"])
    fru_p = _soft(scores["frustration"])

    dept = resp.answers["department"]
    assert dept.choice == "billing"
    assert dept.probabilities["billing"] == pytest.approx(dept_p["billing"], abs=1e-6)
    # TypeSafe formula: (n*peak - 1)/(n - 1)
    assert dept.confidence == pytest.approx(2 * dept_p["billing"] - 1, abs=1e-6)

    urg = resp.answers["urgency"]
    assert urg.noul == pytest.approx(urg_p["true"], abs=1e-6)

    frust = resp.answers["frustration"]
    assert frust.probabilities["frustrated"] == pytest.approx(fru_p["frustrated"], abs=1e-6)
    expected_score = fru_p["calm"] * 0 + fru_p["frustrated"] * 1 + fru_p["very_frustrated"] * 2
    assert frust.score == pytest.approx(expected_score, abs=1e-4)

    # Logprob path skips the completion pass entirely -> no usage accounting.
    assert resp.usage.input_tokens is None
    assert resp.usage.output_tokens is None


async def test_logprob_fallback_when_backend_unsupported():
    backend = LogprobMockBackend(None)  # score_prefixes -> None
    engine = SystemOneEngine(_logprob_profile(), backend=backend)
    resp = await engine.system_one(sample_request())

    # Tried logprob (first question), then fell back to one_shot.
    assert len(backend.score_calls) == 1
    assert backend.complete_calls >= 1
    # Answers come from the parsed mock_text() JSON, not from scores.
    assert resp.answers["department"].choice == "billing"
    assert resp.usage.input_tokens == 100


async def test_logprob_renormalizes_raw_distributions():
    scores = {
        "department": {"billing": 0.9, "returns": 0.5},  # sums to 1.4
        "urgency": {"true": 2.0, "false": 0.0},          # sums to 2.0
        "frustration": {"calm": 1.0, "frustrated": 1.0, "very_frustrated": 1.0},
    }
    backend = LogprobMockBackend(scores, normalize=False)
    engine = SystemOneEngine(_logprob_profile(), backend=backend)
    resp = await engine.system_one(sample_request())

    dept = resp.answers["department"]
    assert sum(dept.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    # Ratio preserved: 0.9 : 0.5 -> 9/14
    assert dept.probabilities["billing"] == pytest.approx(0.9 / 1.4, abs=1e-6)

    urg = resp.answers["urgency"]
    assert urg.noul == pytest.approx(1.0, abs=1e-6)

    frust = resp.answers["frustration"]
    # Uniform mass -> score expectation is the midpoint index (1.0).
    assert frust.score == pytest.approx(1.0, abs=1e-4)
    assert frust.confidence == pytest.approx(0.0, abs=1e-6)


async def test_default_strategy_stays_one_shot():
    backend = LogprobMockBackend({"department": {"billing": 1.0, "returns": 0.0}})
    engine = SystemOneEngine(mock_profile(), backend=backend)  # strategy: one_shot
    resp = await engine.system_one(sample_request())

    assert backend.score_calls == []          # logprob never attempted
    assert backend.complete_calls == 1
    assert resp.answers["department"].choice == "billing"


def test_scoring_prompt_and_expected_options():
    req = sample_request()
    assert expected_options(req) == {
        "department": ["billing", "returns"],
        "urgency": ["true", "false"],
        "frustration": ["calm", "frustrated", "very_frustrated"],
    }

    msg = compile_scoring_user_message(req.state, "department", req.questions["department"])
    assert msg.startswith("STATE:\nCustomer ticket")
    payload = json.loads(msg.split("\nQUESTION:\n", 1)[1])
    assert payload["id"] == "department"
    assert [o["name"] for o in payload["options"]] == ["billing", "returns"]

    # Noul questions score over literal true/false labels.
    noul_msg = compile_scoring_user_message(req.state, "urgency", req.questions["urgency"])
    noul_payload = json.loads(noul_msg.split("\nQUESTION:\n", 1)[1])
    assert [o["name"] for o in noul_payload["options"]] == ["true", "false"]

    assert "one option label" in SCORING_SYSTEM_PROMPT
