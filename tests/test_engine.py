import pytest

from jeviathan.backends.base import CompletionResult, DecisionBackend
from jeviathan.calibration.calibrator import Calibrator, expected_calibration_error, fit_platt
from jeviathan.compiler.prompt_compiler import CompileError
from jeviathan.engine.systemone_engine import EngineValidationError, SystemOneEngine
from tests.helpers import MockBackend, mock_profile, mock_text, sample_request


class SequenceBackend(DecisionBackend):
    """Returns canned texts in order (last one repeats)."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.i = 0

    async def complete(self, system_prompt: str, user_message: str) -> CompletionResult:
        t = self.texts[min(self.i, len(self.texts) - 1)]
        self.i += 1
        return CompletionResult(text=t)


async def test_end_to_end_shapes_and_values():
    backend = MockBackend(mock_text(), input_tokens=1234, output_tokens=67)
    engine = SystemOneEngine(mock_profile(), backend=backend)
    resp = await engine.system_one(sample_request())

    assert resp.model == "mock-model"
    assert resp.usage.input_tokens == 1234
    assert resp.usage.output_tokens == 67

    dept = resp.answers["department"]
    assert dept.type == "choice"
    assert dept.choice == "billing"
    assert dept.confidence == pytest.approx(0.56, abs=1e-6)
    assert dept.probabilities["returns"] == pytest.approx(0.22)

    urgency = resp.answers["urgency"]
    assert urgency.type == "noul"
    assert urgency.noul == pytest.approx(0.62)

    frust = resp.answers["frustration"]
    assert frust.type == "score"
    assert frust.score == pytest.approx(1.25, abs=1e-4)
    # (3*0.55 - 1)/2 = 0.325
    assert frust.confidence == pytest.approx(0.325, abs=1e-6)


async def test_request_model_override():
    backend = MockBackend(mock_text())
    engine = SystemOneEngine(mock_profile(), backend=backend)
    req = sample_request()
    req.model = "custom-model"
    resp = await engine.system_one(req)
    assert resp.model == "custom-model"


async def test_limits_enforced():
    profile = mock_profile()
    profile.limits.max_questions = 2
    backend = MockBackend(mock_text())
    engine = SystemOneEngine(profile, backend=backend)
    with pytest.raises(EngineValidationError, match="too many questions"):
        await engine.system_one(sample_request())


async def test_bad_model_output_maps_to_compile_error():
    backend = MockBackend("Sure! Here is my answer: billing.")
    engine = SystemOneEngine(mock_profile(), backend=backend)
    with pytest.raises(CompileError):
        await engine.system_one(sample_request())


def test_calibrator_applies_platt_params():
    cal = Calibrator(params={"choice": (2.0, 0.0)})
    dist = {"a": 0.6, "b": 0.4}
    out = cal.apply_distribution(dist, "choice", "some-qid")
    # Sharper: a=2 doubles the logit gap -> a gains probability mass.
    assert out["a"] > 0.6
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_calibrator_identity_when_no_params():
    cal = Calibrator(params={})
    dist = {"a": 0.3, "b": 0.7}
    out = cal.apply_distribution(dist, "choice", "unknown")
    assert out == dist


def test_fit_platt_recovers_separating_transform():
    # Raw probabilities systematically underconfident: p_raw = sqrt(p_true).
    import math

    raw, labels = [], []
    for p_true in (0.55, 0.6, 0.7, 0.8, 0.9, 0.95):
        raw.append(math.sqrt(p_true))
        labels.append(1)
    a, b = fit_platt(raw, labels)
    # After calibration, high true-probability rows should be pushed up.
    assert a > 1.0


async def test_corrective_retry_recovers_bad_json():
    backend = SequenceBackend(["{bad json", mock_text()])
    engine = SystemOneEngine(mock_profile(), backend=backend)
    resp = await engine.system_one(sample_request())
    assert resp.answers["department"].choice == "billing"
    assert backend.i == 2


async def test_corrective_retry_exhaustion_raises():
    backend = SequenceBackend(["{bad json"])  # repeats forever
    engine = SystemOneEngine(mock_profile(), backend=backend)
    with pytest.raises(CompileError):
        await engine.system_one(sample_request())
    assert backend.i == 3  # initial + 2 corrective retries


def test_ece_perfectly_calibrated_is_low():
    probs = [0.8] * 50 + [0.2] * 50
    labels = [1] * 40 + [0] * 10 + [0] * 40 + [1] * 10  # 80% of .8-bucket correct, etc.
    ece = expected_calibration_error(probs, labels)
    assert 0.0 <= ece < 0.2
