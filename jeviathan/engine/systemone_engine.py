"""System One engine — orchestrates one decision pass.

Flow: validate limits -> compile prompt -> backend completion -> parse +
normalize distributions -> calibrate (if artifact present) -> derive typed
answers with confidence, TypeSafe-shaped.
"""

from __future__ import annotations

import json

from ..backends.base import DecisionBackend
from ..backends.openai_backend import OpenAICompatibleBackend
from ..calibration.calibrator import Calibrator
from ..compiler.prompt_compiler import CompileError, compile_request, parse_answers
from ..confidence.derive import argmax, distribution_confidence, score_expectation
from ..config import Profile
from ..schemas.typesafe import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)


class EngineValidationError(ValueError):
    """Raised when the request violates profile limits."""


# Corrective retries when the model's JSON is malformed (mirrors
# system-one-adapter-python's n_retry_malformed_structure).
MAX_CORRECTIVE_RETRIES = 2


class SystemOneEngine:
    def __init__(self, profile: Profile, backend: DecisionBackend | None = None) -> None:
        self.profile = profile
        self.backend = backend or OpenAICompatibleBackend(profile.backend, profile.sampling)
        self.calibrator: Calibrator | None = None
        if profile.calibration.enabled and profile.calibration.path:
            try:
                self.calibrator = Calibrator.from_artifact(profile.calibration.path)
            except (OSError, ValueError):
                self.calibrator = None

    async def close(self) -> None:
        await self.backend.close()

    def _validate_limits(self, req: SystemOneRequest) -> None:
        limits = self.profile.limits
        state_chars = (
            len(req.state) if isinstance(req.state, str) else len(json.dumps(req.state))
        )
        if state_chars > limits.max_state_chars:
            raise EngineValidationError(
                f"state too large: {state_chars} chars > limit {limits.max_state_chars}"
            )
        if len(req.questions) > limits.max_questions:
            raise EngineValidationError(
                f"too many questions: {len(req.questions)} > limit {limits.max_questions}"
            )
        for qid, q in req.questions.items():
            if isinstance(q, ChoiceQuestion) and len(q.criteria) > limits.max_options_per_choice:
                raise EngineValidationError(
                    f"question {qid!r}: {len(q.criteria)} options > limit "
                    f"{limits.max_options_per_choice}"
                )

    async def system_one(self, req: SystemOneRequest) -> SystemOneResponse:
        self._validate_limits(req)
        system_prompt, user_message = compile_request(req)
        result = await self.backend.complete(system_prompt, user_message)

        last_exc: CompileError | None = None
        for attempt in range(MAX_CORRECTIVE_RETRIES + 1):
            try:
                raw_dists = parse_answers(result.text, req)
                break
            except CompileError as exc:
                last_exc = exc
                if attempt == MAX_CORRECTIVE_RETRIES:
                    raise
                repair_message = (
                    "Your previous output was not valid JSON.\n"
                    f"Previous output:\n{result.text[:1500]}\n\n"
                    "Return ONLY the corrected JSON object with exactly the same shape."
                )
                result = await self.backend.complete(system_prompt, repair_message)
        else:  # pragma: no cover - loop always breaks or raises
            raise last_exc  # type: ignore[misc]

        answers: dict[str, object] = {}
        for qid, q in req.questions.items():
            dist = raw_dists[qid]
            if isinstance(q, NoulQuestion):
                p_true = dist.get("true", 0.5)
                if self.calibrator is not None:
                    p_true = self.calibrator.apply_noul(p_true, qid)
                answers[qid] = NoulAnswer(noul=round(min(max(p_true, 0.0), 1.0), 6))
            else:
                if self.calibrator is not None:
                    dist = self.calibrator.apply_distribution(dist, q.type, qid)
                probs = {k: round(v, 6) for k, v in dist.items()}
                confidence = round(distribution_confidence(dist), 6)
                if isinstance(q, ChoiceQuestion):
                    answers[qid] = ChoiceAnswer(
                        choice=argmax(dist), confidence=confidence, probabilities=probs
                    )
                else:  # ScoreQuestion
                    names = q.level_names()
                    score = round(score_expectation(dist, names), 4)
                    answers[qid] = ScoreAnswer(
                        score=score, confidence=confidence, probabilities=probs
                    )

        return SystemOneResponse(
            model=req.model or self.profile.backend.model,
            answers=answers,  # type: ignore[arg-type]
            usage=Usage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            ),
        )
