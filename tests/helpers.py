"""Shared fixtures: mock backend + sample request (no GPU required)."""

from __future__ import annotations

import json

from jeviathan.backends.base import CompletionResult, DecisionBackend
from jeviathan.config import BackendConfig, CalibrationConfig, LimitsConfig, Profile, SamplingConfig
from jeviathan.schemas.typesafe import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreLevel,
    ScoreQuestion,
    SystemOneRequest,
)


class MockBackend(DecisionBackend):
    def __init__(self, text: str, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    async def complete(self, system_prompt: str, user_message: str) -> CompletionResult:
        self.calls += 1
        return CompletionResult(
            text=self.text,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


def sample_request() -> SystemOneRequest:
    return SystemOneRequest(
        state=(
            "Customer ticket: 'My card was charged twice for order #4512 on March 3rd. "
            "I want a refund and I am not happy about this.'"
        ),
        questions={
            "department": ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={
                    "billing": "Charges, invoices, payment problems",
                    "returns": "Exchanges, wrong or damaged items",
                },
            ),
            "urgency": NoulQuestion(instructions="Is this ticket urgent?"),
            "frustration": ScoreQuestion(
                instructions="How frustrated is the customer?",
                levels=[
                    ScoreLevel(name="calm", description="Polite, factual"),
                    ScoreLevel(name="frustrated", description="Impatient or annoyed"),
                    ScoreLevel(name="very_frustrated", description="Angry, demanding action"),
                ],
            ),
        },
    )


def mock_text() -> str:
    return json.dumps(
        {
            "answers": {
                "department": {"probabilities": {"billing": 0.78, "returns": 0.22}},
                "urgency": {"probabilities": {"true": 0.62, "false": 0.38}},
                "frustration": {
                    "probabilities": {"calm": 0.10, "frustrated": 0.55, "very_frustrated": 0.35}
                },
            }
        }
    )


def mock_profile() -> Profile:
    return Profile(
        name="mock",
        backend=BackendConfig(base_url="http://mock.invalid/v1", model="mock-model"),
        sampling=SamplingConfig(),
        calibration=CalibrationConfig(enabled=False),
        limits=LimitsConfig(),
    )
