"""Pydantic models mirroring the TypeSafe System One API contract.

Request/response shapes follow docs.typesafe.ai so that Jeviathan is a
drop-in stand-in for (and A/B-comparable with) the real TypeSafe API:
POST /v1/systemone  {state, model?, questions} -> {model, answers, usage}.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- questions


class ChoiceQuestion(BaseModel):
    """Pick one option from a fixed set (up to 255 options)."""

    type: Literal["choice"] = "choice"
    instructions: str | dict[str, Any] | list[Any]
    criteria: dict[str, str | None] = Field(min_length=1)


class ScoreLevel(BaseModel):
    name: str | None = None
    description: str | None = None


class ScoreQuestion(BaseModel):
    """Rate the state on an ordered set of levels (0-based)."""

    type: Literal["score"] = "score"
    instructions: str | dict[str, Any] | list[Any]
    levels: list[ScoreLevel] = Field(min_length=2)

    def level_names(self) -> list[str]:
        return [lv.name or str(i) for i, lv in enumerate(self.levels)]


class NoulQuestion(BaseModel):
    """Is this statement true? Returns a probability in [0, 1]."""

    type: Literal["noul"] = "noul"
    instructions: str | dict[str, Any] | list[Any]


Question = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, NoulQuestion],
    Field(discriminator="type"),
]


# ------------------------------------------------------------------ request


class SystemOneRequest(BaseModel):
    state: Union[str, dict[str, Any], list[Any]]
    model: str | None = None
    questions: dict[str, Question] = Field(min_length=1)


# ----------------------------------------------------------------- responses


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


Answer = Annotated[
    Union[ChoiceAnswer, ScoreAnswer, NoulAnswer],
    Field(discriminator="type"),
]


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
