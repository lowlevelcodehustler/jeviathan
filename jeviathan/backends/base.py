from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class BackendError(RuntimeError):
    """Raised when the model backend fails after retries."""


class DecisionBackend(ABC):
    """One decision pass over (system_prompt, user_message)."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_message: str) -> CompletionResult: ...

    async def score_prefixes(
        self,
        system_prompt: str,
        user_message: str,
        prefixes: list[str],
    ) -> dict[str, float] | None:
        """Logprob strategy (v1.1): P(prefix | state) for each candidate prefix.

        Returns a normalized distribution over `prefixes`, or None when the
        backend cannot score logprobs (engine falls back to one_shot).
        The user_message should end with an "ANSWER:" cue; prefixes are the
        option labels themselves.
        """
        return None  # default: unsupported

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None
