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

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None
