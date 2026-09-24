"""Backend for any OpenAI-compatible /chat/completions server.

Covers vLLM (RTX 5090), Ollama and llama.cpp servers (laptop), the local
transformers shim (scripts/transformers_server.py), or a remote endpoint.
"""

from __future__ import annotations

import asyncio

import httpx

from ..config import BackendConfig, SamplingConfig
from .base import BackendError, CompletionResult, DecisionBackend


class OpenAICompatibleBackend(DecisionBackend):
    def __init__(self, backend: BackendConfig, sampling: SamplingConfig) -> None:
        self._cfg = backend
        self._sampling = sampling
        headers = {"Content-Type": "application/json"}
        if backend.api_key:
            headers["Authorization"] = f"Bearer {backend.api_key}"
        self._client = httpx.AsyncClient(
            base_url=backend.base_url.rstrip("/"),
            headers=headers,
            timeout=backend.timeout_s,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(self, system_prompt: str, user_message: str) -> CompletionResult:
        body = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self._sampling.temperature,
            "top_p": self._sampling.top_p,
            "max_tokens": self._sampling.max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                resp = await self._client.post("/chat/completions", json=body)
                if resp.status_code == 429 or resp.status_code >= 500:
                    retry_after = resp.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else min(2**attempt, 16.0)
                    last_error = BackendError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                else:
                    resp.raise_for_status()
                    data = resp.json()
                    text = (data["choices"][0]["message"]["content"]) or ""
                    usage = data.get("usage") or {}
                    return CompletionResult(
                        text=text,
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                    )
            except (httpx.TransportError, BackendError, KeyError, ValueError) as exc:
                last_error = exc

            if attempt < self._cfg.max_retries:
                await asyncio.sleep(min(2**attempt, 16.0))

        raise BackendError(f"backend failed after retries: {last_error}") from last_error
