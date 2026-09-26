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

    async def score_prefixes(
        self,
        system_prompt: str,
        user_message: str,
        prefixes: list[str],
    ) -> dict[str, float] | None:
        """Per-option prefix scoring via assistant-prefill + logprobs.

        Best effort across OpenAI-compatible servers (vLLM/SGLang/our shim):
        prefill the assistant turn with each option label and read the mean
        logprob of that span. Returns None on any failure so the engine can
        fall back to one_shot.
        """
        import math

        raw: dict[str, float] = {}
        for prefix in prefixes:
            body = {
                "model": self._cfg.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": prefix},  # prefill
                ],
                "temperature": 0.0,
                "max_tokens": 2,
                "logprobs": True,
                "top_logprobs": 1,
            }
            try:
                resp = await self._client.post("/chat/completions", json=body)
                if resp.status_code == 429 or resp.status_code >= 500:
                    return None
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                lp = (choice.get("logprobs") or {}).get("content") or []
                if not lp:
                    # Some servers only logprob generated tokens; use those as a
                    # proxy for continuation quality after the prefix.
                    return None
                mean_logprob = sum(item["logprob"] for item in lp) / len(lp)
                raw[prefix] = mean_logprob
            except (httpx.HTTPError, KeyError, ValueError, TypeError):
                return None

        # Normalize log-scores into a distribution.
        m = max(raw.values())
        exps = {k: math.exp(v - m) for k, v in raw.items()}
        total = sum(exps.values()) or 1.0
        return {k: e / total for k, e in exps.items()}

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
