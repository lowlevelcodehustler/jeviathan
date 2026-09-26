"""End-to-end smoke test for the v1.1 logprob strategy against a running backend.

Runs one 3-question System One pass (noul + score + choice) through the full
engine with `strategy: logprob` — exercises assistant-prefill + logprobs on
whatever serving stack is behind JEVIATHAN_BASE_URL, then prints typed answers.

Usage:
    python scripts/smoke_logprob.py [profile]     # default profile: rtx5090

Laptop shim example (logprob strategy against the local NF4 Llama):
    set JEVIATHAN_BASE_URL=http://localhost:8200/v1
    set JEVIATHAN_MODEL=llama3.1-8b-local
    python scripts/smoke_logprob.py rtx5090

First thing to run on the 5090 box after setup_5090.sh — verifies vLLM's
assistant-prefill + logprobs support empirically (the engine falls back to
one_shot automatically if the server can't do it, so a green exit alone is
not proof; check that usage.input_tokens is null in the output).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

from jeviathan.config import load_profile
from jeviathan.engine.systemone_engine import SystemOneEngine
from jeviathan.schemas.typesafe import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreLevel,
    ScoreQuestion,
    SystemOneRequest,
)


def build_request() -> SystemOneRequest:
    return SystemOneRequest(
        state=(
            "Claim: Metformin is first-line pharmacotherapy for type 2 diabetes "
            "in most patients without contraindications."
        ),
        questions={
            "claim_supported": NoulQuestion(
                instructions="Is the factual claim accurate and supported by well-established domain knowledge?"
            ),
            "risk_level": ScoreQuestion(
                instructions="How high is the risk if this claim turns out to be wrong?",
                levels=[
                    ScoreLevel(name="low", description="Minor impact; easily corrected"),
                    ScoreLevel(name="medium", description="Material impact on a decision or customer"),
                    ScoreLevel(name="high", description="Safety, regulatory, or financial exposure"),
                ],
            ),
            "vertical": ChoiceQuestion(
                instructions="Which domain does this claim belong to?",
                criteria={
                    "healthcare": None,
                    "aviation": None,
                    "financial": None,
                    "legal": None,
                    "gaming": None,
                    "support": None,
                    "other": None,
                },
            ),
        },
    )


async def main(profile_name: str) -> int:
    profile = load_profile(profile_name)  # applies JEVIATHAN_BASE_URL/MODEL overrides
    strategy = getattr(profile.sampling, "strategy", "one_shot")
    if strategy != "logprob":
        print(
            f"note: profile {profile_name!r} uses strategy={strategy!r}; pass a logprob "
            "profile (rtx5090 / laptop-4050-logprob) to exercise v1.1",
            flush=True,
        )
    engine = SystemOneEngine(profile)
    try:
        t0 = time.time()
        resp = await engine.system_one(build_request())
    finally:
        await engine.close()
    print(
        f"elapsed: {time.time() - t0:.1f}s  (profile={profile.name}, model={resp.model})",
        flush=True,
    )
    dump = resp.model_dump()
    print(json.dumps(dump, indent=2))
    if strategy == "logprob" and dump["usage"]["input_tokens"] is not None:
        print(
            "\nWARNING: usage tokens present -> the one_shot fallback ran; the server's "
            "assistant-prefill/logprobs path failed. Check its logs.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "rtx5090"
    sys.exit(asyncio.run(main(name)))
