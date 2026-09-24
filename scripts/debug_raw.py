"""Debug: run one compiled pass against the live backend and print the raw output."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jeviathan.backends.openai_backend import OpenAICompatibleBackend  # noqa: E402
from jeviathan.compiler.prompt_compiler import compile_request, parse_answers  # noqa: E402
from jeviathan.config import load_profile  # noqa: E402
from jeviathan.schemas.typesafe import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreLevel,
    ScoreQuestion,
    SystemOneRequest,
)


async def main() -> None:
    profile = load_profile(os.environ.get("JEVIATHAN_PROFILE", "laptop-4050"))
    backend = OpenAICompatibleBackend(profile.backend, profile.sampling)
    req = SystemOneRequest(
        state=(
            'Customer ticket: "My card was charged twice for order #4512 on March 3rd. '
            'I want a refund and I am not happy about this."'
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
    system_prompt, user_message = compile_request(req)
    print("=== PROMPT (user part) ===")
    print(user_message[:800])
    result = await backend.complete(system_prompt, user_message)
    print("\n=== RAW MODEL OUTPUT ===")
    print(result.text)
    try:
        dists = parse_answers(result.text, req)
        print("\n=== PARSED OK ===")
        print(json.dumps(dists, indent=2))
    except Exception as exc:
        print(f"\n=== PARSE FAILED: {exc} ===")
    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
