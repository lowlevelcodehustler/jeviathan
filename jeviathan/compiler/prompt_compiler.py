"""Compiles a System One request (state + questions) into one prompt pass.

This is the "parallel sampler" equivalent for open-weight models: the state
is ingested once and every question is evaluated against it in a single
completion, mirroring Jev's one-request / one-response semantics.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..schemas.typesafe import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

SYSTEM_PROMPT = """You are Jeviathan, a System One decision model. You evaluate the STATE and answer every QUESTION in one pass. Each question is judged independently against the same STATE.

Contract:
1. Judge exactly what each question asks — literally, using only the STATE. Do not use outside knowledge unless a question explicitly allows it.
2. For each question return a probability distribution over its options (or levels) that sums to 1. Use decimals between 0 and 1 with up to 4 places.
3. Concentrate probability on the best-supported option; spread it when genuinely ambiguous or evidence is thin.
4. Every option name must appear with a numeric value (use 0 for options that get no mass). No nulls, no missing keys.
5. Output ONLY one JSON object of exactly this shape:
{"answers": {"<question_id>": {"probabilities": {"<option_name>": <p>, ...}}}}
Use the exact question ids and option names given. No prose, no markdown, no code fences.

Worked example (2 questions: "a" is a choice over x/y, "b" is noul):
{"answers": {"a": {"probabilities": {"x": 0.7, "y": 0.3}}, "b": {"probabilities": {"true": 1.0, "false": 0.0}}}}
Note how each question id maps to its own object containing one "probabilities" object."""


class CompileError(ValueError):
    """Raised when the model's output cannot be turned into valid distributions."""


def _repair_json(text: str) -> str:
    """Cheap repairs for common LLM JSON slips (trailing commas)."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _hoist_missing_answers(payload: dict, expected_qids) -> None:
    """Structural repair: small models sometimes nest later questions inside an
    earlier question's object (dropped closing brace). Hoist them to the top.
    """
    if not isinstance(payload, dict):
        return
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return
    for qid in list(expected_qids):
        if qid in answers:
            continue
        for wrapper_key, wrapper in list(answers.items()):
            if isinstance(wrapper, dict) and qid in wrapper:
                answers[qid] = wrapper.pop(qid)
                break


def _first_balanced_object(text: str) -> str | None:
    """Extract the first brace-balanced JSON object, ignoring braces in strings.

    Handles trailing garbage (e.g. Llama-3.1's <|eot_id>N</eot_id> loops).
    Falls back to a greedy first-{..last-} slice when unbalanced.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    end = text.rfind("}")
    return text[start : end + 1] if end > start else None


SCORING_SYSTEM_PROMPT = """You are Jeviathan, a System One decision model. You evaluate the STATE and answer ONE question by outputting exactly one option label from its options — nothing else: no punctuation, no quotes, no explanation. Judge literally using only the STATE."""


def _render_state(state: Any) -> str:
    if isinstance(state, str):
        return state.strip()
    return json.dumps(state, ensure_ascii=False)


def _question_payload(qid: str, q) -> dict[str, Any]:
    if isinstance(q, ChoiceQuestion):
        options = [
            {"name": name, "description": desc} for name, desc in q.criteria.items()
        ]
    elif isinstance(q, ScoreQuestion):
        names = q.level_names()
        options = [
            {"name": n, "description": lv.description or ""}
            for n, lv in zip(names, q.levels)
        ]
    else:  # NoulQuestion
        options = [
            {"name": "true", "description": None},
            {"name": "false", "description": None},
        ]
    return {
        "id": qid,
        "type": q.type,
        "instructions": q.instructions,
        "options": options,
    }


def compile_scoring_user_message(state: Any, qid: str, q) -> str:
    """Single-question prompt for the logprob strategy (ends before ANSWER:)."""
    return (
        f"STATE:\n{_render_state(state)}\n\n"
        f"QUESTION:\n{json.dumps(_question_payload(qid, q), ensure_ascii=False)}"
    )


def compile_request(req: SystemOneRequest) -> tuple[str, str]:
    """Return (system_prompt, user_message) for one decision pass."""
    questions_payload = [_question_payload(qid, q) for qid, q in req.questions.items()]
    user = (
        f"STATE:\n{_render_state(req.state)}\n\n"
        f"QUESTIONS:\n{json.dumps(questions_payload, ensure_ascii=False)}"
    )
    return SYSTEM_PROMPT, user


def expected_options(req: SystemOneRequest) -> dict[str, list[str]]:
    """Map question id -> ordered option/level names (true/false for Noul)."""
    out: dict[str, list[str]] = {}
    for qid, q in req.questions.items():
        if isinstance(q, ChoiceQuestion):
            out[qid] = list(q.criteria.keys())
        elif isinstance(q, ScoreQuestion):
            out[qid] = q.level_names()
        else:
            out[qid] = ["true", "false"]
    return out


def parse_answers(raw: str, req: SystemOneRequest) -> dict[str, dict[str, float]]:
    """Parse + validate + renormalize the model's JSON into raw distributions.

    Mirrors system-one-adapter-python behaviour: strip code fences, extract
    the first JSON object if needed, clamp probabilities to [0, 1], and
    rescale each distribution so it sums to exactly 1.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Candidate ladder, cheapest first:
    #   1. whole text            2. balanced object (cuts trailing garbage)
    #   3-4. same with trailing commas stripped
    #   5+. brace-completion: append missing '}' (small models drop closers)
    bases = [text]
    balanced = _first_balanced_object(text)
    if balanced and balanced != text:
        bases.append(balanced)
    candidates: list[str] = []
    for base in bases:
        for cand in (base, _repair_json(base)):
            if cand not in candidates:
                candidates.append(cand)
    target = balanced or text
    for k in range(1, 5):
        cand = _repair_json(target + "}" * k)
        if cand not in candidates:
            candidates.append(cand)

    last_exc: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_exc = exc
    else:
        if "{" not in text:
            raise CompileError(f"model returned non-JSON output: {raw[:200]!r}") from None
        raise CompileError(f"unparseable model JSON: {last_exc}") from last_exc

    _hoist_missing_answers(payload, expected_options(req).keys())

    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise CompileError('model JSON missing "answers" object')

    result: dict[str, dict[str, float]] = {}
    for qid, options in expected_options(req).items():
        entry = answers.get(qid)
        probs = entry.get("probabilities") if isinstance(entry, dict) else None
        if not isinstance(probs, dict):
            raise CompileError(f"missing probabilities for question {qid!r}")

        dist: dict[str, float] = {}
        for name in options:
            val = probs.get(name)
            if val is None:  # model omitted mass for this option
                p = 0.0
            else:
                try:
                    p = float(val)
                except (TypeError, ValueError):
                    raise CompileError(
                        f"non-numeric probability for {qid}.{name}: {val!r}"
                    ) from None
            dist[name] = max(0.0, min(1.0, p))

        total = sum(dist.values())
        if total <= 0:
            flat = 1.0 / len(options)
            dist = {name: flat for name in options}
        else:
            dist = {name: p / total for name, p in dist.items()}
        result[qid] = dist
    return result
