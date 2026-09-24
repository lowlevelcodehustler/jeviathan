import pytest

from jeviathan.compiler.prompt_compiler import (
    CompileError,
    compile_request,
    expected_options,
    parse_answers,
)
from tests.helpers import mock_text, sample_request


def test_compile_produces_system_and_user():
    system, user = compile_request(sample_request())
    assert "System One decision model" in system
    assert "STATE:" in user and "QUESTIONS:" in user
    assert '"department"' in user and '"frustration"' in user


def test_expected_options_shape():
    opts = expected_options(sample_request())
    assert opts["department"] == ["billing", "returns"]
    assert opts["urgency"] == ["true", "false"]
    assert opts["frustration"] == ["calm", "frustrated", "very_frustrated"]


def test_parse_clean_json():
    dists = parse_answers(mock_text(), sample_request())
    for qid, dist in dists.items():
        assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dists["department"]["billing"] == pytest.approx(0.78)


def test_parse_strips_code_fences():
    raw = "```json\n" + mock_text() + "\n```"
    dists = parse_answers(raw, sample_request())
    assert dists["urgency"]["true"] == pytest.approx(0.62)


def test_parse_renormalizes_when_sum_not_one():
    import json

    raw = json.dumps(
        {
            "answers": {
                "department": {"probabilities": {"billing": 0.5, "returns": 0.3}},
                "urgency": {"probabilities": {"true": 0.4, "false": 0.4}},
                "frustration": {
                    "probabilities": {"calm": 0.2, "frustrated": 0.3, "very_frustrated": 0.3}
                },
            }
        }
    )
    dists = parse_answers(raw, sample_request())
    assert dists["department"]["billing"] == pytest.approx(0.625)


def test_parse_all_zero_becomes_flat_prior():
    import json

    raw = json.dumps(
        {
            "answers": {
                "department": {"probabilities": {"billing": 0.0, "returns": 0.0}},
                "urgency": {"probabilities": {"true": 0.0, "false": 0.0}},
                "frustration": {
                    "probabilities": {"calm": 0.0, "frustrated": 0.0, "very_frustrated": 0.0}
                },
            }
        }
    )
    dists = parse_answers(raw, sample_request())
    assert dists["department"]["billing"] == pytest.approx(0.5)


def test_parse_missing_question_raises():
    import json

    raw = json.dumps({"answers": {"urgency": {"probabilities": {"true": 1.0}}}})
    with pytest.raises(CompileError, match="missing probabilities"):
        parse_answers(raw, sample_request())


def test_parse_null_probability_treated_as_zero_mass():
    import json

    raw = json.dumps(
        {
            "answers": {
                "department": {"probabilities": {"billing": 0.8, "returns": None}},
                "urgency": {"probabilities": {"true": 1.0, "false": 0.0}},
                "frustration": {
                    "probabilities": {"calm": 1.0, "frustrated": 0.0, "very_frustrated": 0.0}
                },
            }
        }
    )
    dists = parse_answers(raw, sample_request())
    assert dists["department"] == {"billing": 1.0, "returns": 0.0}  # renormalized


def test_parse_non_numeric_raises():
    import json

    raw = json.dumps(
        {
            "answers": {
                "department": {"probabilities": {"billing": "high", "returns": 0.2}},
                "urgency": {"probabilities": {"true": 1.0, "false": 0.0}},
                "frustration": {
                    "probabilities": {"calm": 1.0, "frustrated": 0.0, "very_frustrated": 0.0}
                },
            }
        }
    )
    with pytest.raises(CompileError, match="non-numeric"):
        parse_answers(raw, sample_request())


def test_parse_non_json_raises():
    with pytest.raises(CompileError, match="non-JSON"):
        parse_answers("Sure! Here is my answer: billing.", sample_request())


# Regression: exact raw output captured from Llama-3.1-8B (NF4) — dropped a
# closing brace after "department" and looped <|eot_id>N</eot_id> afterwards.
LLAMA_RAW = (
    '{"answers": {"department": {"probabilities": {"billing": 0.8, "returns": 0.2}, '
    '"urgency": {"probabilities": {"true": 0.9, "false": 0.1}, '
    '"frustration": {"probabilities": {"calm": 0.0, "frustrated": 0.7, "very_frustrated": 0.3}}}}'
    '<|eot_id>1</eot_id>2</eot_id>3</eot_id>4</eot_id>5</eot_id>'
)


def test_parse_llama_real_output_missing_brace_and_eot_garbage():
    dists = parse_answers(LLAMA_RAW, sample_request())
    assert set(dists) == {"department", "urgency", "frustration"}
    assert dists["department"] == {"billing": 0.8, "returns": 0.2}
    assert dists["urgency"] == {"true": 0.9, "false": 0.1}
    assert dists["frustration"]["very_frustrated"] == pytest.approx(0.3)


def test_balanced_object_cuts_trailing_garbage():
    from jeviathan.compiler.prompt_compiler import _first_balanced_object

    text = '{"a": {"b": 1}} trailing junk {not json}\n'
    assert _first_balanced_object(text) == '{"a": {"b": 1}}'


def test_hoist_missing_answers():
    from jeviathan.compiler.prompt_compiler import _hoist_missing_answers

    payload = {
        "answers": {
            "department": {
                "probabilities": {"billing": 0.5, "returns": 0.5},
                "urgency": {"probabilities": {"true": 1.0, "false": 0.0}},
            }
        }
    }
    _hoist_missing_answers(payload, ["department", "urgency"])
    assert set(payload["answers"]) == {"department", "urgency"}
    assert "urgency" not in payload["answers"]["department"]
