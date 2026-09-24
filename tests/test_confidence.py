from jeviathan.confidence.derive import (
    argmax,
    distribution_confidence,
    score_expectation,
)


def test_choice_confidence_three_options_matches_typesafe_demo():
    # TypeSafe docs demo: [90%, 6%, 4%] over 3 options -> (3*0.9 - 1)/2 = 0.85
    import pytest

    assert distribution_confidence({"A": 0.90, "B": 0.06, "C": 0.04}) == pytest.approx(0.85)


def test_flat_distribution_is_zero_confidence():
    flat = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    assert distribution_confidence(flat) < 1e-9


def test_two_option_extremes():
    assert distribution_confidence({"yes": 1.0, "no": 0.0}) == 1.0
    assert distribution_confidence({"yes": 0.5, "no": 0.5}) == 0.0
    # (2*0.78 - 1)/1 = 0.56
    assert abs(distribution_confidence({"billing": 0.78, "returns": 0.22}) - 0.56) < 1e-9


def test_single_option_is_full_confidence():
    assert distribution_confidence({"only": 1.0}) == 1.0


def test_empty_distribution():
    assert distribution_confidence({}) == 0.0


def test_argmax_deterministic_first_max():
    probs = {"a": 0.5, "b": 0.5, "c": 0.0}
    assert argmax(probs) == "a"
    assert argmax({"x": 0.1, "y": 0.9}) == "y"


def test_score_expectation():
    dist = {"calm": 0.1, "frustrated": 0.55, "very_frustrated": 0.35}
    names = ["calm", "frustrated", "very_frustrated"]
    # 0*0.1 + 1*0.55 + 2*0.35 = 1.25
    assert abs(score_expectation(dist, names) - 1.25) < 1e-9
