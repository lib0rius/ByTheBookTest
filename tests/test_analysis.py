import math
import random

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from bythebooktest import analysis
from bythebooktest.records import ClaimRecord


def _claim(i, relation="neutral", expected="TRUE", *, kb="K1", level=3, depth=2, group="g", subtype="s"):
    return ClaimRecord(
        claim_id=f"C{i}",
        kb_id=kb,
        level=level,
        group=group,
        subtype=subtype,
        text="t",
        logic="l",
        expected=expected,
        z3=expected,
        world_relation=relation,
        category=None,
        depth=depth,
        reasoning=[],
        support=[],
        traps=[],
    )


def test_world_answer():
    assert analysis.world_answer(_claim(1, "agree", "TRUE")) == "TRUE"
    assert analysis.world_answer(_claim(2, "conflict", "TRUE")) == "FALSE"
    assert analysis.world_answer(_claim(3, "conflict", "FALSE")) == "TRUE"
    assert analysis.world_answer(_claim(4, "unsupported_true", "UNKNOWN")) == "TRUE"
    assert analysis.world_answer(_claim(5, "unsupported_false", "UNKNOWN")) == "FALSE"
    assert analysis.world_answer(_claim(6, "neutral", "TRUE")) is None


def test_fixed_rules_by_construction():
    claims = [_claim(1, "agree", "TRUE"), _claim(2, "conflict", "TRUE"), _claim(3, "unsupported_true", "UNKNOWN")]
    ans = {"C1": "TRUE", "C2": "TRUE", "C3": "UNKNOWN"}
    r = analysis.fixed_rules(claims, ans)
    assert r["all_real"]["believe_world"]["k"] == 1  # right only on the agreeing claim
    assert r["all_real"]["contradict_world"]["k"] == 1  # right only on the conflicting claim
    assert r["all_real"]["model"]["k"] == 3
    assert r["unsupported_true"]["believe_world"]["k"] == 0


def test_adaptive_ece_perfect_and_worst():
    assert analysis.adaptive_ece([1.0] * 10, [True] * 10) == pytest.approx(0.0)
    assert analysis.adaptive_ece([1.0] * 10, [False] * 10) == pytest.approx(1.0)


def test_t_interval_matches_tables():
    r = analysis.t_interval([0.40, 0.42, 0.44, 0.42])
    sd = math.sqrt(((0.02) ** 2 * 2) / 3)
    half = 3.182446 * sd / 2  # Student t, 97.5% quantile with 3 degrees of freedom
    assert r["value"] == pytest.approx(0.42)
    assert r["ci95"][1] - r["value"] == pytest.approx(half, rel=1e-5)


def test_uniformity_identical_rulebooks_are_explained_by_specification():
    # Two specifications, always right on one and always wrong on the other: the rulebooks are identical,
    # and shuffling within a specification cannot change anything.
    claims, ans = [], {}
    i = 0
    for kb in ("K1", "K2", "K3", "K4"):
        for spec in ("a", "b"):
            for _ in range(5):
                c = _claim(i, kb=kb, subtype=spec)
                claims.append(c)
                ans[c.claim_id] = "TRUE" if spec == "a" else "FALSE"
                i += 1
    u = analysis.uniformity(claims, ans, random.Random(0))
    assert [v["right"] for v in u["per_kb"].values()] == [5, 5, 5, 5]
    assert u["binomial"]["chi2"] == pytest.approx(0.0)
    assert u["permutation"]["lower_tail"] == pytest.approx(1.0)
    assert u["specifications"]["same_answer_share"] == pytest.approx(1.0)


def test_confidence_auroc_separates():
    claims = [_claim(i) for i in range(20)]
    responses = {
        c.claim_id: {"status": "ok", "probs": {"TRUE": 0.5}, "confidence": 0.9 if i < 10 else 0.4}
        for i, c in enumerate(claims)
    }
    ans = {c.claim_id: ("TRUE" if i < 10 else "FALSE") for i, c in enumerate(claims)}
    r = analysis.confidence(claims, responses, ans, random.Random(0))
    assert r["auroc"]["value"] == pytest.approx(1.0)
    assert sum(b["n"] for b in r["fixed_bins"]) == 20


def test_answer_preference_shares():
    claims = [_claim(1, "neutral", "TRUE"), _claim(2, "neutral", "FALSE"), _claim(3, "agree", "TRUE")]
    ans = {"C1": "FALSE", "C2": "FALSE", "C3": "TRUE"}
    r = analysis.answer_preference(claims, ans)
    assert r["invented_words"]["share"]["FALSE"]["k"] == 2
    assert r["invented_words"]["accuracy_by_expected"]["FALSE"]["k"] == 1
    assert r["agree"]["share"]["TRUE"]["k"] == 1
