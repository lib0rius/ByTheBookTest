"""The measures of the test against independent implementations.

Each measure of :mod:`bythebooktest.metrics` is recomputed on random data with scipy, scikit-learn or a direct
formula written here, and must agree. The data are random but seeded, so the test is reproducible.
"""

from __future__ import annotations

import math
import random

import pytest

from bythebooktest import metrics
from bythebooktest.records import ClaimRecord

stats = pytest.importorskip("scipy.stats")
skm = pytest.importorskip("sklearn.metrics")

LABELS = list(metrics.LABELS)


def _claim(i: int, expected: str, relation: str = "neutral") -> ClaimRecord:
    return ClaimRecord(
        claim_id=f"C{i}",
        kb_id="K",
        level=1,
        group="g",
        subtype="s",
        text="t",
        logic="l",
        expected=expected,
        z3=expected,
        world_relation=relation,
        category=None,
        depth=0,
        reasoning=[],
        support=[],
        traps=[],
    )


def _random_run(n: int, seed: int):
    rng = random.Random(seed)
    claims, ans, responses = [], {}, {}
    for i in range(n):
        c = _claim(i, rng.choice(LABELS))
        weights = [rng.random() + (1.5 if lab == c.expected else 0) for lab in LABELS]
        total = sum(weights)
        probs = {lab: w / total for lab, w in zip(LABELS, weights)}
        answer = max(probs, key=probs.__getitem__)
        claims.append(c)
        ans[c.claim_id] = answer
        responses[c.claim_id] = {"status": "ok", "probs": probs, "confidence": probs[answer]}
    return claims, ans, responses


@pytest.mark.parametrize(("k", "n"), [(0, 10), (3, 10), (580, 1400), (1400, 1400), (78, 140)])
def test_wilson_matches_scipy(k, n):
    lo, hi = metrics.wilson(k, n)
    ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert lo == pytest.approx(ref.low, abs=1e-6)
    assert hi == pytest.approx(ref.high, abs=1e-6)


def test_newcombe_is_built_from_the_two_wilson_intervals():
    """Newcombe's hybrid score interval for p1 - p2: d - sqrt((p1-l1)^2 + (u2-p2)^2), d + sqrt((u1-p1)^2 +
    (p2-l2)^2), with the Wilson bounds computed here by scipy, not by the project.
    """
    for k1, n1, k2, n2 in [(221, 504, 78, 140), (214, 324, 65, 90), (221, 504, 96, 280)]:
        a, b = metrics.proportion(k1, n1), metrics.proportion(k2, n2)
        d = metrics.difference(a, b)
        w1 = stats.binomtest(k1, n1).proportion_ci(method="wilson")
        w2 = stats.binomtest(k2, n2).proportion_ci(method="wilson")
        p1, p2 = k1 / n1, k2 / n2
        lo = (p1 - p2) - math.sqrt((p1 - w1.low) ** 2 + (w2.high - p2) ** 2)
        hi = (p1 - p2) + math.sqrt((w1.high - p1) ** 2 + (p2 - w2.low) ** 2)
        assert d["ci95"][0] == pytest.approx(lo, abs=1e-6)
        assert d["ci95"][1] == pytest.approx(hi, abs=1e-6)


def test_long_rulebook_effects_reproduce_the_published_values():
    """The conflict effect on the long rulebooks: 221/504 - 78/140 = -11.9 [-20.9, -2.5]; the invented-words
    effect 221/504 - 96/280 = +9.6 [+2.4, +16.4], as published.
    """
    ce = metrics.difference(metrics.proportion(221, 504), metrics.proportion(78, 140))
    assert [round(100 * v, 1) for v in (ce["value"], *ce["ci95"])] == [-11.9, -20.9, -2.5]
    iw = metrics.difference(metrics.proportion(221, 504), metrics.proportion(96, 280))
    assert [round(100 * v, 1) for v in (iw["value"], *iw["ci95"])] == [9.6, 2.4, 16.4]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_macro_f1_and_kappa_match_scikit_learn(seed):
    claims, ans, _ = _random_run(300, seed)
    out = metrics.label_measures(claims, ans)
    y_true = [c.expected for c in claims]
    y_pred = [ans[c.claim_id] for c in claims]
    ref_f1 = skm.f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    assert out["macro_f1"] == pytest.approx(ref_f1, abs=1e-12)
    assert out["kappa"] == pytest.approx(skm.cohen_kappa_score(y_true, y_pred, labels=LABELS), abs=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_brier_and_ece_match_direct_formulas(seed):
    """Brier: mean over claims of the sum over the three answers of (p_a - [a = expected])^2. ECE: 10 bins of
    equal width on the confidence, weighted mean of |accuracy - mean confidence|.
    """
    claims, ans, responses = _random_run(500, seed)
    out = metrics.calibration(claims, responses, ans)
    n = len(claims)
    brier = sum(sum((responses[c.claim_id]["probs"][a] - (a == c.expected)) ** 2 for a in LABELS) for c in claims) / n
    assert out["brier"] == pytest.approx(brier, abs=1e-12)
    assert out["brier_skill"] == pytest.approx(1 - brier / (2 / 3), abs=1e-12)
    bins: dict[int, list[tuple[float, bool]]] = {}
    for c in claims:
        conf = responses[c.claim_id]["confidence"]
        bins.setdefault(min(int(conf * 10), 9), []).append((conf, ans[c.claim_id] == c.expected))
    ece = sum(abs(sum(r for _, r in b) / len(b) - sum(p for p, _ in b) / len(b)) * len(b) for b in bins.values()) / n
    assert out["ece"] == pytest.approx(ece, abs=1e-12)


def test_uniform_probabilities_give_brier_two_thirds():
    """With p = 1/3 for every answer: (2/3)^2 + 2 (1/3)^2 = 2/3, the reference of the Brier skill score."""
    brier_uniform = (2 / 3) ** 2 + 2 * (1 / 3) ** 2
    assert brier_uniform == pytest.approx(2 / 3)


def test_silence_effect_standard_error_is_the_score_standard_error():
    rng = random.Random(7)
    ut = [_claim(i, "UNKNOWN", "unsupported_true") for i in range(80)]
    uf = [_claim(100 + i, "UNKNOWN", "unsupported_false") for i in range(70)]
    ans = {c.claim_id: rng.choice(LABELS) for c in ut + uf}
    out = metrics.silence_effect(ut, uf, ans)
    scores = [
        -1 if ans[c.claim_id] == ("TRUE" if c in ut else "FALSE") else (1 if ans[c.claim_id] != "UNKNOWN" else 0)
        for c in ut + uf
    ]
    mean = sum(scores) / len(scores)
    sd0 = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))  # population standard deviation
    assert out["value"] == pytest.approx(mean, abs=1e-12)
    half = float(stats.norm.ppf(0.975)) * sd0 / math.sqrt(len(scores))
    assert out["ci95"][0] == pytest.approx(max(-1.0, mean - half), abs=1e-6)
    assert out["ci95"][1] == pytest.approx(min(1.0, mean + half), abs=1e-6)


def test_auroc_equals_the_mann_whitney_statistic():
    """AUROC = U / (n_right * n_wrong), the probability that a right answer has higher confidence."""
    rng = random.Random(11)
    right = [rng.random() for _ in range(60)]
    wrong = [rng.random() * 0.9 for _ in range(40)]
    u = stats.mannwhitneyu(right, wrong, alternative="two-sided").statistic
    auc = skm.roc_auc_score([1] * 60 + [0] * 40, right + wrong)
    assert auc == pytest.approx(u / (60 * 40), abs=1e-12)


def test_student_t_quantile_with_three_degrees_of_freedom():
    """The interval over the four rulebooks of a length uses t_{0.975, 3} = 3.182."""
    assert round(float(stats.t.ppf(0.975, 3)), 3) == 3.182
