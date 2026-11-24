"""Extra analyses of a run, computed from the saved answers only (no model is called).

Usage::

    bythebooktest analysis <run folder> [<run folder> ...] [--root <repository>] [--out <folder>]

Writes ``analysis.json`` and ``analysis.md`` (default folder: ``<first run>/analysis``). Six analyses,
each by level (short, medium, long rulebooks):

- **uniformity**: how similar the rulebooks of a level are. The accuracy of each rulebook is compared with
  the spread expected if every claim were an independent draw (binomial), and with the spread left once
  claims are matched by their specification (group, variant, expected answer, relation with the world,
  depth), through a stratified permutation test: the rulebook labels are shuffled among claims with the
  same specification. Also the share of claims whose specification gets the same answer in every case.
- **per_rulebook**: accuracy, conflict effect and silence effect with a 95% interval built from the
  per-rulebook values (Student t with K - 1 degrees of freedom), the cautious interval when claims of one
  rulebook are not independent.
- **confidence**: AUROC of the confidence of the chosen answer for telling right from wrong answers
  (0.5 is chance, 1 is perfect), with a bootstrap interval, and the expected calibration error over 10
  bins of equal size (adaptive) beside the 10 bins of equal width used by :mod:`metrics`.
- **depth_by_length**: accuracy on TRUE or FALSE claims for every number of steps and every level.
- **fixed_rules**: accuracy that two answers fixed in advance would get on the claims about real facts:
  "believe the world" (answer what is true in the real world) and "contradict the world".
- **answer_preference**: the share of each answer on the TRUE or FALSE claims made of invented words,
  against the claims that contradict or agree with the world.

Needs the optional dependencies of the ``analysis`` extra (scikit-learn and scipy).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from . import metrics
from .records import ClaimRecord

SEED = 20261122
"""Seed of the permutation test and of the bootstrap, so that every number can be reproduced."""
PERMUTATIONS = 10_000
BOOTSTRAP = 2_000
BINS = 10
LABELS = metrics.LABELS

Spec = tuple[str, str, str, str, int]


def spec_of(c: ClaimRecord) -> Spec:
    """The specification a claim was built from: what the plan fixes before the claim is generated."""
    return (c.group, c.subtype, c.expected, c.world_relation, c.depth)


def _answers(claims: list[ClaimRecord], responses: dict[str, dict]) -> dict[str, str | None]:
    return {c.claim_id: metrics._answer(responses.get(c.claim_id), metrics.MAIN == "lenient") for c in claims}


def _stats():
    try:
        from scipy import stats
    except ImportError as e:  # pragma: no cover - depends on the installed extras
        raise SystemExit("this command needs the analysis extra: pip install 'bythebooktest[analysis]'") from e
    return stats


# --- 1. Uniformity of the rulebooks of a level -------------------------------------------------------------


def _between_kb_ss(correct_by_kb: dict[str, list[bool]]) -> float:
    """Sum over rulebooks of the squared distance between right answers and their share of the total."""
    total = sum(len(v) for v in correct_by_kb.values())
    right = sum(sum(v) for v in correct_by_kb.values())
    return sum((sum(v) - len(v) * right / total) ** 2 for v in correct_by_kb.values())


def uniformity(claims: list[ClaimRecord], ans: dict[str, str | None], rng: random.Random) -> dict:
    """How much the accuracy of the rulebooks of one level differs, against two references.

    ``binomial``: χ² = Σ (k − n·p)² / (n·p·(1 − p)) over the K rulebooks, with K − 1 degrees of freedom;
    ``lower_tail`` is the probability of a value this small or smaller if claims were independent draws.
    ``permutation``: the same sum of squares (``between_ss``) after shuffling the rulebook of claims that
    share a specification; ``lower_tail`` is the share of shuffles with a sum of squares no larger than
    the observed one. A small value there means that rulebooks agree even more than their identical
    composition explains.
    """
    stats = _stats()
    by_kb: dict[str, list[ClaimRecord]] = defaultdict(list)
    for c in claims:
        by_kb[c.kb_id].append(c)
    kbs = sorted(by_kb)
    correct = {c.claim_id: ans[c.claim_id] == c.expected for c in claims}
    k = {kb: sum(correct[c.claim_id] for c in by_kb[kb]) for kb in kbs}
    n = {kb: len(by_kb[kb]) for kb in kbs}
    p = sum(k.values()) / sum(n.values())
    chi2 = sum((k[kb] - n[kb] * p) ** 2 / (n[kb] * p * (1 - p)) for kb in kbs) if 0 < p < 1 else math.nan
    df = len(kbs) - 1
    observed = _between_kb_ss({kb: [correct[c.claim_id] for c in by_kb[kb]] for kb in kbs})

    # Stratified permutation: within each specification, the rulebook labels are shuffled.
    strata: dict[Spec, list[ClaimRecord]] = defaultdict(list)
    for c in claims:
        strata[spec_of(c)].append(c)
    labels = {s: [c.kb_id for c in cs] for s, cs in strata.items()}
    values = {s: [correct[c.claim_id] for c in cs] for s, cs in strata.items()}
    not_larger = 0
    for _ in range(PERMUTATIONS):
        groups: dict[str, list[bool]] = defaultdict(list)
        for s in strata:
            shuffled = labels[s][:]
            rng.shuffle(shuffled)
            for kb, v in zip(shuffled, values[s]):
                groups[kb].append(v)
        not_larger += _between_kb_ss(groups) <= observed + 1e-12

    same = [cs for cs in strata.values() if len(cs) > 1]
    claims_in = sum(len(cs) for cs in same)
    always_same_answer = sum(len(cs) for cs in same if len({ans[c.claim_id] for c in cs}) == 1)
    always_same_outcome = sum(len(cs) for cs in same if len({correct[c.claim_id] for c in cs}) == 1)
    return {
        "per_kb": {kb: {"right": k[kb], "n": n[kb]} for kb in kbs},
        "accuracy": p,
        "binomial": {
            "chi2": chi2,
            "df": df,
            "lower_tail": float(stats.chi2.cdf(chi2, df)) if not math.isnan(chi2) else math.nan,
            "sd_expected": [math.sqrt(n[kb] * p * (1 - p)) for kb in kbs],
            "sd_observed": _sd([k[kb] for kb in kbs]),
        },
        "permutation": {"between_ss": observed, "lower_tail": not_larger / PERMUTATIONS, "shuffles": PERMUTATIONS},
        "specifications": {
            "count": len(strata),
            "with_more_than_one_claim": len(same),
            "claims_in_them": claims_in,
            "same_answer_share": always_same_answer / claims_in if claims_in else math.nan,
            "same_outcome_share": always_same_outcome / claims_in if claims_in else math.nan,
        },
    }


def _sd(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else math.nan


# --- 2. Intervals from the per-rulebook values ------------------------------------------------------------


def t_interval(values: list[float]) -> dict:
    """Mean of the per-rulebook values with a 95% Student t interval on K − 1 degrees of freedom."""
    stats = _stats()
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return {"value": math.nan, "ci95": [math.nan, math.nan], "values": values}
    mean = sum(vals) / len(vals)
    half = float(stats.t.ppf(0.975, len(vals) - 1)) * _sd(vals) / math.sqrt(len(vals))
    return {"value": mean, "ci95": [mean - half, mean + half], "values": values, "df": len(vals) - 1}


def per_rulebook(claims: list[ClaimRecord], responses: dict[str, dict]) -> dict:
    by_kb: dict[str, list[ClaimRecord]] = defaultdict(list)
    for c in claims:
        by_kb[c.kb_id].append(c)
    per = {kb: metrics.measures(cs, responses, lenient=metrics.MAIN == "lenient") for kb, cs in sorted(by_kb.items())}
    return {
        name: t_interval([m[key]["value"] for m in per.values()])
        for name, key in (
            ("accuracy", "accuracy"),
            ("conflict_effect", "conflict_effect"),
            ("silence_effect", "silence_effect"),
        )
    } | {"kbs": sorted(per)}


# --- 3. Confidence: discrimination and adaptive calibration ------------------------------------------------


def adaptive_ece(conf: list[float], right: list[bool], bins: int = BINS) -> float:
    """Expected calibration error with bins of equal size, in order of confidence."""
    order = sorted(range(len(conf)), key=lambda i: conf[i])
    total = len(order)
    ece = 0.0
    for b in range(bins):
        idx = order[b * total // bins : (b + 1) * total // bins]
        if idx:
            ece += abs(sum(right[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx)) * len(idx)
    return ece / total if total else math.nan


def confidence(claims: list[ClaimRecord], responses: dict[str, dict], ans: dict, rng: random.Random) -> dict:
    from sklearn.metrics import roc_auc_score

    rows = [
        (ans[c.claim_id] == c.expected, responses[c.claim_id]["confidence"])
        for c in claims
        if responses.get(c.claim_id, {}).get("status") == "ok" and responses[c.claim_id].get("probs")
    ]
    if not rows:
        return {}
    right = [r for r, _ in rows]
    conf = [s for _, s in rows]
    fixed = [[0, 0] for _ in range(BINS)]
    for r, s in rows:
        fixed[min(int(s * BINS), BINS - 1)][0] += 1
        fixed[min(int(s * BINS), BINS - 1)][1] += r
    out: dict = {
        "n": len(rows),
        "adaptive_ece": adaptive_ece(conf, right),
        "fixed_bins": [{"from": i / BINS, "to": (i + 1) / BINS, "n": n, "right": k} for i, (n, k) in enumerate(fixed)],
    }
    if len(set(right)) < 2:
        out["auroc"] = {"value": math.nan, "ci95": [math.nan, math.nan]}
        return out
    value = float(roc_auc_score(right, conf))
    boot = []
    for _ in range(BOOTSTRAP):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        ys = [r for r, _ in sample]
        if len(set(ys)) == 2:
            boot.append(float(roc_auc_score(ys, [s for _, s in sample])))
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[min(len(boot) - 1, int(0.975 * len(boot)))]
    out["auroc"] = {"value": value, "ci95": [lo, hi], "bootstrap": len(boot)}
    return out


# --- 4. Depth by length -------------------------------------------------------------------------------------


def depth_by_length(claims: list[ClaimRecord], ans: dict) -> dict:
    """Accuracy on TRUE or FALSE claims, by level and number of steps."""
    cells: dict[tuple[int, int], list[ClaimRecord]] = defaultdict(list)
    for c in claims:
        if c.expected != "UNKNOWN":
            cells[(c.level, c.depth)].append(c)
    out: dict = defaultdict(dict)
    for (level, depth), cs in sorted(cells.items()):
        out[str(level)][str(depth)] = metrics.proportion(sum(ans[c.claim_id] == c.expected for c in cs), len(cs))
    return dict(out)


# --- 5. Fixed rules -----------------------------------------------------------------------------------------


def world_answer(c: ClaimRecord) -> str | None:
    """What is true in the real world for a claim about real facts; ``None`` for invented words."""
    if c.world_relation == "agree":
        return c.expected
    if c.world_relation == "conflict":
        return "FALSE" if c.expected == "TRUE" else "TRUE"
    if c.world_relation == "unsupported_true":
        return "TRUE"
    if c.world_relation == "unsupported_false":
        return "FALSE"
    return None


def fixed_rules(claims: list[ClaimRecord], ans: dict) -> dict:
    """Accuracy of "believe the world" and "contradict the world" on the claims about real facts, by
    relation with the world and in total, beside the accuracy of the model on the same claims.
    """
    real = [c for c in claims if world_answer(c) is not None]
    opposite = {"TRUE": "FALSE", "FALSE": "TRUE"}

    def score(subset: list[ClaimRecord]) -> dict:
        return {
            "believe_world": metrics.proportion(sum(world_answer(c) == c.expected for c in subset), len(subset)),
            "contradict_world": metrics.proportion(
                sum(opposite[world_answer(c) or ""] == c.expected for c in subset), len(subset)
            ),
            "model": metrics.proportion(sum(ans[c.claim_id] == c.expected for c in subset), len(subset)),
        }

    relations = ("conflict", "agree", "unsupported_true", "unsupported_false")
    return {"all_real": score(real)} | {r: score([c for c in real if c.world_relation == r]) for r in relations}


# --- 6. Answer preference -----------------------------------------------------------------------------------


def answer_preference(claims: list[ClaimRecord], ans: dict) -> dict:
    """Share of each answer on TRUE or FALSE claims, for invented words, conflict and agreement."""
    out = {}
    for name, rel in (("invented_words", "neutral"), ("conflict", "conflict"), ("agree", "agree")):
        cs = [c for c in claims if c.world_relation == rel and c.expected != "UNKNOWN"]
        out[name] = {
            "n": len(cs),
            "share": {lab: metrics.proportion(sum(ans[c.claim_id] == lab for c in cs), len(cs)) for lab in LABELS},
            "accuracy_by_expected": {
                lab: metrics.proportion(
                    sum(ans[c.claim_id] == lab for c in cs if c.expected == lab),
                    sum(c.expected == lab for c in cs),
                )
                for lab in ("TRUE", "FALSE")
            },
        }
    return out


# --- Run ------------------------------------------------------------------------------------------------------


def compute(run_dirs: list[Path], root: Path) -> dict:
    planned: dict[str, dict[str, ClaimRecord]] = defaultdict(dict)
    answers: dict[str, dict[str, dict]] = defaultdict(dict)
    run_ids = []
    for run_dir in run_dirs:
        cfg, per_model = metrics.run_claims(run_dir, root)
        run_ids.append(cfg["run_id"])
        for name, claims in per_model.items():
            planned[name].update({c.claim_id: c for c in claims})
            answers[name].update(metrics.load_responses(run_dir / "responses" / f"{name}.jsonl"))
    result: dict = {"run_id": "+".join(run_ids), "seed": SEED, "models": {}}
    for name, by_id in planned.items():
        mine = list(by_id.values())
        responses = answers[name]
        ans = _answers(mine, responses)
        rng = random.Random(SEED)
        levels = {}
        for level in sorted({c.level for c in mine}):
            sub = [c for c in mine if c.level == level]
            levels[str(level)] = {
                "uniformity": uniformity(sub, ans, rng),
                "per_rulebook": per_rulebook(sub, responses),
                "confidence": confidence(sub, responses, ans, rng),
                "fixed_rules": fixed_rules(sub, ans),
                "answer_preference": answer_preference(sub, ans),
            }
        result["models"][name] = {"levels": levels, "depth_by_length": depth_by_length(mine, ans)}
    return result


# --- Summary --------------------------------------------------------------------------------------------------


def _pct(p: dict) -> str:
    if not p.get("n"):
        return "-"
    lo, hi = p["ci95"]
    return f"{p['value']:.1%} [{lo:.1%}, {hi:.1%}] ({p['k']}/{p['n']})"


def _pts(d: dict) -> str:
    if math.isnan(d["value"]):
        return "-"
    lo, hi = d["ci95"]
    return f"{d['value'] * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"


def summary_md(result: dict) -> str:
    out = [f"# Analyses of run {result['run_id']}", "", f"Seed {result['seed']}. Lenient reading of the answers.", ""]
    for name, m in result["models"].items():
        lv = m["levels"]
        out += [f"## {name}", "", "### 1. Uniformity of the rulebooks of a level", ""]
        out += [
            "| Level | Right per rulebook | SD expected / observed | χ² (df) | P(as uniform), binomial | "
            "P(as uniform), shuffling within specification | Same answer within a specification |",
            "|---|---|---|---|---|---|---|",
        ]
        for level, e in lv.items():
            u = e["uniformity"]
            b = u["binomial"]
            ks = ", ".join(f"{v['right']}/{v['n']}" for v in u["per_kb"].values())
            sd_exp = sum(b["sd_expected"]) / len(b["sd_expected"])
            out.append(
                f"| {level} | {ks} | {sd_exp:.1f} / {b['sd_observed']:.1f} | {b['chi2']:.3f} ({b['df']}) | "
                f"{b['lower_tail']:.3%} | {u['permutation']['lower_tail']:.1%} | "
                f"{u['specifications']['same_answer_share']:.1%} of {u['specifications']['claims_in_them']} |"
            )
        out += ["", "### 2. Intervals from the per-rulebook values (t, K − 1 df)", ""]
        out += ["| Level | Accuracy | Conflict effect | Silence effect |", "|---|---|---|---|"]
        for level, e in lv.items():
            r = e["per_rulebook"]
            out.append(
                f"| {level} | {_pts(r['accuracy'])} | {_pts(r['conflict_effect'])} | {_pts(r['silence_effect'])} |"
            )
        out += [
            "",
            "### 3. Confidence",
            "",
            "| Level | AUROC [bootstrap 95%] | ECE, 10 equal-size bins |",
            "|---|---|---|",
        ]
        for level, e in lv.items():
            c = e["confidence"]
            if c:
                a = c["auroc"]
                lo, hi = a["ci95"]
                out.append(f"| {level} | {a['value']:.3f} [{lo:.3f}, {hi:.3f}] | {c['adaptive_ece']:.3f} |")
        out += ["", "### 4. Accuracy on TRUE or FALSE claims, by steps and length", ""]
        depths = sorted({int(d) for v in m["depth_by_length"].values() for d in v})
        out += ["| Steps | " + " | ".join(f"Level {lv_}" for lv_ in m["depth_by_length"]) + " |"]
        out += ["|---|" + "---|" * len(m["depth_by_length"])]
        for d in depths:
            cells = [_pct(v[str(d)]) if str(d) in v else "-" for v in m["depth_by_length"].values()]
            out.append(f"| {d} | " + " | ".join(cells) + " |")
        out += ["", "### 5. Fixed rules on the claims about real facts", ""]
        out += ["| Level | Claims | Believe the world | Contradict the world | Model |", "|---|---|---|---|---|"]
        for level, e in lv.items():
            for rel, f in e["fixed_rules"].items():
                rules = f"{_pct(f['believe_world'])} | {_pct(f['contradict_world'])}"
                out.append(f"| {level} | {rel} | {rules} | {_pct(f['model'])} |")
        out += ["", "### 6. Answers on TRUE or FALSE claims", ""]
        out += ["| Level | Claims | TRUE | FALSE | UNKNOWN | Right when TRUE | Right when FALSE |"]
        out += ["|---|---|---|---|---|---|---|"]
        for level, e in lv.items():
            for grp, a in e["answer_preference"].items():
                s, r = a["share"], a["accuracy_by_expected"]
                out.append(
                    f"| {level} | {grp} ({a['n']}) | {_pct(s['TRUE'])} | {_pct(s['FALSE'])} | {_pct(s['UNKNOWN'])} | "
                    f"{_pct(r['TRUE'])} | {_pct(r['FALSE'])} |"
                )
        out.append("")
    return "\n".join(out)


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``analysis`` command."""
    p = sub.add_parser(
        "analysis",
        help="extra analyses of one or more runs, from the saved answers",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dirs", type=Path, nargs="+", help="one or more run folders, merged by model")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="folder that dataset paths start from")
    p.add_argument("--out", type=Path, help="output folder (default: <first run>/analysis)")
    p.set_defaults(func=command)


def command(args: argparse.Namespace) -> int:
    result = compute(args.run_dirs, args.root)
    out = args.out or args.run_dirs[0] / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "analysis.md").write_text(summary_md(result), encoding="utf-8")
    print(summary_md(result))
    return 0
