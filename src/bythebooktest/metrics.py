"""Measures computed from the recorded answers of one or more runs.

Usage::

    bythebooktest metrics <run folder> [<run folder> ...] [--root <repository>] [--out <folder>]

Writes ``metrics.json`` (every measure, by model and level) and ``summary.md`` (the main tables).

- The main measures use the **lenient** reading of the answer; the **exact** reading, which also
  measures compliance with the answer format, is kept under ``exact`` and ``all_exact``.
- Every proportion has its 95% Wilson interval; differences between two proportions have the Newcombe
  interval built from the two Wilson intervals.
- Invalid answers and claims without an answer count as errors.
- All measures are oriented so that higher is better, except the effects, where 0 is ideal and a
  negative value means that world knowledge interferes with the rulebook.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from .records import ClaimRecord

Z = 1.959964
"""Normal quantile for 95% intervals."""
MAIN = "lenient"
"""Reading used by the main measures."""
LABELS = ("TRUE", "FALSE", "UNKNOWN")


# --- Proportions and intervals ------------------------------------------------------------------------


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for ``k`` successes out of ``n``."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion(k: int, n: int) -> dict:
    """``k / n`` with its Wilson interval, as ``{"value", "ci95", "k", "n"}``."""
    lo, hi = wilson(k, n)
    return {"value": k / n if n else math.nan, "ci95": [lo, hi], "k": k, "n": n}


def difference(a: dict, b: dict) -> dict:
    """``a - b`` for two proportions, with the Newcombe hybrid score interval (method 10)."""
    pa, pb = a["value"], b["value"]
    (la, ua), (lb, ub) = a["ci95"], b["ci95"]
    d = pa - pb
    lo = d - math.sqrt((pa - la) ** 2 + (ub - pb) ** 2)
    hi = d + math.sqrt((ua - pa) ** 2 + (pb - lb) ** 2)
    return {"value": d, "ci95": [lo, hi]}


# --- Measures on a set of claims -------------------------------------------------------------------------


def silence_effect(
    true_in_world: list[ClaimRecord], false_in_world: list[ClaimRecord], ans: dict[str, str | None]
) -> dict:
    """How far the model fills the silence of the rulebook with what it knows.

    On UNKNOWN claims about real facts, the share of answers against the real world minus the share in
    the direction of the real world (TRUE for a true fact, FALSE for a false one). 0 is ideal; negative
    means world knowledge leaks into the answers. The 95% interval uses the normal approximation on a
    per-claim score: -1 toward the world, +1 against it, 0 otherwise.
    """
    toward = sum(ans[c.claim_id] == "TRUE" for c in true_in_world) + sum(
        ans[c.claim_id] == "FALSE" for c in false_in_world
    )
    away = sum(ans[c.claim_id] == "FALSE" for c in true_in_world) + sum(
        ans[c.claim_id] == "TRUE" for c in false_in_world
    )
    n = len(true_in_world) + len(false_in_world)
    if n == 0:
        return {"value": math.nan, "ci95": [math.nan, math.nan], "toward": 0, "away": 0, "n": 0}
    d = (away - toward) / n
    se = math.sqrt(max(0.0, (toward + away) / n - d * d) / n)
    return {"value": d, "ci95": [max(-1.0, d - Z * se), min(1.0, d + Z * se)], "toward": toward, "away": away, "n": n}


def _answer(rec: dict | None, lenient: bool) -> str | None:
    if rec is None or rec.get("status") != "ok":
        return None
    return rec.get("answer_lenient" if lenient else "answer_exact")


def measures(claims: list[ClaimRecord], responses: dict[str, dict], lenient: bool = False) -> dict:
    """All the measures on one set of claims; a claim without an answer counts as an error.

    The **conflict effect** compares claims that contradict the real world with claims that agree with
    it (real words in both); the **invented-words effect** compares conflict claims with TRUE or FALSE
    claims made of invented words only.
    """
    ans = {c.claim_id: _answer(responses.get(c.claim_id), lenient) for c in claims}

    def acc(subset: list[ClaimRecord]) -> dict:
        return proportion(sum(ans[c.claim_id] == c.expected for c in subset), len(subset))

    out: dict = {"n": len(claims), "accuracy": acc(claims)}
    out["per_label"] = {}
    for lab in LABELS:
        predicted = [c for c in claims if ans[c.claim_id] == lab]
        actual = [c for c in claims if c.expected == lab]
        tp = sum(c.expected == lab for c in predicted)
        out["per_label"][lab] = {"precision": proportion(tp, len(predicted)), "recall": proportion(tp, len(actual))}
    conflict = acc([c for c in claims if c.world_relation == "conflict"])
    agree = acc([c for c in claims if c.world_relation == "agree"])
    neutral = [c for c in claims if c.world_relation == "neutral"]
    neutral_tf = acc([c for c in neutral if c.expected != "UNKNOWN"])
    out["conflict_accuracy"] = conflict
    out["agree_accuracy"] = agree
    out["neutral_accuracy"] = acc(neutral)
    out["neutral_tf_accuracy"] = neutral_tf
    out["neutral_unknown_accuracy"] = acc([c for c in neutral if c.expected == "UNKNOWN"])
    out["conflict_effect"] = difference(conflict, agree)
    out["invented_words_effect"] = difference(conflict, neutral_tf)
    ut = [c for c in claims if c.world_relation == "unsupported_true"]
    uf = [c for c in claims if c.world_relation == "unsupported_false"]
    out["intrusion_true"] = proportion(sum(ans[c.claim_id] == "TRUE" for c in ut), len(ut))
    out["intrusion_false"] = proportion(sum(ans[c.claim_id] == "FALSE" for c in uf), len(uf))
    out["silence_effect"] = silence_effect(ut, uf, ans)
    # The same measures without the modal UNKNOWN claims ("can" against "does"), reported apart because
    # they test one reading rule only.
    rest = [c for c in claims if not (c.group == "modal" and c.expected == "UNKNOWN")]
    out["without_modal_unknown"] = {
        "n": len(rest),
        "accuracy": acc(rest),
        "unknown_recall": acc([c for c in rest if c.expected == "UNKNOWN"]),
        "modal_unknown": acc([c for c in claims if c.group == "modal" and c.expected == "UNKNOWN"]),
    }
    out["invalid"] = proportion(
        sum(responses.get(c.claim_id, {}).get("status") == "ok" and ans[c.claim_id] is None for c in claims),
        len(claims),
    )
    out["not_completed"] = proportion(
        sum(responses.get(c.claim_id, {}).get("status") != "ok" for c in claims), len(claims)
    )
    out["budget_hit"] = proportion(
        sum(bool(responses.get(c.claim_id, {}).get("budget_hit")) for c in claims), len(claims)
    )
    out["confusion"] = {
        e: {str(a): sum(1 for c in claims if c.expected == e and ans[c.claim_id] == a) for a in (*LABELS, None)}
        for e in LABELS
    }
    out.update(label_measures(claims, ans))
    out["calibration"] = calibration(claims, responses, ans)
    kbs: dict = defaultdict(list)
    for c in claims:
        kbs[c.kb_id].append(c)
    out["per_kb"] = {k: acc(v) for k, v in sorted(kbs.items())}
    return out


def label_measures(claims: list[ClaimRecord], ans: dict[str, str | None]) -> dict:
    """Share of each answer, the three kinds of error, Cohen's kappa and macro-F1."""
    n = len(claims)
    tf = [c for c in claims if c.expected != "UNKNOWN"]
    unk = [c for c in claims if c.expected == "UNKNOWN"]
    out: dict = {"answer_share": {lab: proportion(sum(ans[c.claim_id] == lab for c in claims), n) for lab in LABELS}}
    out["answerable_accuracy"] = proportion(sum(ans[c.claim_id] == c.expected for c in tf), len(tf))
    out["unknown_recall"] = proportion(sum(ans[c.claim_id] == "UNKNOWN" for c in unk), len(unk))
    out["errors"] = {
        # TRUE instead of FALSE or the other way round, on TRUE or FALSE claims
        "inversion": proportion(
            sum(ans[c.claim_id] in ("TRUE", "FALSE") and ans[c.claim_id] != c.expected for c in tf), len(tf)
        ),
        # UNKNOWN when the answer was TRUE or FALSE
        "abstention": proportion(sum(ans[c.claim_id] == "UNKNOWN" for c in tf), len(tf)),
        # TRUE or FALSE when the answer was UNKNOWN
        "invention": proportion(sum(ans[c.claim_id] in ("TRUE", "FALSE") for c in unk), len(unk)),
    }
    if n:
        po = sum(ans[c.claim_id] == c.expected for c in claims) / n
        pe = sum(
            (sum(c.expected == lab for c in claims) / n) * (sum(ans[c.claim_id] == lab for c in claims) / n)
            for lab in LABELS
        )
        out["kappa"] = (po - pe) / (1 - pe) if pe < 1 else math.nan
        out["chance_accuracy"] = pe
    else:
        out["kappa"] = out["chance_accuracy"] = math.nan
    f1 = []
    for lab in LABELS:
        tp = sum(ans[c.claim_id] == lab and c.expected == lab for c in claims)
        pred = sum(ans[c.claim_id] == lab for c in claims)
        real = sum(c.expected == lab for c in claims)
        p, r = (tp / pred if pred else 0.0), (tp / real if real else 0.0)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    out["macro_f1"] = sum(f1) / len(f1)
    return out


def calibration(claims: list[ClaimRecord], responses: dict[str, dict], ans: dict[str, str | None]) -> dict:
    """Brier score, Brier skill score, ECE over 10 bins, and mean confidence on right and wrong answers.

    Only for models that return the probability of each option (decision models); empty otherwise. The
    Brier skill score compares with uniform probabilities (Brier 2/3 with three options).
    """
    rows = [
        (c, responses[c.claim_id])
        for c in claims
        if responses.get(c.claim_id, {}).get("status") == "ok" and responses[c.claim_id].get("probs")
    ]
    if not rows:
        return {}
    brier = sum(sum((r["probs"].get(lab, 0.0) - (lab == c.expected)) ** 2 for lab in LABELS) for c, r in rows) / len(
        rows
    )
    bins: dict = defaultdict(lambda: [0, 0, 0.0])
    right: list[float] = []
    wrong: list[float] = []
    for c, r in rows:
        ok = ans[c.claim_id] == c.expected
        (right if ok else wrong).append(r["confidence"])
        b = bins[min(int(r["confidence"] * 10), 9)]
        b[0] += 1
        b[1] += ok
        b[2] += r["confidence"]
    ece = sum(abs(k / n - s / n) * n for n, k, s in bins.values()) / len(rows)
    return {
        "n": len(rows),
        "brier": brier,
        "brier_skill": 1 - brier / (2 / 3),
        "ece": ece,
        "confidence_right": _mean(right),
        "confidence_wrong": _mean(wrong),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def curves(claims: list[ClaimRecord], responses: dict[str, dict]) -> dict:
    """Accuracy by depth, by reasoning group and by fact category; ``depth_tf`` uses TRUE or FALSE
    claims only ("same text, more steps").
    """
    ans = {c.claim_id: _answer(responses.get(c.claim_id), MAIN == "lenient") for c in claims}
    result: dict = {}
    keys: tuple[tuple[str, Callable[[ClaimRecord], object]], ...] = (
        ("depth", lambda c: c.depth),
        ("group", lambda c: c.group),
        ("category", lambda c: c.category or "-"),
    )
    for name, key in keys:
        buckets: dict = defaultdict(list)
        for c in claims:
            buckets[key(c)].append(c)
        result[name] = {
            str(k): proportion(sum(ans[c.claim_id] == c.expected for c in v), len(v))
            for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))
        }
    tf = [c for c in claims if c.expected != "UNKNOWN"]
    by_depth: dict = defaultdict(list)
    for c in tf:
        by_depth[c.depth].append(c)
    result["depth_tf"] = {
        str(d): proportion(sum(ans[c.claim_id] == c.expected for c in v), len(v)) for d, v in sorted(by_depth.items())
    }
    return result


def same_question(claims: list[ClaimRecord], responses: dict[str, dict]) -> dict:
    """Same question, longer text: for each depth, the groups with TRUE or FALSE claims at that depth in
    every level that has it, and the accuracy of each level on those groups.
    """
    ans = {c.claim_id: _answer(responses.get(c.claim_id), MAIN == "lenient") for c in claims}
    tf = [c for c in claims if c.expected != "UNKNOWN"]
    out: dict = {}
    for depth in sorted({c.depth for c in tf}):
        at = [c for c in tf if c.depth == depth]
        levels = sorted({c.level for c in at})
        if len(levels) < 2:
            continue
        groups = set.intersection(*({c.group for c in at if c.level == lv} for lv in levels))
        if not groups:
            continue
        out[str(depth)] = {
            "groups": sorted(groups),
            "levels": {
                str(lv): proportion(
                    sum(ans[c.claim_id] == c.expected for c in at if c.level == lv and c.group in groups),
                    sum(1 for c in at if c.level == lv and c.group in groups),
                )
                for lv in levels
            },
        }
    return out


def secondary(claims: list[ClaimRecord], responses: dict[str, dict]) -> dict:
    """Mean time, tokens and server speeds of the answered claims."""
    ok = [responses[c.claim_id] for c in claims if responses.get(c.claim_id, {}).get("status") == "ok"]
    if not ok:
        return {}
    return {
        "wall_s_mean": _mean([r["wall_s"] for r in ok]),
        "reasoning_tokens_mean": _mean([r["reasoning_tokens"] for r in ok if r.get("reasoning_tokens") is not None]),
        "completion_tokens_mean": _mean([r["completion_tokens"] for r in ok if r.get("completion_tokens")]),
        "predicted_per_second_mean": _mean([r["timings"].get("predicted_per_second", math.nan) for r in ok]),
        "prompt_per_second_mean": _mean([r["timings"].get("prompt_per_second", math.nan) for r in ok]),
        "cache_n_mean": _mean([r["timings"].get("cache_n", math.nan) for r in ok]),
    }


def equal_weight(per_level: dict) -> dict:
    """Overall accuracy with the same weight for every level (mean of the level accuracies).

    The 95% interval uses the normal approximation: variance = sum of p(1-p)/n, divided by L².
    """
    accs = [lv[MAIN]["accuracy"] for lv in per_level.values() if lv[MAIN]["accuracy"]["n"]]
    if not accs:
        return {"value": math.nan, "ci95": [math.nan, math.nan], "levels": 0}
    value = sum(a["value"] for a in accs) / len(accs)
    se = math.sqrt(sum(a["value"] * (1 - a["value"]) / a["n"] for a in accs)) / len(accs)
    return {"value": value, "ci95": [max(0.0, value - Z * se), min(1.0, value + Z * se)], "levels": len(accs)}


# --- Reading runs ---------------------------------------------------------------------------------------


def load_claims(root: Path, datasets: list[str]) -> list[ClaimRecord]:
    return [
        ClaimRecord.model_validate_json(line)
        for d in datasets
        for line in (root / d / "claims.jsonl").open(encoding="utf-8")
    ]


def load_responses(path: Path) -> dict[str, dict]:
    """The last response of each claim (later lines replace earlier ones, as after a retry)."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["claim_id"]] = rec
    return out


def run_claims(run_dir: Path, root: Path) -> tuple[dict, dict[str, list[ClaimRecord]]]:
    """The configuration of a run and, for each model, the claims it had to answer (those without an
    answer included: they count as errors).
    """
    from .run import load_dataset  # imported here: run pulls in the server code, not needed otherwise

    cfg = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["key"]["config"]
    _, tasks, _ = load_dataset(
        [root / d for d in cfg["datasets"]],
        cfg.get("claims_per_kb"),
        cfg.get("kbs"),
        cfg.get("levels"),
        cfg.get("fraction"),
    )
    claims = [t.claim for t in tasks]
    return cfg, {m["name"]: claims for m in cfg["models"]}


def compute(run_dirs: Path | list[Path], root: Path) -> dict:
    """All measures from one or more runs (for example one per level), merged by model."""
    if isinstance(run_dirs, Path):
        run_dirs = [run_dirs]
    planned: dict[str, dict[str, ClaimRecord]] = defaultdict(dict)
    answers: dict[str, dict[str, dict]] = defaultdict(dict)
    run_ids = []
    for run_dir in run_dirs:
        cfg, per_model = run_claims(run_dir, root)
        run_ids.append(cfg["run_id"])
        for name, claims in per_model.items():
            for c in claims:
                if c.claim_id in planned[name]:
                    raise SystemExit(f"{name}: claim {c.claim_id} appears in more than one run")
                planned[name][c.claim_id] = c
            answers[name].update(load_responses(run_dir / "responses" / f"{name}.jsonl"))
    result: dict = {"run_id": "+".join(run_ids), "models": {}}
    for name in planned:
        mine = list(planned[name].values())
        responses = answers[name]
        per_level = {}
        for level in sorted({c.level for c in mine}):
            sub = [c for c in mine if c.level == level]
            per_level[str(level)] = {
                "exact": measures(sub, responses),
                "lenient": measures(sub, responses, lenient=True),
                "curves": curves(sub, responses),
                "secondary": secondary(sub, responses),
            }
        result["models"][name] = {
            "answered": sum(c.claim_id in responses for c in mine),
            "levels": per_level,
            "same_question": same_question(mine, responses),
            "all": measures(mine, responses, lenient=MAIN == "lenient"),
            "all_exact": measures(mine, responses),
            "all_equal_weight": equal_weight(per_level),
        }
    return result


# --- Summary tables ------------------------------------------------------------------------------------


def _pct(p: dict) -> str:
    if p.get("n") == 0 or math.isnan(p["value"]):
        return "-"
    lo, hi = p["ci95"]
    return f"{p['value']:.1%} [{lo:.1%}, {hi:.1%}]"


def _diff(d: dict) -> str:
    if math.isnan(d["value"]):
        return "-"
    return f"{d['value']:+.1%} [{d['ci95'][0]:+.1%}, {d['ci95'][1]:+.1%}]"


def _share(e: dict, lab: str) -> str:
    return f"{e['answer_share'][lab]['value']:.0%}"


def summary_md(result: dict) -> str:
    """The main measures as Markdown tables."""
    lines = [
        f"# Run {result['run_id']}: main measures",
        "",
        "Lenient reading of the answers; the exact reading also measures compliance with the answer "
        "format. In brackets, the 95% confidence interval. The conflict effect compares claims that "
        "contradict the real world with claims that agree with it; the silence effect, on UNKNOWN "
        "claims about real facts, is the share of answers against the real world minus the share in "
        "its direction. For both, 0 is ideal and a negative value means that world knowledge "
        "interferes. The invented-words effect compares conflict claims with TRUE or FALSE claims "
        "made of invented words.",
        "",
        "| Model | Level | Claims | Accuracy | Conflict effect | Silence effect | "
        "Invented-words effect | Intrusion (TRUE) | Intrusion (FALSE) | "
        "Accuracy when agreeing | Invalid | Thinking budget hit | Exact accuracy | Invalid (exact) |",
        "|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in result["models"].items():
        for level, lv in m["levels"].items():
            e, x = lv[MAIN], lv["exact"]
            lines.append(
                f"| {name} | {level} | {e['n']} | {_pct(e['accuracy'])} | {_diff(e['conflict_effect'])} | "
                f"{_diff(e['silence_effect'])} | {_diff(e['invented_words_effect'])} | "
                f"{_pct(e['intrusion_true'])} | {_pct(e['intrusion_false'])} | {_pct(e['agree_accuracy'])} | "
                f"{_pct(e['invalid'])} | {_pct(e['budget_hit'])} | {_pct(x['accuracy'])} | "
                f"{_pct(x['invalid'])} |"
            )
    lines += [
        "",
        "## Without the modal UNKNOWN claims",
        "",
        "| Model | Level | Claims | Accuracy | UNKNOWN recall | Modal UNKNOWN claims right |",
        "|---|---:|---:|---|---|---|",
    ]
    for name, m in result["models"].items():
        for level, lv in m["levels"].items():
            w = lv[MAIN]["without_modal_unknown"]
            mu = w["modal_unknown"]
            lines.append(
                f"| {name} | {level} | {w['n']} | {_pct(w['accuracy'])} | "
                f"{_pct(w['unknown_recall'])} | {mu['k']}/{mu['n']} |"
            )
    lines += [
        "",
        "## Same text, more steps (TRUE or FALSE claims)",
        "",
        "| Model | Level | Depth | Accuracy |",
        "|---|---:|---:|---|",
    ]
    for name, m in result["models"].items():
        for level, lv in m["levels"].items():
            for d, p in lv["curves"]["depth_tf"].items():
                lines.append(f"| {name} | {level} | {d} | {_pct(p)} |")
    lines += [
        "",
        "## Same question, longer text (TRUE or FALSE claims)",
        "",
        "| Model | Depth | Groups | Level | Accuracy |",
        "|---|---:|---|---:|---|",
    ]
    for name, m in result["models"].items():
        for d, s in m.get("same_question", {}).items():
            for level, p in s["levels"].items():
                lines.append(f"| {name} | {d} | {', '.join(s['groups'])} | {level} | {_pct(p)} |")
    lines += [
        "",
        "## The three answers",
        "",
        "Higher is better. Answerable claims: accuracy on TRUE or FALSE claims, and how they split into "
        "right / inverted (TRUE instead of FALSE or the other way round) / abstained (UNKNOWN). "
        "UNKNOWN recall: how often the model admits that the rulebook is silent.",
        "",
        "| Model | Level | Accuracy | Chance | Kappa | Answers T / F / U | Answerable claims | "
        "Right / inverted / abstained | UNKNOWN recall | Recall T / F / U | Precision T / F / U | "
        "Macro-F1 |",
        "|---|---:|---|---:|---:|---|---|---|---|---|---|---:|",
    ]
    for name, m in result["models"].items():
        for level, lv in m["levels"].items():
            e = lv[MAIN]
            pl = e["per_label"]
            rec = " / ".join(f"{pl[lab]['recall']['value']:.0%}" for lab in LABELS)
            prec = " / ".join(
                "-" if math.isnan(pl[lab]["precision"]["value"]) else f"{pl[lab]['precision']['value']:.0%}"
                for lab in LABELS
            )
            lines.append(
                f"| {name} | {level} | {_pct(e['accuracy'])} | {e['chance_accuracy']:.1%} | {e['kappa']:.3f} | "
                f"{' / '.join(_share(e, lab) for lab in LABELS)} | {_pct(e['answerable_accuracy'])} | "
                f"{e['answerable_accuracy']['value']:.0%} / {e['errors']['inversion']['value']:.0%} / "
                f"{e['errors']['abstention']['value']:.0%} | {_pct(e['unknown_recall'])} | {rec} | {prec} | "
                f"{e['macro_f1']:.3f} |"
            )
    lines += [
        "",
        "## Confidence and accuracy per rulebook",
        "",
        "| Model | Level | Brier skill score ↑ | Brier ↓ | ECE ↓ | Confidence, right / wrong | "
        "Accuracy per rulebook ↑ |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name, m in result["models"].items():
        for level, lv in m["levels"].items():
            e = lv[MAIN]
            cal = e["calibration"]
            kb = ", ".join(f"{k} {v['value']:.1%}" for k, v in e["per_kb"].items())
            if cal:
                lines.append(
                    f"| {name} | {level} | {cal['brier_skill']:.3f} | {cal['brier']:.3f} | "
                    f"{cal['ece']:.3f} | {cal['confidence_right']:.2f} / {cal['confidence_wrong']:.2f} | "
                    f"{kb} |"
                )
            else:
                lines.append(f"| {name} | {level} | - | - | - | - | {kb} |")
    lines += [
        "",
        "## Overall, by model",
        "",
        "Accuracy on the claims as they are (Level 3 weighs more) and with the same weight for every level.",
        "",
        "| Model | Claims | Answers | Accuracy | Accuracy, levels weighted equally | Conflict effect |",
        "|---|---:|---:|---|---|---|",
    ]
    for name, m in result["models"].items():
        a, w, ce = m["all"], m["all_equal_weight"], m["all"]["conflict_effect"]
        lines.append(
            f"| {name} | {a['n']} | {m['answered']} | {_pct(a['accuracy'])} | "
            f"{w['value']:.1%} [{w['ci95'][0]:.1%}, {w['ci95'][1]:.1%}] | {_diff(ce)} |"
        )
    return "\n".join(lines) + "\n"


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``metrics`` command."""
    p = sub.add_parser(
        "metrics",
        help="compute the measures of one or more runs",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dirs", type=Path, nargs="+", help="one or more run folders, merged by model")
    p.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="folder that dataset paths start from (default: the current folder)",
    )
    p.add_argument("--out", type=Path, help="output folder (default: <first run>/metrics)")
    p.set_defaults(func=command)


def command(args: argparse.Namespace) -> int:
    result = compute(args.run_dirs, args.root)
    out = args.out or args.run_dirs[0] / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_md(result), encoding="utf-8")
    print(summary_md(result))
    return 0
