"""Knowledge probe: which facts of the curated tables does the model know, with no rulebook at all?

Usage::

    bythebooktest probe build [--tables data/tables] [--out out/world-probe]
    bythebooktest run --config configs/d1-2b-world.toml
    bythebooktest probe analyze <probe run> <benchmark run> [--root <repository>]

``build`` writes a dataset in the format of the test program: an empty rulebook and, as claims, the
sentences written from the tables (for every fact its true and false versions, for every ability the
sentence with "can"), each in affirmative and negative form; the expected answer is the value in the real
world. The probe runs with ``task = "world"``.

``analyze`` decides which sentence pairs the model knows (right on the sentence and on its negation),
links the real-world claims of the benchmark to those pairs, and computes the conflict and silence
effects separately for known and unknown facts, with the difference between the two groups.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import render
from .metrics import MAIN, Z, _answer, difference, load_responses, proportion, run_claims, silence_effect
from .records import ClaimRecord, Expected, KBRecord
from .tables import load_tables

KB_ID = "W-KB1"
WORLD = ("conflict", "agree", "unsupported_true", "unsupported_false")
"""World relations of the benchmark claims that are about real facts."""
MIN_UNKNOWN_SHARE = 0.10
"""Below this share of unknown facts, known and unknown facts are not compared."""


def sentences(tables_dir: Path) -> list[dict]:
    """The probe sentences, in pairs (affirmative and negative).

    Every sentence carries its pair, its table row, its category and its value in the real world.
    """
    t = load_tables(tables_dir)
    out: list[dict] = []
    for f in t.facts:
        for i, (vp, true_in_world) in enumerate([(f.true, True)] + [(v, False) for v in f.false]):
            pair = f"{f.id}#{i}"
            for polarity, text, value in (("pos", vp.pos, true_in_world), ("neg", vp.neg, not true_in_world)):
                out.append(
                    {
                        "pair": pair,
                        "row": f.id,
                        "kind": "fact",
                        "category": f.category,
                        "polarity": polarity,
                        "text": render.fact(f.subject, text),
                        "world": value,
                    }
                )
    for a in t.abilities:
        subject = render.with_article(a.singular)
        pair = f"{a.id}#can"
        for polarity, phrase, value in (
            ("pos", f"can {a.verb}", a.world_can),
            ("neg", f"cannot {a.verb}", not a.world_can),
        ):
            out.append(
                {
                    "pair": pair,
                    "row": a.id,
                    "kind": "ability",
                    "category": a.category,
                    "polarity": polarity,
                    "text": render.fact(subject, phrase),
                    "world": value,
                    "singular": a.singular,
                    "plural": a.plural,
                    "verb": a.verb,
                    "third": a.third,
                }
            )
    return out


def build(tables_dir: Path, out_dir: Path) -> int:
    """Write the probe dataset in ``out_dir``."""
    items = sentences(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = KBRecord(kb_id=KB_ID, level=0, statements=[])
    (out_dir / "kbs.jsonl").write_text(kb.model_dump_json() + "\n", encoding="utf-8")
    with (out_dir / "claims.jsonl").open("w", encoding="utf-8") as f:
        for i, s in enumerate(items):
            expected: Expected = "TRUE" if s["world"] else "FALSE"
            # The pair goes in ``subtype`` and the polarity in ``logic``: the probe needs no logic.
            c = ClaimRecord(
                claim_id=f"W-{i:05d}",
                kb_id=KB_ID,
                level=0,
                group=f"world_{s['kind']}",
                subtype=s["pair"],
                text=s["text"],
                logic=s["polarity"],
                expected=expected,
                z3=expected,
                world_relation="agree",
                category=s["category"],
                depth=0,
                reasoning=[],
                support=[],
                traps=[],
            )
            f.write(c.model_dump_json() + "\n")
    pairs = len({s["pair"] for s in items})
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "what": "knowledge probe, no rulebook",
                "tables_sha256": load_tables(tables_dir).sha256,
                "sentences": len(items),
                "pairs": pairs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(items)} sentences ({pairs} pairs) in {out_dir}")
    return 0


def knowledge(probe_claims: list[ClaimRecord], responses: dict[str, dict]) -> dict[str, dict]:
    """For every pair: known if the model is right on the sentence and on its negation."""
    pairs: dict[str, dict] = defaultdict(lambda: {"answers": {}, "right": 0, "n": 0})
    for c in probe_claims:
        a = _answer(responses.get(c.claim_id), MAIN == "lenient")
        p = pairs[c.subtype]
        p["answers"][c.logic] = a
        p["right"] += a == c.expected
        p["n"] += 1
        p["category"] = c.category
        p["kind"] = c.group
    for p in pairs.values():
        p["known"] = p["n"] == 2 and p["right"] == 2
    return dict(pairs)


def link(claim: ClaimRecord, by_text: dict[str, str], abilities: list[dict]) -> str | None:
    """The sentence pair a benchmark claim is about: the sentence with the same text, or the ability it
    names (class and verb). ``None`` if the claim links to none (invented names).
    """
    t = claim.text.lower()
    if t in by_text:
        return by_text[t]
    hits = {
        a["pair"]
        for a in abilities
        if (f" {a['singular']} " in f" {t} " or f" {a['plural']} " in f" {t} ")
        and (f" {a['verb']}" in f" {t}" or f" {a['third']}" in f" {t}")
    }
    return hits.pop() if len(hits) == 1 else None


def _diff_of_diffs(a: dict, b: dict) -> dict:
    """``a - b`` for two differences, with the normal approximation: the standard error of each comes
    from half of its 95% interval.
    """
    if any(math.isnan(x) for x in (a["value"], b["value"], *a["ci95"], *b["ci95"])):
        return {"value": math.nan, "ci95": [math.nan, math.nan]}
    se = math.sqrt(((a["ci95"][1] - a["ci95"][0]) / (2 * Z)) ** 2 + ((b["ci95"][1] - b["ci95"][0]) / (2 * Z)) ** 2)
    d = a["value"] - b["value"]
    return {"value": d, "ci95": [d - Z * se, d + Z * se]}


def effects(claims: list[ClaimRecord], ans: dict[str, str | None]) -> dict:
    """Conflict and silence effects on a set of benchmark claims."""

    def acc(s: list[ClaimRecord]) -> dict:
        return proportion(sum(ans[c.claim_id] == c.expected for c in s), len(s))

    conflict = acc([c for c in claims if c.world_relation == "conflict"])
    agree = acc([c for c in claims if c.world_relation == "agree"])
    ut = [c for c in claims if c.world_relation == "unsupported_true"]
    uf = [c for c in claims if c.world_relation == "unsupported_false"]
    return {
        "n": len(claims),
        "conflict_accuracy": conflict,
        "agree_accuracy": agree,
        "conflict_effect": difference(conflict, agree),
        "silence_effect": silence_effect(ut, uf, ans),
    }


def analyze(probe_dir: Path, bench_dir: Path, root: Path, tables_dir: Path) -> dict:
    """Compare the benchmark effects on facts the model knows and on facts it does not know."""
    _, probe_models = run_claims(probe_dir, root)
    _, bench_models = run_claims(bench_dir, root)
    items = sentences(tables_dir)
    by_text = {s["text"].lower(): s["pair"] for s in items}
    abilities = [s for s in items if s["kind"] == "ability" and s["polarity"] == "pos"]
    result: dict = {"probe_run": probe_dir.name, "bench_run": bench_dir.name, "models": {}}
    for name, probe_claims in probe_models.items():
        probe_ans = load_responses(probe_dir / "responses" / f"{name}.jsonl")
        pairs = knowledge(probe_claims, probe_ans)
        m: dict = {
            "pairs": len(pairs),
            "known": proportion(sum(p["known"] for p in pairs.values()), len(pairs)),
            "sentences": proportion(sum(p["right"] for p in pairs.values()), sum(p["n"] for p in pairs.values())),
            "answers": {
                lab: sum(_answer(probe_ans.get(c.claim_id), True) == lab for c in probe_claims)
                for lab in ("TRUE", "FALSE", "UNKNOWN")
            },
        }
        cats: dict = defaultdict(list)
        for p in pairs.values():
            cats[(p["kind"], p["category"])].append(p["known"])
        m["by_category"] = {f"{k}/{c}": proportion(sum(v), len(v)) for (k, c), v in sorted(cats.items())}
        bench = bench_models.get(name)
        if bench is None:
            result["models"][name] = m
            continue
        responses = load_responses(bench_dir / "responses" / f"{name}.jsonl")
        ans = {c.claim_id: _answer(responses.get(c.claim_id), MAIN == "lenient") for c in bench}
        levels: dict = {}
        for level in sorted({c.level for c in bench}):
            world = [c for c in bench if c.level == level and c.world_relation in WORLD]
            pair_of = {c.claim_id: link(c, by_text, abilities) for c in world}
            linked = [(c, pairs.get(pair, {}).get("known", False)) for c in world if (pair := pair_of[c.claim_id])]
            known = [c for c, k in linked if k]
            unknown = [c for c, k in linked if not k]
            n_linked = len(known) + len(unknown)
            lv: dict[str, Any] = {
                "world_claims": len(world),
                "linked": n_linked,
                "not_linked": len(world) - n_linked,
                "known_share": proportion(len(known), n_linked),
                "known": effects(known, ans),
                "unknown": effects(unknown, ans),
            }
            lv["comparison"] = n_linked > 0 and len(unknown) / n_linked >= MIN_UNKNOWN_SHARE
            if lv["comparison"]:
                lv["difference"] = {
                    k: _diff_of_diffs(lv["known"][k], lv["unknown"][k]) for k in ("conflict_effect", "silence_effect")
                }
            levels[str(level)] = lv
        m["levels"] = levels
        result["models"][name] = m
    return result


def _pct(p: dict) -> str:
    if not p.get("n") or math.isnan(p["value"]):
        return "-"
    return f"{p['value']:.1%} [{p['ci95'][0]:.1%}, {p['ci95'][1]:.1%}] ({p['k']}/{p['n']})"


def _d(d: dict) -> str:
    if math.isnan(d["value"]):
        return "-"
    return f"{d['value']:+.1%} [{d['ci95'][0]:+.1%}, {d['ci95'][1]:+.1%}]"


def summary_md(r: dict) -> str:
    """The probe results as Markdown tables."""
    lines = [
        f"# Knowledge probe: {r['probe_run']} with {r['bench_run']}",
        "",
        "A sentence pair (affirmative and negative) is known if the model is right on both. Effects: "
        "0 is ideal, negative when world knowledge interferes.",
        "",
    ]
    for name, m in r["models"].items():
        a = m["answers"]
        lines += [
            f"## {name}",
            "",
            f"- Known pairs: {_pct(m['known'])}",
            f"- Right sentences: {_pct(m['sentences'])}",
            f"- Answers TRUE / FALSE / UNKNOWN: {a['TRUE']} / {a['FALSE']} / {a['UNKNOWN']}",
            "",
            "| Table / category | Known pairs |",
            "|---|---|",
        ]
        lines += [f"| {k} | {_pct(v)} |" for k, v in m["by_category"].items()]
        if "levels" not in m:
            continue
        lines += [
            "",
            "| Level | Real-world claims | Linked | Known share | Group | Conflict effect | Silence effect |",
            "|---:|---:|---:|---|---|---|---|",
        ]
        for level, lv in m["levels"].items():
            for g in ("known", "unknown"):
                e = lv[g]
                lines.append(
                    f"| {level} | {lv['world_claims']} | {lv['linked']} | {_pct(lv['known_share'])} | "
                    f"{g} ({e['n']}) | {_d(e['conflict_effect'])} | {_d(e['silence_effect'])} |"
                )
            if lv["comparison"]:
                d = lv["difference"]
                lines.append(f"| {level} | | | | difference | {_d(d['conflict_effect'])} | {_d(d['silence_effect'])} |")
            else:
                lines.append(
                    f"| {level} | | | | difference | not computed: unknown facts below {MIN_UNKNOWN_SHARE:.0%} | |"
                )
    return "\n".join(lines) + "\n"


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``probe`` command and its ``build`` and ``analyze`` subcommands."""
    p = sub.add_parser(
        "probe",
        help="knowledge probe: what the model knows with no rulebook",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    psub = p.add_subparsers(dest="probe_command", required=True)
    b = psub.add_parser("build", help="write the probe dataset")
    b.add_argument("--tables", type=Path, default=Path("data/tables"), help="fact tables (default: %(default)s)")
    b.add_argument("--out", type=Path, default=Path("out/world-probe"), help="output folder (default: %(default)s)")
    b.set_defaults(func=lambda args: build(args.tables, args.out))
    a = psub.add_parser("analyze", help="compare the benchmark effects on known and unknown facts")
    a.add_argument("probe_run", type=Path, help="run folder of the probe")
    a.add_argument("bench_run", type=Path, help="run folder of the benchmark")
    a.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="folder that dataset paths start from (default: the current folder)",
    )
    a.add_argument("--tables", type=Path, default=None, help="fact tables (default: <root>/data/tables)")
    a.set_defaults(func=command_analyze)


def command_analyze(args: argparse.Namespace) -> int:
    tables = args.tables or args.root / "data" / "tables"
    result = analyze(args.probe_run, args.bench_run, args.root, tables)
    out = args.probe_run / "probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_md(result), encoding="utf-8")
    print(summary_md(result))
    return 0
