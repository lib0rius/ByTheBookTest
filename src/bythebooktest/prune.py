"""Pruning the noise of a dataset already generated.

For every rulebook saved in the generator checkpoints, noise statements that no claim needs are removed
at random, with a fixed seed, until noise is the requested share of all statements. Claims, useful
statements, texts and statement order stay identical. Z3 then checks the rulebook again: it must stay
consistent and every claim must keep its expected answer, or pruning stops with an error.

This is how the released Level 2 and Level 3 datasets were made shorter without changing any claim.

Usage::

    bythebooktest prune --source out/level3 --noise-ratio 0.2 --out out/level3-noise20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import time
from collections.abc import Callable
from pathlib import Path

from .config import GeneratorConfig, LevelConfig
from .generate import GeneratedKB, _code_sha256, file_logger, write_dataset
from .logic import Evaluation, SlicedEvaluation, evaluate, evaluate_sliced
from .status import STALE_SECONDS, StatusWriter
from .tables import load_tables


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source(source: Path) -> tuple[dict, list[dict], dict[str, set[str]]]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    kbs = [json.loads(line) for line in (source / "kbs.jsonl").open(encoding="utf-8")]
    claims: dict[str, set[str]] = {}
    for line in (source / "claims.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        claims.setdefault(c["kb_id"], set()).add(c["claim_id"])
    return manifest, kbs, claims


def prune_kb(g: GeneratedKB, ratio: float, rng: random.Random) -> dict:
    """Remove free noise (statements no claim uses) from ``g`` in place; return the counts.

    Raises:
        RuntimeError: If reaching ``ratio`` would require removing noise that some claim uses.
    """
    kb = g.builder
    noise = [sid for sid in g.order if kb.statements[sid].role == "noise"]
    free = [sid for sid in noise if not kb.owners(sid)]
    useful = len(g.order) - len(noise)
    target = round(ratio / (1 - ratio) * useful)
    n_remove = max(0, len(noise) - target)
    if n_remove > len(free):
        raise RuntimeError(
            f"{kb.kb_id}: {n_remove} noise statements should be removed, but only {len(free)} are not used by any claim"
        )
    for sid in rng.sample(free, n_remove):
        kb.remove_noise(sid)
    g.order = [sid for sid in g.order if sid in kb.statements]
    return {
        "useful": useful,
        "noise_before": len(noise),
        "noise_after": len(noise) - n_remove,
        "removed": n_remove,
        "statements": len(g.order),
    }


def check_kb(
    g: GeneratedKB, lvl: LevelConfig, workers: int, progress: Callable[[int, int], None] | None = None
) -> None:
    """Check the pruned rulebook with Z3: consistent, and every claim with its expected answer.

    Raises:
        RuntimeError: If the rulebook contradicts itself or any claim changed its answer.
    """
    kb = g.builder
    claims = list(kb.claims.values())
    logic = kb.logic_statements()
    timeout_ms = lvl.z3_timeout_s * 1000
    ev: Evaluation | SlicedEvaluation
    if lvl.check == "sliced":
        ev = evaluate_sliced(
            logic,
            [c.logic for c in claims],
            timeout_ms=timeout_ms,
            workers=workers,
            progress=progress,
            chunk=lvl.chunk,
            memory_mb=lvl.z3_memory_mb,
        )
        inconsistent = bool(ev.inconsistent)
    else:
        ev = evaluate(logic, [c.logic for c in claims], timeout_ms=timeout_ms, workers=workers, progress=progress)
        inconsistent = not ev.consistent
    if inconsistent:
        raise RuntimeError(f"{kb.kb_id}: the pruned rulebook contradicts itself")
    changed = [
        (c.id, c.expected.value, None if v is None else v.value) for c, v in zip(claims, ev.verdicts) if v != c.expected
    ]
    if changed:
        raise RuntimeError(
            f"{kb.kb_id}: {len(changed)} claims changed answer after pruning (claim, expected, Z3): {changed[:20]}"
        )
    for c, v in zip(claims, ev.verdicts):
        c.z3 = v


def _load_saved(
    source: Path, src_kbs: list[dict], src_claims: dict[str, set[str]], raw: dict, tables_sha256: str
) -> tuple[list[GeneratedKB], list[str]]:
    """Load the checkpoints of the source dataset and check they match its files exactly."""
    saved: list[GeneratedKB] = []
    code_changed = []
    for rec in src_kbs:
        path = source / "checkpoints" / f"{rec['kb_id']}.pkl"
        with path.open("rb") as f:
            data = pickle.load(f)
        if data["key"]["config"] != raw or data["key"]["tables_sha256"] != tables_sha256:
            raise RuntimeError(f"{path}: configuration or tables differ from the source dataset")
        if data["key"]["code_sha256"] != _code_sha256():
            code_changed.append(rec["kb_id"])
        g = data["kb"]
        if g.order != [s["id"] for s in rec["statements"]]:
            raise RuntimeError(f"{path}: statements differ from kbs.jsonl")
        if set(g.builder.claims) != src_claims[rec["kb_id"]]:
            raise RuntimeError(f"{path}: claims differ from claims.jsonl")
        saved.append(g)
    return saved, code_changed


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``prune`` command."""
    p = sub.add_parser(
        "prune",
        help="remove part of the noise of a generated dataset",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, required=True, help="generated dataset, with its checkpoints/")
    p.add_argument("--noise-ratio", type=float, required=True, help="requested share of noise, for example 0.2")
    p.add_argument("--out", type=Path, required=True, help="output folder, different from --source")
    p.add_argument("--tables", type=Path, default=Path("data/tables"), help="fact tables (default: %(default)s)")
    p.add_argument("--status", type=Path, default=None, help="status file (default: <out>/status.json)")
    p.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processes for the Z3 checks (default: all CPU threads)",
    )
    p.set_defaults(func=command)


def command(args: argparse.Namespace) -> int:
    if not 0 <= args.noise_ratio < 1:
        raise SystemExit("--noise-ratio must be at least 0 and less than 1")
    if args.out.resolve() == args.source.resolve():
        raise SystemExit("--out must differ from --source: the source dataset stays as it is")
    log = file_logger(args.out / "prune.log")

    manifest, src_kbs, src_claims = _load_source(args.source)
    raw = manifest["config"]
    config = GeneratorConfig.from_dict(raw)
    tables = load_tables(args.tables)
    if tables.sha256 != manifest["tables_sha256"]:
        raise RuntimeError("the fact tables changed since the source dataset was generated")

    saved, code_changed = _load_saved(args.source, src_kbs, src_claims, raw, manifest["tables_sha256"])
    if code_changed:
        log(
            f"Note: the package code changed after {code_changed} were saved; the saved rulebooks are "
            "still those of the source dataset (checked above)"
        )

    plan_total = sum(len(g.builder.claims) for g in saved)
    stale = max(STALE_SECONDS, max(lv.z3_timeout_s for lv in config.levels.values()))
    status = StatusWriter(args.status or args.out / "status.json", plan_total, "claim checks", stale_after_s=stale)
    started = time.monotonic()
    log(
        f"Start: {args.source} -> {args.out}, noise {args.noise_ratio:.0%}, {len(saved)} rulebooks, "
        f"{plan_total} claims, {args.workers} processes"
    )

    counts: dict[str, dict] = {}
    done_before = 0
    for g in saved:
        kb_id = g.builder.kb_id
        lvl = config.levels[g.builder.level]
        rng = random.Random(f"{config.seed}-{kb_id}-prune-{args.noise_ratio}")
        counts[kb_id] = prune_kb(g, args.noise_ratio, rng)
        c = counts[kb_id]
        log(
            f"{kb_id}: useful {c['useful']}, noise {c['noise_before']} -> {c['noise_after']} "
            f"({c['removed']} removed), statements {c['statements']}"
        )
        total = len(g.builder.claims)
        t0 = time.monotonic()

        def progress(done: int, tot: int, kb_id: str = kb_id, base: int = done_before) -> None:
            rate = (base + done) / max(1e-6, time.monotonic() - started)
            status.update(
                phase="checking again with Z3 after pruning",
                step=kb_id,
                step_done=done,
                step_total=tot,
                plan_done=base + done,
                last={"rulebooks done": len(counts) - 1, "claims per second": f"{rate:.2f}", "processes": args.workers},
            )

        progress(0, total)
        check_kb(g, lvl, args.workers, progress)
        done_before += total
        progress(total, total)
        log(f"{kb_id}: check passed, {total} claims with their expected answer ({time.monotonic() - t0:.1f} s)")

    out_manifest = write_dataset(args.out, config, tables, saved)
    out_manifest["prune"] = {
        "source": str(args.source),
        "source_manifest_sha256": _sha256(args.source / "manifest.json"),
        "source_kbs_sha256": _sha256(args.source / "kbs.jsonl"),
        "noise_ratio": args.noise_ratio,
        "rng": "random.Random(f'{seed}-{kb_id}-prune-{noise_ratio}')",
        "prune_code_sha256": _code_sha256(),
        "kbs": counts,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    status.update(
        phase="finished", step="writing the dataset", step_done=1, step_total=1, plan_done=plan_total, finished=True
    )
    log(
        f"Done: {out_manifest['claims_total']} claims in {len(saved)} rulebooks, "
        f"{time.monotonic() - started:.1f} s. Output in {args.out}"
    )
    return 0
