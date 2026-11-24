"""Dataset generator: plan, build, add noise, check with Z3, write.

For every level of the configuration, the generator plans the claims of each cell (level times
reasoning group), builds each rulebook with the statements those claims need, adds noise, and checks
every expected answer with Z3 against the whole rulebook. A claim whose answer Z3 does not confirm, or
that makes the rulebook contradict itself, is rebuilt and the rulebook checked again.

Usage::

    bythebooktest generate --config configs/level1.toml --out out/level1

Each rulebook is saved in ``<out>/checkpoints`` as soon as it is checked: an interrupted run resumes from
the first missing rulebook, with the same result as an uninterrupted one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import z3

from . import __version__
from .config import GeneratorConfig, LevelConfig
from .groups import Context
from .kb import KBBuilder, membership
from .logic import Evaluation, SlicedEvaluation, evaluate, evaluate_sliced, inconsistency_core, to_text
from .plan import ClaimSpec, cell_specs, split_among_kbs
from .records import ClaimRecord, KBRecord, StatementRecord
from .registry import GROUPS
from .status import STALE_SECONDS, StatusWriter
from .tables import Tables, load_tables
from .vocab import Vocabulary

MAX_CHECK_ROUNDS = 5
"""Check-and-rebuild rounds per rulebook before giving up."""

Log = Callable[[str], None]


@dataclass
class GeneratedKB:
    """A checked rulebook: its builder (statements and claims) and the order shown to the model."""

    builder: KBBuilder
    order: list[str]
    rebuilt: int
    """Claims rebuilt because Z3 did not confirm them."""


def _look_alike(name: str, rng: random.Random, vocab: Vocabulary) -> str | None:
    """Return a new name that differs by one vowel: ``"Kelvorax"`` -> ``"Kelvorix"``."""
    vowels = [i for i, ch in enumerate(name) if ch in "aeiou"]
    for _ in range(10):
        i = rng.choice(vowels)
        new = name[:i] + rng.choice([v for v in "aeiou" if v != name[i]]) + name[i + 1 :]
        if new.lower() not in vocab.used:
            vocab.used.add(new.lower())
            return new
    return None


def add_noise(
    kb: KBBuilder, ratio: float, rng: random.Random, vocab: Vocabulary, tables: Tables, style: str = "isolated"
) -> None:
    """Add statements that no claim needs, until they are ``ratio`` of the rulebook.

    ``isolated`` noise uses invented classes and names, unconnected to the rest. ``similar`` noise looks
    like the useful statements without changing any expected answer:

    - true predicates from the tables about invented entities ("Kelvorix was born in Australia.");
    - rules from invented classes to predicates of the tables ("Every zelkar was born in Japan.");
    - names that differ by one vowel from those of the claims, with the same classes
      ("Kelvorix is a trusap.");
    - subjects of the tables in classes that lead nowhere ("Napoleon is a zelkar.").
    """
    if ratio <= 0:
        return
    useful = len(kb.statements)
    target = round(ratio / (1 - ratio) * useful)
    # Noise is written in small groups of invented classes, renewed often: classes shared by all the
    # noise would connect unrelated parts of the rulebook and make the Z3 check needlessly heavy
    # (see logic.components).
    classes: list = []
    common: dict[str, Any] = dict(role="noise", world="neutral", category=None, owner=None)
    table_subjects = {r.subject for r in tables.facts}
    real_vps = [vp for r in tables.facts for vp in (r.true, *r.false)]
    from .kb import vp_pred
    from .logic import Atom, Not

    copies = []
    if style == "similar":
        for s in kb.statements.values():
            atom = s.logic.f if isinstance(s.logic, Not) else s.logic
            if (
                s.role == "proof"
                and isinstance(atom, Atom)
                and atom.pred.startswith("isa:")
                and len(atom.args) == 1
                and isinstance(atom.args[0], str)
                and " " not in atom.args[0]
                and atom.args[0] not in table_subjects
            ):
                copies.append((atom.args[0], atom.pred[4:], not isinstance(s.logic, Not)))

    subjects = sorted(table_subjects)
    added = 0
    while added < target:
        before = len(kb.statements)
        if added % 8 == 0:
            classes = [membership(vocab.noun()) for _ in range(3)]
            # Classes of a group may be linked ("Every bromupruk is a gumures"): a predicate taken
            # with one sign keeps that sign in the whole group, or "Every bromupruk is round" and
            # "No gumures is round" would contradict each other.
            polarity: dict[str, bool] = {}
        r = rng.random()
        if style != "similar" or r < 0.25:
            if rng.random() < 0.5:
                kb.fact(vocab.name(), rng.choice(classes), rng.random() < 0.8, **common)
            else:
                a, b = rng.sample(classes, 2)
                kb.rule(a, b, True, person=False, **common)
        elif r < 0.45:
            vp = rng.choice(real_vps)
            kb.fact(vocab.name(), vp_pred(vp.pos, vp.neg), rng.random() < 0.7, **common)
        elif r < 0.6:
            vp = rng.choice(real_vps)
            cls = rng.choice(classes)
            pred = vp_pred(vp.pos, vp.neg)
            positive = polarity.setdefault(pred.id, rng.random() < 0.7)
            kb.rule(cls, pred, positive, person=False, **common)
        elif r < 0.85 and copies:
            name, noun, positive = rng.choice(copies)
            other = _look_alike(name, rng, vocab)
            if other is not None:
                kb.fact(other, membership(noun), positive, **common)
        else:
            # A class that leads nowhere, new every time.
            kb.fact(rng.choice(subjects), membership(vocab.noun()), True, **common)
        added += len(kb.statements) - before  # a statement already present does not count


def build_kb(
    kb_id: str,
    level: int,
    specs: list[ClaimSpec],
    lvl: LevelConfig,
    seed: int,
    tables: Tables,
    vocab: Vocabulary,
    log: Log = print,
    workers: int = 1,
    progress: Callable[[str, int, int, int], None] | None = None,
) -> GeneratedKB:
    """Build one rulebook with its claims, add noise, and check it with Z3 until every answer holds.

    Raises:
        RuntimeError: If the expected answers still differ from Z3 after :data:`MAX_CHECK_ROUNDS`.
    """
    rng = random.Random(f"{seed}-{kb_id}")
    kb = KBBuilder(kb_id, level)
    ctx = Context(rng=rng, vocab=vocab, tables=tables, kb=kb)
    spec_of: dict[str, ClaimSpec] = {}

    specs = list(specs)
    rng.shuffle(specs)
    for spec in specs:
        claim = GROUPS[spec.group].build(spec, ctx)
        spec_of[claim.id] = spec

    add_noise(kb, lvl.noise_ratio, rng, vocab, tables, lvl.noise_style)

    rebuilt = 0
    for round_no in range(1, MAX_CHECK_ROUNDS + 1):
        claims = list(kb.claims.values())
        started = time.monotonic()
        on_progress = None
        if progress is not None:
            on_progress = lambda done, total, r=round_no: progress(kb_id, r, done, total)
        logic = kb.logic_statements()
        timeout_ms = lvl.z3_timeout_s * 1000
        ev: Evaluation | SlicedEvaluation
        if lvl.check == "sliced":
            ev = evaluate_sliced(
                logic,
                [c.logic for c in claims],
                timeout_ms=timeout_ms,
                workers=workers,
                progress=on_progress,
                chunk=lvl.chunk,
                memory_mb=lvl.z3_memory_mb,
            )
            bad_parts = ev.inconsistent
        else:
            ev = evaluate(
                logic, [c.logic for c in claims], timeout_ms=timeout_ms, workers=workers, progress=on_progress
            )
            bad_parts = [] if ev.consistent else [list(range(len(logic)))]
        if bad_parts:
            # The rulebook must never contradict itself: find the statements responsible and rebuild
            # the claims that brought them in.
            ids = kb.logic_statement_ids()
            core = []
            for part in bad_parts:
                core += [ids[part[i]] for i in inconsistency_core([logic[p] for p in part])]
            owners = sorted({o for sid in core for o in kb.owners(sid)})
            log(
                f"{kb_id}: the rulebook contradicts itself; statements: "
                + " | ".join(kb.statements[s].text for s in core)
            )
            noise = [s for s in core if kb.statements[s].role == "noise" and not kb.owners(s)]
            for sid in noise:
                kb.remove_noise(sid)
            if not owners and not noise:
                raise RuntimeError(f"{kb_id}: contradiction with no claim to rebuild")
            for cid in owners:
                spec = spec_of.pop(cid)
                kb.remove_claim(cid)
                new = GROUPS[spec.group].build(spec, ctx)
                spec_of[new.id] = spec
                rebuilt += 1
            continue
        bad = []
        for claim, verdict in zip(claims, ev.verdicts):
            claim.z3 = verdict
            if verdict != claim.expected:
                bad.append(claim)
        undecided = sum(1 for c in bad if c.z3 is None)
        why = ""
        reasons = getattr(ev, "undecided_reasons", [])
        if reasons:
            counts = Counter(r.splitlines()[0][:80] for r in reasons)
            why = "; consistency undecided in " + ", ".join(f"{n} parts ({r})" for r, n in counts.items())
        if isinstance(ev, SlicedEvaluation) and ev.retried:
            why += f"; {ev.retried} undecided claims checked again with more memory, {ev.recovered} decided"
        log(
            f"{kb_id}: check {round_no}: {len(claims)} claims, {len(kb.statements)} statements, "
            f"{len(bad)} to rebuild, of which {undecided} undecided by Z3 "
            f"({time.monotonic() - started:.1f} s){why}"
        )
        if not bad:
            break
        for claim in bad:
            spec = spec_of.pop(claim.id)
            kb.remove_claim(claim.id)
            new = GROUPS[spec.group].build(spec, ctx)
            spec_of[new.id] = spec
            rebuilt += 1
    else:
        raise RuntimeError(f"{kb_id}: expected answers still differ from Z3 after {MAX_CHECK_ROUNDS} checks")

    order = list(kb.statements)
    rng.shuffle(order)
    return GeneratedKB(builder=kb, order=order, rebuilt=rebuilt)


def planned_claims(config: GeneratorConfig) -> int:
    """Number of claims the configuration asks for."""
    return sum(lv.claims_per_cell * len(lv.groups) for lv in config.levels.values())


def _code_sha256() -> str:
    """Fingerprint of the package source: a code change can change the rulebooks produced."""
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _checkpoint_key(config: GeneratorConfig, tables: Tables, kb_id: str) -> dict:
    """What must match for a saved rulebook to be reused: code, configuration and tables."""
    return {
        "generator_version": __version__,
        "code_sha256": _code_sha256(),
        "config": config.raw,
        "tables_sha256": tables.sha256,
        "kb_id": kb_id,
    }


def _load_checkpoint(path: Path, key: dict) -> tuple[GeneratedKB, Vocabulary] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        saved = pickle.load(f)
    if saved["key"] != key:
        raise RuntimeError(
            f"{path}: rulebook saved with different code, configuration or tables; "
            "move or delete it to generate it again"
        )
    return saved["kb"], saved["vocab"]


def _save_checkpoint(path: Path, key: dict, kb: GeneratedKB, vocab: Vocabulary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"key": key, "kb": kb, "vocab": vocab}, f)
    tmp.replace(path)


def generate(
    config: GeneratorConfig,
    tables: Tables,
    log: Log = print,
    workers: int = 1,
    status: StatusWriter | None = None,
    checkpoints: Path | None = None,
) -> list[GeneratedKB]:
    """Generate all the rulebooks of the configuration.

    Args:
        config: The generator configuration.
        tables: The curated fact tables.
        log: Where progress messages go.
        workers: Processes for the Z3 checks.
        status: Progress reporting, if any.
        checkpoints: Folder where each rulebook is saved, with the vocabulary state, as soon as it is
            checked. On a new start, saved rulebooks are reused and work resumes from the first
            missing one, with the same result as an uninterrupted run.
    """
    vocab = Vocabulary(random.Random(f"{config.seed}-vocab"), tables.reserved_words())
    result: list[GeneratedKB] = []
    done_before = 0
    started = time.monotonic()

    # A further check round (after rebuilding claims) checks all the claims of the rulebook again: it
    # is added to the plan, so progress never goes back and the estimate includes the extra work.
    rounds: dict = {"current": None, "extra": 0}

    def progress(kb_id: str, round_no: int, done: int, total: int) -> None:
        if status is None:
            return
        if rounds["current"] != (kb_id, round_no):
            rounds["current"] = (kb_id, round_no)
            if round_no > 1:
                rounds["extra"] += total
                status.plan_total += total
        plan_done = done_before + rounds["extra"] + done
        rate = (plan_done - status.plan_offset) / max(1e-6, time.monotonic() - started)
        step = f"{kb_id}, round {round_no}"
        if round_no > 1:
            step += " (check again after rebuilding claims)"
        status.update(
            phase="checking every claim with Z3",
            step=step,
            step_done=done,
            step_total=total,
            plan_done=plan_done,
            last={"rulebooks done": len(result), "claims per second": f"{rate:.1f}", "processes": workers},
        )

    for level in sorted(config.levels):
        lvl = config.levels[level]
        rng = random.Random(f"{config.seed}-L{level}")
        per_kb: list[list[ClaimSpec]] = [[] for _ in range(lvl.kbs)]
        for group in lvl.groups:
            depths = GROUPS[group].depths(level)
            variants = GROUPS[group].variants(level)
            specs = cell_specs(group, lvl.claims_per_cell, depths, rng, variants)
            for i, bucket in enumerate(split_among_kbs(specs, lvl.kbs, rng)):
                per_kb[i].extend(bucket)
        for i, specs in enumerate(per_kb):
            kb_id = f"L{level}-KB{i + 1}"
            path = None if checkpoints is None else checkpoints / f"{kb_id}.pkl"
            key = _checkpoint_key(config, tables, kb_id)
            saved = None if path is None else _load_checkpoint(path, key)
            if saved is not None:
                g, vocab = saved
                if status is not None:
                    status.plan_offset += len(specs)
                log(f"{kb_id}: resumed from {path} ({len(g.order)} statements, {len(g.builder.claims)} claims)")
            else:
                if status is not None:
                    status.update(
                        phase="building rulebooks and claims",
                        step=kb_id,
                        step_done=0,
                        step_total=len(specs),
                        plan_done=done_before + rounds["extra"],
                    )
                g = build_kb(kb_id, level, specs, lvl, config.seed, tables, vocab, log, workers, progress)
                if path is not None:
                    _save_checkpoint(path, key, g, vocab)
                    log(f"{kb_id}: saved in {path}")
            result.append(g)
            done_before += len(specs)
    return result


def to_records(kbs: list[GeneratedKB]) -> tuple[list[KBRecord], list[ClaimRecord]]:
    """Convert generated rulebooks to the records written in the dataset files."""
    kb_records: list[KBRecord] = []
    claim_records: list[ClaimRecord] = []
    for g in kbs:
        kb = g.builder
        statements = [
            StatementRecord.model_validate(
                {
                    "id": s.id,
                    "text": s.text,
                    "form": s.form,
                    "logic": to_text(s.logic) if s.logic is not None else None,
                    "role": s.role,
                    "world": s.world,
                    "category": s.category,
                }
            )
            for s in (kb.statements[sid] for sid in g.order)
        ]
        kb_records.append(KBRecord(kb_id=kb.kb_id, level=kb.level, statements=statements))
        for c in kb.claims.values():
            if c.z3 is None:
                raise RuntimeError(f"{c.id}: claim not checked by Z3")
            claim_records.append(
                ClaimRecord.model_validate(
                    {
                        "claim_id": c.id,
                        "kb_id": kb.kb_id,
                        "level": kb.level,
                        "group": c.group,
                        "subtype": c.subtype,
                        "text": c.text,
                        "logic": to_text(c.logic),
                        "expected": c.expected.value,
                        "z3": c.z3.value,
                        "world_relation": c.world_relation,
                        "category": c.category,
                        "depth": c.depth,
                        "reasoning": c.reasoning,
                        "support": c.support,
                        "traps": c.traps,
                    }
                )
            )
    return kb_records, claim_records


def summary(claims: list[ClaimRecord]) -> dict:
    """Claim counts by level, group, expected answer and world relation."""
    counts: dict[str, int] = {}
    for c in claims:
        key = f"L{c.level}/{c.group}/{c.expected}/{c.world_relation}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_dataset(out: Path, config: GeneratorConfig, tables: Tables, kbs: list[GeneratedKB]) -> dict:
    """Write ``kbs.jsonl``, ``claims.jsonl`` and ``manifest.json`` in ``out``; return the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    kb_records, claim_records = to_records(kbs)
    with (out / "kbs.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in kb_records:
            f.write(r.model_dump_json() + "\n")
    with (out / "claims.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for c in claim_records:
            f.write(c.model_dump_json() + "\n")
    manifest = {
        "generator_version": __version__,
        "z3_version": z3.get_version_string(),
        "seed": config.seed,
        "config": config.raw,
        "tables_sha256": tables.sha256,
        "kbs": {
            g.builder.kb_id: {"statements": len(g.order), "claims": len(g.builder.claims), "rebuilt": g.rebuilt}
            for g in kbs
        },
        "claims_total": len(claim_records),
        "counts": summary(claim_records),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def file_logger(path: Path) -> Log:
    """Return a logger that prints timestamped lines and appends them to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    return log


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``generate`` command."""
    p = sub.add_parser(
        "generate", help="generate a dataset", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True, help="generator file, for example configs/level1.toml")
    p.add_argument("--out", type=Path, required=True, help="output folder")
    p.add_argument("--tables", type=Path, default=Path("data/tables"), help="fact tables (default: %(default)s)")
    p.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processes for the Z3 checks (default: all CPU threads)",
    )
    p.set_defaults(func=command)


def command(args: argparse.Namespace) -> int:
    log = file_logger(args.out / "generate.log")
    config = GeneratorConfig.load(args.config)
    tables = load_tables(args.tables)
    # In the sliced check a group of claims can take as long as the Z3 time limit: until then, no
    # update is normal, not a stall.
    stale = max(STALE_SECONDS, max(lv.z3_timeout_s for lv in config.levels.values()))
    status = StatusWriter(args.out / "status.json", planned_claims(config), "claim checks", stale_after_s=stale)
    started = time.monotonic()
    log(f"Start: {args.config}, {planned_claims(config)} claims, {args.workers} processes")
    kbs = generate(config, tables, log=log, workers=args.workers, status=status, checkpoints=args.out / "checkpoints")
    manifest = write_dataset(args.out, config, tables, kbs)
    status.update(
        phase="finished",
        step="writing the dataset",
        step_done=1,
        step_total=1,
        plan_done=status.plan_total,
        finished=True,
    )
    log(
        f"Done: {manifest['claims_total']} claims in {len(kbs)} rulebooks, "
        f"{time.monotonic() - started:.1f} s. Output in {args.out}"
    )
    for kb_id, info in manifest["kbs"].items():
        if info["rebuilt"]:
            log(f"  {kb_id}: {info['rebuilt']} claims rebuilt after the Z3 check")
    return 0
