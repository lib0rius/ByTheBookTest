"""The generator on the small test configuration (configs/smoke.toml)."""

import json
import os
from pathlib import Path

import pytest

from bythebooktest.config import GeneratorConfig
from bythebooktest.generate import generate, to_records, write_dataset
from bythebooktest.plan import LEVEL_DEPTHS, scaled_counts
from bythebooktest.tables import load_tables

ROOT = Path(__file__).resolve().parents[1]
WORKERS = os.cpu_count() or 1


@pytest.fixture(scope="module")
def dataset():
    config = GeneratorConfig.load(ROOT / "configs" / "smoke.toml")
    tables = load_tables(ROOT / "data" / "tables")
    kbs = generate(config, tables, log=lambda m: None, workers=WORKERS)
    kb_records, claim_records = to_records(kbs)
    return config, tables, kbs, kb_records, claim_records


def test_every_claim_matches_z3(dataset):
    *_, claims = dataset
    assert claims
    assert all(c.expected == c.z3 for c in claims)


def test_cell_sizes_and_claim_mix(dataset):
    config, _, _, _, claims = dataset
    for level, lvl in config.levels.items():
        expected = {(v.value, r): n for v, r, n in scaled_counts(lvl.claims_per_cell)}
        for group in lvl.groups:
            cell = [c for c in claims if c.level == level and c.group == group]
            assert len(cell) == lvl.claims_per_cell, (level, group)
            got: dict[tuple[str, str], int] = {}
            for c in cell:
                key = (c.expected, c.world_relation)
                got[key] = got.get(key, 0) + 1
            assert got == {k: v for k, v in expected.items() if v}, (level, group)


def test_variants_are_spread_within_cells(dataset):
    config, _, _, _, claims = dataset
    for level, lvl in config.levels.items():
        for group in lvl.groups:
            counts: dict[str, int] = {}
            for c in claims:
                if c.level == level and c.group == group:
                    counts[c.subtype] = counts.get(c.subtype, 0) + 1
            if group in ("modal", "direct_fact"):
                continue
            assert max(counts.values()) - min(counts.values()) <= 4, (level, group, counts)


def test_support_and_traps_exist(dataset):
    _, _, _, kbs, claims = dataset
    ids = {s.id for kb in kbs for s in kb.statements}
    for c in claims:
        assert set(c.support) <= ids and set(c.traps) <= ids
        if c.expected != "UNKNOWN":
            assert c.support


def test_direct_true_claims_are_written_in_kb(dataset):
    _, _, _, kbs, claims = dataset
    texts = {kb.kb_id: {s.text for s in kb.statements} for kb in kbs}
    for c in claims:
        if c.group == "direct_fact" and c.expected == "TRUE":
            assert c.text in texts[c.kb_id]
        if c.expected == "UNKNOWN":
            assert c.text not in texts[c.kb_id]


def test_fact_categories_are_balanced(dataset):
    # In every rulebook the non-neutral claims split about evenly among the five fact categories.
    _, _, _, kbs, claims = dataset
    for kb in kbs:
        counts: dict[str, int] = {}
        for c in claims:
            if c.kb_id == kb.kb_id and c.category is not None:
                counts[c.category] = counts.get(c.category, 0) + 1
        assert len(counts) == 5, (kb.kb_id, counts)
        assert max(counts.values()) - min(counts.values()) <= 4, (kb.kb_id, counts)


def test_depths_follow_levels(dataset):
    # Groups built on rule chains follow the depths of their level. direct_fact (depth 0) and modal
    # (depth 1) stay the same at every level, because every level contains the forms of the previous ones.
    _, _, _, _, claims = dataset
    for c in claims:
        if c.group == "direct_fact":
            assert c.depth == 0
        elif c.group == "modal":
            assert c.depth == 1
        else:
            lo, hi = LEVEL_DEPTHS[c.level]
            assert max(lo, 1) <= c.depth <= hi, c


def test_noise_ratio(dataset):
    config, _, _, kbs, _ = dataset
    for kb in kbs:
        ratio = config.levels[kb.level].noise_ratio
        noise = sum(1 for s in kb.statements if s.role == "noise")
        assert abs(noise / len(kb.statements) - ratio) < 0.05, kb.kb_id


def test_one_year_per_subject(dataset):
    _, _, _, kbs, _ = dataset
    for kb in kbs:
        years: dict[str, set[str]] = {}
        for s in kb.statements:
            if s.logic and s.logic.startswith("year:"):
                subject, year = s.logic.split("(", 1)[1].rstrip(")").rsplit(", ", 1)
                years.setdefault(subject, set()).add(year)
        assert all(len(v) == 1 for v in years.values()), (kb.kb_id, years)


def test_shared_chain_exits_use_each_predicate_once(dataset):
    # Two exits from the same chain with the same predicate ("No femoven equals nine" and "Every
    # buzastis equals nine") contradict each other as soon as an entity joins the chain.
    _, _, kbs, _, _ = dataset
    for g in kbs:
        kb = g.builder
        hub_of = {p.noun: h for h, hub in enumerate(kb.hubs) for p in hub}
        exits: dict[int, list[str]] = {}
        for s in kb.statements.values():
            words = s.text.rstrip(".").split(" ", 2)
            if len(words) < 3 or words[0] not in ("Every", "No") or words[1] not in hub_of:
                continue
            rest = words[2]
            if rest.startswith(("is a ", "is an ")) and rest.split()[-1] in hub_of:
                continue  # an inner step of the chain
            exits.setdefault(hub_of[words[1]], []).append(rest)
        for h, rests in exits.items():
            assert len(rests) == len(set(rests)), (kb.kb_id, h, rests)


def test_sliced_check_matches_full_check(dataset):
    # The sliced check (used on Level 3) must give the same answers as the check on the whole rulebook,
    # which is still feasible on the small test rulebooks.
    from bythebooktest.logic import evaluate, evaluate_sliced

    _, _, kbs, _, _ = dataset
    for g in kbs:
        kb = g.builder
        claims = [c.logic for c in kb.claims.values()]
        full = evaluate(kb.logic_statements(), claims, workers=WORKERS)
        sliced = evaluate_sliced(kb.logic_statements(), claims, workers=WORKERS)
        assert full.consistent and sliced.consistent, kb.kb_id
        assert full.verdicts == sliced.verdicts, kb.kb_id


def test_resume_from_checkpoints_gives_same_dataset(dataset, tmp_path):
    # Checked rulebooks are saved; started again, the generator reuses them and the result is the same.
    config, tables, kbs, _, _ = dataset
    write_dataset(tmp_path / "a", config, tables, kbs)
    ckpt = tmp_path / "checkpoints"
    first = generate(config, tables, log=lambda m: None, workers=WORKERS, checkpoints=ckpt)
    saved = sorted(ckpt.glob("*.pkl"))
    assert len(saved) == len(first)
    saved[-1].unlink()  # as if stopped during the last rulebook
    messages: list[str] = []
    again = generate(config, tables, log=messages.append, workers=WORKERS, checkpoints=ckpt)
    assert sum("resumed from" in m for m in messages) == len(first) - 1
    write_dataset(tmp_path / "b", config, tables, again)
    for name in ("kbs.jsonl", "claims.jsonl"):
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()


def test_sliced_check_survives_undecided_groups(dataset, monkeypatch):
    # If Z3 does not decide consistency for a group of claims (time or memory), those claims stay
    # undecided and the others hold; another check proves the consistency of the part.
    from bythebooktest import logic

    _, _, kbs, _, _ = dataset
    kb = kbs[-1].builder
    claims = [c.logic for c in kb.claims.values()]
    real = logic._component_task
    calls = {"n": 0}

    def flaky(stmts, cls, timeout_ms, memory_mb=0):
        calls["n"] += bool(cls)
        if cls and calls["n"] == 1:
            return "unknown", [None] * len(cls), "memout"
        return real(stmts, cls, timeout_ms, memory_mb)

    monkeypatch.setattr(logic, "_component_task", flaky)
    ev = logic.evaluate_sliced(kb.logic_statements(), claims, workers=1, chunk=2)
    assert ev.consistent and ev.undecided_reasons == ["memout"]
    undecided = sum(v is None for v in ev.verdicts)
    assert 0 < undecided <= 2


def test_z3_internal_error_on_a_claim_leaves_it_undecided(dataset, monkeypatch):
    # An internal Z3 error on one claim does not stop the check: that claim stays undecided, and the
    # others of the group are checked with a new solver.
    import z3

    from bythebooktest import logic

    _, _, kbs, _, _ = dataset
    kb = kbs[-1].builder
    claims = [c.logic for c in kb.claims.values()]
    stmts, cls = max(logic.components(kb.logic_statements(), claims), key=lambda p: len(p[1]))
    part = [kb.logic_statements()[i] for i in stmts]
    some = [claims[j] for j in cls[:3]]
    expected = logic._component_task(part, some, 30000)[1]
    real = logic._verdict
    calls = {"n": 0}

    def broken_once(solver, c):
        calls["n"] += 1
        if calls["n"] == 1:
            raise z3.Z3Exception(b"unreachable")
        return real(solver, c)

    monkeypatch.setattr(logic, "_verdict", broken_once)
    status, verdicts, _ = logic._component_task(part, some, 30000)
    assert status == "sat"
    assert verdicts[0] is None and verdicts[1:] == expected[1:]


def test_generation_is_deterministic(dataset, tmp_path):
    config, tables, kbs, _, claims = dataset
    write_dataset(tmp_path / "a", config, tables, kbs)
    again = generate(config, tables, log=lambda m: None, workers=WORKERS)
    write_dataset(tmp_path / "b", config, tables, again)
    for name in ("kbs.jsonl", "claims.jsonl"):
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["claims_total"] == len(claims)
