"""The test program against a fake llama-server (no model): prompts, answers, resuming, timeouts, server
crashes, measures, progress and the knowledge probe."""

import json
import math
import os
import socket
import stat
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from bythebooktest import probe, prompt
from bythebooktest.answer import parse_exact, parse_lenient
from bythebooktest.config import GeneratorConfig, ModelSpec, RunConfig, ServerOptions
from bythebooktest.generate import generate, write_dataset
from bythebooktest.metrics import compute, measures, same_question, wilson
from bythebooktest.records import ClaimRecord, KBRecord, StatementRecord
from bythebooktest.run import Runner
from bythebooktest.server import ServerError
from bythebooktest.status import render
from bythebooktest.tables import load_tables

ROOT = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).with_name("fake_llama_server.py")


# --- reading the answer -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content, exact, lenient",
    [
        ('{"answer": "TRUE"}', "TRUE", "TRUE"),
        ('  {"answer": "UNKNOWN"}\n', "UNKNOWN", "UNKNOWN"),
        ('<think>maybe FALSE</think>{"answer": "TRUE"}', "TRUE", "TRUE"),
        ('```json\n{"answer": "FALSE"}\n```', None, "FALSE"),
        ("The answer is TRUE.", None, "TRUE"),
        ("TRUE or FALSE?", None, None),
        ('{"answer": "true"}', None, None),
        ('{"answer": "TRUE", "why": "x"}', None, "TRUE"),
        ("", None, None),
    ],
)
def test_answer_parsing(content, exact, lenient):
    assert parse_exact(content) == exact
    assert parse_lenient(content) == lenient


# --- prompts --------------------------------------------------------------------------------------------


def test_prompt_puts_kb_before_claim():
    kb = KBRecord(
        kb_id="K",
        level=1,
        statements=[
            StatementRecord(
                id="s1",
                text="Every tarv is a pelm.",
                form="f",
                logic=None,
                role="proof",
                world="neutral",
                category=None,
            ),
            StatementRecord(
                id="s2", text="Kiro is a tarv.", form="f", logic=None, role="proof", world="neutral", category=None
            ),
        ],
    )
    msgs = prompt.messages(kb, "Kiro is a pelm.")
    assert msgs[0]["role"] == "system" and "10. Years are numbers." in msgs[0]["content"]
    assert msgs[1]["content"] == ("KNOWLEDGE BASE\nEvery tarv is a pelm.\nKiro is a tarv.\n\nCLAIM\nKiro is a pelm.")
    assert msgs[0]["content"].endswith('{"answer": "TRUE"}, {"answer": "FALSE"} or {"answer": "UNKNOWN"}.')


def test_systemone_request_uses_the_same_rules():
    kb = KBRecord(
        kb_id="K",
        level=1,
        statements=[
            StatementRecord(
                id="s1", text="Kiro is a tarv.", form="f", logic=None, role="proof", world="neutral", category=None
            )
        ],
    )
    req = prompt.systemone_request(kb, "Kiro is a pelm.")
    assert req["state"] == "KNOWLEDGE BASE\nKiro is a tarv."
    q = req["questions"]["claim"]
    assert q["type"] == "choice" and list(q["criteria"]) == ["TRUE", "FALSE", "UNKNOWN"]
    assert "10. Years are numbers." in q["instructions"] and "JSON" not in q["instructions"]
    assert q["instructions"].endswith("CLAIM\nKiro is a pelm.")


def test_request_hash_covers_systemone_options(monkeypatch):
    # The request fingerprint changes with the text of the options; the chat prompt fingerprint does not.
    req, pr = prompt.request_sha256(), prompt.prompt_sha256()
    monkeypatch.setitem(prompt.SYSTEMONE_CRITERIA, "UNKNOWN", "UNKNOWN: something else.")
    assert prompt.request_sha256() != req and prompt.prompt_sha256() == pr


# --- intervals ------------------------------------------------------------------------------------------


def test_wilson_interval():
    lo, hi = wilson(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3


# --- configuration ------------------------------------------------------------------------------------------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(
        'run_id = "x"\nout = "o"\ndatasets = ["ds"]\n' + text + '[[models]]\nname = "M-2B"\npath = "m.gguf"\n'
    )
    return path


def test_fraction_and_claims_per_kb_are_exclusive(tmp_path):
    with pytest.raises(ValidationError, match="not both"):
        RunConfig.load(_write(tmp_path, "claims_per_kb = 2\n"), fraction=0.1)


def test_world_task_needs_decision_models(tmp_path):
    with pytest.raises(ValidationError, match="decision models"):
        RunConfig.load(_write(tmp_path, 'task = "world"\n'))


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="claim_timeout"):
        RunConfig.load(_write(tmp_path, "claim_timeout = 10\n"))


def test_overrides_and_model_filter(tmp_path):
    cfg = RunConfig.load(_write(tmp_path, ""), models=["2B"], parallel=3, run_id="y", levels=None)
    assert cfg.run_id == "y" and cfg.server.parallel == 3 and cfg.levels is None
    with pytest.raises(SystemExit, match="no model"):
        RunConfig.load(_write(tmp_path, ""), models=["9B"])


def test_generator_config_checks_groups(tmp_path):
    path = tmp_path / "gen.toml"
    path.write_text(
        'seed = 1\n[level.1]\nkbs = 1\nclaims_per_cell = 4\nnoise_ratio = 0.5\ngroups = ["direct_fact", "telepathy"]\n'
    )
    with pytest.raises(ValidationError, match="telepathy"):
        GeneratorConfig.load(path)


# --- full runs with the fake server -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    config = GeneratorConfig.load(ROOT / "configs" / "smoke.toml")
    tables = load_tables(ROOT / "data" / "tables")
    kbs = generate(config, tables, log=lambda m: None, workers=os.cpu_count() or 1)
    write_dataset(root / "ds", config, tables, kbs)
    return root


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _config(tmp_path: Path, **kw) -> RunConfig:
    launcher = tmp_path / "fake-llama-server"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    models = []
    for name in ("Fake-A", "Fake-B"):
        path = tmp_path / f"{name}.gguf"
        path.write_bytes(b"x" * (1000 if name == "Fake-A" else 3000))
        models.append(ModelSpec(name=name, path=str(path)))
    base = dict(
        run_id="t",
        out=str(tmp_path / "runs"),
        datasets=["ds"],
        models=models,
        sampling={"temperature": 1.0, "top_p": 0.95},
        claims_per_kb=3,
        server=ServerOptions(binary=str(launcher), port=_free_port()),
    )
    base.update(kw)
    return RunConfig(**base)


def _records(run_dir: Path, model: str) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "responses" / f"{model}.jsonl").open()]


def test_full_run_records_every_claim(dataset_root, tmp_path):
    cfg = _config(tmp_path)
    runner = Runner(cfg, dataset_root, log=lambda m: None)
    assert runner.run() == 0
    run_dir = Path(cfg.out) / "t"
    n = len(runner.tasks)
    for model in ("Fake-A", "Fake-B"):
        recs = _records(run_dir, model)
        assert len(recs) == n and len({r["claim_id"] for r in recs}) == n
        assert all(r["status"] == "ok" and r["answer_exact"] in ("TRUE", "FALSE", "UNKNOWN") for r in recs)
        assert all(r["reasoning"] and r["reasoning_tokens"] == 4 and not r["budget_hit"] for r in recs)
    saved = json.loads((run_dir / "run.json").read_text())
    assert set(saved["models"]) == {"Fake-A", "Fake-B"}
    assert len(saved["models"]["Fake-A"]["sha256"]) == 64
    assert "out" not in saved["key"]["config"] and len(saved["key"]["request_sha256"]) == 64
    state = json.loads((run_dir / "status.json").read_text())
    assert state["finished"] and state["plan_done"] == state["plan_total"] == 2 * n
    metrics = compute(run_dir, dataset_root)
    assert metrics["models"]["Fake-A"]["all"]["n"] == n
    assert metrics["models"]["Fake-A"]["all"]["invalid"]["k"] == 0


def test_payload_follows_config(dataset_root, tmp_path):
    cfg = _config(tmp_path, seed=7)
    runner = Runner(cfg, dataset_root, log=lambda m: None)
    base = cfg.payload_base()
    assert base["seed"] == 7 and base["max_tokens"] == 8192 + 1024
    assert base["chat_template_kwargs"] == {"enable_thinking": True}
    assert base["temperature"] == 1.0 and runner.tasks


def test_resume_skips_done_claims(dataset_root, tmp_path):
    cfg = _config(tmp_path)
    Runner(cfg, dataset_root, log=lambda m: None).run()
    run_dir = Path(cfg.out) / "t"
    path = run_dir / "responses" / "Fake-B.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-2]) + lines[-2][:10])  # last answer lost, the one before cut
    messages: list[str] = []
    Runner(cfg, dataset_root, log=messages.append).run()
    assert any("Fake-A: already complete" in m for m in messages)
    ids = [json.loads(line)["claim_id"] for line in path.read_text().splitlines()]
    assert sorted(set(ids)) == sorted(ids) and len(ids) == len(lines)


def test_changed_config_is_refused(dataset_root, tmp_path):
    cfg = _config(tmp_path)
    Runner(cfg, dataset_root, log=lambda m: None).run()
    cfg.seed = 43
    with pytest.raises(SystemExit, match="run_id"):
        Runner(cfg, dataset_root, log=lambda m: None).run()


def test_timeout_marks_claim_and_run_continues(dataset_root, tmp_path, monkeypatch):
    cfg = _config(tmp_path, claim_timeout_s=2)
    cfg.models = cfg.models[:1]
    runner = Runner(cfg, dataset_root, log=lambda m: None)
    target = runner.tasks[1].claim.text
    monkeypatch.setenv("FAKE_SLEEP_ON", target)
    runner.run()
    recs = _records(Path(cfg.out) / "t", "Fake-A")
    by_text = {t.claim.claim_id: t.claim.text for t in runner.tasks}
    assert [r["status"] for r in recs if by_text[r["claim_id"]] == target] == ["timeout"]
    assert sum(r["status"] == "ok" for r in recs) == len(recs) - 1


def test_retry_timeouts_redoes_only_timed_out_claims(dataset_root, tmp_path, monkeypatch):
    cfg = _config(tmp_path, claim_timeout_s=2)
    cfg.models = cfg.models[:1]
    runner = Runner(cfg, dataset_root, log=lambda m: None)
    target = runner.tasks[1].claim
    monkeypatch.setenv("FAKE_SLEEP_ON", target.text)
    runner.run()
    monkeypatch.delenv("FAKE_SLEEP_ON")
    run_dir = Path(cfg.out) / "t"
    n_before = len(_records(run_dir, "Fake-A"))
    # Without retry_timeouts the run is complete: nothing to do.
    messages: list[str] = []
    Runner(cfg, dataset_root, log=messages.append).run()
    assert any("already complete" in m for m in messages)
    # With retry_timeouts and a different number of slots, only the timed-out claim is asked again.
    cfg.server.parallel = 1
    Runner(cfg, dataset_root, log=lambda m: None, retry_timeouts=True).run()
    recs = _records(run_dir, "Fake-A")
    assert len(recs) == n_before + 1
    assert recs[-1]["claim_id"] == target.claim_id and recs[-1]["status"] == "ok"
    assert recs[-1]["retry_of"] == "timeout" and recs[-1]["parallel"] == 1
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["retries"][0]["parallel"] == 1
    metrics = compute(run_dir, dataset_root)
    assert metrics["models"]["Fake-A"]["all"]["not_completed"]["k"] == 0  # the new answer counts
    # Any other change (not the number of slots) is still refused when retrying.
    cfg.seed = 43
    with pytest.raises(SystemExit, match="parallel"):
        Runner(cfg, dataset_root, log=lambda m: None, retry_timeouts=True).run()


def test_engine_crash_is_retried(dataset_root, tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.models = cfg.models[:1]
    flag = tmp_path / "crash"
    flag.write_text("1")
    monkeypatch.setenv("FAKE_CRASH_FILE", str(flag))
    messages: list[str] = []
    runner = Runner(cfg, dataset_root, log=messages.append)
    runner.run()
    recs = _records(Path(cfg.out) / "t", "Fake-A")
    assert all(r["status"] == "ok" for r in recs) and len(recs) == len(runner.tasks)
    assert recs[0]["attempt"] == 2 and any("server error" in m for m in messages)


def test_prose_answers_are_invalid_but_lenient(dataset_root, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_REPLY", "prose")
    cfg = _config(tmp_path)
    cfg.models = cfg.models[:1]
    Runner(cfg, dataset_root, log=lambda m: None).run()
    recs = _records(Path(cfg.out) / "t", "Fake-A")
    assert all(r["answer_exact"] is None and r["answer_lenient"] for r in recs)
    m = compute(Path(cfg.out) / "t", dataset_root)["models"]["Fake-A"]
    # The main (lenient) reading accepts them; the exact reading counts them as invalid.
    assert m["all_exact"]["invalid"]["k"] == m["all_exact"]["n"] and m["all_exact"]["accuracy"]["k"] == 0
    assert m["all"]["invalid"]["k"] == 0


def test_parallel_slots(dataset_root, tmp_path):
    cfg = _config(tmp_path)
    cfg.server.parallel = 3
    runner = Runner(cfg, dataset_root, log=lambda m: None)
    runner.run()
    recs = _records(Path(cfg.out) / "t", "Fake-B")
    assert sorted(r["claim_id"] for r in recs) == sorted(t.claim.claim_id for t in runner.tasks)


def test_no_gpu_refuses_to_start(dataset_root, tmp_path, monkeypatch):
    # Without a visible GPU llama-server would silently fall back to the CPU: the run does not start.
    monkeypatch.setenv("FAKE_NO_GPU", "1")
    cfg = _config(tmp_path)
    with pytest.raises(ServerError, match="sees no GPU"):
        Runner(cfg, dataset_root, log=lambda m: None).run()


def test_devices_are_recorded(dataset_root, tmp_path):
    cfg = _config(tmp_path)
    cfg.models = cfg.models[:1]
    Runner(cfg, dataset_root, log=lambda m: None).run()
    saved = json.loads((Path(cfg.out) / "t" / "run.json").read_text())
    assert saved["models"]["Fake-A"]["devices"] == ["FakeGPU0: fake"]


def test_stop_signal_ends_quickly_and_resumes(dataset_root, tmp_path, monkeypatch):
    # A stop during a slow answer stops the server at once, writes no interrupted answer, and the
    # resumed run completes the rest.
    cfg = _config(tmp_path, claim_timeout_s=60)
    cfg.models = cfg.models[:1]
    runner = Runner(cfg, dataset_root, log=lambda m: None)
    monkeypatch.setenv("FAKE_SLEEP_ON", runner.tasks[2].claim.text)
    result: dict = {}
    worker = threading.Thread(target=lambda: result.setdefault("code", runner.run()))
    worker.start()
    path = Path(cfg.out) / "t" / "responses" / "Fake-A.jsonl"
    deadline = time.time() + 30
    while time.time() < deadline and not (path.exists() and len(path.read_text().splitlines()) >= 2):
        time.sleep(0.1)
    time.sleep(0.5)  # the third, slow request is in progress
    t0 = time.time()
    with pytest.raises(SystemExit):
        runner.handle_stop(15, None)
    worker.join(20)
    assert not worker.is_alive() and time.time() - t0 < 15 and result["code"] == 143
    recs = _records(Path(cfg.out) / "t", "Fake-A")
    assert len(recs) == 2 and all(r["status"] == "ok" for r in recs)
    monkeypatch.delenv("FAKE_SLEEP_ON")
    Runner(cfg, dataset_root, log=lambda m: None).run()
    assert len(_records(Path(cfg.out) / "t", "Fake-A")) == len(runner.tasks)


def test_systemone_model_answers_with_an_option(dataset_root, tmp_path):
    # A decision model gets rulebook and claim on /v1/systemone and picks TRUE, FALSE or UNKNOWN; the
    # choice, the option probabilities and the confidence are recorded.
    cfg = _config(tmp_path)
    cfg.models = [ModelSpec(name="D1", path=cfg.models[0].path, kind="systemone")]
    cfg.server.reasoning = False
    runner = Runner(cfg, dataset_root, log=lambda m: None)
    assert runner.run() == 0
    recs = _records(Path(cfg.out) / "t", "D1")
    assert len(recs) == len(runner.tasks)
    for r in recs:
        assert r["status"] == "ok" and r["answer_exact"] in ("TRUE", "FALSE", "UNKNOWN")
        assert r["answer_lenient"] == r["answer_exact"] and r["confidence"] == 0.8
        assert abs(sum(r["probs"].values()) - 1) < 1e-9 and max(r["probs"], key=r["probs"].get) == r["answer_exact"]
    cmd = cfg.server.command(Path("m"))
    assert cmd[cmd.index("--reasoning") + 1] == "off" and "--reasoning-budget" not in cmd


# --- choosing claims ------------------------------------------------------------------------------------------


def _levels(dataset_root: Path) -> list[int]:
    return sorted({json.loads(line)["level"] for line in (dataset_root / "ds" / "kbs.jsonl").open()})


def test_levels_and_fraction_select_claims(dataset_root, tmp_path):
    level = _levels(dataset_root)[-1]
    cfg = _config(tmp_path, claims_per_kb=None, levels=[level], fraction=0.1)
    tasks = Runner(cfg, dataset_root, log=lambda m: None).tasks
    assert tasks and all(t.kb.level == level for t in tasks)
    claims = [json.loads(line) for line in (dataset_root / "ds" / "claims.jsonl").open()]
    total = sum(c["level"] == level for c in claims)
    assert len(tasks) == round(total * 0.1)
    assert len({t.kb.kb_id for t in tasks}) == len({c["kb_id"] for c in claims if c["level"] == level})


def test_fraction_keeps_proportions(dataset_root, tmp_path):
    # Over all the rulebooks of a level together, every (world relation, expected answer) block gets its
    # quota rounded down or up, so the three answers stay as balanced as in the full dataset.
    cfg = _config(tmp_path, claims_per_kb=None, fraction=0.1)
    tasks = Runner(cfg, dataset_root, log=lambda m: None).tasks
    claims = [json.loads(line) for line in (dataset_root / "ds" / "claims.jsonl").open()]
    for level in _levels(dataset_root):
        pool = [c for c in claims if c["level"] == level]
        mine = [t.claim for t in tasks if t.claim.level == level]
        full = Counter((c["world_relation"], c["expected"]) for c in pool)
        got = Counter((c.world_relation, c.expected) for c in mine)
        for k, size in full.items():
            q = len(mine) * size / len(pool)
            assert math.floor(q) <= got.get(k, 0) <= math.ceil(q), (level, k, got.get(k, 0), q)
        # Groups and rulebooks: drawn at random within the blocks, so present and not concentrated.
        for field in ("group", "kb_id"):
            sizes = Counter(c[field] for c in pool)
            have = Counter(getattr(c, field) for c in mine)
            for name, size in sizes.items():
                q = len(mine) * size / len(pool)
                assert have.get(name, 0) <= 2 * q + 2, (level, field, have, q)
                if q >= 3:
                    assert have.get(name, 0) >= 1, (level, field, have, q)


# --- measures --------------------------------------------------------------------------------------------------


def test_metrics_merge_one_run_per_level(dataset_root, tmp_path):
    # One run per level; the measures merge them by model and also give the equal-weight accuracy.
    levels = _levels(dataset_root)[:2]
    dirs, sizes = [], 0
    for lv in levels:
        cfg = _config(tmp_path, claims_per_kb=None, levels=[lv], fraction=0.1, run_id=f"l{lv}")
        cfg.models = cfg.models[:1]
        runner = Runner(cfg, dataset_root, log=lambda m: None)
        runner.run()
        dirs.append(Path(cfg.out) / f"l{lv}")
        sizes += len(runner.tasks)
    m = compute(dirs, dataset_root)["models"]["Fake-A"]
    assert m["all"]["n"] == sizes and m["answered"] == sizes
    assert sorted(m["levels"]) == [str(lv) for lv in levels]
    w = m["all_equal_weight"]
    assert w["levels"] == 2
    accs = [m["levels"][str(lv)]["lenient"]["accuracy"]["value"] for lv in levels]
    assert abs(w["value"] - sum(accs) / 2) < 1e-12 and w["ci95"][0] <= w["value"] <= w["ci95"][1]
    with pytest.raises(SystemExit, match="more than one run"):
        compute([dirs[0], dirs[0]], dataset_root)


def _claim(i, relation="neutral", expected="TRUE", *, group="direct_fact", kb="K", level=1, depth=0):
    return ClaimRecord(
        claim_id=f"C{i}",
        kb_id=kb,
        level=level,
        group=group,
        subtype="s",
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


def _answers(claims, answers):
    return {c.claim_id: {"status": "ok", "answer_exact": a, "answer_lenient": a} for c, a in zip(claims, answers)}


def test_knowledge_effects():
    # Conflict effect: conflict against agree claims. Invented-words effect: against neutral TRUE or FALSE
    # claims (neutral UNKNOWN ones stay out). Silence effect: on UNKNOWN claims about real facts.
    spec = [
        ("conflict", "TRUE", "TRUE"),
        ("conflict", "FALSE", "TRUE"),  # 1 of 2
        ("agree", "TRUE", "TRUE"),
        ("agree", "FALSE", "FALSE"),  # 2 of 2
        ("neutral", "TRUE", "TRUE"),
        ("neutral", "FALSE", "TRUE"),  # 1 of 2
        ("neutral", "UNKNOWN", "FALSE"),  # outside the comparisons
        ("unsupported_true", "UNKNOWN", "TRUE"),  # toward the world
        ("unsupported_true", "UNKNOWN", "UNKNOWN"),  # no direction
        ("unsupported_false", "UNKNOWN", "FALSE"),  # toward the world
        ("unsupported_false", "UNKNOWN", "TRUE"),
    ]  # against the world
    claims = [_claim(i, rel, exp) for i, (rel, exp, _) in enumerate(spec)]
    m = measures(claims, _answers(claims, [a for *_, a in spec]), lenient=True)
    assert m["conflict_effect"]["value"] == -0.5
    assert m["invented_words_effect"]["value"] == 0 and m["neutral_tf_accuracy"]["n"] == 2
    p = m["silence_effect"]  # negative when the model answers in the direction of the world
    assert (p["toward"], p["away"], p["n"]) == (2, 1, 4) and p["value"] == -0.25
    assert p["ci95"][0] < -0.25 < p["ci95"][1]


def test_scores_without_modal_unknown_claims():
    spec = [
        ("modal", "UNKNOWN", "TRUE"),
        ("modal", "TRUE", "TRUE"),
        ("direct_fact", "UNKNOWN", "UNKNOWN"),
        ("direct_fact", "FALSE", "TRUE"),
    ]
    claims = [_claim(i, expected=e, group=g, depth=1) for i, (g, e, _) in enumerate(spec)]
    w = measures(claims, _answers(claims, [a for *_, a in spec]), lenient=True)["without_modal_unknown"]
    assert w["n"] == 3 and w["accuracy"]["k"] == 2
    assert (w["unknown_recall"]["k"], w["unknown_recall"]["n"]) == (1, 1)
    assert (w["modal_unknown"]["k"], w["modal_unknown"]["n"]) == (0, 1)


def test_label_measures_and_calibration():
    # Answer shares, the three kinds of error, kappa, macro-F1, calibration and accuracy per rulebook.
    spec = [
        ("K1", "TRUE", "TRUE", 0.9),
        ("K1", "FALSE", "TRUE", 0.6),  # right, inversion
        ("K2", "TRUE", "UNKNOWN", 0.5),
        ("K2", "UNKNOWN", "FALSE", 0.7),
    ]  # abstention, invention
    claims = [_claim(i, expected=e, kb=kb) for i, (kb, e, _, _) in enumerate(spec)]
    responses = {}
    for c, (_, _, a, conf) in zip(claims, spec):
        probs = {lab: (conf if lab == a else (1 - conf) / 2) for lab in ("TRUE", "FALSE", "UNKNOWN")}
        responses[c.claim_id] = {
            "status": "ok",
            "answer_exact": a,
            "answer_lenient": a,
            "probs": probs,
            "confidence": conf,
        }
    m = measures(claims, responses, lenient=True)
    e = m["errors"]
    assert (e["inversion"]["k"], e["inversion"]["n"]) == (1, 3)
    assert (e["abstention"]["k"], e["invention"]["k"], e["invention"]["n"]) == (1, 1, 1)
    assert m["answer_share"]["TRUE"]["k"] == 2
    # expected and given answers: half TRUE, a quarter FALSE, a quarter UNKNOWN; one right answer
    assert abs(m["chance_accuracy"] - 0.375) < 1e-12
    assert abs(m["kappa"] - (0.25 - 0.375) / (1 - 0.375)) < 1e-12
    assert abs(m["macro_f1"] - 0.5 / 3) < 1e-12  # F1 of TRUE 0.5; FALSE and UNKNOWN 0
    assert (m["answerable_accuracy"]["k"], m["answerable_accuracy"]["n"]) == (1, 3)
    assert (m["unknown_recall"]["k"], m["unknown_recall"]["n"]) == (0, 1)
    cal = m["calibration"]
    assert cal["n"] == 4 and abs(cal["confidence_right"] - 0.9) < 1e-12
    assert abs(cal["brier_skill"] - (1 - cal["brier"] / (2 / 3))) < 1e-12
    assert m["per_kb"]["K1"]["k"] == 1 and m["per_kb"]["K2"]["k"] == 0


def test_same_question_uses_groups_shared_by_levels():
    # At depth 2 group "b" exists only at level 3, so it stays out of the comparison; UNKNOWN claims do
    # not count; a depth present at one level only does not appear.
    spec = [
        (2, "a", 2, "TRUE", "TRUE"),
        (2, "a", 2, "FALSE", "TRUE"),
        (2, "a", 2, "UNKNOWN", "TRUE"),
        (3, "a", 2, "TRUE", "TRUE"),
        (3, "b", 2, "TRUE", "FALSE"),
        (3, "a", 5, "TRUE", "TRUE"),
    ]
    claims = [_claim(i, expected=e, group=g, level=lv, depth=d) for i, (lv, g, d, e, _) in enumerate(spec)]
    s = same_question(claims, _answers(claims, [a for *_, a in spec]))
    assert list(s) == ["2"] and s["2"]["groups"] == ["a"]
    assert (s["2"]["levels"]["2"]["k"], s["2"]["levels"]["2"]["n"]) == (1, 2)
    assert (s["2"]["levels"]["3"]["k"], s["2"]["levels"]["3"]["n"]) == (1, 1)


def test_status_shows_three_levels():
    state = {
        "pid": os.getpid(),
        "phase": "answers",
        "step": "L2-KB3",
        "step_done": 20,
        "step_total": 225,
        "plan_label": "claims",
        "plan_done": 3020,
        "plan_total": 13000,
        "updated_at": "x",
        "updated_ts": 0,
        "elapsed_s": 10,
        "eta_s": 100,
        "finished": False,
        "stale_after_s": 1e12,
        "last": {},
        "step_label": "Rulebook",
        "bars": [["Model", 620, 2600, "Model-4B (3 of 5)"]],
    }
    lines = render(state, now=1).splitlines()
    assert lines[1].startswith("Plan") and lines[2].startswith("Model") and lines[3].startswith("Rulebook")
    assert "Model-4B (3 of 5)" in lines[2] and "20/225" in lines[3]


# --- knowledge probe ------------------------------------------------------------------------------------------


def test_probe_sentences_come_in_opposite_pairs():
    # For every fact its true and false versions, for every ability "can"; every sentence with its
    # negation, and opposite real-world values.
    items = probe.sentences(ROOT / "data" / "tables")
    pairs: dict = {}
    for s in items:
        pairs.setdefault(s["pair"], {})[s["polarity"]] = s
    t = load_tables(ROOT / "data" / "tables")
    assert len(pairs) == sum(1 + len(r.false) for r in t.facts) + len(t.abilities)
    assert all(set(p) == {"pos", "neg"} and p["pos"]["world"] != p["neg"]["world"] for p in pairs.values())
    assert len({s["text"] for s in items}) == len(items)


def test_probe_knowledge_needs_both_sentences_right():
    def claim(i, pair, logic, expected):
        return ClaimRecord(
            claim_id=f"W{i}",
            kb_id="W",
            level=0,
            group="world_fact",
            subtype=pair,
            text="t",
            logic=logic,
            expected=expected,
            z3=expected,
            world_relation="agree",
            category="nature",
            depth=0,
            reasoning=[],
            support=[],
            traps=[],
        )

    cs = [
        claim(0, "a#0", "pos", "TRUE"),
        claim(1, "a#0", "neg", "FALSE"),
        claim(2, "b#0", "pos", "TRUE"),
        claim(3, "b#0", "neg", "FALSE"),
    ]
    answers = ["TRUE", "FALSE", "TRUE", "TRUE"]  # "b" always answers TRUE: not known
    k = probe.knowledge(cs, _answers(cs, answers))
    assert k["a#0"]["known"] and not k["b#0"]["known"]


def test_world_probe_runs_and_links_benchmark_claims(dataset_root, tmp_path):
    # The probe runs with the test program (task "world"), and the analysis links the real-world claims
    # of the benchmark to the table sentences.
    probe.build(ROOT / "data" / "tables", dataset_root / "wp")
    base = _config(tmp_path, claims_per_kb=None)
    d1 = ModelSpec(name="D1", path=base.models[0].path, kind="systemone")
    world = _config(tmp_path, run_id="w", datasets=["wp"], claims_per_kb=None, task="world", models=[d1])
    world.server.reasoning = False
    assert Runner(world, dataset_root, log=lambda m: None).run() == 0
    bench = _config(tmp_path, run_id="b", models=[d1])
    bench.server.reasoning = False
    assert Runner(bench, dataset_root, log=lambda m: None).run() == 0
    out = Path(world.out)
    r = probe.analyze(out / "w", out / "b", dataset_root, ROOT / "data" / "tables")
    m = r["models"]["D1"]
    assert m["pairs"] == len({s["pair"] for s in probe.sentences(ROOT / "data" / "tables")})
    assert sum(m["answers"].values()) == 2 * m["pairs"]
    lv = next(iter(m["levels"].values()))
    assert lv["linked"] > 0 and lv["linked"] + lv["not_linked"] == lv["world_claims"]
    assert probe.summary_md(r).startswith("# Knowledge probe")
