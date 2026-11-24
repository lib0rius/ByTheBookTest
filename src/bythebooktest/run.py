"""The test program: ask a model every claim of the datasets and record every answer.

Usage::

    bythebooktest run --config configs/d1-2b.toml

For each model it starts ``llama-server``, sends the claims rulebook by rulebook, and appends every
response to ``<out>/<run_id>/responses/<model>.jsonl`` as soon as it arrives. A run can be stopped at any
time: started again, it resumes from the first missing claim. Progress is shown by
``bythebooktest status <out>/<run_id>/status.json``.

The run key in ``run.json`` records the configuration and the fingerprints of prompts and datasets; a run
can only be resumed with the same key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import FrameType

from . import prompt
from .answer import parse_exact, parse_lenient
from .config import ModelSpec, RunConfig
from .records import ClaimRecord, KBRecord
from .server import LlamaServer, RequestTimeout, ServerError
from .status import StatusWriter

ATTEMPTS = 3
"""Attempts per claim when the server fails; a timeout is not retried."""
BUDGET_SLACK = 16
"""Tolerance, in tokens, to recognise a thinking that reached its budget."""


@dataclass
class Task:
    """One claim to ask, with its rulebook."""

    kb: KBRecord
    claim: ClaimRecord


def load_dataset(
    dirs: list[Path],
    claims_per_kb: int | None = None,
    only_kbs: list[str] | None = None,
    levels: list[int] | None = None,
    fraction: float | None = None,
) -> tuple[list[KBRecord], list[Task], str]:
    """Read the datasets and choose the claims to ask.

    Args:
        dirs: Dataset folders, each with ``kbs.jsonl`` and ``claims.jsonl``.
        claims_per_kb: Ask only this many claims per rulebook (stratified sample).
        only_kbs: Ask only the claims of these rulebooks.
        levels: Ask only the claims of these levels.
        fraction: Ask only this share of the claims (stratified sample).

    Returns:
        All rulebooks, the tasks in file order, and the fingerprint of the dataset files.
    """
    digest = hashlib.sha256()
    kbs: list[KBRecord] = []
    tasks: list[Task] = []
    for d in dirs:
        for name in ("kbs.jsonl", "claims.jsonl"):
            digest.update((d / name).read_bytes())
        level_kbs = [KBRecord.model_validate_json(line) for line in (d / "kbs.jsonl").open(encoding="utf-8")]
        by_id = {k.kb_id: k for k in level_kbs}
        claims = [ClaimRecord.model_validate_json(line) for line in (d / "claims.jsonl").open(encoding="utf-8")]
        chosen_kbs = [
            kb
            for kb in level_kbs
            if (only_kbs is None or kb.kb_id in only_kbs) and (levels is None or kb.level in levels)
        ]
        per_kb = {kb.kb_id: [c for c in claims if c.kb_id == kb.kb_id] for kb in chosen_kbs}
        if claims_per_kb is not None or fraction is not None:
            # The sample is drawn over all the chosen rulebooks of the level together: within a single
            # rulebook most quotas would be fractions, and rounding would favour the large strata.
            pool = [c for kb in chosen_kbs for c in per_kb[kb.kb_id]]
            if claims_per_kb is not None:
                n = claims_per_kb * len(chosen_kbs)
            else:
                assert fraction is not None
                n = max(len(chosen_kbs), round(len(pool) * fraction))
            keep = {c.claim_id for c in stratified_sample(pool, n)}
            per_kb = {k: [c for c in v if c.claim_id in keep] for k, v in per_kb.items()}
        for kb in chosen_kbs:
            tasks.extend(Task(by_id[kb.kb_id], c) for c in per_kb[kb.kb_id])
        kbs.extend(level_kbs)
    return kbs, tasks, digest.hexdigest()


def _stratum(c: ClaimRecord) -> tuple[int, str, str]:
    # Strata follow the mix of expected answers and world relations the main measures depend on;
    # groups and rulebooks are spread by the random draw.
    return (c.level, c.world_relation, c.expected)


def stratified_sample(claims: list[ClaimRecord], n: int) -> list[ClaimRecord]:
    """Draw a stratified random sample of ``n`` claims, deterministically, keeping file order.

    Strata are level, world relation and expected answer. Each stratum gets its proportional quota,
    rounded by the largest-remainder method; within a stratum, claims are drawn with a fixed seed, so
    groups and rulebooks are spread without preference.
    """
    if n >= len(claims):
        return claims
    strata: dict[tuple, list[ClaimRecord]] = {}
    for c in claims:
        strata.setdefault(_stratum(c), []).append(c)
    exact = {k: n * len(v) / len(claims) for k, v in strata.items()}
    quota = {k: int(q) for k, q in exact.items()}
    for k in sorted(strata, key=lambda k: (-(exact[k] - quota[k]), k))[: n - sum(quota.values())]:
        quota[k] += 1
    rng = random.Random("bythebook-subset-v1")  # fixed seed name: changing it changes every sample
    chosen: set[str] = set()
    for k in sorted(strata):
        members = sorted(strata[k], key=lambda c: c.claim_id)
        chosen.update(c.claim_id for c in rng.sample(members, quota[k]))
    return [c for c in claims if c.claim_id in chosen]


def systemone_result(resp: dict) -> dict:
    """Fields of a response record from a ``/v1/systemone`` reply: choice, option probabilities and
    confidence, in the same layout as chat responses.
    """
    answer = (resp.get("answers") or {}).get("claim") or {}
    choice = answer.get("choice")
    valid = choice if choice in prompt.SYSTEMONE_CRITERIA else None
    usage = resp.get("usage") or {}
    return {
        "answer_exact": valid,
        "answer_lenient": valid,
        "content": json.dumps(answer, ensure_ascii=False),
        "reasoning": "",
        "finish_reason": None,
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": 0,
        "budget_hit": False,
        "probs": answer.get("probabilities"),
        "confidence": answer.get("confidence"),
        "timings": resp.get("timings") or {},
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Runner:
    """Runs one configuration: one server per model, claims in parallel slots, answers on disk.

    Args:
        cfg: The run configuration.
        root: Folder that relative dataset paths start from.
        log: Where log lines are printed (they are also appended to ``run.log``).
        retry_timeouts: Ask again only the claims whose last response is a timeout. The new response is
            appended to the file and the measures read the last one. Only ``server.parallel`` may differ
            from the original run.
    """

    def __init__(
        self, cfg: RunConfig, root: Path, log: Callable[[str], None] = print, retry_timeouts: bool = False
    ) -> None:
        self.cfg = cfg
        self.root = root
        self.retry_timeouts = retry_timeouts
        self.dir = Path(cfg.out) / cfg.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (self.dir / "run.log").open("a", encoding="utf-8")
        self._print = log
        self.lock = threading.Lock()
        self.restart_lock = threading.Lock()
        self.stopping = threading.Event()
        self.server: LlamaServer | None = None
        _, self.tasks, self.dataset_sha = load_dataset(
            [root / d for d in cfg.datasets], cfg.claims_per_kb, cfg.kbs, cfg.levels, cfg.fraction
        )

    def handle_stop(self, signum: int, frame: FrameType | None) -> None:
        """Handle a stop request (SIGTERM, for example ``systemctl stop``).

        The server is stopped at once, so requests in progress end instead of waiting for their time
        limit, and nothing is restarted. Responses already written stay; interrupted ones are asked
        again when the run resumes.
        """
        self.stopping.set()
        self.log(f"stop requested (signal {signum}): stopping llama-server")
        if self.server is not None:
            self.server.stop()
        raise SystemExit(128 + signum)

    def log(self, msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        with self.lock:
            self.log_file.write(line + "\n")
            self.log_file.flush()
        self._print(line)

    # --- run key and record -------------------------------------------------------------------------

    def key(self) -> dict:
        """What identifies the run: configuration (without ``out``), prompts and datasets."""
        cfg = self.cfg.model_dump()
        cfg.pop("out")
        return {
            "config": cfg,
            "prompt_sha256": prompt.prompt_sha256(),
            "request_sha256": prompt.request_sha256(),
            "dataset_sha256": self.dataset_sha,
        }

    def check_key(self) -> dict:
        """Create ``run.json``, or check that the existing one has the same key; return its content."""
        path = self.dir / "run.json"
        key = self.key()
        if not path.exists():
            saved = {"key": key, "created_at": datetime.now().isoformat(timespec="seconds"), "models": {}}
            self.save_run(saved)
            return saved
        saved = json.loads(path.read_text(encoding="utf-8"))
        if self.retry_timeouts:
            # Only the number of slots may change (for example 1 instead of 4); when and with how many
            # slots the retry ran is recorded.
            old, new = json.loads(json.dumps(saved["key"])), json.loads(json.dumps(key))
            old["config"]["server"].pop("parallel", None)
            new["config"]["server"].pop("parallel", None)
            if old != new:
                raise SystemExit(
                    f"{path}: to retry timed-out claims, configuration, prompts and datasets "
                    "must stay those of the run (only server.parallel may change)"
                )
            saved.setdefault("retries", []).append(
                {
                    "at": datetime.now().isoformat(timespec="seconds"),
                    "parallel": self.cfg.server.parallel,
                    "what": "timed-out claims",
                }
            )
            self.save_run(saved)
            return saved
        if saved["key"] != key:
            raise SystemExit(
                f"{path}: the existing run has a different configuration, prompts or datasets; use another run_id"
            )
        return saved

    def save_run(self, saved: dict) -> None:
        path = self.dir / "run.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)

    def responses_path(self, model: ModelSpec) -> Path:
        return self.dir / "responses" / f"{model.name}.jsonl"

    def done_ids(self, model: ModelSpec) -> set[str]:
        """Claims that already have a response (with ``retry_timeouts``: a response that is not a timeout)."""
        path = self.responses_path(model)
        if not path.exists():
            return set()
        # A stop while writing leaves the last line incomplete: it is cut, so the next response does not
        # join that line, and the claim is asked again.
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            with path.open("r+b") as f:
                f.truncate(data.rfind(b"\n") + 1)
        last: dict[str, str] = {}
        for line in path.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                last[rec["claim_id"]] = rec.get("status", "")
            except (json.JSONDecodeError, KeyError):
                continue
        if self.retry_timeouts:
            return {c for c, s in last.items() if s != "timeout"}
        return set(last)

    # --- one response ----------------------------------------------------------------------------------

    def ask(self, server: LlamaServer, task: Task, kind: str = "chat") -> dict:
        """Ask one claim and return the response record.

        A server error restarts the server and tries again, up to :data:`ATTEMPTS` times. A timeout
        restarts the server and is recorded as such, without retrying.
        """
        if kind == "systemone":
            payload = (
                prompt.world_request(task.claim.text)
                if self.cfg.task == "world"
                else prompt.systemone_request(task.kb, task.claim.text)
            )
            call = server.systemone
        else:
            payload = {**self.cfg.payload_base(), "messages": prompt.messages(task.kb, task.claim.text)}
            call = server.chat
        record = {
            "claim_id": task.claim.claim_id,
            "kb_id": task.kb.kb_id,
            "level": task.kb.level,
            "expected": task.claim.expected,
            "seed": self.cfg.seed,
        }
        error = ""
        for attempt in range(1, ATTEMPTS + 1):
            t0 = time.time()
            if self.stopping.is_set():
                return {**record, "status": "stopped"}
            failure: Exception | None
            try:
                resp = call(payload, self.cfg.claim_timeout_s)
            except (RequestTimeout, ServerError) as e:
                if self.stopping.is_set():
                    return {**record, "status": "stopped"}  # server stopped by the shutdown: not written
                failure = e
            else:
                failure = None
            if isinstance(failure, RequestTimeout):
                self.log(
                    f"{task.claim.claim_id}: time limit of {self.cfg.claim_timeout_s:.0f} s exceeded, "
                    "restarting the server"
                )
                self._restart(server, t0)
                return {**record, "status": "timeout", "attempt": attempt, "wall_s": round(time.time() - t0, 2)}
            if failure is not None:
                error = str(failure)
                self.log(f"{task.claim.claim_id}: server error ({error[:200]}), attempt {attempt}")
                self._restart(server, t0)
                continue
            if kind == "systemone":
                return {
                    **record,
                    "status": "ok",
                    "attempt": attempt,
                    "wall_s": round(time.time() - t0, 2),
                    "t_start": round(t0, 3),
                    "t_end": round(time.time(), 3),
                    **systemone_result(resp),
                }
            choice = resp["choices"][0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            timings = resp.get("timings", {})
            usage = resp.get("usage", {})
            reasoning_tokens = self._count_tokens(server, reasoning)
            return {
                **record,
                "status": "ok",
                "attempt": attempt,
                "wall_s": round(time.time() - t0, 2),
                "t_start": round(t0, 3),
                "t_end": round(time.time(), 3),
                "answer_exact": parse_exact(content),
                "answer_lenient": parse_lenient(content),
                "content": content,
                "reasoning": reasoning,
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": reasoning_tokens,
                # Counting the separated thinking with /tokenize can differ by a few tokens from the
                # server's own count (end-of-thinking markers): 4,095 out of 4,096 counts as the budget.
                "budget_hit": (
                    reasoning_tokens is not None and reasoning_tokens >= self.cfg.server.reasoning_budget - BUDGET_SLACK
                ),
                "timings": timings,
            }
        return {**record, "status": "error", "attempt": ATTEMPTS, "error": error[:1000]}

    def _count_tokens(self, server: LlamaServer, text: str) -> int | None:
        if not text:
            return 0
        try:
            return server.count_tokens(text)
        except (ServerError, RequestTimeout, KeyError):
            return None

    def _restart(self, server: LlamaServer, since: float) -> None:
        with self.restart_lock:  # with several slots, one restart at a time
            if self.stopping.is_set():
                raise ServerError("shutting down: no restart")
            if server.started_at > since and server.alive():
                return  # another request already restarted it after this one began
            try:
                server.restart()
            except ServerError as e:
                self.log(f"server restart failed: {e}")
                raise

    # --- main loop ---------------------------------------------------------------------------------------

    def run(self) -> int:
        """Run every model of the configuration; return the process exit code."""
        saved = self.check_key()
        per_model = len(self.tasks)
        total = per_model * len(self.cfg.models)
        stale = max(300.0, self.cfg.claim_timeout_s + 120)
        status = StatusWriter(self.dir / "status.json", total, "claims", stale_after_s=stale)
        sizes = {m.name: Path(m.path).stat().st_size for m in self.cfg.models}
        done_all = {m.name: self.done_ids(m) for m in self.cfg.models}
        status.plan_offset = sum(len(v) for v in done_all.values())
        self.log(
            f"Start of run {self.cfg.run_id}: {len(self.cfg.models)} models, {per_model} claims per model, "
            f"{total} in all; {status.plan_offset} already done"
        )
        plan_done = status.plan_offset
        for mi, model in enumerate(self.cfg.models):
            done = done_all[model.name]
            todo = [t for t in self.tasks if t.claim.claim_id not in done]
            if not todo:
                self.log(f"{model.name}: already complete")
                continue
            if model.name not in saved["models"]:
                status.update(
                    phase=f"fingerprint of model {model.name}",
                    step=model.name,
                    step_done=0,
                    step_total=1,
                    plan_done=plan_done,
                    step_label="Rulebook",
                )
                saved["models"][model.name] = {
                    "path": model.path,
                    "size": sizes[model.name],
                    "sha256": file_sha256(Path(model.path)),
                }
                self.save_run(saved)
            server = LlamaServer(self.cfg.server, Path(model.path), self.dir / "server" / f"{model.name}.log")
            status.update(
                phase=f"starting llama-server with {model.name}",
                step=model.name,
                step_done=0,
                step_total=1,
                plan_done=plan_done,
                step_label="Rulebook",
            )
            self.log(f"{model.name}: starting llama-server ({len(todo)} claims to ask)")
            self.server = server
            try:
                server.start()
                props = server.props()
                saved["models"][model.name]["server_props"] = {
                    k: props.get(k) for k in ("build_info", "model_path", "n_ctx", "total_slots", "chat_template")
                }
                saved["models"][model.name]["devices"] = server.devices()
                self.save_run(saved)
                self.log(f"{model.name}: devices {saved['models'][model.name]['devices']}")
                plan_done = self._run_model(server, model, mi, done, todo, status, plan_done, sizes)
            finally:
                server.stop()
                self.server = None
            if self.stopping.is_set():
                self.log("run stopped; started again, it resumes from the first missing claim")
                return 143
            self.log(f"{model.name}: complete")
        status.update(
            phase="finished", step="", step_done=0, step_total=0, plan_done=total, finished=True, step_label="Rulebook"
        )
        self.log("Run complete")
        return 0

    def _run_model(
        self,
        server: LlamaServer,
        model: ModelSpec,
        mi: int,
        done: set[str],
        todo: list[Task],
        status: StatusWriter,
        plan_done: int,
        sizes: dict,
    ) -> int:
        per_model = len(self.tasks)
        model_done = len(done)
        started = time.time()
        here = 0
        stats = {
            "ok": 0,
            "correct": 0,
            "invalid": 0,
            "budget": 0,
            "timeout": 0,
            "tps": 0.0,
            "cache": 0,
            "tokens": 0,
            "invalid_exact": 0,
        }
        out = self.responses_path(model)
        out.parent.mkdir(parents=True, exist_ok=True)
        kb_order = list(dict.fromkeys(t.kb.kb_id for t in self.tasks))
        kb_sizes = {k: sum(1 for t in self.tasks if t.kb.kb_id == k) for k in kb_order}
        later = self.cfg.models[mi + 1 :]

        def eta() -> tuple[float | None, float | None]:
            if here == 0:
                return None, None
            per_claim = (time.time() - started) / here
            this = per_claim * (per_model - model_done)
            # Later models: the same speed scaled by file size (generation is bound by memory
            # bandwidth); replaced by the real speed when they start.
            rest = sum(per_claim * sizes[m.name] / sizes[model.name] * per_model for m in later)
            return this, this + rest

        with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(self.cfg.server.parallel) as pool:
            for kb_id in kb_order:
                batch = [t for t in todo if t.kb.kb_id == kb_id]
                if not batch:
                    continue
                kb_done = kb_sizes[kb_id] - len(batch)
                # Responses are written as they arrive, not in sending order: a slow claim must not
                # hold back the responses of the other slots.
                futures = [pool.submit(self.ask, server, t, model.kind) for t in batch]
                for fut in as_completed(futures):
                    rec = fut.result()
                    if rec["status"] == "stopped" or self.stopping.is_set():
                        continue  # interrupted by the shutdown: asked again when the run resumes
                    rec["model"] = model.name
                    if self.retry_timeouts:
                        rec["retry_of"] = "timeout"
                        rec["parallel"] = self.cfg.server.parallel
                    with self.lock:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                    here += 1
                    model_done += 1
                    plan_done += 1
                    kb_done += 1
                    if rec["status"] == "ok":
                        stats["ok"] += 1
                        stats["correct"] += rec["answer_lenient"] == rec["expected"]
                        stats["invalid"] += rec["answer_lenient"] is None
                        stats["invalid_exact"] += rec["answer_exact"] is None
                        stats["budget"] += bool(rec["budget_hit"])
                        stats["tps"] = rec["timings"].get("predicted_per_second") or stats["tps"]
                        stats["cache"] = rec["timings"].get("cache_n", stats["cache"])
                        stats["tokens"] += rec.get("completion_tokens") or 0
                    elif rec["status"] == "timeout":
                        stats["timeout"] += 1
                    this_eta, plan_eta = eta()
                    status.update(
                        phase=f"answers of {model.name}",
                        step=kb_id,
                        step_done=kb_done,
                        step_total=kb_sizes[kb_id],
                        plan_done=plan_done,
                        step_label="Rulebook",
                        bars=[("Model", model_done, per_model, f"{model.name} ({mi + 1} of {len(self.cfg.models)})")],
                        eta_s=plan_eta,
                        last={
                            "accuracy so far": round(stats["correct"] / stats["ok"], 3) if stats["ok"] else "-",
                            "invalid": stats["invalid"],
                            "invalid (exact)": stats["invalid_exact"],
                            "thinking budget hit": stats["budget"],
                            "timeouts": stats["timeout"],
                            # With several slots the speed of one claim is not the speed of the run:
                            # both are shown, with the average time per finished claim.
                            "tokens/s per claim": round(stats["tps"], 1),
                            "tokens/s overall": round(stats["tokens"] / (time.time() - started), 1),
                            "s per claim": round((time.time() - started) / here, 1),
                            "tokens from cache": stats["cache"],
                            "left for this model": fmt_hms(this_eta) if this_eta else "-",
                        },
                    )
                self.log(
                    f"{model.name} {kb_id}: {kb_sizes[kb_id]} claims; so far {stats['ok']} answers, "
                    f"{stats['correct']} right, {stats['invalid']} invalid, "
                    f"{stats['budget']} at the thinking budget, {stats['timeout']} timeouts"
                )
        return plan_done


def _csv(value: str | None) -> list[str] | None:
    return value.split(",") if value else None


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``run`` command."""
    p = sub.add_parser(
        "run",
        help="ask a model every claim and record the answers",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="run file, for example configs/d1-2b.toml")
    p.add_argument("--models", help="only the models whose name ends with these, comma-separated")
    p.add_argument("--run-id", help="replaces run_id of the configuration")
    p.add_argument("--parallel", type=int, help="requests served at the same time (server slots)")
    p.add_argument("--kbs", help="only these rulebooks, for example L1-KB1,L3-KB1")
    p.add_argument("--levels", help="only these levels, for example 1 or 1,2")
    p.add_argument("--claims-per-kb", type=int, help="only N claims per rulebook (stratified sample)")
    p.add_argument("--fraction", type=float, help="only this share of the claims (stratified sample), for example 0.1")
    p.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="ask again only the claims whose last answer is a timeout; only --parallel may change",
    )
    p.set_defaults(func=command)


def command(args: argparse.Namespace) -> int:
    cfg = RunConfig.load(
        args.config,
        models=_csv(args.models),
        run_id=args.run_id,
        parallel=args.parallel,
        kbs=_csv(args.kbs),
        claims_per_kb=args.claims_per_kb,
        fraction=args.fraction,
        levels=[int(x) for x in args.levels.split(",")] if args.levels else None,
    )
    runner = Runner(cfg, Path.cwd(), retry_timeouts=args.retry_timeouts)
    signal.signal(signal.SIGTERM, runner.handle_stop)
    return runner.run()
