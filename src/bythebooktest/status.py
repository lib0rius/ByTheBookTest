"""Progress reporting for long runs.

The working program writes a small JSON status file after every step (:class:`StatusWriter`). The
``bythebooktest status <file>`` command prints it with progress bars, elapsed time, estimated time left,
a sign of life, and a "possible stall" warning when the last update is older than a threshold. The file
can be read at any time, from another machine too, without disturbing the run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

STALE_SECONDS = 120
"""Default time without updates after which a run is reported as possibly stalled."""


def _bar(fraction: float, width: int = 20) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _hms(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class StatusWriter:
    """Writes the status file: phase, progress of the current step and of the whole plan, times, metrics.

    Args:
        path: The status file; written atomically, so readers never see half a file.
        plan_total: Total units of work in the plan (for example claims).
        plan_label: What a unit is, shown next to the counts (for example ``"claims"``).
        stale_after_s: Time without updates after which readers report a possible stall.
    """

    def __init__(self, path: Path, plan_total: int, plan_label: str, stale_after_s: float = STALE_SECONDS) -> None:
        self.path = path
        self.stale_after_s = stale_after_s
        self.started = time.time()
        self.plan_total = plan_total
        self.plan_label = plan_label
        self.plan_done = 0
        self.plan_offset = 0
        """Work resumed from a checkpoint: counted as done, but not part of this run's pace."""
        self.state: dict = {}
        path.parent.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        *,
        phase: str,
        step: str,
        step_done: int,
        step_total: int,
        plan_done: int | None = None,
        last: dict | None = None,
        finished: bool = False,
        bars: list[tuple[str, int, int, str]] | None = None,
        step_label: str = "Step",
        eta_s: float | None = None,
    ) -> None:
        """Record the current state.

        Args:
            phase: What the program is doing, in words.
            step: The name of the current step.
            step_done: Units done in the current step.
            step_total: Units in the current step.
            plan_done: Units done in the whole plan, if changed.
            last: Latest useful measures (for example accuracy so far); kept until replaced.
            finished: True on the final update.
            bars: Extra progress rows between plan and step, as ``(label, done, total, name)``.
            step_label: Label of the step row.
            eta_s: Time left estimated by the caller, replacing the proportional estimate.
        """
        if plan_done is not None:
            self.plan_done = plan_done
        elapsed = time.time() - self.started
        # Only work done in this run sets the pace: work resumed from a checkpoint cost no time here.
        here = self.plan_done - self.plan_offset
        eta = elapsed * (self.plan_total - self.plan_done) / here if here > 0 else None
        if eta_s is not None:
            eta = eta_s
        self.state = {
            "pid": os.getpid(),
            "phase": phase,
            "step": step,
            "step_done": step_done,
            "step_total": step_total,
            "plan_label": self.plan_label,
            "plan_done": self.plan_done,
            "plan_total": self.plan_total,
            "started_at": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "updated_ts": time.time(),
            "elapsed_s": round(elapsed, 1),
            "eta_s": None if eta is None else round(eta, 1),
            "finished": finished,
            "stale_after_s": self.stale_after_s,
            "last": last or self.state.get("last", {}),
            "bars": [list(b) for b in bars] if bars else [],
            "step_label": step_label,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def render(state: dict, now: float | None = None) -> str:
    """Format a status as a few lines of text for the terminal."""
    now = now or time.time()
    age = now - state["updated_ts"]
    plan = state["plan_done"] / state["plan_total"] if state["plan_total"] else 0.0
    step = state["step_done"] / state["step_total"] if state["step_total"] else 0.0
    eta = state["eta_s"]
    end = "--" if eta is None else datetime.fromtimestamp(state["updated_ts"] + eta).strftime("%Y-%m-%d %H:%M")
    stale_after = int(state.get("stale_after_s", STALE_SECONDS))
    if state["finished"]:
        life = "finished"
    elif not _alive(state["pid"]):
        life = "PROCESS NOT RUNNING"
    elif age > stale_after:
        life = f"POSSIBLE STALL (no update for {int(age)} s, threshold {stale_after} s)"
    else:
        life = "process running"
    lines = [
        f"Phase    {state['phase']}",
        f"Plan     {_bar(plan)} {plan:5.1%}  {state['plan_done']}/{state['plan_total']} {state['plan_label']}",
        *(
            f"{label:<8} {_bar(done / total if total else 0.0)} "
            f"{(done / total if total else 0.0):5.1%}  {done}/{total}  {name}"
            for label, done, total, name in state.get("bars", [])
        ),
        f"{state.get('step_label', 'Step'):<8} {_bar(step)} {step:5.1%}  "
        f"{state['step_done']}/{state['step_total']}  {state['step']}",
        f"Elapsed  {_hms(state['elapsed_s'])}   Left {_hms(eta)}   Expected end {end}",
        f"Last update {state['updated_at']} ({int(age)} s ago)   {life}",
    ]
    if state.get("last"):
        lines.append("Latest   " + "   ".join(f"{k} {v}" for k, v in state["last"].items()))
    return "\n".join(lines)


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``status`` command."""
    p = sub.add_parser(
        "status",
        help="show the progress of a running command",
        description="Print the status file written by generate, prune or run.",
    )
    p.add_argument("status_file", type=Path)
    p.set_defaults(func=command)


def command(args: argparse.Namespace) -> int:
    path: Path = args.status_file
    if not path.exists():
        print(f"{path}: no status yet (has the run started?)")
        return 1
    print(render(json.loads(path.read_text(encoding="utf-8"))))
    return 0
