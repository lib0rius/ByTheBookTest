"""Progress files of long runs."""

import json

from bythebooktest.status import StatusWriter, render


def test_status_file_and_render(tmp_path):
    path = tmp_path / "status.json"
    writer = StatusWriter(path, plan_total=200, plan_label="claim checks")
    writer.update(
        phase="checking",
        step="L2-KB1, round 1",
        step_done=50,
        step_total=100,
        plan_done=50,
        last={"claims per second": "12.0"},
    )
    state = json.loads(path.read_text())
    text = render(state)
    assert "25.0%" in text and "50.0%" in text
    assert "process running" in text
    assert "claims per second 12.0" in text


def test_stale_status_is_flagged(tmp_path):
    path = tmp_path / "status.json"
    writer = StatusWriter(path, plan_total=10, plan_label="steps")
    writer.update(phase="x", step="y", step_done=1, step_total=10, plan_done=1)
    state = json.loads(path.read_text())
    assert "POSSIBLE STALL" in render(state, now=state["updated_ts"] + 600)


def test_resumed_work_does_not_inflate_the_estimate(tmp_path):
    # 75 steps resumed from a checkpoint, 5 done now: the estimate uses the 5, not the 80.
    path = tmp_path / "status.json"
    writer = StatusWriter(path, plan_total=100, plan_label="steps")
    writer.plan_offset = 75
    writer.started -= 50  # 50 s for 5 steps: 10 s per step
    writer.update(phase="x", step="y", step_done=5, step_total=25, plan_done=80)
    state = json.loads(path.read_text())
    assert 190 <= state["eta_s"] <= 210  # 20 steps left


def test_stale_threshold_follows_the_run(tmp_path):
    # On Level 3 a group of claims can take minutes: the stall threshold follows the time limit.
    path = tmp_path / "status.json"
    writer = StatusWriter(path, plan_total=10, plan_label="steps", stale_after_s=900)
    writer.update(phase="x", step="y", step_done=1, step_total=10, plan_done=1)
    state = json.loads(path.read_text())
    assert "process running" in render(state, now=state["updated_ts"] + 600)
    assert "POSSIBLE STALL" in render(state, now=state["updated_ts"] + 1000)
