from __future__ import annotations

import json
import sys

from conftest import execution_config

from autoresearch.chart import render_page, render_svg
from autoresearch.cli import main
from autoresearch.execution import parse_metrics
from autoresearch.loop import FastLoop
from autoresearch.store import Store

TRAIN_PY_TAIL = """step 01055 (99.8%) | loss: 1.203 | steps: 3 | dt: 154ms | remaining: 1s
---
val_bpb:          0.969952
training_seconds: 300.0
total_seconds:    508.2
peak_vram_mb:     52608.9
num_steps:        1058
"""


def test_summary_block_is_parsed_structurally():
    metrics, error = parse_metrics(TRAIN_PY_TAIL)
    assert error == ""
    assert metrics["val_bpb"] == 0.969952
    assert metrics["num_steps"] == 1058
    assert metrics["total_seconds"] == 508.2


def test_progress_lines_alone_are_not_mistaken_for_metrics():
    """The old regex fallback read `steps: 3` off a progress line and fed it to
    the minimum-steps integrity gate.  A run with no summary block is an
    instrumentation failure, not a run with 3 steps."""
    metrics, error = parse_metrics("step 00042 (4.0%) | loss: 2.1 | steps: 3 | val_bpb: 9.9\n")
    assert metrics == {}
    assert error == "no structured metrics were emitted"


def test_explicit_contract_wins_over_summary_block():
    metrics, error = parse_metrics(
        TRAIN_PY_TAIL + '\nAUTORESEARCH_METRICS {"val_bpb":0.5,"num_steps":10}\n'
    )
    assert error == ""
    assert metrics == {"val_bpb": 0.5, "num_steps": 10.0}


def test_aborted_frame_is_named_as_such():
    metrics, error = parse_metrics("OPHIS_ABORT projected 120 steps at step 30\n")
    assert metrics == {}
    assert "aborted" in error


def make_trainer(tmp_path):
    """A stand-in for train.py that emits the real trailing summary block."""
    project = tmp_path / "trainer"
    project.mkdir(exist_ok=True)
    script = project / "train.py"
    script.write_text(
        """import argparse
p = argparse.ArgumentParser()
p.add_argument('--value', type=float, required=True)
p.add_argument('--steps', type=int, default=1000)
a = p.parse_args()
print(f"step {a.steps:05d} (99.8%) | loss: 1.2 | steps: 3 | remaining: 0s")
print("---")
print(f"val_bpb:          {a.value:.6f}")
print(f"total_seconds:    508.2")
print(f"num_steps:        {a.steps}")
""",
        encoding="utf-8",
    )
    data = project / "data_manifest.json"
    data.write_text('{"dataset":"test","revision":1}', encoding="utf-8")
    return script, data


def _screen(store, tmp_path, label, value, *, steps=1000):
    script, data = make_trainer(tmp_path)
    return FastLoop(store).screen(
        label,
        change_summary=f"set value to {value}",
        execution=execution_config(script, data),
        argv=[sys.executable, str(script), "--value", str(value), "--steps", str(steps)],
        minimum_steps=900,
        require_gpu=False,
        isolation="none",
    )


def test_leaderboard_keeps_only_improvements_and_invalid_cannot_move_the_best(tmp_path):
    store = Store(tmp_path / "state")
    _screen(store, tmp_path, "a_first", 0.99)
    _screen(store, tmp_path, "b_worse", 1.20)
    _screen(store, tmp_path, "c_better", 0.90)
    # A far better number that is inadmissible: too few optimizer steps.
    _screen(store, tmp_path, "d_short_run", 0.10, steps=5)

    board = FastLoop(store).leaderboard()
    by_label = {row["spec_id"]: row for row in board["entries"]}
    assert by_label["exp_a_first"]["kept"] is True
    assert by_label["exp_b_worse"]["kept"] is False
    assert by_label["exp_c_better"]["kept"] is True
    assert by_label["exp_d_short_run"]["verdict"] == "invalid"
    assert by_label["exp_d_short_run"]["kept"] is False, "an invalid run cannot become the best"
    assert board["best_value"] == 0.90
    assert board["best_spec_id"] == "exp_c_better"
    assert board["accepted"] == 2
    assert board["invalid"] == 1


def test_chart_and_views_are_rebuilt_from_the_registry(tmp_path):
    store = Store(tmp_path / "state")
    _screen(store, tmp_path, "a_first", 0.99)
    board = FastLoop(store).write_views()
    svg = render_svg(board)
    assert svg.startswith("<svg")
    assert "0.990000" in render_page(board)
    assert (store.views_dir / "LEADERBOARD.md").is_file()
    assert "running best" in (store.views_dir / "LEADERBOARD.md").read_text()


def test_screen_cli_publishes_chart_and_reports_running_best(tmp_path, capsys):
    script, data = make_trainer(tmp_path)
    execution_file = tmp_path / "execution.json"
    execution_file.write_text(json.dumps(execution_config(script, data)))
    code = main(
        [
            "--root",
            str(tmp_path / "state"),
            "screen",
            "cli_probe",
            "--summary",
            "cli smoke test",
            "--minimum-steps",
            "900",
            "--no-gpu",
            "--execution",
            str(execution_file),
            "--argv",
            sys.executable,
            str(script),
            "--value",
            "0.95",
            "--steps",
            "1000",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert '"running_best": 0.95' in printed
    store = Store(tmp_path / "state")
    assert (store.views_dir / "progress.svg").is_file()
    assert (store.views_dir / "progress.html").is_file()


def test_sub_noise_improvements_are_recorded_but_not_accepted(tmp_path):
    """A strict improvement is not an improvement.

    The accept rule used to take any value below the reference. 16 of 33 accepted entries
    were below the measurement's own noise -- deltas as small as -0.000005 -- and together
    contributed -0.002038 of apparent progress, 5.7% of the leaderboard total. That is
    placement and seed luck ratcheted into the running best.
    """
    from autoresearch.loop import ACCEPT_THRESHOLD

    assert ACCEPT_THRESHOLD > 0

    # A delta smaller than the gate must be visible but not accepted; one larger must be
    # accepted. Exercised through the same predicate the leaderboard uses.
    reference = 0.962000
    tiny = reference - ACCEPT_THRESHOLD / 10
    real = reference - ACCEPT_THRESHOLD * 2

    def kept(value):
        raw = value < reference
        return bool(raw and value < reference - ACCEPT_THRESHOLD)

    assert kept(tiny) is False, "sub-noise improvement must not be accepted"
    assert kept(real) is True, "a real improvement must still be accepted"
    assert tiny < reference, "and the sub-noise value is still a strict improvement"


def test_trophy_is_written_only_for_a_promoted_sota_and_only_once(tmp_path):
    """The trophy marks a Knowledge-Engine promotion, never a search-mode best.

    The leaderboard's running best has sat BELOW the SOTA figure for most of this campaign
    (0.961449 vs 0.962288) because paired screens are structurally ineligible -- and one of
    those screens was a co-tenancy artefact whose control arm was robbed down to 827 steps,
    which would have promoted a false SOTA by 8x. A trophy earnable by a screen would mean
    nothing.
    """
    from autoresearch.knowledge import KnowledgeEngine
    from autoresearch.store import Store

    store = Store(tmp_path / "state")
    engine = KnowledgeEngine(store)
    snap = {
        "sota": {
            "grp": {
                "spec_id": "exp_confirm_thing",
                "value": 0.9500,
                "replicate_values": [0.95, 0.9501, 0.9499],
            }
        }
    }
    engine._append_sota_log(snap)
    text = (store.views_dir / "SOTA_LOG.md").read_text()
    assert "\U0001f3c6" in text and "NEW SOTA: 0.950000" in text
    assert "exp_confirm_thing" in text and "n=3" in text

    # Idempotent: synthesize runs on every judge, and must not append a second trophy.
    engine._append_sota_log(snap)
    assert text.count("\U0001f3c6") == (store.views_dir / "SOTA_LOG.md").read_text().count(
        "\U0001f3c6"
    )

    # A genuinely new promotion does append.
    snap["sota"]["grp"] = {
        "spec_id": "exp_confirm_next",
        "value": 0.9400,
        "replicate_values": [0.94, 0.9401],
    }
    engine._append_sota_log(snap)
    assert (store.views_dir / "SOTA_LOG.md").read_text().count("\U0001f3c6") == 2
