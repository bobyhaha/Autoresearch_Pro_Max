from __future__ import annotations

import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from autoresearch.bank import BANK_MAX_USES, BankIndex
from autoresearch.campaign import CampaignQueue
from autoresearch.knowledge import KnowledgeEngine, _select_sota
from autoresearch.records import RecordError
from autoresearch.store import Store
from autoresearch.workflow import V2Workflow


def _project(tmp_path, *, training_seconds: float = 0.045):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "train.py"
    script.write_text(
        """import argparse, json, os
p = argparse.ArgumentParser()
p.add_argument('--value', type=float, required=True)
a = p.parse_args()
seed = int(os.environ['AUTORESEARCH_SEED'])
print('AUTORESEARCH_METRICS ' + json.dumps({
    'val_bpb': a.value,
    'num_steps': 1000,
    'seed': seed,
    'training_seconds': TRAINING_SECONDS,
    'total_seconds': 0.05,
}))
""".replace("TRAINING_SECONDS", repr(training_seconds)),
        encoding="utf-8",
    )
    data = project / "data.json"
    data.write_text(json.dumps({"split": "train-0-9/val-10"}), encoding="utf-8")
    return script, data


def _scope():
    return {
        "id": "test_fixed_frame_v2",
        "hardware_class": "cpu-test",
        "dataset_split": "train-0-9/val-10",
        "tokenizer": "test-byte-fallback-v2",
        "evaluator": "byte-bpb-v2",
        "precision": "fp32",
        "metric": {"name": "val_bpb", "direction": "minimize"},
        "budget": {"kind": "wall_seconds", "value": 0.05},
    }


def _execution(script, data, *, resources=("cpu_a", "cpu_b")):
    return {
        "code_bindings": [{"source": str(script), "execution_path": script.name}],
        "data_bindings": [{"source": str(data), "execution_path": data.name}],
        "resources": [
            {
                "id": resource_id,
                "hardware_class": "cpu-test",
                "backend": "local",
                "workdir": str(script.parent),
                "gpus": [],
            }
            for resource_id in resources
        ],
        "runtime": {
            "timeout_seconds_per_arm": 10,
            "telemetry_interval_seconds": 0.01,
            "resource_wait_seconds": 0,
        },
    }


def _argv(script, value):
    return [sys.executable, script.name, f"--value={value}"]


def _calibrated(tmp_path, *, training_seconds: float = 0.045, resources=("cpu_a",)):
    script, data = _project(tmp_path, training_seconds=training_seconds)
    store = Store(tmp_path / "state")
    queue = CampaignQueue(store)
    workflow = V2Workflow(store, queue=queue)
    execution = _execution(script, data, resources=resources)
    calibration = workflow.stage_calibration(
        "champion_1",
        "initial",
        execution,
        _argv(script, 1.0),
        _scope(),
        [script.name],
    )
    queue.work(workers=len(resources))
    return workflow, queue, store, script, execution, calibration


def test_cpu_calibrate_candidate_and_direct_promotion_workflow(tmp_path):
    workflow, queue, store, script, execution, calibration = _calibrated(
        tmp_path, resources=("cpu_a", "cpu_b")
    )

    assert len(calibration["staged"]) == 2
    assert {row["manifest"]["payload"]["resources"][0]["id"] for row in calibration["staged"]} == {
        "cpu_a",
        "cpu_b",
    }
    assert len(BankIndex.from_store(store).controls) == 2

    first = workflow.stage_candidate(
        "faster_update",
        execution,
        _argv(script, 0.9),
        _scope(),
        [script.name],
        "champion_1",
        subsystem="optimizer",
    )
    second = workflow.stage_candidate(
        "faster_update_2",
        execution,
        _argv(script, 0.8),
        _scope(),
        [script.name],
        "champion_1",
        subsystem="optimizer",
    )
    assert first["reference_control"]["resource_id"] == "cpu_a"
    assert second["reference_control"]["resource_id"] == "cpu_b"
    assert len(first["spec"]["payload"]["search"]["reference_controls"]) == 1
    assert first["spec"]["payload"]["analysis"]["sota_eligible"] is False
    assert first["manifest"]["payload"]["resources"][0]["id"] == "cpu_a"

    with pytest.raises(RecordError, match="landed candidate"):
        workflow.promotion_proposal(first["spec"]["id"])

    queue.work(workers=2)
    score = BankIndex.from_store(store).score_candidate(
        next(
            row["id"]
            for row in store.list("result_bundle")
            if row["payload"]["spec_id"] == first["spec"]["id"]
        )
    )
    assert score["status"] == "scored"
    assert score["promotion_due"] is True

    with pytest.raises(RecordError, match="held out"):
        workflow.promotion_proposal(
            first["spec"]["id"], seeds=(42, 43, 44), minimum_valid_replicates=3
        )

    proposal = workflow.promotion_proposal(first["spec"]["id"])
    assert [row["seed"] for row in proposal["plan"]] == [43, 44, 45, 46, 47]
    assert [arm["name"] for arm in proposal["plan"][0]["arms"]] == [
        "control",
        "candidate",
    ]
    assert [arm["name"] for arm in proposal["plan"][1]["arms"]] == [
        "candidate",
        "control",
    ]
    assert all(
        arm["env"]["AUTORESEARCH_SEED"] == str(replicate["seed"])
        for replicate in proposal["plan"]
        for arm in replicate["arms"]
    )
    assert proposal["analysis"]["effect"] == "difference"
    assert proposal["analysis"]["sota_eligible"] is True
    assert proposal["search"]["source_result_id"] == score["result_id"]

    staged = workflow.stage_promotion(first["spec"]["id"])
    assert staged["manifest"]["payload"]["reviews"] == []
    assert staged["job"]["state"] == "pending"
    assert staged["reviews_supplied"] is False


def test_same_argv_code_edit_promotes_with_isolated_control_and_candidate_snapshots(tmp_path):
    workflow, queue, store, script, execution, _ = _calibrated(tmp_path)
    original = script.read_text(encoding="utf-8")
    script.write_text(
        original.replace("'val_bpb': a.value", "'val_bpb': a.value - 0.1"),
        encoding="utf-8",
    )

    candidate = workflow.stage_candidate(
        "in_place_train_edit",
        execution,
        _argv(script, 1.0),
        _scope(),
        [script.name],
        "champion_1",
        subsystem="optimizer",
    )
    assert candidate["reference_control"]["mutable_code_changes"] == [script.name]
    queue.work()

    promotion = workflow.stage_promotion(candidate["spec"]["id"])
    code_paths = {
        row["execution_path"] for row in promotion["manifest"]["payload"]["code_bindings"]
    }
    roots = promotion["spec"]["payload"]["search"]["arm_code_roots"]
    assert f"{roots['control']}/train.py" in code_paths
    assert f"{roots['candidate']}/train.py" in code_paths
    first_arms = promotion["manifest"]["payload"]["plan"][0]["arms"]
    assert first_arms[0]["argv"][1] == f"{roots['control']}/train.py"
    assert first_arms[1]["argv"][1] == f"{roots['candidate']}/train.py"

    queue.work()
    snapshot = KnowledgeEngine(store).synthesize()
    sota = snapshot["sota"][_scope()["id"]]
    assert sota["value"] == pytest.approx(0.9)
    assert sota["replicate_seeds"] == [43, 44, 45, 46, 47]

    # Knowledge recomputes lineage; it does not trust a promotion label or its
    # claimed pilot delta when selecting SOTA.
    specs = {row["id"]: row for row in store.list("experiment_spec")}
    manifests = {row["id"]: row for row in store.list("execution_manifest")}
    results = {row["id"]: row for row in store.list("result_bundle")}
    decisions = {row["payload"]["result_id"]: row for row in store.list("evidence_decision")}
    promotion_spec = promotion["spec"]
    tampered = copy.deepcopy(promotion_spec)
    tampered["payload"]["search"]["source_delta"] += 0.01
    specs[promotion_spec["id"]] = tampered
    selected, blockers = _select_sota(snapshot["beliefs"], specs, manifests, results, decisions)
    assert selected == {}
    assert "source_delta" in blockers[_scope()["id"]]

    colliding = copy.deepcopy(promotion_spec)
    colliding["payload"]["search"]["arm_code_roots"] = {
        "control": "snap",
        "candidate": "snap/sub",
    }
    specs[promotion_spec["id"]] = colliding
    selected, blockers = _select_sota(snapshot["beliefs"], specs, manifests, results, decisions)
    assert selected == {}
    assert "canonical disjoint namespaces" in blockers[_scope()["id"]]


def test_candidate_rejections_and_overhead_override(tmp_path):
    workflow, _, _, script, execution, _ = _calibrated(tmp_path, training_seconds=0.005)

    with pytest.raises(RecordError, match="repeats long option"):
        workflow.stage_candidate(
            "duplicate",
            execution,
            [sys.executable, str(script), "--value=0.9", "--value", "0.8"],
            _scope(),
            [script.name],
            "champion_1",
        )
    with pytest.raises(RecordError, match="absolute binding source"):
        workflow.stage_candidate(
            "absolute_source_bypass",
            execution,
            [sys.executable, str(script), "--value=0.9"],
            _scope(),
            [script.name],
            "champion_1",
        )
    with pytest.raises(RecordError, match="absolute payload paths"):
        workflow.stage_candidate(
            "unsealed_absolute_program",
            execution,
            [sys.executable, "/tmp/unsealed.py", script.name, "--value=0.9"],
            _scope(),
            [script.name],
            "champion_1",
        )
    with pytest.raises(RecordError, match="identical"):
        workflow.stage_candidate(
            "identical",
            execution,
            _argv(script, 1.0),
            _scope(),
            [script.name],
            "champion_1",
            subsystem="optimizer",
        )
    with pytest.raises(RecordError, match="overhead-dominated"):
        workflow.stage_candidate(
            "model_change",
            execution,
            _argv(script, 0.9),
            _scope(),
            [script.name],
            "champion_1",
            subsystem="architecture",
        )
    accepted = workflow.stage_candidate(
        "model_change_override",
        execution,
        _argv(script, 0.9),
        _scope(),
        [script.name],
        "champion_1",
        subsystem="architecture",
        allow_overhead_dominated=True,
    )
    assert accepted["job"]["state"] == "pending"


def test_calibration_timing_relationship_and_sealed_frame_are_evidence_gates(tmp_path):
    _, _, inconsistent_store, _, _, _ = _calibrated(tmp_path, training_seconds=0.055)
    inconsistent = inconsistent_store.list("evidence_decision")[0]["payload"]
    assert inconsistent["measurement_verdict"] == "invalid"
    assert any("inconsistent timing" in reason for reason in inconsistent["reasons"])

    off_frame_root = tmp_path / "off_frame"
    off_frame_root.mkdir()
    script, data = _project(off_frame_root, training_seconds=0.02)
    store = Store(tmp_path / "off_frame_state")
    queue = CampaignQueue(store)
    workflow = V2Workflow(store, queue=queue)
    execution = _execution(script, data, resources=("cpu_off_frame",))
    scope = _scope()
    scope["budget"]["value"] = 5
    workflow.stage_calibration(
        "off_frame_bank",
        "initial",
        execution,
        _argv(script, 1.0),
        scope,
        [script.name],
    )
    queue.work()
    off_frame = store.list("evidence_decision")[0]["payload"]
    assert off_frame["measurement_verdict"] == "invalid"
    assert any("outside the sealed 5s frame" in reason for reason in off_frame["reasons"])


def test_stale_missing_and_losing_banks_cannot_promote(tmp_path):
    workflow, queue, store, script, execution, _ = _calibrated(tmp_path)
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    stale_workflow = V2Workflow(store, queue=queue, now=lambda: future)
    with pytest.raises(RecordError, match="no fresh exact bank"):
        stale_workflow.stage_candidate(
            "stale",
            execution,
            _argv(script, 0.9),
            _scope(),
            [script.name],
            "champion_1",
            subsystem="optimizer",
        )

    with pytest.raises(RecordError, match="no fresh exact bank"):
        workflow.stage_candidate(
            "missing",
            execution,
            _argv(script, 0.9),
            _scope(),
            [script.name],
            "unknown_bank",
            subsystem="optimizer",
        )

    losing = workflow.stage_candidate(
        "losing",
        execution,
        _argv(script, 1.1),
        _scope(),
        [script.name],
        "champion_1",
        subsystem="optimizer",
    )
    queue.work()
    with pytest.raises(RecordError, match="did not clear"):
        workflow.promotion_proposal(losing["spec"]["id"])


def test_failed_seal_does_not_leak_a_bank_reservation(tmp_path):
    workflow, _, _, script, execution, _ = _calibrated(tmp_path)
    broken = copy.deepcopy(execution)
    broken["resources"][0]["workdir"] = str(tmp_path / "missing_workdir")
    with pytest.raises(RecordError, match="workdir does not exist"):
        workflow.stage_candidate(
            "orphan_after_spec",
            broken,
            _argv(script, 0.9),
            _scope(),
            [script.name],
            "champion_1",
            subsystem="optimizer",
        )

    accepted = workflow.stage_candidate(
        "after_failed_seal",
        execution,
        _argv(script, 0.9),
        _scope(),
        [script.name],
        "champion_1",
        subsystem="optimizer",
    )
    assert accepted["reference_control"]["pending_reservations"] == 0


def test_concurrent_search_reservations_cannot_exceed_bank_use_cap(tmp_path):
    workflow, _, _, script, execution, _ = _calibrated(tmp_path)

    def stage(index):
        try:
            return workflow.stage_candidate(
                f"parallel_{index}",
                execution,
                _argv(script, 0.9 - index * 0.001),
                _scope(),
                [script.name],
                "champion_1",
                subsystem="optimizer",
                # This test is about the reservation cap, not the exploration
                # budget, so every arm opens a distinct mechanism family and the
                # budget's repair-loop rule stays out of the way.
                family=f"family_{index}",
            )
        except RecordError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=10) as pool:
        outcomes = list(pool.map(stage, range(10)))

    accepted = [row for row in outcomes if isinstance(row, dict)]
    rejected = [row for row in outcomes if isinstance(row, RecordError)]
    assert len(accepted) == BANK_MAX_USES
    assert len(rejected) == 10 - BANK_MAX_USES
    assert all("exhausted" in str(error) for error in rejected)
