from __future__ import annotations

import copy
import os
import site
import subprocess
import sys

import pytest
from conftest import approvals, execution_config, make_project, make_spec

import autoresearch.execution as execution_module
from autoresearch.evidence import EvidenceEngine, assess, throughput_key, throughput_reason
from autoresearch.execution import NoResourceAvailable, ResourceScheduler
from autoresearch.knowledge import KnowledgeEngine, _belief, _select_sota
from autoresearch.protocol import normalize_scope, validate_code_entry_argv
from autoresearch.records import RecordError
from autoresearch.research import ResearchEngine
from autoresearch.sealing import SealingAuthority
from autoresearch.store import Store


def _protocol_v2_spec(script, *, stage="confirmation", replicates=3, sota_eligible=True):
    spec = make_spec(
        script,
        stage=stage,
        replicates=replicates,
        sota_eligible=sota_eligible,
    )
    spec["protocol_version"] = 2
    spec["scope"] = {
        "benchmark": "test_fixed_budget_v1",
        "budget": {"kind": "wall_seconds", "value": 0.05},
    }
    spec["search"] = {
        "lane": "promotion" if stage == "confirmation" and sota_eligible else "explore",
        "baseline": {"name": "frozen_control"},
        "context_fingerprint": "a" * 64,
        "mutable_code_paths": [script.name],
        "future_extension": {"is_allowed": True},
    }
    spec["requirements"]["required_metrics"].extend(["seed", "training_seconds", "total_seconds"])
    for index, replicate in enumerate(spec["plan"]):
        seed = 41 + index
        replicate["seed"] = seed
        for arm in replicate["arms"]:
            arm["argv"] = [script.name if token == str(script) else token for token in arm["argv"]]
            arm["env"]["AUTORESEARCH_SEED"] = str(seed)
    return spec


def test_resource_lease_is_visible_exclusive_and_cleaned_up(tmp_path):
    store = Store(tmp_path / "state")
    name = "worker_gpu0"
    with store.resource_lease(name, {"resource_id": "worker", "gpu": 0}) as lease:
        visible = store.resource_leases()
        assert visible == [lease]
        assert visible[0]["owner_pid"] == os.getpid()
        assert visible[0]["lease_id"]
        with pytest.raises(BlockingIOError), store.resource_lease(name, blocking=False):
            pass
    assert store.resource_leases() == []

    with pytest.raises(RuntimeError, match="inside lease"), store.resource_lease(name):
        raise RuntimeError("inside lease")
    assert store.resource_leases() == []


def test_scheduler_records_physical_gpu_and_enforces_host_slots(tmp_path, monkeypatch):
    store = Store(tmp_path / "state")

    def idle(_resource, gpu):
        return {
            "state": "available",
            "gpu": gpu,
            "uuid": f"GPU-{gpu}",
            "memory_used_mb": 0,
            "utilization_percent": 0,
            "process_count": 0,
        }

    monkeypatch.setattr(execution_module, "probe", idle)
    resource = {
        "id": "worker",
        "host_id": "host-a",
        "backend": "local",
        "workdir": str(tmp_path),
        "gpus": [0, 1],
        "max_concurrent_jobs": 1,
        "reservation": {"mode": "externally_reserved", "id": "job-123"},
    }
    scheduler = ResourceScheduler(store, [resource])
    with scheduler.allocate(require_gpu=True, wait_seconds=0) as first:
        assert first.public["host_id"] == "host-a"
        assert first.public["gpu"] == 0
        assert first.public["gpu_uuid"] == "GPU-0"
        assert first.public["reservation"]["id"] == "job-123"
        assert store.resource_leases()[0]["host_slot"] == 0
        with (
            pytest.raises(NoResourceAvailable),
            ResourceScheduler(store, [resource]).allocate(require_gpu=True, wait_seconds=0),
        ):
            pass

    two_wide = {**resource, "max_concurrent_jobs": 2}
    with (
        ResourceScheduler(store, [two_wide]).allocate(require_gpu=True, wait_seconds=0),
        pytest.raises(NoResourceAvailable),
        ResourceScheduler(store, [two_wide]).allocate(require_gpu=True, wait_seconds=0),
    ):
        pass

    second_workdir = tmp_path / "gpu1"
    second_workdir.mkdir()
    gpu1 = {
        **two_wide,
        "id": "worker-gpu1",
        "workdir": str(second_workdir),
        "gpus": [1],
    }
    with (
        ResourceScheduler(store, [two_wide]).allocate(require_gpu=True, wait_seconds=0) as first,
        ResourceScheduler(store, [gpu1]).allocate(require_gpu=True, wait_seconds=0) as second,
    ):
        assert {first.gpu, second.gpu} == {0, 1}
        assert len(store.resource_leases()) == 2


def test_resource_contract_is_sealed_and_external_reservation_needs_id(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    proposal = _protocol_v2_spec(script)
    spec = ResearchEngine(store).create(proposal)
    execution = execution_config(script, data)
    execution["resources"][0].update(
        {
            "host_id": "test-host",
            "max_concurrent_jobs": 1,
            "reservation": {"mode": "externally_reserved", "id": "allocation-7"},
        }
    )
    manifest = SealingAuthority(store).seal(spec["id"], execution, approvals(spec))
    assert manifest["payload"]["protocol_version"] == 2
    assert manifest["payload"]["search"] == proposal["search"]
    assert manifest["payload"]["scope"] == proposal["scope"]
    assert manifest["payload"]["resources"][0]["host_id"] == "test-host"
    assert manifest["payload"]["resources"][0]["max_concurrent_jobs"] == 1
    assert manifest["payload"]["resources"][0]["reservation"]["id"] == "allocation-7"

    bad = execution_config(script, data)
    bad["resources"][0]["reservation"] = {"mode": "externally_reserved"}
    with pytest.raises(RecordError, match="reservation id"):
        SealingAuthority(store).seal(spec["id"], bad, approvals(spec))


def test_protocol_v2_spec_has_exactly_one_immutable_manifest(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state_single_manifest")
    proposal = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    spec = ResearchEngine(store).create(proposal)
    authority = SealingAuthority(store)
    execution = execution_config(script, data)
    first = authority.seal(spec["id"], execution)
    assert authority.seal(spec["id"], execution) == first

    changed_execution = copy.deepcopy(execution)
    changed_execution["resources"][0]["id"] = "different_resource_contract"
    with pytest.raises(RecordError, match="exactly one immutable manifest"):
        authority.seal(spec["id"], changed_execution)


def test_protocol_v2_seed_and_mutable_path_contracts_are_hard_gates(tmp_path):
    script, data = make_project(tmp_path)

    missing_seed = _protocol_v2_spec(script)
    del missing_seed["plan"][0]["seed"]
    with pytest.raises(RecordError, match=r"plan\[0\].seed"):
        ResearchEngine(Store(tmp_path / "state_missing_seed")).create(missing_seed)

    duplicate_seed = _protocol_v2_spec(script)
    duplicate_seed["plan"][1]["seed"] = duplicate_seed["plan"][0]["seed"]
    for arm in duplicate_seed["plan"][1]["arms"]:
        arm["env"]["AUTORESEARCH_SEED"] = str(duplicate_seed["plan"][1]["seed"])
    with pytest.raises(RecordError, match="seeds must be distinct"):
        ResearchEngine(Store(tmp_path / "state_duplicate_seed")).create(duplicate_seed)

    store = Store(tmp_path / "state_mutable")
    proposal = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    proposal["search"]["mutable_code_paths"] = ["not_sealed.py"]
    spec = ResearchEngine(store).create(proposal)
    with pytest.raises(RecordError, match="mutable_code_paths"):
        SealingAuthority(store).seal(spec["id"], execution_config(script, data))

    absolute = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    for arm in absolute["plan"][0]["arms"]:
        arm["argv"] = [str(script) if token == script.name else token for token in arm["argv"]]
    absolute_store = Store(tmp_path / "state_absolute")
    absolute_spec = ResearchEngine(absolute_store).create(absolute)
    with pytest.raises(RecordError, match="absolute payload paths"):
        SealingAuthority(absolute_store).seal(absolute_spec["id"], execution_config(script, data))

    indirect_commands = {
        "python_option_argument": [sys.executable, "-W", script.name, "unsealed.py"],
        "shell_command": ["/bin/bash", "-c", "python /tmp/unsealed.py", script.name],
    }
    for label, argv in indirect_commands.items():
        indirect = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
        for arm in indirect["plan"][0]["arms"]:
            arm["argv"] = list(argv)
        indirect_store = Store(tmp_path / f"state_{label}")
        indirect_spec = ResearchEngine(indirect_store).create(indirect)
        with pytest.raises(RecordError, match="direct sealed-entry grammar"):
            SealingAuthority(indirect_store).seal(
                indirect_spec["id"], execution_config(script, data)
            )

    untrusted_launcher = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    for arm in untrusted_launcher["plan"][0]["arms"]:
        arm["argv"] = ["/tmp/attacker/python3", script.name]
    untrusted_store = Store(tmp_path / "state_untrusted_launcher")
    untrusted_spec = ResearchEngine(untrusted_store).create(untrusted_launcher)
    with pytest.raises(RecordError, match="not explicitly trusted"):
        SealingAuthority(untrusted_store).seal(untrusted_spec["id"], execution_config(script, data))

    for variable in ("PATH", "PYTHONUSERBASE"):
        path_override = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
        path_override["plan"][0]["arms"][0]["env"][variable] = "/tmp/attacker"
        with pytest.raises(RecordError, match="launcher/module loading variables"):
            ResearchEngine(Store(tmp_path / f"state_{variable.lower()}_override")).create(
                path_override
            )

    with pytest.raises(RecordError, match="PATH-resolved"):
        validate_code_entry_argv(["python", "-c", "print(1)"], ["python"])
    with pytest.raises(RecordError, match="at least one sealed code binding"):
        validate_code_entry_argv([sys.executable, "-c", "print(1)"], ["-c"])
    with pytest.raises(RecordError, match="at least one sealed code binding"):
        validate_code_entry_argv(["/bin/bash", "+O", "extglob", "-c", "printf BYPASS"], ["+O"])

    shadowed_scope = _protocol_v2_spec(script)
    shadowed_scope["search"]["scope"] = {"benchmark": "different"}
    with pytest.raises(RecordError, match="only authority"):
        ResearchEngine(Store(tmp_path / "state_shadowed_scope")).create(shadowed_scope)


def test_protocol_v2_requires_program_timing_metrics(tmp_path):
    script, _ = make_project(tmp_path)
    proposal = _protocol_v2_spec(script)
    proposal["requirements"]["required_metrics"].remove("total_seconds")

    with pytest.raises(RecordError, match="timing metrics"):
        ResearchEngine(Store(tmp_path / "state_missing_timing")).create(proposal)


def test_training_seconds_budget_normalizes_and_seals_explicit_time_budget(tmp_path):
    normalized = normalize_scope(
        {
            "id": "karpathy_training_clock_test",
            "hardware_class": "cpu-test",
            "dataset_split": "test",
            "tokenizer": "test",
            "evaluator": "test",
            "precision": "test",
            "metric": {"name": "val_bpb", "direction": "minimize"},
            "budget": {"kind": "training_seconds", "value": 300},
        }
    )
    assert normalized["budget"] == {"kind": "training_seconds", "value": 300.0}

    script, data = make_project(tmp_path)
    proposal = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    proposal["scope"]["budget"] = {"kind": "training_seconds", "value": 0.05}
    proposal["plan"][0]["arms"][0]["argv"].append("--time-budget=0.05")
    store = Store(tmp_path / "state_training_budget_seal")
    spec = ResearchEngine(store).create(proposal)
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    assert manifest["payload"]["scope"]["budget"]["kind"] == "training_seconds"

    conflicting = copy.deepcopy(proposal)
    conflicting["plan"][0]["arms"][0]["argv"][-1] = "--time-budget=0.06"
    conflicting_store = Store(tmp_path / "state_training_budget_conflict")
    conflicting_spec = ResearchEngine(conflicting_store).create(conflicting)
    with pytest.raises(RecordError, match="training_seconds"):
        SealingAuthority(conflicting_store).seal(
            conflicting_spec["id"], execution_config(script, data)
        )


def test_training_seconds_evidence_uses_inner_clock_and_reconciles_runner(tmp_path):
    script, data = make_project(tmp_path)
    script.write_text(
        """import json, os, time
seed = int(os.environ['AUTORESEARCH_SEED'])
started = time.monotonic()
time.sleep(0.06)
total = time.monotonic() - started
print('AUTORESEARCH_METRICS '+json.dumps({
    'score':0.9,
    'num_steps':1000,
    'seed':seed,
    'training_seconds':0.05,
    'total_seconds':total,
}))
""",
        encoding="utf-8",
    )
    store = Store(tmp_path / "state_training_frame")
    proposal = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    proposal["scope"]["budget"] = {"kind": "training_seconds", "value": 0.05}
    spec = ResearchEngine(store).create(proposal)
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    result = execution_module.ExecutionService(store).execute_next(manifest["id"])
    decision = EvidenceEngine(store).judge(result["id"])
    assert decision["payload"]["measurement_verdict"] == "valid", decision["payload"]["reasons"]

    result_payload = copy.deepcopy(result["payload"])
    manifest_payload = copy.deepcopy(manifest["payload"])
    spec_payload = copy.deepcopy(spec["payload"])
    for payload in (manifest_payload, spec_payload):
        payload["scope"]["budget"] = {"kind": "training_seconds", "value": 300}
    for arm in result_payload["arms"]:
        arm["metrics"]["training_seconds"] = 300.1
        arm["metrics"]["total_seconds"] = 325.9
        arm["wall_seconds"] = 326.0

    verdict, reasons, _ = assess(result_payload, manifest_payload, spec_payload)
    assert verdict == "valid", reasons

    off_training = copy.deepcopy(result_payload)
    off_training["arms"][0]["metrics"]["training_seconds"] = 270
    verdict, reasons, _ = assess(off_training, manifest_payload, spec_payload)
    assert verdict == "invalid"
    assert any("training_seconds" in reason and "training frame" in reason for reason in reasons)

    short_runner = copy.deepcopy(result_payload)
    short_runner["arms"][0]["metrics"]["training_seconds"] = 300
    short_runner["arms"][0]["metrics"]["total_seconds"] = 300
    short_runner["arms"][0]["wall_seconds"] = 299.9
    verdict, reasons, _ = assess(short_runner, manifest_payload, spec_payload)
    assert verdict == "invalid"
    assert any("shorter than" in reason for reason in reasons)

    impossible_training = copy.deepcopy(result_payload)
    impossible_training["arms"][0]["metrics"]["training_seconds"] = 315
    impossible_training["arms"][0]["metrics"]["total_seconds"] = 315
    impossible_training["arms"][0]["wall_seconds"] = 300
    verdict, reasons, _ = assess(impossible_training, manifest_payload, spec_payload)
    assert verdict == "invalid"
    assert any("longer than runner" in reason for reason in reasons)

    mismatched_total = copy.deepcopy(result_payload)
    mismatched_total["arms"][0]["metrics"]["total_seconds"] = 350
    verdict, reasons, _ = assess(mismatched_total, manifest_payload, spec_payload)
    assert verdict == "invalid"
    assert any("does not match runner" in reason for reason in reasons)

    # The pre-existing wall-clock mode remains strict about end-to-end duration.
    wall_spec = copy.deepcopy(spec_payload)
    wall_manifest = copy.deepcopy(manifest_payload)
    for payload in (wall_manifest, wall_spec):
        payload["scope"]["budget"] = {"kind": "wall_seconds", "value": 300}
    verdict, reasons, _ = assess(result_payload, wall_manifest, wall_spec)
    assert verdict == "invalid"
    assert any("runner wall_seconds" in reason for reason in reasons)


def test_evidence_v2_verifies_seed_and_gpu_identity(tmp_path):
    script, data = make_project(tmp_path)
    script.write_text(
        """import json, os
seed = int(os.environ['AUTORESEARCH_SEED'])
print('AUTORESEARCH_METRICS '+json.dumps({
    'score':0.9,
    'num_steps':1000,
    'seed':seed,
    'training_seconds':0.04,
    'total_seconds':0.05,
}))
""",
        encoding="utf-8",
    )
    store = Store(tmp_path / "state")
    proposal = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    spec = ResearchEngine(store).create(proposal)
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    result = execution_module.ExecutionService(store).execute_next(manifest["id"])
    decision = EvidenceEngine(store).judge(result["id"])
    assert decision["payload"]["policy_version"] == "evidence-v3"
    assert decision["payload"]["measurement_verdict"] == "valid"
    assert decision["payload"]["verified_seed"] == 41
    assert decision["payload"]["gpu_key"] is None

    result_payload = copy.deepcopy(result["payload"])
    manifest_payload = copy.deepcopy(manifest["payload"])
    spec_payload = copy.deepcopy(spec["payload"])
    spec_payload["requirements"]["require_gpu"] = True
    spec_payload["requirements"]["isolation"] = "launch"
    manifest_payload["requirements"] = copy.deepcopy(spec_payload["requirements"])
    manifest_payload["resources"] = [
        {
            "id": "gpu-worker",
            "host_id": "host-a",
            "backend": "local",
            "workdir": str(script.parent),
            "gpus": [2],
        }
    ]
    result_payload["resource"] = {
        "id": "gpu-worker",
        "host_id": "host-a",
        "backend": "local",
        "workdir": str(script.parent),
        "gpu": 2,
        "gpu_uuid": "GPU-good",
    }
    result_payload["launch_telemetry"] = {
        "state": "available",
        "gpu": 2,
        "uuid": "GPU-good",
        "process_count": 0,
    }
    verdict, reasons, measurements = assess(result_payload, manifest_payload, spec_payload)
    assert verdict == "valid", reasons
    assert measurements["verified_seed"] == 41
    assert measurements["gpu_key"] == "host-a::GPU-good"

    result_payload["launch_telemetry"]["gpu"] = 3
    result_payload["launch_telemetry"]["uuid"] = "GPU-other"
    verdict, reasons, measurements = assess(result_payload, manifest_payload, spec_payload)
    assert verdict == "invalid"
    assert measurements["gpu_key"] is None
    assert any("GPU index" in reason for reason in reasons)
    assert any("GPU UUID" in reason for reason in reasons)


def test_evidence_v2_checks_runner_wall_clock_independently(tmp_path, monkeypatch):
    script, data = make_project(tmp_path)
    if site.ENABLE_USER_SITE is not True:
        pytest.skip("Python user-site startup is disabled for this interpreter")
    user_base = tmp_path / "unsealed_user_base"
    user_site = (
        user_base
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    user_site.mkdir(parents=True)
    (user_site / "usercustomize.py").write_text(
        "import os\nos.environ['OPHIS_USERCUSTOMIZE_RAN'] = '1'\n",
        encoding="utf-8",
    )
    probe_environment = os.environ.copy()
    probe_environment["PYTHONUSERBASE"] = str(user_base)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('OPHIS_USERCUSTOMIZE_RAN', '0'))",
        ],
        env=probe_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "1"
    script.write_text(
        """import json, os
seed = int(os.environ['AUTORESEARCH_SEED'])
print('AUTORESEARCH_METRICS '+json.dumps({
    'score':0.9,
    'num_steps':1000,
    'seed':seed,
    'training_seconds':0.04,
    'total_seconds':0.05,
    'ambient_loader_seen':int('PYTHONPATH' in os.environ),
    'usercustomize_seen':int(os.environ.get('OPHIS_USERCUSTOMIZE_RAN', '0')),
}))
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", "/tmp/unsealed-python-path")
    monkeypatch.setenv("PYTHONUSERBASE", str(user_base))
    store = Store(tmp_path / "state_wall_frame")
    proposal = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    spec = ResearchEngine(store).create(proposal)
    execution = execution_config(script, data)
    execution["runtime"]["telemetry_interval_seconds"] = 1
    manifest = SealingAuthority(store).seal(spec["id"], execution)
    result = execution_module.ExecutionService(store).execute_next(manifest["id"])
    assert result["payload"]["arms"][0]["wall_seconds"] < 0.5
    assert result["payload"]["arms"][0]["metrics"]["ambient_loader_seen"] == 0
    assert result["payload"]["arms"][0]["metrics"]["usercustomize_seen"] == 0

    honest = EvidenceEngine(store).judge(result["id"])
    assert honest["payload"]["measurement_verdict"] == "valid"

    off_frame = copy.deepcopy(result["payload"])
    off_frame["arms"][0]["wall_seconds"] = 1.0
    verdict, reasons, _ = assess(off_frame, manifest["payload"], spec["payload"])
    assert verdict == "invalid"
    assert any("runner wall_seconds" in reason for reason in reasons)

    del off_frame["arms"][0]["wall_seconds"]
    verdict, reasons, _ = assess(off_frame, manifest["payload"], spec["payload"])
    assert verdict == "invalid"
    assert any("runner wall clock" in reason for reason in reasons)


def test_throughput_history_is_gpu_specific_without_pooled_fallback():
    argv = ["python", "train.py", "--arm=control"]
    gpu0 = {"host_id": "host-a", "gpu": 0, "gpu_uuid": "GPU-0"}
    gpu1 = {"host_id": "host-a", "gpu": 1, "gpu_uuid": "GPU-1"}
    key0 = throughput_key(argv, gpu0)
    key1 = throughput_key(argv, gpu1)
    assert key0 != key1
    baselines = {key0: 1000.0, key1: 800.0}

    assert throughput_reason("control", 850, argv, baselines, gpu0) is not None
    assert throughput_reason("control", 850, argv, baselines, gpu1) is None
    # Even if a legacy pooled key exists, an unidentified placement never uses it.
    pooled = {" ".join(argv): 1000.0}
    assert throughput_reason("control", 500, argv, pooled, {}) is None


def test_throughput_history_uses_only_valid_completed_evidence(tmp_path, monkeypatch):
    argv = ["python", "train.py"]
    results = []
    decisions = []
    for index, (steps, verdict) in enumerate([(1000, "valid")] * 5 + [(100, "invalid")] * 5):
        result_id = f"result_{index}"
        results.append(
            {
                "id": result_id,
                "payload": {
                    "status": "completed",
                    "resource": {
                        "id": "worker",
                        "host_id": "host-a",
                        "gpu": 0,
                        "gpu_uuid": "GPU-0",
                    },
                    "launch_telemetry": {"uuid": "GPU-0"},
                    "arms": [
                        {
                            "status": "completed",
                            "return_code": 0,
                            "payload_argv": argv,
                            "metrics": {"num_steps": steps},
                            "telemetry": {"max_compute_processes": 1},
                        }
                    ],
                },
            }
        )
        decisions.append(
            {
                "id": f"evidence_{index}",
                "created_at": f"2026-08-15T00:00:{index:02d}Z",
                "payload": {
                    "result_id": result_id,
                    "measurement_verdict": verdict,
                },
            }
        )

    store = Store(tmp_path / "state")

    def rows(kind):
        return decisions if kind == "evidence_decision" else results

    monkeypatch.setattr(store, "list", rows)
    baselines = EvidenceEngine(store)._step_baselines()
    key = throughput_key(argv, results[0]["payload"])
    assert baselines[key] == 1000


def test_protocol_v2_confirmation_counts_and_publishes_unique_verified_seeds(tmp_path):
    script, data = make_project(tmp_path)
    script.write_text(
        """from __future__ import annotations
import argparse, json, os
p=argparse.ArgumentParser()
p.add_argument('--value', type=float, required=True)
p.add_argument('--steps', type=int, default=1000)
a=p.parse_args()
seed=int(os.environ['AUTORESEARCH_SEED'])
print('AUTORESEARCH_METRICS '+json.dumps({
    'score':a.value,
    'num_steps':a.steps,
    'seed':seed,
    'training_seconds':0.04,
    'total_seconds':0.05,
}))
""",
        encoding="utf-8",
    )
    store = Store(tmp_path / "state")
    proposal = _protocol_v2_spec(script)
    spec = ResearchEngine(store).create(proposal)

    # Protocol v2 confirmations deliberately do not block execution on a five-person
    # council. Scientific promotion is guarded by sealed scope and independent seeds.
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    assert manifest["payload"]["reviews"] == []
    results, unfinished = execution_module.ExecutionService(store).execute_all(
        manifest["id"], workers=2
    )
    assert unfinished == []
    decisions = EvidenceEngine(store).judge_all()
    assert len(results) == len(decisions) == 3

    snapshot = KnowledgeEngine(store).synthesize()
    belief = snapshot["beliefs"][0]
    assert belief["status"] == "supported"
    assert belief["valid_replicates"] == 3
    assert belief["replicate_seeds"] == [41, 42, 43]
    # A hand-authored object cannot become SOTA merely by labelling itself a
    # promotion.  Only V2Workflow's immutable candidate/control lineage can cross
    # that boundary; the real positive path is covered by test_workflow.py.
    assert snapshot["sota"] == {}
    assert "source links" in snapshot["sota_blockers"]["test_fixed_budget_v1"]
    assert store.validate()["valid"] is True

    # minimum_valid_replicates tolerates a terminal invalid seed; it must not let
    # Knowledge publish while any preregistered seed has not landed and been judged.
    incomplete_results = {row["id"]: row for row in results[:-1]}
    incomplete_decisions = {
        row["payload"]["result_id"]: row
        for row in decisions
        if row["payload"]["result_id"] in incomplete_results
    }
    incomplete, _ = _select_sota(
        snapshot["beliefs"],
        {spec["id"]: spec},
        {manifest["id"]: manifest},
        incomplete_results,
        incomplete_decisions,
    )
    assert incomplete == {}

    # Selection independently defends the lane boundary even if handed an object that
    # did not come through record validation.
    nonpromotion = copy.deepcopy(spec)
    nonpromotion["payload"]["search"]["lane"] = "candidate"
    selected, _ = _select_sota(
        snapshot["beliefs"],
        {spec["id"]: nonpromotion},
        {manifest["id"]: manifest},
        {row["id"]: row for row in results},
        {row["payload"]["result_id"]: row for row in decisions},
    )
    assert selected == {}

    # Reusing one scope id cannot turn two hand-authored promotions into a SOTA,
    # even when their self-declared context strings differ.
    other_spec = copy.deepcopy(spec)
    other_spec["id"] = "exp_other_context"
    other_spec["payload"]["search"]["context_fingerprint"] = "b" * 64
    other_manifest = copy.deepcopy(manifest)
    other_manifest["id"] = "manifest_other_context"
    other_manifest["payload"]["spec_id"] = other_spec["id"]
    other_results = []
    other_decisions = []
    for index, (result, decision) in enumerate(zip(results, decisions, strict=True)):
        cloned_result = copy.deepcopy(result)
        cloned_result["id"] = f"result_other_{index}"
        cloned_result["payload"]["spec_id"] = other_spec["id"]
        cloned_result["payload"]["manifest_id"] = other_manifest["id"]
        other_results.append(cloned_result)
        cloned_decision = copy.deepcopy(decision)
        cloned_decision["id"] = f"evidence_other_{index}"
        cloned_decision["payload"]["result_id"] = cloned_result["id"]
        cloned_decision["payload"]["spec_id"] = other_spec["id"]
        other_decisions.append(cloned_decision)
    other_belief = copy.deepcopy(snapshot["beliefs"][0])
    other_belief["spec_id"] = other_spec["id"]
    mixed, blockers = _select_sota(
        [snapshot["beliefs"][0], other_belief],
        {spec["id"]: spec, other_spec["id"]: other_spec},
        {manifest["id"]: manifest, other_manifest["id"]: other_manifest},
        {row["id"]: row for row in [*results, *other_results]},
        {row["payload"]["result_id"]: row for row in [*decisions, *other_decisions]},
    )
    assert mixed == {}
    assert "promotion provenance rejected" in blockers["test_fixed_budget_v1"]


def test_protocol_v2_belief_deduplicates_and_excludes_unverified_seeds():
    spec = {
        "id": "exp_seed_accounting",
        "payload": {
            "protocol_version": 2,
            "stage": "confirmation",
            "title": "seed accounting",
            "hypothesis": {"statement": "seeded effects improve"},
            "knowledge": {"direction": "optimizer", "subsystem": "update"},
            "plan": [{"seed": seed} for seed in (41, 42, 43)],
            "analysis": {
                "minimum_valid_replicates": 3,
                "success_rule": {"op": "lt", "value": -0.05},
                "falsifier_rule": {"op": "gte", "value": 0.0},
            },
        },
    }

    def decision(record_id, seed, effect, *, policy="evidence-v3"):
        payload = {
            "measurement_verdict": "valid",
            "claim_status": "eligible",
            "policy_version": policy,
            "measurements": {"effect_value": effect},
        }
        if seed is not None:
            payload["verified_seed"] = seed
        return {
            "id": record_id,
            "created_at": f"2026-08-15T00:00:0{len(record_id)}Z",
            "payload": payload,
        }

    rows = [
        decision("e1", 41, -0.1),
        decision("e1_duplicate", 41, -0.9),
        decision("e_missing", None, -0.9),
        decision("e2", 42, -0.1),
        decision("e3", 43, -0.1),
    ]
    belief = _belief(spec, rows)
    assert belief["replicate_seeds"] == [42, 43]
    assert belief["valid_replicates"] == 2
    assert belief["status"] == "preliminary"
    assert belief["evidence_ids"] == ["e2", "e3"]
    assert belief["effect_mean"] == pytest.approx(-0.1)

    otherwise_supported = [
        decision("e1_only", 41, -0.1),
        decision("e2_only", 42, -0.1),
        decision("e3_only", 43, -0.1),
    ]
    unfinished = _belief(spec, otherwise_supported, plan_complete=False)
    assert unfinished["status"] == "preliminary"
    assert unfinished["plan_complete"] is False

    retired_policy = [
        decision("old_e1", 41, -0.1, policy="evidence-v2"),
        decision("old_e2", 42, -0.1, policy="evidence-v2"),
        decision("old_e3", 43, -0.1, policy="evidence-v2"),
    ]
    retired_belief = _belief(spec, retired_policy, plan_complete=True)
    assert retired_belief["valid_replicates"] == 0
    assert retired_belief["status"] == "untested"

    legacy = copy.deepcopy(spec)
    del legacy["payload"]["protocol_version"]
    legacy_belief = _belief(legacy, rows)
    assert legacy_belief["valid_replicates"] == 5
    assert "replicate_seeds" not in legacy_belief


def test_protocol_v2_frame_comes_from_structured_scope(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    proposal = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    proposal["scope"]["budget"]["value"] = 123
    proposal["plan"][0]["arms"][0]["argv"].extend(["--time-budget", "123"])
    proposal["plan"][0]["arms"][1]["argv"].append("--time-budget=123")
    spec = ResearchEngine(store).create(proposal)
    assert SealingAuthority(store).seal(spec["id"], execution_config(script, data))

    conflicting = _protocol_v2_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    conflicting["scope"]["budget"]["value"] = 123
    conflicting["plan"][0]["arms"][0]["argv"].append("--time-budget=300")
    conflict_spec = ResearchEngine(Store(tmp_path / "conflict")).create(conflicting)
    with pytest.raises(RecordError, match="conflicts with scope"):
        SealingAuthority(Store(tmp_path / "conflict")).seal(
            conflict_spec["id"], execution_config(script, data)
        )


def test_protocol_v2_sota_confirmation_requires_promotion_lane(tmp_path):
    script, _ = make_project(tmp_path)
    proposal = _protocol_v2_spec(script)
    proposal["search"]["lane"] = "candidate"
    with pytest.raises(RecordError, match="search.lane=promotion"):
        ResearchEngine(Store(tmp_path / "state")).create(proposal)


def test_protocol_v2_paper_gate_returns_before_registry_cadence_lookup(tmp_path, monkeypatch):
    authority = SealingAuthority(Store(tmp_path / "state"))

    def unexpected_list(_kind):
        raise AssertionError("protocol v2 paper gate should not inspect paper cadence")

    monkeypatch.setattr(authority.store, "list", unexpected_list)
    authority._enforce_paper_gate(
        {"id": "exp_v2", "payload": {"stage": "confirmation", "protocol_version": 2}}
    )
