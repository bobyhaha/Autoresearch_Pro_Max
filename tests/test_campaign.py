from __future__ import annotations

import time

import pytest
from conftest import execution_config, make_project, make_spec

from autoresearch.campaign import CampaignQueue, job_id_for_manifest
from autoresearch.evidence import EvidenceEngine
from autoresearch.execution import ExecutionService, NoResourceAvailable
from autoresearch.research import ResearchEngine
from autoresearch.sealing import SealingAuthority
from autoresearch.store import Store


def pilot_manifest(
    store: Store,
    script,
    data,
    label: str,
    *,
    value: float = 0.9,
    fail: bool = False,
):
    proposal = make_spec(
        script,
        spec_id=f"exp_{label}",
        stage="pilot",
        replicates=1,
        candidate_values=[value],
        fail_control=fail,
        sota_eligible=False,
    )
    spec = ResearchEngine(store).create(proposal)
    execution = execution_config(script, data)
    execution["resources"][0]["id"] = f"cpu_{label}"
    return SealingAuthority(store).seal(spec["id"], execution)


def test_enqueue_is_deterministic_idempotent_and_one_job_per_manifest(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    manifest = pilot_manifest(store, script, data, "one")
    queue = CampaignQueue(store)

    first = queue.enqueue(manifest["id"])
    second = queue.enqueue(manifest["id"])

    assert first == second
    assert first["job_id"] == job_id_for_manifest(manifest["id"])
    assert first["manifest_digest"] == manifest["digest"]
    assert first["state"] == "pending"
    assert len(queue.jobs()) == 1
    assert list(queue.queue_dir.glob("*.tmp")) == []


def test_workers_drain_multiple_manifests_judge_immediately_and_never_reexecute(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    queue = CampaignQueue(store)
    manifests = [pilot_manifest(store, script, data, f"batch_{index}") for index in range(4)]
    for manifest in manifests:
        queue.enqueue(manifest["id"])

    report = queue.work(workers=4)

    assert len(report["completed_job_ids"]) == 4
    assert len(store.list("result_bundle")) == 4
    assert len(store.list("evidence_decision")) == 4
    assert all(row["state"] == "complete" for row in queue.jobs())
    assert all(row["progress"] == {"total": 1, "completed": 1, "judged": 1} for row in queue.jobs())
    assert queue.status()["queue_depth"] == 0
    assert queue.status()["active"] == 0

    # Re-enqueue and re-run are both idempotent at the manifest boundary.
    for manifest in manifests:
        assert queue.enqueue(manifest["id"])["state"] == "complete"
    assert queue.work(workers=2)["completed_job_ids"] == []
    assert len(store.list("result_bundle")) == 4


class NoResourceExecution:
    def execute_next(self, manifest_id):
        raise NoResourceAvailable(f"no slot for {manifest_id}")


def test_no_resource_is_an_explicit_waiting_state_with_backoff(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    manifest = pilot_manifest(store, script, data, "waiting")
    queue = CampaignQueue(
        store,
        execution=NoResourceExecution(),
        no_resource_backoff_seconds=7,
        max_no_resource_backoff_seconds=30,
    )
    job = queue.enqueue(manifest["id"])

    report = queue.work()
    waiting = queue.get_job(job["job_id"])

    assert report["waiting_job_ids"] == [job["job_id"]]
    assert waiting["state"] == "waiting"
    assert waiting["no_resource_count"] == 1
    assert waiting["next_eligible_at"] is not None
    assert "NoResourceAvailable" in waiting["last_error"]
    assert waiting["transitions"][-1]["reason"] == "no_resource"
    assert queue.status()["queue_depth"] == 1


class CrashingExecution:
    def execute_next(self, manifest_id):
        raise RuntimeError(f"uncertain runner failure for {manifest_id}")


def test_uncertain_execution_blocks_and_is_raised_not_retried_or_swallowed(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    manifest = pilot_manifest(store, script, data, "crash")
    queue = CampaignQueue(store, execution=CrashingExecution())
    job = queue.enqueue(manifest["id"])

    with pytest.raises(RuntimeError, match="uncertain runner failure"):
        queue.work()

    blocked = queue.get_job(job["job_id"])
    assert blocked["state"] == "blocked"
    assert blocked["run_attempts"] == 1
    assert "RuntimeError" in blocked["last_error"]
    assert store.list("result_bundle") == []
    assert queue.status()["blocked"] == 1


def test_reconcile_completes_and_judges_a_result_that_landed_before_queue_recovery(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    manifest = pilot_manifest(store, script, data, "landed")
    queue = CampaignQueue(store)
    job = queue.enqueue(manifest["id"])

    result = ExecutionService(store).execute_next(manifest["id"])
    assert store.list("evidence_decision") == []

    queue.reconcile()
    recovered = queue.get_job(job["job_id"])
    assert recovered["state"] == "complete"
    assert recovered["progress"] == {"total": 1, "completed": 1, "judged": 1}
    assert EvidenceEngine(store).judge(result["id"])["payload"]["measurement_verdict"] == "valid"


def test_result_from_another_manifest_cannot_complete_a_queue_job(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    first = pilot_manifest(store, script, data, "manifest_identity")
    second_execution = execution_config(script, data)
    second_execution["resources"][0]["id"] = "cpu_manifest_identity"
    second_execution["runtime"]["timeout_seconds_per_arm"] = 11
    second = SealingAuthority(store).seal(first["payload"]["spec_id"], second_execution)
    assert first["id"] != second["id"]

    job = CampaignQueue(store).enqueue(second["id"])
    ExecutionService(store).execute_next(first["id"])
    queue = CampaignQueue(store)
    queue.reconcile()

    pending = queue.get_job(job["job_id"])
    assert pending["state"] == "pending"
    assert pending["progress"] == {"total": 1, "completed": 0, "judged": 0}


def test_reconcile_requeues_only_an_unstarted_dead_owner_and_blocks_uncertain_claim(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    manifest = pilot_manifest(store, script, data, "recover")
    queue = CampaignQueue(store, pid_alive=lambda pid: False)
    queue.enqueue(manifest["id"])

    claimed = queue._claim_next_job(ignore_health=True)
    assert claimed["state"] == "running"
    recovered = queue.reconcile()[0]
    assert recovered["state"] == "pending"
    assert recovered["transitions"][-1]["reason"] == "recovered_unstarted_job"

    claimed = queue._claim_next_job(ignore_health=True)
    store.write_operational(
        store.inflight_dir / "uncertain.json",
        {
            "manifest_id": manifest["id"],
            "spec_id": manifest["payload"]["spec_id"],
            "replicate_id": "seed_1",
            "owner_pid": 999999,
            "state": "uncertain_after_internal_failure",
        },
    )
    blocked = queue.reconcile()[0]
    assert claimed["state"] == "running"
    assert blocked["state"] == "blocked"
    assert blocked["transitions"][-1]["reason"] == "uncertain_inflight_claim"

    store.release_claim("uncertain")
    resumed = queue.reconcile()[0]
    assert resumed["state"] == "pending"
    assert resumed["transitions"][-1]["reason"] == "operator_released_unstarted_job"


def test_live_local_pid_and_inflight_claim_are_never_stolen(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    manifest = pilot_manifest(store, script, data, "live")
    queue = CampaignQueue(store, pid_alive=lambda pid: pid == 12345)
    job = queue.enqueue(manifest["id"])
    queue._claim_next_job(ignore_health=True)
    store.write_operational(
        store.inflight_dir / "live.json",
        {
            "manifest_id": manifest["id"],
            "spec_id": manifest["payload"]["spec_id"],
            "replicate_id": "seed_1",
            "owner_pid": 12345,
            "state": "running",
        },
    )

    assert queue.reconcile()[0]["state"] == "running"
    assert queue.get_job(job["job_id"])["run_attempts"] == 1


def test_three_consecutive_invalid_pilots_pause_before_the_next_claim(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    queue = CampaignQueue(store)

    for index in range(3):
        manifest = pilot_manifest(store, script, data, f"invalid_{index}", fail=True)
        queue.enqueue(manifest["id"])
        queue.work()

    health = queue.health()
    assert health["paused"] is True
    assert health["consecutive_invalid"] == 3

    healthy = pilot_manifest(store, script, data, "held_healthy")
    held = queue.enqueue(healthy["id"])
    report = queue.work()
    assert report["completed_job_ids"] == []
    assert queue.get_job(held["job_id"])["state"] == "pending"


def test_health_pauses_above_twenty_five_percent_invalid_in_last_eight(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    # No three invalids are adjacent; the rate circuit is independently responsible.
    invalid = {0, 3, 7}
    for index in range(8):
        manifest = pilot_manifest(store, script, data, f"rate_{index}", fail=index in invalid)
        result = ExecutionService(store).execute_next(manifest["id"])
        EvidenceEngine(store).judge(result["id"])

    health = CampaignQueue(store).health()
    assert health["window_size"] == 8
    assert health["invalid_count"] == 3
    assert health["invalid_rate"] == pytest.approx(3 / 8)
    assert health["consecutive_invalid"] == 1
    assert health["paused"] is True
    assert any("invalid rate" in reason for reason in health["reasons"])


def test_follow_mode_obeys_idle_timeout(tmp_path):
    store = Store(tmp_path / "state")
    queue = CampaignQueue(store)
    started = time.monotonic()
    report = queue.work(
        workers=2,
        follow=True,
        poll_seconds=0.01,
        idle_timeout_seconds=0.03,
    )
    elapsed = time.monotonic() - started
    assert elapsed >= 0.02
    assert elapsed < 1
    assert report["status"]["queue_depth"] == 0
