from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from autoresearch.bank import (
    BANK_MAX_USES,
    BANK_TTL_SECONDS,
    PROMOTION_GATE,
    BankIndex,
    baseline_fingerprint,
    context_fingerprint,
    latest_decisions,
)
from autoresearch.store import Store

NOW = datetime(2026, 8, 15, 20, 0, tzinfo=timezone.utc)


def stamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def spec_record(
    spec_id: str,
    lane: str,
    *,
    seed: int = 42,
    bank: str = "champion_1",
    expected_fingerprint: str | None = None,
    references: list[dict] | None = None,
    mutable: list[str] | None = None,
) -> dict:
    arm_name = "control" if lane == "bank" else "candidate"
    search = {
        "lane": lane,
        "bank_id": bank,
        "mutable_code_paths": ["train.py"] if mutable is None else mutable,
    }
    if expected_fingerprint is not None:
        search["baseline_fingerprint"] = expected_fingerprint
    if references is not None:
        search["reference_controls"] = references
    return {
        "id": spec_id,
        "digest": f"digest_{spec_id}",
        "created_at": stamp(NOW - timedelta(hours=2)),
        "payload": {
            "stage": "pilot",
            "comparison_group": "fixed_frame_val_bpb_v2",
            "scope": {"hardware": "h200", "seconds": 300},
            "metric": {"name": "val_bpb", "direction": "minimize"},
            "requirements": {
                "required_metrics": ["val_bpb", "num_steps", "seed"],
                "minimum_steps": 900,
                "require_gpu": True,
                "isolation": "continuous",
            },
            "analysis": {
                "effect": "single",
                "primary_arm": arm_name,
                "minimum_valid_replicates": 1,
                "success_rule": {"op": "lt", "value": 1e9},
                "falsifier_rule": {"op": "gte", "value": 1e9},
                "sota_eligible": False,
            },
            "plan": [
                {
                    "replicate_id": f"seed_{seed}",
                    "seed": seed,
                    "arms": [
                        {
                            "name": arm_name,
                            "argv": ["python", "train.py", f"--mode={lane}"],
                            "env": {"AUTORESEARCH_SEED": str(seed)},
                        }
                    ],
                }
            ],
            "search": search,
        },
    }


def manifest_record(
    spec: dict,
    *,
    train_digest: str = "train-v1",
    evaluator_digest: str = "eval-v1",
    data_digest: str = "data-v1",
) -> dict:
    spec_id = spec["id"]
    return {
        "id": f"manifest_{spec_id}",
        "digest": f"manifest_digest_{spec_id}",
        "created_at": stamp(NOW - timedelta(hours=2)),
        "payload": {
            "spec_id": spec_id,
            "spec_digest": spec["digest"],
            "plan": copy.deepcopy(spec["payload"]["plan"]),
            "code_bindings": [
                {"execution_path": "train.py", "sha256": train_digest},
                {"execution_path": "evaluate.py", "sha256": evaluator_digest},
            ],
            "data_bindings": [{"execution_path": "data_manifest.json", "sha256": data_digest}],
        },
    }


def result_record(
    spec: dict,
    manifest: dict,
    result_id: str,
    *,
    gpu_uuid: str | None = "GPU-A",
    resource_id: str = "host_1",
    gpu: int | None = 0,
    started: datetime,
    ended: datetime,
) -> dict:
    return {
        "id": result_id,
        "digest": f"digest_{result_id}",
        "created_at": stamp(ended),
        "payload": {
            "spec_id": spec["id"],
            "manifest_id": manifest["id"],
            "resource": {"id": resource_id, "gpu": gpu},
            "launch_telemetry": {"state": "available", "uuid": gpu_uuid},
            "lifecycle": {"started_at": stamp(started), "ended_at": stamp(ended)},
        },
    }


def decision_record(
    spec: dict,
    result: dict,
    decision_id: str,
    *,
    value: float,
    seed: int | None = 42,
    verdict: str = "valid",
    created: datetime | None = None,
) -> dict:
    primary = spec["payload"]["analysis"]["primary_arm"]
    measurements = {"arms": {primary: {"val_bpb": value, "seed": seed}}}
    if seed is not None:
        measurements["verified_seed"] = seed
    return {
        "id": decision_id,
        "created_at": stamp(created or (NOW - timedelta(minutes=1))),
        "payload": {
            "result_id": result["id"],
            "result_digest": result["digest"],
            "spec_id": spec["id"],
            "measurement_verdict": verdict,
            "policy_version": "evidence-v3",
            "measurements": measurements,
        },
    }


def make_bank(
    suffix: str,
    *,
    gpu_uuid: str = "GPU-A",
    resource_id: str = "host_1",
    ended: datetime | None = None,
    value: float = 1.0,
    verdict: str = "valid",
    train_digest: str = "train-v1",
    evaluator_digest: str = "eval-v1",
) -> tuple[dict, dict, dict, dict]:
    ended = ended or (NOW - timedelta(minutes=10))
    spec = spec_record(f"bank_{suffix}", "bank")
    manifest = manifest_record(spec, train_digest=train_digest, evaluator_digest=evaluator_digest)
    result = result_record(
        spec,
        manifest,
        f"result_bank_{suffix}",
        gpu_uuid=gpu_uuid,
        resource_id=resource_id,
        started=ended - timedelta(minutes=5),
        ended=ended,
    )
    decision = decision_record(
        spec, result, f"evidence_bank_{suffix}", value=value, verdict=verdict
    )
    return spec, manifest, result, decision


def flatten(bundles):
    return [[row[index] for row in bundles] for index in range(4)]


def candidate_bundle(
    suffix: str,
    *,
    baseline_fp: str,
    references: list[dict],
    gpu_uuid: str = "GPU-A",
    resource_id: str = "host_1",
    started: datetime | None = None,
    value: float = 0.999,
    evaluator_digest: str = "eval-v1",
) -> tuple[dict, dict, dict, dict]:
    started = started or (NOW - timedelta(minutes=3))
    spec = spec_record(
        f"candidate_{suffix}",
        "candidate",
        expected_fingerprint=baseline_fp,
        references=copy.deepcopy(references),
    )
    manifest = manifest_record(
        spec, train_digest=f"candidate-code-{suffix}", evaluator_digest=evaluator_digest
    )
    result = result_record(
        spec,
        manifest,
        f"result_candidate_{suffix}",
        gpu_uuid=gpu_uuid,
        resource_id=resource_id,
        started=started,
        ended=started + timedelta(minutes=5),
    )
    decision = decision_record(
        spec, result, f"evidence_candidate_{suffix}", value=value, created=result_time(result)
    )
    return spec, manifest, result, decision


def result_time(result: dict) -> datetime:
    return datetime.fromisoformat(result["payload"]["lifecycle"]["ended_at"].replace("Z", "+00:00"))


def bank_query(index: BankIndex, spec: dict, manifest: dict, *, at=NOW, gpu=None):
    return index.eligible_controls(
        bank_id="champion_1",
        baseline_fingerprint=baseline_fingerprint(spec, manifest),
        context_fingerprint=context_fingerprint(spec, manifest),
        seed=42,
        at=at,
        gpu_key=gpu,
    )


def test_latest_decision_per_result_prevents_policy_double_counting():
    spec, _, result, _ = make_bank("latest")
    old = decision_record(
        spec,
        result,
        "evidence_old",
        value=1.0,
        verdict="valid",
        created=NOW - timedelta(minutes=2),
    )
    new = decision_record(
        spec,
        result,
        "evidence_new",
        value=1.0,
        verdict="invalid",
        created=NOW - timedelta(minutes=1),
    )
    assert latest_decisions([new, old])[result["id"]]["id"] == "evidence_new"


def test_retired_evidence_policy_cannot_seed_the_control_bank():
    spec, manifest, result, decision = make_bank("retired_policy")
    decision["payload"]["policy_version"] = "evidence-v2"
    index = BankIndex([spec], [manifest], [result], [decision], now=NOW)
    assert index.controls == []


def test_fingerprints_separate_baseline_identity_from_mutable_code_context():
    spec, manifest, *_ = make_bank("fingerprint")
    changed_train = copy.deepcopy(manifest)
    changed_train["payload"]["code_bindings"][0]["sha256"] = "new-train"
    changed_eval = copy.deepcopy(manifest)
    changed_eval["payload"]["code_bindings"][1]["sha256"] = "new-eval"

    assert baseline_fingerprint(spec, changed_train) != baseline_fingerprint(spec, manifest)
    assert context_fingerprint(spec, changed_train) == context_fingerprint(spec, manifest)
    assert context_fingerprint(spec, changed_eval) != context_fingerprint(spec, manifest)


def test_bank_requires_latest_valid_evidence_verified_seed_and_physical_gpu():
    valid = make_bank("valid")
    invalid = make_bank("invalid", gpu_uuid="GPU-B", verdict="invalid")
    missing_seed = list(make_bank("no_seed", gpu_uuid="GPU-C"))
    missing_seed[3] = decision_record(
        missing_seed[0], missing_seed[2], "evidence_no_seed", value=1.0, seed=None
    )
    no_uuid = list(make_bank("no_uuid", gpu_uuid="GPU-D"))
    no_uuid[2]["payload"]["launch_telemetry"].pop("uuid")
    specs, manifests, results, decisions = flatten([valid, invalid, missing_seed, no_uuid])

    index = BankIndex(specs, manifests, results, decisions, now=NOW)

    assert [row["result_id"] for row in index.controls] == [valid[2]["id"]]


def test_eligible_controls_are_exact_fresh_prior_same_gpu_and_never_fallback():
    fresh_a = make_bank("fresh_a", gpu_uuid="GPU-A", value=1.0)
    stale_a = make_bank(
        "stale_a",
        gpu_uuid="GPU-A",
        ended=NOW - timedelta(seconds=BANK_TTL_SECONDS + 1),
        value=0.1,
    )
    future_a = make_bank("future_a", gpu_uuid="GPU-A", ended=NOW + timedelta(seconds=1), value=0.2)
    invalid_b = make_bank("invalid_b", gpu_uuid="GPU-B", verdict="invalid", value=0.3)
    specs, manifests, results, decisions = flatten([fresh_a, stale_a, future_a, invalid_b])
    index = BankIndex(specs, manifests, results, decisions, now=NOW)
    fp = baseline_fingerprint(fresh_a[0], fresh_a[1])
    context = context_fingerprint(fresh_a[0], fresh_a[1])

    rows = index.eligible_controls(
        bank_id="champion_1",
        baseline_fingerprint=fp,
        context_fingerprint=context,
        seed=42,
        at=NOW,
        gpu_key="host_1:GPU-A",
    )
    assert [row["result_id"] for row in rows] == [fresh_a[2]["id"]]
    assert (
        index.eligible_controls(
            bank_id="champion_1",
            baseline_fingerprint=fp,
            context_fingerprint=context,
            seed=42,
            at=NOW,
            gpu_key="host_1:GPU-B",
        )
        == []
    ), "invalid B must not cause a cross-GPU fallback to A"
    assert (
        index.eligible_controls(
            bank_id="champion_1",
            baseline_fingerprint=fp,
            context_fingerprint="wrong-context",
            seed=42,
            at=NOW,
        )
        == []
    )
    assert (
        index.eligible_controls(
            bank_id="wrong-bank",
            baseline_fingerprint=fp,
            context_fingerprint=context,
            seed=42,
            at=NOW,
        )
        == []
    )


def test_eighth_use_exhausts_a_control_and_ninth_candidate_cannot_score():
    bank = make_bank("cap", ended=NOW - timedelta(minutes=45))
    base_index = BankIndex(*flatten([bank]), now=NOW - timedelta(minutes=40))
    fp = baseline_fingerprint(bank[0], bank[1])
    context = context_fingerprint(bank[0], bank[1])
    refs = base_index.eligible_controls(
        bank_id="champion_1",
        baseline_fingerprint=fp,
        context_fingerprint=context,
        seed=42,
        at=NOW - timedelta(minutes=40),
    )
    candidates = []
    for index in range(BANK_MAX_USES):
        candidates.append(
            candidate_bundle(
                f"prior_{index}",
                baseline_fp=fp,
                references=refs,
                started=NOW - timedelta(minutes=38 - index * 4),
            )
        )
    ninth = candidate_bundle(
        "ninth", baseline_fp=fp, references=refs, started=NOW - timedelta(minutes=3)
    )
    all_bundles = [bank, *candidates, ninth]
    index = BankIndex(*flatten(all_bundles), now=NOW)

    assert index.use_count(bank[2]["id"], before=NOW) == BANK_MAX_USES + 1
    assert (
        index.eligible_controls(
            bank_id="champion_1",
            baseline_fingerprint=fp,
            context_fingerprint=context,
            seed=42,
            at=NOW - timedelta(minutes=3),
        )
        == []
    )
    score = index.score_candidate(ninth[2]["id"])
    assert score["status"] == "unscored"
    assert "overused" in score["reason"]


def test_candidate_uses_only_frozen_same_gpu_control_not_newer_or_global_best():
    frozen_a = make_bank("frozen_a", gpu_uuid="GPU-A", value=1.0)
    other_b = make_bank("other_b", gpu_uuid="GPU-B", value=0.8)
    bank_only = BankIndex(*flatten([frozen_a, other_b]), now=NOW - timedelta(minutes=4))
    fp = baseline_fingerprint(frozen_a[0], frozen_a[1])
    context = context_fingerprint(frozen_a[0], frozen_a[1])
    refs = bank_only.eligible_controls(
        bank_id="champion_1",
        baseline_fingerprint=fp,
        context_fingerprint=context,
        seed=42,
        at=NOW - timedelta(minutes=4),
    )
    # This control appears after references were frozen but before the candidate.
    # It is much worse, so a dynamic lookup would make the candidate look magical.
    newer_a = make_bank(
        "newer_a", gpu_uuid="GPU-A", ended=NOW - timedelta(minutes=3, seconds=30), value=1.2
    )
    candidate = candidate_bundle(
        "frozen",
        baseline_fp=fp,
        references=refs,
        gpu_uuid="GPU-A",
        started=NOW - timedelta(minutes=3),
        value=0.999,
    )
    index = BankIndex(*flatten([frozen_a, other_b, newer_a, candidate]), now=NOW)

    score = index.score_candidate(candidate[2]["id"])
    assert score["status"] == "scored"
    assert score["control_result_id"] == frozen_a[2]["id"]
    assert score["control_value"] == 1.0
    assert score["delta"] == pytest.approx(-0.001)
    assert score["promotion_due"] is True
    assert score["sota_eligible"] is False
    assert abs(score["delta"]) > PROMOTION_GATE


def test_candidate_with_no_frozen_same_gpu_reference_is_unscored_without_fallback():
    bank_a = make_bank("only_a", gpu_uuid="GPU-A")
    bank_index = BankIndex(*flatten([bank_a]), now=NOW - timedelta(minutes=4))
    fp = baseline_fingerprint(bank_a[0], bank_a[1])
    context = context_fingerprint(bank_a[0], bank_a[1])
    refs = bank_index.eligible_controls(
        bank_id="champion_1",
        baseline_fingerprint=fp,
        context_fingerprint=context,
        seed=42,
        at=NOW - timedelta(minutes=4),
    )
    candidate_b = candidate_bundle("gpu_b", baseline_fp=fp, references=refs, gpu_uuid="GPU-B")
    index = BankIndex(*flatten([bank_a, candidate_b]), now=NOW)

    score = index.score_candidate(candidate_b[2]["id"])
    assert score["status"] == "unscored"
    assert score["delta"] is None
    assert "physical GPU" in score["reason"]


def test_candidate_context_mismatch_blocks_frozen_control():
    bank = make_bank("context")
    bank_index = BankIndex(*flatten([bank]), now=NOW - timedelta(minutes=4))
    fp = baseline_fingerprint(bank[0], bank[1])
    refs = bank_query(bank_index, bank[0], bank[1], at=NOW - timedelta(minutes=4))
    candidate = candidate_bundle(
        "context_bad",
        baseline_fp=fp,
        references=refs,
        evaluator_digest="eval-changed",
    )
    index = BankIndex(*flatten([bank, candidate]), now=NOW)

    score = index.score_candidate(candidate[2]["id"])
    assert score["status"] == "unscored"
    assert "context" in score["reason"]


def test_cpu_slot_fallback_and_rebuildable_views(tmp_path):
    bank = list(make_bank("cpu"))
    bank[2] = result_record(
        bank[0],
        bank[1],
        "result_bank_cpu",
        gpu_uuid=None,
        gpu=None,
        started=NOW - timedelta(minutes=15),
        ended=NOW - timedelta(minutes=10),
    )
    bank[3] = decision_record(bank[0], bank[2], "evidence_bank_cpu", value=1.0)
    index = BankIndex(*flatten([bank]), now=NOW)
    assert index.controls[0]["gpu_key"] == "host_1:cpu"

    store = Store(tmp_path / "state")
    written = index.write_views(store)
    bank_view = json.loads((store.views_dir / "BANK.json").read_text())
    queue_view = json.loads((store.views_dir / "PROMOTION_QUEUE.json").read_text())
    assert bank_view == written["bank"]
    assert queue_view["promotion_queue"] == []
    assert queue_view["note"].endswith("never become SOTA")
