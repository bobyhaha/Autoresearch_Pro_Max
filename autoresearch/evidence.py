"""Evidence Engine: separate measurement integrity from claim authority."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from .protocol import EVIDENCE_POLICY_VERSION
from .records import RecordError, canonical_json, make_record
from .store import Store

POLICY_VERSION = "evidence-v1"
POLICY_VERSION_V2 = EVIDENCE_POLICY_VERSION
FRAME_RELATIVE_TOLERANCE = 0.05
FRAME_ABSOLUTE_TOLERANCE_SECONDS = 0.25
FRAME_SMALL_BUDGET_TOLERANCE_MULTIPLIER = 2.0
TRAINING_RUNNER_TOTAL_MAX_TOLERANCE_SECONDS = 15.0


class EvidenceEngine:
    def __init__(self, store: Store) -> None:
        self.store = store

    def judge(self, result_id: str) -> dict[str, Any]:
        result = self.store.get("result_bundle", result_id)
        manifest = self.store.get("execution_manifest", result["payload"]["manifest_id"])
        spec = self.store.get("experiment_spec", result["payload"]["spec_id"])
        policy_version = (
            POLICY_VERSION_V2 if spec["payload"].get("protocol_version") == 2 else POLICY_VERSION
        )
        verdict, reasons, measurements = assess(
            result["payload"],
            manifest["payload"],
            spec["payload"],
            step_baselines=self._step_baselines(),
        )
        link_errors: list[str] = []
        if result["payload"].get("manifest_digest") != manifest["digest"]:
            link_errors.append("ResultBundle does not bind the loaded ExecutionManifest digest")
        if manifest["payload"].get("spec_id") != spec["id"]:
            link_errors.append("ExecutionManifest names a different ExperimentSpec")
        if manifest["payload"].get("spec_digest") != spec["digest"]:
            link_errors.append("ExecutionManifest does not bind the loaded ExperimentSpec digest")
        if result["payload"].get("stage") != spec["payload"].get("stage"):
            link_errors.append("ResultBundle stage disagrees with the ExperimentSpec")
        for field in ("protocol_version", "search", "scope"):
            if manifest["payload"].get(field) != spec["payload"].get(field):
                link_errors.append(f"ExecutionManifest changes ExperimentSpec field {field}")
        if link_errors:
            verdict = "invalid"
            reasons = sorted(set(reasons + link_errors))
        if verdict == "valid" and spec["payload"]["stage"] == "confirmation":
            claim_status = "eligible"
        elif verdict == "valid":
            claim_status = "ineligible"
            reasons.append("pilot measurements are exploratory and cannot support claims or SOTA")
        else:
            claim_status = "blocked"
        payload = {
            "result_id": result_id,
            "result_digest": result["digest"],
            "spec_id": spec["id"],
            "stage": spec["payload"]["stage"],
            "measurement_verdict": verdict,
            "claim_status": claim_status,
            "reasons": reasons,
            "measurements": measurements,
            "policy_version": policy_version,
        }
        if policy_version == POLICY_VERSION_V2:
            payload["verified_seed"] = measurements.get("verified_seed")
            payload["gpu_key"] = measurements.get("gpu_key")
        evidence_id = f"evidence_{result['digest'][:12]}_{policy_version.replace('-', '_')}"
        path = self.store.record_path("evidence_decision", evidence_id)
        if path.exists():
            existing = self.store.get("evidence_decision", evidence_id)
            if existing["payload"] == payload:
                return existing
            raise RecordError(f"evidence id collision: {evidence_id}")
        return self.store.put(make_record("evidence_decision", evidence_id, payload))

    def _step_baselines(self) -> dict[str, float]:
        """Median step count per exact argv and physical GPU placement.

        Self-calibrating and conservative in three ways that matter:

        * Keyed on the FULL argv plus host/GPU UUID, so a 600 s arm or a
          naturally slower card is compared only against its own history.
        * Built only from arms whose replicate carried no co-tenancy, so a contended run
          cannot lower the bar that catches contended runs.
        * Requires THROUGHPUT_MIN_HISTORY samples before it will judge anything, so a new
          configuration is never invalidated for lack of a baseline. New work is always
          admissible; only configurations with an established clean rate get policed.

        The median, not the mean, because the failure being detected is a heavy left tail.
        """
        latest_decisions: dict[str, Mapping[str, Any]] = {}
        for decision in self.store.list("evidence_decision"):
            result_id = decision["payload"].get("result_id")
            if not isinstance(result_id, str):
                continue
            previous = latest_decisions.get(result_id)
            if previous is None or (decision["created_at"], decision["id"]) > (
                previous["created_at"],
                previous["id"],
            ):
                latest_decisions[result_id] = decision

        v2_spec_ids = {
            row["id"]
            for row in self.store.list("experiment_spec")
            if row["payload"].get("protocol_version") == 2
        }
        counts: dict[str, list[float]] = {}
        for row in self.store.list("result_bundle"):
            payload = row["payload"]
            decision = latest_decisions.get(row["id"])
            if (
                payload.get("status") != "completed"
                or decision is None
                or decision["payload"].get("measurement_verdict") != "valid"
                or (
                    payload.get("spec_id") in v2_spec_ids
                    and decision["payload"].get("policy_version") != POLICY_VERSION_V2
                )
            ):
                continue
            arms = payload.get("arms", [])
            mps = [(a.get("telemetry") or {}).get("max_compute_processes") for a in arms]
            if max([m for m in mps if m is not None] or [1]) > 1:
                continue
            for arm in arms:
                if arm.get("status") != "completed" or arm.get("return_code") != 0:
                    continue
                steps = (arm.get("metrics") or {}).get("num_steps")
                if not isinstance(steps, (int, float)) or isinstance(steps, bool):
                    continue
                key = throughput_key(arm.get("payload_argv") or [], payload)
                if key is None:
                    # Legacy results without a verifiable physical GPU identity
                    # cannot safely establish a pooled throughput baseline.
                    continue
                counts.setdefault(key, []).append(float(steps))
        return {
            key: statistics.median(values)
            for key, values in counts.items()
            if len(values) >= THROUGHPUT_MIN_HISTORY
        }

    def judge_all(self) -> list[dict[str, Any]]:
        existing = {
            (
                row["payload"]["result_id"],
                row["payload"]["result_digest"],
                row["payload"]["policy_version"],
            )
            for row in self.store.list("evidence_decision")
        }
        decisions: list[dict[str, Any]] = []
        for result in self.store.list("result_bundle"):
            spec = self.store.get("experiment_spec", result["payload"]["spec_id"])
            policy_version = (
                POLICY_VERSION_V2
                if spec["payload"].get("protocol_version") == 2
                else POLICY_VERSION
            )
            key = (result["id"], result["digest"], policy_version)
            if key not in existing:
                decisions.append(self.judge(result["id"]))
        return decisions


# A run whose step count is far below what the SAME argv normally achieves was slowed by
# something outside the experiment.  On this box that is usually a neighbour: co-tenancy is
# already caught by max_compute_processes, but HOST-level contention -- CPU, PCIe, power --
# leaves process_count at 1 and still removes 20% of the optimizer steps.
#
# This matters more than it sounds.  Pairing cancels PLACEMENT because both arms share a
# leased GPU, but the arms run SEQUENTIALLY, minutes apart, so a load that arrives between
# them hits one and not the other.  Observed 2026-08-12 on exp_cxb_turn_1x2: the control got
# 803 steps (82% of its clean 975) while the candidate got 1936 (105% of its clean 1845).
# The pair reported -0.008689 -- the OPPOSITE sign to the hypothesis it was testing, twice,
# because the comparison was between two moments rather than two configurations.
THROUGHPUT_FLOOR = 0.90  # fraction of the argv's own clean median
THROUGHPUT_MIN_HISTORY = 5  # never judge against a baseline thinner than this


def gpu_key(result: Mapping[str, Any]) -> str | None:
    """Stable execution placement key, preferring physical UUID over index.

    ``result`` may be a ResultBundle payload or a resource-like mapping. Legacy
    bundles can recover UUID from launch telemetry and host from the recorded
    execution environment; if either is unavailable there is intentionally no
    pooled fallback.
    """
    nested = result.get("resource")
    resource = nested if isinstance(nested, Mapping) else result
    launch = result.get("launch_telemetry", {})
    environment = result.get("environment", {})
    host_id = resource.get("host_id") or environment.get("host") or resource.get("id")
    gpu_uuid = resource.get("gpu_uuid") or launch.get("uuid")
    gpu_index = resource.get("gpu")
    if not isinstance(host_id, str) or not host_id.strip():
        return None
    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        return None
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
        return None
    return f"{host_id.strip()}::{gpu_uuid.strip()}"


def throughput_key(argv: Any, placement: Mapping[str, Any]) -> str | None:
    placement_key = gpu_key(placement)
    if placement_key is None:
        return None
    return canonical_json(
        {"gpu_key": placement_key, "argv": [str(token) for token in (argv or [])]}
    )


def throughput_reason(
    arm_name: str,
    steps: Any,
    argv: Any,
    step_baselines: Mapping[str, float] | None,
    placement: Mapping[str, Any] | None = None,
) -> str | None:
    """Reject an arm that ran far slower than the same argv normally does.

    Pure so it can be tested without constructing a whole admissible ResultBundle.
    Returns None when there is no baseline, no step count, or the arm is healthy --
    a new configuration is never punished for lacking history.
    """
    if not step_baselines:
        return None
    if not isinstance(steps, (int, float)) or isinstance(steps, bool):
        return None
    # Refuse a pooled fallback. The same argv has measurably different healthy
    # rates on different cards, so missing placement means "no baseline", not
    # permission to borrow one from another GPU.
    if placement is None:
        return None
    key = throughput_key(argv, placement)
    if key is None:
        return None
    baseline = step_baselines.get(key)
    if not baseline or float(steps) >= baseline * THROUGHPUT_FLOOR:
        return None
    return (
        f"arm {arm_name} ran {float(steps):g} steps against a clean median of "
        f"{baseline:g} for the same argv ({float(steps) / baseline:.0%}); "
        "host contention, not a measurement"
    )


def assess(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    step_baselines: Mapping[str, float] | None = None,
) -> tuple[str, list[str], dict[str, Any]]:
    invalid: list[str] = []
    unknown: list[str] = []
    protocol_v2 = spec.get("protocol_version") == 2
    if result["status"] != "completed":
        invalid.append(f"execution status is {result['status']}")
    if result.get("manifest_digest") is None:
        unknown.append("manifest digest is absent")
    expected_bindings = {
        (row["execution_path"], row["sha256"])
        for row in list(manifest.get("code_bindings", [])) + list(manifest.get("data_bindings", []))
    }
    pre_bindings = list(result.get("binding_checks", []))
    post_bindings = list(result.get("post_binding_checks", []))
    binding_rows = pre_bindings + post_bindings
    if not binding_rows:
        unknown.append("no code/data binding checks were recorded")
    else:
        for phase, rows in (("launch", pre_bindings), ("completion", post_bindings)):
            observed = {(row.get("execution_path"), row.get("expected_sha256")) for row in rows}
            if observed != expected_bindings or len(rows) != len(expected_bindings):
                invalid.append(f"{phase} binding checks do not exactly cover the manifest")
        if any(row.get("state") == "mismatch" for row in binding_rows):
            invalid.append("code or data did not match the sealed manifest")
        elif any(row.get("state") != "verified" for row in binding_rows):
            unknown.append("a code/data binding could not be verified")
        if any(
            row.get("state") == "verified"
            and row.get("actual_sha256") != row.get("expected_sha256")
            for row in binding_rows
        ):
            invalid.append("a verified binding has inconsistent hash fields")

    requirements = spec["requirements"]
    environment = result.get("environment", {})
    if environment.get("scope") != "execution_host" or environment.get("state") == "unknown":
        unknown.append("execution-host environment could not be established")
    result_resource = result.get("resource", {})
    launch = result.get("launch_telemetry", {})
    verified_gpu_key: str | None = None
    authorized_resource = next(
        (
            row
            for row in manifest.get("resources", [])
            if row.get("id") == result_resource.get("id")
        ),
        None,
    )
    if authorized_resource is None:
        invalid.append("ResultBundle names a resource absent from the manifest")
    else:
        for field in ("backend", "workdir"):
            if result_resource.get(field) != authorized_resource.get(field):
                invalid.append(f"execution resource {field} disagrees with the manifest")
        gpu = result_resource.get("gpu")
        allowed_gpus = authorized_resource.get("gpus", [])
        if gpu is not None and gpu not in allowed_gpus:
            invalid.append("execution used a GPU absent from the authorized resource")
        if requirements.get("require_gpu", True) and gpu is None:
            invalid.append("the ExperimentSpec requires a GPU but none was recorded")
        if protocol_v2:
            expected_host_id = authorized_resource.get("host_id", authorized_resource.get("id"))
            if result_resource.get("host_id") != expected_host_id:
                invalid.append("execution resource host_id disagrees with the manifest")
            scope = spec.get("scope", {})
            expected_hardware = scope.get("hardware_class") if isinstance(scope, Mapping) else None
            if (
                expected_hardware is not None
                and authorized_resource.get("hardware_class") != expected_hardware
            ):
                invalid.append(
                    "authorized resource hardware_class disagrees with the structured scope"
                )
    if protocol_v2 and requirements.get("require_gpu", True):
        resource_gpu = result_resource.get("gpu")
        resource_uuid = result_resource.get("gpu_uuid")
        launch_gpu = launch.get("gpu")
        launch_uuid = launch.get("uuid")
        if launch_gpu != resource_gpu:
            invalid.append("launch GPU index disagrees with the allocated resource")
        if not resource_uuid or not launch_uuid:
            invalid.append("protocol v2 requires launch and resource GPU UUIDs")
        elif launch_uuid != resource_uuid:
            invalid.append("launch GPU UUID disagrees with the allocated resource")
        elif launch_gpu == resource_gpu:
            verified_gpu_key = gpu_key(result)
            if verified_gpu_key is None:
                invalid.append("protocol v2 could not establish a physical GPU key")
    plan = next(
        (row for row in manifest["plan"] if row["replicate_id"] == result["replicate_id"]),
        None,
    )
    if plan is None:
        invalid.append("replicate is not present in the sealed plan")
        planned_arm_names: list[str] = []
    else:
        planned_arm_names = [row["name"] for row in plan["arms"]]
    arm_rows = list(result.get("arms", []))
    actual_arm_names = [row.get("name") for row in arm_rows]
    arms = {row.get("name"): row for row in arm_rows}
    if actual_arm_names != planned_arm_names:
        invalid.append("result does not preserve the exact preregistered arm order")
    if plan is not None:
        for planned_arm, actual_arm in zip(plan["arms"], arm_rows, strict=False):
            if actual_arm.get("payload_argv") != planned_arm.get("argv"):
                invalid.append(f"arm {planned_arm['name']} argv differs from the manifest")
            if actual_arm.get("payload_env") != planned_arm.get("env", {}):
                invalid.append(f"arm {planned_arm['name']} environment differs from the manifest")

    required_metrics = list(requirements["required_metrics"])
    minimum_steps = float(requirements.get("minimum_steps", 0))
    arm_metrics: dict[str, dict[str, float]] = {}
    seed_verified_arms: set[str] = set()
    planned_seed = plan.get("seed") if plan is not None else None
    frame_kind: str | None = None
    frame_seconds: float | None = None
    frame_tolerance: float | None = None
    if protocol_v2:
        scope = spec.get("scope", {})
        budget = scope.get("budget", {}) if isinstance(scope, Mapping) else {}
        raw_kind = budget.get("kind") if isinstance(budget, Mapping) else None
        raw_frame = budget.get("value") if isinstance(budget, Mapping) else None
        if (
            raw_kind in {"wall_seconds", "training_seconds"}
            and isinstance(raw_frame, (int, float))
            and not isinstance(raw_frame, bool)
            and math.isfinite(float(raw_frame))
            and float(raw_frame) > 0
        ):
            frame_kind = str(raw_kind)
            frame_seconds = float(raw_frame)
            # Five percent absorbs normal launch/poll/SSH teardown on the real 300 s
            # frame.  Short smoke-test frames get a bounded absolute allowance, but
            # never an attacker-inflatable allowance derived from runtime settings.
            frame_tolerance = max(
                frame_seconds * FRAME_RELATIVE_TOLERANCE,
                min(
                    FRAME_ABSOLUTE_TOLERANCE_SECONDS,
                    frame_seconds * FRAME_SMALL_BUDGET_TOLERANCE_MULTIPLIER,
                ),
            )
        else:
            invalid.append("protocol v2 could not establish a sealed timing frame")
    for arm_name in planned_arm_names:
        arm = arms.get(arm_name)
        if arm is None:
            continue
        if arm.get("status") != "completed" or arm.get("return_code") != 0:
            invalid.append(f"arm {arm_name} did not complete successfully")
        raw_metrics = arm.get("metrics", {})
        values: dict[str, float] = {}
        for name in required_metrics:
            value = raw_metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                invalid.append(f"arm {arm_name} is missing required metric {name}")
                continue
            number = float(value)
            if not math.isfinite(number):
                invalid.append(f"arm {arm_name} metric {name} is not finite")
                continue
            values[name] = number
        if protocol_v2:
            runner_wall_value: float | None = None
            runner_wall_seconds = arm.get("wall_seconds")
            if (
                isinstance(runner_wall_seconds, bool)
                or not isinstance(runner_wall_seconds, (int, float))
                or not math.isfinite(float(runner_wall_seconds))
                or float(runner_wall_seconds) <= 0
            ):
                invalid.append(f"arm {arm_name} lacks a positive finite runner wall clock")
            else:
                runner_wall_value = float(runner_wall_seconds)
                values["runner_wall_seconds"] = runner_wall_value
                if (
                    frame_kind == "wall_seconds"
                    and frame_seconds is not None
                    and frame_tolerance is not None
                    and abs(runner_wall_value - frame_seconds) > frame_tolerance
                ):
                    invalid.append(
                        f"arm {arm_name} runner wall_seconds={runner_wall_value:g}, outside "
                        f"the sealed {frame_seconds:g}s frame "
                        f"(tolerance {frame_tolerance:g}s)"
                    )
            emitted_seed = raw_metrics.get("seed")
            if (
                isinstance(emitted_seed, bool)
                or not isinstance(emitted_seed, (int, float))
                or not math.isfinite(float(emitted_seed))
                or not float(emitted_seed).is_integer()
            ):
                invalid.append(f"arm {arm_name} emitted a non-integral seed")
            elif int(emitted_seed) != planned_seed:
                invalid.append(
                    f"arm {arm_name} emitted seed {int(emitted_seed)} instead of {planned_seed}"
                )
            else:
                seed_verified_arms.add(arm_name)
            training_seconds = raw_metrics.get("training_seconds")
            total_seconds = raw_metrics.get("total_seconds")
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (training_seconds, total_seconds)
            ):
                training_value = float(training_seconds)
                total_value = float(total_seconds)
                if training_value < 0 or total_value <= 0 or training_value > total_value:
                    invalid.append(f"arm {arm_name} emitted inconsistent timing metrics")
                if frame_kind == "wall_seconds" and frame_seconds is not None:
                    tolerance = max(0.05, frame_seconds * 0.05)
                    if abs(total_value - frame_seconds) > tolerance:
                        invalid.append(
                            f"arm {arm_name} reported total_seconds={total_value:g}, outside "
                            f"the sealed {frame_seconds:g}s frame (tolerance {tolerance:g}s)"
                        )
                elif frame_kind == "training_seconds" and frame_seconds is not None:
                    tolerance = max(0.05, frame_seconds * FRAME_RELATIVE_TOLERANCE)
                    if abs(training_value - frame_seconds) > tolerance:
                        invalid.append(
                            f"arm {arm_name} reported training_seconds={training_value:g}, "
                            f"outside the sealed {frame_seconds:g}s training frame "
                            f"(tolerance {tolerance:g}s)"
                        )
                    if runner_wall_value is not None:
                        if runner_wall_value < frame_seconds:
                            invalid.append(
                                f"arm {arm_name} runner wall_seconds={runner_wall_value:g} "
                                f"is shorter than the sealed {frame_seconds:g}s training frame"
                            )
                        if runner_wall_value + FRAME_ABSOLUTE_TOLERANCE_SECONDS < training_value:
                            invalid.append(
                                f"arm {arm_name} reported training_seconds={training_value:g} "
                                f"longer than runner wall_seconds={runner_wall_value:g}"
                            )
                        reconciliation_tolerance = max(
                            FRAME_ABSOLUTE_TOLERANCE_SECONDS,
                            min(
                                TRAINING_RUNNER_TOTAL_MAX_TOLERANCE_SECONDS,
                                frame_seconds * FRAME_RELATIVE_TOLERANCE,
                            ),
                        )
                        if abs(total_value - runner_wall_value) > reconciliation_tolerance:
                            invalid.append(
                                f"arm {arm_name} emitted total_seconds={total_value:g}, which "
                                f"does not match runner wall_seconds={runner_wall_value:g} "
                                f"within {reconciliation_tolerance:g}s"
                            )
        steps = raw_metrics.get("num_steps", raw_metrics.get("steps"))
        if minimum_steps:
            if not isinstance(steps, (int, float)) or isinstance(steps, bool):
                invalid.append(f"arm {arm_name} is missing a step count")
            elif float(steps) < minimum_steps:
                invalid.append(
                    f"arm {arm_name} completed {float(steps):g} steps below {minimum_steps:g}"
                )
        arm_metrics[arm_name] = values

        isolation = requirements.get("isolation", "launch")
        telemetry = arm.get("telemetry", {})
        if requirements.get("require_gpu", True):
            if isolation == "continuous" and not telemetry.get("sample_count"):
                unknown.append(f"arm {arm_name} lacks continuous GPU telemetry")
            if isolation == "continuous" and telemetry.get("unavailable_samples"):
                unknown.append(f"arm {arm_name} has unavailable GPU telemetry samples")
            if isolation == "continuous":
                process_count = telemetry.get("max_compute_processes")
                if isinstance(process_count, int) and process_count > 1:
                    invalid.append(f"arm {arm_name} observed GPU co-tenancy")
            if launch.get("state") != "available":
                unknown.append("launch-time GPU telemetry is unavailable")
            elif int(launch.get("process_count", 1)) != 0:
                invalid.append("GPU was not isolated at launch")

        # Throughput health, independent of process count and of the launch gate.  Host
        # contention (CPU, PCIe, power) leaves max_compute_processes at 1 and still removes
        # ~20% of the optimizer steps.  Compared against this exact argv's own history, so a
        # deliberately slower configuration is judged against itself.
        reason = throughput_reason(
            arm_name,
            steps,
            arm.get("payload_argv"),
            step_baselines,
            result,
        )
        if reason:
            invalid.append(reason)

    measurements: dict[str, Any] = {"arms": arm_metrics, "effect_value": None}
    if protocol_v2:
        measurements["verified_seed"] = (
            planned_seed
            if planned_seed is not None and seed_verified_arms == set(planned_arm_names)
            else None
        )
        measurements["gpu_key"] = verified_gpu_key
    analysis = spec["analysis"]
    metric_name = spec["metric"]["name"]
    primary = arm_metrics.get(analysis["primary_arm"], {}).get(metric_name)
    if primary is not None:
        effect = analysis["effect"]
        if effect == "single":
            measurements["effect_value"] = primary
        else:
            reference = arm_metrics.get(analysis["reference_arm"], {}).get(metric_name)
            if reference is not None:
                if effect == "difference":
                    measurements["effect_value"] = primary - reference
                elif reference == 0:
                    invalid.append("cannot compute a ratio against zero")
                else:
                    measurements["effect_value"] = primary / reference

    if invalid:
        return "invalid", sorted(set(invalid + unknown)), measurements
    if unknown:
        return "unknown", sorted(set(unknown)), measurements
    return "valid", [], measurements
