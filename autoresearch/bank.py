"""Rebuildable per-GPU controls for the accelerated search lane.

The bank is deliberately *not* another scientific record.  It is a projection of
immutable ExperimentSpecs, ExecutionManifests, ResultBundles, and the latest
EvidenceDecision for each result.  A candidate may freeze the projected control
rows into its own immutable spec; scoring then resolves those rows back to their
source records and never substitutes a newer, prettier, or cross-GPU reference.

This module does not launch work and it cannot promote SOTA.  Its only authority is
to answer two operational questions:

* which clean controls are currently eligible to be frozen for a candidate; and
* whether a finished candidate cleared the cheap search gate against the exact
  same-GPU control it froze before launch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from .protocol import EVIDENCE_POLICY_VERSION
from .records import RecordError, canonical_json, sha256_object
from .store import Store

# A banked control is only a valid comparator for as long as the machine it was
# measured on has not changed underneath it.  On the shared H200 host this
# campaign runs on, that window is short and measured, not assumed:
#
#   * same-seed repeat sigma on a quiet slot .............. ~0.0001
#   * spread across seed-42 controls over a campaign ...... ~0.013  (130x)
#   * a contended window can move val_bpb by .............. ~0.02
#   * the promotion gate is ............................... 0.000426
#
# A one-hour-old control therefore carries more contention drift than the gate it
# is being compared against, which is exactly the confound that made an earlier
# campaign's fp8 "win" an artifact.  Twenty minutes with at most three uses keeps
# a control adjacent in time to the candidates it scores, at a cost of roughly one
# control slot in four -- the same control:treatment ratio a paired-wave campaign
# on this host converged to empirically.
BANK_TTL_SECONDS = 20 * 60
BANK_MAX_USES = 3
PROMOTION_GATE = 0.000426

SEARCH_LANES = {"bank", "candidate", "promotion"}


def _payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("payload", {})
    return value if isinstance(value, Mapping) else {}


def _records_by_id(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["id"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _decision_order(row: Mapping[str, Any]) -> tuple[datetime, str]:
    return (
        _parse_time(row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
        str(row.get("id", "")),
    )


def latest_decisions(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Return exactly one deterministic EvidenceDecision per ResultBundle.

    Evidence policies are versioned, so a result can legitimately have several
    decisions.  Any rebuildable search view that iterates all of them double-counts
    the result after a policy bump.  Newest ``created_at`` wins; id is a stable tie
    breaker for adversarial or hand-constructed records with identical timestamps.
    """

    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        result_id = _payload(row).get("result_id")
        if not isinstance(result_id, str) or not result_id:
            continue
        previous = latest.get(result_id)
        if previous is None or _decision_order(row) > _decision_order(previous):
            latest[result_id] = row
    return latest


def search_metadata(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _payload(spec).get("search", {})
    return value if isinstance(value, Mapping) else {}


def search_lane(spec: Mapping[str, Any]) -> str | None:
    lane = search_metadata(spec).get("lane")
    return str(lane) if lane in SEARCH_LANES else None


def bank_id(spec: Mapping[str, Any]) -> str | None:
    """The bank revision named by a search spec.

    ``baseline_id`` is accepted as a compatibility alias for early v2 drafts.
    Views always publish the canonical ``bank_id`` spelling.
    """

    search = search_metadata(spec)
    value = search.get("bank_id", search.get("baseline_id"))
    return str(value) if isinstance(value, str) and value else None


def expected_baseline_fingerprint(spec: Mapping[str, Any]) -> str | None:
    search = search_metadata(spec)
    value = search.get("baseline_fingerprint", search.get("bank_fingerprint"))
    return str(value) if isinstance(value, str) and value else None


def _scope(spec: Mapping[str, Any]) -> Any:
    # Scope has one authority: the top-level, spec-digest-bound object.  Early v2
    # drafts allowed search.scope to shadow it, which let bank matching reason about
    # a different benchmark than Evidence and Knowledge.
    return _payload(spec).get("scope")


def _bindings(
    manifest: Mapping[str, Any],
    group: str,
    *,
    excluded_paths: set[str] | None = None,
) -> list[dict[str, str]]:
    excluded_paths = excluded_paths or set()
    rows: list[dict[str, str]] = []
    raw_rows = _payload(manifest).get(group, [])
    if not isinstance(raw_rows, list):
        raise RecordError(f"manifest {group} must be a list")
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise RecordError(f"manifest {group} contains a non-object binding")
        path = raw.get("execution_path")
        digest = raw.get("sha256")
        if not isinstance(path, str) or not path or not isinstance(digest, str) or not digest:
            raise RecordError(f"manifest {group} contains an incomplete binding")
        if path in excluded_paths:
            continue
        rows.append({"execution_path": path, "sha256": digest})
    return sorted(rows, key=lambda row: (row["execution_path"], row["sha256"]))


def _control_arm(spec: Mapping[str, Any], manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    search = search_metadata(spec)
    requested = search.get("control_arm", "control")
    plan = _payload(manifest).get("plan", _payload(spec).get("plan", []))
    if not isinstance(plan, list) or not plan or not isinstance(plan[0], Mapping):
        raise RecordError("bank fingerprint requires a non-empty sealed plan")
    arms = plan[0].get("arms", [])
    if not isinstance(arms, list) or not arms:
        raise RecordError("bank fingerprint requires at least one sealed arm")
    matches = [row for row in arms if isinstance(row, Mapping) and row.get("name") == requested]
    if len(matches) == 1:
        return matches[0]
    if len(arms) == 1 and isinstance(arms[0], Mapping):
        # A bank is normally a one-arm pilot.  Accept its sole arm even when the
        # generic pilot builder called it "candidate".
        return arms[0]
    raise RecordError(f"sealed plan has no unique control arm named {requested!r}")


def baseline_fingerprint(spec: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """Identity of the exact control configuration represented by a bank row."""

    payload = _payload(spec)
    arm = _control_arm(spec, manifest)
    identity = {
        "comparison_group": payload.get("comparison_group"),
        "scope": _scope(spec),
        "metric": payload.get("metric"),
        "requirements": payload.get("requirements"),
        "control": {
            "argv": arm.get("argv"),
            "env": arm.get("env", {}),
        },
        "code_bindings": _bindings(manifest, "code_bindings"),
        "data_bindings": _bindings(manifest, "data_bindings"),
    }
    return sha256_object(identity)


def mutable_code_paths(spec: Mapping[str, Any]) -> set[str]:
    raw = search_metadata(spec).get("mutable_code_paths", [])
    if raw is None:
        return set()
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        raise RecordError("search.mutable_code_paths must contain non-empty strings")
    return set(raw)


def context_fingerprint(spec: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """Identity of everything that must remain fixed around the edited code.

    All data are always included.  Code is included unless the ExperimentSpec
    explicitly declares its execution path mutable.  The declaration is itself
    digest-bound through the spec, so the exclusion cannot be changed after a run.
    """

    payload = _payload(spec)
    excluded = mutable_code_paths(spec)
    bound_code = {row["execution_path"] for row in _bindings(manifest, "code_bindings")}
    missing = sorted(excluded - bound_code)
    if missing:
        raise RecordError(
            f"search.mutable_code_paths names code absent from the sealed manifest: {missing}"
        )
    identity = {
        "comparison_group": payload.get("comparison_group"),
        "scope": _scope(spec),
        "metric": payload.get("metric"),
        "requirements": payload.get("requirements"),
        "stable_code_bindings": _bindings(manifest, "code_bindings", excluded_paths=excluded),
        "data_bindings": _bindings(manifest, "data_bindings"),
    }
    return sha256_object(identity)


def gpu_key(result: Mapping[str, Any]) -> str | None:
    """Stable execution slot identity: resource plus physical GPU UUID.

    GPU indices are not physical identities and may be remapped.  CPU executions
    have an explicit fallback so fast unit/smoke tests can exercise the policy.
    """

    payload = _payload(result)
    resource = payload.get("resource", {})
    launch = payload.get("launch_telemetry", {})
    if not isinstance(resource, Mapping) or not isinstance(launch, Mapping):
        return None
    resource_id = resource.get("id")
    if not isinstance(resource_id, str) or not resource_id:
        return None
    gpu = resource.get("gpu")
    if gpu is None:
        return f"{resource_id}:cpu"
    uuid = launch.get("uuid", resource.get("gpu_uuid"))
    if not isinstance(uuid, str) or not uuid:
        return None
    return f"{resource_id}:{uuid}"


def verified_seed(decision: Mapping[str, Any]) -> int | None:
    measurements = _payload(decision).get("measurements", {})
    if not isinstance(measurements, Mapping):
        return None
    raw = measurements.get("verified_seed")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    number = int(raw)
    return number if float(raw) == number and number >= 0 else None


def _metric_value(
    spec: Mapping[str, Any], decision: Mapping[str, Any], *, arm_name: str | None = None
) -> float | None:
    spec_payload = _payload(spec)
    metric = spec_payload.get("metric", {})
    analysis = spec_payload.get("analysis", {})
    if not isinstance(metric, Mapping) or not isinstance(analysis, Mapping):
        return None
    metric_name = metric.get("name")
    primary = arm_name or analysis.get("primary_arm")
    measurements = _payload(decision).get("measurements", {})
    arms = measurements.get("arms", {}) if isinstance(measurements, Mapping) else {}
    values = arms.get(primary, {}) if isinstance(arms, Mapping) else {}
    value = values.get(metric_name) if isinstance(values, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _result_start(result: Mapping[str, Any]) -> datetime | None:
    lifecycle = _payload(result).get("lifecycle", {})
    return _parse_time(lifecycle.get("started_at")) if isinstance(lifecycle, Mapping) else None


def _result_end(result: Mapping[str, Any]) -> datetime | None:
    lifecycle = _payload(result).get("lifecycle", {})
    return _parse_time(lifecycle.get("ended_at")) if isinstance(lifecycle, Mapping) else None


def _reference_result_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("result_id", row.get("control_result_id"))
    return str(value) if isinstance(value, str) and value else None


class BankIndex:
    """A deterministic projection of immutable search records."""

    def __init__(
        self,
        specs: Sequence[Mapping[str, Any]],
        manifests: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
        decisions: Sequence[Mapping[str, Any]],
        *,
        now: datetime | None = None,
        store: Store | None = None,
    ) -> None:
        self.specs = _records_by_id(specs)
        self.manifests = _records_by_id(manifests)
        self.results = _records_by_id(results)
        self.decisions = latest_decisions(decisions)
        self.now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self.store = store
        self.controls = self._build_controls()
        self._controls_by_result = {row["result_id"]: row for row in self.controls}
        self._usage_events = self._build_usage_events()

    @classmethod
    def from_store(cls, store: Store, *, now: datetime | None = None) -> BankIndex:
        return cls(
            store.list("experiment_spec"),
            store.list("execution_manifest"),
            store.list("result_bundle"),
            store.list("evidence_decision"),
            now=now,
            store=store,
        )

    def _linked_records(
        self, result: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None:
        result_payload = _payload(result)
        decision = self.decisions.get(str(result.get("id", "")))
        spec = self.specs.get(str(result_payload.get("spec_id", "")))
        manifest = self.manifests.get(str(result_payload.get("manifest_id", "")))
        if decision is None or spec is None or manifest is None:
            return None
        decision_payload = _payload(decision)
        manifest_payload = _payload(manifest)
        if decision_payload.get("spec_id") != spec.get("id"):
            return None
        if manifest_payload.get("spec_id") != spec.get("id"):
            return None
        if decision_payload.get("result_id") != result.get("id"):
            return None
        if result_payload.get("manifest_id") != manifest.get("id"):
            return None
        result_digest = decision_payload.get("result_digest")
        if result_digest is not None and result_digest != result.get("digest"):
            return None
        return spec, manifest, decision

    def _build_controls(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in self.results.values():
            linked = self._linked_records(result)
            if linked is None:
                continue
            spec, manifest, decision = linked
            if search_lane(spec) != "bank":
                continue
            decision_payload = _payload(decision)
            if (
                decision_payload.get("measurement_verdict") != "valid"
                or decision_payload.get("policy_version") != EVIDENCE_POLICY_VERSION
            ):
                continue
            seed = verified_seed(decision)
            slot = gpu_key(result)
            ended = _result_end(result)
            revision = bank_id(spec)
            if seed is None or slot is None or ended is None or revision is None:
                continue
            try:
                base_fp = baseline_fingerprint(spec, manifest)
                context_fp = context_fingerprint(spec, manifest)
                arm = _control_arm(spec, manifest)
            except RecordError:
                continue
            value = _metric_value(spec, decision, arm_name=str(arm.get("name", "")))
            if value is None:
                continue
            resource = _payload(result).get("resource", {})
            launch = _payload(result).get("launch_telemetry", {})
            rows.append(
                {
                    "bank_id": revision,
                    "baseline_fingerprint": base_fp,
                    "context_fingerprint": context_fp,
                    "seed": seed,
                    "gpu_key": slot,
                    "resource_id": resource.get("id") if isinstance(resource, Mapping) else None,
                    "gpu": resource.get("gpu") if isinstance(resource, Mapping) else None,
                    "gpu_uuid": launch.get("uuid") if isinstance(launch, Mapping) else None,
                    "spec_id": spec["id"],
                    "manifest_id": manifest["id"],
                    "result_id": result["id"],
                    "evidence_id": decision["id"],
                    "metric": _payload(spec).get("metric"),
                    "value": value,
                    "ended_at": _iso(ended),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                row["ended_at"],
                row["bank_id"],
                row["gpu_key"],
                row["result_id"],
            ),
        )

    def _build_usage_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for result in self.results.values():
            spec = self.specs.get(str(_payload(result).get("spec_id", "")))
            if spec is None or search_lane(spec) != "candidate":
                continue
            started = _result_start(result)
            slot = gpu_key(result)
            if started is None or slot is None:
                continue
            refs = search_metadata(spec).get("reference_controls", [])
            if not isinstance(refs, list):
                continue
            same_slot = [
                row for row in refs if isinstance(row, Mapping) and row.get("gpu_key") == slot
            ]
            if len(same_slot) != 1:
                continue
            control_id = _reference_result_id(same_slot[0])
            if control_id not in self._controls_by_result:
                continue
            events.append(
                {
                    "control_result_id": control_id,
                    "candidate_result_id": result["id"],
                    "gpu_key": slot,
                    "started_at": _iso(started),
                }
            )
        return sorted(events, key=lambda row: (row["started_at"], row["candidate_result_id"]))

    def use_count(
        self,
        control_result_id: str,
        *,
        before: datetime | None = None,
        excluding_candidate: str | None = None,
    ) -> int:
        count = 0
        for row in self._usage_events:
            if row["control_result_id"] != control_result_id:
                continue
            if excluding_candidate and row["candidate_result_id"] == excluding_candidate:
                continue
            started = _parse_time(row["started_at"])
            if started is None:
                continue
            if before is not None and started >= before:
                continue
            count += 1
        return count

    def _control_is_eligible(
        self,
        row: Mapping[str, Any],
        *,
        at: datetime,
        excluding_candidate: str | None = None,
    ) -> bool:
        ended = _parse_time(row.get("ended_at"))
        if ended is None or ended > at:
            return False
        if (at - ended).total_seconds() > BANK_TTL_SECONDS:
            return False
        return (
            self.use_count(
                str(row["result_id"]),
                before=at,
                excluding_candidate=excluding_candidate,
            )
            < BANK_MAX_USES
        )

    def eligible_controls(
        self,
        *,
        bank_id: str,
        baseline_fingerprint: str,
        context_fingerprint: str,
        seed: int,
        at: datetime | None = None,
        gpu_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the newest eligible exact control for each physical GPU slot."""

        moment = (at or self.now).astimezone(timezone.utc)
        matching = [
            row
            for row in self.controls
            if row["bank_id"] == bank_id
            and row["baseline_fingerprint"] == baseline_fingerprint
            and row["context_fingerprint"] == context_fingerprint
            and row["seed"] == seed
            and (gpu_key is None or row["gpu_key"] == gpu_key)
            and self._control_is_eligible(row, at=moment)
        ]
        newest: dict[str, dict[str, Any]] = {}
        for row in matching:
            previous = newest.get(row["gpu_key"])
            if previous is None or (row["ended_at"], row["result_id"]) > (
                previous["ended_at"],
                previous["result_id"],
            ):
                newest[row["gpu_key"]] = row
        frozen: list[dict[str, Any]] = []
        for slot, row in sorted(newest.items()):
            ended = _parse_time(row["ended_at"])
            assert ended is not None
            uses = self.use_count(row["result_id"], before=moment)
            frozen.append(
                {
                    **row,
                    "uses": uses,
                    "remaining_uses": BANK_MAX_USES - uses,
                    "expires_at": _iso(ended + timedelta(seconds=BANK_TTL_SECONDS)),
                    "frozen_at": _iso(moment),
                    "gpu_key": slot,
                }
            )
        return frozen

    def freeze_references(
        self,
        candidate_spec: Mapping[str, Any],
        candidate_manifest: Mapping[str, Any],
        *,
        at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Build immutable ``search.reference_controls`` rows for a candidate draft."""

        revision = bank_id(candidate_spec)
        base_fp = expected_baseline_fingerprint(candidate_spec)
        plan = _payload(candidate_spec).get("plan", [])
        seed = plan[0].get("seed") if isinstance(plan, list) and plan else None
        if (
            revision is None
            or base_fp is None
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise RecordError(
                "candidate search metadata requires bank_id, baseline_fingerprint, and plan seed"
            )
        return self.eligible_controls(
            bank_id=revision,
            baseline_fingerprint=base_fp,
            context_fingerprint=context_fingerprint(candidate_spec, candidate_manifest),
            seed=seed,
            at=at,
        )

    @staticmethod
    def _unscored(result_id: str, spec_id: str | None, reason: str) -> dict[str, Any]:
        return {
            "result_id": result_id,
            "spec_id": spec_id,
            "status": "unscored",
            "reason": reason,
            "delta": None,
            "promotion_due": False,
        }

    def score_candidate(self, result_id: str) -> dict[str, Any]:
        """Score one candidate only against its frozen, same-GPU bank source."""

        result = self.results.get(result_id)
        if result is None:
            return self._unscored(result_id, None, "candidate ResultBundle does not exist")
        linked = self._linked_records(result)
        spec_id = str(_payload(result).get("spec_id", "")) or None
        if linked is None:
            return self._unscored(result_id, spec_id, "candidate record links are incomplete")
        spec, manifest, decision = linked
        if search_lane(spec) != "candidate":
            return self._unscored(result_id, spec["id"], "result is not from the candidate lane")
        if (
            _payload(decision).get("measurement_verdict") != "valid"
            or _payload(decision).get("policy_version") != EVIDENCE_POLICY_VERSION
        ):
            return self._unscored(result_id, spec["id"], "candidate evidence is not valid")
        seed = verified_seed(decision)
        slot = gpu_key(result)
        started = _result_start(result)
        if seed is None or slot is None or started is None:
            return self._unscored(
                result_id, spec["id"], "candidate lacks verified seed, physical GPU, or start time"
            )
        revision = bank_id(spec)
        expected_fp = expected_baseline_fingerprint(spec)
        if revision is None or expected_fp is None:
            return self._unscored(
                result_id, spec["id"], "candidate did not bind a bank id and fingerprint"
            )
        try:
            candidate_context = context_fingerprint(spec, manifest)
        except RecordError as exc:
            return self._unscored(result_id, spec["id"], str(exc))

        refs = search_metadata(spec).get("reference_controls", [])
        if not isinstance(refs, list) or not refs:
            return self._unscored(
                result_id,
                spec["id"],
                "candidate froze no reference_controls; dynamic fallback is forbidden",
            )
        same_slot = [row for row in refs if isinstance(row, Mapping) and row.get("gpu_key") == slot]
        if len(same_slot) != 1:
            return self._unscored(
                result_id,
                spec["id"],
                "candidate has no unique frozen reference for its physical GPU",
            )
        frozen = same_slot[0]
        control_id = _reference_result_id(frozen)
        control = self._controls_by_result.get(str(control_id))
        if control is None:
            return self._unscored(
                result_id, spec["id"], "frozen reference is not a valid bank control"
            )
        # Frozen rows are redundant on purpose.  Resolve their source record, then
        # require every supplied identity field to agree with the reconstruction.
        for key in (
            "bank_id",
            "baseline_fingerprint",
            "context_fingerprint",
            "seed",
            "gpu_key",
            "evidence_id",
        ):
            if key in frozen and frozen[key] != control[key]:
                return self._unscored(
                    result_id, spec["id"], f"frozen reference {key} disagrees with its source"
                )
        if (
            control["bank_id"] != revision
            or control["baseline_fingerprint"] != expected_fp
            or control["context_fingerprint"] != candidate_context
            or control["seed"] != seed
            or control["gpu_key"] != slot
        ):
            return self._unscored(
                result_id,
                spec["id"],
                "frozen control does not exactly match bank, context, seed, and physical GPU",
            )
        if not self._control_is_eligible(control, at=started, excluding_candidate=result_id):
            return self._unscored(
                result_id, spec["id"], "frozen control was future, stale, or overused at launch"
            )
        candidate_value = _metric_value(spec, decision)
        if candidate_value is None:
            return self._unscored(result_id, spec["id"], "candidate primary metric is absent")
        delta = candidate_value - float(control["value"])
        metric = _payload(spec).get("metric", {})
        direction = metric.get("direction") if isinstance(metric, Mapping) else None
        promotion_due = (
            delta < -PROMOTION_GATE if direction == "minimize" else delta > PROMOTION_GATE
        )
        return {
            "result_id": result_id,
            "spec_id": spec["id"],
            "status": "scored",
            "reason": "",
            "bank_id": revision,
            "baseline_fingerprint": expected_fp,
            "context_fingerprint": candidate_context,
            "seed": seed,
            "gpu_key": slot,
            "candidate_value": candidate_value,
            "control_value": float(control["value"]),
            "control_result_id": control["result_id"],
            "control_evidence_id": control["evidence_id"],
            "delta": delta,
            "gate": PROMOTION_GATE,
            "promotion_due": promotion_due,
            "sota_eligible": False,
            "started_at": _iso(started),
        }

    def bank_view(self) -> dict[str, Any]:
        controls: list[dict[str, Any]] = []
        for row in self.controls:
            ended = _parse_time(row["ended_at"])
            assert ended is not None
            uses = self.use_count(row["result_id"], before=self.now)
            future = ended > self.now
            stale = future or (self.now - ended).total_seconds() > BANK_TTL_SECONDS
            controls.append(
                {
                    **row,
                    "uses": uses,
                    "remaining_uses": max(0, BANK_MAX_USES - uses),
                    "expires_at": _iso(ended + timedelta(seconds=BANK_TTL_SECONDS)),
                    "eligible_now": not stale and uses < BANK_MAX_USES,
                }
            )
        return {
            "generated_at": _iso(self.now),
            "policy": {
                "ttl_seconds": BANK_TTL_SECONDS,
                "max_uses": BANK_MAX_USES,
                "promotion_gate": PROMOTION_GATE,
                "cross_gpu_fallback": False,
            },
            "controls": controls,
        }

    def promotion_view(self) -> dict[str, Any]:
        candidates = []
        for result in self.results.values():
            spec = self.specs.get(str(_payload(result).get("spec_id", "")))
            if spec is not None and search_lane(spec) == "candidate":
                candidates.append(self.score_candidate(str(result["id"])))
        candidates.sort(
            key=lambda row: (
                str(row.get("started_at", "")),
                str(row.get("spec_id", "")),
                str(row.get("result_id", "")),
            )
        )
        return {
            "generated_at": _iso(self.now),
            "gate": PROMOTION_GATE,
            "candidates": candidates,
            "promotion_queue": [row for row in candidates if row["promotion_due"]],
            "note": "search candidates are exploratory and can never become SOTA",
        }

    def write_views(self, store: Store | None = None) -> dict[str, Any]:
        destination = store or self.store
        if destination is None:
            raise RecordError("write_views requires a Store")
        bank = self.bank_view()
        promotions = self.promotion_view()
        destination.write_view("BANK.json", canonical_json(bank) + "\n")
        destination.write_view("PROMOTION_QUEUE.json", canonical_json(promotions) + "\n")
        return {"bank": bank, "promotions": promotions}


def rebuild_bank_views(store: Store, *, now: datetime | None = None) -> dict[str, Any]:
    """Convenience integration point for the loop/CLI."""

    return BankIndex.from_store(store, now=now).write_views()


__all__ = [
    "BANK_MAX_USES",
    "BANK_TTL_SECONDS",
    "PROMOTION_GATE",
    "BankIndex",
    "bank_id",
    "baseline_fingerprint",
    "context_fingerprint",
    "expected_baseline_fingerprint",
    "gpu_key",
    "latest_decisions",
    "mutable_code_paths",
    "rebuild_bank_views",
    "search_lane",
    "search_metadata",
    "verified_seed",
]
