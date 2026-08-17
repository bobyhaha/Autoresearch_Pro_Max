"""Sealing Authority: bind science, review, code, data, and allowed resources."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import trusted_launchers_for_resources, validate_code_entry_argv
from .records import (
    REVIEW_ROLES,
    RecordError,
    canonical_json,
    make_record,
    read_json,
    sha256_object,
)
from .store import Store

# The comparison group IS the frame. See _enforce_frame.
FRAME_SECONDS = 300


class SealingAuthority:
    def __init__(self, store: Store) -> None:
        self.store = store

    def review_template(self, spec_id: str) -> dict[str, Any]:
        spec = self.store.get("experiment_spec", spec_id)
        return {
            "spec_id": spec_id,
            "spec_digest": spec["digest"],
            "reviews": [
                {
                    "role": role,
                    "reviewer_id": f"replace_{role}_reviewer",
                    "session_id": f"replace_{role}_session",
                    "decision": "approve_or_reject",
                    "reviewed_at": "replace_with_utc_timestamp",
                    "notes": "",
                }
                for role in sorted(REVIEW_ROLES)
            ],
        }

    def seal_from_files(
        self,
        spec_id: str,
        execution_path: str | Path,
        reviews_path: str | Path | None = None,
    ) -> dict[str, Any]:
        execution_file = Path(execution_path).resolve()
        execution = read_json(execution_file)
        reviews = read_json(Path(reviews_path).resolve()) if reviews_path else None
        return self.seal(spec_id, execution, reviews, base=execution_file.parent)

    def _enforce_frame(self, spec_payload: Mapping[str, Any]) -> None:
        """Refuse to seal an explicit time budget that disagrees with the scope.

        THE FRAME IS THE EXPERIMENT.  `fixed_frame_val_bpb_v1` means one H200 for 300
        seconds; val_bpb is only meaningful relative to that budget.  A run given 600
        seconds is not a better model, it is a model given twice the compute, and its number
        is not comparable to anything else in the group.

        This is a HARD gate at seal time rather than a filter at report time, because a
        filter only prevents the bad number from being *ranked* -- it still gets measured,
        recorded, and read.  On 2026-08-12 a 600 s arm scored 0.926958 against a 300 s SOTA
        of 0.962288 and was surfaced as the leaderboard's "running best": a number that read
        as a 3.7 percent breakthrough and was purely the extra budget.  It was believed for
        several hours.

        The specs that produced those runs were sealed by me with
        `comparison_group = fixed_frame_val_bpb_v1` while overriding `--time-budget`, so the
        group's own definition was violated by the experiment design rather than by the
        harness.  Nothing in the pipeline objected.  Now it does.
        """
        if spec_payload.get("protocol_version") == 2:
            scope = spec_payload.get("scope")
            if not isinstance(scope, Mapping):
                raise RecordError("protocol v2 requires structured scope with a timing budget")
            budget = scope.get("budget")
            budget_kind = budget.get("kind") if isinstance(budget, Mapping) else None
            if budget_kind not in {"wall_seconds", "training_seconds"}:
                raise RecordError(
                    "protocol v2 scope.budget.kind must be wall_seconds or training_seconds"
                )
            frame_seconds = budget.get("value")
            if (
                isinstance(frame_seconds, bool)
                or not isinstance(frame_seconds, (int, float))
                or not math.isfinite(float(frame_seconds))
                or float(frame_seconds) <= 0
            ):
                raise RecordError("protocol v2 scope.budget.value must be a positive finite number")
            expected = float(frame_seconds)
            for row in spec_payload.get("plan", []) or []:
                for arm in row.get("arms", []) or []:
                    argv = [str(token) for token in arm.get("argv", []) or []]
                    index = 0
                    while index < len(argv):
                        token = argv[index]
                        raw_value: str | None = None
                        if token.startswith("--time-budget="):
                            raw_value = token.split("=", 1)[1]
                        elif token == "--time-budget":
                            if index + 1 >= len(argv):
                                raise RecordError(
                                    f"arm {arm.get('name')} has --time-budget without a value"
                                )
                            raw_value = argv[index + 1]
                            index += 1
                        if raw_value is not None:
                            try:
                                requested = float(raw_value)
                            except ValueError as exc:
                                raise RecordError(
                                    f"arm {arm.get('name')} has an unparsable time budget "
                                    f"{raw_value!r}"
                                ) from exc
                            if not math.isfinite(requested) or requested != expected:
                                raise RecordError(
                                    f"arm {arm.get('name')} requests a {raw_value}s frame, "
                                    f"which conflicts with scope.budget.value={frame_seconds} "
                                    f"({budget_kind})"
                                )
                        index += 1
            return

        for row in spec_payload.get("plan", []) or []:
            for arm in row.get("arms", []) or []:
                for token in arm.get("argv", []) or []:
                    token = str(token)
                    if not token.startswith("--time-budget="):
                        continue
                    try:
                        seconds = int(token.split("=", 1)[1])
                    except ValueError as exc:
                        raise RecordError(
                            f"arm {arm.get('name')} has an unparsable {token}; the "
                            "comparison group requires a 300 second frame"
                        ) from exc
                    if seconds != FRAME_SECONDS:
                        raise RecordError(
                            f"arm {arm.get('name')} requests a {seconds}s frame; "
                            f"{spec_payload.get('comparison_group')} is DEFINED by a "
                            f"{FRAME_SECONDS}s budget and off-frame runs are not "
                            "comparable. Seal it under a different comparison group."
                        )

    def seal(
        self,
        spec_id: str,
        execution: Mapping[str, Any],
        reviews: Mapping[str, Any] | None = None,
        *,
        base: Path | None = None,
    ) -> dict[str, Any]:
        spec = self.store.get("experiment_spec", spec_id)
        spec_payload = spec["payload"]
        self._enforce_frame(spec_payload)
        self._enforce_paper_gate(spec)
        review_rows = self._validate_reviews(spec, reviews)
        runtime = dict(execution.get("runtime", {}))
        timeout = float(runtime.get("timeout_seconds_per_arm", 900))
        interval = float(runtime.get("telemetry_interval_seconds", 2))
        wait_seconds = float(runtime.get("resource_wait_seconds", 30))
        if timeout <= 0 or interval <= 0 or wait_seconds < 0:
            raise RecordError("runtime durations must be positive")
        runtime = {
            "timeout_seconds_per_arm": timeout,
            "telemetry_interval_seconds": interval,
            "resource_wait_seconds": wait_seconds,
        }

        code_bindings = self._seal_bindings(execution.get("code_bindings", []), base)
        data_bindings = self._seal_bindings(execution.get("data_bindings", []), base)
        if not code_bindings:
            raise RecordError("at least one code binding is required")
        if not data_bindings:
            raise RecordError("at least one data binding is required")
        execution_paths = [row["execution_path"] for row in code_bindings + data_bindings]
        if len(execution_paths) != len(set(execution_paths)):
            raise RecordError("every code/data binding must have a unique execution_path")
        search = spec_payload.get("search", {})
        mutable_code_paths = list(search.get("mutable_code_paths", [])) if search else []
        sealed_code_paths = {row["execution_path"] for row in code_bindings}
        missing_mutable_paths = sorted(set(mutable_code_paths) - sealed_code_paths)
        if missing_mutable_paths:
            raise RecordError(
                "every search.mutable_code_paths entry must match a sealed code binding; "
                f"missing {missing_mutable_paths}"
            )
        resources = self._validate_resources(execution.get("resources"), base=base)
        if spec_payload.get("protocol_version") == 2:
            trusted_launchers = trusted_launchers_for_resources(resources)
            for replicate in spec_payload.get("plan", []):
                for arm in replicate.get("arms", []):
                    try:
                        validate_code_entry_argv(
                            arm.get("argv", []),
                            sorted(sealed_code_paths),
                            trusted_launchers=trusted_launchers,
                        )
                    except RecordError as exc:
                        raise RecordError(f"arm {arm.get('name')}: {exc}") from exc
        if spec_payload["requirements"].get("require_gpu", True) and not any(
            resource.get("gpus") for resource in resources
        ):
            raise RecordError("the ExperimentSpec requires a GPU but no resource authorizes one")

        payload = {
            "spec_id": spec_id,
            "spec_digest": spec["digest"],
            "stage": spec_payload["stage"],
            "plan": spec_payload["plan"],
            "requirements": spec_payload["requirements"],
            "metric": spec_payload["metric"],
            "analysis": spec_payload["analysis"],
            "comparison_group": spec_payload["comparison_group"],
            "reviews": review_rows,
            "code_bindings": code_bindings,
            "data_bindings": data_bindings,
            "resources": resources,
            "runtime": runtime,
        }
        for optional_field in ("protocol_version", "search", "scope"):
            if optional_field in spec_payload:
                payload[optional_field] = spec_payload[optional_field]
        semantic_digest = sha256_object(payload)
        manifest_id = f"manifest_{semantic_digest[:24]}"

        def commit() -> dict[str, Any]:
            path = self.store.record_path("execution_manifest", manifest_id)
            if path.exists():
                existing = self.store.get("execution_manifest", manifest_id)
                if existing["payload"] == payload:
                    return existing
                raise RecordError(f"manifest id collision: {manifest_id}")
            return self.store.put(make_record("execution_manifest", manifest_id, payload))

        if spec_payload.get("protocol_version") != 2:
            return commit()

        # A v2 plan is one immutable scientific object, not a menu whose seeds may
        # quietly run against different code snapshots.  This lock makes the check
        # atomic across concurrent sealers while retaining idempotent re-sealing of
        # the byte-identical manifest.
        with self.store.lock(f"v2_manifest_{spec_id}"):
            existing_for_spec = [
                row
                for row in self.store.list("execution_manifest")
                if row["payload"].get("spec_id") == spec_id
            ]
            if existing_for_spec:
                if (
                    len(existing_for_spec) == 1
                    and existing_for_spec[0]["id"] == manifest_id
                    and existing_for_spec[0]["payload"] == payload
                ):
                    return existing_for_spec[0]
                raise RecordError(
                    "protocol v2 permits exactly one immutable manifest per "
                    f"ExperimentSpec; {spec_id} is already sealed"
                )
            return commit()

    def _seal_bindings(self, rows: Any, base: Path | None) -> list[dict[str, Any]]:
        if not isinstance(rows, list):
            raise RecordError("bindings must be lists")
        result: list[dict[str, Any]] = []
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                raise RecordError(f"binding {index} must be an object")
            source = Path(str(raw.get("source", ""))).expanduser()
            if base is not None and not source.is_absolute():
                source = base / source
            execution_path = str(raw.get("execution_path", "")).strip()
            execution_parts = Path(execution_path).parts
            if not execution_path or Path(execution_path).is_absolute() or ".." in execution_parts:
                raise RecordError("binding execution_path must be a non-empty relative path")
            digest, blob = self.store.add_blob(source)
            source_name = str(raw.get("source_name", source.name)).strip()
            if not source_name:
                raise RecordError("binding source_name must be non-empty when supplied")
            result.append(
                {
                    "source_name": source_name,
                    "execution_path": execution_path,
                    "sha256": digest,
                    "blob": blob,
                }
            )
        return result

    def _validate_reviews(
        self, spec: Mapping[str, Any], declaration: Mapping[str, Any] | None
    ) -> list[dict[str, Any]]:
        stage = spec["payload"]["stage"]
        protocol_v2_confirmation = (
            stage == "confirmation" and spec["payload"].get("protocol_version") == 2
        )
        if declaration is None:
            if stage == "pilot" or protocol_v2_confirmation:
                return []
            raise RecordError("confirmation sealing requires an independent review declaration")
        if declaration.get("spec_id") != spec["id"]:
            raise RecordError("review declaration names a different spec")
        if declaration.get("spec_digest") != spec["digest"]:
            raise RecordError("review declaration is stale for this ExperimentSpec")
        rows = declaration.get("reviews")
        if not isinstance(rows, list):
            raise RecordError("reviews must be a list")
        normalized: list[dict[str, Any]] = []
        roles: set[str] = set()
        reviewers: set[str] = set()
        sessions: set[str] = set()
        spec_time = _parse_time(str(spec["created_at"]), "spec.created_at")
        now = datetime.now(timezone.utc)
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise RecordError("each review must be an object")
            role = str(raw.get("role", "")).strip().lower()
            reviewer = str(raw.get("reviewer_id", "")).strip().lower()
            session = str(raw.get("session_id", "")).strip().lower()
            decision = str(raw.get("decision", "")).strip().lower()
            reviewed_at = str(raw.get("reviewed_at", "")).strip()
            if role not in REVIEW_ROLES:
                raise RecordError(f"unknown review role: {role}")
            if not reviewer or not session or not reviewed_at:
                raise RecordError("reviewer_id, session_id, and reviewed_at are required")
            review_time = _parse_time(reviewed_at, f"review {role}.reviewed_at")
            if review_time < spec_time:
                raise RecordError(f"review role {role} predates the ExperimentSpec")
            if review_time > now:
                raise RecordError(f"review role {role} is future-dated")
            if decision != "approve":
                raise RecordError(f"review role {role} did not approve")
            if role in roles or reviewer in reviewers or session in sessions:
                raise RecordError("review roles, reviewers, and sessions must be independent")
            roles.add(role)
            reviewers.add(reviewer)
            sessions.add(session)
            normalized.append(
                {
                    "role": role,
                    "reviewer_id": reviewer,
                    "session_id": session,
                    "decision": decision,
                    "reviewed_at": reviewed_at,
                    "notes": str(raw.get("notes", "")),
                }
            )
        if stage == "confirmation" and not protocol_v2_confirmation and roles != REVIEW_ROLES:
            missing = sorted(REVIEW_ROLES - roles)
            raise RecordError(f"confirmation is missing review roles: {missing}")
        return sorted(normalized, key=lambda row: row["role"])

    def _enforce_paper_gate(self, candidate: Mapping[str, Any]) -> None:
        """Block the sixth unpublished completed confirmation round."""
        if candidate["payload"]["stage"] != "confirmation":
            return
        if candidate["payload"].get("protocol_version") == 2:
            return
        covered = {
            spec_id
            for paper in self.store.list("paper")
            for spec_id in paper["payload"].get("spec_ids", [])
        }
        valid_by_spec: dict[str, set[str]] = {}
        for decision in self.store.list("evidence_decision"):
            payload = decision["payload"]
            if payload["measurement_verdict"] == "valid" and payload["claim_status"] == "eligible":
                valid_by_spec.setdefault(payload["spec_id"], set()).add(payload["result_id"])
        completed: set[str] = set()
        for spec in self.store.list("experiment_spec"):
            if spec["payload"]["stage"] != "confirmation":
                continue
            required = int(spec["payload"]["analysis"]["minimum_valid_replicates"])
            if len(valid_by_spec.get(spec["id"], set())) >= required:
                completed.add(spec["id"])
        unpublished = completed - covered
        if len(unpublished) >= 5 and candidate["id"] not in completed:
            raise RecordError(
                "paper gate: five completed confirmation rounds remain unpublished; "
                f"register a paper covering {sorted(unpublished)} before sealing a new round"
            )

    @staticmethod
    def _validate_resources(value: Any, *, base: Path | None) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise RecordError("execution resources must be a non-empty list")
        result: list[dict[str, Any]] = []
        ids: set[str] = set()
        for raw in value:
            if not isinstance(raw, Mapping):
                raise RecordError("each resource must be an object")
            row = json.loads(canonical_json(dict(raw)))
            resource_id = str(row.get("id", "")).strip()
            backend = str(row.get("backend", "")).strip()
            workdir = str(row.get("workdir", "")).strip()
            if not resource_id or resource_id in ids:
                raise RecordError("resource ids must be non-empty and unique")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", resource_id):
                raise RecordError(
                    "resource ids may contain only letters, digits, dot, dash, underscore"
                )
            if "host_id" in row:
                host_id = str(row.get("host_id", "")).strip()
                if not host_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", host_id):
                    raise RecordError(
                        "resource host_id may contain only letters, digits, dot, dash, underscore"
                    )
                row["host_id"] = host_id
            if "hardware_class" in row:
                hardware_class = str(row.get("hardware_class", "")).strip()
                if not hardware_class:
                    raise RecordError("resource hardware_class must be non-empty text")
                row["hardware_class"] = hardware_class
            if backend not in {"local", "ssh"}:
                raise RecordError("resource backend must be local or ssh")
            # Launcher paths are part of the resource runtime trust boundary. They
            # must be explicit paths; a bare name would reintroduce PATH substitution.
            trusted_launchers_for_resources([row])
            if not workdir:
                raise RecordError("resource workdir is required")
            if backend == "local":
                local_workdir = Path(workdir).expanduser()
                if base is not None and not local_workdir.is_absolute():
                    local_workdir = base / local_workdir
                row["workdir"] = str(local_workdir.resolve())
                if not Path(row["workdir"]).is_dir():
                    raise RecordError(f"local resource workdir does not exist: {row['workdir']}")
            elif not workdir.startswith("/"):
                raise RecordError("ssh resource workdir must be an absolute POSIX path")
            gpus = row.get("gpus", [])
            if (
                not isinstance(gpus, list)
                or not all(
                    isinstance(item, int) and not isinstance(item, bool) and item >= 0
                    for item in gpus
                )
                or len(gpus) != len(set(gpus))
            ):
                raise RecordError("resource gpus must be unique non-negative integer indices")
            if "max_concurrent_jobs" in row:
                max_jobs = row["max_concurrent_jobs"]
                if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1:
                    raise RecordError("max_concurrent_jobs must be a positive integer")
            if "reservation" in row:
                reservation = row["reservation"]
                if not isinstance(reservation, Mapping):
                    raise RecordError("resource reservation must be an object")
                mode = str(reservation.get("mode", "")).strip()
                if mode not in {"shared", "externally_reserved"}:
                    raise RecordError(
                        "resource reservation mode must be shared or externally_reserved"
                    )
                reservation_id = reservation.get("id")
                if reservation_id is not None and (
                    not isinstance(reservation_id, str) or not reservation_id.strip()
                ):
                    raise RecordError("resource reservation id must be non-empty when present")
                if mode == "externally_reserved" and not reservation_id:
                    raise RecordError("externally_reserved resources require a reservation id")
                row["reservation"] = {
                    "mode": mode,
                    **({"id": reservation_id.strip()} if reservation_id is not None else {}),
                }
            memory_limit = row.get("max_idle_memory_mb", 512)
            utilization_limit = row.get("max_idle_utilization_percent", 5)
            if (
                isinstance(memory_limit, bool)
                or not isinstance(memory_limit, (int, float))
                or float(memory_limit) < 0
            ):
                raise RecordError("max_idle_memory_mb must be a non-negative number")
            if (
                isinstance(utilization_limit, bool)
                or not isinstance(utilization_limit, (int, float))
                or not 0 <= float(utilization_limit) <= 100
            ):
                raise RecordError("max_idle_utilization_percent must be between 0 and 100")
            if backend == "ssh":
                ssh_argv = row.get("ssh_argv")
                if not isinstance(ssh_argv, list) or not ssh_argv:
                    raise RecordError("ssh resources require ssh_argv")
                if not all(isinstance(item, str) and item for item in ssh_argv):
                    raise RecordError("ssh_argv must contain strings")
            ids.add(resource_id)
            result.append(row)
        physical_slots: set[tuple[str, int]] = set()
        for row in result:
            host_id = str(row.get("host_id", row["id"]))
            for gpu in row.get("gpus", []):
                slot = (host_id, int(gpu))
                if slot in physical_slots:
                    raise RecordError(
                        f"GPU {gpu} on host_id {host_id} is authorized by multiple resources"
                    )
                physical_slots.add(slot)
        return result


def _parse_time(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecordError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise RecordError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)
