"""Durable campaign queue built on the immutable execution and evidence services.

The queue is operational state, not scientific evidence.  Each manifest has one
deterministic job file whose state is atomically replaced under a store lock.  ResultBundle
and EvidenceDecision records remain the authority for what ran and whether it was valid.

This module deliberately owns the long-running loop.  A campaign should not need a shell
``while`` loop, a pid-file convention, or an agent between one completed manifest and the
next launch.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .evidence import EvidenceEngine
from .execution import (
    ExecutionService,
    NoPendingReplicate,
    NoResourceAvailable,
)
from .records import read_json
from .store import Store

QUEUE_SCHEMA_VERSION = 1
QUEUE_STATES = frozenset({"pending", "running", "complete", "waiting", "blocked"})
HEALTH_WINDOW = 12
HEALTH_MIN_RATE_SAMPLE = 8
HEALTH_MAX_INVALID_RATE = 0.25
HEALTH_CONSECUTIVE_INVALID = 3


class CampaignError(RuntimeError):
    """Base class for durable campaign orchestration failures."""


class CampaignStateError(CampaignError):
    """The operational queue violates its state-machine invariants."""


def job_id_for_manifest(manifest_id: str) -> str:
    """Return a stable filesystem-safe job id for one immutable manifest id."""
    digest = hashlib.sha256(manifest_id.encode("utf-8")).hexdigest()[:24]
    return f"job_{digest}"


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise CampaignStateError("campaign clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise CampaignStateError(f"invalid queue timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise CampaignStateError(f"queue timestamp lacks timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class CampaignQueue:
    """Atomic manifest queue with resident workers and evidence-aware recovery.

    ``execution`` and ``evidence`` are injectable so fault handling can be tested without
    pretending an uncertain execution succeeded.  Production callers normally omit them.
    """

    def __init__(
        self,
        store: Store,
        *,
        execution: ExecutionService | None = None,
        evidence: EvidenceEngine | None = None,
        now: Callable[[], datetime] = _default_now,
        sleep: Callable[[float], None] = time.sleep,
        pid_alive: Callable[[int], bool] = _pid_alive,
        no_resource_backoff_seconds: float = 15.0,
        max_no_resource_backoff_seconds: float = 300.0,
    ) -> None:
        if no_resource_backoff_seconds < 0 or max_no_resource_backoff_seconds < 0:
            raise ValueError("NoResource backoff durations must be non-negative")
        self.store = store
        self.execution = execution or ExecutionService(store)
        self.evidence = evidence or EvidenceEngine(store)
        self._now = now
        self._sleep = sleep
        self._pid_alive = pid_alive
        self.no_resource_backoff_seconds = float(no_resource_backoff_seconds)
        self.max_no_resource_backoff_seconds = float(max_no_resource_backoff_seconds)
        self.store.init()
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    @property
    def queue_dir(self) -> Path:
        return self.store.root / "operational" / "queue"

    def job_path(self, job_id: str) -> Path:
        return self.queue_dir / f"{job_id}.json"

    def enqueue(self, manifest_id: str) -> dict[str, Any]:
        """Idempotently queue an existing immutable ExecutionManifest."""
        manifest = self.store.get("execution_manifest", manifest_id)
        job_id = job_id_for_manifest(manifest_id)
        path = self.job_path(job_id)
        with self.store.lock("campaign_queue"):
            if path.exists():
                existing = self._read_job(path)
                if (
                    existing["manifest_id"] != manifest_id
                    or existing["manifest_digest"] != manifest["digest"]
                ):
                    raise CampaignStateError(f"deterministic queue id collision at {path}")
                return existing
            timestamp = _iso(self._now())
            total = len(manifest["payload"]["plan"])
            job = {
                "schema_version": QUEUE_SCHEMA_VERSION,
                "job_id": job_id,
                "manifest_id": manifest_id,
                "manifest_digest": manifest["digest"],
                "spec_id": manifest["payload"]["spec_id"],
                "stage": manifest["payload"]["stage"],
                "state": "pending",
                "created_at": timestamp,
                "updated_at": timestamp,
                "last_progress_at": timestamp,
                "run_attempts": 0,
                "no_resource_count": 0,
                "next_eligible_at": None,
                "owner_pid": None,
                "owner_thread": None,
                "lease_token": None,
                "last_error": None,
                "progress": {"total": total, "completed": 0, "judged": 0},
                "transitions": [
                    {"from": None, "to": "pending", "at": timestamp, "reason": "enqueued"}
                ],
            }
            self._write_job(job)
            return job

    def jobs(self) -> list[dict[str, Any]]:
        """Return a consistent snapshot of every operational queue job."""
        with self.store.lock("campaign_queue"):
            rows = [self._read_job(path) for path in sorted(self.queue_dir.glob("*.json"))]
        return sorted(rows, key=lambda row: (row["created_at"], row["job_id"]))

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.store.lock("campaign_queue"):
            path = self.job_path(job_id)
            if not path.exists():
                raise CampaignStateError(f"missing campaign job: {job_id}")
            return self._read_job(path)

    def health(self) -> dict[str, Any]:
        """Derive the rolling pilot-health circuit entirely from immutable evidence."""
        results = {row["id"]: row for row in self.store.list("result_bundle")}
        latest: dict[str, dict[str, Any]] = {}
        for decision in self.store.list("evidence_decision"):
            payload = decision["payload"]
            result = results.get(payload["result_id"])
            if result is None or result["payload"].get("stage") != "pilot":
                continue
            previous = latest.get(payload["result_id"])
            if previous is None or (decision["created_at"], decision["id"]) > (
                previous["created_at"],
                previous["id"],
            ):
                latest[payload["result_id"]] = decision

        ordered = sorted(
            latest.values(),
            key=lambda row: (
                results[row["payload"]["result_id"]]["created_at"],
                row["payload"]["result_id"],
            ),
        )[-HEALTH_WINDOW:]
        verdicts = [row["payload"]["measurement_verdict"] for row in ordered]
        invalid_only = sum(verdict == "invalid" for verdict in verdicts)
        unknown_count = sum(verdict == "unknown" for verdict in verdicts)
        # Unknown telemetry/provenance is just as unusable as invalid evidence and
        # should not let a broken campaign consume the rest of its budget.
        invalid_count = invalid_only + unknown_count
        consecutive = 0
        for verdict in reversed(verdicts):
            if verdict == "valid":
                break
            consecutive += 1
        rate = invalid_count / len(verdicts) if verdicts else 0.0
        reasons: list[str] = []
        if consecutive >= HEALTH_CONSECUTIVE_INVALID:
            reasons.append(
                f"{consecutive} consecutive pilot measurements are invalid or unknown "
                f"(limit {HEALTH_CONSECUTIVE_INVALID})"
            )
        if len(verdicts) >= HEALTH_MIN_RATE_SAMPLE and rate > HEALTH_MAX_INVALID_RATE:
            reasons.append(
                f"pilot invalid rate (including unknown) is {rate:.1%} over the last "
                f"{len(verdicts)} "
                f"(limit {HEALTH_MAX_INVALID_RATE:.0%})"
            )
        return {
            "state": "paused" if reasons else "healthy",
            "paused": bool(reasons),
            "window_size": len(verdicts),
            "invalid_count": invalid_count,
            "invalid_only_count": invalid_only,
            "unknown_count": unknown_count,
            "invalid_rate": rate,
            "consecutive_invalid": consecutive,
            "reasons": reasons,
            "result_ids": [row["payload"]["result_id"] for row in ordered],
        }

    def status(self) -> dict[str, Any]:
        rows = self.jobs()
        counts = {state: 0 for state in sorted(QUEUE_STATES)}
        for row in rows:
            counts[row["state"]] += 1
        last_progress = max(
            (str(row["last_progress_at"]) for row in rows if row.get("last_progress_at")),
            default=None,
        )
        return {
            "queue_depth": counts["pending"] + counts["waiting"],
            "active": counts["running"],
            "blocked": counts["blocked"],
            "complete": counts["complete"],
            "states": counts,
            "health": self.health(),
            "last_progress": last_progress,
        }

    def work(
        self,
        *,
        workers: int = 1,
        follow: bool = False,
        poll_seconds: float = 1.0,
        idle_timeout_seconds: float | None = None,
        ignore_health: bool = False,
    ) -> dict[str, Any]:
        """Drain ready jobs with resident threads.

        With ``follow=True`` workers remain available for jobs enqueued by another process.
        ``idle_timeout_seconds=None`` follows indefinitely.  The rolling health circuit stops
        new claims unless an operator explicitly supplies ``ignore_health=True``; already
        running scientific work is never killed by the queue.

        Unexpected runner/evidence failures transition the owning job to ``blocked`` and are
        re-raised after the worker pool settles.  They are never converted into retries.
        """
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be > 0")
        if idle_timeout_seconds is not None and idle_timeout_seconds < 0:
            raise ValueError("idle_timeout_seconds must be non-negative")

        self.reconcile()
        stop = threading.Event()
        activity_lock = threading.Lock()
        last_activity = [time.monotonic()]
        completed: list[str] = []
        waiting: list[str] = []
        activity_errors: list[BaseException] = []

        def touched() -> None:
            with activity_lock:
                last_activity[0] = time.monotonic()

        def idle_expired() -> bool:
            if idle_timeout_seconds is None:
                return False
            with activity_lock:
                return time.monotonic() - last_activity[0] >= idle_timeout_seconds

        def worker() -> None:
            while not stop.is_set():
                job = self._claim_next_job(ignore_health=ignore_health)
                if job is not None:
                    touched()
                    try:
                        state = self._process_job(job)
                    except BaseException as exc:  # noqa: BLE001 - worker faults must stop the pool
                        stop.set()
                        activity_errors.append(exc)
                        return
                    if state == "complete":
                        completed.append(job["job_id"])
                    elif state == "waiting":
                        waiting.append(job["job_id"])
                    touched()
                    continue
                if not ignore_health and self.health()["paused"]:
                    stop.set()
                    return
                if not follow or idle_expired():
                    return
                self._sleep(poll_seconds)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker) for _ in range(workers)]
            for future in as_completed(futures):
                future.result()

        if activity_errors:
            raise activity_errors[0]
        snapshot = self.status()
        return {
            "completed_job_ids": sorted(set(completed)),
            "waiting_job_ids": sorted(set(waiting)),
            "status": snapshot,
        }

    def reconcile(self) -> list[dict[str, Any]]:
        """Recover queue state without guessing that uncertain execution is safe.

        A dead queue owner with no claim and no artifacts is safe to requeue: execution never
        crossed its durable boundary.  An inflight claim or artifacts without a ResultBundle
        means the scientific replicate may have run, so the job is blocked and the evidence is
        left intact for an operator.  A live local owner/runner PID leaves the job running.
        """
        reconciled: list[dict[str, Any]] = []
        for job in self.jobs():
            reconciled.append(self._reconcile_job(job["job_id"]))
        return reconciled

    def _claim_next_job(self, *, ignore_health: bool) -> dict[str, Any] | None:
        if not ignore_health and self.health()["paused"]:
            return None
        with self.store.lock("campaign_queue"):
            rows = [self._read_job(path) for path in sorted(self.queue_dir.glob("*.json"))]
            now = self._now()
            eligible = []
            for row in rows:
                if row["state"] == "pending":
                    eligible.append(row)
                elif row["state"] == "waiting":
                    deadline = row.get("next_eligible_at")
                    if deadline is None or _parse_time(str(deadline)) <= now:
                        eligible.append(row)
            if not eligible:
                return None
            job = min(eligible, key=lambda row: (row["created_at"], row["job_id"]))
            token = uuid.uuid4().hex
            return self._transition_locked(
                job,
                "running",
                reason="worker_claimed",
                updates={
                    "owner_pid": os.getpid(),
                    "owner_thread": threading.get_ident(),
                    "lease_token": token,
                    "run_attempts": int(job.get("run_attempts", 0)) + 1,
                    "next_eligible_at": None,
                    "last_error": None,
                },
            )

    def _process_job(self, claimed: Mapping[str, Any]) -> str:
        job_id = str(claimed["job_id"])
        lease_token = str(claimed["lease_token"])
        manifest_id = str(claimed["manifest_id"])
        try:
            while True:
                if self._complete_if_finished(job_id, lease_token=lease_token):
                    return "complete"
                try:
                    result = self.execution.execute_next(manifest_id)
                except NoResourceAvailable as exc:
                    current = self.get_job(job_id)
                    count = int(current.get("no_resource_count", 0)) + 1
                    delay = min(
                        self.max_no_resource_backoff_seconds,
                        self.no_resource_backoff_seconds * (2 ** min(count - 1, 30)),
                    )
                    deadline = self._now() + timedelta(seconds=delay)
                    self._transition_owned(
                        job_id,
                        lease_token,
                        "waiting",
                        reason="no_resource",
                        updates={
                            "no_resource_count": count,
                            "next_eligible_at": _iso(deadline),
                            "last_error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    return "waiting"
                except NoPendingReplicate:
                    row = self._reconcile_job(job_id)
                    if row["state"] == "complete":
                        return "complete"
                    if row["state"] == "blocked":
                        raise CampaignError(
                            f"{job_id} has no claimable replicate and reconciliation "
                            f"blocked it: {row.get('last_error') or 'uncertain execution'}"
                        )
                    raise CampaignStateError(
                        f"{job_id} has no claimable replicate but reconciled to "
                        f"{row['state']}; refusing to guess or retry"
                    )

                # Judging is part of landing a result, not a later reporting pass.
                self.evidence.judge(result["id"])
                self._record_progress(job_id, lease_token)
        except BaseException as exc:  # noqa: BLE001 - preserve and surface uncertain execution
            # No retry: ExecutionService preserves an uncertain inflight claim, and this
            # queue preserves the error and surfaces it to the caller.
            try:
                current = self.get_job(job_id)
                if current["state"] == "running" and current.get("lease_token") == lease_token:
                    self._transition_owned(
                        job_id,
                        lease_token,
                        "blocked",
                        reason="uncertain_failure",
                        updates={"last_error": f"{type(exc).__name__}: {exc}"},
                    )
            finally:
                raise

    def _complete_if_finished(self, job_id: str, *, lease_token: str | None = None) -> bool:
        job = self.get_job(job_id)
        progress = self._manifest_progress(job)
        if progress["completed"] != progress["total"]:
            return False
        for result_id in progress["result_ids"]:
            self.evidence.judge(result_id)
        progress = self._manifest_progress(job)
        updates = {"progress": self._public_progress(progress), "last_error": None}
        if lease_token is not None:
            self._transition_owned(
                job_id,
                lease_token,
                "complete",
                reason="all_replicates_judged",
                updates=updates,
            )
        else:
            self._transition(
                job_id,
                "complete",
                reason="reconciled_complete",
                updates=updates,
            )
        return True

    def _record_progress(self, job_id: str, lease_token: str) -> dict[str, Any]:
        with self.store.lock("campaign_queue"):
            job = self._read_job(self.job_path(job_id))
            self._assert_owner(job, lease_token)
            progress = self._manifest_progress(job)
            timestamp = _iso(self._now())
            job["progress"] = self._public_progress(progress)
            job["updated_at"] = timestamp
            job["last_progress_at"] = timestamp
            self._write_job(job)
            return job

    def _reconcile_job(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        progress = self._manifest_progress(job)
        if progress["completed"] == progress["total"]:
            try:
                for result_id in progress["result_ids"]:
                    self.evidence.judge(result_id)
            except BaseException as exc:
                if job["state"] != "complete":
                    self._transition(
                        job_id,
                        "blocked",
                        reason="evidence_reconciliation_failed",
                        updates={"last_error": f"{type(exc).__name__}: {exc}"},
                    )
                raise
            progress = self._manifest_progress(job)
            if job["state"] != "complete":
                return self._transition(
                    job_id,
                    "complete",
                    reason="reconciled_complete",
                    updates={"progress": self._public_progress(progress), "last_error": None},
                )
            return job

        claims = self._claims_for_manifest(str(job["manifest_id"]))
        uncertain = [row for row in claims if "uncertain" in str(row.get("state", "")).lower()]
        if uncertain:
            if job["state"] != "blocked":
                return self._transition(
                    job_id,
                    "blocked",
                    reason="uncertain_inflight_claim",
                    updates={
                        "last_error": "an inflight claim records an uncertain execution failure"
                    },
                )
            return job

        if claims:
            if any(self._claim_has_live_local_pid(row) for row in claims):
                # It may be a worker in another process. Never steal its scientific slot.
                if job["state"] != "running":
                    return self._transition(
                        job_id,
                        "running",
                        reason="reconciled_live_claim",
                        updates={"last_error": None},
                    )
                return job
            if job["state"] != "blocked":
                return self._transition(
                    job_id,
                    "blocked",
                    reason="orphaned_inflight_claim",
                    updates={
                        "last_error": "inflight claim exists but no recorded local PID is live"
                    },
                )
            return job

        if self._missing_replicate_has_artifacts(job, progress["missing_replicates"]):
            if job["state"] != "blocked":
                return self._transition(
                    job_id,
                    "blocked",
                    reason="artifacts_without_result",
                    updates={"last_error": "execution artifacts exist without a ResultBundle"},
                )
            return job

        if job["state"] in {"running", "blocked"}:
            owner = job.get("owner_pid")
            if job["state"] == "running" and isinstance(owner, int) and self._pid_alive(owner):
                return job
            return self._transition(
                job_id,
                "pending",
                reason=(
                    "operator_released_unstarted_job"
                    if job["state"] == "blocked"
                    else "recovered_unstarted_job"
                ),
                updates={"last_error": None},
            )
        return job

    def _manifest_progress(self, job: Mapping[str, Any]) -> dict[str, Any]:
        manifest = self.store.get("execution_manifest", str(job["manifest_id"]))
        if manifest["digest"] != job["manifest_digest"]:
            raise CampaignStateError(f"{job['job_id']} manifest digest changed")
        spec_id = manifest["payload"]["spec_id"]
        planned = [str(row["replicate_id"]) for row in manifest["payload"]["plan"]]
        planned_set = set(planned)
        by_replicate: dict[str, dict[str, Any]] = {}
        for result in self.store.list("result_bundle"):
            payload = result["payload"]
            replicate_id = str(payload["replicate_id"])
            if (
                payload["spec_id"] != spec_id
                or payload.get("manifest_id") != manifest["id"]
                or payload.get("manifest_digest") != manifest["digest"]
                or replicate_id not in planned_set
            ):
                continue
            if replicate_id in by_replicate:
                raise CampaignStateError(
                    f"scientific replicate {(spec_id, replicate_id)} has multiple results"
                )
            by_replicate[replicate_id] = result
        decisions = {
            row["payload"]["result_id"]
            for row in self.store.list("evidence_decision")
            if row["payload"]["result_id"] in {item["id"] for item in by_replicate.values()}
        }
        result_ids = [by_replicate[item]["id"] for item in planned if item in by_replicate]
        return {
            "total": len(planned),
            "completed": len(by_replicate),
            "judged": sum(result_id in decisions for result_id in result_ids),
            "result_ids": result_ids,
            "missing_replicates": [item for item in planned if item not in by_replicate],
        }

    @staticmethod
    def _public_progress(progress: Mapping[str, Any]) -> dict[str, int]:
        return {
            "total": int(progress["total"]),
            "completed": int(progress["completed"]),
            "judged": int(progress["judged"]),
        }

    def _claims_for_manifest(self, manifest_id: str) -> list[dict[str, Any]]:
        claims = []
        for path in sorted(self.store.inflight_dir.glob("*.json")):
            row = read_json(path)  # corrupt operational state is a hard stop, not ignored
            if row.get("manifest_id") == manifest_id:
                claims.append(row)
        return claims

    def _claim_has_live_local_pid(self, claim: Mapping[str, Any]) -> bool:
        for field in ("runner_pid", "owner_pid"):
            value = claim.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and self._pid_alive(value):
                return True
        return False

    def _missing_replicate_has_artifacts(
        self, job: Mapping[str, Any], missing_replicates: list[str]
    ) -> bool:
        manifest = self.store.get("execution_manifest", str(job["manifest_id"]))
        for replicate_id in missing_replicates:
            path = self.store.artifacts_dir / f"result_{manifest['digest'][:12]}_{replicate_id}"
            if path.is_dir() and any(path.iterdir()):
                return True
        return False

    def _transition_owned(
        self,
        job_id: str,
        lease_token: str,
        target: str,
        *,
        reason: str,
        updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.store.lock("campaign_queue"):
            job = self._read_job(self.job_path(job_id))
            self._assert_owner(job, lease_token)
            return self._transition_locked(job, target, reason=reason, updates=updates)

    def _transition(
        self,
        job_id: str,
        target: str,
        *,
        reason: str,
        updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.store.lock("campaign_queue"):
            job = self._read_job(self.job_path(job_id))
            return self._transition_locked(job, target, reason=reason, updates=updates)

    def _transition_locked(
        self,
        job: dict[str, Any],
        target: str,
        *,
        reason: str,
        updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = str(job["state"])
        allowed = {
            "pending": {"running", "complete", "blocked"},
            "running": {"pending", "complete", "waiting", "blocked"},
            "waiting": {"running", "complete", "blocked"},
            # Reconciliation permits blocked -> pending only after the operator has
            # removed the orphan claim and no artifacts/result indicate execution.
            "blocked": {"pending", "complete"},
            "complete": set(),
        }
        if target not in allowed[source]:
            raise CampaignStateError(f"illegal queue transition {source} -> {target}")
        timestamp = _iso(self._now())
        job.update(dict(updates or {}))
        job["state"] = target
        job["updated_at"] = timestamp
        job["last_progress_at"] = timestamp
        if target != "running":
            job["owner_pid"] = None
            job["owner_thread"] = None
            job["lease_token"] = None
        job.setdefault("transitions", []).append(
            {"from": source, "to": target, "at": timestamp, "reason": reason}
        )
        self._write_job(job)
        return job

    @staticmethod
    def _assert_owner(job: Mapping[str, Any], lease_token: str) -> None:
        if job["state"] != "running" or job.get("lease_token") != lease_token:
            raise CampaignStateError(f"worker no longer owns {job['job_id']}")

    def _read_job(self, path: Path) -> dict[str, Any]:
        row = read_json(path)
        if row.get("schema_version") != QUEUE_SCHEMA_VERSION:
            raise CampaignStateError(f"unsupported queue schema in {path}")
        if row.get("state") not in QUEUE_STATES:
            raise CampaignStateError(f"invalid queue state in {path}: {row.get('state')!r}")
        if path != self.job_path(str(row.get("job_id"))):
            raise CampaignStateError(f"queue job id/path mismatch in {path}")
        if row.get("job_id") != job_id_for_manifest(str(row.get("manifest_id", ""))):
            raise CampaignStateError(f"queue job id/manifest mismatch in {path}")
        return row

    def _write_job(self, job: Mapping[str, Any]) -> None:
        job_id = str(job.get("job_id", ""))
        if not job_id or job.get("state") not in QUEUE_STATES:
            raise CampaignStateError("cannot write malformed campaign job")
        self.store.write_operational(self.job_path(job_id), job)
