"""Execution Service: scheduler, orchestrator, and hardened runner in one boundary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import posixpath
import re
import shlex
import signal
import subprocess
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .records import (
    PROTOCOL_V2_FORBIDDEN_ENV,
    RecordError,
    make_record,
    read_json,
    sha256_file,
    utc_now,
)
from .store import Store

METRICS_PREFIX = "AUTORESEARCH_METRICS "
# upstream train.py closes with a `---` rule followed by a `key: value` summary
# block.  That is a defined format, so it is parsed structurally.  Loose
# whole-log regex scanning is deliberately NOT used: it silently picks up
# progress-line numbers and then feeds them to the minimum-steps integrity gate.
SUMMARY_RULE = "---"
SUMMARY_LINE = re.compile(r"^([a-z][a-z0-9_]*)\s*:\s*([-+0-9.eE]+)\s*$")
ABORT_PREFIX = "OPHIS_ABORT"


class NoResourceAvailable(RuntimeError):
    pass


class NoPendingReplicate(RuntimeError):
    pass


class ReplicateAlreadyExecuted(RuntimeError):
    """A replicate was claimed whose artifacts already exist on disk.

    ``result_id`` is deterministic in ``(manifest_digest, replicate_id)`` and finished
    artifacts are chmod'd to 0444 so a recorded hash stays true.  Those two protections
    interact badly: a second attempt at the same replicate does not merely get refused,
    it dies partway through ``run_arm`` with

        PermissionError: [Errno 13] Permission denied: .../control.stdout.log

    which the runner reports as ``uncertain_after_internal_failure`` -- an operator has to
    reason about whether a training process was left running somewhere before releasing
    the claim by hand.  Observed on exp_confirm_attn_sink seed_42.

    Raising this instead turns a confusing incident into a stated invariant: artifacts
    exist, therefore this replicate ran, therefore there is nothing to do.  It is a clean
    no-op, not a failure, and the caller removes the claim rather than quarantining it.
    """


@dataclass(frozen=True)
class Allocation:
    resource: Mapping[str, Any]
    gpu: int | None
    launch_telemetry: Mapping[str, Any]
    lease: Mapping[str, Any] | None = None

    @property
    def public(self) -> dict[str, Any]:
        public = {
            "id": self.resource["id"],
            "backend": self.resource["backend"],
            "workdir": self.resource["workdir"],
            "host_id": self.resource.get("host_id", self.resource["id"]),
            "gpu": self.gpu,
            "gpu_uuid": self.launch_telemetry.get("uuid") if self.gpu is not None else None,
        }
        if self.resource.get("reservation") is not None:
            public["reservation"] = dict(self.resource["reservation"])
        if self.lease:
            public["lease_id"] = self.lease.get("lease_id")
        return public


class ResourceScheduler:
    def __init__(self, store: Store, resources: Sequence[Mapping[str, Any]]) -> None:
        self.store = store
        self.resources = list(resources)

    @contextmanager
    def allocate(self, *, require_gpu: bool, wait_seconds: float) -> Iterator[Allocation]:
        deadline = time.monotonic() + wait_seconds
        while True:
            for resource in self.resources:
                candidates: list[int | None] = list(resource.get("gpus", []))
                if not require_gpu:
                    candidates = candidates or [None]
                if require_gpu and not candidates:
                    continue
                # With no explicit host limit, preserve v1's one-worker-per-GPU
                # behavior. A shared host can opt into a stricter cross-GPU cap.
                host_id = str(resource.get("host_id", resource["id"]))
                host_slots = int(resource.get("max_concurrent_jobs", max(1, len(candidates))))
                for host_slot in range(host_slots):
                    try:
                        host_lock = f"host_{_safe_name(host_id)}_{host_slot}"
                        with self.store.lock(host_lock, blocking=False):
                            # Staging mutates workdir-relative paths.  Different GPU
                            # leases may run concurrently only when their workdirs differ;
                            # otherwise one manifest can overwrite another mid-run.
                            workdir_identity = f"{host_id}\0{resource['workdir']}"
                            workdir_digest = hashlib.sha256(
                                workdir_identity.encode("utf-8")
                            ).hexdigest()[:24]
                            with self.store.lock(f"workdir_{workdir_digest}", blocking=False):
                                for gpu in candidates:
                                    lease_name = (
                                        f"{_safe_name(host_id)}_"
                                        f"{_safe_name(str(resource['id']))}_{gpu}"
                                    )
                                    metadata = {
                                        "resource_id": resource["id"],
                                        "host_id": host_id,
                                        "gpu": gpu,
                                        "host_slot": host_slot,
                                        "workdir": resource["workdir"],
                                        "reservation": dict(
                                            resource.get("reservation", {"mode": "shared"})
                                        ),
                                    }
                                    try:
                                        with self.store.resource_lease(
                                            lease_name, metadata, blocking=False
                                        ) as lease:
                                            telemetry = probe(resource, gpu)
                                            if _is_available(
                                                resource, telemetry, require_gpu=require_gpu
                                            ):
                                                yield Allocation(resource, gpu, telemetry, lease)
                                                return
                                    except BlockingIOError:
                                        continue
                    except BlockingIOError:
                        continue
            if time.monotonic() >= deadline:
                raise NoResourceAvailable("no allowed resource passed the launch gate")
            time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))


class ExecutionService:
    def __init__(self, store: Store) -> None:
        self.store = store

    def execute_next(self, manifest_id: str) -> dict[str, Any]:
        manifest = self.store.get("execution_manifest", manifest_id)
        spec = self.store.get("experiment_spec", manifest["payload"]["spec_id"])
        replicate, claim_path, claim = self._claim_next(manifest, spec)
        requirements = manifest["payload"]["requirements"]
        runtime = manifest["payload"]["runtime"]
        scheduler = ResourceScheduler(self.store, manifest["payload"]["resources"])
        remove_claim = False
        try:
            with scheduler.allocate(
                require_gpu=bool(requirements.get("require_gpu", True)),
                wait_seconds=float(runtime["resource_wait_seconds"]),
            ) as allocation:
                claim["resource"] = allocation.public
                claim["allocated_at"] = utc_now()
                self.store.write_operational(claim_path, claim)
                result = self._run_replicate(
                    manifest, spec, replicate, allocation, claim, claim_path
                )
                remove_claim = True
                return result
        except (NoResourceAvailable, ReplicateAlreadyExecuted):
            # Neither of these leaves anything running or half-written, so the claim is
            # released rather than quarantined.  Quarantining a claim that provably never
            # launched a process is what forces an operator to reason about ghosts.
            remove_claim = True
            raise
        except BaseException as exc:
            claim["state"] = "uncertain_after_internal_failure"
            claim["internal_error"] = f"{type(exc).__name__}: {exc}"
            claim["updated_at"] = utc_now()
            self.store.write_operational(claim_path, claim)
            raise
        finally:
            if remove_claim:
                claim_path.unlink(missing_ok=True)

    def execute_all(
        self, manifest_id: str, *, workers: int = 1
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Run every remaining replicate.

        Returns ``(results, unfinished_replicate_ids)``.  A replicate that never
        found a free resource is reported, never raised over the top of results
        that already landed on disk -- those are immutable facts and the caller
        must always get to see them.
        """
        if workers < 1:
            raise RecordError("workers must be >= 1")
        results: list[dict[str, Any]] = []

        def worker() -> list[dict[str, Any]]:
            mine: list[dict[str, Any]] = []
            while True:
                try:
                    mine.append(self.execute_next(manifest_id))
                except NoPendingReplicate:
                    return mine
                except NoResourceAvailable:
                    # More workers than slots is normal.  Stop this worker and let
                    # the post-hoc reconciliation below report the shortfall.
                    return mine

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker) for _ in range(workers)]
            for future in as_completed(futures):
                results.extend(future.result())

        manifest = self.store.get("execution_manifest", manifest_id)
        completed = {
            (row["payload"]["spec_id"], row["payload"]["replicate_id"])
            for row in self.store.list("result_bundle")
        }
        spec_id = manifest["payload"]["spec_id"]
        unfinished = [
            row["replicate_id"]
            for row in manifest["payload"]["plan"]
            if (spec_id, row["replicate_id"]) not in completed
        ]
        return sorted(results, key=lambda row: row["payload"]["replicate_id"]), unfinished

    def _claim_next(
        self, manifest: Mapping[str, Any], spec: Mapping[str, Any]
    ) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        with self.store.lock("replicate_claims"):
            completed = {
                (row["payload"]["spec_id"], row["payload"]["replicate_id"])
                for row in self.store.list("result_bundle")
            }
            inflight: set[tuple[str, str]] = set()
            for path in self.store.inflight_dir.glob("*.json"):
                try:
                    row = read_json(path)
                    inflight.add((str(row.get("spec_id")), str(row.get("replicate_id"))))
                except RecordError:
                    inflight.add(("corrupt", path.name))

            def _has_artifacts(replicate_id: str) -> bool:
                """Filesystem evidence that this replicate already ran.

                ``completed`` is derived from listing result_bundle records, which is a
                directory scan of its own and can lag a concurrent worker.  The artifact
                directory is written *before* the record and is the earliest durable
                trace, so it closes the window in which two workers both believe a
                replicate is free.  Observed once on exp_confirm_attn_sink seed_42, which
                was claimed twice and died on its own read-only log.
                """
                path = self.store.artifacts_dir / f"result_{manifest['digest'][:12]}_{replicate_id}"
                return path.is_dir() and any(path.iterdir())

            replicate = next(
                (
                    dict(item)
                    for item in manifest["payload"]["plan"]
                    if (spec["id"], item["replicate_id"]) not in completed
                    and (spec["id"], item["replicate_id"]) not in inflight
                    and not _has_artifacts(str(item["replicate_id"]))
                ),
                None,
            )
            if replicate is None:
                raise NoPendingReplicate(f"no unclaimed replicate remains for {manifest['id']}")
            token = uuid.uuid4().hex
            claim = {
                "token": token,
                "manifest_id": manifest["id"],
                "manifest_digest": manifest["digest"],
                "spec_id": spec["id"],
                "replicate_id": replicate["replicate_id"],
                "claimed_at": utc_now(),
                "owner_pid": os.getpid(),
            }
            claim_path = self.store.inflight_dir / f"{token}.json"
            self.store.write_operational(claim_path, claim)
            return replicate, claim_path, claim

    def _run_replicate(
        self,
        manifest: Mapping[str, Any],
        spec: Mapping[str, Any],
        replicate: Mapping[str, Any],
        allocation: Allocation,
        claim: dict[str, Any],
        claim_path: Path,
    ) -> dict[str, Any]:
        started_at = utc_now()
        result_id = f"result_{manifest['digest'][:12]}_{replicate['replicate_id']}"
        artifact_dir = self.store.artifacts_dir / result_id
        # Artifacts are the filesystem's record that this replicate ran, and unlike the
        # record listing they cannot go stale between the claim and the launch.  Check
        # them before touching anything: a non-empty directory means a previous attempt
        # already wrote read-only logs here, and proceeding would fault on one of them
        # halfway through the first arm.
        if artifact_dir.is_dir() and any(artifact_dir.iterdir()):
            raise ReplicateAlreadyExecuted(
                f"artifacts already exist for {result_id}; a replicate cannot be re-run "
                "under the same manifest"
            )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        bindings = manifest["payload"]["code_bindings"] + manifest["payload"]["data_bindings"]
        # Place the sealed bytes before checking for them.  Staging is addressed by the
        # sha256 already in the manifest, so it cannot change what runs -- and the
        # verification below is unchanged and still decides whether the run proceeds.
        staging = stage_bindings(allocation.resource, bindings, self.store.root)
        binding_checks = verify_bindings(allocation.resource, bindings)
        corpus_checks = verify_data_corpus(
            allocation.resource, manifest["payload"]["data_bindings"], self.store.root
        )
        environment = execution_environment(allocation)
        arms: list[dict[str, Any]] = []
        failure = ""
        if not all(row["state"] == "verified" for row in binding_checks):
            status = "preflight_failed"
            failure = "one or more code/data bindings failed launch-time verification"
        elif not all(row["state"] == "verified" for row in corpus_checks):
            status = "preflight_failed"
            failure = next(row["error"] for row in corpus_checks if row["state"] != "verified")
        else:
            status = "completed"
            require_gpu = bool(manifest["payload"]["requirements"].get("require_gpu", True))
            for index, arm in enumerate(replicate["arms"]):
                arm_name = str(arm["name"])

                # Re-verify the launch gate immediately before each arm.  The box is
                # shared: a neighbour's job can land on this GPU between allocation
                # and launch, and a contended arm is a confound, not a measurement.
                if require_gpu and allocation.gpu is not None:
                    recheck = _await_idle(allocation.resource, allocation.gpu)
                    if not _is_available(allocation.resource, recheck, require_gpu=True):
                        status = "partial" if index else "preflight_failed"
                        failure = (
                            f"GPU {allocation.gpu} stopped being idle before arm "
                            f"{arm_name}: {recheck.get('error') or recheck}"
                        )
                        break

                def process_started(
                    runner_argv: list[str], pid: int, current_arm: str = arm_name
                ) -> None:
                    claim["state"] = "running"
                    claim["active_arm"] = current_arm
                    claim["runner_argv"] = runner_argv
                    claim["runner_pid"] = pid
                    claim["updated_at"] = utc_now()
                    self.store.write_operational(claim_path, claim)

                arm_result = run_arm(
                    allocation,
                    arm,
                    artifact_dir=artifact_dir,
                    timeout_seconds=float(
                        manifest["payload"]["runtime"]["timeout_seconds_per_arm"]
                    ),
                    telemetry_interval=float(
                        manifest["payload"]["runtime"]["telemetry_interval_seconds"]
                    ),
                    on_start=process_started,
                    artifact_root=self.store.root,
                    sanitize_environment=spec["payload"].get("protocol_version") == 2,
                )
                arms.append(arm_result)
                claim["completed_arms"] = [row["name"] for row in arms]
                claim.pop("active_arm", None)
                claim.pop("runner_pid", None)
                claim["updated_at"] = utc_now()
                self.store.write_operational(claim_path, claim)
                if arm_result["status"] != "completed":
                    status = "partial" if index else arm_result["status"]
                    failure = f"arm {arm['name']} ended as {arm_result['status']}"
                    break
        post_binding_checks = verify_bindings(allocation.resource, bindings)
        if status == "completed" and not all(
            row["state"] == "verified" for row in post_binding_checks
        ):
            status = "failed"
            failure = "a code/data binding changed during execution"

        payload = {
            "manifest_id": manifest["id"],
            "manifest_digest": manifest["digest"],
            "spec_id": spec["id"],
            "replicate_id": replicate["replicate_id"],
            "stage": spec["payload"]["stage"],
            "status": status,
            "failure": failure,
            "lifecycle": {
                "claimed_at": claim["claimed_at"],
                "started_at": started_at,
                "ended_at": utc_now(),
            },
            "resource": allocation.public,
            "environment": environment,
            "launch_telemetry": dict(allocation.launch_telemetry),
            "binding_checks": binding_checks,
            "corpus_checks": corpus_checks,
            "staging": staging,
            "post_binding_checks": post_binding_checks,
            "arms": arms,
        }
        return self.store.put(make_record("result_bundle", result_id, payload))


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def _prefix(resource: Mapping[str, Any]) -> list[str]:
    return list(resource.get("ssh_argv", [])) if resource["backend"] == "ssh" else []


def _control_command(resource: Mapping[str, Any], command: str) -> list[str]:
    if resource["backend"] == "ssh":
        return _prefix(resource) + [command]
    return ["sh", "-lc", command]


def _run_control(
    resource: Mapping[str, Any], command: str, timeout: float = 15
) -> subprocess.CompletedProcess:
    return subprocess.run(
        _control_command(resource, command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


PROBE_SEPARATOR = "___AUTORESEARCH_APPS___"
# One round trip, not two.  An SSH handshake per telemetry sample is itself load
# on a shared box, and perturbing the machine you are measuring is how a
# contention artifact gets recorded as a result.
PROBE_COMMAND = (
    "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu "
    "--format=csv,noheader,nounits; "
    f"echo {PROBE_SEPARATOR}; "
    "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits"
)


def probe(resource: Mapping[str, Any], gpu: int | None) -> dict[str, Any]:
    if gpu is None:
        return {"state": "available", "gpu": None, "observed_at": utc_now()}
    try:
        completed = _run_control(resource, PROBE_COMMAND)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "unavailable", "gpu": gpu, "error": str(exc), "observed_at": utc_now()}
    if completed.returncode != 0:
        return {
            "state": "unavailable",
            "gpu": gpu,
            "error": completed.stderr.strip(),
            "observed_at": utc_now(),
        }
    gpu_text, separator, apps_text = completed.stdout.partition(PROBE_SEPARATOR)
    if not separator:
        return {
            "state": "unavailable",
            "gpu": gpu,
            "error": "compute-process telemetry is unavailable",
            "observed_at": utc_now(),
        }
    selected: dict[str, Any] | None = None
    for line in gpu_text.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 4 and fields[0].isdigit() and int(fields[0]) == gpu:
            selected = {
                "gpu": gpu,
                "uuid": fields[1],
                "memory_used_mb": float(fields[2]),
                "utilization_percent": float(fields[3]),
            }
            break
    if selected is None:
        return {
            "state": "unavailable",
            "gpu": gpu,
            "error": "GPU index not found",
            "observed_at": utc_now(),
        }
    processes: list[int] = []
    for line in apps_text.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 2 and fields[0] == selected["uuid"] and fields[1].isdigit():
            processes.append(int(fields[1]))
    selected.update(
        {
            "state": "available",
            "compute_pids": processes,
            "process_count": len(processes),
            "observed_at": utc_now(),
        }
    )
    return selected


# How long to let a GPU settle before declaring it occupied, and how often to look.
# nvidia-smi keeps reporting non-zero utilization for a few hundred milliseconds after a
# process exits, so a paired replicate -- two arms back-to-back on ONE leased GPU, which
# is the design that makes a screen placement-free -- reliably re-checks the gate while
# its OWN first arm is still draining.  Observed on exp_flr02_paired: the pre-arm probe
# fired 378 ms after the control arm's process ended and read
# utilization 72 percent with ZERO compute processes and ZERO memory in use, so the
# candidate arm was refused and the replicate was consumed without a comparison.
IDLE_SETTLE_ATTEMPTS = 6
IDLE_SETTLE_DELAY_SECONDS = 2.0


def _await_idle(
    resource: Mapping[str, Any],
    gpu: int,
    *,
    attempts: int = IDLE_SETTLE_ATTEMPTS,
    delay: float = IDLE_SETTLE_DELAY_SECONDS,
) -> dict[str, Any]:
    """Probe until the GPU reads idle, or until the settle window is exhausted.

    This does NOT relax the idle predicate -- `_is_available` is unchanged, so a GPU with
    another tenant's process, or with memory in use, still fails exactly as before.  The
    only thing that changes is that a *transient* reading gets a second look instead of
    aborting the arm on its first sample.

    That distinction is the whole point.  The gate exists because a co-tenant arriving
    mid-run destroyed a six-replicate confirmation and flipped the sign of its apparent
    effect; weakening it would give that failure mode back.  A real neighbour occupies the
    GPU for minutes and will still be there after the settle window.  Our own teardown
    will not be.  Waiting distinguishes them without trusting either.

    Returns the last telemetry observed, so the caller reports what it actually saw.
    """
    telemetry = probe(resource, gpu)
    for _ in range(max(0, attempts - 1)):
        if _is_available(resource, telemetry, require_gpu=True):
            return telemetry
        time.sleep(delay)
        telemetry = probe(resource, gpu)
    return telemetry


def _is_available(
    resource: Mapping[str, Any], telemetry: Mapping[str, Any], *, require_gpu: bool
) -> bool:
    if telemetry.get("state") != "available":
        return False
    if not require_gpu:
        return True
    return (
        int(telemetry.get("process_count", 1)) == 0
        and float(telemetry.get("memory_used_mb", float("inf")))
        <= float(resource.get("max_idle_memory_mb", 512))
        and float(telemetry.get("utilization_percent", float("inf")))
        <= float(resource.get("max_idle_utilization_percent", 5))
    )


def verify_data_corpus(
    resource: Mapping[str, Any], data_bindings: Sequence[Mapping[str, Any]], store_root: Path
) -> list[dict[str, Any]]:
    """Check that the host's data directory is exactly the corpus the manifest describes.

    THE HOLE THIS CLOSES.  ``data_manifest.json`` is pinned by sha256 and verified before
    every run, which looks like the corpus is bound.  It is not.  ``prepare.py`` builds its
    shard list with

        sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))

    so the loader reads *whatever parquet files happen to be in the directory* while the
    binding check only confirms that a JSON file **describing** an intended corpus is
    unchanged.  Dropping one extra shard into that directory would change the training
    data of every subsequent run -- and of any run already in flight -- while every
    binding check continued to pass and every record continued to claim the same data
    provenance.  Of all the ways this harness could produce a confident wrong answer,
    this is the quietest.

    The check compares the set of parquet basenames and their byte sizes against the
    manifest.  Sizes rather than hashes because the corpus is ~1 GB and re-hashing it per
    arm would cost more than the training run; sizes catch additions, removals and
    truncations, which are the realistic failure modes.  Small artifacts explicitly listed
    below ``tokenizer/`` are different: they are cheap to hash and directly define both the
    token stream and the BPB denominator, so their size *and* sha256 must match on every
    launch.  This includes the byte-accounting version marker.

    A mismatch fails preflight rather than silently adjusting anything -- the point is to
    make a corpus or tokenizer change impossible to perform by accident, which in turn
    means expanding the corpus or rebuilding the tokenizer has to go through a new data
    manifest, and therefore a new comparison group.
    """
    tokenizer_artifact_limit = 64 * 1024 * 1024
    rows: list[dict[str, Any]] = []
    for binding in data_bindings:
        blob = store_root / str(binding.get("blob") or "")
        if not blob.is_file():
            continue
        try:
            described = json.loads(blob.read_text())
        except (OSError, ValueError) as exc:
            rows.append({"state": "unreadable", "error": str(exc), "corpus": None})
            continue
        cache_dir = described.get("cache_dir")
        files = described.get("files")
        if not cache_dir or not isinstance(files, list):
            continue  # not a corpus-describing manifest; nothing to enforce
        if not isinstance(cache_dir, str) or any(char in cache_dir for char in "\r\n\t"):
            rows.append(
                {
                    "state": "unreadable",
                    "error": "data manifest cache_dir must be control-character-free text",
                    "corpus": None,
                }
            )
            continue
        expected: dict[str, int] = {}
        tokenizer_expected: dict[str, dict[str, Any]] = {}
        try:
            for entry in files:
                if not isinstance(entry, Mapping):
                    raise TypeError("data manifest files must contain objects")
                raw_path = entry.get("path")
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError("data manifest file paths must be non-empty text")
                path = PurePosixPath(raw_path)
                tokenizer_declared = raw_path == "tokenizer" or raw_path.startswith("tokenizer/")
                if tokenizer_declared and (
                    path.is_absolute()
                    or len(path.parts) < 2
                    or path.parts[0] != "tokenizer"
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or str(path) != raw_path
                    or any(char in raw_path for char in "\r\n\t")
                ):
                    raise ValueError(f"unsafe tokenizer artifact path {raw_path!r}")
                if not raw_path.endswith(".parquet") and not tokenizer_declared:
                    continue  # unrelated manifest metadata/artifacts are not runtime inputs

                byte_count = entry.get("bytes")
                if (
                    isinstance(byte_count, bool)
                    or not isinstance(byte_count, int)
                    or byte_count < 0
                ):
                    raise ValueError(f"data manifest has invalid byte size for {raw_path!r}")
                if raw_path.endswith(".parquet"):
                    basename = posixpath.basename(raw_path)
                    if basename in expected:
                        raise ValueError(f"data manifest repeats parquet basename {basename!r}")
                    expected[basename] = byte_count
                    continue

                normalized = str(path)
                if normalized in tokenizer_expected:
                    raise ValueError(f"data manifest repeats tokenizer path {normalized!r}")
                if byte_count > tokenizer_artifact_limit:
                    raise ValueError(
                        f"tokenizer artifact {normalized!r} exceeds the "
                        f"{tokenizer_artifact_limit}-byte preflight hashing limit"
                    )
                digest = entry.get("sha256")
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
                    raise ValueError(f"tokenizer artifact {normalized!r} requires a sha256 digest")
                tokenizer_expected[normalized] = {
                    "bytes": byte_count,
                    "sha256": digest.lower(),
                }
        except (TypeError, ValueError) as exc:
            rows.append({"state": "unreadable", "error": str(exc), "corpus": cache_dir})
            continue

        data_dir = posixpath.join(str(cache_dir), "data")
        listing = f"cd {shlex.quote(data_dir)} && stat -c '%n %s' -- *.parquet 2>/dev/null"
        try:
            if resource["backend"] == "local":
                actual = {
                    path.name: path.stat().st_size for path in Path(data_dir).glob("*.parquet")
                }
            else:
                completed = _run_control(resource, listing)
                actual = {}
                for line in completed.stdout.splitlines():
                    parts = line.rsplit(" ", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        actual[posixpath.basename(parts[0])] = int(parts[1])

            tokenizer_actual: dict[str, dict[str, Any]] = {}
            if tokenizer_expected and resource["backend"] == "local":
                for relative in tokenizer_expected:
                    artifact = Path(cache_dir).joinpath(*PurePosixPath(relative).parts)
                    if not artifact.is_file():
                        continue
                    tokenizer_actual[relative] = {
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
            elif tokenizer_expected:
                remote_to_relative = {
                    posixpath.join(cache_dir, relative): relative for relative in tokenizer_expected
                }
                targets = " ".join(shlex.quote(path) for path in remote_to_relative)
                inspect = _run_control(
                    resource,
                    (
                        f"stat -c '%s %n' -- {targets} 2>/dev/null; "
                        f"sha256sum -- {targets} 2>/dev/null"
                    ),
                )
                sizes: dict[str, int] = {}
                digests: dict[str, str] = {}
                for line in inspect.stdout.splitlines():
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].isdigit():
                        relative = remote_to_relative.get(parts[1])
                        if relative is not None:
                            sizes[relative] = int(parts[0])
                        continue
                    if len(parts) != 2 or re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]) is None:
                        continue
                    relative = remote_to_relative.get(parts[1])
                    if relative is not None:
                        digests[relative] = parts[0].lower()
                for relative in set(sizes) | set(digests):
                    tokenizer_actual[relative] = {
                        "bytes": sizes.get(relative),
                        "sha256": digests.get(relative),
                    }
        except (OSError, subprocess.TimeoutExpired) as exc:
            rows.append({"state": "unreachable", "error": str(exc), "corpus": data_dir})
            continue
        extra = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        resized = sorted(
            name for name in set(expected) & set(actual) if expected[name] != actual[name]
        )
        tokenizer_missing = sorted(set(tokenizer_expected) - set(tokenizer_actual))
        tokenizer_resized = sorted(
            relative
            for relative in set(tokenizer_expected) & set(tokenizer_actual)
            if tokenizer_expected[relative]["bytes"] != tokenizer_actual[relative].get("bytes")
        )
        tokenizer_hash_mismatch = sorted(
            relative
            for relative in set(tokenizer_expected) & set(tokenizer_actual)
            if tokenizer_expected[relative]["sha256"] != tokenizer_actual[relative].get("sha256")
        )
        mismatch = any(
            (
                extra,
                missing,
                resized,
                tokenizer_missing,
                tokenizer_resized,
                tokenizer_hash_mismatch,
            )
        )
        state = "mismatch" if mismatch else "verified"
        rows.append(
            {
                "state": state,
                "corpus": data_dir,
                "shards_expected": len(expected),
                "shards_found": len(actual),
                "extra": extra,
                "missing": missing,
                "resized": resized,
                "tokenizer_expected": len(tokenizer_expected),
                "tokenizer_found": len(tokenizer_actual),
                "tokenizer_missing": tokenizer_missing,
                "tokenizer_resized": tokenizer_resized,
                "tokenizer_hash_mismatch": tokenizer_hash_mismatch,
                "error": (
                    ""
                    if state == "verified"
                    else (
                        f"the host corpus does not match the sealed data manifest: "
                        f"parquet={len(extra)} extra/{len(missing)} missing/"
                        f"{len(resized)} resized; tokenizer={len(tokenizer_missing)} missing/"
                        f"{len(tokenizer_resized)} resized/"
                        f"{len(tokenizer_hash_mismatch)} hash mismatches. "
                        "prepare.py enumerates this directory with os.listdir, so training "
                        "would read a corpus or tokenizer the record does not describe."
                    )
                ),
            }
        )
    return rows


def stage_bindings(
    resource: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    store_root: Path,
) -> list[dict[str, Any]]:
    """Copy each sealed binding to the execution host if it is not already there.

    WHY THIS IS SAFE, which is the only interesting thing about it.  Staging copies from
    ``blobs/sha256/<digest>`` where ``<digest>`` is the sha256 recorded in the immutable
    manifest.  The bytes are therefore selected *by* the hash the record already commits
    to, so staging cannot introduce code that differs from what was sealed -- the worst it
    can do is put the right file somewhere it already was.  ``verify_bindings`` still runs
    afterwards and still has to pass, and the post-run check still has to pass, so nothing
    about provenance is being taken on trust.

    WHY IT IS NEEDED.  Before this, the harness verified bindings but never placed them.
    A newly written training script failed preflight with "No such file or directory" and
    -- because ``_claim_next`` excludes any ``(spec_id, replicate_id)`` that already has a
    ResultBundle, whatever its verdict -- the replicate was **permanently consumed**.  Two
    spec pairs were burned this way in one session, each requiring a fresh spec, a fresh
    seal, and in the confirmation case a fresh review council.  The operator's only
    defence was remembering to scp by hand.

    Returns one row per binding describing what was done, so the record shows whether a
    file was already present or had to be placed.
    """
    staged: list[dict[str, Any]] = []
    workdir = Path(str(resource["workdir"]))
    for binding in bindings:
        relative = str(binding["execution_path"])
        expected = str(binding["sha256"])
        blob = store_root / str(binding.get("blob") or f"blobs/sha256/{expected}")
        row: dict[str, Any] = {"execution_path": relative, "sha256": expected}
        if not blob.is_file():
            # Nothing to stage from.  Not an error here: verify_bindings will fail the
            # preflight if the host does not already have the file, and that failure is
            # the one that carries the right diagnostic.
            staged.append({**row, "action": "unavailable", "error": f"missing blob {blob}"})
            continue
        try:
            if resource["backend"] == "local":
                destination = workdir / relative
                actual = sha256_file(destination) if destination.is_file() else None
            else:
                remote_probe = posixpath.join(str(resource["workdir"]), relative)
                completed = _run_control(resource, f"sha256sum -- {shlex.quote(remote_probe)}")
                actual = (
                    completed.stdout.split()[0]
                    if completed.returncode == 0 and completed.stdout.split()
                    else None
                )

            if actual == expected:
                staged.append({**row, "action": "already_present", "error": ""})
                continue

            # A file that EXISTS with different content is not a staging problem, it is
            # information, and overwriting it would destroy the only signal that this
            # host is not what the operator thinks it is.  Staging fills a gap; it does
            # not repair a discrepancy.  The one exception is content this campaign
            # itself sealed at some point -- a stale version of our own file is a known
            # quantity, not a foreign mutation, and refusing to advance it would make
            # every code revision require a manual cleanup.
            if actual is not None:
                known = (store_root / "blobs" / "sha256" / actual).is_file()
                if not known:
                    staged.append(
                        {
                            **row,
                            "action": "conflict",
                            "error": (
                                f"{relative} exists with sha256 {actual}, which this "
                                "store has never sealed; refusing to overwrite. "
                                "verify_bindings will fail this preflight."
                            ),
                        }
                    )
                    continue

            if resource["backend"] == "local":
                destination.parent.mkdir(parents=True, exist_ok=True)
                # The blob we copy from lives in a content-addressed store and is kept
                # read-only, and copying propagates that mode.  A file this function placed
                # earlier is therefore UNWRITABLE, so advancing it to a new revision fails
                # unless the old one is removed first.  Staging could place a file once and
                # never update it -- observed as
                #   scp: dest open ".../train_ngw.py": Permission denied
                # after a script was edited and re-sealed.
                destination.unlink(missing_ok=True)
                destination.write_bytes(blob.read_bytes())
                destination.chmod(0o644)
                staged.append(
                    {
                        **row,
                        "action": "staged" if actual is None else "advanced",
                        "error": "",
                    }
                )
                continue
            remote = posixpath.join(str(resource["workdir"]), relative)
            parent = posixpath.dirname(remote)
            if parent:
                _run_control(resource, f"mkdir -p -- {shlex.quote(parent)}")
            # Same reason as the local branch: a previously staged file is read-only, and
            # scp cannot open it for writing.  Remove before replacing.  Only reached when
            # the existing content is either absent or a digest this store has sealed, so
            # nothing unknown is ever destroyed.
            if actual is not None:
                _run_control(resource, f"rm -f -- {shlex.quote(remote)}")
            copied = _copy_to_host(resource, blob, remote)
            staged.append(
                {
                    **row,
                    "action": "staged" if copied.returncode == 0 else "failed",
                    "error": copied.stderr.strip() if copied.returncode else "",
                }
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            staged.append({**row, "action": "failed", "error": str(exc)})
    return staged


def _copy_to_host(
    resource: Mapping[str, Any], local: Path, remote_path: str
) -> subprocess.CompletedProcess[str]:
    """scp one file to the execution host, reusing the ssh options from ``ssh_argv``.

    The resource declares its connection as an ``ssh`` argv.  Rather than inventing a
    second, separately-configured transport, translate that argv into the equivalent scp
    invocation so both share one identity file, one port, and one ControlMaster socket --
    a mismatch between them would show up as an authentication failure at the worst
    possible moment.
    """
    ssh_argv = [str(token) for token in resource["ssh_argv"]]
    target = ssh_argv[-1]
    options = ssh_argv[1:-1]
    scp_argv = ["scp"]
    index = 0
    while index < len(options):
        token = options[index]
        if token == "-p":  # ssh port flag is -p, scp's is -P
            scp_argv += ["-P", options[index + 1]]
            index += 2
            continue
        if token in {"-i", "-o"}:
            scp_argv += [token, options[index + 1]]
            index += 2
            continue
        scp_argv.append(token)
        index += 1
    scp_argv += [str(local), f"{target}:{remote_path}"]
    return subprocess.run(scp_argv, capture_output=True, text=True, timeout=300, check=False)


def verify_bindings(
    resource: Mapping[str, Any], bindings: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    workdir = Path(str(resource["workdir"]))
    for binding in bindings:
        relative = str(binding["execution_path"])
        expected = str(binding["sha256"])
        if resource["backend"] == "local":
            path = workdir / relative
            actual = sha256_file(path) if path.is_file() else None
            error = "" if actual else "execution path is missing"
        else:
            remote_path = shlex.quote(posixpath.join(str(resource["workdir"]), relative))
            try:
                completed = _run_control(resource, f"sha256sum -- {remote_path}")
                actual = completed.stdout.split()[0] if completed.returncode == 0 else None
                error = completed.stderr.strip() if completed.returncode else ""
            except (OSError, subprocess.TimeoutExpired) as exc:
                actual = None
                error = str(exc)
        result.append(
            {
                "execution_path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "state": "verified" if actual == expected else "mismatch",
                "error": error,
            }
        )
    return result


def execution_environment(allocation: Allocation) -> dict[str, Any]:
    resource = allocation.resource
    if resource["backend"] == "local":
        return {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "scope": "execution_host",
        }
    python = str(resource.get("python", "python3"))
    command = (
        f"cd {shlex.quote(str(resource['workdir']))} && "
        f"{shlex.quote(python)} -c "
        + shlex.quote(
            "import json,platform; print(json.dumps({'host':platform.node(),"
            "'platform':platform.platform(),'python':platform.python_version(),"
            "'scope':'execution_host'},sort_keys=True))"
        )
    )
    try:
        completed = _run_control(resource, command)
        if completed.returncode == 0:
            return json.loads(completed.stdout.strip().splitlines()[-1])
        return {"scope": "execution_host", "state": "unknown", "error": completed.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"scope": "execution_host", "state": "unknown", "error": str(exc)}


def build_runner_argv(
    allocation: Allocation,
    payload_argv: Sequence[str],
    arm_env: Mapping[str, str],
    *,
    timeout_seconds: float,
    sanitize_environment: bool = False,
) -> tuple[list[str], dict[str, str]]:
    resource = allocation.resource
    env = dict(arm_env)
    if allocation.gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(allocation.launch_telemetry.get("uuid", allocation.gpu))
    if resource["backend"] == "local":
        return list(payload_argv), env
    environment_argv = ["env"]
    if sanitize_environment:
        for key in sorted(PROTOCOL_V2_FORBIDDEN_ENV):
            environment_argv.extend(["-u", key])
    environment_argv.extend(f"{key}={value}" for key, value in sorted(env.items()))
    # The brace group must OPEN BEFORE `env`: `env VAR=x { ... }` is a SYNTAX ERROR,
    # because env takes a COMMAND and `{` is a shell keyword. I verified the brace
    # construct in isolation, shipped it, and the very next run died in 20s with
    # "syntax error near unexpected token `}'". Test the COMPOSITION, not the fragment.
    command = f"cd {shlex.quote(str(resource['workdir']))} && {{ "
    if len(environment_argv) > 1:
        command += f"{shlex.join(environment_argv)} "
    # The remote watchdog outlives the local SSH client.  If the connection is
    # severed, GNU timeout still terminates the remote process at the sealed
    # deadline.  The local watchdog below adds a small cleanup grace period.
    # OPHIS_REMOTE_EXIT: the REMOTE status, echoed to stdout so it survives a dead pipe.
    #
    # `return_code` below is the LOCAL ssh client's exit code. When the connection drops,
    # ssh exits 255 and the remote status is lost -- which is exactly what happened to
    # seed_192 and seed_210 (rc=255, zero-byte stderr, stdout truncated mid-stream with
    # healthy metrics). Nothing in the record could distinguish "the run crashed" from
    # "the pipe went away", and the confirmation then sealed on exactly its minimum n.
    # Line present -> the remote process finished and its TRUE status is on the line.
    # Line absent  -> the transport died, as a POSITIVE observation rather than a guess.
    #
    # The braces and `exit $__rc` are load-bearing: `A; echo` returns ECHO's status, so
    # the naive form would report rc=0 for EVERY run and silently mask all failures --
    # including the 124/137 timeout codes checked below. Verified in a shell before
    # being written here.
    command += (
        "timeout --signal=TERM --kill-after=5s "
        f"{max(1, math.ceil(timeout_seconds))}s {shlex.join(payload_argv)}"
        '; __rc=$?; echo "OPHIS_REMOTE_EXIT=$__rc"; exit $__rc; }'
    )
    return _prefix(resource) + [command], {}


def run_arm(
    allocation: Allocation,
    arm: Mapping[str, Any],
    *,
    artifact_dir: Path,
    timeout_seconds: float,
    telemetry_interval: float,
    on_start: Any | None = None,
    artifact_root: Path | None = None,
    sanitize_environment: bool = False,
) -> dict[str, Any]:
    """Run one arm.

    ``artifact_root`` is the state root.  Artifact locations are recorded
    RELATIVE to it, never as absolute host paths: an absolute path baked into a
    digest-sealed record makes the state directory unmovable, and lets
    validation pass by hashing some *other* root's logs.
    """
    name = str(arm["name"])
    stdout_path = artifact_dir / f"{name}.stdout.log"
    stderr_path = artifact_dir / f"{name}.stderr.log"
    runner_argv, extra_env = build_runner_argv(
        allocation,
        arm["argv"],
        arm.get("env", {}),
        timeout_seconds=timeout_seconds,
        sanitize_environment=sanitize_environment,
    )
    environment = os.environ.copy()
    if sanitize_environment:
        for key in PROTOCOL_V2_FORBIDDEN_ENV:
            environment.pop(key, None)
    environment.update(extra_env)
    started_at = utc_now()
    telemetry: list[dict[str, Any]] = []
    status = "failed"
    return_code: int | None = None
    failure = ""
    start = time.monotonic()
    process: subprocess.Popen | None = None
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        try:
            process = subprocess.Popen(
                runner_argv,
                cwd=str(allocation.resource["workdir"])
                if allocation.resource["backend"] == "local"
                else None,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
            if on_start is not None:
                on_start(runner_argv, process.pid)
            next_telemetry_at = 0.0
            while process.poll() is None:
                elapsed = time.monotonic() - start
                outer_deadline = timeout_seconds
                if allocation.resource["backend"] == "ssh":
                    outer_deadline += 7
                if elapsed > outer_deadline:
                    _terminate_process(process)
                    status = "timed_out"
                    failure = f"arm exceeded {timeout_seconds:.1f}s timeout"
                    break
                if allocation.gpu is not None and elapsed >= next_telemetry_at:
                    telemetry.append(probe(allocation.resource, allocation.gpu))
                    next_telemetry_at = elapsed + telemetry_interval
                # Telemetry cadence must not inflate the independent runner clock.
                # The old sleep(interval) loop made an instant process look one full
                # interval long; setting interval=the benchmark frame could therefore
                # manufacture an on-frame result. Poll process completion separately
                # while sampling telemetry only when its own deadline arrives.
                remaining = max(0.01, outer_deadline - elapsed)
                until_telemetry = (
                    max(0.01, next_telemetry_at - elapsed) if allocation.gpu is not None else 0.1
                )
                wait_seconds = min(0.1, remaining, until_telemetry)
                try:
                    process.wait(timeout=wait_seconds)
                except subprocess.TimeoutExpired:
                    pass
            return_code = process.returncode
            if status != "timed_out":
                if allocation.resource["backend"] == "ssh" and return_code in {124, 137}:
                    status = "timed_out"
                    failure = f"remote watchdog exceeded {timeout_seconds:.1f}s timeout"
                else:
                    status = "completed" if return_code == 0 else "failed"
                if return_code and not failure:
                    failure = f"runner exited with code {return_code}"
        except OSError as exc:
            failure = str(exc)
            status = "failed"
        except BaseException:
            if process is not None and process.poll() is None:
                _terminate_process(process)
            raise
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    metrics, metric_error = parse_metrics(text)
    if metric_error and not failure:
        failure = metric_error
    process_counts = [
        int(sample["process_count"])
        for sample in telemetry
        if sample.get("state") == "available" and "process_count" in sample
    ]
    unavailable_samples = sum(sample.get("state") != "available" for sample in telemetry)
    stdout_digest = sha256_file(stdout_path)
    stderr_digest = sha256_file(stderr_path)
    stdout_path.chmod(0o444)
    stderr_path.chmod(0o444)
    root = artifact_root.resolve() if artifact_root is not None else None

    def relative(path: Path) -> str:
        return str(path.resolve().relative_to(root)) if root is not None else str(path)

    return {
        "name": name,
        "payload_argv": list(arm["argv"]),
        "payload_env": dict(arm.get("env", {})),
        "runner_argv": runner_argv,
        "status": status,
        "return_code": return_code,
        "failure": failure,
        "started_at": started_at,
        "ended_at": utc_now(),
        "wall_seconds": time.monotonic() - start,
        "metrics": metrics,
        "telemetry": {
            "sample_count": len(telemetry),
            "unavailable_samples": unavailable_samples,
            "max_compute_processes": max(process_counts) if process_counts else None,
            "samples": telemetry,
        },
        "artifacts": {
            "stdout": relative(stdout_path),
            "stdout_sha256": stdout_digest,
            "stderr": relative(stderr_path),
            "stderr_sha256": stderr_digest,
        },
    }


def _terminate_process(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def parse_metrics(text: str) -> tuple[dict[str, float], str]:
    """Extract endpoint metrics from a run log.

    Two structured formats are accepted, in priority order:

    1. ``AUTORESEARCH_METRICS {json}`` -- the explicit contract;
    2. upstream ``train.py``'s trailing ``---`` summary block.

    Anything else is an error.  A run whose metrics cannot be read structurally
    is an instrumentation failure, not a run to be guessed at.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        if line.startswith(METRICS_PREFIX):
            try:
                raw = json.loads(line[len(METRICS_PREFIX) :])
                metrics = {
                    str(key): float(value)
                    for key, value in raw.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                return metrics, ""
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                return {}, f"invalid {METRICS_PREFIX.strip()} payload: {exc}"

    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() != SUMMARY_RULE:
            continue
        metrics = {}
        for line in lines[index + 1 :]:
            match = SUMMARY_LINE.match(line.strip())
            if match is None:
                continue
            try:
                metrics[match.group(1)] = float(match.group(2))
            except ValueError:
                continue
        if "val_bpb" in metrics:
            return metrics, ""
        break

    if any(line.lstrip().startswith(ABORT_PREFIX) for line in lines):
        return {}, "train.py aborted the frame before producing a scorable val_bpb"
    return {}, "no structured metrics were emitted"
