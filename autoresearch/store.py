"""Content-addressed immutable store with rebuildable indexes and views."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .records import (
    KINDS,
    ConflictError,
    RecordError,
    canonical_json,
    read_json,
    sha256_file,
    utc_now,
    verify_record,
)


class Store:
    def __init__(self, root: str | Path = ".autoresearch") -> None:
        self.root = Path(root).resolve()

    @property
    def records_dir(self) -> Path:
        return self.root / "records"

    @property
    def blobs_dir(self) -> Path:
        return self.root / "blobs" / "sha256"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def inflight_dir(self) -> Path:
        return self.root / "operational" / "inflight"

    @property
    def locks_dir(self) -> Path:
        return self.root / "operational" / "locks"

    @property
    def leases_dir(self) -> Path:
        return self.root / "operational" / "leases"

    @property
    def views_dir(self) -> Path:
        return self.root / "views"

    def init(self) -> None:
        for kind in sorted(KINDS):
            (self.records_dir / kind).mkdir(parents=True, exist_ok=True)
        for path in (
            self.blobs_dir,
            self.artifacts_dir,
            self.inflight_dir,
            self.locks_dir,
            self.leases_dir,
            self.views_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def record_path(self, kind: str, record_id: str) -> Path:
        if kind not in KINDS:
            raise RecordError(f"unknown kind: {kind}")
        return self.records_dir / kind / f"{record_id}.json"

    def put(self, record: Mapping[str, Any]) -> dict[str, Any]:
        self.init()
        verify_record(record)
        normalized = json.loads(canonical_json(dict(record)))
        path = self.record_path(str(normalized["kind"]), str(normalized["id"]))
        payload = (canonical_json(normalized) + "\n").encode("utf-8")
        temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = read_json(path)
                if existing != normalized:
                    raise ConflictError(f"immutable record conflict at {path}")
                return existing
            self._fsync_dir(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return normalized

    def get(self, kind: str, record_id: str) -> dict[str, Any]:
        path = self.record_path(kind, record_id)
        if not path.exists():
            raise RecordError(f"missing {kind} record: {record_id}")
        record = read_json(path)
        verify_record(record, expected_kind=kind)
        return record

    def list(self, kind: str) -> list[dict[str, Any]]:
        directory = self.records_dir / kind
        if not directory.exists():
            return []
        records = [read_json(path) for path in sorted(directory.glob("*.json"))]
        for record in records:
            verify_record(record, expected_kind=kind)
        return sorted(records, key=lambda record: (record["created_at"], record["id"]))

    def add_blob(self, source: Path) -> tuple[str, str]:
        self.init()
        source = source.resolve()
        if not source.is_file():
            raise RecordError(f"binding source is not a file: {source}")
        digest = sha256_file(source)
        destination = self.blobs_dir / digest
        if not destination.exists():
            temporary = destination.parent / f".{digest}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            try:
                with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output:
                    for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    pass
                self._fsync_dir(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256_file(destination) != digest:
            raise ConflictError(f"blob content mismatch: {destination}")
        return digest, str(destination.relative_to(self.root))

    def verify_blob(self, digest: str, relative_path: str) -> bool:
        path = self.root / relative_path
        return path.is_file() and sha256_file(path) == digest

    @contextmanager
    def lock(self, name: str, *, blocking: bool = True) -> Iterator[Any]:
        self.init()
        path = self.locks_dir / f"{name}.lock"
        handle = path.open("a+")
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        acquired = False
        try:
            fcntl.flock(handle.fileno(), flags)
            acquired = True
            yield handle
        finally:
            try:
                if acquired:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    @contextmanager
    def resource_lease(
        self,
        name: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        blocking: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Coordinate one resource and expose its live ownership as JSON.

        The flock is the authority: it is released automatically if the owner
        process dies.  The JSON file is deliberately only operational state. It
        makes a live lease inspectable and leaves a crash breadcrumb, but it is
        never treated as proof that an unrelated process cannot use the device.
        In particular, this coordinates autoresearch workers sharing this state
        root; physical exclusivity still requires an external reservation.
        """
        if not name or Path(name).name != name or name in {".", ".."}:
            raise RecordError("resource lease name must be a non-empty path-safe name")
        with self.lock(f"resource_{name}", blocking=blocking):
            lease = dict(metadata or {})
            lease.update(
                {
                    "lease_id": uuid.uuid4().hex,
                    "name": name,
                    "owner_pid": os.getpid(),
                    "owner_host": platform.node(),
                    "acquired_at": utc_now(),
                }
            )
            path = self.leases_dir / f"{name}.json"
            self.write_operational(path, lease)
            try:
                yield lease
            finally:
                # Do not erase a manually replaced breadcrumb. This should not
                # happen while the flock is held, but token checking makes the
                # cleanup safe even under operator intervention.
                try:
                    current = read_json(path)
                except RecordError:
                    current = {}
                if current.get("lease_id") == lease["lease_id"]:
                    path.unlink(missing_ok=True)
                    self._fsync_dir(path.parent)

    def resource_leases(self) -> list[dict[str, Any]]:
        """Return visible coordination leases, including stale crash breadcrumbs."""
        self.init()
        rows: list[dict[str, Any]] = []
        for path in sorted(self.leases_dir.glob("*.json")):
            try:
                lease = read_json(path)
            except RecordError as exc:
                rows.append({"name": path.stem, "state": "corrupt", "error": str(exc)})
                continue
            rows.append(lease)
        return sorted(rows, key=lambda row: str(row.get("acquired_at", "")), reverse=True)

    def write_operational(self, path: Path, value: Mapping[str, Any]) -> None:
        self.init()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        data = (canonical_json(dict(value)) + "\n").encode("utf-8")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self._fsync_dir(path.parent)

    def write_view(self, name: str, content: str) -> Path:
        self.init()
        path = self.views_dir / name
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
        return path

    def validate(self) -> dict[str, Any]:
        self.init()
        errors: list[str] = []
        counts: dict[str, int] = {}
        records: dict[str, dict[str, dict[str, Any]]] = {}
        for kind in sorted(KINDS):
            try:
                items = self.list(kind)
            except RecordError as exc:
                errors.append(str(exc))
                items = []
            counts[kind] = len(items)
            records[kind] = {item["id"]: item for item in items}

        v2_manifests_by_spec: dict[str, list[str]] = {}
        for manifest in records["execution_manifest"].values():
            payload = manifest["payload"]
            spec = records["experiment_spec"].get(payload["spec_id"])
            if spec is None:
                errors.append(f"{manifest['id']} references missing spec {payload['spec_id']}")
            elif spec["digest"] != payload["spec_digest"]:
                errors.append(f"{manifest['id']} spec digest mismatch")
            else:
                for field in (
                    "stage",
                    "plan",
                    "requirements",
                    "metric",
                    "analysis",
                    "comparison_group",
                ):
                    if payload[field] != spec["payload"][field]:
                        errors.append(f"{manifest['id']} changes sealed scientific field {field}")
                for field in ("protocol_version", "search", "scope"):
                    if payload.get(field) != spec["payload"].get(field):
                        errors.append(f"{manifest['id']} changes sealed scientific field {field}")
                if spec["payload"].get("protocol_version") == 2:
                    v2_manifests_by_spec.setdefault(spec["id"], []).append(manifest["id"])
                roles = {row.get("role") for row in payload.get("reviews", [])}
                if (
                    payload["stage"] == "confirmation"
                    and payload.get("protocol_version") != 2
                    and roles
                    != {
                        "mechanism",
                        "falsifier",
                        "novelty",
                        "provenance",
                        "methodology",
                    }
                ):
                    errors.append(f"{manifest['id']} lacks the complete confirmation council")
            for binding in payload["code_bindings"] + payload["data_bindings"]:
                if not self.verify_blob(binding["sha256"], binding["blob"]):
                    errors.append(
                        f"{manifest['id']} has missing or corrupt blob {binding['sha256']}"
                    )

        for spec_id, manifest_ids in sorted(v2_manifests_by_spec.items()):
            if len(manifest_ids) > 1:
                errors.append(
                    f"protocol v2 spec {spec_id} has multiple immutable manifests: "
                    f"{sorted(manifest_ids)}"
                )

        seen_replicates: dict[tuple[str, str], str] = {}
        for result in records["result_bundle"].values():
            payload = result["payload"]
            manifest = records["execution_manifest"].get(payload["manifest_id"])
            if manifest is None:
                errors.append(f"{result['id']} references missing manifest")
            elif manifest["digest"] != payload["manifest_digest"]:
                errors.append(f"{result['id']} manifest digest mismatch")
            elif payload["spec_id"] != manifest["payload"]["spec_id"]:
                errors.append(f"{result['id']} spec id disagrees with its manifest")
            elif payload["stage"] != manifest["payload"]["stage"]:
                errors.append(f"{result['id']} stage disagrees with its manifest")
            elif payload["replicate_id"] not in {
                row["replicate_id"] for row in manifest["payload"]["plan"]
            }:
                errors.append(f"{result['id']} replicate is absent from its manifest")
            else:
                planned = next(
                    row
                    for row in manifest["payload"]["plan"]
                    if row["replicate_id"] == payload["replicate_id"]
                )
                if payload["status"] == "completed":
                    actual_signature = [
                        (row.get("name"), row.get("payload_argv"), row.get("payload_env"))
                        for row in payload.get("arms", [])
                    ]
                    planned_signature = [
                        (row["name"], row["argv"], row.get("env", {})) for row in planned["arms"]
                    ]
                    if actual_signature != planned_signature:
                        errors.append(f"{result['id']} execution payload differs from its plan")
                authorized = {
                    (row["id"], row["backend"], row["workdir"])
                    for row in manifest["payload"]["resources"]
                }
                resource_key = (
                    payload["resource"].get("id"),
                    payload["resource"].get("backend"),
                    payload["resource"].get("workdir"),
                )
                if resource_key not in authorized:
                    errors.append(f"{result['id']} used a resource outside its manifest")
            key = (payload["spec_id"], payload["replicate_id"])
            previous = seen_replicates.get(key)
            if previous:
                errors.append(
                    f"replicate {key} has multiple ResultBundles: {previous}, {result['id']}"
                )
            seen_replicates[key] = result["id"]

        seen_decisions: dict[tuple[str, str], str] = {}
        for decision in records["evidence_decision"].values():
            payload = decision["payload"]
            result = records["result_bundle"].get(payload["result_id"])
            if result is None:
                errors.append(f"{decision['id']} references missing result")
            elif result["digest"] != payload["result_digest"]:
                errors.append(f"{decision['id']} result digest mismatch")
            elif result["payload"]["spec_id"] != payload["spec_id"]:
                errors.append(f"{decision['id']} spec id disagrees with its result")
            elif result["payload"]["stage"] != payload.get("stage"):
                errors.append(f"{decision['id']} stage disagrees with its result")
            if result is not None:
                spec = records["experiment_spec"].get(payload["spec_id"])
                if (
                    spec is not None
                    and spec["payload"].get("protocol_version") == 2
                    and payload["measurement_verdict"] == "valid"
                ):
                    manifest = records["execution_manifest"].get(result["payload"]["manifest_id"])
                    planned_seed = None
                    if manifest is not None:
                        planned = next(
                            (
                                row
                                for row in manifest["payload"]["plan"]
                                if row["replicate_id"] == result["payload"]["replicate_id"]
                            ),
                            None,
                        )
                        if planned is not None:
                            planned_seed = planned.get("seed")
                    verified_seed = payload.get("verified_seed")
                    measurement_seed = payload.get("measurements", {}).get("verified_seed")
                    if payload.get("policy_version") not in {"evidence-v2", "evidence-v3"}:
                        errors.append(
                            f"{decision['id']} uses a non-v2 policy for protocol v2 evidence"
                        )
                    if verified_seed != planned_seed or measurement_seed != planned_seed:
                        errors.append(
                            f"{decision['id']} verified seed disagrees with its sealed plan"
                        )
            key = (payload["result_id"], payload["policy_version"])
            previous = seen_decisions.get(key)
            if previous:
                errors.append(
                    f"result/policy {key} has multiple decisions: {previous}, {decision['id']}"
                )
            seen_decisions[key] = decision["id"]

        for paper in records["paper"].values():
            payload = paper["payload"]
            for spec_id in payload["spec_ids"]:
                if spec_id not in records["experiment_spec"]:
                    errors.append(f"{paper['id']} references missing spec {spec_id}")
            for evidence_id in payload.get("evidence_ids", []):
                if evidence_id not in records["evidence_decision"]:
                    errors.append(f"{paper['id']} references missing evidence {evidence_id}")
            if not self.verify_blob(payload["content_sha256"], payload["blob"]):
                errors.append(f"{paper['id']} has a missing or corrupt content snapshot")

        agendas = records["research_agenda"]
        sources = records["literature_source"]
        searches = records["literature_search"]
        claims = records["scientific_claim"]
        mechanisms = records["scientific_mechanism"]
        hypotheses = records["scientific_hypothesis"]
        evidence_rows = records["evidence_decision"]

        for search in searches.values():
            payload = search["payload"]
            agenda_id = payload.get("agenda_id")
            if agenda_id and agenda_id not in agendas:
                errors.append(f"{search['id']} references missing agenda {agenda_id}")
            if agenda_id and payload.get("topic_id"):
                agenda = agendas.get(agenda_id)
                topic_ids = {row["id"] for row in agenda["payload"]["topics"]} if agenda else set()
                if payload["topic_id"] not in topic_ids:
                    errors.append(
                        f"{search['id']} references missing agenda topic {payload['topic_id']}"
                    )
            for source_id in payload["result_source_ids"]:
                if source_id not in sources:
                    errors.append(f"{search['id']} references missing source {source_id}")

        for source in sources.values():
            payload = source["payload"]
            search_id = payload["retrieval"].get("search_id")
            if search_id and search_id not in searches:
                errors.append(f"{source['id']} references missing search {search_id}")
            for topic_id in payload.get("topics", []):
                if not any(
                    topic_id in {row["id"] for row in agenda["payload"]["topics"]}
                    for agenda in agendas.values()
                ):
                    errors.append(f"{source['id']} references unknown topic {topic_id}")
            content = payload["content"]
            if content["status"] == "fulltext_snapshot" and not self.verify_blob(
                content["sha256"], content["blob"]
            ):
                errors.append(f"{source['id']} has a missing or corrupt full-text snapshot")

        for claim in claims.values():
            payload = claim["payload"]
            for source_id in payload["source_ids"]:
                if source_id not in sources:
                    errors.append(f"{claim['id']} references missing source {source_id}")
            for evidence_id in payload["evidence_ids"]:
                if evidence_id not in evidence_rows:
                    errors.append(f"{claim['id']} references missing evidence {evidence_id}")
            for source_claim_id in payload["derived_from_claim_ids"]:
                if source_claim_id not in claims:
                    errors.append(
                        f"{claim['id']} references missing derived claim {source_claim_id}"
                    )
            for locator in payload.get("locators", []):
                if locator["source_id"] not in payload["source_ids"]:
                    errors.append(
                        f"{claim['id']} locator source {locator['source_id']} is not cited"
                    )

        for mechanism in mechanisms.values():
            for claim_id in mechanism["payload"]["source_claim_ids"]:
                if claim_id not in claims:
                    errors.append(f"{mechanism['id']} references missing claim {claim_id}")
                elif claims[claim_id]["payload"]["stance"] != "supports":
                    errors.append(
                        f"{mechanism['id']} uses opposing claim {claim_id} as a causal edge"
                    )
            for assumption in mechanism["payload"].get("assumptions", []):
                for claim_id in assumption.get("claim_ids", []):
                    if claim_id not in claims:
                        errors.append(
                            f"{mechanism['id']} assumption references missing claim {claim_id}"
                        )

        for hypothesis in hypotheses.values():
            payload = hypothesis["payload"]
            known_topics = {
                row["id"] for agenda in agendas.values() for row in agenda["payload"]["topics"]
            }
            for topic_id in payload["topics"]:
                if topic_id not in known_topics:
                    errors.append(f"{hypothesis['id']} references unknown topic {topic_id}")
            for claim_id in payload["claim_ids"]:
                if claim_id not in claims:
                    errors.append(f"{hypothesis['id']} references missing claim {claim_id}")
            for mechanism_id in payload["mechanism_ids"]:
                if mechanism_id not in mechanisms:
                    errors.append(f"{hypothesis['id']} references missing mechanism {mechanism_id}")
            for competing_id in payload.get("competing_hypothesis_ids", []):
                if competing_id not in hypotheses:
                    errors.append(
                        f"{hypothesis['id']} references missing competing hypothesis {competing_id}"
                    )

        for spec in records["experiment_spec"].values():
            search = spec["payload"].get("search", {})
            for hypothesis_id in search.get("hypothesis_ids", []):
                if hypothesis_id not in hypotheses:
                    errors.append(f"{spec['id']} references missing hypothesis {hypothesis_id}")
                elif hypothesis_id not in spec["payload"]["knowledge"]["source_ids"]:
                    errors.append(
                        f"{spec['id']} scientific hypothesis is absent from knowledge.source_ids"
                    )

        for result in records["result_bundle"].values():
            for arm in result["payload"].get("arms", []):
                artifacts = arm.get("artifacts", {})
                for name in ("stdout", "stderr"):
                    recorded = str(artifacts.get(name, ""))
                    if not recorded:
                        errors.append(f"{result['id']} has no recorded {name} artifact")
                        continue
                    # Artifact locations are root-relative so the state directory
                    # stays movable and can never validate against another root.
                    if Path(recorded).is_absolute():
                        errors.append(f"{result['id']} records an absolute {name} artifact path")
                        continue
                    path = self.root / recorded
                    digest = artifacts.get(f"{name}_sha256")
                    if not path.is_file() or not digest or sha256_file(path) != digest:
                        errors.append(f"{result['id']} has missing or corrupt {name} artifact")

        # An in-flight claim is a live-process fact, not a corrupt record.  It is
        # reported as a warning so one orphan cannot freeze synthesis for every
        # unrelated spec in the registry; `autoresearch claims` resolves them.
        warnings: list[str] = []
        inflight_paths = list(self.inflight_dir.glob("*.json"))
        for path in inflight_paths:
            try:
                read_json(path)
            except RecordError as exc:
                errors.append(f"corrupt inflight claim {path.name}: {exc}")
        if inflight_paths:
            warnings.append(f"{len(inflight_paths)} unresolved in-flight execution claim(s)")

        return {
            "valid": not errors,
            "counts": counts,
            "inflight": len(inflight_paths),
            "errors": errors,
            "warnings": warnings,
        }

    def claims(self) -> list[dict[str, Any]]:
        """Every in-flight execution claim, newest first."""
        rows: list[dict[str, Any]] = []
        for path in sorted(self.inflight_dir.glob("*.json")):
            try:
                claim = read_json(path)
            except RecordError as exc:
                rows.append({"token": path.stem, "state": "corrupt", "error": str(exc)})
                continue
            claim["token"] = path.stem
            rows.append(claim)
        return sorted(rows, key=lambda row: str(row.get("claimed_at", "")), reverse=True)

    def release_claim(self, token: str) -> dict[str, Any]:
        """Remove one in-flight claim after an operator confirmed the process is dead.

        Only operational state is removed.  Immutable scientific records are never
        touched, so a released claim can never fabricate a terminal result.
        """
        # Tokens are filenames, never paths.  Without this check a value such as
        # ``../../records/experiment_spec/exp_x`` escapes operational state and can
        # unlink an immutable scientific record.
        if (
            not isinstance(token, str)
            or not token
            or Path(token).name != token
            or token in {".", ".."}
            or any(character in token for character in ("/", "\\", "\x00"))
        ):
            raise RecordError("in-flight claim token must be a path-safe filename stem")
        path = self.inflight_dir / f"{token}.json"
        if not path.is_file():
            raise RecordError(f"no in-flight claim with token {token}")
        try:
            claim = read_json(path)
        except RecordError:
            claim = {"token": token, "state": "corrupt"}
        path.unlink()
        self._fsync_dir(path.parent)
        return claim

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
