"""Canonical immutable records and compact schema validation.

Execution records and scientific-library records are immutable authorities.
Belief confidence, research gaps, hypothesis status, and leaderboards are
rebuildable projections over those records; they are never mutable truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KINDS = {
    "experiment_spec",
    "execution_manifest",
    "result_bundle",
    "evidence_decision",
    "paper",
    "research_agenda",
    "literature_search",
    "literature_source",
    "scientific_claim",
    "scientific_mechanism",
    "scientific_hypothesis",
}
ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{2,159}$")
ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RULE_OPS = {"lt", "lte", "gt", "gte", "between"}
REVIEW_ROLES = {"mechanism", "falsifier", "novelty", "provenance", "methodology"}
PROTOCOL_V2_REQUIRED_METRICS = frozenset({"seed", "training_seconds", "total_seconds"})
PROTOCOL_V2_BUDGET_KINDS = frozenset({"wall_seconds", "training_seconds"})
PROTOCOL_V2_FORBIDDEN_ENV = frozenset(
    {
        "BASH_ENV",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "ENV",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NODE_OPTIONS",
        "NODE_PATH",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "PYTHONWARNINGS",
    }
)

CLAIM_TYPES = frozenset(
    {"empirical", "mechanistic", "theoretical", "methodological", "negative_result"}
)
CLAIM_ORIGINS = frozenset({"literature", "experiment", "synthesis"})
CLAIM_STANCES = frozenset({"supports", "opposes"})
STUDY_DESIGNS = frozenset(
    {
        "meta_analysis",
        "randomized_controlled",
        "controlled_benchmark",
        "ablation",
        "observational",
        "theoretical",
        "case_study",
        "anecdotal",
        "unknown",
    }
)
ARTIFACT_STATUSES = frozenset({"verified", "available", "partial", "none", "unknown"})
REPRODUCTION_STATUSES = frozenset(
    {"independent_success", "independent_failure", "author_only", "not_attempted", "unknown"}
)
DIRECTNESS_LEVELS = frozenset({"direct", "indirect", "speculative"})
SCOPE_MATCH_LEVELS = frozenset({"exact", "close", "partial", "distant", "unknown"})
RISK_LEVELS = frozenset({"low", "medium", "high", "unknown"})
VENUE_TIERS = frozenset({"top", "selective", "peer_reviewed", "workshop", "preprint", "unknown"})


class RecordError(ValueError):
    """Raised when a record or its provenance is malformed."""


class ConflictError(RecordError):
    """Raised when immutable content would be replaced."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_object(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordError(f"{field} must be an object")
    return dict(value)


def _list(value: Any, field: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise RecordError(f"{field} must be a list")
    if nonempty and not value:
        raise RecordError(f"{field} must not be empty")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{field} must be non-empty text")
    return value.strip()


def _identifier(value: Any, field: str) -> str:
    text = _text(value, field)
    if not ID_RE.fullmatch(text):
        raise RecordError(f"{field} is not a valid identifier: {text!r}")
    return text


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RecordError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise RecordError(f"{field} must be >= {minimum}")
    return number


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordError(f"{field} must be an integer")
    if value < minimum:
        raise RecordError(f"{field} must be >= {minimum}")
    return value


def _digest(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise RecordError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _argv(value: Any, field: str) -> list[str]:
    argv = _list(value, field, nonempty=True)
    if not all(isinstance(item, str) and item for item in argv):
        raise RecordError(f"{field} must contain non-empty strings")
    return argv


def _rule(value: Any, field: str) -> dict[str, Any]:
    rule = _mapping(value, field)
    op = _text(rule.get("op"), f"{field}.op")
    if op not in RULE_OPS:
        raise RecordError(f"{field}.op must be one of {sorted(RULE_OPS)}")
    if op == "between":
        low = _number(rule.get("low"), f"{field}.low")
        high = _number(rule.get("high"), f"{field}.high")
        if low > high:
            raise RecordError(f"{field}.low must not exceed high")
    else:
        _number(rule.get("value"), f"{field}.value")
    return rule


def rule_matches(value: float, rule: Mapping[str, Any]) -> bool:
    op = str(rule["op"])
    if op == "lt":
        return value < float(rule["value"])
    if op == "lte":
        return value <= float(rule["value"])
    if op == "gt":
        return value > float(rule["value"])
    if op == "gte":
        return value >= float(rule["value"])
    if op == "between":
        return float(rule["low"]) <= value <= float(rule["high"])
    raise RecordError(f"unknown rule operator: {op}")


def _validate_protocol_metadata(value: Mapping[str, Any], prefix: str) -> int | None:
    protocol_version = value.get("protocol_version")
    if protocol_version is not None and (
        isinstance(protocol_version, bool)
        or not isinstance(protocol_version, int)
        or protocol_version != 2
    ):
        raise RecordError(f"{prefix}.protocol_version must be 2 when present")
    if "scope" in value:
        scope = _mapping(value.get("scope"), f"{prefix}.scope")
        if protocol_version == 2 and "budget" in scope:
            budget = _mapping(scope.get("budget"), f"{prefix}.scope.budget")
            if budget.get("kind") not in PROTOCOL_V2_BUDGET_KINDS:
                raise RecordError(
                    f"{prefix}.scope.budget.kind must be wall_seconds or training_seconds"
                )
            seconds = _number(
                budget.get("value"),
                f"{prefix}.scope.budget.value",
                minimum=0,
            )
            if seconds == 0:
                raise RecordError(f"{prefix}.scope.budget.value must be greater than zero")
    if "search" in value:
        search = _mapping(value.get("search"), f"{prefix}.search")
        if protocol_version == 2 and "scope" in search:
            raise RecordError(
                f"{prefix}.search.scope is forbidden; top-level scope is the only authority"
            )
        if "lane" in search:
            lane = _text(search.get("lane"), f"{prefix}.search.lane")
            if protocol_version == 2 and lane in {"candidate", "promotion"}:
                _digest(
                    search.get("context_fingerprint"),
                    f"{prefix}.search.context_fingerprint",
                )
        if "baseline" in search:
            baseline = search.get("baseline")
            if isinstance(baseline, str):
                _text(baseline, f"{prefix}.search.baseline")
            elif not isinstance(baseline, Mapping):
                raise RecordError(f"{prefix}.search.baseline must be text or an object")
        if "mutable_code_paths" in search:
            paths = _list(
                search.get("mutable_code_paths"),
                f"{prefix}.search.mutable_code_paths",
            )
            if not all(isinstance(path, str) and path.strip() for path in paths):
                raise RecordError(f"{prefix}.search.mutable_code_paths must contain text")
            if len(paths) != len(set(paths)):
                raise RecordError(f"{prefix}.search.mutable_code_paths must be unique")
        if "hypothesis_ids" in search:
            _identifiers(search.get("hypothesis_ids"), f"{prefix}.search.hypothesis_ids")
    return protocol_version


def validate_experiment_spec(payload: Mapping[str, Any]) -> None:
    spec = _mapping(payload, "experiment_spec")
    protocol_version = _validate_protocol_metadata(spec, "experiment_spec")
    stage = _text(spec.get("stage"), "experiment_spec.stage")
    if stage not in {"pilot", "confirmation"}:
        raise RecordError("experiment_spec.stage must be pilot or confirmation")
    _text(spec.get("title"), "experiment_spec.title")
    _text(spec.get("question"), "experiment_spec.question")

    mechanism = _mapping(spec.get("mechanism"), "experiment_spec.mechanism")
    _text(mechanism.get("cause"), "experiment_spec.mechanism.cause")
    _text(mechanism.get("effect"), "experiment_spec.mechanism.effect")
    chain = _list(mechanism.get("chain"), "experiment_spec.mechanism.chain", nonempty=True)
    if not all(isinstance(item, str) and item.strip() for item in chain):
        raise RecordError("experiment_spec.mechanism.chain must contain non-empty text")

    hypothesis = _mapping(spec.get("hypothesis"), "experiment_spec.hypothesis")
    _text(hypothesis.get("statement"), "experiment_spec.hypothesis.statement")
    _text(hypothesis.get("prediction"), "experiment_spec.hypothesis.prediction")
    falsifier = _mapping(spec.get("falsifier"), "experiment_spec.falsifier")
    _text(falsifier.get("statement"), "experiment_spec.falsifier.statement")

    metric = _mapping(spec.get("metric"), "experiment_spec.metric")
    _text(metric.get("name"), "experiment_spec.metric.name")
    direction = _text(metric.get("direction"), "experiment_spec.metric.direction")
    if direction not in {"minimize", "maximize"}:
        raise RecordError("experiment_spec.metric.direction must be minimize or maximize")

    plan = _list(spec.get("plan"), "experiment_spec.plan", nonempty=True)
    replicate_ids: set[str] = set()
    planned_seeds: list[int] = []
    arm_counts: list[int] = []
    expected_arm_names: list[str] | None = None
    for index, raw_replicate in enumerate(plan):
        replicate = _mapping(raw_replicate, f"experiment_spec.plan[{index}]")
        replicate_id = _identifier(
            replicate.get("replicate_id"), f"experiment_spec.plan[{index}].replicate_id"
        )
        if replicate_id in replicate_ids:
            raise RecordError(f"duplicate replicate_id: {replicate_id}")
        replicate_ids.add(replicate_id)
        planned_seed: int | None = None
        if protocol_version == 2:
            planned_seed = _integer(
                replicate.get("seed"), f"experiment_spec.plan[{index}].seed", minimum=0
            )
            planned_seeds.append(planned_seed)
        arms = _list(replicate.get("arms"), f"experiment_spec.plan[{index}].arms", nonempty=True)
        arm_counts.append(len(arms))
        arm_names: set[str] = set()
        for arm_index, raw_arm in enumerate(arms):
            arm = _mapping(raw_arm, f"experiment_spec.plan[{index}].arms[{arm_index}]")
            name = _identifier(
                arm.get("name"), f"experiment_spec.plan[{index}].arms[{arm_index}].name"
            )
            if name in arm_names:
                raise RecordError(f"duplicate arm name {name} in {replicate_id}")
            arm_names.add(name)
            _argv(arm.get("argv"), f"experiment_spec.plan[{index}].arms[{arm_index}].argv")
            env = _mapping(
                arm.get("env", {}), f"experiment_spec.plan[{index}].arms[{arm_index}].env"
            )
            if not all(isinstance(key, str) and isinstance(val, str) for key, val in env.items()):
                raise RecordError("arm env keys and values must be strings")
            if not all(ENV_RE.fullmatch(key) for key in env):
                raise RecordError("arm env keys must be valid environment variable names")
            if protocol_version == 2 and env.get("AUTORESEARCH_SEED") != str(planned_seed):
                raise RecordError(
                    f"experiment_spec.plan[{index}].arms[{arm_index}] "
                    "AUTORESEARCH_SEED must match the planned seed"
                )
            forbidden_env = sorted(set(env) & PROTOCOL_V2_FORBIDDEN_ENV)
            if protocol_version == 2 and forbidden_env:
                raise RecordError(
                    "protocol v2 arm env cannot override launcher/module loading variables: "
                    f"{forbidden_env}"
                )
        # Every replicate must contain the same SET of arms, but the ORDER may differ
        # between replicates so that arm order can be counterbalanced.
        #
        # Arms within a replicate run sequentially on one leased GPU, so a fixed order
        # makes the later arm systematically vulnerable to load that arrives during the
        # replicate.  This is not hypothetical: a confirmation on this hardware had four
        # of six replicates invalidated by a co-tenant, and the sign of the apparent
        # effect was decided by which arm was hit.  Alternating the order across
        # replicates converts that systematic bias into noise that averages out.
        #
        # Per-replicate order is still binding: the ExecutionManifest records each
        # replicate's own arm sequence and the Evidence Engine checks the executed order
        # against that replicate's plan, so this relaxation does not weaken any gate.
        current_arm_names = [str(arm["name"]) for arm in arms]
        if expected_arm_names is None:
            expected_arm_names = current_arm_names
        elif sorted(current_arm_names) != sorted(expected_arm_names):
            raise RecordError("every replicate must contain the same set of arm names")

    if len(set(arm_counts)) != 1:
        raise RecordError("every replicate must contain the same number of arms")
    if stage == "pilot" and (len(plan) > 1 or arm_counts[0] > 2):
        raise RecordError("a pilot is limited to one replicate and at most two arms")

    analysis = _mapping(spec.get("analysis"), "experiment_spec.analysis")
    effect = _text(analysis.get("effect"), "experiment_spec.analysis.effect")
    if effect not in {"single", "difference", "ratio"}:
        raise RecordError("experiment_spec.analysis.effect must be single, difference, or ratio")
    primary_arm = _identifier(analysis.get("primary_arm"), "experiment_spec.analysis.primary_arm")
    arm_names = {str(arm["name"]) for arm in plan[0]["arms"]}
    if primary_arm not in arm_names:
        raise RecordError("analysis.primary_arm must name an arm in every replicate")
    if effect != "single":
        reference_arm = _identifier(
            analysis.get("reference_arm"), "experiment_spec.analysis.reference_arm"
        )
        if reference_arm not in arm_names or reference_arm == primary_arm:
            raise RecordError("analysis.reference_arm must name a different arm")
    minimum = _integer(
        analysis.get("minimum_valid_replicates"),
        "experiment_spec.analysis.minimum_valid_replicates",
        minimum=1,
    )
    if minimum > len(plan):
        raise RecordError("minimum_valid_replicates exceeds the preregistered plan")
    _rule(analysis.get("success_rule"), "experiment_spec.analysis.success_rule")
    _rule(analysis.get("falsifier_rule"), "experiment_spec.analysis.falsifier_rule")
    if not isinstance(analysis.get("sota_eligible", False), bool):
        raise RecordError("experiment_spec.analysis.sota_eligible must be boolean")
    if protocol_version == 2 and stage == "confirmation" and analysis.get("sota_eligible", False):
        search = _mapping(spec.get("search", {}), "experiment_spec.search")
        if search.get("lane") != "promotion":
            raise RecordError("protocol v2 SOTA confirmations require search.lane=promotion")
        if minimum < 3:
            raise RecordError("protocol v2 SOTA confirmations require at least 3 valid replicates")
        if len(planned_seeds) != len(set(planned_seeds)):
            raise RecordError("protocol v2 SOTA confirmation seeds must be distinct")

    requirements = _mapping(spec.get("requirements"), "experiment_spec.requirements")
    required_metrics = _list(
        requirements.get("required_metrics"),
        "experiment_spec.requirements.required_metrics",
        nonempty=True,
    )
    if metric["name"] not in required_metrics:
        raise RecordError("the primary metric must be listed in required_metrics")
    if not all(isinstance(item, str) and item for item in required_metrics):
        raise RecordError("requirements.required_metrics must contain strings")
    if protocol_version == 2:
        missing_v2_metrics = sorted(PROTOCOL_V2_REQUIRED_METRICS - set(required_metrics))
        if missing_v2_metrics:
            raise RecordError(
                "protocol v2 requires seed and timing metrics in "
                f"requirements.required_metrics; missing {missing_v2_metrics}"
            )
    _number(requirements.get("minimum_steps", 0), "requirements.minimum_steps", minimum=0)
    if not isinstance(requirements.get("require_gpu", True), bool):
        raise RecordError("requirements.require_gpu must be boolean")
    isolation = _text(requirements.get("isolation", "launch"), "requirements.isolation")
    if isolation not in {"none", "launch", "continuous"}:
        raise RecordError("requirements.isolation must be none, launch, or continuous")

    knowledge = _mapping(spec.get("knowledge"), "experiment_spec.knowledge")
    _text(knowledge.get("direction"), "experiment_spec.knowledge.direction")
    _text(knowledge.get("subsystem"), "experiment_spec.knowledge.subsystem")
    source_ids = _list(
        knowledge.get("source_ids", []), "experiment_spec.knowledge.source_ids", nonempty=True
    )
    if not all(isinstance(item, str) and item for item in source_ids):
        raise RecordError("knowledge.source_ids must contain strings")
    _text(spec.get("comparison_group"), "experiment_spec.comparison_group")


def validate_execution_manifest(payload: Mapping[str, Any]) -> None:
    manifest = _mapping(payload, "execution_manifest")
    protocol_version = _validate_protocol_metadata(manifest, "execution_manifest")
    _identifier(manifest.get("spec_id"), "execution_manifest.spec_id")
    _digest(manifest.get("spec_digest"), "execution_manifest.spec_digest")
    if manifest.get("stage") not in {"pilot", "confirmation"}:
        raise RecordError("execution_manifest.stage is invalid")
    plan = _list(manifest.get("plan"), "execution_manifest.plan", nonempty=True)
    planned_seeds: list[int] = []
    for replicate_index, raw_replicate in enumerate(plan):
        replicate = _mapping(raw_replicate, f"execution_manifest.plan[{replicate_index}]")
        _identifier(
            replicate.get("replicate_id"),
            f"execution_manifest.plan[{replicate_index}].replicate_id",
        )
        planned_seed: int | None = None
        if protocol_version == 2:
            planned_seed = _integer(
                replicate.get("seed"),
                f"execution_manifest.plan[{replicate_index}].seed",
                minimum=0,
            )
            planned_seeds.append(planned_seed)
        arms = _list(
            replicate.get("arms"),
            f"execution_manifest.plan[{replicate_index}].arms",
            nonempty=True,
        )
        for arm_index, raw_arm in enumerate(arms):
            arm = _mapping(raw_arm, f"execution_manifest.plan[{replicate_index}].arms[{arm_index}]")
            _identifier(
                arm.get("name"),
                f"execution_manifest.plan[{replicate_index}].arms[{arm_index}].name",
            )
            _argv(
                arm.get("argv"),
                f"execution_manifest.plan[{replicate_index}].arms[{arm_index}].argv",
            )
            env = _mapping(
                arm.get("env", {}),
                f"execution_manifest.plan[{replicate_index}].arms[{arm_index}].env",
            )
            if not all(
                isinstance(key, str) and isinstance(value, str) for key, value in env.items()
            ):
                raise RecordError("manifest arm env keys and values must be strings")
            if not all(ENV_RE.fullmatch(key) for key in env):
                raise RecordError("manifest arm env keys must be valid environment variable names")
            if protocol_version == 2 and env.get("AUTORESEARCH_SEED") != str(planned_seed):
                raise RecordError(
                    f"execution_manifest.plan[{replicate_index}].arms[{arm_index}] "
                    "AUTORESEARCH_SEED must match the planned seed"
                )
            forbidden_env = sorted(set(env) & PROTOCOL_V2_FORBIDDEN_ENV)
            if protocol_version == 2 and forbidden_env:
                raise RecordError(
                    "protocol v2 arm env cannot override launcher/module loading variables: "
                    f"{forbidden_env}"
                )
    for group in ("code_bindings", "data_bindings"):
        for index, raw_binding in enumerate(_list(manifest.get(group, []), group, nonempty=True)):
            binding = _mapping(raw_binding, f"{group}[{index}]")
            _digest(binding.get("sha256"), f"{group}[{index}].sha256")
            _text(binding.get("blob"), f"{group}[{index}].blob")
            _text(binding.get("execution_path"), f"{group}[{index}].execution_path")
    resources = _list(manifest.get("resources"), "execution_manifest.resources", nonempty=True)
    for index, raw_resource in enumerate(resources):
        resource = _mapping(raw_resource, f"execution_manifest.resources[{index}]")
        _text(resource.get("id"), f"execution_manifest.resources[{index}].id")
        if "host_id" in resource:
            host_id = _text(
                resource.get("host_id"), f"execution_manifest.resources[{index}].host_id"
            )
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", host_id):
                raise RecordError("resource host_id contains unsupported characters")
        if resource.get("backend") not in {"local", "ssh"}:
            raise RecordError(f"execution_manifest.resources[{index}].backend is invalid")
        _text(resource.get("workdir"), f"execution_manifest.resources[{index}].workdir")
        if resource.get("backend") == "ssh":
            _argv(
                resource.get("ssh_argv"),
                f"execution_manifest.resources[{index}].ssh_argv",
            )
        for launcher_field in ("python",):
            if launcher_field in resource:
                launcher = _text(
                    resource.get(launcher_field),
                    f"execution_manifest.resources[{index}].{launcher_field}",
                )
                if not any(separator in launcher for separator in ("/", "\\")):
                    raise RecordError("resource python launcher must be an explicit path")
        if "trusted_launchers" in resource:
            launchers = _list(
                resource.get("trusted_launchers"),
                f"execution_manifest.resources[{index}].trusted_launchers",
            )
            if not all(
                isinstance(launcher, str)
                and launcher
                and any(separator in launcher for separator in ("/", "\\"))
                and ".." not in Path(launcher).parts
                for launcher in launchers
            ):
                raise RecordError("resource trusted_launchers must contain explicit safe paths")
        gpus = _list(resource.get("gpus", []), f"execution_manifest.resources[{index}].gpus")
        if not all(isinstance(gpu, int) and not isinstance(gpu, bool) and gpu >= 0 for gpu in gpus):
            raise RecordError("resource GPU indices must be non-negative integers")
        if len(gpus) != len(set(gpus)):
            raise RecordError("resource GPU indices must be unique")
        if "max_concurrent_jobs" in resource:
            _integer(
                resource.get("max_concurrent_jobs"),
                f"execution_manifest.resources[{index}].max_concurrent_jobs",
                minimum=1,
            )
        if "reservation" in resource:
            reservation = _mapping(
                resource.get("reservation"),
                f"execution_manifest.resources[{index}].reservation",
            )
            mode = reservation.get("mode")
            if mode not in {"shared", "externally_reserved"}:
                raise RecordError("resource reservation mode is invalid")
            reservation_id = reservation.get("id")
            if reservation_id is not None:
                _text(
                    reservation_id,
                    f"execution_manifest.resources[{index}].reservation.id",
                )
            if mode == "externally_reserved" and not reservation_id:
                raise RecordError("externally_reserved resources require a reservation id")
        _number(
            resource.get("max_idle_memory_mb", 512),
            f"execution_manifest.resources[{index}].max_idle_memory_mb",
            minimum=0,
        )
        utilization = _number(
            resource.get("max_idle_utilization_percent", 5),
            f"execution_manifest.resources[{index}].max_idle_utilization_percent",
            minimum=0,
        )
        if utilization > 100:
            raise RecordError("max_idle_utilization_percent must not exceed 100")
    runtime = _mapping(manifest.get("runtime"), "execution_manifest.runtime")
    timeout = _number(
        runtime.get("timeout_seconds_per_arm"),
        "execution_manifest.runtime.timeout_seconds_per_arm",
        minimum=0,
    )
    interval = _number(
        runtime.get("telemetry_interval_seconds"),
        "execution_manifest.runtime.telemetry_interval_seconds",
        minimum=0,
    )
    _number(
        runtime.get("resource_wait_seconds"),
        "execution_manifest.runtime.resource_wait_seconds",
        minimum=0,
    )
    if timeout == 0 or interval == 0:
        raise RecordError("execution timeout and telemetry interval must be greater than zero")
    reviews = _list(manifest.get("reviews", []), "execution_manifest.reviews")
    roles: set[str] = set()
    reviewers: set[str] = set()
    sessions: set[str] = set()
    for index, raw_review in enumerate(reviews):
        review = _mapping(raw_review, f"execution_manifest.reviews[{index}]")
        role = _text(review.get("role"), f"execution_manifest.reviews[{index}].role")
        reviewer = _text(
            review.get("reviewer_id"), f"execution_manifest.reviews[{index}].reviewer_id"
        )
        session = _text(review.get("session_id"), f"execution_manifest.reviews[{index}].session_id")
        if role not in REVIEW_ROLES or review.get("decision") != "approve":
            raise RecordError("manifest reviews must have a known role and approve decision")
        _text(review.get("reviewed_at"), f"execution_manifest.reviews[{index}].reviewed_at")
        if role in roles or reviewer in reviewers or session in sessions:
            raise RecordError("manifest review roles, reviewers, and sessions must be unique")
        roles.add(role)
        reviewers.add(reviewer)
        sessions.add(session)
    if manifest.get("stage") == "confirmation" and protocol_version != 2 and roles != REVIEW_ROLES:
        raise RecordError("confirmation manifest must contain the complete review council")
    requirements = _mapping(manifest.get("requirements"), "execution_manifest.requirements")
    _mapping(manifest.get("metric"), "execution_manifest.metric")
    analysis = _mapping(manifest.get("analysis"), "execution_manifest.analysis")
    if protocol_version == 2:
        required_metrics = _list(
            requirements.get("required_metrics"),
            "execution_manifest.requirements.required_metrics",
            nonempty=True,
        )
        missing_v2_metrics = sorted(PROTOCOL_V2_REQUIRED_METRICS - set(required_metrics))
        if missing_v2_metrics:
            raise RecordError(
                "protocol v2 requires seed and timing metrics in "
                f"requirements.required_metrics; missing {missing_v2_metrics}"
            )
        if manifest.get("stage") == "confirmation" and analysis.get("sota_eligible", False):
            search = _mapping(manifest.get("search", {}), "execution_manifest.search")
            if search.get("lane") != "promotion":
                raise RecordError("protocol v2 SOTA confirmations require search.lane=promotion")
            minimum = _integer(
                analysis.get("minimum_valid_replicates"),
                "execution_manifest.analysis.minimum_valid_replicates",
                minimum=1,
            )
            if minimum < 3:
                raise RecordError(
                    "protocol v2 SOTA confirmations require at least 3 valid replicates"
                )
            if len(planned_seeds) != len(set(planned_seeds)):
                raise RecordError("protocol v2 SOTA confirmation seeds must be distinct")
    _text(manifest.get("comparison_group"), "execution_manifest.comparison_group")


def validate_result_bundle(payload: Mapping[str, Any]) -> None:
    result = _mapping(payload, "result_bundle")
    _identifier(result.get("manifest_id"), "result_bundle.manifest_id")
    _digest(result.get("manifest_digest"), "result_bundle.manifest_digest")
    _identifier(result.get("spec_id"), "result_bundle.spec_id")
    _identifier(result.get("replicate_id"), "result_bundle.replicate_id")
    if result.get("stage") not in {"pilot", "confirmation"}:
        raise RecordError("result_bundle.stage is invalid")
    if result.get("status") not in {
        "completed",
        "failed",
        "timed_out",
        "preflight_failed",
        "partial",
    }:
        raise RecordError("result_bundle.status is invalid")
    lifecycle = _mapping(result.get("lifecycle"), "result_bundle.lifecycle")
    for field in ("claimed_at", "started_at", "ended_at"):
        _text(lifecycle.get(field), f"result_bundle.lifecycle.{field}")
    _mapping(result.get("resource"), "result_bundle.resource")
    _mapping(result.get("environment"), "result_bundle.environment")
    _mapping(result.get("launch_telemetry"), "result_bundle.launch_telemetry")
    for group in ("binding_checks", "post_binding_checks"):
        for index, raw_check in enumerate(_list(result.get(group), f"result_bundle.{group}")):
            check = _mapping(raw_check, f"result_bundle.{group}[{index}]")
            if check.get("state") not in {"verified", "mismatch", "unknown"}:
                raise RecordError(f"result_bundle.{group}[{index}].state is invalid")
            _digest(check.get("expected_sha256"), f"result_bundle.{group}[{index}].expected")
    arms = _list(result.get("arms"), "result_bundle.arms")
    if result.get("status") == "completed" and not arms:
        raise RecordError("a completed ResultBundle must contain arms")
    for index, raw_arm in enumerate(arms):
        arm = _mapping(raw_arm, f"result_bundle.arms[{index}]")
        _identifier(arm.get("name"), f"result_bundle.arms[{index}].name")
        _argv(arm.get("payload_argv"), f"result_bundle.arms[{index}].payload_argv")
        env = _mapping(arm.get("payload_env"), f"result_bundle.arms[{index}].payload_env")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
            raise RecordError("result arm environment keys and values must be strings")
        _argv(arm.get("runner_argv"), f"result_bundle.arms[{index}].runner_argv")
        if arm.get("status") not in {"completed", "failed", "timed_out"}:
            raise RecordError(f"result_bundle.arms[{index}].status is invalid")
        metrics = _mapping(arm.get("metrics"), f"result_bundle.arms[{index}].metrics")
        for name, value in metrics.items():
            _number(value, f"result_bundle.arms[{index}].metrics.{name}")
        artifacts = _mapping(arm.get("artifacts"), f"result_bundle.arms[{index}].artifacts")
        for name in ("stdout", "stderr"):
            _text(artifacts.get(name), f"result_bundle.arms[{index}].artifacts.{name}")
            _digest(
                artifacts.get(f"{name}_sha256"),
                f"result_bundle.arms[{index}].artifacts.{name}_sha256",
            )


def validate_evidence_decision(payload: Mapping[str, Any]) -> None:
    evidence = _mapping(payload, "evidence_decision")
    _identifier(evidence.get("result_id"), "evidence_decision.result_id")
    _digest(evidence.get("result_digest"), "evidence_decision.result_digest")
    _identifier(evidence.get("spec_id"), "evidence_decision.spec_id")
    if evidence.get("measurement_verdict") not in {"valid", "invalid", "unknown"}:
        raise RecordError("evidence_decision.measurement_verdict is invalid")
    if evidence.get("claim_status") not in {"eligible", "ineligible", "blocked"}:
        raise RecordError("evidence_decision.claim_status is invalid")
    _list(evidence.get("reasons"), "evidence_decision.reasons")
    _mapping(evidence.get("measurements"), "evidence_decision.measurements")
    policy_version = _text(evidence.get("policy_version"), "evidence_decision.policy_version")
    if policy_version in {"evidence-v2", "evidence-v3"}:
        verified_seed = evidence.get("verified_seed")
        if verified_seed is not None:
            _integer(verified_seed, "evidence_decision.verified_seed", minimum=0)
        decision_gpu_key = evidence.get("gpu_key")
        if decision_gpu_key is not None:
            _text(decision_gpu_key, "evidence_decision.gpu_key")


def validate_paper(payload: Mapping[str, Any]) -> None:
    paper = _mapping(payload, "paper")
    _text(paper.get("title"), "paper.title")
    _text(paper.get("path"), "paper.path")
    _digest(paper.get("content_sha256"), "paper.content_sha256")
    _text(paper.get("blob"), "paper.blob")
    spec_ids = _list(paper.get("spec_ids"), "paper.spec_ids", nonempty=True)
    evidence_ids = _list(paper.get("evidence_ids", []), "paper.evidence_ids")
    for index, spec_id in enumerate(spec_ids):
        _identifier(spec_id, f"paper.spec_ids[{index}]")
    for index, evidence_id in enumerate(evidence_ids):
        _identifier(evidence_id, f"paper.evidence_ids[{index}]")


def _identifiers(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    rows = _list(value, field, nonempty=nonempty)
    result = [_identifier(item, f"{field}[{index}]") for index, item in enumerate(rows)]
    if len(result) != len(set(result)):
        raise RecordError(f"{field} must be unique")
    return result


def _texts(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    rows = _list(value, field, nonempty=nonempty)
    result = [_text(item, f"{field}[{index}]") for index, item in enumerate(rows)]
    if len(result) != len(set(result)):
        raise RecordError(f"{field} must be unique")
    return result


def validate_research_agenda(payload: Mapping[str, Any]) -> None:
    agenda = _mapping(payload, "research_agenda")
    _text(agenda.get("name"), "research_agenda.name")
    _text(agenda.get("objective"), "research_agenda.objective")
    _mapping(agenda.get("scope"), "research_agenda.scope")
    if agenda.get("confidence_policy_version") != "confidence-v1":
        raise RecordError("research_agenda.confidence_policy_version must be confidence-v1")
    topics = _list(agenda.get("topics"), "research_agenda.topics", nonempty=True)
    topic_ids: set[str] = set()
    for index, raw_topic in enumerate(topics):
        topic = _mapping(raw_topic, f"research_agenda.topics[{index}]")
        topic_id = _identifier(topic.get("id"), f"research_agenda.topics[{index}].id")
        if topic_id in topic_ids:
            raise RecordError("research_agenda topic ids must be unique")
        topic_ids.add(topic_id)
        _text(topic.get("question"), f"research_agenda.topics[{index}].question")
        _texts(topic.get("queries"), f"research_agenda.topics[{index}].queries", nonempty=True)
        _texts(topic.get("keywords", []), f"research_agenda.topics[{index}].keywords")
        _integer(
            topic.get("refresh_days", 14),
            f"research_agenda.topics[{index}].refresh_days",
            minimum=1,
        )
        _integer(
            topic.get("minimum_sources", 5),
            f"research_agenda.topics[{index}].minimum_sources",
            minimum=1,
        )
        _integer(
            topic.get("minimum_claims", 3),
            f"research_agenda.topics[{index}].minimum_claims",
            minimum=1,
        )


def validate_literature_search(payload: Mapping[str, Any]) -> None:
    search = _mapping(payload, "literature_search")
    if search.get("provider") not in {"openalex", "manual", "semantic_scholar", "arxiv"}:
        raise RecordError("literature_search.provider is invalid")
    _text(search.get("query"), "literature_search.query")
    _text(search.get("searched_at"), "literature_search.searched_at")
    if "agenda_id" in search:
        _identifier(search.get("agenda_id"), "literature_search.agenda_id")
    if "topic_id" in search:
        _identifier(search.get("topic_id"), "literature_search.topic_id")
    _mapping(search.get("filters", {}), "literature_search.filters")
    _identifiers(search.get("result_source_ids", []), "literature_search.result_source_ids")
    _integer(search.get("raw_result_count", 0), "literature_search.raw_result_count")
    if "request_url" in search:
        _text(search.get("request_url"), "literature_search.request_url")


def validate_literature_source(payload: Mapping[str, Any]) -> None:
    source = _mapping(payload, "literature_source")
    _identifier(source.get("work_key"), "literature_source.work_key")
    _text(source.get("title"), "literature_source.title")
    _texts(source.get("authors", []), "literature_source.authors")
    year = source.get("year")
    if year is not None:
        _integer(year, "literature_source.year", minimum=1600)
    venue = _mapping(source.get("venue"), "literature_source.venue")
    _text(venue.get("name", "unknown"), "literature_source.venue.name")
    if venue.get("tier") not in VENUE_TIERS:
        raise RecordError(f"literature_source.venue.tier must be one of {sorted(VENUE_TIERS)}")
    if venue.get("peer_reviewed") not in {"yes", "no", "unknown"}:
        raise RecordError("literature_source.venue.peer_reviewed must be yes, no, or unknown")
    _text(source.get("publication_type"), "literature_source.publication_type")
    identifiers = _mapping(source.get("identifiers", {}), "literature_source.identifiers")
    if not all(
        isinstance(key, str) and isinstance(value, str) for key, value in identifiers.items()
    ):
        raise RecordError("literature_source.identifiers must map strings to strings")
    _texts(source.get("urls", []), "literature_source.urls")
    if "abstract" in source and not isinstance(source.get("abstract"), str):
        raise RecordError("literature_source.abstract must be text")
    _identifiers(source.get("topics", []), "literature_source.topics")
    retrieval = _mapping(source.get("retrieval"), "literature_source.retrieval")
    _text(retrieval.get("provider"), "literature_source.retrieval.provider")
    _text(retrieval.get("retrieved_at"), "literature_source.retrieval.retrieved_at")
    if "query" in retrieval:
        _text(retrieval.get("query"), "literature_source.retrieval.query")
    if "search_id" in retrieval:
        _identifier(retrieval.get("search_id"), "literature_source.retrieval.search_id")
    content = _mapping(source.get("content"), "literature_source.content")
    if content.get("status") not in {"metadata_only", "abstract_only", "fulltext_snapshot"}:
        raise RecordError("literature_source.content.status is invalid")
    if content.get("status") == "fulltext_snapshot":
        _digest(content.get("sha256"), "literature_source.content.sha256")
        _text(content.get("blob"), "literature_source.content.blob")
        _text(content.get("mime_type"), "literature_source.content.mime_type")
    if "citation_count" in source:
        _integer(source.get("citation_count"), "literature_source.citation_count")


def validate_scientific_claim(payload: Mapping[str, Any]) -> None:
    claim = _mapping(payload, "scientific_claim")
    _identifier(claim.get("belief_key"), "scientific_claim.belief_key")
    _text(claim.get("statement"), "scientific_claim.statement")
    if claim.get("claim_type") not in CLAIM_TYPES:
        raise RecordError(f"scientific_claim.claim_type must be one of {sorted(CLAIM_TYPES)}")
    if claim.get("origin") not in CLAIM_ORIGINS:
        raise RecordError(f"scientific_claim.origin must be one of {sorted(CLAIM_ORIGINS)}")
    if claim.get("stance") not in CLAIM_STANCES:
        raise RecordError(f"scientific_claim.stance must be one of {sorted(CLAIM_STANCES)}")
    _mapping(claim.get("scope"), "scientific_claim.scope")
    source_ids = _identifiers(claim.get("source_ids", []), "scientific_claim.source_ids")
    evidence_ids = _identifiers(claim.get("evidence_ids", []), "scientific_claim.evidence_ids")
    derived = _identifiers(
        claim.get("derived_from_claim_ids", []), "scientific_claim.derived_from_claim_ids"
    )
    if not source_ids and not evidence_ids and not derived:
        raise RecordError(
            "scientific_claim requires literature, experiment, or derived-claim provenance"
        )
    if claim.get("origin") == "literature" and not source_ids:
        raise RecordError("a literature claim requires source_ids")
    if claim.get("origin") == "experiment" and not evidence_ids:
        raise RecordError("an experiment claim requires evidence_ids")
    _identifiers(claim.get("topics", []), "scientific_claim.topics")
    locators = _list(claim.get("locators", []), "scientific_claim.locators")
    for index, raw_locator in enumerate(locators):
        locator = _mapping(raw_locator, f"scientific_claim.locators[{index}]")
        _identifier(locator.get("source_id"), f"scientific_claim.locators[{index}].source_id")
        _text(locator.get("location"), f"scientific_claim.locators[{index}].location")
        if "excerpt" in locator:
            _text(locator.get("excerpt"), f"scientific_claim.locators[{index}].excerpt")
    evidence = _mapping(claim.get("evidence"), "scientific_claim.evidence")
    enum_fields = {
        "study_design": STUDY_DESIGNS,
        "artifact_status": ARTIFACT_STATUSES,
        "reproduction_status": REPRODUCTION_STATUSES,
        "directness": DIRECTNESS_LEVELS,
        "scope_match": SCOPE_MATCH_LEVELS,
        "risk_of_bias": RISK_LEVELS,
    }
    for field, allowed in enum_fields.items():
        if evidence.get(field) not in allowed:
            raise RecordError(f"scientific_claim.evidence.{field} must be one of {sorted(allowed)}")
    metrics = _list(evidence.get("metrics", []), "scientific_claim.evidence.metrics")
    for index, raw_metric in enumerate(metrics):
        metric = _mapping(raw_metric, f"scientific_claim.evidence.metrics[{index}]")
        _text(metric.get("name"), f"scientific_claim.evidence.metrics[{index}].name")
        _text(metric.get("result"), f"scientific_claim.evidence.metrics[{index}].result")
    assessment = _mapping(claim.get("assessment"), "scientific_claim.assessment")
    _text(assessment.get("assessor"), "scientific_claim.assessment.assessor")
    _text(assessment.get("assessed_at"), "scientific_claim.assessment.assessed_at")
    _text(assessment.get("rationale"), "scientific_claim.assessment.rationale")
    if assessment.get("confidence_policy_version") != "confidence-v1":
        raise RecordError("scientific_claim assessment must use confidence-v1")


def validate_scientific_mechanism(payload: Mapping[str, Any]) -> None:
    mechanism = _mapping(payload, "scientific_mechanism")
    _text(mechanism.get("name"), "scientific_mechanism.name")
    _text(mechanism.get("statement"), "scientific_mechanism.statement")
    _mapping(mechanism.get("scope"), "scientific_mechanism.scope")
    nodes = _list(mechanism.get("nodes"), "scientific_mechanism.nodes", nonempty=True)
    node_ids: set[str] = set()
    for index, raw_node in enumerate(nodes):
        node = _mapping(raw_node, f"scientific_mechanism.nodes[{index}]")
        node_id = _identifier(node.get("id"), f"scientific_mechanism.nodes[{index}].id")
        if node_id in node_ids:
            raise RecordError("scientific_mechanism node ids must be unique")
        node_ids.add(node_id)
        _text(node.get("label"), f"scientific_mechanism.nodes[{index}].label")
    edges = _list(mechanism.get("edges"), "scientific_mechanism.edges", nonempty=True)
    edge_claims: set[str] = set()
    for index, raw_edge in enumerate(edges):
        edge = _mapping(raw_edge, f"scientific_mechanism.edges[{index}]")
        source = _identifier(edge.get("from"), f"scientific_mechanism.edges[{index}].from")
        target = _identifier(edge.get("to"), f"scientific_mechanism.edges[{index}].to")
        if source not in node_ids or target not in node_ids or source == target:
            raise RecordError("scientific_mechanism edges must connect different declared nodes")
        _text(edge.get("relation"), f"scientific_mechanism.edges[{index}].relation")
        edge_claims.update(
            _identifiers(
                edge.get("claim_ids", []),
                f"scientific_mechanism.edges[{index}].claim_ids",
                nonempty=True,
            )
        )
    source_claims = set(
        _identifiers(
            mechanism.get("source_claim_ids"),
            "scientific_mechanism.source_claim_ids",
            nonempty=True,
        )
    )
    if not edge_claims.issubset(source_claims):
        raise RecordError("every mechanism edge claim must be in source_claim_ids")
    assumptions = _list(mechanism.get("assumptions", []), "scientific_mechanism.assumptions")
    for index, raw_assumption in enumerate(assumptions):
        assumption = _mapping(raw_assumption, f"scientific_mechanism.assumptions[{index}]")
        _text(assumption.get("statement"), f"scientific_mechanism.assumptions[{index}].statement")
        _identifiers(
            assumption.get("claim_ids", []),
            f"scientific_mechanism.assumptions[{index}].claim_ids",
        )
    _texts(mechanism.get("alternatives", []), "scientific_mechanism.alternatives")
    _texts(mechanism.get("predictions"), "scientific_mechanism.predictions", nonempty=True)
    _texts(mechanism.get("falsifiers"), "scientific_mechanism.falsifiers", nonempty=True)


def validate_scientific_hypothesis(payload: Mapping[str, Any]) -> None:
    hypothesis = _mapping(payload, "scientific_hypothesis")
    _identifier(hypothesis.get("belief_key"), "scientific_hypothesis.belief_key")
    _text(hypothesis.get("statement"), "scientific_hypothesis.statement")
    _text(hypothesis.get("rationale"), "scientific_hypothesis.rationale")
    _mapping(hypothesis.get("scope"), "scientific_hypothesis.scope")
    _identifiers(hypothesis.get("topics"), "scientific_hypothesis.topics", nonempty=True)
    _identifiers(
        hypothesis.get("mechanism_ids"), "scientific_hypothesis.mechanism_ids", nonempty=True
    )
    _identifiers(hypothesis.get("claim_ids"), "scientific_hypothesis.claim_ids", nonempty=True)
    prediction = _mapping(hypothesis.get("prediction"), "scientific_hypothesis.prediction")
    _text(prediction.get("metric"), "scientific_hypothesis.prediction.metric")
    if prediction.get("direction") not in {"increase", "decrease", "non_monotonic", "no_change"}:
        raise RecordError("scientific_hypothesis.prediction.direction is invalid")
    _text(prediction.get("statement"), "scientific_hypothesis.prediction.statement")
    if "minimum_effect" in prediction:
        _number(prediction.get("minimum_effect"), "scientific_hypothesis.prediction.minimum_effect")
    _texts(hypothesis.get("falsifiers"), "scientific_hypothesis.falsifiers", nonempty=True)
    intervention = _mapping(hypothesis.get("intervention"), "scientific_hypothesis.intervention")
    _text(intervention.get("summary"), "scientific_hypothesis.intervention.summary")
    _text(intervention.get("subsystem"), "scientific_hypothesis.intervention.subsystem")
    _texts(
        intervention.get("mutable_code_paths", []),
        "scientific_hypothesis.intervention.mutable_code_paths",
    )
    _texts(intervention.get("diagnostics", []), "scientific_hypothesis.intervention.diagnostics")
    _identifiers(
        hypothesis.get("competing_hypothesis_ids", []),
        "scientific_hypothesis.competing_hypothesis_ids",
    )
    _text(hypothesis.get("proposed_by"), "scientific_hypothesis.proposed_by")


VALIDATORS: dict[str, Callable[[Mapping[str, Any]], None]] = {
    "experiment_spec": validate_experiment_spec,
    "execution_manifest": validate_execution_manifest,
    "result_bundle": validate_result_bundle,
    "evidence_decision": validate_evidence_decision,
    "paper": validate_paper,
    "research_agenda": validate_research_agenda,
    "literature_search": validate_literature_search,
    "literature_source": validate_literature_source,
    "scientific_claim": validate_scientific_claim,
    "scientific_mechanism": validate_scientific_mechanism,
    "scientific_hypothesis": validate_scientific_hypothesis,
}


def make_record(
    kind: str,
    record_id: str,
    payload: Mapping[str, Any],
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise RecordError(f"unknown record kind: {kind}")
    _identifier(record_id, f"{kind}.id")
    normalized = json.loads(canonical_json(dict(payload)))
    VALIDATORS[kind](normalized)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "id": record_id,
        "created_at": created_at or utc_now(),
        "payload": normalized,
    }
    envelope["digest"] = sha256_object(envelope)
    return envelope


def verify_record(record: Mapping[str, Any], *, expected_kind: str | None = None) -> None:
    envelope = _mapping(record, "record")
    if envelope.get("schema_version") != SCHEMA_VERSION:
        raise RecordError("unsupported schema_version")
    kind = _text(envelope.get("kind"), "record.kind")
    if kind not in KINDS:
        raise RecordError(f"unknown record kind: {kind}")
    if expected_kind and kind != expected_kind:
        raise RecordError(f"expected {expected_kind}, found {kind}")
    _identifier(envelope.get("id"), "record.id")
    _text(envelope.get("created_at"), "record.created_at")
    payload = _mapping(envelope.get("payload"), "record.payload")
    VALIDATORS[kind](payload)
    expected = sha256_object({key: value for key, value in envelope.items() if key != "digest"})
    if envelope.get("digest") != expected:
        raise RecordError(f"record {envelope.get('id')} digest mismatch")


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_DOTENV_LOADED = False


def _load_dotenv() -> None:
    """Load repo-root .env into the environment once, without overriding real vars.

    Execution configs name a host, user, port and key path. Those are deployment
    facts, not science, and publishing them in a shared repository exposes an SSH
    endpoint for no benefit. Keeping them in a gitignored .env lets the committed
    config describe the *shape* of the fleet while each operator supplies their own
    box. Existing environment variables win, so CI and one-off overrides still work.
    """

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    dotenv = Path(__file__).resolve().parent.parent / ".env"
    try:
        lines = dotenv.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} in every string of a loaded JSON document.

    Deliberately strict: an undefined variable raises rather than expanding to an
    empty string, because a silently blank ssh host produces a confusing failure
    deep inside the runner instead of a clear one at config load.
    """

    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            resolved = os.environ.get(name)
            if resolved is None:
                raise RecordError(
                    f"config references ${{{name}}} but it is not set; "
                    "copy .env.example to .env and fill it in"
                )
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordError(f"cannot read JSON from {path}: {exc}") from exc
    _load_dotenv()
    return _mapping(_expand_env(value), str(path))


def mean(values: Sequence[float]) -> float:
    if not values:
        raise RecordError("cannot compute a mean of no values")
    return sum(values) / len(values)
