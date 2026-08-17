"""The small, explicit scientific contract used by the v2 search lane.

V1 encoded the benchmark mostly in a comparison-group string and in command-line
flags.  That made two runs look comparable even when their shard split, tokenizer,
evaluator, precision, or compute frame differed.  V2 makes those fields data.

The low-level record API remains able to read legacy v1-style specs.  Every spec
created by the v2 campaign commands, however, carries this complete scope and an
executable seed contract.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .records import RecordError, canonical_json, sha256_object

PROTOCOL_VERSION = 2
EVIDENCE_POLICY_VERSION = "evidence-v3"
SEARCH_SEED = 42
PROMOTION_SEEDS = (43, 44, 45, 46, 47)
PROMOTION_CODE_ROOT = "__autoresearch_v2__"
PRIMARY_METRIC = "val_bpb"
SEARCH_GATE = 0.000426
SCOPE_BUDGET_KINDS = frozenset({"wall_seconds", "training_seconds"})

# These are intentionally required in the fast lane.  `total_seconds -
# training_seconds` is the cheapest universally useful indication that data loading,
# compilation, or evaluation -- rather than the model -- owns the frame.
CORE_METRICS = (
    PRIMARY_METRIC,
    "num_steps",
    "seed",
    "training_seconds",
    "total_seconds",
)

SYSTEM_TRUSTED_LAUNCHERS = frozenset(
    {"/bin/bash", "/bin/sh", "/bin/zsh", "/usr/bin/bash", "/usr/bin/sh", "/usr/bin/zsh"}
)
DEFAULT_TRUSTED_LAUNCHERS = SYSTEM_TRUSTED_LAUNCHERS | {str(Path(sys.executable))}


def _explicit_launcher(value: Any) -> str:
    launcher = str(value).strip()
    parts = Path(launcher).parts
    if (
        not launcher
        or not any(separator in launcher for separator in ("/", "\\"))
        or ".." in parts
        or launcher.endswith(("/", "\\"))
    ):
        raise RecordError("trusted launchers must be explicit absolute or workdir-relative paths")
    return launcher


def trusted_launchers_for_resources(resources: Any) -> set[str]:
    """Launchers explicitly trusted on every resource a manifest may select."""

    if not isinstance(resources, list) or not resources:
        return set(DEFAULT_TRUSTED_LAUNCHERS)
    common: set[str] | None = None
    for raw in resources:
        if not isinstance(raw, Mapping):
            raise RecordError("execution resources must contain objects")
        allowed = set(SYSTEM_TRUSTED_LAUNCHERS)
        if raw.get("backend") == "local":
            allowed.add(str(Path(sys.executable)))
        if raw.get("python") is not None:
            allowed.add(_explicit_launcher(raw["python"]))
        declared = raw.get("trusted_launchers", [])
        if not isinstance(declared, list):
            raise RecordError("resource trusted_launchers must be a list")
        allowed.update(_explicit_launcher(value) for value in declared)
        common = allowed if common is None else common & allowed
    return common or set()


def promotion_code_roots(control_manifest_id: str, candidate_manifest_id: str) -> dict[str, str]:
    """Collision-free workdir roots derived from the immutable source pair."""

    namespace = sha256_object(
        {
            "control_manifest_id": control_manifest_id,
            "candidate_manifest_id": candidate_manifest_id,
        }
    )[:16]
    base = f"{PROMOTION_CODE_ROOT}/{namespace}"
    return {"control": f"{base}/control", "candidate": f"{base}/candidate"}


SCOPE_TEXT_FIELDS = (
    "id",
    "hardware_class",
    "dataset_split",
    "tokenizer",
    "evaluator",
    "precision",
)


def normalize_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the canonical structured benchmark scope."""
    if not isinstance(scope, Mapping):
        raise RecordError("v2 search requires a structured benchmark scope")
    normalized = dict(scope)
    for field in SCOPE_TEXT_FIELDS:
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RecordError(f"scope.{field} must be non-empty text")
        normalized[field] = value.strip()

    metric = normalized.get("metric")
    if not isinstance(metric, Mapping):
        raise RecordError("scope.metric must be an object")
    metric_name = metric.get("name")
    direction = metric.get("direction")
    if metric_name != PRIMARY_METRIC or direction != "minimize":
        raise RecordError(f"v2 fast search requires scope.metric={PRIMARY_METRIC!r}/'minimize'")
    normalized["metric"] = {"name": PRIMARY_METRIC, "direction": "minimize"}

    budget = normalized.get("budget")
    if not isinstance(budget, Mapping):
        raise RecordError("scope.budget must be an object")
    budget_kind = budget.get("kind")
    if budget_kind not in SCOPE_BUDGET_KINDS:
        raise RecordError("scope.budget.kind must be 'wall_seconds' or 'training_seconds'")
    seconds = budget.get("value")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
        raise RecordError("scope.budget.value must be a positive number")
    normalized["budget"] = {"kind": budget_kind, "value": float(seconds)}

    # Round-trip through canonical JSON so callers cannot mutate nested input after a
    # spec is constructed.
    import json

    return json.loads(canonical_json(normalized))


def scope_digest(scope: Mapping[str, Any]) -> str:
    return sha256_object(normalize_scope(scope))


def seeded_env(seed: int, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Bind the seed to the process environment, not merely the replicate label."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RecordError("seed must be a non-negative integer")
    result = {str(key): str(value) for key, value in (env or {}).items()}
    declared = result.get("AUTORESEARCH_SEED")
    if declared is not None and declared != str(seed):
        raise RecordError(f"AUTORESEARCH_SEED={declared!r} disagrees with planned seed {seed}")
    result["AUTORESEARCH_SEED"] = str(seed)
    return result


def normalize_mutable_paths(paths: Sequence[str]) -> list[str]:
    """Validate code paths that may differ between a bank control and candidate."""
    normalized: list[str] = []
    for raw in paths:
        value = str(raw).strip()
        parts = value.split("/")
        if (
            not value
            or value.startswith(("/", "~"))
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise RecordError("mutable code paths must be normalized relative execution paths")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise RecordError("mutable code paths must be unique")
    return sorted(normalized)


def validate_code_entry_argv(
    argv: Sequence[str],
    code_paths: Sequence[str],
    *,
    trusted_launchers: Sequence[str] | set[str] | None = None,
) -> list[str]:
    """Prove the command enters through a sealed relative code binding.

    The supported command shape is deliberately narrow: a sealed file may be the
    executable itself, or it may follow a known interpreter/launcher and option-only
    prefix (for example ``python -u train.py``).  Unsealed positional programs,
    ``-c``/``-m`` indirection, and absolute payload arguments are rejected because the
    binding checks could not prove which bytes they execute or consume.
    """

    tokens = [str(token) for token in argv]
    sealed = set(code_paths)
    absolute_payloads = [token for token in tokens[1:] if Path(token).is_absolute()]
    if absolute_payloads:
        raise RecordError(
            f"argv contains absolute payload paths outside sealed staging: {absolute_payloads}"
        )
    matches = [
        (index, token.removeprefix("./"))
        for index, token in enumerate(tokens)
        if not Path(token).is_absolute()
        and (not token.startswith(("-", "+")) or token.startswith("./"))
        and token.removeprefix("./") in sealed
    ]
    if not matches:
        raise RecordError(
            "argv must execute at least one sealed code binding by its relative "
            "execution_path (for example: python train.py)"
        )
    first_index = matches[0][0]
    if first_index == 0:
        # Popen PATH-resolves a bare argv[0].  A binding called "python" therefore
        # does not prove that the staged file ran.  Direct sealed executables must
        # contain a slash so the OS resolves them from the staged workdir.
        if not any(separator in tokens[0] for separator in ("/", "\\")):
            raise RecordError(
                "a direct sealed argv[0] must use an explicit relative path such as "
                "./launcher; bare names are PATH-resolved"
            )
        return sorted({path for _, path in matches})
    if first_index:
        launcher_token = tokens[0]
        launcher_path = Path(launcher_token)
        launcher = launcher_path.name
        allowed_launchers = set(trusted_launchers or DEFAULT_TRUSTED_LAUNCHERS)
        if launcher_token not in allowed_launchers:
            raise RecordError(
                "argv launcher is not explicitly trusted by every execution resource; "
                "use @python locally, declare a resource launcher path, or invoke a "
                "sealed launcher through /bin/bash"
            )
        prefix = tokens[1:first_index]
        python_launcher = bool(re.fullmatch(r"python(?:w)?(?:\d+(?:\.\d+)*)?|pypy\d*", launcher))
        # Parsing every interpreter's option grammar would itself become a security
        # boundary.  Keep the accepted grammar intentionally tiny: these Python
        # switches consume no following argument, so the next token is provably the
        # program.  In particular, ``-W train.py evil.py`` is rejected rather than
        # mistaking train.py for the executed entry when it is actually -W's value.
        safe_python_switches = {
            "-B",
            "-E",
            "-I",
            "-O",
            "-OO",
            "-P",
            "-S",
            "-s",
            "-u",
            "-v",
            "-x",
        }
        direct_launcher = launcher in {"bash", "sh", "zsh", "node"}
        torchrun_launcher = launcher == "torchrun"
        safe_torchrun_prefix = all(
            token == "--standalone" or (token.startswith("--") and "=" in token) for token in prefix
        )
        accepted = (
            (python_launcher and all(token in safe_python_switches for token in prefix))
            or (direct_launcher and not prefix)
            or (torchrun_launcher and safe_torchrun_prefix)
        )
        if not accepted:
            raise RecordError(
                "argv does not use the supported direct sealed-entry grammar; "
                "use an interpreter followed by the relative sealed execution_path, "
                "or put complex launch logic in a sealed launcher"
            )
    return sorted({path for _, path in matches})


def profile_health(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Diagnose whether overhead, rather than training, dominates the measured run."""
    training = metrics.get("training_seconds")
    total = metrics.get("total_seconds")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (training, total)
    ):
        return {
            "state": "unprofiled",
            "reason": "training_seconds and total_seconds are required",
        }
    training_value = float(training)
    total_value = float(total)
    if training_value < 0 or total_value <= 0 or training_value > total_value:
        return {
            "state": "invalid",
            "reason": "timing metrics are inconsistent",
            "training_seconds": training_value,
            "total_seconds": total_value,
        }
    overhead = total_value - training_value
    fraction = overhead / total_value
    return {
        "state": "healthy" if fraction <= 0.25 else "overhead_dominated",
        "training_seconds": training_value,
        "total_seconds": total_value,
        "overhead_seconds": overhead,
        "overhead_fraction": fraction,
        "reason": (
            "training owns at least 75% of elapsed time"
            if fraction <= 0.25
            else "input/compile/evaluation overhead exceeds 25% of elapsed time"
        ),
    }
