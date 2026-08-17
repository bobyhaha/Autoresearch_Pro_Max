"""High-level v2 workflow built only from existing immutable authorities.

``V2Workflow`` stages scientific records through ``ResearchEngine`` and
``SealingAuthority``, then hands their manifests to ``CampaignQueue``.  It never
launches a command itself and never creates a mutable scientific pointer.  Bank
selection is reconstructed from ``BankIndex`` and frozen into the candidate spec
before that candidate is sealed.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bank import (
    BANK_MAX_USES,
    BankIndex,
    context_fingerprint,
    search_lane,
    search_metadata,
)
from .campaign import CampaignQueue
from .exploration import enforce as enforce_exploration_budget
from .exploration import normalize_family, normalize_track
from .protocol import (
    CORE_METRICS,
    PROMOTION_SEEDS,
    PROTOCOL_VERSION,
    SEARCH_GATE,
    normalize_mutable_paths,
    normalize_scope,
    profile_health,
    promotion_code_roots,
    seeded_env,
    trusted_launchers_for_resources,
    validate_code_entry_argv,
)
from .records import RecordError, canonical_json, sha256_file
from .research import ResearchEngine
from .science import ScientificLibrary
from .sealing import SealingAuthority
from .store import Store


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9_.-]+", "_", value.strip().lower()).strip("._-")
    if not result:
        raise RecordError("workflow labels must contain an identifier character")
    if not result[0].isalpha():
        result = f"x_{result}"
    if len(result) > 100:
        digest = hashlib.sha256(result.encode("utf-8")).hexdigest()[:12]
        result = f"{result[:86]}_{digest}"
    return result


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(canonical_json(dict(value)))


def _source_path(raw: Mapping[str, Any], base: Path | None) -> Path:
    source = Path(str(raw.get("source", ""))).expanduser()
    if base is not None and not source.is_absolute():
        source = base / source
    source = source.resolve()
    if not source.is_file():
        raise RecordError(f"binding source is not a file: {source}")
    return source


def preview_execution_bindings(
    execution: Mapping[str, Any], *, base: Path | None = None
) -> dict[str, list[dict[str, str]]]:
    """Hash an execution draft without creating records or blobs."""

    preview: dict[str, list[dict[str, str]]] = {}
    all_paths: list[str] = []
    for group in ("code_bindings", "data_bindings"):
        raw_rows = execution.get(group, [])
        if not isinstance(raw_rows, list) or not raw_rows:
            raise RecordError(f"execution {group} must be a non-empty list")
        rows: list[dict[str, str]] = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise RecordError(f"execution {group} must contain objects")
            path = str(raw.get("execution_path", "")).strip()
            parts = Path(path).parts
            if not path or Path(path).is_absolute() or ".." in parts:
                raise RecordError("binding execution_path must be a safe relative path")
            source = _source_path(raw, base)
            rows.append(
                {
                    "source_name": source.name,
                    "execution_path": path,
                    "sha256": sha256_file(source),
                }
            )
            all_paths.append(path)
        preview[group] = sorted(rows, key=lambda row: (row["execution_path"], row["sha256"]))
    if len(all_paths) != len(set(all_paths)):
        raise RecordError("every code/data binding must have a unique execution_path")
    return preview


def validate_bound_argv(
    execution: Mapping[str, Any],
    argv: Sequence[str],
    *,
    base: Path | None = None,
) -> list[str]:
    """Require the command to enter through a sealed, workdir-relative code path.

    Verifying a staged blob is meaningless if ``argv`` executes a mutable absolute
    source file somewhere else.  V2 therefore supports the ordinary explicit form
    (for example ``python train.py``) and records exactly which sealed entry paths
    the command names.  Shell snippets and ``python -m`` can be wrapped in a small
    sealed launcher when needed.
    """

    raw_rows = execution.get("code_bindings", [])
    if not isinstance(raw_rows, list) or not raw_rows:
        raise RecordError("execution code_bindings must be a non-empty list")
    tokens = [str(token) for token in argv]
    code_paths: set[str] = set()
    absolute_sources: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise RecordError("execution code_bindings must contain objects")
        path = str(raw.get("execution_path", "")).strip()
        if path:
            code_paths.add(path)
        absolute_sources.add(str(_source_path(raw, base)))
    bypasses = sorted(set(tokens) & absolute_sources)
    if bypasses:
        raise RecordError(
            "argv executes an absolute binding source instead of its sealed "
            f"execution_path: {bypasses}"
        )
    return validate_code_entry_argv(
        tokens,
        sorted(code_paths),
        trusted_launchers=trusted_launchers_for_resources(execution.get("resources")),
    )


def duplicate_long_options(argv: Sequence[str]) -> list[str]:
    """Return repeated ``--long-option`` names, including ``--name=value`` forms."""

    names = [str(token).split("=", 1)[0] for token in argv if str(token).startswith("--")]
    counts = Counter(names)
    return sorted(name for name, count in counts.items() if count > 1)


def _snapshot_code_alias(root: str, execution_path: str) -> str:
    return f"{root}/{execution_path}"


def _rewrite_snapshot_argv(
    arm: Mapping[str, Any], manifest: Mapping[str, Any], lane: str, root: str
) -> list[str]:
    """Point a promotion arm at its own immutable code snapshot.

    Control and candidate commonly both execute ``train.py`` with different bytes.
    A paired manifest cannot stage both at that one path, so confirmation materializes
    complete code trees under separate roots and rewrites the sealed entry token.
    """

    code_paths = {
        str(row["execution_path"])
        for row in manifest["payload"].get("code_bindings", [])
        if isinstance(row, Mapping) and isinstance(row.get("execution_path"), str)
    }
    rewritten: list[str] = []
    matched = False
    for raw in arm.get("argv", []):
        token = str(raw)
        normalized = token.removeprefix("./")
        if not Path(token).is_absolute() and normalized in code_paths:
            rewritten.append(_snapshot_code_alias(root, normalized))
            matched = True
        else:
            rewritten.append(token)
    if not matched:
        raise RecordError(
            f"{lane} argv does not execute a sealed relative code binding; "
            "promotion cannot prove which snapshot ran"
        )
    return rewritten


def _single_arm_spec(
    *,
    spec_id: str,
    lane: str,
    title: str,
    argv: Sequence[str],
    seed: int,
    scope: Mapping[str, Any],
    mutable_code_paths: Sequence[str],
    bank_id: str,
    require_gpu: bool,
    subsystem: str,
    source_ids: Sequence[str],
    direction: str | None = None,
    reference_controls: Sequence[Mapping[str, Any]] | None = None,
    baseline_fp: str | None = None,
    context_fp: str | None = None,
    minimum_steps: int = 1,
    scientific: Mapping[str, Any] | None = None,
    exploration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_scope = normalize_scope(scope)
    arm_name = "control" if lane == "bank" else "candidate"
    search: dict[str, Any] = {
        "lane": lane,
        "bank_id": bank_id,
        "mutable_code_paths": normalize_mutable_paths(mutable_code_paths),
    }
    if exploration is not None:
        # The budget report is recorded whether it passed cleanly or was overridden,
        # so an audit can tell the two apart without replaying the campaign.
        search["track"] = str(exploration["track"])
        search["family"] = str(exploration["family"])
        search["exploration_budget"] = copy.deepcopy(dict(exploration))
    if baseline_fp is not None:
        search["baseline_fingerprint"] = baseline_fp
    if context_fp is not None:
        search["context_fingerprint"] = context_fp
    if reference_controls is not None:
        search["reference_controls"] = [copy.deepcopy(dict(row)) for row in reference_controls]
    if scientific is not None:
        search["hypothesis_ids"] = list(scientific["hypothesis_ids"])
    mechanism = {
        "cause": title,
        "effect": "fixed-scope validation bits per byte",
        "chain": [title, "changed training trajectory", "changed val_bpb"],
    }
    hypothesis = {
        "statement": f"{title} lowers val_bpb.",
        "prediction": "candidate beats its frozen same-GPU bank control",
    }
    falsifier = {"statement": "A delta at or above the promotion gate rejects fast promotion."}
    question = f"Does {title} lower val_bpb in the frozen v2 scope?"
    if scientific is not None:
        mechanism = copy.deepcopy(scientific["mechanism"])
        hypothesis = copy.deepcopy(scientific["hypothesis"])
        falsifier = copy.deepcopy(scientific["falsifier"])
        question = str(scientific["question"])
    return {
        "id": spec_id,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "pilot",
        "title": title,
        "question": question,
        "mechanism": mechanism,
        "hypothesis": hypothesis,
        "falsifier": falsifier,
        "metric": copy.deepcopy(normalized_scope["metric"]),
        "plan": [
            {
                "replicate_id": f"seed_{seed}",
                "seed": seed,
                "arms": [
                    {
                        "name": arm_name,
                        "argv": [str(token) for token in argv],
                        "env": seeded_env(seed),
                    }
                ],
            }
        ],
        "analysis": {
            "effect": "single",
            "primary_arm": arm_name,
            "minimum_valid_replicates": 1,
            "success_rule": {"op": "lt", "value": 1e9},
            "falsifier_rule": {"op": "gte", "value": 1e9},
            "sota_eligible": False,
        },
        "requirements": {
            "required_metrics": list(CORE_METRICS),
            "minimum_steps": minimum_steps,
            "require_gpu": require_gpu,
            "isolation": "continuous" if require_gpu else "none",
        },
        "knowledge": {
            "source_ids": list(source_ids),
            "direction": direction or ("calibration" if lane == "bank" else "search"),
            "subsystem": subsystem,
        },
        "comparison_group": normalized_scope["id"],
        "scope": normalized_scope,
        "search": search,
    }


def _resource_slots(execution: Mapping[str, Any]) -> list[tuple[dict[str, Any], int | None]]:
    resources = execution.get("resources")
    if not isinstance(resources, list) or not resources:
        raise RecordError("execution resources must be a non-empty list")
    slots: list[tuple[dict[str, Any], int | None]] = []
    for raw in resources:
        if not isinstance(raw, Mapping):
            raise RecordError("execution resources must contain objects")
        resource = _copy(raw)
        gpus = resource.get("gpus", [])
        if not isinstance(gpus, list):
            raise RecordError("resource gpus must be a list")
        if gpus:
            slots.extend((resource, int(gpu)) for gpu in gpus)
        else:
            slots.append((resource, None))
    return slots


def _validate_scope_resources(execution: Mapping[str, Any], scope: Mapping[str, Any]) -> None:
    expected = str(scope["hardware_class"])
    for resource, _ in _resource_slots(execution):
        observed = resource.get("hardware_class")
        if observed != expected:
            raise RecordError(
                f"resource {resource.get('id')} hardware_class={observed!r} does not "
                f"match scope.hardware_class={expected!r}"
            )


def _pin_execution(
    execution: Mapping[str, Any], resource: Mapping[str, Any], gpu: int | None
) -> dict[str, Any]:
    pinned = _copy(execution)
    selected = _copy(resource)
    selected["gpus"] = [] if gpu is None else [gpu]
    pinned["resources"] = [selected]
    return pinned


def _manifest_preview(
    spec: Mapping[str, Any], binding_preview: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "id": "preview_manifest",
        "payload": {
            "spec_id": spec["id"],
            "plan": copy.deepcopy(spec["plan"]),
            "code_bindings": copy.deepcopy(binding_preview["code_bindings"]),
            "data_bindings": copy.deepcopy(binding_preview["data_bindings"]),
        },
    }


def _result_spec_ids(store: Store) -> set[str]:
    return {str(row["payload"]["spec_id"]) for row in store.list("result_bundle")}


def _pending_control_reservations(store: Store, queue: CampaignQueue) -> Counter[str]:
    """Frozen controls held by candidates that do not have a ResultBundle yet."""

    landed = _result_spec_ids(store)
    active_specs = {
        str(job["spec_id"])
        for job in queue.jobs()
        if job.get("state") in {"pending", "running", "waiting"}
    }
    reservations: Counter[str] = Counter()
    for spec in store.list("experiment_spec"):
        if (
            search_lane(spec) != "candidate"
            or spec["id"] in landed
            or spec["id"] not in active_specs
        ):
            continue
        refs = search_metadata(spec).get("reference_controls", [])
        if not isinstance(refs, list):
            continue
        for row in refs:
            if not isinstance(row, Mapping):
                continue
            result_id = row.get("result_id", row.get("control_result_id"))
            if isinstance(result_id, str) and result_id:
                reservations[result_id] += 1
    return reservations


def _is_model_subsystem(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_")
    bottleneck_lanes = (
        "calibration",
        "compile",
        "data",
        "evaluation",
        "evaluator",
        "input",
        "instrumentation",
        "tokenizer",
    )
    return not normalized.startswith(bottleneck_lanes)


class V2Workflow:
    def __init__(
        self,
        store: Store,
        *,
        queue: CampaignQueue | None = None,
        now: Callable[[], datetime] = _default_now,
    ) -> None:
        self.store = store
        self.research = ResearchEngine(store)
        self.sealing = SealingAuthority(store)
        self.queue = queue or CampaignQueue(store)
        self._now = now

    def _scientific_context(
        self, hypothesis_ids: Sequence[str], *, allow_weak_science: bool = False
    ) -> dict[str, Any] | None:
        """Resolve one primary hypothesis into the immutable experiment chain."""

        ids = list(dict.fromkeys(str(item) for item in hypothesis_ids if str(item)))
        if not ids:
            return None
        if len(ids) != 1:
            raise RecordError("a candidate must test exactly one primary scientific hypothesis")
        hypothesis_record = self.store.get("scientific_hypothesis", ids[0])
        payload = hypothesis_record["payload"]
        projection = ScientificLibrary(self.store).synthesize(write=False)
        projected = next(
            (row for row in projection["hypotheses"] if row["hypothesis_id"] == ids[0]),
            None,
        )
        if projected is None:
            raise RecordError("scientific hypothesis has no rebuildable confidence projection")
        topics = set(payload["topics"])
        topic_gaps = [row for row in projection["research_gaps"] if row["topic_id"] in topics]
        covered_topics = {row["topic_id"] for row in topic_gaps}
        blocking_gaps = [row for row in topic_gaps if row["search_due"] or row["analysis_due"]]
        weak_reasons = []
        if projected["foundation_confidence"] < 0.60:
            weak_reasons.append("foundation confidence is below 0.60")
        if projected["foundation_evidence_units"] < 2:
            weak_reasons.append("fewer than two independent evidence units")
        if topics - covered_topics:
            weak_reasons.append(f"topics lack a research agenda: {sorted(topics - covered_topics)}")
        if blocking_gaps:
            weak_reasons.append(
                "agenda research remains due: "
                + ", ".join(f"{row['topic_id']}={row['reasons']}" for row in blocking_gaps)
            )
        if weak_reasons and not allow_weak_science:
            raise RecordError(
                "scientific hypothesis is not research-ready ("
                + "; ".join(weak_reasons)
                + "); gather literature/claims or pass allow_weak_science=True for explicit exploration"
            )
        mechanism_ids = payload["mechanism_ids"]
        if not mechanism_ids:
            raise RecordError("scientific hypothesis has no mechanism")
        mechanism_payload = self.store.get("scientific_mechanism", mechanism_ids[0])["payload"]
        node_labels = {str(row["id"]): str(row["label"]) for row in mechanism_payload["nodes"]}
        chain = [
            f"{node_labels[edge['from']]} -> {edge['relation']} -> {node_labels[edge['to']]}"
            for edge in mechanism_payload["edges"]
        ]
        prediction = payload["prediction"]
        return {
            "hypothesis_ids": ids,
            "question": f"Does the preregistered hypothesis hold: {payload['statement']}",
            "mechanism": {
                "cause": payload["intervention"]["summary"],
                "effect": mechanism_payload["statement"],
                "chain": chain,
            },
            "hypothesis": {
                "statement": payload["statement"],
                "prediction": prediction["statement"],
            },
            "falsifier": {"statement": "; ".join(payload["falsifiers"])},
        }

    def stage_calibration(
        self,
        bank_id: str,
        label: str,
        execution: Mapping[str, Any],
        argv: Sequence[str],
        scope: Mapping[str, Any],
        mutable_code_paths: Sequence[str],
        seed: int = 42,
        *,
        base: Path | None = None,
        minimum_steps: int = 1,
    ) -> dict[str, Any]:
        """Stage one pinned bank run per declared resource/GPU slot."""

        validate_bound_argv(execution, argv, base=base)
        normalized_scope = normalize_scope(scope)
        _validate_scope_resources(execution, normalized_scope)
        revision = _slug(bank_id)
        staged: list[dict[str, Any]] = []
        for resource, gpu in _resource_slots(execution):
            resource_id = _slug(str(resource.get("id", "resource")))
            slot = "cpu" if gpu is None else f"gpu_{gpu}"
            spec_id = f"exp_bank_{_slug(label)}_{resource_id}_{slot}"
            proposal = _single_arm_spec(
                spec_id=spec_id,
                lane="bank",
                title=f"bank calibration {label} on {resource_id}/{slot}",
                argv=argv,
                seed=seed,
                scope=normalized_scope,
                mutable_code_paths=mutable_code_paths,
                bank_id=revision,
                require_gpu=gpu is not None,
                subsystem="calibration",
                source_ids=[f"bank_{revision}"],
                minimum_steps=minimum_steps,
            )
            spec = self.research.create(proposal)
            pinned = _pin_execution(execution, resource, gpu)
            manifest = self.sealing.seal(spec["id"], pinned, base=base)
            job = self.queue.enqueue(manifest["id"])
            staged.append(
                {
                    "resource_id": resource["id"],
                    "gpu": gpu,
                    "spec": spec,
                    "manifest": manifest,
                    "job": job,
                }
            )
        return {
            "bank_id": revision,
            "scope": normalized_scope,
            "staged": staged,
        }

    def _staged_candidates(self, comparison_group: str) -> list[Mapping[str, Any]]:
        """Candidate-lane spec payloads in this scope, oldest staged first.

        The budget counts *staged* candidates rather than landed results, because the
        decision it constrains is what to launch next.  Waiting for results before
        counting a slot would let eight knobs be queued while the first is still
        running -- which is precisely how a queue drains into a knob sweep.
        """

        rows = [
            spec
            for spec in self.store.list("experiment_spec")
            if search_lane(spec) == "candidate"
            and str(spec.get("payload", spec).get("comparison_group", "")) == comparison_group
        ]
        rows.sort(key=lambda spec: (str(spec.get("created_at", "")), str(spec.get("id", ""))))
        return [dict(spec.get("payload", spec)) for spec in rows]

    def _gate_cleared_families(self) -> list[str]:
        """Families with at least one candidate that reached the promotion queue."""

        try:
            bank = BankIndex.from_store(self.store, now=self._now())
            view = bank.promotion_view()
        except RecordError:  # pragma: no cover - an unscorable bank clears no family
            return []
        specs = {
            str(spec["id"]): dict(spec.get("payload", spec))
            for spec in self.store.list("experiment_spec")
        }
        cleared: set[str] = set()
        for row in view.get("promotion_queue", []):
            spec = specs.get(str(row.get("spec_id", "")))
            if spec is None:
                continue
            search = spec.get("search")
            if isinstance(search, Mapping) and isinstance(search.get("family"), str):
                cleared.add(str(search["family"]))
                continue
            knowledge = spec.get("knowledge")
            if isinstance(knowledge, Mapping) and isinstance(knowledge.get("subsystem"), str):
                cleared.add(str(knowledge["subsystem"]).strip().lower().replace("-", "_"))
        return sorted(cleared)

    def _control_profile(self, index: BankIndex, control: Mapping[str, Any]) -> dict[str, Any]:
        decision = index.decisions.get(str(control["result_id"]))
        spec = index.specs.get(str(control["spec_id"]))
        if decision is None or spec is None:
            return {"state": "unprofiled", "reason": "control evidence is unavailable"}
        arms = decision["payload"].get("measurements", {}).get("arms", {})
        primary = spec["payload"].get("analysis", {}).get("primary_arm", "control")
        metrics = arms.get(primary, {}) if isinstance(arms, Mapping) else {}
        return profile_health(metrics if isinstance(metrics, Mapping) else {})

    @staticmethod
    def _slot_exists(
        execution: Mapping[str, Any], control: Mapping[str, Any]
    ) -> tuple[dict[str, Any], int | None] | None:
        for resource, gpu in _resource_slots(execution):
            if resource.get("id") == control.get("resource_id") and gpu == control.get("gpu"):
                return resource, gpu
        return None

    def stage_candidate(
        self,
        label: str,
        execution: Mapping[str, Any],
        argv: Sequence[str],
        scope: Mapping[str, Any],
        mutable_code_paths: Sequence[str],
        bank_id: str,
        *,
        seed: int = 42,
        summary: str | None = None,
        direction: str = "unassigned",
        subsystem: str = "model",
        source_ids: Sequence[str] = ("banked_control",),
        hypothesis_ids: Sequence[str] = (),
        base: Path | None = None,
        minimum_steps: int = 1,
        allow_overhead_dominated: bool = False,
        allow_weak_science: bool = False,
        track: str = "mechanism",
        family: str | None = None,
        exploration_override: str | None = None,
    ) -> dict[str, Any]:
        """Atomically reserve a control, pin its slot, and enqueue one candidate."""

        duplicates = duplicate_long_options(argv)
        if duplicates:
            raise RecordError(f"candidate argv repeats long option flags: {duplicates}")
        validate_bound_argv(execution, argv, base=base)
        # Selection, immutable spec creation, sealing, and queue admission form one
        # reservation transaction.  Without this lock two search processes can both
        # consume the final bank use.  Failed seal/enqueue attempts do not count because
        # reservation rebuilding considers active queue jobs, not orphan specs.
        with self.store.lock("v2_candidate_reservations"):
            return self._stage_candidate_locked(
                label=label,
                execution=execution,
                argv=argv,
                scope=scope,
                mutable_code_paths=mutable_code_paths,
                bank_id=bank_id,
                seed=seed,
                summary=summary,
                direction=direction,
                subsystem=subsystem,
                source_ids=source_ids,
                hypothesis_ids=hypothesis_ids,
                base=base,
                minimum_steps=minimum_steps,
                allow_overhead_dominated=allow_overhead_dominated,
                allow_weak_science=allow_weak_science,
                track=track,
                family=family,
                exploration_override=exploration_override,
            )

    def _stage_candidate_locked(
        self,
        *,
        label: str,
        execution: Mapping[str, Any],
        argv: Sequence[str],
        scope: Mapping[str, Any],
        mutable_code_paths: Sequence[str],
        bank_id: str,
        seed: int,
        summary: str | None,
        direction: str,
        subsystem: str,
        source_ids: Sequence[str],
        hypothesis_ids: Sequence[str],
        base: Path | None,
        minimum_steps: int,
        allow_overhead_dominated: bool,
        allow_weak_science: bool,
        track: str = "mechanism",
        family: str | None = None,
        exploration_override: str | None = None,
    ) -> dict[str, Any]:
        """Implementation of ``stage_candidate`` while its reservation lock is held."""

        normalized_scope = normalize_scope(scope)
        _validate_scope_resources(execution, normalized_scope)
        # The exploration budget runs inside the reservation lock, before a control is
        # selected, so a refused candidate cannot consume a bank use or race another
        # staging process into the same window.
        exploration = enforce_exploration_budget(
            track=normalize_track(track),
            family=normalize_family(family, subsystem=subsystem),
            staged_candidates=self._staged_candidates(normalized_scope["id"]),
            cleared_families=self._gate_cleared_families(),
            override_reason=exploration_override,
            subsystem=subsystem,
        )
        revision = _slug(bank_id)
        slots = _resource_slots(execution)
        require_gpu = any(gpu is not None for _, gpu in slots)
        title = summary or label
        scientific = self._scientific_context(hypothesis_ids, allow_weak_science=allow_weak_science)
        linked_source_ids = list(
            dict.fromkeys([*source_ids, *[str(item) for item in hypothesis_ids]])
        )

        # Build the scientific and binding context before writing an immutable spec.
        draft = _single_arm_spec(
            spec_id=f"exp_{_slug(label)}",
            lane="candidate",
            title=title,
            argv=argv,
            seed=seed,
            scope=normalized_scope,
            mutable_code_paths=mutable_code_paths,
            bank_id=revision,
            require_gpu=require_gpu,
            subsystem=subsystem,
            source_ids=linked_source_ids,
            direction=direction,
            reference_controls=[],
            baseline_fp="0" * 64,
            minimum_steps=minimum_steps,
            scientific=scientific,
            exploration=exploration,
        )
        preview = preview_execution_bindings(execution, base=base)
        preview_manifest = _manifest_preview(draft, preview)
        candidate_context = context_fingerprint(
            {"id": draft["id"], "payload": draft}, preview_manifest
        )

        bank = BankIndex.from_store(self.store, now=self._now())
        matching_controls = [
            row
            for row in bank.controls
            if row["bank_id"] == revision
            and row["context_fingerprint"] == candidate_context
            and row["seed"] == seed
            and self._slot_exists(execution, row) is not None
        ]
        fingerprints = sorted({row["baseline_fingerprint"] for row in matching_controls})
        eligible_by_fingerprint: dict[str, list[dict[str, Any]]] = {}
        for fingerprint in fingerprints:
            rows = bank.eligible_controls(
                bank_id=revision,
                baseline_fingerprint=fingerprint,
                context_fingerprint=candidate_context,
                seed=seed,
                at=self._now(),
            )
            rows = [row for row in rows if self._slot_exists(execution, row) is not None]
            if rows:
                eligible_by_fingerprint[fingerprint] = rows
        if not eligible_by_fingerprint:
            raise RecordError(
                "no fresh exact bank control matches this scope, context, seed, and resources"
            )
        if len(eligible_by_fingerprint) != 1:
            raise RecordError(
                "bank id is ambiguous: multiple fresh baseline fingerprints match the candidate"
            )
        baseline_fp, eligible = next(iter(eligible_by_fingerprint.items()))

        # An identical command is a real experiment when a declared mutable binding
        # changed.  That is the normal autoresearch loop: edit train.py, keep the exact
        # benchmark command.  It is a no-op only when both command and mutable bytes match.
        control_argv_by_result: dict[str, list[str]] = {}
        mutable_changes_by_result: dict[str, list[str]] = {}
        candidate_code = {row["execution_path"]: row["sha256"] for row in preview["code_bindings"]}
        declared_mutable = set(normalize_mutable_paths(mutable_code_paths))
        for row in eligible:
            manifest = bank.manifests[row["manifest_id"]]
            arms = manifest["payload"]["plan"][0]["arms"]
            control_arm = next(
                (arm for arm in arms if arm.get("name") == "control"),
                arms[0] if len(arms) == 1 else None,
            )
            if control_arm is None:
                continue
            control_argv_by_result[row["result_id"]] = list(control_arm["argv"])
            control_code = {
                binding["execution_path"]: binding["sha256"]
                for binding in manifest["payload"]["code_bindings"]
            }
            mutable_changes_by_result[row["result_id"]] = sorted(
                path
                for path in declared_mutable
                if candidate_code.get(path) != control_code.get(path)
            )
        no_op_controls = {
            result_id
            for result_id, control_argv in control_argv_by_result.items()
            if list(argv) == control_argv and not mutable_changes_by_result.get(result_id)
        }
        eligible = [row for row in eligible if row["result_id"] not in no_op_controls]
        if not eligible:
            raise RecordError(
                "candidate is identical to its bank control: argv and all declared "
                "mutable code bindings are unchanged"
            )

        reservations = _pending_control_reservations(self.store, self.queue)
        selectable: list[tuple[int, str, dict[str, Any]]] = []
        overhead_rejections: list[dict[str, Any]] = []
        invalid_profile_rejections: list[dict[str, Any]] = []
        for row in eligible:
            total_reserved = int(row.get("uses", 0)) + reservations[row["result_id"]]
            if total_reserved >= BANK_MAX_USES:
                continue
            health = self._control_profile(bank, row)
            if health.get("state") in {"invalid", "unprofiled"}:
                invalid_profile_rejections.append(
                    {"result_id": row["result_id"], "profile": health}
                )
                continue
            if (
                _is_model_subsystem(subsystem)
                and health.get("state") == "overhead_dominated"
                and not allow_overhead_dominated
            ):
                overhead_rejections.append({"result_id": row["result_id"], "profile": health})
                continue
            selectable.append((total_reserved, row["gpu_key"], row))
        if not selectable:
            if invalid_profile_rejections:
                raise RecordError(
                    "all eligible bank controls have missing or inconsistent timing "
                    "instrumentation; recalibrate before search"
                )
            if overhead_rejections:
                raise RecordError(
                    "all eligible bank controls are overhead-dominated; first search data, "
                    "compile, evaluation, or instrumentation, or pass "
                    "allow_overhead_dominated=True to override explicitly"
                )
            raise RecordError("all matching bank controls have exhausted their use reservations")
        _, _, selected = min(selectable, key=lambda item: (item[0], item[1]))
        selected_resource = self._slot_exists(execution, selected)
        assert selected_resource is not None
        resource, gpu = selected_resource
        frozen = copy.deepcopy(selected)
        frozen["pending_reservations"] = reservations[selected["result_id"]]
        frozen["effective_uses"] = int(selected.get("uses", 0)) + frozen["pending_reservations"]
        frozen["mutable_code_changes"] = mutable_changes_by_result.get(selected["result_id"], [])

        proposal = _single_arm_spec(
            spec_id=draft["id"],
            lane="candidate",
            title=title,
            argv=argv,
            seed=seed,
            scope=normalized_scope,
            mutable_code_paths=mutable_code_paths,
            bank_id=revision,
            require_gpu=gpu is not None,
            subsystem=subsystem,
            source_ids=linked_source_ids,
            direction=direction,
            reference_controls=[frozen],
            baseline_fp=baseline_fp,
            context_fp=candidate_context,
            minimum_steps=minimum_steps,
            scientific=scientific,
            exploration=exploration,
        )
        spec = self.research.create(proposal)
        pinned_execution = _pin_execution(execution, resource, gpu)
        manifest = self.sealing.seal(spec["id"], pinned_execution, base=base)
        sealed_context = context_fingerprint(spec, manifest)
        if sealed_context != candidate_context:
            raise RecordError("sealed candidate context differs from its binding preview")
        job = self.queue.enqueue(manifest["id"])
        return {
            "spec": spec,
            "manifest": manifest,
            "job": job,
            "reference_control": frozen,
            "preview": {
                **preview,
                "context_fingerprint": candidate_context,
                "baseline_fingerprint": baseline_fp,
            },
            "resource_id": resource["id"],
            "gpu": gpu,
        }

    def _manifest_for_spec(self, spec_id: str) -> Mapping[str, Any]:
        rows = [
            row
            for row in self.store.list("execution_manifest")
            if row["payload"].get("spec_id") == spec_id
        ]
        if len(rows) != 1:
            raise RecordError(
                f"workflow requires exactly one sealed manifest for {spec_id}; found {len(rows)}"
            )
        return rows[0]

    def _promotion_sources(
        self, candidate_spec_id: str
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        candidate = self.store.get("experiment_spec", candidate_spec_id)
        if search_lane(candidate) != "candidate":
            raise RecordError("promotion source must be a candidate-lane ExperimentSpec")
        refs = search_metadata(candidate).get("reference_controls", [])
        if not isinstance(refs, list) or len(refs) != 1 or not isinstance(refs[0], Mapping):
            raise RecordError("promotion source must freeze exactly one bank control")
        control_result_id = refs[0].get("result_id", refs[0].get("control_result_id"))
        if not isinstance(control_result_id, str):
            raise RecordError("frozen bank control has no ResultBundle id")
        control_result = self.store.get("result_bundle", control_result_id)
        control_manifest = self.store.get(
            "execution_manifest", control_result["payload"]["manifest_id"]
        )
        candidate_manifest = self._manifest_for_spec(candidate_spec_id)
        return candidate, candidate_manifest, control_result, control_manifest

    def _promotion_score(self, candidate_spec_id: str) -> tuple[Mapping[str, Any], dict[str, Any]]:
        """Require one landed, valid, gate-clearing candidate result.

        Promotion is deliberately downstream of the rebuildable bank index.  A queued
        candidate, an invalid run, or a result that fails the search gate is not a
        promotion source merely because its immutable spec exists.
        """

        results = [
            row
            for row in self.store.list("result_bundle")
            if row["payload"].get("spec_id") == candidate_spec_id
        ]
        if len(results) != 1:
            raise RecordError(
                "promotion requires exactly one landed candidate ResultBundle; "
                f"found {len(results)}"
            )
        result = results[0]
        score = BankIndex.from_store(self.store, now=self._now()).score_candidate(result["id"])
        if score.get("status") != "scored":
            raise RecordError(
                "promotion requires a scored candidate: "
                f"{score.get('reason', 'bank score is unavailable')}"
            )
        if score.get("promotion_due") is not True:
            raise RecordError(
                "candidate did not clear the bank promotion gate "
                f"(delta={score.get('delta')}, gate={score.get('gate')})"
            )
        return result, score

    def promotion_proposal(
        self,
        candidate_spec_id: str,
        seeds: Sequence[int] = PROMOTION_SEEDS,
        minimum_valid_replicates: int = 4,
    ) -> dict[str, Any]:
        """Create the ordinary reviewed confirmation implied by one candidate."""

        candidate_result, candidate_score = self._promotion_score(candidate_spec_id)
        candidate, candidate_manifest, control_result, control_manifest = self._promotion_sources(
            candidate_spec_id
        )
        normalized_seeds = list(seeds)
        if (
            not normalized_seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in normalized_seeds
            )
            or len(normalized_seeds) != len(set(normalized_seeds))
        ):
            raise RecordError("promotion seeds must be distinct non-negative integers")
        if minimum_valid_replicates < 3 or minimum_valid_replicates > len(normalized_seeds):
            raise RecordError("promotion minimum must be between 3 and the planned seed count")

        observed_seeds = {
            int(row["seed"])
            for manifest in (candidate_manifest, control_manifest)
            for row in manifest["payload"].get("plan", [])
            if isinstance(row.get("seed"), int) and not isinstance(row.get("seed"), bool)
        }
        reused = sorted(set(normalized_seeds) & observed_seeds)
        if reused:
            raise RecordError(
                f"promotion seeds must be held out from bank/search pilots; reused {reused}"
            )

        candidate_arm = candidate_manifest["payload"]["plan"][0]["arms"][0]
        control_arms = control_manifest["payload"]["plan"][0]["arms"]
        control_arm = next(
            (arm for arm in control_arms if arm.get("name") == "control"),
            control_arms[0] if len(control_arms) == 1 else None,
        )
        if control_arm is None:
            raise RecordError("bank manifest has no unique control arm")
        code_roots = promotion_code_roots(control_manifest["id"], candidate_manifest["id"])

        def reseed(
            arm: Mapping[str, Any],
            name: str,
            seed: int,
            manifest: Mapping[str, Any],
            lane: str,
        ) -> dict[str, Any]:
            env = dict(arm.get("env", {}))
            env.pop("AUTORESEARCH_SEED", None)
            return {
                "name": name,
                "argv": _rewrite_snapshot_argv(arm, manifest, lane, code_roots[lane]),
                "env": seeded_env(seed, env),
            }

        plan = []
        for index, seed in enumerate(normalized_seeds):
            control = reseed(control_arm, "control", seed, control_manifest, "control")
            experiment = reseed(candidate_arm, "candidate", seed, candidate_manifest, "candidate")
            arms = [control, experiment] if index % 2 == 0 else [experiment, control]
            plan.append({"replicate_id": f"seed_{seed}", "seed": seed, "arms": arms})

        candidate_payload = candidate["payload"]
        search = search_metadata(candidate)
        mutable = set(search.get("mutable_code_paths", []))
        control_spec = self.store.get("experiment_spec", control_result["payload"]["spec_id"])
        mutable.update(search_metadata(control_spec).get("mutable_code_paths", []))
        promotion_mutable = sorted(
            {
                _snapshot_code_alias(code_roots[lane], path)
                for lane, manifest in (
                    ("control", control_manifest),
                    ("candidate", candidate_manifest),
                )
                for path in mutable
                if any(
                    row.get("execution_path") == path
                    for row in manifest["payload"].get("code_bindings", [])
                )
            }
        )
        promotion_id = f"exp_promote_{_slug(candidate_spec_id.removeprefix('exp_'))}"
        return {
            "id": promotion_id,
            "protocol_version": PROTOCOL_VERSION,
            "stage": "confirmation",
            "title": f"Promotion confirmation: {candidate_payload['title']}",
            "question": "Does the candidate beat its exact bank control on held-out seeds?",
            "mechanism": copy.deepcopy(candidate_payload["mechanism"]),
            "hypothesis": {
                "statement": candidate_payload["hypothesis"]["statement"],
                "prediction": (
                    f"{candidate_payload['hypothesis']['prediction']}; confirmation mean "
                    f"paired difference is below -{SEARCH_GATE:g}"
                ),
            },
            "falsifier": {
                "statement": f"A mean paired difference at or above -{SEARCH_GATE:g} blocks promotion."
            },
            "metric": copy.deepcopy(candidate_payload["metric"]),
            "plan": plan,
            "analysis": {
                "effect": "difference",
                "primary_arm": "candidate",
                "reference_arm": "control",
                "minimum_valid_replicates": minimum_valid_replicates,
                "success_rule": {"op": "lt", "value": -SEARCH_GATE},
                "falsifier_rule": {"op": "gte", "value": -SEARCH_GATE},
                "sota_eligible": True,
            },
            "requirements": copy.deepcopy(candidate_payload["requirements"]),
            "knowledge": {
                "source_ids": list(
                    dict.fromkeys(
                        [candidate_spec_id, *candidate_payload["knowledge"]["source_ids"]]
                    )
                ),
                "direction": candidate_payload["knowledge"]["direction"],
                "subsystem": candidate_payload["knowledge"]["subsystem"],
            },
            "comparison_group": candidate_payload["comparison_group"],
            "scope": copy.deepcopy(candidate_payload["scope"]),
            "search": {
                "lane": "promotion",
                "bank_id": search.get("bank_id", search.get("baseline_id")),
                "baseline_fingerprint": search.get(
                    "baseline_fingerprint", search.get("bank_fingerprint")
                ),
                "context_fingerprint": search.get("context_fingerprint"),
                "mutable_code_paths": promotion_mutable,
                "hypothesis_ids": list(search.get("hypothesis_ids", [])),
                "arm_code_roots": code_roots,
                "source_spec_id": candidate_spec_id,
                "source_manifest_id": candidate_manifest["id"],
                "source_result_id": candidate_result["id"],
                "source_delta": candidate_score["delta"],
                "control_result_id": control_result["id"],
                "control_manifest_id": control_manifest["id"],
            },
        }

    def _combined_immutable_bindings(
        self, candidate_spec_id: str
    ) -> dict[str, list[dict[str, str]]]:
        _, candidate_manifest, _, control_manifest = self._promotion_sources(candidate_spec_id)
        code_roots = promotion_code_roots(control_manifest["id"], candidate_manifest["id"])
        code_bindings: list[dict[str, str]] = []
        for lane, manifest in (
            ("control", control_manifest),
            ("candidate", candidate_manifest),
        ):
            for row in manifest["payload"]["code_bindings"]:
                code_bindings.append(
                    {
                        "source": str((self.store.root / row["blob"]).resolve()),
                        "source_name": str(row["source_name"]),
                        "execution_path": _snapshot_code_alias(
                            code_roots[lane], str(row["execution_path"])
                        ),
                    }
                )

        # Data, tokenizer artifacts, and evaluator inputs are shared scientific
        # context.  Unlike mutable code, differing bytes at one data path are not two
        # arms of an experiment; they are two incomparable benchmark scopes.
        control_data = {
            str(row["execution_path"]): row for row in control_manifest["payload"]["data_bindings"]
        }
        candidate_data = {
            str(row["execution_path"]): row
            for row in candidate_manifest["payload"]["data_bindings"]
        }
        if set(control_data) != set(candidate_data):
            raise RecordError("promotion data binding paths differ from the frozen control")
        for path in sorted(control_data):
            if control_data[path]["sha256"] != candidate_data[path]["sha256"]:
                raise RecordError(
                    f"promotion data binding conflict at {path}: "
                    f"{control_data[path]['sha256']} != {candidate_data[path]['sha256']}"
                )
        data_bindings = [
            {
                "source": str((self.store.root / row["blob"]).resolve()),
                "source_name": str(row["source_name"]),
                "execution_path": path,
            }
            for path, row in sorted(control_data.items())
        ]
        return {"code_bindings": code_bindings, "data_bindings": data_bindings}

    def stage_promotion(
        self,
        candidate_spec_id: str,
        *,
        execution: Mapping[str, Any] | None = None,
        resources: Sequence[Mapping[str, Any]] | None = None,
        runtime: Mapping[str, Any] | None = None,
        reviews: Mapping[str, Any] | None = None,
        seeds: Sequence[int] = PROMOTION_SEEDS,
        minimum_valid_replicates: int = 4,
    ) -> dict[str, Any]:
        """Register, seal, and enqueue a gate-clearing promotion confirmation.

        Protocol-v2 confirmations are reviewless by default.  A caller may still
        supply an optional digest-bound review declaration.  Bindings always come
        from the immutable bank and candidate manifests; callers supply only
        resources and runtime.
        """

        proposal = self.promotion_proposal(
            candidate_spec_id,
            seeds=seeds,
            minimum_valid_replicates=minimum_valid_replicates,
        )
        spec = self.research.create(proposal)
        if execution is not None and (resources is not None or runtime is not None):
            raise RecordError("pass either execution or explicit resources/runtime, not both")
        if execution is not None:
            forbidden = {"code_bindings", "data_bindings"} & set(execution)
            if forbidden:
                raise RecordError(
                    "promotion bindings are immutable; execution may supply only resources/runtime"
                )
            selected_resources = execution.get("resources")
            selected_runtime = execution.get("runtime")
        else:
            selected_resources = resources
            selected_runtime = runtime
        if selected_resources is None or selected_runtime is None:
            candidate_manifest = self._manifest_for_spec(candidate_spec_id)
            if selected_resources is None:
                selected_resources = candidate_manifest["payload"]["resources"]
            if selected_runtime is None:
                selected_runtime = candidate_manifest["payload"]["runtime"]
        bindings = self._combined_immutable_bindings(candidate_spec_id)
        promotion_execution = {
            **bindings,
            "resources": copy.deepcopy(list(selected_resources)),
            "runtime": copy.deepcopy(dict(selected_runtime)),
        }
        manifest = self.sealing.seal(spec["id"], promotion_execution, reviews)
        job = self.queue.enqueue(manifest["id"])
        return {
            "spec": spec,
            "reviews_supplied": reviews is not None,
            "manifest": manifest,
            "job": job,
        }


__all__ = [
    "V2Workflow",
    "duplicate_long_options",
    "preview_execution_bindings",
]
