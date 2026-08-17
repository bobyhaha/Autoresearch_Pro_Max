"""Knowledge Engine: rebuild beliefs, portfolio, paper cadence, and verified SOTA."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from .bank import BankIndex, context_fingerprint, search_lane, search_metadata
from .protocol import EVIDENCE_POLICY_VERSION, promotion_code_roots
from .records import RecordError, canonical_json, mean, rule_matches, sha256_object, utc_now
from .science import ScientificLibrary
from .store import Store


class KnowledgeEngine:
    def __init__(self, store: Store) -> None:
        self.store = store

    def synthesize(self) -> dict[str, Any]:
        validation = self.store.validate()
        if not validation["valid"]:
            preview = "; ".join(validation["errors"][:3])
            raise RecordError(f"knowledge synthesis requires a valid registry: {preview}")
        specs = {row["id"]: row for row in self.store.list("experiment_spec")}
        manifests = {row["id"]: row for row in self.store.list("execution_manifest")}
        results = {row["id"]: row for row in self.store.list("result_bundle")}
        decisions = _latest_decisions(self.store.list("evidence_decision"))
        papers = self.store.list("paper")

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for decision in decisions.values():
            grouped[decision["payload"]["spec_id"]].append(decision)

        beliefs: list[dict[str, Any]] = []
        evidence_counts = Counter(
            decision["payload"]["measurement_verdict"] for decision in decisions.values()
        )
        for spec_id, spec in sorted(specs.items(), key=lambda item: item[1]["created_at"]):
            rows = grouped.get(spec_id, [])
            plan_complete: bool | None = None
            if (
                spec["payload"].get("protocol_version") == 2
                and spec["payload"].get("stage") == "confirmation"
            ):
                planned = [
                    str(replicate["replicate_id"]) for replicate in spec["payload"].get("plan", [])
                ]
                terminal: dict[str, list[str]] = defaultdict(list)
                for result in results.values():
                    if result["payload"].get("spec_id") != spec_id:
                        continue
                    if result["id"] not in decisions:
                        continue
                    terminal[str(result["payload"].get("replicate_id"))].append(result["id"])
                plan_complete = all(
                    len(terminal.get(replicate_id, [])) == 1 for replicate_id in planned
                )
            beliefs.append(_belief(spec, rows, plan_complete=plan_complete))

        covered_specs = {
            spec_id for paper in papers for spec_id in paper["payload"].get("spec_ids", [])
        }
        completed_confirmation_specs = {
            row["spec_id"]
            for row in beliefs
            if row["stage"] == "confirmation" and row["status"] not in {"untested", "preliminary"}
        }
        unpublished = sorted(completed_confirmation_specs - covered_specs)
        paper_status = {
            "completed_papers": len(papers),
            "unpublished_confirmation_rounds": len(unpublished),
            "unpublished_spec_ids": unpublished,
            "paper_due": len(unpublished) >= 5,
            "rounds_until_due": max(0, 5 - len(unpublished)),
        }

        sota, sota_blockers = _select_sota(beliefs, specs, manifests, results, decisions)
        portfolio = _portfolio(specs, beliefs, evidence_counts)
        snapshot = {
            "generated_at": utc_now(),
            "counts": {
                "specs": len(specs),
                "manifests": len(manifests),
                "results": len(results),
                "evidence": len(decisions),
                "papers": len(papers),
            },
            "evidence_counts": {
                "valid": evidence_counts["valid"],
                "invalid": evidence_counts["invalid"],
                "unknown": evidence_counts["unknown"],
            },
            "beliefs": beliefs,
            "portfolio": portfolio,
            "paper_status": paper_status,
            "sota": sota,
            "sota_blockers": sota_blockers,
        }
        scientific = ScientificLibrary(self.store).synthesize()
        snapshot["scientific_library"] = scientific["summary"]
        self._write_views(snapshot)
        return snapshot

    def _write_views(self, snapshot: Mapping[str, Any]) -> None:
        self.store.write_view("KNOWLEDGE.json", canonical_json(snapshot) + "\n")
        self.store.write_view("SOTA.json", canonical_json(snapshot["sota"]) + "\n")
        self.store.write_view("BELIEFS.md", _beliefs_markdown(snapshot))
        self.store.write_view("PORTFOLIO.md", _portfolio_markdown(snapshot))
        self.store.write_view("PAPERS.md", _papers_markdown(snapshot))
        self._append_sota_log(snapshot)

    def _append_sota_log(self, snapshot: Mapping[str, Any]) -> None:
        """Append a trophy entry to SOTA_LOG.md when, and only when, SOTA actually moves.

        The visual weight is deliberately scarce.  A trophy marks a *promotion by the
        Knowledge Engine*, which requires supported, claim-eligible CONFIRMATION evidence
        in the comparison group.  It never marks a search-mode running best -- the
        leaderboard's running best has been below the SOTA figure for most of this
        campaign (0.961449 against 0.962288), because single paired screens are
        structurally ineligible and one of them was a co-tenancy artefact that would have
        promoted a false SOTA by 8x.  If a trophy could be earned by a screen it would
        stop meaning anything, which is the whole reason for the convention.

        Appends only on CHANGE, so re-running synthesize is idempotent.
        """
        views = self.store.views_dir
        views.mkdir(parents=True, exist_ok=True)
        log = views / "SOTA_LOG.md"
        header = (
            "# SOTA log\n\n"
            "A trophy is written **only** when the Knowledge Engine promotes a new SOTA:\n"
            "supported, claim-eligible confirmation evidence in the comparison group.\n"
            "Search-mode running bests, inconclusive confirmations and single paired\n"
            "screens stay in plain text no matter how good the number looks.\n\n"
        )
        if not log.exists():
            log.write_text(header)
        existing = log.read_text()
        lines = []
        for group, entry in sorted((snapshot.get("sota") or {}).items()):
            value = entry.get("value")
            spec = entry.get("spec_id")
            if value is None or spec is None:
                continue
            marker = f"<!-- {group}:{spec}:{value!r} -->"
            if marker in existing:
                continue  # already logged; synthesize is idempotent
            n = len(entry.get("replicate_values") or [])
            lines.append(
                f"\n## \U0001f3c6 **NEW SOTA: {value:.6f}** \u2014 `{spec}`, n={n}\n\n"
                f"{marker}\n\n"
                f"- comparison group: `{group}`\n"
                f"- replicate values: "
                + ", ".join(f"{v:.6f}" for v in (entry.get("replicate_values") or []))
                + (
                    "\n- verified seeds: "
                    + ", ".join(str(seed) for seed in entry["replicate_seeds"])
                    if entry.get("replicate_seeds") is not None
                    else ""
                )
                + "\n- promoted by the Knowledge Engine from confirmation evidence;"
                " not a search-mode result\n"
            )
        if lines:
            log.write_text(existing + "".join(lines))


def _latest_decisions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        result_id = row["payload"]["result_id"]
        previous = latest.get(result_id)
        if previous is None or row["created_at"] > previous["created_at"]:
            latest[result_id] = row
    return latest


def _belief(
    spec: Mapping[str, Any],
    decisions: list[dict[str, Any]],
    *,
    plan_complete: bool | None = None,
) -> dict[str, Any]:
    payload = spec["payload"]
    valid_effects = [
        row
        for row in decisions
        if row["payload"]["measurement_verdict"] == "valid"
        and isinstance(row["payload"]["measurements"].get("effect_value"), (int, float))
    ]
    eligible = [row for row in valid_effects if row["payload"]["claim_status"] == "eligible"]
    exploratory = [row for row in valid_effects if row["payload"]["claim_status"] == "ineligible"]
    required = int(payload["analysis"]["minimum_valid_replicates"])
    selected = exploratory if payload["stage"] == "pilot" else eligible
    protocol_v2_confirmation = (
        payload.get("protocol_version") == 2 and payload["stage"] == "confirmation"
    )
    replicate_seeds: list[int] | None = None
    if protocol_v2_confirmation:
        rows_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted(selected, key=lambda item: (item["created_at"], item["id"])):
            row_payload = row["payload"]
            seed = row_payload.get("verified_seed")
            if (
                row_payload.get("policy_version") != EVIDENCE_POLICY_VERSION
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
            ):
                continue
            rows_by_seed[seed].append(row)
        by_seed = {seed: rows[0] for seed, rows in rows_by_seed.items() if len(rows) == 1}
        planned_order = [
            row.get("seed")
            for row in payload.get("plan", [])
            if isinstance(row.get("seed"), int) and not isinstance(row.get("seed"), bool)
        ]
        replicate_seeds = []
        for seed in planned_order:
            if seed in by_seed and seed not in replicate_seeds:
                replicate_seeds.append(seed)
        selected = [by_seed[seed] for seed in replicate_seeds]
    values = [float(row["payload"]["measurements"]["effect_value"]) for row in selected]
    aggregate = mean(values) if values else None
    if payload["stage"] == "pilot":
        status = "exploratory" if values else "untested"
    elif (protocol_v2_confirmation and plan_complete is False) or len(values) < required:
        status = "preliminary" if values else "untested"
    elif rule_matches(float(aggregate), payload["analysis"]["success_rule"]):
        status = "supported"
    elif rule_matches(float(aggregate), payload["analysis"]["falsifier_rule"]):
        status = "refuted"
    else:
        status = "inconclusive"
    belief = {
        "spec_id": spec["id"],
        "stage": payload["stage"],
        "title": payload["title"],
        "hypothesis": payload["hypothesis"]["statement"],
        "direction": payload["knowledge"]["direction"],
        "subsystem": payload["knowledge"]["subsystem"],
        "status": status,
        "valid_replicates": len(values),
        "required_replicates": required,
        "effect_mean": aggregate,
        "effect_stdev": statistics.stdev(values) if len(values) > 1 else None,
        "evidence_ids": [row["id"] for row in selected],
    }
    if replicate_seeds is not None:
        belief["replicate_seeds"] = replicate_seeds
        if plan_complete is not None:
            belief["plan_complete"] = plan_complete
    return belief


def _binding_map(manifest: Mapping[str, Any], group: str) -> dict[str, str] | None:
    rows = manifest.get("payload", {}).get(group, [])
    if not isinstance(rows, list):
        return None
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        path = row.get("execution_path")
        digest = row.get("sha256")
        if not isinstance(path, str) or not path or not isinstance(digest, str) or not digest:
            return None
        if path in result:
            return None
        result[path] = digest
    return result


def _source_arm(
    manifest: Mapping[str, Any], name: str, *, sole_fallback: bool = False
) -> Mapping[str, Any] | None:
    plan = manifest.get("payload", {}).get("plan", [])
    if not isinstance(plan, list) or not plan or not isinstance(plan[0], Mapping):
        return None
    arms = plan[0].get("arms", [])
    if not isinstance(arms, list):
        return None
    matches = [row for row in arms if isinstance(row, Mapping) and row.get("name") == name]
    if len(matches) == 1:
        return matches[0]
    if sole_fallback and len(arms) == 1 and isinstance(arms[0], Mapping):
        return arms[0]
    return None


def _snapshot_argv(
    arm: Mapping[str, Any], manifest: Mapping[str, Any], root: str
) -> list[str] | None:
    code = _binding_map(manifest, "code_bindings")
    argv = arm.get("argv")
    if code is None or not isinstance(argv, list):
        return None
    rewritten: list[str] = []
    entered = False
    for raw in argv:
        token = str(raw)
        normalized = token.removeprefix("./")
        if normalized in code and not token.startswith("/"):
            rewritten.append(f"{root}/{normalized}")
            entered = True
        else:
            rewritten.append(token)
    return rewritten if entered else None


def _promotion_provenance_reason(
    spec: Mapping[str, Any],
    *,
    specs: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    bank_index: BankIndex,
) -> str | None:
    """Recompute a v2 promotion's lineage from immutable pilot records.

    ``lane=promotion`` is a label, not evidence.  This gate proves that the promoted
    arms are byte-for-byte snapshots of one scored candidate and its exact frozen
    bank control, and that confirmation uses held-out seeds under one manifest.
    """

    payload = spec.get("payload", {})
    search = search_metadata(spec)
    link_fields = (
        "source_spec_id",
        "source_manifest_id",
        "source_result_id",
        "control_result_id",
        "control_manifest_id",
    )
    if any(
        not isinstance(search.get(field), str) or not search.get(field) for field in link_fields
    ):
        return "immutable candidate/control source links are incomplete"

    source_spec_id = str(search["source_spec_id"])
    source_manifest_id = str(search["source_manifest_id"])
    source_result_id = str(search["source_result_id"])
    control_result_id = str(search["control_result_id"])
    control_manifest_id = str(search["control_manifest_id"])
    candidate_spec = specs.get(source_spec_id)
    candidate_manifest = manifests.get(source_manifest_id)
    candidate_result = results.get(source_result_id)
    control_result = results.get(control_result_id)
    control_manifest = manifests.get(control_manifest_id)
    if any(
        row is None
        for row in (
            candidate_spec,
            candidate_manifest,
            candidate_result,
            control_result,
            control_manifest,
        )
    ):
        return "a linked candidate/control source record is missing"
    assert candidate_spec is not None
    assert candidate_manifest is not None
    assert candidate_result is not None
    assert control_result is not None
    assert control_manifest is not None
    if search_lane(candidate_spec) != "candidate":
        return "source_spec_id is not a candidate-lane pilot"
    if (
        candidate_manifest["payload"].get("spec_id") != source_spec_id
        or candidate_result["payload"].get("spec_id") != source_spec_id
        or candidate_result["payload"].get("manifest_id") != source_manifest_id
    ):
        return "candidate source links disagree"
    if control_result["payload"].get("manifest_id") != control_manifest_id:
        return "control source links disagree"
    control_spec = specs.get(str(control_result["payload"].get("spec_id", "")))
    if control_spec is None or search_lane(control_spec) != "bank":
        return "control_result_id is not a bank-lane result"
    if control_manifest["payload"].get("spec_id") != control_spec["id"]:
        return "control manifest disagrees with its bank spec"

    candidate_results = [
        row for row in results.values() if row["payload"].get("spec_id") == source_spec_id
    ]
    if len(candidate_results) != 1 or candidate_results[0]["id"] != source_result_id:
        return "promotion source is not the candidate's unique landed result"
    candidate_manifests = [
        row for row in manifests.values() if row["payload"].get("spec_id") == source_spec_id
    ]
    if len(candidate_manifests) != 1 or candidate_manifests[0]["id"] != source_manifest_id:
        return "candidate pilot is not bound to exactly one immutable manifest"

    score = bank_index.score_candidate(source_result_id)
    if score.get("status") != "scored" or score.get("promotion_due") is not True:
        return "linked candidate did not pass the frozen bank promotion gate"
    if score.get("control_result_id") != control_result_id:
        return "linked control is not the candidate's frozen bank reference"
    candidate_search = search_metadata(candidate_spec)
    try:
        computed_context = context_fingerprint(candidate_spec, candidate_manifest)
    except RecordError as exc:
        return f"candidate benchmark context cannot be reconstructed: {exc}"
    for field, computed in (
        ("bank_id", score.get("bank_id")),
        ("baseline_fingerprint", score.get("baseline_fingerprint")),
        ("context_fingerprint", computed_context),
    ):
        if search.get(field) != computed or candidate_search.get(field) != computed:
            return f"promotion {field} does not match the scored candidate"
    declared_delta = search.get("source_delta")
    if (
        isinstance(declared_delta, bool)
        or not isinstance(declared_delta, (int, float))
        or not math.isclose(float(declared_delta), float(score["delta"]), abs_tol=1e-12)
    ):
        return "promotion source_delta does not match the recomputed bank score"

    for field in ("comparison_group", "scope", "metric", "requirements"):
        if payload.get(field) != candidate_spec["payload"].get(field):
            return f"promotion changes candidate {field}"

    promotion_results = [
        row for row in results.values() if row["payload"].get("spec_id") == spec["id"]
    ]
    promotion_manifest_ids = {str(row["payload"].get("manifest_id")) for row in promotion_results}
    registered_manifests = [
        row for row in manifests.values() if row["payload"].get("spec_id") == spec["id"]
    ]
    if len(promotion_manifest_ids) != 1 or len(registered_manifests) != 1:
        return "all promotion replicates must share one immutable manifest"
    promotion_manifest_id = next(iter(promotion_manifest_ids))
    promotion_manifest = manifests.get(promotion_manifest_id)
    if promotion_manifest is None or registered_manifests[0]["id"] != promotion_manifest_id:
        return "promotion results disagree with the sole sealed manifest"

    roots = search.get("arm_code_roots")
    if not isinstance(roots, Mapping):
        return "promotion does not declare separate control/candidate code roots"
    control_root = roots.get("control")
    candidate_root = roots.get("candidate")
    expected_roots = promotion_code_roots(control_manifest_id, source_manifest_id)
    if dict(roots) != expected_roots:
        return "promotion arm_code_roots are not the canonical disjoint namespaces"
    if (
        not isinstance(control_root, str)
        or not isinstance(candidate_root, str)
        or not control_root
        or not candidate_root
        or control_root == candidate_root
        or any(
            root.startswith("/") or ".." in root.split("/")
            for root in (control_root, candidate_root)
        )
    ):
        return "promotion arm_code_roots are unsafe or ambiguous"

    control_code = _binding_map(control_manifest, "code_bindings")
    candidate_code = _binding_map(candidate_manifest, "code_bindings")
    promotion_code = _binding_map(promotion_manifest, "code_bindings")
    if control_code is None or candidate_code is None or promotion_code is None:
        return "promotion code binding identity is malformed"
    expanded_control = {f"{control_root}/{path}": digest for path, digest in control_code.items()}
    expanded_candidate = {
        f"{candidate_root}/{path}": digest for path, digest in candidate_code.items()
    }
    collisions = sorted(set(expanded_control) & set(expanded_candidate))
    if collisions:
        return f"promotion control/candidate code namespaces collide: {collisions}"
    expected_code = {**expanded_control, **expanded_candidate}
    if len(expected_code) != len(control_code) + len(candidate_code):
        return "promotion code namespace expansion is not one-to-one"
    if promotion_code != expected_code:
        return "promotion code bindings are not exact isolated source snapshots"
    control_data = _binding_map(control_manifest, "data_bindings")
    candidate_data = _binding_map(candidate_manifest, "data_bindings")
    promotion_data = _binding_map(promotion_manifest, "data_bindings")
    if control_data is None or control_data != candidate_data or control_data != promotion_data:
        return "promotion changes the candidate/control data bindings"

    control_arm = _source_arm(control_manifest, "control", sole_fallback=True)
    candidate_arm = _source_arm(candidate_manifest, "candidate", sole_fallback=True)
    if control_arm is None or candidate_arm is None:
        return "source manifests do not have unique control/candidate arms"
    expected_argv = {
        "control": _snapshot_argv(control_arm, control_manifest, control_root),
        "candidate": _snapshot_argv(candidate_arm, candidate_manifest, candidate_root),
    }
    if any(value is None for value in expected_argv.values()):
        return "source arm does not directly execute its sealed code binding"

    promotion_plan = promotion_manifest["payload"].get("plan", [])
    if not isinstance(promotion_plan, list):
        return "promotion plan is malformed"
    observed_seeds = {
        row.get("seed")
        for manifest in (candidate_manifest, control_manifest)
        for row in manifest["payload"].get("plan", [])
        if isinstance(row, Mapping)
    }
    for index, replicate in enumerate(promotion_plan):
        if not isinstance(replicate, Mapping):
            return "promotion plan contains a malformed replicate"
        seed = replicate.get("seed")
        if seed in observed_seeds:
            return "promotion reuses a bank/search seed instead of a held-out seed"
        arms = replicate.get("arms", [])
        expected_order = ["control", "candidate"] if index % 2 == 0 else ["candidate", "control"]
        if not isinstance(arms, list) or [arm.get("name") for arm in arms] != expected_order:
            return "promotion arm order is not preregistered AB/BA counterbalancing"
        for arm in arms:
            name = str(arm["name"])
            source_arm = control_arm if name == "control" else candidate_arm
            if arm.get("argv") != expected_argv[name]:
                return f"promotion {name} argv differs from its immutable source"
            expected_env = dict(source_arm.get("env", {}))
            expected_env["AUTORESEARCH_SEED"] = str(seed)
            if arm.get("env", {}) != expected_env:
                return f"promotion {name} environment differs from its immutable source"
    return None


def _select_sota(
    beliefs: list[dict[str, Any]],
    specs: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    supported = {row["spec_id"] for row in beliefs if row["status"] == "supported"}
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    bank_index = BankIndex(
        list(specs.values()),
        list(manifests.values()),
        list(results.values()),
        list(decisions.values()),
    )
    provenance_cache: dict[str, str | None] = {}
    provenance_blockers: dict[str, str] = {}
    for result_id, decision in decisions.items():
        payload = decision["payload"]
        spec = specs.get(payload["spec_id"])
        result = results.get(result_id)
        if (
            spec is None
            or result is None
            or spec["id"] not in supported
            or payload["measurement_verdict"] != "valid"
            or payload["claim_status"] != "eligible"
            or not spec["payload"]["analysis"].get("sota_eligible", False)
        ):
            continue
        protocol_v2 = spec["payload"].get("protocol_version") == 2
        if protocol_v2 and spec["payload"].get("search", {}).get("lane") != "promotion":
            continue
        if protocol_v2:
            if spec["id"] not in provenance_cache:
                provenance_cache[spec["id"]] = _promotion_provenance_reason(
                    spec,
                    specs=specs,
                    manifests=manifests,
                    results=results,
                    bank_index=bank_index,
                )
            reason = provenance_cache[spec["id"]]
            if reason is not None:
                group = str(spec["payload"].get("comparison_group", spec["id"]))
                provenance_blockers[group] = f"promotion provenance rejected: {reason}"
                continue
        verified_seed: int | None = None
        if protocol_v2:
            candidate_seed = payload.get("verified_seed")
            if (
                payload.get("policy_version") != EVIDENCE_POLICY_VERSION
                or isinstance(candidate_seed, bool)
                or not isinstance(candidate_seed, int)
                or candidate_seed < 0
            ):
                continue
            verified_seed = candidate_seed
        metric = spec["payload"]["metric"]["name"]
        arm = spec["payload"]["analysis"]["primary_arm"]
        value = payload["measurements"].get("arms", {}).get(arm, {}).get(metric)
        if not isinstance(value, (int, float)):
            continue
        candidates[spec["payload"]["comparison_group"]][spec["id"]].append(
            {
                "value": float(value),
                "result_id": result_id,
                "spec_id": spec["id"],
                "manifest_id": result["payload"]["manifest_id"],
                "metric": metric,
                "direction": spec["payload"]["metric"]["direction"],
                "protocol_version": spec["payload"].get("protocol_version"),
                "verified_seed": verified_seed,
            }
        )
    selected: dict[str, Any] = {}
    blockers: dict[str, str] = dict(provenance_blockers)
    for group, by_spec in candidates.items():
        scope_signatures = {
            canonical_json(
                {
                    "scope": specs[spec_id]["payload"].get("scope"),
                    # Generated v2 workflows derive this from all data plus stable
                    # evaluator/code bindings.  Equal prose scope labels cannot make
                    # different tokenizer/evaluator artifacts comparable.
                    "context_fingerprint": specs[spec_id]["payload"]
                    .get("search", {})
                    .get("context_fingerprint"),
                }
            )
            for spec_id in by_spec
        }
        if len(scope_signatures) != 1:
            blockers[group] = (
                "candidate specs reuse one comparison group across different structured "
                "scopes or sealed benchmark contexts"
            )
            continue
        aggregates: list[dict[str, Any]] = []
        for spec_id, rows in by_spec.items():
            spec_payload = specs[spec_id]["payload"]
            protocol_v2 = spec_payload.get("protocol_version") == 2
            replicate_seeds: list[int] | None = None
            if protocol_v2:
                planned_replicates = [str(row["replicate_id"]) for row in spec_payload["plan"]]
                terminal_results = [
                    result
                    for result in results.values()
                    if result["payload"].get("spec_id") == spec_id
                ]
                terminal_by_replicate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
                for result in terminal_results:
                    terminal_by_replicate[str(result["payload"].get("replicate_id"))].append(result)
                # Minimum-valid is tolerance for terminal invalid measurements, not
                # permission to publish while preregistered seeds are still pending.
                if any(
                    len(terminal_by_replicate.get(replicate_id, [])) != 1
                    for replicate_id in planned_replicates
                ) or any(
                    rows_for_replicate[0]["id"] not in decisions
                    for replicate_id, rows_for_replicate in terminal_by_replicate.items()
                    if replicate_id in set(planned_replicates) and len(rows_for_replicate) == 1
                ):
                    continue
                rows_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for row in sorted(rows, key=lambda item: item["result_id"]):
                    rows_by_seed[row["verified_seed"]].append(row)
                by_seed = {
                    seed: seed_rows[0]
                    for seed, seed_rows in rows_by_seed.items()
                    if len(seed_rows) == 1
                }
                planned_order = [row["seed"] for row in spec_payload["plan"]]
                replicate_seeds = []
                for seed in planned_order:
                    if seed in by_seed and seed not in replicate_seeds:
                        replicate_seeds.append(seed)
                rows = [by_seed[seed] for seed in replicate_seeds]
                required = int(spec_payload["analysis"]["minimum_valid_replicates"])
                if len(rows) < required:
                    continue
            manifest_ids = sorted({row["manifest_id"] for row in rows})
            code_snapshots: dict[str, dict[str, Any]] = {}
            data_snapshots: dict[str, dict[str, Any]] = {}
            for manifest_id in manifest_ids:
                manifest = manifests[manifest_id]
                for binding in manifest["payload"]["code_bindings"]:
                    code_snapshots[binding["sha256"]] = {
                        "sha256": binding["sha256"],
                        "blob": binding["blob"],
                        "name": binding["source_name"],
                    }
                for binding in manifest["payload"]["data_bindings"]:
                    data_snapshots[binding["sha256"]] = {
                        "sha256": binding["sha256"],
                        "blob": binding["blob"],
                        "name": binding["source_name"],
                    }
            aggregate = {
                "value": mean([row["value"] for row in rows]),
                "replicate_values": [row["value"] for row in rows],
                "result_ids": [row["result_id"] for row in rows],
                "spec_id": spec_id,
                "manifest_ids": manifest_ids,
                "metric": rows[0]["metric"],
                "direction": rows[0]["direction"],
                "code_snapshots": list(code_snapshots.values()),
                "data_snapshots": list(data_snapshots.values()),
                "scope": spec_payload.get("scope"),
                "scope_digest": sha256_object(spec_payload.get("scope")),
            }
            if replicate_seeds is not None:
                aggregate["replicate_seeds"] = replicate_seeds
            aggregates.append(aggregate)
        if not aggregates:
            continue
        signatures = {(row["metric"], row["direction"]) for row in aggregates}
        if len(signatures) != 1:
            blockers[group] = "candidate specs disagree on metric name or optimization direction"
            continue
        direction = aggregates[0]["direction"]
        selected[group] = (
            min(aggregates, key=lambda row: row["value"])
            if direction == "minimize"
            else max(aggregates, key=lambda row: row["value"])
        )
        blockers.pop(group, None)
    return selected, blockers


def _portfolio(
    specs: Mapping[str, Mapping[str, Any]],
    beliefs: list[dict[str, Any]],
    evidence_counts: Counter,
) -> dict[str, Any]:
    confirmation = [row for row in beliefs if row["stage"] == "confirmation"]
    recent = confirmation[-3:]
    recent_directions = [row["direction"] for row in recent]
    preliminary = [row["spec_id"] for row in confirmation if row["status"] == "preliminary"]
    invalid = evidence_counts["invalid"]
    valid = evidence_counts["valid"]
    if invalid > valid and invalid >= 2:
        next_mode = "resolve_integrity"
        reason = "invalid measurements outnumber valid measurements"
    elif preliminary:
        next_mode = "confirm"
        reason = "a preregistered confirmation has not reached its minimum replicate count"
    elif len(recent_directions) == 3 and len(set(recent_directions)) == 1:
        next_mode = "explore"
        reason = "three consecutive confirmation rounds target the same direction"
    else:
        next_mode = "develop"
        reason = "the portfolio is not blocked or saturated"
    return {
        "required_next_mode": next_mode,
        "reason": reason,
        "preliminary_spec_ids": preliminary,
        "direction_counts": dict(Counter(row["direction"] for row in confirmation)),
        "recent_directions": recent_directions,
        "total_specs": len(specs),
    }


def _beliefs_markdown(snapshot: Mapping[str, Any]) -> str:
    lines = ["# Beliefs", "", "Generated only from immutable EvidenceDecisions.", ""]
    for row in snapshot["beliefs"]:
        effect = "n/a" if row["effect_mean"] is None else f"{row['effect_mean']:.6g}"
        lines.extend(
            [
                f"## {row['title']}",
                "",
                f"- Spec: `{row['spec_id']}`",
                f"- Stage/status: **{row['stage']} / {row['status']}**",
                f"- Hypothesis: {row['hypothesis']}",
                f"- Valid replicates: {row['valid_replicates']}/{row['required_replicates']}",
                f"- Mean preregistered effect: {effect}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _portfolio_markdown(snapshot: Mapping[str, Any]) -> str:
    row = snapshot["portfolio"]
    return (
        "# Portfolio\n\n"
        f"- Required next mode: **{row['required_next_mode']}**\n"
        f"- Reason: {row['reason']}\n"
        f"- Direction counts: `{json.dumps(row['direction_counts'], sort_keys=True)}`\n"
        f"- Preliminary specs: `{json.dumps(row['preliminary_spec_ids'])}`\n"
    )


def _papers_markdown(snapshot: Mapping[str, Any]) -> str:
    row = snapshot["paper_status"]
    return (
        "# Paper cadence\n\n"
        f"- Paper due: **{str(row['paper_due']).lower()}**\n"
        f"- Unpublished confirmation rounds: {row['unpublished_confirmation_rounds']}\n"
        f"- Rounds until gate: {row['rounds_until_due']}\n"
        f"- Unpublished specs: `{json.dumps(row['unpublished_spec_ids'])}`\n"
    )
