"""Fast loop: one seed, one arm, one run per experiment.

This is the Karpathy-style search mode.  It keeps the parts that make a result
mean something -- sealed code/data hashes, exact argv, GPU isolation, immutable
records, an integrity verdict -- and drops the parts that cost GPU time:
replicates, paired arms, and the review council.

Two rules are enforced rather than documented, because both are locally
attractive and globally wrong:

* the search seed is fixed for every screen, so comparisons are paired by
  construction (common random numbers) instead of paying full across-seed
  variance on every experiment;
* the seed may never be changed after seeing a result.  Re-rolling a seed to
  improve a number turns the leaderboard into a record of lucky draws.  A seed
  is changed only by `audit`, which re-runs an *already accepted* config on
  held-out seeds to check that the gains transfer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bank import latest_decisions
from .evidence import EvidenceEngine
from .execution import ExecutionService
from .records import RecordError, canonical_json, sha256_file, utc_now
from .research import ResearchEngine
from .sealing import SealingAuthority
from .store import Store

# 2x the paired replicate sd (0.000213) measured on exp_confirm_attn_sink at n=7.
# See mech_placement_bias: the unpaired figure is 10x larger and was what the campaign
# scored against for its first 150 experiments.
# THE COMPARISON GROUP IS DEFINED BY THE FRAME.  fixed_frame_val_bpb_v1 means a 300-second
# wall-clock budget; a run given 600 seconds is not a competitor in it, it is a different
# experiment that happens to report the same metric.
#
# This was a real defect, not a labelling quibble.  The v(s) sweep and the corpus x budget
# 2x2 were sealed with comparison_group = fixed_frame_val_bpb_v1 while overriding
# --time-budget, so eight off-frame arms entered the leaderboard.  One of them ran 600 s,
# scored 0.926958 against a 300 s SOTA of 0.962288, and was reported as the "running best"
# -- a number that looked like a 3.7% breakthrough and was simply double the compute.
#
# Filtering here rather than at the spec is deliberate: ExperimentSpecs are immutable, so
# the eight existing records cannot be relabelled, and the leaderboard is a derived view
# that can and should refuse to rank them.
FRAME_SECONDS = 300
FRAME_OVERRIDE = "--time-budget="

ACCEPT_THRESHOLD = 0.000426


def _off_frame(result_payload: Mapping[str, Any]) -> bool:
    """True if any arm ran a wall-clock budget other than the group's 300 s frame.

    Checked on the recorded argv rather than on a declared field, because the argv is what
    actually ran and is hash-pinned in the manifest. Any override is treated as off-frame,
    including one that happens to equal 300, since the flag's presence means the frame was
    a variable in that experiment rather than the group's fixed definition.
    """
    for arm in result_payload.get("arms", []) or []:
        for token in arm.get("payload_argv") or []:
            token = str(token)
            if token.startswith(FRAME_OVERRIDE):
                try:
                    if int(token.split("=", 1)[1]) != FRAME_SECONDS:
                        return True
                except ValueError:
                    return True
    return False


SEARCH_SEED = 42
AUDIT_SEEDS = (43, 44, 45)
AUDIT_EVERY = 10
COMPARISON_GROUP = "fixed_frame_val_bpb_v1"
METRIC = "val_bpb"


def screen_spec(
    label: str,
    *,
    change_summary: str,
    seed: int,
    argv: Sequence[str],
    minimum_steps: int,
    source_ids: Sequence[str],
    direction: str,
    subsystem: str,
    require_gpu: bool = True,
    isolation: str = "continuous",
) -> dict[str, Any]:
    """A one-replicate, one-arm pilot: the cheapest admissible experiment."""
    spec_id = f"exp_{label}"
    return {
        "id": spec_id,
        "stage": "pilot",
        "title": change_summary,
        "question": f"Does '{change_summary}' lower {METRIC} in the fixed frame?",
        "mechanism": {
            "cause": change_summary,
            "effect": f"{METRIC} at the fixed time budget",
            "chain": [change_summary, "changed training trajectory", f"changed {METRIC}"],
        },
        "hypothesis": {
            "statement": f"{change_summary} lowers {METRIC}.",
            "prediction": f"{METRIC} below the current running best on seed {seed}",
        },
        "falsifier": {"statement": f"{METRIC} at or above the running best refutes the change."},
        "metric": {"name": METRIC, "direction": "minimize"},
        "plan": [
            {
                "replicate_id": f"seed_{seed}",
                "arms": [{"name": "candidate", "argv": list(argv), "env": {}}],
            }
        ],
        "analysis": {
            "effect": "single",
            "primary_arm": "candidate",
            "minimum_valid_replicates": 1,
            # A screen records the endpoint; accept/reject is the leaderboard's
            # job, so the registered rules stay deliberately wide.
            "success_rule": {"op": "lt", "value": 1e9},
            "falsifier_rule": {"op": "gte", "value": 1e9},
            "sota_eligible": False,
        },
        "requirements": {
            "required_metrics": [METRIC, "num_steps"],
            "minimum_steps": minimum_steps,
            "require_gpu": require_gpu,
            "isolation": isolation,
        },
        "knowledge": {
            "source_ids": list(source_ids),
            "direction": direction,
            "subsystem": subsystem,
        },
        "comparison_group": COMPARISON_GROUP,
    }


class FastLoop:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.research = ResearchEngine(store)
        self.sealing = SealingAuthority(store)
        self.execution = ExecutionService(store)
        self.evidence = EvidenceEngine(store)

    def screen(
        self,
        label: str,
        *,
        change_summary: str,
        execution: Mapping[str, Any],
        argv: Sequence[str],
        seed: int = SEARCH_SEED,
        minimum_steps: int = 1,
        source_ids: Sequence[str] = ("baseline",),
        direction: str = "unassigned",
        subsystem: str = "train_py",
        require_gpu: bool = True,
        isolation: str = "continuous",
        base: Path | None = None,
        workers: int = 1,
    ) -> dict[str, Any]:
        """Design -> seal -> execute -> judge -> leaderboard, as one step."""
        proposal = screen_spec(
            label,
            change_summary=change_summary,
            seed=seed,
            argv=argv,
            minimum_steps=minimum_steps,
            source_ids=source_ids,
            direction=direction,
            subsystem=subsystem,
            require_gpu=require_gpu,
            isolation=isolation,
        )
        spec = self.research.create(proposal)
        manifest = self.sealing.seal(spec["id"], execution, None, base=base)
        results, unfinished = self.execution.execute_all(manifest["id"], workers=workers)
        decisions = [self.evidence.judge(row["id"]) for row in results]
        return {
            "spec": spec,
            "manifest": manifest,
            "results": results,
            "decisions": decisions,
            "unfinished": unfinished,
        }

    def leaderboard(self) -> dict[str, Any]:
        """Rebuild the running-best history from immutable EvidenceDecisions only."""
        specs = {row["id"]: row for row in self.store.list("experiment_spec")}
        results = {row["id"]: row for row in self.store.list("result_bundle")}
        entries: list[dict[str, Any]] = []
        # Evidence policy is versioned.  Re-judging a result must replace its
        # projection, not add a second leaderboard point.
        for decision in latest_decisions(self.store.list("evidence_decision")).values():
            payload = decision["payload"]
            result = results.get(payload["result_id"])
            spec = specs.get(payload["spec_id"])
            if result is None or spec is None:
                continue
            if spec["payload"]["comparison_group"] != COMPARISON_GROUP:
                continue
            if _off_frame(result["payload"]):
                # Ran a different wall-clock budget. The comparison group IS the frame, so
                # this is not a competitor -- it is a different experiment reporting the
                # same metric. Excluded from ranking entirely rather than shown and
                # discounted, because a "running best" is read as a claim.
                continue
            arms = payload["measurements"].get("arms", {})
            value = arms.get("candidate", {}).get(METRIC)
            entries.append(
                {
                    "spec_id": spec["id"],
                    "label": spec["payload"]["title"],
                    "seed": result["payload"]["replicate_id"].removeprefix("seed_"),
                    "value": float(value) if isinstance(value, (int, float)) else None,
                    "verdict": payload["measurement_verdict"],
                    "reasons": payload["reasons"],
                    "result_id": payload["result_id"],
                    "evidence_id": decision["id"],
                    "created_at": result["created_at"],
                    "started_at": result["payload"]["lifecycle"]["started_at"],
                    "gpu": result["payload"]["resource"].get("gpu"),
                }
            )
        entries.sort(key=lambda row: (row["created_at"], row["spec_id"]))

        best: float | None = None
        best_spec: str | None = None
        accepted = 0
        for index, row in enumerate(entries):
            row["experiment"] = index
            usable = row["verdict"] == "valid" and row["value"] is not None
            # `kept` compares against the best that was already KNOWN when this
            # screen started, not against whatever happened to finish first.
            # Screens run concurrently on several GPUs, so a plain sequential
            # accept-if-better rule would let arrival order decide which member
            # of a simultaneous batch looks like an improvement.
            known = [
                other["value"]
                for other in entries[:index]
                if other["verdict"] == "valid"
                and other["value"] is not None
                and other["created_at"] <= row["started_at"]
            ]
            reference = min(known) if known else None
            row["baseline_value"] = reference
            row["delta"] = row["value"] - reference if usable and reference is not None else None
            # A STRICT IMPROVEMENT IS NOT AN IMPROVEMENT.  Until 2026-08-09 this line
            # accepted any value below the reference, with no significance test, and 16 of
            # 33 accepted entries turned out to sit below the measurement's own noise --
            # deltas of -0.000005, -0.000012, -0.000035.  Those 16 contributed -0.002038 of
            # apparent progress, 5.7% of the leaderboard's total, which is placement and
            # seed luck ratcheted into the running best and then defended as a result.
            #
            # ACCEPT_THRESHOLD is 2x the paired replicate sd measured on the attention-sink
            # confirmation (0.000213, n=7).  A single paired screen cannot resolve below
            # that, so anything smaller is recorded, charted, and NOT accepted.  `kept_raw`
            # preserves the old predicate so nothing is hidden by the change.
            row["kept_raw"] = bool(usable and (reference is None or row["value"] < reference))
            row["kept"] = bool(
                row["kept_raw"]
                and (reference is None or row["value"] < reference - ACCEPT_THRESHOLD)
            )
            if row["kept"]:
                accepted += 1
            if usable and (best is None or row["value"] < best):
                best = row["value"]
                best_spec = row["spec_id"]
            row["running_best"] = best
        return {
            "generated_at": utc_now(),
            "comparison_group": COMPARISON_GROUP,
            "metric": METRIC,
            "search_seed": SEARCH_SEED,
            "experiments": len(entries),
            "accepted": accepted,
            "invalid": sum(row["verdict"] != "valid" for row in entries),
            "best_value": best,
            "best_spec_id": best_spec,
            "audit_due": accepted > 0 and accepted % AUDIT_EVERY == 0,
            "audit_every": AUDIT_EVERY,
            "audit_seeds": list(AUDIT_SEEDS),
            "entries": entries,
        }

    def write_views(self) -> dict[str, Any]:
        board = self.leaderboard()
        self.store.write_view("LEADERBOARD.json", canonical_json(board) + "\n")
        self.store.write_view("LEADERBOARD.md", _leaderboard_markdown(board))
        return board


def code_identity(execution: Mapping[str, Any], base: Path | None) -> dict[str, str]:
    """SHA-256 of every code binding, so a screen can name the code that ran."""
    identity: dict[str, str] = {}
    for row in execution.get("code_bindings", []):
        source = Path(str(row.get("source", ""))).expanduser()
        if base is not None and not source.is_absolute():
            source = base / source
        if not source.is_file():
            raise RecordError(f"code binding source does not exist: {source}")
        identity[source.name] = sha256_file(source)
    return identity


def _leaderboard_markdown(board: Mapping[str, Any]) -> str:
    best = "n/a" if board["best_value"] is None else f"{board['best_value']:.6f}"
    lines = [
        "# Leaderboard",
        "",
        f"- Metric: **{board['metric']}** (lower is better)",
        f"- Search seed: **{board['search_seed']}** (fixed; never changed after a result)",
        (
            f"- Experiments: {board['experiments']} | kept: {board['accepted']} "
            f"(gate: delta < -{ACCEPT_THRESHOLD:g}) "
            f"| invalid: {board['invalid']}"
        ),
        f"- Running best: **{best}** (`{board['best_spec_id'] or 'none'}`)",
        (
            f"- Seed audit due: **{str(board['audit_due']).lower()}** "
            f"(every {board['audit_every']} accepted changes on seeds {board['audit_seeds']})"
        ),
        "",
        "| # | change | seed | val_bpb | vs. launch baseline | verdict | kept | running best |",
        "|---:|---|---:|---:|---:|---|---|---:|",
    ]
    for row in board["entries"]:
        value = "—" if row["value"] is None else f"{row['value']:.6f}"
        running = "—" if row["running_best"] is None else f"{row['running_best']:.6f}"
        delta = "—" if row.get("delta") is None else f"{row['delta']:+.6f}"
        kept = "**kept**" if row["kept"] else ""
        lines.append(
            f"| {row['experiment']} | {row['label']} | {row['seed']} | {value} | {delta} "
            f"| {row['verdict']} | {kept} | {running} |"
        )
    return "\n".join(lines).rstrip() + "\n"
