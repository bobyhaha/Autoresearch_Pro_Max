"""Profile-first diagnostics for a calibrated search bank."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .bank import BankIndex
from .campaign import CampaignQueue
from .protocol import profile_health
from .store import Store


def diagnose(store: Store, *, bank_id: str | None = None) -> dict[str, Any]:
    """Summarize timing and validity before architecture search consumes a frame.

    The report is derived.  It does not bless a benchmark or mutate its bank.  A
    control is model-search-ready only when it is currently bank-eligible and its
    structured timing metrics show that training owns at least 75% of elapsed time.
    Data/compile work may deliberately proceed while this gate is red.
    """

    index = BankIndex.from_store(store)
    results = {row["id"]: row for row in store.list("result_bundle")}
    rows: list[dict[str, Any]] = []
    eligible_by_result = {row["result_id"]: row for row in index.bank_view()["controls"]}
    for control in index.controls:
        if bank_id is not None and control["bank_id"] != bank_id:
            continue
        result = results.get(control["result_id"])
        arms = result["payload"].get("arms", []) if result else []
        arm = next(
            (item for item in arms if isinstance(item, Mapping) and item.get("name") == "control"),
            arms[0] if arms else {},
        )
        metrics = dict(arm.get("metrics", {})) if isinstance(arm, Mapping) else {}
        timing = profile_health(metrics)
        eligibility = eligible_by_result.get(control["result_id"], {})
        rows.append(
            {
                **control,
                "eligible_now": bool(eligibility.get("eligible_now", False)),
                "profile": timing,
                "metrics": {
                    key: metrics[key]
                    for key in (
                        "num_steps",
                        "training_seconds",
                        "total_seconds",
                        "dataloader_seconds",
                        "compile_seconds",
                        "eval_seconds",
                        "total_tokens",
                        "mfu",
                    )
                    if key in metrics
                },
            }
        )

    eligible = [row for row in rows if row["eligible_now"]]
    healthy = [row for row in eligible if row["profile"]["state"] == "healthy"]
    profile_states: dict[str, int] = {}
    for row in rows:
        state = str(row["profile"]["state"])
        profile_states[state] = profile_states.get(state, 0) + 1
    warnings: list[str] = []
    if not rows:
        warnings.append("no valid calibration controls exist for this bank")
    elif not eligible:
        warnings.append("all calibration controls are stale or overused")
    if eligible and len(healthy) != len(eligible):
        warnings.append(
            "one or more eligible controls are unprofiled or overhead-dominated; "
            "optimize input/compile/evaluation before model architecture"
        )
    queue_health = CampaignQueue(store).health()
    if queue_health["paused"]:
        warnings.extend(queue_health["reasons"])
    return {
        "bank_id": bank_id,
        "controls": rows,
        "eligible_controls": len(eligible),
        "healthy_profiles": len(healthy),
        "profile_states": profile_states,
        "model_search_ready": bool(eligible) and len(healthy) == len(eligible),
        "queue_health": queue_health,
        "warnings": warnings,
    }


__all__ = ["diagnose"]
