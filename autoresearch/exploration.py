"""Structural enforcement of the exploration budget at candidate staging time.

V2 arrived with strong guarantees about *measurement* -- immutable scope, per-GPU
banked controls, held-out-seed promotion -- and no guarantee at all about *what
gets measured*.  Two prior campaigns on this benchmark failed in exactly that gap:

  * A 102-run campaign moved val_bpb 0.9910 -> 0.9726 and then spent its last ~60
    runs producing a top-30 leaderboard whose entire spread was ~3x its own
    control-repeat noise.  Sixty runs that measured nothing.  It had converged
    onto tuning constants inside its own noise floor.
  * A second campaign closed twenty-one directions and found that every lever was
    either too small to move token count at all (bounded by ~1.3e-3 against a
    2.6e-3 gate) or pushed past the repetition wall.  Its own conclusion:
    "the 12-knob box is closed; what is left needs code variants."

Both were fixed afterwards by writing a rule down.  The rule was then followed
correctly and the failure class recurred anyway, which is the finding that this
module exists to act on: *a recurring failure stops recurring when it becomes
structurally impossible, not when it becomes discouraged.*  So the budget lives
inside the same reservation lock that pins a control and seals a manifest, and a
violating candidate is rejected before it can consume a bank use.

Three rules, each traceable to a specific measured failure:

1. **Breadth.**  A non-mechanism candidate is refused unless at least
   ``MINIMUM_MECHANISM_SLOTS`` of the trailing ``WINDOW_SLOTS`` staged candidates
   were mechanisms.  Early in a campaign this means the first several slots must
   all be mechanisms, which is the intended reading.
2. **No premature knob tuning.**  A ``knob`` candidate is refused unless its
   family already produced a candidate that cleared the bank gate.  Tuning a
   constant whose mechanism is still inside the noise band is the exact motion
   that burned those sixty runs.
3. **No repair loop.**  At most ``MAX_CONSECUTIVE_FAMILY`` consecutive candidates
   may come from one family unless that family has cleared the gate.  "A mechanism
   that fails is a result; do not repair-loop a dead idea."

Every rule can be overridden, and an override is never silent: it demands a written
reason that is copied into the immutable ExperimentSpec, so the audit trail shows
both that the budget fired and why a human overrode it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .records import RecordError

#: A candidate must declare which kind of search it is.  ``control`` is not a
#: candidate track -- unchanged controls enter through the bank lane.
CANDIDATE_TRACKS = ("mechanism", "knob", "throughput")

#: Trailing candidates considered by the breadth rule.
WINDOW_SLOTS = 8

#: How many of ``WINDOW_SLOTS`` must be mechanisms.
MINIMUM_MECHANISM_SLOTS = 5

#: Consecutive candidates permitted from one family before it must clear the gate.
MAX_CONSECUTIVE_FAMILY = 2


class ExplorationBudgetError(RecordError):
    """A candidate would violate the exploration budget."""


#: Subsystem prefixes that describe the measuring apparatus rather than the model.
#: These are exempt from the breadth rule for the same reason v2 already exempts
#: them from `--hypothesis-id`: they are not model search, so spending a slot on
#: one is not spending a slot instead of exploring.  Without this exemption the two
#: gates contradict each other -- `doctor` refuses model search while a control is
#: overhead-dominated, and the breadth rule would simultaneously refuse the
#: input/compile/evaluation work that is the only way to make it healthy.
DIAGNOSTIC_SUBSYSTEM_PREFIXES = (
    "calibration",
    "compile",
    "data",
    "evaluation",
    "evaluator",
    "input",
    "instrumentation",
    "tokenizer",
)


def is_diagnostic_subsystem(value: Any) -> bool:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return normalized.startswith(DIAGNOSTIC_SUBSYSTEM_PREFIXES)


def normalize_track(value: Any) -> str:
    track = str(value or "").strip().lower().replace("-", "_")
    if track not in CANDIDATE_TRACKS:
        raise RecordError(
            f"candidate track must be one of {list(CANDIDATE_TRACKS)}; got {value!r}. "
            "Classify the change before staging it: a mechanism is new code on a "
            "structure that has never been run, a knob is a new value for an "
            "existing constant, and a throughput change buys steps rather than "
            "per-step quality."
        )
    return track


def normalize_family(value: Any, *, subsystem: str) -> str:
    family = str(value or subsystem or "").strip().lower().replace("-", "_")
    if not family:
        raise RecordError("candidate family must be non-empty text")
    return family


def _track_of(spec: Mapping[str, Any]) -> str | None:
    search = spec.get("search")
    if not isinstance(search, Mapping):
        return None
    value = search.get("track")
    return str(value) if isinstance(value, str) and value else None


def _family_of(spec: Mapping[str, Any]) -> str:
    search = spec.get("search")
    if isinstance(search, Mapping):
        value = search.get("family")
        if isinstance(value, str) and value:
            return value
    knowledge = spec.get("knowledge")
    if isinstance(knowledge, Mapping):
        value = knowledge.get("subsystem")
        if isinstance(value, str) and value:
            return str(value).strip().lower().replace("-", "_")
    return "unclassified"


def evaluate(
    *,
    track: str,
    family: str,
    staged_candidates: Sequence[Mapping[str, Any]],
    cleared_families: Sequence[str],
    subsystem: str | None = None,
) -> dict[str, Any]:
    """Decide whether one candidate may be staged.

    ``staged_candidates`` are prior candidate-lane spec payloads in staging order,
    oldest first.  ``cleared_families`` are families with at least one candidate
    that reached the promotion queue.  The return value is always a full report so
    the caller can record why staging was permitted, not only why it was refused.
    """

    cleared = {str(item).strip().lower().replace("-", "_") for item in cleared_families}
    # The breadth window looks at the slots this candidate is joining, so it holds
    # WINDOW_SLOTS - 1 prior candidates plus the one being staged.
    window = list(staged_candidates)[-(WINDOW_SLOTS - 1) :]
    mechanism_slots = sum(1 for spec in window if _track_of(spec) == "mechanism")
    trailing_families = [_family_of(spec) for spec in staged_candidates]
    consecutive_family = 0
    for name in reversed(trailing_families):
        if name != family:
            break
        consecutive_family += 1

    diagnostic = is_diagnostic_subsystem(subsystem)

    violations: list[str] = []
    if not diagnostic and track != "mechanism" and mechanism_slots < MINIMUM_MECHANISM_SLOTS:
        violations.append(
            f"breadth: only {mechanism_slots} of the trailing {len(window)} staged "
            f"candidates are mechanisms (need {MINIMUM_MECHANISM_SLOTS} of "
            f"{WINDOW_SLOTS} before a {track} slot). Stage a mechanism that has "
            "never been run in this campaign instead."
        )
    if not diagnostic and track == "knob" and family not in cleared:
        violations.append(
            f"premature knob tuning: family {family!r} has never cleared the bank "
            "gate, so its constants are being tuned inside the noise band. Clear "
            "the mechanism first, or stage this as a mechanism if it is one."
        )
    if not diagnostic and consecutive_family >= MAX_CONSECUTIVE_FAMILY and family not in cleared:
        violations.append(
            f"repair loop: the last {consecutive_family} staged candidates were "
            f"already family {family!r} and it has not cleared the gate. A "
            "mechanism that fails is a result -- record why and open a family no "
            "prior candidate has touched."
        )

    return {
        "track": track,
        "family": family,
        "diagnostic_subsystem": diagnostic,
        "window_slots": len(window),
        "mechanism_slots": mechanism_slots,
        "consecutive_family": consecutive_family,
        "family_cleared_gate": family in cleared,
        "violations": violations,
    }


def enforce(
    *,
    track: str,
    family: str,
    staged_candidates: Sequence[Mapping[str, Any]],
    cleared_families: Sequence[str],
    override_reason: str | None = None,
    subsystem: str | None = None,
) -> dict[str, Any]:
    """Raise unless the candidate fits the budget or carries a written override."""

    report = evaluate(
        track=track,
        family=family,
        staged_candidates=staged_candidates,
        cleared_families=cleared_families,
        subsystem=subsystem,
    )
    if not report["violations"]:
        report["override_reason"] = None
        return report
    reason = str(override_reason or "").strip()
    if not reason:
        joined = "; ".join(report["violations"])
        raise ExplorationBudgetError(f"exploration budget refuses this candidate -- {joined}")
    if len(reason) < 20:
        raise ExplorationBudgetError(
            "an exploration-budget override must state a reason of at least 20 "
            "characters; it is copied into the immutable spec"
        )
    report["override_reason"] = reason
    return report


__all__ = [
    "CANDIDATE_TRACKS",
    "DIAGNOSTIC_SUBSYSTEM_PREFIXES",
    "MAX_CONSECUTIVE_FAMILY",
    "MINIMUM_MECHANISM_SLOTS",
    "WINDOW_SLOTS",
    "ExplorationBudgetError",
    "enforce",
    "evaluate",
    "is_diagnostic_subsystem",
    "normalize_family",
    "normalize_track",
]
