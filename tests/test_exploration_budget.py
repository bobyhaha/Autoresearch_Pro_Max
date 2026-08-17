from __future__ import annotations

import pytest

from autoresearch.exploration import (
    MAX_CONSECUTIVE_FAMILY,
    MINIMUM_MECHANISM_SLOTS,
    ExplorationBudgetError,
    enforce,
    evaluate,
    is_diagnostic_subsystem,
    normalize_family,
    normalize_track,
)
from autoresearch.records import RecordError


def _staged(track: str, family: str) -> dict:
    return {"search": {"track": track, "family": family}, "knowledge": {"subsystem": family}}


def test_track_must_be_classified_before_staging():
    with pytest.raises(RecordError):
        normalize_track("interesting")
    with pytest.raises(RecordError):
        normalize_track("")
    assert normalize_track("Mechanism") == "mechanism"
    assert normalize_track("  THROUGHPUT ") == "throughput"


def test_family_defaults_to_subsystem():
    assert normalize_family(None, subsystem="Short-Window") == "short_window"
    assert normalize_family("Attention", subsystem="ignored") == "attention"
    with pytest.raises(RecordError):
        normalize_family(None, subsystem="")


def test_an_early_campaign_may_only_stage_mechanisms():
    # The first slots of a campaign have no mechanism history to draw down, so a
    # knob or throughput candidate is refused outright.
    report = evaluate(track="knob", family="optimizer", staged_candidates=[], cleared_families=[])
    assert report["mechanism_slots"] == 0
    assert any("breadth" in violation for violation in report["violations"])

    ok = evaluate(track="mechanism", family="optimizer", staged_candidates=[], cleared_families=[])
    assert ok["violations"] == []


def test_breadth_opens_a_non_mechanism_slot_once_enough_mechanisms_ran():
    history = [_staged("mechanism", f"family_{index}") for index in range(MINIMUM_MECHANISM_SLOTS)]
    report = evaluate(
        track="throughput",
        family="kernel",
        staged_candidates=history,
        cleared_families=[],
    )
    assert report["mechanism_slots"] == MINIMUM_MECHANISM_SLOTS
    assert report["violations"] == []


def test_a_knob_is_refused_until_its_family_clears_the_bank_gate():
    history = [_staged("mechanism", f"family_{index}") for index in range(MINIMUM_MECHANISM_SLOTS)]
    blocked = evaluate(
        track="knob",
        family="optimizer",
        staged_candidates=history,
        cleared_families=[],
    )
    assert any("premature knob tuning" in violation for violation in blocked["violations"])

    allowed = evaluate(
        track="knob",
        family="optimizer",
        staged_candidates=history,
        cleared_families=["optimizer"],
    )
    assert allowed["violations"] == []
    assert allowed["family_cleared_gate"] is True


def test_repair_loop_rule_blocks_a_third_consecutive_candidate_in_one_family():
    history = [
        *[_staged("mechanism", f"family_{index}") for index in range(MINIMUM_MECHANISM_SLOTS)],
        *[_staged("mechanism", "attention") for _ in range(MAX_CONSECUTIVE_FAMILY)],
    ]
    report = evaluate(
        track="mechanism",
        family="attention",
        staged_candidates=history,
        cleared_families=[],
    )
    assert report["consecutive_family"] == MAX_CONSECUTIVE_FAMILY
    assert any("repair loop" in violation for violation in report["violations"])

    # A family that already cleared the gate is being confirmed, not repaired.
    cleared = evaluate(
        track="mechanism",
        family="attention",
        staged_candidates=history,
        cleared_families=["attention"],
    )
    assert cleared["violations"] == []


def test_enforce_raises_without_an_override_and_records_one_with():
    with pytest.raises(ExplorationBudgetError):
        enforce(track="knob", family="optimizer", staged_candidates=[], cleared_families=[])

    # A token override is refused: the reason is copied into an immutable spec, so
    # it has to say something.
    with pytest.raises(ExplorationBudgetError):
        enforce(
            track="knob",
            family="optimizer",
            staged_candidates=[],
            cleared_families=[],
            override_reason="why not",
        )

    reason = "operator directive: reproduce the published 2^18 batch rung exactly"
    report = enforce(
        track="knob",
        family="optimizer",
        staged_candidates=[],
        cleared_families=[],
        override_reason=reason,
    )
    assert report["override_reason"] == reason
    assert report["violations"]


def test_diagnostic_subsystems_are_exempt_so_the_two_gates_do_not_contradict():
    # `doctor` refuses model search while a control is overhead-dominated, and the
    # only way to fix that is input/compile/evaluation work.  If the breadth rule
    # also refused that work, a campaign whose first calibration comes back
    # overhead-dominated could stage nothing at all.
    assert is_diagnostic_subsystem("compile_mode")
    assert is_diagnostic_subsystem("Data-Loader")
    assert not is_diagnostic_subsystem("attention_span")

    blocked = evaluate(
        track="throughput",
        family="attention_span",
        staged_candidates=[],
        cleared_families=[],
        subsystem="attention_span",
    )
    assert blocked["violations"]

    exempt = evaluate(
        track="throughput",
        family="compile_cache",
        staged_candidates=[],
        cleared_families=[],
        subsystem="compile_mode",
    )
    assert exempt["diagnostic_subsystem"] is True
    assert exempt["violations"] == []
