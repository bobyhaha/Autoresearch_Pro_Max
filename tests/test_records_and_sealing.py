from __future__ import annotations

import copy

import pytest
from conftest import approvals, execution_config, make_project, make_spec

from autoresearch.records import ConflictError, RecordError, make_record
from autoresearch.research import ResearchEngine
from autoresearch.sealing import SealingAuthority
from autoresearch.store import Store


def test_design_requires_the_full_scientific_chain(tmp_path):
    script, _ = make_project(tmp_path)
    proposal = make_spec(script)
    del proposal["mechanism"]
    with pytest.raises(RecordError, match="mechanism"):
        ResearchEngine(Store(tmp_path / "state")).create(proposal)


def test_design_requires_knowledge_provenance(tmp_path):
    script, _ = make_project(tmp_path)
    proposal = make_spec(script)
    proposal["knowledge"]["source_ids"] = []
    with pytest.raises(RecordError, match="source_ids"):
        ResearchEngine(Store(tmp_path / "state")).create(proposal)


def test_pilot_is_bounded_structurally(tmp_path):
    script, _ = make_project(tmp_path)
    proposal = make_spec(script, stage="pilot", replicates=2)
    with pytest.raises(RecordError, match="pilot"):
        ResearchEngine(Store(tmp_path / "state")).create(proposal)


def test_immutable_record_conflict_and_idempotent_design(tmp_path):
    script, _ = make_project(tmp_path)
    engine = ResearchEngine(Store(tmp_path / "state"))
    proposal = make_spec(script)
    first = engine.create(proposal)
    assert engine.create(proposal) == first
    changed = copy.deepcopy(proposal)
    changed["title"] = "Different title"
    with pytest.raises(ConflictError):
        engine.create(changed)


def test_confirmation_needs_five_independent_digest_bound_reviews(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    spec = ResearchEngine(store).create(make_spec(script))
    authority = SealingAuthority(store)
    with pytest.raises(RecordError, match="requires an independent review"):
        authority.seal(spec["id"], execution_config(script, data))
    review = approvals(spec)
    review["reviews"][1]["reviewer_id"] = review["reviews"][0]["reviewer_id"]
    with pytest.raises(RecordError, match="independent"):
        authority.seal(spec["id"], execution_config(script, data), review)

    stale = approvals(spec)
    stale["spec_digest"] = "0" * 64
    with pytest.raises(RecordError, match="stale"):
        authority.seal(spec["id"], execution_config(script, data), stale)


def test_scientific_review_survives_runtime_resealing(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    spec = ResearchEngine(store).create(make_spec(script))
    review = approvals(spec)
    authority = SealingAuthority(store)
    first = authority.seal(spec["id"], execution_config(script, data), review)
    changed_execution = execution_config(script, data)
    changed_execution["runtime"]["resource_wait_seconds"] = 7
    second = authority.seal(spec["id"], changed_execution, review)
    assert first["id"] != second["id"]
    assert first["payload"]["spec_digest"] == second["payload"]["spec_digest"]


def test_zero_execution_deadline_is_rejected(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    spec = ResearchEngine(store).create(
        make_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    )
    execution = execution_config(script, data)
    execution["runtime"]["timeout_seconds_per_arm"] = 0
    with pytest.raises(RecordError, match="runtime durations"):
        SealingAuthority(store).seal(spec["id"], execution)


def test_binding_execution_paths_must_be_globally_unique(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    spec = ResearchEngine(store).create(
        make_spec(script, stage="pilot", replicates=1, sota_eligible=False)
    )
    execution = execution_config(script, data)
    execution["data_bindings"][0]["execution_path"] = script.name
    with pytest.raises(RecordError, match="unique execution_path"):
        SealingAuthority(store).seal(spec["id"], execution)


def test_store_rejects_direct_immutable_replacement(tmp_path):
    store = Store(tmp_path / "state")
    script, _ = make_project(tmp_path)
    proposal = make_spec(script)
    record = make_record("experiment_spec", proposal.pop("id"), proposal)
    store.put(record)
    replacement_payload = copy.deepcopy(proposal)
    replacement_payload["title"] = "replacement"
    replacement = make_record("experiment_spec", record["id"], replacement_payload)
    with pytest.raises(ConflictError):
        store.put(replacement)


def test_arm_order_may_be_counterbalanced_across_replicates():
    """Arms run sequentially on one GPU, so fixed order makes the later arm
    systematically vulnerable to load arriving mid-replicate.  Counterbalancing
    is the standard remedy and the schema must permit it."""
    from autoresearch.records import RecordError, validate_experiment_spec

    def arms(order):
        return [{"name": name, "argv": [name], "env": {}} for name in order]

    spec = {
        "stage": "confirmation",
        "title": "t",
        "question": "q",
        "mechanism": {"cause": "c", "effect": "e", "chain": ["a"]},
        "hypothesis": {"statement": "s", "prediction": "p"},
        "falsifier": {"statement": "f"},
        "metric": {"name": "val_bpb", "direction": "minimize"},
        "plan": [
            {"replicate_id": "seed_42", "arms": arms(["control", "candidate"])},
            {"replicate_id": "seed_43", "arms": arms(["candidate", "control"])},
        ],
        "analysis": {
            "effect": "difference",
            "primary_arm": "candidate",
            "reference_arm": "control",
            "minimum_valid_replicates": 1,
            "success_rule": {"op": "lt", "value": -0.001},
            "falsifier_rule": {"op": "gte", "value": 0.0},
            "sota_eligible": False,
        },
        "requirements": {
            "required_metrics": ["val_bpb"],
            "minimum_steps": 1,
            "require_gpu": True,
            "isolation": "continuous",
        },
        "knowledge": {"source_ids": ["x"], "direction": "d", "subsystem": "s"},
        "comparison_group": "g",
    }
    validate_experiment_spec(spec)

    # A different arm SET is still rejected -- only the order is free.
    spec["plan"][1]["arms"] = arms(["candidate", "other_arm"])
    with pytest.raises(RecordError, match="same set of arm names"):
        validate_experiment_spec(spec)


def test_sealing_refuses_a_frame_other_than_300_seconds(tmp_path):
    """The comparison group IS the frame. 300 seconds is a hard requirement.

    A 600 s arm is not a better model, it is a model given twice the compute. One scored
    0.926958 against a 300 s SOTA of 0.962288 and was surfaced as the leaderboard's
    "running best" -- reading as a 3.7% breakthrough that was purely the extra budget.
    A report-time filter is not enough: the run still gets measured, recorded and believed.
    """
    from autoresearch.records import RecordError
    from autoresearch.sealing import SealingAuthority
    from autoresearch.store import Store

    store = Store(tmp_path / "state")
    authority = SealingAuthority(store)

    ok = {
        "comparison_group": "fixed_frame_val_bpb_v1",
        "plan": [
            {
                "replicate_id": "seed_42",
                "arms": [
                    {"name": "control", "argv": [".venv/bin/python", "train.py", "--arm=control"]},
                    {
                        "name": "candidate",
                        "argv": [".venv/bin/python", "train.py", "--time-budget=300"],
                    },
                ],
            }
        ],
    }
    authority._enforce_frame(ok)  # 300 is the frame; explicit is fine

    for bad_seconds in (150, 450, 600):
        bad = {
            "comparison_group": "fixed_frame_val_bpb_v1",
            "plan": [
                {
                    "replicate_id": "seed_42",
                    "arms": [
                        {"name": "control", "argv": [".venv/bin/python", "train.py"]},
                        {
                            "name": "candidate",
                            "argv": [
                                ".venv/bin/python",
                                "train.py",
                                f"--time-budget={bad_seconds}",
                            ],
                        },
                    ],
                }
            ],
        }
        with pytest.raises(RecordError) as excinfo:
            authority._enforce_frame(bad)
        assert str(bad_seconds) in str(excinfo.value)
        assert "300" in str(excinfo.value)
