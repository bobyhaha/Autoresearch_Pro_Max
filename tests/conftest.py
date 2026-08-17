from __future__ import annotations

import json
import sys
from pathlib import Path

from autoresearch.records import REVIEW_ROLES, utc_now


def make_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    script = project / "metric.py"
    script.write_text(
        """from __future__ import annotations
import argparse, json, sys
p=argparse.ArgumentParser()
p.add_argument('--value', type=float, required=True)
p.add_argument('--steps', type=int, default=1000)
p.add_argument('--fail', action='store_true')
a=p.parse_args()
if a.fail:
    print('intentional failure', file=sys.stderr)
    raise SystemExit(7)
print('AUTORESEARCH_METRICS '+json.dumps({'score':a.value,'num_steps':a.steps}))
""",
        encoding="utf-8",
    )
    data = project / "data_manifest.json"
    data.write_text(json.dumps({"dataset": "test", "revision": 1}), encoding="utf-8")
    return script, data


def make_spec(
    script: Path,
    *,
    spec_id: str = "exp_candidate",
    stage: str = "confirmation",
    replicates: int = 2,
    candidate_values: list[float] | None = None,
    fail_control: bool = False,
    sota_eligible: bool = True,
) -> dict:
    candidate_values = candidate_values or [0.9 - 0.1 * index for index in range(replicates)]
    plan = []
    for index in range(replicates):
        control_argv = [sys.executable, str(script), "--value", "1.0", "--steps", "1000"]
        if fail_control:
            control_argv.append("--fail")
        plan.append(
            {
                "replicate_id": f"seed_{index + 1}",
                "arms": [
                    {"name": "control", "argv": control_argv, "env": {}},
                    {
                        "name": "candidate",
                        "argv": [
                            sys.executable,
                            str(script),
                            "--value",
                            str(candidate_values[index]),
                            "--steps",
                            "1000",
                        ],
                        "env": {},
                    },
                ],
            }
        )
    return {
        "id": spec_id,
        "stage": stage,
        "title": "Candidate mechanism test",
        "question": "Does the candidate improve the fixed-budget score?",
        "mechanism": {
            "cause": "the candidate changes the update",
            "effect": "endpoint score changes",
            "chain": ["change update", "change trajectory", "change endpoint"],
        },
        "hypothesis": {
            "statement": "The candidate reduces score.",
            "prediction": "candidate minus control is less than -0.05",
        },
        "falsifier": {"statement": "A non-negative difference refutes the mechanism."},
        "metric": {"name": "score", "direction": "minimize"},
        "plan": plan,
        "analysis": {
            "effect": "difference",
            "primary_arm": "candidate",
            "reference_arm": "control",
            "minimum_valid_replicates": replicates,
            "success_rule": {"op": "lt", "value": -0.05},
            "falsifier_rule": {"op": "gte", "value": 0.0},
            "sota_eligible": sota_eligible,
        },
        "requirements": {
            "required_metrics": ["score", "num_steps"],
            "minimum_steps": 900,
            "require_gpu": False,
            "isolation": "none",
        },
        "knowledge": {
            "source_ids": ["obs_prior_001"],
            "direction": "optimizer",
            "subsystem": "update_rule",
        },
        "comparison_group": "test_fixed_budget_v1",
    }


def execution_config(script: Path, data: Path) -> dict:
    return {
        "code_bindings": [{"source": str(script), "execution_path": script.name}],
        "data_bindings": [{"source": str(data), "execution_path": data.name}],
        "resources": [
            {
                "id": "local_cpu",
                "backend": "local",
                "workdir": str(script.parent),
                "gpus": [],
            }
        ],
        "runtime": {
            "timeout_seconds_per_arm": 10,
            "telemetry_interval_seconds": 0.02,
            "resource_wait_seconds": 3,
        },
    }


def approvals(spec: dict) -> dict:
    return {
        "spec_id": spec["id"],
        "spec_digest": spec["digest"],
        "reviews": [
            {
                "role": role,
                "reviewer_id": f"reviewer_{index}",
                "session_id": f"session_{index}",
                "decision": "approve",
                "reviewed_at": utc_now(),
                "notes": "approved for test",
            }
            for index, role in enumerate(sorted(REVIEW_ROLES), start=1)
        ],
    }
