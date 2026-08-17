from __future__ import annotations

import json

from autoresearch.cli import main
from autoresearch.store import Store


def _files(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "metric.py"
    script.write_text(
        """import argparse, json, os
p = argparse.ArgumentParser()
p.add_argument('--value', type=float, required=True)
a = p.parse_args()
print('AUTORESEARCH_METRICS ' + json.dumps({
    'val_bpb': a.value,
    'num_steps': 1000,
    'seed': int(os.environ['AUTORESEARCH_SEED']),
    'training_seconds': 0.04,
    'total_seconds': 0.05,
}))
""",
        encoding="utf-8",
    )
    data = project / "data.json"
    data.write_text('{"split":"demo"}', encoding="utf-8")
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps(
            {
                "id": "cli_demo_v2",
                "hardware_class": "cpu-test",
                "dataset_split": "demo",
                "tokenizer": "none",
                "evaluator": "metric-emitter-v1",
                "precision": "fp64",
                "metric": {"name": "val_bpb", "direction": "minimize"},
                "budget": {"kind": "wall_seconds", "value": 0.05},
            }
        ),
        encoding="utf-8",
    )
    execution = tmp_path / "execution.json"
    execution.write_text(
        json.dumps(
            {
                "code_bindings": [{"source": str(script), "execution_path": script.name}],
                "data_bindings": [{"source": str(data), "execution_path": data.name}],
                "resources": [
                    {
                        "id": "cpu",
                        "hardware_class": "cpu-test",
                        "backend": "local",
                        "workdir": str(project),
                        "gpus": [],
                    }
                ],
                "runtime": {
                    "timeout_seconds_per_arm": 10,
                    "telemetry_interval_seconds": 0.01,
                    "resource_wait_seconds": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return script, scope, execution


def test_cli_calibrate_search_and_resident_run(tmp_path, capsys):
    script, scope, execution = _files(tmp_path)
    root = tmp_path / "state"
    prefix = ["--root", str(root)]

    assert main([*prefix, "init"]) == 0
    assert (
        main(
            [
                *prefix,
                "calibrate",
                "demo_bank",
                "initial",
                "--scope",
                str(scope),
                "--execution",
                str(execution),
                "--mutable-code-path",
                script.name,
                "--minimum-steps",
                "900",
                "--argv",
                "@python",
                script.name,
                "--value=1.0",
            ]
        )
        == 0
    )
    assert main([*prefix, "run", "--workers", "2"]) == 0
    assert main([*prefix, "doctor", "--bank-id", "demo_bank"]) == 0
    assert (
        main(
            [
                *prefix,
                "search",
                "better",
                "--bank-id",
                "demo_bank",
                "--summary",
                "lower deterministic value",
                "--scope",
                str(scope),
                "--execution",
                str(execution),
                "--mutable-code-path",
                script.name,
                "--minimum-steps",
                "900",
                "--subsystem",
                "optimizer",
                "--argv",
                "@python",
                script.name,
                "--value=0.9",
            ]
        )
        == 0
    )
    assert main([*prefix, "run", "--workers", "2"]) == 0
    assert main([*prefix, "bank"]) == 0

    output = capsys.readouterr().out
    assert '"promotion_due": 1' in output
    store = Store(root)
    assert (store.views_dir / "BANK.json").is_file()
    assert (store.views_dir / "PROMOTION_QUEUE.json").is_file()
