"""Tiny executable used by the local smoke test and documentation."""

from __future__ import annotations

import argparse
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--value", type=float, required=True)
parser.add_argument("--steps", type=int, default=1000)
args = parser.parse_args()
seed = int(os.environ.get("AUTORESEARCH_SEED", "0"))
print("starting bounded example run")
time.sleep(0.9)
print(
    "AUTORESEARCH_METRICS "
    + json.dumps(
        {
            "score": args.value,
            "val_bpb": args.value,
            "num_steps": args.steps,
            "seed": seed,
            "training_seconds": 0.9,
            "total_seconds": 1.0,
        },
        sort_keys=True,
    )
)
