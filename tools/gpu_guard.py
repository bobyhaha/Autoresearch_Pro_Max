#!/usr/bin/env python3
"""Every 10 minutes, guarantee the GPUs are actually running our work.

`watchdog.py` restarts a dead pool but deliberately will not put work in the queue,
because staging a *candidate* is a scientific decision. `wake.py` tells an agent what
is wrong but needs an agent to be listening. Between them sits the failure that has
now happened three times in this project: the queue drains, every GPU goes to 0%, and
nothing at all happens until a human or an agent notices. Once it cost 12.8 of 21.6
hours; once four H200s idled through a pool crash; once this session left the fleet
idle for two hours because a loop was cancelled and never re-armed.

This closes that gap, and it closes it narrowly:

    IT STAGES CALIBRATION CONTROLS ONLY. NEVER A CANDIDATE.

That distinction is the whole safety argument. A candidate encodes a hypothesis and
choosing one is research; a control is a re-measurement of the frozen baseline that
is already sealed in the scope. Re-measuring costs nothing scientifically and pays
twice: it keeps the fleet warm, and every control replicate tightens the noise floor
that currently blocks reading any candidate at all (sigma 0.00556 against a 0.000426
gate). When in doubt this tool does LESS, and escalates instead.

    uv run python tools/gpu_guard.py --once          # one pass, for a heartbeat
    nohup uv run python tools/gpu_guard.py &          # resident, every 10 minutes
    uv run python tools/gpu_guard.py --once --dry-run # say what it would do

Exit 0 healthy or repaired, 3 escalation needed, 4 host unreachable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "runs" / "gpu_guard.log"
STATE = REPO / "runs" / "GPU_GUARD_STATE.json"

# Never stage more than this many waves without a landed result. If the fleet is
# burning controls that never land, something is broken that staging cannot fix and
# the honest move is to stop and say so rather than keep the GPUs warm for show.
MAX_CONSECUTIVE_STAGES = 4
POOL_PATTERN = "autoresearch --root .autoresearch run --workers"


def _load_dotenv() -> None:
    try:
        lines = (REPO / ".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()
SSH = [
    "ssh", "-i", os.environ.get("OPHIS_SSH_KEY", ""),
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    # Share one TCP connection across every tool and tick. Without this the
    # watchdog, wake, guard, preflight and eight worker threads each open their
    # own session; the host started refusing them, and the harness reported the
    # refusal as "code or data did not match the sealed manifest" on every
    # binding at once -- an environmental fault wearing a scientific fault's
    # clothes.
    "-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/cm/%r@%h:%p",
    "-o", "ControlPersist=600", "-o", "ConnectTimeout=20",
    "-p", os.environ.get("OPHIS_SSH_PORT", "22"), os.environ.get("OPHIS_SSH_TARGET", ""),
]
REMOTE_ROOT = os.environ.get("OPHIS_REMOTE_ROOT", "")


def run(argv: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              cwd=REPO, check=False)
        return done.returncode, (done.stdout or "").strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 124, f"{type(exc).__name__}: {exc}"


def log(line: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(f"{stamp} {line}\n")
    print(f"{stamp} {line}", flush=True)


def read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"consecutive_stages": 0, "last_result_count": -1}


def write_state(state: dict[str, Any]) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def result_count() -> int:
    directory = REPO / ".autoresearch" / "records" / "result_bundle"
    try:
        return len(list(directory.glob("*.json")))
    except OSError:
        return 0


def queue_state() -> dict[str, Any]:
    code, out = run(["uv", "run", "autoresearch", "--root", ".autoresearch", "status"])
    if code != 0 or not out:
        return {"error": out[:160] or f"exit {code}"}
    try:
        queue = json.loads(out).get("queue", {}) or {}
    except ValueError:
        return {"error": "status did not return JSON"}
    health = queue.get("health", {}) or {}
    states = queue.get("states", {}) or {}
    return {
        "running": int(states.get("running", 0) or 0),
        "pending": int(states.get("pending", 0) or 0),
        "depth": int(queue.get("queue_depth") or 0),
        "paused": bool(health.get("paused")),
    }


def our_training_processes() -> int | None:
    """How many of OUR training processes are alive on the box. None = unreachable."""
    if not REMOTE_ROOT:
        return None
    code, out = run([*SSH, f"pgrep -u $(whoami) -f '{REMOTE_ROOT}/gpu.*train.py' | wc -l"], 90)
    if code != 0:
        return None
    try:
        return int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def free_gpus() -> list[str]:
    """Our GPUs with no compute process on them. Empty list also means unreachable."""
    code, out = run([*SSH,
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"], 90)
    if code != 0:
        return []
    free = []
    for line in out.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 2 or cells[0] not in ("0", "3", "4", "7"):
            continue
        try:
            if float(cells[1].split()[0]) < 512:
                free.append(cells[0])
        except ValueError:
            continue
    return free


def ensure_pool(dry_run: bool) -> bool:
    code, _ = run(["pgrep", "-f", POOL_PATTERN], 20)
    if code == 0:
        return False
    if dry_run:
        log("WOULD restart pool (none alive)")
        return True
    handle = (REPO / "runs" / "worker_pool.log").open("ab")
    subprocess.Popen(
        ["uv", "run", "autoresearch", "--root", ".autoresearch", "run",
         "--workers", "8", "--follow", "--poll-seconds", "5",
         "--idle-timeout-seconds", "86400", "--ignore-health"],
        cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    log("RESTARTED pool")
    return True


def execution_for(free: list[str]) -> str:
    """Write an execution config containing only the GPUs that are actually free.

    Jobs are GPU-pinned, so staging a control for a card a foreign tenant holds
    creates a claim that can never run. Observed 2026-08-17: three `custom_molprop.py`
    processes from another user sat on gpu0 for over four hours, and every calibration
    wave added one more permanently stuck gpu0 claim -- seven of them, while gpu3/4/7
    ran fine. The queue read `running=7` the whole time, which looks healthy and is not.

    So the fleet is chosen from measurement, not from a fixed file. Four GPUs are ours
    on paper; the number usable right now is whatever nvidia-smi says.
    """
    base = json.loads((REPO / "runs" / "execution_fleet4.json").read_text())
    keep = [r for r in base.get("resources", []) if str(r.get("gpus", [None])[0]) in free]
    if not keep:
        return "runs/execution_fleet4.json"
    base["resources"] = keep
    for r in base["resources"]:
        r["max_concurrent_jobs"] = len(keep)
    out = REPO / "runs" / "execution_auto.json"
    out.write_text(json.dumps(base, indent=2) + "\n")
    return "runs/execution_auto.json"


def stage_controls(free: list[str], dry_run: bool) -> bool:
    """Stage one calibration control per declared GPU. Controls only, by design."""
    label = f"guard_{datetime.now(timezone.utc).strftime('%m%d_%H%M%S')}"
    argv = [
        "uv", "run", "autoresearch", "--root", ".autoresearch", "calibrate",
        "karpathy_228791f", label,
        "--scope", "runs/scope.json",
        "--execution", execution_for(free),
        "--mutable-code-path", "train.py",
        "--seed", "42",
        "--argv", "/bin/bash", "launch.sh",
    ]
    if dry_run:
        log(f"WOULD stage calibration wave {label} on free GPUs {free}")
        return True
    code, out = run(argv, 300)
    if code != 0:
        log(f"STAGE FAILED {label}: {out[:200]}")
        return False
    try:
        staged = json.loads(out)
        gpus = [s["gpu"] for s in staged.get("staged", staged.get("controls", []))]
    except (ValueError, KeyError):
        gpus = ["?"]
    log(f"STAGED calibration wave {label} on GPUs {gpus} (free={free})")
    return True


def check(dry_run: bool) -> int:
    state = read_state()
    results_now = result_count()
    # A landed result since the last pass means staging is working; reset the brake.
    if results_now != state.get("last_result_count"):
        state["consecutive_stages"] = 0
    state["last_result_count"] = results_now

    restarted = ensure_pool(dry_run)
    queue = queue_state()
    if "error" in queue:
        log(f"ESCALATE store unreadable: {queue['error']}")
        write_state(state)
        return 3

    procs = our_training_processes()
    if procs is None:
        log("host unreachable — retrying next tick, concluding nothing")
        write_state(state)
        return 4

    busy = queue["running"] + queue["pending"] + queue["depth"]
    free = free_gpus()
    # Work executing is not the same as work QUEUED. If the queue is empty, the fleet
    # goes idle the moment the current arms finish, and nothing notices until the next
    # tick -- ten minutes of four idle H200s, repeatedly, across a 24h run. Keep the
    # queue at least as deep as the capacity that will shortly be free.
    queued = queue["running"] + queue["pending"] + queue["depth"]
    capacity = max(len(free) + procs, 1)
    if procs > 0 and queued >= capacity:
        log(f"ok procs={procs} running={queue['running']} depth={queue['depth']}"
            f"{' (pool restarted)' if restarted else ''}")
        write_state(state)
        return 0
    if procs > 0 and queued < capacity:
        log(f"TOPPING UP: {procs} running but only {queued} job(s) queued against "
            f"capacity {capacity} (free={free}) — staging so the fleet does not drain")
        can_stage = (state["consecutive_stages"] < MAX_CONSECUTIVE_STAGES
                     and not queue["paused"])
        if can_stage and stage_controls(free, dry_run):
            state["consecutive_stages"] = state.get("consecutive_stages", 0) + 1
        write_state(state)
        return 0

    # No training processes. Queue depth alone is NOT sufficient reason to sit still:
    # jobs are pinned to a specific GPU, so a queue full of work for a tenant-held card
    # leaves free cards idle while `running` looks healthy. Observed 2026-08-17: six
    # jobs all pinned to gpu0 (foreign-held) while gpu3/4/7 sat free, and this branch
    # declined to stage for 15 minutes. Only decline if the queue has work AND we have
    # no free GPU to put it on.
    if busy > 0 and not free:
        log(f"claimed but not executing: running={queue['running']} depth={queue['depth']}, "
            f"no free GPU — resource-wait or setup, not staging")
        write_state(state)
        return 0
    if busy > 0 and free:
        log(f"IDLE CAPACITY: {len(free)} free GPU(s) {free} while {busy} job(s) are pinned "
            f"elsewhere — staging a wave to cover them")

    # Nothing running and nothing queued: the expensive state. Stage controls.
    if state["consecutive_stages"] >= MAX_CONSECUTIVE_STAGES:
        log(f"ESCALATE {state['consecutive_stages']} waves staged with no new result — "
            f"staging cannot fix this; a human or agent must look")
        write_state(state)
        return 3
    if queue["paused"]:
        log("health circuit paused — staging anyway is wrong; escalating")
        write_state(state)
        return 3

    log("IDLE GPUS, EMPTY QUEUE — staging calibration controls (never candidates)")
    if stage_controls(free, dry_run):
        state["consecutive_stages"] = state.get("consecutive_stages", 0) + 1
    write_state(state)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=600)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.once:
        sys.exit(check(args.dry_run))
    while True:
        try:
            check(args.dry_run)
        except Exception as exc:  # noqa: BLE001 - a guard must never die
            log(f"GUARD_ERROR {type(exc).__name__}: {exc}")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
