#!/usr/bin/env python3
"""The 10-minute agent wake: are the GPUs running our work, and if not, why not?

`tools/watchdog.py` is the unattended half -- it restarts a dead pool and writes a
log. It deliberately cannot decide *what* to run, because staging a candidate is a
scientific decision and a monitor has no business making one. So the watchdog can
be perfectly healthy while every GPU sits at 0% with an empty queue, which is the
exact state that cost a predecessor campaign 12.8 of its 21.6 hours.

This file is the other half: the briefing a *research agent* reads on waking, on
the clock, whether or not it remembered to look. It answers the two questions
HEARTBEAT.md opens with -- am I running, are the GPUs running -- plus the one the
watchdog cannot: is there a decision waiting for me right now.

Exit codes are the point. A scheduler can act on them:

    0   GPUs are executing our work and nothing needs a decision
    3   a decision is waiting (idle GPUs, empty queue, paused circuit, dead pool)
    4   the host is unreachable -- retry, do not conclude anything

Arm it for a session with the harness scheduler, which is what actually delivers
the wake:

    /loop 10m uv run python tools/wake.py

or a cron-scheduled agent at */10 * * * * running the same command.

    uv run python tools/wake.py            # the briefing
    uv run python tools/wake.py --json     # same, machine-readable
    uv run python tools/wake.py --arm      # print the schedule command and exit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load repo-root .env so deployment facts stay out of this file and out of git."""
    dotenv = Path(__file__).resolve().parent.parent / ".env"
    try:
        lines = dotenv.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


WAKE_LOG = REPO / "runs" / "wake.log"

POOL_PATTERN = "autoresearch --root .autoresearch run --workers"
WATCHDOG_PATTERN = "tools/watchdog.py"

# Ours. GPUs 1, 2, 5, 6 belong to the plain_v2 campaign and are counted as
# occupied even when nvidia-smi shows them free -- see HEARTBEAT.md.
OUR_GPUS = {
    "GPU-b2887def-dbd6-2033-0d4e-df854a8c4f06": "h200_gpu0",
    "GPU-3b9b0622-a181-60f7-00fc-02438c6fcb7a": "h200_gpu3",
    "GPU-d5063bf5-ac9a-da34-6a3c-f7e9b5a44ac4": "h200_gpu4",
    "GPU-f6cb0281-2829-ddcf-883a-df6eb0b18058": "h200_gpu7",
}
_load_dotenv()
# Deployment facts come from .env (gitignored), never from this file: the repo
# describes the shape of the fleet, the operator supplies the box.
OPHIS_SSH_TARGET = os.environ.get("OPHIS_SSH_TARGET", "")
REMOTE_USER = OPHIS_SSH_TARGET.split("@", 1)[0] if "@" in OPHIS_SSH_TARGET else ""
SSH = [
    "ssh", "-i", os.environ.get("OPHIS_SSH_KEY", ""),
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
    "-p", os.environ.get("OPHIS_SSH_PORT", "22"), OPHIS_SSH_TARGET,
]

# One remote round trip: our training processes, per-GPU occupancy, and load.
PROBE = (
    f"ps -u {REMOTE_USER} -o pid=,etime=,cmd= | grep -E 'train.py|launch.sh' "
    "| grep -v grep; echo '---PROCS---'; "
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; "
    "echo '---MEM---'; "
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; "
    "echo '---LOAD---'; uptime"
)


def _run(argv: list[str], timeout: int = 90) -> tuple[int, str]:
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, cwd=REPO, check=False
        )
        return done.returncode, done.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 124, f"{type(exc).__name__}: {exc}"


def _pids(pattern: str) -> list[int]:
    code, out = _run(["pgrep", "-f", pattern], timeout=15)
    if code != 0 or not out:
        return []
    return [int(x) for x in out.splitlines() if x.strip().isdigit()]


def _section(text: str, name: str) -> str:
    """Pull one ---NAME--- delimited block out of the single-round-trip probe."""

    marker = f"---{name}---"
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].split("---", 1)[0].strip()


def sleep_guard() -> dict[str, Any]:
    """Is macOS idle-sleep held off, and did we nap recently?

    `wall_seconds` is measured here with `time.monotonic()`, which does not advance
    across system sleep, while `total_seconds` is measured inside train.py on the
    box. A nap therefore makes the runner under-measure a run that was perfectly
    healthy, and `evidence.py` correctly refuses to reconcile them. On 2026-08-16 a
    36s idle sleep invalidated two concurrent H200 runs at once. The laptop is the
    fault, so it is worth one cheap check per wake.
    """

    held = bool(_pids("caffeinate"))
    napped = ""
    code, out = _run(
        ["bash", "-lc", "pmset -g log | grep -E '\\bSleep\\b' | tail -3"], timeout=45
    )
    if code == 0 and out:
        napped = out.splitlines()[-1][:120]
    return {"idle_sleep_held": held, "last_sleep_log": napped}


def local_state() -> dict[str, Any]:
    code, out = _run(
        ["uv", "run", "autoresearch", "--root", ".autoresearch", "status"], timeout=120
    )
    state: dict[str, Any] = {
        "pool_pids": _pids(POOL_PATTERN),
        "watchdog_pids": _pids(WATCHDOG_PATTERN),
    }
    if code != 0 or not out:
        state["queue_error"] = out[:200] or f"exit {code}"
        return state
    try:
        document = json.loads(out)
    except ValueError:
        state["queue_error"] = "status did not return JSON"
        return state
    queue = document.get("queue", {}) or {}
    health = queue.get("health", {}) or {}
    state.update(
        running=int((queue.get("states", {}) or {}).get("running", 0) or 0),
        pending=int((queue.get("states", {}) or {}).get("pending", 0) or 0),
        depth=int(queue.get("queue_depth") or 0),
        paused=bool(health.get("paused")),
        health=health.get("state"),
        last_progress=queue.get("last_progress"),
    )
    return state


def remote_state() -> dict[str, Any]:
    code, out = _run([*SSH, PROBE], timeout=90)
    if code != 0:
        return {"reachable": False, "error": out[:200] or f"ssh exit {code}"}

    our_procs = [ln for ln in out.split("---PROCS---")[0].strip().splitlines() if ln.strip()]

    busy = sorted(
        {
            OUR_GPUS[uuid]
            for line in _section(out, "PROCS").splitlines()
            if (uuid := line.split(",")[0].strip()) in OUR_GPUS
        }
    )
    load_tail = _section(out, "LOAD")
    load1 = ""
    if "load average:" in load_tail:
        load1 = load_tail.split("load average:")[1].strip().split(",")[0]

    return {
        "reachable": True,
        "our_training_procs": len(our_procs),
        "proc_lines": [ln[:110] for ln in our_procs],
        "gpus_with_any_process": busy,
        "gpus_without_any_process": sorted(set(OUR_GPUS.values()) - set(busy)),
        "per_gpu": _section(out, "MEM").splitlines(),
        "load1": load1,
    }


def assess(local: dict[str, Any], remote: dict[str, Any]) -> tuple[list[str], int]:
    """Decide whether a research agent is needed, and how loudly."""

    decisions: list[str] = []

    if not remote.get("reachable"):
        # Never conclude anything from an unreachable host -- retry, per HEARTBEAT.
        return [f"HOST_UNREACHABLE: {remote.get('error', '')} -- retry, do not conclude"], 4

    if not local.get("pool_pids"):
        decisions.append("POOL_DEAD: no resident worker pool -- restart it NOW")
    if not local.get("watchdog_pids"):
        decisions.append("WATCHDOG_DEAD: the unattended repair loop is not running")
    if not local.get("sleep", {}).get("idle_sleep_held"):
        decisions.append(
            "IDLE_SLEEP_NOT_HELD: no caffeinate assertion -- a nap will invalidate "
            "every run in flight via the total_seconds/wall_seconds check. Run: "
            "nohup caffeinate -i -m >/dev/null 2>&1 &"
        )
    if local.get("paused"):
        decisions.append(
            "HEALTH_CIRCUIT_PAUSED: read doctor + the last EvidenceDecision reasons "
            "BEFORE reaching for run --ignore-health"
        )
    if local.get("queue_error"):
        decisions.append(f"STORE_UNREADABLE: {local['queue_error']}")

    # The expensive state, and the one the watchdog is not allowed to fix:
    # capacity is available, nothing is queued, and nobody is deciding what to run.
    idle = remote.get("gpus_without_any_process", [])
    running = local.get("running", 0)
    depth = local.get("depth", 0)
    if remote.get("our_training_procs", 0) == 0 and running == 0 and depth == 0:
        decisions.append(
            f"IDLE_GPUS_EMPTY_QUEUE({','.join(idle)}): STAGE WORK -- this is the "
            "single most expensive failure available to this campaign"
        )
    elif remote.get("our_training_procs", 0) == 0 and (running or depth):
        decisions.append(
            f"CLAIMED_BUT_NOT_EXECUTING: {running} running / {depth} queued but zero "
            "of our training processes on the box -- jobs are in resource-wait "
            "(a foreign tenant holds the GPU) or the arm is dying in setup"
        )
    elif idle and depth > 0:
        decisions.append(f"QUEUED_BUT_NOT_LAUNCHING({','.join(idle)})")

    return decisions, 3 if decisions else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable briefing")
    parser.add_argument(
        "--arm", action="store_true", help="print the schedule command and exit"
    )
    args = parser.parse_args()

    if args.arm:
        print("Arm the 10-minute agent wake with the harness scheduler:\n")
        print("    /loop 10m uv run python tools/wake.py\n")
        print("and the unattended repair loop separately:\n")
        print("    pgrep -f 'tools/watchdog.py' || nohup uv run python tools/watchdog.py \\")
        print("      --interval-seconds 600 --workers 4 --idle-timeout-seconds 14400 \\")
        print("      > runs/watchdog.out 2>&1 &")
        sys.exit(0)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    local = local_state()
    local["sleep"] = sleep_guard()
    remote = remote_state()
    decisions, code = assess(local, remote)

    briefing = {
        "checked_at": now,
        "verdict": "OK" if code == 0 else ("UNREACHABLE" if code == 4 else "NEEDS_DECISION"),
        "decisions": decisions,
        "local": local,
        "remote": remote,
    }

    WAKE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WAKE_LOG.open("a") as handle:
        handle.write(
            f"{now} verdict={briefing['verdict']} "
            f"pool={'up' if local.get('pool_pids') else 'DOWN'} "
            f"running={local.get('running', '?')} depth={local.get('depth', '?')} "
            f"our_procs={remote.get('our_training_procs', '?')} "
            f"busy={len(remote.get('gpus_with_any_process', []))}/4 "
            f"load1={remote.get('load1', '?')} "
            f"{';'.join(decisions) if decisions else 'OK'}\n"
        )

    if args.json:
        print(json.dumps(briefing, indent=2))
        sys.exit(code)

    print(f"=== WAKE {now} -- {briefing['verdict']} ===")
    print(
        f"pool={'up' if local.get('pool_pids') else 'DOWN'}  "
        f"watchdog={'up' if local.get('watchdog_pids') else 'DOWN'}  "
        f"queue running={local.get('running', '?')} pending={local.get('pending', '?')} "
        f"depth={local.get('depth', '?')}  health={local.get('health')}  "
        f"idle-sleep={'held' if local.get('sleep', {}).get('idle_sleep_held') else 'NOT HELD'}"
    )
    if remote.get("reachable"):
        print(
            f"our training processes on the box: {remote['our_training_procs']}   "
            f"load1={remote['load1']}"
        )
        for line in remote.get("per_gpu", []):
            print(f"  gpu {line}")
        for line in remote.get("proc_lines", []):
            print(f"  proc {line}")
    else:
        print(f"host unreachable: {remote.get('error')}")

    if decisions:
        print("\nDECISIONS WAITING FOR YOU:")
        for item in decisions:
            print(f"  - {item}")
        print("\nFix these BEFORE analysis, writing, or staging the next candidate.")
    else:
        print("\nGPUs are executing our work. Nothing needs a decision.")

    sys.exit(code)


if __name__ == "__main__":
    main()
