#!/usr/bin/env python3
"""Every 10 minutes: check whether we are running, and restart the pool if not.

This exists because the written version of this rule has failed twice. A
predecessor campaign spent 12.8 of 21.6 hours idle and used about 5% of the
compute it had; the post-mortem's own conclusion was that a failure class stops
recurring only when it becomes structurally impossible, not when it becomes
discouraged. Earlier in this campaign the resident pool drained and exited while
analysis continued, and four H200s sat at 0% until a manual check happened to
catch it. So this is a process, not a paragraph.

Three questions are asked separately, because they fail separately:

  1. Is the pool alive?      It can exit on idle timeout, or crash.
  2. Is work queued?         The queue can be empty while the pool is healthy.
  3. Are the GPUs running?   Leases can be held with nothing executing on them.

Only the first is auto-repaired. Restarting a resident pool is safe -- the queue
is durable and each manifest maps to one idempotent job -- so the watchdog just
does it. The other two need a human or a research agent to decide *what* to run,
so the watchdog escalates loudly instead of inventing work: staging a candidate is
a scientific decision and a monitor has no business making one.

Writes a one-line-per-tick log and a machine-readable status file the heartbeat
reads, so "when was this last actually verified" is never a matter of memory.

    nohup uv run python tools/watchdog.py > runs/watchdog.out 2>&1 &
    uv run python tools/watchdog.py --once      # single check, for a heartbeat
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


STATE_ROOT = REPO / ".autoresearch"
LOG_PATH = REPO / "runs" / "watchdog.log"
STATUS_PATH = REPO / "runs" / "WATCHDOG_STATUS.json"

POOL_PATTERN = "autoresearch --root .autoresearch run --workers"
OUR_GPU_UUIDS = {
    "GPU-b2887def-dbd6-2033-0d4e-df854a8c4f06": "h200_gpu0",
    "GPU-3b9b0622-a181-60f7-00fc-02438c6fcb7a": "h200_gpu3",
    "GPU-d5063bf5-ac9a-da34-6a3c-f7e9b5a44ac4": "h200_gpu4",
    "GPU-f6cb0281-2829-ddcf-883a-df6eb0b18058": "h200_gpu7",
}
_load_dotenv()
SSH = [
    "ssh", "-i", os.environ.get("OPHIS_SSH_KEY", ""),
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
    "-p", os.environ.get("OPHIS_SSH_PORT", "22"),
    os.environ.get("OPHIS_SSH_TARGET", ""),
]


def _run(argv: list[str], timeout: int = 90) -> tuple[int, str]:
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, cwd=REPO, check=False
        )
        return done.returncode, done.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 124, f"{type(exc).__name__}: {exc}"


def pool_pids() -> list[int]:
    code, out = _run(["pgrep", "-f", POOL_PATTERN], timeout=15)
    if code != 0 or not out:
        return []
    return [int(line) for line in out.splitlines() if line.strip().isdigit()]


def start_pool(workers: int, idle_timeout: int, ignore_health: bool = False) -> int | None:
    """Relaunch the resident pool, detached so it outlives this watchdog.

    `ignore_health` exists for unattended multi-hour operation on this specific
    box. The dominant invalid here is environmental, not scientific: a foreign
    tenant grabs a GPU mid-launch and the co-tenancy guard correctly discards the
    run. Those results never reach the bank, so the circuit breaker is protecting
    against a config fault that is not occurring while costing every remaining GPU.
    It is NOT a licence to ignore a real fault -- read the EvidenceDecision reasons
    at the next heartbeat, and if the invalids are binding mismatches or timing
    reconciliations rather than co-tenancy, stop and fix the cause.
    """

    log = (REPO / "runs" / "worker_pool.log").open("ab")
    try:
        process = subprocess.Popen(
            [
                "uv", "run", "autoresearch", "--root", ".autoresearch", "run",
                "--workers", str(workers), "--follow", "--poll-seconds", "5",
                "--idle-timeout-seconds", str(idle_timeout),
                *(["--ignore-health"] if ignore_health else []),
            ],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError:
        return None
    return process.pid


def queue_state() -> dict[str, Any]:
    code, out = _run(
        ["uv", "run", "autoresearch", "--root", ".autoresearch", "queue"], timeout=120
    )
    if code != 0 or not out:
        return {"error": out[:200] or f"exit {code}"}
    try:
        document = json.loads(out)
    except ValueError:
        return {"error": "queue did not return JSON"}

    def find(node: Any) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if "states" in node:
                return node
            for value in node.values():
                found = find(value)
                if found:
                    return found
        return None

    block = find(document) or {}
    health = block.get("health", {})
    return {
        "states": block.get("states", {}),
        "depth": block.get("queue_depth"),
        "paused": bool(health.get("paused")),
        "health": health.get("state"),
    }


def gpu_state() -> dict[str, Any]:
    code, out = _run(
        [*SSH, "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory "
               "--format=csv,noheader; echo ---; uptime"],
        timeout=90,
    )
    if code != 0:
        return {"error": out[:200] or f"ssh exit {code}", "busy": [], "reachable": False}
    apps, _, tail = out.partition("---")
    busy = sorted(
        {
            OUR_GPU_UUIDS[uuid]
            for line in apps.splitlines()
            if (uuid := line.split(",")[0].strip()) in OUR_GPU_UUIDS
        }
    )
    load = ""
    if "load average:" in tail:
        load = tail.split("load average:")[1].strip().split(",")[0]
    return {"busy": busy, "idle": sorted(set(OUR_GPU_UUIDS.values()) - set(busy)),
            "load1": load, "reachable": True}


def refresh_chart() -> str:
    """Rebuild the scored view and redraw the chart, every tick.

    HEARTBEAT.md requires the chart after *every* landed experiment and never
    batched. Tying it to the watchdog tick rather than to an agent remembering is
    the same argument the rest of this file makes: a rule that depends on memory
    has already failed twice here. Both commands read the immutable store and
    write derived views, so running them on a quiet tick is a no-op.
    """

    code, _ = _run(
        ["uv", "run", "autoresearch", "--root", ".autoresearch", "bank"], timeout=180
    )
    if code != 0:
        return "chart=bank_failed"
    code, out = _run(["uv", "run", "python", "tools/make_chart.py"], timeout=180)
    if code != 0:
        return "chart=draw_failed"
    # make_chart prints "wrote <path>: N landed runs, M valid, ..."
    return "chart=ok" + (f" ({out.split(':', 1)[1].strip()})" if ":" in out else "")


def check(*, repair: bool, workers: int, idle_timeout: int, chart: bool = True,
          ignore_health: bool = False) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pids = pool_pids()
    action = ""
    if not pids and repair:
        # Safe unconditionally: the queue is durable and jobs are idempotent, so a
        # spurious restart costs nothing while a missed one costs every GPU.
        started = start_pool(workers, idle_timeout, ignore_health)
        action = f"RESTARTED pool pid={started}" if started else "RESTART FAILED"
        time.sleep(8)
        pids = pool_pids()
    elif not pids:
        action = "POOL DEAD (no --repair)"

    queue = queue_state()
    gpus = gpu_state()
    states = queue.get("states", {})
    running = int(states.get("running", 0) or 0)
    depth = queue.get("depth") or 0

    alerts = []
    if not pids:
        alerts.append("POOL_DEAD")
    if queue.get("paused"):
        alerts.append("HEALTH_CIRCUIT_PAUSED")
    if not gpus.get("reachable"):
        alerts.append("HOST_UNREACHABLE")
    elif gpus.get("idle") and running == 0 and depth == 0:
        # The expensive state: capacity available, nothing queued, nobody deciding.
        alerts.append(f"IDLE_GPUS_EMPTY_QUEUE({','.join(gpus['idle'])})")
    elif gpus.get("idle") and depth > 0:
        alerts.append(f"QUEUED_BUT_NOT_LAUNCHING({','.join(gpus['idle'])})")

    chart_note = refresh_chart() if chart else ""

    status = {
        "checked_at": now,
        "pool_pids": pids,
        "queue": queue,
        "gpus": gpus,
        "alerts": alerts,
        "action": action,
        "chart": chart_note,
        "ok": not alerts,
    }

    line = (
        f"{now} pool={'up' if pids else 'DOWN'} "
        f"running={running} depth={depth} "
        f"busy={len(gpus.get('busy', []))}/4 load1={gpus.get('load1', '?')} "
        f"{'OK' if not alerts else ' '.join(alerts)}"
        f"{'  ' + action if action else ''}"
        f"{'  ' + chart_note if chart_note else ''}"
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as handle:
        handle.write(line + "\n")
    STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")
    print(line, flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--idle-timeout-seconds", type=int, default=7200)
    parser.add_argument("--once", action="store_true", help="single check, then exit")
    parser.add_argument(
        "--no-repair", action="store_true", help="report only; never restart the pool"
    )
    parser.add_argument(
        "--no-chart", action="store_true", help="skip the per-tick bank + chart refresh"
    )
    parser.add_argument(
        "--ignore-health",
        action="store_true",
        help="restart the pool with --ignore-health (unattended runs on a contended box)",
    )
    args = parser.parse_args()

    if args.once:
        status = check(
            repair=not args.no_repair,
            workers=args.workers,
            idle_timeout=args.idle_timeout_seconds,
            chart=not args.no_chart,
            ignore_health=args.ignore_health,
        )
        sys.exit(0 if status["ok"] else 3)

    while True:
        try:
            check(
                repair=not args.no_repair,
                workers=args.workers,
                idle_timeout=args.idle_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - a watchdog must never die
            with LOG_PATH.open("a") as handle:
                handle.write(
                    f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                    f"WATCHDOG_ERROR {type(exc).__name__}: {exc}\n"
                )
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
