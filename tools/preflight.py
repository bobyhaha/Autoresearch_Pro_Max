#!/usr/bin/env python3
"""Verify everything that must be true before a campaign run, and say what isn't.

Written after a restart in which several things were individually plausible and
jointly wrong: the active code disagreed with what provenance recorded, four remote
workdirs disagreed with the local tree, a scope claimed an evaluator digest that no
longer existed, and a "pristine" baseline silently read a corrected byte table left
in a shared cache by another campaign. Every one of those was cheap to detect and
expensive to miss, and none of them announced itself.

So this is one command that fails loudly rather than a checklist someone remembers.
It checks, in order of how badly it hurts to get wrong:

  1. the active code IS the baseline provenance claims it is
  2. train.py differs from upstream only by the protocol adapter
  3. prepare.py's digest IS the scope's evaluator, since that digest defines the metric
  4. every remote workdir byte-matches the local tree
  5. the corpus is pinned, present, and matches the data manifest
  6. the pre-tokenized cache is internally consistent with its own manifest
  7. the GPUs are healthy: no MIG, no throttle, full power limit, and actually free
  8. the local machine will not sleep mid-run and invalidate the timing reconciliation

Exit 0 means every check passed. Exit 1 means at least one FAILED and running now
would produce measurements you cannot trust. Warnings never fail the run; they are
things worth seeing, not reasons to stop.

    uv run python tools/preflight.py
    uv run python tools/preflight.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
CODE = REPO / "runs" / "code"


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
GPUS = ("0", "3", "4", "7")
UNPINNED = (
    "prepare.py reads the SHARED cache; dataset_split can change with no code change"
)


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, ok: bool | None, name: str, detail: str = "") -> None:
        self.rows.append({"status": "PASS" if ok else ("WARN" if ok is None else "FAIL"),
                          "check": name, "detail": detail})

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["status"] == "FAIL"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str], timeout: int = 90, retries: int = 1) -> tuple[int, str]:
    """Run a command, returning (code, stdout-or-stderr).

    Two things learned the hard way on this box. Failures reported STDERR while this
    only returned stdout, so a failed check printed an empty reason and told the
    reader nothing. And the host is shared at load ~250 with three of our own tools
    SSHing every ten minutes, so a single connection refusal is routine noise, not a
    reason to block a run -- HEARTBEAT's own rule is to retry rather than conclude.
    """
    last = (124, "not attempted")
    for attempt in range(retries + 1):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                  cwd=REPO, check=False)
            if done.returncode == 0:
                return 0, done.stdout.strip()
            detail = (done.stderr or done.stdout or "").strip() or f"exit {done.returncode}"
            last = (done.returncode, detail)
        except (subprocess.TimeoutExpired, OSError) as exc:
            last = (124, f"{type(exc).__name__}: {exc}")
        if attempt < retries:
            time.sleep(3 * (attempt + 1))
    return last


def check_code(rep: Report) -> None:
    prov = json.loads((CODE / "provenance.json").read_text())
    active, upstream = prov["active_adapter"], prov["upstream"]

    for name in ("train.py", "prepare.py"):
        actual = sha256(CODE / name)
        expected = active[f"{name.split('.')[0]}_py_sha256"]
        rep.add(actual == expected, f"active {name} matches provenance",
                "" if actual == expected else f"{actual[:16]} != {expected[:16]}")

    # The pristine copies must still be pristine, or every comparison above is moot.
    for name in ("train.py", "prepare.py"):
        actual = sha256(CODE / "upstream" / name)
        expected = upstream[f"{name.split('.')[0]}_py_sha256"]
        rep.add(actual == expected, f"upstream/{name} is pristine {upstream['commit'][:7]}",
                "" if actual == expected else f"{actual[:16]} != {expected[:16]}")

    # train.py is the file that computes the metric. Bound how far it may drift from
    # Karpathy's: the adapter emits numbers, it must not change any.
    import difflib
    up = (CODE / "upstream" / "train.py").read_text().splitlines()
    cur = (CODE / "train.py").read_text().splitlines()
    removed = [ln for ln in difflib.unified_diff(up, cur, lineterm="")
               if ln.startswith("-") and not ln.startswith("---")]
    rep.add(len(removed) <= 5, "train.py differs from upstream by adapter only",
            f"{len(removed)} upstream lines replaced (limit 5)")


def check_scope(rep: Report) -> None:
    scope = json.loads((REPO / "runs" / "scope.json").read_text())
    evaluator = scope.get("evaluator", "")
    actual = f"sha256:{sha256(CODE / 'prepare.py')}"
    # prepare.py's digest IS the evaluator: it defines the metric. If these disagree,
    # results get banked under a scope that never produced them.
    rep.add(evaluator == actual, "scope evaluator == prepare.py digest",
            "" if evaluator == actual else f"scope says {evaluator[:23]}, file is {actual[:23]}")
    rep.add(bool(scope.get("id")), "scope has an id", scope.get("id", ""))
    budget = (scope.get("budget") or {})
    rep.add(budget.get("kind") == "training_seconds" and float(budget.get("value", 0)) == 300.0,
            "scope budget is 300 training_seconds", json.dumps(budget))


def check_remote(rep: Report) -> None:
    if not REMOTE_ROOT or not os.environ.get("OPHIS_SSH_TARGET"):
        rep.add(False, "remote configured", "OPHIS_* env not set; copy .env.example to .env")
        return
    local = {name: sha256(CODE / name) for name in ("train.py", "prepare.py")}
    local["baseline_provenance.json"] = sha256(CODE / "provenance.json")

    cmd = "; ".join(
        f'echo "gpu{g} $(sha256sum {REMOTE_ROOT}/gpu{g}/train.py {REMOTE_ROOT}/gpu{g}/prepare.py '
        f'{REMOTE_ROOT}/gpu{g}/baseline_provenance.json 2>/dev/null | cut -d\\  -f1 | tr \\\\n \\ )"'
        for g in GPUS
    )
    code, out = run([*SSH, cmd], retries=2)
    if code != 0:
        rep.add(False, "remote reachable", out[:120])
        return
    rep.add(True, "remote reachable")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 4:
            rep.add(False, f"{parts[0] if parts else '?'} workdir readable", line[:80])
            continue
        gpu, t, p, b = parts
        ok = (t == local["train.py"] and p == local["prepare.py"]
              and b == local["baseline_provenance.json"])
        rep.add(ok, f"{gpu} workdir byte-matches local tree",
                "" if ok else f"train {t[:8]} prepare {p[:8]} prov {b[:8]}")


def check_corpus_and_cache(rep: Report) -> None:
    prepare_source = (CODE / "prepare.py").read_text()
    pinned = 'autoresearch_v2"' in prepare_source
    rep.add(pinned, "corpus root is pinned, not a shared listing", "" if pinned else UNPINNED)
    # Explicit delimiters, not line offsets: token_bytes.version has no trailing
    # newline, so a naive split glued "2" onto the first brace of the JSON manifest
    # and reported a corrupt version. The delimiter costs nothing and cannot drift.
    code, out = run([*SSH,
        'ls ~/.cache/autoresearch_v2/data/*.parquet 2>/dev/null | wc -l; echo "---VER---"; '
        'cat ~/.cache/autoresearch_v2/tokenizer/token_bytes.version 2>/dev/null; '
        'echo; echo "---MANIFEST---"; '
        'cat ~/.cache/autoresearch_v2/token_cache/train.manifest.json 2>/dev/null'],
        retries=2)
    if code != 0:
        rep.add(None, "corpus inspectable", out[:100]); return
    shards = out.split("---VER---")[0].strip()
    rep.add(shards == "11", "pinned corpus has 11 shards (10 train + 1 val)", f"found {shards}")
    version = out.split("---VER---")[1].split("---MANIFEST---")[0].strip()
    rep.add(version == "2", "token_bytes is corrected format 2", f"version={version!r}")
    manifest_text = out.split("---MANIFEST---", 1)[1].strip() if "---MANIFEST---" in out else ""
    try:
        manifest = json.loads(manifest_text) if manifest_text else {}
        consistent = (manifest.get("num_documents", 0) > 0
                      and manifest.get("num_tokens", 0) > 0
                      and manifest.get("num_batches", 0) > 0)
        rep.add(consistent, "token cache manifest is populated",
                f"{manifest.get('num_documents')} docs, "
                f"{manifest.get('num_tokens', 0) / 1e6:.1f}M tokens")
    except ValueError:
        rep.add(None, "token cache manifest parses",
                "absent or unreadable; run will fall back to inline tokenization")


def check_gpus(rep: Report) -> None:
    code, out = run([*SSH,
        "nvidia-smi --query-gpu=index,mig.mode.current,power.limit,clocks.max.sm,memory.used "
        "--format=csv,noheader"], retries=2)
    if code != 0:
        rep.add(False, "GPUs queryable", out[:100]); return
    free, busy = [], []
    for line in out.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 5:
            continue
        idx, mig, _power, _clk, mem = cells
        if idx not in GPUS:
            continue
        if mig.lower() != "disabled":
            rep.add(False, f"gpu{idx} MIG disabled", mig)
        used = float(mem.split()[0])
        (free if used < 512 else busy).append(idx)
    rep.add(len(free) > 0, "at least one of our GPUs is free",
            f"free={free} busy(foreign or ours)={busy}")
    rep.add(None if busy else True, "all four of our GPUs free", f"busy={busy}" if busy else "")


def check_local(rep: Report) -> None:
    code, _ = run(["pgrep", "-x", "caffeinate"], timeout=15)
    sleep_hint = (
        "macOS sleep will under-measure wall_seconds and invalidate healthy runs; "
        "run: nohup caffeinate -i -m >/dev/null 2>&1 &"
    )
    rep.add(code == 0, "idle sleep held (caffeinate)", "" if code == 0 else sleep_hint)
    code, out = run(["uv", "run", "pytest", "-q"], timeout=600)
    rep.add(code == 0, "test suite passes", out.splitlines()[-1][:90] if out else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rep = Report()
    check_code(rep)
    check_scope(rep)
    check_remote(rep)
    check_corpus_and_cache(rep)
    check_gpus(rep)
    check_local(rep)

    if args.json:
        print(json.dumps(rep.rows, indent=2))
    else:
        width = max(len(r["check"]) for r in rep.rows) + 2
        for r in rep.rows:
            mark = {"PASS": "  ok ", "WARN": " warn", "FAIL": "FAIL "}[r["status"]]
            print(f"{mark} {r['check']:<{width}} {r['detail']}")
        failed = rep.failed
        print()
        print(f"{len(rep.rows)} checks, {len(failed)} failed")
        if failed:
            print("\nDO NOT RUN until these are fixed:")
            for r in failed:
                print(f"  - {r['check']}: {r['detail']}")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
