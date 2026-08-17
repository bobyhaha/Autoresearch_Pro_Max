#!/usr/bin/env python3
"""Regenerate the val_bpb progress chart from the immutable v2 store.

Same shape as the upstream `autoresearch` progress chart -- self-contained HTML,
one SVG scatter of val_bpb against experiment number, a running-best frontier, a
paired-control marker joined to each candidate, stat tiles above and a full table
below -- but every number is read out of `.autoresearch/records/`, never from a
hand-maintained TSV. If a run did not land a ResultBundle it does not appear, and
if its EvidenceDecision is not `valid` it is drawn as a failure rather than as a
score.

Two things differ from upstream because v2's record model is stricter:

  * The paired control is not "the champion re-run beside it". It is the *exact
    bank control frozen into the candidate's spec before launch*, on the same
    physical GPU, and the chart reads it back out of that immutable reference
    rather than pairing by adjacency.
  * Status is derived, never declared. `promotion-due` means the bank scored the
    candidate past the gate; it is a work-allocation decision and is labelled as
    one, because in v2 nothing on this chart can be SOTA.

Usage:  uv run python tools/make_chart.py [--root .autoresearch] [--output ...]
"""

from __future__ import annotations

import argparse
import html
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

W, H = 980, 430
PAD_L, PAD_R, PAD_T, PAD_B = 68, 26, 22, 46


def _load_records(root: Path, kind: str) -> list[dict[str, Any]]:
    directory = root / "records" / kind
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return rows


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("payload")
    return value if isinstance(value, dict) else record


def collect(root: Path) -> list[dict[str, Any]]:
    """Build one chart row per landed ResultBundle, oldest first."""

    specs = {str(r["id"]): _payload(r) for r in _load_records(root, "experiment_spec")}

    # Latest decision per result. Older evidence-v2 rows stay in the store but the
    # current policy authority is whatever judged the bundle most recently.
    decisions: dict[str, dict[str, Any]] = {}
    for record in _load_records(root, "evidence_decision"):
        payload = _payload(record)
        result_id = str(payload.get("result_id", ""))
        if not result_id:
            continue
        previous = decisions.get(result_id)
        if previous is None or str(record.get("created_at", "")) >= previous["_at"]:
            decisions[result_id] = {**payload, "_at": str(record.get("created_at", ""))}

    # Bank scores, keyed by result, give the delta against the frozen control.
    scores: dict[str, dict[str, Any]] = {}
    promotion_view = root / "views" / "PROMOTION_QUEUE.json"
    if promotion_view.is_file():
        try:
            view = json.loads(promotion_view.read_text())
            for row in view.get("candidates", []):
                if isinstance(row, dict) and row.get("result_id"):
                    scores[str(row["result_id"])] = row
        except (OSError, ValueError):
            pass

    rows: list[dict[str, Any]] = []
    for record in _load_records(root, "result_bundle"):
        payload = _payload(record)
        spec_id = str(payload.get("spec_id", ""))
        spec = specs.get(spec_id, {})
        row_scope = spec.get("scope") if isinstance(spec.get("scope"), dict) else {}
        search = spec.get("search") if isinstance(spec.get("search"), dict) else {}
        knowledge = spec.get("knowledge") if isinstance(spec.get("knowledge"), dict) else {}

        arms = [a for a in payload.get("arms", []) if isinstance(a, dict)]
        metrics: dict[str, Any] = {}
        failure = ""
        for arm in arms:
            arm_metrics = arm.get("metrics") or {}
            if arm_metrics:
                metrics = arm_metrics
            if arm.get("failure"):
                failure = str(arm["failure"])

        decision = decisions.get(str(record.get("id", "")), {})
        verdict = str(decision.get("measurement_verdict", "unknown"))
        score = scores.get(str(record.get("id", "")), {})

        value = metrics.get("val_bpb")
        value = float(value) if isinstance(value, (int, float)) else 0.0

        lane = str(search.get("lane", "")) or "unknown"
        if verdict != "valid" or value <= 0:
            status = "failed"
        elif lane == "bank":
            status = "control"
        elif score.get("promotion_due"):
            status = "promotion-due"
        else:
            status = "measured"

        total = metrics.get("total_seconds")
        train = metrics.get("training_seconds")
        overhead = ""
        if isinstance(total, (int, float)) and isinstance(train, (int, float)) and total > 0:
            overhead = f"{100 * (total - train) / total:.0f}%"

        rows.append(
            {
                "result_id": str(record.get("id", "")),
                "created_at": str(record.get("created_at", "")),
                "spec_id": spec_id,
                "scope_id": str(row_scope.get("id", "")),
                "title": str(spec.get("title", spec_id)),
                "lane": lane,
                "track": str(search.get("track", "")) or ("control" if lane == "bank" else ""),
                "family": str(search.get("family", "")) or str(knowledge.get("subsystem", "")),
                "slot": _slot_of(payload),
                "seed": metrics.get("seed", ""),
                "val": value,
                "steps": float(metrics.get("num_steps") or 0),
                "mfu": metrics.get("mfu_percent"),
                "tokens": metrics.get("total_tokens_M"),
                "training_s": train,
                "total_s": total,
                "overhead": overhead,
                "verdict": verdict,
                "status": status,
                "failure": failure,
                "ctrl_bpb": float(score.get("control_value") or 0.0),
                "delta": score.get("delta"),
            }
        )

    rows.sort(key=lambda row: (row["created_at"], row["result_id"]))
    for index, row in enumerate(rows, start=1):
        row["i"] = index
    return rows


def _slot_of(payload: dict[str, Any]) -> str:
    """The resource id the arm actually ran on, from the bundle's own resource block.

    Deliberately not the GPU *index*: v2 keys bank identity by the physical GPU
    UUID because an index is remappable, and the chart should show the same
    identity the bank compared against.
    """

    resource = payload.get("resource")
    if isinstance(resource, dict):
        name = str(resource.get("id", "") or "")
        if name:
            return name
    return ""


def _ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    raw = (hi - lo) / count
    if raw <= 0:
        return [lo]
    magnitude = 10 ** math.floor(math.log10(raw))
    step = magnitude * 10
    for multiple in (1, 2, 2.5, 5, 10):
        if magnitude * multiple >= raw:
            step = magnitude * multiple
            break
    out, tick = [], math.ceil(lo / step) * step
    while tick <= hi + 1e-12:
        out.append(tick)
        tick += step
    return out


def render(rows: list[dict[str, Any]], scope_id: str, excluded: int = 0) -> str:
    measured = [r for r in rows if r["val"] > 0 and r["status"] != "failed"]
    failed = [r for r in rows if r["status"] == "failed"]
    controls = [r for r in measured if r["status"] == "control"]

    best = math.inf
    for row in rows:
        if row["val"] > 0 and row["status"] != "failed" and row["val"] < best:
            best = row["val"]
        row["best"] = None if best is math.inf else best

    n = max(len(rows), 1)
    baseline = controls[0]["val"] if controls else (measured[0]["val"] if measured else 0.0)
    current_best = min((r["val"] for r in measured), default=0.0)
    best_row = min(measured, key=lambda r: r["val"], default=None)

    # Control spread is the noise floor every delta on this chart is read against.
    control_sd = 0.0
    if len(controls) > 1:
        values = [r["val"] for r in controls]
        mean = sum(values) / len(values)
        control_sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    pool = [r["val"] for r in measured] + [r["ctrl_bpb"] for r in rows if r["ctrl_bpb"] > 0]
    if pool:
        vmin, vmax = min(pool), max(pool)
        span = max(vmax - vmin, 1e-4)
        vmin -= span * 0.16
        vmax += span * 0.14
    else:
        vmin, vmax = 0.9, 1.2

    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def sx(i: int) -> float:
        return PAD_L + plot_w / 2 if n == 1 else PAD_L + (i - 1) / (n - 1) * plot_w

    def sy(v: float) -> float:
        return PAD_T + (vmax - v) / (vmax - vmin) * plot_h

    grid = "".join(
        f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
        f'<text class="tick" x="{PAD_L - 10}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:.3f}</text>'
        for t in _ticks(vmin, vmax)
    )
    xstep = max(1, n // 14)
    xticks = "".join(
        f'<text class="tick" x="{sx(r["i"]):.1f}" y="{H - PAD_B + 20}" text-anchor="middle">{r["i"]}</text>'
        for r in rows
        if r["i"] % xstep == 0 or r["i"] == 1
    )

    base_rule = ""
    if baseline:
        base_rule = (
            f'<line class="baserule" x1="{PAD_L}" x2="{W - PAD_R}" '
            f'y1="{sy(baseline):.1f}" y2="{sy(baseline):.1f}"/>'
            f'<text class="baselab" x="{W - PAD_R}" y="{sy(baseline) - 7:.1f}" '
            f'text-anchor="end">first control {baseline:.4f}</text>'
        )

    # The +/-2sd band around the control mean: anything inside it is an unordered tie.
    noise_band = ""
    if control_sd > 0:
        mean = sum(r["val"] for r in controls) / len(controls)
        top, bottom = sy(mean + 2 * control_sd), sy(mean - 2 * control_sd)
        noise_band = (
            f'<rect class="noise" x="{PAD_L}" y="{min(top, bottom):.1f}" '
            f'width="{plot_w}" height="{abs(bottom - top):.1f}"/>'
        )

    frontier = ""
    points = [r for r in rows if r["best"] is not None]
    if len(points) > 1:
        d = [f'M {sx(points[0]["i"]):.1f} {sy(points[0]["best"]):.1f}']
        for previous, row in pairwise(points):
            d.append(f'L {sx(row["i"]):.1f} {sy(previous["best"]):.1f}')
            d.append(f'L {sx(row["i"]):.1f} {sy(row["best"]):.1f}')
        frontier = f'<path class="frontier" d="{" ".join(d)}"/>'

    def data_attrs(row: dict[str, Any], kind: str, value: float) -> str:
        return (
            f'data-i="{row["i"]}" data-val="{value:.6f}" data-kind="{kind}" '
            f'data-status="{html.escape(row["status"])}" '
            f'data-title="{html.escape(row["title"], quote=True)}" '
            f'data-lane="{html.escape(row["lane"])}" '
            f'data-track="{html.escape(row["track"] or "-")}" '
            f'data-slot="{html.escape(row["slot"])}" '
            f'data-seed="{html.escape(str(row["seed"]))}" '
            f'data-steps="{row["steps"]:.0f}" '
            f'data-mfu="{html.escape(str(row["mfu"] or ""))}" '
            f'data-overhead="{html.escape(row["overhead"])}" '
            f'data-verdict="{html.escape(row["verdict"])}" '
            f'data-failure="{html.escape(row["failure"], quote=True)}"'
        )

    marks = []
    for row in rows:
        if row["ctrl_bpb"] > 0 and row["val"] > 0:
            x = sx(row["i"])
            marks.append(
                f'<line class="pairlink" x1="{x:.1f}" y1="{sy(row["val"]):.1f}" '
                f'x2="{x:.1f}" y2="{sy(row["ctrl_bpb"]):.1f}"/>'
                f'<circle class="dot ctrl" cx="{x:.1f}" cy="{sy(row["ctrl_bpb"]):.1f}" r="4" '
                f'{data_attrs(row, "frozen-control", row["ctrl_bpb"])}/>'
            )
    for row in measured:
        css = {"promotion-due": "m-due", "control": "m-control"}.get(row["status"], "m-measured")
        radius = 6 if row["status"] == "promotion-due" else 4.5
        shape = "rect" if row["status"] == "control" else "circle"
        x, y = sx(row["i"]), sy(row["val"])
        if shape == "rect":
            marks.append(
                f'<rect class="dot {css}" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
                f'{data_attrs(row, "control", row["val"])}/>'
            )
        else:
            marks.append(
                f'<circle class="dot {css}" cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
                f'{data_attrs(row, "candidate", row["val"])}/>'
            )
    for row in failed:
        x, y = sx(row["i"]), H - PAD_B - 8
        marks.append(
            f'<path class="dot dot-failed" d="M {x - 5:.1f} {y - 5:.1f} l 10 10 '
            f'M {x + 5:.1f} {y - 5:.1f} l -10 10" {data_attrs(row, "failed", 0.0)}/>'
        )

    best_label = ""
    if best_row:
        bx, by = sx(best_row["i"]), sy(best_row["val"])
        anchor = "end" if bx > W - 150 else "start"
        best_label = (
            f'<text class="best-label" x="{bx + (-12 if anchor == "end" else 12):.1f}" '
            f'y="{by + 16:.1f}" text-anchor="{anchor}">best {best_row["val"]:.4f}</text>'
        )

    def pair_cell(row: dict[str, Any]) -> str:
        if row["ctrl_bpb"] > 0 and isinstance(row["delta"], (int, float)):
            delta = float(row["delta"])
            sign = "−" if delta < 0 else "+"
            css = "win" if delta < 0 else "loss"
            return f'{row["ctrl_bpb"]:.6f} <span class="{css}">({sign}{abs(delta):.4f})</span>' 
        return "—"

    def failure_note(row: dict[str, Any]) -> str:
        if not row["failure"]:
            return ""
        return f'<br><span class="m">{html.escape(row["failure"])}</span>'

    # The baseline is ONE row, not fifteen. Listing every control individually made a
    # table where the first fifteen rows were the same experiment repeated, all
    # numbered 1 -- which is less readable than the numbering it replaced. What a
    # reader wants from the baseline is its centre, its spread and how many runs back
    # it; the individual controls are in the chart at x=1 and in the immutable store.
    control_rows = [r for r in rows if r["status"] == "control"]
    candidate_rows = [r for r in rows if r["status"] != "control"]

    baseline_row = ""
    if control_rows:
        vals = sorted(r["val"] for r in control_rows)
        steps = sorted(r["steps"] for r in control_rows)
        mean = sum(vals) / len(vals)
        sd = (
            (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            if len(vals) > 1
            else 0.0
        )
        slots = sorted({r["slot"] for r in control_rows if r["slot"]})
        baseline_row = (
            '<tr><td>1</td>'
            '<td class="mono">bank</td><td class="mono">baseline</td>'
            f'<td class="mono">{html.escape(", ".join(s.replace("h200_", "") for s in slots))}</td>'
            f'<td class="num strong">{mean:.6f}</td>'
            f'<td class="num">{steps[0]:.0f}–{steps[-1]:.0f}</td>'
            f'<td class="num">—</td><td class="num">—</td>'
            '<td><span class="pill pill-control">baseline</span></td>'
            f'<td>the frozen baseline, re-measured {len(vals)} times · '
            f'σ {sd:.6f} · range {vals[0]:.6f}–{vals[-1]:.6f}<br>'
            '<span class="m">spread is contention, not drift: step count explains it '
            '(see the chart at experiment 1)</span></td></tr>'
        )

    table_rows = "".join(
        f'<tr><td>{r["i"]}</td>'
        f'<td class="mono">{html.escape(r["lane"])}</td>'
        f'<td class="mono">{html.escape(r["track"] or "—")}</td>'
        f'<td class="mono">{html.escape(r["slot"])}</td>'
        f'<td class="num strong">{format(r["val"], ".6f") if r["val"] > 0 else "—"}</td>'
        f'<td class="num">{r["steps"]:.0f}</td>'
        f'<td class="num">{html.escape(str(r["overhead"]))}</td>'
        f'<td class="num">{pair_cell(r)}</td>'
        f'<td><span class="pill pill-{r["status"]}">{r["status"]}</span></td>'
        f'<td>{html.escape(r["title"])}{failure_note(r)}</td></tr>'
        for r in reversed(candidate_rows)
    ) + baseline_row

    n_due = sum(1 for r in rows if r["status"] == "promotion-due")
    sd_text = f"σ {control_sd:.5f} over {len(controls)}" if control_sd else f"{len(controls)} control(s)"

    # The scope suffix belongs in the title, not just the subtitle: each scope is a
    # separate leaderboard with its own artifact, and a title that does not name the
    # scope is the easiest way to overwrite one campaign's chart with another's.
    scope_tag = scope_id.rsplit("_", 1)[-1] if "_" in scope_id else scope_id
    excluded_note = (
        f", and {excluded} landed run{'s' if excluded != 1 else ''} whose decision was not "
        "<code>valid</code> {verb} excluded here — they remain in the immutable store, so "
        "nothing is deleted, only kept off a scale it does not belong on.".replace(
            "{verb}", "were" if excluded != 1 else "was"
        )
        if excluded
        else ". No landed run has been excluded."
    )
    return f"""<title>OPHIS {html.escape(scope_tag)} — val_bpb progress</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1:#fcfcfb; --plane:#f9f9f7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
    --series-1:#2a78d6; --good:#0ca30c; --critical:#b3261e; --success-text:#006300;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface-1:#1a1a19; --plane:#0d0d0d;
      --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
      --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
      --series-1:#3987e5; --good:#2aa82a; --critical:#e35c7a; --success-text:#2aa82a;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --series-1:#3987e5; --good:#2aa82a; --critical:#e35c7a; --success-text:#2aa82a;
  }}
  body {{
    background: var(--plane); color: var(--text-primary); margin: 0;
  }}
  .viz-root {{
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--plane); color: var(--text-primary);
    padding: 28px 20px 48px; margin: 0 auto; max-width: 1080px;
  }}
  a:focus-visible, .dot:focus-visible {{ outline: 2px solid var(--series-1); outline-offset: 2px; }}
  @media (prefers-reduced-motion: reduce) {{ .tt {{ transition: none; }} }}
  h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; max-width: 78ch; line-height: 1.55; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  .tiles {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }}
  .tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
           padding: 14px 18px; min-width: 140px; flex: 1 1 140px; }}
  .tile .k {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
  .tile .v {{ font-size: 26px; font-weight: 600; margin-top: 4px; line-height: 1.1;
              font-variant-numeric: tabular-nums; }}
  .tile .d {{ font-size: 12px; color: var(--text-secondary); margin-top: 3px; }}
  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
           padding: 16px 8px 8px; }}
  .wrap {{ overflow-x: auto; }}
  svg {{ display: block; min-width: {W}px; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .noise {{ fill: var(--muted); opacity: 0.13; }}
  .baserule {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 5 4; }}
  .baselab {{ fill: var(--muted); font-size: 11px; }}
  .tick {{ fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
  .alab {{ fill: var(--muted); font-size: 11px; }}
  .frontier {{ fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; }}
  .pairlink {{ stroke: var(--muted); stroke-width: 1.5; opacity: 0.5; }}
  .dot {{ cursor: pointer; }}
  circle.dot, rect.dot {{ stroke: var(--surface-1); stroke-width: 2; }}
  circle.ctrl {{ fill: none; stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 2 2; }}
  .m-due {{ fill: var(--good); }}
  .m-measured {{ fill: var(--series-1); }}
  rect.m-control {{ fill: var(--muted); }}
  path.dot-failed {{ stroke: var(--critical); stroke-width: 2.5; fill: none; stroke-linecap: round; }}
  .best-label {{ fill: var(--text-secondary); font-size: 12px; font-weight: 600; }}
  .legend {{ display: flex; gap: 18px; flex-wrap: wrap; padding: 12px 12px 4px;
             font-size: 12px; color: var(--text-secondary); }}
  .legend span.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                     margin-right: 6px; vertical-align: -1px; }}
  .legend span.sq {{ display: inline-block; width: 9px; height: 9px; margin-right: 6px;
                     vertical-align: -1px; background: var(--muted); }}
  .legend span.ring {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%;
                       border: 1.5px dashed var(--muted); margin-right: 6px; vertical-align: -1px; }}
  .legend span.line {{ display: inline-block; width: 16px; height: 2px; margin-right: 6px;
                       vertical-align: 4px; background: var(--series-1); }}
  .legend span.dash {{ display: inline-block; width: 16px; height: 0; margin-right: 6px;
                       vertical-align: 4px; border-top: 1px dashed var(--muted); }}
  .legend span.x {{ display: inline-block; width: 10px; margin-right: 6px;
                    color: var(--critical); font-weight: 700; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 26px; }}
  th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.05em; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.strong {{ font-weight: 600; }}
  td.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--text-secondary); }}
  .m {{ color: var(--muted); font-size: 11.5px; }}
  .win {{ color: var(--success-text); }}
  .loss {{ color: var(--muted); }}
  .pill {{ font-size: 11px; padding: 2px 8px; border-radius: 99px; font-weight: 600; white-space: nowrap; }}
  .pill-promotion-due {{ color: var(--success-text); background: color-mix(in srgb, var(--good) 14%, transparent); }}
  .pill-measured {{ color: var(--text-secondary); background: color-mix(in srgb, var(--series-1) 14%, transparent); }}
  .pill-control {{ color: var(--text-secondary); background: color-mix(in srgb, var(--muted) 16%, transparent); }}
  .pill-failed {{ color: var(--critical); background: color-mix(in srgb, var(--critical) 14%, transparent); }}
  .tablewrap {{ overflow-x: auto; }}
  .tt {{ position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
         background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border);
         border-radius: 8px; padding: 9px 11px; font-size: 12px; max-width: 340px;
         box-shadow: 0 6px 24px rgba(0,0,0,0.18); z-index: 10; line-height: 1.5; }}
  .tt b {{ font-variant-numeric: tabular-nums; }}
</style>
<div class="viz-root">
  <h1>OPHIS {html.escape(scope_tag)} — val_bpb over experiments</h1>
  <p class="sub">Every point is a val_bpb an actual run produced under one frozen scope
  (<code>{html.escape(scope_id)}</code>): one H200, a fixed 300&nbsp;s charged-training budget,
  corrected byte-fallback BPB. Nothing is normalised, fitted, or inferred. <strong>Only runs
  whose EvidenceDecision is <code>valid</code> are plotted</strong>{excluded_note}
  <strong>Experiment&nbsp;1 is the baseline itself</strong>: every bank control sits at x&nbsp;=&nbsp;1,
  so their vertical spread <em>is</em> the noise floor, and only a candidate advances the
  counter. Controls are re-measurements of one frozen baseline, not separate experiments,
  and the 1200&nbsp;s bank TTL forces a steady supply of them — counting each as an experiment
  would make the campaign look busier than it is.
  The host is shared, so a candidate is only ever read against the <strong>exact bank control
  frozen into its spec before launch</strong> on the same physical GPU — both are plotted and
  joined by a line. The grey band is ±2σ of the control replicates: anything inside it is an
  unordered tie. <strong>Nothing on this chart is SOTA</strong> — <code>promotion-due</code>
  is a work-allocation decision, and only a held-out-seed confirmation can move the record.</p>

  <div class="tiles">
    <div class="tile"><div class="k">Best measured</div><div class="v">{current_best:.4f}</div>
      <div class="d">exploratory · not SOTA</div></div>
    <div class="tile"><div class="k">First control</div><div class="v">{baseline:.4f}</div>
      <div class="d">{(controls[0]["steps"] if controls else 0):.0f} steps in 300&nbsp;s</div></div>
    <div class="tile"><div class="k">Experiments</div><div class="v">{max((r["i"] for r in rows), default=1)}</div>
      <div class="d">#1 is the baseline ({sum(1 for r in rows if r["status"] == "control")} control runs) ·
      {max((r["i"] for r in rows), default=1) - 1} idea(s) tested · {excluded} invalid excluded</div></div>
    <div class="tile"><div class="k">Control noise</div><div class="v">{control_sd:.5f}</div>
      <div class="d">{sd_text} · gate 0.000426</div></div>
  </div>

  <div class="card">
    <div class="wrap"><svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img"
      aria-label="measured val_bpb by experiment number with frozen paired controls">
      {noise_band}
      {grid}
      <line class="axis" x1="{PAD_L}" x2="{PAD_L}" y1="{PAD_T}" y2="{H - PAD_B}"/>
      <line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{H - PAD_B}" y2="{H - PAD_B}"/>
      {base_rule}
      {frontier}
      {"".join(marks)}
      {best_label}
      {xticks}
      <text class="alab" x="{PAD_L}" y="{H - 6}">experiment #</text>
      <text class="alab" x="{PAD_L - 52}" y="{PAD_T - 8}">val_bpb</text>
    </svg></div>
    <div class="legend">
      <span><span class="sq"></span>bank control</span>
      <span><span class="sw" style="background:var(--series-1)"></span>candidate, below gate</span>
      <span><span class="sw" style="background:var(--good)"></span>candidate, promotion-due</span>
      <span><span class="ring"></span>frozen paired control</span>
      <span><span class="line"></span>best so far</span>
      <span><span class="dash"></span>first control</span>
      <span><span class="x">✕</span>invalid / failed</span>
    </div>
  </div>

  <div class="tablewrap">
  <table>
    <thead><tr><th>#</th><th>lane</th><th>track</th><th>slot</th><th>val_bpb</th><th>steps</th>
      <th>overhead</th><th>frozen control (Δ)</th><th>status</th><th>what ran</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  </div>
  <div class="tt" id="tt"></div>
</div>
<script>
(function () {{
  var tt = document.getElementById('tt');
  document.querySelectorAll('.dot').forEach(function (el) {{
    el.addEventListener('mouseenter', function () {{
      var d = el.dataset;
      var head = d.kind === 'failed'
        ? '<b>' + d.verdict + '</b>'
        : (d.kind === 'frozen-control'
            ? 'frozen control <b>' + Number(d.val).toFixed(6) + '</b>'
            : 'val_bpb <b>' + Number(d.val).toFixed(6) + '</b>');
      tt.innerHTML = '<div>#' + d.i + ' \\u00b7 ' + head + '</div>' +
        '<div class="m">' + (d.title || '') + '</div>' +
        '<div class="m">' + d.lane + ' \\u00b7 track ' + d.track + ' \\u00b7 ' + d.slot + ' \\u00b7 seed ' + d.seed + '</div>' +
        (Number(d.steps) > 0
          ? '<div class="m">' + d.steps + ' steps \\u00b7 ' + d.mfu + '% mfu \\u00b7 ' + d.overhead + ' overhead</div>'
          : '') +
        (d.failure ? '<div class="m">' + d.failure + '</div>' : '');
      tt.style.opacity = 1;
    }});
    el.addEventListener('mousemove', function (e) {{
      var x = e.clientX + 14, y = e.clientY + 14;
      if (x + 350 > window.innerWidth) x = e.clientX - 350;
      if (y + 150 > window.innerHeight) y = e.clientY - 150;
      tt.style.left = x + 'px'; tt.style.top = y + 'px';
    }});
    el.addEventListener('mouseleave', function () {{ tt.style.opacity = 0; }});
  }});
}})();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".autoresearch")
    parser.add_argument("--scope", default="runs/scope.json")
    parser.add_argument("--output", default="runs/val_bpb_progress.html")
    parser.add_argument(
        "--show-invalid",
        action="store_true",
        help="also plot runs whose EvidenceDecision was not valid (drawn as ✕)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect(root)
    if not rows:
        raise SystemExit(f"no ResultBundles under {root}/records/result_bundle")

    scope_id = "unknown-scope"
    scope_path = Path(args.scope)
    if scope_path.is_file():
        try:
            scope_id = str(json.loads(scope_path.read_text()).get("id", scope_id))
        except (OSError, ValueError):
            pass

    # One chart per scope, and this filter runs first. A scope change means a
    # different corpus, byte accounting, or evaluator, so its runs are a different
    # leaderboard: they must not share an axis, a noise band, or a numbering with
    # the previous scope. That is exactly the cross-scope comparison the campaign
    # rules forbid, and the chart is where it would be committed by accident.
    off_scope = 0
    if scope_id != "unknown-scope":
        in_scope = [r for r in rows if r["scope_id"] == scope_id]
        off_scope = len(rows) - len(in_scope)
        if not in_scope:
            raise SystemExit(
                f"no landed runs under scope {scope_id} "
                f"({off_scope} belong to earlier scopes)"
            )
        rows = in_scope

    # Operator preference: plot measurements only. An invalid run is a harness or
    # host failure, not a score, and mixing the two makes the frontier hard to read.
    # The count is still carried into the subtitle rather than dropped in silence --
    # an executed replicate that vanishes without a trace is how selection effects
    # get laundered, and the immutable store keeps every one of them regardless.
    excluded = 0
    if not args.show_invalid:
        kept = [r for r in rows if r["status"] != "failed"]
        excluded = len(rows) - len(kept)
        if not kept:
            raise SystemExit(
                f"no valid runs yet ({excluded} landed but invalid); "
                "pass --show-invalid to plot them anyway"
            )
        rows = kept

    # Experiment #1 IS the baseline, and only candidates advance the counter.
    #
    # Numbering every landed run made the axis mostly bookkeeping: 15 of the first 17
    # points were bank controls, so "experiment 17" described a campaign that had
    # tested two ideas. Controls are re-measurements of one frozen baseline, not
    # separate experiments, and the bank TTL (1200s) forces a steady supply of them --
    # counting each as an experiment makes the campaign look busier than it is.
    #
    # So every control sits at x=1, where their vertical spread shows the noise floor
    # directly, and each candidate takes the next integer in time order. The axis now
    # answers "how many ideas have been tested", which is the question worth asking.
    controls = [r for r in rows if r["status"] == "control"]
    candidates = [r for r in rows if r["status"] != "control"]
    for row in controls:
        row["i"] = 1
    for index, row in enumerate(candidates, start=2):
        row["i"] = index

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(rows, scope_id, excluded=excluded))

    valid = [r for r in rows if r["status"] != "failed"]
    best = min((r["val"] for r in valid), default=0.0)
    print(
        f"wrote {output}: {len(rows)} plotted, {len(valid)} valid, "
        f"{excluded} invalid excluded, {off_scope} from earlier scopes excluded, "
        f"best measured {best:.6f} (exploratory, not SOTA)"
    )


if __name__ == "__main__":
    main()
