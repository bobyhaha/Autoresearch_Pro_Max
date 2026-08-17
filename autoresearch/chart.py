"""Live val_bpb progress chart, rendered from the leaderboard.

Pure stdlib SVG: no plotting dependency, so the chart can be redrawn after every
single experiment without the control plane growing a scientific-python stack.
Every point resolves to a ResultBundle id -- the chart cannot show a number that
is not in the immutable registry.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

WIDTH = 1200
HEIGHT = 620
LEFT = 82
RIGHT = 28
TOP = 54
BOTTOM = 62
KEPT = "#34a853"
DISCARDED = "#c8ccd0"
INVALID = "#d93025"


def _nice_bounds(low: float, high: float) -> tuple[float, float, float]:
    if high <= low:
        high = low + 1e-3
    span = high - low
    padded_low = low - span * 0.12
    padded_high = high + span * 0.12
    raw = (padded_high - padded_low) / 5
    magnitude = 10 ** _floor_log10(raw)
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if step >= raw:
            break
    start = step * int(padded_low / step)
    while start > padded_low:
        start -= step
    return start, padded_high, step


def _floor_log10(value: float) -> int:
    exponent = 0
    if value <= 0:
        return 0
    while value < 1:
        value *= 10
        exponent -= 1
    while value >= 10:
        value /= 10
        exponent += 1
    return exponent


def render_svg(board: Mapping[str, Any]) -> str:
    entries: Sequence[Mapping[str, Any]] = [
        row for row in board.get("entries", []) if row.get("value") is not None
    ]
    title = (
        f"Autoresearch progress: {board.get('experiments', 0)} experiments, "
        f"{board.get('accepted', 0)} kept improvements"
    )
    if not entries:
        return _empty_svg(title)

    values = [float(row["value"]) for row in entries]
    count = max(len(board.get("entries", [])), 1)
    y_low, y_high, y_step = _nice_bounds(min(values), max(values))
    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM

    def x_of(index: int) -> float:
        return LEFT + (plot_w * index / max(count - 1, 1))

    def y_of(value: float) -> float:
        return TOP + plot_h * (1 - (value - y_low) / (y_high - y_low))

    style = (
        "<style>"
        ".bg{fill:var(--bg,#ffffff)}"
        ".tx{fill:var(--fg,#1f2328);font-family:ui-sans-serif,system-ui,sans-serif}"
        ".mut{fill:var(--mut,#6b7280);font-family:ui-sans-serif,system-ui,sans-serif}"
        ".grid{stroke:var(--grid,#e8eaed);stroke-width:1}"
        ".ax{stroke:var(--fg,#1f2328);stroke-width:1.2}"
        "@media (prefers-color-scheme:dark){"
        ".bg{fill:#0d1117}.tx{fill:#e6edf3}.mut{fill:#9198a1}.grid{stroke:#21262d}"
        ".ax{stroke:#e6edf3}}"
        "</style>"
    )
    parts: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="100%" role="img" aria-label="{html.escape(title)}">'
        ),
        style,
        f'<rect class="bg" width="{WIDTH}" height="{HEIGHT}"/>',
        (
            f'<text class="tx" x="{WIDTH / 2}" y="30" text-anchor="middle" '
            f'font-size="17" font-weight="600">{html.escape(title)}</text>'
        ),
    ]

    tick = y_low
    while tick <= y_high + 1e-12:
        y = y_of(tick)
        if TOP - 1 <= y <= TOP + plot_h + 1:
            parts.append(
                f'<line class="grid" x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + plot_w}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text class="mut" x="{LEFT - 10}" y="{y + 4:.1f}" '
                f'text-anchor="end" font-size="12">{tick:.4f}</text>'
            )
        tick += y_step

    parts.append(
        f'<line class="ax" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + plot_h}"/>'
        f'<line class="ax" x1="{LEFT}" y1="{TOP + plot_h}" '
        f'x2="{LEFT + plot_w}" y2="{TOP + plot_h}"/>'
    )

    step = max(1, count // 10)
    for index in range(0, count, step):
        x = x_of(index)
        parts.append(
            f'<text class="mut" x="{x:.1f}" y="{TOP + plot_h + 22}" '
            f'text-anchor="middle" font-size="12">{index}</text>'
        )
    parts.append(
        f'<text class="mut" x="{LEFT + plot_w / 2}" y="{HEIGHT - 18}" '
        f'text-anchor="middle" font-size="13">Experiment #</text>'
    )
    parts.append(
        f'<text class="mut" transform="translate(20 {TOP + plot_h / 2}) rotate(-90)" '
        f'text-anchor="middle" font-size="13">Validation BPB (lower is better)</text>'
    )

    # Running-best staircase.
    staircase: list[str] = []
    previous_y: float | None = None
    for row in board.get("entries", []):
        best = row.get("running_best")
        if best is None:
            continue
        x = x_of(int(row["experiment"]))
        y = y_of(float(best))
        if previous_y is None:
            staircase.append(f"M {x:.1f} {y:.1f}")
        else:
            staircase.append(f"L {x:.1f} {previous_y:.1f} L {x:.1f} {y:.1f}")
        previous_y = y
    if staircase and previous_y is not None:
        staircase.append(f"L {x_of(count - 1):.1f} {previous_y:.1f}")
        parts.append(
            f'<path d="{" ".join(staircase)}" fill="none" stroke="{KEPT}" '
            f'stroke-width="2.2" stroke-linejoin="round"/>'
        )

    for row in entries:
        x = x_of(int(row["experiment"]))
        y = y_of(float(row["value"]))
        if row.get("verdict") != "valid":
            colour, radius, opacity = INVALID, 3.4, 0.75
        elif row.get("kept"):
            colour, radius, opacity = KEPT, 6.0, 1.0
        else:
            colour, radius, opacity = DISCARDED, 3.4, 0.9
        tip = (
            f"#{row['experiment']} {row['label']} | {row['value']:.6f} "
            f"| seed {row['seed']} | {row['verdict']} | {row['result_id']}"
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{colour}" '
            f'opacity="{opacity}" stroke="#ffffff" stroke-width="1">'
            f"<title>{html.escape(tip)}</title></circle>"
        )

    for row in entries:
        if not row.get("kept"):
            continue
        x = x_of(int(row["experiment"]))
        y = y_of(float(row["value"]))
        label = row["label"]
        if len(label) > 46:
            label = label[:45] + "…"
        parts.append(
            f'<text class="tx" x="{x + 9:.1f}" y="{y - 8:.1f}" font-size="11" '
            f'fill="{KEPT}" transform="rotate(-18 {x + 9:.1f} {y - 8:.1f})">'
            f"{html.escape(label)}</text>"
        )

    legend = [("Kept", KEPT, 6.0), ("Discarded", DISCARDED, 3.4), ("Invalid", INVALID, 3.4)]
    for index, (name, colour, radius) in enumerate(legend):
        x = LEFT + plot_w - 210 + index * 74
        parts.append(
            f'<circle cx="{x}" cy="{TOP + 8}" r="{radius}" fill="{colour}"/>'
            f'<text class="mut" x="{x + 10}" y="{TOP + 12}" font-size="11">{name}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _empty_svg(title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} 200" width="100%">'
        f'<rect width="{WIDTH}" height="200" fill="#ffffff"/>'
        f'<text x="{WIDTH / 2}" y="100" text-anchor="middle" font-size="15" '
        f'fill="#6b7280" font-family="ui-sans-serif,system-ui,sans-serif">'
        f"{html.escape(title)} — no scorable runs yet</text></svg>"
    )


PAGE_CSS = """
:root{
  --paper:#f4f6f7; --surface:#ffffff; --ink:#0e1418; --mut:#66757f;
  --rule:#dde3e6; --rule-soft:#e9eef0;
  --keep:#1f8a4c; --void:#a7b0b8; --bad:#b8402c;
  --plate:#fbfcfc;
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#0b1114; --surface:#131b20; --ink:#e3ebef; --mut:#8b9aa4;
    --rule:#232e35; --rule-soft:#1b242a; --plate:#0f171b;
    --keep:#3fb972; --void:#5d6b74; --bad:#e0705c;
  }
}
:root[data-theme="dark"]{
  --paper:#0b1114; --surface:#131b20; --ink:#e3ebef; --mut:#8b9aa4;
  --rule:#232e35; --rule-soft:#1b242a; --plate:#0f171b;
  --keep:#3fb972; --void:#5d6b74; --bad:#e0705c;
}
:root[data-theme="light"]{
  --paper:#f4f6f7; --surface:#ffffff; --ink:#0e1418; --mut:#66757f;
  --rule:#dde3e6; --rule-soft:#e9eef0; --plate:#fbfcfc;
  --keep:#1f8a4c; --void:#a7b0b8; --bad:#b8402c;
}
*{box-sizing:border-box}
body{
  margin:0; padding:32px 20px 64px; background:var(--paper); color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  line-height:1.5; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,"Cascadia Mono",monospace;
  font-variant-numeric:tabular-nums}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);
  font-weight:600}
h1{font-size:clamp(26px,3.4vw,36px);line-height:1.1;margin:6px 0 0;letter-spacing:-.02em;
  font-weight:700;text-wrap:balance}
.sub{color:var(--mut);font-size:14px;margin:8px 0 0;max-width:64ch}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.chip{font-size:11.5px;padding:4px 10px;border:1px solid var(--rule);border-radius:999px;
  color:var(--mut);background:var(--surface)}
.chip b{color:var(--ink);font-weight:600}
.readout{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
  border-radius:12px;overflow:hidden;
  grid-template-columns:repeat(auto-fit,minmax(148px,1fr))}
.cell{background:var(--surface);padding:14px 16px;display:flex;flex-direction:column;gap:5px}
.cell .k{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);
  font-weight:600}
.cell .v{font-size:22px;font-weight:600;letter-spacing:-.01em}
.cell.lead{grid-column:span 2}
.cell.lead .v{font-size:34px;color:var(--keep)}
.plate{background:var(--plate);border:1px solid var(--rule);border-radius:12px;
  padding:10px 12px 4px}
.scroll{overflow-x:auto}
h2{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);
  margin:0 0 10px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:12.5px;min-width:760px}
th{text-align:left;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--mut);font-weight:600;padding:0 12px 8px;border-bottom:1px solid var(--rule)}
td{padding:9px 12px;border-bottom:1px solid var(--rule-soft);vertical-align:top}
tr:last-child td{border-bottom:none}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tag{display:inline-block;font-size:10.5px;padding:2px 7px;border-radius:4px;font-weight:600;
  letter-spacing:.03em}
.tag.valid{color:var(--keep);background:color-mix(in srgb,var(--keep) 12%,transparent)}
.tag.invalid{color:var(--bad);background:color-mix(in srgb,var(--bad) 13%,transparent)}
.tag.unknown{color:var(--mut);background:color-mix(in srgb,var(--mut) 13%,transparent)}
.kept{color:var(--keep);font-weight:700}
.rid{font-size:10.5px;color:var(--mut)}
.foot{color:var(--mut);font-size:12px;border-top:1px solid var(--rule);padding-top:14px}
"""


def render_page(board: Mapping[str, Any]) -> str:
    """Self-contained control-panel page: readouts, chart, then the full run log."""
    best = "—" if board.get("best_value") is None else f"{board['best_value']:.6f}"
    entries = list(board.get("entries", []))
    row_html: list[str] = []
    for row in reversed(entries):
        value = "—" if row["value"] is None else f"{row['value']:.6f}"
        running = "—" if row["running_best"] is None else f"{row['running_best']:.6f}"
        verdict = html.escape(str(row["verdict"]))
        delta = "—" if row.get("delta") is None else f"{row['delta']:+.6f}"
        kept = "KEPT" if row["kept"] else ""
        row_html.append(
            f'<tr><td class="n mono">{row["experiment"]}</td>'
            f"<td>{html.escape(str(row['label']))}</td>"
            f'<td class="n mono">{row["seed"]}</td>'
            f'<td class="n mono">{value}</td>'
            f'<td class="n mono">{delta}</td>'
            f'<td><span class="tag {verdict}">{verdict}</span></td>'
            f'<td class="kept">{kept}</td>'
            f'<td class="n mono">{running}</td>'
            f'<td class="rid mono">{html.escape(str(row["result_id"]))}</td></tr>'
        )
    rows = "".join(row_html)
    audit = "due now" if board.get("audit_due") else "not due"
    return f"""<title>val_bpb progress</title>
<style>{PAGE_CSS}</style>
<div class="wrap">
  <header>
    <div class="eyebrow">OPHIS · fixed-frame search</div>
    <h1>val_bpb progress</h1>
    <p class="sub">One seed, one arm, one run per experiment. Every point resolves to a
    ResultBundle in the immutable registry — this chart cannot show a number that was not
    measured and judged admissible.</p>
    <div class="chips">
      <span class="chip">search seed <b>{board.get("search_seed")}</b> · fixed</span>
      <span class="chip">comparison group
        <b>{html.escape(str(board.get("comparison_group", "")))}</b></span>
      <span class="chip">seed audit <b>{audit}</b>
        (every {board.get("audit_every")} kept)</span>
    </div>
  </header>

  <section class="readout">
    <div class="cell lead">
      <span class="k">Running best</span><span class="v mono">{best}</span>
    </div>
    <div class="cell"><span class="k">Experiments</span>
      <span class="v mono">{board.get("experiments", 0)}</span></div>
    <div class="cell"><span class="k">Kept</span>
      <span class="v mono">{board.get("accepted", 0)}</span></div>
    <div class="cell"><span class="k">Invalid</span>
      <span class="v mono">{board.get("invalid", 0)}</span></div>
  </section>

  <section class="plate scroll">{render_svg(board)}</section>

  <section>
    <h2>Run log</h2>
    <div class="scroll"><table>
      <thead><tr><th>#</th><th>Change</th><th>Seed</th><th>val_bpb</th><th>vs. baseline</th>
      <th>Verdict</th><th>Kept</th><th>Running best</th><th>ResultBundle</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
  </section>

  <p class="foot">Rebuilt {html.escape(str(board.get("generated_at", "")))} from
  EvidenceDecisions alone. An invalid measurement can never become the running best.</p>
</div>
"""
