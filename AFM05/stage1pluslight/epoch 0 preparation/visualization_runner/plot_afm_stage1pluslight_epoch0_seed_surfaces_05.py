"""AFM05 epoch-0-preparation mechanistic loss surface.

This mirrors
AFM05/stage1pluslight/runner/visualization/plot_afm_stage1pluslight_seed_surfaces_05.py,
but the plotted loss is the epoch-0-preparation loss produced after the st1pl
endpoint:

    epoch0_train_loss

The epoch-0 preparation run evaluates one selected st1pl-best viable trial per
mechanistic grid point. Therefore the meaningful visualization here is a single
full 20x50 mechanistic surface after epoch-0 preparation, not separate fixed-seed
surfaces.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


CSV_NAME = "afm05_st1pl_epoch0_preparation_best_mech_winners.csv"


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_e(value: Any) -> str:
    out = finite_float(value)
    return f"{out:.6e}" if math.isfinite(out) else "None"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    return rows


def row_loss(row: dict[str, str]) -> float:
    return finite_float(row.get("epoch0_train_loss", row.get("train_loss", math.nan)))


def row_seed_idx(row: dict[str, str]) -> int | None:
    return finite_int(row.get("nn_seed_bank_idx"))


def row_trial_id(row: dict[str, str]) -> int | None:
    return finite_int(row.get("trial_id"))


def row_node(row: dict[str, str]) -> tuple[int | None, int | None]:
    return finite_int(row.get("ks_node_idx")), finite_int(row.get("cs_node_idx"))


def row_ks_cs(row: dict[str, str]) -> tuple[float, float] | None:
    ks = finite_float(row.get("ks"))
    cs = finite_float(row.get("cs"))
    if not (math.isfinite(ks) and ks > 0.0 and math.isfinite(cs) and cs > 0.0):
        return None
    return ks, cs


def viable_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        loss = row_loss(row)
        if not (math.isfinite(loss) and loss > 0.0):
            continue
        status = str(row.get("status", "ok")).lower()
        if status and status != "ok":
            continue
        if row_ks_cs(row) is None:
            continue
        out.append(row)
    if not out:
        raise RuntimeError("No finite ok epoch0 rows available for plotting.")
    return out


def grid_values(rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    ks_values: dict[str, float] = {}
    cs_values: dict[str, float] = {}
    for row in rows:
        pair = row_ks_cs(row)
        if pair is None:
            continue
        ks, cs = pair
        ks_values[f"{ks:.16e}"] = ks
        cs_values[f"{cs:.16e}"] = cs
    return sorted(ks_values.values()), sorted(cs_values.values())


def fixed_seed_surface(rows: list[dict[str, str]], seed_idx: int) -> dict[tuple[str, str], tuple[float, dict[str, str]]]:
    points: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}
    for row in rows:
        if row_seed_idx(row) != seed_idx:
            continue
        pair = row_ks_cs(row)
        loss = row_loss(row)
        if pair is None or not (math.isfinite(loss) and loss > 0.0):
            continue
        key = (f"{pair[0]:.16e}", f"{pair[1]:.16e}")
        prev = points.get(key)
        if prev is None or loss < prev[0]:
            points[key] = (loss, row)
    return points


def mean_surface(rows: list[dict[str, str]]) -> dict[tuple[str, str], tuple[float, dict[str, str]]]:
    grouped: dict[tuple[str, str], list[tuple[float, dict[str, str]]]] = defaultdict(list)
    for row in rows:
        pair = row_ks_cs(row)
        loss = row_loss(row)
        if pair is None or not (math.isfinite(loss) and loss > 0.0):
            continue
        key = (f"{pair[0]:.16e}", f"{pair[1]:.16e}")
        grouped[key].append((loss, row))

    points: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}
    for key, items in grouped.items():
        mean_loss = sum(loss for loss, _ in items) / len(items)
        representative = min(items, key=lambda item: item[0])[1]
        points[key] = (mean_loss, representative)
    return points


def best_epoch0_surface(rows: list[dict[str, str]]) -> dict[tuple[str, str], tuple[float, dict[str, str]]]:
    points: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}
    for row in rows:
        pair = row_ks_cs(row)
        loss = row_loss(row)
        if pair is None or not (math.isfinite(loss) and loss > 0.0):
            continue
        key = (f"{pair[0]:.16e}", f"{pair[1]:.16e}")
        prev = points.get(key)
        if prev is None or loss < prev[0]:
            points[key] = (loss, row)
    return points


def sci_ticks(values: list[float]) -> tuple[list[int], list[str]]:
    finite = [v for v in values if math.isfinite(v) and v > 0.0]
    lo = math.floor(math.log10(min(finite)))
    hi = math.ceil(math.log10(max(finite)))
    vals = list(range(int(lo), int(hi) + 1))
    return vals, [f"1e{v}" for v in vals]


def make_hover(
    *,
    title: str,
    ks: float,
    cs: float,
    loss: float,
    row: dict[str, str],
    seed_label: str,
) -> str:
    node = row_node(row)
    return (
        f"{title}"
        f"<br>{seed_label}"
        f"<br>epoch0_train_loss={format_e(loss)}"
        f"<br>log10(epoch0_train_loss)={math.log10(loss):.6f}"
        f"<br>st1pl_train_loss={format_e(row.get('st1pl_train_loss'))}"
        f"<br>ratio={format_e(row.get('epoch0_ratio_to_st1pl_train'))}"
        f"<br>k_s={format_e(ks)} N/m"
        f"<br>c_s={format_e(cs)} N&middot;s/m"
        f"<br>node=({node[0]}, {node[1]})"
        f"<br>trial={row_trial_id(row)}"
        f"<br>seed_bank_idx={row.get('nn_seed_bank_idx', 'None')}"
        f"<br>nn_init_seed={row.get('nn_init_seed', 'None')}"
    )


def write_surface_html(
    *,
    out_path: Path,
    source_csv: Path,
    title: str,
    points: dict[tuple[str, str], tuple[float, dict[str, str]]],
    ks_values: list[float],
    cs_values: list[float],
    seed_label: str,
    highlight_lowest_n: int = 0,
    highlight_name: str | None = None,
) -> None:
    if not points:
        raise RuntimeError(f"No points available for {title}.")

    ks_keys = [f"{ks:.16e}" for ks in ks_values]
    cs_keys = [f"{cs:.16e}" for cs in cs_values]
    x = [math.log10(ks) for ks in ks_values]
    y = [math.log10(cs) for cs in cs_values]
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    px: list[float] = []
    py: list[float] = []
    pz: list[float] = []
    ptext: list[str] = []
    point_rows: list[tuple[float, float, float, float, str]] = []

    for cs_key, cs in zip(cs_keys, cs_values):
        zrow: list[float | None] = []
        trow: list[str] = []
        for ks_key, ks in zip(ks_keys, ks_values):
            item = points.get((ks_key, cs_key))
            if item is None:
                zrow.append(None)
                trow.append("")
                continue
            loss, row = item
            log_loss = math.log10(loss)
            hover = make_hover(title=title, ks=ks, cs=cs, loss=loss, row=row, seed_label=seed_label)
            zrow.append(log_loss)
            trow.append(hover)
            px.append(math.log10(ks))
            py.append(math.log10(cs))
            pz.append(log_loss)
            ptext.append(hover)
            point_rows.append((loss, math.log10(ks), math.log10(cs), log_loss, hover))
        z.append(zrow)
        text.append(trow)

    zfinite = [v for row in z for v in row if v is not None and math.isfinite(v)]
    zlo, zhi = min(zfinite), max(zfinite)
    zpad = max(0.015, (zhi - zlo) * 0.08)
    xticks, xticklabels = sci_ticks(ks_values)
    yticks, yticklabels = sci_ticks(cs_values)

    traces: list[dict[str, Any]] = [
        {
            "type": "surface",
            "name": title,
            "x": x,
            "y": y,
            "z": z,
            "text": text,
            "hovertemplate": "%{text}<extra></extra>",
            "opacity": 0.42,
            "connectgaps": False,
            "showscale": False,
            "colorscale": [
                [0.00, "#d9f0ff"],
                [0.35, "#74add1"],
                [0.70, "#2b83ba"],
                [1.00, "#08306b"],
            ],
            "contours": {"z": {"show": True, "usecolormap": True, "project": {"z": True}}},
        },
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "evaluated mechanistic points",
            "x": px,
            "y": py,
            "z": pz,
            "hovertext": ptext,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {"size": 3, "color": "#000000", "opacity": 0.96, "line": {"width": 0}},
        },
    ]

    if highlight_lowest_n > 0:
        lowest = sorted(point_rows, key=lambda item: item[0])[:highlight_lowest_n]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": highlight_name or f"lowest {len(lowest)} points",
                "x": [item[1] for item in lowest],
                "y": [item[2] for item in lowest],
                "z": [item[3] for item in lowest],
                "hovertext": [item[4] for item in lowest],
                "hovertemplate": "%{hovertext}<extra></extra>",
                "marker": {
                    "size": 8,
                    "color": "#d62728",
                    "opacity": 1.0,
                    "line": {"color": "#7f0000", "width": 1},
                },
            }
        )

    layout = {
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "legend": {"orientation": "h", "x": 0, "y": 1.02},
        "margin": {"l": 0, "r": 0, "t": 38, "b": 0},
        "annotations": [
            {
                "text": f"source: {str(source_csv).replace(chr(92), chr(92) * 2)}<br>{title}; points={len(px)}",
                "x": 0,
                "y": 1.06,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"size": 12, "color": "#444444"},
            }
        ],
        "scene": {
            "xaxis": {
                "title": "k_s [N/m]",
                "tickmode": "array",
                "tickvals": xticks,
                "ticktext": xticklabels,
                "range": [min(x), max(x)],
            },
            "yaxis": {
                "title": "c_s [N&middot;s/m]",
                "tickmode": "array",
                "tickvals": yticks,
                "ticktext": yticklabels,
                "range": [min(y), max(y)],
            },
            "zaxis": {
                "title": "log10(epoch0_train_loss)",
                "range": [zlo - zpad, zhi + zpad],
            },
            "camera": {"eye": {"x": 1.5, "y": 1.32, "z": 0.92}},
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #ffffff; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const traces = {json.dumps(traces, ensure_ascii=False)};
    const layout = {json.dumps(layout, ensure_ascii=False)};
    Plotly.newPlot('plot', traces, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    default_csv = base_dir / "logs" / CSV_NAME
    source_csv = Path(argv[0]).resolve() if argv else default_csv.resolve()
    out_dir = Path(argv[1]).resolve() if len(argv) > 1 else base_dir / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = viable_rows(load_rows(source_csv))
    ks_values, cs_values = grid_values(rows)

    points = best_epoch0_surface(rows)
    out_path = out_dir / "afm_param_stage1pluslight_05_epoch0_best_seed_envelope_points_ks_logcs_train_loss_3d.html"
    write_surface_html(
        out_path=out_path,
        source_csv=source_csv,
        title="AFM05 epoch0 best-seed envelope mechanistic loss surface",
        points=points,
        ks_values=ks_values,
        cs_values=cs_values,
        seed_label="selected st1pl-best seed after epoch0 preparation",
        highlight_lowest_n=30,
        highlight_name="lowest 30 epoch0 points",
    )
    print(f"saved {out_path} ({len(points)} evaluated grid points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
