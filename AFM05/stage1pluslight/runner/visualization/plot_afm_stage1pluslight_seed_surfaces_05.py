"""AFM05 stage1pluslight seed-specific mechanistic loss surfaces.

This complements the existing best-over-seeds envelope plot by drawing:
- fixed NN seed 45
- fixed NN seed 55
- fixed NN seed 142
- mean over all finite viable seeds at each mechanistic grid point

The z axis is log10(train_loss).  The underlying loss is not redefined; only
the grouping over NN seeds is changed for visualization.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def finite_float(value: Any, default: float = math.nan) -> float:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    return default


def record_params(rec: dict[str, Any]) -> dict[str, Any]:
    params = rec.get("params", {})
    return params if isinstance(params, dict) else {}


def record_loss(rec: dict[str, Any]) -> float:
    return finite_float(rec.get("train_loss", rec.get("loss", math.nan)))


def record_seed_idx(rec: dict[str, Any]) -> int | None:
    val = record_params(rec).get("nn_seed_bank_idx")
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def record_trial_id(rec: dict[str, Any]) -> int | None:
    val = record_params(rec).get("trial_id")
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def record_ks_cs(rec: dict[str, Any]) -> tuple[float, float] | None:
    params = record_params(rec)
    ks = finite_float(params.get("ks0", rec.get("ks_hat", math.nan)))
    cs = finite_float(params.get("cs0", rec.get("cs_hat", math.nan)))
    if not (math.isfinite(ks) and ks > 0.0 and math.isfinite(cs) and cs > 0.0):
        return None
    return ks, cs


def record_node(rec: dict[str, Any]) -> tuple[int | None, int | None]:
    params = record_params(rec)
    out: list[int | None] = []
    for key in ("ks_node_idx", "cs_node_idx"):
        try:
            out.append(int(params.get(key)))
        except (TypeError, ValueError):
            out.append(None)
    return out[0], out[1]


def sci_ticks(values: list[float]) -> tuple[list[int], list[str]]:
    finite = [v for v in values if math.isfinite(v) and v > 0.0]
    lo = math.floor(math.log10(min(finite)))
    hi = math.ceil(math.log10(max(finite)))
    vals = list(range(int(lo), int(hi) + 1))
    return vals, [f"1e{v}" for v in vals]


def format_e(value: Any) -> str:
    out = finite_float(value)
    return f"{out:.6e}" if math.isfinite(out) else "None"


def load_records(result_path: Path) -> list[dict[str, Any]]:
    with result_path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload: {result_path}")
    raw_records = payload.get("trial_parameters")
    if not isinstance(raw_records, list):
        raise KeyError(f"Payload missing list field trial_parameters: {result_path}")
    records = [rec for rec in raw_records if isinstance(rec, dict)]
    if not records:
        raise RuntimeError(f"No trial records found: {result_path}")
    return records


def viable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        loss = record_loss(rec)
        if not (math.isfinite(loss) and loss > 0.0):
            continue
        if bool(rec.get("trial_failed", False)):
            continue
        if not bool(rec.get("is_viable", True)):
            continue
        out.append(rec)
    return out


def grid_values(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    ks_values: dict[str, float] = {}
    cs_values: dict[str, float] = {}
    for rec in records:
        pair = record_ks_cs(rec)
        if pair is None:
            continue
        ks, cs = pair
        ks_values[f"{ks:.16e}"] = ks
        cs_values[f"{cs:.16e}"] = cs
    return sorted(ks_values.values()), sorted(cs_values.values())


def fixed_seed_surface(records: list[dict[str, Any]], seed_idx: int) -> dict[tuple[str, str], tuple[float, dict[str, Any]]]:
    points: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    for rec in records:
        if record_seed_idx(rec) != seed_idx:
            continue
        pair = record_ks_cs(rec)
        loss = record_loss(rec)
        if pair is None or not (math.isfinite(loss) and loss > 0.0):
            continue
        key = (f"{pair[0]:.16e}", f"{pair[1]:.16e}")
        prev = points.get(key)
        if prev is None or loss < prev[0]:
            points[key] = (loss, rec)
    return points


def mean_seed_surface(records: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[float, dict[str, Any]]]:
    losses_by_grid: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for rec in records:
        pair = record_ks_cs(rec)
        loss = record_loss(rec)
        if pair is None or not (math.isfinite(loss) and loss > 0.0):
            continue
        key = (f"{pair[0]:.16e}", f"{pair[1]:.16e}")
        losses_by_grid[key].append((loss, rec))

    points: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    for key, items in losses_by_grid.items():
        if not items:
            continue
        mean_loss = sum(loss for loss, _ in items) / len(items)
        representative = min(items, key=lambda item: item[0])[1]
        points[key] = (mean_loss, representative)
    return points


def make_hover(
    *,
    title: str,
    ks: float,
    cs: float,
    loss: float,
    rec: dict[str, Any],
    seed_label: str,
) -> str:
    params = record_params(rec)
    node = record_node(rec)
    return (
        f"{title}"
        f"<br>{seed_label}"
        f"<br>train_loss={format_e(loss)}"
        f"<br>log10(train_loss)={math.log10(loss):.6f}"
        f"<br>ks={format_e(ks)} N/m"
        f"<br>cs={format_e(cs)} N·s/m"
        f"<br>node=({node[0]}, {node[1]})"
        f"<br>trial={record_trial_id(rec)}"
        f"<br>seed_bank_idx={params.get('nn_seed_bank_idx', 'None')}"
        f"<br>nn_init_seed={params.get('nn_init_seed', 'None')}"
    )


def write_surface_html(
    *,
    out_path: Path,
    result_path: Path,
    title: str,
    points: dict[tuple[str, str], tuple[float, dict[str, Any]]],
    ks_values: list[float],
    cs_values: list[float],
    seed_label: str,
    highlight_lowest_n: int = 0,
    highlight_name: str | None = None,
    gt_ks: float | None = None,
    gt_cs: float | None = None,
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
            loss, rec = item
            log_loss = math.log10(loss)
            hover = make_hover(title=title, ks=ks, cs=cs, loss=loss, rec=rec, seed_label=seed_label)
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

    traces = [
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
            "name": "mechanistic grid points",
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
    if gt_ks is not None and math.isfinite(gt_ks) and gt_ks > 0.0:
        gt_x = math.log10(gt_ks)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "\\bar{k_s}",
                "x": [gt_x, gt_x],
                "y": [min(y), max(y)],
                "z": [zlo - zpad * 0.35, zlo - zpad * 0.35],
                "hovertemplate": f"\\bar{{k_s}}={format_e(gt_ks)} N/m<extra></extra>",
                "line": {"color": "#000000", "width": 7},
            }
        )
    if gt_cs is not None and math.isfinite(gt_cs) and gt_cs > 0.0:
        gt_y = math.log10(gt_cs)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "\\bar{c_s}",
                "x": [min(x), max(x)],
                "y": [gt_y, gt_y],
                "z": [zlo - zpad * 0.35, zlo - zpad * 0.35],
                "hovertemplate": f"\\bar{{c_s}}={format_e(gt_cs)} N路s/m<extra></extra>",
                "line": {"color": "#000000", "width": 7},
            }
        )
    layout = {
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "legend": {"orientation": "h", "x": 0, "y": 1.02},
        "margin": {"l": 0, "r": 0, "t": 38, "b": 0},
        "annotations": [
            {
                "text": f"source: {str(result_path).replace(chr(92), chr(92) * 2)}<br>{title}; points={len(px)}",
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
                "title": "c_s [N·s/m]",
                "tickmode": "array",
                "tickvals": yticks,
                "ticktext": yticklabels,
                "range": [min(y), max(y)],
            },
            "zaxis": {
                "title": "log10(train_loss)",
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
    default_result = script_dir.parents[1] / "results_afm" / "afm_param_stage1pluslight_05.pkl"
    result_path = Path(argv[0]).resolve() if argv else default_result.resolve()
    out_dir = Path(argv[1]).resolve() if len(argv) > 1 else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = viable_records(load_records(result_path))
    ks_values, cs_values = grid_values(records)

    jobs: list[tuple[str, str, dict[tuple[str, str], tuple[float, dict[str, Any]]], str, int]] = []
    for seed_idx in (45, 55, 142):
        jobs.append(
            (
                f"AFM05 st1pl seed {seed_idx} mechanistic loss surface",
                f"afm_param_stage1pluslight_05_seed{seed_idx:03d}_ks_logcs_train_loss_3d.html",
                fixed_seed_surface(records, seed_idx),
                f"fixed NN seed_idx={seed_idx}",
                0,
            )
        )
    jobs.append(
        (
            "AFM05 st1pl mean-over-seeds mechanistic loss surface",
            "afm_param_stage1pluslight_05_mean_over_seeds_ks_logcs_train_loss_3d.html",
            mean_seed_surface(records),
            "mean over finite viable NN seeds at this mech point",
            30,
        )
    )

    for title, filename, points, seed_label, highlight_lowest_n in jobs:
        out_path = out_dir / filename
        write_surface_html(
            out_path=out_path,
            result_path=result_path,
            title=title,
            points=points,
            ks_values=ks_values,
            cs_values=cs_values,
            seed_label=seed_label,
            highlight_lowest_n=highlight_lowest_n,
            highlight_name=f"lowest {highlight_lowest_n} mean-loss points" if highlight_lowest_n else None,
        )
        print(f"saved {out_path} ({len(points)} grid points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
