"""Plot a 3D ks/cs/loss surface from a DMT-KV identifiability sweep CSV.

The companion sweep script writes one row per (ks, cs) pair.  This script
reconstructs the regular grid and emits a Plotly HTML surface:

    x = log10(ks)
    y = log10(cs)
    z = log10(max(selected normalized loss, loss_floor))

Using log coordinates keeps the plot consistent with the AFM04 mechanistic
grid visualizations, where ks/cs span orders of magnitude.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PATTERN = "dmt_kv_ks_cs_loss_surface_*_600x600.csv"

METRIC_TO_COLUMN = {
    "observed": "loss_observed",
    "state": "loss_state",
    "full": "loss_full",
    "x1": "loss_x1",
    "x2": "loss_x2",
    "x3": "loss_x3",
    "x2dot": "loss_x2dot",
    "fts": "loss_fts",
}

METRIC_LABEL = {
    "observed": "observed loss",
    "state": "state loss",
    "full": "full loss",
    "x1": "x1 loss",
    "x2": "x2 loss",
    "x3": "x3 loss",
    "x2dot": "x2dot loss",
    "fts": "Fts loss",
}

RAINBOW_PURPLE_LOW_RED_HIGH = [
    [0.0, "#4b0082"],
    [1.0 / 6.0, "#0000ff"],
    [2.0 / 6.0, "#00ffff"],
    [3.0 / 6.0, "#00aa00"],
    [4.0 / 6.0, "#ffff00"],
    [5.0 / 6.0, "#ff7f00"],
    [1.0, "#ff0000"],
]


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def find_default_csv() -> Path:
    candidates = sorted(THIS_DIR.glob(DEFAULT_PATTERN), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(THIS_DIR.glob("dmt_kv_ks_cs_loss_surface_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No sweep CSV found in {THIS_DIR}")
    return candidates[0]


def infer_metric_from_name(path: Path) -> str:
    name = path.stem
    for metric in METRIC_TO_COLUMN:
        if re.search(rf"_{re.escape(metric)}_", name) or name.endswith(f"_{metric}"):
            return metric
    return "observed"


def log10_or_nan(value: float, *, floor: float | None = None) -> float:
    if not (math.isfinite(value) and value > 0.0):
        return math.nan
    if floor is not None and math.isfinite(floor) and floor > 0.0:
        value = max(value, floor)
    return math.log10(value)


def read_surface_csv(
    path: Path,
    metric: str,
    *,
    loss_floor: float,
) -> tuple[list[float], list[float], list[list[float]], float, float, float]:
    column = METRIC_TO_COLUMN[metric]
    rows: list[tuple[float, float, float]] = []
    ks_seen: dict[str, float] = {}
    cs_seen: dict[str, float] = {}

    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        if column not in reader.fieldnames:
            raise KeyError(f"Metric column {column!r} not found in {path}")

        for row in reader:
            ks = finite_float(row.get("ks"))
            cs = finite_float(row.get("cs"))
            loss = finite_float(row.get(column))
            if not (math.isfinite(ks) and ks > 0.0 and math.isfinite(cs) and cs > 0.0):
                continue
            rows.append((ks, cs, loss))
            ks_seen[f"{ks:.17e}"] = ks
            cs_seen[f"{cs:.17e}"] = cs

    if not rows:
        raise RuntimeError(f"No finite ks/cs rows loaded from {path}")

    ks_values = sorted(ks_seen.values())
    cs_values = sorted(cs_seen.values())
    ks_index = {f"{value:.17e}": i for i, value in enumerate(ks_values)}
    cs_index = {f"{value:.17e}": i for i, value in enumerate(cs_values)}

    # Plotly surface wants z as rows over y, columns over x.
    z_grid = [[math.nan for _ in ks_values] for _ in cs_values]
    best_ks = math.nan
    best_cs = math.nan
    best_loss = math.inf
    for ks, cs, loss in rows:
        i = ks_index[f"{ks:.17e}"]
        j = cs_index[f"{cs:.17e}"]
        z_grid[j][i] = log10_or_nan(loss, floor=loss_floor)
        if math.isfinite(loss) and loss > 0.0 and loss < best_loss:
            best_ks = ks
            best_cs = cs
            best_loss = loss

    return ks_values, cs_values, z_grid, best_ks, best_cs, best_loss


def tick_spec(values: list[float]) -> tuple[list[float], list[str]]:
    if not values:
        return [], []
    lo = min(values)
    hi = max(values)
    lo_exp = math.floor(math.log10(lo))
    hi_exp = math.ceil(math.log10(hi))
    ticks = [float(exp) for exp in range(int(lo_exp), int(hi_exp) + 1)]
    labels = [f"1e{exp}" for exp in range(int(lo_exp), int(hi_exp) + 1)]
    return ticks, labels


def html_template(*, title: str, data_json: str, layout_json: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; font-family: Arial, sans-serif; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const data = {data_json};
    const layout = {layout_json};
    Plotly.newPlot("plot", data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""


def write_plot(
    *,
    out_path: Path,
    csv_path: Path,
    metric: str,
    ks_values: list[float],
    cs_values: list[float],
    z_grid: list[list[float]],
    best_ks: float,
    best_cs: float,
    best_loss: float,
    loss_floor: float,
    surface_opacity: float,
) -> None:
    x = [log10_or_nan(value) for value in ks_values]
    y = [log10_or_nan(value) for value in cs_values]
    best_z = log10_or_nan(best_loss, floor=loss_floor)
    metric_label = METRIC_LABEL.get(metric, f"{metric} loss")

    x_ticks, x_ticktext = tick_spec(ks_values)
    y_ticks, y_ticktext = tick_spec(cs_values)

    title = f"DMT-KV ks/cs/loss 3D surface | metric={metric}"
    surface = {
        "type": "surface",
        "x": x,
        "y": y,
        "z": z_grid,
        "opacity": float(surface_opacity),
        "colorscale": RAINBOW_PURPLE_LOW_RED_HIGH,
        "cmin": math.log10(loss_floor),
        "colorbar": {
            "title": {"text": f"log10({metric_label})", "font": {"size": 20}},
            "tickfont": {"size": 18},
        },
        "hovertemplate": (
            "log10(ks)=%{x:.6f}<br>"
            "log10(cs)=%{y:.6f}<br>"
            f"log10(max({metric} loss, {loss_floor:.0e}))=%{{z:.6f}}"
            "<extra></extra>"
        ),
        "contours": {
            "z": {
                "show": True,
                "usecolormap": True,
                "highlightcolor": "#ffffff",
                "project": {"z": True},
            }
        },
    }

    traces: list[dict[str, Any]] = [surface]
    if math.isfinite(best_ks) and math.isfinite(best_cs) and math.isfinite(best_z):
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": "minimum",
                "x": [log10_or_nan(best_ks)],
                "y": [log10_or_nan(best_cs)],
                "z": [best_z],
                "marker": {"size": 4, "color": "black", "symbol": "circle"},
                "hovertemplate": (
                    f"best ks={best_ks:.9e}<br>"
                    f"best cs={best_cs:.9e}<br>"
                    f"best {metric} loss={best_loss:.9e}<br>"
                    f"plotted floor={loss_floor:.9e}<br>"
                    f"log10(plotted loss)={best_z:.6f}<extra></extra>"
                ),
            }
        )

    layout = {
        "scene": {
            "xaxis": {
                "title": {"text": "ks [N/m]", "font": {"size": 22}},
                "tickmode": "array",
                "tickvals": x_ticks,
                "ticktext": x_ticktext,
                "tickfont": {"size": 18},
            },
            "yaxis": {
                "title": {"text": "cs [N·s/m]", "font": {"size": 22}},
                "tickmode": "array",
                "tickvals": y_ticks,
                "ticktext": y_ticktext,
                "tickfont": {"size": 18},
            },
            "zaxis": {
                "title": {"text": f"log10({metric_label})", "font": {"size": 22}},
                "tickfont": {"size": 18},
            },
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 0.95}},
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 0},
        "legend": {"orientation": "h", "x": 0.02, "y": 0.98},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        html_template(
            title=title,
            data_json=json.dumps(traces, separators=(",", ":"), allow_nan=True),
            layout_json=json.dumps(layout, separators=(",", ":"), allow_nan=True),
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="Sweep CSV. Defaults to newest 600x600 sweep in this folder.")
    parser.add_argument("--metric", choices=sorted(METRIC_TO_COLUMN), default="", help="Loss metric to plot. Defaults to metric inferred from filename.")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path.")
    parser.add_argument("--loss-floor", type=float, default=1.0e-12, help="Clamp positive losses to this floor before log10 plotting.")
    parser.add_argument("--surface-opacity", type=float, default=0.78, help="Surface opacity; 1.0 is opaque.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv if args.csv is not None else find_default_csv()
    csv_path = csv_path.resolve()
    metric = args.metric.strip() if args.metric else infer_metric_from_name(csv_path)
    if metric not in METRIC_TO_COLUMN:
        raise ValueError(f"Unsupported metric {metric!r}")

    out_path = args.out
    if out_path is None:
        out_path = csv_path.with_name(f"{csv_path.stem}_3d.html")

    loss_floor = float(args.loss_floor)
    if not (math.isfinite(loss_floor) and loss_floor > 0.0):
        raise ValueError(f"--loss-floor must be positive and finite, got {loss_floor!r}")

    ks_values, cs_values, z_grid, best_ks, best_cs, best_loss = read_surface_csv(csv_path, metric, loss_floor=loss_floor)
    write_plot(
        out_path=out_path,
        csv_path=csv_path,
        metric=metric,
        ks_values=ks_values,
        cs_values=cs_values,
        z_grid=z_grid,
        best_ks=best_ks,
        best_cs=best_cs,
        best_loss=best_loss,
        loss_floor=loss_floor,
        surface_opacity=args.surface_opacity,
    )

    print(f"Loaded CSV: {csv_path}")
    print(f"Grid: ks={len(ks_values)} x cs={len(cs_values)}")
    print(f"Metric: {metric} ({METRIC_TO_COLUMN[metric]})")
    print(f"Loss floor: {loss_floor:.9e}")
    print(f"Best: ks={best_ks:.9e}, cs={best_cs:.9e}, loss={best_loss:.9e}")
    print(f"Saved HTML: {out_path.resolve()}")


if __name__ == "__main__":
    main()
