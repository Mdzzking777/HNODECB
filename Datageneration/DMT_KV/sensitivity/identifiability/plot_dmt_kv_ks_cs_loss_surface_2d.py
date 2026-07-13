"""Plot 2-D DMT-KV ks/cs loss surfaces from an existing sweep CSV.

This is a no-sweep post-plotter.  It reads the CSV written by
``sweep_ks_cs_dmt_kv_loss_surface.py`` and writes a two-panel image:

    left:  log10(selected loss)
    right: selected loss on the raw linear color axis
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


THIS_DIR = Path(__file__).resolve().parent

RAINBOW_PURPLE_LOW_RED_HIGH = LinearSegmentedColormap.from_list(
    "rainbow_purple_low_red_high",
    ["#4b0082", "#0000ff", "#00ffff", "#00aa00", "#ffff00", "#ff7f00", "#ff0000"],
)

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


def infer_metric_from_name(path: Path) -> str:
    name = path.stem
    for metric in METRIC_TO_COLUMN:
        if re.search(rf"_{re.escape(metric)}_", name) or name.endswith(f"_{metric}"):
            return metric
    return "observed"


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".png")


def load_grid(csv_path: Path, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    column = METRIC_TO_COLUMN[metric]
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise RuntimeError(f"No rows loaded from {csv_path}")
    if column not in data.dtype.names:
        raise KeyError(f"Column {column!r} not found in {csv_path}")

    ks_flat = np.asarray(data["ks"], dtype=float)
    cs_flat = np.asarray(data["cs"], dtype=float)
    loss_flat = np.asarray(data[column], dtype=float)

    valid = np.isfinite(ks_flat) & (ks_flat > 0.0) & np.isfinite(cs_flat) & (cs_flat > 0.0)
    if not np.any(valid):
        raise RuntimeError(f"No finite positive ks/cs rows in {csv_path}")

    ks_flat = ks_flat[valid]
    cs_flat = cs_flat[valid]
    loss_flat = loss_flat[valid]

    ks_values = np.unique(ks_flat)
    cs_values = np.unique(cs_flat)
    ks_values.sort()
    cs_values.sort()

    grid = np.full((ks_values.size, cs_values.size), np.nan, dtype=float)
    i = np.searchsorted(ks_values, ks_flat)
    j = np.searchsorted(cs_values, cs_flat)
    grid[i, j] = loss_flat
    return ks_values, cs_values, grid


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("No finite values available for plotting")
    return float(np.nanpercentile(finite, float(percentile)))


def display_labels(metric: str) -> tuple[str, str]:
    if metric == "observed":
        return r"$\log_{10}(\mathcal{L})$", r"$\mathcal{L}$"
    return f"log10({metric} loss)", f"{metric} loss"


def plot_dual_surface(
    *,
    csv_path: Path,
    png_path: Path,
    pdf_path: Path | None,
    metric: str,
    clip_percentile: float,
    dpi: int,
    plot_mode: str,
    contours: bool,
    contour_levels: int,
    contour_color: str,
    contour_linewidth: float,
    log_floor: float | None,
    log_ceiling: float | None,
) -> None:
    ks_values, cs_values, loss_grid = load_grid(csv_path, metric)
    kk, cc = np.meshgrid(ks_values, cs_values, indexing="ij")

    positive_loss = np.where(loss_grid > 0.0, loss_grid, np.nan)
    log_grid = np.log10(positive_loss)
    if log_floor is not None or log_ceiling is not None:
        lo = -np.inf if log_floor is None else float(log_floor)
        hi = np.inf if log_ceiling is None else float(log_ceiling)
        log_grid = np.clip(log_grid, lo, hi)

    log_finite = log_grid[np.isfinite(log_grid)]
    raw_finite = positive_loss[np.isfinite(positive_loss)]
    if log_finite.size == 0 or raw_finite.size == 0:
        raise RuntimeError("No finite positive loss values available for plotting")

    log_vmin = float(np.nanmin(log_finite))
    log_vmax = finite_percentile(log_grid, clip_percentile)
    if not np.isfinite(log_vmax) or log_vmax <= log_vmin:
        log_vmax = float(np.nanmax(log_finite))

    raw_vmin = float(np.nanmin(raw_finite))
    raw_vmax = finite_percentile(positive_loss, clip_percentile)
    if not np.isfinite(raw_vmax) or raw_vmax <= raw_vmin:
        raw_vmax = float(np.nanmax(raw_finite))

    if plot_mode == "both":
        log_label, raw_label = display_labels(metric)
        fig, axes_obj = plt.subplots(1, 2, figsize=(16.5, 6.8), constrained_layout=True)
        axes = list(np.ravel(axes_obj))
        panels = [
            (axes[0], log_grid, log_vmin, log_vmax, log_label, log_label),
            (axes[1], positive_loss, raw_vmin, raw_vmax, raw_label, raw_label),
        ]
    elif plot_mode == "log":
        log_label, _raw_label = display_labels(metric)
        fig, ax = plt.subplots(figsize=(9.5, 7.6), constrained_layout=True)
        panels = [(ax, log_grid, log_vmin, log_vmax, log_label, log_label)]
    elif plot_mode == "raw":
        _log_label, raw_label = display_labels(metric)
        fig, ax = plt.subplots(figsize=(9.5, 7.6), constrained_layout=True)
        panels = [(ax, positive_loss, raw_vmin, raw_vmax, raw_label, raw_label)]
    else:
        raise ValueError(f"Unsupported plot_mode: {plot_mode}")

    for ax, values, vmin, vmax, title, cbar_label in panels:
        mesh = ax.pcolormesh(
            kk,
            cc,
            values,
            shading="auto",
            cmap=RAINBOW_PURPLE_LOW_RED_HIGH,
            vmin=vmin,
            vmax=vmax,
        )
        if contours:
            finite_values = values[np.isfinite(values)]
            if finite_values.size > 0:
                lo = float(np.nanmin(finite_values))
                hi = float(np.nanmax(finite_values))
                if hi > lo:
                    levels = np.linspace(lo, hi, max(2, int(contour_levels)))
                    ax.contour(
                        kk,
                        cc,
                        values,
                        levels=levels,
                        colors=str(contour_color),
                        linewidths=float(contour_linewidth),
                        alpha=0.86,
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ks [N/m]")
        ax.set_ylabel("cs [N s/m]")
        ax.set_title(title)
        ax.grid(False)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(cbar_label)
        if values is positive_loss:
            cbar.formatter = mticker.ScalarFormatter(useMathText=True)
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.update_ticks()

    fig.suptitle(f"DMT-KV ks/cs loss surface | source={csv_path.name}", fontsize=13)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=int(dpi))
    if pdf_path is not None:
        fig.savefig(pdf_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Sweep CSV with ks, cs, and loss_* columns.",
    )
    parser.add_argument("--metric", choices=tuple(METRIC_TO_COLUMN), default="")
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--plot-mode", choices=("both", "log", "raw"), default="both")
    parser.add_argument("--contours", action="store_true")
    parser.add_argument("--contour-levels", type=int, default=30)
    parser.add_argument("--contour-color", default="black")
    parser.add_argument("--contour-linewidth", type=float, default=0.65)
    parser.add_argument("--log-floor", type=float, default=float("nan"))
    parser.add_argument("--log-ceiling", type=float, default=float("nan"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    metric = args.metric.strip() or infer_metric_from_name(csv_path)
    png_path = args.png.resolve() if args.png is not None else default_output_path(csv_path)
    pdf_path = args.pdf.resolve() if args.pdf is not None else csv_path.with_suffix(".pdf")

    plot_dual_surface(
        csv_path=csv_path,
        png_path=png_path,
        pdf_path=pdf_path,
        metric=metric,
        clip_percentile=args.clip_percentile,
        dpi=args.dpi,
        plot_mode=args.plot_mode,
        contours=bool(args.contours),
        contour_levels=int(args.contour_levels),
        contour_color=str(args.contour_color),
        contour_linewidth=float(args.contour_linewidth),
        log_floor=None if not np.isfinite(args.log_floor) else float(args.log_floor),
        log_ceiling=None if not np.isfinite(args.log_ceiling) else float(args.log_ceiling),
    )
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
