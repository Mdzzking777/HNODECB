"""Render a static 3D ks/cs/loss surface PNG from a DMT-KV sweep CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


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

RAINBOW_PURPLE_LOW_RED_HIGH = LinearSegmentedColormap.from_list(
    "rainbow_purple_low_red_high",
    ["#4b0082", "#0000ff", "#00ffff", "#00aa00", "#ffff00", "#ff7f00", "#ff0000"],
)


def infer_metric_from_name(path: Path) -> str:
    name = path.stem
    for metric in METRIC_TO_COLUMN:
        if re.search(rf"_{re.escape(metric)}_", name) or name.endswith(f"_{metric}"):
            return metric
    return "observed"


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

    valid = (
        np.isfinite(ks_flat)
        & (ks_flat > 0.0)
        & np.isfinite(cs_flat)
        & (cs_flat > 0.0)
        & np.isfinite(loss_flat)
        & (loss_flat > 0.0)
    )
    if not np.any(valid):
        raise RuntimeError(f"No finite positive ks/cs/loss rows in {csv_path}")

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
    grid[i, j] = np.log10(loss_flat)
    return ks_values, cs_values, grid


def decade_ticks(values: np.ndarray) -> tuple[list[float], list[str]]:
    lo = int(np.floor(np.log10(np.nanmin(values))))
    hi = int(np.ceil(np.log10(np.nanmax(values))))
    ticks = [float(v) for v in range(lo, hi + 1)]
    labels = [rf"$10^{{{v}}}$" for v in range(lo, hi + 1)]
    return ticks, labels


def z_axis_label(metric: str) -> str:
    if metric == "observed":
        return r"$\log_{10}(\mathcal{L})$"
    z_label = z_axis_label(metric)
    return f"log10({metric_label})"


def render_static_3d(
    *,
    csv_path: Path,
    png_path: Path,
    pdf_path: Path | None,
    metric: str,
    dpi: int,
    surface_count: int,
    z_floor_log10: float | None,
    alpha: float,
    bottom_contours: bool,
    contour_levels: int,
    contour_color: str,
    contour_linewidth: float,
) -> None:
    ks_values, cs_values, z = load_grid(csv_path, metric)
    if z_floor_log10 is not None:
        z = np.maximum(z, float(z_floor_log10))
    x_log = np.log10(ks_values)
    y_log = np.log10(cs_values)
    xx, yy = np.meshgrid(x_log, y_log, indexing="ij")
    z_label = z_axis_label(metric)

    fig = plt.figure(figsize=(13.5, 8.8))
    ax = fig.add_axes([0.10, 0.06, 0.66, 0.90], projection="3d")
    surf = ax.plot_surface(
        xx,
        yy,
        z,
        cmap=RAINBOW_PURPLE_LOW_RED_HIGH,
        rcount=max(2, int(surface_count)),
        ccount=max(2, int(surface_count)),
        linewidth=0.0,
        antialiased=True,
        shade=False,
        alpha=float(alpha),
        vmin=float(z_floor_log10) if z_floor_log10 is not None else None,
        vmax=float(np.nanmax(z)),
    )
    if bottom_contours:
        finite_z = z[np.isfinite(z)]
        if finite_z.size > 0:
            z_offset = float(z_floor_log10) if z_floor_log10 is not None else float(np.nanmin(finite_z))
            lo = float(np.nanmin(finite_z))
            hi = float(np.nanmax(finite_z))
            if hi > lo:
                levels = np.linspace(lo, hi, max(2, int(contour_levels)))
                ax.contour(
                    xx,
                    yy,
                    z,
                    levels=levels,
                    zdir="z",
                    offset=z_offset,
                    colors=str(contour_color),
                    linewidths=float(contour_linewidth),
                    alpha=0.82,
                )

    x_ticks, x_ticklabels = decade_ticks(ks_values)
    y_ticks, y_ticklabels = decade_ticks(cs_values)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)
    ax.set_xlabel("ks [N/m]", labelpad=14, fontsize=20)
    ax.set_ylabel("cs [N·s/m]", labelpad=14, fontsize=20)
    ax.set_zlabel("")
    fig.text(0.035, 0.50, z_label, rotation=90, fontsize=20, va="center", ha="center")
    ax.tick_params(axis="both", which="major", labelsize=18, pad=7)
    ax.tick_params(axis="z", which="major", labelsize=18, pad=7)
    if z_floor_log10 is not None:
        ax.set_zlim(float(z_floor_log10), float(np.nanmax(z)))
    ax.view_init(elev=30, azim=-135)
    ax.set_box_aspect((5.2, 5.2, 3.0))

    cax = fig.add_axes([0.83, 0.18, 0.025, 0.66])
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label(z_label, fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.08)
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metric", choices=sorted(METRIC_TO_COLUMN), default="")
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--surface-count", type=int, default=300)
    parser.add_argument(
        "--z-floor-log10",
        type=float,
        default=-15.0,
        help="Clamp plotted z=log10(loss) to this lower bound. Use nan to disable.",
    )
    parser.add_argument("--alpha", type=float, default=0.78, help="Surface transparency; 1.0 is opaque.")
    parser.add_argument("--bottom-contours", action="store_true", help="Project log-loss contour lines onto the bottom z plane.")
    parser.add_argument("--contour-levels", type=int, default=12)
    parser.add_argument("--contour-color", default="black")
    parser.add_argument("--contour-linewidth", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    metric = args.metric.strip() or infer_metric_from_name(csv_path)
    png_path = args.png.resolve() if args.png is not None else csv_path.with_name(f"{csv_path.stem}_3d.png")
    pdf_path = args.pdf.resolve() if args.pdf is not None else None
    render_static_3d(
        csv_path=csv_path,
        png_path=png_path,
        pdf_path=pdf_path,
        metric=metric,
        dpi=args.dpi,
        surface_count=args.surface_count,
        z_floor_log10=None if not np.isfinite(args.z_floor_log10) else float(args.z_floor_log10),
        alpha=args.alpha,
        bottom_contours=bool(args.bottom_contours),
        contour_levels=int(args.contour_levels),
        contour_color=str(args.contour_color),
        contour_linewidth=float(args.contour_linewidth),
    )
    print(f"Saved PNG: {png_path}")
    if pdf_path is not None:
        print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
