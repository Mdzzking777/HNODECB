"""Render a 3x3 DMT-KV ks/cs loss landscape summary.

Rows: full simulation time span, transient window, stable window.
Columns: 3D log10 loss, 2D log10 loss, 2D raw loss.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
CASES = [
    (
        "full simulation time span",
        THIS_DIR / "Full time span" / "dmt_kv_ks_cs_loss_surface_full_time_span_observed_600x600.csv",
    ),
    (
        "transient window",
        THIS_DIR / "W0" / "dmt_kv_ks_cs_loss_surface_first_contact_observed_600x600.csv",
    ),
    (
        "stable window",
        THIS_DIR / "W1" / "dmt_kv_ks_cs_loss_surface_w1_observed_600x600.csv",
    ),
]

CMAP = LinearSegmentedColormap.from_list(
    "rainbow_purple_low_red_high",
    ["#4b0082", "#0000ff", "#00ffff", "#00aa00", "#ffff00", "#ff7f00", "#ff0000"],
)

LOG_FLOOR = -15.0
CONTOUR_LEVELS = 30
FONT_SCALE = 1.5


def load_loss_grid(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise RuntimeError(f"No rows loaded from {csv_path}")

    ks_flat = np.asarray(data["ks"], dtype=float)
    cs_flat = np.asarray(data["cs"], dtype=float)
    loss_flat = np.asarray(data["loss_observed"], dtype=float)
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
    grid[i, j] = loss_flat
    return ks_values, cs_values, grid


def decade_ticks(values: np.ndarray) -> tuple[list[float], list[str]]:
    lo = int(np.floor(np.log10(np.nanmin(values))))
    hi = int(np.ceil(np.log10(np.nanmax(values))))
    ticks = [float(v) for v in range(lo, hi + 1)]
    labels = [rf"$10^{{{v}}}$" for v in range(lo, hi + 1)]
    return ticks, labels


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("No finite values available")
    return float(np.nanpercentile(finite, float(percentile)))


def add_colorbar(
    fig: plt.Figure,
    mappable,
    ax,
    label: str,
    *,
    raw: bool = False,
    font_scale: float = FONT_SCALE,
) -> None:
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.035, shrink=0.78)
    cbar.set_label(label, fontsize=12 * font_scale)
    cbar.ax.tick_params(labelsize=10 * font_scale)
    if raw:
        cbar.formatter = mticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()


def style_2d_axis(
    ax,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    font_scale: float = FONT_SCALE,
) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=10 * font_scale, pad=2)
    if show_xlabel:
        ax.set_xlabel("ks [N/m]", fontsize=12 * font_scale)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("cs [N·s/m]", fontsize=12 * font_scale)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)


def render() -> None:
    fig = plt.figure(figsize=(26.0, 20.5), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        3,
        left=0.06,
        right=0.985,
        bottom=0.055,
        top=0.935,
        wspace=0.30,
        hspace=0.24,
        width_ratios=[1.08, 1.0, 1.0],
    )

    column_labels = [
        r"3D landscape $\log_{10}(\mathcal{L})$",
        r"2D landscape $\log_{10}(\mathcal{L})$",
        r"2D landscape $\mathcal{L}$",
    ]

    for row_idx, (row_label, csv_path) in enumerate(CASES):
        ks_values, cs_values, loss = load_loss_grid(csv_path)
        positive_loss = np.where(loss > 0.0, loss, np.nan)
        log_loss = np.log10(positive_loss)
        log_top = float(np.nanmax(log_loss[np.isfinite(log_loss)]))
        log_plot = np.clip(log_loss, LOG_FLOOR, log_top)
        log_levels = np.linspace(LOG_FLOOR, log_top, CONTOUR_LEVELS)

        raw_vmin = float(np.nanmin(positive_loss[np.isfinite(positive_loss)]))
        raw_vmax = finite_percentile(positive_loss, 99.5)
        if not np.isfinite(raw_vmax) or raw_vmax <= raw_vmin:
            raw_vmax = float(np.nanmax(positive_loss[np.isfinite(positive_loss)]))
        raw_levels = np.linspace(raw_vmin, raw_vmax, CONTOUR_LEVELS)

        kk, cc = np.meshgrid(ks_values, cs_values, indexing="ij")
        x_log = np.log10(ks_values)
        y_log = np.log10(cs_values)
        xx, yy = np.meshgrid(x_log, y_log, indexing="ij")

        ax3d = fig.add_subplot(gs[row_idx, 0], projection="3d")
        surf = ax3d.plot_surface(
            xx,
            yy,
            log_plot,
            cmap=CMAP,
            rcount=220,
            ccount=220,
            linewidth=0.0,
            antialiased=True,
            shade=False,
            alpha=0.78,
            vmin=LOG_FLOOR,
            vmax=log_top,
        )
        ax3d.contour(
            xx,
            yy,
            log_plot,
            levels=log_levels,
            zdir="z",
            offset=LOG_FLOOR,
            colors="black",
            linewidths=0.55,
            alpha=0.82,
        )
        x_ticks, x_ticklabels = decade_ticks(ks_values)
        y_ticks, y_ticklabels = decade_ticks(cs_values)
        ax3d.set_xticks(x_ticks)
        ax3d.set_xticklabels(x_ticklabels)
        ax3d.set_yticks(y_ticks)
        ax3d.set_yticklabels(y_ticklabels)
        ax3d.set_zlim(LOG_FLOOR, log_top)
        ax3d.view_init(elev=30, azim=-135)
        ax3d.set_box_aspect((5.2, 5.2, 3.0))
        ax3d.tick_params(axis="both", which="major", labelsize=9 * FONT_SCALE, pad=1)
        ax3d.tick_params(axis="z", which="major", labelsize=9 * FONT_SCALE, pad=1)
        ax3d.set_xlabel("ks [N/m]", fontsize=10 * FONT_SCALE, labelpad=4)
        ax3d.set_ylabel("cs [N·s/m]", fontsize=10 * FONT_SCALE, labelpad=4)
        ax3d.set_zlabel("")
        if row_idx == 0:
            ax3d.set_title(column_labels[0], fontsize=14 * FONT_SCALE, pad=8)
        add_colorbar(fig, surf, ax3d, r"$\log_{10}(\mathcal{L})$")

        ax_log = fig.add_subplot(gs[row_idx, 1])
        mesh_log = ax_log.pcolormesh(
            kk,
            cc,
            log_plot,
            shading="auto",
            cmap=CMAP,
            vmin=LOG_FLOOR,
            vmax=log_top,
        )
        ax_log.contour(kk, cc, log_plot, levels=log_levels, colors="black", linewidths=0.55, alpha=0.86)
        style_2d_axis(ax_log, show_xlabel=row_idx == 2, show_ylabel=False)
        if row_idx == 0:
            ax_log.set_title(column_labels[1], fontsize=14 * FONT_SCALE, pad=8)
        add_colorbar(fig, mesh_log, ax_log, r"$\log_{10}(\mathcal{L})$")

        ax_raw = fig.add_subplot(gs[row_idx, 2])
        raw_plot = np.minimum(positive_loss, raw_vmax)
        mesh_raw = ax_raw.pcolormesh(
            kk,
            cc,
            raw_plot,
            shading="auto",
            cmap=CMAP,
            vmin=raw_vmin,
            vmax=raw_vmax,
        )
        ax_raw.contour(kk, cc, raw_plot, levels=raw_levels, colors="black", linewidths=0.55, alpha=0.86)
        style_2d_axis(ax_raw, show_xlabel=row_idx == 2, show_ylabel=False)
        if row_idx == 0:
            ax_raw.set_title(column_labels[2], fontsize=14 * FONT_SCALE, pad=8)
        add_colorbar(fig, mesh_raw, ax_raw, r"$\mathcal{L}$", raw=True)

        fig.text(
            0.025,
            0.795 - row_idx * 0.294,
            row_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=15 * FONT_SCALE,
        )

    out_png = THIS_DIR / "dmt_kv_ks_cs_loss_landscape_3x3_summary.png"
    out_pdf = THIS_DIR / "dmt_kv_ks_cs_loss_landscape_3x3_summary.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    render()
