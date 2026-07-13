"""Render a 2x2 DMT-KV ks/cs log-loss landscape summary.

Rows: transient window, stable window.
Columns: 3D log10 loss, 2D log10 loss.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_dmt_kv_ks_cs_loss_landscape_3x3_summary import (
    CMAP,
    CONTOUR_LEVELS,
    FONT_SCALE,
    LOG_FLOOR,
    THIS_DIR,
    add_colorbar,
    load_loss_grid,
    style_2d_axis,
)


CASES = [
    (
        "transient window",
        THIS_DIR / "W0" / "dmt_kv_ks_cs_loss_surface_first_contact_observed_600x600.csv",
    ),
    (
        "stable window",
        THIS_DIR / "W1" / "dmt_kv_ks_cs_loss_surface_w1_observed_600x600.csv",
    ),
]

FIG_FONT_SCALE = 2.0 * FONT_SCALE


def interior_decade_ticks(values: np.ndarray) -> tuple[list[float], list[str]]:
    lo = int(np.ceil(np.log10(np.nanmin(values))))
    hi = int(np.floor(np.log10(np.nanmax(values))))
    ticks = [float(value) for value in range(lo, hi + 1)]
    labels = [rf"$10^{{{value}}}$" for value in range(lo, hi + 1)]
    return ticks, labels


def render() -> None:
    fig = plt.figure(figsize=(24.0, 18.8), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.10,
        right=0.975,
        bottom=0.08,
        top=0.90,
        wspace=0.44,
        hspace=0.30,
        width_ratios=[1.24, 1.0],
    )

    column_labels = [
        r"3D landscape $\log_{10}(\mathcal{L})$",
        r"2D landscape $\log_{10}(\mathcal{L})$",
    ]

    row_y_positions = [0.705, 0.295]

    for row_idx, (row_label, csv_path) in enumerate(CASES):
        ks_values, cs_values, loss = load_loss_grid(csv_path)
        positive_loss = np.where(loss > 0.0, loss, np.nan)
        log_loss = np.log10(positive_loss)
        log_top = float(np.nanmax(log_loss[np.isfinite(log_loss)]))
        log_plot = np.clip(log_loss, LOG_FLOOR, log_top)
        log_levels = np.linspace(LOG_FLOOR, log_top, CONTOUR_LEVELS)

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
        x_ticks, x_ticklabels = interior_decade_ticks(ks_values)
        y_ticks, y_ticklabels = interior_decade_ticks(cs_values)
        ax3d.set_xticks(x_ticks)
        ax3d.set_xticklabels(x_ticklabels)
        ax3d.set_yticks(y_ticks)
        ax3d.set_yticklabels(y_ticklabels)
        ax3d.set_zlim(LOG_FLOOR, log_top)
        ax3d.view_init(elev=30, azim=-135)
        ax3d.set_box_aspect((5.2, 5.2, 3.0))
        ax3d.tick_params(axis="both", which="major", labelsize=9 * FIG_FONT_SCALE, pad=1)
        ax3d.tick_params(axis="z", which="major", labelsize=9 * FIG_FONT_SCALE, pad=10)
        ax3d.set_xlabel("ks [N/m]", fontsize=10 * FIG_FONT_SCALE, labelpad=28)
        ax3d.set_ylabel("cs [N·s/m]", fontsize=10 * FIG_FONT_SCALE, labelpad=32)
        ax3d.set_zlabel("")
        if row_idx == 0:
            ax3d.set_title(column_labels[0], fontsize=14 * FIG_FONT_SCALE, pad=12)
        add_colorbar(fig, surf, ax3d, r"$\log_{10}(\mathcal{L})$", font_scale=FIG_FONT_SCALE)

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
        style_2d_axis(ax_log, show_xlabel=True, show_ylabel=True, font_scale=FIG_FONT_SCALE)
        if row_idx == 0:
            ax_log.set_title(column_labels[1], fontsize=14 * FIG_FONT_SCALE, pad=12)
        add_colorbar(fig, mesh_log, ax_log, r"$\log_{10}(\mathcal{L})$", font_scale=FIG_FONT_SCALE)

        fig.text(
            0.03,
            row_y_positions[row_idx],
            row_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=15 * FIG_FONT_SCALE,
        )

    out_png = THIS_DIR / "dmt_kv_ks_cs_loss_landscape_2x2_transient_stable_log_summary.png"
    out_pdf = THIS_DIR / "dmt_kv_ks_cs_loss_landscape_2x2_transient_stable_log_summary.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    render()
