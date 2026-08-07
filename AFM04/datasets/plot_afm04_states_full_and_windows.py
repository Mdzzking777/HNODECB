"""Plot the AFM04 state data over the full horizon and formal W0-W3 windows."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM04.datasets.non_perturbed_dataset_generator import load_table_npz
from AFM04.stage1pluslight.config import default_config
from AFM04.stage1pluslight.windows import (
    WindowManifest,
    first_contact_window_indices,
    stage2_window_manifest,
)


DATA_PATH = Path(__file__).resolve().parent / "e0.0" / "data" / "pert_df_afm_dmt_kv.npz"
OUTPUT_PATH = Path(__file__).resolve().parent / "afm04_x1_x2_x2dot_x3_full_and_windows.png"


def _w0_manifest(
    times: np.ndarray,
    contact: np.ndarray,
    window_span: float,
) -> WindowManifest:
    idxs = first_contact_window_indices(times, contact, window_span)
    return WindowManifest(
        role="first_contact",
        label="first_contact_window",
        start_idx=int(idxs[0]),
        stop_idx=int(idxs[-1]),
        length=int(idxs.size),
        t_start=float(times[idxs[0]]),
        t_stop=float(times[idxs[-1]]),
        idxs=idxs,
    )


def _window_definitions(
    times: np.ndarray,
    contact: np.ndarray,
    x1: np.ndarray,
) -> list[tuple[str, str, WindowManifest]]:
    config = default_config(REPO_ROOT)
    w0 = _w0_manifest(times, contact, config.arch_window_us)
    stage2_windows = stage2_window_manifest(times, contact, x1)
    role_labels = {
        "middle": ("W1", "middle"),
        "max_x1_pp_change": ("W2", "maximum amplitude change"),
        "tail_stable": ("W3", "tail stable"),
    }
    windows: list[tuple[str, str, WindowManifest]] = [
        ("W0", "first contact", w0),
    ]
    for window in stage2_windows:
        label, description = role_labels[window.role]
        windows.append((label, description, window))
    return windows


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.75)
    ax.tick_params(axis="both", labelsize=9, direction="out", length=3.5, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"AFM04 dataset not found: {DATA_PATH}")

    table = load_table_npz(str(DATA_PATH))
    times = np.asarray(table["t"], dtype=float)
    contact = np.asarray(table["contact"], dtype=bool)

    series = [
        (r"$x_1$ [nm]", np.asarray(table["x1"], dtype=float) * 1.0e9, "#1f77b4"),
        (r"$x_2$ [m s$^{-1}$]", np.asarray(table["x2"], dtype=float), "#d62728"),
        (r"$\dot{x}_2$ [m s$^{-2}$]", np.asarray(table["x2dot"], dtype=float), "#2ca02c"),
        (r"$x_3$ [nm]", np.asarray(table["x3"], dtype=float) * 1.0e9, "#9467bd"),
    ]
    windows = _window_definitions(times, contact, np.asarray(table["x1"], dtype=float))

    columns: list[tuple[str, str, np.ndarray | None]] = [
        ("Full time span", f"{times[0] * 1.0e3:.3f}-{times[-1] * 1.0e3:.3f} ms", None)
    ]
    for label, description, window in windows:
        interval = f"{window.t_start * 1.0e6:.3f}-{window.t_stop * 1.0e6:.3f} us"
        columns.append((f"{label}: {description}", interval, np.asarray(window.idxs, dtype=int)))

    fig, axes = plt.subplots(
        nrows=4,
        ncols=len(columns),
        figsize=(21.0, 11.2),
        gridspec_kw={"width_ratios": [1.32, 1.0, 1.0, 1.0, 1.0]},
        squeeze=False,
    )

    window_colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for col, (title, interval, idxs) in enumerate(columns):
        axes[0, col].set_title(f"{title}\n{interval}", fontsize=11.5, pad=8)
        if idxs is None:
            t_plot = times * 1.0e3
            x_label = "Time [ms]"
        else:
            t_plot = times[idxs] * 1.0e6
            x_label = "Time [us]"

        for row, (y_label, values, color) in enumerate(series):
            ax = axes[row, col]
            y_plot = values if idxs is None else values[idxs]
            ax.plot(t_plot, y_plot, color=color, linewidth=0.82 if idxs is None else 1.05)
            ax.set_xlim(float(t_plot[0]), float(t_plot[-1]))
            _style_axis(ax)

            if row == len(series) - 1:
                ax.set_xlabel(x_label, fontsize=10.5)
            if col == 0:
                ax.set_ylabel(y_label, fontsize=11.5)

            if row in (1, 2):
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 3))
                ax.yaxis.set_major_formatter(formatter)
                ax.yaxis.get_offset_text().set_fontsize(8.5)

    # Mark the four formal windows in the full-horizon column.
    for row in range(len(series)):
        full_ax = axes[row, 0]
        for color, (_, _, window) in zip(window_colors, windows):
            full_ax.axvspan(
                window.t_start * 1.0e3,
                window.t_stop * 1.0e3,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.18,
                zorder=0,
            )

    top_full = axes[0, 0]
    y_lo, y_hi = top_full.get_ylim()
    label_levels = [0.92, 0.80, 0.92, 0.80]
    for color, level, (label, _, window) in zip(window_colors, label_levels, windows):
        x_mid = 0.5 * (window.t_start + window.t_stop) * 1.0e3
        top_full.text(
            x_mid,
            y_lo + level * (y_hi - y_lo),
            label,
            color=color,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            clip_on=False,
        )

    fig.subplots_adjust(left=0.065, right=0.992, top=0.92, bottom=0.075, wspace=0.28, hspace=0.24)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Source: {DATA_PATH}")
    for label, description, window in windows:
        print(
            f"{label} ({description}): "
            f"{window.t_start * 1.0e6:.3f}-{window.t_stop * 1.0e6:.3f} us, "
            f"{window.length} points"
        )


if __name__ == "__main__":
    main()
