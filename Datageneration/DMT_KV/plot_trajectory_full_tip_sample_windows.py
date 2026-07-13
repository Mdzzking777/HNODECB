"""Plot AFM04 DMT-KV full trajectory with transient/stable windows.

This is a focused two-panel variant of plots/trajectory_full.png:
- tip displacement
- sample motion

The original W0/W1 window construction is preserved, but the labels are
rendered as "transient window" and "stable window" for presentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM04.stage1pluslight.windows import window_manifests


WINDOW_US = 6.288e-6


def _scale_figure_text(axes: np.ndarray, factor: float = 2.0) -> None:
    tick_size = axes.ravel()[0].xaxis.label.get_fontsize() * factor
    for ax in axes.ravel():
        ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * factor)
        ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * factor)
        ax.tick_params(axis="both", which="both", labelsize=tick_size)
        ax.xaxis.offsetText.set_fontsize(ax.xaxis.offsetText.get_fontsize() * factor)
        ax.yaxis.offsetText.set_fontsize(ax.yaxis.offsetText.get_fontsize() * factor)


def _training_windows(times: np.ndarray, contact_mask: np.ndarray, x1_signal: np.ndarray):
    transient = window_manifests(times, contact_mask, "stage2_w0", WINDOW_US, x1_signal=x1_signal)[0]
    stable = window_manifests(times, contact_mask, "stage2_w1", WINDOW_US, x1_signal=x1_signal)[0]
    return [
        ("transient window", transient, "tab:orange"),
        ("stable window", stable, "tab:purple"),
    ]


def _annotate_windows(axes: np.ndarray, windows) -> None:
    for label, win, color in windows:
        start_us = 1.0e6 * float(win.t_start)
        stop_us = 1.0e6 * float(win.t_stop)
        for ax in axes.ravel():
            ax.axvspan(
                start_us,
                stop_us,
                facecolor=color,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.16,
                zorder=0,
            )
            ax.axvline(start_us, color=color, linewidth=1.0, alpha=0.9)
            ax.axvline(stop_us, color=color, linewidth=1.0, alpha=0.9)
        axes[0].text(
            0.5 * (start_us + stop_us),
            0.95,
            label,
            color=color,
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
            transform=axes[0].get_xaxis_transform(),
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.78, "pad": 2.5},
        )


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "trajectory.csv"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = np.asarray(data["time_s"], dtype=float)
    x1 = np.asarray(data["x_tip_m"], dtype=float)
    x3 = np.asarray(data["y_sample_m"], dtype=float)
    contact = np.asarray(data["contact_status"], dtype=bool)

    t_us = t * 1.0e6
    x1_nm = x1 * 1.0e9
    x3_nm = x3 * 1.0e9

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    axes[0].plot(t_us, x1_nm, color="tab:blue", linewidth=0.5)
    axes[0].set_ylabel("Tip displacement [nm]")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.0, color="k", linestyle="--", linewidth=0.5)

    axes[1].plot(t_us, x3_nm, color="tab:red", linewidth=0.5)
    axes[1].set_ylabel("Sample motion [nm]")
    axes[1].set_xlabel(r"Time [$\mu$s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.0, color="k", linestyle="--", linewidth=0.5)

    _annotate_windows(axes, _training_windows(t, contact, x1))
    _scale_figure_text(axes, factor=2.0)

    fig.tight_layout(pad=2.0)

    out_png = plots_dir / "trajectory_full_tip_sample_transient_stable.png"
    out_pdf = plots_dir / "trajectory_full_tip_sample_transient_stable.pdf"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
