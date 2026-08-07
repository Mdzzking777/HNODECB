from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ALIGNMENT_NPZ = SCRIPT_DIR / "middle_gp_template_to_400_800_alignment.npz"
OUTPUT_PNG = SCRIPT_DIR / "middle_gp_template_to_400_800_x2_raw_vs_aligned_background.png"
ZOOM_WINDOWS_US = (
    (410.0, 430.0),
    (590.0, 610.0),
    (770.0, 790.0),
)


def plot_panel(ax, time_us, x2_raw, x2_bg, transition_idx, xlim=None, show_legend=False):
    ax.plot(
        time_us,
        x2_raw,
        color="black",
        linewidth=0.45,
        alpha=0.86,
        label=r"$x_2^{raw}$",
        zorder=2,
    )
    ax.plot(
        time_us,
        x2_bg,
        color="#1f77b4",
        linewidth=1.15,
        label="aligned middle-GP background",
        zorder=3,
    )
    for idx in transition_idx:
        t_transition = time_us[int(idx)]
        if xlim is None or (xlim[0] <= t_transition <= xlim[1]):
            ax.axvline(t_transition, color="0.4", linewidth=0.25, alpha=0.13, zorder=1)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.23)
    ax.margins(x=0.0)
    if show_legend:
        ax.legend(loc="upper right")


def main() -> int:
    with np.load(ALIGNMENT_NPZ, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        x2_raw = np.asarray(data["x2_raw_m_s"], dtype=float)
        x2_bg = np.asarray(data["x2_background_aligned_m_s"], dtype=float)
        transition_idx = np.asarray(data["transition_idx"], dtype=int)

    time_us = time_s * 1.0e6

    fig, axes = plt.subplots(4, 1, figsize=(13.5, 11.2), constrained_layout=True)
    plot_panel(axes[0], time_us, x2_raw, x2_bg, transition_idx, show_legend=True)
    axes[0].set_title(r"400--800 $\mu$s")

    for ax, window in zip(axes[1:], ZOOM_WINDOWS_US):
        plot_panel(ax, time_us, x2_raw, x2_bg, transition_idx, xlim=window)
        ax.set_title(rf"{window[0]:.0f}--{window[1]:.0f} $\mu$s")

    for ax in axes:
        ax.set_ylabel(r"$x_2$ [m s$^{-1}$]")
    axes[-1].set_xlabel(r"Time [$\mu$s]")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
