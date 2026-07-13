"""Plot AFM04 W0/W1 x1/x2/x3 trajectory windows from trajectory.csv."""

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
SUPTITLE_FONTSIZE = 32
AX_TITLE_FONTSIZE = 28
AX_LABEL_FONTSIZE = 25
TICK_LABEL_FONTSIZE = 21


def load_trajectory(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing trajectory CSV: {path}")
    return np.genfromtxt(path, delimiter=",", names=True)


def afm04_window_slice(data: np.ndarray, window_mode: str, window_name: str) -> tuple[slice, object]:
    times = np.asarray(data["time_s"], dtype=float)
    contact = np.asarray(data["contact_status"], dtype=float) > 0.5
    x1 = np.asarray(data["x_tip_m"], dtype=float)
    manifests = window_manifests(times, contact, window_mode, WINDOW_US, x1_signal=x1)
    if len(manifests) != 1:
        raise RuntimeError(f"AFM04 {window_name} expected one manifest, got {len(manifests)}")
    win = manifests[0]
    return slice(int(win.start_idx), int(win.stop_idx) + 1), win


def save_plot(
    data: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    *,
    window_mode: str,
    window_name: str,
) -> None:
    sl, _win = afm04_window_slice(data, window_mode, window_name)
    t_us = np.asarray(data["time_s"], dtype=float)[sl] * 1.0e6

    panels = [
        ("x1 trajectory", np.asarray(data["x_tip_m"], dtype=float)[sl] * 1.0e9, "x1 [nm]", "tab:blue", True),
        ("x2 trajectory", np.asarray(data["x1dot_velocity"], dtype=float)[sl] * 1.0e6, "x2 [um/s]", "tab:cyan", True),
        ("x3 trajectory", np.asarray(data["y_sample_m"], dtype=float)[sl] * 1.0e9, "x3 [nm]", "tab:green", True),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(15.5, 11.5), sharex=True)
    for ax, (title, values, ylabel, color, show_curve) in zip(axes, panels):
        if show_curve:
            ax.plot(t_us, values, color=color, linewidth=1.35)
        if t_us.size and values.size and np.isfinite(t_us[0]) and np.isfinite(values[0]):
            ax.scatter(
                t_us[0],
                values[0],
                s=44,
                color="red",
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
            )
        ax.axhline(0.0, color="black", linewidth=0.65, alpha=0.65)
        ax.set_title(title, fontsize=AX_TITLE_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AX_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    axes[-1].set_xlabel("time [us]", fontsize=AX_LABEL_FONTSIZE)

    fig.suptitle(f"Training {window_name} sliced from full-time span", fontsize=SUPTITLE_FONTSIZE)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    plots_dir = root / "plots"
    data = load_trajectory(root / "trajectory.csv")
    outputs = [
        (
            plots_dir / "afm04_w0_state_force_six_panel.png",
            plots_dir / "afm04_w0_state_force_six_panel.pdf",
            "stage2_w0",
            "W0",
        ),
        (
            plots_dir / "afm04_w1_state_force_six_panel.png",
            plots_dir / "afm04_w1_state_force_six_panel.pdf",
            "stage2_w1",
            "W1",
        ),
    ]
    for out_png, out_pdf, window_mode, window_name in outputs:
        save_plot(
            data,
            out_png,
            out_pdf,
            window_mode=window_mode,
            window_name=window_name,
        )
        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
