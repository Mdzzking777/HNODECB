"""Plot AFM05 pixel_184_152 x1, x2, and x2dot traces.

Outputs:
  - full time span
  - current stage1pluslight window_05_initial sampled training window
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "AFM05" / "DATAgeneration"
SOURCE_NPZ = DATA_DIR / "PS_cantilever_disp_vel_time_3_pixels_Z_85.0_a0_0.07108_file_scan04143.imp_.npz"
OUT_DIR = DATA_DIR / "plot"
THESIS_OUT_DIR = OUT_DIR / "for thesis"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else int(raw)


def _load_pixel_trace(pixel_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(SOURCE_NPZ, allow_pickle=True) as data:
        pixel = np.asarray(data[pixel_key], dtype=float)
        meta = dict(data["add_data_dict"].item())
    if pixel.ndim != 2 or pixel.shape[1] < 4:
        raise ValueError(f"{pixel_key} must be [x1_m, x2_m_per_s, x2dot_m_per_s2, t_s].")
    x1 = pixel[:, 0]
    x2 = pixel[:, 1]
    x2dot = pixel[:, 2]
    t = pixel[:, 3]
    return x1, x2, x2dot, t, meta


def _current_window_indices(t: np.ndarray, meta: dict, pixel_tag: str) -> np.ndarray:
    del pixel_tag
    window_start_s = float(meta["AFM05_first_contact_time_s"])
    start_idx = int(np.searchsorted(t, window_start_s, side="left"))
    if start_idx >= t.size:
        raise ValueError("The AFM05 training-window start is outside the stored trace.")

    span_s = _env_float("HNODECB_AFM05_STAGE1PLUS_ARCH_WINDOW_US", 25.152e-6)
    stride = max(1, _env_int("HNODECB_AFM05_STAGE1PLUS_WINDOW_SAMPLE_STRIDE", 16))
    local = np.flatnonzero((t[start_idx:] - float(t[start_idx])) <= span_s)
    if local.size == 0:
        return np.array([start_idx], dtype=int)
    return (local + start_idx)[::stride]


def _plot_three_panel(
    t: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x2dot: np.ndarray,
    out_path: Path,
    *,
    time_scale: float,
    time_label: str,
    title: str,
    train_marker_mask: np.ndarray | None = None,
    val_marker_mask: np.ndarray | None = None,
    window_box_range: tuple[float, float] | None = None,
    first_contact_x: float | None = None,
    absolute_time: bool = False,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.8), sharex=True)
    traces = (
        (x1 * 1.0e9, r"$x_1$ [nm]", "#1f77b4"),
        (x2, r"$x_2$ [m/s]", "#2ca02c"),
        (x2dot, r"$\dot{x}_2$ [m/s$^2$]", "#d62728"),
    )
    tx = t * time_scale if absolute_time else (t - float(t[0])) * time_scale
    if train_marker_mask is None:
        train_marker_mask = np.zeros_like(tx, dtype=bool)
    else:
        train_marker_mask = np.asarray(train_marker_mask, dtype=bool)
        if train_marker_mask.shape != tx.shape:
            raise ValueError("train_marker_mask must match the plotted time vector length")
    if val_marker_mask is None:
        val_marker_mask = np.zeros_like(tx, dtype=bool)
    else:
        val_marker_mask = np.asarray(val_marker_mask, dtype=bool)
        if val_marker_mask.shape != tx.shape:
            raise ValueError("val_marker_mask must match the plotted time vector length")
    for ax, (values, ylabel, line_color) in zip(axes, traces):
        ax.plot(tx, values, color=line_color, linewidth=1.1)
        if np.any(train_marker_mask):
            ax.scatter(tx[train_marker_mask], values[train_marker_mask], s=6, c="#1f77b4", marker="o", zorder=5)
        if np.any(val_marker_mask):
            ax.scatter(tx[val_marker_mask], values[val_marker_mask], s=6, c="#d62728", marker="o", zorder=6)
        if window_box_range is not None:
            x0, x1_box = window_box_range
            y0, y1 = ax.get_ylim()
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1_box - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.4,
                    zorder=7,
                    clip_on=False,
                )
            )
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=10)
        if first_contact_x is not None:
            ax.axvline(first_contact_x, color="black", linestyle="--", linewidth=1.2, zorder=8)
    if first_contact_x is not None:
        axes[0].text(
            first_contact_x,
            0.97,
            "first contact",
            transform=axes[0].get_xaxis_transform(),
            fontsize=11,
            color="black",
            va="top",
            ha="left",
        )
    axes[0].set_title(title, fontsize=13, pad=10)
    axes[-1].set_xlabel(time_label, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_full_timespan_for_thesis(
    t: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    first_contact_time_s: float,
    window_start_s: float,
    window_stop_s: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True)
    time_ms = (t - float(t[0])) * 1.0e3
    first_contact_ms = (first_contact_time_s - float(t[0])) * 1.0e3
    window_start_ms = (window_start_s - float(t[0])) * 1.0e3
    window_stop_ms = (window_stop_s - float(t[0])) * 1.0e3
    traces = (
        (x1 * 1.0e9, r"$x_1$ [nm]", "#1f77b4"),
        (x2, r"$x_2$ [m/s]", "#2ca02c"),
    )
    for axis, (values, ylabel, color) in zip(axes, traces):
        axis.plot(time_ms, values, color=color, linewidth=1.1)
        axis.axvline(
            first_contact_ms,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            zorder=9,
        )
        axis.set_ylabel(ylabel, fontsize=18)
        axis.grid(True, alpha=0.25, linewidth=0.8)
        axis.tick_params(axis="both", labelsize=15)
        y0, y1 = axis.get_ylim()
        axis.add_patch(
            Rectangle(
                (window_start_ms, y0),
                window_stop_ms - window_start_ms,
                y1 - y0,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                zorder=7,
            )
        )
    axes[0].text(
        first_contact_ms + 0.02,
        0.96,
        "first contact",
        transform=axes[0].get_xaxis_transform(),
        fontsize=16.5,
        color="#d62728",
        va="top",
        ha="left",
        zorder=10,
    )
    axes[0].text(
        window_stop_ms + 0.04,
        0.82,
        "window for training & validation",
        transform=axes[0].get_xaxis_transform(),
        fontsize=14,
        color="black",
        va="top",
        ha="left",
    )
    axes[-1].set_xlabel("time [ms]", fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_initial_window_two_panel(
    t: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    out_path: Path,
    *,
    font_multiplier: float = 1.0,
    time_label: str = r"time [$\mu$s]",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0))
    time_us = t * 1.0e6
    traces = (
        (x1 * 1.0e9, r"$x_1$ [nm]", "#1f77b4"),
        (x2, r"$x_2$ [m/s]", "#2ca02c"),
    )
    for axis, (values, ylabel, color) in zip(axes, traces):
        axis.plot(time_us, values, color=color, linewidth=1.1)
        axis.set_ylabel(ylabel, fontsize=18 * font_multiplier)
        axis.set_xlabel(time_label, fontsize=18 * font_multiplier)
        axis.tick_params(axis="both", labelsize=15 * font_multiplier)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(True, alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    pixel_tag = os.environ.get("HNODECB_AFM05_STAGE1PLUS_PIXEL_TAG", "184_152").strip() or "184_152"
    pixel_key = f"pixel_{pixel_tag}"
    x1, x2, x2dot, t, meta = _load_pixel_trace(pixel_key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idxs = _current_window_indices(t, meta, pixel_tag)
    stride = max(1, _env_int("HNODECB_AFM05_STAGE1PLUS_WINDOW_SAMPLE_STRIDE", 16))
    first_contact_time_s = float(meta["AFM05_first_contact_time_s"])

    THESIS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    thesis_out = THESIS_OUT_DIR / f"afm05_{pixel_key}_full_timespan_x1_x2.png"
    _plot_full_timespan_for_thesis(
        t,
        x1,
        x2,
        first_contact_time_s,
        float(t[idxs[0]]),
        float(t[idxs[-1]]),
        thesis_out,
    )

    full_out = OUT_DIR / f"afm05_{pixel_key}_full_timespan_x1_x2_x2dot.png"
    _plot_three_panel(
        t,
        x1,
        x2,
        x2dot,
        full_out,
        time_scale=1.0e3,
        time_label="time from trace start [ms]",
        title=f"AFM05 {pixel_key}: full time span",
        first_contact_x=(first_contact_time_s - float(t[0])) * 1.0e3,
    )

    val_stride = max(1, _env_int("HNODECB_AFM05_STAGE1_VAL_STRIDE", 5))
    val_offset = max(1, _env_int("HNODECB_AFM05_STAGE1_VAL_OFFSET", 2))
    local_idx = np.arange(idxs.size, dtype=int)
    val_local_mask = ((local_idx + 1 - val_offset) % val_stride) == 0
    train_marker_mask = ~val_local_mask
    val_marker_mask = ~train_marker_mask
    window_out = OUT_DIR / f"afm05_{pixel_key}_window_05_initial_stride{stride}_x1_x2_x2dot.png"
    _plot_initial_window_two_panel(
        t[idxs],
        x1[idxs],
        x2[idxs],
        window_out,
    )
    thesis_window_out = THESIS_OUT_DIR / window_out.name
    _plot_initial_window_two_panel(
        t[idxs],
        x1[idxs],
        x2[idxs],
        thesis_window_out,
        font_multiplier=4.0 / 3.0,
        time_label=r"time [$\mu$s]",
    )

    print(f"Saved: {full_out}")
    print(f"Saved: {window_out}")
    print(f"Saved: {thesis_window_out}")
    print(f"Saved: {thesis_out}")
    print(
        f"Window points: {idxs.size}, train_markers={int(np.count_nonzero(train_marker_mask))}, "
        f"start_idx={int(idxs[0])}, stop_idx={int(idxs[-1])}"
    )


if __name__ == "__main__":
    main()
