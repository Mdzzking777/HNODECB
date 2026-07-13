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


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "AFM05" / "DATAgeneration"
SOURCE_NPZ = DATA_DIR / "PS_cantilever_disp_vel_time_3_pixels_Z_63.77_a0_0.07108_file_scan04143.imp_.npz"
OUT_DIR = DATA_DIR / "plot"


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
    pixel_key = f"pixel_{pixel_tag}"
    start_keys = (
        f"{pixel_key}_AFM05_initial_condition_index",
        f"{pixel_key}_initial_condition_index",
        f"{pixel_key}_first_contact_index",
        "AFM05_initial_condition_index",
        "AFM05_first_contact_index",
    )
    start_idx = None
    for key in start_keys:
        if key in meta:
            start_idx = int(meta[key])
            break
    if start_idx is None:
        raise KeyError(f"Could not find AFM05 initial-condition index for {pixel_key}.")

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
    z_a0_x1_nm: float | None = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.8), sharex=True)
    traces = (
        (x1 * 1.0e9, r"$x_1$ [nm]", "#1f77b4"),
        (x2, r"$x_2$ [m/s]", "#2ca02c"),
        (x2dot, r"$\dot{x}_2$ [m/s$^2$]", "#d62728"),
    )
    tx = (t - float(t[0])) * time_scale
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
    if z_a0_x1_nm is not None:
        axes[0].axhline(z_a0_x1_nm, color="black", linestyle="--", linewidth=1.2, zorder=8)
        x_text = tx[0] + 0.012 * (tx[-1] - tx[0])
        axes[0].text(
            x_text,
            z_a0_x1_nm,
            r"$z=a_0$",
            fontsize=11,
            color="black",
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            zorder=9,
        )
    axes[0].set_title(title, fontsize=13, pad=10)
    axes[-1].set_xlabel(time_label, fontsize=12)
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
    z_a0_x1_nm = (float(meta["a0"]) - float(meta["Z"])) * 1.0e9

    full_out = OUT_DIR / f"afm05_{pixel_key}_full_timespan_x1_x2_x2dot.png"
    full_window_box = (
        float((t[int(idxs[0])] - t[0]) * 1.0e3),
        float((t[int(idxs[-1])] - t[0]) * 1.0e3),
    )
    _plot_three_panel(
        t,
        x1,
        x2,
        x2dot,
        full_out,
        time_scale=1.0e3,
        time_label="time from trace start [ms]",
        title=f"AFM05 {pixel_key}: full time span",
        window_box_range=full_window_box,
        z_a0_x1_nm=z_a0_x1_nm,
    )

    val_stride = max(1, _env_int("HNODECB_AFM05_STAGE1_VAL_STRIDE", 5))
    val_offset = max(1, _env_int("HNODECB_AFM05_STAGE1_VAL_OFFSET", 2))
    all_global_idx = np.arange(t.size, dtype=int)
    val_global_idx = all_global_idx[((all_global_idx + 1 - val_offset) % val_stride) == 0]
    train_global_mask = np.ones(t.size, dtype=bool)
    train_global_mask[val_global_idx] = False
    train_marker_mask = train_global_mask[idxs]
    val_marker_mask = ~train_marker_mask
    window_out = OUT_DIR / f"afm05_{pixel_key}_window_05_initial_stride{stride}_x1_x2_x2dot.png"
    _plot_three_panel(
        t[idxs],
        x1[idxs],
        x2[idxs],
        x2dot[idxs],
        window_out,
        time_scale=1.0e6,
        time_label="time from window start [us]",
        title=f"AFM05 {pixel_key}: current window_05_initial",
        train_marker_mask=train_marker_mask,
        val_marker_mask=val_marker_mask,
        z_a0_x1_nm=z_a0_x1_nm,
    )

    print(f"Saved: {full_out}")
    print(f"Saved: {window_out}")
    print(
        f"Window points: {idxs.size}, train_markers={int(np.count_nonzero(train_marker_mask))}, "
        f"start_idx={int(idxs[0])}, stop_idx={int(idxs[-1])}"
    )


if __name__ == "__main__":
    main()
