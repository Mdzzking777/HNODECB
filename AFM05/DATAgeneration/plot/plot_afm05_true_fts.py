"""Plot the AFM05 true tip-sample interaction force.

This is a data-generation diagnostic only.  The stored true F_ts is never
passed to stage losses, optimizers, gain initialization, or model inputs by
this script.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
AFM05_ROOT = REPO_ROOT / "AFM05"
DATA_PATH = (
    AFM05_ROOT
    / "DATAgeneration"
    / "PS_cantilever_disp_vel_time_3_pixels_Z_85.0_a0_0.07108_file_scan04143.imp_.npz"
)
OUT_PATH = AFM05_ROOT / "DATAgeneration" / "plot" / "afm05_pixel_184_152_true_Fts_full_and_window.png"
OBSERVATION_OUT_PATH = (
    AFM05_ROOT
    / "DATAgeneration"
    / "plot"
    / "afm05_pixel_184_152_true_Fts_observation_window_1p0ms.png"
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else float(raw)


def _load_true_force() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(DATA_PATH, allow_pickle=True) as data:
        pixel = np.asarray(data["pixel_184_152"], dtype=float)
        if pixel.ndim != 2 or pixel.shape[1] < 4:
            raise ValueError("pixel_184_152 must contain time in its fourth column.")
        x1 = np.asarray(pixel[:, 0], dtype=float)
        t = np.asarray(pixel[:, 3], dtype=float)
        f_ts = np.asarray(data["F_ts_184_152"], dtype=float)
        metadata = dict(data["add_data_dict"].item())
        start_idx = int(metadata["AFM05_first_contact_index"])

    if t.ndim != 1 or f_ts.ndim != 1 or x1.ndim != 1 or not (t.shape == f_ts.shape == x1.shape):
        raise ValueError("Stored t, x1, and F_ts arrays must be matching one-dimensional vectors.")
    if t.size < 2 or not np.all(np.diff(t) > 0.0):
        raise ValueError("Stored F_ts time samples must be strictly increasing.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(f_ts)) or not np.all(np.isfinite(x1)):
        raise ValueError("Stored t, x1, or true F_ts data contain non-finite values.")
    if start_idx < 0 or start_idx >= t.size:
        raise ValueError(f"AFM05 initial-condition index is outside the trace: {start_idx}.")
    return t, x1, f_ts, start_idx


def main() -> None:
    t, x1, f_ts, start_idx = _load_true_force()
    window_span_s = _env_float("HNODECB_AFM05_STAGE1PLUS_ARCH_WINDOW_US", 25.152e-6)
    window_mask = (np.arange(t.size) >= start_idx) & ((t - float(t[start_idx])) <= window_span_s)
    if not np.any(window_mask):
        window_mask[start_idx] = True

    t_ms = t * 1.0e3
    t_window_ms = t[window_mask] * 1.0e3
    f_ts_nN = f_ts * 1.0e9
    x1_nm = x1 * 1.0e9

    # The complete trace has more than 600,000 points.  A modest display-only
    # decimation keeps the raster rendering light while preserving the waveform.
    full_display_stride = max(1, int(np.ceil(t.size / 80000)))
    full_idxs = np.arange(0, t.size, full_display_stride, dtype=int)
    if full_idxs[-1] != t.size - 1:
        full_idxs = np.append(full_idxs, t.size - 1)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.0, 9.5),
        gridspec_kw={"height_ratios": [1.0, 1.15, 1.15], "hspace": 0.34},
    )

    axes[0].plot(
        t_ms[full_idxs],
        f_ts_nN[full_idxs],
        color="black",
        linewidth=0.65,
        rasterized=True,
    )
    window_start_ms = float(t[window_mask][0] * 1.0e3)
    window_stop_ms = float(t[window_mask][-1] * 1.0e3)
    axes[0].axvspan(window_start_ms, window_stop_ms, color="#8ecae6", alpha=0.42, linewidth=0.0)
    axes[0].set_title("Full time span", fontsize=13, pad=7)
    axes[0].set_xlabel("Time [ms]", fontsize=12)

    axes[1].plot(
        t_window_ms,
        f_ts_nN[window_mask],
        color="black",
        linewidth=1.05,
    )
    axes[1].set_title("Current training window", fontsize=13, pad=7)
    axes[1].set_xlabel("Time [ms]", fontsize=12)

    axes[2].plot(
        t_window_ms,
        x1_nm[window_mask],
        color="tab:blue",
        linewidth=1.05,
    )
    axes[2].set_title(r"$x_1$ in current training window", fontsize=13, pad=7)
    axes[2].set_xlabel("Time [ms]", fontsize=12)
    axes[2].set_ylabel(r"$x_1$ [nm]", fontsize=12)

    for ax in axes[:2]:
        ax.set_ylabel(r"$F_{ts}$ [nN]", fontsize=12)

    for ax in axes:
        ax.grid(True, color="#b0b0b0", alpha=0.28, linewidth=0.75)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.margins(x=0.0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.965)
    fig.savefig(OUT_PATH, dpi=300, facecolor="white")
    plt.close(fig)

    observation_start_s = 1.0e-3
    observation_mask = (t >= observation_start_s) & (t - observation_start_s <= window_span_s)
    if not np.any(observation_mask):
        raise ValueError("The requested 1.0 ms observation window is outside the stored trace.")

    observation_start_ms = float(t[observation_mask][0] * 1.0e3)
    observation_stop_ms = float(t[observation_mask][-1] * 1.0e3)
    fig_obs, axes_obs = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.0),
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.30},
    )
    axes_obs[0].plot(
        t_ms[full_idxs],
        f_ts_nN[full_idxs],
        color="black",
        linewidth=0.65,
        rasterized=True,
    )
    axes_obs[0].axvspan(
        observation_start_ms,
        observation_stop_ms,
        color="#8ecae6",
        alpha=0.42,
        linewidth=0.0,
    )
    axes_obs[0].set_title("Full time span", fontsize=13, pad=7)
    axes_obs[0].set_xlabel("Time [ms]", fontsize=12)

    axes_obs[1].plot(
        t_ms[observation_mask],
        f_ts_nN[observation_mask],
        color="black",
        linewidth=1.05,
    )
    axes_obs[1].set_title("Observation window", fontsize=13, pad=7)
    axes_obs[1].set_xlabel("Time [ms]", fontsize=12)

    for ax in axes_obs:
        ax.set_ylabel(r"$F_{ts}$ [nN]", fontsize=12)
        ax.grid(True, color="#b0b0b0", alpha=0.28, linewidth=0.75)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.margins(x=0.0)

    fig_obs.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.965)
    fig_obs.savefig(OBSERVATION_OUT_PATH, dpi=300, facecolor="white")
    plt.close(fig_obs)

    print(f"Saved: {OUT_PATH}")
    print(f"Source: {DATA_PATH}")
    print(f"Full samples: {t.size}")
    print(f"Window samples: {int(np.count_nonzero(window_mask))}")
    print(f"Window start: {t[window_mask][0] * 1.0e3:.6f} ms")
    print(f"Window stop: {t[window_mask][-1] * 1.0e3:.6f} ms")
    print(f"Window span: {(t[window_mask][-1] - t[window_mask][0]) * 1.0e6:.6f} us")
    print(f"F_ts range: [{np.min(f_ts):.12e}, {np.max(f_ts):.12e}] N")
    print(f"Saved: {OBSERVATION_OUT_PATH}")
    print(f"Observation start: {observation_start_ms:.6f} ms")
    print(f"Observation stop: {observation_stop_ms:.6f} ms")
    print(
        "Observation span: "
        f"{(t[observation_mask][-1] - t[observation_mask][0]) * 1.0e6:.6f} us"
    )


if __name__ == "__main__":
    main()
