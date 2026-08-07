"""Plot the AFM05 pixel_184_152 force residual and detailed windows."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


AFM05_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = (
    AFM05_ROOT
    / "DATAgeneration"
    / "PS_cantilever_disp_vel_time_3_pixels_Z_85.0_a0_0.07108_file_scan04143.imp_.npz"
)
IN_AIR_PATH = (
    AFM05_ROOT
    / "DATAgeneration"
    / "PS_cantilever_disp_in_air_file_scan04143.imp_.npz"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "afm05_pixel_184_152_force_residual_full.png"
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else float(raw)


def _window_mask(
    t: np.ndarray,
    start_s: float,
    span_s: float,
) -> np.ndarray:
    stop_s = min(float(t[-1]), float(start_s) + float(span_s))
    mask = (t >= float(start_s)) & (t <= stop_s)
    if not np.any(mask):
        raise ValueError(
            f"AFM05 residual window [{start_s}, {stop_s}] s is empty"
        )
    return mask


def main() -> int:
    if not TRACE_PATH.is_file():
        raise FileNotFoundError(f"AFM05 trace dataset not found: {TRACE_PATH}")
    if not IN_AIR_PATH.is_file():
        raise FileNotFoundError(f"AFM05 in-air dataset not found: {IN_AIR_PATH}")

    with np.load(TRACE_PATH, allow_pickle=True) as archive:
        pixel = np.asarray(archive["pixel_184_152"], dtype=float)
        f_ts = np.asarray(archive["F_ts_184_152"], dtype=float)
        metadata = dict(archive["add_data_dict"].item())

    if pixel.ndim != 2 or pixel.shape[1] != 4:
        raise ValueError(
            "pixel_184_152 must have columns [x1, x2, x2dot, t]"
        )
    x1 = pixel[:, 0]
    x2 = pixel[:, 1]
    x2dot = pixel[:, 2]
    t = pixel[:, 3]

    with np.load(IN_AIR_PATH, allow_pickle=True) as archive:
        f_actuation = np.asarray(
            archive["eta1_in_air_F_actuation_t"], dtype=float
        )

    arrays = (x1, x2, x2dot, f_ts, f_actuation)
    if t.size == 0 or any(value.shape != t.shape for value in arrays):
        raise ValueError("AFM05 trajectory and force arrays must be aligned")
    if not all(np.all(np.isfinite(value)) for value in (t, *arrays)):
        raise ValueError("AFM05 trajectory contains non-finite values")
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("AFM05 time vector must be strictly increasing")

    k_eff = float(metadata["k"])
    m_eff = float(metadata["F_ts_184_152_m_eff"])
    c_eff = float(metadata["F_ts_184_152_c_eff"])
    first_contact_idx = int(metadata["AFM05_first_contact_index"])
    if first_contact_idx < 0 or first_contact_idx >= t.size:
        raise ValueError(
            "AFM05_first_contact_index is outside the stored trajectory"
        )

    force_residual_lhs = m_eff * x2dot + c_eff * x2 + k_eff * x1
    force_residual_rhs = f_actuation + f_ts
    closure_error = force_residual_lhs - force_residual_rhs
    max_closure_error = float(np.max(np.abs(closure_error)))

    force_residual_nn = force_residual_lhs * 1e9
    window_span_s = _env_float(
        "HNODECB_AFM05_STAGE1PLUS_ARCH_WINDOW_US",
        25.152e-6,
    )
    first_contact_s = float(t[first_contact_idx])
    middle_start_s = 1.0e-3
    tail_start_s = max(float(t[0]), float(t[-1]) - window_span_s)
    windows = (
        (
            "Training window",
            _window_mask(t, first_contact_s, window_span_s),
            "#4c9fdb",
        ),
        (
            "Observation window 1",
            _window_mask(t, middle_start_s, window_span_s),
            "#e78a32",
        ),
        (
            "Observation window 2",
            _window_mask(t, tail_start_s, window_span_s),
            "#4ca66b",
        ),
    )

    # The full trace has more than 700,000 samples. Display decimation is used
    # only in the overview panel; every stored point is retained in each detail.
    full_stride = max(1, int(np.ceil(t.size / 100000)))
    full_indices = np.arange(0, t.size, full_stride, dtype=int)
    if full_indices[-1] != t.size - 1:
        full_indices = np.append(full_indices, t.size - 1)

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), squeeze=False)
    axes_flat = list(axes.ravel())
    overview = axes_flat[0]
    overview.plot(
        t[full_indices] * 1e6,
        force_residual_nn[full_indices],
        color="tab:blue",
        linewidth=0.65,
        rasterized=True,
    )
    for label, mask, color in windows:
        t_window_us = t[mask] * 1e6
        overview.axvspan(
            float(t_window_us[0]),
            float(t_window_us[-1]),
            color=color,
            alpha=0.24,
            linewidth=0.0,
            label=label,
        )
    overview.set_title("Full time span", fontsize=15, pad=7)
    overview.legend(loc="upper right", fontsize=10.5, framealpha=0.92)
    overview.set_xlim(float(t[0] * 1e6), float(t[-1] * 1e6))

    for axis, (label, mask, color) in zip(
        axes_flat[1:],
        windows,
        strict=True,
    ):
        t_window_us = t[mask] * 1e6
        axis.plot(
            t_window_us,
            force_residual_nn[mask],
            color=color,
            linewidth=1.0,
        )
        axis.set_title(
            f"{label}: "
            f"{t_window_us[0]:.3f}-{t_window_us[-1]:.3f} "
            r"$\mu$s",
            fontsize=15,
            pad=7,
        )
        axis.set_xlim(float(t_window_us[0]), float(t_window_us[-1]))

    for axis in axes_flat:
        axis.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.6,
            alpha=0.8,
        )
        axis.set_xlabel(r"Time [$\mu$s]", fontsize=14)
        axis.set_ylabel(
            r"$F_{\mathrm{res}}=F_{\mathrm{act}}+F_{ts}$ [nN]",
            fontsize=14,
        )
        axis.tick_params(axis="both", labelsize=11.5)
        axis.grid(True, alpha=0.28, linewidth=0.8)
        axis.margins(x=0.0)

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.085,
        top=0.965,
        wspace=0.20,
        hspace=0.29,
    )
    fig.savefig(OUTPUT_PATH, dpi=300, facecolor="white")
    plt.close(fig)

    print(f"m_eff = {m_eff:.12g} kg")
    print(f"c_eff = {c_eff:.12g} N s/m")
    print(f"k_eff = {k_eff:.12g} N/m")
    print(
        "F_res range = "
        f"[{force_residual_nn.min():.12g}, "
        f"{force_residual_nn.max():.12g}] nN"
    )
    print(
        "F_res RMS = "
        f"{np.sqrt(np.mean(force_residual_nn**2)):.12g} nN"
    )
    for label, mask, _color in windows:
        print(
            f"{label}: "
            f"[{t[mask][0] * 1e6:.6f}, "
            f"{t[mask][-1] * 1e6:.6f}] us, "
            f"samples={int(np.count_nonzero(mask))}"
        )
    print(f"LHS/RHS max closure error = {max_closure_error:.12g} N")
    print(f"Saved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
