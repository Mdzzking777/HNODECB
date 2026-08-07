"""Regenerate raw silicon index109 x1/x2 full-span and window plots from MAT."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MAT = SCRIPT_DIR / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
RAW_MAT = (
    Path(r"E:\External data SI_SI hard sample raw")
    / "OneDrive_2_2026-7-26"
    / "dataAFM_Si"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)

INDEX = 109
DT_S = 1.0 / 250.0e6
DRIVE_FREQUENCY_HZ = 164526.16455353564
PERIOD_S = 1.0 / DRIVE_FREQUENCY_HZ

FULL_OUTPUT = SCRIPT_DIR / "silicon_raw_forward_index109_x1_x2.png"
WINDOW_OUTPUT = SCRIPT_DIR / "silicon_raw_forward_index109_x1_x2_windows.png"


def _load_index109() -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    errors: list[str] = []
    for mat_path in (LOCAL_MAT, RAW_MAT):
        if not mat_path.is_file():
            errors.append(f"{mat_path} does not exist")
            continue
        try:
            data = sio.loadmat(
                str(mat_path),
                variable_names=["displacement_fwd_sweep", "velocity_fwd_sweep"],
                squeeze_me=True,
                struct_as_record=False,
            )
            x1 = np.asarray(data["displacement_fwd_sweep"][INDEX], dtype=float).ravel()
            x2 = np.asarray(data["velocity_fwd_sweep"][INDEX], dtype=float).ravel()
            if x1.shape != x2.shape or x1.size < 2:
                raise ValueError(f"invalid shapes: x1={x1.shape}, x2={x2.shape}")
            time_s = np.arange(x1.size, dtype=float) * DT_S
            return time_s, x1, x2, mat_path
        except Exception as exc:
            errors.append(f"{mat_path}: {exc}")
    raise RuntimeError("Could not read silicon MAT file:\n" + "\n".join(errors))


def _second_zero_crossing_time(time_s: np.ndarray, x1: np.ndarray) -> float:
    crossings = np.flatnonzero(np.signbit(x1[:-1]) != np.signbit(x1[1:]))
    if crossings.size < 2:
        return 0.0
    left = int(crossings[1])
    right = left + 1
    x_left = float(x1[left])
    x_right = float(x1[right])
    t_left = float(time_s[left])
    t_right = float(time_s[right])
    return t_left - x_left * (t_right - t_left) / (x_right - x_left)


def _plot_full(time_s: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> None:
    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    axes[0].plot(time_us, x1 * 1.0e9, color="black", linewidth=0.8)
    axes[1].plot(time_us, x2, color="black", linewidth=0.8)
    axes[0].set_ylabel(r"$x_1$ [nm]")
    axes[1].set_ylabel(r"$x_2$ [m s$^{-1}$]")
    axes[1].set_xlabel(r"time [$\mu$s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FULL_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _window_bounds(time_s: np.ndarray, x1: np.ndarray) -> list[tuple[str, float, float]]:
    t0 = _second_zero_crossing_time(time_s, x1)
    starts = [
        ("early", t0),
        ("quarter span", 0.25 * float(time_s[-1])),
        ("middle", 0.50 * float(time_s[-1])),
        ("tail", 0.75 * float(time_s[-1])),
    ]
    bounds: list[tuple[str, float, float]] = []
    for label, start in starts:
        start = max(0.0, min(float(start), float(time_s[-1]) - PERIOD_S))
        bounds.append((label, start, start + PERIOD_S))
    return bounds


def _plot_windows(time_s: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> None:
    x2dot = np.gradient(x2, time_s, edge_order=2)
    bounds = _window_bounds(time_s, x1)
    fig, axes = plt.subplots(
        3,
        len(bounds),
        figsize=(15, 8.0),
        sharex=False,
        gridspec_kw={"hspace": 0.18, "wspace": 0.22},
    )
    rows = (
        (x1 * 1.0e9, r"$x_1$ [nm]"),
        (x2, r"$x_2$ [m s$^{-1}$]"),
        (x2dot, r"$\dot{x}_2$ [m s$^{-2}$]"),
    )
    for col, (label, start, stop) in enumerate(bounds):
        mask = (time_s >= start) & (time_s <= stop)
        local_time_us = (time_s[mask] - start) * 1.0e6
        title = f"{label}\n{start * 1.0e6:.3f}-{stop * 1.0e6:.3f} us"
        for row, (values, ylabel) in enumerate(rows):
            ax = axes[row, col]
            ax.plot(local_time_us, values[mask], color="black", linewidth=0.9)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(rows) - 1:
                ax.set_xlabel(r"local time [$\mu$s]")
            ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(WINDOW_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    time_s, x1, x2, mat_path = _load_index109()
    _plot_full(time_s, x1, x2)
    _plot_windows(time_s, x1, x2)
    print(f"Read MAT: {mat_path}")
    print(f"index={INDEX} samples={time_s.size} time=[{time_s[0] * 1e6:.6g}, {time_s[-1] * 1e6:.6g}] us")
    print(f"Saved: {FULL_OUTPUT}")
    print(f"Saved: {WINDOW_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
