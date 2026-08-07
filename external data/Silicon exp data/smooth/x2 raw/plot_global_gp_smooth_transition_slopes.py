from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SILICON_DIR = SCRIPT_DIR.parents[1]
SMOOTH_NPZ = SCRIPT_DIR / "silicon_middle_x2_raw_global_gp_smooth.npz"
CONTACT_NPZ = (
    SILICON_DIR
    / "GP"
    / "middle"
    / "x1 for contact judge"
    / "silicon_gp_index109_mid5000_490_510us_x1_contact_judge.npz"
)
OUTPUT_PNG = SCRIPT_DIR / "silicon_middle_x2_raw_global_gp_smooth_transition_slopes.png"
JUMP_SUMMARY_JSON = (
    SILICON_DIR
    / "acceleration jump judge"
    / "index109_400_800us_transition_slope_jumps_summary.json"
)
OUTPUT_JUMP_PNG = SCRIPT_DIR / "silicon_middle_x2_raw_global_gp_smooth_transition_jump_slopes.png"

SLOPE_WINDOW_US = 0.36


def _transition_type(contact: np.ndarray, idx: int) -> str:
    before = bool(contact[idx - 1]) if idx > 0 else False
    after = bool(contact[idx]) if idx < contact.size else False
    if (not before) and after:
        return "N->C"
    if before and (not after):
        return "C->N"
    return "unknown"


def _line_fit_segment(
    *,
    time_us: np.ndarray,
    values: np.ndarray,
    idx: int,
    side: str,
    window_us: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if side not in {"plus", "minus"}:
        raise ValueError(f"unexpected side: {side}")
    dt_us = float(np.median(np.diff(time_us)))
    n_win = max(8, int(round(window_us / dt_us)))
    if side == "plus":
        start = idx
        stop = min(values.size, idx + n_win + 1)
    else:
        start = max(0, idx - n_win)
        stop = idx + 1
    x = time_us[start:stop]
    y = values[start:stop]
    if x.size < 3:
        raise RuntimeError(f"not enough points for local slope near index {idx}")
    slope, intercept = np.polyfit(x, y, deg=1)
    return x, slope * x + intercept, float(slope)


def _local_slope(
    *,
    time_us: np.ndarray,
    values: np.ndarray,
    idx: int,
    side: str,
    window_us: float,
) -> float:
    _, _, slope_per_us = _line_fit_segment(
        time_us=time_us,
        values=values,
        idx=idx,
        side=side,
        window_us=window_us,
    )
    return slope_per_us * 1.0e6


def _reference_line_from_transition(
    *,
    time_us: np.ndarray,
    values: np.ndarray,
    idx: int,
    side: str,
    slope_m_s2: float,
    window_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt_us = float(np.median(np.diff(time_us)))
    n_win = max(8, int(round(window_us / dt_us)))
    if side == "plus":
        start = idx
        stop = min(values.size, idx + n_win + 1)
    elif side == "minus":
        start = max(0, idx - n_win)
        stop = idx + 1
    else:
        raise ValueError(f"unexpected side: {side}")
    x = time_us[start:stop]
    t0 = time_us[idx]
    y0 = values[idx]
    y = y0 + slope_m_s2 * ((x - t0) * 1.0e-6)
    return x, y


def _load_jump_priors() -> tuple[float, float]:
    summary = json.loads(JUMP_SUMMARY_JSON.read_text(encoding="utf-8"))
    priors = summary["summary_by_fit_half_width"]["64_ns"]
    j_nc = float(priors["N_to_C"]["mean_delta_time_x2dot_raw_m_s2"])
    j_cn = float(priors["C_to_N"]["mean_delta_time_x2dot_raw_m_s2"])
    return j_nc, j_cn


def main() -> int:
    with np.load(SMOOTH_NPZ, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        x2_raw = np.asarray(data["x2_raw_m_s"], dtype=float)
        x2_tilde = np.asarray(data["x2_tilde_m_s"], dtype=float)

    with np.load(CONTACT_NPZ, allow_pickle=False) as data:
        contact = np.asarray(data["contact_tilde"], dtype=bool)
        transition_idx = np.asarray(data["transition_idx"], dtype=int)
    j_nc, j_cn = _load_jump_priors()

    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 6.5),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )

    axes[0].plot(time_us, x2_raw, color="0.38", linewidth=0.75, label=r"$x_2^{raw}$")
    axes[0].set_ylabel(r"$x_2^{raw}$ [m s$^{-1}$]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_us, x2_tilde, color="#1f77b4", linewidth=1.2, label=r"$\tilde{x}_2$ (GP)")
    axes[1].set_ylabel(r"$\tilde{x}_2$ [m s$^{-1}$]")
    axes[1].set_xlabel(r"Time [$\mu$s]")

    seen_nc = False
    seen_cn = False
    for idx in transition_idx:
        idx = int(idx)
        kind = _transition_type(contact, idx)
        for ax in axes:
            ax.axvline(time_us[idx], color="black", linewidth=0.5, alpha=0.18, zorder=1)
        if kind == "N->C":
            x, y, _ = _line_fit_segment(
                time_us=time_us,
                values=x2_tilde,
                idx=idx,
                side="plus",
                window_us=SLOPE_WINDOW_US,
            )
            axes[1].plot(
                x,
                y,
                color="#ff7f0e",
                linestyle="--",
                linewidth=2.1,
                zorder=4,
                label="N-C slope at transition +" if not seen_nc else None,
            )
            seen_nc = True
        elif kind == "C->N":
            x, y, _ = _line_fit_segment(
                time_us=time_us,
                values=x2_tilde,
                idx=idx,
                side="minus",
                window_us=SLOPE_WINDOW_US,
            )
            axes[1].plot(
                x,
                y,
                color="#9467bd",
                linestyle="--",
                linewidth=2.1,
                zorder=4,
                label="C-N slope at transition -" if not seen_cn else None,
            )
            seen_cn = True

    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 6.5),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    axes[0].plot(time_us, x2_raw, color="0.38", linewidth=0.75, label=r"$x_2^{raw}$")
    axes[0].set_ylabel(r"$x_2^{raw}$ [m s$^{-1}$]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_us, x2_tilde, color="#1f77b4", linewidth=1.2, label=r"$\tilde{x}_2$ (GP)")
    axes[1].set_ylabel(r"$\tilde{x}_2$ [m s$^{-1}$]")
    axes[1].set_xlabel(r"Time [$\mu$s]")

    seen_nc = False
    seen_cn = False
    for idx in transition_idx:
        idx = int(idx)
        kind = _transition_type(contact, idx)
        for ax in axes:
            ax.axvline(time_us[idx], color="black", linewidth=0.5, alpha=0.18, zorder=1)
        if kind == "N->C":
            pre_slope = _local_slope(
                time_us=time_us,
                values=x2_tilde,
                idx=idx,
                side="minus",
                window_us=SLOPE_WINDOW_US,
            )
            x, y = _reference_line_from_transition(
                time_us=time_us,
                values=x2_tilde,
                idx=idx,
                side="plus",
                slope_m_s2=pre_slope + j_nc,
                window_us=SLOPE_WINDOW_US,
            )
            axes[1].plot(
                x,
                y,
                color="#ff7f0e",
                linestyle="--",
                linewidth=0.65,
                zorder=4,
                label=r"N-C jump-imposed slope at transition +" if not seen_nc else None,
            )
            seen_nc = True
        elif kind == "C->N":
            post_slope = _local_slope(
                time_us=time_us,
                values=x2_tilde,
                idx=idx,
                side="plus",
                window_us=SLOPE_WINDOW_US,
            )
            x, y = _reference_line_from_transition(
                time_us=time_us,
                values=x2_tilde,
                idx=idx,
                side="minus",
                slope_m_s2=post_slope - j_cn,
                window_us=SLOPE_WINDOW_US,
            )
            axes[1].plot(
                x,
                y,
                color="#9467bd",
                linestyle="--",
                linewidth=0.65,
                zorder=4,
                label=r"C-N jump-imposed slope at transition -" if not seen_cn else None,
            )
            seen_cn = True

    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)
    fig.savefig(OUTPUT_JUMP_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_JUMP_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
