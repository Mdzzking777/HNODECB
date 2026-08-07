from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from GP_data_denoise_abhi_silicon import (
    AFM06A_PERIOD_S,
    AFM06A_TRANSITION_HALF_WIDTH_S,
    DRIVE_FREQUENCY_HZ,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MIDDLE_RAW_NPZ = (
    SCRIPT_DIR
    / "GP"
    / "middle"
    / "silicon_gp_index109_mid5000_490_510us_x2_x2dot.npz"
)
CONTACT_JUDGE_NPZ = (
    SCRIPT_DIR
    / "GP"
    / "middle"
    / "x1 for contact judge"
    / "silicon_gp_index109_mid5000_490_510us_x1_contact_judge.npz"
)
OUTPUT_DIR = SCRIPT_DIR / "GP" / "middle" / "x2 filtered"
OUTPUT_STEM = "silicon_gp_index109_mid5000_490_510us_x2_regime_aware_sg"
TWO_PANEL_OUTPUT = OUTPUT_DIR / "silicon_middle_x2_raw_vs_tilde.png"

POLYORDER = 3
WINDOW_BY_REGIME = {
    "noncontact": 101,
    "contact": 41,
    "transition": 11,
}


def _odd_window_at_most(length: int, preferred: int, polyorder: int) -> int | None:
    if length <= polyorder:
        return None
    window = min(int(preferred), int(length))
    if window % 2 == 0:
        window -= 1
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    if window < minimum:
        return None
    return window


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks] + 1, idx[-1] + 1]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _transition_band_mask(
    *,
    time_s: np.ndarray,
    transition_idx: np.ndarray,
    half_width_s: float,
) -> np.ndarray:
    transition_band = np.zeros(time_s.shape, dtype=bool)
    if transition_idx.size == 0:
        return transition_band
    for idx in transition_idx:
        transition_time = float(time_s[int(idx)])
        transition_band |= np.abs(time_s - transition_time) <= half_width_s
    return transition_band


def _apply_segmented_sg(
    *,
    values: np.ndarray,
    time_s: np.ndarray,
    masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[dict[str, int | str]]]]:
    dt_s = float(np.median(np.diff(time_s)))
    filtered = np.asarray(values, dtype=np.float64).copy()
    derivative = np.gradient(filtered, time_s, edge_order=2)
    segment_report: dict[str, list[dict[str, int | str]]] = {
        "noncontact": [],
        "contact": [],
        "transition": [],
    }

    for regime in ("noncontact", "contact", "transition"):
        preferred_window = WINDOW_BY_REGIME[regime]
        for start, stop in _contiguous_runs(masks[regime]):
            segment = np.asarray(values[start:stop], dtype=np.float64)
            window = _odd_window_at_most(segment.size, preferred_window, POLYORDER)
            record: dict[str, int | str] = {
                "start_index": start,
                "stop_index_exclusive": stop,
                "length": int(segment.size),
                "preferred_window": int(preferred_window),
            }
            if window is None:
                record["status"] = "kept_raw_segment_too_short"
                segment_report[regime].append(record)
                continue
            filtered[start:stop] = savgol_filter(
                segment,
                window_length=window,
                polyorder=POLYORDER,
                mode="interp",
            )
            derivative[start:stop] = savgol_filter(
                segment,
                window_length=window,
                polyorder=POLYORDER,
                deriv=1,
                delta=dt_s,
                mode="interp",
            )
            record["status"] = "filtered"
            record["actual_window"] = int(window)
            segment_report[regime].append(record)

    return filtered, derivative, segment_report


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(MIDDLE_RAW_NPZ, allow_pickle=False) as raw_data:
        time_s = np.asarray(raw_data["time_s"], dtype=np.float64)
        source_idx = np.asarray(raw_data["source_idx"], dtype=np.int64)
        x1_raw_m = np.asarray(raw_data["x1_raw_m"], dtype=np.float64)
        x2_raw_m_s = np.asarray(raw_data["x2_raw_m_s"], dtype=np.float64)

    with np.load(CONTACT_JUDGE_NPZ, allow_pickle=False) as judge_data:
        judge_source_idx = np.asarray(judge_data["source_idx"], dtype=np.int64)
        x1_tilde_m = np.asarray(judge_data["x1_tilde_m"], dtype=np.float64)
        contact_tilde = np.asarray(judge_data["contact_tilde"], dtype=bool)
        transition_idx = np.asarray(judge_data["transition_idx"], dtype=np.int64)

    if not np.array_equal(source_idx, judge_source_idx):
        raise ValueError("middle raw data and contact-judge data use different source indices")

    external_period_s = 1.0 / DRIVE_FREQUENCY_HZ
    transition_half_width_s = (
        AFM06A_TRANSITION_HALF_WIDTH_S * external_period_s / AFM06A_PERIOD_S
    )
    transition_band = _transition_band_mask(
        time_s=time_s,
        transition_idx=transition_idx,
        half_width_s=transition_half_width_s,
    )
    contact_interior = contact_tilde & ~transition_band
    noncontact_interior = ~contact_tilde & ~transition_band

    masks = {
        "noncontact": noncontact_interior,
        "contact": contact_interior,
        "transition": transition_band,
    }
    x2_tilde_m_s, x2dot_tilde_m_s2, segment_report = _apply_segmented_sg(
        values=x2_raw_m_s,
        time_s=time_s,
        masks=masks,
    )
    x2dot_tilde_gradient_check_m_s2 = np.gradient(
        x2_tilde_m_s,
        time_s,
        edge_order=2,
    )

    output_npz = OUTPUT_DIR / f"{OUTPUT_STEM}.npz"
    output_json = OUTPUT_DIR / f"{OUTPUT_STEM}.json"
    output_png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"

    np.savez_compressed(
        output_npz,
        time_s=time_s,
        source_idx=source_idx,
        x1_raw_m=x1_raw_m,
        x1_tilde_m=x1_tilde_m,
        x2_raw_m_s=x2_raw_m_s,
        x2_tilde_m_s=x2_tilde_m_s,
        x2dot_tilde_m_s2=x2dot_tilde_m_s2,
        x2dot_tilde_gradient_check_m_s2=x2dot_tilde_gradient_check_m_s2,
        contact_tilde=contact_tilde,
        contact_interior=contact_interior,
        noncontact_interior=noncontact_interior,
        transition_band=transition_band,
        transition_idx=transition_idx,
        transition_half_width_s=np.array(transition_half_width_s, dtype=np.float64),
    )

    metadata = {
        "input_middle_raw_npz": str(MIDDLE_RAW_NPZ.resolve()),
        "input_contact_judge_npz": str(CONTACT_JUDGE_NPZ.resolve()),
        "output_npz": str(output_npz.resolve()),
        "notation": {
            "x_tilde": "smoothed / denoised data",
            "x_hat": "model prediction only; not used for these filtered observations",
        },
        "usage": (
            "x1_tilde is used only to judge contact / transition regime. "
            "x2_tilde and x2dot_tilde are the candidate smoothed observation data."
        ),
        "regime_rule": "contact_tilde comes from z_static + x1_tilde <= a0",
        "transition_half_width_s": float(transition_half_width_s),
        "transition_half_width_us": float(transition_half_width_s * 1.0e6),
        "sg_polyorder": POLYORDER,
        "sg_windows_points": WINDOW_BY_REGIME,
        "sample_count": int(time_s.size),
        "contact_points": int(np.count_nonzero(contact_tilde)),
        "contact_interior_points": int(np.count_nonzero(contact_interior)),
        "noncontact_interior_points": int(np.count_nonzero(noncontact_interior)),
        "transition_band_points": int(np.count_nonzero(transition_band)),
        "transition_points": int(transition_idx.size),
        "segment_report": segment_report,
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 9.0),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    axes[0].plot(time_us, 1.0e9 * x1_raw_m, color="0.65", linewidth=0.7, label="raw")
    axes[0].plot(
        time_us,
        1.0e9 * x1_tilde_m,
        color="#1f77b4",
        linewidth=1.0,
        label=r"$\tilde{x}_1$ for contact judge",
    )
    axes[1].plot(time_us, x2_raw_m_s, color="0.65", linewidth=0.7, label="raw")
    axes[1].plot(
        time_us,
        x2_tilde_m_s,
        color="#1f77b4",
        linewidth=1.0,
        label=r"$\tilde{x}_2$",
    )
    axes[2].plot(time_us, x2_tilde_m_s, color="#1f77b4", linewidth=1.0)
    axes[3].plot(
        time_us,
        x2dot_tilde_m_s2,
        color="#d62728",
        linewidth=0.9,
        label=r"$\dot{\tilde{x}}_2$",
    )

    for ax in axes:
        for idx in transition_idx:
            ax.axvline(time_us[int(idx)], color="black", linewidth=0.7, alpha=0.45)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)

    axes[0].set_ylabel(r"$x_1$ [nm]")
    axes[1].set_ylabel(r"$x_2$ [m s$^{-1}$]")
    axes[2].set_ylabel(r"$\tilde{x}_2$ [m s$^{-1}$]")
    axes[3].set_ylabel(r"$\dot{\tilde{x}}_2$ [m s$^{-2}$]")
    axes[3].set_xlabel(r"Time [$\mu$s]")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[3].legend(loc="upper right")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(
        2,
        1,
        figsize=(10, 5.8),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    axes2[0].plot(time_us, x2_raw_m_s, color="0.35", linewidth=0.8)
    axes2[1].plot(time_us, x2_tilde_m_s, color="#1f77b4", linewidth=1.0)
    axes2[0].set_ylabel(r"$x_2^{\mathrm{raw}}$ [m s$^{-1}$]")
    axes2[1].set_ylabel(r"$\tilde{x}_2$ [m s$^{-1}$]")
    axes2[1].set_xlabel(r"Time [$\mu$s]")
    for axis in axes2:
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.0)
    fig2.savefig(TWO_PANEL_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved: {output_npz}")
    print(f"Saved: {output_json}")
    print(f"Saved: {output_png}")
    print(f"Saved: {TWO_PANEL_OUTPUT}")
    print(f"transition_half_width_us={transition_half_width_s * 1.0e6:.6f}")
    print(
        "points: "
        f"noncontact={np.count_nonzero(noncontact_interior)}, "
        f"contact={np.count_nonzero(contact_interior)}, "
        f"transition={np.count_nonzero(transition_band)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
