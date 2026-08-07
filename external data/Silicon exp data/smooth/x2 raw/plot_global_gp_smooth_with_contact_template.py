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
JUMP_SUMMARY_JSON = (
    SILICON_DIR
    / "acceleration jump judge"
    / "index109_400_800us_transition_slope_jumps_summary.json"
)

OUTPUT_NPZ = SCRIPT_DIR / "silicon_middle_x2_raw_global_gp_smooth_with_contact_template.npz"
OUTPUT_JSON = SCRIPT_DIR / "silicon_middle_x2_raw_global_gp_smooth_with_contact_template_summary.json"
OUTPUT_PNG = SCRIPT_DIR / "silicon_middle_x2_raw_global_gp_smooth_with_contact_template.png"

SLOPE_WINDOW_US = 0.36


def _transition_type(contact: np.ndarray, idx: int) -> str:
    before = bool(contact[idx - 1]) if idx > 0 else False
    after = bool(contact[idx]) if idx < contact.size else False
    if (not before) and after:
        return "N->C"
    if before and (not after):
        return "C->N"
    return "unknown"


def _contact_segments(contact: np.ndarray) -> list[tuple[int, int]]:
    contact = np.asarray(contact, dtype=bool)
    starts = np.flatnonzero(contact & np.r_[True, ~contact[:-1]])
    stops = np.flatnonzero(contact & np.r_[~contact[1:], True])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _local_slope(
    *,
    time_us: np.ndarray,
    values: np.ndarray,
    idx: int,
    side: str,
    window_us: float,
) -> float:
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
    slope_per_us, _ = np.polyfit(x, y, deg=1)
    return float(slope_per_us * 1.0e6)


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


def _cubic_hermite(
    *,
    time_s: np.ndarray,
    y0: float,
    y1: float,
    slope0: float,
    slope1: float,
) -> tuple[np.ndarray, np.ndarray]:
    if time_s.size < 2:
        raise RuntimeError("Hermite segment needs at least two samples")
    duration_s = float(time_s[-1] - time_s[0])
    if duration_s <= 0.0:
        raise RuntimeError("Hermite segment has non-positive duration")
    tau = (time_s - time_s[0]) / duration_s
    h00 = 2.0 * tau**3 - 3.0 * tau**2 + 1.0
    h10 = tau**3 - 2.0 * tau**2 + tau
    h01 = -2.0 * tau**3 + 3.0 * tau**2
    h11 = tau**3 - tau**2
    values = h00 * y0 + h10 * duration_s * slope0 + h01 * y1 + h11 * duration_s * slope1

    dh00 = 6.0 * tau**2 - 6.0 * tau
    dh10 = 3.0 * tau**2 - 4.0 * tau + 1.0
    dh01 = -6.0 * tau**2 + 6.0 * tau
    dh11 = 3.0 * tau**2 - 2.0 * tau
    slopes = (
        dh00 * y0
        + dh10 * duration_s * slope0
        + dh01 * y1
        + dh11 * duration_s * slope1
    ) / duration_s
    return values, slopes


def _build_x2_tilda_fixed_curve(
    *,
    time_s: np.ndarray,
    time_us: np.ndarray,
    x2_background: np.ndarray,
    x2_background_dot: np.ndarray,
    contact: np.ndarray,
    jump_n_to_c_m_s2: float,
    jump_c_to_n_m_s2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    x2_fixed = np.asarray(x2_background, dtype=float).copy()
    x2_fixed_dot = np.asarray(x2_background_dot, dtype=float).copy()
    segment_metadata: list[dict[str, float | int]] = []

    for start, stop in _contact_segments(contact):
        c_to_n_idx = stop + 1
        if c_to_n_idx >= x2_background.size:
            continue

        if start <= 0 or c_to_n_idx >= x2_background_dot.size:
            continue

        # Use the same sampled derivative data that will be saved and plotted.
        # This keeps the imposed transition jump and the visualized jump in one
        # definition, instead of mixing local-fit slopes with pointwise slopes.
        pre_slope = float(x2_background_dot[start - 1])
        post_slope = float(x2_background_dot[c_to_n_idx])
        slope_start = pre_slope + float(jump_n_to_c_m_s2)
        slope_stop = post_slope - float(jump_c_to_n_m_s2)

        local_slice = slice(start, c_to_n_idx + 1)
        local_time = time_s[local_slice]
        local_x2, local_dot = _cubic_hermite(
            time_s=local_time,
            y0=float(x2_background[start]),
            y1=float(x2_background[c_to_n_idx]),
            slope0=float(slope_start),
            slope1=float(slope_stop),
        )

        x2_fixed[local_slice] = local_x2
        x2_fixed_dot[local_slice] = local_dot

        # Encode the one-sided transition jumps directly in the stored sampled
        # derivative. The fixed curve remains continuous; the derivative has the
        # prescribed jumps at the transition boundaries.
        x2_fixed_dot[start - 1] = pre_slope
        x2_fixed_dot[start] = slope_start
        x2_fixed_dot[c_to_n_idx - 1] = slope_stop
        x2_fixed_dot[c_to_n_idx] = post_slope

        n_to_c_array_jump = float(x2_fixed_dot[start] - x2_fixed_dot[start - 1])
        c_to_n_array_jump = float(x2_fixed_dot[c_to_n_idx] - x2_fixed_dot[c_to_n_idx - 1])

        segment_metadata.append(
            {
                "n_to_c_index": int(start),
                "c_to_n_index": int(c_to_n_idx),
                "contact_stop_index": int(stop),
                "n_to_c_time_us": float(time_us[start]),
                "c_to_n_time_us": float(time_us[c_to_n_idx]),
                "contact_stop_time_us": float(time_us[stop]),
                "contact_duration_us": float(time_us[c_to_n_idx] - time_us[start]),
                "background_pre_sampled_slope_m_s2": float(pre_slope),
                "background_post_sampled_slope_m_s2": float(post_slope),
                "fixed_start_slope_m_s2": float(slope_start),
                "fixed_stop_slope_m_s2": float(slope_stop),
                "constructed_N_to_C_jump_m_s2": float(slope_start - pre_slope),
                "constructed_C_to_N_jump_m_s2": float(post_slope - slope_stop),
                "array_adjacent_N_to_C_jump_m_s2": n_to_c_array_jump,
                "array_adjacent_C_to_N_jump_m_s2": c_to_n_array_jump,
                "array_adjacent_N_to_C_over_introduced": float(
                    n_to_c_array_jump / jump_n_to_c_m_s2
                ),
                "array_adjacent_C_to_N_over_introduced": float(
                    c_to_n_array_jump / jump_c_to_n_m_s2
                ),
                "background_contact_delta_m_s": float(
                    x2_background[c_to_n_idx] - x2_background[start]
                ),
                "fixed_contact_delta_m_s": float(local_x2[-1] - local_x2[0]),
                "n_to_c_value_gap_m_s": float(local_x2[0] - x2_background[start]),
                "c_to_n_value_gap_m_s": float(
                    local_x2[-1] - x2_background[c_to_n_idx]
                ),
            }
        )

    direct_gradient_dot = np.gradient(x2_fixed, time_s, edge_order=2)
    return x2_fixed, x2_fixed_dot, direct_gradient_dot, segment_metadata


def main() -> int:
    with np.load(SMOOTH_NPZ, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        x2_raw = np.asarray(data["x2_raw_m_s"], dtype=float)
        x2_background = np.asarray(data["x2_tilde_m_s"], dtype=float)
        x2_background_dot = np.asarray(data["x2dot_tilde_m_s2"], dtype=float)

    with np.load(CONTACT_NPZ, allow_pickle=False) as data:
        contact = np.asarray(data["contact_tilde"], dtype=bool)
        transition_idx = np.asarray(data["transition_idx"], dtype=int)

    if not (time_s.shape == x2_raw.shape == x2_background.shape == contact.shape):
        raise RuntimeError("middle-slice x2 and contact arrays are not aligned")

    time_us = time_s * 1.0e6
    j_nc, j_cn = _load_jump_priors()
    x2_fixed, x2_fixed_dot, x2_fixed_direct_gradient_dot, segments = (
        _build_x2_tilda_fixed_curve(
            time_s=time_s,
            time_us=time_us,
            x2_background=x2_background,
            x2_background_dot=x2_background_dot,
            contact=contact,
            jump_n_to_c_m_s2=j_nc,
            jump_c_to_n_m_s2=j_cn,
        )
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        time_s=time_s,
        time_us=time_us,
        x2_raw_m_s=x2_raw,
        x2_background_m_s=x2_background,
        x2_background_dot_m_s2=x2_background_dot,
        contact=contact,
        transition_idx=transition_idx,
        x2_tilda_pure_m_s=x2_background,
        x2_tilda_pure_dot_m_s2=x2_background_dot,
        x2_tilda_fixed_m_s=x2_fixed,
        x2_tilda_fixed_dot_m_s2=x2_fixed_dot,
        x2_tilda_fixed_direct_gradient_dot_m_s2=x2_fixed_direct_gradient_dot,
        jump_prior_N_to_C_m_s2=j_nc,
        jump_prior_C_to_N_m_s2=j_cn,
        # Compatibility aliases for scripts that still read the previous names.
        x2_smooth_change_m_s=x2_fixed,
        x2_smooth_change_dot_m_s2=x2_fixed_dot,
        x2_smooth_change_gradient_dot_m_s2=x2_fixed_direct_gradient_dot,
        x2_background_plus_contact_template_mean_tilde_m_s=x2_fixed,
        x2_background_plus_contact_template_median_tilde_m_s=x2_fixed,
        x2_background_plus_contact_template_mean_tilde_dot_m_s2=x2_fixed_dot,
    )

    summary = {
        "definition": (
            "Corrected pipeline: x2 raw is first strongly GP-smoothed into "
            "x2_tilda_pure. External transition slope jumps are imposed directly "
            "on the same sampled derivative that is saved and plotted. The contact "
            "red curve is a cubic Hermite curve constrained by the sampled "
            "one-sided derivative values and the pure-GP endpoint values, giving "
            "x2_tilda_fixed. x2_tilda_fixed_dot is the canonical sampled "
            "derivative used for downstream force-residual calculations."
        ),
        "previous_background_plus_bc_template_removed": True,
        "order": [
            "x2 raw strong GP -> x2_tilda_pure",
            "external transition slope jumps -> one-sided boundary slopes",
            "contact-regime smooth curve -> x2_tilda_fixed",
            "differentiate x2_tilda_fixed -> x2_tilda_fixed_dot",
            "check constructed and adjacent slope jumps against introduced jumps",
        ],
        "smooth_background_npz": str(SMOOTH_NPZ),
        "contact_judge_npz": str(CONTACT_NPZ),
        "jump_summary_json": str(JUMP_SUMMARY_JSON),
        "output_npz": str(OUTPUT_NPZ),
        "output_png": str(OUTPUT_PNG),
        "time_range_us": [float(time_us[0]), float(time_us[-1])],
        "slope_window_us": SLOPE_WINDOW_US,
        "jump_prior_N_to_C_m_s2": float(j_nc),
        "jump_prior_C_to_N_m_s2": float(j_cn),
        "contact_segments": segments,
        "stored_derivative": (
            "x2_tilda_fixed_dot_m_s2 is the canonical sampled derivative: pure-GP "
            "derivative in noncontact, Hermite derivative in contact, and explicit "
            "one-sided sampled values at each transition so that the saved and "
            "plotted derivative has the prescribed N-C and C-N jump magnitudes. "
            "x2_tilda_fixed_direct_gradient_dot_m_s2 is also saved as a direct "
            "finite-difference gradient of x2_tilda_fixed."
        ),
        "compatibility_aliases": {
            "x2_smooth_change_m_s": "x2_tilda_fixed_m_s",
            "x2_smooth_change_dot_m_s2": "x2_tilda_fixed_dot_m_s2",
            "x2_background_plus_contact_template_mean_tilde_m_s": "x2_tilda_fixed_m_s",
            "x2_background_plus_contact_template_mean_tilde_dot_m_s2": "x2_tilda_fixed_dot_m_s2",
        },
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.0, 8.6),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    axes[0].plot(time_us, x2_raw, color="0.38", linewidth=0.75, label=r"$x_2^{raw}$")
    axes[0].set_ylabel(r"$x_2^{raw}$ [m s$^{-1}$]")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        time_us,
        x2_background,
        color="#1f77b4",
        linewidth=1.2,
        label=r"$\tilde{x}_2$ pure",
        zorder=2,
    )

    first_segment = True
    for start, stop in _contact_segments(contact):
        axes[1].plot(
            time_us[start : stop + 1],
            x2_fixed[start : stop + 1],
            color="#d62728",
            linewidth=1.15,
            label=r"$\tilde{x}_2$ fixed" if first_segment else None,
            zorder=5,
        )
        first_segment = False

    axes[1].set_ylabel(r"$x_2$ [m s$^{-1}$]")

    axes[2].plot(
        time_us,
        x2_fixed_dot,
        color="#2ca02c",
        linewidth=1.0,
        label=r"$\dot{\tilde{x}}_{2,\mathrm{fixed}}$",
        zorder=2,
    )
    axes[2].set_ylabel(r"$\dot{\tilde{x}}_2$ [m s$^{-2}$]")
    axes[2].set_xlabel(r"Time [$\mu$s]")
    axes[2].legend(loc="upper right")

    seen_nc = False
    seen_cn = False
    for idx in transition_idx:
        idx = int(idx)
        kind = _transition_type(contact, idx)
        for ax in axes:
            ax.axvline(time_us[idx], color="black", linewidth=0.5, alpha=0.18, zorder=1)
        if kind == "N->C":
            pre_slope = float(x2_fixed_dot[idx - 1])
            x, y = _reference_line_from_transition(
                time_us=time_us,
                values=x2_background,
                idx=idx,
                side="plus",
                slope_m_s2=float(x2_fixed_dot[idx]),
                window_us=SLOPE_WINDOW_US,
            )
            axes[1].plot(
                x,
                y,
                color="#ff7f0e",
                linestyle="--",
                linewidth=0.65,
                zorder=6,
                label=r"N-C jump-imposed slope at transition +" if not seen_nc else None,
            )
            axes[2].vlines(
                time_us[idx],
                pre_slope,
                float(x2_fixed_dot[idx]),
                color="#ff7f0e",
                linewidth=1.2,
                zorder=7,
                label=r"N-C jump" if not seen_nc else None,
            )
            seen_nc = True
        elif kind == "C->N":
            contact_left_slope = float(x2_fixed_dot[idx - 1])
            post_slope = float(x2_fixed_dot[idx])
            x, y = _reference_line_from_transition(
                time_us=time_us,
                values=x2_background,
                idx=idx,
                side="minus",
                slope_m_s2=contact_left_slope,
                window_us=SLOPE_WINDOW_US,
            )
            axes[1].plot(
                x,
                y,
                color="#9467bd",
                linestyle="--",
                linewidth=0.65,
                zorder=6,
                label=r"C-N jump-imposed slope at transition -" if not seen_cn else None,
            )
            axes[2].vlines(
                time_us[idx],
                contact_left_slope,
                post_slope,
                color="#9467bd",
                linewidth=1.2,
                zorder=7,
                label=r"C-N jump" if not seen_cn else None,
            )
            seen_cn = True

    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_NPZ}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
