from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SILICON_DIR = SCRIPT_DIR.parent

X1_STRONG_NPZ = (
    SILICON_DIR
    / "GP"
    / "middle"
    / "x1 for strong smoothness"
    / "silicon_middle_x1_strong_gp_margin20.npz"
)
X2_CONSTRAINTS_NPZ = (
    SILICON_DIR
    / "smooth"
    / "x2 raw"
    / "silicon_middle_x2_raw_global_gp_smooth_with_contact_template.npz"
)

OUTPUT_NPZ = SCRIPT_DIR / "silicon_middle_f_residual_x1strong_x2constraints.npz"
OUTPUT_JSON = SCRIPT_DIR / "silicon_middle_f_residual_x1strong_x2constraints_summary.json"
OUTPUT_PNG = SCRIPT_DIR / "silicon_middle_f_residual_x1strong_x2constraints.png"

MARGIN_FRACTION_PER_SIDE = 0.10
FACT_AVG_N = 18.9e-9


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def main() -> int:
    with np.load(X1_STRONG_NPZ, allow_pickle=False) as x1_data:
        time_s = np.asarray(x1_data["middle_time_s"], dtype=float)
        time_us = np.asarray(x1_data["middle_time_us"], dtype=float)
        x1_tilde_strong = np.asarray(x1_data["middle_x1_tilde_strong_m"], dtype=float)
        transition_idx = np.asarray(x1_data["middle_transition_idx"], dtype=int)
        contact = np.asarray(x1_data["middle_contact"], dtype=bool)
        mass_kg = float(x1_data["m_kg"])
        damping_n_s_m = float(x1_data["c_N_s_m"])
        stiffness_n_m = float(x1_data["k_N_m"])
        f0_hz = float(x1_data["f0_Hz"])
        q_factor = float(x1_data["Q"])

    with np.load(X2_CONSTRAINTS_NPZ, allow_pickle=False) as x2_data:
        x2_time_s = np.asarray(x2_data["time_s"], dtype=float)
        x2_tilde_constraints = np.asarray(x2_data["x2_tilda_fixed_m_s"], dtype=float)
        x2dot_tilde_constraints = np.asarray(
            x2_data["x2_tilda_fixed_dot_m_s2"], dtype=float
        )
        x2dot_gradient_constraints = np.asarray(
            x2_data["x2_tilda_fixed_direct_gradient_dot_m_s2"], dtype=float
        )

    if not (
        time_s.shape
        == x1_tilde_strong.shape
        == x2_tilde_constraints.shape
        == x2dot_tilde_constraints.shape
        == contact.shape
    ):
        raise RuntimeError("middle-slice arrays are not shape-aligned")
    if not np.allclose(time_s, x2_time_s, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("x1 strong and x2 constraints are not on the same time grid")

    f_residual = (
        mass_kg * x2dot_tilde_constraints
        + damping_n_s_m * x2_tilde_constraints
        + stiffness_n_m * x1_tilde_strong
    )
    inertial_force = mass_kg * x2dot_tilde_constraints
    damping_force = damping_n_s_m * x2_tilde_constraints
    stiffness_force = stiffness_n_m * x1_tilde_strong
    f_actuation = FACT_AVG_N * np.sin(-2.0 * np.pi * f0_hz * time_s)
    f_residual_minus_actuation = f_residual - f_actuation

    n = int(time_s.size)
    margin_points = int(round(MARGIN_FRACTION_PER_SIDE * n))
    display_start = margin_points
    display_stop = n - margin_points
    if display_start >= display_stop:
        raise RuntimeError("margin removes the whole middle slice")
    display_slice = slice(display_start, display_stop)
    display_mask = np.zeros(n, dtype=bool)
    display_mask[display_slice] = True

    np.savez_compressed(
        OUTPUT_NPZ,
        time_s=time_s,
        time_us=time_us,
        x1_tilde_strong_m=x1_tilde_strong,
        x2_tilde_constraints_m_s=x2_tilde_constraints,
        x2dot_tilde_constraints_m_s2=x2dot_tilde_constraints,
        x2dot_gradient_constraints_m_s2=x2dot_gradient_constraints,
        f_residual_N=f_residual,
        f_actuation_N=f_actuation,
        f_residual_minus_actuation_N=f_residual_minus_actuation,
        inertial_force_N=inertial_force,
        damping_force_N=damping_force,
        stiffness_force_N=stiffness_force,
        contact=contact,
        transition_idx=transition_idx,
        display_mask=display_mask,
        margin_fraction_per_side=np.asarray(MARGIN_FRACTION_PER_SIDE, dtype=float),
        margin_points_per_side=np.asarray(margin_points, dtype=int),
        display_start_index=np.asarray(display_start, dtype=int),
        display_stop_index_exclusive=np.asarray(display_stop, dtype=int),
        m_kg=np.asarray(mass_kg, dtype=float),
        c_N_s_m=np.asarray(damping_n_s_m, dtype=float),
        k_N_m=np.asarray(stiffness_n_m, dtype=float),
        f0_Hz=np.asarray(f0_hz, dtype=float),
        Q=np.asarray(q_factor, dtype=float),
    )

    summary = {
        "definition": "F_residual = m*x2dot_tilde_constraints + c*x2_tilde_constraints + k*x1_tilde_strong on the middle slice",
        "actuation_subtraction_definition": "F_act = Fact_avg * sin(-2*pi*f_index109*t), no fitted phase shift; fifth subplot shows F_residual - F_act",
        "Fact_avg_N": FACT_AVG_N,
        "Fact_avg_nN": FACT_AVG_N * 1.0e9,
        "index109_drive_frequency_Hz": f0_hz,
        "x1_source": str(X1_STRONG_NPZ),
        "x2_source": str(X2_CONSTRAINTS_NPZ),
        "x2_constraints_definition": "x2_tilda_fixed: x2 raw is strongly GP-smoothed into x2_tilda_pure; external transition slope jumps are imposed as one-sided contact boundary slopes; each contact segment is replaced by a cubic Hermite curve constrained by those slopes and pure-GP endpoint values; noncontact remains x2_tilda_pure.",
        "output_npz": str(OUTPUT_NPZ),
        "output_png": str(OUTPUT_PNG),
        "time_range_us_full_middle": [float(time_us[0]), float(time_us[-1])],
        "sample_count_full_middle": n,
        "margin_fraction_per_side": MARGIN_FRACTION_PER_SIDE,
        "margin_points_per_side": margin_points,
        "display_index_range": [display_start, display_stop],
        "display_time_range_us": [
            float(time_us[display_start]),
            float(time_us[display_stop - 1]),
        ],
        "cantilever": {
            "m_kg": mass_kg,
            "c_N_s_m": damping_n_s_m,
            "k_N_m": stiffness_n_m,
            "f0_Hz": f0_hz,
            "Q": q_factor,
        },
        "full_middle_stats": {
            "x1_tilde_strong_nm": _stats(x1_tilde_strong * 1.0e9),
            "x2_tilde_constraints_m_s": _stats(x2_tilde_constraints),
            "x2dot_tilde_constraints_m_s2": _stats(x2dot_tilde_constraints),
            "f_residual_nN": _stats(f_residual * 1.0e9),
            "f_actuation_nN": _stats(f_actuation * 1.0e9),
            "f_residual_minus_actuation_nN": _stats(
                f_residual_minus_actuation * 1.0e9
            ),
        },
        "display_region_stats": {
            "x1_tilde_strong_nm": _stats(x1_tilde_strong[display_slice] * 1.0e9),
            "x2_tilde_constraints_m_s": _stats(x2_tilde_constraints[display_slice]),
            "x2dot_tilde_constraints_m_s2": _stats(
                x2dot_tilde_constraints[display_slice]
            ),
            "f_residual_nN": _stats(f_residual[display_slice] * 1.0e9),
            "f_actuation_nN": _stats(f_actuation[display_slice] * 1.0e9),
            "f_residual_minus_actuation_nN": _stats(
                f_residual_minus_actuation[display_slice] * 1.0e9
            ),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(12.0, 11.0),
        sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    t_plot = time_us[display_slice]
    axes[0].plot(
        t_plot,
        x1_tilde_strong[display_slice] * 1.0e9,
        color="black",
        linewidth=1.0,
    )
    axes[0].set_ylabel(r"$\tilde{x}_{1,\mathrm{strong}}$ [nm]")

    axes[1].plot(
        t_plot,
        x2_tilde_constraints[display_slice],
        color="#1f77b4",
        linewidth=1.0,
    )
    axes[1].set_ylabel(r"$\tilde{x}_{2,\mathrm{constraints}}$ [m s$^{-1}$]")

    axes[2].plot(
        t_plot,
        x2dot_tilde_constraints[display_slice],
        color="#2ca02c",
        linewidth=1.0,
    )
    axes[2].set_ylabel(r"$\dot{\tilde{x}}_{2,\mathrm{constraints}}$ [m s$^{-2}$]")

    axes[3].plot(
        t_plot,
        f_residual[display_slice] * 1.0e9,
        color="black",
        linewidth=1.0,
    )
    axes[3].set_ylabel(r"$F_{\mathrm{residual}}$ [nN]")

    axes[4].plot(
        t_plot,
        f_residual_minus_actuation[display_slice] * 1.0e9,
        color="black",
        linewidth=1.0,
    )
    axes[4].set_ylabel(r"$F_{\mathrm{residual}}-F_{\mathrm{act}}$ [nN]")
    axes[4].set_xlabel(r"Time [$\mu$s]")

    transition_in_display = [
        int(idx)
        for idx in np.asarray(transition_idx, dtype=int)
        if display_start <= int(idx) < display_stop
    ]
    if transition_in_display:
        transition_in_display_arr = np.asarray(transition_in_display, dtype=int)
        transition_time_us = time_us[transition_in_display_arr]
        point_series = [
            x1_tilde_strong * 1.0e9,
            x2_tilde_constraints,
            x2dot_tilde_constraints,
            f_residual * 1.0e9,
            f_residual_minus_actuation * 1.0e9,
        ]
        for axis, values in zip(axes, point_series):
            for t_transition in transition_time_us:
                axis.axvline(
                    t_transition,
                    color="black",
                    linestyle="--",
                    linewidth=0.65,
                    alpha=0.55,
                    zorder=1,
                )
            axis.scatter(
                transition_time_us,
                values[transition_in_display_arr],
                s=22,
                color="black",
                zorder=5,
                label="transition point",
            )
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            axes[-1].legend(loc="upper right")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.0)

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_NPZ}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_PNG}")
    print(
        "Display time range: "
        f"{time_us[display_start]:.3f}-{time_us[display_stop - 1]:.3f} us"
    )
    print(
        "Display F_residual range: "
        f"{np.min(f_residual[display_slice]) * 1.0e9:.6g} to "
        f"{np.max(f_residual[display_slice]) * 1.0e9:.6g} nN"
    )
    print(
        "Display F_residual - F_act range: "
        f"{np.min(f_residual_minus_actuation[display_slice]) * 1.0e9:.6g} to "
        f"{np.max(f_residual_minus_actuation[display_slice]) * 1.0e9:.6g} nN"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
