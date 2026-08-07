from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SILICON_DIR = SCRIPT_DIR.parents[1]
MERGED_X2_NPZ = SCRIPT_DIR / "silicon_middle_x2_raw_global_gp_smooth_with_contact_template.npz"
CONTACT_JUDGE_NPZ = (
    SILICON_DIR
    / "GP"
    / "middle"
    / "x1 for contact judge"
    / "silicon_gp_index109_mid5000_490_510us_x1_contact_judge.npz"
)

OUTPUT_NPZ = SCRIPT_DIR / "silicon_middle_fresidual_from_x1_x2_tilde_contact_template.npz"
OUTPUT_JSON = SCRIPT_DIR / "silicon_middle_fresidual_from_x1_x2_tilde_contact_template_summary.json"
OUTPUT_PNG = SCRIPT_DIR / "silicon_middle_fresidual_from_x1_x2_tilde_contact_template.png"

# Cantilever constants used by the external silicon processing notebook/script.
# The notebook states: Silicon cantilever NCLR, NanoWorld AG, k=22.68 N/m,
# f0=164.52 kHz, Q=428. Then m and c are derived from k = m*omega0^2 and
# c = m*omega0/Q.
CANTILEVER_STIFFNESS_N_M = 22.68
CANTILEVER_RESONANCE_HZ = 164.52e3
CANTILEVER_QUALITY_FACTOR = 428.0


def _transition_type(contact: np.ndarray, idx: int) -> str:
    before = bool(contact[idx - 1]) if idx > 0 else False
    after = bool(contact[idx]) if idx < contact.size else False
    if (not before) and after:
        return "N->C"
    if before and (not after):
        return "C->N"
    return "unknown"


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> int:
    with np.load(MERGED_X2_NPZ, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        time_us = np.asarray(data["time_us"], dtype=float)
        x2_tilde = np.asarray(
            data["x2_background_plus_contact_template_mean_tilde_m_s"], dtype=float
        )
        x2dot_tilde = np.asarray(
            data["x2_background_plus_contact_template_mean_tilde_dot_m_s2"], dtype=float
        )
        contact = np.asarray(data["contact"], dtype=bool)
        transition_idx = np.asarray(data["transition_idx"], dtype=int)

    with np.load(CONTACT_JUDGE_NPZ, allow_pickle=False) as data:
        x1_time_s = np.asarray(data["time_s"], dtype=float)
        x1_tilde = np.asarray(data["x1_tilde_m"], dtype=float)

    if time_s.shape != x1_time_s.shape or not np.allclose(time_s, x1_time_s, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("x1_tilde and merged x2_tilde are not on the same middle-slice time grid")

    omega0 = 2.0 * np.pi * CANTILEVER_RESONANCE_HZ
    mass_kg = CANTILEVER_STIFFNESS_N_M / (omega0 * omega0)
    damping_n_s_m = mass_kg * omega0 / CANTILEVER_QUALITY_FACTOR

    inertial_force = mass_kg * x2dot_tilde
    damping_force = damping_n_s_m * x2_tilde
    stiffness_force = CANTILEVER_STIFFNESS_N_M * x1_tilde
    f_residual = inertial_force + damping_force + stiffness_force

    np.savez_compressed(
        OUTPUT_NPZ,
        time_s=time_s,
        time_us=time_us,
        x1_tilde_m=x1_tilde,
        x2_tilde_m_s=x2_tilde,
        x2dot_tilde_m_s2=x2dot_tilde,
        contact=contact,
        transition_idx=transition_idx,
        inertial_force_N=inertial_force,
        damping_force_N=damping_force,
        stiffness_force_N=stiffness_force,
        f_residual_N=f_residual,
        k_N_m=CANTILEVER_STIFFNESS_N_M,
        f0_Hz=CANTILEVER_RESONANCE_HZ,
        omega0_rad_s=omega0,
        Q=CANTILEVER_QUALITY_FACTOR,
        m_kg=mass_kg,
        c_N_s_m=damping_n_s_m,
    )

    summary = {
        "definition": "F_residual = m*x2dot_tilde + c*x2_tilde + k*x1_tilde on the middle slice",
        "merged_x2_source_npz": str(MERGED_X2_NPZ),
        "x1_tilde_source_npz": str(CONTACT_JUDGE_NPZ),
        "output_npz": str(OUTPUT_NPZ),
        "output_png": str(OUTPUT_PNG),
        "time_range_us": [float(time_us[0]), float(time_us[-1])],
        "cantilever": {
            "k_N_m": CANTILEVER_STIFFNESS_N_M,
            "f0_Hz": CANTILEVER_RESONANCE_HZ,
            "omega0_rad_s": omega0,
            "Q": CANTILEVER_QUALITY_FACTOR,
            "m_kg": mass_kg,
            "c_N_s_m": damping_n_s_m,
            "source_note": "k, f0, and Q follow the external silicon notebook/script constants; m and c are derived.",
        },
        "f_residual_stats_N": _stats(f_residual),
        "f_residual_stats_nN": _stats(f_residual * 1.0e9),
        "inertial_force_stats_nN": _stats(inertial_force * 1.0e9),
        "damping_force_stats_nN": _stats(damping_force * 1.0e9),
        "stiffness_force_stats_nN": _stats(stiffness_force * 1.0e9),
        "contact_transition_count": int(transition_idx.size),
        "contact_transitions": [
            {
                "time_us": float(time_us[int(idx)]),
                "type": _transition_type(contact, int(idx)),
            }
            for idx in transition_idx
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    ax.plot(
        time_us,
        f_residual * 1.0e9,
        color="black",
        linewidth=0.95,
        label=r"$F_{\mathrm{residual}}$",
    )
    for idx in transition_idx:
        ax.axvline(time_us[int(idx)], color="0.45", linewidth=0.55, alpha=0.28, zorder=1)
    if transition_idx.size:
        ax.scatter(
            time_us[transition_idx],
            f_residual[transition_idx] * 1.0e9,
            s=18,
            color="black",
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
            label="contact transition",
        )
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylabel(r"$F_{\mathrm{residual}}$ [nN]")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.0)
    ax.legend(loc="upper right")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_NPZ}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"m={mass_kg:.12e} kg, c={damping_n_s_m:.12e} N*s/m, k={CANTILEVER_STIFFNESS_N_M:.12e} N/m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
