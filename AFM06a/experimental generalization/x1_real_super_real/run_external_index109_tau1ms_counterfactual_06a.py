"""Counterfactual AFM06a rollout with a 1 ms amplitude-relaxation time."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

optimizer = importlib.import_module(
    "optimize_external_index109_Fd_phase_full_rollout_06a"
)
full_rollout = importlib.import_module(
    "run_external_index109_full_rollout_06a_framework"
)

BASE_NPZ = SCRIPT_DIR / "external_index109_full_rollout_06a_framework_x1.npz"
OPTIMIZED_NPZ = (
    SCRIPT_DIR / "external_index109_full_rollout_optimized_Fd_phase_06a.npz"
)
OUTPUT_STEM = "external_index109_full_rollout_tau1ms_06a"
TAU_S = 1.0e-3


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray) -> float:
    error_rms = float(np.sqrt(np.mean(np.square(prediction - truth))))
    truth_rms = max(float(np.sqrt(np.mean(np.square(truth)))), 1.0e-30)
    return 100.0 * error_rms / truth_rms


def _half_amplitude(values: np.ndarray) -> float:
    return 0.5 * float(np.max(values) - np.min(values))


def main() -> int:
    with np.load(BASE_NPZ) as payload:
        time_s = np.asarray(payload["time_s"], dtype=np.float64)
        true_x1_m = np.asarray(payload["external_true_x1_m"], dtype=np.float64)
        true_x2_m_s = np.asarray(payload["external_true_x2_m_s"], dtype=np.float64)
        force_x1_grid_m = np.asarray(payload["force_lookup_x1_m"], dtype=np.float64)
        force_grid_N = np.asarray(payload["force_lookup_fts_N"], dtype=np.float64)
    with np.load(OPTIMIZED_NPZ) as payload:
        fd_N = float(payload["optimized_Fd_N"])
        phase_rad = float(payload["optimized_phase_rad"])

    parameters = dict(full_rollout.one_step._load_rhs_parameters())
    original_c = float(parameters["c"])
    parameters["c"] = 2.0 * float(parameters["m"]) / TAU_S
    effective_q = float(parameters["m"] * parameters["omega0"] / parameters["c"])
    initial_state = np.asarray((true_x1_m[0], true_x2_m_s[0]), dtype=np.float64)
    trajectory, predicted_force_N = optimizer._rollout(
        time_s,
        initial_state,
        force_x1_grid_m,
        force_grid_N,
        parameters,
        fd_N,
        phase_rad,
    )
    predicted_x1_m = trajectory[0]

    period_s = 2.0 * np.pi / float(parameters["omega0"])
    centers = (0.10 * time_s[-1], 0.50 * time_s[-1], 0.90 * time_s[-1])
    panels: list[tuple[str, np.ndarray]] = [
        ("Full time-span", np.arange(time_s.size, dtype=int))
    ]
    for label, center in zip(
        ("Early window", "Middle window", "Tail window"), centers, strict=True
    ):
        panels.append(
            (
                label,
                np.flatnonzero(
                    (time_s >= center - period_s) & (time_s <= center + period_s)
                ),
            )
        )

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2))
    time_us = time_s * 1.0e6
    for axis, (label, indices) in zip(axes, panels, strict=True):
        draw = indices
        if label == "Full time-span" and indices.size > 20_000:
            draw = indices[np.linspace(0, indices.size - 1, 20_000, dtype=int)]
        axis.plot(
            time_us[draw],
            true_x1_m[draw] * 1.0e9,
            color="black",
            linewidth=1.15,
            label="external true $x_1$",
            zorder=1,
        )
        axis.plot(
            time_us[draw],
            predicted_x1_m[draw] * 1.0e9,
            color="#d62728",
            linewidth=0.75,
            label=r"rollout, $\tau=1\,\mathrm{ms}$",
            zorder=2,
        )
        axis.set_title(label)
        axis.set_xlabel(r"Time [$\mu$s]")
        axis.set_ylabel(r"$x_1$ [nm]")
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.0)
    axes[0].legend(loc="upper right")
    fig.suptitle(
        rf"$\tau=1$ ms, $Q_{{\mathrm{{eff}}}}={effective_q:.3f}$, "
        rf"$F_d={fd_N * 1.0e9:.6f}$ nN, $\phi_d={phase_rad:.6f}$ rad",
        y=1.02,
    )
    fig.tight_layout()
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    output_npz = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    np.savez_compressed(
        output_npz,
        time_s=time_s,
        external_true_x1_m=true_x1_m,
        external_true_x2_m_s=true_x2_m_s,
        predicted_x1_m=predicted_x1_m,
        predicted_x2_m_s=trajectory[1],
        predicted_fts_N=predicted_force_N,
        tau_s=np.asarray(TAU_S),
        effective_Q=np.asarray(effective_q),
        damping_c_N_s_m=np.asarray(parameters["c"]),
        Fd_N=np.asarray(fd_N),
        phase_rad=np.asarray(phase_rad),
    )

    amplitudes: dict[str, dict[str, float]] = {}
    for label, center in zip(("early", "middle", "tail"), centers, strict=True):
        mask = np.abs(time_s - center) <= period_s
        amplitudes[label] = {
            "external_true_half_amplitude_nm": _half_amplitude(true_x1_m[mask]) * 1.0e9,
            "predicted_half_amplitude_nm": _half_amplitude(predicted_x1_m[mask]) * 1.0e9,
        }
    summary = {
        "counterfactual": "tau = 1 ms by changing c only",
        "kept_fixed": ["omega_d", "k", "m", "trained KAN", "Fd", "drive phase"],
        "tau_s": TAU_S,
        "original_c_N_s_m": original_c,
        "counterfactual_c_N_s_m": float(parameters["c"]),
        "effective_Q": effective_q,
        "x1_relative_rmse_pct": _relative_rmse_pct(predicted_x1_m, true_x1_m),
        "window_half_amplitudes": amplitudes,
        "outputs": {"npz": str(output_npz), "png": str(output_png)},
    }
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {output_png}")
    print(f"Saved: {output_npz}")
    print(f"Saved: {output_json}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
