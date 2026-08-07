"""Fit drive amplitude, phase, and additional contact damping to index109."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

full_rollout = importlib.import_module("run_external_index109_full_rollout_06a_framework")

BASE_NPZ = SCRIPT_DIR / "external_index109_full_rollout_06a_framework_x1.npz"
OUTPUT_STEM = "external_index109_full_rollout_optimized_Fd_phase_contact_damping_06a"
OPTIMIZATION_STRIDE = 20
FD_BOUNDS_NN = (3.0, 8.0)
PHASE_BOUNDS_RAD = (-np.pi, np.pi)
DELTA_C_RATIO_BOUNDS = (0.0, 20.0)
DIST_M = 100.0e-9
A0_M = 0.165e-9


def _rhs(
    t: float,
    state: np.ndarray,
    force_x1_grid_m: np.ndarray,
    force_grid_N: np.ndarray,
    parameters: dict[str, float],
    fd_N: float,
    phase_rad: float,
    delta_c_N_s_m: float,
) -> np.ndarray:
    x1 = float(state[0])
    x2 = float(state[1])
    force_N = float(np.interp(x1, force_x1_grid_m, force_grid_N))
    in_contact = float(DIST_M + x1 <= A0_M)
    c_eff = parameters["c"] + in_contact * delta_c_N_s_m
    x2dot = (
        fd_N * np.sin(parameters["omega0"] * t + phase_rad)
        - c_eff * x2
        - parameters["k"] * x1
        + force_N
    ) / parameters["m"]
    return np.asarray((x2, x2dot), dtype=np.float64)


def _rollout(
    time_s: np.ndarray,
    initial_state: np.ndarray,
    force_x1_grid_m: np.ndarray,
    force_grid_N: np.ndarray,
    parameters: dict[str, float],
    fd_N: float,
    phase_rad: float,
    delta_c_N_s_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trajectory = np.empty((2, time_s.size), dtype=np.float64)
    trajectory[:, 0] = initial_state
    for index in range(time_s.size - 1):
        t = float(time_s[index])
        h = float(time_s[index + 1] - time_s[index])
        y = trajectory[:, index]
        args = (
            force_x1_grid_m,
            force_grid_N,
            parameters,
            fd_N,
            phase_rad,
            delta_c_N_s_m,
        )
        k1 = _rhs(t, y, *args)
        k2 = _rhs(t + 0.5 * h, y + 0.5 * h * k1, *args)
        k3 = _rhs(t + 0.5 * h, y + 0.5 * h * k2, *args)
        k4 = _rhs(t + h, y + h * k3, *args)
        trajectory[:, index + 1] = y + (h / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
    force_N = np.interp(trajectory[0], force_x1_grid_m, force_grid_N)
    contact = DIST_M + trajectory[0] <= A0_M
    return trajectory, force_N, contact


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray) -> float:
    error_rms = float(np.sqrt(np.mean(np.square(prediction - truth))))
    truth_rms = max(float(np.sqrt(np.mean(np.square(truth)))), 1.0e-30)
    return 100.0 * error_rms / truth_rms


def _half_amplitude(values: np.ndarray) -> float:
    return 0.5 * float(np.max(values) - np.min(values))


def _plot(
    time_s: np.ndarray,
    true_x1_m: np.ndarray,
    predicted_x1_m: np.ndarray,
    fd_nN: float,
    phase_rad: float,
    delta_c_ratio: float,
    output_path: Path,
) -> None:
    period_s = 2.0 * np.pi / 1033709.6467371855
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
            label="three-parameter rollout",
            zorder=2,
        )
        axis.set_title(label)
        axis.set_xlabel(r"Time [$\mu$s]")
        axis.set_ylabel(r"$x_1$ [nm]")
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.0)
    axes[0].legend(loc="upper right")
    fig.suptitle(
        rf"$F_d={fd_nN:.6f}$ nN, $\phi_d={phase_rad:.6f}$ rad, "
        rf"$\Delta c_{{\rm contact}}/c_{{\rm air}}={delta_c_ratio:.6f}$",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    with np.load(BASE_NPZ) as payload:
        time_full = np.asarray(payload["time_s"], dtype=np.float64)
        true_x1_full = np.asarray(payload["external_true_x1_m"], dtype=np.float64)
        true_x2_full = np.asarray(payload["external_true_x2_m_s"], dtype=np.float64)
        force_x1_grid_m = np.asarray(payload["force_lookup_x1_m"], dtype=np.float64)
        force_grid_N = np.asarray(payload["force_lookup_fts_N"], dtype=np.float64)
        baseline_initial_state = np.asarray(payload["initial_state"], dtype=np.float64)

    parameters = dict(full_rollout.one_step._load_rhs_parameters())
    c_air = float(parameters["c"])
    initial_state = baseline_initial_state.reshape(2)
    source_idx = np.arange(0, time_full.size, OPTIMIZATION_STRIDE, dtype=int)
    if source_idx[-1] != time_full.size - 1:
        source_idx = np.concatenate((source_idx, [time_full.size - 1]))
    time_fit = time_full[source_idx]
    true_x1_fit = true_x1_full[source_idx]
    truth_energy = max(float(np.mean(np.square(true_x1_fit))), 1.0e-30)
    evaluation_count = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluation_count
        evaluation_count += 1
        trajectory, _, _ = _rollout(
            time_fit,
            initial_state,
            force_x1_grid_m,
            force_grid_N,
            parameters,
            float(theta[0]) * 1.0e-9,
            float(theta[1]),
            float(theta[2]) * c_air,
        )
        return float(np.mean(np.square(trajectory[0] - true_x1_fit)) / truth_energy)

    bounds = (FD_BOUNDS_NN, PHASE_BOUNDS_RAD, DELTA_C_RATIO_BOUNDS)
    global_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=20260804,
        popsize=8,
        maxiter=15,
        tol=2.0e-4,
        polish=False,
        workers=1,
        updating="immediate",
    )
    local_result = minimize(
        objective,
        np.asarray(global_result.x, dtype=np.float64),
        method="Powell",
        bounds=bounds,
        options={"xtol": 1.0e-7, "ftol": 1.0e-9, "maxiter": 100},
    )
    best = local_result if float(local_result.fun) <= float(global_result.fun) else global_result
    best_fd_nN = float(best.x[0])
    best_phase_rad = float(best.x[1])
    best_delta_c_ratio = float(best.x[2])
    best_delta_c = best_delta_c_ratio * c_air
    trajectory, predicted_force_N, contact = _rollout(
        time_full,
        initial_state,
        force_x1_grid_m,
        force_grid_N,
        parameters,
        best_fd_nN * 1.0e-9,
        best_phase_rad,
        best_delta_c,
    )

    output_npz = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
    np.savez_compressed(
        output_npz,
        time_s=time_full,
        external_true_x1_m=true_x1_full,
        external_true_x2_m_s=true_x2_full,
        predicted_x1_m=trajectory[0],
        predicted_x2_m_s=trajectory[1],
        predicted_fts_N=predicted_force_N,
        predicted_contact_mask=contact,
        optimized_Fd_N=np.asarray(best_fd_nN * 1.0e-9),
        optimized_phase_rad=np.asarray(best_phase_rad),
        optimized_delta_c_contact_N_s_m=np.asarray(best_delta_c),
    )
    _plot(
        time_full,
        true_x1_full,
        trajectory[0],
        best_fd_nN,
        best_phase_rad,
        best_delta_c_ratio,
        output_png,
    )

    period_s = 2.0 * np.pi / float(parameters["omega0"])
    centers = (0.10 * time_full[-1], 0.50 * time_full[-1], 0.90 * time_full[-1])
    amplitudes = {}
    for label, center in zip(("early", "middle", "tail"), centers, strict=True):
        mask = np.abs(time_full - center) <= period_s
        amplitudes[label] = {
            "external_true_half_amplitude_nm": _half_amplitude(true_x1_full[mask]) * 1.0e9,
            "predicted_half_amplitude_nm": _half_amplitude(trajectory[0, mask]) * 1.0e9,
        }
    summary = {
        "objective": "full-span normalized mean squared x1 trajectory error",
        "free_parameters": ["Fd_N", "drive_phase_rad", "delta_c_contact_N_s_m"],
        "contact_criterion": "Z + predicted_x1 <= a0",
        "contact_threshold_x1_nm": (A0_M - DIST_M) * 1.0e9,
        "initial_state_SI": {
            "x1_m": float(initial_state[0]),
            "x2_m_s": float(initial_state[1]),
            "x2_policy": "inherited from baseline periodic fit to external x1 raw",
        },
        "fixed_c_air_N_s_m": c_air,
        "bounds": {
            "Fd_nN": list(FD_BOUNDS_NN),
            "phase_rad": list(PHASE_BOUNDS_RAD),
            "delta_c_contact_over_c_air": list(DELTA_C_RATIO_BOUNDS),
        },
        "evaluation_count": evaluation_count,
        "optimized": {
            "Fd_N": best_fd_nN * 1.0e-9,
            "Fd_nN": best_fd_nN,
            "phase_rad": best_phase_rad,
            "phase_deg": float(np.degrees(best_phase_rad)),
            "delta_c_contact_N_s_m": best_delta_c,
            "delta_c_contact_over_c_air": best_delta_c_ratio,
            "c_contact_total_N_s_m": c_air + best_delta_c,
        },
        "full_resolution_metrics": {
            "x1_relative_rmse_pct": _relative_rmse_pct(trajectory[0], true_x1_full),
            "x2_relative_rmse_pct": _relative_rmse_pct(trajectory[1], true_x2_full),
            "predicted_contact_fraction": float(np.mean(contact)),
            "window_half_amplitudes": amplitudes,
        },
        "outputs": {"npz": str(output_npz), "png": str(output_png)},
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {output_npz}")
    print(f"Saved: {output_png}")
    print(f"Saved: {output_json}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
