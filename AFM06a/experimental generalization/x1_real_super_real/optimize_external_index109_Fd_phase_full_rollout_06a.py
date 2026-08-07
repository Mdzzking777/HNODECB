"""Fit only AFM06a drive amplitude and phase to external index109 x1."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

full_rollout = importlib.import_module("run_external_index109_full_rollout_06a_framework")


BASE_ROLLOUT_NPZ = SCRIPT_DIR / "external_index109_full_rollout_06a_framework_x1.npz"
OUTPUT_STEM = "external_index109_full_rollout_optimized_Fd_phase_06a"
OPTIMIZATION_STRIDE = 20  # 80 ns; about 76 samples per actuation period.
FD_BOUNDS_NN = (3.0, 8.0)
PHASE_BOUNDS_RAD = (-np.pi, np.pi)


def _rhs(
    t: float,
    state: np.ndarray,
    force_x1_grid_m: np.ndarray,
    force_grid_N: np.ndarray,
    parameters: dict[str, float],
    fd_N: float,
    phase_rad: float,
) -> np.ndarray:
    x1 = float(state[0])
    x2 = float(state[1])
    force_N = float(np.interp(x1, force_x1_grid_m, force_grid_N))
    x2dot = (
        fd_N * np.sin(parameters["omega0"] * t + phase_rad)
        - parameters["c"] * x2
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
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = np.empty((2, time_s.size), dtype=np.float64)
    trajectory[:, 0] = initial_state
    for index in range(time_s.size - 1):
        t = float(time_s[index])
        h = float(time_s[index + 1] - time_s[index])
        y = trajectory[:, index]
        k1 = _rhs(t, y, force_x1_grid_m, force_grid_N, parameters, fd_N, phase_rad)
        k2 = _rhs(
            t + 0.5 * h,
            y + 0.5 * h * k1,
            force_x1_grid_m,
            force_grid_N,
            parameters,
            fd_N,
            phase_rad,
        )
        k3 = _rhs(
            t + 0.5 * h,
            y + 0.5 * h * k2,
            force_x1_grid_m,
            force_grid_N,
            parameters,
            fd_N,
            phase_rad,
        )
        k4 = _rhs(
            t + h,
            y + h * k3,
            force_x1_grid_m,
            force_grid_N,
            parameters,
            fd_N,
            phase_rad,
        )
        trajectory[:, index + 1] = y + (h / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
    force_N = np.interp(trajectory[0], force_x1_grid_m, force_grid_N)
    return trajectory, force_N


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray) -> float:
    numerator = float(np.sqrt(np.mean(np.square(prediction - truth))))
    denominator = max(float(np.sqrt(np.mean(np.square(truth)))), 1.0e-30)
    return 100.0 * numerator / denominator


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _plot(
    time_s: np.ndarray,
    true_x1_m: np.ndarray,
    predicted_x1_m: np.ndarray,
    fd_nN: float,
    phase_rad: float,
    output_path: Path,
) -> None:
    period_s = full_rollout.pointwise.PERIOD_S
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
            label="optimized rollout",
            zorder=2,
        )
        axis.set_title(label)
        axis.set_xlabel(r"Time [$\mu$s]")
        axis.set_ylabel(r"$x_1$ [nm]")
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.0)
    axes[0].legend(loc="upper right")
    fig.suptitle(
        rf"$F_d={fd_nN:.6f}$ nN, $\phi_d={phase_rad:.6f}$ rad",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if not BASE_ROLLOUT_NPZ.is_file():
        raise FileNotFoundError(
            f"baseline rollout data is missing; run the full-rollout script first: {BASE_ROLLOUT_NPZ}"
        )
    with np.load(BASE_ROLLOUT_NPZ) as payload:
        time_full = np.asarray(payload["time_s"], dtype=np.float64)
        true_x1_full = np.asarray(payload["external_true_x1_m"], dtype=np.float64)
        true_x2_full = np.asarray(payload["external_true_x2_m_s"], dtype=np.float64)
        force_x1_grid_m = np.asarray(payload["force_lookup_x1_m"], dtype=np.float64)
        force_grid_N = np.asarray(payload["force_lookup_fts_N"], dtype=np.float64)
    parameters = full_rollout.one_step._load_rhs_parameters()
    initial_state = np.asarray((true_x1_full[0], true_x2_full[0]), dtype=np.float64)

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
        fd_N = float(theta[0]) * 1.0e-9
        phase_rad = float(theta[1])
        trajectory, _ = _rollout(
            time_fit,
            initial_state,
            force_x1_grid_m,
            force_grid_N,
            parameters,
            fd_N,
            phase_rad,
        )
        return float(np.mean(np.square(trajectory[0] - true_x1_fit)) / truth_energy)

    global_result = differential_evolution(
        objective,
        bounds=(FD_BOUNDS_NN, PHASE_BOUNDS_RAD),
        seed=20260804,
        popsize=8,
        maxiter=10,
        tol=2.0e-4,
        polish=False,
        workers=1,
        updating="immediate",
    )
    local_result = minimize(
        objective,
        np.asarray(global_result.x, dtype=np.float64),
        method="Powell",
        bounds=(FD_BOUNDS_NN, PHASE_BOUNDS_RAD),
        options={"xtol": 1.0e-7, "ftol": 1.0e-9, "maxiter": 80},
    )
    best = local_result if float(local_result.fun) <= float(global_result.fun) else global_result
    best_fd_nN = float(best.x[0])
    best_phase_rad = float(best.x[1])
    trajectory, predicted_force_N = _rollout(
        time_full,
        initial_state,
        force_x1_grid_m,
        force_grid_N,
        parameters,
        best_fd_nN * 1.0e-9,
        best_phase_rad,
    )
    predicted_x1_m = trajectory[0]
    predicted_x2_m_s = trajectory[1]

    output_npz = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
    np.savez_compressed(
        output_npz,
        time_s=time_full,
        external_true_x1_m=true_x1_full,
        external_true_x2_m_s=true_x2_full,
        predicted_x1_m=predicted_x1_m,
        predicted_x2_m_s=predicted_x2_m_s,
        predicted_fts_N=predicted_force_N,
        initial_state=initial_state,
        optimized_Fd_N=np.asarray(best_fd_nN * 1.0e-9),
        optimized_phase_rad=np.asarray(best_phase_rad),
    )
    _plot(
        time_full,
        true_x1_full,
        predicted_x1_m,
        best_fd_nN,
        best_phase_rad,
        output_png,
    )

    summary = {
        "objective": "full-span normalized mean squared x1 trajectory error",
        "free_parameters": ["Fd_N", "drive_phase_rad"],
        "fixed_parameters": {
            "m_kg": parameters["m"],
            "c_N_s_m": parameters["c"],
            "k_N_m": parameters["k"],
            "omega_d_rad_s": parameters["omega0"],
            "trained_kan_force_lookup": str(BASE_ROLLOUT_NPZ),
        },
        "optimization_grid": {
            "stride": OPTIMIZATION_STRIDE,
            "dt_s": float(np.median(np.diff(time_fit))),
            "sample_count": int(time_fit.size),
        },
        "bounds": {
            "Fd_nN": list(FD_BOUNDS_NN),
            "phase_rad": list(PHASE_BOUNDS_RAD),
        },
        "evaluation_count": evaluation_count,
        "global_result": {
            "x": np.asarray(global_result.x, dtype=float).tolist(),
            "fun": float(global_result.fun),
            "success": bool(global_result.success),
            "message": str(global_result.message),
        },
        "local_result": {
            "x": np.asarray(local_result.x, dtype=float).tolist(),
            "fun": float(local_result.fun),
            "success": bool(local_result.success),
            "message": str(local_result.message),
        },
        "optimized": {
            "Fd_N": best_fd_nN * 1.0e-9,
            "Fd_nN": best_fd_nN,
            "phase_rad": best_phase_rad,
            "phase_deg": float(np.degrees(best_phase_rad)),
        },
        "full_resolution_metrics": {
            "x1_relative_rmse_pct": _relative_rmse_pct(predicted_x1_m, true_x1_full),
            "x2_relative_rmse_pct": _relative_rmse_pct(predicted_x2_m_s, true_x2_full),
            "x1_error_nm": _stats((predicted_x1_m - true_x1_full) * 1.0e9),
            "predicted_x1_nm": _stats(predicted_x1_m * 1.0e9),
            "predicted_fts_nN": _stats(predicted_force_N * 1.0e9),
        },
        "outputs": {"npz": str(output_npz), "png": str(output_png)},
    }
    _write_json(output_json, summary)
    print(f"Saved: {output_npz}")
    print(f"Saved: {output_json}")
    print(f"Saved: {output_png}")
    print(f"evaluations={evaluation_count}")
    print(f"Fd={best_fd_nN:.12g} nN")
    print(f"phase={best_phase_rad:.12g} rad ({np.degrees(best_phase_rad):.9g} deg)")
    print(
        "full-resolution x1 relative RMSE="
        f"{summary['full_resolution_metrics']['x1_relative_rmse_pct']:.9g}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
