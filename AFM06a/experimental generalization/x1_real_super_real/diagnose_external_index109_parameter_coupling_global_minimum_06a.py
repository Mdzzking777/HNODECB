"""Diagnose three-parameter coupling and repeatability of the external rollout fit."""

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

fit_module = importlib.import_module(
    "optimize_external_index109_Fd_phase_contact_damping_06a"
)
full_rollout = importlib.import_module(
    "run_external_index109_full_rollout_06a_framework"
)

OUTPUT_STEM = "external_index109_three_parameter_coupling_global_search_06a"
FIT_STRIDE = 40
SEEDS = (20260804, 20260805, 20260806, 20260807)
BOUNDS = (
    fit_module.FD_BOUNDS_NN,
    fit_module.PHASE_BOUNDS_RAD,
    fit_module.DELTA_C_RATIO_BOUNDS,
)
PARAMETER_NAMES = (r"$F_d$", r"$\phi_d$", r"$\Delta c/c_{air}$")


def _rollout_residual(
    theta: np.ndarray,
    time_s: np.ndarray,
    truth_x1_m: np.ndarray,
    initial_state: np.ndarray,
    force_x1_grid_m: np.ndarray,
    force_grid_N: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    trajectory, _, _ = fit_module._rollout(
        time_s,
        initial_state,
        force_x1_grid_m,
        force_grid_N,
        parameters,
        float(theta[0]) * 1.0e-9,
        float(theta[1]),
        float(theta[2]) * float(parameters["c"]),
    )
    truth_rms = max(float(np.sqrt(np.mean(np.square(truth_x1_m)))), 1.0e-30)
    return (trajectory[0] - truth_x1_m) / truth_rms


def _objective(*args: object) -> float:
    residual = _rollout_residual(*args)
    return float(np.mean(np.square(residual)))


def _sensitivity_diagnostics(
    theta: np.ndarray,
    residual_args: tuple[object, ...],
) -> dict[str, object]:
    lower = np.asarray([bound[0] for bound in BOUNDS], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in BOUNDS], dtype=np.float64)
    widths = upper - lower
    step_fraction = 2.0e-3
    columns = []
    for index in range(3):
        step = step_fraction * widths[index]
        plus = theta.copy()
        minus = theta.copy()
        plus[index] = min(upper[index], theta[index] + step)
        minus[index] = max(lower[index], theta[index] - step)
        derivative = (
            _rollout_residual(plus, *residual_args)
            - _rollout_residual(minus, *residual_args)
        ) / ((plus[index] - minus[index]) / widths[index])
        columns.append(derivative)
    jacobian = np.column_stack(columns)
    column_norms = np.linalg.norm(jacobian, axis=0)
    safe_norms = np.maximum(column_norms, 1.0e-30)
    cosine = (jacobian.T @ jacobian) / np.outer(safe_norms, safe_norms)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    condition_number = float(singular_values[0] / max(singular_values[-1], 1.0e-30))
    information = jacobian.T @ jacobian
    covariance = np.linalg.pinv(information, rcond=1.0e-12)
    covariance_scale = np.sqrt(np.maximum(np.diag(covariance), 1.0e-30))
    parameter_correlation = covariance / np.outer(covariance_scale, covariance_scale)
    return {
        "bound_scaled_jacobian_column_norms": column_norms.tolist(),
        "sensitivity_column_cosine": cosine.tolist(),
        "singular_values": singular_values.tolist(),
        "condition_number": condition_number,
        "local_parameter_correlation": parameter_correlation.tolist(),
    }


def _plot(
    runs: list[dict[str, object]],
    sensitivity: dict[str, object],
    output_path: Path,
) -> None:
    final = np.asarray([run["local_x"] for run in runs], dtype=np.float64)
    objective = np.asarray([run["local_fun"] for run in runs], dtype=np.float64)
    correlation = np.asarray(sensitivity["local_parameter_correlation"], dtype=np.float64)
    singular_values = np.asarray(sensitivity["singular_values"], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    run_axis = axes[0, 0]
    run_axis.semilogy(np.arange(1, len(runs) + 1), objective, "o-", color="#1f77b4")
    run_axis.set_xlabel("Independent search run")
    run_axis.set_ylabel("Final normalized MSE")
    run_axis.set_title("Independent-search repeatability")
    run_axis.grid(True, alpha=0.25)

    scatter = axes[0, 1].scatter(
        final[:, 0], final[:, 2], c=objective, cmap="viridis_r", s=70
    )
    axes[0, 1].set_xlabel(r"$F_d$ [nN]")
    axes[0, 1].set_ylabel(r"$\Delta c_{contact}/c_{air}$")
    axes[0, 1].set_title("Converged parameter pairs")
    fig.colorbar(scatter, ax=axes[0, 1], label="Normalized MSE")

    image = axes[1, 0].imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1, 0].set_xticks(range(3), PARAMETER_NAMES)
    axes[1, 0].set_yticks(range(3), PARAMETER_NAMES)
    axes[1, 0].set_title("Local parameter correlation")
    for row in range(3):
        for column in range(3):
            axes[1, 0].text(
                column,
                row,
                f"{correlation[row, column]:.3f}",
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(image, ax=axes[1, 0])

    axes[1, 1].semilogy(range(1, 4), singular_values, "o-", color="#d62728")
    axes[1, 1].set_xticks(range(1, 4))
    axes[1, 1].set_xlabel("Singular-value index")
    axes[1, 1].set_ylabel("Singular value")
    axes[1, 1].set_title(
        f"Residual-Jacobian spectrum (condition={sensitivity['condition_number']:.3g})"
    )
    axes[1, 1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    model, _, _, _, _ = full_rollout.pointwise._restore_stage2_model(
        full_rollout.ARCHIVE_DIR
    )
    time_full, true_x1_full, mat_source = full_rollout.pointwise._load_external_x1_raw(
        None
    )
    true_x2_full = full_rollout.one_step._load_external_x2_raw(mat_source)
    parameters = dict(full_rollout.one_step._load_rhs_parameters())
    fitted_x2_t0, _ = full_rollout._fit_initial_velocity_from_x1(
        time_full, true_x1_full, float(parameters["omega0"])
    )
    initial_state = np.asarray(
        (true_x1_full[0], fitted_x2_t0), dtype=np.float64
    )
    force_x1_grid_m, force_grid_N = full_rollout._build_force_lookup(
        model, true_x1_full
    )

    fit_indices = np.arange(0, time_full.size, FIT_STRIDE, dtype=int)
    if fit_indices[-1] != time_full.size - 1:
        fit_indices = np.concatenate((fit_indices, [time_full.size - 1]))
    time_fit = time_full[fit_indices]
    true_x1_fit = true_x1_full[fit_indices]
    residual_args = (
        time_fit,
        true_x1_fit,
        initial_state,
        force_x1_grid_m,
        force_grid_N,
        parameters,
    )
    objective_args = residual_args

    runs: list[dict[str, object]] = []
    for seed in SEEDS:
        global_result = differential_evolution(
            lambda theta: _objective(theta, *objective_args),
            bounds=BOUNDS,
            seed=seed,
            popsize=6,
            maxiter=8,
            tol=5.0e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        local_result = minimize(
            lambda theta: _objective(theta, *objective_args),
            np.asarray(global_result.x, dtype=np.float64),
            method="Powell",
            bounds=BOUNDS,
            options={"xtol": 2.0e-6, "ftol": 1.0e-8, "maxiter": 60},
        )
        runs.append(
            {
                "seed": seed,
                "global_x": np.asarray(global_result.x, dtype=float).tolist(),
                "global_fun": float(global_result.fun),
                "local_x": np.asarray(local_result.x, dtype=float).tolist(),
                "local_fun": float(local_result.fun),
                "local_success": bool(local_result.success),
                "local_message": str(local_result.message),
            }
        )

    best_run = min(runs, key=lambda item: float(item["local_fun"]))
    best_theta = np.asarray(best_run["local_x"], dtype=np.float64)
    sensitivity = _sensitivity_diagnostics(best_theta, residual_args)

    final_theta = np.asarray([run["local_x"] for run in runs], dtype=np.float64)
    final_objective = np.asarray([run["local_fun"] for run in runs], dtype=np.float64)
    best_objective = float(np.min(final_objective))
    near_best = final_objective <= best_objective * 1.01 + 1.0e-12
    spread = np.ptp(final_theta[near_best], axis=0) if np.any(near_best) else np.zeros(3)
    repeatability = {
        "best_normalized_mse": best_objective,
        "worst_over_best_ratio": float(np.max(final_objective) / best_objective),
        "runs_within_1pct_of_best": int(np.count_nonzero(near_best)),
        "parameter_spread_among_near_best": {
            "Fd_nN": float(spread[0]),
            "phase_rad": float(spread[1]),
            "delta_c_over_c_air": float(spread[2]),
        },
    }
    summary = {
        "scope": (
            "Four independent differential-evolution searches followed by Powell "
            "refinement on the corrected periodic-fit initial state. This supplies "
            "evidence, not a mathematical proof, of global optimality."
        ),
        "fit_stride": FIT_STRIDE,
        "bounds": [list(bound) for bound in BOUNDS],
        "runs": runs,
        "best_theta": {
            "Fd_nN": float(best_theta[0]),
            "phase_rad": float(best_theta[1]),
            "delta_c_over_c_air": float(best_theta[2]),
        },
        "repeatability": repeatability,
        "local_sensitivity": sensitivity,
    }
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}.json"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(runs, sensitivity, output_png)
    print(f"Saved: {output_json}")
    print(f"Saved: {output_png}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
