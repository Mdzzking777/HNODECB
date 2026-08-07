"""Autonomous external-index109 rollout using the current AFM06a framework."""

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
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (REPO_ROOT, EXPERIMENT_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pointwise = importlib.import_module("predict_external_index109_with_trained_kan_06a")
one_step = importlib.import_module("run_external_index109_true_state_reset_one_step_06a")


ARCHIVE_DIR = (
    REPO_ROOT / "AFM06a" / "archive" / "st2l" / "rank1_20260803_183835"
)
OUTPUT_STEM = "external_index109_full_rollout_06a_framework_x1"
LOOKUP_GRID_SIZE = 100_001
LOOKUP_MARGIN_FRACTION = 0.25


def _fit_initial_velocity_from_x1(
    time_s: np.ndarray,
    x1_m: np.ndarray,
    omega_d_rad_s: float,
) -> tuple[float, dict[str, float]]:
    """Fit the fundamental harmonic of external x1 and differentiate at t0."""
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    x1_m = np.asarray(x1_m, dtype=np.float64).reshape(-1)
    shifted_time_s = time_s - float(time_s[0])
    design = np.column_stack(
        (
            np.sin(omega_d_rad_s * shifted_time_s),
            np.cos(omega_d_rad_s * shifted_time_s),
            np.ones_like(shifted_time_s),
        )
    )
    sin_coefficient_m, cos_coefficient_m, offset_m = np.linalg.lstsq(
        design, x1_m, rcond=None
    )[0]
    fitted_x2_t0_m_s = float(omega_d_rad_s * sin_coefficient_m)
    return fitted_x2_t0_m_s, {
        "sin_coefficient_m": float(sin_coefficient_m),
        "cos_coefficient_m": float(cos_coefficient_m),
        "offset_m": float(offset_m),
        "fundamental_amplitude_m": float(
            np.hypot(sin_coefficient_m, cos_coefficient_m)
        ),
        "fitted_x2_t0_m_s": fitted_x2_t0_m_s,
        "fit_sample_count": int(time_s.size),
    }


@torch.no_grad()
def _build_force_lookup(
    model: torch.nn.Module,
    x1_reference_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(x1_reference_m, dtype=np.float64).reshape(-1)
    lo = float(np.min(reference))
    hi = float(np.max(reference))
    span = max(hi - lo, 1.0e-12)
    grid = np.linspace(
        lo - LOOKUP_MARGIN_FRACTION * span,
        hi + LOOKUP_MARGIN_FRACTION * span,
        LOOKUP_GRID_SIZE,
        dtype=np.float64,
    )
    force = pointwise._predict_fts_N(model, grid, chunk_size=32768)
    return grid, force


def _rhs(
    t: float,
    state: np.ndarray,
    force_x1_grid_m: np.ndarray,
    force_grid_N: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    x1 = float(state[0])
    x2 = float(state[1])
    force_N = float(np.interp(x1, force_x1_grid_m, force_grid_N))
    x2dot = (
        parameters["Fd"] * np.sin(parameters["omega0"] * t)
        - parameters["c"] * x2
        - parameters["k"] * x1
        + force_N
    ) / parameters["m"]
    return np.asarray((x2, x2dot), dtype=np.float64)


def _rk4_rollout(
    time_s: np.ndarray,
    initial_state: np.ndarray,
    force_x1_grid_m: np.ndarray,
    force_grid_N: np.ndarray,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = np.empty((2, time_s.size), dtype=np.float64)
    trajectory[:, 0] = np.asarray(initial_state, dtype=np.float64).reshape(2)
    for index in range(time_s.size - 1):
        t = float(time_s[index])
        h = float(time_s[index + 1] - time_s[index])
        y = trajectory[:, index]
        k1 = _rhs(t, y, force_x1_grid_m, force_grid_N, parameters)
        k2 = _rhs(t + 0.5 * h, y + 0.5 * h * k1, force_x1_grid_m, force_grid_N, parameters)
        k3 = _rhs(t + 0.5 * h, y + 0.5 * h * k2, force_x1_grid_m, force_grid_N, parameters)
        k4 = _rhs(t + h, y + h * k3, force_x1_grid_m, force_grid_N, parameters)
        trajectory[:, index + 1] = y + (h / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        if not np.all(np.isfinite(trajectory[:, index + 1])):
            raise RuntimeError(
                f"nonfinite rollout state at index {index + 1}, t={time_s[index + 1]:.12g} s"
            )
    rollout_force_N = np.interp(trajectory[0], force_x1_grid_m, force_grid_N)
    return trajectory, rollout_force_N


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
    output_path: Path,
) -> None:
    period_s = pointwise.PERIOD_S
    centers = (
        max(float(time_s[0]) + period_s, 0.10 * float(time_s[-1])),
        0.50 * float(time_s[-1]),
        min(float(time_s[-1]) - period_s, 0.90 * float(time_s[-1])),
    )
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
                    (time_s >= center - period_s)
                    & (time_s <= center + period_s)
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
            label="autonomous rollout",
            zorder=2,
        )
        axis.set_title(label)
        axis.set_xlabel(r"Time [$\mu$s]")
        axis.set_ylabel(r"$x_1$ [nm]")
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.0)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    torch.set_num_threads(1)
    model, endpoint, payload, stage2_result, stage1_result = pointwise._restore_stage2_model(
        ARCHIVE_DIR
    )
    normalizer = pointwise._record_training_normalizer(model)
    time_s, true_x1_m, mat_source = pointwise._load_external_x1_raw(None)
    true_x2_m_s = one_step._load_external_x2_raw(mat_source)
    if true_x2_m_s.shape != true_x1_m.shape:
        raise ValueError("external x1 and x2 arrays do not share the same sample axis")
    parameters = one_step._load_rhs_parameters()
    fitted_x2_t0_m_s, periodic_fit = _fit_initial_velocity_from_x1(
        time_s, true_x1_m, float(parameters["omega0"])
    )
    initial_state = np.asarray(
        (true_x1_m[0], fitted_x2_t0_m_s), dtype=np.float64
    )

    force_x1_grid_m, force_grid_N = _build_force_lookup(model, true_x1_m)
    trajectory, predicted_force_N = _rk4_rollout(
        time_s,
        initial_state,
        force_x1_grid_m,
        force_grid_N,
        parameters,
    )
    for _ in range(2):
        outside = (
            float(np.min(trajectory[0])) < float(force_x1_grid_m[0])
            or float(np.max(trajectory[0])) > float(force_x1_grid_m[-1])
        )
        if not outside:
            break
        force_x1_grid_m, force_grid_N = _build_force_lookup(
            model, np.concatenate((true_x1_m, trajectory[0]))
        )
        trajectory, predicted_force_N = _rk4_rollout(
            time_s,
            initial_state,
            force_x1_grid_m,
            force_grid_N,
            parameters,
        )

    predicted_x1_m = trajectory[0]
    predicted_x2_m_s = trajectory[1]
    output_npz = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
    np.savez_compressed(
        output_npz,
        time_s=time_s,
        external_true_x1_m=true_x1_m,
        external_true_x2_m_s=true_x2_m_s,
        predicted_x1_m=predicted_x1_m,
        predicted_x2_m_s=predicted_x2_m_s,
        predicted_fts_N=predicted_force_N,
        force_lookup_x1_m=force_x1_grid_m,
        force_lookup_fts_N=force_grid_N,
        initial_state=initial_state,
        raw_external_x2_t0_m_s=np.asarray(true_x2_m_s[0]),
        periodic_fit_x2_t0_m_s=np.asarray(fitted_x2_t0_m_s),
    )
    _plot(time_s, true_x1_m, predicted_x1_m, output_png)

    metadata = model.metadata()
    summary = {
        "definition": (
            "Fully autonomous rollout: external true x1(t0) is used directly, while "
            "x2(t0) is obtained by differentiating a fundamental-period fit to the "
            "complete external x1 raw trajectory. Every later KAN force evaluation "
            "uses predicted x1, and every later state is generated by the current "
            "AFM06a e0.0_real RHS."
        ),
        "archive_dir": str(ARCHIVE_DIR),
        "stage2_result": str(stage2_result),
        "stage1_dependency": str(stage1_result),
        "external_mat_source": str(mat_source),
        "sample_count": int(time_s.size),
        "time_range_us": [float(time_s[0] * 1.0e6), float(time_s[-1] * 1.0e6)],
        "dt_s": float(np.median(np.diff(time_s))),
        "initial_state_SI": {"x1_m": float(initial_state[0]), "x2_m_s": float(initial_state[1])},
        "initial_velocity_policy": "differentiate full-span fundamental-period fit of external x1 raw",
        "raw_external_x2_t0_m_s_not_used": float(true_x2_m_s[0]),
        "periodic_x1_fit": periodic_fit,
        "normalizer": normalizer,
        "force_output_policy": str(metadata["force_output_policy"]),
        "force_output_quantity": "Fts_N",
        "rhs_parameter_source": str(one_step.DATA_MANIFEST),
        "rhs_parameters_SI": parameters,
        "external_Favg_used": False,
        "integration": "fixed-step RK4 on the complete external 4 ns time grid",
        "metrics": {
            "x1_relative_rmse_pct": _relative_rmse_pct(predicted_x1_m, true_x1_m),
            "x1_error_nm": _stats((predicted_x1_m - true_x1_m) * 1.0e9),
            "x2_relative_rmse_pct": _relative_rmse_pct(predicted_x2_m_s, true_x2_m_s),
            "predicted_fts_nN": _stats(predicted_force_N * 1.0e9),
            "predicted_x1_nm": _stats(predicted_x1_m * 1.0e9),
        },
        "outputs": {"npz": str(output_npz), "png": str(output_png)},
    }
    _write_json(output_json, summary)
    print(f"Saved: {output_npz}")
    print(f"Saved: {output_json}")
    print(f"Saved: {output_png}")
    print(f"x1 rollout relative RMSE: {summary['metrics']['x1_relative_rmse_pct']:.9g}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
