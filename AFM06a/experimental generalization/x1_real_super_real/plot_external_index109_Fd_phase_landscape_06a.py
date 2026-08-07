"""Generate an 81 x 81 Fd-phase rollout-loss landscape with delta_c fixed to zero."""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

full_rollout = importlib.import_module(
    "run_external_index109_full_rollout_06a_framework"
)
fit_module = importlib.import_module(
    "optimize_external_index109_Fd_phase_contact_damping_06a"
)

OUTPUT_STEM = "external_index109_Fd_phase_loss_landscape_delta_c_zero_06a"
GRID_SIZE = 81
FIT_STRIDE = 40
FD_BOUNDS_NN = fit_module.FD_BOUNDS_NN
PHASE_BOUNDS_RAD = fit_module.PHASE_BOUNDS_RAD

_WORKER_DATA: tuple[np.ndarray, ...] | None = None


def _init_worker(
    time_s: np.ndarray,
    truth_x1_m: np.ndarray,
    initial_state: np.ndarray,
    force_x1_grid_m: np.ndarray,
    force_grid_N: np.ndarray,
    parameter_values: tuple[float, float, float, float],
    phase_values_rad: np.ndarray,
    truth_energy: float,
) -> None:
    global _WORKER_DATA
    _WORKER_DATA = (
        time_s,
        truth_x1_m,
        initial_state,
        force_x1_grid_m,
        force_grid_N,
        np.asarray(parameter_values, dtype=np.float64),
        phase_values_rad,
        np.asarray([truth_energy], dtype=np.float64),
    )


def _evaluate_fd_row(task: tuple[int, float]) -> tuple[int, np.ndarray]:
    if _WORKER_DATA is None:
        raise RuntimeError("landscape worker was not initialized")
    (
        time_s,
        truth_x1_m,
        initial_state,
        force_x1_grid_m,
        force_grid_N,
        parameter_values,
        phase_values_rad,
        truth_energy_array,
    ) = _WORKER_DATA
    parameters = {
        "m": float(parameter_values[0]),
        "c": float(parameter_values[1]),
        "k": float(parameter_values[2]),
        "omega0": float(parameter_values[3]),
    }
    truth_energy = float(truth_energy_array[0])
    row_index, fd_nN = task
    phase_count = phase_values_rad.size
    state = np.repeat(initial_state[:, None], phase_count, axis=1)
    squared_error = np.square(state[0] - truth_x1_m[0])
    fd_N = float(fd_nN) * 1.0e-9

    def rhs(t: float, current: np.ndarray) -> np.ndarray:
        x1 = current[0]
        x2 = current[1]
        force_N = np.interp(x1, force_x1_grid_m, force_grid_N)
        x2dot = (
            fd_N * np.sin(parameters["omega0"] * t + phase_values_rad)
            - parameters["c"] * x2
            - parameters["k"] * x1
            + force_N
        ) / parameters["m"]
        return np.vstack((x2, x2dot))

    for time_index in range(time_s.size - 1):
        t = float(time_s[time_index])
        h = float(time_s[time_index + 1] - time_s[time_index])
        k1 = rhs(t, state)
        k2 = rhs(t + 0.5 * h, state + 0.5 * h * k1)
        k3 = rhs(t + 0.5 * h, state + 0.5 * h * k2)
        k4 = rhs(t + h, state + h * k3)
        state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        squared_error += np.square(state[0] - truth_x1_m[time_index + 1])

    row = squared_error / (float(time_s.size) * truth_energy)
    return row_index, row


def _plot(
    fd_values_nN: np.ndarray,
    phase_values_rad: np.ndarray,
    loss: np.ndarray,
    output_path: Path,
) -> None:
    fd_grid, phase_grid = np.meshgrid(fd_values_nN, phase_values_rad, indexing="ij")
    log_loss = np.log10(np.maximum(loss, np.finfo(np.float64).tiny))
    minimum_index = np.unravel_index(int(np.argmin(loss)), loss.shape)

    fig = plt.figure(figsize=(10.5, 7.5))
    axis = fig.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        fd_grid,
        phase_grid,
        log_loss,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        rcount=GRID_SIZE,
        ccount=GRID_SIZE,
    )
    axis.scatter(
        [fd_values_nN[minimum_index[0]]],
        [phase_values_rad[minimum_index[1]]],
        [log_loss[minimum_index]],
        color="#d62728",
        s=48,
        depthshade=False,
        label="grid minimum",
    )
    axis.set_xlabel(r"$F_d$ [nN]", labelpad=10)
    axis.set_ylabel(r"$\phi_d$ [rad]", labelpad=10)
    axis.set_zlabel(r"$\log_{10}$ normalized $x_1$ MSE", labelpad=10)
    axis.view_init(elev=29.0, azim=-133.0)
    axis.legend(loc="upper right")
    colorbar = fig.colorbar(surface, ax=axis, shrink=0.68, pad=0.10)
    colorbar.set_label(r"$\log_{10}$ normalized $x_1$ MSE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    started = time.perf_counter()
    model, _, _, _, _ = full_rollout.pointwise._restore_stage2_model(
        full_rollout.ARCHIVE_DIR
    )
    time_full, true_x1_full, mat_source = (
        full_rollout.pointwise._load_external_x1_raw(None)
    )
    parameters = dict(full_rollout.one_step._load_rhs_parameters())
    fitted_x2_t0, initial_fit = full_rollout._fit_initial_velocity_from_x1(
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
    truth_energy = max(float(np.mean(np.square(true_x1_fit))), 1.0e-30)

    fd_values_nN = np.linspace(*FD_BOUNDS_NN, GRID_SIZE, dtype=np.float64)
    phase_values_rad = np.linspace(
        *PHASE_BOUNDS_RAD, GRID_SIZE, dtype=np.float64
    )
    loss = np.empty((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    parameter_values = (
        float(parameters["m"]),
        float(parameters["c"]),
        float(parameters["k"]),
        float(parameters["omega0"]),
    )
    worker_count = max(1, min(8, (os.cpu_count() or 2) - 1, GRID_SIZE))
    context = mp.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(
            time_fit,
            true_x1_fit,
            initial_state,
            force_x1_grid_m,
            force_grid_N,
            parameter_values,
            phase_values_rad,
            truth_energy,
        ),
    ) as pool:
        for completed, (row_index, row) in enumerate(
            pool.imap_unordered(
                _evaluate_fd_row,
                enumerate(fd_values_nN),
                chunksize=1,
            ),
            start=1,
        ):
            loss[row_index] = row
            if completed == 1 or completed % 10 == 0 or completed == GRID_SIZE:
                print(f"[landscape] completed {completed}/{GRID_SIZE} Fd rows", flush=True)

    minimum_index = np.unravel_index(int(np.argmin(loss)), loss.shape)
    elapsed_s = time.perf_counter() - started
    output_npz = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}.json"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
    np.savez_compressed(
        output_npz,
        fd_values_nN=fd_values_nN,
        phase_values_rad=phase_values_rad,
        normalized_x1_mse=loss,
        fit_indices=fit_indices,
        time_fit_s=time_fit,
        true_x1_fit_m=true_x1_fit,
        initial_state=initial_state,
    )
    summary = {
        "definition": "81x81 Fd-phase autonomous-rollout landscape with delta_c fixed to zero",
        "trained_kan_archive": str(full_rollout.ARCHIVE_DIR),
        "external_mat_source": str(mat_source),
        "grid_size": GRID_SIZE,
        "fit_stride": FIT_STRIDE,
        "rollout_sample_count": int(time_fit.size),
        "worker_count": worker_count,
        "delta_c_contact_N_s_m": 0.0,
        "Fd_bounds_nN": list(FD_BOUNDS_NN),
        "phase_bounds_rad": list(PHASE_BOUNDS_RAD),
        "grid_minimum": {
            "Fd_nN": float(fd_values_nN[minimum_index[0]]),
            "phase_rad": float(phase_values_rad[minimum_index[1]]),
            "normalized_x1_mse": float(loss[minimum_index]),
            "relative_x1_rmse_pct": float(100.0 * np.sqrt(loss[minimum_index])),
        },
        "initial_state_SI": {
            "x1_m": float(initial_state[0]),
            "x2_m_s": float(initial_state[1]),
        },
        "initial_velocity_fit": initial_fit,
        "elapsed_seconds": float(elapsed_s),
        "outputs": {"npz": str(output_npz), "png": str(output_png)},
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(fd_values_nN, phase_values_rad, loss, output_png)
    print(json.dumps(summary["grid_minimum"], indent=2), flush=True)
    print(f"[done] elapsed={elapsed_s:.1f} s", flush=True)
    print(f"[done] {output_png}", flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
