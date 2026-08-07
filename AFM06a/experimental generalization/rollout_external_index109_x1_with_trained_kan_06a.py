"""Autonomous external-silicon x1 rollout using one archived AFM06a KAN.

This is a diagnostic rollout, not a training script. The current AFM06a force
head outputs physical ``Fts`` in N, so the RHS uses ``model([x1, 0]) / m`` as
the interaction acceleration term.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.stage1pluslight.checkpoint import load_checkpoint  # noqa: E402
from AFM06a.stage2light.config import default_config  # noqa: E402
from AFM06a.stage2light.data import load_stage1_endpoint  # noqa: E402
from AFM06a.stage2light.kan_backend import restore_exact_stage1_model  # noqa: E402


DEFAULT_ARCHIVE_DIR = (
    REPO_ROOT / "AFM06a" / "archive" / "st2l" / "rank1_20260803_142741"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "x1_real_super_real"
)
DEFAULT_LOCAL_MAT = (
    REPO_ROOT
    / "external data"
    / "Silicon exp data"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)
DEFAULT_RAW_MAT = (
    Path(r"E:\External data SI_SI hard sample raw")
    / "OneDrive_2_2026-7-26"
    / "dataAFM_Si"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)
DEFAULT_OUTPUT_STEM = "external_index109_raw_x1_trained_kan_x1_rollout_from_initial"

INDEX = 109
DT_S = 1.0 / 250.0e6
K_N_PER_M = 22.68
Q_FACTOR = 428.0
DRIVE_FREQUENCY_HZ = 164520.0
OMEGA_RAD_S = 2.0 * np.pi * DRIVE_FREQUENCY_HZ
M_KG = K_N_PER_M / (OMEGA_RAD_S**2)
C_N_S_PER_M = M_KG * OMEGA_RAD_S / Q_FACTOR
FD_AVG_N = 18.9e-9
PERIOD_S = 1.0 / DRIVE_FREQUENCY_HZ


def _required_dict(path: Path) -> dict[str, Any]:
    payload = load_checkpoint(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a dictionary checkpoint payload: {path}")
    return payload


def _resolve_archive_files(archive_dir: Path) -> tuple[Path, Path]:
    archive = archive_dir.expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"AFM06a archive is missing: {archive}")
    result_files = sorted((archive / "result").glob("afm_param_stage2light_06a_rank*.pkl"))
    stage1_files = sorted(
        (archive / "conditional dependency").glob("afm_param_stage1pluslight_06a.pkl")
    )
    if len(result_files) != 1:
        raise RuntimeError(
            f"Expected one stage2light result in {archive / 'result'}, found {len(result_files)}"
        )
    if len(stage1_files) != 1:
        raise RuntimeError(
            "Expected one st1pl dependency in "
            f"{archive / 'conditional dependency'}, found {len(stage1_files)}"
        )
    return result_files[0], stage1_files[0]


def _restore_stage2_model(archive_dir: Path):
    stage2_result, stage1_result = _resolve_archive_files(archive_dir)
    payload = _required_dict(stage2_result)
    if not bool(payload.get("complete")):
        raise RuntimeError(f"Stage2light result is not complete: {stage2_result}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Stage2light result has no model_state_dict: {stage2_result}")

    base = default_config(REPO_ROOT)
    saved = payload.get("config")
    updates: dict[str, Any] = {
        "input_rank": int(payload.get("input_rank", base.input_rank)),
        "device": "cpu",
        "stage1_result_path": stage1_result,
    }
    if isinstance(saved, dict):
        if saved.get("pykan_root") is not None:
            updates["pykan_root"] = Path(saved["pykan_root"])
        for key in (
            "gain_enabled",
            "gain_learnable",
            "soft_mask_enabled",
            "soft_mask_trainable",
        ):
            if saved.get(key) is not None:
                updates[key] = bool(saved[key])
    config = replace(base, **updates)
    endpoint = load_stage1_endpoint(config, enforce_current_contract=False)
    model = restore_exact_stage1_model(config, endpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, endpoint, payload, stage2_result, stage1_result


def _load_external_raw(mat_path: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    candidates = []
    if mat_path is not None:
        candidates.append(mat_path.expanduser())
    candidates.extend([DEFAULT_RAW_MAT, DEFAULT_LOCAL_MAT])
    errors: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            errors.append(f"{candidate} does not exist")
            continue
        try:
            data = sio.loadmat(
                str(candidate),
                variable_names=["displacement_fwd_sweep", "velocity_fwd_sweep"],
                squeeze_me=True,
                struct_as_record=False,
            )
            x1 = np.asarray(data["displacement_fwd_sweep"][INDEX], dtype=np.float64).ravel()
            x2 = np.asarray(data["velocity_fwd_sweep"][INDEX], dtype=np.float64).ravel()
            if x1.shape != x2.shape or x1.size < 2:
                raise ValueError(f"invalid x1/x2 shapes: {x1.shape}, {x2.shape}")
            if not (np.all(np.isfinite(x1)) and np.all(np.isfinite(x2))):
                raise ValueError("x1/x2 contain nonfinite values")
            time_s = np.arange(x1.size, dtype=np.float64) * DT_S
            return time_s, x1, x2, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("Could not load external raw x1/x2:\n" + "\n".join(errors))


@torch.no_grad()
def _fts_N(model: torch.nn.Module, x1: float) -> float:
    states = torch.zeros((1, 2), dtype=torch.float64, device="cpu")
    states[0, 0] = float(x1)
    return float(model(states).detach().cpu().reshape(-1)[0])


@torch.no_grad()
def _build_force_lookup(
    model: torch.nn.Module,
    x1_reference: np.ndarray,
    grid_size: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x1 = np.asarray(x1_reference, dtype=np.float64).reshape(-1)
    lo = float(np.min(x1))
    hi = float(np.max(x1))
    span = max(hi - lo, 1.0e-12)
    lo -= 0.25 * span
    hi += 0.25 * span
    grid = np.linspace(lo, hi, max(1024, int(grid_size)), dtype=np.float64)
    predictions: list[np.ndarray] = []
    zeros = np.zeros_like(grid)
    for start in range(0, grid.size, max(1, int(chunk_size))):
        stop = min(grid.size, start + max(1, int(chunk_size)))
        states = np.column_stack((grid[start:stop], zeros[start:stop]))
        tensor = torch.as_tensor(states, dtype=torch.float64, device="cpu")
        pred = model(tensor).detach().cpu().numpy().reshape(-1)
        predictions.append(np.asarray(pred, dtype=np.float64))
    return grid, np.concatenate(predictions)


def _lookup_fts_N(x1: float, x1_grid: np.ndarray, fts_grid_N: np.ndarray) -> float:
    return float(np.interp(float(x1), x1_grid, fts_grid_N))


@torch.no_grad()
def _rhs(
    x1_grid: np.ndarray,
    fts_grid_N: np.ndarray,
    t: float,
    state: np.ndarray,
) -> np.ndarray:
    x1 = float(state[0])
    x2 = float(state[1])
    fts_N = _lookup_fts_N(x1, x1_grid, fts_grid_N)
    x2dot = (
        (FD_AVG_N / M_KG) * np.sin(OMEGA_RAD_S * float(t))
        - (K_N_PER_M / M_KG) * x1
        - (C_N_S_PER_M / M_KG) * x2
        + fts_N / M_KG
    )
    return np.asarray([x2, x2dot], dtype=np.float64)


def _rk4_rollout(
    x1_grid: np.ndarray,
    fts_grid_N: np.ndarray,
    times: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = np.empty((2, times.size), dtype=np.float64)
    trajectory[:, 0] = np.asarray(initial_state, dtype=np.float64).reshape(2)
    for i in range(times.size - 1):
        t = float(times[i])
        h = float(times[i + 1] - times[i])
        y = trajectory[:, i]
        k1 = _rhs(x1_grid, fts_grid_N, t, y)
        k2 = _rhs(x1_grid, fts_grid_N, t + 0.5 * h, y + 0.5 * h * k1)
        k3 = _rhs(x1_grid, fts_grid_N, t + 0.5 * h, y + 0.5 * h * k2)
        k4 = _rhs(x1_grid, fts_grid_N, t + h, y + h * k3)
        trajectory[:, i + 1] = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(trajectory[:, i + 1])):
            raise RuntimeError(f"nonfinite rollout state at step {i + 1}, t={times[i + 1]:.9e}")
    fts_N = np.interp(trajectory[0], x1_grid, fts_grid_N)
    return trajectory, fts_N


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray, times: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(truth, dtype=float)
    t = np.asarray(times, dtype=float)
    numerator = float(np.sqrt(np.trapezoid(np.square(pred - ref), t)))
    denominator = max(float(np.sqrt(np.trapezoid(np.square(ref), t))), 1.0e-30)
    return 100.0 * numerator / denominator


def _plot_indices(indices: np.ndarray, max_points: int = 15000) -> np.ndarray:
    if indices.size <= max_points:
        return indices
    selected = np.linspace(0, indices.size - 1, max_points, dtype=int)
    return indices[np.unique(selected)]


def _windows(times: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    full = np.arange(times.size, dtype=int)
    centers = (
        max(0.5 * PERIOD_S, 0.10 * float(times[-1])),
        0.50 * float(times[-1]),
        min(float(times[-1]) - 0.5 * PERIOD_S, 0.90 * float(times[-1])),
    )
    out: list[tuple[str, np.ndarray]] = [("Full time-span", full)]
    for label, center in zip(("Early window", "Middle window", "Tail window"), centers, strict=True):
        idx = np.flatnonzero(
            (times >= max(float(times[0]), center - PERIOD_S))
            & (times <= min(float(times[-1]), center + PERIOD_S))
        )
        out.append((label, idx))
    return tuple(out)


def _plot(
    *,
    output_png: Path,
    times: np.ndarray,
    true_x1: np.ndarray,
    pred_x1: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 4.6), sharey=False)
    for axis, (title, idx) in zip(axes, _windows(times), strict=True):
        plot_idx = _plot_indices(idx)
        time_us = times[plot_idx] * 1.0e6
        axis.plot(
            time_us,
            true_x1[plot_idx] * 1.0e9,
            color="black",
            linewidth=1.05,
            label=r"external raw $x_1$",
            zorder=1,
        )
        axis.plot(
            time_us,
            pred_x1[plot_idx] * 1.0e9,
            color="crimson",
            linewidth=0.95,
            label=r"predicted rollout",
            zorder=2,
        )
        axis.set_title(title, fontsize=14)
        axis.set_xlabel(r"Time [$\mu$s]", fontsize=12)
        axis.set_ylabel(r"$x_1$ [nm]", fontsize=12)
        axis.grid(True, alpha=0.28, linewidth=0.8)
        axis.tick_params(axis="both", labelsize=11)
        axis.legend(loc="upper right", fontsize=10, framealpha=0.82)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--mat-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-step-stride", type=int, default=10)
    parser.add_argument("--force-lookup-grid-size", type=int, default=250000)
    parser.add_argument("--force-lookup-chunk-size", type=int, default=32768)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / f"{DEFAULT_OUTPUT_STEM}.npz"
    output_json = output_dir / f"{DEFAULT_OUTPUT_STEM}_summary.json"
    output_png = output_dir / f"{DEFAULT_OUTPUT_STEM}.png"

    torch.set_num_threads(1)
    model, endpoint, payload, stage2_result, stage1_result = _restore_stage2_model(args.archive_dir)
    time_raw, x1_raw, x2_raw, mat_source = _load_external_raw(args.mat_path)
    stride = max(1, int(args.time_step_stride))
    source_idx = np.arange(0, time_raw.size, stride, dtype=int)
    if source_idx[-1] != time_raw.size - 1:
        source_idx = np.concatenate((source_idx, np.asarray([time_raw.size - 1], dtype=int)))
    times = time_raw[source_idx]
    true_x1 = np.interp(times, time_raw, x1_raw)
    true_x2 = np.interp(times, time_raw, x2_raw)
    initial_state = np.asarray([true_x1[0], true_x2[0]], dtype=np.float64)

    force_x1_grid, force_fts_grid_N = _build_force_lookup(
        model,
        x1_raw,
        int(args.force_lookup_grid_size),
        int(args.force_lookup_chunk_size),
    )
    trajectory, pred_fts_N = _rk4_rollout(force_x1_grid, force_fts_grid_N, times, initial_state)
    if (
        float(np.min(trajectory[0])) < float(force_x1_grid[0])
        or float(np.max(trajectory[0])) > float(force_x1_grid[-1])
    ):
        combined = np.concatenate((x1_raw.reshape(-1), trajectory[0].reshape(-1)))
        force_x1_grid, force_fts_grid_N = _build_force_lookup(
            model,
            combined,
            int(args.force_lookup_grid_size),
            int(args.force_lookup_chunk_size),
        )
        trajectory, pred_fts_N = _rk4_rollout(
            force_x1_grid,
            force_fts_grid_N,
            times,
            initial_state,
        )
    pred_fts_eff_m_s2 = pred_fts_N / M_KG
    pred_x1 = trajectory[0]
    pred_x2 = trajectory[1]

    np.savez_compressed(
        output_npz,
        source_idx=source_idx,
        time_s=times,
        time_us=times * 1.0e6,
        external_x1_raw_m=true_x1,
        external_x2_raw_m_s=true_x2,
        pred_x1_m=pred_x1,
        pred_x2_m_s=pred_x2,
        pred_fts_N=pred_fts_N,
        pred_fts_eff_m_s2=pred_fts_eff_m_s2,
        force_lookup_x1_grid_m=force_x1_grid,
        force_lookup_fts_grid_N=force_fts_grid_N,
        initial_state_m_m_s=initial_state,
        time_step_stride=np.asarray(stride, dtype=np.int64),
        index=np.asarray(INDEX, dtype=np.int64),
        dt_raw_s=np.asarray(DT_S, dtype=np.float64),
        m_kg=np.asarray(M_KG, dtype=np.float64),
        c_N_s_per_m=np.asarray(C_N_S_PER_M, dtype=np.float64),
        k_N_per_m=np.asarray(K_N_PER_M, dtype=np.float64),
        q_factor=np.asarray(Q_FACTOR, dtype=np.float64),
        drive_frequency_Hz=np.asarray(DRIVE_FREQUENCY_HZ, dtype=np.float64),
        fd_avg_N=np.asarray(FD_AVG_N, dtype=np.float64),
    )
    _plot(output_png=output_png, times=times, true_x1=true_x1, pred_x1=pred_x1)

    digest = str(payload.get("model_state_dict_sha256", "")).strip()
    if not digest:
        digest = hashlib.sha256(stage2_result.read_bytes()).hexdigest()
    summary = {
        "definition": (
            "Autonomous x1 rollout from external index109 initial state. "
            "The archived KAN output is interpreted as physical Fts [N] "
            "and is inserted into the external cantilever RHS as Fts/m."
        ),
        "archive_dir": str(args.archive_dir.expanduser().resolve()),
        "stage2_result": str(stage2_result.resolve()),
        "stage1_dependency": str(stage1_result.resolve()),
        "external_mat_source": str(mat_source.resolve()),
        "index": INDEX,
        "sample_count_raw": int(time_raw.size),
        "sample_count_rollout": int(times.size),
        "time_step_stride": stride,
        "time_range_us": [float(times[0] * 1.0e6), float(times[-1] * 1.0e6)],
        "initial_state": {
            "x1_m": float(initial_state[0]),
            "x1_nm": float(initial_state[0] * 1.0e9),
            "x2_m_s": float(initial_state[1]),
        },
        "model_rank": int(payload.get("input_rank", endpoint.rank)),
        "model_state_dict_sha256": digest,
        "force_output_interpretation": "current_archive_model_output_is_Fts_N",
        "force_lookup_grid_size": int(force_x1_grid.size),
        "force_lookup_range_nm": [
            float(force_x1_grid[0] * 1.0e9),
            float(force_x1_grid[-1] * 1.0e9),
        ],
        "normalizer_application": "archived_trained_kan_training_mean_std",
        "external_rhs_parameters": {
            "m_kg": float(M_KG),
            "c_N_s_per_m": float(C_N_S_PER_M),
            "k_N_per_m": float(K_N_PER_M),
            "Q": float(Q_FACTOR),
            "drive_frequency_Hz": float(DRIVE_FREQUENCY_HZ),
            "omega_rad_s": float(OMEGA_RAD_S),
            "Fd_avg_N": float(FD_AVG_N),
            "Fd_avg_nN": float(FD_AVG_N * 1.0e9),
        },
        "x1_relative_rmse_pct": _relative_rmse_pct(pred_x1, true_x1, times),
        "external_x1_nm_range": [float(np.min(true_x1) * 1.0e9), float(np.max(true_x1) * 1.0e9)],
        "pred_x1_nm_range": [float(np.min(pred_x1) * 1.0e9), float(np.max(pred_x1) * 1.0e9)],
        "output_npz": str(output_npz),
        "output_png": str(output_png),
    }
    _write_json(output_json, summary)

    print(f"Saved: {output_npz}")
    print(f"Saved: {output_json}")
    print(f"Saved: {output_png}")
    print(f"x1 rollout relative RMSE: {summary['x1_relative_rmse_pct']:.6g}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
