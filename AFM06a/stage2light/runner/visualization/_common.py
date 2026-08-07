"""Shared AFM06a stage2light visualization data loading."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from AFM06a.stage1pluslight.checkpoint import write_checkpoint_atomic
from AFM06a.stage1pluslight.data import load_dataset, transition_resolved_source_indices
from AFM06a.stage1pluslight.kan_backend import state_dict_digest
from AFM06a.stage2light.config import Stage2LightConfig, default_config
from AFM06a.stage2light.data import Stage1Endpoint, load_stage1_endpoint
from AFM06a.stage2light.kan_backend import restore_exact_stage1_model
from AFM06a.stage2light.rollout import StateGuardTriggered, x2dot_rhs_torch
from AFM06a.stage2light.train import _forward_evaluation

def find_repo_root(start: str | Path) -> Path:
    here = Path(start).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM06a" / "stage2light").is_dir():
            return path
    raise RuntimeError(f"Could not locate HNODECB root from {start}")


REPO_ROOT = find_repo_root(__file__)
ROLLOUT_TARGET_POINTS_PER_PERIOD = 40
ROLLOUT_CONTACT_STRIDE = 1
ROLLOUT_VISUALIZATION_GUARD_MIN_MULTIPLIER = 1000.0
ROLLOUT_ODE_MAX_STEP_POLICY = "adaptive_solver_unconstrained_v1"
ROLLOUT_INTEGRATOR_POLICY = "fixed_step_rk4_on_rollout_output_grid_v1"
CURRENT_WINDOW_DIAGNOSTIC_END_TIME_S = 10.5e-3


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _full_rollout_cache_contract(
    context: VisualizationContext,
    model_hash: str,
    *,
    start_source_idx: int,
    initial_state_source: str,
    end_time_s: float | None,
) -> dict[str, Any]:
    saved_config = context.endpoint.payload["config"]
    warmstart = context.endpoint.warmstart
    return {
        "schema_version": 15,
        "rollout_grid_policy": "period_adaptive_with_dense_contact_v5",
        "rollout_integrator": ROLLOUT_INTEGRATOR_POLICY,
        "diagnostic_end_time_s": None if end_time_s is None else float(end_time_s),
        "start_source_idx": int(start_source_idx),
        "initial_state_source": str(initial_state_source),
        "force_input_source": "predicted_x1_from_autonomous_rollout",
        "regime_source": "predicted_x1_from_autonomous_rollout",
        "model_state_dict_sha256": model_hash,
        "window_source_indices_sha256": _array_sha256(context.endpoint.window.source_idx),
        "window_times_sha256": _array_sha256(context.endpoint.window.times),
        "train_idx_sha256": _array_sha256(context.endpoint.window.train_idx),
        "val_idx_sha256": _array_sha256(context.endpoint.window.val_idx),
        "sample_stride": int(saved_config["sample_stride"]),
        **_rollout_grid_metadata(context),
        "sample_count": (
            None
            if saved_config.get("sample_count") is None
            else int(saved_config["sample_count"])
        ),
        "transition_half_width_s": float(saved_config["transition_half_width_s"]),
        "ode_method": str(saved_config["ode_method"]),
        "ode_rtol": float(saved_config["ode_rtol"]),
        "ode_atol": float(saved_config["ode_atol"]),
        "ode_step_budget": int(saved_config["ode_step_budget"]),
        "state_guard_multiplier": float(saved_config["state_guard_multiplier"]),
        "visualization_state_guard_multiplier": max(
            float(saved_config["state_guard_multiplier"]),
            ROLLOUT_VISUALIZATION_GUARD_MIN_MULTIPLIER,
        ),
        "dataset_root": str(Path(saved_config["dataset_root"]).resolve()),
        "error_level": str(saved_config["error_level"]),
        "normalizer_policy": str(saved_config["normalizer_policy"]),
        "force_output_policy": str(warmstart["force_output_policy"]),
        "force_output_quantity": str(warmstart.get("force_output_quantity", "")),
        "force_mean": float(warmstart["force_mean"]),
        "force_scale": float(warmstart["force_scale"]),
        "physics_parameter_vector": [
            float(value) for value in context.endpoint.window.settings.parameter_vector
        ],
    }


@dataclass(frozen=True)
class VisualizationContext:
    config: Stage2LightConfig
    payload: dict[str, Any]
    endpoint: Stage1Endpoint
    model: torch.nn.Module
    history: list[dict[str, Any]]
    predicted_states: np.ndarray
    predicted_x2dot: np.ndarray
    predicted_fts: np.ndarray
    predicted_bar_fts: np.ndarray
    source_path: Path


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected stage2light payload type in {path}: {type(payload)!r}")
    return payload


def _candidate_payload_paths(config: Stage2LightConfig) -> list[Path]:
    paths = list(config.result_dir.glob("afm_param_stage2light_06a_rank*.pkl"))
    paths.extend(config.checkpoint_dir.glob("afm_param_stage2light_06a_rank*.checkpoint.pkl"))
    return sorted((path for path in paths if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)


def _config_for_payload(
    base: Stage2LightConfig,
    payload: dict[str, Any],
    *,
    preserve_io_paths: bool = False,
) -> Stage2LightConfig:
    saved = payload.get("config")
    rank = int(payload.get("input_rank", base.input_rank))
    updates: dict[str, Any] = {"input_rank": rank, "device": "cpu"}
    if isinstance(saved, dict):
        for key in ("epochs", "adam_epochs", "lbfgs_epochs"):
            if saved.get(key) is not None:
                updates[key] = int(saved[key])
        for key in (
            "gain_enabled",
            "gain_learnable",
            "soft_mask_enabled",
            "soft_mask_trainable",
        ):
            if saved.get(key) is not None:
                updates[key] = bool(saved[key])
        if not preserve_io_paths:
            for key in (
                "stage1_result_path",
                "result_dir",
                "checkpoint_dir",
                "log_dir",
                "visualization_dir",
                "archive_root",
                "pykan_root",
            ):
                if saved.get(key) is not None:
                    updates[key] = Path(saved[key])
    return replace(base, **updates)


def load_visualization_context(
    config: Stage2LightConfig | None = None,
    *,
    preserve_io_paths: bool = False,
    enforce_current_contract: bool = True,
) -> VisualizationContext:
    base = default_config(REPO_ROOT) if config is None else config
    paths = _candidate_payload_paths(base)
    if not paths:
        raise FileNotFoundError(
            f"No AFM06a stage2light result under {base.result_dir} or checkpoint under {base.checkpoint_dir}"
        )
    source_path = paths[0]
    payload = _load_pickle(source_path)
    cfg = _config_for_payload(base, payload, preserve_io_paths=preserve_io_paths)
    endpoint = load_stage1_endpoint(
        cfg,
        enforce_current_contract=enforce_current_contract,
    )
    model = restore_exact_stage1_model(cfg, endpoint)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Payload has no model_state_dict: {source_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    if (
        all(key in payload for key in ("predicted_states", "predicted_x2dot", "predicted_bar_fts"))
        and payload.get("predicted_states") is not None
        and payload.get("predicted_x2dot") is not None
    ):
        predicted_states = np.asarray(payload["predicted_states"], dtype=float)
        predicted_x2dot = np.asarray(payload["predicted_x2dot"], dtype=float)
        predicted_bar_fts = np.asarray(payload["predicted_bar_fts"], dtype=float)
        if payload.get("predicted_fts") is not None:
            predicted_fts = np.asarray(payload["predicted_fts"], dtype=float)
        else:
            predicted_fts = predicted_bar_fts * float(endpoint.window.settings.mass_kg)
    else:
        with torch.no_grad():
            evaluation = _forward_evaluation(model, endpoint, include_rollout=False)
        predicted_states = np.full_like(endpoint.window.states, np.nan, dtype=float)
        predicted_x2dot = np.full_like(endpoint.window.x2dot, np.nan, dtype=float)
        predicted_fts = evaluation.predicted_fts.detach().cpu().numpy()
        predicted_bar_fts = evaluation.predicted_bar_fts.detach().cpu().numpy()
    return VisualizationContext(
        config=cfg,
        payload=payload,
        endpoint=endpoint,
        model=model,
        history=list(payload.get("history", [])),
        predicted_states=predicted_states,
        predicted_x2dot=predicted_x2dot,
        predicted_fts=predicted_fts,
        predicted_bar_fts=predicted_bar_fts,
        source_path=source_path,
    )


def output_path(context: VisualizationContext, filename: str) -> Path:
    context.config.visualization_dir.mkdir(parents=True, exist_ok=True)
    return context.config.visualization_dir / filename


def finalize_and_save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {path}")
    return path


def finite_history_series(history: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    epochs: list[int] = []
    values: list[float] = []
    for row in history:
        try:
            epoch = int(row["epoch"])
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            epochs.append(epoch)
            values.append(value)
    return np.asarray(epochs, dtype=int), np.asarray(values, dtype=float)


def nested_history_series(
    history: list[dict[str, Any]],
    group: str,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    epochs: list[int] = []
    values: list[float] = []
    for row in history:
        nested = row.get(group)
        if not isinstance(nested, dict):
            continue
        try:
            epoch = int(row["epoch"])
            value = float(nested[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            epochs.append(epoch)
            values.append(value)
    return np.asarray(epochs, dtype=int), np.asarray(values, dtype=float)


def relative_rmse_pct(
    prediction: np.ndarray,
    truth: np.ndarray,
    times: np.ndarray | None = None,
) -> float:
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(truth, dtype=float)
    if times is not None:
        t = np.asarray(times, dtype=float)
        if t.size != pred.size or np.any(np.diff(t) <= 0.0):
            raise ValueError("times must match the signals and be strictly increasing")
        squared_error = float(np.trapezoid(np.square(pred - ref), t))
        squared_reference = float(np.trapezoid(np.square(ref), t))
        return 100.0 * float(np.sqrt(squared_error)) / max(
            float(np.sqrt(squared_reference)), 1.0e-30
        )
    return 100.0 * float(np.sqrt(np.mean(np.square(pred - ref)))) / max(
        float(np.sqrt(np.mean(np.square(ref)))), 1.0e-30
    )


def _contact_resolved_rollout_source_indices(
    times: np.ndarray,
    contact: np.ndarray,
    *,
    start: int,
    stride: int,
    contact_stride: int,
    transition_half_width_s: float,
) -> np.ndarray:
    """Use a plotting/output grid dense enough to preserve narrow force peaks."""

    base = transition_resolved_source_indices(
        times,
        contact,
        start=start,
        stride=stride,
        transition_half_width_s=transition_half_width_s,
    )
    indices = np.arange(start, times.size, dtype=int)
    contact_stride = max(1, int(contact_stride))
    contact_band = indices[np.asarray(contact[start:], dtype=bool)][::contact_stride]
    return np.unique(
        np.concatenate(
            (
                base,
                contact_band,
                np.asarray([times.size - 1], dtype=int),
            )
        )
    )


def _rollout_grid_metadata(context: VisualizationContext) -> dict[str, float | int]:
    """Derive the rollout visualization grid from the current dataset frequency."""

    saved_config = context.endpoint.payload["config"]
    loaded = load_dataset(
        Path(saved_config["dataset_root"]),
        str(saved_config["error_level"]),
    )
    times = np.asarray(loaded.table["t"], dtype=float)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise RuntimeError("AFM06a rollout visualization requires increasing dataset times")
    raw_dt = float(np.median(np.diff(times)))
    omega_d = float(context.endpoint.window.settings.omega0)
    if omega_d <= 0.0 or not np.isfinite(omega_d):
        raise RuntimeError(f"invalid AFM06a driving angular frequency: {omega_d}")
    period = float(2.0 * np.pi / omega_d)
    base_stride = max(1, int(saved_config["sample_stride"]))
    period_stride = max(
        1,
        int(np.floor(period / (ROLLOUT_TARGET_POINTS_PER_PERIOD * raw_dt))),
    )
    plot_stride = max(1, min(base_stride, period_stride))
    return {
        "rollout_target_points_per_period": int(ROLLOUT_TARGET_POINTS_PER_PERIOD),
        "rollout_base_sample_stride": int(base_stride),
        "rollout_period_adaptive_stride": int(period_stride),
        "rollout_plot_stride": int(plot_stride),
        "rollout_contact_stride": int(ROLLOUT_CONTACT_STRIDE),
        "rollout_raw_dt_s": float(raw_dt),
        "rollout_drive_period_s": float(period),
        "ode_max_step_policy": ROLLOUT_ODE_MAX_STEP_POLICY,
        "ode_output_grid_step_s": float(plot_stride * raw_dt),
        "ode_max_step_s": None,
    }


def _rollout_max_step_s(context: VisualizationContext) -> None:
    """Return the optional adaptive-solver step cap for rollout diagnostics."""

    _ = context
    return None


def _fixed_step_rk4_rollout_torch(
    model: torch.nn.Module,
    settings,
    initial_state: torch.Tensor,
    times: torch.Tensor,
    *,
    x1_abs_guard: float | None,
    x2_abs_guard: float | None,
    max_rhs_evaluations: int,
) -> tuple[torch.Tensor, int]:
    """Roll out on the requested output grid with classical RK4.

    This mirrors the AFM06a data-generation integrator style while replacing
    the analytical interaction force by the trained KAN force module.
    """

    if times.ndim != 1 or times.numel() < 2 or bool(torch.any(times[1:] <= times[:-1])):
        raise ValueError("RK4 rollout times must be strictly increasing")
    state = torch.as_tensor(
        initial_state,
        dtype=times.dtype,
        device=times.device,
    ).reshape(-1)
    if state.numel() != 2:
        raise ValueError("AFM06a RK4 rollout initial state must contain x1 and x2")

    k_n_m, omega0, mass_kg, c_n_s_m, fd_n, *_ = settings.parameter_vector
    k_over_m = float(k_n_m) / float(mass_kg)
    c_over_m = float(c_n_s_m) / float(mass_kg)
    fd_over_m = float(fd_n) / float(mass_kg)
    inv_mass = 1.0 / float(mass_kg)
    omega = float(omega0)

    trajectory = torch.empty((2, times.numel()), dtype=times.dtype, device=times.device)
    force_state = torch.zeros((1, 2), dtype=times.dtype, device=times.device)
    rhs_evaluations = 0

    def rhs(t_value: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        nonlocal rhs_evaluations
        rhs_evaluations += 1
        if rhs_evaluations > int(max_rhs_evaluations):
            raise StateGuardTriggered(
                "rk4_rhs_evaluation_budget_exceeded "
                f"calls={rhs_evaluations} budget={int(max_rhs_evaluations)} "
                f"t={float(t_value.detach()):.9e}"
            )
        if not bool(torch.all(torch.isfinite(current_state))):
            raise StateGuardTriggered(
                f"rk4_nonfinite_state t={float(t_value.detach()):.9e}"
            )
        abs_x1 = float(torch.abs(current_state[0]).detach())
        abs_x2 = float(torch.abs(current_state[1]).detach())
        if x1_abs_guard is not None and abs_x1 > float(x1_abs_guard):
            raise StateGuardTriggered(
                f"state_guard_x1 t={float(t_value.detach()):.9e} "
                f"abs={abs_x1:.9e} guard={float(x1_abs_guard):.9e}"
            )
        if x2_abs_guard is not None and abs_x2 > float(x2_abs_guard):
            raise StateGuardTriggered(
                f"state_guard_x2 t={float(t_value.detach()):.9e} "
                f"abs={abs_x2:.9e} guard={float(x2_abs_guard):.9e}"
            )

        force_state[0, 0] = current_state[0]
        force_state[0, 1] = 0.0
        fts = model(force_state)[0]
        x1 = current_state[0]
        x2 = current_state[1]
        return torch.stack(
            (
                x2,
                fd_over_m * torch.sin(omega * t_value)
                - k_over_m * x1
                - c_over_m * x2
                + inv_mass * fts,
            )
        )

    trajectory[:, 0] = state
    for index in range(times.numel() - 1):
        t0 = times[index]
        dt = times[index + 1] - t0
        half_dt = 0.5 * dt
        k1 = rhs(t0, state)
        k2 = rhs(t0 + half_dt, state + half_dt * k1)
        k3 = rhs(t0 + half_dt, state + half_dt * k2)
        k4 = rhs(t0 + dt, state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[:, index + 1] = state

    if trajectory.shape != (2, times.numel()) or not bool(torch.all(torch.isfinite(trajectory))):
        raise RuntimeError(f"AFM06a RK4 rollout returned invalid trajectory: {tuple(trajectory.shape)}")
    return trajectory, rhs_evaluations


def _build_full_rollout_payload(
    context: VisualizationContext,
    *,
    start_source_idx: int,
    cache_filename: str,
    initial_state_source: str,
    end_time_s: float | None,
) -> dict[str, Any]:
    cache_path = output_path(
        context,
        cache_filename,
    )
    model_hash = state_dict_digest(context.model.frozen_state_dict())
    cache_contract = _full_rollout_cache_contract(
        context,
        model_hash,
        start_source_idx=start_source_idx,
        initial_state_source=initial_state_source,
        end_time_s=end_time_s,
    )
    if cache_path.is_file():
        cached = _load_pickle(cache_path)
        if cached.get("cache_contract") == cache_contract:
            return cached

    saved_config = context.endpoint.payload["config"]
    loaded = load_dataset(Path(saved_config["dataset_root"]), str(saved_config["error_level"]))
    settings = loaded.settings
    all_times = np.asarray(loaded.table["t"], dtype=float)
    start = int(start_source_idx)
    if start < 0 or start >= all_times.size - 1:
        raise ValueError(f"invalid full-rollout start source index: {start}")
    if end_time_s is None:
        stop = all_times.size
    else:
        end_time = float(end_time_s)
        stop = int(np.searchsorted(all_times, end_time, side="right"))
        if stop <= start + 1:
            raise ValueError(
                "AFM06a current-window rollout diagnostic has no samples between "
                f"source index {start} and {end_time:.6g} s"
            )
    grid_metadata = _rollout_grid_metadata(context)
    stride = int(grid_metadata["rollout_plot_stride"])
    source_idx = _contact_resolved_rollout_source_indices(
        all_times,
        np.asarray(loaded.table["contact"], dtype=bool),
        start=start,
        stride=stride,
        contact_stride=int(grid_metadata["rollout_contact_stride"]),
        transition_half_width_s=float(saved_config["transition_half_width_s"]),
    )
    if stop < all_times.size:
        source_idx = source_idx[source_idx < stop]
        end_source_idx = stop - 1
        if source_idx.size == 0 or int(source_idx[-1]) != end_source_idx:
            source_idx = np.unique(
                np.concatenate((source_idx, np.asarray([end_source_idx], dtype=int)))
            )
    times = all_times[source_idx]
    true_states = np.asarray(loaded.states[:, source_idx], dtype=float)
    model_dtype = context.model.state_mean.dtype
    model_device = context.model.state_mean.device
    times_t = torch.as_tensor(times, dtype=model_dtype, device=model_device)
    initial_state = torch.as_tensor(true_states[:, 0], dtype=model_dtype, device=model_device)
    guard_multiplier = max(
        float(saved_config["state_guard_multiplier"]),
        ROLLOUT_VISUALIZATION_GUARD_MIN_MULTIPLIER,
    )
    x1_abs_guard = guard_multiplier * max(float(np.max(np.abs(true_states[0]))), 1.0e-30)
    x2_abs_guard = guard_multiplier * max(float(np.max(np.abs(true_states[1]))), 1.0e-30)
    max_rhs_evaluations = max(200_000, int(source_idx.size) * 200)
    max_step_s = None
    with torch.no_grad():
        trajectory, rk4_rhs_evaluations = _fixed_step_rk4_rollout_torch(
            context.model,
            settings,
            initial_state,
            times_t,
            x1_abs_guard=x1_abs_guard,
            x2_abs_guard=x2_abs_guard,
            max_rhs_evaluations=max_rhs_evaluations,
        )
        predicted_x2dot = x2dot_rhs_torch(
            trajectory,
            times_t,
            context.model,
            settings,
        )
        force_states = torch.stack(
            (
                trajectory[0],
                torch.zeros(times_t.numel(), dtype=torch.float64),
            ),
            dim=1,
        )
        predicted_fts = context.model(force_states)
        predicted_bar_fts = predicted_fts / float(settings.mass_kg)
    true_bar_fts = np.asarray(loaded.table["bar_fts"], dtype=float)[source_idx]
    true_fts = np.asarray(loaded.table["Fts"], dtype=float)[source_idx]
    payload = {
        "cache_contract": cache_contract,
        "model_state_dict_sha256": model_hash,
        "source_idx": source_idx,
        "times": times,
        "true_states": true_states,
        "true_x2dot": np.asarray(loaded.table["x2dot"], dtype=float)[source_idx],
        "true_bar_fts": true_bar_fts,
        "true_fts": true_fts,
        "true_contact": np.asarray(loaded.table["contact"], dtype=bool)[source_idx],
        "predicted_states": trajectory.detach().cpu().numpy(),
        "predicted_x2dot": predicted_x2dot.detach().cpu().numpy(),
        "predicted_fts": predicted_fts.detach().cpu().numpy(),
        "predicted_bar_fts": predicted_bar_fts.detach().cpu().numpy(),
        "initial_state_source": str(initial_state_source),
        "force_input_source": "predicted_x1_from_autonomous_rollout",
        "regime_source": "predicted_x1_from_autonomous_rollout",
        "x1_abs_guard": x1_abs_guard,
        "x2_abs_guard": x2_abs_guard,
        "max_rhs_evaluations": max_rhs_evaluations,
        "rk4_rhs_evaluations": int(rk4_rhs_evaluations),
        "rollout_integrator": ROLLOUT_INTEGRATOR_POLICY,
        "ode_max_step_s": max_step_s,
        "ode_max_step_policy": ROLLOUT_ODE_MAX_STEP_POLICY,
        "rollout_grid_metadata": dict(grid_metadata),
    }
    write_checkpoint_atomic(cache_path, payload)
    return payload


def full_rollout_payload(context: VisualizationContext) -> dict[str, Any]:
    return _build_full_rollout_payload(
        context,
        start_source_idx=int(context.endpoint.window.source_idx[0]),
        cache_filename=(
            f"afm06a_stage2light_full_rollout_cache_rank{context.endpoint.rank}.pkl"
        ),
        initial_state_source="numerical_true_data_at_W1_t0",
        end_time_s=CURRENT_WINDOW_DIAGNOSTIC_END_TIME_S,
    )


def full_rollout_payload_from_w0(context: VisualizationContext) -> dict[str, Any]:
    return _build_full_rollout_payload(
        context,
        start_source_idx=0,
        cache_filename=(
            f"afm06a_stage2light_full_rollout_from_W0_cache_rank"
            f"{context.endpoint.rank}.pkl"
        ),
        initial_state_source="numerical_true_data_at_global_t0",
        end_time_s=None,
    )


def _pointwise_full_resolution_payload(
    context: VisualizationContext,
    *,
    start_source_idx: int,
    force_input_source: str,
    end_time_s: float | None,
) -> dict[str, Any]:
    """Evaluate the trained force map at every true x1 point from a source index onward."""

    saved_config = context.endpoint.payload["config"]
    loaded = load_dataset(
        Path(saved_config["dataset_root"]),
        str(saved_config["error_level"]),
    )

    start = int(start_source_idx)
    all_times = np.asarray(loaded.table["t"], dtype=float)
    if start < 0 or start >= all_times.size:
        raise ValueError(f"invalid pointwise start source index: {start}")
    if end_time_s is None:
        stop = all_times.size
    else:
        stop = int(np.searchsorted(all_times, float(end_time_s), side="right"))
        if stop <= start:
            raise ValueError(
                "AFM06a current-window pointwise diagnostic has no samples between "
                f"source index {start} and {float(end_time_s):.6g} s"
            )
    times = all_times[start:stop]
    true_x1 = np.asarray(loaded.states[0], dtype=float)[start:stop]
    true_bar_fts = np.asarray(loaded.table["bar_fts"], dtype=float)[start:stop]
    true_fts = np.asarray(loaded.table["Fts"], dtype=float)[start:stop]
    predicted_fts_chunks: list[np.ndarray] = []
    chunk_size = 131_072
    model_dtype = context.model.state_mean.dtype
    model_device = context.model.state_mean.device
    with torch.no_grad():
        for chunk_start in range(0, true_x1.size, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, true_x1.size)
            force_states = torch.zeros(
                (chunk_stop - chunk_start, 2),
                dtype=model_dtype,
                device=model_device,
            )
            force_states[:, 0] = torch.as_tensor(
                true_x1[chunk_start:chunk_stop],
                dtype=model_dtype,
                device=model_device,
            )
            predicted_fts_chunks.append(
                context.model(force_states).detach().cpu().numpy()
            )
    predicted_fts = np.concatenate(predicted_fts_chunks)
    predicted_bar_fts = predicted_fts / float(loaded.settings.mass_kg)

    return {
        "source_idx": np.arange(start, start + times.size, dtype=int),
        "times": times,
        "true_bar_fts": true_bar_fts,
        "true_fts": true_fts,
        "predicted_fts": predicted_fts,
        "predicted_bar_fts": predicted_bar_fts,
        "force_input_source": str(force_input_source),
        "normalizer_source": "trained_kan_inherited_normalizer",
        "evaluation_mode": "pointwise_no_rollout",
    }


def pointwise_full_resolution_payload(
    context: VisualizationContext,
) -> dict[str, Any]:
    """Evaluate the trained force map from the current training-window start onward."""

    return _pointwise_full_resolution_payload(
        context,
        start_source_idx=int(context.endpoint.window.source_idx[0]),
        force_input_source=(
            "original_full_resolution_true_x1_pointwise_from_current_window"
        ),
        end_time_s=CURRENT_WINDOW_DIAGNOSTIC_END_TIME_S,
    )


def pointwise_full_resolution_payload_from_w0(
    context: VisualizationContext,
) -> dict[str, Any]:
    """Evaluate the trained force map at every true x1 point from global t=0 onward."""

    return _pointwise_full_resolution_payload(
        context,
        start_source_idx=0,
        force_input_source="original_full_resolution_true_x1_pointwise_from_global_t0",
        end_time_s=None,
    )


def full_rollout_panels(
    context: VisualizationContext,
    payload: dict[str, Any],
    *,
    initial_window_label: str | None = None,
) -> list[tuple[str, np.ndarray]]:
    if initial_window_label is None:
        initial_window_label = (
            "Training window"
            if context.endpoint.window.val_idx.size == 0
            else "Training & validation window"
        )
    times = np.asarray(payload["times"], dtype=float)
    full = np.arange(times.size, dtype=int)
    window_duration = float(context.endpoint.window.times[-1] - context.endpoint.window.times[0])
    training = np.flatnonzero(times <= float(times[0]) + window_duration)
    middle_time = 0.5 * (float(times[0]) + float(times[-1]))
    observation = np.flatnonzero(np.abs(times - middle_time) <= 0.5 * window_duration)
    tail = np.flatnonzero(times >= float(times[-1]) - window_duration)
    return [
        ("Full time-span", full),
        (initial_window_label, training),
        ("Observation window, middle", observation),
        ("Observation window, tail", tail),
    ]


def full_rollout_panels_from_w0(
    context: VisualizationContext,
    payload: dict[str, Any],
) -> list[tuple[str, np.ndarray]]:
    times = np.asarray(payload["times"], dtype=float)
    full = np.arange(times.size, dtype=int)
    window_duration = float(context.endpoint.window.times[-1] - context.endpoint.window.times[0])
    contact = np.asarray(payload.get("true_contact", []), dtype=bool)
    if contact.shape == times.shape and np.any(contact):
        first_contact_time = float(times[np.flatnonzero(contact)[0]])
    else:
        saved_config = context.endpoint.payload["config"]
        loaded = load_dataset(
            Path(saved_config["dataset_root"]),
            str(saved_config["error_level"]),
        )
        contact_all = np.asarray(loaded.table["contact"], dtype=bool)
        contact_indices = np.flatnonzero(contact_all)
        if contact_indices.size == 0:
            raise RuntimeError("AFM06a dataset contains no contact point for W0 panel")
        first_contact_time = float(np.asarray(loaded.table["t"], dtype=float)[contact_indices[0]])
    initial = np.flatnonzero(
        (times >= first_contact_time)
        & (times <= first_contact_time + window_duration)
    )
    middle_time = 0.5 * (float(times[0]) + float(times[-1]))
    observation = np.flatnonzero(np.abs(times - middle_time) <= 0.5 * window_duration)
    tail = np.flatnonzero(times >= float(times[-1]) - window_duration)
    return [
        ("Full time-span", full),
        ("Initial window, W0", initial),
        ("Observation window, middle", observation),
        ("Observation window, tail", tail),
    ]


__all__ = [
    "REPO_ROOT",
    "VisualizationContext",
    "finalize_and_save",
    "finite_history_series",
    "full_rollout_panels",
    "full_rollout_panels_from_w0",
    "full_rollout_payload",
    "full_rollout_payload_from_w0",
    "pointwise_full_resolution_payload",
    "pointwise_full_resolution_payload_from_w0",
    "load_visualization_context",
    "nested_history_series",
    "output_path",
    "relative_rmse_pct",
]
