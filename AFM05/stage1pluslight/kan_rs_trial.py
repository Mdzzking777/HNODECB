"""KAN random-search trial backend for AFM05 stage1pluslight.

This is intentionally AFM05-specific:
- no true ks/cs, x3(t), x3dot(t), delta_dot(t), or Fts(t) diagnostics;
- train ranking is based on AFM05 train loss;
- KAN gain initialization keeps the AFM04 hook, using the AFM05 reconstructed
  pixel-tagged gain-force-reference training-window trajectory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch

from AFM05.stage2light.kan_backend import (
    KANForceModule,
    initial_grid_support_from_raw_inputs,
    initial_grid_support_to_meta,
)
from AFM05.stage1pluslight.losses import (
    FTS_RANGE_EPS,
    FTS_RANGE_WEIGHT,
    SCALE_EPS,
    X3_RANGE_EPS,
    X3_RANGE_WEIGHT,
    relative_rmse_pct,
    softplus,
)
from AFM05.stage1pluslight.rollout import actuation_values_at, fts_from_x2dot_signal, rollout_single_shooting, x2dot_rhs


X3_INITIAL_GRID_SUPPORT_OFFSET_LO = -2.0
X3_INITIAL_GRID_SUPPORT_OFFSET_HI = 1.0


def _safe_rms_np(values: np.ndarray, *, floor: float = SCALE_EPS) -> float | np.ndarray:
    arr = np.asarray(values, dtype=float)
    rms = np.sqrt(np.mean(np.square(arr), axis=-1))
    return np.maximum(rms, float(floor))


def _mean_abs_amp_np(values: np.ndarray, *, floor: float) -> float:
    amp = float(np.mean(np.abs(np.asarray(values, dtype=float).reshape(-1))))
    return float(max(amp, float(floor)))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    return raw not in ("0", "false", "False", "no", "NO")


def _env_pct_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    out: list[float] = []
    for chunk in raw.replace(";", ",").split(","):
        item = chunk.strip()
        if item == "":
            continue
        val = float(item)
        if 0.0 < val < 100.0:
            out.append(val)
    return tuple(out) if out else default


def _torch_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name!r}")


@dataclass(frozen=True)
class KANStage1Config:
    pykan_root: Path
    device: str
    dtype_name: str
    seed: int
    width: tuple[int, int, int]
    grid: int
    spline_k: int
    base_fun: str
    symbolic_enabled: bool
    auto_save: bool
    noise_scale: float
    affine_trainable: bool
    grid_eps: float
    grid_range_lo: float
    grid_range_hi: float
    adaptive_grid_enabled: bool
    gnn_learnable: bool
    random_search_use_val: bool
    random_search_fail_fast_enabled: bool
    random_search_x1_guard_mult: float
    x1_online_error_guard_enabled: bool
    x1_online_error_guard_pct: float
    x1_online_error_guard_task_pcts: tuple[float, ...]
    window_mode: str
    pixel_tag: str
    arch_window_us: float
    window_sample_stride: int
    ode_solver: str
    ode_fallback_solver: str
    ode_rtol: float
    ode_atol: float
    ode_max_step: float

    @property
    def dtype(self) -> torch.dtype:
        return _torch_dtype(self.dtype_name)


def default_kan_stage1_config(repo_root: str | Path, stage_cfg: Any) -> KANStage1Config:
    repo_root = Path(repo_root)
    pykan_root = Path(os.environ.get("HNODECB_PYKAN_ROOT", str(repo_root.parent / "pykan")))
    hidden = max(1, _env_int("HNODECB_AFM05_KAN_TEST_WIDTH_HIDDEN", getattr(stage_cfg, "fixed_num_hidden_nodes", 7)))
    online_task_pcts = _env_pct_tuple(
        "HNODECB_AFM05_KAN_TEST_X1_ONLINE_ERROR_GUARD_TASK_PCTS",
        tuple(float(v) for v in range(5, 100, 5)),
    )
    return KANStage1Config(
        pykan_root=pykan_root,
        device=os.environ.get("HNODECB_AFM05_KAN_TEST_DEVICE", "cpu").strip() or "cpu",
        dtype_name=os.environ.get("HNODECB_AFM05_KAN_TEST_DTYPE", "float64").strip() or "float64",
        seed=max(1, _env_int("HNODECB_AFM05_STAGE1_RUN_SEED", getattr(stage_cfg, "run_seed", 20260406))),
        width=(3, hidden, 1),
        grid=max(1, _env_int("HNODECB_AFM05_KAN_TEST_GRID", 11)),
        spline_k=max(1, _env_int("HNODECB_AFM05_KAN_TEST_SPLINE_K", 3)),
        base_fun=os.environ.get("HNODECB_AFM05_KAN_TEST_BASE_FUN", "silu").strip() or "silu",
        symbolic_enabled=_env_bool("HNODECB_AFM05_KAN_TEST_SYMBOLIC", False),
        auto_save=_env_bool("HNODECB_AFM05_KAN_TEST_AUTO_SAVE", False),
        noise_scale=_env_float("HNODECB_AFM05_KAN_TEST_NOISE_SCALE", 0.1),
        affine_trainable=_env_bool("HNODECB_AFM05_KAN_TEST_AFFINE_TRAINABLE", False),
        grid_eps=_env_float("HNODECB_AFM05_KAN_TEST_GRID_EPS", 0.02),
        grid_range_lo=_env_float("HNODECB_AFM05_KAN_TEST_GRID_RANGE_LO", -1.0),
        grid_range_hi=_env_float("HNODECB_AFM05_KAN_TEST_GRID_RANGE_HI", 1.0),
        adaptive_grid_enabled=_env_bool("HNODECB_AFM05_KAN_TEST_ADAPTIVE_GRID", True),
        gnn_learnable=_env_bool("HNODECB_AFM05_KAN_TEST_GNN_LEARNABLE", False),
        random_search_use_val=_env_bool("HNODECB_AFM05_KAN_TEST_RANDOM_SEARCH_USE_VAL", False),
        random_search_fail_fast_enabled=_env_bool("HNODECB_AFM05_KAN_TEST_RANDOM_SEARCH_FAIL_FAST", True),
        random_search_x1_guard_mult=_env_float("HNODECB_AFM05_KAN_TEST_RANDOM_SEARCH_X1_GUARD_MULT", 1.0),
        x1_online_error_guard_enabled=_env_bool("HNODECB_AFM05_KAN_TEST_X1_ONLINE_ERROR_GUARD", True),
        x1_online_error_guard_pct=_env_float("HNODECB_AFM05_KAN_TEST_X1_ONLINE_ERROR_GUARD_PCT", 0.05),
        x1_online_error_guard_task_pcts=online_task_pcts,
        window_mode=str(getattr(stage_cfg, "window_mode", "window_05_initial")),
        pixel_tag=str(getattr(stage_cfg, "pixel_tag", "184_152")),
        arch_window_us=float(getattr(stage_cfg, "arch_window_us", 6.288e-6)),
        window_sample_stride=max(1, int(getattr(stage_cfg, "window_sample_stride", 4))),
        ode_solver=str(getattr(stage_cfg, "ode_solver", "Radau")),
        ode_fallback_solver=str(getattr(stage_cfg, "ode_fallback_solver", "BDF")),
        ode_rtol=float(getattr(stage_cfg, "ode_rtol", 1.0e-8)),
        ode_atol=float(getattr(stage_cfg, "ode_atol", 1.0e-8)),
        ode_max_step=float(getattr(stage_cfg, "ode_max_step", 0.0)),
    )


def _trial_seed(base_seed: int, seed_bank_index: int) -> int:
    return int(base_seed + seed_bank_index)


def _safe_scale(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    scale = float(np.std(arr))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = float(np.max(np.abs(arr)))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = 1.0
    return scale


def _formal_state_normalizer(train_states_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(train_states_all, dtype=float)
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError(f"train_states_all must have shape (n, >=3), got {states.shape}")
    x1_mean = float(np.mean(states[:, 0]))
    x2_mean = float(np.mean(states[:, 1]))
    x1_scale = _safe_scale(states[:, 0])
    x2_scale = _safe_scale(states[:, 1])
    # x3 is measured from the undeformed sample surface; no trajectory mean is observed.
    mean = np.asarray([x1_mean, x2_mean, 0.0], dtype=float)
    scale = np.asarray([x1_scale, x2_scale, max(0.1 * x1_scale, 1.0e-30)], dtype=float)
    return mean, scale


def _observable_grid_inputs_from_ode(*, ode_train: torch.Tensor, state_mean: np.ndarray, state_scale: np.ndarray) -> torch.Tensor:
    n = int(ode_train.shape[1])
    inputs = torch.empty((n, 3), dtype=ode_train.dtype, device=ode_train.device)
    inputs[:, 0:2] = ode_train[0:2, :].transpose(0, 1)
    if n <= 1:
        inputs[:, 2] = ode_train[2, 0]
    else:
        mean = np.asarray(state_mean, dtype=float).reshape(-1)
        scale = np.asarray(state_scale, dtype=float).reshape(-1)
        center = float(mean[2])
        amp = max(float(abs(scale[2])), 1.0e-30)
        x3_init = float(ode_train[2, 0].detach().cpu())
        x3_init_norm = (x3_init - center) / amp
        inputs[:, 2] = torch.linspace(
            center + (x3_init_norm + X3_INITIAL_GRID_SUPPORT_OFFSET_LO) * amp,
            center + (x3_init_norm + X3_INITIAL_GRID_SUPPORT_OFFSET_HI) * amp,
            n,
            dtype=ode_train.dtype,
            device=ode_train.device,
        )
    return inputs


def _x3_pred_normalizer_meta(traj: np.ndarray) -> dict[str, float | str | bool]:
    values = np.asarray(traj[2, :], dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return {
            "rs_x3_normalizer_valid": False,
            "rs_x3_normalizer_source": "AFM05_stage1pluslight_rs_rollout_x3_pred",
            "rs_x3_pred_mean": float("nan"),
            "rs_x3_pred_scale": float("nan"),
            "rs_x3_pred_min": float("nan"),
            "rs_x3_pred_max": float("nan"),
            "rs_x3_norm_support_min": float("nan"),
            "rs_x3_norm_support_max": float("nan"),
        }
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        norm_min = norm_max = float("nan")
        valid = False
    else:
        norm = (values - mean) / scale
        norm_min = float(np.min(norm))
        norm_max = float(np.max(norm))
        valid = True
    return {
        "rs_x3_normalizer_valid": bool(valid),
        "rs_x3_normalizer_source": "AFM05_stage1pluslight_rs_rollout_x3_pred",
        "rs_x3_pred_mean": mean,
        "rs_x3_pred_scale": float(scale),
        "rs_x3_pred_min": float(np.min(values)),
        "rs_x3_pred_max": float(np.max(values)),
        "rs_x3_norm_support_min": norm_min,
        "rs_x3_norm_support_max": norm_max,
    }


def _x3_norm_kan_meta(
    traj: np.ndarray | None,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
) -> dict[str, float]:
    """Summarize rollout x3 in the KAN input coordinate used by AGU/grid support."""

    out = {
        "x3_norm_KAN_min": float("nan"),
        "x3_norm_KAN_max": float("nan"),
        "x3_norm_KAN_outside_1_frac": float("nan"),
        "x3_norm_KAN_outside_2_frac": float("nan"),
    }
    if traj is None:
        return out
    values = np.asarray(traj, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] == 0:
        return out
    mean = np.asarray(state_mean, dtype=float).reshape(-1)
    scale = np.asarray(state_scale, dtype=float).reshape(-1)
    if mean.size < 3 or scale.size < 3 or not np.isfinite(mean[2]) or not np.isfinite(scale[2]) or abs(scale[2]) <= 1.0e-30:
        return out
    norm = (values[2, :] - float(mean[2])) / float(scale[2])
    finite = norm[np.isfinite(norm)]
    if finite.size == 0:
        return out
    abs_norm = np.abs(finite)
    out["x3_norm_KAN_min"] = float(np.min(finite))
    out["x3_norm_KAN_max"] = float(np.max(finite))
    out["x3_norm_KAN_outside_1_frac"] = float(np.mean(abs_norm > 1.0))
    out["x3_norm_KAN_outside_2_frac"] = float(np.mean(abs_norm > 2.0))
    return out


class _KANContactModel:
    def __init__(self, force_module: KANForceModule, *, dtype: torch.dtype, device: str) -> None:
        self.force_module = force_module
        self.dtype = dtype
        self.device = device

    def __call__(self, state: np.ndarray, _params: Any = None) -> float:
        with torch.no_grad():
            states = torch.as_tensor(np.asarray(state, dtype=float).reshape(1, 3), dtype=self.dtype, device=self.device)
            out = self.force_module(states)
        return float(out.reshape(-1)[0].detach().cpu())


def _make_failure_record(
    *,
    global_trial_id: int,
    ks_fixed: float,
    cs_fixed: float,
    ks_node_idx: int,
    cs_node_idx: int,
    nn_seed_bank_idx: int,
    seed: int,
    cfg: KANStage1Config,
    failure_reason: str,
    dt_total: float,
    init_gnn: float,
    init_gain_status: str,
    initial_grid_fields: dict[str, Any],
    x3_norm_fields: dict[str, float] | None = None,
) -> dict[str, Any]:
    x3_norm_fields = x3_norm_fields or _x3_norm_kan_meta(None, np.zeros(3), np.ones(3))
    metric_parts = {
        "state": float("nan"),
        "x1_state": float("nan"),
        "x2_state": float("nan"),
        "x2dot": float("nan"),
        "x3_range": float("nan"),
        "fts_range": float("nan"),
        "cont": float("nan"),
        "x1_rec": float("nan"),
        "x2_rec": float("nan"),
        "x2dot_rec": float("nan"),
    }
    return {
        "loss": float("inf"),
        "train_loss": float("inf"),
        "val_loss": float("nan"),
        "val_loss_start": float("inf"),
        "params": {
            "trial_id": int(global_trial_id),
            "ks0": float(ks_fixed),
            "cs0": float(cs_fixed),
            "ks_node_idx": int(ks_node_idx),
            "cs_node_idx": int(cs_node_idx),
            "node_label": f"node_{ks_node_idx}*{cs_node_idx}",
            "nn_seed_bank_idx": int(nn_seed_bank_idx),
            "nn_init_seed": int(seed),
            "nn_force_mode": "kan_force_model",
            "nn_backend": "pykan",
            "stage1plus_standalone": True,
            "stage1plus_search_mode": f"kscs_grid_kan_seedbank_{cfg.window_mode}",
            "stage1plus_window_mode": str(cfg.window_mode),
            "stage1plus_window_pixel_tag": str(cfg.pixel_tag),
            "stage1plus_arch_window_us": float(cfg.arch_window_us),
            "stage1plus_window_sample_stride": int(cfg.window_sample_stride),
            "window_role": "window_05_initial",
            "window_pixel_tag": str(cfg.pixel_tag),
            "ranking_metric": "train_loss",
            "kan_grid": int(cfg.grid),
            "kan_spline_k": int(cfg.spline_k),
            "kan_base_fun": str(cfg.base_fun),
            "kan_gain_init_status": str(init_gain_status),
            "kan_gain_init_force_reference_q95_abs": float("nan"),
            "x1_amp_guard": (
                float(cfg.random_search_x1_guard_mult) if bool(cfg.random_search_fail_fast_enabled) else float("nan")
            ),
            "x1_online_error_guard_enabled": bool(
                cfg.random_search_fail_fast_enabled and cfg.x1_online_error_guard_enabled
            ),
            "x1_online_error_guard_pct": float(cfg.x1_online_error_guard_pct),
            "x1_online_error_guard_task_pcts": [float(v) for v in cfg.x1_online_error_guard_task_pcts],
            "AFM05_true_side_available": False,
            **initial_grid_fields,
            **x3_norm_fields,
        },
        "train_parts": metric_parts,
        "val_parts": metric_parts,
        "ks_hat": float(ks_fixed),
        "cs_hat": float(cs_fixed),
        "is_viable": False,
        "train_reason": str(failure_reason),
        "val_reason": str(failure_reason),
        "p_net_vec": np.zeros(0, dtype=float),
        "nn_gain": float(init_gnn),
        "nn_force_mode": "kan_force_model",
        "completed_epochs": 0,
        "retry_count_total": 0,
        "early_stopped": False,
        "early_stop_reason": "",
        "trial_failed": True,
        "failure_reason": str(failure_reason),
        "failure_epoch": 0,
        "lr_last": float("nan"),
        "time_per_epoch": float(dt_total),
    }


def _eval_loss_from_traj(
    *,
    traj: np.ndarray,
    ode_data: np.ndarray,
    x2dot_data: np.ndarray,
    contact_mask: np.ndarray | None,
    times: np.ndarray,
    state12_scale: np.ndarray,
    x2dot_scale: float,
    x3_scale: float,
    mech: np.ndarray,
    model: _KANContactModel,
    known_pars: Mapping[str, Any],
    zero_contact_override: bool,
    loss_indices: np.ndarray,
) -> tuple[float, dict[str, float]]:
    idx = np.asarray(loss_indices, dtype=int)
    ode_full = np.asarray(ode_data, dtype=float)
    times_full = np.asarray(times, dtype=float)
    x3_range_amp = _mean_abs_amp_np(ode_full[0, :], floor=X3_RANGE_EPS)
    f_act_full = actuation_values_at(known_pars, times_full)
    fts_range_amp = _mean_abs_amp_np(f_act_full, floor=FTS_RANGE_EPS)
    pred = np.asarray(traj[:, idx], dtype=float)
    obs = np.asarray(ode_full[:, idx], dtype=float)
    x2dot_obs = np.asarray(x2dot_data[idx], dtype=float)
    t = np.asarray(times_full[idx], dtype=float)
    weights = np.ones(idx.size, dtype=float)
    if contact_mask is not None:
        contact = np.asarray(contact_mask, dtype=bool)[idx]
        weights = np.where(contact, 1.0, 1.0).astype(float)
    weights_sum = float(max(np.sum(weights), 1.0e-30))

    _ = state12_scale
    state_scale = np.maximum(
        np.sqrt(np.mean(np.square(obs[0:2, :]), axis=1)),
        SCALE_EPS,
    )
    x1_err = np.square((obs[0, :] - pred[0, :]) / state_scale[0])
    x2_err = np.square((obs[1, :] - pred[1, :]) / state_scale[1])
    x1_state = float(np.sum(weights * x1_err) / weights_sum)
    x2_state = float(np.sum(weights * x2_err) / weights_sum)
    state = float(x1_state + x2_state)

    x2dot_pred = np.array(
        [
            x2dot_rhs(pred[:, j], mech, model, None, known_pars, float(t[j]), zero_contact_override=zero_contact_override)
            for j in range(t.size)
        ],
        dtype=float,
    )
    _ = x2dot_scale
    x2dot_scale_eval = float(max(float(np.sqrt(np.mean(np.square(x2dot_obs)))), SCALE_EPS))
    x2dot_loss = float(np.sum(weights * np.square((x2dot_obs - x2dot_pred) / x2dot_scale_eval)) / weights_sum)

    fts_pred = fts_from_x2dot_signal(pred, x2dot_pred, t, known_pars)
    fts_pen = np.square(
        np.asarray(softplus(np.abs(fts_pred) - fts_range_amp, FTS_RANGE_EPS), dtype=float) / fts_range_amp
    )
    fts_range = float(FTS_RANGE_WEIGHT * np.sum(weights * fts_pen) / weights_sum)

    _ = x3_scale
    x3_pen = np.square(np.asarray(softplus(np.abs(pred[2, :]) - x3_range_amp, X3_RANGE_EPS), dtype=float) / x3_range_amp)
    x3_range = float(X3_RANGE_WEIGHT * np.sum(weights * x3_pen) / weights_sum)

    x1_rec = relative_rmse_pct(
        float(np.sum(np.square(obs[0, :] - pred[0, :]))),
        float(np.sum(np.square(obs[0, :]))),
        max(1, t.size),
        SCALE_EPS,
    )
    x2_rec = relative_rmse_pct(
        float(np.sum(np.square(obs[1, :] - pred[1, :]))),
        float(np.sum(np.square(obs[1, :]))),
        max(1, t.size),
        SCALE_EPS,
    )
    x2dot_rec = relative_rmse_pct(
        float(np.sum(np.square(x2dot_obs - x2dot_pred))),
        float(np.sum(np.square(x2dot_obs))),
        max(1, t.size),
        SCALE_EPS,
    )
    parts = {
        "state": state,
        "x1_state": x1_state,
        "x2_state": x2_state,
        "x2dot": x2dot_loss,
        "x3_range": x3_range,
        "fts_range": fts_range,
        "x1_rms_scale": float(state_scale[0]),
        "x2_rms_scale": float(state_scale[1]),
        "x2dot_rms_scale": float(x2dot_scale_eval),
        "x3_range_amp": float(x3_range_amp),
        "fts_range_amp": float(fts_range_amp),
        "cont": 0.0,
        "x1_rec": float(x1_rec),
        "x2_rec": float(x2_rec),
        "x2dot_rec": float(x2dot_rec),
    }
    total = float(state + x2dot_loss + x3_range + fts_range)
    return total, parts


def _guard_trajectory(traj: np.ndarray, *, x1_abs_guard: float | None) -> str:
    if not np.all(np.isfinite(traj)):
        return "nonfinite_rollout"
    if x1_abs_guard is not None and float(np.max(np.abs(traj[0, :]))) > float(x1_abs_guard):
        return f"x1_guard_triggered:{float(np.max(np.abs(traj[0, :]))):.6e}>{float(x1_abs_guard):.6e}"
    return ""


def _x1_rec_prefix_pct(x1_pred: np.ndarray, x1_obs: np.ndarray) -> float:
    pred = np.asarray(x1_pred, dtype=float).reshape(-1)
    obs = np.asarray(x1_obs, dtype=float).reshape(-1)
    if pred.size != obs.size:
        raise ValueError(f"x1 prefix length mismatch: pred={pred.size} obs={obs.size}")
    if pred.size == 0:
        return float("inf")
    return float(
        relative_rmse_pct(
            float(np.sum(np.square(obs - pred))),
            float(np.sum(np.square(obs))),
            int(pred.size),
            SCALE_EPS,
        )
    )


def _online_guard_checkpoint_indices(n: int, task_pcts: tuple[float, ...]) -> tuple[tuple[int, float], ...]:
    if n < 2:
        return tuple()
    checkpoints: dict[int, float] = {}
    last = n - 1
    for pct in task_pcts:
        pct_f = float(pct)
        if not np.isfinite(pct_f) or pct_f <= 0.0 or pct_f >= 100.0:
            continue
        idx = int(np.ceil((pct_f / 100.0) * float(last)))
        idx = max(1, min(last, idx))
        checkpoints.setdefault(idx, pct_f)
    return tuple(sorted(checkpoints.items(), key=lambda item: item[0]))


def _rollout_with_x1_online_error_guard(
    *,
    mech: np.ndarray,
    model: _KANContactModel,
    known_pars: Mapping[str, Any],
    u0: np.ndarray,
    times_full: np.ndarray,
    x1_obs_full: np.ndarray,
    x1_abs_guard: float | None,
    cfg: KANStage1Config,
) -> np.ndarray:
    times = np.asarray(times_full, dtype=float).reshape(-1)
    x1_obs = np.asarray(x1_obs_full, dtype=float).reshape(-1)
    if times.size != x1_obs.size:
        raise ValueError(f"online guard time/x1 length mismatch: {times.size} vs {x1_obs.size}")
    checkpoints = _online_guard_checkpoint_indices(times.size, cfg.x1_online_error_guard_task_pcts)
    guard_enabled = (
        bool(cfg.random_search_fail_fast_enabled)
        and bool(cfg.x1_online_error_guard_enabled)
        and np.isfinite(float(cfg.x1_online_error_guard_pct))
        and float(cfg.x1_online_error_guard_pct) > 0.0
        and len(checkpoints) > 0
    )
    if not guard_enabled:
        return rollout_single_shooting(
            mech,
            model,
            None,
            known_pars,
            u0,
            times,
            zero_contact_override=False,
            ode_solver=cfg.ode_solver,
            ode_fallback_solver=cfg.ode_fallback_solver,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            ode_max_step=cfg.ode_max_step,
            x1_abs_guard=x1_abs_guard,
        )

    checkpoint_map = dict(checkpoints)
    stop_indices = sorted(set([idx for idx, _pct in checkpoints] + [times.size - 1]))
    out = np.empty((3, times.size), dtype=float)
    current_u = np.asarray(u0, dtype=float).reshape(3).copy()
    out[:, 0] = current_u
    start = 0
    threshold = float(cfg.x1_online_error_guard_pct)
    for stop in stop_indices:
        if stop <= start:
            continue
        segment_times = times[start : stop + 1]
        segment = rollout_single_shooting(
            mech,
            model,
            None,
            known_pars,
            current_u,
            segment_times,
            zero_contact_override=False,
            ode_solver=cfg.ode_solver,
            ode_fallback_solver=cfg.ode_fallback_solver,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            ode_max_step=cfg.ode_max_step,
            x1_abs_guard=x1_abs_guard,
        )
        out[:, start : stop + 1] = segment
        current_u = np.asarray(segment[:, -1], dtype=float)
        start = stop
        if stop in checkpoint_map:
            rec = _x1_rec_prefix_pct(out[0, : stop + 1], x1_obs[: stop + 1])
            if not np.isfinite(rec) or rec > threshold:
                task_pct = float(checkpoint_map[stop])
                raise RuntimeError(
                    "x1_online_error_guard_triggered:"
                    f"task_pct={task_pct:.1f},idx={stop},"
                    f"x1_rec_prefix={rec:.8f}%>{threshold:.8f}%"
                )
    return out


def prepare_kan_stage1_runtime(
    repo_root: str | Path,
    *,
    stage_config: Any,
    window_bundle: dict[str, Any],
    known_pars: Mapping[str, Any],
    state12_scale: np.ndarray,
    x2dot_scale: float,
    x3_scale: float,
) -> dict[str, Any]:
    cfg = default_kan_stage1_config(repo_root, stage_config)
    dtype = cfg.dtype
    device = cfg.device
    ode_train = np.asarray(window_bundle["ode_train"], dtype=float)
    state_mean, state_scale = _formal_state_normalizer(ode_train.T)

    train_max_abs_x1 = float(np.max(np.abs(np.asarray(window_bundle["ode_full"], dtype=float)[0, :])))
    x1_abs_guard = (
        float(cfg.random_search_x1_guard_mult) * train_max_abs_x1
        if cfg.random_search_fail_fast_enabled and np.isfinite(train_max_abs_x1) and train_max_abs_x1 > 0.0
        else None
    )

    train_gain_force_reference = np.asarray(
        window_bundle.get("train_gain_force_reference", np.asarray([], dtype=float)),
        dtype=float,
    ).reshape(-1)
    if train_gain_force_reference.size != int(ode_train.shape[1]) or not np.all(np.isfinite(train_gain_force_reference)):
        raise ValueError(
            "AFM05 gain initialization requires finite train_gain_force_reference with the same length as ode_train"
        )
    train_gain_force_reference_source = f"F_ts_{cfg.pixel_tag}_{cfg.window_mode}_train"
    train_gain_force_reference_q95_abs = float(np.quantile(np.abs(train_gain_force_reference), 0.95))

    return {
        "cfg": cfg,
        "dtype": dtype,
        "device": device,
        "window_bundle": window_bundle,
        "known_pars": dict(known_pars),
        "state_mean": state_mean,
        "state_scale": state_scale,
        "state12_scale": np.asarray(state12_scale, dtype=float),
        "x2dot_scale": float(x2dot_scale),
        "x3_scale": float(x3_scale),
        "x1_abs_guard": x1_abs_guard,
        "x1_online_error_guard_enabled": bool(
            cfg.random_search_fail_fast_enabled and cfg.x1_online_error_guard_enabled
        ),
        "x1_online_error_guard_pct": float(cfg.x1_online_error_guard_pct),
        "x1_online_error_guard_task_pcts": [float(v) for v in cfg.x1_online_error_guard_task_pcts],
        "train_gain_force_reference": train_gain_force_reference,
        "train_gain_force_reference_source": train_gain_force_reference_source,
        "train_gain_force_reference_q95_abs": train_gain_force_reference_q95_abs,
    }


def stage1pluslight_kan_random_trial(
    global_trial_id: int,
    ks_fixed: float,
    cs_fixed: float,
    ks_node_idx: int,
    cs_node_idx: int,
    nn_seed_bank_idx: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    cfg: KANStage1Config = runtime["cfg"]
    dtype: torch.dtype = runtime["dtype"]
    device: str = runtime["device"]
    split: dict[str, Any] = runtime["window_bundle"]
    known_pars: dict[str, Any] = runtime["known_pars"]

    seed = _trial_seed(int(cfg.seed), int(nn_seed_bank_idx))
    mech = np.asarray([float(ks_fixed), float(cs_fixed)], dtype=float)
    t0 = perf_counter()

    ode_train_t = torch.as_tensor(np.asarray(split["ode_train"], dtype=float), dtype=dtype, device=device)
    grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=ode_train_t,
        state_mean=runtime["state_mean"],
        state_scale=runtime["state_scale"],
    )
    initial_grid_support = initial_grid_support_from_raw_inputs(grid_inputs, runtime["state_mean"], runtime["state_scale"])
    initial_grid_support_meta = initial_grid_support_to_meta(
        initial_grid_support,
        source="AFM05_stage1pluslight_rs_observed_x1x2_dynamic_x3_init",
    )
    initial_grid_fields = {
        "initial_grid_support_source": str(initial_grid_support_meta["source"]),
        "initial_grid_support": initial_grid_support_meta["support"],
        "initial_grid_support_x1_min": float(initial_grid_support_meta["x1_min"]),
        "initial_grid_support_x1_max": float(initial_grid_support_meta["x1_max"]),
        "initial_grid_support_x2_min": float(initial_grid_support_meta["x2_min"]),
        "initial_grid_support_x2_max": float(initial_grid_support_meta["x2_max"]),
        "initial_grid_support_x3_min": float(initial_grid_support_meta["x3_min"]),
        "initial_grid_support_x3_max": float(initial_grid_support_meta["x3_max"]),
    }

    init_gnn = 1.0
    init_gain_status = "not_initialized"
    try:
        model = KANForceModule(
            pykan_root=cfg.pykan_root,
            state_mean=runtime["state_mean"],
            state_scale=runtime["state_scale"],
            seed=seed,
            width=cfg.width,
            grid=cfg.grid,
            spline_k=cfg.spline_k,
            base_fun=cfg.base_fun,
            symbolic_enabled=cfg.symbolic_enabled,
            auto_save=cfg.auto_save,
            noise_scale=cfg.noise_scale,
            affine_trainable=cfg.affine_trainable,
            grid_eps=cfg.grid_eps,
            grid_range=(cfg.grid_range_lo, cfg.grid_range_hi),
            initial_grid_support=initial_grid_support,
            dist=float(known_pars.get("Z", known_pars.get("dist", np.nan))),
            a0=float(known_pars.get("a0", np.nan)),
            gnn_learnable=cfg.gnn_learnable,
            soft_mask_enabled=True,
            soft_mask_trainable=False,
            soft_mask_s0_a0=20.0,
            soft_mask_s0_min_a0=1.0,
            soft_mask_s0_max_a0=100.0,
            soft_mask_alpha_a0=0.25,
            soft_mask_alpha_min_a0=0.02,
            soft_mask_alpha_max_a0=5.0,
            device=device,
            dtype=dtype,
        ).to(device)
        with torch.no_grad():
            if cfg.adaptive_grid_enabled:
                model.update_grid_from_normalized_inputs(grid_inputs)
            gain_ref = runtime.get("train_gain_force_reference")
            if gain_ref is not None:
                gain_ref_t = torch.as_tensor(gain_ref, dtype=dtype, device=device)
                if torch.isfinite(gain_ref_t).all():
                    train_states = torch.as_tensor(np.asarray(split["ode_train"], dtype=float).T, dtype=dtype, device=device)
                    init_gnn = float(model.initialize_gain_from_truth(train_states, gain_ref_t))
                    init_gain_status = str(runtime.get("train_gain_force_reference_source", "gain_from_force_reference"))
        np_model = _KANContactModel(model, dtype=dtype, device=device)

        ode_full = np.asarray(split["ode_full"], dtype=float)
        times_full = np.asarray(split["times_full"], dtype=float)
        u0 = np.asarray([ode_full[0, 0], ode_full[1, 0], float(split["x3_t0_val"])], dtype=float)
        traj = _rollout_with_x1_online_error_guard(
            mech=mech,
            model=np_model,
            known_pars=known_pars,
            u0=u0,
            times_full=times_full,
            x1_obs_full=ode_full[0, :],
            x1_abs_guard=runtime["x1_abs_guard"],
            cfg=cfg,
        )
        x3_norm_fields = _x3_norm_kan_meta(traj, runtime["state_mean"], runtime["state_scale"])
        guard_reason = _guard_trajectory(
            traj,
            x1_abs_guard=runtime["x1_abs_guard"],
        )
        if guard_reason:
            dt_total = float(perf_counter() - t0)
            return _make_failure_record(
                global_trial_id=global_trial_id,
                ks_fixed=ks_fixed,
                cs_fixed=cs_fixed,
                ks_node_idx=ks_node_idx,
                cs_node_idx=cs_node_idx,
                nn_seed_bank_idx=nn_seed_bank_idx,
                seed=seed,
                cfg=cfg,
                failure_reason=guard_reason,
                dt_total=dt_total,
                init_gnn=init_gnn,
                init_gain_status=init_gain_status,
                initial_grid_fields=initial_grid_fields,
                x3_norm_fields=x3_norm_fields,
            )

        train_loss, train_parts = _eval_loss_from_traj(
            traj=traj,
            ode_data=ode_full,
            x2dot_data=np.asarray(split["x2dot_full"], dtype=float),
            contact_mask=np.asarray(split["contact_full"], dtype=bool) if "contact_full" in split else None,
            times=times_full,
            state12_scale=runtime["state12_scale"],
            x2dot_scale=runtime["x2dot_scale"],
            x3_scale=runtime["x3_scale"],
            mech=mech,
            model=np_model,
            known_pars=known_pars,
            zero_contact_override=False,
            loss_indices=np.asarray(split["train_idx"], dtype=int),
        )
        val_loss = float("nan")
        val_parts = train_parts
        if cfg.random_search_use_val and np.asarray(split["val_idx"], dtype=int).size > 0:
            val_loss, val_parts = _eval_loss_from_traj(
                traj=traj,
                ode_data=ode_full,
                x2dot_data=np.asarray(split["x2dot_full"], dtype=float),
                contact_mask=np.asarray(split["contact_full"], dtype=bool) if "contact_full" in split else None,
                times=times_full,
                state12_scale=runtime["state12_scale"],
                x2dot_scale=runtime["x2dot_scale"],
                x3_scale=runtime["x3_scale"],
                mech=mech,
                model=np_model,
                known_pars=known_pars,
                zero_contact_override=False,
                loss_indices=np.asarray(split["val_idx"], dtype=int),
            )
        ranking_loss = float(train_loss)
    except Exception as exc:
        dt_total = float(perf_counter() - t0)
        return _make_failure_record(
            global_trial_id=global_trial_id,
            ks_fixed=ks_fixed,
            cs_fixed=cs_fixed,
            ks_node_idx=ks_node_idx,
            cs_node_idx=cs_node_idx,
            nn_seed_bank_idx=nn_seed_bank_idx,
            seed=seed,
            cfg=cfg,
            failure_reason=f"{type(exc).__name__}: {exc}",
            dt_total=dt_total,
            init_gnn=init_gnn,
            init_gain_status=init_gain_status,
            initial_grid_fields=initial_grid_fields,
        )

    dt_total = float(perf_counter() - t0)
    x3_pred_fields = _x3_pred_normalizer_meta(traj)
    x3_norm_fields = _x3_norm_kan_meta(traj, runtime["state_mean"], runtime["state_scale"])
    metric_parts = dict(val_parts)

    return {
        "loss": float(ranking_loss),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_loss_start": float(ranking_loss),
        "params": {
            "trial_id": int(global_trial_id),
            "ks0": float(ks_fixed),
            "cs0": float(cs_fixed),
            "ks_node_idx": int(ks_node_idx),
            "cs_node_idx": int(cs_node_idx),
            "node_label": f"node_{ks_node_idx}*{cs_node_idx}",
            "nn_seed_bank_idx": int(nn_seed_bank_idx),
            "nn_init_seed": int(seed),
            "nn_force_mode": "kan_force_model",
            "nn_backend": "pykan",
            "stage1plus_standalone": True,
            "stage1plus_search_mode": f"kscs_grid_kan_seedbank_{cfg.window_mode}",
            "stage1plus_window_mode": str(cfg.window_mode),
            "stage1plus_window_pixel_tag": str(cfg.pixel_tag),
            "stage1plus_arch_window_us": float(cfg.arch_window_us),
            "stage1plus_window_sample_stride": int(cfg.window_sample_stride),
            "window_role": str(split.get("role", "window_05_initial")),
            "window_pixel_tag": str(split.get("pixel_tag", cfg.pixel_tag)),
            "ranking_metric": "train_loss",
            "kan_grid": int(cfg.grid),
            "kan_spline_k": int(cfg.spline_k),
            "kan_base_fun": str(cfg.base_fun),
            "kan_gain_init_status": str(init_gain_status),
            "kan_gain_init_force_reference_q95_abs": float(runtime.get("train_gain_force_reference_q95_abs", float("nan"))),
            "x1_amp_guard": (
                float(cfg.random_search_x1_guard_mult) if bool(cfg.random_search_fail_fast_enabled) else float("nan")
            ),
            "x1_online_error_guard_enabled": bool(
                cfg.random_search_fail_fast_enabled and cfg.x1_online_error_guard_enabled
            ),
            "x1_online_error_guard_pct": float(cfg.x1_online_error_guard_pct),
            "x1_online_error_guard_task_pcts": [float(v) for v in cfg.x1_online_error_guard_task_pcts],
            "AFM05_true_side_available": False,
            **initial_grid_fields,
            **x3_pred_fields,
            **x3_norm_fields,
        },
        "train_parts": dict(train_parts),
        "val_parts": metric_parts,
        "ks_hat": float(ks_fixed),
        "cs_hat": float(cs_fixed),
        "is_viable": bool(np.isfinite(ranking_loss)),
        "train_reason": "",
        "val_reason": "",
        "p_net_vec": np.zeros(0, dtype=float),
        "nn_gain": float(init_gnn),
        "nn_force_mode": "kan_force_model",
        "completed_epochs": 0,
        "retry_count_total": 0,
        "early_stopped": False,
        "early_stop_reason": "",
        "trial_failed": False,
        "failure_reason": "",
        "failure_epoch": 0,
        "lr_last": float("nan"),
        "time_per_epoch": dt_total,
    }


__all__ = ["prepare_kan_stage1_runtime", "stage1pluslight_kan_random_trial"]
