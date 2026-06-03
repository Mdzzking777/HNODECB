"""Training loop for the isolated AFM04 KAN full functional test."""

from __future__ import annotations

import copy
import json
import pickle
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from AFM04.KAN_full_test.config import default_config
from AFM04.KAN_full_test.data import (
    PreparedData,
    WindowSplit,
    prepare_data,
    x3dot_init_from_split,
    x3dot_scale_from_split,
)
from AFM04.KAN_full_test.kan_backend import KANForceModule
from AFM04.KAN_full_test.losses import TorchLossParts, evaluate_split
from AFM04.KAN_full_test.optim.kft_lbfgs import LBFGS as KFTLBFGS
from AFM04.KAN_full_test.rollout import (
    force_inputs_for_module,
    fts_truth_from_states_torch,
    rollout_single_shooting_torch,
    x2dot_rhs_torch,
)

def _torch_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def _grid_update_due(epoch: int, update_num: int, start_step: int, stop_step: int) -> bool:
    if update_num <= 0 or epoch < start_step or epoch >= stop_step:
        return False
    freq = max(1, int(np.ceil(stop_step / update_num)))
    return epoch % freq == 0


def _recent_val_plateau_stats(
    history: list[dict[str, float]],
    *,
    min_epochs: int,
    compare_gap: int,
) -> dict[str, float] | None:
    if len(history) < max(int(min_epochs), int(compare_gap) + 1):
        return None
    prev = float(history[-(int(compare_gap) + 1)]["val_loss"])
    cur = float(history[-1]["val_loss"])
    if not np.isfinite(prev) or prev <= 0.0:
        return None
    if not np.isfinite(cur) or cur <= 0.0:
        return None
    rel_gap_to_prev = abs(cur - prev) / prev
    return {
        "previous": prev,
        "current": cur,
        "compare_gap": float(compare_gap),
        "rel_gap_to_prev": rel_gap_to_prev,
    }


def _window_title(role: str) -> str:
    role_norm = role.strip().lower()
    if role_norm == "modified_w0":
        return "modified W0"
    if role_norm == "first_contact":
        return "W0: first-contact window"
    if role_norm == "middle":
        return "W1: middle window"
    if role_norm == "max_x1_pp_change":
        return "window2: the most drastic region"
    if role_norm == "tail_stable":
        return "window3: stable region at the end"
    return role


def _timestamp_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_line(log, message: str) -> None:
    line = f"[{_timestamp_human()}] {message}"
    log.write(line + "\n")
    log.flush()


def _window_to_torch(split: WindowSplit, *, dtype: torch.dtype, device: str) -> dict[str, torch.Tensor]:
    return {
        "ode_full": torch.as_tensor(split.ode_full, dtype=dtype, device=device),
        "ode_train": torch.as_tensor(split.ode_train, dtype=dtype, device=device),
        "ode_val": torch.as_tensor(split.ode_val, dtype=dtype, device=device),
        "x2dot_full": torch.as_tensor(split.x2dot_full, dtype=dtype, device=device),
        "x2dot_train": torch.as_tensor(split.x2dot_train, dtype=dtype, device=device),
        "x2dot_val": torch.as_tensor(split.x2dot_val, dtype=dtype, device=device),
        "contact_full": torch.as_tensor(split.contact_full.astype(float), dtype=dtype, device=device),
        "contact_train": torch.as_tensor(split.contact_train.astype(float), dtype=dtype, device=device),
        "contact_val": torch.as_tensor(split.contact_val.astype(float), dtype=dtype, device=device),
        "times_full": torch.as_tensor(split.times_full, dtype=dtype, device=device),
        "times_train": torch.as_tensor(split.times_train, dtype=dtype, device=device),
        "times_val": torch.as_tensor(split.times_val, dtype=dtype, device=device),
        "train_idx": torch.as_tensor(split.train_idx, dtype=torch.long, device=device),
        "val_idx": torch.as_tensor(split.val_idx, dtype=torch.long, device=device),
    }


def _split_loss_indices(tensors: dict[str, torch.Tensor], part_name: str) -> torch.Tensor | None:
    if part_name == "full":
        return None
    if part_name == "train":
        return tensors["train_idx"]
    if part_name == "val":
        return tensors["val_idx"]
    raise ValueError(f"unsupported split part: {part_name!r}")


def _select_from_full(tensors: dict[str, torch.Tensor], part_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_indices = _split_loss_indices(tensors, part_name)
    if loss_indices is None:
        return tensors["ode_full"], tensors["x2dot_full"], tensors["contact_full"], tensors["times_full"]
    return (
        tensors["ode_full"][:, loss_indices],
        tensors["x2dot_full"][loss_indices],
        tensors["contact_full"][loss_indices],
        tensors["times_full"][loss_indices],
    )


def _metrics_row(total: torch.Tensor, parts: TorchLossParts) -> dict[str, float]:
    return {
        "loss": float(total.detach()),
        "state": float(parts.state),
        "x1_state": float(parts.x1_state),
        "x2_state": float(parts.x2_state),
        "x2dot": float(parts.x2dot),
        "x3_range": float(parts.x3_range),
        "fts_range": float(parts.fts_range),
        "cont": float(parts.cont),
        "x1_rec": float(parts.x1_rec),
        "x3_rec": float(parts.x3_rec),
        "fts_rollout_rec": float(parts.fts_rollout_rec),
    }


def _grad_norm(model: torch.nn.Module) -> float:
    sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        g = param.grad.detach()
        sq += float(torch.sum(g * g))
    return float(np.sqrt(max(sq, 0.0)))


def _capture_grads(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in model.named_parameters()
    }


def _restore_grads(model: torch.nn.Module, grads: dict[str, torch.Tensor | None]) -> None:
    for name, param in model.named_parameters():
        grad = grads.get(name)
        if grad is None:
            param.grad = None
        else:
            param.grad = grad.detach().clone().to(device=param.device, dtype=param.dtype)


def _capture_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }


def _param_delta_stats(
    model: torch.nn.Module,
    base_params: dict[str, torch.Tensor],
) -> tuple[float, float]:
    sq = 0.0
    max_abs = 0.0
    for name, param in model.named_parameters():
        base = base_params.get(name)
        if base is None:
            continue
        delta = param.detach() - base.to(device=param.device, dtype=param.dtype)
        sq += float(torch.sum(delta * delta))
        if delta.numel() > 0:
            max_abs = max(max_abs, float(torch.max(torch.abs(delta))))
    return float(np.sqrt(max(sq, 0.0))), float(max_abs)


def _new_group_stat() -> dict[str, float]:
    return {"sq": 0.0, "max_abs": 0.0, "numel": 0.0}


def _add_group_tensor(stats: dict[str, dict[str, float]], group: str, tensor: torch.Tensor) -> None:
    if tensor.numel() <= 0:
        return
    value = tensor.detach()
    rec = stats.setdefault(group, _new_group_stat())
    rec["sq"] += float(torch.sum(value * value))
    rec["max_abs"] = max(rec["max_abs"], float(torch.max(torch.abs(value))))
    rec["numel"] += float(value.numel())


def _q_input_slice(name: str, tensor: torch.Tensor) -> torch.Tensor | None:
    if not name.startswith("kan."):
        return None
    if ".0." not in name:
        return None
    q_index = 3
    if name.startswith("kan.act_fun.0.") and tensor.ndim >= 1 and tensor.shape[0] > q_index:
        return tensor[q_index]
    if name.startswith("kan.symbolic_fun.0.") and tensor.ndim >= 2 and tensor.shape[1] > q_index:
        return tensor[:, q_index]
    return None


def _param_group_name(name: str) -> str:
    if name == "x3dot_q_init":
        return "q_init"
    if name == "log_gnn":
        return "g_nn"
    if name.startswith("soft_mask_"):
        return "soft_mask"
    if name.startswith("kan."):
        return "kan_all"
    return "other"


def _finalize_group_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for group, rec in stats.items():
        out[group] = {
            "l2": float(np.sqrt(max(rec["sq"], 0.0))),
            "max": float(rec["max_abs"]),
            "numel": float(rec["numel"]),
        }
    return out


def _param_delta_group_stats(
    model: torch.nn.Module,
    base_params: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        base = base_params.get(name)
        if base is None:
            continue
        delta = param.detach() - base.to(device=param.device, dtype=param.dtype)
        _add_group_tensor(stats, "all", delta)
        _add_group_tensor(stats, _param_group_name(name), delta)
        q_delta = _q_input_slice(name, delta)
        if q_delta is not None:
            _add_group_tensor(stats, "kan_q_input_slice", q_delta)
    return _finalize_group_stats(stats)


def _grad_group_stats(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        _add_group_tensor(stats, "all", grad)
        _add_group_tensor(stats, _param_group_name(name), grad)
        q_grad = _q_input_slice(name, grad)
        if q_grad is not None:
            _add_group_tensor(stats, "kan_q_input_slice", q_grad)
    return _finalize_group_stats(stats)


def _fmt_group_stats(stats: dict[str, dict[str, float]], group: str) -> str:
    rec = stats.get(group, {"l2": float("nan"), "max": float("nan"), "numel": 0.0})
    return f"{group}:l2={rec['l2']:.3e},max={rec['max']:.3e},n={int(rec['numel'])}"


def _log_diag_group_stats(
    log: Path,
    *,
    prefix: str,
    stats: dict[str, dict[str, float]],
) -> None:
    groups = ("all", "kan_all", "kan_q_input_slice", "soft_mask", "g_nn", "q_init", "other")
    _log_line(log, prefix + " | " + " | ".join(_fmt_group_stats(stats, group) for group in groups))


def _grad_dot_delta(
    model: torch.nn.Module,
    base_params: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor | None],
) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        grad = grads.get(name)
        base = base_params.get(name)
        if grad is None or base is None:
            continue
        delta = param.detach() - base.to(device=param.device, dtype=param.dtype)
        total += float(torch.sum(grad.to(device=param.device, dtype=param.dtype) * delta))
    return total


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Clear Adam/AMSGrad momentum buffers while preserving param groups."""
    optimizer.state.clear()


def _make_optimizer(cfg, model: torch.nn.Module) -> tuple[torch.optim.Optimizer, str]:
    optimizer_name = str(cfg.optimizer_name).strip().lower()
    amsgrad = optimizer_name == "amsgrad"
    if optimizer_name not in ("adam", "amsgrad"):
        optimizer_name = "amsgrad"
        amsgrad = True
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        amsgrad=amsgrad,
    )
    return optimizer, optimizer_name


def _make_lbfgs_optimizer(cfg, model: torch.nn.Module) -> tuple[torch.optim.Optimizer, str]:
    line_search_fn = "strong_wolfe" if bool(cfg.lbfgs_strong_wolfe) else None
    max_iter = max(1, int(cfg.lbfgs_max_iter))
    max_eval = max(max_iter, int(cfg.lbfgs_max_eval))
    optimizer = KFTLBFGS(
        model.parameters(),
        lr=float(cfg.lbfgs_lr),
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=max(1, int(cfg.lbfgs_history_size)),
        tolerance_grad=float(cfg.lbfgs_tolerance_grad),
        tolerance_change=float(cfg.lbfgs_tolerance_change),
        line_search_fn=line_search_fn,
        ys_threshold=float(cfg.lbfgs_ys_threshold),
    )
    return optimizer, ("strong_wolfe" if line_search_fn == "strong_wolfe" else "none")


def _lbfgs_internal_debug(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    params = list(getattr(optimizer, "_params", []))
    if not params:
        return {}
    state = optimizer.state.get(params[0], {})
    debug = state.get("last_step_debug", {})
    if not isinstance(debug, dict):
        return {}
    return copy.deepcopy(debug)


def _dbg_float(debug: dict[str, Any], key: str) -> float:
    try:
        return float(debug.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _terminal_step_failure(reason: str) -> bool:
    return reason == "step_trial_loss_nonfinite" or reason.startswith("step_trial_exception:")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _write_event(event_log, event: dict[str, Any]) -> None:
    event_log.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
    event_log.flush()


def _role_short(role: str) -> str:
    role_norm = role.strip().lower()
    if role_norm == "modified_w0":
        return "modified W0"
    if role_norm == "first_contact":
        return "W0"
    if role_norm == "middle":
        return "W1"
    if role_norm == "max_x1_pp_change":
        return "W2"
    if role_norm == "tail_stable":
        return "W3"
    return role


def _window_file_tag(role: str) -> str:
    return _role_short(role).strip().lower().replace(" ", "_")


def _selected_window_indices(cfg, splits: tuple[WindowSplit, ...]) -> list[int]:
    requested = int(getattr(cfg, "train_window_index", 0))
    if requested <= 0:
        return list(range(1, len(splits) + 1))
    if requested > len(splits):
        raise ValueError(f"invalid train_window_index={requested}; available windows=1..{len(splits)}")
    return [requested]


def _observable_grid_inputs_from_ode(
    *,
    ode_train: torch.Tensor,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    cfg,
) -> torch.Tensor:
    """Build AGU fallback samples in raw physical coordinates.

    x1/x2 come from observed data.  Hidden axes use conservative non-teacher
    physical priors, not true x3 or true x3dot statistics.
    """
    n = int(ode_train.shape[1])
    input_dim = int(getattr(cfg, "width", (3, 7, 1))[0])
    inputs = torch.empty((n, input_dim), dtype=ode_train.dtype, device=ode_train.device)
    inputs[:, 0:2] = ode_train[0:2, :].transpose(0, 1)

    if input_dim >= 3:
        x1_abs = float(torch.max(torch.abs(ode_train[0, :])).detach().cpu()) if n > 0 else 0.0
        x3_amp = max(0.1 * x1_abs, 1.0e-12)
        if n <= 1:
            inputs[:, 2] = 0.0
        else:
            inputs[:, 2] = torch.linspace(-x3_amp, x3_amp, n, dtype=ode_train.dtype, device=ode_train.device)

    if input_dim >= 4:
        x2_abs = float(torch.max(torch.abs(ode_train[1, :])).detach().cpu()) if n > 0 else 0.0
        q_amp = max(0.1 * x2_abs, 1.0e-12)
        if n <= 1:
            inputs[:, 3] = 0.0
        else:
            idx = torch.arange(n, dtype=ode_train.dtype, device=ode_train.device)
            frac = torch.remainder((idx + 0.5) * 0.6180339887498949, 1.0)
            inputs[:, 3] = q_amp * (2.0 * frac - 1.0)

    for axis in range(4, input_dim):
        if n <= 1:
            inputs[:, axis] = 0.0
        else:
            idx = torch.arange(n, dtype=ode_train.dtype, device=ode_train.device)
            frac = torch.remainder((idx + 0.5 + 0.137 * axis) * 0.6180339887498949, 1.0)
            inputs[:, axis] = 2.0 * frac - 1.0
    return inputs


def _model_normalizer_arrays(model: KANForceModule) -> tuple[np.ndarray, np.ndarray]:
    return (
        model.state_mean.detach().cpu().numpy().astype(float).copy(),
        model.state_scale.detach().cpu().numpy().astype(float).copy(),
    )


def _make_grid_update_inputs(
    *,
    cfg,
    base_inputs: torch.Tensor | None = None,
    known_pars: tuple[float, ...],
    model: KANForceModule | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
    mech_true: torch.Tensor | None = None,
    ode_method: str | None = None,
    ode_rtol: float | None = None,
    ode_atol: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    train_wpred_enabled = bool(getattr(cfg, "train_wpred_enabled", getattr(cfg, "wpred_enabled", False)))
    source = "raw_observed_x1x2_neutral_x3_q_axes"
    grid_inputs: torch.Tensor | None = None
    rollout_grid_enabled = bool(getattr(cfg, "rollout_grid_update_enabled", False))
    if rollout_grid_enabled and model is not None and tensors is not None and mech_true is not None:
        try:
            rollout_out = rollout_single_shooting_torch(
                model,
                known_pars,
                mech_true,
                tensors["ode_full"][:, 0],
                tensors["times_full"],
                method=str(ode_method or getattr(cfg, "ode_method", "dopri5")),
                rtol=float(ode_rtol if ode_rtol is not None else getattr(cfg, "ode_rtol", 1.0e-10)),
                atol=float(ode_atol if ode_atol is not None else getattr(cfg, "ode_atol", 1.0e-12)),
                return_aux=True,
            )
            traj, aux = rollout_out
            train_idx = tensors["train_idx"].to(dtype=torch.long, device=traj.device)
            states = traj.transpose(0, 1)
            q_pre = aux.q_pre
            if q_pre is not None:
                q_pre = q_pre.to(device=states.device, dtype=states.dtype)
            raw_inputs = force_inputs_for_module(model, states, q_pre)
            grid_inputs = raw_inputs[train_idx, :].detach()
            source = "raw_current_rollout_pred_x3_qpre"
        except Exception:
            grid_inputs = None
            source = "raw_observed_x1x2_neutral_x3_q_axes_fallback"
    if grid_inputs is None:
        if base_inputs is None:
            raise ValueError("grid update requires base_inputs")
        grid_inputs = base_inputs.detach()
    n = float(grid_inputs.shape[0])
    meta = {
        "source": source,
        "base_samples": n,
        "extra_samples": 0.0,
        "total_samples": n,
        "mean_w_pred": float("nan"),
        "max_w_pred": float("nan"),
        "q50_w_pred": float("nan"),
        "q90_w_pred": float("nan"),
        "q99_w_pred": float("nan"),
        "frac_s_le_a0": float("nan"),
        "frac_s_le_1p5a0": float("nan"),
        "frac_wpred_ge_05": float("nan"),
        "frac_wpred_ge_08": float("nan"),
    }
    if train_wpred_enabled:
        meta["wpred_meta_skipped"] = 1.0
    return grid_inputs, meta


class _OracleForceModule(torch.nn.Module):
    def __init__(
        self,
        known_pars: tuple[float, ...],
        *,
        eta_star_true: float,
        mech_true: torch.Tensor,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        super().__init__()
        self.known_pars = known_pars
        self.eta_star_true = float(eta_star_true)
        self.register_buffer("_gain_one", torch.ones(1, dtype=dtype, device=device))
        self.register_buffer("mech_true", torch.as_tensor(mech_true, dtype=dtype, device=device).reshape(2))

    def gain(self) -> torch.Tensor:
        return self._gain_one

    def raw_output(self, states: torch.Tensor) -> torch.Tensor:
        return fts_truth_from_states_torch(
            states,
            self.known_pars,
            eta_star=self.eta_star_true,
            mech_true=self.mech_true,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.raw_output(states)


def _oracle_nn_metrics(
    *,
    known_pars: tuple[float, ...],
    eta_star_true: float,
    mech_true: torch.Tensor,
    states_full: torch.Tensor,
) -> dict[str, float]:
    raw = fts_truth_from_states_torch(
        states_full,
        known_pars,
        eta_star=eta_star_true,
        mech_true=mech_true,
    ).reshape(-1)
    raw_min = float(torch.min(raw).detach())
    raw_max = float(torch.max(raw).detach())
    raw_mean = float(torch.mean(raw).detach())
    raw_neg_frac = float(torch.mean((raw < 0.0).to(raw.dtype)).detach())
    return {
        "fcontact_err": 0.0,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_mean": raw_mean,
        "raw_neg_frac": raw_neg_frac,
    }


def _run_oracle_sanity(
    *,
    log,
    split: WindowSplit,
    tensors: dict[str, torch.Tensor],
    known_pars: tuple[float, ...],
    eta_star_true: float,
    mech_true: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
) -> dict[str, Any]:
    _log_line(log, "=== KAN FULL TEST SANITY (oracle F_contact + true mech) ===")
    oracle = _OracleForceModule(
        known_pars,
        eta_star_true=eta_star_true,
        mech_true=mech_true,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        sanity_loss, sanity_parts, _ = evaluate_split(
            force_module=oracle,
            known_pars=known_pars,
            mech_true=mech_true,
            ode_true=tensors["ode_full"],
            x2dot_true=tensors["x2dot_full"],
            contact_mask=tensors["contact_full"],
            times=tensors["times_full"],
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            eta_star_true=eta_star_true,
            loss_indices=tensors["train_idx"],
        )
        nn_metrics = _oracle_nn_metrics(
            known_pars=known_pars,
            eta_star_true=eta_star_true,
            mech_true=mech_true,
            states_full=tensors["ode_full"].transpose(0, 1),
        )

    _log_line(log, f"  sanity loss={float(sanity_loss.detach()):.6e}")
    _log_line(
        log,
        "  sanity parts: "
        f"state={float(sanity_parts.state):.3e} "
        f"x2dot={float(sanity_parts.x2dot):.3e} "
        f"x3r={float(sanity_parts.x3_range):.3e} "
        f"ftsr={float(sanity_parts.fts_range):.3e} "
        f"cont={float(sanity_parts.cont):.3e}",
    )
    _log_line(log, f"  sanity rec: x1={float(sanity_parts.x1_rec):.2f}% x3={float(sanity_parts.x3_rec):.2f}%")
    _log_line(log, f"  sanity nn: F_contact err={float(nn_metrics['fcontact_err']):.2f}%")
    return {
        "loss": float(sanity_loss.detach()),
        "parts": {
            "state": float(sanity_parts.state),
            "x1_state": float(sanity_parts.x1_state),
            "x2_state": float(sanity_parts.x2_state),
            "x2dot": float(sanity_parts.x2dot),
            "x3_range": float(sanity_parts.x3_range),
            "fts_range": float(sanity_parts.fts_range),
            "cont": float(sanity_parts.cont),
            "x1_rec": float(sanity_parts.x1_rec),
            "x3_rec": float(sanity_parts.x3_rec),
            "fts_rollout_rec": float(sanity_parts.fts_rollout_rec),
        },
        "nn": nn_metrics,
        "window_meta": {
            "role": split.role,
            "label": split.label,
            "title": _window_title(split.role),
            "start_idx": split.start_idx,
            "stop_idx": split.stop_idx,
            "t_start": split.t_start,
            "t_stop": split.t_stop,
        },
    }


def _snapshot_split(
    *,
    force_module: KANForceModule,
    known_pars: tuple[float, ...],
    eta_star_true: float,
    mech_true: torch.Tensor,
    split: WindowSplit,
    tensors: dict[str, torch.Tensor],
    dtype: torch.dtype,
    device: str,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "role": split.role,
        "label": split.label,
        "title": _window_title(split.role),
        "window_short": _role_short(split.role),
        "start_idx": split.start_idx,
        "stop_idx": split.stop_idx,
        "t_start": split.t_start,
        "t_stop": split.t_stop,
    }

    with torch.no_grad():
        for part_name in ("full", "train", "val"):
            loss_indices = _split_loss_indices(tensors, part_name)
            total, parts, traj = evaluate_split(
                force_module=force_module,
                known_pars=known_pars,
                mech_true=mech_true,
                ode_true=tensors["ode_full"],
                x2dot_true=tensors["x2dot_full"],
                contact_mask=tensors["contact_full"],
                times=tensors["times_full"],
                ode_method=ode_method,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                eta_star_true=eta_star_true,
                loss_indices=loss_indices,
            )
            ode_true, x2dot_true, contact_mask, times = _select_from_full(tensors, part_name)
            q_pre = getattr(traj, "_plan_z_q_pre", None)
            if loss_indices is not None:
                if q_pre is not None:
                    q_pre = q_pre[loss_indices]
                traj = traj[:, loss_indices]
            x2dot_pred = x2dot_rhs_torch(traj, times, force_module, known_pars, q_pre=q_pre)
            teacher_states = ode_true.transpose(0, 1)
            fts_teacher_true = fts_truth_from_states_torch(
                teacher_states,
                known_pars,
                eta_star=eta_star_true,
                mech_true=mech_true,
            )
            rollout_states = traj.transpose(0, 1)
            rollout_inputs = force_inputs_for_module(force_module, rollout_states, q_pre)
            fts_rollout_pred = force_module(rollout_inputs)
            nn_raw_rollout = force_module.raw_output(rollout_inputs)
            w_pred_rollout = force_module.w_pred(rollout_inputs)
            nn_weighted_rollout = force_module.weighted_raw_output(rollout_inputs)
            soft_mask_rollout = force_module.soft_mask(rollout_inputs)
            fts_unmasked_rollout = force_module.gain() * nn_weighted_rollout
            out[part_name] = {
                "metrics": _metrics_row(total, parts),
                "times": times.detach().cpu().numpy(),
                "ode_true": ode_true.detach().cpu().numpy(),
                "traj_pred": traj.detach().cpu().numpy(),
                "x2dot_true": x2dot_true.detach().cpu().numpy(),
                "x2dot_pred": x2dot_pred.detach().cpu().numpy(),
                "contact_mask": contact_mask.detach().cpu().numpy(),
                "fts_teacher_true": fts_teacher_true.detach().cpu().numpy(),
                "fts_rollout_pred": fts_rollout_pred.detach().cpu().numpy(),
                "fts_unmasked_rollout": fts_unmasked_rollout.detach().cpu().numpy(),
                "nn_raw_rollout": nn_raw_rollout.detach().cpu().numpy(),
                "w_pred_rollout": w_pred_rollout.detach().cpu().numpy(),
                "nn_weighted_rollout": nn_weighted_rollout.detach().cpu().numpy(),
                "soft_mask_rollout": soft_mask_rollout.detach().cpu().numpy(),
            }
            if q_pre is not None:
                out[part_name]["x3dot_q_pre"] = q_pre.detach().cpu().numpy()
        out["g_nn"] = float(force_module.gain().detach().cpu().item())
        out["soft_mask"] = force_module.soft_mask_summary()
        if hasattr(force_module, "x3dot_summary"):
            out["x3dot_input"] = force_module.x3dot_summary()
    return out


def _save_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _save_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _handoff_path(cfg, file_tag: str) -> Path:
    raw = str(getattr(cfg, "handoff_path", "")).strip()
    if raw:
        return Path(raw)
    return cfg.checkpoint_dir / f"kan_full_test_handoff_{file_tag}.pt"


def _handoff_summary_path(cfg, file_tag: str) -> Path:
    return cfg.result_dir / f"kan_full_test_handoff_{file_tag}_summary.json"


def _handoff_epoch_path(cfg, file_tag: str, handoff_epoch: int) -> Path:
    raw = str(getattr(cfg, "handoff_path", "")).strip()
    if raw:
        raw_path = Path(raw)
        return raw_path.with_name(f"{raw_path.stem}_adam{handoff_epoch}{raw_path.suffix}")
    return cfg.checkpoint_dir / f"kan_full_test_handoff_{file_tag}_adam{handoff_epoch}.pt"


def _handoff_epoch_summary_path(cfg, file_tag: str, handoff_epoch: int) -> Path:
    return cfg.result_dir / f"kan_full_test_handoff_{file_tag}_adam{handoff_epoch}_summary.json"


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Any) -> list[str]:
    restored: list[str] = []
    if not isinstance(state, dict):
        return restored
    try:
        if "python_random" in state:
            random.setstate(state["python_random"])
            restored.append("python")
    except Exception:
        pass
    try:
        if "numpy_random" in state:
            np.random.set_state(state["numpy_random"])
            restored.append("numpy")
    except Exception:
        pass
    try:
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
            restored.append("torch_cpu")
    except Exception:
        pass
    try:
        cuda_state = state.get("torch_cuda_all")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
            restored.append("torch_cuda_all")
    except Exception:
        pass
    return restored


def _window_meta_dict(split: WindowSplit) -> dict[str, Any]:
    return {
        "role": split.role,
        "label": split.label,
        "title": _window_title(split.role),
        "start_idx": split.start_idx,
        "stop_idx": split.stop_idx,
        "t_start": split.t_start,
        "t_stop": split.t_stop,
    }


def _save_handoff_checkpoint(
    *,
    path: Path,
    summary_path: Path,
    epoch_path: Path | None,
    epoch_summary_path: Path | None,
    cfg,
    model: KANForceModule,
    optimizer: torch.optim.Optimizer,
    split: WindowSplit,
    history: list[dict[str, Any]],
    best_payload: dict[str, Any] | None,
    best_epoch: int,
    best_val: float,
    sanity_payload: dict[str, Any] | None,
    prepared: PreparedData,
    lr: float,
    grad_ema: float,
    grad_target: float,
    recent_losses: list[float],
    train_loss_last: float,
    warmstart_payload: dict[str, Any] | None,
) -> None:
    handoff_epoch = int(len(history))
    payload = {
        "format": "kft_adam_to_lbfgs_handoff_v1",
        "epoch": handoff_epoch,
        "state_dict": copy.deepcopy(model.state_dict()),
        "adam_optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "history": copy.deepcopy(history),
        "best_payload": copy.deepcopy(best_payload),
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
        "sanity": copy.deepcopy(sanity_payload),
        "config": asdict(cfg),
        "window_meta": _window_meta_dict(split),
        "state_mean": _model_normalizer_arrays(model)[0],
        "state_scale": _model_normalizer_arrays(model)[1],
        "hidden_normalizer": {
            "enabled": bool(getattr(model, "x3dot_input_enabled", False)),
            "source": "active_model_at_handoff",
            "new_state_mean": _model_normalizer_arrays(model)[0],
            "new_state_scale": _model_normalizer_arrays(model)[1],
        },
        "known_pars": prepared.known_pars,
        "eta_star_true": prepared.eta_star_true,
        "mech_true": prepared.mech_true,
        "lr": float(lr),
        "grad_ema": float(grad_ema),
        "grad_target": float(grad_target),
        "recent_losses": list(recent_losses),
        "train_loss_last": float(train_loss_last),
        "warmstart_payload": copy.deepcopy(warmstart_payload),
        "rng_state": _capture_rng_state(),
    }
    _save_torch(path, payload)
    if epoch_path is not None and epoch_path != path:
        _save_torch(epoch_path, payload)

    summary = {
        "format": payload["format"],
        "path": str(path),
        "epoch_tagged_path": str(epoch_path) if epoch_path is not None else "",
        "epoch": handoff_epoch,
        "window_meta": payload["window_meta"],
        "adam_epochs_config": int(cfg.adam_epochs),
        "lbfgs_epochs_config": int(cfg.lbfgs_steps),
        "history_rows": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
        "train_loss_last": float(train_loss_last),
        "lr": float(lr),
        "soft_mask": model.soft_mask_summary(),
        "warmstart_trial": (
            int(warmstart_payload["best_trial"]["trial"])
            if isinstance(warmstart_payload, dict)
            and isinstance(warmstart_payload.get("best_trial"), dict)
            and "trial" in warmstart_payload["best_trial"]
            else None
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if epoch_summary_path is not None and epoch_summary_path != summary_path:
        epoch_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with epoch_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def _load_handoff_checkpoint(path: Path, *, cfg, file_tag: str, split: WindowSplit) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing KFT handoff checkpoint: {path}")
    payload = torch.load(path, map_location=cfg.device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid KFT handoff checkpoint payload: {path}")
    if payload.get("format") != "kft_adam_to_lbfgs_handoff_v1":
        raise RuntimeError(f"Unsupported KFT handoff checkpoint format: {payload.get('format')!r}")
    if "state_dict" not in payload:
        raise RuntimeError(f"KFT handoff checkpoint missing state_dict: {path}")
    window_meta = payload.get("window_meta", {})
    if isinstance(window_meta, dict) and str(window_meta.get("role", "")) != str(split.role):
        raise RuntimeError(
            "KFT handoff window mismatch: "
            f"checkpoint_role={window_meta.get('role')!r} current_role={split.role!r} path={path}"
        )
    handoff_epoch = int(payload.get("epoch", -1))
    if handoff_epoch != int(cfg.adam_epochs):
        raise RuntimeError(
            "KFT handoff epoch mismatch: "
            f"checkpoint_epoch={handoff_epoch} cfg.adam_epochs={cfg.adam_epochs} "
            f"file_tag={file_tag} path={path}"
        )
    return payload


def _load_prerun_candidate_rows(cfg, *, tag: str) -> tuple[Path, list[dict[str, Any]]] | None:
    summary_path = cfg.prerun_result_dir / f"kan_full_test_prerun_summary_{tag}.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("candidates", []) if isinstance(payload, dict) else []
        if isinstance(rows, list) and rows:
            return summary_path, [dict(row) for row in rows if isinstance(row, dict)]

    for path in (
        cfg.prerun_result_dir / f"kan_full_test_prerun_candidates_b_{tag}.pkl",
        cfg.prerun_result_dir / f"kan_full_test_prerun_{tag}.pkl",
    ):
        if not path.is_file():
            continue
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            continue
        rows = payload.get("candidate_records", payload.get("candidate_b_records", []))
        if isinstance(rows, list) and rows:
            return path, [dict(row) for row in rows if isinstance(row, dict)]
    return None


def _prerun_candidate_rank(row: dict[str, Any], fallback: int) -> int:
    for key in ("candidate_rank", "candidate_b", "candidate"):
        try:
            value = int(row.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return fallback


def _candidate_loss_value(row: dict[str, Any]) -> float:
    for key in ("candidate_loss", "best_train_loss", "final_train_loss"):
        try:
            value = float(row.get(key, float("inf")))
        except (TypeError, ValueError):
            value = float("inf")
        if np.isfinite(value):
            return value
    return float("inf")


def _maybe_load_prerun_candidate_warmstart(
    *,
    cfg,
    model: KANForceModule,
    warmstart_window_index: int,
    log,
) -> dict[str, Any] | None:
    if not getattr(cfg, "warmstart_from_prerun", False):
        return None
    prepared = prepare_data(cfg)
    if warmstart_window_index < 1 or warmstart_window_index > len(prepared.splits):
        raise ValueError(f"invalid warmstart_window_index={warmstart_window_index}; available windows=1..{len(prepared.splits)}")
    split = prepared.splits[warmstart_window_index - 1]
    tag = _window_file_tag(split.role)
    loaded = _load_prerun_candidate_rows(cfg, tag=tag)
    if loaded is None:
        if log is not None:
            _log_line(log, f"warmstart from prerun: no candidate result found for {tag}; trying random-search warmstart")
        return None

    source_path, rows = loaded
    ranked_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        if bool(row.get("failed", False)):
            continue
        loss = _candidate_loss_value(row)
        if not np.isfinite(loss):
            continue
        row = dict(row)
        row["_candidate_rank_resolved"] = _prerun_candidate_rank(row, idx)
        row["_candidate_loss_resolved"] = loss
        ranked_rows.append(row)
    ranked_rows.sort(key=lambda row: (int(row["_candidate_rank_resolved"]), float(row["_candidate_loss_resolved"])))

    requested_candidate = int(getattr(cfg, "prerun_warmstart_candidate", 1))
    selected_row = None
    for row in ranked_rows:
        if int(row["_candidate_rank_resolved"]) == requested_candidate:
            selected_row = row
            break
    if selected_row is None and 1 <= requested_candidate <= len(ranked_rows):
        selected_row = ranked_rows[requested_candidate - 1]
    if selected_row is None:
        if log is not None:
            _log_line(
                log,
                "warmstart from prerun: requested candidate not available; "
                f"candidate={requested_candidate} source={source_path}; trying random-search warmstart",
            )
        return None

    from AFM04.KAN_full_test.random_search import _rebuild_trial_state_dict

    trial = int(selected_row["trial"])
    seed = int(selected_row.get("seed", cfg.seed + trial))
    state_dict = _rebuild_trial_state_dict(
        cfg=cfg,
        prepared=prepared,
        split=split,
        seed=seed,
        dtype=next(model.parameters()).dtype,
    )
    model.load_state_dict(state_dict)
    candidate_rank = int(selected_row["_candidate_rank_resolved"])
    candidate_loss = float(selected_row["_candidate_loss_resolved"])
    payload = {
        "best_trial": {
            "trial": int(trial),
            "seed": int(seed),
            "ranking_metric": "prerun_best_train_loss",
            "ranking_loss": float(candidate_loss),
            "train_loss": float(selected_row.get("best_train_loss", candidate_loss)),
            "val_loss": float(selected_row.get("best_val_loss", float("nan"))),
        },
        "prerun_candidate": dict(selected_row),
        "best_state_dict": state_dict,
        "window_meta": {
            "role": split.role,
            "label": split.label,
            "title": _window_title(split.role),
            "start_idx": split.start_idx,
            "stop_idx": split.stop_idx,
            "t_start": split.t_start,
            "t_stop": split.t_stop,
        },
        "config": asdict(cfg),
        "state_mean": prepared.state_mean,
        "state_scale": prepared.state_scale,
        "known_pars": prepared.known_pars,
        "mech_true": prepared.mech_true,
        "merge_meta": {
            "window_index": warmstart_window_index,
            "selection_mode": "prerun_candidate_from_rs_initial_state",
            "source_window_index": warmstart_window_index,
            "candidate": candidate_rank,
            "trial": int(trial),
        },
    }
    if log is not None:
        _log_line(
            log,
            "warmstart from prerun candidate: "
            f"source={source_path} candidate={candidate_rank} "
            f"trial={trial} seed={seed} "
            f"rs_rank={int(selected_row.get('rs_rank', 0))} "
            f"layer_a_winner={int(selected_row.get('layer_a_winner_rank', 0))} "
            f"best_train={float(selected_row.get('best_train_loss', candidate_loss)):.6e} "
            f"best_val={float(selected_row.get('best_val_loss', float('nan'))):.6e} "
            "state=rebuilt_original_rs_trial_state epoch_start=1",
        )
    return payload


def _maybe_load_random_search_warmstart(
    *,
    cfg,
    model: KANForceModule,
    warmstart_window_index: int,
    log,
) -> dict[str, Any] | None:
    if not getattr(cfg, "warmstart_from_random_search", False):
        return None
    prepared = prepare_data(cfg)
    if warmstart_window_index < 1 or warmstart_window_index > len(prepared.splits):
        raise ValueError(f"invalid warmstart_window_index={warmstart_window_index}; available windows=1..{len(prepared.splits)}")
    split = prepared.splits[warmstart_window_index - 1]
    tag = _window_file_tag(split.role)
    requested_trial = int(getattr(cfg, "random_search_warmstart_trial", 0))
    if requested_trial > 0:
        from AFM04.KAN_full_test.random_search import _rebuild_trial_state_dict

        trials_path = cfg.random_search_result_dir / f"kan_full_test_random_search_trials_{tag}.json"
        if not trials_path.is_file():
            raise FileNotFoundError(f"Missing random-search trial ranking file: {trials_path}")
        with trials_path.open("r", encoding="utf-8") as f:
            trial_rows = json.load(f)
        selected_row = None
        for row in trial_rows:
            if int(row.get("trial", -1)) == requested_trial:
                selected_row = dict(row)
                break
        if selected_row is None:
            raise ValueError(f"Requested random-search warmstart trial {requested_trial} was not found in {trials_path}")
        if bool(selected_row.get("failed", False)):
            raise ValueError(f"Requested random-search warmstart trial {requested_trial} failed: {selected_row.get('failure_reason')}")
        ranking_loss = float(selected_row.get("ranking_loss", selected_row.get("train_loss", float("inf"))))
        train_loss = float(selected_row.get("train_loss", float("inf")))
        if not np.isfinite(ranking_loss) or not np.isfinite(train_loss):
            raise ValueError(f"Requested random-search warmstart trial {requested_trial} has non-finite loss")
        seed = int(selected_row.get("seed", cfg.seed + requested_trial))
        state_dict = _rebuild_trial_state_dict(
            cfg=cfg,
            prepared=prepared,
            split=split,
            seed=seed,
            dtype=next(model.parameters()).dtype,
        )
        model.load_state_dict(state_dict)
        payload = {
            "best_trial": selected_row,
            "best_state_dict": state_dict,
            "window_meta": {
                "role": split.role,
                "label": split.label,
                "title": _window_title(split.role),
                "start_idx": split.start_idx,
                "stop_idx": split.stop_idx,
                "t_start": split.t_start,
                "t_stop": split.t_stop,
            },
            "config": asdict(cfg),
            "state_mean": prepared.state_mean,
            "state_scale": prepared.state_scale,
            "known_pars": prepared.known_pars,
            "mech_true": prepared.mech_true,
            "merge_meta": {
                "window_index": warmstart_window_index,
                "selection_mode": "manual_trial_override",
                "source_window_index": warmstart_window_index,
                "common_trial": requested_trial,
            },
        }
        if log is not None:
            ranking_metric = str(selected_row.get("ranking_metric", "train_loss"))
            _log_line(
                log,
                "warmstart from random search trial override: "
                f"trials={trials_path} trial={requested_trial} seed={seed} "
                f"rank({ranking_metric})={ranking_loss:.6e} "
                f"train={train_loss:.6e} "
                f"val={float(selected_row.get('val_loss', float('nan'))):.6e} "
                "selection=manual_trial_override",
            )
        return payload

    best_path = cfg.random_search_checkpoint_dir / f"kan_full_test_random_search_best_{tag}.pt"
    if not best_path.is_file():
        from AFM04.KAN_full_test.random_search import _rebuild_trial_state_dict

        trials_path = cfg.random_search_result_dir / f"kan_full_test_random_search_trials_{tag}.json"
        if trials_path.is_file():
            with trials_path.open("r", encoding="utf-8") as f:
                trial_rows = json.load(f)
            if isinstance(trial_rows, list):
                valid_rows = []
                for row in trial_rows:
                    if not isinstance(row, dict) or bool(row.get("failed", False)):
                        continue
                    ranking_loss = float(row.get("ranking_loss", row.get("train_loss", float("inf"))))
                    train_loss = float(row.get("train_loss", float("inf")))
                    if np.isfinite(ranking_loss) and np.isfinite(train_loss):
                        valid_rows.append(dict(row))
                valid_rows.sort(key=lambda row: float(row.get("ranking_loss", row.get("train_loss", float("inf")))))
                if valid_rows:
                    selected_row = valid_rows[0]
                    trial = int(selected_row["trial"])
                    seed = int(selected_row.get("seed", cfg.seed + trial))
                    state_dict = _rebuild_trial_state_dict(
                        cfg=cfg,
                        prepared=prepared,
                        split=split,
                        seed=seed,
                        dtype=next(model.parameters()).dtype,
                    )
                    model.load_state_dict(state_dict)
                    payload = {
                        "best_trial": selected_row,
                        "best_state_dict": state_dict,
                        "window_meta": {
                            "role": split.role,
                            "label": split.label,
                            "title": _window_title(split.role),
                            "start_idx": split.start_idx,
                            "stop_idx": split.stop_idx,
                            "t_start": split.t_start,
                            "t_stop": split.t_stop,
                        },
                        "config": asdict(cfg),
                        "state_mean": prepared.state_mean,
                        "state_scale": prepared.state_scale,
                        "known_pars": prepared.known_pars,
                        "mech_true": prepared.mech_true,
                        "merge_meta": {
                            "window_index": warmstart_window_index,
                            "selection_mode": "random_search_rank1_rebuilt_from_trials",
                            "source_window_index": warmstart_window_index,
                            "common_trial": trial,
                        },
                    }
                    if log is not None:
                        ranking_metric = str(selected_row.get("ranking_metric", "train_loss"))
                        _log_line(
                            log,
                            "warmstart from random search rank1 rebuilt from trials: "
                            f"trials={trials_path} trial={trial} seed={seed} "
                            f"rank({ranking_metric})={float(selected_row.get('ranking_loss', float('nan'))):.6e} "
                            f"train={float(selected_row.get('train_loss', float('nan'))):.6e} "
                            f"val={float(selected_row.get('val_loss', float('nan'))):.6e} "
                            "selection=random_search_rank1_rebuilt_from_trials",
                        )
                    return payload
        if getattr(cfg, "warmstart_fallback_random", False):
            if log is not None:
                _log_line(
                    log,
                    "warmstart from random search: missing checkpoint, fallback to random init | "
                    f"path={best_path}",
                )
            return None
        raise FileNotFoundError(f"Missing random-search warmstart checkpoint: {best_path}")
    payload = torch.load(best_path, map_location=cfg.device, weights_only=False)
    payload_cfg = payload.get("config", {})
    if bool(getattr(cfg, "x3dot_input_enabled", False)) and isinstance(payload_cfg, dict):
        payload_init_policy = str(payload_cfg.get("x3dot_init_policy", "")).strip().lower()
        current_init_policy = str(getattr(cfg, "x3dot_init_policy", "")).strip().lower()
        if payload_init_policy != current_init_policy:
            raise RuntimeError(
                "Random-search checkpoint is incompatible with current Plan Z settings. "
                f"checkpoint init_policy={payload_init_policy or 'NA'}; "
                f"current init_policy={current_init_policy or 'NA'}. "
                "Clear KFT random-search outputs and rerun KFT-RS."
            )
    state_dict = payload.get("best_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Random-search checkpoint missing 'best_state_dict': {best_path}")
    model.load_state_dict(state_dict)
    if log is not None:
        best_trial = payload.get("best_trial", {})
        ranking_metric = str(best_trial.get("ranking_metric", "val_loss"))
        merge_meta = payload.get("merge_meta", {})
        selection_mode = str(merge_meta.get("selection_mode", "")).strip()
        common_trial = merge_meta.get("common_trial", None)
        source_window_index = merge_meta.get("source_window_index", None)
        _log_line(
            log,
            "warmstart from random search: "
            f"path={best_path} trial={best_trial.get('trial', 'NA')} "
            f"seed={best_trial.get('seed', 'NA')} "
            f"rank({ranking_metric})={best_trial.get('ranking_loss', float('nan')):.6e} "
            f"train={best_trial.get('train_loss', float('nan')):.6e} "
            f"val={best_trial.get('val_loss', float('nan')):.6e} "
            f"selection={selection_mode or 'legacy'} "
            f"source_window={source_window_index if source_window_index is not None else 'NA'} "
            f"common_trial={common_trial if common_trial is not None else 'NA'}",
        )
    return payload


def _maybe_load_kft_warmstart(
    *,
    cfg,
    model: KANForceModule,
    warmstart_window_index: int,
    log,
) -> dict[str, Any] | None:
    if int(getattr(cfg, "random_search_warmstart_trial", 0)) > 0:
        return _maybe_load_random_search_warmstart(
            cfg=cfg,
            model=model,
            warmstart_window_index=warmstart_window_index,
            log=log,
        )
    prerun_payload = _maybe_load_prerun_candidate_warmstart(
        cfg=cfg,
        model=model,
        warmstart_window_index=warmstart_window_index,
        log=log,
    )
    if prerun_payload is not None:
        return prerun_payload
    return _maybe_load_random_search_warmstart(
        cfg=cfg,
        model=model,
        warmstart_window_index=warmstart_window_index,
        log=log,
    )


def _make_viz_payload(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("best")
    best_viz = None
    if isinstance(best, dict):
        best_viz = {key: value for key, value in best.items() if key != "state_dict"}
    return {
        "best": best_viz,
        "history": payload.get("history", []),
        "state_mean": payload.get("state_mean"),
        "state_scale": payload.get("state_scale"),
        "known_pars": payload.get("known_pars"),
        "eta_star_true": payload.get("eta_star_true"),
        "mech_true": payload.get("mech_true"),
        "sanity": payload.get("sanity"),
        "hidden_normalizer": payload.get("hidden_normalizer"),
        "window_meta": payload.get("window_meta"),
        "final_snapshot": payload.get("final_snapshot"),
        "config": payload.get("config"),
        "best_epoch": payload.get("best_epoch"),
        "best_val_loss": payload.get("best_val_loss"),
    }


def run_full_test_shard(cfg=None, *, shard_index: int | None = None, log_path: Path | None = None) -> dict[str, object]:
    cfg = default_config() if cfg is None else cfg
    train_wpred_enabled = bool(getattr(cfg, "train_wpred_enabled", getattr(cfg, "wpred_enabled", False)))
    shard_index = int(cfg.shard_index if shard_index is None else shard_index)
    dtype = _torch_dtype(cfg.dtype)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    prepared = prepare_data(cfg)
    selected_window_indices = _selected_window_indices(cfg, prepared.splits)
    if cfg.shard_count != len(selected_window_indices):
        raise ValueError(
            f"train shard_count={cfg.shard_count} must match selected windows={len(selected_window_indices)}"
        )
    if shard_index < 1 or shard_index > cfg.shard_count:
        raise ValueError(f"invalid shard_index={shard_index}, shard_count={cfg.shard_count}")

    window_index = int(selected_window_indices[shard_index - 1])
    split = prepared.splits[window_index - 1]
    mech_true_t = torch.as_tensor(prepared.mech_true, dtype=dtype, device=cfg.device)
    tensors = _window_to_torch(split, dtype=dtype, device=cfg.device)
    observable_grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=tensors["ode_train"],
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        cfg=cfg,
    )

    train_states = torch.as_tensor(split.ode_train.T, dtype=dtype, device=cfg.device)
    train_fts = torch.as_tensor(split.fts_train_true, dtype=dtype, device=cfg.device)
    model = KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=cfg.seed,
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
        dist=float(prepared.known_pars[6]),
        a0=float(prepared.known_pars[9]),
        wpred_enabled=train_wpred_enabled,
        wpred_eps=cfg.wpred_eps,
        gnn_learnable=cfg.gnn_learnable,
        soft_mask_enabled=cfg.soft_mask_enabled,
        soft_mask_trainable=cfg.soft_mask_trainable,
        soft_mask_s0_a0=cfg.soft_mask_s0_a0,
        soft_mask_s0_min_a0=cfg.soft_mask_s0_min_a0,
        soft_mask_s0_max_a0=cfg.soft_mask_s0_max_a0,
        soft_mask_alpha_a0=cfg.soft_mask_alpha_a0,
        soft_mask_alpha_min_a0=cfg.soft_mask_alpha_min_a0,
        soft_mask_alpha_max_a0=cfg.soft_mask_alpha_max_a0,
        x3dot_input_enabled=cfg.x3dot_input_enabled,
        x3dot_init_trainable=cfg.x3dot_init_trainable,
        x3dot_init_value=(
            x3dot_init_from_split(
                split,
                policy=cfg.x3dot_init_policy,
                fallback=cfg.x3dot_init_value,
            )
            if cfg.x3dot_input_enabled
            else cfg.x3dot_init_value
        ),
        x3dot_scale=x3dot_scale_from_split(
            split,
            configured_scale=cfg.x3dot_scale,
            scale_mode=cfg.x3dot_scale_mode,
            a0=float(prepared.known_pars[9]),
            window_span=float(split.t_stop - split.t_start),
        ),
        x3dot_lag_detach=cfg.x3dot_lag_detach,
        device=cfg.device,
        dtype=dtype,
    ).to(cfg.device)
    init_gain = model.initialize_gain_from_truth(train_states, train_fts)

    optimizer, optimizer_name = _make_optimizer(cfg, model)
    lr = float(cfg.lr)
    _set_optimizer_lr(optimizer, lr)
    step_controller = str(getattr(cfg, "step_controller", "armijo_backtracking")).strip().lower().replace("-", "_")
    if step_controller not in ("armijo_backtracking", "legacy_guard", "off"):
        step_controller = "off"
    lr_adapt_active = bool(cfg.lr_adapt) and step_controller != "armijo_backtracking"

    role_short = _role_short(split.role)
    file_tag = _window_file_tag(split.role)
    history_path = cfg.result_dir / f"kan_full_test_history_{file_tag}.json"
    event_log_dir = cfg.log_dir / "optimizer_events"
    event_log_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = event_log_dir / f"kan_full_test_optimizer_events_{file_tag}.jsonl"
    checkpoint_path = cfg.checkpoint_dir / f"kan_full_test_best_{file_tag}.pt"
    periodic_checkpoint_path = cfg.checkpoint_dir / f"kan_full_test_checkpoint_{file_tag}.pt"
    handoff_path = _handoff_path(cfg, file_tag)
    handoff_summary_path = _handoff_summary_path(cfg, file_tag)
    result_path = cfg.result_dir / f"kan_full_test_result_{file_tag}.pt"
    viz_result_path = cfg.result_dir / f"kan_full_test_result_{file_tag}.viz.pkl"
    log_path = cfg.shard_log_dir / f"log2_04_step2a_kan_full_test_local_{file_tag}.txt" if log_path is None else Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    best_payload: dict[str, Any] | None = None
    sanity_payload: dict[str, Any] | None = None
    grad_ema = float(cfg.lr_target_init)
    grad_target = float(cfg.lr_target_init)
    recent_losses: list[float] = []
    failure_reason = ""
    failure_epoch = 0
    stop_reason = ""
    stop_epoch = 0
    stop_kind = ""
    train_loss_last = float("inf")
    good_enough_reached = False
    warmstart_payload: dict[str, Any] | None = None
    hidden_normalizer_meta: dict[str, Any] | None = None

    with log_path.open("w", encoding="utf-8") as log, event_log_path.open("w", encoding="utf-8") as event_log:
        _log_line(
            log,
            "KAN full test windowed horizon: "
            f"full_points={prepared.full_points} | window=[{split.start_idx}, {split.stop_idx}] "
            f"len={split.stop_idx - split.start_idx + 1} | tspan=[{split.t_start:.6e}, {split.t_stop:.6e}]",
        )
        _log_line(
            log,
            "KAN full test window meta: "
            f"label={split.label} | role={split.role} | "
            f"t_us=[{split.t_start * 1.0e6:.9f}, {split.t_stop * 1.0e6:.9f}]",
        )
        _log_line(log, "=== AFM04 KAN FULL TEST ===")
        _log_line(
            log,
            "Switches: "
            f"adaptive_grid={str(cfg.adaptive_grid_enabled).upper()} | "
            f"warmstart_from_prerun={'ON' if cfg.warmstart_from_prerun else 'OFF'} | "
            f"warmstart_from_rs={'ON' if cfg.warmstart_from_random_search else 'OFF'} | "
            f"warmstart_fallback_random={'ON' if cfg.warmstart_fallback_random else 'OFF'} | "
            f"handoff_save={'ON' if cfg.handoff_enabled else 'OFF'} | "
            f"handoff_replay={'ON' if cfg.handoff_replay else 'OFF'} | "
            "random_init=ON | "
            f"symbolic={str(cfg.symbolic_enabled).upper()} | auto_save={str(cfg.auto_save).upper()}",
        )
        _log_line(log, f"Solver: torchdiffeq odeint | method={cfg.ode_method} | differentiable=ON")
        _log_line(
            log,
            f"KAN shard assignment: {shard_index}/{cfg.shard_count} | local_windows=1 | "
            f"window_index={window_index} | role={role_short}",
        )
        _log_line(
            log,
            "KAN config: "
            f"width={list(cfg.width)} grid={cfg.grid} k={cfg.spline_k} base={cfg.base_fun} "
            f"noise_scale={cfg.noise_scale}",
        )
        _log_line(
            log,
            "contact gate: "
            f"w_pred={'ON' if train_wpred_enabled else 'OFF'} "
            f"eps={cfg.wpred_eps:.2e} "
            "target=(soft_mask * g_nn * w_pred * nn_raw)",
        )
        _log_line(
            log,
            "handoff: "
            f"save={'ON' if cfg.handoff_enabled else 'OFF'} "
            f"replay={'ON' if cfg.handoff_replay else 'OFF'} "
            f"path={handoff_path}",
        )
        if cfg.handoff_replay:
            handoff_payload = _load_handoff_checkpoint(handoff_path, cfg=cfg, file_tag=file_tag, split=split)
            model.load_state_dict(copy.deepcopy(handoff_payload["state_dict"]))
            history = copy.deepcopy(handoff_payload.get("history", []))
            best_payload = copy.deepcopy(handoff_payload.get("best_payload"))
            best_epoch = int(handoff_payload.get("best_epoch", -1))
            best_val = float(handoff_payload.get("best_val", float("inf")))
            lr = float(handoff_payload.get("lr", lr))
            grad_ema = float(handoff_payload.get("grad_ema", grad_ema))
            grad_target = float(handoff_payload.get("grad_target", grad_target))
            recent_losses = list(handoff_payload.get("recent_losses", []))
            train_loss_last = float(handoff_payload.get("train_loss_last", train_loss_last))
            restored_rng = _restore_rng_state(handoff_payload.get("rng_state"))
            init_gain = float(model.gain().detach().cpu().item())
            _log_line(
                log,
                "handoff replay loaded: "
                f"epoch={len(history)} best_epoch={best_epoch} best_val={best_val:.6e} "
                f"rng_restored={','.join(restored_rng) if restored_rng else 'none'} "
                "adam_loop=SKIP",
            )
        else:
            warmstart_payload = _maybe_load_kft_warmstart(
                cfg=cfg,
                model=model,
                warmstart_window_index=window_index,
                log=log,
            )
        if (not cfg.handoff_replay) and warmstart_payload is not None:
            init_gain = float(model.gain().detach().cpu().item())
        if not cfg.handoff_replay:
            hidden_normalizer_meta = {
                "enabled": False,
                "source": "raw_input_no_state_mean_scale_refit",
                "state_mean": _model_normalizer_arrays(model)[0],
                "state_scale": _model_normalizer_arrays(model)[1],
            }
        else:
            hidden_normalizer_meta = copy.deepcopy(handoff_payload.get("hidden_normalizer")) if isinstance(handoff_payload, dict) else None
            if not isinstance(hidden_normalizer_meta, dict):
                hidden_normalizer_meta = {
                    "enabled": bool(getattr(model, "x3dot_input_enabled", False)),
                    "source": "handoff_replay_state_dict",
                    "new_state_mean": _model_normalizer_arrays(model)[0],
                    "new_state_scale": _model_normalizer_arrays(model)[1],
                }
        soft_mask_meta = model.soft_mask_summary()
        x3dot_meta = model.x3dot_summary() if hasattr(model, "x3dot_summary") else {
            "enabled": False,
            "q_init": float("nan"),
            "q_mean": float("nan"),
            "q_scale": float("nan"),
            "q_init_trainable": False,
            "lag_detach": False,
        }
        _log_line(
            log,
            "soft mask: "
            f"enabled={'ON' if soft_mask_meta['enabled'] else 'OFF'} "
            f"trainable={'ON' if soft_mask_meta['trainable'] else 'OFF'} "
            "scope=hard_noncontact_only "
            f"s0={soft_mask_meta['s0']:.6e} ({soft_mask_meta['s0_a0']:.3f} a0) "
            f"alpha={soft_mask_meta['alpha']:.6e} (alpha*a0={soft_mask_meta['alpha_a0']:.3f}) "
            "m_min=0",
        )
        _log_line(
            log,
            "Plan Z x3dot input: "
            f"enabled={'ON' if x3dot_meta['enabled'] else 'OFF'} "
            f"mode={cfg.x3dot_input_mode} "
            f"q_init_policy={cfg.x3dot_init_policy} "
            f"q_init={float(x3dot_meta['q_init']):.6e} "
            f"q_init_trainable={'ON' if x3dot_meta['q_init_trainable'] else 'OFF'} "
            f"q_mean={float(x3dot_meta['q_mean']):.6e} "
            f"q_scale={float(x3dot_meta['q_scale']):.6e} "
            f"lag_detach={'ON' if x3dot_meta['lag_detach'] else 'OFF'}",
        )
        _log_line(
            log,
            f"optimizer={optimizer_name.upper()} | amsgrad={'ON' if optimizer_name == 'amsgrad' else 'OFF'} | "
            f"lr={lr:.2e} weight_decay={cfg.weight_decay:.2e}",
        )
        _log_line(
            log,
            "training horizon: "
            f"total_epochs={cfg.epochs} "
            f"adam_epochs={cfg.adam_epochs} "
            f"lbfgs_epochs={cfg.lbfgs_steps}",
        )
        _log_line(
            log,
            "optimizer controls: "
            f"step_controller={step_controller} "
            f"lr_adapt={'ON' if lr_adapt_active else 'OFF'} "
            f"legacy_step_guard_switch={'ON' if (step_controller == 'legacy_guard' and cfg.step_guard_enabled) else 'OFF'} "
            f"plateau_early_stop={'ON' if cfg.plateau_early_stop else 'OFF'} "
            f"recent_val_early_stop={'ON' if cfg.recent_val_early_stop else 'OFF'} "
            f"good_enough={cfg.good_enough_loss:.1e}",
        )
        if step_controller == "off":
            _log_line(
                log,
                "optimizer bounds: "
                f"lr=[{cfg.lr_min:.2e}, {cfg.lr_max:.2e}] "
                f"epoch_retry={cfg.epoch_retry_max} | step_retry=OFF",
            )
        else:
            _log_line(
                log,
                "optimizer bounds: "
                f"lr=[{cfg.lr_min:.2e}, {cfg.lr_max:.2e}] "
                f"retry(epoch={cfg.epoch_retry_max}, step={cfg.step_retry_max}) "
                f"step_jump_frac={cfg.step_max_loss_increase_frac:.3f}",
            )
        if step_controller == "armijo_backtracking":
            _log_line(
                log,
                "armijo/backtracking config: "
                f"c1={cfg.armijo_c1:.2e} shrink={cfg.backtrack_shrink:.3f} "
                f"max={cfg.backtrack_max} min_alpha={cfg.backtrack_min_alpha:.1e} "
                f"event_log={event_log_path}",
            )
        else:
            _log_line(
                log,
                "Adam step controller: native optimizer.step() only; "
                "epoch retry is reserved for nonfinite loss/grad or runtime exceptions.",
            )
        _log_line(
            log,
            "LBFGS config: "
            f"enabled={'ON' if cfg.lbfgs_enabled else 'OFF'} "
            f"epochs={cfg.lbfgs_steps} lr={cfg.lbfgs_lr:.3e} "
            f"max_iter={cfg.lbfgs_max_iter} max_eval={cfg.lbfgs_max_eval} "
            f"history_size={cfg.lbfgs_history_size} "
            f"tol_grad={cfg.lbfgs_tolerance_grad:.1e} "
            f"tol_change={cfg.lbfgs_tolerance_change:.1e} "
            f"ys_threshold={cfg.lbfgs_ys_threshold:.1e} "
            f"strong_wolfe={'ON' if cfg.lbfgs_strong_wolfe else 'OFF'} "
            "AGU_frozen_in_lbfgs=ON",
        )
        _log_line(
            log,
            "early-stop config: "
            f"recent_val_window={cfg.recent_val_window} "
            f"recent_val_compare_gap={cfg.recent_val_compare_gap} "
            f"recent_val_rel_current_frac={cfg.recent_val_rel_current_frac:.4f}",
        )
        _log_line(log, "optimizer grouping: single_group=ON | group_adapt=OFF")
        _log_line(log, f"g_nn init={init_gain:.6e} | learnable={'ON' if cfg.gnn_learnable else 'OFF'}")
        sanity_payload = _run_oracle_sanity(
            log=log,
            split=split,
            tensors=tensors,
            known_pars=prepared.known_pars,
            eta_star_true=prepared.eta_star_true,
            mech_true=mech_true_t,
            dtype=dtype,
            device=cfg.device,
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
        )

        adam_epoch_iter = range(cfg.adam_epochs)
        if cfg.handoff_replay:
            adam_epoch_iter = range(int(cfg.adam_epochs), int(cfg.adam_epochs))

        for epoch in adam_epoch_iter:
            epoch_wall_start = perf_counter()
            grid_update_sec_epoch = 0.0
            grid_update_runs_epoch = 0
            grid_update_meta_epoch = {
                "base_samples": float("nan"),
                "extra_samples": 0.0,
                "total_samples": float("nan"),
                "mean_w_pred": float("nan"),
                "max_w_pred": float("nan"),
                "q50_w_pred": float("nan"),
                "q90_w_pred": float("nan"),
                "q99_w_pred": float("nan"),
                "frac_s_le_a0": float("nan"),
                "frac_s_le_1p5a0": float("nan"),
                "frac_wpred_ge_05": float("nan"),
                "frac_wpred_ge_08": float("nan"),
            }
            epoch_start_state = copy.deepcopy(model.state_dict())
            epoch_start_opt_state = copy.deepcopy(optimizer.state_dict())
            epoch_attempt_max = max(1, int(cfg.epoch_retry_max) + 1)
            epoch_attempt = 0
            recovered = False
            train_total: torch.Tensor | None = None
            train_parts: TorchLossParts | None = None
            val_total: torch.Tensor | None = None
            val_parts: TorchLossParts | None = None
            grad_norm = float("nan")
            grad_norm_raw = float("nan")
            step_retry_count = 0
            step_accept_reason = ""
            epoch_fail_reason = ""

            while epoch_attempt < epoch_attempt_max:
                epoch_attempt += 1
                model.load_state_dict(copy.deepcopy(epoch_start_state))
                optimizer.load_state_dict(copy.deepcopy(epoch_start_opt_state))
                _set_optimizer_lr(optimizer, lr)
                model.train()

                if cfg.adaptive_grid_enabled and _grid_update_due(epoch, cfg.grid_update_num, cfg.start_grid_update_step, cfg.stop_grid_update_step):
                    try:
                        grid_update_t0 = perf_counter()
                        with torch.no_grad():
                            grid_source = "raw_current_rollout_pred_x3_qpre"
                            grid_inputs, grid_meta = _make_grid_update_inputs(
                                cfg=cfg,
                                base_inputs=observable_grid_inputs,
                                known_pars=prepared.known_pars,
                                model=model,
                                tensors=tensors,
                                mech_true=mech_true_t,
                                ode_method=cfg.ode_method,
                                ode_rtol=cfg.ode_rtol,
                                ode_atol=cfg.ode_atol,
                            )
                            grid_source = str(grid_meta.get("source", "raw_current_rollout_pred_x3_qpre"))
                            model.update_grid_from_inputs(grid_inputs)
                        grid_update_dt = perf_counter() - grid_update_t0
                        grid_update_sec_epoch += float(grid_update_dt)
                        grid_update_runs_epoch += 1
                        grid_update_meta_epoch = dict(grid_meta)
                        _log_line(
                            log,
                            "grid update | "
                            f"epoch={epoch + 1} | source={grid_source} | "
                            f"samples={int(grid_meta['total_samples'])} | "
                            f"mean_w_pred={grid_meta['mean_w_pred']:.3f} "
                            f"max_w_pred={grid_meta['max_w_pred']:.3f}",
                        )
                    except Exception as err:
                        epoch_fail_reason = f"grid_update_exception:{err}"

                if epoch_fail_reason != "":
                    if epoch_attempt < epoch_attempt_max:
                        lr = max(lr * cfg.epoch_retry_lr_factor, cfg.epoch_retry_lr_floor)
                        _log_line(
                            log,
                            f"retry epoch {epoch + 1} attempt {epoch_attempt}/{epoch_attempt_max} "
                            f"-- reason={epoch_fail_reason} | lr={lr:.3e}",
                        )
                        continue
                    failure_reason = f"epoch_retry_exhausted:{epoch_fail_reason}"
                    failure_epoch = epoch + 1
                    break

                optimizer.zero_grad()
                train_traj: torch.Tensor | None = None
                accepted_train_traj: torch.Tensor | None = None
                diag_epoch1 = epoch == 0
                try:
                    if diag_epoch1:
                        diag_eval_t0 = perf_counter()
                        _log_line(log, f"DIAG epoch=1 attempt={epoch_attempt} before evaluate_split")
                    train_total, train_parts, train_traj = evaluate_split(
                        force_module=model,
                        known_pars=prepared.known_pars,
                        mech_true=mech_true_t,
                        ode_true=tensors["ode_full"],
                        x2dot_true=tensors["x2dot_full"],
                        contact_mask=tensors["contact_full"],
                        times=tensors["times_full"],
                        ode_method=cfg.ode_method,
                        ode_rtol=cfg.ode_rtol,
                        ode_atol=cfg.ode_atol,
                        eta_star_true=prepared.eta_star_true,
                        loss_indices=tensors["train_idx"],
                    )
                    if diag_epoch1:
                        diag_eval_sec = perf_counter() - diag_eval_t0
                        diag_q_count = getattr(train_traj, "_plan_z_accepted_history_count", "NA")
                        _log_line(
                            log,
                            f"DIAG epoch=1 attempt={epoch_attempt} after evaluate_split "
                            f"sec={diag_eval_sec:.3f} "
                            f"loss={float(train_total.detach()):.6e} "
                            f"q_count={diag_q_count}",
                        )
                    if not torch.isfinite(train_total):
                        epoch_fail_reason = "train_loss_nonfinite"
                    else:
                        if diag_epoch1:
                            diag_backward_t0 = perf_counter()
                            _log_line(log, f"DIAG epoch=1 attempt={epoch_attempt} before backward")
                        train_total.backward()
                        grad_norm_raw = _grad_norm(model)
                        if diag_epoch1:
                            diag_backward_sec = perf_counter() - diag_backward_t0
                            _log_line(
                                log,
                                f"DIAG epoch=1 attempt={epoch_attempt} after backward "
                                f"sec={diag_backward_sec:.3f} grad_norm_raw={grad_norm_raw:.6e}",
                            )
                        if not np.isfinite(grad_norm_raw):
                            epoch_fail_reason = "grad_norm_nonfinite"
                except Exception as err:
                    epoch_fail_reason = f"exception:{err}"

                if epoch_fail_reason == "":
                    grad_norm = grad_norm_raw
                    if not np.isfinite(grad_norm):
                        epoch_fail_reason = "grad_scaled_norm_nonfinite"

                if epoch_fail_reason != "":
                    if epoch_attempt < epoch_attempt_max:
                        lr = max(lr * cfg.epoch_retry_lr_factor, cfg.epoch_retry_lr_floor)
                        _log_line(
                            log,
                            f"retry epoch {epoch + 1} attempt {epoch_attempt}/{epoch_attempt_max} "
                            f"-- reason={epoch_fail_reason} | lr={lr:.3e}",
                        )
                        continue
                    failure_reason = f"epoch_retry_exhausted:{epoch_fail_reason}"
                    failure_epoch = epoch + 1
                    break

                prev_loss_ref = train_loss_last if np.isfinite(train_loss_last) else float("nan")
                train_loss_before = float(train_total.detach())
                model_base_state = copy.deepcopy(model.state_dict())
                opt_base_state = copy.deepcopy(optimizer.state_dict())
                base_params = _capture_params(model)
                grad_cache = _capture_grads(model)
                step_base_lr = float(lr)
                step_candidate_loss = float("nan")
                step_alpha = 1.0
                step_effective_lr = float(lr)
                step_grad_dot_p = float("nan")
                step_armijo_rhs = float("nan")
                step_retry_count = 0
                step_accept_reason = "accepted"
                step_reject_reason = ""
                step_accepted = False
                step_update_accepted = False
                adam_state_reset_attempted = False
                adam_state_reset_accepted = False
                adam_state_reset_trigger_retries = 0
                adam_state_reset_retries = 0

                if step_controller == "armijo_backtracking":
                    c1 = float(cfg.armijo_c1)
                    if not np.isfinite(c1) or c1 <= 0.0:
                        c1 = 1.0e-4
                    shrink = float(cfg.backtrack_shrink)
                    if not np.isfinite(shrink) or not (0.0 < shrink < 1.0):
                        shrink = 0.5
                    min_alpha = float(cfg.backtrack_min_alpha)
                    if not np.isfinite(min_alpha) or min_alpha <= 0.0:
                        min_alpha = 1.0e-8
                    attempts_total = max(1, int(cfg.backtrack_max) + 1)
                    last_reason = "armijo_no_trial"
                    trials_run = 0

                    for step_attempt in range(1, attempts_total + 1):
                        alpha = float(shrink ** (step_attempt - 1))
                        if alpha < min_alpha:
                            last_reason = "armijo_min_alpha_reached"
                            break
                        trials_run = step_attempt
                        trial_lr = float(step_base_lr * alpha)
                        model.load_state_dict(copy.deepcopy(model_base_state))
                        optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                        _set_optimizer_lr(optimizer, trial_lr)
                        _restore_grads(model, grad_cache)
                        optimizer.step()
                        model.train()
                        grad_dot_delta = _grad_dot_delta(model, base_params, grad_cache)
                        armijo_rhs = float(train_loss_before + c1 * grad_dot_delta)
                        try:
                            trial_total, trial_parts, trial_traj = evaluate_split(
                                force_module=model,
                                known_pars=prepared.known_pars,
                                mech_true=mech_true_t,
                                ode_true=tensors["ode_full"],
                                x2dot_true=tensors["x2dot_full"],
                                contact_mask=tensors["contact_full"],
                                times=tensors["times_full"],
                                ode_method=cfg.ode_method,
                                ode_rtol=cfg.ode_rtol,
                                ode_atol=cfg.ode_atol,
                                eta_star_true=prepared.eta_star_true,
                                loss_indices=tensors["train_idx"],
                            )
                            trial_loss = float(trial_total.detach())
                        except Exception as err:
                            step_accept_reason = f"step_trial_exception:{err}"
                            trial_total = None
                            trial_parts = None
                            trial_traj = None
                            trial_loss = float("inf")

                        accepted_trial = False
                        reject_reason = ""
                        if step_accept_reason != "accepted":
                            reject_reason = step_accept_reason
                        elif not np.isfinite(trial_loss):
                            reject_reason = "step_trial_loss_nonfinite"
                        elif not np.isfinite(grad_dot_delta):
                            reject_reason = "armijo_grad_dot_delta_nonfinite"
                        elif grad_dot_delta >= 0.0:
                            reject_reason = "armijo_not_descent"
                        elif trial_loss > train_loss_before:
                            reject_reason = "armijo_loss_increase"
                        elif trial_loss > armijo_rhs:
                            reject_reason = "armijo_insufficient_decrease"
                        else:
                            accepted_trial = True
                            reject_reason = ""

                        _write_event(
                            event_log,
                            {
                                "event": "armijo_trial",
                                "epoch": int(epoch + 1),
                                "trial_index": int(step_attempt),
                                "alpha": float(alpha),
                                "base_lr": float(step_base_lr),
                                "effective_lr": float(trial_lr),
                                "loss_before": float(train_loss_before),
                                "candidate_loss": float(trial_loss),
                                "grad_dot_p": float(grad_dot_delta),
                                "armijo_rhs": float(armijo_rhs),
                                "accepted": bool(accepted_trial),
                                "reject_reason": reject_reason,
                                "armijo_pass": "base",
                            },
                        )

                        step_candidate_loss = float(trial_loss)
                        step_alpha = float(alpha)
                        step_effective_lr = float(trial_lr)
                        step_grad_dot_p = float(grad_dot_delta)
                        step_armijo_rhs = float(armijo_rhs)

                        if accepted_trial:
                            train_total = trial_total
                            train_parts = trial_parts
                            accepted_train_traj = trial_traj
                            step_retry_count = step_attempt - 1
                            step_accept_reason = "accepted"
                            step_reject_reason = ""
                            step_accepted = True
                            step_update_accepted = True
                            _set_optimizer_lr(optimizer, step_base_lr)
                            break

                        last_reason = reject_reason
                        step_accept_reason = "accepted"

                    resettable_armijo_reasons = {
                        "armijo_not_descent",
                        "armijo_loss_increase",
                        "armijo_insufficient_decrease",
                    }
                    if (
                        not step_accepted
                        and last_reason in resettable_armijo_reasons
                        and max(0, trials_run - 1) >= int(cfg.backtrack_max)
                    ):
                        adam_state_reset_attempted = True
                        adam_state_reset_trigger_retries = max(0, trials_run - 1)
                        _write_event(
                            event_log,
                            {
                                "event": "adam_state_reset_after_armijo_cap",
                                "epoch": int(epoch + 1),
                                "loss_before": float(train_loss_before),
                                "trigger_retries": int(adam_state_reset_trigger_retries),
                                "trigger_reason": last_reason,
                            },
                        )

                        model.load_state_dict(copy.deepcopy(model_base_state))
                        optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                        _reset_optimizer_state(optimizer)
                        _set_optimizer_lr(optimizer, step_base_lr)
                        reset_opt_base_state = copy.deepcopy(optimizer.state_dict())
                        reset_last_reason = "armijo_no_trial"
                        reset_trials_run = 0

                        for step_attempt in range(1, attempts_total + 1):
                            alpha = float(shrink ** (step_attempt - 1))
                            if alpha < min_alpha:
                                reset_last_reason = "armijo_min_alpha_reached"
                                break
                            reset_trials_run = step_attempt
                            trial_lr = float(step_base_lr * alpha)
                            model.load_state_dict(copy.deepcopy(model_base_state))
                            optimizer.load_state_dict(copy.deepcopy(reset_opt_base_state))
                            _set_optimizer_lr(optimizer, trial_lr)
                            _restore_grads(model, grad_cache)
                            optimizer.step()
                            model.train()
                            grad_dot_delta = _grad_dot_delta(model, base_params, grad_cache)
                            armijo_rhs = float(train_loss_before + c1 * grad_dot_delta)
                            try:
                                trial_total, trial_parts, trial_traj = evaluate_split(
                                    force_module=model,
                                    known_pars=prepared.known_pars,
                                    mech_true=mech_true_t,
                                    ode_true=tensors["ode_full"],
                                    x2dot_true=tensors["x2dot_full"],
                                    contact_mask=tensors["contact_full"],
                                    times=tensors["times_full"],
                                    ode_method=cfg.ode_method,
                                    ode_rtol=cfg.ode_rtol,
                                    ode_atol=cfg.ode_atol,
                                    eta_star_true=prepared.eta_star_true,
                                    loss_indices=tensors["train_idx"],
                                )
                                trial_loss = float(trial_total.detach())
                            except Exception as err:
                                step_accept_reason = f"step_trial_exception:{err}"
                                trial_total = None
                                trial_parts = None
                                trial_traj = None
                                trial_loss = float("inf")

                            accepted_trial = False
                            reject_reason = ""
                            if step_accept_reason != "accepted":
                                reject_reason = step_accept_reason
                            elif not np.isfinite(trial_loss):
                                reject_reason = "step_trial_loss_nonfinite"
                            elif not np.isfinite(grad_dot_delta):
                                reject_reason = "armijo_grad_dot_delta_nonfinite"
                            elif grad_dot_delta >= 0.0:
                                reject_reason = "armijo_not_descent"
                            elif trial_loss > train_loss_before:
                                reject_reason = "armijo_loss_increase"
                            elif trial_loss > armijo_rhs:
                                reject_reason = "armijo_insufficient_decrease"
                            else:
                                accepted_trial = True
                                reject_reason = ""

                            _write_event(
                                event_log,
                                {
                                    "event": "armijo_trial",
                                    "epoch": int(epoch + 1),
                                    "trial_index": int(step_attempt),
                                    "alpha": float(alpha),
                                    "base_lr": float(step_base_lr),
                                    "effective_lr": float(trial_lr),
                                    "loss_before": float(train_loss_before),
                                    "candidate_loss": float(trial_loss),
                                    "grad_dot_p": float(grad_dot_delta),
                                    "armijo_rhs": float(armijo_rhs),
                                    "accepted": bool(accepted_trial),
                                    "reject_reason": reject_reason,
                                    "armijo_pass": "adam_state_reset",
                                },
                            )

                            step_candidate_loss = float(trial_loss)
                            step_alpha = float(alpha)
                            step_effective_lr = float(trial_lr)
                            step_grad_dot_p = float(grad_dot_delta)
                            step_armijo_rhs = float(armijo_rhs)

                            if accepted_trial:
                                train_total = trial_total
                                train_parts = trial_parts
                                accepted_train_traj = trial_traj
                                step_retry_count = step_attempt - 1
                                adam_state_reset_retries = step_retry_count
                                step_accept_reason = "accepted"
                                step_reject_reason = ""
                                step_accepted = True
                                step_update_accepted = True
                                adam_state_reset_accepted = True
                                _set_optimizer_lr(optimizer, step_base_lr)
                                break

                            reset_last_reason = reject_reason
                            step_accept_reason = "accepted"

                        if not step_accepted:
                            last_reason = reset_last_reason
                            trials_run = reset_trials_run
                            adam_state_reset_retries = max(0, reset_trials_run - 1)
                            _write_event(
                                event_log,
                                {
                                    "event": "adam_state_reset_armijo_rejected",
                                    "epoch": int(epoch + 1),
                                    "loss_before": float(train_loss_before),
                                    "candidate_loss": float(step_candidate_loss),
                                    "trials_run": int(reset_trials_run),
                                    "backtracking_retries": int(adam_state_reset_retries),
                                    "reject_reason": reset_last_reason,
                                },
                            )

                    if not step_accepted:
                        step_retry_count = max(0, trials_run - 1)
                        model.load_state_dict(copy.deepcopy(model_base_state))
                        optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                        _set_optimizer_lr(optimizer, step_base_lr)
                        train_total = torch.as_tensor(train_loss_before, dtype=dtype, device=cfg.device)
                        accepted_train_traj = train_traj
                        step_accept_reason = "armijo_rejected_keep_previous"
                        step_reject_reason = last_reason
                        step_accepted = True
                        step_update_accepted = False
                        _write_event(
                            event_log,
                            {
                                "event": "armijo_reject_epoch",
                                "epoch": int(epoch + 1),
                                "loss_before": float(train_loss_before),
                                "candidate_loss": float(step_candidate_loss),
                                "trials_run": int(trials_run),
                                "backtracking_retries": int(step_retry_count),
                                "reject_reason": step_reject_reason,
                            },
                        )

                elif step_controller == "legacy_guard":
                    attempts_total = max(1, int(cfg.step_retry_max) + 1)
                    for step_attempt in range(1, attempts_total + 1):
                        model.load_state_dict(copy.deepcopy(model_base_state))
                        optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                        _set_optimizer_lr(optimizer, lr)
                        _restore_grads(model, grad_cache)
                        optimizer.step()
                        model.train()
                        try:
                            trial_total, trial_parts, trial_traj = evaluate_split(
                                force_module=model,
                                known_pars=prepared.known_pars,
                                mech_true=mech_true_t,
                                ode_true=tensors["ode_full"],
                                x2dot_true=tensors["x2dot_full"],
                                contact_mask=tensors["contact_full"],
                                times=tensors["times_full"],
                                ode_method=cfg.ode_method,
                                ode_rtol=cfg.ode_rtol,
                                ode_atol=cfg.ode_atol,
                                eta_star_true=prepared.eta_star_true,
                                loss_indices=tensors["train_idx"],
                            )
                            trial_loss = float(trial_total.detach())
                        except Exception as err:
                            step_accept_reason = f"step_trial_exception:{err}"
                            trial_total = None
                            trial_parts = None
                            trial_traj = None
                            trial_loss = float("inf")

                        step_candidate_loss = float(trial_loss)
                        step_effective_lr = float(lr)
                        if step_accept_reason == "accepted":
                            if not np.isfinite(trial_loss):
                                step_accept_reason = "step_trial_loss_nonfinite"
                            elif (
                                np.isfinite(cfg.step_max_loss_increase_frac)
                                and trial_loss > train_loss_before * (1.0 + cfg.step_max_loss_increase_frac)
                            ):
                                step_accept_reason = "step_trial_loss_jump"
                            elif (
                                np.isfinite(prev_loss_ref)
                                and trial_loss > prev_loss_ref * (1.0 + cfg.step_max_loss_increase_frac)
                            ):
                                step_accept_reason = "step_prev_epoch_loss_jump"

                        if step_accept_reason == "accepted":
                            train_total = trial_total
                            train_parts = trial_parts
                            accepted_train_traj = trial_traj
                            step_accepted = True
                            step_update_accepted = True
                            step_retry_count = step_attempt - 1
                            break

                        if step_attempt >= attempts_total:
                            step_retry_count = max(0, attempts_total - 1)
                            if _terminal_step_failure(step_accept_reason):
                                model.load_state_dict(copy.deepcopy(model_base_state))
                                optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                                _set_optimizer_lr(optimizer, lr)
                                failure_reason = f"trial_failed:{step_accept_reason}"
                                failure_epoch = epoch + 1
                                break
                            model.load_state_dict(copy.deepcopy(model_base_state))
                            optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                            _set_optimizer_lr(optimizer, lr)
                            train_total = torch.as_tensor(train_loss_before, dtype=dtype, device=cfg.device)
                            accepted_train_traj = train_traj
                            step_reject_reason = step_accept_reason
                            step_accept_reason = "step_rejected_keep_previous"
                            step_accepted = True
                            step_update_accepted = False
                            break

                        lr = max(lr * cfg.step_retry_lr_factor, cfg.epoch_retry_lr_floor)
                        step_retry_count = step_attempt
                        _log_line(
                            log,
                            f"step-guard retry {step_attempt}/{cfg.step_retry_max} -- reason={step_accept_reason} | lr={lr:.3e}",
                        )
                        step_accept_reason = "accepted"

                else:
                    model.load_state_dict(copy.deepcopy(model_base_state))
                    optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                    _set_optimizer_lr(optimizer, lr)
                    _restore_grads(model, grad_cache)
                    if diag_epoch1:
                        diag_step_t0 = perf_counter()
                        _log_line(log, f"DIAG epoch=1 attempt={epoch_attempt} before optimizer.step")
                        _log_diag_group_stats(
                            log,
                            prefix=f"DIAG epoch=1 attempt={epoch_attempt} before_step_grad_group_stats",
                            stats=_grad_group_stats(model),
                        )
                        _log_diag_group_stats(
                            log,
                            prefix=f"DIAG epoch=1 attempt={epoch_attempt} before_step_param_delta_group_stats",
                            stats=_param_delta_group_stats(model, model_base_state),
                        )
                    optimizer.step()
                    if diag_epoch1:
                        diag_step_sec = perf_counter() - diag_step_t0
                        _log_line(log, f"DIAG epoch=1 attempt={epoch_attempt} after optimizer.step sec={diag_step_sec:.6f}")
                        _log_diag_group_stats(
                            log,
                            prefix=f"DIAG epoch=1 attempt={epoch_attempt} after_step_param_delta_group_stats",
                            stats=_param_delta_group_stats(model, model_base_state),
                        )
                    model.train()
                    try:
                        if diag_epoch1:
                            diag_trial_eval_t0 = perf_counter()
                            _log_line(log, f"DIAG epoch=1 attempt={epoch_attempt} before post_step evaluate_split")
                        trial_total, trial_parts, trial_traj = evaluate_split(
                            force_module=model,
                            known_pars=prepared.known_pars,
                            mech_true=mech_true_t,
                            ode_true=tensors["ode_full"],
                            x2dot_true=tensors["x2dot_full"],
                            contact_mask=tensors["contact_full"],
                            times=tensors["times_full"],
                            ode_method=cfg.ode_method,
                            ode_rtol=cfg.ode_rtol,
                            ode_atol=cfg.ode_atol,
                            eta_star_true=prepared.eta_star_true,
                            loss_indices=tensors["train_idx"],
                        )
                        trial_loss = float(trial_total.detach())
                        if diag_epoch1:
                            diag_trial_eval_sec = perf_counter() - diag_trial_eval_t0
                            diag_trial_q_count = getattr(trial_traj, "_plan_z_accepted_history_count", "NA")
                            _log_line(
                                log,
                                f"DIAG epoch=1 attempt={epoch_attempt} after post_step evaluate_split "
                                f"sec={diag_trial_eval_sec:.3f} loss={trial_loss:.6e} "
                                f"q_count={diag_trial_q_count}",
                            )
                    except Exception as err:
                        step_accept_reason = f"step_trial_exception:{err}"
                        trial_total = None
                        trial_parts = None
                        trial_traj = None
                        trial_loss = float("inf")

                    step_candidate_loss = float(trial_loss)
                    step_effective_lr = float(lr)
                    if step_accept_reason == "accepted" and np.isfinite(trial_loss):
                        train_total = trial_total
                        train_parts = trial_parts
                        accepted_train_traj = trial_traj
                        step_accepted = True
                        step_update_accepted = True
                    else:
                        if step_accept_reason == "accepted":
                            step_accept_reason = "step_trial_loss_nonfinite"
                        model.load_state_dict(copy.deepcopy(model_base_state))
                        optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                        _set_optimizer_lr(optimizer, lr)
                        epoch_fail_reason = step_accept_reason

                if epoch_fail_reason != "":
                    if epoch_attempt < epoch_attempt_max:
                        lr = max(lr * cfg.epoch_retry_lr_factor, cfg.epoch_retry_lr_floor)
                        _log_line(
                            log,
                            f"retry epoch {epoch + 1} attempt {epoch_attempt}/{epoch_attempt_max} "
                            f"-- reason={epoch_fail_reason} | lr={lr:.3e}",
                        )
                        continue
                    failure_reason = f"epoch_retry_exhausted:{epoch_fail_reason}"
                    failure_epoch = epoch + 1
                    break

                if failure_reason != "":
                    break
                if step_accepted:
                    recovered = epoch_attempt > 1
                    break

            if failure_reason != "":
                break

            model.eval()
            try:
                with torch.no_grad():
                    val_total, val_parts, _ = evaluate_split(
                        force_module=model,
                        known_pars=prepared.known_pars,
                        mech_true=mech_true_t,
                        ode_true=tensors["ode_full"],
                        x2dot_true=tensors["x2dot_full"],
                        contact_mask=tensors["contact_full"],
                        times=tensors["times_full"],
                        ode_method=cfg.ode_method,
                        ode_rtol=cfg.ode_rtol,
                        ode_atol=cfg.ode_atol,
                        eta_star_true=prepared.eta_star_true,
                        loss_indices=tensors["val_idx"],
                    )
                    if not torch.isfinite(val_total):
                        failure_reason = "val_loss_nonfinite"
                        failure_epoch = epoch + 1
                        break
            except Exception as err:
                failure_reason = f"val_exception:{err}"
                failure_epoch = epoch + 1
                break

            retry_cap = max(0, int(cfg.backtrack_max)) if step_controller == "armijo_backtracking" else max(0, int(cfg.step_retry_max))
            previous_retry_count = float("nan")
            for previous_row in reversed(history):
                if isinstance(previous_row, dict) and "step_retry_count" in previous_row:
                    try:
                        previous_retry_count = float(previous_row["step_retry_count"])
                    except Exception:
                        previous_retry_count = float("nan")
                    break
            retry_hit_cap = step_controller == "armijo_backtracking" and retry_cap > 0 and int(step_retry_count) >= retry_cap
            retry_spike = bool(
                retry_hit_cap
                and np.isfinite(previous_retry_count)
                and float(previous_retry_count) < float(retry_cap)
            )
            optimizer_warning = ""
            if retry_hit_cap:
                optimizer_warning = "armijo_retry_hit_configured_cap"
                if retry_spike:
                    optimizer_warning = (
                        f"armijo_retry_spike_to_cap:{int(previous_retry_count)}->{int(step_retry_count)}"
                    )

            soft_mask_meta = model.soft_mask_summary()
            x3dot_meta = model.x3dot_summary() if hasattr(model, "x3dot_summary") else {
                "enabled": False,
                "q_init": float("nan"),
                "q_mean": float("nan"),
                "q_scale": float("nan"),
                "q_init_trainable": False,
                "lag_detach": False,
            }
            row = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_total.detach()),
                "val_loss": float(val_total.detach()),
                "epoch_sec": float(perf_counter() - epoch_wall_start),
                "grid_update_sec": float(grid_update_sec_epoch),
                "grid_update_runs": float(grid_update_runs_epoch),
                "train_state": float(train_parts.state),
                "val_state": float(val_parts.state),
                "train_x1_state": float(train_parts.x1_state),
                "val_x1_state": float(val_parts.x1_state),
                "train_x2_state": float(train_parts.x2_state),
                "val_x2_state": float(val_parts.x2_state),
                "train_x2dot": float(train_parts.x2dot),
                "val_x2dot": float(val_parts.x2dot),
                "train_x3_range": float(train_parts.x3_range),
                "val_x3_range": float(val_parts.x3_range),
                "train_fts_range": float(train_parts.fts_range),
                "val_fts_range": float(val_parts.fts_range),
                "train_x1_rec": float(train_parts.x1_rec),
                "val_x1_rec": float(val_parts.x1_rec),
                "train_x3_rec": float(train_parts.x3_rec),
                "val_x3_rec": float(val_parts.x3_rec),
                "train_fts_rollout_rec": float(train_parts.fts_rollout_rec),
                "val_fts_rollout_rec": float(val_parts.fts_rollout_rec),
                "g_nn": float(model.gain().detach().cpu().item()),
                "soft_mask_enabled": bool(soft_mask_meta["enabled"]),
                "soft_mask_trainable": bool(soft_mask_meta["trainable"]),
                "soft_mask_s0": float(soft_mask_meta["s0"]),
                "soft_mask_s0_a0": float(soft_mask_meta["s0_a0"]),
                "soft_mask_alpha": float(soft_mask_meta["alpha"]),
                "soft_mask_alpha_a0": float(soft_mask_meta["alpha_a0"]),
                "soft_mask_m_min": 0.0,
                "x3dot_input_enabled": bool(x3dot_meta["enabled"]),
                "x3dot_q_init": float(x3dot_meta["q_init"]),
                "x3dot_q_init_trainable": bool(x3dot_meta["q_init_trainable"]),
                "x3dot_q_mean": float(x3dot_meta["q_mean"]),
                "x3dot_q_scale": float(x3dot_meta["q_scale"]),
                "x3dot_lag_detach": bool(x3dot_meta["lag_detach"]),
                "grad_norm": float(grad_norm),
                "grad_norm_raw": float(grad_norm_raw),
                "lr": float(lr),
                "phase": "adam_backtracking" if step_controller == "armijo_backtracking" else (
                    "adam_legacy_guard" if step_controller == "legacy_guard" else "adam_no_step_controller"
                ),
                "optimizer": optimizer_name,
                "step_controller": step_controller,
                "loss_before": float(train_loss_before),
                "candidate_loss": float(step_candidate_loss),
                "accepted_loss": float(train_total.detach()),
                "base_lr": float(step_base_lr),
                "alpha": float(step_alpha),
                "effective_lr": float(step_effective_lr),
                "grad_dot_p": float(step_grad_dot_p),
                "armijo_rhs": float(step_armijo_rhs),
                "backtracking_retries": float(step_retry_count if step_controller == "armijo_backtracking" else 0),
                "step_update_accepted": bool(step_update_accepted),
                "accepted": bool(step_update_accepted),
                "reject_reason": step_reject_reason,
                "lbfgs_outer_step": 0.0,
                "lbfgs_closure_calls": 0.0,
                "loss_before_lbfgs_step": float("nan"),
                "loss_after_lbfgs_step": float("nan"),
                "lbfgs_returned_loss": float("nan"),
                "lbfgs_wall_time": float("nan"),
                "lbfgs_line_search": "",
                "epoch_retry_count": float(max(0, epoch_attempt - 1)),
                "step_retry_count": float(step_retry_count),
                "optimizer_warning": optimizer_warning,
                "retry_hit_cap": bool(retry_hit_cap),
                "retry_spike": bool(retry_spike),
                "previous_step_retry_count": float(previous_retry_count),
                "adam_state_reset_attempted": bool(adam_state_reset_attempted),
                "adam_state_reset_accepted": bool(adam_state_reset_accepted),
                "adam_state_reset_trigger_retries": float(adam_state_reset_trigger_retries),
                "adam_state_reset_retries": float(adam_state_reset_retries),
                "grid_base_samples": float(grid_update_meta_epoch["base_samples"]),
                "grid_extra_samples": float(grid_update_meta_epoch["extra_samples"]),
                "grid_total_samples": float(grid_update_meta_epoch["total_samples"]),
                "wpred_mean": float(grid_update_meta_epoch["mean_w_pred"]),
                "wpred_max": float(grid_update_meta_epoch["max_w_pred"]),
                "wpred_q50": float(grid_update_meta_epoch["q50_w_pred"]),
                "wpred_q90": float(grid_update_meta_epoch["q90_w_pred"]),
                "wpred_q99": float(grid_update_meta_epoch["q99_w_pred"]),
                "wpred_frac_s_le_a0": float(grid_update_meta_epoch["frac_s_le_a0"]),
                "wpred_frac_s_le_1p5a0": float(grid_update_meta_epoch["frac_s_le_1p5a0"]),
                "wpred_frac_ge_05": float(grid_update_meta_epoch["frac_wpred_ge_05"]),
                "wpred_frac_ge_08": float(grid_update_meta_epoch["frac_wpred_ge_08"]),
            }
            history.append(row)
            train_loss_last = float(train_total.detach())

            if float(val_total.detach()) < best_val:
                best_val = float(val_total.detach())
                best_epoch = epoch + 1
                best_snapshot = _snapshot_split(
                    force_module=model,
                    known_pars=prepared.known_pars,
                    eta_star_true=prepared.eta_star_true,
                    mech_true=mech_true_t,
                    split=split,
                    tensors=tensors,
                    dtype=dtype,
                    device=cfg.device,
                    ode_method=cfg.ode_method,
                    ode_rtol=cfg.ode_rtol,
                    ode_atol=cfg.ode_atol,
                )
                best_payload = {
                    "epoch": best_epoch,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "config": asdict(cfg),
                    "state_mean": _model_normalizer_arrays(model)[0],
                    "state_scale": _model_normalizer_arrays(model)[1],
                    "hidden_normalizer": copy.deepcopy(hidden_normalizer_meta),
                    "known_pars": prepared.known_pars,
                    "eta_star_true": prepared.eta_star_true,
                    "mech_true": prepared.mech_true,
                    "sanity": sanity_payload,
                    "window_meta": {
                        "role": split.role,
                        "label": split.label,
                        "title": _window_title(split.role),
                        "start_idx": split.start_idx,
                        "stop_idx": split.stop_idx,
                        "t_start": split.t_start,
                        "t_stop": split.t_stop,
                    },
                    "history": history,
                    "best_snapshot": best_snapshot,
                }
                _save_torch(checkpoint_path, best_payload)

            if ((epoch + 1) % cfg.log_every) == 0 or epoch == 0 or epoch + 1 == cfg.adam_epochs:
                _log_line(log, f"KAN epoch {epoch + 1} train={float(train_total.detach()):.6e}")
                _log_line(
                    log,
                    "  optimizer summary: "
                    f"phase={row['phase']} controller={step_controller} "
                    f"loss={train_loss_before:.6e}->{float(train_total.detach()):.6e} "
                    f"accepted={'YES' if step_update_accepted else 'NO'} "
                    f"retries={step_retry_count} alpha={step_alpha:.3e} "
                    f"effective_lr={step_effective_lr:.3e} "
                    f"reason={step_reject_reason or step_accept_reason}",
                )
                if optimizer_warning:
                    _log_line(
                        log,
                        "  optimizer warning: "
                        f"{optimizer_warning} "
                        f"| previous_retries={previous_retry_count} "
                        f"current_retries={step_retry_count} "
                        f"cap={retry_cap}",
                    )
                if adam_state_reset_attempted:
                    _log_line(
                        log,
                        "  adam state reset: "
                        f"trigger_retries={adam_state_reset_trigger_retries} "
                        f"accepted={'YES' if adam_state_reset_accepted else 'NO'} "
                        f"post_reset_retries={adam_state_reset_retries}",
                    )
                _log_line(log, f"  grad_norm={grad_norm:.3e} lr={lr:.6g}")
                _log_line(log, f"  grad_norm_raw={grad_norm_raw:.3e} grad_norm_scaled={grad_norm:.3e}")
                _log_line(
                    log,
                    "  soft_mask: "
                    f"enabled={'ON' if row['soft_mask_enabled'] else 'OFF'} "
                    f"trainable={'ON' if row['soft_mask_trainable'] else 'OFF'} "
                    f"s0={row['soft_mask_s0']:.6e} "
                    f"s0/a0={row['soft_mask_s0_a0']:.6g} "
                    f"alpha={row['soft_mask_alpha']:.6e} "
                    f"alpha*a0={row['soft_mask_alpha_a0']:.6g} "
                    f"m_min={row['soft_mask_m_min']:.1f}",
                )
                _log_line(
                    log,
                    "  PlanZ q: "
                    f"enabled={'ON' if row['x3dot_input_enabled'] else 'OFF'} "
                    f"q_init={row['x3dot_q_init']:.6e} "
                    f"trainable={'ON' if row['x3dot_q_init_trainable'] else 'OFF'} "
                    f"q_mean={row['x3dot_q_mean']:.6e} "
                    f"q_scale={row['x3dot_q_scale']:.6e} "
                    f"detach={'ON' if row['x3dot_lag_detach'] else 'OFF'}",
                )
                if recovered:
                    _log_line(log, f"  retry: recovered_after={epoch_attempt - 1} rollback(s)")
                if step_retry_count > 0 and step_accept_reason == "accepted":
                    label = "armijo/backtracking" if step_controller == "armijo_backtracking" else "step-guard"
                    _log_line(log, f"  {label}: accepted_after={step_retry_count} retry/reduction(s)")
                epoch_sec = float(row["epoch_sec"])
                grid_sec = float(row["grid_update_sec"])
                grid_pct = 100.0 * grid_sec / epoch_sec if np.isfinite(epoch_sec) and epoch_sec > 0.0 else float("nan")
                _log_line(
                    log,
                    f"  timing: epoch={epoch_sec:.3f}s grid_update={grid_sec:.3f}s "
                    f"({grid_pct:.1f}%) runs={grid_update_runs_epoch}",
                )
                if grid_update_runs_epoch > 0:
                    _log_line(
                        log,
                        "  w_pred: "
                        f"samples={int(row['grid_total_samples'])} "
                        f"mean={row['wpred_mean']:.3f} q50={row['wpred_q50']:.3f} "
                        f"q90={row['wpred_q90']:.3f} q99={row['wpred_q99']:.3f} "
                        f"max={row['wpred_max']:.3f}",
                    )
                    _log_line(
                        log,
                        "  w_pred focus: "
                        f"s<=a0={100.0 * row['wpred_frac_s_le_a0']:.1f}% "
                        f"s<=1.5a0={100.0 * row['wpred_frac_s_le_1p5a0']:.1f}% "
                        f"w>=0.5={100.0 * row['wpred_frac_ge_05']:.1f}% "
                        f"w>=0.8={100.0 * row['wpred_frac_ge_08']:.1f}%",
                    )
                _log_line(
                    log,
                    "  parts: "
                    f"state={float(train_parts.state):.3e} "
                    f"x1_state={float(train_parts.x1_state):.3e} "
                    f"x2_state={float(train_parts.x2_state):.3e} "
                    f"x2dot={float(train_parts.x2dot):.3e} "
                    f"x3r={float(train_parts.x3_range):.3e} "
                    f"ftsr={float(train_parts.fts_range):.3e}",
                )
                _log_line(log, f"  rec: x1={float(train_parts.x1_rec):.2f}% x3={float(train_parts.x3_rec):.2f}%")
                _log_line(log, f"  nn: F_contact err={float(train_parts.fts_rollout_rec):.2f}%")
                _log_line(log, f"  mech: ks={prepared.mech_true[0]:.6e} (known) cs={prepared.mech_true[1]:.6e} (known)")
                _log_line(log, f"KAN val epoch {epoch + 1} val={float(val_total.detach()):.6e}")
                _log_line(
                    log,
                    "  val parts: "
                    f"state={float(val_parts.state):.3e} "
                    f"x1_state={float(val_parts.x1_state):.3e} "
                    f"x2_state={float(val_parts.x2_state):.3e} "
                    f"x2dot={float(val_parts.x2dot):.3e} "
                    f"x3r={float(val_parts.x3_range):.3e} "
                    f"ftsr={float(val_parts.fts_range):.3e}",
                )
                _log_line(log, f"  val rec: x1={float(val_parts.x1_rec):.2f}% x3={float(val_parts.x3_rec):.2f}%")
                _log_line(log, f"  val nn: F_contact err={float(val_parts.fts_rollout_rec):.2f}%")

            if ((epoch + 1) % cfg.checkpoint_every) == 0:
                payload = {
                    "epoch": int(epoch + 1),
                    "history": history,
                    "config": asdict(cfg),
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "state_mean": _model_normalizer_arrays(model)[0],
                    "state_scale": _model_normalizer_arrays(model)[1],
                    "hidden_normalizer": copy.deepcopy(hidden_normalizer_meta),
                    "known_pars": prepared.known_pars,
                    "eta_star_true": prepared.eta_star_true,
                    "mech_true": prepared.mech_true,
                    "lr": float(lr),
                    "grad_ema": float(grad_ema),
                    "grad_target": float(grad_target),
                    "recent_losses": list(recent_losses),
                    "window_meta": {
                        "role": split.role,
                        "label": split.label,
                        "title": _window_title(split.role),
                        "start_idx": split.start_idx,
                        "stop_idx": split.stop_idx,
                        "t_start": split.t_start,
                        "t_stop": split.t_stop,
                    },
                }
                _save_torch(periodic_checkpoint_path, payload)
                _log_line(log, f"checkpoint saved | epoch={epoch + 1} | path={periodic_checkpoint_path}")

            if not step_update_accepted:
                if retry_hit_cap:
                    stop_reason = (
                        f"adam_retry_cap_hit:{step_reject_reason or step_accept_reason}"
                    )
                    stop_kind = "adam_retry_cap_stop"
                else:
                    stop_reason = f"adam_step_rejected:{step_reject_reason or step_accept_reason}"
                    stop_kind = "adam_reject_stop"
                stop_epoch = int(epoch + 1)
                _write_event(
                    event_log,
                    {
                        "event": stop_kind,
                        "epoch": int(stop_epoch),
                        "loss_before": float(train_loss_before),
                        "accepted_loss": float(train_total.detach()),
                        "candidate_loss": float(step_candidate_loss),
                        "backtracking_retries": int(step_retry_count),
                        "retry_cap": int(retry_cap),
                        "retry_spike": bool(retry_spike),
                        "previous_step_retry_count": float(previous_retry_count),
                        "adam_state_reset_attempted": bool(adam_state_reset_attempted),
                        "adam_state_reset_accepted": bool(adam_state_reset_accepted),
                        "adam_state_reset_trigger_retries": int(adam_state_reset_trigger_retries),
                        "adam_state_reset_retries": int(adam_state_reset_retries),
                        "reject_reason": step_reject_reason or step_accept_reason,
                    },
                )
                _log_line(
                    log,
                    "KAN Adam stop -- rejected update is terminal "
                    f"| epoch={stop_epoch} "
                    f"| reason={step_reject_reason or step_accept_reason} "
                    f"| retries={step_retry_count} "
                    f"| loss={train_loss_before:.6e}->{float(train_total.detach()):.6e}; "
                    "no further epochs will be consumed",
                )
                break

            if lr_adapt_active and np.isfinite(grad_norm) and grad_norm > 0.0:
                if not np.isfinite(grad_ema):
                    grad_ema = grad_norm
                else:
                    grad_ema = cfg.lr_ema_alpha * grad_ema + (1.0 - cfg.lr_ema_alpha) * grad_norm
                if not np.isfinite(grad_target):
                    grad_target = grad_ema
                ratio = grad_target / (grad_ema + cfg.lr_eps)
                lr = max(cfg.lr_min, min(float(lr * (ratio ** cfg.lr_eta)), cfg.lr_max))
                _set_optimizer_lr(optimizer, lr)

            if cfg.plateau_early_stop:
                recent_losses.append(float(train_total.detach()))
                if len(recent_losses) > int(cfg.plateau_window):
                    recent_losses.pop(0)

            if float(train_total.detach()) < float(cfg.good_enough_loss):
                _log_line(log, f"  early-stop: good_enough (loss < {cfg.good_enough_loss:.1e})")
                good_enough_reached = True
                break
            if (
                cfg.plateau_early_stop
                and len(recent_losses) == int(cfg.plateau_window)
                and abs(recent_losses[-1] - recent_losses[0]) < float(cfg.plateau_tol)
            ):
                _log_line(
                    log,
                    f"  early-stop: plateau_{cfg.plateau_window}ep (|Δloss| < {cfg.plateau_tol:.1e})",
                )
                break
            if cfg.recent_val_early_stop:
                stats = _recent_val_plateau_stats(
                    history,
                    min_epochs=int(cfg.recent_val_window),
                    compare_gap=int(cfg.recent_val_compare_gap),
                )
                if stats is not None and stats["rel_gap_to_prev"] < float(cfg.recent_val_rel_current_frac):
                    _log_line(
                        log,
                        "  early-stop: recent_val_plateau "
                        f"(min_epochs={cfg.recent_val_window}, gap={cfg.recent_val_compare_gap} | "
                        f"|val_n+gap-val_n|/val_n={stats['rel_gap_to_prev']:.6f} < "
                        f"{cfg.recent_val_rel_current_frac:.6f}) "
                        f"| val_n={stats['previous']:.6e} val_n+gap={stats['current']:.6e}",
                    )
                    break

        if (
            failure_reason == ""
            and bool(cfg.handoff_enabled)
            and not bool(cfg.handoff_replay)
            and len(history) == int(cfg.adam_epochs)
        ):
            handoff_epoch = int(len(history))
            handoff_epoch_path = _handoff_epoch_path(cfg, file_tag, handoff_epoch)
            handoff_epoch_summary_path = _handoff_epoch_summary_path(cfg, file_tag, handoff_epoch)
            _save_handoff_checkpoint(
                path=handoff_path,
                summary_path=handoff_summary_path,
                epoch_path=handoff_epoch_path,
                epoch_summary_path=handoff_epoch_summary_path,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                split=split,
                history=history,
                best_payload=best_payload,
                best_epoch=best_epoch,
                best_val=best_val,
                sanity_payload=sanity_payload,
                prepared=prepared,
                lr=lr,
                grad_ema=grad_ema,
                grad_target=grad_target,
                recent_losses=recent_losses,
                train_loss_last=train_loss_last,
                warmstart_payload=warmstart_payload,
            )
            _log_line(
                log,
                "handoff checkpoint saved: "
                f"epoch={handoff_epoch} path={handoff_path} "
                f"epoch_tagged_path={handoff_epoch_path} "
                f"summary={handoff_summary_path}",
            )

        if (
            failure_reason == ""
            and stop_reason == ""
            and bool(cfg.lbfgs_enabled)
            and int(cfg.lbfgs_steps) > 0
            and not good_enough_reached
        ):
            lbfgs_effective_steps = int(cfg.lbfgs_steps)
            planned_total_epochs = int(cfg.adam_epochs) + int(cfg.lbfgs_steps)
            lbfgs_optimizer, lbfgs_line_search = _make_lbfgs_optimizer(cfg, model)
            _log_line(
                log,
                "LBFGS phase start: "
                f"after_adam_epochs={len(history)} "
                f"planned_total_epochs={planned_total_epochs} "
                f"lbfgs_epochs={lbfgs_effective_steps} lr={cfg.lbfgs_lr:.3e} "
                f"line_search={lbfgs_line_search} "
                f"max_iter={cfg.lbfgs_max_iter} max_eval={cfg.lbfgs_max_eval} "
                f"tol_grad={cfg.lbfgs_tolerance_grad:.1e} "
                f"tol_change={cfg.lbfgs_tolerance_change:.1e} "
                f"ys_threshold={cfg.lbfgs_ys_threshold:.1e} "
                "AGU=frozen",
            )

            for lbfgs_outer_step in range(1, lbfgs_effective_steps + 1):
                lbfgs_wall_start = perf_counter()
                outer_epoch = len(history) + 1
                closure_calls = 0
                closure_last_loss = float("nan")
                closure_last_grad_norm = float("nan")
                closure_error = ""
                lbfgs_start_state = copy.deepcopy(model.state_dict())
                lbfgs_start_opt_state = copy.deepcopy(lbfgs_optimizer.state_dict())
                lbfgs_start_params = _capture_params(model)
                lbfgs_param_delta_l2 = float("nan")
                lbfgs_param_delta_max_abs = float("nan")
                lbfgs_loss_drop = float("nan")
                lbfgs_internal_debug: dict[str, Any] = {}

                model.eval()
                try:
                    with torch.no_grad():
                        loss_before_t, parts_before, traj_before = evaluate_split(
                            force_module=model,
                            known_pars=prepared.known_pars,
                            mech_true=mech_true_t,
                            ode_true=tensors["ode_full"],
                            x2dot_true=tensors["x2dot_full"],
                            contact_mask=tensors["contact_full"],
                            times=tensors["times_full"],
                            ode_method=cfg.ode_method,
                            ode_rtol=cfg.ode_rtol,
                            ode_atol=cfg.ode_atol,
                            eta_star_true=prepared.eta_star_true,
                            loss_indices=tensors["train_idx"],
                        )
                    lbfgs_loss_before = float(loss_before_t.detach())
                except Exception as err:
                    _log_line(log, f"LBFGS stop -- pre-step train eval failed at step {lbfgs_outer_step}: {err}")
                    break

                def closure() -> torch.Tensor:
                    nonlocal closure_calls, closure_last_loss, closure_last_grad_norm, closure_error
                    closure_calls += 1
                    lbfgs_optimizer.zero_grad()
                    try:
                        closure_total, _, _ = evaluate_split(
                            force_module=model,
                            known_pars=prepared.known_pars,
                            mech_true=mech_true_t,
                            ode_true=tensors["ode_full"],
                            x2dot_true=tensors["x2dot_full"],
                            contact_mask=tensors["contact_full"],
                            times=tensors["times_full"],
                            ode_method=cfg.ode_method,
                            ode_rtol=cfg.ode_rtol,
                            ode_atol=cfg.ode_atol,
                            eta_star_true=prepared.eta_star_true,
                            loss_indices=tensors["train_idx"],
                        )
                        if not torch.isfinite(closure_total):
                            closure_error = "lbfgs_closure_loss_nonfinite"
                            _write_event(
                                event_log,
                                {
                                    "event": "lbfgs_closure",
                                    "epoch": int(outer_epoch),
                                    "lbfgs_outer_step": int(lbfgs_outer_step),
                                    "closure_call": int(closure_calls),
                                    "loss": float(closure_total.detach()),
                                    "grad_norm": float("nan"),
                                    "accepted": False,
                                    "reject_reason": closure_error,
                                },
                            )
                            raise RuntimeError(closure_error)
                        closure_total.backward()
                        closure_last_loss = float(closure_total.detach())
                        closure_last_grad_norm = _grad_norm(model)
                        _write_event(
                            event_log,
                            {
                                "event": "lbfgs_closure",
                                "epoch": int(outer_epoch),
                                "lbfgs_outer_step": int(lbfgs_outer_step),
                                "closure_call": int(closure_calls),
                                "loss": float(closure_last_loss),
                                "grad_norm": float(closure_last_grad_norm),
                                "line_search": lbfgs_line_search,
                                "accepted": True,
                                "reject_reason": "",
                            },
                        )
                        return closure_total
                    except Exception as err:
                        if closure_error == "":
                            closure_error = f"lbfgs_closure_exception:{err}"
                        raise

                model.train()
                lbfgs_returned_loss = float("nan")
                lbfgs_stop_reason = ""
                try:
                    returned_loss = lbfgs_optimizer.step(closure)
                    lbfgs_internal_debug = _lbfgs_internal_debug(lbfgs_optimizer)
                    if isinstance(returned_loss, torch.Tensor):
                        lbfgs_returned_loss = float(returned_loss.detach())
                    else:
                        lbfgs_returned_loss = float(returned_loss)
                except Exception as err:
                    lbfgs_internal_debug = _lbfgs_internal_debug(lbfgs_optimizer)
                    lbfgs_stop_reason = closure_error or f"lbfgs_step_exception:{err}"
                    model.load_state_dict(copy.deepcopy(lbfgs_start_state))
                    lbfgs_optimizer.load_state_dict(copy.deepcopy(lbfgs_start_opt_state))
                    _write_event(
                        event_log,
                        {
                            "event": "lbfgs_step_exception",
                            "epoch": int(outer_epoch),
                            "lbfgs_outer_step": int(lbfgs_outer_step),
                            "closure_calls": int(closure_calls),
                            "loss_before": float(lbfgs_loss_before),
                            "returned_loss": float(lbfgs_returned_loss),
                            "reject_reason": lbfgs_stop_reason,
                            "lbfgs_internal": lbfgs_internal_debug,
                        },
                    )
                    _log_line(
                        log,
                        f"LBFGS stop -- step={lbfgs_outer_step} reason={lbfgs_stop_reason} "
                        f"closure_calls={closure_calls}",
                    )
                    break

                model.train()
                lbfgs_reject_reason = ""
                step_update_accepted = True
                grad_norm_raw = float("nan")
                try:
                    lbfgs_optimizer.zero_grad()
                    train_total, train_parts, accepted_train_traj = evaluate_split(
                        force_module=model,
                        known_pars=prepared.known_pars,
                        mech_true=mech_true_t,
                        ode_true=tensors["ode_full"],
                        x2dot_true=tensors["x2dot_full"],
                        contact_mask=tensors["contact_full"],
                        times=tensors["times_full"],
                        ode_method=cfg.ode_method,
                        ode_rtol=cfg.ode_rtol,
                        ode_atol=cfg.ode_atol,
                        eta_star_true=prepared.eta_star_true,
                        loss_indices=tensors["train_idx"],
                    )
                    train_loss_after = float(train_total.detach())
                    lbfgs_loss_drop = float(lbfgs_loss_before - train_loss_after)
                    lbfgs_param_delta_l2, lbfgs_param_delta_max_abs = _param_delta_stats(
                        model,
                        lbfgs_start_params,
                    )
                    if not torch.isfinite(train_total):
                        lbfgs_reject_reason = "lbfgs_after_loss_nonfinite"
                    else:
                        train_total.backward()
                        grad_norm_raw = _grad_norm(model)
                except Exception as err:
                    train_loss_after = float("inf")
                    train_parts = parts_before
                    accepted_train_traj = traj_before
                    lbfgs_loss_drop = float("-inf")
                    lbfgs_reject_reason = f"lbfgs_after_eval_exception:{err}"

                if lbfgs_reject_reason != "":
                    model.load_state_dict(copy.deepcopy(lbfgs_start_state))
                    lbfgs_optimizer.load_state_dict(copy.deepcopy(lbfgs_start_opt_state))
                    _write_event(
                        event_log,
                        {
                            "event": "lbfgs_safety_stop",
                            "epoch": int(outer_epoch),
                            "lbfgs_outer_step": int(lbfgs_outer_step),
                            "closure_calls": int(closure_calls),
                            "loss_before": float(lbfgs_loss_before),
                            "loss_after": float(train_loss_after),
                            "reject_reason": lbfgs_reject_reason,
                            "action": "rollback_and_stop",
                        },
                    )
                    _log_line(
                        log,
                        f"LBFGS stop -- safety reject at step={lbfgs_outer_step} "
                        f"epoch={outer_epoch} reason={lbfgs_reject_reason}; "
                        "rollback parameters and stop",
                    )
                    break

                grad_norm = float(grad_norm_raw)
                model.eval()
                try:
                    with torch.no_grad():
                        val_total, val_parts, _ = evaluate_split(
                            force_module=model,
                            known_pars=prepared.known_pars,
                            mech_true=mech_true_t,
                            ode_true=tensors["ode_full"],
                            x2dot_true=tensors["x2dot_full"],
                            contact_mask=tensors["contact_full"],
                            times=tensors["times_full"],
                            ode_method=cfg.ode_method,
                            ode_rtol=cfg.ode_rtol,
                            ode_atol=cfg.ode_atol,
                            eta_star_true=prepared.eta_star_true,
                            loss_indices=tensors["val_idx"],
                        )
                        if not torch.isfinite(val_total):
                            _log_line(log, f"LBFGS stop -- val_loss_nonfinite at step {lbfgs_outer_step}")
                            break
                except Exception as err:
                    _log_line(log, f"LBFGS stop -- val_exception at step {lbfgs_outer_step}: {err}")
                    break

                epoch_sec = float(perf_counter() - lbfgs_wall_start)
                soft_mask_meta = model.soft_mask_summary()
                x3dot_meta = model.x3dot_summary() if hasattr(model, "x3dot_summary") else {
                    "enabled": False,
                    "q_init": float("nan"),
                    "q_mean": float("nan"),
                    "q_scale": float("nan"),
                    "q_init_trainable": False,
                    "lag_detach": False,
                }
                row = {
                    "epoch": float(outer_epoch),
                    "train_loss": float(train_total.detach()),
                    "val_loss": float(val_total.detach()),
                    "epoch_sec": epoch_sec,
                    "grid_update_sec": 0.0,
                    "grid_update_runs": 0.0,
                    "train_state": float(train_parts.state),
                    "val_state": float(val_parts.state),
                    "train_x1_state": float(train_parts.x1_state),
                    "val_x1_state": float(val_parts.x1_state),
                    "train_x2_state": float(train_parts.x2_state),
                    "val_x2_state": float(val_parts.x2_state),
                    "train_x2dot": float(train_parts.x2dot),
                    "val_x2dot": float(val_parts.x2dot),
                    "train_x3_range": float(train_parts.x3_range),
                    "val_x3_range": float(val_parts.x3_range),
                    "train_fts_range": float(train_parts.fts_range),
                    "val_fts_range": float(val_parts.fts_range),
                    "train_x1_rec": float(train_parts.x1_rec),
                    "val_x1_rec": float(val_parts.x1_rec),
                    "train_x3_rec": float(train_parts.x3_rec),
                    "val_x3_rec": float(val_parts.x3_rec),
                    "train_fts_rollout_rec": float(train_parts.fts_rollout_rec),
                    "val_fts_rollout_rec": float(val_parts.fts_rollout_rec),
                    "g_nn": float(model.gain().detach().cpu().item()),
                    "soft_mask_enabled": bool(soft_mask_meta["enabled"]),
                    "soft_mask_trainable": bool(soft_mask_meta["trainable"]),
                    "soft_mask_s0": float(soft_mask_meta["s0"]),
                    "soft_mask_s0_a0": float(soft_mask_meta["s0_a0"]),
                    "soft_mask_alpha": float(soft_mask_meta["alpha"]),
                    "soft_mask_alpha_a0": float(soft_mask_meta["alpha_a0"]),
                    "soft_mask_m_min": 0.0,
                    "x3dot_input_enabled": bool(x3dot_meta["enabled"]),
                    "x3dot_q_init": float(x3dot_meta["q_init"]),
                    "x3dot_q_init_trainable": bool(x3dot_meta["q_init_trainable"]),
                    "x3dot_q_mean": float(x3dot_meta["q_mean"]),
                    "x3dot_q_scale": float(x3dot_meta["q_scale"]),
                    "x3dot_lag_detach": bool(x3dot_meta["lag_detach"]),
                    "grad_norm": float(grad_norm),
                    "grad_norm_raw": float(grad_norm_raw),
                    "lr": float(cfg.lbfgs_lr),
                    "phase": "lbfgs_strong_wolfe" if lbfgs_line_search == "strong_wolfe" else "lbfgs",
                    "optimizer": "lbfgs",
                    "step_controller": lbfgs_line_search,
                    "loss_before": float(lbfgs_loss_before),
                    "candidate_loss": float(train_loss_after),
                    "accepted_loss": float(train_total.detach()),
                    "base_lr": float(cfg.lbfgs_lr),
                    "alpha": float("nan"),
                    "effective_lr": float(cfg.lbfgs_lr),
                    "grad_dot_p": float("nan"),
                    "armijo_rhs": float("nan"),
                    "backtracking_retries": 0.0,
                    "step_update_accepted": bool(step_update_accepted),
                    "accepted": bool(step_update_accepted),
                    "reject_reason": lbfgs_reject_reason,
                    "lbfgs_outer_step": float(lbfgs_outer_step),
                    "lbfgs_closure_calls": float(closure_calls),
                    "loss_before_lbfgs_step": float(lbfgs_loss_before),
                    "loss_after_lbfgs_step": float(train_total.detach()),
                    "lbfgs_returned_loss": float(lbfgs_returned_loss),
                    "lbfgs_wall_time": float(epoch_sec),
                    "lbfgs_line_search": lbfgs_line_search,
                    "lbfgs_loss_drop": float(lbfgs_loss_drop),
                    "lbfgs_param_delta_l2": float(lbfgs_param_delta_l2),
                    "lbfgs_param_delta_max_abs": float(lbfgs_param_delta_max_abs),
                    "epoch_retry_count": 0.0,
                    "step_retry_count": 0.0,
                    "grid_base_samples": float("nan"),
                    "grid_extra_samples": 0.0,
                    "grid_total_samples": float("nan"),
                    "wpred_mean": float("nan"),
                    "wpred_max": float("nan"),
                    "wpred_q50": float("nan"),
                    "wpred_q90": float("nan"),
                    "wpred_q99": float("nan"),
                    "wpred_frac_s_le_a0": float("nan"),
                    "wpred_frac_s_le_1p5a0": float("nan"),
                    "wpred_frac_ge_05": float("nan"),
                    "wpred_frac_ge_08": float("nan"),
                }
                history.append(row)
                train_loss_last = float(train_total.detach())

                _write_event(
                    event_log,
                    {
                        "event": "lbfgs_outer_step",
                        "epoch": int(outer_epoch),
                        "lbfgs_outer_step": int(lbfgs_outer_step),
                        "closure_calls": int(closure_calls),
                        "loss_before": float(lbfgs_loss_before),
                        "loss_after": float(train_total.detach()),
                        "returned_loss": float(lbfgs_returned_loss),
                        "accepted": bool(step_update_accepted),
                        "reject_reason": lbfgs_reject_reason,
                        "loss_drop": float(lbfgs_loss_drop),
                        "param_delta_l2": float(lbfgs_param_delta_l2),
                        "param_delta_max_abs": float(lbfgs_param_delta_max_abs),
                        "wall_time": float(epoch_sec),
                        "line_search": lbfgs_line_search,
                        "lbfgs_internal": lbfgs_internal_debug,
                    },
                )

                if float(val_total.detach()) < best_val:
                    best_val = float(val_total.detach())
                    best_epoch = int(outer_epoch)
                    best_snapshot = _snapshot_split(
                        force_module=model,
                        known_pars=prepared.known_pars,
                        eta_star_true=prepared.eta_star_true,
                        mech_true=mech_true_t,
                        split=split,
                        tensors=tensors,
                        dtype=dtype,
                        device=cfg.device,
                        ode_method=cfg.ode_method,
                        ode_rtol=cfg.ode_rtol,
                        ode_atol=cfg.ode_atol,
                    )
                    best_payload = {
                        "epoch": best_epoch,
                        "state_dict": copy.deepcopy(model.state_dict()),
                        "config": asdict(cfg),
                        "state_mean": _model_normalizer_arrays(model)[0],
                        "state_scale": _model_normalizer_arrays(model)[1],
                        "hidden_normalizer": copy.deepcopy(hidden_normalizer_meta),
                        "known_pars": prepared.known_pars,
                        "eta_star_true": prepared.eta_star_true,
                        "mech_true": prepared.mech_true,
                        "sanity": sanity_payload,
                        "window_meta": {
                            "role": split.role,
                            "label": split.label,
                            "title": _window_title(split.role),
                            "start_idx": split.start_idx,
                            "stop_idx": split.stop_idx,
                            "t_start": split.t_start,
                            "t_stop": split.t_stop,
                        },
                        "history": history,
                        "best_snapshot": best_snapshot,
                    }
                    _save_torch(checkpoint_path, best_payload)

                _log_line(log, f"KAN LBFGS step {lbfgs_outer_step} epoch {outer_epoch} train={float(train_total.detach()):.6e}")
                _log_line(
                    log,
                    "  optimizer summary: "
                    f"phase={row['phase']} line_search={lbfgs_line_search} "
                    f"loss={lbfgs_loss_before:.6e}->{float(train_total.detach()):.6e} "
                    f"accepted={'YES' if step_update_accepted else 'NO'} "
                    f"closure_calls={closure_calls} "
                    f"returned_loss={lbfgs_returned_loss:.6e} "
                    f"reason={lbfgs_reject_reason or 'accepted'}",
                )
                _log_line(log, f"  grad_norm={grad_norm:.3e} lbfgs_lr={cfg.lbfgs_lr:.6g}")
                _log_line(log, f"  grad_norm_raw={grad_norm_raw:.3e} grad_norm_scaled={grad_norm:.3e}")
                _log_line(
                    log,
                    "  lbfgs progress: "
                    f"loss_drop={lbfgs_loss_drop:.3e} "
                    f"param_delta_l2={lbfgs_param_delta_l2:.3e} "
                    f"param_delta_max_abs={lbfgs_param_delta_max_abs:.3e}",
                )
                if lbfgs_internal_debug:
                    ls_debug = lbfgs_internal_debug.get("line_search_debug", {})
                    if not isinstance(ls_debug, dict):
                        ls_debug = {}
                    _log_line(
                        log,
                        "  lbfgs internal: "
                        f"t_init={_dbg_float(lbfgs_internal_debug, 't_initial'):.3e} "
                        f"t_final={_dbg_float(lbfgs_internal_debug, 't_final'):.3e} "
                        f"gtd={_dbg_float(lbfgs_internal_debug, 'gtd'):.3e} "
                        f"ys={_dbg_float(lbfgs_internal_debug, 'ys'):.3e} "
                        f"ys_thr={_dbg_float(lbfgs_internal_debug, 'ys_threshold'):.3e} "
                        f"curv={'YES' if bool(lbfgs_internal_debug.get('curvature_update', False)) else 'NO'} "
                        f"hist={int(lbfgs_internal_debug.get('history_len_after', 0) or 0)} "
                        f"H={_dbg_float(lbfgs_internal_debug, 'H_diag_final'):.3e} "
                        f"step_max={_dbg_float(lbfgs_internal_debug, 'actual_step_max_abs'):.3e} "
                        f"ls_reason={ls_debug.get('final_reason', '') or lbfgs_internal_debug.get('break_reason', '')}",
                    )
                _log_line(
                    log,
                    "  soft_mask: "
                    f"enabled={'ON' if row['soft_mask_enabled'] else 'OFF'} "
                    f"trainable={'ON' if row['soft_mask_trainable'] else 'OFF'} "
                    f"s0={row['soft_mask_s0']:.6e} "
                    f"s0/a0={row['soft_mask_s0_a0']:.6g} "
                    f"alpha={row['soft_mask_alpha']:.6e} "
                    f"alpha*a0={row['soft_mask_alpha_a0']:.6g} "
                    f"m_min={row['soft_mask_m_min']:.1f}",
                )
                _log_line(
                    log,
                    "  PlanZ q: "
                    f"enabled={'ON' if row['x3dot_input_enabled'] else 'OFF'} "
                    f"q_init={row['x3dot_q_init']:.6e} "
                    f"trainable={'ON' if row['x3dot_q_init_trainable'] else 'OFF'} "
                    f"q_mean={row['x3dot_q_mean']:.6e} "
                    f"q_scale={row['x3dot_q_scale']:.6e} "
                    f"detach={'ON' if row['x3dot_lag_detach'] else 'OFF'}",
                )
                _log_line(log, f"  timing: lbfgs_outer_step={epoch_sec:.3f}s AGU=frozen closure_calls={closure_calls}")
                _log_line(
                    log,
                    "  parts: "
                    f"state={float(train_parts.state):.3e} "
                    f"x1_state={float(train_parts.x1_state):.3e} "
                    f"x2_state={float(train_parts.x2_state):.3e} "
                    f"x2dot={float(train_parts.x2dot):.3e} "
                    f"x3r={float(train_parts.x3_range):.3e} "
                    f"ftsr={float(train_parts.fts_range):.3e}",
                )
                _log_line(log, f"  rec: x1={float(train_parts.x1_rec):.2f}% x3={float(train_parts.x3_rec):.2f}%")
                _log_line(log, f"  nn: F_contact err={float(train_parts.fts_rollout_rec):.2f}%")
                _log_line(log, f"KAN val epoch {outer_epoch} val={float(val_total.detach()):.6e}")
                _log_line(
                    log,
                    "  val parts: "
                    f"state={float(val_parts.state):.3e} "
                    f"x1_state={float(val_parts.x1_state):.3e} "
                    f"x2_state={float(val_parts.x2_state):.3e} "
                    f"x2dot={float(val_parts.x2dot):.3e} "
                    f"x3r={float(val_parts.x3_range):.3e} "
                    f"ftsr={float(val_parts.fts_range):.3e}",
                )
                _log_line(log, f"  val rec: x1={float(val_parts.x1_rec):.2f}% x3={float(val_parts.x3_rec):.2f}%")
                _log_line(log, f"  val nn: F_contact err={float(val_parts.fts_rollout_rec):.2f}%")

                if (int(outer_epoch) % cfg.checkpoint_every) == 0:
                    payload = {
                        "epoch": int(outer_epoch),
                        "history": history,
                        "config": asdict(cfg),
                        "state_dict": copy.deepcopy(model.state_dict()),
                        "state_mean": _model_normalizer_arrays(model)[0],
                        "state_scale": _model_normalizer_arrays(model)[1],
                        "hidden_normalizer": copy.deepcopy(hidden_normalizer_meta),
                        "known_pars": prepared.known_pars,
                        "eta_star_true": prepared.eta_star_true,
                        "mech_true": prepared.mech_true,
                        "lr": float(cfg.lbfgs_lr),
                        "grad_ema": float(grad_ema),
                        "grad_target": float(grad_target),
                        "recent_losses": list(recent_losses),
                        "window_meta": {
                            "role": split.role,
                            "label": split.label,
                            "title": _window_title(split.role),
                            "start_idx": split.start_idx,
                            "stop_idx": split.stop_idx,
                            "t_start": split.t_start,
                            "t_stop": split.t_stop,
                        },
                    }
                    _save_torch(periodic_checkpoint_path, payload)
                    _log_line(log, f"checkpoint saved | epoch={outer_epoch} | path={periodic_checkpoint_path}")

        if failure_reason != "":
            _log_line(log, f"KAN shard failure -- epoch={failure_epoch} reason={failure_reason}")
        if stop_reason != "":
            _log_line(log, f"KAN shard stopped -- epoch={stop_epoch} kind={stop_kind} reason={stop_reason}")

        final_snapshot = _snapshot_split(
            force_module=model,
            known_pars=prepared.known_pars,
            eta_star_true=prepared.eta_star_true,
            mech_true=mech_true_t,
            split=split,
            tensors=tensors,
            dtype=dtype,
            device=cfg.device,
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
        )
        final_payload = {
            "best": best_payload,
            "history": history,
            "state_mean": _model_normalizer_arrays(model)[0],
            "state_scale": _model_normalizer_arrays(model)[1],
            "hidden_normalizer": copy.deepcopy(hidden_normalizer_meta),
            "known_pars": prepared.known_pars,
            "eta_star_true": prepared.eta_star_true,
            "mech_true": prepared.mech_true,
            "sanity": sanity_payload,
            "window_meta": {
                "role": split.role,
                "label": split.label,
                "title": _window_title(split.role),
                "start_idx": split.start_idx,
                "stop_idx": split.stop_idx,
                "t_start": split.t_start,
                "t_stop": split.t_stop,
            },
            "final_snapshot": final_snapshot,
            "config": asdict(cfg),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "optimizer_event_log_path": str(event_log_path),
            "handoff_checkpoint_path": str(handoff_path),
            "handoff_summary_path": str(handoff_summary_path),
            "handoff_replay": bool(cfg.handoff_replay),
            "stop_kind": stop_kind,
            "stop_epoch": int(stop_epoch),
            "stop_reason": stop_reason,
            "failure_reason": failure_reason,
            "failure_epoch": int(failure_epoch),
        }
        _save_torch(result_path, final_payload)
        _save_pickle(viz_result_path, _make_viz_payload(final_payload))
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        _log_line(
            log,
            f"KAN shard done -- train={history[-1]['train_loss']:.6e} val={history[-1]['val_loss']:.6e} "
            f"| best_epoch={best_epoch} best_val={best_val:.6e}",
        )

    return {
        "shard_index": shard_index,
        "role": split.role,
        "label": split.label,
        "log_path": str(log_path),
        "history_path": str(history_path),
        "optimizer_event_log_path": str(event_log_path),
        "checkpoint_path": str(checkpoint_path),
        "result_path": str(result_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }


def merge_shard_results(cfg=None) -> dict[str, Any]:
    cfg = default_config() if cfg is None else cfg
    prepared = prepare_data(cfg)
    selected_window_indices = _selected_window_indices(cfg, prepared.splits)
    merged: list[dict[str, Any]] = []
    for shard_index in range(1, cfg.shard_count + 1):
        window_index = int(selected_window_indices[shard_index - 1])
        split = prepared.splits[window_index - 1]
        result_path = cfg.result_dir / f"kan_full_test_result_{_window_file_tag(split.role)}.pt"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing shard result: {result_path}")
        payload = torch.load(result_path, map_location="cpu", weights_only=False)
        meta = payload.get("window_meta", {})
        merged.append(
            {
                "shard_index": shard_index,
                "role": str(meta.get("role", "")),
                "label": str(meta.get("label", "")),
                "title": str(meta.get("title", "")),
                "best_epoch": int(payload.get("best_epoch", -1)),
                "best_val_loss": float(payload.get("best_val_loss", float("inf"))),
                "result_path": str(result_path),
            }
        )
    merged = sorted(merged, key=lambda item: item["shard_index"])
    best_loss = min(item["best_val_loss"] for item in merged)
    out = {
        "total_shards": len(merged),
        "best_val_loss": float(best_loss),
        "shards": merged,
    }
    summary_path = cfg.result_dir / "kan_full_test_merged_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_json_default)
    return out


def run_full_test(cfg=None) -> dict[str, object]:
    return run_full_test_shard(cfg)


__all__ = ["merge_shard_results", "run_full_test", "run_full_test_shard"]
