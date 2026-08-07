"""Training loop for the standalone AFM05 Python stage2light library."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from AFM05.rng_state import capture_rng_state, restore_rng_state
from AFM05.stage2light.config import default_config
from AFM05.stage2light.data import (
    PreparedData,
    Prestage2WarmstartCandidate,
    Stage1WarmstartCandidate,
    WindowSplit,
    load_stage1_warmstart_record,
    prepare_data,
    select_prestage2_candidate,
    select_stage1_candidate,
    select_stage1_candidate_by_trial_id,
)
from AFM05.stage2light.kan_backend import (
    KANForceModule,
    initial_grid_support_from_raw_inputs,
    initial_grid_support_to_meta,
)
from AFM05.stage2light.losses import TorchLossParts, evaluate_split
from AFM05.stage2light.optim.stage2_lbfgs import LBFGS as Stage2LBFGS
from AFM05.stage2light.rollout import (
    LearnableMechModule,
    _afm05_known_fields,
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


def _is_w0_window_mode(window_mode: str | None) -> bool:
    mode = str(window_mode or "").strip().lower()
    return mode in ("w0", "stage2_w0", "stage2-w0", "first_contact", "first-contact")


def _window_title(role: str, window_mode: str | None = None) -> str:
    role_norm = role.strip().lower()
    if role_norm == "first_contact":
        return "W0: first-contact window"
    if role_norm == "middle":
        return "W1: middle window"
    if role_norm == "max_x1_pp_change":
        return "window2: the most drastic region"
    if role_norm == "tail_stable":
        return "window3: stable region at the end"
    return role


def _window_meta_dict(split: WindowSplit, window_mode: str | None = None) -> dict[str, Any]:
    return {
        "role": str(split.role),
        "label": str(split.label),
        "title": _window_title(split.role, window_mode),
        "start_idx": int(split.start_idx),
        "stop_idx": int(split.stop_idx),
        "length": int(split.stop_idx) - int(split.start_idx) + 1,
        "t_start": float(split.t_start),
        "t_stop": float(split.t_stop),
    }


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
        "x3_range_amp": float(parts.x3_range_amp),
        "fts_range_amp": float(parts.fts_range_amp),
        "cont": float(parts.cont),
        "x1_rec": float(parts.x1_rec),
        "x2_rec": float(parts.x2_rec),
        "x2dot_rec": float(parts.x2dot_rec),
    }


def _validation_final_only(cfg) -> bool:
    return str(getattr(cfg, "val_eval_mode", "per_epoch")).strip().lower() == "final_only"


def _should_eval_validation(cfg, epoch_number: int) -> bool:
    _ = epoch_number
    return not _validation_final_only(cfg)


def _nan_loss_parts() -> TorchLossParts:
    nan = float("nan")
    return TorchLossParts(
        state=nan,
        x1_state=nan,
        x2_state=nan,
        x2dot=nan,
        x3_range=nan,
        fts_range=nan,
        x3_range_amp=nan,
        fts_range_amp=nan,
        cont=nan,
        x1_rec=nan,
        x2_rec=nan,
        x2dot_rec=nan,
    )


def _nan_loss_tensor(*, dtype: torch.dtype, device: str) -> torch.Tensor:
    return torch.as_tensor(float("nan"), dtype=dtype, device=device)


def _iter_named_params(*modules: tuple[str, torch.nn.Module]):
    for prefix, module in modules:
        for name, param in module.named_parameters():
            yield f"{prefix}.{name}", param


def _grad_norm(*modules: tuple[str, torch.nn.Module]) -> float:
    sq = 0.0
    for _name, param in _iter_named_params(*modules):
        if param.grad is None:
            continue
        g = param.grad.detach()
        sq += float(torch.sum(g * g))
    return float(np.sqrt(max(sq, 0.0)))


def _capture_grads(*modules: tuple[str, torch.nn.Module]) -> dict[str, torch.Tensor | None]:
    return {name: None if param.grad is None else param.grad.detach().clone() for name, param in _iter_named_params(*modules)}


def _restore_grads(grads: dict[str, torch.Tensor | None], *modules: tuple[str, torch.nn.Module]) -> None:
    for name, param in _iter_named_params(*modules):
        grad = grads.get(name)
        if grad is None:
            param.grad = None
        else:
            param.grad = grad.detach().clone().to(device=param.device, dtype=param.dtype)


def _capture_params(*modules: tuple[str, torch.nn.Module]) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in _iter_named_params(*modules)
    }


def _param_delta_stats(
    base_params: dict[str, torch.Tensor],
    *modules: tuple[str, torch.nn.Module],
) -> tuple[float, float]:
    sq = 0.0
    max_abs = 0.0
    for name, param in _iter_named_params(*modules):
        base = base_params.get(name)
        if base is None:
            continue
        delta = param.detach() - base.to(device=param.device, dtype=param.dtype)
        sq += float(torch.sum(delta * delta))
        if delta.numel() > 0:
            max_abs = max(max_abs, float(torch.max(torch.abs(delta))))
    return float(np.sqrt(max(sq, 0.0))), float(max_abs)


def _grad_dot_delta(
    base_params: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor | None],
    *modules: tuple[str, torch.nn.Module],
) -> float:
    total = 0.0
    for name, param in _iter_named_params(*modules):
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


def _canonical_mech_parameterization(value: Any, *, default: str = "direct_unbounded") -> str:
    mode = str(value if value is not None else default).strip().lower().replace("-", "_")
    aliases = {
        "direct": "direct_unbounded",
        "physical": "direct_unbounded",
        "unbounded": "direct_unbounded",
        "direct_unbounded": "direct_unbounded",
        "exp": "log_relative",
        "log": "log_relative",
        "relative": "log_relative",
        "log_relative": "log_relative",
        "log_relative_bounded": "log_relative",
        "sigmoid": "sigmoid_bounded",
        "bounded": "sigmoid_bounded",
        "sigmoid_bound": "sigmoid_bounded",
        "sigmoid_bounds": "sigmoid_bounded",
        "sigmoid_bounded": "sigmoid_bounded",
        "legacy": "sigmoid_bounded",
        "legacy_sigmoid": "sigmoid_bounded",
    }
    return aliases.get(mode, default)


def _make_optimizer(cfg, model: torch.nn.Module, mech_module: LearnableMechModule) -> tuple[torch.optim.Optimizer, str]:
    optimizer_name = str(cfg.optimizer_name).strip().lower()
    amsgrad = optimizer_name == "amsgrad"
    if optimizer_name not in ("adam", "amsgrad"):
        optimizer_name = "amsgrad"
        amsgrad = True
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mech_module.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        amsgrad=amsgrad,
    )
    return optimizer, optimizer_name


def _make_lbfgs_optimizer(cfg, model: torch.nn.Module, mech_module: LearnableMechModule) -> tuple[torch.optim.Optimizer, str]:
    line_search_fn = "strong_wolfe" if bool(cfg.lbfgs_strong_wolfe) else None
    max_iter = max(1, int(cfg.lbfgs_max_iter))
    max_eval = max(max_iter, int(cfg.lbfgs_max_eval))
    optimizer = Stage2LBFGS(
        list(model.parameters()) + list(mech_module.parameters()),
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


def _fingerprint_update(hasher: "hashlib._Hash", obj: Any) -> None:
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        hasher.update(b"tensor")
        hasher.update(str(t.dtype).encode("utf-8"))
        hasher.update(str(tuple(t.shape)).encode("utf-8"))
        if t.numel() > 0:
            hasher.update(t.numpy().tobytes())
        return
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        hasher.update(b"ndarray")
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(str(tuple(arr.shape)).encode("utf-8"))
        hasher.update(arr.tobytes())
        return
    if isinstance(obj, dict):
        hasher.update(b"dict")
        for key in sorted(obj.keys(), key=lambda item: repr(item)):
            hasher.update(repr(key).encode("utf-8"))
            _fingerprint_update(hasher, obj[key])
        return
    if isinstance(obj, (list, tuple)):
        hasher.update(type(obj).__name__.encode("utf-8"))
        hasher.update(str(len(obj)).encode("utf-8"))
        for item in obj:
            _fingerprint_update(hasher, item)
        return
    hasher.update(repr(obj).encode("utf-8"))


def _fingerprint_hash(obj: Any) -> str:
    hasher = hashlib.sha256()
    _fingerprint_update(hasher, obj)
    return hasher.hexdigest()


def _filtered_state_dict(module: torch.nn.Module, needles: tuple[str, ...]) -> dict[str, torch.Tensor]:
    lowered = tuple(needle.lower() for needle in needles)
    return {
        key: value
        for key, value in module.state_dict().items()
        if any(needle in key.lower() for needle in lowered)
    }


def _fingerprint_payload(
    *,
    model: torch.nn.Module,
    mech_module: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    grid_inputs: torch.Tensor | None = None,
) -> dict[str, str]:
    payload = {
        "model_state_hash": _fingerprint_hash(model.state_dict()),
        "mech_state_hash": _fingerprint_hash(mech_module.state_dict()),
        "kan_grid_hash": _fingerprint_hash(_filtered_state_dict(model, ("grid",))),
        "kan_spline_hash": _fingerprint_hash(_filtered_state_dict(model, ("coef", "spline", "scale"))),
        "rng_state_hash": _fingerprint_hash(capture_rng_state()),
    }
    if optimizer is not None:
        payload["optimizer_state_hash"] = _fingerprint_hash(optimizer.state_dict())
    else:
        payload["optimizer_state_hash"] = "none"
    if grid_inputs is not None:
        payload["grid_inputs_hash"] = _fingerprint_hash(grid_inputs)
    else:
        payload["grid_inputs_hash"] = "none"
    return payload


def _log_fingerprint(
    log,
    event_log,
    *,
    epoch: int,
    stage: str,
    phase: str,
    model: torch.nn.Module,
    mech_module: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    grid_inputs: torch.Tensor | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = _fingerprint_payload(
        model=model,
        mech_module=mech_module,
        optimizer=optimizer,
        grid_inputs=grid_inputs,
    )
    event = {
        "event": "fingerprint",
        "epoch": int(epoch),
        "phase": str(phase),
        "stage": str(stage),
        **payload,
    }
    if extra:
        event.update(extra)
    _write_event(event_log, event)
    _log_line(
        log,
        "fingerprint | "
        f"epoch={int(epoch)} | phase={phase} | stage={stage} | "
        f"model={payload['model_state_hash']} | mech={payload['mech_state_hash']} | "
        f"grid={payload['kan_grid_hash']} | spline={payload['kan_spline_hash']} | "
        f"optimizer={payload['optimizer_state_hash']} | rng={payload['rng_state_hash']} | "
        f"grid_inputs={payload['grid_inputs_hash']}",
    )


def _role_short(role: str, window_mode: str | None = None) -> str:
    role_norm = role.strip().lower()
    if role_norm == "first_contact":
        return "W0"
    if role_norm == "middle":
        return "W1"
    if role_norm == "max_x1_pp_change":
        return "W2"
    if role_norm == "tail_stable":
        return "W3"
    return role


def _selected_window_indices(cfg, splits: tuple[WindowSplit, ...]) -> list[int]:
    requested = int(getattr(cfg, "train_window_index", 0))
    if requested > len(splits):
        raise ValueError(f"invalid train_window_index={requested}; available windows=1..{len(splits)}")
    if requested > 0:
        return [requested]

    # Default to the canonical single window for the selected manifest:
    # W0 for first-contact mode, W1 for the legacy stage2 three-window mode.
    role_to_index = {split.role.strip().lower(): idx for idx, split in enumerate(splits, start=1)}
    preferred_roles = ("first_contact",) if _is_w0_window_mode(getattr(cfg, "window_mode", None)) else ("middle", "first_contact")
    preferred = [role_to_index[role] for role in preferred_roles if role in role_to_index]
    if len(preferred) == 1:
        return preferred
    return list(range(1, len(splits) + 1))


def _geometric_midpoint(lo: float, hi: float) -> float:
    return float(math.sqrt(max(lo, 1.0e-30) * max(hi, 1.0e-30)))


def _current_mech_numpy(mech_module: LearnableMechModule) -> tuple[float, float]:
    mech = mech_module().detach().cpu().numpy()
    return float(mech[0]), float(mech[1])


def _observable_grid_inputs_from_ode(
    *,
    ode_train: torch.Tensor,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    x3_norm_support: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Build AGU inputs in raw physical coordinates before normalization.

    x1/x2 come from observed data. When no rollout-derived x3 support is
    available, the x3 axis follows the current normalized x3 initial state.
    """

    n = int(ode_train.shape[1])
    inputs = torch.empty((n, 3), dtype=ode_train.dtype, device=ode_train.device)
    inputs[:, 0:2] = ode_train[0:2, :].transpose(0, 1)
    if n <= 1:
        inputs[:, 2] = float(np.asarray(state_mean, dtype=float)[2])
    else:
        mean = np.asarray(state_mean, dtype=float).reshape(-1)
        scale = np.asarray(state_scale, dtype=float).reshape(-1)
        x3_center = float(mean[2])
        x3_scale = max(float(abs(scale[2])), 1.0e-30)
        if x3_norm_support is None:
            x3_init = float(ode_train[2, 0].detach().cpu())
            x3_init_norm = (x3_init - x3_center) / x3_scale
            norm_lo, norm_hi = x3_init_norm - 2.0, x3_init_norm + 1.0
        else:
            norm_lo = float(x3_norm_support[0])
            norm_hi = float(x3_norm_support[1])
            if not (np.isfinite(norm_lo) and np.isfinite(norm_hi) and norm_hi > norm_lo):
                raise ValueError(f"invalid x3_norm_support: {x3_norm_support}")
        x3_norm_axis = torch.linspace(
            norm_lo,
            norm_hi,
            n,
            dtype=ode_train.dtype,
            device=ode_train.device,
        )
        inputs[:, 2] = torch.as_tensor(x3_center, dtype=ode_train.dtype, device=ode_train.device) + (
            torch.as_tensor(x3_scale, dtype=ode_train.dtype, device=ode_train.device) * x3_norm_axis
        )
    return inputs


def _x3_norm_monitor_from_traj(
    traj: torch.Tensor | None,
    force_module: KANForceModule,
    x3_refit_meta: dict[str, Any] | None,
    x3_current_support: tuple[float, float] | None = None,
) -> dict[str, float | bool]:
    if traj is None:
        return {
            "x3_norm_min_epoch": float("nan"),
            "x3_norm_max_epoch": float("nan"),
            "x3_neutral_support_min": float("nan"),
            "x3_neutral_support_max": float("nan"),
            "x3_neutral_support_exceeded": False,
            "x3_neutral_support_lower_exceed": float("nan"),
            "x3_neutral_support_upper_exceed": float("nan"),
        }
    support_min = -1.0
    support_max = 1.0
    if _valid_x3_norm_support(x3_current_support):
        support_min = float(x3_current_support[0])
        support_max = float(x3_current_support[1])
    elif isinstance(x3_refit_meta, dict):
        support_min = float(x3_refit_meta.get("x3_norm_support_min", support_min))
        support_max = float(x3_refit_meta.get("x3_norm_support_max", support_max))
    mean = force_module.state_mean.detach()[2]
    scale = torch.clamp(torch.abs(force_module.state_scale.detach()[2]), min=1.0e-30)
    x3_norm = (traj.detach()[2, :] - mean) / scale
    x3_min = float(torch.min(x3_norm).detach().cpu())
    x3_max = float(torch.max(x3_norm).detach().cpu())
    lower_exceed = max(0.0, float(support_min) - x3_min)
    upper_exceed = max(0.0, x3_max - float(support_max))
    return {
        "x3_norm_min_epoch": x3_min,
        "x3_norm_max_epoch": x3_max,
        "x3_neutral_support_min": float(support_min),
        "x3_neutral_support_max": float(support_max),
        "x3_neutral_support_exceeded": bool(lower_exceed > 0.0 or upper_exceed > 0.0),
        "x3_neutral_support_lower_exceed": float(lower_exceed),
        "x3_neutral_support_upper_exceed": float(upper_exceed),
    }


def _valid_x3_norm_support(support: tuple[float, float] | list[float] | None) -> bool:
    if support is None:
        return False
    try:
        lo = float(support[0])
        hi = float(support[1])
    except Exception:
        return False
    return bool(np.isfinite(lo) and np.isfinite(hi) and hi > lo)


def _x3_norm_support_from_monitor(stats: dict[str, float | bool]) -> tuple[float, float] | None:
    try:
        support = (float(stats["x3_norm_min_epoch"]), float(stats["x3_norm_max_epoch"]))
    except Exception:
        return None
    return support if _valid_x3_norm_support(support) else None


def _x3_norm_support_to_payload(support: tuple[float, float] | list[float] | None) -> list[float] | None:
    if not _valid_x3_norm_support(support):
        return None
    return [float(support[0]), float(support[1])]


def _x3_drift_from_epoch0(
    traj: torch.Tensor | None,
    x3_epoch0: torch.Tensor | None,
) -> dict[str, float]:
    if traj is None or x3_epoch0 is None:
        return {
            "x3_drift_from_epoch0_pct": float("nan"),
            "x3_drift_from_epoch0_abs": float("nan"),
            "x3_drift_from_epoch0_max_abs": float("nan"),
        }
    try:
        x3_now = traj.detach()[2, :].reshape(-1)
        x3_ref = x3_epoch0.detach().to(device=x3_now.device, dtype=x3_now.dtype).reshape(-1)
    except Exception:
        return {
            "x3_drift_from_epoch0_pct": float("nan"),
            "x3_drift_from_epoch0_abs": float("nan"),
            "x3_drift_from_epoch0_max_abs": float("nan"),
        }
    if x3_now.numel() != x3_ref.numel() or x3_now.numel() == 0:
        return {
            "x3_drift_from_epoch0_pct": float("nan"),
            "x3_drift_from_epoch0_abs": float("nan"),
            "x3_drift_from_epoch0_max_abs": float("nan"),
        }
    finite = torch.isfinite(x3_now) & torch.isfinite(x3_ref)
    if not bool(torch.any(finite).detach().cpu()):
        return {
            "x3_drift_from_epoch0_pct": float("nan"),
            "x3_drift_from_epoch0_abs": float("nan"),
            "x3_drift_from_epoch0_max_abs": float("nan"),
        }
    x3_now = x3_now[finite]
    x3_ref = x3_ref[finite]
    diff = x3_now - x3_ref
    drift_abs = torch.sqrt(torch.mean(torch.square(diff)))
    ref_rms = torch.sqrt(torch.mean(torch.square(x3_ref)))
    drift_pct = torch.where(
        ref_rms > torch.as_tensor(1.0e-30, dtype=x3_now.dtype, device=x3_now.device),
        100.0 * drift_abs / ref_rms,
        torch.as_tensor(float("nan"), dtype=x3_now.dtype, device=x3_now.device),
    )
    return {
        "x3_drift_from_epoch0_pct": float(drift_pct.detach().cpu()),
        "x3_drift_from_epoch0_abs": float(drift_abs.detach().cpu()),
        "x3_drift_from_epoch0_max_abs": float(torch.max(torch.abs(diff)).detach().cpu()),
    }


def _format_x3_norm_monitor(stats: dict[str, float | bool]) -> str:
    exceeded = "YES" if bool(stats.get("x3_neutral_support_exceeded", False)) else "NO"
    return (
        "x3_norm: "
        f"epoch=[{float(stats['x3_norm_min_epoch']):.3f}, {float(stats['x3_norm_max_epoch']):.3f}] "
        f"support=[{float(stats['x3_neutral_support_min']):.3f}, {float(stats['x3_neutral_support_max']):.3f}] "
        f"exceeded={exceeded} "
        f"lower={float(stats['x3_neutral_support_lower_exceed']):.3e} "
        f"upper={float(stats['x3_neutral_support_upper_exceed']):.3e}"
    )


def _make_grid_update_inputs(
    *,
    base_inputs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    grid_inputs = base_inputs.detach()
    n = float(grid_inputs.shape[0])
    meta = {
        "base_samples": n,
        "extra_samples": 0.0,
        "total_samples": n,
    }
    return grid_inputs, meta


def _strict_mean_scale_from_x3_pred(x3_pred: torch.Tensor) -> tuple[float, float]:
    values = x3_pred.detach().reshape(-1).cpu().numpy().astype(float)
    if values.size == 0:
        raise ValueError("x3_pred has no samples")
    if not np.all(np.isfinite(values)):
        bad_count = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
        raise ValueError(f"x3_pred contains nonfinite value(s): count={bad_count}")
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(mean):
        raise ValueError(f"x3_pred mean is nonfinite: {mean}")
    if not np.isfinite(scale) or scale <= 1.0e-30:
        raise ValueError(f"x3_pred scale is invalid: {scale:.6e}")
    return mean, scale


def _refit_x3_normalizer_from_preopt_rollout(
    *,
    log,
    cfg,
    prepared: PreparedData,
    split: WindowSplit,
    tensors: dict[str, torch.Tensor],
    train_states: torch.Tensor,
    train_gain_force_reference: torch.Tensor,
    model: KANForceModule,
    mech_module: LearnableMechModule,
) -> tuple[PreparedData, torch.Tensor, dict[str, Any], float]:
    prior_mean = np.asarray(prepared.state_mean, dtype=float).copy()
    prior_scale = np.asarray(prepared.state_scale, dtype=float).copy()

    pre_grid_meta = {"total_samples": 0.0}
    with torch.no_grad():
        gain_before = float(model.initialize_gain_from_reference(train_states, train_gain_force_reference))
        pre_total, _pre_parts, pre_traj = evaluate_split(
            force_module=model,
            known_pars=prepared.known_pars,
            mech_module=mech_module,
            ode_true=tensors["ode_full"],
            x2dot_true=tensors["x2dot_full"],
            contact_mask=tensors["contact_full"],
            times=tensors["times_full"],
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            loss_indices=tensors["train_idx"],
        )
        x3_mean, x3_scale = _strict_mean_scale_from_x3_pred(pre_traj[2, :])
        x3_pred_norm = (pre_traj[2, :].detach() - float(x3_mean)) / float(x3_scale)
        x3_norm_support_min = float(torch.min(x3_pred_norm).detach().cpu())
        x3_norm_support_max = float(torch.max(x3_pred_norm).detach().cpu())
        if not (
            np.isfinite(x3_norm_support_min)
            and np.isfinite(x3_norm_support_max)
            and x3_norm_support_max > x3_norm_support_min
        ):
            raise ValueError(
                "x3_pred normalized support is invalid: "
                f"min={x3_norm_support_min:.6e} max={x3_norm_support_max:.6e}"
            )

        refit_mean = prior_mean.copy()
        refit_scale = prior_scale.copy()
        refit_mean[2] = float(x3_mean)
        refit_scale[2] = float(x3_scale)

        model.set_state_normalizer(refit_mean, refit_scale)
        refit_prepared = replace(prepared, state_mean=refit_mean, state_scale=refit_scale)
        refit_observable_grid_inputs = _observable_grid_inputs_from_ode(
            ode_train=tensors["ode_train"],
            state_mean=refit_prepared.state_mean,
            state_scale=refit_prepared.state_scale,
            x3_norm_support=(x3_norm_support_min, x3_norm_support_max),
        )
        refit_grid_meta = {"total_samples": float("nan")}
        if cfg.adaptive_grid_enabled:
            grid_inputs, refit_grid_meta = _make_grid_update_inputs(base_inputs=refit_observable_grid_inputs)
            model.update_grid_from_normalized_inputs(grid_inputs)
        gain_after = float(model.initialize_gain_from_reference(train_states, train_gain_force_reference))
        post_total, _post_parts, post_traj = evaluate_split(
            force_module=model,
            known_pars=refit_prepared.known_pars,
            mech_module=mech_module,
            ode_true=tensors["ode_full"],
            x2dot_true=tensors["x2dot_full"],
            contact_mask=tensors["contact_full"],
            times=tensors["times_full"],
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            loss_indices=tensors["train_idx"],
        )

    x3_pred_values = pre_traj[2, :].detach()
    meta = {
        "enabled": True,
        "source": "preopt_rollout_x3_pred_full_window",
        "window_role": str(split.role),
        "window_label": str(split.label),
        "prior_x3_mean": float(prior_mean[2]),
        "prior_x3_scale": float(prior_scale[2]),
        "refit_x3_mean": float(x3_mean),
        "refit_x3_scale": float(x3_scale),
        "x3_pred_min": float(torch.min(x3_pred_values).detach()),
        "x3_pred_max": float(torch.max(x3_pred_values).detach()),
        "x3_norm_support_source": "preopt_x3_pred_norm_minmax",
        "x3_norm_support_min": float(x3_norm_support_min),
        "x3_norm_support_max": float(x3_norm_support_max),
        "x3_norm_support_manual_margin": 0.0,
        "loss_before_x3_refit": float(pre_total.detach()),
        "loss_after_x3_refit": float(post_total.detach()),
        "gain_before_refit": float(gain_before),
        "gain_after_refit": float(gain_after),
        "pre_refit_agu_samples": float(pre_grid_meta["total_samples"]),
        "post_refit_agu_samples": float(refit_grid_meta["total_samples"]),
    }
    _log_line(
        log,
        "x3 normalizer refit: "
        "source=preopt_rollout_x3_pred_full_window "
        f"prior_mean={meta['prior_x3_mean']:.6e} prior_scale={meta['prior_x3_scale']:.6e} "
        f"refit_mean={meta['refit_x3_mean']:.6e} refit_scale={meta['refit_x3_scale']:.6e} "
        f"x3_norm_support=[{meta['x3_norm_support_min']:.6e}, {meta['x3_norm_support_max']:.6e}] "
        "manual_margin=0",
    )
    _log_line(
        log,
        "x3 normalizer refit loss: "
        f"train={meta['loss_before_x3_refit']:.6e}->{meta['loss_after_x3_refit']:.6e} "
        f"gain={meta['gain_before_refit']:.6e}->{meta['gain_after_refit']:.6e}",
    )
    _log_line(
        log,
        "x3 normalizer refit AGU: "
        f"pre_samples={int(meta['pre_refit_agu_samples']) if np.isfinite(meta['pre_refit_agu_samples']) else 'NA'} "
        f"post_samples={int(meta['post_refit_agu_samples']) if np.isfinite(meta['post_refit_agu_samples']) else 'NA'} "
        f"x3_norm_neutral_range=[{meta['x3_norm_support_min']:.6e}, {meta['x3_norm_support_max']:.6e}] "
        "distribution=uniform",
    )
    _ = post_traj
    return refit_prepared, refit_observable_grid_inputs, meta, float(gain_after)


def _snapshot_split(
    *,
    force_module: KANForceModule,
    known_pars: dict[str, Any] | tuple[Any, ...],
    mech_module: LearnableMechModule,
    split: WindowSplit,
    window_mode: str | None,
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
        "title": _window_title(split.role, window_mode),
        "window_short": _role_short(split.role, window_mode),
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
                mech_module=mech_module,
                ode_true=tensors["ode_full"],
                x2dot_true=tensors["x2dot_full"],
                contact_mask=tensors["contact_full"],
                times=tensors["times_full"],
                ode_method=ode_method,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                loss_indices=loss_indices,
            )
            ode_true, x2dot_true, contact_mask, times = _select_from_full(tensors, part_name)
            if loss_indices is not None:
                traj = traj[:, loss_indices]
            x2dot_pred = x2dot_rhs_torch(traj, times, force_module, known_pars)
            rollout_states = traj.transpose(0, 1)
            fts_rollout_pred = force_module(rollout_states)
            nn_raw_rollout = force_module.raw_output(rollout_states)
            soft_mask_rollout = force_module.soft_mask(rollout_states)
            fts_unmasked_rollout = force_module.gain() * nn_raw_rollout
            out[part_name] = {
                "metrics": _metrics_row(total, parts),
                "times": times.detach().cpu().numpy(),
                "ode_true": ode_true.detach().cpu().numpy(),
                "traj_pred": traj.detach().cpu().numpy(),
                "x2dot_true": x2dot_true.detach().cpu().numpy(),
                "x2dot_pred": x2dot_pred.detach().cpu().numpy(),
                "contact_mask": contact_mask.detach().cpu().numpy(),
                "fts_rollout_pred": fts_rollout_pred.detach().cpu().numpy(),
                "fts_unmasked_rollout": fts_unmasked_rollout.detach().cpu().numpy(),
                "nn_raw_rollout": nn_raw_rollout.detach().cpu().numpy(),
                "soft_mask_rollout": soft_mask_rollout.detach().cpu().numpy(),
            }
        out["g_nn"] = float(force_module.gain().detach().cpu().item())
        out["soft_mask"] = force_module.soft_mask_summary()
        mech_pred = mech_module().detach().cpu().numpy()
        out["mech_pred"] = mech_pred
    return out


def _save_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(2):
        try:
            if tmp.exists():
                tmp.unlink()
            torch.save(payload, tmp)
            tmp.replace(path)
            return
        except Exception as exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt == 0:
                continue
            raise RuntimeError(f"failed to save torch checkpoint: path={path} tmp={tmp}") from exc


def _save_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(2):
        try:
            if tmp.exists():
                tmp.unlink()
            with tmp.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
            return
        except Exception as exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt == 0:
                continue
            raise RuntimeError(f"failed to save pickle checkpoint: path={path} tmp={tmp}") from exc


_AGU_WARMSTART_KEYS = (
    "initial_grid_support_source",
    "initial_grid_support",
    "initial_grid_support_meta",
    "initial_grid_support_x1_min",
    "initial_grid_support_x1_max",
    "initial_grid_support_x2_min",
    "initial_grid_support_x2_max",
    "initial_grid_support_x3_min",
    "initial_grid_support_x3_max",
    "rs_x3_normalizer_valid",
    "rs_x3_normalizer_source",
    "rs_x3_pred_mean",
    "rs_x3_pred_scale",
    "rs_x3_pred_min",
    "rs_x3_pred_max",
    "rs_x3_norm_support_min",
    "rs_x3_norm_support_max",
    "x3_refit_meta",
    "x3_agu_support_current",
)


def _copy_agu_warmstart_fields(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
    for key in _AGU_WARMSTART_KEYS:
        if key in record:
            out[key] = record[key]
        elif key in params:
            out[key] = params[key]
    return out


def _x3_init_from_warmstart_meta(
    warmstart_meta: dict[str, Any] | None,
) -> tuple[float, float, tuple[float, float], str] | None:
    if not isinstance(warmstart_meta, dict):
        return None

    current_support_raw = warmstart_meta.get("x3_agu_support_current")
    current_support = (
        (float(current_support_raw[0]), float(current_support_raw[1]))
        if _valid_x3_norm_support(current_support_raw)
        else None
    )

    refit = warmstart_meta.get("x3_refit_meta")
    if isinstance(refit, dict):
        try:
            mean = float(refit["refit_x3_mean"])
            scale = float(refit["refit_x3_scale"])
            support = (float(refit["x3_norm_support_min"]), float(refit["x3_norm_support_max"]))
        except Exception:
            mean = scale = float("nan")
            support = (float("nan"), float("nan"))
        source = str(refit.get("source", "prestage2_stage2light_x3_refit_meta"))
        if current_support is not None:
            support = current_support
            source = f"{source}__latest_x3_agu_support"
        if np.isfinite(mean) and np.isfinite(scale) and scale > 1.0e-30 and all(np.isfinite(support)) and support[1] > support[0]:
            return mean, scale, support, source

    valid = warmstart_meta.get("rs_x3_normalizer_valid", True)
    if isinstance(valid, str):
        valid = valid.strip().lower() not in ("0", "false", "no")
    if not bool(valid):
        return None
    try:
        mean = float(warmstart_meta["rs_x3_pred_mean"])
        scale = float(warmstart_meta["rs_x3_pred_scale"])
        support = (
            float(warmstart_meta["rs_x3_norm_support_min"]),
            float(warmstart_meta["rs_x3_norm_support_max"]),
        )
    except Exception:
        return None
    if not (np.isfinite(mean) and np.isfinite(scale) and scale > 1.0e-30):
        return None
    if not (np.isfinite(support[0]) and np.isfinite(support[1]) and support[1] > support[0]):
        return None
    source = str(warmstart_meta.get("rs_x3_normalizer_source", "stage1pluslight_rs_rollout_x3_pred_full_window"))
    if current_support is not None:
        support = current_support
        source = f"{source}__latest_x3_agu_support"
    return mean, scale, support, source


def _prepared_with_initial_x3_from_warmstart(
    prepared: PreparedData,
    warmstart_meta: dict[str, Any] | None,
) -> tuple[PreparedData, tuple[float, float] | None, str]:
    init = _x3_init_from_warmstart_meta(warmstart_meta)
    if init is None:
        return prepared, None, "x1_prior_dynamic_x3_init"
    x3_mean, x3_scale, x3_support, source = init
    mean = np.asarray(prepared.state_mean, dtype=float).copy()
    scale = np.asarray(prepared.state_scale, dtype=float).copy()
    mean[2] = float(x3_mean)
    scale[2] = float(x3_scale)
    return replace(prepared, state_mean=mean, state_scale=scale), x3_support, source


def _stage1_entry_x3_meta(warmstart_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(warmstart_meta, dict):
        return None
    meta = dict(warmstart_meta)
    # A prest2 record can carry its own x3_refit_meta, but formal st2l/prest2
    # should branch from the original st1pl endpoint, not inherit a previous
    # screening refit as a second-stage truth.
    if str(meta.get("source", "")).strip().lower() == "prest2":
        meta.pop("x3_refit_meta", None)
    return meta


def _x3_refit_meta_from_stage1_entry(
    *,
    prepared_before: PreparedData,
    prepared_after: PreparedData,
    split: WindowSplit,
    x3_support: tuple[float, float] | None,
    source: str,
    gain_before: float,
    gain_after: float,
    grid_meta: dict[str, Any],
) -> dict[str, Any]:
    prior_mean = np.asarray(prepared_before.state_mean, dtype=float).reshape(-1)
    prior_scale = np.asarray(prepared_before.state_scale, dtype=float).reshape(-1)
    mean = np.asarray(prepared_after.state_mean, dtype=float).reshape(-1)
    scale = np.asarray(prepared_after.state_scale, dtype=float).reshape(-1)
    support = x3_support if _valid_x3_norm_support(x3_support) else (-1.0, 1.0)
    return {
        "enabled": True,
        "source": str(source),
        "window_role": str(split.role),
        "window_label": str(split.label),
        "prior_x3_mean": float(prior_mean[2]),
        "prior_x3_scale": float(prior_scale[2]),
        "refit_x3_mean": float(mean[2]),
        "refit_x3_scale": float(scale[2]),
        "x3_pred_min": float("nan"),
        "x3_pred_max": float("nan"),
        "x3_norm_support_source": "stage1pluslight_saved_x3_pred_norm_support",
        "x3_norm_support_min": float(support[0]),
        "x3_norm_support_max": float(support[1]),
        "x3_norm_support_manual_margin": 0.0,
        "loss_before_x3_refit": float("nan"),
        "loss_after_x3_refit": float("nan"),
        "gain_before_refit": float(gain_before),
        "gain_after_refit": float(gain_after),
        "pre_refit_agu_samples": 0.0,
        "post_refit_agu_samples": float(grid_meta.get("total_samples", float("nan"))),
        "preopt_reproduce_rollout_skipped": True,
    }


def _x3_norm_support_from_refit_meta(meta: dict[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(meta, dict):
        return None
    try:
        support = (float(meta["x3_norm_support_min"]), float(meta["x3_norm_support_max"]))
    except Exception:
        return None
    if np.isfinite(support[0]) and np.isfinite(support[1]) and support[1] > support[0]:
        return support
    return None


def _warmstart_payload(candidate: Stage1WarmstartCandidate | Prestage2WarmstartCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    if isinstance(candidate, Prestage2WarmstartCandidate):
        payload = {
            "source": "prest2",
            "path": str(candidate.path),
            "candidate": int(candidate.candidate),
            "candidate_b": int(candidate.candidate),
            "original_rank": int(candidate.original_rank),
            "source_stage1_rank": int(candidate.original_rank),
            "source_mech_winner": int(candidate.source_mech_winner),
            "label": (
                f"candidate B {int(candidate.candidate)} from mech winner {int(candidate.source_mech_winner)} "
                f"(rank {int(candidate.original_rank)})"
            ),
            "trial_id": int(candidate.trial_id),
            "loss": float(candidate.ranking_loss),
            "seed": int(candidate.init_seed),
            "ks0": float(candidate.ks0),
            "cs0": float(candidate.cs0),
            "result_path": str(candidate.result_path),
            "st2l_start_policy": "restart_from_stage1_initial_state",
            "prest2_final_state_used": False,
        }
        payload.update(_copy_agu_warmstart_fields(candidate.record))
        return payload
    payload = {
        "source": "stage1",
        "path": str(candidate.path),
        "rank": int(candidate.rank),
        "mech_winner": int(candidate.mech_winner),
        "label": str(candidate.warmstart_label),
        "trial_id": int(candidate.trial_id),
        "loss": float(candidate.ranking_loss),
        "seed": int(candidate.init_seed),
        "ks0": float(candidate.ks0),
        "cs0": float(candidate.cs0),
    }
    payload.update(_copy_agu_warmstart_fields(candidate.record))
    return payload


def _best_from_history(history: list[dict[str, Any]]) -> tuple[int, float]:
    best_epoch = -1
    best_val = float("inf")
    for row in history:
        if not isinstance(row, dict):
            continue
        try:
            val = float(row.get("val_loss", float("inf")))
            epoch = int(float(row.get("epoch", -1)))
        except Exception:
            continue
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_epoch = epoch
    return best_epoch, best_val


def _resume_identity(
    *,
    shard_index: int,
    window_index: int,
    split: WindowSplit,
    warmstart_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "shard_index": int(shard_index),
        "window_index": int(window_index),
        "window_role": str(split.role),
        "window_label": str(split.label),
        "window_start_idx": int(split.start_idx),
        "window_stop_idx": int(split.stop_idx),
        "window_t_start": float(split.t_start),
        "window_t_stop": float(split.t_stop),
        "x3_normalizer_source": "preopt_rollout_x3_pred_full_window",
    }
    if isinstance(warmstart_meta, dict):
        source = str(warmstart_meta.get("source", "")).strip().lower()
        identity["warmstart_source"] = source
        if source == "stage1":
            if "rank" in warmstart_meta:
                identity["stage1_rank"] = int(warmstart_meta["rank"])
            if "mech_winner" in warmstart_meta:
                identity["stage1_mech_winner"] = int(warmstart_meta["mech_winner"])
            if "trial_id" in warmstart_meta:
                identity["stage1_trial_id"] = int(warmstart_meta["trial_id"])
        elif source == "prest2":
            if "candidate" in warmstart_meta:
                identity["prestage2_candidate"] = int(warmstart_meta["candidate"])
            if "trial_id" in warmstart_meta:
                identity["prestage2_trial_id"] = int(warmstart_meta["trial_id"])
    return identity


def _resume_metadata_matches(
    payload: dict[str, Any],
    *,
    split: WindowSplit,
    identity: dict[str, Any],
) -> tuple[bool, str]:
    def _float_match(old: Any, cur: Any) -> bool:
        try:
            old_f = float(old)
            cur_f = float(cur)
        except Exception:
            return False
        return abs(old_f - cur_f) <= max(1.0e-15, 1.0e-9 * max(abs(old_f), abs(cur_f), 1.0))

    def _identity_match(key: str, old_value: Any, cur_value: Any) -> bool:
        if key in ("window_t_start", "window_t_stop"):
            return _float_match(old_value, cur_value)
        if key in ("window_start_idx", "window_stop_idx", "shard_index", "window_index"):
            try:
                return int(old_value) == int(cur_value)
            except Exception:
                return False
        return str(old_value) == str(cur_value)

    window_meta = payload.get("window_meta", {})
    if not isinstance(window_meta, dict):
        return False, "window_meta missing in checkpoint"
    role = str(window_meta.get("role", "")).strip().lower()
    label = str(window_meta.get("label", "")).strip()
    if role != "" and role != str(split.role).strip().lower():
        return False, f"window role mismatch: checkpoint={role} current={split.role}"
    if label != "" and label != str(split.label):
        return False, f"window label mismatch: checkpoint={label} current={split.label}"
    for key, cur_value in (
        ("start_idx", int(split.start_idx)),
        ("stop_idx", int(split.stop_idx)),
        ("t_start", float(split.t_start)),
        ("t_stop", float(split.t_stop)),
    ):
        old_value = window_meta.get(key)
        if old_value is None:
            return False, f"window {key} missing in checkpoint metadata"
        if key in ("t_start", "t_stop"):
            if not _float_match(old_value, cur_value):
                return False, f"window {key} mismatch: checkpoint={old_value} current={cur_value}"
        else:
            try:
                if int(old_value) != int(cur_value):
                    return False, f"window {key} mismatch: checkpoint={old_value} current={cur_value}"
            except Exception:
                return False, f"window {key} invalid in checkpoint metadata: {old_value}"

    resume_identity = payload.get("resume_identity")
    warmstart = payload.get("warmstart")
    if isinstance(resume_identity, dict):
        for key, cur_value in identity.items():
            old_value = resume_identity.get(key)
            if old_value is None:
                return False, f"resume identity missing {key}"
            if not _identity_match(key, old_value, cur_value):
                return False, f"resume identity mismatch on {key}: checkpoint={old_value} current={cur_value}"
        return True, "identity match"

    if isinstance(warmstart, dict):
        source = str(warmstart.get("source", "")).strip().lower()
        cur_source = str(identity.get("warmstart_source", "")).strip().lower()
        if source != "" and cur_source != "" and source != cur_source:
            return False, f"warmstart source mismatch: checkpoint={source} current={cur_source}"
        if source == "stage1" and "stage1_rank" in identity and "rank" in warmstart:
            if int(warmstart["rank"]) != int(identity["stage1_rank"]):
                return False, f"stage1 rank mismatch: checkpoint={warmstart['rank']} current={identity['stage1_rank']}"
        if source == "stage1" and "stage1_trial_id" in identity and "trial_id" in warmstart:
            if int(warmstart["trial_id"]) != int(identity["stage1_trial_id"]):
                return False, (
                    f"stage1 trial_id mismatch: checkpoint={warmstart['trial_id']} "
                    f"current={identity['stage1_trial_id']}"
                )
        if source == "stage1" and "stage1_mech_winner" in identity and "mech_winner" in warmstart:
            if int(warmstart["mech_winner"]) != int(identity["stage1_mech_winner"]):
                return False, (
                    f"stage1 mech winner mismatch: checkpoint={warmstart['mech_winner']} "
                    f"current={identity['stage1_mech_winner']}"
                )
        if source == "prest2" and "prestage2_candidate" in identity and "candidate" in warmstart:
            if int(warmstart["candidate"]) != int(identity["prestage2_candidate"]):
                return False, (
                    f"prest2 candidate mismatch: checkpoint={warmstart['candidate']} "
                    f"current={identity['prestage2_candidate']}"
                )
        return True, "warmstart match"

    return True, "legacy checkpoint (no identity metadata)"


def _maybe_resume_from_running_checkpoint(
    *,
    cfg,
    running_checkpoint_path: Path,
    best_checkpoint_path: Path,
    result_path: Path,
    shard_index: int,
    window_index: int,
    split: WindowSplit,
    warmstart_meta: dict[str, Any] | None,
    model: torch.nn.Module,
    mech_module: LearnableMechModule,
    optimizer: torch.optim.Optimizer,
    log,
) -> dict[str, Any] | None:
    if not getattr(cfg, "resume_from_checkpoint", False):
        return None
    if result_path.is_file():
        if not running_checkpoint_path.is_file():
            if log is not None:
                _log_line(
                    log,
                    "resume skipped: completed result already exists and no running checkpoint is available "
                    f"| result={result_path}",
                )
            return None
        try:
            existing_checkpoint = torch.load(running_checkpoint_path, map_location="cpu", weights_only=False)
            if not isinstance(existing_checkpoint, dict):
                raise TypeError(f"unexpected checkpoint payload type: {type(existing_checkpoint)!r}")
            existing_history = existing_checkpoint.get("history", [])
            existing_epoch = max(
                int(existing_checkpoint.get("epoch", 0) or 0),
                len(existing_history) if isinstance(existing_history, list) else 0,
            )
        except Exception as err:
            if log is not None:
                _log_line(
                    log,
                    "resume skipped: completed result already exists and running checkpoint could not be read "
                    f"| result={result_path} checkpoint={running_checkpoint_path} error={err}",
                )
            return None
        if existing_epoch >= int(cfg.epochs):
            if log is not None:
                _log_line(
                    log,
                    "resume skipped: completed result already covers requested epoch budget "
                    f"| checkpoint_epoch={existing_epoch} target_epochs={cfg.epochs} result={result_path}",
                )
            return None
        if log is not None:
            _log_line(
                log,
                "resume allowed: completed result exists but checkpoint is shorter than requested epoch budget "
                f"| checkpoint_epoch={existing_epoch} target_epochs={cfg.epochs} checkpoint={running_checkpoint_path}",
            )
    if not running_checkpoint_path.is_file():
        return None

    payload = torch.load(running_checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        if log is not None:
            _log_line(log, f"resume skipped: checkpoint payload is not a dict | path={running_checkpoint_path}")
        return None

    identity = _resume_identity(
        shard_index=shard_index,
        window_index=window_index,
        split=split,
        warmstart_meta=warmstart_meta,
    )
    matches, reason = _resume_metadata_matches(payload, split=split, identity=identity)
    if not matches:
        if log is not None:
            _log_line(log, f"resume skipped: {reason} | checkpoint={running_checkpoint_path}")
        return None
    checkpoint_cfg = payload.get("config", {})
    checkpoint_mech_parameterization = _canonical_mech_parameterization(
        checkpoint_cfg.get("mech_parameterization") if isinstance(checkpoint_cfg, dict) else None,
        default="sigmoid_bounded",
    )
    current_mech_parameterization = _canonical_mech_parameterization(
        getattr(cfg, "mech_parameterization", "direct_unbounded"),
        default="direct_unbounded",
    )
    if checkpoint_mech_parameterization != current_mech_parameterization:
        if log is not None:
            _log_line(
                log,
                "resume skipped: mech parameterization mismatch "
                f"| checkpoint={checkpoint_mech_parameterization} current={current_mech_parameterization} "
                f"| checkpoint={running_checkpoint_path}",
            )
        return None

    state_dict = payload.get("state_dict")
    mech_state_dict = payload.get("mech_state_dict")
    if not isinstance(state_dict, dict) or not isinstance(mech_state_dict, dict):
        if log is not None:
            _log_line(log, f"resume skipped: checkpoint missing state dicts | checkpoint={running_checkpoint_path}")
        return None

    model.load_state_dict(state_dict)
    mech_module.load_state_dict(mech_state_dict)

    optimizer_phase = str(payload.get("optimizer_phase", "adam")).strip().lower() or "adam"
    restored_rng = restore_rng_state(payload.get("rng_state"))
    rng_status = ",".join(restored_rng) if restored_rng else (
        "legacy_checkpoint_no_rng_state" if "rng_state" not in payload else "rng_restore_failed"
    )
    optimizer_restored = False
    deferred_optimizer_state = None
    opt_state = payload.get("optimizer_state_dict")
    if isinstance(opt_state, dict):
        if optimizer_phase == "lbfgs":
            deferred_optimizer_state = opt_state
        else:
            optimizer.load_state_dict(opt_state)
            optimizer_restored = True

    history_raw = payload.get("history", [])
    history = [row for row in history_raw if isinstance(row, dict)]
    start_epoch = max(int(payload.get("epoch", 0)), len(history))
    lr = float(payload.get("lr", float(cfg.lr)))
    grad_ema = float(payload.get("grad_ema", float(cfg.lr_target_init)))
    grad_target = float(payload.get("grad_target", float(cfg.lr_target_init)))
    recent_losses_raw = payload.get("recent_losses", [])
    recent_losses = [float(x) for x in recent_losses_raw] if isinstance(recent_losses_raw, list) else []
    if getattr(cfg, "plateau_early_stop", False) and not recent_losses:
        recent_losses = [float(row["train_loss"]) for row in history[-int(cfg.plateau_window):] if "train_loss" in row]
    train_loss_last = float(history[-1]["train_loss"]) if history else float("inf")

    best_payload = None
    best_epoch = int(payload.get("best_epoch", -1))
    best_val = float(payload.get("best_val_loss", float("inf")))
    final_val_epoch = int(payload.get("final_val_epoch", -1))
    final_val_loss = float(payload.get("final_val_loss", float("nan")))
    if best_checkpoint_path.is_file():
        best_payload = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(best_payload, dict):
            best_epoch = int(best_payload.get("epoch", best_epoch))
            best_hist = best_payload.get("history", [])
            if isinstance(best_hist, list) and best_hist:
                try:
                    best_val = float(best_hist[-1].get("val_loss", best_val))
                except Exception:
                    pass
    if not np.isfinite(best_val) or best_epoch < 0:
        best_epoch_hist, best_val_hist = _best_from_history(history)
        if best_epoch < 0:
            best_epoch = best_epoch_hist
        if not np.isfinite(best_val):
            best_val = best_val_hist

    if log is not None and _validation_final_only(cfg):
        _log_line(
            log,
            "resume from checkpoint: "
            f"epoch={start_epoch} | {reason} | optimizer={'restored' if optimizer_restored else 'fresh'} | "
            f"checkpoint_optimizer_phase={optimizer_phase} | "
            f"rng_restored={rng_status} | "
            f"final_val_epoch={final_val_epoch} final_val={final_val_loss:.6e} | checkpoint={running_checkpoint_path}",
        )
    elif log is not None:
        _log_line(
            log,
            "resume from checkpoint: "
            f"epoch={start_epoch} | {reason} | optimizer={'restored' if optimizer_restored else 'fresh'} | "
            f"checkpoint_optimizer_phase={optimizer_phase} | "
            f"rng_restored={rng_status} | "
            f"best_epoch={best_epoch} best_val={best_val:.6e} | checkpoint={running_checkpoint_path}",
        )

    return {
        "start_epoch": start_epoch,
        "history": history,
        "lr": lr,
        "grad_ema": grad_ema,
        "grad_target": grad_target,
        "recent_losses": recent_losses,
        "train_loss_last": train_loss_last,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_val_epoch": final_val_epoch,
        "final_val_loss": final_val_loss,
        "best_payload": best_payload if isinstance(best_payload, dict) else None,
        "optimizer_phase": optimizer_phase,
        "deferred_optimizer_state_dict": deferred_optimizer_state,
        "state_mean": payload.get("state_mean"),
        "state_scale": payload.get("state_scale"),
        "initial_grid_support": payload.get("initial_grid_support"),
        "initial_grid_support_meta": payload.get("initial_grid_support_meta"),
        "x3_refit_meta": payload.get("x3_refit_meta"),
        "x3_agu_support_current": payload.get("x3_agu_support_current"),
        "x3_drift_epoch0_x3": payload.get("x3_drift_epoch0_x3"),
    }


def _maybe_load_warmstart(
    *,
    cfg,
    log,
) -> Stage1WarmstartCandidate | Prestage2WarmstartCandidate | None:
    if not getattr(cfg, "warmstart_from_stage1", False):
        return None
    source = str(getattr(cfg, "warmstart_source", "stage1")).strip().lower()
    if source == "prest2":
        best_path = cfg.prestage2_input_path
    else:
        source = "stage1"
        best_path = cfg.stage1_input_path
    record_path = Path(getattr(cfg, "stage1_warmstart_record_path", Path()))
    if source == "stage1" and str(record_path).strip() != "" and record_path.is_file():
        candidate = load_stage1_warmstart_record(record_path)
        if log is not None:
            label = candidate.warmstart_label or (
                f"mech winner {candidate.mech_winner}" if candidate.mech_winner > 0 else f"rank {candidate.rank}"
            )
            _log_line(
                log,
                "warmstart from stage1pluslight extracted record: "
                f"record={record_path} "
                f"source_path={candidate.path} {label} "
                f"trial={candidate.trial_id} "
                f"loss={candidate.ranking_loss:.6e} "
                f"seed={candidate.init_seed} "
                f"ks0={candidate.ks0:.6e} cs0={candidate.cs0:.6e}",
            )
        return candidate
    if not best_path.is_file():
        if getattr(cfg, "warmstart_fallback_random", False):
            if log is not None:
                _log_line(
                    log,
                    f"warmstart from {source}: missing input, fallback to random init | "
                    f"path={best_path}",
                )
            return None
        raise FileNotFoundError(f"Missing {source} warmstart file: {best_path}")
    if source == "prest2":
        candidate = select_prestage2_candidate(best_path, candidate=int(cfg.prestage2_input_candidate))
        if log is not None:
            _log_line(
                log,
                "select prest2 trial-run candidate; formal stage2light starts from stage1 initial state: "
                f"path={best_path} candidate B={candidate.candidate} "
                f"source_mech_winner={candidate.source_mech_winner} "
                f"(rank={candidate.original_rank}) "
                f"trial={candidate.trial_id} "
                f"loss={candidate.ranking_loss:.6e} "
                f"seed={candidate.init_seed} "
                f"ks0={candidate.ks0:.6e} cs0={candidate.cs0:.6e} "
                f"prest2_result_for_audit={candidate.result_path}",
        )
        return candidate

    if int(getattr(cfg, "stage1_input_trial_id", 0)) > 0:
        candidate = select_stage1_candidate_by_trial_id(
            best_path,
            trial_id=int(cfg.stage1_input_trial_id),
            mech_winner=int(getattr(cfg, "stage1_input_mech_winner", 0)),
        )
    else:
        candidate = select_stage1_candidate(best_path, rank=int(cfg.stage1_input_rank))
    if log is not None:
        label = candidate.warmstart_label or (
            f"mech winner {candidate.mech_winner}" if candidate.mech_winner > 0 else f"rank {candidate.rank}"
        )
        _log_line(
            log,
            "warmstart from stage1pluslight: "
            f"path={best_path} {label} "
            f"trial={candidate.trial_id} "
            f"loss={candidate.ranking_loss:.6e} "
            f"seed={candidate.init_seed} "
            f"ks0={candidate.ks0:.6e} cs0={candidate.cs0:.6e}",
        )
    return candidate


def _log_prestage2_selection_policy(
    *,
    candidate: Stage1WarmstartCandidate | Prestage2WarmstartCandidate | None,
    log,
) -> bool:
    if not isinstance(candidate, Prestage2WarmstartCandidate):
        return False
    if log is not None:
        _log_line(
            log,
            "prest2 candidate selected for screening identity only; "
            "stage2light restarts from the original stage1 initial state: "
            f"candidate B={candidate.candidate} source_mech_winner={candidate.source_mech_winner} "
            f"(rank={candidate.original_rank}) trial={candidate.trial_id} "
            f"seed={candidate.init_seed} ks0={candidate.ks0:.6e} cs0={candidate.cs0:.6e}",
        )
    return True


def _make_viz_payload(payload: dict[str, Any]) -> dict[str, Any]:
    validation_eval_mode = str(payload.get("validation_eval_mode", "")).strip().lower()
    final_only_validation = validation_eval_mode == "final_only"
    final_viz = None
    final_val_epoch = payload.get("final_val_epoch")
    final_val_loss = payload.get("final_val_loss")
    if final_only_validation:
        final_viz = {
            "epoch": final_val_epoch,
            "final_snapshot": payload.get("final_snapshot"),
        }
    best = payload.get("best")
    best_viz = None
    best_epoch = payload.get("best_epoch")
    best_val_loss = payload.get("best_val_loss")
    if final_only_validation:
        best_viz = None
        best_epoch = None
        best_val_loss = None
    elif isinstance(best, dict):
        best_viz = {key: value for key, value in best.items() if key not in ("state_dict", "mech_state_dict")}
    elif isinstance(payload.get("best_snapshot"), dict):
        best_epoch = payload.get("epoch", best_epoch)
        history = payload.get("history", [])
        if best_val_loss is None and isinstance(history, list) and history:
            last_row = history[-1]
            if isinstance(last_row, dict):
                best_val_loss = last_row.get("val_loss", best_val_loss)
        best_viz = {
            "epoch": best_epoch,
            "best_snapshot": payload.get("best_snapshot"),
        }
    viz_payload = {
        "final": final_viz,
        "history": payload.get("history", []),
        "state_mean": payload.get("state_mean"),
        "state_scale": payload.get("state_scale"),
        "initial_grid_support": payload.get("initial_grid_support"),
        "initial_grid_support_meta": payload.get("initial_grid_support_meta"),
        "x3_refit_meta": payload.get("x3_refit_meta"),
        "x3_agu_support_current": payload.get("x3_agu_support_current"),
        "known_pars": payload.get("known_pars"),
        "warmstart": payload.get("warmstart"),
        "stage1_warmstart": payload.get("stage1_warmstart"),
        "prestage2_warmstart": payload.get("prestage2_warmstart"),
        "warmstart_source": payload.get("warmstart_source"),
        "window_meta": payload.get("window_meta"),
        "final_snapshot": payload.get("final_snapshot"),
        "config": payload.get("config"),
        "validation_eval_mode": payload.get("validation_eval_mode"),
        "final_val_epoch": final_val_epoch,
        "final_val_loss": final_val_loss,
    }
    if not final_only_validation:
        viz_payload.update(
            {
                "best": best_viz,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }
        )
    return viz_payload


def run_stage2light_shard(cfg=None, *, shard_index: int | None = None, log_path: Path | None = None) -> dict[str, object]:
    cfg = default_config() if cfg is None else cfg
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
    tensors = _window_to_torch(split, dtype=dtype, device=cfg.device)
    observable_grid_inputs: torch.Tensor | None = None
    initial_grid_support: np.ndarray | None = None
    initial_grid_support_meta: dict[str, Any] | None = None

    train_states = torch.as_tensor(split.ode_train.T, dtype=dtype, device=cfg.device)
    train_gain_force_reference = torch.as_tensor(split.train_gain_force_reference, dtype=dtype, device=cfg.device)
    role_short = _role_short(split.role, cfg.window_mode)
    final_only_validation = _validation_final_only(cfg)
    best_or_final_stem = "stage2light_final" if final_only_validation else "stage2light_best"
    history_path = cfg.result_dir / f"stage2light_history_p{shard_index}.json"
    checkpoint_path = cfg.checkpoint_dir / f"{best_or_final_stem}_p{shard_index}.pt"
    checkpoint_viz_path = cfg.checkpoint_dir / f"{best_or_final_stem}_p{shard_index}.viz.pkl"
    running_checkpoint_path = cfg.checkpoint_dir / f"stage2light_checkpoint_p{shard_index}.pt"
    result_path = cfg.result_dir / f"stage2light_result_p{shard_index}.pt"
    viz_result_path = cfg.result_dir / f"stage2light_result_p{shard_index}.viz.pkl"
    log_path = cfg.shard_log_dir / f"log2_05_step2a_stage2light_local_p{shard_index}.txt" if log_path is None else Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    best_payload: dict[str, Any] | None = None
    final_val_loss = float("nan")
    final_val_epoch = -1
    grad_ema = float(cfg.lr_target_init)
    grad_target = float(cfg.lr_target_init)
    recent_losses: list[float] = []
    failure_reason = ""
    failure_epoch = 0
    stop_reason = ""
    stop_kind = ""
    stop_epoch = 0
    train_loss_last = float("inf")
    cached_grid_traj: torch.Tensor | None = None
    start_epoch = 0
    good_enough_reached = False
    resume_optimizer_phase = ""
    deferred_optimizer_state_dict = None
    x3_refit_meta: dict[str, Any] | None = None
    x3_agu_support_current: tuple[float, float] | None = None
    x3_drift_epoch0_x3: torch.Tensor | None = None

    step_controller = str(getattr(cfg, "step_controller", "armijo_backtracking")).strip().lower().replace("-", "_")
    if step_controller not in ("armijo_backtracking", "legacy_guard", "off"):
        step_controller = "off"
    lr_adapt_active = bool(cfg.lr_adapt) and step_controller != "armijo_backtracking"
    event_log_dir = cfg.log_dir / "optimizer_events"
    event_log_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = event_log_dir / f"stage2light_optimizer_events_p{shard_index}.jsonl"
    log_mode = "a" if log_path.exists() else "w"
    with log_path.open(log_mode, encoding="utf-8") as log, event_log_path.open("a", encoding="utf-8") as event_log:
        if log_mode == "a":
            log.write("\n")
            log.flush()
            _log_line(log, "----- append existing shard log; preserve previous session -----")
        warmstart = _maybe_load_warmstart(cfg=cfg, log=log)
        warmstart_meta = _warmstart_payload(warmstart)
        stage1_warmstart_meta = warmstart_meta if isinstance(warmstart, Stage1WarmstartCandidate) else None
        prestage2_warmstart_meta = warmstart_meta if isinstance(warmstart, Prestage2WarmstartCandidate) else None
        stage1_entry_meta = _stage1_entry_x3_meta(warmstart_meta)
        prepared_before_stage1_entry = prepared
        prepared, x3_norm_support_for_init, x3_init_source = _prepared_with_initial_x3_from_warmstart(
            prepared,
            stage1_entry_meta,
        )
        x3_agu_support_current = x3_norm_support_for_init if _valid_x3_norm_support(x3_norm_support_for_init) else None
        observable_grid_inputs = _observable_grid_inputs_from_ode(
            ode_train=tensors["ode_train"],
            state_mean=prepared.state_mean,
            state_scale=prepared.state_scale,
            x3_norm_support=x3_agu_support_current,
        )
        initial_grid_support = initial_grid_support_from_raw_inputs(
            observable_grid_inputs,
            prepared.state_mean,
            prepared.state_scale,
        )
        initial_grid_support_meta = initial_grid_support_to_meta(
            initial_grid_support,
            source=f"stage2light_initial_observed_x1x2_dynamic_x3__{x3_init_source}",
        )
        model_seed = int(warmstart.init_seed) if warmstart is not None else int(cfg.seed)
        ks_init = float(warmstart.ks0) if warmstart is not None else _geometric_midpoint(cfg.ks_lo, cfg.ks_hi)
        cs_init = float(warmstart.cs0) if warmstart is not None else _geometric_midpoint(cfg.cs_lo, cfg.cs_hi)
        known_fields = _afm05_known_fields(prepared.known_pars)
        model = KANForceModule(
            pykan_root=cfg.pykan_root,
            state_mean=prepared.state_mean,
            state_scale=prepared.state_scale,
            seed=model_seed,
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
            dist=float(known_fields["Z"]),
            a0=float(known_fields["a0"]),
            gnn_learnable=cfg.gnn_learnable,
            soft_mask_enabled=cfg.soft_mask_enabled,
            soft_mask_trainable=cfg.soft_mask_trainable,
            soft_mask_s0_a0=cfg.soft_mask_s0_a0,
            soft_mask_s0_min_a0=cfg.soft_mask_s0_min_a0,
            soft_mask_s0_max_a0=cfg.soft_mask_s0_max_a0,
            soft_mask_alpha_a0=cfg.soft_mask_alpha_a0,
            soft_mask_alpha_min_a0=cfg.soft_mask_alpha_min_a0,
            soft_mask_alpha_max_a0=cfg.soft_mask_alpha_max_a0,
            device=cfg.device,
            dtype=dtype,
        ).to(cfg.device)
        mech_module = LearnableMechModule(
            ks_init=ks_init,
            cs_init=cs_init,
            ks_bounds=(cfg.ks_lo, cfg.ks_hi),
            cs_bounds=(cfg.cs_lo, cfg.cs_hi),
            dtype=dtype,
            device=cfg.device,
            parameterization=cfg.mech_parameterization,
        ).to(cfg.device)
        prest2_candidate_selected = _log_prestage2_selection_policy(
            candidate=warmstart,
            log=log,
        )
        _ = prest2_candidate_selected
        optimizer, optimizer_name = _make_optimizer(cfg, model, mech_module)
        lr = float(cfg.lr)
        _set_optimizer_lr(optimizer, lr)
        resume_state = _maybe_resume_from_running_checkpoint(
            cfg=cfg,
            running_checkpoint_path=running_checkpoint_path,
            best_checkpoint_path=checkpoint_path,
            result_path=result_path,
            shard_index=shard_index,
            window_index=window_index,
            split=split,
            warmstart_meta=warmstart_meta,
            model=model,
            mech_module=mech_module,
            optimizer=optimizer,
            log=log,
        )
        if resume_state is not None:
            start_epoch = int(resume_state["start_epoch"])
            history = list(resume_state["history"])
            best_val = float(resume_state["best_val_loss"])
            best_epoch = int(resume_state["best_epoch"])
            final_val_loss = float(resume_state.get("final_val_loss", final_val_loss))
            final_val_epoch = int(resume_state.get("final_val_epoch", final_val_epoch))
            best_payload = resume_state["best_payload"]
            grad_ema = float(resume_state["grad_ema"])
            grad_target = float(resume_state["grad_target"])
            recent_losses = list(resume_state["recent_losses"])
            train_loss_last = float(resume_state["train_loss_last"])
            lr = float(resume_state["lr"])
            resume_optimizer_phase = str(resume_state.get("optimizer_phase", "")).strip().lower()
            deferred_optimizer_state_dict = resume_state.get("deferred_optimizer_state_dict")
            _set_optimizer_lr(optimizer, lr)
            init_gain = float(model.gain().detach().cpu()) if hasattr(model, "gain") else float("nan")
            x3_refit_meta = resume_state.get("x3_refit_meta")
            x3_drift_epoch0_raw = resume_state.get("x3_drift_epoch0_x3")
            if x3_drift_epoch0_raw is not None:
                try:
                    x3_drift_epoch0_x3 = torch.as_tensor(
                        x3_drift_epoch0_raw,
                        dtype=dtype,
                        device=cfg.device,
                    ).detach().clone()
                except Exception:
                    x3_drift_epoch0_x3 = None
            initial_grid_support_meta = resume_state.get("initial_grid_support_meta") or initial_grid_support_meta
            resume_x3_support = resume_state.get("x3_agu_support_current")
            if _valid_x3_norm_support(resume_x3_support):
                x3_agu_support_current = (float(resume_x3_support[0]), float(resume_x3_support[1]))
            resume_mean = resume_state.get("state_mean")
            resume_scale = resume_state.get("state_scale")
            if resume_mean is not None and resume_scale is not None:
                resume_mean_arr = np.asarray(resume_mean, dtype=float).reshape(-1)
                resume_scale_arr = np.asarray(resume_scale, dtype=float).reshape(-1)
                if resume_mean_arr.size == prepared.state_mean.size and resume_scale_arr.size == prepared.state_scale.size:
                    prepared = replace(prepared, state_mean=resume_mean_arr, state_scale=resume_scale_arr)
                    if not _valid_x3_norm_support(x3_agu_support_current):
                        x3_agu_support_current = _x3_norm_support_from_refit_meta(x3_refit_meta)
                    observable_grid_inputs = _observable_grid_inputs_from_ode(
                        ode_train=tensors["ode_train"],
                        state_mean=prepared.state_mean,
                        state_scale=prepared.state_scale,
                        x3_norm_support=x3_agu_support_current,
                    )
        else:
            if isinstance(stage1_entry_meta, dict) and str(x3_init_source) != "x1_prior_dynamic_x3_init":
                try:
                    with torch.no_grad():
                        gain_before = float(model.initialize_gain_from_reference(train_states, train_gain_force_reference))
                        grid_meta = {"total_samples": float("nan")}
                        if cfg.adaptive_grid_enabled:
                            grid_inputs, grid_meta = _make_grid_update_inputs(base_inputs=observable_grid_inputs)
                            model.update_grid_from_normalized_inputs(grid_inputs)
                        gain_after = float(model.initialize_gain_from_reference(train_states, train_gain_force_reference))
                    init_gain = float(gain_after)
                    x3_refit_meta = _x3_refit_meta_from_stage1_entry(
                        prepared_before=prepared_before_stage1_entry,
                        prepared_after=prepared,
                        split=split,
                        x3_support=x3_agu_support_current,
                        source=x3_init_source,
                        gain_before=gain_before,
                        gain_after=gain_after,
                        grid_meta=grid_meta,
                    )
                    _log_line(
                        log,
                        "x3 normalizer entry: "
                        f"source={x3_init_source} "
                        f"mean={float(prepared.state_mean[2]):.6e} scale={float(prepared.state_scale[2]):.6e} "
                        f"x3_norm_support=[{float(x3_agu_support_current[0]) if x3_agu_support_current else float('nan'):.6e}, "
                        f"{float(x3_agu_support_current[1]) if x3_agu_support_current else float('nan'):.6e}] "
                        "preopt_reproduce_rollout=skipped",
                    )
                    _log_fingerprint(
                        log,
                        event_log,
                        epoch=0,
                        phase="stage1_entry",
                        stage="after_stage1_x3_entry",
                        model=model,
                        mech_module=mech_module,
                        optimizer=optimizer,
                        grid_inputs=observable_grid_inputs,
                        extra={
                            "refit_x3_mean": float(x3_refit_meta.get("refit_x3_mean", float("nan"))),
                            "refit_x3_scale": float(x3_refit_meta.get("refit_x3_scale", float("nan"))),
                            "x3_norm_support_min": float(x3_refit_meta.get("x3_norm_support_min", float("nan"))),
                            "x3_norm_support_max": float(x3_refit_meta.get("x3_norm_support_max", float("nan"))),
                            "preopt_reproduce_rollout_skipped": True,
                        },
                    )
                except Exception as err:
                    _log_line(log, f"stage1 x3 entry failed: {type(err).__name__}: {err}")
                    raise RuntimeError(f"stage1_x3_entry_failed:{err}") from err
            else:
                try:
                    prepared, observable_grid_inputs, x3_refit_meta, init_gain = _refit_x3_normalizer_from_preopt_rollout(
                        log=log,
                        cfg=cfg,
                        prepared=prepared,
                        split=split,
                        tensors=tensors,
                        train_states=train_states,
                        train_gain_force_reference=train_gain_force_reference,
                        model=model,
                        mech_module=mech_module,
                    )
                    x3_agu_support_current = _x3_norm_support_from_refit_meta(x3_refit_meta)
                    _log_fingerprint(
                        log,
                        event_log,
                        epoch=0,
                        phase="preopt",
                        stage="after_x3_refit",
                        model=model,
                        mech_module=mech_module,
                        optimizer=optimizer,
                        grid_inputs=observable_grid_inputs,
                        extra={
                            "refit_x3_mean": float(x3_refit_meta.get("refit_x3_mean", float("nan"))),
                            "refit_x3_scale": float(x3_refit_meta.get("refit_x3_scale", float("nan"))),
                            "x3_norm_support_min": float(x3_refit_meta.get("x3_norm_support_min", float("nan"))),
                            "x3_norm_support_max": float(x3_refit_meta.get("x3_norm_support_max", float("nan"))),
                        },
                    )
                except Exception as err:
                    _log_line(log, f"x3 normalizer refit failed: {type(err).__name__}: {err}")
                    raise RuntimeError(f"x3_normalizer_refit_failed:{err}") from err

        if x3_drift_epoch0_x3 is None:
            if start_epoch > 0:
                _log_line(
                    log,
                    "x3 drift baseline missing in resume checkpoint; initializing from current resumed state",
                )
            with torch.no_grad():
                _baseline_total, _baseline_parts, baseline_traj = evaluate_split(
                    force_module=model,
                    known_pars=prepared.known_pars,
                    mech_module=mech_module,
                    ode_true=tensors["ode_full"],
                    x2dot_true=tensors["x2dot_full"],
                    contact_mask=tensors["contact_full"],
                    times=tensors["times_full"],
                    ode_method=cfg.ode_method,
                    ode_rtol=cfg.ode_rtol,
                    ode_atol=cfg.ode_atol,
                    loss_indices=tensors["train_idx"],
                )
                x3_drift_epoch0_x3 = baseline_traj[2, :].detach().clone()
            _log_line(
                log,
                f"x3 drift baseline initialized | source=post_x3_refit_pre_epoch1 | samples={int(x3_drift_epoch0_x3.numel())}",
            )

        _log_line(
            log,
            "STAGE2LIGHT windowed horizon: "
            f"full_points={prepared.full_points} | window=[{split.start_idx}, {split.stop_idx}] "
            f"len={split.stop_idx - split.start_idx + 1} | tspan=[{split.t_start:.6e}, {split.t_stop:.6e}]",
        )
        _log_line(
            log,
            "STAGE2LIGHT window meta: "
            f"label={split.label} | role={split.role} | "
            f"t_us=[{split.t_start * 1.0e6:.9f}, {split.t_stop * 1.0e6:.9f}]",
        )
        _log_line(log, "=== AFM05 STAGE2LIGHT ===")
        _log_line(
            log,
            "Switches: "
            f"adaptive_grid={str(cfg.adaptive_grid_enabled).upper()} | "
            f"warmstart={'ON' if cfg.warmstart_from_stage1 else 'OFF'} | "
            f"warmstart_source={getattr(cfg, 'warmstart_source', 'stage1')} | "
            f"warmstart_fallback_random={'ON' if cfg.warmstart_fallback_random else 'OFF'} | "
            "random_init=ON | "
            f"symbolic={str(cfg.symbolic_enabled).upper()} | auto_save={str(cfg.auto_save).upper()}",
        )
        solver_name = "torchdiffeq odeint_adjoint" if getattr(cfg, "use_adjoint", False) else "torchdiffeq odeint"
        _log_line(log, f"Solver: {solver_name} | method={cfg.ode_method} | differentiable=ON")
        _log_line(
            log,
            f"STAGE2LIGHT shard assignment: {shard_index}/{cfg.shard_count} | local_windows=1 | "
            f"window_index={window_index} | role={role_short}",
        )
        _log_line(
            log,
            "KAN config: "
            f"width={list(cfg.width)} grid={cfg.grid} k={cfg.spline_k} base={cfg.base_fun} "
            f"noise_scale={cfg.noise_scale}",
        )
        if isinstance(initial_grid_support_meta, dict):
            _log_line(
                log,
                "KAN initial grid support: "
                f"source={initial_grid_support_meta.get('source', '')} "
                f"x1=[{float(initial_grid_support_meta['x1_min']):.6e}, {float(initial_grid_support_meta['x1_max']):.6e}] "
                f"x2=[{float(initial_grid_support_meta['x2_min']):.6e}, {float(initial_grid_support_meta['x2_max']):.6e}] "
                f"x3=[{float(initial_grid_support_meta['x3_min']):.6e}, {float(initial_grid_support_meta['x3_max']):.6e}]",
            )
        if _valid_x3_norm_support(x3_agu_support_current):
            _log_line(
                log,
                "x3 AGU adaptive support init: "
                f"current=[{float(x3_agu_support_current[0]):.6e}, {float(x3_agu_support_current[1]):.6e}] "
                "density=uniform | normalizer=fixed_after_refit",
            )
        _log_line(
            log,
            "force chain: target=(soft_mask * g_nn * nn_raw) | legacy contact multiplier removed",
        )
        _log_line(
            log,
            "mech init: "
            f"ks0={ks_init:.6e} cs0={cs_init:.6e} "
            f"| parameterization={cfg.mech_parameterization}",
        )
        _log_line(
            log,
            f"optimizer={optimizer_name.upper()} | amsgrad={'ON' if optimizer_name == 'amsgrad' else 'OFF'} | "
            f"lr={lr:.2e} weight_decay={cfg.weight_decay:.2e}",
        )
        _log_line(
            log,
            "training horizon: "
            f"total_epochs={cfg.epochs} adam_epochs={cfg.adam_epochs} lbfgs_epochs={cfg.lbfgs_steps}",
        )
        _log_line(
            log,
            "validation evaluation: "
            f"mode={cfg.val_eval_mode} "
            f"| checkpoint_semantics={'final' if final_only_validation else 'best_val'}",
        )
        _log_line(
            log,
            "optimizer controls: "
            f"step_controller={step_controller} "
            f"lr_adapt={'UP_ONLY' if (lr_adapt_active and cfg.lr_adapt_up_only) else ('ON' if lr_adapt_active else 'OFF')} "
            f"legacy_step_guard_switch={'ON' if (step_controller == 'legacy_guard' and cfg.step_guard_enabled) else 'OFF'} "
            f"step_retry_reset={'INITIAL_LR' if cfg.step_retry_reset_lr_each_epoch else 'OFF'} "
            f"plateau_early_stop={'ON' if cfg.plateau_early_stop else 'OFF'} "
            f"recent_val_early_stop={'ON' if cfg.recent_val_early_stop else 'OFF'} "
            f"good_enough={cfg.good_enough_loss:.1e}",
        )
        step_retry_label = "inf" if cfg.step_retry_max < 0 else str(cfg.step_retry_max)
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
                f"retry(epoch={cfg.epoch_retry_max}, step={step_retry_label}) "
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
            f"recent_val_min_epochs={cfg.recent_val_window} "
            f"recent_val_compare_gap={cfg.recent_val_compare_gap} "
            f"recent_val_rel_current_frac={cfg.recent_val_rel_current_frac:.4f}",
        )
        _log_line(log, "optimizer grouping: single_group=ON | group_adapt=OFF | mech_joint=ON")
        _log_line(log, f"g_nn init={init_gain:.6e} | learnable={'ON' if cfg.gnn_learnable else 'OFF'}")
        soft_mask_meta = model.soft_mask_summary()
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

        for epoch in range(start_epoch, cfg.adam_epochs):
            if cfg.step_retry_reset_lr_each_epoch:
                lr = float(cfg.lr)
                _set_optimizer_lr(optimizer, lr)
            epoch_wall_start = perf_counter()
            grid_update_sec_epoch = 0.0
            grid_update_runs_epoch = 0
            grid_update_meta_epoch = {
                "base_samples": float("nan"),
                "extra_samples": 0.0,
                "total_samples": float("nan"),
            }
            epoch_start_state = copy.deepcopy(model.state_dict())
            epoch_start_mech_state = copy.deepcopy(mech_module.state_dict())
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
            grad_norm_nn = float("nan")
            grad_norm_mech = float("nan")
            step_retry_count = 0
            step_accept_reason = ""
            step_effective_lr = float(lr)
            step_next_lr = float(lr)
            epoch_fail_reason = ""

            while epoch_attempt < epoch_attempt_max:
                epoch_attempt += 1
                model.load_state_dict(copy.deepcopy(epoch_start_state))
                mech_module.load_state_dict(copy.deepcopy(epoch_start_mech_state))
                optimizer.load_state_dict(copy.deepcopy(epoch_start_opt_state))
                _set_optimizer_lr(optimizer, lr)
                model.train()
                _log_fingerprint(
                    log,
                    event_log,
                    epoch=epoch + 1,
                    phase="adam",
                    stage="epoch_start",
                    model=model,
                    mech_module=mech_module,
                    optimizer=optimizer,
                    grid_inputs=observable_grid_inputs,
                    extra={"attempt": int(epoch_attempt), "lr": float(lr)},
                )

                if cfg.adaptive_grid_enabled and _grid_update_due(epoch, cfg.grid_update_num, cfg.start_grid_update_step, cfg.stop_grid_update_step):
                    try:
                        grid_update_t0 = perf_counter()
                        with torch.no_grad():
                            grid_source = "normalized_observed_x1x2_pred_informed_uniform_x3_axis"
                            grid_inputs, grid_meta = _make_grid_update_inputs(
                                base_inputs=observable_grid_inputs,
                            )
                            _log_fingerprint(
                                log,
                                event_log,
                                epoch=epoch + 1,
                                phase="adam",
                                stage="before_grid_update",
                                model=model,
                                mech_module=mech_module,
                                optimizer=optimizer,
                                grid_inputs=grid_inputs,
                                extra={
                                    "attempt": int(epoch_attempt),
                                    "grid_source": grid_source,
                                    "grid_base_samples": float(grid_meta.get("base_samples", float("nan"))),
                                    "grid_extra_samples": float(grid_meta.get("extra_samples", float("nan"))),
                                    "grid_total_samples": float(grid_meta.get("total_samples", float("nan"))),
                                },
                            )
                            model.update_grid_from_normalized_inputs(grid_inputs)
                            _log_fingerprint(
                                log,
                                event_log,
                                epoch=epoch + 1,
                                phase="adam",
                                stage="after_grid_update",
                                model=model,
                                mech_module=mech_module,
                                optimizer=optimizer,
                                grid_inputs=grid_inputs,
                                extra={
                                    "attempt": int(epoch_attempt),
                                    "grid_source": grid_source,
                                    "grid_base_samples": float(grid_meta.get("base_samples", float("nan"))),
                                    "grid_extra_samples": float(grid_meta.get("extra_samples", float("nan"))),
                                    "grid_total_samples": float(grid_meta.get("total_samples", float("nan"))),
                                },
                            )
                        grid_update_dt = perf_counter() - grid_update_t0
                        grid_update_sec_epoch += float(grid_update_dt)
                        grid_update_runs_epoch += 1
                        grid_update_meta_epoch = dict(grid_meta)
                        if _valid_x3_norm_support(x3_agu_support_current):
                            x3_support_label = (
                                f"[{float(x3_agu_support_current[0]):.6e}, "
                                f"{float(x3_agu_support_current[1]):.6e}]"
                            )
                        else:
                            x3_support_label = "default"
                        _log_line(
                            log,
                            "grid update | "
                            f"epoch={epoch + 1} | source={grid_source} | "
                            f"samples={int(grid_meta['total_samples'])} | "
                            f"x3_support={x3_support_label}",
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
                try:
                    train_total, train_parts, train_traj = evaluate_split(
                        force_module=model,
                        known_pars=prepared.known_pars,
                        mech_module=mech_module,
                        ode_true=tensors["ode_full"],
                        x2dot_true=tensors["x2dot_full"],
                        contact_mask=tensors["contact_full"],
                        times=tensors["times_full"],
                        ode_method=cfg.ode_method,
                        ode_rtol=cfg.ode_rtol,
                        ode_atol=cfg.ode_atol,
                        loss_indices=tensors["train_idx"],
                    )
                    if not torch.isfinite(train_total):
                        epoch_fail_reason = "train_loss_nonfinite"
                    else:
                        train_total.backward()
                        grad_norm_nn = _grad_norm(("model", model))
                        grad_norm_mech = _grad_norm(("mech", mech_module))
                        grad_norm_raw = _grad_norm(("model", model), ("mech", mech_module))
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

                step_retry_count = 0
                step_accept_reason = "accepted"
                step_accepted = False
                step_update_accepted = False
                step_reject_reason = ""
                train_loss_before = float(train_total.detach())
                step_candidate_loss = float("nan")
                step_alpha = 1.0
                step_base_lr = float(lr)
                step_grad_dot_p = float("nan")
                step_armijo_rhs = float("nan")
                accepted_val_total: torch.Tensor | None = None
                accepted_val_parts: TorchLossParts | None = None
                adam_state_reset_attempted = False
                adam_state_reset_accepted = False
                adam_state_reset_trigger_retries = 0
                adam_state_reset_retries = 0

                if step_controller == "armijo_backtracking":
                    step_base_model_state = copy.deepcopy(model.state_dict())
                    step_base_mech_state = copy.deepcopy(mech_module.state_dict())
                    step_base_opt_state = copy.deepcopy(optimizer.state_dict())
                    base_params = _capture_params(("model", model), ("mech", mech_module))
                    grad_cache = _capture_grads(("model", model), ("mech", mech_module))
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
                        model.load_state_dict(copy.deepcopy(step_base_model_state))
                        mech_module.load_state_dict(copy.deepcopy(step_base_mech_state))
                        optimizer.load_state_dict(copy.deepcopy(step_base_opt_state))
                        _set_optimizer_lr(optimizer, trial_lr)
                        _restore_grads(grad_cache, ("model", model), ("mech", mech_module))
                        optimizer.step()
                        model.train()
                        grad_dot_delta = _grad_dot_delta(base_params, grad_cache, ("model", model), ("mech", mech_module))
                        armijo_rhs = float(train_loss_before + c1 * grad_dot_delta)
                        try:
                            trial_total, trial_parts, trial_traj = evaluate_split(
                                force_module=model,
                                known_pars=prepared.known_pars,
                                mech_module=mech_module,
                                ode_true=tensors["ode_full"],
                                x2dot_true=tensors["x2dot_full"],
                                contact_mask=tensors["contact_full"],
                                times=tensors["times_full"],
                                ode_method=cfg.ode_method,
                                ode_rtol=cfg.ode_rtol,
                                ode_atol=cfg.ode_atol,
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

                        model.load_state_dict(copy.deepcopy(step_base_model_state))
                        mech_module.load_state_dict(copy.deepcopy(step_base_mech_state))
                        optimizer.load_state_dict(copy.deepcopy(step_base_opt_state))
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
                            model.load_state_dict(copy.deepcopy(step_base_model_state))
                            mech_module.load_state_dict(copy.deepcopy(step_base_mech_state))
                            optimizer.load_state_dict(copy.deepcopy(reset_opt_base_state))
                            _set_optimizer_lr(optimizer, trial_lr)
                            _restore_grads(grad_cache, ("model", model), ("mech", mech_module))
                            optimizer.step()
                            model.train()
                            grad_dot_delta = _grad_dot_delta(base_params, grad_cache, ("model", model), ("mech", mech_module))
                            armijo_rhs = float(train_loss_before + c1 * grad_dot_delta)
                            try:
                                trial_total, trial_parts, trial_traj = evaluate_split(
                                    force_module=model,
                                    known_pars=prepared.known_pars,
                                    mech_module=mech_module,
                                    ode_true=tensors["ode_full"],
                                    x2dot_true=tensors["x2dot_full"],
                                    contact_mask=tensors["contact_full"],
                                    times=tensors["times_full"],
                                    ode_method=cfg.ode_method,
                                    ode_rtol=cfg.ode_rtol,
                                    ode_atol=cfg.ode_atol,
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
                        model.load_state_dict(copy.deepcopy(step_base_model_state))
                        mech_module.load_state_dict(copy.deepcopy(step_base_mech_state))
                        optimizer.load_state_dict(copy.deepcopy(step_base_opt_state))
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

                elif step_controller == "off" or not cfg.step_guard_enabled:
                    native_base_model_state = copy.deepcopy(model.state_dict())
                    native_base_mech_state = copy.deepcopy(mech_module.state_dict())
                    native_base_opt_state = copy.deepcopy(optimizer.state_dict())
                    _set_optimizer_lr(optimizer, lr)
                    _log_fingerprint(
                        log,
                        event_log,
                        epoch=epoch + 1,
                        phase="adam",
                        stage="before_optimizer_step",
                        model=model,
                        mech_module=mech_module,
                        optimizer=optimizer,
                        grid_inputs=observable_grid_inputs,
                        extra={"attempt": int(epoch_attempt), "lr": float(lr)},
                    )
                    optimizer.step()
                    model.train()
                    _log_fingerprint(
                        log,
                        event_log,
                        epoch=epoch + 1,
                        phase="adam",
                        stage="after_optimizer_step",
                        model=model,
                        mech_module=mech_module,
                        optimizer=optimizer,
                        grid_inputs=observable_grid_inputs,
                        extra={"attempt": int(epoch_attempt), "lr": float(lr)},
                    )
                    try:
                        trial_total, trial_parts, trial_traj = evaluate_split(
                            force_module=model,
                            known_pars=prepared.known_pars,
                            mech_module=mech_module,
                            ode_true=tensors["ode_full"],
                            x2dot_true=tensors["x2dot_full"],
                            contact_mask=tensors["contact_full"],
                            times=tensors["times_full"],
                            ode_method=cfg.ode_method,
                            ode_rtol=cfg.ode_rtol,
                            ode_atol=cfg.ode_atol,
                            loss_indices=tensors["train_idx"],
                        )
                        trial_loss = float(trial_total.detach())
                    except Exception as err:
                        epoch_fail_reason = f"step_trial_exception:{err}"
                        trial_total = None
                        trial_parts = None
                        trial_traj = None
                        trial_loss = float("inf")

                    step_candidate_loss = float(trial_loss)
                    step_effective_lr = float(lr)
                    step_next_lr = float(lr)
                    if epoch_fail_reason == "" and not np.isfinite(trial_loss):
                        epoch_fail_reason = "step_trial_loss_nonfinite"

                    if epoch_fail_reason == "":
                        train_total = trial_total
                        train_parts = trial_parts
                        accepted_train_traj = trial_traj
                        step_accepted = True
                        step_update_accepted = True
                        break
                    model.load_state_dict(copy.deepcopy(native_base_model_state))
                    mech_module.load_state_dict(copy.deepcopy(native_base_mech_state))
                    optimizer.load_state_dict(copy.deepcopy(native_base_opt_state))
                    _set_optimizer_lr(optimizer, lr)
                else:
                    prev_loss_ref = train_loss_last if np.isfinite(train_loss_last) else float("nan")
                    train_loss_before = float(train_total.detach())
                    val_loss_before = float("nan")
                    step_base_model_state = copy.deepcopy(model.state_dict())
                    step_base_mech_state = copy.deepcopy(mech_module.state_dict())
                    step_base_opt_state = copy.deepcopy(optimizer.state_dict())
                    step_base_train_total = train_total
                    step_base_train_parts = train_parts
                    step_base_train_traj = train_traj

                    if cfg.step_guard_validate_val:
                        try:
                            model.eval()
                            with torch.no_grad():
                                step_base_val_total, _step_base_val_parts, _ = evaluate_split(
                                    force_module=model,
                                    known_pars=prepared.known_pars,
                                    mech_module=mech_module,
                                    ode_true=tensors["ode_full"],
                                    x2dot_true=tensors["x2dot_full"],
                                    contact_mask=tensors["contact_full"],
                                    times=tensors["times_full"],
                                    ode_method=cfg.ode_method,
                                    ode_rtol=cfg.ode_rtol,
                                    ode_atol=cfg.ode_atol,
                                    loss_indices=tensors["val_idx"],
                                )
                            val_loss_before = float(step_base_val_total.detach())
                            model.train()
                            if not np.isfinite(val_loss_before):
                                epoch_fail_reason = "step_base_val_loss_nonfinite"
                        except Exception as err:
                            model.train()
                            epoch_fail_reason = f"step_base_val_exception:{err}"

                    step_retry_infinite = int(cfg.step_retry_max) < 0
                    attempts_total = max(1, int(cfg.step_retry_max) + 1)
                    step_attempt = 1
                    step_trial_lr = float(lr)
                    while epoch_fail_reason == "":
                        _set_optimizer_lr(optimizer, step_trial_lr)
                        optimizer.step()
                        model.train()
                        try:
                            trial_total, trial_parts, trial_traj = evaluate_split(
                                force_module=model,
                                known_pars=prepared.known_pars,
                                mech_module=mech_module,
                                ode_true=tensors["ode_full"],
                                x2dot_true=tensors["x2dot_full"],
                                contact_mask=tensors["contact_full"],
                                times=tensors["times_full"],
                                ode_method=cfg.ode_method,
                                ode_rtol=cfg.ode_rtol,
                                ode_atol=cfg.ode_atol,
                                loss_indices=tensors["train_idx"],
                            )
                            trial_loss = float(trial_total.detach())
                        except Exception as err:
                            step_accept_reason = f"step_trial_exception:{err}"
                            trial_total = None
                            trial_parts = None
                            trial_traj = None
                            trial_loss = float("inf")

                        if step_accept_reason == "accepted":
                            if not np.isfinite(trial_loss):
                                step_accept_reason = "step_trial_loss_nonfinite"
                            elif np.isfinite(cfg.step_max_loss_increase_frac) and trial_loss > train_loss_before * (1.0 + cfg.step_max_loss_increase_frac):
                                step_accept_reason = "step_trial_loss_jump"
                            elif np.isfinite(prev_loss_ref) and trial_loss > prev_loss_ref * (1.0 + cfg.step_max_loss_increase_frac):
                                step_accept_reason = "step_prev_epoch_loss_jump"

                        trial_val_loss = float("nan")
                        trial_val_total = None
                        trial_val_parts = None
                        if step_accept_reason == "accepted" and cfg.step_guard_validate_val:
                            try:
                                model.eval()
                                with torch.no_grad():
                                    trial_val_total, trial_val_parts, _ = evaluate_split(
                                        force_module=model,
                                        known_pars=prepared.known_pars,
                                        mech_module=mech_module,
                                        ode_true=tensors["ode_full"],
                                        x2dot_true=tensors["x2dot_full"],
                                        contact_mask=tensors["contact_full"],
                                        times=tensors["times_full"],
                                        ode_method=cfg.ode_method,
                                        ode_rtol=cfg.ode_rtol,
                                        ode_atol=cfg.ode_atol,
                                        loss_indices=tensors["val_idx"],
                                    )
                                trial_val_loss = float(trial_val_total.detach())
                                model.train()
                            except Exception as err:
                                model.train()
                                step_accept_reason = f"step_trial_val_exception:{err}"

                            if step_accept_reason == "accepted":
                                if not np.isfinite(trial_val_loss):
                                    step_accept_reason = "step_trial_val_loss_nonfinite"
                                elif (
                                    np.isfinite(cfg.step_max_loss_increase_frac)
                                    and np.isfinite(val_loss_before)
                                    and trial_val_loss > val_loss_before * (1.0 + cfg.step_max_loss_increase_frac)
                                ):
                                    step_accept_reason = "step_trial_val_loss_jump"

                        if step_accept_reason == "accepted":
                            train_total = trial_total
                            train_parts = trial_parts
                            accepted_train_traj = trial_traj
                            accepted_val_total = trial_val_total
                            accepted_val_parts = trial_val_parts
                            step_accepted = True
                            step_update_accepted = True
                            step_retry_count = step_attempt - 1
                            step_effective_lr = float(step_trial_lr)
                            if cfg.step_retry_reset_lr_each_epoch:
                                _set_optimizer_lr(optimizer, lr)
                            else:
                                lr = float(step_trial_lr)
                            step_next_lr = float(lr)
                            if step_retry_count > 0:
                                _log_line(
                                    log,
                                    f"step-guard: accepted after {step_retry_count} retry/reduction(s) "
                                    f"| trial_lr={step_effective_lr:.3e} | next_lr={step_next_lr:.3e} "
                                    f"| train={trial_loss:.6e} "
                                    f"| val={trial_val_loss:.6e}",
                                )
                            break

                        if (not step_retry_infinite) and step_attempt >= attempts_total:
                            step_retry_count = max(0, attempts_total - 1)
                            model.load_state_dict(copy.deepcopy(step_base_model_state))
                            mech_module.load_state_dict(copy.deepcopy(step_base_mech_state))
                            optimizer.load_state_dict(copy.deepcopy(step_base_opt_state))
                            train_total = step_base_train_total
                            train_parts = step_base_train_parts
                            accepted_train_traj = step_base_train_traj
                            step_accept_reason = "step_retries_exhausted_keep_base"
                            step_effective_lr = float(step_trial_lr)
                            if cfg.step_retry_reset_lr_each_epoch:
                                _set_optimizer_lr(optimizer, lr)
                            else:
                                lr = float(step_trial_lr)
                            step_next_lr = float(lr)
                            _log_line(
                                log,
                                f"step-guard: retries exhausted after {step_retry_count} retries; keep base parameters "
                                f"| reason={step_accept_reason} | trial_lr={step_effective_lr:.3e} | next_lr={step_next_lr:.3e}",
                            )
                            step_accepted = True
                            step_reject_reason = step_accept_reason
                            step_update_accepted = False
                            break

                        step_trial_lr = step_trial_lr * cfg.step_retry_lr_factor
                        step_retry_count = step_attempt
                        step_retry_limit_label = "inf" if step_retry_infinite else str(cfg.step_retry_max)
                        _log_line(
                                log,
                                f"step-guard retry {step_attempt}/{step_retry_limit_label} -- reason={step_accept_reason} | trial_lr={step_trial_lr:.3e} | next_lr={lr:.3e}",
                            )
                        model.load_state_dict(copy.deepcopy(step_base_model_state))
                        mech_module.load_state_dict(copy.deepcopy(step_base_mech_state))
                        optimizer.load_state_dict(copy.deepcopy(step_base_opt_state))
                        optimizer.zero_grad(set_to_none=True)
                        model.train()
                        try:
                            train_total, train_parts, train_traj = evaluate_split(
                                force_module=model,
                                known_pars=prepared.known_pars,
                                mech_module=mech_module,
                                ode_true=tensors["ode_full"],
                                x2dot_true=tensors["x2dot_full"],
                                contact_mask=tensors["contact_full"],
                                times=tensors["times_full"],
                                ode_method=cfg.ode_method,
                                ode_rtol=cfg.ode_rtol,
                                ode_atol=cfg.ode_atol,
                                loss_indices=tensors["train_idx"],
                            )
                            if not torch.isfinite(train_total):
                                failure_reason = "trial_failed:retry_base_train_loss_nonfinite"
                                failure_epoch = epoch + 1
                                break
                            train_total.backward()
                            grad_norm_nn = _grad_norm(("model", model))
                            grad_norm_mech = _grad_norm(("mech", mech_module))
                            grad_norm_raw = _grad_norm(("model", model), ("mech", mech_module))
                        except Exception as err:
                            failure_reason = f"trial_failed:retry_base_exception:{err}"
                            failure_epoch = epoch + 1
                            break
                        accepted_train_traj = None
                        if not np.isfinite(grad_norm_raw):
                            failure_reason = "trial_failed:retry_base_grad_norm_nonfinite"
                            failure_epoch = epoch + 1
                            break
                        grad_norm = grad_norm_raw
                        step_accept_reason = "accepted"
                        step_attempt += 1

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

            eval_val_this_epoch = _should_eval_validation(cfg, epoch + 1)
            model.eval()
            try:
                if not eval_val_this_epoch:
                    val_total = _nan_loss_tensor(dtype=dtype, device=cfg.device)
                    val_parts = _nan_loss_parts()
                elif accepted_val_total is not None and accepted_val_parts is not None:
                    val_total = accepted_val_total
                    val_parts = accepted_val_parts
                else:
                    with torch.no_grad():
                        val_total, val_parts, _ = evaluate_split(
                            force_module=model,
                            known_pars=prepared.known_pars,
                            mech_module=mech_module,
                            ode_true=tensors["ode_full"],
                            x2dot_true=tensors["x2dot_full"],
                            contact_mask=tensors["contact_full"],
                            times=tensors["times_full"],
                            ode_method=cfg.ode_method,
                            ode_rtol=cfg.ode_rtol,
                            ode_atol=cfg.ode_atol,
                            loss_indices=tensors["val_idx"],
                        )
                if eval_val_this_epoch and not torch.isfinite(val_total):
                    failure_reason = "val_loss_nonfinite"
                    failure_epoch = epoch + 1
                    break
            except Exception as err:
                failure_reason = f"val_exception:{err}"
                failure_epoch = epoch + 1
                break

            soft_mask_meta = model.soft_mask_summary()
            x3_support_used_epoch = x3_agu_support_current
            x3_norm_monitor = _x3_norm_monitor_from_traj(
                accepted_train_traj,
                model,
                x3_refit_meta,
                x3_current_support=x3_support_used_epoch,
            )
            x3_support_next = _x3_norm_support_from_monitor(x3_norm_monitor)
            x3_support_updated = False
            if _valid_x3_norm_support(x3_support_next):
                x3_agu_support_current = (float(x3_support_next[0]), float(x3_support_next[1]))
                observable_grid_inputs = _observable_grid_inputs_from_ode(
                    ode_train=tensors["ode_train"],
                    state_mean=prepared.state_mean,
                    state_scale=prepared.state_scale,
                    x3_norm_support=x3_agu_support_current,
                )
                x3_support_updated = True
                _log_fingerprint(
                    log,
                    event_log,
                    epoch=epoch + 1,
                    phase="adam",
                    stage="after_x3_support_update",
                    model=model,
                    mech_module=mech_module,
                    optimizer=optimizer,
                    grid_inputs=observable_grid_inputs,
                    extra={
                        "x3_agu_support_min": float(x3_agu_support_current[0]),
                        "x3_agu_support_max": float(x3_agu_support_current[1]),
                    },
                )
            row = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_total.detach()),
                "val_loss": float(val_total.detach()),
                "validation_eval_mode": str(cfg.val_eval_mode),
                "validation_evaluated": bool(eval_val_this_epoch),
                "validation_is_final": bool(final_only_validation and eval_val_this_epoch),
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
                "train_x3_range_amp": float(train_parts.x3_range_amp),
                "val_x3_range_amp": float(val_parts.x3_range_amp),
                "train_fts_range_amp": float(train_parts.fts_range_amp),
                "val_fts_range_amp": float(val_parts.fts_range_amp),
                "train_x1_rec": float(train_parts.x1_rec),
                "val_x1_rec": float(val_parts.x1_rec),
                "train_x2_rec": float(train_parts.x2_rec),
                "val_x2_rec": float(val_parts.x2_rec),
                "train_x2dot_rec": float(train_parts.x2dot_rec),
                "val_x2dot_rec": float(val_parts.x2dot_rec),
                "g_nn": float(model.gain().detach().cpu().item()),
                "soft_mask_enabled": bool(soft_mask_meta["enabled"]),
                "soft_mask_trainable": bool(soft_mask_meta["trainable"]),
                "soft_mask_s0": float(soft_mask_meta["s0"]),
                "soft_mask_s0_a0": float(soft_mask_meta["s0_a0"]),
                "soft_mask_alpha": float(soft_mask_meta["alpha"]),
                "soft_mask_alpha_a0": float(soft_mask_meta["alpha_a0"]),
                "soft_mask_m_min": 0.0,
                "grad_norm": float(grad_norm),
                "grad_norm_raw": float(grad_norm_raw),
                "grad_norm_nn": float(grad_norm_nn),
                "grad_norm_mech": float(grad_norm_mech),
                "lr": float(step_effective_lr),
                "step_effective_lr": float(step_effective_lr),
                "step_next_lr": float(step_next_lr),
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
                "adam_state_reset_attempted": bool(adam_state_reset_attempted),
                "adam_state_reset_accepted": bool(adam_state_reset_accepted),
                "adam_state_reset_trigger_retries": float(adam_state_reset_trigger_retries),
                "adam_state_reset_retries": float(adam_state_reset_retries),
                "grid_base_samples": float(grid_update_meta_epoch["base_samples"]),
                "grid_extra_samples": float(grid_update_meta_epoch["extra_samples"]),
                "grid_total_samples": float(grid_update_meta_epoch["total_samples"]),
                "x3_agu_support_used_min": (
                    float(x3_support_used_epoch[0]) if _valid_x3_norm_support(x3_support_used_epoch) else float("nan")
                ),
                "x3_agu_support_used_max": (
                    float(x3_support_used_epoch[1]) if _valid_x3_norm_support(x3_support_used_epoch) else float("nan")
                ),
                "x3_agu_next_support_min": (
                    float(x3_support_next[0]) if _valid_x3_norm_support(x3_support_next) else float("nan")
                ),
                "x3_agu_next_support_max": (
                    float(x3_support_next[1]) if _valid_x3_norm_support(x3_support_next) else float("nan")
                ),
                "x3_agu_support_updated": bool(x3_support_updated),
            }
            row.update(x3_norm_monitor)
            row.update(_x3_drift_from_epoch0(accepted_train_traj, x3_drift_epoch0_x3))
            ks_hat, cs_hat = _current_mech_numpy(mech_module)
            row["ks_hat"] = ks_hat
            row["cs_hat"] = cs_hat
            history.append(row)
            train_loss_last = float(train_total.detach())
            if accepted_train_traj is not None:
                cached_grid_traj = accepted_train_traj.detach().clone()

            if final_only_validation and eval_val_this_epoch:
                final_val_loss = float(val_total.detach())
                final_val_epoch = epoch + 1
                final_checkpoint_snapshot = _snapshot_split(
                    force_module=model,
                    known_pars=prepared.known_pars,
                    mech_module=mech_module,
                    split=split,
                    window_mode=cfg.window_mode,
                    tensors=tensors,
                    dtype=dtype,
                    device=cfg.device,
                    ode_method=cfg.ode_method,
                    ode_rtol=cfg.ode_rtol,
                    ode_atol=cfg.ode_atol,
                )
                final_checkpoint_payload = {
                    "epoch": final_val_epoch,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "mech_state_dict": copy.deepcopy(mech_module.state_dict()),
                    "config": asdict(cfg),
                    "rng_state": capture_rng_state(),
                    "state_mean": prepared.state_mean,
                    "state_scale": prepared.state_scale,
                    "initial_grid_support": None if initial_grid_support is None else initial_grid_support.tolist(),
                    "initial_grid_support_meta": initial_grid_support_meta,
                    "x3_refit_meta": x3_refit_meta,
                    "x3_agu_support_current": _x3_norm_support_to_payload(x3_agu_support_current),
                    "x3_drift_epoch0_x3": None
                    if x3_drift_epoch0_x3 is None
                    else x3_drift_epoch0_x3.detach().cpu(),
                    "known_pars": prepared.known_pars,
                    "warmstart_source": None if warmstart_meta is None else str(warmstart_meta.get("source", "")),
                    "warmstart": warmstart_meta,
                    "stage1_warmstart": stage1_warmstart_meta,
                    "prestage2_warmstart": prestage2_warmstart_meta,
                    "resume_identity": _resume_identity(
                        shard_index=shard_index,
                        window_index=window_index,
                        split=split,
                        warmstart_meta=warmstart_meta,
                    ),
                    "window_meta": _window_meta_dict(split, cfg.window_mode),
                    "history": history,
                    "final_snapshot": final_checkpoint_snapshot,
                    "final_val_loss": float(final_val_loss),
                    "final_val_epoch": int(final_val_epoch),
                    "validation_eval_mode": str(cfg.val_eval_mode),
                }
                _save_torch(checkpoint_path, final_checkpoint_payload)
                _save_pickle(checkpoint_viz_path, _make_viz_payload(final_checkpoint_payload))
            elif (not final_only_validation) and float(val_total.detach()) < best_val:
                best_val = float(val_total.detach())
                best_epoch = epoch + 1
                best_snapshot = _snapshot_split(
                    force_module=model,
                    known_pars=prepared.known_pars,
                    mech_module=mech_module,
                    split=split,
                    window_mode=cfg.window_mode,
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
                    "mech_state_dict": copy.deepcopy(mech_module.state_dict()),
                    "config": asdict(cfg),
                    "rng_state": capture_rng_state(),
                    "state_mean": prepared.state_mean,
                    "state_scale": prepared.state_scale,
                    "initial_grid_support": None if initial_grid_support is None else initial_grid_support.tolist(),
                    "initial_grid_support_meta": initial_grid_support_meta,
                    "x3_refit_meta": x3_refit_meta,
                    "x3_agu_support_current": _x3_norm_support_to_payload(x3_agu_support_current),
                    "x3_drift_epoch0_x3": None
                    if x3_drift_epoch0_x3 is None
                    else x3_drift_epoch0_x3.detach().cpu(),
                    "known_pars": prepared.known_pars,
                    "warmstart_source": None if warmstart_meta is None else str(warmstart_meta.get("source", "")),
                    "warmstart": warmstart_meta,
                    "stage1_warmstart": stage1_warmstart_meta,
                    "prestage2_warmstart": prestage2_warmstart_meta,
                    "resume_identity": _resume_identity(
                        shard_index=shard_index,
                        window_index=window_index,
                        split=split,
                        warmstart_meta=warmstart_meta,
                    ),
                    "window_meta": _window_meta_dict(split, cfg.window_mode),
                    "history": history,
                    "best_snapshot": best_snapshot,
                }
                _save_torch(checkpoint_path, best_payload)
                _save_pickle(checkpoint_viz_path, _make_viz_payload(best_payload))

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
                _log_line(
                    log,
                    f"  grad_norm={grad_norm:.3e} lr={step_effective_lr:.6g} next_lr={step_next_lr:.6g}",
                )
                _log_line(log, f"  grad_norm_raw={grad_norm_raw:.3e} grad_norm_scaled={grad_norm:.3e}")
                _log_line(log, f"  grad_norm_nn={grad_norm_nn:.3e} grad_norm_mech={grad_norm_mech:.3e}")
                _log_line(
                    log,
                    "  x3 drift from epoch0: "
                    f"{float(row['x3_drift_from_epoch0_pct']):.3e}% "
                    f"rmse_abs={float(row['x3_drift_from_epoch0_abs']):.3e}",
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
                if recovered:
                    _log_line(log, f"  retry: recovered_after={epoch_attempt - 1} retry/reduction(s)")
                if step_retry_count > 0 and step_accept_reason == "accepted":
                    _log_line(log, f"  step-guard: accepted_after={step_retry_count} retry/reduction(s)")
                if adam_state_reset_attempted:
                    _log_line(
                        log,
                        "  adam state reset: "
                        f"trigger_retries={adam_state_reset_trigger_retries} "
                        f"accepted={'YES' if adam_state_reset_accepted else 'NO'} "
                        f"post_reset_retries={adam_state_reset_retries}",
                    )
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
                        "  AGU: "
                        f"samples={int(row['grid_total_samples'])} "
                        "source=normalized_observed_x1x2_pred_informed_range_uniform_x3",
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
                _log_line(
                    log,
                    "  range amp: "
                    f"x3={float(train_parts.x3_range_amp):.6e} "
                    f"Fts={float(train_parts.fts_range_amp):.6e} "
                    "source=mean_abs_observed_x1_and_F_actuation_current_window",
                )
                _log_line(
                    log,
                    "  rec: "
                    f"x1={float(train_parts.x1_rec):.2f}% "
                    f"x2={float(train_parts.x2_rec):.2f}% "
                    f"x2dot={float(train_parts.x2dot_rec):.2f}%",
                )
                _log_line(log, f"  {_format_x3_norm_monitor(x3_norm_monitor)}")
                _log_line(
                    log,
                    "  x3 AGU next support: "
                    f"[{row['x3_agu_next_support_min']:.6e}, {row['x3_agu_next_support_max']:.6e}] "
                    f"updated={'YES' if row['x3_agu_support_updated'] else 'NO'} "
                    "source=accepted_epoch_x3_norm_range density=uniform",
                )
                _log_line(
                    log,
                    "  mech: "
                    f"ks={row['ks_hat']:.6e} "
                    f"cs={row['cs_hat']:.6e}",
                )
                if eval_val_this_epoch:
                    _log_line(log, f"KAN final val epoch {epoch + 1} val={float(val_total.detach()):.6e}" if final_only_validation else f"KAN val epoch {epoch + 1} val={float(val_total.detach()):.6e}")
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
                    _log_line(
                        log,
                        "  val range amp: "
                        f"x3={float(val_parts.x3_range_amp):.6e} "
                        f"Fts={float(val_parts.fts_range_amp):.6e} "
                        "source=mean_abs_observed_x1_and_F_actuation_current_window",
                    )
                    _log_line(
                        log,
                        "  val rec: "
                        f"x1={float(val_parts.x1_rec):.2f}% "
                        f"x2={float(val_parts.x2_rec):.2f}% "
                        f"x2dot={float(val_parts.x2dot_rec):.2f}%",
                    )

            if lr_adapt_active and np.isfinite(grad_norm) and grad_norm > 0.0:
                if not np.isfinite(grad_ema):
                    grad_ema = grad_norm
                else:
                    grad_ema = cfg.lr_ema_alpha * grad_ema + (1.0 - cfg.lr_ema_alpha) * grad_norm
                if not np.isfinite(grad_target):
                    grad_target = grad_ema
                ratio = grad_target / (grad_ema + cfg.lr_eps)
                candidate_lr = max(cfg.lr_min, min(float(lr * (ratio ** cfg.lr_eta)), cfg.lr_max))
                if cfg.lr_adapt_up_only:
                    lr = max(lr, candidate_lr)
                else:
                    lr = candidate_lr
                _set_optimizer_lr(optimizer, lr)

            if cfg.plateau_early_stop:
                recent_losses.append(float(train_total.detach()))
                if len(recent_losses) > int(cfg.plateau_window):
                    recent_losses.pop(0)

            if ((epoch + 1) % cfg.checkpoint_every) == 0 or (epoch + 1) == int(cfg.epochs):
                payload = {
                    "epoch": int(epoch + 1),
                    "history": history,
                    "config": asdict(cfg),
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "mech_state_dict": copy.deepcopy(mech_module.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "optimizer_phase": "adam",
                    "rng_state": capture_rng_state(),
                    "state_mean": prepared.state_mean,
                    "state_scale": prepared.state_scale,
                    "initial_grid_support": None if initial_grid_support is None else initial_grid_support.tolist(),
                    "initial_grid_support_meta": initial_grid_support_meta,
                    "x3_refit_meta": x3_refit_meta,
                    "x3_agu_support_current": _x3_norm_support_to_payload(x3_agu_support_current),
                    "x3_drift_epoch0_x3": None
                    if x3_drift_epoch0_x3 is None
                    else x3_drift_epoch0_x3.detach().cpu(),
                    "known_pars": prepared.known_pars,
                    "warmstart_source": None if warmstart_meta is None else str(warmstart_meta.get("source", "")),
                    "warmstart": warmstart_meta,
                    "stage1_warmstart": stage1_warmstart_meta,
                    "prestage2_warmstart": prestage2_warmstart_meta,
                    "validation_eval_mode": str(cfg.val_eval_mode),
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val),
                    "final_val_epoch": int(final_val_epoch),
                    "final_val_loss": float(final_val_loss),
                    "lr": float(lr),
                    "grad_ema": float(grad_ema),
                    "grad_target": float(grad_target),
                    "recent_losses": list(recent_losses),
                    "resume_identity": _resume_identity(
                        shard_index=shard_index,
                        window_index=window_index,
                        split=split,
                        warmstart_meta=warmstart_meta,
                    ),
                    "window_meta": _window_meta_dict(split, cfg.window_mode),
                }
                _save_torch(running_checkpoint_path, payload)
                _log_line(log, f"checkpoint saved | epoch={epoch + 1} | path={running_checkpoint_path}")

            if not step_update_accepted:
                if step_controller == "armijo_backtracking" and int(step_retry_count) >= int(cfg.backtrack_max):
                    stop_reason = f"adam_retry_cap_hit:{step_reject_reason or step_accept_reason}"
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
                        "retry_cap": int(cfg.backtrack_max if step_controller == "armijo_backtracking" else cfg.step_retry_max),
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
                    f"| adam_state_reset_attempted={'YES' if adam_state_reset_attempted else 'NO'} "
                    f"| adam_state_reset_accepted={'YES' if adam_state_reset_accepted else 'NO'} "
                    f"| loss={train_loss_before:.6e}->{float(train_total.detach()):.6e}; "
                    "no further epochs will be consumed",
                )
                break

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
                    f"  early-stop: plateau_{cfg.plateau_window}ep (|螖loss| < {cfg.plateau_tol:.1e})",
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
            and stop_reason == ""
            and bool(cfg.lbfgs_enabled)
            and int(cfg.lbfgs_steps) > 0
            and not good_enough_reached
        ):
            planned_total_epochs = int(cfg.adam_epochs) + int(cfg.lbfgs_steps)
            lbfgs_effective_steps = max(0, planned_total_epochs - len(history))
            if lbfgs_effective_steps <= 0:
                _log_line(
                    log,
                    "LBFGS phase skipped: "
                    f"history already reached planned_total_epochs={planned_total_epochs}",
                )
                lbfgs_effective_steps = 0
            lbfgs_optimizer, lbfgs_line_search = _make_lbfgs_optimizer(cfg, model, mech_module)
            if resume_optimizer_phase == "lbfgs" and isinstance(deferred_optimizer_state_dict, dict):
                lbfgs_optimizer.load_state_dict(deferred_optimizer_state_dict)
                _log_line(log, "LBFGS phase resume: restored LBFGS optimizer state from checkpoint")
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

            lbfgs_outer_step = 1
            lbfgs_zero_step_restart_used = False
            while lbfgs_outer_step <= lbfgs_effective_steps:
                lbfgs_wall_start = perf_counter()
                outer_epoch = len(history) + 1
                closure_calls = 0
                closure_calls_total = 0
                closure_last_loss = float("nan")
                closure_last_grad_norm = float("nan")
                closure_error = ""
                lbfgs_attempt_index = 1
                lbfgs_start_model_state = copy.deepcopy(model.state_dict())
                lbfgs_start_mech_state = copy.deepcopy(mech_module.state_dict())
                lbfgs_start_opt_state = copy.deepcopy(lbfgs_optimizer.state_dict())
                lbfgs_start_params = _capture_params(("model", model), ("mech", mech_module))
                lbfgs_param_delta_l2 = float("nan")
                lbfgs_param_delta_max_abs = float("nan")
                lbfgs_loss_drop = float("nan")

                model.eval()
                try:
                    with torch.no_grad():
                        loss_before_t, parts_before, traj_before = evaluate_split(
                            force_module=model,
                            known_pars=prepared.known_pars,
                            mech_module=mech_module,
                            ode_true=tensors["ode_full"],
                            x2dot_true=tensors["x2dot_full"],
                            contact_mask=tensors["contact_full"],
                            times=tensors["times_full"],
                            ode_method=cfg.ode_method,
                            ode_rtol=cfg.ode_rtol,
                            ode_atol=cfg.ode_atol,
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
                            mech_module=mech_module,
                            ode_true=tensors["ode_full"],
                            x2dot_true=tensors["x2dot_full"],
                            contact_mask=tensors["contact_full"],
                            times=tensors["times_full"],
                            ode_method=cfg.ode_method,
                            ode_rtol=cfg.ode_rtol,
                            ode_atol=cfg.ode_atol,
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
                                    "attempt": int(lbfgs_attempt_index),
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
                        closure_last_grad_norm = _grad_norm(("model", model), ("mech", mech_module))
                        _write_event(
                            event_log,
                            {
                                "event": "lbfgs_closure",
                                "epoch": int(outer_epoch),
                                "lbfgs_outer_step": int(lbfgs_outer_step),
                                "attempt": int(lbfgs_attempt_index),
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
                try:
                    _log_fingerprint(
                        log,
                        event_log,
                        epoch=outer_epoch,
                        phase="lbfgs",
                        stage="before_lbfgs_step",
                        model=model,
                        mech_module=mech_module,
                        optimizer=lbfgs_optimizer,
                        grid_inputs=observable_grid_inputs,
                        extra={
                            "lbfgs_outer_step": int(lbfgs_outer_step),
                            "attempt": int(lbfgs_attempt_index),
                            "line_search": lbfgs_line_search,
                        },
                    )
                    returned_loss = lbfgs_optimizer.step(closure)
                    lbfgs_returned_loss = float(returned_loss.detach()) if isinstance(returned_loss, torch.Tensor) else float(returned_loss)
                    closure_calls_total = int(closure_calls)
                    _log_fingerprint(
                        log,
                        event_log,
                        epoch=outer_epoch,
                        phase="lbfgs",
                        stage="after_lbfgs_step",
                        model=model,
                        mech_module=mech_module,
                        optimizer=lbfgs_optimizer,
                        grid_inputs=observable_grid_inputs,
                        extra={
                            "lbfgs_outer_step": int(lbfgs_outer_step),
                            "attempt": int(lbfgs_attempt_index),
                            "line_search": lbfgs_line_search,
                            "closure_calls": int(closure_calls_total),
                            "returned_loss": float(lbfgs_returned_loss),
                        },
                    )
                except Exception as err:
                    lbfgs_stop_reason = closure_error or f"lbfgs_step_exception:{err}"
                    model.load_state_dict(copy.deepcopy(lbfgs_start_model_state))
                    mech_module.load_state_dict(copy.deepcopy(lbfgs_start_mech_state))
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
                grad_norm_nn = float("nan")
                grad_norm_mech = float("nan")
                try:
                    lbfgs_optimizer.zero_grad()
                    train_total, train_parts, accepted_train_traj = evaluate_split(
                        force_module=model,
                        known_pars=prepared.known_pars,
                        mech_module=mech_module,
                        ode_true=tensors["ode_full"],
                        x2dot_true=tensors["x2dot_full"],
                        contact_mask=tensors["contact_full"],
                        times=tensors["times_full"],
                        ode_method=cfg.ode_method,
                        ode_rtol=cfg.ode_rtol,
                        ode_atol=cfg.ode_atol,
                        loss_indices=tensors["train_idx"],
                    )
                    train_loss_after = float(train_total.detach())
                    lbfgs_loss_drop = float(lbfgs_loss_before - train_loss_after)
                    lbfgs_param_delta_l2, lbfgs_param_delta_max_abs = _param_delta_stats(
                        lbfgs_start_params,
                        ("model", model),
                        ("mech", mech_module),
                    )
                    if not torch.isfinite(train_total):
                        lbfgs_reject_reason = "lbfgs_after_loss_nonfinite"
                    else:
                        train_total.backward()
                        grad_norm_nn = _grad_norm(("model", model))
                        grad_norm_mech = _grad_norm(("mech", mech_module))
                        grad_norm_raw = _grad_norm(("model", model), ("mech", mech_module))
                except Exception as err:
                    train_loss_after = float("inf")
                    train_parts = parts_before
                    accepted_train_traj = traj_before
                    lbfgs_loss_drop = float("-inf")
                    lbfgs_reject_reason = f"lbfgs_after_eval_exception:{err}"

                if lbfgs_reject_reason != "":
                    model.load_state_dict(copy.deepcopy(lbfgs_start_model_state))
                    mech_module.load_state_dict(copy.deepcopy(lbfgs_start_mech_state))
                    lbfgs_optimizer.load_state_dict(copy.deepcopy(lbfgs_start_opt_state))
                    _write_event(
                        event_log,
                        {
                            "event": "lbfgs_safety_stop",
                            "epoch": int(outer_epoch),
                            "lbfgs_outer_step": int(lbfgs_outer_step),
                            "closure_calls": int(closure_calls_total),
                            "loss_before": float(lbfgs_loss_before),
                            "loss_after": float(train_loss_after),
                            "reject_reason": lbfgs_reject_reason,
                            "action": "rollback_and_stop",
                        },
                    )
                    stop_epoch = int(outer_epoch)
                    stop_kind = "lbfgs_safety_stop"
                    stop_reason = lbfgs_reject_reason
                    _log_line(
                        log,
                        f"LBFGS stop -- safety reject at step={lbfgs_outer_step} epoch={outer_epoch} "
                        f"reason={lbfgs_reject_reason}; rollback parameters and stop",
                    )
                    break

                hard_zero_step = (
                    math.isfinite(float(lbfgs_param_delta_max_abs))
                    and float(lbfgs_param_delta_max_abs) <= float(cfg.lbfgs_tolerance_change)
                )
                if hard_zero_step:
                    if not lbfgs_zero_step_restart_used:
                        model.load_state_dict(copy.deepcopy(lbfgs_start_model_state))
                        mech_module.load_state_dict(copy.deepcopy(lbfgs_start_mech_state))
                        lbfgs_optimizer, lbfgs_line_search = _make_lbfgs_optimizer(cfg, model, mech_module)
                        lbfgs_zero_step_restart_used = True
                        _write_event(
                            event_log,
                            {
                                "event": "lbfgs_hard_zero_step_restart",
                                "epoch": int(outer_epoch),
                                "lbfgs_outer_step": int(lbfgs_outer_step),
                                "closure_calls": int(closure_calls_total),
                                "loss_before": float(lbfgs_loss_before),
                                "loss_after": float(train_loss_after),
                                "loss_drop": float(lbfgs_loss_drop),
                                "param_delta_l2": float(lbfgs_param_delta_l2),
                                "param_delta_max_abs": float(lbfgs_param_delta_max_abs),
                                "tolerance_change": float(cfg.lbfgs_tolerance_change),
                                "action": "rollback_fresh_lbfgs_optimizer_and_retry_same_epoch",
                            },
                        )
                        _log_line(
                            log,
                            "LBFGS hard zero-step detected -- "
                            f"step={lbfgs_outer_step} epoch={outer_epoch} "
                            f"param_delta_max_abs={lbfgs_param_delta_max_abs:.3e} "
                            f"<= tol_change={cfg.lbfgs_tolerance_change:.1e}; "
                            "rollback and fresh-restart LBFGS optimizer for same epoch",
                        )
                        continue

                    model.load_state_dict(copy.deepcopy(lbfgs_start_model_state))
                    mech_module.load_state_dict(copy.deepcopy(lbfgs_start_mech_state))
                    stop_epoch = int(outer_epoch)
                    stop_kind = "lbfgs_hard_zero_step_stop"
                    stop_reason = "lbfgs_hard_zero_step_after_fresh_restart"
                    _write_event(
                        event_log,
                        {
                            "event": stop_kind,
                            "epoch": int(outer_epoch),
                            "lbfgs_outer_step": int(lbfgs_outer_step),
                            "closure_calls": int(closure_calls_total),
                            "loss_before": float(lbfgs_loss_before),
                            "loss_after": float(train_loss_after),
                            "loss_drop": float(lbfgs_loss_drop),
                            "param_delta_l2": float(lbfgs_param_delta_l2),
                            "param_delta_max_abs": float(lbfgs_param_delta_max_abs),
                            "tolerance_change": float(cfg.lbfgs_tolerance_change),
                            "reject_reason": stop_reason,
                            "action": "rollback_and_stop",
                        },
                    )
                    _log_line(
                        log,
                        "LBFGS stop -- hard zero-step persisted after fresh restart "
                        f"| step={lbfgs_outer_step} epoch={outer_epoch} "
                        f"| param_delta_max_abs={lbfgs_param_delta_max_abs:.3e} "
                        f"<= tol_change={cfg.lbfgs_tolerance_change:.1e} "
                        f"| reason={stop_reason}; rollback parameters and stop",
                    )
                    break

                grad_norm = float(grad_norm_raw)
                eval_val_this_epoch = _should_eval_validation(cfg, outer_epoch)
                model.eval()
                try:
                    if not eval_val_this_epoch:
                        val_total = _nan_loss_tensor(dtype=dtype, device=cfg.device)
                        val_parts = _nan_loss_parts()
                    else:
                        with torch.no_grad():
                            val_total, val_parts, _ = evaluate_split(
                                force_module=model,
                                known_pars=prepared.known_pars,
                                mech_module=mech_module,
                                ode_true=tensors["ode_full"],
                                x2dot_true=tensors["x2dot_full"],
                                contact_mask=tensors["contact_full"],
                                times=tensors["times_full"],
                                ode_method=cfg.ode_method,
                                ode_rtol=cfg.ode_rtol,
                                ode_atol=cfg.ode_atol,
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
                x3_norm_monitor = _x3_norm_monitor_from_traj(
                    accepted_train_traj,
                    model,
                    x3_refit_meta,
                    x3_current_support=x3_agu_support_current,
                )
                x3_support_next = _x3_norm_support_from_monitor(x3_norm_monitor)
                row = {
                    "epoch": float(outer_epoch),
                    "train_loss": float(train_total.detach()),
                    "val_loss": float(val_total.detach()),
                    "validation_eval_mode": str(cfg.val_eval_mode),
                    "validation_evaluated": bool(eval_val_this_epoch),
                    "validation_is_final": bool(final_only_validation and eval_val_this_epoch),
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
                    "train_x3_range_amp": float(train_parts.x3_range_amp),
                    "val_x3_range_amp": float(val_parts.x3_range_amp),
                    "train_fts_range_amp": float(train_parts.fts_range_amp),
                    "val_fts_range_amp": float(val_parts.fts_range_amp),
                    "train_x1_rec": float(train_parts.x1_rec),
                    "val_x1_rec": float(val_parts.x1_rec),
                    "train_x2_rec": float(train_parts.x2_rec),
                    "val_x2_rec": float(val_parts.x2_rec),
                    "train_x2dot_rec": float(train_parts.x2dot_rec),
                    "val_x2dot_rec": float(val_parts.x2dot_rec),
                    "g_nn": float(model.gain().detach().cpu().item()),
                    "soft_mask_enabled": bool(soft_mask_meta["enabled"]),
                    "soft_mask_trainable": bool(soft_mask_meta["trainable"]),
                    "soft_mask_s0": float(soft_mask_meta["s0"]),
                    "soft_mask_s0_a0": float(soft_mask_meta["s0_a0"]),
                    "soft_mask_alpha": float(soft_mask_meta["alpha"]),
                    "soft_mask_alpha_a0": float(soft_mask_meta["alpha_a0"]),
                    "soft_mask_m_min": 0.0,
                    "grad_norm": float(grad_norm),
                    "grad_norm_raw": float(grad_norm_raw),
                    "grad_norm_nn": float(grad_norm_nn),
                    "grad_norm_mech": float(grad_norm_mech),
                    "lr": float(cfg.lbfgs_lr),
                    "step_effective_lr": float(cfg.lbfgs_lr),
                    "step_next_lr": float(cfg.lbfgs_lr),
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
                    "lbfgs_closure_calls": float(closure_calls_total),
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
                    "x3_agu_support_used_min": (
                        float(x3_agu_support_current[0])
                        if _valid_x3_norm_support(x3_agu_support_current)
                        else float("nan")
                    ),
                    "x3_agu_support_used_max": (
                        float(x3_agu_support_current[1])
                        if _valid_x3_norm_support(x3_agu_support_current)
                        else float("nan")
                    ),
                    "x3_agu_next_support_min": (
                        float(x3_support_next[0]) if _valid_x3_norm_support(x3_support_next) else float("nan")
                    ),
                    "x3_agu_next_support_max": (
                        float(x3_support_next[1]) if _valid_x3_norm_support(x3_support_next) else float("nan")
                    ),
                    "x3_agu_support_updated": False,
                }
                row.update(x3_norm_monitor)
                row.update(_x3_drift_from_epoch0(accepted_train_traj, x3_drift_epoch0_x3))
                ks_hat, cs_hat = _current_mech_numpy(mech_module)
                row["ks_hat"] = ks_hat
                row["cs_hat"] = cs_hat
                history.append(row)
                train_loss_last = float(train_total.detach())

                _write_event(
                    event_log,
                    {
                        "event": "lbfgs_outer_step",
                        "epoch": int(outer_epoch),
                        "lbfgs_outer_step": int(lbfgs_outer_step),
                        "closure_calls": int(closure_calls_total),
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
                    },
                )

                if final_only_validation and eval_val_this_epoch:
                    final_val_loss = float(val_total.detach())
                    final_val_epoch = int(outer_epoch)
                    final_checkpoint_snapshot = _snapshot_split(
                        force_module=model,
                        known_pars=prepared.known_pars,
                        mech_module=mech_module,
                        split=split,
                        window_mode=cfg.window_mode,
                        tensors=tensors,
                        dtype=dtype,
                        device=cfg.device,
                        ode_method=cfg.ode_method,
                        ode_rtol=cfg.ode_rtol,
                        ode_atol=cfg.ode_atol,
                    )
                    final_checkpoint_payload = {
                        "epoch": final_val_epoch,
                        "state_dict": copy.deepcopy(model.state_dict()),
                        "mech_state_dict": copy.deepcopy(mech_module.state_dict()),
                        "config": asdict(cfg),
                        "rng_state": capture_rng_state(),
                        "state_mean": prepared.state_mean,
                        "state_scale": prepared.state_scale,
                        "initial_grid_support": None if initial_grid_support is None else initial_grid_support.tolist(),
                        "initial_grid_support_meta": initial_grid_support_meta,
                        "x3_refit_meta": x3_refit_meta,
                        "x3_agu_support_current": _x3_norm_support_to_payload(x3_agu_support_current),
                        "x3_drift_epoch0_x3": None
                        if x3_drift_epoch0_x3 is None
                        else x3_drift_epoch0_x3.detach().cpu(),
                        "known_pars": prepared.known_pars,
                        "warmstart_source": None if warmstart_meta is None else str(warmstart_meta.get("source", "")),
                        "warmstart": warmstart_meta,
                        "stage1_warmstart": stage1_warmstart_meta,
                        "prestage2_warmstart": prestage2_warmstart_meta,
                        "resume_identity": _resume_identity(
                            shard_index=shard_index,
                            window_index=window_index,
                            split=split,
                            warmstart_meta=warmstart_meta,
                        ),
                        "window_meta": _window_meta_dict(split, cfg.window_mode),
                        "history": history,
                        "final_snapshot": final_checkpoint_snapshot,
                        "final_val_loss": float(final_val_loss),
                        "final_val_epoch": int(final_val_epoch),
                        "validation_eval_mode": str(cfg.val_eval_mode),
                    }
                    _save_torch(checkpoint_path, final_checkpoint_payload)
                    _save_pickle(checkpoint_viz_path, _make_viz_payload(final_checkpoint_payload))
                elif (not final_only_validation) and float(val_total.detach()) < best_val:
                    best_val = float(val_total.detach())
                    best_epoch = int(outer_epoch)
                    best_snapshot = _snapshot_split(
                        force_module=model,
                        known_pars=prepared.known_pars,
                        mech_module=mech_module,
                        split=split,
                        window_mode=cfg.window_mode,
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
                        "mech_state_dict": copy.deepcopy(mech_module.state_dict()),
                        "config": asdict(cfg),
                        "rng_state": capture_rng_state(),
                        "state_mean": prepared.state_mean,
                        "state_scale": prepared.state_scale,
                        "initial_grid_support": None if initial_grid_support is None else initial_grid_support.tolist(),
                        "initial_grid_support_meta": initial_grid_support_meta,
                        "x3_refit_meta": x3_refit_meta,
                        "x3_agu_support_current": _x3_norm_support_to_payload(x3_agu_support_current),
                        "x3_drift_epoch0_x3": None
                        if x3_drift_epoch0_x3 is None
                        else x3_drift_epoch0_x3.detach().cpu(),
                        "known_pars": prepared.known_pars,
                        "warmstart_source": None if warmstart_meta is None else str(warmstart_meta.get("source", "")),
                        "warmstart": warmstart_meta,
                        "stage1_warmstart": stage1_warmstart_meta,
                        "prestage2_warmstart": prestage2_warmstart_meta,
                        "resume_identity": _resume_identity(
                            shard_index=shard_index,
                            window_index=window_index,
                            split=split,
                            warmstart_meta=warmstart_meta,
                        ),
                        "window_meta": _window_meta_dict(split, cfg.window_mode),
                        "history": history,
                        "best_snapshot": best_snapshot,
                    }
                    _save_torch(checkpoint_path, best_payload)
                    _save_pickle(checkpoint_viz_path, _make_viz_payload(best_payload))

                _log_line(log, f"KAN LBFGS step {lbfgs_outer_step} epoch {outer_epoch} train={float(train_total.detach()):.6e}")
                _log_line(
                    log,
                    "  optimizer summary: "
                    f"phase={row['phase']} line_search={lbfgs_line_search} "
                    f"loss={lbfgs_loss_before:.6e}->{float(train_total.detach()):.6e} "
                    f"accepted={'YES' if step_update_accepted else 'NO'} "
                    f"closure_calls={closure_calls_total} "
                    f"returned_loss={lbfgs_returned_loss:.6e} "
                    f"reason={lbfgs_reject_reason or 'accepted'}",
                )
                _log_line(log, f"  grad_norm={grad_norm:.3e} lbfgs_lr={cfg.lbfgs_lr:.6g}")
                _log_line(log, f"  grad_norm_raw={grad_norm_raw:.3e} grad_norm_scaled={grad_norm:.3e}")
                _log_line(log, f"  grad_norm_nn={grad_norm_nn:.3e} grad_norm_mech={grad_norm_mech:.3e}")
                _log_line(
                    log,
                    "  x3 drift from epoch0: "
                    f"{float(row['x3_drift_from_epoch0_pct']):.3e}% "
                    f"rmse_abs={float(row['x3_drift_from_epoch0_abs']):.3e}",
                )
                _log_line(
                    log,
                    "  lbfgs progress: "
                    f"loss_drop={lbfgs_loss_drop:.3e} "
                    f"param_delta_l2={lbfgs_param_delta_l2:.3e} "
                    f"param_delta_max_abs={lbfgs_param_delta_max_abs:.3e}",
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
                _log_line(log, f"  timing: lbfgs_outer_step={epoch_sec:.3f}s AGU=frozen closure_calls={closure_calls_total}")
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
                _log_line(
                    log,
                    "  range amp: "
                    f"x3={float(train_parts.x3_range_amp):.6e} "
                    f"Fts={float(train_parts.fts_range_amp):.6e} "
                    "source=mean_abs_observed_x1_and_F_actuation_current_window",
                )
                _log_line(
                    log,
                    "  rec: "
                    f"x1={float(train_parts.x1_rec):.2f}% "
                    f"x2={float(train_parts.x2_rec):.2f}% "
                    f"x2dot={float(train_parts.x2dot_rec):.2f}%",
                )
                _log_line(log, f"  {_format_x3_norm_monitor(x3_norm_monitor)}")
                _log_line(
                    log,
                    "  mech: "
                    f"ks={row['ks_hat']:.6e} "
                    f"cs={row['cs_hat']:.6e}",
                )
                if eval_val_this_epoch:
                    _log_line(log, f"KAN final val epoch {outer_epoch} val={float(val_total.detach()):.6e}" if final_only_validation else f"KAN val epoch {outer_epoch} val={float(val_total.detach()):.6e}")
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
                    _log_line(
                        log,
                        "  val range amp: "
                        f"x3={float(val_parts.x3_range_amp):.6e} "
                        f"Fts={float(val_parts.fts_range_amp):.6e} "
                        "source=mean_abs_observed_x1_and_F_actuation_current_window",
                    )
                    _log_line(
                        log,
                        "  val rec: "
                        f"x1={float(val_parts.x1_rec):.2f}% "
                        f"x2={float(val_parts.x2_rec):.2f}% "
                        f"x2dot={float(val_parts.x2dot_rec):.2f}%",
                    )

                if (int(outer_epoch) % cfg.checkpoint_every) == 0 or int(outer_epoch) == int(planned_total_epochs):
                    payload = {
                        "epoch": int(outer_epoch),
                        "history": history,
                        "config": asdict(cfg),
                        "state_dict": copy.deepcopy(model.state_dict()),
                        "mech_state_dict": copy.deepcopy(mech_module.state_dict()),
                        "optimizer_state_dict": copy.deepcopy(lbfgs_optimizer.state_dict()),
                        "optimizer_phase": "lbfgs",
                        "rng_state": capture_rng_state(),
                        "state_mean": prepared.state_mean,
                        "state_scale": prepared.state_scale,
                        "initial_grid_support": None if initial_grid_support is None else initial_grid_support.tolist(),
                        "initial_grid_support_meta": initial_grid_support_meta,
                        "x3_refit_meta": x3_refit_meta,
                        "x3_agu_support_current": _x3_norm_support_to_payload(x3_agu_support_current),
                        "x3_drift_epoch0_x3": None
                        if x3_drift_epoch0_x3 is None
                        else x3_drift_epoch0_x3.detach().cpu(),
                        "known_pars": prepared.known_pars,
                        "warmstart_source": None if warmstart_meta is None else str(warmstart_meta.get("source", "")),
                        "warmstart": warmstart_meta,
                        "stage1_warmstart": stage1_warmstart_meta,
                        "prestage2_warmstart": prestage2_warmstart_meta,
                        "validation_eval_mode": str(cfg.val_eval_mode),
                        "best_epoch": int(best_epoch),
                        "best_val_loss": float(best_val),
                        "final_val_epoch": int(final_val_epoch),
                        "final_val_loss": float(final_val_loss),
                        "lr": float(cfg.lbfgs_lr),
                        "grad_ema": float(grad_ema),
                        "grad_target": float(grad_target),
                        "recent_losses": list(recent_losses),
                        "resume_identity": _resume_identity(
                            shard_index=shard_index,
                            window_index=window_index,
                            split=split,
                            warmstart_meta=warmstart_meta,
                        ),
                        "window_meta": _window_meta_dict(split, cfg.window_mode),
                    }
                    _save_torch(running_checkpoint_path, payload)
                    _log_line(log, f"checkpoint saved | epoch={outer_epoch} | path={running_checkpoint_path}")

                lbfgs_zero_step_restart_used = False
                lbfgs_outer_step += 1

        if failure_reason != "":
            _log_line(log, f"KAN shard failure -- epoch={failure_epoch} reason={failure_reason}")
        if stop_reason != "":
            _log_line(log, f"KAN shard stopped -- epoch={stop_epoch} kind={stop_kind} reason={stop_reason}")

        final_snapshot = _snapshot_split(
            force_module=model,
            known_pars=prepared.known_pars,
            mech_module=mech_module,
            split=split,
            window_mode=cfg.window_mode,
            tensors=tensors,
            dtype=dtype,
            device=cfg.device,
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
        )
        final_val_metrics = final_snapshot.get("val", {}).get("metrics", {})
        if isinstance(final_val_metrics, dict):
            final_val_loss = float(final_val_metrics.get("loss", final_val_loss))
            final_val_epoch = int(history[-1].get("epoch", len(history))) if history else -1
            if final_only_validation and history:
                final_row = history[-1]
                final_row["val_loss"] = float(final_val_loss)
                final_row["validation_eval_mode"] = str(cfg.val_eval_mode)
                final_row["validation_evaluated"] = True
                final_row["validation_is_final"] = True
                final_row["val_state"] = float(final_val_metrics.get("state", float("nan")))
                final_row["val_x1_state"] = float(final_val_metrics.get("x1_state", float("nan")))
                final_row["val_x2_state"] = float(final_val_metrics.get("x2_state", float("nan")))
                final_row["val_x2dot"] = float(final_val_metrics.get("x2dot", float("nan")))
                final_row["val_x3_range"] = float(final_val_metrics.get("x3_range", float("nan")))
                final_row["val_fts_range"] = float(final_val_metrics.get("fts_range", float("nan")))
                final_row["val_x3_range_amp"] = float(final_val_metrics.get("x3_range_amp", float("nan")))
                final_row["val_fts_range_amp"] = float(final_val_metrics.get("fts_range_amp", float("nan")))
                final_row["val_x1_rec"] = float(final_val_metrics.get("x1_rec", float("nan")))
                final_row["val_x2_rec"] = float(final_val_metrics.get("x2_rec", float("nan")))
                final_row["val_x2dot_rec"] = float(final_val_metrics.get("x2dot_rec", float("nan")))
        if final_only_validation:
            final_checkpoint_payload = {
                "epoch": int(final_val_epoch),
                "state_dict": copy.deepcopy(model.state_dict()),
                "mech_state_dict": copy.deepcopy(mech_module.state_dict()),
                "config": asdict(cfg),
                "rng_state": capture_rng_state(),
                "state_mean": prepared.state_mean,
                "state_scale": prepared.state_scale,
                "initial_grid_support": None if initial_grid_support is None else initial_grid_support.tolist(),
                "initial_grid_support_meta": initial_grid_support_meta,
                "x3_refit_meta": x3_refit_meta,
                "x3_agu_support_current": _x3_norm_support_to_payload(x3_agu_support_current),
                "x3_drift_epoch0_x3": None
                if x3_drift_epoch0_x3 is None
                else x3_drift_epoch0_x3.detach().cpu(),
                "known_pars": prepared.known_pars,
                "warmstart_source": None if warmstart_meta is None else str(warmstart_meta.get("source", "")),
                "warmstart": warmstart_meta,
                "stage1_warmstart": stage1_warmstart_meta,
                "prestage2_warmstart": prestage2_warmstart_meta,
                "resume_identity": _resume_identity(
                    shard_index=shard_index,
                    window_index=window_index,
                    split=split,
                    warmstart_meta=warmstart_meta,
                ),
                "window_meta": _window_meta_dict(split, cfg.window_mode),
                "history": history,
                "final_snapshot": final_snapshot,
                "final_val_loss": float(final_val_loss),
                "final_val_epoch": int(final_val_epoch),
                "validation_eval_mode": str(cfg.val_eval_mode),
            }
            _save_torch(checkpoint_path, final_checkpoint_payload)
            _save_pickle(checkpoint_viz_path, _make_viz_payload(final_checkpoint_payload))
        final_payload = {
            "history": history,
            "state_mean": prepared.state_mean,
            "state_scale": prepared.state_scale,
            "initial_grid_support": None if initial_grid_support is None else initial_grid_support.tolist(),
            "initial_grid_support_meta": initial_grid_support_meta,
            "x3_refit_meta": x3_refit_meta,
            "x3_agu_support_current": _x3_norm_support_to_payload(x3_agu_support_current),
            "x3_drift_epoch0_x3": None
            if x3_drift_epoch0_x3 is None
            else x3_drift_epoch0_x3.detach().cpu(),
            "known_pars": prepared.known_pars,
            "warmstart_source": None if warmstart_meta is None else str(warmstart_meta.get("source", "")),
            "warmstart": warmstart_meta,
            "stage1_warmstart": stage1_warmstart_meta,
            "prestage2_warmstart": prestage2_warmstart_meta,
            "resume_identity": _resume_identity(
                shard_index=shard_index,
                window_index=window_index,
                split=split,
                warmstart_meta=warmstart_meta,
            ),
            "window_meta": _window_meta_dict(split, cfg.window_mode),
            "final_snapshot": final_snapshot,
            "config": asdict(cfg),
            "rng_state": capture_rng_state(),
            "validation_eval_mode": str(cfg.val_eval_mode),
            "final_val_epoch": int(final_val_epoch),
            "final_val_loss": float(final_val_loss),
            "final_state_dict": copy.deepcopy(model.state_dict()),
            "final_mech_state_dict": copy.deepcopy(mech_module.state_dict()),
            "optimizer_event_log_path": str(event_log_path),
            "stop_kind": stop_kind,
            "stop_epoch": int(stop_epoch),
            "stop_reason": stop_reason,
            "failure_reason": failure_reason,
            "failure_epoch": int(failure_epoch),
        }
        if not final_only_validation:
            final_payload.update(
                {
                    "best": best_payload,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                }
            )
        _save_torch(result_path, final_payload)
        _save_pickle(viz_result_path, _make_viz_payload(final_payload))
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if final_only_validation:
            _log_line(
                log,
                f"STAGE2LIGHT shard done -- train={history[-1]['train_loss']:.6e} final_val={final_val_loss:.6e} "
                f"| final_val_epoch={final_val_epoch}",
            )
        else:
            _log_line(
                log,
                f"STAGE2LIGHT shard done -- train={history[-1]['train_loss']:.6e} val={history[-1]['val_loss']:.6e} "
                f"| best_epoch={best_epoch} best_val={best_val:.6e}",
            )

    result_summary = {
        "shard_index": shard_index,
        "role": split.role,
        "label": split.label,
        "log_path": str(log_path),
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path),
        "result_path": str(result_path),
        "validation_eval_mode": str(cfg.val_eval_mode),
        "final_val_epoch": int(final_val_epoch),
        "final_val_loss": float(final_val_loss),
    }
    if not final_only_validation:
        result_summary.update(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
            }
        )
    return result_summary


def merge_stage2light_results(cfg=None) -> dict[str, Any]:
    cfg = default_config() if cfg is None else cfg
    merged: list[dict[str, Any]] = []
    for shard_index in range(1, cfg.shard_count + 1):
        result_path = cfg.result_dir / f"stage2light_result_p{shard_index}.pt"
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
    summary_path = cfg.result_dir / "stage2light_merged_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_json_default)
    return out


def run_stage2light(cfg=None) -> dict[str, object]:
    return run_stage2light_shard(cfg)


run_full_test_shard = run_stage2light_shard
run_full_test = run_stage2light
merge_shard_results = merge_stage2light_results


__all__ = [
    "merge_shard_results",
    "merge_stage2light_results",
    "run_full_test",
    "run_full_test_shard",
    "run_stage2light",
    "run_stage2light_shard",
]
