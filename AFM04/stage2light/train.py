"""Training loop for the standalone AFM04 Python stage2light library."""

from __future__ import annotations

import copy
import json
import math
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from AFM04.stage2light.config import default_config
from AFM04.stage2light.data import PreparedData, Stage1WarmstartCandidate, WindowSplit, prepare_data, select_stage1_candidate
from AFM04.stage2light.kan_backend import KANForceModule, summarize_wpred_from_states
from AFM04.stage2light.losses import TorchLossParts, evaluate_split
from AFM04.stage2light.rollout import LearnableMechModule, fts_truth_from_states_torch, x2dot_rhs_torch


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
    window: int,
) -> dict[str, float] | None:
    if len(history) < window + 1:
        return None
    recent = history[-(window + 1) :]
    prev_vals = [float(row["val_loss"]) for row in recent[:-1]]
    cur = float(recent[-1]["val_loss"])
    if not np.isfinite(cur) or cur <= 0.0:
        return None
    if any((not np.isfinite(v)) for v in prev_vals):
        return None
    best_window = min(prev_vals + [cur])
    if cur > best_window:
        return None
    max_prev = max(prev_vals)
    rel_gap_to_current = (max_prev - cur) / cur
    return {
        "current": cur,
        "max_prev": max_prev,
        "best_window": best_window,
        "rel_gap_to_current": rel_gap_to_current,
    }


def _window_title(role: str) -> str:
    role_norm = role.strip().lower()
    if role_norm == "first_contact":
        return "window1: right after first contact"
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
    }


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
        "fts_teacher_rec": float(parts.fts_teacher_rec),
    }


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


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


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


def _role_short(role: str) -> str:
    role_norm = role.strip().lower()
    if role_norm == "first_contact":
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

    # Default stage2light behavior trains only on W1 when the standard
    # stage2 window manifest is available. If that canonical role is missing,
    # fall back to all available windows.
    role_to_index = {split.role.strip().lower(): idx for idx, split in enumerate(splits, start=1)}
    preferred = [role_to_index[role] for role in ("first_contact",) if role in role_to_index]
    if len(preferred) == 1:
        return preferred
    return list(range(1, len(splits) + 1))


def _geometric_midpoint(lo: float, hi: float) -> float:
    return float(math.sqrt(max(lo, 1.0e-30) * max(hi, 1.0e-30)))


def _current_mech_numpy(mech_module: LearnableMechModule) -> tuple[float, float]:
    mech = mech_module().detach().cpu().numpy()
    return float(mech[0]), float(mech[1])


def _mech_err_pct(est: float, truth: float) -> float:
    return 100.0 * abs(float(est) - float(truth)) / max(abs(float(truth)), 1.0e-30)


def _make_grid_update_states(
    *,
    cfg,
    traj_pred: torch.Tensor,
    known_pars: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, float]]:
    train_wpred_enabled = bool(getattr(cfg, "train_wpred_enabled", getattr(cfg, "wpred_enabled", False)))
    pred_states = traj_pred.transpose(0, 1).detach()
    n = float(pred_states.shape[0])
    meta = {
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
    if not train_wpred_enabled:
        return pred_states, meta
    dist = float(known_pars[6])
    a0 = float(known_pars[9])
    meta.update(
        summarize_wpred_from_states(
            pred_states,
            dist=dist,
            a0=a0,
            eps=float(cfg.wpred_eps),
        )
    )
    return pred_states, meta


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
    _log_line(log, "=== STAGE2LIGHT SANITY (oracle F_contact + true mech) ===")
    oracle = _OracleForceModule(
        known_pars,
        eta_star_true=eta_star_true,
        mech_true=mech_true,
        dtype=dtype,
        device=device,
    )
    mech_oracle = LearnableMechModule(
        ks_init=float(mech_true[0].detach().cpu().item()),
        cs_init=float(mech_true[1].detach().cpu().item()),
        ks_bounds=(float(mech_true[0].detach().cpu().item()), float(mech_true[0].detach().cpu().item()) + 1.0e-12),
        cs_bounds=(float(mech_true[1].detach().cpu().item()), float(mech_true[1].detach().cpu().item()) + 1.0e-18),
        dtype=dtype,
        device=device,
    )
    for param in mech_oracle.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        sanity_loss, sanity_parts, _ = evaluate_split(
            force_module=oracle,
            known_pars=known_pars,
            mech_module=mech_oracle,
            ode_true=tensors["ode_train"],
            x2dot_true=tensors["x2dot_train"],
            contact_mask=tensors["contact_train"],
            times=tensors["times_train"],
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            mech_true=mech_true,
            eta_star_true=eta_star_true,
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
            "fts_teacher_rec": float(sanity_parts.fts_teacher_rec),
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
    mech_module: LearnableMechModule,
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
            ode_true = tensors[f"ode_{part_name}"]
            x2dot_true = tensors[f"x2dot_{part_name}"]
            contact_mask = tensors[f"contact_{part_name}"]
            times = tensors[f"times_{part_name}"]
            total, parts, traj = evaluate_split(
                force_module=force_module,
                known_pars=known_pars,
                mech_module=mech_module,
                ode_true=ode_true,
                x2dot_true=x2dot_true,
                contact_mask=contact_mask,
                times=times,
                ode_method=ode_method,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                mech_true=mech_true,
                eta_star_true=eta_star_true,
            )
            x2dot_pred = x2dot_rhs_torch(traj, times, force_module, known_pars)
            teacher_states = ode_true.transpose(0, 1)
            fts_teacher_true = fts_truth_from_states_torch(
                teacher_states,
                known_pars,
                eta_star=eta_star_true,
                mech_true=mech_true,
            )
            fts_teacher_pred = force_module(teacher_states)
            rollout_states = traj.transpose(0, 1)
            fts_rollout_true = fts_truth_from_states_torch(
                rollout_states,
                known_pars,
                eta_star=eta_star_true,
                mech_true=mech_true,
            )
            fts_rollout_pred = force_module(rollout_states)
            nn_raw_rollout = force_module.raw_output(rollout_states)
            w_pred_rollout = force_module.w_pred(rollout_states)
            nn_weighted_rollout = force_module.weighted_raw_output(rollout_states)
            out[part_name] = {
                "metrics": _metrics_row(total, parts),
                "times": times.detach().cpu().numpy(),
                "ode_true": ode_true.detach().cpu().numpy(),
                "traj_pred": traj.detach().cpu().numpy(),
                "x2dot_true": x2dot_true.detach().cpu().numpy(),
                "x2dot_pred": x2dot_pred.detach().cpu().numpy(),
                "contact_mask": contact_mask.detach().cpu().numpy(),
                "fts_teacher_true": fts_teacher_true.detach().cpu().numpy(),
                "fts_teacher_pred": fts_teacher_pred.detach().cpu().numpy(),
                "fts_rollout_true": fts_rollout_true.detach().cpu().numpy(),
                "fts_rollout_pred": fts_rollout_pred.detach().cpu().numpy(),
                "nn_raw_rollout": nn_raw_rollout.detach().cpu().numpy(),
                "w_pred_rollout": w_pred_rollout.detach().cpu().numpy(),
                "nn_weighted_rollout": nn_weighted_rollout.detach().cpu().numpy(),
            }
        out["g_nn"] = float(force_module.gain().detach().cpu().item())
        mech_pred = mech_module().detach().cpu().numpy()
        out["mech_pred"] = mech_pred
        out["mech_true"] = mech_true.detach().cpu().numpy()
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


def _maybe_load_stage1_warmstart(
    *,
    cfg,
    log,
) -> Stage1WarmstartCandidate | None:
    if not getattr(cfg, "warmstart_from_stage1", False):
        return None
    best_path = cfg.stage1_input_path
    if not best_path.is_file():
        if getattr(cfg, "warmstart_fallback_random", False):
            if log is not None:
                _log_line(
                    log,
                    "warmstart from stage1pluslight: missing input, fallback to random init | "
                    f"path={best_path}",
                )
            return None
        raise FileNotFoundError(f"Missing stage1pluslight warmstart file: {best_path}")
    candidate = select_stage1_candidate(best_path, rank=int(cfg.stage1_input_rank))
    if log is not None:
        _log_line(
            log,
            "warmstart from stage1pluslight: "
            f"path={best_path} rank={candidate.rank} "
            f"trial={candidate.trial_id} "
            f"loss={candidate.ranking_loss:.6e} "
            f"seed={candidate.init_seed} "
            f"ks0={candidate.ks0:.6e} cs0={candidate.cs0:.6e}",
        )
    return candidate


def _make_viz_payload(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("best")
    best_viz = None
    best_epoch = payload.get("best_epoch")
    best_val_loss = payload.get("best_val_loss")
    if isinstance(best, dict):
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
    return {
        "best": best_viz,
        "history": payload.get("history", []),
        "state_mean": payload.get("state_mean"),
        "state_scale": payload.get("state_scale"),
        "known_pars": payload.get("known_pars"),
        "eta_star_true": payload.get("eta_star_true"),
        "mech_true": payload.get("mech_true"),
        "stage1_warmstart": payload.get("stage1_warmstart"),
        "sanity": payload.get("sanity"),
        "window_meta": payload.get("window_meta"),
        "final_snapshot": payload.get("final_snapshot"),
        "config": payload.get("config"),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def run_stage2light_shard(cfg=None, *, shard_index: int | None = None, log_path: Path | None = None) -> dict[str, object]:
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

    train_states = torch.as_tensor(split.ode_train.T, dtype=dtype, device=cfg.device)
    train_fts = torch.as_tensor(split.fts_train_true, dtype=dtype, device=cfg.device)
    role_short = _role_short(split.role)
    history_path = cfg.result_dir / f"stage2light_history_p{shard_index}.json"
    checkpoint_path = cfg.checkpoint_dir / f"stage2light_best_p{shard_index}.pt"
    checkpoint_viz_path = cfg.checkpoint_dir / f"stage2light_best_p{shard_index}.viz.pkl"
    running_checkpoint_path = cfg.checkpoint_dir / f"stage2light_checkpoint_p{shard_index}.pt"
    result_path = cfg.result_dir / f"stage2light_result_p{shard_index}.pt"
    viz_result_path = cfg.result_dir / f"stage2light_result_p{shard_index}.viz.pkl"
    log_path = cfg.shard_log_dir / f"log2_04_step2a_stage2light_local_p{shard_index}.txt" if log_path is None else Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    best_payload: dict[str, Any] | None = None
    sanity_payload: dict[str, Any] | None = None
    grad_ema = float(cfg.lr_target_init)
    grad_target = float(cfg.lr_target_init)
    recent_losses: list[float] = []
    failure_reason = ""
    failure_epoch = 0
    train_loss_last = float("inf")
    cached_grid_traj: torch.Tensor | None = None

    with log_path.open("w", encoding="utf-8") as log:
        stage1_warmstart = _maybe_load_stage1_warmstart(cfg=cfg, log=log)
        model_seed = int(stage1_warmstart.init_seed) if stage1_warmstart is not None else int(cfg.seed)
        ks_init = float(stage1_warmstart.ks0) if stage1_warmstart is not None else _geometric_midpoint(cfg.ks_lo, cfg.ks_hi)
        cs_init = float(stage1_warmstart.cs0) if stage1_warmstart is not None else _geometric_midpoint(cfg.cs_lo, cfg.cs_hi)
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
            dist=float(prepared.known_pars[6]),
            a0=float(prepared.known_pars[9]),
            wpred_enabled=train_wpred_enabled,
            wpred_eps=cfg.wpred_eps,
            gnn_learnable=cfg.gnn_learnable,
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
        ).to(cfg.device)
        init_gain = model.initialize_gain_from_truth(train_states, train_fts)
        optimizer, optimizer_name = _make_optimizer(cfg, model, mech_module)
        lr = float(cfg.lr)
        _set_optimizer_lr(optimizer, lr)

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
        _log_line(log, "=== AFM04 STAGE2LIGHT ===")
        _log_line(
            log,
            "Switches: "
            f"adaptive_grid={str(cfg.adaptive_grid_enabled).upper()} | "
            f"warmstart_from_stage1={'ON' if cfg.warmstart_from_stage1 else 'OFF'} | "
            f"warmstart_fallback_random={'ON' if cfg.warmstart_fallback_random else 'OFF'} | "
            "random_init=ON | "
            f"symbolic={str(cfg.symbolic_enabled).upper()} | auto_save={str(cfg.auto_save).upper()}",
        )
        _log_line(log, f"Solver: torchdiffeq odeint | method={cfg.ode_method} | differentiable=ON")
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
        _log_line(
            log,
            "contact gate: "
            f"w_pred={'ON' if train_wpred_enabled else 'OFF'} "
            f"eps={cfg.wpred_eps:.2e} "
            "target=(g_nn * w_pred * nn_raw)",
        )
        _log_line(
            log,
            "mech init: "
            f"ks0={ks_init:.6e} cs0={cs_init:.6e} "
            f"| truth_ks={prepared.mech_true[0]:.6e} truth_cs={prepared.mech_true[1]:.6e}",
        )
        _log_line(
            log,
            f"optimizer={optimizer_name.upper()} | amsgrad={'ON' if optimizer_name == 'amsgrad' else 'OFF'} | "
            f"lr={lr:.2e} weight_decay={cfg.weight_decay:.2e}",
        )
        _log_line(
            log,
            "optimizer controls: "
            f"lr_adapt={'ON' if cfg.lr_adapt else 'OFF'} "
            f"step_guard={'ON' if cfg.step_guard_enabled else 'OFF'} "
            f"plateau_early_stop={'ON' if cfg.plateau_early_stop else 'OFF'} "
            f"recent_val_early_stop={'ON' if cfg.recent_val_early_stop else 'OFF'} "
            f"good_enough={cfg.good_enough_loss:.1e}",
        )
        _log_line(
            log,
            "optimizer bounds: "
            f"lr=[{cfg.lr_min:.2e}, {cfg.lr_max:.2e}] "
            f"retry(epoch={cfg.epoch_retry_max}, step={cfg.step_retry_max}) "
            f"step_jump_frac={cfg.step_max_loss_increase_frac:.3f}",
        )
        _log_line(
            log,
            "early-stop config: "
            f"recent_val_window={cfg.recent_val_window} "
            f"recent_val_rel_current_frac={cfg.recent_val_rel_current_frac:.4f}",
        )
        _log_line(log, "optimizer grouping: single_group=ON | group_adapt=OFF | mech_joint=ON")
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

        for epoch in range(cfg.epochs):
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
            step_retry_count = 0
            step_accept_reason = ""
            epoch_fail_reason = ""

            while epoch_attempt < epoch_attempt_max:
                epoch_attempt += 1
                model.load_state_dict(copy.deepcopy(epoch_start_state))
                mech_module.load_state_dict(copy.deepcopy(epoch_start_mech_state))
                optimizer.load_state_dict(copy.deepcopy(epoch_start_opt_state))
                _set_optimizer_lr(optimizer, lr)
                model.train()

                if cfg.adaptive_grid_enabled and _grid_update_due(epoch, cfg.grid_update_num, cfg.start_grid_update_step, cfg.stop_grid_update_step):
                    try:
                        grid_update_t0 = perf_counter()
                        with torch.no_grad():
                            grid_source = "cached_pred"
                            grid_traj = cached_grid_traj
                            if grid_traj is None:
                                model.eval()
                                _, _, grid_traj = evaluate_split(
                                    force_module=model,
                                    known_pars=prepared.known_pars,
                                    mech_module=mech_module,
                                    ode_true=tensors["ode_train"],
                                    x2dot_true=tensors["x2dot_train"],
                                    contact_mask=tensors["contact_train"],
                                    times=tensors["times_train"],
                                    ode_method=cfg.ode_method,
                                    ode_rtol=cfg.ode_rtol,
                                    ode_atol=cfg.ode_atol,
                                    mech_true=mech_true_t,
                                    eta_star_true=prepared.eta_star_true,
                                )
                                model.train()
                                grid_source = "warmup_pred"
                            grid_states, grid_meta = _make_grid_update_states(
                                cfg=cfg,
                                traj_pred=grid_traj,
                                known_pars=prepared.known_pars,
                            )
                            model.update_grid_from_states(grid_states)
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
                try:
                    train_total, train_parts, train_traj = evaluate_split(
                        force_module=model,
                        known_pars=prepared.known_pars,
                        mech_module=mech_module,
                        ode_true=tensors["ode_train"],
                        x2dot_true=tensors["x2dot_train"],
                        contact_mask=tensors["contact_train"],
                        times=tensors["times_train"],
                        ode_method=cfg.ode_method,
                        ode_rtol=cfg.ode_rtol,
                        ode_atol=cfg.ode_atol,
                        mech_true=mech_true_t,
                        eta_star_true=prepared.eta_star_true,
                    )
                    if not torch.isfinite(train_total):
                        epoch_fail_reason = "train_loss_nonfinite"
                    else:
                        train_total.backward()
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

                prev_loss_ref = train_loss_last if np.isfinite(train_loss_last) else float("nan")
                train_loss_before = float(train_total.detach())
                model_base_state = copy.deepcopy(model.state_dict())
                mech_base_state = copy.deepcopy(mech_module.state_dict())
                opt_base_state = copy.deepcopy(optimizer.state_dict())
                grad_cache = _capture_grads(("model", model), ("mech", mech_module))
                step_retry_count = 0
                step_accept_reason = "accepted"
                step_accepted = False

                attempts_total = 1 if not cfg.step_guard_enabled else max(1, int(cfg.step_retry_max) + 1)
                for step_attempt in range(1, attempts_total + 1):
                    model.load_state_dict(copy.deepcopy(model_base_state))
                    mech_module.load_state_dict(copy.deepcopy(mech_base_state))
                    optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                    _set_optimizer_lr(optimizer, lr)
                    _restore_grads(grad_cache, ("model", model), ("mech", mech_module))
                    optimizer.step()
                    model.train()
                    try:
                        trial_total, trial_parts, trial_traj = evaluate_split(
                            force_module=model,
                            known_pars=prepared.known_pars,
                            mech_module=mech_module,
                            ode_true=tensors["ode_train"],
                            x2dot_true=tensors["x2dot_train"],
                            contact_mask=tensors["contact_train"],
                            times=tensors["times_train"],
                            ode_method=cfg.ode_method,
                            ode_rtol=cfg.ode_rtol,
                            ode_atol=cfg.ode_atol,
                            mech_true=mech_true_t,
                            eta_star_true=prepared.eta_star_true,
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

                    if step_accept_reason == "accepted":
                        train_total = trial_total
                        train_parts = trial_parts
                        accepted_train_traj = trial_traj
                        step_accepted = True
                        step_retry_count = step_attempt - 1
                        if step_retry_count > 0:
                            _log_line(
                                log,
                                f"step-guard: accepted after {step_retry_count} retry/reduction(s) "
                                f"| lr={lr:.3e} | train={trial_loss:.6e}",
                            )
                        break

                    if step_attempt >= attempts_total:
                        step_retry_count = max(0, attempts_total - 1)
                        if _terminal_step_failure(step_accept_reason):
                            model.load_state_dict(copy.deepcopy(model_base_state))
                            mech_module.load_state_dict(copy.deepcopy(mech_base_state))
                            optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                            _set_optimizer_lr(optimizer, lr)
                            failure_reason = f"trial_failed:{step_accept_reason}"
                            failure_epoch = epoch + 1
                            break
                        model.load_state_dict(copy.deepcopy(model_base_state))
                        mech_module.load_state_dict(copy.deepcopy(mech_base_state))
                        optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                        _set_optimizer_lr(optimizer, lr)
                        train_total = torch.as_tensor(train_loss_before, dtype=dtype, device=cfg.device)
                        accepted_train_traj = train_traj
                        step_accept_reason = "step_rejected_keep_previous"
                        _log_line(
                            log,
                            f"step-guard: rejected update after {step_retry_count} retries; keep previous parameters "
                            f"| reason={step_accept_reason} | lr={lr:.3e}",
                        )
                        step_accepted = True
                        break

                    lr = max(lr * cfg.step_retry_lr_factor, cfg.epoch_retry_lr_floor)
                    step_retry_count = step_attempt
                    _log_line(
                        log,
                        f"step-guard retry {step_attempt}/{cfg.step_retry_max} -- reason={step_accept_reason} | lr={lr:.3e}",
                    )
                    step_accept_reason = "accepted"

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
                        mech_module=mech_module,
                        ode_true=tensors["ode_val"],
                        x2dot_true=tensors["x2dot_val"],
                        contact_mask=tensors["contact_val"],
                        times=tensors["times_val"],
                        ode_method=cfg.ode_method,
                        ode_rtol=cfg.ode_rtol,
                        ode_atol=cfg.ode_atol,
                        mech_true=mech_true_t,
                        eta_star_true=prepared.eta_star_true,
                    )
                    if not torch.isfinite(val_total):
                        failure_reason = "val_loss_nonfinite"
                        failure_epoch = epoch + 1
                        break
            except Exception as err:
                failure_reason = f"val_exception:{err}"
                failure_epoch = epoch + 1
                break

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
                "train_fts_teacher_rec": float(train_parts.fts_teacher_rec),
                "val_fts_teacher_rec": float(val_parts.fts_teacher_rec),
                "g_nn": float(model.gain().detach().cpu().item()),
                "grad_norm": float(grad_norm),
                "grad_norm_raw": float(grad_norm_raw),
                "lr": float(lr),
                "epoch_retry_count": float(max(0, epoch_attempt - 1)),
                "step_retry_count": float(step_retry_count),
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
            ks_hat, cs_hat = _current_mech_numpy(mech_module)
            row["ks_hat"] = ks_hat
            row["cs_hat"] = cs_hat
            row["ks_err_pct"] = _mech_err_pct(ks_hat, float(prepared.mech_true[0]))
            row["cs_err_pct"] = _mech_err_pct(cs_hat, float(prepared.mech_true[1]))
            history.append(row)
            train_loss_last = float(train_total.detach())
            if accepted_train_traj is not None:
                cached_grid_traj = accepted_train_traj.detach().clone()

            if float(val_total.detach()) < best_val:
                best_val = float(val_total.detach())
                best_epoch = epoch + 1
                best_snapshot = _snapshot_split(
                    force_module=model,
                    known_pars=prepared.known_pars,
                    eta_star_true=prepared.eta_star_true,
                    mech_module=mech_module,
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
                    "mech_state_dict": copy.deepcopy(mech_module.state_dict()),
                    "config": asdict(cfg),
                    "state_mean": prepared.state_mean,
                    "state_scale": prepared.state_scale,
                    "known_pars": prepared.known_pars,
                    "eta_star_true": prepared.eta_star_true,
                    "mech_true": prepared.mech_true,
                    "stage1_warmstart": None if stage1_warmstart is None else {
                        "path": str(stage1_warmstart.path),
                        "rank": int(stage1_warmstart.rank),
                        "trial_id": int(stage1_warmstart.trial_id),
                        "loss": float(stage1_warmstart.ranking_loss),
                        "seed": int(stage1_warmstart.init_seed),
                        "ks0": float(stage1_warmstart.ks0),
                        "cs0": float(stage1_warmstart.cs0),
                    },
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
                _save_pickle(checkpoint_viz_path, _make_viz_payload(best_payload))

            if ((epoch + 1) % cfg.log_every) == 0 or epoch == 0 or epoch + 1 == cfg.epochs:
                _log_line(log, f"KAN epoch {epoch + 1} train={float(train_total.detach()):.6e}")
                _log_line(log, f"  grad_norm={grad_norm:.3e} lr={lr:.6g}")
                _log_line(log, f"  grad_norm_raw={grad_norm_raw:.3e} grad_norm_scaled={grad_norm:.3e}")
                if recovered:
                    _log_line(log, f"  retry: recovered_after={epoch_attempt - 1} rollback(s)")
                if step_retry_count > 0 and step_accept_reason == "accepted":
                    _log_line(log, f"  step-guard: accepted_after={step_retry_count} retry/reduction(s)")
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
                _log_line(log, f"  nn: F_contact err={float(train_parts.fts_teacher_rec):.2f}%")
                _log_line(
                    log,
                    "  mech: "
                    f"ks={row['ks_hat']:.6e} ({row['ks_err_pct']:.2f}%) "
                    f"cs={row['cs_hat']:.6e} ({row['cs_err_pct']:.2f}%)",
                )
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
                _log_line(log, f"  val nn: F_contact err={float(val_parts.fts_teacher_rec):.2f}%")

            if ((epoch + 1) % cfg.checkpoint_every) == 0:
                payload = {
                    "epoch": int(epoch + 1),
                    "history": history,
                    "config": asdict(cfg),
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "mech_state_dict": copy.deepcopy(mech_module.state_dict()),
                    "state_mean": prepared.state_mean,
                    "state_scale": prepared.state_scale,
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
                _save_torch(running_checkpoint_path, payload)
                _log_line(log, f"checkpoint saved | epoch={epoch + 1} | path={running_checkpoint_path}")

            if cfg.lr_adapt and np.isfinite(grad_norm) and grad_norm > 0.0:
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
                stats = _recent_val_plateau_stats(history, window=int(cfg.recent_val_window))
                if stats is not None and stats["rel_gap_to_current"] < float(cfg.recent_val_rel_current_frac):
                    _log_line(
                        log,
                        "  early-stop: recent_val_plateau "
                        f"(window={cfg.recent_val_window} | "
                        f"(max_prev-current)/current={stats['rel_gap_to_current']:.6f} < "
                        f"{cfg.recent_val_rel_current_frac:.6f}) "
                        f"| current={stats['current']:.6e} max_prev={stats['max_prev']:.6e}",
                    )
                    break

        if failure_reason != "":
            _log_line(log, f"KAN shard failure -- epoch={failure_epoch} reason={failure_reason}")

        final_snapshot = _snapshot_split(
            force_module=model,
            known_pars=prepared.known_pars,
            eta_star_true=prepared.eta_star_true,
            mech_module=mech_module,
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
            "state_mean": prepared.state_mean,
            "state_scale": prepared.state_scale,
            "known_pars": prepared.known_pars,
            "eta_star_true": prepared.eta_star_true,
            "mech_true": prepared.mech_true,
            "stage1_warmstart": None if stage1_warmstart is None else {
                "path": str(stage1_warmstart.path),
                "rank": int(stage1_warmstart.rank),
                "trial_id": int(stage1_warmstart.trial_id),
                "loss": float(stage1_warmstart.ranking_loss),
                "seed": int(stage1_warmstart.init_seed),
                "ks0": float(stage1_warmstart.ks0),
                "cs0": float(stage1_warmstart.cs0),
            },
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
        }
        _save_torch(result_path, final_payload)
        _save_pickle(viz_result_path, _make_viz_payload(final_payload))
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        _log_line(
            log,
            f"STAGE2LIGHT shard done -- train={history[-1]['train_loss']:.6e} val={history[-1]['val_loss']:.6e} "
            f"| best_epoch={best_epoch} best_val={best_val:.6e}",
        )

    return {
        "shard_index": shard_index,
        "role": split.role,
        "label": split.label,
        "log_path": str(log_path),
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path),
        "result_path": str(result_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }


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
