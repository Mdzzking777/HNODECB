"""Hybrid supervised KFT quick check: rollout loss plus supervised Fts loss."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np
import torch

from AFM04.KAN_full_test.data import WindowSplit, prepare_data
from AFM04.KAN_full_test.kan_backend import KANForceModule
from AFM04.KAN_full_test.quick_check_supervised.config import QuickCheckSupervisedConfig, default_config
from AFM04.KAN_full_test.quick_check_supervised.losses import (
    HybridLossParts,
    fts_scale_from_truth,
    hybrid_rollout_supervised_loss,
)
from AFM04.KAN_full_test.rollout import fts_truth_from_states_torch


def _torch_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def _timestamp_file() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _timestamp_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_line(log, message: str) -> None:
    line = f"[{_timestamp_human()}] {message}"
    log.write(line + "\n")
    log.flush()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _grid_update_due(epoch: int, update_num: int, start_step: int, stop_step: int) -> bool:
    if update_num <= 0 or epoch < start_step or epoch >= stop_step:
        return False
    freq = max(1, int(np.ceil(stop_step / update_num)))
    return epoch % freq == 0


def _make_optimizer(cfg: QuickCheckSupervisedConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        amsgrad=bool(cfg.optimizer_amsgrad),
    )


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


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.sum(torch.square(grad)).cpu())
    return float(np.sqrt(total))


def _capture_grads(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {name: None if param.grad is None else param.grad.detach().clone() for name, param in model.named_parameters()}


def _restore_grads(model: torch.nn.Module, grads: dict[str, torch.Tensor | None]) -> None:
    for name, param in model.named_parameters():
        grad = grads.get(name)
        param.grad = None if grad is None else grad.detach().clone()


def _terminal_step_failure(reason: str) -> bool:
    return reason.startswith("step_trial_loss_nonfinite") or reason.startswith("step_trial_exception")


def _selected_splits(cfg: QuickCheckSupervisedConfig, splits: tuple[WindowSplit, ...]) -> list[WindowSplit]:
    if cfg.train_window_index <= 0:
        return list(splits)
    index0 = int(cfg.train_window_index) - 1
    if index0 >= len(splits):
        raise IndexError(f"train_window_index={cfg.train_window_index} but only {len(splits)} split(s) are available")
    return [splits[index0]]


def _window_file_tag(role: str) -> str:
    return role.strip().lower().replace(" ", "_")


def _observable_grid_inputs_from_ode(
    *,
    ode_train: torch.Tensor,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    cfg: QuickCheckSupervisedConfig,
) -> torch.Tensor:
    """Build AGU samples from observed x1/x2 and a neutral normalized x3 axis."""
    n = int(ode_train.shape[1])
    inputs = torch.empty((n, 3), dtype=ode_train.dtype, device=ode_train.device)
    mean_x1x2 = torch.as_tensor(
        np.asarray(state_mean[0:2], dtype=float),
        dtype=ode_train.dtype,
        device=ode_train.device,
    )
    scale_x1x2 = torch.as_tensor(
        np.asarray(state_scale[0:2], dtype=float),
        dtype=ode_train.dtype,
        device=ode_train.device,
    )
    inputs[:, 0:2] = (ode_train[0:2, :].transpose(0, 1) - mean_x1x2[None, :]) / scale_x1x2[None, :]

    if n <= 1:
        inputs[:, 2] = 0.0
    else:
        idx = torch.arange(n, dtype=ode_train.dtype, device=ode_train.device)
        frac = torch.remainder((idx + 0.5) * 0.6180339887498949, 1.0)
        inputs[:, 2] = float(cfg.grid_range_lo) + (float(cfg.grid_range_hi) - float(cfg.grid_range_lo)) * frac
    return inputs


def _parts_dict(prefix: str, parts: HybridLossParts) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in asdict(parts).items()}


def _unique_tmp_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return path.with_name(f"{path.name}.{os.getpid()}.{stamp}.tmp")


def _replace_with_windows_retry(tmp: Path, path: Path) -> Path:
    for attempt in range(20):
        try:
            tmp.replace(path)
            return path
        except PermissionError:
            sleep(min(0.05 * (attempt + 1), 1.0))

    fallback = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_pid{os.getpid()}{path.suffix}")
    tmp.replace(fallback)
    return fallback


def _save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp_path(path)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return _replace_with_windows_retry(tmp, path)


def _save_torch(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp_path(path)
    torch.save(payload, tmp)
    return _replace_with_windows_retry(tmp, path)


def _split_tensors(split: WindowSplit, part: str, *, dtype: torch.dtype, device: str) -> dict[str, torch.Tensor]:
    ode = getattr(split, f"ode_{part}")
    x2dot = getattr(split, f"x2dot_{part}")
    contact = getattr(split, f"contact_{part}")
    times = getattr(split, f"times_{part}")
    return {
        "ode": torch.as_tensor(ode, dtype=dtype, device=device),
        "states": torch.as_tensor(np.asarray(ode).T, dtype=dtype, device=device),
        "x2dot": torch.as_tensor(x2dot, dtype=dtype, device=device),
        "contact": torch.as_tensor(contact, dtype=dtype, device=device),
        "times": torch.as_tensor(times, dtype=dtype, device=device),
        "fts": torch.as_tensor(getattr(split, f"fts_{part}_true"), dtype=dtype, device=device),
    }


def _loss_for_part(
    model: KANForceModule,
    prepared,
    split_tensors: dict[str, torch.Tensor],
    cfg: QuickCheckSupervisedConfig,
    fts_scale: torch.Tensor,
) -> tuple[torch.Tensor, HybridLossParts, torch.Tensor, torch.Tensor, torch.Tensor]:
    mech_true = torch.as_tensor(prepared.mech_true, dtype=split_tensors["ode"].dtype, device=split_tensors["ode"].device)
    return hybrid_rollout_supervised_loss(
        model,
        prepared.known_pars,
        mech_true,
        split_tensors["ode"],
        split_tensors["x2dot"],
        split_tensors["contact"],
        split_tensors["times"],
        ode_method=cfg.ode_method,
        ode_rtol=cfg.ode_rtol,
        ode_atol=cfg.ode_atol,
        eta_star_true=prepared.eta_star_true,
        fts_scale=fts_scale,
    )


def _build_model(
    *,
    cfg: QuickCheckSupervisedConfig,
    prepared,
    seed: int,
    dtype: torch.dtype,
) -> KANForceModule:
    _, _, _, _, _, _, dist, _, _, a0, _ = prepared.known_pars
    return KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
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
        gnn_learnable=cfg.gnn_learnable,
        dist=dist,
        a0=a0,
        wpred_enabled=cfg.wpred_enabled,
        wpred_eps=cfg.wpred_eps,
        device=cfg.device,
        dtype=dtype,
    )


def _rebuild_trial_state_dict(
    *,
    cfg: QuickCheckSupervisedConfig,
    prepared,
    split_tensors: dict[str, torch.Tensor],
    seed: int,
    dtype: torch.dtype,
    fts_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model = _build_model(cfg=cfg, prepared=prepared, seed=seed, dtype=dtype)
    observable_grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=split_tensors["ode"],
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        cfg=cfg,
    )
    with torch.no_grad():
        if cfg.adaptive_grid_enabled:
            model.update_grid_from_normalized_inputs(observable_grid_inputs)
        model.initialize_gain_from_truth(split_tensors["states"], split_tensors["fts"])
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _maybe_load_random_search_warmstart(
    *,
    cfg: QuickCheckSupervisedConfig,
    prepared,
    split: WindowSplit,
    split_tensors: dict[str, torch.Tensor],
    model: KANForceModule,
    dtype: torch.dtype,
    fts_scale: torch.Tensor,
    log,
) -> dict[str, Any] | None:
    if not bool(cfg.warmstart_from_random_search):
        return None

    tag = _window_file_tag(split.role)
    requested_trial = int(cfg.random_search_warmstart_trial)
    if requested_trial > 0:
        trials_path = cfg.random_search_result_dir / f"qcs_random_search_trials_{tag}.json"
        if not trials_path.is_file():
            raise FileNotFoundError(f"Missing QCS random-search trial ranking file: {trials_path}")
        with trials_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        selected = None
        for row in rows:
            if int(row.get("trial", -1)) == requested_trial:
                selected = dict(row)
                break
        if selected is None:
            raise ValueError(f"Requested QCS random-search trial {requested_trial} was not found in {trials_path}")
        if bool(selected.get("failed", False)):
            raise ValueError(f"Requested QCS random-search trial {requested_trial} failed: {selected.get('failure_reason')}")
        state_dict = _rebuild_trial_state_dict(
            cfg=cfg,
            prepared=prepared,
            split_tensors=split_tensors,
            seed=int(selected.get("seed", cfg.seed + requested_trial)),
            dtype=dtype,
            fts_scale=fts_scale,
        )
        model.load_state_dict(state_dict)
        _log_line(
            log,
            "warmstart from QCS random search trial override: "
            f"trial={requested_trial} seed={int(selected.get('seed', -1))} "
            f"rank({selected.get('ranking_metric', 'train_loss')})={float(selected.get('ranking_loss', float('nan'))):.6e}",
        )
        return {"best_trial": selected, "selection_mode": "manual_trial_override", "source": str(trials_path)}

    best_path = cfg.random_search_checkpoint_dir / f"qcs_random_search_best_{tag}.pt"
    if not best_path.is_file():
        if bool(cfg.warmstart_fallback_random):
            _log_line(log, f"warmstart from QCS random search: missing checkpoint, fallback to random init | path={best_path}")
            return None
        raise FileNotFoundError(f"Missing QCS random-search warmstart checkpoint: {best_path}")
    payload = torch.load(best_path, map_location=cfg.device, weights_only=False)
    state_dict = payload.get("best_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"QCS random-search checkpoint missing best_state_dict: {best_path}")
    model.load_state_dict(state_dict)
    best_trial = dict(payload.get("best_trial") or {})
    _log_line(
        log,
        "warmstart from QCS random search: "
        f"path={best_path} trial={best_trial.get('trial', 'NA')} seed={best_trial.get('seed', 'NA')} "
        f"rank({best_trial.get('ranking_metric', 'train_loss')})={float(best_trial.get('ranking_loss', float('nan'))):.6e}",
    )
    return {"best_trial": best_trial, "selection_mode": "best_trial", "source": str(best_path)}


def _snapshot_for_split(
    model: KANForceModule,
    prepared,
    split_tensors: dict[str, torch.Tensor],
    cfg: QuickCheckSupervisedConfig,
    fts_scale: torch.Tensor,
) -> tuple[HybridLossParts, dict[str, np.ndarray]]:
    _, parts, traj, fts_pred, fts_true = _loss_for_part(model, prepared, split_tensors, cfg, fts_scale)
    payload = {
        "times": split_tensors["times"].detach().cpu().numpy(),
        "states_true": split_tensors["states"].detach().cpu().numpy(),
        "states_pred": traj.detach().transpose(0, 1).cpu().numpy(),
        "fts_true": fts_true.detach().cpu().numpy(),
        "fts_pred": fts_pred.detach().cpu().numpy(),
        "contact": split_tensors["contact"].detach().cpu().numpy().astype(bool),
    }
    return parts, payload


def _run_oracle_sanity(
    *,
    log,
    cfg: QuickCheckSupervisedConfig,
    prepared,
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    full: dict[str, torch.Tensor],
    fts_scale: torch.Tensor,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    _log_line(log, "=== QCS SANITY (teacher Fts target consistency) ===")
    oracle = _OracleForceModule(
        prepared.known_pars,
        eta_star_true=prepared.eta_star_true,
        mech_true=torch.as_tensor(prepared.mech_true, dtype=dtype, device=device),
        dtype=dtype,
        device=device,
    )
    out: dict[str, Any] = {}
    with torch.no_grad():
        teacher_states = train["states"]
        teacher_fts_formula = oracle(teacher_states).reshape(-1)
        teacher_fts_target = train["fts"].reshape(-1)
        scale = torch.clamp(torch.as_tensor(fts_scale, dtype=teacher_fts_formula.dtype, device=teacher_fts_formula.device), min=1.0e-30)
        teacher_fts_norm_mse = torch.mean(torch.square((teacher_fts_formula - teacher_fts_target) / scale))
        teacher_fts_rel_rmse_pct = 100.0 * torch.sqrt(torch.mean(torch.square(teacher_fts_formula - teacher_fts_target))) / torch.clamp(
            torch.sqrt(torch.mean(torch.square(teacher_fts_target))),
            min=1.0e-30,
        )
        row = {
            "teacher_total": float(teacher_fts_norm_mse.detach()),
            "fts_teacher_norm_mse": float(teacher_fts_norm_mse.detach()),
            "fts_teacher_rel_rmse_pct": float(teacher_fts_rel_rmse_pct.detach()),
        }
        out["train"] = row
        _log_line(log, f"  sanity FtsSup_teacher={row['fts_teacher_norm_mse']:.6e}")
        _log_line(log, f"  sanity Fts_teacher={row['fts_teacher_rel_rmse_pct']:.3e}%")
    return out


def _save_plots(cfg: QuickCheckSupervisedConfig, history: list[dict[str, float]], snapshot: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if history:
        epochs = np.asarray([row["epoch"] for row in history], dtype=float)
        train = np.asarray([row["train_loss"] for row in history], dtype=float)
        val = np.asarray([row["val_loss"] for row in history], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(epochs, train, label="train")
        ax.semilogy(epochs, val, label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel("hybrid normalized loss")
        ax.set_title("KFT quick check hybrid supervised rollout loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(cfg.visualization_dir / "qcs_supervised_loss.png", dpi=160)
        plt.close(fig)

    t_us = np.asarray(snapshot["times_full"], dtype=float) * 1.0e6
    true_nN = np.asarray(snapshot["fts_full_true"], dtype=float) * 1.0e9
    pred_nN = np.asarray(snapshot["fts_full_pred"], dtype=float) * 1.0e9
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_us, true_nN, color="black", linewidth=1.5, label="Fts teacher true")
    axes[0].plot(t_us, pred_nN, color="tab:blue", linewidth=1.2, label="Fts rollout pred")
    axes[0].set_ylabel("Fts [nN]")
    axes[0].set_title("KFT quick check hybrid supervised rollout Fts fit")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(t_us, pred_nN - true_nN, color="tab:red", linewidth=1.1)
    axes[1].set_xlabel("time [us]")
    axes[1].set_ylabel("pred - true [nN]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.visualization_dir / "qcs_supervised_fts_fit.png", dpi=160)
    plt.close(fig)


def run_quick_check(cfg: QuickCheckSupervisedConfig | None = None) -> dict[str, Any]:
    cfg = cfg or default_config()
    dtype = _torch_dtype(cfg.dtype)
    device = cfg.device
    prepared = prepare_data(cfg)
    selected = _selected_splits(cfg, prepared.splits)
    if len(selected) != 1:
        raise RuntimeError("hybrid quick_check_supervised currently expects exactly one selected window")
    split = selected[0]
    role_tag = split.role

    log_path = cfg.log_dir / f"qcs_supervised_driver_{_timestamp_file()}.txt"
    with log_path.open("w", encoding="utf-8") as log:
        _log_line(log, "KFT quick check hybrid supervised rollout start")
        _log_line(log, f"output_root={cfg.output_root}")
        _log_line(log, f"window_mode={cfg.window_mode} train_window_index={cfg.train_window_index} role={role_tag}")
        _log_line(log, f"width={cfg.width} grid={cfg.grid} k={cfg.spline_k} base_fun={cfg.base_fun}")
        _log_line(log, f"ode={cfg.ode_method} rtol={cfg.ode_rtol:.1e} atol={cfg.ode_atol:.1e}")
        _log_line(
            log,
            "controls: "
            f"step_guard={'ON' if cfg.step_guard_enabled else 'OFF'} "
            f"early_stop={'ON' if cfg.recent_val_early_stop else 'OFF'} "
            f"agu={'ON' if cfg.adaptive_grid_enabled else 'OFF'}",
        )
        _log_line(log, "loss = formal KFT rollout loss + normalized Fts rollout supervised loss; no lambda_fts")

        train = _split_tensors(split, "train", dtype=dtype, device=device)
        val = _split_tensors(split, "val", dtype=dtype, device=device)
        full = _split_tensors(split, "full", dtype=dtype, device=device)
        observable_grid_inputs = _observable_grid_inputs_from_ode(
            ode_train=train["ode"],
            state_mean=prepared.state_mean,
            state_scale=prepared.state_scale,
            cfg=cfg,
        )

        if int(cfg.width[0]) != int(train["states"].shape[1]):
            raise ValueError(f"width input dimension {cfg.width[0]} does not match state dimension {train['states'].shape[1]}")
        if int(cfg.width[-1]) != 1:
            raise ValueError("quick_check_supervised expects scalar KAN output width[-1] == 1")

        model = _build_model(cfg=cfg, prepared=prepared, seed=cfg.seed, dtype=dtype)
        init_gain = model.initialize_gain_from_truth(train["states"], train["fts"])
        fts_scale = fts_scale_from_truth(train["fts"], mode=cfg.fts_scale_mode).to(dtype=dtype, device=device)
        warmstart_payload = _maybe_load_random_search_warmstart(
            cfg=cfg,
            prepared=prepared,
            split=split,
            split_tensors=train,
            model=model,
            dtype=dtype,
            fts_scale=fts_scale,
            log=log,
        )
        if warmstart_payload is not None:
            init_gain = float(model.gain().detach().cpu().item())
        optimizer = _make_optimizer(cfg, model)
        lr = float(cfg.lr)
        grad_ema = float(cfg.lr_target_init)
        grad_target = float(cfg.lr_target_init)
        best_val_loss = float("inf")
        best_epoch = -1
        best_payload: dict[str, Any] | None = None
        best_checkpoint_path = cfg.checkpoint_dir / f"qcs_best_{role_tag}.pt"
        bad_val_epochs = 0
        train_loss_last = float("nan")
        failure_reason = ""
        history: list[dict[str, float]] = []

        _log_line(
            log,
            f"samples: train={train['states'].shape[0]} val={val['states'].shape[0]} full={full['states'].shape[0]} "
            f"fts_scale={float(fts_scale.detach()):.6e} init_gain={init_gain:.6e}",
        )
        _log_line(log, "AGU source = normalized_observed_x1x2_neutral_x3_axis")
        sanity_payload = _run_oracle_sanity(
            log=log,
            cfg=cfg,
            prepared=prepared,
            train=train,
            val=val,
            full=full,
            fts_scale=fts_scale,
            dtype=dtype,
            device=device,
        )

        for epoch in range(cfg.epochs):
            epoch_start = perf_counter()
            epoch_start_state = copy.deepcopy(model.state_dict())
            epoch_start_opt_state = copy.deepcopy(optimizer.state_dict())
            train_total: torch.Tensor | None = None
            train_parts: HybridLossParts | None = None
            val_total: torch.Tensor | None = None
            val_parts: HybridLossParts | None = None
            grad_norm = float("nan")
            step_retry_count = 0
            step_accept_reason = "accepted"
            epoch_fail_reason = ""

            for epoch_attempt in range(1, max(1, cfg.epoch_retry_max + 1) + 1):
                model.load_state_dict(copy.deepcopy(epoch_start_state))
                optimizer.load_state_dict(copy.deepcopy(epoch_start_opt_state))
                _set_optimizer_lr(optimizer, lr)
                model.train()

                if cfg.adaptive_grid_enabled and _grid_update_due(
                    epoch,
                    cfg.grid_update_num,
                    cfg.start_grid_update_step,
                    cfg.stop_grid_update_step,
                ):
                    try:
                        with torch.no_grad():
                            model.update_grid_from_normalized_inputs(observable_grid_inputs)
                    except Exception as err:
                        epoch_fail_reason = f"grid_update_exception:{err}"

                if epoch_fail_reason == "":
                    optimizer.zero_grad()
                    try:
                        train_total, train_parts, _, _, _ = _loss_for_part(model, prepared, train, cfg, fts_scale)
                        if not torch.isfinite(train_total):
                            epoch_fail_reason = "train_loss_nonfinite"
                        else:
                            train_total.backward()
                            grad_norm = _grad_norm(model)
                            if not np.isfinite(grad_norm):
                                epoch_fail_reason = "grad_norm_nonfinite"
                    except Exception as err:
                        epoch_fail_reason = f"train_exception:{err}"

                if epoch_fail_reason != "":
                    if epoch_attempt <= cfg.epoch_retry_max:
                        lr = max(lr * cfg.epoch_retry_lr_factor, cfg.epoch_retry_lr_floor)
                        _log_line(log, f"retry epoch {epoch + 1} attempt {epoch_attempt} reason={epoch_fail_reason} lr={lr:.3e}")
                        continue
                    failure_reason = f"epoch_retry_exhausted:{epoch_fail_reason}"
                    break

                prev_loss_ref = train_loss_last if np.isfinite(train_loss_last) else float("nan")
                train_loss_before = float(train_total.detach())
                model_base_state = copy.deepcopy(model.state_dict())
                opt_base_state = copy.deepcopy(optimizer.state_dict())
                grad_cache = _capture_grads(model)
                attempts_total = max(1, cfg.step_retry_max + 1) if cfg.step_guard_enabled else 1

                for step_attempt in range(1, attempts_total + 1):
                    model.load_state_dict(copy.deepcopy(model_base_state))
                    optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                    _set_optimizer_lr(optimizer, lr)
                    _restore_grads(model, grad_cache)
                    optimizer.step()
                    try:
                        trial_total, trial_parts, _, _, _ = _loss_for_part(model, prepared, train, cfg, fts_scale)
                        trial_loss = float(trial_total.detach())
                        step_accept_reason = "accepted"
                    except Exception as err:
                        trial_total = None
                        trial_parts = None
                        trial_loss = float("inf")
                        step_accept_reason = f"step_trial_exception:{err}"

                    if step_accept_reason == "accepted":
                        if not np.isfinite(trial_loss):
                            step_accept_reason = "step_trial_loss_nonfinite"
                        elif cfg.step_guard_enabled:
                            max_frac = float(cfg.step_max_loss_increase_frac)
                            if np.isfinite(max_frac) and trial_loss > train_loss_before * (1.0 + max_frac):
                                step_accept_reason = "step_trial_loss_jump"
                            elif np.isfinite(prev_loss_ref) and trial_loss > prev_loss_ref * (1.0 + max_frac):
                                step_accept_reason = "step_prev_epoch_loss_jump"

                    if step_accept_reason == "accepted":
                        train_total = trial_total
                        train_parts = trial_parts
                        step_retry_count = step_attempt - 1
                        if step_retry_count > 0:
                            _log_line(
                                log,
                                f"step-guard: accepted after {step_retry_count} retry/reduction(s) "
                                f"lr={lr:.3e} train={trial_loss:.6e}",
                            )
                        break

                    if step_attempt >= attempts_total:
                        step_retry_count = max(0, attempts_total - 1)
                        if (not cfg.step_guard_enabled) or _terminal_step_failure(step_accept_reason):
                            failure_reason = f"trial_failed:{step_accept_reason}"
                            model.load_state_dict(copy.deepcopy(model_base_state))
                            optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                            _set_optimizer_lr(optimizer, lr)
                            break
                        model.load_state_dict(copy.deepcopy(model_base_state))
                        optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                        _set_optimizer_lr(optimizer, lr)
                        train_total = torch.as_tensor(train_loss_before, dtype=dtype, device=device)
                        step_accept_reason = "step_rejected_keep_previous"
                        _log_line(
                            log,
                            f"step-guard: rejected update after {step_retry_count} retries; keep previous parameters "
                            f"lr={lr:.3e}",
                        )
                        break

                    lr = max(lr * cfg.step_retry_lr_factor, cfg.epoch_retry_lr_floor)
                    step_retry_count = step_attempt
                    _log_line(log, f"step-guard retry {step_attempt}/{cfg.step_retry_max} reason={step_accept_reason} lr={lr:.3e}")

                break

            if failure_reason != "":
                _log_line(log, f"failure: {failure_reason}")
                break

            model.eval()
            with torch.no_grad():
                val_total, val_parts, _, _, _ = _loss_for_part(model, prepared, val, cfg, fts_scale)
            train_loss = float(train_total.detach())
            val_loss = float(val_total.detach())
            train_loss_last = train_loss
            improved = np.isfinite(val_loss) and val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                bad_val_epochs = 0
                best_payload = {
                    "epoch": best_epoch,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_parts": asdict(train_parts),
                    "val_parts": asdict(val_parts),
                    "config": asdict(cfg),
                    "state_mean": prepared.state_mean,
                    "state_scale": prepared.state_scale,
                    "fts_scale": float(fts_scale.detach()),
                    "window_role": split.role,
                    "warmstart": warmstart_payload,
                    "sanity": sanity_payload,
                }
                best_checkpoint_path = _save_torch(cfg.checkpoint_dir / f"qcs_best_{role_tag}.pt", best_payload)
            else:
                bad_val_epochs += 1

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

            row = {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(lr),
                "grad_norm": float(grad_norm),
                "grad_ema": float(grad_ema),
                "grad_target": float(grad_target),
                "step_retries": float(step_retry_count),
                "step_reason": step_accept_reason,
                "seconds": float(perf_counter() - epoch_start),
                **_parts_dict("train", train_parts),
                **_parts_dict("val", val_parts),
            }
            history.append(row)

            if (epoch + 1) % cfg.log_every == 0:
                _log_line(log, f"QCS epoch {epoch + 1} train={train_loss:.6e}")
                _log_line(log, f"  grad_norm={grad_norm:.3e} lr={lr:.6g}")
                if step_retry_count > 0 and step_accept_reason == "accepted":
                    _log_line(log, f"  step-guard: accepted_after={step_retry_count} retry/reduction(s)")
                _log_line(
                    log,
                    "  parts: "
                    f"formal={train_parts.formal_total:.3e} "
                    f"FtsSup={train_parts.fts_norm_mse:.3e} "
                    f"state={train_parts.state:.3e} "
                    f"x1_state={train_parts.x1_state:.3e} "
                    f"x2_state={train_parts.x2_state:.3e} "
                    f"x2dot={train_parts.x2dot:.3e} "
                    f"x3r={train_parts.x3_range:.3e} "
                    f"ftsr={train_parts.fts_range:.3e}",
                )
                _log_line(log, f"  rec: x1={train_parts.x1_rec:.2f}% x3={train_parts.x3_rec:.2f}%")
                _log_line(log, f"  nn: F_contact err={train_parts.fts_rel_rmse_pct:.2f}%")
                _log_line(log, f"QCS val epoch {epoch + 1} val={val_loss:.6e}")
                _log_line(
                    log,
                    "  val parts: "
                    f"formal={val_parts.formal_total:.3e} "
                    f"FtsSup={val_parts.fts_norm_mse:.3e} "
                    f"state={val_parts.state:.3e} "
                    f"x1_state={val_parts.x1_state:.3e} "
                    f"x2_state={val_parts.x2_state:.3e} "
                    f"x2dot={val_parts.x2dot:.3e} "
                    f"x3r={val_parts.x3_range:.3e} "
                    f"ftsr={val_parts.fts_range:.3e}",
                )
                _log_line(log, f"  val rec: x1={val_parts.x1_rec:.2f}% x3={val_parts.x3_rec:.2f}%")
                _log_line(log, f"  val nn: F_contact err={val_parts.fts_rel_rmse_pct:.2f}%")

            if (epoch + 1) % cfg.checkpoint_every == 0:
                _save_torch(
                    cfg.checkpoint_dir / f"qcs_checkpoint_{role_tag}.pt",
                    {
                        "epoch": epoch + 1,
                        "state_dict": copy.deepcopy(model.state_dict()),
                        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                        "history": history,
                        "config": asdict(cfg),
                    },
                )

            if cfg.good_enough_loss > 0.0 and val_loss <= cfg.good_enough_loss:
                _log_line(log, f"early stop: good enough val loss {val_loss:.6e} <= {cfg.good_enough_loss:.6e}")
                break
            if cfg.recent_val_early_stop and bad_val_epochs >= cfg.recent_val_window:
                _log_line(log, f"early stop: no validation improvement for {bad_val_epochs} epochs")
                break

        if best_payload is not None:
            model.load_state_dict(best_payload["state_dict"])
        model.eval()
        with torch.no_grad():
            train_parts, train_snapshot = _snapshot_for_split(model, prepared, train, cfg, fts_scale)
            val_parts, val_snapshot = _snapshot_for_split(model, prepared, val, cfg, fts_scale)
            full_parts, full_snapshot = _snapshot_for_split(model, prepared, full, cfg, fts_scale)

        snapshot = {
            "times_full": full_snapshot["times"],
            "states_full_true": full_snapshot["states_true"],
            "states_full_pred": full_snapshot["states_pred"],
            "fts_full_true": full_snapshot["fts_true"],
            "fts_full_pred": full_snapshot["fts_pred"],
            "contact_full": full_snapshot["contact"],
            "times_train": train_snapshot["times"],
            "states_train_true": train_snapshot["states_true"],
            "states_train_pred": train_snapshot["states_pred"],
            "fts_train_true": train_snapshot["fts_true"],
            "fts_train_pred": train_snapshot["fts_pred"],
            "times_val": val_snapshot["times"],
            "states_val_true": val_snapshot["states_true"],
            "states_val_pred": val_snapshot["states_pred"],
            "fts_val_true": val_snapshot["fts_true"],
            "fts_val_pred": val_snapshot["fts_pred"],
        }
        np.savez_compressed(cfg.result_dir / f"qcs_snapshot_{role_tag}.npz", **snapshot)
        history_path = _save_json(cfg.result_dir / f"qcs_history_{role_tag}.json", history)
        summary = {
            "status": "failed" if failure_reason else "ok",
            "failure_reason": failure_reason,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "log_path": log_path,
            "history_path": history_path,
            "snapshot_path": cfg.result_dir / f"qcs_snapshot_{role_tag}.npz",
            "best_checkpoint_path": best_checkpoint_path,
            "train_final": asdict(train_parts),
            "val_final": asdict(val_parts),
            "full_final": asdict(full_parts),
            "config": asdict(cfg),
            "warmstart": warmstart_payload,
            "sanity": sanity_payload,
        }
        _save_json(cfg.result_dir / f"qcs_summary_{role_tag}.json", summary)
        _save_torch(
            cfg.result_dir / f"qcs_result_{role_tag}.pt",
            {
                "summary": summary,
                "history": history,
                "snapshot": snapshot,
                "best": best_payload,
                "final_state_dict": copy.deepcopy(model.state_dict()),
            },
        )
        _save_plots(cfg, history, snapshot)
        _log_line(
            log,
            f"done status={summary['status']} best_epoch={best_epoch} best_val={best_val_loss:.6e} "
            f"full_Fts={full_parts.fts_rel_rmse_pct:.3f}%",
        )
        return summary


def main() -> None:
    run_quick_check(default_config())


if __name__ == "__main__":
    main()
