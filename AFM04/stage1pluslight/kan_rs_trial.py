"""KAN random-search style trial backend for AFM04 stage1pluslight."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from AFM04.KAN_full_test.config import default_config as default_kan_config
from AFM04.KAN_full_test.data import PreparedData, WindowSplit, prepare_data
from AFM04.KAN_full_test.losses import TorchLossParts, evaluate_split
from AFM04.KAN_full_test.rollout import StateGuardTriggered, fts_truth_from_states_torch
from AFM04.stage2light.kan_backend import KANForceModule
from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import CS, KS


def _torch_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


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


def _rel_err_pct(est: float, truth: float, eps: float) -> float:
    return 100.0 * abs(float(est) - float(truth)) / max(abs(float(truth)), float(eps))


def _parts_dict(parts: TorchLossParts) -> dict[str, float]:
    return {
        "state": float(parts.state),
        "x1_state": float(parts.x1_state),
        "x2_state": float(parts.x2_state),
        "x2dot": float(parts.x2dot),
        "x3_range": float(parts.x3_range),
        "fts_range": float(parts.fts_range),
        "cont": float(parts.cont),
        "x1_rec": float(parts.x1_rec),
        "x3_rec": float(parts.x3_rec),
        "fts_teacher_rec": float(getattr(parts, "fts_teacher_rec", getattr(parts, "fts_rollout_rec", float("nan")))),
    }


def _rollout_left_nn_err(
    *,
    force_module: KANForceModule,
    traj: torch.Tensor,
    ode_true: torch.Tensor,
    known_pars: tuple[float, ...],
    mech_true: torch.Tensor,
    eta_star_true: float,
) -> float:
    pred_states = traj.transpose(0, 1)
    teacher_states = ode_true.transpose(0, 1)
    pred_force = force_module(pred_states)
    true_force = fts_truth_from_states_torch(
        teacher_states,
        known_pars,
        eta_star=eta_star_true,
        mech_true=mech_true,
    )
    err = torch.sum(torch.square(pred_force - true_force))
    den = torch.sum(torch.square(true_force))
    count = torch.tensor(float(max(int(true_force.numel()), 1)), dtype=pred_force.dtype, device=pred_force.device)
    truth_rms = torch.sqrt(den / count)
    return float((100.0 * torch.sqrt(err / count) / truth_rms).detach())


def _trial_seed(base_seed: int, seed_bank_index: int) -> int:
    return int(base_seed + seed_bank_index)


def _observable_grid_inputs_from_ode(
    *,
    ode_train: torch.Tensor,
) -> torch.Tensor:
    """Build AGU samples in raw physical coordinates.

    x1/x2 come from observed data.  x3 is a neutral raw-domain axis only; it
    does not encode true/predicted x3 and therefore does not guide AGU.
    """

    n = int(ode_train.shape[1])
    inputs = torch.empty((n, 3), dtype=ode_train.dtype, device=ode_train.device)
    inputs[:, 0:2] = ode_train[0:2, :].transpose(0, 1)
    if n <= 1:
        inputs[:, 2] = 0.0
    else:
        x1_abs = float(torch.max(torch.abs(ode_train[0, :])).detach().cpu()) if n > 0 else 0.0
        x3_amp = max(0.1 * x1_abs, 1.0e-12)
        inputs[:, 2] = torch.linspace(-x3_amp, x3_amp, n, dtype=ode_train.dtype, device=ode_train.device)
    return inputs


def prepare_kan_stage1_runtime(
    repo_root: str | Path,
    *,
    auto_generate_dataset: bool = False,
    window_mode: str | None = None,
    arch_window_us: float | None = None,
    val_stride: int | None = None,
    val_offset: int | None = None,
) -> dict[str, Any]:
    kcfg = default_kan_config(repo_root)
    kcfg = replace(
        kcfg,
        auto_generate_dataset=bool(auto_generate_dataset),
        window_mode=str(window_mode).strip() if window_mode is not None else kcfg.window_mode,
        arch_window_us=float(arch_window_us) if arch_window_us is not None else kcfg.arch_window_us,
        val_stride=int(val_stride) if val_stride is not None else kcfg.val_stride,
        val_offset=int(val_offset) if val_offset is not None else kcfg.val_offset,
        random_search_window_index=1,
        train_window_index=0,
    )
    prepared = prepare_data(kcfg)
    split = prepared.splits[0]
    dtype = _torch_dtype(kcfg.dtype)
    tensors = _window_to_torch(split, dtype=dtype, device=kcfg.device)
    train_states = torch.as_tensor(split.ode_train.T, dtype=dtype, device=kcfg.device)
    train_fts = torch.as_tensor(split.fts_train_true, dtype=dtype, device=kcfg.device)

    train_max_abs_x1 = float(np.max(np.abs(split.ode_full[0, :]))) if split.ode_full.shape[1] > 0 else float("nan")
    train_max_abs_x2 = float(np.max(np.abs(split.ode_full[1, :]))) if split.ode_full.shape[1] > 0 else float("nan")
    x1_abs_guard = (
        float(kcfg.random_search_x1_guard_mult) * train_max_abs_x1
        if kcfg.random_search_fail_fast_enabled and np.isfinite(train_max_abs_x1) and train_max_abs_x1 > 0.0
        else None
    )
    x2_abs_guard = (
        float(kcfg.random_search_x2_guard_mult) * train_max_abs_x2
        if kcfg.random_search_fail_fast_enabled and np.isfinite(train_max_abs_x2) and train_max_abs_x2 > 0.0
        else None
    )

    return {
        "cfg": kcfg,
        "prepared": prepared,
        "split": split,
        "dtype": dtype,
        "tensors": tensors,
        "train_states": train_states,
        "train_fts": train_fts,
        "x1_abs_guard": x1_abs_guard,
        "x2_abs_guard": x2_abs_guard,
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
    cfg = runtime["cfg"]
    prepared: PreparedData = runtime["prepared"]
    split: WindowSplit = runtime["split"]
    dtype: torch.dtype = runtime["dtype"]
    tensors: dict[str, torch.Tensor] = runtime["tensors"]
    train_states: torch.Tensor = runtime["train_states"]
    train_fts: torch.Tensor = runtime["train_fts"]
    x1_abs_guard = runtime["x1_abs_guard"]
    x2_abs_guard = runtime["x2_abs_guard"]

    seed = _trial_seed(int(cfg.seed), int(nn_seed_bank_idx))
    mech = torch.as_tensor([float(ks_fixed), float(cs_fixed)], dtype=dtype, device=cfg.device)

    t0 = perf_counter()
    model = KANForceModule(
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
        dist=float(prepared.known_pars[6]),
        a0=float(prepared.known_pars[9]),
        gnn_learnable=cfg.gnn_learnable,
        soft_mask_enabled=True,
        soft_mask_trainable=False,
        soft_mask_s0_a0=20.0,
        soft_mask_s0_min_a0=1.0,
        soft_mask_s0_max_a0=100.0,
        soft_mask_alpha_a0=0.25,
        soft_mask_alpha_min_a0=0.02,
        soft_mask_alpha_max_a0=5.0,
        device=cfg.device,
        dtype=dtype,
    ).to(cfg.device)
    with torch.no_grad():
        if cfg.adaptive_grid_enabled:
            grid_inputs = _observable_grid_inputs_from_ode(
                ode_train=tensors["ode_train"],
            )
            model.update_grid_from_normalized_inputs(grid_inputs)
        init_gnn = float(model.initialize_gain_from_truth(train_states, train_fts))

    failure_reason = ""
    train_total = None
    train_parts = None
    train_traj = None
    val_total = None
    val_parts = None
    val_traj = None
    try:
        with torch.no_grad():
            train_total, train_parts, train_traj = evaluate_split(
                force_module=model,
                known_pars=prepared.known_pars,
                mech_true=mech,
                ode_true=tensors["ode_full"],
                x2dot_true=tensors["x2dot_full"],
                contact_mask=tensors["contact_full"],
                times=tensors["times_full"],
                ode_method=cfg.ode_method,
                ode_rtol=cfg.ode_rtol,
                ode_atol=cfg.ode_atol,
                eta_star_true=prepared.eta_star_true,
                x1_abs_guard=x1_abs_guard,
                x2_abs_guard=x2_abs_guard,
                loss_indices=tensors["train_idx"],
            )
            if cfg.random_search_use_val:
                val_total, val_parts, val_traj = evaluate_split(
                    force_module=model,
                    known_pars=prepared.known_pars,
                    mech_true=mech,
                    ode_true=tensors["ode_full"],
                    x2dot_true=tensors["x2dot_full"],
                    contact_mask=tensors["contact_full"],
                    times=tensors["times_full"],
                    ode_method=cfg.ode_method,
                    ode_rtol=cfg.ode_rtol,
                    ode_atol=cfg.ode_atol,
                    eta_star_true=prepared.eta_star_true,
                    x1_abs_guard=x1_abs_guard,
                    x2_abs_guard=x2_abs_guard,
                    loss_indices=tensors["val_idx"],
                )
    except StateGuardTriggered as exc:
        failure_reason = str(exc)
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"

    dt_total = float(perf_counter() - t0)

    if failure_reason != "" or train_total is None or train_parts is None:
        metric_parts = {
            "state": float("nan"),
            "x1_state": float("nan"),
            "x2_state": float("nan"),
            "x2dot": float("nan"),
            "x3_range": float("nan"),
            "fts_range": float("nan"),
            "cont": float("nan"),
            "x1_rec": float("nan"),
            "x3_rec": float("nan"),
            "fts_teacher_rec": float("nan"),
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
                "stage1plus_arch_window_us": float(cfg.arch_window_us),
                "window_role": str(split.role),
                "ranking_metric": "train_loss",
                "kan_grid": int(cfg.grid),
                "kan_spline_k": int(cfg.spline_k),
                "kan_base_fun": str(cfg.base_fun),
            },
            "val_parts": metric_parts,
            "ks_hat": float(ks_fixed),
            "cs_hat": float(cs_fixed),
            "ks_err_pct": _rel_err_pct(float(ks_fixed), float(KS), 1.0e-9),
            "cs_err_pct": _rel_err_pct(float(cs_fixed), float(CS), 1.0e-9),
            "val_nn_err_start": float("nan"),
            "val_nn_err": float("nan"),
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
            "time_per_epoch": dt_total,
        }

    train_loss = float(train_total.detach())
    val_loss = float(val_total.detach()) if val_total is not None else float("nan")
    ranking_loss = train_loss
    metric_parts = _parts_dict(val_parts) if val_parts is not None else _parts_dict(train_parts)
    metric_traj = val_traj if val_traj is not None else train_traj
    metric_indices = tensors["val_idx"] if val_traj is not None else tensors["train_idx"]
    metric_traj = metric_traj[:, metric_indices]
    metric_ode_true = tensors["ode_full"][:, metric_indices]
    metric_nn_err = _rollout_left_nn_err(
        force_module=model,
        traj=metric_traj,
        ode_true=metric_ode_true,
        known_pars=prepared.known_pars,
        mech_true=mech,
        eta_star_true=prepared.eta_star_true,
    )
    metric_parts["fts_teacher_rec"] = float(metric_nn_err)

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
            "stage1plus_arch_window_us": float(cfg.arch_window_us),
            "window_role": str(split.role),
            "ranking_metric": "train_loss",
            "kan_grid": int(cfg.grid),
            "kan_spline_k": int(cfg.spline_k),
            "kan_base_fun": str(cfg.base_fun),
        },
        "val_parts": metric_parts,
        "ks_hat": float(ks_fixed),
        "cs_hat": float(cs_fixed),
        "ks_err_pct": _rel_err_pct(float(ks_fixed), float(KS), 1.0e-9),
        "cs_err_pct": _rel_err_pct(float(cs_fixed), float(CS), 1.0e-9),
        "val_nn_err_start": float(metric_nn_err),
        "val_nn_err": float(metric_nn_err),
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
