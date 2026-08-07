"""One-seed trial contract for AFM06a stage1pluslight."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch

from AFM06a.stage1pluslight.config import Stage1PlusLightConfig
from AFM06a.stage1pluslight.data import PreparedWindow
from AFM06a.stage1pluslight.kan_backend import build_kan_runtime, training_force_normalizer
from AFM06a.stage1pluslight.losses import (
    evaluate_dynamics_residual_loss,
    nan_loss_parts,
    parts_as_dict,
    validate_loss_policy,
)
from AFM06a.stage1pluslight.seed_search import SeedTrial
TrialExecutor = Callable[[SeedTrial, Stage1PlusLightConfig, PreparedWindow], dict[str, Any]]


def failed_trial_record(trial: SeedTrial, reason: str) -> dict[str, Any]:
    return {
        "loss": float("inf"),
        "train_loss": float("inf"),
        "val_loss": float("inf"),
        "params": {
            "trial_id": int(trial.global_trial_id),
            "nn_seed_bank_idx": int(trial.nn_seed_bank_idx),
            "nn_init_seed": int(trial.nn_init_seed),
            "trial_label": str(trial.trial_label),
            "kan_width": [1, 3, 1],
            "search_mode": "kan_seed_bank_only",
            "force_output": "Fts_N",
        },
        "loss_parts": {},
        "is_viable": False,
        "trial_failed": True,
        "failure_reason": str(reason),
        "time_per_trial": float("inf"),
    }


def run_seed_trial(
    trial: SeedTrial,
    config: Stage1PlusLightConfig,
    window: PreparedWindow,
    *,
    executor: TrialExecutor | None = None,
) -> dict[str, Any]:
    if window.states.shape[0] != 2:
        return failed_trial_record(trial, f"expected two states, got {window.states.shape}")
    if executor is None:
        return failed_trial_record(
            trial,
            "AFM06a KAN/loss runtime is pending policy decisions; framework check only",
        )
    try:
        record = executor(trial, config, window)
    except Exception as exc:
        return failed_trial_record(trial, f"{type(exc).__name__}: {exc}")
    if not isinstance(record, dict):
        return failed_trial_record(trial, "trial executor did not return a record dictionary")
    return record


def execute_seed_trial(
    trial: SeedTrial,
    config: Stage1PlusLightConfig,
    window: PreparedWindow,
) -> dict[str, Any]:
    """Evaluate one initialized KAN seed without back-propagation."""

    validate_loss_policy(config.loss_policy)
    started = perf_counter()
    device = torch.device(config.device)
    dtype = torch.float64
    training_states = np.asarray(window.states[:, window.train_idx].T, dtype=float)
    gain_reference = np.asarray(window.fts[window.train_idx], dtype=float)
    model = build_kan_runtime(
        config,
        nn_init_seed=trial.nn_init_seed,
        training_states=training_states,
        gain_reference=gain_reference,
        global_state_mean=window.global_state_mean,
        global_state_scale=window.global_state_scale,
        global_force_mean=window.global_bar_fts_mean,
        global_force_scale=window.global_bar_fts_scale,
        settings=window.settings,
    )
    model.eval()

    true_states = torch.as_tensor(window.states, dtype=dtype, device=device)
    times = torch.as_tensor(window.times, dtype=dtype, device=device)
    true_x2dot = torch.as_tensor(window.x2dot, dtype=dtype, device=device)
    train_idx = torch.as_tensor(window.train_idx, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(window.val_idx, dtype=torch.long, device=device)
    _residual_mean, residual_scale_value = training_force_normalizer(
        window.bar_fts_rhs_residual[window.train_idx]
    )
    residual_scale = torch.as_tensor(residual_scale_value, dtype=dtype, device=device)
    settings = window.settings
    with torch.no_grad():
        force_states = torch.stack(
            (true_states[0], torch.zeros_like(true_states[0])),
            dim=1,
        )
        predicted_fts = model(force_states)
        train_total, train_parts = evaluate_dynamics_residual_loss(
            predicted_fts=predicted_fts,
            true_states=true_states,
            true_x2dot=true_x2dot,
            times=times,
            indices=train_idx,
            residual_scale=residual_scale,
            settings=settings,
            smoothness_weight=config.smoothness_loss_weight,
        )
        if val_idx.numel() > 0:
            val_total, val_parts = evaluate_dynamics_residual_loss(
                predicted_fts=predicted_fts,
                true_states=true_states,
                true_x2dot=true_x2dot,
                times=times,
                indices=val_idx,
                residual_scale=residual_scale,
                settings=settings,
                smoothness_weight=config.smoothness_loss_weight,
            )
        else:
            val_total = torch.as_tensor(float("nan"), dtype=dtype, device=device)
            val_parts = nan_loss_parts()

    model_meta = model.metadata()
    state_dict = model.frozen_state_dict()
    elapsed = float(perf_counter() - started)
    predicted_fts_np = predicted_fts.detach().cpu().numpy()
    predicted_bar_fts_np = predicted_fts_np / float(settings.mass_kg)
    return {
        "loss": float(train_total.detach().cpu()),
        "train_loss": float(train_total.detach().cpu()),
        "val_loss": float(val_total.detach().cpu()),
        "params": {
            "trial_id": int(trial.global_trial_id),
            "nn_seed_bank_idx": int(trial.nn_seed_bank_idx),
            "nn_init_seed": int(trial.nn_init_seed),
            "trial_label": str(trial.trial_label),
            "kan_width": list(config.kan_width),
            "search_mode": "kan_seed_bank_only",
            "force_output": "Fts_N",
            "force_input_source": "observed_x1_at_every_time",
            "loss_target": "zero_dynamics_residual",
            "loss_terms": "dynamics_residual_plus_residual_smoothness",
            "evaluation_policy": "pointwise_true_x1_no_rollout",
        },
        "loss_parts": parts_as_dict(train_parts),
        "val_loss_parts": parts_as_dict(val_parts),
        "is_viable": True,
        "trial_failed": False,
        "failure_reason": "",
        "time_per_trial": elapsed,
        "warmstart": {
            "kan_state_dict": state_dict,
            "kan_state_dict_sha256": str(model_meta["state_dict_sha256"]),
            "kan_construction": model_meta["construction"],
            "input_policy": str(model_meta["input_policy"]),
            "state_mean": np.asarray(model_meta["state_mean"], dtype=float),
            "state_scale": np.asarray(model_meta["state_scale"], dtype=float),
            "force_output_policy": str(model_meta["force_output_policy"]),
            "force_output_quantity": str(model_meta["force_output_quantity"]),
            "force_mean": float(model_meta["force_mean"]),
            "force_scale": float(model_meta["force_scale"]),
            "initial_grid_support": np.asarray(model_meta["initial_grid_support"], dtype=float),
            "nn_gain": float(model_meta["nn_gain"]),
            "soft_mask": dict(model_meta["soft_mask"]),
            "initial_state": np.asarray(window.states[:, 0], dtype=float).copy(),
        },
        "predicted_fts": predicted_fts_np,
        "predicted_bar_fts": predicted_bar_fts_np,
        "diagnostics": {
            "contact_fraction": float(np.mean(window.contact)),
            "contact_fraction_train": float(np.mean(window.contact[window.train_idx])),
            "contact_fraction_val": (
                float(np.mean(window.contact[window.val_idx]))
                if window.val_idx.size
                else float("nan")
            ),
            "gain_reference_q95": float(np.quantile(np.abs(gain_reference), 0.95)),
            "global_state_mean": np.asarray(window.global_state_mean, dtype=float),
            "global_state_scale": np.asarray(window.global_state_scale, dtype=float),
            "global_bar_fts_mean": float(window.global_bar_fts_mean),
            "global_bar_fts_scale": float(window.global_bar_fts_scale),
            "fts_pred_min_N": float(np.min(predicted_fts_np)),
            "fts_pred_max_N": float(np.max(predicted_fts_np)),
            "bar_fts_pred_min": float(np.min(predicted_bar_fts_np)),
            "bar_fts_pred_max": float(np.max(predicted_bar_fts_np)),
            "dynamics_residual_target": 0.0,
            "smoothness_loss_weight": float(config.smoothness_loss_weight),
            "kan_input_source": "observed_x1_at_every_time",
            "st1pl_rollout_performed": False,
        },
    }


__all__ = ["TrialExecutor", "execute_seed_trial", "failed_trial_record", "run_seed_trial"]
