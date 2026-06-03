"""Hybrid rollout + supervised Fts losses for the KFT quick check."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from AFM04.KAN_full_test.losses import evaluate_split
from AFM04.KAN_full_test.rollout import fts_truth_from_states_torch


SCALE_EPS = 1.0e-30


@dataclass(frozen=True)
class HybridLossParts:
    total: float
    formal_total: float
    state: float
    x1_state: float
    x2_state: float
    x2dot: float
    x3_range: float
    fts_range: float
    cont: float
    x1_rec: float
    x3_rec: float
    fts_rollout_rec: float
    fts_norm_mse: float
    fts_raw_mse: float
    fts_rel_rmse_pct: float
    fts_scale: float


def fts_scale_from_truth(fts_true: torch.Tensor, *, mode: str = "rms") -> torch.Tensor:
    truth = torch.as_tensor(fts_true).reshape(-1)
    if mode == "maxabs":
        scale = torch.max(torch.abs(truth))
    else:
        scale = torch.sqrt(torch.mean(torch.square(truth)))
    return torch.clamp(scale, min=SCALE_EPS)


def _relative_rmse_pct(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1)
    truth = truth.reshape(-1)
    denom = torch.sqrt(torch.mean(torch.square(truth)))
    denom = torch.clamp(denom, min=SCALE_EPS)
    return 100.0 * torch.sqrt(torch.mean(torch.square(pred - truth))) / denom


def hybrid_rollout_supervised_loss(
    force_module,
    known_pars,
    mech_true: torch.Tensor,
    ode_true: torch.Tensor,
    x2dot_true: torch.Tensor,
    contact_mask: torch.Tensor,
    times: torch.Tensor,
    *,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    eta_star_true: float,
    fts_scale: torch.Tensor,
    x1_abs_guard: float | None = None,
    x2_abs_guard: float | None = None,
) -> tuple[torch.Tensor, HybridLossParts, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return equal-weight normalized formal rollout loss plus supervised Fts loss.

    No lambda or extra loss weight is applied here. The formal KFT loss terms are
    normalized inside ``evaluate_split``; the Fts term is normalized by
    ``fts_scale`` before being added.
    """

    formal_total, formal_parts, traj = evaluate_split(
        force_module=force_module,
        known_pars=known_pars,
        mech_true=mech_true,
        ode_true=ode_true,
        x2dot_true=x2dot_true,
        contact_mask=contact_mask,
        times=times,
        ode_method=ode_method,
        ode_rtol=float(ode_rtol),
        ode_atol=float(ode_atol),
        eta_star_true=eta_star_true,
        x1_abs_guard=x1_abs_guard,
        x2_abs_guard=x2_abs_guard,
    )
    rollout_states = traj.transpose(0, 1)
    teacher_states = ode_true.transpose(0, 1)
    fts_pred = force_module(rollout_states).reshape(-1)
    fts_true = fts_truth_from_states_torch(
        teacher_states,
        known_pars,
        eta_star=eta_star_true,
        mech_true=mech_true,
    ).reshape(-1)
    scale = torch.clamp(torch.as_tensor(fts_scale, dtype=fts_pred.dtype, device=fts_pred.device), min=SCALE_EPS)
    fts_err_norm = (fts_pred - fts_true) / scale
    fts_norm_mse = torch.mean(torch.square(fts_err_norm))
    total = formal_total + fts_norm_mse
    fts_raw_mse = torch.mean(torch.square(fts_pred - fts_true))
    fts_rel = _relative_rmse_pct(fts_pred, fts_true)
    parts = HybridLossParts(
        total=float(total.detach()),
        formal_total=float(formal_total.detach()),
        state=float(formal_parts.state),
        x1_state=float(formal_parts.x1_state),
        x2_state=float(formal_parts.x2_state),
        x2dot=float(formal_parts.x2dot),
        x3_range=float(formal_parts.x3_range),
        fts_range=float(formal_parts.fts_range),
        cont=float(formal_parts.cont),
        x1_rec=float(formal_parts.x1_rec),
        x3_rec=float(formal_parts.x3_rec),
        fts_rollout_rec=float(formal_parts.fts_rollout_rec),
        fts_norm_mse=float(fts_norm_mse.detach()),
        fts_raw_mse=float(fts_raw_mse.detach()),
        fts_rel_rmse_pct=float(fts_rel.detach()),
        fts_scale=float(scale.detach()),
    )
    return total, parts, traj, fts_pred, fts_true


__all__ = ["HybridLossParts", "fts_scale_from_truth", "hybrid_rollout_supervised_loss"]
