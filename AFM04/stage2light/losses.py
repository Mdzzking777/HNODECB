"""Torch loss kernel for AFM04 stage2light."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from AFM04.stage1pluslight.losses import (
    CONTACT_LOSS_WEIGHT,
    FTS_RANGE_AMP,
    FTS_RANGE_EPS,
    FTS_RANGE_WEIGHT,
    NONCONTACT_LOSS_WEIGHT,
    SCALE_EPS,
    X3_RANGE_AMP,
    X3_RANGE_EPS,
    X3_RANGE_WEIGHT,
)
from AFM04.stage2light.rollout import fts_truth_from_states_torch, rollout_single_shooting_torch, x2dot_rhs_torch


@dataclass(frozen=True)
class TorchLossParts:
    state: float
    x1_state: float
    x2_state: float
    x2dot: float
    x3_range: float
    fts_range: float
    cont: float
    x1_rec: float
    x3_rec: float
    fts_teacher_rec: float


def _softplus_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
    return eps * torch.nn.functional.softplus(x / eps)


def _relative_rmse_pct(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    err = torch.sum(torch.square(pred - truth))
    den = torch.sum(torch.square(truth))
    count = torch.tensor(float(max(int(truth.numel()), 1)), dtype=truth.dtype, device=truth.device)
    truth_rms = torch.sqrt(den / count)
    return 100.0 * torch.sqrt(err / count) / truth_rms


def _window_loss(
    *,
    traj: torch.Tensor,
    ode_true: torch.Tensor,
    x2dot_true: torch.Tensor,
    contact_mask: torch.Tensor,
    times: torch.Tensor,
    force_module,
    known_pars: tuple[float, ...],
    mech_true: torch.Tensor,
    eta_star_true: float,
) -> tuple[torch.Tensor, TorchLossParts]:
    weights = torch.where(
        contact_mask > 0.5,
        torch.full_like(contact_mask, CONTACT_LOSS_WEIGHT),
        torch.full_like(contact_mask, NONCONTACT_LOSS_WEIGHT),
    )
    weights_sum = torch.clamp(torch.sum(weights), min=1.0)

    state_true = ode_true[0:2, :]
    state_pred = traj[0:2, :]
    state_scale = torch.sqrt(torch.mean(torch.square(state_true), dim=1))
    x1_err = torch.square((state_true[0, :] - state_pred[0, :]) / state_scale[0])
    x2_err_state = torch.square((state_true[1, :] - state_pred[1, :]) / state_scale[1])
    state_err = x1_err + x2_err_state
    state_loss = torch.sum(weights * state_err) / weights_sum
    x1_state_loss = torch.sum(weights * x1_err) / weights_sum
    x2_state_loss = torch.sum(weights * x2_err_state) / weights_sum

    x2dot_pred = x2dot_rhs_torch(traj, times, force_module, known_pars)
    x2dot_scale = torch.sqrt(torch.mean(torch.square(x2dot_true)))
    x2_err = torch.square((x2dot_true - x2dot_pred) / x2dot_scale)
    x2dot_loss = torch.sum(weights * x2_err) / weights_sum

    fts_pred = force_module(traj.transpose(0, 1))
    fts_scale = torch.tensor(max(float(FTS_RANGE_AMP), float(SCALE_EPS)), dtype=traj.dtype, device=traj.device)
    fts_exceed = torch.abs(fts_pred) - FTS_RANGE_AMP
    fts_pen = torch.square(_softplus_torch(fts_exceed, FTS_RANGE_EPS) / fts_scale)
    fts_range_loss = FTS_RANGE_WEIGHT * (torch.sum(weights * fts_pen) / weights_sum)

    x3_exceed = torch.abs(traj[2, :]) - X3_RANGE_AMP
    x3_scale = torch.tensor(max(float(X3_RANGE_AMP), float(SCALE_EPS)), dtype=traj.dtype, device=traj.device)
    x3_pen = torch.square(_softplus_torch(x3_exceed, X3_RANGE_EPS) / x3_scale)
    x3_range_loss = X3_RANGE_WEIGHT * (torch.sum(weights * x3_pen) / weights_sum)

    x1_rec = _relative_rmse_pct(traj[0, :], ode_true[0, :])
    x3_rec = _relative_rmse_pct(traj[2, :], ode_true[2, :])

    teacher_states = ode_true.transpose(0, 1)
    fts_teacher_pred = force_module(teacher_states)
    fts_teacher_true = fts_truth_from_states_torch(
        teacher_states,
        known_pars,
        eta_star=eta_star_true,
        mech_true=mech_true,
    )
    fts_teacher_rec = _relative_rmse_pct(fts_teacher_pred, fts_teacher_true)

    total = state_loss + x2dot_loss + x3_range_loss + fts_range_loss
    parts = TorchLossParts(
        state=float(state_loss.detach()),
        x1_state=float(x1_state_loss.detach()),
        x2_state=float(x2_state_loss.detach()),
        x2dot=float(x2dot_loss.detach()),
        x3_range=float(x3_range_loss.detach()),
        fts_range=float(fts_range_loss.detach()),
        cont=0.0,
        x1_rec=float(x1_rec.detach()),
        x3_rec=float(x3_rec.detach()),
        fts_teacher_rec=float(fts_teacher_rec.detach()),
    )
    return total, parts


def evaluate_split(
    *,
    force_module,
    known_pars: tuple[float, ...],
    mech_module,
    ode_true: torch.Tensor,
    x2dot_true: torch.Tensor,
    contact_mask: torch.Tensor,
    times: torch.Tensor,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    mech_true: torch.Tensor,
    eta_star_true: float,
    x1_abs_guard: float | None = None,
    x2_abs_guard: float | None = None,
) -> tuple[torch.Tensor, TorchLossParts, torch.Tensor]:
    u0 = ode_true[:, 0]
    traj = rollout_single_shooting_torch(
        force_module,
        known_pars,
        mech_module,
        u0,
        times,
        method=ode_method,
        rtol=ode_rtol,
        atol=ode_atol,
        x1_abs_guard=x1_abs_guard,
        x2_abs_guard=x2_abs_guard,
    )
    total, parts = _window_loss(
        traj=traj,
        ode_true=ode_true,
        x2dot_true=x2dot_true,
        contact_mask=contact_mask,
        times=times,
        force_module=force_module,
        known_pars=known_pars,
        mech_true=mech_true,
        eta_star_true=eta_star_true,
    )
    return total, parts, traj


__all__ = ["TorchLossParts", "evaluate_split"]
