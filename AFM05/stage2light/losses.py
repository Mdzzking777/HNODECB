"""Torch loss kernel for AFM05 stage2light.

AFM05 has no true/observed x3(t), x3dot(t), delta_dot(t), or Fts(t).  The loss
therefore follows the AFM05 stage1pluslight training side: supervise x1, x2,
and x2dot; keep model-side x3/Fts range penalties; do not emit true-side
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from AFM05.stage1pluslight.losses import (
    CONTACT_LOSS_WEIGHT,
    FTS_RANGE_EPS,
    FTS_RANGE_WEIGHT,
    NONCONTACT_LOSS_WEIGHT,
    SCALE_EPS,
    X3_RANGE_EPS,
    X3_RANGE_WEIGHT,
)
from AFM05.stage2light.rollout import _actuation_values_torch, rollout_single_shooting_torch, x2dot_rhs_torch


@dataclass(frozen=True)
class TorchLossParts:
    state: float
    x1_state: float
    x2_state: float
    x2dot: float
    x3_range: float
    fts_range: float
    x3_range_amp: float
    fts_range_amp: float
    cont: float
    x1_rec: float
    x2_rec: float
    x2dot_rec: float


def _softplus_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
    return eps * torch.nn.functional.softplus(x / eps)


def _relative_rmse_pct(pred: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    count = torch.tensor(float(max(int(reference.numel()), 1)), dtype=reference.dtype, device=reference.device)
    err = torch.sum(torch.square(pred - reference))
    den = torch.sum(torch.square(reference))
    reference_rms = torch.sqrt(den / count)
    return 100.0 * torch.sqrt(err / count) / torch.clamp(
        reference_rms,
        min=torch.as_tensor(SCALE_EPS, dtype=reference.dtype, device=reference.device),
    )


def _safe_rms(values: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(torch.square(values)))
    return torch.clamp(rms, min=torch.as_tensor(SCALE_EPS, dtype=values.dtype, device=values.device))


def _mean_abs_amp(values: torch.Tensor, *, floor: float) -> torch.Tensor:
    amp = torch.mean(torch.abs(values.reshape(-1)))
    return torch.clamp(amp, min=torch.as_tensor(floor, dtype=values.dtype, device=values.device))


def _window_range_amps(
    *,
    ode_window: torch.Tensor,
    times_window: torch.Tensor,
    known_pars: dict[str, Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """AFM05 model-side range scales from the current observed training window."""

    x3_range_amp = _mean_abs_amp(ode_window[0, :], floor=X3_RANGE_EPS)
    f_act_window = _actuation_values_torch(known_pars, times_window)
    fts_range_amp = _mean_abs_amp(f_act_window, floor=FTS_RANGE_EPS)
    return x3_range_amp, fts_range_amp


def _window_loss(
    *,
    traj: torch.Tensor,
    ode_true: torch.Tensor,
    x2dot_true: torch.Tensor,
    contact_mask: torch.Tensor,
    times: torch.Tensor,
    force_module,
    known_pars: dict[str, Any] | tuple[Any, ...],
    loss_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, TorchLossParts]:
    x3_range_amp, fts_range_amp = _window_range_amps(
        ode_window=ode_true,
        times_window=times,
        known_pars=known_pars,
    )

    if loss_indices is not None:
        idx = loss_indices.to(device=traj.device, dtype=torch.long)
        traj = traj[:, idx]
        ode_true = ode_true[:, idx]
        x2dot_true = x2dot_true[idx]
        contact_mask = contact_mask[idx]
        times = times[idx]

    weights = torch.where(
        contact_mask > 0.5,
        torch.full_like(contact_mask, CONTACT_LOSS_WEIGHT),
        torch.full_like(contact_mask, NONCONTACT_LOSS_WEIGHT),
    )
    weights_sum = torch.clamp(torch.sum(weights), min=1.0)

    state_true = ode_true[0:2, :]
    state_pred = traj[0:2, :]
    state_scale = torch.stack((_safe_rms(state_true[0, :]), _safe_rms(state_true[1, :])))
    x1_err = torch.square((state_true[0, :] - state_pred[0, :]) / state_scale[0])
    x2_err_state = torch.square((state_true[1, :] - state_pred[1, :]) / state_scale[1])
    state_loss = torch.sum(weights * (x1_err + x2_err_state)) / weights_sum
    x1_state_loss = torch.sum(weights * x1_err) / weights_sum
    x2_state_loss = torch.sum(weights * x2_err_state) / weights_sum

    x2dot_pred = x2dot_rhs_torch(traj, times, force_module, known_pars)
    x2dot_scale = _safe_rms(x2dot_true)
    x2dot_err = torch.square((x2dot_true - x2dot_pred) / x2dot_scale)
    x2dot_loss = torch.sum(weights * x2dot_err) / weights_sum

    rollout_states = traj.transpose(0, 1)
    fts_pred = force_module(rollout_states)
    fts_scale = torch.clamp(fts_range_amp, min=torch.as_tensor(FTS_RANGE_EPS, dtype=traj.dtype, device=traj.device))
    fts_exceed = torch.abs(fts_pred) - fts_range_amp
    fts_pen = torch.square(_softplus_torch(fts_exceed, FTS_RANGE_EPS) / fts_scale)
    fts_range_loss = FTS_RANGE_WEIGHT * (torch.sum(weights * fts_pen) / weights_sum)

    x3_exceed = torch.abs(traj[2, :]) - x3_range_amp
    x3_scale = torch.clamp(x3_range_amp, min=torch.as_tensor(X3_RANGE_EPS, dtype=traj.dtype, device=traj.device))
    x3_pen = torch.square(_softplus_torch(x3_exceed, X3_RANGE_EPS) / x3_scale)
    x3_range_loss = X3_RANGE_WEIGHT * (torch.sum(weights * x3_pen) / weights_sum)

    total = state_loss + x2dot_loss + x3_range_loss + fts_range_loss
    parts = TorchLossParts(
        state=float(state_loss.detach()),
        x1_state=float(x1_state_loss.detach()),
        x2_state=float(x2_state_loss.detach()),
        x2dot=float(x2dot_loss.detach()),
        x3_range=float(x3_range_loss.detach()),
        fts_range=float(fts_range_loss.detach()),
        x3_range_amp=float(x3_range_amp.detach()),
        fts_range_amp=float(fts_range_amp.detach()),
        cont=0.0,
        x1_rec=float(_relative_rmse_pct(traj[0, :], ode_true[0, :]).detach()),
        x2_rec=float(_relative_rmse_pct(traj[1, :], ode_true[1, :]).detach()),
        x2dot_rec=float(_relative_rmse_pct(x2dot_pred, x2dot_true).detach()),
    )
    return total, parts


def evaluate_split(
    *,
    force_module,
    known_pars: dict[str, Any] | tuple[Any, ...],
    mech_module,
    ode_true: torch.Tensor,
    x2dot_true: torch.Tensor,
    contact_mask: torch.Tensor,
    times: torch.Tensor,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    x1_abs_guard: float | None = None,
    x2_abs_guard: float | None = None,
    loss_indices: torch.Tensor | None = None,
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
        loss_indices=loss_indices,
    )
    return total, parts, traj


__all__ = ["TorchLossParts", "evaluate_split"]
