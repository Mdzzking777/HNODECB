"""Loss kernel cloned from AFM03 stage1pluslight with AFM04 physics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from AFM04.stage1pluslight.mathutils import (
    bound_param,
    group_ranges,
    relative_rmse_pct,
    softplus,
)
from AFM04.stage1pluslight.rollout import fts_from_x2dot_signal, rollout_single_shooting, x2dot_rhs


X3_RANGE_AMP = 100e-9
X3_RANGE_EPS = 1e-9
X3_RANGE_WEIGHT = 1.0
FTS_RANGE_AMP = 1e-8
FTS_RANGE_EPS = 1e-10
FTS_RANGE_WEIGHT = 1.0
CONTACT_LOSS_WEIGHT = 1.0
NONCONTACT_LOSS_WEIGHT = 1.0
SCALE_EPS = 1e-9


@dataclass(frozen=True)
class LossParts:
    state: float
    x2dot: float
    x3_range: float
    fts_range: float
    cont: float
    x1_rec: float
    x3_rec: float


INF_LOSS_PARTS = LossParts(
    state=float("inf"),
    x2dot=float("inf"),
    x3_range=float("inf"),
    fts_range=float("inf"),
    cont=float("inf"),
    x1_rec=float("inf"),
    x3_rec=float("inf"),
)


def _theta_field(theta: Any, key: str, default: Any = None) -> Any:
    if isinstance(theta, Mapping):
        return theta.get(key, default)
    return getattr(theta, key, default)


def _model_l2_penalty(model_params: Any) -> float:
    if model_params is None:
        return 0.0
    if isinstance(model_params, np.ndarray):
        return float(np.sum(np.square(model_params)))
    if isinstance(model_params, (list, tuple)):
        arr = np.asarray(model_params, dtype=float)
        return float(np.sum(np.square(arr)))
    return 0.0


def _store_pred_traj(ref: Any, value: np.ndarray) -> None:
    if ref is None:
        return
    if isinstance(ref, dict):
        ref["traj"] = np.asarray(value, dtype=float).copy()
        return
    if isinstance(ref, list):
        ref.clear()
        ref.append(np.asarray(value, dtype=float).copy())
        return
    if hasattr(ref, "__setitem__"):
        ref[0] = np.asarray(value, dtype=float).copy()


def _loss_from_pred(
    uhat: np.ndarray,
    ode_data: np.ndarray,
    x2dot_data: np.ndarray,
    contact_mask: np.ndarray,
    times: np.ndarray,
    state12_scale: np.ndarray,
    x2dot_scale: float,
    x3_scale: float,
    mech: np.ndarray,
    model: Any,
    model_params: Any,
    known_pars: tuple[float, ...],
    *,
    zero_contact_override: bool,
) -> tuple[float, float, float, float, np.ndarray]:
    weights = np.where(contact_mask, CONTACT_LOSS_WEIGHT, NONCONTACT_LOSS_WEIGHT)
    weights_sum = float(np.sum(weights))
    if not np.isfinite(weights_sum) or weights_sum <= 0.0:
        return float("inf"), float("inf"), float("inf"), float("inf"), np.full(times.shape, np.nan)

    state_err = np.sum(np.square((ode_data[0:2, :] - uhat[0:2, :]) / state12_scale[:, None]), axis=0)
    total_state = float(np.sum(weights * state_err) / weights_sum)

    x2dot_pred = np.array(
        [
            x2dot_rhs(uhat[:, j], mech, model, model_params, known_pars, float(times[j]), zero_contact_override=zero_contact_override)
            for j in range(times.size)
        ],
        dtype=float,
    )
    x2_err = np.square((x2dot_data - x2dot_pred) / float(x2dot_scale))
    total_x2dot = float(np.sum(weights * x2_err) / weights_sum)

    fts_pred = fts_from_x2dot_signal(uhat, x2dot_pred, times, known_pars)
    fts_exceed = np.abs(fts_pred) - FTS_RANGE_AMP
    fts_pen = np.square(np.asarray(softplus(fts_exceed, FTS_RANGE_EPS), dtype=float) / FTS_RANGE_AMP)
    total_ftsrange = float(FTS_RANGE_WEIGHT * (np.sum(weights * fts_pen) / weights_sum))

    exceed = np.abs(uhat[2, :]) - X3_RANGE_AMP
    range_pen = np.square(np.asarray(softplus(exceed, X3_RANGE_EPS), dtype=float) / float(x3_scale))
    total_x3range = float(X3_RANGE_WEIGHT * (np.sum(weights * range_pen) / weights_sum))

    return total_state, total_x2dot, total_x3range, total_ftsrange, x2dot_pred


def loss_single_or_ms(
    theta: Any,
    ode_data: np.ndarray,
    x2dot_data: np.ndarray,
    contact_mask: np.ndarray,
    times: np.ndarray,
    state12_scale: np.ndarray,
    x2dot_scale: float,
    x3_scale: float,
    use_multiple_shooting: bool,
    ms_group_size: int,
    ms_continuity_term: float,
    model: Any,
    known_pars: tuple[float, ...],
    x3_t0_val: float,
    *,
    l2_weight: float = 0.0,
    zero_contact_override: bool = False,
    pred_traj_ref: Any = None,
    ode_solver: str = "Radau",
    ode_fallback_solver: str = "BDF",
    ode_rtol: float = 1.0e-8,
    ode_atol: float = 1.0e-8,
    ode_max_step: float = 0.0,
    loss_indices: np.ndarray | None = None,
) -> tuple[float, LossParts]:
    mech_raw = np.asarray(_theta_field(theta, "mech_raw"), dtype=float)
    mech = np.array(
        [
            bound_param(float(mech_raw[0]), 0.005, 5.0),
            bound_param(float(mech_raw[1]), 5.0e-8, 5.0e-6),
        ],
        dtype=float,
    )
    model_params = _theta_field(theta, "model_params", None)

    ode_data = np.asarray(ode_data, dtype=float)
    x2dot_data = np.asarray(x2dot_data, dtype=float)
    contact_mask = np.asarray(contact_mask, dtype=bool)
    times = np.asarray(times, dtype=float)
    if loss_indices is None:
        loss_idx = None
    else:
        loss_idx = np.asarray(loss_indices, dtype=int)
        if loss_idx.size == 0:
            return float("inf"), INF_LOSS_PARTS
    state12_scale = np.maximum(np.asarray(state12_scale, dtype=float), SCALE_EPS)
    x2dot_scale = float(max(float(x2dot_scale), SCALE_EPS))
    x3_scale = float(max(float(x3_scale), SCALE_EPS))

    total_cont = 0.0
    pred_full = np.empty_like(ode_data)

    try:
        if use_multiple_shooting:
            ranges = group_ranges(len(times), ms_group_size)
            preds: list[np.ndarray] = []
            for rg in ranges:
                u0 = np.array([ode_data[0, rg[0]], ode_data[1, rg[0]], x3_t0_val], dtype=float)
                uhat = rollout_single_shooting(
                    mech,
                    model,
                    model_params,
                    known_pars,
                    u0,
                    times[rg],
                    zero_contact_override=zero_contact_override,
                    ode_solver=ode_solver,
                    ode_fallback_solver=ode_fallback_solver,
                    ode_rtol=ode_rtol,
                    ode_atol=ode_atol,
                    ode_max_step=ode_max_step,
                )
                preds.append(uhat)
                pred_full[:, rg] = uhat
            for i in range(1, len(preds)):
                total_cont += float(ms_continuity_term * np.sum(np.square(preds[i - 1][:, -1] - preds[i][:, 0])))
        else:
            u0 = np.array([ode_data[0, 0], ode_data[1, 0], x3_t0_val], dtype=float)
            pred_full = rollout_single_shooting(
                mech,
                model,
                model_params,
                known_pars,
                u0,
                times,
                zero_contact_override=zero_contact_override,
                ode_solver=ode_solver,
                ode_fallback_solver=ode_fallback_solver,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                ode_max_step=ode_max_step,
            )
    except Exception:
        return float("inf"), INF_LOSS_PARTS

    if not np.all(np.isfinite(pred_full)):
        return float("inf"), INF_LOSS_PARTS

    _store_pred_traj(pred_traj_ref, pred_full)

    if loss_idx is None:
        pred_loss = pred_full
        ode_loss = ode_data
        x2dot_loss = x2dot_data
        contact_loss = contact_mask
        times_loss = times
    else:
        pred_loss = pred_full[:, loss_idx]
        ode_loss = ode_data[:, loss_idx]
        x2dot_loss = x2dot_data[loss_idx]
        contact_loss = contact_mask[loss_idx]
        times_loss = times[loss_idx]

    total_state, total_x2dot, total_x3range, total_ftsrange, _ = _loss_from_pred(
        pred_loss,
        ode_loss,
        x2dot_loss,
        contact_loss,
        times_loss,
        state12_scale,
        x2dot_scale,
        x3_scale,
        mech,
        model,
        model_params,
        known_pars,
        zero_contact_override=zero_contact_override,
    )

    x1_err_sum = float(np.sum(np.square(ode_loss[0, :] - pred_loss[0, :])))
    x1_truth_sum = float(np.sum(np.square(ode_loss[0, :])))
    x3_err_sum = float(np.sum(np.square(ode_loss[2, :] - pred_loss[2, :])))
    x3_truth_sum = float(np.sum(np.square(ode_loss[2, :])))
    count = max(1, len(times_loss))
    x1_rec = relative_rmse_pct(x1_err_sum, x1_truth_sum, count, SCALE_EPS)
    x3_rec = relative_rmse_pct(x3_err_sum, x3_truth_sum, count, SCALE_EPS)

    l2_penalty = float(l2_weight) * _model_l2_penalty(model_params)
    total = float(total_state + total_x2dot + total_x3range + total_ftsrange + total_cont + l2_penalty)
    if not np.isfinite(total):
        return float("inf"), INF_LOSS_PARTS

    parts = LossParts(
        state=float(total_state),
        x2dot=float(total_x2dot),
        x3_range=float(total_x3range),
        fts_range=float(total_ftsrange),
        cont=float(total_cont),
        x1_rec=float(x1_rec),
        x3_rec=float(x3_rec),
    )
    return total, parts


def parts_as_dict(parts: LossParts) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(parts).items()}


__all__ = [
    "CONTACT_LOSS_WEIGHT",
    "FTS_RANGE_AMP",
    "FTS_RANGE_EPS",
    "FTS_RANGE_WEIGHT",
    "INF_LOSS_PARTS",
    "LossParts",
    "NONCONTACT_LOSS_WEIGHT",
    "SCALE_EPS",
    "X3_RANGE_AMP",
    "X3_RANGE_EPS",
    "X3_RANGE_WEIGHT",
    "loss_single_or_ms",
    "parts_as_dict",
]
