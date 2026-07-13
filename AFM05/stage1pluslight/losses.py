"""AFM05 stage1pluslight loss kernel, cloned from AFM04 then adapted.

AFM05 keeps the AFM04 training-side structure:
- supervise x1 and x2 trajectory;
- supervise x2dot, the tip/cantilever acceleration derived from x2(t);
- keep model-side x3 and Fts range penalties;
- optionally use contact/noncontact loss weighting.

AFM05 does not have true/observed x3(t), x3dot(t), delta_dot(t), or Fts(t).
Accordingly, true-side diagnostics are not emitted here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from AFM05.stage1pluslight.grid import CS_BOUNDS, KS_BOUNDS
from AFM05.stage1pluslight.rollout import actuation_values_at, fts_from_x2dot_signal, rollout_single_shooting, x2dot_rhs


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
    x2_rec: float
    x2dot_rec: float


INF_LOSS_PARTS = LossParts(
    state=float("inf"),
    x2dot=float("inf"),
    x3_range=float("inf"),
    fts_range=float("inf"),
    cont=float("inf"),
    x1_rec=float("inf"),
    x2_rec=float("inf"),
    x2dot_rec=float("inf"),
)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(x, dtype=float)
    clipped = np.clip(arr, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(-clipped))
    if np.isscalar(x):
        return float(out)
    return out


def bound_param(raw: float, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * sigmoid(raw))


def softplus(x: float | np.ndarray, eps: float) -> float | np.ndarray:
    arr = np.asarray(x, dtype=float)
    z = arr / float(eps)
    out = np.where(
        z > 50.0,
        arr,
        np.where(z < -50.0, np.zeros_like(arr), float(eps) * np.log1p(np.exp(z))),
    )
    if np.isscalar(x):
        return float(out)
    return out


def relative_rmse_pct(err_sum: float, truth_sum: float, count: int, eps: float) -> float:
    denom = math.sqrt(float(truth_sum) / float(count)) + float(eps)
    return 100.0 * math.sqrt(float(err_sum) / float(count)) / denom


def _safe_rms_np(values: np.ndarray, *, floor: float = SCALE_EPS) -> float | np.ndarray:
    arr = np.asarray(values, dtype=float)
    rms = np.sqrt(np.mean(np.square(arr), axis=-1))
    return np.maximum(rms, float(floor))


def _mean_abs_amp_np(values: np.ndarray, *, floor: float) -> float:
    amp = float(np.mean(np.abs(np.asarray(values, dtype=float).reshape(-1))))
    return float(max(amp, float(floor)))


def group_ranges(n: int, group_size: int) -> list[np.ndarray]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return [np.arange(start, min(start + group_size, n), dtype=int) for start in range(0, n, group_size)]


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


def _loss_weights(contact_mask: np.ndarray | None, n: int) -> np.ndarray:
    if contact_mask is None:
        return np.ones(int(n), dtype=float)
    contact = np.asarray(contact_mask, dtype=bool)
    if contact.size != int(n):
        raise ValueError(f"contact_mask length mismatch: {contact.size} vs {n}")
    return np.where(contact, CONTACT_LOSS_WEIGHT, NONCONTACT_LOSS_WEIGHT).astype(float)


def _loss_from_pred(
    uhat: np.ndarray,
    ode_data: np.ndarray,
    x2dot_data: np.ndarray,
    contact_mask: np.ndarray | None,
    times: np.ndarray,
    state12_scale: np.ndarray,
    x2dot_scale: float,
    x3_scale: float,
    mech: np.ndarray,
    model: Any,
    model_params: Any,
    known_pars: tuple[Any, ...] | Mapping[str, Any],
    *,
    zero_contact_override: bool,
    x3_range_amp: float,
    fts_range_amp: float,
) -> tuple[float, float, float, float, np.ndarray]:
    weights = _loss_weights(contact_mask, times.size)
    weights_sum = float(np.sum(weights))
    if not np.isfinite(weights_sum) or weights_sum <= 0.0:
        return float("inf"), float("inf"), float("inf"), float("inf"), np.full(times.shape, np.nan)

    _ = state12_scale
    state_scale = np.maximum(_safe_rms_np(ode_data[0:2, :]), SCALE_EPS)
    state_err = np.sum(np.square((ode_data[0:2, :] - uhat[0:2, :]) / state_scale[:, None]), axis=0)
    total_state = float(np.sum(weights * state_err) / weights_sum)

    x2dot_pred = np.array(
        [
            x2dot_rhs(uhat[:, j], mech, model, model_params, known_pars, float(times[j]), zero_contact_override=zero_contact_override)
            for j in range(times.size)
        ],
        dtype=float,
    )
    _ = x2dot_scale
    x2dot_scale_eval = float(max(float(_safe_rms_np(x2dot_data)), SCALE_EPS))
    x2_err = np.square((x2dot_data - x2dot_pred) / x2dot_scale_eval)
    total_x2dot = float(np.sum(weights * x2_err) / weights_sum)

    fts_pred = fts_from_x2dot_signal(uhat, x2dot_pred, times, known_pars)
    fts_range_amp = float(max(float(fts_range_amp), FTS_RANGE_EPS))
    fts_exceed = np.abs(fts_pred) - fts_range_amp
    fts_pen = np.square(np.asarray(softplus(fts_exceed, FTS_RANGE_EPS), dtype=float) / fts_range_amp)
    total_ftsrange = float(FTS_RANGE_WEIGHT * (np.sum(weights * fts_pen) / weights_sum))

    _ = x3_scale
    x3_range_amp = float(max(float(x3_range_amp), X3_RANGE_EPS))
    exceed = np.abs(uhat[2, :]) - x3_range_amp
    range_pen = np.square(np.asarray(softplus(exceed, X3_RANGE_EPS), dtype=float) / x3_range_amp)
    total_x3range = float(X3_RANGE_WEIGHT * (np.sum(weights * range_pen) / weights_sum))

    return total_state, total_x2dot, total_x3range, total_ftsrange, x2dot_pred


def loss_single_or_ms(
    theta: Any,
    ode_data: np.ndarray,
    x2dot_data: np.ndarray,
    contact_mask: np.ndarray | None,
    times: np.ndarray,
    state12_scale: np.ndarray,
    x2dot_scale: float,
    x3_scale: float,
    use_multiple_shooting: bool,
    ms_group_size: int,
    ms_continuity_term: float,
    model: Any,
    known_pars: tuple[Any, ...] | Mapping[str, Any],
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
            bound_param(float(mech_raw[0]), KS_BOUNDS[0], KS_BOUNDS[1]),
            bound_param(float(mech_raw[1]), CS_BOUNDS[0], CS_BOUNDS[1]),
        ],
        dtype=float,
    )
    model_params = _theta_field(theta, "model_params", None)

    ode_data = np.asarray(ode_data, dtype=float)
    x2dot_data = np.asarray(x2dot_data, dtype=float)
    times = np.asarray(times, dtype=float)
    if contact_mask is None:
        contact_mask_arr: np.ndarray | None = None
    else:
        contact_mask_arr = np.asarray(contact_mask, dtype=bool)
    if loss_indices is None:
        loss_idx = None
    else:
        loss_idx = np.asarray(loss_indices, dtype=int)
        if loss_idx.size == 0:
            return float("inf"), INF_LOSS_PARTS
    state12_scale = np.maximum(np.asarray(state12_scale, dtype=float), SCALE_EPS)
    x2dot_scale = float(max(float(x2dot_scale), SCALE_EPS))
    x3_scale = float(max(float(x3_scale), SCALE_EPS))
    x3_range_amp_full = _mean_abs_amp_np(ode_data[0, :], floor=X3_RANGE_EPS)
    fts_range_amp_full = _mean_abs_amp_np(actuation_values_at(known_pars, times), floor=FTS_RANGE_EPS)

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
        contact_loss = contact_mask_arr
        times_loss = times
    else:
        pred_loss = pred_full[:, loss_idx]
        ode_loss = ode_data[:, loss_idx]
        x2dot_loss = x2dot_data[loss_idx]
        contact_loss = None if contact_mask_arr is None else contact_mask_arr[loss_idx]
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
        x3_range_amp=x3_range_amp_full,
        fts_range_amp=fts_range_amp_full,
    )

    x1_err_sum = float(np.sum(np.square(ode_loss[0, :] - pred_loss[0, :])))
    x1_truth_sum = float(np.sum(np.square(ode_loss[0, :])))
    x2_err_sum = float(np.sum(np.square(ode_loss[1, :] - pred_loss[1, :])))
    x2_truth_sum = float(np.sum(np.square(ode_loss[1, :])))
    count = max(1, len(times_loss))
    x1_rec = relative_rmse_pct(x1_err_sum, x1_truth_sum, count, SCALE_EPS)
    x2_rec = relative_rmse_pct(x2_err_sum, x2_truth_sum, count, SCALE_EPS)
    x2dot_pred_loss = np.array(
        [
            x2dot_rhs(
                pred_loss[:, j],
                mech,
                model,
                model_params,
                known_pars,
                float(times_loss[j]),
                zero_contact_override=zero_contact_override,
            )
            for j in range(times_loss.size)
        ],
        dtype=float,
    )
    x2dot_err_sum = float(np.sum(np.square(x2dot_loss - x2dot_pred_loss)))
    x2dot_truth_sum = float(np.sum(np.square(x2dot_loss)))
    x2dot_rec = relative_rmse_pct(x2dot_err_sum, x2dot_truth_sum, count, SCALE_EPS)

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
        x2_rec=float(x2_rec),
        x2dot_rec=float(x2dot_rec),
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
