"""RHS, rollout, and model-side force reconstruction for AFM05 stage1pluslight.

This file intentionally mirrors the AFM04 stage1pluslight rollout interface,
with the AFM05-specific drive term:

    AFM04: Fd*cos(wd*t)
    AFM05: F_actuation(t)

AFM05 has no true/observed x3(t), x3dot(t), delta_dot(t), or Fts(t).  The only
known x3 value is x3_init.  All post-initialization x3-related quantities are
model-side quantities generated during rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

try:
    from scipy.integrate import solve_ivp
except Exception:  # pragma: no cover - environment dependent
    solve_ivp = None


ContactModel = Callable[[np.ndarray, Any], float]


@dataclass(frozen=True)
class AFM05KnownPars:
    k_eff: float
    m_eff: float
    c_eff: float
    actuation_times: np.ndarray
    actuation_values: np.ndarray
    Z: float
    a0: float


def _rk4_step(rhs: Callable[[float, np.ndarray], np.ndarray], t: float, u: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs(t, u)
    k2 = rhs(t + 0.5 * dt, u + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, u + 0.5 * dt * k2)
    k4 = rhs(t + dt, u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _mapping_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def parse_known_pars(known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars) -> AFM05KnownPars:
    """Parse AFM05 known parameters.

    Preferred mapping keys:
      k_eff, m_eff, c_eff, actuation_times, F_actuation, Z, a0

    Tuple fallback order:
      (k_eff, m_eff, c_eff, actuation_times, F_actuation, Z, a0)
    """
    if isinstance(known_pars, AFM05KnownPars):
        parsed = known_pars
    elif isinstance(known_pars, Mapping):
        parsed = AFM05KnownPars(
            k_eff=float(_mapping_get(known_pars, "k_eff", "k")),
            m_eff=float(_mapping_get(known_pars, "m_eff", "m")),
            c_eff=float(_mapping_get(known_pars, "c_eff", "c")),
            actuation_times=np.asarray(
                _mapping_get(known_pars, "actuation_times", "F_actuation_times", "times"),
                dtype=float,
            ),
            actuation_values=np.asarray(
                _mapping_get(known_pars, "F_actuation", "F_actuation_values", "actuation_values"),
                dtype=float,
            ),
            Z=float(_mapping_get(known_pars, "Z", "dist", default=np.nan)),
            a0=float(_mapping_get(known_pars, "a0", default=np.nan)),
        )
    else:
        if len(known_pars) < 5:
            raise ValueError(
                "AFM05 known_pars tuple must contain at least "
                "(k_eff, m_eff, c_eff, actuation_times, F_actuation)"
            )
        parsed = AFM05KnownPars(
            k_eff=float(known_pars[0]),
            m_eff=float(known_pars[1]),
            c_eff=float(known_pars[2]),
            actuation_times=np.asarray(known_pars[3], dtype=float),
            actuation_values=np.asarray(known_pars[4], dtype=float),
            Z=float(known_pars[5]) if len(known_pars) > 5 else float("nan"),
            a0=float(known_pars[6]) if len(known_pars) > 6 else float("nan"),
        )

    times = np.asarray(parsed.actuation_times, dtype=float).reshape(-1)
    values = np.asarray(parsed.actuation_values, dtype=float).reshape(-1)
    if times.size == 0 or values.size == 0:
        raise ValueError("AFM05 F_actuation(t) requires nonempty time and force arrays")
    if times.size != values.size:
        raise ValueError(f"F_actuation times/value length mismatch: {times.size} vs {values.size}")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("F_actuation times must be strictly increasing")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("F_actuation times and values must be finite")
    if not np.isfinite(parsed.k_eff) or not np.isfinite(parsed.m_eff) or not np.isfinite(parsed.c_eff):
        raise ValueError("k_eff, m_eff, and c_eff must be finite")
    if parsed.m_eff == 0.0:
        raise ValueError("m_eff must be nonzero")

    return AFM05KnownPars(
        k_eff=float(parsed.k_eff),
        m_eff=float(parsed.m_eff),
        c_eff=float(parsed.c_eff),
        actuation_times=times,
        actuation_values=values,
        Z=float(parsed.Z),
        a0=float(parsed.a0),
    )


def _actuation_at_parsed(pars: AFM05KnownPars, t: float) -> float:
    if pars.actuation_times.size == 1:
        return float(pars.actuation_values[0])
    return float(
        np.interp(
            float(t),
            pars.actuation_times,
            pars.actuation_values,
            left=float(pars.actuation_values[0]),
            right=float(pars.actuation_values[-1]),
        )
    )


def actuation_at(known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars, t: float) -> float:
    return _actuation_at_parsed(parse_known_pars(known_pars), t)


def actuation_values_at(known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars, times: np.ndarray) -> np.ndarray:
    pars = parse_known_pars(known_pars)
    query = np.asarray(times, dtype=float)
    if pars.actuation_times.size == 1:
        return np.full(query.shape, float(pars.actuation_values[0]), dtype=float)
    return np.interp(
        query,
        pars.actuation_times,
        pars.actuation_values,
        left=float(pars.actuation_values[0]),
        right=float(pars.actuation_values[-1]),
    )


def nn_input_from_state(u: np.ndarray) -> np.ndarray:
    return np.asarray(u[:3], dtype=float)


def s_from_state(u: np.ndarray, known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars) -> float:
    pars = parse_known_pars(known_pars)
    return float(pars.Z + float(u[0]) - float(u[2]))


def contact_from_state(u: np.ndarray, known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars) -> bool:
    pars = parse_known_pars(known_pars)
    if not np.isfinite(pars.a0):
        return False
    return bool(s_from_state(u, pars) <= pars.a0)


def _call_contact_model(model: ContactModel | None, u: np.ndarray, model_params: Any) -> float:
    if model is None:
        raise ValueError("AFM05 requires an explicit contact model/KAN; no true Fts fallback exists")
    try:
        out = model(nn_input_from_state(u), model_params)
    except TypeError:
        out = model(nn_input_from_state(u))
    if isinstance(out, np.ndarray):
        return float(np.asarray(out, dtype=float).reshape(-1)[0])
    if isinstance(out, (list, tuple)):
        return float(out[0])
    return float(out)


def f_contact_from_state(
    u: np.ndarray,
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars,
    *,
    zero_contact_override: bool = False,
) -> float:
    _ = (mech, known_pars)
    if zero_contact_override:
        return 0.0
    return _call_contact_model(model, u, model_params)


def model_side_terms_from_state(
    u: np.ndarray,
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars,
    *,
    zero_contact_override: bool = False,
) -> tuple[float, float, float, float, bool]:
    """Return model-side (Fts, x3dot, delta_dot, s, contact).

    These are constructed/model-side quantities in AFM05, never observed truth.
    """
    u = np.asarray(u, dtype=float)
    ks, cs = np.asarray(mech, dtype=float)
    f_contact = f_contact_from_state(
        u,
        np.array([ks, cs], dtype=float),
        model,
        model_params,
        known_pars,
        zero_contact_override=zero_contact_override,
    )
    x3dot = float((-f_contact - ks * u[2]) / cs)
    pars = parse_known_pars(known_pars)
    s = float(pars.Z + u[0] - u[2])
    contact = bool(np.isfinite(pars.a0) and s <= pars.a0)
    if contact:
        delta_dot = float(x3dot - u[1])
    else:
        delta_dot = 0.0
    return float(f_contact), float(x3dot), float(delta_dot), float(s), contact


def make_uode_rhs(
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars,
    *,
    zero_contact_override: bool = False,
) -> Callable[[float, np.ndarray], np.ndarray]:
    pars = parse_known_pars(known_pars)
    ks, cs = np.asarray(mech, dtype=float)
    mech_arr = np.array([ks, cs], dtype=float)
    act_times = pars.actuation_times
    act_values = pars.actuation_values
    act_left = float(act_values[0])
    act_right = float(act_values[-1])
    act_is_scalar = act_times.size == 1

    def rhs(t: float, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if zero_contact_override:
            f_contact = 0.0
        else:
            f_contact = f_contact_from_state(
                u,
                mech_arr,
                model,
                model_params,
                pars,
                zero_contact_override=False,
            )
        du1 = u[1]
        if act_is_scalar:
            f_act = act_left
        else:
            f_act = float(np.interp(float(t), act_times, act_values, left=act_left, right=act_right))
        du2 = (f_act - pars.k_eff * u[0] - pars.c_eff * u[1] + f_contact) / pars.m_eff
        du3 = (-f_contact - ks * u[2]) / cs
        return np.array([du1, du2, du3], dtype=float)

    return rhs


def x2dot_rhs(
    u: np.ndarray,
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars,
    t: float,
    *,
    zero_contact_override: bool = False,
) -> float:
    pars = parse_known_pars(known_pars)
    f_contact = f_contact_from_state(u, mech, model, model_params, pars, zero_contact_override=zero_contact_override)
    return float((_actuation_at_parsed(pars, float(t)) - pars.k_eff * u[0] - pars.c_eff * u[1] + f_contact) / pars.m_eff)


def fts_pred_from_state(
    u: np.ndarray,
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars,
    *,
    zero_contact_override: bool = False,
) -> float:
    return f_contact_from_state(u, mech, model, model_params, known_pars, zero_contact_override=zero_contact_override)


def fts_from_x2dot_signal(
    uhat: np.ndarray,
    x2dot_pred: np.ndarray,
    times: np.ndarray,
    known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars,
) -> np.ndarray:
    pars = parse_known_pars(known_pars)
    x1 = np.asarray(uhat[0, :], dtype=float)
    x2 = np.asarray(uhat[1, :], dtype=float)
    times = np.asarray(times, dtype=float)
    x2dot_pred = np.asarray(x2dot_pred, dtype=float)
    return pars.m_eff * x2dot_pred - actuation_values_at(pars, times) + pars.k_eff * x1 + pars.c_eff * x2


def rollout_single_shooting(
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: Mapping[str, Any] | tuple[Any, ...] | AFM05KnownPars,
    u0: np.ndarray,
    times: np.ndarray,
    *,
    zero_contact_override: bool = False,
    ode_solver: str = "Radau",
    ode_fallback_solver: str = "BDF",
    ode_rtol: float = 1.0e-8,
    ode_atol: float = 1.0e-8,
    ode_max_step: float = 0.0,
    x1_abs_guard: float | None = None,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1D array with at least two points")
    rhs = make_uode_rhs(mech, model, model_params, known_pars, zero_contact_override=zero_contact_override)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    x1_guard = None
    if x1_abs_guard is not None and np.isfinite(float(x1_abs_guard)) and float(x1_abs_guard) > 0.0:
        x1_guard = float(x1_abs_guard)

    def _raise_if_x1_guarded(u: np.ndarray) -> None:
        if x1_guard is not None and abs(float(np.asarray(u, dtype=float)[0])) > x1_guard:
            raise RuntimeError(f"x1_guard_triggered:{abs(float(np.asarray(u, dtype=float)[0])):.6e}>{x1_guard:.6e}")

    method_primary = str(ode_solver).strip() or "Radau"
    method_fallback = str(ode_fallback_solver).strip()
    methods: list[str] = [method_primary]
    if method_fallback and method_fallback.lower() != method_primary.lower():
        methods.append(method_fallback)

    if method_primary.lower() == "rk4":
        u = np.asarray(u0, dtype=float).copy()
        _raise_if_x1_guarded(u)
        out = np.empty((3, times.size), dtype=float)
        out[:, 0] = u
        for i in range(times.size - 1):
            t = float(times[i])
            dt = float(times[i + 1] - times[i])
            u = _rk4_step(rhs, t, u, dt)
            _raise_if_x1_guarded(u)
            out[:, i + 1] = u
        return out

    if solve_ivp is None:
        raise ImportError("scipy is required for adaptive rollout solvers (e.g. Radau/BDF). Install scipy or set ode_solver='rk4'.")

    t_span = (float(times[0]), float(times[-1]))
    y0 = np.asarray(u0, dtype=float).copy()
    _raise_if_x1_guarded(y0)
    max_step = float(ode_max_step)
    max_step = np.inf if not np.isfinite(max_step) or max_step <= 0.0 else max_step
    last_error = "adaptive_solver_not_attempted"

    def rhs_scipy(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(rhs(float(t), np.asarray(y, dtype=float)), dtype=float)

    events = None
    if x1_guard is not None:
        def x1_amp_event(_t: float, y: np.ndarray) -> float:
            return x1_guard - abs(float(np.asarray(y, dtype=float)[0]))

        x1_amp_event.terminal = True  # type: ignore[attr-defined]
        x1_amp_event.direction = -1.0  # type: ignore[attr-defined]
        events = x1_amp_event

    for method in methods:
        try:
            sol = solve_ivp(
                rhs_scipy,
                t_span,
                y0,
                method=method,
                t_eval=times,
                vectorized=False,
                rtol=float(ode_rtol),
                atol=float(ode_atol),
                max_step=max_step,
                events=events,
            )
        except Exception as err:
            last_error = f"{method}_exception:{err}"
            continue
        if events is not None and getattr(sol, "t_events", None) and sol.t_events[0].size > 0:
            y_event = np.asarray(sol.y_events[0][0], dtype=float)
            raise RuntimeError(f"x1_guard_triggered:{abs(float(y_event[0])):.6e}>{float(x1_guard):.6e}")
        if not bool(sol.success):
            last_error = f"{method}_failed:{getattr(sol, 'message', 'unknown')}"
            continue
        if sol.y.shape != (3, times.size):
            last_error = f"{method}_shape_mismatch:{sol.y.shape}"
            continue
        if not np.all(np.isfinite(sol.y)):
            last_error = f"{method}_nonfinite_solution"
            continue
        return np.asarray(sol.y, dtype=float)

    raise RuntimeError(f"adaptive rollout failed: {last_error}")


__all__ = [
    "AFM05KnownPars",
    "ContactModel",
    "actuation_at",
    "actuation_values_at",
    "contact_from_state",
    "f_contact_from_state",
    "fts_from_x2dot_signal",
    "fts_pred_from_state",
    "make_uode_rhs",
    "model_side_terms_from_state",
    "nn_input_from_state",
    "parse_known_pars",
    "rollout_single_shooting",
    "s_from_state",
    "x2dot_rhs",
]
