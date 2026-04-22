"""RHS, rollout, and diagnostic force reconstruction for AFM04 stage1pluslight."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from AFM04.datasets.non_perturbed_dataset_generator import rk4_step
from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_functions import f_ts_from_state

try:
    from scipy.integrate import solve_ivp
except Exception:  # pragma: no cover - environment dependent
    solve_ivp = None


ContactModel = Callable[[np.ndarray, Any], float]


def nn_input_from_state(u: np.ndarray) -> np.ndarray:
    return np.asarray(u[:3], dtype=float)


def _call_contact_model(model: ContactModel | None, u: np.ndarray, model_params: Any) -> float:
    if model is None:
        raise ValueError("contact model is None")
    try:
        out = model(nn_input_from_state(u), model_params)
    except TypeError:
        out = model(nn_input_from_state(u))
    if isinstance(out, np.ndarray):
        return float(np.asarray(out, dtype=float).reshape(-1)[0])
    if isinstance(out, (list, tuple)):
        return float(out[0])
    return float(out)


def _truth_force_terms_from_state(
    u: np.ndarray,
    mech: np.ndarray,
    known_pars: tuple[float, ...],
) -> tuple[float, float, float, float]:
    k, wd, m, c, Fd, R, dist, Estar, eta_star, A, a0, beta = known_pars
    _ = (k, wd, m, c, Fd)
    ks, cs = np.asarray(mech, dtype=float)
    f_ts, x3dot, delta_dot, s = f_ts_from_state(
        float(u[0]),
        float(u[1]),
        float(u[2]),
        dist=float(dist),
        Estar=float(Estar),
        eta_star=float(eta_star),
        R=float(R),
        A=float(A),
        a0=float(a0),
        beta=float(beta),
        ks=float(ks),
        cs=float(cs),
    )
    return float(f_ts), float(x3dot), float(delta_dot), float(s)


def f_contact_from_state(
    u: np.ndarray,
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: tuple[float, ...],
    *,
    zero_contact_override: bool = False,
) -> float:
    if zero_contact_override:
        return 0.0
    if model is None:
        f_ts, _, _, _ = _truth_force_terms_from_state(u, mech, known_pars)
        return f_ts
    return _call_contact_model(model, u, model_params)


def make_uode_rhs(
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: tuple[float, ...],
    *,
    zero_contact_override: bool = False,
) -> Callable[[float, np.ndarray], np.ndarray]:
    k, wd, m, c, Fd, R, dist, Estar, eta_star, A, a0, beta = known_pars
    ks, cs = np.asarray(mech, dtype=float)
    _ = (R, dist, Estar, eta_star, A, a0, beta)

    def rhs(t: float, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if zero_contact_override:
            f_contact = 0.0
            du3 = (-ks * u[2]) / cs
        elif model is None:
            f_contact, du3, _, _ = _truth_force_terms_from_state(u, np.array([ks, cs], dtype=float), known_pars)
        else:
            f_contact = f_contact_from_state(
                u,
                np.array([ks, cs], dtype=float),
                model,
                model_params,
                known_pars,
                zero_contact_override=False,
            )
            du3 = (-f_contact - ks * u[2]) / cs
        du1 = u[1]
        du2 = (Fd * math.cos(wd * t) - k * u[0] - c * u[1] + f_contact) / m
        return np.array([du1, du2, du3], dtype=float)

    return rhs


def x2dot_rhs(
    u: np.ndarray,
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: tuple[float, ...],
    t: float,
    *,
    zero_contact_override: bool = False,
) -> float:
    k, wd, m, c, Fd, _, _, _, _, _, _, _ = known_pars
    f_contact = f_contact_from_state(u, mech, model, model_params, known_pars, zero_contact_override=zero_contact_override)
    return float((Fd * math.cos(wd * t) - k * u[0] - c * u[1] + f_contact) / m)


def fts_pred_from_state(
    u: np.ndarray,
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: tuple[float, ...],
    *,
    zero_contact_override: bool = False,
) -> float:
    return f_contact_from_state(u, mech, model, model_params, known_pars, zero_contact_override=zero_contact_override)


def fts_from_x2dot_signal(uhat: np.ndarray, x2dot_pred: np.ndarray, times: np.ndarray, known_pars: tuple[float, ...]) -> np.ndarray:
    k, wd, m, c, Fd, _, _, _, _, _, _, _ = known_pars
    x1 = np.asarray(uhat[0, :], dtype=float)
    x2 = np.asarray(uhat[1, :], dtype=float)
    times = np.asarray(times, dtype=float)
    x2dot_pred = np.asarray(x2dot_pred, dtype=float)
    return m * x2dot_pred - Fd * np.cos(wd * times) + k * x1 + c * x2


def rollout_single_shooting(
    mech: np.ndarray,
    model: ContactModel | None,
    model_params: Any,
    known_pars: tuple[float, ...],
    u0: np.ndarray,
    times: np.ndarray,
    *,
    zero_contact_override: bool = False,
    ode_solver: str = "Radau",
    ode_fallback_solver: str = "BDF",
    ode_rtol: float = 1.0e-8,
    ode_atol: float = 1.0e-8,
    ode_max_step: float = 0.0,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1D array with at least two points")
    rhs = make_uode_rhs(mech, model, model_params, known_pars, zero_contact_override=zero_contact_override)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")

    method_primary = str(ode_solver).strip() or "Radau"
    method_fallback = str(ode_fallback_solver).strip()
    methods: list[str] = [method_primary]
    if method_fallback and method_fallback.lower() != method_primary.lower():
        methods.append(method_fallback)

    if method_primary.lower() == "rk4":
        u = np.asarray(u0, dtype=float).copy()
        out = np.empty((3, times.size), dtype=float)
        out[:, 0] = u
        for i in range(times.size - 1):
            t = float(times[i])
            dt = float(times[i + 1] - times[i])
            u = rk4_step(rhs, t, u, dt)
            out[:, i + 1] = u
        return out

    if solve_ivp is None:
        raise ImportError("scipy is required for adaptive rollout solvers (e.g. Radau/BDF). Install scipy or set ode_solver='rk4'.")

    t_span = (float(times[0]), float(times[-1]))
    y0 = np.asarray(u0, dtype=float).copy()
    max_step = float(ode_max_step)
    max_step = np.inf if not np.isfinite(max_step) or max_step <= 0.0 else max_step
    last_error = "adaptive_solver_not_attempted"

    def rhs_scipy(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(rhs(float(t), np.asarray(y, dtype=float)), dtype=float)

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
            )
        except Exception as err:
            last_error = f"{method}_exception:{err}"
            continue
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
    "ContactModel",
    "f_contact_from_state",
    "fts_from_x2dot_signal",
    "fts_pred_from_state",
    "make_uode_rhs",
    "nn_input_from_state",
    "rollout_single_shooting",
    "x2dot_rhs",
]
