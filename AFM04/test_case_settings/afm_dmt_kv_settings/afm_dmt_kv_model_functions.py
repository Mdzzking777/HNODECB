"""Shared AFM04 DMT-KV physics functions.

This file now mirrors `Datageneration/DMT_KV`:

    g(s) = sigmoid(beta * (s - a0))
    delta = max(a0 - s, 0)

    F_static(s) =
        -A*R / (6 * (g(s) * (s - a0) + a0)^2)
        + (1 - g(s)) * (4/3) * Estar * sqrt(R) * delta^(3/2)

    F_ts = F_static + eta_star * sqrt(R) * delta^(1/2) * delta_dot

with the Kelvin-Voigt surface dynamics solved explicitly inside the shared
truth layer so that x3dot stays in explicit state-space form.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - scalar fallback path
    torch = None

from .afm_dmt_kv_model_settings import original_parameters


def _is_tensor(x: Any) -> bool:
    return torch is not None and torch.is_tensor(x)


def _field_or(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[key]
    return getattr(obj, key)


def _split_state(u: Any) -> tuple[Any, Any, Any]:
    if _is_tensor(u):
        return u[0], u[1], u[2]
    return u[0], u[1], u[2]


def _pack_state(x1dot: Any, x2dot: Any, x3dot: Any, like: Any) -> Any:
    if _is_tensor(like):
        return torch.stack((x1dot, x2dot, x3dot))
    return [x1dot, x2dot, x3dot]


def _as_like_tensor(values: Sequence[float], like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def _mul_parameters(lhs: Any, rhs: Sequence[float]) -> Any:
    if _is_tensor(lhs):
        return lhs * _as_like_tensor(rhs, lhs)
    return [a * b for a, b in zip(lhs, rhs)]


def _network_scalar(out: Any) -> Any:
    if isinstance(out, tuple):
        out = out[0]
    if _is_tensor(out):
        if out.ndim == 0:
            return out
        return out.reshape(-1)[0]
    if isinstance(out, Sequence):
        return out[0]
    return out


def sigmoid(x: Any) -> Any:
    """Clipped logistic, matching Datageneration/DMT_KV."""

    if _is_tensor(x):
        x_clipped = torch.clamp(x, -60.0, 60.0)
        return 1.0 / (1.0 + torch.exp(-x_clipped))
    x_clipped = min(max(float(x), -60.0), 60.0)
    return 1.0 / (1.0 + math.exp(-x_clipped))


def f_ts_from_distance(
    s: Any,
    *,
    Estar: float,
    R: float,
    A: float,
    a0: float,
    beta: float,
) -> Any:
    """Exact DMT_KV force law used by the AFM04 shared truth layer."""

    g = sigmoid(beta * (s - a0))

    if _is_tensor(s):
        denom = g * (s - a0) + a0
        denom = torch.clamp(denom, min=1.0e-15)
        inv_denom = denom.reciprocal()
        adhesion = -(A * R / 6.0) * inv_denom.pow(2)
        indentation = torch.clamp(a0 - s, min=0.0)
        f_hertz = (4.0 / 3.0) * Estar * math.sqrt(R) * indentation.pow(1.5)
        return adhesion + (1.0 - g) * f_hertz

    denom = g * (s - a0) + a0
    denom = max(denom, 1.0e-15)
    inv_denom = 1.0 / denom
    adhesion = -(A * R / 6.0) * (inv_denom**2)
    indentation = max(a0 - s, 0.0)
    f_hertz = (4.0 / 3.0) * Estar * math.sqrt(R) * (indentation**1.5)
    return adhesion + (1.0 - g) * f_hertz


def f_ts_from_state(
    x1: Any,
    x2: Any,
    x3: Any,
    *,
    dist: float,
    Estar: float,
    eta_star: float,
    R: float,
    A: float,
    a0: float,
    beta: float,
    ks: float,
    cs: float,
) -> tuple[Any, Any, Any, Any]:
    """Return full DMT-KV force, explicit x3dot, indentation rate, and distance.

    The Kelvin-Voigt term depends on ``delta_dot = (a0 - s)_dot = x3dot - x2``.
    To keep the state-space equation explicit, solve x3dot from

        cs * x3dot = -F_ts - ks * x3

    after substituting the full ``F_ts`` definition.
    """

    s = dist + x1 - x3
    g = sigmoid(beta * (s - a0))

    if _is_tensor(s):
        denom = g * (s - a0) + a0
        denom = torch.clamp(denom, min=1.0e-15)
        inv_denom = denom.reciprocal()
        adhesion = -(A * R / 6.0) * inv_denom.pow(2)
        delta = torch.clamp(a0 - s, min=0.0)
        f_hertz = (4.0 / 3.0) * Estar * math.sqrt(R) * delta.pow(1.5)
        kv_coeff = eta_star * math.sqrt(R) * torch.sqrt(delta)
        f_static = adhesion + (1.0 - g) * f_hertz
        x3dot = (-f_static + kv_coeff * x2 - ks * x3) / torch.clamp(cs + kv_coeff, min=1.0e-15)
        zeros = torch.zeros_like(x3dot)
        delta_dot = torch.where(delta > 0.0, x3dot - x2, zeros)
        f_ts = f_static + kv_coeff * delta_dot
        return f_ts, x3dot, delta_dot, s

    denom = g * (s - a0) + a0
    denom = max(denom, 1.0e-15)
    inv_denom = 1.0 / denom
    adhesion = -(A * R / 6.0) * (inv_denom**2)
    delta = max(a0 - s, 0.0)
    f_hertz = (4.0 / 3.0) * Estar * math.sqrt(R) * (delta**1.5)
    kv_coeff = eta_star * math.sqrt(R) * math.sqrt(delta)
    f_static = adhesion + (1.0 - g) * f_hertz
    x3dot = (-f_static + kv_coeff * x2 - ks * x3) / max(cs + kv_coeff, 1.0e-15)
    delta_dot = (x3dot - x2) if delta > 0.0 else 0.0
    f_ts = f_static + kv_coeff * delta_dot
    return f_ts, x3dot, delta_dot, s


def ground_truth_rhs(t: Any, u: Any, p: Sequence[float] = original_parameters) -> Any:
    """Return AFM04 DMT-KV RHS as a 3-vector."""

    k, wd, m, c, Fd, R, dist, Estar, eta_star, A, a0, beta, ks, cs = p
    x1, x2, x3 = _split_state(u)

    f_ts, x3dot, _, _ = f_ts_from_state(
        x1,
        x2,
        x3,
        dist=dist,
        Estar=Estar,
        eta_star=eta_star,
        R=R,
        A=A,
        a0=a0,
        beta=beta,
        ks=ks,
        cs=cs,
    )

    x1dot = x2
    x2dot = (Fd * torch.cos(wd * t) - k * x1 - c * x2 + f_ts) / m if _is_tensor(x1) else (
        (Fd * math.cos(wd * t) - k * x1 - c * x2 + f_ts) / m
    )
    return _pack_state(x1dot, x2dot, x3dot, u)


def ground_truth_function(du: Any, u: Any, p: Sequence[float], t: Any) -> Any:
    """In-place flavored wrapper mirroring the Julia API."""

    rhs = ground_truth_rhs(t, u, p)
    if _is_tensor(du):
        du[...] = rhs
    else:
        du[0], du[1], du[2] = rhs[0], rhs[1], rhs[2]
    return du


def get_uode_model_function(
    appr_neural_network: Callable[..., Any],
    state: Any,
    original_parameters_opt: Sequence[float],
) -> Callable[[Any, Any, Any], Any]:
    """Residual-on-x2dot UODE builder with DMT_KV baseline physics."""

    def rhs(t: Any, u: Any, p: Any) -> Any:
        ode_par = _mul_parameters(_field_or(p, "ode_par"), original_parameters_opt)
        k, wd, m, c, Fd, R, dist, Estar, eta_star, A, a0, beta, ks, cs = ode_par

        x1, x2, x3 = _split_state(u)
        f_ts, x3dot, _, _ = f_ts_from_state(
            x1,
            x2,
            x3,
            dist=dist,
            Estar=Estar,
            eta_star=eta_star,
            R=R,
            A=A,
            a0=a0,
            beta=beta,
            ks=ks,
            cs=cs,
        )

        x1dot = x2
        x2dot = (Fd * torch.cos(wd * t) - k * x1 - c * x2 + f_ts) / m if _is_tensor(x1) else (
            (Fd * math.cos(wd * t) - k * x1 - c * x2 + f_ts) / m
        )

        out = appr_neural_network(u, _field_or(p, "p_net"), state)
        x2dot = x2dot + _network_scalar(out)
        return _pack_state(x1dot, x2dot, x3dot, u)

    return rhs


def get_uode_model_function_hertz_nn(
    appr_neural_network: Callable[..., Any],
    state: Any,
) -> Callable[[Any, Any, Any], Any]:
    """Direct NN-contact replacement builder without external gate/gain."""

    def rhs(t: Any, u: Any, p: Any) -> Any:
        ode_par = _field_or(p, "ode_par")
        k, wd, m, c, Fd, R, dist, Estar, eta_star, A, a0, beta, ks, cs = ode_par
        _ = eta_star

        x1, x2, x3 = _split_state(u)
        nn_in = u if _is_tensor(u) else [x1, x2, x3]
        f_contact = _network_scalar(appr_neural_network(nn_in, _field_or(p, "p_net"), state))

        x1dot = x2
        x2dot = (Fd * torch.cos(wd * t) - k * x1 - c * x2 + f_contact) / m if _is_tensor(x1) else (
            (Fd * math.cos(wd * t) - k * x1 - c * x2 + f_contact) / m
        )
        x3dot = (-f_contact - ks * x3) / cs
        return _pack_state(x1dot, x2dot, x3dot, u)

    return rhs


__all__ = [
    "sigmoid",
    "f_ts_from_distance",
    "f_ts_from_state",
    "ground_truth_rhs",
    "ground_truth_function",
    "get_uode_model_function",
    "get_uode_model_function_hertz_nn",
]
