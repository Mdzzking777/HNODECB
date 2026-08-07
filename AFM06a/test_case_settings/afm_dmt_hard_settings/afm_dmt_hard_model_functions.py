"""AFM06a mass-scaled hard-sample force law and two-state RHS."""

from __future__ import annotations

import math
from typing import Any, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - NumPy generation needs no torch
    torch = None

def _is_tensor(x: Any) -> bool:
    return torch is not None and torch.is_tensor(x)


def sigmoid(x: Any) -> Any:
    """Clipped logistic contact transition."""

    if _is_tensor(x):
        x_clipped = torch.clamp(x, -60.0, 60.0)
        return 1.0 / (1.0 + torch.exp(-x_clipped))
    x_clipped = min(max(float(x), -60.0), 60.0)
    return 1.0 / (1.0 + math.exp(-x_clipped))


def effective_damping_from_distance(
    s: Any,
    *,
    d1: float,
    d2: float,
    a0: float,
    beta: float,
) -> Any:
    """Apply the AFM06a hard switch from non-contact to contact damping."""

    _ = beta  # Retained for compatibility with the saved parameter-vector layout.
    if _is_tensor(s):
        d1_tensor = torch.as_tensor(d1, dtype=s.dtype, device=s.device)
        d2_tensor = torch.as_tensor(d2, dtype=s.dtype, device=s.device)
        return torch.where(s <= a0, d2_tensor, d1_tensor)
    return float(d2) if float(s) <= float(a0) else float(d1)


def bar_f_ts_from_distance(
    s: Any,
    *,
    ca: float | None = None,
    ch: float | None = None,
    omega0: float | None = None,
    c1: float | None = None,
    c2: float | None = None,
    eta_star_length: float | None = None,
    a0: float,
    beta: float,
) -> Any:
    """Return the hard-switched mass-scaled interaction acceleration.

    Non-contact (s > a0):
        adhesion = C_A / s^2
        Hertz = 0

    Contact (s <= a0):
        adhesion = C_A / a0^2
        Hertz = C_H * (a0 - s)^(3/2)
    """

    _ = beta  # Retained for compatibility with the saved parameter-vector layout.
    if ca is None or ch is None:
        if omega0 is None or c1 is None or c2 is None or eta_star_length is None:
            raise ValueError("either ca/ch or omega0/c1/c2/eta_star_length must be supplied")
        adhesion_coefficient = float(c1) * (float(omega0) ** 2) * (float(eta_star_length) ** 3)
        hertz_coefficient = float(c2) * (float(omega0) ** 2) / math.sqrt(float(eta_star_length))
    else:
        adhesion_coefficient = float(ca)
        hertz_coefficient = float(ch)
    if _is_tensor(s):
        a0_tensor = torch.as_tensor(a0, dtype=s.dtype, device=s.device)
        contact = s <= a0_tensor
        denom = torch.where(contact, a0_tensor, s)
        denom = torch.clamp(denom, min=1.0e-15)
        inv_denom = denom.reciprocal()
        adhesion = adhesion_coefficient * inv_denom.pow(2)
        indentation = torch.clamp(a0_tensor - s, min=0.0)
        hertz = hertz_coefficient * indentation.pow(1.5)
        return adhesion + torch.where(contact, hertz, torch.zeros_like(hertz))

    contact = float(s) <= float(a0)
    denom = max(float(a0) if contact else float(s), 1.0e-15)
    inv_denom = 1.0 / denom
    adhesion = adhesion_coefficient * (inv_denom**2)
    if not contact:
        return adhesion
    indentation = max(float(a0) - float(s), 0.0)
    return adhesion + hertz_coefficient * (indentation**1.5)


def interaction_from_tip_state(
    x1: Any,
    *,
    dist: float,
    ca: float | None = None,
    ch: float | None = None,
    omega0: float | None = None,
    c1: float | None = None,
    c2: float | None = None,
    eta_star_length: float | None = None,
    a0: float,
    beta: float,
) -> tuple[Any, Any, Any]:
    """Return mass-scaled interaction, separation, and indentation."""

    s = dist + x1
    if _is_tensor(s):
        delta = torch.clamp(a0 - s, min=0.0)
    else:
        delta = max(a0 - s, 0.0)
    bar_f_ts = bar_f_ts_from_distance(
        s,
        ca=ca,
        ch=ch,
        omega0=omega0,
        c1=c1,
        c2=c2,
        eta_star_length=eta_star_length,
        a0=a0,
        beta=beta,
    )
    return bar_f_ts, s, delta


def _physical_rhs_from_vector(t: Any, u: Any, p: Sequence[float]) -> Any:
    k, omega0, mass, damping, fd, ca, ch, dist, a0, beta = p
    x1, x2 = u[0], u[1]
    bar_f_ts, _s, _ = interaction_from_tip_state(
        x1,
        dist=dist,
        ca=ca,
        ch=ch,
        a0=a0,
        beta=beta,
    )
    x1dot = x2
    if _is_tensor(x1):
        x2dot = (
            (fd * torch.sin(omega0 * t) - damping * x2 - k * x1) / mass
            + bar_f_ts
        )
        return torch.stack((x1dot, x2dot))
    x2dot = (
        (fd * math.sin(omega0 * t) - damping * x2 - k * x1) / mass
        + bar_f_ts
    )
    return [x1dot, x2dot]


def _legacy_rhs_from_vector(t: Any, u: Any, p: Sequence[float]) -> Any:
    omega0, c1, c2, b1, d1, d2, eta_star_length, y_bar, omega_bar, dist, a0, beta = p
    x1, x2 = u[0], u[1]
    bar_f_ts, s, _ = interaction_from_tip_state(
        x1,
        dist=dist,
        omega0=omega0,
        c1=c1,
        c2=c2,
        eta_star_length=eta_star_length,
        a0=a0,
        beta=beta,
    )
    d_eff = effective_damping_from_distance(s, d1=d1, d2=d2, a0=a0, beta=beta)
    x1dot = x2
    actuation_coefficient = (omega0**2) * eta_star_length * b1 * (omega_bar**2) * y_bar
    if _is_tensor(x1):
        x2dot = (
            actuation_coefficient * torch.sin(omega0 * t)
            - (omega0**2) * x1
            - d_eff * omega0 * x2
            + bar_f_ts
        )
        return torch.stack((x1dot, x2dot))
    x2dot = (
        actuation_coefficient * math.sin(omega0 * t)
        - (omega0**2) * x1
        - d_eff * omega0 * x2
        + bar_f_ts
    )
    return [x1dot, x2dot]


def ground_truth_rhs(t: Any, u: Any, p: Sequence[float]) -> Any:
    """Evaluate the two-state hard-sample cantilever RHS."""

    if len(p) == 10:
        return _physical_rhs_from_vector(t, u, p)
    if len(p) >= 12:
        return _legacy_rhs_from_vector(t, u, p)
    raise ValueError(f"unsupported AFM06a parameter vector length: {len(p)}")


def ground_truth_function(du: Any, u: Any, p: Sequence[float], t: Any) -> Any:
    """In-place-style wrapper matching the legacy model API."""

    rhs = ground_truth_rhs(t, u, p)
    if _is_tensor(du):
        du[...] = rhs
    else:
        du[0], du[1] = rhs[0], rhs[1]
    return du


__all__ = [
    "sigmoid",
    "effective_damping_from_distance",
    "bar_f_ts_from_distance",
    "interaction_from_tip_state",
    "ground_truth_rhs",
    "ground_truth_function",
]
