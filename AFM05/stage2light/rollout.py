"""Differentiable rollout for AFM05 stage2light with joint mech learning."""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint


class StateGuardTriggered(RuntimeError):
    pass


def sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def f_ts_from_distance_torch(
    s: torch.Tensor,
    *,
    Estar: float,
    R: float,
    A: float,
    a0: float,
    beta: float,
) -> torch.Tensor:
    s = torch.as_tensor(s)
    g = sigmoid_torch(beta * (s - a0))
    denom = torch.square(g * (s - a0) + a0)
    adh = -(A * R) / (6.0 * denom)
    hertz = (1.0 - g) * (4.0 / 3.0) * Estar * (R ** 0.5) * torch.pow(torch.clamp(a0 - s, min=0.0), 1.5)
    return adh + hertz


def _logit(p: float) -> float:
    p = min(max(float(p), 1.0e-9), 1.0 - 1.0e-9)
    return math.log(p / (1.0 - p))


def _positive_geometric_midpoint(lo: float, hi: float) -> float:
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("log-relative mech parameterization requires positive reference bounds")
    return math.sqrt(float(lo) * float(hi))


def _mapping_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _afm05_known_fields(known_pars: Mapping[str, Any] | tuple[Any, ...]) -> dict[str, Any]:
    if isinstance(known_pars, Mapping):
        return {
            "k_eff": float(_mapping_get(known_pars, "k_eff", "k")),
            "m_eff": float(_mapping_get(known_pars, "m_eff", "m")),
            "c_eff": float(_mapping_get(known_pars, "c_eff", "c")),
            "actuation_times": _mapping_get(known_pars, "actuation_times", "F_actuation_times", "times"),
            "F_actuation": _mapping_get(known_pars, "F_actuation", "F_actuation_values", "actuation_values"),
            "Z": float(_mapping_get(known_pars, "Z", "dist", default=float("nan"))),
            "a0": float(_mapping_get(known_pars, "a0", default=float("nan"))),
        }
    if len(known_pars) >= 7 and not isinstance(known_pars[3], (int, float)):
        return {
            "k_eff": float(known_pars[0]),
            "m_eff": float(known_pars[1]),
            "c_eff": float(known_pars[2]),
            "actuation_times": known_pars[3],
            "F_actuation": known_pars[4],
            "Z": float(known_pars[5]),
            "a0": float(known_pars[6]),
        }
    raise ValueError(
        "AFM05 stage2light known_pars must provide "
        "k_eff, m_eff, c_eff, actuation_times, F_actuation, Z, and a0"
    )


def _actuation_values_torch(
    known_pars: Mapping[str, Any] | tuple[Any, ...],
    query_times: torch.Tensor,
) -> torch.Tensor:
    fields = _afm05_known_fields(known_pars)
    act_t = torch.as_tensor(fields["actuation_times"], dtype=query_times.dtype, device=query_times.device).reshape(-1)
    act_v = torch.as_tensor(fields["F_actuation"], dtype=query_times.dtype, device=query_times.device).reshape(-1)
    if act_t.numel() == 0 or act_v.numel() == 0:
        raise ValueError("AFM05 F_actuation(t) requires nonempty time and force arrays")
    if act_t.numel() != act_v.numel():
        raise ValueError(f"F_actuation time/value length mismatch: {act_t.numel()} vs {act_v.numel()}")
    if act_t.numel() == 1:
        return torch.full_like(query_times, act_v[0])
    qt = query_times.reshape(-1)
    idx = torch.searchsorted(act_t, qt).clamp(1, act_t.numel() - 1)
    t0 = act_t[idx - 1]
    t1 = act_t[idx]
    v0 = act_v[idx - 1]
    v1 = act_v[idx]
    w = (qt - t0) / torch.clamp(t1 - t0, min=torch.finfo(query_times.dtype).eps)
    interp = v0 + w * (v1 - v0)
    interp = torch.where(qt <= act_t[0], act_v[0], interp)
    interp = torch.where(qt >= act_t[-1], act_v[-1], interp)
    return interp.reshape_as(query_times)


class LearnableMechModule(nn.Module):
    def __init__(
        self,
        *,
        ks_init: float,
        cs_init: float,
        ks_bounds: tuple[float, float],
        cs_bounds: tuple[float, float],
        dtype: torch.dtype,
        device: str,
        parameterization: str = "direct_unbounded",
    ) -> None:
        super().__init__()
        ks_lo, ks_hi = float(ks_bounds[0]), float(ks_bounds[1])
        cs_lo, cs_hi = float(cs_bounds[0]), float(cs_bounds[1])
        if not (ks_hi > ks_lo and cs_hi > cs_lo):
            raise ValueError("mech bounds must satisfy hi > lo")
        mode = str(parameterization).strip().lower().replace("-", "_")
        aliases = {
            "direct": "direct_unbounded",
            "physical": "direct_unbounded",
            "unbounded": "direct_unbounded",
            "direct_unbounded": "direct_unbounded",
            "exp": "log_relative",
            "log": "log_relative",
            "relative": "log_relative",
            "log_relative": "log_relative",
            "log_relative_bounded": "log_relative",
            "sigmoid": "sigmoid_bounded",
            "bounded": "sigmoid_bounded",
            "sigmoid_bound": "sigmoid_bounded",
            "sigmoid_bounds": "sigmoid_bounded",
            "sigmoid_bounded": "sigmoid_bounded",
            "legacy": "sigmoid_bounded",
            "legacy_sigmoid": "sigmoid_bounded",
        }
        if mode not in aliases:
            raise ValueError(f"unsupported mech parameterization: {parameterization!r}")
        self.parameterization = aliases[mode]

        self.register_buffer("ks_lo", torch.as_tensor(ks_lo, dtype=dtype, device=device))
        self.register_buffer("ks_hi", torch.as_tensor(ks_hi, dtype=dtype, device=device))
        self.register_buffer("cs_lo", torch.as_tensor(cs_lo, dtype=dtype, device=device))
        self.register_buffer("cs_hi", torch.as_tensor(cs_hi, dtype=dtype, device=device))
        ks_ref = _positive_geometric_midpoint(ks_lo, ks_hi)
        cs_ref = _positive_geometric_midpoint(cs_lo, cs_hi)
        self.register_buffer("ks_ref", torch.as_tensor(ks_ref, dtype=dtype, device=device))
        self.register_buffer("cs_ref", torch.as_tensor(cs_ref, dtype=dtype, device=device))

        if self.parameterization == "sigmoid_bounded":
            ks_frac = (float(ks_init) - ks_lo) / (ks_hi - ks_lo)
            cs_frac = (float(cs_init) - cs_lo) / (cs_hi - cs_lo)
            raw_ks_init = _logit(ks_frac)
            raw_cs_init = _logit(cs_frac)
        elif self.parameterization == "log_relative":
            if float(ks_init) <= 0.0 or float(cs_init) <= 0.0:
                raise ValueError("log-relative mech parameterization requires positive initial ks/cs")
            raw_ks_init = math.log(float(ks_init) / ks_ref)
            raw_cs_init = math.log(float(cs_init) / cs_ref)
        else:
            raw_ks_init = float(ks_init)
            raw_cs_init = float(cs_init)
        self.raw_ks = nn.Parameter(torch.as_tensor(raw_ks_init, dtype=dtype, device=device))
        self.raw_cs = nn.Parameter(torch.as_tensor(raw_cs_init, dtype=dtype, device=device))

    @staticmethod
    def _bounded(raw: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return lo + (hi - lo) * sigmoid_torch(raw)

    @staticmethod
    def _log_relative(raw: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return ref * torch.exp(raw)

    def ks(self) -> torch.Tensor:
        if self.parameterization == "direct_unbounded":
            return self.raw_ks.reshape(())
        if self.parameterization == "log_relative":
            return self._log_relative(self.raw_ks, self.ks_ref)
        return self._bounded(self.raw_ks, self.ks_lo, self.ks_hi)

    def cs(self) -> torch.Tensor:
        if self.parameterization == "direct_unbounded":
            return self.raw_cs.reshape(())
        if self.parameterization == "log_relative":
            return self._log_relative(self.raw_cs, self.cs_ref)
        return self._bounded(self.raw_cs, self.cs_lo, self.cs_hi)

    def forward(self) -> torch.Tensor:
        return torch.stack((self.ks(), self.cs()))


class JointAFMRHS(nn.Module):
    def __init__(
        self,
        force_module: nn.Module,
        known_pars: Mapping[str, Any] | tuple[Any, ...],
        mech_module: LearnableMechModule,
        *,
        x1_abs_guard: float | None = None,
        x2_abs_guard: float | None = None,
    ) -> None:
        super().__init__()
        self.force_module = force_module
        self.mech_module = mech_module
        fields = _afm05_known_fields(known_pars)
        self.k_eff = float(fields["k_eff"])
        self.m_eff = float(fields["m_eff"])
        self.c_eff = float(fields["c_eff"])
        self.Z = float(fields["Z"])
        self.a0 = float(fields["a0"])
        self.register_buffer("actuation_times", torch.as_tensor(fields["actuation_times"], dtype=torch.float64).reshape(-1))
        self.register_buffer("actuation_values", torch.as_tensor(fields["F_actuation"], dtype=torch.float64).reshape(-1))
        self.x1_abs_guard = None if x1_abs_guard is None else float(x1_abs_guard)
        self.x2_abs_guard = None if x2_abs_guard is None else float(x2_abs_guard)

    def _actuation_at(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        act_t = self.actuation_times.to(dtype=like.dtype, device=like.device)
        act_v = self.actuation_values.to(dtype=like.dtype, device=like.device)
        if act_t.numel() == 1:
            return act_v[0]
        tq = t.to(dtype=like.dtype, device=like.device).reshape(())
        idx = torch.searchsorted(act_t, tq.reshape(1)).clamp(1, act_t.numel() - 1)[0]
        t0 = act_t[idx - 1]
        t1 = act_t[idx]
        v0 = act_v[idx - 1]
        v1 = act_v[idx]
        w = (tq - t0) / torch.clamp(t1 - t0, min=torch.finfo(like.dtype).eps)
        out = v0 + w * (v1 - v0)
        out = torch.where(tq <= act_t[0], act_v[0], out)
        out = torch.where(tq >= act_t[-1], act_v[-1], out)
        return out

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if u.ndim != 1 or u.numel() != 3:
            raise ValueError("JointAFMRHS expects a 3-vector state")
        x1 = float(torch.abs(u[0]).detach())
        x2 = float(torch.abs(u[1]).detach())
        if self.x1_abs_guard is not None and x1 > self.x1_abs_guard:
            raise StateGuardTriggered(
                f"state_guard_x1 | t={float(t.detach()):.6e} | abs_x1={x1:.6e} | guard={self.x1_abs_guard:.6e}"
            )
        if self.x2_abs_guard is not None and x2 > self.x2_abs_guard:
            raise StateGuardTriggered(
                f"state_guard_x2 | t={float(t.detach()):.6e} | abs_x2={x2:.6e} | guard={self.x2_abs_guard:.6e}"
            )
        mech = self.mech_module()
        ks = mech[0]
        cs = mech[1]
        f_contact = self.force_module(u.unsqueeze(0))[0]
        du1 = u[1]
        f_act = self._actuation_at(t, u)
        du2 = (f_act - self.k_eff * u[0] - self.c_eff * u[1] + f_contact) / self.m_eff
        du3 = (-f_contact - ks * u[2]) / cs
        return torch.stack((du1, du2, du3))


def rollout_single_shooting_torch(
    force_module: nn.Module,
    known_pars: Mapping[str, Any] | tuple[Any, ...],
    mech_module: LearnableMechModule,
    u0: torch.Tensor,
    times: torch.Tensor,
    *,
    method: str = "dopri5",
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
    x1_abs_guard: float | None = None,
    x2_abs_guard: float | None = None,
    use_adjoint: bool | None = None,
) -> torch.Tensor:
    rhs = JointAFMRHS(
        force_module,
        known_pars,
        mech_module,
        x1_abs_guard=x1_abs_guard,
        x2_abs_guard=x2_abs_guard,
    )
    if use_adjoint is None:
        raw = os.environ.get("HNODECB_AFM05_STAGE2LIGHT_USE_ADJOINT", "").strip()
        use_adjoint = raw not in ("", "0", "false", "False", "no", "NO")
    if use_adjoint:
        adjoint_params = tuple(p for p in rhs.parameters() if p.requires_grad)
        traj = odeint_adjoint(
            rhs,
            u0,
            times,
            method=method,
            rtol=rtol,
            atol=atol,
            adjoint_method=method,
            adjoint_rtol=rtol,
            adjoint_atol=atol,
            adjoint_params=adjoint_params,
        )
    else:
        traj = odeint(rhs, u0, times, method=method, rtol=rtol, atol=atol)
    return traj.transpose(0, 1)


def x2dot_rhs_torch(
    traj: torch.Tensor,
    times: torch.Tensor,
    force_module: nn.Module,
    known_pars: Mapping[str, Any] | tuple[Any, ...],
) -> torch.Tensor:
    fields = _afm05_known_fields(known_pars)
    k_eff = float(fields["k_eff"])
    m_eff = float(fields["m_eff"])
    c_eff = float(fields["c_eff"])
    states = traj.transpose(0, 1)
    f_contact = force_module(states)
    f_act = _actuation_values_torch(known_pars, times)
    return (f_act - k_eff * traj[0, :] - c_eff * traj[1, :] + f_contact) / m_eff


__all__ = [
    "JointAFMRHS",
    "LearnableMechModule",
    "StateGuardTriggered",
    "f_ts_from_distance_torch",
    "rollout_single_shooting_torch",
    "x2dot_rhs_torch",
]
