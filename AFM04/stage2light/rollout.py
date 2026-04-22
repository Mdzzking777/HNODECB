"""Differentiable rollout for AFM04 stage2light with joint mech learning."""

from __future__ import annotations

import math

import torch
from torch import nn
from torchdiffeq import odeint

from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_functions import f_ts_from_state


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
    ) -> None:
        super().__init__()
        ks_lo, ks_hi = float(ks_bounds[0]), float(ks_bounds[1])
        cs_lo, cs_hi = float(cs_bounds[0]), float(cs_bounds[1])
        if not (ks_hi > ks_lo and cs_hi > cs_lo):
            raise ValueError("mech bounds must satisfy hi > lo")

        self.register_buffer("ks_lo", torch.as_tensor(ks_lo, dtype=dtype, device=device))
        self.register_buffer("ks_hi", torch.as_tensor(ks_hi, dtype=dtype, device=device))
        self.register_buffer("cs_lo", torch.as_tensor(cs_lo, dtype=dtype, device=device))
        self.register_buffer("cs_hi", torch.as_tensor(cs_hi, dtype=dtype, device=device))

        ks_frac = (float(ks_init) - ks_lo) / (ks_hi - ks_lo)
        cs_frac = (float(cs_init) - cs_lo) / (cs_hi - cs_lo)
        self.raw_ks = nn.Parameter(torch.as_tensor(_logit(ks_frac), dtype=dtype, device=device))
        self.raw_cs = nn.Parameter(torch.as_tensor(_logit(cs_frac), dtype=dtype, device=device))

    @staticmethod
    def _bounded(raw: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return lo + (hi - lo) * sigmoid_torch(raw)

    def ks(self) -> torch.Tensor:
        return self._bounded(self.raw_ks, self.ks_lo, self.ks_hi)

    def cs(self) -> torch.Tensor:
        return self._bounded(self.raw_cs, self.cs_lo, self.cs_hi)

    def forward(self) -> torch.Tensor:
        return torch.stack((self.ks(), self.cs()))


class JointAFMRHS(nn.Module):
    def __init__(
        self,
        force_module: nn.Module,
        known_pars: tuple[float, ...],
        mech_module: LearnableMechModule,
        *,
        x1_abs_guard: float | None = None,
        x2_abs_guard: float | None = None,
    ) -> None:
        super().__init__()
        self.force_module = force_module
        self.mech_module = mech_module
        self.k, self.wd, self.m, self.c, self.Fd, self.R, self.dist, self.Estar, self.A, self.a0, self.beta = known_pars
        self.x1_abs_guard = None if x1_abs_guard is None else float(x1_abs_guard)
        self.x2_abs_guard = None if x2_abs_guard is None else float(x2_abs_guard)

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
        du2 = (self.Fd * torch.cos(self.wd * t) - self.k * u[0] - self.c * u[1] + f_contact) / self.m
        du3 = (-f_contact - ks * u[2]) / cs
        return torch.stack((du1, du2, du3))


def rollout_single_shooting_torch(
    force_module: nn.Module,
    known_pars: tuple[float, ...],
    mech_module: LearnableMechModule,
    u0: torch.Tensor,
    times: torch.Tensor,
    *,
    method: str = "dopri5",
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
    x1_abs_guard: float | None = None,
    x2_abs_guard: float | None = None,
) -> torch.Tensor:
    rhs = JointAFMRHS(
        force_module,
        known_pars,
        mech_module,
        x1_abs_guard=x1_abs_guard,
        x2_abs_guard=x2_abs_guard,
    )
    traj = odeint(rhs, u0, times, method=method, rtol=rtol, atol=atol)
    return traj.transpose(0, 1)


def x2dot_rhs_torch(
    traj: torch.Tensor,
    times: torch.Tensor,
    force_module: nn.Module,
    known_pars: tuple[float, ...],
) -> torch.Tensor:
    k, wd, m, c, Fd, _, _, _, _, _, _ = known_pars
    states = traj.transpose(0, 1)
    f_contact = force_module(states)
    return (Fd * torch.cos(wd * times) - k * traj[0, :] - c * traj[1, :] + f_contact) / m


def fts_truth_from_states_torch(
    states: torch.Tensor,
    known_pars: tuple[float, ...],
    *,
    eta_star: float,
    mech_true,
) -> torch.Tensor:
    _, _, _, _, _, R, dist, Estar, A, a0, beta = known_pars
    mech_t = torch.as_tensor(mech_true, dtype=states.dtype, device=states.device).reshape(-1)
    if mech_t.numel() < 2:
        raise ValueError("mech_true must contain ks and cs")
    f_ts, _, _, _ = f_ts_from_state(
        states[:, 0],
        states[:, 1],
        states[:, 2],
        dist=dist,
        Estar=Estar,
        eta_star=float(eta_star),
        R=R,
        A=A,
        a0=a0,
        beta=beta,
        ks=mech_t[0],
        cs=mech_t[1],
    )
    return f_ts.reshape(-1)


__all__ = [
    "JointAFMRHS",
    "LearnableMechModule",
    "StateGuardTriggered",
    "f_ts_from_distance_torch",
    "fts_truth_from_states_torch",
    "rollout_single_shooting_torch",
    "x2dot_rhs_torch",
]
