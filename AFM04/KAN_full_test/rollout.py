"""Differentiable rollout for the isolated KAN full functional test."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchdiffeq import odeint

from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_functions import f_ts_from_state


class StateGuardTriggered(RuntimeError):
    pass


@dataclass
class RolloutAux:
    q_pre: torch.Tensor | None
    accepted_history_count: int = 0


def _check_state_guard_at_t(
    *,
    state: torch.Tensor,
    t: torch.Tensor,
    x1_abs_guard: float | None,
    x2_abs_guard: float | None,
) -> None:
    x1 = float(torch.abs(state[0]).detach())
    x2 = float(torch.abs(state[1]).detach())
    t_val = float(t.detach())
    if x1_abs_guard is not None and x1 > float(x1_abs_guard):
        raise StateGuardTriggered(
            f"state_guard_x1 | t={t_val:.6e} | abs_x1={x1:.6e} | guard={float(x1_abs_guard):.6e}"
        )
    if x2_abs_guard is not None and x2 > float(x2_abs_guard):
        raise StateGuardTriggered(
            f"state_guard_x2 | t={t_val:.6e} | abs_x2={x2:.6e} | guard={float(x2_abs_guard):.6e}"
        )


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


class KnownAFMRHS(nn.Module):
    def __init__(
        self,
        force_module: nn.Module,
        known_pars: tuple[float, ...],
        mech_true: torch.Tensor,
        *,
        fixed_q: torch.Tensor | None = None,
        x1_abs_guard: float | None = None,
        x2_abs_guard: float | None = None,
        t_guard_min: float | None = None,
        t_guard_max: float | None = None,
    ) -> None:
        super().__init__()
        self.force_module = force_module
        self.k, self.wd, self.m, self.c, self.Fd, self.R, self.dist, self.Estar, self.A, self.a0, self.beta = known_pars
        self.register_buffer("mech_true", torch.as_tensor(mech_true, dtype=torch.float64))
        self.x1_abs_guard = None if x1_abs_guard is None else float(x1_abs_guard)
        self.x2_abs_guard = None if x2_abs_guard is None else float(x2_abs_guard)
        self.t_guard_min = None if t_guard_min is None else float(t_guard_min)
        self.t_guard_max = None if t_guard_max is None else float(t_guard_max)
        self.x3dot_enabled = bool(getattr(force_module, "x3dot_input_enabled", False))
        self.fixed_q = None if fixed_q is None else torch.as_tensor(fixed_q).reshape(())

    def _initial_q(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if hasattr(self.force_module, "x3dot_initial_q"):
            q0 = self.force_module.x3dot_initial_q()
            return q0.to(dtype=dtype, device=device).reshape(())
        return torch.zeros((), dtype=dtype, device=device)

    def _current_q(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.x3dot_enabled:
            return torch.zeros((), dtype=dtype, device=device)
        if self.fixed_q is None:
            return self._initial_q(dtype=dtype, device=device)
        return self.fixed_q.to(dtype=dtype, device=device).reshape(())

    def _force_input(self, u: torch.Tensor, q_pre: torch.Tensor | None = None) -> torch.Tensor:
        states = u.unsqueeze(0)
        if hasattr(self.force_module, "prepare_force_inputs"):
            return self.force_module.prepare_force_inputs(states, None if q_pre is None else q_pre.reshape(1))
        return states

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if u.ndim != 1 or u.numel() != 3:
            raise ValueError("KnownAFMRHS expects a 3-vector state")
        x1 = float(torch.abs(u[0]).detach())
        x2 = float(torch.abs(u[1]).detach())
        t_val = float(t.detach())
        guard_in_window = True
        if self.t_guard_min is not None:
            guard_in_window = guard_in_window and t_val >= self.t_guard_min - 1.0e-15
        if self.t_guard_max is not None:
            guard_in_window = guard_in_window and t_val <= self.t_guard_max + 1.0e-15
        if guard_in_window:
            if self.x1_abs_guard is not None and x1 > self.x1_abs_guard:
                raise StateGuardTriggered(
                    f"state_guard_x1 | t={t_val:.6e} | abs_x1={x1:.6e} | guard={self.x1_abs_guard:.6e}"
                )
            if self.x2_abs_guard is not None and x2 > self.x2_abs_guard:
                raise StateGuardTriggered(
                    f"state_guard_x2 | t={t_val:.6e} | abs_x2={x2:.6e} | guard={self.x2_abs_guard:.6e}"
                )
        ks = self.mech_true[0]
        cs = self.mech_true[1]
        q_pre = self._current_q(dtype=u.dtype, device=u.device) if self.x3dot_enabled else None
        f_contact = self.force_module(self._force_input(u, q_pre))[0]
        du1 = u[1]
        du2 = (self.Fd * torch.cos(self.wd * t) - self.k * u[0] - self.c * u[1] + f_contact) / self.m
        du3 = (-f_contact - ks * u[2]) / cs
        return torch.stack((du1, du2, du3))


def _x3dot_from_state_and_q(
    *,
    force_module: nn.Module,
    known_pars: tuple[float, ...],
    mech_true: torch.Tensor,
    state: torch.Tensor,
    q_pre: torch.Tensor,
) -> torch.Tensor:
    mech_t = torch.as_tensor(mech_true, dtype=state.dtype, device=state.device).reshape(-1)
    if mech_t.numel() < 2:
        raise ValueError("mech_true must contain ks and cs")
    ks = mech_t[0]
    cs = mech_t[1]
    if hasattr(force_module, "prepare_force_inputs"):
        inputs = force_module.prepare_force_inputs(state.reshape(1, -1), q_pre.reshape(1))
    else:
        inputs = state.reshape(1, -1)
    f_contact = force_module(inputs)[0]
    return ((-f_contact - ks * state[2]) / cs).reshape(())


def rollout_single_shooting_torch(
    force_module: nn.Module,
    known_pars: tuple[float, ...],
    mech_true: torch.Tensor,
    u0: torch.Tensor,
    times: torch.Tensor,
    *,
    method: str = "dopri5",
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
    x1_abs_guard: float | None = None,
    x2_abs_guard: float | None = None,
    ode_step_budget: int | None = None,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, RolloutAux]:
    x3dot_enabled = bool(getattr(force_module, "x3dot_input_enabled", False))
    if x3dot_enabled:
        if str(method).strip().lower() != "dopri5":
            raise ValueError("Plan Z T-level x3dot input currently requires torchdiffeq method='dopri5'")
        if times.ndim != 1:
            raise ValueError("rollout_single_shooting_torch expects 1-D times")
        if u0.ndim != 1 or u0.numel() != 3:
            raise ValueError("rollout_single_shooting_torch expects a 3-vector initial state")

        states: list[torch.Tensor] = [u0]
        q_values: list[torch.Tensor] = []
        if hasattr(force_module, "x3dot_initial_q"):
            q_current = force_module.x3dot_initial_q().to(dtype=u0.dtype, device=u0.device).reshape(())
        else:
            q_current = torch.zeros((), dtype=u0.dtype, device=u0.device)
        q_values.append(q_current)

        options_dict: dict[str, object] = {}
        if ode_step_budget is not None and int(ode_step_budget) > 0:
            options_dict["max_num_steps"] = int(ode_step_budget)
        options = options_dict or None

        u_current = u0
        for k in range(int(times.numel()) - 1):
            t_pair = times[k : k + 2]
            rhs = KnownAFMRHS(
                force_module,
                known_pars,
                mech_true,
                fixed_q=q_current,
                # In Plan Z T-level rollout, fail-fast state guards must be
                # evaluated on accepted physical sampling states, not on
                # dopri5 internal stage states.
                x1_abs_guard=None,
                x2_abs_guard=None,
                t_guard_min=float(torch.min(times).detach()),
                t_guard_max=float(torch.max(times).detach()),
            )
            seg = odeint(rhs, u_current, t_pair, method=method, rtol=rtol, atol=atol, options=options)
            u_next = seg[-1]
            _check_state_guard_at_t(
                state=u_next,
                t=t_pair[-1],
                x1_abs_guard=x1_abs_guard,
                x2_abs_guard=x2_abs_guard,
            )
            q_next = _x3dot_from_state_and_q(
                force_module=force_module,
                known_pars=known_pars,
                mech_true=mech_true,
                state=u_next,
                q_pre=q_current,
            )
            if bool(getattr(force_module, "x3dot_lag_detach", False)):
                q_next = q_next.detach()
            states.append(u_next)
            q_values.append(q_next)
            u_current = u_next
            q_current = q_next

        traj_out = torch.stack(states, dim=1)
        if not return_aux:
            return traj_out
        q_pre = torch.stack(q_values).to(dtype=times.dtype, device=times.device)
        return traj_out, RolloutAux(q_pre=q_pre, accepted_history_count=int(q_pre.numel()))

    rhs = KnownAFMRHS(
        force_module,
        known_pars,
        mech_true,
        x1_abs_guard=x1_abs_guard,
        x2_abs_guard=x2_abs_guard,
        t_guard_min=float(torch.min(times).detach()),
        t_guard_max=float(torch.max(times).detach()),
    )
    options_dict: dict[str, object] = {}
    if ode_step_budget is not None and int(ode_step_budget) > 0:
        options_dict["max_num_steps"] = int(ode_step_budget)
    options = options_dict or None
    traj = odeint(rhs, u0, times, method=method, rtol=rtol, atol=atol, options=options)
    traj_out = traj.transpose(0, 1)
    if not return_aux:
        return traj_out
    return traj_out, RolloutAux(q_pre=None, accepted_history_count=0)


def force_inputs_for_module(force_module: nn.Module, states: torch.Tensor, q_pre: torch.Tensor | None = None) -> torch.Tensor:
    if hasattr(force_module, "prepare_force_inputs"):
        return force_module.prepare_force_inputs(states, q_pre)
    return states


def x2dot_rhs_torch(
    traj: torch.Tensor,
    times: torch.Tensor,
    force_module: nn.Module,
    known_pars: tuple[float, ...],
    q_pre: torch.Tensor | None = None,
) -> torch.Tensor:
    k, wd, m, c, Fd, _, _, _, _, _, _ = known_pars
    states = traj.transpose(0, 1)
    f_contact = force_module(force_inputs_for_module(force_module, states, q_pre))
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
    "KnownAFMRHS",
    "RolloutAux",
    "StateGuardTriggered",
    "force_inputs_for_module",
    "f_ts_from_distance_torch",
    "fts_truth_from_states_torch",
    "rollout_single_shooting_torch",
    "x2dot_rhs_torch",
]
