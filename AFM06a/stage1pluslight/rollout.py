"""Differentiable two-state AFM06a rollout shared by all learning stages."""

from __future__ import annotations

import torch
from torch import nn
from torchdiffeq import odeint

from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (
    AFM06aHardSampleInputs,
)


class StateGuardTriggered(RuntimeError):
    pass


def _interpolate_observed_x1(
    query_time: torch.Tensor,
    observed_times: torch.Tensor,
    observed_x1: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate the fixed observed x1 trajectory at one RHS time."""

    query = torch.clamp(query_time, min=observed_times[0], max=observed_times[-1])
    right = torch.searchsorted(observed_times, query, right=False)
    right = torch.clamp(right, min=1, max=observed_times.numel() - 1)
    left = right - 1
    t_left = observed_times[left]
    t_right = observed_times[right]
    fraction = (query - t_left) / torch.clamp(
        t_right - t_left,
        min=torch.finfo(observed_times.dtype).eps,
    )
    return observed_x1[left] + fraction * (observed_x1[right] - observed_x1[left])


class AFM06aKnownRHS(nn.Module):
    def __init__(
        self,
        force_module: nn.Module,
        settings: AFM06aHardSampleInputs,
        *,
        observed_times: torch.Tensor | None = None,
        observed_x1: torch.Tensor | None = None,
        x1_abs_guard: float | None = None,
        x2_abs_guard: float | None = None,
        max_rhs_evaluations: int | None = None,
    ) -> None:
        super().__init__()
        settings.validate()
        self.force_module = force_module
        self.settings = settings
        self.x1_abs_guard = None if x1_abs_guard is None else float(x1_abs_guard)
        self.x2_abs_guard = None if x2_abs_guard is None else float(x2_abs_guard)
        self.max_rhs_evaluations = (
            None if max_rhs_evaluations is None else int(max_rhs_evaluations)
        )
        self._rhs_evaluations = 0
        if (observed_times is None) != (observed_x1 is None):
            raise ValueError("observed_times and observed_x1 must be supplied together")
        if observed_times is None:
            self.register_buffer("observed_times", None)
            self.register_buffer("observed_x1", None)
        else:
            times = torch.as_tensor(observed_times).reshape(-1)
            x1_values = torch.as_tensor(
                observed_x1,
                dtype=times.dtype,
                device=times.device,
            ).reshape(-1)
            if (
                times.numel() < 2
                or x1_values.shape != times.shape
                or bool(torch.any(times[1:] <= times[:-1]))
            ):
                raise ValueError(
                    "observed conditioning requires matching, strictly increasing "
                    "time/x1 vectors"
                )
            self.register_buffer("observed_times", times)
            self.register_buffer("observed_x1", x1_values)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self._rhs_evaluations += 1
        if (
            self.max_rhs_evaluations is not None
            and self._rhs_evaluations > self.max_rhs_evaluations
        ):
            raise StateGuardTriggered(
                "rhs_evaluation_budget_exceeded "
                f"calls={self._rhs_evaluations} "
                f"budget={self.max_rhs_evaluations} "
                f"t={float(t.detach()):.9e}"
            )
        if state.ndim != 1 or state.numel() != 2:
            raise ValueError(f"AFM06a RHS expects a two-vector state, got {state.shape}")
        abs_x1 = float(torch.abs(state[0]).detach())
        abs_x2 = float(torch.abs(state[1]).detach())
        if self.x1_abs_guard is not None and abs_x1 > self.x1_abs_guard:
            raise StateGuardTriggered(
                f"state_guard_x1 t={float(t.detach()):.9e} abs={abs_x1:.9e} guard={self.x1_abs_guard:.9e}"
            )
        if self.x2_abs_guard is not None and abs_x2 > self.x2_abs_guard:
            raise StateGuardTriggered(
                f"state_guard_x2 t={float(t.detach()):.9e} abs={abs_x2:.9e} guard={self.x2_abs_guard:.9e}"
            )

        k_n_m, omega0, mass_kg, c_n_s_m, fd_n, _ca, _ch, _dist, _a0, _beta = (
            self.settings.parameter_vector
        )
        x1, x2 = state[0], state[1]
        force_x1 = (
            x1
            if self.observed_times is None
            else _interpolate_observed_x1(t, self.observed_times, self.observed_x1)
        )
        force_state = torch.stack((force_x1, torch.zeros_like(force_x1))).reshape(1, 2)
        fts = self.force_module(force_state)[0]
        bar_fts = fts / float(mass_kg)
        actuation = float(fd_n) / float(mass_kg)
        return torch.stack(
            (
                x2,
                actuation * torch.sin(float(omega0) * t)
                - (float(k_n_m) / float(mass_kg)) * x1
                - (float(c_n_s_m) / float(mass_kg)) * x2
                + bar_fts,
            )
        )


def rollout_single_shooting_torch(
    force_module: nn.Module,
    settings: AFM06aHardSampleInputs,
    initial_state: torch.Tensor,
    times: torch.Tensor,
    *,
    method: str,
    rtol: float,
    atol: float,
    observed_x1: torch.Tensor | None = None,
    x1_abs_guard: float | None = None,
    x2_abs_guard: float | None = None,
    ode_step_budget: int | None = None,
    max_step: float | None = None,
    max_rhs_evaluations: int | None = None,
) -> torch.Tensor:
    if times.ndim != 1 or times.numel() < 2 or bool(torch.any(times[1:] <= times[:-1])):
        raise ValueError("times must be strictly increasing and contain at least two points")
    initial_state = torch.as_tensor(initial_state, dtype=times.dtype, device=times.device).reshape(-1)
    if initial_state.numel() != 2:
        raise ValueError("AFM06a initial state must contain x1 and x2")
    rhs = AFM06aKnownRHS(
        force_module,
        settings,
        observed_times=times if observed_x1 is not None else None,
        observed_x1=observed_x1,
        x1_abs_guard=x1_abs_guard,
        x2_abs_guard=x2_abs_guard,
        max_rhs_evaluations=max_rhs_evaluations,
    )
    options = None
    option_values: dict[str, float | int] = {}
    if ode_step_budget is not None and int(ode_step_budget) > 0:
        option_values["max_num_steps"] = int(ode_step_budget)
    if max_step is not None and float(max_step) > 0.0:
        option_values["max_step"] = float(max_step)
    if option_values:
        options = option_values
    trajectory = odeint(
        rhs,
        initial_state,
        times,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        options=options,
    ).transpose(0, 1)
    if trajectory.shape != (2, times.numel()) or not bool(torch.all(torch.isfinite(trajectory))):
        raise RuntimeError(f"AFM06a rollout returned invalid trajectory: {tuple(trajectory.shape)}")
    return trajectory


def x2dot_rhs_torch(
    trajectory: torch.Tensor,
    times: torch.Tensor,
    force_module: nn.Module,
    settings: AFM06aHardSampleInputs,
    *,
    observed_x1: torch.Tensor | None = None,
) -> torch.Tensor:
    if trajectory.shape != (2, times.numel()):
        raise ValueError("trajectory must have shape (2, len(times))")
    k_n_m, omega0, mass_kg, c_n_s_m, fd_n, _ca, _ch, _dist, _a0, _beta = (
        settings.parameter_vector
    )
    x1 = trajectory[0]
    x2 = trajectory[1]
    force_x1 = (
        x1
        if observed_x1 is None
        else torch.as_tensor(
            observed_x1,
            dtype=trajectory.dtype,
            device=trajectory.device,
        ).reshape(-1)
    )
    if force_x1.shape != x1.shape:
        raise ValueError("observed_x1 must match the trajectory time dimension")
    force_states = torch.stack((force_x1, torch.zeros_like(force_x1)), dim=1)
    fts = force_module(force_states)
    bar_fts = fts / float(mass_kg)
    actuation = float(fd_n) / float(mass_kg)
    return (
        actuation * torch.sin(float(omega0) * times)
        - (float(k_n_m) / float(mass_kg)) * x1
        - (float(c_n_s_m) / float(mass_kg)) * x2
        + bar_fts
    )


__all__ = [
    "AFM06aKnownRHS",
    "StateGuardTriggered",
    "_interpolate_observed_x1",
    "rollout_single_shooting_torch",
    "x2dot_rhs_torch",
]
