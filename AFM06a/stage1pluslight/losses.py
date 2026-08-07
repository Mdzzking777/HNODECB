"""Shared AFM06a noiseless governing-equation residual loss."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (
    AFM06aHardSampleInputs,
)


SCALE_EPS = 1.0e-30
SUPPORTED_POLICY = "force_scaled_dynamics_residual_true_x1_residual_smooth"


@dataclass(frozen=True)
class LossParts:
    dynamics_residual: float
    smoothness: float
    smoothness_unweighted: float
    residual_rmse: float
    residual_max_abs: float
    residual_scale: float
    x1_rec: float
    x2_rec: float


def _relative_rmse_pct(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    mse = torch.mean(torch.square(prediction - truth))
    truth_rms = torch.sqrt(torch.mean(torch.square(truth)))
    return 100.0 * torch.sqrt(mse) / torch.clamp(truth_rms, min=SCALE_EPS)


def dynamics_residual_torch(
    *,
    predicted_fts: torch.Tensor,
    true_states: torch.Tensor,
    true_x2dot: torch.Tensor,
    times: torch.Tensor,
    settings: AFM06aHardSampleInputs | None = None,
) -> torch.Tensor:
    """Return the noiseless cantilever residual evaluated on observed states.

    The KAN prediction is the only learned term. Every state, derivative,
    contact decision, damping coefficient, and actuation value comes from the
    observed trajectory:

        r_dyn = x2dot_true - f2(x1_true, x2_true, Fts_pred(x1_true)/m, t).
    """

    settings = AFM06aHardSampleInputs() if settings is None else settings
    settings.validate()
    if true_states.ndim != 2 or true_states.shape[0] != 2:
        raise ValueError(f"true_states must have shape (2, N), got {true_states.shape}")
    sample_count = int(true_states.shape[1])
    predicted_fts = predicted_fts.reshape(-1)
    true_x2dot = true_x2dot.reshape(-1)
    times = times.reshape(-1)
    if (
        predicted_fts.numel() != sample_count
        or true_x2dot.numel() != sample_count
        or times.numel() != sample_count
    ):
        raise ValueError("force, x2dot, and time vectors must match true_states")

    k_n_m, omega0, mass_kg, c_n_s_m, fd_n, _ca, _ch, _dist, _a0, _beta = (
        settings.parameter_vector
    )
    true_x1 = true_states[0]
    true_x2 = true_states[1]
    actuation = float(fd_n) / float(mass_kg)
    predicted_rhs_x2dot = (
        actuation * torch.sin(float(omega0) * times)
        - (float(k_n_m) / float(mass_kg)) * true_x1
        - (float(c_n_s_m) / float(mass_kg)) * true_x2
        + predicted_fts / float(mass_kg)
    )
    return true_x2dot - predicted_rhs_x2dot


def residual_smoothness_torch(
    *,
    normalized_residual: torch.Tensor,
    times: torch.Tensor,
    indices: torch.Tensor,
    contact_indicator: torch.Tensor,
) -> torch.Tensor:
    """Penalize within-regime residual curvature on a nonuniform time grid."""

    residual = normalized_residual.reshape(-1)
    time_vector = times.reshape(-1)
    idx = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=residual.device,
    ).reshape(-1)
    contact = contact_indicator.to(
        device=residual.device,
        dtype=torch.bool,
    ).reshape(-1)
    if residual.numel() != idx.numel():
        raise ValueError("normalized_residual must contain one value per selected index")
    if time_vector.numel() != contact.numel():
        raise ValueError("times and contact_indicator must have the same length")
    if idx.numel() == 0:
        raise ValueError("residual smoothness indices cannot be empty")

    transition = contact[1:] != contact[:-1]
    segment_ids = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=residual.device),
            torch.cumsum(transition.to(torch.long), dim=0),
        )
    )
    selected_times = time_vector[idx]
    selected_segments = segment_ids[idx]
    order = torch.argsort(selected_times)
    selected_times = selected_times[order]
    selected_residual = residual[order]
    selected_segments = selected_segments[order]

    second_differences: list[torch.Tensor] = []
    for segment_id in torch.unique_consecutive(selected_segments):
        in_segment = selected_segments == segment_id
        segment_residual = selected_residual[in_segment]
        segment_times = selected_times[in_segment]
        if segment_residual.numel() < 3:
            continue
        h_left = segment_times[1:-1] - segment_times[:-2]
        h_right = segment_times[2:] - segment_times[1:-1]
        if torch.any(h_left <= 0.0) or torch.any(h_right <= 0.0):
            raise ValueError("residual smoothness requires strictly increasing times")
        slope_left = (segment_residual[1:-1] - segment_residual[:-2]) / h_left
        slope_right = (segment_residual[2:] - segment_residual[1:-1]) / h_right
        second_derivative = 2.0 * (slope_right - slope_left) / (
            h_left + h_right
        )
        local_spacing = 0.5 * (h_left + h_right)
        second_differences.append(
            second_derivative * torch.square(local_spacing)
        )

    if not second_differences:
        return torch.sum(residual) * 0.0
    return torch.mean(torch.square(torch.cat(second_differences)))


def evaluate_dynamics_residual_loss(
    *,
    predicted_fts: torch.Tensor,
    true_states: torch.Tensor,
    true_x2dot: torch.Tensor,
    times: torch.Tensor,
    indices: torch.Tensor,
    residual_scale: float | torch.Tensor,
    predicted_states: torch.Tensor | None = None,
    settings: AFM06aHardSampleInputs | None = None,
    smoothness_weight: float = 0.0,
) -> tuple[torch.Tensor, LossParts]:
    total, parts, _ = evaluate_dynamics_residual_loss_with_terms(
        predicted_fts=predicted_fts,
        true_states=true_states,
        true_x2dot=true_x2dot,
        times=times,
        indices=indices,
        residual_scale=residual_scale,
        predicted_states=predicted_states,
        settings=settings,
        smoothness_weight=smoothness_weight,
    )
    return total, parts


def evaluate_dynamics_residual_loss_with_terms(
    *,
    predicted_fts: torch.Tensor,
    true_states: torch.Tensor,
    true_x2dot: torch.Tensor,
    times: torch.Tensor,
    indices: torch.Tensor,
    residual_scale: float | torch.Tensor,
    predicted_states: torch.Tensor | None = None,
    settings: AFM06aHardSampleInputs | None = None,
    smoothness_weight: float = 0.0,
) -> tuple[torch.Tensor, LossParts, dict[str, torch.Tensor]]:
    """Evaluate force-scale-normalized residual and residual-curvature losses."""

    idx = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=predicted_fts.device,
    ).reshape(-1)
    if idx.numel() == 0:
        raise ValueError("loss indices cannot be empty")
    if smoothness_weight < 0.0:
        raise ValueError("residual smoothness loss weight must be nonnegative")
    settings = AFM06aHardSampleInputs() if settings is None else settings
    settings.validate()
    residual = dynamics_residual_torch(
        predicted_fts=predicted_fts,
        true_states=true_states,
        true_x2dot=true_x2dot,
        times=times,
        settings=settings,
    )
    selected_residual = residual[idx]
    scale = torch.as_tensor(
        residual_scale,
        dtype=predicted_fts.dtype,
        device=predicted_fts.device,
    ).reshape(())
    force_scale = torch.clamp(
        torch.abs(scale),
        min=SCALE_EPS,
    )
    normalized_residual = selected_residual / force_scale
    dynamics_loss = torch.mean(torch.square(normalized_residual))
    _k, _omega0, _mass, _c, _fd, _ca, _ch, dist, a0, _beta = (
        settings.parameter_vector
    )
    contact_indicator = float(dist) + true_states[0] <= float(a0)
    smoothness_raw = residual_smoothness_torch(
        normalized_residual=normalized_residual,
        times=times,
        indices=idx,
        contact_indicator=contact_indicator,
    )
    smoothness = float(smoothness_weight) * smoothness_raw
    total_loss = dynamics_loss + smoothness

    x1_rec = float("nan")
    x2_rec = float("nan")
    if predicted_states is not None:
        if predicted_states.shape != true_states.shape:
            raise ValueError(
                "predicted_states must match true_states when supplied for diagnostics"
            )
        state_pred = predicted_states[:, idx]
        state_true = true_states[:, idx]
        x1_rec = float(
            _relative_rmse_pct(state_pred[0], state_true[0]).detach().cpu()
        )
        x2_rec = float(
            _relative_rmse_pct(state_pred[1], state_true[1]).detach().cpu()
        )

    parts = LossParts(
        dynamics_residual=float(dynamics_loss.detach().cpu()),
        smoothness=float(smoothness.detach().cpu()),
        smoothness_unweighted=float(smoothness_raw.detach().cpu()),
        residual_rmse=float(
            torch.sqrt(torch.mean(torch.square(selected_residual))).detach().cpu()
        ),
        residual_max_abs=float(torch.max(torch.abs(selected_residual)).detach().cpu()),
        residual_scale=float(force_scale.detach().cpu()),
        x1_rec=x1_rec,
        x2_rec=x2_rec,
    )
    return total_loss, parts, {
        "dynamics_residual": dynamics_loss,
        "smoothness": smoothness,
    }


def parts_as_dict(parts: LossParts) -> dict[str, float]:
    return {name: float(value) for name, value in asdict(parts).items()}


def nan_loss_parts() -> LossParts:
    return LossParts(
        dynamics_residual=float("nan"),
        smoothness=float("nan"),
        smoothness_unweighted=float("nan"),
        residual_rmse=float("nan"),
        residual_max_abs=float("nan"),
        residual_scale=float("nan"),
        x1_rec=float("nan"),
        x2_rec=float("nan"),
    )


def validate_loss_policy(policy_name: str | None) -> str:
    policy = "" if policy_name is None else str(policy_name).strip().lower()
    if policy != SUPPORTED_POLICY:
        raise ValueError(f"unsupported AFM06a loss policy: {policy_name!r}")
    return policy


__all__ = [
    "LossParts",
    "SUPPORTED_POLICY",
    "dynamics_residual_torch",
    "residual_smoothness_torch",
    "evaluate_dynamics_residual_loss",
    "evaluate_dynamics_residual_loss_with_terms",
    "nan_loss_parts",
    "parts_as_dict",
    "validate_loss_policy",
]
