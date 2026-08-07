"""AFM06a stage2light preserves the st1pl dynamics-residual definition."""

from AFM06a.stage1pluslight.losses import (
    LossParts,
    SUPPORTED_POLICY,
    dynamics_residual_torch,
    evaluate_dynamics_residual_loss,
    evaluate_dynamics_residual_loss_with_terms,
    nan_loss_parts,
    parts_as_dict,
    residual_smoothness_torch,
    validate_loss_policy,
)

__all__ = [
    "LossParts",
    "SUPPORTED_POLICY",
    "dynamics_residual_torch",
    "evaluate_dynamics_residual_loss",
    "evaluate_dynamics_residual_loss_with_terms",
    "nan_loss_parts",
    "parts_as_dict",
    "residual_smoothness_torch",
    "validate_loss_policy",
]
