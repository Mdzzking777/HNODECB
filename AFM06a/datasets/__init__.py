"""Dataset-generation utilities for AFM06a."""

from typing import Any

from .non_perturbed_dataset_generator import generate_non_perturbed_training_set


def generate_afm_dmt_hard_dataset(*args: Any, **kwargs: Any) -> dict[str, object]:
    """Lazily import and run the AFM06a hard-sample generator."""

    from .afm_dataset_generator import generate_afm_dmt_hard_dataset as _generate

    return _generate(*args, **kwargs)

__all__ = [
    "generate_afm_dmt_hard_dataset",
    "generate_non_perturbed_training_set",
]
