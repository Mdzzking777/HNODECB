"""Single-trial compatibility shell for AFM05 stage1pluslight.

AFM05 stage1pluslight uses the KAN random-search backend in
``kan_rs_trial.py``.  This module keeps the AFM04-shaped trial bundle API
available without importing AFM04-only MLP or true-parameter helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class TrialModelBundle:
    model: Callable[[np.ndarray, Any], float] | None
    model_params: Any
    model_param_vec: np.ndarray
    unflatten: Callable[[np.ndarray], Any] | None
    force_mode: str
    backend_label: str


def build_zero_contact_model(seed: int = 0, num_hidden_layers: int = 0, num_hidden_nodes: int = 1) -> TrialModelBundle:
    _ = (seed, num_hidden_layers, num_hidden_nodes)

    def model(u: np.ndarray, _params: Any = None) -> float:
        _ = u
        return 0.0

    return TrialModelBundle(
        model=model,
        model_params=np.zeros(0, dtype=float),
        model_param_vec=np.zeros(0, dtype=float),
        unflatten=lambda flat: np.asarray(flat, dtype=float).reshape(-1),
        force_mode="direct_contact_model",
        backend_label="zero_contact",
    )


__all__ = ["TrialModelBundle", "build_zero_contact_model"]
