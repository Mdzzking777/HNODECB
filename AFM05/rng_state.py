"""RNG-state helpers shared by AFM05 checkpoint/resume code."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for utility import.
    torch = None  # type: ignore[assignment]


RNG_STATE_FORMAT = "afm05_rng_state_v1"


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "format": RNG_STATE_FORMAT,
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
    }
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Any) -> list[str]:
    restored: list[str] = []
    if not isinstance(state, dict):
        return restored
    try:
        if "python_random" in state:
            random.setstate(state["python_random"])
            restored.append("python")
    except Exception:
        pass
    try:
        if "numpy_random" in state:
            np.random.set_state(state["numpy_random"])
            restored.append("numpy")
    except Exception:
        pass
    if torch is None:
        return restored
    try:
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
            restored.append("torch_cpu")
    except Exception:
        pass
    try:
        cuda_state = state.get("torch_cuda_all")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
            restored.append("torch_cuda_all")
    except Exception:
        pass
    return restored


__all__ = ["RNG_STATE_FORMAT", "capture_rng_state", "restore_rng_state"]
