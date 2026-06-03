"""Shared formal stage2light optimizer schedule defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EPOCHS = 200
DEFAULT_ADAM_EPOCHS = 20
DEFAULT_LBFGS_EPOCHS = 180


@dataclass(frozen=True)
class Stage2LightSchedule:
    epochs: int = DEFAULT_EPOCHS
    adam_epochs: int = DEFAULT_ADAM_EPOCHS
    lbfgs_epochs: int = DEFAULT_LBFGS_EPOCHS


def _positive_int(payload: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(payload.get(key, default))
    except Exception:
        return int(default)
    return max(0, value)


def load_stage2light_schedule(repo_root: str | Path) -> Stage2LightSchedule:
    repo = Path(repo_root)
    path = repo / "AFM04" / "stage2light" / "stage2light_schedule.json"
    if not path.is_file():
        return Stage2LightSchedule()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return Stage2LightSchedule()
    if not isinstance(payload, dict):
        return Stage2LightSchedule()
    epochs = max(1, _positive_int(payload, "epochs", DEFAULT_EPOCHS))
    adam_epochs = min(epochs, _positive_int(payload, "adam_epochs", DEFAULT_ADAM_EPOCHS))
    lbfgs_epochs = _positive_int(payload, "lbfgs_epochs", max(0, epochs - adam_epochs))
    return Stage2LightSchedule(epochs=epochs, adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)


__all__ = [
    "DEFAULT_ADAM_EPOCHS",
    "DEFAULT_EPOCHS",
    "DEFAULT_LBFGS_EPOCHS",
    "Stage2LightSchedule",
    "load_stage2light_schedule",
]
