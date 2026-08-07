"""AFM06a stage2light optimizer schedule, migrated from AFM04."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage2LightSchedule:
    epochs: int
    adam_epochs: int
    lbfgs_epochs: int


def load_stage2light_schedule(repo_root: str | Path) -> Stage2LightSchedule:
    path = Path(repo_root) / "AFM06a" / "stage2light" / "stage2light_schedule.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    schedule = Stage2LightSchedule(
        epochs=int(payload["epochs"]),
        adam_epochs=int(payload["adam_epochs"]),
        lbfgs_epochs=int(payload["lbfgs_epochs"]),
    )
    if schedule.epochs <= 0 or schedule.adam_epochs < 0 or schedule.lbfgs_epochs < 0:
        raise ValueError("stage2light schedule contains invalid epoch counts")
    if schedule.adam_epochs + schedule.lbfgs_epochs != schedule.epochs:
        raise ValueError("adam_epochs + lbfgs_epochs must equal epochs")
    return schedule


__all__ = ["Stage2LightSchedule", "load_stage2light_schedule"]
