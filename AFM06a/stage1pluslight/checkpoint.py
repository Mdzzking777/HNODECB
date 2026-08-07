"""Atomic checkpoint and resume helpers for AFM06a stage1pluslight."""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Any


def trial_id_or_zero(record: Any) -> int:
    if not isinstance(record, dict):
        return 0
    params = record.get("params", {})
    if isinstance(params, dict):
        return int(params.get("trial_id", 0) or 0)
    return int(record.get("trial_id", 0) or 0)


def write_checkpoint_atomic(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(50):
        try:
            os.replace(temporary, target)
            return
        except OSError as exc:
            transient_windows_lock = (
                os.name == "nt" and getattr(exc, "winerror", None) in {5, 32}
            )
            if not transient_windows_lock or attempt == 49:
                raise
            time.sleep(0.1)


def load_checkpoint(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    with target.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid AFM06a stage1pluslight checkpoint: {target}")
    return payload


def load_resume_trials(path: str | Path) -> dict[int, dict[str, Any]]:
    payload = load_checkpoint(path)
    if payload is None:
        return {}
    records = payload.get("trial_parameters", [])
    if not isinstance(records, list):
        raise RuntimeError("checkpoint field trial_parameters must be a list")
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        trial_id = trial_id_or_zero(record)
        if trial_id <= 0 or not isinstance(record, dict):
            continue
        if trial_id in indexed:
            raise RuntimeError(f"duplicate trial_id in checkpoint: {trial_id}")
        indexed[trial_id] = record
    return indexed


__all__ = ["load_checkpoint", "load_resume_trials", "trial_id_or_zero", "write_checkpoint_atomic"]
