"""Completeness checks for merged AFM06a stage1pluslight exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from AFM06a.stage1pluslight.checkpoint import trial_id_or_zero


def validate_complete_stage1plus_payload(
    payload: Mapping[str, Any],
    *,
    path: str | Path | None = None,
    require_merged: bool = True,
) -> int:
    label = str(path) if path is not None else "AFM06a stage1pluslight payload"
    if bool(payload.get("stage1plus_partial", False)):
        raise RuntimeError(f"refusing partial result: {label}")
    if not bool(payload.get("stage1plus_complete", False)):
        raise RuntimeError(f"refusing incomplete result: {label}")
    if require_merged and not bool(payload.get("stage1plus_merged", False)):
        raise RuntimeError(f"refusing non-merged result: {label}")
    records = payload.get("trial_parameters")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise RuntimeError(f"invalid trial_parameters in {label}")
    expected = int(payload.get("stage1plus_total_trials", 0) or 0)
    if expected <= 0 or len(records) != expected:
        raise RuntimeError(f"incomplete coverage in {label}: records={len(records)} expected={expected}")
    trial_ids = [trial_id_or_zero(record) for record in records]
    if any(trial_id <= 0 for trial_id in trial_ids) or len(set(trial_ids)) != len(trial_ids):
        raise RuntimeError(f"invalid or duplicate trial IDs in {label}")
    return len(records)


__all__ = ["validate_complete_stage1plus_payload"]
