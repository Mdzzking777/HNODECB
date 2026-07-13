"""Validation helpers for complete AFM05 stage1pluslight merged results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from AFM05.stage1pluslight.checkpoint import trial_id_or_zero


def _label(path: str | Path | None) -> str:
    return str(path) if path is not None else "stage1pluslight payload"


def _records(payload: Mapping[str, Any], path: str | Path | None) -> list[dict[str, Any]]:
    trial_parameters = payload.get("trial_parameters")
    if not isinstance(trial_parameters, list):
        raise RuntimeError(f"{_label(path)} is missing list 'trial_parameters'")
    records = [rec for rec in trial_parameters if isinstance(rec, dict)]
    if len(records) != len(trial_parameters):
        raise RuntimeError(f"{_label(path)} contains non-dict trial records")
    return records


def expected_stage1plus_trial_count(payload: Mapping[str, Any]) -> int:
    for key in ("stage1plus_total_trials", "searches_per_candidate"):
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            return int(value)
    return 0


def validate_complete_stage1plus_payload(
    payload: Mapping[str, Any],
    *,
    path: str | Path | None = None,
    require_merged: bool = True,
) -> int:
    """Raise if a payload is not a complete AFM05 all-trials result."""

    if bool(payload.get("stage1plus_partial", False)):
        raise RuntimeError(f"Refusing partial AFM05 stage1pluslight result: {_label(path)}")
    if not bool(payload.get("stage1plus_complete", False)):
        raise RuntimeError(f"Refusing incomplete AFM05 stage1pluslight result: {_label(path)}")
    if require_merged and not bool(payload.get("stage1plus_merged", False)):
        raise RuntimeError(f"Refusing non-merged AFM05 stage1pluslight result: {_label(path)}")

    records = _records(payload, path)
    expected_total = expected_stage1plus_trial_count(payload)
    if expected_total <= 0:
        raise RuntimeError(f"{_label(path)} is missing positive expected trial count")
    if len(records) != expected_total:
        raise RuntimeError(
            f"Refusing AFM05 stage1pluslight result with incomplete trial coverage: "
            f"{_label(path)} records={len(records)} expected={expected_total}"
        )

    trial_ids = [trial_id_or_zero(rec) for rec in records]
    if any(tid <= 0 for tid in trial_ids):
        raise RuntimeError(f"{_label(path)} contains trial records without positive trial_id")
    if len(set(trial_ids)) != len(trial_ids):
        raise RuntimeError(f"{_label(path)} contains duplicate trial_id records")

    for rec in records:
        if not isinstance(rec, dict):
            continue
        if bool(rec.get("is_viable", False)):
            loss = float(rec.get("loss", math.inf))
            train_loss = float(rec.get("train_loss", math.inf))
            if not math.isfinite(loss) or not math.isfinite(train_loss):
                raise RuntimeError(f"{_label(path)} has viable record with non-finite loss")
            if float(rec.get("ks_hat", math.nan)) <= 0.0 or float(rec.get("cs_hat", math.nan)) <= 0.0:
                raise RuntimeError(f"{_label(path)} has viable record with non-positive ks/cs")

    return len(records)


__all__ = [
    "expected_stage1plus_trial_count",
    "validate_complete_stage1plus_payload",
]
