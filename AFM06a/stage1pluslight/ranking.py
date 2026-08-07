"""Deterministic ranking for AFM06a seed-screening records."""

from __future__ import annotations

from typing import Any, Iterable


def _trial_id(record: dict[str, Any]) -> int:
    params = record.get("params", {})
    return int(params.get("trial_id", record.get("trial_id", 0))) if isinstance(params, dict) else 0


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    viable = bool(record.get("is_viable", False)) and not bool(record.get("trial_failed", False))
    loss = float(record.get("loss", float("inf")))
    return (0 if viable else 1, loss, _trial_id(record))


def rank_trial_records(
    records: Iterable[dict[str, Any]],
    *,
    topk: int | None = None,
    viable_only: bool = False,
) -> list[dict[str, Any]]:
    selected = [dict(record) for record in records]
    if viable_only:
        selected = [record for record in selected if bool(record.get("is_viable", False))]
    ranked = sorted(selected, key=_record_key)
    return ranked if topk is None else ranked[: max(0, int(topk))]


__all__ = ["rank_trial_records"]
