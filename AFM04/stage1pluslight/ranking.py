"""Ranking helpers for AFM04 stage1pluslight trial records."""

from __future__ import annotations

from typing import Any, Iterable

import math


def _rec_key(rec: dict[str, Any]) -> tuple[Any, ...]:
    loss = float(rec.get("loss", math.inf))
    viable = bool(rec.get("is_viable", False))
    ks_err = float(rec.get("ks_err_pct", math.inf))
    cs_err = float(rec.get("cs_err_pct", math.inf))
    params = rec.get("params", {}) if isinstance(rec, dict) else {}
    trial_id = int(params.get("trial_id", 0)) if isinstance(params, dict) else 0
    return (not viable, not math.isfinite(loss), loss, ks_err, cs_err, trial_id)


def rank_trial_records(records: Iterable[dict[str, Any]], *, topk: int | None = None, viable_only: bool = False) -> list[dict[str, Any]]:
    recs = [rec for rec in records if isinstance(rec, dict)]
    if viable_only:
        recs = [rec for rec in recs if bool(rec.get("is_viable", False))]
    ranked = sorted(recs, key=_rec_key)
    if topk is not None:
        ranked = ranked[: max(0, int(topk))]
    return ranked


__all__ = ["rank_trial_records"]
