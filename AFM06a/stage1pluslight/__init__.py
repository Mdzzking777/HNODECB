"""AFM06a stage1pluslight control-plane package.

The package intentionally keeps imports lazy.  The first migration pass fixes
the AFM06a contracts (two ODE states, a single-input 1-3-1 KAN, and seed-only screening) while
leaving experiment policies behind an explicit readiness gate.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "Stage1PlusLightConfig": ("AFM06a.stage1pluslight.config", "Stage1PlusLightConfig"),
    "default_config": ("AFM06a.stage1pluslight.config", "default_config"),
    "load_dataset": ("AFM06a.stage1pluslight.data", "load_dataset"),
    "prepare_window": ("AFM06a.stage1pluslight.data", "prepare_window"),
    "SeedTrial": ("AFM06a.stage1pluslight.seed_search", "SeedTrial"),
    "decode_seed_trial": ("AFM06a.stage1pluslight.seed_search", "decode_seed_trial"),
    "compute_shard_assignments": ("AFM06a.stage1pluslight.seed_search", "compute_shard_assignments"),
    "rank_trial_records": ("AFM06a.stage1pluslight.ranking", "rank_trial_records"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
