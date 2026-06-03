"""Standalone AFM04 Python stage2light package.

Keep package import lightweight so visualization scripts can run without
pulling in the full training stack and optional deps like torch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "KANForceModule": ("AFM04.stage2light.kan_backend", "KANForceModule"),
    "LearnableMechModule": ("AFM04.stage2light.rollout", "LearnableMechModule"),
    "PreparedData": ("AFM04.stage2light.data", "PreparedData"),
    "Prestage2WarmstartCandidate": ("AFM04.stage2light.data", "Prestage2WarmstartCandidate"),
    "Stage1WarmstartCandidate": ("AFM04.stage2light.data", "Stage1WarmstartCandidate"),
    "Stage2LightConfig": ("AFM04.stage2light.config", "Stage2LightConfig"),
    "WindowSplit": ("AFM04.stage2light.data", "WindowSplit"),
    "default_config": ("AFM04.stage2light.config", "default_config"),
    "merge_shard_results": ("AFM04.stage2light.train", "merge_shard_results"),
    "merge_stage2light_results": ("AFM04.stage2light.train", "merge_stage2light_results"),
    "prepare_data": ("AFM04.stage2light.data", "prepare_data"),
    "run_full_test": ("AFM04.stage2light.train", "run_full_test"),
    "run_full_test_shard": ("AFM04.stage2light.train", "run_full_test_shard"),
    "run_stage2light": ("AFM04.stage2light.train", "run_stage2light"),
    "run_stage2light_shard": ("AFM04.stage2light.train", "run_stage2light_shard"),
    "select_prestage2_candidate": ("AFM04.stage2light.data", "select_prestage2_candidate"),
    "select_stage1_candidate": ("AFM04.stage2light.data", "select_stage1_candidate"),
    "select_stage1_candidate_by_trial_id": ("AFM04.stage2light.data", "select_stage1_candidate_by_trial_id"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
