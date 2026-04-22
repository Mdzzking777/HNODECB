"""Stage1pluslight control-plane helpers for AFM04.

Package import stays lazy so callers that only need lightweight modules, such
as grid/config helpers, do not pay the cost of importing the KAN backend.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "Stage1PlusLightConfig": ("AFM04.stage1pluslight.config", "Stage1PlusLightConfig"),
    "WindowManifest": ("AFM04.stage1pluslight.windows", "WindowManifest"),
    "LossParts": ("AFM04.stage1pluslight.losses", "LossParts"),
    "MLPParams": ("AFM04.stage1pluslight.mlp", "MLPParams"),
    "bound_param": ("AFM04.stage1pluslight.mathutils", "bound_param"),
    "build_mlp_bundle": ("AFM04.stage1pluslight.mlp", "build_mlp_bundle"),
    "build_mlp_model": ("AFM04.stage1pluslight.trial", "build_mlp_model"),
    "build_mlp_params": ("AFM04.stage1pluslight.mlp", "build_mlp_params"),
    "build_zero_contact_model": ("AFM04.stage1pluslight.trial", "build_zero_contact_model"),
    "compute_shard_assignments": ("AFM04.stage1pluslight.grid", "compute_shard_assignments"),
    "decode_grid_trial": ("AFM04.stage1pluslight.grid", "decode_grid_trial"),
    "default_config": ("AFM04.stage1pluslight.config", "default_config"),
    "export_stage1pluslight_final": ("AFM04.stage1pluslight.runner", "export_stage1pluslight_final"),
    "fts_pred_from_state": ("AFM04.stage1pluslight.rollout", "fts_pred_from_state"),
    "hidden_width_from_nodes": ("AFM04.stage1pluslight.mlp", "hidden_width_from_nodes"),
    "launch_local_shards": ("AFM04.stage1pluslight.runner", "launch_local_shards"),
    "load_resume_trials": ("AFM04.stage1pluslight.checkpoint", "load_resume_trials"),
    "loggrid_value": ("AFM04.stage1pluslight.grid", "loggrid_value"),
    "loss_single_or_ms": ("AFM04.stage1pluslight.losses", "loss_single_or_ms"),
    "make_uode_rhs": ("AFM04.stage1pluslight.rollout", "make_uode_rhs"),
    "merge_shard_exports": ("AFM04.stage1pluslight.runner", "merge_shard_exports"),
    "mlp_forward": ("AFM04.stage1pluslight.mlp", "mlp_forward"),
    "prepare_stage1pluslight_context": ("AFM04.stage1pluslight.runner", "prepare_stage1pluslight_context"),
    "raw_from_value": ("AFM04.stage1pluslight.mathutils", "raw_from_value"),
    "rank_context_trials": ("AFM04.stage1pluslight.runner", "rank_context_trials"),
    "rank_trial_records": ("AFM04.stage1pluslight.ranking", "rank_trial_records"),
    "rollout_single_shooting": ("AFM04.stage1pluslight.rollout", "rollout_single_shooting"),
    "run_shard_mainloop": ("AFM04.stage1pluslight.runner", "run_shard_mainloop"),
    "run_trial_from_context": ("AFM04.stage1pluslight.runner", "run_trial_from_context"),
    "save_stage1pluslight_partial": ("AFM04.stage1pluslight.runner", "save_stage1pluslight_partial"),
    "stage1pluslight_joint_trial": ("AFM04.stage1pluslight.trial", "stage1pluslight_joint_trial"),
    "stage1pluslight_joint_trial_multiwindow": ("AFM04.stage1pluslight.trial", "stage1pluslight_joint_trial_multiwindow"),
    "stage1pluslight_nn_bank_seed": ("AFM04.stage1pluslight.mathutils", "stage1pluslight_nn_bank_seed"),
    "stage2_w1_window_indices": ("AFM04.stage1pluslight.windows", "stage2_w1_window_indices"),
    "stage2_window_manifest": ("AFM04.stage1pluslight.windows", "stage2_window_manifest"),
    "total_trials": ("AFM04.stage1pluslight.grid", "total_trials"),
    "trial_id_or_zero": ("AFM04.stage1pluslight.checkpoint", "trial_id_or_zero"),
    "window_manifests": ("AFM04.stage1pluslight.windows", "window_manifests"),
    "write_checkpoint_atomic": ("AFM04.stage1pluslight.checkpoint", "write_checkpoint_atomic"),
    "x2dot_rhs": ("AFM04.stage1pluslight.rollout", "x2dot_rhs"),
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
