"""Checkpoint helpers for AFM05 stage1pluslight."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping

from AFM05.rng_state import restore_rng_state


def trial_id_or_zero(record: Any) -> int:
    params = record.get("params") if isinstance(record, Mapping) else getattr(record, "params", None)
    if not isinstance(params, Mapping):
        return 0
    trial_id = params.get("trial_id", 0)
    return int(trial_id) if isinstance(trial_id, int) else 0


def load_resume_trials(
    path: str | Path,
    *,
    searches_total: int,
    shard_idx: int,
    shard_cnt: int,
    ks_nodes: int,
    cs_nodes: int,
    nn_seeds_per_node: int,
    window_mode: str,
    pixel_tag: str,
    arch_window_us: float,
    window_sample_stride: int,
) -> tuple[list[Any], str]:
    checkpoint_path = Path(path)
    backup_path = checkpoint_path.with_name(checkpoint_path.name + ".bak")
    candidates: list[Path] = []
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0:
        candidates.append(checkpoint_path)
    if backup_path.is_file() and backup_path.stat().st_size > 0:
        candidates.append(backup_path)
    if not candidates:
        return [], ""

    data = None
    loaded_from = ""
    last_reason = ""
    for candidate in candidates:
        try:
            with open(candidate, "rb") as fh:
                data = pickle.load(fh)
        except Exception as err:
            last_reason = f"deserialize_failed:{err}"
            continue
        loaded_from = "backup" if candidate == backup_path else "primary"
        break

    if data is None:
        return [], last_reason or "missing_checkpoint_payload"
    if "trial_parameters" not in data:
        return [], "missing_trial_parameters"
    if data.get("searches_per_candidate", searches_total) != searches_total:
        return [], "searches_total_mismatch"
    if data.get("stage1plus_shard_index", shard_idx) != shard_idx:
        return [], "shard_index_mismatch"
    if data.get("stage1plus_shard_count", shard_cnt) != shard_cnt:
        return [], "shard_count_mismatch"
    if data.get("stage1plus_grid_ks_nodes", ks_nodes) != ks_nodes:
        return [], "ks_nodes_mismatch"
    if data.get("stage1plus_grid_cs_nodes", cs_nodes) != cs_nodes:
        return [], "cs_nodes_mismatch"
    if data.get("stage1plus_grid_nn_seeds_per_node", nn_seeds_per_node) != nn_seeds_per_node:
        return [], "nn_seeds_mismatch"
    if str(data.get("stage1plus_window_mode", window_mode)) != str(window_mode):
        return [], "window_mode_mismatch"
    if "stage1plus_window_pixel_tag" not in data:
        return [], "window_pixel_tag_missing"
    if str(data.get("stage1plus_window_pixel_tag")) != str(pixel_tag):
        return [], "window_pixel_tag_mismatch"
    old_arch_window_us = data.get("stage1plus_arch_window_us")
    if old_arch_window_us is None:
        return [], "arch_window_us_missing"
    try:
        old_arch_window_us_f = float(old_arch_window_us)
    except Exception:
        return [], "arch_window_us_invalid"
    if abs(old_arch_window_us_f - float(arch_window_us)) > max(1.0e-15, 1.0e-9 * abs(float(arch_window_us))):
        return [], "arch_window_us_mismatch"
    try:
        old_sample_stride = int(data.get("stage1plus_window_sample_stride"))
    except Exception:
        return [], "window_sample_stride_missing_or_invalid"
    if old_sample_stride != int(window_sample_stride):
        return [], "window_sample_stride_mismatch"
    notes: list[str] = []
    if loaded_from == "backup":
        notes.append("loaded_from_backup")
    restored_rng = restore_rng_state(data.get("rng_state"))
    if restored_rng:
        notes.append(f"rng_restored={','.join(restored_rng)}")
    elif "rng_state" not in data:
        notes.append("legacy_checkpoint_no_rng_state")
    return list(data["trial_parameters"]), ";".join(notes)


def write_checkpoint_atomic(path: str | Path, payload: Any) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    backup_path = checkpoint_path.with_name(checkpoint_path.name + ".bak")

    if tmp_path.exists():
        tmp_path.unlink()
    with open(tmp_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())

    if checkpoint_path.exists():
        if backup_path.exists():
            backup_path.unlink()
        os.replace(checkpoint_path, backup_path)

    os.replace(tmp_path, checkpoint_path)


__all__ = ["load_resume_trials", "trial_id_or_zero", "write_checkpoint_atomic"]
