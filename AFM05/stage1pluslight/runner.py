"""Runner entrypoints for AFM05 stage1pluslight."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from AFM05.rng_state import capture_rng_state
from AFM05.stage1pluslight.checkpoint import load_resume_trials, trial_id_or_zero, write_checkpoint_atomic
from AFM05.stage1pluslight.config import Stage1PlusLightConfig, default_config
from AFM05.stage1pluslight.data import initial_condition_aligned_dataset, load_dataset, make_train_val_masks
from AFM05.stage1pluslight.grid import CS_BOUNDS, KS_BOUNDS, compute_shard_assignments, decode_grid_trial, total_trials
from AFM05.stage1pluslight.kan_rs_trial import prepare_kan_stage1_runtime, stage1pluslight_kan_random_trial
from AFM05.stage1pluslight.ranking import rank_trial_records
from AFM05.stage1pluslight.trial import TrialModelBundle
from AFM05.stage1pluslight.windows import WindowManifest, window_manifests


def timestamp_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Stage1Logger:
    path: Path
    echo: bool = True

    def log(self, message: str) -> None:
        line = f"[{timestamp_human()}] {message}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            print(line, flush=True)


def make_shard_log_path(log_dir: Path, *, shard_index: int, shard_count: int, run_tag: str = "") -> Path:
    suffix = "" if run_tag.strip() == "" else f"_{run_tag.strip()}"
    return log_dir / f"afm05_stage1pluslight_local_p{shard_index}of{shard_count}{suffix}_{timestamp_compact()}.txt"


def make_driver_log_path(log_dir: Path, *, run_tag: str = "") -> Path:
    suffix = "" if run_tag.strip() == "" else f"_{run_tag.strip()}"
    return log_dir / f"afm05_stage1pluslight_driver{suffix}_{timestamp_compact()}.txt"


def format_trial_line(record: dict[str, Any]) -> str:
    params = record.get("params", {}) if isinstance(record, dict) else {}
    parts = record.get("val_parts", {}) if isinstance(record, dict) else {}
    trial_id = int(params.get("trial_id", 0))
    node_label = str(params.get("node_label", "node_unknown"))
    seedbank = int(params.get("nn_seed_bank_idx", 0))
    initseed = int(params.get("nn_init_seed", seedbank))
    loss = float(record.get("loss", float("inf")))
    train_loss = float(record.get("train_loss", float("inf")))
    val_loss = float(record.get("val_loss", float("nan")))
    ks_hat = float(record.get("ks_hat", float("nan")))
    cs_hat = float(record.get("cs_hat", float("nan")))
    x1_rec = float(parts.get("x1_rec", float("nan"))) if isinstance(parts, dict) else float("nan")
    x2_rec = float(parts.get("x2_rec", float("nan"))) if isinstance(parts, dict) else float("nan")
    x2dot_rec = float(parts.get("x2dot_rec", float("nan"))) if isinstance(parts, dict) else float("nan")
    x3n_min = float(params.get("x3_norm_KAN_min", float("nan")))
    x3n_max = float(params.get("x3_norm_KAN_max", float("nan")))
    x3n_out1 = float(params.get("x3_norm_KAN_outside_1_frac", float("nan")))
    x3n_out2 = float(params.get("x3_norm_KAN_outside_2_frac", float("nan")))
    viable = bool(record.get("is_viable", False))
    failed = bool(record.get("trial_failed", False))
    reason = str(record.get("failure_reason", "") or record.get("train_reason", "") or record.get("val_reason", "") or "")
    return (
        f"trial {trial_id} | {node_label} | seedbank={seedbank} initseed={initseed} | "
        f"train={train_loss:.6e} val={val_loss:.6e} loss={loss:.6e} | "
        f"ks={ks_hat:.6e} cs={cs_hat:.6e} | "
        f"x1_rec={x1_rec:.5f}% x2_rec={x2_rec:.5f}% x2dot_rec={x2dot_rec:.5f}% | "
        f"x3n_KAN=[{x3n_min:.3f},{x3n_max:.3f}] out1={x3n_out1:.3f} out2={x3n_out2:.3f} | "
        f"viable={viable} failed={failed}"
        + (f" | reason={reason}" if reason else "")
    )


def format_progress_line(done: int, total: int, *, pending: int, best_loss: float | None) -> str:
    pct = 100.0 * float(done) / max(1, int(total))
    best_txt = "nan" if best_loss is None else f"{best_loss:.6e}"
    return f"progress {done}/{total} ({pct:.2f}%) | pending={pending} | best_loss={best_txt}"


@dataclass
class Stage1PlusLightContext:
    config: Stage1PlusLightConfig
    kan_runtime: dict[str, Any]
    ode_data_full: np.ndarray
    solution_table_full: dict[str, np.ndarray]
    all_times: np.ndarray
    x2dot_all: np.ndarray
    gain_force_reference_all: np.ndarray
    contact_all: np.ndarray
    s_all: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    main_windows: list[WindowManifest]
    window_bundles: list[dict[str, Any]]
    searches_total: int
    assignments: list[int]
    trial_parameters: list[Any]
    pending_assignments: list[int]
    state12_scale_full: np.ndarray
    x2dot_scale_full: float
    x3_scale: float
    resume_reason: str


def _window_manifest_payload(windows: list[WindowManifest] | None, *, pixel_tag: str) -> list[dict[str, Any]]:
    if not windows:
        return []
    return [
        {
            "role": str(win.role),
            "label": str(win.label),
            "pixel_tag": str(pixel_tag),
            "start_idx": int(win.start_idx),
            "stop_idx": int(win.stop_idx),
            "length": int(win.length),
            "t_start": float(win.t_start),
            "t_stop": float(win.t_stop),
            "sample_stride": int(win.sample_stride),
        }
        for win in windows
    ]


def _stage1plus_window_identity(config: Stage1PlusLightConfig, windows: list[WindowManifest] | None) -> dict[str, Any]:
    return {
        "stage1plus_window_mode": str(config.window_mode),
        "stage1plus_window_pixel_tag": str(config.pixel_tag),
        "stage1plus_arch_window_us": float(config.arch_window_us),
        "stage1plus_window_sample_stride": int(config.window_sample_stride),
        "stage1plus_windows": _window_manifest_payload(windows, pixel_tag=str(config.pixel_tag)),
    }


def _stage1plus_loss_scale_payload() -> dict[str, Any]:
    return {
        "stage1plus_loss_scale_mode": "stage2_prestage2_aligned_rms",
        "stage1plus_state_x2dot_scale": "rms_over_current_loss_indices",
        "stage1plus_model_side_range_amp": "mean_abs_x1_and_F_actuation_over_current_window",
    }


def _payload_matches_stage1plus_window(payload: dict[str, Any], *, config: Stage1PlusLightConfig) -> tuple[bool, str]:
    if str(payload.get("stage1plus_window_mode", "")) != str(config.window_mode):
        return False, "window_mode_mismatch"
    if str(payload.get("stage1plus_window_pixel_tag", "")) != str(config.pixel_tag):
        return False, "window_pixel_tag_mismatch"
    try:
        old_arch = float(payload.get("stage1plus_arch_window_us"))
    except Exception:
        return False, "arch_window_us_missing_or_invalid"
    if abs(old_arch - float(config.arch_window_us)) > max(1.0e-15, 1.0e-9 * abs(float(config.arch_window_us))):
        return False, f"arch_window_us_mismatch:{old_arch:.12e}!={float(config.arch_window_us):.12e}"
    try:
        old_stride = int(payload.get("stage1plus_window_sample_stride"))
    except Exception:
        return False, "window_sample_stride_missing_or_invalid"
    if old_stride != int(config.window_sample_stride):
        return False, f"window_sample_stride_mismatch:{old_stride}!={int(config.window_sample_stride)}"
    return True, "match"


def _slice_with_mask(values: np.ndarray, idxs: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.asarray(values)[idxs][keep]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _data_dir(config: Stage1PlusLightConfig) -> Path:
    return config.dataset_root / config.error_level / "data"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _load_known_pars(config: Stage1PlusLightConfig) -> dict[str, Any]:
    data_dir = _data_dir(config)
    metadata = _load_json(data_dir / "afm05_st1pl_entry_metadata.json")
    if str(metadata.get("pixel_tag", "")).strip() != str(config.pixel_tag):
        raise ValueError(
            "AFM05 metadata pixel_tag does not match config: "
            f"{metadata.get('pixel_tag')!r} != {config.pixel_tag!r}"
        )
    with np.load(data_dir / "afm05_F_actuation.npz") as data:
        act_t = np.asarray(data["t"], dtype=float)
        f_act = np.asarray(data["F_actuation"], dtype=float)
    if act_t.ndim != 1 or f_act.ndim != 1 or act_t.size != f_act.size:
        raise ValueError("AFM05 F_actuation npz must contain equal-length 1D t and F_actuation arrays")
    if not np.all(np.isfinite(act_t)) or not np.all(np.isfinite(f_act)):
        raise ValueError("AFM05 F_actuation trajectory contains non-finite values")
    return {
        "k_eff": float(metadata["k_eff"]),
        "m_eff": float(metadata["m_eff"]),
        "c_eff": float(metadata["c_eff"]),
        "actuation_times": act_t,
        "F_actuation": f_act,
        "Z": float(metadata["dist_Z_m"]),
        "a0": float(metadata["a0_m"]),
        "metadata": metadata,
    }


def _load_gain_force_reference(config: Stage1PlusLightConfig, expected_times: np.ndarray) -> np.ndarray:
    data_dir = _data_dir(config)
    key = f"F_ts_{config.pixel_tag}"
    path = data_dir / f"afm05_F_ts_{config.pixel_tag}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"AFM05 gain force reference file is missing: {path}")
    with np.load(path) as data:
        times = np.asarray(data["t"], dtype=float)
        if key not in data.files:
            raise KeyError(f"AFM05 gain force reference file {path} is missing key {key!r}")
        values = np.asarray(data[key], dtype=float)
    expected_times = np.asarray(expected_times, dtype=float)
    if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
        raise ValueError(f"AFM05 {key} npz must contain equal-length 1D t and force arrays")
    if times.size != expected_times.size or not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-15):
        raise ValueError(f"AFM05 {key} time grid does not match the stage1pluslight entry dataset")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"AFM05 {key} contains non-finite values")
    return values


def save_stage1pluslight_partial(
    config: Stage1PlusLightConfig,
    trial_parameters: list[Any],
    *,
    searches_total: int,
    windows: list[WindowManifest] | None = None,
) -> None:
    payload = {
        "trial_parameters": trial_parameters,
        "bounds": {"ks": KS_BOUNDS, "cs": CS_BOUNDS},
        "true_values": {},
        "val_stride": config.val_stride,
        "val_offset": config.val_offset,
        "error_level": config.error_level,
        "stage1plus_standalone": True,
        "stage1plus_kan_search": True,
        "stage1plus_search_mode": f"kscs_grid_kan_seedbank_{config.window_mode}",
        "stage1plus_grid_ks_nodes": config.ks_node_count,
        "stage1plus_grid_cs_nodes": config.cs_node_count,
        "stage1plus_grid_nn_seeds_per_node": config.nn_seed_bank_size,
        **_stage1plus_loss_scale_payload(),
        **_stage1plus_window_identity(config, windows),
        "run_seed": config.run_seed,
        "use_multiple_shooting": config.use_multiple_shooting,
        "ms_group_size": config.ms_group_size,
        "ms_continuity_term": config.ms_continuity_term,
        "trial_opt_enabled": config.trial_opt_enabled,
        "trial_epochs": config.trial_epochs,
        "trial_lr": config.trial_lr,
        "ode_solver": config.ode_solver,
        "ode_fallback_solver": config.ode_fallback_solver,
        "ode_rtol": config.ode_rtol,
        "ode_atol": config.ode_atol,
        "ode_max_step": config.ode_max_step,
        "searches_per_candidate": searches_total,
        "final_topk": config.final_topk,
        "stage1plus_partial": True,
        "stage1plus_shard_index": config.shard_index,
        "stage1plus_shard_count": config.shard_count,
        "AFM05_true_side_available": False,
        "rng_state": capture_rng_state(),
    }
    write_checkpoint_atomic(config.result_path, payload)


def prepare_stage1pluslight_context(
    config: Stage1PlusLightConfig | None = None,
    *,
    auto_generate_dataset: bool = False,
) -> Stage1PlusLightContext:
    if config is None:
        config = default_config()

    ode_data_full, solution_table_full = load_dataset(
        config.dataset_root,
        config.error_level,
        pixel_tag=config.pixel_tag,
        auto_generate=auto_generate_dataset,
    )
    ode_data_full, solution_table_full = initial_condition_aligned_dataset(ode_data_full, solution_table_full)

    all_times = np.asarray(solution_table_full["t"], dtype=float)
    x2dot_all = np.asarray(solution_table_full["x2dot"], dtype=float)
    gain_force_reference_all = _load_gain_force_reference(config, all_times)
    contact_all = np.asarray(solution_table_full.get("contact", np.zeros(all_times.size, dtype=int)), dtype=int).astype(bool)
    s_all = np.asarray(solution_table_full.get("s", np.full(all_times.size, np.nan)), dtype=float)
    train_idx, val_idx = make_train_val_masks(len(all_times), config.val_stride, config.val_offset)
    main_windows = window_manifests(
        all_times,
        config.window_mode,
        config.arch_window_us,
        sample_stride=config.window_sample_stride,
    )

    window_bundles: list[dict[str, Any]] = []
    for win in main_windows:
        local_train_keep = np.isin(win.idxs, train_idx)
        local_val_keep = np.isin(win.idxs, val_idx)
        window_bundles.append(
            {
                "role": win.role,
                "label": win.label,
                "pixel_tag": str(config.pixel_tag),
                "start_idx": win.start_idx,
                "stop_idx": win.stop_idx,
                "len": win.length,
                "t_start": win.t_start,
                "t_stop": win.t_stop,
                "ode_full": ode_data_full[:, win.idxs],
                "times_full": np.asarray(all_times[win.idxs], dtype=float),
                "x2dot_full": np.asarray(x2dot_all[win.idxs], dtype=float),
                "gain_force_reference_full": np.asarray(gain_force_reference_all[win.idxs], dtype=float),
                "contact_full": np.asarray(contact_all[win.idxs], dtype=bool),
                "s_full": np.asarray(s_all[win.idxs], dtype=float),
                "train_idx": np.flatnonzero(local_train_keep),
                "val_idx": np.flatnonzero(local_val_keep),
                "ode_train": ode_data_full[:, win.idxs][:, local_train_keep],
                "ode_val": ode_data_full[:, win.idxs][:, local_val_keep],
                "times_train": _slice_with_mask(all_times, win.idxs, local_train_keep),
                "times_val": _slice_with_mask(all_times, win.idxs, local_val_keep),
                "x2dot_train": _slice_with_mask(x2dot_all, win.idxs, local_train_keep),
                "x2dot_val": _slice_with_mask(x2dot_all, win.idxs, local_val_keep),
                "train_gain_force_reference": _slice_with_mask(gain_force_reference_all, win.idxs, local_train_keep),
                "val_gain_force_reference": _slice_with_mask(gain_force_reference_all, win.idxs, local_val_keep),
                "contact_train": _slice_with_mask(contact_all.astype(bool), win.idxs, local_train_keep),
                "contact_val": _slice_with_mask(contact_all.astype(bool), win.idxs, local_val_keep),
                "x3_t0_val": float(ode_data_full[2, win.idxs[0]]),
            }
        )
    if not window_bundles:
        raise RuntimeError("AFM05 stage1pluslight selected no windows")

    searches_total = total_trials(config.ks_node_count, config.cs_node_count, config.nn_seed_bank_size)
    assignments = compute_shard_assignments(searches_total, config.shard_index, config.shard_count)

    trial_parameters: list[Any] = []
    resume_reason = ""
    if config.resume_enabled:
        trial_parameters, resume_reason = load_resume_trials(
            config.result_path,
            searches_total=searches_total,
            shard_idx=config.shard_index,
            shard_cnt=config.shard_count,
            ks_nodes=config.ks_node_count,
            cs_nodes=config.cs_node_count,
            nn_seeds_per_node=config.nn_seed_bank_size,
            window_mode=config.window_mode,
            pixel_tag=config.pixel_tag,
            arch_window_us=config.arch_window_us,
            window_sample_stride=config.window_sample_stride,
            expected_windows=_window_manifest_payload(main_windows, pixel_tag=str(config.pixel_tag)),
        )

    completed_trial_ids = {trial_id_or_zero(rec) for rec in trial_parameters if trial_id_or_zero(rec) > 0}
    pending_assignments = [i for i in assignments if i not in completed_trial_ids]

    scale_window = window_bundles[0]
    scale_train_idx = np.asarray(scale_window["train_idx"], dtype=int)
    if scale_train_idx.size <= 0:
        scale_train_idx = np.arange(np.asarray(scale_window["ode_full"]).shape[1], dtype=int)
    scale_ode_train = np.asarray(scale_window["ode_full"], dtype=float)[0:2, :][:, scale_train_idx]
    scale_x2dot_train = np.asarray(scale_window["x2dot_full"], dtype=float)[scale_train_idx]
    state12_scale_full = np.sqrt(np.mean(np.square(scale_ode_train), axis=1))
    state12_scale_full = np.maximum(state12_scale_full, 1.0e-9)
    x2dot_scale_full = float(max(float(np.sqrt(np.mean(np.square(scale_x2dot_train)))), 1.0e-9))
    x3_scale = 100.0e-9
    known_pars = _load_known_pars(config)
    kan_runtime = prepare_kan_stage1_runtime(
        config.repo_root,
        stage_config=config,
        window_bundle=window_bundles[0],
        known_pars=known_pars,
        state12_scale=state12_scale_full,
        x2dot_scale=x2dot_scale_full,
        x3_scale=x3_scale,
    )

    return Stage1PlusLightContext(
        config=config,
        kan_runtime=kan_runtime,
        ode_data_full=ode_data_full,
        solution_table_full=solution_table_full,
        all_times=all_times,
        x2dot_all=x2dot_all,
        gain_force_reference_all=gain_force_reference_all,
        contact_all=contact_all,
        s_all=s_all,
        train_idx=train_idx,
        val_idx=val_idx,
        main_windows=main_windows,
        window_bundles=window_bundles,
        searches_total=searches_total,
        assignments=assignments,
        trial_parameters=trial_parameters,
        pending_assignments=pending_assignments,
        state12_scale_full=state12_scale_full,
        x2dot_scale_full=x2dot_scale_full,
        x3_scale=x3_scale,
        resume_reason=resume_reason,
    )


def run_trial_from_context(context: Stage1PlusLightContext, global_trial_id: int, *, model_factory: Any = None) -> dict[str, Any]:
    _ = model_factory
    cfg = context.config
    grid_trial = decode_grid_trial(global_trial_id, cfg.ks_node_count, cfg.cs_node_count, cfg.nn_seed_bank_size)
    return stage1pluslight_kan_random_trial(
        grid_trial.global_trial_id,
        grid_trial.ks0,
        grid_trial.cs0,
        grid_trial.ks_node_idx,
        grid_trial.cs_node_idx,
        grid_trial.nn_seed_bank_idx,
        context.kan_runtime,
    )


def rank_context_trials(context: Stage1PlusLightContext, *, topk: int | None = None, viable_only: bool = False) -> list[dict[str, Any]]:
    records = [rec for rec in context.trial_parameters if isinstance(rec, dict)]
    return rank_trial_records(records, topk=topk, viable_only=viable_only)


def _safe_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else float("nan")


def _record_train_loss(rec: dict[str, Any]) -> float:
    if not isinstance(rec, dict):
        return float("inf")
    value = rec.get("train_loss", rec.get("loss", float("inf")))
    try:
        return float(value)
    except Exception:
        return float("inf")


def summarize_trial_records(records: list[dict[str, Any]], *, final_topk: int) -> dict[str, Any]:
    ranked_all = rank_trial_records(records)
    ranked_viable = rank_trial_records(records, viable_only=True)
    topk = rank_trial_records(records, topk=final_topk)
    losses = [float(rec.get("loss", float("inf"))) for rec in ranked_all if isinstance(rec, dict)]
    return {
        "total_trials": int(len(records)),
        "viable_trials": int(sum(bool(rec.get("is_viable", False)) for rec in records if isinstance(rec, dict))),
        "failed_trials": int(sum(bool(rec.get("trial_failed", False)) for rec in records if isinstance(rec, dict))),
        "best_loss": float(topk[0]["loss"]) if topk else float("inf"),
        "mean_loss": float(_safe_mean(losses)),
        "top_trial_ids": [int(rec.get("params", {}).get("trial_id", 0)) for rec in topk],
        "ranked_topk": topk,
        "ranked_viable_topk": ranked_viable[:final_topk],
    }


def export_stage1pluslight_final(
    config: Stage1PlusLightConfig,
    trial_parameters: list[dict[str, Any]],
    *,
    searches_total: int,
    windows: list[WindowManifest] | None = None,
) -> dict[str, Any]:
    summary = summarize_trial_records(trial_parameters, final_topk=config.final_topk)
    payload = {
        "trial_parameters": trial_parameters,
        "ranked_topk": summary["ranked_topk"],
        "ranked_viable_topk": summary["ranked_viable_topk"],
        "searches_per_candidate": searches_total,
        "final_topk": config.final_topk,
        "stage1plus_partial": False,
        "stage1plus_complete": True,
        "stage1plus_kan_search": True,
        "stage1plus_search_mode": f"kscs_grid_kan_seedbank_{config.window_mode}",
        **_stage1plus_window_identity(config, windows),
        **_stage1plus_loss_scale_payload(),
        "stage1plus_shard_index": config.shard_index,
        "stage1plus_shard_count": config.shard_count,
        "stage1plus_grid_ks_nodes": config.ks_node_count,
        "stage1plus_grid_cs_nodes": config.cs_node_count,
        "stage1plus_grid_nn_seeds_per_node": config.nn_seed_bank_size,
        **_stage1plus_loss_scale_payload(),
        "bounds": {"ks": KS_BOUNDS, "cs": CS_BOUNDS},
        "true_values": {},
        "run_seed": config.run_seed,
        "use_multiple_shooting": config.use_multiple_shooting,
        "ms_group_size": config.ms_group_size,
        "ms_continuity_term": config.ms_continuity_term,
        "trial_opt_enabled": config.trial_opt_enabled,
        "trial_epochs": config.trial_epochs,
        "trial_lr": config.trial_lr,
        "ode_solver": config.ode_solver,
        "ode_fallback_solver": config.ode_fallback_solver,
        "ode_rtol": config.ode_rtol,
        "ode_atol": config.ode_atol,
        "ode_max_step": config.ode_max_step,
        "AFM05_true_side_available": False,
        "summary": summary,
        "rng_state": capture_rng_state(),
    }
    write_checkpoint_atomic(config.result_path, payload)

    summary_path = config.result_path.with_suffix(config.result_path.suffix + ".summary.json")
    topk_path = config.result_path.with_suffix(config.result_path.suffix + ".topk.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=_json_default)
    with open(topk_path, "w", encoding="utf-8") as fh:
        json.dump(summary["ranked_topk"], fh, indent=2, ensure_ascii=False, default=_json_default)
    return summary


def _load_payload(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return data if isinstance(data, dict) else None


def merge_shard_exports(config: Stage1PlusLightConfig, *, shard_count: int | None = None) -> dict[str, Any]:
    merged_records: list[dict[str, Any]] = []
    shard_paths: list[str] = []
    merged_windows: list[dict[str, Any]] | None = None
    shard_count = max(1, int(shard_count if shard_count is not None else config.shard_count))
    expected_total = total_trials(config.ks_node_count, config.cs_node_count, config.nn_seed_bank_size)
    if shard_count == 1:
        candidate_paths = [config.result_path]
    else:
        stem = config.result_stem if config.run_tag == "" else f"{config.result_stem}_{config.run_tag}"
        candidate_paths = [config.result_dir / f"{stem}_p{shard_idx}{config.result_ext}" for shard_idx in range(1, shard_count + 1)]

    for shard_path in candidate_paths:
        payload = _load_payload(shard_path)
        if not payload:
            continue
        matches, reason = _payload_matches_stage1plus_window(payload, config=config)
        if not matches:
            raise RuntimeError(f"Refusing to merge AFM05 shards with incompatible window identity: path={shard_path} reason={reason}")
        shard_windows = payload.get("stage1plus_windows")
        if isinstance(shard_windows, list):
            if merged_windows is None:
                merged_windows = shard_windows
            elif json.dumps(merged_windows, sort_keys=True, default=_json_default) != json.dumps(
                shard_windows,
                sort_keys=True,
                default=_json_default,
            ):
                raise RuntimeError(f"Refusing to merge AFM05 shards with different window manifests: path={shard_path}")
        shard_paths.append(str(shard_path))
        for rec in payload.get("trial_parameters", []):
            if isinstance(rec, dict):
                merged_records.append(rec)

    trial_ids = [trial_id_or_zero(rec) for rec in merged_records]
    if len(merged_records) != expected_total:
        raise RuntimeError(f"Refusing to merge incomplete AFM05 shards: records={len(merged_records)} expected={expected_total}")
    if any(tid <= 0 for tid in trial_ids):
        raise RuntimeError("Refusing to merge AFM05 shards: missing trial_id in one or more records")
    if len(set(trial_ids)) != len(trial_ids):
        raise RuntimeError("Refusing to merge AFM05 shards: duplicate trial_id records detected")

    summary = summarize_trial_records(merged_records, final_topk=config.final_topk)
    merged_payload = {
        "trial_parameters": merged_records,
        "ranked_topk": summary["ranked_topk"],
        "ranked_viable_topk": summary["ranked_viable_topk"],
        "searches_per_candidate": expected_total,
        "stage1plus_total_trials": expected_total,
        "stage1plus_merged_trial_count": len(merged_records),
        "final_topk": config.final_topk,
        "stage1plus_partial": False,
        "stage1plus_complete": True,
        "stage1plus_merged": True,
        "stage1plus_kan_search": True,
        "stage1plus_search_mode": f"kscs_grid_kan_seedbank_{config.window_mode}",
        "stage1plus_window_mode": str(config.window_mode),
        "stage1plus_window_pixel_tag": str(config.pixel_tag),
        "stage1plus_arch_window_us": float(config.arch_window_us),
        "stage1plus_window_sample_stride": int(config.window_sample_stride),
        "stage1plus_windows": list(merged_windows or []),
        "stage1plus_shard_count": shard_count,
        "stage1plus_grid_ks_nodes": config.ks_node_count,
        "stage1plus_grid_cs_nodes": config.cs_node_count,
        "stage1plus_grid_nn_seeds_per_node": config.nn_seed_bank_size,
        "bounds": {"ks": KS_BOUNDS, "cs": CS_BOUNDS},
        "true_values": {},
        "AFM05_true_side_available": False,
        "summary": summary,
        "source_shards": shard_paths,
        "rng_state": capture_rng_state(),
    }
    write_checkpoint_atomic(config.merged_result_path, merged_payload)

    summary_path = config.merged_result_path.with_suffix(config.merged_result_path.suffix + ".summary.json")
    topk_path = config.merged_result_path.with_suffix(config.merged_result_path.suffix + ".topk.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=_json_default)
    with open(topk_path, "w", encoding="utf-8") as fh:
        json.dump(summary["ranked_topk"], fh, indent=2, ensure_ascii=False, default=_json_default)
    return summary


def build_stage1pluslight_mechanistic_winners(config: Stage1PlusLightConfig) -> dict[str, Any]:
    payload = _load_payload(config.merged_result_path)
    if not payload:
        raise RuntimeError(f"AFM05 merged stage1pluslight result missing: {config.merged_result_path}")
    records = [rec for rec in payload.get("trial_parameters", []) if isinstance(rec, dict)]
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for rec in records:
        params = rec.get("params", {})
        key = (int(params.get("ks_node_idx", 0)), int(params.get("cs_node_idx", 0)))
        groups.setdefault(key, []).append(rec)

    rows: list[dict[str, Any]] = []
    for (ks_idx, cs_idx), recs in sorted(groups.items()):
        viable = [rec for rec in recs if bool(rec.get("is_viable", False)) and np.isfinite(_record_train_loss(rec))]
        ranked = sorted(recs, key=lambda rec: (not bool(rec.get("is_viable", False)), not np.isfinite(_record_train_loss(rec)), _record_train_loss(rec), int(rec.get("params", {}).get("trial_id", 0))))
        ranked_viable = sorted(viable, key=lambda rec: (_record_train_loss(rec), int(rec.get("params", {}).get("trial_id", 0))))
        best = ranked_viable[0] if ranked_viable else (ranked[0] if ranked else {})
        losses = [_record_train_loss(rec) for rec in viable]
        params = best.get("params", {}) if isinstance(best, dict) else {}
        viable_rate = float(len(viable)) / max(1, int(len(recs)))
        rows.append(
            {
                "ks_node_idx": int(ks_idx),
                "cs_node_idx": int(cs_idx),
                "node_label": f"node_{ks_idx}*{cs_idx}",
                "count": int(len(recs)),
                "viable_count": int(len(viable)),
                "viable_rate": float(viable_rate),
                "mean_loss_viable": float(_safe_mean(losses)),
                "best_loss": float(_record_train_loss(best)) if isinstance(best, dict) else float("inf"),
                "best_trial_id": int(params.get("trial_id", 0)),
                "ks": float(best.get("ks_hat", np.nan)) if isinstance(best, dict) else float("nan"),
                "cs": float(best.get("cs_hat", np.nan)) if isinstance(best, dict) else float("nan"),
            }
        )

    mean_rows = sorted(
        rows,
        key=lambda row: (
            not np.isfinite(row["mean_loss_viable"]),
            row["mean_loss_viable"],
            not np.isfinite(row["best_loss"]),
            row["best_loss"],
            -float(row["viable_rate"]),
            int(row["ks_node_idx"]),
            int(row["cs_node_idx"]),
        ),
    )
    best_rows = sorted(
        rows,
        key=lambda row: (
            not np.isfinite(row["best_loss"]),
            row["best_loss"],
            not np.isfinite(row["mean_loss_viable"]),
            row["mean_loss_viable"],
            -float(row["viable_rate"]),
            int(row["ks_node_idx"]),
            int(row["cs_node_idx"]),
        ),
    )

    def _write_report(paths: dict[str, Path], sorted_rows: list[dict[str, Any]], *, behavior: str) -> dict[str, Any]:
        best_row = sorted_rows[0] if sorted_rows else None
        with open(paths["txt"], "w", encoding="utf-8") as fh:
            fh.write(f"AFM05 stage1pluslight mechanistic winner report ({behavior})\n")
            fh.write("ranking_loss=train_loss\n")
            fh.write(
                "mean_behavior: rank by each mechanical node's mean viable train loss; "
                "best_behavior: rank by each mechanical node's best viable train loss.\n"
            )
            fh.write(f"source={config.merged_result_path}\n")
            fh.write(f"bounds_ks={KS_BOUNDS} bounds_cs={CS_BOUNDS}\n")
            fh.write("rank,node_label,count,viable_count,viable_rate,mean_loss_viable,best_loss,best_trial_id,ks,cs\n")
            for rank, row in enumerate(sorted_rows, start=1):
                fh.write(
                    f"{rank},{row['node_label']},{row['count']},{row['viable_count']},"
                    f"{row['viable_rate']:.12e},{row['mean_loss_viable']:.12e},"
                    f"{row['best_loss']:.12e},{row['best_trial_id']},"
                    f"{row['ks']:.12e},{row['cs']:.12e}\n"
                )
        with open(paths["json"], "w", encoding="utf-8") as fh:
            json.dump({"behavior": behavior, "rows": sorted_rows, "best": best_row}, fh, indent=2, ensure_ascii=False, default=_json_default)
        return {
            "behavior": behavior,
            "txt_path": str(paths["txt"]),
            "json_path": str(paths["json"]),
            "best": best_row,
        }

    legacy_paths = {
        "txt": config.log_dir / "afm05_stage1pluslight_mechanistic_winners.txt",
        "json": config.log_dir / "afm05_stage1pluslight_mechanistic_winners.json",
    }
    mean_paths = {
        "txt": config.log_dir / "afm05_stage1pluslight_mechanistic_winners_mean_behavior.txt",
        "json": config.log_dir / "afm05_stage1pluslight_mechanistic_winners_mean_behavior.json",
    }
    best_paths = {
        "txt": config.log_dir / "afm05_stage1pluslight_mechanistic_winners_best_behavior.txt",
        "json": config.log_dir / "afm05_stage1pluslight_mechanistic_winners_best_behavior.json",
    }
    mean_report = _write_report(mean_paths, mean_rows, behavior="mean_behavior")
    best_report = _write_report(best_paths, best_rows, behavior="best_behavior")
    legacy_report = _write_report(legacy_paths, mean_rows, behavior="mean_behavior")

    return {
        "expected_groups": config.ks_node_count * config.cs_node_count,
        "expected_seeds": config.nn_seed_bank_size,
        "mean_behavior": mean_report,
        "best_behavior": best_report,
        "legacy_mean_behavior": legacy_report,
        "txt_path": legacy_report["txt_path"],
        "json_path": legacy_report["json_path"],
        "best": mean_report["best"],
    }


def _log_top_trial_ranks(logger: Stage1Logger, ranked_topk: list[dict[str, Any]], *, max_rows: int) -> None:
    count = min(max(0, int(max_rows)), len(ranked_topk))
    logger.log(f"stage1plus top trial ranks | count={count}")
    for idx, rec in enumerate(ranked_topk[:count], start=1):
        params = rec.get("params", {}) if isinstance(rec, dict) else {}
        parts = rec.get("val_parts", {}) if isinstance(rec, dict) else {}
        logger.log(
            f"rank {idx:03d} | trial={int(params.get('trial_id', 0))} | "
            f"{params.get('node_label', 'node_unknown')} | seedbank={int(params.get('nn_seed_bank_idx', 0))} "
            f"initseed={int(params.get('nn_init_seed', 0))} | "
            f"train={float(rec.get('train_loss', np.inf)):.6e} loss={float(rec.get('loss', np.inf)):.6e} | "
            f"ks={float(rec.get('ks_hat', np.nan)):.6e} cs={float(rec.get('cs_hat', np.nan)):.6e} | "
            f"x1_rec={float(parts.get('x1_rec', np.nan)):.5f}% "
            f"x2_rec={float(parts.get('x2_rec', np.nan)):.5f}% "
            f"x2dot_rec={float(parts.get('x2dot_rec', np.nan)):.5f}% | "
            f"x3n_KAN=[{float(params.get('x3_norm_KAN_min', np.nan)):.3f},"
            f"{float(params.get('x3_norm_KAN_max', np.nan)):.3f}] "
            f"out1={float(params.get('x3_norm_KAN_outside_1_frac', np.nan)):.3f} "
            f"out2={float(params.get('x3_norm_KAN_outside_2_frac', np.nan)):.3f} | "
            f"viable={bool(rec.get('is_viable', False))}"
        )


def run_shard_mainloop(
    config: Stage1PlusLightConfig | None = None,
    *,
    auto_generate_dataset: bool = False,
    logger: Stage1Logger | None = None,
    model_factory: Any = None,
) -> tuple[Stage1PlusLightContext, dict[str, Any]]:
    if config is None:
        config = default_config()
    context = prepare_stage1pluslight_context(config, auto_generate_dataset=auto_generate_dataset)

    if logger is not None:
        logger.log(
            "context prepared | "
            f"shard={config.shard_index}/{config.shard_count} "
            f"window_mode={config.window_mode} "
            f"pixel_tag={config.pixel_tag} "
            f"sample_stride={config.window_sample_stride} "
            f"backend=kan_rs_{config.window_mode} "
            f"AFM05_true_side_available=False "
            f"F_actuation=experimental_in_air_reconstructed "
            f"contact_noncontact_weight=1:1 "
            f"resume_reason={context.resume_reason or 'fresh_start'} "
            f"completed={len(context.trial_parameters)} "
            f"pending={len(context.pending_assignments)}"
        )

    total_assigned = len(context.assignments)
    for loop_idx, global_trial_id in enumerate(context.pending_assignments, start=1):
        record = run_trial_from_context(context, global_trial_id, model_factory=model_factory)
        context.trial_parameters.append(record)

        if logger is not None:
            logger.log(format_trial_line(record))

        completed_now = len(context.trial_parameters)
        if logger is not None and (loop_idx == 1 or loop_idx % max(1, config.checkpoint_every) == 0 or loop_idx == len(context.pending_assignments)):
            ranked = rank_trial_records([rec for rec in context.trial_parameters if isinstance(rec, dict)], topk=1)
            best_loss = float(ranked[0]["loss"]) if ranked else None
            logger.log(format_progress_line(completed_now, total_assigned, pending=max(0, total_assigned - completed_now), best_loss=best_loss))

        if loop_idx % max(1, config.checkpoint_every) == 0:
            save_stage1pluslight_partial(config, context.trial_parameters, searches_total=context.searches_total, windows=context.main_windows)
            if logger is not None:
                logger.log(f"checkpoint saved | path={config.result_path}")

    save_stage1pluslight_partial(config, context.trial_parameters, searches_total=context.searches_total, windows=context.main_windows)
    if logger is not None:
        logger.log(f"final partial checkpoint saved | path={config.result_path}")

    summary = export_stage1pluslight_final(config, context.trial_parameters, searches_total=context.searches_total, windows=context.main_windows)
    if logger is not None:
        logger.log(
            "final export complete | "
            f"path={config.result_path} total_trials={summary['total_trials']} "
            f"viable={summary['viable_trials']} best_loss={summary['best_loss']:.6e}"
        )
    return context, summary


def launch_local_shards(
    config: Stage1PlusLightConfig | None = None,
    *,
    shard_count: int | None = None,
    auto_generate_dataset: bool = False,
) -> dict[str, Any]:
    if config is None:
        config = default_config()
    shard_count = max(1, int(shard_count if shard_count is not None else config.shard_count))
    driver_log_path = make_driver_log_path(config.log_dir, run_tag=config.run_tag)
    driver_logger = Stage1Logger(driver_log_path, echo=True)
    driver_logger.log(f"launcher start | shard_count={shard_count} result_dir={config.result_dir}")

    procs: list[tuple[int, subprocess.Popen[Any], Path, Any]] = []
    base_env = dict(os.environ)
    base_env["HNODECB_AFM05_STAGE1_SHARD_COUNT"] = str(shard_count)

    for shard_idx in range(1, shard_count + 1):
        shard_log_path = make_shard_log_path(config.log_dir, shard_index=shard_idx, shard_count=shard_count, run_tag=config.run_tag)
        env = dict(base_env)
        env["HNODECB_AFM05_STAGE1_SHARD_INDEX"] = str(shard_idx)
        env["HNODECB_AFM05_STAGE1_LOG_PATH"] = str(shard_log_path)
        cmd = [sys.executable, "-m", "AFM05.stage1pluslight.main"]
        if auto_generate_dataset:
            cmd.append("--auto-generate-dataset")
        shard_fh = open(shard_log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(config.repo_root), env=env, stdout=shard_fh, stderr=subprocess.STDOUT, text=True)
        procs.append((shard_idx, proc, shard_log_path, shard_fh))
        driver_logger.log(f"launched shard {shard_idx}/{shard_count} | pid={proc.pid} | log={shard_log_path}")

    exit_codes: dict[int, int] = {}
    try:
        for shard_idx, proc, shard_log_path, shard_fh in procs:
            code = proc.wait()
            exit_codes[shard_idx] = int(code)
            shard_fh.close()
            driver_logger.log(f"shard {shard_idx}/{shard_count} exited | code={code} | log={shard_log_path}")
    finally:
        for _shard_idx, proc, _shard_log_path, shard_fh in procs:
            if proc.poll() is None:
                proc.kill()
            if not shard_fh.closed:
                shard_fh.close()

    failed = {idx: code for idx, code in exit_codes.items() if code != 0}
    if failed:
        driver_logger.log(f"launcher failed | failed_shards={failed}")
        raise RuntimeError(f"AFM05 stage1pluslight launcher failed: {failed}")

    merged_summary = merge_shard_exports(config, shard_count=shard_count)
    driver_logger.log(
        "launcher merge complete | "
        f"merged_result={config.merged_result_path} best_loss={merged_summary['best_loss']:.6e}"
    )
    _log_top_trial_ranks(driver_logger, merged_summary["ranked_topk"], max_rows=config.final_topk)
    mech_report = build_stage1pluslight_mechanistic_winners(config)
    mean_report = mech_report.get("mean_behavior") if isinstance(mech_report, dict) else None
    best_report = mech_report.get("best_behavior") if isinstance(mech_report, dict) else None
    mean_best = mean_report.get("best") if isinstance(mean_report, dict) else None
    best_best = best_report.get("best") if isinstance(best_report, dict) else None
    if isinstance(mean_best, dict):
        driver_logger.log(
            "mechanistic winner report complete | behavior=mean_behavior | "
            f"txt={mean_report['txt_path']} | winner={mean_best['node_label']} "
            f"mean_loss_viable={float(mean_best['mean_loss_viable']):.6e} "
            f"best_trial={int(mean_best['best_trial_id'])}"
        )
    if isinstance(best_best, dict):
        driver_logger.log(
            "mechanistic winner report complete | behavior=best_behavior | "
            f"txt={best_report['txt_path']} | winner={best_best['node_label']} "
            f"best_loss={float(best_best['best_loss']):.6e} "
            f"best_trial={int(best_best['best_trial_id'])}"
        )
    return {
        "driver_log": str(driver_log_path),
        "exit_codes": exit_codes,
        "merged_summary": merged_summary,
        "mechanistic_winner_report": mech_report,
    }


__all__ = [
    "Stage1Logger",
    "Stage1PlusLightContext",
    "TrialModelBundle",
    "build_stage1pluslight_mechanistic_winners",
    "export_stage1pluslight_final",
    "format_progress_line",
    "format_trial_line",
    "launch_local_shards",
    "make_driver_log_path",
    "make_shard_log_path",
    "merge_shard_exports",
    "prepare_stage1pluslight_context",
    "rank_context_trials",
    "run_shard_mainloop",
    "run_trial_from_context",
    "save_stage1pluslight_partial",
    "summarize_trial_records",
]
