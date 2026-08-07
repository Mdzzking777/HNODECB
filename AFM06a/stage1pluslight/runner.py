"""AFM06a stage1pluslight control-plane framework."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from AFM06a.stage1pluslight.checkpoint import load_checkpoint, load_resume_trials, trial_id_or_zero, write_checkpoint_atomic
from AFM06a.stage1pluslight.config import Stage1PlusLightConfig, default_config
from AFM06a.stage1pluslight.data import (
    PreparedWindow,
    dataset_paths,
    load_dataset,
    make_all_train_mask,
    make_train_val_masks,
    prepare_window,
    select_window_sample_times,
)
from AFM06a.stage1pluslight.kan_rs_trial import TrialExecutor, run_seed_trial
from AFM06a.stage1pluslight.ranking import rank_trial_records
from AFM06a.stage1pluslight.seed_search import (
    compute_shard_assignments,
    decode_seed_trial,
    total_trials,
)


_RESUME_CONTRACT_FIELDS = (
    "error_level",
    "nn_seed_bank_size",
    "run_seed",
    "all_points_training",
    "val_stride",
    "val_offset",
    "window_start_s",
    "window_stop_s",
    "sample_stride",
    "sample_count",
    "sampling_policy",
    "noncontact_sampling_weight",
    "contact_sampling_weight",
    "transition_sampling_weight",
    "transition_half_width_s",
    "smoothness_loss_weight",
    "kan_width",
    "kan_grid",
    "kan_spline_k",
    "kan_base_fun",
    "kan_noise_scale",
    "kan_grid_eps",
    "kan_grid_range_lo",
    "kan_grid_range_hi",
    "adaptive_grid_enabled",
    "dtype",
    "device",
    "gain_enabled",
    "gain_learnable",
    "soft_mask_enabled",
    "soft_mask_trainable",
    "soft_mask_s0_a0",
    "soft_mask_s0_min_a0",
    "soft_mask_s0_max_a0",
    "soft_mask_alpha_a0",
    "soft_mask_alpha_min_a0",
    "soft_mask_alpha_max_a0",
    "state_guard_multiplier",
    "ode_step_budget",
    "normalizer_policy",
    "force_output_policy",
    "gain_policy",
    "soft_mask_policy",
    "loss_policy",
    "ode_method",
    "ode_rtol",
    "ode_atol",
)


def timestamp_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_shard_log_path(log_dir: Path, *, shard_index: int, shard_count: int, stamp: str) -> Path:
    return log_dir / f"afm06a_stage1pluslight_local_p{shard_index}of{shard_count}_{stamp}.txt"


def make_driver_log_path(log_dir: Path, *, stamp: str) -> Path:
    return log_dir / f"afm06a_stage1pluslight_driver_{stamp}.txt"


class Stage1Logger:
    def __init__(self, path: str | Path, *, echo: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = bool(echo)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.echo:
            print(line, flush=True)


@dataclass(frozen=True)
class Stage1PlusLightContext:
    config: Stage1PlusLightConfig
    window: PreparedWindow
    assigned_trial_ids: list[int]
    data_manifest: dict[str, dict[str, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_manifest(config: Stage1PlusLightConfig) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for name, path in dataset_paths(config.dataset_root, config.error_level).items():
        if name == "data_dir":
            continue
        resolved = path.resolve()
        manifest[name] = {
            "path": str(resolved),
            "size_bytes": int(resolved.stat().st_size),
            "sha256": _sha256_file(resolved),
        }
    return manifest


def readiness_report(config: Stage1PlusLightConfig | None = None) -> dict[str, Any]:
    cfg = default_config() if config is None else config
    dataset_status: dict[str, Any]
    try:
        loaded = load_dataset(cfg.dataset_root, cfg.error_level)
        sample_times, _sample_source_idx, _interpolated = select_window_sample_times(cfg, loaded)
        if cfg.all_points_training:
            train_idx, val_idx = make_all_train_mask(sample_times.size)
        else:
            train_idx, val_idx = make_train_val_masks(sample_times.size, cfg.val_stride, cfg.val_offset)
        dataset_status = {
            "ok": True,
            "state_shape": list(loaded.states.shape),
            "time_start_s": float(loaded.table["t"][0]),
            "time_stop_s": float(loaded.table["t"][-1]),
            "contact_fraction": float(loaded.table["contact"].mean()),
            "prepared_window": {
                "sample_count": int(sample_times.size),
                "train_count": int(train_idx.size),
                "validation_count": int(val_idx.size),
                "start_s": float(sample_times[0]),
                "stop_s": float(sample_times[-1]),
            },
        }
    except Exception as exc:
        dataset_status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "stage": "AFM06a stage1pluslight",
        "architecture": list(cfg.kan_width),
        "search_space": "NN seed bank only",
        "nn_seed_bank_size": int(cfg.nn_seed_bank_size),
        "state_names": ["x1", "x2"],
        "force_output": "Fts [N]",
        "kan_input_source": "observed x1 at every sampled time",
        "loss_target": "zero dynamics residual",
        "loss_terms": {
            "force_scale_normalized_dynamics_residual_weight": 1.0,
            "residual_smoothness_weight": cfg.smoothness_loss_weight,
        },
        "formal_window": {
            "name": "selected_training_window",
            "start_s": cfg.window_start_s,
            "stop_s": cfg.window_stop_s,
            "sample_stride": cfg.sample_stride,
            "sample_count": cfg.sample_count,
            "sampling_policy": cfg.sampling_policy,
            "sampling_density_transition_contact_noncontact": [
                cfg.transition_sampling_weight,
                cfg.contact_sampling_weight,
                cfg.noncontact_sampling_weight,
            ],
        },
        "train_validation_split": {
            "policy": (
                "all_points_training_no_validation"
                if cfg.all_points_training
                else "periodic_4_to_1_train_validation_split"
            ),
            "val_stride": int(cfg.val_stride),
            "val_offset": int(cfg.val_offset),
        },
        "kan": {
            "grid": cfg.kan_grid,
            "spline_k": cfg.kan_spline_k,
            "base_fun": cfg.kan_base_fun,
            "noise_scale": cfg.kan_noise_scale,
            "grid_eps": cfg.kan_grid_eps,
            "dtype": cfg.dtype,
            "device": cfg.device,
        },
        "policies": {
            "normalizer": cfg.normalizer_policy,
            "force_output": cfg.force_output_policy,
            "gain": cfg.gain_policy,
            "soft_mask": cfg.soft_mask_policy,
            "loss": cfg.loss_policy,
            "ode_method": cfg.ode_method,
            "ode_rtol": cfg.ode_rtol,
            "ode_atol": cfg.ode_atol,
        },
        "dataset": dataset_status,
        "ready": not cfg.readiness_gaps() and bool(dataset_status.get("ok")),
        "gaps": cfg.readiness_gaps(),
    }


def prepare_stage1pluslight_context(config: Stage1PlusLightConfig | None = None) -> Stage1PlusLightContext:
    cfg = default_config() if config is None else config
    cfg.validate_for_formal_run()
    window = prepare_window(cfg)
    assigned = compute_shard_assignments(
        total_trials(cfg.nn_seed_bank_size),
        cfg.shard_index,
        cfg.shard_count,
    )
    return Stage1PlusLightContext(
        config=cfg,
        window=window,
        assigned_trial_ids=assigned,
        data_manifest=_data_manifest(cfg),
    )


def _checkpoint_payload(context: Stage1PlusLightContext, records: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
    return {
        "stage": "AFM06a stage1pluslight",
        "schema_version": 1,
        "stage1plus_partial": not complete,
        "stage1plus_complete": bool(complete),
        "stage1plus_merged": context.config.shard_count == 1,
        "stage1plus_total_trials": int(context.config.nn_seed_bank_size),
        "shard_index": int(context.config.shard_index),
        "shard_count": int(context.config.shard_count),
        "kan_width": list(context.config.kan_width),
        "search_mode": "kan_seed_bank_only",
        "window_source_indices": context.window.source_idx.copy(),
        "train_idx": context.window.train_idx.copy(),
        "val_idx": context.window.val_idx.copy(),
        "times": context.window.times.copy(),
        "window_states": context.window.states.copy(),
        "window_x2dot": context.window.x2dot.copy(),
        "window_fts": context.window.fts.copy(),
        "window_bar_fts": context.window.bar_fts.copy(),
        "window_contact": context.window.contact.copy(),
        "initial_state": context.window.states[:, 0].copy(),
        "global_normalization": {
            "state_mean": context.window.global_state_mean.copy(),
            "state_scale": context.window.global_state_scale.copy(),
            "bar_fts_mean": float(context.window.global_bar_fts_mean),
            "bar_fts_scale": float(context.window.global_bar_fts_scale),
        },
        "data_manifest": dict(context.data_manifest),
        "config": asdict(context.config),
        "trial_parameters": list(records),
    }


def _validate_resume_grid(
    payload: dict[str, Any] | None,
    context: Stage1PlusLightContext,
    path: Path,
) -> None:
    if payload is None:
        return
    saved_fields = {
        "window_source_indices": payload.get("window_source_indices"),
        "train_idx": payload.get("train_idx"),
        "val_idx": payload.get("val_idx"),
    }
    expected_fields = {
        "window_source_indices": context.window.source_idx,
        "train_idx": context.window.train_idx,
        "val_idx": context.window.val_idx,
    }
    for name, saved in saved_fields.items():
        if saved is None:
            raise RuntimeError(
                f"AFM06a st1pl resume checkpoint lacks {name}: {path}. "
                "Archive or clear the old result before starting this configuration."
            )
        saved_values = tuple(int(value) for value in saved)
        expected_values = tuple(int(value) for value in expected_fields[name])
        if saved_values != expected_values:
            raise RuntimeError(
                f"AFM06a st1pl resume grid mismatch for {name}: {path}. "
                "The saved result belongs to a different window or sampling policy; "
                "archive or clear it before starting this configuration."
            )
    saved_config = payload.get("config")
    if not isinstance(saved_config, dict):
        raise RuntimeError(f"AFM06a st1pl resume checkpoint lacks its saved configuration: {path}")
    expected_config = asdict(context.config)
    for name in _RESUME_CONTRACT_FIELDS:
        if saved_config.get(name) != expected_config.get(name):
            raise RuntimeError(
                f"AFM06a st1pl resume contract mismatch for {name}: {path}. "
                "Archive or clear the old result before starting this configuration."
            )
    saved_manifest = payload.get("data_manifest")
    if saved_manifest != context.data_manifest:
        raise RuntimeError(
            f"AFM06a st1pl resume dataset manifest mismatch: {path}. "
            "The saved result belongs to different source data; archive or clear it."
        )


def run_shard_mainloop(
    config: Stage1PlusLightConfig | None = None,
    *,
    executor: TrialExecutor | None = None,
    logger: Stage1Logger | None = None,
) -> tuple[Stage1PlusLightContext, dict[str, Any]]:
    context = prepare_stage1pluslight_context(config)
    cfg = context.config
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if executor is None:
        raise RuntimeError("AFM06a formal trial executor is not implemented; use --check-only")
    if cfg.resume_enabled:
        _validate_resume_grid(load_checkpoint(cfg.result_path), context, cfg.result_path)
    resumed = load_resume_trials(cfg.result_path) if cfg.resume_enabled else {}
    records = [resumed[trial_id] for trial_id in context.assigned_trial_ids if trial_id in resumed]
    completed = set(resumed)
    for trial_id in context.assigned_trial_ids:
        if trial_id in completed:
            continue
        trial = decode_seed_trial(trial_id, cfg.nn_seed_bank_size, cfg.run_seed)
        record = run_seed_trial(trial, cfg, context.window, executor=executor)
        records.append(record)
        if logger is not None:
            logger.log(
                f"trial={trial_id}/{cfg.nn_seed_bank_size} seed={trial.nn_init_seed} "
                f"loss={float(record.get('loss', float('inf'))):.9e} viable={bool(record.get('is_viable', False))}"
            )
        if len(records) % cfg.checkpoint_every == 0:
            write_checkpoint_atomic(cfg.result_path, _checkpoint_payload(context, records, complete=False))
    write_checkpoint_atomic(cfg.result_path, _checkpoint_payload(context, records, complete=True))
    ranked = rank_trial_records(records, topk=cfg.final_topk)
    summary = {
        "total_trials": len(records),
        "viable_trials": sum(bool(record.get("is_viable", False)) for record in records),
        "ranked_topk": ranked,
        "best_loss": float(ranked[0]["loss"]) if ranked else float("inf"),
    }
    return context, summary


def merge_shard_exports(config: Stage1PlusLightConfig, *, shard_count: int) -> dict[str, Any]:
    records_by_id: dict[int, dict[str, Any]] = {}
    template: dict[str, Any] | None = None
    context = prepare_stage1pluslight_context(config)
    stem = config.result_stem if config.run_tag == "" else f"{config.result_stem}_{config.run_tag}"
    for shard_index in range(1, int(shard_count) + 1):
        path = config.result_dir / f"{stem}_p{shard_index}.pkl"
        payload = load_checkpoint(path)
        if payload is None:
            raise RuntimeError(f"missing AFM06a shard result: {path}")
        if not bool(payload.get("stage1plus_complete", False)):
            raise RuntimeError(f"incomplete AFM06a shard result: {path}")
        if int(payload.get("shard_index", 0)) != shard_index or int(payload.get("shard_count", 0)) != shard_count:
            raise RuntimeError(f"AFM06a shard identity mismatch: {path}")
        _validate_resume_grid(payload, context, path)
        template = dict(payload) if template is None else template
        for record in payload.get("trial_parameters", []):
            trial_id = trial_id_or_zero(record)
            if trial_id <= 0 or not isinstance(record, dict):
                raise RuntimeError(f"invalid AFM06a trial record in {path}")
            if trial_id in records_by_id:
                raise RuntimeError(f"duplicate AFM06a trial_id across shards: {trial_id}")
            records_by_id[trial_id] = record

    expected = int(config.nn_seed_bank_size)
    expected_ids = set(range(1, expected + 1))
    if set(records_by_id) != expected_ids:
        missing = sorted(expected_ids - set(records_by_id))
        extra = sorted(set(records_by_id) - expected_ids)
        raise RuntimeError(f"AFM06a merged coverage mismatch: missing={missing[:10]} extra={extra[:10]}")
    if template is None:
        raise RuntimeError("no AFM06a shard payloads were available for merging")

    records = [records_by_id[index] for index in range(1, expected + 1)]
    ranked = rank_trial_records(records, topk=config.final_topk)
    merged = dict(template)
    merged.update(
        {
            "stage1plus_partial": False,
            "stage1plus_complete": True,
            "stage1plus_merged": True,
            "stage1plus_total_trials": expected,
            "shard_index": 0,
            "shard_count": int(shard_count),
            "trial_parameters": records,
            "ranked_topk": ranked,
        }
    )
    write_checkpoint_atomic(config.merged_result_path, merged)
    return {
        "total_trials": len(records),
        "viable_trials": sum(bool(record.get("is_viable", False)) for record in records),
        "best_loss": float(ranked[0]["loss"]) if ranked else float("inf"),
        "result_path": str(config.merged_result_path),
    }


def launch_local_shards(
    config: Stage1PlusLightConfig | None = None,
    *,
    shard_count: int = 5,
) -> dict[str, Any]:
    cfg = default_config() if config is None else config
    shard_count = max(1, int(shard_count))
    cfg.validate_for_formal_run()
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_compact()
    driver = Stage1Logger(make_driver_log_path(cfg.log_dir, stamp=stamp))
    driver.log(f"launcher start | shards={shard_count} seeds={cfg.nn_seed_bank_size}")

    base_env = dict(os.environ)
    base_env["HNODECB_AFM06a_STAGE1_SHARD_COUNT"] = str(shard_count)
    base_env["HNODECB_AFM06a_STAGE1_NN_SEEDS"] = str(cfg.nn_seed_bank_size)
    base_env["HNODECB_AFM06a_STAGE1_RESULT_DIR"] = str(cfg.result_dir)
    base_env["HNODECB_AFM06a_STAGE1_LOG_DIR"] = str(cfg.log_dir)
    base_env["HNODECB_AFM06a_STAGE1_RESULT_STEM"] = str(cfg.result_stem)
    base_env["HNODECB_AFM06a_STAGE1_RUN_TAG"] = str(cfg.run_tag)
    base_env["HNODECB_AFM06a_STAGE1_LOG_ECHO"] = "0"

    processes: list[tuple[int, subprocess.Popen[Any], Any, Path]] = []
    try:
        for shard_index in range(1, shard_count + 1):
            shard_log = make_shard_log_path(
                cfg.log_dir,
                shard_index=shard_index,
                shard_count=shard_count,
                stamp=stamp,
            )
            console_log = shard_log.with_name(shard_log.stem + "_console.txt")
            env = dict(base_env)
            env["HNODECB_AFM06a_STAGE1_SHARD_INDEX"] = str(shard_index)
            env["HNODECB_AFM06a_STAGE1_LOG_PATH"] = str(shard_log)
            console_handle = console_log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-m", "AFM06a.stage1pluslight.main", "--run"],
                cwd=str(cfg.repo_root),
                env=env,
                stdout=console_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((shard_index, process, console_handle, console_log))
            driver.log(f"launched shard {shard_index}/{shard_count} | pid={process.pid} | log={shard_log}")

        exit_codes: dict[int, int] = {}
        for shard_index, process, console_handle, console_log in processes:
            code = int(process.wait())
            exit_codes[shard_index] = code
            console_handle.close()
            driver.log(f"shard {shard_index}/{shard_count} exited | code={code} | console={console_log}")
        failed = {index: code for index, code in exit_codes.items() if code != 0}
        if failed:
            raise RuntimeError(f"AFM06a stage1pluslight shards failed: {failed}")
        merged = merge_shard_exports(cfg, shard_count=shard_count)
        driver.log(
            f"launcher complete | total={merged['total_trials']} viable={merged['viable_trials']} "
            f"best_loss={merged['best_loss']:.9e} merged={merged['result_path']}"
        )
        return {"driver_log": str(driver.path), "exit_codes": exit_codes, "merged_summary": merged}
    finally:
        for _index, process, console_handle, _console_log in processes:
            if process.poll() is None:
                process.kill()
            if not console_handle.closed:
                console_handle.close()


def write_readiness_report(path: str | Path, config: Stage1PlusLightConfig | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(readiness_report(config), indent=2), encoding="utf-8")
    return target


__all__ = [
    "Stage1Logger",
    "Stage1PlusLightContext",
    "prepare_stage1pluslight_context",
    "readiness_report",
    "launch_local_shards",
    "make_driver_log_path",
    "make_shard_log_path",
    "merge_shard_exports",
    "run_shard_mainloop",
    "timestamp_compact",
    "write_readiness_report",
]
