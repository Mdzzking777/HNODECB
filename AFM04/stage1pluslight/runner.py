"""Compact runner entrypoints for AFM04 stage1pluslight."""

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

from AFM04.stage1pluslight.checkpoint import load_resume_trials, trial_id_or_zero, write_checkpoint_atomic
from AFM04.stage1pluslight.kan_rs_trial import prepare_kan_stage1_runtime, stage1pluslight_kan_random_trial
from AFM04.stage1pluslight.config import Stage1PlusLightConfig, default_config
from AFM04.stage1pluslight.data import load_dataset, make_train_val_masks, truncate_to_first_contact
from AFM04.stage1pluslight.grid import KS_BOUNDS, CS_BOUNDS, compute_shard_assignments, decode_grid_trial, total_trials
from AFM04.stage1pluslight.ranking import rank_trial_records
from AFM04.stage1pluslight.trial import TrialModelBundle
from AFM04.stage1pluslight.windows import WindowManifest, window_manifests


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
    stamp = timestamp_compact()
    return log_dir / f"afm04_stage1pluslight_local_p{shard_index}of{shard_count}{suffix}_{stamp}.txt"


def make_driver_log_path(log_dir: Path, *, run_tag: str = "") -> Path:
    suffix = "" if run_tag.strip() == "" else f"_{run_tag.strip()}"
    stamp = timestamp_compact()
    return log_dir / f"afm04_stage1pluslight_driver{suffix}_{stamp}.txt"


def format_trial_line(record: dict[str, Any]) -> str:
    params = record.get("params", {}) if isinstance(record, dict) else {}
    trial_id = int(params.get("trial_id", 0))
    node_label = str(params.get("node_label", "node_unknown"))
    nn_seed_bank_idx = int(params.get("nn_seed_bank_idx", 0))
    nn_init_seed = int(params.get("nn_init_seed", nn_seed_bank_idx))
    loss = float(record.get("loss", float("inf")))
    train_loss = float(record.get("train_loss", float("inf")))
    val_loss = float(record.get("val_loss", float("inf")))
    ks_hat = float(record.get("ks_hat", float("nan")))
    cs_hat = float(record.get("cs_hat", float("nan")))
    ks_err = float(record.get("ks_err_pct", float("nan")))
    cs_err = float(record.get("cs_err_pct", float("nan")))
    nn_err = float(record.get("val_nn_err", record.get("val_nn_err_start", float("nan"))))
    parts = record.get("val_parts", {}) if isinstance(record, dict) else {}
    x1_rec = float(parts.get("x1_rec", float("nan"))) if isinstance(parts, dict) else float("nan")
    x3_rec = float(parts.get("x3_rec", float("nan"))) if isinstance(parts, dict) else float("nan")
    viable = bool(record.get("is_viable", False))
    failed = bool(record.get("trial_failed", False))
    reason = str(record.get("failure_reason", "") or record.get("val_reason", "") or "")
    return (
        f"trial {trial_id} | {node_label} | seedbank={nn_seed_bank_idx} initseed={nn_init_seed} | "
        f"train={train_loss:.6e} val={val_loss:.6e} loss={loss:.6e} | "
        f"ks={ks_hat:.6e} ({ks_err:.2f}%) cs={cs_hat:.6e} ({cs_err:.2f}%) | "
        f"x1_rec={x1_rec:.2f}% x3_rec={x3_rec:.2f}% nn={nn_err:.2f}% | "
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


def save_stage1pluslight_partial(
    config: Stage1PlusLightConfig,
    trial_parameters: list[Any],
    *,
    searches_total: int,
) -> None:
    payload = {
        "study": None,
        "trial_parameters": trial_parameters,
        "warm_start_top": [],
        "bounds": {"ks": KS_BOUNDS, "cs": CS_BOUNDS},
        "true_values": {},
        "use_multiple_shooting": False,
        "use_l2_regularization": False,
        "val_stride": config.val_stride,
        "val_offset": config.val_offset,
        "error_level": config.error_level,
        "stage1_input_file": "",
        "stage1_input_topk": 0,
        "stage1plus_standalone": True,
        "stage1plus_joint_random_search": False,
        "stage1plus_joint_grid_search": False,
        "stage1plus_kan_search": True,
        "stage1plus_search_mode": "kscs_grid_kan_seedbank_w1",
        "stage1plus_grid_ks_nodes": config.ks_node_count,
        "stage1plus_grid_cs_nodes": config.cs_node_count,
        "stage1plus_grid_nn_seeds_per_node": config.nn_seed_bank_size,
        "stage1plus_window_mode": config.window_mode,
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
    }
    write_checkpoint_atomic(config.result_path, payload)


def _slice_with_mask(values: np.ndarray, idxs: np.ndarray, keep: np.ndarray) -> np.ndarray:
    selected = values[idxs]
    return selected[keep]


def prepare_stage1pluslight_context(
    config: Stage1PlusLightConfig | None = None,
    *,
    auto_generate_dataset: bool = False,
) -> Stage1PlusLightContext:
    if config is None:
        config = default_config()

    ode_data_full, solution_table_full = load_dataset(config.dataset_root, config.error_level, auto_generate=auto_generate_dataset)
    ode_data_full, solution_table_full = truncate_to_first_contact(ode_data_full, solution_table_full)

    all_times = np.asarray(solution_table_full["t"], dtype=float)
    x2dot_all = np.asarray(solution_table_full["x2dot"], dtype=float)
    contact_all = np.asarray(solution_table_full["contact"], dtype=int).astype(bool)
    s_all = np.asarray(solution_table_full["s"], dtype=float)
    x1_signal = np.asarray(solution_table_full["x1"], dtype=float)
    x3_signal = np.asarray(solution_table_full["x3"], dtype=float)

    train_idx, val_idx = make_train_val_masks(len(all_times), config.val_stride, config.val_offset)
    main_windows = window_manifests(all_times, contact_all, config.window_mode, config.arch_window_us, x1_signal=x1_signal)

    window_bundles: list[dict[str, Any]] = []
    for win in main_windows:
        local_train_keep = np.isin(win.idxs, train_idx)
        local_val_keep = np.isin(win.idxs, val_idx)
        window_bundles.append(
            {
                "role": win.role,
                "label": win.label,
                "start_idx": win.start_idx,
                "stop_idx": win.stop_idx,
                "len": win.length,
                "t_start": win.t_start,
                "t_stop": win.t_stop,
                "ode_train": ode_data_full[:, win.idxs][:, local_train_keep],
                "ode_val": ode_data_full[:, win.idxs][:, local_val_keep],
                "times_train": _slice_with_mask(all_times, win.idxs, local_train_keep),
                "times_val": _slice_with_mask(all_times, win.idxs, local_val_keep),
                "x2dot_train": _slice_with_mask(x2dot_all, win.idxs, local_train_keep),
                "x2dot_val": _slice_with_mask(x2dot_all, win.idxs, local_val_keep),
                "contact_train": _slice_with_mask(contact_all.astype(bool), win.idxs, local_train_keep),
                "contact_val": _slice_with_mask(contact_all.astype(bool), win.idxs, local_val_keep),
                "x3_t0_val": float(x3_signal[win.idxs[0]]),
            }
        )

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
        )

    completed_trial_ids = {trial_id_or_zero(rec) for rec in trial_parameters if trial_id_or_zero(rec) > 0}
    pending_assignments = [i for i in assignments if i not in completed_trial_ids]

    state12_scale_full = np.max(ode_data_full[0:2, :], axis=1) - np.min(ode_data_full[0:2, :], axis=1)
    state12_scale_full = np.maximum(state12_scale_full, 1.0e-9)
    x2dot_scale_full = float(max(np.max(x2dot_all) - np.min(x2dot_all), 1.0e-9))
    x3_scale = 100.0e-9
    kan_runtime = prepare_kan_stage1_runtime(config.repo_root, auto_generate_dataset=auto_generate_dataset)

    return Stage1PlusLightContext(
        config=config,
        kan_runtime=kan_runtime,
        ode_data_full=ode_data_full,
        solution_table_full=solution_table_full,
        all_times=all_times,
        x2dot_all=x2dot_all,
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


def run_trial_from_context(
    context: Stage1PlusLightContext,
    global_trial_id: int,
    *,
    model_factory: Any = None,
) -> dict[str, Any]:
    cfg = context.config
    grid_trial = decode_grid_trial(global_trial_id, cfg.ks_node_count, cfg.cs_node_count, cfg.nn_seed_bank_size)
    _ = model_factory  # kept for API compatibility; stage1pluslight now uses the KAN backend directly.
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
    finite = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(finite) / len(finite) if finite else float("nan")


def _json_default(obj: Any) -> Any:
    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


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
        "stage1plus_search_mode": "kscs_grid_kan_seedbank_w1",
        "stage1plus_window_mode": config.window_mode,
        "stage1plus_shard_index": config.shard_index,
        "stage1plus_shard_count": config.shard_count,
        "stage1plus_grid_ks_nodes": config.ks_node_count,
        "stage1plus_grid_cs_nodes": config.cs_node_count,
        "stage1plus_grid_nn_seeds_per_node": config.nn_seed_bank_size,
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
        "summary": summary,
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
    shard_count = max(1, int(shard_count if shard_count is not None else config.shard_count))
    if shard_count == 1:
        candidate_paths = [config.result_path]
    else:
        stem = config.result_stem if config.run_tag == "" else f"{config.result_stem}_{config.run_tag}"
        candidate_paths = [config.result_dir / f"{stem}_p{shard_idx}{config.result_ext}" for shard_idx in range(1, shard_count + 1)]

    for shard_path in candidate_paths:
        payload = _load_payload(shard_path)
        if not payload:
            continue
        shard_paths.append(str(shard_path))
        for rec in payload.get("trial_parameters", []):
            if isinstance(rec, dict):
                merged_records.append(rec)

    summary = summarize_trial_records(merged_records, final_topk=config.final_topk)
    merged_payload = {
        "trial_parameters": merged_records,
        "ranked_topk": summary["ranked_topk"],
        "ranked_viable_topk": summary["ranked_viable_topk"],
        "searches_per_candidate": len(merged_records),
        "final_topk": config.final_topk,
        "stage1plus_partial": False,
        "stage1plus_complete": True,
        "stage1plus_merged": True,
        "stage1plus_kan_search": True,
        "stage1plus_search_mode": "kscs_grid_kan_seedbank_w1",
        "stage1plus_shard_count": shard_count,
        "summary": summary,
        "source_shards": shard_paths,
    }
    write_checkpoint_atomic(config.merged_result_path, merged_payload)

    summary_path = config.merged_result_path.with_suffix(config.merged_result_path.suffix + ".summary.json")
    topk_path = config.merged_result_path.with_suffix(config.merged_result_path.suffix + ".topk.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=_json_default)
    with open(topk_path, "w", encoding="utf-8") as fh:
        json.dump(summary["ranked_topk"], fh, indent=2, ensure_ascii=False, default=_json_default)
    return summary


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
            f"backend=kan_rs_w1 "
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
            logger.log(
                format_progress_line(
                    completed_now,
                    total_assigned,
                    pending=max(0, total_assigned - completed_now),
                    best_loss=best_loss,
                )
            )

        if loop_idx % max(1, config.checkpoint_every) == 0:
            save_stage1pluslight_partial(config, context.trial_parameters, searches_total=context.searches_total)
            if logger is not None:
                logger.log(f"checkpoint saved | path={config.result_path}")

    save_stage1pluslight_partial(config, context.trial_parameters, searches_total=context.searches_total)
    if logger is not None:
        logger.log(f"final partial checkpoint saved | path={config.result_path}")

    summary = export_stage1pluslight_final(config, context.trial_parameters, searches_total=context.searches_total)
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
    base_env["HNODECB_AFM04_STAGE1_SHARD_COUNT"] = str(shard_count)

    for shard_idx in range(1, shard_count + 1):
        shard_log_path = make_shard_log_path(config.log_dir, shard_index=shard_idx, shard_count=shard_count, run_tag=config.run_tag)
        env = dict(base_env)
        env["HNODECB_AFM04_STAGE1_SHARD_INDEX"] = str(shard_idx)
        env["HNODECB_AFM04_STAGE1_LOG_PATH"] = str(shard_log_path)
        cmd = [sys.executable, "-m", "AFM04.stage1pluslight.main"]
        if auto_generate_dataset:
            cmd.append("--auto-generate-dataset")
        shard_fh = open(shard_log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(config.repo_root),
            env=env,
            stdout=shard_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
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
        raise RuntimeError(f"AFM04 stage1pluslight launcher failed: {failed}")

    merged_summary = merge_shard_exports(config, shard_count=shard_count)
    driver_logger.log(
        "launcher merge complete | "
        f"merged_result={config.merged_result_path} "
        f"best_loss={merged_summary['best_loss']:.6e}"
    )
    return {
        "driver_log": str(driver_log_path),
        "exit_codes": exit_codes,
        "merged_summary": merged_summary,
    }


__all__ = [
    "Stage1Logger",
    "Stage1PlusLightContext",
    "TrialModelBundle",
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
