"""Two-layer prerun promotion stage for AFM04 KAN full functional test."""

from __future__ import annotations

import copy
import json
import os
import pickle
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from AFM04.KAN_full_test.config import default_config
from AFM04.KAN_full_test.data import WindowSplit, prepare_data, x3dot_init_from_split, x3dot_scale_from_split
from AFM04.KAN_full_test.kan_backend import KANForceModule
from AFM04.KAN_full_test.losses import evaluate_split
from AFM04.KAN_full_test.random_search import (
    _rebuild_trial_state_dict,
    _row_is_valid,
    _row_sort_key,
    _window_file_tag,
    _window_title,
)
from AFM04.KAN_full_test.train import (
    _capture_grads,
    _grid_update_due,
    _grad_norm,
    _json_default,
    _make_grid_update_inputs,
    _make_optimizer,
    _metrics_row,
    _observable_grid_inputs_from_ode,
    _recent_val_plateau_stats,
    _restore_grads,
    _selected_window_indices,
    _set_optimizer_lr,
    _terminal_step_failure,
    _torch_dtype,
    _window_to_torch,
)


LAYER_A = "layer_a"
LAYER_B = "layer_b"

_STEP_RETRY_RE = re.compile(r"step-guard retry\s+(\d+)/(\w+)")
_STEP_ACCEPTED_RE = re.compile(r"accepted after\s+(\d+)\s+retry/reduction")
_STEP_REJECTED_RE = re.compile(r"(?:rejected update|retries exhausted) after\s+(\d+)\s+retries")
_EPOCH_RETRY_RE = re.compile(r"retry epoch\s+(\d+)\s+attempt\s+(\d+)/(\d+)")


def _timestamp_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_line(log, message: str, *, echo: bool = True) -> None:
    line = f"[{_timestamp_human()}] {message}"
    log.write(line + "\n")
    log.flush()
    if echo:
        print(line, flush=True)


def _canonical_layer_name(layer_name: str) -> str:
    value = str(layer_name).strip().lower()
    if value in ("a", "layera", "layer_a"):
        return LAYER_A
    if value in ("b", "layerb", "layer_b", "candidate", "candidates"):
        return LAYER_B
    raise ValueError(f"Unsupported KFT prerun layer name: {layer_name!r}")


def _layer_token(layer_name: str) -> str:
    return "layerA" if _canonical_layer_name(layer_name) == LAYER_A else "layerB"


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    tmp.replace(path)


def _save_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _save_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _read_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected KFT prerun pickle payload for {path}: {type(payload)!r}")
    return payload


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_root(cfg, trial: int) -> Path:
    return cfg.output_root / "candidate_runs" / "prerun" / f"trial_{int(trial):05d}"


def _candidate_history_path(candidate_root: Path, tag: str = "w0") -> Path:
    return candidate_root / "results" / f"kan_full_test_prerun_history_{tag}.json"


def _candidate_result_path(candidate_root: Path, tag: str = "w0") -> Path:
    return candidate_root / "results" / f"kan_full_test_prerun_result_{tag}.pt"


def _candidate_running_checkpoint_path(candidate_root: Path, tag: str = "w0") -> Path:
    return candidate_root / "checkpoints" / f"kan_full_test_prerun_checkpoint_{tag}.pt"


def _candidate_best_checkpoint_path(candidate_root: Path, tag: str = "w0") -> Path:
    return candidate_root / "checkpoints" / f"kan_full_test_prerun_best_{tag}.pt"


def _candidate_log_path(candidate_root: Path, layer_name: str, tag: str = "w0") -> Path:
    return (
        candidate_root
        / "logs"
        / "window_per_shard"
        / f"log2_04_step2a_kan_full_test_prerun_{_layer_token(layer_name)}_{tag}.txt"
    )


def _assigned_positions(n_items: int, shard_count: int, shard_index: int) -> list[int]:
    if shard_index < 1 or shard_index > shard_count:
        raise ValueError(f"invalid prerun shard_index={shard_index}, shard_count={shard_count}")
    return [idx for idx in range(1, n_items + 1) if ((idx - 1) % shard_count) + 1 == shard_index]


def _assignment_path(cfg, layer_name: str, shard_index: int) -> Path:
    return cfg.prerun_result_dir / "assignments" / f"kan_full_test_prerun_{_layer_token(layer_name)}_assignments_s{shard_index}.json"


def _shard_result_path(cfg, layer_name: str, shard_index: int) -> Path:
    return cfg.prerun_result_dir / f"kan_full_test_prerun_{_layer_token(layer_name)}_p{shard_index}.pkl"


def _merged_result_path(cfg, layer_name: str, tag: str) -> Path:
    layer = _canonical_layer_name(layer_name)
    if layer == LAYER_A:
        return cfg.prerun_result_dir / f"kan_full_test_prerun_top_trials_a_{tag}.pkl"
    return cfg.prerun_result_dir / f"kan_full_test_prerun_candidates_b_{tag}.pkl"


def _legacy_merged_result_path(cfg, layer_name: str, tag: str) -> Path:
    layer = _canonical_layer_name(layer_name)
    if layer == LAYER_A:
        return cfg.prerun_result_dir / f"kan_full_test_prerun_newrank_a_{tag}.pkl"
    return cfg.prerun_result_dir / f"kan_full_test_prerun_{tag}.pkl"


def _driver_log_path(cfg) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg.prerun_log_dir / f"kan_full_test_prerun_driver_{stamp}.txt"


def _shard_log_path(cfg, layer_name: str, shard_index: int) -> Path:
    return cfg.prerun_log_dir / f"kan_full_test_prerun_{_layer_token(layer_name)}_s{shard_index}.txt"


def _load_rs_ranked_rows(cfg, split: WindowSplit) -> tuple[Path, list[dict[str, Any]]]:
    tag = _window_file_tag(split.role)
    path = cfg.random_search_result_dir / f"kan_full_test_random_search_trials_{tag}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing random-search trial history for prerun: {path}")
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"Random-search trial history is not a list: {path}")
    valid_rows = [dict(row) for row in rows if _row_is_valid(row)]
    valid_rows.sort(key=_row_sort_key)
    for rank, row in enumerate(valid_rows, start=1):
        row["rs_rank"] = int(rank)
    return path, valid_rows


def _summarize_child_retry_log(log_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "child_log_path": str(log_path.resolve()),
        "child_log_exists": bool(log_path.is_file()),
        "step_retry_lines": 0,
        "step_retry_max": 0,
        "step_retry_denoms": [],
        "step_accepted_after_count": 0,
        "step_accepted_after_max": 0,
        "step_rejected_count": 0,
        "step_rejected_after_max": 0,
        "epoch_retry_lines": 0,
        "epoch_retry_max_attempt": 0,
        "epoch_retry_max_denom": 0,
    }
    if not log_path.is_file():
        return summary

    denoms: set[str] = set()
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            step_match = _STEP_RETRY_RE.search(line)
            if step_match:
                retry_idx = int(step_match.group(1))
                denoms.add(step_match.group(2))
                summary["step_retry_lines"] += 1
                summary["step_retry_max"] = max(int(summary["step_retry_max"]), retry_idx)

            accepted_match = _STEP_ACCEPTED_RE.search(line)
            if accepted_match:
                retry_count = int(accepted_match.group(1))
                summary["step_accepted_after_count"] += 1
                summary["step_accepted_after_max"] = max(int(summary["step_accepted_after_max"]), retry_count)

            rejected_match = _STEP_REJECTED_RE.search(line)
            if rejected_match:
                retry_count = int(rejected_match.group(1))
                summary["step_rejected_count"] += 1
                summary["step_rejected_after_max"] = max(int(summary["step_rejected_after_max"]), retry_count)

            epoch_match = _EPOCH_RETRY_RE.search(line)
            if epoch_match:
                attempt = int(epoch_match.group(2))
                denom = int(epoch_match.group(3))
                summary["epoch_retry_lines"] += 1
                summary["epoch_retry_max_attempt"] = max(int(summary["epoch_retry_max_attempt"]), attempt)
                summary["epoch_retry_max_denom"] = max(int(summary["epoch_retry_max_denom"]), denom)

    summary["step_retry_denoms"] = sorted(denoms, key=str)
    return summary


def _retry_summary_line(summary: dict[str, Any]) -> str:
    denoms = ",".join(str(x) for x in summary.get("step_retry_denoms", [])) or "none"
    return (
        f"child_log={summary.get('child_log_path', '')} | "
        f"exists={summary.get('child_log_exists', False)} | "
        f"step_retry_lines={int(summary.get('step_retry_lines', 0))} | "
        f"step_retry_max={int(summary.get('step_retry_max', 0))}/{denoms} | "
        f"accepted_after_max={int(summary.get('step_accepted_after_max', 0))} | "
        f"rejected_count={int(summary.get('step_rejected_count', 0))} | "
        f"rejected_after_max={int(summary.get('step_rejected_after_max', 0))} | "
        f"epoch_retry_lines={int(summary.get('epoch_retry_lines', 0))} | "
        f"epoch_retry_max={int(summary.get('epoch_retry_max_attempt', 0))}/"
        f"{int(summary.get('epoch_retry_max_denom', 0))}"
    )


def _state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        out[key] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    return out


def _build_model_from_state(
    *,
    cfg,
    prepared,
    dtype: torch.dtype,
    seed: int,
    state_dict: dict[str, Any],
) -> KANForceModule:
    train_wpred_enabled = bool(getattr(cfg, "train_wpred_enabled", getattr(cfg, "wpred_enabled", False)))
    model = KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=seed,
        width=cfg.width,
        grid=cfg.grid,
        spline_k=cfg.spline_k,
        base_fun=cfg.base_fun,
        symbolic_enabled=cfg.symbolic_enabled,
        auto_save=cfg.auto_save,
        noise_scale=cfg.noise_scale,
        affine_trainable=cfg.affine_trainable,
        grid_eps=cfg.grid_eps,
        grid_range=(cfg.grid_range_lo, cfg.grid_range_hi),
        dist=float(prepared.known_pars[6]),
        a0=float(prepared.known_pars[9]),
        wpred_enabled=train_wpred_enabled,
        wpred_eps=cfg.wpred_eps,
        gnn_learnable=cfg.gnn_learnable,
        soft_mask_enabled=cfg.soft_mask_enabled,
        soft_mask_trainable=cfg.soft_mask_trainable,
        soft_mask_s0_a0=cfg.soft_mask_s0_a0,
        soft_mask_s0_min_a0=cfg.soft_mask_s0_min_a0,
        soft_mask_s0_max_a0=cfg.soft_mask_s0_max_a0,
        soft_mask_alpha_a0=cfg.soft_mask_alpha_a0,
        soft_mask_alpha_min_a0=cfg.soft_mask_alpha_min_a0,
        soft_mask_alpha_max_a0=cfg.soft_mask_alpha_max_a0,
        x3dot_input_enabled=cfg.x3dot_input_enabled,
        x3dot_init_trainable=cfg.x3dot_init_trainable,
        x3dot_init_value=(
            x3dot_init_from_split(
                split,
                policy=cfg.x3dot_init_policy,
                fallback=cfg.x3dot_init_value,
            )
            if cfg.x3dot_input_enabled
            else cfg.x3dot_init_value
        ),
        x3dot_scale=x3dot_scale_from_split(
            split,
            configured_scale=cfg.x3dot_scale,
            scale_mode=cfg.x3dot_scale_mode,
            a0=float(prepared.known_pars[9]),
            window_span=float(split.t_stop - split.t_start),
        ),
        x3dot_lag_detach=cfg.x3dot_lag_detach,
        device=cfg.device,
        dtype=dtype,
    ).to(cfg.device)
    model.load_state_dict(state_dict)
    return model


def _eval_metrics(
    *,
    cfg,
    prepared,
    tensors: dict[str, torch.Tensor],
    mech_true_t: torch.Tensor,
    model: KANForceModule,
    split_name: str,
) -> tuple[float, dict[str, float]]:
    loss_indices = tensors["train_idx"] if split_name == "train" else tensors["val_idx"]
    total, parts, _traj = evaluate_split(
        force_module=model,
        known_pars=prepared.known_pars,
        mech_true=mech_true_t,
        ode_true=tensors["ode_full"],
        x2dot_true=tensors["x2dot_full"],
        contact_mask=tensors["contact_full"],
        times=tensors["times_full"],
        ode_method=cfg.ode_method,
        ode_rtol=cfg.ode_rtol,
        ode_atol=cfg.ode_atol,
        eta_star_true=prepared.eta_star_true,
        loss_indices=loss_indices,
    )
    return float(total.detach()), _metrics_row(total, parts)


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "loss":
            continue
        out[f"{prefix}_{key}"] = float(value)
    return out


def _history_best_train(history: list[dict[str, Any]]) -> tuple[float, float, int]:
    best_train = float("inf")
    best_val = float("inf")
    best_epoch = 0
    for row in history:
        try:
            train_loss = float(row.get("train_loss", float("inf")))
            val_loss = float(row.get("val_loss", float("inf")))
        except Exception:
            continue
        if np.isfinite(train_loss) and (train_loss, int(row.get("epoch", 10**9))) < (best_train, best_epoch or 10**9):
            best_train = train_loss
            best_val = val_loss
            best_epoch = int(row.get("epoch", 0))
    return best_train, best_val, best_epoch


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, int]:
    return (float(row.get("best_train_loss", float("inf"))), int(row.get("trial", -1)))


def _safe_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _format_loss(value: Any) -> str:
    try:
        loss = float(value)
    except Exception:
        return "nan"
    return f"{loss:.6e}" if np.isfinite(loss) else "nan"


def _format_percent(value: Any) -> str:
    try:
        pct = float(value)
    except Exception:
        return "nan%"
    return f"{pct:.2f}%" if np.isfinite(pct) else "nan%"


def _flow_metric_tail(row: dict[str, Any]) -> str:
    return (
        f"| trial={_safe_int(row, 'trial')} "
        f"| seed={_safe_int(row, 'seed')} "
        f"| epochs_total={_safe_int(row, 'epochs_completed')} "
        f"epochs_layer={_safe_int(row, 'current_layer_epochs')} "
        f"| final_train={_format_loss(row.get('final_train_loss'))} "
        f"| final_val={_format_loss(row.get('final_val_loss'))} "
        f"| best_train={_format_loss(row.get('best_train_loss'))} @ e{_safe_int(row, 'best_epoch')} "
        f"| best_val={_format_loss(row.get('best_val_loss'))} "
        f"| train_x1={_format_percent(row.get('train_x1_rec'))} "
        f"| train_x3={_format_percent(row.get('train_x3_rec'))} "
        f"| train_nn={_format_percent(row.get('train_fts_rollout_rec'))} "
        f"| val_x1={_format_percent(row.get('val_x1_rec'))} "
        f"| val_x3={_format_percent(row.get('val_x3_rec'))} "
        f"| val_nn={_format_percent(row.get('val_fts_rollout_rec'))} "
        f"| step_retry_lines={_safe_int(row, 'step_retry_lines')} "
        f"| epoch_retry_lines={_safe_int(row, 'epoch_retry_lines')}"
    )


def _log_layer_a_promotion_flow(
    log,
    cfg,
    *,
    merged_a: dict[str, Any],
    layer_a_topk: int,
) -> None:
    all_records = [dict(row) for row in merged_a.get("all_records", []) if isinstance(row, dict)]
    ranked_records = [dict(row) for row in merged_a.get("candidate_records", []) if isinstance(row, dict)]
    rank_by_trial = {_safe_int(row, "trial"): row for row in ranked_records}
    records = all_records or ranked_records
    records.sort(key=lambda row: (_safe_int(row, "rs_rank", 10**9), _safe_int(row, "trial", 10**9)))

    _log_line(log, "KFT prerun layer A rerank summary:")
    _log_line(
        log,
        f"  all_input_trials={len(records)} expected={cfg.prerun_input_topk} "
        f"promote_top={layer_a_topk} metric=best_train_loss",
    )
    for row in records:
        trial = _safe_int(row, "trial")
        ranked = rank_by_trial.get(trial)
        display = ranked or row
        rs_rank = _safe_int(display, "rs_rank")
        failed = bool(row.get("failed", False)) or ranked is None
        if failed:
            _log_line(
                log,
                f"RS rank {rs_rank} -> NewRank_A FAILED "
                f"{_flow_metric_tail(display)}",
            )
            _log_line(log, f"  not promoted | reason={row.get('failure_reason', '')}")
            continue

        layer_a_rank = _safe_int(display, "layer_a_winner_rank")
        promoted = layer_a_rank > 0 and layer_a_rank <= int(layer_a_topk)
        next_step = "yes" if promoted else "no"
        _log_line(
            log,
            f"RS rank {rs_rank} -> NewRank_A {layer_a_rank} "
            f"{_flow_metric_tail(display)} | promote_to_layerB={next_step}",
        )


def _log_layer_b_promotion_flow(log, *, merged_b: dict[str, Any]) -> None:
    all_records = [dict(row) for row in merged_b.get("all_records", []) if isinstance(row, dict)]
    ranked_records = [dict(row) for row in merged_b.get("candidate_records", []) if isinstance(row, dict)]
    rank_by_trial = {_safe_int(row, "trial"): row for row in ranked_records}
    records = all_records or ranked_records

    def sort_key(row: dict[str, Any]) -> tuple[int, float, int, int]:
        ranked = rank_by_trial.get(_safe_int(row, "trial"))
        candidate_rank = _safe_int(ranked or row, "candidate_rank", 10**9)
        return (
            candidate_rank,
            float((ranked or row).get("best_train_loss", float("inf"))),
            _safe_int(row, "layer_a_winner_rank", 10**9),
            _safe_int(row, "trial", 10**9),
        )

    records.sort(key=sort_key)
    _log_line(log, "KFT prerun final candidate summary:")
    _log_line(log, f"  candidates={len(records)} metric=best_train_loss")
    for row in records:
        trial = _safe_int(row, "trial")
        ranked = rank_by_trial.get(trial)
        display = ranked or row
        rs_rank = _safe_int(display, "rs_rank")
        layer_a_rank = _safe_int(display, "layer_a_winner_rank")
        failed = bool(row.get("failed", False)) or ranked is None
        if failed:
            _log_line(
                log,
                f"RS rank {rs_rank} -> NewRank_A {layer_a_rank} -> candidate FAILED "
                f"{_flow_metric_tail(display)} | reason={row.get('failure_reason', '')}",
            )
            continue

        candidate_rank = _safe_int(display, "candidate_rank")
        _log_line(
            log,
            f"RS rank {rs_rank} -> NewRank_A {layer_a_rank} -> candidate {candidate_rank} "
            f"{_flow_metric_tail(display)}",
        )


def _record_from_run(
    *,
    layer_name: str,
    source: dict[str, Any],
    candidate_root: Path,
    history: list[dict[str, Any]],
    best_train_loss: float,
    best_val_loss: float,
    best_epoch: int,
    failed: bool,
    failure_reason: str,
    dt_total_sec: float,
    log_path: Path,
    tag: str,
) -> dict[str, Any]:
    canonical_layer = _canonical_layer_name(layer_name)
    trial = int(source["trial"])
    rs_rank = int(source.get("rs_rank", 0))
    layer_a_winner_rank = int(source.get("layer_a_winner_rank", 0))
    final = history[-1] if history else {}
    layer_tag = "A" if canonical_layer == LAYER_A else "B"
    current_layer_epochs = len([row for row in history if str(row.get("layer_tag", "")) == layer_tag])
    retry_summary = _summarize_child_retry_log(log_path)
    record = {
        "layer_name": canonical_layer,
        "layer": layer_tag,
        "trial": int(trial),
        "seed": int(source.get("seed", 0)),
        "rs_rank": int(rs_rank),
        "layer_a_winner_rank": int(layer_a_winner_rank),
        "epochs_completed": int(len(history)),
        "current_layer_epochs": int(current_layer_epochs),
        "failed": bool(failed),
        "failure_reason": failure_reason,
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(final.get("train_loss", float("inf"))),
        "final_val_loss": float(final.get("val_loss", float("inf"))),
        "candidate_metric": "best_train_loss",
        "candidate_loss": float(best_train_loss),
        "candidate_root": str(candidate_root),
        "result_path": str(_candidate_result_path(candidate_root, tag)),
        "running_checkpoint_path": str(_candidate_running_checkpoint_path(candidate_root, tag)),
        "best_checkpoint_path": str(_candidate_best_checkpoint_path(candidate_root, tag)),
        "history_path": str(_candidate_history_path(candidate_root, tag)),
        "log_path": str(log_path.resolve()),
        "history": history,
        "retry_summary": retry_summary,
        "step_retry_lines": int(retry_summary["step_retry_lines"]),
        "step_retry_max": int(retry_summary["step_retry_max"]),
        "step_accepted_after_max": int(retry_summary["step_accepted_after_max"]),
        "step_rejected_count": int(retry_summary["step_rejected_count"]),
        "step_rejected_after_max": int(retry_summary["step_rejected_after_max"]),
        "epoch_retry_lines": int(retry_summary["epoch_retry_lines"]),
        "epoch_retry_max_attempt": int(retry_summary["epoch_retry_max_attempt"]),
        "epoch_retry_max_denom": int(retry_summary["epoch_retry_max_denom"]),
        "dt_total_sec": float(dt_total_sec),
    }
    for key in (
        "train_x1_rec",
        "train_x3_rec",
        "train_fts_rollout_rec",
        "val_x1_rec",
        "val_x3_rec",
        "val_fts_rollout_rec",
        "g_nn",
        "grad_norm",
        "grad_norm_raw",
        "lr",
        "epoch_retry_count",
        "step_retry_count",
    ):
        if key in final:
            record[key] = final[key]
    return record


def _run_candidate_to_epoch(
    *,
    cfg,
    prepared,
    split: WindowSplit,
    tensors: dict[str, torch.Tensor],
    mech_true_t: torch.Tensor,
    source: dict[str, Any],
    layer_name: str,
    target_total_epochs: int,
    log,
    resume_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    dtype = _torch_dtype(cfg.dtype)
    canonical_layer = _canonical_layer_name(layer_name)
    layer_tag = "A" if canonical_layer == LAYER_A else "B"
    trial = int(source["trial"])
    seed = int(source.get("seed", cfg.seed + trial))
    tag = _window_file_tag(split.role)
    candidate_root = _candidate_root(cfg, trial)
    candidate_root.mkdir(parents=True, exist_ok=True)
    history_path = _candidate_history_path(candidate_root, tag)
    result_path = _candidate_result_path(candidate_root, tag)
    running_checkpoint_path = _candidate_running_checkpoint_path(candidate_root, tag)
    best_checkpoint_path = _candidate_best_checkpoint_path(candidate_root, tag)
    candidate_log_path = _candidate_log_path(candidate_root, canonical_layer, tag)

    if resume_checkpoint_path is not None:
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing prerun running checkpoint for seamless resume: {resume_checkpoint_path}")
        resume = torch.load(resume_checkpoint_path, map_location=cfg.device, weights_only=False)
        start_state_dict = resume["state_dict"]
        optimizer_state_dict = resume.get("optimizer_state_dict")
        history = list(resume.get("history", []))
        start_epoch = int(resume.get("epoch", len(history)))
        cached_best_train = float(resume.get("best_train_loss", float("inf")))
        cached_best_val = float(resume.get("best_val_loss", float("inf")))
        cached_best_epoch = int(resume.get("best_epoch", 0))
        lr = float(resume.get("lr", float(cfg.lr)))
        grad_ema = float(resume.get("grad_ema", float(cfg.lr_target_init)))
        grad_target = float(resume.get("grad_target", float(cfg.lr_target_init)))
        recent_losses_raw = resume.get("recent_losses", [])
        recent_losses = [float(x) for x in recent_losses_raw] if isinstance(recent_losses_raw, list) else []
        _log_line(
            log,
            f"layer B seamless resume | trial={trial} rs_rank={int(source.get('rs_rank', 0))} "
            f"from_epoch={start_epoch} target_total_epochs={target_total_epochs} "
            f"checkpoint={resume_checkpoint_path}",
        )
    else:
        if candidate_root.exists():
            for child in candidate_root.iterdir():
                if child.name == "logs":
                    continue
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
        candidate_root.mkdir(parents=True, exist_ok=True)
        start_state_dict = _rebuild_trial_state_dict(
            cfg=cfg,
            prepared=prepared,
            split=split,
            seed=seed,
            dtype=dtype,
        )
        optimizer_state_dict = None
        history = []
        start_epoch = 0
        cached_best_train = float("inf")
        cached_best_val = float("inf")
        cached_best_epoch = 0
        lr = float(cfg.lr)
        grad_ema = float(cfg.lr_target_init)
        grad_target = float(cfg.lr_target_init)
        recent_losses = []
        _log_line(
            log,
            f"layer A start | trial={trial} rs_rank={int(source.get('rs_rank', 0))} "
            f"target_total_epochs={target_total_epochs}",
        )

    model = _build_model_from_state(
        cfg=cfg,
        prepared=prepared,
        dtype=dtype,
        seed=seed,
        state_dict=start_state_dict,
    )
    optimizer, optimizer_name = _make_optimizer(cfg, model)
    _set_optimizer_lr(optimizer, lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        _set_optimizer_lr(optimizer, lr)
    observable_grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=tensors["ode_train"],
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        cfg=cfg,
    )

    if len(history) >= int(target_total_epochs):
        best_train_loss, best_val_loss, best_epoch = _history_best_train(history)
        if np.isfinite(cached_best_train) and cached_best_train <= best_train_loss:
            best_train_loss, best_val_loss, best_epoch = cached_best_train, cached_best_val, cached_best_epoch
        return _record_from_run(
            layer_name=canonical_layer,
            source=source,
            candidate_root=candidate_root,
            history=history,
            best_train_loss=best_train_loss,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            failed=False,
            failure_reason="",
            dt_total_sec=0.0,
            log_path=candidate_log_path,
            tag=tag,
        )

    best_train_loss, best_val_loss, best_epoch = _history_best_train(history)
    if np.isfinite(cached_best_train) and cached_best_train <= best_train_loss:
        best_train_loss, best_val_loss, best_epoch = cached_best_train, cached_best_val, cached_best_epoch
    if cfg.plateau_early_stop and not recent_losses:
        recent_losses = [
            float(row["train_loss"])
            for row in history[-int(cfg.plateau_window):]
            if isinstance(row, dict) and "train_loss" in row
        ]
    failure_reason = ""
    failed = False
    wall_t0 = perf_counter()
    _log_line(
        log,
        f"layer {layer_tag} optimizer | trial={trial} optimizer={optimizer_name.upper()} lr={lr:.2e}",
    )

    for epoch in range(start_epoch + 1, int(target_total_epochs) + 1):
        epoch_t0 = perf_counter()
        epoch_start_state = copy.deepcopy(model.state_dict())
        epoch_start_opt_state = copy.deepcopy(optimizer.state_dict())
        epoch_attempt_max = max(1, int(cfg.epoch_retry_max) + 1)
        epoch_attempt = 0
        recovered = False
        train_total: torch.Tensor | None = None
        train_parts = None
        val_total: torch.Tensor | None = None
        val_parts = None
        grad_norm = float("nan")
        grad_norm_raw = float("nan")
        step_retry_count = 0
        step_accept_reason = ""
        epoch_fail_reason = ""
        grid_update_sec_epoch = 0.0
        grid_update_runs_epoch = 0

        while epoch_attempt < epoch_attempt_max:
            epoch_attempt += 1
            model.load_state_dict(copy.deepcopy(epoch_start_state))
            optimizer.load_state_dict(copy.deepcopy(epoch_start_opt_state))
            _set_optimizer_lr(optimizer, lr)
            model.train()

            if (
                cfg.adaptive_grid_enabled
                and _grid_update_due(epoch - 1, cfg.grid_update_num, cfg.start_grid_update_step, cfg.stop_grid_update_step)
            ):
                try:
                    grid_t0 = perf_counter()
                    with torch.no_grad():
                        grid_source = "raw_current_rollout_pred_x3_qpre"
                        grid_inputs, grid_meta = _make_grid_update_inputs(
                            cfg=cfg,
                            base_inputs=observable_grid_inputs,
                            known_pars=prepared.known_pars,
                            model=model,
                            tensors=tensors,
                            mech_true=mech_true_t,
                            ode_method=cfg.ode_method,
                            ode_rtol=cfg.ode_rtol,
                            ode_atol=cfg.ode_atol,
                        )
                        grid_source = str(grid_meta.get("source", grid_source))
                        model.update_grid_from_inputs(grid_inputs)
                    grid_update_sec_epoch += float(perf_counter() - grid_t0)
                    grid_update_runs_epoch += 1
                    _log_line(
                        log,
                        "grid update | "
                        f"epoch={epoch} | source={grid_source} | "
                        f"samples={int(grid_meta['total_samples'])} | "
                        f"mean_w_pred={grid_meta['mean_w_pred']:.3f} "
                        f"max_w_pred={grid_meta['max_w_pred']:.3f}",
                    )
                except Exception as err:  # noqa: BLE001 - same retry semantics as KFT GBO.
                    epoch_fail_reason = f"grid_update_exception:{err}"

            if epoch_fail_reason != "":
                if epoch_attempt < epoch_attempt_max:
                    lr = max(lr * cfg.epoch_retry_lr_factor, cfg.epoch_retry_lr_floor)
                    _log_line(
                        log,
                        f"retry epoch {epoch} attempt {epoch_attempt}/{epoch_attempt_max} "
                        f"-- reason={epoch_fail_reason} | lr={lr:.3e}",
                    )
                    continue
                failure_reason = f"epoch_retry_exhausted:{epoch_fail_reason}"
                break

            optimizer.zero_grad()
            train_traj: torch.Tensor | None = None
            accepted_train_traj: torch.Tensor | None = None
            try:
                train_total, train_parts, train_traj = evaluate_split(
                    force_module=model,
                    known_pars=prepared.known_pars,
                    mech_true=mech_true_t,
                    ode_true=tensors["ode_full"],
                    x2dot_true=tensors["x2dot_full"],
                    contact_mask=tensors["contact_full"],
                    times=tensors["times_full"],
                    ode_method=cfg.ode_method,
                    ode_rtol=cfg.ode_rtol,
                    ode_atol=cfg.ode_atol,
                    eta_star_true=prepared.eta_star_true,
                    loss_indices=tensors["train_idx"],
                )
                if not torch.isfinite(train_total):
                    epoch_fail_reason = "train_loss_nonfinite"
                else:
                    train_total.backward()
                    grad_norm_raw = _grad_norm(model)
                    if not np.isfinite(grad_norm_raw):
                        epoch_fail_reason = "grad_norm_nonfinite"
            except Exception as err:  # noqa: BLE001 - same retry semantics as KFT GBO.
                epoch_fail_reason = f"exception:{err}"

            if epoch_fail_reason == "":
                grad_norm = grad_norm_raw
                if not np.isfinite(grad_norm):
                    epoch_fail_reason = "grad_scaled_norm_nonfinite"

            if epoch_fail_reason != "":
                if epoch_attempt < epoch_attempt_max:
                    lr = max(lr * cfg.epoch_retry_lr_factor, cfg.epoch_retry_lr_floor)
                    _log_line(
                        log,
                        f"retry epoch {epoch} attempt {epoch_attempt}/{epoch_attempt_max} "
                        f"-- reason={epoch_fail_reason} | lr={lr:.3e}",
                    )
                    continue
                failure_reason = f"epoch_retry_exhausted:{epoch_fail_reason}"
                break

            prev_loss_ref = float(history[-1]["train_loss"]) if history else float("nan")
            train_loss_before = float(train_total.detach())
            model_base_state = copy.deepcopy(model.state_dict())
            opt_base_state = copy.deepcopy(optimizer.state_dict())
            grad_cache = _capture_grads(model)
            step_retry_count = 0
            step_accept_reason = "accepted"
            step_accepted = False

            attempts_total = 1 if not cfg.step_guard_enabled else max(1, int(cfg.step_retry_max) + 1)
            for step_attempt in range(1, attempts_total + 1):
                model.load_state_dict(copy.deepcopy(model_base_state))
                optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                _set_optimizer_lr(optimizer, lr)
                _restore_grads(model, grad_cache)
                optimizer.step()
                model.train()
                try:
                    trial_total, trial_parts, trial_traj = evaluate_split(
                        force_module=model,
                        known_pars=prepared.known_pars,
                        mech_true=mech_true_t,
                        ode_true=tensors["ode_full"],
                        x2dot_true=tensors["x2dot_full"],
                        contact_mask=tensors["contact_full"],
                        times=tensors["times_full"],
                        ode_method=cfg.ode_method,
                        ode_rtol=cfg.ode_rtol,
                        ode_atol=cfg.ode_atol,
                        eta_star_true=prepared.eta_star_true,
                        loss_indices=tensors["train_idx"],
                    )
                    trial_loss = float(trial_total.detach())
                except Exception as err:  # noqa: BLE001 - same retry semantics as KFT GBO.
                    step_accept_reason = f"step_trial_exception:{err}"
                    trial_total = None
                    trial_parts = None
                    trial_traj = None
                    trial_loss = float("inf")

                if step_accept_reason == "accepted":
                    if not np.isfinite(trial_loss):
                        step_accept_reason = "step_trial_loss_nonfinite"
                    elif np.isfinite(cfg.step_max_loss_increase_frac) and trial_loss > train_loss_before * (1.0 + cfg.step_max_loss_increase_frac):
                        step_accept_reason = "step_trial_loss_jump"
                    elif np.isfinite(prev_loss_ref) and trial_loss > prev_loss_ref * (1.0 + cfg.step_max_loss_increase_frac):
                        step_accept_reason = "step_prev_epoch_loss_jump"

                if step_accept_reason == "accepted":
                    train_total = trial_total
                    train_parts = trial_parts
                    accepted_train_traj = trial_traj
                    step_accepted = True
                    step_retry_count = step_attempt - 1
                    if step_retry_count > 0:
                        _log_line(
                            log,
                            f"step-guard: accepted after {step_retry_count} retry/reduction(s) "
                            f"| lr={lr:.3e} | train={trial_loss:.6e}",
                        )
                    break

                if step_attempt >= attempts_total:
                    step_retry_count = max(0, attempts_total - 1)
                    if _terminal_step_failure(step_accept_reason):
                        model.load_state_dict(copy.deepcopy(model_base_state))
                        optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                        _set_optimizer_lr(optimizer, lr)
                        failure_reason = f"trial_failed:{step_accept_reason}"
                        break
                    model.load_state_dict(copy.deepcopy(model_base_state))
                    optimizer.load_state_dict(copy.deepcopy(opt_base_state))
                    _set_optimizer_lr(optimizer, lr)
                    train_total = torch.as_tensor(train_loss_before, dtype=dtype, device=cfg.device)
                    train_parts = train_parts
                    accepted_train_traj = train_traj
                    step_accept_reason = "step_rejected_keep_previous"
                    _log_line(
                        log,
                        f"step-guard: rejected update after {step_retry_count} retries; keep previous parameters "
                        f"| reason={step_accept_reason} | lr={lr:.3e}",
                    )
                    step_accepted = True
                    break

                lr = max(lr * cfg.step_retry_lr_factor, cfg.epoch_retry_lr_floor)
                step_retry_count = step_attempt
                _log_line(
                    log,
                    f"step-guard retry {step_attempt}/{cfg.step_retry_max} -- reason={step_accept_reason} | lr={lr:.3e}",
                )
                step_accept_reason = "accepted"

            if failure_reason != "":
                break
            if step_accepted:
                recovered = epoch_attempt > 1
                break

        if failure_reason != "":
            failed = True
            _log_line(log, f"layer {layer_tag} failed | trial={trial} epoch={epoch} reason={failure_reason}")
            break

        model.eval()
        try:
            with torch.no_grad():
                train_loss, train_row = _eval_metrics(
                    cfg=cfg,
                    prepared=prepared,
                    tensors=tensors,
                    mech_true_t=mech_true_t,
                    model=model,
                    split_name="train",
                )
                val_loss, val_row = _eval_metrics(
                    cfg=cfg,
                    prepared=prepared,
                    tensors=tensors,
                    mech_true_t=mech_true_t,
                    model=model,
                    split_name="val",
                )
            if not np.isfinite(train_loss):
                failure_reason = "post_step_train_loss_nonfinite"
                failed = True
                break
            if not np.isfinite(val_loss):
                failure_reason = "post_step_val_loss_nonfinite"
                failed = True
                break
        except Exception as exc:  # noqa: BLE001 - failures become candidate metadata.
            failed = True
            failure_reason = f"{type(exc).__name__}: {exc}"
            _log_line(log, f"layer {layer_tag} failed | trial={trial} epoch={epoch} reason={failure_reason}")
            break

        row = {
            "epoch": int(epoch),
            "layer_epoch": int(epoch - start_epoch),
            "layer_tag": layer_tag,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "grad_norm": float(grad_norm),
            "grad_norm_raw": float(grad_norm_raw),
            "lr": float(lr),
            "epoch_retry_count": float(max(0, epoch_attempt - 1)),
            "step_retry_count": float(step_retry_count),
            "dt_epoch_sec": float(perf_counter() - epoch_t0),
            "epoch_sec": float(perf_counter() - epoch_t0),
            "grid_update_sec": float(grid_update_sec_epoch),
            "grid_update_runs": float(grid_update_runs_epoch),
            "g_nn": float(model.gain().detach().cpu().item()),
            "train": train_row,
            "val": val_row,
        }
        row.update(_prefixed_metrics("train", train_row))
        row.update(_prefixed_metrics("val", val_row))
        history.append(row)

        if float(train_loss) < float(best_train_loss):
            best_train_loss = float(train_loss)
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            _save_torch(
                best_checkpoint_path,
                {
                    "trial": int(trial),
                    "seed": int(seed),
                    "rs_rank": int(source.get("rs_rank", 0)),
                    "layer": layer_tag,
                    "epoch": int(epoch),
                    "best_epoch": int(best_epoch),
                    "best_train_loss": float(best_train_loss),
                    "best_val_loss": float(best_val_loss),
                    "state_dict": _state_dict_to_cpu(model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "history": history,
                    "source": dict(source),
                    "config": asdict(cfg),
                },
            )

        if ((epoch % cfg.log_every) == 0) or epoch == 1 or epoch == int(target_total_epochs):
            _log_line(log, f"KAN prerun epoch {epoch} train={float(train_loss):.6e}")
            _log_line(log, f"  grad_norm={grad_norm:.3e} lr={lr:.6g}")
            if recovered:
                _log_line(log, f"  retry: recovered_after={epoch_attempt - 1} rollback(s)")
            if step_retry_count > 0 and step_accept_reason == "accepted":
                _log_line(log, f"  step-guard: accepted_after={step_retry_count} retry/reduction(s)")
            _log_line(
                log,
                "  rec: "
                f"x1={float(train_row.get('x1_rec', float('nan'))):.2f}% "
                f"x3={float(train_row.get('x3_rec', float('nan'))):.2f}%",
            )
            _log_line(log, f"  nn: F_contact err={float(train_row.get('fts_rollout_rec', float('nan'))):.2f}%")
            _log_line(log, f"KAN prerun val epoch {epoch} val={float(val_loss):.6e}")

        if cfg.lr_adapt and np.isfinite(grad_norm) and grad_norm > 0.0:
            if not np.isfinite(grad_ema):
                grad_ema = grad_norm
            else:
                grad_ema = cfg.lr_ema_alpha * grad_ema + (1.0 - cfg.lr_ema_alpha) * grad_norm
            if not np.isfinite(grad_target):
                grad_target = grad_ema
            ratio = grad_target / (grad_ema + cfg.lr_eps)
            lr = max(cfg.lr_min, min(float(lr * (ratio ** cfg.lr_eta)), cfg.lr_max))
            _set_optimizer_lr(optimizer, lr)

        if cfg.plateau_early_stop:
            recent_losses.append(float(train_loss))
            if len(recent_losses) > int(cfg.plateau_window):
                recent_losses.pop(0)

        checkpoint_payload = {
            "trial": int(trial),
            "seed": int(seed),
            "rs_rank": int(source.get("rs_rank", 0)),
            "layer": layer_tag,
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_train_loss": float(best_train_loss),
            "best_val_loss": float(best_val_loss),
            "state_dict": _state_dict_to_cpu(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "history": history,
            "source": dict(source),
            "config": asdict(cfg),
            "lr": float(lr),
            "grad_ema": float(grad_ema),
            "grad_target": float(grad_target),
            "recent_losses": list(recent_losses),
        }
        _save_torch(running_checkpoint_path, checkpoint_payload)
        _save_json(history_path, history)

        if float(train_loss) < float(cfg.good_enough_loss):
            _log_line(log, f"  early-stop: good_enough (loss < {cfg.good_enough_loss:.1e})")
            break
        if (
            cfg.plateau_early_stop
            and len(recent_losses) == int(cfg.plateau_window)
            and abs(recent_losses[-1] - recent_losses[0]) < float(cfg.plateau_tol)
        ):
            _log_line(log, f"  early-stop: plateau_{cfg.plateau_window}ep (loss change < {cfg.plateau_tol:.1e})")
            break
        if cfg.recent_val_early_stop:
            stats = _recent_val_plateau_stats(
                history,
                min_epochs=int(cfg.recent_val_window),
                compare_gap=int(cfg.recent_val_compare_gap),
            )
            if stats is not None and stats["rel_gap_to_prev"] < float(cfg.recent_val_rel_current_frac):
                _log_line(
                    log,
                    "  early-stop: recent_val_plateau "
                    f"(min_epochs={cfg.recent_val_window}, gap={cfg.recent_val_compare_gap} | "
                    f"|val_n+gap-val_n|/val_n={stats['rel_gap_to_prev']:.6f} < "
                    f"{cfg.recent_val_rel_current_frac:.6f}) "
                    f"| val_n={stats['previous']:.6e} val_n+gap={stats['current']:.6e}",
                )
                break

    record = _record_from_run(
        layer_name=canonical_layer,
        source=source,
        candidate_root=candidate_root,
        history=history,
        best_train_loss=best_train_loss,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        failed=failed or not history,
        failure_reason=failure_reason,
        dt_total_sec=float(perf_counter() - wall_t0),
        log_path=candidate_log_path,
        tag=tag,
    )
    final_payload = {
        "trial": int(trial),
        "seed": int(seed),
        "rs_rank": int(source.get("rs_rank", 0)),
        "layer": layer_tag,
        "history": history,
        "config": asdict(cfg),
        "source": dict(source),
        "state_dict": _state_dict_to_cpu(model.state_dict()),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "candidate_metric": "best_train_loss",
        "candidate_loss": float(best_train_loss),
        "window_meta": {
            "role": split.role,
            "label": split.label,
            "title": _window_title(split.role),
            "start_idx": split.start_idx,
            "stop_idx": split.stop_idx,
            "t_start": split.t_start,
            "t_stop": split.t_stop,
        },
        "final_record": history[-1] if history else None,
        "record": record,
    }
    _save_torch(result_path, final_payload)
    _log_line(
        log,
        f"layer {layer_tag} done | trial={trial} rs_rank={int(source.get('rs_rank', 0))} "
        f"epochs_total={record['epochs_completed']} epochs_layer={record['current_layer_epochs']} "
        f"best_train={record['best_train_loss']:.6e} best_val={record['best_val_loss']:.6e} "
        f"best_epoch={record['best_epoch']} failed={record['failed']}",
    )
    return record


def _write_assignments(cfg, layer_name: str, items: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for shard_index in range(1, cfg.prerun_shard_count + 1):
        assigned = [
            items[pos - 1]
            for pos in _assigned_positions(len(items), cfg.prerun_shard_count, shard_index)
        ]
        path = _assignment_path(cfg, layer_name, shard_index)
        _save_json(path, assigned)
        paths.append(path)
    return paths


def _load_assignment_items(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing KFT prerun assignment file: {path}")
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise RuntimeError(f"KFT prerun assignment payload is not a list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _prepare_runtime(cfg):
    dtype = _torch_dtype(cfg.dtype)
    prepared = prepare_data(cfg)
    selected_window_indices = _selected_window_indices(cfg, prepared.splits)
    if len(selected_window_indices) != 1:
        raise ValueError(f"KFT prerun expects exactly one train window, got {selected_window_indices}")
    window_index = int(selected_window_indices[0])
    split = prepared.splits[window_index - 1]
    tensors = _window_to_torch(split, dtype=dtype, device=cfg.device)
    mech_true_t = torch.as_tensor(prepared.mech_true, dtype=dtype, device=cfg.device)
    return dtype, prepared, split, tensors, mech_true_t


def run_prerun_shard(
    cfg=None,
    *,
    shard_index: int | None = None,
    layer_name: str = LAYER_A,
    assignment_path: str | Path | None = None,
) -> dict[str, Any]:
    cfg = default_config() if cfg is None else cfg
    canonical_layer = _canonical_layer_name(layer_name)
    shard_index = int(cfg.shard_index if shard_index is None else shard_index)
    if assignment_path is None:
        assignment_path = _assignment_path(cfg, canonical_layer, shard_index)
    assignment_path = Path(assignment_path)
    items = _load_assignment_items(assignment_path)
    _dtype, prepared, split, tensors, mech_true_t = _prepare_runtime(cfg)
    tag = _window_file_tag(split.role)
    target_epochs = int(cfg.prerun_layer_a_epochs) if canonical_layer == LAYER_A else (
        int(cfg.prerun_layer_a_epochs) + int(cfg.prerun_layer_b_epochs)
    )
    shard_log = _shard_log_path(cfg, canonical_layer, shard_index)
    shard_log.parent.mkdir(parents=True, exist_ok=True)
    shard_result = _shard_result_path(cfg, canonical_layer, shard_index)
    records: list[dict[str, Any]] = []

    with shard_log.open("w", encoding="utf-8") as log:
        _log_line(
            log,
            f"AFM04 KFT prerun shard start | layer={canonical_layer} "
            f"shard={shard_index}/{cfg.prerun_shard_count} items={len(items)} "
            f"target_total_epochs={target_epochs}",
        )
        for item in items:
            trial = int(item["trial"])
            candidate_root = _candidate_root(cfg, trial)
            resume_path = None
            if canonical_layer == LAYER_B:
                resume_path = _candidate_running_checkpoint_path(candidate_root, tag)
            candidate_log_path = _candidate_log_path(candidate_root, canonical_layer, tag)
            candidate_log_path.parent.mkdir(parents=True, exist_ok=True)
            with candidate_log_path.open("w", encoding="utf-8") as child_log:
                record = _run_candidate_to_epoch(
                    cfg=cfg,
                    prepared=prepared,
                    split=split,
                    tensors=tensors,
                    mech_true_t=mech_true_t,
                    source=item,
                    layer_name=canonical_layer,
                    target_total_epochs=target_epochs,
                    log=child_log,
                    resume_checkpoint_path=resume_path,
                )
            records.append(record)
            _save_pickle(
                shard_result,
                {
                    "layer_name": canonical_layer,
                    "window_tag": tag,
                    "shard_index": int(shard_index),
                    "candidate_records": records,
                    "assignment_path": str(assignment_path),
                },
            )
            retry_summary = record.get("retry_summary", {})
            _log_line(
                log,
                "item done | "
                f"trial rank {int(record.get('rs_rank', 0))} "
                f"(trial {int(record.get('trial', 0))}) | "
                f"epochs_total={int(record.get('epochs_completed', 0))} | "
                f"epochs_layer={int(record.get('current_layer_epochs', 0))} | "
                f"best_train={float(record.get('best_train_loss', float('inf'))):.6e} | "
                f"final_train={float(record.get('final_train_loss', float('inf'))):.6e} | "
                f"final_val={float(record.get('final_val_loss', float('inf'))):.6e} | "
                f"{_retry_summary_line(retry_summary)}",
            )
        _log_line(log, f"AFM04 KFT prerun shard done | layer={canonical_layer} completed={len(records)}")

    return {
        "layer_name": canonical_layer,
        "shard_index": int(shard_index),
        "log_path": str(shard_log),
        "result_path": str(shard_result),
        "completed": len(records),
    }


def merge_prerun_results(cfg=None, *, layer_name: str = LAYER_B, tag: str | None = None) -> dict[str, Any]:
    cfg = default_config() if cfg is None else cfg
    _dtype, _prepared, split, _tensors, _mech_true_t = _prepare_runtime(cfg)
    tag = _window_file_tag(split.role) if tag is None else str(tag)
    canonical_layer = _canonical_layer_name(layer_name)
    all_records: list[dict[str, Any]] = []
    source_shards: list[str] = []
    for shard_index in range(1, cfg.prerun_shard_count + 1):
        path = _shard_result_path(cfg, canonical_layer, shard_index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing KFT prerun shard result: {path}")
        payload = _read_pickle(path)
        source_shards.append(str(path.resolve()))
        records = payload.get("candidate_records", []) if isinstance(payload, dict) else []
        all_records.extend([dict(row) for row in records if isinstance(row, dict)])

    ranked = [row for row in all_records if not bool(row.get("failed", False))]
    ranked.sort(key=_candidate_sort_key)
    if canonical_layer == LAYER_A:
        for idx, row in enumerate(ranked, start=1):
            row["layer_a_winner_rank"] = int(idx)
            row["top_trial_a"] = int(idx)
            row["newrank_a"] = int(idx)
    else:
        for idx, row in enumerate(ranked, start=1):
            row["candidate_rank"] = int(idx)
            row["candidate_b"] = int(idx)
            row["candidate"] = int(idx)

    summary = {
        "layer_name": canonical_layer,
        "ranking_label": "top trial A" if canonical_layer == LAYER_A else "candidate B",
        "total_candidates": len(ranked),
        "failed_candidates": len(all_records) - len(ranked),
        "candidate_metric": "best_train_loss",
        "best_candidate_loss": float(ranked[0].get("best_train_loss", float("inf"))) if ranked else float("inf"),
        "best_train_loss": float(ranked[0].get("best_train_loss", float("inf"))) if ranked else float("inf"),
        "source_shards": source_shards,
    }
    payload = {
        "layer_name": canonical_layer,
        "window_tag": tag,
        "ranking_label": summary["ranking_label"],
        "ranking_metric": "best_train_loss",
        "candidate_records": ranked,
        "all_records": all_records,
        "source_shards": source_shards,
        "summary": summary,
    }
    if canonical_layer == LAYER_A:
        payload["top_trial_a_records"] = ranked
        payload["newrank_a_records"] = ranked
    else:
        payload["candidate_b_records"] = ranked
    path = _merged_result_path(cfg, canonical_layer, tag)
    _save_pickle(path, payload)
    legacy_path = _legacy_merged_result_path(cfg, canonical_layer, tag)
    if legacy_path != path:
        _save_pickle(legacy_path, payload)
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    topk_path = path.with_suffix(path.suffix + ".topk.json")
    _save_json(summary_path, summary)
    _save_json(topk_path, ranked)
    return payload


def _run_layer_processes(
    cfg,
    *,
    layer_name: str,
    driver_log,
    assignment_paths: list[Path],
) -> None:
    canonical_layer = _canonical_layer_name(layer_name)
    procs: list[tuple[int, subprocess.Popen[str], Path]] = []
    base_env = dict(os.environ)
    base_env["HNODECB_AFM04_KAN_TEST_SHARD_COUNT"] = str(cfg.shard_count)
    base_env["HNODECB_AFM04_KAN_TEST_PRERUN_SHARD_COUNT"] = str(cfg.prerun_shard_count)
    base_env["HNODECB_AFM04_KAN_TEST_TRAIN_WINDOW_INDEX"] = str(cfg.train_window_index)
    base_env["HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_WINDOW_INDEX"] = str(cfg.random_search_window_index)
    base_env["HNODECB_AFM04_KAN_TEST_PRERUN_INPUT_TOPK"] = str(cfg.prerun_input_topk)
    base_env["HNODECB_AFM04_KAN_TEST_PRERUN_LAYER_A_EPOCHS"] = str(cfg.prerun_layer_a_epochs)
    base_env["HNODECB_AFM04_KAN_TEST_PRERUN_LAYER_A_TOPK"] = str(cfg.prerun_layer_a_topk)
    base_env["HNODECB_AFM04_KAN_TEST_PRERUN_LAYER_B_EPOCHS"] = str(cfg.prerun_layer_b_epochs)

    for shard_index in range(1, cfg.prerun_shard_count + 1):
        env = dict(base_env)
        env["HNODECB_AFM04_KAN_TEST_SHARD_INDEX"] = str(shard_index)
        args = [
            sys.executable,
            "-m",
            "AFM04.KAN_full_test.main",
            "--prerun-shard",
            "--prerun-layer",
            canonical_layer,
            "--shard-index",
            str(shard_index),
            "--prerun-assignment-path",
            str(assignment_paths[shard_index - 1]),
        ]
        shard_log = _shard_log_path(cfg, canonical_layer, shard_index)
        proc = subprocess.Popen(
            args,
            cwd=str(cfg.repo_root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        procs.append((shard_index, proc, shard_log))
        _log_line(
            driver_log,
            f"launched {canonical_layer} shard {shard_index}/{cfg.prerun_shard_count} | pid={proc.pid} | log={shard_log}",
        )

    exit_codes: dict[int, int] = {}
    try:
        for shard_index, proc, shard_log in procs:
            code = proc.wait()
            exit_codes[shard_index] = int(code)
            _log_line(
                driver_log,
                f"{canonical_layer} shard {shard_index}/{cfg.prerun_shard_count} exited | code={code} | log={shard_log}",
            )
    finally:
        for _shard_index, proc, _shard_log in procs:
            if proc.poll() is None:
                proc.kill()

    failed = {idx: code for idx, code in exit_codes.items() if code != 0}
    if failed:
        raise RuntimeError(f"AFM04 KFT prerun {canonical_layer} failed: {failed}")


def _write_promotion_flow(cfg, *, tag: str, candidates: list[dict[str, Any]]) -> Path:
    flow_path = cfg.prerun_result_dir / f"kan_full_test_prerun_promotion_flow_{tag}.txt"
    flow_lines = [
        "KFT prerun promotion flow",
        "ranking metric: best_train_loss",
        f"RS input top-k: {cfg.prerun_input_topk}",
        f"layer A: {cfg.prerun_layer_a_epochs} epochs, promote top {cfg.prerun_layer_a_topk}",
        f"layer B: seamless resume to total {cfg.prerun_layer_a_epochs + cfg.prerun_layer_b_epochs} epochs",
        "",
    ]
    for row in candidates:
        flow_lines.append(
            "trial rank "
            f"{int(row.get('rs_rank', 0))} (trial {int(row['trial'])}) "
            f"-> winner layer A {int(row.get('layer_a_winner_rank', 0))} "
            f"-> candidate {int(row.get('candidate_rank', 0))} "
            f"| best_train={float(row.get('best_train_loss', float('inf'))):.6e} "
            f"best_val={float(row.get('best_val_loss', float('inf'))):.6e}"
        )
    flow_path.parent.mkdir(parents=True, exist_ok=True)
    flow_path.write_text("\n".join(flow_lines) + "\n", encoding="utf-8")
    return flow_path


def run_prerun_driver(cfg=None) -> dict[str, Any]:
    cfg = default_config() if cfg is None else cfg
    _dtype, _prepared, split, _tensors, _mech_true_t = _prepare_runtime(cfg)
    tag = _window_file_tag(split.role)
    rs_history_path, rs_ranked_rows = _load_rs_ranked_rows(cfg, split)
    input_topk = min(int(cfg.prerun_input_topk), len(rs_ranked_rows))
    layer_a_topk = min(int(cfg.prerun_layer_a_topk), input_topk)
    layer_a_inputs = rs_ranked_rows[:input_topk]
    layer_a_assignments = _write_assignments(cfg, LAYER_A, layer_a_inputs)
    driver_log_path = _driver_log_path(cfg)
    driver_log_path.parent.mkdir(parents=True, exist_ok=True)

    with driver_log_path.open("w", encoding="utf-8") as log:
        _log_line(
            log,
            "AFM04 KFT prerun driver start | "
            f"shards={cfg.prerun_shard_count} window={_window_title(split.role)} "
            f"input_topk={input_topk} layerA_epochs={cfg.prerun_layer_a_epochs} "
            f"layerA_topk={layer_a_topk} layerB_epochs={cfg.prerun_layer_b_epochs} "
            f"metric=best_train_loss rs_history={rs_history_path}",
        )
        _run_layer_processes(cfg, layer_name=LAYER_A, driver_log=log, assignment_paths=layer_a_assignments)
        merged_a = merge_prerun_results(cfg, layer_name=LAYER_A, tag=tag)
        layer_a_records = [dict(row) for row in merged_a.get("candidate_records", [])]
        layer_a_winners = layer_a_records[:layer_a_topk]
        layer_b_assignments = _write_assignments(cfg, LAYER_B, layer_a_winners)
        _log_line(log, f"Layer A merge complete | winners={len(layer_a_winners)}")
        _log_layer_a_promotion_flow(log, cfg, merged_a=merged_a, layer_a_topk=layer_a_topk)

        _run_layer_processes(cfg, layer_name=LAYER_B, driver_log=log, assignment_paths=layer_b_assignments)
        merged_b = merge_prerun_results(cfg, layer_name=LAYER_B, tag=tag)
        candidates = [dict(row) for row in merged_b.get("candidate_records", [])]
        flow_path = _write_promotion_flow(cfg, tag=tag, candidates=candidates)
        summary_path = cfg.prerun_result_dir / f"kan_full_test_prerun_summary_{tag}.json"
        summary = {
            "window_tag": tag,
            "window_meta": {
                "role": split.role,
                "label": split.label,
                "title": _window_title(split.role),
                "start_idx": split.start_idx,
                "stop_idx": split.stop_idx,
                "t_start": split.t_start,
                "t_stop": split.t_stop,
            },
            "ranking_metric": "best_train_loss",
            "shard_count": int(cfg.prerun_shard_count),
            "rs_history_path": str(rs_history_path),
            "layer_a_path": str(_merged_result_path(cfg, LAYER_A, tag)),
            "layer_b_path": str(_merged_result_path(cfg, LAYER_B, tag)),
            "promotion_flow_path": str(flow_path),
            "log_path": str(driver_log_path),
            "best_candidate": candidates[0] if candidates else None,
            "candidates": candidates,
        }
        _save_json(summary_path, summary)
        _log_layer_b_promotion_flow(log, merged_b=merged_b)
        _log_line(log, f"AFM04 KFT prerun done | candidates={len(candidates)} | summary={summary_path}")
        if candidates:
            best = candidates[0]
            _log_line(
                log,
                "best promotion flow: "
                f"trial rank {int(best.get('rs_rank', 0))} -> "
                f"winner layer A {int(best.get('layer_a_winner_rank', 0))} -> "
                f"candidate {int(best.get('candidate_rank', 0))}",
            )
            _log_line(log, f"driver merge complete | best_candidate_loss={float(best.get('best_train_loss', float('inf'))):.6e}")

    return {
        "summary_path": str(summary_path),
        "promotion_flow_path": str(flow_path),
        "log_path": str(driver_log_path),
    }


__all__ = ["merge_prerun_results", "run_prerun_driver", "run_prerun_shard"]
