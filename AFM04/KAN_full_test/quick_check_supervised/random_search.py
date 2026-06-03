"""Sharded random-search front stage for the QCS hybrid supervised rollout check."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from AFM04.KAN_full_test.data import prepare_data
from AFM04.KAN_full_test.quick_check_supervised.config import QuickCheckSupervisedConfig, default_config
from AFM04.KAN_full_test.quick_check_supervised.losses import fts_scale_from_truth
from AFM04.KAN_full_test.quick_check_supervised.main import (
    _build_model,
    _loss_for_part,
    _observable_grid_inputs_from_ode,
    _save_json,
    _save_torch,
    _selected_splits,
    _split_tensors,
    _torch_dtype,
    _window_file_tag,
)


def _timestamp_file() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _timestamp_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_line(log, message: str, *, echo: bool = False) -> None:
    line = f"[{_timestamp_human()}] {message}"
    log.write(line + "\n")
    log.flush()
    if echo:
        print(line, flush=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _trial_seed(base_seed: int, trial_index: int) -> int:
    return int(base_seed + trial_index)


def _trial_seed_bank_path(cfg: QuickCheckSupervisedConfig) -> Path:
    return cfg.random_search_result_dir / "qcs_random_search_trial_seeds.json"


def _generate_trial_seed_bank(cfg: QuickCheckSupervisedConfig) -> list[int]:
    trial_count = int(cfg.random_search_trials)
    if not bool(cfg.random_search_true_random):
        return [_trial_seed(cfg.seed, trial_index) for trial_index in range(1, trial_count + 1)]
    max_seed = (1 << 63) - 1
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < trial_count:
        seed = int(secrets.randbelow(max_seed - 1) + 1)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def write_random_search_seed_bank(cfg: QuickCheckSupervisedConfig) -> dict[str, Any]:
    path = _trial_seed_bank_path(cfg)
    seeds = _generate_trial_seed_bank(cfg)
    payload = {
        "trial_count": int(cfg.random_search_trials),
        "true_random": bool(cfg.random_search_true_random),
        "generated_at": _timestamp_human(),
        "trial_seeds": [int(seed) for seed in seeds],
    }
    _save_json(path, payload)
    payload["path"] = str(path)
    return payload


def load_random_search_seed_bank(cfg: QuickCheckSupervisedConfig, *, create_if_missing: bool = False) -> list[int]:
    path = _trial_seed_bank_path(cfg)
    if not path.is_file():
        if not create_if_missing:
            raise FileNotFoundError(f"Missing QCS random-search seed bank: {path}")
        write_random_search_seed_bank(cfg)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    seeds = payload.get("trial_seeds", [])
    if not isinstance(seeds, list):
        raise RuntimeError(f"Invalid QCS random-search seed bank format: {path}")
    trial_count = int(payload.get("trial_count", len(seeds)))
    if trial_count != int(cfg.random_search_trials) or len(seeds) != int(cfg.random_search_trials):
        raise RuntimeError(
            "QCS random-search seed bank size mismatch: "
            f"path={path} file_trials={trial_count} seeds={len(seeds)} cfg_trials={cfg.random_search_trials}"
        )
    return [int(seed) for seed in seeds]


def _topk_insert(topk: list[dict[str, Any]], row: dict[str, Any], k: int) -> None:
    topk.append(dict(row))
    topk.sort(key=lambda item: (float(item["ranking_loss"]), float(item["train_loss"]), int(item["trial"])))
    del topk[k:]


def _row_is_valid(row: dict[str, Any] | None) -> bool:
    return row is not None and not bool(row.get("failed", False)) and np.isfinite(float(row.get("ranking_loss", float("inf"))))


def _row_sort_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (float(row["ranking_loss"]), float(row["train_loss"]), int(row["trial"]))


def prepare_data_for_qcs(cfg: QuickCheckSupervisedConfig) -> dict[str, Any]:
    prepared_data = prepare_data(cfg)
    selected = _selected_splits(cfg, prepared_data.splits)
    if len(selected) != 1:
        raise RuntimeError("QCS random search currently expects exactly one selected window")
    split = selected[0]
    dtype = _torch_dtype(cfg.dtype)
    train = _split_tensors(split, "train", dtype=dtype, device=cfg.device)
    val = _split_tensors(split, "val", dtype=dtype, device=cfg.device)
    observable_grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=train["ode"],
        state_mean=prepared_data.state_mean,
        state_scale=prepared_data.state_scale,
        cfg=cfg,
    )
    fts_scale = fts_scale_from_truth(train["fts"], mode=cfg.fts_scale_mode).to(dtype=dtype, device=cfg.device)
    return {
        "prepared": prepared_data,
        "split": split,
        "train": train,
        "val": val,
        "observable_grid_inputs": observable_grid_inputs,
        "fts_scale": fts_scale,
    }


def _trial_row_for_seed(
    *,
    cfg: QuickCheckSupervisedConfig,
    prepared: dict[str, Any],
    dtype: torch.dtype,
    trial_index: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    train = prepared["train"]
    val = prepared["val"]
    ranking_metric = "val_loss" if cfg.random_search_use_val else "train_loss"
    t0 = perf_counter()
    failure_reason = ""
    train_total = None
    train_parts = None
    val_total = None
    val_parts = None
    state_dict = None
    try:
        model = _build_model(cfg=cfg, prepared=prepared["prepared"], seed=seed, dtype=dtype)
        with torch.no_grad():
            if cfg.adaptive_grid_enabled:
                model.update_grid_from_normalized_inputs(prepared["observable_grid_inputs"])
            init_gain = model.initialize_gain_from_truth(train["states"], train["fts"])
            train_total, train_parts, _, _, _ = _loss_for_part(
                model,
                prepared["prepared"],
                train,
                cfg,
                prepared["fts_scale"],
            )
            if cfg.random_search_use_val:
                val_total, val_parts, _, _, _ = _loss_for_part(
                    model,
                    prepared["prepared"],
                    val,
                    cfg,
                    prepared["fts_scale"],
                )
            state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    except Exception as err:
        failure_reason = f"{type(err).__name__}: {err}"
        init_gain = float("nan")

    train_loss = float(train_total.detach()) if train_total is not None else float("inf")
    val_loss = float(val_total.detach()) if val_total is not None else float("nan")
    ranking_loss = val_loss if cfg.random_search_use_val and np.isfinite(val_loss) else train_loss
    row = {
        "trial": int(trial_index),
        "seed": int(seed),
        "ranking_metric": ranking_metric,
        "ranking_loss": float(ranking_loss),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "failed": failure_reason != "",
        "failure_reason": failure_reason,
        "init_gnn": float(init_gain),
        "train_formal": float(train_parts.formal_total) if train_parts is not None else float("nan"),
        "train_fts_norm_mse": float(train_parts.fts_norm_mse) if train_parts is not None else float("nan"),
        "train_fts_rel_rmse_pct": float(train_parts.fts_rel_rmse_pct) if train_parts is not None else float("nan"),
        "val_formal": float(val_parts.formal_total) if val_parts is not None else float("nan"),
        "val_fts_norm_mse": float(val_parts.fts_norm_mse) if val_parts is not None else float("nan"),
        "val_fts_rel_rmse_pct": float(val_parts.fts_rel_rmse_pct) if val_parts is not None else float("nan"),
        "seconds": float(perf_counter() - t0),
    }
    return row, state_dict


def run_random_search_shard(
    cfg: QuickCheckSupervisedConfig | None = None,
    *,
    shard_index: int | None = None,
) -> dict[str, Any]:
    cfg = cfg or default_config()
    shard_index = int(cfg.random_search_shard_index if shard_index is None else shard_index)
    shard_count = int(cfg.random_search_shard_count)
    if shard_index < 1 or shard_index > shard_count:
        raise ValueError(f"invalid QCS shard_index={shard_index}, shard_count={shard_count}")

    dtype = _torch_dtype(cfg.dtype)
    prepared = prepare_data_for_qcs(cfg)
    split = prepared["split"]
    tag = _window_file_tag(split.role)
    seeds = load_random_search_seed_bank(cfg, create_if_missing=(shard_count == 1))
    ranking_metric = "val_loss" if cfg.random_search_use_val else "train_loss"
    trial_indices = list(range(shard_index, int(cfg.random_search_trials) + 1, shard_count))
    log_path = cfg.random_search_shard_log_dir / f"qcs_random_search_local_s{shard_index}.txt"

    rows: list[dict[str, Any]] = []
    topk: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None

    with log_path.open("w", encoding="utf-8") as log:
        _log_line(
            log,
            "QCS random search shard start | "
            f"shard={shard_index}/{shard_count} trials_total={cfg.random_search_trials} "
            f"local_trials={len(trial_indices)} ranking={ranking_metric}",
        )
        _log_line(log, f"window={split.role} width={cfg.width} grid={cfg.grid} k={cfg.spline_k} base_fun={cfg.base_fun}")
        _log_line(log, f"AGU source=normalized_observed_x1x2_neutral_x3_axis enabled={cfg.adaptive_grid_enabled}")
        _log_line(log, f"loss=hybrid formal rollout + normalized Fts supervised | ode={cfg.ode_method} rtol={cfg.ode_rtol:.1e} atol={cfg.ode_atol:.1e}")

        for local_i, trial_index in enumerate(trial_indices, start=1):
            seed = int(seeds[trial_index - 1])
            row, state_dict = _trial_row_for_seed(
                cfg=cfg,
                prepared=prepared,
                dtype=dtype,
                trial_index=trial_index,
                seed=seed,
            )
            row["shard_index"] = int(shard_index)
            rows.append(row)
            if _row_is_valid(row):
                _topk_insert(topk, row, int(cfg.random_search_topk))
                if best_row is None or _row_sort_key(row) < _row_sort_key(best_row):
                    best_row = dict(row)
                    best_state_dict = state_dict
                    _log_line(
                        log,
                        f"new best | trial={trial_index} seed={seed} rank({ranking_metric})={row['ranking_loss']:.6e} "
                        f"train={row['train_loss']:.6e} val={row['val_loss']:.6e} Fts={row['train_fts_rel_rmse_pct']:.3f}%",
                    )

            if local_i == 1 or local_i % int(cfg.random_search_log_every) == 0 or local_i == len(trial_indices):
                best_loss = float(best_row["ranking_loss"]) if best_row is not None else float("nan")
                status = "FAIL" if row["failed"] else "OK"
                _log_line(
                    log,
                    f"local {local_i}/{len(trial_indices)} trial {trial_index}/{cfg.random_search_trials} "
                    f"status={status} rank({ranking_metric})={row['ranking_loss']:.6e} "
                    f"train={row['train_loss']:.6e} best={best_loss:.6e} seconds={row['seconds']:.3f}",
                )

        if best_row is None or best_state_dict is None:
            raise RuntimeError(f"QCS random-search shard {shard_index} found no valid trial")

        trials_path = _save_json(cfg.random_search_result_dir / f"qcs_random_search_trials_{tag}_s{shard_index}.json", rows)
        summary = {
            "status": "ok",
            "window_role": split.role,
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "local_trials": len(trial_indices),
            "trials_total": int(cfg.random_search_trials),
            "ranking_metric": ranking_metric,
            "best_trial": dict(best_row),
            "topk": topk,
            "trials_path": trials_path,
            "seed_bank_path": _trial_seed_bank_path(cfg),
            "log_path": log_path,
            "config": asdict(cfg),
        }
        summary_path = _save_json(cfg.random_search_result_dir / f"qcs_random_search_summary_{tag}_s{shard_index}.json", summary)
        best_path = cfg.random_search_checkpoint_dir / f"qcs_random_search_best_{tag}_s{shard_index}.pt"
        _save_torch(
            best_path,
            {
                "best_trial": dict(best_row),
                "best_state_dict": best_state_dict,
                "window_meta": {
                    "role": split.role,
                    "label": split.label,
                    "start_idx": split.start_idx,
                    "stop_idx": split.stop_idx,
                    "t_start": split.t_start,
                    "t_stop": split.t_stop,
                },
                "config": asdict(cfg),
                "state_mean": prepared["prepared"].state_mean,
                "state_scale": prepared["prepared"].state_scale,
                "known_pars": prepared["prepared"].known_pars,
                "eta_star_true": prepared["prepared"].eta_star_true,
                "mech_true": prepared["prepared"].mech_true,
                "summary_path": summary_path,
                "trials_path": trials_path,
            },
        )
        _log_line(log, f"QCS random-search shard done | best_trial={best_row['trial']} best_rank={best_row['ranking_loss']:.6e}")

    return {
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "best_path": str(best_path),
        "best_trial": dict(best_row),
    }


def merge_random_search_results(cfg: QuickCheckSupervisedConfig | None = None) -> dict[str, Any]:
    cfg = cfg or default_config()
    prepared = prepare_data_for_qcs(cfg)
    split = prepared["split"]
    tag = _window_file_tag(split.role)
    all_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for shard_index in range(1, int(cfg.random_search_shard_count) + 1):
        trials_path = cfg.random_search_result_dir / f"qcs_random_search_trials_{tag}_s{shard_index}.json"
        summary_path = cfg.random_search_result_dir / f"qcs_random_search_summary_{tag}_s{shard_index}.json"
        if not trials_path.is_file():
            raise FileNotFoundError(f"Missing QCS random-search shard trials: {trials_path}")
        with trials_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected QCS shard trial payload: {trials_path}")
        all_rows.extend(dict(row) for row in rows)
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as f:
                shard_summaries.append(json.load(f))

    all_rows.sort(key=lambda row: int(row.get("trial", 10**12)))
    valid = [row for row in all_rows if _row_is_valid(row)]
    if not valid:
        raise RuntimeError("QCS random-search merge found no valid trial")
    valid.sort(key=_row_sort_key)
    best_row = dict(valid[0])
    topk = [dict(row) for row in valid[: int(cfg.random_search_topk)]]
    best_shard = int(best_row.get("shard_index", 1))
    shard_best_path = cfg.random_search_checkpoint_dir / f"qcs_random_search_best_{tag}_s{best_shard}.pt"
    if not shard_best_path.is_file():
        raise FileNotFoundError(f"Missing QCS best shard checkpoint: {shard_best_path}")
    shard_payload = torch.load(shard_best_path, map_location=cfg.device, weights_only=False)
    state_dict = shard_payload.get("best_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"QCS shard checkpoint missing best_state_dict: {shard_best_path}")

    trials_path = _save_json(cfg.random_search_result_dir / f"qcs_random_search_trials_{tag}.json", all_rows)
    summary = {
        "status": "ok",
        "window_role": split.role,
        "trials": int(cfg.random_search_trials),
        "shard_count": int(cfg.random_search_shard_count),
        "ranking_metric": str(best_row.get("ranking_metric", "train_loss")),
        "best_trial": best_row,
        "topk": topk,
        "trials_path": trials_path,
        "seed_bank_path": _trial_seed_bank_path(cfg),
        "shard_summaries": shard_summaries,
        "config": asdict(cfg),
    }
    summary_path = _save_json(cfg.random_search_result_dir / f"qcs_random_search_summary_{tag}.json", summary)
    merged_summary_path = _save_json(
        cfg.random_search_result_dir / "qcs_random_search_merged_summary.json",
        {
            "status": "ok",
            "best_trial": int(best_row["trial"]),
            "best_seed": int(best_row["seed"]),
            "best_ranking_loss": float(best_row["ranking_loss"]),
            "best_train_loss": float(best_row["train_loss"]),
            "shard_count": int(cfg.random_search_shard_count),
            "summary_path": summary_path,
            "trials_path": trials_path,
        },
    )
    best_path = cfg.random_search_checkpoint_dir / f"qcs_random_search_best_{tag}.pt"
    _save_torch(
        best_path,
        {
            "best_trial": best_row,
            "best_state_dict": state_dict,
            "window_meta": {
                "role": split.role,
                "label": split.label,
                "start_idx": split.start_idx,
                "stop_idx": split.stop_idx,
                "t_start": split.t_start,
                "t_stop": split.t_stop,
            },
            "config": asdict(cfg),
            "state_mean": prepared["prepared"].state_mean,
            "state_scale": prepared["prepared"].state_scale,
            "known_pars": prepared["prepared"].known_pars,
            "eta_star_true": prepared["prepared"].eta_star_true,
            "mech_true": prepared["prepared"].mech_true,
            "summary_path": summary_path,
            "merged_summary_path": merged_summary_path,
            "trials_path": trials_path,
        },
    )
    return {"summary_path": str(summary_path), "best_path": str(best_path), "best_trial": best_row}


def run_random_search_driver(cfg: QuickCheckSupervisedConfig | None = None) -> dict[str, Any]:
    cfg = cfg or default_config()
    driver_log = cfg.random_search_log_dir / f"qcs_random_search_driver_{_timestamp_file()}.txt"
    write_random_search_seed_bank(cfg)
    procs: list[tuple[int, subprocess.Popen[str], Path]] = []
    base_env = dict(os.environ)
    base_env["HNODECB_AFM04_KFT_QCS_RANDOM_SEARCH_SHARD_COUNT"] = str(cfg.random_search_shard_count)
    base_env["HNODECB_AFM04_KFT_QCS_RANDOM_SEARCH_TRIALS"] = str(cfg.random_search_trials)
    base_env["HNODECB_AFM04_KFT_QCS_TRAIN_WINDOW_INDEX"] = str(cfg.train_window_index)
    base_env["HNODECB_AFM04_KFT_QCS_WINDOW_MODE"] = str(cfg.window_mode)
    base_env["HNODECB_AFM04_KFT_QCS_WINDOW_US"] = str(cfg.arch_window_us)

    with driver_log.open("w", encoding="utf-8") as log:
        _log_line(
            log,
            "QCS random-search driver start | "
            f"window_index={cfg.train_window_index} trials={cfg.random_search_trials} "
            f"shards={cfg.random_search_shard_count}",
            echo=True,
        )
        _log_line(log, f"Logs -> {cfg.random_search_shard_log_dir}", echo=True)
        _log_line(log, f"Results -> {cfg.random_search_result_dir}", echo=True)
        for shard_index in range(1, int(cfg.random_search_shard_count) + 1):
            env = dict(base_env)
            env["HNODECB_AFM04_KFT_QCS_RANDOM_SEARCH_SHARD_INDEX"] = str(shard_index)
            shard_log = cfg.random_search_shard_log_dir / f"qcs_random_search_local_s{shard_index}.txt"
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "AFM04.KAN_full_test.quick_check_supervised.random_search",
                    "--shard",
                    "--shard-index",
                    str(shard_index),
                ],
                cwd=str(cfg.repo_root),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            procs.append((shard_index, proc, shard_log))
            _log_line(log, f"launched QCS-RS shard {shard_index}/{cfg.random_search_shard_count} | pid={proc.pid} | log={shard_log}", echo=True)

        exit_codes: dict[int, int] = {}
        try:
            for shard_index, proc, shard_log in procs:
                code = proc.wait()
                exit_codes[shard_index] = int(code)
                _log_line(log, f"QCS-RS shard {shard_index}/{cfg.random_search_shard_count} exited | code={code} | log={shard_log}", echo=True)
        finally:
            for _idx, proc, _log_path in procs:
                if proc.poll() is None:
                    proc.kill()

        failed = {idx: code for idx, code in exit_codes.items() if code != 0}
        if failed:
            _log_line(log, f"QCS random-search driver failed | failed_shards={failed}", echo=True)
            raise RuntimeError(f"QCS random-search driver failed: {failed}")

        merged = merge_random_search_results(cfg)
        _log_line(
            log,
            f"QCS random-search merge complete | best_trial={int(merged['best_trial']['trial'])} "
            f"best_rank={float(merged['best_trial']['ranking_loss']):.6e}",
            echo=True,
        )
        return {"driver_log": str(driver_log), "exit_codes": exit_codes, "merged_summary": merged}


def run_random_search(cfg: QuickCheckSupervisedConfig | None = None) -> dict[str, Any]:
    return run_random_search_driver(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--shard", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    args = parser.parse_args()
    cfg = default_config()
    if args.shard_index is not None:
        cfg = replace(cfg, random_search_shard_index=max(1, int(args.shard_index)))
    if args.shard:
        result = run_random_search_shard(cfg, shard_index=args.shard_index)
    elif args.merge:
        result = merge_random_search_results(cfg)
    else:
        result = run_random_search_driver(cfg)
    print("QCS random search completed")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
