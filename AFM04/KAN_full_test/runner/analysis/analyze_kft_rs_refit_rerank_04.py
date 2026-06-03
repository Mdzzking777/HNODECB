from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import fields, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.config import default_config
from AFM04.KAN_full_test.data import prepare_data, x3dot_init_from_split, x3dot_scale_from_split
from AFM04.KAN_full_test.kan_backend import KANForceModule
from AFM04.KAN_full_test.losses import evaluate_split
from AFM04.KAN_full_test.random_search import _rebuild_trial_state_dict
from AFM04.KAN_full_test.rollout import force_inputs_for_module
from AFM04.KAN_full_test.train import _torch_dtype, _window_file_tag, _window_to_torch


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "analysis"
RESULT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "results" / "random_search"
CHECKPOINT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "checkpoints" / "random_search"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _config_from_payload(payload: dict[str, Any]):
    cfg = default_config(REPO_ROOT)
    raw = payload.get("config", {})
    if not isinstance(raw, dict):
        return cfg
    known_fields = {field.name for field in fields(cfg)}
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known_fields:
            continue
        if key == "width":
            updates[key] = tuple(int(v) for v in value)
        elif key.endswith("_root") or key.endswith("_dir"):
            updates[key] = Path(value)
        elif key == "device":
            updates[key] = "cpu"
        else:
            updates[key] = value
    return replace(cfg, **updates)


def _load_best_payload(tag: str) -> dict[str, Any]:
    path = CHECKPOINT_DIR / f"kan_full_test_random_search_best_{tag}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RS best checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid RS best checkpoint payload: {path}")
    return payload


def _load_all_trial_rows(tag: str) -> tuple[Path, list[dict[str, Any]]]:
    path = RESULT_DIR / f"kan_full_test_random_search_trials_{tag}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RS trial rows: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid RS trial rows: {path}")
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            clean_rows.append(dict(row))
    return path, clean_rows


def _valid_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or bool(row.get("failed", False)):
            continue
        old_loss = float(row.get("ranking_loss", row.get("train_loss", float("inf"))))
        train_loss = float(row.get("train_loss", float("inf")))
        if np.isfinite(old_loss) and np.isfinite(train_loss):
            valid.append(dict(row))
    valid.sort(key=lambda r: (float(r.get("ranking_loss", r.get("train_loss", float("inf")))), float(r.get("train_loss", float("inf"))), int(r.get("trial", -1))))
    for rank, row in enumerate(valid, start=1):
        row["_old_rank"] = int(rank)
    return valid


def _trial_file_stats(rows: list[dict[str, Any]], valid: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row for row in rows if bool(row.get("failed", False))]
    reasons: dict[str, int] = {}
    for row in failed:
        reason = str(row.get("failure_reason", row.get("error", "unknown")))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "total_trial_rows": int(len(rows)),
        "failed_trial_rows": int(len(failed)),
        "valid_trial_rows": int(len(valid)),
        "failure_reasons": [
            {"reason": reason, "count": int(count)}
            for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def _build_model(cfg, prepared, split, *, seed: int, dtype: torch.dtype) -> KANForceModule:
    model = KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=int(seed),
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
        wpred_enabled=cfg.wpred_enabled,
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
        x3dot_init_trainable=False,
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
        device="cpu",
        dtype=dtype,
    ).to("cpu")
    return model


def _normalizer_arrays(model: KANForceModule) -> tuple[np.ndarray, np.ndarray]:
    return (
        model.state_mean.detach().cpu().numpy().astype(float).copy(),
        model.state_scale.detach().cpu().numpy().astype(float).copy(),
    )


def _set_normalizer(model: KANForceModule, mean: np.ndarray, scale: np.ndarray) -> None:
    if mean.shape != tuple(model.state_mean.shape) or scale.shape != tuple(model.state_scale.shape):
        raise ValueError("normalizer shape mismatch")
    with torch.no_grad():
        model.state_mean.copy_(torch.as_tensor(mean, dtype=model.state_mean.dtype, device=model.state_mean.device))
        model.state_scale.copy_(torch.as_tensor(scale, dtype=model.state_scale.dtype, device=model.state_scale.device))


def _maxabs_scale(values: torch.Tensor, mean: torch.Tensor, *, floor: float = 1.0e-12) -> float:
    scale = float(torch.max(torch.abs(values - mean)).detach())
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(floor)
    return max(scale, float(floor))


def _finite_positive(value: Any) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v > 0.0)


def _rs_rollout_stats_available(row: dict[str, Any], *, q_required: bool) -> bool:
    if not bool(row.get("rs_rollout_stats_available", False)):
        return False
    required = ("rs_rollout_x3_mean", "rs_rollout_x3_scale")
    for key in required:
        if key.endswith("_scale"):
            if not _finite_positive(row.get(key)):
                return False
        else:
            try:
                if not np.isfinite(float(row.get(key))):
                    return False
            except (TypeError, ValueError):
                return False
    if q_required:
        if not bool(row.get("rs_rollout_q_available", False)):
            return False
        if not _finite_positive(row.get("rs_rollout_q_scale")):
            return False
        try:
            if not np.isfinite(float(row.get("rs_rollout_q_mean"))):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _evaluate_and_refit_one(*, cfg, prepared, split, tensors, row: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    seed = int(row.get("seed", int(cfg.seed) + int(row["trial"])))
    model = _build_model(cfg, prepared, split, seed=seed, dtype=dtype)
    state_dict = _rebuild_trial_state_dict(
        cfg=cfg,
        prepared=prepared,
        split=split,
        seed=seed,
        dtype=dtype,
    )
    model.load_state_dict(state_dict)
    model.eval()
    mech_true_t = torch.as_tensor(prepared.mech_true, dtype=dtype, device="cpu")
    train_idx = tensors["train_idx"]
    old_mean, old_scale = _normalizer_arrays(model)
    old_loss_value = float("nan")
    old_parts_values = {
        "x1_rec": float(row.get("x1_rec", row.get("train_x1_rec", float("nan")))),
        "x3_rec": float(row.get("x3_rec", row.get("train_x3_rec", float("nan")))),
        "fts_rec": float(row.get("nn_err", row.get("train_nn_err", float("nan")))),
    }
    diagnostic_q_current_mean = float("nan")
    diagnostic_q_current_scale = float("nan")
    stats_source = "replayed_old_rollout"

    with torch.no_grad():
        new_mean = old_mean.copy()
        new_scale = old_scale.copy()
        if _rs_rollout_stats_available(row, q_required=bool(getattr(model, "x3dot_input_enabled", False))):
            stats_source = "rs_saved_train_rollout_stats"
            old_loss_value = float(row.get("train_loss", row.get("ranking_loss", float("nan"))))
            new_mean[2] = float(row["rs_rollout_x3_mean"])
            new_scale[2] = float(row["rs_rollout_x3_scale"])
            if bool(getattr(model, "x3dot_input_enabled", False)):
                new_mean[3] = float(row["rs_rollout_q_mean"])
                new_scale[3] = float(row["rs_rollout_q_scale"])
        else:
            old_loss_t, old_parts, traj = evaluate_split(
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
                loss_indices=train_idx,
            )
            old_loss_value = float(old_loss_t.detach())
            old_parts_values = {
                "x1_rec": float(old_parts.x1_rec),
                "x3_rec": float(old_parts.x3_rec),
                "fts_rec": float(old_parts.fts_rollout_rec),
            }
            q_pre = getattr(traj, "_plan_z_q_pre", None)
            if bool(getattr(model, "x3dot_input_enabled", False)) and q_pre is None:
                raise RuntimeError("Plan Z refit rerank requires q_pre from rollout")
            x3_train = traj[2, train_idx]
            x3_mean_t = torch.mean(x3_train)
            new_mean[2] = float(x3_mean_t.detach())
            new_scale[2] = _maxabs_scale(x3_train, x3_mean_t)
            if q_pre is not None:
                q_train = q_pre[train_idx]
                q_mean_t = torch.mean(q_train)
                new_mean[3] = float(q_mean_t.detach())
                new_scale[3] = _maxabs_scale(q_train, q_mean_t)

                states_full = traj.transpose(0, 1)
                f_pred = model(force_inputs_for_module(model, states_full, q_pre))
                ks = torch.as_tensor(prepared.mech_true, dtype=dtype).reshape(-1)[0]
                cs = torch.as_tensor(prepared.mech_true, dtype=dtype).reshape(-1)[1]
                q_current = (-f_pred - ks * traj[2, :]) / cs
                q_current_train = q_current[train_idx]
                q_current_mean_t = torch.mean(q_current_train)
                diagnostic_q_current_mean = float(q_current_mean_t.detach())
                diagnostic_q_current_scale = _maxabs_scale(q_current_train, q_current_mean_t)

        _set_normalizer(model, new_mean, new_scale)
        refit_loss_t, refit_parts, _ = evaluate_split(
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
            loss_indices=train_idx,
        )

    return {
        "old_rank": int(row["_old_rank"]),
        "trial": int(row["trial"]),
        "seed": seed,
        "old_ranking_loss_json": float(row.get("ranking_loss", row.get("train_loss", float("nan")))),
        "old_train_loss_json": float(row.get("train_loss", float("nan"))),
        "old_train_loss_recomputed": float(old_loss_value),
        "normalizer_stats_source": stats_source,
        "refit_train_loss": float(refit_loss_t.detach()),
        "old_x1_rec": float(old_parts_values["x1_rec"]),
        "old_x3_rec": float(old_parts_values["x3_rec"]),
        "old_fts_rec": float(old_parts_values["fts_rec"]),
        "refit_x1_rec": float(refit_parts.x1_rec),
        "refit_x3_rec": float(refit_parts.x3_rec),
        "refit_fts_rec": float(refit_parts.fts_rollout_rec),
        "old_x3_mean": float(old_mean[2]),
        "old_x3_scale": float(old_scale[2]),
        "old_q_mean": float(old_mean[3]) if old_mean.size >= 4 else float("nan"),
        "old_q_scale": float(old_scale[3]) if old_scale.size >= 4 else float("nan"),
        "refit_x3_mean": float(new_mean[2]),
        "refit_x3_scale": float(new_scale[2]),
        "refit_q_mean": float(new_mean[3]) if new_mean.size >= 4 else float("nan"),
        "refit_q_scale": float(new_scale[3]) if new_scale.size >= 4 else float("nan"),
        "diagnostic_q_current_mean": float(diagnostic_q_current_mean),
        "diagnostic_q_current_scale": float(diagnostic_q_current_scale),
    }


def _failed_refit_row(row: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    old_loss = float(row.get("ranking_loss", row.get("train_loss", float("inf"))))
    train_loss = float(row.get("train_loss", float("inf")))
    err_type = type(exc).__name__
    err_msg = str(exc)
    return {
        "old_rank": int(row["_old_rank"]),
        "trial": int(row["trial"]),
        "seed": int(row.get("seed", -1)),
        "old_ranking_loss_json": old_loss,
        "old_train_loss_json": train_loss,
        "old_train_loss_recomputed": float("nan"),
        "refit_train_loss": float("inf"),
        "refit_failed": True,
        "refit_failure_type": err_type,
        "refit_failure_reason": err_msg,
        "refit_failure_traceback_tail": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        "old_x1_rec": float("nan"),
        "old_x3_rec": float("nan"),
        "old_fts_rec": float("nan"),
        "refit_x1_rec": float("nan"),
        "refit_x3_rec": float("nan"),
        "refit_fts_rec": float("nan"),
        "old_x3_mean": float("nan"),
        "old_x3_scale": float("nan"),
        "old_q_mean": float("nan"),
        "old_q_scale": float("nan"),
        "refit_x3_mean": float("nan"),
        "refit_x3_scale": float("nan"),
        "refit_q_mean": float("nan"),
        "refit_q_scale": float("nan"),
        "diagnostic_q_current_mean": float("nan"),
        "diagnostic_q_current_scale": float("nan"),
    }


def _spearman_from_ranks(rows: list[dict[str, Any]]) -> float:
    n = len(rows)
    if n < 2:
        return float("nan")
    old_order = sorted(rows, key=lambda row: (int(row["old_rank"]), int(row["trial"])))
    new_order = sorted(rows, key=lambda row: (int(row["new_rank"]), int(row["trial"])))
    old_local = {int(row["trial"]): rank for rank, row in enumerate(old_order, start=1)}
    new_local = {int(row["trial"]): rank for rank, row in enumerate(new_order, start=1)}
    d2 = sum((old_local[int(row["trial"])] - new_local[int(row["trial"])]) ** 2 for row in rows)
    return float(1.0 - 6.0 * d2 / (n * (n * n - 1)))


def _topk_jaccard(rows: list[dict[str, Any]], k: int) -> float:
    n = len(rows)
    if n == 0:
        return float("nan")
    k_eff = min(int(k), n)
    old = {int(row["trial"]) for row in sorted(rows, key=lambda r: int(r["old_rank"]))[:k_eff]}
    new = {int(row["trial"]) for row in sorted(rows, key=lambda r: int(r["new_rank"]))[:k_eff]}
    union = old | new
    if not union:
        return float("nan")
    return float(len(old & new) / len(union))


def _write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], prefix: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{prefix}.json"
    csv_path = OUT_DIR / f"{prefix}.csv"
    payload = {"summary": summary, "rows": rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _effective_top_k(top_k: int, *, total_trial_rows: int, valid_trial_rows: int) -> int:
    """Resolve rerank scope.

    top_k = -1: default policy, analyze the top 10% of the original RS trial
    budget. For the current 500-trial KFT RS this is 50.
    top_k =  0: analyze all valid RS trials.
    top_k >  0: analyze exactly top_k valid ranked trials, clipped by valid rows.
    """

    top_k = int(top_k)
    valid_trial_rows = int(valid_trial_rows)
    if top_k == 0:
        return valid_trial_rows
    if top_k < 0:
        return min(valid_trial_rows, max(1, int(np.ceil(0.10 * int(total_trial_rows)))))
    return min(valid_trial_rows, max(1, top_k))


def _scope_name(top_k: int, *, effective_top_k: int | None = None) -> str:
    if int(top_k) == 0:
        return "all"
    if int(top_k) < 0:
        return "top10pct" if effective_top_k is None else f"top10pct_{int(effective_top_k)}"
    return f"top{int(top_k)}"


def _default_prefix(tag: str, top_k: int, *, effective_top_k: int | None = None) -> str:
    return f"kft_rs_refit_rerank_{tag}_{_scope_name(top_k, effective_top_k=effective_top_k)}"


def _shard_prefix(base_prefix: str, shard_index: int, shard_count: int) -> str:
    return f"{base_prefix}_s{int(shard_index)}of{int(shard_count)}"


def _rank_refit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        copy.pop("new_rank", None)
        copy.pop("rank_delta", None)
        copy.pop("new_rank_within_shard", None)
        copy.pop("rank_delta_within_shard", None)
        ranked.append(copy)
    ranked.sort(key=lambda r: (float(r["refit_train_loss"]), int(r["trial"])))
    for new_rank, row in enumerate(ranked, start=1):
        row["new_rank"] = int(new_rank)
        row["rank_delta"] = int(new_rank) - int(row["old_rank"])
    ranked.sort(key=lambda r: int(r["old_rank"]))
    return ranked


def _summary_for_rows(
    *,
    rows: list[dict[str, Any]],
    window_index: int,
    split,
    tag: str,
    source_trial_file: Path,
    trial_stats: dict[str, Any],
    started: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "window_index": int(window_index),
        "window_role": split.role,
        "window_label": split.label,
        "analyzed_trials": int(len(rows)),
        "source_trial_file": str(source_trial_file),
        "best_checkpoint": str(CHECKPOINT_DIR / f"kan_full_test_random_search_best_{tag}.pt"),
        "trial_file_stats": trial_stats,
        "total_sec": float(perf_counter() - started),
        "note": (
            "old rank is original RS ranking under observable-prior hidden normalizer; "
            "new rank is after per-trial rollout-based x3/q normalizer refit and re-evaluation. "
            "failed RS trials are intentionally excluded from refit re-rank."
        ),
    }
    if rows:
        failed_refit = [row for row in rows if bool(row.get("refit_failed", False))]
        summary.update(
            {
                "old_rank1_trial": int(min(rows, key=lambda r: int(r["old_rank"]))["trial"]),
                "new_rank1_trial": int(min(rows, key=lambda r: int(r["new_rank"]))["trial"]),
                "spearman_old_vs_new": _spearman_from_ranks(rows),
                "top5_jaccard": _topk_jaccard(rows, 5),
                "top10_jaccard": _topk_jaccard(rows, 10),
                "top20_jaccard": _topk_jaccard(rows, 20),
                "max_abs_rank_delta": int(max(abs(int(row["rank_delta"])) for row in rows)),
                "mean_abs_rank_delta": float(np.mean([abs(int(row["rank_delta"])) for row in rows])),
                "refit_failed_trials": int(len(failed_refit)),
                "refit_valid_trials": int(len(rows) - len(failed_refit)),
                "refit_failure_reasons": [
                    {"reason": reason, "count": int(count)}
                    for reason, count in sorted(
                        {
                            str(row.get("refit_failure_reason", "unknown")): sum(
                                1
                                for item in failed_refit
                                if str(item.get("refit_failure_reason", "unknown"))
                                == str(row.get("refit_failure_reason", "unknown"))
                            )
                            for row in failed_refit
                        }.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
            }
        )
    if extra:
        summary.update(extra)
    return summary


def _load_shard_rows(base_prefix: str, shard_count: int) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    counts: dict[str, int] = {}
    for shard_index in range(1, int(shard_count) + 1):
        shard_path = OUT_DIR / f"{_shard_prefix(base_prefix, shard_index, shard_count)}.json"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing shard output: {shard_path}")
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
        shard_rows = payload.get("rows", [])
        if not isinstance(shard_rows, list):
            raise RuntimeError(f"Invalid shard rows in {shard_path}")
        for row in shard_rows:
            if isinstance(row, dict):
                rows.append(dict(row))
        paths.append(str(shard_path))
        counts[f"s{shard_index}"] = int(len(shard_rows))
    return rows, paths, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-rank KFT random-search trials after trial-specific hidden normalizer refit."
    )
    parser.add_argument("--window-index", type=int, default=1, help="1-based KFT window index; default W0.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help=(
            "Only analyze old top K valid trials. Default -1 means top 10% of the RS trial budget "
            "(50 for 500 trials). Use 0 to analyze all valid trials."
        ),
    )
    parser.add_argument("--output-prefix", type=str, default="", help="Optional output filename prefix under logs/analysis.")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of analysis shards; default 1.")
    parser.add_argument("--shard-index", type=int, default=1, help="1-based shard index for worker mode; default 1.")
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge existing shard outputs into one global old/new rank table without re-running trial evaluations.",
    )
    args = parser.parse_args()
    if int(args.shard_count) < 1:
        raise ValueError("--shard-count must be >= 1")
    if int(args.shard_index) < 1 or int(args.shard_index) > int(args.shard_count):
        raise ValueError("--shard-index must be within 1..shard-count")

    base_cfg = default_config(REPO_ROOT)
    prepared_for_tag = prepare_data(base_cfg)
    window_index = int(args.window_index)
    if window_index < 1 or window_index > len(prepared_for_tag.splits):
        raise ValueError(f"invalid window-index={window_index}; available=1..{len(prepared_for_tag.splits)}")
    tag = _window_file_tag(prepared_for_tag.splits[window_index - 1].role)

    best_payload = _load_best_payload(tag)
    cfg = _config_from_payload(best_payload)
    prepared = prepare_data(cfg)
    split = prepared.splits[window_index - 1]
    trial_path, all_trial_rows = _load_all_trial_rows(tag)
    valid_rows = _valid_trial_rows(all_trial_rows)
    trial_stats = _trial_file_stats(all_trial_rows, valid_rows)
    rs_trial_budget = int(getattr(cfg, "random_search_trials", len(all_trial_rows)))
    effective_top_k = _effective_top_k(
        int(args.top_k),
        total_trial_rows=rs_trial_budget,
        valid_trial_rows=len(valid_rows),
    )
    prefix_base = args.output_prefix.strip() or _default_prefix(tag, int(args.top_k), effective_top_k=effective_top_k)
    rows = valid_rows[:effective_top_k]

    started = perf_counter()

    if bool(args.merge_shards):
        merged_rows_raw, shard_paths, shard_counts = _load_shard_rows(prefix_base, int(args.shard_count))
        merged_rows = _rank_refit_rows(merged_rows_raw)
        summary = _summary_for_rows(
            rows=merged_rows,
            window_index=window_index,
            split=split,
            tag=tag,
            source_trial_file=trial_path,
            trial_stats=trial_stats,
            started=started,
            extra={
                "scope": _scope_name(int(args.top_k)),
                "effective_top_k": int(effective_top_k),
                "rs_trial_budget_for_top10pct": int(rs_trial_budget),
                "top_k_policy": "top_10_percent_of_rs_trial_budget" if int(args.top_k) < 0 else ("all_valid" if int(args.top_k) == 0 else "explicit_top_k"),
                "merge_shards": True,
                "shard_count": int(args.shard_count),
                "shard_paths": shard_paths,
                "shard_row_counts": shard_counts,
            },
        )
        json_path, csv_path = _write_outputs(merged_rows, summary, f"{prefix_base}_merged")
        print("\n=== refit rerank merged summary ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
        print(f"json -> {json_path}")
        print(f"csv  -> {csv_path}")
        return

    if int(args.shard_count) > 1:
        shard_index = int(args.shard_index)
        shard_count = int(args.shard_count)
        rows = [
            row
            for pos, row in enumerate(rows, start=1)
            if ((pos - 1) % shard_count) == (shard_index - 1)
        ]

    tensors = _window_to_torch(split, dtype=_torch_dtype(cfg.dtype), device="cpu")
    out_rows: list[dict[str, Any]] = []
    dtype = _torch_dtype(cfg.dtype)
    for i, row in enumerate(rows, start=1):
        t0 = perf_counter()
        try:
            out = _evaluate_and_refit_one(cfg=cfg, prepared=prepared, split=split, tensors=tensors, row=row, dtype=dtype)
            out["refit_failed"] = False
            status = "ok"
        except Exception as exc:
            out = _failed_refit_row(row, exc)
            status = f"failed:{type(exc).__name__}"
        out["analysis_index"] = int(i)
        out["shard_index"] = int(args.shard_index)
        out["shard_count"] = int(args.shard_count)
        out["analysis_sec"] = float(perf_counter() - t0)
        out_rows.append(out)
        print(
            f"[{i}/{len(rows)}] trial={out['trial']} old_rank={out['old_rank']} "
            f"old={out['old_train_loss_recomputed']:.6e} refit={out['refit_train_loss']:.6e} "
            f"status={status} dt={out['analysis_sec']:.2f}s",
            flush=True,
        )
        if bool(out.get("refit_failed", False)):
            print(
                f"  refit failure: {out['refit_failure_type']}: {out['refit_failure_reason']}",
                flush=True,
            )

    ranked_rows = _rank_refit_rows(out_rows)
    if int(args.shard_count) > 1:
        for row in ranked_rows:
            row["new_rank_within_shard"] = int(row["new_rank"])
            row["rank_delta_within_shard"] = int(row["rank_delta"])
            row.pop("new_rank", None)
            row.pop("rank_delta", None)
    summary = _summary_for_rows(
        rows=ranked_rows if int(args.shard_count) == 1 else _rank_refit_rows(out_rows),
        window_index=window_index,
        split=split,
        tag=tag,
        source_trial_file=trial_path,
        trial_stats=trial_stats,
        started=started,
        extra={
            "scope": _scope_name(int(args.top_k)),
            "effective_top_k": int(effective_top_k),
            "rs_trial_budget_for_top10pct": int(rs_trial_budget),
            "top_k_policy": "top_10_percent_of_rs_trial_budget" if int(args.top_k) < 0 else ("all_valid" if int(args.top_k) == 0 else "explicit_top_k"),
            "merge_shards": False,
            "shard_count": int(args.shard_count),
            "shard_index": int(args.shard_index),
            "selected_valid_trials_before_sharding": int(len(rows) if int(args.shard_count) <= 1 else effective_top_k),
            "worker_trials": int(len(out_rows)),
        },
    )
    prefix = prefix_base
    if int(args.shard_count) > 1:
        prefix = _shard_prefix(prefix_base, int(args.shard_index), int(args.shard_count))
    json_path, csv_path = _write_outputs(ranked_rows, summary, prefix)

    print("\n=== refit rerank summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    print(f"json -> {json_path}")
    print(f"csv  -> {csv_path}")


if __name__ == "__main__":
    main()
