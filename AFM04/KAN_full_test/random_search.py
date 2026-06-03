"""Random-search front stage for AFM04 KAN full test."""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from AFM04.KAN_full_test.data import WindowSplit, prepare_data, x3dot_init_from_split, x3dot_scale_from_split
from AFM04.KAN_full_test.kan_backend import KANForceModule
from AFM04.KAN_full_test.losses import evaluate_split
from AFM04.KAN_full_test.rollout import StateGuardTriggered


def _torch_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def _role_short(role: str) -> str:
    role_norm = role.strip().lower()
    if role_norm == "modified_w0":
        return "modified W0"
    if role_norm == "first_contact":
        return "W0"
    if role_norm == "middle":
        return "W1"
    if role_norm == "max_x1_pp_change":
        return "W2"
    if role_norm == "tail_stable":
        return "W3"
    return role


def _window_title(role: str) -> str:
    role_norm = role.strip().lower()
    if role_norm == "modified_w0":
        return "modified W0"
    if role_norm == "first_contact":
        return "W0: first-contact window"
    if role_norm == "middle":
        return "W1: middle window"
    if role_norm == "max_x1_pp_change":
        return "window2: the most drastic region"
    if role_norm == "tail_stable":
        return "window3: stable region at the end"
    return role


def _window_file_tag(role: str) -> str:
    return _role_short(role).strip().lower().replace(" ", "_")


def _timestamp_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_line(log, message: str) -> None:
    line = f"[{_timestamp_human()}] {message}"
    log.write(line + "\n")
    log.flush()


def _window_to_torch(split: WindowSplit, *, dtype: torch.dtype, device: str) -> dict[str, torch.Tensor]:
    return {
        "ode_full": torch.as_tensor(split.ode_full, dtype=dtype, device=device),
        "ode_train": torch.as_tensor(split.ode_train, dtype=dtype, device=device),
        "ode_val": torch.as_tensor(split.ode_val, dtype=dtype, device=device),
        "x2dot_full": torch.as_tensor(split.x2dot_full, dtype=dtype, device=device),
        "x2dot_train": torch.as_tensor(split.x2dot_train, dtype=dtype, device=device),
        "x2dot_val": torch.as_tensor(split.x2dot_val, dtype=dtype, device=device),
        "contact_full": torch.as_tensor(split.contact_full.astype(float), dtype=dtype, device=device),
        "contact_train": torch.as_tensor(split.contact_train.astype(float), dtype=dtype, device=device),
        "contact_val": torch.as_tensor(split.contact_val.astype(float), dtype=dtype, device=device),
        "times_full": torch.as_tensor(split.times_full, dtype=dtype, device=device),
        "times_train": torch.as_tensor(split.times_train, dtype=dtype, device=device),
        "times_val": torch.as_tensor(split.times_val, dtype=dtype, device=device),
        "train_idx": torch.as_tensor(split.train_idx, dtype=torch.long, device=device),
        "val_idx": torch.as_tensor(split.val_idx, dtype=torch.long, device=device),
    }


def _rs_observable_grid_inputs(
    *,
    split: WindowSplit,
    prepared,
    cfg,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build RS AGU samples in raw physical coordinates.

    x1/x2 come from the observed training trajectory.  Hidden axes use fixed
    neutral coordinates, not true/predicted x3 or true/predicted x3dot samples.
    """
    n = int(split.ode_train.shape[1])
    input_dim = int(getattr(cfg, "width", (3, 7, 1))[0])
    inputs = np.empty((n, input_dim), dtype=float)
    inputs[:, 0:2] = np.asarray(split.ode_train[0:2, :], dtype=float).T

    if input_dim >= 3:
        x1_abs = float(np.max(np.abs(np.asarray(split.ode_train[0, :], dtype=float)))) if n > 0 else 0.0
        x3_amp = max(0.1 * x1_abs, 1.0e-12)
        if n <= 1:
            inputs[:, 2] = 0.0
        else:
            frac = np.linspace(-1.0, 1.0, n, dtype=float)
            inputs[:, 2] = x3_amp * frac

    if input_dim >= 4:
        x2_abs = float(np.max(np.abs(np.asarray(split.ode_train[1, :], dtype=float)))) if n > 0 else 0.0
        q_amp = max(0.1 * x2_abs, 1.0e-12)
        if n <= 1:
            inputs[:, 3] = 0.0
        else:
            frac = np.mod((np.arange(n, dtype=float) + 0.5) * 0.6180339887498949, 1.0)
            inputs[:, 3] = q_amp * (2.0 * frac - 1.0)

    for axis in range(4, input_dim):
        if n <= 1:
            inputs[:, axis] = 0.0
        else:
            idx = np.arange(n, dtype=float)
            frac = np.mod((idx + 0.5 + 0.137 * axis) * 0.6180339887498949, 1.0)
            inputs[:, axis] = 2.0 * frac - 1.0
    return torch.as_tensor(inputs, dtype=dtype, device=cfg.device)


def _trial_seed(base_seed: int, trial_index: int) -> int:
    return int(base_seed + trial_index)


def _trial_seed_bank_path(cfg) -> Path:
    return cfg.random_search_result_dir / "kan_full_test_random_search_trial_seeds.json"


def _generate_trial_seed_bank(cfg) -> list[int]:
    trial_count = int(cfg.random_search_trials)
    if trial_count <= 0:
        return []
    if not bool(getattr(cfg, "random_search_true_random", True)):
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


def write_random_search_seed_bank(cfg) -> dict[str, Any]:
    path = _trial_seed_bank_path(cfg)
    seeds = _generate_trial_seed_bank(cfg)
    payload = {
        "trial_count": int(cfg.random_search_trials),
        "true_random": bool(getattr(cfg, "random_search_true_random", True)),
        "generated_at": _timestamp_human(),
        "trial_seeds": [int(seed) for seed in seeds],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    tmp.replace(path)
    payload["path"] = str(path)
    return payload


def load_random_search_seed_bank(cfg, *, create_if_missing: bool = False) -> list[int]:
    path = _trial_seed_bank_path(cfg)
    if not path.is_file():
        if not create_if_missing:
            raise FileNotFoundError(f"Missing random-search trial seed bank: {path}")
        write_random_search_seed_bank(cfg)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    seeds = payload.get("trial_seeds", [])
    if not isinstance(seeds, list):
        raise RuntimeError(f"Invalid trial seed bank format: {path}")
    trial_count = int(payload.get("trial_count", len(seeds)))
    if trial_count != int(cfg.random_search_trials) or len(seeds) != int(cfg.random_search_trials):
        raise RuntimeError(
            "Random-search trial seed bank size mismatch: "
            f"path={path} file_trials={trial_count} seeds={len(seeds)} cfg_trials={cfg.random_search_trials}"
        )
    return [int(seed) for seed in seeds]


def _save_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _topk_insert(topk: list[dict[str, Any]], row: dict[str, Any], k: int) -> None:
    topk.append(dict(row))
    topk.sort(key=lambda item: (float(item["ranking_loss"]), float(item["train_loss"]), int(item["trial"])))
    del topk[k:]


def _pct(part: float, total: float) -> float:
    total = float(total)
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    return 100.0 * float(part) / total


def _fmt_optional_loss(value: float) -> str:
    return "NA" if not np.isfinite(float(value)) else f"{float(value):.6e}"


def _fmt_optional_pct(value: float) -> str:
    return "NA" if not np.isfinite(float(value)) else f"{float(value):.2f}%"


def _min_finite_or_nan(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(min(finite)) if finite else float("nan")


def _format_trial_failure(exc: Exception) -> str:
    raw = f"{type(exc).__name__}: {exc}"
    if "max_num_steps exceeded" in raw:
        return f"ODE step-budget guard: {raw}"
    return raw


def _maxabs_scale_torch(values: torch.Tensor, mean: torch.Tensor, *, floor: float = 1.0e-12) -> float:
    if values.numel() <= 0:
        return float(floor)
    scale = float(torch.max(torch.abs(values - mean)).detach().cpu())
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(floor)
    return max(scale, float(floor))


def _rs_rollout_hidden_normalizer_stats(
    *,
    traj: torch.Tensor | None,
    train_idx: torch.Tensor,
) -> dict[str, Any]:
    """Summarize hidden input axes from the RS train rollout.

    These are predicted-rollout statistics, not true hidden-state statistics.
    They let later refit-rerank reuse the RS rollout information without
    replaying the first rollout just to recover x3/q coordinates.
    """

    out: dict[str, Any] = {
        "rs_rollout_stats_available": False,
        "rs_rollout_stats_source": "unavailable",
        "rs_rollout_stats_points": 0,
        "rs_rollout_x3_mean": float("nan"),
        "rs_rollout_x3_scale": float("nan"),
        "rs_rollout_q_mean": float("nan"),
        "rs_rollout_q_scale": float("nan"),
        "rs_rollout_q_available": False,
        "rs_rollout_accepted_history_count": 0,
    }
    if traj is None:
        return out
    idx = train_idx.to(device=traj.device, dtype=torch.long).reshape(-1)
    if int(idx.numel()) <= 0:
        return out
    with torch.no_grad():
        x3_train = traj[2, idx]
        x3_mean_t = torch.mean(x3_train)
        out.update(
            {
                "rs_rollout_stats_available": True,
                "rs_rollout_stats_source": "rs_train_rollout_predicted",
                "rs_rollout_stats_points": int(idx.numel()),
                "rs_rollout_x3_mean": float(x3_mean_t.detach().cpu()),
                "rs_rollout_x3_scale": _maxabs_scale_torch(x3_train, x3_mean_t),
            }
        )
        q_pre = getattr(traj, "_plan_z_q_pre", None)
        if q_pre is not None:
            q_train = q_pre[idx]
            q_mean_t = torch.mean(q_train)
            out.update(
                {
                    "rs_rollout_q_available": True,
                    "rs_rollout_q_mean": float(q_mean_t.detach().cpu()),
                    "rs_rollout_q_scale": _maxabs_scale_torch(q_train, q_mean_t),
                    "rs_rollout_accepted_history_count": int(
                        getattr(traj, "_plan_z_accepted_history_count", 0)
                    ),
                }
            )
    return out


def _row_ranking_loss(row: dict[str, Any]) -> float:
    return float(row.get("ranking_loss", row.get("val_loss", row.get("train_loss", float("inf")))))


def _row_sort_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        _row_ranking_loss(row),
        float(row.get("train_loss", float("inf"))),
        int(row.get("trial", -1)),
    )


def _row_is_valid(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    if bool(row.get("failed", False)):
        return False
    return np.isfinite(_row_ranking_loss(row))


def _metric_split_name(use_val: bool) -> str:
    return "val" if use_val else "train"


def _selected_window_indices(cfg, splits: tuple[WindowSplit, ...]) -> list[int]:
    requested = int(cfg.random_search_window_index)
    if requested <= 0:
        return list(range(1, len(splits) + 1))
    if requested > len(splits):
        raise ValueError(
            f"invalid random_search_window_index={requested}; available windows=1..{len(splits)}"
        )
    return [requested]


def _rebuild_trial_state_dict(
    *,
    cfg,
    prepared,
    split: WindowSplit,
    seed: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
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
        device=cfg.device,
        dtype=dtype,
    ).to(cfg.device)
    train_states = torch.as_tensor(split.ode_train.T, dtype=dtype, device=cfg.device)
    grid_inputs = _rs_observable_grid_inputs(split=split, prepared=prepared, cfg=cfg, dtype=dtype)
    train_fts = torch.as_tensor(split.fts_train_true, dtype=dtype, device=cfg.device)
    with torch.no_grad():
        if cfg.adaptive_grid_enabled:
            model.update_grid_from_inputs(grid_inputs)
        model.initialize_gain_from_truth(train_states, train_fts)
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def run_random_search_shard(
    cfg=None,
    *,
    subshard_index: int | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    from AFM04.KAN_full_test.config import default_config

    cfg = default_config() if cfg is None else cfg
    subshard_index = int(cfg.random_search_subshard_index if subshard_index is None else subshard_index)
    dtype = _torch_dtype(cfg.dtype)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    trial_seeds = load_random_search_seed_bank(
        cfg,
        create_if_missing=(int(cfg.random_search_subshard_count) == 1),
    )

    prepared = prepare_data(cfg)
    selected_window_indices = _selected_window_indices(cfg, prepared.splits)
    if subshard_index < 1 or subshard_index > cfg.random_search_subshard_count:
        raise ValueError(
            f"invalid subshard_index={subshard_index}, subshard_count={cfg.random_search_subshard_count}"
        )

    mech_true_t = torch.as_tensor(prepared.mech_true, dtype=dtype, device=cfg.device)
    window_ctxs: dict[int, dict[str, Any]] = {}
    for window_index in selected_window_indices:
        split = prepared.splits[window_index - 1]
        tensors = _window_to_torch(split, dtype=dtype, device=cfg.device)
        train_states = torch.as_tensor(split.ode_train.T, dtype=dtype, device=cfg.device)
        grid_inputs = _rs_observable_grid_inputs(split=split, prepared=prepared, cfg=cfg, dtype=dtype)
        train_fts = torch.as_tensor(split.fts_train_true, dtype=dtype, device=cfg.device)
        train_max_abs_x1 = float(torch.max(torch.abs(tensors["ode_full"][0, :])).detach().cpu())
        train_max_abs_x2 = float(torch.max(torch.abs(tensors["ode_full"][1, :])).detach().cpu())
        x1_abs_guard = (
            float(cfg.random_search_x1_guard_mult) * train_max_abs_x1
            if cfg.random_search_fail_fast_enabled and np.isfinite(train_max_abs_x1) and train_max_abs_x1 > 0.0
            else None
        )
        x2_abs_guard = (
            float(cfg.random_search_x2_guard_mult) * train_max_abs_x2
            if cfg.random_search_fail_fast_enabled and np.isfinite(train_max_abs_x2) and train_max_abs_x2 > 0.0
            else None
        )
        window_ctxs[window_index] = {
            "split": split,
            "tag": _window_file_tag(split.role),
            "label": _role_short(split.role),
            "tensors": tensors,
            "train_states": train_states,
            "grid_inputs": grid_inputs,
            "train_fts": train_fts,
            "x1_abs_guard": x1_abs_guard,
            "x2_abs_guard": x2_abs_guard,
        }

    log_path = (
        cfg.random_search_shard_log_dir / f"log2_04_step2a_kan_full_test_random_search_local_s{subshard_index}.txt"
        if log_path is None
        else Path(log_path)
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ranking_metric = "val_loss" if cfg.random_search_use_val else "train_loss"
    metric_split = _metric_split_name(cfg.random_search_use_val)
    trial_rows_by_window: dict[int, list[dict[str, Any]]] = {window_index: [] for window_index in selected_window_indices}
    topk_by_window: dict[int, list[dict[str, Any]]] = {window_index: [] for window_index in selected_window_indices}
    best_row_by_window: dict[int, dict[str, Any] | None] = {window_index: None for window_index in selected_window_indices}
    completed_trials_by_window: dict[int, int] = {window_index: 0 for window_index in selected_window_indices}
    total_tasks = len(selected_window_indices) * cfg.random_search_trials
    task_indices = list(range(subshard_index, total_tasks + 1, cfg.random_search_subshard_count))
    local_completed = 0

    with log_path.open("w", encoding="utf-8") as log:
        _log_line(
            log,
            "KAN random search start | "
            f"compute_shard={subshard_index}/{cfg.random_search_subshard_count} "
            f"| windows={len(selected_window_indices)} selected={selected_window_indices}",
        )
        _log_line(
            log,
            "KAN random search config: "
            f"trials={cfg.random_search_trials} width={list(cfg.width)} grid={cfg.grid} "
            f"k={cfg.spline_k} base={cfg.base_fun} noise_scale={cfg.noise_scale} "
            f"adaptive_grid={'ON' if cfg.adaptive_grid_enabled else 'OFF'} "
            f"w_pred={'ON' if cfg.wpred_enabled else 'OFF'} eps={cfg.wpred_eps:.2e} "
            f"soft_mask={'ON' if cfg.soft_mask_enabled else 'OFF'} "
            f"soft_mask_trainable={'ON' if cfg.soft_mask_trainable else 'OFF'} "
            f"s0={cfg.soft_mask_s0_a0:.3g}a0 alpha={cfg.soft_mask_alpha_a0:.3g}/a0 m_min=0 "
            f"x3dot_input={'ON' if cfg.x3dot_input_enabled else 'OFF'} "
            f"x3dot_mode={cfg.x3dot_input_mode} "
            f"q_init_policy={cfg.x3dot_init_policy} "
            f"q_init_trainable=OFF "
            "q_coordinate=raw_physical "
            f"q_lag_detach={'ON' if cfg.x3dot_lag_detach else 'OFF'} "
            f"ode_step_budget_guard={cfg.random_search_ode_step_budget if cfg.random_search_ode_step_budget > 0 else 'OFF'} "
            f"val_eval={'ON' if cfg.random_search_use_val else 'OFF'} "
            f"ranking={ranking_metric} metrics={metric_split} "
            f"seed_mode={'true-random' if cfg.random_search_true_random else 'deterministic-bank'}",
        )
        _log_line(log, f"mech known: ks={prepared.mech_true[0]:.6e} cs={prepared.mech_true[1]:.6e}")
        for window_index, ctx in window_ctxs.items():
            split = ctx["split"]
            _log_line(
                log,
                f"{ctx['label']} meta: "
                f"label={split.label} | role={split.role} | "
                f"t_us=[{split.t_start * 1.0e6:.9f}, {split.t_stop * 1.0e6:.9f}] | "
                f"q_init={x3dot_init_from_split(split, policy=cfg.x3dot_init_policy, fallback=cfg.x3dot_init_value):.6e} | "
                f"x1_guard={_fmt_optional_loss(ctx['x1_abs_guard'] if ctx['x1_abs_guard'] is not None else float('nan'))} "
                f"x2_guard={_fmt_optional_loss(ctx['x2_abs_guard'] if ctx['x2_abs_guard'] is not None else float('nan'))}"
            )

        for task_index in task_indices:
            local_completed += 1
            zero_based = task_index - 1
            selected_count = len(selected_window_indices)
            trial_index = int(zero_based // selected_count) + 1
            window_index = int(selected_window_indices[zero_based % selected_count])
            ctx = window_ctxs[window_index]
            split = ctx["split"]
            window_tag = str(ctx["tag"])
            window_label = str(ctx["label"])
            tensors = ctx["tensors"]
            train_states = ctx["train_states"]
            grid_inputs = ctx["grid_inputs"]
            train_fts = ctx["train_fts"]
            x1_abs_guard = ctx["x1_abs_guard"]
            x2_abs_guard = ctx["x2_abs_guard"]

            seed = int(trial_seeds[trial_index - 1])
            t0 = perf_counter()
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
                device=cfg.device,
                dtype=dtype,
            ).to(cfg.device)
            t1 = perf_counter()
            with torch.no_grad():
                if cfg.adaptive_grid_enabled:
                    model.update_grid_from_inputs(grid_inputs)
                t2 = perf_counter()
                init_gnn = model.initialize_gain_from_truth(train_states, train_fts)
                t3 = perf_counter()
            t4 = t3
            t5 = t3
            failure_reason: str | None = None
            train_traj = None
            try:
                with torch.no_grad():
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
                        x1_abs_guard=x1_abs_guard,
                        x2_abs_guard=x2_abs_guard,
                        ode_step_budget=cfg.random_search_ode_step_budget,
                        loss_indices=tensors["train_idx"],
                    )
                    t4 = perf_counter()
                    if cfg.random_search_use_val:
                        val_total, val_parts, _ = evaluate_split(
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
                            x1_abs_guard=x1_abs_guard,
                            x2_abs_guard=x2_abs_guard,
                            ode_step_budget=cfg.random_search_ode_step_budget,
                            loss_indices=tensors["val_idx"],
                        )
                    else:
                        val_total = None
                        val_parts = None
                    t5 = perf_counter()
            except StateGuardTriggered as exc:
                t_fail = perf_counter()
                if t4 <= t3:
                    t4 = t_fail
                    t5 = t_fail
                else:
                    t5 = t_fail
                failure_reason = str(exc)
                train_total = None
                train_parts = None
                val_total = None
                val_parts = None
            except Exception as exc:
                t_fail = perf_counter()
                if t4 <= t3:
                    t4 = t_fail
                    t5 = t_fail
                else:
                    t5 = t_fail
                failure_reason = _format_trial_failure(exc)
                train_total = None
                train_parts = None
                val_total = None
                val_parts = None

            dt_init = float(t1 - t0)
            dt_grid = float(t2 - t1)
            dt_gnn = float(t3 - t2)
            dt_train = float(t4 - t3)
            dt_val = float(t5 - t4)
            dt_total = float(t5 - t0)
            train_loss = float(train_total.detach()) if train_total is not None else float("inf")
            val_loss = float(val_total.detach()) if val_total is not None else float("nan")
            ranking_loss = train_loss if not cfg.random_search_use_val else (val_loss if np.isfinite(val_loss) else float("inf"))
            x1_rec = float(val_parts.x1_rec) if cfg.random_search_use_val and val_parts is not None else (
                float(train_parts.x1_rec) if train_parts is not None else float("nan")
            )
            x3_rec = float(val_parts.x3_rec) if cfg.random_search_use_val and val_parts is not None else (
                float(train_parts.x3_rec) if train_parts is not None else float("nan")
            )
            nn_err = float(val_parts.fts_rollout_rec) if cfg.random_search_use_val and val_parts is not None else (
                float(train_parts.fts_rollout_rec) if train_parts is not None else float("nan")
            )
            rollout_hidden_stats = _rs_rollout_hidden_normalizer_stats(
                traj=train_traj if train_parts is not None else None,
                train_idx=tensors["train_idx"],
            )

            row = {
                "window_index": window_index,
                "trial": trial_index,
                "seed": seed,
                "ranking_metric": ranking_metric,
                "metric_split": metric_split,
                "ranking_loss": ranking_loss,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "failed": failure_reason is not None,
                "failure_reason": failure_reason,
                "x1_rec": x1_rec,
                "x3_rec": x3_rec,
                "nn_err": nn_err,
                "train_x1_rec": float(train_parts.x1_rec) if train_parts is not None else float("nan"),
                "train_x3_rec": float(train_parts.x3_rec) if train_parts is not None else float("nan"),
                "train_nn_err": float(train_parts.fts_rollout_rec) if train_parts is not None else float("nan"),
                "val_x1_rec": float(val_parts.x1_rec) if val_parts is not None else float("nan"),
                "val_x3_rec": float(val_parts.x3_rec) if val_parts is not None else float("nan"),
                "val_nn_err": float(val_parts.fts_rollout_rec) if val_parts is not None else float("nan"),
                "g_nn": float(model.gain().detach().cpu().item()),
                "init_gnn": float(init_gnn),
                "soft_mask_enabled": bool(model.soft_mask_enabled),
                "soft_mask_trainable": bool(model.soft_mask_trainable),
                "soft_mask_s0_a0": float(model.soft_mask_summary()["s0_a0"]),
                "soft_mask_alpha_a0": float(model.soft_mask_summary()["alpha_a0"]),
                "soft_mask_m_min": 0.0,
                **rollout_hidden_stats,
                "dt_total_sec": dt_total,
                "dt_step1_model_init_sec": dt_init,
                "dt_step2_grid_update_sec": dt_grid,
                "dt_step3_gnn_init_sec": dt_gnn,
                "dt_step4_train_eval_sec": dt_train,
                "dt_step5_val_eval_sec": dt_val,
                "dt_step1_model_init_pct": _pct(dt_init, dt_total),
                "dt_step2_grid_update_pct": _pct(dt_grid, dt_total),
                "dt_step3_gnn_init_pct": _pct(dt_gnn, dt_total),
                "dt_step4_train_eval_pct": _pct(dt_train, dt_total),
                "dt_step5_val_eval_pct": _pct(dt_val, dt_total),
            }
            trial_rows_by_window[window_index].append(row)
            completed_trials_by_window[window_index] += 1
            _topk_insert(topk_by_window[window_index], row, cfg.random_search_topk)

            best_row = best_row_by_window[window_index]
            if (not row["failed"]) and (best_row is None or (row["ranking_loss"], row["train_loss"], row["trial"]) < (
                float(best_row["ranking_loss"]),
                float(best_row["train_loss"]),
                int(best_row["trial"]),
            )):
                best_row = dict(row)
                best_row_by_window[window_index] = best_row
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_path = cfg.random_search_checkpoint_dir / f"kan_full_test_random_search_best_{window_tag}_s{subshard_index}.pt"
                best_payload = {
                    "best_trial": best_row,
                    "best_state_dict": best_state_dict,
                    "window_meta": {
                        "role": split.role,
                        "label": split.label,
                        "title": _window_title(split.role),
                        "start_idx": split.start_idx,
                        "stop_idx": split.stop_idx,
                        "t_start": split.t_start,
                        "t_stop": split.t_stop,
                    },
                    "config": asdict(cfg),
                    "state_mean": prepared.state_mean,
                    "state_scale": prepared.state_scale,
                    "known_pars": prepared.known_pars,
                    "eta_star_true": prepared.eta_star_true,
                    "mech_true": prepared.mech_true,
                }
                _save_torch(best_path, best_payload)
                _log_line(
                    log,
                    f"{window_label} new best | trial={trial_index} seed={seed} rank({ranking_metric})={row['ranking_loss']:.6e} "
                    f"train={row['train_loss']:.6e} val={_fmt_optional_loss(row['val_loss'])} "
                    f"x1={_fmt_optional_pct(row['x1_rec'])} x3={_fmt_optional_pct(row['x3_rec'])} "
                    f"nn={_fmt_optional_pct(row['nn_err'])}",
                )

            best_row = best_row_by_window[window_index]
            if row["failed"]:
                _log_line(
                    log,
                    f"{window_label} trial {trial_index}/{cfg.random_search_trials} | "
                    f"rank({ranking_metric})=inf train=inf val={_fmt_optional_loss(row['val_loss'])} "
                    f"status=FAIL reason={row['failure_reason']} "
                    f"best_rank={float(best_row['ranking_loss']) if best_row is not None else float('nan'):.6e} | "
                    f"dt_total={dt_total:.3f}s | "
                    f"1:init={dt_init:.3f}s({_pct(dt_init, dt_total):.1f}%) "
                    f"2:grid={dt_grid:.3f}s({_pct(dt_grid, dt_total):.1f}%) "
                    f"3:gnn={dt_gnn:.3f}s({_pct(dt_gnn, dt_total):.1f}%) "
                    f"4:train={dt_train:.3f}s({_pct(dt_train, dt_total):.1f}%) "
                    f"5:val={dt_val:.3f}s({_pct(dt_val, dt_total):.1f}%)",
                )
            else:
                _log_line(
                    log,
                    f"{window_label} trial {trial_index}/{cfg.random_search_trials} | "
                    f"rank({ranking_metric})={row['ranking_loss']:.6e} "
                    f"train={row['train_loss']:.6e} val={_fmt_optional_loss(row['val_loss'])} "
                    f"x1={_fmt_optional_pct(row['x1_rec'])} x3={_fmt_optional_pct(row['x3_rec'])} "
                    f"nn={_fmt_optional_pct(row['nn_err'])} "
                    f"best_rank={float(best_row['ranking_loss']) if best_row is not None else float('nan'):.6e} | "
                    f"dt_total={dt_total:.3f}s | "
                    f"1:init={dt_init:.3f}s({_pct(dt_init, dt_total):.1f}%) "
                    f"2:grid={dt_grid:.3f}s({_pct(dt_grid, dt_total):.1f}%) "
                    f"3:gnn={dt_gnn:.3f}s({_pct(dt_gnn, dt_total):.1f}%) "
                    f"4:train={dt_train:.3f}s({_pct(dt_train, dt_total):.1f}%) "
                    f"5:val={dt_val:.3f}s({_pct(dt_val, dt_total):.1f}%)",
                )

            if completed_trials_by_window[window_index] % cfg.random_search_checkpoint_every == 0:
                ckpt_path = (
                    cfg.random_search_checkpoint_dir
                    / f"kan_full_test_random_search_checkpoint_{window_tag}_s{subshard_index}.pt"
                )
                with ckpt_path.with_suffix(".tmp").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "completed_trials": completed_trials_by_window[window_index],
                            "window_index": window_index,
                            "subshard_index": subshard_index,
                            "best_trial": best_row_by_window[window_index],
                            "topk": topk_by_window[window_index],
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=_json_default,
                    )
                ckpt_path.with_suffix(".tmp").replace(ckpt_path)
                _log_line(log, f"{window_label} checkpoint saved | trial={trial_index} | path={ckpt_path}")

        worker_summary_windows: list[dict[str, Any]] = []
        for window_index in selected_window_indices:
            ctx = window_ctxs[window_index]
            split = ctx["split"]
            window_tag = str(ctx["tag"])
            window_label = str(ctx["label"])
            history_path = (
                cfg.random_search_result_dir
                / f"kan_full_test_random_search_trials_{window_tag}_s{subshard_index}.json"
            )
            summary_path = (
                cfg.random_search_result_dir
                / f"kan_full_test_random_search_summary_{window_tag}_s{subshard_index}.json"
            )
            best_path = (
                cfg.random_search_checkpoint_dir
                / f"kan_full_test_random_search_best_{window_tag}_s{subshard_index}.pt"
            )
            rows = sorted(trial_rows_by_window[window_index], key=lambda item: int(item["trial"]))
            with history_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False, default=_json_default)

            summary = {
                "window_index": window_index,
                "subshard_index": subshard_index,
                "subshard_count": cfg.random_search_subshard_count,
                "window_meta": {
                    "role": split.role,
                    "label": split.label,
                    "title": _window_title(split.role),
                    "start_idx": split.start_idx,
                    "stop_idx": split.stop_idx,
                    "t_start": split.t_start,
                    "t_stop": split.t_stop,
                },
                "trials": cfg.random_search_trials,
                "completed_trials": completed_trials_by_window[window_index],
                "ranking_metric": ranking_metric,
                "best_trial": best_row_by_window[window_index],
                "topk": topk_by_window[window_index],
                "history_path": str(history_path),
                "best_path": str(best_path),
            }
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)
            _log_line(
                log,
                f"{window_label} random search done | completed_trials={completed_trials_by_window[window_index]} "
                f"best_trial={int(best_row_by_window[window_index]['trial']) if best_row_by_window[window_index] else -1} "
                f"best_rank({ranking_metric})="
                f"{float(best_row_by_window[window_index]['ranking_loss']) if best_row_by_window[window_index] else float('nan'):.6e}",
            )
            worker_summary_windows.append(
                {
                    "window_index": window_index,
                    "completed_trials": completed_trials_by_window[window_index],
                    "best_trial": None if best_row_by_window[window_index] is None else int(best_row_by_window[window_index]["trial"]),
                }
            )

    return {
        "subshard_index": subshard_index,
        "compute_shard_count": cfg.random_search_subshard_count,
        "windows": worker_summary_windows,
    }


def merge_random_search_results(cfg=None) -> dict[str, Any]:
    from AFM04.KAN_full_test.config import default_config

    cfg = default_config() if cfg is None else cfg
    prepared = prepare_data(cfg)
    dtype = _torch_dtype(cfg.dtype)
    selected_window_indices = _selected_window_indices(cfg, prepared.splits)
    merged: list[dict[str, Any]] = []
    per_window: dict[int, dict[str, Any]] = {}
    ranking_metric_global = ""

    for window_index in selected_window_indices:
        split = prepared.splits[window_index - 1]
        window_tag = _window_file_tag(split.role)
        canonical_history_path = cfg.random_search_result_dir / f"kan_full_test_random_search_trials_{window_tag}.json"
        canonical_summary_path = cfg.random_search_result_dir / f"kan_full_test_random_search_summary_{window_tag}.json"
        canonical_best_path = cfg.random_search_checkpoint_dir / f"kan_full_test_random_search_best_{window_tag}.pt"

        partial_summaries: list[dict[str, Any]] = []
        merged_records: list[dict[str, Any]] = []
        topk: list[dict[str, Any]] = []
        ranking_metric = ""
        meta: dict[str, Any] = {}

        for subshard_index in range(1, cfg.random_search_subshard_count + 1):
            partial_summary_path = (
                cfg.random_search_result_dir
                / f"kan_full_test_random_search_summary_{window_tag}_s{subshard_index}.json"
            )
            if not partial_summary_path.is_file():
                raise FileNotFoundError(f"Missing random-search partial summary: {partial_summary_path}")
            with partial_summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            partial_summaries.append(summary)

            summary_metric = str(summary.get("ranking_metric", "val_loss")).strip() or "val_loss"
            if ranking_metric == "":
                ranking_metric = summary_metric
            elif summary_metric != ranking_metric:
                raise RuntimeError(
                    f"Inconsistent ranking_metric for window {window_index}: {ranking_metric!r} vs {summary_metric!r}"
                )

            if not meta:
                meta = dict(summary.get("window_meta") or {})

            history_path = Path(str(summary.get("history_path", "")))
            if not history_path.is_file():
                raise FileNotFoundError(f"Missing random-search partial history: {history_path}")
            with history_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                raise RuntimeError(f"Random-search partial history is not a list: {history_path}")
            for record in records:
                merged_records.append(dict(record))
                _topk_insert(topk, record, cfg.random_search_topk)

        merged_records.sort(key=lambda item: int(item.get("trial", -1)))
        rows_by_trial = {int(item.get("trial", -1)): dict(item) for item in merged_records}
        window_best_row = None
        for item in merged_records:
            if not _row_is_valid(item):
                continue
            if window_best_row is None or _row_sort_key(item) < _row_sort_key(window_best_row):
                window_best_row = dict(item)

        with canonical_history_path.open("w", encoding="utf-8") as f:
            json.dump(merged_records, f, indent=2, ensure_ascii=False, default=_json_default)

        per_window[window_index] = {
            "canonical_history_path": canonical_history_path,
            "canonical_summary_path": canonical_summary_path,
            "canonical_best_path": canonical_best_path,
            "window_index": window_index,
            "window_meta": meta,
            "ranking_metric": ranking_metric or "val_loss",
            "merged_records": merged_records,
            "rows_by_trial": rows_by_trial,
            "topk": topk,
            "window_best_trial": window_best_row,
            "partial_summaries": partial_summaries,
        }
        if ranking_metric_global == "":
            ranking_metric_global = ranking_metric or "val_loss"
        elif ranking_metric_global != (ranking_metric or "val_loss"):
            raise RuntimeError(
                f"Inconsistent ranking_metric across windows: {ranking_metric_global!r} vs {(ranking_metric or 'val_loss')!r}"
            )

    if len(selected_window_indices) == 1:
        source_window_index = int(selected_window_indices[0])
        source_window_info = per_window[source_window_index]
        selected_row = source_window_info["window_best_trial"]
        if not _row_is_valid(selected_row):
            raise RuntimeError(f"No valid random-search best trial found for window {source_window_index}")
        common_best = {
            "trial": int(selected_row["trial"]),
            "rows": [dict(selected_row)],
            "mean_ranking_loss": float(_row_ranking_loss(selected_row)),
            "mean_train_loss": float(selected_row.get("train_loss", float("inf"))),
        }
        common_trial_index = int(selected_row["trial"])
        target_window_indices = list(selected_window_indices)
        common_rows_by_window = {window_index: dict(selected_row) for window_index in target_window_indices}
        selection_mode = "source_window_best_ranking_loss"
    else:
        source_window_index = 0
        common_candidates: list[dict[str, Any]] = []
        for trial_index in range(1, cfg.random_search_trials + 1):
            rows = []
            valid = True
            for window_index in selected_window_indices:
                row = per_window[window_index]["rows_by_trial"].get(trial_index)
                if not _row_is_valid(row):
                    valid = False
                    break
                rows.append(dict(row))
            if not valid:
                continue
            mean_ranking_loss = float(np.mean([_row_ranking_loss(row) for row in rows]))
            mean_train_loss = float(np.mean([float(row.get("train_loss", float("inf"))) for row in rows]))
            common_candidates.append(
                {
                    "trial": int(trial_index),
                    "rows": rows,
                    "mean_ranking_loss": mean_ranking_loss,
                    "mean_train_loss": mean_train_loss,
                }
            )

        if not common_candidates:
            raise RuntimeError("No valid common random-search trial found across all windows")

        common_best = min(
            common_candidates,
            key=lambda item: (float(item["mean_ranking_loss"]), float(item["mean_train_loss"]), int(item["trial"])),
        )
        common_trial_index = int(common_best["trial"])
        common_rows_by_window = {
            int(row["window_index"]): dict(row)
            for row in common_best["rows"]
        }
        target_window_indices = list(selected_window_indices)
        selection_mode = "common_trial_mean_ranking_loss"

    for window_index in target_window_indices:
        if window_index in per_window:
            window_info = per_window[window_index]
        else:
            split = prepared.splits[window_index - 1]
            source_window_info = per_window[source_window_index]
            window_tag = _window_file_tag(split.role)
            canonical_history_path = cfg.random_search_result_dir / f"kan_full_test_random_search_trials_{window_tag}.json"
            with canonical_history_path.open("w", encoding="utf-8") as f:
                json.dump(
                    source_window_info["merged_records"],
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                )
            canonical_summary_path = cfg.random_search_result_dir / f"kan_full_test_random_search_summary_{window_tag}.json"
            canonical_best_path = cfg.random_search_checkpoint_dir / f"kan_full_test_random_search_best_{window_tag}.pt"
            window_info = {
                "canonical_history_path": canonical_history_path,
                "canonical_summary_path": canonical_summary_path,
                "canonical_best_path": canonical_best_path,
                "window_index": window_index,
                "window_meta": {
                    "role": split.role,
                    "label": split.label,
                    "title": split.label,
                },
                "ranking_metric": ranking_metric_global or "val_loss",
                "merged_records": source_window_info["merged_records"],
                "rows_by_trial": source_window_info["rows_by_trial"],
                "topk": source_window_info["topk"],
                "window_best_trial": None,
                "partial_summaries": source_window_info["partial_summaries"],
            }
        split = prepared.splits[window_index - 1]
        selected_row = common_rows_by_window[window_index]
        canonical_best_path = window_info["canonical_best_path"]
        canonical_summary_path = window_info["canonical_summary_path"]

        best_state_dict = _rebuild_trial_state_dict(
            cfg=cfg,
            prepared=prepared,
            split=split,
            seed=int(selected_row.get("seed", cfg.seed + common_trial_index)),
            dtype=dtype,
        )
        best_payload = {
            "best_trial": dict(selected_row),
            "best_state_dict": best_state_dict,
            "window_meta": dict(window_info["window_meta"]),
            "config": asdict(cfg),
            "state_mean": prepared.state_mean,
            "state_scale": prepared.state_scale,
            "known_pars": prepared.known_pars,
            "mech_true": prepared.mech_true,
            "merge_meta": {
                "window_index": window_index,
                "compute_shard_count": cfg.random_search_subshard_count,
                "selection_mode": selection_mode,
                "source_window_index": source_window_index if source_window_index > 0 else window_index,
                "common_trial": common_trial_index,
                "common_trial_mean_ranking_loss": float(common_best["mean_ranking_loss"]),
                "common_trial_mean_train_loss": float(common_best["mean_train_loss"]),
                "window_best_trial": None
                if window_info["window_best_trial"] is None
                else int(window_info["window_best_trial"]["trial"]),
            },
        }
        _save_torch(canonical_best_path, best_payload)

        canonical_summary = {
            "window_index": window_index,
            "window_meta": window_info["window_meta"],
            "trials": cfg.random_search_trials,
            "completed_trials": len(window_info["merged_records"]),
            "ranking_metric": ranking_metric_global or "val_loss",
            "selection_mode": selection_mode,
            "best_trial": dict(selected_row),
            "window_best_trial": window_info["window_best_trial"],
            "source_window_index": source_window_index if source_window_index > 0 else window_index,
            "common_trial": {
                "trial": common_trial_index,
                "mean_ranking_loss": float(common_best["mean_ranking_loss"]),
                "mean_train_loss": float(common_best["mean_train_loss"]),
            },
            "topk": window_info["topk"],
            "history_path": str(window_info["canonical_history_path"]),
            "best_path": str(canonical_best_path),
            "compute_shard_count": cfg.random_search_subshard_count,
            "partial_summaries": [
                str(
                    cfg.random_search_result_dir
                    / (
                        "kan_full_test_random_search_summary_"
                        f"{_window_file_tag(prepared.splits[(source_window_index if source_window_index > 0 else window_index) - 1].role)}"
                        f"_s{subshard_index}.json"
                    )
                )
                for subshard_index in range(1, cfg.random_search_subshard_count + 1)
            ],
        }
        with canonical_summary_path.open("w", encoding="utf-8") as f:
            json.dump(canonical_summary, f, indent=2, ensure_ascii=False, default=_json_default)

        merged.append(
            {
                "shard_index": window_index,
                "role": str(window_info["window_meta"].get("role", "")),
                "label": str(window_info["window_meta"].get("label", "")),
                "title": str(window_info["window_meta"].get("title", "")),
                "trials": int(cfg.random_search_trials),
                "ranking_metric": ranking_metric_global or "val_loss",
                "best_trial": int(selected_row.get("trial", -1)),
                "best_seed": int(selected_row.get("seed", -1)),
                "best_ranking_loss": _row_ranking_loss(selected_row),
                "best_train_loss": float(selected_row.get("train_loss", float("inf"))),
                "best_val_loss": float(selected_row.get("val_loss", float("inf"))),
                "source_window_index": source_window_index if source_window_index > 0 else window_index,
                "summary_path": str(canonical_summary_path),
            }
        )

    merged = sorted(merged, key=lambda item: item["shard_index"])
    ranking_metric = str(merged[0]["ranking_metric"]) if merged else "val_loss"
    out = {
        "total_shards": len(merged),
        "trials_per_shard": cfg.random_search_trials,
        "compute_shards": cfg.random_search_subshard_count,
        "selection_mode": selection_mode,
        "common_trial": common_trial_index,
        "common_trial_mean_ranking_loss": float(common_best["mean_ranking_loss"]),
        "common_trial_mean_train_loss": float(common_best["mean_train_loss"]),
        "ranking_metric": ranking_metric,
        "best_ranking_loss": float(min(item["best_ranking_loss"] for item in merged)),
        "best_val_loss": _min_finite_or_nan([item["best_val_loss"] for item in merged]),
        "shards": merged,
    }
    out_path = cfg.random_search_result_dir / "kan_full_test_random_search_merged_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_json_default)
    return out


__all__ = ["load_random_search_seed_bank", "merge_random_search_results", "run_random_search_shard", "write_random_search_seed_bank"]
