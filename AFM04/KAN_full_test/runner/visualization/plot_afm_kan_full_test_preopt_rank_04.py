from __future__ import annotations

import json
import pickle
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.config import default_config
from AFM04.KAN_full_test.data import (
    PreparedData,
    WindowSplit,
    prepare_data,
    x3dot_init_from_split,
    x3dot_scale_from_split,
)
from AFM04.KAN_full_test.kan_backend import KANForceModule
from AFM04.KAN_full_test.losses import evaluate_split
from AFM04.KAN_full_test.random_search import _rebuild_trial_state_dict
from AFM04.KAN_full_test.rollout import force_inputs_for_module, fts_truth_from_states_torch, x2dot_rhs_torch
from AFM04.KAN_full_test.runner.visualization._common import (
    REPO_ROOT,
    finalize_and_save,
    load_result_payloads,
    out_path,
    stage_title,
    window_title,
)


DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "visualization"
DEFAULT_OUT_FILE = "afm_param_kan_full_test_04_preopt_rank_grid.png"


def _torch_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def _window_file_tag(role: str) -> str:
    role_norm = str(role).strip().lower()
    if role_norm == "modified_w0":
        return "modified_w0"
    if role_norm == "first_contact":
        return "w0"
    if role_norm == "middle":
        return "w1"
    if role_norm == "max_x1_pp_change":
        return "w2"
    if role_norm == "tail_stable":
        return "w3"
    return role_norm.replace(" ", "_")


def _is_path_field(name: str) -> bool:
    return name.endswith("_root") or name.endswith("_dir")


def _config_from_payload(payload: dict[str, Any] | None):
    cfg = default_config(REPO_ROOT)
    raw = {} if payload is None else payload.get("config", {})
    if not isinstance(raw, dict):
        return cfg

    known_fields = {field.name for field in fields(cfg)}
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known_fields:
            continue
        if key == "width":
            updates[key] = tuple(int(v) for v in value)
        elif _is_path_field(key):
            updates[key] = Path(value)
        else:
            updates[key] = value
    return replace(cfg, **updates)


def _latest_source_payload() -> dict[str, Any] | None:
    try:
        payloads = load_result_payloads()
    except FileNotFoundError:
        payloads = []
    if payloads:
        return payloads[0]

    checkpoint_dir = REPO_ROOT / "AFM04" / "KAN_full_test" / "checkpoints" / "random_search"
    candidate_paths = [
        checkpoint_dir / "kan_full_test_random_search_best_modified_w0.pt",
        checkpoint_dir / "kan_full_test_random_search_best_w0.pt",
        checkpoint_dir / "kan_full_test_random_search_best_w1.pt",
        checkpoint_dir / "kan_full_test_random_search_best_w2.pt",
        checkpoint_dir / "kan_full_test_random_search_best_w3.pt",
    ]
    existing = [path for path in candidate_paths if path.is_file()]
    if not existing:
        return None
    latest = max(existing, key=lambda path: path.stat().st_mtime)
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unexpected checkpoint payload type: {latest}")
    return checkpoint


def _select_split(prepared: PreparedData, payload: dict[str, Any] | None, cfg) -> WindowSplit:
    meta = {} if payload is None else payload.get("window_meta", {})
    role = str(meta.get("role", "")).strip()
    label = str(meta.get("label", "")).strip()

    for split in prepared.splits:
        if role and split.role == role:
            return split
    for split in prepared.splits:
        if label and split.label == label:
            return split

    requested = int(getattr(cfg, "train_window_index", 0))
    if requested <= 0:
        requested = 1
    if requested > len(prepared.splits):
        raise ValueError(f"invalid train_window_index={requested}; available windows=1..{len(prepared.splits)}")
    return prepared.splits[requested - 1]


def _valid_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank_loss(row: dict[str, Any]) -> float:
        return float(row.get("ranking_loss", row.get("val_loss", row.get("train_loss", float("inf")))))

    valid = []
    for row in rows:
        if bool(row.get("failed", False)):
            continue
        if not np.isfinite(rank_loss(row)):
            continue
        valid.append(dict(row))
    valid.sort(key=lambda row: (rank_loss(row), float(row.get("train_loss", float("inf"))), int(row.get("trial", -1))))
    return valid


def _load_manual_trial_row(cfg, tag: str, trial: int) -> tuple[dict[str, Any], int | None]:
    trials_path = cfg.random_search_result_dir / f"kan_full_test_random_search_trials_{tag}.json"
    if not trials_path.is_file():
        raise FileNotFoundError(f"Missing random-search trial ranking file: {trials_path}")
    with trials_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise RuntimeError(f"Random-search trial ranking file is not a list: {trials_path}")

    selected = None
    for row in rows:
        if int(row.get("trial", -1)) == int(trial):
            selected = dict(row)
            break
    if selected is None:
        raise ValueError(f"Requested random-search trial {trial} was not found in {trials_path}")
    if bool(selected.get("failed", False)):
        raise ValueError(f"Requested random-search trial {trial} failed: {selected.get('failure_reason')}")

    rank = None
    for idx, row in enumerate(_valid_trial_rows([dict(row) for row in rows]), start=1):
        if int(row.get("trial", -1)) == int(trial):
            rank = idx
            break
    return selected, rank


def _load_best_checkpoint(cfg, tag: str) -> dict[str, Any]:
    best_path = cfg.random_search_checkpoint_dir / f"kan_full_test_random_search_best_{tag}.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing random-search best checkpoint for preopt visualization: {best_path}")
    payload = torch.load(best_path, map_location=cfg.device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected checkpoint payload type: {best_path}")
    state_dict = payload.get("best_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Random-search checkpoint missing best_state_dict: {best_path}")
    return payload


def _load_rank1_trial_row(cfg, tag: str) -> tuple[dict[str, Any], int] | None:
    trials_path = cfg.random_search_result_dir / f"kan_full_test_random_search_trials_{tag}.json"
    if not trials_path.is_file():
        return None
    with trials_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise RuntimeError(f"Random-search trial ranking file is not a list: {trials_path}")
    valid_rows = _valid_trial_rows([dict(row) for row in rows if isinstance(row, dict)])
    if not valid_rows:
        return None
    return valid_rows[0], 1


def _build_model(cfg, prepared: PreparedData, split: WindowSplit, dtype: torch.dtype) -> KANForceModule:
    train_wpred_enabled = bool(getattr(cfg, "train_wpred_enabled", getattr(cfg, "wpred_enabled", False)))
    return KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=cfg.seed,
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


def _load_preopt_state(
    *,
    cfg,
    prepared: PreparedData,
    split: WindowSplit,
    model: KANForceModule,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], int | None]:
    tag = _window_file_tag(split.role)
    requested_trial = int(getattr(cfg, "random_search_warmstart_trial", 0))

    if requested_trial > 0:
        selected_row, rank = _load_manual_trial_row(cfg, tag, requested_trial)
        seed = int(selected_row.get("seed", int(cfg.seed) + requested_trial))
        state_dict = _rebuild_trial_state_dict(
            cfg=cfg,
            prepared=prepared,
            split=split,
            seed=seed,
            dtype=dtype,
        )
        model.load_state_dict(state_dict)
        return selected_row, rank

    try:
        checkpoint = _load_best_checkpoint(cfg, tag)
    except FileNotFoundError:
        rank1 = _load_rank1_trial_row(cfg, tag)
        if rank1 is not None:
            selected_row, rank = rank1
            trial = int(selected_row.get("trial", 0))
            seed = int(selected_row.get("seed", int(cfg.seed) + trial))
            state_dict = _rebuild_trial_state_dict(
                cfg=cfg,
                prepared=prepared,
                split=split,
                seed=seed,
                dtype=dtype,
            )
            model.load_state_dict(state_dict)
            selected_row["preopt_source"] = "random_search_rank1_rebuilt_from_trials"
            return selected_row, rank

        seed = int(cfg.seed)
        state_dict = _rebuild_trial_state_dict(
            cfg=cfg,
            prepared=prepared,
            split=split,
            seed=seed,
            dtype=dtype,
        )
        model.load_state_dict(state_dict)
        return {
            "trial": 0,
            "seed": seed,
            "ranking_metric": "deterministic_random_init",
            "ranking_loss": float("nan"),
            "train_loss": float("nan"),
            "val_loss": float("nan"),
            "preopt_source": "deterministic_random_init_fallback",
        }, None

    model.load_state_dict(checkpoint["best_state_dict"])
    best_trial = dict(checkpoint.get("best_trial") or {})
    best_trial["preopt_source"] = "random_search_best_checkpoint"
    return best_trial, 1


def _build_preopt_snapshot(payload: dict[str, Any] | None) -> dict[str, Any]:
    cfg = _config_from_payload(payload)
    prepared = prepare_data(cfg)
    split = _select_split(prepared, payload, cfg)
    dtype = _torch_dtype(cfg.dtype)
    device = str(cfg.device)
    model = _build_model(cfg, prepared, split, dtype)
    trial_row, trial_rank = _load_preopt_state(cfg=cfg, prepared=prepared, split=split, model=model, dtype=dtype)

    ode_true = torch.as_tensor(split.ode_full, dtype=dtype, device=device)
    x2dot_true = torch.as_tensor(split.x2dot_full, dtype=dtype, device=device)
    contact_mask = torch.as_tensor(split.contact_full.astype(float), dtype=dtype, device=device)
    times = torch.as_tensor(split.times_full, dtype=dtype, device=device)
    mech_true = torch.as_tensor(prepared.mech_true, dtype=dtype, device=device)

    with torch.no_grad():
        total, parts, traj_pred = evaluate_split(
            force_module=model,
            known_pars=prepared.known_pars,
            mech_true=mech_true,
            ode_true=ode_true,
            x2dot_true=x2dot_true,
            contact_mask=contact_mask,
            times=times,
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            eta_star_true=prepared.eta_star_true,
        )
        q_pre = getattr(traj_pred, "_plan_z_q_pre", None)
        x2dot_pred = x2dot_rhs_torch(traj_pred, times, model, prepared.known_pars, q_pre=q_pre)
        teacher_states = ode_true.transpose(0, 1)
        fts_true = fts_truth_from_states_torch(
            teacher_states,
            prepared.known_pars,
            eta_star=prepared.eta_star_true,
            mech_true=mech_true,
        )
        rollout_states = traj_pred.transpose(0, 1)
        fts_pred = model(force_inputs_for_module(model, rollout_states, q_pre))

    return {
        "payload": payload,
        "cfg": cfg,
        "window_meta": {
            "role": split.role,
            "label": split.label,
            "title": window_title({"window_meta": {"role": split.role, "label": split.label}}),
            "start_idx": split.start_idx,
            "stop_idx": split.stop_idx,
            "t_start": split.t_start,
            "t_stop": split.t_stop,
        },
        "trial": trial_row,
        "trial_rank": trial_rank,
        "metrics": {
            "loss": float(total.detach()),
            "state": float(parts.state),
            "x2dot": float(parts.x2dot),
            "x1_rec": float(parts.x1_rec),
            "x3_rec": float(parts.x3_rec),
            "fts_rollout_rec": float(parts.fts_rollout_rec),
        },
        "times_us": 1.0e6 * times.detach().cpu().numpy(),
        "ode_true": ode_true.detach().cpu().numpy(),
        "traj_pred": traj_pred.detach().cpu().numpy(),
        "x2dot_true": x2dot_true.detach().cpu().numpy(),
        "x2dot_pred": x2dot_pred.detach().cpu().numpy(),
        "fts_true": fts_true.detach().cpu().numpy(),
        "fts_pred": fts_pred.detach().cpu().numpy(),
    }


def _title_prefix(snapshot: dict[str, Any]) -> str:
    payload = snapshot.get("payload")
    trial = snapshot.get("trial", {})
    meta = snapshot["window_meta"]
    if isinstance(payload, dict):
        stage = f"{stage_title(payload)}: {window_title(payload)}"
    else:
        stage = window_title({"window_meta": meta})

    trial_id = int(trial.get("trial", -1))
    seed = int(trial.get("seed", -1))
    rank = snapshot.get("trial_rank")
    source = str(trial.get("preopt_source", "")).strip()
    rank_text = f"rank={rank}" if rank is not None else "rank=NA"
    metric = str(trial.get("ranking_metric", "train_loss"))
    loss = float(trial.get("ranking_loss", trial.get("train_loss", float("nan"))))
    if source == "deterministic_random_init_fallback":
        return f"KFT pre-GBO: {stage} | deterministic random init seed={seed}"
    return f"KFT pre-GBO: {stage} | RS {rank_text} trial={trial_id} seed={seed} {metric}={loss:.3e}"


def run_one(*, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    payload = _latest_source_payload()
    snapshot = _build_preopt_snapshot(payload)
    times_us = np.asarray(snapshot["times_us"], dtype=float)
    ode_true = np.asarray(snapshot["ode_true"], dtype=float)
    traj_pred = np.asarray(snapshot["traj_pred"], dtype=float)
    x2dot_true = np.asarray(snapshot["x2dot_true"], dtype=float)
    x2dot_pred = np.asarray(snapshot["x2dot_pred"], dtype=float)
    fts_true = np.asarray(snapshot["fts_true"], dtype=float)
    fts_pred = np.asarray(snapshot["fts_pred"], dtype=float)
    title_prefix = _title_prefix(snapshot)
    metrics = snapshot["metrics"]

    fig, axes = plt.subplots(1, 5, figsize=(30, 5.2), squeeze=False)
    axes = axes[0]

    ax0 = axes[0]
    ax0.plot(times_us, fts_true, color="black", linewidth=2, label="Fts teacher true")
    ax0.plot(times_us, fts_pred, color="crimson", linewidth=2, linestyle="--", label="Fts rollout pred")
    ax0.set_title(title_prefix + f"\nFts before GBO | err={metrics['fts_rollout_rec']:.2f}%")
    ax0.set_xlabel("time (us)")
    ax0.set_ylabel("force (N)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best")

    ax1 = axes[1]
    ax1.plot(times_us, ode_true[0, :], color="black", linewidth=2, label="x1 true")
    ax1.plot(times_us, traj_pred[0, :], color="crimson", linewidth=2, linestyle="--", label="x1 pred")
    ax1.set_title(title_prefix + f"\nx1 before GBO | err={metrics['x1_rec']:.3f}%")
    ax1.set_xlabel("time (us)")
    ax1.set_ylabel("x1")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2 = axes[2]
    ax2.plot(times_us, ode_true[1, :], color="black", linewidth=2, label="x2 true")
    ax2.plot(times_us, traj_pred[1, :], color="crimson", linewidth=2, linestyle="--", label="x2 pred")
    ax2.set_title(title_prefix + "\nx2 before GBO")
    ax2.set_xlabel("time (us)")
    ax2.set_ylabel("x2")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    ax3 = axes[3]
    ax3.plot(times_us, x2dot_true, color="black", linewidth=2, label="x2dot true")
    ax3.plot(times_us, x2dot_pred, color="crimson", linewidth=2, linestyle="--", label="x2dot pred")
    ax3.set_title(title_prefix + f"\nx2dot before GBO | loss={metrics['x2dot']:.3e}")
    ax3.set_xlabel("time (us)")
    ax3.set_ylabel("x2dot")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")

    ax4 = axes[4]
    ax4.plot(times_us, ode_true[2, :], color="black", linewidth=2, label="x3 true")
    ax4.plot(times_us, traj_pred[2, :], color="crimson", linewidth=2, linestyle="--", label="x3 pred")
    ax4.set_title(title_prefix + f"\nx3 before GBO | err={metrics['x3_rec']:.2f}%")
    ax4.set_xlabel("time (us)")
    ax4.set_ylabel("x3")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="best")

    fig.suptitle(
        "AFM04 KAN full-test random-search warmstart snapshot before gradient-based optimization",
        fontsize=14,
        y=1.03,
    )
    return finalize_and_save(fig, out_path(DEFAULT_OUT_FILE, out_dir))


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_OUT_DIR
    run_one(out_dir=out_dir)


if __name__ == "__main__":
    main()
