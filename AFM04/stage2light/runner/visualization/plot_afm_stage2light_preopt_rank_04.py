from __future__ import annotations

import pickle
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.stage2light.config import default_config
from AFM04.stage2light.data import PreparedData, WindowSplit, prepare_data
from AFM04.stage2light.kan_backend import KANForceModule
from AFM04.stage2light.rollout import (
    LearnableMechModule,
    fts_truth_from_states_torch,
    rollout_single_shooting_torch,
    x2dot_rhs_torch,
)
from AFM04.stage2light.runner.visualization._common import (
    REPO_ROOT,
    finalize_and_save,
    load_result_payloads,
    out_path,
    stage_title,
    window_title,
)


DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "AFM04" / "stage2light" / "checkpoints"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "stage2light" / "logs" / "visualization"
DEFAULT_OUT_FILE = "afm_param_stage2light_04_preopt_warmstart_grid.png"
_BEST_VIZ_RE = re.compile(r"stage2light_best_p(\d+)\.viz\.pkl$")


def _role_sort_key(role: str) -> tuple[int, str]:
    role_norm = role.strip().lower()
    if role_norm == "first_contact":
        return (0, role_norm)
    if role_norm == "middle":
        return (1, role_norm)
    if role_norm == "max_x1_pp_change":
        return (2, role_norm)
    if role_norm == "tail_stable":
        return (3, role_norm)
    return (99, role_norm)


def _payload_key(payload: dict[str, Any]) -> str:
    meta = payload.get("window_meta", {})
    role = str(meta.get("role", "")).strip()
    label = str(meta.get("label", "")).strip()
    return role if role else label


def _load_checkpoint_viz_payloads(checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> list[dict[str, Any]]:
    selected: dict[int, Path] = {}
    for path in checkpoint_dir.glob("stage2light_best_p*.viz.pkl"):
        match = _BEST_VIZ_RE.match(path.name)
        if match is not None:
            selected[int(match.group(1))] = path

    payloads: list[dict[str, Any]] = []
    for shard_index in sorted(selected):
        with selected[shard_index].open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def load_available_payloads() -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for payload in _load_checkpoint_viz_payloads():
        merged[_payload_key(payload)] = payload

    try:
        result_payloads = load_result_payloads()
    except FileNotFoundError:
        result_payloads = []
    for payload in result_payloads:
        merged[_payload_key(payload)] = payload

    if not merged:
        raise FileNotFoundError("No stage2light completed results or checkpoint viz payloads found.")

    return sorted(
        merged.values(),
        key=lambda payload: _role_sort_key(str(payload.get("window_meta", {}).get("role", ""))),
    )


def _torch_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def _config_from_payload(payload: dict[str, Any]):
    cfg = default_config(REPO_ROOT)
    raw = payload.get("config", {})
    if not isinstance(raw, dict):
        return cfg

    updates: dict[str, Any] = {}
    for key in (
        "error_level",
        "auto_generate_dataset",
        "window_mode",
        "arch_window_us",
        "val_stride",
        "val_offset",
        "dtype",
        "device",
        "grid",
        "spline_k",
        "base_fun",
        "symbolic_enabled",
        "auto_save",
        "noise_scale",
        "affine_trainable",
        "grid_eps",
        "grid_range_lo",
        "grid_range_hi",
        "gnn_learnable",
        "soft_mask_enabled",
        "soft_mask_trainable",
        "soft_mask_s0_a0",
        "soft_mask_s0_min_a0",
        "soft_mask_s0_max_a0",
        "soft_mask_alpha_a0",
        "soft_mask_alpha_min_a0",
        "soft_mask_alpha_max_a0",
        "ks_lo",
        "ks_hi",
        "cs_lo",
        "cs_hi",
        "ode_method",
        "ode_rtol",
        "ode_atol",
    ):
        if key in raw:
            updates[key] = raw[key]

    if "width" in raw:
        updates["width"] = tuple(int(v) for v in raw["width"])
    if "repo_root" in raw:
        updates["repo_root"] = Path(raw["repo_root"])
    if "pykan_root" in raw:
        updates["pykan_root"] = Path(raw["pykan_root"])
    if "dataset_root" in raw:
        updates["dataset_root"] = Path(raw["dataset_root"])

    return replace(cfg, **updates)


def _select_split(prepared: PreparedData, payload: dict[str, Any]) -> WindowSplit:
    meta = payload.get("window_meta", {})
    role = str(meta.get("role", "")).strip()
    label = str(meta.get("label", "")).strip()

    for split in prepared.splits:
        if role and split.role == role:
            return split
    for split in prepared.splits:
        if label and split.label == label:
            return split
    raise KeyError(f"Could not match payload window_meta to a prepared split: {meta}")


def _build_preopt_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    warmstart = payload.get("warmstart", payload.get("prestage2_warmstart", payload.get("stage1_warmstart", {})))
    if not isinstance(warmstart, dict) or not warmstart:
        raise KeyError("Payload is missing warmstart metadata required for pre-opt visualization.")

    cfg = _config_from_payload(payload)
    prepared = prepare_data(cfg)
    split = _select_split(prepared, payload)
    dtype = _torch_dtype(cfg.dtype)
    device = str(cfg.device)

    model_seed = int(warmstart["seed"])
    ks_init = float(warmstart["ks0"])
    cs_init = float(warmstart["cs0"])

    model = KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=model_seed,
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
        soft_mask_enabled=bool(cfg.soft_mask_enabled),
        soft_mask_trainable=bool(cfg.soft_mask_trainable),
        soft_mask_s0_a0=float(cfg.soft_mask_s0_a0),
        soft_mask_s0_min_a0=float(cfg.soft_mask_s0_min_a0),
        soft_mask_s0_max_a0=float(cfg.soft_mask_s0_max_a0),
        soft_mask_alpha_a0=float(cfg.soft_mask_alpha_a0),
        soft_mask_alpha_min_a0=float(cfg.soft_mask_alpha_min_a0),
        soft_mask_alpha_max_a0=float(cfg.soft_mask_alpha_max_a0),
        gnn_learnable=cfg.gnn_learnable,
        device=device,
        dtype=dtype,
    ).to(device)
    mech_module = LearnableMechModule(
        ks_init=ks_init,
        cs_init=cs_init,
        ks_bounds=(cfg.ks_lo, cfg.ks_hi),
        cs_bounds=(cfg.cs_lo, cfg.cs_hi),
        dtype=dtype,
        device=device,
    ).to(device)

    train_states = torch.as_tensor(split.ode_train.T, dtype=dtype, device=device)
    train_fts = torch.as_tensor(split.fts_train_true, dtype=dtype, device=device)
    ode_true = torch.as_tensor(split.ode_full, dtype=dtype, device=device)
    times = torch.as_tensor(split.times_full, dtype=dtype, device=device)
    mech_true = torch.as_tensor(prepared.mech_true, dtype=dtype, device=device)

    source_detail = ""
    with torch.no_grad():
        warmstart_source = str(warmstart.get("source", "stage1")).strip().lower()
        if warmstart_source == "prest2":
            init_gain = float(model.initialize_gain_from_truth(train_states, train_fts))
            source_detail = "shown = st1pl source of selected prest2 candidate"
        else:
            init_gain = float(model.initialize_gain_from_truth(train_states, train_fts))
            source_detail = "shown = stage1pluslight warmstart"
        traj_pred = rollout_single_shooting_torch(
            force_module=model,
            known_pars=prepared.known_pars,
            mech_module=mech_module,
            u0=ode_true[:, 0],
            times=times,
            method=cfg.ode_method,
            rtol=cfg.ode_rtol,
            atol=cfg.ode_atol,
        )
        x2dot_pred = x2dot_rhs_torch(traj_pred, times, model, prepared.known_pars)
        teacher_states = ode_true.transpose(0, 1)
        fts_true = fts_truth_from_states_torch(
            teacher_states,
            prepared.known_pars,
            eta_star=prepared.eta_star_true,
            mech_true=mech_true,
        )
        rollout_states = traj_pred.transpose(0, 1)
        fts_pred = model(rollout_states)
        mech_pred = mech_module().detach().cpu().numpy()

    return {
        "times_us": 1.0e6 * times.detach().cpu().numpy(),
        "ode_true": ode_true.detach().cpu().numpy(),
        "traj_pred": traj_pred.detach().cpu().numpy(),
        "x2dot_true": np.asarray(split.x2dot_full, dtype=float),
        "x2dot_pred": x2dot_pred.detach().cpu().numpy(),
        "fts_true": fts_true.detach().cpu().numpy(),
        "fts_pred": fts_pred.detach().cpu().numpy(),
        "mech_pred": mech_pred,
        "mech_true": prepared.mech_true,
        "init_gain": init_gain,
        "warmstart": warmstart,
        "warmstart_source": warmstart_source,
        "source_detail": source_detail,
    }


def run_one(*, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    payloads = load_available_payloads()
    fig, axes = plt.subplots(len(payloads), 5, figsize=(30, 4.8 * len(payloads)), squeeze=False, sharex=False)
    row_labels: list[str] = []

    for row, payload in enumerate(payloads):
        snap = _build_preopt_snapshot(payload)
        times_us = np.asarray(snap["times_us"], dtype=float)
        ode_true = np.asarray(snap["ode_true"], dtype=float)
        traj_pred = np.asarray(snap["traj_pred"], dtype=float)
        x2dot_true = np.asarray(snap["x2dot_true"], dtype=float)
        x2dot_pred = np.asarray(snap["x2dot_pred"], dtype=float)
        fts_true = np.asarray(snap["fts_true"], dtype=float)
        fts_pred = np.asarray(snap["fts_pred"], dtype=float)
        warmstart = snap["warmstart"]
        warmstart_source = str(snap.get("warmstart_source", warmstart.get("source", "stage1"))).strip().lower()
        source_detail = str(snap.get("source_detail", "")).strip()

        if warmstart_source == "prest2" and int(warmstart.get("candidate_b", warmstart.get("candidate", 0)) or 0) > 0:
            candidate_b = int(warmstart.get("candidate_b", warmstart.get("candidate", 0)))
            rank = int(warmstart.get("source_stage1_rank", warmstart.get("original_rank", 0)) or 0)
            mech = int(warmstart.get("source_mech_winner", 0) or 0)
            parts = [f"source of prest2 candidate B={candidate_b}"]
            if rank > 0:
                parts.append(f"source st1 rank={rank}")
            if mech > 0:
                parts.append(f"mech winner={mech}")
            warmstart_label = ", ".join(parts)
        elif int(warmstart.get("mech_winner", 0) or 0) > 0:
            warmstart_label = f"stage1pluslight mech winner={int(warmstart['mech_winner'])}"
        elif int(warmstart.get("rank", 0) or 0) > 0:
            warmstart_label = f"stage1pluslight rank={int(warmstart['rank'])}"
        else:
            warmstart_label = "stage1pluslight trial"
        if source_detail:
            warmstart_label = f"{source_detail} | {warmstart_label}"
        window_label = window_title(payload)
        stage_label = stage_title(payload)
        if not window_label.startswith(f"{stage_label}:"):
            window_label = f"{stage_label}: {window_label}"
        row_labels.append(f"{window_label} | {warmstart_label} | trial={int(warmstart['trial_id'])}")

        ax0 = axes[row, 0]
        ax0.plot(times_us, fts_true, color="black", linewidth=2, label="Fts teacher true")
        ax0.plot(times_us, fts_pred, color="crimson", linewidth=2, linestyle="--", label="Fts rollout pred")
        ax0.set_title("Fts after st1pl")
        ax0.set_xlabel("time (us)")
        ax0.set_ylabel("force (N)")
        ax0.grid(True, alpha=0.25)
        ax0.legend(loc="best")

        ax1 = axes[row, 1]
        ax1.plot(times_us, ode_true[0, :], color="black", linewidth=2, label="x1 true")
        ax1.plot(times_us, traj_pred[0, :], color="crimson", linewidth=2, linestyle="--", label="x1 pred")
        ax1.set_title("x1 after st1pl")
        ax1.set_xlabel("time (us)")
        ax1.set_ylabel("x1")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[row, 2]
        ax2.plot(times_us, ode_true[1, :], color="black", linewidth=2, label="x2 true")
        ax2.plot(times_us, traj_pred[1, :], color="crimson", linewidth=2, linestyle="--", label="x2 pred")
        ax2.set_title("x2 after st1pl")
        ax2.set_xlabel("time (us)")
        ax2.set_ylabel("x2")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

        ax3 = axes[row, 3]
        ax3.plot(times_us, x2dot_true, color="black", linewidth=2, label="x2dot true")
        ax3.plot(times_us, x2dot_pred, color="crimson", linewidth=2, linestyle="--", label="x2dot pred")
        ax3.set_title("x2dot after st1pl")
        ax3.set_xlabel("time (us)")
        ax3.set_ylabel("x2dot")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="best")

        ax4 = axes[row, 4]
        ax4.plot(times_us, ode_true[2, :], color="black", linewidth=2, label="x3 true")
        ax4.plot(times_us, traj_pred[2, :], color="crimson", linewidth=2, linestyle="--", label="x3 pred")
        ax4.set_title("x3 after st1pl")
        ax4.set_xlabel("time (us)")
        ax4.set_ylabel("x3")
        ax4.grid(True, alpha=0.25)
        ax4.legend(loc="best")

    source_text = "\n".join(row_labels)
    fig.suptitle(
        "AFM04 stage2light warmstart source snapshot after st1pl\n" + source_text,
        fontsize=14,
        y=1.02,
    )
    return finalize_and_save(fig, out_path(DEFAULT_OUT_FILE, out_dir))


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_OUT_DIR
    run_one(out_dir=out_dir)


if __name__ == "__main__":
    main()
