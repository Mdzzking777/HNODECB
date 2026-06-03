from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from AFM04.KAN_full_test.kan_backend import KANForceModule, plan_z_x3dot_scale
from AFM04.KAN_full_test.rollout import (
    force_inputs_for_module,
    fts_truth_from_states_torch,
    rollout_single_shooting_torch,
)
from AFM04.KAN_full_test.runner.visualization._common import (
    CHECKPOINT_DIR,
    OUT_DIR,
    REPO_ROOT,
    finalize_and_save,
)
from AFM04.stage1pluslight.data import load_dataset, truncate_to_first_contact
from AFM04.stage1pluslight.windows import window_manifests


RESULT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "results"
RS_MERGED_SUMMARY = RESULT_DIR / "random_search" / "kan_full_test_random_search_merged_summary.json"
DEFAULT_MAX_ROLLOUT_POINTS = 4000
DEFAULT_ROLLOUT_METHOD = ""
FIXED_GRID_METHODS = {"euler", "midpoint", "rk4", "explicit_adams", "implicit_adams"}


@dataclass(frozen=True)
class FullRolloutContext:
    payload: dict[str, Any]
    source: str
    cfg: dict[str, Any]
    model: KANForceModule
    known_pars: tuple[float, ...]
    mech_true_t: torch.Tensor
    eta_star_true: float
    dtype: torch.dtype
    ode_true: np.ndarray
    all_times: np.ndarray
    window_meta: dict[str, Any]
    rollout_start_idx: int
    rollout_role: str
    times_rollout_full: np.ndarray
    panels: list[tuple[str, int, int]]
    rollout_method: str
    solve_note: str
    solve_idx: np.ndarray
    plot_idx: np.ndarray
    plot_pos: np.ndarray
    times_rollout: np.ndarray
    times_us: np.ndarray
    ode_rollout_true: np.ndarray
    traj_pred: np.ndarray
    q_pre: np.ndarray | None = None


def torch_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def checkpoint_path_for_tag(tag: str) -> Path:
    return CHECKPOINT_DIR / f"kan_full_test_checkpoint_{tag}.pt"


def best_checkpoint_path_for_tag(tag: str) -> Path:
    return CHECKPOINT_DIR / f"kan_full_test_best_{tag}.pt"


def load_state_payload() -> tuple[dict[str, Any], str]:
    """Load the model-bearing KFT payload used for full-horizon rollout.

    Normal KFT result files intentionally keep lightweight visualization
    snapshots.  The current model weights live in checkpoint payloads.
    """

    for tag in ("modified_w0", "w0"):
        checkpoint = checkpoint_path_for_tag(tag)
        if checkpoint.is_file():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or "state_dict" not in payload:
                raise RuntimeError(f"Invalid KFT checkpoint payload: {checkpoint}")
            return payload, f"checkpoint:{tag}"

        best_checkpoint = best_checkpoint_path_for_tag(tag)
        if best_checkpoint.is_file():
            payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or "state_dict" not in payload:
                raise RuntimeError(f"Invalid KFT best checkpoint payload: {best_checkpoint}")
            return payload, f"best_checkpoint:{tag}"

    raise FileNotFoundError(f"No KFT checkpoint payload with model weights found under: {CHECKPOINT_DIR}")


def load_full_dataset(cfg_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    dataset_root = Path(cfg_dict["dataset_root"])
    error_level = str(cfg_dict["error_level"])
    ode_data, pert_df = load_dataset(dataset_root, error_level, auto_generate=False)
    ode_data, pert_df = truncate_to_first_contact(ode_data, pert_df)
    all_times = np.asarray(pert_df["t"], dtype=float)
    contact_all = np.asarray(pert_df["contact"], dtype=bool)
    x1_signal = np.asarray(ode_data[0, :], dtype=float)

    arch_window_us = float(cfg_dict.get("arch_window_us", 6.288e-6))
    manifests = window_manifests(all_times, contact_all, "stage2_w123", arch_window_us, x1_signal=x1_signal)
    role_to_manifest = {str(man.role).strip().lower(): man for man in manifests}
    return ode_data, all_times, role_to_manifest


def build_model(payload: dict[str, Any]) -> tuple[KANForceModule, tuple[float, ...], torch.Tensor, float, torch.dtype]:
    cfg = payload["config"]
    dtype = torch_dtype(str(cfg["dtype"]))
    device = "cpu"
    known_pars = tuple(float(v) for v in payload["known_pars"])
    mech_true = torch.as_tensor(np.asarray(payload["mech_true"], dtype=float), dtype=dtype, device=device)
    window_meta = payload.get("window_meta", {}) if isinstance(payload.get("window_meta", {}), dict) else {}
    window_span = float(window_meta.get("t_stop", 0.0)) - float(window_meta.get("t_start", 0.0))
    if not np.isfinite(window_span) or window_span <= 0.0:
        window_span = float(cfg.get("arch_window_us", 6.288e-6))

    model = KANForceModule(
        pykan_root=Path(cfg["pykan_root"]),
        state_mean=np.asarray(payload["state_mean"], dtype=float),
        state_scale=np.asarray(payload["state_scale"], dtype=float),
        seed=int(cfg["seed"]),
        width=tuple(int(v) for v in cfg["width"]),
        grid=int(cfg["grid"]),
        spline_k=int(cfg["spline_k"]),
        base_fun=str(cfg["base_fun"]),
        symbolic_enabled=bool(cfg["symbolic_enabled"]),
        auto_save=bool(cfg["auto_save"]),
        noise_scale=float(cfg["noise_scale"]),
        affine_trainable=bool(cfg["affine_trainable"]),
        grid_eps=float(cfg["grid_eps"]),
        grid_range=(float(cfg["grid_range_lo"]), float(cfg["grid_range_hi"])),
        gnn_learnable=bool(cfg["gnn_learnable"]),
        dist=float(known_pars[6]),
        a0=float(known_pars[9]),
        wpred_enabled=bool(cfg.get("train_wpred_enabled", cfg.get("wpred_enabled", False))),
        wpred_eps=float(cfg.get("wpred_eps", 1.0e-10)),
        soft_mask_enabled=bool(cfg.get("soft_mask_enabled", False)),
        soft_mask_trainable=bool(cfg.get("soft_mask_trainable", False)),
        soft_mask_s0_a0=float(cfg.get("soft_mask_s0_a0", 20.0)),
        soft_mask_s0_min_a0=float(cfg.get("soft_mask_s0_min_a0", 1.0)),
        soft_mask_s0_max_a0=float(cfg.get("soft_mask_s0_max_a0", 100.0)),
        soft_mask_alpha_a0=float(cfg.get("soft_mask_alpha_a0", 0.25)),
        soft_mask_alpha_min_a0=float(cfg.get("soft_mask_alpha_min_a0", 0.02)),
        soft_mask_alpha_max_a0=float(cfg.get("soft_mask_alpha_max_a0", 5.0)),
        x3dot_input_enabled=bool(cfg.get("x3dot_input_enabled", False)),
        x3dot_init_trainable=bool(cfg.get("x3dot_init_trainable", False)),
        x3dot_init_value=float(cfg.get("x3dot_init_value", 0.0)),
        x3dot_scale=plan_z_x3dot_scale(
            configured_scale=float(cfg.get("x3dot_scale", 0.0)),
            scale_mode=str(cfg.get("x3dot_scale_mode", "x2_scale_tenth")),
            a0=float(known_pars[9]),
            window_span=window_span,
            x2_scale=float(np.asarray(payload["state_scale"], dtype=float).reshape(-1)[1]),
        ),
        x3dot_lag_detach=bool(cfg.get("x3dot_lag_detach", False)),
        device=device,
        dtype=dtype,
    ).to(device)

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("KFT checkpoint payload is missing state_dict")
    model.load_state_dict(state_dict)
    model.eval()
    return model, known_pars, mech_true, float(payload["eta_star_true"]), dtype


def double_window_bounds(start_idx: int, stop_idx: int, total_len: int) -> tuple[int, int]:
    length = int(stop_idx) - int(start_idx) + 1
    target = min(int(total_len), max(length, 2 * length))
    center = 0.5 * (int(start_idx) + int(stop_idx))
    new_start = int(round(center - 0.5 * (target - 1)))
    new_stop = new_start + target - 1
    if new_start < 0:
        new_stop += -new_start
        new_start = 0
    if new_stop >= total_len:
        shift = new_stop - total_len + 1
        new_start = max(0, new_start - shift)
        new_stop = total_len - 1
    return int(new_start), int(new_stop)


def window_short(role: str) -> str:
    role_norm = str(role).strip().lower()
    if role_norm == "first_contact":
        return "W0"
    if role_norm == "middle":
        return "W1"
    if role_norm == "max_x1_pp_change":
        return "W2"
    if role_norm == "tail_stable":
        return "W3"
    if role_norm == "modified_w0":
        return "modified W0"
    return role or "W?"


def panel_title(name: str, start: int, stop: int, times: np.ndarray) -> str:
    return f"{name}\nt = [{1.0e6 * times[start]:.3f}, {1.0e6 * times[stop]:.3f}] us"


def true_based_ylim(values_true: np.ndarray) -> tuple[float, float]:
    lo = float(np.nanmin(values_true))
    hi = float(np.nanmax(values_true))
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        scale = max(abs(lo), abs(hi), 1.0e-12)
        pad = 0.1 * scale
    else:
        pad = 0.12 * span
    return lo - pad, hi + pad


def rs_trial_text() -> str:
    if not RS_MERGED_SUMMARY.is_file():
        return ""
    try:
        data = json.loads(RS_MERGED_SUMMARY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    trial = data.get("common_trial")
    if trial is None:
        return ""
    return f" | RS trial {int(trial)}"


def max_rollout_points() -> int:
    raw = os.environ.get("HNODECB_AFM04_KAN_TEST_FULL_ROLLOUT_MAX_POINTS", "").strip()
    if raw == "":
        return DEFAULT_MAX_ROLLOUT_POINTS
    value = int(raw)
    return value if value > 0 else 0


def rollout_method(cfg_dict: dict[str, Any]) -> str:
    raw = os.environ.get("HNODECB_AFM04_KAN_TEST_FULL_ROLLOUT_METHOD", "").strip()
    return raw or DEFAULT_ROLLOUT_METHOD or str(cfg_dict["ode_method"])


def is_fixed_grid_method(method: str) -> bool:
    return str(method).strip().lower() in FIXED_GRID_METHODS


def fast_adaptive_rollout_enabled() -> bool:
    raw = os.environ.get("HNODECB_AFM04_KAN_TEST_FULL_ROLLOUT_FAST", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def rollout_eval_indices(total_len: int, panels: list[tuple[str, int, int]]) -> np.ndarray:
    max_points = max_rollout_points()
    if max_points <= 0 or total_len <= max_points:
        return np.arange(total_len, dtype=int)

    base = np.linspace(0, total_len - 1, max_points, dtype=int)
    extra: list[np.ndarray] = [base, np.asarray([0, total_len - 1], dtype=int)]
    for name, start, stop in panels:
        if name == "Full horizon from W0 IC":
            continue
        extra.append(np.arange(max(0, start), min(total_len - 1, stop) + 1, dtype=int))
    return np.unique(np.concatenate(extra)).astype(int)


def source_note(payload: dict[str, Any], source: str) -> str:
    epoch = payload.get("epoch")
    if epoch is None:
        return source
    return f"{source}, epoch {int(epoch)}"


def build_panels(
    *,
    window_meta: dict[str, Any],
    role_to_manifest: dict[str, object],
    rollout_start_idx: int,
    times_rollout_full: np.ndarray,
    rollout_role: str,
) -> list[tuple[str, int, int]]:
    panels: list[tuple[str, int, int]] = [("Full horizon from W0 IC", 0, len(times_rollout_full) - 1)]
    stop_rel = int(window_meta.get("stop_idx", rollout_start_idx)) - rollout_start_idx
    w_start, w_stop = double_window_bounds(0, stop_rel, len(times_rollout_full))
    panels.append((f"{window_short(rollout_role)} (2x default)", w_start, w_stop))

    for role in ("max_x1_pp_change", "tail_stable"):
        man = role_to_manifest.get(role)
        if man is None:
            continue
        start_rel = int(man.start_idx) - rollout_start_idx
        stop_rel = int(man.stop_idx) - rollout_start_idx
        if stop_rel < 0:
            continue
        start, stop = double_window_bounds(max(0, start_rel), stop_rel, len(times_rollout_full))
        panels.append((f"{window_short(man.role)} (2x default)", start, stop))
    return panels


def prepare_full_rollout_context() -> FullRolloutContext:
    payload, source = load_state_payload()
    cfg = payload["config"]
    model, known_pars, mech_true_t, eta_star_true, dtype = build_model(payload)
    ode_true, all_times, role_to_manifest = load_full_dataset(cfg)

    window_meta = payload.get("window_meta", {})
    if not isinstance(window_meta, dict):
        window_meta = {}
    rollout_start_idx = int(window_meta.get("start_idx", 0))
    rollout_role = str(window_meta.get("role", "first_contact")).strip().lower() or "first_contact"
    times_rollout_full = np.asarray(all_times[rollout_start_idx:], dtype=float)
    panels = build_panels(
        window_meta=window_meta,
        role_to_manifest=role_to_manifest,
        rollout_start_idx=rollout_start_idx,
        times_rollout_full=times_rollout_full,
        rollout_role=rollout_role,
    )

    method = rollout_method(cfg)
    plot_idx = rollout_eval_indices(len(times_rollout_full), panels)
    if is_fixed_grid_method(method) or not fast_adaptive_rollout_enabled():
        solve_idx = np.arange(len(times_rollout_full), dtype=int)
        solve_note = "full-grid"
    else:
        solve_idx = plot_idx
        solve_note = "adaptive-sampled-preview"

    times_rollout = times_rollout_full[solve_idx]
    ode_rollout_true = np.asarray(ode_true[:, rollout_start_idx + solve_idx], dtype=float)
    times_t = torch.as_tensor(times_rollout, dtype=dtype, device="cpu")
    u0 = torch.as_tensor(ode_true[:, rollout_start_idx], dtype=dtype, device="cpu")

    with torch.no_grad():
        rollout_out = rollout_single_shooting_torch(
            model,
            known_pars,
            mech_true_t,
            u0,
            times_t,
            method=method,
            rtol=float(cfg["ode_rtol"]),
            atol=float(cfg["ode_atol"]),
            return_aux=True,
        )
        traj_pred_t, aux = rollout_out

    traj_pred = traj_pred_t.detach().cpu().numpy()
    q_pre_np = None if aux.q_pre is None else aux.q_pre.detach().cpu().numpy()
    times_us = 1.0e6 * times_rollout
    plot_pos = np.searchsorted(solve_idx, plot_idx)
    if not np.array_equal(solve_idx[plot_pos], plot_idx):
        raise RuntimeError("Internal plot index mapping failed for KFT full rollout visualization")

    return FullRolloutContext(
        payload=payload,
        source=source,
        cfg=cfg,
        model=model,
        known_pars=known_pars,
        mech_true_t=mech_true_t,
        eta_star_true=eta_star_true,
        dtype=dtype,
        ode_true=ode_true,
        all_times=all_times,
        window_meta=window_meta,
        rollout_start_idx=rollout_start_idx,
        rollout_role=rollout_role,
        times_rollout_full=times_rollout_full,
        panels=panels,
        rollout_method=method,
        solve_note=solve_note,
        solve_idx=solve_idx,
        plot_idx=plot_idx,
        plot_pos=plot_pos,
        times_rollout=times_rollout,
        times_us=times_us,
        ode_rollout_true=ode_rollout_true,
        traj_pred=traj_pred,
        q_pre=q_pre_np,
    )


def compute_rollout_fts(ctx: FullRolloutContext) -> tuple[np.ndarray, np.ndarray]:
    rollout_states_pred = torch.as_tensor(ctx.traj_pred.T, dtype=ctx.dtype, device="cpu")
    rollout_states_true = torch.as_tensor(ctx.ode_rollout_true.T, dtype=ctx.dtype, device="cpu")
    q_pre = None if ctx.q_pre is None else torch.as_tensor(ctx.q_pre, dtype=ctx.dtype, device="cpu")
    with torch.no_grad():
        fts_true_t = fts_truth_from_states_torch(
            rollout_states_true,
            ctx.known_pars,
            eta_star=ctx.eta_star_true,
            mech_true=ctx.mech_true_t,
        )
        fts_pred_t = ctx.model(force_inputs_for_module(ctx.model, rollout_states_pred, q_pre))
    return (
        fts_true_t.detach().cpu().numpy().reshape(-1),
        fts_pred_t.detach().cpu().numpy().reshape(-1),
    )


def figure_title(ctx: FullRolloutContext, quantity: str) -> str:
    mech_true = ctx.mech_true_t.detach().cpu().numpy().reshape(-1)
    return (
        f"AFM04 KAN full-test {quantity} postprocess rollout from W0 known IC\n"
        f"final KAN + ground-truth ks/cs{rs_trial_text()} | "
        f"W0 start={1.0e6 * ctx.times_rollout_full[0]:.3f} us | "
        f"ks_true={float(mech_true[0]):.6e}, cs_true={float(mech_true[1]):.6e} | "
        f"{source_note(ctx.payload, ctx.source)} | rollout={ctx.rollout_method}/{ctx.solve_note}, "
        f"solve_points={len(ctx.solve_idx)}, plot_points={len(ctx.plot_idx)}"
    )
