from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.stage1pluslight.data import load_dataset, truncate_to_first_contact
from AFM04.stage1pluslight.windows import window_manifests
from AFM04.stage2light.config import default_config as default_stage2light_config
from AFM04.stage2light.kan_backend import KANForceModule
from AFM04.stage2light.rollout import LearnableMechModule, rollout_single_shooting_torch
from AFM04.stage2light.runner.visualization._common import (
    OUT_DIR,
    REPO_ROOT,
    finalize_and_save,
    load_checkpoint_payloads,
)


RESULT_DIR = REPO_ROOT / "AFM04" / "stage2light" / "results"
_RESULT_PT_RE = re.compile(r"stage2light_result_p(\d+)\.pt$")
FULL_PANEL_MAX_POINTS = 4000


def _torch_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def _discover_result_pt_paths(result_dir: Path = RESULT_DIR) -> list[Path]:
    selected: list[tuple[int, Path]] = []
    for path in result_dir.glob("stage2light_result_p*.pt"):
        match = _RESULT_PT_RE.match(path.name)
        if match is not None:
            selected.append((int(match.group(1)), path))
    return [path for _, path in sorted(selected, key=lambda item: item[0])]


def _load_result_payload() -> dict:
    payloads: list[dict] = []
    for result_path in _discover_result_pt_paths():
        payload = torch.load(result_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            payloads.append(payload)
    if not payloads:
        payloads = load_checkpoint_payloads()
    preferred = [p for p in payloads if str(p.get("window_meta", {}).get("role", "")).strip().lower() == "first_contact"]
    payload = preferred[0] if preferred else payloads[0]
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected payload type: {type(payload)!r}")
    return payload


def _arch_window_us(cfg_dict: dict) -> float:
    raw = cfg_dict.get("arch_window_us")
    if raw is not None:
        return float(raw)
    return float(default_stage2light_config(REPO_ROOT).arch_window_us)


def _load_full_dataset(cfg_dict: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, dict[str, object]]:
    dataset_root = Path(cfg_dict["dataset_root"])
    error_level = str(cfg_dict["error_level"])
    ode_data, pert_df = load_dataset(dataset_root, error_level, auto_generate=False)
    ode_data, pert_df = truncate_to_first_contact(ode_data, pert_df)
    all_times = np.asarray(pert_df["t"], dtype=float)
    contact_all = np.asarray(pert_df["contact"], dtype=bool)
    x1_signal = np.asarray(ode_data[0, :], dtype=float)
    w0_manifests = window_manifests(
        all_times,
        contact_all,
        "w0",
        _arch_window_us(cfg_dict),
        x1_signal=x1_signal,
    )
    stage2_manifests = window_manifests(
        all_times,
        contact_all,
        "stage2_w123",
        _arch_window_us(cfg_dict),
        x1_signal=x1_signal,
    )
    w0_manifest = w0_manifests[0]
    stage2_role_to_manifest = {str(man.role).strip().lower(): man for man in stage2_manifests}
    return ode_data, all_times, contact_all, w0_manifest, stage2_role_to_manifest


def _build_model_and_mech(payload: dict) -> tuple[KANForceModule, LearnableMechModule, tuple[float, ...], torch.dtype]:
    cfg = payload["config"]
    dtype = _torch_dtype(cfg["dtype"])
    device = "cpu"
    known_pars = tuple(float(v) for v in payload["known_pars"])

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
        soft_mask_enabled=bool(cfg.get("soft_mask_enabled", True)),
        soft_mask_trainable=bool(cfg.get("soft_mask_trainable", True)),
        soft_mask_s0_a0=float(cfg.get("soft_mask_s0_a0", 20.0)),
        soft_mask_s0_min_a0=float(cfg.get("soft_mask_s0_min_a0", 1.0)),
        soft_mask_s0_max_a0=float(cfg.get("soft_mask_s0_max_a0", 100.0)),
        soft_mask_alpha_a0=float(cfg.get("soft_mask_alpha_a0", 0.25)),
        soft_mask_alpha_min_a0=float(cfg.get("soft_mask_alpha_min_a0", 0.02)),
        soft_mask_alpha_max_a0=float(cfg.get("soft_mask_alpha_max_a0", 5.0)),
        device=device,
        dtype=dtype,
    ).to(device)

    mech_module = LearnableMechModule(
        ks_init=float(payload["mech_true"][0]),
        cs_init=float(payload["mech_true"][1]),
        ks_bounds=(float(cfg["ks_lo"]), float(cfg["ks_hi"])),
        cs_bounds=(float(cfg["cs_lo"]), float(cfg["cs_hi"])),
        dtype=dtype,
        device=device,
    ).to(device)

    state_dict = payload.get("final_state_dict")
    mech_state_dict = payload.get("final_mech_state_dict")
    if not isinstance(state_dict, dict) or not isinstance(mech_state_dict, dict):
        state_dict = payload.get("state_dict")
        mech_state_dict = payload.get("mech_state_dict")
    if not isinstance(state_dict, dict) or not isinstance(mech_state_dict, dict):
        raise RuntimeError("State payload is missing final_state_dict/final_mech_state_dict or state_dict/mech_state_dict")

    model.load_state_dict(state_dict)
    mech_module.load_state_dict(mech_state_dict)
    model.eval()
    mech_module.eval()
    return model, mech_module, known_pars, dtype


def _double_window_bounds(start_idx: int, stop_idx: int, total_len: int) -> tuple[int, int]:
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


def _panel_indices(start: int, stop: int, *, max_points: int | None = None) -> np.ndarray:
    start_i = int(start)
    stop_i = int(stop)
    if max_points is None or stop_i - start_i + 1 <= int(max_points):
        return np.arange(start_i, stop_i + 1, dtype=int)
    return np.unique(np.linspace(start_i, stop_i, int(max_points), dtype=int))


def _window_short(role: str) -> str:
    role_norm = str(role).strip().lower()
    if role_norm == "first_contact":
        return "W0"
    if role_norm == "middle":
        return "W1"
    if role_norm == "max_x1_pp_change":
        return "W2"
    if role_norm == "tail_stable":
        return "W3"
    return role or "W?"


def _panel_title(name: str, start: int, stop: int, times: np.ndarray) -> str:
    return (
        f"{name}\n"
        f"t = [{1.0e6 * times[start]:.3f}, {1.0e6 * times[stop]:.3f}] μs"
    )


def _relative_rmse_pct_np(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    count = max(int(truth_arr.size), 1)
    err = float(np.sum(np.square(pred_arr - truth_arr)))
    den = float(np.sum(np.square(truth_arr)))
    truth_rms = np.sqrt(den / float(count))
    if truth_rms <= 0.0:
        return 0.0
    return 100.0 * np.sqrt(err / float(count)) / truth_rms


def main() -> None:
    payload = _load_result_payload()
    cfg = payload["config"]
    model, mech_module, known_pars, dtype = _build_model_and_mech(payload)
    ode_true, all_times, _contact_all, w0_manifest, stage2_role_to_manifest = _load_full_dataset(cfg)

    rollout_start_idx = int(w0_manifest.start_idx)
    times_rollout = np.asarray(all_times[rollout_start_idx:], dtype=float)
    ode_rollout_true = np.asarray(ode_true[:, rollout_start_idx:], dtype=float)

    ordered_roles = ["max_x1_pp_change", "tail_stable"]
    panels: list[tuple[str, int, int, np.ndarray]] = [
        (
            "Full horizon from W0 IC",
            0,
            len(times_rollout) - 1,
            _panel_indices(0, len(times_rollout) - 1, max_points=FULL_PANEL_MAX_POINTS),
        )
    ]
    w0_start, w0_stop = _double_window_bounds(
        int(w0_manifest.start_idx) - rollout_start_idx,
        int(w0_manifest.stop_idx) - rollout_start_idx,
        len(times_rollout),
    )
    panels.append(("W0 (2× default)", w0_start, w0_stop, _panel_indices(w0_start, w0_stop)))
    for role in ordered_roles:
        man = stage2_role_to_manifest.get(role)
        if man is None:
            continue
        start_rel = int(man.start_idx) - rollout_start_idx
        stop_rel = int(man.stop_idx) - rollout_start_idx
        start, stop = _double_window_bounds(start_rel, stop_rel, len(times_rollout))
        panels.append((f"{_window_short(man.role)} (2× default)", start, stop, _panel_indices(start, stop)))

    eval_indices = np.unique(np.concatenate([panel[-1] for panel in panels]))
    times_eval = np.asarray(times_rollout[eval_indices], dtype=float)

    times_t = torch.as_tensor(times_eval, dtype=dtype, device="cpu")
    u0 = torch.as_tensor(ode_true[:, rollout_start_idx], dtype=dtype, device="cpu")
    with torch.no_grad():
        traj_pred_t = rollout_single_shooting_torch(
            model,
            known_pars,
            mech_module,
            u0,
            times_t,
            method=str(cfg["ode_method"]),
            rtol=float(cfg["ode_rtol"]),
            atol=float(cfg["ode_atol"]),
        )
        mech_pred = mech_module().detach().cpu().numpy().reshape(-1)

    traj_pred_eval = traj_pred_t.detach().cpu().numpy()
    times_eval_us = 1.0e6 * times_eval

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=False, squeeze=False)
    flat_axes = list(axes.ravel())
    for ax, (name, start, stop, indices) in zip(flat_axes, panels, strict=False):
        pos = np.searchsorted(eval_indices, indices)
        ax.plot(times_eval_us[pos], ode_rollout_true[0, indices], color="black", linewidth=2.0, label="x1 true", zorder=1)
        ax.plot(
            times_eval_us[pos],
            traj_pred_eval[0, pos],
            color="crimson",
            linewidth=2.0,
            linestyle="--",
            label="x1 pred",
            zorder=3,
        )
        ax.set_title(_panel_title(name, start, stop, times_rollout))
        ax.set_xlabel("time (μs)")
        ax.set_ylabel("x1")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        if name != "Full horizon from W0 IC":
            err_pct = _relative_rmse_pct_np(traj_pred_eval[0, pos], ode_rollout_true[0, indices])
            ax.text(
                0.03,
                0.97,
                f"error = {err_pct:.2f}%",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.80),
                zorder=5,
            )

    for ax in flat_axes[len(panels):]:
        ax.axis("off")

    stage1_meta = payload.get("warmstart") or payload.get("stage1_warmstart") or {}
    rank_text = ""
    if isinstance(stage1_meta, dict):
        if int(stage1_meta.get("candidate_b", 0) or 0) > 0:
            rank_text = f" | candidate B {int(stage1_meta['candidate_b'])}"
        elif int(stage1_meta.get("mech_winner", 0) or 0) > 0:
            rank_text = f" | mech winner {int(stage1_meta['mech_winner'])}"
        elif int(stage1_meta.get("rank", 0) or 0) > 0:
            rank_text = f" | stage1 rank {int(stage1_meta['rank'])}"
    fig.suptitle(
        "AFM04 stage2light postprocess rollout from W0 known IC\n"
        f"final optimized KAN + final ks/cs{rank_text} | "
        f"W0 start={1.0e6 * times_rollout[0]:.3f} μs | "
        f"ks_hat={float(mech_pred[0]):.6e}, cs_hat={float(mech_pred[1]):.6e}",
        fontsize=14,
        y=0.98,
    )

    finalize_and_save(fig, OUT_DIR / "afm_param_stage2light_04_x1_full_rollout_grid.png")


if __name__ == "__main__":
    main()
