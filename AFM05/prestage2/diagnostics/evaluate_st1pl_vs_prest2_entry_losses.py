"""Diagnose AFM05 st1pl-to-prest2 loss discontinuity.

This script does not train.  It compares four quantities for prest2 B
candidates:

1. Completed st1pl trial evaluated by the st1pl loss, read from the st1pl pkl.
2. The same st1pl error components converted to the stage2 RMS normalization.
3. The prest2 pre-refit rollout evaluated by a st1pl-style loss.
4. The prest2 pre-refit rollout evaluated by the stage2 loss.

The second quantity is a scale-only estimate because the st1pl pkl does not
store the full st1pl trajectory.  The third and fourth quantities are recomputed
by reproducing the prest2 pre-refit rollout without optimizer steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM05.stage1pluslight.losses import (
    FTS_RANGE_AMP,
    FTS_RANGE_EPS,
    FTS_RANGE_WEIGHT,
    SCALE_EPS,
    X3_RANGE_AMP,
    X3_RANGE_EPS,
    X3_RANGE_WEIGHT,
    softplus,
)
from AFM05.stage2light.config import default_config
from AFM05.stage2light.data import prepare_data, select_stage1_candidate_by_trial_id
from AFM05.stage2light.kan_backend import (
    KANForceModule,
    initial_grid_support_from_raw_inputs,
    initial_grid_support_to_meta,
)
from AFM05.stage2light.losses import evaluate_split
from AFM05.stage2light.rollout import LearnableMechModule, _afm05_known_fields, x2dot_rhs_torch
from AFM05.stage2light.train import (
    _make_grid_update_inputs,
    _observable_grid_inputs_from_ode,
    _prepared_with_initial_x3_from_warmstart,
    _selected_window_indices,
    _strict_mean_scale_from_x3_pred,
    _torch_dtype,
    _window_to_torch,
)


DEFAULT_ST1PL = REPO_ROOT / "AFM05" / "stage1pluslight" / "results_afm" / "afm_param_stage1pluslight_05.pkl"
DEFAULT_PREST2 = REPO_ROOT / "AFM05" / "prestage2" / "results" / "afm_prest2_05_candidates_b.pkl"
DEFAULT_OUTDIR = REPO_ROOT / "AFM05" / "prestage2" / "diagnostics"


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _candidate_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("candidate_b_records", "candidate_records"):
        records = payload.get(key)
        if isinstance(records, list) and records:
            return [rec for rec in records if isinstance(rec, dict)]
    return []


def _stage1_records_by_trial_id(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for rec in payload.get("trial_parameters", []):
        if not isinstance(rec, dict):
            continue
        params = rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}
        trial_id = int(params.get("trial_id", 0))
        if trial_id > 0:
            out[trial_id] = rec
    return out


def _config_from_saved_dict(saved: dict[str, Any], *, trial_id: int, mech_winner: int) -> Any:
    cfg = default_config(REPO_ROOT)
    kwargs: dict[str, Any] = {}
    field_by_name = {f.name: f for f in fields(cfg)}
    for key, value in saved.items():
        if key not in field_by_name:
            continue
        current = getattr(cfg, key)
        if isinstance(current, Path):
            kwargs[key] = Path(value)
        elif key == "width":
            kwargs[key] = tuple(int(v) for v in value)
        else:
            kwargs[key] = value
    kwargs.update(
        {
            "repo_root": REPO_ROOT,
            "pykan_root": REPO_ROOT / "pykan",
            "dataset_root": REPO_ROOT / "AFM05" / "datasets",
            "stage1_input_path": DEFAULT_ST1PL,
            "shard_index": 1,
            "shard_count": 1,
            "device": "cpu",
            "resume_from_checkpoint": False,
            "stage1_input_trial_id": int(trial_id),
            "stage1_input_mech_winner": int(mech_winner),
            "warmstart_source": "stage1",
        }
    )
    return replace(cfg, **kwargs)


def _safe_rms_np(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    out = float(np.sqrt(np.mean(np.square(arr))))
    return max(out if np.isfinite(out) else 0.0, SCALE_EPS)


def _scale_only_stage2_estimate(
    *,
    stage1_parts: dict[str, Any],
    split: Any,
    state12_range_scale: np.ndarray,
    x2dot_range_scale: float,
) -> dict[str, float]:
    train_idx = np.asarray(split.train_idx, dtype=int)
    x1_rms = _safe_rms_np(split.ode_full[0, train_idx])
    x2_rms = _safe_rms_np(split.ode_full[1, train_idx])
    x2dot_rms = _safe_rms_np(split.x2dot_full[train_idx])
    x1 = float(stage1_parts.get("x1_state", float("nan"))) * (float(state12_range_scale[0]) / x1_rms) ** 2
    x2 = float(stage1_parts.get("x2_state", float("nan"))) * (float(state12_range_scale[1]) / x2_rms) ** 2
    x2dot = float(stage1_parts.get("x2dot", float("nan"))) * (float(x2dot_range_scale) / x2dot_rms) ** 2
    x3_range = float(stage1_parts.get("x3_range", 0.0))
    fts_range = float(stage1_parts.get("fts_range", 0.0))
    return {
        "loss": float(x1 + x2 + x2dot + x3_range + fts_range),
        "x1_state": x1,
        "x2_state": x2,
        "x2dot": x2dot,
        "x3_range_carried_from_st1pl": x3_range,
        "fts_range_carried_from_st1pl": fts_range,
        "x1_rms_scale_stage2": x1_rms,
        "x2_rms_scale_stage2": x2_rms,
        "x2dot_rms_scale_stage2": x2dot_rms,
    }


def _relative_rmse_pct_np(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    count = max(int(ref.size), 1)
    err = float(np.sum(np.square(pred - ref)))
    den = float(np.sum(np.square(ref)))
    denom = math.sqrt(den / count) + SCALE_EPS
    return float(100.0 * math.sqrt(err / count) / denom)


def _stage1_style_loss_from_traj(
    *,
    traj: torch.Tensor,
    force_module: torch.nn.Module,
    known_pars: dict[str, Any],
    split: Any,
    tensors: dict[str, torch.Tensor],
    state12_range_scale: np.ndarray,
    x2dot_range_scale: float,
    x3_range_scale: float,
) -> dict[str, float]:
    idx_t = tensors["train_idx"]
    idx_np = idx_t.detach().cpu().numpy().astype(int)
    pred_t = traj[:, idx_t]
    times_t = tensors["times_full"][idx_t]
    x2dot_pred_t = x2dot_rhs_torch(pred_t, times_t, force_module, known_pars)
    fts_pred_t = force_module(pred_t.transpose(0, 1))

    pred = pred_t.detach().cpu().numpy()
    obs = np.asarray(split.ode_full[:, idx_np], dtype=float)
    x2dot_obs = np.asarray(split.x2dot_full[idx_np], dtype=float)
    x2dot_pred = x2dot_pred_t.detach().cpu().numpy()
    fts_pred = fts_pred_t.detach().cpu().numpy()

    s1 = max(float(state12_range_scale[0]), SCALE_EPS)
    s2 = max(float(state12_range_scale[1]), SCALE_EPS)
    sd = max(float(x2dot_range_scale), SCALE_EPS)

    x1_state = float(np.mean(np.square((obs[0, :] - pred[0, :]) / s1)))
    x2_state = float(np.mean(np.square((obs[1, :] - pred[1, :]) / s2)))
    x2dot_loss = float(np.mean(np.square((x2dot_obs - x2dot_pred) / sd)))

    fts_pen = np.square(np.asarray(softplus(np.abs(fts_pred) - FTS_RANGE_AMP, FTS_RANGE_EPS), dtype=float) / FTS_RANGE_AMP)
    fts_range = float(FTS_RANGE_WEIGHT * np.mean(fts_pen))

    x3_scale = max(float(x3_range_scale), SCALE_EPS)
    x3_pen = np.square(np.asarray(softplus(np.abs(pred[2, :]) - X3_RANGE_AMP, X3_RANGE_EPS), dtype=float) / x3_scale)
    x3_range = float(X3_RANGE_WEIGHT * np.mean(x3_pen))

    return {
        "loss": float(x1_state + x2_state + x2dot_loss + x3_range + fts_range),
        "state": float(x1_state + x2_state),
        "x1_state": x1_state,
        "x2_state": x2_state,
        "x2dot": x2dot_loss,
        "x3_range": x3_range,
        "fts_range": fts_range,
        "x1_rec": _relative_rmse_pct_np(pred[0, :], obs[0, :]),
        "x2_rec": _relative_rmse_pct_np(pred[1, :], obs[1, :]),
        "x2dot_rec": _relative_rmse_pct_np(x2dot_pred, x2dot_obs),
    }


def _build_entry_and_preopt(
    *,
    cfg: Any,
    trial_id: int,
    mech_winner: int,
) -> dict[str, Any]:
    dtype = _torch_dtype(cfg.dtype)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    prepared = prepare_data(cfg)
    selected = _selected_window_indices(cfg, prepared.splits)
    split = prepared.splits[int(selected[0]) - 1]
    tensors = _window_to_torch(split, dtype=dtype, device=cfg.device)
    train_states = torch.as_tensor(split.ode_train.T, dtype=dtype, device=cfg.device)
    train_gain_force_reference = torch.as_tensor(split.train_gain_force_reference, dtype=dtype, device=cfg.device)

    warmstart = select_stage1_candidate_by_trial_id(
        cfg.stage1_input_path,
        trial_id=int(trial_id),
        mech_winner=int(mech_winner),
    )

    prepared, x3_support, x3_init_source = _prepared_with_initial_x3_from_warmstart(prepared, None)
    x3_support = x3_support if x3_support is not None else None
    observable_grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=tensors["ode_train"],
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        x3_norm_support=x3_support,
    )
    initial_grid_support = initial_grid_support_from_raw_inputs(
        observable_grid_inputs,
        prepared.state_mean,
        prepared.state_scale,
    )
    initial_grid_support_meta = initial_grid_support_to_meta(
        initial_grid_support,
        source=f"diagnostic_stage2light_initial__{x3_init_source}",
    )

    known_fields = _afm05_known_fields(prepared.known_pars)
    model = KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=int(warmstart.init_seed),
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
        initial_grid_support=initial_grid_support,
        dist=float(known_fields["Z"]),
        a0=float(known_fields["a0"]),
        gnn_learnable=cfg.gnn_learnable,
        soft_mask_enabled=cfg.soft_mask_enabled,
        soft_mask_trainable=cfg.soft_mask_trainable,
        soft_mask_s0_a0=cfg.soft_mask_s0_a0,
        soft_mask_s0_min_a0=cfg.soft_mask_s0_min_a0,
        soft_mask_s0_max_a0=cfg.soft_mask_s0_max_a0,
        soft_mask_alpha_a0=cfg.soft_mask_alpha_a0,
        soft_mask_alpha_min_a0=cfg.soft_mask_alpha_min_a0,
        soft_mask_alpha_max_a0=cfg.soft_mask_alpha_max_a0,
        device=cfg.device,
        dtype=dtype,
    ).to(cfg.device)
    mech_module = LearnableMechModule(
        ks_init=float(warmstart.ks0),
        cs_init=float(warmstart.cs0),
        ks_bounds=(cfg.ks_lo, cfg.ks_hi),
        cs_bounds=(cfg.cs_lo, cfg.cs_hi),
        dtype=dtype,
        device=cfg.device,
        parameterization=cfg.mech_parameterization,
    ).to(cfg.device)

    with torch.no_grad():
        gain_before = float(model.initialize_gain_from_reference(train_states, train_gain_force_reference))
        pre_total, pre_parts, pre_traj = evaluate_split(
            force_module=model,
            known_pars=prepared.known_pars,
            mech_module=mech_module,
            ode_true=tensors["ode_full"],
            x2dot_true=tensors["x2dot_full"],
            contact_mask=tensors["contact_full"],
            times=tensors["times_full"],
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            loss_indices=tensors["train_idx"],
        )
        pre_stage1_style = _stage1_style_loss_from_traj(
            traj=pre_traj,
            force_module=model,
            known_pars=prepared.known_pars,
            split=split,
            tensors=tensors,
            state12_range_scale=np.maximum(np.ptp(split.ode_full[0:2, :], axis=1), SCALE_EPS),
            x2dot_range_scale=max(float(np.ptp(split.x2dot_full)), SCALE_EPS),
            x3_range_scale=100.0e-9,
        )

        x3_mean, x3_scale = _strict_mean_scale_from_x3_pred(pre_traj[2, :])
        x3_pred_norm = (pre_traj[2, :].detach() - float(x3_mean)) / float(x3_scale)
        x3_norm_min = float(torch.min(x3_pred_norm).detach().cpu())
        x3_norm_max = float(torch.max(x3_pred_norm).detach().cpu())

        refit_mean = np.asarray(prepared.state_mean, dtype=float).copy()
        refit_scale = np.asarray(prepared.state_scale, dtype=float).copy()
        refit_mean[2] = float(x3_mean)
        refit_scale[2] = float(x3_scale)
        model.set_state_normalizer(refit_mean, refit_scale)
        refit_prepared = replace(prepared, state_mean=refit_mean, state_scale=refit_scale)
        refit_inputs = _observable_grid_inputs_from_ode(
            ode_train=tensors["ode_train"],
            state_mean=refit_prepared.state_mean,
            state_scale=refit_prepared.state_scale,
            x3_norm_support=(x3_norm_min, x3_norm_max),
        )
        if cfg.adaptive_grid_enabled:
            grid_inputs, _grid_meta = _make_grid_update_inputs(base_inputs=refit_inputs)
            model.update_grid_from_normalized_inputs(grid_inputs)
        gain_after = float(model.initialize_gain_from_reference(train_states, train_gain_force_reference))
        post_total, post_parts, post_traj = evaluate_split(
            force_module=model,
            known_pars=refit_prepared.known_pars,
            mech_module=mech_module,
            ode_true=tensors["ode_full"],
            x2dot_true=tensors["x2dot_full"],
            contact_mask=tensors["contact_full"],
            times=tensors["times_full"],
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            loss_indices=tensors["train_idx"],
        )
        post_stage1_style = _stage1_style_loss_from_traj(
            traj=post_traj,
            force_module=model,
            known_pars=refit_prepared.known_pars,
            split=split,
            tensors=tensors,
            state12_range_scale=np.maximum(np.ptp(split.ode_full[0:2, :], axis=1), SCALE_EPS),
            x2dot_range_scale=max(float(np.ptp(split.x2dot_full)), SCALE_EPS),
            x3_range_scale=100.0e-9,
        )

    return {
        "split": split,
        "initial_grid_support_meta": initial_grid_support_meta,
        "gain_before": gain_before,
        "gain_after": gain_after,
        "pre_stage2_loss": float(pre_total.detach()),
        "pre_stage2_parts": pre_parts,
        "pre_stage1_style": pre_stage1_style,
        "post_stage2_loss": float(post_total.detach()),
        "post_stage2_parts": post_parts,
        "post_stage1_style": post_stage1_style,
        "x3_refit_mean": float(x3_mean),
        "x3_refit_scale": float(x3_scale),
        "x3_refit_support_min": x3_norm_min,
        "x3_refit_support_max": x3_norm_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, default=DEFAULT_ST1PL)
    parser.add_argument("--prest2", type=Path, default=DEFAULT_PREST2)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    stage1_payload = _load_pickle(args.stage1)
    prest2_payload = _load_pickle(args.prest2)
    stage1_by_trial = _stage1_records_by_trial_id(stage1_payload)
    candidates = _candidate_records(prest2_payload)
    if args.limit > 0:
        candidates = candidates[: args.limit]
    if not candidates:
        raise RuntimeError(f"No prest2 candidates found in {args.prest2}")

    first_result = Path(str(candidates[0]["result_path"]))
    result_payload = torch.load(first_result, map_location="cpu", weights_only=False)
    base_config_dict = result_payload.get("config", {})
    if not isinstance(base_config_dict, dict):
        raise RuntimeError(f"Missing saved config in {first_result}")

    rows: list[dict[str, Any]] = []
    for cand in candidates:
        trial_id = int(cand.get("trial_id", cand.get("source_stage1_trial_id", 0)))
        mech_winner = int(cand.get("source_mech_winner", 0))
        if trial_id not in stage1_by_trial:
            raise RuntimeError(f"trial_id {trial_id} not found in stage1 payload")
        st1 = stage1_by_trial[trial_id]
        st1_parts = st1.get("train_parts", {}) if isinstance(st1.get("train_parts"), dict) else {}

        cfg = _config_from_saved_dict(base_config_dict, trial_id=trial_id, mech_winner=mech_winner)
        diag = _build_entry_and_preopt(cfg=cfg, trial_id=trial_id, mech_winner=mech_winner)
        split = diag["split"]
        state12_range_scale = np.maximum(np.ptp(split.ode_full[0:2, :], axis=1), SCALE_EPS)
        x2dot_range_scale = max(float(np.ptp(split.x2dot_full)), SCALE_EPS)
        st1_to_stage2_est = _scale_only_stage2_estimate(
            stage1_parts=st1_parts,
            split=split,
            state12_range_scale=state12_range_scale,
            x2dot_range_scale=x2dot_range_scale,
        )

        meta = cand.get("x3_refit_meta", {}) if isinstance(cand.get("x3_refit_meta"), dict) else {}
        row = {
            "candidate_b": int(cand.get("candidate_b", cand.get("candidate", 0))),
            "trial_id": trial_id,
            "source_mech_winner": mech_winner,
            "ks0": float(cand.get("ks0", float("nan"))),
            "cs0": float(cand.get("cs0", float("nan"))),
            "eval1_st1pl_rollout_st1pl_loss_cached": float(st1.get("train_loss", st1.get("loss", float("nan")))),
            "eval1_st1pl_x1_rec_pct": float(st1_parts.get("x1_rec", float("nan"))),
            "eval1_st1pl_x2_rec_pct": float(st1_parts.get("x2_rec", float("nan"))),
            "eval1_st1pl_x2dot_rec_pct": float(st1_parts.get("x2dot_rec", float("nan"))),
            "eval2_st1pl_rollout_stage2_loss_scale_only_est": st1_to_stage2_est["loss"],
            "eval2_stage2_scale_est_x1": st1_to_stage2_est["x1_state"],
            "eval2_stage2_scale_est_x2": st1_to_stage2_est["x2_state"],
            "eval2_stage2_scale_est_x2dot": st1_to_stage2_est["x2dot"],
            "eval3_prest2_pre_refit_rollout_st1pl_style_loss": diag["pre_stage1_style"]["loss"],
            "eval3_pre_st1pl_style_x1_rec_pct": diag["pre_stage1_style"]["x1_rec"],
            "eval3_pre_st1pl_style_x2_rec_pct": diag["pre_stage1_style"]["x2_rec"],
            "eval3_pre_st1pl_style_x2dot_rec_pct": diag["pre_stage1_style"]["x2dot_rec"],
            "eval4_prest2_pre_refit_rollout_stage2_loss_recomputed": diag["pre_stage2_loss"],
            "eval4_prest2_pre_refit_rollout_stage2_loss_saved": float(meta.get("loss_before_x3_refit", float("nan"))),
            "eval4_recompute_minus_saved": diag["pre_stage2_loss"] - float(meta.get("loss_before_x3_refit", float("nan"))),
            "post_refit_stage2_loss_recomputed": diag["post_stage2_loss"],
            "post_refit_stage2_loss_saved": float(meta.get("loss_after_x3_refit", float("nan"))),
            "post_refit_st1pl_style_loss": diag["post_stage1_style"]["loss"],
            "train_loss_start_saved_epoch1": float(cand.get("train_loss_start", float("nan"))),
            "final_train_loss": float(cand.get("final_train_loss", float("nan"))),
            "ratio_eval2_to_eval1": st1_to_stage2_est["loss"] / max(float(st1.get("train_loss", st1.get("loss", float("nan")))), 1.0e-300),
            "ratio_eval3_to_eval1": diag["pre_stage1_style"]["loss"] / max(float(st1.get("train_loss", st1.get("loss", float("nan")))), 1.0e-300),
            "ratio_eval4_to_eval1": diag["pre_stage2_loss"] / max(float(st1.get("train_loss", st1.get("loss", float("nan")))), 1.0e-300),
            "stage1_x1_range_scale": float(state12_range_scale[0]),
            "stage1_x2_range_scale": float(state12_range_scale[1]),
            "stage1_x2dot_range_scale": float(x2dot_range_scale),
            "stage2_x1_rms_scale": st1_to_stage2_est["x1_rms_scale_stage2"],
            "stage2_x2_rms_scale": st1_to_stage2_est["x2_rms_scale_stage2"],
            "stage2_x2dot_rms_scale": st1_to_stage2_est["x2dot_rms_scale_stage2"],
            "x3_refit_mean": diag["x3_refit_mean"],
            "x3_refit_scale": diag["x3_refit_scale"],
            "x3_refit_support_min": diag["x3_refit_support_min"],
            "x3_refit_support_max": diag["x3_refit_support_max"],
            "gain_before_refit_recomputed": diag["gain_before"],
            "gain_after_refit_recomputed": diag["gain_after"],
            "gain_before_refit_saved": float(meta.get("gain_before_refit", float("nan"))),
            "gain_after_refit_saved": float(meta.get("gain_after_refit", float("nan"))),
        }
        rows.append(row)
        print(
            f"B{row['candidate_b']:02d} trial={trial_id} "
            f"eval1={row['eval1_st1pl_rollout_st1pl_loss_cached']:.3e} "
            f"eval2={row['eval2_st1pl_rollout_stage2_loss_scale_only_est']:.3e} "
            f"eval3={row['eval3_prest2_pre_refit_rollout_st1pl_style_loss']:.3e} "
            f"eval4={row['eval4_prest2_pre_refit_rollout_stage2_loss_recomputed']:.3e}"
        )

    csv_path = args.outdir / "afm05_st1pl_vs_prest2_entry_loss_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def stats(key: str) -> dict[str, float]:
        vals = np.asarray([float(r[key]) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
        return {"min": float(np.min(vals)), "median": float(np.median(vals)), "max": float(np.max(vals))}

    summary = {
        "n": len(rows),
        "csv": str(csv_path),
        "eval1_st1pl_loss": stats("eval1_st1pl_rollout_st1pl_loss_cached"),
        "eval2_scale_only_stage2_loss_est": stats("eval2_st1pl_rollout_stage2_loss_scale_only_est"),
        "eval3_prest2_pre_refit_st1pl_style_loss": stats("eval3_prest2_pre_refit_rollout_st1pl_style_loss"),
        "eval4_prest2_pre_refit_stage2_loss": stats("eval4_prest2_pre_refit_rollout_stage2_loss_recomputed"),
        "ratio_eval2_to_eval1": stats("ratio_eval2_to_eval1"),
        "ratio_eval3_to_eval1": stats("ratio_eval3_to_eval1"),
        "ratio_eval4_to_eval1": stats("ratio_eval4_to_eval1"),
        "max_abs_eval4_recompute_minus_saved": float(
            np.nanmax(np.abs(np.asarray([float(r["eval4_recompute_minus_saved"]) for r in rows], dtype=float)))
        ),
    }
    summary_path = args.outdir / "afm05_st1pl_vs_prest2_entry_loss_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    txt_path = args.outdir / "afm05_st1pl_vs_prest2_entry_loss_diagnostics_summary.txt"
    lines = [
        "AFM05 st1pl vs prest2 entry loss diagnostics",
        f"n = {len(rows)}",
        f"CSV = {csv_path}",
        "",
        "Definitions:",
        "eval1 = cached st1pl rollout under st1pl loss",
        "eval2 = st1pl saved x1/x2/x2dot components converted to stage2 RMS scales (scale-only estimate)",
        "eval3 = recomputed prest2 pre-refit rollout under st1pl-style loss",
        "eval4 = recomputed prest2 pre-refit rollout under stage2 loss",
        "",
    ]
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.append(f"{key}: min={value['min']:.6e}, median={value['median']:.6e}, max={value['max']:.6e}")
    lines.append(f"max_abs_eval4_recompute_minus_saved = {summary['max_abs_eval4_recompute_minus_saved']:.6e}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
