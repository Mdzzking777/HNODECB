"""Rerank stage1pluslight advanced mech-grid winners after x3 normalizer refit.

For each mechanical grid point, this script selects the best NN seed from the
completed RS payload.  That pair is the "advanced mech grid" object:

    one mech grid point + its best RS NN seed

The script then rebuilds each selected RS object, rolls it out under the old
x1-prior x3 normalizer, computes an x3_pred-based normalizer from that rollout,
refits the KAN input normalizer/grid, reevaluates the train loss, and writes an
old-rank -> new-rank report.

It does not modify training results or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from AFM04.KAN_full_test.losses import evaluate_split  # noqa: E402
from AFM04.stage1pluslight.kan_rs_trial import (  # noqa: E402
    _observable_grid_inputs_from_ode,
    prepare_kan_stage1_runtime,
)
from AFM04.stage2light.kan_backend import (  # noqa: E402
    KANForceModule,
    initial_grid_support_from_raw_inputs,
)


STAGE_ROOT = ROOT / "AFM04" / "stage1pluslight"
DEFAULT_RESULT_PATH = STAGE_ROOT / "results_afm" / "afm_param_stage1pluslight_04.pkl"
DEFAULT_OUT_DIR = STAGE_ROOT / "logs"
DEFAULT_PREFIX = "afm04_stage1pluslight_x3_refit_advanced_mech_rerank"


def _finite_float(value: Any, default: float = math.nan) -> float:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _params(rec: dict[str, Any]) -> dict[str, Any]:
    params = rec.get("params", {})
    return params if isinstance(params, dict) else {}


def _record_loss(rec: dict[str, Any]) -> float:
    return _finite_float(rec.get("loss", rec.get("train_loss", math.nan)), math.inf)


def _is_usable_record(rec: dict[str, Any]) -> bool:
    if not isinstance(rec, dict):
        return False
    if not math.isfinite(_record_loss(rec)):
        return False
    if bool(rec.get("trial_failed", False)):
        return False
    if rec.get("is_viable", True) is False:
        return False
    return True


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"stage1pluslight result payload not found: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected payload type: {type(payload)!r}")
    return payload


def _trial_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("trial_parameters", [])
    if not isinstance(records, list) or not records:
        records = payload.get("ranked_topk", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError("stage1pluslight payload has no trial records")
    return [rec for rec in records if isinstance(rec, dict)]


def _select_advanced_mech_winners(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one best-seed record for every mech grid point."""

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for rec in records:
        params = _params(rec)
        try:
            key = (int(params["ks_node_idx"]), int(params["cs_node_idx"]))
        except Exception:
            continue
        groups.setdefault(key, []).append(rec)

    winners: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        usable = [rec for rec in group if _is_usable_record(rec)]
        pool = usable if usable else [rec for rec in group if math.isfinite(_record_loss(rec))]
        if not pool:
            continue
        winners.append(min(pool, key=_record_loss))
    return sorted(winners, key=_record_loss)


def _x3_mean_scale_from_traj(traj: torch.Tensor) -> tuple[float, float, float, float, float, float]:
    values = traj[2, :].detach().reshape(-1).cpu().numpy().astype(float)
    if values.size == 0:
        raise ValueError("x3_pred has no samples")
    if not np.all(np.isfinite(values)):
        bad_count = int(values.size - np.count_nonzero(np.isfinite(values)))
        raise ValueError(f"x3_pred contains nonfinite values: count={bad_count}")
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(mean):
        raise ValueError(f"x3_pred mean is nonfinite: {mean}")
    if not np.isfinite(scale) or scale <= 1.0e-30:
        raise ValueError(f"x3_pred scale is invalid: {scale:.6e}")
    norm = (values - mean) / scale
    return (
        mean,
        scale,
        float(np.min(values)),
        float(np.max(values)),
        float(np.min(norm)),
        float(np.max(norm)),
    )


def _observable_grid_inputs_with_x3_norm_support(
    *,
    ode_train: torch.Tensor,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    x3_norm_support: tuple[float, float],
) -> torch.Tensor:
    """Build raw AGU samples whose normalized x3 axis spans x3_norm_support."""

    n = int(ode_train.shape[1])
    inputs = torch.empty((n, 3), dtype=ode_train.dtype, device=ode_train.device)
    inputs[:, 0:2] = ode_train[0:2, :].transpose(0, 1)

    mean = np.asarray(state_mean, dtype=float).reshape(-1)
    scale = np.asarray(state_scale, dtype=float).reshape(-1)
    if n <= 1:
        inputs[:, 2] = float(mean[2])
        return inputs

    lo, hi = float(x3_norm_support[0]), float(x3_norm_support[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(f"invalid x3_norm_support: {x3_norm_support}")
    x3_norm_axis = torch.linspace(lo, hi, n, dtype=ode_train.dtype, device=ode_train.device)
    inputs[:, 2] = torch.as_tensor(float(mean[2]), dtype=ode_train.dtype, device=ode_train.device) + (
        torch.as_tensor(float(scale[2]), dtype=ode_train.dtype, device=ode_train.device) * x3_norm_axis
    )
    return inputs


def _build_model(
    *,
    runtime: dict[str, Any],
    rec: dict[str, Any],
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    initial_grid_inputs: torch.Tensor,
) -> KANForceModule:
    cfg = runtime["cfg"]
    prepared = runtime["prepared"]
    dtype: torch.dtype = runtime["dtype"]
    params = _params(rec)

    cfg = replace(
        cfg,
        grid=int(params.get("kan_grid", cfg.grid)),
        spline_k=int(params.get("kan_spline_k", cfg.spline_k)),
        base_fun=str(params.get("kan_base_fun", cfg.base_fun)),
    )
    initial_grid_support = initial_grid_support_from_raw_inputs(
        initial_grid_inputs,
        state_mean,
        state_scale,
    )
    return KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=state_mean,
        state_scale=state_scale,
        seed=int(params["nn_init_seed"]),
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
        dist=float(prepared.known_pars[6]),
        a0=float(prepared.known_pars[9]),
        gnn_learnable=cfg.gnn_learnable,
        soft_mask_enabled=True,
        soft_mask_trainable=False,
        soft_mask_s0_a0=20.0,
        soft_mask_s0_min_a0=1.0,
        soft_mask_s0_max_a0=100.0,
        soft_mask_alpha_a0=0.25,
        soft_mask_alpha_min_a0=0.02,
        soft_mask_alpha_max_a0=5.0,
        device=cfg.device,
        dtype=dtype,
    ).to(cfg.device)


def _evaluate_record_refit(
    *,
    runtime: dict[str, Any],
    rec: dict[str, Any],
) -> dict[str, Any]:
    cfg = runtime["cfg"]
    prepared = runtime["prepared"]
    tensors = runtime["tensors"]
    train_states = runtime["train_states"]
    train_fts = runtime["train_fts"]
    dtype: torch.dtype = runtime["dtype"]
    params = _params(rec)

    old_mean = np.asarray(prepared.state_mean, dtype=float).copy()
    old_scale = np.asarray(prepared.state_scale, dtype=float).copy()
    old_grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=tensors["ode_train"],
        state_mean=old_mean,
        state_scale=old_scale,
    )
    mech = torch.as_tensor([float(params["ks0"]), float(params["cs0"])], dtype=dtype, device=cfg.device)
    model = _build_model(
        runtime=runtime,
        rec=rec,
        state_mean=old_mean,
        state_scale=old_scale,
        initial_grid_inputs=old_grid_inputs,
    )

    with torch.no_grad():
        if cfg.adaptive_grid_enabled:
            model.update_grid_from_normalized_inputs(old_grid_inputs)
        old_gain = float(model.initialize_gain_from_truth(train_states, train_fts))
        old_total, old_parts, old_traj = evaluate_split(
            force_module=model,
            known_pars=prepared.known_pars,
            mech_true=mech,
            ode_true=tensors["ode_full"],
            x2dot_true=tensors["x2dot_full"],
            contact_mask=tensors["contact_full"],
            times=tensors["times_full"],
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            eta_star_true=prepared.eta_star_true,
            x1_abs_guard=runtime.get("x1_abs_guard"),
            x2_abs_guard=runtime.get("x2_abs_guard"),
            loss_indices=tensors["train_idx"],
        )
        x3_mean, x3_scale, x3_min, x3_max, x3_norm_min, x3_norm_max = _x3_mean_scale_from_traj(old_traj)

        new_mean = old_mean.copy()
        new_scale = old_scale.copy()
        new_mean[2] = float(x3_mean)
        new_scale[2] = float(x3_scale)
        model.set_state_normalizer(new_mean, new_scale)
        new_grid_inputs = _observable_grid_inputs_with_x3_norm_support(
            ode_train=tensors["ode_train"],
            state_mean=new_mean,
            state_scale=new_scale,
            x3_norm_support=(x3_norm_min, x3_norm_max),
        )
        if cfg.adaptive_grid_enabled:
            model.update_grid_from_normalized_inputs(new_grid_inputs)
        new_gain = float(model.initialize_gain_from_truth(train_states, train_fts))
        new_total, new_parts, _new_traj = evaluate_split(
            force_module=model,
            known_pars=prepared.known_pars,
            mech_true=mech,
            ode_true=tensors["ode_full"],
            x2dot_true=tensors["x2dot_full"],
            contact_mask=tensors["contact_full"],
            times=tensors["times_full"],
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            eta_star_true=prepared.eta_star_true,
            x1_abs_guard=runtime.get("x1_abs_guard"),
            x2_abs_guard=runtime.get("x2_abs_guard"),
            loss_indices=tensors["train_idx"],
        )

    old_loss_payload = _record_loss(rec)
    old_loss_eval = float(old_total.detach())
    new_loss = float(new_total.detach())
    return {
        "old_loss_payload": old_loss_payload,
        "old_loss_eval": old_loss_eval,
        "old_eval_minus_payload": old_loss_eval - old_loss_payload,
        "new_loss": new_loss,
        "loss_delta_new_minus_old_payload": new_loss - old_loss_payload,
        "loss_ratio_new_over_old_payload": new_loss / old_loss_payload if old_loss_payload > 0.0 else float("inf"),
        "x3_refit_mean": float(x3_mean),
        "x3_refit_scale": float(x3_scale),
        "x3_pred_min": float(x3_min),
        "x3_pred_max": float(x3_max),
        "x3_norm_support_min": float(x3_norm_min),
        "x3_norm_support_max": float(x3_norm_max),
        "old_x3_prior_mean": float(old_mean[2]),
        "old_x3_prior_scale": float(old_scale[2]),
        "normalizer_mean_shift_old_scale": abs(float(x3_mean) - float(old_mean[2])) / max(abs(float(old_scale[2])), 1.0e-30),
        "normalizer_scale_ratio": float(x3_scale) / max(abs(float(old_scale[2])), 1.0e-30),
        "old_gain": float(old_gain),
        "new_gain": float(new_gain),
        "old_x1_rec": float(old_parts.x1_rec),
        "new_x1_rec": float(new_parts.x1_rec),
        "old_x3_rec": float(old_parts.x3_rec),
        "new_x3_rec": float(new_parts.x3_rec),
        "old_fts_rec": float(getattr(old_parts, "fts_rollout_rec", float("nan"))),
        "new_fts_rec": float(getattr(new_parts, "fts_rollout_rec", float("nan"))),
        "failure_reason": "",
    }


def _base_row_from_record(rec: dict[str, Any]) -> dict[str, Any]:
    params = _params(rec)
    return {
        "node_label": str(params.get("node_label", "")),
        "ks_node_idx": int(params.get("ks_node_idx", -1)),
        "cs_node_idx": int(params.get("cs_node_idx", -1)),
        "trial_id": int(params.get("trial_id", -1)),
        "nn_seed_bank_idx": int(params.get("nn_seed_bank_idx", -1)),
        "nn_init_seed": int(params.get("nn_init_seed", -1)),
        "ks": float(params.get("ks0", rec.get("ks_hat", float("nan")))),
        "cs": float(params.get("cs0", rec.get("cs_hat", float("nan")))),
        "ks_err_pct": float(rec.get("ks_err_pct", float("nan"))),
        "cs_err_pct": float(rec.get("cs_err_pct", float("nan"))),
        "old_loss_payload": _record_loss(rec),
        "old_x3_rec_payload": float(rec.get("val_parts", {}).get("x3_rec", float("nan"))),
        "old_fts_rec_payload": float(rec.get("val_parts", {}).get("fts_teacher_rec", rec.get("val_nn_err", float("nan")))),
    }


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old_order = sorted(rows, key=lambda r: (not math.isfinite(float(r["old_loss_payload"])), float(r["old_loss_payload"])))
    new_order = sorted(rows, key=lambda r: (not math.isfinite(float(r["new_loss"])), float(r["new_loss"])))
    for idx, row in enumerate(old_order, start=1):
        row["old_advanced_mech_rank"] = int(idx)
    for idx, row in enumerate(new_order, start=1):
        row["new_advanced_mech_rank"] = int(idx)
    for row in rows:
        row["rank_delta_new_minus_old"] = int(row["new_advanced_mech_rank"]) - int(row["old_advanced_mech_rank"])
    return sorted(rows, key=lambda r: int(r["new_advanced_mech_rank"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6e}"


def _write_txt(path: Path, rows: list[dict[str, Any]], *, metadata: dict[str, Any]) -> None:
    old_order = sorted(rows, key=lambda r: int(r["old_advanced_mech_rank"]))
    new_order = sorted(rows, key=lambda r: int(r["new_advanced_mech_rank"]))

    def write_rank_row(row: dict[str, Any], *, primary: str) -> None:
        if primary == "old":
            f.write(
                f"{int(row['old_advanced_mech_rank']):8d}  "
                f"{int(row['new_advanced_mech_rank']):8d}  "
            )
        else:
            f.write(
                f"{int(row['new_advanced_mech_rank']):8d}  "
                f"{int(row['old_advanced_mech_rank']):8d}  "
            )
        f.write(
            f"{int(row['rank_delta_new_minus_old']):5d}  "
            f"{row['node_label']:<8}  "
            f"{int(row['trial_id']):7d}  "
            f"{int(row['nn_seed_bank_idx']):4d}  "
            f"{float(row['ks']):11.4e}  "
            f"{float(row['ks_err_pct']):8.2f}  "
            f"{float(row['cs']):11.4e}  "
            f"{float(row['cs_err_pct']):8.2f}  "
            f"{_fmt(row['old_loss_payload']):>11}  "
            f"{_fmt(row['new_loss']):>11}  "
            f"{_fmt(row['loss_ratio_new_over_old_payload']):>9}  "
            f"{float(row['x3_refit_mean']) * 1.0e9:10.4f}  "
            f"{float(row['x3_refit_scale']) * 1.0e9:11.4f}  "
            f"{float(row['old_x3_rec']):7.2f}->{float(row['new_x3_rec']):7.2f}\n"
        )

    with path.open("w", encoding="utf-8") as f:
        f.write("AFM04 stage1pluslight x3-refit advanced mech rerank\n")
        f.write("Object = one mech grid point + its old-RS best NN seed\n")
        f.write("Old framework = x3_mean=x1_mean, x3_scale=0.1*x1_scale\n")
        f.write("New framework = refit x3_mean/x3_scale from that object's RS x3_pred rollout\n")
        f.write("New AGU x3 support = min/max of the same normalized x3_pred, no manual margin\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\nTop by old rank:\n")
        f.write(
            "old_rank  new_rank  delta  node      trial    seed  "
            "ks          ks_err%   cs          cs_err%   "
            "old_loss      new_loss      ratio      x3_mean_nm  x3_scale_nm  x3_rec_old->new\n"
        )
        f.write("-" * 178 + "\n")
        for row in old_order[: min(30, len(old_order))]:
            write_rank_row(row, primary="old")
        f.write("\nTop by new rank:\n")
        f.write(
            "new_rank  old_rank  delta  node      trial    seed  "
            "ks          ks_err%   cs          cs_err%   "
            "old_loss      new_loss      ratio      x3_mean_nm  x3_scale_nm  x3_rec_old->new\n"
        )
        f.write("-" * 178 + "\n")
        for row in new_order[: min(30, len(new_order))]:
            write_rank_row(row, primary="new")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=0, help="debug limit; default 0 means all mech winners")
    parser.add_argument("--no-timestamp", action="store_true", help="write stable filenames without a timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_payload(args.result)
    records = _trial_records(payload)
    winners = _select_advanced_mech_winners(records)
    if args.limit and args.limit > 0:
        winners = winners[: int(args.limit)]
    if not winners:
        raise RuntimeError("no advanced mech-grid winners found")

    runtime = prepare_kan_stage1_runtime(
        ROOT,
        window_mode=str(payload.get("stage1plus_window_mode")) if payload.get("stage1plus_window_mode") else None,
        arch_window_us=float(payload.get("stage1plus_arch_window_us")) if payload.get("stage1plus_arch_window_us") else None,
    )
    prepared = runtime["prepared"]
    old_mean = np.asarray(prepared.state_mean, dtype=float)
    old_scale = np.asarray(prepared.state_scale, dtype=float)
    split = runtime["split"]

    rows: list[dict[str, Any]] = []
    t0 = perf_counter()
    for idx, rec in enumerate(winners, start=1):
        row = _base_row_from_record(rec)
        try:
            row.update(_evaluate_record_refit(runtime=runtime, rec=rec))
        except Exception as exc:
            row.update(
                {
                    "old_loss_eval": float("nan"),
                    "old_eval_minus_payload": float("nan"),
                    "new_loss": float("inf"),
                    "loss_delta_new_minus_old_payload": float("inf"),
                    "loss_ratio_new_over_old_payload": float("inf"),
                    "x3_refit_mean": float("nan"),
                    "x3_refit_scale": float("nan"),
                    "x3_pred_min": float("nan"),
                    "x3_pred_max": float("nan"),
                    "x3_norm_support_min": float("nan"),
                    "x3_norm_support_max": float("nan"),
                    "old_x3_prior_mean": float(old_mean[2]),
                    "old_x3_prior_scale": float(old_scale[2]),
                    "normalizer_mean_shift_old_scale": float("nan"),
                    "normalizer_scale_ratio": float("nan"),
                    "old_gain": float("nan"),
                    "new_gain": float("nan"),
                    "old_x1_rec": float("nan"),
                    "new_x1_rec": float("nan"),
                    "old_x3_rec": float("nan"),
                    "new_x3_rec": float("nan"),
                    "old_fts_rec": float("nan"),
                    "new_fts_rec": float("nan"),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)
        print(
            f"[{idx:03d}/{len(winners):03d}] {row['node_label']} trial={row['trial_id']} "
            f"old={_fmt(row['old_loss_payload'])} new={_fmt(row['new_loss'])} "
            f"reason={row.get('failure_reason', '')}"
        )

    rows = _rank_rows(rows)
    elapsed_sec = float(perf_counter() - t0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.tag.strip() or ("stable" if args.no_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S"))
    prefix = f"{DEFAULT_PREFIX}_{suffix}"
    csv_path = args.out_dir / f"{prefix}.csv"
    json_path = args.out_dir / f"{prefix}.json"
    txt_path = args.out_dir / f"{prefix}.txt"

    metadata = {
        "result_path": str(args.result.resolve()),
        "window_mode": str(payload.get("stage1plus_window_mode", "")),
        "arch_window_us": float(payload.get("stage1plus_arch_window_us", float("nan"))),
        "window_role": str(split.role),
        "window_label": str(split.label),
        "advanced_mech_objects": len(rows),
        "old_x3_prior_mean_m": float(old_mean[2]),
        "old_x3_prior_scale_m": float(old_scale[2]),
        "old_x3_prior_mean_nm": float(old_mean[2] * 1.0e9),
        "old_x3_prior_scale_nm": float(old_scale[2] * 1.0e9),
        "elapsed_sec": elapsed_sec,
    }
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _write_txt(txt_path, rows, metadata=metadata)

    print(f"wrote CSV  -> {csv_path}")
    print(f"wrote JSON -> {json_path}")
    print(f"wrote TXT  -> {txt_path}")


if __name__ == "__main__":
    main()
