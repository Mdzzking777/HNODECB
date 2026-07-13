from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(REPO_ROOT))

from AFM04.prestage2.visualization_runner.plot_afm_prest2_preopt_rank_04 import _payload_from_record  # noqa: E402
from AFM04.stage2light.data import prepare_data  # noqa: E402
from AFM04.stage2light.kan_backend import KANForceModule, initial_grid_support_from_raw_inputs  # noqa: E402
from AFM04.stage2light.rollout import LearnableMechModule, rollout_single_shooting_torch  # noqa: E402
from AFM04.stage2light.runner.visualization.plot_afm_stage2light_preopt_rank_04 import (  # noqa: E402
    _config_from_payload,
    _select_split,
    _torch_dtype,
)
from AFM04.stage2light.train import _observable_grid_inputs_from_ode, _prepared_with_initial_x3_from_warmstart  # noqa: E402

DEFAULT_RESULT_PATH = REPO_ROOT / "AFM04" / "prestage2" / "results" / "afm_prest2_04_candidates_b.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "prestage2" / "visualization"
BATCH_CANDIDATE_START = 1
BATCH_CANDIDATE_END = 20
BATCH_DIR_STEM = "candidate_b_001_020_recon_nn"


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected payload type for {path}: {type(payload)!r}")
    return payload


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("candidate_b_records", "candidate_records", "top_mech_winner_a_records", "newrank_a_records"):
        records = payload.get(key)
        if isinstance(records, list) and records:
            return [rec for rec in records if isinstance(rec, dict)]
    return []


def _candidate_id(record: dict[str, Any]) -> int:
    for key in ("candidate_b", "candidate", "top_mech_winner_a", "newrank_a"):
        try:
            value = int(record.get(key, 0))
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _find_candidate_record(records: list[dict[str, Any]], candidate: int) -> dict[str, Any]:
    wanted = int(candidate)
    for rec in records:
        if _candidate_id(rec) == wanted:
            return rec
    raise ValueError(f"Candidate {wanted} not found. Available candidates: 1..{len(records)}")


def _batch_candidate_ids(records: list[dict[str, Any]]) -> list[int]:
    available = sorted({_candidate_id(rec) for rec in records if _candidate_id(rec) > 0})
    wanted = list(range(BATCH_CANDIDATE_START, BATCH_CANDIDATE_END + 1))
    return [candidate for candidate in wanted if candidate in available]


def _timestamped_subdir(parent: Path, stem: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = parent / f"{stem}_{stamp}"
    if not base.exists():
        return base
    for idx in range(1, 100):
        candidate = parent / f"{stem}_{stamp}_{idx:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique output directory under {parent}")


def _load_history(candidate_record: dict[str, Any]) -> list[dict[str, Any]]:
    history = candidate_record.get("history", [])
    if isinstance(history, list) and history and all(isinstance(row, dict) for row in history):
        return history

    history_path_raw = str(candidate_record.get("history_path", "")).strip()
    if history_path_raw:
        history_path = Path(history_path_raw)
        if history_path.is_file():
            data = _load_json(history_path)
            if isinstance(data, list) and all(isinstance(row, dict) for row in data):
                return data

    raise RuntimeError("Could not load candidate history from merged payload or history_path.")


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _relative_rmse_pct_np(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=float).reshape(-1)
    truth_arr = np.asarray(truth, dtype=float).reshape(-1)
    if pred_arr.shape != truth_arr.shape:
        return float("nan")
    finite = np.isfinite(pred_arr) & np.isfinite(truth_arr)
    if not np.any(finite):
        return float("nan")
    pred_arr = pred_arr[finite]
    truth_arr = truth_arr[finite]
    count = float(max(int(truth_arr.size), 1))
    truth_rms = math.sqrt(float(np.sum(np.square(truth_arr))) / count)
    if not math.isfinite(truth_rms) or truth_rms <= 0.0:
        return float("nan")
    err_rms = math.sqrt(float(np.sum(np.square(pred_arr - truth_arr))) / count)
    return 100.0 * err_rms / truth_rms


def _epoch0_train_metrics(candidate_record: dict[str, Any]) -> dict[str, Any]:
    payload = _payload_from_record(candidate_record)
    warmstart = payload.get("warmstart", {})
    if not isinstance(warmstart, dict) or not warmstart:
        return {
            "train_x1_rec": float("nan"),
            "train_x3_rec": float("nan"),
            "train_nn_err": float("nan"),
            "x3_pred": None,
        }

    cfg = _config_from_payload(payload)
    prepared = prepare_data(cfg)
    split = _select_split(prepared, payload)
    dtype = _torch_dtype(cfg.dtype)
    device = str(cfg.device)
    prepared, x3_norm_support, _x3_source = _prepared_with_initial_x3_from_warmstart(prepared, warmstart)

    ode_train_for_grid = torch.as_tensor(split.ode_train, dtype=dtype, device=device)
    grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=ode_train_for_grid,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        x3_norm_support=x3_norm_support,
    )
    initial_grid_support = initial_grid_support_from_raw_inputs(
        grid_inputs,
        prepared.state_mean,
        prepared.state_scale,
    )

    model = KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=int(warmstart["seed"]),
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
        ks_init=float(warmstart["ks0"]),
        cs_init=float(warmstart["cs0"]),
        ks_bounds=(cfg.ks_lo, cfg.ks_hi),
        cs_bounds=(cfg.cs_lo, cfg.cs_hi),
        dtype=dtype,
        device=device,
        parameterization=getattr(cfg, "mech_parameterization", "sigmoid_bounded"),
    ).to(device)

    train_states_for_gain = torch.as_tensor(split.ode_train.T, dtype=dtype, device=device)
    train_fts = torch.as_tensor(split.fts_train_true, dtype=dtype, device=device)
    ode_true = torch.as_tensor(split.ode_train, dtype=dtype, device=device)
    times = torch.as_tensor(split.times_train, dtype=dtype, device=device)

    with torch.no_grad():
        model.initialize_gain_from_truth(train_states_for_gain, train_fts)
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
        fts_pred = model(traj_pred.transpose(0, 1))

    traj_np = traj_pred.detach().cpu().numpy()
    ode_np = np.asarray(split.ode_train, dtype=float)
    fts_pred_np = fts_pred.detach().cpu().numpy()
    fts_true_np = np.asarray(split.fts_train_true, dtype=float)
    return {
        "train_x1_rec": _relative_rmse_pct_np(traj_np[0, :], ode_np[0, :]),
        "train_x3_rec": _relative_rmse_pct_np(traj_np[2, :], ode_np[2, :]),
        "train_nn_err": _relative_rmse_pct_np(fts_pred_np, fts_true_np),
        "x3_pred": np.asarray(traj_np[2, :], dtype=float).copy(),
    }


def _source_label(record: dict[str, Any]) -> str:
    candidate = _candidate_id(record)
    rank = int(record.get("source_stage1_rank", 0) or 0)
    mech = int(record.get("source_mech_winner", 0) or 0)
    seed = int(record.get("source_stage1_best_seedbank", 0) or 0)
    trial = int(record.get("trial_id", record.get("source_stage1_trial_id", 0)) or 0)
    bits = [f"candidate B={candidate}"]
    if rank > 0:
        bits.append(f"source st1 rank={rank}")
    if mech > 0:
        bits.append(f"mech winner={mech}")
    if seed > 0:
        bits.append(f"NN seed={seed}")
    if trial > 0:
        bits.append(f"trial={trial}")
    return " | ".join(bits)


def _out_path(out_dir: Path, candidate: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"afm_prest2_candidate_b_{int(candidate):03d}_recon_nn.png"


def plot_candidate_recon_nn(candidate_record: dict[str, Any], history: list[dict[str, Any]], out_dir: Path) -> Path:
    candidate = _candidate_id(candidate_record)
    prest2_epochs = [int(float(row.get("epoch", idx + 1))) for idx, row in enumerate(history)]
    epochs = [0, *prest2_epochs]
    epoch0 = _epoch0_train_metrics(candidate_record)
    train_x1 = [float(epoch0["train_x1_rec"]), *[_finite_or_nan(row.get("train_x1_rec")) for row in history]]
    train_x3 = [float(epoch0["train_x3_rec"]), *[_finite_or_nan(row.get("train_x3_rec")) for row in history]]
    train_nn = [
        float(epoch0["train_nn_err"]),
        *[_finite_or_nan(row.get("train_fts_rollout_rec", row.get("train_nn_err"))) for row in history],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=False, squeeze=False)
    axes = axes[0]

    axes[0].plot(epochs, train_x1, color="seagreen", linewidth=2, marker="o", markersize=3, label="x1_rec")
    axes[0].plot(epochs, train_x3, color="purple", linewidth=2, marker="o", markersize=3, label="x3_rec")
    axes[0].set_title("prest2 train rec")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("reconstruction error (%)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="center right")

    axes[1].plot(epochs, train_nn, color="black", linewidth=2, marker="o", markersize=3, label="F_contact err")
    axes[1].set_title("prest2 train NN")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("NN F_contact error (%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle(
        "AFM04 prestage2 training reconstruction / NN metrics\n" + _source_label(candidate_record),
        fontsize=14,
        y=1.03,
    )

    out_path = _out_path(out_dir, candidate)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {out_path}")
    print(_source_label(candidate_record))
    return out_path


def run_one(result_path: Path, out_dir: Path, candidate: int) -> Path:
    payload = _load_pickle(result_path)
    records = _candidate_records(payload)
    if not records:
        raise FileNotFoundError(f"No prest2 candidate records found in: {result_path}")
    record = _find_candidate_record(records, candidate)
    history = _load_history(record)
    return plot_candidate_recon_nn(record, history, out_dir)


def run_all(result_path: Path, out_parent: Path) -> list[Path]:
    payload = _load_pickle(result_path)
    records = _candidate_records(payload)
    if not records:
        raise FileNotFoundError(f"No prest2 candidate records found in: {result_path}")

    candidate_ids = _batch_candidate_ids(records)
    if not candidate_ids:
        raise FileNotFoundError(
            f"No prest2 candidate B records found in the requested range "
            f"{BATCH_CANDIDATE_START}..{BATCH_CANDIDATE_END}: {result_path}"
        )

    out_dir = _timestamped_subdir(out_parent, BATCH_DIR_STEM)
    out_paths: list[Path] = []
    for candidate in candidate_ids:
        record = _find_candidate_record(records, candidate)
        history = _load_history(record)
        out_paths.append(plot_candidate_recon_nn(record, history, out_dir))

    print(f"Saved {len(out_paths)} plots to: {out_dir}")
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prest2 candidate reconstruction and NN metrics over epochs.")
    parser.add_argument("--candidate", "-c", type=int, help="generate only one candidate B id")
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.candidate is not None:
        run_one(args.result_path.resolve(), args.out_dir.resolve(), int(args.candidate))
    else:
        run_all(args.result_path.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
