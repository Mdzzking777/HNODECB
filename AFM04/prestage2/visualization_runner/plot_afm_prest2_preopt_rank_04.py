from __future__ import annotations

import argparse
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from AFM04.stage2light.runner.visualization.plot_afm_stage2light_preopt_rank_04 import (
    _build_preopt_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREST2_FINAL = REPO_ROOT / "AFM04" / "prestage2" / "results" / "afm_prest2_04_candidates_b.pkl"
DEFAULT_PREST2_LAYER_A = REPO_ROOT / "AFM04" / "prestage2" / "results" / "afm_prest2_04_top_mech_winners_a.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "prestage2" / "visualization"
DEFAULT_CANDIDATE = 11
BATCH_CANDIDATE_START = 1
BATCH_CANDIDATE_END = 20
BATCH_DIR_STEM = "candidate_b_001_020_preopt_grid"

_AGU_WARMSTART_KEYS = (
    "initial_grid_support_source",
    "initial_grid_support",
    "initial_grid_support_meta",
    "initial_grid_support_x1_min",
    "initial_grid_support_x1_max",
    "initial_grid_support_x2_min",
    "initial_grid_support_x2_max",
    "initial_grid_support_x3_min",
    "initial_grid_support_x3_max",
    "rs_x3_normalizer_valid",
    "rs_x3_normalizer_source",
    "rs_x3_pred_mean",
    "rs_x3_pred_scale",
    "rs_x3_pred_min",
    "rs_x3_pred_max",
    "rs_x3_norm_support_min",
    "rs_x3_norm_support_max",
    "x3_refit_meta",
    "x3_agu_support_current",
)


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected payload type for {path}: {type(payload)!r}")
    return payload


def _candidate_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "candidate_b_records",
        "candidate_records",
        "top_mech_winner_a_records",
        "newrank_a_records",
        "ranked_candidates",
        "candidates",
    ):
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


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _record_label(record: dict[str, Any]) -> str:
    candidate_b = int(record.get("candidate_b", record.get("candidate", 0)) or 0)
    layer_a = int(record.get("top_mech_winner_a", record.get("newrank_a", 0)) or 0)
    rank = int(record.get("source_stage1_rank", 0) or 0)
    mech = int(record.get("source_mech_winner", 0) or 0)
    trial = int(record.get("trial_id", record.get("source_stage1_trial_id", 0)) or 0)
    seed = int(record.get("source_stage1_best_seedbank", 0) or 0)

    parts: list[str] = []
    if candidate_b > 0:
        parts.append(f"candidate B={candidate_b}")
    elif layer_a > 0:
        parts.append(f"Layer-A top mech winner={layer_a}")
    if rank > 0:
        parts.append(f"source st1 rank={rank}")
    if mech > 0:
        parts.append(f"mech winner={mech}")
    if seed > 0:
        parts.append(f"NN seed={seed}")
    if trial > 0:
        parts.append(f"trial={trial}")
    return " | ".join(parts) if parts else "prest2 source record"


def _x3_refit_label(record: dict[str, Any]) -> str:
    refit = record.get("x3_refit_meta")
    if isinstance(refit, dict):
        try:
            mean = float(refit["refit_x3_mean"])
            scale = float(refit["refit_x3_scale"])
            lo = float(refit["x3_norm_support_min"])
            hi = float(refit["x3_norm_support_max"])
        except Exception:
            return "x3 refit: after prest2 x3-refit"
        return (
            "x3 refit: after prest2 x3-refit | "
            f"mean={mean:.6e}, scale={scale:.6e}, support=[{lo:.6f}, {hi:.6f}]"
        )

    if bool(record.get("rs_x3_normalizer_valid", False)):
        mean = _finite_or_nan(record.get("rs_x3_pred_mean"))
        scale = _finite_or_nan(record.get("rs_x3_pred_scale"))
        lo = _finite_or_nan(record.get("rs_x3_norm_support_min"))
        hi = _finite_or_nan(record.get("rs_x3_norm_support_max"))
        return (
            "x3 refit: after prest2 x3-refit | "
            f"mean={mean:.6e}, scale={scale:.6e}, support=[{lo:.6f}, {hi:.6f}]"
        )

    return "x3 refit: not available"


def _record_to_warmstart(record: dict[str, Any]) -> dict[str, Any]:
    candidate = _candidate_id(record)
    original_rank = int(record.get("source_stage1_rank", record.get("original_rank", 0)) or 0)
    source_mech_winner = int(record.get("source_mech_winner", 0) or 0)
    trial_id = int(record.get("trial_id", record.get("source_stage1_trial_id", 0)) or 0)
    ranking_loss = _finite_or_nan(
        record.get("candidate_loss", record.get("final_train_loss", record.get("final_val_loss", float("nan"))))
    )
    seed = int(record.get("seed", record.get("source_stage1_best_initseed", 0)) or 0)
    ks0 = _finite_or_nan(record.get("ks0", float("nan")))
    cs0 = _finite_or_nan(record.get("cs0", float("nan")))

    if seed <= 0:
        raise RuntimeError(f"Selected prest2 record is missing a positive seed: {_record_label(record)}")
    if not (math.isfinite(ks0) and math.isfinite(cs0)):
        raise RuntimeError(f"Selected prest2 record is missing finite ks0/cs0: {_record_label(record)}")
    if trial_id <= 0:
        raise RuntimeError(f"Selected prest2 record is missing a positive trial_id: {_record_label(record)}")

    warmstart: dict[str, Any] = {
        "source": "prest2",
        "path": str(DEFAULT_PREST2_FINAL),
        "candidate": int(candidate),
        "candidate_b": int(record.get("candidate_b", record.get("candidate", 0)) or 0),
        "original_rank": int(original_rank),
        "source_stage1_rank": int(original_rank),
        "source_mech_winner": int(source_mech_winner),
        "label": _record_label(record),
        "trial_id": int(trial_id),
        "loss": float(ranking_loss),
        "seed": int(seed),
        "ks0": float(ks0),
        "cs0": float(cs0),
        "result_path": str(record.get("result_path", "")),
        "st2l_start_policy": "restart_from_stage1_initial_state",
        "prest2_final_state_used": False,
    }
    for key in _AGU_WARMSTART_KEYS:
        if key in record:
            warmstart[key] = record[key]
    return warmstart


def _payload_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "warmstart": _record_to_warmstart(record),
        "warmstart_source": "prest2",
        "window_meta": record.get("window_meta", {}),
    }


def _find_by_candidate(records: list[dict[str, Any]], candidate: int) -> dict[str, Any]:
    wanted = int(candidate)
    for rec in records:
        if _candidate_id(rec) == wanted:
            return rec
    raise ValueError(f"Candidate {wanted} was not found in the selected prest2 payload.")


def _find_by_source_rank(records: list[dict[str, Any]], rank: int) -> dict[str, Any] | None:
    wanted = int(rank)
    for rec in records:
        try:
            source_rank = int(rec.get("source_stage1_rank", rec.get("original_rank", 0)) or 0)
        except Exception:
            source_rank = 0
        if source_rank == wanted:
            return rec
    return None


def select_record(
    *,
    result_path: Path,
    layer_a_path: Path,
    candidate: int | None,
    rank: int | None,
) -> tuple[dict[str, Any], str]:
    payload = _load_pickle(result_path)
    records = _candidate_records(payload)
    if not records:
        raise FileNotFoundError(f"No prest2 candidate records found in: {result_path}")

    if candidate is not None:
        rec = _find_by_candidate(records, candidate)
        return rec, _candidate_suffix(rec)

    if rank is not None:
        rec = _find_by_source_rank(records, rank)
        if rec is not None:
            return rec, _candidate_suffix(rec)

        if layer_a_path != result_path and layer_a_path.is_file():
            layer_a_payload = _load_pickle(layer_a_path)
            layer_a_records = _candidate_records(layer_a_payload)
            rec = _find_by_source_rank(layer_a_records, rank)
            if rec is not None:
                return rec, _candidate_suffix(rec)

        raise ValueError(f"Source stage1 rank {int(rank)} was not found in final B or Layer-A prest2 payloads.")

    rec = _find_by_candidate(records, DEFAULT_CANDIDATE)
    return rec, _candidate_suffix(rec)


def _candidate_suffix(record: dict[str, Any]) -> str:
    candidate_b = int(record.get("candidate_b", record.get("candidate", 0)) or 0)
    if candidate_b > 0:
        return f"candidate_b_{candidate_b:03d}"
    layer_a = int(record.get("top_mech_winner_a", record.get("newrank_a", 0)) or 0)
    if layer_a > 0:
        return f"candidate_a_{layer_a:03d}"
    source_rank = int(record.get("source_stage1_rank", record.get("original_rank", 0)) or 0)
    if source_rank > 0:
        return f"candidate_unknown_source_rank_{source_rank:03d}"
    return "candidate_unknown"


def _out_path(out_dir: Path, suffix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"afm_prest2_preopt_{suffix}_grid.png"


def plot_snapshot(record: dict[str, Any], suffix: str, out_dir: Path) -> Path:
    snap = _build_preopt_snapshot(_payload_from_record(record))
    times_us = np.asarray(snap["times_us"], dtype=float)
    ode_true = np.asarray(snap["ode_true"], dtype=float)
    traj_pred = np.asarray(snap["traj_pred"], dtype=float)
    x2dot_true = np.asarray(snap["x2dot_true"], dtype=float)
    x2dot_pred = np.asarray(snap["x2dot_pred"], dtype=float)
    fts_true = np.asarray(snap["fts_true"], dtype=float)
    fts_pred = np.asarray(snap["fts_pred"], dtype=float)

    fig, axes = plt.subplots(1, 5, figsize=(30, 4.8), squeeze=False, sharex=False)
    axes = axes[0]

    series = [
        (
            "Fts before prest2 optimization\nafter x3-refit",
            fts_true,
            fts_pred,
            "force (N)",
            "Fts teacher true",
            "Fts rollout pred",
        ),
        (
            "x1 before prest2 optimization\nafter x3-refit",
            ode_true[0, :],
            traj_pred[0, :],
            "x1",
            "x1 true",
            "x1 pred",
        ),
        (
            "x2 before prest2 optimization\nafter x3-refit",
            ode_true[1, :],
            traj_pred[1, :],
            "x2",
            "x2 true",
            "x2 pred",
        ),
        (
            "x2dot before prest2 optimization\nafter x3-refit",
            x2dot_true,
            x2dot_pred,
            "x2dot",
            "x2dot true",
            "x2dot pred",
        ),
        (
            "x3 before prest2 optimization\nafter x3-refit",
            ode_true[2, :],
            traj_pred[2, :],
            "x3",
            "x3 true",
            "x3 pred",
        ),
    ]

    for ax, (title, true_values, pred_values, ylabel, true_label, pred_label) in zip(axes, series):
        ax.plot(times_us, true_values, color="black", linewidth=2, label=true_label)
        ax.plot(times_us, pred_values, color="crimson", linewidth=2, linestyle="--", label=pred_label)
        ax.set_title(title)
        ax.set_xlabel("time (us)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle(
        "AFM04 prestage2 warmstart source snapshot | X3-REFIT: AFTER\n"
        "before prest2 optimization, after x3-refit\n"
        + _x3_refit_label(record)
        + "\n"
        + _record_label(record),
        fontsize=14,
        y=1.03,
    )

    out_path = _out_path(out_dir, suffix)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {out_path}")
    print(_record_label(record))
    return out_path


def run_one(
    *,
    result_path: Path = DEFAULT_PREST2_FINAL,
    layer_a_path: Path = DEFAULT_PREST2_LAYER_A,
    out_dir: Path = DEFAULT_OUT_DIR,
    candidate: int | None = None,
    rank: int | None = None,
) -> Path:
    record, suffix = select_record(
        result_path=result_path,
        layer_a_path=layer_a_path,
        candidate=candidate,
        rank=rank,
    )
    return plot_snapshot(record, suffix, out_dir)


def run_all(
    *,
    result_path: Path = DEFAULT_PREST2_FINAL,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> list[Path]:
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

    batch_out_dir = _timestamped_subdir(out_dir, BATCH_DIR_STEM)
    out_paths: list[Path] = []
    for candidate in candidate_ids:
        record = _find_by_candidate(records, candidate)
        out_paths.append(plot_snapshot(record, _candidate_suffix(record), batch_out_dir))

    print(f"Saved {len(out_paths)} plots to: {batch_out_dir}")
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AFM04 prestage2 pre-optimization warmstart rollout for a selected candidate or source rank."
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--candidate", "-c", type=int, help="prest2 candidate B id, e.g. 11")
    selector.add_argument("--rank", "-r", type=int, help="source stage1pluslight rank, e.g. 247")
    parser.add_argument("--result-path", type=Path, default=DEFAULT_PREST2_FINAL)
    parser.add_argument("--layer-a-path", type=Path, default=DEFAULT_PREST2_LAYER_A)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.candidate is None and args.rank is None:
        run_all(
            result_path=args.result_path.resolve(),
            out_dir=args.out_dir.resolve(),
        )
    else:
        run_one(
            result_path=args.result_path.resolve(),
            layer_a_path=args.layer_a_path.resolve(),
            out_dir=args.out_dir.resolve(),
            candidate=args.candidate,
            rank=args.rank,
        )


if __name__ == "__main__":
    main()
