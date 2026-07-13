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


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(REPO_ROOT))

from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import CS, KS  # noqa: E402

DEFAULT_RESULT_PATH = REPO_ROOT / "AFM04" / "prestage2" / "results" / "afm_prest2_04_candidates_b.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "prestage2" / "visualization"
BATCH_CANDIDATE_START = 1
BATCH_CANDIDATE_END = 20
BATCH_DIR_STEM = "candidate_b_001_020_mech"


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


def _mech_err_pct(est: float, truth: float) -> float:
    if not np.isfinite(est) or not np.isfinite(truth):
        return float("nan")
    return 100.0 * abs(float(est) - float(truth)) / max(abs(float(truth)), 1.0e-30)


def _mech_relative_change_from_epoch0_pct(values: np.ndarray) -> float:
    series = np.asarray(values, dtype=float).reshape(-1)
    finite_idx = np.flatnonzero(np.isfinite(series))
    if finite_idx.size == 0:
        return float("nan")
    start = float(series[finite_idx[0]])
    final = float(series[finite_idx[-1]])
    if not np.isfinite(start) or abs(start) <= 1.0e-30:
        return float("nan")
    return 100.0 * abs(final - start) / abs(start)


def _annotate_epoch0_relative_change(ax, *, values: np.ndarray, label: str) -> None:
    rel_change = _mech_relative_change_from_epoch0_pct(values)
    value_text = f"{rel_change:.3g}%" if np.isfinite(rel_change) else "n/a"
    ax.text(
        0.02,
        0.50,
        f"||{label}(e30)-{label}(e0)||\n/ |{label}(e0)|\n= {value_text}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9,
        family="monospace",
        color="purple",
    )


def _annotation_epochs(epochs: list[int], step: int = 10) -> list[int]:
    epoch_list = [int(ep) for ep in epochs if math.isfinite(float(ep))]
    if not epoch_list:
        return []
    selected = [ep for ep in epoch_list if ep == 0 or ep == 1 or ep % int(step) == 0]
    if epoch_list[-1] not in selected:
        selected.append(epoch_list[-1])
    return selected


def _annotate_err_points(
    ax,
    *,
    epochs: list[int],
    values: np.ndarray,
    truth: float,
    step: int = 10,
) -> None:
    if not np.isfinite(truth):
        return
    epoch_to_idx = {int(ep): idx for idx, ep in enumerate(epochs)}
    for ann_idx, epoch in enumerate(_annotation_epochs(epochs, step=step)):
        idx = epoch_to_idx.get(int(epoch))
        if idx is None:
            continue
        value = float(values[idx])
        err_pct = _mech_err_pct(value, truth)
        if not np.isfinite(value) or not np.isfinite(err_pct):
            continue
        ax.scatter([epoch], [value], color="crimson", s=24, zorder=4)
        offset_y = 10 if ann_idx % 2 == 0 else -14
        va = "bottom" if offset_y > 0 else "top"
        ax.annotate(
            f"{err_pct:.2f}%",
            xy=(epoch, value),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color="crimson",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )


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
    return out_dir / f"afm_prest2_candidate_b_{int(candidate):03d}_mech.png"


def plot_candidate_mech(candidate_record: dict[str, Any], history: list[dict[str, Any]], out_dir: Path) -> Path:
    candidate = _candidate_id(candidate_record)
    prest2_epochs = [int(float(row.get("epoch", idx + 1))) for idx, row in enumerate(history)]
    ks0 = _finite_or_nan(candidate_record.get("ks0"))
    cs0 = _finite_or_nan(candidate_record.get("cs0"))
    epochs = [0, *prest2_epochs]
    ks_series = np.asarray([ks0, *[_finite_or_nan(row.get("ks_hat")) for row in history]], dtype=float)
    cs_series = np.asarray([cs0, *[_finite_or_nan(row.get("cs_hat")) for row in history]], dtype=float)
    mech_true = np.asarray([float(KS), float(CS)], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.8), sharex=False, squeeze=False)
    ax1 = axes[0, 0]
    ax1.plot(epochs, ks_series, color="black", linewidth=2, marker="o", markersize=3, label="ks_hat")
    ax1.axhline(mech_true[0], color="royalblue", linestyle="--", linewidth=2, label="ks_true")
    ax1.set_title("prest2 ks trajectory")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("ks")
    ax1.grid(True, alpha=0.25)
    _annotate_err_points(ax1, epochs=epochs, values=ks_series, truth=float(mech_true[0]))
    _annotate_epoch0_relative_change(ax1, values=ks_series, label="ks")
    ax1.legend(loc="best")

    ax2 = axes[1, 0]
    ax2.plot(epochs, cs_series, color="black", linewidth=2, marker="o", markersize=3, label="cs_hat")
    ax2.axhline(mech_true[1], color="royalblue", linestyle="--", linewidth=2, label="cs_true")
    ax2.set_title("prest2 cs trajectory")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cs")
    ax2.grid(True, alpha=0.25)
    _annotate_err_points(ax2, epochs=epochs, values=cs_series, truth=float(mech_true[1]))
    _annotate_epoch0_relative_change(ax2, values=cs_series, label="cs")
    ax2.legend(loc="best")

    fig.suptitle(
        "AFM04 prestage2 mechanistic parameters\n"
        + _source_label(candidate_record),
        fontsize=14,
        y=1.02,
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
    return plot_candidate_mech(record, history, out_dir)


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
        out_paths.append(plot_candidate_mech(record, history, out_dir))

    print(f"Saved {len(out_paths)} plots to: {out_dir}")
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prest2 candidate mechanistic parameters over epochs.")
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
