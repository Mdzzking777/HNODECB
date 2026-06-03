from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
DEFAULT_RESULT_PATH = REPO_ROOT / "AFM04" / "prestage2" / "results" / "afm_prest2_04_candidates_b.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "prestage2" / "visualization"
DEFAULT_CANDIDATE = 5


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
    records = payload.get("candidate_b_records", [])
    if isinstance(records, list) and records:
        return [rec for rec in records if isinstance(rec, dict)]
    records = payload.get("candidate_records", [])
    if isinstance(records, list) and records:
        return [rec for rec in records if isinstance(rec, dict)]
    records = payload.get("top_mech_winner_a_records", payload.get("newrank_a_records", []))
    return [rec for rec in records if isinstance(rec, dict)]


def _find_candidate_record(records: list[dict[str, Any]], candidate: int) -> dict[str, Any]:
    for rec in records:
        rank_id = int(rec.get("candidate_b", rec.get("candidate", rec.get("top_mech_winner_a", rec.get("newrank_a", -1)))))
        if rank_id == int(candidate):
            return rec
    raise ValueError(f"Candidate B {candidate} not found. Available candidates: 1..{len(records)}")


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

    result_path_raw = str(candidate_record.get("result_path", "")).strip()
    if result_path_raw:
        result_path = Path(result_path_raw)
        if result_path.is_file():
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Loading prest2 result history from '.pt' requires a Python environment with 'torch'."
                ) from exc
            payload = torch.load(result_path, map_location="cpu", weights_only=False)
            hist = payload.get("history", [])
            if isinstance(hist, list) and all(isinstance(row, dict) for row in hist):
                return hist

    raise RuntimeError("Could not load candidate history from merged payload, history_path, or result_path.")


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _positive_values(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v) and v > 0.0]


def _mw_seed_rank_label(record: dict[str, Any]) -> str:
    mw = int(record.get("source_mech_winner", 0))
    seed = int(record.get("source_stage1_best_seedbank", 0))
    rank = int(record.get("source_stage1_rank", 0))
    if mw > 0 and seed > 0 and rank > 0:
        return f"mech winner {mw} +NN seed {seed} (rank {rank})"
    if mw > 0 and seed > 0:
        return f"mech winner {mw} +NN seed {seed}"
    if mw > 0 and rank > 0:
        return f"mech winner {mw} (rank {rank})"
    return f"mech winner {mw}"


def _out_path(out_dir: Path, candidate: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"afm_prest2_candidate_b_{candidate:03d}_loss.png"


def plot_candidate_losses(candidate_record: dict[str, Any], history: list[dict[str, Any]], out_dir: Path) -> Path:
    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(history)]
    train_losses = [_finite_or_nan(row.get("train_loss", float("nan"))) for row in history]
    val_losses = [_finite_or_nan(row.get("val_loss", float("nan"))) for row in history]

    y_values = _positive_values(train_losses) + _positive_values(val_losses)
    y_min = min(y_values) / 1.8 if y_values else 1.0e-12
    y_max = max(y_values) * 1.8 if y_values else 1.0e-6

    candidate = int(candidate_record.get("candidate_b", candidate_record.get("candidate", candidate_record.get("top_mech_winner_a", candidate_record.get("newrank_a", 0)))))
    source_label = _mw_seed_rank_label(candidate_record)
    trial_id = int(candidate_record.get("trial_id", 0))
    final_val_epoch = int(candidate_record.get("final_val_epoch", -1))
    final_val = _finite_or_nan(candidate_record.get("final_val_loss", float("nan")))
    rank_label = "candidate B" if int(candidate_record.get("candidate_b", candidate_record.get("candidate", 0))) > 0 else "top mech winner A"

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, max(epochs) + 1 if epochs else 1)

    ax.plot(epochs, train_losses, label="train", color="steelblue", linewidth=2, marker="o", markersize=4)
    if any(math.isfinite(v) for v in val_losses):
        ax.plot(epochs, val_losses, label="final validation", color="firebrick", linewidth=2, linestyle="--", marker="D", markersize=4)

    if math.isfinite(final_val):
        title = (
            f"AFM04 prest2 {rank_label} {candidate} loss curve\n"
            f"from {source_label} | trial {trial_id} | final val={final_val:.6e} @ e{final_val_epoch}"
        )
    else:
        title = (
            f"AFM04 prest2 {rank_label} {candidate} loss curve\n"
            f"from {source_label} | trial {trial_id}"
        )
    ax.set_title(title)
    ax.legend(loc="best")

    text_lines = [
        f"epochs={len(history)}",
        f"final train={_finite_or_nan(candidate_record.get('final_train_loss')):.6e}",
        f"final val={_finite_or_nan(candidate_record.get('final_val_loss')):.6e}",
        f"ks={_finite_or_nan(candidate_record.get('ks_hat')):.6e} ({_finite_or_nan(candidate_record.get('ks_err_pct')):.2f}%)",
        f"cs={_finite_or_nan(candidate_record.get('cs_hat')):.6e} ({_finite_or_nan(candidate_record.get('cs_err_pct')):.2f}%)",
        f"val x1={_finite_or_nan(candidate_record.get('val_x1_rec')):.2f}%",
        f"val x3={_finite_or_nan(candidate_record.get('val_x3_rec')):.2f}%",
        f"val nn={_finite_or_nan(candidate_record.get('val_nn_err')):.2f}%",
    ]
    ax.text(
        1.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    out_path = _out_path(out_dir, candidate)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_one(result_path: Path, out_dir: Path, candidate: int) -> Path:
    payload = _load_pickle(result_path)
    records = _candidate_records(payload)
    if not records:
        raise FileNotFoundError(f"No prest2 candidate records found in: {result_path}")

    candidate_record = _find_candidate_record(records, candidate)
    history = _load_history(candidate_record)
    out_path = plot_candidate_losses(candidate_record, history, out_dir)

    print(f"Saved plot to: {out_path}")
    rank_id = int(candidate_record.get("candidate_b", candidate_record.get("candidate", candidate_record.get("top_mech_winner_a", candidate_record.get("newrank_a", 0)))))
    rank_label = "candidate B" if int(candidate_record.get("candidate_b", candidate_record.get("candidate", 0))) > 0 else "top mech winner A"
    print(
        f"{rank_label}={rank_id} | "
        f"from {_mw_seed_rank_label(candidate_record)} | "
        f"trial={int(candidate_record.get('trial_id', 0))} | "
        f"epochs={len(history)} | "
        f"final_val={_finite_or_nan(candidate_record.get('final_val_loss')):.6e} @ e{int(candidate_record.get('final_val_epoch', -1))}"
    )
    return out_path


def main() -> None:
    result_path = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_RESULT_PATH
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    candidate = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_CANDIDATE
    run_one(result_path, out_dir, candidate)


if __name__ == "__main__":
    main()
