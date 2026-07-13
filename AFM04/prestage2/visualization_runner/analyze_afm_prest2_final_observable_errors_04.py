from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "AFM04" / "archive" / "prest2" / "10x10x1000_W0"
DEFAULT_CANDIDATES_PATH = DEFAULT_ARCHIVE_ROOT / "results" / "afm_prest2_04_candidates_b.pkl"
DEFAULT_OUT_DIR = DEFAULT_ARCHIVE_ROOT / "logs"


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected payload type for {path}: {type(payload)!r}")
    return payload


def _candidate_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("candidate_b_records", [])
    if not isinstance(records, list):
        return []
    return [rec for rec in records if isinstance(rec, dict)]


def _candidate_id(record: dict[str, Any]) -> int:
    for key in ("candidate_b", "candidate"):
        try:
            value = int(record.get(key, 0))
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _candidate_viz_path(archive_root: Path, record: dict[str, Any]) -> Path:
    try:
        mech_winner = int(record.get("source_mech_winner", 0))
    except Exception:
        mech_winner = 0
    if mech_winner <= 0:
        raise ValueError(f"Candidate B {_candidate_id(record)} does not have source_mech_winner.")
    return (
        archive_root
        / "candidate_runs"
        / "stage1_origin"
        / f"mech_winner_{mech_winner:03d}"
        / "results"
        / "stage2light_result_p1.viz.pkl"
    )


def _relative_rmse_pct(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    if pred_arr.shape != truth_arr.shape:
        raise ValueError(f"Shape mismatch: pred {pred_arr.shape}, truth {truth_arr.shape}")
    denom = float(np.sqrt(np.mean(np.square(truth_arr))))
    numer = float(np.sqrt(np.mean(np.square(pred_arr - truth_arr))))
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return 100.0 * numer / denom


def _snapshot_errors(payload: dict[str, Any], split: str) -> dict[str, float]:
    final_snapshot = payload.get("final_snapshot")
    if not isinstance(final_snapshot, dict):
        raise RuntimeError("Missing final_snapshot in viz payload.")
    snap = final_snapshot.get(split)
    if not isinstance(snap, dict):
        raise RuntimeError(f"Missing final_snapshot[{split!r}] in viz payload.")

    ode_true = np.asarray(snap["ode_true"], dtype=float)
    traj_pred = np.asarray(snap["traj_pred"], dtype=float)
    x2dot_true = np.asarray(snap["x2dot_true"], dtype=float)
    x2dot_pred = np.asarray(snap["x2dot_pred"], dtype=float)

    if ode_true.shape[0] < 2 or traj_pred.shape[0] < 2:
        raise ValueError(f"Expected state arrays with at least x1/x2 rows, got {ode_true.shape} and {traj_pred.shape}")

    return {
        "x1_rel_rmse_pct": _relative_rmse_pct(traj_pred[0, :], ode_true[0, :]),
        "x2_rel_rmse_pct": _relative_rmse_pct(traj_pred[1, :], ode_true[1, :]),
        "x2dot_rel_rmse_pct": _relative_rmse_pct(x2dot_pred, x2dot_true),
        "n_points": float(ode_true.shape[1]),
    }


def _fmt(value: Any, digits: int = 10) -> str:
    try:
        out = float(value)
    except Exception:
        return ""
    if not math.isfinite(out):
        return "nan"
    return f"{out:.{digits}f}"


def build_rows(archive_root: Path, candidates_path: Path) -> list[dict[str, Any]]:
    payload = _load_pickle(candidates_path)
    records = _candidate_records(payload)
    if not records:
        raise RuntimeError(f"No candidate_b_records found in {candidates_path}")

    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=_candidate_id):
        candidate_b = _candidate_id(record)
        viz_path = _candidate_viz_path(archive_root, record)
        if not viz_path.is_file():
            raise FileNotFoundError(f"Candidate B {candidate_b:02d} viz payload not found: {viz_path}")
        viz_payload = _load_pickle(viz_path)
        for split in ("full", "train", "val"):
            errs = _snapshot_errors(viz_payload, split)
            rows.append(
                {
                    "candidate_b": candidate_b,
                    "source_stage1_rank": int(record.get("source_stage1_rank", 0)),
                    "source_mech_winner": int(record.get("source_mech_winner", 0)),
                    "trial_id": int(record.get("trial_id", record.get("source_stage1_trial_id", 0))),
                    "epochs_completed": int(record.get("epochs_completed", 0)),
                    "split": split,
                    "n_points": int(errs["n_points"]),
                    "x1_rel_rmse_pct": errs["x1_rel_rmse_pct"],
                    "x2_rel_rmse_pct": errs["x2_rel_rmse_pct"],
                    "x2dot_rel_rmse_pct": errs["x2dot_rel_rmse_pct"],
                    "final_train_loss": float(record.get("final_train_loss", float("nan"))),
                    "final_val_loss": float(record.get("final_val_loss", float("nan"))),
                    "viz_path": str(viz_path),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_b",
        "source_stage1_rank",
        "source_mech_winner",
        "trial_id",
        "epochs_completed",
        "split",
        "n_points",
        "x1_rel_rmse_pct",
        "x2_rel_rmse_pct",
        "x2dot_rel_rmse_pct",
        "final_train_loss",
        "final_val_loss",
        "viz_path",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_txt(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("AFM04 prest2 final observable errors from saved viz payloads\n")
        f.write("Formula: relative RMSE (%) = 100 * sqrt(mean((pred - true)^2)) / sqrt(mean(true^2))\n")
        f.write("Source: final_snapshot split arrays in stage2light_result_p1.viz.pkl\n")
        f.write("Observables reported here: x1, x2, x2dot. No x3_true-based metric is used.\n\n")

        for row in rows:
            f.write(
                "B{candidate_b:02d} | rank {source_stage1_rank:>4d} | mech {source_mech_winner:03d} "
                "| split={split:<5s} | n={n_points:>3d} | "
                "x1={x1} % | x2={x2} % | x2dot={x2dot} %\n".format(
                    candidate_b=int(row["candidate_b"]),
                    source_stage1_rank=int(row["source_stage1_rank"]),
                    source_mech_winner=int(row["source_mech_winner"]),
                    split=str(row["split"]),
                    n_points=int(row["n_points"]),
                    x1=_fmt(row["x1_rel_rmse_pct"]),
                    x2=_fmt(row["x2_rel_rmse_pct"]),
                    x2dot=_fmt(row["x2dot_rel_rmse_pct"]),
                )
            )

        f.write("\nSplit summaries:\n")
        for split in ("full", "train", "val"):
            split_rows = [row for row in rows if row["split"] == split]
            if not split_rows:
                continue
            f.write(f"[{split}]\n")
            for key in ("x1_rel_rmse_pct", "x2_rel_rmse_pct", "x2dot_rel_rmse_pct"):
                values = np.asarray([float(row[key]) for row in split_rows], dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    f.write(f"  {key}: no finite values\n")
                    continue
                f.write(
                    f"  {key}: min={finite.min():.10f}% | median={np.median(finite):.10f}% "
                    f"| max={finite.max():.10f}%\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute high-precision final x1/x2/x2dot relative RMSE percentages from prest2 viz payloads."
    )
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", type=str, default="afm_prest2_final_observable_errors_from_viz")
    args = parser.parse_args()

    rows = build_rows(args.archive_root.resolve(), args.candidates_path.resolve())
    csv_path = args.out_dir / f"{args.stem}.csv"
    txt_path = args.out_dir / f"{args.stem}.txt"
    write_csv(rows, csv_path)
    write_txt(rows, txt_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")
    print(f"Rows: {len(rows)} ({len(rows) // 3} candidates x full/train/val)")
    for row in rows:
        if row["split"] == "full":
            print(
                "B{candidate_b:02d} rank={source_stage1_rank} full "
                "x1={x1:.10f}% x2={x2:.10f}% x2dot={x2dot:.10f}%".format(
                    candidate_b=int(row["candidate_b"]),
                    source_stage1_rank=int(row["source_stage1_rank"]),
                    x1=float(row["x1_rel_rmse_pct"]),
                    x2=float(row["x2_rel_rmse_pct"]),
                    x2dot=float(row["x2dot_rel_rmse_pct"]),
                )
            )


if __name__ == "__main__":
    main()
