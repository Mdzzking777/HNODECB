from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FLOAT_TOKEN = r"[-+]?(?:nan|inf|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
TRAIN_EPOCH_RE = re.compile(rf"\b(?:QCS\s+)?epoch\s+(\d+)\s+train=({FLOAT_TOKEN})", re.IGNORECASE)
VAL_EPOCH_RE = re.compile(rf"\bQCS\s+val\s+epoch\s+(\d+)\s+val=({FLOAT_TOKEN})", re.IGNORECASE)
DONE_RE = re.compile(
    rf"done status=(\S+)\s+best_epoch=([0-9\-]+)\s+best_val=({FLOAT_TOKEN})\s+full_Fts=({FLOAT_TOKEN})%",
    re.IGNORECASE,
)


def _value_from_line(line: str, key: str) -> float:
    match = re.search(rf"(?:^|\s){re.escape(key)}=({FLOAT_TOKEN})%?", line, re.IGNORECASE)
    if match is None:
        return float("nan")
    return float(match.group(1))


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
QCS_ROOT = REPO_ROOT / "AFM04" / "KAN_full_test" / "quick_check_supervised"
LOG_DIR = QCS_ROOT / "logs"
RESULT_DIR = QCS_ROOT / "results"
OUT_DIR = LOG_DIR / "visualization"


def out_path(filename: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def finalize_and_save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {path}")
    return path


def latest_file(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under: {directory}")
    return paths[0]


def parse_training_log(log_path: Path | None = None) -> dict:
    path = log_path or latest_file(LOG_DIR, "qcs_supervised_driver_*.txt")
    rows_by_epoch: dict[int, dict[str, float]] = {}
    current_epoch: int | None = None
    done: dict[str, float | str] = {}

    def row_for(epoch: int) -> dict[str, float]:
        if epoch not in rows_by_epoch:
            rows_by_epoch[epoch] = {"epoch": float(epoch)}
        return rows_by_epoch[epoch]

    def fill_parts(row: dict[str, float], prefix: str, line: str) -> None:
        row[f"{prefix}_formal_total"] = _value_from_line(line, "formal")
        row[f"{prefix}_fts_norm_mse"] = _value_from_line(line, "FtsSup")
        row[f"{prefix}_state"] = _value_from_line(line, "state")
        row[f"{prefix}_x1_state"] = _value_from_line(line, "x1_state")
        row[f"{prefix}_x2_state"] = _value_from_line(line, "x2_state")
        row[f"{prefix}_x2dot"] = _value_from_line(line, "x2dot")
        row[f"{prefix}_x3_range"] = _value_from_line(line, "x3r")
        row[f"{prefix}_fts_range"] = _value_from_line(line, "ftsr")

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRAIN_EPOCH_RE.search(line)
        if match is not None:
            current_epoch = int(match.group(1))
            row = row_for(current_epoch)
            row["train_loss"] = float(match.group(2))
            # Legacy single-line QCS logs are still accepted for old partial runs.
            row["val_loss"] = _value_from_line(line, "val")
            row["train_fts_rel_rmse_pct"] = _value_from_line(line, "train_Fts")
            row["val_fts_rel_rmse_pct"] = _value_from_line(line, "val_Fts")
            row["lr"] = _value_from_line(line, "lr")
            row["grad_norm"] = _value_from_line(line, "grad")
            row["train_formal_total"] = _value_from_line(line, "train_formal")
            row["val_formal_total"] = _value_from_line(line, "val_formal")
            row["train_fts_norm_mse"] = _value_from_line(line, "train_FtsSup")
            row["val_fts_norm_mse"] = _value_from_line(line, "val_FtsSup")
            row["train_x1_state"] = _value_from_line(line, "train_x1State")
            row["val_x1_state"] = _value_from_line(line, "val_x1State")
            row["train_x3_range"] = _value_from_line(line, "train_x3Range")
            row["val_x3_range"] = _value_from_line(line, "val_x3Range")
            row["train_x1_rec"] = _value_from_line(line, "train_x1Rec")
            row["val_x1_rec"] = _value_from_line(line, "val_x1Rec")
            row["train_x3_rec"] = _value_from_line(line, "train_x3Rec")
            row["val_x3_rec"] = _value_from_line(line, "val_x3Rec")
            continue
        match = VAL_EPOCH_RE.search(line)
        if match is not None:
            current_epoch = int(match.group(1))
            row_for(current_epoch)["val_loss"] = float(match.group(2))
            continue
        if current_epoch is not None:
            row = row_for(current_epoch)
            if "grad_norm=" in line:
                row["grad_norm"] = _value_from_line(line, "grad_norm")
                row["lr"] = _value_from_line(line, "lr")
                continue
            if "  val parts:" in line:
                fill_parts(row, "val", line)
                continue
            if "  parts:" in line:
                fill_parts(row, "train", line)
                continue
            if "  val rec:" in line:
                row["val_x1_rec"] = _value_from_line(line, "x1")
                row["val_x3_rec"] = _value_from_line(line, "x3")
                continue
            if "  rec:" in line:
                row["train_x1_rec"] = _value_from_line(line, "x1")
                row["train_x3_rec"] = _value_from_line(line, "x3")
                continue
            if "  val nn:" in line:
                row["val_fts_rel_rmse_pct"] = _value_from_line(line, "err")
                continue
            if "  nn:" in line:
                row["train_fts_rel_rmse_pct"] = _value_from_line(line, "err")
                continue
        match = DONE_RE.search(line)
        if match is not None:
            done = {
                "status": match.group(1),
                "best_epoch": float(match.group(2)),
                "best_val": float(match.group(3)),
                "full_fts_rel_rmse_pct": float(match.group(4)),
            }
    rows = [rows_by_epoch[key] for key in sorted(rows_by_epoch)]
    return {"path": path, "rows": rows, "done": done}


def load_latest_history() -> tuple[Path, list[dict]]:
    path = latest_file(RESULT_DIR, "qcs_history_*.json")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected history payload type: {path}")
    return path, data


def load_latest_summary() -> tuple[Path, dict]:
    path = latest_file(RESULT_DIR, "qcs_summary_*.json")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected summary payload type: {path}")
    return path, data


def load_latest_snapshot() -> tuple[Path, dict[str, np.ndarray]]:
    path = latest_file(RESULT_DIR, "qcs_snapshot_*.npz")
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    return path, payload


def history_array(history: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in history], dtype=float)


def finite_minmax(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(arr, dtype=float).reshape(-1) for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = 1.0 if lo == 0.0 else abs(lo) * 0.05
        return lo - pad, hi + pad
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


__all__ = [
    "LOG_DIR",
    "OUT_DIR",
    "QCS_ROOT",
    "RESULT_DIR",
    "finite_minmax",
    "finalize_and_save",
    "find_repo_root",
    "history_array",
    "latest_file",
    "load_latest_history",
    "load_latest_snapshot",
    "load_latest_summary",
    "out_path",
    "parse_training_log",
]

