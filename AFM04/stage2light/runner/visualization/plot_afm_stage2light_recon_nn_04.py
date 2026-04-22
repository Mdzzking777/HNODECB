from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.stage2light.runner.visualization._common import (
    REPO_ROOT,
    discover_log_paths,
    finalize_and_save,
    load_result_payloads,
    out_path,
    stage_title,
    window_title,
)


ROLE_RE = re.compile(r"role=([^|]+)")
LABEL_RE = re.compile(r"label=([^|]+)")
EPOCH_RE = re.compile(r"KAN epoch (\d+) train=([0-9eE+\-.]+)")
VAL_EPOCH_RE = re.compile(r"KAN val epoch (\d+) val=([0-9eE+\-.]+|NaN|Inf|-Inf)")
REC_RE = re.compile(r"rec:\s*x1=([0-9eE+\-.]+)%\s+x3=([0-9eE+\-.]+)%")
VAL_REC_RE = re.compile(r"val rec:\s*x1=([0-9eE+\-.]+)%\s+x3=([0-9eE+\-.]+)%")
NN_RE = re.compile(r"nn:\s*F_contact err=([0-9eE+\-.]+)%")
VAL_NN_RE = re.compile(r"val nn:\s*F_contact err=([0-9eE+\-.]+)%")

DEFAULT_LOG_DIR = REPO_ROOT / "AFM04" / "stage2light" / "logs" / "window_per_shard"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "stage2light" / "logs" / "visualization"
DEFAULT_OUT_FILE = "afm_param_stage2light_04_recon_nn_grid.png"


def _role_sort_key(role: str) -> tuple[int, str]:
    role_norm = role.strip().lower()
    if role_norm == "first_contact":
        return (1, role_norm)
    if role_norm == "max_x1_pp_change":
        return (2, role_norm)
    if role_norm == "tail_stable":
        return (3, role_norm)
    return (99, role_norm)


def _series_key(role: str, label: str) -> str:
    role_s = role.strip()
    label_s = label.strip()
    return role_s if role_s else label_s


def shard_title(role: str, label: str) -> str:
    role_norm = role.strip().lower()
    if role_norm == "first_contact":
        return "W1: window1: right after first contact"
    if role_norm == "max_x1_pp_change":
        return "W2: window2: the most drastic region"
    if role_norm == "tail_stable":
        return "W3: window3: stable region at the end"
    return role or label


def parse_log_metrics(log_path: Path) -> dict:
    role = ""
    label = log_path.name
    current_epoch: int | None = None
    current_val_epoch: int | None = None
    train_x1: dict[int, float] = {}
    train_x3: dict[int, float] = {}
    train_nn: dict[int, float] = {}
    val_x1: dict[int, float] = {}
    val_x3: dict[int, float] = {}
    val_nn: dict[int, float] = {}

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not role:
            match = ROLE_RE.search(line)
            if match is not None:
                role = match.group(1).strip()
        if label == log_path.name:
            match = LABEL_RE.search(line)
            if match is not None:
                label = match.group(1).strip()

        match = EPOCH_RE.search(line)
        if match is not None:
            current_epoch = int(match.group(1))
            continue

        match = VAL_EPOCH_RE.search(line)
        if match is not None:
            current_val_epoch = int(match.group(1))
            continue

        match = VAL_REC_RE.search(line)
        if match is not None and current_val_epoch is not None:
            val_x1[current_val_epoch] = float(match.group(1))
            val_x3[current_val_epoch] = float(match.group(2))
            continue

        match = VAL_NN_RE.search(line)
        if match is not None and current_val_epoch is not None:
            val_nn[current_val_epoch] = float(match.group(1))
            continue

        match = REC_RE.search(line)
        if match is not None and current_epoch is not None:
            train_x1[current_epoch] = float(match.group(1))
            train_x3[current_epoch] = float(match.group(2))
            continue

        match = NN_RE.search(line)
        if match is not None and current_epoch is not None:
            train_nn[current_epoch] = float(match.group(1))

    # Prefer validation metrics when they exist, otherwise fall back to train-side metrics.
    epochs = sorted(set(val_x1) | set(val_x3) | set(val_nn))
    use_train = not epochs
    if use_train:
        epochs = sorted(set(train_x1) | set(train_x3) | set(train_nn))

    x1 = [(train_x1 if use_train else val_x1).get(epoch, float("nan")) for epoch in epochs]
    x3 = [(train_x3 if use_train else val_x3).get(epoch, float("nan")) for epoch in epochs]
    nn = [(train_nn if use_train else val_nn).get(epoch, float("nan")) for epoch in epochs]
    return {
        "source": "log",
        "role": role,
        "label": label,
        "epochs": epochs,
        "x1": x1,
        "x3": x3,
        "nn": nn,
        "title": shard_title(role if role else label, label),
        "metric_split": "train" if use_train else "val",
    }


def load_log_series(log_dir: Path = DEFAULT_LOG_DIR) -> list[dict]:
    log_paths = discover_log_paths(log_dir)
    if not log_paths:
        raise FileNotFoundError(f"No stage2light shard logs found under: {log_dir}")
    return [parse_log_metrics(path) for path in log_paths]


def load_merged_series(log_dir: Path = DEFAULT_LOG_DIR) -> list[dict]:
    merged: dict[str, dict] = {}

    try:
        log_payloads = load_log_series(log_dir)
    except FileNotFoundError:
        log_payloads = []
    for payload in log_payloads:
        merged[_series_key(str(payload.get("role", "")), str(payload.get("label", "")))] = payload

    try:
        result_payloads = load_result_payloads()
    except FileNotFoundError:
        result_payloads = []
    for payload in result_payloads:
        meta = payload.get("window_meta", {})
        role = str(meta.get("role", ""))
        label = str(meta.get("label", ""))
        merged[_series_key(role, label)] = {
            "source": "result",
            "payload": payload,
            "role": role,
            "label": label,
        }

    if not merged:
        raise FileNotFoundError("No stage2light results or logs found.")

    return sorted(merged.values(), key=lambda item: _role_sort_key(str(item.get("role", ""))))


def run_one(*, log_dir: Path | None = None, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    payloads = load_merged_series(log_dir if log_dir is not None else DEFAULT_LOG_DIR)

    fig, axes = plt.subplots(2, len(payloads), figsize=(6 * len(payloads), 9), sharex=False, squeeze=False)
    for col, payload in enumerate(payloads):
        if payload.get("source") == "log":
            epochs = payload["epochs"]
            x1 = payload["x1"]
            x3 = payload["x3"]
            nn = payload["nn"]
            title = f"{payload['title']} ({payload['metric_split']})"
        else:
            result_payload = payload["payload"]
            hist = result_payload.get("history", [])
            epochs = [int(row["epoch"]) for row in hist]
            x1 = [float(row.get("val_x1_rec", float("nan"))) for row in hist]
            x3 = [float(row.get("val_x3_rec", float("nan"))) for row in hist]
            nn = [float(row.get("val_fts_teacher_rec", float("nan"))) for row in hist]
            title = f"{stage_title(result_payload)}: {window_title(result_payload)}"

        ax1 = axes[0, col]
        ax1.plot(epochs, x1, color="seagreen", linewidth=2, label="x1_rec")
        ax1.plot(epochs, x3, color="purple", linewidth=2, label="x3_rec")
        ax1.set_title(title)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("reconstruction error (%)")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        ax2.plot(epochs, nn, color="black", linewidth=2, label="F_contact err")
        ax2.set_title(title)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("NN F_contact error (%)")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

    out_file = out_path(DEFAULT_OUT_FILE, out_dir)
    finalize_and_save(fig, out_file)
    return out_file


def main() -> None:
    log_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else None
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    run_one(log_dir=log_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
