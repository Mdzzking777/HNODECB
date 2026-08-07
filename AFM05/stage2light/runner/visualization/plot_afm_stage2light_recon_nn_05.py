from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM05.stage2light.runner.visualization._common import (
    LOG_DIR,
    OUT_DIR,
    REPO_ROOT,
    discover_log_paths,
    finalize_and_save,
    load_available_payloads,
    out_path,
    stage_title,
    window_title,
)


ROLE_RE = re.compile(r"role=([^|]+)")
LABEL_RE = re.compile(r"label=([^|]+)")
EPOCH_RE = re.compile(r"KAN (?:LBFGS step \d+ epoch|epoch) (\d+) train=([0-9eE+\-.]+)")
VAL_EPOCH_RE = re.compile(r"KAN val epoch (\d+) val=([0-9eE+\-.]+|NaN|Inf|-Inf)")
REC_RE = re.compile(
    r"rec:\s*x1=([0-9eE+\-.]+)%\s+x2=([0-9eE+\-.]+)%\s+x2dot=([0-9eE+\-.]+)%"
)
VAL_REC_RE = re.compile(
    r"val rec:\s*x1=([0-9eE+\-.]+)%\s+x2=([0-9eE+\-.]+)%\s+x2dot=([0-9eE+\-.]+)%"
)

DEFAULT_LOG_DIR = LOG_DIR
DEFAULT_OUT_DIR = OUT_DIR
DEFAULT_OUT_FILE = "afm_param_stage2light_05_recon_nn_grid.png"


def _role_sort_key(role: str) -> tuple[int, str]:
    role_norm = role.strip().lower()
    if role_norm == "first_contact":
        return (0, role_norm)
    if role_norm == "middle":
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
        return "W0: first-contact window"
    if role_norm == "middle":
        return "W1: middle window"
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
    train_x2: dict[int, float] = {}
    train_x2dot: dict[int, float] = {}
    val_x1: dict[int, float] = {}
    val_x2: dict[int, float] = {}
    val_x2dot: dict[int, float] = {}

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
            val_x2[current_val_epoch] = float(match.group(2))
            val_x2dot[current_val_epoch] = float(match.group(3))
            continue

        match = REC_RE.search(line)
        if match is not None and current_epoch is not None:
            train_x1[current_epoch] = float(match.group(1))
            train_x2[current_epoch] = float(match.group(2))
            train_x2dot[current_epoch] = float(match.group(3))
            continue

    train_epochs = sorted(set(train_x1) | set(train_x2) | set(train_x2dot))
    val_epochs = sorted(set(val_x1) | set(val_x2) | set(val_x2dot))
    return {
        "source": "log",
        "role": role,
        "label": label,
        "train_epochs": train_epochs,
        "train_x1": [train_x1.get(epoch, float("nan")) for epoch in train_epochs],
        "train_x2": [train_x2.get(epoch, float("nan")) for epoch in train_epochs],
        "train_x2dot": [train_x2dot.get(epoch, float("nan")) for epoch in train_epochs],
        "val_epochs": val_epochs,
        "val_x1": [val_x1.get(epoch, float("nan")) for epoch in val_epochs],
        "val_x2": [val_x2.get(epoch, float("nan")) for epoch in val_epochs],
        "val_x2dot": [val_x2dot.get(epoch, float("nan")) for epoch in val_epochs],
        "title": shard_title(role if role else label, label),
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
        result_payloads = load_available_payloads()
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

    fig, axes = plt.subplots(len(payloads), 4, figsize=(24, 4.8 * len(payloads)), sharex=False, squeeze=False)
    for row, payload in enumerate(payloads):
        if payload.get("source") == "log":
            title = payload["title"]
            train_epochs = payload["train_epochs"]
            train_x1 = payload["train_x1"]
            train_x2 = payload["train_x2"]
            train_x2dot = payload["train_x2dot"]
            val_epochs = payload["val_epochs"]
            val_x1 = payload["val_x1"]
            val_x2 = payload["val_x2"]
            val_x2dot = payload["val_x2dot"]
        else:
            result_payload = payload["payload"]
            hist = result_payload.get("history", [])
            train_epochs = [int(item["epoch"]) for item in hist]
            val_epochs = [int(item["epoch"]) for item in hist]
            train_x1 = [float(item.get("train_x1_rec", float("nan"))) for item in hist]
            train_x2 = [float(item.get("train_x2_rec", float("nan"))) for item in hist]
            train_x2dot = [float(item.get("train_x2dot_rec", float("nan"))) for item in hist]
            val_x1 = [float(item.get("val_x1_rec", float("nan"))) for item in hist]
            val_x2 = [float(item.get("val_x2_rec", float("nan"))) for item in hist]
            val_x2dot = [float(item.get("val_x2dot_rec", float("nan"))) for item in hist]
            title = f"{stage_title(result_payload)}: {window_title(result_payload)}"

        ax_train_rec = axes[row, 0]
        ax_train_rec.plot(train_epochs, train_x1, color="seagreen", linewidth=2, label="x1_rec")
        ax_train_rec.plot(train_epochs, train_x2, color="purple", linewidth=2, label="x2_rec")
        ax_train_rec.set_title(f"{title} | train rec")
        ax_train_rec.set_xlabel("epoch")
        ax_train_rec.set_ylabel("reconstruction error (%)")
        ax_train_rec.grid(True, alpha=0.25)
        ax_train_rec.legend(loc="best")

        ax_val_rec = axes[row, 1]
        ax_val_rec.plot(val_epochs, val_x1, color="seagreen", linewidth=2, label="x1_rec")
        ax_val_rec.plot(val_epochs, val_x2, color="purple", linewidth=2, label="x2_rec")
        ax_val_rec.set_title(f"{title} | val rec")
        ax_val_rec.set_xlabel("epoch")
        ax_val_rec.set_ylabel("reconstruction error (%)")
        ax_val_rec.grid(True, alpha=0.25)
        ax_val_rec.legend(loc="best")

        ax_train_x2dot = axes[row, 2]
        ax_train_x2dot.plot(train_epochs, train_x2dot, color="black", linewidth=2, label="x2dot_rec")
        ax_train_x2dot.set_title(f"{title} | train x2dot")
        ax_train_x2dot.set_xlabel("epoch")
        ax_train_x2dot.set_ylabel("x2dot reconstruction error (%)")
        ax_train_x2dot.grid(True, alpha=0.25)
        ax_train_x2dot.legend(loc="best")

        ax_val_x2dot = axes[row, 3]
        ax_val_x2dot.plot(val_epochs, val_x2dot, color="black", linewidth=2, label="x2dot_rec")
        ax_val_x2dot.set_title(f"{title} | val x2dot")
        ax_val_x2dot.set_xlabel("epoch")
        ax_val_x2dot.set_ylabel("x2dot reconstruction error (%)")
        ax_val_x2dot.grid(True, alpha=0.25)
        ax_val_x2dot.legend(loc="best")

    out_file = out_path(DEFAULT_OUT_FILE, out_dir)
    finalize_and_save(fig, out_file)
    return out_file


def main() -> None:
    log_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else None
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    run_one(log_dir=log_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
