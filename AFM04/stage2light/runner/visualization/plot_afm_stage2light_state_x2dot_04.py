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
PARTS_RE = re.compile(
    r"parts:\s*state=([0-9eE+\-.]+)\s+x1_state=([0-9eE+\-.]+)\s+x2_state=([0-9eE+\-.]+)\s+x2dot=([0-9eE+\-.]+)"
)
VAL_PARTS_RE = re.compile(
    r"val parts:\s*state=([0-9eE+\-.]+)\s+x1_state=([0-9eE+\-.]+)\s+x2_state=([0-9eE+\-.]+)\s+x2dot=([0-9eE+\-.]+)"
)

DEFAULT_LOG_DIR = REPO_ROOT / "AFM04" / "stage2light" / "logs" / "window_per_shard"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "stage2light" / "logs" / "visualization"


def shard_title(role: str, label: str) -> str:
    role_norm = role.strip().lower()
    if role_norm == "first_contact":
        return "W1: window1: right after first contact"
    if role_norm == "max_x1_pp_change":
        return "W2: window2: the most drastic region"
    if role_norm == "tail_stable":
        return "W3: window3: stable region at the end"
    return role or label


def parse_log_series(log_path: Path) -> dict:
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

        match = VAL_PARTS_RE.search(line)
        if match is not None and current_val_epoch is not None:
            val_x1[current_val_epoch] = float(match.group(2))
            val_x2[current_val_epoch] = float(match.group(3))
            val_x2dot[current_val_epoch] = float(match.group(4))
            continue

        match = PARTS_RE.search(line)
        if match is not None and current_epoch is not None:
            train_x1[current_epoch] = float(match.group(2))
            train_x2[current_epoch] = float(match.group(3))
            train_x2dot[current_epoch] = float(match.group(4))

    epochs = sorted(set(val_x1) | set(val_x2) | set(val_x2dot))
    use_train = not epochs
    if use_train:
        epochs = sorted(set(train_x1) | set(train_x2) | set(train_x2dot))

    source_x1 = train_x1 if use_train else val_x1
    source_x2 = train_x2 if use_train else val_x2
    source_x2dot = train_x2dot if use_train else val_x2dot
    return {
        "epochs": epochs,
        "x1_state": [source_x1.get(epoch, float("nan")) for epoch in epochs],
        "x2_state": [source_x2.get(epoch, float("nan")) for epoch in epochs],
        "x2dot": [source_x2dot.get(epoch, float("nan")) for epoch in epochs],
        "title": shard_title(role if role else label, label),
        "metric_split": "train" if use_train else "val",
    }


def load_log_payloads(log_dir: Path = DEFAULT_LOG_DIR) -> list[dict]:
    log_paths = discover_log_paths(log_dir)
    if not log_paths:
        raise FileNotFoundError(f"No stage2light shard logs found under: {log_dir}")
    return [parse_log_series(path) for path in log_paths]


def run_one(*, log_dir: Path | None = None, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    use_logs = log_dir is not None
    if use_logs:
        payloads = load_log_payloads(log_dir)
    else:
        try:
            payloads = load_result_payloads()
        except FileNotFoundError:
            use_logs = True
            payloads = load_log_payloads()

    fig, axes = plt.subplots(3, len(payloads), figsize=(6 * len(payloads), 13), sharex=False, squeeze=False)
    for col, payload in enumerate(payloads):
        if use_logs:
            epochs = payload["epochs"]
            x1_state = payload["x1_state"]
            x2_state = payload["x2_state"]
            x2dot = payload["x2dot"]
            title = f"{payload['title']} ({payload['metric_split']})"
        else:
            hist = payload.get("history", [])
            epochs = [int(row["epoch"]) for row in hist]
            x1_state = [float(row.get("val_x1_state", float("nan"))) for row in hist]
            x2_state = [float(row.get("val_x2_state", float("nan"))) for row in hist]
            x2dot = [float(row.get("val_x2dot", float("nan"))) for row in hist]
            title = f"{stage_title(payload)}: {window_title(payload)}"

        ax1 = axes[0, col]
        ax1.plot(epochs, x1_state, color="seagreen", linewidth=2, label="x1_state")
        ax1.set_title(title)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("x1 state loss")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        ax2.plot(epochs, x2_state, color="royalblue", linewidth=2, label="x2_state")
        ax2.set_title(title)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("x2 state loss")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

        ax3 = axes[2, col]
        ax3.plot(epochs, x2dot, color="darkorange", linewidth=2, label="x2dot")
        ax3.set_title(title)
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("x2dot loss")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="best")

    return finalize_and_save(fig, out_path("afm_param_stage2light_04_x1x2state_x2dot_grid.png", out_dir))


def main() -> None:
    log_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else None
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    run_one(log_dir=log_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
