from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
MECH_RE = re.compile(
    r"mech:\s*ks=([0-9eE+\-.]+)\s+cs=([0-9eE+\-.]+)"
)

DEFAULT_LOG_DIR = LOG_DIR
DEFAULT_OUT_DIR = OUT_DIR


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


def parse_log_mech(log_path: Path) -> dict:
    role = ""
    label = log_path.name
    current_epoch: int | None = None
    ks_by_epoch: dict[int, float] = {}
    cs_by_epoch: dict[int, float] = {}

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

        match = MECH_RE.search(line)
        if match is not None and current_epoch is not None:
            ks_by_epoch[current_epoch] = float(match.group(1))
            cs_by_epoch[current_epoch] = float(match.group(2))

    epochs = sorted(set(ks_by_epoch) | set(cs_by_epoch))
    return {
        "source": "log",
        "role": role,
        "label": label,
        "epochs": epochs,
        "ks": [ks_by_epoch.get(epoch, float("nan")) for epoch in epochs],
        "cs": [cs_by_epoch.get(epoch, float("nan")) for epoch in epochs],
        "title": shard_title(role if role else label, label),
    }


def load_log_series(log_dir: Path = DEFAULT_LOG_DIR) -> list[dict]:
    log_paths = discover_log_paths(log_dir)
    if not log_paths:
        raise FileNotFoundError(f"No stage2light shard logs found under: {log_dir}")
    return [parse_log_mech(path) for path in log_paths]


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

    fig, axes = plt.subplots(2, len(payloads), figsize=(6 * len(payloads), 9), sharex=False, squeeze=False)
    for col, payload in enumerate(payloads):
        if payload.get("source") == "log":
            epochs = payload["epochs"]
            ks_series = np.asarray(payload["ks"], dtype=float)
            cs_series = np.asarray(payload["cs"], dtype=float)
            title = payload["title"]
        else:
            result_payload = payload["payload"]
            hist = result_payload.get("history", [])
            epochs = [int(row["epoch"]) for row in hist]
            ks_series = np.asarray([float(row.get("ks_hat", np.nan)) for row in hist], dtype=float)
            cs_series = np.asarray([float(row.get("cs_hat", np.nan)) for row in hist], dtype=float)
            title = f"{stage_title(result_payload)}: {window_title(result_payload)}"

        ax1 = axes[0, col]
        ax1.plot(epochs, ks_series, color="black", linewidth=2, label="ks_hat")
        ax1.set_title(title)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("ks")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        ax2.plot(epochs, cs_series, color="black", linewidth=2, label="cs_hat")
        ax2.set_title(title)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("cs")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

    return finalize_and_save(fig, out_path("afm_param_stage2light_05_mech_grid.png", out_dir))


def main() -> None:
    log_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else None
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    run_one(log_dir=log_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
