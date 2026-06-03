from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.runner.visualization._common import (
    REPO_ROOT,
    finalize_and_save,
    history_phases,
    load_epoch0_warmstart_record,
    load_replayed_handoff_history,
    load_result_payloads,
    out_path,
    plot_phase_series,
    prepend_epoch0_history,
    sample_epoch_series,
    stage_title,
    window_title,
)


ROLE_RE = re.compile(r"role=([^|]+)")
LABEL_RE = re.compile(r"label=([^|]+)")
EPOCH_RE = re.compile(r"KAN epoch (\d+) train=([0-9eE+\-.]+)")
LBFGS_EPOCH_RE = re.compile(r"KAN LBFGS step \d+ epoch (\d+) train=([0-9eE+\-.]+)")
VAL_EPOCH_RE = re.compile(r"KAN val epoch (\d+) val=([0-9eE+\-.]+|NaN|Inf|-Inf)")
REC_RE = re.compile(r"rec:\s*x1=([0-9eE+\-.]+)%\s+x3=([0-9eE+\-.]+)%")
VAL_REC_RE = re.compile(r"val rec:\s*x1=([0-9eE+\-.]+)%\s+x3=([0-9eE+\-.]+)%")
NN_RE = re.compile(r"nn:\s*F_contact err=([0-9eE+\-.]+)%")
VAL_NN_RE = re.compile(r"val nn:\s*F_contact err=([0-9eE+\-.]+)%")

DEFAULT_LOG_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "window_per_shard"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "visualization"
DEFAULT_OUT_FILE = "afm_param_kan_full_test_04_recon_nn_grid.png"


def _robust_error_limit(values: list[float]) -> tuple[float, bool, int, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size < 4:
        return float(np.nanmax(arr)) if arr.size else 1.0, False, 0, float("nan")

    arr = np.maximum(arr, 0.0)
    full_max = float(np.nanmax(arr))
    q25, q50, q75 = np.nanpercentile(arr, [25, 50, 75])
    iqr = max(float(q75 - q25), 0.0)
    p90 = float(np.nanpercentile(arr, 90))
    tukey = float(q75 + 3.0 * iqr)

    if iqr <= 1.0e-12:
        robust_top = max(float(q50) * 1.25, p90, 1.0e-9)
    else:
        robust_top = max(min(p90, tukey), float(q75), float(q50), 1.0e-9)

    clip = full_max > max(robust_top * 1.8, robust_top + 1.0e-9)
    if not clip:
        return full_max, False, 0, full_max

    top = robust_top * 1.12
    clipped = int(np.sum(arr > top))
    return top, True, clipped, full_max


def _apply_robust_error_axis(
    ax,
    series: list[tuple[list[int], list[float], str]],
) -> None:
    values: list[float] = []
    for _, ys, _ in series:
        values.extend(float(y) for y in ys)

    top, clipped, clipped_count, full_max = _robust_error_limit(values)
    if not clipped:
        return

    top = max(float(top), 1.0e-9)
    ax.set_ylim(0.0, top)
    marker_y = top * 0.965

    marker_drawn = False
    for xs, ys, color in series:
        clipped_x = [x for x, y in zip(xs, ys, strict=False) if np.isfinite(y) and float(y) > top]
        if not clipped_x:
            continue
        ax.scatter(
            clipped_x,
            [marker_y] * len(clipped_x),
            marker="v",
            s=34,
            color=color,
            edgecolors="crimson",
            linewidths=0.8,
            zorder=6,
            label="clipped outlier" if not marker_drawn else None,
        )
        marker_drawn = True

    ax.text(
        0.02,
        0.96,
        f"y clipped at {top:.3g}%\nmax={full_max:.3g}% | outliers={clipped_count}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )


def shard_title(role: str, label: str) -> str:
    role_norm = role.strip().lower()
    if role_norm == "modified_w0":
        return "modified W0"
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
    train_x3: dict[int, float] = {}
    train_nn: dict[int, float] = {}
    val_x1: dict[int, float] = {}
    val_x3: dict[int, float] = {}
    val_nn: dict[int, float] = {}
    train_phase: dict[int, str] = {}
    val_phase: dict[int, str] = {}
    current_phase = "adam"

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
            current_phase = "adam"
            train_phase[current_epoch] = current_phase
            continue

        match = LBFGS_EPOCH_RE.search(line)
        if match is not None:
            current_epoch = int(match.group(1))
            current_phase = "lbfgs"
            train_phase[current_epoch] = current_phase
            continue

        match = VAL_EPOCH_RE.search(line)
        if match is not None:
            current_val_epoch = int(match.group(1))
            val_phase[current_val_epoch] = train_phase.get(current_val_epoch, current_phase)
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

    for row in load_replayed_handoff_history(log_path):
        epoch = int(row.get("epoch", 0))
        if epoch <= 0:
            continue
        if epoch not in val_x1:
            val_x1[epoch] = float(row.get("val_x1_rec", float("nan")))
        if epoch not in val_x3:
            val_x3[epoch] = float(row.get("val_x3_rec", float("nan")))
        if epoch not in val_nn:
            val_nn[epoch] = float(row.get("val_fts_rollout_rec", float("nan")))
        val_phase[epoch] = "adam"

    epoch0 = load_epoch0_warmstart_record(role)
    if epoch0 is not None:
        train_x1.setdefault(0, float(epoch0.get("train_x1_rec", float("nan"))))
        train_x3.setdefault(0, float(epoch0.get("train_x3_rec", float("nan"))))
        train_nn.setdefault(0, float(epoch0.get("train_fts_rollout_rec", float("nan"))))
        val_x1.setdefault(0, float(epoch0.get("val_x1_rec", float("nan"))))
        val_x3.setdefault(0, float(epoch0.get("val_x3_rec", float("nan"))))
        val_nn.setdefault(0, float(epoch0.get("val_fts_rollout_rec", float("nan"))))
        train_phase.setdefault(0, "warmstart")
        val_phase.setdefault(0, "warmstart")

    # Prefer validation metrics when they exist, otherwise fall back to train-side metrics.
    epochs = sorted(set(val_x1) | set(val_x3) | set(val_nn))
    use_train = not epochs
    if use_train:
        epochs = sorted(set(train_x1) | set(train_x3) | set(train_nn))

    x1 = [(train_x1 if use_train else val_x1).get(epoch, float("nan")) for epoch in epochs]
    x3 = [(train_x3 if use_train else val_x3).get(epoch, float("nan")) for epoch in epochs]
    nn = [(train_nn if use_train else val_nn).get(epoch, float("nan")) for epoch in epochs]
    phases = [(train_phase if use_train else val_phase).get(epoch, "adam") for epoch in epochs]
    return {
        "epochs": epochs,
        "x1": x1,
        "x3": x3,
        "nn": nn,
        "phases": phases,
        "title": shard_title(role if role else label, label),
        "metric_split": "train" if use_train else "val",
    }


def load_log_series(log_dir: Path = DEFAULT_LOG_DIR) -> list[dict]:
    log_paths = sorted(log_dir.glob("log2_04_step2a_kan_full_test_local_*.txt"))
    if not log_paths:
        raise FileNotFoundError(f"No KAN full-test shard logs found under: {log_dir}")
    return [parse_log_metrics(path) for path in log_paths]


def run_one(*, log_dir: Path | None = None, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    use_logs = log_dir is not None
    if use_logs:
        payloads = load_log_series(log_dir)
    else:
        try:
            payloads = load_result_payloads()
        except FileNotFoundError:
            use_logs = True
            payloads = load_log_series()

    ncols = max(1, len(payloads))
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 9), sharex=False, squeeze=False)
    for col, payload in enumerate(payloads):
        if use_logs:
            epochs = payload["epochs"]
            x1 = payload["x1"]
            x3 = payload["x3"]
            nn = payload["nn"]
            phases = payload.get("phases", ["adam"] * len(epochs))
            title = f"{payload['title']} ({payload['metric_split']})"
        else:
            hist = payload.get("history", [])
            hist = prepend_epoch0_history(hist, payload.get("window_meta", {}).get("role", ""))
            epochs = [int(row["epoch"]) for row in hist]
            x1 = [float(row.get("val_x1_rec", float("nan"))) for row in hist]
            x3 = [float(row.get("val_x3_rec", float("nan"))) for row in hist]
            nn = [float(row.get("val_fts_rollout_rec", float("nan"))) for row in hist]
            phases = history_phases(hist)
            title = f"{stage_title(payload)}: {window_title(payload)}"

        epochs_x1, x1_sampled, phases_x1 = sample_epoch_series(epochs, x1, phases)
        epochs_x3, x3_sampled, phases_x3 = sample_epoch_series(epochs, x3, phases)
        epochs_nn, nn_sampled, phases_nn = sample_epoch_series(epochs, nn, phases)

        ax1 = axes[0, col]
        plot_phase_series(ax1, epochs_x1, x1_sampled, phases_x1, color="seagreen", linewidth=2, label="x1_rec")
        plot_phase_series(ax1, epochs_x3, x3_sampled, phases_x3, color="purple", linewidth=2, label="x3_rec")
        ax1.set_title(title)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("reconstruction error (%)")
        ax1.grid(True, alpha=0.25)
        _apply_robust_error_axis(
            ax1,
            [
                (epochs_x1, x1_sampled, "seagreen"),
                (epochs_x3, x3_sampled, "purple"),
            ],
        )
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        plot_phase_series(ax2, epochs_nn, nn_sampled, phases_nn, color="black", linewidth=2, label="F_contact err")
        ax2.set_title(title)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("NN F_contact error (%)")
        ax2.grid(True, alpha=0.25)
        _apply_robust_error_axis(ax2, [(epochs_nn, nn_sampled, "black")])
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
