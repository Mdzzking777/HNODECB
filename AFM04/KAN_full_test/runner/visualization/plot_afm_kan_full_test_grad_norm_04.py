from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.runner.visualization._common import (
    finalize_and_save,
    load_replayed_handoff_history,
    out_path,
    plot_phase_series,
)
from AFM04.KAN_full_test.runner.visualization.plot_afm_kan_full_test_losses_04 import sample_points, shard_title


EPOCH_RE = re.compile(r"KAN epoch (\d+) train=")
LBFGS_EPOCH_RE = re.compile(r"KAN LBFGS step \d+ epoch (\d+) train=")
GRAD_RE = re.compile(r"\bgrad_norm=([0-9eE+\-.]+|NaN|Inf|-Inf)\b")
EPOCH0_GRAD_RE = re.compile(r"DIAG epoch=1 .* after backward .* grad_norm_raw=([0-9eE+\-.]+|NaN|Inf|-Inf)")
ROLE_RE = re.compile(r"role=([^|]+)")
LABEL_RE = re.compile(r"label=([^|]+)")


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
DEFAULT_LOG_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "window_per_shard"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "visualization"
DEFAULT_OUT_FILE = "afm_param_kan_full_test_04_grad_norm_grid.png"
DEFAULT_SAMPLE_EVERY = 1


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def parse_grad_norms(log_path: Path) -> dict:
    grad_norms_by_epoch: dict[int, float] = {}
    phases_by_epoch: dict[int, str] = {}
    role = ""
    label = log_path.name
    current_epoch: int | None = None
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
            continue

        match = LBFGS_EPOCH_RE.search(line)
        if match is not None:
            current_epoch = int(match.group(1))
            current_phase = "lbfgs"
            continue

        match = GRAD_RE.search(line)
        if match is not None and current_epoch is not None:
            grad_norms_by_epoch[current_epoch] = parse_float(match.group(1))
            phases_by_epoch[current_epoch] = current_phase
            current_epoch = None
            continue

        match = EPOCH0_GRAD_RE.search(line)
        if match is not None:
            grad_norms_by_epoch.setdefault(0, parse_float(match.group(1)))
            phases_by_epoch.setdefault(0, "warmstart")

    for row in load_replayed_handoff_history(log_path):
        epoch = int(row.get("epoch", 0))
        if epoch <= 0 or epoch in grad_norms_by_epoch:
            continue
        grad_norm = float(row.get("grad_norm", float("nan")))
        if is_finite(grad_norm):
            grad_norms_by_epoch[epoch] = grad_norm
            phases_by_epoch[epoch] = "adam"

    epochs = sorted(grad_norms_by_epoch)
    return {
        "path": str(log_path),
        "label": label,
        "role": role if role else label,
        "epochs": epochs,
        "grad_norms": [grad_norms_by_epoch[e] for e in epochs],
        "phases": [phases_by_epoch.get(e, "adam") for e in epochs],
    }


def build_subplot(ax, rec: dict, *, sample_every: int) -> None:
    epochs = rec["epochs"]
    grad_norms = rec["grad_norms"]
    ax.set_title(shard_title(rec["role"], rec["label"]))
    ax.set_xlabel("epoch")
    ax.set_ylabel("grad_norm")
    ax.grid(True, alpha=0.25)

    if not epochs:
        ax.text(0.5, 0.5, "no grad_norm data", transform=ax.transAxes, ha="center", va="center", color="darkred")
        return

    sample_epoch, sample_grad, sample_phase = sample_points(epochs, grad_norms, sample_every, rec.get("phases"))
    y_values = [v for v in sample_grad if v > 0]
    y_min = min(y_values) / 1.8 if y_values else 1.0e-12
    y_max = max(y_values) * 1.8 if y_values else 1.0

    plot_phase_series(
        ax,
        sample_epoch,
        sample_grad,
        sample_phase,
        label="grad_norm",
        color="purple",
        linewidth=2,
        marker="o",
        markersize=3,
    )

    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, epochs[-1] + 1)
    ax.legend(loc="best")


def run_one(log_dir: Path, out_dir: Path, sample_every: int) -> Path:
    log_paths = sorted(log_dir.glob("log2_04_step2a_kan_full_test_local_*.txt"))
    if not log_paths:
        raise FileNotFoundError(f"No KAN full-test shard logs found under: {log_dir}")

    recs = [parse_grad_norms(path) for path in log_paths]
    ncols = max(1, len(recs))
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.2), sharey=False, squeeze=False)
    for ax, rec in zip(axes[0], recs):
        build_subplot(ax, rec, sample_every=sample_every)

    out_file = out_path(DEFAULT_OUT_FILE, out_dir)
    finalize_and_save(fig, out_file)

    print(f"Sampled every {sample_every} epochs.")
    for rec in recs:
        if not rec["epochs"]:
            print(f"{shard_title(rec['role'], rec['label'])}: no grad_norm data found")
            continue
        print(
            f"{shard_title(rec['role'], rec['label'])}: "
            f"epochs={rec['epochs'][0]}-{rec['epochs'][-1]} | "
            f"grad_norm_points={len(rec['epochs'])}"
        )
    return out_file


def main() -> None:
    log_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_LOG_DIR
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    sample_every = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_SAMPLE_EVERY
    run_one(log_dir, out_dir, sample_every)


if __name__ == "__main__":
    main()
