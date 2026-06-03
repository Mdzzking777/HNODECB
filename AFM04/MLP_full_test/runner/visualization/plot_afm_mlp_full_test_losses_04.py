from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.MLP_full_test.runner.visualization._common import finalize_and_save, out_path


EPOCH_RE = re.compile(r"MLP epoch (\d+) train=([0-9eE+\-.]+)")
VAL_EPOCH_RE = re.compile(r"MLP val epoch (\d+) val=([0-9eE+\-.]+|NaN|Inf|-Inf)")
SANITY_RE = re.compile(r"sanity loss=([0-9eE+\-.]+)")
ROLE_RE = re.compile(r"role=([^|]+)")
LABEL_RE = re.compile(r"label=([^|]+)")
DONE_RE = re.compile(r"done -- train=([0-9eE+\-.]+)\s+val=([0-9eE+\-.]+)")
BEST_RE = re.compile(r"best: train=([0-9eE+\-.]+)\s+val=([0-9eE+\-.]+)")


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
DEFAULT_LOG_DIR = REPO_ROOT / "AFM04" / "MLP_full_test" / "logs" / "window_per_shard"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "MLP_full_test" / "logs" / "visualization"
DEFAULT_OUT_FILE = "afm_param_MLP_full_test_04_loss_triptych.png"
DEFAULT_SAMPLE_EVERY = 25


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def sample_points(epochs: list[int], losses: list[float], sample_every: int) -> tuple[list[int], list[float]]:
    sample_epoch: list[int] = []
    sample_loss: list[float] = []
    for epoch, loss in zip(epochs, losses):
        if (epoch - 1) % sample_every == 0:
            sample_epoch.append(epoch)
            sample_loss.append(loss)
    if not sample_epoch and epochs:
        sample_epoch.append(epochs[-1])
        sample_loss.append(losses[-1])
    elif epochs and sample_epoch[-1] != epochs[-1]:
        sample_epoch.append(epochs[-1])
        sample_loss.append(losses[-1])
    return sample_epoch, sample_loss


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


def parse_epoch_losses(log_path: Path) -> dict:
    train_losses_by_epoch: dict[int, float] = {}
    val_losses_by_epoch: dict[int, float] = {}
    role = ""
    label = log_path.name
    final_val = float("nan")
    sanity_loss = float("nan")

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
            train_losses_by_epoch[int(match.group(1))] = parse_float(match.group(2))
            continue

        match = VAL_EPOCH_RE.search(line)
        if match is not None:
            val_epoch = int(match.group(1))
            val_loss = parse_float(match.group(2))
            if val_loss == val_loss and val_loss not in (float("inf"), float("-inf")):
                val_losses_by_epoch[val_epoch] = val_loss
            continue

        match = SANITY_RE.search(line)
        if match is not None:
            sanity_loss = parse_float(match.group(1))
            continue

        match = DONE_RE.search(line)
        if match is not None:
            final_val = parse_float(match.group(2))
            continue

        match = BEST_RE.search(line)
        if match is not None:
            final_val = parse_float(match.group(2))

    epochs = sorted(train_losses_by_epoch)
    train_losses = [train_losses_by_epoch[e] for e in epochs]
    val_epochs = sorted(val_losses_by_epoch)
    val_losses = [val_losses_by_epoch[e] for e in val_epochs]
    return {
        "path": str(log_path),
        "label": label,
        "role": role if role else label,
        "epochs": epochs,
        "train_losses": train_losses,
        "val_epochs": val_epochs,
        "val_losses": val_losses,
        "final_val": final_val,
        "sanity_loss": sanity_loss,
    }


def build_subplot(ax, rec: dict, *, sample_every: int) -> None:
    epochs = rec["epochs"]
    losses = rec["train_losses"]
    ax.set_title(shard_title(rec["role"], rec["label"]))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)

    if not epochs:
        ax.text(0.5, 0.5, "no epoch data", transform=ax.transAxes, ha="center", va="center", color="darkred")
        return

    sample_epoch, sample_loss = sample_points(epochs, losses, sample_every)
    sample_val_epoch, sample_val_loss = sample_points(rec["val_epochs"], rec["val_losses"], sample_every)

    y_values = [v for v in sample_loss if v > 0]
    y_values.extend(v for v in sample_val_loss if v > 0)
    if rec["final_val"] == rec["final_val"] and rec["final_val"] > 0:
        y_values.append(rec["final_val"])
    if rec["sanity_loss"] == rec["sanity_loss"] and rec["sanity_loss"] > 0:
        y_values.append(rec["sanity_loss"])

    y_min = min(y_values) / 1.8 if y_values else 1e-12
    y_max = max(y_values) * 1.8 if y_values else 1e-6

    ax.plot(sample_epoch, sample_loss, label="train", color="steelblue", linewidth=2, marker="o", markersize=3)
    if sample_val_epoch:
        ax.plot(
            sample_val_epoch,
            sample_val_loss,
            label="validation",
            color="firebrick",
            linewidth=2,
            linestyle="--",
            marker="D",
            markersize=3,
        )
    elif rec["final_val"] == rec["final_val"]:
        ax.scatter([epochs[-1]], [rec["final_val"]], label="final val", color="firebrick", marker="D", s=30)

    if rec["sanity_loss"] == rec["sanity_loss"] and rec["sanity_loss"] > 0:
        ax.axhline(rec["sanity_loss"], label="sanity", color="blue", linestyle="--", linewidth=2)

    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, max(epochs[-1], rec["val_epochs"][-1] if rec["val_epochs"] else 0) + 1)
    ax.legend(loc="best")


def run_one(log_dir: Path, out_dir: Path, sample_every: int) -> Path:
    log_paths = sorted(log_dir.glob("log2_04_step2a_MLP_full_test_local_w0.txt"))
    if not log_paths:
        raise FileNotFoundError(f"No MLP full-test shard logs found under: {log_dir}")

    recs = [parse_epoch_losses(path) for path in log_paths]
    ncols = max(1, len(recs))
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.2), sharey=False, squeeze=False)
    for ax, rec in zip(axes[0], recs):
        build_subplot(ax, rec, sample_every=sample_every)

    out_file = out_path(DEFAULT_OUT_FILE, out_dir)
    finalize_and_save(fig, out_file)

    print(f"Sampled every {sample_every} epochs.")
    for index, rec in enumerate(recs, start=1):
        if not rec["epochs"]:
            print(f"W{index}: no epoch data found")
            continue
        print(
            f"W{index} [{shard_title(rec['role'], rec['label'])}]: "
            f"epochs={rec['epochs'][0]}-{rec['epochs'][-1]} | "
            f"train_points={len(rec['epochs'])} | "
            f"val_points={len(rec['val_epochs'])} | "
            f"sanity={rec['sanity_loss']:.4e}" if rec["sanity_loss"] == rec["sanity_loss"] else
            f"W{index} [{shard_title(rec['role'], rec['label'])}]: "
            f"epochs={rec['epochs'][0]}-{rec['epochs'][-1]} | "
            f"train_points={len(rec['epochs'])} | "
            f"val_points={len(rec['val_epochs'])} | "
            f"sanity=missing"
        )
    return out_file


def main() -> None:
    log_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_LOG_DIR
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    sample_every = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_SAMPLE_EVERY
    run_one(log_dir, out_dir, sample_every)


if __name__ == "__main__":
    main()
