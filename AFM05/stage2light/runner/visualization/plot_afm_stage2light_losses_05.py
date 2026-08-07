from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM05.stage2light.runner.visualization._common import (
    LOG_DIR,
    OUT_DIR,
    discover_log_paths,
    out_path,
    plot_phase_series,
)


EPOCH_RE = re.compile(r"KAN epoch (\d+) train=([0-9eE+\-.]+)")
LBFGS_EPOCH_RE = re.compile(r"KAN LBFGS step \d+ epoch (\d+) train=([0-9eE+\-.]+)")
VAL_EPOCH_RE = re.compile(r"KAN (?:final )?val epoch (\d+) val=([0-9eE+\-.]+|NaN|Inf|-Inf)")
ROLE_RE = re.compile(r"role=([^|]+)")
LABEL_RE = re.compile(r"label=([^|]+)")
DONE_RE = re.compile(r"done -- train=([0-9eE+\-.]+)\s+val=([0-9eE+\-.]+)")
BEST_RE = re.compile(r"best: train=([0-9eE+\-.]+)\s+val=([0-9eE+\-.]+)")


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM05" / "stage2light").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
DEFAULT_LOG_DIR = LOG_DIR
DEFAULT_OUT_DIR = OUT_DIR
DEFAULT_OUT_FILE = "afm_param_stage2light_05_loss_triptych.png"
DEFAULT_SAMPLE_EVERY = 1


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def sample_points(
    epochs: list[int],
    losses: list[float],
    sample_every: int,
    phases: list[str] | None = None,
) -> tuple[list[int], list[float], list[str]]:
    sample_epoch: list[int] = []
    sample_loss: list[float] = []
    sample_phase: list[str] = []
    if phases is None or len(phases) != len(epochs):
        phases = ["adam"] * len(epochs)
    for idx, (epoch, loss, phase) in enumerate(zip(epochs, losses, phases)):
        phase_changed = idx > 0 and phase != phases[idx - 1]
        phase_will_change = idx + 1 < len(phases) and phase != phases[idx + 1]
        if idx == 0 or (epoch - 1) % sample_every == 0 or phase_changed or phase_will_change:
            sample_epoch.append(epoch)
            sample_loss.append(loss)
            sample_phase.append(phase)
    if not sample_epoch and epochs:
        sample_epoch.append(epochs[-1])
        sample_loss.append(losses[-1])
        sample_phase.append(phases[-1])
    elif epochs and sample_epoch[-1] != epochs[-1]:
        sample_epoch.append(epochs[-1])
        sample_loss.append(losses[-1])
        sample_phase.append(phases[-1])
    return sample_epoch, sample_loss, sample_phase


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


def add_final_train_reference(ax, final_train_loss: float) -> None:
    if not (is_finite(final_train_loss) and final_train_loss > 0.0):
        return

    ax.axhline(
        final_train_loss,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.82,
        zorder=1.5,
    )
    y_min, y_max = sorted(ax.get_ylim())
    default_ticks = [
        float(value)
        for value in ax.get_yticks()
        if is_finite(float(value)) and y_min <= float(value) <= y_max
    ]
    ticks = sorted({*default_ticks, float(final_train_loss)})

    def format_tick(value: float, _position: int) -> str:
        if math.isclose(value, final_train_loss, rel_tol=1.0e-10, abs_tol=0.0):
            return f"{final_train_loss:.6e}"
        exponent = int(round(math.log10(value))) if value > 0.0 else 0
        if value > 0.0 and math.isclose(value, 10.0**exponent, rel_tol=1.0e-10, abs_tol=0.0):
            return rf"$10^{{{exponent}}}$"
        return ""

    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
    for tick, value in zip(ax.yaxis.get_major_ticks(), ticks):
        if math.isclose(value, final_train_loss, rel_tol=1.0e-10, abs_tol=0.0):
            tick.label1.set_color("steelblue")
            tick.label1.set_fontweight("bold")


def parse_epoch_losses(log_path: Path) -> dict:
    train_losses_by_epoch: dict[int, float] = {}
    val_losses_by_epoch: dict[int, float] = {}
    train_phases_by_epoch: dict[int, str] = {}
    val_phases_by_epoch: dict[int, str] = {}
    role = ""
    label = log_path.name
    final_val = float("nan")
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
            epoch = int(match.group(1))
            current_phase = "adam"
            train_losses_by_epoch[epoch] = parse_float(match.group(2))
            train_phases_by_epoch[epoch] = current_phase
            continue

        match = LBFGS_EPOCH_RE.search(line)
        if match is not None:
            epoch = int(match.group(1))
            current_phase = "lbfgs"
            train_losses_by_epoch[epoch] = parse_float(match.group(2))
            train_phases_by_epoch[epoch] = current_phase
            continue

        match = VAL_EPOCH_RE.search(line)
        if match is not None:
            val_epoch = int(match.group(1))
            val_loss = parse_float(match.group(2))
            if is_finite(val_loss):
                val_losses_by_epoch[val_epoch] = val_loss
                val_phases_by_epoch[val_epoch] = train_phases_by_epoch.get(val_epoch, current_phase)
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
    train_phases = [train_phases_by_epoch.get(e, "adam") for e in epochs]
    val_epochs = sorted(val_losses_by_epoch)
    val_losses = [val_losses_by_epoch[e] for e in val_epochs]
    val_phases = [val_phases_by_epoch.get(e, train_phases_by_epoch.get(e, "adam")) for e in val_epochs]
    return {
        "path": str(log_path),
        "label": label,
        "role": role if role else label,
        "epochs": epochs,
        "train_losses": train_losses,
        "train_phases": train_phases,
        "val_epochs": val_epochs,
        "val_losses": val_losses,
        "val_phases": val_phases,
        "final_val": final_val,
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

    sample_epoch, sample_loss, sample_phase = sample_points(
        epochs,
        losses,
        sample_every,
        rec.get("train_phases"),
    )
    sample_val_epoch, sample_val_loss, sample_val_phase = sample_points(
        rec["val_epochs"],
        rec["val_losses"],
        sample_every,
        rec.get("val_phases"),
    )

    y_values = [v for v in sample_loss if v > 0]
    y_values.extend(v for v in sample_val_loss if v > 0)
    if rec["final_val"] == rec["final_val"] and rec["final_val"] > 0:
        y_values.append(rec["final_val"])

    y_min = min(y_values) / 1.8 if y_values else 1e-12
    y_max = max(y_values) * 1.8 if y_values else 1e-6

    plot_phase_series(
        ax,
        sample_epoch,
        sample_loss,
        sample_phase,
        label="train",
        color="steelblue",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    if sample_val_epoch:
        plot_phase_series(
            ax,
            sample_val_epoch,
            sample_val_loss,
            sample_val_phase,
            label="validation",
            color="firebrick",
            linewidth=2,
            linestyle="--",
            marker="D",
            markersize=3,
        )
    elif rec["final_val"] == rec["final_val"]:
        ax.scatter([epochs[-1]], [rec["final_val"]], label="final val", color="firebrick", marker="D", s=30)

    lbfgs_epochs = [epoch for epoch, phase in zip(epochs, rec.get("train_phases", [])) if phase == "lbfgs"]
    if lbfgs_epochs:
        lbfgs_handoff_epoch = min(lbfgs_epochs) - 0.5
        ax.axvline(
            lbfgs_handoff_epoch,
            label="LBFGS starts",
            color="0.35",
            linestyle=":",
            linewidth=1.8,
            alpha=0.85,
        )

    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, max(epochs[-1], rec["val_epochs"][-1] if rec["val_epochs"] else 0) + 1)
    add_final_train_reference(ax, losses[-1])
    ax.legend(loc="best")


def run_one(log_dir: Path, out_dir: Path, sample_every: int) -> Path:
    log_paths = discover_log_paths(log_dir)
    if not log_paths:
        raise FileNotFoundError(f"No stage2light shard logs found under: {log_dir}")
    recs = [parse_epoch_losses(path) for path in log_paths]
    fig, axes = plt.subplots(1, len(recs), figsize=(6 * len(recs), 5.2), sharey=False, squeeze=False)
    axes = axes[0]
    for ax, rec in zip(axes, recs):
        build_subplot(ax, rec, sample_every=sample_every)

    out_file = out_path(DEFAULT_OUT_FILE, out_dir)
    fig.tight_layout()
    fig.subplots_adjust(left=max(0.22, fig.subplotpars.left))
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {out_file}")

    print(f"Plotted every {sample_every} epoch(s).")
    for rec in recs:
        title = shard_title(rec["role"], rec["label"])
        if not rec["epochs"]:
            print(f"{title}: no epoch data found")
            continue
        print(
            f"{title}: "
            f"epochs={rec['epochs'][0]}-{rec['epochs'][-1]} | "
            f"train_points={len(rec['epochs'])} | "
            f"val_points={len(rec['val_epochs'])}"
        )
    return out_file


def main() -> None:
    log_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_LOG_DIR
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    sample_every = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_SAMPLE_EVERY
    run_one(log_dir, out_dir, sample_every)


if __name__ == "__main__":
    main()
