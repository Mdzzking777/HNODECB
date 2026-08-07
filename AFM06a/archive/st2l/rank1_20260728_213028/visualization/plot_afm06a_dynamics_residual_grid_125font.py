"""Regenerate AFM06a dynamics-residual grid with 1.25x text size."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VIS_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = VIS_DIR.parent
RESULT_PATH = ARCHIVE_ROOT / "result" / "afm_param_stage2light_06a_rank1.pkl"
OUTPUT_PATH = VIS_DIR / "afm_param_stage2light_06a_dynamics_residual_grid.png"
FONT_SIZE = 15


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected pickle payload type in {path}: {type(payload)!r}")
    return payload


def _nested_series(history: list[dict], group: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    epochs: list[int] = []
    values: list[float] = []
    for row in history:
        nested = row.get(group)
        if not isinstance(nested, dict):
            continue
        try:
            epoch = int(row["epoch"])
            value = float(nested[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            epochs.append(epoch)
            values.append(value)
    return np.asarray(epochs, dtype=int), np.asarray(values, dtype=float)


def _mark_lbfgs_start(ax: plt.Axes, adam_epochs: int, *, label: bool = False) -> None:
    ax.axvline(
        int(adam_epochs) + 1,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        label="L-BFGS starts" if label else None,
        zorder=5,
    )


def main() -> int:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )
    result = _load_pickle(RESULT_PATH)
    history = list(result.get("history", []))
    config = result.get("config", {})
    adam_epochs = int(config.get("adam_epochs", 10))

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
    items = (
        ("dynamics_residual", r"$\mathcal{L}_{\mathrm{dyn}}$"),
        ("smoothness", r"$\lambda_{\mathrm{sm}}\mathcal{L}_{\mathrm{sm}}$"),
    )
    for panel_index, (ax, (key, ylabel)) in enumerate(zip(axes, items, strict=True)):
        for group, label, color, style in (
            ("loss_parts", "training", "royalblue", "-"),
            ("val_loss_parts", "validation", "crimson", "--"),
        ):
            epochs, values = _nested_series(history, group, key)
            if values.size:
                ax.semilogy(
                    epochs,
                    np.maximum(values, np.finfo(float).tiny),
                    color=color,
                    linestyle=style,
                    linewidth=1.8,
                    label=label,
                )
        _mark_lbfgs_start(ax, adam_epochs, label=panel_index == 0)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
