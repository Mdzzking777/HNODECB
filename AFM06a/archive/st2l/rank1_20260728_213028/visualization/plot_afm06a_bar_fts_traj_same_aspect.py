"""Generate an AFM06a bar_Fts trajectory figure with thesis-panel aspect."""

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
STAGE1_PATH = ARCHIVE_ROOT / "conditional dependency" / "afm_param_stage1pluslight_06a.pkl"
DATA_PATH = ARCHIVE_ROOT / "data" / "pert_df_afm_dmt_hard.npz"
OUTPUT_PATH = VIS_DIR / "afm_param_stage2light_06a_bar_fts_traj_same_aspect.png"
FONT_SIZE = 15


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(truth, dtype=float)
    return 100.0 * float(np.sqrt(np.mean(np.square(pred - ref)))) / max(
        float(np.sqrt(np.mean(np.square(ref)))), 1.0e-30
    )


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected pickle payload type in {path}: {type(payload)!r}")
    return payload


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
    stage1 = _load_pickle(STAGE1_PATH)
    source_idx = np.asarray(stage1["window_source_indices"], dtype=int)
    times = np.asarray(result["times"], dtype=float)
    predicted = np.asarray(result["predicted_bar_fts"], dtype=float)
    with np.load(DATA_PATH, allow_pickle=False) as data:
        truth = np.asarray(data["bar_fts"], dtype=float)[source_idx]

    if times.shape != predicted.shape or times.shape != truth.shape:
        raise ValueError(
            f"mismatched arrays: times={times.shape}, predicted={predicted.shape}, truth={truth.shape}"
        )

    times_ms = times * 1.0e3
    fig, ax = plt.subplots(figsize=(8.02, 6.4))
    ax.plot(
        times_ms,
        truth,
        color="black",
        linewidth=2.0,
        label=r"$\bar{F}_{ts}$",
        zorder=2,
    )
    ax.plot(
        times_ms,
        predicted,
        color="crimson",
        linestyle="-",
        linewidth=1.0,
        label=r"$\hat{F}_{ts}$",
        zorder=3,
    )
    ax.text(
        0.98,
        0.95,
        f"relative RMSE = {_relative_rmse_pct(predicted, truth):.4f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZE,
    )
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$\bar{F}_{ts}$ (m s$^{-2}$)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
