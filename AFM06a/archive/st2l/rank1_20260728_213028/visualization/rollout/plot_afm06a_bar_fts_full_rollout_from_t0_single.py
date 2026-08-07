from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_PATH = SCRIPT_DIR / "afm06a_stage2light_full_rollout_from_t0_cache_rank1.pkl"
X1_ROLLOUT_SCRIPT = SCRIPT_DIR / "plot_afm06a_x1_full_rollout_from_t0_grid.py"
OUTPUT_PATH = SCRIPT_DIR / "rollout Fts from t0.png"
OUTPUT_CANONICAL_PATH = (
    SCRIPT_DIR / "afm_param_stage2light_06a_bar_fts_full_rollout_from_t0_single.png"
)


def _load_x1_rollout_module():
    spec = importlib.util.spec_from_file_location(
        "afm06a_x1_full_rollout_from_t0_grid", X1_ROLLOUT_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rollout helper: {X1_ROLLOUT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return 100.0 * float(np.sqrt(np.mean((prediction - truth) ** 2))) / max(
        float(np.sqrt(np.mean(truth**2))), 1.0e-30
    )


def main() -> int:
    rollout_module = _load_x1_rollout_module()
    payload = rollout_module._build_rollout_from_t0()

    time_ms = 1.0e3 * np.asarray(payload["times"], dtype=float)
    true_bar_fts = np.asarray(payload["true_bar_fts"], dtype=float)
    pred_bar_fts = np.asarray(payload["predicted_bar_fts"], dtype=float)
    rmse_pct = _relative_rmse_pct(pred_bar_fts, true_bar_fts)

    fig, ax = plt.subplots(figsize=(8.11, 6.0))
    ax.plot(
        time_ms,
        true_bar_fts,
        color="#3168ff",
        linewidth=1.6,
        label="simulated true trajectory",
    )
    ax.plot(
        time_ms,
        pred_bar_fts,
        color="#ff8c00",
        linestyle="--",
        linewidth=1.4,
        label="prediction",
    )
    ax.text(
        0.01,
        0.98,
        f"relative RMSE = {rmse_pct:.3f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
    )
    ax.set_xlabel("time (ms)", fontsize=18)
    ax.set_ylabel(r"$\bar{F}_{ts}$ (m s$^{-2}$)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_CANONICAL_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {OUTPUT_CANONICAL_PATH}")
    print(f"t range: {time_ms[0]:.6f} ms to {time_ms[-1]:.6f} ms")
    print(f"relative RMSE: {rmse_pct:.6f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
