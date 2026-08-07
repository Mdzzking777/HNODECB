"""Plot AFM06a archived rollout in x1-x2-time state space.

Data source:
AFM06a/archive/st2l/rank1_20260728_213028/visualization/
    afm06a_stage2light_full_rollout_cache_rank1.pkl

This deliberately does not read AFM06 live data.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARCHIVE_DIR = Path(__file__).resolve().parent
VIS_DIR = ARCHIVE_DIR / "st2l" / "rank1_20260728_213028" / "visualization"
CACHE_PATH = VIS_DIR / "afm06a_stage2light_full_rollout_cache_rank1.pkl"
OUTPUT = ARCHIVE_DIR / "rollout state space.png"
MAX_RENDER_POINTS = 25_000


def render_indices(point_count: int) -> np.ndarray:
    stride = max(1, int(np.ceil(point_count / MAX_RENDER_POINTS)))
    indices = np.arange(0, point_count, stride, dtype=int)
    if indices[-1] != point_count - 1:
        indices = np.append(indices, point_count - 1)
    return indices


def main() -> int:
    with CACHE_PATH.open("rb") as handle:
        payload = pickle.load(handle)

    times = np.asarray(payload["times"], dtype=float)
    true_states = np.asarray(payload["true_states"], dtype=float)
    predicted_states = np.asarray(payload["predicted_states"], dtype=float)
    if true_states.shape != predicted_states.shape or true_states.shape[0] != 2:
        raise ValueError(
            "expected true_states and predicted_states with matching shape (2, N), "
            f"got {true_states.shape} and {predicted_states.shape}"
        )
    if true_states.shape[1] != times.size:
        raise ValueError("times and state arrays do not have matching lengths")

    idx = render_indices(times.size)
    time_ms = times[idx] * 1.0e3
    true_x1_nm = true_states[0, idx] * 1.0e9
    true_x2_um_s = true_states[1, idx] * 1.0e6
    pred_x1_nm = predicted_states[0, idx] * 1.0e9
    pred_x2_um_s = predicted_states[1, idx] * 1.0e6

    fig = plt.figure(figsize=(10.5, 8.2))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(
        true_x1_nm,
        true_x2_um_s,
        time_ms,
        color="royalblue",
        linewidth=0.9,
        alpha=0.9,
        label="simulated true trajectory",
        zorder=2,
    )
    axis.plot(
        pred_x1_nm,
        pred_x2_um_s,
        time_ms,
        color="darkorange",
        linestyle="--",
        linewidth=1.35,
        alpha=0.95,
        label="prediction",
        zorder=3,
    )
    axis.set_xlabel(r"Tip displacement, $x_1$ [nm]", labelpad=11, fontsize=14)
    axis.set_ylabel(
        r"Tip velocity, $x_2$ [$\mu\mathrm{m}\,\mathrm{s}^{-1}$]",
        labelpad=13,
        fontsize=14,
    )
    axis.set_zlabel("")
    axis.set_zlim(float(time_ms[0]), float(time_ms[-1]))
    axis.view_init(elev=23, azim=-56)
    axis.tick_params(axis="both", labelsize=14)
    axis.grid(True, alpha=0.22)
    axis.legend(loc="upper left", frameon=False, fontsize=11)
    fig.text(
        0.835,
        0.515,
        "Time [ms]",
        rotation=90,
        ha="center",
        va="center",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.01, right=0.88, bottom=0.03, top=0.99)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"Saved: {OUTPUT}")
    print(
        f"source={CACHE_PATH} | time=[{times[0] * 1e3:.9g}, {times[-1] * 1e3:.9g}] ms | "
        f"source points={times.size} | rendered points={idx.size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
