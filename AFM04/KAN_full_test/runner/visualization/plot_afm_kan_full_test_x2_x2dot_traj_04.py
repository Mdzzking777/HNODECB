from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.runner.visualization._common import (
    finalize_and_save,
    load_result_payloads,
    out_path,
    stage_title,
    time_snapshot,
    title_with_progress,
    window_title,
)


def main() -> None:
    payloads = load_result_payloads()
    ncols = max(1, len(payloads))
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 9), sharex=False, squeeze=False)
    for col, payload in enumerate(payloads):
        snap = time_snapshot(payload).get("full") or time_snapshot(payload).get("val")
        if snap is None:
            raise KeyError("Neither 'full' nor 'val' snapshot found in time snapshot")
        times_us = 1.0e6 * np.asarray(snap["times"], dtype=float)
        ode_true = np.asarray(snap["ode_true"], dtype=float)
        traj_pred = np.asarray(snap["traj_pred"], dtype=float)
        x2dot_true = np.asarray(snap["x2dot_true"], dtype=float)
        x2dot_pred = np.asarray(snap["x2dot_pred"], dtype=float)
        title = title_with_progress(payload, f"{stage_title(payload)}: {window_title(payload)}")

        ax1 = axes[0, col]
        ax1.plot(times_us, ode_true[1, :], color="black", linewidth=2, label="x2 true")
        ax1.plot(times_us, traj_pred[1, :], color="crimson", linewidth=2, linestyle="--", label="x2 pred")
        ax1.set_title(title)
        ax1.set_xlabel("time (μs)")
        ax1.set_ylabel("x2")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        ax2.plot(times_us, x2dot_true, color="black", linewidth=2, label="x2dot true")
        ax2.plot(times_us, x2dot_pred, color="crimson", linewidth=2, linestyle="--", label="x2dot pred")
        ax2.set_title(title)
        ax2.set_xlabel("time (μs)")
        ax2.set_ylabel("x2dot")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

    finalize_and_save(fig, out_path("afm_param_kan_full_test_04_x2_x2dot_traj_grid.png"))


if __name__ == "__main__":
    main()
