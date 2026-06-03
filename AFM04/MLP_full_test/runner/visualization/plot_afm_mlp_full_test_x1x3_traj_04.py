from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.MLP_full_test.runner.visualization._common import load_result_payloads, out_path, finalize_and_save, stage_title, window_title


def main() -> None:
    payloads = load_result_payloads()
    ncols = max(1, len(payloads))
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 9), sharex=False, squeeze=False)
    for col, payload in enumerate(payloads):
        best_snapshot = payload["best"]["best_snapshot"]
        snap = best_snapshot.get("full") or best_snapshot.get("val")
        if snap is None:
            raise KeyError("Neither 'full' nor 'val' snapshot found in payload['best']['best_snapshot']")
        times_us = 1.0e6 * np.asarray(snap["times"], dtype=float)
        ode_true = np.asarray(snap["ode_true"], dtype=float)
        traj_pred = np.asarray(snap["traj_pred"], dtype=float)
        title = f"{stage_title(payload)}: {window_title(payload)}"

        ax1 = axes[0, col]
        ax1.plot(times_us, ode_true[0, :], color="black", linewidth=2, label="x1 true")
        ax1.plot(times_us, traj_pred[0, :], color="crimson", linewidth=2, linestyle="--", label="x1 pred")
        ax1.set_title(title)
        ax1.set_xlabel("time (μs)")
        ax1.set_ylabel("x1")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        ax2.plot(times_us, ode_true[2, :], color="black", linewidth=2, label="x3 true")
        ax2.plot(times_us, traj_pred[2, :], color="crimson", linewidth=2, linestyle="--", label="x3 pred")
        ax2.set_title(title)
        ax2.set_xlabel("time (μs)")
        ax2.set_ylabel("x3")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

    finalize_and_save(fig, out_path("afm_param_MLP_full_test_04_x1x3_traj_grid.png"))


if __name__ == "__main__":
    main()
