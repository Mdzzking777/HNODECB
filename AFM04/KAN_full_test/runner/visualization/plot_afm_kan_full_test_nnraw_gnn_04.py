from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.runner.visualization._common import load_result_payloads, out_path, finalize_and_save, stage_title, window_title


def main() -> None:
    payloads = load_result_payloads()
    fig, axes = plt.subplots(5, 3, figsize=(18, 19), sharex=False)
    for col, payload in enumerate(payloads):
        snap = payload["best"]["best_snapshot"]["full"]
        times_us = 1.0e6 * np.asarray(snap["times"], dtype=float)
        nn_raw = np.asarray(snap["nn_raw_rollout"], dtype=float)
        w_pred = np.asarray(snap.get("w_pred_rollout", np.ones_like(nn_raw)), dtype=float)
        nn_weighted = np.asarray(snap.get("nn_weighted_rollout", nn_raw), dtype=float)
        f_true = np.asarray(snap["fts_rollout_true"], dtype=float)
        g_nn = float(payload["best"]["best_snapshot"].get("g_nn", float("nan")))
        nn_scaled = g_nn * nn_weighted
        title = f"{stage_title(payload)}: {window_title(payload)}"

        ax1 = axes[0, col]
        ax1.plot(times_us, nn_raw, color="crimson", linewidth=2, label="KAN raw")
        ax1.set_title(title)
        ax1.set_xlabel("time (μs)")
        ax1.set_ylabel("NN_raw(t)")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        ax2.plot(times_us, w_pred, color="darkorange", linewidth=2, label="w_pred")
        ax2.set_title(title)
        ax2.set_xlabel("time (μs)")
        ax2.set_ylabel("w_pred(t)")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

        ax3 = axes[2, col]
        ax3.plot(times_us, nn_weighted, color="purple", linewidth=2, label="w_pred * NN_raw")
        ax3.set_title(title)
        ax3.set_xlabel("time (μs)")
        ax3.set_ylabel("gated NN")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="best")

        ax4 = axes[3, col]
        ax4.plot(times_us, f_true, color="black", linewidth=2, label="F_contact true")
        ax4.plot(times_us, nn_scaled, color="teal", linewidth=2, linestyle="--", label="g_nn * w_pred * NN_raw")
        ax4.set_title(title)
        ax4.set_xlabel("time (μs)")
        ax4.set_ylabel("force")
        ax4.grid(True, alpha=0.25)
        ax4.legend(loc="best")

        ax5 = axes[4, col]
        ax5.plot(times_us, np.full_like(times_us, g_nn), color="royalblue", linewidth=2, label="g_nn")
        ax5.set_title(title)
        ax5.set_xlabel("time (μs)")
        ax5.set_ylabel("g_nn")
        ax5.grid(True, alpha=0.25)
        ax5.legend(loc="best")

    finalize_and_save(fig, out_path("afm_param_kan_full_test_04_nnraw_gnn_grid.png"))


if __name__ == "__main__":
    main()
