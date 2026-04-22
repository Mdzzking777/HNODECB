from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.runner.visualization._common import load_result_payloads, out_path, finalize_and_save, stage_title, window_title


def main() -> None:
    payloads = load_result_payloads()
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=False)
    for col, payload in enumerate(payloads):
        hist = payload.get("history", [])
        epochs = [int(row["epoch"]) for row in hist]
        x1_state = [float(row.get("val_x1_state", float("nan"))) for row in hist]
        x2_state = [float(row.get("val_x2_state", float("nan"))) for row in hist]
        x2dot = [float(row.get("val_x2dot", float("nan"))) for row in hist]
        title = f"{stage_title(payload)}: {window_title(payload)}"

        ax1 = axes[0, col]
        ax1.plot(epochs, x1_state, color="seagreen", linewidth=2, label="x1_state")
        ax1.set_title(title)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("x1 state loss")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        ax2.plot(epochs, x2_state, color="royalblue", linewidth=2, label="x2_state")
        ax2.set_title(title)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("x2 state loss")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

        ax3 = axes[2, col]
        ax3.plot(epochs, x2dot, color="darkorange", linewidth=2, label="x2dot")
        ax3.set_title(title)
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("x2dot loss")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="best")

    finalize_and_save(fig, out_path("afm_param_kan_full_test_04_x1x2state_x2dot_grid.png"))


if __name__ == "__main__":
    main()
