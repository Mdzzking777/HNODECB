from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.runner.visualization._common import load_result_payloads, out_path, finalize_and_save, stage_title, window_title


def main() -> None:
    payloads = load_result_payloads()
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=False)
    for col, payload in enumerate(payloads):
        hist = payload.get("history", [])
        epochs = [int(row["epoch"]) for row in hist]
        mech_true = np.asarray(payload.get("mech_true", [np.nan, np.nan]), dtype=float)
        ks_series = np.full(len(epochs), mech_true[0], dtype=float)
        cs_series = np.full(len(epochs), mech_true[1], dtype=float)
        title = f"{stage_title(payload)}: {window_title(payload)}"

        ax1 = axes[0, col]
        ax1.plot(epochs, ks_series, color="black", linewidth=2, label="ks (known)")
        ax1.axhline(mech_true[0], color="royalblue", linestyle="--", linewidth=2, label="ks_true")
        ax1.set_title(title)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("ks")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        ax2.plot(epochs, cs_series, color="black", linewidth=2, label="cs (known)")
        ax2.axhline(mech_true[1], color="royalblue", linestyle="--", linewidth=2, label="cs_true")
        ax2.set_title(title)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("cs")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

    finalize_and_save(fig, out_path("afm_param_kan_full_test_04_mech_grid.png"))


if __name__ == "__main__":
    main()
