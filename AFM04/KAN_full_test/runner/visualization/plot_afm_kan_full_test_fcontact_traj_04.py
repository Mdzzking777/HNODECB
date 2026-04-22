from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.runner.visualization._common import load_result_payloads, out_path, finalize_and_save, stage_title, window_title


def main() -> None:
    payloads = load_result_payloads()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=False)
    for ax, payload in zip(axes, payloads):
        best_snapshot = payload["best"]["best_snapshot"]
        snap = best_snapshot.get("full") or best_snapshot.get("val")
        if snap is None:
            raise KeyError("Neither 'full' nor 'val' snapshot found in payload['best']['best_snapshot']")
        times_us = 1.0e6 * np.asarray(snap["times"], dtype=float)
        f_true = np.asarray(snap["fts_teacher_true"], dtype=float)
        f_pred = np.asarray(snap["fts_teacher_pred"], dtype=float)
        title = f"{stage_title(payload)}: {window_title(payload)}"
        ax.plot(times_us, f_true, color="black", linewidth=2, label="Fts true")
        ax.plot(times_us, f_pred, color="crimson", linewidth=2, linestyle="--", label="Fts pred")
        ax.set_title(title)
        ax.set_xlabel("time (μs)")
        ax.set_ylabel("force (N)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    finalize_and_save(fig, out_path("afm_param_kan_full_test_04_fcontact_grid.png"))


if __name__ == "__main__":
    main()
