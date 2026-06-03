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
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.2), sharex=False, squeeze=False)
    for ax, payload in zip(axes[0], payloads):
        snap = time_snapshot(payload).get("full") or time_snapshot(payload).get("val")
        if snap is None:
            raise KeyError("Neither 'full' nor 'val' snapshot found in time snapshot")
        times_us = 1.0e6 * np.asarray(snap["times"], dtype=float)
        f_true = np.asarray(snap["fts_teacher_true"], dtype=float)
        f_pred = np.asarray(snap["fts_rollout_pred"], dtype=float)
        title = title_with_progress(payload, f"{stage_title(payload)}: {window_title(payload)}")
        ax.plot(times_us, f_true, color="black", linewidth=2, label="Fts teacher true", zorder=2)
        ax.plot(times_us, f_pred, color="crimson", linewidth=2, linestyle="-", label="Fts rollout pred", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("time (μs)")
        ax.set_ylabel("force (N)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    finalize_and_save(fig, out_path("afm_param_kan_full_test_04_fcontact_grid.png"))


if __name__ == "__main__":
    main()
