from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM05.stage2light.runner.visualization._common import (  # noqa: E402
    CHECKPOINT_DIR,
    OUT_DIR,
    RESULT_DIR,
    finalize_and_save,
    load_available_payloads,
)


OUTPUT_NAME = "afm_param_stage2light_05_x3_traj_grid.png"


def _final_epoch(payload: dict[str, Any]) -> int:
    history = payload.get("history")
    if not isinstance(history, list):
        raise RuntimeError("The stage2light payload has no epoch history.")
    epochs = [
        int(round(float(row["epoch"])))
        for row in history
        if isinstance(row, dict)
        and row.get("epoch") is not None
        and math.isfinite(float(row["epoch"]))
    ]
    if not epochs:
        raise RuntimeError("The stage2light payload has no finite final epoch.")
    return max(epochs)


def _final_full_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    final_snapshot = payload.get("final_snapshot")
    if not isinstance(final_snapshot, dict):
        raise RuntimeError(
            "The final-epoch x3 plot requires a completed stage2light result with final_snapshot."
        )
    snapshot = final_snapshot.get("full")
    if not isinstance(snapshot, dict):
        raise RuntimeError("The stage2light final_snapshot has no full-window payload.")
    return snapshot


def _candidate_text(payload: dict[str, Any]) -> str:
    warmstart = payload.get("stage1_warmstart") or payload.get("warmstart") or {}
    if not isinstance(warmstart, dict):
        return ""
    candidate = int(warmstart.get("candidate_b", 0) or 0)
    rank = int(warmstart.get("rank", 0) or 0)
    if candidate > 0 and rank > 0:
        return f"B{candidate:02d}, rank {rank}"
    if candidate > 0:
        return f"B{candidate:02d}"
    if rank > 0:
        return f"rank {rank}"
    return ""


def _window_text(payload: dict[str, Any]) -> str:
    meta = payload.get("window_meta")
    if not isinstance(meta, dict):
        return "window"
    return str(meta.get("title") or meta.get("label") or meta.get("role") or "window")


def plot_final_x3(payloads: list[dict[str, Any]], output_path: Path) -> Path:
    if not payloads:
        raise RuntimeError("No completed stage2light payloads were found.")

    fig, axes = plt.subplots(
        1,
        len(payloads),
        figsize=(6.4 * len(payloads), 4.6),
        squeeze=False,
    )
    for column, payload in enumerate(payloads):
        snapshot = _final_full_snapshot(payload)
        epoch = _final_epoch(payload)
        times_us = 1.0e6 * np.asarray(snapshot["times"], dtype=float).reshape(-1)
        traj_pred = np.asarray(snapshot["traj_pred"], dtype=float)
        if traj_pred.ndim != 2 or traj_pred.shape[0] < 3:
            raise RuntimeError(
                "Final snapshot traj_pred must have shape [at least 3 states, time]."
            )
        x3_pred_nm = 1.0e9 * np.asarray(traj_pred[2, :], dtype=float).reshape(-1)
        if times_us.size != x3_pred_nm.size:
            raise RuntimeError(
                "Final snapshot time/x3 length mismatch: "
                f"times={times_us.size}, x3={x3_pred_nm.size}"
            )
        if (
            times_us.size == 0
            or not np.all(np.isfinite(times_us))
            or not np.all(np.isfinite(x3_pred_nm))
        ):
            raise RuntimeError("Final snapshot contains empty or non-finite time/x3 values.")

        ax = axes[0, column]
        ax.plot(
            times_us,
            x3_pred_nm,
            color="seagreen",
            linewidth=2.0,
            label=r"$\hat{x}_3$",
            zorder=3,
        )
        candidate = _candidate_text(payload)
        suffix = f" | {candidate}" if candidate else ""
        ax.set_title(f"{_window_text(payload)} | final epoch {epoch}{suffix}")
        ax.set_xlabel(r"time ($\mu$s)")
        ax.set_ylabel(r"$\hat{x}_3$ (nm)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    return finalize_and_save(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot only the AFM05 stage2light final-epoch predicted x3 trajectory."
    )
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    payloads = load_available_payloads(
        result_dir=args.result_dir.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
    )
    plot_final_x3(payloads, args.out_dir.resolve() / OUTPUT_NAME)


if __name__ == "__main__":
    main()
