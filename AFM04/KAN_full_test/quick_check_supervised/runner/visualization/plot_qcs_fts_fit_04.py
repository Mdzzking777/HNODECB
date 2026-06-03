from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from AFM04.KAN_full_test.quick_check_supervised.runner.visualization._common import (  # noqa: E402
    OUT_DIR,
    finite_minmax,
    finalize_and_save,
    load_latest_snapshot,
    out_path,
)


DEFAULT_OUT_FILE = "qcs_supervised_fts_fit.png"


def run_one(out_dir: Path = OUT_DIR) -> Path:
    snapshot_path, snap = load_latest_snapshot()
    t_us = np.asarray(snap["times_full"], dtype=float) * 1.0e6
    true = np.asarray(snap["fts_full_true"], dtype=float) * 1.0e9
    pred = np.asarray(snap["fts_full_pred"], dtype=float) * 1.0e9
    contact = np.asarray(snap["contact_full"], dtype=bool)
    residual = pred - true

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    axes[0].plot(t_us, true, color="black", linewidth=1.8, label="Fts teacher true")
    axes[0].plot(t_us, pred, color="tab:blue", linewidth=1.4, label="Fts rollout pred")
    if np.any(contact):
        axes[0].fill_between(t_us, np.nanmin(true), np.nanmax(true), where=contact, color="tab:red", alpha=0.08, label="contact")
    axes[0].set_ylabel("Fts [nN]")
    axes[0].set_title("KFT quick check hybrid supervised Fts fit")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(loc="best")

    axes[1].plot(t_us, residual, color="tab:red", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("time [us]")
    axes[1].set_ylabel("pred - true [nN]")
    axes[1].grid(True, alpha=0.28)

    lo, hi = finite_minmax(true, pred)
    axes[2].scatter(true[~contact], pred[~contact], s=16, alpha=0.65, label="noncontact", color="tab:blue")
    if np.any(contact):
        axes[2].scatter(true[contact], pred[contact], s=20, alpha=0.75, label="contact", color="tab:red")
    axes[2].plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    axes[2].set_xlabel("Fts true [nN]")
    axes[2].set_ylabel("Fts pred [nN]")
    axes[2].grid(True, alpha=0.28)
    axes[2].legend(loc="best")

    print(f"Source snapshot: {snapshot_path}")
    return finalize_and_save(fig, out_path(DEFAULT_OUT_FILE, out_dir))


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else OUT_DIR
    try:
        run_one(out_dir)
    except FileNotFoundError as exc:
        print(f"[skip] {exc}")


if __name__ == "__main__":
    main()

