from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from AFM04.KAN_full_test.quick_check_supervised.runner.visualization._common import (  # noqa: E402
    OUT_DIR,
    finalize_and_save,
    history_array,
    load_latest_history,
    out_path,
    parse_training_log,
)


DEFAULT_OUT_FILE = "qcs_supervised_losses.png"
ALLOW_LOG_FALLBACK_ENV = "HNODECB_AFM04_KFT_QCS_VIS_ALLOW_LOG_FALLBACK"


def _load_loss_rows() -> tuple[Path, list[dict], str]:
    try:
        history_path, history = load_latest_history()
        return history_path, history, "history"
    except FileNotFoundError as exc:
        if os.environ.get(ALLOW_LOG_FALLBACK_ENV) != "1":
            raise FileNotFoundError(
                "No QCS-GBO result history was found. This visualization reads current logs only "
                "when launched by run_qcs_core_visualizations_04.py."
            ) from exc
        log_payload = parse_training_log()
        rows = log_payload["rows"]
        if not rows:
            raise FileNotFoundError(
                "No QCS-GBO history or epoch rows were found. Run QCS-GBO first; "
                "random-search-only output does not contain training loss curves."
            )
        return log_payload["path"], rows, "driver log"


def _has_finite(values) -> bool:
    arr = np.asarray(values, dtype=float)
    return bool(np.any(np.isfinite(arr)))


def _positive_for_semilogy(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr) & (arr > 0.0), arr, np.nan)


def _plot_semilogy_if_finite(ax, epochs: np.ndarray, values: np.ndarray, **kwargs) -> bool:
    arr = _positive_for_semilogy(values)
    if not _has_finite(arr):
        return False
    ax.semilogy(epochs, arr, **kwargs)
    return True


def _plot_if_finite(ax, epochs: np.ndarray, values: np.ndarray, **kwargs) -> bool:
    arr = np.asarray(values, dtype=float)
    if not _has_finite(arr):
        return False
    ax.plot(epochs, arr, **kwargs)
    return True


def run_one(out_dir: Path = OUT_DIR) -> Path:
    history_path, history, source_kind = _load_loss_rows()
    epochs = history_array(history, "epoch")
    train = history_array(history, "train_loss")
    val = history_array(history, "val_loss")
    train_formal = history_array(history, "train_formal_total")
    val_formal = history_array(history, "val_formal_total")
    train_fts = history_array(history, "train_fts_norm_mse")
    val_fts = history_array(history, "val_fts_norm_mse")
    train_x1_state = history_array(history, "train_x1_state")
    val_x1_state = history_array(history, "val_x1_state")
    train_x3_range = history_array(history, "train_x3_range")
    val_x3_range = history_array(history, "val_x3_range")
    train_x1_rec = history_array(history, "train_x1_rec")
    val_x1_rec = history_array(history, "val_x1_rec")
    train_x3_rec = history_array(history, "train_x3_rec")
    val_x3_rec = history_array(history, "val_x3_rec")

    has_components = _has_finite(train_formal) or _has_finite(val_formal) or _has_finite(train_fts) or _has_finite(val_fts)
    has_state_terms = (
        _has_finite(train_x1_state)
        or _has_finite(val_x1_state)
        or _has_finite(train_x3_range)
        or _has_finite(val_x3_range)
        or _has_finite(train_x1_rec)
        or _has_finite(val_x1_rec)
        or _has_finite(train_x3_rec)
        or _has_finite(val_x3_rec)
    )
    nrows = 1 + int(has_components) + int(has_state_terms)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 + 3 * (nrows - 1)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    _plot_semilogy_if_finite(axes[0], epochs, train, color="steelblue", linewidth=2, label="train")
    _plot_semilogy_if_finite(axes[0], epochs, val, color="firebrick", linewidth=2, linestyle="--", label="validation")
    axes[0].set_ylabel("hybrid loss")
    axes[0].set_title("KFT quick check hybrid supervised losses")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(loc="best")

    axis_index = 1
    if has_components:
        ax = axes[axis_index]
        _plot_semilogy_if_finite(ax, epochs, train_formal, color="steelblue", linewidth=1.7, label="train formal")
        _plot_semilogy_if_finite(ax, epochs, val_formal, color="firebrick", linewidth=1.7, linestyle="--", label="val formal")
        _plot_semilogy_if_finite(ax, epochs, train_fts, color="darkgreen", linewidth=1.7, label="train FtsSup")
        _plot_semilogy_if_finite(ax, epochs, val_fts, color="purple", linewidth=1.7, linestyle="--", label="val FtsSup")
        _plot_semilogy_if_finite(ax, epochs, train_x1_state, color="tab:cyan", linewidth=1.4, label="train x1 state")
        _plot_semilogy_if_finite(ax, epochs, val_x1_state, color="tab:orange", linewidth=1.4, linestyle="--", label="val x1 state")
        _plot_semilogy_if_finite(ax, epochs, train_x3_range, color="tab:gray", linewidth=1.4, label="train x3 range")
        _plot_semilogy_if_finite(ax, epochs, val_x3_range, color="black", linewidth=1.4, linestyle="--", label="val x3 range")
        ax.set_ylabel("normalized loss")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best", ncols=2, fontsize=8)
        axis_index += 1

    if has_state_terms:
        ax = axes[axis_index]
        _plot_if_finite(ax, epochs, train_x1_rec, color="tab:blue", linewidth=1.8, label="train x1 rec")
        _plot_if_finite(ax, epochs, val_x1_rec, color="tab:red", linewidth=1.8, linestyle="--", label="val x1 rec")
        _plot_if_finite(ax, epochs, train_x3_rec, color="tab:green", linewidth=1.8, label="train x3 rec")
        _plot_if_finite(ax, epochs, val_x3_rec, color="tab:purple", linewidth=1.8, linestyle="--", label="val x3 rec")
        ax.set_ylabel("rollout rel RMSE [%]")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best", ncols=2, fontsize=8)

    axes[-1].set_xlabel("epoch")

    print(f"Source {source_kind}: {history_path}")
    return finalize_and_save(fig, out_path(DEFAULT_OUT_FILE, out_dir))


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else OUT_DIR
    try:
        run_one(out_dir)
    except FileNotFoundError as exc:
        print(f"[skip] {exc}")


if __name__ == "__main__":
    main()

