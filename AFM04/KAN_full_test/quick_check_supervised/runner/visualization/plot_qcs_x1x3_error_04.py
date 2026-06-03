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


DEFAULT_OUT_FILE = "qcs_supervised_x1x3_error.png"
ALLOW_LOG_FALLBACK_ENV = "HNODECB_AFM04_KFT_QCS_VIS_ALLOW_LOG_FALLBACK"


def _load_error_rows() -> tuple[Path, list[dict], str]:
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
                "random-search-only output does not contain x1/x3 error curves."
            )
        return log_payload["path"], rows, "driver log"


def _has_finite(values: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(np.asarray(values, dtype=float))))


def run_one(out_dir: Path = OUT_DIR) -> Path:
    history_path, history, source_kind = _load_error_rows()
    epochs = history_array(history, "epoch")
    train_x1 = history_array(history, "train_x1_rec")
    val_x1 = history_array(history, "val_x1_rec")
    train_x3 = history_array(history, "train_x3_rec")
    val_x3 = history_array(history, "val_x3_rec")

    if not (_has_finite(train_x1) or _has_finite(val_x1) or _has_finite(train_x3) or _has_finite(val_x3)):
        raise FileNotFoundError(
            "No x1/x3 reconstruction-error fields were found in the selected QCS-GBO history/log. "
            "Run QCS-GBO after the latest logging update."
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_x1, color="tab:blue", linewidth=2, label="train x1")
    ax.plot(epochs, val_x1, color="tab:red", linewidth=2, linestyle="--", label="val x1")
    ax.plot(epochs, train_x3, color="tab:green", linewidth=2, label="train x3")
    ax.plot(epochs, val_x3, color="tab:purple", linewidth=2, linestyle="--", label="val x3")
    ax.set_xlabel("epoch")
    ax.set_ylabel("rollout rel RMSE (%)")
    ax.set_title("KFT quick check supervised x1/x3 rollout error")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", ncols=2)

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
