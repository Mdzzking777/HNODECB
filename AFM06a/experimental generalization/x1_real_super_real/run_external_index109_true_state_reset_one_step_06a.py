"""Teacher-forced one-step AFM06a prediction on external index109 data.

Every true external x1 sample is evaluated by the trained KAN first.  Each
subsequent one-step RK4 prediction is independently initialized from the true
external x1 and x2 at that step; predicted states are never fed into the next
step or back into the KAN.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

pointwise = importlib.import_module("predict_external_index109_with_trained_kan_06a")


ARCHIVE_DIR = (
    REPO_ROOT
    / "AFM06a"
    / "archive"
    / "st2l"
    / "rank1_20260803_183835"
)
DATA_MANIFEST = (
    REPO_ROOT
    / "AFM06a"
    / "datasets"
    / "e0.0_real"
    / "data"
    / "afm06a_generation_manifest.json"
)
OUTPUT_STEM = "external_index109_true_state_reset_one_step_x1_prediction"


def _load_external_x2_raw(mat_path: Path) -> np.ndarray:
    payload = sio.loadmat(
        str(mat_path),
        variable_names=["velocity_fwd_sweep"],
        squeeze_me=True,
        struct_as_record=False,
    )
    if "velocity_fwd_sweep" not in payload:
        raise KeyError(f"velocity_fwd_sweep is missing from {mat_path}")
    x2 = np.asarray(
        payload["velocity_fwd_sweep"][pointwise.INDEX],
        dtype=np.float64,
    ).reshape(-1)
    if x2.size < 2 or not np.all(np.isfinite(x2)):
        raise ValueError(f"invalid external x2 raw array: shape={x2.shape}")
    return x2


def _load_rhs_parameters() -> dict[str, float]:
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    names = [str(value) for value in manifest["parameter_names"]]
    values = [float(value) for value in manifest["parameter_vector"]]
    parameters = dict(zip(names, values, strict=True))
    required = ("k", "omega0", "m", "c", "Fd")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise KeyError(f"missing AFM06a RHS parameters: {missing}")
    return {name: parameters[name] for name in required}


def _rhs(
    time_s: np.ndarray,
    states: np.ndarray,
    force_N: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    x1 = states[:, 0]
    x2 = states[:, 1]
    x2dot = (
        parameters["Fd"] * np.sin(parameters["omega0"] * time_s)
        - parameters["c"] * x2
        - parameters["k"] * x1
        + force_N
    ) / parameters["m"]
    return np.column_stack((x2, x2dot))


def _true_state_reset_rk4(
    time_s: np.ndarray,
    x1_true_m: np.ndarray,
    x2_true_m_s: np.ndarray,
    force_N: np.ndarray,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.diff(time_s)
    if np.any(h <= 0.0):
        raise ValueError("external time samples must be strictly increasing")
    y0 = np.column_stack((x1_true_m[:-1], x2_true_m_s[:-1]))
    t0 = time_s[:-1]
    t1 = time_s[1:]
    f0 = force_N[:-1]
    f1 = force_N[1:]
    fmid = 0.5 * (f0 + f1)

    k1 = _rhs(t0, y0, f0, parameters)
    k2 = _rhs(t0 + 0.5 * h, y0 + 0.5 * h[:, None] * k1, fmid, parameters)
    k3 = _rhs(t0 + 0.5 * h, y0 + 0.5 * h[:, None] * k2, fmid, parameters)
    k4 = _rhs(t1, y0 + h[:, None] * k3, f1, parameters)
    predicted_next = y0 + (h[:, None] / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return predicted_next[:, 0], predicted_next[:, 1], k1[:, 1]


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray) -> float:
    error_rms = float(np.sqrt(np.mean(np.square(prediction - truth))))
    truth_rms = max(float(np.sqrt(np.mean(np.square(truth)))), 1.0e-30)
    return 100.0 * error_rms / truth_rms


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _plot(
    time_s: np.ndarray,
    true_x1_m: np.ndarray,
    predicted_x1_m: np.ndarray,
    output_path: Path,
) -> None:
    period_s = pointwise.PERIOD_S
    centers = (
        max(float(time_s[0]) + period_s, 0.10 * float(time_s[-1])),
        0.50 * float(time_s[-1]),
        min(float(time_s[-1]) - period_s, 0.90 * float(time_s[-1])),
    )
    panels: list[tuple[str, np.ndarray]] = [
        ("Full time-span", np.arange(time_s.size, dtype=int))
    ]
    for label, center in zip(
        ("Early window", "Middle window", "Tail window"), centers, strict=True
    ):
        panels.append(
            (
                label,
                np.flatnonzero(
                    (time_s >= center - period_s)
                    & (time_s <= center + period_s)
                ),
            )
        )

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2))
    time_us = time_s * 1.0e6
    for axis, (label, indices) in zip(axes, panels, strict=True):
        axis.plot(
            time_us[indices],
            true_x1_m[indices] * 1.0e9,
            color="black",
            linewidth=1.15,
            label="external true $x_1$",
            zorder=1,
        )
        axis.plot(
            time_us[indices],
            predicted_x1_m[indices] * 1.0e9,
            color="#d62728",
            linewidth=0.75,
            label="one-step prediction",
            zorder=2,
        )
        axis.set_title(label)
        axis.set_xlabel(r"Time [$\mu$s]")
        axis.set_ylabel(r"$x_1$ [nm]")
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.0)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    torch.set_num_threads(1)
    model, endpoint, payload, stage2_result, stage1_result = pointwise._restore_stage2_model(
        ARCHIVE_DIR
    )
    time_s, x1_true_m, mat_source = pointwise._load_external_x1_raw(None)
    x2_true_m_s = _load_external_x2_raw(mat_source)
    if x2_true_m_s.shape != x1_true_m.shape:
        raise ValueError(
            f"external x1/x2 length mismatch: {x1_true_m.shape} vs {x2_true_m_s.shape}"
        )

    normalizer = pointwise._record_training_normalizer(model)
    force_N = pointwise._predict_fts_N(model, x1_true_m, chunk_size=32768)
    parameters = _load_rhs_parameters()
    predicted_x1_next_m, predicted_x2_next_m_s, predicted_x2dot_m_s2 = (
        _true_state_reset_rk4(
            time_s,
            x1_true_m,
            x2_true_m_s,
            force_N,
            parameters,
        )
    )

    target_time_s = time_s[1:]
    target_x1_m = x1_true_m[1:]
    target_x2_m_s = x2_true_m_s[1:]
    x1_error_m = predicted_x1_next_m - target_x1_m
    x2_error_m_s = predicted_x2_next_m_s - target_x2_m_s

    output_npz = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
    np.savez_compressed(
        output_npz,
        source_time_s=time_s[:-1],
        target_time_s=target_time_s,
        true_x1_source_m=x1_true_m[:-1],
        true_x2_source_m_s=x2_true_m_s[:-1],
        predicted_fts_source_N=force_N[:-1],
        predicted_x2dot_source_m_s2=predicted_x2dot_m_s2,
        true_x1_target_m=target_x1_m,
        predicted_x1_target_m=predicted_x1_next_m,
        x1_one_step_error_m=x1_error_m,
        true_x2_target_m_s=target_x2_m_s,
        predicted_x2_target_m_s=predicted_x2_next_m_s,
        x2_one_step_error_m_s=x2_error_m_s,
        dt_s=np.diff(time_s),
    )
    _plot(target_time_s, target_x1_m, predicted_x1_next_m, output_png)

    metadata = model.metadata()
    summary = {
        "definition": (
            "Teacher-forced one-step prediction: all external true x1 samples are "
            "fed pointwise to the trained KAN first. Each RK4 interval is reset to "
            "the true external x1 and x2 at its left endpoint; no predicted state is "
            "carried into the next interval or fed back to the KAN."
        ),
        "archive_dir": str(ARCHIVE_DIR),
        "stage2_result": str(stage2_result),
        "stage1_dependency": str(stage1_result),
        "external_mat_source": str(mat_source),
        "sample_count": int(time_s.size),
        "one_step_count": int(target_time_s.size),
        "time_range_us": [float(time_s[0] * 1.0e6), float(time_s[-1] * 1.0e6)],
        "normalizer": normalizer,
        "force_output_policy": str(metadata["force_output_policy"]),
        "force_output_quantity": "Fts_N",
        "rhs_parameters_SI": parameters,
        "integration": "independent true-state-reset RK4 over every 4 ns interval",
        "metrics": {
            "x1_relative_rmse_pct": _relative_rmse_pct(predicted_x1_next_m, target_x1_m),
            "x1_error_nm": _stats(x1_error_m * 1.0e9),
            "x2_relative_rmse_pct": _relative_rmse_pct(predicted_x2_next_m_s, target_x2_m_s),
            "x2_error_m_s": _stats(x2_error_m_s),
            "predicted_x2dot_m_s2": _stats(predicted_x2dot_m_s2),
            "predicted_fts_nN": _stats(force_N[:-1] * 1.0e9),
        },
        "outputs": {
            "npz": str(output_npz),
            "png": str(output_png),
        },
    }
    _write_json(output_json, summary)
    print(f"Saved: {output_npz}")
    print(f"Saved: {output_json}")
    print(f"Saved: {output_png}")
    print(
        "x1 one-step relative RMSE: "
        f"{summary['metrics']['x1_relative_rmse_pct']:.9g}%"
    )
    print(
        "x2 one-step relative RMSE: "
        f"{summary['metrics']['x2_relative_rmse_pct']:.9g}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
