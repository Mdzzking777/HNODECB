"""Plot the observable trajectories for one ranked AFM06a st1pl trial."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_repo_root(start: str | Path) -> Path:
    here = Path(start).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "AFM06a" / "stage1pluslight").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the HNODECB repository from {start}")


REPO_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(REPO_ROOT))

from AFM06a.stage1pluslight.checkpoint import load_checkpoint  # noqa: E402
from AFM06a.stage1pluslight.config import default_config  # noqa: E402
from AFM06a.stage1pluslight.data import load_dataset  # noqa: E402
from AFM06a.stage1pluslight.ranking import rank_trial_records  # noqa: E402
from AFM06a.stage1pluslight.result_validation import validate_complete_stage1plus_payload  # noqa: E402


DEFAULT_RESULT = (
    REPO_ROOT
    / "AFM06a"
    / "stage1pluslight"
    / "results_afm"
    / "afm_param_stage1pluslight_06a.pkl"
)
DEFAULT_OUT_DIR = REPO_ROOT / "AFM06a" / "stage1pluslight" / "visualization"


def _finite_viable_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("trial_parameters", [])
    if not isinstance(records, list):
        raise RuntimeError("AFM06a st1pl payload has no trial_parameters list")
    selected = [
        record
        for record in records
        if isinstance(record, dict)
        and bool(record.get("is_viable", False))
        and not bool(record.get("trial_failed", False))
        and np.isfinite(float(record.get("loss", np.inf)))
    ]
    if not selected:
        raise RuntimeError("AFM06a st1pl payload has no finite viable trials")
    return rank_trial_records(selected)


def _relative_rmse_pct(prediction: np.ndarray, observation: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    obs = np.asarray(observation, dtype=float).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(obs)
    if not np.any(valid):
        return float("nan")
    denominator = float(np.sqrt(np.mean(np.square(obs[valid]))))
    numerator = float(np.sqrt(np.mean(np.square(pred[valid] - obs[valid]))))
    return 100.0 * numerator / max(denominator, np.finfo(float).tiny)


def _load_observations(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_idx = np.asarray(payload.get("window_source_indices", []), dtype=int)
    saved_times = np.asarray(payload.get("times", []), dtype=float)
    if source_idx.ndim != 1 or source_idx.size == 0 or saved_times.shape != source_idx.shape:
        raise RuntimeError("AFM06a st1pl payload has invalid training-window source indices/times")

    config = default_config(REPO_ROOT)
    loaded = load_dataset(config.dataset_root, config.error_level)
    if np.min(source_idx) < 0 or np.max(source_idx) >= loaded.states.shape[1]:
        raise RuntimeError("AFM06a st1pl training-window source indices are outside the dataset")
    times = np.asarray(loaded.table["t"], dtype=float)[source_idx]
    if not np.allclose(times, saved_times, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("AFM06a st1pl saved times do not match the source dataset")
    states = np.asarray(loaded.states[:, source_idx], dtype=float)
    x2dot = np.asarray(loaded.table["x2dot"], dtype=float)[source_idx]
    return times, states, x2dot


def plot_rank(
    *,
    rank: int = 1,
    result_path: Path = DEFAULT_RESULT,
    out_dir: Path = DEFAULT_OUT_DIR,
    dpi: int = 240,
) -> Path:
    if rank < 1:
        raise ValueError("rank must be at least 1")
    payload = load_checkpoint(result_path)
    if payload is None:
        raise FileNotFoundError(f"AFM06a st1pl merged result was not found: {result_path}")
    validate_complete_stage1plus_payload(payload, path=result_path)
    ranked = _finite_viable_records(payload)
    if rank > len(ranked):
        raise IndexError(f"rank {rank} is outside the viable range 1..{len(ranked)}")
    record = ranked[rank - 1]

    times, observed_states, observed_x2dot = _load_observations(payload)
    has_rollout = "predicted_states" in record and "predicted_x2dot" in record
    if has_rollout:
        predicted_states = np.asarray(record.get("predicted_states"), dtype=float)
        predicted_x2dot = np.asarray(record.get("predicted_x2dot"), dtype=float)
        if predicted_states.shape != observed_states.shape:
            raise RuntimeError(
                f"rank {rank} predicted state shape {predicted_states.shape} != observed {observed_states.shape}"
            )
        if predicted_x2dot.shape != observed_x2dot.shape:
            raise RuntimeError(
                f"rank {rank} predicted x2dot shape {predicted_x2dot.shape} != observed {observed_x2dot.shape}"
            )
    else:
        predicted_states = np.full_like(observed_states, np.nan, dtype=float)
        predicted_x2dot = np.full_like(observed_x2dot, np.nan, dtype=float)

    time_us = 1.0e6 * times
    series = [
        (
            r"$x_1$ [nm]",
            1.0e9 * observed_states[0],
            1.0e9 * predicted_states[0],
            r"$x_1$",
        ),
        (
            r"$x_2$ [$\mu$m s$^{-1}$]",
            1.0e6 * observed_states[1],
            1.0e6 * predicted_states[1],
            r"$x_2$",
        ),
        (
            r"$\dot{x}_2$ [m s$^{-2}$]",
            observed_x2dot,
            predicted_x2dot,
            r"$\dot{x}_2$",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.2), sharex=True)
    for ax, (ylabel, observed, predicted, symbol) in zip(axes, series, strict=True):
        error = _relative_rmse_pct(predicted, observed)
        ax.plot(time_us, observed, color="black", linewidth=1.8, label=f"{symbol} observed", zorder=2)
        ax.plot(
            time_us,
            predicted,
            color="crimson",
            linewidth=1.7,
            linestyle="--",
            label=f"{symbol} predicted (relative RMSE = {error:.6f}%)",
            zorder=3,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=True)

    params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
    trial_id = int(params.get("trial_id", -1))
    seed_index = int(params.get("nn_seed_bank_idx", -1))
    loss = float(record.get("loss", np.nan))
    axes[-1].set_xlabel(r"Time [$\mu$s]")
    fig.suptitle(
        f"AFM06a stage1pluslight Rank {rank} | trial {trial_id} | seed-bank {seed_index} | "
        f"loss = {loss:.9e} | rollout = {'available' if has_rollout else 'not performed'}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"afm_param_stage1pluslight_06a_rank{rank}_x1_x2_x2dot.png"
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=1, help="One-based viable trial rank (default: 1)")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT, help="Merged AFM06a st1pl checkpoint")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--dpi", type=int, default=240, help="PNG resolution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = plot_rank(
        rank=int(args.rank),
        result_path=args.result.resolve(),
        out_dir=args.out_dir.resolve(),
        dpi=int(args.dpi),
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
