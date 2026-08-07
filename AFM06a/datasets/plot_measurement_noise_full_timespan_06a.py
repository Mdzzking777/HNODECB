"""Plot full-span AFM06a noisy x1 and its finite-difference derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.stage1pluslight.config import default_config
from AFM06a.stage1pluslight.data import load_dataset
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (
    AFM06aHardSampleInputs,
)


def sha256_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            SCRIPT_PATH.parent
            / "noisy"
            / "gaussian_white_measurement_x1"
            / "W1_current_sampling"
        ),
    )
    parser.add_argument(
        "--max-display-points",
        type=int,
        default=180_000,
        help="Maximum plotted points per line; derivatives are still computed at full resolution.",
    )
    return parser.parse_args()


def plot_case(
    *,
    case_root: Path,
    times: np.ndarray,
    x1_true: np.ndarray,
    x2_true: np.ndarray,
    x2dot_true: np.ndarray,
    full_standard_noise: np.ndarray,
    sigma: float,
    level_percent: float,
    drive_period_s: float,
    max_display_points: int,
) -> Path:
    x1_noisy = np.asarray(x1_true + sigma * full_standard_noise, dtype=np.float64)
    x2_from_noisy_x1 = np.gradient(x1_noisy, times, edge_order=2)
    x2dot_from_noisy_x1 = np.gradient(x2_from_noisy_x1, times, edge_order=2)

    del x2_true, x2dot_true
    full_stride = max(1, int(np.ceil(times.size / max_display_points)))
    full_idx = np.arange(0, times.size, full_stride, dtype=np.int64)
    if full_idx[-1] != times.size - 1:
        full_idx = np.append(full_idx, times.size - 1)

    start = float(times[0])
    stop = float(times[-1])
    midpoint = 0.5 * (start + stop)
    window_bounds = (
        (start, min(stop, start + drive_period_s)),
        (
            max(start, midpoint - 0.5 * drive_period_s),
            min(stop, midpoint + 0.5 * drive_period_s),
        ),
        (max(start, stop - drive_period_s), stop),
    )
    detail_indices = tuple(
        np.flatnonzero((times >= left) & (times <= right))
        for left, right in window_bounds
    )
    if any(indices.size < 3 for indices in detail_indices):
        raise RuntimeError("a detailed full-span noise window contains fewer than three points")

    series = (
        x1_noisy * 1.0e9,
        x2_from_noisy_x1 * 1.0e3,
        x2dot_from_noisy_x1 * 1.0e-6,
    )
    labels = (
        r"$x_1$ [nm]",
        r"$x_2$ [mm s$^{-1}$]",
        r"$\dot{x}_2$ [$10^6$ m s$^{-2}$]",
    )
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    column_titles = (
        "Full time span",
        "Detailed window 1, early",
        "Detailed window 2, middle",
        "Detailed window 3, tail",
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(3, 4, figsize=(19.5, 10.2), squeeze=False)
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)

    for row, (values, ylabel, color) in enumerate(zip(series, labels, colors)):
        axes[row, 0].plot(
            times[full_idx] * 1.0e3,
            values[full_idx],
            color=color,
            linewidth=0.45,
            rasterized=True,
        )
        axes[row, 0].set_ylabel(ylabel)
        axes[row, 0].set_xlim(start * 1.0e3, stop * 1.0e3)
        for column, indices in enumerate(detail_indices, start=1):
            axes[row, column].plot(
                times[indices] * 1.0e6,
                values[indices],
                color=color,
                linewidth=0.8,
                rasterized=True,
            )
            axes[row, column].set_xlim(
                float(times[indices[0]] * 1.0e6),
                float(times[indices[-1]] * 1.0e6),
            )
        for column in range(4):
            axes[row, column].grid(
                True,
                color="#d9d9d9",
                linewidth=0.5,
                alpha=0.7,
            )

    axes[-1, 0].set_xlabel("Time [ms]")
    for column in range(1, 4):
        axes[-1, column].set_xlabel(r"Time [$\mu$s]")
    fig.subplots_adjust(
        left=0.065,
        right=0.992,
        bottom=0.07,
        top=0.955,
        wspace=0.22,
        hspace=0.18,
    )

    visualization = case_root / "visualization"
    visualization.mkdir(parents=True, exist_ok=True)
    output = visualization / (
        f"afm06a_noise_{int(round(level_percent)):02d}pct_"
        "x1_x2_x2dot_full_timespan.png"
    )
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    if args.max_display_points < 2:
        raise ValueError("--max-display-points must be at least two")

    config = default_config()
    clean = load_dataset(config.dataset_root, "e0.0")
    times = np.asarray(clean.table["t"], dtype=np.float64)
    x1_true = np.asarray(clean.table["x1"], dtype=np.float64)
    x2_true = np.asarray(clean.table["x2"], dtype=np.float64)
    x2dot_true = np.asarray(clean.table["x2dot"], dtype=np.float64)
    omega0 = float(AFM06aHardSampleInputs().omega0)
    drive_period_s = float(2.0 * np.pi / omega0)

    case_roots = sorted(path.parent for path in args.data_root.glob("noise_*pct/manifest.json"))
    if not case_roots:
        raise FileNotFoundError(f"no generated noisy cases were found under {args.data_root}")

    outputs: list[Path] = []
    for case_root in case_roots:
        manifest = json.loads((case_root / "manifest.json").read_text(encoding="utf-8"))
        noise = manifest["noise"]
        seed = int(noise["seed"])
        sigma = float(noise["sigma_x1_m"])
        level_percent = float(noise["level_percent"])
        rng = np.random.default_rng(seed)
        full_standard_noise = np.asarray(
            rng.standard_normal(times.size),
            dtype=np.float64,
        )
        expected_hash = str(noise["full_span_standard_noise_sha256"])
        if sha256_array(full_standard_noise) != expected_hash:
            raise RuntimeError(f"full-span noise hash mismatch for {case_root}")
        outputs.append(
            plot_case(
                case_root=case_root,
                times=times,
                x1_true=x1_true,
                x2_true=x2_true,
                x2dot_true=x2dot_true,
                full_standard_noise=full_standard_noise,
                sigma=sigma,
                level_percent=level_percent,
                drive_period_s=drive_period_s,
                max_display_points=args.max_display_points,
            )
        )

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
