"""Plot x1 full time-span rollout from W0 using the archived AFM06a cache.

This mirrors afm_param_stage2light_06a_bar_fts_full_rollout_from_W0_grid.png
and changes only the plotted quantity from bar_Fts to x1.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = SCRIPT_DIR.parent
CACHE_PATH = SCRIPT_DIR / "afm06a_stage2light_full_rollout_from_W0_cache_rank1.pkl"
STAGE1_PATH = ARCHIVE_ROOT / "conditional dependency" / "afm_param_stage1pluslight_06a.pkl"
DATA_PATH = ARCHIVE_ROOT / "data" / "pert_df_afm_dmt_hard.npz"
OUTPUT = SCRIPT_DIR / "afm_param_stage2light_06a_x1_full_rollout_from_W0_grid.png"


def relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(truth, dtype=float)
    return 100.0 * float(np.sqrt(np.mean(np.square(pred - ref)))) / max(
        float(np.sqrt(np.mean(np.square(ref)))), 1.0e-30
    )


def training_style_panel_subset(
    *,
    indices: np.ndarray,
    target_count: int,
    payload_times: np.ndarray,
    contact: np.ndarray,
    config: dict,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if indices.size <= target_count:
        return indices

    times = payload_times[indices]
    local_contact = contact[indices]
    weights = np.where(
        local_contact,
        float(config["contact_sampling_weight"]),
        float(config["noncontact_sampling_weight"]),
    )
    half_width = float(config["transition_half_width_s"])
    switch_idx = np.flatnonzero(local_contact[1:] != local_contact[:-1]) + 1
    for switch in switch_idx:
        boundary = 0.5 * (times[switch - 1] + times[switch])
        weights[np.abs(times - boundary) <= half_width] = float(
            config["transition_sampling_weight"]
        )

    cell_width = np.empty_like(times)
    cell_width[1:-1] = 0.5 * (times[2:] - times[:-2])
    cell_width[0] = 0.5 * (times[1] - times[0])
    cell_width[-1] = 0.5 * (times[-1] - times[-2])
    cumulative = np.cumsum(weights * cell_width)
    targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), target_count)

    positions: list[int] = []
    used = np.zeros(indices.size, dtype=bool)
    for target in targets:
        insertion = int(np.searchsorted(cumulative, target, side="left"))
        insertion = min(max(insertion, 0), indices.size - 1)
        left = insertion
        right = insertion + 1
        while True:
            candidates: list[int] = []
            if left >= 0 and not used[left]:
                candidates.append(left)
            if right < indices.size and not used[right]:
                candidates.append(right)
            if candidates:
                selected = min(
                    candidates,
                    key=lambda pos: abs(float(cumulative[pos]) - float(target)),
                )
                used[selected] = True
                positions.append(selected)
                break
            left -= 1
            right += 1

    return indices[np.asarray(sorted(positions), dtype=int)]


def full_rollout_panels_from_w0(
    payload_times: np.ndarray,
    window_times: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    full = np.arange(payload_times.size, dtype=int)
    window_duration = float(window_times[-1] - window_times[0])
    training = np.flatnonzero(payload_times <= float(payload_times[0]) + window_duration)
    middle_time = 0.5 * (float(payload_times[0]) + float(payload_times[-1]))
    middle = np.flatnonzero(np.abs(payload_times - middle_time) <= 0.5 * window_duration)
    tail = np.flatnonzero(payload_times >= float(payload_times[-1]) - window_duration)
    return [
        ("Full time-span", full),
        ("Initial window, W0", training),
        ("Observation window, middle", middle),
        ("Observation window, tail", tail),
    ]


def main() -> int:
    with CACHE_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    with STAGE1_PATH.open("rb") as handle:
        stage1 = pickle.load(handle)
    with np.load(DATA_PATH, allow_pickle=True) as data:
        raw_times_ms = 1.0e3 * np.asarray(data["t"], dtype=float)
        raw_x1_nm = 1.0e9 * np.asarray(data["x1"], dtype=float)

    config = stage1["config"]
    window_times = np.asarray(stage1["times"], dtype=float)
    window_count = int(window_times.size)
    payload_times = np.asarray(payload["times"], dtype=float)
    payload_times_ms = 1.0e3 * payload_times
    contact = np.asarray(payload["true_contact"], dtype=bool)
    true_x1 = 1.0e9 * np.asarray(payload["true_states"], dtype=float)[0]
    pred_x1 = 1.0e9 * np.asarray(payload["predicted_states"], dtype=float)[0]
    panels = full_rollout_panels_from_w0(payload_times, window_times)

    panel_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    window_duration = float(window_times[-1] - window_times[0])
    for name, indices in panels:
        panel_indices = np.asarray(indices, dtype=int)
        if name.startswith("Observation window") or name == "Initial window, W0":
            target_count = window_count
        else:
            panel_duration = float(
                payload_times[panel_indices[-1]] - payload_times[panel_indices[0]]
            )
            target_count = max(2, int(round(window_count * panel_duration / window_duration)))
        panel_indices = training_style_panel_subset(
            indices=panel_indices,
            target_count=target_count,
            payload_times=payload_times,
            contact=contact,
            config=config,
        )
        panel_series[name] = (
            payload_times_ms[panel_indices],
            true_x1[panel_indices],
            pred_x1[panel_indices],
        )

    full_name = "Full time-span"
    full_times, full_truth, full_prediction = panel_series[full_name]
    for detail_name, _ in panels[1:]:
        detail_times, detail_truth, detail_prediction = panel_series[detail_name]
        outside_detail = (full_times < detail_times[0]) | (full_times > detail_times[-1])
        full_times = np.concatenate((full_times[outside_detail], detail_times))
        full_truth = np.concatenate((full_truth[outside_detail], detail_truth))
        full_prediction = np.concatenate((full_prediction[outside_detail], detail_prediction))
        order = np.argsort(full_times)
        full_times = full_times[order]
        full_truth = full_truth[order]
        full_prediction = full_prediction[order]
    panel_series[full_name] = (full_times, full_truth, full_prediction)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for ax, (name, _) in zip(axes.ravel(), panels, strict=True):
        panel_times, panel_truth, panel_prediction = panel_series[name]
        in_panel = (raw_times_ms >= panel_times[0]) & (raw_times_ms <= panel_times[-1])
        ax.plot(
            raw_times_ms[in_panel],
            raw_x1_nm[in_panel],
            color="black",
            linewidth=1.8,
            label="Original full-resolution data",
        )
        ax.plot(
            panel_times,
            panel_prediction,
            color="crimson",
            linestyle="-",
            linewidth=1.0,
            label="Rollout prediction",
        )
        ax.text(
            0.02,
            0.95,
            f"relative RMSE = {relative_rmse_pct(panel_prediction, panel_truth):.3f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        ax.set_title(name)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(r"$x_1$ (nm)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
