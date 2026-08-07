"""Create archived AFM06a diagnostics from the W1 initial time to 10.05 ms."""

from __future__ import annotations

import sys
import math
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, ConnectionPatch


SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = SCRIPT_DIR.parents[1]
for parent in SCRIPT_DIR.parents:
    if (parent / "AFM06a" / "stage2light").is_dir():
        REPO_ROOT = parent
        break
else:
    raise RuntimeError(f"Could not locate HNODECB root from {SCRIPT_DIR}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.stage2light.config import default_config
from AFM06a.stage2light.runner.visualization._common import (
    _build_full_rollout_payload,
    _pointwise_full_resolution_payload,
    load_visualization_context,
)


END_TIME_S = 10.05e-3
POINTWISE_OUTPUT = (
    SCRIPT_DIR
    / "afm_param_stage2light_06a_bar_fts_pointwise_original_full_resolution_single_to_10p05ms.png"
)
STATE_SPACE_OUTPUT = (
    SCRIPT_DIR
    / "afm_param_stage2light_06a_state_space_x1_x2_time_full_rollout_from_t0_to_10p05ms.png"
)
ROLLOUT_CACHE_NAME = "afm06a_stage2light_rollout_initial_to_10p05ms_cache_rank1.pkl"
ARCHIVED_FULL_RESOLUTION_DATA = ARCHIVE_ROOT / "data" / "pert_df_afm_dmt_hard.npz"


def _context():
    config = replace(
        default_config(REPO_ROOT),
        stage1_result_path=(
            ARCHIVE_ROOT
            / "conditional dependency"
            / "afm_param_stage1pluslight_06a.pkl"
        ),
        result_dir=ARCHIVE_ROOT / "result",
        checkpoint_dir=ARCHIVE_ROOT / "checkpoint",
        log_dir=ARCHIVE_ROOT / "logs",
        visualization_dir=SCRIPT_DIR,
    )
    return load_visualization_context(
        config,
        preserve_io_paths=True,
        enforce_current_contract=False,
    )


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    denominator = max(float(np.sqrt(np.mean(np.asarray(truth) ** 2))), 1.0e-30)
    return 100.0 * float(np.sqrt(np.mean(error**2))) / denominator


def _third_contact_transition_times_ms(
    times_ms: np.ndarray,
    contact: np.ndarray,
) -> tuple[float, float]:
    transitions = np.flatnonzero(np.diff(contact.astype(np.int8)) != 0) + 1
    transitions = transitions[
        (times_ms[transitions] >= 10.0)
        & (times_ms[transitions] <= END_TIME_S * 1.0e3)
    ]
    if transitions.size < 6:
        raise RuntimeError("The archived interval does not contain three contact cycles")
    return float(times_ms[transitions[4]]), float(times_ms[transitions[5]])


def _dense_true_force_4ns(context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_time_s = float(context.endpoint.window.times[0])
    dt_s = 4.0e-9
    point_count = int(round((END_TIME_S - start_time_s) / dt_s)) + 1
    times_s = start_time_s + dt_s * np.arange(point_count, dtype=float)
    times_s[-1] = END_TIME_S

    with np.load(ARCHIVED_FULL_RESOLUTION_DATA, allow_pickle=True) as data:
        archived_times = np.asarray(data["t"], dtype=float)
        archived_x1 = np.asarray(data["x1"], dtype=float)
        archived_x2 = np.asarray(data["x2"], dtype=float)
    initial_index = int(np.argmin(np.abs(archived_times - start_time_s)))
    if abs(float(archived_times[initial_index]) - start_time_s) > 1.0e-15:
        raise RuntimeError("Archived data does not contain the exact W1 initial time")

    settings = context.endpoint.window.settings
    mass = float(settings.mass_kg)
    stiffness = float(settings.k_n_m)
    damping = float(settings.c_n_s_m)
    drive_force = float(settings.fd_n)
    drive_frequency = float(settings.omega0)
    dist = float(settings.dist)
    a0 = float(settings.a0)
    ca_force = mass * float(settings.ca)
    ch_force = mass * float(settings.ch)

    def force_from_x1(x1: float) -> float:
        separation = dist + float(x1)
        if separation <= a0:
            indentation = max(a0 - separation, 0.0)
            return ca_force / (a0**2) + ch_force * (indentation**1.5)
        return ca_force / (max(separation, 1.0e-15) ** 2)

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        x1, x2 = float(state[0]), float(state[1])
        x2dot = (
            drive_force * math.sin(drive_frequency * time_s)
            - stiffness * x1
            - damping * x2
            + force_from_x1(x1)
        ) / mass
        return np.asarray((x2, x2dot), dtype=float)

    states = np.empty((2, point_count), dtype=float)
    states[:, 0] = (archived_x1[initial_index], archived_x2[initial_index])
    for index in range(point_count - 1):
        time_s = float(times_s[index])
        step_s = float(times_s[index + 1] - times_s[index])
        state = states[:, index]
        k1 = rhs(time_s, state)
        k2 = rhs(time_s + 0.5 * step_s, state + 0.5 * step_s * k1)
        k3 = rhs(time_s + 0.5 * step_s, state + 0.5 * step_s * k2)
        k4 = rhs(time_s + step_s, state + step_s * k3)
        states[:, index + 1] = state + (step_s / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

    separation = dist + states[0]
    contact = separation <= a0
    indentation = np.maximum(a0 - separation, 0.0)
    denominator = np.where(contact, a0, np.maximum(separation, 1.0e-15))
    force_n = ca_force / denominator**2
    force_n = force_n + np.where(contact, ch_force * indentation**1.5, 0.0)
    return times_s * 1.0e3, force_n * 1.0e9, contact


def _add_transition_inset(
    axis,
    truth_time_ms: np.ndarray,
    truth_nN: np.ndarray,
    prediction_time_ms: np.ndarray,
    prediction_nN: np.ndarray,
    *,
    transition_ms: float,
    bounds: tuple[float, float, float, float],
) -> None:
    half_width_ms = 0.000096
    truth_selected = np.abs(truth_time_ms - transition_ms) <= half_width_ms
    prediction_selected = np.abs(prediction_time_ms - transition_ms) <= half_width_ms
    if np.count_nonzero(truth_selected) < 3 or np.count_nonzero(prediction_selected) < 3:
        raise RuntimeError(f"Insufficient full-resolution points near {transition_ms:.9f} ms")

    local_truth = truth_nN[truth_selected]
    local_prediction = prediction_nN[prediction_selected]
    local_minimum = float(np.min(np.concatenate((local_truth, local_prediction))))
    negative_scale = max(abs(min(local_minimum, 0.0)), 5.0e-4)
    y_min = local_minimum - 0.35 * negative_scale
    y_max = max(3.0e-3, 4.0 * negative_scale)

    inset = axis.inset_axes(bounds, zorder=8)
    inset.patch.set_alpha(0.0)
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_xlim(transition_ms - half_width_ms, transition_ms + half_width_ms)
    inset.set_ylim(y_min, y_max)

    circle = Circle(
        (0.5, 0.5),
        0.495,
        transform=inset.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.35,
        zorder=0,
    )
    inset.add_patch(circle)
    true_line, = inset.plot(
        truth_time_ms,
        truth_nN,
        color="#3168ff",
        linewidth=1.6,
        zorder=2,
    )
    prediction_line, = inset.plot(
        prediction_time_ms,
        prediction_nN,
        color="#ff8c00",
        linestyle="--",
        linewidth=1.4,
        zorder=3,
    )
    true_line.set_clip_path(circle)
    prediction_line.set_clip_path(circle)

    connector = ConnectionPatch(
        xyA=(transition_ms, -0.012),
        coordsA=axis.transData,
        xyB=(0.5, 0.01),
        coordsB=inset.transAxes,
        color="black",
        linewidth=0.9,
        zorder=7,
    )
    axis.add_artist(connector)


def _plot_pointwise(context) -> None:
    payload = _pointwise_full_resolution_payload(
        context,
        start_source_idx=int(context.endpoint.window.source_idx[0]),
        force_input_source="original_full_resolution_true_x1_pointwise_from_W1_t0",
        end_time_s=END_TIME_S,
    )
    time_ms = np.asarray(payload["times"], dtype=float) * 1.0e3
    prediction_nN = np.asarray(payload["predicted_fts"], dtype=float) * 1.0e9
    truth_time_ms, truth_nN, truth_contact = _dense_true_force_4ns(context)
    truth_at_prediction_times = np.interp(time_ms, truth_time_ms, truth_nN)
    relative_rmse = _relative_rmse_pct(prediction_nN, truth_at_prediction_times)

    fig, axis = plt.subplots(figsize=(8.11, 6.0))
    axis.plot(
        truth_time_ms,
        truth_nN,
        color="#3168ff",
        linewidth=1.6,
        label="simulated true trajectory",
        zorder=2,
    )
    axis.plot(
        time_ms,
        prediction_nN,
        color="#ff8c00",
        linestyle="--",
        linewidth=1.4,
        label="prediction",
        zorder=3,
    )
    axis.text(
        0.01,
        0.98,
        f"relative RMSE = {relative_rmse:.3f}%",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=15,
    )
    axis.set_xlim(float(time_ms[0]), END_TIME_S * 1.0e3)
    axis.set_xlabel("time (ms)", fontsize=18)
    axis.set_ylabel(r"$F_{ts}$ (nN)", fontsize=18)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", fontsize=15)
    transition_in, transition_out = _third_contact_transition_times_ms(
        truth_time_ms,
        truth_contact,
    )
    _add_transition_inset(
        axis,
        truth_time_ms,
        truth_nN,
        time_ms,
        prediction_nN,
        transition_ms=transition_in,
        bounds=(0.13, 0.49, 0.20, 0.27),
    )
    _add_transition_inset(
        axis,
        truth_time_ms,
        truth_nN,
        time_ms,
        prediction_nN,
        transition_ms=transition_out,
        bounds=(0.43, 0.49, 0.20, 0.27),
    )
    fig.tight_layout()
    fig.savefig(POINTWISE_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _render_indices(point_count: int, maximum: int = 25_000) -> np.ndarray:
    stride = max(1, int(np.ceil(point_count / maximum)))
    indices = np.arange(0, point_count, stride, dtype=int)
    if indices[-1] != point_count - 1:
        indices = np.append(indices, point_count - 1)
    return indices


def _plot_state_space(context) -> None:
    cache_path = SCRIPT_DIR / ROLLOUT_CACHE_NAME
    payload = _build_full_rollout_payload(
        context,
        start_source_idx=int(context.endpoint.window.source_idx[0]),
        cache_filename=ROLLOUT_CACHE_NAME,
        initial_state_source="sampled_true_data_at_W1_t0",
        end_time_s=END_TIME_S,
    )
    times_ms = np.asarray(payload["times"], dtype=float) * 1.0e3
    true_states = np.asarray(payload["true_states"], dtype=float)
    predicted_states = np.asarray(payload["predicted_states"], dtype=float)
    true_idx = _render_indices(times_ms.size)
    pred_idx = _render_indices(times_ms.size)

    fig = plt.figure(figsize=(10.5, 8.2))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(
        true_states[0, true_idx] * 1.0e9,
        true_states[1, true_idx] * 1.0e6,
        times_ms[true_idx],
        color="royalblue",
        linewidth=0.9,
        alpha=0.9,
        label="simulated true trajectory",
        zorder=2,
    )
    axis.plot(
        predicted_states[0, pred_idx] * 1.0e9,
        predicted_states[1, pred_idx] * 1.0e6,
        times_ms[pred_idx],
        color="darkorange",
        linestyle="--",
        linewidth=1.35,
        alpha=0.95,
        label="prediction",
        zorder=3,
    )
    axis.set_xlabel(r"Tip displacement, $x_1$ [nm]", labelpad=11, fontsize=15)
    axis.set_ylabel(
        r"Tip velocity, $x_2$ [$\mu$m s$^{-1}$]",
        labelpad=13,
        fontsize=15,
    )
    axis.set_zlabel("")
    axis.set_zlim(float(times_ms[0]), END_TIME_S * 1.0e3)
    axis.view_init(elev=23, azim=-56)
    axis.tick_params(axis="x", labelsize=15)
    axis.tick_params(axis="y", labelsize=15)
    axis.tick_params(axis="z", labelsize=15)
    axis.grid(True, alpha=0.22)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.90),
        frameon=False,
        fontsize=15,
    )
    fig.text(
        0.885,
        0.52,
        "Time [ms]",
        rotation=90,
        ha="center",
        va="center",
        fontsize=18,
    )
    fig.subplots_adjust(left=0.01, right=0.94, bottom=0.03, top=0.99)
    fig.savefig(STATE_SPACE_OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    cache_path.unlink(missing_ok=True)


def main() -> None:
    context = _context()
    _plot_pointwise(context)
    _plot_state_space(context)
    print(f"Saved: {POINTWISE_OUTPUT}")
    print(f"Saved: {STATE_SPACE_OUTPUT}")


if __name__ == "__main__":
    main()
