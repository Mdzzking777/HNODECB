"""Plot true and governing-equation residual F_ts for AFM04."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM04.datasets.non_perturbed_dataset_generator import load_table_npz
from AFM04.stage1pluslight.config import default_config
from AFM04.stage1pluslight.windows import (
    stage2_window_manifest,
)
from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import (
    DEFAULT_SETTINGS,
    original_u0,
)


DATA_PATH = Path(__file__).resolve().parent / "e0.0" / "data" / "pert_df_afm_dmt_kv.npz"
OUTPUT_PATH = Path(__file__).resolve().parent / "afm04_true_vs_residual_fts_full_W0_W1_W3.png"
FULL_DT = 4.0e-9
DETAIL_DT = 0.25e-9


def _rhs_scalar(time: float, state: np.ndarray) -> np.ndarray:
    settings = DEFAULT_SETTINGS
    x1, x2, x3 = (float(value) for value in state)
    separation = settings.dist + x1 - x3
    transition_arg = min(max(settings.beta * (separation - settings.a0), -60.0), 60.0)
    gate = 1.0 / (1.0 + math.exp(-transition_arg))
    denominator = max(gate * (separation - settings.a0) + settings.a0, 1.0e-15)
    adhesion = -(settings.A * settings.R / 6.0) / (denominator**2)
    indentation = max(settings.a0 - separation, 0.0)
    hertz = (
        (4.0 / 3.0)
        * settings.Estar
        * math.sqrt(settings.R)
        * (indentation**1.5)
    )
    kv_coefficient = settings.eta_star * math.sqrt(settings.R) * math.sqrt(indentation)
    static_force = adhesion + (1.0 - gate) * hertz
    x3dot = (
        -static_force + kv_coefficient * x2 - settings.ks * x3
    ) / max(settings.cs + kv_coefficient, 1.0e-15)
    indentation_rate = x3dot - x2 if indentation > 0.0 else 0.0
    true_fts = static_force + kv_coefficient * indentation_rate
    x2dot = (
        settings.Fd * math.cos(settings.wd * time)
        - settings.k * x1
        - settings.c * x2
        + true_fts
    ) / settings.m
    return np.asarray((x2, x2dot, x3dot), dtype=float)


def _integrate_fixed_rk4(
    start_time: float,
    stop_time: float,
    initial_state: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    step_count = int(round((float(stop_time) - float(start_time)) / float(dt)))
    if step_count < 1:
        raise ValueError("integration interval must contain at least one step")
    times = np.linspace(float(start_time), float(stop_time), step_count + 1, dtype=float)
    states = np.empty((step_count + 1, 3), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)
    for index in range(step_count):
        time = float(times[index])
        step = float(times[index + 1] - times[index])
        state = states[index]
        k1 = _rhs_scalar(time, state)
        k2 = _rhs_scalar(time + 0.5 * step, state + 0.5 * step * k1)
        k3 = _rhs_scalar(time + 0.5 * step, state + 0.5 * step * k2)
        k4 = _rhs_scalar(time + step, state + step * k3)
        states[index + 1] = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return times, states


def _force_signals(times: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    settings = DEFAULT_SETTINGS
    x1 = np.asarray(states[:, 0], dtype=float)
    x2 = np.asarray(states[:, 1], dtype=float)
    x3 = np.asarray(states[:, 2], dtype=float)
    separation = settings.dist + x1 - x3
    gate = 1.0 / (
        1.0
        + np.exp(
            -np.clip(settings.beta * (separation - settings.a0), -60.0, 60.0)
        )
    )
    denominator = np.maximum(
        gate * (separation - settings.a0) + settings.a0,
        1.0e-15,
    )
    adhesion = -(settings.A * settings.R / 6.0) / np.square(denominator)
    indentation = np.maximum(settings.a0 - separation, 0.0)
    hertz = (
        (4.0 / 3.0)
        * settings.Estar
        * math.sqrt(settings.R)
        * np.power(indentation, 1.5)
    )
    kv_coefficient = (
        settings.eta_star * math.sqrt(settings.R) * np.sqrt(indentation)
    )
    static_force = adhesion + (1.0 - gate) * hertz
    x3dot = (
        -static_force + kv_coefficient * x2 - settings.ks * x3
    ) / np.maximum(settings.cs + kv_coefficient, 1.0e-15)
    indentation_rate = np.where(indentation > 0.0, x3dot - x2, 0.0)
    true_fts = static_force + kv_coefficient * indentation_rate
    x2dot = (
        settings.Fd * np.cos(settings.wd * times)
        - settings.k * x1
        - settings.c * x2
        + true_fts
    ) / settings.m
    residual_fts = (
        settings.m * x2dot
        + settings.c * x2
        + settings.k * x1
        - settings.Fd * np.cos(settings.wd * times)
    )
    return true_fts, residual_fts


def _formal_window_bounds(
    times: np.ndarray,
    contact: np.ndarray,
    x1: np.ndarray,
) -> dict[str, tuple[float, float]]:
    windows = {window.role: window for window in stage2_window_manifest(times, contact, x1)}
    return {
        "W1": (float(windows["middle"].t_start), float(windows["middle"].t_stop)),
        "W3": (
            float(windows["tail_stable"].t_start),
            float(windows["tail_stable"].t_stop),
        ),
    }


def _state_at_time(
    full_times: np.ndarray,
    full_states: np.ndarray,
    target_time: float,
) -> np.ndarray:
    index = int(np.argmin(np.abs(full_times - float(target_time))))
    if abs(float(full_times[index]) - float(target_time)) > 0.51 * FULL_DT:
        raise RuntimeError(f"fine full trajectory does not cover target time {target_time:.12e}")
    return np.asarray(full_states[index], dtype=float).copy()


def _build_fine_windows(
    full_times: np.ndarray,
    full_states: np.ndarray,
    formal_bounds: dict[str, tuple[float, float]],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    settings = DEFAULT_SETTINGS
    separation = settings.dist + full_states[:, 0] - full_states[:, 2]
    contact_locations = np.flatnonzero(separation <= settings.a0)
    if contact_locations.size == 0:
        raise RuntimeError("fine AFM04 full trajectory contains no contact point")

    coarse_contact = int(contact_locations[0])
    precontact_index = max(0, coarse_contact - 2)
    w0_refine_start = float(full_times[precontact_index])
    w0_refine_stop = (
        float(full_times[coarse_contact])
        + float(default_config(REPO_ROOT).arch_window_us)
        + 3.0 * FULL_DT
    )
    w0_times_all, w0_states_all = _integrate_fixed_rk4(
        w0_refine_start,
        w0_refine_stop,
        full_states[precontact_index],
        DETAIL_DT,
    )
    w0_separation = settings.dist + w0_states_all[:, 0] - w0_states_all[:, 2]
    w0_contacts = np.flatnonzero(w0_separation <= settings.a0)
    if w0_contacts.size == 0:
        raise RuntimeError("fine W0 refinement contains no contact point")
    w0_start_index = int(w0_contacts[0])
    w0_stop_time = (
        float(w0_times_all[w0_start_index])
        + float(default_config(REPO_ROOT).arch_window_us)
    )
    w0_stop_index = int(np.searchsorted(w0_times_all, w0_stop_time, side="right") - 1)
    windows: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "W0: first contact",
            w0_times_all[w0_start_index : w0_stop_index + 1],
            w0_states_all[w0_start_index : w0_stop_index + 1],
        )
    ]

    for label in ("W1", "W3"):
        start_time, stop_time = formal_bounds[label]
        local_times, local_states = _integrate_fixed_rk4(
            start_time,
            stop_time,
            _state_at_time(full_times, full_states, start_time),
            DETAIL_DT,
        )
        description = "middle" if label == "W1" else "tail stable"
        windows.append((f"{label}: {description}", local_times, local_states))
    return windows


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.8)
    ax.tick_params(axis="both", labelsize=10.5, direction="out", length=3.5, width=0.75)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"AFM04 dataset not found: {DATA_PATH}")

    table = load_table_npz(str(DATA_PATH))
    saved_times = np.asarray(table["t"], dtype=float)
    saved_x1 = np.asarray(table["x1"], dtype=float)
    saved_contact = np.asarray(table["contact"], dtype=bool)
    formal_bounds = _formal_window_bounds(saved_times, saved_contact, saved_x1)

    full_times, full_states = _integrate_fixed_rk4(
        float(saved_times[0]),
        float(saved_times[-1]),
        np.asarray(original_u0, dtype=float),
        FULL_DT,
    )
    fine_windows = _build_fine_windows(full_times, full_states, formal_bounds)
    full_true, full_residual = _force_signals(full_times, full_states)

    columns: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("Full time span", full_times, full_true, full_residual)
    ]
    for title, local_times, local_states in fine_windows:
        local_true, local_residual = _force_signals(local_times, local_states)
        columns.append((title, local_times, local_true, local_residual))

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18.5, 7.2),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.28, 1.0, 1.0, 1.0]},
    )

    for col, (title, column_times, true_fts, residual_fts) in enumerate(columns):
        if col == 0:
            t_plot = column_times * 1.0e3
            xlabel = "Time [ms]"
            interval = f"{column_times[0] * 1.0e3:.3f}-{column_times[-1] * 1.0e3:.3f} ms"
        else:
            t_plot = column_times * 1.0e6
            xlabel = "Time [us]"
            interval = f"{column_times[0] * 1.0e6:.3f}-{column_times[-1] * 1.0e6:.3f} us"
        axes[0, col].set_title(f"{title}\n{interval}", fontsize=12.5, pad=8)

        rows = [
            (r"True $F_{ts}$ [nN]", true_fts * 1.0e9, "#111111"),
            (r"Residual $F_{ts}$ [nN]", residual_fts * 1.0e9, "#d62728"),
        ]
        column_values = [values for _, values, _ in rows]
        y_all = np.concatenate(column_values)
        y_span = float(np.ptp(y_all))
        y_pad = 0.06 * y_span if y_span > 0.0 else 1.0
        y_limits = (float(np.min(y_all) - y_pad), float(np.max(y_all) + y_pad))

        for row, (ylabel, values, color) in enumerate(rows):
            ax = axes[row, col]
            ax.plot(t_plot, values, color=color, linewidth=0.72 if col == 0 else 1.05)
            ax.set_xlim(float(t_plot[0]), float(t_plot[-1]))
            ax.set_ylim(*y_limits)
            ax.set_xlabel(xlabel, fontsize=11.5)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=12.5)
            _style_axis(ax)

    fig.subplots_adjust(left=0.068, right=0.992, top=0.88, bottom=0.095, wspace=0.25, hspace=0.30)
    fig.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    all_errors = [full_residual - full_true]
    for _, local_times, local_states in fine_windows:
        local_true, local_residual = _force_signals(local_times, local_states)
        all_errors.append(local_residual - local_true)
    error = np.concatenate(all_errors)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Source: {DATA_PATH}")
    print(f"Fine full time-span dt [ns]: {FULL_DT * 1.0e9:.6f}")
    print(f"Fine W0/W1/W3 dt [ns]: {DETAIL_DT * 1.0e9:.6f}")
    print(f"Maximum absolute residual mismatch [nN]: {np.max(np.abs(error)) * 1.0e9:.12e}")
    print(f"Residual mismatch RMSE [nN]: {np.sqrt(np.mean(error**2)) * 1.0e9:.12e}")


if __name__ == "__main__":
    main()
