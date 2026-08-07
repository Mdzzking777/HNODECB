"""Create standard AFM06a e0.0_real visualizations with physical Fts units."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.stage1pluslight.data import (  # noqa: E402
    build_regime_sampling_weights,
    make_train_val_masks,
)


ERROR_LEVEL = "e0.0_real"
WINDOW_RAW_POINTS_INCLUSIVE = 834
SAMPLE_COUNT = 394
VAL_STRIDE = 5
VAL_OFFSET = 2
TRANSITION_WEIGHT = 10.0
BASE_CONTACT_WEIGHT = 5.0
CONTACT_WEIGHT = 10.0
NONCONTACT_WEIGHT = 1.0
TRANSITION_HALF_WIDTH_S = 1.3565187713310766e-7
W1_START_S = 10.0e-3
FIG_DPI = 260


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_table(data_dir: Path) -> dict[str, np.ndarray]:
    with np.load(data_dir / "pert_df_afm_dmt_hard.npz", allow_pickle=True) as payload:
        columns = [str(x) for x in payload["__columns__"].tolist()]
        return {name: np.asarray(payload[name]) for name in columns}


def _load_manifest(data_dir: Path) -> dict[str, Any]:
    return json.loads((data_dir / "afm06a_generation_manifest.json").read_text(encoding="utf-8"))


def _transition_times(times: np.ndarray, contact: np.ndarray) -> np.ndarray:
    contact_bool = np.asarray(contact, dtype=bool)
    switch_idx = np.flatnonzero(contact_bool[1:] != contact_bool[:-1]) + 1
    if switch_idx.size == 0:
        return np.empty(0, dtype=float)
    return 0.5 * (times[switch_idx - 1] + times[switch_idx])


def _regime_masks(
    times: np.ndarray,
    contact: np.ndarray,
    *,
    transition_half_width_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    contact = np.asarray(contact, dtype=bool)
    transition = np.zeros(times.shape, dtype=bool)
    switch_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1
    for index in switch_idx:
        boundary_time = 0.5 * (float(times[index - 1]) + float(times[index]))
        transition |= np.abs(times - boundary_time) <= float(transition_half_width_s)
    contact_interior = contact & ~transition
    noncontact_interior = ~contact & ~transition
    return transition, contact_interior, noncontact_interior


def _fixed_count_weighted_positions(density: np.ndarray, target_count: int) -> np.ndarray:
    cumulative = np.cumsum(np.asarray(density, dtype=float))
    targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), int(target_count))
    positions = np.searchsorted(cumulative, targets, side="left").astype(int)
    positions[0] = 0
    positions[-1] = density.size - 1
    for i in range(1, positions.size):
        positions[i] = max(positions[i], positions[i - 1] + 1)
    for i in range(positions.size - 2, -1, -1):
        positions[i] = min(positions[i], positions[i + 1] - 1)
    if positions[0] < 0 or positions[-1] >= density.size or np.any(np.diff(positions) <= 0):
        raise RuntimeError("failed to construct a strictly increasing weighted sample grid")
    return positions


def _even_subset(candidates: np.ndarray, count: int) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=int)
    if count <= 0:
        return np.empty(0, dtype=int)
    if count > candidates.size:
        return candidates.copy()
    positions = np.floor(np.arange(count, dtype=float) * candidates.size / count).astype(int)
    return candidates[positions]


def _runs(indices: np.ndarray) -> list[np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    return [part for part in np.split(indices, breaks) if part.size]


def _allocate_counts(sizes: np.ndarray, total: int) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=float)
    if total <= 0 or sizes.size == 0:
        return np.zeros(sizes.shape, dtype=int)
    raw = total * sizes / float(np.sum(sizes))
    counts = np.floor(raw).astype(int)
    remainder = int(total - np.sum(counts))
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def _interpolated_contact_times(
    local_times: np.ndarray,
    contact_mask: np.ndarray,
    extra_count: int,
) -> np.ndarray:
    contact_runs = _runs(np.flatnonzero(contact_mask))
    if extra_count <= 0 or not contact_runs:
        return np.empty(0, dtype=float)
    sizes = np.asarray([run.size for run in contact_runs], dtype=int)
    counts = _allocate_counts(sizes, extra_count)
    pieces: list[np.ndarray] = []
    for run, count in zip(contact_runs, counts, strict=True):
        if count <= 0:
            continue
        left = float(local_times[run[0]])
        right = float(local_times[run[-1]])
        if right <= left:
            continue
        pieces.append(np.linspace(left, right, count + 2, dtype=float)[1:-1])
    if not pieces:
        return np.empty(0, dtype=float)
    return np.concatenate(pieces)


def _window_record(
    *,
    name: str,
    description: str,
    color: str,
    basis: str,
    start_idx: int,
    stop_idx: int,
    table: dict[str, np.ndarray],
    period_s: float,
) -> dict[str, Any]:
    times = np.asarray(table["t"], dtype=float)
    idx = np.arange(int(start_idx), int(stop_idx) + 1, dtype=int)
    x1 = np.asarray(table["x1"], dtype=float)
    fts = np.asarray(table["Fts"], dtype=float)
    contact = np.asarray(table["contact"], dtype=bool)
    start_s = float(times[start_idx])
    stop_s = float(times[stop_idx])
    return {
        "name": name,
        "description": description,
        "color": color,
        "selection_basis": basis,
        "start_source_idx": int(start_idx),
        "stop_source_idx": int(stop_idx),
        "start_s": start_s,
        "stop_s": stop_s,
        "start_us": start_s * 1.0e6,
        "stop_us": stop_s * 1.0e6,
        "start_ms": start_s * 1.0e3,
        "stop_ms": stop_s * 1.0e3,
        "duration_s": stop_s - start_s,
        "raw_point_count_inclusive": int(idx.size),
        "contact_fraction": float(np.mean(contact[idx])),
        "Fts_min_N": float(np.min(fts[idx])),
        "Fts_max_N": float(np.max(fts[idx])),
        "Fts_min_nN": float(np.min(fts[idx]) * 1.0e9),
        "Fts_max_nN": float(np.max(fts[idx]) * 1.0e9),
        "x1_half_range_nm": float(0.5 * (np.max(x1[idx]) - np.min(x1[idx])) * 1.0e9),
        "actuation_period_s": float(period_s),
        "period_count": float((stop_s - start_s) / period_s),
    }


def _build_windows(table: dict[str, np.ndarray], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    times = np.asarray(table["t"], dtype=float)
    contact = np.asarray(table["contact"], dtype=bool)
    omega0 = float(manifest["physical_constants"]["omega0_rad_s"])
    period_s = float(2.0 * np.pi / omega0)
    window_span = WINDOW_RAW_POINTS_INCLUSIVE - 1
    last = times.size - 1
    contact_idx = np.flatnonzero(contact)
    w0_start = int(contact_idx[0]) if contact_idx.size else 0
    w0_stop = min(last, w0_start + window_span)
    dt_s = float(times[1] - times[0])
    w1_start = int(round((W1_START_S - float(times[0])) / dt_s))
    w1_start = max(0, min(last - window_span, w1_start))
    w1_stop = w1_start + window_span
    w2_stop = last
    w2_start = max(0, w2_stop - window_span)

    windows = [
        _window_record(
            name="W0",
            description="first-contact near-two-cycle window",
            color="#E3A018",
            basis="first contact + 834 full-resolution points",
            start_idx=w0_start,
            stop_idx=w0_stop,
            table=table,
            period_s=period_s,
        ),
        _window_record(
            name="W1",
            description="10 ms near-two-cycle observation window",
            color="#4E9F63",
            basis="fixed start at 10.000 ms",
            start_idx=w1_start,
            stop_idx=w1_stop,
            table=table,
            period_s=period_s,
        ),
        _window_record(
            name="W2",
            description="tail near-two-cycle observation window",
            color="#7E57C2",
            basis="final 834-point window before the full time-span endpoint",
            start_idx=w2_start,
            stop_idx=w2_stop,
            table=table,
            period_s=period_s,
        ),
    ]
    w1_idx = np.arange(w1_start, w1_stop + 1, dtype=int)
    base_density = build_regime_sampling_weights(
        times[w1_idx],
        contact[w1_idx],
        noncontact_weight=NONCONTACT_WEIGHT,
        contact_weight=BASE_CONTACT_WEIGHT,
        transition_weight=TRANSITION_WEIGHT,
        transition_half_width_s=TRANSITION_HALF_WIDTH_S,
    )
    base_weighted_positions = _fixed_count_weighted_positions(base_density, SAMPLE_COUNT)
    transition_mask, contact_mask, noncontact_mask = _regime_masks(
        times[w1_idx],
        contact[w1_idx],
        transition_half_width_s=TRANSITION_HALF_WIDTH_S,
    )
    base_transition_count = int(np.count_nonzero(transition_mask[base_weighted_positions]))
    base_contact_count = int(np.count_nonzero(contact_mask[base_weighted_positions]))
    base_noncontact_count = int(np.count_nonzero(noncontact_mask[base_weighted_positions]))
    requested_contact_count = 2 * base_contact_count
    extra_contact_count = base_contact_count
    weighted_time_s = np.sort(
        np.concatenate(
            [
                times[w1_idx[base_weighted_positions]],
                _interpolated_contact_times(times[w1_idx], contact_mask, extra_contact_count),
            ]
        )
    )
    uniform_positions = np.rint(np.linspace(0, w1_idx.size - 1, SAMPLE_COUNT)).astype(int)
    uniform_positions[0] = 0
    uniform_positions[-1] = w1_idx.size - 1
    uniform_idx = w1_idx[uniform_positions]
    uniform_train_local = np.arange(uniform_idx.size, dtype=int)
    uniform_val_local = np.empty(0, dtype=int)
    weighted_train_local = np.arange(weighted_time_s.size, dtype=int)
    weighted_val_local = np.empty(0, dtype=int)
    final_transition_count = base_transition_count
    final_contact_count = base_contact_count + extra_contact_count
    final_noncontact_count = base_noncontact_count
    windows_manifest = {
        "schema_version": 4,
        "dataset": "AFM06a/datasets/e0.0_real/data",
            "status": "full time-span data generated; W0/W1/W2 selected for visualization",
        "full_time_span_s": [float(times[0]), float(times[-1])],
        "dt_s": float(times[1] - times[0]),
        "nsteps": int(times.size - 1),
        "num_points": int(times.size),
        "window_point_count_on_source_grid": WINDOW_RAW_POINTS_INCLUSIVE,
        "sample_count": SAMPLE_COUNT,
        "weighted_sample_count": int(weighted_time_s.size),
        "force_column": "Fts",
        "force_plot_unit": "nN",
        "windows": windows,
        "W1_sampling": {
            "uniform_count": SAMPLE_COUNT,
            "base_weighted_count": SAMPLE_COUNT,
            "weighted_count": int(weighted_time_s.size),
            "base_weighted_density_transition_contact_noncontact": [
                TRANSITION_WEIGHT,
                BASE_CONTACT_WEIGHT,
                NONCONTACT_WEIGHT,
            ],
            "weighted_density_transition_contact_noncontact": [
                TRANSITION_WEIGHT,
                CONTACT_WEIGHT,
                NONCONTACT_WEIGHT,
            ],
            "contact_oversampling_policy": (
                "Keep transition and noncontact samples from the base 10:5:1 grid, "
                "then add contact-interior samples until the contact count is doubled."
            ),
            "base_regime_counts_transition_contact_noncontact": [
                base_transition_count,
                base_contact_count,
                base_noncontact_count,
            ],
            "requested_final_contact_count": requested_contact_count,
            "available_unique_contact_interior_points": int(np.count_nonzero(contact_mask)),
            "interpolated_extra_contact_count": int(extra_contact_count),
            "final_regime_counts_transition_contact_noncontact": [
                final_transition_count,
                final_contact_count,
                final_noncontact_count,
            ],
            "transition_half_width_s": TRANSITION_HALF_WIDTH_S,
            "validation_policy": "all selected samples are training points; no validation split",
            "uniform_train_count": int(uniform_train_local.size),
            "uniform_validation_count": int(uniform_val_local.size),
            "train_count": int(weighted_train_local.size),
            "validation_count": int(weighted_val_local.size),
        },
    }
    selections = {
        "w1_window": w1_idx,
        "uniform": uniform_idx,
        "weighted": np.searchsorted(times, weighted_time_s, side="left"),
        "weighted_time_s": weighted_time_s,
        "uniform_train_local": uniform_train_local,
        "uniform_val_local": uniform_val_local,
        "weighted_train_local": weighted_train_local,
        "weighted_val_local": weighted_val_local,
    }
    return windows_manifest, selections


def _plot_full_trajectory(table: dict[str, np.ndarray], windows: list[dict[str, Any]], output: Path) -> None:
    times_ms = np.asarray(table["t"]) * 1.0e3
    series = [
        (np.asarray(table["x1"]) * 1.0e9, r"$x_1$ [nm]", "tab:blue"),
        (np.asarray(table["x2"]) * 1.0e6, r"$x_2$ [$\mu$m s$^{-1}$]", "tab:red"),
        (np.asarray(table["x2dot"]), r"$\dot{x}_2$ [m s$^{-2}$]", "tab:purple"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(15.0, 8.4), sharex=True)
    for axis, (values, ylabel, color) in zip(axes, series, strict=True):
        axis.plot(times_ms, values, color=color, linewidth=0.65)
        for window in windows:
            axis.axvspan(
                float(window["start_ms"]),
                float(window["stop_ms"]),
                color=str(window["color"]),
                alpha=0.16,
                linewidth=0,
            )
        axis.set_ylabel(ylabel, fontsize=13)
        axis.tick_params(axis="both", labelsize=11)
        axis.grid(True, alpha=0.28, linewidth=0.7)
    axes[-1].set_xlabel("Time [ms]", fontsize=13)
    axes[-1].set_xlim(float(times_ms[0]), float(times_ms[-1]))
    fig.tight_layout(pad=1.2)
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_fts_full(table: dict[str, np.ndarray], windows: list[dict[str, Any]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(15.0, 4.8))
    axis.plot(np.asarray(table["t"]) * 1.0e3, np.asarray(table["Fts"]) * 1.0e9, color="black", linewidth=0.65)
    for window in windows:
        axis.axvspan(float(window["start_ms"]), float(window["stop_ms"]), color=str(window["color"]), alpha=0.16)
    axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.7)
    axis.set_xlabel("Time [ms]", fontsize=13)
    axis.set_ylabel(r"$F_{ts}$ [nN]", fontsize=13)
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(True, alpha=0.28, linewidth=0.7)
    fig.tight_layout(pad=1.1)
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_fts_vs_x1(table: dict[str, np.ndarray], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 5.2))
    axis.plot(np.asarray(table["x1"]) * 1.0e9, np.asarray(table["Fts"]) * 1.0e9, color="black", linewidth=0.55)
    axis.set_xlabel(r"$x_1$ [nm]", fontsize=13)
    axis.set_ylabel(r"$F_{ts}$ [nN]", fontsize=13)
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(True, alpha=0.28, linewidth=0.7)
    fig.tight_layout(pad=1.1)
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_w1_samples(
    table: dict[str, np.ndarray],
    selections: dict[str, np.ndarray],
    *,
    sample_key: str,
    output: Path,
    include_x2ddot: bool = False,
) -> None:
    times = np.asarray(table["t"], dtype=float)
    window_idx = selections["w1_window"]
    source_idx = selections[sample_key]
    sample_times = selections.get(f"{sample_key}_time_s")
    if sample_times is None:
        sample_times = times[source_idx]
    sample_times = np.asarray(sample_times, dtype=float)
    train_local = selections[f"{sample_key}_train_local"]
    val_local = selections[f"{sample_key}_val_local"]
    transition_times = _transition_times(times, np.asarray(table["contact"]))
    start = float(times[window_idx[0]])
    stop = float(times[window_idx[-1]])
    transitions = transition_times[(transition_times >= start) & (transition_times <= stop)]
    series = [
        (np.asarray(table["x1"]) * 1.0e9, r"$x_1$ [nm]"),
        (np.asarray(table["x2"]) * 1.0e6, r"$x_2$ [$\mu$m s$^{-1}$]"),
        (np.asarray(table["x2dot"]), r"$\dot{x}_2$ [m s$^{-2}$]"),
        (np.asarray(table["Fts"]) * 1.0e9, r"$F_{ts}$ [nN]"),
    ]
    if include_x2ddot:
        x2ddot = np.gradient(np.asarray(table["x2dot"], dtype=float), times, edge_order=2)
        series.insert(3, (x2ddot, r"$\ddot{x}_2$ [m s$^{-3}$]"))
    fig, axes = plt.subplots(len(series), 1, figsize=(14.5, 2.5 * len(series)), sharex=True)
    time_us = times[window_idx] * 1.0e6
    for axis, (values, ylabel) in zip(axes, series, strict=True):
        axis.plot(time_us, values[window_idx], color="black", linewidth=0.9)
        sample_values = np.interp(sample_times, times, values)
        axis.scatter(
            sample_times[train_local] * 1.0e6,
            sample_values[train_local],
            s=18,
            color="tab:blue",
            zorder=4,
            label="training data",
        )
        axis.scatter(
            sample_times[val_local] * 1.0e6,
            sample_values[val_local],
            s=18,
            color="tab:red",
            zorder=5,
            label="validation data",
        )
        for transition in transitions:
            axis.axvline(float(transition) * 1.0e6, color="black", linestyle=":", linewidth=0.8, alpha=0.75)
        axis.set_ylabel(ylabel, fontsize=13)
        axis.tick_params(axis="both", labelsize=11)
        axis.grid(True, alpha=0.28, linewidth=0.7)
    axes[-1].set_xlabel(r"Time [$\mu$s]", fontsize=13)
    axes[-1].set_xlim(float(time_us[0]), float(time_us[-1]))
    axes[0].legend(loc="best", fontsize=10, frameon=True, framealpha=0.9)
    fig.tight_layout(pad=1.1)
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data_dir = REPO_ROOT / "AFM06a" / "datasets" / ERROR_LEVEL / "data"
    table = _load_table(data_dir)
    manifest = _load_manifest(data_dir)
    if "Fts" not in table:
        raise RuntimeError("e0.0_real visualization requires a physical Fts column")
    windows_manifest, selections = _build_windows(table, manifest)
    _write_json(data_dir / "windows_manifest.json", windows_manifest)
    visualization = data_dir / "visualization"
    visualization.mkdir(parents=True, exist_ok=True)
    outputs = [
        visualization / "afm06_real_trajectory_full_x1_x2_x2dot.png",
        visualization / "afm06_real_Fts_full_hard_switch.png",
        visualization / "afm06_real_Fts_vs_x1.png",
        visualization / "afm06_real_training_validation_W1_x1_x2_x2dot_Fts.png",
        visualization / "afm06_real_training_validation_W1_weighted_10_5_1_x1_x2_x2dot_Fts_hard_switch.png",
    ]
    _plot_full_trajectory(table, windows_manifest["windows"], outputs[0])
    _plot_fts_full(table, windows_manifest["windows"], outputs[1])
    _plot_fts_vs_x1(table, outputs[2])
    _plot_w1_samples(table, selections, sample_key="uniform", output=outputs[3])
    _plot_w1_samples(table, selections, sample_key="weighted", output=outputs[4], include_x2ddot=True)
    _write_json(
        visualization / "visualization_summary.json",
        {
            "generated_by": str(SCRIPT_PATH),
            "data_dir": str(data_dir),
            "force_column": "Fts",
            "force_plot_unit": "nN",
            "outputs": [path.name for path in outputs],
        },
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
