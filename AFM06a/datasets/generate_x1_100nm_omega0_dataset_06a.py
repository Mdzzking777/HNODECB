"""Regenerate the AFM06a x1_100nm_omega0 dataset and standard plots.

This case uses the AFM04-style physical cantilever equation

    m*x2dot = Fd*sin(omega0*t) - c*x2 - k*x1 + Fts

with the hard-sample DMT force represented through the mass-normalized
coefficients CA=-AR/(6m) and CH=4E*sqrt(R)/(3m).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
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

from AFM06a.datasets.afm_dataset_generator import generate_afm_dmt_hard_dataset  # noqa: E402
from AFM06a.stage1pluslight.config import default_config  # noqa: E402
from AFM06a.stage1pluslight.data import (  # noqa: E402
    LoadedDataset,
    build_regime_sampling_weights,
    make_train_val_masks,
    select_window_source_indices,
)
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (  # noqa: E402
    AFM06aHardSampleInputs,
)


ERROR_LEVEL = "e0.0_x1_100nm_\u03c90"
ACTUATION_SCALE = 2.5
FIG_DPI = 260


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_current_manifest(data_dir: Path) -> dict[str, Any]:
    manifest = json.loads((data_dir / "afm06a_generation_manifest.json").read_text(encoding="utf-8"))
    return manifest


def _transition_times(times: np.ndarray, contact: np.ndarray) -> np.ndarray:
    contact = np.asarray(contact, dtype=bool)
    switch_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1
    if switch_idx.size == 0:
        return np.empty(0, dtype=float)
    return 0.5 * (times[switch_idx - 1] + times[switch_idx])


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
    contact = np.asarray(table["contact"], dtype=bool)
    x1 = np.asarray(table["x1"], dtype=float)
    bar_fts = np.asarray(table["bar_fts"], dtype=float)
    idx = np.arange(start_idx, stop_idx + 1, dtype=int)
    start = float(times[start_idx])
    stop = float(times[stop_idx])
    duration = stop - start
    return {
        "name": name,
        "description": description,
        "color": color,
        "selection_basis": basis,
        "start_source_idx": int(start_idx),
        "stop_source_idx": int(stop_idx),
        "start_s": start,
        "stop_s": stop,
        "start_us": start * 1.0e6,
        "stop_us": stop * 1.0e6,
        "start_ms": start * 1.0e3,
        "stop_ms": stop * 1.0e3,
        "duration_s": duration,
        "raw_point_count_inclusive": int(idx.size),
        "contact_fraction": float(np.mean(contact[idx])),
        "bar_fts_min": float(np.min(bar_fts[idx])),
        "bar_fts_max": float(np.max(bar_fts[idx])),
        "x1_half_range_nm": float(0.5 * (np.max(x1[idx]) - np.min(x1[idx])) * 1.0e9),
        "actuation_period_s": float(period_s),
        "period_count": float(duration / period_s),
    }


def _nearest_idx(times: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(times - float(value))))


def build_windows_manifest(
    *,
    data_dir: Path,
    table: dict[str, np.ndarray],
    settings: AFM06aHardSampleInputs,
    config: Any,
) -> dict[str, Any]:
    times = np.asarray(table["t"], dtype=float)
    contact = np.asarray(table["contact"], dtype=bool)
    period_s = float(2.0 * np.pi / settings.omega0)

    w1_start = float(config.window_start_s)
    w1_stop = float(config.window_stop_s)
    w1_start_idx = _nearest_idx(times, w1_start)
    w1_stop_idx = _nearest_idx(times, w1_stop)
    window_duration_s = float(times[w1_stop_idx] - times[w1_start_idx])

    contact_idx = np.flatnonzero(contact)
    w0_start_idx = int(contact_idx[0]) if contact_idx.size else 0
    w0_stop_idx = min(times.size - 1, _nearest_idx(times, times[w0_start_idx] + window_duration_s))
    w2_stop_idx = times.size - 1
    w2_start_idx = max(0, _nearest_idx(times, times[w2_stop_idx] - window_duration_s))

    windows = [
        _window_record(
            name="W0",
            description="first-contact near-two-cycle window",
            color="#E3A018",
            basis="first contact + W1 near-two-cycle duration",
            start_idx=w0_start_idx,
            stop_idx=w0_stop_idx,
            table=table,
            period_s=period_s,
        ),
        _window_record(
            name="W1",
            description="middle stable near-two-cycle window",
            color="#4E9F63",
            basis="centered at 0.375 ms in the 0.75 ms full time span",
            start_idx=w1_start_idx,
            stop_idx=w1_stop_idx,
            table=table,
            period_s=period_s,
        ),
        _window_record(
            name="W2",
            description="tail near-two-cycle window",
            color="#7E57C2",
            basis="final W1-duration window before the 0.75 ms endpoint",
            start_idx=w2_start_idx,
            stop_idx=w2_stop_idx,
            table=table,
            period_s=period_s,
        ),
    ]
    source_idx = select_window_source_indices(
        config,
        LoadedDataset(
            states=np.vstack((table["x1"], table["x2"])),
            table=table,
            generation_manifest={},
            settings=settings,
        ),
    )
    train_idx_local, val_idx_local = make_train_val_masks(
        source_idx.size,
        int(config.val_stride),
        int(config.val_offset),
    )
    first_contact = windows[0]["start_source_idx"]
    manifest = {
        "schema_version": 4,
        "dataset": f"AFM06a/datasets/{ERROR_LEVEL}/data",
        "reference_policy": (
            "0.75 ms full time span; W1 is centered at 0.375 ms, "
            "with the formal near-two-cycle W1 train/validation window."
        ),
        "full_time_span_s": [float(times[0]), float(times[-1])],
        "dt_s": float(times[1] - times[0]),
        "sample_stride": int(config.sample_stride),
        "window_point_count_on_source_grid": int(windows[1]["raw_point_count_inclusive"]),
        "window_duration_s": float(window_duration_s),
        "first_contact": {
            "source_idx": int(first_contact),
            "time_s": float(times[first_contact]),
            "time_us": float(times[first_contact] * 1.0e6),
            "time_ms": float(times[first_contact] * 1.0e3),
        },
        "contact_threshold": {
            "x1_m": float(settings.a0 - settings.dist),
            "x1_nm": float((settings.a0 - settings.dist) * 1.0e9),
            "dist_m": float(settings.dist),
            "a0_m": float(settings.a0),
        },
        "windows": windows,
        "W1_sampling": {
            "uniform_count": int(config.sample_count),
            "weighted_count": int(source_idx.size),
            "source_count": int(source_idx.size),
            "target_count": int(config.sample_count),
            "weighted_density_transition_contact_noncontact": [
                float(config.transition_sampling_weight),
                float(config.contact_sampling_weight),
                float(config.noncontact_sampling_weight),
            ],
            "transition_half_width_s": float(config.transition_half_width_s),
            "validation_policy": "periodic 4:1 training/validation split",
            "train_count": int(train_idx_local.size),
            "validation_count": int(val_idx_local.size),
        },
    }
    _write_json(data_dir / "windows_manifest.json", manifest)
    return manifest


def _window_indices(times: np.ndarray, start: float, stop: float) -> np.ndarray:
    return np.flatnonzero((times >= float(start)) & (times <= float(stop)))


def _plot_full_trajectory(table: dict[str, np.ndarray], manifest: dict[str, Any], output: Path) -> None:
    times_ms = np.asarray(table["t"], dtype=float) * 1.0e3
    series = [
        (np.asarray(table["x1"]) * 1.0e9, r"$x_1$ [nm]", "tab:blue"),
        (np.asarray(table["x2"]) * 1.0e6, r"$x_2$ [$\mu$m s$^{-1}$]", "tab:red"),
        (np.asarray(table["x2dot"]), r"$\dot{x}_2$ [m s$^{-2}$]", "tab:purple"),
    ]
    first_contact_ms = float(manifest["first_contact"]["time_ms"])
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 8.2), sharex=True)
    for axis, (values, label, color) in zip(axes, series, strict=True):
        axis.plot(times_ms, values, color=color, linewidth=0.75)
        axis.axvline(first_contact_ms, color="tab:red", linestyle="--", linewidth=1.0)
        axis.set_ylabel(label, fontsize=13)
        axis.tick_params(axis="both", labelsize=11)
        axis.grid(True, alpha=0.28, linewidth=0.7)
    axes[-1].set_xlabel("Time [ms]", fontsize=13)
    axes[-1].set_xlim(float(times_ms[0]), float(times_ms[-1]))
    fig.tight_layout(pad=1.2)
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_bar_fts_full(table: dict[str, np.ndarray], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(14.5, 4.6))
    axis.plot(np.asarray(table["t"]) * 1.0e3, np.asarray(table["bar_fts"]), color="black", linewidth=0.75)
    axis.axhline(0.0, color="0.3", linestyle="--", linewidth=0.7)
    axis.set_xlabel("Time [ms]", fontsize=13)
    axis.set_ylabel(r"$\bar{F}_{ts}$ [m s$^{-2}$]", fontsize=13)
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(True, alpha=0.28, linewidth=0.7)
    fig.tight_layout(pad=1.1)
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_fts_vs_x1(table: dict[str, np.ndarray], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 5.2))
    axis.plot(np.asarray(table["x1"]) * 1.0e9, np.asarray(table["bar_fts"]), color="black", linewidth=0.65)
    axis.set_xlabel(r"$x_1$ [nm]", fontsize=13)
    axis.set_ylabel(r"$\bar{F}_{ts}$ [m s$^{-2}$]", fontsize=13)
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(True, alpha=0.28, linewidth=0.7)
    fig.tight_layout(pad=1.1)
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_w1_samples(
    *,
    table: dict[str, np.ndarray],
    config: Any,
    settings: AFM06aHardSampleInputs,
    output: Path,
    weighted: bool,
) -> None:
    loaded = LoadedDataset(
        states=np.vstack((table["x1"], table["x2"])),
        table=table,
        generation_manifest={},
        settings=settings,
    )
    source_idx = select_window_source_indices(config, loaded)
    if not weighted:
        window_idx = _window_indices(table["t"], config.window_start_s, config.window_stop_s)
        positions = np.rint(np.linspace(0, window_idx.size - 1, int(config.sample_count))).astype(int)
        positions[0] = 0
        positions[-1] = window_idx.size - 1
        source_idx = window_idx[positions]
    train_local, val_local = make_train_val_masks(source_idx.size, int(config.val_stride), int(config.val_offset))
    train_idx = source_idx[train_local]
    val_idx = source_idx[val_local]
    display_idx = _window_indices(table["t"], config.window_start_s, config.window_stop_s)

    times = np.asarray(table["t"], dtype=float)
    time_us = times[display_idx] * 1.0e6
    train_us = times[train_idx] * 1.0e6
    val_us = times[val_idx] * 1.0e6
    series = [
        ("x1", np.asarray(table["x1"]) * 1.0e9, r"$x_1$ [nm]"),
        ("x2", np.asarray(table["x2"]) * 1.0e6, r"$x_2$ [$\mu$m s$^{-1}$]"),
        ("x2dot", np.asarray(table["x2dot"]), r"$\dot{x}_2$ [m s$^{-2}$]"),
        ("bar_fts", np.asarray(table["bar_fts"]), r"$\bar{F}_{ts}$ [m s$^{-2}$]"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(14.5, 10.0), sharex=True)
    transition_times = _transition_times(times, table["contact"])
    transition_in_window = transition_times[
        (transition_times >= float(config.window_start_s)) & (transition_times <= float(config.window_stop_s))
    ]
    for axis, (_name, values, ylabel) in zip(axes, series, strict=True):
        axis.plot(time_us, values[display_idx], color="black", linewidth=0.9)
        axis.scatter(train_us, values[train_idx], s=18, color="tab:blue", zorder=4, label="training data")
        axis.scatter(val_us, values[val_idx], s=18, color="tab:red", zorder=5, label="validation data")
        for t0 in transition_in_window:
            axis.axvline(float(t0) * 1.0e6, color="black", linestyle=":", linewidth=0.8, alpha=0.75)
        axis.set_ylabel(ylabel, fontsize=13)
        axis.tick_params(axis="both", labelsize=11)
        axis.grid(True, alpha=0.28, linewidth=0.7)
    axes[-1].set_xlabel(r"Time [$\mu$s]", fontsize=13)
    axes[0].legend(loc="best", fontsize=10, frameon=True, framealpha=0.9)
    axes[-1].set_xlim(float(time_us[0]), float(time_us[-1]))
    fig.tight_layout(pad=1.1)
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    settings = AFM06aHardSampleInputs(actuation_scale=ACTUATION_SCALE)
    result = generate_afm_dmt_hard_dataset(
        settings=settings,
        error_level=ERROR_LEVEL,
        save_outputs=True,
    )
    data_dir = Path(result["data_dir"])
    table = result["afm_table"]
    if not isinstance(table, dict):
        raise TypeError("generator returned an invalid AFM table")
    generation_manifest = _load_current_manifest(data_dir)
    config = default_config(REPO_ROOT)
    manifest = build_windows_manifest(data_dir=data_dir, table=table, settings=settings, config=config)

    # Extend the generation manifest with the formal window metadata.
    generation_manifest["schema_version"] = 5
    generation_manifest["formula_format"] = (
        "AFM04-style physical cantilever equation using m,c,k,omega0,Fd "
        "and mass-normalized hard-sample DMT coefficients CA,CH"
    )
    generation_manifest["source_constants"] = {
        "m_c_k_omega0": "external silicon data",
        "Fd": "current effective Fd/m multiplied by external-data mass",
        "CA": "C1*eta_star_length^3*reference_omega0^2",
        "CH": "C2*reference_omega0^2/sqrt(eta_star_length)",
    }
    generation_manifest["contact_and_amplitude_summary"] = {
        "full_contact_fraction": float(np.mean(np.asarray(table["contact"], dtype=bool))),
        "full_bar_fts_min": float(np.min(table["bar_fts"])),
        "full_bar_fts_max": float(np.max(table["bar_fts"])),
        "full_x1_half_range_nm": float(0.5 * (np.max(table["x1"]) - np.min(table["x1"])) * 1.0e9),
        "first_contact_time_s": manifest["first_contact"]["time_s"],
        "actuation_period_s": float(2.0 * np.pi / settings.omega0),
        "windows": manifest["windows"],
    }
    _write_json(data_dir / "afm06a_generation_manifest.json", generation_manifest)

    viz = data_dir / "visualization"
    viz.mkdir(parents=True, exist_ok=True)
    generated_plots: list[Path] = []
    generated_plots.append(viz / "afm06_trajectory_full_x1_x2_transient_stable_hard_switch.png")
    _plot_full_trajectory(table, manifest, generated_plots[-1])
    generated_plots.append(viz / "afm06_bar_fts_full_hard_switch.png")
    _plot_bar_fts_full(table, generated_plots[-1])
    generated_plots.append(viz / "afm06_bar_fts_vs_x1.png")
    _plot_fts_vs_x1(table, generated_plots[-1])
    generated_plots.append(viz / "afm06_training_validation_W1_x1_x2_x2dot_fts.png")
    _plot_w1_samples(
        table=table,
        config=config,
        settings=settings,
        weighted=False,
        output=generated_plots[-1],
    )
    generated_plots.append(viz / "afm06_training_validation_W1_weighted_10_5_1_x1_x2_x2dot_fts_hard_switch.png")
    _plot_w1_samples(
        table=table,
        config=config,
        settings=settings,
        weighted=True,
        output=generated_plots[-1],
    )
    _write_json(
        viz / "visualization_summary.json",
        {
            "generated_by": str(SCRIPT_PATH),
            "data_dir": str(data_dir),
            "formula_format": generation_manifest["formula_format"],
            "outputs": [path.name for path in generated_plots],
        },
    )
    print(f"Generated AFM06a physical-form dataset: {data_dir}")
    print(f"m={settings.mass_kg:.12e} kg, c={settings.c_n_s_m:.12e} N*s/m, k={settings.k_n_m:.12e} N/m")
    print(f"omega0={settings.omega0:.12e} rad/s, Fd={settings.fd:.12e} N, Fd/m={settings.fd_over_m:.12e} m/s^2")
    print(f"CA={settings.ca:.12e}, CH={settings.ch:.12e}")


if __name__ == "__main__":
    main()
