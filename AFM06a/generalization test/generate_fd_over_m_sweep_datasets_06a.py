"""Generate AFM06a datasets for frozen-KAN Fd/m generalization tests.

The sweep changes only the actuation acceleration amplitude Fd/m.  The
hard-sample force law, damping, time span, and sampling interval remain
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.datasets.afm_dataset_generator import (  # noqa: E402
    AFM_DATA_COLUMNS,
    generate_afm_dmt_hard_dataset,
)
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (  # noqa: E402
    AFM06aHardSampleInputs,
)


DEFAULT_MAX_FACTOR = 1.5
DEFAULT_NUM_FD_OVER_M = 21
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "fd_over_m_sweep_n021"
MAX_PLOT_POINTS = 50_000


def reference_fd_over_m(settings: AFM06aHardSampleInputs) -> float:
    """Return the reference actuation acceleration amplitude Fd/m."""

    return float(settings.fd_over_m)


def build_fd_over_m_design(
    settings: AFM06aHardSampleInputs,
    *,
    max_factor: float = DEFAULT_MAX_FACTOR,
    num_fd_over_m: int = DEFAULT_NUM_FD_OVER_M,
) -> list[dict[str, Any]]:
    """Return a log-symmetric Fd/m multiplier grid."""

    if not math.isfinite(max_factor) or max_factor <= 1.0:
        raise ValueError("max_factor must be finite and greater than 1")
    if isinstance(num_fd_over_m, bool) or num_fd_over_m < 2:
        raise ValueError("num_fd_over_m must be an integer of at least 2")

    fd0 = reference_fd_over_m(settings)
    log_factors = np.linspace(-math.log(max_factor), math.log(max_factor), num_fd_over_m)
    factors = np.exp(log_factors)
    if num_fd_over_m % 2 == 1:
        factors[num_fd_over_m // 2] = 1.0

    q0 = float(settings.mass_kg) * float(settings.omega0) / float(settings.c_n_s_m)
    design: list[dict[str, Any]] = []
    for index, factor in enumerate(factors, start=1):
        fd_over_m = fd0 * float(factor)
        design.append(
            {
                "index": index,
                "fd_over_m_id": f"fdm{index:03d}",
                "fd_over_m_multiplier": float(factor),
                "actuation_scale": float(settings.actuation_scale) * float(factor),
                "fd_over_m": fd_over_m,
                "reference_fd_over_m": fd0,
                "q_effective": q0,
                "d1": float(settings.d1),
                "d2": float(settings.d2),
                "c_N_s_m": float(settings.c_n_s_m),
                "is_reference": math.isclose(float(factor), 1.0, rel_tol=0.0, abs_tol=1.0e-15),
            }
        )
    if not math.isclose(design[0]["fd_over_m_multiplier"], 1.0 / max_factor, rel_tol=1.0e-14):
        raise RuntimeError("Fd/m sweep lower endpoint is not the log-symmetric 1/max_factor point")
    if not math.isclose(design[-1]["fd_over_m_multiplier"], max_factor, rel_tol=1.0e-14):
        raise RuntimeError("Fd/m sweep upper endpoint is not the log-symmetric max_factor point")
    if num_fd_over_m % 2 == 1 and not math.isclose(
        design[num_fd_over_m // 2]["fd_over_m_multiplier"], 1.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise RuntimeError("Fd/m sweep center point is not the exact reference multiplier")
    return design


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_design(
    output_dir: Path,
    design: list[dict[str, Any]],
    settings: AFM06aHardSampleInputs,
) -> None:
    """Write the complete sweep definition before integrations begin."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        "index",
        "fd_over_m_id",
        "fd_over_m_multiplier",
        "actuation_scale",
        "fd_over_m",
        "reference_fd_over_m",
        "q_effective",
        "d1",
        "d2",
        "c_N_s_m",
        "is_reference",
    )
    with (output_dir / "fd_over_m_sweep_design.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(design)

    _write_json(
        output_dir / "fd_over_m_sweep_design.json",
        {
            "design": "actuation_acceleration_amplitude_variation",
            "spacing": "log_symmetric",
            "factor_grid": "exp(linspace(-log(max_factor), log(max_factor), num_fd_over_m))",
            "mapping": "Fd_over_m = (Fd_over_m)_ref * multiplier",
            "implementation": "Fd is set explicitly so Fd/m follows the requested sweep point",
            "reference_fd_over_m": reference_fd_over_m(settings),
            "fd_over_m_min": design[0]["fd_over_m"],
            "fd_over_m_max": design[-1]["fd_over_m"],
            "max_factor": design[-1]["fd_over_m_multiplier"],
            "num_fd_over_m": len(design),
            "fixed_q_effective": (
                float(settings.mass_kg) * float(settings.omega0) / float(settings.c_n_s_m)
            ),
            "fixed_c_N_s_m": float(settings.c_n_s_m),
            "time_span_s": [float(settings.initial_time), float(settings.end_time)],
            "data_nsteps": int(settings.data_nsteps),
            "data_num_points": int(settings.data_nsteps) + 1,
            "points": design,
        },
    )


def _dataset_path(output_dir: Path, point: dict[str, Any]) -> Path:
    value_text = f"{point['fd_over_m']:.6e}".replace("+", "").replace("-", "m").replace(".", "p")
    return output_dir / point["fd_over_m_id"] / f"afm06a_{point['fd_over_m_id']}_FdOverM{value_text}.npz"


def save_compact_dataset(
    path: Path,
    result: dict[str, object],
    point: dict[str, Any],
    settings: AFM06aHardSampleInputs,
) -> None:
    """Save one canonical compressed table without duplicate CSV/std copies."""

    table = result["afm_table"]
    if not isinstance(table, dict):
        raise TypeError("generator returned an invalid AFM table")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{name: np.asarray(table[name]) for name in AFM_DATA_COLUMNS},
        fd_over_m=np.asarray(point["fd_over_m"], dtype=float),
        fd_over_m_multiplier=np.asarray(point["fd_over_m_multiplier"], dtype=float),
        actuation_scale=np.asarray(point["actuation_scale"], dtype=float),
        reference_fd_over_m=np.asarray(point["reference_fd_over_m"], dtype=float),
        q_effective=np.asarray(point["q_effective"], dtype=float),
        c_N_s_m=np.asarray(settings.c_n_s_m, dtype=float),
        omega0=np.asarray(settings.omega0, dtype=float),
    )


def _plot_indices(num_points: int) -> np.ndarray:
    if num_points <= MAX_PLOT_POINTS:
        return np.arange(num_points, dtype=int)
    return np.unique(np.linspace(0, num_points - 1, MAX_PLOT_POINTS, dtype=int))


def _load_compact_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in AFM_DATA_COLUMNS}


def _write_dataset_parameters(
    folder: Path,
    point: dict[str, Any],
    settings: AFM06aHardSampleInputs,
    data_path: Path,
) -> Path:
    path = folder / "parameters.json"
    _write_json(
        path,
        {
            **point,
            "dataset_path": str(data_path),
            "nsteps": int(settings.data_nsteps),
            "num_points": int(settings.data_nsteps) + 1,
            "time_span_s": [float(settings.initial_time), float(settings.end_time)],
            "varied_parameters": ["Fd_over_m"],
            "fixed_force_law": True,
            "fixed_damping": True,
            "fixed_c_N_s_m": float(settings.c_n_s_m),
        },
    )
    return path


def plot_fd_over_m_dataset(
    folder: Path,
    table: dict[str, np.ndarray],
    point: dict[str, Any],
) -> tuple[Path, Path]:
    visualization = folder / "visualization"
    visualization.mkdir(parents=True, exist_ok=True)
    indices = _plot_indices(len(table["t"]))
    time_us = np.asarray(table["t"])[indices] * 1.0e6
    series = (
        (np.asarray(table["x1"])[indices] * 1.0e9, r"$x_1$ [nm]", "tab:blue"),
        (np.asarray(table["x2"])[indices] * 1.0e6, r"$x_2$ [$\mu$m s$^{-1}$]", "tab:red"),
        (np.asarray(table["x2dot"])[indices], r"$\dot{x}_2$ [m s$^{-2}$]", "tab:purple"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(18, 10.5), sharex=True)
    for axis, (values, ylabel, color) in zip(axes, series, strict=True):
        axis.plot(time_us, values, color=color, linewidth=0.8)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.6, alpha=0.8)
        axis.set_ylabel(ylabel, fontsize=17)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(True, alpha=0.28, linewidth=0.8)
    axes[-1].set_xlabel(r"Time [$\mu$s]", fontsize=17)
    axes[-1].set_xlim(float(time_us[0]), float(time_us[-1]))
    fig.suptitle(
        rf"{point['fd_over_m_id']}: $F_d/m={point['fd_over_m']:.6g}$ m s$^{{-2}}$, "
        rf"multiplier={point['fd_over_m_multiplier']:.6g}",
        fontsize=16,
    )
    fig.tight_layout(pad=2.0)
    state_path = visualization / f"afm06a_{point['fd_over_m_id']}_x1_x2_x2dot.png"
    fig.savefig(state_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(18, 4.8))
    axis.plot(time_us, np.asarray(table["bar_fts"])[indices], color="tab:green", linewidth=0.8)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.6, alpha=0.8)
    axis.set_xlabel(r"Time [$\mu$s]", fontsize=17)
    axis.set_ylabel(r"$\bar{F}_{ts}$ [m s$^{-2}$]", fontsize=17)
    axis.tick_params(axis="both", labelsize=14)
    axis.grid(True, alpha=0.28, linewidth=0.8)
    axis.set_xlim(float(time_us[0]), float(time_us[-1]))
    axis.set_title(
        rf"{point['fd_over_m_id']}: $F_d/m={point['fd_over_m']:.6g}$ m s$^{{-2}}$",
        fontsize=16,
    )
    fig.tight_layout(pad=2.0)
    force_path = visualization / f"afm06a_{point['fd_over_m_id']}_bar_fts.png"
    fig.savefig(force_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return state_path, force_path


def run_sweep(
    *,
    output_dir: Path,
    nsteps: int,
    max_factor: float,
    num_fd_over_m: int,
    overwrite: bool,
    design_only: bool,
    smoke_test: bool,
) -> None:
    reference = AFM06aHardSampleInputs()
    design = build_fd_over_m_design(
        reference,
        max_factor=max_factor,
        num_fd_over_m=num_fd_over_m,
    )
    write_design(output_dir, design, reference)

    print(
        "AFM06a Fd/m sweep: "
        "spacing=log-symmetric, "
        f"reference={reference_fd_over_m(reference):.9g} m/s^2, "
        f"range=[{design[0]['fd_over_m']:.9g}, {design[-1]['fd_over_m']:.9g}], "
        f"points={len(design)}"
    )
    if design_only:
        print(f"Design written to: {output_dir}")
        return

    if smoke_test:
        point = design[len(design) // 2]
        smoke_steps = min(int(nsteps), 2000)
        settings = replace(
            reference,
            fd_n=float(point["fd_over_m"]) * float(reference.mass_kg),
            data_nsteps=smoke_steps,
        )
        result = generate_afm_dmt_hard_dataset(
            settings=settings,
            nsteps=smoke_steps,
            save_outputs=False,
        )
        table = result["afm_table"]
        print(
            "Smoke test passed: "
            f"Fd/m={point['fd_over_m']:.9g} m/s^2, points={smoke_steps + 1}, "
            f"x1=[{np.min(table['x1']):.6e}, {np.max(table['x1']):.6e}]"
        )
        return

    manifest_path = output_dir / "generation_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "output_dir": str(output_dir),
        "nsteps": int(nsteps),
        "num_points_per_dataset": int(nsteps) + 1,
        "datasets": [],
    }
    _write_json(manifest_path, manifest)

    started = time.perf_counter()
    for point in design:
        path = _dataset_path(output_dir, point)
        folder = path.parent
        entry = dict(point)
        entry["path"] = str(path)
        if path.exists() and not overwrite:
            entry["status"] = "existing"
            print(f"[{point['index']:02d}/{len(design)}] skip existing {path.name}")
            table = _load_compact_table(path)
            settings = replace(
                reference,
                fd_n=float(point["fd_over_m"]) * float(reference.mass_kg),
                data_nsteps=int(nsteps),
            )
        else:
            print(
                f"[{point['index']:02d}/{len(design)}] generating "
                f"Fd/m={point['fd_over_m']:.9g} m/s^2, "
                f"multiplier={point['fd_over_m_multiplier']:.9g}"
            )
            item_started = time.perf_counter()
            settings = replace(
                reference,
                fd_n=float(point["fd_over_m"]) * float(reference.mass_kg),
                data_nsteps=int(nsteps),
            )
            result = generate_afm_dmt_hard_dataset(
                settings=settings,
                nsteps=int(nsteps),
                save_outputs=False,
            )
            save_compact_dataset(path, result, point, settings)
            table = result["afm_table"]
            if not isinstance(table, dict):
                raise TypeError("generator returned an invalid AFM table")
            entry["status"] = "generated"
            entry["walltime_s"] = time.perf_counter() - item_started
            entry["size_bytes"] = path.stat().st_size
        parameters_path = _write_dataset_parameters(folder, point, settings, path)
        state_plot, force_plot = plot_fd_over_m_dataset(folder, table, point)
        entry["parameters_path"] = str(parameters_path)
        entry["visualizations"] = [str(state_plot), str(force_plot)]
        manifest["datasets"].append(entry)
        _write_json(manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["total_walltime_s"] = time.perf_counter() - started
    _write_json(manifest_path, manifest)
    print(f"Complete: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nsteps", type=int, default=AFM06aHardSampleInputs().data_nsteps)
    parser.add_argument("--max-factor", type=float, default=DEFAULT_MAX_FACTOR)
    parser.add_argument("--num-fd-over-m", type=int, default=DEFAULT_NUM_FD_OVER_M)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nsteps < 1:
        raise SystemExit("--nsteps must be at least 1")
    run_sweep(
        output_dir=args.output_dir.resolve(),
        nsteps=args.nsteps,
        max_factor=args.max_factor,
        num_fd_over_m=args.num_fd_over_m,
        overwrite=args.overwrite,
        design_only=args.design_only,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
