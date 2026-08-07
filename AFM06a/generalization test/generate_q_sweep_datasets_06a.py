"""Generate isolated AFM06a datasets for frozen-KAN generalization tests.

The sweep changes only the effective cantilever damping level.  The reference
hard-sample model, force law, forcing, time span, and sampling interval remain
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
DEFAULT_NUM_Q = 21
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "q_sweep_n021"
MAX_PLOT_POINTS = 50_000


def build_q_design(
    settings: AFM06aHardSampleInputs,
    *,
    max_factor: float = DEFAULT_MAX_FACTOR,
    num_q: int = DEFAULT_NUM_Q,
) -> list[dict[str, Any]]:
    """Return a log-symmetric Q grid and its physical damping values."""

    if not math.isfinite(max_factor) or max_factor <= 1.0:
        raise ValueError("max_factor must be finite and greater than 1")
    if isinstance(num_q, bool) or num_q < 2:
        raise ValueError("num_q must be an integer of at least 2")
    if settings.c_n_s_m <= 0.0:
        raise ValueError("the reference c must be positive to define Q0=m*omega0/c")

    q0 = float(settings.mass_kg) * float(settings.omega0) / float(settings.c_n_s_m)
    log_factors = np.linspace(-math.log(max_factor), math.log(max_factor), num_q)
    factors = np.exp(log_factors)
    if num_q % 2 == 1:
        factors[num_q // 2] = 1.0

    design: list[dict[str, Any]] = []
    for index, factor in enumerate(factors, start=1):
        q_value = q0 * float(factor)
        damping_scale = q0 / q_value
        c_n_s_m = float(settings.c_n_s_m) * damping_scale
        damping_ratio = c_n_s_m / (float(settings.mass_kg) * float(settings.omega0))
        if not math.isclose(
            c_n_s_m,
            float(settings.mass_kg) * float(settings.omega0) / q_value,
            rel_tol=2.0e-15,
            abs_tol=0.0,
        ):
            raise RuntimeError("internal Q-to-c consistency check failed")
        design.append(
            {
                "index": index,
                "q_id": f"q{index:03d}",
                "q_multiplier": float(factor),
                "q_effective": q_value,
                "damping_scale": damping_scale,
                "c_N_s_m": c_n_s_m,
                "d1": damping_ratio,
                "d2": damping_ratio,
                "is_reference": math.isclose(float(factor), 1.0, rel_tol=0.0, abs_tol=1.0e-15),
            }
        )
    return design


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_design(output_dir: Path, design: list[dict[str, Any]], settings: AFM06aHardSampleInputs) -> None:
    """Write the complete sweep definition before expensive integrations begin."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        "index",
        "q_id",
        "q_multiplier",
        "q_effective",
        "damping_scale",
        "c_N_s_m",
        "d1",
        "d2",
        "is_reference",
    )
    with (output_dir / "q_sweep_design.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(design)

    _write_json(
        output_dir / "q_sweep_design.json",
        {
            "design": "global_damping_variation",
            "mapping": "c(Q)=c_ref*Q0/Q with Q0=m*omega0/c_ref",
            "q0_definition": "Q0=m*omega0/c_ref",
            "q0": float(settings.mass_kg) * float(settings.omega0) / float(settings.c_n_s_m),
            "q_min": design[0]["q_effective"],
            "q_max": design[-1]["q_effective"],
            "max_factor": design[-1]["q_multiplier"],
            "num_q": len(design),
            "reference_c_N_s_m": float(settings.c_n_s_m),
            "time_span_s": [float(settings.initial_time), float(settings.end_time)],
            "data_nsteps": int(settings.data_nsteps),
            "data_num_points": int(settings.data_nsteps) + 1,
            "points": design,
        },
    )


def _dataset_path(output_dir: Path, point: dict[str, Any]) -> Path:
    q_text = f"{point['q_effective']:.6f}".replace(".", "p")
    return output_dir / point["q_id"] / f"afm06a_{point['q_id']}_Q{q_text}.npz"


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
        q_effective=np.asarray(point["q_effective"], dtype=float),
        q_multiplier=np.asarray(point["q_multiplier"], dtype=float),
        damping_scale=np.asarray(point["damping_scale"], dtype=float),
        d1=np.asarray(point["d1"], dtype=float),
        d2=np.asarray(point["d2"], dtype=float),
        c_N_s_m=np.asarray(point["c_N_s_m"], dtype=float),
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
            "varied_parameters": ["Q", "c"],
            "fixed_force_law": True,
        },
    )
    return path


def plot_q_dataset(
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
        rf"{point['q_id']}: $Q={point['q_effective']:.6g}$, "
        rf"$c={point['c_N_s_m']:.6g}$ N s m$^{{-1}}$",
        fontsize=16,
    )
    fig.tight_layout(pad=2.0)
    state_path = visualization / f"afm06a_{point['q_id']}_x1_x2_x2dot.png"
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
    axis.set_title(rf"{point['q_id']}: $Q={point['q_effective']:.6g}$", fontsize=16)
    fig.tight_layout(pad=2.0)
    force_path = visualization / f"afm06a_{point['q_id']}_bar_fts.png"
    fig.savefig(force_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return state_path, force_path


def run_sweep(
    *,
    output_dir: Path,
    nsteps: int,
    max_factor: float,
    num_q: int,
    overwrite: bool,
    design_only: bool,
    smoke_test: bool,
) -> None:
    reference = AFM06aHardSampleInputs()
    design = build_q_design(reference, max_factor=max_factor, num_q=num_q)
    write_design(output_dir, design, reference)

    q0 = float(reference.mass_kg) * float(reference.omega0) / float(reference.c_n_s_m)
    print(
        f"AFM06a Q sweep: Q0={q0:.9g}, range=[{design[0]['q_effective']:.9g}, "
        f"{design[-1]['q_effective']:.9g}], points={len(design)}"
    )
    if design_only:
        print(f"Design written to: {output_dir}")
        return

    if smoke_test:
        point = design[len(design) // 2]
        smoke_steps = min(int(nsteps), 2000)
        settings = replace(
            reference,
            d1=float(point["d1"]),
            d2=float(point["d2"]),
            c_n_s_m=float(point["c_N_s_m"]),
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
            f"Q={point['q_effective']:.9g}, points={smoke_steps + 1}, "
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
                d1=float(point["d1"]),
                d2=float(point["d2"]),
                c_n_s_m=float(point["c_N_s_m"]),
                data_nsteps=int(nsteps),
            )
        else:
            print(
                f"[{point['index']:02d}/{len(design)}] generating "
                f"Q={point['q_effective']:.9g}, c={point['c_N_s_m']:.9g} N*s/m"
            )
            item_started = time.perf_counter()
            settings = replace(
                reference,
                d1=float(point["d1"]),
                d2=float(point["d2"]),
                c_n_s_m=float(point["c_N_s_m"]),
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
        state_plot, force_plot = plot_q_dataset(folder, table, point)
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
    parser.add_argument("--num-q", type=int, default=DEFAULT_NUM_Q)
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
        num_q=args.num_q,
        overwrite=args.overwrite,
        design_only=args.design_only,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
