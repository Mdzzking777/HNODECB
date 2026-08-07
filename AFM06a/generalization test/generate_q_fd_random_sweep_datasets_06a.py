"""Generate the four boundary AFM06a Q x Fd/m datasets.

All parameters except Q/c and Fd/m are loaded from the latest e0.0_real
reference dataset.  The four combinations of the minimum/maximum Q and
minimum/maximum Fd/m values are integrated over 0--12 ms.
"""

from __future__ import annotations

import argparse
import json
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from AFM06a.datasets.afm_dataset_generator import (  # noqa: E402
    AFM_DATA_COLUMNS,
    generate_afm_dmt_hard_dataset,
)
from AFM06a.stage1pluslight.config import default_config as stage1_default_config  # noqa: E402
from AFM06a.stage1pluslight.data import settings_from_generation_manifest  # noqa: E402
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (  # noqa: E402
    AFM06aHardSampleInputs,
)
from generate_fd_over_m_sweep_datasets_06a import (  # noqa: E402
    DEFAULT_NUM_FD_OVER_M,
    build_fd_over_m_design,
    reference_fd_over_m,
)
from generate_q_sweep_datasets_06a import DEFAULT_NUM_Q, build_q_design  # noqa: E402


DEFAULT_MAX_FACTOR = 1.5
DEFAULT_NUM_SAMPLES = 4
END_TIME_S = 0.012
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "data"
DEFAULT_REFERENCE_MANIFEST = (
    REPO_ROOT / "AFM06a" / "datasets" / "e0.0_real" / "data" / "afm06a_generation_manifest.json"
)
MAX_FULL_PLOT_POINTS = 60_000


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load_reference_settings(reference_manifest: Path) -> tuple[AFM06aHardSampleInputs, dict[str, Any]]:
    if not reference_manifest.is_file():
        raise FileNotFoundError(f"AFM06a reference generation manifest not found: {reference_manifest}")
    payload = json.loads(reference_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"AFM06a reference generation manifest must be a JSON object: {reference_manifest}")
    settings = settings_from_generation_manifest(payload)
    settings.validate()
    return settings, payload


def _reference_summary(settings: AFM06aHardSampleInputs) -> dict[str, Any]:
    return {
        "q_effective": float(settings.mass_kg) * float(settings.omega0) / float(settings.c_n_s_m),
        "d1": float(settings.d1),
        "d2": float(settings.d2),
        "c_N_s_m": float(settings.c_n_s_m),
        "dist_m": float(settings.dist),
        "dist_nm": float(settings.dist) * 1.0e9,
        "actuation_scale": float(settings.actuation_scale),
        "fd_over_m": float(reference_fd_over_m(settings)),
        "time_span_s": [float(settings.initial_time), float(settings.end_time)],
        "data_nsteps": int(settings.data_nsteps),
        "data_num_points": int(settings.data_nsteps) + 1,
        "parameter_vector": [float(value) for value in settings.parameter_vector],
    }


def _folder_name(q_point: dict[str, Any], fd_point: dict[str, Any]) -> str:
    q_index = int(q_point["index"])
    fd_index = int(fd_point["index"])
    return f"q_{q_index:03d}_Fd_{fd_index:03d}"


def _plot_indices(indices: np.ndarray, max_points: int = MAX_FULL_PLOT_POINTS) -> np.ndarray:
    if indices.size <= max_points:
        return indices
    selected = np.linspace(0, indices.size - 1, max_points, dtype=int)
    return indices[np.unique(selected)]


def _window_indices(
    times: np.ndarray,
    contact: np.ndarray,
    *,
    window_duration_s: float,
) -> list[tuple[str, np.ndarray]]:
    full = np.arange(times.size, dtype=int)
    contact_idx = np.flatnonzero(contact.astype(bool))
    if contact_idx.size == 0:
        w0_start = float(times[0])
    else:
        w0_start = float(times[int(contact_idx[0])])
    w0_stop = min(float(times[-1]), w0_start + window_duration_s)

    middle_center = 0.5 * (float(times[0]) + float(times[-1]))
    w1_start = max(float(times[0]), middle_center - 0.5 * window_duration_s)
    w1_stop = min(float(times[-1]), middle_center + 0.5 * window_duration_s)

    w2_start = max(float(times[0]), float(times[-1]) - window_duration_s)
    w2_stop = float(times[-1])

    ranges = [
        ("Full time span", float(times[0]), float(times[-1])),
        ("W0: first contact", w0_start, w0_stop),
        ("W1: middle", w1_start, w1_stop),
        ("W2: tail", w2_start, w2_stop),
    ]
    out: list[tuple[str, np.ndarray]] = []
    for name, start, stop in ranges:
        idx = np.flatnonzero((times >= start) & (times <= stop))
        if idx.size == 0:
            closest = int(np.argmin(np.abs(times - start)))
            idx = np.asarray([closest], dtype=int)
        out.append((name, idx))
    return out


def _save_compact_dataset(
    path: Path,
    table: dict[str, np.ndarray],
    q_point: dict[str, Any],
    fd_point: dict[str, Any],
    settings: AFM06aHardSampleInputs,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Match the latest e0.0_real contract: physical Fts [N] is the canonical
    # learned-force field.  bar_fts=Fts/m remains available for RHS use.
    physical_columns = {
        name: np.asarray(table[name])
        for name in AFM_DATA_COLUMNS
        if name != "fts_N"
    }
    physical_columns["Fts"] = np.asarray(table["fts_N"], dtype=float)
    np.savez_compressed(
        path,
        **physical_columns,
        q_index=np.asarray(q_point["index"], dtype=int),
        q_multiplier=np.asarray(q_point["q_multiplier"], dtype=float),
        q_effective=np.asarray(q_point["q_effective"], dtype=float),
        damping_scale=np.asarray(q_point["damping_scale"], dtype=float),
        d1=np.asarray(q_point["d1"], dtype=float),
        d2=np.asarray(q_point["d2"], dtype=float),
        c_N_s_m=np.asarray(q_point["c_N_s_m"], dtype=float),
        fd_over_m_index=np.asarray(fd_point["index"], dtype=int),
        fd_over_m_multiplier=np.asarray(fd_point["fd_over_m_multiplier"], dtype=float),
        fd_over_m=np.asarray(fd_point["fd_over_m"], dtype=float),
        reference_fd_over_m=np.asarray(fd_point["reference_fd_over_m"], dtype=float),
        actuation_scale=np.asarray(fd_point["actuation_scale"], dtype=float),
        omega0=np.asarray(settings.omega0, dtype=float),
    )


def _plot_dataset(
    *,
    output_path: Path,
    table: dict[str, np.ndarray],
    q_point: dict[str, Any],
    fd_point: dict[str, Any],
    window_duration_s: float,
) -> Path:
    times = np.asarray(table["t"], dtype=float)
    contact = np.asarray(table["contact"], dtype=bool)
    x1_nm = np.asarray(table["x1"], dtype=float) * 1.0e9
    fts_nN = np.asarray(table["fts_N"], dtype=float) * 1.0e9
    windows = _window_indices(times, contact, window_duration_s=window_duration_s)

    fig, axes = plt.subplots(2, 4, figsize=(22.0, 8.6), sharex=False)
    for col, (title, idx) in enumerate(windows):
        plot_idx = _plot_indices(idx)
        time_ms = times[plot_idx] * 1.0e3
        axes[0, col].plot(time_ms, x1_nm[plot_idx], color="tab:blue", linewidth=0.9)
        axes[1, col].plot(time_ms, fts_nN[plot_idx], color="tab:green", linewidth=0.9)
        axes[0, col].set_title(title, fontsize=15)
        for row in range(2):
            axes[row, col].grid(True, alpha=0.28, linewidth=0.8)
            axes[row, col].tick_params(axis="both", labelsize=12)
            axes[row, col].set_xlabel("Time [ms]", fontsize=13)
    axes[0, 0].set_ylabel(r"$x_1$ [nm]", fontsize=14)
    axes[1, 0].set_ylabel(r"$F_{ts}$ [nN]", fontsize=14)
    fig.suptitle(
        rf"Q x Fd/m sample: $Q={q_point['q_effective']:.6g}$, "
        rf"$F_d/m={fd_point['fd_over_m']:.6g}$ m s$^{{-2}}$",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.6)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _select_boundary_pairs(
    *,
    q_design: list[dict[str, Any]],
    fd_design: list[dict[str, Any]],
    output_root: Path,
    overwrite: bool,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = [
        (q_design[0], fd_design[0]),
        (q_design[0], fd_design[-1]),
        (q_design[-1], fd_design[0]),
        (q_design[-1], fd_design[-1]),
    ]
    if overwrite:
        return pairs
    return [
        pair for pair in pairs
        if not (output_root / _folder_name(*pair)).exists()
    ]


def run(
    *,
    output_root: Path,
    reference_manifest: Path,
    num_samples: int,
    nsteps: int,
    max_factor: float,
    num_q: int,
    num_fd_over_m: int,
    overwrite: bool,
    smoke_test: bool,
) -> None:
    if num_samples != DEFAULT_NUM_SAMPLES:
        raise ValueError("this generator always creates exactly four Q x Fd/m boundary datasets")
    if nsteps < 1:
        raise ValueError("nsteps must be positive")

    reference, reference_payload = _load_reference_settings(reference_manifest)
    q_design = build_q_design(reference, max_factor=max_factor, num_q=num_q)
    fd_design = build_fd_over_m_design(reference, max_factor=max_factor, num_fd_over_m=num_fd_over_m)
    selected = _select_boundary_pairs(
        q_design=q_design,
        fd_design=fd_design,
        output_root=output_root,
        overwrite=overwrite,
    )
    if smoke_test:
        selected = selected[:1]

    stage1_config = stage1_default_config(REPO_ROOT)
    window_duration_s = float(stage1_config.window_stop_s) - float(stage1_config.window_start_s)
    if not np.isfinite(window_duration_s) or window_duration_s <= 0.0:
        raise RuntimeError(f"Invalid AFM06a window duration: {window_duration_s}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "q_fd_random_sweep_manifest.json"
    manifest_nsteps = 2000 if smoke_test else int(nsteps)
    manifest: dict[str, Any] = {
        "status": "running",
        "selection_policy": "four boundary pairs: Q min/max crossed with Fd/m min/max",
        "reproducible": True,
        "num_requested": DEFAULT_NUM_SAMPLES,
        "num_generated_this_run": 0,
        "requested_nsteps": int(nsteps),
        "nsteps": int(manifest_nsteps),
        "num_points_per_dataset": int(manifest_nsteps) + 1,
        "max_factor": float(max_factor),
        "num_q": int(num_q),
        "num_fd_over_m": int(num_fd_over_m),
        "window_duration_s": window_duration_s,
        "reference_manifest": str(reference_manifest.resolve()),
        "reference_manifest_system": reference_payload.get("system"),
        "reference_settings": _reference_summary(reference),
        "datasets": [],
    }
    _write_json(manifest_path, manifest)

    started = time.perf_counter()
    for ordinal, (q_point, fd_point) in enumerate(selected, start=1):
        folder_name = _folder_name(q_point, fd_point)
        folder = output_root / folder_name
        data_path = folder / f"afm06a_{folder_name}_full_timespan.npz"
        plot_path = folder / f"afm06a_{folder_name}_x1_Fts_full_W0_W1_W2.png"
        parameters_path = folder / "parameters.json"
        print(
            f"[{ordinal:02d}/{len(selected)}] {folder_name}: "
            f"Q={q_point['q_effective']:.9g}, Fd/m={fd_point['fd_over_m']:.9g} m/s^2",
            flush=True,
        )

        if smoke_test:
            run_nsteps = 2000
        else:
            run_nsteps = int(nsteps)
        settings = replace(
            reference,
            d1=float(q_point["d1"]),
            d2=float(q_point["d2"]),
            c_n_s_m=float(q_point["c_N_s_m"]),
            fd_n=float(fd_point["fd_over_m"]) * float(reference.mass_kg),
            initial_time=0.0,
            end_time=END_TIME_S,
            data_nsteps=run_nsteps,
        )
        item_started = time.perf_counter()
        result = generate_afm_dmt_hard_dataset(
            settings=settings,
            nsteps=run_nsteps,
            save_outputs=False,
        )
        table = result["afm_table"]
        if not isinstance(table, dict):
            raise TypeError("generator returned an invalid AFM table")
        if not smoke_test:
            _save_compact_dataset(data_path, table, q_point, fd_point, settings)
            _write_json(
                parameters_path,
                {
                    "folder": folder_name,
                    "dataset_path": str(data_path),
                    "plot_path": str(plot_path),
                    "q": q_point,
                    "fd_over_m": fd_point,
                    "reference_manifest": str(reference_manifest.resolve()),
                    "reference_settings": _reference_summary(reference),
                    "nsteps": int(run_nsteps),
                    "num_points": int(run_nsteps) + 1,
                    "time_span_s": [float(settings.initial_time), float(settings.end_time)],
                    "window_definitions": {
                        "full": "complete generated time span",
                        "W0": "window_duration_s starting from first detected contact",
                        "W1": "window_duration_s centered at the full time span midpoint",
                        "W2": "window_duration_s ending at the final time point",
                    },
                },
            )
            _plot_dataset(
                output_path=plot_path,
                table=table,
                q_point=q_point,
                fd_point=fd_point,
                window_duration_s=window_duration_s,
            )
        record = {
            "folder": folder_name,
            "q_index": int(q_point["index"]),
            "fd_over_m_index": int(fd_point["index"]),
            "q_effective": float(q_point["q_effective"]),
            "q_multiplier": float(q_point["q_multiplier"]),
            "fd_over_m": float(fd_point["fd_over_m"]),
            "fd_over_m_multiplier": float(fd_point["fd_over_m_multiplier"]),
            "data_path": str(data_path),
            "plot_path": str(plot_path),
            "walltime_s": time.perf_counter() - item_started,
            "smoke_test": bool(smoke_test),
        }
        manifest["datasets"].append(record)
        manifest["num_generated_this_run"] = int(len(manifest["datasets"]))
        _write_json(manifest_path, manifest)
        if smoke_test:
            print("Smoke test integration passed; no full dataset was saved.", flush=True)
            break

    manifest["status"] = "complete"
    manifest["total_walltime_s"] = time.perf_counter() - started
    _write_json(manifest_path, manifest)
    print(f"Complete: {manifest_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    default_reference = settings_from_generation_manifest(
        json.loads(DEFAULT_REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    ) if DEFAULT_REFERENCE_MANIFEST.is_file() else AFM06aHardSampleInputs()
    reference_dt = (
        float(default_reference.end_time) - float(default_reference.initial_time)
    ) / int(default_reference.data_nsteps)
    parser.add_argument("--nsteps", type=int, default=int(round(END_TIME_S / reference_dt)))
    parser.add_argument("--max-factor", type=float, default=DEFAULT_MAX_FACTOR)
    parser.add_argument("--num-q", type=int, default=DEFAULT_NUM_Q)
    parser.add_argument("--num-fd-over-m", type=int, default=DEFAULT_NUM_FD_OVER_M)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        output_root=args.output_root.resolve(),
        reference_manifest=args.reference_manifest.resolve(),
        num_samples=int(args.num_samples),
        nsteps=int(args.nsteps),
        max_factor=float(args.max_factor),
        num_q=int(args.num_q),
        num_fd_over_m=int(args.num_fd_over_m),
        overwrite=bool(args.overwrite),
        smoke_test=bool(args.smoke_test),
    )


if __name__ == "__main__":
    main()
