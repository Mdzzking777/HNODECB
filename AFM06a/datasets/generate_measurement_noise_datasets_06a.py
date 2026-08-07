"""Generate strict AFM06a Gaussian measurement-noise benchmark datasets.

Only x1 is exposed as a noisy observation. Synthetic truth is written to a
separate evaluation-only directory and must never be opened by a trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.stage1pluslight.config import default_config
from AFM06a.stage1pluslight.data import (
    dataset_paths,
    load_dataset,
    make_train_val_masks,
    select_window_source_indices,
)


SCHEMA_VERSION = "afm06a.measurement-noise.v1"
OBSERVATION_KEYS = ("t", "x1_obs", "x1_obs_std", "train_idx", "val_idx")
TRUTH_KEYS = (
    "x1_true",
    "x2_true",
    "x2dot_true",
    "bar_fts_true",
    "contact_true",
    "source_idx",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def validate_observations(
    path: Path,
    *,
    expected_count: int,
    expected_sigma: float,
) -> None:
    with np.load(path, allow_pickle=False) as payload:
        if tuple(payload.files) != OBSERVATION_KEYS:
            raise RuntimeError(
                f"{path} violates the observation whitelist: {payload.files}"
            )
        t = np.asarray(payload["t"], dtype=np.float64)
        x1_obs = np.asarray(payload["x1_obs"], dtype=np.float64)
        x1_obs_std = np.asarray(payload["x1_obs_std"], dtype=np.float64)
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        val_idx = np.asarray(payload["val_idx"], dtype=np.int64)

    if t.shape != (expected_count,) or x1_obs.shape != (expected_count,):
        raise RuntimeError(f"{path} contains an invalid observation shape")
    if x1_obs_std.shape != (expected_count,):
        raise RuntimeError(f"{path} contains an invalid uncertainty shape")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x1_obs)):
        raise RuntimeError(f"{path} contains non-finite observations")
    if not np.all(x1_obs_std == expected_sigma):
        raise RuntimeError(f"{path} contains inconsistent noise standard deviations")
    if np.intersect1d(train_idx, val_idx).size:
        raise RuntimeError(f"{path} has overlapping train/validation indices")
    combined = np.sort(np.concatenate((train_idx, val_idx)))
    if not np.array_equal(combined, np.arange(expected_count, dtype=np.int64)):
        raise RuntimeError(f"{path} train/validation indices do not partition the data")


def validate_truth(path: Path, *, expected_count: int) -> None:
    with np.load(path, allow_pickle=False) as payload:
        if tuple(payload.files) != TRUTH_KEYS:
            raise RuntimeError(f"{path} violates the truth schema: {payload.files}")
        for key in TRUTH_KEYS:
            if np.asarray(payload[key]).shape != (expected_count,):
                raise RuntimeError(f"{path}:{key} has an invalid shape")


def level_directory_name(level: float) -> str:
    percentage = level * 100.0
    if not np.isclose(percentage, round(percentage)):
        raise ValueError("noise levels must be whole percentages in this benchmark")
    return f"noise_{int(round(percentage)):02d}pct"


def parse_args() -> argparse.Namespace:
    default_output = (
        SCRIPT_PATH.parent
        / "noisy"
        / "gaussian_white_measurement_x1"
        / "W1_current_sampling"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=(0.01, 0.05),
        help="Noise standard deviations as fractions of the clean W1 x1 peak-to-peak range.",
    )
    parser.add_argument("--output-root", type=Path, default=default_output)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = tuple(float(level) for level in args.levels)
    if not levels or any(not (0.0 < level < 1.0) for level in levels):
        raise ValueError("every noise level must lie strictly between 0 and 1")
    if len({level_directory_name(level) for level in levels}) != len(levels):
        raise ValueError("noise levels map to duplicate output directory names")

    config = default_config()
    clean_error_level = "e0.0"
    clean_paths = dataset_paths(config.dataset_root, clean_error_level)
    clean_files = (clean_paths["ode_data"], clean_paths["pert_df"])
    clean_hashes_before = {relative_repo_path(path): sha256_file(path) for path in clean_files}

    clean = load_dataset(config.dataset_root, clean_error_level)
    source_idx = np.asarray(
        select_window_source_indices(config, clean),
        dtype=np.int64,
    )
    train_idx, val_idx = make_train_val_masks(
        source_idx.size,
        config.val_stride,
        config.val_offset,
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    t = np.asarray(clean.table["t"], dtype=np.float64)[source_idx]
    x1_true = np.asarray(clean.table["x1"], dtype=np.float64)[source_idx]
    x2_true = np.asarray(clean.table["x2"], dtype=np.float64)[source_idx]
    x2dot_true = np.asarray(clean.table["x2dot"], dtype=np.float64)[source_idx]
    bar_fts_true = np.asarray(clean.table["bar_fts"], dtype=np.float64)[source_idx]
    contact_true = np.asarray(clean.table["contact"], dtype=bool)[source_idx]

    x1_peak_to_peak = float(np.ptp(x1_true))
    if not np.isfinite(x1_peak_to_peak) or x1_peak_to_peak <= 0.0:
        raise RuntimeError("the selected clean W1 x1 range is invalid")

    rng = np.random.default_rng(args.seed)
    full_standard_noise = np.asarray(
        rng.standard_normal(clean.states.shape[1]),
        dtype=np.float64,
    )
    standard_noise = np.asarray(full_standard_noise[source_idx], dtype=np.float64)
    standard_noise_hash = sha256_array(standard_noise)
    full_standard_noise_hash = sha256_array(full_standard_noise)

    output_root = args.output_root.resolve()
    shared_root = output_root / "shared"
    truth_path = shared_root / "evaluation_only" / "truth.npz"
    atomic_savez(
        truth_path,
        x1_true=x1_true,
        x2_true=x2_true,
        x2dot_true=x2dot_true,
        bar_fts_true=bar_fts_true,
        contact_true=contact_true,
        source_idx=source_idx,
    )
    validate_truth(truth_path, expected_count=source_idx.size)
    truth_hash = sha256_file(truth_path)

    sampling_manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "training_visible_file": "observations.npz",
            "training_visible_keys": list(OBSERVATION_KEYS),
            "evaluation_truth_is_forbidden_to_training": True,
            "evaluation_truth_keys": list(TRUTH_KEYS),
        },
        "clean_source": {
            "error_level": clean_error_level,
            "files_sha256": clean_hashes_before,
        },
        "window": {
            "name": "W1",
            "start_s": float(config.window_start_s),
            "stop_s": float(config.window_stop_s),
        },
        "sampling": {
            "policy": config.sampling_policy,
            "sample_stride_reference": int(config.sample_stride),
            "noncontact_weight": float(config.noncontact_sampling_weight),
            "contact_weight": float(config.contact_sampling_weight),
            "transition_weight": float(config.transition_sampling_weight),
            "transition_half_width_s": float(config.transition_half_width_s),
            "sample_count": int(source_idx.size),
            "train_count": int(train_idx.size),
            "validation_count": int(val_idx.size),
            "validation_stride": int(config.val_stride),
            "validation_offset": int(config.val_offset),
            "source_indices_sha256": sha256_array(source_idx),
        },
        "units": {
            "t": "s",
            "x1": "m",
            "x2": "m s^-1",
            "x2dot": "m s^-2",
            "bar_fts": "m s^-2",
        },
        "x1_clean_peak_to_peak_m": x1_peak_to_peak,
        "evaluation_truth": {
            "path": relative_repo_path(truth_path),
            "sha256": truth_hash,
        },
    }
    atomic_write_json(shared_root / "sampling_manifest.json", sampling_manifest)

    generated_cases: list[dict[str, Any]] = []
    normalized_noise_by_level: list[np.ndarray] = []
    for level in levels:
        sigma = level * x1_peak_to_peak
        x1_obs = np.asarray(x1_true + sigma * standard_noise, dtype=np.float64)
        x1_obs_std = np.full(source_idx.size, sigma, dtype=np.float64)
        case_root = output_root / level_directory_name(level)
        observations_path = case_root / "observations.npz"
        atomic_savez(
            observations_path,
            t=t,
            x1_obs=x1_obs,
            x1_obs_std=x1_obs_std,
            train_idx=train_idx,
            val_idx=val_idx,
        )
        validate_observations(
            observations_path,
            expected_count=source_idx.size,
            expected_sigma=sigma,
        )

        normalized_noise = np.asarray((x1_obs - x1_true) / sigma, dtype=np.float64)
        normalized_noise_by_level.append(normalized_noise)
        if not np.allclose(normalized_noise, standard_noise, rtol=0.0, atol=2.0e-14):
            raise RuntimeError("written observations do not reproduce the requested noise vector")

        case_manifest = {
            "schema_version": SCHEMA_VERSION,
            "measurement_model": "x1_obs = x1_true + sigma_x1 * epsilon",
            "noise": {
                "distribution": "Gaussian white measurement noise",
                "epsilon_distribution": "independent standard normal",
                "level_fraction_of_clean_W1_x1_peak_to_peak": level,
                "level_percent": 100.0 * level,
                "sigma_x1_m": sigma,
                "sigma_x1_nm": sigma * 1.0e9,
                "seed": int(args.seed),
                "numpy_generator": "default_rng",
                "numpy_bit_generator": type(rng.bit_generator).__name__,
                "standard_noise_sha256": standard_noise_hash,
                "full_span_standard_noise_sha256": full_standard_noise_hash,
                "nonnegative_clipping_applied": False,
            },
            "observations": {
                "path": relative_repo_path(observations_path),
                "sha256": sha256_file(observations_path),
                "keys": list(OBSERVATION_KEYS),
                "sample_count": int(source_idx.size),
            },
            "shared_sampling_manifest": relative_repo_path(
                shared_root / "sampling_manifest.json"
            ),
            "evaluation_truth": {
                "path": relative_repo_path(truth_path),
                "sha256": truth_hash,
                "forbidden_to_training": True,
            },
        }
        atomic_write_json(case_root / "manifest.json", case_manifest)
        generated_cases.append(case_manifest)

    reference_noise = normalized_noise_by_level[0]
    for candidate_noise in normalized_noise_by_level[1:]:
        if not np.allclose(reference_noise, candidate_noise, rtol=0.0, atol=3.0e-14):
            raise RuntimeError("noise levels do not share the same standard-normal realization")

    clean_hashes_after = {relative_repo_path(path): sha256_file(path) for path in clean_files}
    if clean_hashes_after != clean_hashes_before:
        raise RuntimeError("a clean AFM06a source file changed during noisy-data generation")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "output_root": relative_repo_path(output_root),
        "sample_count": int(source_idx.size),
        "train_count": int(train_idx.size),
        "validation_count": int(val_idx.size),
        "x1_clean_peak_to_peak_m": x1_peak_to_peak,
        "standard_noise_sha256": standard_noise_hash,
        "full_span_standard_noise_sha256": full_standard_noise_hash,
        "all_levels_share_standard_noise": True,
        "clean_sources_unchanged": True,
        "cases": generated_cases,
    }
    atomic_write_json(output_root / "generation_summary.json", summary)

    print(f"AFM06a noisy measurement datasets generated: {output_root}")
    print(
        f"samples={source_idx.size} train={train_idx.size} val={val_idx.size} "
        f"x1_peak_to_peak={x1_peak_to_peak:.12e} m"
    )
    for case in generated_cases:
        noise = case["noise"]
        print(
            f"{noise['level_percent']:.0f}%: "
            f"sigma={noise['sigma_x1_m']:.12e} m "
            f"({noise['sigma_x1_nm']:.9f} nm)"
        )
    print("Protocol validation: PASS")
    print("Clean-source hash validation: PASS")
    print("Shared standard-noise validation: PASS")


if __name__ == "__main__":
    main()
