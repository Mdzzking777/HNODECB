"""Apply one archived AFM06a trained KAN to generated Q x Fd/m datasets.

For each q_XXX_Fd_YYY dataset folder, the script writes a fresh ``prediction``
folder containing:

1. autonomous 0.5 ms rollout using only the initial state;
2. pointwise force evaluation by feeding every true x1 point to the trained KAN.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.stage1pluslight.checkpoint import load_checkpoint  # noqa: E402
from AFM06a.stage1pluslight.data import settings_from_generation_manifest  # noqa: E402
from AFM06a.stage2light.config import default_config as stage2_default_config  # noqa: E402
from AFM06a.stage2light.data import load_stage1_endpoint  # noqa: E402
from AFM06a.stage2light.kan_backend import restore_exact_stage1_model  # noqa: E402
from AFM06a.stage2light.rollout import rollout_single_shooting_torch, x2dot_rhs_torch  # noqa: E402
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (  # noqa: E402
    AFM06aHardSampleInputs,
)


DEFAULT_DATA_ROOT = SCRIPT_DIR / "data"
DEFAULT_ARCHIVE_DIR = (
    REPO_ROOT / "AFM06a" / "archive" / "st2l" / "rank1_20260801_131123"
)
DEFAULT_REFERENCE_MANIFEST = (
    REPO_ROOT / "AFM06a" / "datasets" / "e0.0_real" / "data" / "afm06a_generation_manifest.json"
)
# Match afm_param_stage2light_06a_bar_fts_full_rollout_grid.png: 10.0--10.5 ms.
PREDICTION_DURATION_S = 0.5e-3
MAX_FULL_PLOT_POINTS = 60_000
POINTWISE_CHUNK_SIZE = 131_072
TRAINED_NORMALIZER_POLICY = "trained_archive_x1_mean_std"
DATASET_NORMALIZER_POLICY = "dataset_specific_full_span_x1_mean_std"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected pickle payload in {path}: {type(payload)!r}")
    return payload


def _archive_stage_paths(archive_dir: Path) -> tuple[Path, Path]:
    archive = archive_dir.resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Archive directory not found: {archive}")
    result_files = sorted((archive / "result").glob("afm_param_stage2light_06a_rank*.pkl"))
    stage1_files = sorted(
        (archive / "conditional dependency").glob("afm_param_stage1pluslight_06a.pkl")
    )
    if len(result_files) != 1:
        raise RuntimeError(
            f"Expected exactly one AFM06a st2l result in {archive / 'result'}, found {len(result_files)}"
        )
    if len(stage1_files) != 1:
        raise RuntimeError(
            "Expected exactly one AFM06a st1pl dependency in "
            f"{archive / 'conditional dependency'}, found {len(stage1_files)}"
        )
    return result_files[0].resolve(), stage1_files[0].resolve()


def restore_archived_model(archive_dir: Path):
    stage2_result, stage1_result = _archive_stage_paths(archive_dir)
    payload = _load_pickle(stage2_result)
    if not bool(payload.get("complete")):
        raise RuntimeError(f"Archived AFM06a st2l result is incomplete: {stage2_result}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Archived AFM06a st2l result has no model_state_dict: {stage2_result}")

    saved = payload.get("config")
    base = stage2_default_config(REPO_ROOT)
    updates: dict[str, Any] = {
        "input_rank": int(payload.get("input_rank", base.input_rank)),
        "device": "cpu",
        "stage1_result_path": stage1_result,
    }
    if isinstance(saved, dict):
        if saved.get("pykan_root") is not None:
            updates["pykan_root"] = Path(saved["pykan_root"])
        for key in (
            "gain_enabled",
            "gain_learnable",
            "soft_mask_enabled",
            "soft_mask_trainable",
        ):
            if saved.get(key) is not None:
                updates[key] = bool(saved[key])
    config = replace(base, **updates)
    endpoint = load_stage1_endpoint(config, enforce_current_contract=False)
    model = restore_exact_stage1_model(config, endpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, endpoint, payload, stage2_result, stage1_result


def _dataset_roots(data_root: Path) -> list[Path]:
    return sorted(
        path
        for path in data_root.glob("q_[0-9][0-9][0-9]_Fd_[0-9][0-9][0-9]")
        if path.is_dir()
    )


def _load_dataset(dataset_root: Path) -> tuple[Path, dict[str, np.ndarray], dict[str, Any]]:
    parameters_path = dataset_root / "parameters.json"
    parameters: dict[str, Any] = {}
    if parameters_path.is_file():
        loaded = json.loads(parameters_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            parameters = loaded
    configured = parameters.get("dataset_path")
    candidates: list[Path] = []
    if configured is not None and Path(configured).is_file():
        candidates.append(Path(configured))
    candidates.extend(sorted(dataset_root.glob("afm06a_q_*_Fd_*_full_timespan.npz")))
    unique = list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
    if len(unique) != 1:
        raise RuntimeError(f"Expected one q-Fd dataset NPZ in {dataset_root}, found {len(unique)}")
    data_path = unique[0]
    with np.load(data_path, allow_pickle=False) as payload:
        required = ("t", "x1", "x2", "x2dot", "contact", "bar_fts")
        missing = [name for name in required if name not in payload]
        if missing:
            raise RuntimeError(f"Dataset is missing {missing}: {data_path}")
        table = {name: np.asarray(payload[name]) for name in required}
        for optional in (
            "q_index",
            "q_multiplier",
            "q_effective",
            "damping_scale",
            "d1",
            "d2",
            "fd_over_m_index",
            "fd_over_m_multiplier",
            "fd_over_m",
            "actuation_scale",
        ):
            if optional in payload:
                table[optional] = np.asarray(payload[optional])
    return data_path, table, parameters


def _scalar_from_table_or_params(
    table: dict[str, np.ndarray],
    parameters: dict[str, Any],
    name: str,
    fallback: float | None = None,
) -> float:
    if name in table:
        return float(np.asarray(table[name]).reshape(()))
    node: Any = parameters
    if name == "q_effective" and isinstance(parameters.get("q"), dict):
        node = parameters["q"]
    elif name in {"d1", "d2", "q_multiplier"} and isinstance(parameters.get("q"), dict):
        node = parameters["q"]
    elif name in {"fd_over_m", "fd_over_m_multiplier", "actuation_scale"} and isinstance(
        parameters.get("fd_over_m"), dict
    ):
        node = parameters["fd_over_m"]
    elif name in {"dist_m", "dist_nm"} and isinstance(parameters.get("reference_settings"), dict):
        node = parameters["reference_settings"]
    if isinstance(node, dict) and name in node:
        return float(node[name])
    if fallback is None:
        raise RuntimeError(f"Could not resolve required scalar {name!r}")
    return float(fallback)


def _load_reference_settings(reference_manifest: Path) -> AFM06aHardSampleInputs:
    if not reference_manifest.is_file():
        raise FileNotFoundError(f"AFM06a reference manifest not found: {reference_manifest}")
    payload = json.loads(reference_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"AFM06a reference manifest must contain a JSON object: {reference_manifest}")
    settings = settings_from_generation_manifest(payload)
    settings.validate()
    return settings


def _settings_for_dataset(
    table: dict[str, np.ndarray],
    parameters: dict[str, Any],
    reference: AFM06aHardSampleInputs,
) -> AFM06aHardSampleInputs:
    times = np.asarray(table["t"], dtype=float)
    return replace(
        reference,
        d1=_scalar_from_table_or_params(table, parameters, "d1"),
        d2=_scalar_from_table_or_params(table, parameters, "d2"),
        actuation_scale=_scalar_from_table_or_params(table, parameters, "actuation_scale", 1.0),
        dist_m=_scalar_from_table_or_params(table, parameters, "dist_m", reference.dist),
        x1_start=float(np.asarray(table["x1"], dtype=float)[0]),
        x2_start=float(np.asarray(table["x2"], dtype=float)[0]),
        initial_time=float(times[0]),
        end_time=float(times[-1]),
        data_nsteps=int(times.size - 1),
    )


def _safe_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = 1.0
    return scale


def _normalizer_for_dataset(table: dict[str, np.ndarray], policy: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    normalized = str(policy).strip().lower()
    if normalized == TRAINED_NORMALIZER_POLICY:
        return None, None
    if normalized != DATASET_NORMALIZER_POLICY:
        raise ValueError(
            f"unsupported normalizer policy {policy!r}; expected "
            f"{TRAINED_NORMALIZER_POLICY!r} or {DATASET_NORMALIZER_POLICY!r}"
        )
    x1 = np.asarray(table["x1"], dtype=float).reshape(-1)
    if x1.size < 2 or not np.all(np.isfinite(x1)):
        raise ValueError("dataset-specific normalizer requires at least two finite x1 points")
    mean = np.asarray([np.mean(x1, dtype=float)], dtype=float)
    scale = np.asarray([_safe_scale(x1)], dtype=float)
    return mean, scale


def _set_model_normalizer(model: Any, mean: np.ndarray, scale: np.ndarray) -> None:
    with torch.no_grad():
        model.state_mean.copy_(
            torch.as_tensor(mean, dtype=model.state_mean.dtype, device=model.state_mean.device)
        )
        model.state_scale.copy_(
            torch.as_tensor(scale, dtype=model.state_scale.dtype, device=model.state_scale.device)
        )


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
    w0_start = float(times[int(contact_idx[0])]) if contact_idx.size else float(times[0])
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


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray, times: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(truth, dtype=float)
    t = np.asarray(times, dtype=float)
    numerator = float(np.sqrt(np.trapezoid(np.square(pred - ref), t)))
    denominator = max(float(np.sqrt(np.trapezoid(np.square(ref), t))), 1.0e-30)
    return 100.0 * numerator / denominator


def _pointwise_force(model: Any, x1: np.ndarray) -> np.ndarray:
    dtype = model.state_mean.dtype
    device = model.state_mean.device
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x1.size, POINTWISE_CHUNK_SIZE):
            stop = min(start + POINTWISE_CHUNK_SIZE, x1.size)
            states = torch.zeros((stop - start, 2), dtype=dtype, device=device)
            states[:, 0] = torch.as_tensor(x1[start:stop], dtype=dtype, device=device)
            out.append(model(states).detach().cpu().numpy())
    return np.concatenate(out)


def _full_rollout(
    model: Any,
    settings: AFM06aHardSampleInputs,
    times: np.ndarray,
    initial_state: np.ndarray,
    stage1_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = torch.float64
    times_t = torch.as_tensor(times, dtype=dtype)
    initial_t = torch.as_tensor(initial_state, dtype=dtype)
    with torch.no_grad():
        trajectory_t = rollout_single_shooting_torch(
            model,
            settings,
            initial_t,
            times_t,
            method=str(stage1_config["ode_method"]),
            rtol=float(stage1_config["ode_rtol"]),
            atol=float(stage1_config["ode_atol"]),
            ode_step_budget=int(stage1_config["ode_step_budget"]),
        )
        x2dot_t = x2dot_rhs_torch(trajectory_t, times_t, model, settings)
        force_states = torch.stack(
            (
                trajectory_t[0],
                torch.zeros(times_t.numel(), dtype=dtype),
            ),
            dim=1,
        )
        fts_t = model(force_states)
        bar_fts_t = fts_t / float(settings.mass_kg)
    return (
        trajectory_t.detach().cpu().numpy(),
        x2dot_t.detach().cpu().numpy(),
        bar_fts_t.detach().cpu().numpy(),
    )


def _plot_rollout_grid(
    *,
    path: Path,
    times: np.ndarray,
    contact: np.ndarray,
    true_x1: np.ndarray,
    true_bar_fts: np.ndarray,
    pred_x1: np.ndarray,
    pred_bar_fts: np.ndarray,
    window_duration_s: float,
) -> Path:
    windows = _window_indices(times, contact, window_duration_s=window_duration_s)
    fig, axes = plt.subplots(2, 4, figsize=(22.0, 8.6), sharex=False)
    for col, (title, idx) in enumerate(windows):
        plot_idx = _plot_indices(idx)
        time_ms = times[plot_idx] * 1.0e3
        axes[0, col].plot(time_ms, true_x1[plot_idx] * 1.0e9, color="black", linewidth=1.2, label="true")
        axes[0, col].plot(
            time_ms,
            pred_x1[plot_idx] * 1.0e9,
            color="crimson",
            linewidth=1.0,
            label="prediction",
        )
        axes[1, col].plot(time_ms, true_bar_fts[plot_idx], color="black", linewidth=1.2, label="true")
        axes[1, col].plot(
            time_ms,
            pred_bar_fts[plot_idx],
            color="crimson",
            linewidth=1.0,
            label="prediction",
        )
        axes[0, col].set_title(title, fontsize=15)
        for row in range(2):
            axes[row, col].grid(True, alpha=0.28, linewidth=0.8)
            axes[row, col].tick_params(axis="both", labelsize=12)
            axes[row, col].set_xlabel("Time [ms]", fontsize=13)
        if col == 0:
            axes[0, col].legend(loc="best", fontsize=11)
            axes[1, col].legend(loc="best", fontsize=11)
    axes[0, 0].set_ylabel(r"$x_1$ [nm]", fontsize=14)
    axes[1, 0].set_ylabel(r"$\bar{F}_{ts}$ [m s$^{-2}$]", fontsize=14)
    fig.tight_layout(pad=1.6)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pointwise_force_grid(
    *,
    path: Path,
    times: np.ndarray,
    contact: np.ndarray,
    true_bar_fts: np.ndarray,
    pointwise_bar_fts: np.ndarray,
    window_duration_s: float,
) -> Path:
    windows = _window_indices(times, contact, window_duration_s=window_duration_s)
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 4.2), sharex=False)
    for col, (title, idx) in enumerate(windows):
        plot_idx = _plot_indices(idx)
        time_ms = times[plot_idx] * 1.0e3
        axes[col].plot(time_ms, true_bar_fts[plot_idx], color="black", linewidth=1.2, label="true")
        axes[col].plot(
            time_ms,
            pointwise_bar_fts[plot_idx],
            color="crimson",
            linewidth=1.0,
            label="prediction",
        )
        axes[col].set_title(title, fontsize=15)
        axes[col].grid(True, alpha=0.28, linewidth=0.8)
        axes[col].tick_params(axis="both", labelsize=12)
        axes[col].set_xlabel("Time [ms]", fontsize=13)
        if col == 0:
            axes[col].set_ylabel(r"$\bar{F}_{ts}$ [m s$^{-2}$]", fontsize=14)
            axes[col].legend(loc="best", fontsize=11)
    fig.tight_layout(pad=1.6)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def _fresh_prediction_dir(dataset_root: Path) -> Path:
    prediction_dir = dataset_root / "prediction"
    if prediction_dir.is_dir():
        shutil.rmtree(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    return prediction_dir


def _same_resolved_path(left: Any, right: Path) -> bool:
    try:
        return Path(str(left)).resolve() == right.resolve()
    except Exception:
        return False


def _metadata_file_exists(metadata: dict[str, Any], section: str, key: str) -> bool:
    node = metadata.get(section)
    if not isinstance(node, dict):
        return False
    value = node.get(key)
    if not isinstance(value, str) or not value.strip():
        return False
    return Path(value).is_file()


def _existing_complete_prediction(
    *,
    dataset_root: Path,
    archive_dir: Path,
    reference_manifest: Path,
    normalizer_policy: str,
    model_state_dict_sha256: str,
) -> dict[str, Any] | None:
    metadata_path = dataset_root / "prediction" / "prediction_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    if not _same_resolved_path(loaded.get("archive_dir", ""), archive_dir):
        return None
    if not _same_resolved_path(loaded.get("reference_manifest", ""), reference_manifest):
        return None
    if not np.isclose(
        float(loaded.get("prediction_duration_s", float("nan"))),
        PREDICTION_DURATION_S,
        rtol=0.0,
        atol=1.0e-15,
    ):
        return None
    if str(loaded.get("model_state_dict_sha256", "")) != str(model_state_dict_sha256):
        return None
    normalizer = loaded.get("normalizer")
    if not isinstance(normalizer, dict) or str(normalizer.get("policy", "")) != str(normalizer_policy):
        return None
    required_outputs = (
        ("rollout", "data_path"),
        ("rollout", "plot_path"),
        ("pointwise", "data_path"),
        ("pointwise", "plot_path"),
    )
    if not all(_metadata_file_exists(loaded, section, key) for section, key in required_outputs):
        return None
    loaded["status"] = "complete"
    loaded["resume_action"] = "skipped_existing_complete_prediction"
    loaded["walltime_s"] = 0.0
    return loaded


def run_one_dataset(
    *,
    dataset_root: Path,
    restored: tuple[Any, Any, dict[str, Any], Path, Path],
    archive_dir: Path,
    reference: AFM06aHardSampleInputs,
    reference_manifest: Path,
    normalizer_policy: str = DATASET_NORMALIZER_POLICY,
    trained_state_mean: np.ndarray | None = None,
    trained_state_scale: np.ndarray | None = None,
) -> dict[str, Any]:
    model, endpoint, payload, stage2_result, stage1_result = restored
    archived_state_mean = (
        model.state_mean.detach().cpu().numpy().copy()
        if trained_state_mean is None
        else np.asarray(trained_state_mean, dtype=float).copy()
    )
    archived_state_scale = (
        model.state_scale.detach().cpu().numpy().copy()
        if trained_state_scale is None
        else np.asarray(trained_state_scale, dtype=float).copy()
    )
    data_path, table, parameters = _load_dataset(dataset_root)
    prediction_dir = _fresh_prediction_dir(dataset_root)

    all_times = np.asarray(table["t"], dtype=float)
    prediction_stop = float(all_times[0]) + PREDICTION_DURATION_S
    keep = np.flatnonzero(all_times <= prediction_stop + 1.0e-15)
    if keep.size < 2:
        raise RuntimeError(f"Dataset does not cover the required {PREDICTION_DURATION_S * 1e3:g} ms")
    times = all_times[keep]
    x1 = np.asarray(table["x1"], dtype=float)[keep]
    x2 = np.asarray(table["x2"], dtype=float)[keep]
    x2dot = np.asarray(table["x2dot"], dtype=float)[keep]
    contact = np.asarray(table["contact"], dtype=bool)[keep]
    true_bar_fts = np.asarray(table["bar_fts"], dtype=float)[keep]
    cropped_table = dict(table)
    cropped_table.update(t=times, x1=x1, x2=x2, x2dot=x2dot, contact=contact, bar_fts=true_bar_fts)
    settings = _settings_for_dataset(cropped_table, parameters, reference)
    settings.validate()
    stage1_config = endpoint.payload.get("config")
    if not isinstance(stage1_config, dict):
        raise RuntimeError("Archived st1pl endpoint has no saved configuration")
    window_duration_s = float(endpoint.window.times[-1] - endpoint.window.times[0])
    if not np.isfinite(window_duration_s) or window_duration_s <= 0.0:
        raise RuntimeError(f"Invalid archived training-window duration: {window_duration_s}")

    dataset_state_mean, dataset_state_scale = _normalizer_for_dataset(cropped_table, normalizer_policy)
    if dataset_state_mean is None or dataset_state_scale is None:
        applied_state_mean = archived_state_mean
        applied_state_scale = archived_state_scale
    else:
        applied_state_mean = dataset_state_mean
        applied_state_scale = dataset_state_scale
    _set_model_normalizer(model, applied_state_mean, applied_state_scale)

    rollout_states, rollout_x2dot, rollout_bar_fts = _full_rollout(
        model,
        settings,
        times,
        np.asarray([x1[0], x2[0]], dtype=float),
        stage1_config,
    )
    pointwise_fts = _pointwise_force(model, x1)
    pointwise_bar_fts = pointwise_fts / float(settings.mass_kg)
    true_fts = true_bar_fts * float(settings.mass_kg)

    rollout_data_path = prediction_dir / f"{dataset_root.name}_trained_kan_full_rollout.npz"
    pointwise_data_path = prediction_dir / f"{dataset_root.name}_trained_kan_pointwise_x1_to_fts.npz"
    np.savez_compressed(
        rollout_data_path,
        t=times,
        true_x1=x1,
        true_x2=x2,
        true_x2dot=x2dot,
        true_bar_fts=true_bar_fts,
        true_Fts=true_fts,
        pred_x1=rollout_states[0],
        pred_x2=rollout_states[1],
        pred_x2dot=rollout_x2dot,
        pred_bar_fts=rollout_bar_fts,
        pred_Fts=rollout_bar_fts * float(settings.mass_kg),
        contact=contact.astype(int),
        normalizer_policy=np.asarray(str(normalizer_policy)),
        applied_state_mean=applied_state_mean,
        applied_state_scale=applied_state_scale,
        archived_state_mean=archived_state_mean,
        archived_state_scale=archived_state_scale,
    )
    np.savez_compressed(
        pointwise_data_path,
        t=times,
        true_x1=x1,
        true_bar_fts=true_bar_fts,
        true_Fts=true_fts,
        pred_bar_fts=pointwise_bar_fts,
        pred_Fts=pointwise_fts,
        contact=contact.astype(int),
        normalizer_policy=np.asarray(str(normalizer_policy)),
        applied_state_mean=applied_state_mean,
        applied_state_scale=applied_state_scale,
        archived_state_mean=archived_state_mean,
        archived_state_scale=archived_state_scale,
    )

    rollout_plot_path = prediction_dir / f"{dataset_root.name}_trained_kan_full_rollout_x1_bar_fts_full_W0_W1_W2.png"
    pointwise_plot_path = prediction_dir / f"{dataset_root.name}_trained_kan_pointwise_bar_fts_full_W0_W1_W2.png"
    _plot_rollout_grid(
        path=rollout_plot_path,
        times=times,
        contact=contact,
        true_x1=x1,
        true_bar_fts=true_bar_fts,
        pred_x1=rollout_states[0],
        pred_bar_fts=rollout_bar_fts,
        window_duration_s=window_duration_s,
    )
    _plot_pointwise_force_grid(
        path=pointwise_plot_path,
        times=times,
        contact=contact,
        true_bar_fts=true_bar_fts,
        pointwise_bar_fts=pointwise_bar_fts,
        window_duration_s=window_duration_s,
    )

    metadata = {
        "dataset_root": str(dataset_root),
        "source_dataset": str(data_path),
        "reference_manifest": str(reference_manifest.resolve()),
        "prediction_time_span_s": [float(times[0]), float(times[-1])],
        "prediction_duration_s": float(times[-1] - times[0]),
        "prediction_duration_reference": (
            "matches stage2light afm_param_stage2light_06a_bar_fts_full_rollout_grid.png "
            "duration (10.0--10.5 ms)"
        ),
        "archive_dir": str(archive_dir.resolve()),
        "stage2_result": str(stage2_result),
        "stage1_result": str(stage1_result),
        "stage2_rank": int(endpoint.rank),
        "stage2_completed_epoch": int(payload.get("completed_epoch", -1)),
        "model_state_dict_sha256": str(payload.get("model_state_dict_sha256", "")),
        "q_effective": _scalar_from_table_or_params(table, parameters, "q_effective"),
        "q_multiplier": _scalar_from_table_or_params(table, parameters, "q_multiplier"),
        "fd_over_m": _scalar_from_table_or_params(table, parameters, "fd_over_m"),
        "fd_over_m_multiplier": _scalar_from_table_or_params(table, parameters, "fd_over_m_multiplier"),
        "actuation_scale": settings.actuation_scale,
        "d1": settings.d1,
        "d2": settings.d2,
        "dist_m": settings.dist,
        "dist_nm": settings.dist * 1.0e9,
        "normalizer": {
            "policy": str(normalizer_policy),
            "applied_state_mean": applied_state_mean.tolist(),
            "applied_state_scale": applied_state_scale.tolist(),
            "archived_state_mean": archived_state_mean.tolist(),
            "archived_state_scale": archived_state_scale.tolist(),
            "dataset_specific_state_mean": (
                None if dataset_state_mean is None else dataset_state_mean.tolist()
            ),
            "dataset_specific_state_scale": (
                None if dataset_state_scale is None else dataset_state_scale.tolist()
            ),
        },
        "rollout": {
            "definition": "autonomous full-time-span rollout from dataset initial state",
            "force_input_source": "predicted x1 during rollout",
            "data_path": str(rollout_data_path),
            "plot_path": str(rollout_plot_path),
            "x1_relative_rmse_pct": _relative_rmse_pct(rollout_states[0], x1, times),
            "bar_fts_relative_rmse_pct": _relative_rmse_pct(rollout_bar_fts, true_bar_fts, times),
        },
        "pointwise": {
            "definition": "feed every true x1 point to the trained KAN to predict physical Fts; no rollout/time integration",
            "force_input_source": "true x1 from this generated dataset",
            "data_path": str(pointwise_data_path),
            "plot_path": str(pointwise_plot_path),
            "bar_fts_relative_rmse_pct": _relative_rmse_pct(pointwise_bar_fts, true_bar_fts, times),
        },
    }
    _write_json(prediction_dir / "prediction_metadata.json", metadata)
    return metadata


def run_all(
    *,
    data_root: Path,
    archive_dir: Path,
    reference_manifest: Path,
    max_datasets: int | None,
    normalizer_policy: str,
    resume: bool,
) -> Path:
    dataset_roots = _dataset_roots(data_root.resolve())
    if max_datasets is not None:
        dataset_roots = dataset_roots[: int(max_datasets)]
    if not dataset_roots:
        raise RuntimeError(f"No q_XXX_Fd_YYY dataset folders found under {data_root}")

    reference = _load_reference_settings(reference_manifest.resolve())
    restored = restore_archived_model(archive_dir)
    model = restored[0]
    trained_state_mean = model.state_mean.detach().cpu().numpy().copy()
    trained_state_scale = model.state_scale.detach().cpu().numpy().copy()
    manifest_path = data_root.resolve() / "q_fd_trained_kan_prediction_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "data_root": str(data_root.resolve()),
        "archive_dir": str(archive_dir.resolve()),
        "reference_manifest": str(reference_manifest.resolve()),
        "prediction_duration_s": PREDICTION_DURATION_S,
        "num_datasets": len(dataset_roots),
        "overwrite_policy": (
            "complete prediction folders matching the current archive and normalizer are kept; "
            "missing or incomplete prediction folders are deleted and rewritten"
            if resume
            else "each dataset prediction folder is deleted and rewritten"
        ),
        "resume": bool(resume),
        "normalizer_policy": str(normalizer_policy),
        "trained_state_mean": trained_state_mean.tolist(),
        "trained_state_scale": trained_state_scale.tolist(),
        "datasets": [],
    }
    _write_json(manifest_path, manifest)
    started = time.perf_counter()
    for ordinal, dataset_root in enumerate(dataset_roots, start=1):
        print(f"[{ordinal:02d}/{len(dataset_roots)}] predicting {dataset_root.name}", flush=True)
        item_started = time.perf_counter()
        if resume:
            existing = _existing_complete_prediction(
                dataset_root=dataset_root,
                archive_dir=archive_dir,
                reference_manifest=reference_manifest,
                normalizer_policy=normalizer_policy,
                model_state_dict_sha256=str(restored[2].get("model_state_dict_sha256", "")),
            )
            if existing is not None:
                print(f"[{ordinal:02d}/{len(dataset_roots)}] skipping complete {dataset_root.name}", flush=True)
                manifest["datasets"].append(existing)
                _write_json(manifest_path, manifest)
                continue
        try:
            metadata = run_one_dataset(
                dataset_root=dataset_root,
                restored=restored,
                archive_dir=archive_dir,
                reference=reference,
                reference_manifest=reference_manifest,
                normalizer_policy=normalizer_policy,
                trained_state_mean=trained_state_mean,
                trained_state_scale=trained_state_scale,
            )
            metadata["status"] = "complete"
        except Exception as exc:
            metadata = {
                "dataset_root": str(dataset_root),
                "status": "failed",
                "error": repr(exc),
            }
            _write_json(dataset_root / "prediction_error.json", metadata)
            print(f"Failed {dataset_root.name}: {exc!r}", flush=True)
        metadata["walltime_s"] = time.perf_counter() - item_started
        manifest["datasets"].append(metadata)
        _write_json(manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["total_walltime_s"] = time.perf_counter() - started
    _write_json(manifest_path, manifest)
    print(f"Complete: {manifest_path}", flush=True)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument(
        "--normalizer-policy",
        choices=(TRAINED_NORMALIZER_POLICY, DATASET_NORMALIZER_POLICY),
        default=TRAINED_NORMALIZER_POLICY,
        help=(
            "Use the archived trained KAN normalizer or recompute the full-span "
            "x1 mean/std for each generated Q x Fd/m dataset."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip dataset prediction folders that already contain complete outputs "
            "for the current archive and normalizer policy."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all(
        data_root=args.data_root,
        archive_dir=args.archive_dir,
        reference_manifest=args.reference_manifest,
        max_datasets=args.max_datasets,
        normalizer_policy=str(args.normalizer_policy),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
