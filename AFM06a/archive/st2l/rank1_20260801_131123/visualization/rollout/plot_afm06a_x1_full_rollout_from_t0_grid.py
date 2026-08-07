"""Plot archived AFM06a x1 full rollout started at the true t=0 state.

This script intentionally uses only the self-contained archive beside it:
the archived stage2light model, the archived st1pl warmstart, and the
archived generated data table. It does not reuse the legacy first-contact
``from_W0`` rollout cache.
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
VIS_DIR = SCRIPT_DIR.parent
ARCHIVE_ROOT = VIS_DIR.parent
REPO_ROOT = ARCHIVE_ROOT.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.stage1pluslight.data import transition_resolved_source_indices
from AFM06a.stage2light.config import default_config
from AFM06a.stage2light.rollout import rollout_single_shooting_torch, x2dot_rhs_torch
from AFM06a.stage2light.runner.visualization._common import load_visualization_context
from AFM06a.stage2light.kan_backend import state_dict_digest

RESULT_PATH = ARCHIVE_ROOT / "result" / "afm_param_stage2light_06a_rank1.pkl"
STAGE1_PATH = ARCHIVE_ROOT / "conditional dependency" / "afm_param_stage1pluslight_06a.pkl"
DATA_PATH = ARCHIVE_ROOT / "data" / "pert_df_afm_dmt_hard.npz"
CACHE_PATH = SCRIPT_DIR / "afm06a_stage2light_full_rollout_from_t0_cache_rank1.pkl"
OUTPUT_PATH = SCRIPT_DIR / "afm_param_stage2light_06a_x1_full_rollout_from_W0_grid.png"
X2_CSV_PATH = SCRIPT_DIR / "afm_param_stage2light_06a_x1_full_rollout_from_W0_pred_x2_trajectory.csv"
X2_NPZ_PATH = SCRIPT_DIR / "afm_param_stage2light_06a_x1_full_rollout_from_W0_pred_x2_trajectory.npz"


def _archive_config():
    return replace(
        default_config(REPO_ROOT),
        stage1_result_path=STAGE1_PATH,
        result_dir=RESULT_PATH.parent,
        checkpoint_dir=ARCHIVE_ROOT / ".visualization_no_checkpoint",
        log_dir=ARCHIVE_ROOT / "logs",
        visualization_dir=VIS_DIR,
        archive_root=ARCHIVE_ROOT.parent,
    )


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_pickle_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(truth, dtype=float)
    return 100.0 * float(np.sqrt(np.mean(np.square(pred - ref)))) / max(
        float(np.sqrt(np.mean(np.square(ref)))), 1.0e-30
    )


def _training_style_panel_subset(
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


def _panel_indices(payload_times: np.ndarray, window_times: np.ndarray) -> list[tuple[str, np.ndarray]]:
    full = np.arange(payload_times.size, dtype=int)
    window_duration = float(window_times[-1] - window_times[0])
    initial = np.flatnonzero(payload_times <= float(payload_times[0]) + window_duration)
    middle_time = 0.5 * (float(payload_times[0]) + float(payload_times[-1]))
    middle = np.flatnonzero(np.abs(payload_times - middle_time) <= 0.5 * window_duration)
    tail = np.flatnonzero(payload_times >= float(payload_times[-1]) - window_duration)
    return [
        ("Full time-span", full),
        ("Initial window from t=0", initial),
        ("Observation window, middle", middle),
        ("Observation window, tail", tail),
    ]


def _build_rollout_from_t0() -> dict:
    if CACHE_PATH.is_file():
        return _load_pickle(CACHE_PATH)

    stage1 = _load_pickle(STAGE1_PATH)
    saved_config = stage1["config"]
    context = load_visualization_context(
        _archive_config(),
        preserve_io_paths=True,
        enforce_current_contract=False,
    )

    with np.load(DATA_PATH, allow_pickle=False) as data:
        all_times = np.asarray(data["t"], dtype=float)
        all_x1 = np.asarray(data["x1"], dtype=float)
        all_x2 = np.asarray(data["x2"], dtype=float)
        all_x2dot = np.asarray(data["x2dot"], dtype=float)
        all_contact = np.asarray(data["contact"], dtype=bool)
        all_bar_fts = np.asarray(data["bar_fts"], dtype=float)

    settings = context.endpoint.window.settings
    model_hash = state_dict_digest(context.model.frozen_state_dict())
    cache_contract = {
        "schema_version": 2,
        "start_source_idx": 0,
        "initial_state_source": "sampled_true_data_at_global_t0",
        "force_input_source": "predicted_x1_from_autonomous_rollout",
        "regime_source": "predicted_x1_from_autonomous_rollout",
        "model_state_dict_sha256": model_hash,
        "physics_parameter_vector": [
            float(value) for value in settings.parameter_vector
        ],
        "data_path": str(DATA_PATH.resolve()),
        "sample_stride": int(saved_config["sample_stride"]),
        "transition_half_width_s": float(saved_config["transition_half_width_s"]),
    }
    if CACHE_PATH.is_file():
        cached = _load_pickle(CACHE_PATH)
        if cached.get("cache_contract") == cache_contract:
            return cached

    source_idx = transition_resolved_source_indices(
        all_times,
        all_contact,
        start=0,
        stride=max(1, int(saved_config["sample_stride"])),
        transition_half_width_s=float(saved_config["transition_half_width_s"]),
    )
    times = all_times[source_idx]
    true_states = np.vstack((all_x1[source_idx], all_x2[source_idx]))

    times_t = torch.as_tensor(times, dtype=torch.float64)
    initial_state_t = torch.as_tensor(true_states[:, 0], dtype=torch.float64)
    with torch.no_grad():
        trajectory = rollout_single_shooting_torch(
            context.model,
            settings,
            initial_state_t,
            times_t,
            method=str(saved_config["ode_method"]),
            rtol=float(saved_config["ode_rtol"]),
            atol=float(saved_config["ode_atol"]),
            ode_step_budget=int(saved_config["ode_step_budget"]),
        )
        predicted_x2dot = x2dot_rhs_torch(
            trajectory,
            times_t,
            context.model,
            settings,
        )
        force_states = torch.stack(
            (
                trajectory[0],
                torch.zeros(times_t.numel(), dtype=torch.float64),
            ),
            dim=1,
        )
        predicted_bar_fts = context.model(force_states)

    payload = {
        "cache_contract": cache_contract,
        "source_idx": source_idx,
        "times": times,
        "true_states": true_states,
        "true_x2dot": all_x2dot[source_idx],
        "true_bar_fts": all_bar_fts[source_idx],
        "true_contact": all_contact[source_idx],
        "predicted_states": trajectory.detach().cpu().numpy(),
        "predicted_x2dot": predicted_x2dot.detach().cpu().numpy(),
        "predicted_bar_fts": predicted_bar_fts.detach().cpu().numpy(),
    }
    _write_pickle_atomic(CACHE_PATH, payload)
    return payload


def _save_predicted_x2(payload: dict) -> None:
    times = np.asarray(payload["times"], dtype=float)
    source_idx = np.asarray(payload["source_idx"], dtype=int)
    predicted_states = np.asarray(payload["predicted_states"], dtype=float)
    true_states = np.asarray(payload["true_states"], dtype=float)
    rows = zip(
        source_idx,
        times,
        1.0e3 * times,
        predicted_states[0],
        1.0e9 * predicted_states[0],
        predicted_states[1],
        1.0e3 * predicted_states[1],
        true_states[0],
        1.0e9 * true_states[0],
        true_states[1],
        1.0e3 * true_states[1],
        strict=True,
    )
    with X2_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_idx",
                "t_s",
                "t_ms",
                "pred_x1_m",
                "pred_x1_nm",
                "pred_x2_m_per_s",
                "pred_x2_mm_per_s",
                "true_x1_m",
                "true_x1_nm",
                "true_x2_m_per_s",
                "true_x2_mm_per_s",
            ]
        )
        writer.writerows(rows)
    np.savez_compressed(
        X2_NPZ_PATH,
        source_idx=source_idx,
        t_s=times,
        t_ms=1.0e3 * times,
        pred_x1_m=predicted_states[0],
        pred_x1_nm=1.0e9 * predicted_states[0],
        pred_x2_m_per_s=predicted_states[1],
        pred_x2_mm_per_s=1.0e3 * predicted_states[1],
        true_x1_m=true_states[0],
        true_x1_nm=1.0e9 * true_states[0],
        true_x2_m_per_s=true_states[1],
        true_x2_mm_per_s=1.0e3 * true_states[1],
    )


def _plot_x1(payload: dict) -> None:
    stage1 = _load_pickle(STAGE1_PATH)
    config = stage1["config"]
    window_times = np.asarray(stage1["times"], dtype=float)
    window_count = int(window_times.size)
    payload_times = np.asarray(payload["times"], dtype=float)
    payload_times_ms = 1.0e3 * payload_times
    contact = np.asarray(payload["true_contact"], dtype=bool)
    true_x1 = 1.0e9 * np.asarray(payload["true_states"], dtype=float)[0]
    pred_x1 = 1.0e9 * np.asarray(payload["predicted_states"], dtype=float)[0]

    with np.load(DATA_PATH, allow_pickle=False) as data:
        raw_times_ms = 1.0e3 * np.asarray(data["t"], dtype=float)
        raw_x1_nm = 1.0e9 * np.asarray(data["x1"], dtype=float)

    panels = _panel_indices(payload_times, window_times)
    panel_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    window_duration = float(window_times[-1] - window_times[0])
    for name, indices in panels:
        panel_indices = np.asarray(indices, dtype=int)
        if name.startswith("Observation window") or name == "Initial window from t=0":
            target_count = window_count
        else:
            panel_duration = float(
                payload_times[panel_indices[-1]] - payload_times[panel_indices[0]]
            )
            target_count = max(2, int(round(window_count * panel_duration / window_duration)))
        panel_indices = _training_style_panel_subset(
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
            f"relative RMSE = {_relative_rmse_pct(panel_prediction, panel_truth):.3f}%",
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
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    payload = _build_rollout_from_t0()
    _save_predicted_x2(payload)
    _plot_x1(payload)
    print(f"Saved x1 plot to: {OUTPUT_PATH}")
    print(f"Saved t=0 rollout cache to: {CACHE_PATH}")
    print(f"Saved predicted x2 CSV to: {X2_CSV_PATH}")
    print(f"Saved predicted x2 NPZ to: {X2_NPZ_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
