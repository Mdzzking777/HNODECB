"""Load one exact AFM06a st1pl endpoint for stage2light."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from AFM06a.stage1pluslight.checkpoint import load_checkpoint
from AFM06a.stage1pluslight.kan_backend import (
    GLOBAL_FORCE_OUTPUT_POLICY,
    GLOBAL_NORMALIZED_INPUT_POLICY,
    IDENTITY_FORCE_OUTPUT_POLICY,
    LOCAL_FORCE_OUTPUT_POLICY,
    METERS_PER_NANOMETER,
    NANOMETER_INPUT_POLICY,
    NORMALIZED_INPUT_POLICY,
    RAW_INPUT_POLICY,
    SINGLE_INPUT_KAN_WIDTH,
    training_force_normalizer,
    training_state_normalizer,
)
from AFM06a.stage1pluslight.data import (
    PreparedWindow,
    bar_fts_residual_from_observables,
    global_normalization_statistics,
    load_dataset,
)
from AFM06a.stage1pluslight.ranking import rank_trial_records
from AFM06a.stage2light.config import Stage2LightConfig


REQUIRED_WARMSTART_KEYS = {
    "initial_grid_support",
    "initial_state",
    "kan_construction",
    "kan_state_dict",
    "kan_state_dict_sha256",
    "nn_gain",
    "force_mean",
    "force_output_policy",
    "force_scale",
    "soft_mask",
    "state_mean",
    "state_scale",
}


@dataclass(frozen=True)
class Stage1Endpoint:
    rank: int
    record: dict[str, Any]
    warmstart: dict[str, Any]
    payload: dict[str, Any]
    window: PreparedWindow
    true_fts: np.ndarray
    true_bar_fts: np.ndarray


def _exact_window(payload: dict[str, Any]) -> tuple[PreparedWindow, np.ndarray, np.ndarray]:
    saved_config = payload.get("config")
    if not isinstance(saved_config, dict):
        raise RuntimeError("st1pl payload does not contain its saved configuration")
    dataset_root = Path(saved_config["dataset_root"])
    error_level = str(saved_config["error_level"])
    loaded = load_dataset(dataset_root, error_level)
    global_state_mean, global_state_scale, global_bar_fts_mean, global_bar_fts_scale = (
        global_normalization_statistics(loaded)
    )
    source_idx = np.asarray(payload["window_source_indices"], dtype=int)
    times = np.asarray(payload["times"], dtype=float)
    train_idx = np.asarray(payload["train_idx"], dtype=int)
    val_idx = np.asarray(payload["val_idx"], dtype=int)
    if source_idx.ndim != 1 or source_idx.size != times.size:
        raise RuntimeError("st1pl window indices/times are inconsistent")
    has_exact_window_arrays = all(
        key in payload
        for key in ("window_states", "window_x2dot", "window_bar_fts", "window_contact")
    )
    if has_exact_window_arrays:
        states = np.asarray(payload["window_states"], dtype=float)
        x2dot = np.asarray(payload["window_x2dot"], dtype=float)
        if "window_fts" in payload:
            fts = np.asarray(payload["window_fts"], dtype=float)
        else:
            fts = np.asarray(payload["window_bar_fts"], dtype=float) * float(loaded.settings.mass_kg)
        bar_fts = np.asarray(payload["window_bar_fts"], dtype=float)
        contact = np.asarray(payload["window_contact"], dtype=bool)
        if (
            states.shape != (2, times.size)
            or x2dot.shape != (times.size,)
            or fts.shape != (times.size,)
            or bar_fts.shape != (times.size,)
            or contact.shape != (times.size,)
        ):
            raise RuntimeError("saved st1pl exact window arrays have inconsistent shapes")
    else:
        table_times = np.asarray(loaded.table["t"], dtype=float)[source_idx]
        if not np.array_equal(table_times, times):
            raise RuntimeError("current dataset no longer reproduces the saved st1pl time window")
        states = np.asarray(loaded.states[:, source_idx], dtype=float)
        x2dot = np.asarray(loaded.table["x2dot"], dtype=float)[source_idx]
        fts = np.asarray(loaded.table["Fts"], dtype=float)[source_idx]
        bar_fts = np.asarray(loaded.table["bar_fts"], dtype=float)[source_idx]
        contact = np.asarray(loaded.table["contact"], dtype=bool)[source_idx]
    window = PreparedWindow(
        states=states,
        times=times,
        x2dot=x2dot,
        fts=fts,
        bar_fts=bar_fts,
        contact=contact,
        train_idx=train_idx,
        val_idx=val_idx,
        source_idx=source_idx,
        bar_fts_rhs_residual=bar_fts_residual_from_observables(
            states,
            x2dot,
            times,
            settings=loaded.settings,
        ),
        global_state_mean=global_state_mean,
        global_state_scale=global_state_scale,
        global_bar_fts_mean=global_bar_fts_mean,
        global_bar_fts_scale=global_bar_fts_scale,
        settings=loaded.settings,
    )
    true_fts = window.fts
    true_bar_fts = window.bar_fts
    return window, true_fts, true_bar_fts


def load_stage1_endpoint(
    config: Stage2LightConfig,
    *,
    enforce_current_contract: bool = True,
) -> Stage1Endpoint:
    payload = load_checkpoint(config.stage1_result_path)
    if payload is None:
        raise FileNotFoundError(config.stage1_result_path)
    saved_config = payload.get("config")
    if not isinstance(saved_config, dict):
        raise RuntimeError("st1pl payload does not contain its saved configuration")
    saved_width = tuple(int(value) for value in saved_config.get("kan_width", ()))
    if enforce_current_contract and saved_width != SINGLE_INPUT_KAN_WIDTH:
        raise RuntimeError(
            "AFM06a stage2light requires a fresh single-input 1-3-1 st1pl result; "
            f"saved KAN width is {saved_width}"
        )
    if enforce_current_contract and (
        str(saved_config.get("error_level", "")) != "e0.0_real"
        or abs(float(saved_config.get("window_start_s", float("nan"))) - 0.010000000) > 1.0e-15
        or abs(float(saved_config.get("window_stop_s", float("nan"))) - 0.010013328) > 1.0e-15
    ):
        raise RuntimeError(
            "AFM06a stage2light requires a fresh st1pl endpoint from "
            "datasets/e0.0_real W1=10.000-10.013328 ms"
        )
    if len(saved_width) < 2 or saved_width[0] <= 0 or saved_width[-1] != 1:
        raise RuntimeError(f"saved AFM06a KAN width is invalid: {saved_width}")
    input_dimension = int(saved_width[0])
    if enforce_current_contract and (
        bool(saved_config.get("gain_enabled", True))
        or bool(saved_config.get("gain_learnable", True))
    ):
        raise RuntimeError(
            "AFM06a stage2light requires a fresh st1pl endpoint with Gain disabled"
        )
    if enforce_current_contract and (
        str(saved_config.get("sampling_policy", "")) != "full_resolution_all_train"
        or not bool(saved_config.get("all_points_training", False))
    ):
        raise RuntimeError(
            "AFM06a stage2light requires a fresh full-resolution 16 ns "
            "all-training st1pl endpoint"
        )
    if enforce_current_contract and (
        str(saved_config.get("loss_policy", ""))
        != "force_scaled_dynamics_residual_true_x1_residual_smooth"
    ):
        raise RuntimeError(
            "AFM06a stage2light requires a fresh st1pl endpoint using the "
            "force-scale-normalized r_dyn plus within-regime residual-smoothness loss "
            "with observed x1"
        )
    if enforce_current_contract and (
        float(saved_config.get("smoothness_loss_weight", -1.0)) < 0.0
    ):
        raise RuntimeError(
            "AFM06a stage2light requires a fresh st1pl endpoint with a "
            "nonnegative residual-smoothness loss weight"
        )
    if enforce_current_contract and (
        bool(saved_config.get("soft_mask_enabled", True))
        or bool(saved_config.get("soft_mask_trainable", True))
    ):
        raise RuntimeError(
            "AFM06a stage2light requires a fresh st1pl endpoint with Soft Mask disabled"
        )
    if not bool(payload.get("stage1plus_complete")) or not bool(payload.get("stage1plus_merged")):
        raise RuntimeError("AFM06a stage2light requires a complete merged st1pl result")
    records = payload.get("trial_parameters")
    if not isinstance(records, list):
        raise RuntimeError("st1pl trial_parameters is missing or invalid")
    ranked = rank_trial_records(records, viable_only=True)
    if config.input_rank > len(ranked):
        raise IndexError(f"requested rank {config.input_rank}, but only {len(ranked)} viable st1pl trials exist")
    record = ranked[config.input_rank - 1]
    saved_warmstart = record.get("warmstart")
    if not isinstance(saved_warmstart, dict):
        raise RuntimeError(f"st1pl rank {config.input_rank} has no complete warmstart")
    warmstart = dict(saved_warmstart)
    if not enforce_current_contract:
        construction = warmstart.get("kan_construction")
        construction_output_policy = (
            construction.get("force_output_policy", IDENTITY_FORCE_OUTPUT_POLICY)
            if isinstance(construction, dict)
            else IDENTITY_FORCE_OUTPUT_POLICY
        )
        warmstart.setdefault("force_output_policy", construction_output_policy)
        warmstart.setdefault("force_mean", 0.0)
        warmstart.setdefault("force_scale", 1.0)
    missing = sorted(REQUIRED_WARMSTART_KEYS.difference(warmstart))
    if missing:
        raise RuntimeError(f"st1pl rank {config.input_rank} warmstart is incomplete: {missing}")
    nn_gain = float(warmstart["nn_gain"])
    if not np.isfinite(nn_gain) or nn_gain <= 0.0:
        raise RuntimeError(
            f"st1pl rank {config.input_rank} has an invalid initialized Gain: {nn_gain}"
        )
    soft_mask = warmstart["soft_mask"]
    if not isinstance(soft_mask, dict):
        raise RuntimeError(f"st1pl rank {config.input_rank} has no saved Soft Mask state")
    if enforce_current_contract and bool(soft_mask.get("enabled", True)):
        raise RuntimeError(
            f"st1pl rank {config.input_rank} unexpectedly contains an enabled Soft Mask"
        )
    if enforce_current_contract and bool(soft_mask.get("trainable_in_st1pl", True)):
        raise RuntimeError(
            f"st1pl rank {config.input_rank} contains trainable Soft Mask parameters"
        )
    window, true_fts, true_bar_fts = _exact_window(payload)
    if not np.array_equal(np.asarray(warmstart["initial_state"], dtype=float), window.states[:, 0]):
        raise RuntimeError("saved warmstart initial state does not match the st1pl window")
    input_policy = str(saved_config.get("normalizer_policy", ""))
    if enforce_current_contract and input_policy != NORMALIZED_INPUT_POLICY:
        raise RuntimeError(
            "AFM06a stage2light requires a single-input training_mean_std "
            "x1 st1pl endpoint from the selected W1 window; "
            f"saved policy is {input_policy!r}"
        )
    training_states = np.asarray(window.states[:, window.train_idx].T, dtype=float)
    saved_state_mean = np.asarray(warmstart["state_mean"], dtype=float).reshape(-1)
    saved_state_scale = np.asarray(warmstart["state_scale"], dtype=float).reshape(-1)
    if not enforce_current_contract:
        if (
            saved_state_mean.shape != (input_dimension,)
            or saved_state_scale.shape != (input_dimension,)
            or not np.all(np.isfinite(saved_state_mean))
            or not np.all(np.isfinite(saved_state_scale))
            or np.any(saved_state_scale <= 1.0e-30)
        ):
            raise RuntimeError(
                "archived st1pl input transform is invalid: "
                f"mean={saved_state_mean}, scale={saved_state_scale}"
            )
        expected_state_mean = saved_state_mean
        expected_state_scale = saved_state_scale
    elif input_policy == NORMALIZED_INPUT_POLICY:
        expected_state_mean, expected_state_scale = training_state_normalizer(training_states)
        expected_state_mean = np.asarray(expected_state_mean, dtype=float).reshape(1)
        expected_state_scale = np.asarray(expected_state_scale, dtype=float).reshape(1)
    elif input_policy == GLOBAL_NORMALIZED_INPUT_POLICY:
        expected_state_mean = np.asarray(
            [window.global_state_mean[0]],
            dtype=float,
        )
        expected_state_scale = np.asarray(
            [window.global_state_scale[0]],
            dtype=float,
        )
    elif input_policy == RAW_INPUT_POLICY:
        expected_state_mean = np.zeros(input_dimension, dtype=float)
        expected_state_scale = np.ones(input_dimension, dtype=float)
    elif input_policy == NANOMETER_INPUT_POLICY:
        expected_state_mean = np.zeros(input_dimension, dtype=float)
        expected_state_scale = np.full(
            input_dimension,
            METERS_PER_NANOMETER,
            dtype=float,
        )
    else:
        raise RuntimeError(f"unsupported saved AFM06a input policy: {input_policy!r}")
    np.testing.assert_allclose(
        saved_state_mean,
        expected_state_mean,
        rtol=1.0e-14,
        atol=0.0,
        err_msg="saved st1pl state mean no longer matches its normalization policy",
    )
    np.testing.assert_allclose(
        saved_state_scale,
        expected_state_scale,
        rtol=1.0e-14,
        atol=0.0,
        err_msg="saved st1pl state scale no longer matches its normalization policy",
    )
    output_policy = str(warmstart["force_output_policy"])
    construction = warmstart.get("kan_construction")
    construction_output_policy = (
        str(construction.get("force_output_policy", ""))
        if isinstance(construction, dict)
        else ""
    )
    construction_output_quantity = (
        str(construction.get("force_output_quantity", ""))
        if isinstance(construction, dict)
        else ""
    )
    if construction_output_policy and construction_output_policy != output_policy:
        raise RuntimeError(
            "st1pl warmstart has inconsistent force-output policies: "
            f"warmstart={output_policy!r}, construction={construction_output_policy!r}"
        )
    if enforce_current_contract and construction_output_quantity != "Fts_N":
        raise RuntimeError(
            "AFM06a stage2light requires a fresh st1pl endpoint whose KAN "
            "output quantity is physical Fts [N]; "
            f"saved quantity is {construction_output_quantity!r}"
        )
    if enforce_current_contract and output_policy != LOCAL_FORCE_OUTPUT_POLICY:
        raise RuntimeError(
            "AFM06a stage2light requires a fresh st1pl endpoint whose KAN "
            "raw head is mapped to physical Fts [N] using the selected-window "
            "training_mean_std inverse; "
            f"saved policy is {output_policy!r}"
        )
    saved_force_mean = float(warmstart["force_mean"])
    saved_force_scale = float(warmstart["force_scale"])
    if not enforce_current_contract:
        if (
            output_policy
            not in {
                LOCAL_FORCE_OUTPUT_POLICY,
                GLOBAL_FORCE_OUTPUT_POLICY,
                IDENTITY_FORCE_OUTPUT_POLICY,
            }
            or not np.isfinite(saved_force_mean)
            or not np.isfinite(saved_force_scale)
            or saved_force_scale <= 1.0e-30
        ):
            raise RuntimeError(
                "archived st1pl force-output transform is invalid: "
                f"policy={output_policy!r}, mean={saved_force_mean}, "
                f"scale={saved_force_scale}"
            )
        expected_force_mean = saved_force_mean
        expected_force_scale = saved_force_scale
    elif output_policy == LOCAL_FORCE_OUTPUT_POLICY:
        expected_force_mean, expected_force_scale = training_force_normalizer(
            window.fts[window.train_idx]
        )
    elif output_policy == GLOBAL_FORCE_OUTPUT_POLICY:
        expected_force_mean = window.global_bar_fts_mean
        expected_force_scale = window.global_bar_fts_scale
    elif output_policy == IDENTITY_FORCE_OUTPUT_POLICY:
        expected_force_mean = 0.0
        expected_force_scale = 1.0
    else:
        raise RuntimeError(f"unsupported saved AFM06a force-output policy: {output_policy!r}")
    np.testing.assert_allclose(
        saved_force_mean,
        expected_force_mean,
        rtol=1.0e-14,
        atol=0.0,
        err_msg="saved st1pl force mean no longer matches its output policy",
    )
    np.testing.assert_allclose(
        saved_force_scale,
        expected_force_scale,
        rtol=1.0e-14,
        atol=0.0,
        err_msg="saved st1pl force scale no longer matches its output policy",
    )
    return Stage1Endpoint(
        rank=int(config.input_rank),
        record=record,
        warmstart=warmstart,
        payload=payload,
        window=window,
        true_fts=true_fts,
        true_bar_fts=true_bar_fts,
    )


__all__ = ["REQUIRED_WARMSTART_KEYS", "Stage1Endpoint", "load_stage1_endpoint"]
