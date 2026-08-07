"""AFM06a two-state dataset loading, windowing, and split construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from AFM06a.datasets.non_perturbed_dataset_generator import load_table_npz
from AFM06a.stage1pluslight.config import Stage1PlusLightConfig
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (
    AFM06aHardSampleInputs,
    B1,
)


@dataclass(frozen=True)
class LoadedDataset:
    states: np.ndarray
    table: dict[str, np.ndarray]
    generation_manifest: dict[str, Any]
    settings: AFM06aHardSampleInputs


@dataclass(frozen=True)
class PreparedWindow:
    states: np.ndarray
    times: np.ndarray
    x2dot: np.ndarray
    fts: np.ndarray
    bar_fts: np.ndarray
    contact: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    source_idx: np.ndarray
    bar_fts_rhs_residual: np.ndarray
    global_state_mean: np.ndarray
    global_state_scale: np.ndarray
    global_bar_fts_mean: float
    global_bar_fts_scale: float
    settings: AFM06aHardSampleInputs


def make_train_val_masks(n: int, val_stride: int, val_offset: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("at least two samples are required")
    if val_stride <= 1 or not (1 <= val_offset <= val_stride):
        raise ValueError("val_stride must exceed 1 and val_offset must be in 1..val_stride")
    idx = np.arange(n, dtype=int)
    val = idx[((idx + 1 - val_offset) % val_stride) == 0]
    train_mask = np.ones(n, dtype=bool)
    train_mask[val] = False
    return idx[train_mask], val


def make_all_train_mask(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("at least two samples are required")
    return np.arange(n, dtype=int), np.empty(0, dtype=int)


def build_regime_sampling_weights(
    times: np.ndarray,
    contact: np.ndarray,
    *,
    noncontact_weight: float,
    contact_weight: float,
    transition_weight: float,
    transition_half_width_s: float,
) -> np.ndarray:
    """Build point-density weights on a dense, strictly increasing time grid."""

    times = np.asarray(times, dtype=float).reshape(-1)
    contact = np.asarray(contact, dtype=bool).reshape(-1)
    if times.size < 2 or contact.shape != times.shape:
        raise ValueError("times/contact must be equal-length vectors with at least two points")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")
    if min(noncontact_weight, contact_weight, transition_weight) <= 0.0:
        raise ValueError("sampling-density weights must be positive")
    if transition_half_width_s < 0.0:
        raise ValueError("transition half-width cannot be negative")

    weights = np.where(contact, float(contact_weight), float(noncontact_weight))
    switch_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1
    for index in switch_idx:
        boundary_time = 0.5 * (times[index - 1] + times[index])
        near_transition = np.abs(times - boundary_time) <= float(transition_half_width_s)
        weights[near_transition] = float(transition_weight)
    return np.asarray(weights, dtype=float)


def _regime_masks(
    times: np.ndarray,
    contact: np.ndarray,
    *,
    transition_half_width_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float).reshape(-1)
    contact = np.asarray(contact, dtype=bool).reshape(-1)
    transition = np.zeros(times.shape, dtype=bool)
    switch_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1
    for index in switch_idx:
        boundary_time = 0.5 * (float(times[index - 1]) + float(times[index]))
        transition |= np.abs(times - boundary_time) <= float(transition_half_width_s)
    return transition, contact & ~transition, ~contact & ~transition


def _fixed_count_weighted_positions(density: np.ndarray, target_count: int) -> np.ndarray:
    density = np.asarray(density, dtype=float)
    if target_count < 2 or target_count > density.size:
        raise ValueError("invalid fixed sampling target count")
    cumulative = np.cumsum(density)
    targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), int(target_count))
    positions = np.searchsorted(cumulative, targets, side="left").astype(int)
    positions[0] = 0
    positions[-1] = density.size - 1
    for i in range(1, positions.size):
        positions[i] = max(positions[i], positions[i - 1] + 1)
    for i in range(positions.size - 2, -1, -1):
        positions[i] = min(positions[i], positions[i + 1] - 1)
    if positions[0] < 0 or positions[-1] >= density.size or np.any(np.diff(positions) <= 0):
        raise RuntimeError("failed to construct a strictly increasing weighted AFM06a sample grid")
    return positions


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
    counts = _allocate_counts(np.asarray([run.size for run in contact_runs], dtype=int), extra_count)
    pieces: list[np.ndarray] = []
    for run, count in zip(contact_runs, counts, strict=True):
        if count <= 0:
            continue
        left = float(local_times[run[0]])
        right = float(local_times[run[-1]])
        if right > left:
            pieces.append(np.linspace(left, right, count + 2, dtype=float)[1:-1])
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=float)


def _nearest_indices(times_all: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(times_all, sample_times, side="left")
    right = np.clip(right, 0, times_all.size - 1)
    left = np.clip(right - 1, 0, times_all.size - 1)
    choose_left = np.abs(sample_times - times_all[left]) <= np.abs(times_all[right] - sample_times)
    return np.where(choose_left, left, right).astype(int)


def transition_resolved_source_indices(
    times: np.ndarray,
    contact: np.ndarray,
    *,
    start: int,
    stride: int,
    transition_half_width_s: float,
) -> np.ndarray:
    """Combine a regular time grid with dense samples around contact switches."""

    times = np.asarray(times, dtype=float)
    contact = np.asarray(contact, dtype=bool)
    if times.ndim != 1 or contact.shape != times.shape:
        raise ValueError("times and contact must be matching one-dimensional arrays")
    if start < 0 or start >= times.size:
        raise ValueError("start lies outside the available time grid")
    if stride <= 0 or transition_half_width_s < 0.0:
        raise ValueError("stride must be positive and transition width nonnegative")

    regular = np.arange(start, times.size, stride, dtype=int)
    switch_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1
    dense_parts: list[np.ndarray] = []
    for index in switch_idx[switch_idx >= start]:
        boundary_time = 0.5 * (float(times[index - 1]) + float(times[index]))
        left = max(
            start,
            int(np.searchsorted(times, boundary_time - transition_half_width_s, side="left")),
        )
        right = int(
            np.searchsorted(times, boundary_time + transition_half_width_s, side="right")
        )
        if right > left:
            dense_parts.append(np.arange(left, right, dtype=int))

    pieces = [regular, np.asarray([times.size - 1], dtype=int), *dense_parts]
    return np.unique(np.concatenate(pieces))


def select_window_source_indices(
    config: Stage1PlusLightConfig,
    loaded: LoadedDataset,
) -> np.ndarray:
    """Select a fixed-size uniform or regime-weighted AFM06a time grid."""

    if config.window_start_s is None or config.window_stop_s is None or config.sample_stride is None:
        raise RuntimeError("window start, stop, and sample stride must be selected before preparing AFM06a data")
    times_all = np.asarray(loaded.table["t"], dtype=float)
    window_idx = np.flatnonzero(
        (times_all >= float(config.window_start_s)) & (times_all <= float(config.window_stop_s))
    )
    if window_idx.size < 2:
        raise ValueError("selected AFM06a training window contains fewer than two raw points")

    if config.sampling_policy == "full_resolution_all_train":
        return window_idx

    explicit_count = config.sample_count
    target_count = (
        int(explicit_count)
        if explicit_count is not None
        else int(window_idx[:: int(config.sample_stride)].size)
    )
    if target_count < 2 or target_count > window_idx.size:
        raise ValueError("invalid fixed sampling target count")
    if config.sampling_policy == "uniform":
        if explicit_count is None:
            return window_idx[:: int(config.sample_stride)]
        positions = np.rint(np.linspace(0, window_idx.size - 1, target_count)).astype(int)
        positions[0] = 0
        positions[-1] = window_idx.size - 1
        if np.unique(positions).size != positions.size:
            raise RuntimeError("failed to construct a strictly increasing uniform AFM06a sample grid")
        return window_idx[positions]
    if config.sampling_policy != "regime_weighted_fixed_count":
        raise ValueError(f"unsupported AFM06a sampling policy: {config.sampling_policy}")

    window_times = times_all[window_idx]
    window_contact = np.asarray(loaded.table["contact"], dtype=bool)[window_idx]
    density = build_regime_sampling_weights(
        window_times,
        window_contact,
        noncontact_weight=config.noncontact_sampling_weight,
        contact_weight=config.contact_sampling_weight,
        transition_weight=config.transition_sampling_weight,
        transition_half_width_s=config.transition_half_width_s,
    )
    positions = _fixed_count_weighted_positions(density, target_count)
    return window_idx[positions]


def select_window_sample_times(
    config: Stage1PlusLightConfig,
    loaded: LoadedDataset,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return sampled times, nearest source indices, and whether interpolation is required."""

    if config.sampling_policy != "regime_weighted_contact_doubled":
        selected = select_window_source_indices(config, loaded)
        return np.asarray(loaded.table["t"], dtype=float)[selected], selected, False

    if config.window_start_s is None or config.window_stop_s is None or config.sample_stride is None:
        raise RuntimeError("window start, stop, and sample stride must be selected before preparing AFM06a data")
    times_all = np.asarray(loaded.table["t"], dtype=float)
    contact_all = np.asarray(loaded.table["contact"], dtype=bool)
    window_idx = np.flatnonzero(
        (times_all >= float(config.window_start_s)) & (times_all <= float(config.window_stop_s))
    )
    if window_idx.size < 2:
        raise ValueError("selected AFM06a training window contains fewer than two raw points")
    target_count = (
        int(config.sample_count)
        if config.sample_count is not None
        else int(window_idx[:: int(config.sample_stride)].size)
    )
    window_times = times_all[window_idx]
    window_contact = contact_all[window_idx]
    base_contact_weight = max(float(config.noncontact_sampling_weight), float(config.contact_sampling_weight) / 2.0)
    base_density = build_regime_sampling_weights(
        window_times,
        window_contact,
        noncontact_weight=config.noncontact_sampling_weight,
        contact_weight=base_contact_weight,
        transition_weight=config.transition_sampling_weight,
        transition_half_width_s=config.transition_half_width_s,
    )
    base_positions = _fixed_count_weighted_positions(base_density, target_count)
    _transition_mask, contact_mask, _noncontact_mask = _regime_masks(
        window_times,
        window_contact,
        transition_half_width_s=config.transition_half_width_s,
    )
    base_contact_count = int(np.count_nonzero(contact_mask[base_positions]))
    extra_contact_times = _interpolated_contact_times(window_times, contact_mask, base_contact_count)
    sample_times = np.sort(np.concatenate([window_times[base_positions], extra_contact_times]))
    return sample_times, _nearest_indices(times_all, sample_times), True


def dataset_paths(dataset_root: Path, error_level: str) -> dict[str, Path]:
    data_dir = Path(dataset_root) / error_level / "data"
    return {
        "data_dir": data_dir,
        "ode_data": data_dir / "ode_data_afm_dmt_hard.npz",
        "pert_df": data_dir / "pert_df_afm_dmt_hard.npz",
        "generation_manifest": data_dir / "afm06a_generation_manifest.json",
    }


def settings_from_generation_manifest(manifest: dict[str, Any]) -> AFM06aHardSampleInputs:
    """Reconstruct the exact AFM06a physics settings used to generate a dataset."""

    params = manifest.get("parameter_vector")
    if not isinstance(params, list):
        return AFM06aHardSampleInputs()
    initial_state = manifest.get("initial_state", [0.0, 0.0])
    if not isinstance(initial_state, list) or len(initial_state) < 2:
        initial_state = [0.0, 0.0]
    common = {
        "x1_start": float(initial_state[0]),
        "x2_start": float(initial_state[1]),
        "initial_time": float(manifest.get("initial_time_s", 0.0)),
        "end_time": float(manifest.get("end_time_s", AFM06aHardSampleInputs().end_time)),
        "data_nsteps": int(manifest.get("nsteps", AFM06aHardSampleInputs().data_nsteps)),
    }
    parameter_names = manifest.get("parameter_names")
    if isinstance(parameter_names, list) and list(parameter_names[:10]) == [
        "k",
        "omega0",
        "m",
        "c",
        "Fd",
        "CA",
        "CH",
        "dist",
        "a0",
        "beta",
    ]:
        k_n_m, omega0, mass_kg, c_n_s_m, fd_n, ca, ch, dist, a0, beta = (
            float(value) for value in params[:10]
        )
        damping_ratio = c_n_s_m / (mass_kg * omega0)
        return AFM06aHardSampleInputs(
            beta=beta,
            d1=damping_ratio,
            d2=damping_ratio,
            k_n_m=k_n_m,
            mass_kg=mass_kg,
            c_n_s_m=c_n_s_m,
            ca=ca,
            ch=ch,
            fd_n=fd_n,
            omega0=omega0,
            dist_m=dist,
            a0_m=a0,
            **common,
        )

    if isinstance(parameter_names, list) and list(parameter_names[:10]) == [
        "k",
        "omega0",
        "m",
        "c",
        "Fd",
        "CA_F",
        "CH_F",
        "dist",
        "a0",
        "beta",
    ]:
        k_n_m, omega0, mass_kg, c_n_s_m, fd_n, ca_f, ch_f, dist, a0, beta = (
            float(value) for value in params[:10]
        )
        damping_ratio = c_n_s_m / (mass_kg * omega0)
        return AFM06aHardSampleInputs(
            beta=beta,
            d1=damping_ratio,
            d2=damping_ratio,
            k_n_m=k_n_m,
            mass_kg=mass_kg,
            c_n_s_m=c_n_s_m,
            ca=ca_f / mass_kg,
            ch=ch_f / mass_kg,
            fd_n=fd_n,
            omega0=omega0,
            dist_m=dist,
            a0_m=a0,
            **common,
        )

    if len(params) < 12:
        return AFM06aHardSampleInputs(**common)
    omega0, c1, c2, b1, d1, d2, eta, y_bar, omega_bar, dist, a0, beta = (
        float(value) for value in params[:12]
    )
    actuation_scale = float(b1) / float(B1)
    k_n_m = AFM06aHardSampleInputs().k_n_m
    mass_kg = k_n_m / (omega0**2)
    c_n_s_m = mass_kg * omega0 * float(d1)
    fd_n = (
        (omega0**2)
        * float(eta)
        * float(b1)
        * (float(omega_bar) ** 2)
        * float(y_bar)
        * mass_kg
    )
    ca = float(c1) * (omega0**2) * (float(eta) ** 3)
    ch = float(c2) * (omega0**2) / np.sqrt(float(eta))
    return AFM06aHardSampleInputs(
        beta=beta,
        d1=d1,
        d2=d2,
        k_n_m=k_n_m,
        mass_kg=mass_kg,
        c_n_s_m=c_n_s_m,
        ca=ca,
        ch=ch,
        fd_n=fd_n,
        omega0=omega0,
        actuation_scale=actuation_scale,
        dist_m=dist,
        a0_m=a0,
        **common,
    )


def load_dataset(dataset_root: Path, error_level: str) -> LoadedDataset:
    paths = dataset_paths(dataset_root, error_level)
    required_path_keys = ("ode_data", "pert_df")
    missing = [str(paths[key]) for key in required_path_keys if not paths[key].is_file()]
    if missing:
        raise FileNotFoundError("AFM06a dataset is incomplete:\n  " + "\n  ".join(missing))
    manifest: dict[str, Any] = {}
    if paths["generation_manifest"].is_file():
        manifest = json.loads(paths["generation_manifest"].read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("AFM06a generation manifest must be a JSON object")
    settings = settings_from_generation_manifest(manifest)
    settings.validate()
    with np.load(paths["ode_data"]) as payload:
        states = np.asarray(payload["data"], dtype=float)
    table = load_table_npz(str(paths["pert_df"]))
    if states.ndim != 2 or states.shape[0] != 2:
        raise ValueError(f"AFM06a requires state matrix shape (2, N), got {states.shape}")
    if "Fts" not in table and "fts_N" in table:
        table["Fts"] = np.asarray(table["fts_N"], dtype=float)
    if "Fts" not in table and "bar_fts" in table:
        table["Fts"] = np.asarray(table["bar_fts"], dtype=float) * float(settings.mass_kg)
    if "bar_fts" not in table and "Fts" in table:
        table["bar_fts"] = np.asarray(table["Fts"], dtype=float) / float(settings.mass_kg)
    # AFM06a trains the KAN to output the physical force Fts [N].  The
    # mass-scaled bar_fts = Fts/m is retained for RHS diagnostics and plots.
    required = {"t", "x1", "x2", "x2dot", "Fts", "bar_fts", "contact"}
    absent = sorted(required.difference(table))
    if absent:
        raise ValueError(f"AFM06a table is missing columns: {absent}")
    n = states.shape[1]
    if any(np.asarray(table[name]).shape != (n,) for name in required):
        raise ValueError("AFM06a table columns do not match the state trajectory length")
    if not np.allclose(states[0], table["x1"], rtol=0.0, atol=0.0):
        raise ValueError("x1 table/state mismatch")
    if not np.allclose(states[1], table["x2"], rtol=0.0, atol=0.0):
        raise ValueError("x2 table/state mismatch")
    return LoadedDataset(
        states=states,
        table=table,
        generation_manifest=manifest,
        settings=settings,
    )


def _safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = 1.0
    return scale


def bar_fts_residual_from_observables(
    states: np.ndarray,
    x2dot: np.ndarray,
    times: np.ndarray,
    *,
    settings: AFM06aHardSampleInputs | None = None,
) -> np.ndarray:
    """Reconstruct the AFM06a mass-scaled interaction from its own RHS.

    This signal may support gain initialization or offline diagnostics, but it
    is neither a pointwise force label nor a force-range loss reference.
    """

    settings = AFM06aHardSampleInputs() if settings is None else settings
    settings.validate()
    states = np.asarray(states, dtype=float)
    x2dot = np.asarray(x2dot, dtype=float)
    times = np.asarray(times, dtype=float)
    if states.ndim != 2 or states.shape[0] != 2:
        raise ValueError(f"states must have shape (2, N), got {states.shape}")
    if x2dot.shape != (states.shape[1],) or times.shape != (states.shape[1],):
        raise ValueError("x2dot/times must match the AFM06a state trajectory length")
    k_n_m, omega0, mass_kg, c_n_s_m, fd_n, _ca, _ch, _dist, _a0, _beta = (
        settings.parameter_vector
    )
    x1 = states[0]
    x2 = states[1]
    actuation = fd_n / mass_kg
    return np.asarray(
        x2dot
        - actuation * np.sin(omega0 * times)
        + (k_n_m / mass_kg) * x1
        + (c_n_s_m / mass_kg) * x2,
        dtype=float,
    )


def fts_residual_from_observables(
    states: np.ndarray,
    x2dot: np.ndarray,
    times: np.ndarray,
    *,
    settings: AFM06aHardSampleInputs | None = None,
) -> np.ndarray:
    """Reconstruct the physical interaction force Fts [N] from the RHS."""

    settings = AFM06aHardSampleInputs() if settings is None else settings
    settings.validate()
    return (
        bar_fts_residual_from_observables(
            states,
            x2dot,
            times,
            settings=settings,
        )
        * float(settings.mass_kg)
    )


def global_normalization_statistics(
    dataset: LoadedDataset,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return one fixed input/output transform from the complete time span."""

    states = np.asarray(dataset.states, dtype=float)
    state_mean = np.mean(states, axis=1, dtype=float)
    state_scale = np.asarray([_safe_std(states[index]) for index in range(2)], dtype=float)
    if "bar_fts" in dataset.table:
        bar_fts = np.asarray(dataset.table["bar_fts"], dtype=float)
    else:
        bar_fts = bar_fts_residual_from_observables(
            states,
            np.asarray(dataset.table["x2dot"], dtype=float),
            np.asarray(dataset.table["t"], dtype=float),
            settings=dataset.settings,
        )
    if bar_fts.shape != (states.shape[1],) or not np.all(np.isfinite(bar_fts)):
        raise ValueError("full-span AFM06a force reference is invalid")
    return (
        np.asarray(state_mean, dtype=float),
        state_scale,
        float(np.mean(bar_fts, dtype=float)),
        _safe_std(bar_fts),
    )


def prepare_window(config: Stage1PlusLightConfig, dataset: LoadedDataset | None = None) -> PreparedWindow:
    loaded = load_dataset(config.dataset_root, config.error_level) if dataset is None else dataset
    global_state_mean, global_state_scale, global_bar_fts_mean, global_bar_fts_scale = (
        global_normalization_statistics(loaded)
    )
    times_all = np.asarray(loaded.table["t"], dtype=float)
    sample_times, selected, interpolated = select_window_sample_times(config, loaded)
    if sample_times.size < 2:
        raise ValueError("selected AFM06a training window contains fewer than two sampled points")
    if config.all_points_training:
        train_idx, val_idx = make_all_train_mask(sample_times.size)
    else:
        train_idx, val_idx = make_train_val_masks(sample_times.size, config.val_stride, config.val_offset)
    if interpolated:
        states = np.vstack(
            [
                np.interp(sample_times, times_all, np.asarray(loaded.states[index], dtype=float))
                for index in range(loaded.states.shape[0])
            ]
        )
        times = np.asarray(sample_times, dtype=float)
        x2dot = np.interp(times, times_all, np.asarray(loaded.table["x2dot"], dtype=float))
        fts = np.interp(times, times_all, np.asarray(loaded.table["Fts"], dtype=float))
        bar_fts = np.interp(times, times_all, np.asarray(loaded.table["bar_fts"], dtype=float))
        contact = np.interp(times, times_all, np.asarray(loaded.table["contact"], dtype=float)) >= 0.5
    else:
        states = np.asarray(loaded.states[:, selected], dtype=float)
        times = np.asarray(times_all[selected], dtype=float)
        x2dot = np.asarray(loaded.table["x2dot"], dtype=float)[selected]
        fts = np.asarray(loaded.table["Fts"], dtype=float)[selected]
        bar_fts = np.asarray(loaded.table["bar_fts"], dtype=float)[selected]
        contact = np.asarray(loaded.table["contact"], dtype=bool)[selected]
    return PreparedWindow(
        states=states,
        times=times,
        x2dot=x2dot,
        fts=fts,
        bar_fts=bar_fts,
        contact=contact,
        train_idx=train_idx,
        val_idx=val_idx,
        source_idx=selected,
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


__all__ = [
    "LoadedDataset",
    "PreparedWindow",
    "build_regime_sampling_weights",
    "dataset_paths",
    "settings_from_generation_manifest",
    "bar_fts_residual_from_observables",
    "fts_residual_from_observables",
    "global_normalization_statistics",
    "load_dataset",
    "make_all_train_mask",
    "make_train_val_masks",
    "prepare_window",
    "select_window_sample_times",
    "select_window_source_indices",
    "transition_resolved_source_indices",
]
