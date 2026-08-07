"""AFM05 data, windowing, and warmstart helpers for stage2light.

This module is the AFM05-specific replacement for the stage2light data
entry.  It deliberately does not expose true x3(t), true Fts(t), ks/cs truth,
or eta truth.  The available training information is the experimental
observable side: x1, x2, x2dot, t, F_actuation(t), the AFM05 initial condition,
and the force-reference signal used only for gain initialization.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from AFM05.stage1pluslight.data import (
    ensure_dataset,
    initial_condition_aligned_dataset,
    load_dataset,
    make_train_val_masks,
)
from AFM05.stage1pluslight.result_validation import validate_complete_stage1plus_payload
from AFM05.stage1pluslight.windows import window_manifests


@dataclass(frozen=True)
class WindowSplit:
    role: str
    label: str
    pixel_tag: str
    start_idx: int
    stop_idx: int
    t_start: float
    t_stop: float
    sample_stride: int
    times_full: np.ndarray
    ode_full: np.ndarray
    x2dot_full: np.ndarray
    contact_full: np.ndarray
    gain_force_reference_full: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    times_train: np.ndarray
    times_val: np.ndarray
    ode_train: np.ndarray
    ode_val: np.ndarray
    x2dot_train: np.ndarray
    x2dot_val: np.ndarray
    contact_train: np.ndarray
    contact_val: np.ndarray
    train_gain_force_reference: np.ndarray
    val_gain_force_reference: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    splits: tuple[WindowSplit, ...]
    train_states_all: np.ndarray
    train_gain_force_reference_all: np.ndarray
    state_mean: np.ndarray
    state_scale: np.ndarray
    known_pars: dict[str, Any]
    metadata: dict[str, Any]
    full_points: int
    true_side_available: bool = False


@dataclass(frozen=True)
class Stage1WarmstartCandidate:
    path: Path
    rank: int
    record: dict[str, Any]
    trial_id: int
    ranking_loss: float
    ks0: float
    cs0: float
    init_seed: int
    mech_winner: int = 0
    warmstart_label: str = ""


@dataclass(frozen=True)
class Prestage2WarmstartCandidate:
    path: Path
    candidate: int
    record: dict[str, Any]
    result_path: Path
    original_rank: int
    source_mech_winner: int
    trial_id: int
    ranking_loss: float
    ks0: float
    cs0: float
    init_seed: int


def _safe_scale(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    scale = float(np.std(arr))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = float(np.max(np.abs(arr)))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = 1.0
    return scale


def _state_normalizer(train_states_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Formal KAN input normalizer without hidden true x3 statistics."""

    states = np.asarray(train_states_all, dtype=float)
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError(f"train_states_all must have shape (n, >=3), got {states.shape}")
    x1_mean = float(np.mean(states[:, 0]))
    x2_mean = float(np.mean(states[:, 1]))
    x1_scale = _safe_scale(states[:, 0])
    x2_scale = _safe_scale(states[:, 1])
    # Match AFM05 st1pl: use the undeformed sample surface as the x3 center.
    mean = np.asarray([x1_mean, x2_mean, 0.0], dtype=float)
    scale = np.asarray([x1_scale, x2_scale, max(0.1 * x1_scale, 1.0e-30)], dtype=float)
    return mean, scale


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _data_dir(cfg) -> Path:
    return Path(cfg.dataset_root) / str(cfg.error_level) / "data"


def _load_gain_force_reference(cfg, expected_times: np.ndarray) -> np.ndarray:
    key = f"F_ts_{cfg.pixel_tag}"
    path = _data_dir(cfg) / f"afm05_F_ts_{cfg.pixel_tag}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"AFM05 gain force reference file is missing: {path}")
    with np.load(path) as data:
        times = np.asarray(data["t"], dtype=float)
        if key not in data.files:
            raise KeyError(f"AFM05 gain force reference file {path} is missing key {key!r}")
        values = np.asarray(data[key], dtype=float)
    expected_times = np.asarray(expected_times, dtype=float)
    if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
        raise ValueError(f"{path} must contain equal-length 1D t and {key} arrays")
    if times.size != expected_times.size or not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-15):
        raise ValueError(f"AFM05 {key} time grid does not match the entry dataset")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"AFM05 {key} contains non-finite values")
    return values


def _load_known_pars(cfg, expected_times: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    data_dir = _data_dir(cfg)
    metadata = _load_json(data_dir / "afm05_st1pl_entry_metadata.json")
    if str(metadata.get("pixel_tag", "")).strip() != str(cfg.pixel_tag):
        raise ValueError(
            "AFM05 metadata pixel_tag does not match config: "
            f"{metadata.get('pixel_tag')!r} != {cfg.pixel_tag!r}"
        )
    with np.load(data_dir / "afm05_F_actuation.npz") as data:
        act_t = np.asarray(data["t"], dtype=float)
        f_act = np.asarray(data["F_actuation"], dtype=float)
    expected_times = np.asarray(expected_times, dtype=float)
    if act_t.ndim != 1 or f_act.ndim != 1 or act_t.size != f_act.size:
        raise ValueError("AFM05 F_actuation npz must contain equal-length 1D t and F_actuation arrays")
    if act_t.size != expected_times.size or not np.allclose(act_t, expected_times, rtol=0.0, atol=1.0e-15):
        raise ValueError("AFM05 F_actuation time grid does not match the entry dataset")
    if not np.all(np.isfinite(act_t)) or not np.all(np.isfinite(f_act)):
        raise ValueError("AFM05 F_actuation trajectory contains non-finite values")
    known_pars = {
        "k_eff": float(metadata["k_eff"]),
        "m_eff": float(metadata["m_eff"]),
        "c_eff": float(metadata["c_eff"]),
        "actuation_times": act_t,
        "F_actuation": f_act,
        "Z": float(metadata["dist_Z_m"]),
        "a0": float(metadata["a0_m"]),
    }
    return known_pars, metadata


def load_stage1_payload(path: str | Path) -> dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.is_file():
        raise FileNotFoundError(f"Missing stage1pluslight result: {payload_path}")
    with payload_path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected stage1pluslight payload type: {type(payload)!r}")
    return payload


def _record_loss_key(record: dict[str, Any]) -> tuple[bool, float]:
    viable = bool(record.get("is_viable", False))
    loss = float(record.get("loss", float("inf")))
    if not np.isfinite(loss):
        loss = float("inf")
    return (not viable, loss)


def stage1_candidate_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    validate_complete_stage1plus_payload(payload)
    records = [rec for rec in payload.get("trial_parameters", []) if isinstance(rec, dict)]
    return sorted(records, key=_record_loss_key)


def _stage1_record_to_warmstart(
    *,
    payload_path: Path,
    record: dict[str, Any],
    rank: int = 0,
    mech_winner: int = 0,
) -> Stage1WarmstartCandidate:
    params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
    trial_id = int(params.get("trial_id", 0))
    ranking_loss = float(record.get("loss", float("inf")))
    ks0 = float(record.get("ks_hat", params.get("ks0", float("nan"))))
    cs0 = float(record.get("cs_hat", params.get("cs0", float("nan"))))
    init_seed = int(params.get("nn_init_seed", params.get("nn_seed_bank_idx", 0)))
    if not np.isfinite(ks0) or not np.isfinite(cs0):
        raise RuntimeError(f"Stage1 candidate is missing finite ks/cs init values: {payload_path}")
    if init_seed <= 0:
        raise RuntimeError(f"Stage1 candidate is missing a positive nn_init_seed: {payload_path}")
    if trial_id <= 0:
        raise RuntimeError(f"Stage1 candidate is missing a positive trial_id: {payload_path}")
    seedbank = int(params.get("nn_seed_bank_idx", 0))
    if mech_winner > 0 and seedbank > 0 and rank > 0:
        label = f"mech winner {int(mech_winner)} +NN seed {seedbank} (rank {int(rank)})"
    elif rank > 0:
        label = f"stage1 rank {int(rank)}"
    else:
        label = f"stage1 trial {int(trial_id)}"
    return Stage1WarmstartCandidate(
        path=payload_path,
        rank=int(rank),
        record=record,
        trial_id=trial_id,
        ranking_loss=ranking_loss,
        ks0=ks0,
        cs0=cs0,
        init_seed=init_seed,
        mech_winner=int(mech_winner),
        warmstart_label=label,
    )


def load_stage1_warmstart_record(path: str | Path) -> Stage1WarmstartCandidate:
    """Load one pre-extracted stage1 warmstart record for prest2/st2l."""

    record_path = Path(path)
    if not record_path.is_file():
        raise FileNotFoundError(f"Missing stage1 warmstart record: {record_path}")
    with record_path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected stage1 warmstart record type: {type(payload)!r}")

    if isinstance(payload.get("record"), dict):
        record = dict(payload["record"])
        source_path_raw = str(payload.get("source_stage1_input_path", "")).strip()
        payload_path = Path(source_path_raw) if source_path_raw != "" else record_path
        rank = int(payload.get("source_stage1_rank", payload.get("rank", 0)))
        mech_winner = int(payload.get("source_mech_winner", payload.get("mech_winner", 0)))
    else:
        record = dict(payload)
        payload_path = record_path
        rank = int(payload.get("source_stage1_rank", payload.get("rank", 0)))
        mech_winner = int(payload.get("source_mech_winner", payload.get("mech_winner", 0)))

    record["stage1_warmstart_record_path"] = str(record_path.resolve())
    return _stage1_record_to_warmstart(
        payload_path=payload_path,
        record=record,
        rank=rank,
        mech_winner=mech_winner,
    )


def select_stage1_candidate(path: str | Path, *, rank: int = 1) -> Stage1WarmstartCandidate:
    payload_path = Path(path)
    payload = load_stage1_payload(payload_path)
    validate_complete_stage1plus_payload(payload, path=payload_path)
    records = sorted([rec for rec in payload.get("trial_parameters", []) if isinstance(rec, dict)], key=_record_loss_key)
    if not records:
        raise RuntimeError(f"No stage1 candidate records in: {payload_path}")
    if rank < 1 or rank > len(records):
        raise ValueError(f"Requested stage1 rank {rank} but only {len(records)} candidate(s) exist")
    return _stage1_record_to_warmstart(payload_path=payload_path, record=records[rank - 1], rank=int(rank))


def select_stage1_candidate_by_trial_id(
    path: str | Path,
    *,
    trial_id: int,
    mech_winner: int = 0,
) -> Stage1WarmstartCandidate:
    payload_path = Path(path)
    payload = load_stage1_payload(payload_path)
    validate_complete_stage1plus_payload(payload, path=payload_path)
    records = [rec for rec in payload.get("trial_parameters", []) if isinstance(rec, dict)]
    rank_by_trial_id: dict[int, int] = {}
    for idx, rec in enumerate(sorted(records, key=_record_loss_key), start=1):
        params = rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}
        rec_trial_id = int(params.get("trial_id", 0))
        if rec_trial_id > 0:
            rank_by_trial_id[rec_trial_id] = int(idx)
    wanted = int(trial_id)
    if wanted <= 0:
        raise ValueError(f"Requested stage1 trial_id must be positive, got {trial_id}")
    for rec in records:
        params = rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}
        if int(params.get("trial_id", 0)) == wanted:
            return _stage1_record_to_warmstart(
                payload_path=payload_path,
                record=rec,
                rank=int(rank_by_trial_id.get(wanted, 0)),
                mech_winner=int(mech_winner),
            )
    raise ValueError(f"Requested stage1 trial_id {wanted} was not found in: {payload_path}")


def prestage2_candidate_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("candidate_b_records", "candidate_records", "ranked_candidates", "candidates"):
        records = payload.get(key)
        if isinstance(records, list) and records:
            return [rec for rec in records if isinstance(rec, dict)]
    return []


def select_prestage2_candidate(path: str | Path, *, candidate: int = 1) -> Prestage2WarmstartCandidate:
    payload_path = Path(path)
    payload = load_stage1_payload(payload_path)
    records = prestage2_candidate_records(payload)
    if not records:
        raise RuntimeError(f"No prestage2 candidate records in: {payload_path}")
    if candidate < 1 or candidate > len(records):
        raise ValueError(f"Requested prestage2 candidate {candidate} but only {len(records)} candidate(s) exist")
    record = records[candidate - 1]
    result_path_raw = str(record.get("result_path", "")).strip()
    result_path = Path(result_path_raw) if result_path_raw != "" else Path()
    if result_path_raw != "" and not result_path.is_absolute():
        result_path = (payload_path.parent / result_path).resolve()
    ranking_loss = float(record.get("candidate_loss", record.get("final_train_loss", record.get("final_val_loss", float("inf")))))
    return Prestage2WarmstartCandidate(
        path=payload_path,
        candidate=int(candidate),
        record=record,
        result_path=result_path,
        original_rank=int(record.get("source_stage1_rank", record.get("original_rank", 0))),
        source_mech_winner=int(record.get("source_mech_winner", 0)),
        trial_id=int(record.get("trial_id", record.get("source_stage1_trial_id", 0))),
        ranking_loss=ranking_loss,
        ks0=float(record.get("ks0", float("nan"))),
        cs0=float(record.get("cs0", float("nan"))),
        init_seed=int(record.get("seed", 0)),
    )


def prepare_data(cfg) -> PreparedData:
    ensure_dataset(
        Path(cfg.dataset_root),
        str(cfg.error_level),
        pixel_tag=str(cfg.pixel_tag),
        auto_generate=bool(cfg.auto_generate_dataset),
    )
    ode_data, pert_df = load_dataset(
        Path(cfg.dataset_root),
        str(cfg.error_level),
        pixel_tag=str(cfg.pixel_tag),
        auto_generate=bool(cfg.auto_generate_dataset),
    )
    ode_data, pert_df = initial_condition_aligned_dataset(ode_data, pert_df)

    all_times = np.asarray(pert_df["t"], dtype=float)
    x2dot_all = np.asarray(pert_df["x2dot"], dtype=float)
    contact_all = np.asarray(pert_df.get("contact", np.zeros(all_times.size, dtype=int)), dtype=int).astype(bool)
    gain_force_reference = _load_gain_force_reference(cfg, all_times)
    known_pars, metadata = _load_known_pars(cfg, all_times)

    manifests = window_manifests(
        all_times,
        cfg.window_mode,
        cfg.arch_window_us,
        sample_stride=cfg.window_sample_stride,
    )

    splits: list[WindowSplit] = []
    train_states_all: list[np.ndarray] = []
    train_gain_refs: list[np.ndarray] = []
    for win in manifests:
        idxs = np.asarray(win.idxs, dtype=int)
        ode_window = np.asarray(ode_data[:, idxs], dtype=float)
        times_window = np.asarray(all_times[idxs], dtype=float)
        x2dot_window = np.asarray(x2dot_all[idxs], dtype=float)
        contact_window = np.asarray(contact_all[idxs], dtype=bool)
        gain_window = np.asarray(gain_force_reference[idxs], dtype=float)
        train_idx, val_idx = make_train_val_masks(len(times_window), cfg.val_stride, cfg.val_offset)
        splits.append(
            WindowSplit(
                role=win.role,
                label=win.label,
                pixel_tag=str(cfg.pixel_tag),
                start_idx=int(win.start_idx),
                stop_idx=int(win.stop_idx),
                t_start=float(win.t_start),
                t_stop=float(win.t_stop),
                sample_stride=int(win.sample_stride),
                times_full=times_window,
                ode_full=ode_window,
                x2dot_full=x2dot_window,
                contact_full=contact_window,
                gain_force_reference_full=gain_window,
                train_idx=train_idx,
                val_idx=val_idx,
                times_train=times_window[train_idx],
                times_val=times_window[val_idx],
                ode_train=ode_window[:, train_idx],
                ode_val=ode_window[:, val_idx],
                x2dot_train=x2dot_window[train_idx],
                x2dot_val=x2dot_window[val_idx],
                contact_train=contact_window[train_idx],
                contact_val=contact_window[val_idx],
                train_gain_force_reference=gain_window[train_idx],
                val_gain_force_reference=gain_window[val_idx],
            )
        )
        train_states_all.append(np.asarray(ode_window[:, train_idx].T, dtype=float))
        train_gain_refs.append(np.asarray(gain_window[train_idx], dtype=float))

    if not splits:
        raise RuntimeError("AFM05 stage2light selected no windows")
    train_states = np.vstack(train_states_all)
    train_gain_force_reference = np.concatenate(train_gain_refs)
    state_mean, state_scale = _state_normalizer(train_states)
    return PreparedData(
        splits=tuple(splits),
        train_states_all=train_states,
        train_gain_force_reference_all=train_gain_force_reference,
        state_mean=state_mean,
        state_scale=state_scale,
        known_pars=known_pars,
        metadata=metadata,
        full_points=int(len(all_times)),
        true_side_available=False,
    )


__all__ = [
    "PreparedData",
    "Prestage2WarmstartCandidate",
    "Stage1WarmstartCandidate",
    "WindowSplit",
    "load_stage1_warmstart_record",
    "load_stage1_payload",
    "prepare_data",
    "prestage2_candidate_records",
    "select_prestage2_candidate",
    "select_stage1_candidate",
    "select_stage1_candidate_by_trial_id",
    "stage1_candidate_records",
]
