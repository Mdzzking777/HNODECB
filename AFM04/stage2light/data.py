"""Dataset, windowing, and stage1 warmstart helpers for AFM04 stage2light."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from AFM04.datasets.afm_dataset_generator import generate_afm_dmt_kv_dataset
from AFM04.stage1pluslight.data import load_dataset, make_train_val_masks, truncate_to_first_contact
from AFM04.stage1pluslight.windows import window_manifests
from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import DEFAULT_SETTINGS


@dataclass(frozen=True)
class WindowSplit:
    role: str
    label: str
    start_idx: int
    stop_idx: int
    t_start: float
    t_stop: float
    times_full: np.ndarray
    ode_full: np.ndarray
    x2dot_full: np.ndarray
    contact_full: np.ndarray
    fts_full_true: np.ndarray
    times_train: np.ndarray
    times_val: np.ndarray
    ode_train: np.ndarray
    ode_val: np.ndarray
    x2dot_train: np.ndarray
    x2dot_val: np.ndarray
    contact_train: np.ndarray
    contact_val: np.ndarray
    fts_train_true: np.ndarray
    fts_val_true: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    splits: tuple[WindowSplit, ...]
    train_states_all: np.ndarray
    train_fts_all: np.ndarray
    state_mean: np.ndarray
    state_scale: np.ndarray
    known_pars: tuple[float, ...]
    eta_star_true: float
    mech_true: np.ndarray
    full_points: int


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos = x >= 0.0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def f_ts_from_distance_numpy(s: np.ndarray, *, Estar: float, R: float, A: float, a0: float, beta: float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    g = _sigmoid(beta * (s - a0))
    denom = np.square(g * (s - a0) + a0)
    adh = -(A * R) / (6.0 * denom)
    hertz = (1.0 - g) * (4.0 / 3.0) * Estar * np.sqrt(R) * np.power(np.maximum(a0 - s, 0.0), 1.5)
    return adh + hertz


def _compute_true_fts(
    ode: np.ndarray,
    known_pars: tuple[float, ...],
    *,
    eta_star: float,
    mech_true: np.ndarray,
) -> np.ndarray:
    _, _, _, _, _, R, dist, Estar, A, a0, beta = known_pars
    x1 = np.asarray(ode[0, :], dtype=float)
    x2 = np.asarray(ode[1, :], dtype=float)
    x3 = np.asarray(ode[2, :], dtype=float)
    ks = float(np.asarray(mech_true, dtype=float)[0])
    cs = float(np.asarray(mech_true, dtype=float)[1])
    s = dist + x1 - x3
    g = _sigmoid(beta * (s - a0))
    denom = np.maximum(g * (s - a0) + a0, 1.0e-15)
    adhesion = -(A * R) / (6.0 * np.square(denom))
    delta = np.maximum(a0 - s, 0.0)
    f_hertz = (4.0 / 3.0) * Estar * np.sqrt(R) * np.power(delta, 1.5)
    kv_coeff = float(eta_star) * np.sqrt(R) * np.sqrt(delta)
    f_static = adhesion + (1.0 - g) * f_hertz
    x3dot = (-f_static + kv_coeff * x2 - ks * x3) / np.maximum(cs + kv_coeff, 1.0e-15)
    delta_dot = np.where(delta > 0.0, x3dot - x2, 0.0)
    return f_static + kv_coeff * delta_dot


def _state_normalizer(train_states_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_states_all, axis=0)
    centered = train_states_all - mean[None, :]
    scale = np.max(np.abs(centered), axis=0)
    scale = np.maximum(scale, 1.0e-12)
    return mean.astype(float), scale.astype(float)


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
    ranked = payload.get("ranked_topk")
    if isinstance(ranked, list) and ranked:
        return [rec for rec in ranked if isinstance(rec, dict)]
    trials = payload.get("trial_parameters", [])
    records = [rec for rec in trials if isinstance(rec, dict)]
    return sorted(records, key=_record_loss_key)


def select_stage1_candidate(path: str | Path, *, rank: int = 1) -> Stage1WarmstartCandidate:
    payload_path = Path(path)
    payload = load_stage1_payload(payload_path)
    ranked = payload.get("ranked_topk")
    ranked_records = [rec for rec in ranked if isinstance(rec, dict)] if isinstance(ranked, list) else []
    all_trials = payload.get("trial_parameters", [])
    all_records = sorted([rec for rec in all_trials if isinstance(rec, dict)], key=_record_loss_key)

    if ranked_records and rank <= len(ranked_records):
        records = ranked_records
    else:
        records = all_records

    if not records:
        raise RuntimeError(f"No stage1 candidate records in: {payload_path}")
    if rank < 1 or rank > len(records):
        raise ValueError(f"Requested stage1 rank {rank} but only {len(records)} candidate(s) exist")

    record = records[rank - 1]
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

    return Stage1WarmstartCandidate(
        path=payload_path,
        rank=int(rank),
        record=record,
        trial_id=trial_id,
        ranking_loss=ranking_loss,
        ks0=ks0,
        cs0=cs0,
        init_seed=init_seed,
    )


def prepare_data(cfg) -> PreparedData:
    if cfg.auto_generate_dataset:
        generate_afm_dmt_kv_dataset(error_level=cfg.error_level, output_root=cfg.dataset_root, save_outputs=True)

    ode_data, pert_df = load_dataset(cfg.dataset_root, cfg.error_level, auto_generate=cfg.auto_generate_dataset)
    ode_data, pert_df = truncate_to_first_contact(ode_data, pert_df)

    all_times = np.asarray(pert_df["t"], dtype=float)
    x2dot_all = np.asarray(pert_df["x2dot"], dtype=float)
    contact_all = np.asarray(pert_df["contact"], dtype=bool)
    x1_signal = np.asarray(ode_data[0, :], dtype=float)

    manifests = window_manifests(all_times, contact_all, cfg.window_mode, cfg.arch_window_us, x1_signal=x1_signal)
    settings = DEFAULT_SETTINGS
    known_pars = (
        settings.k,
        settings.wd,
        settings.m,
        settings.c,
        settings.Fd,
        settings.R,
        settings.dist,
        settings.Estar,
        settings.A,
        settings.a0,
        settings.beta,
    )
    mech_true = np.array([settings.ks, settings.cs], dtype=float)
    eta_star_true = float(settings.eta_star)

    splits: list[WindowSplit] = []
    train_states_all: list[np.ndarray] = []
    train_fts_all: list[np.ndarray] = []

    for win in manifests:
        idxs = np.asarray(win.idxs, dtype=int)
        ode_window = np.asarray(ode_data[:, idxs], dtype=float)
        times_window = np.asarray(all_times[idxs], dtype=float)
        x2dot_window = np.asarray(x2dot_all[idxs], dtype=float)
        contact_window = np.asarray(contact_all[idxs], dtype=bool)
        fts_window_true = _compute_true_fts(
            ode_window,
            known_pars,
            eta_star=eta_star_true,
            mech_true=mech_true,
        )

        train_idx, val_idx = make_train_val_masks(len(times_window), cfg.val_stride, cfg.val_offset)
        ode_train = ode_window[:, train_idx]
        ode_val = ode_window[:, val_idx]
        times_train = times_window[train_idx]
        times_val = times_window[val_idx]
        x2dot_train = x2dot_window[train_idx]
        x2dot_val = x2dot_window[val_idx]
        contact_train = contact_window[train_idx]
        contact_val = contact_window[val_idx]
        fts_train_true = fts_window_true[train_idx]
        fts_val_true = fts_window_true[val_idx]

        splits.append(
            WindowSplit(
                role=win.role,
                label=win.label,
                start_idx=int(win.start_idx),
                stop_idx=int(win.stop_idx),
                t_start=float(win.t_start),
                t_stop=float(win.t_stop),
                times_full=times_window,
                ode_full=ode_window,
                x2dot_full=x2dot_window,
                contact_full=contact_window,
                fts_full_true=fts_window_true,
                times_train=times_train,
                times_val=times_val,
                ode_train=ode_train,
                ode_val=ode_val,
                x2dot_train=x2dot_train,
                x2dot_val=x2dot_val,
                contact_train=contact_train,
                contact_val=contact_val,
                fts_train_true=fts_train_true,
                fts_val_true=fts_val_true,
            )
        )
        train_states_all.append(np.asarray(ode_train.T, dtype=float))
        train_fts_all.append(np.asarray(fts_train_true, dtype=float))

    train_states = np.vstack(train_states_all)
    train_fts = np.concatenate(train_fts_all)
    state_mean, state_scale = _state_normalizer(train_states)

    return PreparedData(
        splits=tuple(splits),
        train_states_all=train_states,
        train_fts_all=train_fts,
        state_mean=state_mean,
        state_scale=state_scale,
        known_pars=known_pars,
        eta_star_true=eta_star_true,
        mech_true=mech_true,
        full_points=int(len(all_times)),
    )


__all__ = [
    "PreparedData",
    "Stage1WarmstartCandidate",
    "WindowSplit",
    "f_ts_from_distance_numpy",
    "load_stage1_payload",
    "prepare_data",
    "select_stage1_candidate",
    "stage1_candidate_records",
]
