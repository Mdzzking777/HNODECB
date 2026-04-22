"""Window selection helpers for AFM04 stage1pluslight."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowManifest:
    role: str
    label: str
    start_idx: int
    stop_idx: int
    length: int
    t_start: float
    t_stop: float
    idxs: np.ndarray


def window_indices(times: np.ndarray, window_span: float) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        raise ValueError("window selection received an empty time vector")
    if not np.isfinite(window_span) or window_span <= 0.0:
        return np.arange(times.size, dtype=int)
    t0 = float(times[0])
    idxs = np.flatnonzero((times - t0) <= window_span)
    return idxs if idxs.size > 0 else np.array([0], dtype=int)


def peak_to_peak(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.max(values) - np.min(values)) if values.size > 0 else float("nan")


def contact_onsets(contact_mask: np.ndarray) -> np.ndarray:
    contact = np.asarray(contact_mask, dtype=bool)
    if contact.size == 0:
        return np.array([], dtype=int)
    starts = contact & np.concatenate(([True], ~contact[:-1]))
    return np.flatnonzero(starts)


def nearest_time_index(t_us: np.ndarray, target_us: float) -> int:
    return int(np.argmin(np.abs(np.asarray(t_us, dtype=float) - float(target_us))))


def start_anchor_index(t_us: np.ndarray, target_us: float) -> int:
    idx = int(np.searchsorted(np.asarray(t_us, dtype=float), float(target_us), side="left"))
    return min(max(idx, 0), len(t_us) - 1)


def extend_stop_to_contact_end(contact_mask: np.ndarray, stop_idx: int) -> int:
    contact = np.asarray(contact_mask, dtype=bool)
    stop_idx = min(max(int(stop_idx), 0), len(contact) - 1)
    if not bool(contact[stop_idx]):
        return stop_idx
    new_stop = stop_idx
    while new_stop + 1 < len(contact) and bool(contact[new_stop + 1]):
        new_stop += 1
    return new_stop


def relocate_window_with_anchor(t_us: np.ndarray, start_idx: int, stop_idx: int, *, anchor: str, target_us: float) -> tuple[int, int]:
    length = stop_idx - start_idx + 1
    n = len(t_us)
    if length > n:
        raise ValueError("window length exceeds available trajectory length")
    if anchor == "start":
        new_start = start_anchor_index(t_us, target_us)
        new_start = min(max(new_start, 0), n - length)
        new_stop = new_start + length - 1
    elif anchor == "stop":
        new_stop = nearest_time_index(t_us, target_us)
        new_stop = min(max(new_stop, length - 1), n - 1)
        new_start = new_stop - length + 1
    else:
        raise ValueError(f"unsupported anchor: {anchor}")
    return new_start, new_stop


def stage2_w1_window_indices(times: np.ndarray, contact_mask: np.ndarray, x1_signal: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    contact = np.asarray(contact_mask, dtype=bool)
    x1_signal = np.asarray(x1_signal, dtype=float)
    if times.size == 0:
        raise ValueError("stage2_w1 window: empty time vector")
    if contact.size != times.size or x1_signal.size != times.size:
        raise ValueError("stage2_w1 window: signal lengths must match times")
    onset_post = contact_onsets(contact)
    if onset_post.size < 3:
        raise ValueError("stage2_w1 window needs at least 3 contact onsets")
    start_idx = int(onset_post[0])
    stop_idx = int(onset_post[2] - 1)
    t_us = times * 1e6
    new_start, new_stop = relocate_window_with_anchor(t_us, start_idx, stop_idx, anchor="start", target_us=1034.0)
    new_stop = extend_stop_to_contact_end(contact, new_stop)
    return np.arange(new_start, new_stop + 1, dtype=int)


def _apply_stage2_window_time_overrides(selected: list[dict], times: np.ndarray, contact_mask: np.ndarray) -> list[dict]:
    overrides = {
        "first_contact": ("start", 1034.0),
        "max_x1_pp_change": ("stop", 1057.0),
        "tail_stable": ("stop", 1998.0),
    }
    t_us = np.asarray(times, dtype=float) * 1e6
    contact = np.asarray(contact_mask, dtype=bool)
    adjusted: list[dict] = []
    for win in selected:
        anchor, target_us = overrides[win["role"]]
        new_start, new_stop = relocate_window_with_anchor(t_us, win["start_idx"], win["stop_idx"], anchor=anchor, target_us=target_us)
        if win["role"] == "first_contact":
            new_stop = extend_stop_to_contact_end(contact, new_stop)
        adjusted.append({**win, "start_idx": new_start, "stop_idx": new_stop, "length": new_stop - new_start + 1, "t_start": float(times[new_start]), "t_stop": float(times[new_stop]), "idxs": np.arange(new_start, new_stop + 1, dtype=int)})
    return adjusted


def stage2_window_manifest(times: np.ndarray, contact_mask: np.ndarray, x1_signal: np.ndarray) -> list[WindowManifest]:
    times = np.asarray(times, dtype=float)
    contact = np.asarray(contact_mask, dtype=bool)
    x1_signal = np.asarray(x1_signal, dtype=float)
    if times.size == 0:
        raise ValueError("stage2 window manifest: empty time vector")
    if contact.size != times.size or x1_signal.size != times.size:
        raise ValueError("stage2 window manifest: signal lengths must match times")

    onset_post = contact_onsets(contact)
    if onset_post.size < 3:
        raise ValueError("stage2 window manifest needs at least 3 contact onsets")

    cycle_pp: list[float] = []
    for k in range(len(onset_post) - 1):
        lo = int(onset_post[k])
        hi = int(onset_post[k + 1] - 1)
        if hi < lo:
            raise ValueError("invalid cycle bounds while building stage2 window manifest")
        cycle_pp.append(peak_to_peak(x1_signal[lo : hi + 1]))

    candidates: list[dict] = []
    for k in range(len(onset_post) - 2):
        start_idx = int(onset_post[k])
        stop_idx = int(onset_post[k + 2] - 1)
        pp1 = cycle_pp[k]
        pp2 = cycle_pp[k + 1]
        candidates.append({"candidate_index": k + 1, "start_idx": start_idx, "stop_idx": stop_idx, "length": stop_idx - start_idx + 1, "t_start": float(times[start_idx]), "t_stop": float(times[stop_idx]), "cycle1_index": k + 1, "cycle2_index": k + 2, "x1_pp_cycle1": pp1, "x1_pp_cycle2": pp2, "x1_pp_delta": abs(pp2 - pp1), "role": "", "label": "", "idxs": np.arange(start_idx, stop_idx + 1, dtype=int)})
    if not candidates:
        raise ValueError("no valid stage2-style windows were built")

    first_idx = 0
    deltas = np.asarray([cand["x1_pp_delta"] for cand in candidates], dtype=float)
    change_order = list(np.argsort(deltas)[::-1])
    tail_order = list(range(len(candidates) - 1, -1, -1))

    selected: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def add_window(role: str, candidate_idx: int) -> bool:
        cand = candidates[int(candidate_idx)]
        key = (cand["start_idx"], cand["stop_idx"])
        if key in seen:
            return False
        seen.add(key)
        selected.append({**cand, "role": role, "label": f"{role}_window"})
        return True

    def add_first_unique(role: str, candidate_order: list[int]) -> None:
        for candidate_idx in candidate_order:
            if add_window(role, int(candidate_idx)):
                return

    add_window("first_contact", first_idx)
    add_first_unique("max_x1_pp_change", change_order)
    add_first_unique("tail_stable", tail_order)
    if len(selected) < 3:
        raise ValueError("stage2 window manifest could not find 3 unique windows")

    adjusted = _apply_stage2_window_time_overrides(selected, times, contact)
    return [WindowManifest(role=win["role"], label=win["label"], start_idx=int(win["start_idx"]), stop_idx=int(win["stop_idx"]), length=int(win["length"]), t_start=float(win["t_start"]), t_stop=float(win["t_stop"]), idxs=np.asarray(win["idxs"], dtype=int)) for win in adjusted]


def first_contact_window_indices(times: np.ndarray, contact_mask: np.ndarray, window_us: float) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    contact = np.asarray(contact_mask, dtype=bool)
    if times.size == 0:
        raise ValueError("first-contact window: empty time vector")
    if contact.size != times.size:
        raise ValueError("first-contact window: contact mask length does not match times")
    idx = np.flatnonzero(contact)
    if idx.size == 0:
        return window_indices(times, window_us)
    first_contact_idx = int(idx[0])
    t_start = float(times[first_contact_idx])
    t_stop = t_start + float(window_us)
    stop_idx = int(np.searchsorted(times, t_stop, side="right") - 1)
    stop_idx = min(max(stop_idx, first_contact_idx), len(times) - 1)
    return np.arange(first_contact_idx, stop_idx + 1, dtype=int)


def select_window_indices(times: np.ndarray, contact_mask: np.ndarray, window_mode: str, window_us: float, *, x1_signal: np.ndarray | None = None) -> np.ndarray:
    mode = window_mode.strip().lower()
    if mode in ("full", "full_horizon", "full-horizon"):
        return np.arange(len(times), dtype=int)
    if mode in ("stage2_w123", "stage2-w123", "stage2_w1", "stage2-w1"):
        if x1_signal is None:
            raise ValueError(f"{window_mode} requires x1_signal")
        return stage2_w1_window_indices(times, contact_mask, x1_signal)
    if mode in ("first_contact", "first-contact"):
        return first_contact_window_indices(times, contact_mask, window_us)
    raise ValueError(f"unsupported window mode: {window_mode!r}")


def window_manifests(times: np.ndarray, contact_mask: np.ndarray, window_mode: str, window_us: float, *, x1_signal: np.ndarray | None = None) -> list[WindowManifest]:
    mode = window_mode.strip().lower()
    if mode in ("stage2_w123", "stage2-w123"):
        if x1_signal is None:
            raise ValueError(f"{window_mode} requires x1_signal")
        return stage2_window_manifest(times, contact_mask, x1_signal)

    idxs = select_window_indices(times, contact_mask, window_mode, window_us, x1_signal=x1_signal)
    role = "full_horizon" if mode in ("full", "full_horizon", "full-horizon") else "first_contact"
    return [WindowManifest(role=role, label=f"{role}_window", start_idx=int(idxs[0]), stop_idx=int(idxs[-1]), length=int(len(idxs)), t_start=float(times[idxs[0]]), t_stop=float(times[idxs[-1]]), idxs=np.asarray(idxs, dtype=int))]


__all__ = [
    "WindowManifest",
    "contact_onsets",
    "first_contact_window_indices",
    "select_window_indices",
    "stage2_w1_window_indices",
    "stage2_window_manifest",
    "window_manifests",
]
