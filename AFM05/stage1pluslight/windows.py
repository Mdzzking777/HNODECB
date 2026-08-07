"""Window selection helpers for AFM05 stage1pluslight."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from AFM05.DATAgeneration.afm05_st1pl_entry_generator import DEFAULT_TRAINING_WINDOW_START_S


@dataclass(frozen=True)
class WindowManifest:
    role: str
    label: str
    start_idx: int
    stop_idx: int
    length: int
    t_start: float
    t_stop: float
    sample_stride: int
    idxs: np.ndarray


def _as_times(times: np.ndarray) -> np.ndarray:
    out = np.asarray(times, dtype=float)
    if out.ndim != 1 or out.size == 0:
        raise ValueError("AFM05 window selection received an empty or non-1D time vector")
    if not np.all(np.isfinite(out)):
        raise ValueError("AFM05 window selection received non-finite time values")
    if out.size > 1 and not np.all(np.diff(out) > 0.0):
        raise ValueError("AFM05 time vector must be strictly increasing")
    return out


def window_05_initial_indices(times: np.ndarray, window_span: float) -> np.ndarray:
    """Select the fixed AFM05 training/validation window.

    The entry dataset and the loss window both begin at the current operational
    AFM05 initial/contact-side point.
    """

    times = _as_times(times)
    if not np.isfinite(window_span) or window_span <= 0.0:
        return np.arange(times.size, dtype=int)
    start_idx = int(np.searchsorted(times, DEFAULT_TRAINING_WINDOW_START_S, side="left"))
    if start_idx >= times.size:
        raise ValueError("AFM05 default training-window start is outside the entry dataset")
    t0 = float(times[start_idx])
    idxs = np.flatnonzero((times - t0) <= float(window_span))
    idxs = idxs[idxs >= start_idx]
    return idxs if idxs.size > 0 else np.array([0], dtype=int)


def select_window_indices(
    times: np.ndarray,
    window_mode: str,
    window_us: float,
    sample_stride: int = 1,
) -> np.ndarray:
    mode = str(window_mode).strip().lower()
    if mode in ("full", "full_horizon", "full-horizon"):
        idxs = np.arange(_as_times(times).size, dtype=int)
    elif mode == "window_05_initial":
        idxs = window_05_initial_indices(times, window_us)
    else:
        raise ValueError(f"unsupported AFM05 window mode: {window_mode!r}")

    stride = max(1, int(sample_stride))
    if stride > 1:
        idxs = idxs[::stride]
    return idxs if idxs.size > 0 else np.array([0], dtype=int)


def window_manifests(
    times: np.ndarray,
    window_mode: str,
    window_us: float,
    sample_stride: int = 1,
) -> list[WindowManifest]:
    times = _as_times(times)
    stride = max(1, int(sample_stride))
    idxs = select_window_indices(times, window_mode, window_us, sample_stride=stride)
    mode = str(window_mode).strip().lower()
    role = "full_horizon" if mode in ("full", "full_horizon", "full-horizon") else "window_05_initial"
    label = f"{role}_window_stride{stride}"
    return [
        WindowManifest(
            role=role,
            label=label,
            start_idx=int(idxs[0]),
            stop_idx=int(idxs[-1]),
            length=int(len(idxs)),
            t_start=float(times[idxs[0]]),
            t_stop=float(times[idxs[-1]]),
            sample_stride=int(stride),
            idxs=np.asarray(idxs, dtype=int),
        )
    ]


__all__ = [
    "WindowManifest",
    "select_window_indices",
    "window_05_initial_indices",
    "window_manifests",
]
