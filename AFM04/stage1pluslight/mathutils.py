"""Math helpers used by the AFM04 stage1pluslight kernel."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


MASK64 = 0xFFFFFFFFFFFFFFFF
INTMAX64 = (1 << 63) - 1


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(x, dtype=float)
    clipped = np.clip(arr, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(-clipped))
    if np.isscalar(x):
        return float(out)
    return out


def logit(x: float) -> float:
    return math.log(x / (1.0 - x))


def bound_param(raw: float, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * sigmoid(raw))


def raw_from_value(val: float, lo: float, hi: float) -> float:
    z = (float(val) - lo) / (hi - lo)
    z = min(max(z, 1.0e-6), 1.0 - 1.0e-6)
    return logit(z)


def softplus(x: float | np.ndarray, eps: float) -> float | np.ndarray:
    arr = np.asarray(x, dtype=float)
    z = arr / float(eps)
    out = np.where(
        z > 50.0,
        arr,
        np.where(z < -50.0, np.zeros_like(arr), float(eps) * np.log1p(np.exp(z))),
    )
    if np.isscalar(x):
        return float(out)
    return out


def relative_rmse_pct(err_sum: float, truth_sum: float, count: int, eps: float) -> float:
    denom = math.sqrt(float(truth_sum) / float(count)) + float(eps)
    return 100.0 * math.sqrt(float(err_sum) / float(count)) / denom


def rel_err_pct(est: float, truth: float, eps: float) -> float:
    return 100.0 * abs(float(est) - float(truth)) / (abs(float(truth)) + float(eps))


def mean_finite(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def join_nonempty_unique(values: Iterable[str]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text == "" or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return " | ".join(out)


def group_ranges(n: int, group_size: int) -> list[np.ndarray]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return [np.arange(start, min(start + group_size, n), dtype=int) for start in range(0, n, group_size)]


def _mix64(x: int) -> int:
    z = (int(x) + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def stage1plus_hash_seed(run_seed: int, item_id: int, stream_id: int) -> int:
    base = abs(int(run_seed)) % INTMAX64
    item = abs(int(item_id)) % INTMAX64
    stream = abs(int(stream_id)) % INTMAX64
    mixed = _mix64(base ^ ((item * 0x94D049BB133111EB) & MASK64) ^ ((stream * 0x369DEA0F31A53F85) & MASK64))
    seed = int(mixed % INTMAX64)
    return 1 if seed == 0 else seed


def stage1pluslight_nn_bank_seed(run_seed: int, num_hidden_layers: int, num_hidden_nodes: int, nn_seed_bank_idx: int) -> int:
    item_id = 200_000_000 + 10_000_000 * (1 + int(num_hidden_layers)) + 1_000_000 * (1 + int(num_hidden_nodes)) + int(nn_seed_bank_idx)
    return stage1plus_hash_seed(run_seed, item_id, 23)


__all__ = [
    "bound_param",
    "group_ranges",
    "join_nonempty_unique",
    "mean_finite",
    "raw_from_value",
    "rel_err_pct",
    "relative_rmse_pct",
    "sigmoid",
    "softplus",
    "stage1plus_hash_seed",
    "stage1pluslight_nn_bank_seed",
]
