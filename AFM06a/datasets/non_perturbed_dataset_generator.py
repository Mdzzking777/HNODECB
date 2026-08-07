"""Noise-free trajectory-generation helpers for AFM06a."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np


def rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance one fixed-size step with the classical fourth-order RK method."""

    k1 = f(t, u)
    k2 = f(t + 0.5 * dt, u + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, u + 0.5 * dt * k2)
    k4 = f(t + dt, u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_table(column_names: Sequence[str], matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a two-dimensional matrix into a dictionary of columns."""

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2D")
    if arr.shape[1] != len(column_names):
        raise ValueError("column_names length must match matrix column count")
    return {name: arr[:, i].copy() for i, name in enumerate(column_names)}


def table_to_matrix(table: Mapping[str, Any], column_names: Sequence[str]) -> np.ndarray:
    """Convert a dictionary of columns into a two-dimensional matrix."""

    cols = [np.asarray(table[name]) for name in column_names]
    lengths = {len(col) for col in cols}
    if len(lengths) != 1:
        raise ValueError("all table columns must have the same length")
    return np.column_stack(cols)


def save_table_npz(path: str, table: Mapping[str, Any], column_names: Sequence[str]) -> None:
    """Persist a dictionary of columns in a portable NumPy archive."""

    payload = {name: np.asarray(table[name]) for name in column_names}
    payload["__columns__"] = np.asarray(list(column_names), dtype=object)
    np.savez(path, **payload)


def load_table_npz(path: str) -> dict[str, np.ndarray]:
    """Load a dictionary of columns written by :func:`save_table_npz`."""

    with np.load(path, allow_pickle=True) as data:
        columns = [str(x) for x in data["__columns__"].tolist()]
        return {name: np.asarray(data[name]) for name in columns}


def generate_non_perturbed_training_set(
    ground_truth_rhs: Callable[[float, Sequence[float], Sequence[float]], Sequence[float]],
    u0: Sequence[float],
    p: Sequence[float],
    tspan: tuple[float, float],
    save_at: Sequence[float],
    *,
    column_names: Sequence[str] = ("t", "s1", "s2", "s3"),
) -> dict[str, np.ndarray]:
    """Roll out a noise-free trajectory and return a dictionary of columns."""

    save_at = np.asarray(save_at, dtype=float)
    if save_at.ndim != 1 or save_at.size < 2:
        raise ValueError("save_at must be a 1D array with at least 2 points")

    u = np.asarray(u0, dtype=float).copy()
    n_state = u.size
    if len(column_names) != n_state + 1:
        raise ValueError("column_names must contain time plus one name per state variable")
    if abs(save_at[0] - float(tspan[0])) > 1.0e-12 or abs(save_at[-1] - float(tspan[1])) > 1.0e-12:
        raise ValueError("save_at must start/end at tspan boundaries")

    states = np.zeros((save_at.size, n_state), dtype=float)
    states[0, :] = u

    for i in range(save_at.size - 1):
        t = float(save_at[i])
        dt = float(save_at[i + 1] - save_at[i])
        if dt <= 0.0:
            raise ValueError("save_at must be strictly increasing")

        def rhs_wrapped(tt: float, uu: np.ndarray) -> np.ndarray:
            return np.asarray(ground_truth_rhs(tt, uu.tolist(), p), dtype=float)

        u = rk4_step(rhs_wrapped, t, u, dt)
        states[i + 1, :] = u

    matrix = np.column_stack((save_at, states))
    return build_table(column_names, matrix)


__all__ = [
    "rk4_step",
    "build_table",
    "table_to_matrix",
    "save_table_npz",
    "load_table_npz",
    "generate_non_perturbed_training_set",
]

