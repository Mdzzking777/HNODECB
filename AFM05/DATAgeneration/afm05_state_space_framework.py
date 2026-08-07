"""AFM05 generation-layer state-space equation skeleton.

This file intentionally contains no experimental constants and no true hidden
variables.  AFM05 starts from experimental data, so the generation-layer
framework must not silently fill values for

    ks, cs, x3(t), or Fts(t).

At this step we do fill the experimentally known cantilever-side values:

    k_eff, omega0, m_eff, c_eff, dist, a0, x1(t), x2(t).

The in-air force balance provides a time-dependent actuation force:

    F_actuation(t) = m_eff*xddot_air + c_eff*xdot_air + k_eff*x_air

We still do not fill the sample-side unknowns ``ks``, ``cs``, and ``x3(t)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


Array = np.ndarray

DEFAULT_TRACE_NPZ = Path(__file__).with_name(
    "PS_cantilever_disp_vel_time_3_pixels_Z_85.0_a0_0.07108_file_scan04143.imp_.npz"
)


@dataclass(frozen=True)
class AFM05KnownValues:
    """Known cantilever-side values for AFM05, in SI units."""

    k_eff: float
    omega0: float
    omega1: float
    omega2: float
    m_eff: float
    c_eff: float
    dist: float
    a0: float


@dataclass(frozen=True)
class AFM05ObservedValues:
    """Observed experimental channels used by the generation-layer framework."""

    t: Array
    x1: Array
    x2: Array
    pixel_key: str


@dataclass(frozen=True)
class AFM05UnknownPlaceholders:
    """Unknown quantities that AFM05 must estimate or constrain later."""

    F_actuation: Array
    ks: float
    cs: float
    x3: Array
    Fts: Array


def _load_add_data_dict(path: str | Path) -> dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
    raw = data["add_data_dict"].item()
    if not isinstance(raw, Mapping):
        raise TypeError("add_data_dict must be a mapping")
    return dict(raw)


def load_known_values(trace_npz: str | Path = DEFAULT_TRACE_NPZ) -> AFM05KnownValues:
    """Load the known AFM05 constants and compute m_eff/c_eff.

    Required source quantities:

        k_eff  <- stored k
        omega0 <- stored omega0, or 2*pi*f0
        omega1 <- omega0 - d_omega
        omega2 <- omega0 + d_omega
        dist   <- stored Z
        a0     <- stored a0

    Derived quantities:

        m_eff = k_eff / omega0^2
        c_eff = k_eff / (Q * omega0)
    """

    meta = _load_add_data_dict(trace_npz)
    k_eff = float(meta["k"])
    omega0 = float(meta.get("omega0", 2.0 * np.pi * float(meta["f0"])))
    omega1 = float(meta.get("omega1", omega0 - float(meta["d_omega"])))
    omega2 = float(meta.get("omega2", omega0 + float(meta["d_omega"])))
    q_factor = float(meta["Q_factor"])
    m_eff = k_eff / (omega0**2)
    c_eff = k_eff / (q_factor * omega0)
    dist = float(meta["Z"])
    a0 = float(meta["a0"])
    return AFM05KnownValues(
        k_eff=k_eff,
        omega0=omega0,
        omega1=omega1,
        omega2=omega2,
        m_eff=m_eff,
        c_eff=c_eff,
        dist=dist,
        a0=a0,
    )


def list_pixel_keys(trace_npz: str | Path = DEFAULT_TRACE_NPZ) -> tuple[str, ...]:
    data = np.load(Path(trace_npz), allow_pickle=True)
    return tuple(key for key in data.files if key.startswith("pixel_"))


def load_observed_values(
    trace_npz: str | Path = DEFAULT_TRACE_NPZ,
    *,
    pixel_key: str | None = None,
    x1_scale_to_m: float = 1.0e-9,
    x2_scale_to_m_per_s: float = 1.0e-9,
) -> AFM05ObservedValues:
    """Load x1(t), x2(t), and t from one experimental pixel.

    Current AFM05 file convention:

        column 0 -> x1, stored in nm
        column 1 -> x2, stored in nm/s
        column 2 -> t, stored in s
    """

    data = np.load(Path(trace_npz), allow_pickle=True)
    pixel_keys = tuple(key for key in data.files if key.startswith("pixel_"))
    if not pixel_keys:
        raise KeyError("No pixel_* arrays found in trace file")
    if pixel_key is None:
        pixel_key = pixel_keys[0]
    arr = np.asarray(data[pixel_key], dtype=float)
    return AFM05ObservedValues(
        t=arr[:, 2].copy(),
        x1=arr[:, 0] * x1_scale_to_m,
        x2=arr[:, 1] * x2_scale_to_m_per_s,
        pixel_key=pixel_key,
    )


def observed_x2dot(observed: AFM05ObservedValues) -> Array:
    """Compute x2dot from observed x2(t), outside the hidden-state model."""

    return np.gradient(observed.x2, observed.t, edge_order=2)


def cantilever_rhs(
    *,
    t: Array,
    x1: Array,
    x2: Array,
    Fts: Array,
    F_actuation: Array,
    m_eff: float,
    c_eff: float,
    k_eff: float,
) -> tuple[Array, Array]:
    """Cantilever state equation.

    State variables on the observable cantilever side:

        x1dot = x2
        x2dot = (F_actuation(t) - k_eff*x1 - c_eff*x2 + Fts) / m_eff

    This does not infer ``Fts`` or ``F_actuation``.  Both are explicit inputs.
    """

    x1dot = x2
    x2dot = (F_actuation - k_eff * x1 - c_eff * x2 + Fts) / m_eff
    return x1dot, x2dot


def in_air_rhs(
    *,
    t: Array,
    x1: Array,
    x2: Array,
    F_actuation: Array,
    m_eff: float,
    c_eff: float,
    k_eff: float,
) -> tuple[Array, Array]:
    """In-air two-state cantilever equation.

    This RHS is for the in-air trajectory ``eta1_in_air``.  It has no
    tip-sample interaction force:

        x1dot = x2
        x2dot = (F_actuation(t) - k_eff*x1 - c_eff*x2) / m_eff
    """

    x1dot = x2
    x2dot = (F_actuation - k_eff * x1 - c_eff * x2) / m_eff
    return x1dot, x2dot


def in_air_balance_residual(
    *,
    t: Array,
    x1: Array,
    x2: Array,
    x2dot: Array,
    F_actuation: Array,
    m_eff: float,
    c_eff: float,
    k_eff: float,
) -> Array:
    """Residual form of the in-air equation.

        m_eff*x2dot - (F_actuation(t) - k_eff*x1 - c_eff*x2) = 0.
    """

    return m_eff * x2dot - (F_actuation - k_eff * x1 - c_eff * x2)


def cantilever_balance_residual(
    *,
    t: Array,
    x1: Array,
    x2: Array,
    x2dot: Array,
    Fts: Array,
    F_actuation: Array,
    m_eff: float,
    c_eff: float,
    k_eff: float,
) -> Array:
    """Residual form of the cantilever equation.

    A consistent candidate set should satisfy

        m_eff*x2dot - (F_actuation(t) - k_eff*x1 - c_eff*x2 + Fts) = 0.
    """

    return m_eff * x2dot - (F_actuation - k_eff * x1 - c_eff * x2 + Fts)


def surface_rhs(*, x3: Array, Fts: Array, ks: float, cs: float) -> Array:
    """AFM04-compatible surface/Kelvin-Voigt equation.

    Sign convention inherited from AFM04:

        x3dot = (-Fts - ks*x3) / cs

    This function does not generate true ``x3``.  It only computes the RHS if a
    candidate ``x3``, ``Fts``, ``ks``, and ``cs`` are supplied.
    """

    return (-Fts - ks * x3) / cs


def surface_balance_residual(*, x3: Array, x3dot: Array, Fts: Array, ks: float, cs: float) -> Array:
    """Residual form of the surface equation.

        cs*x3dot + ks*x3 + Fts = 0.
    """

    return cs * x3dot + ks * x3 + Fts


def tip_sample_distance(*, x1: Array, x3: Array, dist: float) -> Array:
    """Tip-sample distance relation.

        s = dist + x1 - x3

    This is only evaluable after a candidate hidden trajectory ``x3`` exists.
    """

    return dist + x1 - x3


def contact_gap(*, x1: Array, x3: Array, dist: float, a0: float) -> Array:
    """Gap relative to the nominal contact threshold.

        gap = s - a0

    Negative values correspond to indentation/contact under this convention.
    """

    return tip_sample_distance(x1=x1, x3=x3, dist=dist) - a0


def equation_summary() -> str:
    """Text summary of known values and remaining placeholders."""

    known = load_known_values()
    observed = load_observed_values()

    return (
        "AFM05 generation-layer state-space skeleton\n"
        "===========================================\n\n"
        "Known values filled at this stage:\n"
        f"  k_eff [N/m]      = {known.k_eff:.12e}\n"
        f"  omega0 [rad/s]   = {known.omega0:.12e}\n"
        f"  omega1 [rad/s]   = {known.omega1:.12e}\n"
        f"  omega2 [rad/s]   = {known.omega2:.12e}\n"
        f"  m_eff [kg]       = {known.m_eff:.12e}\n"
        f"  c_eff [N*s/m]    = {known.c_eff:.12e}\n"
        f"  dist [m]         = {known.dist:.12e}\n"
        f"  a0 [m]           = {known.a0:.12e}\n"
        f"  observed pixel   = {observed.pixel_key}\n"
        f"  n_points         = {observed.t.size}\n"
        f"  x1(t) [m]        = filled array, range [{float(np.min(observed.x1)):.12e}, {float(np.max(observed.x1)):.12e}]\n"
        f"  x2(t) [m/s]      = filled array, range [{float(np.min(observed.x2)):.12e}, {float(np.max(observed.x2)):.12e}]\n\n"
        "Unknown placeholders not filled at this stage:\n"
        "  Fts(t)\n"
        "  x3(t)\n"
        "  ks, cs\n\n"
        "Actuation from in-air force balance:\n"
        "  F_actuation(t) = m_eff*x2dot_air + c_eff*x2_air + k_eff*x1_air\n\n"
        "State-space equations:\n"
        "  in_air:\n"
        "    x1dot = x2\n"
        "    x2dot = (F_actuation(t) - k_eff*x1 - c_eff*x2) / m_eff\n"
        "  normal/contact-capable:\n"
        "  x1dot = x2\n"
        "  x2dot = (F_actuation(t) - k_eff*x1 - c_eff*x2 + Fts) / m_eff\n"
        "  x3dot = (-Fts - ks*x3) / cs\n\n"
        "Equivalent residual equations:\n"
        "  m_eff*x2dot - (F_actuation(t) - k_eff*x1 - c_eff*x2 + Fts) = 0\n"
        "  cs*x3dot + ks*x3 + Fts = 0\n\n"
        "Geometry relation:\n"
        "  s = dist + x1 - x3\n"
        "  gap = s - a0\n"
    )


def main() -> None:
    print(equation_summary())


if __name__ == "__main__":
    main()
