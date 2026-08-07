"""Constants and required inputs for AFM06a hard-sample data generation."""

from __future__ import annotations

import math
from dataclasses import dataclass


PARAMETER_NAMES = (
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
)

LEGACY_PARAMETER_NAMES = (
    "omega0",
    "c1",
    "c2",
    "b1",
    "d1",
    "d2",
    "eta_star_length",
    "y_bar",
    "omega_bar",
    "dist",
    "a0",
    "beta",
)

# AFM06a constants supplied for the hard-sample formulation.
REFERENCE_FORCE_OMEGA0 = 11.804e3 * 2.0 * math.pi  # rad/s; coefficient reference angular frequency
C1 = -1.27462e-6
C2 = 4.63118
B1 = 1.56598
Y_BAR = 0.05585
OMEGA_BAR = 1.002
ETA_STAR_LENGTH = 8.88249e-9  # m; geometric length, not KV viscosity
A0_M = 0.165e-9  # m
A0_NORM = A0_M / ETA_STAR_LENGTH

# External silicon cantilever constants.
K_N_M = 22.68
F0_HZ = 164.52e3
OMEGA0 = 2.0 * math.pi * F0_HZ
Q = 428.0
M_KG = K_N_M / (OMEGA0**2)
C_N_S_M = M_KG * OMEGA0 / Q
D1 = C_N_S_M / (M_KG * OMEGA0)
D2 = D1

# Mass-normalized DMT force coefficients.  These are invariant when the
# cantilever/drive frequency is changed.
CA = C1 * (REFERENCE_FORCE_OMEGA0**2) * (ETA_STAR_LENGTH**3)
CH = C2 * (REFERENCE_FORCE_OMEGA0**2) / math.sqrt(ETA_STAR_LENGTH)

# Retained in the parameter vector for backward-compatible file interfaces.
# The current hard-contact law does not use a transition sharpness.
BETA = 5.0e11  # 1/m; inactive under CONTACT_SWITCH_POLICY="hard"
CONTACT_SWITCH_POLICY = "hard"
DIST = 100.0e-9  # m
X1_START = 0.0  # m
X2_START = 0.0  # m/s
INITIAL_TIME = 0.0  # s
END_TIME = 0.75e-3  # s
# Preserve the AFM04 sampling interval: 2 ms / 125000 = 16 ns.
DATA_NSTEPS = 46875
DATA_NUM_POINTS = DATA_NSTEPS + 1


def force_constants_for_omega0(omega0: float) -> tuple[float, float]:
    """Return C1/C2 rescaled so the effective DMT coefficients stay fixed."""

    value = float(omega0)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("omega0 must be positive and finite")
    scale = (float(REFERENCE_FORCE_OMEGA0) / value) ** 2
    return float(C1) * scale, float(C2) * scale


@dataclass(frozen=True)
class AFM06aHardSampleInputs:
    """Assigned configuration for one AFM06a simulation."""

    beta: float = BETA
    d1: float = D1
    d2: float = D2
    k_n_m: float = K_N_M
    mass_kg: float = M_KG
    c_n_s_m: float = C_N_S_M
    ca: float = CA
    ch: float = CH
    fd_n: float | None = None
    x1_start: float = X1_START
    x2_start: float = X2_START
    initial_time: float = INITIAL_TIME
    end_time: float = END_TIME
    data_nsteps: int = DATA_NSTEPS
    omega0: float = OMEGA0
    actuation_scale: float = 1.0
    dist_m: float = DIST
    a0_m: float = A0_M

    @property
    def a0(self) -> float:
        """DMT contact distance."""

        return float(self.a0_m)

    @property
    def dist(self) -> float:
        """Equilibrium tip-sample separation fixed by ``eta_star_length``."""

        return float(self.dist_m)

    @property
    def fd_over_m(self) -> float:
        """Return the mass-normalized actuation amplitude."""

        if self.fd_n is not None:
            return float(self.fd_n) / float(self.mass_kg)
        return (
            float(self.omega0) ** 2
            * float(ETA_STAR_LENGTH)
            * float(B1)
            * float(self.actuation_scale)
            * (float(OMEGA_BAR) ** 2)
            * float(Y_BAR)
        )

    @property
    def fd(self) -> float:
        """Return the physical actuation-force amplitude."""

        if self.fd_n is not None:
            return float(self.fd_n)
        return float(self.fd_over_m) * float(self.mass_kg)

    @property
    def initial_state(self) -> tuple[float, float]:
        """Return the two-state hard-sample initial condition."""

        return float(self.x1_start), float(self.x2_start)

    @property
    def parameter_vector(self) -> tuple[float, ...]:
        """Return parameters in the order expected by the RHS."""

        return (
            float(self.k_n_m),
            float(self.omega0),
            float(self.mass_kg),
            float(self.c_n_s_m),
            float(self.fd),
            float(self.ca),
            float(self.ch),
            float(self.dist),
            float(self.a0),
            float(self.beta),
        )

    @property
    def legacy_parameter_vector(self) -> tuple[float, ...]:
        """Return the older mass-scaled coefficient vector for archived tools."""

        c1 = float(self.ca) / (
            (float(self.omega0) ** 2) * (float(ETA_STAR_LENGTH) ** 3)
        )
        c2 = float(self.ch) * math.sqrt(float(ETA_STAR_LENGTH)) / (
            float(self.omega0) ** 2
        )
        b1 = float(self.fd_over_m) / (
            (float(self.omega0) ** 2)
            * float(ETA_STAR_LENGTH)
            * (float(OMEGA_BAR) ** 2)
            * float(Y_BAR)
        )
        damping_ratio = float(self.c_n_s_m) / (
            float(self.mass_kg) * float(self.omega0)
        )
        return (
            float(self.omega0),
            c1,
            c2,
            b1,
            damping_ratio,
            damping_ratio,
            float(ETA_STAR_LENGTH),
            float(Y_BAR),
            float(OMEGA_BAR),
            float(self.dist),
            float(self.a0),
            float(self.beta),
        )

    def validate(self) -> None:
        """Reject incomplete, non-finite, or physically invalid inputs."""

        scalar_fields = (
            "omega0",
            "beta",
            "d1",
            "d2",
            "k_n_m",
            "mass_kg",
            "c_n_s_m",
            "ca",
            "ch",
            "dist",
            "a0",
            "x1_start",
            "x2_start",
            "initial_time",
            "end_time",
            "actuation_scale",
        )
        for name in scalar_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.fd_n is not None and not math.isfinite(float(self.fd_n)):
            raise ValueError("fd_n must be finite when supplied")
        for name in ("omega0", "beta", "dist", "a0", "k_n_m", "mass_kg"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.actuation_scale) <= 0.0:
            raise ValueError("actuation_scale must be positive")
        for name in ("d1", "d2", "c_n_s_m"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.end_time <= self.initial_time:
            raise ValueError("end_time must be greater than initial_time")
        if isinstance(self.data_nsteps, bool) or int(self.data_nsteps) != self.data_nsteps:
            raise ValueError("data_nsteps must be an integer")
        if int(self.data_nsteps) < 1:
            raise ValueError("data_nsteps must be at least 1")


__all__ = [
    "AFM06aHardSampleInputs",
    "PARAMETER_NAMES",
    "LEGACY_PARAMETER_NAMES",
    "K_N_M",
    "F0_HZ",
    "Q",
    "M_KG",
    "C_N_S_M",
    "OMEGA0",
    "C1",
    "C2",
    "CA",
    "CH",
    "REFERENCE_FORCE_OMEGA0",
    "force_constants_for_omega0",
    "B1",
    "D1",
    "D2",
    "A0_M",
    "A0_NORM",
    "Y_BAR",
    "OMEGA_BAR",
    "ETA_STAR_LENGTH",
    "BETA",
    "CONTACT_SWITCH_POLICY",
    "DIST",
    "X1_START",
    "X2_START",
    "INITIAL_TIME",
    "END_TIME",
    "DATA_NSTEPS",
    "DATA_NUM_POINTS",
]
