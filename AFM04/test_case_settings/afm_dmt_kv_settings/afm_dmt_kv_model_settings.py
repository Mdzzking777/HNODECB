"""Shared AFM04 DMT-KV settings.

This file now follows the current force-law choice used in
`Datageneration/DMT_KV`, rather than the older AFM03 soft-gated `Fad*w`
definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Cantilever
k = 29.9
f0 = 313.57e3
wd = 2.0 * math.pi * f0
Q = 371.0
m = k / (wd**2)
c = m * wd / Q

# Contact & geometry
ESTAR = 15e6
ETA_STAR = 1.3
R = 10e-9
A = 6.0e-20
A0 = 3.0e-10
BETA = 5.0e11
DIST = 24e-9

# Drive
FD = 4.10e-9

# Surface (Kelvin-Voigt)
KS = 0.1
CS = 0.24e-6

# Dataset sampling density: match AFM03 exactly
DATA_NSTEPS = 125000
DATA_NUM_POINTS = DATA_NSTEPS + 1

NN_MONITOR_EFFECTIVE_CONTACT_FRAC = 0.01

PARAMETER_NAMES = (
    "k",
    "wd",
    "m",
    "c",
    "Fd",
    "R",
    "dist",
    "Estar",
    "eta_star",
    "A",
    "a0",
    "beta",
    "ks",
    "cs",
)

# Original model parameters.
# Order: k, wd, m, c, Fd, R, dist, Estar, eta_star, A, a0, beta, ks, cs
original_parameters = (
    k,
    wd,
    m,
    c,
    FD,
    R,
    DIST,
    ESTAR,
    ETA_STAR,
    A,
    A0,
    BETA,
    KS,
    CS,
)

# Initial conditions (x1, x2, x3)
original_u0 = (0.0, 0.0, 0.0)

# Initial time / end time
initial_time_training = 0.0
end_time_training = 2.0e-3


@dataclass(frozen=True)
class AFMDMTKVSettings:
    """Container for the shared AFM DMT-KV constants."""

    k: float = k
    f0: float = f0
    wd: float = wd
    Q: float = Q
    m: float = m
    c: float = c
    Estar: float = ESTAR
    eta_star: float = ETA_STAR
    R: float = R
    A: float = A
    a0: float = A0
    beta: float = BETA
    dist: float = DIST
    Fd: float = FD
    ks: float = KS
    cs: float = CS
    data_nsteps: int = DATA_NSTEPS
    data_num_points: int = DATA_NUM_POINTS
    nn_monitor_effective_contact_frac: float = NN_MONITOR_EFFECTIVE_CONTACT_FRAC
    original_parameters: tuple[float, ...] = original_parameters
    original_u0: tuple[float, float, float] = original_u0
    initial_time_training: float = initial_time_training
    end_time_training: float = end_time_training


DEFAULT_SETTINGS = AFMDMTKVSettings()
