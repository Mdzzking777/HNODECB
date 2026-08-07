"""Standalone one-dimensional AFM05 surface model.

The model is the Kelvin-Voigt balance

    cs * x3dot(t) + ks * x3(t) = -Fts(t),

where the complete AFM05 reconstructed force trajectory is prescribed directly
from the unsliced DATAgeneration trace.  Integration starts at the global time
origin with ``x3(0) = 0``.  This module does not use the cantilever states, a
KAN, or any stage1/stage2 training result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


Array = np.ndarray
AFM05_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORCE_PATH = AFM05_ROOT / "DATAgeneration" / (
    "PS_cantilever_disp_vel_time_3_pixels_Z_85.0_a0_0.07108_file_scan04143.imp_.npz"
)
PIXEL_KEY = "pixel_184_152"
FORCE_KEY = "F_ts_184_152"


@dataclass(frozen=True)
class PrescribedForce:
    """Full-domain prescribed force samples in SI units."""

    t: Array
    fts: Array


@dataclass(frozen=True)
class SurfaceTrajectory:
    """Integrated one-dimensional surface trajectory in SI units."""

    t: Array
    fts: Array
    x3: Array
    x3dot: Array


@dataclass(frozen=True)
class KelvinVoigt1D:
    """One-dimensional Kelvin-Voigt surface with prescribed force."""

    ks: float
    cs: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.ks) or self.ks <= 0.0:
            raise ValueError("ks must be a finite positive value in N/m")
        if not np.isfinite(self.cs) or self.cs <= 0.0:
            raise ValueError("cs must be a finite positive value in N*s/m")

    def rhs(self, x3: Array | float, fts: Array | float) -> Array:
        """Evaluate x3dot = (-Fts - ks*x3) / cs."""

        return (-np.asarray(fts, dtype=float) - self.ks * np.asarray(x3, dtype=float)) / self.cs

    def residual(self, x3: Array, x3dot: Array, fts: Array) -> Array:
        """Evaluate cs*x3dot + ks*x3 + Fts, whose target is zero."""

        return self.cs * np.asarray(x3dot) + self.ks * np.asarray(x3) + np.asarray(fts)

    def integrate(self, force: PrescribedForce, x3_initial: float) -> SurfaceTrajectory:
        """Integrate over every full-domain force sample.

        On each time interval, the force is represented by its midpoint value
        and the resulting linear ODE is advanced analytically.  This avoids a
        numerical-stability restriction when cs/ks is small.
        """

        if not np.isfinite(x3_initial):
            raise ValueError("x3_initial must be finite")

        t = force.t
        fts = force.fts
        dt = np.diff(t)
        decay = np.exp(-(self.ks / self.cs) * dt)
        force_mid = 0.5 * (fts[:-1] + fts[1:])
        equilibrium = -force_mid / self.ks

        x3 = np.empty_like(t)
        x3[0] = float(x3_initial)
        for index in range(dt.size):
            x3[index + 1] = decay[index] * x3[index] + (1.0 - decay[index]) * equilibrium[index]

        x3dot = self.rhs(x3, fts)
        return SurfaceTrajectory(t=t.copy(), fts=fts.copy(), x3=x3, x3dot=x3dot)


def load_full_domain_force(path: Path = DEFAULT_FORCE_PATH) -> PrescribedForce:
    """Load and validate the complete reconstructed AFM05 Fts trajectory."""

    with np.load(path, allow_pickle=True) as archive:
        trace = np.asarray(archive[PIXEL_KEY], dtype=float)
        t = np.asarray(trace[:, 3], dtype=float)
        fts = np.asarray(archive[FORCE_KEY], dtype=float)

    if t.ndim != 1 or fts.ndim != 1 or t.shape != fts.shape:
        raise ValueError("t and Fts must be matching one-dimensional arrays")
    if t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("the prescribed force time grid must be strictly increasing")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(fts)):
        raise ValueError("the prescribed force trajectory contains non-finite values")
    if not np.isclose(t[0], 0.0, rtol=0.0, atol=1.0e-15):
        raise ValueError(f"the full-domain force must begin at t=0, got {t[0]:.12e} s")
    return PrescribedForce(t=t, fts=fts)


def run_model(*, ks: float, cs: float, output: Path | None = None) -> SurfaceTrajectory:
    """Load the AFM05 force, integrate the 1D model, and optionally save it."""

    trajectory = KelvinVoigt1D(ks=ks, cs=cs).integrate(
        load_full_domain_force(),
        x3_initial=0.0,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            t=trajectory.t,
            Fts=trajectory.fts,
            x3=trajectory.x3,
            x3dot=trajectory.x3dot,
            ks_N_per_m=float(ks),
            cs_Ns_per_m=float(cs),
            x3_initial_m=0.0,
        )
    return trajectory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks", type=float, required=True, help="surface stiffness [N/m]")
    parser.add_argument("--cs", type=float, required=True, help="surface damping [N*s/m]")
    parser.add_argument("--output", type=Path, help="optional output .npz path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    trajectory = run_model(ks=args.ks, cs=args.cs, output=args.output)
    print(
        f"Integrated {trajectory.t.size} samples from "
        f"{trajectory.t[0] * 1e6:.6f} to {trajectory.t[-1] * 1e6:.6f} us."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
