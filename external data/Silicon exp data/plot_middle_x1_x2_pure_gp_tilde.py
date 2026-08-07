from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
X1_TILDE_NPZ = (
    ROOT
    / "GP"
    / "middle"
    / "x1 for contact judge"
    / "silicon_gp_index109_mid5000_490_510us_x1_contact_judge.npz"
)
X2_PURE_GP_NPZ = (
    ROOT
    / "smooth"
    / "x2 raw"
    / "silicon_middle_x2_raw_global_gp_smooth.npz"
)
OUTPUT = ROOT / "silicon_middle_x1_x2_pure_gp_tilde.png"


def main() -> int:
    with np.load(X1_TILDE_NPZ, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        x1_tilde = np.asarray(data["x1_tilde_m"], dtype=float)
        transition_idx = np.asarray(data["transition_idx"], dtype=int)

    with np.load(X2_PURE_GP_NPZ, allow_pickle=False) as data:
        x2_time_s = np.asarray(data["time_s"], dtype=float)
        x2_tilde = np.asarray(data["x2_tilde_m_s"], dtype=float)
        x2dot_tilde = np.asarray(data["x2dot_tilde_m_s2"], dtype=float)

    if time_s.shape != x2_time_s.shape or not np.allclose(time_s, x2_time_s, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("x1_tilde and pure-GP x2_tilde are not on the same middle-slice time grid")

    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 7.8), sharex=True)
    panels = (
        (x1_tilde * 1.0e9, r"$\tilde{x}_1$ [nm]"),
        (x2_tilde * 1.0e3, r"$\tilde{x}_{2,\mathrm{pure}}$ [mm s$^{-1}$]"),
        (x2dot_tilde, r"$\dot{\tilde{x}}_{2,\mathrm{pure}}$ [m s$^{-2}$]"),
    )
    for ax, (values, ylabel) in zip(axes, panels):
        ax.plot(time_us, values, color="black", linewidth=0.95)
        for idx in transition_idx:
            ax.axvline(time_us[int(idx)], color="0.45", linewidth=0.55, alpha=0.28)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)

    axes[-1].set_xlabel(r"Time [$\mu$s]")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
