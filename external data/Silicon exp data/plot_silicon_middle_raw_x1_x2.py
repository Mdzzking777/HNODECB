from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MIDDLE_NPZ = (
    SCRIPT_DIR
    / "GP"
    / "middle"
    / "silicon_gp_index109_mid5000_490_510us_x2_x2dot.npz"
)
OUTPUT = SCRIPT_DIR / "silicon_raw_forward_index109_x1_x2_windows.png"


def main() -> int:
    data = np.load(MIDDLE_NPZ)
    time_us = data["time_s"].astype(float) * 1.0e6
    x1_nm = data["x1_raw_m"].astype(float) * 1.0e9
    x2 = data["x2_raw_m_s"].astype(float)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5.6),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    axes[0].plot(time_us, x1_nm, color="black", linewidth=0.9)
    axes[1].plot(time_us, x2, color="black", linewidth=0.9)

    axes[0].set_ylabel(r"$x_1^{\mathrm{raw}}$ [nm]")
    axes[1].set_ylabel(r"$x_2^{\mathrm{raw}}$ [m s$^{-1}$]")
    axes[1].set_xlabel(r"Time [$\mu$s]")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.0)

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
