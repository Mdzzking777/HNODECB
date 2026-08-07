from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from GP_data_denoise_abhi_silicon import A0_M, Z_STATIC_M, force_residual


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_NPZ = SCRIPT_DIR / "silicon_gp_index109_mid5000_490_510us_x2_x2dot.npz"
OUTPUT_PNG = SCRIPT_DIR / "silicon_gp_index109_mid5000_490_510us_x2_x2dot.png"


def main() -> None:
    data = np.load(INPUT_NPZ)
    time_s = data["time_s"]
    x1_raw = data["x1_raw_m"]
    x2_raw = data["x2_raw_m_s"]
    x2_hat = data["x2_hat_m_s"]
    x2dot_hat = data["x2dot_hat_m_s2"]

    f_residual, _omega0, _mass, _damping = force_residual(
        x1_m=x1_raw,
        x2_hat_m_s=x2_hat,
        x2dot_hat_m_s2=x2dot_hat,
    )

    contact = (Z_STATIC_M + x1_raw) <= A0_M
    transition_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1

    time_us = time_s * 1.0e6
    series = [
        (x1_raw * 1e9, r"$x_1^{\mathrm{raw}}$ [nm]", "black"),
        (x2_raw, r"$x_2^{\mathrm{raw}}$ [m s$^{-1}$]", "black"),
        (x2_hat, r"$\hat{x}_2$ [m s$^{-1}$]", "blue"),
        (x2dot_hat, r"$\widehat{\dot{x}}_2$ [m s$^{-2}$]", "red"),
        (f_residual * 1e9, r"$F_{\mathrm{residual}}$ [nN]", "black"),
    ]

    figure, axes = plt.subplots(
        5,
        1,
        figsize=(10, 11),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    for axis, (values, ylabel, color) in zip(axes, series):
        axis.plot(time_us, values, color=color, linewidth=0.8, alpha=0.8)
        if transition_idx.size:
            axis.scatter(
                time_us[transition_idx],
                values[transition_idx],
                s=22,
                color="black",
                edgecolors="white",
                linewidths=0.35,
                zorder=6,
            )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel(r"Time [$\mu$s]")
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Contact transitions: {int(transition_idx.size)}")


if __name__ == "__main__":
    main()
