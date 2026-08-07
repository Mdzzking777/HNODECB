from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_NPZ = SCRIPT_DIR / "silicon_gp_index109_training_window_2cycles.npz"
OUTPUT_PNG = SCRIPT_DIR / "silicon_gp_index109_training_window_2cycles_no_points.png"


def main() -> None:
    data = np.load(INPUT_NPZ)

    time_us = data["time_s"] * 1.0e6
    series = [
        (data["x1_raw_m"] * 1e9, r"$x_1^{\mathrm{raw}}$ [nm]"),
        (data["x2_raw_m_s"], r"$x_2^{\mathrm{raw}}$ [m s$^{-1}$]"),
        (data["x2_hat_m_s"], r"$\hat{x}_2$ [m s$^{-1}$]"),
        (data["x2dot_hat_m_s2"], r"$\widehat{\dot{x}}_2$ [m s$^{-2}$]"),
        (data["f_residual_N"] * 1e9, r"$F_{\mathrm{residual}}$ [nN]"),
    ]

    figure, axes = plt.subplots(
        5,
        1,
        figsize=(10, 11),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    for axis, (values, ylabel) in zip(axes, series):
        axis.plot(time_us, values, color="black", linewidth=0.7, alpha=0.65)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel(r"Time [$\mu$s]")
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
