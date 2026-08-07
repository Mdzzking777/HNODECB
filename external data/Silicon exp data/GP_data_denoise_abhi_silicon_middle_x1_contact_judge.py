from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from GP_data_denoise_abhi_silicon import (
    A0_M,
    GP_LENGTH_SCALE_REFERENCE,
    Z_STATIC_M,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MIDDLE_DIR = SCRIPT_DIR / "GP" / "middle"
INPUT_NPZ = MIDDLE_DIR / "silicon_gp_index109_mid5000_490_510us_x2_x2dot.npz"
OUTPUT_DIR = MIDDLE_DIR / "x1 for contact judge"
OUTPUT_STEM = "silicon_gp_index109_mid5000_490_510us_x1_contact_judge"


def _fit_gp_x1_contact_judge(
    *,
    time_s: np.ndarray,
    x1_raw_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    # Fit in standardized nm coordinates so the GP kernel is not dominated by
    # the small SI-unit scale of x1.
    x1_raw_nm = 1.0e9 * np.asarray(x1_raw_m, dtype=np.float64)
    y_mean = float(np.mean(x1_raw_nm))
    y_scale = float(np.std(x1_raw_nm))
    if y_scale <= 0.0:
        raise ValueError("x1 has zero standard deviation")
    y = (x1_raw_nm - y_mean) / y_scale

    kernel = RBF(
        length_scale=1.0 / GP_LENGTH_SCALE_REFERENCE,
        length_scale_bounds=(
            1e-1 / GP_LENGTH_SCALE_REFERENCE,
            1e3 / GP_LENGTH_SCALE_REFERENCE,
        ),
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(time_s.reshape(-1, 1), y)
    y_tilde, y_std = gp.predict(time_s.reshape(-1, 1), return_std=True)

    x1_tilde_nm = y_tilde * y_scale + y_mean
    x1_std_nm = y_std * y_scale
    metadata = {
        "gp_kernel_initial": str(kernel),
        "gp_kernel_optimized": str(gp.kernel_),
        "x1_gp_internal_units": "standardized nm",
        "x1_gp_output_units": "m",
        "x1_standardization_mean_nm": y_mean,
        "x1_standardization_std_nm": y_scale,
        "usage": (
            "x1_tilde is only for contact-point / regime judgment; "
            "it is not a model prediction and should not be denoted with hat."
        ),
    }
    return 1.0e-9 * x1_tilde_nm, 1.0e-9 * x1_std_nm, metadata


def _transition_indices(contact: np.ndarray) -> np.ndarray:
    contact = np.asarray(contact, dtype=bool)
    return np.flatnonzero(contact[1:] != contact[:-1]) + 1


def main() -> int:
    start_wall = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(INPUT_NPZ, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        source_idx = np.asarray(data["source_idx"], dtype=np.int64)
        x1_raw_m = np.asarray(data["x1_raw_m"], dtype=np.float64)

    x1_tilde_m, x1_gp_std_m, gp_metadata = _fit_gp_x1_contact_judge(
        time_s=time_s,
        x1_raw_m=x1_raw_m,
    )

    separation_tilde_m = Z_STATIC_M + x1_tilde_m
    contact_tilde = separation_tilde_m <= A0_M
    transition_idx = _transition_indices(contact_tilde)

    output_npz = OUTPUT_DIR / f"{OUTPUT_STEM}.npz"
    output_png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    output_json = OUTPUT_DIR / f"{OUTPUT_STEM}.json"

    np.savez_compressed(
        output_npz,
        time_s=time_s,
        source_idx=source_idx,
        x1_raw_m=x1_raw_m,
        x1_tilde_m=x1_tilde_m,
        x1_gp_std_m=x1_gp_std_m,
        z_static_m=np.array(Z_STATIC_M, dtype=np.float64),
        a0_m=np.array(A0_M, dtype=np.float64),
        separation_tilde_m=separation_tilde_m,
        contact_tilde=contact_tilde,
        transition_idx=transition_idx,
    )

    metadata = {
        "dataset": "silicon forward sweep",
        "index": 109,
        "segment": "middle 5000 points centered near 500 us",
        "input_npz": str(INPUT_NPZ.resolve()),
        "output_npz": str(output_npz.resolve()),
        "start_time_s": float(time_s[0]),
        "end_time_s": float(time_s[-1]),
        "sample_count": int(time_s.size),
        "z_static_m": Z_STATIC_M,
        "a0_m": A0_M,
        "contact_rule": "contact if z_static + x1_tilde <= a0",
        "transition_count": int(transition_idx.size),
        **gp_metadata,
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5.8),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    axes[0].plot(time_us, 1.0e9 * x1_raw_m, color="0.45", linewidth=0.7, label="raw")
    axes[0].plot(
        time_us,
        1.0e9 * x1_tilde_m,
        color="#1f77b4",
        linewidth=1.0,
        label=r"$\tilde{x}_1$",
    )
    axes[1].plot(time_us, 1.0e9 * separation_tilde_m, color="#1f77b4", linewidth=1.0)
    axes[1].axhline(1.0e9 * A0_M, color="crimson", linestyle="--", linewidth=1.0)

    for axis in axes:
        if transition_idx.size:
            axis.scatter(
                time_us[transition_idx],
                np.interp(time_us[transition_idx], time_us, axis.lines[0].get_ydata()),
                s=18,
                color="black",
                zorder=5,
            )
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.0)

    axes[0].set_ylabel(r"$x_1$ [nm]")
    axes[1].set_ylabel(r"$Z+\tilde{x}_1$ [nm]")
    axes[1].set_xlabel(r"Time [$\mu$s]")
    axes[0].legend(loc="upper right")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - start_wall
    print(f"Saved: {output_npz}")
    print(f"Saved: {output_png}")
    print(f"Saved: {output_json}")
    print(f"Contact transitions: {transition_idx.size}")
    print(f"Wall time: {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
