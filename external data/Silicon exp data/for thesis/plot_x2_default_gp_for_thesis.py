"""Plot the default index109 x2 GP result used by the original GP script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent
SOURCE_SCRIPT = SOURCE_DIR / "GP_data_denoise_abhi_silicon.py"
OUTPUT_PATH = SCRIPT_DIR / "silicon_index109_x2_default_gp_x2dot.png"
WINDOW_START_US = 470.0
WINDOW_DURATION_US = 60.0
GP_FIT_STRIDE = 4


def _load_source_module():
    spec = importlib.util.spec_from_file_location("silicon_default_gp", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    source = _load_source_module()
    start_index = int(round(WINDOW_START_US * 1.0e-6 / source.EXPERIMENTAL_DT_S))
    sample_count = int(round(WINDOW_DURATION_US * 1.0e-6 / source.EXPERIMENTAL_DT_S)) + 1
    _, x2_raw, time_s, _ = source.load_index_signals(
        source.DEFAULT_MAT_PATH,
        index=source.INDEX,
        start_index=start_index,
        sample_count=sample_count,
    )
    fit_indices = np.arange(0, time_s.size, GP_FIT_STRIDE, dtype=int)
    if fit_indices[-1] != time_s.size - 1:
        fit_indices = np.append(fit_indices, time_s.size - 1)
    kernel = source.RBF(
        length_scale=1.0 / source.GP_LENGTH_SCALE_REFERENCE,
        length_scale_bounds=(
            1.0e-1 / source.GP_LENGTH_SCALE_REFERENCE,
            1.0e3 / source.GP_LENGTH_SCALE_REFERENCE,
        ),
    ) + source.WhiteKernel(noise_level=0.1, noise_level_bounds=(1.0e-5, 1.0e1))
    gp = source.GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
    )
    gp.fit(time_s[fit_indices, None], x2_raw[fit_indices])
    x2_tilde = np.asarray(gp.predict(time_s[:, None]), dtype=np.float64)
    x2dot_tilde = np.gradient(x2_tilde, time_s, edge_order=2)

    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 8.0),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    axes[0].plot(time_us, x2_raw, color="black", linewidth=0.7)
    axes[1].plot(time_us, x2_tilde, color="royalblue", linewidth=1.0)
    axes[2].plot(time_us, x2dot_tilde, color="#d62728", linewidth=1.0)
    axes[0].set_ylabel(r"$x_2^{\mathrm{raw}}$ [m s$^{-1}$]")
    axes[1].set_ylabel(r"$\widetilde{x}_2$ [m s$^{-1}$]")
    axes[2].set_ylabel(r"$\dot{\widetilde{x}}_2$ [m s$^{-2}$]")
    axes[2].set_xlabel(r"Time [$\mu$s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.0)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")
    print(
        f"window=[{time_s[0] * 1e6:.6f}, {time_s[-1] * 1e6:.6f}] us | "
        f"raw points={time_s.size} | GP support points={fit_indices.size} | "
        f"optimized kernel={gp.kernel_}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
