from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from GP_data_denoise_abhi_silicon import (
    DEFAULT_MAT_PATH,
    EXPERIMENTAL_DT_S,
    fit_gp_x2,
    load_index_signals,
)


SCRIPT_DIR = Path(__file__).resolve().parent
INDEX = 109
CENTER_TIME_S = 500.0e-6
SAMPLE_COUNT = 5000

START_INDEX = int(round(CENTER_TIME_S / EXPERIMENTAL_DT_S)) - SAMPLE_COUNT // 2
OUTPUT_STEM = "silicon_gp_index109_mid5000_490_510us_x2_x2dot"


def main() -> None:
    start_wall = time.time()
    x1_raw, x2_raw, time_s, source_idx = load_index_signals(
        DEFAULT_MAT_PATH,
        index=INDEX,
        start_index=START_INDEX,
        sample_count=SAMPLE_COUNT,
    )
    x2_hat = fit_gp_x2(x2_raw, time_s)
    x2dot_hat = np.gradient(x2_hat, time_s, edge_order=2)

    output_npz = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}.json"

    np.savez_compressed(
        output_npz,
        time_s=time_s,
        source_idx=source_idx,
        segment_start_index=np.array(int(source_idx[0]), dtype=np.int64),
        segment_end_index_exclusive=np.array(int(source_idx[-1]) + 1, dtype=np.int64),
        x1_raw_m=x1_raw,
        x2_raw_m_s=x2_raw,
        x2_hat_m_s=x2_hat,
        x2dot_hat_m_s2=x2dot_hat,
    )

    metadata = {
        "dataset": "silicon forward sweep",
        "index": INDEX,
        "segment": "middle 5000 points centered near 500 us",
        "start_index": int(source_idx[0]),
        "end_index_exclusive": int(source_idx[-1]) + 1,
        "sample_count": int(time_s.size),
        "start_time_s": float(time_s[0]),
        "end_time_s": float(time_s[-1]),
        "dt_s": EXPERIMENTAL_DT_S,
        "gp_target": "x2_raw_m_s",
        "x2dot_method": "np.gradient(x2_hat, time_s, edge_order=2)",
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    time_us = time_s * 1.0e6
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(10, 9),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    axes[0].plot(time_us, x1_raw * 1e9, color="black", linewidth=0.8)
    axes[1].plot(time_us, x2_raw, color="black", linewidth=0.8)
    axes[2].plot(time_us, x2_hat, color="blue", linewidth=0.8)
    axes[3].plot(time_us, x2dot_hat, color="red", linewidth=0.8)
    axes[0].set_ylabel(r"$x_1^{\mathrm{raw}}$ [nm]")
    axes[1].set_ylabel(r"$x_2^{\mathrm{raw}}$ [m s$^{-1}$]")
    axes[2].set_ylabel(r"$\hat{x}_2$ [m s$^{-1}$]")
    axes[3].set_ylabel(r"$\widehat{\dot{x}}_2$ [m s$^{-2}$]")
    axes[3].set_xlabel(r"Time [$\mu$s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)

    minutes, seconds = divmod(time.time() - start_wall, 60.0)
    print(f"Saved: {output_npz}")
    print(f"Saved: {output_png}")
    print(f"Saved: {output_json}")
    print(f"Segment: {time_us[0]:.6f} us to {time_us[-1]:.6f} us")
    print(f"Wall time: {int(minutes)} min {seconds:.1f} s")


if __name__ == "__main__":
    main()
