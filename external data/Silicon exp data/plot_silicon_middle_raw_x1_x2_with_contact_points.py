from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MIDDLE_RAW_NPZ = (
    SCRIPT_DIR
    / "GP"
    / "middle"
    / "silicon_gp_index109_mid5000_490_510us_x2_x2dot.npz"
)
CONTACT_JUDGE_NPZ = (
    SCRIPT_DIR
    / "GP"
    / "middle"
    / "x1 for contact judge"
    / "silicon_gp_index109_mid5000_490_510us_x1_contact_judge.npz"
)
OUTPUT_DIR = SCRIPT_DIR / "data" / "middle"
OUTPUT_PATH = OUTPUT_DIR / "silicon_middle_raw_x1_x2_contact_points.png"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(MIDDLE_RAW_NPZ, allow_pickle=False) as raw_data:
        time_s = np.asarray(raw_data["time_s"], dtype=float)
        source_idx = np.asarray(raw_data["source_idx"], dtype=np.int64)
        x1_raw_nm = 1.0e9 * np.asarray(raw_data["x1_raw_m"], dtype=float)
        x2_raw = np.asarray(raw_data["x2_raw_m_s"], dtype=float)

    with np.load(CONTACT_JUDGE_NPZ, allow_pickle=False) as judge_data:
        judge_source_idx = np.asarray(judge_data["source_idx"], dtype=np.int64)
        contact_tilde = np.asarray(judge_data["contact_tilde"], dtype=bool)
        transition_idx = np.asarray(judge_data["transition_idx"], dtype=np.int64)

    if not np.array_equal(source_idx, judge_source_idx):
        raise ValueError("middle raw data and contact-judge data use different source indices")
    if contact_tilde.shape != time_s.shape:
        raise ValueError("contact mask length does not match middle raw time array")

    time_us = time_s * 1.0e6
    transition_points_idx = transition_idx

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5.8),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    series = (
        (x1_raw_nm, r"$x_1^{\mathrm{raw}}$ [nm]"),
        (x2_raw, r"$x_2^{\mathrm{raw}}$ [m s$^{-1}$]"),
    )

    for ax, (values, ylabel) in zip(axes, series, strict=True):
        ax.plot(time_us, values, color="0.45", linewidth=0.8, label="raw")
        ax.scatter(
            time_us[transition_points_idx],
            values[transition_points_idx],
            s=36,
            color="black",
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
            label="transition points",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)

    axes[1].set_xlabel(r"Time [$\mu$s]")
    handles, labels = axes[0].get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=True):
        dedup.setdefault(label, handle)
    axes[0].legend(dedup.values(), dedup.keys(), loc="upper right")

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Transition points: {transition_points_idx.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
