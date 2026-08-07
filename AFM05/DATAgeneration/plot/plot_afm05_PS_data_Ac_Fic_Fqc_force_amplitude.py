"""Plot AFM05 force quadratures against oscillation amplitude."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "AFM05" / "DATAgeneration"
MAT_PATH = DATA_DIR / "PS_data.mat"
TRACE_PATH = DATA_DIR / "PS_cantilever_disp_vel_time_3_pixels_Z_85.0_a0_0.07108_file_scan04143.imp_.npz"
OUT_PATH = DATA_DIR / "plot" / "afm05_PS_data_Ac_Fic_Fqc_force_amplitude.png"


def _initial_contact_amplitude_nm() -> float:
    with np.load(TRACE_PATH, allow_pickle=True) as data:
        metadata = dict(data["add_data_dict"].item())
    return abs(float(metadata["AFM05_x1_init_nm"]))


def main() -> None:
    source = loadmat(MAT_PATH, squeeze_me=True)
    amplitude_nm = np.asarray(source["Ac"], dtype=float).reshape(-1) * 1.0e9
    fi_nN = np.asarray(source["Fic"], dtype=float).reshape(-1) * 1.0e9
    fq_nN = np.asarray(source["Fqc"], dtype=float).reshape(-1) * 1.0e9
    if not (amplitude_nm.size == fi_nN.size == fq_nN.size):
        raise ValueError("Ac, Fic, and Fqc must have equal lengths")

    # PS_data contains repeated decreasing/increasing sweeps. The first
    # increasing branch is indices 239..467 (inclusive).
    decreasing_runs = ((0, 239), (467, 724), (951, amplitude_nm.size - 1))
    increasing_runs = ((239, 467), (724, 951))

    fig, ax = plt.subplots(figsize=(8.2, 7.65))
    blue = "#1f77b4"
    orange = "#e67e2f"

    for run_index, (lo, hi) in enumerate(decreasing_runs):
        sl = slice(lo, hi + 1)
        ax.plot(
            amplitude_nm[sl],
            fi_nN[sl],
            color=blue,
            linewidth=3.0,
            label=r"In phase ($F_I$), decreasing amplitude" if run_index == 0 else "_nolegend_",
        )
        ax.plot(
            amplitude_nm[sl],
            fq_nN[sl],
            color=orange,
            linewidth=3.0,
            label=r"Out of phase ($F_Q$), decreasing amplitude" if run_index == 0 else "_nolegend_",
        )

    for run_index, (lo, hi) in enumerate(increasing_runs):
        sl = slice(lo, hi + 1)
        ax.plot(
            amplitude_nm[sl],
            fi_nN[sl],
            color=blue,
            linestyle="--",
            linewidth=3.0,
            label=r"In phase ($F_I$), increasing amplitude" if run_index == 0 else "_nolegend_",
        )
        ax.plot(
            amplitude_nm[sl],
            fq_nN[sl],
            color=orange,
            linestyle="--",
            linewidth=3.0,
            label=r"Out of phase ($F_Q$), increasing amplitude" if run_index == 0 else "_nolegend_",
        )

    contact_amplitude_nm = _initial_contact_amplitude_nm()
    ax.axvline(
        contact_amplitude_nm,
        color="black",
        linestyle="--",
        linewidth=1.5,
        zorder=1,
    )

    ax.set_xlabel("Amplitude (nm)", fontsize=28, fontfamily="serif")
    ax.set_ylabel("Force (nN)", fontsize=28, fontfamily="serif")
    ax.set_xlim(-2.0, 89.0)
    ax.set_ylim(-3.3, 0.37)
    ax.tick_params(which="major", direction="in", top=True, right=True, width=2.6, length=9, labelsize=22)
    ax.tick_params(which="minor", direction="in", top=True, right=True, width=2.0, length=5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(2.8)
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.055, 0.235),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        fontsize=15.5,
        handlelength=2.0,
    )
    legend.get_frame().set_linewidth(1.3)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, facecolor="white")
    plt.close(fig)

    lo, hi = increasing_runs[0]
    branch_a = amplitude_nm[lo : hi + 1]
    branch_fi = fi_nN[lo : hi + 1]
    branch_fq = fq_nN[lo : hi + 1]
    fi_contact = float(np.interp(contact_amplitude_nm, branch_a, branch_fi))
    fq_contact = float(np.interp(contact_amplitude_nm, branch_a, branch_fq))
    print(f"Saved: {OUT_PATH}")
    print(f"Initial-contact amplitude: {contact_amplitude_nm:.12f} nm")
    print(f"First increasing sweep F_I: {fi_contact:.12f} nN")
    print(f"First increasing sweep F_Q: {fq_contact:.12f} nN")
    print(f"First increasing sweep F_I + F_Q: {fi_contact + fq_contact:.12f} nN")


if __name__ == "__main__":
    main()
