"""Integrate representative (ks, cs) pairs with the standalone AFM05 model."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from afm05_1d_surface_model import KelvinVoigt1D, load_full_domain_force


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURE_PATH = OUTPUT_DIR / "afm05_1d_x3_representative_ks_cs.png"
W0_FIGURE_PATH = OUTPUT_DIR / "afm05_1d_x3_representative_ks_cs_W0.png"
SUMMARY_PATH = OUTPUT_DIR / "afm05_1d_x3_representative_ks_cs_summary.csv"
W0_START_S = 0.605488e-3
W0_STOP_S = W0_START_S + 25.152e-6

# Representative points drawn from the current 10 x 10 logarithmic grid.
CASES = (
    ("low ks, low cs", 1.0e-3, 5.0e-6),
    ("low ks, high cs", 1.0e-3, 5.0e-3),
    ("middle grid", 1.583223e-2, 1.077217e-4),
    ("archive-overlap region", 1.256605e-1, 2.320794e-5),
    ("high ks, middle cs", 5.0e-1, 5.0e-4),
    ("high ks, low cs", 5.0e-1, 5.0e-6),
)


def main() -> int:
    force = load_full_domain_force()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), sharex=True)
    w0_fig, w0_axes = plt.subplots(3, 2, figsize=(12.0, 10.0), sharex=True)
    records: list[dict[str, float | str]] = []
    display_stride = max(1, force.t.size // 12000)
    w0_mask = (force.t >= W0_START_S) & (force.t <= W0_STOP_S)

    for ax, w0_ax, (label, ks, cs) in zip(axes.flat, w0_axes.flat, CASES, strict=True):
        model = KelvinVoigt1D(ks=ks, cs=cs)
        trajectory = model.integrate(force, x3_initial=0.0)
        x3_nm = trajectory.x3 * 1.0e9
        residual = model.residual(trajectory.x3, trajectory.x3dot, trajectory.fts)

        ax.plot(
            trajectory.t[::display_stride] * 1.0e3,
            x3_nm[::display_stride],
            color="tab:purple",
            linewidth=1.0,
        )
        ax.set_title(
            rf"{label}: $k_s={ks:.3g}$ N/m, $c_s={cs:.3g}$ N s/m"
            "\n"
            rf"$\tau_s=c_s/k_s={cs / ks * 1.0e6:.3g}$ $\mu$s"
        )
        ax.set_ylabel(r"$x_3$ [nm]")
        ax.grid(True, alpha=0.25)

        w0_x3_nm = x3_nm[w0_mask]
        w0_ax.plot(
            trajectory.t[w0_mask] * 1.0e6,
            w0_x3_nm,
            color="tab:purple",
            linewidth=1.0,
        )
        w0_ax.set_title(
            rf"{label}: $k_s={ks:.3g}$ N/m, $c_s={cs:.3g}$ N s/m"
            "\n"
            rf"$\tau_s={cs / ks * 1.0e6:.3g}$ $\mu$s"
        )
        w0_ax.set_ylabel(r"$x_3$ [nm]")
        w0_ax.grid(True, alpha=0.25)

        records.append(
            {
                "case": label,
                "ks_N_per_m": ks,
                "cs_Ns_per_m": cs,
                "tau_s": cs / ks,
                "x3_initial_nm": x3_nm[0],
                "x3_final_nm": x3_nm[-1],
                "x3_min_nm": float(np.min(x3_nm)),
                "x3_max_nm": float(np.max(x3_nm)),
                "x3_mean_nm": float(np.mean(x3_nm)),
                "W0_x3_initial_nm": float(w0_x3_nm[0]),
                "W0_x3_final_nm": float(w0_x3_nm[-1]),
                "W0_x3_min_nm": float(np.min(w0_x3_nm)),
                "W0_x3_max_nm": float(np.max(w0_x3_nm)),
                "max_abs_balance_residual_N": float(np.max(np.abs(residual))),
            }
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("time [ms]")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=240, facecolor="white")
    plt.close(fig)

    for ax in w0_axes[-1, :]:
        ax.set_xlabel(r"time [$\mu$s]")
    w0_fig.tight_layout()
    w0_fig.savefig(W0_FIGURE_PATH, dpi=240, facecolor="white")
    plt.close(w0_fig)

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved: {FIGURE_PATH}")
    print(f"Saved: {W0_FIGURE_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
