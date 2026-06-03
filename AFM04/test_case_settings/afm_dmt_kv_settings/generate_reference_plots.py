"""Generate AFM04-local DMT-KV reference plots from the shared truth layer.

This script is intentionally independent from `Datageneration/DMT_KV`.
It uses the formal AFM04 dataset generator and shared AFM04 physics, then
emits a small set of inspection plots so the current AFM04 force law can be
checked by eye.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from AFM04.datasets.afm_dataset_generator import generate_afm_dmt_kv_dataset
from AFM04.stage1pluslight.windows import stage2_window_manifest
from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import DEFAULT_SETTINGS


ZOOM_START_US = 1600.0
ZOOM_END_US = 1650.0
CSV_COLUMNS = ("t", "x1", "x2", "x3", "x2dot", "contact", "s", "x3dot", "delta_dot", "fts")


def _save_reference_csv(output_dir: Path, afm_table: dict[str, np.ndarray]) -> Path:
    csv_path = output_dir / "reference_trajectory.csv"
    matrix = np.column_stack([np.asarray(afm_table[name], dtype=float) for name in CSV_COLUMNS])
    np.savetxt(
        csv_path,
        matrix,
        delimiter=",",
        header=",".join(CSV_COLUMNS),
        comments="",
    )
    return csv_path


def _save_trajectory_full_plot(output_dir: Path, afm_table: dict[str, np.ndarray], settings) -> tuple[Path, Path]:
    t_us = np.asarray(afm_table["t"], dtype=float) * 1e6
    x_nm = np.asarray(afm_table["x1"], dtype=float) * 1e9
    y_nm = np.asarray(afm_table["x3"], dtype=float) * 1e9
    s_nm = np.asarray(afm_table["s"], dtype=float) * 1e9

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_us, x_nm, "b-", linewidth=0.5)
    axes[0].set_ylabel("Tip displacement x [nm]")
    axes[0].set_title("AFM04 DMT-KV Simulation: Full Trajectory")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.0, color="k", linestyle="--", linewidth=0.5)

    axes[1].plot(t_us, y_nm, "r-", linewidth=0.5)
    axes[1].set_ylabel("Sample motion y [nm]")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.0, color="k", linestyle="--", linewidth=0.5)

    axes[2].plot(t_us, s_nm, "g-", linewidth=0.5)
    axes[2].axhline(y=settings.a0 * 1e9, color="r", linestyle="-", linewidth=1.0, label="Hertz threshold (s=a0)")
    axes[2].fill_between(
        t_us,
        s_nm,
        settings.a0 * 1e9,
        where=(s_nm <= settings.a0 * 1e9),
        alpha=0.3,
        color="red",
        label="Hertz-active region",
    )
    axes[2].set_ylabel("Distance s [nm]")
    axes[2].set_xlabel("Time [us]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    png_path = output_dir / "trajectory_full.png"
    pdf_path = output_dir / "trajectory_full.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _save_trajectory_zoomed_plot(output_dir: Path, afm_table: dict[str, np.ndarray], settings) -> tuple[Path, Path]:
    t_us = np.asarray(afm_table["t"], dtype=float) * 1e6
    x_nm = np.asarray(afm_table["x1"], dtype=float) * 1e9
    y_nm = np.asarray(afm_table["x3"], dtype=float) * 1e9
    s_nm = np.asarray(afm_table["s"], dtype=float) * 1e9
    zoom_mask = (t_us >= ZOOM_START_US) & (t_us <= ZOOM_END_US)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_us[zoom_mask], x_nm[zoom_mask], "b-", linewidth=1.0)
    axes[0].set_ylabel("Tip displacement x [nm]")
    axes[0].set_title(f"AFM04 DMT-KV Simulation: Steady-State Zoomed View ({ZOOM_START_US:.0f}-{ZOOM_END_US:.0f} us)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.0, color="k", linestyle="--", linewidth=0.5)

    axes[1].plot(t_us[zoom_mask], y_nm[zoom_mask], "r-", linewidth=1.0)
    axes[1].set_ylabel("Sample motion y [nm]")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.0, color="k", linestyle="--", linewidth=0.5)

    axes[2].plot(t_us[zoom_mask], s_nm[zoom_mask], "g-", linewidth=1.0)
    axes[2].axhline(y=settings.a0 * 1e9, color="r", linestyle="-", linewidth=1.0, label="Hertz threshold (s=a0)")
    axes[2].fill_between(
        t_us[zoom_mask],
        s_nm[zoom_mask],
        settings.a0 * 1e9,
        where=(s_nm[zoom_mask] <= settings.a0 * 1e9),
        alpha=0.3,
        color="red",
        label="Hertz-active region",
    )
    axes[2].set_ylabel("Distance s [nm]")
    axes[2].set_xlabel("Time [us]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    png_path = output_dir / "trajectory_zoomed.png"
    pdf_path = output_dir / "trajectory_zoomed.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _save_f_ts_zoomed_plot(output_dir: Path, afm_table: dict[str, np.ndarray]) -> tuple[Path, Path]:
    times = np.asarray(afm_table["t"], dtype=float)
    contact = np.asarray(afm_table["contact"], dtype=bool)
    x1 = np.asarray(afm_table["x1"], dtype=float)
    fts_n_n = np.asarray(afm_table["fts"], dtype=float) * 1e9
    t_us = times * 1e6

    manifests = stage2_window_manifest(times, contact, x1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharey=True)
    axes = np.atleast_1d(axes)

    segments = [fts_n_n[np.asarray(win.idxs, dtype=int)] for win in manifests]
    y_all = np.concatenate(segments)
    y_pad = 0.08 * max(1.0e-6, float(y_all.max() - y_all.min()))
    y_lo = float(y_all.min() - y_pad)
    y_hi = float(y_all.max() + y_pad)

    role_to_rank = {
        "first_contact": "W0",
        "middle": "W1",
        "max_x1_pp_change": "W2",
        "tail_stable": "W3",
    }

    for ax, win in zip(axes, manifests):
        idxs = np.asarray(win.idxs, dtype=int)
        ax.plot(t_us[idxs], fts_n_n[idxs], color="tab:purple", linewidth=1.0)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.7)
        ax.set_title(
            f"{role_to_rank.get(win.role, win.role)} [{win.role}]  "
            f"{win.t_start * 1e6:.3f}-{win.t_stop * 1e6:.3f} us"
        )
        ax.set_ylabel("F_ts [nN]")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_lo, y_hi)

    axes[-1].set_xlabel("Time [us]")
    fig.suptitle("AFM04 DMT-KV F_ts Time-Domain Signal (training-style W1 / W2 / W3 windows)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    png_path = output_dir / "F_ts_time_domain_zoomed.png"
    pdf_path = output_dir / "F_ts_time_domain_zoomed.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    here = Path(__file__).resolve().parent
    output_dir = here / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = DEFAULT_SETTINGS
    result = generate_afm_dmt_kv_dataset(save_outputs=False)
    afm_table = result["afm_table"]

    csv_path = _save_reference_csv(output_dir, afm_table)
    tf_png, tf_pdf = _save_trajectory_full_plot(output_dir, afm_table, settings)
    tz_png, tz_pdf = _save_trajectory_zoomed_plot(output_dir, afm_table, settings)
    fz_png, fz_pdf = _save_f_ts_zoomed_plot(output_dir, afm_table)

    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {tf_png}")
    print(f"  {tf_pdf}")
    print(f"  {tz_png}")
    print(f"  {tz_pdf}")
    print(f"  {fz_png}")
    print(f"  {fz_pdf}")
    print(f"F_ts min/max [nN]: {float(np.min(afm_table['fts']) * 1e9):.6f}, {float(np.max(afm_table['fts']) * 1e9):.6f}")
    print(f"Contact fraction [%]: {100.0 * float(np.mean(afm_table['contact'])):.2f}")


if __name__ == "__main__":
    main()
