"""Test AFM05 W0 x3 trends using the observed force residual as Fts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "AFM05" / "datasets" / "e0.0" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "x3_w0_residual_forced_grid"

W0_START_S = 0.605488e-3
W0_SPAN_S = 25.152e-6
KS_VALUES = np.geomspace(1.0e-3, 5.0e-1, 10)
CS_VALUES = np.geomspace(5.0e-6, 5.0e-3, 10)


def _load_w0() -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(DATA_DIR / "afm05_F_ts_184_152.npz") as archive:
        times = np.asarray(archive["t"], dtype=float)
        fts = np.asarray(archive["F_ts_184_152"], dtype=float)
    with np.load(DATA_DIR / "pert_df_afm_dmt_kv.npz", allow_pickle=True) as archive:
        state_times = np.asarray(archive["t"], dtype=float)
        x3 = np.asarray(archive["x3"], dtype=float)

    if times.shape != fts.shape or times.shape != state_times.shape:
        raise ValueError("AFM05 residual-force and state time grids are not aligned")
    if not np.allclose(times, state_times, rtol=0.0, atol=1.0e-15):
        raise ValueError("AFM05 residual-force and state time values do not match")

    start = int(np.searchsorted(times, W0_START_S, side="left"))
    stop = int(np.searchsorted(times, W0_START_S + W0_SPAN_S, side="right"))
    if stop - start < 2:
        raise ValueError("AFM05 W0 contains fewer than two full-resolution samples")
    return times[start:stop], fts[start:stop], float(x3[start])


def _integrate_linear_x3(
    times: np.ndarray,
    fts: np.ndarray,
    x3_initial: float,
    ks: float,
    cs: float,
) -> np.ndarray:
    """Integrate x3dot=(-Fts-ks*x3)/cs with midpoint-constant forcing."""

    out = np.empty(times.size, dtype=float)
    out[0] = x3_initial
    dt = np.diff(times)
    force_mid = 0.5 * (fts[:-1] + fts[1:])
    decay = np.exp(-(ks / cs) * dt)
    equilibrium = -force_mid / ks
    for index in range(dt.size):
        out[index + 1] = decay[index] * out[index] + (1.0 - decay[index]) * equilibrium[index]
    return out


def main() -> int:
    times, fts, x3_initial = _load_w0()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | int | bool]] = []
    delta_grid = np.empty((CS_VALUES.size, KS_VALUES.size), dtype=float)
    for cs_index, cs in enumerate(CS_VALUES):
        for ks_index, ks in enumerate(KS_VALUES):
            x3 = _integrate_linear_x3(times, fts, x3_initial, float(ks), float(cs))
            increments = np.diff(x3)
            cumulative_rise = float(np.sum(increments[increments > 0.0]))
            cumulative_fall = float(-np.sum(increments[increments < 0.0]))
            delta_nm = float((x3[-1] - x3[0]) * 1.0e9)
            delta_grid[cs_index, ks_index] = delta_nm
            records.append(
                {
                    "ks_index": ks_index + 1,
                    "cs_index": cs_index + 1,
                    "ks_N_per_m": float(ks),
                    "cs_Ns_per_m": float(cs),
                    "tau_us_diagnostic": float(cs / ks * 1.0e6),
                    "x3_initial_nm": float(x3[0] * 1.0e9),
                    "x3_final_nm": float(x3[-1] * 1.0e9),
                    "delta_x3_nm": delta_nm,
                    "cumulative_rise_nm": cumulative_rise * 1.0e9,
                    "cumulative_fall_nm": cumulative_fall * 1.0e9,
                    "positive_step_fraction": float(np.mean(increments > 0.0)),
                    "x3_min_nm": float(np.min(x3) * 1.0e9),
                    "x3_max_nm": float(np.max(x3) * 1.0e9),
                    "net_upward": bool(delta_nm > 0.0),
                }
            )

    csv_path = OUTPUT_DIR / "afm05_x3_w0_residual_forced_10x10_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=(8.4, 6.5))
    scale = float(np.max(np.abs(delta_grid)))
    image = ax.imshow(
        delta_grid,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
    )
    ax.set_xticks(np.arange(KS_VALUES.size), [f"{value:.2e}" for value in KS_VALUES], rotation=45, ha="right")
    ax.set_yticks(np.arange(CS_VALUES.size), [f"{value:.2e}" for value in CS_VALUES])
    ax.set_xlabel(r"$k_s$ [N m$^{-1}$]")
    ax.set_ylabel(r"$c_s$ [N s m$^{-1}$]")
    ax.set_title(r"W0 net sample displacement, $\Delta x_3$")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$x_3(t_{W0,\mathrm{end}})-x_3(t_{W0,0})$ [nm]")
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "afm05_x3_w0_residual_forced_delta_grid.png"
    fig.savefig(figure_path, dpi=250, facecolor="white")
    plt.close(fig)

    ordered = sorted(records, key=lambda item: float(item["delta_x3_nm"]), reverse=True)
    upward = [item for item in records if bool(item["net_upward"])]
    summary_path = OUTPUT_DIR / "afm05_x3_w0_residual_forced_summary.txt"
    lines = [
        "AFM05 W0 residual-forced x3 side validation",
        "============================================",
        f"W0: {times[0] * 1e6:.6f} to {times[-1] * 1e6:.6f} us",
        f"full-resolution samples: {times.size}",
        f"median dt: {np.median(np.diff(times)) * 1e9:.6f} ns",
        f"x3 initial: {x3_initial * 1e9:.12f} nm",
        f"Fts residual range: [{np.min(fts) * 1e9:.12f}, {np.max(fts) * 1e9:.12f}] nN",
        f"Fts residual mean: {np.mean(fts) * 1e9:.12f} nN",
        "forcing rule: x3dot=(-Fts_residual-ks*x3)/cs",
        f"grid: {KS_VALUES.size} x {CS_VALUES.size} = {len(records)} mechanical combinations",
        f"net-upward combinations: {len(upward)}/{len(records)}",
        "",
        "Top 10 by net W0 rise:",
    ]
    for rank, item in enumerate(ordered[:10], start=1):
        lines.append(
            f"{rank:2d}. ks={float(item['ks_N_per_m']):.9e} N/m, "
            f"cs={float(item['cs_Ns_per_m']):.9e} N*s/m, "
            f"delta_x3={float(item['delta_x3_nm']):+.9f} nm, "
            f"rise={float(item['cumulative_rise_nm']):.9f} nm, "
            f"fall={float(item['cumulative_fall_nm']):.9f} nm"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(summary_path.read_text(encoding="utf-8"))
    print(f"Saved: {csv_path}")
    print(f"Saved: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
