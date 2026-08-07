from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "acceleration jump judge"
LOCAL_MAT = SCRIPT_DIR / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
RAW_MAT = (
    Path(r"E:\External data SI_SI hard sample raw")
    / "OneDrive_2_2026-7-26"
    / "dataAFM_Si"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)

INDEX = 109
DT_S = 1.0 / 250.0e6
START_US = 400.0
END_US = 800.0

Z_STATIC_M = 100.0e-9
A0_M = 0.165e-9

X1_SG_WINDOW = 201
X1_SG_POLYORDER = 3

FIT_HALF_WIDTHS_NS = (64.0,)
FIT_GAP_NS = 16.0


def _load_index_signals() -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    errors: list[str] = []
    for mat_path in (LOCAL_MAT, RAW_MAT):
        if not mat_path.is_file():
            errors.append(f"{mat_path} does not exist")
            continue
        try:
            data = sio.loadmat(
                str(mat_path),
                variable_names=["displacement_fwd_sweep", "velocity_fwd_sweep"],
                squeeze_me=True,
                struct_as_record=False,
            )
            x1_all = np.asarray(data["displacement_fwd_sweep"][INDEX], dtype=float).ravel()
            x2_all = np.asarray(data["velocity_fwd_sweep"][INDEX], dtype=float).ravel()
            if x1_all.shape != x2_all.shape:
                raise ValueError(f"x1/x2 shapes differ: {x1_all.shape} vs {x2_all.shape}")
            time_all = np.arange(x1_all.size, dtype=float) * DT_S
            return time_all, x1_all, x2_all, mat_path
        except Exception as exc:
            errors.append(f"{mat_path}: {exc}")
    raise RuntimeError("Could not read MAT file:\n" + "\n".join(errors))


def _odd_window_at_most(length: int, preferred: int, polyorder: int) -> int:
    window = min(length, preferred)
    if window % 2 == 0:
        window -= 1
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    if window < minimum:
        raise ValueError("segment is too short for Savitzky-Golay smoothing")
    return window


def _fit_transition_x2dot_jumps(
    *,
    time_s: np.ndarray,
    x2_raw: np.ndarray,
    x2dot_raw: np.ndarray,
    transition_idx: np.ndarray,
    contact: np.ndarray,
    fit_half_width_s: float,
    fit_gap_s: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for idx in transition_idx:
        idx = int(idx)
        ts = float(time_s[idx])
        left = (time_s >= ts - fit_gap_s - fit_half_width_s) & (time_s <= ts - fit_gap_s)
        right = (time_s >= ts + fit_gap_s) & (time_s <= ts + fit_gap_s + fit_half_width_s)
        if np.count_nonzero(left) < 3 or np.count_nonzero(right) < 3:
            continue
        x2dot_before = float(np.mean(x2dot_raw[left]))
        x2dot_after = float(np.mean(x2dot_raw[right]))
        transition_type = "N_to_C" if bool(contact[idx]) else "C_to_N"
        rows.append(
            {
                "transition_index": idx,
                "transition_time_s": ts,
                "transition_time_us": ts * 1.0e6,
                "transition_type": transition_type,
                "x2_raw_m_per_s": float(x2_raw[idx]),
                "fit_half_width_ns": fit_half_width_s * 1.0e9,
                "fit_gap_ns": fit_gap_s * 1.0e9,
                "left_point_count": int(np.count_nonzero(left)),
                "right_point_count": int(np.count_nonzero(right)),
                "mean_x2dot_raw_before_m_s2": x2dot_before,
                "mean_x2dot_raw_after_m_s2": x2dot_after,
                "delta_time_mean_x2dot_raw_m_s2": x2dot_after - x2dot_before,
            }
        )
    return rows


def _summarize(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for transition_type in ("N_to_C", "C_to_N"):
        values = np.asarray(
            [
                float(row["delta_time_mean_x2dot_raw_m_s2"])
                for row in rows
                if row["transition_type"] == transition_type
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        positive_fraction = float(np.mean(values > 0.0))
        summary[transition_type] = {
            "count": int(values.size),
            "mean_delta_time_x2dot_raw_m_s2": float(np.mean(values)),
            "median_delta_time_x2dot_raw_m_s2": float(np.median(values)),
            "std_delta_time_x2dot_raw_m_s2": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "positive_fraction": positive_fraction,
        }
    return summary


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(
    *,
    time_s: np.ndarray,
    x1_raw: np.ndarray,
    x1_tilde: np.ndarray,
    x2_raw: np.ndarray,
    x2dot_raw: np.ndarray,
    transition_idx: np.ndarray,
    rows_by_window: dict[float, list[dict[str, float | int | str]]],
    output_path: Path,
) -> None:
    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(12, 12.6),
        sharex=False,
        gridspec_kw={"hspace": 0.22},
    )

    axes[0].plot(time_us, 1.0e9 * x1_raw, color="0.65", linewidth=0.7, label="raw")
    axes[0].plot(time_us, 1.0e9 * x1_tilde, color="#1f77b4", linewidth=1.0, label=r"$\tilde{x}_1$")
    axes[0].set_ylabel(r"$x_1$ [nm]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_us, x2_raw, color="0.35", linewidth=0.7)
    axes[1].scatter(
        time_us[transition_idx],
        x2_raw[transition_idx],
        s=18,
        color="black",
        zorder=5,
    )
    axes[1].set_ylabel(r"$x_2^{raw}$ [m s$^{-1}$]")

    axes[2].plot(time_us, x2dot_raw, color="#d62728", linewidth=0.55)
    axes[2].set_ylabel(r"$\dot{x}_2^{raw}$ [m s$^{-2}$]")

    for ax in axes[:3]:
        for idx in transition_idx:
            ax.axvline(time_us[int(idx)], color="black", linewidth=0.45, alpha=0.18)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)

    colors = {"N_to_C": "#1f77b4", "C_to_N": "#d62728"}
    markers = {"N_to_C": "o", "C_to_N": "s"}
    for half_width_ns, rows in rows_by_window.items():
        for transition_type in ("N_to_C", "C_to_N"):
            selected = [row for row in rows if row["transition_type"] == transition_type]
            if not selected:
                continue
            axes[3].scatter(
                [float(row["transition_time_us"]) for row in selected],
                [float(row["delta_time_mean_x2dot_raw_m_s2"]) for row in selected],
                s=18,
                marker=markers[transition_type],
                color=colors[transition_type],
                alpha=0.72,
                label=f"{transition_type}, {half_width_ns:.0f} ns",
            )

    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_ylabel(r"$J_{\dot{x}_2}^{time}$ [m s$^{-2}$]")
    axes[3].set_xlabel(r"Transition time [$\mu$s]")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right", ncols=2, fontsize=8)

    window_labels = []
    means_nc = []
    means_cn = []
    for half_width_ns, rows in rows_by_window.items():
        summary = _summarize(rows)
        window_labels.append(f"{half_width_ns:.0f} ns")
        means_nc.append(summary.get("N_to_C", {}).get("mean_delta_time_x2dot_raw_m_s2", np.nan))
        means_cn.append(summary.get("C_to_N", {}).get("mean_delta_time_x2dot_raw_m_s2", np.nan))
    xpos = np.arange(len(window_labels), dtype=float)
    axes[4].bar(xpos - 0.18, means_nc, width=0.36, color=colors["N_to_C"], label="N_to_C")
    axes[4].bar(xpos + 0.18, means_cn, width=0.36, color=colors["C_to_N"], label="C_to_N")
    axes[4].axhline(0.0, color="black", linewidth=0.8)
    axes[4].set_xticks(xpos)
    axes[4].set_xticklabels(window_labels)
    axes[4].set_xlabel("before/after region half-width")
    axes[4].set_ylabel(r"mean $J_{\dot{x}_2}^{time}$ [m s$^{-2}$]")
    axes[4].grid(True, axis="y", alpha=0.25)
    axes[4].legend(loc="upper right")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    time_all, x1_all, x2_all, mat_path = _load_index_signals()
    mask = (time_all >= START_US * 1.0e-6) & (time_all <= END_US * 1.0e-6)
    time_s = time_all[mask]
    x1_raw = x1_all[mask]
    x2_raw = x2_all[mask]
    x2dot_raw = np.gradient(x2_raw, time_s, edge_order=2)
    source_idx = np.flatnonzero(mask)

    window = _odd_window_at_most(time_s.size, X1_SG_WINDOW, X1_SG_POLYORDER)
    x1_tilde = savgol_filter(
        x1_raw,
        window_length=window,
        polyorder=X1_SG_POLYORDER,
        mode="interp",
    )
    contact = (Z_STATIC_M + x1_tilde) <= A0_M
    transition_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1

    all_rows: list[dict[str, float | int | str]] = []
    rows_by_window: dict[float, list[dict[str, float | int | str]]] = {}
    fit_gap_s = FIT_GAP_NS * 1.0e-9
    summary_by_window: dict[str, dict[str, dict[str, float | int]]] = {}
    for half_width_ns in FIT_HALF_WIDTHS_NS:
        rows = _fit_transition_x2dot_jumps(
            time_s=time_s,
            x2_raw=x2_raw,
            x2dot_raw=x2dot_raw,
            transition_idx=transition_idx,
            contact=contact,
            fit_half_width_s=half_width_ns * 1.0e-9,
            fit_gap_s=fit_gap_s,
        )
        rows_by_window[float(half_width_ns)] = rows
        all_rows.extend(rows)
        summary_by_window[f"{half_width_ns:.0f}_ns"] = _summarize(rows)

    csv_path = OUTPUT_DIR / "index109_400_800us_transition_slope_jumps.csv"
    json_path = OUTPUT_DIR / "index109_400_800us_transition_slope_jumps_summary.json"
    txt_path = OUTPUT_DIR / "index109_400_800us_transition_slope_jumps_summary.txt"
    npz_path = OUTPUT_DIR / "index109_400_800us_transition_slope_jumps_data.npz"
    png_path = OUTPUT_DIR / "index109_400_800us_transition_slope_jumps.png"

    _write_csv(csv_path, all_rows)
    metadata = {
        "mat_path": str(mat_path),
        "index": INDEX,
        "time_range_us": [START_US, END_US],
        "dt_s": DT_S,
        "source_start_index": int(source_idx[0]),
        "source_end_index_inclusive": int(source_idx[-1]),
        "sample_count": int(time_s.size),
        "x1_contact_judge": {
            "method": "Savitzky-Golay smoothing of x1_raw only for contact judgment",
            "window_points": int(window),
            "window_us": float(window * DT_S * 1.0e6),
            "polyorder": X1_SG_POLYORDER,
            "contact_rule": "contact if z_static + x1_tilde <= a0",
            "z_static_m": Z_STATIC_M,
            "a0_m": A0_M,
        },
        "fit_gap_ns": FIT_GAP_NS,
        "fit_half_widths_ns": list(FIT_HALF_WIDTHS_NS),
        "x2dot_source": "np.gradient(x2_raw, time_s, edge_order=2)",
        "jump_definition": "time-forward after-before: mean_x2dot_raw_after - mean_x2dot_raw_before",
        "transition_count": int(transition_idx.size),
        "transition_count_N_to_C": int(np.count_nonzero(contact[transition_idx])),
        "transition_count_C_to_N": int(np.count_nonzero(~contact[transition_idx])),
        "summary_by_fit_half_width": summary_by_window,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "Acceleration jump judge for external silicon index109",
        "=====================================================",
        "",
        f"Time range: {START_US:.1f}-{END_US:.1f} us",
        f"Samples: {time_s.size}",
        f"Transition points: {transition_idx.size}",
        f"x1 contact judge: SG window={window} points ({window * DT_S * 1e6:.3f} us), polyorder={X1_SG_POLYORDER}",
        f"Contact rule: z_static + x1_tilde <= a0; z_static={Z_STATIC_M:.6e} m, a0={A0_M:.6e} m",
        "x2dot source: numerical derivative of x2_raw",
        "Jump definition: time-forward after-before",
        f"Fit gap around transition: {FIT_GAP_NS:.1f} ns",
        "",
    ]
    for label, summary in summary_by_window.items():
        lines.append(f"Fit half-width: {label.replace('_', ' ')}")
        for transition_type in ("N_to_C", "C_to_N"):
            stats = summary.get(transition_type)
            if not stats:
                continue
            lines.append(
                "  "
                f"{transition_type}: n={stats['count']}, "
                f"mean={stats['mean_delta_time_x2dot_raw_m_s2']:.6e} m s^-2, "
                f"median={stats['median_delta_time_x2dot_raw_m_s2']:.6e} m s^-2, "
                f"std={stats['std_delta_time_x2dot_raw_m_s2']:.6e} m s^-2, "
                f"positive fraction={100.0 * stats['positive_fraction']:.1f}%"
            )
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    np.savez_compressed(
        npz_path,
        time_s=time_s,
        source_idx=source_idx,
        x1_raw_m=x1_raw,
        x1_tilde_m=x1_tilde,
        x2_raw_m_s=x2_raw,
        x2dot_raw_m_s2=x2dot_raw,
        contact=contact,
        transition_idx=transition_idx,
    )

    _plot_results(
        time_s=time_s,
        x1_raw=x1_raw,
        x1_tilde=x1_tilde,
        x2_raw=x2_raw,
        x2dot_raw=x2dot_raw,
        transition_idx=transition_idx,
        rows_by_window=rows_by_window,
        output_path=png_path,
    )

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {png_path}")
    print(f"Transition points: {transition_idx.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
