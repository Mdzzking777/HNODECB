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
OUTPUT_STEM = "index109_400_800us_points_only_delta_x2dot"


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


def _summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean_delta_time_x2dot_raw_m_s2": float("nan"),
            "median_delta_time_x2dot_raw_m_s2": float("nan"),
            "std_delta_time_x2dot_raw_m_s2": float("nan"),
            "positive_fraction": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean_delta_time_x2dot_raw_m_s2": float(np.mean(values)),
        "median_delta_time_x2dot_raw_m_s2": float(np.median(values)),
        "std_delta_time_x2dot_raw_m_s2": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    time_all, x1_all, x2_all, mat_path = _load_index_signals()
    mask = (time_all >= START_US * 1.0e-6) & (time_all <= END_US * 1.0e-6)
    time_s = time_all[mask]
    x1_raw = x1_all[mask]
    x2_raw = x2_all[mask]
    source_idx = np.flatnonzero(mask)

    x2dot_raw = np.gradient(x2_raw, time_s, edge_order=2)
    window = _odd_window_at_most(time_s.size, X1_SG_WINDOW, X1_SG_POLYORDER)
    x1_tilde = savgol_filter(
        x1_raw,
        window_length=window,
        polyorder=X1_SG_POLYORDER,
        mode="interp",
    )
    contact = (Z_STATIC_M + x1_tilde) <= A0_M
    transition_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1

    rows: list[dict[str, float | int | str]] = []
    for idx_raw in transition_idx:
        idx = int(idx_raw)
        if idx <= 0 or idx >= x2dot_raw.size:
            continue
        before_idx = idx - 1
        after_idx = idx
        transition_type = "N_to_C" if bool(contact[idx]) else "C_to_N"
        before = float(x2dot_raw[before_idx])
        after = float(x2dot_raw[after_idx])
        rows.append(
            {
                "transition_index": idx,
                "transition_time_s": float(time_s[idx]),
                "transition_time_us": float(time_s[idx] * 1.0e6),
                "transition_type": transition_type,
                "before_index": before_idx,
                "after_index": after_idx,
                "x2_raw_m_per_s_at_transition": float(x2_raw[idx]),
                "x2dot_raw_before_m_s2": before,
                "x2dot_raw_after_m_s2": after,
                "delta_time_x2dot_raw_m_s2": after - before,
            }
        )

    n_to_c = np.asarray(
        [
            float(row["delta_time_x2dot_raw_m_s2"])
            for row in rows
            if row["transition_type"] == "N_to_C"
        ],
        dtype=float,
    )
    c_to_n = np.asarray(
        [
            float(row["delta_time_x2dot_raw_m_s2"])
            for row in rows
            if row["transition_type"] == "C_to_N"
        ],
        dtype=float,
    )
    summary = {
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
        "x2dot_source": "np.gradient(x2_raw, time_s, edge_order=2)",
        "jump_definition": "point-only time-forward after-before: x2dot_raw[transition] - x2dot_raw[transition-1]",
        "transition_count": int(transition_idx.size),
        "transition_count_N_to_C": int(np.count_nonzero(contact[transition_idx])),
        "transition_count_C_to_N": int(np.count_nonzero(~contact[transition_idx])),
        "N_to_C": _summarize(n_to_c),
        "C_to_N": _summarize(c_to_n),
    }

    csv_path = OUTPUT_DIR / f"{OUTPUT_STEM}.csv"
    json_path = OUTPUT_DIR / f"{OUTPUT_STEM}_summary.json"
    txt_path = OUTPUT_DIR / f"{OUTPUT_STEM}_summary.txt"
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    npz_path = OUTPUT_DIR / f"{OUTPUT_STEM}_data.npz"

    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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

    lines = [
        "Point-only acceleration jump judge for external silicon index109",
        "===============================================================",
        "",
        f"Time range: {START_US:.1f}-{END_US:.1f} us",
        f"Samples: {time_s.size}",
        f"Transition points: {transition_idx.size}",
        f"x1 contact judge: SG window={window} points ({window * DT_S * 1e6:.3f} us), polyorder={X1_SG_POLYORDER}",
        "x2dot source: numerical derivative of x2_raw",
        "Jump definition: point-only time-forward after-before",
        "",
    ]
    for name in ("N_to_C", "C_to_N"):
        stats = summary[name]
        lines.append(
            f"{name}: n={stats['count']}, "
            f"mean={stats['mean_delta_time_x2dot_raw_m_s2']:.6e} m s^-2, "
            f"median={stats['median_delta_time_x2dot_raw_m_s2']:.6e} m s^-2, "
            f"std={stats['std_delta_time_x2dot_raw_m_s2']:.6e} m s^-2, "
            f"positive fraction={100.0 * stats['positive_fraction']:.1f}%"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    time_us = time_s * 1.0e6
    colors = {"N_to_C": "#1f77b4", "C_to_N": "#d62728"}
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10.5),
        sharex=False,
        gridspec_kw={"hspace": 0.22},
    )
    axes[0].plot(time_us, 1.0e9 * x1_raw, color="0.65", linewidth=0.7, label="raw")
    axes[0].plot(time_us, 1.0e9 * x1_tilde, color="#1f77b4", linewidth=1.0, label=r"$\tilde{x}_1$")
    axes[0].set_ylabel(r"$x_1$ [nm]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_us, x2_raw, color="0.35", linewidth=0.65)
    axes[1].scatter(time_us[transition_idx], x2_raw[transition_idx], s=12, color="black", zorder=5)
    axes[1].set_ylabel(r"$x_2^{raw}$ [m s$^{-1}$]")

    axes[2].plot(time_us, x2dot_raw, color="#d62728", linewidth=0.45)
    axes[2].set_ylabel(r"$\dot{x}_2^{raw}$ [m s$^{-2}$]")
    for row in rows:
        color = colors[str(row["transition_type"])]
        transition_time_us = float(row["transition_time_us"])
        for ax in axes[:3]:
            ax.axvline(transition_time_us, color=color, linewidth=0.45, alpha=0.2)
        axes[2].scatter(
            [time_us[int(row["before_index"])], time_us[int(row["after_index"])]],
            [float(row["x2dot_raw_before_m_s2"]), float(row["x2dot_raw_after_m_s2"])],
            color=color,
            s=10,
            zorder=5,
        )

    for transition_type in ("N_to_C", "C_to_N"):
        selected = [row for row in rows if row["transition_type"] == transition_type]
        axes[3].scatter(
            [float(row["transition_time_us"]) for row in selected],
            [float(row["delta_time_x2dot_raw_m_s2"]) for row in selected],
            s=13,
            color=colors[transition_type],
            label=transition_type,
        )
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_ylabel(r"$J_{\dot{x}_2}^{time}$ [m s$^{-2}$]")
    axes[3].set_xlabel(r"Transition time [$\mu$s]")
    axes[3].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {png_path}")
    print(lines[-2])
    print(lines[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
