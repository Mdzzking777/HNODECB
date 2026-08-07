from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "acceleration jump judge"
FILTERED_NPZ = (
    SCRIPT_DIR
    / "GP"
    / "middle"
    / "x2 filtered"
    / "silicon_gp_index109_mid5000_490_510us_x2_regime_aware_sg.npz"
)
OUTPUT_STEM = "middle_points_only_delta_x2dot"


def _summarize(values: np.ndarray, expected_positive: bool) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean_delta_x2dot_m_s2": float("nan"),
            "median_delta_x2dot_m_s2": float("nan"),
            "std_delta_x2dot_m_s2": float("nan"),
            "sign_consistency_fraction": float("nan"),
        }
    positive_fraction = float(np.mean(values > 0.0))
    return {
        "count": int(values.size),
        "mean_delta_x2dot_m_s2": float(np.mean(values)),
        "median_delta_x2dot_m_s2": float(np.median(values)),
        "std_delta_x2dot_m_s2": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "positive_fraction": positive_fraction,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(FILTERED_NPZ, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        x2_raw = np.asarray(data["x2_raw_m_s"], dtype=float)
        contact = np.asarray(data["contact_tilde"], dtype=bool)
        transition_idx = np.asarray(data["transition_idx"], dtype=int)
    x2dot_raw = np.gradient(x2_raw, time_s, edge_order=2)

    rows: list[dict[str, float | int | str]] = []
    for idx in transition_idx:
        idx = int(idx)
        if idx <= 0 or idx >= x2dot_raw.size:
            continue
        left_idx = idx - 1
        right_idx = idx
        transition_type = "N_to_C" if bool(contact[idx]) else "C_to_N"
        rows.append(
            {
                "transition_index": idx,
                "transition_time_s": float(time_s[idx]),
                "transition_time_us": float(time_s[idx] * 1.0e6),
                "transition_type": transition_type,
                "left_index": left_idx,
                "right_index": right_idx,
                "x2_raw_m_per_s_at_transition": float(x2_raw[idx]),
                "x2dot_raw_before_m_s2": float(x2dot_raw[left_idx]),
                "x2dot_raw_after_m_s2": float(x2dot_raw[right_idx]),
                "delta_time_x2dot_raw_m_s2": float(x2dot_raw[right_idx] - x2dot_raw[left_idx]),
            }
        )

    csv_path = OUTPUT_DIR / f"{OUTPUT_STEM}.csv"
    json_path = OUTPUT_DIR / f"{OUTPUT_STEM}_summary.json"
    txt_path = OUTPUT_DIR / f"{OUTPUT_STEM}_summary.txt"
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

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
        "input_npz": str(FILTERED_NPZ.resolve()),
        "method": (
            "Point-only transition jump: x2dot_raw at first point after the "
            "contact-state switch minus x2dot_raw at the immediately previous point."
        ),
        "x2dot_source": "np.gradient(x2_raw, time_s, edge_order=2)",
        "jump_definition": "time-forward after-before",
        "transition_count_used": len(rows),
        "N_to_C": _summarize(n_to_c, expected_positive=True),
        "C_to_N": _summarize(c_to_n, expected_positive=True),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "Middle-slice point-only delta x2dot",
        "==================================",
        "",
        f"Input: {FILTERED_NPZ}",
        "x2dot source: numerical derivative of x2_raw",
        "Definition: time-forward after-before, x2dot_raw[right transition point] - x2dot_raw[left previous point]",
        f"Transitions used: {len(rows)}",
        "",
    ]
    for name in ("N_to_C", "C_to_N"):
        stats = summary[name]
        lines.append(
            f"{name}: n={stats['count']}, "
            f"mean={stats['mean_delta_x2dot_m_s2']:.6e} m s^-2, "
            f"median={stats['median_delta_x2dot_m_s2']:.6e} m s^-2, "
            f"std={stats['std_delta_x2dot_m_s2']:.6e} m s^-2, "
            f"positive fraction={100.0 * stats['positive_fraction']:.1f}%"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    time_us = time_s * 1.0e6
    colors = {"N_to_C": "#1f77b4", "C_to_N": "#d62728"}
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.5, 8.2),
        sharex=False,
        gridspec_kw={"hspace": 0.22},
    )
    axes[0].plot(time_us, x2_raw, color="0.35", linewidth=0.8, label=r"$x_2^{raw}$")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel(r"$x_2^{raw}$ [m s$^{-1}$]")

    axes[1].plot(time_us, x2dot_raw, color="#d62728", linewidth=0.65)
    axes[1].set_ylabel(r"$\dot{x}_2^{raw}$ [m s$^{-2}$]")

    for row in rows:
        transition_time_us = float(row["transition_time_us"])
        color = colors[str(row["transition_type"])]
        for ax in axes[:2]:
            ax.axvline(transition_time_us, color=color, linewidth=0.9, alpha=0.4)
        axes[1].scatter(
            [time_us[int(row["left_index"])], time_us[int(row["right_index"])]],
            [float(row["x2dot_raw_before_m_s2"]), float(row["x2dot_raw_after_m_s2"])],
            color=color,
            s=22,
            zorder=5,
        )

    labels = ["N_to_C", "C_to_N"]
    means = [
        float(summary["N_to_C"]["mean_delta_x2dot_m_s2"]),
        float(summary["C_to_N"]["mean_delta_x2dot_m_s2"]),
    ]
    medians = [
        float(summary["N_to_C"]["median_delta_x2dot_m_s2"]),
        float(summary["C_to_N"]["median_delta_x2dot_m_s2"]),
    ]
    x = np.arange(len(labels), dtype=float)
    axes[2].bar(x - 0.18, means, width=0.36, color=["#1f77b4", "#d62728"], alpha=0.85, label="mean")
    axes[2].bar(x + 0.18, medians, width=0.36, color=["#1f77b4", "#d62728"], alpha=0.45, label="median")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel(r"$J_{\dot{x}_2}^{time}$ [m s$^{-2}$]")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)
    axes[1].set_xlabel(r"Time [$\mu$s]")
    axes[2].set_xlabel("transition type")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {png_path}")
    print(lines[-2])
    print(lines[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
