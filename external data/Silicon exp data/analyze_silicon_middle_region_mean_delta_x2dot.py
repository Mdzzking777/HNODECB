from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FILTERED_DIR = SCRIPT_DIR / "GP" / "middle" / "x2 filtered"
FILTERED_NPZ = (
    FILTERED_DIR / "silicon_gp_index109_mid5000_490_510us_x2_regime_aware_sg.npz"
)
OUTPUT_STEM = "silicon_middle_region_mean_delta_x2dot"

REGION_HALF_WIDTH_NS = 64.0
REGION_GAP_NS = 16.0


def _region_mask(
    *,
    time_s: np.ndarray,
    center_s: float,
    side: str,
    half_width_s: float,
    gap_s: float,
) -> np.ndarray:
    if side == "left":
        return (time_s >= center_s - gap_s - half_width_s) & (time_s <= center_s - gap_s)
    if side == "right":
        return (time_s >= center_s + gap_s) & (time_s <= center_s + gap_s + half_width_s)
    raise ValueError(f"unknown side: {side}")


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
    if expected_positive:
        sign_consistency = float(np.mean(values > 0.0))
    else:
        sign_consistency = float(np.mean(values < 0.0))
    return {
        "count": int(values.size),
        "mean_delta_x2dot_m_s2": float(np.mean(values)),
        "median_delta_x2dot_m_s2": float(np.median(values)),
        "std_delta_x2dot_m_s2": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "sign_consistency_fraction": sign_consistency,
    }


def main() -> int:
    with np.load(FILTERED_NPZ, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        x2_raw = np.asarray(data["x2_raw_m_s"], dtype=float)
        x2_tilde = np.asarray(data["x2_tilde_m_s"], dtype=float)
        contact = np.asarray(data["contact_tilde"], dtype=bool)
        transition_idx = np.asarray(data["transition_idx"], dtype=int)

    x2dot_raw = np.gradient(x2_raw, time_s)
    half_width_s = REGION_HALF_WIDTH_NS * 1.0e-9
    gap_s = REGION_GAP_NS * 1.0e-9
    rows: list[dict[str, float | int | str]] = []

    for idx in transition_idx:
        idx = int(idx)
        center_s = float(time_s[idx])
        left = _region_mask(
            time_s=time_s,
            center_s=center_s,
            side="left",
            half_width_s=half_width_s,
            gap_s=gap_s,
        )
        right = _region_mask(
            time_s=time_s,
            center_s=center_s,
            side="right",
            half_width_s=half_width_s,
            gap_s=gap_s,
        )
        if np.count_nonzero(left) < 2 or np.count_nonzero(right) < 2:
            continue
        left_mean = float(np.mean(x2dot_raw[left]))
        right_mean = float(np.mean(x2dot_raw[right]))
        transition_type = "N_to_C" if bool(contact[idx]) else "C_to_N"
        rows.append(
            {
                "transition_index": idx,
                "transition_time_s": center_s,
                "transition_time_us": center_s * 1.0e6,
                "transition_type": transition_type,
                "x2_raw_m_per_s_at_transition": float(x2_raw[idx]),
                "x2_tilde_m_per_s_at_transition": float(x2_tilde[idx]),
                "region_half_width_ns": REGION_HALF_WIDTH_NS,
                "region_gap_ns": REGION_GAP_NS,
                "left_point_count": int(np.count_nonzero(left)),
                "right_point_count": int(np.count_nonzero(right)),
                "mean_x2dot_raw_before_m_s2": left_mean,
                "mean_x2dot_raw_after_m_s2": right_mean,
                "delta_time_mean_x2dot_raw_m_s2": right_mean - left_mean,
            }
        )

    csv_path = FILTERED_DIR / f"{OUTPUT_STEM}.csv"
    json_path = FILTERED_DIR / f"{OUTPUT_STEM}_summary.json"
    txt_path = FILTERED_DIR / f"{OUTPUT_STEM}_summary.txt"
    png_path = FILTERED_DIR / f"{OUTPUT_STEM}.png"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_to_c = np.asarray(
        [
            float(row["delta_time_mean_x2dot_raw_m_s2"])
            for row in rows
            if row["transition_type"] == "N_to_C"
        ],
        dtype=float,
    )
    c_to_n = np.asarray(
        [
            float(row["delta_time_mean_x2dot_raw_m_s2"])
            for row in rows
            if row["transition_type"] == "C_to_N"
        ],
        dtype=float,
    )
    summary = {
        "input_npz": str(FILTERED_NPZ.resolve()),
        "method": (
            "For each middle-slice transition, compute x2dot_raw by differentiating "
            "x2_raw, average x2dot_raw over the before and after regions, then compute "
            "time-forward jump = after_mean - before_mean."
        ),
        "jump_sign_convention": (
            "time-forward after-before: both N_to_C and C_to_N are expected positive."
        ),
        "region_half_width_ns": REGION_HALF_WIDTH_NS,
        "region_gap_ns": REGION_GAP_NS,
        "transition_count_used": len(rows),
        "N_to_C": _summarize(n_to_c, expected_positive=True),
        "C_to_N": _summarize(c_to_n, expected_positive=True),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "Middle-slice time-forward region-mean jump in raw-derived x2dot",
        "==============================================================",
        "",
        f"Input: {FILTERED_NPZ}",
        "x2dot source: numerical derivative of x2_raw, not x2_tilde",
        "Jump definition: after-before in time. Expected signs: N_to_C > 0, C_to_N > 0.",
        f"Region half-width: {REGION_HALF_WIDTH_NS:.1f} ns",
        f"Gap around transition: {REGION_GAP_NS:.1f} ns",
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
            f"sign consistency={100.0 * stats['sign_consistency_fraction']:.1f}%"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.5, 8.2),
        sharex=False,
        gridspec_kw={"hspace": 0.22},
    )
    axes[0].plot(time_us, x2_raw, color="0.6", linewidth=0.8, label=r"$x_2^{raw}$")
    axes[0].plot(time_us, x2_tilde, color="#1f77b4", linewidth=1.0, label=r"$\tilde{x}_2$")
    axes[0].set_ylabel(r"$x_2$ [m s$^{-1}$]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_us, x2dot_raw, color="#d62728", linewidth=0.65)
    axes[1].set_ylabel(r"$\dot{x}_2^{raw}$ [m s$^{-2}$]")

    for row in rows:
        transition_time_us = float(row["transition_time_us"])
        color = "#1f77b4" if row["transition_type"] == "N_to_C" else "#d62728"
        axes[0].axvline(transition_time_us, color=color, linewidth=0.8, alpha=0.35)
        axes[1].axvline(transition_time_us, color=color, linewidth=0.8, alpha=0.35)
        axes[1].hlines(
            float(row["mean_x2dot_raw_before_m_s2"]),
            transition_time_us - REGION_GAP_NS * 1.0e-3 - REGION_HALF_WIDTH_NS * 1.0e-3,
            transition_time_us - REGION_GAP_NS * 1.0e-3,
            colors=color,
            linewidth=2.0,
        )
        axes[1].hlines(
            float(row["mean_x2dot_raw_after_m_s2"]),
            transition_time_us + REGION_GAP_NS * 1.0e-3,
            transition_time_us + REGION_GAP_NS * 1.0e-3 + REGION_HALF_WIDTH_NS * 1.0e-3,
            colors=color,
            linewidth=2.0,
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
