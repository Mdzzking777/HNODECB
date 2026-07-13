"""Post-analysis for AFM05 step3 identifiability results.

This script reads an already-computed AFM05 step3 result JSON and performs the
author-style null-space projection analysis:

  - eigenvalue spectrum with AFM05 null threshold
  - threshold sweep counts
  - projection of mech.raw_ks / mech.raw_cs unit directions onto the null space
  - group decomposition of the projection energy

It is intentionally post-process only. It does not rerun stage2light rollouts
and does not mutate the original step3 result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
RESULT_SCHEMA_PREFIX = "afm05_step3_identifiability"

GROUP_ORDER = (
    "kan_numeric",
    "gain",
    "soft_mask",
    "mech.raw_ks",
    "mech.raw_cs",
    "mech.other",
)
GROUP_COLORS = {
    "kan_numeric": "#4C78A8",
    "gain": "#F58518",
    "soft_mask": "#54A24B",
    "mech.raw_ks": "#B279A2",
    "mech.raw_cs": "#E45756",
    "mech.other": "#9D755D",
}
GROUP_LABELS = {
    "kan_numeric": "KAN",
    "gain": "Gain",
    "soft_mask": "Soft Mask",
    "mech.raw_ks": r"$k_s$",
    "mech.raw_cs": r"$c_s$",
    "mech.other": "Other",
}
TARGET_LABELS = ("mech.raw_ks[0]", "mech.raw_cs[0]")
SPECTRUM_MARKER_THRESHOLDS = tuple(float(f"1e{exp}") for exp in range(-12, -3))
PROJECTION_GROUP_TAUS = (1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)
PROJECTION_GROUP_BAR_X = np.asarray([-0.48, 0.48], dtype=float)
PROJECTION_GROUP_BAR_WIDTH = 0.36
PROJECTION_GROUP_PERCENT_MIN = 1.0
PROJECTION_GROUP_TITLE_FONTSIZE = 40
PROJECTION_GROUP_AXIS_LABEL_FONTSIZE = 40
PROJECTION_GROUP_TICK_FONTSIZE = 32
PROJECTION_GROUP_PERCENT_FONTSIZE = 26
PROJECTION_GROUP_TOTAL_FONTSIZE = 32
PROJECTION_GROUP_LEGEND_FONTSIZE = 40


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_text()}] {message}", flush=True)


def format_projection_percent(value: float) -> str:
    if value >= 99.95:
        return "100%"
    if value >= 10.0:
        return f"{value:.0f}%"
    if value >= 1.0:
        return f"{value:.1f}%"
    return f"{value:.2f}%"


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for raw in str(text).replace(";", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("threshold list is empty")
    return values


def make_sweep_thresholds(args: argparse.Namespace) -> list[float]:
    if args.threshold_points and int(args.threshold_points) > 0:
        if args.threshold_log_min <= 0.0 or args.threshold_log_max <= 0.0:
            raise ValueError("threshold logspace bounds must be positive")
        if args.threshold_log_min >= args.threshold_log_max:
            raise ValueError("threshold logspace min must be smaller than max")
        if int(args.threshold_points) < 2:
            raise ValueError("threshold_points must be >= 2 when logspace sweep is enabled")
        return [
            float(x)
            for x in np.logspace(
                math.log10(float(args.threshold_log_min)),
                math.log10(float(args.threshold_log_max)),
                int(args.threshold_points),
            )
        ]
    return parse_float_list(args.thresholds)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return repo_rel(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def group_for_label(label: str) -> str:
    if label.startswith("mech.raw_ks"):
        return "mech.raw_ks"
    if label.startswith("mech.raw_cs"):
        return "mech.raw_cs"
    if label.startswith("mech."):
        return "mech.other"
    if "soft_mask" in label:
        return "soft_mask"
    if label.startswith("model.log_gnn"):
        return "gain"
    return "kan_numeric"


def load_step3_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    schema = str(result.get("schema_version", ""))
    if not schema.startswith(RESULT_SCHEMA_PREFIX):
        raise ValueError(f"Unexpected result schema in {path}: {schema!r}")
    return result


def prepare_eigensystem(result: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels = [str(x) for x in result["parameter_coordinates"]]
    eigenvalues = np.asarray(result["eigenvalues"], dtype=float).reshape(-1)
    eigenvectors = np.asarray(result["eigenvectors_columns"], dtype=float)

    n = len(labels)
    if eigenvalues.shape[0] != n:
        raise ValueError(f"eigenvalue count {eigenvalues.shape[0]} != parameter count {n}")
    if eigenvectors.shape != (n, n):
        if eigenvectors.T.shape == (n, n):
            eigenvectors = eigenvectors.T
        else:
            raise ValueError(f"eigenvector matrix shape {eigenvectors.shape} incompatible with n={n}")

    # np.linalg.eigh returns eigenvectors as columns. The compute script stores
    # that object directly, so eigenvectors[:, i] is v_i.
    order = np.argsort(eigenvalues)
    return labels, eigenvalues[order], eigenvectors[:, order]


def threshold_label(value: float) -> str:
    return f"{value:.6e}".replace("+", "").replace(".", "p")


def unique_thresholds(primary: float, sweep: list[float]) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = [("primary", float(primary))]
    for th in sweep:
        label = f"sweep_{threshold_label(th)}"
        if not any(math.isclose(th, existing, rel_tol=0.0, abs_tol=max(abs(th), 1.0) * 1.0e-15) for _, existing in items):
            items.append((label, float(th)))
        else:
            items.append((label, float(th)))
    return items


def projection_analysis_for_threshold(
    *,
    labels: list[str],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    threshold_name: str,
    threshold_value: float,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mask = np.abs(eigenvalues) <= threshold_value
    null_count = int(np.count_nonzero(mask))
    null_vectors = eigenvectors[:, mask]
    max_abs_eval = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else float("nan")

    threshold_summary = {
        "threshold_name": threshold_name,
        "threshold_value": float(threshold_value),
        "null_count": null_count,
        "parameter_count": int(eigenvalues.size),
        "null_fraction": float(null_count / eigenvalues.size) if eigenvalues.size else float("nan"),
        "max_abs_eigenvalue": max_abs_eval,
        "relative_threshold": float(threshold_value / max_abs_eval) if max_abs_eval > 0.0 else float("nan"),
    }

    rows: list[dict[str, Any]] = []
    for target in TARGET_LABELS:
        if target not in labels:
            continue
        target_idx = labels.index(target)
        if null_vectors.size == 0:
            projection = np.zeros(len(labels), dtype=float)
        else:
            # Projection of the target unit vector e_i into span(V_null):
            # P e_i = V_null @ (V_null.T @ e_i) = V_null @ V_null[i, :].
            projection = null_vectors @ null_vectors[target_idx, :]

        projection_energy = float(np.dot(projection, projection))
        projection_norm = float(math.sqrt(max(projection_energy, 0.0)))
        residual_norm = float(math.sqrt(max(1.0 - min(projection_energy, 1.0), 0.0)))
        self_component = float(projection[target_idx])
        self_energy = float(self_component * self_component)

        group_energy = {name: 0.0 for name in GROUP_ORDER}
        for idx, label in enumerate(labels):
            group = group_for_label(label)
            group_energy[group] = group_energy.get(group, 0.0) + float(projection[idx] * projection[idx])

        if projection_energy > 0.0:
            group_fraction_within_projection = {
                group: float(value / projection_energy) for group, value in group_energy.items()
            }
        else:
            group_fraction_within_projection = {group: float("nan") for group in group_energy}

        other_energy = float(max(projection_energy - self_energy, 0.0))
        other_fraction = float(other_energy / projection_energy) if projection_energy > 0.0 else float("nan")

        abs_order = np.argsort(-np.abs(projection))
        top_components = []
        for rank, param_idx in enumerate(abs_order[: max(1, int(top_k))], start=1):
            top_components.append(
                {
                    "rank": rank,
                    "parameter": labels[int(param_idx)],
                    "component": float(projection[int(param_idx)]),
                    "energy": float(projection[int(param_idx)] ** 2),
                    "group": group_for_label(labels[int(param_idx)]),
                }
            )

        row: dict[str, Any] = {
            "threshold_name": threshold_name,
            "threshold_value": float(threshold_value),
            "null_count": null_count,
            "target_parameter": target,
            "target_index": target_idx,
            "projection_norm": projection_norm,
            "projection_energy": projection_energy,
            "residual_norm": residual_norm,
            "self_component": self_component,
            "self_energy": self_energy,
            "compensation_energy_excluding_self": other_energy,
            "compensation_fraction_within_projection": other_fraction,
            "top_components": top_components,
            "projection_vector": projection,
        }
        for group in GROUP_ORDER:
            row[f"{group}_energy"] = float(group_energy.get(group, 0.0))
            row[f"{group}_fraction_within_projection"] = float(group_fraction_within_projection.get(group, float("nan")))
        rows.append(row)

    return threshold_summary, rows


def csv_rows_from_projection(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        out = {k: v for k, v in row.items() if k not in {"top_components", "projection_vector"}}
        for rank in range(1, top_k + 1):
            prefix = f"top{rank}"
            if rank <= len(row["top_components"]):
                comp = row["top_components"][rank - 1]
                out[f"{prefix}_parameter"] = comp["parameter"]
                out[f"{prefix}_component"] = comp["component"]
                out[f"{prefix}_energy"] = comp["energy"]
                out[f"{prefix}_group"] = comp["group"]
            else:
                out[f"{prefix}_parameter"] = ""
                out[f"{prefix}_component"] = ""
                out[f"{prefix}_energy"] = ""
                out[f"{prefix}_group"] = ""
        flat_rows.append(out)
    return flat_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text_summary(
    path: Path,
    *,
    source_result: Path,
    rank_label: str,
    primary_threshold: float,
    threshold_summaries: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("AFM05 step3 post-analysis")
    lines.append("==========================")
    lines.append("")
    lines.append(f"source_result: {repo_rel(source_result)}")
    lines.append(f"rank_label: {rank_label}")
    lines.append(f"primary_tau_null: {primary_threshold:.16e}")
    lines.append("")
    lines.append("Threshold sweep")
    lines.append("---------------")
    for item in threshold_summaries:
        lines.append(
            f"{item['threshold_name']}: tau={item['threshold_value']:.3e}, "
            f"null={item['null_count']}/{item['parameter_count']} "
            f"({100.0 * item['null_fraction']:.2f}%), "
            f"relative={item['relative_threshold']:.3e}"
        )
    lines.append("")
    lines.append("Mechanistic null-space projection")
    lines.append("---------------------------------")
    for row in projection_rows:
        lines.append(
            f"{row['threshold_name']} | {row['target_parameter']} | "
            f"projection_energy={row['projection_energy']:.6e}, "
            f"projection_norm={row['projection_norm']:.6e}, "
            f"residual_norm={row['residual_norm']:.6e}, "
            f"compensation_excluding_self={row['compensation_energy_excluding_self']:.6e}"
        )
        for group in GROUP_ORDER:
            lines.append(
                f"  {group}: energy={row[f'{group}_energy']:.6e}, "
                f"within_projection={row[f'{group}_fraction_within_projection']:.6e}"
            )
        lines.append("  top components:")
        for comp in row["top_components"]:
            lines.append(
                f"    {comp['rank']}. {comp['parameter']} "
                f"component={comp['component']:.6e}, energy={comp['energy']:.6e}, group={comp['group']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_eigen_spectrum(
    path: Path,
    *,
    eigenvalues: np.ndarray,
    primary_threshold: float,
    sweep_thresholds: list[float],
    rank_label: str,
) -> None:
    x = np.arange(1, eigenvalues.size + 1)
    y = np.log10(np.maximum(np.abs(eigenvalues), 1.0e-300))

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    ax.plot(x, y, color="#2F5597", linewidth=1.6)

    marker_x: list[float] = []
    marker_y: list[float] = []
    for th in SPECTRUM_MARKER_THRESHOLDS:
        target_y = math.log10(th)
        intersections: list[float] = []
        for idx in range(y.size - 1):
            y0 = float(y[idx])
            y1 = float(y[idx + 1])
            if y0 == target_y:
                intersections.append(float(x[idx]))
            if y0 == y1:
                continue
            if (y0 - target_y) * (y1 - target_y) <= 0.0:
                frac = (target_y - y0) / (y1 - y0)
                if 0.0 <= frac <= 1.0:
                    intersections.append(float(x[idx] + frac * (x[idx + 1] - x[idx])))
        for x_intersect in sorted(set(round(value, 12) for value in intersections)):
            marker_x.append(float(x_intersect))
            marker_y.append(float(target_y))

    ax.scatter(marker_x, marker_y, color="#B00020", s=28, zorder=4)
    ax.set_xlim(0, eigenvalues.size + 8)
    ax.set_ylim(float(np.min(y)) - 0.8, float(np.max(y)) + 0.8)
    ax.set_title(f"AFM05 Step3 Eigen Spectrum ({rank_label})")
    ax.set_xlabel("Eigenvalue index (ascending)")
    ax.set_ylabel("log10(abs(eigenvalue))")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def rows_for_threshold(
    projection_rows: list[dict[str, Any]],
    threshold_value: float,
) -> list[dict[str, Any]]:
    if not projection_rows:
        return []

    candidates = [
        row
        for row in projection_rows
        if math.isclose(float(row["threshold_value"]), threshold_value, rel_tol=1.0e-8, abs_tol=0.0)
    ]
    if not candidates:
        return []

    selected: dict[str, dict[str, Any]] = {}
    for target in TARGET_LABELS:
        target_rows = [row for row in candidates if row["target_parameter"] == target]
        if target_rows:
            selected[target] = target_rows[0]
    return [selected[target] for target in TARGET_LABELS if target in selected]


def plot_projection_groups(
    path: Path,
    *,
    projection_rows: list[dict[str, Any]],
    primary_threshold: float,
    rank_label: str,
    display_taus: tuple[float, float, float, float] = PROJECTION_GROUP_TAUS,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(28.0, 18.2), dpi=160)
    axes_flat = list(axes.ravel())
    legend_handles = []
    legend_labels = []
    legend_groups_seen = set()

    for ax, tau in zip(axes_flat, display_taus):
        tau_title = rf"$\tau = 10^{{{int(round(math.log10(tau)))}}}$"
        rows = rows_for_threshold(projection_rows, tau)
        if not rows:
            ax.text(
                0.5,
                0.5,
                "not computed",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=PROJECTION_GROUP_TITLE_FONTSIZE,
                color="#555555",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(tau_title, fontsize=PROJECTION_GROUP_TITLE_FONTSIZE)
            continue

        x = PROJECTION_GROUP_BAR_X[: len(rows)]
        labels = [row["target_parameter"].replace("mech.raw_", "").replace("[0]", "") for row in rows]
        bottoms = np.zeros(len(rows), dtype=float)
        totals = np.zeros(len(rows), dtype=float)
        group_values: dict[str, np.ndarray] = {}
        component_pct_labels: list[list[tuple[float, str, str, str]]] = [[] for _ in rows]
        for group in GROUP_ORDER:
            vals = np.asarray([row[f"{group}_energy"] for row in rows], dtype=float)
            group_values[group] = vals
            totals += np.maximum(vals, 0.0)

        for group in GROUP_ORDER:
            vals = group_values[group]
            if np.all(vals <= 0.0):
                continue
            color = GROUP_COLORS.get(group)
            bars = ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=group,
                color=color,
                width=PROJECTION_GROUP_BAR_WIDTH,
            )
            for idx, value in enumerate(vals):
                if value <= 0.0 or totals[idx] <= 0.0:
                    continue
                pct = 100.0 * float(value) / float(totals[idx])
                if pct < PROJECTION_GROUP_PERCENT_MIN:
                    continue
                component_pct_labels[idx].append(
                    (float(bottoms[idx] + value * 0.5), format_projection_percent(pct), str(color), group)
                )
            bottoms += vals
            if group not in legend_groups_seen:
                legend_handles.append(bars[0])
                legend_labels.append(GROUP_LABELS.get(group, group))
                legend_groups_seen.add(group)

        max_bar = float(np.max(bottoms)) if bottoms.size else 0.0
        y_top = max(max_bar * 1.18, 1.0e-12)
        for idx, items in enumerate(component_pct_labels):
            if not items:
                continue
            items = sorted(items, key=lambda item: item[0])
            for item_index, (component_mid_y, text, color, _group) in enumerate(items):
                side = -1 if item_index % 2 == 0 else 1
                x_label = x[idx] + side * (PROJECTION_GROUP_BAR_WIDTH * 0.5 + 0.025)
                ha = "left" if side > 0 else "right"
                ax.text(
                    x_label,
                    component_mid_y,
                    text,
                    ha=ha,
                    va="center",
                    fontsize=PROJECTION_GROUP_PERCENT_FONTSIZE,
                    fontweight="bold",
                    color=color,
                    clip_on=False,
                )
        for idx, total in enumerate(bottoms):
            ax.text(
                x[idx],
                total + y_top * 0.025,
                f"{total:.2e}",
                ha="center",
                va="bottom",
                fontsize=PROJECTION_GROUP_TOTAL_FONTSIZE,
                fontweight="bold",
                color="#222222",
            )
        ax.set_title(tau_title, fontsize=PROJECTION_GROUP_TITLE_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="both", which="major", labelsize=PROJECTION_GROUP_TICK_FONTSIZE)
        ax.set_xlim(-0.90, 0.90)
        ax.set_ylim(0.0, y_top)
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes[:, 0]:
        ax.set_ylabel("Projection energy", fontsize=PROJECTION_GROUP_AXIS_LABEL_FONTSIZE)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=min(len(legend_labels), 6),
            frameon=False,
            fontsize=PROJECTION_GROUP_LEGEND_FONTSIZE,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=0.9,
        )
    fig.tight_layout(rect=(0, 0.14, 1.0, 1.0), pad=0.8, w_pad=0.8, h_pad=1.0)
    fig.savefig(path)
    plt.close(fig)


def plot_threshold_sweep(
    path: Path,
    *,
    projection_rows: list[dict[str, Any]],
    rank_label: str,
) -> None:
    sweep_rows = [row for row in projection_rows if row["threshold_name"].startswith("sweep_")]
    if not sweep_rows:
        return

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=180)
    for target in TARGET_LABELS:
        rows = sorted(
            [row for row in sweep_rows if row["target_parameter"] == target],
            key=lambda row: float(row["threshold_value"]),
        )
        if not rows:
            continue
        x = [float(row["threshold_value"]) for row in rows]
        y = [float(row["projection_energy"]) for row in rows]
        ax.semilogx(
            x,
            y,
            marker="o",
            markersize=2.4,
            linewidth=1.4,
            label=target.replace("mech.", ""),
        )
    ax.set_xlabel("abs(eigenvalue) null threshold")
    ax.set_ylabel("Projection energy in null space")
    ax.set_title(f"AFM05 Step3 Null Projection Threshold Sweep ({rank_label})")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def copy_latest(src: Path, latest: Path) -> None:
    latest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, latest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-result",
        type=Path,
        default=THIS_DIR / "results" / "afm05_step3_identifiability_trained_latest.json",
        help="Step3 trained result JSON to analyze.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=THIS_DIR / "results" / "post_analysis",
        help="Directory for post-analysis tables.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=THIS_DIR / "visualization",
        help="Directory for post-analysis figures.",
    )
    parser.add_argument("--abs-floor", type=float, default=1.0e-8)
    parser.add_argument("--rel-factor", type=float, default=0.0)
    parser.add_argument("--thresholds", default="1e-12,1e-10,1e-8")
    parser.add_argument("--threshold-log-min", type=float, default=0.0)
    parser.add_argument("--threshold-log-max", type=float, default=0.0)
    parser.add_argument(
        "--threshold-points",
        type=int,
        default=0,
        help="If >0, ignore --thresholds and use logspace(threshold-log-min, threshold-log-max, threshold-points).",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output-tag", default="")
    parser.add_argument(
        "--stable-output-tag",
        action="store_true",
        help="Use parameter_set + rank_label as the default output tag instead of appending a timestamp.",
    )
    parser.add_argument(
        "--plots",
        default="none",
        help="Deprecated for this post-analysis entry. Figures are generated by visualization runners.",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not write latest aliases. Useful for timestamped visualization runners.",
    )
    args = parser.parse_args()

    input_result = args.input_result.resolve()
    if not input_result.exists():
        raise FileNotFoundError(f"Step3 result not found: {input_result}")

    results_dir = args.results_dir.resolve()
    visualization_dir = args.visualization_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading step3 result -> {input_result}")
    result = load_step3_result(input_result)
    labels, eigenvalues, eigenvectors = prepare_eigensystem(result)

    rank_label = str(result.get("rank_label") or f"rank{result.get('rank', 'unknown')}")
    parameter_set = str(result.get("parameter_set", "unknown"))
    max_abs_eval = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    primary_threshold = float(max(args.abs_floor, args.rel_factor * max_abs_eval))
    sweep_thresholds = make_sweep_thresholds(args)
    threshold_items = unique_thresholds(primary_threshold, sweep_thresholds)

    log(f"rank_label={rank_label} parameter_set={parameter_set} n={len(labels)}")
    log(f"max_abs_eigenvalue={max_abs_eval:.16e} primary_tau={primary_threshold:.16e}")

    threshold_summaries: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for threshold_name, threshold_value in threshold_items:
        log(f"analyzing threshold {threshold_name} tau={threshold_value:.3e}")
        summary, rows = projection_analysis_for_threshold(
            labels=labels,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            top_k=args.top_k,
        )
        threshold_summaries.append(summary)
        projection_rows.extend(rows)

    stamp = now_stamp()
    tag = args.output_tag.strip()
    if not tag:
        if args.stable_output_tag:
            tag = f"{parameter_set}_{rank_label}"
        else:
            tag = f"{parameter_set}_{rank_label}_{stamp}"
    base = f"afm05_step3_post_analysis_{tag}"

    csv_path = results_dir / f"{base}.csv"
    json_path = results_dir / f"{base}.json"
    txt_path = results_dir / f"{base}.txt"

    log(f"writing tables -> {results_dir}")
    csv_rows = csv_rows_from_projection(projection_rows, top_k=args.top_k)
    write_csv(csv_path, csv_rows)

    compact_projection_rows: list[dict[str, Any]] = []
    for row in projection_rows:
        compact = dict(row)
        # Keep the primary projection vector for future perturbation tests; omit
        # sweep vectors to avoid noisy JSON growth.
        if row["threshold_name"] != "primary":
            compact.pop("projection_vector", None)
        compact_projection_rows.append(compact)

    json_payload = {
        "schema_version": "afm05_step3_post_analysis_v1",
        "source_result": repo_rel(input_result),
        "rank_label": rank_label,
        "rank": result.get("rank"),
        "candidate_b": result.get("candidate_b"),
        "parameter_set": parameter_set,
        "observable_components": result.get("observable_components"),
        "parameter_count": len(labels),
        "target_parameters": list(TARGET_LABELS),
        "threshold_rule": {
            "abs_floor": float(args.abs_floor),
            "rel_factor": float(args.rel_factor),
            "max_abs_eigenvalue": max_abs_eval,
            "primary_threshold": primary_threshold,
            "sweep_thresholds": sweep_thresholds,
        },
        "threshold_summaries": threshold_summaries,
        "projection_rows": compact_projection_rows,
    }
    json_path.write_text(json.dumps(jsonable(json_payload), indent=2), encoding="utf-8")

    write_text_summary(
        txt_path,
        source_result=input_result,
        rank_label=rank_label,
        primary_threshold=primary_threshold,
        threshold_summaries=threshold_summaries,
        projection_rows=projection_rows,
    )

    log("writing figures -> disabled in post-analysis; use runner/visualization runner scripts")

    if not args.no_latest:
        latest_base = "afm05_step3_post_analysis_latest"
        copy_latest(csv_path, results_dir / f"{latest_base}.csv")
        copy_latest(json_path, results_dir / f"{latest_base}.json")
        copy_latest(txt_path, results_dir / f"{latest_base}.txt")

    log(f"done -> {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
