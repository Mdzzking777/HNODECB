from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM05").is_dir() and ((path / "user requirements").is_dir() or (path / "Requirements").is_dir()):
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(REPO_ROOT))

from AFM05.stage1pluslight.grid import CS_BOUNDS, KS_BOUNDS, loggrid_value  # noqa: E402

DEFAULT_RESULT_PATH = REPO_ROOT / "AFM05" / "prestage2" / "results" / "afm_prest2_05_candidates_b.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM05" / "prestage2" / "visualization"
KS_GRID_NODE_COUNT = max(1, int(os.environ.get("HNODECB_AFM05_STAGE1PLUS_GRID_KS_NODES", "20")))
CS_GRID_NODE_COUNT = max(1, int(os.environ.get("HNODECB_AFM05_STAGE1PLUS_GRID_CS_NODES", "50")))
KS_VIS_RANGE = (float(KS_BOUNDS[0]), float(KS_BOUNDS[1]))
CS_VIS_RANGE = (float(CS_BOUNDS[0]), float(CS_BOUNDS[1]))


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected payload type for {path}: {type(payload)!r}")
    return payload


def _candidate_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("candidate_b_records", "candidate_records"):
        records = payload.get(key)
        if isinstance(records, list) and records:
            out = [rec for rec in records if isinstance(rec, dict)]
            return sorted(out, key=_candidate_id)
    return []


def _candidate_id(record: dict[str, Any]) -> int:
    for key in ("candidate_b", "candidate", "top_mech_winner_a", "newrank_a"):
        try:
            value = int(record.get(key, 0))
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _out_path(out_dir: Path, records: list[dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_candidate = max((_candidate_id(record) for record in records), default=len(records))
    return out_dir / f"afm_prest2_candidates_b_001_{max_candidate:03d}_st1pl_to_final_ks_cs_2d.png"


def _source_point(record: dict[str, Any]) -> tuple[int, float, float]:
    candidate = _candidate_id(record)
    ks = _finite_or_nan(record.get("ks0", float("nan")))
    cs = _finite_or_nan(record.get("cs0", float("nan")))
    if not (math.isfinite(ks) and math.isfinite(cs)):
        raise RuntimeError(f"Candidate B {candidate} has no finite st1pl source ks0/cs0.")
    return candidate, ks, cs


def _final_point(record: dict[str, Any]) -> tuple[int, float, float]:
    candidate = _candidate_id(record)
    ks = _finite_or_nan(record.get("ks_hat", record.get("ks", float("nan"))))
    cs = _finite_or_nan(record.get("cs_hat", record.get("cs", float("nan"))))
    if not (math.isfinite(ks) and math.isfinite(cs)):
        raise RuntimeError(f"Candidate B {candidate} has no finite final ks_hat/cs_hat.")
    return candidate, ks, cs


def _source_node(record: dict[str, Any]) -> tuple[int, int]:
    try:
        ks_node = int(record.get("source_ks_node_idx", 0))
        cs_node = int(record.get("source_cs_node_idx", 0))
    except Exception:
        ks_node = 0
        cs_node = 0
    if ks_node <= 0:
        ks_node = _nearest_grid_node("ks", record.get("ks0", float("nan")))
    if cs_node <= 0:
        cs_node = _nearest_grid_node("cs", record.get("cs0", float("nan")))
    if ks_node <= 0 or cs_node <= 0:
        raise RuntimeError(f"Candidate B {_candidate_id(record)} has no finite source grid node.")
    return ks_node, cs_node


def _nearest_grid_node(axis: str, value: Any) -> int:
    value = _finite_or_nan(value)
    if not (math.isfinite(value) and value > 0.0):
        return 0
    log_value = math.log10(value)
    best_idx = 0
    best_dist = float("inf")
    for node_idx in range(1, _axis_node_count(axis) + 1):
        node_value = _node_value(axis, node_idx)
        if node_value <= 0.0:
            continue
        dist = abs(log_value - math.log10(node_value))
        if dist < best_dist:
            best_dist = dist
            best_idx = node_idx
    return best_idx


def _grid_distance_one(node_a: tuple[int, int], node_b: tuple[int, int]) -> bool:
    dx = abs(int(node_a[0]) - int(node_b[0]))
    dy = abs(int(node_a[1]) - int(node_b[1]))
    return max(dx, dy) == 1


def _is_boundary_node(node: tuple[int, int]) -> bool:
    ks_node, cs_node = int(node[0]), int(node[1])
    return ks_node in (1, KS_GRID_NODE_COUNT) or cs_node in (1, CS_GRID_NODE_COUNT)


def _axis_node_count(axis: str) -> int:
    if axis == "ks":
        return KS_GRID_NODE_COUNT
    if axis == "cs":
        return CS_GRID_NODE_COUNT
    raise ValueError(axis)


def _cluster_cores(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    nodes = {id(record): _source_node(record) for record in records}
    core_records = [record for record in records if not _is_boundary_node(nodes[id(record)])]
    neighbor_records = [record for record in records if not _is_boundary_node(nodes[id(record)])]
    scores: list[tuple[dict[str, Any], int]] = []
    for record in core_records:
        node = nodes[id(record)]
        score = sum(1 for other in neighbor_records if other is not record and _grid_distance_one(node, nodes[id(other)]))
        scores.append((record, score))
    max_score = max((score for _, score in scores), default=0)
    if max_score <= 0:
        return []
    return [(record, score) for record, score in scores if score == max_score]


def _node_value(axis: str, node_idx: int) -> float:
    if axis == "ks":
        lo, hi = KS_BOUNDS
    elif axis == "cs":
        lo, hi = CS_BOUNDS
    else:
        raise ValueError(axis)
    return loggrid_value(int(node_idx) - 1, _axis_node_count(axis), float(lo), float(hi))


def _one_grid_step_down_cs(value: float) -> float:
    if not (math.isfinite(value) and value > 0.0):
        return value
    lo, hi = float(CS_BOUNDS[0]), float(CS_BOUNDS[1])
    step_ratio = 10.0 ** ((math.log10(hi) - math.log10(lo)) / float(CS_GRID_NODE_COUNT - 1))
    return value / step_ratio


def _log_axis_limits(values: list[float], bounds: tuple[float, float], pad_fraction: float = 0.08) -> tuple[float, float]:
    positive_values = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0]
    positive_values.extend([float(bounds[0]), float(bounds[1])])
    lo = min(positive_values)
    hi = max(positive_values)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo > 0.0 and hi > 0.0):
        return float(bounds[0]), float(bounds[1])
    if lo == hi:
        return lo * 0.8, hi * 1.25
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    pad = (log_hi - log_lo) * float(pad_fraction)
    return 10.0 ** (log_lo - pad), 10.0 ** (log_hi + pad)


def _draw_source_grid_clusters(ax: Any, records: list[dict[str, Any]]) -> None:
    cores = _cluster_cores(records)
    rectangles: list[tuple[float, float, float, float]] = []
    for record, _score in cores:
        ks_node, cs_node = _source_node(record)
        ks_nodes = range(max(1, ks_node - 1), min(KS_GRID_NODE_COUNT, ks_node + 1) + 1)
        cs_nodes = range(max(1, cs_node - 1), min(CS_GRID_NODE_COUNT, cs_node + 1) + 1)
        x_values = [_node_value("ks", idx) for idx in ks_nodes]
        y_values = [_node_value("cs", idx) for idx in cs_nodes]
        rectangles.append((min(x_values), max(x_values), min(y_values), max(y_values)))

    merged: list[tuple[float, float, float, float]] = []
    for rectangle in rectangles:
        x_min, x_max, y_min, y_max = rectangle
        changed = True
        while changed:
            changed = False
            remaining: list[tuple[float, float, float, float]] = []
            for other in merged:
                ox_min, ox_max, oy_min, oy_max = other
                overlaps = x_min <= ox_max and ox_min <= x_max and y_min <= oy_max and oy_min <= y_max
                if overlaps:
                    x_min = min(x_min, ox_min)
                    x_max = max(x_max, ox_max)
                    y_min = min(y_min, oy_min)
                    y_max = max(y_max, oy_max)
                    changed = True
                else:
                    remaining.append(other)
            merged = remaining
        merged.append((x_min, x_max, y_min, y_max))

    line_style = {
        "color": "black",
        "linewidth": 0.625,
        "linestyle": (0, (4, 3)),
        "alpha": 0.72,
        "zorder": 3,
    }
    for x_min, x_max, y_min, y_max in merged:
        ax.plot([x_min, x_max], [y_min, y_min], **line_style)
        ax.plot([x_min, x_max], [y_max, y_max], **line_style)
        ax.plot([x_min, x_min], [y_min, y_max], **line_style)
        ax.plot([x_max, x_max], [y_min, y_max], **line_style)


def _draw_grid_bounds(ax: Any) -> None:
    ks_min, ks_max = float(KS_BOUNDS[0]), float(KS_BOUNDS[1])
    cs_min, cs_max = float(CS_BOUNDS[0]), float(CS_BOUNDS[1])
    line_style = {
        "color": "#2563eb",
        "linewidth": 1.5,
        "linestyle": "-",
        "alpha": 0.95,
        "zorder": 0.5,
    }
    ax.plot([ks_min, ks_max], [cs_min, cs_min], **line_style)
    ax.plot([ks_min, ks_max], [cs_max, cs_max], **line_style)
    ax.plot([ks_min, ks_min], [cs_min, cs_max], **line_style)
    ax.plot([ks_max, ks_max], [cs_min, cs_max], **line_style)


def _zoom_limits(values: list[float], full_range: tuple[float, float], *, min_span_decades: float = 0.22) -> tuple[float, float]:
    positive_values = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0]
    if not positive_values:
        return full_range
    log_values = [math.log10(v) for v in positive_values]
    log_lo = min(log_values)
    log_hi = max(log_values)
    span = max(log_hi - log_lo, float(min_span_decades))
    pad = 0.18 * span
    center = 0.5 * (log_lo + log_hi)
    log_lo = center - 0.5 * span - pad
    log_hi = center + 0.5 * span + pad
    return 10.0 ** log_lo, 10.0 ** log_hi


def plot_candidate_ks_cs_paths(
    records: list[dict[str, Any]],
    out_dir: Path,
    visual_shift_down_candidates: set[int] | None = None,
) -> Path:
    if not records:
        raise RuntimeError("No candidate B records found.")
    visual_shift_down_candidates = visual_shift_down_candidates or set()

    candidates: list[int] = []
    ks0_values: list[float] = []
    cs0_values: list[float] = []
    ksf_values: list[float] = []
    csf_values: list[float] = []
    for record in records:
        candidate, ks0, cs0 = _source_point(record)
        final_candidate, ksf, csf = _final_point(record)
        if final_candidate != candidate:
            raise RuntimeError(f"Candidate id mismatch: source B{candidate}, final B{final_candidate}.")
        if candidate in visual_shift_down_candidates:
            cs0 = _one_grid_step_down_cs(cs0)
            csf = _one_grid_step_down_cs(csf)
        candidates.append(candidate)
        ks0_values.append(ks0)
        cs0_values.append(cs0)
        ksf_values.append(ksf)
        csf_values.append(csf)

    fig, (ax, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.6),
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.18},
    )

    zoom_xlim = _zoom_limits([*ks0_values, *ksf_values], KS_VIS_RANGE)
    zoom_ylim = _zoom_limits([*cs0_values, *csf_values], CS_VIS_RANGE, min_span_decades=0.18)
    if max(cs0_values + csf_values) >= float(CS_BOUNDS[1]) * 0.995:
        zoom_ylim = (zoom_ylim[0], max(zoom_ylim[1], float(CS_BOUNDS[1]) * 1.55))

    def draw_panel(panel: Any, *, xlim: tuple[float, float], ylim: tuple[float, float], label_candidates: bool) -> None:
        panel.scatter(ks0_values, cs0_values, color="#16a34a", s=8.4, alpha=0.9, zorder=2)
        panel.scatter(ksf_values, csf_values, facecolors="none", edgecolors="crimson", linewidths=1.0, s=9.6, alpha=1.0, zorder=5)
        _draw_source_grid_clusters(panel, records)
        _draw_grid_bounds(panel)
        for ks0, cs0, ksf, csf in zip(ks0_values, cs0_values, ksf_values, csf_values):
            panel.annotate(
                "",
                xy=(ksf, csf),
                xytext=(ks0, cs0),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "black",
                    "linewidth": 1.0,
                    "alpha": 0.72,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=4,
            )
        panel.grid(True, alpha=0.25)
        panel.set_xlabel(r"$k_s$ (N/m)")
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(*xlim)
        panel.set_ylim(*ylim)
        if label_candidates:
            for candidate, ks, cs in zip(candidates, ks0_values, cs0_values):
                panel.annotate(
                    f"B{candidate:02d}",
                    (ks, cs),
                    xytext=(0.0, 0.0),
                    textcoords="offset points",
                    fontsize=6,
                    color="crimson",
                    alpha=0.9,
                    ha="center",
                    va="center",
                    clip_on=True,
                    zorder=7,
                )

    draw_panel(ax, xlim=KS_VIS_RANGE, ylim=CS_VIS_RANGE, label_candidates=False)
    draw_panel(ax_zoom, xlim=zoom_xlim, ylim=zoom_ylim, label_candidates=True)
    ax.set_ylabel(r"$c_s$ (N s/m)")
    ax_zoom.set_ylabel(r"$c_s$ (N s/m)")
    legend_handles = [
        Line2D([0], [0], color="#2563eb", linewidth=1.5, linestyle="-", label="global search space"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#16a34a",
            markeredgecolor="#16a34a",
            markersize=4.5,
            label="candidates before screening",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="crimson",
            markeredgewidth=1.0,
            markersize=5.0,
            label="final candidates",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="black",
            linewidth=1.0,
            marker=">",
            markevery=[1],
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=5.0,
            label="optimization from screening",
        ),
        Line2D([0], [0], color="black", linewidth=0.625, linestyle=(0, (4, 3)), label="densest cluster"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.025, 0.015),
        frameon=False,
        fontsize=8.5,
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.45,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    out_path = _out_path(out_dir, records)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot st1pl source to prest2 final ks-cs paths for prest2 B candidates.")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--visual-shift-down-candidates",
        type=int,
        nargs="*",
        default=[],
        help="Candidate ids to move down by one cs log-grid step for visual separation only.",
    )
    args = parser.parse_args()

    payload = _load_pickle(args.result)
    records = _candidate_records(payload)
    out_path = plot_candidate_ks_cs_paths(records, args.out_dir, set(args.visual_shift_down_candidates))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
