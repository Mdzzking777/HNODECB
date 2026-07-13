from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(REPO_ROOT))

from AFM04.stage1pluslight.grid import CS_BOUNDS, KS_BOUNDS, loggrid_value  # noqa: E402
from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import CS, KS  # noqa: E402

DEFAULT_RESULT_PATH = REPO_ROOT / "AFM04" / "prestage2" / "results" / "afm_prest2_04_candidates_b.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "prestage2" / "visualization"
GRID_NODE_COUNT = 10
SEARCH_SPACE_PAD_FRACTION = 0.03
CANDIDATE_LABEL_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 12.75
SYMBOL_LINEAR_SCALE = 1.5
SYMBOL_AREA_SCALE = SYMBOL_LINEAR_SCALE**2


class _LegendArrowHandler(HandlerPatch):
    def create_artists(
        self,
        legend: Any,
        original_handle: Any,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Any,
    ) -> list[Any]:
        arrow = FancyArrowPatch(
            (xdescent, ydescent + 0.5 * height),
            (xdescent + width, ydescent + 0.5 * height),
            arrowstyle="->",
            mutation_scale=fontsize * SYMBOL_LINEAR_SCALE,
            linewidth=original_handle.get_linewidth(),
            color=original_handle.get_edgecolor(),
            fill=False,
            transform=trans,
        )
        return [arrow]


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


def _out_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "afm_prest2_candidates_b_001_020_st1pl_to_final_ks_cs_2d.png"


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
    for node_idx in range(1, GRID_NODE_COUNT + 1):
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
    return ks_node in (1, GRID_NODE_COUNT) or cs_node in (1, GRID_NODE_COUNT)


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
    return loggrid_value(int(node_idx) - 1, GRID_NODE_COUNT, float(lo), float(hi))


def _one_grid_step_down_cs(value: float) -> float:
    if not (math.isfinite(value) and value > 0.0):
        return value
    lo, hi = float(CS_BOUNDS[0]), float(CS_BOUNDS[1])
    step_ratio = 10.0 ** ((math.log10(hi) - math.log10(lo)) / float(GRID_NODE_COUNT - 1))
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


def _search_space_limits(bounds: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(bounds[0]), float(bounds[1])
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    pad = (log_hi - log_lo) * SEARCH_SPACE_PAD_FRACTION
    return 10.0 ** (log_lo - pad), 10.0 ** (log_hi + pad)


def _draw_source_grid_clusters(ax: Any, records: list[dict[str, Any]]) -> None:
    cores = _cluster_cores(records)
    if not cores:
        return

    cluster_x_values: list[float] = []
    cluster_y_values: list[float] = []
    for record, _score in cores:
        ks_node, cs_node = _source_node(record)
        ks_nodes = range(max(1, ks_node - 1), min(GRID_NODE_COUNT, ks_node + 1) + 1)
        cs_nodes = range(max(1, cs_node - 1), min(GRID_NODE_COUNT, cs_node + 1) + 1)
        cluster_x_values.extend(_node_value("ks", idx) for idx in ks_nodes)
        cluster_y_values.extend(_node_value("cs", idx) for idx in cs_nodes)

    x_min, x_max = min(cluster_x_values), max(cluster_x_values)
    y_min, y_max = min(cluster_y_values), max(cluster_y_values)
    line_style = {
        "color": "black",
        "linewidth": 0.625 * SYMBOL_LINEAR_SCALE,
        "linestyle": (0, (4, 3)),
        "alpha": 0.72,
        "zorder": 3,
    }
    ax.plot([x_min, x_max], [y_min, y_min], **line_style)
    ax.plot([x_min, x_max], [y_max, y_max], **line_style)
    ax.plot([x_min, x_min], [y_min, y_max], **line_style)
    ax.plot([x_max, x_max], [y_min, y_max], **line_style)


def _draw_grid_bounds(ax: Any) -> None:
    ks_min, ks_max = float(KS_BOUNDS[0]), float(KS_BOUNDS[1])
    cs_min, cs_max = float(CS_BOUNDS[0]), float(CS_BOUNDS[1])
    line_style = {
        "color": "#2563eb",
        "linewidth": 1.5 * SYMBOL_LINEAR_SCALE,
        "linestyle": "-",
        "alpha": 0.95,
        "zorder": 0.5,
    }
    ax.plot([ks_min, ks_max], [cs_min, cs_min], **line_style)
    ax.plot([ks_min, ks_max], [cs_max, cs_max], **line_style)
    ax.plot([ks_min, ks_min], [cs_min, cs_max], **line_style)
    ax.plot([ks_max, ks_max], [cs_min, cs_max], **line_style)


def plot_candidate_ks_cs_paths(
    records: list[dict[str, Any]],
    out_dir: Path,
    visual_shift_down_candidates: set[int] | None = None,
    legend_loc: str = "lower left",
    legend_bbox: tuple[float, float] = (0.025, 0.015),
    legend_frame: bool = False,
    wrap_legend_labels: bool = False,
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

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(
        ks0_values,
        cs0_values,
        color="#16a34a",
        s=8.4 * SYMBOL_AREA_SCALE,
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(
        ksf_values,
        csf_values,
        facecolors="none",
        edgecolors="crimson",
        linewidths=1.0 * SYMBOL_LINEAR_SCALE,
        s=9.6 * SYMBOL_AREA_SCALE,
        alpha=1.0,
        zorder=5,
    )
    _draw_source_grid_clusters(ax, records)
    _draw_grid_bounds(ax)
    ax.scatter(
        [float(KS)],
        [float(CS)],
        marker="x",
        color="red",
        linewidths=1.5 * SYMBOL_LINEAR_SCALE,
        s=42 * SYMBOL_AREA_SCALE,
        zorder=6,
    )
    for ks0, cs0, ksf, csf in zip(ks0_values, cs0_values, ksf_values, csf_values):
        ax.annotate(
            "",
            xy=(ksf, csf),
            xytext=(ks0, cs0),
            arrowprops={
                "arrowstyle": "->",
                "color": "black",
                "linewidth": 1.0 * SYMBOL_LINEAR_SCALE,
                "mutation_scale": 10.0 * SYMBOL_LINEAR_SCALE,
                "alpha": 0.72,
                "shrinkA": 2,
                "shrinkB": 2,
            },
            zorder=4,
        )
    for candidate, ks, cs in zip(candidates, ks0_values, cs0_values):
        ax.annotate(
            f"B{candidate:02d}",
            (ks, cs),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=CANDIDATE_LABEL_FONTSIZE,
            color="crimson",
            alpha=0.9,
        )
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(r"$k_s$ (N/m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$c_s$ (N s/m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*_search_space_limits(KS_BOUNDS))
    ax.set_ylim(*_search_space_limits(CS_BOUNDS))
    candidates_to_optimize_label = (
        "candidates to be\noptimized" if wrap_legend_labels else "candidates to be optimized"
    )
    optimization_direction_label = (
        "overall optimization\ndirection" if wrap_legend_labels else "overall optimization direction"
    )
    optimization_arrow = FancyArrowPatch(
        (0.0, 0.0),
        (1.0, 0.0),
        arrowstyle="->",
        linewidth=1.0 * SYMBOL_LINEAR_SCALE,
        color="black",
        fill=False,
        label=optimization_direction_label,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#2563eb",
            linewidth=1.5 * SYMBOL_LINEAR_SCALE,
            linestyle="-",
            label="global search space",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color="red",
            markeredgewidth=1.5 * SYMBOL_LINEAR_SCALE,
            markersize=6.5 * SYMBOL_LINEAR_SCALE,
            label="ground truth",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#16a34a",
            markeredgecolor="#16a34a",
            markersize=4.5 * SYMBOL_LINEAR_SCALE,
            label=candidates_to_optimize_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="crimson",
            markeredgewidth=1.0 * SYMBOL_LINEAR_SCALE,
            markersize=5.0 * SYMBOL_LINEAR_SCALE,
            label="optimized candidates",
        ),
        optimization_arrow,
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=0.625 * SYMBOL_LINEAR_SCALE,
            linestyle=(0, (4, 3)),
            label="gathering",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        frameon=legend_frame,
        framealpha=0.92 if legend_frame else None,
        facecolor="white" if legend_frame else None,
        edgecolor="0.85" if legend_frame else None,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.45,
        borderaxespad=0.0,
        handler_map={FancyArrowPatch: _LegendArrowHandler()},
    )

    fig.tight_layout()
    out_path = _out_path(out_dir)
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
    parser.add_argument(
        "--legend-loc",
        default="lower left",
        help="Matplotlib legend loc, e.g. 'lower left' or 'upper right'.",
    )
    parser.add_argument(
        "--legend-bbox",
        type=float,
        nargs=2,
        default=(0.025, 0.015),
        metavar=("X", "Y"),
        help="Legend bbox_to_anchor in axes coordinates.",
    )
    parser.add_argument(
        "--legend-frame",
        action="store_true",
        help="Draw a white legend frame to avoid overlaps with plot elements.",
    )
    parser.add_argument(
        "--wrap-legend-labels",
        action="store_true",
        help="Wrap the two longest legend labels to keep the legend narrow.",
    )
    args = parser.parse_args()

    payload = _load_pickle(args.result)
    records = _candidate_records(payload)
    out_path = plot_candidate_ks_cs_paths(
        records,
        args.out_dir,
        set(args.visual_shift_down_candidates),
        legend_loc=str(args.legend_loc),
        legend_bbox=(float(args.legend_bbox[0]), float(args.legend_bbox[1])),
        legend_frame=bool(args.legend_frame),
        wrap_legend_labels=bool(args.wrap_legend_labels),
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
