from __future__ import annotations

import argparse
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in (here, *here.parents):
        if (path / "AFM05" / "stage2light").is_dir():
            return path
    raise RuntimeError(f"Could not locate the AFM05 repository root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(REPO_ROOT))

from AFM05.stage1pluslight.grid import CS_BOUNDS, KS_BOUNDS, loggrid_value  # noqa: E402


DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "AFM05" / "Archive" / "st2l" / "valley"
DEFAULT_OUT_DIR = DEFAULT_ARCHIVE_ROOT / "visualization"
DEFAULT_OUT_NAME = "afm_stage2light_05_archived_candidates_initial_to_final_ks_cs_2d.png"
KS_GRID_NODE_COUNT = 20
CS_GRID_NODE_COUNT = 50


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected payload type for {path}: {type(payload)!r}")
    return payload


def _final_history_record(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    history = payload.get("history")
    if not isinstance(history, list):
        raise RuntimeError(f"Missing history in {source}")
    valid = [
        row
        for row in history
        if isinstance(row, dict)
        and _finite(row.get("ks_hat")) > 0.0
        and _finite(row.get("cs_hat")) > 0.0
        and math.isfinite(_finite(row.get("epoch")))
    ]
    if not valid:
        raise RuntimeError(f"No finite final-epoch mechanistic estimate in {source}")
    return max(valid, key=lambda row: _finite(row.get("epoch")))


def _warmstart(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    for key in ("stage1_warmstart", "warmstart"):
        value = payload.get(key)
        if isinstance(value, dict) and _finite(value.get("ks0")) > 0.0 and _finite(value.get("cs0")) > 0.0:
            return value
    raise RuntimeError(f"No finite st2l warmstart point in {source}")


def load_archived_candidates(archive_root: Path) -> list[dict[str, Any]]:
    directory_pattern = re.compile(r"B(?P<candidate>\d+)_rank(?P<rank>\d+)$", re.IGNORECASE)
    records: list[dict[str, Any]] = []
    for candidate_dir in archive_root.iterdir():
        if not candidate_dir.is_dir():
            continue
        match = directory_pattern.fullmatch(candidate_dir.name)
        if match is None:
            continue
        result_path = candidate_dir / "result" / "stage2light_result_p1.viz.pkl"
        if not result_path.is_file():
            continue
        payload = _load_pickle(result_path)
        warmstart = _warmstart(payload, result_path)
        final = _final_history_record(payload, result_path)
        records.append(
            {
                "candidate": int(match.group("candidate")),
                "rank": int(match.group("rank")),
                "ks0": _finite(warmstart["ks0"]),
                "cs0": _finite(warmstart["cs0"]),
                "ks_final": _finite(final["ks_hat"]),
                "cs_final": _finite(final["cs_hat"]),
                "final_epoch": int(round(_finite(final["epoch"]))),
                "result_path": result_path,
            }
        )
    records.sort(key=lambda row: int(row["candidate"]))
    if not records:
        raise RuntimeError(f"No archived st2l candidates found under {archive_root}")
    return records


def _axis_node_count(axis: str) -> int:
    return KS_GRID_NODE_COUNT if axis == "ks" else CS_GRID_NODE_COUNT


def _node_value(axis: str, node_idx: int) -> float:
    bounds = KS_BOUNDS if axis == "ks" else CS_BOUNDS
    return loggrid_value(node_idx - 1, _axis_node_count(axis), float(bounds[0]), float(bounds[1]))


def _nearest_grid_node(axis: str, value: float) -> int:
    log_value = math.log10(float(value))
    return min(
        range(1, _axis_node_count(axis) + 1),
        key=lambda idx: abs(math.log10(_node_value(axis, idx)) - log_value),
    )


def _source_node(record: dict[str, Any]) -> tuple[int, int]:
    return _nearest_grid_node("ks", record["ks0"]), _nearest_grid_node("cs", record["cs0"])


def _is_boundary_node(node: tuple[int, int]) -> bool:
    return node[0] in (1, KS_GRID_NODE_COUNT) or node[1] in (1, CS_GRID_NODE_COUNT)


def _grid_distance_one(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def _cluster_cores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = {id(record): _source_node(record) for record in records}
    interior = [record for record in records if not _is_boundary_node(nodes[id(record)])]
    scores = [
        (
            record,
            sum(
                1
                for other in interior
                if other is not record and _grid_distance_one(nodes[id(record)], nodes[id(other)])
            ),
        )
        for record in interior
    ]
    maximum = max((score for _, score in scores), default=0)
    return [record for record, score in scores if maximum > 0 and score == maximum]


def _draw_source_clusters(ax: Any, records: list[dict[str, Any]]) -> bool:
    rectangles: list[tuple[float, float, float, float]] = []
    for record in _cluster_cores(records):
        ks_node, cs_node = _source_node(record)
        xs = [_node_value("ks", idx) for idx in range(max(1, ks_node - 1), min(KS_GRID_NODE_COUNT, ks_node + 1) + 1)]
        ys = [_node_value("cs", idx) for idx in range(max(1, cs_node - 1), min(CS_GRID_NODE_COUNT, cs_node + 1) + 1)]
        rectangles.append((min(xs), max(xs), min(ys), max(ys)))

    merged: list[tuple[float, float, float, float]] = []
    for rectangle in rectangles:
        x_min, x_max, y_min, y_max = rectangle
        changed = True
        while changed:
            changed = False
            remaining: list[tuple[float, float, float, float]] = []
            for other in merged:
                ox_min, ox_max, oy_min, oy_max = other
                if x_min <= ox_max and ox_min <= x_max and y_min <= oy_max and oy_min <= y_max:
                    x_min, x_max = min(x_min, ox_min), max(x_max, ox_max)
                    y_min, y_max = min(y_min, oy_min), max(y_max, oy_max)
                    changed = True
                else:
                    remaining.append(other)
            merged = remaining
        merged.append((x_min, x_max, y_min, y_max))

    style = {"color": "black", "linewidth": 0.625, "linestyle": (0, (4, 3)), "alpha": 0.72, "zorder": 3}
    for x_min, x_max, y_min, y_max in merged:
        ax.plot([x_min, x_max], [y_min, y_min], **style)
        ax.plot([x_min, x_max], [y_max, y_max], **style)
        ax.plot([x_min, x_min], [y_min, y_max], **style)
        ax.plot([x_max, x_max], [y_min, y_max], **style)
    return bool(merged)


def _draw_grid_bounds(ax: Any) -> None:
    ks_min, ks_max = map(float, KS_BOUNDS)
    cs_min, cs_max = map(float, CS_BOUNDS)
    style = {"color": "#2563eb", "linewidth": 1.5, "linestyle": "-", "alpha": 0.95, "zorder": 0.5}
    ax.plot([ks_min, ks_max], [cs_min, cs_min], **style)
    ax.plot([ks_min, ks_max], [cs_max, cs_max], **style)
    ax.plot([ks_min, ks_min], [cs_min, cs_max], **style)
    ax.plot([ks_max, ks_max], [cs_min, cs_max], **style)


def _zoom_limits(values: list[float], *, min_span_decades: float = 0.22) -> tuple[float, float]:
    logs = [math.log10(value) for value in values if math.isfinite(value) and value > 0.0]
    span = max(max(logs) - min(logs), min_span_decades)
    center = 0.5 * (min(logs) + max(logs))
    pad = 0.18 * span
    return 10.0 ** (center - 0.5 * span - pad), 10.0 ** (center + 0.5 * span + pad)


def plot_archived_candidate_paths(records: list[dict[str, Any]], out_path: Path) -> None:
    ks0 = [row["ks0"] for row in records]
    cs0 = [row["cs0"] for row in records]
    ksf = [row["ks_final"] for row in records]
    csf = [row["cs_final"] for row in records]

    fig, (ax, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.6),
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.18},
    )
    fig.patch.set_facecolor("white")

    def draw_panel(panel: Any, *, xlim: tuple[float, float], ylim: tuple[float, float], labels: bool) -> bool:
        panel.scatter(ks0, cs0, color="#16a34a", s=8.4, alpha=0.9, zorder=2)
        panel.scatter(ksf, csf, facecolors="none", edgecolors="crimson", linewidths=1.0, s=9.6, zorder=5)
        has_clusters = _draw_source_clusters(panel, records)
        _draw_grid_bounds(panel)
        for x0, y0, xf, yf in zip(ks0, cs0, ksf, csf):
            panel.annotate(
                "",
                xy=(xf, yf),
                xytext=(x0, y0),
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
        if labels:
            for row in records:
                panel.annotate(
                    f"B{row['candidate']:02d}",
                    (row["ks0"], row["cs0"]),
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
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(*xlim)
        panel.set_ylim(*ylim)
        panel.set_xlabel(r"$k_s$ (N/m)")
        panel.set_ylabel(r"$c_s$ (N s/m)")
        panel.grid(True, alpha=0.25)
        return has_clusters

    has_clusters = draw_panel(ax, xlim=tuple(map(float, KS_BOUNDS)), ylim=tuple(map(float, CS_BOUNDS)), labels=False)
    draw_panel(
        ax_zoom,
        xlim=_zoom_limits([*ks0, *ksf]),
        ylim=_zoom_limits([*cs0, *csf], min_span_decades=0.18),
        labels=True,
    )

    legend_handles: list[Any] = [
        Line2D([0], [0], color="#2563eb", linewidth=1.5, label="global search space"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#16a34a", markeredgecolor="#16a34a", markersize=4.5, label="st2l initial estimates"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="crimson", markeredgewidth=1.0, markersize=5.0, label="st2l final estimates"),
        Line2D([0, 1], [0, 0], color="black", linewidth=1.0, marker=r"$\rightarrow$", markevery=[1], markersize=7.0, label="st2l optimization direction"),
    ]
    if has_clusters:
        legend_handles.append(Line2D([0], [0], color="black", linewidth=0.625, linestyle=(0, (4, 3)), label="initial-point gathering"))
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

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.975, wspace=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot archived AFM05 st2l candidate paths in the ks-cs plane.")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    args = parser.parse_args()

    records = load_archived_candidates(args.archive_root.resolve())
    out_path = args.out_dir.resolve() / args.out_name
    plot_archived_candidate_paths(records, out_path)
    print(f"candidates={len(records)}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
