"""Render a static PNG from an AFM04 prestage2 Plotly 3D HTML.

The archived prestage2 HTML already contains the stage1 landscape, promotion
paths, and final candidate endpoints.  This script redraws those traces with
Matplotlib so the figure can be used as a static publication-style image.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from AFM04.stage1pluslight.visualization_runner.plot_afm_stage1pluslight_3d_static_from_html_04 import (
    _apply_axis_range,
    _apply_axis_ticks,
    _axis_range_values,
    _axis_spec,
    _colorscale_to_cmap,
    _compact_sci,
    _extract_plotly_json,
)


DEFAULT_HTML = (
    Path(__file__).resolve().parents[3]
    / "AFM04"
    / "archive"
    / "prest2"
    / "10x10x1000_W1"
    / "afm_prest2_04_allcandidates_ks_logcs_val_loss_3d.html"
)


def _array(trace: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(trace.get(key, []), dtype=float)


def _valid_xyz(trace: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _array(trace, "x")
    y = _array(trace, "y")
    z = _array(trace, "z")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[valid], y[valid], z[valid], valid


def _trace_by_name(traces: list[dict[str, Any]], prefix: str) -> dict[str, Any] | None:
    prefix_norm = prefix.lower()
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        if str(trace.get("name", "")).lower().startswith(prefix_norm):
            return trace
    return None


def _ground_truth_point(traces: list[dict[str, Any]]) -> tuple[float, float] | None:
    trace = _trace_by_name(traces, "true (ks, cs)")
    if trace is None:
        return None
    x = _array(trace, "x")
    y = _array(trace, "y")
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return None
    return float(x[valid][0]), float(y[valid][0])


def _marker_style(trace: dict[str, Any]) -> tuple[Any, Normalize, np.ndarray | None]:
    marker = trace.get("marker", {})
    if not isinstance(marker, dict):
        return plt.get_cmap("turbo"), Normalize(0.0, 1.0), None
    colors = marker.get("color")
    color_values = None
    if isinstance(colors, list):
        color_values = np.asarray(colors, dtype=float)
    cmap = _colorscale_to_cmap(marker.get("colorscale"))
    if color_values is not None and np.any(np.isfinite(color_values)):
        cmin = float(marker.get("cmin", np.nanmin(color_values)))
        cmax = float(marker.get("cmax", np.nanmax(color_values)))
    else:
        cmin = float(marker.get("cmin", 0.0))
        cmax = float(marker.get("cmax", 1.0))
    if not math.isfinite(cmin) or not math.isfinite(cmax) or cmin == cmax:
        cmin, cmax = 0.0, 1.0
    return cmap, Normalize(cmin, cmax), color_values


def _scatter_colored(
    ax: Any,
    trace: dict[str, Any],
    *,
    size_scale: float,
    alpha: float,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    rasterized: bool = True,
) -> tuple[Any, Normalize, np.ndarray | None]:
    x, y, z, valid = _valid_xyz(trace)
    cmap, norm, raw_colors = _marker_style(trace)
    marker = trace.get("marker", {})
    marker = marker if isinstance(marker, dict) else {}
    size = float(marker.get("size", 4.0)) * size_scale

    if raw_colors is not None and len(raw_colors) == len(valid):
        c = raw_colors[valid]
        ax.scatter(
            x,
            y,
            z,
            c=c,
            cmap=cmap,
            norm=norm,
            s=size,
            alpha=alpha,
            edgecolors=edgecolor,
            linewidths=linewidth,
            depthshade=False,
            rasterized=rasterized,
        )
    else:
        color = marker.get("color", "#1f77b4")
        if not isinstance(color, str):
            color = "#1f77b4"
        ax.scatter(
            x,
            y,
            z,
            c=color,
            s=size,
            alpha=alpha,
            edgecolors=edgecolor,
            linewidths=linewidth,
            depthshade=False,
            rasterized=rasterized,
        )
    return cmap, norm, raw_colors[valid] if raw_colors is not None and len(raw_colors) == len(valid) else None


def _draw_line_trace(ax: Any, trace: dict[str, Any], *, z_floor_log10: float | None = None) -> None:
    x, y, z, _ = _valid_xyz(trace)
    if len(x) < 2:
        return
    line = trace.get("line", {})
    line = line if isinstance(line, dict) else {}
    name = str(trace.get("name", "")).lower()
    if z_floor_log10 is not None and name in {"ks_true", "cs_true"}:
        z = np.full_like(z, float(z_floor_log10))
    ax.plot(
        x,
        y,
        z,
        color=line.get("color", "#111111"),
        linewidth=max(float(line.get("width", 2.0)), 0.8),
        alpha=0.95,
    )


def _draw_promotion_lines_and_arrows(ax: Any, traces: list[dict[str, Any]], *, color: str) -> int:
    count = 0
    for trace in traces:
        if not isinstance(trace, dict) or trace.get("type") != "scatter3d":
            continue
        line = trace.get("line", {})
        line = line if isinstance(line, dict) else {}
        if str(line.get("color", "")).lower() != color.lower():
            continue
        x, y, z, _ = _valid_xyz(trace)
        if len(x) < 2:
            continue
        width = 0.75 if color.lower() == "#1a8f3f" else 1.15
        alpha = 0.46 if color.lower() == "#1a8f3f" else 0.90
        ax.plot(x, y, z, color=color, linewidth=width, alpha=alpha)
        dx, dy, dz = x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]
        ax.quiver(
            x[0],
            y[0],
            z[0],
            dx,
            dy,
            dz,
            color=color,
            linewidth=width,
            alpha=alpha,
            arrow_length_ratio=0.18,
            normalize=False,
        )
        count += 1
    return count


def _promotion_start_points(traces: list[dict[str, Any]], *, color: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for trace in traces:
        if not isinstance(trace, dict) or trace.get("type") != "scatter3d":
            continue
        line = trace.get("line", {})
        line = line if isinstance(line, dict) else {}
        if str(line.get("color", "")).lower() != color.lower():
            continue
        x, y, z, _ = _valid_xyz(trace)
        if len(x) < 2:
            continue
        xs.append(float(x[0]))
        ys.append(float(y[0]))
        zs.append(float(z[0]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float)


def _trace_color_by_xyz(trace: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    x, y, z, valid = _valid_xyz(trace)
    _, _, raw_colors = _marker_style(trace)
    if raw_colors is None or len(raw_colors) != len(valid):
        return {}
    colors = raw_colors[valid]
    return {
        (f"{float(xv):.12f}", f"{float(yv):.12f}", f"{float(zv):.12f}"): float(cv)
        for xv, yv, zv, cv in zip(x, y, z, colors)
    }


def _draw_highlight_points(
    ax: Any,
    trace: dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    edgecolor: str,
    size: float,
) -> None:
    if len(xs) == 0:
        return
    cmap, norm, _ = _marker_style(trace)
    color_lookup = _trace_color_by_xyz(trace)
    colors: list[float] = []
    for xv, yv, zv in zip(xs, ys, zs):
        key = (f"{float(xv):.12f}", f"{float(yv):.12f}", f"{float(zv):.12f}")
        colors.append(color_lookup.get(key, math.nan))
    color_arr = np.asarray(colors, dtype=float)
    if np.all(np.isfinite(color_arr)):
        ax.scatter(
            xs,
            ys,
            zs,
            c=color_arr,
            cmap=cmap,
            norm=norm,
            s=size,
            alpha=1.0,
            edgecolors=edgecolor,
            linewidths=0.95,
            depthshade=False,
            rasterized=True,
        )
    else:
        ax.scatter(
            xs,
            ys,
            zs,
            c="#4f8fd8",
            s=size,
            alpha=1.0,
            edgecolors=edgecolor,
            linewidths=0.95,
            depthshade=False,
            rasterized=True,
        )


def _draw_final_envelope(
    ax: Any,
    final_trace: dict[str, Any],
    *,
    color: str = "#4f8fd8",
    alpha: float = 0.34,
) -> tuple[Any, Normalize, int]:
    x, y, z, valid = _valid_xyz(final_trace)
    cmap, norm, raw_colors = _marker_style(final_trace)
    if len(x) >= 3:
        try:
            ax.plot_trisurf(
                x,
                y,
                z,
                color=color,
                alpha=alpha,
                linewidth=0.0,
                antialiased=True,
                shade=False,
            )
        except Exception:
            pass

    if raw_colors is not None and len(raw_colors) == len(valid):
        c = raw_colors[valid]
        ax.scatter(
            x,
            y,
            z,
            c=c,
            cmap=cmap,
            norm=norm,
            s=118.0,
            alpha=1.0,
            edgecolors="black",
            linewidths=1.25,
            depthshade=False,
            rasterized=True,
        )
    else:
        ax.scatter(
            x,
            y,
            z,
            c="#4f8fd8",
            s=118.0,
            alpha=1.0,
            edgecolors="black",
            linewidths=1.25,
            depthshade=False,
            rasterized=True,
        )
    return cmap, norm, int(len(x))


def render_static(
    html_path: Path,
    png_path: Path,
    *,
    dpi: int,
    z_aspect: float,
    legend_fontsize: float,
    font_scale: float,
    wrap_legend_labels: bool,
) -> None:
    font_scale = max(float(font_scale), 0.1)
    traces, layout = _extract_plotly_json(html_path)
    scene = layout.get("scene", {})
    scene = scene if isinstance(scene, dict) else {}
    xaxis = _axis_spec(scene, "xaxis")
    yaxis = _axis_spec(scene, "yaxis")
    zaxis = _axis_spec(scene, "zaxis")
    z_floor_log10 = math.log10(1.0e-9)

    base_trace = _trace_by_name(traces, "stage1 trials with")
    after10_trace = _trace_by_name(traces, "after 10 epochs")
    final_trace = _trace_by_name(traces, "after +20 more epochs")
    if base_trace is None or final_trace is None:
        raise ValueError(f"Missing required base/final traces in {html_path}")

    fig = plt.figure(figsize=(18.0, 14.8))
    ax = fig.add_axes([0.015, 0.11, 0.57, 0.85], projection="3d")

    base_cmap, base_norm, _ = _scatter_colored(
        ax,
        base_trace,
        size_scale=7.0,
        alpha=0.28,
        edgecolor="none",
        linewidth=0.0,
    )

    entrant_origin_trace = _trace_by_name(traces, "prest2 entrant origins")
    if entrant_origin_trace is not None:
        _scatter_colored(
            ax,
            entrant_origin_trace,
            size_scale=7.0,
            alpha=1.0,
            edgecolor="#006b2f",
            linewidth=0.85,
        )

    finalist_origin_trace = _trace_by_name(traces, "prest2 finalist origins")
    if finalist_origin_trace is not None:
        _scatter_colored(
            ax,
            finalist_origin_trace,
            size_scale=5.0,
            alpha=1.0,
            edgecolor="#006b2f",
            linewidth=0.85,
        )

    if after10_trace is not None:
        _scatter_colored(ax, after10_trace, size_scale=7.0, alpha=0.80, edgecolor="none", linewidth=0.0)

    green_count = _draw_promotion_lines_and_arrows(ax, traces, color="#1a8f3f")
    red_count = _draw_promotion_lines_and_arrows(ax, traces, color="#dc2626")

    if after10_trace is not None:
        rx, ry, rz = _promotion_start_points(traces, color="#dc2626")
        _draw_highlight_points(ax, after10_trace, rx, ry, rz, edgecolor="#dc2626", size=35.0)

    final_cmap, final_norm, final_count = _draw_final_envelope(ax, final_trace)
    ground_truth_point = _ground_truth_point(traces)
    if ground_truth_point is not None:
        ax.scatter(
            [ground_truth_point[0]],
            [ground_truth_point[1]],
            [z_floor_log10],
            marker="D",
            c="#d62728",
            s=150.0,
            edgecolors="black",
            linewidths=1.2,
            depthshade=False,
            zorder=20,
        )

    ax.set_xlabel(r"$k_s$ [N/m]", labelpad=14 * font_scale, fontsize=18 * font_scale)
    ax.set_ylabel(r"$c_s$ [N·s/m]", labelpad=14 * font_scale, fontsize=18 * font_scale)
    ax.set_zlabel("")
    ax.tick_params(axis="x", which="major", labelsize=15 * font_scale, pad=font_scale)
    ax.tick_params(axis="y", which="major", labelsize=15 * font_scale, pad=font_scale)
    ax.tick_params(axis="z", which="major", labelsize=13 * font_scale, pad=12 * font_scale)
    _apply_axis_range(ax, "x", xaxis)
    _apply_axis_range(ax, "y", yaxis)
    _apply_axis_range(ax, "z", zaxis, z_floor_log10=z_floor_log10)
    _apply_axis_ticks(ax, "x", xaxis)
    _apply_axis_ticks(ax, "y", yaxis)
    _apply_axis_ticks(ax, "z", zaxis, z_floor_log10=z_floor_log10)
    z_lo, z_hi = sorted(float(value) for value in ax.get_zlim())
    z_tick_exponents = list(range(math.ceil(z_lo), math.floor(z_hi) + 1))
    ax.set_zticks(
        z_tick_exponents,
        labels=[_compact_sci(10.0 ** exponent) for exponent in z_tick_exponents],
    )
    ax.set_box_aspect((1.0, 1.0, max(float(z_aspect), 0.1)))
    ax.view_init(elev=28, azim=-135)

    fig.text(0.03, 0.52, "training loss", rotation=90, fontsize=18 * font_scale, va="center", ha="center")

    cax = fig.add_axes([0.865, 0.12, 0.024, 0.76])
    colorbar_mappable = ScalarMappable(norm=base_norm, cmap=base_cmap)
    colorbar_mappable.set_array([])
    cbar = fig.colorbar(colorbar_mappable, cax=cax)
    cbar.set_label(r"error between Neural Network estimation and $\bar{F}_{ts}$", fontsize=16 * font_scale)
    cbar.ax.tick_params(labelsize=14 * font_scale)

    handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=5,
            label="all trials",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#1a8f3f",
            linewidth=1.6,
            marker=">",
            markevery=[1],
            markerfacecolor="#1a8f3f",
            markeredgecolor="#1a8f3f",
            markersize=7,
            label="first layer promotion\n(10 epochs)" if wrap_legend_labels else "first layer promotion (10 epochs)",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#dc2626",
            linewidth=1.8,
            marker=">",
            markevery=[1],
            markerfacecolor="#dc2626",
            markeredgecolor="#dc2626",
            markersize=7,
            label=(
                "second layer promotion\n(20 more epochs)"
                if wrap_legend_labels
                else "second layer promotion (20 more epochs)"
            ),
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="black",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
            label="final-promotion\nenvelope" if wrap_legend_labels else "final-promotion envelope",
        ),
    ]
    if ground_truth_point is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markerfacecolor="#d62728",
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=8,
                label="ground-truth",
            )
        )
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.525, 0.53),
        ncol=1,
        fontsize=legend_fontsize * font_scale,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.3,
        handletextpad=0.4,
        labelspacing=0.45,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--z-aspect", type=float, default=2.0)
    parser.add_argument("--legend-fontsize", type=float, default=9.0)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--wrap-legend-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_path = args.html.resolve()
    png_path = args.png.resolve() if args.png else html_path.with_suffix(".png")
    render_static(
        html_path,
        png_path,
        dpi=int(args.dpi),
        z_aspect=float(args.z_aspect),
        legend_fontsize=float(args.legend_fontsize),
        font_scale=float(args.font_scale),
        wrap_legend_labels=bool(args.wrap_legend_labels),
    )
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
