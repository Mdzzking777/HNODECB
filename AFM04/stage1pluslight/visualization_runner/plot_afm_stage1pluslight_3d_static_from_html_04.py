"""Render a static PNG/PDF from an AFM04 Stage1pluslight Plotly 3D HTML.

The archived W1 all-trials plot already contains the full Plotly traces and
layout.  This script parses that embedded JSON and redraws the figure with
Matplotlib, so no Plotly/Kaleido dependency is required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np


def _extract_plotly_json(html_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = html_path.read_text(encoding="utf-8")
    traces_prefix = "const traces = "
    layout_prefix = "const layout = "
    traces_start = text.index(traces_prefix) + len(traces_prefix)
    traces_end = text.index(";\n    const layout = ", traces_start)
    layout_start = text.index(layout_prefix, traces_end) + len(layout_prefix)
    layout_end = text.index(";\n    Plotly.newPlot", layout_start)
    traces = json.loads(text[traces_start:traces_end])
    layout = json.loads(text[layout_start:layout_end])
    if not isinstance(traces, list) or not isinstance(layout, dict):
        raise TypeError(f"Unexpected Plotly JSON in {html_path}")
    return traces, layout


def _finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float)
    return arr[np.isfinite(arr)]


def _colorscale_to_cmap(colorscale: Any, *, fallback: str = "turbo") -> Any:
    if not isinstance(colorscale, list) or not colorscale:
        return plt.get_cmap(fallback)
    stops: list[tuple[float, str]] = []
    for item in colorscale:
        if isinstance(item, list) and len(item) >= 2:
            try:
                stops.append((float(item[0]), str(item[1])))
            except Exception:
                continue
    if not stops:
        return plt.get_cmap(fallback)
    return LinearSegmentedColormap.from_list("plotly_colorscale_static", stops)


def _axis_spec(scene: dict[str, Any], key: str) -> dict[str, Any]:
    spec = scene.get(key, {})
    return spec if isinstance(spec, dict) else {}


def _axis_title(spec: dict[str, Any], default: str) -> str:
    title = spec.get("title", default)
    if isinstance(title, dict):
        title = title.get("text", default)
    return str(title)


def _axis_range_values(spec: dict[str, Any]) -> tuple[float, float] | None:
    rng = spec.get("range")
    if not isinstance(rng, list) or len(rng) < 2:
        return None
    lo = float(rng[0])
    hi = float(rng[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return lo, hi


def _compact_sci(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_sign = exponent[0] if exponent and exponent[0] in "+-" else ""
    exponent_digits = exponent[1:] if exponent_sign else exponent
    exponent_digits = exponent_digits.lstrip("0") or "0"
    return f"{mantissa}e{exponent_sign}{exponent_digits}"


def _lower_envelope_surface(traces: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return one minimum-z point per unique (x, y) mechanistic grid node."""

    source: dict[str, Any] | None = None
    for trace in traces:
        if not isinstance(trace, dict) or trace.get("type") != "scatter3d":
            continue
        name = str(trace.get("name", "")).lower()
        if name.startswith("trials with"):
            source = trace
            break
    if source is None:
        return None

    xs = np.asarray(source.get("x", []), dtype=float)
    ys = np.asarray(source.get("y", []), dtype=float)
    zs = np.asarray(source.get("z", []), dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    if not np.any(valid):
        return None

    best: dict[tuple[str, str], tuple[float, float, float]] = {}
    for x, y, z in zip(xs[valid], ys[valid], zs[valid]):
        key = (f"{float(x):.12f}", f"{float(y):.12f}")
        prev = best.get(key)
        if prev is None or float(z) < prev[2]:
            best[key] = (float(x), float(y), float(z))
    if not best:
        return None

    x_values = sorted({value[0] for value in best.values()})
    y_values = sorted({value[1] for value in best.values()})
    x_index = {f"{value:.12f}": i for i, value in enumerate(x_values)}
    y_index = {f"{value:.12f}": i for i, value in enumerate(y_values)}
    z_grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for (x_key, y_key), (_, _, z) in best.items():
        z_grid[y_index[y_key], x_index[x_key]] = float(z)

    xx, yy = np.meshgrid(np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float), indexing="xy")
    return xx, yy, z_grid


def _lower_envelope_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lower_surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if lower_surface is None:
        return np.zeros(np.asarray(x).shape, dtype=bool)
    xx, yy, zz = lower_surface
    finite = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(zz)
    if not np.any(finite):
        return np.zeros(np.asarray(x).shape, dtype=bool)
    best = {
        (f"{float(xv):.12f}", f"{float(yv):.12f}"): float(zv)
        for xv, yv, zv in zip(xx[finite], yy[finite], zz[finite])
    }
    mask = np.zeros(np.asarray(x).shape, dtype=bool)
    for idx, (xv, yv, zv) in enumerate(zip(x, y, z)):
        ref = best.get((f"{float(xv):.12f}", f"{float(yv):.12f}"))
        if ref is not None and abs(float(zv) - ref) <= 1.0e-12:
            mask[idx] = True
    return mask


def _apply_axis_ticks(ax: Any, axis: str, spec: dict[str, Any], *, z_floor_log10: float | None = None) -> None:
    tickvals = spec.get("tickvals")
    ticktext = spec.get("ticktext")
    if not isinstance(tickvals, list) or not isinstance(ticktext, list):
        return
    pairs = [
        (float(v), str(t))
        for v, t in zip(tickvals, ticktext)
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    rng = _axis_range_values(spec)
    if rng is not None:
        lo, hi = rng
        if axis == "z" and z_floor_log10 is not None and math.isfinite(z_floor_log10):
            lo = float(z_floor_log10)
        eps = 1.0e-12
        pairs = [(v, t) for v, t in pairs if lo - eps <= v <= hi + eps]
    vals = [v for v, _ in pairs]
    texts = [t for _, t in pairs]
    if axis == "z" and z_floor_log10 is not None and math.isfinite(z_floor_log10):
        low_tick = float(z_floor_log10)
        if not vals or abs(vals[0] - low_tick) > 1.0e-9:
            vals = [low_tick] + [v for v in vals if v > low_tick + 1.0e-9]
            texts = [_compact_sci(10.0 ** low_tick)] + [t for v, t in zip(vals[1:], texts) if v > low_tick + 1.0e-9]
    if axis == "x":
        ax.set_xticks(vals, labels=texts)
    elif axis == "y":
        ax.set_yticks(vals, labels=texts)
    elif axis == "z":
        ax.set_zticks(vals, labels=texts)


def _apply_axis_range(ax: Any, axis: str, spec: dict[str, Any], *, z_floor_log10: float | None = None) -> None:
    rng = spec.get("range")
    if not isinstance(rng, list) or len(rng) < 2:
        return
    lo = float(rng[0])
    hi = float(rng[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return
    if axis == "z" and z_floor_log10 is not None and math.isfinite(z_floor_log10):
        lo = float(z_floor_log10)
    if axis == "x":
        ax.set_xlim(lo, hi)
    elif axis == "y":
        ax.set_ylim(lo, hi)
    elif axis == "z":
        ax.set_zlim(lo, hi)


def render_static(
    html_path: Path,
    png_path: Path,
    pdf_path: Path | None,
    *,
    dpi: int,
    z_floor_loss: float | None,
    z_aspect: float,
    lower_envelope_alpha: float,
    lower_envelope_color: str,
    legend_fontsize: float,
    font_scale: float,
    legend_x: float,
    legend_y: float,
    fig_width: float,
    fig_height: float,
    colorbar_x: float,
    wrap_legend_labels: bool,
    compact_layout: bool,
) -> None:
    font_scale = max(float(font_scale), 0.1)
    traces, layout = _extract_plotly_json(html_path)
    scene = layout.get("scene", {})
    if not isinstance(scene, dict):
        scene = {}
    xaxis = _axis_spec(scene, "xaxis")
    yaxis = _axis_spec(scene, "yaxis")
    zaxis = _axis_spec(scene, "zaxis")

    z_floor_log10 = None
    if z_floor_loss is not None and math.isfinite(float(z_floor_loss)) and float(z_floor_loss) > 0.0:
        z_floor_log10 = math.log10(float(z_floor_loss))

    fig = plt.figure(figsize=(float(fig_width), float(fig_height)))
    main_axes_rect = [0.035, 0.11, 0.64, 0.84] if compact_layout else [0.075, 0.23, 0.65, 0.68]
    ax = fig.add_axes(main_axes_rect, projection="3d")
    colorbar_mappable = None
    colorbar_label = None
    lower_surface = _lower_envelope_surface(traces)
    ground_truth_point = None
    for trace in traces:
        if not isinstance(trace, dict) or trace.get("type") != "scatter3d":
            continue
        name = str(trace.get("name", "")).replace(" (black outline)", "")
        if name.strip().lower() != "true (ks, cs)":
            continue
        x_true = np.asarray(trace.get("x", []), dtype=float)
        y_true = np.asarray(trace.get("y", []), dtype=float)
        valid_true = np.isfinite(x_true) & np.isfinite(y_true)
        if np.any(valid_true):
            ground_truth_point = (
                float(x_true[valid_true][0]),
                float(y_true[valid_true][0]),
            )
            break

    if lower_envelope_alpha > 0.0 and lower_surface is not None:
        xx, yy, zz = lower_surface
        ax.plot_surface(
            xx,
            yy,
            zz,
            color=lower_envelope_color,
            alpha=float(lower_envelope_alpha),
            linewidth=0.0,
            antialiased=True,
            shade=False,
        )

    lower_envelope_point_count = 0
    if lower_surface is not None:
        xx, yy, zz = lower_surface
        finite_lower = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(zz)
        lower_envelope_point_count = int(np.count_nonzero(finite_lower))

        if z_floor_log10 is not None and math.isfinite(float(z_floor_log10)):
            z_grid_floor = np.full(xx.shape, float(z_floor_log10), dtype=float)
            for row_idx in range(xx.shape[0]):
                row_valid = np.isfinite(xx[row_idx, :]) & np.isfinite(yy[row_idx, :])
                if np.count_nonzero(row_valid) >= 2:
                    ax.plot(
                        xx[row_idx, row_valid],
                        yy[row_idx, row_valid],
                        z_grid_floor[row_idx, row_valid],
                        color="#1f77b4",
                        linewidth=0.55,
                        alpha=0.9,
                    )
            for col_idx in range(xx.shape[1]):
                col_valid = np.isfinite(xx[:, col_idx]) & np.isfinite(yy[:, col_idx])
                if np.count_nonzero(col_valid) >= 2:
                    ax.plot(
                        xx[col_valid, col_idx],
                        yy[col_valid, col_idx],
                        z_grid_floor[col_valid, col_idx],
                        color="#1f77b4",
                        linewidth=0.55,
                        alpha=0.9,
                    )

    for trace in traces:
        if not isinstance(trace, dict) or trace.get("type") != "scatter3d":
            continue
        x = np.asarray(trace.get("x", []), dtype=float)
        y = np.asarray(trace.get("y", []), dtype=float)
        z = np.asarray(trace.get("z", []), dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(valid):
            continue
        x = x[valid]
        y = y[valid]
        z = z[valid]
        mode = str(trace.get("mode", "markers"))
        marker = trace.get("marker", {})
        marker = marker if isinstance(marker, dict) else {}
        line = trace.get("line", {})
        line = line if isinstance(line, dict) else {}
        name = str(trace.get("name", "")).replace(" (black outline)", "")
        name_norm = name.strip().lower()
        if name_norm in {"true (ks, cs)", "ks_true", "cs_true"}:
            continue
        if "(top)" in name_norm:
            continue
        if "lowest" in name_norm:
            continue

        if "lines" in mode and len(x) >= 2:
            color = line.get("color", "#111111")
            width = float(line.get("width", 2.0))
            if z_floor_log10 is not None and name_norm in {"ks_true", "cs_true"}:
                z = np.full_like(z, float(z_floor_log10))
            ax.plot(x, y, z, color=color, linewidth=max(width, 1.0))

        if "markers" in mode:
            is_trials_trace = name_norm.startswith("trials with")
            lower_mask = _lower_envelope_mask(x, y, z, lower_surface) if is_trials_trace else np.zeros(x.shape, dtype=bool)
            draw_mask = ~lower_mask if is_trials_trace else np.ones(x.shape, dtype=bool)
            size = float(marker.get("size", 4.0))
            opacity = float(marker.get("opacity", 1.0))
            if is_trials_trace:
                opacity = min(opacity, 0.28)
            edgecolors = "none"
            linewidths = 0.0
            marker_line = marker.get("line", {})
            if isinstance(marker_line, dict) and marker_line.get("color"):
                edgecolors = marker_line.get("color", "black")
                linewidths = max(float(marker_line.get("width", 1.0)), 0.5)

            color_values = marker.get("color", "#1f77b4")
            if isinstance(color_values, list) and len(color_values) == len(trace.get("x", [])):
                raw_colors_all = np.asarray(color_values, dtype=float)[valid]
                cmap = _colorscale_to_cmap(marker.get("colorscale"))
                cmin = float(marker.get("cmin", np.nanmin(raw_colors_all)))
                cmax = float(marker.get("cmax", np.nanmax(raw_colors_all)))
                norm = Normalize(vmin=cmin, vmax=cmax)
                if np.any(draw_mask):
                    ax.scatter(
                        x[draw_mask],
                        y[draw_mask],
                        z[draw_mask],
                        c=raw_colors_all[draw_mask],
                        cmap=cmap,
                        norm=norm,
                        s=size * 7.0,
                        alpha=opacity,
                        edgecolors=edgecolors,
                        linewidths=linewidths,
                        depthshade=False,
                        rasterized=True,
                    )
                if is_trials_trace and np.any(lower_mask):
                    ax.scatter(
                        x[lower_mask],
                        y[lower_mask],
                        z[lower_mask],
                        c=raw_colors_all[lower_mask],
                        cmap=cmap,
                        norm=norm,
                        s=max(size * 18.0, 64.0),
                        alpha=1.0,
                        edgecolors="black",
                        linewidths=1.25,
                        depthshade=False,
                        rasterized=True,
                    )

                if marker.get("showscale", True) is not False and colorbar_mappable is None:
                    colorbar_mappable = ScalarMappable(norm=norm, cmap=cmap)
                    colorbar_mappable.set_array([])
                    colorbar = marker.get("colorbar", {})
                    if isinstance(colorbar, dict):
                        title = colorbar.get("title", "")
                        if isinstance(title, dict):
                            title = title.get("text", "")
                        colorbar_label = str(title)
            else:
                color = color_values if isinstance(color_values, str) else "#1f77b4"
                if np.any(draw_mask):
                    ax.scatter(
                        x[draw_mask],
                        y[draw_mask],
                        z[draw_mask],
                        c=color,
                        s=size * 7.0,
                        alpha=opacity,
                        edgecolors=edgecolors,
                        linewidths=linewidths,
                        depthshade=False,
                        rasterized=True,
                    )
                if is_trials_trace and np.any(lower_mask):
                    ax.scatter(
                        x[lower_mask],
                        y[lower_mask],
                        z[lower_mask],
                        c=color,
                        s=max(size * 18.0, 64.0),
                        alpha=1.0,
                        edgecolors="black",
                        linewidths=1.25,
                        depthshade=False,
                        rasterized=True,
                    )

    if ground_truth_point is not None:
        ground_truth_z = (
            float(z_floor_log10)
            if z_floor_log10 is not None and math.isfinite(float(z_floor_log10))
            else float(ax.get_zlim()[0])
        )
        ax.scatter(
            [ground_truth_point[0]],
            [ground_truth_point[1]],
            [ground_truth_z],
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
    ax.set_box_aspect((1.0, 1.0, max(float(z_aspect), 0.1)))
    ax.view_init(elev=28, azim=-135)

    fig.text(
        0.065 if compact_layout else 0.13,
        0.52,
        "training loss",
        rotation=90,
        fontsize=18 * font_scale,
        va="center",
        ha="center",
    )

    if colorbar_mappable is not None:
        cbar_rect = (
            [float(colorbar_x), 0.14, 0.028, 0.72]
            if compact_layout
            else [float(colorbar_x), 0.18, 0.028, 0.64]
        )
        cax = fig.add_axes(cbar_rect)
        cbar = fig.colorbar(colorbar_mappable, cax=cax)
        cbar.set_label(
            r"error between Neural Network estimation and $\bar{F}_{ts}$",
            fontsize=16 * font_scale,
        )
        cbar.ax.tick_params(labelsize=14 * font_scale)

    main_trials_label = None
    has_lower_envelope_points = lower_envelope_point_count > 0
    has_mechanistic_grid = lower_surface is not None and z_floor_log10 is not None
    for trace in traces:
        if not isinstance(trace, dict) or trace.get("type") != "scatter3d":
            continue
        name = str(trace.get("name", "")).replace(" (black outline)", "")
        name_norm = name.strip().lower()
        if name_norm.startswith("trials with"):
            main_trials_label = "all trials"

    legend_handles: list[Line2D] = []
    if main_trials_label is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=5,
                label=main_trials_label,
            )
        )
    if has_lower_envelope_points:
        legend_handles.append(
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
                label="lower-envelope\npoints" if wrap_legend_labels else "lower-envelope points",
            )
        )
    if has_mechanistic_grid:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#1f77b4",
                linewidth=1.0,
                label="mechanistic\ngrid" if wrap_legend_labels else "mechanistic grid",
            )
        )
    if ground_truth_point is not None:
        legend_handles.append(
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
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(float(legend_x), float(legend_y)),
            ncol=1,
            fontsize=float(legend_fontsize) * font_scale,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.5,
            handletextpad=0.45,
            labelspacing=0.6,
        )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    if pdf_path is not None:
        fig.savefig(pdf_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--z-floor-loss", type=float, default=None, help="Set z-axis lower bound from a loss value.")
    parser.add_argument("--z-aspect", type=float, default=2.0, help="Relative z-axis box aspect. 2.0 makes z about two times longer.")
    parser.add_argument("--lower-envelope-alpha", type=float, default=0.34, help="Lower-envelope surface transparency. Set 0 to disable.")
    parser.add_argument("--lower-envelope-color", default="#4f8fd8", help="Color for the lower-envelope surface.")
    parser.add_argument("--legend-fontsize", type=float, default=10.0, help="Legend font size.")
    parser.add_argument("--font-scale", type=float, default=1.0, help="Scale every font in the static figure.")
    parser.add_argument("--legend-x", type=float, default=0.695, help="Figure-relative legend x anchor.")
    parser.add_argument("--legend-y", type=float, default=0.53, help="Figure-relative legend y anchor.")
    parser.add_argument("--fig-width", type=float, default=12.4, help="Matplotlib figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=12.2, help="Matplotlib figure height in inches.")
    parser.add_argument("--colorbar-x", type=float, default=0.885, help="Figure-relative colorbar x position.")
    parser.add_argument("--wrap-legend-labels", action="store_true", help="Wrap long legend labels onto two lines.")
    parser.add_argument(
        "--compact-layout",
        action="store_true",
        help="Use a denser main-axis, legend, and colorbar arrangement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_path = args.html.resolve()
    png_path = args.png.resolve() if args.png else html_path.with_suffix(".png")
    pdf_path = args.pdf.resolve() if args.pdf else html_path.with_suffix(".pdf")
    render_static(
        html_path,
        png_path,
        pdf_path,
        dpi=int(args.dpi),
        z_floor_loss=args.z_floor_loss,
        z_aspect=args.z_aspect,
        lower_envelope_alpha=args.lower_envelope_alpha,
        lower_envelope_color=args.lower_envelope_color,
        legend_fontsize=args.legend_fontsize,
        font_scale=args.font_scale,
        legend_x=args.legend_x,
        legend_y=args.legend_y,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        colorbar_x=args.colorbar_x,
        wrap_legend_labels=bool(args.wrap_legend_labels),
        compact_layout=bool(args.compact_layout),
    )
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
