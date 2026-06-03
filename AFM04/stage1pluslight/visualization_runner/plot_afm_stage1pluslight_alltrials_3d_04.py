"""Interactive 3D all-trials visualization for AFM04 stage1pluslight results.

This mirrors the role of:
runner/visualization/step2a/stage1pluslight/plot_afm_stage1pluslight_alltrials_3d_03.jl

Differences:
- reads AFM04 Python-native `.pkl` results
- emits Plotly HTML without requiring the plotly Python package
- falls back to coloring by `x3_rec` when `val_nn_err` is unavailable
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import CS, KS
from AFM04.stage1pluslight.result_validation import validate_complete_stage1plus_payload


def field_or(rec: dict[str, Any], key: str, default: Any) -> Any:
    return rec.get(key, default) if isinstance(rec, dict) else default


def param_or(params: dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default) if isinstance(params, dict) else default


def finite_float_or(*values: Any) -> float:
    for value in values:
        if isinstance(value, (int, float)):
            fval = float(value)
            if math.isfinite(fval):
                return fval
    return math.nan


def empirical_quantile(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("Cannot take quantile of empty list.")
    idx = max(0, min(n - 1, math.ceil(p * n) - 1))
    return float(sorted_vals[idx])


def fmt_e_html(x: Any) -> str:
    return f"{float(x):.6e}" if isinstance(x, (int, float)) and math.isfinite(float(x)) else "None"


def fmt_f_html(x: Any, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}" if isinstance(x, (int, float)) and math.isfinite(float(x)) else "None"


def window_roles(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    params = field_or(records[0], "params", {})
    roles = param_or(params, "window_roles", [])
    return [str(role) for role in roles] if isinstance(roles, list) else []


def window_display(role: str, fallback_idx: int) -> str:
    role_norm = str(role).strip().lower()
    if role_norm == "first_contact":
        return "W0"
    if role_norm == "middle":
        return "W1"
    if role_norm == "max_x1_pp_change":
        return "W2"
    if role_norm == "tail_stable":
        return "W3"
    return f"W{fallback_idx}"


def metric_specs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roles = window_roles(records)
    if not roles:
        return [{"slug": "val_loss", "label": "val_loss", "title": "val_loss", "idx": 0, "role": "single"}]
    if len(roles) == 1:
        display = window_display(roles[0], 1)
        return [{"slug": f"{display.lower()}_val_loss", "label": f"{display} val_loss", "title": f"{display} val_loss ({roles[0]})", "idx": 1, "role": roles[0]}]
    specs: list[dict[str, Any]] = [{"slug": "mean_val_loss", "label": "mean val_loss", "title": "mean val_loss", "idx": 0, "role": "mean"}]
    for i, role in enumerate(roles, start=1):
        display = window_display(role, i)
        specs.append({"slug": f"{display.lower()}_val_loss", "label": f"{display} val_loss", "title": f"{display} val_loss ({role})", "idx": i, "role": role})
    return specs


def metric_value(rec: dict[str, Any], spec: dict[str, Any]) -> float:
    if int(spec["idx"]) > 0:
        vals = param_or(field_or(rec, "params", {}), "window_val_losses", None)
        if isinstance(vals, list) and len(vals) >= int(spec["idx"]):
            try:
                return finite_float_or(vals[int(spec["idx"]) - 1])
            except Exception:
                return math.nan
    return finite_float_or(
        field_or(rec, "val_loss", math.nan),
        field_or(rec, "val_loss_start", math.nan),
    )


def metric_train_value(rec: dict[str, Any], spec: dict[str, Any]) -> float:
    if int(spec["idx"]) > 0:
        vals = param_or(field_or(rec, "params", {}), "window_train_losses", None)
        if isinstance(vals, list) and len(vals) >= int(spec["idx"]):
            try:
                return finite_float_or(vals[int(spec["idx"]) - 1])
            except Exception:
                return math.nan
    return finite_float_or(field_or(rec, "train_loss", math.nan))


def nn_err_value(rec: dict[str, Any]) -> float:
    return finite_float_or(
        field_or(rec, "val_nn_err", math.nan),
        field_or(rec, "val_nn_err_start", math.nan),
    )


def color_metric_values(records: list[dict[str, Any]]) -> tuple[str, list[float], list[float], list[str], float]:
    nn_vals = [nn_err_value(rec) for rec in records]
    if any(math.isfinite(v) for v in nn_vals):
        cap = 200.0
        overflow = cap + 1.0
        vals = [math.nan if not math.isfinite(v) else (overflow if v > cap else max(v, 0.0)) for v in nn_vals]
        ticks = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, overflow]
        texts = ["0%", "25%", "50%", "75%", "100%", "125%", "150%", "175%", "200%", ">200%"]
        return "F_contact err %", vals, ticks, texts, overflow

    vals = []
    x3_vals = []
    for rec in records:
        parts = field_or(rec, "val_parts", {})
        x3 = parts.get("x3_rec", math.nan) if isinstance(parts, dict) else math.nan
        x3 = float(x3) if isinstance(x3, (int, float)) else math.nan
        vals.append(x3)
        if math.isfinite(x3):
            x3_vals.append(x3)
    if not x3_vals:
        x3_vals = [0.0]
    vmax = max(100.0, max(x3_vals))
    ticks = [0.0, 25.0, 50.0, 75.0, 100.0, vmax]
    texts = ["0%", "25%", "50%", "75%", "100%", f"{vmax:.1f}%"]
    return "x3 rec err %", vals, ticks, texts, vmax


def all_window_lines(rec: dict[str, Any]) -> str:
    params = field_or(rec, "params", {})
    roles = param_or(params, "window_roles", [])
    vals = param_or(params, "window_val_losses", [])
    trains = param_or(params, "window_train_losses", [])
    if not isinstance(roles, list) or not isinstance(vals, list):
        return ""
    n = min(len(roles), len(vals))
    if n == 0:
        return ""
    chunks: list[str] = []
    for i in range(n):
        role = str(roles[i])
        train_text = fmt_e_html(trains[i]) if isinstance(trains, list) and len(trains) > i else "None"
        chunks.append(f"<br>W{i+1} ({role}) val_loss={fmt_e_html(vals[i])}<br>W{i+1} ({role}) train_loss={train_text}")
    return "".join(chunks)


def hover_text(rec: dict[str, Any], spec: dict[str, Any]) -> str:
    parts = field_or(rec, "val_parts", {})
    params = field_or(rec, "params", {})
    trial_id = param_or(params, "trial_id", -1)
    ks0 = param_or(params, "ks0", field_or(rec, "ks_hat", math.nan))
    cs0 = param_or(params, "cs0", field_or(rec, "cs_hat", math.nan))
    z_val = metric_value(rec, spec)
    z_train = metric_train_value(rec, spec)
    mean_val_loss = field_or(rec, "val_loss", field_or(rec, "val_loss_start", math.nan))
    mean_train_loss = field_or(rec, "train_loss", math.nan)
    nn_err = nn_err_value(rec)
    x3_rec = parts.get("x3_rec", math.nan) if isinstance(parts, dict) else math.nan

    bits = [
        f"trial={trial_id}",
        f"<br>{spec['label']}={fmt_e_html(z_val)}",
        f"<br>{spec['label']} train={fmt_e_html(z_train)}",
        f"<br>mean val_loss={fmt_e_html(mean_val_loss)}",
        f"<br>mean train_loss={fmt_e_html(mean_train_loss)}",
        f"<br>ks0={fmt_e_html(ks0)}",
        f"<br>cs0={fmt_e_html(cs0)}",
        f"<br>ks_end={fmt_e_html(field_or(rec, 'ks_hat', math.nan))}",
        f"<br>cs_end={fmt_e_html(field_or(rec, 'cs_hat', math.nan))}",
        f"<br>log10(ks0)={fmt_f_html(math.log10(float(ks0)) if isinstance(ks0, (int, float)) and ks0 > 0 else math.nan, 6)}",
        f"<br>log10(cs0)={fmt_f_html(math.log10(float(cs0)) if isinstance(cs0, (int, float)) and cs0 > 0 else math.nan, 6)}",
        f"<br>ks err={fmt_f_html(field_or(rec, 'ks_err_pct', math.nan), 2)}%",
        f"<br>cs err={fmt_f_html(field_or(rec, 'cs_err_pct', math.nan), 2)}%",
        f"<br>F_contact err={fmt_f_html(nn_err, 2)}%",
        f"<br>x3 rec err={fmt_f_html(x3_rec, 2)}%",
    ]
    if isinstance(parts, dict):
        bits.extend(
            [
                f"<br>state={fmt_e_html(parts.get('state', math.nan))}",
                f"<br>x2dot={fmt_e_html(parts.get('x2dot', math.nan))}",
                f"<br>x1 rec err={fmt_f_html(parts.get('x1_rec', math.nan), 2)}%",
                f"<br>x3 rec err={fmt_f_html(parts.get('x3_rec', math.nan), 2)}%",
            ]
        )
    bits.append(all_window_lines(rec))
    return "".join(bits)


def sci_tick_spec(lo: float, hi: float) -> tuple[list[int], list[str]]:
    lo_exp = math.floor(math.log10(lo))
    hi_exp = math.ceil(math.log10(hi))
    vals = list(range(int(lo_exp), int(hi_exp) + 1))
    texts = [f"1e{e}" for e in vals]
    return vals, texts


def _grid_key(ks0: float, cs0: float) -> tuple[str, str]:
    return f"{ks0:.16e}", f"{cs0:.16e}"


def best_seed_surface(records: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any] | None:
    """Build the lower envelope over NN seeds for each mech grid.

    Each mech grid (ks0, cs0) has many NN seeds/trials.  The surface z-value is
    the minimum positive finite metric among those trials, using the same metric
    as the parent 3D plot.
    """

    best_by_grid: dict[tuple[str, str], dict[str, Any]] = {}
    ks_values_by_key: dict[str, float] = {}
    cs_values_by_key: dict[str, float] = {}

    for rec in records:
        params = field_or(rec, "params", {})
        ks0 = finite_float_or(param_or(params, "ks0", field_or(rec, "ks_hat", math.nan)))
        cs0 = finite_float_or(param_or(params, "cs0", field_or(rec, "cs_hat", math.nan)))
        z_val = finite_float_or(metric_value(rec, spec))
        if not (math.isfinite(ks0) and ks0 > 0.0 and math.isfinite(cs0) and cs0 > 0.0 and math.isfinite(z_val) and z_val > 0.0):
            continue

        key = _grid_key(ks0, cs0)
        ks_values_by_key[key[0]] = ks0
        cs_values_by_key[key[1]] = cs0
        prev = best_by_grid.get(key)
        if prev is None or z_val < float(prev["z_val"]):
            best_by_grid[key] = {"z_val": z_val, "record": rec}

    if not best_by_grid:
        return None

    ks_keys = sorted(ks_values_by_key, key=lambda k: ks_values_by_key[k])
    cs_keys = sorted(cs_values_by_key, key=lambda k: cs_values_by_key[k])
    x = [math.log10(ks_values_by_key[key]) for key in ks_keys]
    y = [math.log10(cs_values_by_key[key]) for key in cs_keys]
    z: list[list[float | None]] = []
    text: list[list[str]] = []

    for cs_key in cs_keys:
        z_row: list[float | None] = []
        text_row: list[str] = []
        for ks_key in ks_keys:
            best = best_by_grid.get((ks_key, cs_key))
            if best is None:
                z_row.append(None)
                text_row.append("")
                continue
            rec = best["record"]
            params = field_or(rec, "params", {})
            trial_id = param_or(params, "trial_id", -1)
            ks0 = ks_values_by_key[ks_key]
            cs0 = cs_values_by_key[cs_key]
            loss = float(best["z_val"])
            z_row.append(math.log10(loss))
            text_row.append(
                "".join(
                    [
                        "best seed envelope",
                        f"<br>trial={trial_id}",
                        f"<br>{spec['label']}={fmt_e_html(loss)}",
                        f"<br>ks0={fmt_e_html(ks0)}",
                        f"<br>cs0={fmt_e_html(cs0)}",
                        f"<br>log10(ks0)={fmt_f_html(math.log10(ks0), 6)}",
                        f"<br>log10(cs0)={fmt_f_html(math.log10(cs0), 6)}",
                    ]
                )
            )
        z.append(z_row)
        text.append(text_row)

    return {
        "type": "surface",
        "name": "best NN seed envelope",
        "x": x,
        "y": y,
        "z": z,
        "text": text,
        "hovertemplate": "%{text}<extra></extra>",
        "opacity": 0.38,
        "connectgaps": False,
        "showscale": False,
        "colorscale": [
            [0.00, "#d9f0ff"],
            [0.35, "#74add1"],
            [0.70, "#2b83ba"],
            [1.00, "#08306b"],
        ],
        "contours": {
            "z": {
                "show": True,
                "usecolormap": True,
                "highlightcolor": "#111111",
                "project": {"z": True},
            }
        },
    }


def write_plot_html(
    html_path: Path,
    result_path: Path,
    spec: dict[str, Any],
    kept_records: list[dict[str, Any]],
    true_ks: float,
    true_cs: float,
    excluded_count: int,
    z_display_max: float,
) -> None:
    xs_raw = [float(param_or(field_or(rec, "params", {}), "ks0", field_or(rec, "ks_hat", math.nan))) for rec in kept_records]
    ys_raw = [float(param_or(field_or(rec, "params", {}), "cs0", field_or(rec, "cs_hat", math.nan))) for rec in kept_records]
    zs_raw = [float(metric_value(rec, spec)) for rec in kept_records]
    hovertexts = [hover_text(rec, spec) for rec in kept_records]

    color_title, colors, color_tickvals, color_ticktext, color_cmax = color_metric_values(kept_records)

    xs = [math.log10(x) for x in xs_raw]
    ys = [math.log10(y) for y in ys_raw]
    zs = [math.log10(z) for z in zs_raw]

    highlight_n = min(len(kept_records), 100)
    highlight_order = sorted(range(len(zs_raw)), key=lambda i: zs_raw[i])[:highlight_n]
    highlight_mask = [i in set(highlight_order) for i in range(len(kept_records))]

    xs_base = [x for x, h in zip(xs, highlight_mask) if not h]
    ys_base = [y for y, h in zip(ys, highlight_mask) if not h]
    zs_base = [z for z, h in zip(zs, highlight_mask) if not h]
    colors_base = [c for c, h in zip(colors, highlight_mask) if not h]
    hovertexts_base = [htext for htext, h in zip(hovertexts, highlight_mask) if not h]
    xs_high = [x for x, h in zip(xs, highlight_mask) if h]
    ys_high = [y for y, h in zip(ys, highlight_mask) if h]
    zs_high = [z for z, h in zip(zs, highlight_mask) if h]
    colors_high = [c for c, h in zip(colors, highlight_mask) if h]
    hovertexts_high = [htext for htext, h in zip(hovertexts, highlight_mask) if h]

    sorted_zs_raw = sorted(zs_raw)
    zfocus_lo_raw = empirical_quantile(sorted_zs_raw, 0.01)
    zfocus_hi_raw = empirical_quantile(sorted_zs_raw, 0.995)
    zfocus_lo = math.log10(zfocus_lo_raw)
    zfocus_hi = math.log10(zfocus_hi_raw)
    zpad = max((zfocus_hi - zfocus_lo) * 0.12, 0.015)
    highlight_lo_raw = zfocus_lo_raw if not zs_high else 10.0 ** min(zs_high)
    highlight_lo = math.log10(highlight_lo_raw)
    highlight_pad = max((zfocus_lo - highlight_lo) * 0.08, 0.01)
    zfloor = min(zfocus_lo - zpad, highlight_lo - highlight_pad)
    ztop = zfocus_hi + zpad
    z_hidden_low = sum(1 for z in zs_raw if z < zfocus_lo_raw)
    z_hidden_high = sum(1 for z in zs_raw if z > zfocus_hi_raw)

    truth_x = math.log10(true_ks) if math.isfinite(true_ks) and true_ks > 0 else math.nan
    truth_y = math.log10(true_cs) if math.isfinite(true_cs) and true_cs > 0 else math.nan
    truth_z = zfloor

    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xtickvals, xticktext = sci_tick_spec(min(xs_raw), max(xs_raw))
    ytickvals, yticktext = sci_tick_spec(min(ys_raw), max(ys_raw))
    ztickvals, zticktext = sci_tick_spec(zfocus_lo_raw, zfocus_hi_raw)

    annotation_text = (
        f"source: {str(result_path).replace(chr(92), chr(92)*2)}"
        f"<br>x=ks0 (log10), y=cs0 (log10), z={spec['label']} (log10)"
        f"<br>color uses {color_title}; excluded {excluded_count} trials with {spec['label']} > {z_display_max:.1e} or <= 0;"
        f" z-focus uses q01..q99.5 and hides {z_hidden_low} low / {z_hidden_high} high outliers from the visible z span"
    )

    traces = []
    surface = best_seed_surface(kept_records, spec)
    if surface is not None:
        traces.append(surface)

    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": f"trials with {spec['label']} <= {z_display_max:.1e}",
            "x": xs_base,
            "y": ys_base,
            "z": zs_base,
            "hovertext": hovertexts_base,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 3,
                "opacity": 0.82,
                "color": colors_base,
                "cmin": 0.0,
                "cmax": color_cmax,
                "colorscale": [
                    [0.0000, "#4b00ff"],
                    [0.1110, "#0057ff"],
                    [0.2220, "#00b7ff"],
                    [0.3330, "#00ffd0"],
                    [0.4440, "#2cff5c"],
                    [0.5550, "#b7ff00"],
                    [0.6660, "#ffe600"],
                    [0.7770, "#ff9a00"],
                    [0.8880, "#ff3b00"],
                    [0.9950, "#c40000"],
                    [0.9951, "#8c8c8c"],
                    [1.0000, "#8c8c8c"],
                ],
                "colorbar": {
                    "title": color_title,
                    "tickmode": "array",
                    "tickvals": color_tickvals,
                    "ticktext": color_ticktext,
                },
                "line": {"width": 0},
            },
        }
    )
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": f"lowest {highlight_n} {spec['label']} trials (black outline)",
            "x": xs_high,
            "y": ys_high,
            "z": zs_high,
            "hovertext": hovertexts_high,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 5,
                "opacity": 1.0,
                "color": colors_high,
                "cmin": 0.0,
                "cmax": color_cmax,
                "colorscale": [
                    [0.0000, "#4b00ff"],
                    [0.1110, "#0057ff"],
                    [0.2220, "#00b7ff"],
                    [0.3330, "#00ffd0"],
                    [0.4440, "#2cff5c"],
                    [0.5550, "#b7ff00"],
                    [0.6660, "#ffe600"],
                    [0.7770, "#ff9a00"],
                    [0.8880, "#ff3b00"],
                    [0.9950, "#c40000"],
                    [0.9951, "#8c8c8c"],
                    [1.0000, "#8c8c8c"],
                ],
                "showscale": False,
                "line": {"color": "#000000", "width": 4},
            },
        }
    )
    if math.isfinite(truth_x) and math.isfinite(truth_y):
        traces.extend(
            [
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "name": "ks_true",
                    "x": [truth_x, truth_x],
                    "y": [ylo, yhi],
                    "z": [zfloor, zfloor],
                    "hovertemplate": f"ks_true={true_ks:.6e}<extra></extra>",
                    "line": {"color": "#111111", "width": 2},
                },
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "name": "ks_true (top)",
                    "showlegend": False,
                    "x": [truth_x, truth_x],
                    "y": [ylo, yhi],
                    "z": [ztop, ztop],
                    "hovertemplate": f"ks_true={true_ks:.6e}<extra></extra>",
                    "line": {"color": "#111111", "width": 2},
                },
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "name": "cs_true",
                    "x": [xlo, xhi],
                    "y": [truth_y, truth_y],
                    "z": [zfloor, zfloor],
                    "hovertemplate": f"cs_true={true_cs:.6e}<extra></extra>",
                    "line": {"color": "#111111", "width": 2},
                },
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "name": "cs_true (top)",
                    "showlegend": False,
                    "x": [xlo, xhi],
                    "y": [truth_y, truth_y],
                    "z": [ztop, ztop],
                    "hovertemplate": f"cs_true={true_cs:.6e}<extra></extra>",
                    "line": {"color": "#111111", "width": 2},
                },
                {
                    "type": "scatter3d",
                    "mode": "markers+text",
                    "name": "true (ks, cs)",
                    "x": [truth_x],
                    "y": [truth_y],
                    "z": [truth_z],
                    "text": ["truth"],
                    "textposition": "bottom center",
                    "hovertemplate": f"true ks={true_ks:.6e}<br>true cs={true_cs:.6e}<extra></extra>",
                    "marker": {"size": 7, "symbol": "diamond", "color": "#111111"},
                },
            ]
        )

    layout = {
        "title": f"AFM04 Stage1pluslight trials: ks0 vs cs0 vs {spec['label']}",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "legend": {"orientation": "h", "x": 0, "y": 1.02},
        "margin": {"l": 0, "r": 0, "t": 70, "b": 0},
        "annotations": [
            {
                "text": annotation_text,
                "x": 0,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"size": 12, "color": "#444444"},
            }
        ],
        "scene": {
            "xaxis": {"title": "ks", "tickmode": "array", "tickvals": xtickvals, "ticktext": xticktext, "range": [xlo, xhi]},
            "yaxis": {"title": "cs", "tickmode": "array", "tickvals": ytickvals, "ticktext": yticktext, "range": [ylo, yhi]},
            "zaxis": {"title": spec["label"], "tickmode": "array", "tickvals": ztickvals, "ticktext": zticktext, "range": [zfloor, ztop]},
            "camera": {"eye": {"x": 1.5, "y": 1.32, "z": 0.92}},
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AFM04 Stage1pluslight all-trials log(ks0)-log(cs0)-{spec['label']} 3D</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #ffffff; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const traces = {json.dumps(traces, ensure_ascii=False)};
    const layout = {json.dumps(layout, ensure_ascii=False)};
    Plotly.newPlot('plot', traces, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")


def load_payload(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(data)!r}")
    return data


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    default_result_path = Path(__file__).resolve().parents[1] / "results_afm" / "afm_param_stage1pluslight_04.pkl"
    result_path = Path(argv[0]).resolve() if argv else default_result_path.resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing AFM04 Stage1pluslight result file: {result_path}")

    default_out_dir = Path(__file__).resolve().parents[1] / "visualization"
    out_dir = Path(argv[1]).resolve() if len(argv) >= 2 else default_out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_payload(result_path)
    if "trial_parameters" not in payload:
        raise KeyError(f"Result file does not contain 'trial_parameters': {result_path}")
    validate_complete_stage1plus_payload(payload, path=result_path)

    all_records = [rec for rec in payload["trial_parameters"] if isinstance(rec, dict)]
    records = [
        rec for rec in all_records
        if not bool(field_or(rec, "early_stopped", False))
        and not bool(field_or(rec, "trial_failed", False))
        and bool(field_or(rec, "is_viable", True))
    ]
    if not records:
        raise RuntimeError(f"No viable trial records found in: {result_path}")

    z_display_max = 1e-1
    true_ks = float(KS)
    true_cs = float(CS)
    specs = metric_specs(records)
    stem = result_path.stem
    saved: list[Path] = []

    for spec in specs:
        kept_records = [
            rec for rec in records
            if math.isfinite(metric_value(rec, spec))
            and metric_value(rec, spec) > 0.0
            and metric_value(rec, spec) <= z_display_max
        ]
        excluded_count = len(records) - len(kept_records)
        if not kept_records:
            raise RuntimeError(f"No trial records remain after z cutoff for {spec['label']}.")

        html_path = out_dir / f"{stem}_alltrials_ks_logcs_{spec['slug']}_3d.html"
        write_plot_html(html_path, result_path, spec, kept_records, true_ks, true_cs, excluded_count, z_display_max)
        saved.append(html_path)
        print(f"Saved 3D HTML to: {html_path}")
        print(f"Trials plotted for {spec['label']}: {len(kept_records)}")
        print(f"Trials excluded by z cutoff for {spec['label']}: {excluded_count}")

    print(f"Saved {len(saved)} AFM04 Stage1pluslight 3D plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
