"""Interactive 3D all-trials visualization for AFM05 stage1pluslight results.

This mirrors the AFM04 script:
AFM04/stage1pluslight/visualization_runner/plot_afm_stage1pluslight_alltrials_3d_04.py

AFM05 differences:
- reads AFM05 Python-native `.pkl` results
- uses only AFM05-legal diagnostics in hover/color fields
- does not draw true ks/cs lines because AFM05 has no truth side
- keeps x3_norm_KAN support diagnostics for checking KAN grid coverage
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM05.stage1pluslight.result_validation import validate_complete_stage1plus_payload


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


def fmt_f_html(x: Any, digits: int = 4) -> str:
    return f"{float(x):.{digits}f}" if isinstance(x, (int, float)) and math.isfinite(float(x)) else "None"


def metric_specs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"slug": "train_loss", "label": "train_loss", "title": "train_loss"}]


def metric_value(rec: dict[str, Any], spec: dict[str, Any]) -> float:
    return finite_float_or(field_or(rec, "train_loss", math.nan), field_or(rec, "loss", math.nan))


def parts_or(rec: dict[str, Any]) -> dict[str, Any]:
    parts = field_or(rec, "val_parts", {})
    if isinstance(parts, dict) and parts:
        return parts
    train_parts = field_or(rec, "train_parts", {})
    return train_parts if isinstance(train_parts, dict) else {}


def part_float(rec: dict[str, Any], key: str) -> float:
    return finite_float_or(parts_or(rec).get(key, math.nan))


def color_metric_values(records: list[dict[str, Any]]) -> tuple[str, list[float], list[float], list[str], float]:
    """Use AFM05-observable reconstruction error as the color metric.

    AFM05 has no true-side Fts/x3 metrics.  The cleanest legal color field is
    therefore an observable-side reconstruction diagnostic.  x1_rec is used
    first because it is the primary tip displacement guard/diagnostic; x2_rec
    and x2dot_rec remain visible in hover text.
    """

    candidates = (
        ("x1_rec", "x1 rec err %"),
        ("x2_rec", "x2 rec err %"),
        ("x2dot_rec", "x2dot rec err %"),
    )
    for key, title in candidates:
        vals = [part_float(rec, key) for rec in records]
        finite_vals = sorted(v for v in vals if math.isfinite(v))
        if finite_vals:
            cap = max(1.0e-6, empirical_quantile(finite_vals, 0.995))
            overflow = cap * 1.02 if cap > 0.0 else 1.0
            plot_vals = [
                math.nan if not math.isfinite(v) else (overflow if v > cap else max(v, 0.0))
                for v in vals
            ]
            ticks = [0.0, 0.25 * cap, 0.5 * cap, 0.75 * cap, cap, overflow]
            texts = [
                "0%",
                f"{0.25 * cap:.4g}%",
                f"{0.5 * cap:.4g}%",
                f"{0.75 * cap:.4g}%",
                f"{cap:.4g}%",
                f">{cap:.4g}%",
            ]
            return title, plot_vals, ticks, texts, overflow

    vals = [metric_value(rec, {"slug": "train_loss", "label": "train_loss"}) for rec in records]
    finite_vals = sorted(v for v in vals if math.isfinite(v) and v > 0.0)
    if not finite_vals:
        vals = [0.0 for _ in records]
        return "color metric unavailable", vals, [0.0], ["0"], 1.0
    log_vals = [math.log10(v) if math.isfinite(v) and v > 0.0 else math.nan for v in vals]
    lo = min(v for v in log_vals if math.isfinite(v))
    hi = max(v for v in log_vals if math.isfinite(v))
    return "log10(train_loss)", log_vals, [lo, hi], [f"{lo:.2f}", f"{hi:.2f}"], hi


def hover_text(rec: dict[str, Any], spec: dict[str, Any]) -> str:
    parts = parts_or(rec)
    params = field_or(rec, "params", {})
    trial_id = param_or(params, "trial_id", -1)
    ks0 = param_or(params, "ks0", field_or(rec, "ks_hat", math.nan))
    cs0 = param_or(params, "cs0", field_or(rec, "cs_hat", math.nan))
    train_loss = metric_value(rec, spec)

    bits = [
        f"trial={trial_id}",
        f"<br>node={param_or(params, 'node_label', 'None')}",
        f"<br>seed_bank_idx={fmt_f_html(param_or(params, 'nn_seed_bank_idx', math.nan), 0)}",
        f"<br>nn_init_seed={fmt_f_html(param_or(params, 'nn_init_seed', math.nan), 0)}",
        f"<br>train_loss={fmt_e_html(train_loss)}",
        f"<br>ks0={fmt_e_html(ks0)} N/m",
        f"<br>cs0={fmt_e_html(cs0)} N*s/m",
        f"<br>ks_node_idx={fmt_f_html(param_or(params, 'ks_node_idx', math.nan), 0)}",
        f"<br>cs_node_idx={fmt_f_html(param_or(params, 'cs_node_idx', math.nan), 0)}",
        f"<br>log10(ks0)={fmt_f_html(math.log10(float(ks0)) if isinstance(ks0, (int, float)) and ks0 > 0 else math.nan, 6)}",
        f"<br>log10(cs0)={fmt_f_html(math.log10(float(cs0)) if isinstance(cs0, (int, float)) and cs0 > 0 else math.nan, 6)}",
        f"<br>nn_gain={fmt_e_html(field_or(rec, 'nn_gain', math.nan))}",
        f"<br>gain_ref_q95_abs={fmt_e_html(param_or(params, 'kan_gain_init_force_reference_q95_abs', math.nan))}",
        f"<br>KAN grid={fmt_f_html(param_or(params, 'kan_grid', math.nan), 0)}",
        f"<br>KAN spline_k={fmt_f_html(param_or(params, 'kan_spline_k', math.nan), 0)}",
        f"<br>KAN base={param_or(params, 'kan_base_fun', 'None')}",
        f"<br>x1 rec err={fmt_f_html(parts.get('x1_rec', math.nan), 5)}%",
        f"<br>x2 rec err={fmt_f_html(parts.get('x2_rec', math.nan), 5)}%",
        f"<br>x2dot rec err={fmt_f_html(parts.get('x2dot_rec', math.nan), 5)}%",
        f"<br>state={fmt_e_html(parts.get('state', math.nan))}",
        f"<br>x1_state={fmt_e_html(parts.get('x1_state', math.nan))}",
        f"<br>x2_state={fmt_e_html(parts.get('x2_state', math.nan))}",
        f"<br>x2dot={fmt_e_html(parts.get('x2dot', math.nan))}",
        f"<br>x3_range={fmt_e_html(parts.get('x3_range', math.nan))}",
        f"<br>fts_range={fmt_e_html(parts.get('fts_range', math.nan))}",
        f"<br>x3_norm_KAN_min={fmt_f_html(param_or(params, 'x3_norm_KAN_min', math.nan), 5)}",
        f"<br>x3_norm_KAN_max={fmt_f_html(param_or(params, 'x3_norm_KAN_max', math.nan), 5)}",
        f"<br>outside |x3n|>1={fmt_f_html(param_or(params, 'x3_norm_KAN_outside_1_frac', math.nan), 5)}",
        f"<br>outside |x3n|>2={fmt_f_html(param_or(params, 'x3_norm_KAN_outside_2_frac', math.nan), 5)}",
        f"<br>window={param_or(params, 'stage1plus_window_mode', 'None')}",
        f"<br>pixel={param_or(params, 'stage1plus_window_pixel_tag', 'None')}",
        f"<br>sample_stride={fmt_f_html(param_or(params, 'stage1plus_window_sample_stride', math.nan), 0)}",
        f"<br>AFM05_true_side_available={param_or(params, 'AFM05_true_side_available', False)}",
    ]
    return "".join(bits)


def sci_tick_spec(lo: float, hi: float) -> tuple[list[int], list[str]]:
    lo_exp = math.floor(math.log10(lo))
    hi_exp = math.ceil(math.log10(hi))
    vals = list(range(int(lo_exp), int(hi_exp) + 1))
    texts = [f"1e{e}" for e in vals]
    return vals, texts


def _grid_key(ks0: float, cs0: float) -> tuple[str, str]:
    return f"{ks0:.16e}", f"{cs0:.16e}"


def _stage1_ridge_separator_points(
    log_ks_values: list[float],
    log_cs_values: list[float],
    log_loss_values: list[float],
) -> list[tuple[float, float, float]]:
    """Find the local high-loss ridge separating the corner and valley basins."""

    columns: dict[float, list[tuple[float, float]]] = {}
    for log_ks, log_cs, log_loss in zip(log_ks_values, log_cs_values, log_loss_values):
        if not (math.isfinite(log_ks) and math.isfinite(log_cs) and math.isfinite(log_loss)):
            continue
        columns.setdefault(float(log_ks), []).append((float(log_cs), float(log_loss)))

    ridge: list[tuple[float, float, float]] = []
    for log_ks in sorted(columns):
        column = sorted(columns[log_ks], key=lambda item: item[0])
        if len(column) < 4:
            continue
        losses = [item[1] for item in column]
        local_minima = [
            idx
            for idx in range(1, len(column) - 1)
            if losses[idx] <= losses[idx - 1] and losses[idx] <= losses[idx + 1]
        ]
        if not local_minima:
            continue

        # Use the valley closest to the high-cs side, then find the barrier to
        # the high-cs boundary. Once the barrier reaches the boundary, the
        # corner/valley split no longer continues inside the grid.
        valley_idx = max(local_minima)
        if valley_idx >= len(column) - 1:
            continue
        ridge_idx = max(range(valley_idx, len(column)), key=lambda idx: losses[idx])
        if ridge_idx <= valley_idx:
            continue
        ridge.append((log_ks, column[ridge_idx][0], column[ridge_idx][1]))
        if ridge_idx == len(column) - 1:
            break
    return ridge


def _separator_y_at_log_ks(log_ks: float, separator_points: list[tuple[float, float, float]]) -> float | None:
    if not separator_points:
        return None
    points = sorted(separator_points, key=lambda item: item[0])
    x = float(log_ks)
    if x > points[-1][0]:
        return None
    if x <= points[0][0]:
        return points[0][1]
    for left, right in zip(points, points[1:]):
        x0, y0, _ = left
        x1, y1, _ = right
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            alpha = (x - x0) / (x1 - x0)
            return y0 + alpha * (y1 - y0)
    return points[-1][1]


def _prest2_zone_from_log_coords(
    log_ks: float,
    log_cs: float,
    separator_points: list[tuple[float, float, float]],
) -> str:
    separator_y = _separator_y_at_log_ks(float(log_ks), separator_points)
    if separator_y is None:
        return "valley"
    return "corner" if float(log_cs) >= separator_y else "valley"


def _node_key_from_label(label: Any) -> tuple[int, int] | None:
    if not isinstance(label, str) or not label.startswith("node_") or "*" not in label:
        return None
    left, right = label[5:].split("*", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def _node_key_from_record(rec: dict[str, Any]) -> tuple[int, int] | None:
    params = field_or(rec, "params", {})
    for container in (rec, params):
        if not isinstance(container, dict):
            continue
        for ks_key, cs_key in (("ks_node_idx", "cs_node_idx"), ("source_ks_node_idx", "source_cs_node_idx")):
            ks_idx = container.get(ks_key)
            cs_idx = container.get(cs_key)
            if ks_idx is None or cs_idx is None:
                continue
            try:
                return int(ks_idx), int(cs_idx)
            except (TypeError, ValueError):
                continue
    return _node_key_from_label(
        param_or(
            params,
            "node_label",
            field_or(rec, "node_label", field_or(rec, "source_mech_node_label", "")),
        )
    )


def load_current_prest2_layer_a_nodes() -> set[tuple[int, int]]:
    prest2_layer_a_path = (
        Path(__file__).resolve().parents[3]
        / "prestage2"
        / "results"
        / "afm_prest2_05_top_mech_winners_a.pkl"
    )
    if not prest2_layer_a_path.is_file():
        return set()

    with open(prest2_layer_a_path, "rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict):
        records = (
            payload.get("top_mech_winner_a_records")
            or payload.get("candidate_records")
            or payload.get("newrank_a_records")
            or []
        )
    elif isinstance(payload, list):
        records = payload
    else:
        records = []

    nodes: set[tuple[int, int]] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        node_key = _node_key_from_record(rec)
        if node_key is not None:
            nodes.add(node_key)
    return nodes


def select_top200_valley_nodes_from_stage1(records: list[dict[str, Any]]) -> set[tuple[int, int]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        params = field_or(rec, "params", {})
        try:
            ks_node = int(param_or(params, "ks_node_idx", 0))
            cs_node = int(param_or(params, "cs_node_idx", 0))
        except (TypeError, ValueError):
            continue
        if ks_node <= 0 or cs_node <= 0:
            continue
        groups.setdefault((ks_node, cs_node), []).append(rec)

    rows: list[dict[str, Any]] = []
    for node, node_records in groups.items():
        finite_records = [rec for rec in node_records if math.isfinite(finite_float_or(field_or(rec, "loss", math.nan)))]
        viable_records = [
            rec
            for rec in node_records
            if bool(field_or(rec, "is_viable", False))
            and math.isfinite(finite_float_or(field_or(rec, "loss", math.nan)))
        ]
        losses = [finite_float_or(field_or(rec, "loss", math.nan)) for rec in viable_records]
        first = min(
            node_records,
            key=lambda rec: int(param_or(field_or(rec, "params", {}), "trial_id", 0)),
        )
        first_params = field_or(first, "params", {})
        best = min(
            finite_records,
            key=lambda rec: (
                finite_float_or(field_or(rec, "loss", float("inf"))),
                int(param_or(field_or(rec, "params", {}), "trial_id", 0)),
            ),
        ) if finite_records else first
        ks_value = finite_float_or(field_or(first, "ks_hat", math.nan), param_or(first_params, "ks0", math.nan))
        cs_value = finite_float_or(field_or(first, "cs_hat", math.nan), param_or(first_params, "cs0", math.nan))
        if not (math.isfinite(ks_value) and ks_value > 0.0 and math.isfinite(cs_value) and cs_value > 0.0):
            continue
        rows.append(
            {
                "node": node,
                "ks": ks_value,
                "cs": cs_value,
                "mean_loss_viable": sum(losses) / len(losses) if losses else float("inf"),
                "viable_rate": len(viable_records) / len(node_records) if node_records else 0.0,
                "best_loss": finite_float_or(field_or(best, "loss", float("inf"))),
            }
        )

    separator_points = _stage1_ridge_separator_points(
        [math.log10(float(row["ks"])) for row in rows],
        [math.log10(float(row["cs"])) for row in rows],
        [
            math.log10(float(row["best_loss"]))
            if math.isfinite(float(row["best_loss"])) and float(row["best_loss"]) > 0.0
            else math.nan
            for row in rows
        ],
    )
    for row in rows:
        row["zone"] = _prest2_zone_from_log_coords(
            math.log10(float(row["ks"])),
            math.log10(float(row["cs"])),
            separator_points,
        )

    rows.sort(
        key=lambda row: (
            finite_float_or(row.get("mean_loss_viable", float("inf"))),
            -finite_float_or(row.get("viable_rate", 0.0)),
            finite_float_or(row.get("best_loss", float("inf"))),
            int(row["node"][0]),
            int(row["node"][1]),
        )
    )
    return {row["node"] for row in rows[:200] if row.get("zone") == "valley"}


def best_seed_envelope(records: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any] | None:
    """Build the lower envelope over NN seeds for each mechanistic grid point."""

    best_by_grid: dict[tuple[str, str], dict[str, Any]] = {}
    ks_values_by_key: dict[str, float] = {}
    cs_values_by_key: dict[str, float] = {}

    for rec in records:
        params = field_or(rec, "params", {})
        ks0 = finite_float_or(param_or(params, "ks0", field_or(rec, "ks_hat", math.nan)))
        cs0 = finite_float_or(param_or(params, "cs0", field_or(rec, "cs_hat", math.nan)))
        z_val = finite_float_or(metric_value(rec, spec))
        if not (
            math.isfinite(ks0)
            and ks0 > 0.0
            and math.isfinite(cs0)
            and cs0 > 0.0
            and math.isfinite(z_val)
            and z_val > 0.0
        ):
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
    point_x: list[float] = []
    point_y: list[float] = []
    point_z: list[float] = []
    point_text: list[str] = []
    point_nodes: list[tuple[int, int] | None] = []

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
            log_loss = math.log10(loss)
            z_row.append(log_loss)
            best_text = (
                "".join(
                    [
                        "best seed envelope",
                        f"<br>trial={trial_id}",
                        f"<br>{spec['label']}={fmt_e_html(loss)}",
                        f"<br>ks0={fmt_e_html(ks0)} N/m",
                        f"<br>cs0={fmt_e_html(cs0)} N*s/m",
                        f"<br>node={param_or(params, 'node_label', 'None')}",
                        f"<br>seed_bank_idx={fmt_f_html(param_or(params, 'nn_seed_bank_idx', math.nan), 0)}",
                        f"<br>x3n=[{fmt_f_html(param_or(params, 'x3_norm_KAN_min', math.nan), 5)},"
                        f"{fmt_f_html(param_or(params, 'x3_norm_KAN_max', math.nan), 5)}]",
                    ]
                )
            )
            text_row.append(best_text)
            point_x.append(math.log10(ks0))
            point_y.append(math.log10(cs0))
            point_z.append(log_loss)
            point_text.append(best_text)
            point_nodes.append(_node_key_from_record(rec))
        z.append(z_row)
        text.append(text_row)

    return {
        "surface": {
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
        },
        "point_x": point_x,
        "point_y": point_y,
        "point_z": point_z,
        "point_text": point_text,
        "point_nodes": point_nodes,
        "ks_raw": [ks_values_by_key[key] for key in ks_keys],
        "cs_raw": [cs_values_by_key[key] for key in cs_keys],
    }


def best_seed_surface(records: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any] | None:
    envelope = best_seed_envelope(records, spec)
    return None if envelope is None else envelope["surface"]


def write_plot_html(
    html_path: Path,
    result_path: Path,
    spec: dict[str, Any],
    kept_records: list[dict[str, Any]],
    excluded_count: int,
    z_display_max: float,
) -> None:
    xs_raw = [finite_float_or(param_or(field_or(rec, "params", {}), "ks0", field_or(rec, "ks_hat", math.nan))) for rec in kept_records]
    ys_raw = [finite_float_or(param_or(field_or(rec, "params", {}), "cs0", field_or(rec, "cs_hat", math.nan))) for rec in kept_records]
    zs_raw = [float(metric_value(rec, spec)) for rec in kept_records]
    hovertexts = [hover_text(rec, spec) for rec in kept_records]

    xs = [math.log10(x) for x in xs_raw]
    ys = [math.log10(y) for y in ys_raw]
    zs = [math.log10(z) for z in zs_raw]

    sorted_zs_raw = sorted(zs_raw)
    zmin_raw = sorted_zs_raw[0]
    zfocus_lo_raw = empirical_quantile(sorted_zs_raw, 0.01)
    zfocus_hi_raw = empirical_quantile(sorted_zs_raw, 0.995)
    zmin = math.log10(zmin_raw)
    zfocus_lo = math.log10(zfocus_lo_raw)
    zfocus_hi = math.log10(zfocus_hi_raw)
    zpad = max((zfocus_hi - zfocus_lo) * 0.12, 0.015)
    zfloor = zmin - max((zfocus_hi - zmin) * 0.05, 0.015)
    ztop = zfocus_hi + zpad
    z_hidden_low = 0
    z_hidden_high = sum(1 for z in zs_raw if z > zfocus_hi_raw)

    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xtickvals, xticktext = sci_tick_spec(min(xs_raw), max(xs_raw))
    ytickvals, yticktext = sci_tick_spec(min(ys_raw), max(ys_raw))
    ztickvals, zticktext = sci_tick_spec(zfocus_lo_raw, zfocus_hi_raw)

    annotation_text = (
        f"source: {str(result_path).replace(chr(92), chr(92) * 2)}"
        f"<br>x=ks0 (log10), y=cs0 (log10), z={spec['label']} (log10)"
        f"<br>all trial markers are black; AFM05 truth side is disabled;"
        f" excluded {excluded_count} trials with {spec['label']} > {z_display_max:.1e} or <= 0;"
        f" z-focus uses min..q99.5 and hides {z_hidden_low} low / {z_hidden_high} high outliers from the visible z span"
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
            "x": xs,
            "y": ys,
            "z": zs,
            "hovertext": hovertexts,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 3,
                "opacity": 0.82,
                "color": "#000000",
                "line": {"width": 0},
            },
        }
    )

    layout = {
        "title": f"AFM05 Stage1pluslight trials: ks0 vs cs0 vs {spec['label']}",
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
            "xaxis": {
                "title": "ks [N/m]",
                "tickmode": "array",
                "tickvals": xtickvals,
                "ticktext": xticktext,
                "range": [xlo, xhi],
            },
            "yaxis": {
                "title": "cs [N*s/m]",
                "tickmode": "array",
                "tickvals": ytickvals,
                "ticktext": yticktext,
                "range": [ylo, yhi],
            },
            "zaxis": {
                "title": spec["label"],
                "tickmode": "array",
                "tickvals": ztickvals,
                "ticktext": zticktext,
                "range": [zfloor, ztop],
            },
            "camera": {"eye": {"x": 1.5, "y": 1.32, "z": 0.92}},
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AFM05 Stage1pluslight all-trials log(ks0)-log(cs0)-{spec['label']} 3D</title>
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


def write_envelope_points_html(
    html_path: Path,
    result_path: Path,
    spec: dict[str, Any],
    kept_records: list[dict[str, Any]],
    selection_records: list[dict[str, Any]],
) -> None:
    envelope = best_seed_envelope(kept_records, spec)
    if envelope is None:
        raise RuntimeError(f"No best-seed envelope points are available for {spec['label']}.")

    xs = list(envelope["point_x"])
    ys = list(envelope["point_y"])
    zs = list(envelope["point_z"])
    hovertexts = list(envelope["point_text"])
    point_nodes = list(envelope.get("point_nodes", []))
    if not xs:
        raise RuntimeError(f"Best-seed envelope is empty for {spec['label']}.")

    ks_raw = list(envelope["ks_raw"])
    cs_raw = list(envelope["cs_raw"])
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    zlo, zhi = min(zs), max(zs)
    zpad = max((zhi - zlo) * 0.12, 0.015)
    zfloor = zlo - zpad
    ztop = zhi + zpad

    xtickvals, xticktext = sci_tick_spec(min(ks_raw), max(ks_raw))
    ytickvals, yticktext = sci_tick_spec(min(cs_raw), max(cs_raw))
    ztickvals, zticktext = sci_tick_spec(10.0**zlo, 10.0**zhi)

    separator_points = _stage1_ridge_separator_points(xs, ys, zs)
    non_corner_idx = [
        idx
        for idx, (x, y) in enumerate(zip(xs, ys))
        if _prest2_zone_from_log_coords(float(x), float(y), separator_points) != "corner"
    ]
    red_n = min(100, len(non_corner_idx))
    red_idx = set(sorted(non_corner_idx, key=lambda idx: zs[idx])[:red_n])
    red_mask = [idx in red_idx for idx in range(len(xs))]
    red_label = f"lowest {red_n} non-corner envelope points"
    red_note = f"red points={red_n} lowest-loss envelope points on the non-corner side"

    annotation_text = (
        f"source: {str(result_path).replace(chr(92), chr(92) * 2)}"
        f"<br>best-seed lower-envelope points and their surface are shown;"
        f" points={len(xs)}; separator_ridge_points={len(separator_points)};"
        f" {red_note}; AFM05 truth side is disabled"
    )
    xs_black = [x for x, is_red in zip(xs, red_mask) if not is_red]
    ys_black = [y for y, is_red in zip(ys, red_mask) if not is_red]
    zs_black = [z for z, is_red in zip(zs, red_mask) if not is_red]
    hovertexts_black = [htext for htext, is_red in zip(hovertexts, red_mask) if not is_red]
    xs_red = [x for x, is_red in zip(xs, red_mask) if is_red]
    ys_red = [y for y, is_red in zip(ys, red_mask) if is_red]
    zs_red = [z for z, is_red in zip(zs, red_mask) if is_red]
    hovertexts_red = [htext for htext, is_red in zip(hovertexts, red_mask) if is_red]

    traces = [envelope["surface"]]
    if len(separator_points) >= 2:
        separator_x = [point[0] for point in separator_points]
        separator_y = [point[1] for point in separator_points]
        separator_plane = {
            "type": "surface",
            "name": "corner/valley ridge separator",
            "x": [separator_x, separator_x],
            "y": [separator_y, separator_y],
            "z": [[zfloor] * len(separator_x), [ztop] * len(separator_x)],
            "surfacecolor": [[0.0] * len(separator_x), [0.0] * len(separator_x)],
            "colorscale": [[0.0, "#a7d8ff"], [1.0, "#a7d8ff"]],
            "opacity": 0.30,
            "showscale": False,
            "hoverinfo": "skip",
            "showlegend": True,
        }
        traces.append(separator_plane)
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "best NN seed envelope points",
            "x": xs_black,
            "y": ys_black,
            "z": zs_black,
            "hovertext": hovertexts_black,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 3,
                "opacity": 0.95,
                "color": "#000000",
                "line": {"width": 0},
            },
        }
    )
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": red_label,
            "x": xs_red,
            "y": ys_red,
            "z": zs_red,
            "hovertext": hovertexts_red,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 3,
                "opacity": 0.98,
                "color": "#d62728",
                "line": {"width": 0},
            },
        }
    )

    layout = {
        "title": f"AFM05 Stage1pluslight best-seed envelope points: ks0 vs cs0 vs {spec['label']}",
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
            "xaxis": {
                "title": "ks [N/m]",
                "tickmode": "array",
                "tickvals": xtickvals,
                "ticktext": xticktext,
                "range": [xlo, xhi],
            },
            "yaxis": {
                "title": "cs [N*s/m]",
                "tickmode": "array",
                "tickvals": ytickvals,
                "ticktext": yticktext,
                "range": [ylo, yhi],
            },
            "zaxis": {
                "title": spec["label"],
                "tickmode": "array",
                "tickvals": ztickvals,
                "ticktext": zticktext,
                "range": [zfloor, ztop],
            },
            "camera": {"eye": {"x": 1.5, "y": 1.32, "z": 0.92}},
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AFM05 Stage1pluslight best-seed envelope points log(ks0)-log(cs0)-{spec['label']} 3D</title>
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
    default_result_path = Path(__file__).resolve().parents[2] / "results_afm" / "afm_param_stage1pluslight_05.pkl"
    result_path = Path(argv[0]).resolve() if argv else default_result_path.resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing AFM05 Stage1pluslight result file: {result_path}")

    default_out_dir = Path(__file__).resolve().parent
    out_dir = Path(argv[1]).resolve() if len(argv) >= 2 else default_out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_payload(result_path)
    if "trial_parameters" not in payload:
        raise KeyError(f"Result file does not contain 'trial_parameters': {result_path}")
    validate_complete_stage1plus_payload(payload, path=result_path)

    all_records = [rec for rec in payload["trial_parameters"] if isinstance(rec, dict)]
    records = [
        rec
        for rec in all_records
        if not bool(field_or(rec, "early_stopped", False))
        and not bool(field_or(rec, "trial_failed", False))
        and bool(field_or(rec, "is_viable", True))
    ]
    if not records:
        raise RuntimeError(f"No viable trial records found in: {result_path}")

    z_display_max = 1e-1
    specs = metric_specs(records)
    stem = result_path.stem
    saved: list[Path] = []

    for spec in specs:
        kept_records = [
            rec
            for rec in records
            if math.isfinite(metric_value(rec, spec))
            and metric_value(rec, spec) > 0.0
            and metric_value(rec, spec) <= z_display_max
        ]
        excluded_count = len(records) - len(kept_records)
        if not kept_records:
            raise RuntimeError(f"No trial records remain after z cutoff for {spec['label']}.")

        html_path = out_dir / f"{stem}_alltrials_ks_logcs_{spec['slug']}_3d.html"
        write_plot_html(html_path, result_path, spec, kept_records, excluded_count, z_display_max)
        saved.append(html_path)
        print(f"Saved 3D HTML to: {html_path}")
        envelope_html_path = out_dir / f"{stem}_best_seed_envelope_points_ks_logcs_{spec['slug']}_3d.html"
        write_envelope_points_html(envelope_html_path, result_path, spec, kept_records, all_records)
        saved.append(envelope_html_path)
        print(f"Saved best-seed envelope points 3D HTML to: {envelope_html_path}")
        print(f"Trials plotted for {spec['label']}: {len(kept_records)}")
        print(f"Trials excluded by z cutoff for {spec['label']}: {excluded_count}")

    print(f"Saved {len(saved)} AFM05 Stage1pluslight 3D plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
