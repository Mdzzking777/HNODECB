"""Interactive 3D visualization for AFM04 prestage2 candidates on stage1pluslight landscape.

This intentionally mirrors the standard used by:
AFM04/stage1pluslight/visualization_runner/plot_afm_stage1pluslight_alltrials_3d_04.py

Base cloud:
- full stage1pluslight viable landscape after the same filtering / z cutoff

Overlays:
- 100 prestage2 Layer-A mech winners (black outline)
- 20 final prestage2 candidates B (red outline)
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
from AFM04.stage1pluslight.visualization_runner.plot_afm_stage1pluslight_alltrials_3d_04 import (
    color_metric_values,
    empirical_quantile,
    field_or,
    finite_float_or,
    fmt_e_html,
    fmt_f_html,
    hover_text as stage1_hover_text,
    load_payload,
    metric_specs,
    metric_value,
    param_or,
    sci_tick_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE1_RESULT = REPO_ROOT / "AFM04" / "stage1pluslight" / "results_afm" / "afm_param_stage1pluslight_04.pkl"
DEFAULT_PREST2_LAYER_A = REPO_ROOT / "AFM04" / "prestage2" / "results" / "afm_prest2_04_top_mech_winners_a.pkl"
DEFAULT_PREST2_FINAL = REPO_ROOT / "AFM04" / "prestage2" / "results" / "afm_prest2_04_candidates_b.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "prestage2" / "visualization"


def mech_seed_rank_label(rec: dict[str, Any]) -> str:
    mw = int(field_or(rec, "source_mech_winner", 0))
    seed = int(field_or(rec, "source_stage1_best_seedbank", 0))
    rank = int(field_or(rec, "source_stage1_rank", 0))
    if mw > 0 and seed > 0 and rank > 0:
        return f"mech winner {mw} +NN seed {seed} (rank {rank})"
    if mw > 0 and seed > 0:
        return f"mech winner {mw} +NN seed {seed}"
    if mw > 0 and rank > 0:
        return f"mech winner {mw} (rank {rank})"
    return f"mech winner {mw}"


def prest2_extra_hover(rec: dict[str, Any]) -> str:
    candidate = int(field_or(rec, "candidate_b", field_or(rec, "candidate", 0)))
    return "".join(
        [
            f"<br>top mech winner A={int(field_or(rec, 'top_mech_winner_a', field_or(rec, 'source_top_mech_winner_a', field_or(rec, 'newrank_a', field_or(rec, 'source_newrank_a', 0)))))}",
            f"<br>candidate B={candidate if candidate > 0 else 'None'}",
            f"<br>source {mech_seed_rank_label(rec)}",
            f"<br>prest2 final train_loss={fmt_e_html(field_or(rec, 'final_train_loss', math.nan))}",
            f"<br>prest2 val x1 rec={fmt_f_html(field_or(rec, 'val_x1_rec', math.nan), 2)}%",
            f"<br>prest2 val x3 rec={fmt_f_html(field_or(rec, 'val_x3_rec', math.nan), 2)}%",
            f"<br>prest2 val F_contact err={fmt_f_html(field_or(rec, 'val_nn_err', math.nan), 2)}%",
        ]
    )


def prest2_overlay_color_values(records: list[dict[str, Any]]) -> tuple[list[float], float]:
    nn_vals = [finite_float_or(field_or(rec, "val_nn_err", math.nan)) for rec in records]
    if any(math.isfinite(v) for v in nn_vals):
        cap = 200.0
        overflow = cap + 1.0
        vals = [math.nan if not math.isfinite(v) else (overflow if v > cap else max(v, 0.0)) for v in nn_vals]
        return vals, overflow

    x3_vals = [finite_float_or(field_or(rec, "val_x3_rec", math.nan)) for rec in records]
    finite = [v for v in x3_vals if math.isfinite(v)]
    vmax = max(100.0, max(finite) if finite else 100.0)
    vals = [math.nan if not math.isfinite(v) else max(v, 0.0) for v in x3_vals]
    return vals, vmax


def prest2_endpoint_hover_text(rec: dict[str, Any], *, phase_label: str) -> str:
    candidate = int(field_or(rec, "candidate_b", field_or(rec, "candidate", 0)))
    bits = [
        phase_label,
        f"<br>source {mech_seed_rank_label(rec)}",
        f"<br>trial={int(field_or(rec, 'source_stage1_trial_id', field_or(rec, 'trial_id', -1)))}",
        f"<br>top mech winner A={int(field_or(rec, 'top_mech_winner_a', field_or(rec, 'source_top_mech_winner_a', field_or(rec, 'newrank_a', field_or(rec, 'source_newrank_a', 0)))))}",
        f"<br>candidate B={candidate if candidate > 0 else 'None'}",
        f"<br>epochs_completed={int(field_or(rec, 'epochs_completed', -1))}",
        f"<br>current_layer_epochs={int(field_or(rec, 'current_layer_epochs', -1))}",
        f"<br>final train_loss={fmt_e_html(field_or(rec, 'final_train_loss', math.nan))}",
        f"<br>ks0={fmt_e_html(field_or(rec, 'ks0', math.nan))}",
        f"<br>cs0={fmt_e_html(field_or(rec, 'cs0', math.nan))}",
        f"<br>ks_hat={fmt_e_html(field_or(rec, 'ks_hat', math.nan))}",
        f"<br>cs_hat={fmt_e_html(field_or(rec, 'cs_hat', math.nan))}",
        f"<br>ks err={fmt_f_html(field_or(rec, 'ks_err_pct', math.nan), 2)}%",
        f"<br>cs err={fmt_f_html(field_or(rec, 'cs_err_pct', math.nan), 2)}%",
        f"<br>val x1 rec={fmt_f_html(field_or(rec, 'val_x1_rec', math.nan), 2)}%",
        f"<br>val x3 rec={fmt_f_html(field_or(rec, 'val_x3_rec', math.nan), 2)}%",
        f"<br>val F_contact err={fmt_f_html(field_or(rec, 'val_nn_err', math.nan), 2)}%",
    ]
    return "".join(bits)


def merged_prest2_maps(
    layer_a_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    entrant_by_trial: dict[int, dict[str, Any]] = {}
    finalist_by_trial: dict[int, dict[str, Any]] = {}

    for rec in layer_a_records:
        trial_id = int(field_or(rec, "source_stage1_trial_id", field_or(rec, "trial_id", -1)))
        if trial_id > 0:
            entrant_by_trial[trial_id] = rec

    for rec in final_records:
        trial_id = int(field_or(rec, "source_stage1_trial_id", field_or(rec, "trial_id", -1)))
        if trial_id > 0:
            finalist_by_trial[trial_id] = rec
    return entrant_by_trial, finalist_by_trial


def layer_b_best_envelope_surface(final_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_by_grid: dict[tuple[int, int] | tuple[float, float], dict[str, Any]] = {}
    for rec in final_records:
        if not isinstance(rec, dict):
            continue
        loss = finite_float_or(field_or(rec, "final_train_loss", math.nan))
        if not math.isfinite(loss) or loss <= 0.0:
            continue
        ks_hat = finite_float_or(field_or(rec, "ks_hat", math.nan))
        cs_hat = finite_float_or(field_or(rec, "cs_hat", math.nan))
        if not (math.isfinite(ks_hat) and ks_hat > 0.0 and math.isfinite(cs_hat) and cs_hat > 0.0):
            continue

        ks_node = int(field_or(rec, "source_ks_node_idx", -1))
        cs_node = int(field_or(rec, "source_cs_node_idx", -1))
        if ks_node >= 0 and cs_node >= 0:
            key: tuple[int, int] | tuple[float, float] = (ks_node, cs_node)
        else:
            ks0 = finite_float_or(field_or(rec, "ks0", math.nan))
            cs0 = finite_float_or(field_or(rec, "cs0", math.nan))
            if not (math.isfinite(ks0) and ks0 > 0.0 and math.isfinite(cs0) and cs0 > 0.0):
                continue
            key = (round(math.log10(ks0), 12), round(math.log10(cs0), 12))

        previous = best_by_grid.get(key)
        previous_loss = finite_float_or(field_or(previous or {}, "final_train_loss", math.nan))
        if previous is None or loss < previous_loss:
            best_by_grid[key] = rec

    records = sorted(
        best_by_grid.values(),
        key=lambda r: (
            int(field_or(r, "source_cs_node_idx", 10**9)),
            int(field_or(r, "source_ks_node_idx", 10**9)),
            finite_float_or(field_or(r, "cs0", math.nan), math.inf),
            finite_float_or(field_or(r, "ks0", math.nan), math.inf),
        ),
    )
    if len(records) < 3:
        return None

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    hover: list[str] = []
    for rec in records:
        ks_hat = finite_float_or(field_or(rec, "ks_hat", math.nan))
        cs_hat = finite_float_or(field_or(rec, "cs_hat", math.nan))
        loss = finite_float_or(field_or(rec, "final_train_loss", math.nan))
        xs.append(math.log10(ks_hat))
        ys.append(math.log10(cs_hat))
        zs.append(math.log10(loss))
        hover.append(
            "".join(
                [
                    "Layer-B best envelope",
                    f"<br>candidate B={int(field_or(rec, 'candidate_b', field_or(rec, 'candidate', 0)))}",
                    f"<br>source trial={int(field_or(rec, 'source_stage1_trial_id', field_or(rec, 'trial_id', -1)))}",
                    f"<br>source rank={int(field_or(rec, 'source_stage1_rank', -1))}",
                    f"<br>source mech winner={int(field_or(rec, 'source_mech_winner', -1))}",
                    f"<br>source ks0={fmt_e_html(field_or(rec, 'ks0', math.nan))}",
                    f"<br>source cs0={fmt_e_html(field_or(rec, 'cs0', math.nan))}",
                    f"<br>Layer-B ks_hat={fmt_e_html(ks_hat)}",
                    f"<br>Layer-B cs_hat={fmt_e_html(cs_hat)}",
                    f"<br>Layer-B final train_loss={fmt_e_html(loss)}",
                    f"<br>epochs_completed={int(field_or(rec, 'epochs_completed', -1))}",
                ]
            )
        )

    return {
        "type": "mesh3d",
        "name": "Layer-B best-seed envelope",
        "x": xs,
        "y": ys,
        "z": zs,
        "text": hover,
        "hovertemplate": "%{text}<extra></extra>",
        "opacity": 0.32,
        "alphahull": -1,
        "delaunayaxis": "z",
        "intensity": zs,
        "colorscale": [[0.0, "#ffe5d9"], [0.5, "#fcae91"], [1.0, "#cb181d"]],
        "showscale": False,
        "flatshading": False,
    }


def write_plot_html(
    html_path: Path,
    stage1_result_path: Path,
    spec: dict[str, Any],
    kept_records: list[dict[str, Any]],
    true_ks: float,
    true_cs: float,
    excluded_count: int,
    z_display_max: float,
    entrant_by_trial: dict[int, dict[str, Any]],
    finalist_by_trial: dict[int, dict[str, Any]],
    final_records: list[dict[str, Any]],
    z_lower_loss: float,
) -> None:
    xs_raw = [float(param_or(field_or(rec, "params", {}), "ks0", field_or(rec, "ks_hat", math.nan))) for rec in kept_records]
    ys_raw = [float(param_or(field_or(rec, "params", {}), "cs0", field_or(rec, "cs_hat", math.nan))) for rec in kept_records]
    zs_raw = [float(metric_value(rec, spec)) for rec in kept_records]

    color_title, colors, color_tickvals, color_ticktext, color_cmax = color_metric_values(kept_records)

    xs = [math.log10(x) for x in xs_raw]
    ys = [math.log10(y) for y in ys_raw]
    zs = [math.log10(z) for z in zs_raw]

    hovertexts_all = []
    entrant_idx: list[int] = []
    finalist_idx: list[int] = []
    entrant_set: set[int] = set()
    finalist_set: set[int] = set()
    entrant_stage1_to_layer_a_segments: list[tuple[float, float, float, float, float, float]] = []
    finalist_layer_a_to_final_segments: list[tuple[float, float, float, float, float, float]] = []
    entrant_overlay_records: list[dict[str, Any]] = []
    finalist_overlay_records: list[dict[str, Any]] = []
    entrant_overlay_xyz: list[tuple[float, float, float]] = []
    finalist_overlay_xyz: list[tuple[float, float, float]] = []
    entrant_overlay_hover: list[str] = []
    finalist_overlay_hover: list[str] = []

    for i, rec in enumerate(kept_records):
        trial_id = int(param_or(field_or(rec, "params", {}), "trial_id", -1))
        base_hover = stage1_hover_text(rec, spec)
        if trial_id in entrant_by_trial:
            entrant_rec = entrant_by_trial[trial_id]
            base_hover += prest2_extra_hover(entrant_rec)
            entrant_idx.append(i)
            entrant_set.add(i)
            x0 = math.log10(float(param_or(field_or(rec, "params", {}), "ks0", field_or(rec, "ks_hat", math.nan))))
            y0 = math.log10(float(param_or(field_or(rec, "params", {}), "cs0", field_or(rec, "cs_hat", math.nan))))
            z0 = math.log10(float(metric_value(rec, spec)))
            x1 = math.log10(float(field_or(entrant_rec, "ks_hat", math.nan)))
            y1 = math.log10(float(field_or(entrant_rec, "cs_hat", math.nan)))
            z1 = math.log10(float(field_or(entrant_rec, "final_train_loss", math.nan)))
            if all(math.isfinite(v) for v in [x0, y0, z0, x1, y1, z1]):
                entrant_stage1_to_layer_a_segments.append((x0, y0, z0, x1, y1, z1))
                entrant_overlay_records.append(entrant_rec)
                entrant_overlay_xyz.append((x1, y1, z1))
                entrant_overlay_hover.append(prest2_endpoint_hover_text(entrant_rec, phase_label="Layer-A endpoint after 10 epochs"))
        if trial_id in finalist_by_trial:
            final_rec = finalist_by_trial[trial_id]
            base_hover += prest2_extra_hover(final_rec)
            finalist_idx.append(i)
            finalist_set.add(i)
            x2 = math.log10(float(field_or(final_rec, "ks_hat", math.nan)))
            y2 = math.log10(float(field_or(final_rec, "cs_hat", math.nan)))
            z2 = math.log10(float(field_or(final_rec, "final_train_loss", math.nan)))
            entrant_rec = entrant_by_trial.get(trial_id)
            if entrant_rec is not None:
                x1 = math.log10(float(field_or(entrant_rec, "ks_hat", math.nan)))
                y1 = math.log10(float(field_or(entrant_rec, "cs_hat", math.nan)))
                z1 = math.log10(float(field_or(entrant_rec, "final_train_loss", math.nan)))
                if all(math.isfinite(v) for v in [x1, y1, z1, x2, y2, z2]):
                    finalist_layer_a_to_final_segments.append((x1, y1, z1, x2, y2, z2))
            if all(math.isfinite(v) for v in [x2, y2, z2]):
                finalist_overlay_records.append(final_rec)
                finalist_overlay_xyz.append((x2, y2, z2))
                finalist_overlay_hover.append(prest2_endpoint_hover_text(final_rec, phase_label="Final candidate B endpoint after +20 epochs"))
        hovertexts_all.append(base_hover)

    base_idx = [i for i in range(len(kept_records)) if i not in entrant_set]
    entrant_only_idx = [i for i in entrant_idx if i not in finalist_set]
    entrant_overlay_colors, entrant_overlay_cmax = prest2_overlay_color_values(entrant_overlay_records)
    finalist_overlay_colors, finalist_overlay_cmax = prest2_overlay_color_values(finalist_overlay_records)

    sorted_zs_raw = sorted(zs_raw)
    zfocus_lo_raw = empirical_quantile(sorted_zs_raw, 0.01)
    zfocus_hi_raw = empirical_quantile(sorted_zs_raw, 0.995)
    zfocus_lo = math.log10(zfocus_lo_raw)
    zfocus_hi = math.log10(zfocus_hi_raw)
    zpad = max((zfocus_hi - zfocus_lo) * 0.12, 0.015)
    finalist_lo_raw = zfocus_lo_raw if not finalist_idx else 10.0 ** min(zs[i] for i in finalist_idx)
    finalist_lo = math.log10(finalist_lo_raw)
    highlight_pad = max((zfocus_lo - finalist_lo) * 0.08, 0.01)
    zfloor = min(zfocus_lo - zpad, finalist_lo - highlight_pad)
    ztop = zfocus_hi + zpad
    overlay_xs = [xyz[0] for xyz in entrant_overlay_xyz] + [xyz[0] for xyz in finalist_overlay_xyz]
    overlay_ys = [xyz[1] for xyz in entrant_overlay_xyz] + [xyz[1] for xyz in finalist_overlay_xyz]
    overlay_zs = [xyz[2] for xyz in entrant_overlay_xyz] + [xyz[2] for xyz in finalist_overlay_xyz]
    z_hidden_low = sum(1 for z in zs_raw if z < zfocus_lo_raw)
    z_hidden_high = sum(1 for z in zs_raw if z > zfocus_hi_raw)

    truth_x = math.log10(true_ks) if math.isfinite(true_ks) and true_ks > 0 else math.nan
    truth_y = math.log10(true_cs) if math.isfinite(true_cs) and true_cs > 0 else math.nan

    xlo = min(xs + overlay_xs) if overlay_xs else min(xs)
    xhi = max(xs + overlay_xs) if overlay_xs else max(xs)
    ylo = min(ys + overlay_ys) if overlay_ys else min(ys)
    yhi = max(ys + overlay_ys) if overlay_ys else max(ys)
    if overlay_zs:
        ztop = max(ztop, max(overlay_zs) + 0.03)
    if math.isfinite(z_lower_loss) and z_lower_loss > 0.0:
        zfloor = math.log10(z_lower_loss)
    truth_z = zfloor
    xtickvals, xticktext = sci_tick_spec(min(xs_raw), max(xs_raw))
    ytickvals, yticktext = sci_tick_spec(min(ys_raw), max(ys_raw))
    ztickvals, zticktext = sci_tick_spec(zfocus_lo_raw, zfocus_hi_raw)
    if math.isfinite(z_lower_loss) and z_lower_loss > 0.0:
        lower_tick = math.log10(z_lower_loss)
        if not any(abs(v - lower_tick) < 1e-12 for v in ztickvals):
            ztickvals = [lower_tick] + ztickvals
            zticktext = [f"{z_lower_loss:.2e}"] + zticktext

    annotation_text = (
        f"source: {str(stage1_result_path).replace(chr(92), chr(92) * 2)}"
        f"<br>x=ks0 (log10), y=cs0 (log10), z={spec['label']} (log10)"
        f"<br>base cloud uses stage1pluslight trials; black outline marks 100 prest2 mech winners;"
        f" red outline marks final 20 candidates B;"
        f" filled overlays show their post-GBO positions with arrows"
        f"<br>excluded {excluded_count} stage1 trials with {spec['label']} > {z_display_max:.1e} or <= 0;"
        f" z-focus uses q01..q99.5 and hides {z_hidden_low} low / {z_hidden_high} high outliers from the visible z span"
    )

    colorscale = [
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
    ]

    traces: list[dict[str, Any]] = []
    surface = layer_b_best_envelope_surface(final_records)
    if surface is not None:
        traces.append(surface)
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "showlegend": False,
            "name": f"stage1 trials with {spec['label']} <= {z_display_max:.1e}",
            "x": [xs[i] for i in base_idx],
            "y": [ys[i] for i in base_idx],
            "z": [zs[i] for i in base_idx],
            "hovertext": [hovertexts_all[i] for i in base_idx],
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 3,
                "opacity": 0.82,
                "color": [colors[i] for i in base_idx],
                "cmin": 0.0,
                "cmax": color_cmax,
                "colorscale": colorscale,
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
            "showlegend": False,
            "name": f"prest2 entrant origins ({len(entrant_idx)})",
            "x": [xs[i] for i in entrant_only_idx],
            "y": [ys[i] for i in entrant_only_idx],
            "z": [zs[i] for i in entrant_only_idx],
            "hovertext": [hovertexts_all[i] for i in entrant_only_idx],
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 5,
                "opacity": 1.0,
                "color": [colors[i] for i in entrant_only_idx],
                "cmin": 0.0,
                "cmax": color_cmax,
                "colorscale": colorscale,
                "showscale": False,
                "line": {"color": "#000000", "width": 4},
            },
        }
    )
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "showlegend": False,
            "name": f"prest2 finalist origins ({len(finalist_idx)})",
            "x": [xs[i] for i in finalist_idx],
            "y": [ys[i] for i in finalist_idx],
            "z": [zs[i] for i in finalist_idx],
            "hovertext": [hovertexts_all[i] for i in finalist_idx],
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 7,
                "opacity": 1.0,
                "color": [colors[i] for i in finalist_idx],
                "cmin": 0.0,
                "cmax": color_cmax,
                "colorscale": colorscale,
                "showscale": False,
                "line": {"color": "#dc2626", "width": 5},
            },
        }
    )
    if entrant_overlay_xyz:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "showlegend": False,
                "name": f"after 10 epochs ({len(entrant_overlay_xyz)})",
                "x": [xyz[0] for xyz in entrant_overlay_xyz],
                "y": [xyz[1] for xyz in entrant_overlay_xyz],
                "z": [xyz[2] for xyz in entrant_overlay_xyz],
                "hovertext": entrant_overlay_hover,
                "hovertemplate": "%{hovertext}<extra></extra>",
                "marker": {
                    "size": 5,
                    "opacity": 0.95,
                    "color": entrant_overlay_colors,
                    "cmin": 0.0,
                    "cmax": max(color_cmax, entrant_overlay_cmax),
                    "colorscale": colorscale,
                    "showscale": False,
                    "line": {"width": 0},
                },
            }
        )
    if finalist_overlay_xyz:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "showlegend": False,
                "name": f"after +20 more epochs ({len(finalist_overlay_xyz)})",
                "x": [xyz[0] for xyz in finalist_overlay_xyz],
                "y": [xyz[1] for xyz in finalist_overlay_xyz],
                "z": [xyz[2] for xyz in finalist_overlay_xyz],
                "hovertext": finalist_overlay_hover,
                "hovertemplate": "%{hovertext}<extra></extra>",
                "marker": {
                    "size": 7,
                    "opacity": 1.0,
                    "color": finalist_overlay_colors,
                    "cmin": 0.0,
                    "cmax": max(color_cmax, finalist_overlay_cmax),
                    "colorscale": colorscale,
                    "showscale": False,
                    "line": {"width": 0},
                },
            }
        )
    if entrant_stage1_to_layer_a_segments:
        for seg_idx, (x0, y0, z0, x1, y1, z1) in enumerate(entrant_stage1_to_layer_a_segments):
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "name": "E0 → E10 flow" if seg_idx == 0 else "E0 → E10 flow",
                    "showlegend": seg_idx == 0,
                    "x": [x0, x1],
                    "y": [y0, y1],
                    "z": [z0, z1],
                    "line": {"color": "#1a8f3f", "width": 3},
                    "hoverinfo": "skip",
                }
            )
        traces.append(
            {
                "type": "cone",
                "name": "E0 → E10 arrows",
                "x": [seg[3] for seg in entrant_stage1_to_layer_a_segments],
                "y": [seg[4] for seg in entrant_stage1_to_layer_a_segments],
                "z": [seg[5] for seg in entrant_stage1_to_layer_a_segments],
                "u": [seg[3] - seg[0] for seg in entrant_stage1_to_layer_a_segments],
                "v": [seg[4] - seg[1] for seg in entrant_stage1_to_layer_a_segments],
                "w": [seg[5] - seg[2] for seg in entrant_stage1_to_layer_a_segments],
                "anchor": "tip",
                "colorscale": [[0, "#1a8f3f"], [1, "#1a8f3f"]],
                "showscale": False,
                "sizemode": "absolute",
                "sizeref": 0.085,
                "hoverinfo": "skip",
            }
        )
    if finalist_layer_a_to_final_segments:
        for seg_idx, (x0, y0, z0, x1, y1, z1) in enumerate(finalist_layer_a_to_final_segments):
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "name": "E10 → E30 flow" if seg_idx == 0 else "E10 → E30 flow",
                    "showlegend": seg_idx == 0,
                    "x": [x0, x1],
                    "y": [y0, y1],
                    "z": [z0, z1],
                    "line": {"color": "#dc2626", "width": 4},
                    "hoverinfo": "skip",
                }
            )
        traces.append(
            {
                "type": "cone",
                "name": "E10 → E30 arrows",
                "x": [seg[3] for seg in finalist_layer_a_to_final_segments],
                "y": [seg[4] for seg in finalist_layer_a_to_final_segments],
                "z": [seg[5] for seg in finalist_layer_a_to_final_segments],
                "u": [seg[3] - seg[0] for seg in finalist_layer_a_to_final_segments],
                "v": [seg[4] - seg[1] for seg in finalist_layer_a_to_final_segments],
                "w": [seg[5] - seg[2] for seg in finalist_layer_a_to_final_segments],
                "anchor": "tip",
                "colorscale": [[0, "#dc2626"], [1, "#dc2626"]],
                "showscale": False,
                "sizemode": "absolute",
                "sizeref": 0.12,
                "hoverinfo": "skip",
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
        "title": f"AFM04 prest2 on stage1pluslight landscape: ks0 vs cs0 vs {spec['label']}",
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
  <title>AFM04 prest2 on stage1pluslight landscape 3D</title>
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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    stage1_result_path = Path(argv[0]).resolve() if len(argv) >= 1 else DEFAULT_STAGE1_RESULT.resolve()
    layer_a_result_path = Path(argv[1]).resolve() if len(argv) >= 2 else DEFAULT_PREST2_LAYER_A.resolve()
    final_result_path = Path(argv[2]).resolve() if len(argv) >= 3 else DEFAULT_PREST2_FINAL.resolve()
    out_dir = Path(argv[3]).resolve() if len(argv) >= 4 else DEFAULT_OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not stage1_result_path.is_file():
        raise FileNotFoundError(f"Missing AFM04 stage1pluslight result file: {stage1_result_path}")
    if not layer_a_result_path.is_file():
        raise FileNotFoundError(f"Missing AFM04 prestage2 Layer-A result file: {layer_a_result_path}")
    if not final_result_path.is_file():
        raise FileNotFoundError(f"Missing AFM04 prestage2 final result file: {final_result_path}")

    stage1_payload = load_payload(stage1_result_path)
    layer_a_payload = load_payload(layer_a_result_path)
    final_payload = load_payload(final_result_path)

    if "trial_parameters" not in stage1_payload:
        raise KeyError(f"Result file does not contain 'trial_parameters': {stage1_result_path}")
    validate_complete_stage1plus_payload(stage1_payload, path=stage1_result_path)

    all_stage1_records = [rec for rec in stage1_payload["trial_parameters"] if isinstance(rec, dict)]
    stage1_records = [
        rec for rec in all_stage1_records
        if not bool(field_or(rec, "early_stopped", False))
        and not bool(field_or(rec, "trial_failed", False))
        and bool(field_or(rec, "is_viable", True))
    ]
    if not stage1_records:
        raise RuntimeError(f"No viable stage1 trial records found in: {stage1_result_path}")

    specs = metric_specs(stage1_records)
    spec = specs[0]
    z_display_max = 1e-1

    kept_records = [
        rec for rec in stage1_records
        if math.isfinite(metric_value(rec, spec))
        and metric_value(rec, spec) > 0.0
        and metric_value(rec, spec) <= z_display_max
    ]
    excluded_count = len(stage1_records) - len(kept_records)
    if not kept_records:
        raise RuntimeError(f"No stage1 trial records remain after z cutoff for {spec['label']}.")

    layer_a_records = [
        rec for rec in field_or(layer_a_payload, "top_mech_winner_a_records", field_or(layer_a_payload, "newrank_a_records", []))
        if isinstance(rec, dict)
    ]
    final_records = [
        rec for rec in field_or(final_payload, "candidate_b_records", field_or(final_payload, "candidate_records", []))
        if isinstance(rec, dict)
    ]
    entrant_by_trial, finalist_by_trial = merged_prest2_maps(layer_a_records, final_records)
    candidate1_loss = math.nan
    if final_records:
        candidate1_loss = finite_float_or(field_or(final_records[0], "final_train_loss", math.nan))

    html_path = out_dir / "afm_prest2_04_allcandidates_ks_logcs_final_train_loss_3d.html"
    write_plot_html(
        html_path,
        stage1_result_path,
        spec,
        kept_records,
        float(KS),
        float(CS),
        excluded_count,
        z_display_max,
        entrant_by_trial,
        finalist_by_trial,
        final_records,
        candidate1_loss,
    )
    print(f"Saved prest2 3D HTML to: {html_path}")
    print(f"Stage1 trials plotted: {len(kept_records)}")
    print(f"Stage1 trials excluded by z cutoff: {excluded_count}")
    print(f"prest2 entrants highlighted: {len(entrant_by_trial)}")
    print(f"prest2 finalists highlighted: {len(finalist_by_trial)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
