from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM05").is_dir() and ((path / "user requirements").is_dir() or (path / "Requirements").is_dir()):
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
ST1PL_VIS_SCRIPT = (
    REPO_ROOT
    / "AFM05"
    / "stage1pluslight"
    / "runner"
    / "visualization"
    / "plot_afm_stage1pluslight_alltrials_3d_05.py"
)
DEFAULT_ST1PL_RESULT = (
    REPO_ROOT
    / "AFM05"
    / "stage1pluslight"
    / "results_afm"
    / "afm_param_stage1pluslight_05.pkl"
)
DEFAULT_LAYER_A_PATH = REPO_ROOT / "AFM05" / "prestage2" / "results" / "afm_prest2_05_top_mech_winners_a.pkl"
DEFAULT_CANDIDATES_PATH = REPO_ROOT / "AFM05" / "prestage2" / "results" / "afm_prest2_05_candidates_b.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM05" / "prestage2" / "visualization"
DEFAULT_HTML_NAME = "afm_prest2_05_allcandidates_ks_logcs_final_train_loss_3d.html"
DEFAULT_CSV_NAME = "afm_prest2_05_allcandidates_ks_logcs_final_train_loss_3d_routes.csv"


def _load_stage1_vis_module() -> Any:
    if not ST1PL_VIS_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing stage1pluslight visualization helper: {ST1PL_VIS_SCRIPT}")
    spec = importlib.util.spec_from_file_location("afm05_stage1pluslight_3d_helper", ST1PL_VIS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {ST1PL_VIS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected pickle payload type for {path}: {type(payload)!r}")
    return payload


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _candidate_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("candidate_b_records", "candidate_records"):
        records = payload.get(key, [])
        if isinstance(records, list) and records:
            return sorted((rec for rec in records if isinstance(rec, dict)), key=_candidate_id)
    return []


def _layer_a_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("top_mech_winner_a_records", "newrank_a_records", "candidate_records"):
        records = payload.get(key, [])
        if isinstance(records, list) and records:
            return sorted(
                (rec for rec in records if isinstance(rec, dict)),
                key=lambda rec: int(rec.get("top_mech_winner_a", rec.get("newrank_a", 10**9))),
            )
    return []


def _candidate_id(record: dict[str, Any]) -> int:
    for key in ("candidate_b", "candidate"):
        try:
            value = int(record.get(key, 0))
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _fmt_e(value: Any) -> str:
    fval = _finite_float(value)
    return "nan" if not math.isfinite(fval) else f"{fval:.6e}"


def _trial_id(record: dict[str, Any]) -> int:
    return int(record.get("source_stage1_trial_id", record.get("trial_id", 0)))


def _route_hover(layer_a_record: dict[str, Any], layer_b_record: dict[str, Any] | None = None) -> str:
    candidate = _candidate_id(layer_b_record or {})
    candidate_label = f"candidate B{candidate:02d}" if candidate > 0 else "not promoted to Layer B"
    return "".join(
        [
            candidate_label,
            f"<br>Layer A rank={int(layer_a_record.get('top_mech_winner_a', layer_a_record.get('newrank_a', 0)))}",
            f"<br>source rank={int(layer_a_record.get('source_stage1_rank', 0))}",
            f"<br>source mech winner={int(layer_a_record.get('source_mech_winner', 0))}",
            f"<br>source trial={_trial_id(layer_a_record)}",
            f"<br>Layer A start loss={_fmt_e(layer_a_record.get('train_loss_start'))}",
            f"<br>Layer A final loss={_fmt_e(layer_a_record.get('final_train_loss'))}",
            f"<br>Layer B final loss={_fmt_e((layer_b_record or {}).get('final_train_loss'))}",
        ]
    )


def _point_hover(
    layer_a_record: dict[str, Any],
    layer_b_record: dict[str, Any] | None,
    point: dict[str, Any],
) -> str:
    candidate = _candidate_id(layer_b_record or {})
    candidate_label = f"candidate B{candidate:02d}" if candidate > 0 else "Layer A only"
    return (
        f"{candidate_label}"
        f"<br>stage={point.get('stage', '')}"
        f"<br>source trial={_trial_id(layer_a_record)}"
        f"<br>ks={_fmt_e(point.get('ks'))}"
        f"<br>cs={_fmt_e(point.get('cs'))}"
        f"<br>train_loss={_fmt_e(point.get('train_loss'))}"
    )


def _route_points(
    layer_a_record: dict[str, Any],
    layer_b_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    ks0 = _finite_float(layer_a_record.get("ks0"))
    cs0 = _finite_float(layer_a_record.get("cs0"))
    start_loss = _finite_float(layer_a_record.get("train_loss_start"))
    x3_refit_meta = layer_a_record.get("x3_refit_meta", {})
    refit_start_loss = math.nan
    if isinstance(x3_refit_meta, dict):
        refit_start_loss = _finite_float(x3_refit_meta.get("loss_after_x3_refit"))
    if not (refit_start_loss > 0.0):
        refit_start_loss = start_loss

    if ks0 > 0.0 and cs0 > 0.0 and refit_start_loss > 0.0:
        points.append(
            {"stage": "Layer A start", "layer_tag": "A", "ks": ks0, "cs": cs0, "train_loss": refit_start_loss}
        )

    layer_a_ks = _finite_float(layer_a_record.get("ks_hat"))
    layer_a_cs = _finite_float(layer_a_record.get("cs_hat"))
    layer_a_loss = _finite_float(
        layer_a_record.get("final_train_loss"), _finite_float(layer_a_record.get("candidate_loss"))
    )
    if layer_a_ks > 0.0 and layer_a_cs > 0.0 and layer_a_loss > 0.0:
        points.append(
            {
                "stage": "Layer A end / Layer B start",
                "layer_tag": "A",
                "ks": layer_a_ks,
                "cs": layer_a_cs,
                "train_loss": layer_a_loss,
            }
        )

    if layer_b_record is not None and not bool(layer_b_record.get("candidate_skipped", False)):
        layer_b_ks = _finite_float(layer_b_record.get("ks_hat"))
        layer_b_cs = _finite_float(layer_b_record.get("cs_hat"))
        layer_b_loss = _finite_float(
            layer_b_record.get("final_train_loss"), _finite_float(layer_b_record.get("candidate_loss"))
        )
        if layer_b_ks > 0.0 and layer_b_cs > 0.0 and layer_b_loss > 0.0:
            points.append(
                {
                    "stage": "Layer B end",
                    "layer_tag": "B",
                    "ks": layer_b_ks,
                    "cs": layer_b_cs,
                    "train_loss": layer_b_loss,
                }
            )

    return points


def _write_routes_csv(
    path: Path,
    layer_a_records: list[dict[str, Any]],
    layer_b_by_trial: dict[int, dict[str, Any]],
) -> None:
    columns = [
        "top_mech_winner_a",
        "candidate_b",
        "point_index",
        "stage",
        "layer_tag",
        "ks",
        "cs",
        "train_loss",
        "log10_ks",
        "log10_cs",
        "log10_train_loss",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for layer_a_record in layer_a_records:
            layer_b_record = layer_b_by_trial.get(_trial_id(layer_a_record))
            candidate = _candidate_id(layer_b_record or {})
            for idx, point in enumerate(_route_points(layer_a_record, layer_b_record), start=1):
                ks = _finite_float(point.get("ks"))
                cs = _finite_float(point.get("cs"))
                loss = _finite_float(point.get("train_loss"))
                writer.writerow(
                    {
                        "top_mech_winner_a": int(
                            layer_a_record.get("top_mech_winner_a", layer_a_record.get("newrank_a", 0))
                        ),
                        "candidate_b": candidate,
                        "point_index": idx,
                        "stage": point.get("stage", ""),
                        "layer_tag": point.get("layer_tag", ""),
                        "ks": f"{ks:.16e}",
                        "cs": f"{cs:.16e}",
                        "train_loss": f"{loss:.16e}",
                        "log10_ks": f"{math.log10(ks):.16e}",
                        "log10_cs": f"{math.log10(cs):.16e}",
                        "log10_train_loss": f"{math.log10(loss):.16e}",
                    }
                )


def _build_stage1_envelope(helper: Any, st1pl_result_path: Path, z_display_max: float) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    payload = _load_pickle(st1pl_result_path)
    if "trial_parameters" not in payload:
        raise KeyError(f"Stage1pluslight result does not contain 'trial_parameters': {st1pl_result_path}")
    helper.validate_complete_stage1plus_payload(payload, path=st1pl_result_path)

    all_records = [rec for rec in payload["trial_parameters"] if isinstance(rec, dict)]
    records = [
        rec
        for rec in all_records
        if not bool(helper.field_or(rec, "early_stopped", False))
        and not bool(helper.field_or(rec, "trial_failed", False))
        and bool(helper.field_or(rec, "is_viable", True))
    ]
    if not records:
        raise RuntimeError(f"No viable st1pl records found in {st1pl_result_path}")

    specs = helper.metric_specs(records)
    if len(specs) != 1:
        raise RuntimeError(f"Expected one AFM05 metric spec, got {len(specs)}")
    spec = specs[0]
    kept_records = [
        rec
        for rec in records
        if math.isfinite(helper.metric_value(rec, spec))
        and helper.metric_value(rec, spec) > 0.0
        and helper.metric_value(rec, spec) <= z_display_max
    ]
    if not kept_records:
        raise RuntimeError(f"No st1pl records remain after z cutoff <= {z_display_max:.3e}.")

    envelope = helper.best_seed_envelope(kept_records, spec)
    if envelope is None:
        raise RuntimeError("No best-seed envelope points are available.")
    return envelope, spec, kept_records


def _write_html(
    path: Path,
    *,
    helper: Any,
    envelope: dict[str, Any],
    spec: dict[str, Any],
    layer_a_records: list[dict[str, Any]],
    layer_b_records: list[dict[str, Any]],
    layer_a_path: Path,
    candidates_path: Path,
    st1pl_result_path: Path,
) -> None:
    surface = dict(envelope["surface"])
    surface["name"] = "st1pl best-seed envelope"
    surface["opacity"] = 0.46
    surface["colorscale"] = [[0.0, "#d9f0ff"], [0.45, "#74add1"], [1.0, "#08306b"]]
    surface["showscale"] = False

    ks_raw = list(envelope["ks_raw"])
    cs_raw = list(envelope["cs_raw"])
    xs_surface = list(envelope["point_x"])
    ys_surface = list(envelope["point_y"])
    zs_surface = list(envelope["point_z"])

    route_x_all: list[float] = []
    route_y_all: list[float] = []
    route_z_all: list[float] = []
    traces: list[dict[str, Any]] = [surface]
    layer_style = {
        "A": {"name": "Layer A route", "color": "#d62728"},
        "B": {"name": "Layer B route", "color": "#2ca02c"},
    }
    layer_lines = {
        "A": {"x": [], "y": [], "z": [], "hovertext": []},
        "B": {"x": [], "y": [], "z": [], "hovertext": []},
    }
    layer_arrows = {
        "A": {"x": [], "y": [], "z": [], "u": [], "v": [], "w": [], "hovertext": []},
        "B": {"x": [], "y": [], "z": [], "u": [], "v": [], "w": [], "hovertext": []},
    }

    start_x: list[float] = []
    start_y: list[float] = []
    start_z: list[float] = []
    start_text: list[str] = []
    layer_a_end_x: list[float] = []
    layer_a_end_y: list[float] = []
    layer_a_end_z: list[float] = []
    layer_a_end_text: list[str] = []
    layer_b_end_x: list[float] = []
    layer_b_end_y: list[float] = []
    layer_b_end_z: list[float] = []
    layer_b_end_text: list[str] = []

    layer_b_by_trial = {_trial_id(record): record for record in layer_b_records}
    for layer_a_record in layer_a_records:
        layer_b_record = layer_b_by_trial.get(_trial_id(layer_a_record))
        points = _route_points(layer_a_record, layer_b_record)
        if len(points) < 2:
            continue
        xs = [math.log10(_finite_float(point["ks"])) for point in points]
        ys = [math.log10(_finite_float(point["cs"])) for point in points]
        zs = [math.log10(_finite_float(point["train_loss"])) for point in points]
        route_x_all.extend(xs)
        route_y_all.extend(ys)
        route_z_all.extend(zs)
        for idx in range(1, len(points)):
            layer_tag = str(points[idx].get("layer_tag", "A")).upper()
            if layer_tag not in {"A", "B"}:
                layer_tag = "A"
            layer_lines[layer_tag]["x"].extend([xs[idx - 1], xs[idx], None])
            layer_lines[layer_tag]["y"].extend([ys[idx - 1], ys[idx], None])
            layer_lines[layer_tag]["z"].extend([zs[idx - 1], zs[idx], None])
            layer_lines[layer_tag]["hovertext"].extend(
                [
                    _point_hover(layer_a_record, layer_b_record, points[idx - 1]),
                    _point_hover(layer_a_record, layer_b_record, points[idx]),
                    None,
                ]
            )

            dx = xs[idx] - xs[idx - 1]
            dy = ys[idx] - ys[idx - 1]
            dz = zs[idx] - zs[idx - 1]
            norm = math.sqrt(dx * dx + dy * dy + dz * dz)
            if norm > 1e-12:
                layer_arrows[layer_tag]["x"].append(xs[idx])
                layer_arrows[layer_tag]["y"].append(ys[idx])
                layer_arrows[layer_tag]["z"].append(zs[idx])
                layer_arrows[layer_tag]["u"].append(dx / norm)
                layer_arrows[layer_tag]["v"].append(dy / norm)
                layer_arrows[layer_tag]["w"].append(dz / norm)
                layer_arrows[layer_tag]["hovertext"].append(
                    _point_hover(layer_a_record, layer_b_record, points[idx])
                )

        start_x.append(xs[0])
        start_y.append(ys[0])
        start_z.append(zs[0])
        start_text.append(_route_hover(layer_a_record, layer_b_record))
        layer_a_end_x.append(xs[1])
        layer_a_end_y.append(ys[1])
        layer_a_end_z.append(zs[1])
        layer_a_end_text.append(_route_hover(layer_a_record, layer_b_record))
        if len(points) >= 3:
            layer_b_end_x.append(xs[2])
            layer_b_end_y.append(ys[2])
            layer_b_end_z.append(zs[2])
            layer_b_end_text.append(_route_hover(layer_a_record, layer_b_record))

    for layer_tag in ("A", "B"):
        style = layer_style[layer_tag]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": style["name"],
                "x": layer_lines[layer_tag]["x"],
                "y": layer_lines[layer_tag]["y"],
                "z": layer_lines[layer_tag]["z"],
                "hovertext": layer_lines[layer_tag]["hovertext"],
                "hovertemplate": "%{hovertext}<extra></extra>",
                "legendgroup": f"layer_{layer_tag}",
                "line": {"color": style["color"], "width": 5},
            }
        )
        traces.append(
            {
                "type": "cone",
                "name": f"Layer {layer_tag} arrows",
                "x": layer_arrows[layer_tag]["x"],
                "y": layer_arrows[layer_tag]["y"],
                "z": layer_arrows[layer_tag]["z"],
                "u": layer_arrows[layer_tag]["u"],
                "v": layer_arrows[layer_tag]["v"],
                "w": layer_arrows[layer_tag]["w"],
                "hovertext": layer_arrows[layer_tag]["hovertext"],
                "hovertemplate": "%{hovertext}<extra></extra>",
                "legendgroup": f"layer_{layer_tag}",
                "showscale": False,
                "sizemode": "absolute",
                "sizeref": 0.055,
                "anchor": "tip",
                "opacity": 0.82,
                "colorscale": [[0.0, style["color"]], [1.0, style["color"]]],
                "showlegend": False,
            }
        )

    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "Layer A starts",
            "x": start_x,
            "y": start_y,
            "z": start_z,
            "hovertext": start_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 5,
                "symbol": "circle",
                "color": "#111111",
                "line": {"width": 0.8, "color": "#111111"},
            },
        }
    )
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "Layer A ends / Layer B starts",
            "x": layer_a_end_x,
            "y": layer_a_end_y,
            "z": layer_a_end_z,
            "hovertext": layer_a_end_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 7,
                "symbol": "circle",
                "color": "#d62728",
                "line": {"width": 1.4, "color": "#111111"},
            },
        }
    )
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "Layer B ends",
            "x": layer_b_end_x,
            "y": layer_b_end_y,
            "z": layer_b_end_z,
            "hovertext": layer_b_end_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 8,
                "symbol": "circle",
                "color": "#2ca02c",
                "line": {"width": 1.4, "color": "#111111"},
            },
        }
    )

    xs_all = xs_surface + route_x_all
    ys_all = ys_surface + route_y_all
    zs_all = zs_surface + route_z_all
    xlo, xhi = min(xs_all), max(xs_all)
    ylo, yhi = min(ys_all), max(ys_all)
    zlo, zhi = min(zs_all), max(zs_all)
    zpad = max((zhi - zlo) * 0.08, 0.015)
    xtickvals, xticktext = helper.sci_tick_spec(min(ks_raw), max(ks_raw))
    ytickvals, yticktext = helper.sci_tick_spec(min(cs_raw), max(cs_raw))
    ztickvals, zticktext = helper.sci_tick_spec(10.0**zlo, 10.0**zhi)

    annotation_text = (
        f"Layer A records: {str(layer_a_path).replace(chr(92), chr(92) * 2)}"
        f"<br>Layer B candidates: {str(candidates_path).replace(chr(92), chr(92) * 2)}"
        f"<br>st1pl envelope: {str(st1pl_result_path).replace(chr(92), chr(92) * 2)}"
        f"<br>black = Layer A start; red = Layer A end / Layer B start; green = Layer B end"
    )
    layout = {
        "title": "AFM05 prest2 candidate optimization routes on st1pl best-seed envelope",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "legend": {"orientation": "v", "x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.78)"},
        "margin": {"l": 0, "r": 0, "t": 78, "b": 0},
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
                "title": f"log10({spec['label']})",
                "tickmode": "array",
                "tickvals": ztickvals,
                "ticktext": zticktext,
                "range": [zlo - zpad, zhi + zpad],
            },
            "camera": {"eye": {"x": 1.45, "y": 1.35, "z": 0.92}},
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AFM05 prest2 candidate routes</title>
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot AFM05 prest2 candidate optimization routes on the st1pl best-seed envelope."
    )
    parser.add_argument("--st1pl-result", type=Path, default=DEFAULT_ST1PL_RESULT)
    parser.add_argument("--layer-a", type=Path, default=DEFAULT_LAYER_A_PATH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--html-name", default=DEFAULT_HTML_NAME)
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME)
    parser.add_argument("--z-display-max", type=float, default=1.0e-1)
    args = parser.parse_args(argv)

    helper = _load_stage1_vis_module()
    envelope, spec, _kept_records = _build_stage1_envelope(helper, args.st1pl_result.resolve(), args.z_display_max)

    layer_a_payload = _load_pickle(args.layer_a.resolve())
    layer_a_records = _layer_a_records(layer_a_payload)
    if not layer_a_records:
        raise RuntimeError(f"No Layer A records found in {args.layer_a}")

    candidate_payload = _load_pickle(args.candidates.resolve())
    candidate_records = _candidate_records(candidate_payload)
    if not candidate_records:
        raise RuntimeError(f"No prest2 candidate records found in {args.candidates}")

    layer_a_trial_ids = {_trial_id(record) for record in layer_a_records}
    missing_layer_b_sources = [
        _trial_id(record) for record in candidate_records if _trial_id(record) not in layer_a_trial_ids
    ]
    if missing_layer_b_sources:
        raise RuntimeError(
            "Layer B candidates do not map to Layer A source trials: "
            + ", ".join(str(trial_id) for trial_id in missing_layer_b_sources)
        )

    out_dir = args.out_dir.resolve()
    html_path = out_dir / args.html_name
    csv_path = out_dir / args.csv_name
    layer_b_by_trial = {_trial_id(record): record for record in candidate_records}
    _write_routes_csv(csv_path, layer_a_records, layer_b_by_trial)
    _write_html(
        html_path,
        helper=helper,
        envelope=envelope,
        spec=spec,
        layer_a_records=layer_a_records,
        layer_b_records=candidate_records,
        layer_a_path=args.layer_a.resolve(),
        candidates_path=args.candidates.resolve(),
        st1pl_result_path=args.st1pl_result.resolve(),
    )
    print(f"Loaded Layer A records: {len(layer_a_records)}")
    print(f"Loaded Layer B candidates: {len(candidate_records)}")
    print(f"Saved route HTML: {html_path}")
    print(f"Saved route CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
