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
DEFAULT_LAYER_A_ASSIGNMENTS_GLOB = (
    REPO_ROOT
    / "AFM05"
    / "prestage2"
    / "results"
    / "afm_prest2_05_layerA_assignments_p*.json"
)
DEFAULT_LAYER_B_ASSIGNMENTS_GLOB = (
    REPO_ROOT
    / "AFM05"
    / "prestage2"
    / "results"
    / "afm_prest2_05_layerB_assignments_p*.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "AFM05" / "prestage2" / "visualization"
DEFAULT_HTML_NAME = "afm_prest2_layerB_candidates_on_st1pl_best_seed_envelope_ks_logcs_train_loss_3d.html"
DEFAULT_CSV_NAME = "afm_prest2_layerB_candidates_on_st1pl_best_seed_envelope_points_05.csv"


def _load_stage1_vis_module() -> Any:
    if not ST1PL_VIS_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing stage1pluslight visualization helper: {ST1PL_VIS_SCRIPT}")
    spec = importlib.util.spec_from_file_location("afm05_stage1pluslight_envelope_3d_helper", ST1PL_VIS_SCRIPT)
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


def _glob_paths(pattern: str, *, required: bool = True) -> list[Path]:
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        paths = sorted(pattern_path.parent.glob(pattern_path.name))
    else:
        paths = sorted(Path().glob(pattern))
    if not paths and required:
        raise FileNotFoundError(f"No files match: {pattern}")
    return paths


def _load_layer_assignments(pattern: str, *, rank_key: str, required: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _glob_paths(pattern, required=required):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(f"Expected list in {path}, got {type(data)!r}")
        for item in data:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["_assignment_file"] = path.name
            rows.append(row)
    rows.sort(key=lambda rec: int(rec.get(rank_key, rec.get("source_mech_winner", 10**9))))
    for idx, row in enumerate(rows, start=1):
        row.setdefault(rank_key, idx)
    return rows


def _trial_id(record: dict[str, Any]) -> int:
    params = record.get("params", {})
    if isinstance(params, dict):
        try:
            return int(params.get("trial_id", -1))
        except Exception:
            return -1
    return -1


def _assignment_trial_id(row: dict[str, Any]) -> int:
    return int(row.get("source_stage1_trial_id", row.get("trial_id", -1)))


def _record_params(record: dict[str, Any]) -> dict[str, Any]:
    params = record.get("params", {})
    return params if isinstance(params, dict) else {}


def _write_candidate_csv(
    path: Path,
    helper: Any,
    spec: dict[str, Any],
    highlighted_records: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> None:
    columns = [
        "marker_role",
        "layer_a_rank",
        "candidate_b",
        "source_top_mech_winner_a",
        "source_mech_winner",
        "source_stage1_rank",
        "source_stage1_trial_id",
        "ks0",
        "cs0",
        "ks_node_idx",
        "cs_node_idx",
        "nn_seed_bank_idx",
        "nn_init_seed",
        "node_label",
        "train_loss",
        "assignment_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for role, record, assignment in highlighted_records:
            params = _record_params(record)
            writer.writerow(
                {
                    "marker_role": role,
                    "layer_a_rank": assignment.get("source_mech_winner", assignment.get("top_mech_winner_a", "")),
                    "candidate_b": assignment.get("candidate_b", ""),
                    "source_top_mech_winner_a": assignment.get("source_top_mech_winner_a", ""),
                    "source_mech_winner": assignment.get("source_mech_winner", ""),
                    "source_stage1_rank": assignment.get("source_stage1_rank", ""),
                    "source_stage1_trial_id": params.get("trial_id", assignment.get("source_stage1_trial_id", "")),
                    "ks0": params.get("ks0", record.get("ks_hat", "")),
                    "cs0": params.get("cs0", record.get("cs_hat", "")),
                    "ks_node_idx": params.get("ks_node_idx", assignment.get("ks_node_idx", "")),
                    "cs_node_idx": params.get("cs_node_idx", assignment.get("cs_node_idx", "")),
                    "nn_seed_bank_idx": params.get("nn_seed_bank_idx", assignment.get("source_stage1_best_seedbank", "")),
                    "nn_init_seed": params.get("nn_init_seed", assignment.get("source_stage1_best_initseed", "")),
                    "node_label": params.get("node_label", assignment.get("node_label", "")),
                    "train_loss": helper.metric_value(record, spec),
                    "assignment_file": assignment.get("_assignment_file", ""),
                }
            )


def _highlight_hover_text(
    helper: Any,
    record: dict[str, Any],
    assignment: dict[str, Any],
    spec: dict[str, Any],
    *,
    role: str,
) -> str:
    if role == "layer_b_candidate":
        head = (
            f"Layer B candidate B{int(assignment.get('candidate_b', 0)):02d}"
            f"<br>source_top_mech_winner_a={assignment.get('source_top_mech_winner_a', 'None')}"
        )
    else:
        head = f"Layer A input #{int(assignment.get('source_mech_winner', assignment.get('top_mech_winner_a', 0))):03d}"
    return (
        head
        +
        f"<br>source_mech_winner={assignment.get('source_mech_winner', 'None')}"
        f"<br>source_stage1_rank={assignment.get('source_stage1_rank', 'None')}"
        f"<br>assignment_file={assignment.get('_assignment_file', 'None')}"
        "<br>--- st1pl envelope source point ---<br>"
        + helper.hover_text(record, spec)
    )


def _write_html(
    path: Path,
    helper: Any,
    st1pl_result_path: Path,
    spec: dict[str, Any],
    kept_records: list[dict[str, Any]],
    layer_a_records: list[tuple[dict[str, Any], dict[str, Any]]],
    layer_b_records: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    normal_marker_size: float,
    layer_a_marker_size: float,
    candidate_marker_size: float,
) -> None:
    envelope = helper.best_seed_envelope(kept_records, spec)
    if envelope is None:
        raise RuntimeError("No best-seed envelope points are available.")

    xs = list(envelope["point_x"])
    ys = list(envelope["point_y"])
    zs = list(envelope["point_z"])
    hovertexts = list(envelope["point_text"])
    ks_raw = list(envelope["ks_raw"])
    cs_raw = list(envelope["cs_raw"])
    if not xs:
        raise RuntimeError("Best-seed envelope is empty.")

    layer_b_trial_ids = {_trial_id(record) for record, _assignment in layer_b_records}
    layer_a_only_records = [
        (record, assignment)
        for record, assignment in layer_a_records
        if _trial_id(record) not in layer_b_trial_ids
    ]

    def _scatter_points(
        rows: list[tuple[dict[str, Any], dict[str, Any]]],
        *,
        role: str,
    ) -> tuple[list[float], list[float], list[float], list[str], set[tuple[float, float, float]]]:
        px: list[float] = []
        py: list[float] = []
        pz: list[float] = []
        text: list[str] = []
        keys: set[tuple[float, float, float]] = set()
        for record, assignment in rows:
            params = _record_params(record)
            ks0 = helper.finite_float_or(helper.param_or(params, "ks0", helper.field_or(record, "ks_hat", math.nan)))
            cs0 = helper.finite_float_or(helper.param_or(params, "cs0", helper.field_or(record, "cs_hat", math.nan)))
            loss = float(helper.metric_value(record, spec))
            if ks0 <= 0.0 or cs0 <= 0.0 or loss <= 0.0:
                continue
            x = math.log10(ks0)
            y = math.log10(cs0)
            z = math.log10(loss)
            px.append(x)
            py.append(y)
            pz.append(z)
            text.append(_highlight_hover_text(helper, record, assignment, spec, role=role))
            keys.add((round(x, 12), round(y, 12), round(z, 12)))
        return px, py, pz, text, keys

    layer_a_x, layer_a_y, layer_a_z, layer_a_text, layer_a_point_keys = _scatter_points(
        layer_a_only_records,
        role="layer_a_input",
    )
    cand_x, cand_y, cand_z, cand_text, candidate_point_keys = _scatter_points(
        layer_b_records,
        role="layer_b_candidate",
    )
    highlighted_point_keys = layer_a_point_keys | candidate_point_keys

    highlighted_records: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for record, assignment in layer_a_only_records:
        highlighted_records.append(("layer_a_only", record, assignment))
    for record, assignment in layer_b_records:
        highlighted_records.append(("layer_b_candidate", record, assignment))

    normal_black_x: list[float] = []
    normal_black_y: list[float] = []
    normal_black_z: list[float] = []
    normal_black_text: list[str] = []
    for x, y, z, text in zip(xs, ys, zs, hovertexts):
        if (round(x, 12), round(y, 12), round(z, 12)) in highlighted_point_keys:
            continue
        normal_black_x.append(x)
        normal_black_y.append(y)
        normal_black_z.append(z)
        normal_black_text.append(text)

    _write_candidate_csv(path.with_suffix(".csv"), helper, spec, highlighted_records)

    # The returned Layer B set is a subset of Layer A; candidates are drawn in green
    # so they do not also appear as red points.
    candidate_trial_ids = {_trial_id(record) for record, _assignment in layer_b_records}

    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    zlo, zhi = min(zs), max(zs)
    zpad = max((zhi - zlo) * 0.12, 0.015)
    zfloor = zlo - zpad
    ztop = zhi + zpad
    xtickvals, xticktext = helper.sci_tick_spec(min(ks_raw), max(ks_raw))
    ytickvals, yticktext = helper.sci_tick_spec(min(cs_raw), max(cs_raw))
    ztickvals, zticktext = helper.sci_tick_spec(10.0**zlo, 10.0**zhi)

    traces: list[dict[str, Any]] = [envelope["surface"]]
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "best NN seed envelope points",
            "x": normal_black_x,
            "y": normal_black_y,
            "z": normal_black_z,
            "hovertext": normal_black_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": normal_marker_size,
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
            "name": f"Layer A inputs ({len(layer_a_x)})",
            "x": layer_a_x,
            "y": layer_a_y,
            "z": layer_a_z,
            "hovertext": layer_a_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": layer_a_marker_size,
                "opacity": 0.98,
                "color": "#d62728",
                "line": {"width": 0},
            },
        }
    )
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": f"Layer A + B candidates ({len(cand_x)})",
            "x": cand_x,
            "y": cand_y,
            "z": cand_z,
            "hovertext": cand_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": candidate_marker_size,
                "opacity": 1.0,
                "color": "#2ca02c",
                "line": {"width": 1.2, "color": "#111111"},
            },
        }
    )

    annotation_text = (
        f"st1pl source: {str(st1pl_result_path).replace(chr(92), chr(92) * 2)}"
        f"<br>red points entered prest2 Layer A; green points entered both Layer A and Layer B"
        f"<br>best-seed envelope points={len(xs)}; highlighted trial ids={len(candidate_trial_ids)}"
    )
    layout = {
        "title": "AFM05 prest2 Layer A/B selections on st1pl best-seed envelope",
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
  <title>AFM05 prest2 Layer A/B selections on st1pl best-seed envelope</title>
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
        description="Highlight AFM05 prest2 Layer A and Layer B selections on the st1pl best-seed envelope 3D plot."
    )
    parser.add_argument("--st1pl-result", type=Path, default=DEFAULT_ST1PL_RESULT)
    parser.add_argument("--layer-a-assignments-glob", default=str(DEFAULT_LAYER_A_ASSIGNMENTS_GLOB))
    parser.add_argument("--layer-b-assignments-glob", "--assignments-glob", default=str(DEFAULT_LAYER_B_ASSIGNMENTS_GLOB))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--html-name", default=DEFAULT_HTML_NAME)
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME)
    parser.add_argument("--z-display-max", type=float, default=1.0e-1)
    parser.add_argument("--normal-marker-size", type=float, default=3.0)
    parser.add_argument("--layer-a-marker-size", type=float, default=5.0)
    parser.add_argument("--candidate-marker-size", type=float, default=9.0)
    args = parser.parse_args(argv)

    helper = _load_stage1_vis_module()
    payload = _load_pickle(args.st1pl_result)
    if "trial_parameters" not in payload:
        raise KeyError(f"Stage1pluslight result does not contain 'trial_parameters': {args.st1pl_result}")
    helper.validate_complete_stage1plus_payload(payload, path=args.st1pl_result)

    all_records = [rec for rec in payload["trial_parameters"] if isinstance(rec, dict)]
    records = [
        rec
        for rec in all_records
        if not bool(helper.field_or(rec, "early_stopped", False))
        and not bool(helper.field_or(rec, "trial_failed", False))
        and bool(helper.field_or(rec, "is_viable", True))
    ]
    if not records:
        raise RuntimeError(f"No viable st1pl records found in {args.st1pl_result}")

    specs = helper.metric_specs(records)
    if len(specs) != 1:
        raise RuntimeError(f"Expected one AFM05 metric spec, got {len(specs)}")
    spec = specs[0]
    kept_records = [
        rec
        for rec in records
        if math.isfinite(helper.metric_value(rec, spec))
        and helper.metric_value(rec, spec) > 0.0
        and helper.metric_value(rec, spec) <= args.z_display_max
    ]
    if not kept_records:
        raise RuntimeError(f"No st1pl records remain after z cutoff <= {args.z_display_max:.3e}.")

    layer_a_assignments = _load_layer_assignments(
        args.layer_a_assignments_glob,
        rank_key="source_mech_winner",
        required=True,
    )
    layer_b_assignments = _load_layer_assignments(
        args.layer_b_assignments_glob,
        rank_key="source_top_mech_winner_a",
        required=False,
    )
    assignment_by_trial = {
        _assignment_trial_id(row): row for row in [*layer_a_assignments, *layer_b_assignments]
    }
    kept_by_trial = {_trial_id(rec): rec for rec in kept_records}
    missing = sorted(trial_id for trial_id in assignment_by_trial if trial_id not in kept_by_trial)
    if missing:
        raise RuntimeError(
            "Some Layer A/B source trial ids are missing from kept st1pl records: "
            + ", ".join(str(x) for x in missing)
        )
    layer_a_records = [
        (kept_by_trial[_assignment_trial_id(row)], row)
        for row in sorted(layer_a_assignments, key=lambda rec: int(rec.get("source_mech_winner", 10**9)))
    ]
    layer_b_records = [
        (kept_by_trial[_assignment_trial_id(row)], row)
        for row in sorted(layer_b_assignments, key=lambda rec: int(rec.get("source_top_mech_winner_a", 10**9)))
    ]

    out_dir = args.out_dir.resolve()
    html_path = out_dir / args.html_name
    csv_path = out_dir / args.csv_name
    _write_html(
        html_path,
        helper,
        args.st1pl_result.resolve(),
        spec,
        kept_records,
        layer_a_records,
        layer_b_records,
        normal_marker_size=args.normal_marker_size,
        layer_a_marker_size=args.layer_a_marker_size,
        candidate_marker_size=args.candidate_marker_size,
    )
    # _write_html writes the CSV next to the HTML by default; move it to the
    # caller-requested CSV name when these names differ.
    generated_csv_path = html_path.with_suffix(".csv")
    if generated_csv_path.resolve() != csv_path.resolve() and generated_csv_path.is_file():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        generated_csv_path.replace(csv_path)

    print(f"Loaded Layer A assignments: {len(layer_a_assignments)}")
    print(f"Loaded Layer B assignments: {len(layer_b_assignments)}")
    print(f"Highlighted Layer A-only envelope source points: {len(layer_a_records) - len(layer_b_records)}")
    print(f"Highlighted Layer B candidate envelope source points: {len(layer_b_records)}")
    print(f"Saved HTML: {html_path}")
    print(f"Saved candidate map CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
