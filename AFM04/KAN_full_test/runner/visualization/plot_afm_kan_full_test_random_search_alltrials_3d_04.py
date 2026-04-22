from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "runner").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
RESULT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "results" / "random_search"
OUT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "visualization"
Z_DISPLAY_MAX = 1.0e-1
FERR_CAP = 200.0
FERR_OVERFLOW = FERR_CAP + 1.0


def fmt_e_html(x: Any) -> str:
    return f"{float(x):.6e}" if isinstance(x, (int, float)) and math.isfinite(float(x)) else "None"


def fmt_f_html(x: Any, *, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}" if isinstance(x, (int, float)) and math.isfinite(float(x)) else "None"


def sci_tick_spec(lo: float, hi: float) -> tuple[list[int], list[str]]:
    lo_exp = math.floor(math.log10(lo))
    hi_exp = math.ceil(math.log10(hi))
    vals = list(range(lo_exp, hi_exp + 1))
    texts = [f"1e{exp}" for exp in vals]
    return vals, texts


def empirical_quantile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        raise RuntimeError("Cannot take quantile of empty vector.")
    idx = max(0, min(math.ceil(p * len(sorted_vals)) - 1, len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_shard_payloads(result_dir: Path = RESULT_DIR) -> list[dict[str, Any]]:
    merged_path = result_dir / "kan_full_test_random_search_merged_summary.json"
    merged = load_json(merged_path)
    shards = merged.get("shards", [])
    if not shards:
        raise RuntimeError(f"No shard entries found in {merged_path}")

    payloads: list[dict[str, Any]] = []
    for shard in shards:
        summary_path = Path(shard["summary_path"])
        summary = load_json(summary_path)
        history_path = Path(summary["history_path"])
        records = load_json(history_path)
        payloads.append(
            {
                "merged": merged,
                "shard": shard,
                "summary": summary,
                "records": records,
            }
        )
    return payloads


def stage_title(role: str) -> str:
    if role == "first_contact":
        return "W1"
    if role == "max_x1_pp_change":
        return "W2"
    if role == "tail_stable":
        return "W3"
    return role or "W?"


def ranking_metric_name(summary: dict[str, Any]) -> str:
    metric = str(summary.get("ranking_metric", "val_loss")).strip()
    return metric or "val_loss"


def metric_value(rec: dict[str, Any]) -> float:
    v = rec.get("ranking_loss", rec.get("val_loss", float("nan")))
    return float(v) if isinstance(v, (int, float)) else float("nan")


def metric_train_value(rec: dict[str, Any]) -> float:
    v = rec.get("train_loss", float("nan"))
    return float(v) if isinstance(v, (int, float)) else float("nan")


def nn_err_value(rec: dict[str, Any]) -> float:
    v = rec.get("nn_err", rec.get("val_nn_err", rec.get("train_nn_err", float("nan"))))
    return float(v) if isinstance(v, (int, float)) else float("nan")


def x1_err_value(rec: dict[str, Any]) -> float:
    v = rec.get("x1_rec", rec.get("val_x1_rec", rec.get("train_x1_rec", float("nan"))))
    return float(v) if isinstance(v, (int, float)) else float("nan")


def x3_err_value(rec: dict[str, Any]) -> float:
    v = rec.get("x3_rec", rec.get("val_x3_rec", rec.get("train_x3_rec", float("nan"))))
    return float(v) if isinstance(v, (int, float)) else float("nan")


def gnn_value(rec: dict[str, Any]) -> float:
    for key in ("g_nn", "init_gnn"):
        v = rec.get(key, float("nan"))
        if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0:
            return float(v)
    return float("nan")


def hover_text(rec: dict[str, Any], role: str, ranking_metric: str) -> str:
    lines = [
        f"trial={rec.get('trial', -1)}"
        + f"<br>window={role}"
        + f"<br>ranking_metric={ranking_metric}"
        + f"<br>ranking_loss={fmt_e_html(rec.get('ranking_loss', rec.get('val_loss')))}"
        + f"<br>train_loss={fmt_e_html(rec.get('train_loss'))}"
    ]
    if math.isfinite(float(rec.get("val_loss", float("nan")))):
        lines.append(f"<br>val_loss={fmt_e_html(rec.get('val_loss'))}")
    lines.extend(
        [
            f"<br>seed={rec.get('seed', 'None')}",
            f"<br>g_nn={fmt_e_html(rec.get('g_nn'))}",
            f"<br>init_gnn={fmt_e_html(rec.get('init_gnn'))}",
            f"<br>x1 rec err={fmt_f_html(x1_err_value(rec))}%",
            f"<br>x3 rec err={fmt_f_html(x3_err_value(rec))}%",
            f"<br>F_contact err={fmt_f_html(nn_err_value(rec))}%",
            f"<br>dt_total={fmt_f_html(rec.get('dt_total_sec'), digits=3)} s",
            f"<br>step1 init={fmt_f_html(rec.get('dt_step1_model_init_sec'), digits=3)} s",
            f"<br>step2 grid={fmt_f_html(rec.get('dt_step2_grid_update_sec'), digits=3)} s",
            f"<br>step3 gnn={fmt_f_html(rec.get('dt_step3_gnn_init_sec'), digits=3)} s",
            f"<br>step4 train={fmt_f_html(rec.get('dt_step4_train_eval_sec'), digits=3)} s",
            f"<br>step5 val={fmt_f_html(rec.get('dt_step5_val_eval_sec'), digits=3)} s",
        ]
    )
    return "".join(lines)


def write_plot_html(
    *,
    html_path: Path,
    history_path: Path,
    role: str,
    kept_records: list[dict[str, Any]],
    best_trial_rec: dict[str, Any] | None,
    excluded_count: int,
    z_display_max: float,
    ranking_metric: str,
) -> None:
    xs_raw = [int(rec["trial"]) for rec in kept_records]
    ys_raw = [gnn_value(rec) for rec in kept_records]
    zs_raw = [metric_value(rec) for rec in kept_records]
    ferrs = [nn_err_value(rec) for rec in kept_records]
    hovertexts = [hover_text(rec, role, ranking_metric) for rec in kept_records]

    ferrs_color = [
        float("nan") if not math.isfinite(f) else (FERR_OVERFLOW if f > FERR_CAP else max(f, 0.0))
        for f in ferrs
    ]

    xs = [float(x) for x in xs_raw]
    ys = [math.log10(y) for y in ys_raw]
    zs = [math.log10(z) for z in zs_raw]

    n_highlight = max(1, math.ceil(0.001 * len(kept_records)))
    highlight_order = sorted(range(len(kept_records)), key=lambda i: zs_raw[i])[:n_highlight]
    highlight_mask = [False] * len(kept_records)
    for idx in highlight_order:
        highlight_mask[idx] = True

    xs_base = [x for x, hi in zip(xs, highlight_mask) if not hi]
    ys_base = [y for y, hi in zip(ys, highlight_mask) if not hi]
    zs_base = [z for z, hi in zip(zs, highlight_mask) if not hi]
    ferrs_base = [f for f, hi in zip(ferrs_color, highlight_mask) if not hi]
    hovertexts_base = [h for h, hi in zip(hovertexts, highlight_mask) if not hi]

    xs_high = [x for x, hi in zip(xs, highlight_mask) if hi]
    ys_high = [y for y, hi in zip(ys, highlight_mask) if hi]
    zs_high = [z for z, hi in zip(zs, highlight_mask) if hi]
    ferrs_high = [f for f, hi in zip(ferrs_color, highlight_mask) if hi]
    hovertexts_high = [h for h, hi in zip(hovertexts, highlight_mask) if hi]

    sorted_zs_raw = sorted(zs_raw)
    zfocus_lo_raw = empirical_quantile(sorted_zs_raw, 0.01)
    zfocus_hi_raw = empirical_quantile(sorted_zs_raw, 0.995)
    zfocus_lo = math.log10(zfocus_lo_raw)
    zfocus_hi = math.log10(zfocus_hi_raw)
    zpad = max((zfocus_hi - zfocus_lo) * 0.12, 0.015)
    highlight_lo_raw = 10.0 ** min(zs_high) if zs_high else zfocus_lo_raw
    highlight_lo = math.log10(highlight_lo_raw)
    highlight_pad = max((zfocus_lo - highlight_lo) * 0.08, 0.01)
    zfloor = min(zfocus_lo - zpad, highlight_lo - highlight_pad)
    ztop = zfocus_hi + zpad
    z_hidden_low = sum(1 for z in zs_raw if z < zfocus_lo_raw)
    z_hidden_high = sum(1 for z in zs_raw if z > zfocus_hi_raw)

    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)

    ytickvals, yticktext = sci_tick_spec(min(ys_raw), max(ys_raw))
    ztickvals, zticktext = sci_tick_spec(zfocus_lo_raw, zfocus_hi_raw)
    ferr_tickvals = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, FERR_OVERFLOW]
    ferr_ticktext = ["0%", "25%", "50%", "75%", "100%", "125%", "150%", "175%", "200%", ">200%"]

    best_x = None
    best_y = None
    best_z = None
    best_hover = None
    if best_trial_rec is not None:
        best_gnn = gnn_value(best_trial_rec)
        best_val = metric_value(best_trial_rec)
        if best_gnn > 0.0 and best_val > 0.0:
            best_x = float(best_trial_rec["trial"])
            best_y = math.log10(best_gnn)
            best_z = math.log10(best_val)
            best_hover = hover_text(best_trial_rec, role, ranking_metric)

    annotation_text = (
        "source: " + str(history_path).replace("\\", "\\\\") +
        f"<br>x=trial index, y=g_nn (log10), z={ranking_metric} (log10)" +
        "<br>color uses stored F_contact err; excluded " + str(excluded_count) +
        f" trials with {ranking_metric} > {z_display_max:.1e} or <= 0" +
        "<br>z-focus uses q01..q99.5 and hides " + str(z_hidden_low) +
        " low / " + str(z_hidden_high) + " high outliers from the visible z span"
    )

    traces: list[str] = []
    traces.append(
        "const traceTrials = " + json.dumps(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"trials with {ranking_metric} <= {z_display_max:.1e}",
                "x": xs_base,
                "y": ys_base,
                "z": zs_base,
                "hovertext": hovertexts_base,
                "hovertemplate": "%{hovertext}<extra></extra>",
                "marker": {
                    "size": 3,
                    "opacity": 0.82,
                    "color": ferrs_base,
                    "cmin": 0.0,
                    "cmax": FERR_OVERFLOW,
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
                        "title": "F_contact err %",
                        "tickmode": "array",
                        "tickvals": ferr_tickvals,
                        "ticktext": ferr_ticktext,
                    },
                    "line": {"width": 0},
                },
            }
        )
        + ";"
    )
    traces.append(
        "const traceHighlights = " + json.dumps(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"lowest 0.1% {ranking_metric} (black outline)",
                "x": xs_high,
                "y": ys_high,
                "z": zs_high,
                "hovertext": hovertexts_high,
                "hovertemplate": "%{hovertext}<extra></extra>",
                "marker": {
                    "size": 5,
                    "opacity": 1.0,
                    "color": ferrs_high,
                    "cmin": 0.0,
                    "cmax": FERR_OVERFLOW,
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
        + ";"
    )

    trace_names = ["traceTrials", "traceHighlights"]
    if best_x is not None and best_y is not None and best_z is not None and best_hover is not None:
        traces.append(
            "const traceBest = " + json.dumps(
                {
                    "type": "scatter3d",
                    "mode": "markers+text",
                    "name": "best trial",
                    "x": [best_x],
                    "y": [best_y],
                    "z": [best_z],
                    "text": ["best"],
                    "textposition": "bottom center",
                    "hovertext": [best_hover],
                    "hovertemplate": "%{hovertext}<extra></extra>",
                    "marker": {"size": 7, "symbol": "diamond", "color": "#111111"},
                }
            )
            + ";"
        )
        trace_names.append("traceBest")

    layout = {
        "title": f"AFM04 KAN full test GS: trial vs g_nn vs {ranking_metric} ({stage_title(role)})",
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
            "xaxis": {"title": "trial", "range": [xlo, xhi]},
            "yaxis": {
                "title": "g_nn",
                "tickmode": "array",
                "tickvals": ytickvals,
                "ticktext": yticktext,
                "range": [ylo, yhi],
            },
            "zaxis": {
                "title": ranking_metric,
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
  <title>AFM04 KAN full test GS 3D</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #ffffff; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    {"".join(traces)}
    const layout = {json.dumps(layout)};
    Plotly.newPlot('plot', [{", ".join(trace_names)}], layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = load_shard_payloads()
    saved: list[Path] = []

    for payload in payloads:
        shard = payload["shard"]
        summary = payload["summary"]
        records = payload["records"]
        role = str(shard.get("role", "window"))
        ranking_metric = ranking_metric_name(summary)

        kept_records = [
            rec for rec in records
            if math.isfinite(metric_value(rec))
            and metric_value(rec) > 0.0
            and metric_value(rec) <= Z_DISPLAY_MAX
            and math.isfinite(gnn_value(rec))
            and gnn_value(rec) > 0.0
        ]
        excluded_count = len(records) - len(kept_records)
        if not kept_records:
            raise RuntimeError(f"No records remain after filtering for shard {shard.get('shard_index')}")

        slug = stage_title(role).lower()
        html_path = OUT_DIR / f"kan_full_test_random_search_alltrials_trial_loggnn_{ranking_metric}_3d_{slug}.html"
        write_plot_html(
            html_path=html_path,
            history_path=Path(summary["history_path"]),
            role=role,
            kept_records=kept_records,
            best_trial_rec=summary.get("best_trial"),
            excluded_count=excluded_count,
            z_display_max=Z_DISPLAY_MAX,
            ranking_metric=ranking_metric,
        )
        print(f"Saved 3D HTML to: {html_path}")
        print(f"Trials plotted for {stage_title(role)}: {len(kept_records)}")
        print(f"Trials excluded by z cutoff for {stage_title(role)}: {excluded_count}")
        saved.append(html_path)

    print(f"Saved {len(saved)} KFT random-search 3D plots.")


if __name__ == "__main__":
    main()
