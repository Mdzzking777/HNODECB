"""Visualization-only entry for AFM05 step3 post-analysis results.

The numeric post-analysis runner writes tables only. This script is the only
CLI entry that writes the three AFM05 step3 post-analysis PNG figures.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from post_analyze_afm05_step3_identifiability import (
    THIS_DIR,
    copy_latest,
    load_step3_result,
    log,
    make_sweep_thresholds,
    now_stamp,
    plot_eigen_spectrum,
    plot_projection_groups,
    plot_threshold_sweep,
    prepare_eigensystem,
    parse_float_list,
    projection_analysis_for_threshold,
    unique_thresholds,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-result",
        type=Path,
        default=THIS_DIR / "results" / "afm05_step3_identifiability_trained_latest.json",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=THIS_DIR / "results" / "post_analysis",
        help="Accepted for runner compatibility; figures are written to visualization-dir.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=THIS_DIR / "visualization",
    )
    parser.add_argument("--abs-floor", type=float, default=1.0e-8)
    parser.add_argument("--rel-factor", type=float, default=0.0)
    parser.add_argument("--thresholds", default="1e-12,1e-10,1e-8")
    parser.add_argument(
        "--group-taus",
        default="",
        help="Optional four comma-separated tau values shown in the 2x2 projection-groups figure.",
    )
    parser.add_argument("--threshold-log-min", type=float, default=0.0)
    parser.add_argument("--threshold-log-max", type=float, default=0.0)
    parser.add_argument("--threshold-points", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--stable-output-tag", action="store_true")
    parser.add_argument("--no-latest", action="store_true")
    parser.add_argument(
        "--plots",
        default="spectrum,groups,sweep",
        help="Comma-separated plot set: spectrum,groups,sweep,all.",
    )
    args = parser.parse_args()

    input_result = args.input_result.resolve()
    if not input_result.exists():
        raise FileNotFoundError(f"Step3 result not found: {input_result}")

    visualization_dir = args.visualization_dir.resolve()
    visualization_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading step3 result -> {input_result}")
    result = load_step3_result(input_result)
    labels, eigenvalues, eigenvectors = prepare_eigensystem(result)

    rank_label = str(result.get("rank_label") or f"rank{result.get('rank', 'unknown')}")
    parameter_set = str(result.get("parameter_set", "unknown"))
    max_abs_eval = float(max(abs(eigenvalues))) if eigenvalues.size else 0.0
    primary_threshold = float(max(args.abs_floor, args.rel_factor * max_abs_eval))
    sweep_thresholds = make_sweep_thresholds(args)
    group_taus = parse_float_list(args.group_taus) if args.group_taus.strip() else None
    if group_taus is not None and len(group_taus) != 4:
        raise ValueError(f"--group-taus requires exactly four values, got {len(group_taus)}")

    log(f"rank_label={rank_label} parameter_set={parameter_set} n={len(labels)}")
    log(f"max_abs_eigenvalue={max_abs_eval:.16e} primary_tau={primary_threshold:.16e}")

    requested_plots = {item.strip().lower() for item in str(args.plots).replace(";", ",").split(",") if item.strip()}
    if "all" in requested_plots:
        requested_plots = {"spectrum", "groups", "sweep"}
    valid_plots = {"spectrum", "groups", "sweep"}
    unknown_plots = sorted(requested_plots - valid_plots)
    if unknown_plots:
        raise ValueError(f"Unknown plot names: {unknown_plots}. Valid: {sorted(valid_plots)}")

    projection_rows: list[dict] = []
    if requested_plots & {"groups", "sweep"}:
        threshold_items = unique_thresholds(primary_threshold, sweep_thresholds)
        for threshold_name, threshold_value in threshold_items:
            log(f"analyzing threshold {threshold_name} tau={threshold_value:.3e}")
            _, rows = projection_analysis_for_threshold(
                labels=labels,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                threshold_name=threshold_name,
                threshold_value=threshold_value,
                top_k=args.top_k,
            )
            projection_rows.extend(rows)

    stamp = now_stamp()
    tag = args.output_tag.strip()
    if not tag:
        if args.stable_output_tag:
            tag = f"{parameter_set}_{rank_label}"
        else:
            tag = f"{parameter_set}_{rank_label}_{stamp}"
    base = f"afm05_step3_post_analysis_{tag}"

    log(f"writing figures -> {visualization_dir}")
    written_plots: dict[str, Path] = {}
    if "spectrum" in requested_plots:
        path = visualization_dir / f"{base}_eigen_spectrum.png"
        plot_eigen_spectrum(
            path,
            eigenvalues=eigenvalues,
            primary_threshold=primary_threshold,
            sweep_thresholds=sweep_thresholds,
            rank_label=rank_label,
        )
        written_plots["eigen_spectrum"] = path
    if "groups" in requested_plots:
        path = visualization_dir / f"{base}_null_projection_groups.png"
        plot_projection_groups(
            path,
            projection_rows=projection_rows,
            primary_threshold=primary_threshold,
            rank_label=rank_label,
            **({"display_taus": tuple(group_taus)} if group_taus is not None else {}),
        )
        written_plots["null_projection_groups"] = path
    if "sweep" in requested_plots:
        path = visualization_dir / f"{base}_threshold_sweep.png"
        plot_threshold_sweep(path, projection_rows=projection_rows, rank_label=rank_label)
        written_plots["threshold_sweep"] = path

    if not args.no_latest:
        latest_base = "afm05_step3_post_analysis_latest"
        if "eigen_spectrum" in written_plots and written_plots["eigen_spectrum"].is_file():
            copy_latest(written_plots["eigen_spectrum"], visualization_dir / f"{latest_base}_eigen_spectrum.png")
        if "null_projection_groups" in written_plots and written_plots["null_projection_groups"].is_file():
            copy_latest(written_plots["null_projection_groups"], visualization_dir / f"{latest_base}_null_projection_groups.png")
        if "threshold_sweep" in written_plots and written_plots["threshold_sweep"].is_file():
            copy_latest(written_plots["threshold_sweep"], visualization_dir / f"{latest_base}_threshold_sweep.png")

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
