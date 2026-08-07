"""Generate AFM06a stage2light entry and full-rollout diagnostics."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def _bootstrap_repo_root() -> Path:
    here = Path(__file__).resolve()
    for path in [here.parent, *here.parents]:
        if (path / "AFM06a" / "stage2light").is_dir():
            root = str(path)
            if root not in sys.path:
                sys.path.insert(0, root)
            return path
    raise RuntimeError(f"Could not locate HNODECB root from {here}")


REPO_ROOT = _bootstrap_repo_root()

from AFM06a.stage2light.runner.visualization._common import load_visualization_context
from AFM06a.stage2light.config import default_config
from AFM06a.stage2light.runner.visualization.plots import (
    plot_bar_fts_full_rollout,
    plot_bar_fts_pointwise_full_resolution,
    plot_x1_full_rollout,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AFM06a stage2light full-rollout visualizations."
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help=(
            "Self-contained AFM06a st2l archive. The final result is read from "
            "result/, its st1pl dependency from 'conditional dependency/', and "
            "outputs are written to visualization/."
        ),
    )
    return parser.parse_args()


def _archive_config(archive_dir: Path):
    archive = archive_dir.expanduser().resolve()
    result_path = archive / "result" / "afm_param_stage2light_06a_rank1.pkl"
    stage1_path = (
        archive
        / "conditional dependency"
        / "afm_param_stage1pluslight_06a.pkl"
    )
    if not result_path.is_file():
        raise FileNotFoundError(f"Archived stage2light result is missing: {result_path}")
    if not stage1_path.is_file():
        raise FileNotFoundError(f"Archived st1pl dependency is missing: {stage1_path}")
    return replace(
        default_config(REPO_ROOT),
        stage1_result_path=stage1_path,
        result_dir=result_path.parent,
        checkpoint_dir=archive / ".visualization_no_checkpoint",
        log_dir=archive / "logs",
        visualization_dir=archive / "visualization",
        archive_root=archive.parent,
    )


def main() -> None:
    args = _parse_args()
    if args.archive_dir is None:
        context = load_visualization_context()
    else:
        context = load_visualization_context(
            _archive_config(args.archive_dir),
            preserve_io_paths=True,
            enforce_current_contract=False,
        )
    plot_bar_fts_full_rollout(context)
    plot_bar_fts_pointwise_full_resolution(context)
    plot_x1_full_rollout(context)
    print("[done] AFM06a stage2light full-rollout visualizations generated")


if __name__ == "__main__":
    main()
