"""Post-run visualization and categorized archiving for AFM06a stage2light."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from AFM06a.stage1pluslight.checkpoint import load_checkpoint
from AFM06a.stage2light.config import Stage2LightConfig


ARCHIVE_SUBDIRS = (
    "result",
    "checkpoint",
    "logs",
    "data",
    "conditional dependency",
    "visualization",
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _archive_run_dir(config: Stage2LightConfig) -> Path:
    base = config.archive_root / f"rank{config.input_rank}"
    if not base.exists() or not any(base.iterdir()):
        return base
    return config.archive_root / f"rank{config.input_rank}_{_timestamp()}"


def _layout(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {"root": root}
    for name in ARCHIVE_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


def _copy_file(source: Path, destination: Path) -> int:
    if not source.is_file():
        return 0
    shutil.copy2(source, destination / source.name)
    return 1


def _copy_glob(source: Path, pattern: str, destination: Path) -> int:
    count = 0
    for path in sorted(source.glob(pattern), key=lambda item: item.name.lower()):
        count += _copy_file(path, destination)
    return count


def run_all_visualizations(config: Stage2LightConfig) -> None:
    env = dict(os.environ)
    env["HNODECB_AFM06a_STAGE2LIGHT_INPUT_RANK"] = str(config.input_rank)
    env["HNODECB_AFM06a_STAGE2LIGHT_STAGE1_RESULT"] = str(config.stage1_result_path)
    env["HNODECB_AFM06a_STAGE2LIGHT_RESULT_DIR"] = str(config.result_dir)
    env["HNODECB_AFM06a_STAGE2LIGHT_CHECKPOINT_DIR"] = str(config.checkpoint_dir)
    env["HNODECB_AFM06a_STAGE2LIGHT_LOG_DIR"] = str(config.log_dir)
    env["HNODECB_AFM06a_STAGE2LIGHT_VISUALIZATION_DIR"] = str(config.visualization_dir)
    command = [
        sys.executable,
        "-m",
        "AFM06a.stage2light.runner.visualization.run_afm_stage2light_all_visualizations_06a",
    ]
    print(f"archive: running visualizations | {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(config.repo_root), env=env, check=True)


def archive_completed_stage2light_run(config: Stage2LightConfig) -> dict[str, Any]:
    payload = load_checkpoint(config.result_path)
    if payload is None or not bool(payload.get("complete")):
        raise RuntimeError("AFM06a stage2light archive requires a completed result")
    if int(payload.get("input_rank", -1)) != config.input_rank:
        raise RuntimeError("AFM06a stage2light result rank does not match archive configuration")
    run_all_visualizations(config)
    archive_dir = _archive_run_dir(config)
    layout = _layout(archive_dir)

    counts = {
        "result_files": _copy_file(config.result_path, layout["result"]),
        "checkpoint_files": _copy_file(config.checkpoint_path, layout["checkpoint"]),
        "log_files": _copy_file(config.run_log_path, layout["logs"]),
        "data_files": 0,
        "conditional_dependency_files": 0,
        "visualization_files": 0,
    }
    saved_config = payload.get("config", {})
    stage1_payload = load_checkpoint(config.stage1_result_path)
    if stage1_payload is None:
        raise RuntimeError("AFM06a st1pl dependency disappeared before archiving")
    dataset_root = Path(stage1_payload["config"]["dataset_root"])
    error_level = str(stage1_payload["config"]["error_level"])
    data_dir = dataset_root / error_level / "data"
    counts["data_files"] += _copy_file(data_dir / "ode_data_afm_dmt_hard.npz", layout["data"])
    counts["data_files"] += _copy_file(data_dir / "pert_df_afm_dmt_hard.npz", layout["data"])
    counts["conditional_dependency_files"] += _copy_file(
        config.stage1_result_path,
        layout["conditional dependency"],
    )
    counts["conditional_dependency_files"] += _copy_file(
        config.result_path,
        layout["conditional dependency"],
    )
    counts["visualization_files"] += _copy_glob(
        config.visualization_dir,
        "afm_param_stage2light_06a_*",
        layout["visualization"],
    )
    manifest = {
        "schema_version": 1,
        "stage": "AFM06a_stage2light",
        "archive_dir": str(archive_dir),
        "input_rank": config.input_rank,
        "source_trial": payload.get("source_trial"),
        "entry_policy": payload.get("entry_policy"),
        "epoch0_preparation_performed": payload.get("epoch0_preparation_performed"),
        "completed_epoch": payload.get("completed_epoch"),
        "stopped_early": payload.get("stopped_early", False),
        "stop_kind": payload.get("stop_kind", ""),
        "stop_epoch": payload.get("stop_epoch", 0),
        "stop_reason": payload.get("stop_reason", ""),
        "lbfgs_zero_step_state": payload.get("lbfgs_zero_step_state", {}),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in saved_config.items()},
        **counts,
    }
    (archive_dir / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        "archive complete | "
        f"dir={archive_dir} result={counts['result_files']} checkpoint={counts['checkpoint_files']} "
        f"logs={counts['log_files']} data={counts['data_files']} "
        f"conditional={counts['conditional_dependency_files']} viz={counts['visualization_files']}",
        flush=True,
    )
    return manifest


__all__ = ["archive_completed_stage2light_run", "run_all_visualizations"]
