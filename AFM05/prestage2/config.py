"""Configuration for AFM05 prestage2 (prest2) trial-run screening."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from AFM05.stage2light.schedule import load_stage2light_schedule


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    return raw not in ("0", "false", "False", "no", "NO")


@dataclass(frozen=True)
class Prestage2Config:
    repo_root: Path
    output_root: Path
    log_dir: Path
    shard_log_dir: Path
    visualization_dir: Path
    result_dir: Path
    candidate_run_root: Path
    stage1_input_path: Path
    shard_index: int
    shard_count: int
    layer_a_top_candidates: int
    layer_a_zone_filter: str
    layer_a_epochs: int
    layer_b_top_candidates: int
    layer_b_epochs: int
    stop_after_layer_a: bool
    checkpoint_every: int
    auto_generate_dataset: bool
    candidate_metric: str
    stage2_total_epochs: int
    stage2_adam_epochs: int
    stage2_lbfgs_epochs: int


def default_config(repo_root: str | Path | None = None) -> Prestage2Config:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root).resolve()

    output_root_raw = os.environ.get("HNODECB_AFM05_PREST2_OUTPUT_ROOT", "").strip()
    output_root = Path(output_root_raw).resolve() if output_root_raw != "" else (repo_root / "AFM05" / "prestage2")
    log_dir = output_root / "logs"
    shard_log_dir = log_dir / "window_per_shard"
    visualization_dir = output_root / "visualization"
    result_dir = output_root / "results"
    candidate_run_root = output_root / "candidate_runs"
    stage1_input_raw = os.environ.get("HNODECB_AFM05_PREST2_INPUT_PATH", "").strip()
    stage1_input_path = Path(stage1_input_raw).resolve() if stage1_input_raw != "" else (
        repo_root / "AFM05" / "stage1pluslight" / "results_afm" / "afm_param_stage1pluslight_05.pkl"
    )
    stage2_schedule = load_stage2light_schedule(repo_root)

    for path in (log_dir, shard_log_dir, visualization_dir, result_dir, candidate_run_root):
        path.mkdir(parents=True, exist_ok=True)

    stage2_total_epochs = max(1, _env_int("HNODECB_AFM05_STAGE2LIGHT_EPOCHS", stage2_schedule.epochs))
    stage2_adam_epochs = max(
        0,
        _env_int(
            "HNODECB_AFM05_PREST2_STAGE2_ADAM_EPOCHS",
            _env_int("HNODECB_AFM05_STAGE2LIGHT_ADAM_EPOCHS", stage2_schedule.adam_epochs),
        ),
    )
    stage2_adam_epochs = min(stage2_adam_epochs, stage2_total_epochs)
    stage2_lbfgs_epochs = max(0, stage2_total_epochs - stage2_adam_epochs)

    return Prestage2Config(
        repo_root=repo_root,
        output_root=output_root,
        log_dir=log_dir,
        shard_log_dir=shard_log_dir,
        visualization_dir=visualization_dir,
        result_dir=result_dir,
        candidate_run_root=candidate_run_root,
        stage1_input_path=stage1_input_path,
        shard_index=max(1, _env_int("HNODECB_AFM05_PREST2_SHARD_INDEX", 1)),
        shard_count=max(1, _env_int("HNODECB_AFM05_PREST2_SHARD_COUNT", 6)),
        layer_a_top_candidates=max(1, _env_int("HNODECB_AFM05_PREST2_LAYER_A_TOP_CANDIDATES", 30)),
        layer_a_zone_filter=(
            os.environ.get("HNODECB_AFM05_PREST2_LAYER_A_ZONE_FILTER", "all").strip().lower()
            or "all"
        ),
        layer_a_epochs=max(1, _env_int("HNODECB_AFM05_PREST2_LAYER_A_EPOCHS", 20)),
        layer_b_top_candidates=max(1, _env_int("HNODECB_AFM05_PREST2_LAYER_B_TOP_CANDIDATES", 10)),
        layer_b_epochs=max(1, _env_int("HNODECB_AFM05_PREST2_LAYER_B_EPOCHS", 20)),
        stop_after_layer_a=_env_bool("HNODECB_AFM05_PREST2_STOP_AFTER_LAYER_A", False),
        checkpoint_every=max(1, _env_int("HNODECB_AFM05_PREST2_CHECKPOINT_EVERY", 1)),
        auto_generate_dataset=_env_bool("HNODECB_AFM05_PREST2_AUTOGEN_DATASET", False),
        candidate_metric=(
            os.environ.get("HNODECB_AFM05_PREST2_CANDIDATE_METRIC", "final_train_loss").strip()
            or "final_train_loss"
        ),
        stage2_total_epochs=stage2_total_epochs,
        stage2_adam_epochs=stage2_adam_epochs,
        stage2_lbfgs_epochs=stage2_lbfgs_epochs,
    )


__all__ = ["Prestage2Config", "default_config"]
