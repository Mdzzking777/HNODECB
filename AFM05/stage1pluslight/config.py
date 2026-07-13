"""Configuration for AFM05 stage1pluslight."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    return raw not in ("0", "false", "False", "no", "NO")


@dataclass(frozen=True)
class Stage1PlusLightConfig:
    repo_root: Path
    dataset_root: Path
    result_dir: Path
    log_dir: Path
    error_level: str
    result_stem: str
    result_ext: str
    run_tag: str
    shard_index: int
    shard_count: int
    ks_node_count: int
    cs_node_count: int
    nn_seed_bank_size: int
    checkpoint_every: int
    resume_enabled: bool
    final_topk: int
    window_mode: str
    pixel_tag: str
    arch_window_us: float
    window_sample_stride: int
    val_stride: int
    val_offset: int
    run_seed: int
    use_multiple_shooting: bool
    ms_group_size: int
    ms_continuity_term: float
    zero_contact_override: bool
    trial_opt_enabled: bool
    trial_epochs: int
    trial_lr: float
    trial_lr_min: float
    trial_fd_rel_eps: float
    trial_fd_abs_eps: float
    step_guard_enabled: bool
    step_retry_max: int
    step_retry_lr_factor: float
    step_max_loss_frac: float
    early_stop_epoch: int
    early_stop_min_drop_frac: float
    fixed_num_hidden_layers: int
    fixed_num_hidden_nodes: int
    ode_solver: str
    ode_fallback_solver: str
    ode_rtol: float
    ode_atol: float
    ode_max_step: float

    @property
    def result_basename(self) -> str:
        stem = self.result_stem if self.run_tag == "" else f"{self.result_stem}_{self.run_tag}"
        base = f"{stem}{self.result_ext}"
        return f"{stem}_p{self.shard_index}{self.result_ext}" if self.shard_count > 1 else base

    @property
    def result_path(self) -> Path:
        return self.result_dir / self.result_basename

    @property
    def merged_result_basename(self) -> str:
        stem = self.result_stem if self.run_tag == "" else f"{self.result_stem}_{self.run_tag}"
        return f"{stem}{self.result_ext}"

    @property
    def merged_result_path(self) -> Path:
        return self.result_dir / self.merged_result_basename


def default_config(repo_root: str | Path | None = None) -> Stage1PlusLightConfig:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)
    dataset_root = repo_root / "AFM05" / "datasets"
    result_dir = repo_root / "AFM05" / "stage1pluslight" / "results_afm"
    log_dir = repo_root / "AFM05" / "stage1pluslight" / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return Stage1PlusLightConfig(
        repo_root=repo_root,
        dataset_root=dataset_root,
        result_dir=result_dir,
        log_dir=log_dir,
        error_level=os.environ.get("HNODECB_AFM05_ERROR_LEVEL", "e0.0").strip() or "e0.0",
        result_stem=os.environ.get("HNODECB_AFM05_STAGE1_RESULT_STEM", "afm_param_stage1pluslight_05").strip()
        or "afm_param_stage1pluslight_05",
        result_ext=os.environ.get("HNODECB_AFM05_STAGE1_RESULT_EXT", ".pkl").strip() or ".pkl",
        run_tag=os.environ.get("HNODECB_AFM05_STAGE1_RUN_TAG", "").strip(),
        shard_index=_env_int("HNODECB_AFM05_STAGE1_SHARD_INDEX", 1),
        shard_count=max(1, _env_int("HNODECB_AFM05_STAGE1_SHARD_COUNT", 1)),
        ks_node_count=max(1, _env_int("HNODECB_AFM05_STAGE1PLUS_GRID_KS_NODES", 20)),
        cs_node_count=max(1, _env_int("HNODECB_AFM05_STAGE1PLUS_GRID_CS_NODES", 50)),
        nn_seed_bank_size=max(
            1,
            _env_int(
                "HNODECB_AFM05_STAGE1PLUS_GRID_NN_SEEDS_PER_NODE",
                _env_int("HNODECB_AFM05_KAN_TEST_RANDOM_SEARCH_TRIALS", 200),
            ),
        ),
        checkpoint_every=max(1, _env_int("HNODECB_AFM05_STAGE1PLUS_CHECKPOINT_EVERY", 100)),
        resume_enabled=_env_bool("HNODECB_AFM05_STAGE1PLUS_RESUME", True),
        final_topk=max(1, _env_int("HNODECB_AFM05_STAGE1_FINAL_TOPK", 200)),
        window_mode=os.environ.get("HNODECB_AFM05_STAGE1PLUS_WINDOW_MODE", "window_05_initial").strip()
        or "window_05_initial",
        pixel_tag=os.environ.get(
            "HNODECB_AFM05_STAGE1PLUS_PIXEL_TAG",
            os.environ.get("HNODECB_AFM05_STAGE1PLUS_WINDOW_PIXEL_TAG", "184_152"),
        ).strip()
        or "184_152",
        arch_window_us=_env_float("HNODECB_AFM05_STAGE1PLUS_ARCH_WINDOW_US", 25.152e-6),
        window_sample_stride=max(1, _env_int("HNODECB_AFM05_STAGE1PLUS_WINDOW_SAMPLE_STRIDE", 16)),
        val_stride=max(1, _env_int("HNODECB_AFM05_STAGE1_VAL_STRIDE", 5)),
        val_offset=max(1, _env_int("HNODECB_AFM05_STAGE1_VAL_OFFSET", 2)),
        run_seed=max(1, _env_int("HNODECB_AFM05_STAGE1_RUN_SEED", 20260406)),
        use_multiple_shooting=_env_bool("HNODECB_AFM05_STAGE1_USE_MULTIPLE_SHOOTING", False),
        ms_group_size=max(1, _env_int("HNODECB_AFM05_STAGE1_MS_GROUP_SIZE", 10)),
        ms_continuity_term=_env_float("HNODECB_AFM05_STAGE1_MS_CONTINUITY_TERM", 1.0e-3),
        zero_contact_override=_env_bool("HNODECB_AFM05_STAGE1_ZERO_CONTACT_OVERRIDE", False),
        trial_opt_enabled=False,
        trial_epochs=max(0, _env_int("HNODECB_AFM05_STAGE1PLUS_TRIAL_EPOCHS", 0)),
        trial_lr=_env_float("HNODECB_AFM05_STAGE1PLUS_TRIAL_LR", 1.0e-3),
        trial_lr_min=_env_float("HNODECB_AFM05_STAGE1PLUS_TRIAL_LR_MIN", 1.0e-12),
        trial_fd_rel_eps=_env_float("HNODECB_AFM05_STAGE1PLUS_TRIAL_FD_REL_EPS", 1.0e-4),
        trial_fd_abs_eps=_env_float("HNODECB_AFM05_STAGE1PLUS_TRIAL_FD_ABS_EPS", 1.0e-8),
        step_guard_enabled=_env_bool("HNODECB_AFM05_STAGE1PLUS_TRIAL_STEP_GUARD", True),
        step_retry_max=max(0, _env_int("HNODECB_AFM05_STAGE1PLUS_TRIAL_STEP_RETRIES", 6)),
        step_retry_lr_factor=_env_float("HNODECB_AFM05_STAGE1PLUS_TRIAL_STEP_RETRY_LR_FACTOR", 0.1),
        step_max_loss_frac=_env_float("HNODECB_AFM05_STAGE1PLUS_TRIAL_STEP_MAX_LOSS_FRAC", 0.1),
        early_stop_epoch=max(1, _env_int("HNODECB_AFM05_STAGE1PLUS_TRIAL_EARLY_STOP_EPOCHS", 5)),
        early_stop_min_drop_frac=_env_float("HNODECB_AFM05_STAGE1PLUS_TRIAL_EARLY_STOP_MIN_DROP_FRAC", -1.0),
        fixed_num_hidden_layers=max(0, _env_int("HNODECB_AFM05_STAGE1_FIXED_HIDDEN_LAYERS", 1)),
        fixed_num_hidden_nodes=max(1, _env_int("HNODECB_AFM05_STAGE1_FIXED_HIDDEN_NODES", 7)),
        ode_solver=os.environ.get("HNODECB_AFM05_STAGE1PLUS_ODE_SOLVER", "Radau").strip() or "Radau",
        ode_fallback_solver=os.environ.get("HNODECB_AFM05_STAGE1PLUS_ODE_FALLBACK_SOLVER", "BDF").strip() or "BDF",
        ode_rtol=_env_float("HNODECB_AFM05_STAGE1PLUS_ODE_RTOL", 1.0e-8),
        ode_atol=_env_float("HNODECB_AFM05_STAGE1PLUS_ODE_ATOL", 1.0e-8),
        ode_max_step=_env_float("HNODECB_AFM05_STAGE1PLUS_ODE_MAX_STEP", 0.0),
    )


__all__ = ["Stage1PlusLightConfig", "default_config"]
