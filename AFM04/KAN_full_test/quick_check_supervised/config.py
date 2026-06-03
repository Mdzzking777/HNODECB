"""Configuration for the isolated KFT supervised quick check."""

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


def _env_width(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    parts = raw.replace("x", ",").replace("-", ",").split(",")
    width = tuple(int(p.strip()) for p in parts if p.strip())
    if len(width) < 2:
        raise ValueError(f"{name} must contain at least input and output widths")
    return width


@dataclass(frozen=True)
class QuickCheckSupervisedConfig:
    repo_root: Path
    pykan_root: Path
    dataset_root: Path
    output_root: Path
    log_dir: Path
    visualization_dir: Path
    result_dir: Path
    checkpoint_dir: Path
    random_search_log_dir: Path
    random_search_shard_log_dir: Path
    random_search_result_dir: Path
    random_search_checkpoint_dir: Path
    error_level: str
    auto_generate_dataset: bool
    window_mode: str
    arch_window_us: float
    modified_w0_start_us: float
    val_stride: int
    val_offset: int
    train_window_index: int
    seed: int
    dtype: str
    device: str
    width: tuple[int, ...]
    grid: int
    spline_k: int
    base_fun: str
    symbolic_enabled: bool
    auto_save: bool
    noise_scale: float
    affine_trainable: bool
    grid_eps: float
    grid_range_lo: float
    grid_range_hi: float
    adaptive_grid_enabled: bool
    grid_update_num: int
    start_grid_update_step: int
    stop_grid_update_step: int
    wpred_enabled: bool
    wpred_eps: float
    epochs: int
    checkpoint_every: int
    lr: float
    lr_adapt: bool
    lr_min: float
    lr_max: float
    lr_eta: float
    lr_ema_alpha: float
    lr_eps: float
    lr_target_init: float
    weight_decay: float
    gnn_learnable: bool
    optimizer_name: str
    optimizer_amsgrad: bool
    epoch_retry_max: int
    epoch_retry_lr_factor: float
    epoch_retry_lr_floor: float
    step_guard_enabled: bool
    step_retry_max: int
    step_retry_lr_factor: float
    step_max_loss_increase_frac: float
    recent_val_early_stop: bool
    recent_val_window: int
    good_enough_loss: float
    log_every: int
    fts_scale_mode: str
    ode_method: str
    ode_rtol: float
    ode_atol: float
    random_search_trials: int
    random_search_log_every: int
    random_search_topk: int
    random_search_use_val: bool
    random_search_true_random: bool
    random_search_shard_index: int
    random_search_shard_count: int
    warmstart_from_random_search: bool
    warmstart_fallback_random: bool
    random_search_warmstart_trial: int


def default_config(repo_root: str | Path | None = None) -> QuickCheckSupervisedConfig:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    repo_root = Path(repo_root)
    output_root = repo_root / "AFM04" / "KAN_full_test" / "quick_check_supervised"
    log_dir = output_root / "logs"
    visualization_dir = log_dir / "visualization"
    result_dir = output_root / "results"
    checkpoint_dir = output_root / "checkpoints"
    random_search_log_dir = log_dir / "random_search"
    random_search_shard_log_dir = random_search_log_dir / "shards"
    random_search_result_dir = result_dir / "random_search"
    random_search_checkpoint_dir = checkpoint_dir / "random_search"
    for path in (
        output_root,
        log_dir,
        visualization_dir,
        result_dir,
        checkpoint_dir,
        random_search_log_dir,
        random_search_shard_log_dir,
        random_search_result_dir,
        random_search_checkpoint_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    pykan_root = Path(os.environ.get("HNODECB_AFM04_KFT_QCS_PYKAN_ROOT", str(repo_root.parent / "pykan")))
    dataset_root = Path(os.environ.get("HNODECB_AFM04_KFT_QCS_DATASET_ROOT", str(repo_root / "AFM04" / "datasets")))

    optimizer_name = os.environ.get("HNODECB_AFM04_KFT_QCS_OPTIMIZER", "amsgrad").strip().lower() or "amsgrad"
    if optimizer_name not in ("adam", "amsgrad"):
        optimizer_name = "amsgrad"

    lr_min = _env_float("HNODECB_AFM04_KFT_QCS_LR_MIN", 1.0e-6)
    lr_max = _env_float("HNODECB_AFM04_KFT_QCS_LR_MAX", 1.0e-2)
    if lr_min <= 0.0:
        lr_min = 1.0e-6
    if lr_max < lr_min:
        lr_max = lr_min

    return QuickCheckSupervisedConfig(
        repo_root=repo_root,
        pykan_root=pykan_root,
        dataset_root=dataset_root,
        output_root=output_root,
        log_dir=log_dir,
        visualization_dir=visualization_dir,
        result_dir=result_dir,
        checkpoint_dir=checkpoint_dir,
        random_search_log_dir=random_search_log_dir,
        random_search_shard_log_dir=random_search_shard_log_dir,
        random_search_result_dir=random_search_result_dir,
        random_search_checkpoint_dir=random_search_checkpoint_dir,
        error_level=os.environ.get("HNODECB_AFM04_KFT_QCS_ERROR_LEVEL", "e0.0").strip() or "e0.0",
        auto_generate_dataset=_env_bool("HNODECB_AFM04_KFT_QCS_AUTOGEN_DATASET", False),
        window_mode=os.environ.get("HNODECB_AFM04_KFT_QCS_WINDOW_MODE", "w0").strip() or "w0",
        arch_window_us=_env_float("HNODECB_AFM04_KFT_QCS_WINDOW_US", 6.288e-6),
        modified_w0_start_us=_env_float("HNODECB_AFM04_KFT_QCS_MODIFIED_W0_START_US", 236.0),
        val_stride=max(1, _env_int("HNODECB_AFM04_KFT_QCS_VAL_STRIDE", 5)),
        val_offset=max(1, _env_int("HNODECB_AFM04_KFT_QCS_VAL_OFFSET", 2)),
        train_window_index=max(0, _env_int("HNODECB_AFM04_KFT_QCS_TRAIN_WINDOW_INDEX", 1)),
        seed=max(1, _env_int("HNODECB_AFM04_KFT_QCS_SEED", 20260406)),
        dtype=os.environ.get("HNODECB_AFM04_KFT_QCS_DTYPE", "float64").strip() or "float64",
        device=os.environ.get("HNODECB_AFM04_KFT_QCS_DEVICE", "cpu").strip() or "cpu",
        width=_env_width("HNODECB_AFM04_KFT_QCS_WIDTH", (3, 7, 1)),
        grid=max(1, _env_int("HNODECB_AFM04_KFT_QCS_GRID", 11)),
        spline_k=max(1, _env_int("HNODECB_AFM04_KFT_QCS_SPLINE_K", 3)),
        base_fun=os.environ.get("HNODECB_AFM04_KFT_QCS_BASE_FUN", "silu").strip() or "silu",
        symbolic_enabled=False,
        auto_save=False,
        noise_scale=_env_float("HNODECB_AFM04_KFT_QCS_NOISE_SCALE", 0.2),
        affine_trainable=False,
        grid_eps=_env_float("HNODECB_AFM04_KFT_QCS_GRID_EPS", 0.02),
        grid_range_lo=_env_float("HNODECB_AFM04_KFT_QCS_GRID_RANGE_LO", -1.0),
        grid_range_hi=_env_float("HNODECB_AFM04_KFT_QCS_GRID_RANGE_HI", 1.0),
        adaptive_grid_enabled=_env_bool("HNODECB_AFM04_KFT_QCS_ADAPTIVE_GRID", True),
        grid_update_num=max(1, _env_int("HNODECB_AFM04_KFT_QCS_GRID_UPDATE_NUM", 10)),
        start_grid_update_step=max(0, _env_int("HNODECB_AFM04_KFT_QCS_GRID_UPDATE_START", 0)),
        stop_grid_update_step=max(0, _env_int("HNODECB_AFM04_KFT_QCS_GRID_UPDATE_STOP", 50)),
        wpred_enabled=False,
        wpred_eps=_env_float("HNODECB_AFM04_KFT_QCS_WPRED_EPS", 1.0e-10),
        epochs=max(1, _env_int("HNODECB_AFM04_KFT_QCS_EPOCHS", 1000)),
        checkpoint_every=max(1, _env_int("HNODECB_AFM04_KFT_QCS_CHECKPOINT_EVERY", 25)),
        lr=_env_float("HNODECB_AFM04_KFT_QCS_LR", 1.0e-3),
        lr_adapt=_env_bool("HNODECB_AFM04_KFT_QCS_LR_ADAPT", True),
        lr_min=lr_min,
        lr_max=lr_max,
        lr_eta=_env_float("HNODECB_AFM04_KFT_QCS_LR_ETA", 0.05),
        lr_ema_alpha=_env_float("HNODECB_AFM04_KFT_QCS_LR_EMA", 0.97),
        lr_eps=_env_float("HNODECB_AFM04_KFT_QCS_LR_EPS", 1.0e-30),
        lr_target_init=_env_float("HNODECB_AFM04_KFT_QCS_LR_TARGET", float("nan")),
        weight_decay=_env_float("HNODECB_AFM04_KFT_QCS_WEIGHT_DECAY", 0.0),
        gnn_learnable=_env_bool("HNODECB_AFM04_KFT_QCS_GNN_LEARNABLE", True),
        optimizer_name=optimizer_name,
        optimizer_amsgrad=(optimizer_name == "amsgrad"),
        epoch_retry_max=max(0, _env_int("HNODECB_AFM04_KFT_QCS_EPOCH_RETRIES", 8)),
        epoch_retry_lr_factor=_env_float("HNODECB_AFM04_KFT_QCS_RETRY_LR_FACTOR", 0.3),
        epoch_retry_lr_floor=_env_float("HNODECB_AFM04_KFT_QCS_RETRY_LR_FLOOR", lr_min),
        step_guard_enabled=_env_bool("HNODECB_AFM04_KFT_QCS_STEP_GUARD", True),
        step_retry_max=max(0, _env_int("HNODECB_AFM04_KFT_QCS_STEP_RETRIES", 6)),
        step_retry_lr_factor=_env_float("HNODECB_AFM04_KFT_QCS_STEP_RETRY_LR_FACTOR", 0.3),
        step_max_loss_increase_frac=_env_float("HNODECB_AFM04_KFT_QCS_STEP_MAX_LOSS_INCREASE_FRAC", 0.05),
        recent_val_early_stop=_env_bool("HNODECB_AFM04_KFT_QCS_RECENT_VAL_EARLY_STOP", True),
        recent_val_window=max(1, _env_int("HNODECB_AFM04_KFT_QCS_RECENT_VAL_WINDOW", 100)),
        good_enough_loss=_env_float("HNODECB_AFM04_KFT_QCS_GOOD_ENOUGH_LOSS", 1.0e-10),
        log_every=max(1, _env_int("HNODECB_AFM04_KFT_QCS_LOG_EVERY", 1)),
        fts_scale_mode=os.environ.get("HNODECB_AFM04_KFT_QCS_FTS_SCALE", "rms").strip().lower() or "rms",
        ode_method=os.environ.get("HNODECB_AFM04_KFT_QCS_ODE_METHOD", "dopri5").strip() or "dopri5",
        ode_rtol=_env_float("HNODECB_AFM04_KFT_QCS_ODE_RTOL", 1.0e-10),
        ode_atol=_env_float("HNODECB_AFM04_KFT_QCS_ODE_ATOL", 1.0e-12),
        random_search_trials=max(1, _env_int("HNODECB_AFM04_KFT_QCS_RANDOM_SEARCH_TRIALS", 10000)),
        random_search_log_every=max(1, _env_int("HNODECB_AFM04_KFT_QCS_RANDOM_SEARCH_LOG_EVERY", 25)),
        random_search_topk=max(1, _env_int("HNODECB_AFM04_KFT_QCS_RANDOM_SEARCH_TOPK", 20)),
        random_search_use_val=_env_bool("HNODECB_AFM04_KFT_QCS_RANDOM_SEARCH_USE_VAL", False),
        random_search_true_random=_env_bool("HNODECB_AFM04_KFT_QCS_RANDOM_SEARCH_TRUE_RANDOM", False),
        random_search_shard_index=max(1, _env_int("HNODECB_AFM04_KFT_QCS_RANDOM_SEARCH_SHARD_INDEX", 1)),
        random_search_shard_count=max(1, _env_int("HNODECB_AFM04_KFT_QCS_RANDOM_SEARCH_SHARD_COUNT", 8)),
        warmstart_from_random_search=_env_bool("HNODECB_AFM04_KFT_QCS_WARMSTART_FROM_RANDOM_SEARCH", False),
        warmstart_fallback_random=_env_bool("HNODECB_AFM04_KFT_QCS_WARMSTART_FALLBACK_RANDOM", False),
        random_search_warmstart_trial=max(0, _env_int("HNODECB_AFM04_KFT_QCS_WARMSTART_TRIAL", 0)),
    )


__all__ = ["QuickCheckSupervisedConfig", "default_config"]
