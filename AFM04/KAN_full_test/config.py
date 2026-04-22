"""Configuration for the isolated AFM04 KAN full functional test."""

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
class KANFullTestConfig:
    repo_root: Path
    pykan_root: Path
    dataset_root: Path
    output_root: Path
    log_dir: Path
    shard_log_dir: Path
    visualization_dir: Path
    result_dir: Path
    checkpoint_dir: Path
    random_search_log_dir: Path
    random_search_shard_log_dir: Path
    random_search_result_dir: Path
    random_search_checkpoint_dir: Path
    shard_index: int
    shard_count: int
    error_level: str
    auto_generate_dataset: bool
    window_mode: str
    arch_window_us: float
    val_stride: int
    val_offset: int
    seed: int
    dtype: str
    device: str
    width: tuple[int, int, int]
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
    train_wpred_enabled: bool
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
    grad_scale_kan: float
    grad_scale_gnn: float
    group_adapt: bool
    group_ema_alpha: float
    group_eta: float
    group_eps: float
    grad_scale_kan_min: float
    grad_scale_kan_max: float
    grad_scale_gnn_min: float
    grad_scale_gnn_max: float
    optimizer_name: str
    optimizer_amsgrad: bool
    plateau_early_stop: bool
    plateau_window: int
    plateau_tol: float
    recent_val_early_stop: bool
    recent_val_window: int
    recent_val_rel_current_frac: float
    good_enough_loss: float
    epoch_retry_max: int
    epoch_retry_lr_factor: float
    epoch_retry_lr_floor: float
    step_guard_enabled: bool
    step_retry_max: int
    step_retry_lr_factor: float
    step_max_loss_increase_frac: float
    random_search_trials: int
    random_search_log_every: int
    random_search_checkpoint_every: int
    random_search_topk: int
    random_search_use_val: bool
    random_search_true_random: bool
    random_search_fail_fast_enabled: bool
    random_search_state_guard_mult: float
    random_search_x1_guard_mult: float
    random_search_x2_guard_mult: float
    random_search_window_index: int
    random_search_subshard_index: int
    random_search_subshard_count: int
    train_window_index: int
    warmstart_from_random_search: bool
    warmstart_fallback_random: bool
    ode_method: str
    ode_rtol: float
    ode_atol: float
    log_every: int


def default_config(repo_root: str | Path | None = None) -> KANFullTestConfig:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)
    pykan_root = Path(os.environ.get("HNODECB_PYKAN_ROOT", str(repo_root.parent / "pykan")))
    output_root = repo_root / "AFM04" / "KAN_full_test"
    log_dir = output_root / "logs"
    shard_log_dir = log_dir / "window_per_shard"
    visualization_dir = log_dir / "visualization"
    result_dir = output_root / "results"
    checkpoint_dir = output_root / "checkpoints"
    random_search_log_dir = log_dir / "random_search"
    random_search_shard_log_dir = random_search_log_dir / "window_per_shard"
    random_search_result_dir = result_dir / "random_search"
    random_search_checkpoint_dir = checkpoint_dir / "random_search"
    dataset_root = repo_root / "AFM04" / "datasets"
    log_dir.mkdir(parents=True, exist_ok=True)
    shard_log_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    random_search_log_dir.mkdir(parents=True, exist_ok=True)
    random_search_shard_log_dir.mkdir(parents=True, exist_ok=True)
    random_search_result_dir.mkdir(parents=True, exist_ok=True)
    random_search_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer_name_raw = os.environ.get("HNODECB_AFM04_KAN_TEST_OPTIMIZER", "").strip().lower()
    if optimizer_name_raw == "":
        optimizer_name_raw = "amsgrad"
    if optimizer_name_raw not in ("adam", "amsgrad"):
        optimizer_name_raw = "amsgrad"

    lr_min = _env_float("HNODECB_AFM04_KAN_TEST_LR_MIN", 1.0e-6)
    lr_max = _env_float("HNODECB_AFM04_KAN_TEST_LR_MAX", 1.0e-2)
    if not (lr_min > 0.0):
        lr_min = 1.0e-6
    if not (lr_max >= lr_min):
        lr_max = max(lr_min, 1.0e-2)

    grad_scale_kan_min = _env_float("HNODECB_AFM04_KAN_TEST_GRAD_SCALE_KAN_MIN", 0.02)
    grad_scale_kan_max = _env_float("HNODECB_AFM04_KAN_TEST_GRAD_SCALE_KAN_MAX", 1.0)
    grad_scale_gnn_min = _env_float("HNODECB_AFM04_KAN_TEST_GRAD_SCALE_GNN_MIN", 0.2)
    grad_scale_gnn_max = _env_float("HNODECB_AFM04_KAN_TEST_GRAD_SCALE_GNN_MAX", 2.0)
    if not (grad_scale_kan_min > 0.0):
        grad_scale_kan_min = 0.02
    if not (grad_scale_kan_max >= grad_scale_kan_min):
        grad_scale_kan_max = max(grad_scale_kan_min, 1.0)
    if not (grad_scale_gnn_min > 0.0):
        grad_scale_gnn_min = 0.2
    if not (grad_scale_gnn_max >= grad_scale_gnn_min):
        grad_scale_gnn_max = max(grad_scale_gnn_min, 2.0)

    grad_scale_kan = _env_float("HNODECB_AFM04_KAN_TEST_GRAD_SCALE_KAN", 0.2)
    grad_scale_gnn = _env_float("HNODECB_AFM04_KAN_TEST_GRAD_SCALE_GNN", 1.0)
    if not (grad_scale_kan >= 0.0):
        grad_scale_kan = 1.0
    if not (grad_scale_gnn >= 0.0):
        grad_scale_gnn = 1.0
    grad_scale_kan = max(grad_scale_kan_min, min(grad_scale_kan, grad_scale_kan_max))
    grad_scale_gnn = max(grad_scale_gnn_min, min(grad_scale_gnn, grad_scale_gnn_max))

    random_search_state_guard_mult = _env_float("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_STATE_GUARD_MULT", 100.0)
    random_search_x1_guard_mult = _env_float(
        "HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_X1_GUARD_MULT",
        random_search_state_guard_mult,
    )
    random_search_x2_guard_mult = _env_float(
        "HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_X2_GUARD_MULT",
        random_search_state_guard_mult,
    )

    return KANFullTestConfig(
        repo_root=repo_root,
        pykan_root=pykan_root,
        dataset_root=dataset_root,
        output_root=output_root,
        log_dir=log_dir,
        shard_log_dir=shard_log_dir,
        visualization_dir=visualization_dir,
        result_dir=result_dir,
        checkpoint_dir=checkpoint_dir,
        random_search_log_dir=random_search_log_dir,
        random_search_shard_log_dir=random_search_shard_log_dir,
        random_search_result_dir=random_search_result_dir,
        random_search_checkpoint_dir=random_search_checkpoint_dir,
        shard_index=max(1, _env_int("HNODECB_AFM04_KAN_TEST_SHARD_INDEX", 1)),
        shard_count=max(1, _env_int("HNODECB_AFM04_KAN_TEST_SHARD_COUNT", 3)),
        error_level=os.environ.get("HNODECB_AFM04_KAN_TEST_ERROR_LEVEL", "e0.0").strip() or "e0.0",
        auto_generate_dataset=_env_bool("HNODECB_AFM04_KAN_TEST_AUTOGEN_DATASET", False),
        window_mode=os.environ.get("HNODECB_AFM04_KAN_TEST_WINDOW_MODE", "stage2_w123").strip() or "stage2_w123",
        arch_window_us=_env_float("HNODECB_AFM04_KAN_TEST_WINDOW_US", 5.0e-6),
        val_stride=max(1, _env_int("HNODECB_AFM04_KAN_TEST_VAL_STRIDE", 5)),
        val_offset=max(1, _env_int("HNODECB_AFM04_KAN_TEST_VAL_OFFSET", 2)),
        seed=max(1, _env_int("HNODECB_AFM04_KAN_TEST_SEED", 20260406)),
        dtype=os.environ.get("HNODECB_AFM04_KAN_TEST_DTYPE", "float64").strip() or "float64",
        device=os.environ.get("HNODECB_AFM04_KAN_TEST_DEVICE", "cpu").strip() or "cpu",
        width=(3, 7, 1),
        grid=max(1, _env_int("HNODECB_AFM04_KAN_TEST_GRID", 11)),
        spline_k=max(1, _env_int("HNODECB_AFM04_KAN_TEST_SPLINE_K", 3)),
        base_fun=os.environ.get("HNODECB_AFM04_KAN_TEST_BASE_FUN", "silu").strip() or "silu",
        symbolic_enabled=False,
        auto_save=False,
        noise_scale=_env_float("HNODECB_AFM04_KAN_TEST_NOISE_SCALE", 0.2),
        affine_trainable=False,
        grid_eps=_env_float("HNODECB_AFM04_KAN_TEST_GRID_EPS", 0.02),
        grid_range_lo=_env_float("HNODECB_AFM04_KAN_TEST_GRID_RANGE_LO", -1.0),
        grid_range_hi=_env_float("HNODECB_AFM04_KAN_TEST_GRID_RANGE_HI", 1.0),
        adaptive_grid_enabled=True,
        grid_update_num=max(1, _env_int("HNODECB_AFM04_KAN_TEST_GRID_UPDATE_NUM", 10)),
        start_grid_update_step=_env_int("HNODECB_AFM04_KAN_TEST_GRID_UPDATE_START", 0),
        stop_grid_update_step=max(0, _env_int("HNODECB_AFM04_KAN_TEST_GRID_UPDATE_STOP", 50)),
        wpred_enabled=_env_bool(
            "HNODECB_AFM04_KAN_TEST_WPRED",
            _env_bool("HNODECB_AFM04_KAN_TEST_GRID_UPDATE_WGRID", False),
        ),
        train_wpred_enabled=_env_bool(
            "HNODECB_AFM04_KAN_TEST_TRAIN_WPRED",
            _env_bool(
                "HNODECB_AFM04_KAN_TEST_WPRED",
                _env_bool("HNODECB_AFM04_KAN_TEST_GRID_UPDATE_WGRID", False),
            ),
        ),
        wpred_eps=_env_float(
            "HNODECB_AFM04_KAN_TEST_WPRED_EPS",
            _env_float("HNODECB_AFM04_KAN_TEST_GRID_UPDATE_WGRID_EPS", 1.0e-10),
        ),
        epochs=max(1, _env_int("HNODECB_AFM04_KAN_TEST_EPOCHS", 1000)),
        checkpoint_every=max(1, _env_int("HNODECB_AFM04_KAN_TEST_CHECKPOINT_EVERY", 5)),
        lr=_env_float("HNODECB_AFM04_KAN_TEST_LR", 1.0e-3),
        lr_adapt=_env_bool("HNODECB_AFM04_KAN_TEST_LR_ADAPT", True),
        lr_min=lr_min,
        lr_max=lr_max,
        lr_eta=_env_float("HNODECB_AFM04_KAN_TEST_LR_ETA", 0.05),
        lr_ema_alpha=_env_float("HNODECB_AFM04_KAN_TEST_LR_EMA", 0.97),
        lr_eps=_env_float("HNODECB_AFM04_KAN_TEST_LR_EPS", 1.0e-30),
        lr_target_init=_env_float("HNODECB_AFM04_KAN_TEST_LR_TARGET", float("nan")),
        weight_decay=_env_float("HNODECB_AFM04_KAN_TEST_WEIGHT_DECAY", 0.0),
        gnn_learnable=_env_bool("HNODECB_AFM04_KAN_TEST_GNN_LEARNABLE", True),
        grad_scale_kan=1.0,
        grad_scale_gnn=1.0,
        group_adapt=_env_bool("HNODECB_AFM04_KAN_TEST_GROUP_ADAPT", False),
        group_ema_alpha=_env_float("HNODECB_AFM04_KAN_TEST_GROUP_EMA", 0.90),
        group_eta=_env_float("HNODECB_AFM04_KAN_TEST_GROUP_ETA", 0.15),
        group_eps=_env_float("HNODECB_AFM04_KAN_TEST_GROUP_EPS", 1.0e-30),
        grad_scale_kan_min=grad_scale_kan_min,
        grad_scale_kan_max=grad_scale_kan_max,
        grad_scale_gnn_min=grad_scale_gnn_min,
        grad_scale_gnn_max=grad_scale_gnn_max,
        optimizer_name=optimizer_name_raw,
        optimizer_amsgrad=(optimizer_name_raw == "amsgrad"),
        plateau_early_stop=_env_bool("HNODECB_AFM04_KAN_TEST_PLATEAU_EARLY_STOP", False),
        plateau_window=max(1, _env_int("HNODECB_AFM04_KAN_TEST_PLATEAU_WINDOW", 5)),
        plateau_tol=_env_float("HNODECB_AFM04_KAN_TEST_PLATEAU_TOL", 1.0e-6),
        recent_val_early_stop=_env_bool("HNODECB_AFM04_KAN_TEST_RECENT_VAL_EARLY_STOP", True),
        recent_val_window=max(1, _env_int("HNODECB_AFM04_KAN_TEST_RECENT_VAL_WINDOW", 30)),
        recent_val_rel_current_frac=_env_float("HNODECB_AFM04_KAN_TEST_RECENT_VAL_REL_CURRENT_FRAC", 0.01),
        good_enough_loss=_env_float("HNODECB_AFM04_KAN_TEST_GOOD_ENOUGH_LOSS", 1.0e-10),
        epoch_retry_max=max(0, _env_int("HNODECB_AFM04_KAN_TEST_EPOCH_RETRIES", 8)),
        epoch_retry_lr_factor=_env_float("HNODECB_AFM04_KAN_TEST_RETRY_LR_FACTOR", 0.3),
        epoch_retry_lr_floor=_env_float("HNODECB_AFM04_KAN_TEST_RETRY_LR_FLOOR", lr_min),
        step_guard_enabled=_env_bool("HNODECB_AFM04_KAN_TEST_STEP_GUARD", True),
        step_retry_max=max(0, _env_int("HNODECB_AFM04_KAN_TEST_STEP_RETRIES", 6)),
        step_retry_lr_factor=_env_float("HNODECB_AFM04_KAN_TEST_STEP_RETRY_LR_FACTOR", 0.3),
        step_max_loss_increase_frac=_env_float("HNODECB_AFM04_KAN_TEST_STEP_MAX_LOSS_INCREASE_FRAC", 0.05),
        random_search_trials=max(1, _env_int("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_TRIALS", 1000)),
        random_search_log_every=max(1, _env_int("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_LOG_EVERY", 25)),
        random_search_checkpoint_every=max(1, _env_int("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_CHECKPOINT_EVERY", 100)),
        random_search_topk=max(1, _env_int("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_TOPK", 20)),
        random_search_use_val=_env_bool("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_USE_VAL", False),
        random_search_true_random=_env_bool("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_TRUE_RANDOM", False),
        random_search_fail_fast_enabled=_env_bool("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_FAIL_FAST", True),
        random_search_state_guard_mult=random_search_state_guard_mult,
        random_search_x1_guard_mult=random_search_x1_guard_mult,
        random_search_x2_guard_mult=random_search_x2_guard_mult,
        random_search_window_index=max(0, _env_int("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_WINDOW_INDEX", 0)),
        random_search_subshard_index=max(1, _env_int("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_SUBSHARD_INDEX", 1)),
        random_search_subshard_count=max(1, _env_int("HNODECB_AFM04_KAN_TEST_RANDOM_SEARCH_SUBSHARD_COUNT", 5)),
        train_window_index=max(0, _env_int("HNODECB_AFM04_KAN_TEST_TRAIN_WINDOW_INDEX", 0)),
        warmstart_from_random_search=_env_bool("HNODECB_AFM04_KAN_TEST_WARMSTART_FROM_RANDOM_SEARCH", False),
        warmstart_fallback_random=_env_bool("HNODECB_AFM04_KAN_TEST_WARMSTART_FALLBACK_RANDOM", False),
        ode_method=os.environ.get("HNODECB_AFM04_KAN_TEST_ODE_METHOD", "dopri5").strip() or "dopri5",
        ode_rtol=_env_float("HNODECB_AFM04_KAN_TEST_ODE_RTOL", 1.0e-10),
        ode_atol=_env_float("HNODECB_AFM04_KAN_TEST_ODE_ATOL", 1.0e-12),
        log_every=max(1, _env_int("HNODECB_AFM04_KAN_TEST_LOG_EVERY", 1)),
    )


__all__ = ["KANFullTestConfig", "default_config"]
