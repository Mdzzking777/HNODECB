"""Configuration for the standalone AFM04 Python stage2light library."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from AFM04.stage1pluslight.grid import CS_BOUNDS, KS_BOUNDS
from AFM04.stage2light.schedule import load_stage2light_schedule


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
class Stage2LightConfig:
    repo_root: Path
    pykan_root: Path
    dataset_root: Path
    output_root: Path
    log_dir: Path
    shard_log_dir: Path
    visualization_dir: Path
    result_dir: Path
    checkpoint_dir: Path
    stage1_input_path: Path
    prestage2_input_path: Path
    shard_index: int
    shard_count: int
    error_level: str
    auto_generate_dataset: bool
    window_mode: str
    arch_window_us: float
    val_stride: int
    val_offset: int
    val_eval_mode: str
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
    soft_mask_enabled: bool
    soft_mask_trainable: bool
    soft_mask_s0_a0: float
    soft_mask_s0_min_a0: float
    soft_mask_s0_max_a0: float
    soft_mask_alpha_a0: float
    soft_mask_alpha_min_a0: float
    soft_mask_alpha_max_a0: float
    epochs: int
    adam_epochs: int
    lbfgs_steps: int
    checkpoint_every: int
    lr: float
    lr_adapt: bool
    lr_min: float
    lr_max: float
    lr_eta: float
    lr_ema_alpha: float
    lr_eps: float
    lr_target_init: float
    lr_adapt_up_only: bool
    weight_decay: float
    gnn_learnable: bool
    group_adapt: bool
    group_ema_alpha: float
    group_eta: float
    group_eps: float
    optimizer_name: str
    optimizer_amsgrad: bool
    plateau_early_stop: bool
    plateau_window: int
    plateau_tol: float
    recent_val_early_stop: bool
    recent_val_window: int
    recent_val_compare_gap: int
    recent_val_rel_current_frac: float
    good_enough_loss: float
    epoch_retry_max: int
    epoch_retry_lr_factor: float
    epoch_retry_lr_floor: float
    step_controller: str
    step_guard_enabled: bool
    step_retry_max: int
    step_retry_lr_factor: float
    step_retry_reset_lr_each_epoch: bool
    step_guard_validate_val: bool
    step_max_loss_increase_frac: float
    armijo_c1: float
    backtrack_shrink: float
    backtrack_max: int
    backtrack_min_alpha: float
    lbfgs_enabled: bool
    lbfgs_strong_wolfe: bool
    lbfgs_lr: float
    lbfgs_max_iter: int
    lbfgs_max_eval: int
    lbfgs_history_size: int
    lbfgs_tolerance_grad: float
    lbfgs_tolerance_change: float
    lbfgs_ys_threshold: float
    train_window_index: int
    warmstart_from_stage1: bool
    warmstart_fallback_random: bool
    warmstart_source: str
    stage1_input_rank: int
    stage1_input_trial_id: int
    stage1_input_mech_winner: int
    prestage2_input_candidate: int
    resume_from_checkpoint: bool
    mech_parameterization: str
    ks_lo: float
    ks_hi: float
    cs_lo: float
    cs_hi: float
    ode_method: str
    ode_rtol: float
    ode_atol: float
    log_every: int


def default_config(repo_root: str | Path | None = None) -> Stage2LightConfig:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)
    pykan_root = Path(os.environ.get("HNODECB_PYKAN_ROOT", str(repo_root.parent / "pykan")))
    output_root_raw = os.environ.get("HNODECB_AFM04_STAGE2LIGHT_OUTPUT_ROOT", "").strip()
    output_root = Path(output_root_raw) if output_root_raw != "" else (repo_root / "AFM04" / "stage2light")
    log_dir = output_root / "logs"
    shard_log_dir = log_dir / "window_per_shard"
    visualization_dir = log_dir / "visualization"
    result_dir = output_root / "results"
    checkpoint_dir = output_root / "checkpoints"
    dataset_root = repo_root / "AFM04" / "datasets"
    default_stage1_input = repo_root / "AFM04" / "stage1pluslight" / "results_afm" / "afm_param_stage1pluslight_04.pkl"
    default_prestage2_input = repo_root / "AFM04" / "prestage2" / "results" / "afm_prest2_04_candidates_b.pkl"

    log_dir.mkdir(parents=True, exist_ok=True)
    shard_log_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stage1_input_raw = os.environ.get("HNODECB_AFM04_STAGE2LIGHT_INPUT_PATH", "").strip()
    stage1_input_path = Path(stage1_input_raw) if stage1_input_raw != "" else default_stage1_input
    prestage2_input_raw = os.environ.get("HNODECB_AFM04_STAGE2LIGHT_PREST2_INPUT_PATH", "").strip()
    prestage2_input_path = Path(prestage2_input_raw) if prestage2_input_raw != "" else default_prestage2_input
    warmstart_source = os.environ.get("HNODECB_AFM04_STAGE2LIGHT_WARMSTART_SOURCE", "prest2").strip().lower() or "prest2"
    if warmstart_source not in ("stage1", "prest2"):
        warmstart_source = "prest2"

    optimizer_name_raw = os.environ.get("HNODECB_AFM04_STAGE2LIGHT_OPTIMIZER", "").strip().lower()
    if optimizer_name_raw == "":
        optimizer_name_raw = "amsgrad"
    if optimizer_name_raw not in ("adam", "amsgrad"):
        optimizer_name_raw = "amsgrad"

    step_controller_raw = os.environ.get("HNODECB_AFM04_STAGE2LIGHT_STEP_CONTROLLER", "").strip().lower()
    if step_controller_raw == "":
        step_controller_raw = "off"
    step_controller_raw = step_controller_raw.replace("-", "_")
    step_controller_aliases = {
        "armijo": "armijo_backtracking",
        "backtracking": "armijo_backtracking",
        "armijo_backtracking": "armijo_backtracking",
        "legacy": "legacy_guard",
        "legacy_guard": "legacy_guard",
        "step_guard": "legacy_guard",
        "off": "off",
        "none": "off",
        "0": "off",
    }
    step_controller = step_controller_aliases.get(step_controller_raw, "off")

    lr_min = _env_float("HNODECB_AFM04_STAGE2LIGHT_LR_MIN", 1.0e-6)
    lr_max = _env_float("HNODECB_AFM04_STAGE2LIGHT_LR_MAX", 1.0e-2)
    if not (lr_min > 0.0):
        lr_min = 1.0e-6
    if not (lr_max >= lr_min):
        lr_max = max(lr_min, 1.0e-2)

    schedule = load_stage2light_schedule(repo_root)
    total_epochs = max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_EPOCHS", schedule.epochs))
    adam_epochs = max(0, _env_int("HNODECB_AFM04_STAGE2LIGHT_ADAM_EPOCHS", schedule.adam_epochs))
    if adam_epochs > total_epochs:
        adam_epochs = total_epochs
    schedule_overridden = (
        os.environ.get("HNODECB_AFM04_STAGE2LIGHT_EPOCHS", "").strip() != ""
        or os.environ.get("HNODECB_AFM04_STAGE2LIGHT_ADAM_EPOCHS", "").strip() != ""
    )
    lbfgs_default = max(0, total_epochs - adam_epochs) if schedule_overridden else schedule.lbfgs_epochs
    lbfgs_steps = max(0, _env_int("HNODECB_AFM04_STAGE2LIGHT_LBFGS_EPOCHS", lbfgs_default))
    total_epochs = max(1, adam_epochs + lbfgs_steps)
    val_eval_mode = os.environ.get("HNODECB_AFM04_STAGE2LIGHT_VAL_EVAL_MODE", "per_epoch").strip().lower()
    if val_eval_mode not in ("per_epoch", "final_only"):
        val_eval_mode = "per_epoch"
    mech_parameterization_raw = os.environ.get(
        "HNODECB_AFM04_STAGE2LIGHT_MECH_PARAMETERIZATION",
        "log_relative",
    ).strip().lower().replace("-", "_")
    mech_parameterization_aliases = {
        "direct": "direct_unbounded",
        "physical": "direct_unbounded",
        "unbounded": "direct_unbounded",
        "direct_unbounded": "direct_unbounded",
        "exp": "log_relative",
        "log": "log_relative",
        "relative": "log_relative",
        "log_relative": "log_relative",
        "log_relative_bounded": "log_relative",
        "sigmoid": "sigmoid_bounded",
        "bounded": "sigmoid_bounded",
        "sigmoid_bound": "sigmoid_bounded",
        "sigmoid_bounds": "sigmoid_bounded",
        "sigmoid_bounded": "sigmoid_bounded",
        "legacy": "sigmoid_bounded",
        "legacy_sigmoid": "sigmoid_bounded",
    }
    mech_parameterization = mech_parameterization_aliases.get(mech_parameterization_raw, "log_relative")

    return Stage2LightConfig(
        repo_root=repo_root,
        pykan_root=pykan_root,
        dataset_root=dataset_root,
        output_root=output_root,
        log_dir=log_dir,
        shard_log_dir=shard_log_dir,
        visualization_dir=visualization_dir,
        result_dir=result_dir,
        checkpoint_dir=checkpoint_dir,
        stage1_input_path=stage1_input_path,
        prestage2_input_path=prestage2_input_path,
        shard_index=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_SHARD_INDEX", 1)),
        shard_count=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_SHARD_COUNT", 1)),
        error_level=os.environ.get("HNODECB_AFM04_STAGE2LIGHT_ERROR_LEVEL", "e0.0").strip() or "e0.0",
        auto_generate_dataset=_env_bool("HNODECB_AFM04_STAGE2LIGHT_AUTOGEN_DATASET", False),
        window_mode=os.environ.get("HNODECB_AFM04_STAGE2LIGHT_WINDOW_MODE", "w0").strip() or "w0",
        arch_window_us=_env_float("HNODECB_AFM04_STAGE2LIGHT_WINDOW_US", 6.288e-6),
        val_stride=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_VAL_STRIDE", 5)),
        val_offset=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_VAL_OFFSET", 2)),
        val_eval_mode=val_eval_mode,
        seed=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_SEED", 20260406)),
        dtype=os.environ.get("HNODECB_AFM04_STAGE2LIGHT_DTYPE", "float64").strip() or "float64",
        device=os.environ.get("HNODECB_AFM04_STAGE2LIGHT_DEVICE", "cpu").strip() or "cpu",
        width=(3, 7, 1),
        grid=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_GRID", 11)),
        spline_k=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_SPLINE_K", 3)),
        base_fun=os.environ.get("HNODECB_AFM04_STAGE2LIGHT_BASE_FUN", "silu").strip() or "silu",
        symbolic_enabled=False,
        auto_save=False,
        noise_scale=_env_float("HNODECB_AFM04_STAGE2LIGHT_NOISE_SCALE", 0.2),
        affine_trainable=False,
        grid_eps=_env_float("HNODECB_AFM04_STAGE2LIGHT_GRID_EPS", 0.02),
        grid_range_lo=_env_float("HNODECB_AFM04_STAGE2LIGHT_GRID_RANGE_LO", -1.0),
        grid_range_hi=_env_float("HNODECB_AFM04_STAGE2LIGHT_GRID_RANGE_HI", 1.0),
        adaptive_grid_enabled=_env_bool("HNODECB_AFM04_STAGE2LIGHT_ADAPTIVE_GRID", True),
        grid_update_num=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_GRID_UPDATE_NUM", 10)),
        start_grid_update_step=_env_int("HNODECB_AFM04_STAGE2LIGHT_GRID_UPDATE_START", 0),
        stop_grid_update_step=max(0, _env_int("HNODECB_AFM04_STAGE2LIGHT_GRID_UPDATE_STOP", 50)),
        soft_mask_enabled=_env_bool("HNODECB_AFM04_STAGE2LIGHT_SOFT_MASK", True),
        soft_mask_trainable=_env_bool("HNODECB_AFM04_STAGE2LIGHT_SOFT_MASK_TRAINABLE", True),
        soft_mask_s0_a0=_env_float("HNODECB_AFM04_STAGE2LIGHT_SOFT_MASK_S0_A0", 20.0),
        soft_mask_s0_min_a0=_env_float("HNODECB_AFM04_STAGE2LIGHT_SOFT_MASK_S0_MIN_A0", 1.0),
        soft_mask_s0_max_a0=_env_float("HNODECB_AFM04_STAGE2LIGHT_SOFT_MASK_S0_MAX_A0", 100.0),
        soft_mask_alpha_a0=_env_float("HNODECB_AFM04_STAGE2LIGHT_SOFT_MASK_ALPHA_A0", 0.25),
        soft_mask_alpha_min_a0=_env_float("HNODECB_AFM04_STAGE2LIGHT_SOFT_MASK_ALPHA_MIN_A0", 0.02),
        soft_mask_alpha_max_a0=_env_float("HNODECB_AFM04_STAGE2LIGHT_SOFT_MASK_ALPHA_MAX_A0", 5.0),
        epochs=total_epochs,
        adam_epochs=adam_epochs,
        lbfgs_steps=lbfgs_steps,
        checkpoint_every=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_CHECKPOINT_EVERY", 5)),
        lr=_env_float("HNODECB_AFM04_STAGE2LIGHT_LR", 1.0e-3),
        lr_adapt=_env_bool("HNODECB_AFM04_STAGE2LIGHT_LR_ADAPT", False),
        lr_min=lr_min,
        lr_max=lr_max,
        lr_eta=_env_float("HNODECB_AFM04_STAGE2LIGHT_LR_ETA", 0.05),
        lr_ema_alpha=_env_float("HNODECB_AFM04_STAGE2LIGHT_LR_EMA", 0.97),
        lr_eps=_env_float("HNODECB_AFM04_STAGE2LIGHT_LR_EPS", 1.0e-30),
        lr_target_init=_env_float("HNODECB_AFM04_STAGE2LIGHT_LR_TARGET", float("nan")),
        lr_adapt_up_only=_env_bool("HNODECB_AFM04_STAGE2LIGHT_LR_ADAPT_UP_ONLY", False),
        weight_decay=_env_float("HNODECB_AFM04_STAGE2LIGHT_WEIGHT_DECAY", 0.0),
        gnn_learnable=_env_bool("HNODECB_AFM04_STAGE2LIGHT_GNN_LEARNABLE", True),
        group_adapt=False,
        group_ema_alpha=_env_float("HNODECB_AFM04_STAGE2LIGHT_GROUP_EMA", 0.90),
        group_eta=_env_float("HNODECB_AFM04_STAGE2LIGHT_GROUP_ETA", 0.15),
        group_eps=_env_float("HNODECB_AFM04_STAGE2LIGHT_GROUP_EPS", 1.0e-30),
        optimizer_name=optimizer_name_raw,
        optimizer_amsgrad=(optimizer_name_raw == "amsgrad"),
        plateau_early_stop=_env_bool("HNODECB_AFM04_STAGE2LIGHT_PLATEAU_EARLY_STOP", False),
        plateau_window=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_PLATEAU_WINDOW", 5)),
        plateau_tol=_env_float("HNODECB_AFM04_STAGE2LIGHT_PLATEAU_TOL", 1.0e-6),
        recent_val_early_stop=_env_bool("HNODECB_AFM04_STAGE2LIGHT_RECENT_VAL_EARLY_STOP", False),
        recent_val_window=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_RECENT_VAL_WINDOW", 100)),
        recent_val_compare_gap=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_RECENT_VAL_COMPARE_GAP", 100)),
        recent_val_rel_current_frac=_env_float("HNODECB_AFM04_STAGE2LIGHT_RECENT_VAL_REL_CURRENT_FRAC", 0.01),
        good_enough_loss=_env_float("HNODECB_AFM04_STAGE2LIGHT_GOOD_ENOUGH_LOSS", 1.0e-10),
        epoch_retry_max=max(0, _env_int("HNODECB_AFM04_STAGE2LIGHT_EPOCH_RETRIES", 8)),
        epoch_retry_lr_factor=_env_float("HNODECB_AFM04_STAGE2LIGHT_RETRY_LR_FACTOR", 0.3),
        epoch_retry_lr_floor=_env_float("HNODECB_AFM04_STAGE2LIGHT_RETRY_LR_FLOOR", lr_min),
        step_controller=step_controller,
        step_guard_enabled=_env_bool("HNODECB_AFM04_STAGE2LIGHT_STEP_GUARD", True),
        step_retry_max=max(0, _env_int("HNODECB_AFM04_STAGE2LIGHT_STEP_RETRIES", 6)),
        step_retry_lr_factor=_env_float("HNODECB_AFM04_STAGE2LIGHT_STEP_RETRY_LR_FACTOR", 0.3),
        step_retry_reset_lr_each_epoch=_env_bool(
            "HNODECB_AFM04_STAGE2LIGHT_STEP_RETRY_RESET_LR_EACH_EPOCH",
            False,
        ),
        step_guard_validate_val=_env_bool("HNODECB_AFM04_STAGE2LIGHT_STEP_GUARD_VALIDATE_VAL", False),
        step_max_loss_increase_frac=_env_float("HNODECB_AFM04_STAGE2LIGHT_STEP_MAX_LOSS_INCREASE_FRAC", 0.05),
        armijo_c1=_env_float("HNODECB_AFM04_STAGE2LIGHT_ARMIJO_C1", 1.0e-4),
        backtrack_shrink=_env_float("HNODECB_AFM04_STAGE2LIGHT_BACKTRACK_SHRINK", 0.5),
        backtrack_max=max(0, _env_int("HNODECB_AFM04_STAGE2LIGHT_BACKTRACK_MAX", 10)),
        backtrack_min_alpha=_env_float("HNODECB_AFM04_STAGE2LIGHT_BACKTRACK_MIN_ALPHA", 1.0e-8),
        lbfgs_enabled=_env_bool("HNODECB_AFM04_STAGE2LIGHT_LBFGS_ENABLED", True),
        lbfgs_strong_wolfe=_env_bool("HNODECB_AFM04_STAGE2LIGHT_LBFGS_STRONG_WOLFE", True),
        lbfgs_lr=_env_float("HNODECB_AFM04_STAGE2LIGHT_LBFGS_LR", 1.0),
        lbfgs_max_iter=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_LBFGS_MAX_ITER", 1)),
        lbfgs_max_eval=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_LBFGS_MAX_EVAL", 10)),
        lbfgs_history_size=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_LBFGS_HISTORY_SIZE", 100)),
        lbfgs_tolerance_grad=max(0.0, _env_float("HNODECB_AFM04_STAGE2LIGHT_LBFGS_TOLERANCE_GRAD", 1.0e-12)),
        lbfgs_tolerance_change=max(0.0, _env_float("HNODECB_AFM04_STAGE2LIGHT_LBFGS_TOLERANCE_CHANGE", 1.0e-15)),
        lbfgs_ys_threshold=max(0.0, _env_float("HNODECB_AFM04_STAGE2LIGHT_LBFGS_YS_THRESHOLD", 1.0e-15)),
        train_window_index=max(0, _env_int("HNODECB_AFM04_STAGE2LIGHT_TRAIN_WINDOW_INDEX", 0)),
        warmstart_from_stage1=_env_bool("HNODECB_AFM04_STAGE2LIGHT_WARMSTART_FROM_STAGE1", True),
        warmstart_fallback_random=_env_bool("HNODECB_AFM04_STAGE2LIGHT_WARMSTART_FALLBACK_RANDOM", False),
        warmstart_source=warmstart_source,
        stage1_input_rank=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_INPUT_RANK", 48)),
        stage1_input_trial_id=max(0, _env_int("HNODECB_AFM04_STAGE2LIGHT_INPUT_TRIAL_ID", 0)),
        stage1_input_mech_winner=max(0, _env_int("HNODECB_AFM04_STAGE2LIGHT_INPUT_MECH_WINNER", 0)),
        prestage2_input_candidate=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_INPUT_CANDIDATE", 1)),
        resume_from_checkpoint=_env_bool("HNODECB_AFM04_STAGE2LIGHT_RESUME", True),
        mech_parameterization=mech_parameterization,
        ks_lo=_env_float("HNODECB_AFM04_STAGE2LIGHT_KS_LO", float(KS_BOUNDS[0])),
        ks_hi=_env_float("HNODECB_AFM04_STAGE2LIGHT_KS_HI", float(KS_BOUNDS[1])),
        cs_lo=_env_float("HNODECB_AFM04_STAGE2LIGHT_CS_LO", float(CS_BOUNDS[0])),
        cs_hi=_env_float("HNODECB_AFM04_STAGE2LIGHT_CS_HI", float(CS_BOUNDS[1])),
        ode_method=os.environ.get("HNODECB_AFM04_STAGE2LIGHT_ODE_METHOD", "dopri5").strip() or "dopri5",
        ode_rtol=_env_float("HNODECB_AFM04_STAGE2LIGHT_ODE_RTOL", 1.0e-10),
        ode_atol=_env_float("HNODECB_AFM04_STAGE2LIGHT_ODE_ATOL", 1.0e-12),
        log_every=max(1, _env_int("HNODECB_AFM04_STAGE2LIGHT_LOG_EVERY", 1)),
    )


__all__ = ["Stage2LightConfig", "default_config"]
