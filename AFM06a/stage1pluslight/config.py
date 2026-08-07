"""Configuration and readiness gate for AFM06a stage1pluslight."""

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
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw not in {"0", "false", "no", "off"}


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    return None if raw == "" else float(raw)


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return None if raw == "" else int(raw)


def _env_optional_text(name: str) -> str | None:
    raw = os.environ.get(name, "").strip()
    return None if raw == "" else raw


@dataclass(frozen=True)
class Stage1PlusLightConfig:
    repo_root: Path
    pykan_root: Path
    dataset_root: Path
    result_dir: Path
    log_dir: Path
    error_level: str
    result_stem: str
    run_tag: str
    shard_index: int
    shard_count: int
    nn_seed_bank_size: int
    run_seed: int
    checkpoint_every: int
    resume_enabled: bool
    final_topk: int
    all_points_training: bool
    val_stride: int
    val_offset: int
    window_start_s: float | None
    window_stop_s: float | None
    sample_stride: int | None
    sample_count: int | None
    sampling_policy: str
    noncontact_sampling_weight: float
    contact_sampling_weight: float
    transition_sampling_weight: float
    transition_half_width_s: float
    smoothness_loss_weight: float
    kan_width: tuple[int, int, int]
    kan_grid: int | None
    kan_spline_k: int | None
    kan_base_fun: str | None
    kan_noise_scale: float | None
    kan_grid_eps: float | None
    kan_grid_range_lo: float | None
    kan_grid_range_hi: float | None
    adaptive_grid_enabled: bool
    dtype: str
    device: str
    gain_enabled: bool
    gain_learnable: bool
    soft_mask_enabled: bool
    soft_mask_trainable: bool
    soft_mask_s0_a0: float
    soft_mask_s0_min_a0: float
    soft_mask_s0_max_a0: float
    soft_mask_alpha_a0: float
    soft_mask_alpha_min_a0: float
    soft_mask_alpha_max_a0: float
    state_guard_multiplier: float
    ode_step_budget: int
    normalizer_policy: str | None
    force_output_policy: str | None
    gain_policy: str | None
    soft_mask_policy: str | None
    loss_policy: str | None
    ode_method: str | None
    ode_rtol: float | None
    ode_atol: float | None

    @property
    def result_basename(self) -> str:
        stem = self.result_stem if self.run_tag == "" else f"{self.result_stem}_{self.run_tag}"
        return f"{stem}_p{self.shard_index}.pkl" if self.shard_count > 1 else f"{stem}.pkl"

    @property
    def result_path(self) -> Path:
        return self.result_dir / self.result_basename

    @property
    def merged_result_path(self) -> Path:
        stem = self.result_stem if self.run_tag == "" else f"{self.result_stem}_{self.run_tag}"
        return self.result_dir / f"{stem}.pkl"

    def readiness_gaps(self) -> list[str]:
        gaps: list[str] = []
        if self.kan_width != (1, 3, 1):
            gaps.append(f"KAN width must remain the single-input (1, 3, 1), got {self.kan_width}")
        if not self.pykan_root.is_dir():
            gaps.append(f"pykan source directory is missing: {self.pykan_root}")
        if self.nn_seed_bank_size <= 0:
            gaps.append("NN seed-bank size is not selected")
        if self.window_start_s is None or self.window_stop_s is None:
            gaps.append("training-window start/stop is not selected")
        elif self.window_stop_s <= self.window_start_s:
            gaps.append("training-window stop must be greater than start")
        full_resolution_sampling = self.sampling_policy == "full_resolution_all_train"
        if self.sample_stride is None or self.sample_stride <= 0:
            gaps.append("training sample stride is not selected")
        if self.sample_count is not None and self.sample_count <= 1:
            gaps.append("explicit training sample count must be greater than one")
        if self.sampling_policy not in {
            "full_resolution_all_train",
            "regime_weighted_fixed_count",
            "regime_weighted_contact_doubled",
        }:
            gaps.append("formal AFM06a requires a supported sampling policy")
        if full_resolution_sampling and not self.all_points_training:
            gaps.append("full-resolution AFM06a sampling requires all selected points to be training points")
        if (not full_resolution_sampling) and (
            self.transition_sampling_weight,
            self.contact_sampling_weight,
            self.noncontact_sampling_weight,
        ) != (10.0, 10.0, 1.0):
            gaps.append("formal AFM06a e0.0_real requires 10:10:1 transition/contact/noncontact sampling")
        if self.transition_half_width_s < 0.0:
            gaps.append("transition sampling half-width cannot be negative")
        if self.smoothness_loss_weight < 0.0:
            gaps.append("residual smoothness loss weight must be nonnegative")
        if self.kan_grid is None or self.kan_grid <= 0:
            gaps.append("KAN grid size is not selected")
        if self.kan_spline_k is None or self.kan_spline_k <= 0:
            gaps.append("KAN spline order is not selected")
        if self.kan_base_fun is None:
            gaps.append("KAN base function is not selected")
        if self.kan_noise_scale is None:
            gaps.append("KAN initialization noise scale is not selected")
        if self.kan_grid_eps is None:
            gaps.append("AGU grid_eps is not selected")
        if self.kan_grid_range_lo is None or self.kan_grid_range_hi is None:
            gaps.append("fallback KAN support is not selected")
        elif self.kan_grid_range_hi <= self.kan_grid_range_lo:
            gaps.append("KAN support upper bound must exceed lower bound")
        if self.normalizer_policy != "training_mean_std":
            gaps.append("AFM06a 1-3-1 KAN requires training_mean_std input from the selected W1 window")
        if self.gain_enabled:
            gaps.append("AFM06a st1pl requires Gain to be disabled")
        if self.gain_learnable:
            gaps.append("AFM06a st1pl requires Gain to be non-learnable")
        if self.gain_policy != "disabled":
            gaps.append("AFM06a st1pl requires the disabled Gain policy")
        if self.soft_mask_enabled:
            gaps.append("AFM06a st1pl requires Soft Mask to be disabled")
        if self.soft_mask_trainable:
            gaps.append("AFM06a st1pl requires Soft Mask to be non-learnable")
        if self.soft_mask_policy != "disabled":
            gaps.append("AFM06a st1pl requires the disabled Soft Mask policy")
        if self.force_output_policy != "training_mean_std_inverse":
            gaps.append(
                "AFM06a requires the KAN raw head to be mapped back to physical "
                "Fts [N] with the selected-window training_mean_std inverse"
            )
        for label, value in (
            ("normalizer", self.normalizer_policy),
            ("force output", self.force_output_policy),
            ("gain", self.gain_policy),
            ("soft mask", self.soft_mask_policy),
            ("loss", self.loss_policy),
        ):
            if value is None:
                gaps.append(f"{label} policy is not selected")
        if self.loss_policy != "force_scaled_dynamics_residual_true_x1_residual_smooth":
            gaps.append(
                "AFM06a requires force-scale-normalized r_dyn plus within-regime "
                "residual-smoothness losses conditioned on observed x1"
            )
        if self.ode_method is None or self.ode_rtol is None or self.ode_atol is None:
            gaps.append("rollout solver/tolerances are not selected")
        if self.dtype.strip().lower() not in {"float64", "double"}:
            gaps.append("formal AFM06a st1pl requires float64")
        if self.device.strip() == "":
            gaps.append("device is not selected")
        if self.state_guard_multiplier <= 0.0:
            gaps.append("state guard multiplier must be positive")
        if self.ode_step_budget <= 0:
            gaps.append("ODE step budget must be positive")
        return gaps

    def validate_for_formal_run(self) -> None:
        gaps = self.readiness_gaps()
        if gaps:
            detail = "\n".join(f"  - {gap}" for gap in gaps)
            raise RuntimeError(f"AFM06a stage1pluslight is not ready for a formal run:\n{detail}")
        if not (1 <= self.shard_index <= self.shard_count):
            raise ValueError("shard_index must be in 1..shard_count")
        if (not self.all_points_training) and (
            self.val_stride <= 1 or not (1 <= self.val_offset <= self.val_stride)
        ):
            raise ValueError("validation split requires val_stride > 1 and offset in 1..stride")


def default_config(repo_root: str | Path | None = None) -> Stage1PlusLightConfig:
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)
    stage_root = root / "AFM06a" / "stage1pluslight"
    default_error_level = "e0.0_real"
    return Stage1PlusLightConfig(
        repo_root=root,
        pykan_root=Path(os.environ.get("HNODECB_PYKAN_ROOT", str(root.parent / "pykan"))),
        dataset_root=root / "AFM06a" / "datasets",
        result_dir=Path(os.environ.get("HNODECB_AFM06a_STAGE1_RESULT_DIR", str(stage_root / "results_afm"))),
        log_dir=Path(os.environ.get("HNODECB_AFM06a_STAGE1_LOG_DIR", str(stage_root / "logs"))),
        error_level=os.environ.get("HNODECB_AFM06a_ERROR_LEVEL", default_error_level).strip()
        or default_error_level,
        result_stem=os.environ.get("HNODECB_AFM06a_STAGE1_RESULT_STEM", "afm_param_stage1pluslight_06a").strip()
        or "afm_param_stage1pluslight_06a",
        run_tag=os.environ.get("HNODECB_AFM06a_STAGE1_RUN_TAG", "").strip(),
        shard_index=max(1, _env_int("HNODECB_AFM06a_STAGE1_SHARD_INDEX", 1)),
        shard_count=max(1, _env_int("HNODECB_AFM06a_STAGE1_SHARD_COUNT", 1)),
        nn_seed_bank_size=max(1, _env_int("HNODECB_AFM06a_STAGE1_NN_SEEDS", 2000)),
        run_seed=max(1, _env_int("HNODECB_AFM06a_STAGE1_RUN_SEED", 20260719)),
        checkpoint_every=max(1, _env_int("HNODECB_AFM06a_STAGE1_CHECKPOINT_EVERY", 100)),
        resume_enabled=_env_bool("HNODECB_AFM06a_STAGE1_RESUME", True),
        final_topk=max(1, _env_int("HNODECB_AFM06a_STAGE1_FINAL_TOPK", 200)),
        all_points_training=_env_bool("HNODECB_AFM06a_STAGE1_ALL_TRAIN", True),
        val_stride=max(2, _env_int("HNODECB_AFM06a_STAGE1_VAL_STRIDE", 5)),
        val_offset=max(1, _env_int("HNODECB_AFM06a_STAGE1_VAL_OFFSET", 2)),
        window_start_s=_env_float("HNODECB_AFM06a_STAGE1_WINDOW_START_S", 0.010000000),
        window_stop_s=_env_float("HNODECB_AFM06a_STAGE1_WINDOW_STOP_S", 0.010013328),
        sample_stride=max(1, _env_int("HNODECB_AFM06a_STAGE1_SAMPLE_STRIDE", 1)),
        sample_count=_env_optional_int("HNODECB_AFM06a_STAGE1_SAMPLE_COUNT"),
        sampling_policy=os.environ.get(
            "HNODECB_AFM06a_STAGE1_SAMPLING_POLICY", "full_resolution_all_train"
        ).strip()
        or "full_resolution_all_train",
        noncontact_sampling_weight=_env_float(
            "HNODECB_AFM06a_STAGE1_NONCONTACT_SAMPLING_WEIGHT", 1.0
        ),
        contact_sampling_weight=_env_float(
            "HNODECB_AFM06a_STAGE1_CONTACT_SAMPLING_WEIGHT", 10.0
        ),
        transition_sampling_weight=_env_float(
            "HNODECB_AFM06a_STAGE1_TRANSITION_SAMPLING_WEIGHT", 10.0
        ),
        transition_half_width_s=_env_float(
            "HNODECB_AFM06a_STAGE1_TRANSITION_HALF_WIDTH_S", 1.3565187713310766e-7
        ),
        smoothness_loss_weight=_env_float(
            "HNODECB_AFM06a_STAGE1_SMOOTHNESS_LOSS_WEIGHT", 1.0
        ),
        kan_width=(1, 3, 1),
        kan_grid=max(1, _env_int("HNODECB_AFM06a_STAGE1_KAN_GRID", 11)),
        kan_spline_k=max(1, _env_int("HNODECB_AFM06a_STAGE1_KAN_SPLINE_K", 3)),
        kan_base_fun=os.environ.get("HNODECB_AFM06a_STAGE1_KAN_BASE_FUN", "silu").strip() or "silu",
        kan_noise_scale=_env_float("HNODECB_AFM06a_STAGE1_KAN_NOISE_SCALE", 0.2),
        kan_grid_eps=_env_float("HNODECB_AFM06a_STAGE1_KAN_GRID_EPS", 0.02),
        kan_grid_range_lo=_env_float("HNODECB_AFM06a_STAGE1_KAN_GRID_LO", -1.0),
        kan_grid_range_hi=_env_float("HNODECB_AFM06a_STAGE1_KAN_GRID_HI", 1.0),
        adaptive_grid_enabled=_env_bool("HNODECB_AFM06a_STAGE1_ADAPTIVE_GRID", True),
        dtype=os.environ.get("HNODECB_AFM06a_STAGE1_DTYPE", "float64").strip() or "float64",
        device=os.environ.get("HNODECB_AFM06a_STAGE1_DEVICE", "cpu").strip() or "cpu",
        gain_enabled=_env_bool("HNODECB_AFM06a_STAGE1_GAIN", False),
        gain_learnable=_env_bool("HNODECB_AFM06a_STAGE1_GAIN_LEARNABLE", False),
        soft_mask_enabled=_env_bool("HNODECB_AFM06a_STAGE1_SOFT_MASK", False),
        soft_mask_trainable=_env_bool("HNODECB_AFM06a_STAGE1_SOFT_MASK_TRAINABLE", False),
        soft_mask_s0_a0=_env_float("HNODECB_AFM06a_STAGE1_SOFT_MASK_S0_A0", 20.0),
        soft_mask_s0_min_a0=_env_float("HNODECB_AFM06a_STAGE1_SOFT_MASK_S0_MIN_A0", 1.0),
        soft_mask_s0_max_a0=_env_float("HNODECB_AFM06a_STAGE1_SOFT_MASK_S0_MAX_A0", 100.0),
        soft_mask_alpha_a0=_env_float("HNODECB_AFM06a_STAGE1_SOFT_MASK_ALPHA_A0", 0.25),
        soft_mask_alpha_min_a0=_env_float("HNODECB_AFM06a_STAGE1_SOFT_MASK_ALPHA_MIN_A0", 0.02),
        soft_mask_alpha_max_a0=_env_float("HNODECB_AFM06a_STAGE1_SOFT_MASK_ALPHA_MAX_A0", 5.0),
        state_guard_multiplier=_env_float("HNODECB_AFM06a_STAGE1_STATE_GUARD_MULT", 100.0),
        ode_step_budget=max(1, _env_int("HNODECB_AFM06a_STAGE1_ODE_STEP_BUDGET", 10000)),
        normalizer_policy=os.environ.get(
            "HNODECB_AFM06a_STAGE1_NORMALIZER_POLICY", "training_mean_std"
        ).strip()
        or "training_mean_std",
        force_output_policy=os.environ.get(
            "HNODECB_AFM06a_STAGE1_FORCE_OUTPUT_POLICY",
            "training_mean_std_inverse",
        ).strip()
        or "training_mean_std_inverse",
        gain_policy=os.environ.get(
            "HNODECB_AFM06a_STAGE1_GAIN_POLICY",
            "disabled",
        ).strip()
        or "disabled",
        soft_mask_policy=os.environ.get(
            "HNODECB_AFM06a_STAGE1_SOFT_MASK_POLICY", "disabled"
        ).strip()
        or "disabled",
        loss_policy=os.environ.get(
            "HNODECB_AFM06a_STAGE1_LOSS_POLICY",
            "force_scaled_dynamics_residual_true_x1_residual_smooth",
        ).strip()
        or "force_scaled_dynamics_residual_true_x1_residual_smooth",
        ode_method=os.environ.get("HNODECB_AFM06a_STAGE1_ODE_METHOD", "dopri5").strip() or "dopri5",
        ode_rtol=_env_float("HNODECB_AFM06a_STAGE1_ODE_RTOL", 1.0e-10),
        ode_atol=_env_float("HNODECB_AFM06a_STAGE1_ODE_ATOL", 1.0e-12),
    )


__all__ = ["Stage1PlusLightConfig", "default_config"]
