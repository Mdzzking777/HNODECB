"""Configuration and invariants for AFM06a stage2light."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else int(raw)


def _schedule_values(stage_root: Path) -> tuple[int, int, int]:
    path = stage_root / "stage2light_schedule.json"
    fallback = (400, 10, 390)
    if not path.is_file():
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        epochs = int(payload["epochs"])
        adam = int(payload["adam_epochs"])
        lbfgs = int(payload["lbfgs_epochs"])
    except Exception:
        return fallback
    return epochs, adam, lbfgs


@dataclass(frozen=True)
class Stage2LightConfig:
    repo_root: Path
    pykan_root: Path
    stage1_result_path: Path
    result_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    visualization_dir: Path
    archive_root: Path
    input_rank: int
    device: str
    dtype: str
    epochs: int
    adam_epochs: int
    lbfgs_epochs: int
    adam_lr: float
    adam_amsgrad: bool
    lbfgs_lr: float
    lbfgs_max_iter: int
    lbfgs_max_eval: int
    lbfgs_history_size: int
    lbfgs_tolerance_grad: float
    lbfgs_tolerance_change: float
    lbfgs_ys_threshold: float
    lbfgs_sparsification_enabled: bool
    lbfgs_sparsification_lambda: float
    lbfgs_sparsification_entropy_weight: float
    lbfgs_sparsification_epsilon: float
    prune_on_first_zero_step: bool
    pruning_max_val_loss_increase_fraction: float
    pruning_max_val_loss_increase_absolute: float
    pruning_min_hidden_nodes: int
    lbfgs_zero_step_max_events: int
    gain_enabled: bool
    gain_learnable: bool
    soft_mask_enabled: bool
    soft_mask_trainable: bool
    adaptive_grid_enabled: bool
    agu_update_every_adam_epoch: bool
    resume_enabled: bool
    checkpoint_every: int
    loss_term_grad_every: int
    entry_policy: str
    epoch0_preparation: bool

    @property
    def result_path(self) -> Path:
        return self.result_dir / f"afm_param_stage2light_06a_rank{self.input_rank}.pkl"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"afm_param_stage2light_06a_rank{self.input_rank}.checkpoint.pkl"

    @property
    def run_log_path(self) -> Path:
        return self.log_dir / f"log2_06a_step2a_stage2light_local_rank{self.input_rank}.txt"

    def readiness_gaps(self) -> list[str]:
        gaps: list[str] = []
        if not self.stage1_result_path.is_file():
            gaps.append(f"completed AFM06a st1pl result is missing: {self.stage1_result_path}")
        if not self.pykan_root.is_dir():
            gaps.append(f"pykan source directory is missing: {self.pykan_root}")
        if self.input_rank <= 0:
            gaps.append("input_rank must be positive")
        if self.dtype.strip().lower() not in {"float64", "double"}:
            gaps.append("AFM06a stage2light requires float64")
        if self.device.strip() == "":
            gaps.append("device cannot be empty")
        if self.epochs <= 0 or self.adam_epochs < 0 or self.lbfgs_epochs < 0:
            gaps.append("optimizer epoch counts must be nonnegative and total epochs positive")
        if self.adam_epochs + self.lbfgs_epochs != self.epochs:
            gaps.append("adam_epochs + lbfgs_epochs must equal epochs")
        if self.loss_term_grad_every <= 0:
            gaps.append("loss_term_grad_every must be positive")
        if self.lbfgs_sparsification_lambda < 0.0:
            gaps.append("lbfgs_sparsification_lambda must be nonnegative")
        if self.lbfgs_sparsification_entropy_weight < 0.0:
            gaps.append("lbfgs_sparsification_entropy_weight must be nonnegative")
        if self.lbfgs_sparsification_epsilon <= 0.0:
            gaps.append("lbfgs_sparsification_epsilon must be positive")
        if self.pruning_max_val_loss_increase_fraction < 0.0:
            gaps.append("pruning_max_val_loss_increase_fraction must be nonnegative")
        if self.pruning_max_val_loss_increase_absolute < 0.0:
            gaps.append("pruning_max_val_loss_increase_absolute must be nonnegative")
        if self.pruning_min_hidden_nodes <= 0:
            gaps.append("pruning_min_hidden_nodes must be positive")
        if self.lbfgs_zero_step_max_events != 3:
            gaps.append("AFM06a st2l requires exactly three L-BFGS hard-zero-step events")
        if not self.adaptive_grid_enabled:
            gaps.append("AFM06a st2l currently requires AGU during Adam")
        if not self.agu_update_every_adam_epoch:
            gaps.append("AFM06a st2l currently requires one AGU update in every Adam epoch")
        if self.entry_policy != "direct_complete_st1pl_warmstart":
            gaps.append("AFM06a st2l must use direct_complete_st1pl_warmstart")
        if self.epoch0_preparation:
            gaps.append("epoch-0 preparation is forbidden for AFM06a st2l")
        if self.gain_enabled or self.gain_learnable:
            gaps.append("AFM06a st2l requires Gain to be disabled and non-learnable")
        if self.soft_mask_enabled or self.soft_mask_trainable:
            gaps.append("AFM06a st2l requires Soft Mask to be disabled and non-learnable")
        return gaps

    def validate(self) -> None:
        gaps = self.readiness_gaps()
        if gaps:
            detail = "\n".join(f"  - {gap}" for gap in gaps)
            raise RuntimeError(f"AFM06a stage2light is not ready:\n{detail}")


def default_config(repo_root: str | Path | None = None) -> Stage2LightConfig:
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).resolve()
    stage_root = root / "AFM06a" / "stage2light"
    default_epochs, default_adam, default_lbfgs = _schedule_values(stage_root)
    epochs = _env_int("HNODECB_AFM06a_STAGE2LIGHT_EPOCHS", default_epochs)
    adam_epochs = _env_int("HNODECB_AFM06a_STAGE2LIGHT_ADAM_EPOCHS", default_adam)
    lbfgs_epochs = _env_int("HNODECB_AFM06a_STAGE2LIGHT_LBFGS_EPOCHS", default_lbfgs)
    return Stage2LightConfig(
        repo_root=root,
        pykan_root=Path(os.environ.get("HNODECB_PYKAN_ROOT", str(root.parent / "pykan"))),
        stage1_result_path=Path(
            os.environ.get(
                "HNODECB_AFM06a_STAGE2LIGHT_STAGE1_RESULT",
                str(root / "AFM06a" / "stage1pluslight" / "results_afm" / "afm_param_stage1pluslight_06a.pkl"),
            )
        ),
        result_dir=Path(
            os.environ.get("HNODECB_AFM06a_STAGE2LIGHT_RESULT_DIR", str(stage_root / "results"))
        ),
        checkpoint_dir=Path(
            os.environ.get("HNODECB_AFM06a_STAGE2LIGHT_CHECKPOINT_DIR", str(stage_root / "checkpoints"))
        ),
        log_dir=Path(os.environ.get("HNODECB_AFM06a_STAGE2LIGHT_LOG_DIR", str(stage_root / "logs"))),
        visualization_dir=Path(
            os.environ.get(
                "HNODECB_AFM06a_STAGE2LIGHT_VISUALIZATION_DIR",
                str(stage_root / "logs" / "visualization"),
            )
        ),
        archive_root=Path(
            os.environ.get(
                "HNODECB_AFM06a_STAGE2LIGHT_ARCHIVE_ROOT",
                str(root / "AFM06a" / "Archive" / "st2l"),
            )
        ),
        input_rank=max(1, _env_int("HNODECB_AFM06a_STAGE2LIGHT_INPUT_RANK", 1)),
        device=os.environ.get("HNODECB_AFM06a_STAGE2LIGHT_DEVICE", "cpu").strip() or "cpu",
        dtype=os.environ.get("HNODECB_AFM06a_STAGE2LIGHT_DTYPE", "float64").strip() or "float64",
        epochs=epochs,
        adam_epochs=adam_epochs,
        lbfgs_epochs=lbfgs_epochs,
        adam_lr=_env_float("HNODECB_AFM06a_STAGE2LIGHT_ADAM_LR", 1.0e-4),
        adam_amsgrad=_env_bool("HNODECB_AFM06a_STAGE2LIGHT_ADAM_AMSGRAD", True),
        lbfgs_lr=_env_float("HNODECB_AFM06a_STAGE2LIGHT_LBFGS_LR", 1.0),
        lbfgs_max_iter=max(1, _env_int("HNODECB_AFM06a_STAGE2LIGHT_LBFGS_MAX_ITER", 1)),
        lbfgs_max_eval=max(1, _env_int("HNODECB_AFM06a_STAGE2LIGHT_LBFGS_MAX_EVAL", 10)),
        lbfgs_history_size=max(1, _env_int("HNODECB_AFM06a_STAGE2LIGHT_LBFGS_HISTORY_SIZE", 100)),
        lbfgs_tolerance_grad=_env_float("HNODECB_AFM06a_STAGE2LIGHT_LBFGS_TOLERANCE_GRAD", 1.0e-12),
        lbfgs_tolerance_change=_env_float("HNODECB_AFM06a_STAGE2LIGHT_LBFGS_TOLERANCE_CHANGE", 1.0e-15),
        lbfgs_ys_threshold=_env_float("HNODECB_AFM06a_STAGE2LIGHT_LBFGS_YS_THRESHOLD", 1.0e-15),
        lbfgs_sparsification_enabled=_env_bool(
            "HNODECB_AFM06a_STAGE2LIGHT_LBFGS_SPARSIFICATION",
            False,
        ),
        lbfgs_sparsification_lambda=_env_float(
            "HNODECB_AFM06a_STAGE2LIGHT_LBFGS_SPARSIFICATION_LAMBDA",
            1.0e-7,
        ),
        lbfgs_sparsification_entropy_weight=_env_float(
            "HNODECB_AFM06a_STAGE2LIGHT_LBFGS_SPARSIFICATION_ENTROPY_WEIGHT",
            2.0,
        ),
        lbfgs_sparsification_epsilon=_env_float(
            "HNODECB_AFM06a_STAGE2LIGHT_LBFGS_SPARSIFICATION_EPSILON",
            1.0e-12,
        ),
        prune_on_first_zero_step=_env_bool(
            "HNODECB_AFM06a_STAGE2LIGHT_PRUNE_ON_FIRST_ZERO_STEP",
            False,
        ),
        pruning_max_val_loss_increase_fraction=_env_float(
            "HNODECB_AFM06a_STAGE2LIGHT_PRUNING_MAX_VAL_LOSS_INCREASE_FRACTION",
            5.0e-2,
        ),
        pruning_max_val_loss_increase_absolute=_env_float(
            "HNODECB_AFM06a_STAGE2LIGHT_PRUNING_MAX_VAL_LOSS_INCREASE_ABSOLUTE",
            1.0e-12,
        ),
        pruning_min_hidden_nodes=max(
            1,
            _env_int("HNODECB_AFM06a_STAGE2LIGHT_PRUNING_MIN_HIDDEN_NODES", 1),
        ),
        lbfgs_zero_step_max_events=max(
            1,
            _env_int("HNODECB_AFM06a_STAGE2LIGHT_LBFGS_ZERO_STEP_MAX_EVENTS", 3),
        ),
        gain_enabled=_env_bool("HNODECB_AFM06a_STAGE2LIGHT_GAIN", False),
        gain_learnable=_env_bool("HNODECB_AFM06a_STAGE2LIGHT_GAIN_LEARNABLE", False),
        soft_mask_enabled=_env_bool("HNODECB_AFM06a_STAGE2LIGHT_SOFT_MASK", False),
        soft_mask_trainable=_env_bool("HNODECB_AFM06a_STAGE2LIGHT_SOFT_MASK_TRAINABLE", False),
        adaptive_grid_enabled=_env_bool("HNODECB_AFM06a_STAGE2LIGHT_ADAPTIVE_GRID", True),
        agu_update_every_adam_epoch=_env_bool(
            "HNODECB_AFM06a_STAGE2LIGHT_AGU_EVERY_ADAM_EPOCH", True
        ),
        resume_enabled=_env_bool("HNODECB_AFM06a_STAGE2LIGHT_RESUME", True),
        checkpoint_every=max(1, _env_int("HNODECB_AFM06a_STAGE2LIGHT_CHECKPOINT_EVERY", 1)),
        loss_term_grad_every=max(
            1,
            _env_int("HNODECB_AFM06a_STAGE2LIGHT_LOSS_TERM_GRAD_EVERY", 5),
        ),
        entry_policy="direct_complete_st1pl_warmstart",
        epoch0_preparation=False,
    )


__all__ = ["Stage2LightConfig", "default_config"]
