"""AFM06a-local single-input KAN force module."""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from AFM06a.stage1pluslight.config import Stage1PlusLightConfig
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (
    AFM06aHardSampleInputs,
)

RAW_INPUT_POLICY = "raw_state_identity"
NANOMETER_INPUT_POLICY = "x1_nanometer_units"
NORMALIZED_INPUT_POLICY = "training_mean_std"
GLOBAL_NORMALIZED_INPUT_POLICY = "global_full_span_mean_std"
IDENTITY_FORCE_OUTPUT_POLICY = "identity_raw_physical"
LOCAL_FORCE_OUTPUT_POLICY = "training_mean_std_inverse"
GLOBAL_FORCE_OUTPUT_POLICY = "global_full_span_mean_std_inverse"
METERS_PER_NANOMETER = 1.0e-9
SINGLE_INPUT_KAN_WIDTH = (1, 3, 1)


def import_pykan(pykan_root: str | Path):
    root = Path(pykan_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"pykan source directory not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from kan import KAN

    return KAN


def configure_deterministic_torch(seed: int) -> None:
    """Make pykan AGU coefficient refits reproducible within each shard process."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting the inter-op pool only before parallel work
        # starts. Repeated model construction in one shard is still safe after
        # the first successful call.
        pass
    torch.use_deterministic_algorithms(True)
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


def _safe_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = 1.0
    return scale


def training_state_normalizer(training_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean/std statistics for the single x1 KAN input."""

    states = np.asarray(training_states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 2 or states.shape[0] < 2:
        raise ValueError(f"training_states must have shape (N, 2), got {states.shape}")
    if not np.all(np.isfinite(states)):
        raise ValueError("training_states contains nonfinite values")
    mean = np.asarray([np.mean(states[:, 0], dtype=float)], dtype=float)
    scale = np.asarray([_safe_scale(states[:, 0])], dtype=float)
    return np.asarray(mean, dtype=float), scale


def training_force_normalizer(training_force: np.ndarray) -> tuple[float, float]:
    """Return mean/std statistics for the training-window force reference."""

    force = np.asarray(training_force, dtype=float).reshape(-1)
    if force.size < 2 or not np.all(np.isfinite(force)):
        raise ValueError("training_force must contain at least two finite values")
    return float(np.mean(force, dtype=float)), _safe_scale(force)


def normalized_support_from_raw_inputs(
    raw_inputs: torch.Tensor | np.ndarray,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
) -> np.ndarray:
    values = raw_inputs.detach().cpu().numpy() if isinstance(raw_inputs, torch.Tensor) else raw_inputs
    values = np.asarray(values, dtype=float)
    mean = np.asarray(state_mean, dtype=float).reshape(-1)
    scale = np.asarray(state_scale, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"AFM06a states must have shape (N, 2), got {values.shape}")
    if mean.shape == (2,) and scale.shape == (2,):
        mean = mean[:1]
        scale = scale[:1]
    if mean.shape != (1,) or scale.shape != (1,) or np.any(scale <= 1.0e-30):
        raise ValueError(f"invalid x1 normalizer: mean={mean}, scale={scale}")
    normalized = (values[:, [0]] - mean.reshape(1, 1)) / scale.reshape(1, 1)
    support = np.column_stack((np.min(normalized, axis=0), np.max(normalized, axis=0)))
    if not np.all(np.isfinite(support)) or np.any(support[:, 1] <= support[:, 0]):
        raise ValueError(f"invalid normalized support: {support}")
    return support


def raw_support_from_inputs(raw_inputs: torch.Tensor | np.ndarray) -> np.ndarray:
    values = raw_inputs.detach().cpu().numpy() if isinstance(raw_inputs, torch.Tensor) else raw_inputs
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"AFM06a states must have shape (N, 2), got {values.shape}")
    x1 = values[:, [0]]
    support = np.column_stack((np.min(x1, axis=0), np.max(x1, axis=0)))
    if not np.all(np.isfinite(support)) or np.any(support[:, 1] <= support[:, 0]):
        raise ValueError(f"invalid raw support: {support}")
    return support


def _coerce_support(
    support: Sequence[Sequence[float]] | np.ndarray | None,
    fallback_range: tuple[float, float],
    *,
    input_dim: int,
) -> np.ndarray:
    if support is None:
        support_arr = np.tile(np.asarray([fallback_range], dtype=float), (input_dim, 1))
    else:
        support_arr = np.asarray(support, dtype=float)
    if support_arr.shape != (input_dim, 2):
        raise ValueError(
            f"initial support must have shape ({input_dim}, 2), got {support_arr.shape}"
        )
    if not np.all(np.isfinite(support_arr)) or np.any(support_arr[:, 1] <= support_arr[:, 0]):
        raise ValueError(f"invalid initial support: {support_arr}")
    return support_arr


def _inverse_sigmoid_fraction(value: float, lo: float, hi: float) -> float:
    frac = (float(value) - float(lo)) / max(float(hi) - float(lo), 1.0e-30)
    frac = min(max(frac, 1.0e-12), 1.0 - 1.0e-12)
    return float(np.log(frac / (1.0 - frac)))


def state_dict_digest(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class KANForceModule(nn.Module):
    """Map saved AFM06a state inputs to physical interaction force Fts [N]."""

    def __init__(
        self,
        *,
        config: Stage1PlusLightConfig,
        state_mean: np.ndarray,
        state_scale: np.ndarray,
        seed: int,
        initial_support: Sequence[Sequence[float]] | np.ndarray | None,
        force_mean: float = 0.0,
        force_scale: float = 1.0,
        force_output_policy: str = IDENTITY_FORCE_OUTPUT_POLICY,
        settings: AFM06aHardSampleInputs | None = None,
    ) -> None:
        super().__init__()
        width = tuple(int(value) for value in config.kan_width)
        if width != SINGLE_INPUT_KAN_WIDTH:
            raise ValueError(
                f"AFM06a requires the single-input 1-3-1 KAN, got {config.kan_width}"
            )
        input_dimension = int(width[0])
        if config.kan_grid is None or config.kan_spline_k is None or config.kan_base_fun is None:
            raise ValueError("KAN grid, spline order, and base function must be configured")
        if config.kan_grid_eps is None or config.kan_noise_scale is None:
            raise ValueError("KAN grid_eps and noise_scale must be configured")
        if config.kan_grid_range_lo is None or config.kan_grid_range_hi is None:
            raise ValueError("KAN fallback grid range must be configured")
        dtype = torch.float64
        device = torch.device(config.device)
        support = _coerce_support(
            initial_support,
            (float(config.kan_grid_range_lo), float(config.kan_grid_range_hi)),
            input_dim=int(config.kan_width[0]),
        )
        KAN = import_pykan(config.pykan_root)
        self.kan = KAN(
            width=list(config.kan_width),
            grid=int(config.kan_grid),
            k=int(config.kan_spline_k),
            base_fun=str(config.kan_base_fun),
            symbolic_enabled=False,
            affine_trainable=False,
            grid_eps=float(config.kan_grid_eps),
            grid_range=[float(config.kan_grid_range_lo), float(config.kan_grid_range_hi)],
            noise_scale=float(config.kan_noise_scale),
            auto_save=False,
            save_act=False,
            seed=int(seed),
            device=str(device),
        ).speed()
        self.kan = self.kan.to(device).double()
        self._reset_first_layer_initial_grid(
            support,
            seed=int(seed),
            noise_scale=float(config.kan_noise_scale),
        )

        mean = np.asarray(state_mean, dtype=float).reshape(-1)
        scale = np.asarray(state_scale, dtype=float).reshape(-1)
        expected_shape = (input_dimension,)
        if mean.shape != expected_shape or scale.shape != expected_shape or np.any(scale <= 1.0e-30):
            raise ValueError(f"invalid AFM06a input transform: mean={mean}, scale={scale}")
        self.input_policy = str(config.normalizer_policy)
        if self.input_policy not in {
            RAW_INPUT_POLICY,
            NANOMETER_INPUT_POLICY,
            NORMALIZED_INPUT_POLICY,
            GLOBAL_NORMALIZED_INPUT_POLICY,
        }:
            raise ValueError(f"unsupported AFM06a KAN input policy: {self.input_policy!r}")
        if self.input_policy == RAW_INPUT_POLICY and (
            not np.array_equal(mean, np.zeros(input_dimension, dtype=float))
            or not np.array_equal(scale, np.ones(input_dimension, dtype=float))
        ):
            raise ValueError(
                "raw-state KAN input requires identity transform statistics: "
                f"mean={mean}, scale={scale}"
            )
        if self.input_policy == NANOMETER_INPUT_POLICY and (
            not np.array_equal(mean, np.zeros(input_dimension, dtype=float))
            or not np.array_equal(
                scale,
                np.full(input_dimension, METERS_PER_NANOMETER, dtype=float),
            )
        ):
            raise ValueError(
                "nanometer KAN input requires x1_nm = x1_m / 1e-9: "
                f"mean={mean}, scale={scale}"
            )
        self.register_buffer("state_mean", torch.as_tensor(mean, dtype=dtype, device=device))
        self.register_buffer("state_scale", torch.as_tensor(scale, dtype=dtype, device=device))
        self.force_output_policy = str(force_output_policy)
        if self.force_output_policy not in {
            IDENTITY_FORCE_OUTPUT_POLICY,
            LOCAL_FORCE_OUTPUT_POLICY,
            GLOBAL_FORCE_OUTPUT_POLICY,
        }:
            raise ValueError(f"unsupported AFM06a force-output policy: {self.force_output_policy!r}")
        force_mean_value = float(force_mean)
        force_scale_value = float(force_scale)
        if not np.isfinite(force_mean_value):
            raise ValueError(f"invalid AFM06a force mean: {force_mean_value}")
        if not np.isfinite(force_scale_value) or force_scale_value <= 1.0e-30:
            raise ValueError(f"invalid AFM06a force scale: {force_scale_value}")
        if self.force_output_policy == IDENTITY_FORCE_OUTPUT_POLICY:
            force_mean_value = 0.0
            force_scale_value = 1.0
        self.register_buffer(
            "force_mean",
            torch.tensor(force_mean_value, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "force_scale",
            torch.tensor(force_scale_value, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer("initial_grid_support", torch.as_tensor(support, dtype=dtype, device=device))
        self.gain_enabled = bool(config.gain_enabled)
        self.log_gnn = nn.Parameter(
            torch.zeros(1, dtype=dtype, device=device),
            requires_grad=self.gain_enabled and bool(config.gain_learnable),
        )

        settings = AFM06aHardSampleInputs() if settings is None else settings
        settings.validate()
        self.dist = float(settings.dist)
        self.a0 = float(settings.a0)
        self.soft_mask_enabled = bool(config.soft_mask_enabled)
        self.soft_mask_trainable = bool(config.soft_mask_trainable)
        a0_safe = max(abs(self.a0), 1.0e-30)
        s0_min = max(float(config.soft_mask_s0_min_a0), 1.0e-6) * a0_safe
        s0_max = max(float(config.soft_mask_s0_max_a0), float(config.soft_mask_s0_min_a0) + 1.0e-6) * a0_safe
        s0_init = min(max(float(config.soft_mask_s0_a0) * a0_safe, s0_min), s0_max)
        alpha_min = max(float(config.soft_mask_alpha_min_a0), 1.0e-9) / a0_safe
        alpha_max = max(
            float(config.soft_mask_alpha_max_a0),
            float(config.soft_mask_alpha_min_a0) + 1.0e-9,
        ) / a0_safe
        alpha_init = min(max(float(config.soft_mask_alpha_a0) / a0_safe, alpha_min), alpha_max)
        self.register_buffer("soft_mask_s0_min", torch.tensor(s0_min, dtype=dtype, device=device))
        self.register_buffer("soft_mask_s0_max", torch.tensor(s0_max, dtype=dtype, device=device))
        self.register_buffer("soft_mask_alpha_min", torch.tensor(alpha_min, dtype=dtype, device=device))
        self.register_buffer("soft_mask_alpha_max", torch.tensor(alpha_max, dtype=dtype, device=device))
        self.soft_mask_s0_raw = nn.Parameter(
            torch.tensor([_inverse_sigmoid_fraction(s0_init, s0_min, s0_max)], dtype=dtype, device=device),
            requires_grad=self.soft_mask_enabled and self.soft_mask_trainable,
        )
        self.soft_mask_alpha_raw = nn.Parameter(
            torch.tensor(
                [_inverse_sigmoid_fraction(alpha_init, alpha_min, alpha_max)],
                dtype=dtype,
                device=device,
            ),
            requires_grad=self.soft_mask_enabled and self.soft_mask_trainable,
        )
        self.construction = {
            "width": list(config.kan_width),
            "grid": int(config.kan_grid),
            "spline_k": int(config.kan_spline_k),
            "base_fun": str(config.kan_base_fun),
            "noise_scale": float(config.kan_noise_scale),
            "grid_eps": float(config.kan_grid_eps),
            "fallback_grid_range": [float(config.kan_grid_range_lo), float(config.kan_grid_range_hi)],
            "input_policy": self.input_policy,
            "force_output_policy": self.force_output_policy,
            "force_output_quantity": "Fts_N",
            "seed": int(seed),
            "dtype": "float64",
            "device": str(device),
            "routing_policy": "single_x1_branch",
        }

    @torch.no_grad()
    def _reset_first_layer_initial_grid(self, support: np.ndarray, *, seed: int, noise_scale: float) -> None:
        from kan.spline import curve2coef, extend_grid

        layer = self.kan.act_fun[0]
        base_grid = torch.stack(
            [
                torch.linspace(float(lo), float(hi), steps=int(layer.num) + 1, dtype=layer.grid.dtype, device=layer.grid.device)
                for lo, hi in support
            ],
            dim=0,
        )
        grid = extend_grid(base_grid, k_extend=int(layer.k))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1_000_003)
        noises = torch.rand(
            (int(layer.num) + 1, int(layer.in_dim), int(layer.out_dim)),
            generator=generator,
            dtype=torch.float64,
        ) - 0.5
        noises = (noises * float(noise_scale) / max(int(layer.num), 1)).to(
            device=layer.grid.device,
            dtype=layer.grid.dtype,
        )
        coef = curve2coef(grid[:, int(layer.k) : -int(layer.k)].permute(1, 0), noises, grid, int(layer.k))
        layer.grid.data = grid
        layer.coef.data = coef.to(device=layer.grid.device, dtype=layer.grid.dtype)

    def normalized_inputs(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.state_mean.dtype, device=self.state_mean.device)
        if states.shape[-1] != 2:
            raise ValueError(f"AFM06a force module expects final dimension 2, got {states.shape}")
        return (states[..., [0]] - self.state_mean) / self.state_scale

    def kan_inputs(self, states: torch.Tensor) -> torch.Tensor:
        """Return the coordinates consumed by pykan under the saved input policy."""

        return self.normalized_inputs(states)

    def raw_output(self, states: torch.Tensor) -> torch.Tensor:
        """Return the raw pykan head output before any output affine map."""

        return self.kan(self.kan_inputs(states)).reshape(-1)

    def denormalized_raw_output(self, states: torch.Tensor) -> torch.Tensor:
        """Map the raw KAN output to physical ``Fts`` units."""

        return self.force_mean + self.force_scale * self.raw_output(states)

    def gain(self) -> torch.Tensor:
        if not self.gain_enabled:
            return torch.ones((), dtype=self.log_gnn.dtype, device=self.log_gnn.device)
        return torch.exp(self.log_gnn)

    def soft_mask_s0(self) -> torch.Tensor:
        frac = torch.sigmoid(self.soft_mask_s0_raw.reshape(()))
        return self.soft_mask_s0_min + (self.soft_mask_s0_max - self.soft_mask_s0_min) * frac

    def soft_mask_alpha(self) -> torch.Tensor:
        frac = torch.sigmoid(self.soft_mask_alpha_raw.reshape(()))
        return self.soft_mask_alpha_min + (self.soft_mask_alpha_max - self.soft_mask_alpha_min) * frac

    def soft_mask(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.state_mean.dtype, device=self.state_mean.device)
        if not self.soft_mask_enabled:
            return torch.ones(states.shape[0], dtype=states.dtype, device=states.device)
        separation = self.dist + states[:, 0]
        contact = separation <= self.a0
        noncontact = torch.sigmoid(self.soft_mask_alpha() * (self.soft_mask_s0() - separation))
        return torch.where(contact, torch.ones_like(noncontact), noncontact)

    def masked_raw_output(self, states: torch.Tensor) -> torch.Tensor:
        return self.soft_mask(states) * self.raw_output(states)

    def masked_output(self, states: torch.Tensor) -> torch.Tensor:
        return self.soft_mask(states) * self.denormalized_raw_output(states)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.gain() * self.masked_output(states)

    @torch.no_grad()
    def update_grid_from_states(self, states: torch.Tensor) -> None:
        self.kan.update_grid_from_samples(self.kan_inputs(states))

    @torch.no_grad()
    def initialize_gain_from_reference(self, states: torch.Tensor, reference: torch.Tensor) -> float:
        if not self.gain_enabled:
            self.log_gnn.zero_()
            return 1.0
        masked_raw = self.masked_output(states)
        amp_ref = torch.quantile(torch.abs(reference.reshape(-1)), 0.95)
        amp_raw = torch.quantile(torch.abs(masked_raw.reshape(-1)), 0.95)
        if not torch.isfinite(amp_ref) or float(amp_ref) <= 1.0e-18:
            gain = torch.ones((), dtype=self.log_gnn.dtype, device=self.log_gnn.device)
        else:
            gain = amp_ref / torch.clamp(amp_raw, min=1.0e-12)
        self.log_gnn.data.copy_(torch.log(torch.clamp(gain, min=1.0e-12)).reshape_as(self.log_gnn))
        return float(self.gain().detach().cpu().item())

    def soft_mask_summary(self) -> dict[str, float | bool]:
        with torch.no_grad():
            s0 = float(self.soft_mask_s0().detach().cpu())
            alpha = float(self.soft_mask_alpha().detach().cpu())
        return {
            "enabled": self.soft_mask_enabled,
            "trainable_in_st1pl": self.soft_mask_trainable,
            "s0": s0,
            "s0_a0": s0 / max(abs(self.a0), 1.0e-30),
            "alpha": alpha,
            "alpha_a0": alpha * max(abs(self.a0), 1.0e-30),
        }

    def frozen_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}

    def metadata(self) -> dict[str, object]:
        state = self.frozen_state_dict()
        return {
            "construction": dict(self.construction),
            "state_mean": self.state_mean.detach().cpu().numpy().copy(),
            "state_scale": self.state_scale.detach().cpu().numpy().copy(),
            "force_mean": float(self.force_mean.detach().cpu()),
            "force_scale": float(self.force_scale.detach().cpu()),
            "force_output_policy": self.force_output_policy,
            "force_output_quantity": "Fts_N",
            "initial_grid_support": self.initial_grid_support.detach().cpu().numpy().copy(),
            "input_policy": self.input_policy,
            "gain_enabled": self.gain_enabled,
            "nn_gain": float(self.gain().detach().cpu()),
            "soft_mask": self.soft_mask_summary(),
            "state_dict_sha256": state_dict_digest(state),
        }


def build_kan_runtime(
    config: Stage1PlusLightConfig,
    *,
    nn_init_seed: int,
    training_states: np.ndarray,
    gain_reference: np.ndarray,
    global_state_mean: np.ndarray | None = None,
    global_state_scale: np.ndarray | None = None,
    global_force_mean: float | None = None,
    global_force_scale: float | None = None,
    settings: AFM06aHardSampleInputs | None = None,
) -> KANForceModule:
    configure_deterministic_torch(int(nn_init_seed))
    width = tuple(int(value) for value in config.kan_width)
    if width != SINGLE_INPUT_KAN_WIDTH:
        raise ValueError(f"AFM06a requires the single-input 1-3-1 KAN, got {width}")
    policy = str(config.normalizer_policy)
    if policy == RAW_INPUT_POLICY:
        mean = np.zeros(1, dtype=float)
        scale = np.ones(1, dtype=float)
        support = raw_support_from_inputs(training_states)
    elif policy == NANOMETER_INPUT_POLICY:
        mean = np.zeros(1, dtype=float)
        scale = np.asarray([METERS_PER_NANOMETER], dtype=float)
        support = normalized_support_from_raw_inputs(training_states, mean, scale)
    elif policy == NORMALIZED_INPUT_POLICY:
        mean, scale = training_state_normalizer(training_states)
        support = normalized_support_from_raw_inputs(training_states, mean, scale)
    elif policy == GLOBAL_NORMALIZED_INPUT_POLICY:
        if global_state_mean is None or global_state_scale is None:
            raise ValueError("global input statistics are required by the AFM06a formal policy")
        mean = np.asarray(global_state_mean, dtype=float).reshape(-1)[:1]
        scale = np.asarray(global_state_scale, dtype=float).reshape(-1)[:1]
        support = normalized_support_from_raw_inputs(training_states, mean, scale)
    else:
        raise ValueError(f"unsupported AFM06a KAN input policy: {policy!r}")
    output_policy = str(config.force_output_policy)
    if output_policy == LOCAL_FORCE_OUTPUT_POLICY:
        force_mean, force_scale = training_force_normalizer(gain_reference)
    elif output_policy == GLOBAL_FORCE_OUTPUT_POLICY:
        if global_force_mean is None or global_force_scale is None:
            raise ValueError("global force statistics are required by the AFM06a formal policy")
        force_mean = float(global_force_mean)
        force_scale = float(global_force_scale)
    elif output_policy == IDENTITY_FORCE_OUTPUT_POLICY:
        force_mean = 0.0
        force_scale = 1.0
    else:
        raise ValueError(f"unsupported AFM06a force-output policy: {output_policy!r}")
    model = KANForceModule(
        config=config,
        state_mean=mean,
        state_scale=scale,
        seed=int(nn_init_seed),
        initial_support=support,
        force_mean=force_mean,
        force_scale=force_scale,
        force_output_policy=output_policy,
        settings=settings,
    ).to(config.device)
    states_t = torch.as_tensor(training_states, dtype=torch.float64, device=config.device)
    reference_t = torch.as_tensor(gain_reference, dtype=torch.float64, device=config.device)
    with torch.no_grad():
        if config.adaptive_grid_enabled:
            model.update_grid_from_states(states_t)
        model.initialize_gain_from_reference(states_t, reference_t)
    return model


__all__ = [
    "KANForceModule",
    "build_kan_runtime",
    "configure_deterministic_torch",
    "GLOBAL_FORCE_OUTPUT_POLICY",
    "GLOBAL_NORMALIZED_INPUT_POLICY",
    "IDENTITY_FORCE_OUTPUT_POLICY",
    "import_pykan",
    "LOCAL_FORCE_OUTPUT_POLICY",
    "SINGLE_INPUT_KAN_WIDTH",
    "METERS_PER_NANOMETER",
    "NANOMETER_INPUT_POLICY",
    "NORMALIZED_INPUT_POLICY",
    "normalized_support_from_raw_inputs",
    "RAW_INPUT_POLICY",
    "raw_support_from_inputs",
    "state_dict_digest",
    "training_force_normalizer",
    "training_state_normalizer",
]
