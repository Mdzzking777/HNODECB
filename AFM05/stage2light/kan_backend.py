"""pykan wrapper used as the AFM05 formal force module."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


def import_pykan(pykan_root: str | Path):
    pykan_root = Path(pykan_root)
    if str(pykan_root) not in sys.path:
        sys.path.insert(0, str(pykan_root))
    from kan import KAN

    return KAN


def _inverse_sigmoid_fraction(value: float, lo: float, hi: float) -> float:
    frac = (float(value) - float(lo)) / max(float(hi) - float(lo), 1.0e-30)
    frac = min(max(frac, 1.0e-12), 1.0 - 1.0e-12)
    return float(np.log(frac / (1.0 - frac)))


def _coerce_initial_grid_support(
    support: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    input_dim: int,
    fallback_range: tuple[float, float],
) -> np.ndarray:
    if support is None:
        lo, hi = float(fallback_range[0]), float(fallback_range[1])
        support_arr = np.tile(np.asarray([[lo, hi]], dtype=float), (int(input_dim), 1))
    else:
        support_arr = np.asarray(support, dtype=float)
        if support_arr.shape != (int(input_dim), 2):
            raise ValueError(
                "initial_grid_support must have shape "
                f"({int(input_dim)}, 2), got {support_arr.shape}"
            )
    if not np.all(np.isfinite(support_arr)):
        raise ValueError(f"initial_grid_support contains nonfinite value(s): {support_arr}")
    if np.any(support_arr[:, 1] <= support_arr[:, 0]):
        raise ValueError(f"initial_grid_support requires hi > lo for every input: {support_arr}")
    return support_arr


def initial_grid_support_from_raw_inputs(
    raw_inputs: torch.Tensor | np.ndarray,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
) -> np.ndarray:
    """Return per-input normalized min/max support from raw-coordinate AGU inputs."""

    values = raw_inputs.detach().cpu().numpy() if isinstance(raw_inputs, torch.Tensor) else np.asarray(raw_inputs)
    values = np.asarray(values, dtype=float)
    mean = np.asarray(state_mean, dtype=float).reshape(-1)
    scale = np.asarray(state_scale, dtype=float).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"raw_inputs must be 2D, got shape {values.shape}")
    if values.shape[1] != mean.size or values.shape[1] != scale.size:
        raise ValueError(
            "raw input / normalizer dimension mismatch: "
            f"raw={values.shape[1]} mean={mean.size} scale={scale.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("raw_inputs contains nonfinite value(s)")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(np.abs(scale) <= 1.0e-30):
        raise ValueError(f"invalid state normalizer: mean={mean} scale={scale}")
    norm = (values - mean.reshape(1, -1)) / scale.reshape(1, -1)
    support = np.column_stack([np.min(norm, axis=0), np.max(norm, axis=0)])
    if np.any(support[:, 1] <= support[:, 0]):
        raise ValueError(f"degenerate initial grid support from samples: {support}")
    return support.astype(float, copy=False)


def initial_grid_support_to_meta(
    support: Sequence[Sequence[float]] | np.ndarray,
    *,
    source: str,
) -> dict[str, float | str | list[list[float]]]:
    arr = np.asarray(support, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"initial grid support must have shape (n, 2), got {arr.shape}")
    meta: dict[str, float | str | list[list[float]]] = {
        "source": str(source),
        "support": arr.tolist(),
    }
    labels = ("x1", "x2", "x3")
    for idx, label in enumerate(labels[: arr.shape[0]]):
        meta[f"{label}_min"] = float(arr[idx, 0])
        meta[f"{label}_max"] = float(arr[idx, 1])
    return meta


class KANForceModule(nn.Module):
    def __init__(
        self,
        *,
        pykan_root: str | Path,
        state_mean: np.ndarray,
        state_scale: np.ndarray,
        seed: int,
        width: tuple[int, int, int],
        grid: int,
        spline_k: int,
        base_fun: str,
        symbolic_enabled: bool,
        auto_save: bool,
        noise_scale: float,
        affine_trainable: bool,
        grid_eps: float,
        grid_range: tuple[float, float],
        initial_grid_support: Sequence[Sequence[float]] | np.ndarray | None = None,
        gnn_learnable: bool,
        dist: float = 0.0,
        a0: float = 0.0,
        soft_mask_enabled: bool = True,
        soft_mask_trainable: bool = True,
        soft_mask_s0_a0: float = 20.0,
        soft_mask_s0_min_a0: float = 1.0,
        soft_mask_s0_max_a0: float = 100.0,
        soft_mask_alpha_a0: float = 0.25,
        soft_mask_alpha_min_a0: float = 0.02,
        soft_mask_alpha_max_a0: float = 5.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        KAN = import_pykan(pykan_root)
        self.kan = KAN(
            width=list(width),
            grid=int(grid),
            k=int(spline_k),
            base_fun=base_fun,
            symbolic_enabled=bool(symbolic_enabled),
            affine_trainable=bool(affine_trainable),
            grid_eps=float(grid_eps),
            grid_range=[float(grid_range[0]), float(grid_range[1])],
            noise_scale=float(noise_scale),
            auto_save=bool(auto_save),
            save_act=False,
            seed=int(seed),
            device=str(device),
        ).speed()
        self.kan = self.kan.to(str(device))
        self.kan = self.kan.double() if dtype == torch.float64 else self.kan.float()
        input_dim = int(width[0])
        support_arr = _coerce_initial_grid_support(
            initial_grid_support,
            input_dim=input_dim,
            fallback_range=(float(grid_range[0]), float(grid_range[1])),
        )
        if initial_grid_support is not None:
            self._reset_first_layer_initial_grid(
                support_arr,
                seed=int(seed),
                noise_scale=float(noise_scale),
            )
        mean_arr = np.asarray(state_mean, dtype=float).reshape(-1)
        scale_arr = np.asarray(state_scale, dtype=float).reshape(-1)
        if mean_arr.size != input_dim or scale_arr.size != input_dim:
            raise ValueError(
                "state normalizer dimension mismatch: "
                f"width[0]={input_dim} mean={mean_arr.size} scale={scale_arr.size}"
            )
        scale_arr = np.where(np.isfinite(scale_arr) & (np.abs(scale_arr) > 1.0e-30), scale_arr, 1.0)
        self.register_buffer("state_mean", torch.as_tensor(mean_arr, dtype=dtype))
        self.register_buffer("state_scale", torch.as_tensor(scale_arr, dtype=dtype))
        self.register_buffer("initial_grid_support", torch.as_tensor(support_arr, dtype=dtype), persistent=False)
        self.log_gnn = nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=bool(gnn_learnable))
        self.dist = float(dist)
        self.a0 = float(a0)
        self.soft_mask_enabled = bool(soft_mask_enabled)
        self.soft_mask_trainable = bool(soft_mask_trainable)

        a0_safe = max(abs(float(a0)), 1.0e-30)
        s0_min = max(float(soft_mask_s0_min_a0), 1.0e-6) * a0_safe
        s0_max = max(float(soft_mask_s0_max_a0), float(soft_mask_s0_min_a0) + 1.0e-6) * a0_safe
        s0_init = min(max(float(soft_mask_s0_a0) * a0_safe, s0_min), s0_max)

        alpha_min = max(float(soft_mask_alpha_min_a0), 1.0e-9) / a0_safe
        alpha_max = max(float(soft_mask_alpha_max_a0), float(soft_mask_alpha_min_a0) + 1.0e-9) / a0_safe
        alpha_init = min(max(float(soft_mask_alpha_a0) / a0_safe, alpha_min), alpha_max)

        self.register_buffer("soft_mask_s0_min", torch.tensor(s0_min, dtype=dtype))
        self.register_buffer("soft_mask_s0_max", torch.tensor(s0_max, dtype=dtype))
        self.register_buffer("soft_mask_alpha_min", torch.tensor(alpha_min, dtype=dtype))
        self.register_buffer("soft_mask_alpha_max", torch.tensor(alpha_max, dtype=dtype))
        self.soft_mask_s0_raw = nn.Parameter(
            torch.tensor([_inverse_sigmoid_fraction(s0_init, s0_min, s0_max)], dtype=dtype),
            requires_grad=self.soft_mask_enabled and self.soft_mask_trainable,
        )
        self.soft_mask_alpha_raw = nn.Parameter(
            torch.tensor([_inverse_sigmoid_fraction(alpha_init, alpha_min, alpha_max)], dtype=dtype),
            requires_grad=self.soft_mask_enabled and self.soft_mask_trainable,
        )

    @torch.no_grad()
    def _reset_first_layer_initial_grid(self, support: np.ndarray, *, seed: int, noise_scale: float) -> None:
        """Reinitialize the first KAN layer on per-input grid support before training."""

        from kan.spline import curve2coef, extend_grid

        if not getattr(self.kan, "act_fun", None):
            raise RuntimeError("pykan model has no activation layers to initialize")
        layer = self.kan.act_fun[0]
        in_dim = int(layer.in_dim)
        out_dim = int(layer.out_dim)
        num = int(layer.num)
        k = int(layer.k)
        support_arr = _coerce_initial_grid_support(
            support,
            input_dim=in_dim,
            fallback_range=(float(support[0, 0]), float(support[0, 1])),
        )
        device = layer.grid.device
        dtype = layer.grid.dtype
        base_grid = torch.stack(
            [
                torch.linspace(float(lo), float(hi), steps=num + 1, dtype=dtype, device=device)
                for lo, hi in support_arr
            ],
            dim=0,
        )
        grid = extend_grid(base_grid, k_extend=k)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1_000_003)
        noises = (torch.rand((num + 1, in_dim, out_dim), generator=generator, dtype=torch.float64) - 0.5)
        noises = (noises * float(noise_scale) / max(num, 1)).to(device=device, dtype=dtype)
        coef = curve2coef(grid[:, k:-k].permute(1, 0), noises, grid, k)
        layer.grid.data = grid
        layer.coef.data = coef.to(device=device, dtype=dtype)

    def normalized_inputs(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.state_mean.dtype, device=self.state_mean.device)
        return (states - self.state_mean) / self.state_scale

    @torch.no_grad()
    def set_state_normalizer(self, state_mean: np.ndarray, state_scale: np.ndarray) -> None:
        mean_arr = np.asarray(state_mean, dtype=float).reshape(-1)
        scale_arr = np.asarray(state_scale, dtype=float).reshape(-1)
        if mean_arr.size != self.state_mean.numel() or scale_arr.size != self.state_scale.numel():
            raise ValueError(
                "state normalizer dimension mismatch: "
                f"expected={self.state_mean.numel()} mean={mean_arr.size} scale={scale_arr.size}"
            )
        if not np.all(np.isfinite(mean_arr)):
            raise ValueError(f"state_mean contains nonfinite value(s): {mean_arr}")
        if not np.all(np.isfinite(scale_arr)) or np.any(np.abs(scale_arr) <= 1.0e-30):
            raise ValueError(f"state_scale contains invalid value(s): {scale_arr}")
        self.state_mean.copy_(torch.as_tensor(mean_arr, dtype=self.state_mean.dtype, device=self.state_mean.device))
        self.state_scale.copy_(torch.as_tensor(scale_arr, dtype=self.state_scale.dtype, device=self.state_scale.device))

    def raw_output(self, states: torch.Tensor) -> torch.Tensor:
        x = self.normalized_inputs(states)
        y = self.kan(x)
        return y.reshape(-1)

    def gain(self) -> torch.Tensor:
        return torch.exp(self.log_gnn)

    def soft_mask_s0(self) -> torch.Tensor:
        frac = torch.sigmoid(self.soft_mask_s0_raw.reshape(()))
        return self.soft_mask_s0_min + (self.soft_mask_s0_max - self.soft_mask_s0_min) * frac

    def soft_mask_alpha(self) -> torch.Tensor:
        frac = torch.sigmoid(self.soft_mask_alpha_raw.reshape(()))
        return self.soft_mask_alpha_min + (self.soft_mask_alpha_max - self.soft_mask_alpha_min) * frac

    def soft_mask(self, states: torch.Tensor) -> torch.Tensor:
        states_t = torch.as_tensor(states, dtype=self.state_mean.dtype, device=self.state_mean.device)
        if not self.soft_mask_enabled:
            return torch.ones(states_t.shape[0], dtype=states_t.dtype, device=states_t.device)
        s = self.dist + states_t[:, 0] - states_t[:, 2]
        contact_gate = s <= float(self.a0)
        noncontact_mask = torch.sigmoid(self.soft_mask_alpha() * (self.soft_mask_s0() - s))
        return torch.where(contact_gate, torch.ones_like(noncontact_mask), noncontact_mask)

    def soft_mask_summary(self) -> dict[str, float | bool]:
        with torch.no_grad():
            s0 = float(self.soft_mask_s0().detach().cpu())
            alpha = float(self.soft_mask_alpha().detach().cpu())
        a0_safe = max(abs(float(self.a0)), 1.0e-30)
        return {
            "enabled": bool(self.soft_mask_enabled),
            "trainable": bool(self.soft_mask_trainable),
            "s0": s0,
            "s0_a0": s0 / a0_safe,
            "alpha": alpha,
            "alpha_a0": alpha * a0_safe,
            "m_min": 0.0,
        }

    def masked_raw_output(self, states: torch.Tensor) -> torch.Tensor:
        return self.soft_mask(states) * self.raw_output(states)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.gain() * self.masked_raw_output(states)

    @torch.no_grad()
    def initialize_gain_from_reference(self, states: torch.Tensor, force_reference: torch.Tensor) -> float:
        raw = self.masked_raw_output(states)
        amp_true = torch.quantile(torch.abs(force_reference.reshape(-1)), 0.95)
        amp_pred = torch.quantile(torch.abs(raw.reshape(-1)), 0.95)
        if not torch.isfinite(amp_true) or float(amp_true) <= 1.0e-18:
            gain = torch.tensor(1.0, dtype=self.log_gnn.dtype, device=self.log_gnn.device)
        else:
            gain = amp_true / torch.clamp(amp_pred, min=1.0e-12)
        self.log_gnn.data = torch.log(torch.clamp(gain, min=1.0e-12)).reshape_as(self.log_gnn)
        return float(torch.exp(self.log_gnn.detach())[0])

    @torch.no_grad()
    def initialize_gain_from_truth(self, states: torch.Tensor, fts_true: torch.Tensor) -> float:
        return self.initialize_gain_from_reference(states, fts_true)

    @torch.no_grad()
    def update_grid_from_states(self, states: torch.Tensor) -> None:
        x = self.normalized_inputs(states)
        self.kan.update_grid_from_samples(x)

    @torch.no_grad()
    def update_grid_from_normalized_inputs(self, inputs: torch.Tensor) -> None:
        """Legacy name: inputs are raw physical coordinates before normalization."""

        x = self.normalized_inputs(inputs)
        self.kan.update_grid_from_samples(x)


__all__ = [
    "KANForceModule",
    "import_pykan",
    "initial_grid_support_from_raw_inputs",
    "initial_grid_support_to_meta",
]
