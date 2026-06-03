"""pykan wrapper used as the AFM04 formal force module."""

from __future__ import annotations

import sys
from pathlib import Path

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
        # Formal AFM04 now follows KFT raw-input coordinates: pykan sees
        # physical [x1, x2, x3] directly, and AGU controls the spline grid.
        # Keep identity buffers only for payload/checkpoint compatibility.
        input_dim = int(width[0])
        mean_arr = np.zeros(input_dim, dtype=float)
        scale_arr = np.ones(input_dim, dtype=float)
        if mean_arr.size != input_dim or scale_arr.size != input_dim:
            raise ValueError(
                "state normalizer dimension mismatch: "
                f"width[0]={input_dim} mean={mean_arr.size} scale={scale_arr.size}"
            )
        self.register_buffer("state_mean", torch.as_tensor(mean_arr, dtype=dtype))
        self.register_buffer("state_scale", torch.as_tensor(scale_arr, dtype=dtype))
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

    def normalized_inputs(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.state_mean.dtype, device=self.state_mean.device)
        return states

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
    def initialize_gain_from_truth(self, states: torch.Tensor, fts_true: torch.Tensor) -> float:
        raw = self.masked_raw_output(states)
        amp_true = torch.quantile(torch.abs(fts_true.reshape(-1)), 0.95)
        amp_pred = torch.quantile(torch.abs(raw.reshape(-1)), 0.95)
        if not torch.isfinite(amp_true) or float(amp_true) <= 1.0e-18:
            gain = torch.tensor(1.0, dtype=self.log_gnn.dtype, device=self.log_gnn.device)
        else:
            gain = amp_true / torch.clamp(amp_pred, min=1.0e-12)
        self.log_gnn.data = torch.log(torch.clamp(gain, min=1.0e-12)).reshape_as(self.log_gnn)
        return float(torch.exp(self.log_gnn.detach())[0])

    @torch.no_grad()
    def update_grid_from_states(self, states: torch.Tensor) -> None:
        x = self.normalized_inputs(states)
        self.kan.update_grid_from_samples(x)

    @torch.no_grad()
    def update_grid_from_normalized_inputs(self, inputs: torch.Tensor) -> None:
        """Legacy name: inputs are raw physical KAN coordinates."""

        x = torch.as_tensor(inputs, dtype=self.state_mean.dtype, device=self.state_mean.device)
        self.kan.update_grid_from_samples(x)


__all__ = [
    "KANForceModule",
    "import_pykan",
]
