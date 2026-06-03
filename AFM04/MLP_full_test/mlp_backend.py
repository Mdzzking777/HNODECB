"""Torch MLP force module for the isolated AFM04 full functional test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_eps(x: torch.Tensor, eps: float) -> torch.Tensor:
    return float(eps) * F.softplus(x / float(eps))


def _activation(name: str) -> nn.Module:
    name = str(name).strip().lower()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    return nn.SiLU()


def _build_mlp(width: tuple[int, ...], activation_name: str) -> nn.Sequential:
    if len(width) < 2:
        raise ValueError("MLP width must include at least input and output dimensions")
    layers: list[nn.Module] = []
    last_layer_index = len(width) - 2
    for layer_index, (in_features, out_features) in enumerate(zip(width[:-1], width[1:])):
        layers.append(nn.Linear(int(in_features), int(out_features)))
        if layer_index != last_layer_index:
            layers.append(_activation(activation_name))
    return nn.Sequential(*layers)


def w_pred_from_states(
    states: torch.Tensor,
    *,
    dist: float,
    a0: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.as_tensor(states)
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError("w_pred_from_states expects states shaped [N, 3]")
    s_pred = float(dist) + states[:, 0] - states[:, 2]
    delta = _softplus_eps(float(a0) - s_pred, float(eps))
    weights = 1.0 - torch.exp(-delta / float(eps))
    return s_pred, torch.clamp(weights, min=0.0, max=1.0)


def summarize_wpred_from_states(
    states: torch.Tensor,
    *,
    dist: float,
    a0: float,
    eps: float,
) -> dict[str, float]:
    states = torch.as_tensor(states)
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError("summarize_wpred_from_states expects states shaped [N, 3]")
    n = int(states.shape[0])
    meta = {
        "base_samples": float(n),
        "extra_samples": 0.0,
        "total_samples": float(n),
        "mean_w_pred": float("nan"),
        "max_w_pred": float("nan"),
        "q50_w_pred": float("nan"),
        "q90_w_pred": float("nan"),
        "q99_w_pred": float("nan"),
        "frac_s_le_a0": float("nan"),
        "frac_s_le_1p5a0": float("nan"),
        "frac_wpred_ge_05": float("nan"),
        "frac_wpred_ge_08": float("nan"),
    }
    if n <= 0:
        return meta

    s_pred, weights = w_pred_from_states(states, dist=dist, a0=a0, eps=eps)
    if torch.any(~torch.isfinite(weights)):
        return meta

    meta["mean_w_pred"] = float(torch.mean(weights).detach())
    meta["max_w_pred"] = float(torch.max(weights).detach())
    meta["q50_w_pred"] = float(torch.quantile(weights, 0.50).detach())
    meta["q90_w_pred"] = float(torch.quantile(weights, 0.90).detach())
    meta["q99_w_pred"] = float(torch.quantile(weights, 0.99).detach())
    meta["frac_s_le_a0"] = float(torch.mean((s_pred <= float(a0)).to(weights.dtype)).detach())
    meta["frac_s_le_1p5a0"] = float(torch.mean((s_pred <= (1.5 * float(a0))).to(weights.dtype)).detach())
    meta["frac_wpred_ge_05"] = float(torch.mean((weights >= 0.5).to(weights.dtype)).detach())
    meta["frac_wpred_ge_08"] = float(torch.mean((weights >= 0.8).to(weights.dtype)).detach())
    return meta


class MLPForceModule(nn.Module):
    def __init__(
        self,
        *,
        mlp_root: str | Path | None = None,
        state_mean: np.ndarray,
        state_scale: np.ndarray,
        seed: int,
        width: tuple[int, ...],
        grid: int | None = None,
        spline_k: int | None = None,
        base_fun: str = "silu",
        symbolic_enabled: bool = False,
        auto_save: bool = False,
        noise_scale: float = 0.0,
        affine_trainable: bool = False,
        grid_eps: float = 0.0,
        grid_range: tuple[float, float] | None = None,
        gnn_learnable: bool = True,
        dist: float = 0.0,
        a0: float = 0.0,
        wpred_enabled: bool = False,
        wpred_eps: float = 1.0e-10,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        del mlp_root, grid, spline_k, symbolic_enabled, auto_save, affine_trainable, grid_eps, grid_range

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.mlp = _build_mlp(tuple(int(x) for x in width), base_fun)
            with torch.no_grad():
                for module in self.mlp:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                        if noise_scale > 0.0:
                            module.weight.add_(float(noise_scale) * torch.randn_like(module.weight))

        self.mlp = self.mlp.to(device=str(device), dtype=dtype)
        self.register_buffer("state_mean", torch.as_tensor(np.asarray(state_mean, dtype=float), dtype=dtype))
        self.register_buffer("state_scale", torch.as_tensor(np.asarray(state_scale, dtype=float), dtype=dtype))
        self.log_gnn = nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=bool(gnn_learnable))
        self.dist = float(dist)
        self.a0 = float(a0)
        self.wpred_enabled = bool(wpred_enabled)
        self.wpred_eps = float(wpred_eps)

    def normalized_inputs(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.state_mean.dtype, device=self.state_mean.device)
        return (states - self.state_mean) / self.state_scale

    def raw_output(self, states: torch.Tensor) -> torch.Tensor:
        x = self.normalized_inputs(states)
        y = self.mlp(x)
        return y.reshape(-1)

    def w_pred(self, states: torch.Tensor) -> torch.Tensor:
        states_t = torch.as_tensor(states, dtype=self.state_mean.dtype, device=self.state_mean.device)
        if not self.wpred_enabled:
            return torch.ones(states_t.shape[0], dtype=states_t.dtype, device=states_t.device)
        _, weights = w_pred_from_states(
            states_t,
            dist=self.dist,
            a0=self.a0,
            eps=self.wpred_eps,
        )
        return weights.to(dtype=states_t.dtype, device=states_t.device)

    def weighted_raw_output(self, states: torch.Tensor) -> torch.Tensor:
        raw = self.raw_output(states)
        return self.w_pred(states) * raw

    def gain(self) -> torch.Tensor:
        return torch.exp(self.log_gnn)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.gain() * self.weighted_raw_output(states)

    @torch.no_grad()
    def initialize_gain_from_truth(self, states: torch.Tensor, fts_true: torch.Tensor) -> float:
        raw = self.weighted_raw_output(states)
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
        del states


__all__ = [
    "MLPForceModule",
    "summarize_wpred_from_states",
    "w_pred_from_states",
]
