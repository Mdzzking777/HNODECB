"""pykan wrapper used as an HNODE force module in the isolated full test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def import_pykan(pykan_root: str | Path):
    pykan_root = Path(pykan_root)
    if str(pykan_root) not in sys.path:
        sys.path.insert(0, str(pykan_root))
    from kan import KAN

    return KAN


def _softplus_eps(x: torch.Tensor, eps: float) -> torch.Tensor:
    return float(eps) * F.softplus(x / float(eps))


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

    mean_w = float(torch.mean(weights).detach())
    max_w = float(torch.max(weights).detach())
    meta["mean_w_pred"] = mean_w
    meta["max_w_pred"] = max_w
    meta["q50_w_pred"] = float(torch.quantile(weights, 0.50).detach())
    meta["q90_w_pred"] = float(torch.quantile(weights, 0.90).detach())
    meta["q99_w_pred"] = float(torch.quantile(weights, 0.99).detach())
    meta["frac_s_le_a0"] = float(torch.mean((s_pred <= float(a0)).to(weights.dtype)).detach())
    meta["frac_s_le_1p5a0"] = float(torch.mean((s_pred <= (1.5 * float(a0))).to(weights.dtype)).detach())
    meta["frac_wpred_ge_05"] = float(torch.mean((weights >= 0.5).to(weights.dtype)).detach())
    meta["frac_wpred_ge_08"] = float(torch.mean((weights >= 0.8).to(weights.dtype)).detach())
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
        gnn_learnable: bool,
        dist: float = 0.0,
        a0: float = 0.0,
        wpred_enabled: bool = False,
        wpred_eps: float = 1.0e-10,
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
        y = self.kan(x)
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
        x = self.normalized_inputs(states)
        self.kan.update_grid_from_samples(x)


__all__ = [
    "KANForceModule",
    "import_pykan",
    "summarize_wpred_from_states",
    "w_pred_from_states",
]
