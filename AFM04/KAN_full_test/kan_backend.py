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


def _logit_from_fraction(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return float(np.log(value / (1.0 - value)))


def plan_z_x3dot_scale(
    *,
    configured_scale: float,
    scale_mode: str,
    a0: float,
    window_span: float,
    x2_scale: float | None = None,
) -> float:
    # Legacy helper retained for older call sites.  In raw-input KFT mode the
    # fourth Plan Z channel is not normalized by this value.
    configured = float(configured_scale)
    if np.isfinite(configured) and configured > 0.0:
        return configured
    mode = str(scale_mode).strip().lower()
    if mode.replace("-", "_") in ("x2_scale_tenth", "x2_train_scale_tenth", "x2_tenth", "x2_window", "measured_x2", "x2"):
        if x2_scale is not None and np.isfinite(float(x2_scale)) and float(x2_scale) > 0.0:
            return max(float(x2_scale) / 10.0, 1.0e-12)
        return 1.0
    if mode in ("physics", "physical", "a0_over_window"):
        return max(float(a0), 1.0e-30) / max(float(window_span), 1.0e-30)
    return 1.0


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
        width: tuple[int, ...],
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
        soft_mask_enabled: bool = False,
        soft_mask_trainable: bool = False,
        soft_mask_s0_a0: float = 20.0,
        soft_mask_s0_min_a0: float = 1.0,
        soft_mask_s0_max_a0: float = 100.0,
        soft_mask_alpha_a0: float = 0.25,
        soft_mask_alpha_min_a0: float = 0.02,
        soft_mask_alpha_max_a0: float = 5.0,
        x3dot_input_enabled: bool = False,
        x3dot_init_trainable: bool = False,
        x3dot_init_value: float = 0.0,
        x3dot_scale: float = 1.0,
        x3dot_lag_detach: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.x3dot_input_enabled = bool(x3dot_input_enabled)
        self.x3dot_init_trainable = bool(x3dot_init_trainable)
        self.x3dot_lag_detach = bool(x3dot_lag_detach)
        if self.x3dot_input_enabled and int(width[0]) != 4:
            raise ValueError(f"Plan Z x3dot input requires width[0] == 4, got width={width}")
        if (not self.x3dot_input_enabled) and int(width[0]) != 3:
            raise ValueError(f"3-input KFT requires width[0] == 3, got width={width}")
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
        # Raw-input KFT: keep identity state_mean/state_scale buffers only for
        # backward-compatible payloads and logs.  They are not used to transform
        # KAN inputs.
        input_dim = int(width[0])
        mean_arr = np.zeros(input_dim, dtype=float)
        scale_arr = np.ones(input_dim, dtype=float)
        if self.x3dot_input_enabled:
            mean_arr[-1] = 0.0
            scale_arr[-1] = 1.0
        if mean_arr.size != int(width[0]) or scale_arr.size != int(width[0]):
            raise ValueError(
                "state normalizer dimension mismatch: "
                f"width[0]={int(width[0])} mean={mean_arr.size} scale={scale_arr.size}"
            )
        scale_arr = np.maximum(scale_arr, 1.0e-30)
        self.register_buffer("state_mean", torch.as_tensor(mean_arr, dtype=dtype))
        self.register_buffer("state_scale", torch.as_tensor(scale_arr, dtype=dtype))
        q_init_tensor = torch.tensor([float(x3dot_init_value)], dtype=dtype)
        if self.x3dot_input_enabled and self.x3dot_init_trainable:
            self.x3dot_q_init = nn.Parameter(q_init_tensor, requires_grad=True)
        else:
            self.register_buffer("x3dot_q_init", q_init_tensor)
        self.log_gnn = nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=bool(gnn_learnable))
        self.dist = float(dist)
        self.a0 = float(a0)
        self.wpred_enabled = bool(wpred_enabled)
        self.wpred_eps = float(wpred_eps)
        self.soft_mask_enabled = bool(soft_mask_enabled)
        self.soft_mask_trainable = bool(soft_mask_trainable)

        a0_safe = max(float(a0), 1.0e-15)
        s0_min = max(float(soft_mask_s0_min_a0), 1.0e-6) * a0_safe
        s0_max = max(float(soft_mask_s0_max_a0), float(soft_mask_s0_min_a0) + 1.0e-6) * a0_safe
        s0_init = min(max(float(soft_mask_s0_a0) * a0_safe, s0_min), s0_max)
        s0_frac = (s0_init - s0_min) / max(s0_max - s0_min, 1.0e-30)

        alpha_min = max(float(soft_mask_alpha_min_a0), 1.0e-9) / a0_safe
        alpha_max = max(float(soft_mask_alpha_max_a0), float(soft_mask_alpha_min_a0) + 1.0e-9) / a0_safe
        alpha_init = min(max(float(soft_mask_alpha_a0) / a0_safe, alpha_min), alpha_max)
        alpha_frac = (alpha_init - alpha_min) / max(alpha_max - alpha_min, 1.0e-30)

        self.register_buffer("soft_mask_s0_min", torch.tensor(s0_min, dtype=dtype))
        self.register_buffer("soft_mask_s0_max", torch.tensor(s0_max, dtype=dtype))
        self.register_buffer("soft_mask_alpha_min", torch.tensor(alpha_min, dtype=dtype))
        self.register_buffer("soft_mask_alpha_max", torch.tensor(alpha_max, dtype=dtype))
        self.soft_mask_s0_raw = nn.Parameter(
            torch.tensor([_logit_from_fraction(s0_frac)], dtype=dtype),
            requires_grad=self.soft_mask_enabled and self.soft_mask_trainable,
        )
        self.soft_mask_alpha_raw = nn.Parameter(
            torch.tensor([_logit_from_fraction(alpha_frac)], dtype=dtype),
            requires_grad=self.soft_mask_enabled and self.soft_mask_trainable,
        )

    def x3dot_initial_q(self) -> torch.Tensor:
        return self.x3dot_q_init.reshape(())

    def prepare_force_inputs(self, states: torch.Tensor, q_pre: torch.Tensor | None = None) -> torch.Tensor:
        states_t = torch.as_tensor(states, dtype=self.state_mean.dtype, device=self.state_mean.device)
        if not self.x3dot_input_enabled:
            if states_t.ndim != 2 or states_t.shape[1] != 3:
                raise ValueError("3-input KFT force expects states shaped [N, 3]")
            return states_t
        if states_t.ndim != 2 or states_t.shape[1] not in (3, 4):
            raise ValueError("Plan Z force expects states shaped [N, 3] or [N, 4]")
        if states_t.shape[1] == 4:
            return states_t
        if q_pre is None:
            q = self.x3dot_initial_q().expand(states_t.shape[0])
        else:
            q = torch.as_tensor(q_pre, dtype=states_t.dtype, device=states_t.device).reshape(-1)
            if q.numel() == 1:
                q = q.expand(states_t.shape[0])
            if q.numel() != states_t.shape[0]:
                raise ValueError(f"q_pre length mismatch: q={q.numel()} states={states_t.shape[0]}")
        return torch.cat([states_t, q.reshape(-1, 1)], dim=1)

    def normalized_inputs(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.state_mean.dtype, device=self.state_mean.device)
        if self.x3dot_input_enabled and states.ndim == 2 and states.shape[1] == 3:
            states = self.prepare_force_inputs(states)
        return states

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
        s = float(self.dist) + states_t[:, 0] - states_t[:, 2]
        contact = s <= float(self.a0)
        noncontact_mask = torch.sigmoid(self.soft_mask_alpha() * (self.soft_mask_s0() - s))
        return torch.where(contact, torch.ones_like(noncontact_mask), noncontact_mask)

    def soft_mask_summary(self) -> dict[str, float | bool]:
        with torch.no_grad():
            s0 = float(self.soft_mask_s0().detach().cpu())
            alpha = float(self.soft_mask_alpha().detach().cpu())
            a0_safe = max(float(self.a0), 1.0e-15)
        return {
            "enabled": bool(self.soft_mask_enabled),
            "trainable": bool(self.soft_mask_trainable),
            "s0": s0,
            "s0_a0": s0 / a0_safe,
            "alpha": alpha,
            "alpha_a0": alpha * a0_safe,
            "m_min": 0.0,
        }

    def x3dot_summary(self) -> dict[str, float | bool | str]:
        q_init = float(self.x3dot_initial_q().detach().cpu())
        q_mean = float(self.state_mean[-1].detach().cpu()) if self.x3dot_input_enabled else float("nan")
        q_scale = float(self.state_scale[-1].detach().cpu()) if self.x3dot_input_enabled else float("nan")
        return {
            "enabled": bool(self.x3dot_input_enabled),
            "mode": "t_level_lag" if self.x3dot_input_enabled else "off",
            "input_coordinate": "raw_physical",
            "q_init": q_init,
            "q_init_trainable": bool(self.x3dot_input_enabled and self.x3dot_init_trainable),
            "q_mean": q_mean,
            "q_scale": q_scale,
            "lag_detach": bool(self.x3dot_lag_detach),
        }

    def weighted_raw_output(self, states: torch.Tensor) -> torch.Tensor:
        raw = self.raw_output(states)
        return self.w_pred(states) * raw

    def pre_gain_output(self, states: torch.Tensor) -> torch.Tensor:
        return self.soft_mask(states) * self.weighted_raw_output(states)

    def gain(self) -> torch.Tensor:
        return torch.exp(self.log_gnn)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.gain() * self.pre_gain_output(states)

    @torch.no_grad()
    def initialize_gain_from_truth(self, states: torch.Tensor, fts_true: torch.Tensor) -> float:
        raw = self.pre_gain_output(states)
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
        # Legacy method name.  In raw-input KFT this expects raw physical KAN
        # inputs, not normalized coordinates.
        x = torch.as_tensor(inputs, dtype=self.state_mean.dtype, device=self.state_mean.device)
        self.kan.update_grid_from_samples(x)

    @torch.no_grad()
    def update_grid_from_inputs(self, inputs: torch.Tensor) -> None:
        x = torch.as_tensor(inputs, dtype=self.state_mean.dtype, device=self.state_mean.device)
        self.kan.update_grid_from_samples(x)


__all__ = [
    "KANForceModule",
    "import_pykan",
    "plan_z_x3dot_scale",
    "summarize_wpred_from_states",
    "w_pred_from_states",
]
