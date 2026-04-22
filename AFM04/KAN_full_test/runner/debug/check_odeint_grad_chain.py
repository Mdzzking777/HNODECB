"""Check whether rollout gradients truly flow back through odeint to KAN params."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from AFM04.KAN_full_test.config import default_config
from AFM04.KAN_full_test.data import prepare_data
from AFM04.KAN_full_test.kan_backend import KANForceModule
from AFM04.KAN_full_test.losses import (
    CONTACT_LOSS_WEIGHT,
    NONCONTACT_LOSS_WEIGHT,
    SCALE_EPS,
    X3_RANGE_AMP,
    X3_RANGE_EPS,
    X3_RANGE_WEIGHT,
    evaluate_split,
)
from AFM04.KAN_full_test.rollout import rollout_single_shooting_torch


def _torch_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def _subset_indices(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    idx = np.unique(idx)
    if idx[0] != 0:
        idx[0] = 0
    if idx[-1] != n - 1:
        idx[-1] = n - 1
    return idx


def _to_torch_split(split, idx: np.ndarray, *, dtype: torch.dtype, device: str) -> dict[str, torch.Tensor]:
    return {
        "ode_true": torch.as_tensor(split.ode_train[:, idx], dtype=dtype, device=device),
        "x2dot_true": torch.as_tensor(split.x2dot_train[idx], dtype=dtype, device=device),
        "contact_mask": torch.as_tensor(split.contact_train[idx].astype(float), dtype=dtype, device=device),
        "times": torch.as_tensor(split.times_train[idx], dtype=dtype, device=device),
        "fts_true": torch.as_tensor(split.fts_train_true[idx], dtype=dtype, device=device),
        "states_for_init": torch.as_tensor(split.ode_train[:, idx].T, dtype=dtype, device=device),
    }


def _softplus_eps(x: torch.Tensor, eps: float) -> torch.Tensor:
    return eps * torch.nn.functional.softplus(x / eps)


def _weighted_rollout_only_loss(traj: torch.Tensor, ode_true: torch.Tensor, contact_mask: torch.Tensor) -> torch.Tensor:
    weights = torch.where(
        contact_mask > 0.5,
        torch.full_like(contact_mask, CONTACT_LOSS_WEIGHT),
        torch.full_like(contact_mask, NONCONTACT_LOSS_WEIGHT),
    )
    weights_sum = torch.clamp(torch.sum(weights), min=1.0)

    state_true = ode_true[0:2, :]
    state_pred = traj[0:2, :]
    state_scale = torch.sqrt(torch.mean(torch.square(state_true), dim=1))
    x1_err = torch.square((state_true[0, :] - state_pred[0, :]) / state_scale[0])
    x2_err = torch.square((state_true[1, :] - state_pred[1, :]) / state_scale[1])
    state_loss = torch.sum(weights * (x1_err + x2_err)) / weights_sum

    x3_scale = torch.tensor(max(float(X3_RANGE_AMP), float(SCALE_EPS)), dtype=traj.dtype, device=traj.device)
    x3_exceed = torch.abs(traj[2, :]) - X3_RANGE_AMP
    x3_pen = torch.square(_softplus_eps(x3_exceed, X3_RANGE_EPS) / x3_scale)
    x3_range_loss = X3_RANGE_WEIGHT * (torch.sum(weights * x3_pen) / weights_sum)
    return state_loss + x3_range_loss


def _iter_named_params(model: KANForceModule):
    wanted = (
        "log_gnn",
        "kan.act_fun.0.coef",
        "kan.act_fun.0.scale_base",
        "kan.act_fun.0.scale_sp",
        "kan.act_fun.1.coef",
        "kan.act_fun.1.scale_base",
        "kan.act_fun.1.scale_sp",
    )
    for name, param in model.named_parameters():
        if name in wanted:
            yield name, param


def _grad_summary(model: KANForceModule) -> dict[str, dict[str, float | bool | None]]:
    out: dict[str, dict[str, float | bool | None]] = {}
    for name, param in _iter_named_params(model):
        grad = param.grad
        if grad is None:
            out[name] = {"has_grad": False, "l2": None, "max_abs": None}
        else:
            g = grad.detach()
            out[name] = {
                "has_grad": True,
                "l2": float(torch.linalg.vector_norm(g).cpu()),
                "max_abs": float(torch.max(torch.abs(g)).cpu()),
            }
    return out


def _zero_grads(model: KANForceModule) -> None:
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_check(*, shard_index: int, max_points: int, seed: int | None = None) -> dict[str, Any]:
    cfg = default_config()
    dtype = _torch_dtype(cfg.dtype)
    torch.manual_seed(cfg.seed if seed is None else seed)
    np.random.seed(cfg.seed if seed is None else seed)

    prepared = prepare_data(cfg)
    split = prepared.splits[shard_index - 1]
    idx = _subset_indices(len(split.times_train), max_points)
    batch = _to_torch_split(split, idx, dtype=dtype, device=cfg.device)
    mech_true_t = torch.as_tensor(prepared.mech_true, dtype=dtype, device=cfg.device)

    model = KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=int(cfg.seed if seed is None else seed),
        width=cfg.width,
        grid=cfg.grid,
        spline_k=cfg.spline_k,
        base_fun=cfg.base_fun,
        symbolic_enabled=cfg.symbolic_enabled,
        auto_save=cfg.auto_save,
        noise_scale=cfg.noise_scale,
        affine_trainable=cfg.affine_trainable,
        grid_eps=cfg.grid_eps,
        grid_range=(cfg.grid_range_lo, cfg.grid_range_hi),
        gnn_learnable=cfg.gnn_learnable,
        device=cfg.device,
        dtype=dtype,
    ).to(cfg.device)

    if cfg.adaptive_grid_enabled:
        model.update_grid_from_states(batch["states_for_init"])
    init_gnn = model.initialize_gain_from_truth(batch["states_for_init"], batch["fts_true"])

    # 1. Full training loss path used by KFT.
    _zero_grads(model)
    total_full, parts_full, traj_full = evaluate_split(
        force_module=model,
        known_pars=prepared.known_pars,
        mech_true=mech_true_t,
        ode_true=batch["ode_true"],
        x2dot_true=batch["x2dot_true"],
        contact_mask=batch["contact_mask"],
        times=batch["times"],
        ode_method=cfg.ode_method,
        ode_rtol=cfg.ode_rtol,
        ode_atol=cfg.ode_atol,
    )
    total_full.backward()
    grads_full = _grad_summary(model)

    # 2. Rollout-only path: state/x3 losses only, so gradients must come through odeint.
    _zero_grads(model)
    traj_roll = rollout_single_shooting_torch(
        model,
        prepared.known_pars,
        mech_true_t,
        batch["ode_true"][:, 0],
        batch["times"],
        method=cfg.ode_method,
        rtol=cfg.ode_rtol,
        atol=cfg.ode_atol,
    )
    rollout_only_loss = _weighted_rollout_only_loss(traj_roll, batch["ode_true"], batch["contact_mask"])
    rollout_requires_grad = bool(traj_roll.requires_grad)
    rollout_only_loss.backward()
    grads_rollout = _grad_summary(model)

    # 3. Detached control: same rollout-only loss, but cut the trajectory graph on purpose.
    _zero_grads(model)
    traj_detached = traj_roll.detach()
    detached_loss = _weighted_rollout_only_loss(traj_detached, batch["ode_true"], batch["contact_mask"]) + 0.0 * model.log_gnn.sum()
    detached_requires_grad = bool(detached_loss.requires_grad)
    detached_loss.backward()
    grads_detached = _grad_summary(model)

    return {
        "timestamp": _now_stamp(),
        "shard_index": shard_index,
        "window_role": split.role,
        "window_label": split.label,
        "max_points": int(max_points),
        "selected_points": int(len(idx)),
        "seed": int(cfg.seed if seed is None else seed),
        "ode_method": cfg.ode_method,
        "adaptive_grid_enabled": bool(cfg.adaptive_grid_enabled),
        "init_gnn": float(init_gnn),
        "full_loss": float(total_full.detach().cpu()),
        "full_parts": {
            "state": float(parts_full.state),
            "x2dot": float(parts_full.x2dot),
            "x3_range": float(parts_full.x3_range),
            "fts_range": float(parts_full.fts_range),
            "x1_rec": float(parts_full.x1_rec),
            "x3_rec": float(parts_full.x3_rec),
            "nn_err": float(parts_full.fts_teacher_rec),
        },
        "traj_requires_grad": rollout_requires_grad,
        "rollout_only_loss": float(rollout_only_loss.detach().cpu()),
        "detached_rollout_only_loss": float(detached_loss.detach().cpu()),
        "detached_loss_requires_grad": detached_requires_grad,
        "grads_full": grads_full,
        "grads_rollout_only": grads_rollout,
        "grads_rollout_detached": grads_detached,
    }


def _format_grad_block(title: str, grads: dict[str, dict[str, float | bool | None]]) -> list[str]:
    lines = [title]
    for name, info in grads.items():
        if not info["has_grad"]:
            lines.append(f"  {name}: grad=None")
        else:
            lines.append(
                f"  {name}: l2={info['l2']:.6e} maxabs={info['max_abs']:.6e}"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether KFT rollout gradients truly pass through odeint.")
    parser.add_argument("--shard-index", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = run_check(shard_index=int(args.shard_index), max_points=int(args.max_points), seed=args.seed)

    repo_root = Path(__file__).resolve().parents[4]
    log_dir = repo_root / "AFM04" / "KAN_full_test" / "logs" / "debug"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp"]
    out_json = log_dir / f"odeint_grad_chain_check_{stamp}.json"
    out_txt = log_dir / f"odeint_grad_chain_check_{stamp}.txt"

    lines = [
        "=== KFT ODEINT GRAD CHAIN CHECK ===",
        f"timestamp={stamp}",
        f"shard={result['shard_index']} role={result['window_role']} label={result['window_label']}",
        f"selected_points={result['selected_points']} max_points={result['max_points']}",
        f"ode_method={result['ode_method']} adaptive_grid={'ON' if result['adaptive_grid_enabled'] else 'OFF'}",
        f"init_gnn={result['init_gnn']:.6e}",
        f"full_loss={result['full_loss']:.6e}",
        (
            "full_parts: "
            f"state={result['full_parts']['state']:.6e} "
            f"x2dot={result['full_parts']['x2dot']:.6e} "
            f"x3r={result['full_parts']['x3_range']:.6e} "
            f"ftsr={result['full_parts']['fts_range']:.6e} "
            f"x1={result['full_parts']['x1_rec']:.3f}% "
            f"x3={result['full_parts']['x3_rec']:.3f}% "
            f"nn={result['full_parts']['nn_err']:.3f}%"
        ),
        f"traj_requires_grad={result['traj_requires_grad']}",
        f"rollout_only_loss={result['rollout_only_loss']:.6e}",
        f"detached_rollout_only_loss={result['detached_rollout_only_loss']:.6e}",
        f"detached_loss_requires_grad={result['detached_loss_requires_grad']}",
        "",
    ]
    lines.extend(_format_grad_block("grads_full:", result["grads_full"]))
    lines.append("")
    lines.extend(_format_grad_block("grads_rollout_only:", result["grads_rollout_only"]))
    lines.append("")
    lines.extend(_format_grad_block("grads_rollout_detached:", result["grads_rollout_detached"]))
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved: {out_txt}")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()

