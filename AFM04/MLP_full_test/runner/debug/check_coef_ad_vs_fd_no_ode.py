"""Check coef AD-vs-FD on direct MLP(states_true) teacher loss, bypassing odeint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from AFM04.MLP_full_test.config import default_config
from AFM04.MLP_full_test.data import prepare_data
from AFM04.MLP_full_test.mlp_backend import MLPForceModule


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


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _teacher_loss(model: MLPForceModule, states_true: torch.Tensor, fts_true: torch.Tensor) -> torch.Tensor:
    pred = model(states_true)
    return torch.mean(torch.square(pred - fts_true.reshape(-1)))


def run_check(*, shard_index: int, max_points: int, rel_eps: float, abs_eps: float, seed: int | None = None) -> dict:
    cfg = default_config()
    dtype = _torch_dtype(cfg.dtype)
    base_seed = int(cfg.seed if seed is None else seed)
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    prepared = prepare_data(cfg)
    split = prepared.splits[shard_index - 1]
    idx = _subset_indices(len(split.times_train), max_points)
    states_true = torch.as_tensor(split.ode_train[:, idx].T, dtype=dtype, device=cfg.device)
    fts_true = torch.as_tensor(split.fts_train_true[idx], dtype=dtype, device=cfg.device)

    model = MLPForceModule(
        mlp_root=cfg.mlp_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=base_seed,
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
        model.update_grid_from_states(states_true)
    init_gnn = model.initialize_gain_from_truth(states_true, fts_true)

    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    loss = _teacher_loss(model, states_true, fts_true)
    loss.backward()

    named = dict(model.named_parameters())
    param_specs = [
        ("mlp.0.weight", [0, 1, 2, 5, 10, 20]),
        ("mlp.0.bias", [0, 1, 2, 5]),
        ("mlp.2.weight", [0, 1, 2, 5]),
        ("mlp.2.bias", [0]),
        ("mlp.4.weight", [0, 1, 2, 5]),
        ("mlp.4.bias", [0]),
    ]
    checks = []
    for name, indices in param_specs:
        param = named.get(name)
        if param is None:
            continue
        flat = param.reshape(-1)
        grad = param.grad.reshape(-1)
        for flat_idx in indices:
            if flat_idx >= flat.numel():
                continue
            orig = float(flat[flat_idx].detach().cpu())
            ad = float(grad[flat_idx].detach().cpu())
            h = max(float(abs_eps), float(rel_eps) * max(1.0, abs(orig)))

            with torch.no_grad():
                flat[flat_idx] = orig + h
            loss_plus = float(_teacher_loss(model, states_true, fts_true).detach().cpu())

            with torch.no_grad():
                flat[flat_idx] = orig - h
            loss_minus = float(_teacher_loss(model, states_true, fts_true).detach().cpu())

            with torch.no_grad():
                flat[flat_idx] = orig

            fd = (loss_plus - loss_minus) / (2.0 * h)
            denom = max(abs(ad), abs(fd), 1.0e-12)
            rel_err = abs(ad - fd) / denom
            checks.append(
                {
                    "name": name,
                    "flat_idx": int(flat_idx),
                    "value": orig,
                    "h": h,
                    "ad": ad,
                    "fd": fd,
                    "abs_diff": abs(ad - fd),
                    "rel_err": rel_err,
                    "sign_match": bool(np.sign(ad) == np.sign(fd) or abs(ad) < 1e-12 or abs(fd) < 1e-12),
                }
            )

    return {
        "timestamp": _now_stamp(),
        "shard_index": int(shard_index),
        "window_role": split.role,
        "window_label": split.label,
        "selected_points": int(len(idx)),
        "seed": int(base_seed),
        "adaptive_grid_enabled": bool(cfg.adaptive_grid_enabled),
        "init_gnn": float(init_gnn),
        "teacher_loss": float(loss.detach().cpu()),
        "rel_eps": float(rel_eps),
        "abs_eps": float(abs_eps),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct MLP teacher-loss coef AD-vs-FD check without odeint.")
    parser.add_argument("--shard-index", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=128)
    parser.add_argument("--rel-eps", type=float, default=1.0e-4)
    parser.add_argument("--abs-eps", type=float, default=1.0e-7)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = run_check(
        shard_index=int(args.shard_index),
        max_points=int(args.max_points),
        rel_eps=float(args.rel_eps),
        abs_eps=float(args.abs_eps),
        seed=args.seed,
    )

    repo_root = Path(__file__).resolve().parents[4]
    log_dir = repo_root / "AFM04" / "MLP_full_test" / "logs" / "debug"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp"]
    out_json = log_dir / f"coef_ad_vs_fd_no_ode_{stamp}.json"
    out_txt = log_dir / f"coef_ad_vs_fd_no_ode_{stamp}.txt"

    lines = [
        "=== MFT COEF AD VS FD (NO ODE) ===",
        f"timestamp={stamp}",
        f"shard={result['shard_index']} role={result['window_role']} label={result['window_label']}",
        f"selected_points={result['selected_points']}",
        f"adaptive_grid={'ON' if result['adaptive_grid_enabled'] else 'OFF'}",
        f"init_gnn={result['init_gnn']:.6e}",
        f"teacher_loss={result['teacher_loss']:.6e}",
        f"rel_eps={result['rel_eps']:.3e} abs_eps={result['abs_eps']:.3e}",
        "",
    ]
    for row in result["checks"]:
        lines.append(
            f"{row['name']}[{row['flat_idx']}]: value={row['value']:.6e} h={row['h']:.6e} "
            f"AD={row['ad']:.6e} FD={row['fd']:.6e} absdiff={row['abs_diff']:.6e} "
            f"relerr={row['rel_err']:.6e} sign_match={row['sign_match']}"
        )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved: {out_txt}")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
