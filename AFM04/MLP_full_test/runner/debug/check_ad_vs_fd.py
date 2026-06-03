"""Small-scope AD-vs-FD sanity check for MFT without interrupting GS."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from AFM04.MLP_full_test.config import default_config
from AFM04.MLP_full_test.data import prepare_data
from AFM04.MLP_full_test.mlp_backend import MLPForceModule
from AFM04.MLP_full_test.losses import evaluate_split


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


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_model(cfg, prepared, *, seed: int, dtype: torch.dtype):
    model = MLPForceModule(
        mlp_root=cfg.mlp_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=seed,
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
    return model


def _compute_loss(model, *, prepared, mech_true_t, batch, cfg) -> torch.Tensor:
    total, _, _ = evaluate_split(
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
    return total


def _selected_param_specs(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter, int]]:
    named = dict(model.named_parameters())
    specs: list[tuple[str, torch.nn.Parameter, int]] = []
    wanted = [
        ("log_gnn[0]", "log_gnn", 0),
        ("mlp.0.weight[0]", "mlp.0.weight", 0),
        ("mlp.0.bias[0]", "mlp.0.bias", 0),
        ("mlp.2.weight[0]", "mlp.2.weight", 0),
        ("mlp.2.bias[0]", "mlp.2.bias", 0),
        ("mlp.4.weight[0]", "mlp.4.weight", 0),
        ("mlp.4.bias[0]", "mlp.4.bias", 0),
    ]
    for label, name, flat_idx in wanted:
        param = named.get(name)
        if param is None:
            continue
        if param.numel() <= flat_idx:
            continue
        specs.append((label, param, flat_idx))
    return specs


def run_check(*, shard_index: int, max_points: int, rel_eps: float, abs_eps: float, seed: int | None = None) -> dict[str, Any]:
    cfg = default_config()
    dtype = _torch_dtype(cfg.dtype)
    base_seed = int(cfg.seed if seed is None else seed)
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    prepared = prepare_data(cfg)
    split = prepared.splits[shard_index - 1]
    idx = _subset_indices(len(split.times_train), max_points)
    batch = _to_torch_split(split, idx, dtype=dtype, device=cfg.device)
    mech_true_t = torch.as_tensor(prepared.mech_true, dtype=dtype, device=cfg.device)

    model = _make_model(cfg, prepared, seed=base_seed, dtype=dtype)
    if cfg.adaptive_grid_enabled:
        model.update_grid_from_states(batch["states_for_init"])
    init_gnn = model.initialize_gain_from_truth(batch["states_for_init"], batch["fts_true"])

    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    loss = _compute_loss(model, prepared=prepared, mech_true_t=mech_true_t, batch=batch, cfg=cfg)
    loss.backward()

    checks = []
    for label, param, flat_idx in _selected_param_specs(model):
        flat_param = param.reshape(-1)
        orig = float(flat_param[flat_idx].detach().cpu())
        ad_grad = float(param.grad.reshape(-1)[flat_idx].detach().cpu()) if param.grad is not None else float("nan")
        h = max(float(abs_eps), float(rel_eps) * max(1.0, abs(orig)))

        with torch.no_grad():
            flat_param[flat_idx] = orig + h
        loss_plus = float(_compute_loss(model, prepared=prepared, mech_true_t=mech_true_t, batch=batch, cfg=cfg).detach().cpu())

        with torch.no_grad():
            flat_param[flat_idx] = orig - h
        loss_minus = float(_compute_loss(model, prepared=prepared, mech_true_t=mech_true_t, batch=batch, cfg=cfg).detach().cpu())

        with torch.no_grad():
            flat_param[flat_idx] = orig

        fd_grad = (loss_plus - loss_minus) / (2.0 * h)
        denom = max(abs(ad_grad), abs(fd_grad), 1.0e-12)
        rel_err = abs(ad_grad - fd_grad) / denom
        checks.append(
            {
                "label": label,
                "value": orig,
                "h": h,
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "ad_grad": ad_grad,
                "fd_grad": fd_grad,
                "abs_diff": abs(ad_grad - fd_grad),
                "rel_err": rel_err,
                "sign_match": bool(np.sign(ad_grad) == np.sign(fd_grad) or abs(ad_grad) < 1e-12 or abs(fd_grad) < 1e-12),
            }
        )

    return {
        "timestamp": _now_stamp(),
        "shard_index": shard_index,
        "window_role": split.role,
        "window_label": split.label,
        "selected_points": int(len(idx)),
        "seed": base_seed,
        "ode_method": cfg.ode_method,
        "adaptive_grid_enabled": bool(cfg.adaptive_grid_enabled),
        "init_gnn": float(init_gnn),
        "base_loss": float(loss.detach().cpu()),
        "rel_eps": float(rel_eps),
        "abs_eps": float(abs_eps),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-scope AD-vs-FD check for AFM04 MFT.")
    parser.add_argument("--shard-index", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=96)
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
    out_json = log_dir / f"ad_vs_fd_check_{stamp}.json"
    out_txt = log_dir / f"ad_vs_fd_check_{stamp}.txt"

    lines = [
        "=== MFT AD VS FD CHECK ===",
        f"timestamp={stamp}",
        f"shard={result['shard_index']} role={result['window_role']} label={result['window_label']}",
        f"selected_points={result['selected_points']}",
        f"ode_method={result['ode_method']} adaptive_grid={'ON' if result['adaptive_grid_enabled'] else 'OFF'}",
        f"init_gnn={result['init_gnn']:.6e}",
        f"base_loss={result['base_loss']:.6e}",
        f"rel_eps={result['rel_eps']:.3e} abs_eps={result['abs_eps']:.3e}",
        "",
    ]
    for row in result["checks"]:
        lines.append(
            f"{row['label']}: value={row['value']:.6e} h={row['h']:.6e} "
            f"AD={row['ad_grad']:.6e} FD={row['fd_grad']:.6e} "
            f"absdiff={row['abs_diff']:.6e} relerr={row['rel_err']:.6e} "
            f"sign_match={row['sign_match']}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved: {out_txt}")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
