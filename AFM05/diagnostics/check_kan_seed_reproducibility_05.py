"""Check AFM05 KAN seed-level reproducibility before prest2 smoke tests.

This diagnostic verifies the most basic requirement:
same seed + same KAN construction config -> same KAN parameters and outputs.

It intentionally checks the initial function level, not rollout/loss behavior.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM05.stage2light.kan_backend import KANForceModule


def _pykan_root(repo_root: Path) -> Path:
    configured = os.environ.get("HNODECB_PYKAN_ROOT", "").strip()
    candidates = ([Path(configured)] if configured else []) + [
        repo_root.parent / "pykan",
        repo_root / "pykan",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if importlib.util.find_spec("kan") is not None:
        return repo_root
    raise FileNotFoundError(
        "Could not locate pykan. Install requirements-afm.txt or set HNODECB_PYKAN_ROOT."
    )


def _build_model(seed: int, noise_scale: float, *, pykan_root: Path) -> KANForceModule:
    return KANForceModule(
        pykan_root=pykan_root,
        state_mean=np.asarray([1.0e-8, 2.0e-3, 3.0e-9], dtype=float),
        state_scale=np.asarray([5.0e-8, 1.0e-2, 1.0e-8], dtype=float),
        seed=int(seed),
        width=(3, 7, 1),
        grid=11,
        spline_k=3,
        base_fun="silu",
        symbolic_enabled=False,
        auto_save=False,
        noise_scale=float(noise_scale),
        affine_trainable=False,
        grid_eps=0.02,
        grid_range=(-1.0, 1.0),
        initial_grid_support=np.asarray([[-1.0, 1.0], [-1.0, 1.0], [-2.0, 1.0]], dtype=float),
        gnn_learnable=False,
        dist=0.0,
        a0=1.0e-9,
        soft_mask_enabled=True,
        soft_mask_trainable=False,
        soft_mask_s0_a0=20.0,
        soft_mask_s0_min_a0=1.0,
        soft_mask_s0_max_a0=100.0,
        soft_mask_alpha_a0=0.25,
        soft_mask_alpha_min_a0=0.02,
        soft_mask_alpha_max_a0=5.0,
        device="cpu",
        dtype=torch.float64,
    )


def _state_dict_max_abs_diff(a: KANForceModule, b: KANForceModule) -> tuple[bool, float, list[str]]:
    sda = a.state_dict()
    sdb = b.state_dict()
    if set(sda) != set(sdb):
        missing = sorted(set(sda).symmetric_difference(set(sdb)))
        return False, float("inf"), missing

    max_abs = 0.0
    different: list[str] = []
    for key in sorted(sda):
        va = sda[key].detach().cpu()
        vb = sdb[key].detach().cpu()
        if va.shape != vb.shape:
            different.append(f"{key}: shape {tuple(va.shape)} != {tuple(vb.shape)}")
            continue
        if torch.is_floating_point(va):
            diff = float(torch.max(torch.abs(va - vb)).item()) if va.numel() else 0.0
            max_abs = max(max_abs, diff)
            if diff != 0.0:
                different.append(f"{key}: max_abs_diff={diff:.16e}")
        elif not torch.equal(va, vb):
            different.append(f"{key}: tensor mismatch")
    return len(different) == 0, max_abs, different


def _output_max_abs_diff(a: KANForceModule, b: KANForceModule) -> dict[str, float]:
    probe = torch.tensor(
        [
            [1.1e-8, 2.1e-3, 1.0e-9],
            [2.0e-8, 1.0e-3, -1.0e-8],
            [0.0, -1.0e-3, 2.0e-8],
        ],
        dtype=torch.float64,
    )
    checks = {
        "raw": lambda m: m.raw_output(probe),
        "masked": lambda m: m.masked_raw_output(probe),
        "final": lambda m: m(probe),
    }
    out: dict[str, float] = {}
    for name, fn in checks.items():
        ya = fn(a).detach().cpu()
        yb = fn(b).detach().cpu()
        out[name] = float(torch.max(torch.abs(ya - yb)).item())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    repo = REPO_ROOT
    pykan = _pykan_root(repo)
    out_path = args.out or (repo / "AFM05" / "diagnostics" / "kan_seed_reproducibility_05.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_a = _build_model(args.seed, args.noise_scale, pykan_root=pykan)
    model_b = _build_model(args.seed, args.noise_scale, pykan_root=pykan)
    same, max_abs, different = _state_dict_max_abs_diff(model_a, model_b)
    out_diffs = _output_max_abs_diff(model_a, model_b)

    lines = [
        "AFM05 KAN seed reproducibility diagnostic",
        "=========================================",
        f"pykan_root = {pykan}",
        f"seed = {args.seed}",
        f"noise_scale = {args.noise_scale}",
        "config = width(3,7,1), grid=11, spline_k=3, base_fun=silu, dtype=float64, device=cpu",
        "support = [[-1,1], [-1,1], [-2,1]]",
        "",
        f"state_dict_identical = {same}",
        f"state_dict_max_abs_diff = {max_abs:.16e}",
        f"raw_output_max_abs_diff = {out_diffs['raw']:.16e}",
        f"masked_output_max_abs_diff = {out_diffs['masked']:.16e}",
        f"final_output_max_abs_diff = {out_diffs['final']:.16e}",
    ]
    if different:
        lines.extend(["", "Differences:"])
        lines.extend(f"  - {item}" for item in different[:100])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if same and all(value == 0.0 for value in out_diffs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
