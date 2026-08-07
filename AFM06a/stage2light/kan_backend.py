"""Exact AFM06a st1pl KAN restoration for stage2light."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from AFM06a.stage1pluslight.config import default_config as stage1_default_config
from AFM06a.stage1pluslight.kan_backend import KANForceModule, configure_deterministic_torch, state_dict_digest
from AFM06a.stage2light.config import Stage2LightConfig
from AFM06a.stage2light.data import Stage1Endpoint


def restore_exact_stage1_model(config: Stage2LightConfig, endpoint: Stage1Endpoint) -> KANForceModule:
    """Restore the saved st1pl function without refitting any entry component."""

    warmstart = endpoint.warmstart
    endpoint_window = getattr(endpoint, "window", None)
    endpoint_settings = getattr(endpoint_window, "settings", None)
    construction = warmstart["kan_construction"]
    seed = int(construction["seed"])
    configure_deterministic_torch(seed)
    fallback_range = construction["fallback_grid_range"]
    input_policy = str(construction.get("input_policy", "global_full_span_mean_std"))
    output_policy = str(construction.get("force_output_policy", "identity_raw_physical"))
    stage1_cfg = replace(
        stage1_default_config(config.repo_root),
        pykan_root=config.pykan_root,
        kan_width=tuple(int(v) for v in construction["width"]),
        kan_grid=int(construction["grid"]),
        kan_spline_k=int(construction["spline_k"]),
        kan_base_fun=str(construction["base_fun"]),
        kan_noise_scale=float(construction["noise_scale"]),
        kan_grid_eps=float(construction["grid_eps"]),
        kan_grid_range_lo=float(fallback_range[0]),
        kan_grid_range_hi=float(fallback_range[1]),
        normalizer_policy=input_policy,
        force_output_policy=output_policy,
        device=config.device,
        dtype=config.dtype,
        gain_enabled=config.gain_enabled,
        gain_learnable=config.gain_learnable,
        soft_mask_enabled=config.soft_mask_enabled,
        soft_mask_trainable=config.soft_mask_trainable,
    )
    model = KANForceModule(
        config=stage1_cfg,
        state_mean=np.asarray(warmstart["state_mean"], dtype=float),
        state_scale=np.asarray(warmstart["state_scale"], dtype=float),
        seed=seed,
        initial_support=np.asarray(warmstart["initial_grid_support"], dtype=float),
        force_mean=float(warmstart.get("force_mean", 0.0)),
        force_scale=float(warmstart.get("force_scale", 1.0)),
        force_output_policy=output_policy,
        settings=endpoint_settings,
    ).to(config.device)
    saved_state = warmstart["kan_state_dict"]
    if not isinstance(saved_state, dict):
        raise RuntimeError("saved AFM06a KAN state_dict is invalid")
    state: dict[str, torch.Tensor] = {
        str(name): torch.as_tensor(value, device=config.device).clone()
        for name, value in saved_state.items()
    }
    model.load_state_dict(state, strict=True)
    restored_digest = state_dict_digest(model.frozen_state_dict())
    expected_digest = str(warmstart["kan_state_dict_sha256"])
    if restored_digest != expected_digest:
        raise RuntimeError(
            "restored AFM06a KAN state does not match st1pl: "
            f"expected={expected_digest}, restored={restored_digest}"
        )
    model.train()
    return model


def trainable_parameter_manifest(model: KANForceModule) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            manifest.append({"name": name, "shape": list(parameter.shape), "numel": parameter.numel()})
    return manifest


__all__ = ["restore_exact_stage1_model", "trainable_parameter_manifest"]
