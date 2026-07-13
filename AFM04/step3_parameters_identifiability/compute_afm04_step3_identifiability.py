"""AFM04 step3 identifiability computation from a completed stage2light result.

This script implements the AFM04 counterpart of the HNODECB step3 idea:
build a sensitivity matrix around one completed step2/st2l local point, form
the Gram/Hessian approximation S^T S, and eigendecompose it.

The formal AFM04 default parameter set is "trained": active trainable KAN
numeric parameters + gain + soft-mask parameters + raw_ks/raw_cs, excluding u0
and excluding inactive symbolic parameters.  A fast "mech" diagnostic is also
available for theta = [log(ks), log(cs)].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM04.datasets.non_perturbed_dataset_generator import load_table_npz  # noqa: E402
from AFM04.stage1pluslight.data import truncate_to_first_contact  # noqa: E402
from AFM04.stage2light.kan_backend import KANForceModule  # noqa: E402
from AFM04.stage2light.rollout import (  # noqa: E402
    LearnableMechModule,
    rollout_single_shooting_torch,
    x2dot_rhs_torch,
)


CURRENT_ENTRY_SCHEMA = "afm04_step3_current_entry_v1"
RESULT_SCHEMA = "afm04_step3_identifiability_v2"
SHARD_SCHEMA = "afm04_step3_identifiability_shard_v1"
MECH_PARAMETER_LABELS = ("log_ks", "log_cs")
DEFAULT_COMPONENTS = ("x1", "x2", "x2dot")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_progress(message: str) -> None:
    print(f"[{now_text()}] {message}", flush=True)


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def torch_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name!r}")


def mech_parameterization_from_config(cfg: dict[str, Any]) -> str:
    # Older st2l payloads predate this field and used raw->sigmoid->bounded.
    raw = str(cfg.get("mech_parameterization", "sigmoid_bounded")).strip().lower().replace("-", "_")
    aliases = {
        "direct": "direct_unbounded",
        "physical": "direct_unbounded",
        "unbounded": "direct_unbounded",
        "direct_unbounded": "direct_unbounded",
        "exp": "log_relative",
        "log": "log_relative",
        "relative": "log_relative",
        "log_relative": "log_relative",
        "log_relative_bounded": "log_relative",
        "sigmoid": "sigmoid_bounded",
        "bounded": "sigmoid_bounded",
        "sigmoid_bound": "sigmoid_bounded",
        "sigmoid_bounds": "sigmoid_bounded",
        "sigmoid_bounded": "sigmoid_bounded",
        "legacy": "sigmoid_bounded",
        "legacy_sigmoid": "sigmoid_bounded",
    }
    return aliases.get(raw, "sigmoid_bounded")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return repo_rel(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 1.0
    val = float(np.sqrt(np.mean(np.square(arr))))
    if not np.isfinite(val) or val <= 1.0e-30:
        return 1.0
    return val


def load_current_entry(path: Path) -> dict[str, Any]:
    entry = json.loads(path.read_text(encoding="utf-8"))
    if entry.get("schema_version") != CURRENT_ENTRY_SCHEMA:
        raise ValueError(f"Unexpected current entry schema in {path}: {entry.get('schema_version')!r}")
    return entry


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)!r}")
    return payload


def build_model(payload: dict[str, Any]) -> tuple[KANForceModule, tuple[float, ...], torch.dtype]:
    cfg = payload["config"]
    dtype = torch_dtype(cfg["dtype"])
    device = "cpu"
    known_pars = tuple(float(v) for v in payload["known_pars"])
    model = KANForceModule(
        pykan_root=Path(cfg["pykan_root"]),
        state_mean=np.asarray(payload["state_mean"], dtype=float),
        state_scale=np.asarray(payload["state_scale"], dtype=float),
        seed=int(cfg["seed"]),
        width=tuple(int(v) for v in cfg["width"]),
        grid=int(cfg["grid"]),
        spline_k=int(cfg["spline_k"]),
        base_fun=str(cfg["base_fun"]),
        symbolic_enabled=bool(cfg["symbolic_enabled"]),
        auto_save=bool(cfg["auto_save"]),
        noise_scale=float(cfg["noise_scale"]),
        affine_trainable=bool(cfg["affine_trainable"]),
        grid_eps=float(cfg["grid_eps"]),
        grid_range=(float(cfg["grid_range_lo"]), float(cfg["grid_range_hi"])),
        initial_grid_support=payload.get("initial_grid_support"),
        gnn_learnable=bool(cfg["gnn_learnable"]),
        dist=float(known_pars[6]),
        a0=float(known_pars[9]),
        soft_mask_enabled=bool(cfg.get("soft_mask_enabled", True)),
        soft_mask_trainable=bool(cfg.get("soft_mask_trainable", True)),
        soft_mask_s0_a0=float(cfg.get("soft_mask_s0_a0", 20.0)),
        soft_mask_s0_min_a0=float(cfg.get("soft_mask_s0_min_a0", 1.0)),
        soft_mask_s0_max_a0=float(cfg.get("soft_mask_s0_max_a0", 100.0)),
        soft_mask_alpha_a0=float(cfg.get("soft_mask_alpha_a0", 0.25)),
        soft_mask_alpha_min_a0=float(cfg.get("soft_mask_alpha_min_a0", 0.02)),
        soft_mask_alpha_max_a0=float(cfg.get("soft_mask_alpha_max_a0", 5.0)),
        device=device,
        dtype=dtype,
    ).to(device)
    state_dict = payload.get("final_state_dict")
    if not isinstance(state_dict, dict):
        state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Payload is missing final_state_dict/state_dict")
    model.load_state_dict(state_dict)
    model.eval()
    return model, known_pars, dtype


def build_mech_from_state(payload: dict[str, Any], dtype: torch.dtype) -> LearnableMechModule:
    cfg = payload["config"]
    parameterization = mech_parameterization_from_config(cfg)
    mech = LearnableMechModule(
        ks_init=float(payload["mech_true"][0]),
        cs_init=float(payload["mech_true"][1]),
        ks_bounds=(float(cfg["ks_lo"]), float(cfg["ks_hi"])),
        cs_bounds=(float(cfg["cs_lo"]), float(cfg["cs_hi"])),
        dtype=dtype,
        device="cpu",
        parameterization=parameterization,
    )
    mech_state = payload.get("final_mech_state_dict")
    if not isinstance(mech_state, dict):
        mech_state = payload.get("mech_state_dict")
    if not isinstance(mech_state, dict):
        raise RuntimeError("Payload is missing final_mech_state_dict/mech_state_dict")
    mech.load_state_dict(mech_state, strict=False)
    mech.eval()
    return mech


def build_mech_from_physical(payload: dict[str, Any], dtype: torch.dtype, *, ks: float, cs: float) -> LearnableMechModule:
    cfg = payload["config"]
    parameterization = mech_parameterization_from_config(cfg)
    return LearnableMechModule(
        ks_init=float(ks),
        cs_init=float(cs),
        ks_bounds=(float(cfg["ks_lo"]), float(cfg["ks_hi"])),
        cs_bounds=(float(cfg["cs_lo"]), float(cfg["cs_hi"])),
        dtype=dtype,
        device="cpu",
        parameterization=parameterization,
    ).eval()


def mech_physical(mech: LearnableMechModule) -> tuple[float, float]:
    with torch.no_grad():
        return float(mech.ks().detach().cpu()), float(mech.cs().detach().cpu())


def trained_parameter_refs(
    payload: dict[str, Any],
    model: KANForceModule,
    mech: LearnableMechModule,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return AFM04 formal step3 trainable parameters, excluding u0.

    Inactive symbolic parameters are excluded when symbolic_enabled=False,
    even if pykan exposes them as requires_grad parameters.
    """

    symbolic_enabled = bool(payload["config"].get("symbolic_enabled", False))
    refs: list[tuple[str, torch.nn.Parameter]] = []
    for prefix, module in (("model", model), ("mech", mech)):
        for name, param in module.named_parameters():
            full_name = f"{prefix}.{name}"
            if not bool(param.requires_grad):
                continue
            if ".symbolic_fun." in full_name and not symbolic_enabled:
                continue
            refs.append((full_name, param))
    if not refs:
        raise RuntimeError("No trainable parameters found for AFM04 step3")
    return refs


def parameter_element_labels(refs: list[tuple[str, torch.nn.Parameter]]) -> list[tuple[str, int]]:
    labels: list[tuple[str, int]] = []
    for name, param in refs:
        for flat_idx in range(int(param.numel())):
            labels.append((name, flat_idx))
    return labels


def restore_parameter_refs(
    refs: list[tuple[str, torch.nn.Parameter]],
    base_values: list[torch.Tensor],
) -> None:
    with torch.no_grad():
        for (_name, param), base in zip(refs, base_values, strict=True):
            param.copy_(base.to(device=param.device, dtype=param.dtype))


def set_parameter_flat_value(param: torch.nn.Parameter, flat_idx: int, value: torch.Tensor) -> None:
    with torch.no_grad():
        flat = param.reshape(-1)
        flat[int(flat_idx)].copy_(value.to(device=param.device, dtype=param.dtype))


def parameter_value(base_values: list[torch.Tensor], param_ref_index: int, flat_idx: int) -> float:
    return float(base_values[int(param_ref_index)].reshape(-1)[int(flat_idx)].detach().cpu())


def summarize_label(label: str, flat_idx: int) -> str:
    return f"{label}[{flat_idx}]"


def make_eigen_rows(
    *,
    parameter_labels: list[str],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    max_eval = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    rows: list[dict[str, Any]] = []
    for idx, value in enumerate(eigenvalues):
        vec = eigenvectors[:, idx]
        abs_order = np.argsort(-np.abs(vec))
        dominant_idx = int(abs_order[0]) if abs_order.size else -1
        row: dict[str, Any] = {
            "eigen_index_ascending": idx + 1,
            "eigenvalue": float(value),
            "log10_eigenvalue": float(np.log10(max(abs(float(value)), 1.0e-300))),
            "relative_to_max_abs_eigenvalue": float(abs(float(value)) / max_eval) if max_eval > 0.0 else float("nan"),
            "dominant_parameter": parameter_labels[dominant_idx] if dominant_idx >= 0 else "",
        }
        top_terms: list[dict[str, Any]] = []
        for rank, param_idx in enumerate(abs_order[: max(1, int(top_k))], start=1):
            label = parameter_labels[int(param_idx)]
            value_i = float(vec[int(param_idx)])
            row[f"top{rank}_parameter"] = label
            row[f"top{rank}_component"] = value_i
            top_terms.append({"rank": rank, "parameter": label, "component": value_i})
        row["top_components"] = top_terms
        rows.append(row)
    return rows


def shard_bounds(total: int, shard_count: int, shard_index: int) -> tuple[int, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not (1 <= shard_index <= shard_count):
        raise ValueError(f"shard_index must be in [1, {shard_count}], got {shard_index}")
    start = (shard_index - 1) * total // shard_count
    end = shard_index * total // shard_count
    return start, end


def parameter_group_counts(parameter_labels: list[str]) -> dict[str, int]:
    groups: dict[str, int] = {"kan_numeric": 0, "gain": 0, "soft_mask": 0, "mech": 0}
    for label in parameter_labels:
        if label.startswith("mech."):
            groups["mech"] += 1
        elif "soft_mask" in label:
            groups["soft_mask"] += 1
        elif label.startswith("model.log_gnn"):
            groups["gain"] += 1
        else:
            groups["kan_numeric"] += 1
    return groups


def load_truncated_data(entry: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ode_path = resolve_repo_path(entry["data_files"]["ode_data"])
    pert_path = resolve_repo_path(entry["data_files"]["pert_df"])
    with np.load(ode_path) as data:
        ode_data = np.asarray(data["data"], dtype=float)
    pert_df = load_table_npz(str(pert_path))
    return truncate_to_first_contact(ode_data, pert_df)


def window_arrays(
    payload: dict[str, Any],
    ode_data: np.ndarray,
    pert_df: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    meta = payload["window_meta"]
    start = int(meta["start_idx"])
    stop = int(meta["stop_idx"])
    if start < 0 or stop < start or stop >= ode_data.shape[1]:
        raise ValueError(f"Invalid window_meta indices after truncation: start={start}, stop={stop}, n={ode_data.shape[1]}")
    idx = np.arange(start, stop + 1, dtype=int)
    times = np.asarray(pert_df["t"], dtype=float)[idx]
    ode_window = np.asarray(ode_data[:, idx], dtype=float)
    extra = {k: np.asarray(v)[idx] for k, v in pert_df.items() if k != "__columns__" and np.asarray(v).ndim == 1}
    return times, ode_window, extra


def model_outputs(
    *,
    model: KANForceModule,
    known_pars: tuple[float, ...],
    mech: LearnableMechModule,
    u0: torch.Tensor,
    times: torch.Tensor,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        traj = rollout_single_shooting_torch(
            model,
            known_pars,
            mech,
            u0,
            times,
            method=str(cfg["ode_method"]),
            rtol=float(cfg["ode_rtol"]),
            atol=float(cfg["ode_atol"]),
        )
        states = traj.transpose(0, 1)
        x2dot = x2dot_rhs_torch(traj, times, model, known_pars)
        force = model(states)
    return {
        "x1": traj[0, :].detach().cpu().numpy().reshape(-1),
        "x2": traj[1, :].detach().cpu().numpy().reshape(-1),
        "x3": traj[2, :].detach().cpu().numpy().reshape(-1),
        "x2dot": x2dot.detach().cpu().numpy().reshape(-1),
        "force": force.detach().cpu().numpy().reshape(-1),
    }


def truth_component_scales(ode_window: np.ndarray, extra: dict[str, np.ndarray]) -> dict[str, float]:
    scales = {
        "x1": rms(ode_window[0, :]),
        "x2": rms(ode_window[1, :]),
        "x3": rms(ode_window[2, :]),
    }
    if "x2dot" in extra:
        scales["x2dot"] = rms(extra["x2dot"])
    if "fts" in extra:
        scales["force"] = rms(extra["fts"])
    return scales


def flatten_components(outputs: dict[str, np.ndarray], components: tuple[str, ...], scales: dict[str, float]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for name in components:
        if name not in outputs:
            raise KeyError(f"Output component {name!r} is unavailable")
        scale = float(scales.get(name, rms(outputs[name])))
        if not np.isfinite(scale) or scale <= 1.0e-30:
            scale = 1.0
        chunks.append(np.asarray(outputs[name], dtype=float).reshape(-1) / scale)
    return np.concatenate(chunks, axis=0)


def parse_components(text: str) -> tuple[str, ...]:
    components = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    if not components:
        raise ValueError("At least one observable component is required")
    valid = {"x1", "x2", "x3", "x2dot", "force"}
    unknown = [name for name in components if name not in valid]
    if unknown:
        raise ValueError(f"Unsupported observable component(s): {unknown}; valid={sorted(valid)}")
    return components


def check_bounds(payload: dict[str, Any], *, ks: float, cs: float) -> None:
    cfg = payload["config"]
    ks_lo, ks_hi = float(cfg["ks_lo"]), float(cfg["ks_hi"])
    cs_lo, cs_hi = float(cfg["cs_lo"]), float(cfg["cs_hi"])
    if not (ks_lo < ks < ks_hi):
        raise ValueError(f"Perturbed ks={ks:.9e} is outside open bounds ({ks_lo:.9e}, {ks_hi:.9e})")
    if not (cs_lo < cs < cs_hi):
        raise ValueError(f"Perturbed cs={cs:.9e} is outside open bounds ({cs_lo:.9e}, {cs_hi:.9e})")


def compute_mech_identifiability(
    *,
    entry: dict[str, Any],
    payload: dict[str, Any],
    components: tuple[str, ...],
    finite_diff_eps: float,
) -> dict[str, Any]:
    model, known_pars, dtype = build_model(payload)
    mech0 = build_mech_from_state(payload, dtype)
    ks0, cs0 = mech_physical(mech0)
    ode_data, pert_df = load_truncated_data(entry)
    times_np, ode_window, extra = window_arrays(payload, ode_data, pert_df)
    times = torch.as_tensor(times_np, dtype=dtype, device="cpu")
    u0 = torch.as_tensor(ode_window[:, 0], dtype=dtype, device="cpu")
    cfg = payload["config"]

    parameter_labels = list(MECH_PARAMETER_LABELS)
    log_progress(
        "step3 sensitivity start | "
        f"parameter_set=mech | parameters={len(parameter_labels)} | "
        f"components={','.join(components)}"
    )

    log_progress("baseline rollout start")
    baseline_outputs = model_outputs(
        model=model,
        known_pars=known_pars,
        mech=mech0,
        u0=u0,
        times=times,
        cfg=cfg,
    )
    log_progress("baseline rollout done")
    scales = truth_component_scales(ode_window, extra)
    for name in components:
        scales.setdefault(name, rms(baseline_outputs[name]))
    baseline_vector = flatten_components(baseline_outputs, components, scales)

    sensitivity_columns: list[np.ndarray] = []
    perturbation_meta: list[dict[str, float | str]] = []
    for param_counter, label in enumerate(parameter_labels, start=1):
        log_progress(f"param {param_counter}/{len(parameter_labels)} | parameter={label}")
        if label == "log_ks":
            ks_plus, ks_minus = ks0 * math.exp(finite_diff_eps), ks0 * math.exp(-finite_diff_eps)
            cs_plus = cs_minus = cs0
        elif label == "log_cs":
            ks_plus = ks_minus = ks0
            cs_plus, cs_minus = cs0 * math.exp(finite_diff_eps), cs0 * math.exp(-finite_diff_eps)
        else:
            raise AssertionError(label)

        check_bounds(payload, ks=ks_plus, cs=cs_plus)
        check_bounds(payload, ks=ks_minus, cs=cs_minus)
        plus = build_mech_from_physical(payload, dtype, ks=ks_plus, cs=cs_plus)
        minus = build_mech_from_physical(payload, dtype, ks=ks_minus, cs=cs_minus)
        plus_outputs = model_outputs(model=model, known_pars=known_pars, mech=plus, u0=u0, times=times, cfg=cfg)
        minus_outputs = model_outputs(model=model, known_pars=known_pars, mech=minus, u0=u0, times=times, cfg=cfg)
        plus_vector = flatten_components(plus_outputs, components, scales)
        minus_vector = flatten_components(minus_outputs, components, scales)
        column = (plus_vector - minus_vector) / (2.0 * finite_diff_eps)
        sensitivity_columns.append(column)
        perturbation_meta.append(
            {
                "parameter": label,
                "ks_plus": ks_plus,
                "ks_minus": ks_minus,
                "cs_plus": cs_plus,
                "cs_minus": cs_minus,
            }
        )

    log_progress("sensitivity matrix assembly start")
    if sensitivity_columns:
        sensitivity = np.column_stack(sensitivity_columns)
    else:
        sensitivity = np.empty((baseline_vector.size, 0), dtype=float)
    log_progress(f"sensitivity matrix assembly done | shape={sensitivity.shape[0]}x{sensitivity.shape[1]}")
    log_progress("hessian build start")
    hessian = (sensitivity.T @ sensitivity) / float(max(sensitivity.shape[0], 1))
    hessian = 0.5 * (hessian + hessian.T)
    log_progress(f"hessian build done | shape={hessian.shape[0]}x{hessian.shape[1]}")
    log_progress("eigendecomposition start")
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    log_progress("eigendecomposition done")

    log_progress("eigen row summary start")
    column_norms = np.sqrt(np.sum(np.square(sensitivity), axis=0) / float(max(sensitivity.shape[0], 1)))
    rows = make_eigen_rows(parameter_labels=parameter_labels, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    log_progress("eigen row summary done")

    return {
        "schema_version": RESULT_SCHEMA,
        "rank_label": entry.get("rank_label"),
        "rank": entry.get("rank"),
        "candidate_b": entry.get("candidate_b"),
        "entry_payload": entry.get("entry_payload"),
        "parameter_set": "mech",
        "parameter_coordinates": parameter_labels,
        "parameter_count": len(parameter_labels),
        "mechanistic_point": {"ks": ks0, "cs": cs0},
        "finite_difference": {
            "coordinate": "central difference in log physical parameters",
            "eps": finite_diff_eps,
            "perturbations": perturbation_meta,
        },
        "observable_components": list(components),
        "component_scales": {k: float(v) for k, v in scales.items() if k in components},
        "window_meta": jsonable(payload.get("window_meta", {})),
        "baseline_vector_length": int(baseline_vector.size),
        "sensitivity_column_rms": {
            parameter_labels[i]: float(column_norms[i]) for i in range(len(parameter_labels))
        },
        "hessian": hessian,
        "eigenvalues": eigenvalues,
        "eigenvectors_columns": eigenvectors,
        "eigen_rows": rows,
        "baseline_outputs": {
            name: {
                "min": float(np.min(baseline_outputs[name])),
                "max": float(np.max(baseline_outputs[name])),
                "rms": rms(baseline_outputs[name]),
            }
            for name in components
        },
    }


def compute_trained_identifiability(
    *,
    entry: dict[str, Any],
    payload: dict[str, Any],
    components: tuple[str, ...],
    finite_diff_eps: float,
    parameter_limit: int = 0,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    model, known_pars, dtype = build_model(payload)
    mech0 = build_mech_from_state(payload, dtype)
    ks0, cs0 = mech_physical(mech0)
    ode_data, pert_df = load_truncated_data(entry)
    times_np, ode_window, extra = window_arrays(payload, ode_data, pert_df)
    times = torch.as_tensor(times_np, dtype=dtype, device="cpu")
    u0 = torch.as_tensor(ode_window[:, 0], dtype=dtype, device="cpu")
    cfg = payload["config"]

    refs = trained_parameter_refs(payload, model, mech0)
    base_values = [param.detach().clone() for _name, param in refs]
    all_element_refs = parameter_element_labels(refs)
    if parameter_limit > 0:
        all_element_refs = all_element_refs[: int(parameter_limit)]
    full_parameter_count = len(all_element_refs)
    all_parameter_labels = [summarize_label(name, flat_idx) for name, flat_idx in all_element_refs]
    if shard_count > 1:
        start, end = shard_bounds(full_parameter_count, shard_count, shard_index)
        selected_global_indices = list(range(start, end))
        element_refs = all_element_refs[start:end]
        parameter_labels = all_parameter_labels[start:end]
        shard_text = f" | shard={shard_index}/{shard_count} | columns={start + 1}-{end} of {full_parameter_count}"
    else:
        selected_global_indices = list(range(full_parameter_count))
        element_refs = all_element_refs
        parameter_labels = all_parameter_labels
        shard_text = ""

    log_progress(
        "step3 sensitivity start | "
        f"parameter_set=trained | parameters={len(parameter_labels)} | "
        f"components={','.join(components)} | parameter_limit={int(parameter_limit)}"
        f"{shard_text}"
    )

    log_progress("baseline rollout start")
    baseline_outputs = model_outputs(
        model=model,
        known_pars=known_pars,
        mech=mech0,
        u0=u0,
        times=times,
        cfg=cfg,
    )
    log_progress("baseline rollout done")
    scales = truth_component_scales(ode_window, extra)
    for name in components:
        scales.setdefault(name, rms(baseline_outputs[name]))
    baseline_vector = flatten_components(baseline_outputs, components, scales)

    sensitivity_columns: list[np.ndarray] = []
    perturbation_meta: list[dict[str, Any]] = []
    ref_name_to_index = {name: idx for idx, (name, _param) in enumerate(refs)}
    for local_counter, (global_idx, (param_name, flat_idx)) in enumerate(zip(selected_global_indices, element_refs), start=1):
        param_label = summarize_label(param_name, flat_idx)
        log_progress(
            f"param {global_idx + 1}/{full_parameter_count} | "
            f"shard_param {local_counter}/{len(parameter_labels)} | parameter={param_label}"
        )
        ref_idx = ref_name_to_index[param_name]
        param = refs[ref_idx][1]
        base_value = parameter_value(base_values, ref_idx, flat_idx)

        if abs(base_value) <= 1.0e-30:
            column = np.zeros_like(baseline_vector)
            perturbation_meta.append(
                {
                    "parameter": param_label,
                    "base_value": base_value,
                    "relative_step_skipped": True,
                    "reason": "base value is zero under original relative-parameter convention",
                    "parameter_number": global_idx + 1,
                    "global_index": global_idx,
                }
            )
            sensitivity_columns.append(column)
            continue

        plus_value = torch.as_tensor(base_value * (1.0 + finite_diff_eps), dtype=param.dtype, device=param.device)
        minus_value = torch.as_tensor(base_value * (1.0 - finite_diff_eps), dtype=param.dtype, device=param.device)

        restore_parameter_refs(refs, base_values)
        set_parameter_flat_value(param, flat_idx, plus_value)
        plus_outputs = model_outputs(model=model, known_pars=known_pars, mech=mech0, u0=u0, times=times, cfg=cfg)
        plus_vector = flatten_components(plus_outputs, components, scales)

        restore_parameter_refs(refs, base_values)
        set_parameter_flat_value(param, flat_idx, minus_value)
        minus_outputs = model_outputs(model=model, known_pars=known_pars, mech=mech0, u0=u0, times=times, cfg=cfg)
        minus_vector = flatten_components(minus_outputs, components, scales)

        column = (plus_vector - minus_vector) / (2.0 * finite_diff_eps)
        sensitivity_columns.append(column)
        perturbation_meta.append(
            {
                "parameter": param_label,
                "base_value": base_value,
                "plus_value": float(plus_value.detach().cpu()),
                "minus_value": float(minus_value.detach().cpu()),
                "relative_step_skipped": False,
                "parameter_number": global_idx + 1,
                "global_index": global_idx,
            }
        )

    restore_parameter_refs(refs, base_values)
    log_progress("sensitivity matrix assembly start")
    if sensitivity_columns:
        sensitivity = np.column_stack(sensitivity_columns)
    else:
        sensitivity = np.empty((baseline_vector.size, 0), dtype=float)
    log_progress(f"sensitivity matrix assembly done | shape={sensitivity.shape[0]}x{sensitivity.shape[1]}")

    if shard_count > 1:
        return {
            "schema_version": SHARD_SCHEMA,
            "rank_label": entry.get("rank_label"),
            "rank": entry.get("rank"),
            "candidate_b": entry.get("candidate_b"),
            "entry_payload": entry.get("entry_payload"),
            "parameter_set": "trained",
            "parameter_coordinates": parameter_labels,
            "parameter_global_indices": selected_global_indices,
            "parameter_count": len(parameter_labels),
            "full_parameter_count": full_parameter_count,
            "mechanistic_point": {"ks": ks0, "cs": cs0},
            "finite_difference": {
                "coordinate": "central difference in relative raw trained-parameter coordinates",
                "eps": finite_diff_eps,
                "zero_base_policy": "zero sensitivity column, matching the original parameter-scaled convention",
                "parameter_limit": int(parameter_limit),
                "perturbations": perturbation_meta,
            },
            "observable_components": list(components),
            "component_scales": {k: float(v) for k, v in scales.items() if k in components},
            "window_meta": jsonable(payload.get("window_meta", {})),
            "baseline_vector_length": int(baseline_vector.size),
            "baseline_outputs": {
                name: {
                    "min": float(np.min(baseline_outputs[name])),
                    "max": float(np.max(baseline_outputs[name])),
                    "rms": rms(baseline_outputs[name]),
                }
                for name in components
            },
            "shard": {
                "shard_count": int(shard_count),
                "shard_index": int(shard_index),
                "start_global_index": int(selected_global_indices[0]) if selected_global_indices else int(start),
                "end_global_index_exclusive": int(selected_global_indices[-1] + 1) if selected_global_indices else int(end),
            },
            "sensitivity": sensitivity,
        }

    log_progress("hessian build start")
    hessian = (sensitivity.T @ sensitivity) / float(max(sensitivity.shape[0], 1))
    hessian = 0.5 * (hessian + hessian.T)
    log_progress(f"hessian build done | shape={hessian.shape[0]}x{hessian.shape[1]}")
    log_progress("eigendecomposition start")
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    log_progress("eigendecomposition done")

    log_progress("eigen row summary start")
    column_norms = np.sqrt(np.sum(np.square(sensitivity), axis=0) / float(max(sensitivity.shape[0], 1)))
    rows = make_eigen_rows(parameter_labels=parameter_labels, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    log_progress("eigen row summary done")

    groups = parameter_group_counts(parameter_labels)

    return {
        "schema_version": RESULT_SCHEMA,
        "rank_label": entry.get("rank_label"),
        "rank": entry.get("rank"),
        "candidate_b": entry.get("candidate_b"),
        "entry_payload": entry.get("entry_payload"),
        "parameter_set": "trained",
        "parameter_coordinates": parameter_labels,
        "parameter_count": len(parameter_labels),
        "parameter_group_counts": groups,
        "mechanistic_point": {"ks": ks0, "cs": cs0},
        "finite_difference": {
            "coordinate": "central difference in relative raw trained-parameter coordinates",
            "eps": finite_diff_eps,
            "zero_base_policy": "zero sensitivity column, matching the original parameter-scaled convention",
            "parameter_limit": int(parameter_limit),
            "perturbations": perturbation_meta,
        },
        "observable_components": list(components),
        "component_scales": {k: float(v) for k, v in scales.items() if k in components},
        "window_meta": jsonable(payload.get("window_meta", {})),
        "baseline_vector_length": int(baseline_vector.size),
        "sensitivity_column_rms": {
            parameter_labels[i]: float(column_norms[i]) for i in range(len(parameter_labels))
        },
        "hessian": hessian,
        "eigenvalues": eigenvalues,
        "eigenvectors_columns": eigenvectors,
        "eigen_rows": rows,
        "baseline_outputs": {
            name: {
                "min": float(np.min(baseline_outputs[name])),
                "max": float(np.max(baseline_outputs[name])),
                "rms": rms(baseline_outputs[name]),
            }
            for name in components
        },
    }


def write_outputs(result: dict[str, Any], results_dir: Path) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rank_label = str(result.get("rank_label") or "unknown").replace(" ", "_")
    parameter_set = str(result.get("parameter_set") or "unknown").replace(" ", "_")
    parameter_limit = int(result.get("finite_difference", {}).get("parameter_limit", 0) or 0)
    run_parameter_set = f"{parameter_set}_limit{parameter_limit}" if parameter_limit > 0 else parameter_set
    prefix = results_dir / f"afm04_step3_identifiability_{run_parameter_set}_{rank_label}_{stamp}"
    latest_prefix = results_dir / f"afm04_step3_identifiability_{run_parameter_set}_latest"

    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    txt_path = prefix.with_suffix(".txt")
    npz_path = prefix.with_suffix(".npz")

    rows = result["eigen_rows"]
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_row = dict(row)
        csv_row["top_components_json"] = json.dumps(csv_row.pop("top_components", []), ensure_ascii=False)
        csv_rows.append(csv_row)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        key: jsonable(value)
        for key, value in result.items()
        if key not in {"hessian", "eigenvalues", "eigenvectors_columns"}
    }
    summary["hessian"] = jsonable(result["hessian"])
    summary["eigenvalues"] = jsonable(result["eigenvalues"])
    summary["eigenvectors_columns"] = jsonable(result["eigenvectors_columns"])
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "AFM04 step3 identifiability result",
        f"schema_version: {result['schema_version']}",
        f"object: {result.get('rank_label')} | rank={result.get('rank')} | candidate_b={result.get('candidate_b')}",
        f"entry_payload: {result.get('entry_payload')}",
        f"parameter_set: {result['parameter_set']}",
        f"parameter_count: {result['parameter_count']}",
        f"parameter_coordinates: {', '.join(result['parameter_coordinates'])}",
        f"ks: {result['mechanistic_point']['ks']:.12e}",
        f"cs: {result['mechanistic_point']['cs']:.12e}",
        f"finite_diff_eps: {result['finite_difference']['eps']:.6e}",
        f"observable_components: {', '.join(result['observable_components'])}",
        f"baseline_vector_length: {result['baseline_vector_length']}",
        "",
        "Eigen rows are sorted by ascending eigenvalue.",
    ]
    for row in rows:
        top = ", ".join(
            f"{item['parameter']}={item['component']:+.6f}" for item in row.get("top_components", [])[:5]
        )
        lines.append(
            "  "
            f"#{row['eigen_index_ascending']}: eigenvalue={row['eigenvalue']:.12e}, "
            f"log10={row['log10_eigenvalue']:.6f}, "
            f"rel={row['relative_to_max_abs_eigenvalue']:.6e}, "
            f"dominant={row['dominant_parameter']}, "
            f"top=[{top}]"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    np.savez(
        npz_path,
        hessian=np.asarray(result["hessian"], dtype=float),
        eigenvalues=np.asarray(result["eigenvalues"], dtype=float),
        eigenvectors_columns=np.asarray(result["eigenvectors_columns"], dtype=float),
    )

    latest_paths = {
        "csv": latest_prefix.with_suffix(".csv"),
        "json": latest_prefix.with_suffix(".json"),
        "txt": latest_prefix.with_suffix(".txt"),
        "npz": latest_prefix.with_suffix(".npz"),
    }
    for src, dst in ((csv_path, latest_paths["csv"]), (json_path, latest_paths["json"]), (txt_path, latest_paths["txt"]), (npz_path, latest_paths["npz"])):
        shutil.copyfile(src, dst)

    paths = {"csv": csv_path, "json": json_path, "txt": txt_path, "npz": npz_path}
    paths.update({f"latest_{key}": value for key, value in latest_paths.items()})
    return paths


def write_shard_output(result: dict[str, Any], shard_run_dir: Path) -> dict[str, Path]:
    if result.get("schema_version") != SHARD_SCHEMA:
        raise ValueError("write_shard_output expects a shard result")
    shard_run_dir.mkdir(parents=True, exist_ok=True)
    shard = result["shard"]
    shard_index = int(shard["shard_index"])
    shard_count = int(shard["shard_count"])
    prefix = shard_run_dir / f"shard_{shard_index:03d}_of_{shard_count:03d}"
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")

    np.savez(
        npz_path,
        sensitivity=np.asarray(result["sensitivity"], dtype=float),
        parameter_global_indices=np.asarray(result["parameter_global_indices"], dtype=np.int64),
    )

    meta = {
        key: jsonable(value)
        for key, value in result.items()
        if key != "sensitivity"
    }
    meta["sensitivity_npz"] = repo_rel(npz_path)
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"json": json_path, "npz": npz_path}


def load_shard_pair(json_path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SHARD_SCHEMA:
        raise ValueError(f"Unexpected shard schema in {json_path}: {meta.get('schema_version')!r}")
    npz_path = resolve_repo_path(meta["sensitivity_npz"])
    with np.load(npz_path) as data:
        sensitivity = np.asarray(data["sensitivity"], dtype=float)
        indices = np.asarray(data["parameter_global_indices"], dtype=np.int64)
    return meta, sensitivity, indices


def merge_trained_shards(shard_run_dir: Path, expected_shard_count: int | None = None) -> dict[str, Any]:
    json_paths = sorted(shard_run_dir.glob("shard_*_of_*.json"))
    if not json_paths:
        raise RuntimeError(f"No shard json files found in {shard_run_dir}")

    loaded = [load_shard_pair(path) for path in json_paths]
    first_meta = loaded[0][0]
    shard_count = int(first_meta["shard"]["shard_count"])
    if expected_shard_count is not None and expected_shard_count > 0 and shard_count != int(expected_shard_count):
        raise RuntimeError(f"Expected shard_count={expected_shard_count}, found shard_count={shard_count}")
    if len(loaded) != shard_count:
        raise RuntimeError(f"Expected {shard_count} shard files, found {len(loaded)} in {shard_run_dir}")

    full_parameter_count = int(first_meta["full_parameter_count"])
    n_obs = int(first_meta["baseline_vector_length"])
    sensitivity = np.empty((n_obs, full_parameter_count), dtype=float)
    parameter_labels: list[str | None] = [None] * full_parameter_count
    perturbations: list[Any] = [None] * full_parameter_count
    seen = np.zeros(full_parameter_count, dtype=bool)

    for meta, shard_sensitivity, indices in loaded:
        if int(meta["full_parameter_count"]) != full_parameter_count:
            raise RuntimeError("Shard full_parameter_count mismatch")
        if int(meta["baseline_vector_length"]) != n_obs:
            raise RuntimeError("Shard baseline_vector_length mismatch")
        if list(meta["observable_components"]) != list(first_meta["observable_components"]):
            raise RuntimeError("Shard observable_components mismatch")
        labels = list(meta["parameter_coordinates"])
        if shard_sensitivity.shape[0] != n_obs:
            raise RuntimeError("Shard sensitivity row count mismatch")
        if shard_sensitivity.shape[1] != len(indices) or len(labels) != len(indices):
            raise RuntimeError("Shard sensitivity/label/index count mismatch")
        if np.any(indices < 0) or np.any(indices >= full_parameter_count):
            raise RuntimeError("Shard parameter index out of bounds")

        for local_col, global_idx in enumerate(indices.tolist()):
            if seen[global_idx]:
                raise RuntimeError(f"Duplicate parameter global index {global_idx}")
            sensitivity[:, global_idx] = shard_sensitivity[:, local_col]
            parameter_labels[global_idx] = labels[local_col]
            seen[global_idx] = True
        for item in meta["finite_difference"].get("perturbations", []):
            global_idx = int(item.get("global_index", -1))
            if 0 <= global_idx < full_parameter_count:
                perturbations[global_idx] = item

    missing = np.where(~seen)[0]
    if missing.size:
        preview = ", ".join(str(int(v)) for v in missing[:10])
        raise RuntimeError(f"Missing sensitivity columns for global indices: {preview}")

    final_labels = [str(label) for label in parameter_labels]
    log_progress(f"merged sensitivity matrix | shape={sensitivity.shape[0]}x{sensitivity.shape[1]}")
    log_progress("hessian build start")
    hessian = (sensitivity.T @ sensitivity) / float(max(sensitivity.shape[0], 1))
    hessian = 0.5 * (hessian + hessian.T)
    log_progress(f"hessian build done | shape={hessian.shape[0]}x{hessian.shape[1]}")
    log_progress("eigendecomposition start")
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    log_progress("eigendecomposition done")

    log_progress("eigen row summary start")
    column_norms = np.sqrt(np.sum(np.square(sensitivity), axis=0) / float(max(sensitivity.shape[0], 1)))
    rows = make_eigen_rows(parameter_labels=final_labels, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    log_progress("eigen row summary done")

    finite_difference = dict(first_meta["finite_difference"])
    finite_difference["shard_count"] = shard_count
    finite_difference["shard_run_dir"] = repo_rel(shard_run_dir)
    finite_difference["perturbations"] = [item for item in perturbations if item is not None]

    return {
        "schema_version": RESULT_SCHEMA,
        "rank_label": first_meta.get("rank_label"),
        "rank": first_meta.get("rank"),
        "candidate_b": first_meta.get("candidate_b"),
        "entry_payload": first_meta.get("entry_payload"),
        "parameter_set": "trained",
        "parameter_coordinates": final_labels,
        "parameter_count": len(final_labels),
        "parameter_group_counts": parameter_group_counts(final_labels),
        "mechanistic_point": first_meta["mechanistic_point"],
        "finite_difference": finite_difference,
        "observable_components": list(first_meta["observable_components"]),
        "component_scales": first_meta["component_scales"],
        "window_meta": first_meta.get("window_meta", {}),
        "baseline_vector_length": int(n_obs),
        "sensitivity_column_rms": {
            final_labels[i]: float(column_norms[i]) for i in range(len(final_labels))
        },
        "hessian": hessian,
        "eigenvalues": eigenvalues,
        "eigenvectors_columns": eigenvectors,
        "eigen_rows": rows,
        "baseline_outputs": first_meta.get("baseline_outputs", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry",
        default="AFM04/step3_parameters_identifiability/results/current_step3_entry.json",
        help="Current step3 entry JSON written by set_afm04_step3_entry.py.",
    )
    parser.add_argument(
        "--results-dir",
        default="AFM04/step3_parameters_identifiability/results",
        help="Directory for step3 identifiability outputs.",
    )
    parser.add_argument("--parameter-set", choices=("trained", "mech"), default="trained")
    parser.add_argument("--finite-diff-eps", type=float, default=1.0e-4)
    parser.add_argument(
        "--components",
        default=",".join(DEFAULT_COMPONENTS),
        help="Comma-separated observable components: x1,x2,x2dot,force; x3 is optional but not default.",
    )
    parser.add_argument(
        "--parameter-limit",
        type=int,
        default=0,
        help="Debug/smoke-test only: limit the number of scalar trained parameters. 0 means all.",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0, help="1-based shard index. 0 means unsharded full run.")
    parser.add_argument(
        "--shard-run-dir",
        default="",
        help="Directory for parameter-column shard outputs or merge input.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge previously computed trained parameter-column shards and compute final H/eigen.",
    )
    args = parser.parse_args()

    if args.finite_diff_eps <= 0.0:
        raise ValueError("--finite-diff-eps must be positive")
    results_dir = resolve_repo_path(args.results_dir)
    if args.parameter_limit < 0:
        raise ValueError("--parameter-limit must be non-negative")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")

    if args.merge_shards:
        if args.parameter_set != "trained":
            raise ValueError("--merge-shards is only valid for --parameter-set trained")
        if not args.shard_run_dir:
            raise ValueError("--merge-shards requires --shard-run-dir")
        result = merge_trained_shards(resolve_repo_path(args.shard_run_dir), expected_shard_count=int(args.shard_count))
        paths = write_outputs(result, results_dir)
        print("AFM04 step3 shard merge completed")
        print(f"object -> {result.get('rank_label')} | rank={result.get('rank')} | candidate B{result.get('candidate_b')}")
        print(f"parameter_set -> {result['parameter_set']} | parameter_count={result['parameter_count']}")
        print(f"shard_run_dir -> {resolve_repo_path(args.shard_run_dir)}")
        print(f"csv -> {paths['csv']}")
        print(f"json -> {paths['json']}")
        print(f"txt -> {paths['txt']}")
        print(f"npz -> {paths['npz']}")
        print(f"latest_json -> {paths['latest_json']}")
        return

    if args.shard_count > 1:
        if args.parameter_set != "trained":
            raise ValueError("--shard-count > 1 is only valid for --parameter-set trained")
        if not (1 <= int(args.shard_index) <= int(args.shard_count)):
            raise ValueError("--shard-index must be 1-based and within --shard-count")
        if not args.shard_run_dir:
            raise ValueError("--shard-count > 1 requires --shard-run-dir")
    elif args.shard_index:
        raise ValueError("--shard-index requires --shard-count > 1")

    entry_path = resolve_repo_path(args.entry)
    components = parse_components(args.components)

    entry = load_current_entry(entry_path)
    payload_path = resolve_repo_path(entry["entry_payload"])
    payload = load_payload(payload_path)

    if args.parameter_set == "trained":
        result = compute_trained_identifiability(
            entry=entry,
            payload=payload,
            components=components,
            finite_diff_eps=float(args.finite_diff_eps),
            parameter_limit=int(args.parameter_limit),
            shard_count=int(args.shard_count),
            shard_index=int(args.shard_index),
        )
    else:
        result = compute_mech_identifiability(
            entry=entry,
            payload=payload,
            components=components,
            finite_diff_eps=float(args.finite_diff_eps),
        )

    if result.get("schema_version") == SHARD_SCHEMA:
        paths = write_shard_output(result, resolve_repo_path(args.shard_run_dir))
        shard = result["shard"]
        print("AFM04 step3 sensitivity shard completed")
        print(f"object -> {result.get('rank_label')} | rank={result.get('rank')} | candidate B{result.get('candidate_b')}")
        print(
            f"shard -> {int(shard['shard_index'])}/{int(shard['shard_count'])} | "
            f"columns={int(shard['start_global_index']) + 1}-{int(shard['end_global_index_exclusive'])}"
        )
        print(f"parameter_count -> {result['parameter_count']} of {result['full_parameter_count']}")
        print(f"json -> {paths['json']}")
        print(f"npz -> {paths['npz']}")
        return

    paths = write_outputs(result, results_dir)

    print("AFM04 step3 identifiability completed")
    print(f"object -> {result.get('rank_label')} | rank={result.get('rank')} | candidate B{result.get('candidate_b')}")
    print(f"parameter_set -> {result['parameter_set']} | parameter_count={result['parameter_count']}")
    for row in result["eigen_rows"]:
        top = ", ".join(
            f"{item['parameter']}={item['component']:+.4f}" for item in row.get("top_components", [])[:3]
        )
        print(
            f"eig#{row['eigen_index_ascending']} "
            f"value={row['eigenvalue']:.12e} "
            f"dominant={row['dominant_parameter']} "
            f"top=[{top}]"
        )
    print(f"csv -> {paths['csv']}")
    print(f"json -> {paths['json']}")
    print(f"txt -> {paths['txt']}")
    print(f"npz -> {paths['npz']}")
    print(f"latest_json -> {paths['latest_json']}")


if __name__ == "__main__":
    main()
