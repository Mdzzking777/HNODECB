"""Pre-optimization rollout snapshot for AFM04 stage1pluslight ranks.

This mirrors the stage2light preopt-rank plot, but reads stage1pluslight
random-search ranks directly.  Each selected rank is rebuilt from its
mechanistic grid point and KAN seed, then rolled out before any GBO update.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from AFM04.KAN_full_test.rollout import (  # noqa: E402
    fts_truth_from_states_torch,
    rollout_single_shooting_torch,
    x2dot_rhs_torch,
)
from AFM04.stage1pluslight.kan_rs_trial import (  # noqa: E402
    _observable_grid_inputs_from_ode,
    prepare_kan_stage1_runtime,
)
from AFM04.stage2light.kan_backend import (  # noqa: E402
    KANForceModule,
    initial_grid_support_from_raw_inputs,
)


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
DEFAULT_RESULT_PATH = REPO_ROOT / "AFM04" / "stage1pluslight" / "results_afm" / "afm_param_stage1pluslight_04.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "stage1pluslight" / "visualization"
DEFAULT_OUT_PREFIX = "afm_param_stage1pluslight_04_preopt"


def _finite_float(value: Any, default: float = math.nan) -> float:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    return default


def _record_loss(rec: dict[str, Any]) -> float:
    return _finite_float(rec.get("loss", rec.get("train_loss", math.nan)), math.inf)


def _params(rec: dict[str, Any]) -> dict[str, Any]:
    params = rec.get("params", {})
    return params if isinstance(params, dict) else {}


def _load_payload(path: Path = DEFAULT_RESULT_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"stage1pluslight result payload not found: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected stage1pluslight payload type: {type(payload)!r}")
    return payload


def _finite_ranked_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("trial_parameters", [])
    if not isinstance(records, list) or not records:
        records = payload.get("ranked_topk", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError("stage1pluslight payload has no trial records to rank.")
    finite = [rec for rec in records if isinstance(rec, dict) and math.isfinite(_record_loss(rec))]
    if not finite:
        raise RuntimeError("stage1pluslight payload has no finite trial records.")
    return sorted(finite, key=_record_loss)


def _parse_selections(raw: str | None) -> list[tuple[str, int]]:
    text = (raw or "").strip()
    if not text:
        text = os.environ.get("HNODECB_AFM04_STAGE1PLUS_PREOPT_RANKS", "247")
    out: list[tuple[str, int]] = []
    for item in text.replace(";", ",").split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token.startswith("trial=") or token.startswith("trial:"):
            out.append(("trial", int(token.split("=", 1)[-1].split(":", 1)[-1])))
        elif token.startswith("rank=") or token.startswith("rank:"):
            out.append(("rank", int(token.split("=", 1)[-1].split(":", 1)[-1])))
        else:
            out.append(("rank", int(token)))
    if not out:
        out.append(("rank", 1))
    return out


def _select_records(payload: dict[str, Any], selections: list[tuple[str, int]]) -> list[tuple[int, dict[str, Any]]]:
    ranked = _finite_ranked_records(payload)
    trial_to_rank: dict[int, tuple[int, dict[str, Any]]] = {}
    for idx, rec in enumerate(ranked, start=1):
        trial_id = int(_params(rec).get("trial_id", -1))
        if trial_id >= 0 and trial_id not in trial_to_rank:
            trial_to_rank[trial_id] = (idx, rec)

    selected: list[tuple[int, dict[str, Any]]] = []
    for kind, value in selections:
        if kind == "trial":
            try:
                selected.append(trial_to_rank[int(value)])
            except KeyError as exc:
                raise KeyError(f"trial_id={value} was not found among finite stage1pluslight trials") from exc
        else:
            if value < 1 or value > len(ranked):
                raise IndexError(f"rank={value} is outside the finite-rank range 1..{len(ranked)}")
            selected.append((int(value), ranked[value - 1]))
    return selected


def _output_filename(selected: list[tuple[int, dict[str, Any]]]) -> str:
    ranks = [str(rank) for rank, _ in selected]
    if len(ranks) == 1:
        suffix = f"rank{ranks[0]}"
    else:
        suffix = "ranks" + "_".join(ranks)
    return f"{DEFAULT_OUT_PREFIX}_{suffix}_grid.png"


def _runtime_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    window_mode = payload.get("stage1plus_window_mode", None)
    arch_window_us = payload.get("stage1plus_arch_window_us", None)
    return prepare_kan_stage1_runtime(
        REPO_ROOT,
        window_mode=str(window_mode) if window_mode is not None else None,
        arch_window_us=float(arch_window_us) if arch_window_us is not None else None,
    )


def _build_stage1_model(
    *,
    runtime: dict[str, Any],
    rec: dict[str, Any],
) -> tuple[KANForceModule, torch.Tensor]:
    cfg = runtime["cfg"]
    prepared = runtime["prepared"]
    tensors = runtime["tensors"]
    train_states = runtime["train_states"]
    train_fts = runtime["train_fts"]
    dtype: torch.dtype = runtime["dtype"]
    params = _params(rec)

    cfg = replace(
        cfg,
        grid=int(params.get("kan_grid", cfg.grid)),
        spline_k=int(params.get("kan_spline_k", cfg.spline_k)),
        base_fun=str(params.get("kan_base_fun", cfg.base_fun)),
    )
    grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=tensors["ode_train"],
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
    )
    initial_grid_support = initial_grid_support_from_raw_inputs(
        grid_inputs,
        prepared.state_mean,
        prepared.state_scale,
    )
    model = KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=int(params["nn_init_seed"]),
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
        initial_grid_support=initial_grid_support,
        dist=float(prepared.known_pars[6]),
        a0=float(prepared.known_pars[9]),
        gnn_learnable=cfg.gnn_learnable,
        soft_mask_enabled=True,
        soft_mask_trainable=False,
        soft_mask_s0_a0=20.0,
        soft_mask_s0_min_a0=1.0,
        soft_mask_s0_max_a0=100.0,
        soft_mask_alpha_a0=0.25,
        soft_mask_alpha_min_a0=0.02,
        soft_mask_alpha_max_a0=5.0,
        device=cfg.device,
        dtype=dtype,
    ).to(cfg.device)

    with torch.no_grad():
        if cfg.adaptive_grid_enabled:
            model.update_grid_from_normalized_inputs(grid_inputs)
        model.initialize_gain_from_truth(train_states, train_fts)

    mech = torch.as_tensor(
        [float(params["ks0"]), float(params["cs0"])],
        dtype=dtype,
        device=cfg.device,
    )
    return model, mech


def _build_preopt_snapshot(
    *,
    runtime: dict[str, Any],
    rec: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    cfg = runtime["cfg"]
    prepared = runtime["prepared"]
    split = runtime["split"]
    tensors = runtime["tensors"]
    dtype: torch.dtype = runtime["dtype"]
    model, mech = _build_stage1_model(runtime=runtime, rec=rec)

    ode_true = tensors["ode_full"]
    times = tensors["times_full"]
    mech_true = torch.as_tensor(prepared.mech_true, dtype=dtype, device=cfg.device)

    with torch.no_grad():
        traj_pred = rollout_single_shooting_torch(
            force_module=model,
            known_pars=prepared.known_pars,
            mech_true=mech,
            u0=ode_true[:, 0],
            times=times,
            method=cfg.ode_method,
            rtol=cfg.ode_rtol,
            atol=cfg.ode_atol,
            x1_abs_guard=runtime.get("x1_abs_guard"),
            x2_abs_guard=runtime.get("x2_abs_guard"),
        )
        x2dot_pred = x2dot_rhs_torch(traj_pred, times, model, prepared.known_pars)
        teacher_states = ode_true.transpose(0, 1)
        fts_true = fts_truth_from_states_torch(
            teacher_states,
            prepared.known_pars,
            eta_star=prepared.eta_star_true,
            mech_true=mech_true,
        )
        fts_pred = model(traj_pred.transpose(0, 1))

    return {
        "rank": int(rank),
        "record": rec,
        "times_us": 1.0e6 * times.detach().cpu().numpy(),
        "ode_true": ode_true.detach().cpu().numpy(),
        "traj_pred": traj_pred.detach().cpu().numpy(),
        "x2dot_true": np.asarray(split.x2dot_full, dtype=float),
        "x2dot_pred": x2dot_pred.detach().cpu().numpy(),
        "fts_true": fts_true.detach().cpu().numpy(),
        "fts_pred": fts_pred.detach().cpu().numpy(),
    }


def _row_label(rank: int, rec: dict[str, Any]) -> str:
    params = _params(rec)
    return (
        f"rank={rank} | trial={int(params.get('trial_id', -1))} | "
        f"{params.get('node_label', 'node=?')} | seedbank={int(params.get('nn_seed_bank_idx', -1))} | "
        f"loss={_record_loss(rec):.6e} | "
        f"ks={float(params.get('ks0', rec.get('ks_hat', math.nan))):.6e} "
        f"({float(rec.get('ks_err_pct', math.nan)):.2f}%) | "
        f"cs={float(params.get('cs0', rec.get('cs_hat', math.nan))):.6e} "
        f"({float(rec.get('cs_err_pct', math.nan)):.2f}%)"
    )


def _plot_row(axes: np.ndarray, snap: dict[str, Any]) -> None:
    times_us = np.asarray(snap["times_us"], dtype=float)
    ode_true = np.asarray(snap["ode_true"], dtype=float)
    traj_pred = np.asarray(snap["traj_pred"], dtype=float)
    x2dot_true = np.asarray(snap["x2dot_true"], dtype=float)
    x2dot_pred = np.asarray(snap["x2dot_pred"], dtype=float)
    fts_true = np.asarray(snap["fts_true"], dtype=float)
    fts_pred = np.asarray(snap["fts_pred"], dtype=float)

    specs = [
        ("Fts before prest2 x3-refit", "force (N)", fts_true, fts_pred, "Fts teacher true", "Fts rollout pred"),
        ("x1 before prest2 x3-refit", "x1", ode_true[0, :], traj_pred[0, :], "x1 true", "x1 pred"),
        ("x2 before prest2 x3-refit", "x2", ode_true[1, :], traj_pred[1, :], "x2 true", "x2 pred"),
        ("x2dot before prest2 x3-refit", "x2dot", x2dot_true, x2dot_pred, "x2dot true", "x2dot pred"),
        ("x3 before prest2 x3-refit", "x3", ode_true[2, :], traj_pred[2, :], "x3 true", "x3 pred"),
    ]
    for ax, (title, ylabel, true_y, pred_y, true_label, pred_label) in zip(axes, specs, strict=True):
        ax.plot(times_us, true_y, color="black", linewidth=2, label=true_label)
        ax.plot(times_us, pred_y, color="crimson", linewidth=2, linestyle="--", label=pred_label)
        ax.set_title(title)
        ax.set_xlabel("time (us)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")


def run_one(
    *,
    result_path: Path = DEFAULT_RESULT_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    selections: str | None = None,
) -> Path:
    payload = _load_payload(result_path)
    selected = _select_records(payload, _parse_selections(selections))
    runtime = _runtime_from_payload(payload)

    fig, axes = plt.subplots(len(selected), 5, figsize=(30, 4.8 * len(selected)), squeeze=False, sharex=False)
    row_labels: list[str] = []
    for row, (rank, rec) in enumerate(selected):
        snap = _build_preopt_snapshot(runtime=runtime, rec=rec, rank=rank)
        _plot_row(axes[row, :], snap)
        row_labels.append(_row_label(rank, rec))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle(
        "AFM04 stage1pluslight preopt rank snapshot | X3-REFIT: BEFORE\n"
        "before prest2 x3-refit\n"
        "x3 normalizer: x1-prior, x3 support: neutral\n"
        + "\n".join(row_labels),
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    path = out_dir / _output_filename(selected)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_OUT_DIR
    selections = sys.argv[2] if len(sys.argv) >= 3 else None
    path = run_one(out_dir=out_dir, selections=selections)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
