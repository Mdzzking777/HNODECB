"""Probe A: reproduce AFM05 st1pl entry rollout without running prest2.

This diagnostic deliberately freezes the st1pl entry conditions:

- use the st1pl prior normalizer from AFM05.stage2light.data.prepare_data
  (x3 mean = x3(t0), x3 scale = 0.1 * x1 scale);
- use the st1pl initial AGU/grid support recorded in the trial when available;
- perform the same initial AGU update and gain initialization used by st1pl;
- do one rollout/evaluation only.

It deliberately does NOT do the prest2/st2l x3 refit chain:

- no x3 mean/scale recomputation from st1pl x3_pred(t);
- no rollout-derived x3 support;
- no second AGU/grid update;
- no optimizer step.

The purpose is to answer whether the prest2/st2l entry machinery can reproduce
the st1pl rollout under the original st1pl entry conditions.  If this probe is
close to the cached st1pl loss, the later jump is likely from the intended
x3-refit/support/AGU chain.  If it is not close, the entry reconstruction itself
is already inconsistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM05.stage2light.config import default_config
from AFM05.stage2light.data import prepare_data
from AFM05.stage2light.kan_backend import KANForceModule, initial_grid_support_from_raw_inputs
from AFM05.stage2light.losses import evaluate_split
from AFM05.stage2light.rollout import LearnableMechModule, _afm05_known_fields
from AFM05.stage2light.train import (
    _make_grid_update_inputs,
    _observable_grid_inputs_from_ode,
    _selected_window_indices,
    _torch_dtype,
    _window_to_torch,
)


DEFAULT_ST1PL = REPO_ROOT / "AFM05" / "stage1pluslight" / "results_afm" / "afm_param_stage1pluslight_05.pkl"
DEFAULT_OUTDIR = REPO_ROOT / "AFM05" / "prestage2" / "diagnostics" / "probe_a_st1pl_entry_replay"


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _record_loss(rec: dict[str, Any]) -> float:
    for key in ("train_loss", "loss", "val_loss"):
        try:
            val = float(rec.get(key, float("nan")))
        except Exception:
            val = float("nan")
        if math.isfinite(val):
            return val
    return float("inf")


def _trial_id(rec: dict[str, Any]) -> int:
    params = rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}
    return int(params.get("trial_id", 0))


def _ranked_records(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    records = [rec for rec in payload.get("trial_parameters", []) if isinstance(rec, dict)]
    ranked = sorted(records, key=_record_loss)
    return [(idx, rec) for idx, rec in enumerate(ranked, start=1)]


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in str(raw or "").replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            out.append(int(item))
    return out


def _select_probe_records(
    payload: dict[str, Any],
    *,
    ranks: Iterable[int],
    trial_ids: Iterable[int],
    limit: int,
) -> list[tuple[int, dict[str, Any]]]:
    ranked = _ranked_records(payload)
    by_rank = {rank: rec for rank, rec in ranked}
    by_trial_id = {_trial_id(rec): (rank, rec) for rank, rec in ranked if _trial_id(rec) > 0}

    selected: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()

    for rank in ranks:
        if rank not in by_rank:
            raise ValueError(f"rank {rank} is not available; total records={len(ranked)}")
        rec = by_rank[rank]
        tid = _trial_id(rec)
        if tid not in seen:
            selected.append((rank, rec))
            seen.add(tid)

    for tid in trial_ids:
        if tid not in by_trial_id:
            raise ValueError(f"trial_id {tid} was not found in the st1pl payload")
        rank, rec = by_trial_id[tid]
        if tid not in seen:
            selected.append((rank, rec))
            seen.add(tid)

    if not selected:
        default_ranks = [1, 10, 30]
        for rank in default_ranks:
            if rank <= len(ranked):
                rec = by_rank[rank]
                tid = _trial_id(rec)
                if tid not in seen:
                    selected.append((rank, rec))
                    seen.add(tid)

    if limit > 0:
        selected = selected[:limit]
    return selected


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _gain_set(model: KANForceModule, gain: float) -> None:
    gain = float(gain)
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"cannot set nonpositive/nonfinite gain: {gain!r}")
    with torch.no_grad():
        model.log_gnn.data = torch.log(
            torch.as_tensor([gain], dtype=model.log_gnn.dtype, device=model.log_gnn.device)
        )


def _loss_row(prefix: str, total: torch.Tensor, parts: Any) -> dict[str, float]:
    return {
        f"{prefix}_loss": float(total.detach().cpu()),
        f"{prefix}_state": float(parts.state),
        f"{prefix}_x1_state": float(parts.x1_state),
        f"{prefix}_x2_state": float(parts.x2_state),
        f"{prefix}_x2dot": float(parts.x2dot),
        f"{prefix}_x3_range": float(parts.x3_range),
        f"{prefix}_fts_range": float(parts.fts_range),
        f"{prefix}_x1_rec_pct": float(parts.x1_rec),
        f"{prefix}_x2_rec_pct": float(parts.x2_rec),
        f"{prefix}_x2dot_rec_pct": float(parts.x2dot_rec),
    }


def _traj_x3_meta(prefix: str, traj: torch.Tensor) -> dict[str, float]:
    x3 = traj[2, :].detach().cpu().numpy().astype(float)
    return {
        f"{prefix}_x3_min": float(np.min(x3)),
        f"{prefix}_x3_max": float(np.max(x3)),
        f"{prefix}_x3_mean": float(np.mean(x3)),
        f"{prefix}_x3_std": float(np.std(x3)),
    }


def _build_probe_model(
    *,
    cfg: Any,
    prepared: Any,
    split: Any,
    tensors: dict[str, torch.Tensor],
    record: dict[str, Any],
    dtype: torch.dtype,
) -> tuple[KANForceModule, LearnableMechModule, dict[str, float]]:
    params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
    known_fields = _afm05_known_fields(prepared.known_pars)

    saved_support = params.get("initial_grid_support")
    if saved_support is None:
        saved_support_arr = None
    else:
        saved_support_arr = np.asarray(saved_support, dtype=float)

    prior_grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=tensors["ode_train"],
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        x3_norm_support=None,
    )
    recomputed_support = initial_grid_support_from_raw_inputs(
        prior_grid_inputs,
        prepared.state_mean,
        prepared.state_scale,
    )
    initial_grid_support = saved_support_arr if saved_support_arr is not None else recomputed_support
    support_diff = (
        float(np.max(np.abs(saved_support_arr - recomputed_support)))
        if saved_support_arr is not None and saved_support_arr.shape == recomputed_support.shape
        else float("nan")
    )

    grid = int(params.get("kan_grid", cfg.grid))
    spline_k = int(params.get("kan_spline_k", cfg.spline_k))
    base_fun = str(params.get("kan_base_fun", cfg.base_fun))
    seed = int(params.get("nn_init_seed", cfg.seed))

    model = KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=seed,
        width=cfg.width,
        grid=grid,
        spline_k=spline_k,
        base_fun=base_fun,
        symbolic_enabled=cfg.symbolic_enabled,
        auto_save=cfg.auto_save,
        noise_scale=cfg.noise_scale,
        affine_trainable=cfg.affine_trainable,
        grid_eps=cfg.grid_eps,
        grid_range=(cfg.grid_range_lo, cfg.grid_range_hi),
        initial_grid_support=initial_grid_support,
        dist=float(known_fields["Z"]),
        a0=float(known_fields["a0"]),
        gnn_learnable=cfg.gnn_learnable,
        soft_mask_enabled=True,
        soft_mask_trainable=False,
        soft_mask_s0_a0=cfg.soft_mask_s0_a0,
        soft_mask_s0_min_a0=cfg.soft_mask_s0_min_a0,
        soft_mask_s0_max_a0=cfg.soft_mask_s0_max_a0,
        soft_mask_alpha_a0=cfg.soft_mask_alpha_a0,
        soft_mask_alpha_min_a0=cfg.soft_mask_alpha_min_a0,
        soft_mask_alpha_max_a0=cfg.soft_mask_alpha_max_a0,
        device=cfg.device,
        dtype=dtype,
    ).to(cfg.device)

    with torch.no_grad():
        if bool(cfg.adaptive_grid_enabled):
            grid_inputs, _grid_meta = _make_grid_update_inputs(base_inputs=prior_grid_inputs)
            model.update_grid_from_normalized_inputs(grid_inputs)
        train_states = torch.as_tensor(split.ode_train.T, dtype=dtype, device=cfg.device)
        train_gain_force_reference = torch.as_tensor(split.train_gain_force_reference, dtype=dtype, device=cfg.device)
        gain_reinit = float(model.initialize_gain_from_reference(train_states, train_gain_force_reference))

    mech = LearnableMechModule(
        ks_init=float(record.get("ks_hat", params.get("ks0", float("nan")))),
        cs_init=float(record.get("cs_hat", params.get("cs0", float("nan")))),
        ks_bounds=(cfg.ks_lo, cfg.ks_hi),
        cs_bounds=(cfg.cs_lo, cfg.cs_hi),
        dtype=dtype,
        device=cfg.device,
        parameterization=cfg.mech_parameterization,
    ).to(cfg.device)

    meta = {
        "support_max_abs_diff_saved_vs_recomputed": support_diff,
        "gain_reinitialized": gain_reinit,
        "grid": float(grid),
        "spline_k": float(spline_k),
        "seed": float(seed),
    }
    return model, mech, meta


def _evaluate_once(
    *,
    cfg: Any,
    prepared: Any,
    tensors: dict[str, torch.Tensor],
    model: KANForceModule,
    mech: LearnableMechModule,
) -> tuple[torch.Tensor, Any, torch.Tensor]:
    with torch.no_grad():
        return evaluate_split(
            force_module=model,
            known_pars=prepared.known_pars,
            mech_module=mech,
            ode_true=tensors["ode_full"],
            x2dot_true=tensors["x2dot_full"],
            contact_mask=tensors["contact_full"],
            times=tensors["times_full"],
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            loss_indices=tensors["train_idx"],
        )


def run_probe(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _load_pickle(args.stage1)
    selected = _select_probe_records(
        payload,
        ranks=_parse_int_list(args.ranks),
        trial_ids=_parse_int_list(args.trial_ids),
        limit=int(args.limit),
    )
    if not selected:
        raise RuntimeError("no probe records selected")

    cfg = default_config(REPO_ROOT)
    cfg = replace(
        cfg,
        stage1_input_path=Path(args.stage1),
        shard_index=1,
        shard_count=1,
        device=str(args.device),
        resume_from_checkpoint=False,
        warmstart_source="stage1",
    )
    dtype = _torch_dtype(cfg.dtype)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    prepared = prepare_data(cfg)
    selected_window_indices = _selected_window_indices(cfg, prepared.splits)
    split = prepared.splits[int(selected_window_indices[0]) - 1]
    tensors = _window_to_torch(split, dtype=dtype, device=cfg.device)

    rows: list[dict[str, Any]] = []
    for rank, record in selected:
        params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
        model, mech, build_meta = _build_probe_model(
            cfg=cfg,
            prepared=prepared,
            split=split,
            tensors=tensors,
            record=record,
            dtype=dtype,
        )

        total_reinit, parts_reinit, traj_reinit = _evaluate_once(
            cfg=cfg,
            prepared=prepared,
            tensors=tensors,
            model=model,
            mech=mech,
        )

        saved_gain = _float_or_nan(record.get("nn_gain", float("nan")))
        saved_gain_eval_ok = math.isfinite(saved_gain) and saved_gain > 0.0
        if saved_gain_eval_ok:
            _gain_set(model, saved_gain)
            total_saved, parts_saved, traj_saved = _evaluate_once(
                cfg=cfg,
                prepared=prepared,
                tensors=tensors,
                model=model,
                mech=mech,
            )
        else:
            total_saved = torch.as_tensor(float("nan"), dtype=dtype, device=cfg.device)
            parts_saved = parts_reinit
            traj_saved = traj_reinit

        cached = _float_or_nan(record.get("train_loss", record.get("loss", float("nan"))))
        row: dict[str, Any] = {
            "stage1_rank": int(rank),
            "trial_id": int(params.get("trial_id", _trial_id(record))),
            "ks0": float(record.get("ks_hat", params.get("ks0", float("nan")))),
            "cs0": float(record.get("cs_hat", params.get("cs0", float("nan")))),
            "ks_node_idx": int(params.get("ks_node_idx", 0)),
            "cs_node_idx": int(params.get("cs_node_idx", 0)),
            "nn_seed_bank_idx": int(params.get("nn_seed_bank_idx", -1)),
            "nn_init_seed": int(params.get("nn_init_seed", cfg.seed)),
            "cached_st1pl_train_loss": cached,
            "cached_st1pl_x1_rec_pct": _float_or_nan((record.get("train_parts") or {}).get("x1_rec", float("nan"))),
            "cached_st1pl_x2_rec_pct": _float_or_nan((record.get("train_parts") or {}).get("x2_rec", float("nan"))),
            "cached_st1pl_x2dot_rec_pct": _float_or_nan((record.get("train_parts") or {}).get("x2dot_rec", float("nan"))),
            "saved_nn_gain": saved_gain,
            "gain_reinitialized": float(build_meta["gain_reinitialized"]),
            "gain_reinit_over_saved": float(build_meta["gain_reinitialized"]) / saved_gain
            if saved_gain_eval_ok
            else float("nan"),
            "support_max_abs_diff_saved_vs_recomputed": float(build_meta["support_max_abs_diff_saved_vs_recomputed"]),
            "state_mean_x1": float(prepared.state_mean[0]),
            "state_mean_x2": float(prepared.state_mean[1]),
            "state_mean_x3": float(prepared.state_mean[2]),
            "state_scale_x1": float(prepared.state_scale[0]),
            "state_scale_x2": float(prepared.state_scale[1]),
            "state_scale_x3": float(prepared.state_scale[2]),
        }
        row.update(_loss_row("probe_a_gain_reinit", total_reinit, parts_reinit))
        row.update(_traj_x3_meta("probe_a_gain_reinit", traj_reinit))
        row.update(_loss_row("probe_a_saved_gain", total_saved, parts_saved))
        row.update(_traj_x3_meta("probe_a_saved_gain", traj_saved))
        row["ratio_gain_reinit_to_cached"] = row["probe_a_gain_reinit_loss"] / max(cached, 1.0e-300)
        row["ratio_saved_gain_to_cached"] = row["probe_a_saved_gain_loss"] / max(cached, 1.0e-300)
        rows.append(row)

    summary = {
        "stage1_path": str(Path(args.stage1).resolve()),
        "outdir": str(Path(args.outdir).resolve()),
        "n": len(rows),
        "window_role": str(split.role),
        "window_label": str(split.label),
        "window_start_idx": int(split.start_idx),
        "window_stop_idx": int(split.stop_idx),
        "window_sample_stride": int(split.sample_stride),
        "excluded_by_design": [
            "x3 mean/scale refit from st1pl x3_pred",
            "rollout-derived x3 support",
            "second AGU/grid update",
            "optimizer step",
        ],
    }
    for key in ("ratio_gain_reinit_to_cached", "ratio_saved_gain_to_cached"):
        vals = np.asarray([float(row[key]) for row in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        summary[key] = {
            "min": float(np.min(vals)) if vals.size else float("nan"),
            "median": float(np.median(vals)) if vals.size else float("nan"),
            "max": float(np.max(vals)) if vals.size else float("nan"),
        }
    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"probe_a_st1pl_entry_replay_{stamp}.csv"
    json_path = outdir / f"probe_a_st1pl_entry_replay_{stamp}.json"
    txt_path = outdir / f"probe_a_st1pl_entry_replay_{stamp}.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "AFM05 Probe A: st1pl-entry rollout replay",
        f"n = {summary['n']}",
        f"stage1_path = {summary['stage1_path']}",
        f"window = {summary['window_label']} ({summary['window_role']})",
        "",
        "This probe freezes st1pl entry conditions and excludes:",
    ]
    lines.extend([f"- {item}" for item in summary["excluded_by_design"]])
    lines.append("")
    for key in ("ratio_gain_reinit_to_cached", "ratio_saved_gain_to_cached"):
        stats = summary.get(key, {})
        lines.append(
            f"{key}: min={stats.get('min', float('nan')):.6e}, "
            f"median={stats.get('median', float('nan')):.6e}, "
            f"max={stats.get('max', float('nan')):.6e}"
        )
    lines.append("")
    for row in rows:
        lines.append(
            "rank={stage1_rank} trial={trial_id} seed={nn_seed_bank_idx} "
            "cached={cached_st1pl_train_loss:.6e} "
            "probe_gain_reinit={probe_a_gain_reinit_loss:.6e} "
            "probe_saved_gain={probe_a_saved_gain_loss:.6e} "
            "ratio_reinit={ratio_gain_reinit_to_cached:.3f} "
            "ratio_saved={ratio_saved_gain_to_cached:.3f}".format(**row)
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "txt": txt_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe A: replay st1pl entry rollout under frozen conditions.")
    parser.add_argument("--stage1", type=Path, default=DEFAULT_ST1PL, help="AFM05 st1pl merged result pkl.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Diagnostic output directory.")
    parser.add_argument("--ranks", default="", help="Comma-separated st1pl ranks to probe. Default: 1,10,30.")
    parser.add_argument("--trial-ids", default="", help="Comma-separated st1pl trial_id values to probe.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of selected trials to evaluate.")
    parser.add_argument("--device", default="cpu", help="Torch device for the diagnostic. Default: cpu.")
    args = parser.parse_args()

    rows, summary = run_probe(args)
    paths = write_outputs(rows, summary, Path(args.outdir))
    print("Probe A complete.")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
