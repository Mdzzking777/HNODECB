"""Run AFM05 st1pl-compatible epoch-0 preparation for valid trials.

This script evaluates the same stage1-entry preparation used by AFM05
prest2/st2l after the st1pl endpoint:

1. load each viable st1pl trial;
2. replace the x3 normalizer by the saved st1pl x3_pred statistics;
3. rebuild the neutral x3 AGU support/grid;
4. re-initialize g_NN from the gain reference;
5. evaluate the stage2/prest2 loss once without any optimizer step.

All outputs are written next to this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from contextlib import redirect_stderr, redirect_stdout
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM05.stage1pluslight.result_validation import validate_complete_stage1plus_payload
from AFM05.stage2light.config import default_config
from AFM05.stage2light.data import prepare_data
from AFM05.stage2light.kan_backend import KANForceModule, initial_grid_support_from_raw_inputs, initial_grid_support_to_meta
from AFM05.stage2light.losses import evaluate_split
from AFM05.stage2light.rollout import LearnableMechModule, _afm05_known_fields
from AFM05.stage2light.train import (
    _make_grid_update_inputs,
    _metrics_row,
    _observable_grid_inputs_from_ode,
    _prepared_with_initial_x3_from_warmstart,
    _torch_dtype,
    _window_to_torch,
)


CSV_NAME = "afm05_st1pl_epoch0_preparation_best_mech_winners.csv"
SUMMARY_JSON_NAME = "afm05_st1pl_epoch0_preparation_summary.json"
SUMMARY_TXT_NAME = "afm05_st1pl_epoch0_preparation_summary.txt"
TOP200_TXT_NAME = "afm05_st1pl_epoch0_preparation_rank_top200.txt"
TOP200_JSON_NAME = "afm05_st1pl_epoch0_preparation_rank_top200.json"
BEST_TXT_NAME = "afm05_st1pl_epoch0_preparation_mechanistic_winners_best_behavior.txt"
BEST_JSON_NAME = "afm05_st1pl_epoch0_preparation_mechanistic_winners_best_behavior.json"
RERANKED_CSV_NAME = "afm05_st1pl_epoch0_preparation_reranked_1000_trials.csv"
RERANKED_TXT_NAME = "afm05_st1pl_epoch0_preparation_reranked_1000_trials.txt"
DONE_JSON_NAME = "afm05_st1pl_epoch0_preparation_done.json"


FIELDNAMES = [
    "epoch0_rank",
    "st1pl_rank",
    "trial_id",
    "node_label",
    "ks_node_idx",
    "cs_node_idx",
    "nn_seed_bank_idx",
    "nn_init_seed",
    "ks",
    "cs",
    "st1pl_loss",
    "st1pl_train_loss",
    "st1pl_val_loss",
    "epoch0_train_loss",
    "epoch0_val_loss",
    "epoch0_ratio_to_st1pl_train",
    "epoch0_state",
    "epoch0_x1_state",
    "epoch0_x2_state",
    "epoch0_x2dot",
    "epoch0_x3_range",
    "epoch0_fts_range",
    "epoch0_x3_range_amp",
    "epoch0_fts_range_amp",
    "epoch0_x1_rec",
    "epoch0_x2_rec",
    "epoch0_x2dot_rec",
    "epoch0_val_state",
    "epoch0_val_x1_state",
    "epoch0_val_x2_state",
    "epoch0_val_x2dot",
    "epoch0_val_x3_range",
    "epoch0_val_fts_range",
    "epoch0_gain_before_agu",
    "epoch0_gain_after_agu",
    "epoch0_x3_mean",
    "epoch0_x3_scale",
    "epoch0_x3_support_min",
    "epoch0_x3_support_max",
    "epoch0_initial_grid_x1_min",
    "epoch0_initial_grid_x1_max",
    "epoch0_initial_grid_x2_min",
    "epoch0_initial_grid_x2_max",
    "epoch0_initial_grid_x3_min",
    "epoch0_initial_grid_x3_max",
    "epoch0_agu_samples",
    "status",
    "error",
    "elapsed_s",
]


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected stage1 payload type: {type(payload)!r}")
    validate_complete_stage1plus_payload(payload, path=path, require_merged=True)
    return payload


def _record_loss(record: dict[str, Any]) -> float:
    loss = _finite_float(record.get("train_loss", record.get("loss", float("inf"))), float("inf"))
    return loss if math.isfinite(loss) else float("inf")


def _record_trial_id(record: dict[str, Any]) -> int:
    params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
    for key in ("trial_id", "id"):
        value = params.get(key, record.get(key))
        try:
            tid = int(value)
        except Exception:
            continue
        if tid > 0:
            return tid
    return 0


def _node_key(record: dict[str, Any]) -> tuple[int, int, float, float, str]:
    params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
    ks_idx = int(params.get("ks_node_idx", -1))
    cs_idx = int(params.get("cs_node_idx", -1))
    ks = _finite_float(record.get("ks_hat", params.get("ks0", float("nan"))))
    cs = _finite_float(record.get("cs_hat", params.get("cs0", float("nan"))))
    label = str(params.get("node_label", f"node_{ks_idx}*{cs_idx}"))
    return ks_idx, cs_idx, ks, cs, label


def _valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [r for r in records if isinstance(r, dict) and bool(r.get("is_viable", False))]
    return sorted(valid, key=lambda rec: (_record_loss(rec), _record_trial_id(rec)))


def _best_mech_winner_records(valid_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one st1pl-best valid trial per mechanistic grid point."""

    best_by_node: dict[tuple[int, int], dict[str, Any]] = {}
    for rec in valid_records:
        ks_idx, cs_idx, _ks, _cs, _label = _node_key(rec)
        key = (ks_idx, cs_idx)
        if key not in best_by_node:
            best_by_node[key] = rec
            continue
        prev = best_by_node[key]
        if (_record_loss(rec), _record_trial_id(rec)) < (_record_loss(prev), _record_trial_id(prev)):
            best_by_node[key] = rec
    return sorted(best_by_node.values(), key=lambda rec: (_record_loss(rec), _record_trial_id(rec)))


def _load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _done_trial_ids(rows: list[dict[str, str]]) -> set[int]:
    done: set[int] = set()
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            done.add(int(row.get("trial_id", "0")))
        except Exception:
            pass
    return done


def _row_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    return _finite_float(row.get(key, default), default)


def _row_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def _build_model_and_eval(
    *,
    cfg,
    prepared_base,
    split,
    tensors: dict[str, torch.Tensor],
    record: dict[str, Any],
    eval_val: bool,
) -> dict[str, Any]:
    params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
    prepared_before = prepared_base
    prepared, x3_norm_support, x3_source = _prepared_with_initial_x3_from_warmstart(
        prepared_base,
        params,
    )
    if x3_norm_support is None:
        raise RuntimeError("missing_or_invalid_saved_st1pl_x3_entry_meta")

    observable_grid_inputs = _observable_grid_inputs_from_ode(
        ode_train=tensors["ode_train"],
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        x3_norm_support=x3_norm_support,
    )
    initial_grid_support = initial_grid_support_from_raw_inputs(
        observable_grid_inputs,
        prepared.state_mean,
        prepared.state_scale,
    )
    initial_grid_support_meta = initial_grid_support_to_meta(
        initial_grid_support,
        source=f"epoch0_preparation_observed_x1x2_neutral_x3__{x3_source}",
    )
    known_fields = _afm05_known_fields(prepared.known_pars)
    model_seed = int(params.get("nn_init_seed", record.get("seed", cfg.seed)))
    grid = int(params.get("kan_grid", cfg.grid))
    spline_k = int(params.get("kan_spline_k", cfg.spline_k))
    base_fun = str(params.get("kan_base_fun", cfg.base_fun))
    model = KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=model_seed,
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
        soft_mask_enabled=cfg.soft_mask_enabled,
        soft_mask_trainable=cfg.soft_mask_trainable,
        soft_mask_s0_a0=cfg.soft_mask_s0_a0,
        soft_mask_s0_min_a0=cfg.soft_mask_s0_min_a0,
        soft_mask_s0_max_a0=cfg.soft_mask_s0_max_a0,
        soft_mask_alpha_a0=cfg.soft_mask_alpha_a0,
        soft_mask_alpha_min_a0=cfg.soft_mask_alpha_min_a0,
        soft_mask_alpha_max_a0=cfg.soft_mask_alpha_max_a0,
        device=cfg.device,
        dtype=_torch_dtype(cfg.dtype),
    ).to(cfg.device)
    mech_module = LearnableMechModule(
        ks_init=float(record.get("ks_hat", params.get("ks0"))),
        cs_init=float(record.get("cs_hat", params.get("cs0"))),
        ks_bounds=(cfg.ks_lo, cfg.ks_hi),
        cs_bounds=(cfg.cs_lo, cfg.cs_hi),
        dtype=_torch_dtype(cfg.dtype),
        device=cfg.device,
        parameterization=cfg.mech_parameterization,
    ).to(cfg.device)

    train_states = tensors["ode_train"].transpose(0, 1)
    train_gain_force_reference = tensors["train_gain_force_reference"]
    gain_before = float(model.initialize_gain_from_reference(train_states, train_gain_force_reference))
    grid_meta: dict[str, float] = {"total_samples": float("nan")}
    if bool(cfg.adaptive_grid_enabled):
        grid_inputs, grid_meta = _make_grid_update_inputs(base_inputs=observable_grid_inputs)
        model.update_grid_from_normalized_inputs(grid_inputs)
    gain_after = float(model.initialize_gain_from_reference(train_states, train_gain_force_reference))

    train_total, train_parts, _train_traj = evaluate_split(
        force_module=model,
        known_pars=prepared.known_pars,
        mech_module=mech_module,
        ode_true=tensors["ode_full"],
        x2dot_true=tensors["x2dot_full"],
        contact_mask=tensors["contact_full"],
        times=tensors["times_full"],
        ode_method=cfg.ode_method,
        ode_rtol=cfg.ode_rtol,
        ode_atol=cfg.ode_atol,
        loss_indices=tensors["train_idx"],
    )
    train_metrics = _metrics_row(train_total, train_parts)
    if eval_val:
        val_total, val_parts, _val_traj = evaluate_split(
            force_module=model,
            known_pars=prepared.known_pars,
            mech_module=mech_module,
            ode_true=tensors["ode_full"],
            x2dot_true=tensors["x2dot_full"],
            contact_mask=tensors["contact_full"],
            times=tensors["times_full"],
            ode_method=cfg.ode_method,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            loss_indices=tensors["val_idx"],
        )
        val_metrics = _metrics_row(val_total, val_parts)
    else:
        val_metrics = {key: float("nan") for key in train_metrics}
    mean_before = np.asarray(prepared_before.state_mean, dtype=float)
    scale_before = np.asarray(prepared_before.state_scale, dtype=float)
    mean_after = np.asarray(prepared.state_mean, dtype=float)
    scale_after = np.asarray(prepared.state_scale, dtype=float)
    return {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "gain_before": gain_before,
        "gain_after": gain_after,
        "x3_mean": float(mean_after[2]),
        "x3_scale": float(scale_after[2]),
        "prior_x3_mean": float(mean_before[2]),
        "prior_x3_scale": float(scale_before[2]),
        "x3_support_min": float(x3_norm_support[0]),
        "x3_support_max": float(x3_norm_support[1]),
        "initial_grid_support": initial_grid_support,
        "initial_grid_support_meta": initial_grid_support_meta,
        "grid_meta": grid_meta,
        "x3_source": x3_source,
    }


def _append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def _format_csv_line(values: list[Any]) -> str:
    return ",".join(str(v) for v in values)


def _summarize(
    *,
    rows: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    source: Path,
    outdir: Path,
    elapsed_s: float,
    cfg,
) -> None:
    ok_rows = [r for r in rows if str(r.get("status", "")) == "ok" and math.isfinite(_row_float(r, "epoch0_train_loss"))]
    ok_rows.sort(key=lambda r: (_row_float(r, "epoch0_train_loss", float("inf")), _row_int(r, "trial_id")))
    for idx, row in enumerate(ok_rows, start=1):
        row["epoch0_rank"] = idx

    rank_by_trial = {_row_int(row, "trial_id"): int(row["epoch0_rank"]) for row in ok_rows}
    for row in rows:
        tid = _row_int(row, "trial_id")
        if tid in rank_by_trial:
            row["epoch0_rank"] = rank_by_trial[tid]
    with (outdir / CSV_NAME).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})

    top200 = ok_rows[:200]
    total_by_node: dict[tuple[int, int], int] = defaultdict(int)
    for rec in all_records:
        params = rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}
        key = (int(params.get("ks_node_idx", -1)), int(params.get("cs_node_idx", -1)))
        total_by_node[key] += 1

    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        groups[(_row_int(row, "ks_node_idx", -1), _row_int(row, "cs_node_idx", -1))].append(row)

    mech_rows = []
    for key, group in groups.items():
        group_sorted = sorted(group, key=lambda r: (_row_float(r, "epoch0_train_loss", float("inf")), _row_int(r, "trial_id")))
        best = group_sorted[0]
        losses = np.asarray([_row_float(r, "epoch0_train_loss") for r in group_sorted], dtype=float)
        mean_loss = float(np.mean(losses)) if losses.size else float("nan")
        total_count = int(total_by_node.get(key, len(group_sorted)))
        viable_count = len(group_sorted)
        mech_rows.append(
            {
                "node_label": str(best.get("node_label", f"node_{key[0]}*{key[1]}")),
                "ks_node_idx": key[0],
                "cs_node_idx": key[1],
                "count": total_count,
                "viable_count": viable_count,
                "viable_rate": float(viable_count / max(total_count, 1)),
                "mean_loss_viable": mean_loss,
                "best_loss": _row_float(best, "epoch0_train_loss"),
                "best_trial_id": _row_int(best, "trial_id"),
                "best_st1pl_rank": _row_int(best, "st1pl_rank"),
                "best_nn_seed_bank_idx": _row_int(best, "nn_seed_bank_idx"),
                "best_nn_init_seed": _row_int(best, "nn_init_seed"),
                "ks": _row_float(best, "ks"),
                "cs": _row_float(best, "cs"),
            }
        )

    best_behavior = sorted(mech_rows, key=lambda r: (float(r["best_loss"]), int(r["ks_node_idx"]), int(r["cs_node_idx"])))
    for idx, row in enumerate(best_behavior, start=1):
        row["rank"] = idx

    top_payload = [_json_ready(row) for row in top200]
    (outdir / TOP200_JSON_NAME).write_text(json.dumps(top_payload, indent=2), encoding="utf-8")
    (outdir / BEST_JSON_NAME).write_text(json.dumps(_json_ready(best_behavior), indent=2), encoding="utf-8")

    logs_dir = outdir if outdir.name.lower() == "logs" else outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rerank_fields = [
        "epoch0_rank",
        "st1pl_rank",
        "trial_id",
        "node_label",
        "ks_node_idx",
        "cs_node_idx",
        "nn_seed_bank_idx",
        "nn_init_seed",
        "ks",
        "cs",
        "st1pl_train_loss",
        "epoch0_train_loss",
        "epoch0_ratio_to_st1pl_train",
        "epoch0_gain_after_agu",
        "epoch0_x3_mean",
        "epoch0_x3_scale",
        "epoch0_x3_support_min",
        "epoch0_x3_support_max",
    ]
    with (logs_dir / RERANKED_CSV_NAME).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rerank_fields, extrasaction="ignore")
        writer.writeheader()
        for row in ok_rows:
            writer.writerow({key: row.get(key, "") for key in rerank_fields})
    with (logs_dir / RERANKED_TXT_NAME).open("w", encoding="utf-8") as fh:
        fh.write("AFM05 st1pl epoch-0 preparation reranked 1000 best-mech-winner trials\n")
        fh.write("ranking_loss=epoch0_train_loss\n")
        fh.write(f"source={source}\n")
        fh.write("selection=one st1pl-best viable trial per mechanistic node\n")
        fh.write("stage2_entry=no preopt reproduce rollout; x3 normalizer/support from saved st1pl rs_x3_pred metadata\n")
        fh.write(",".join(rerank_fields) + "\n")
        for row in ok_rows:
            fh.write(",".join(str(row.get(key, "")) for key in rerank_fields) + "\n")

    def write_mech_report(path: Path, title: str, data: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"AFM05 stage1pluslight epoch-0 preparation mechanistic winner report ({title})\n")
            fh.write("ranking_loss=epoch0_train_loss\n")
            fh.write("best_behavior: rank by each mechanical node's best selected epoch0 train loss.\n")
            fh.write(f"source={source}\n")
            fh.write(f"stage2_entry=no preopt reproduce rollout; x3 normalizer/support from saved st1pl rs_x3_pred metadata\n")
            fh.write(f"ks_bounds=({cfg.ks_lo}, {cfg.ks_hi}) cs_bounds=({cfg.cs_lo}, {cfg.cs_hi})\n")
            fh.write(
                "rank,node_label,count,viable_count,viable_rate,mean_loss_viable,best_loss,"
                "best_trial_id,best_st1pl_rank,best_nn_seed_bank_idx,best_nn_init_seed,ks,cs\n"
            )
            for row in data:
                fh.write(
                    _format_csv_line(
                        [
                            row["rank"],
                            row["node_label"],
                            row["count"],
                            row["viable_count"],
                            f"{row['viable_rate']:.12e}",
                            f"{row['mean_loss_viable']:.12e}",
                            f"{row['best_loss']:.12e}",
                            row["best_trial_id"],
                            row["best_st1pl_rank"],
                            row["best_nn_seed_bank_idx"],
                            row["best_nn_init_seed"],
                            f"{row['ks']:.12e}",
                            f"{row['cs']:.12e}",
                        ]
                    )
                    + "\n"
                )

    write_mech_report(outdir / BEST_TXT_NAME, "best_behavior", best_behavior)

    with (outdir / TOP200_TXT_NAME).open("w", encoding="utf-8") as fh:
        fh.write("AFM05 stage1pluslight epoch-0 preparation top 200 rank report\n")
        fh.write("ranking_loss=epoch0_train_loss\n")
        fh.write(f"source={source}\n")
        fh.write("stage2_entry=no preopt reproduce rollout; x3 normalizer/support from saved st1pl rs_x3_pred metadata\n")
        fh.write(
            "epoch0_rank,st1pl_rank,trial_id,node_label,ks_node_idx,cs_node_idx,"
            "nn_seed_bank_idx,nn_init_seed,ks,cs,st1pl_train_loss,epoch0_train_loss,"
            "epoch0_val_loss,ratio_to_st1pl,gain_after_agu,x3_mean,x3_scale,x3_support_min,x3_support_max\n"
        )
        for row in top200:
            fh.write(
                _format_csv_line(
                    [
                        row.get("epoch0_rank"),
                        row.get("st1pl_rank"),
                        row.get("trial_id"),
                        row.get("node_label"),
                        row.get("ks_node_idx"),
                        row.get("cs_node_idx"),
                        row.get("nn_seed_bank_idx"),
                        row.get("nn_init_seed"),
                        f"{_row_float(row, 'ks'):.12e}",
                        f"{_row_float(row, 'cs'):.12e}",
                        f"{_row_float(row, 'st1pl_train_loss'):.12e}",
                        f"{_row_float(row, 'epoch0_train_loss'):.12e}",
                        f"{_row_float(row, 'epoch0_val_loss'):.12e}",
                        f"{_row_float(row, 'epoch0_ratio_to_st1pl_train'):.12e}",
                        f"{_row_float(row, 'epoch0_gain_after_agu'):.12e}",
                        f"{_row_float(row, 'epoch0_x3_mean'):.12e}",
                        f"{_row_float(row, 'epoch0_x3_scale'):.12e}",
                        f"{_row_float(row, 'epoch0_x3_support_min'):.12e}",
                        f"{_row_float(row, 'epoch0_x3_support_max'):.12e}",
                    ]
                )
                + "\n"
            )

    summary = {
        "source": str(source),
        "output_dir": str(outdir),
        "csv": str(outdir / CSV_NAME),
        "total_stage1_records": len(all_records),
        "epoch0_ok_records": len(ok_rows),
        "epoch0_failed_or_pending_records": len(rows) - len(ok_rows),
        "top200_count": len(top200),
        "mechanistic_node_count": len(mech_rows),
        "elapsed_s_for_this_run": elapsed_s,
        "stage2_entry": "no preopt reproduce rollout; saved st1pl rs_x3_pred metadata -> x3 normalizer/support -> AGU -> gain -> loss",
        "selection_note": (
            "Default selection is one st1pl-best viable trial per mechanistic node "
            "(the 20x50 lower-envelope/best-mech-winner layer). In this mode, "
            "epoch0 preparation reports rank top 200 and best_behavior only; "
            "mean_behavior is intentionally not evaluated here."
        ),
        "config": {
            "window_mode": cfg.window_mode,
            "pixel_tag": cfg.pixel_tag,
            "arch_window_us": cfg.arch_window_us,
            "window_sample_stride": cfg.window_sample_stride,
            "val_stride": cfg.val_stride,
            "val_offset": cfg.val_offset,
            "width": cfg.width,
            "grid": cfg.grid,
            "spline_k": cfg.spline_k,
            "base_fun": cfg.base_fun,
            "noise_scale": cfg.noise_scale,
            "grid_eps": cfg.grid_eps,
            "grid_range": [cfg.grid_range_lo, cfg.grid_range_hi],
            "adaptive_grid_enabled": cfg.adaptive_grid_enabled,
            "ode_method": cfg.ode_method,
            "ode_rtol": cfg.ode_rtol,
            "ode_atol": cfg.ode_atol,
        },
        "best_epoch0_trial": _json_ready(top200[0]) if top200 else None,
        "best_mech_winner": _json_ready(best_behavior[0]) if best_behavior else None,
    }
    (outdir / SUMMARY_JSON_NAME).write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    with (outdir / SUMMARY_TXT_NAME).open("w", encoding="utf-8") as fh:
        fh.write("AFM05 st1pl epoch-0 preparation summary\n")
        fh.write("======================================\n")
        fh.write(f"source: {source}\n")
        fh.write(f"output_dir: {outdir}\n")
        fh.write(f"stage2_entry: {summary['stage2_entry']}\n")
        fh.write(f"selection_note: {summary['selection_note']}\n")
        fh.write(f"total_stage1_records: {summary['total_stage1_records']}\n")
        fh.write(f"epoch0_ok_records: {summary['epoch0_ok_records']}\n")
        fh.write(f"epoch0_failed_or_pending_records_in_csv: {summary['epoch0_failed_or_pending_records']}\n")
        fh.write(f"mechanistic_node_count: {summary['mechanistic_node_count']}\n")
        fh.write(f"elapsed_s_for_this_run: {elapsed_s:.3f}\n\n")
        if top200:
            b = top200[0]
            fh.write("Best epoch0-ranked trial:\n")
            fh.write(
                f"  epoch0_rank={b.get('epoch0_rank')} st1pl_rank={b.get('st1pl_rank')} "
                f"trial_id={b.get('trial_id')} node={b.get('node_label')} "
                f"ks={_row_float(b, 'ks'):.12e} cs={_row_float(b, 'cs'):.12e} "
                f"st1pl_train_loss={_row_float(b, 'st1pl_train_loss'):.12e} "
                f"epoch0_train_loss={_row_float(b, 'epoch0_train_loss'):.12e}\n\n"
            )
        if best_behavior:
            b = best_behavior[0]
            fh.write("Best-behavior mechanistic winner:\n")
            fh.write(
                f"  rank=1 node={b['node_label']} ks={b['ks']:.12e} cs={b['cs']:.12e} "
                f"best_loss={b['best_loss']:.12e} best_trial_id={b['best_trial_id']} "
                f"best_st1pl_rank={b['best_st1pl_rank']}\n\n"
            )
        fh.write("Generated files:\n")
        for name in (
            CSV_NAME,
            TOP200_TXT_NAME,
            TOP200_JSON_NAME,
            BEST_TXT_NAME,
            BEST_JSON_NAME,
            SUMMARY_JSON_NAME,
        ):
            fh.write(f"  - {outdir / name}\n")
        fh.write(f"  - {logs_dir / RERANKED_CSV_NAME}\n")
        fh.write(f"  - {logs_dir / RERANKED_TXT_NAME}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-pkl",
        type=Path,
        default=REPO_ROOT / "AFM05" / "stage1pluslight" / "results_afm" / "afm_param_stage1pluslight_05.pkl",
        help="Merged AFM05 st1pl result pkl.",
    )
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR, help="Output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Limit valid trials for smoke tests; 0 means all.")
    parser.add_argument(
        "--all-valid",
        action="store_true",
        help="Run every viable seed-level st1pl trial. Default is only one best trial per mech node.",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        action="append",
        default=[],
        help="Run only selected trial_id(s). Can be repeated.",
    )
    parser.add_argument("--shard-index", type=int, default=1, help="1-based sequential shard index for segmented runs.")
    parser.add_argument(
        "--shard-count",
        type=int,
        default=6,
        help="Sequential shard count. Default is 6. Do not run shards in parallel against the same CSV.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already successful trial_ids in the output CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Remove previous outputs before running.")
    parser.add_argument("--eval-val", action="store_true", help="Also evaluate validation loss. Slower; ranking uses train loss.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional stdout/stderr log file.")
    parser.add_argument("--merge-only", action="store_true", help="Merge per-shard CSV files and regenerate final reports.")
    parser.add_argument("--merge-shard-root", type=Path, default=None, help="Directory containing shard_XX output folders.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N processed trials.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / CSV_NAME
    output_paths = [
        csv_path,
        outdir / SUMMARY_JSON_NAME,
        outdir / SUMMARY_TXT_NAME,
        outdir / TOP200_TXT_NAME,
        outdir / TOP200_JSON_NAME,
        outdir / BEST_TXT_NAME,
        outdir / BEST_JSON_NAME,
        (outdir if outdir.name.lower() == "logs" else outdir / "logs") / RERANKED_CSV_NAME,
        (outdir if outdir.name.lower() == "logs" else outdir / "logs") / RERANKED_TXT_NAME,
    ]
    if args.overwrite:
        for path in output_paths:
            if path.exists():
                path.unlink()

    cfg = default_config(REPO_ROOT)
    t0 = perf_counter()
    payload = _load_payload(args.stage1_pkl)
    records = [rec for rec in payload.get("trial_parameters", []) if isinstance(rec, dict)]
    if args.merge_only:
        if args.merge_shard_root is None:
            raise SystemExit("--merge-only requires --merge-shard-root")
        shard_root = args.merge_shard_root.resolve()
        shard_csvs = sorted(shard_root.glob(f"shard_*/{CSV_NAME}"))
        if not shard_csvs:
            raise SystemExit(f"no shard CSV files found under {shard_root}")
        merged_by_trial: dict[int, dict[str, Any]] = {}
        for shard_csv in shard_csvs:
            for row in _load_existing_rows(shard_csv):
                tid = _row_int(row, "trial_id")
                if tid <= 0:
                    continue
                if tid not in merged_by_trial:
                    merged_by_trial[tid] = dict(row)
                    continue
                old = merged_by_trial[tid]
                if str(old.get("status", "")) != "ok" and str(row.get("status", "")) == "ok":
                    merged_by_trial[tid] = dict(row)
        rows = list(merged_by_trial.values())
        _summarize(
            rows=rows,
            all_records=records,
            source=args.stage1_pkl.resolve(),
            outdir=outdir,
            elapsed_s=perf_counter() - t0,
            cfg=cfg,
        )
        print(f"merged_shard_root={shard_root}")
        print(f"merged_shard_csvs={len(shard_csvs)}")
        print(f"merged_rows={len(rows)}")
        print(f"summary={outdir / SUMMARY_TXT_NAME}")
        logs_dir = outdir if outdir.name.lower() == "logs" else outdir / "logs"
        print(f"reranked={logs_dir / RERANKED_CSV_NAME}")
        return

    dtype = _torch_dtype(cfg.dtype)
    prepared_base = prepare_data(cfg)
    split_index = max(0, int(cfg.train_window_index))
    if split_index >= len(prepared_base.splits):
        split_index = 0
    split = prepared_base.splits[split_index]
    tensors = _window_to_torch(split, dtype=dtype, device=cfg.device)
    tensors["train_gain_force_reference"] = torch.as_tensor(
        split.train_gain_force_reference,
        dtype=dtype,
        device=cfg.device,
    )

    valid_all = _valid_records(records)
    st1pl_rank_by_trial = {_record_trial_id(rec): idx for idx, rec in enumerate(valid_all, start=1)}
    if args.all_valid or args.trial_id:
        valid = valid_all
        selection_label = "all_valid_seed_level_trials"
    else:
        valid = _best_mech_winner_records(valid_all)
        selection_label = "best_mech_winners_one_per_mechanistic_node"
    if args.trial_id:
        selected_ids = set(int(v) for v in args.trial_id)
        valid = [rec for rec in valid if _record_trial_id(rec) in selected_ids]
    shard_count = max(1, int(args.shard_count))
    shard_index = min(max(1, int(args.shard_index)), shard_count)
    if shard_count > 1:
        valid = [rec for idx, rec in enumerate(valid) if (idx % shard_count) == (shard_index - 1)]
    if args.limit and args.limit > 0:
        valid = valid[: int(args.limit)]

    existing_rows = _load_existing_rows(csv_path) if args.resume else []
    done = _done_trial_ids(existing_rows) if args.resume else set()
    pending = [rec for rec in valid if _record_trial_id(rec) not in done]

    print(f"source={args.stage1_pkl}")
    print(
        f"selection={selection_label} valid_records_selected={len(valid)} "
        f"pending={len(pending)} resume_done={len(done)} shard={shard_index}/{shard_count}"
    )
    print(f"output_csv={csv_path}")

    processed = 0
    for rec in pending:
        processed += 1
        trial_start = perf_counter()
        params = rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}
        trial_id = _record_trial_id(rec)
        ks_idx, cs_idx, ks, cs, node_label = _node_key(rec)
        base_row: dict[str, Any] = {
            "epoch0_rank": "",
            "st1pl_rank": st1pl_rank_by_trial.get(trial_id, ""),
            "trial_id": trial_id,
            "node_label": node_label,
            "ks_node_idx": ks_idx,
            "cs_node_idx": cs_idx,
            "nn_seed_bank_idx": int(params.get("nn_seed_bank_idx", -1)),
            "nn_init_seed": int(params.get("nn_init_seed", rec.get("seed", -1))),
            "ks": ks,
            "cs": cs,
            "st1pl_loss": _finite_float(rec.get("loss")),
            "st1pl_train_loss": _finite_float(rec.get("train_loss", rec.get("loss"))),
            "st1pl_val_loss": _finite_float(rec.get("val_loss")),
        }
        try:
            with torch.no_grad():
                result = _build_model_and_eval(
                    cfg=cfg,
                    prepared_base=prepared_base,
                    split=split,
                    tensors=tensors,
                    record=rec,
                    eval_val=bool(args.eval_val),
                )
            train = result["train_metrics"]
            val = result["val_metrics"]
            grid = np.asarray(result["initial_grid_support"], dtype=float)
            st1_loss = _finite_float(base_row["st1pl_train_loss"], float("nan"))
            ep0_loss = float(train["loss"])
            ratio = ep0_loss / st1_loss if math.isfinite(st1_loss) and abs(st1_loss) > 0.0 else float("nan")
            row = {
                **base_row,
                "epoch0_train_loss": ep0_loss,
                "epoch0_val_loss": float(val["loss"]),
                "epoch0_ratio_to_st1pl_train": ratio,
                "epoch0_state": train["state"],
                "epoch0_x1_state": train["x1_state"],
                "epoch0_x2_state": train["x2_state"],
                "epoch0_x2dot": train["x2dot"],
                "epoch0_x3_range": train["x3_range"],
                "epoch0_fts_range": train["fts_range"],
                "epoch0_x3_range_amp": train["x3_range_amp"],
                "epoch0_fts_range_amp": train["fts_range_amp"],
                "epoch0_x1_rec": train["x1_rec"],
                "epoch0_x2_rec": train["x2_rec"],
                "epoch0_x2dot_rec": train["x2dot_rec"],
                "epoch0_val_state": val["state"],
                "epoch0_val_x1_state": val["x1_state"],
                "epoch0_val_x2_state": val["x2_state"],
                "epoch0_val_x2dot": val["x2dot"],
                "epoch0_val_x3_range": val["x3_range"],
                "epoch0_val_fts_range": val["fts_range"],
                "epoch0_gain_before_agu": result["gain_before"],
                "epoch0_gain_after_agu": result["gain_after"],
                "epoch0_x3_mean": result["x3_mean"],
                "epoch0_x3_scale": result["x3_scale"],
                "epoch0_x3_support_min": result["x3_support_min"],
                "epoch0_x3_support_max": result["x3_support_max"],
                "epoch0_initial_grid_x1_min": float(grid[0, 0]),
                "epoch0_initial_grid_x1_max": float(grid[0, 1]),
                "epoch0_initial_grid_x2_min": float(grid[1, 0]),
                "epoch0_initial_grid_x2_max": float(grid[1, 1]),
                "epoch0_initial_grid_x3_min": float(grid[2, 0]),
                "epoch0_initial_grid_x3_max": float(grid[2, 1]),
                "epoch0_agu_samples": float(result["grid_meta"].get("total_samples", float("nan"))),
                "status": "ok",
                "error": "",
                "elapsed_s": perf_counter() - trial_start,
            }
        except Exception as err:
            row = {
                **base_row,
                "status": "failed",
                "error": f"{type(err).__name__}: {err}",
                "elapsed_s": perf_counter() - trial_start,
            }
        _append_row(csv_path, row)
        if args.progress_every > 0 and (processed % args.progress_every == 0 or processed == len(pending)):
            elapsed = perf_counter() - t0
            print(f"processed={processed}/{len(pending)} elapsed_s={elapsed:.1f} last_trial_id={trial_id} status={row['status']}")

    all_rows = _load_existing_rows(csv_path)
    _summarize(
        rows=all_rows,
        all_records=records,
        source=args.stage1_pkl.resolve(),
        outdir=outdir,
        elapsed_s=perf_counter() - t0,
        cfg=cfg,
    )
    done_payload = {
        "status": "done",
        "outdir": str(outdir),
        "csv": str(csv_path),
        "rows": len(_load_existing_rows(csv_path)),
        "elapsed_s": perf_counter() - t0,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
    }
    (outdir / DONE_JSON_NAME).write_text(json.dumps(_json_ready(done_payload), indent=2), encoding="utf-8")
    print(f"summary={outdir / SUMMARY_TXT_NAME}")
    print(f"top200={outdir / TOP200_TXT_NAME}")
    print(f"best_behavior={outdir / BEST_TXT_NAME}")


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.log_file is None:
        run(parsed_args)
    else:
        parsed_args.log_file.parent.mkdir(parents=True, exist_ok=True)
        with parsed_args.log_file.open("w", encoding="utf-8") as log_fh:
            with redirect_stdout(log_fh), redirect_stderr(log_fh):
                run(parsed_args)
