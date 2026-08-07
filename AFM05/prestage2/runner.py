"""Runner helpers for AFM05 prestage2 (prest2)."""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from AFM05.rng_state import capture_rng_state
from AFM05.prestage2.config import Prestage2Config, default_config
from AFM05.stage2light.config import default_config as default_stage2light_config
from AFM05.stage2light.data import load_stage1_payload, prepare_data
from AFM05.stage1pluslight.result_validation import validate_complete_stage1plus_payload

LAYER_A = "layer_a"
LAYER_B = "layer_b"

_ST1PL_LOW_LOSS_SEPARATOR_X0 = -1.0
_ST1PL_LOW_LOSS_SEPARATOR_Y0 = -5.934295301786307
_ST1PL_LOW_LOSS_SEPARATOR_X1 = 0.5789473684210527
_ST1PL_LOW_LOSS_SEPARATOR_Y1 = -4.301029995663981

_STEP_RETRY_RE = re.compile(r"step-guard retry\s+(\d+)/(\w+)")
_STEP_ACCEPTED_RE = re.compile(r"accepted after\s+(\d+)\s+retry/reduction")
_STEP_REJECTED_RE = re.compile(r"rejected update after\s+(\d+)\s+retries")
_EPOCH_RETRY_RE = re.compile(r"retry epoch\s+(\d+)\s+attempt\s+(\d+)/(\d+)")
_ADAM_STATE_RESET_RE = re.compile(
    r"adam state reset:\s+trigger_retries=(\d+)\s+accepted=(YES|NO)\s+post_reset_retries=(\d+)",
    re.IGNORECASE,
)


def _timestamp_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_line(log, message: str, *, echo: bool = False) -> None:
    line = f"[{_timestamp_human()}] {message}"
    log.write(line + "\n")
    log.flush()
    if echo:
        print(line, flush=True)


def _canonical_layer_name(layer_name: str) -> str:
    value = str(layer_name).strip().lower()
    if value in ("a", "layera", "layer_a", "newrank_a", "top_mech_winner_a", "top_mech_winners_a"):
        return LAYER_A
    if value in ("b", "layerb", "layer_b", "candidate", "candidate_b", "candidates_b"):
        return LAYER_B
    raise ValueError(f"Unsupported prest2 layer name: {layer_name!r}")


def _layer_token(layer_name: str) -> str:
    return "layerA" if _canonical_layer_name(layer_name) == LAYER_A else "layerB"


def _rank_label_name(layer_name: str) -> str:
    return "top mech winner A" if _canonical_layer_name(layer_name) == LAYER_A else "candidate B"


def _canonical_low_loss_zone(value: str) -> str:
    zone = str(value).strip().lower()
    if zone in ("", "all", "none", "any"):
        return "all"
    if zone in ("corner", "corner_zone", "corner-zone"):
        return "corner"
    if zone in ("noncorner", "non_corner", "non-corner", "not_corner", "not-corner"):
        return "noncorner"
    if zone in ("valley", "valley_zone", "valley-zone"):
        return "valley"
    raise ValueError(f"Unsupported AFM05 prest2 Layer A zone filter: {value!r}")


def _low_loss_zone_from_log_coords(log_ks: float, log_cs: float) -> str:
    """Classify AFM05 st1pl low-loss regions using the visualization separator.

    The separator is the same line drawn in the AFM05 st1pl best-seed envelope
    visualization.  Points above the line are the corner zone; points below it
    are the valley zone.
    """

    slope = (_ST1PL_LOW_LOSS_SEPARATOR_Y1 - _ST1PL_LOW_LOSS_SEPARATOR_Y0) / (
        _ST1PL_LOW_LOSS_SEPARATOR_X1 - _ST1PL_LOW_LOSS_SEPARATOR_X0
    )
    separator_y = _ST1PL_LOW_LOSS_SEPARATOR_Y0 + slope * (float(log_ks) - _ST1PL_LOW_LOSS_SEPARATOR_X0)
    return "corner" if float(log_cs) >= separator_y else "valley"


def _stage2light_prefix_schedule(*, stage2_adam_epochs: int, target_total_epochs: int) -> tuple[int, int]:
    """Project the formal st2l optimizer schedule onto the first N epochs.

    Layers decide only how many st2l epochs a trial is allowed to reach.  The
    optimizer phase of each epoch still belongs to st2l: epoch 1 follows st2l
    epoch 1, epoch 21 follows st2l epoch 21, and so on.
    """

    total = max(0, int(target_total_epochs))
    adam = min(max(0, int(stage2_adam_epochs)), total)
    lbfgs = max(0, total - adam)
    return adam, lbfgs


def make_driver_log_path(log_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"log2_05_prest2_local_driver_{stamp}.txt"


def make_shard_log_path(log_dir: Path, layer_name: str, shard_index: int) -> Path:
    token = _layer_token(layer_name)
    return log_dir / f"log2_05_prest2_local_{token}_p{shard_index}.txt"


def _save_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = dict(payload)
    payload.setdefault("rng_state", capture_rng_state())
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _is_stage2_w0_mode(window_mode: str) -> bool:
    mode = str(window_mode or "").strip().lower()
    return mode in ("w0", "stage2_w0", "stage2-w0", "first_contact", "first-contact")


def _expected_stage2_window_meta(cfg: Prestage2Config) -> dict[str, Any]:
    stage2_cfg = default_stage2light_config(cfg.repo_root)
    stage2_cfg = replace(stage2_cfg, auto_generate_dataset=bool(cfg.auto_generate_dataset))
    prepared = prepare_data(stage2_cfg)
    requested = int(getattr(stage2_cfg, "train_window_index", 0))
    if requested > 0:
        split = prepared.splits[requested - 1]
    else:
        preferred_role = "first_contact" if _is_stage2_w0_mode(stage2_cfg.window_mode) else "middle"
        role_lookup = {str(split.role).strip().lower(): split for split in prepared.splits}
        split = role_lookup.get(preferred_role, prepared.splits[0])
    return {
        "role": str(split.role),
        "label": str(split.label),
        "start_idx": int(split.start_idx),
        "stop_idx": int(split.stop_idx),
        "t_start": float(split.t_start),
        "t_stop": float(split.t_stop),
    }


def _float_meta_match(old: Any, cur: Any) -> bool:
    try:
        old_f = float(old)
        cur_f = float(cur)
    except Exception:
        return False
    return abs(old_f - cur_f) <= max(1.0e-15, 1.0e-9 * max(abs(old_f), abs(cur_f), 1.0))


def _validate_layer_b_checkpoint_window(checkpoint_path: Path, expected_meta: dict[str, Any]) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Layer B seamless resume checkpoint payload is not a dict: {checkpoint_path}")
    meta = payload.get("window_meta")
    if not isinstance(meta, dict):
        raise RuntimeError(f"Layer B seamless resume checkpoint is missing window_meta: {checkpoint_path}")
    for key in ("role", "label"):
        if str(meta.get(key, "")).strip() != str(expected_meta[key]).strip():
            raise RuntimeError(
                "Layer B seamless resume checkpoint window identity mismatch: "
                f"{key} checkpoint={meta.get(key)!r} current={expected_meta[key]!r} | checkpoint={checkpoint_path}"
            )
    for key in ("start_idx", "stop_idx"):
        try:
            old_value = int(meta.get(key))
        except Exception as err:
            raise RuntimeError(f"Layer B checkpoint has invalid {key}: {meta.get(key)!r}") from err
        if old_value != int(expected_meta[key]):
            raise RuntimeError(
                "Layer B seamless resume checkpoint window identity mismatch: "
                f"{key} checkpoint={old_value} current={expected_meta[key]} | checkpoint={checkpoint_path}"
            )
    for key in ("t_start", "t_stop"):
        if not _float_meta_match(meta.get(key), expected_meta[key]):
            raise RuntimeError(
                "Layer B seamless resume checkpoint window identity mismatch: "
                f"{key} checkpoint={meta.get(key)!r} current={expected_meta[key]!r} | checkpoint={checkpoint_path}"
            )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _candidate_loss(record: dict[str, Any], metric: str) -> float:
    raw = record.get(metric, record.get("final_train_loss", record.get("final_val_loss", float("inf"))))
    try:
        value = float(raw)
    except Exception:
        return float("inf")
    if not math.isfinite(value):
        return float("inf")
    return value


def _finite_float(value: Any, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        out = int(value)
    except Exception:
        return default
    return out if out > 0 else default


def _rank_candidates(records: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda rec: (
            _candidate_loss(rec, metric),
            int(rec.get("source_mech_winner", 10**9)),
            int(rec.get("source_top_mech_winner_a", rec.get("source_newrank_a", 10**9))),
        ),
    )


def _mw_seed_rank_label(record: dict[str, Any]) -> str:
    mw = int(record.get("source_mech_winner", 0))
    seed = int(record.get("source_stage1_best_seedbank", 0))
    rank = int(record.get("source_stage1_rank", 0))
    if mw > 0 and seed > 0 and rank > 0:
        return f"mech winner {mw} +NN seed {seed} (rank {rank})"
    if mw > 0 and seed > 0:
        return f"mech winner {mw} +NN seed {seed}"
    if mw > 0 and rank > 0:
        return f"mech winner {mw} (rank {rank})"
    if mw > 0:
        return f"mech winner {mw}"
    return "mech winner 0"


def _write_stage1_warmstart_record(
    *,
    cfg: Prestage2Config,
    row: dict[str, Any],
    record: dict[str, Any],
) -> Path:
    params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
    trial_id = int(params.get("trial_id", row.get("source_stage1_trial_id", 0)))
    mech_winner = int(row.get("source_mech_winner", 0))
    if trial_id <= 0:
        raise RuntimeError(f"Cannot materialize stage1 warmstart record without trial_id: {row}")
    out_dir = cfg.result_dir / "warmstart_records"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stage1_mw{mech_winner:03d}_trial{trial_id}.pkl"
    payload = {
        "schema": "afm05_stage1_warmstart_record_v1",
        "source_stage1_input_path": str(cfg.stage1_input_path.resolve()),
        "source_mech_winner": int(mech_winner),
        "source_stage1_rank": int(row.get("source_stage1_rank", 0)),
        "source_stage1_trial_id": int(trial_id),
        "source_stage1_best_loss": float(row.get("best_loss", float("inf"))),
        "source_stage1_best_seedbank": int(row.get("source_stage1_best_seedbank", 0)),
        "source_stage1_best_initseed": int(row.get("source_stage1_best_initseed", 0)),
        "record": record,
    }
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def _build_mech_winner_items(stage1_payload: dict[str, Any], cfg: Prestage2Config) -> list[dict[str, Any]]:
    validate_complete_stage1plus_payload(stage1_payload, path=cfg.stage1_input_path)
    all_records = [rec for rec in stage1_payload.get("trial_parameters", []) if isinstance(rec, dict)]
    rank_by_trial_id: dict[int, int] = {}
    for idx, rec in enumerate(
        sorted(
            all_records,
            key=lambda item: (
                not bool(item.get("is_viable", False)),
                _finite_float(item.get("loss")),
            ),
        ),
        start=1,
    ):
        params = rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}
        trial_id = int(params.get("trial_id", 0))
        if trial_id > 0:
            rank_by_trial_id[trial_id] = int(idx)

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for rec in stage1_payload.get("trial_parameters", []):
        if not isinstance(rec, dict):
            continue
        params = rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}
        ks_node = _positive_int(params.get("ks_node_idx"))
        cs_node = _positive_int(params.get("cs_node_idx"))
        if ks_node <= 0 or cs_node <= 0:
            continue
        groups.setdefault((ks_node, cs_node), []).append(rec)

    if not groups:
        raise RuntimeError(f"No mechanical groups found in stage1pluslight payload: {cfg.stage1_input_path}")

    rows: list[dict[str, Any]] = []
    for (ks_node, cs_node), records in groups.items():
        finite_records = [rec for rec in records if math.isfinite(_finite_float(rec.get("loss")))]
        viable_records = [
            rec for rec in records if bool(rec.get("is_viable", False)) and math.isfinite(_finite_float(rec.get("loss")))
        ]
        losses = [_finite_float(rec.get("loss")) for rec in viable_records]
        best_pool = viable_records or finite_records
        best = min(
            best_pool,
            key=lambda rec: (
                _finite_float(rec.get("loss")),
                _positive_int((rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}).get("trial_id")),
            ),
        ) if best_pool else min(
            records,
            key=lambda rec: _positive_int((rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}).get("trial_id")),
        )
        first = min(
            records,
            key=lambda rec: _positive_int((rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}).get("trial_id")),
        )
        first_params = first.get("params", {}) if isinstance(first.get("params"), dict) else {}
        best_params = best.get("params", {}) if isinstance(best.get("params"), dict) else {}
        best_trial_id = int(best_params.get("trial_id", 0))
        ks_value = float(first.get("ks_hat", first_params.get("ks0", float("nan"))))
        cs_value = float(first.get("cs_hat", first_params.get("cs0", float("nan"))))
        low_loss_zone = (
            _low_loss_zone_from_log_coords(math.log10(ks_value), math.log10(cs_value))
            if math.isfinite(ks_value) and ks_value > 0.0 and math.isfinite(cs_value) and cs_value > 0.0
            else "unknown"
        )
        n_total = len(records)
        n_viable = len(viable_records)
        mean_loss_viable = sum(losses) / len(losses) if losses else float("inf")
        rows.append(
            {
                "node_label": f"node_{ks_node}*{cs_node}",
                "ks_node_idx": int(ks_node),
                "cs_node_idx": int(cs_node),
                "ks": ks_value,
                "cs": cs_value,
                "low_loss_zone": low_loss_zone,
                "n_total": int(n_total),
                "viable_count": int(n_viable),
                "finite_count": int(len(finite_records)),
                "viable_rate": float(n_viable / n_total) if n_total else float("nan"),
                "mean_loss_viable": float(mean_loss_viable),
                "best_loss": float(best.get("loss", float("inf"))),
                "source_stage1_trial_id": int(best_trial_id),
                "source_stage1_rank": int(rank_by_trial_id.get(best_trial_id, 0)),
                "source_stage1_best_seedbank": int(best_params.get("nn_seed_bank_idx", 0)),
                "source_stage1_best_initseed": int(best_params.get("nn_init_seed", 0)),
                "_stage1_warmstart_record": best,
            }
        )

    zone_filter = _canonical_low_loss_zone(cfg.layer_a_zone_filter)
    if zone_filter in ("noncorner", "all"):
        rows.sort(
            key=lambda row: (
                _finite_float(row.get("best_loss")),
                int(row.get("ks_node_idx", 10**9)),
                int(row.get("cs_node_idx", 10**9)),
            )
        )
    else:
        rows.sort(
            key=lambda row: (
                _finite_float(row.get("mean_loss_viable")),
                -_finite_float(row.get("viable_rate"), default=0.0),
                _finite_float(row.get("best_loss")),
                int(row.get("ks_node_idx", 10**9)),
                int(row.get("cs_node_idx", 10**9)),
            )
        )

    for idx, row in enumerate(rows, start=1):
        row["source_mech_winner"] = int(idx)
        row["mech_winner_label"] = f"mech winner {idx}"

    candidate_rows = rows
    if zone_filter == "noncorner":
        candidate_rows = [row for row in rows if str(row.get("low_loss_zone", "")) != "corner"]
    elif zone_filter != "all":
        candidate_rows = [row for row in rows if str(row.get("low_loss_zone", "")) == zone_filter]

    if cfg.layer_a_top_candidates > len(candidate_rows):
        raise ValueError(
            f"Requested {cfg.layer_a_top_candidates} mech winners for Layer A, "
            f"but only {len(candidate_rows)} mechanical groups match zone_filter={zone_filter!r}"
        )
    selected_rows = candidate_rows[: int(cfg.layer_a_top_candidates)]
    for row in selected_rows:
        record = row.pop("_stage1_warmstart_record", None)
        if isinstance(record, dict):
            record_path = _write_stage1_warmstart_record(cfg=cfg, row=row, record=record)
            row["source_stage1_warmstart_record_path"] = str(record_path.resolve())
    return selected_rows


def _stage2light_child_log_path(candidate_root: Path) -> Path:
    return candidate_root / "logs" / "window_per_shard" / "log2_05_step2a_stage2light_local_p1.txt"


def _stage2light_log_has_hard_zero_step_stop(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        return any("lbfgs_hard_zero_step" in line for line in f)


def _stage2light_log_has_x3_refit_stop(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        return any(("x3 normalizer refit failed" in line) or ("x3_normalizer_refit_failed" in line) for line in f)


def _stage2light_payload_hard_zero_step_stop(payload: dict[str, Any]) -> bool:
    return str(payload.get("stop_kind", "")).strip() == "lbfgs_hard_zero_step_stop"


def _summarize_stage2light_retry_log(log_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "stage2light_log_path": str(log_path.resolve()),
        "stage2light_log_exists": bool(log_path.is_file()),
        "step_retry_lines": 0,
        "step_retry_max": 0,
        "step_retry_denoms": [],
        "step_accepted_after_count": 0,
        "step_accepted_after_max": 0,
        "step_rejected_count": 0,
        "step_rejected_after_max": 0,
        "epoch_retry_lines": 0,
        "epoch_retry_max_attempt": 0,
        "epoch_retry_max_denom": 0,
        "adam_state_reset_lines": 0,
        "adam_state_reset_accepted_count": 0,
        "adam_state_reset_trigger_max": 0,
        "adam_state_reset_post_max": 0,
    }
    if not log_path.is_file():
        return summary

    denoms: set[str] = set()
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            step_match = _STEP_RETRY_RE.search(line)
            if step_match:
                retry_idx = int(step_match.group(1))
                denoms.add(step_match.group(2))
                summary["step_retry_lines"] += 1
                summary["step_retry_max"] = max(int(summary["step_retry_max"]), retry_idx)

            accepted_match = _STEP_ACCEPTED_RE.search(line)
            if accepted_match:
                retry_count = int(accepted_match.group(1))
                summary["step_accepted_after_count"] += 1
                summary["step_accepted_after_max"] = max(int(summary["step_accepted_after_max"]), retry_count)

            rejected_match = _STEP_REJECTED_RE.search(line)
            if rejected_match:
                retry_count = int(rejected_match.group(1))
                summary["step_rejected_count"] += 1
                summary["step_rejected_after_max"] = max(int(summary["step_rejected_after_max"]), retry_count)

            epoch_match = _EPOCH_RETRY_RE.search(line)
            if epoch_match:
                attempt = int(epoch_match.group(2))
                denom = int(epoch_match.group(3))
                summary["epoch_retry_lines"] += 1
                summary["epoch_retry_max_attempt"] = max(int(summary["epoch_retry_max_attempt"]), attempt)
                summary["epoch_retry_max_denom"] = max(int(summary["epoch_retry_max_denom"]), denom)

            reset_match = _ADAM_STATE_RESET_RE.search(line)
            if reset_match:
                trigger_retries = int(reset_match.group(1))
                accepted = reset_match.group(2).upper() == "YES"
                post_retries = int(reset_match.group(3))
                summary["adam_state_reset_lines"] += 1
                summary["adam_state_reset_accepted_count"] += int(accepted)
                summary["adam_state_reset_trigger_max"] = max(
                    int(summary["adam_state_reset_trigger_max"]),
                    trigger_retries,
                )
                summary["adam_state_reset_post_max"] = max(
                    int(summary["adam_state_reset_post_max"]),
                    post_retries,
                )

    summary["step_retry_denoms"] = sorted(denoms, key=str)
    return summary


def _retry_summary_line(summary: dict[str, Any]) -> str:
    denoms = ",".join(str(x) for x in summary.get("step_retry_denoms", [])) or "none"
    return (
        f"stage2light_log={summary.get('stage2light_log_path', '')} | "
        f"exists={summary.get('stage2light_log_exists', False)} | "
        f"step_retry_lines={int(summary.get('step_retry_lines', 0))} | "
        f"step_retry_max={int(summary.get('step_retry_max', 0))}/{denoms} | "
        f"accepted_after_max={int(summary.get('step_accepted_after_max', 0))} | "
        f"rejected_count={int(summary.get('step_rejected_count', 0))} | "
        f"rejected_after_max={int(summary.get('step_rejected_after_max', 0))} | "
        f"epoch_retry_lines={int(summary.get('epoch_retry_lines', 0))} | "
        f"epoch_retry_max={int(summary.get('epoch_retry_max_attempt', 0))}/"
        f"{int(summary.get('epoch_retry_max_denom', 0))} | "
        f"adam_state_reset_lines={int(summary.get('adam_state_reset_lines', 0))} | "
        f"adam_state_reset_accepted={int(summary.get('adam_state_reset_accepted_count', 0))} | "
        f"adam_state_reset_trigger_max={int(summary.get('adam_state_reset_trigger_max', 0))} | "
        f"adam_state_reset_post_max={int(summary.get('adam_state_reset_post_max', 0))}"
    )


def _assigned_positions(total: int, shard_count: int, shard_index: int) -> list[int]:
    if shard_index < 1 or shard_index > shard_count:
        raise ValueError(f"invalid prest2 shard_index={shard_index}, shard_count={shard_count}")
    base = total // shard_count
    extra = total % shard_count
    start = 1 + (shard_index - 1) * base + min(shard_index - 1, extra)
    count = base + (1 if shard_index <= extra else 0)
    return list(range(start, start + count))


def _layer_shard_result_path(cfg: Prestage2Config, layer_name: str, shard_index: int) -> Path:
    token = _layer_token(layer_name)
    return cfg.result_dir / f"afm_prest2_05_{token}_p{shard_index}.pkl"


def _layer_assignment_path(cfg: Prestage2Config, layer_name: str, shard_index: int) -> Path:
    token = _layer_token(layer_name)
    return cfg.result_dir / f"afm_prest2_05_{token}_assignments_p{shard_index}.json"


def _layer_merged_result_path(cfg: Prestage2Config, layer_name: str) -> Path:
    canonical = _canonical_layer_name(layer_name)
    if canonical == LAYER_A:
        return cfg.result_dir / "afm_prest2_05_top_mech_winners_a.pkl"
    return cfg.result_dir / "afm_prest2_05_candidates_b.pkl"


def _legacy_layer_merged_result_path(cfg: Prestage2Config, layer_name: str) -> Path:
    canonical = _canonical_layer_name(layer_name)
    if canonical == LAYER_A:
        return cfg.result_dir / "afm_prest2_05_newrank_a.pkl"
    return cfg.result_dir / "afm_prest2_05.pkl"


def _stage1_origin_trial_root(
    cfg: Prestage2Config,
    *,
    mech_winner: int,
) -> Path:
    return cfg.candidate_run_root / "stage1_origin" / f"mech_winner_{int(mech_winner):03d}"


def _candidate_root_from_result_path(result_path: str | Path) -> Path:
    path = Path(result_path).resolve()
    return path.parent.parent


def _read_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected pickle payload for {path}: {type(payload)!r}")
    return payload


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_assignments(cfg: Prestage2Config, layer_name: str, items: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    total = len(items)
    for shard_index in range(1, cfg.shard_count + 1):
        positions = _assigned_positions(total, cfg.shard_count, shard_index) if total > 0 else []
        assigned = [items[pos - 1] for pos in positions]
        path = _layer_assignment_path(cfg, layer_name, shard_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(assigned, f, indent=2, ensure_ascii=False)
        paths.append(path)
    return paths


def _load_assignment_items(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prest2 assignment file: {path}")
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected assignment payload type for {path}: {type(payload)!r}")
    return [item for item in payload if isinstance(item, dict)]


def _annotate_history(rows: list[dict[str, Any]], *, layer_tag: str, epoch_offset: int = 0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        new_row = dict(row)
        new_row["epoch"] = epoch_offset + idx
        new_row["layer_epoch"] = idx
        new_row["layer_tag"] = layer_tag
        out.append(new_row)
    return out


def _combined_final_val(history: list[dict[str, Any]]) -> tuple[float, int]:
    final_val = float("nan")
    final_epoch = -1
    for row in history:
        try:
            value = float(row.get("val_loss", float("inf")))
            epoch = int(row.get("epoch", -1))
        except Exception:
            continue
        if math.isfinite(value):
            final_val = value
            final_epoch = epoch
    return final_val, final_epoch


def _layer_a_record_lookup(path: Path) -> dict[int, dict[str, Any]]:
    payload = _read_pickle(path)
    records = payload.get("candidate_records", [])
    lookup: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        key = int(record.get("top_mech_winner_a", record.get("newrank_a", 0)))
        if key > 0:
            lookup[key] = record
    return lookup


def _shard_progress_payload(
    *,
    layer_name: str,
    shard_index: int,
    candidate_records: list[dict[str, Any]],
    skipped_records: list[dict[str, Any]],
    cfg: Prestage2Config,
) -> dict[str, Any]:
    canonical_layer = _canonical_layer_name(layer_name)
    return {
        "layer_name": canonical_layer,
        "candidate_records": candidate_records,
        "skipped_records": skipped_records,
        "summary": {
            "layer_name": canonical_layer,
            "shard_index": int(shard_index),
            "completed": len(candidate_records),
            "skipped": len(skipped_records),
            "candidate_metric": cfg.candidate_metric,
        },
    }


def _save_shard_progress(
    path: Path,
    *,
    layer_name: str,
    shard_index: int,
    candidate_records: list[dict[str, Any]],
    skipped_records: list[dict[str, Any]],
    cfg: Prestage2Config,
) -> None:
    _save_pickle(
        path,
        _shard_progress_payload(
            layer_name=layer_name,
            shard_index=shard_index,
            candidate_records=candidate_records,
            skipped_records=skipped_records,
            cfg=cfg,
        ),
    )


def _layer_record_key(layer_name: str, record: dict[str, Any]) -> int:
    canonical_layer = _canonical_layer_name(layer_name)
    if canonical_layer == LAYER_A:
        return int(record.get("source_mech_winner", 0))
    return int(record.get("source_top_mech_winner_a", record.get("source_newrank_a", record.get("top_mech_winner_a", 0))))


def _layer_item_key(layer_name: str, item: dict[str, Any]) -> int:
    canonical_layer = _canonical_layer_name(layer_name)
    if canonical_layer == LAYER_A:
        return int(item.get("source_mech_winner", 0))
    return int(item.get("source_top_mech_winner_a", item.get("source_newrank_a", 0)))


def _epochs_completed(record: dict[str, Any]) -> int:
    try:
        return int(record.get("epochs_completed", 0))
    except Exception:
        return 0


def _load_existing_shard_records(
    path: Path,
    *,
    layer_name: str,
    log=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file():
        return [], []
    canonical_layer = _canonical_layer_name(layer_name)
    try:
        payload = _read_pickle(path)
    except Exception as err:
        if log is not None:
            _log_line(log, f"resume warning: could not read shard progress {path}: {err}")
        return [], []
    old_layer = str(payload.get("layer_name", canonical_layer)).strip().lower()
    if old_layer and _canonical_layer_name(old_layer) != canonical_layer:
        if log is not None:
            _log_line(log, f"resume warning: ignoring shard progress with layer mismatch: {path}")
        return [], []
    records = [rec for rec in payload.get("candidate_records", []) if isinstance(rec, dict)]
    skipped = [rec for rec in payload.get("skipped_records", []) if isinstance(rec, dict)]
    return records, skipped


def _dedupe_records_by_key(layer_name: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        key = _layer_record_key(layer_name, record)
        if key <= 0 or key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _completed_result_record_if_available(
    *,
    layer_name: str,
    source_mech_winner: int,
    source_top_mech_winner_a: int | None,
    candidate_root: Path,
    target_total_epochs: int,
    cfg: Prestage2Config,
    assignment_item: dict[str, Any],
    previous_record: dict[str, Any] | None,
) -> dict[str, Any] | None:
    result_path = candidate_root / "results" / "stage2light_result_p1.pt"
    if not result_path.is_file():
        return None
    try:
        payload = torch.load(result_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict) or _stage2light_payload_hard_zero_step_stop(payload):
        return None
    history = payload.get("history", [])
    if not isinstance(history, list) or len(history) < int(target_total_epochs):
        return None
    return _extract_candidate_record(
        layer_name=layer_name,
        source_mech_winner=source_mech_winner,
        candidate_root=candidate_root,
        payload=payload,
        cfg=cfg,
        source_top_mech_winner_a=source_top_mech_winner_a,
        assignment_item=assignment_item,
        previous_record=previous_record,
    )


def _extract_candidate_record(
    *,
    layer_name: str,
    source_mech_winner: int,
    candidate_root: Path,
    payload: dict[str, Any],
    cfg: Prestage2Config,
    source_top_mech_winner_a: int | None = None,
    assignment_item: dict[str, Any] | None = None,
    previous_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_history = payload.get("history", [])
    if not isinstance(current_history, list) or not current_history:
        raise RuntimeError(f"Missing prest2 history for mech winner {source_mech_winner}: {candidate_root}")

    warmstart = payload.get("warmstart", payload.get("stage1_warmstart", {}))
    if not isinstance(warmstart, dict):
        warmstart = {}

    previous_history = previous_record.get("history", []) if isinstance(previous_record, dict) else []
    previous_history = previous_history if isinstance(previous_history, list) else []
    previous_annotated = _annotate_history(previous_history, layer_tag="A", epoch_offset=0)

    canonical_layer = _canonical_layer_name(layer_name)
    if canonical_layer == LAYER_B and previous_annotated and len(current_history) >= len(previous_annotated):
        split_at = len(previous_annotated)
        current_annotated = (
            _annotate_history(current_history[:split_at], layer_tag="A", epoch_offset=0)
            + _annotate_history(current_history[split_at:], layer_tag="B", epoch_offset=split_at)
        )
        combined_history = current_annotated
        current_layer_epochs = max(0, len(current_history) - split_at)
    else:
        current_annotated = _annotate_history(
            current_history,
            layer_tag="A" if canonical_layer == LAYER_A else "B",
            epoch_offset=len(previous_annotated),
        )
        combined_history = previous_annotated + current_annotated
        current_layer_epochs = len(current_annotated)

    if not combined_history:
        raise RuntimeError(f"Combined prest2 history is empty for mech winner {source_mech_winner}: {candidate_root}")

    final_row = combined_history[-1]
    final_val_loss, final_val_epoch = _combined_final_val(combined_history)
    result_path = candidate_root / "results" / "stage2light_result_p1.pt"
    history_path = candidate_root / "results" / "stage2light_history_p1.json"
    log_path = _stage2light_child_log_path(candidate_root)
    checkpoint_path = candidate_root / "checkpoints" / "stage2light_final_p1.pt"
    retry_summary = _summarize_stage2light_retry_log(log_path)

    source_stage1_loss = float(warmstart.get("loss", float("inf")))
    if not math.isfinite(source_stage1_loss) and isinstance(previous_record, dict):
        source_stage1_loss = float(previous_record.get("source_stage1_loss", float("inf")))

    item = assignment_item if isinstance(assignment_item, dict) else {}
    source_mech_winner = int(
        source_mech_winner
        or warmstart.get("mech_winner", 0)
        or item.get("source_mech_winner", 0)
        or (previous_record.get("source_mech_winner", 0) if isinstance(previous_record, dict) else 0)
    )
    source_top_mech_winner_a = int(
        source_top_mech_winner_a
        or item.get("source_top_mech_winner_a", 0)
        or item.get("source_newrank_a", 0)
        or 0
    )
    source_stage1_rank = int(
        warmstart.get("rank", 0)
        or item.get("source_stage1_rank", 0)
        or (previous_record.get("source_stage1_rank", 0) if isinstance(previous_record, dict) else 0)
    )

    record = {
        "layer_name": _canonical_layer_name(layer_name),
        "source_mech_winner": int(source_mech_winner),
        "source_stage1_rank": int(source_stage1_rank),
        "source_top_mech_winner_a": int(source_top_mech_winner_a),
        "source_newrank_a": int(source_top_mech_winner_a),  # compatibility alias
        "source_mech_node_label": str(item.get("node_label", previous_record.get("source_mech_node_label", "") if isinstance(previous_record, dict) else "")),
        "source_ks_node_idx": int(item.get("ks_node_idx", previous_record.get("source_ks_node_idx", 0) if isinstance(previous_record, dict) else 0)),
        "source_cs_node_idx": int(item.get("cs_node_idx", previous_record.get("source_cs_node_idx", 0) if isinstance(previous_record, dict) else 0)),
        "source_stage1_trial_id": int(warmstart.get("trial_id", 0)),
        "source_stage1_loss": source_stage1_loss,
        "source_stage1_best_loss": float(item.get("best_loss", previous_record.get("source_stage1_best_loss", source_stage1_loss) if isinstance(previous_record, dict) else source_stage1_loss)),
        "source_stage1_best_seedbank": int(item.get("source_stage1_best_seedbank", previous_record.get("source_stage1_best_seedbank", 0) if isinstance(previous_record, dict) else 0)),
        "source_stage1_best_initseed": int(item.get("source_stage1_best_initseed", previous_record.get("source_stage1_best_initseed", 0) if isinstance(previous_record, dict) else 0)),
        "source_stage1_warmstart_record_path": str(item.get("source_stage1_warmstart_record_path", previous_record.get("source_stage1_warmstart_record_path", "") if isinstance(previous_record, dict) else "")),
        "source_mech_mean_loss_viable": float(item.get("mean_loss_viable", previous_record.get("source_mech_mean_loss_viable", float("inf")) if isinstance(previous_record, dict) else float("inf"))),
        "source_mech_viable_count": int(item.get("viable_count", previous_record.get("source_mech_viable_count", 0) if isinstance(previous_record, dict) else 0)),
        "source_mech_total_count": int(item.get("n_total", previous_record.get("source_mech_total_count", 0) if isinstance(previous_record, dict) else 0)),
        "trial_id": int(warmstart.get("trial_id", 0)),
        "seed": int(warmstart.get("seed", 0)),
        "ks0": float(warmstart.get("ks0", float("nan"))),
        "cs0": float(warmstart.get("cs0", float("nan"))),
        "epochs_completed": int(len(combined_history)),
        "current_layer_epochs": int(current_layer_epochs),
        "train_loss_start": float(combined_history[0].get("train_loss", float("inf"))),
        "val_loss_start": float(combined_history[0].get("val_loss", float("inf"))),
        "final_train_loss": float(final_row.get("train_loss", float("inf"))),
        "final_val_loss": float(final_val_loss),
        "final_val_epoch": int(final_val_epoch),
        "candidate_loss": float(final_row.get("train_loss", float("inf"))),
        "ks_hat": float(final_row.get("ks_hat", float("nan"))),
        "cs_hat": float(final_row.get("cs_hat", float("nan"))),
        "train_x1_rec": float(final_row.get("train_x1_rec", final_row.get("x1_rec", float("nan")))),
        "train_x2_rec": float(final_row.get("train_x2_rec", float("nan"))),
        "train_x2dot_rec": float(final_row.get("train_x2dot_rec", float("nan"))),
        "val_x1_rec": float(final_row.get("val_x1_rec", float("nan"))),
        "val_x2_rec": float(final_row.get("val_x2_rec", float("nan"))),
        "val_x2dot_rec": float(final_row.get("val_x2dot_rec", float("nan"))),
        "result_path": str(result_path.resolve()),
        "history_path": str(history_path.resolve()),
        "log_path": str(log_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "result_has_rng_state": bool("rng_state" in payload),
        "window_meta": payload.get("window_meta", {}),
        "history": combined_history,
        "retry_summary": retry_summary,
        "step_retry_lines": int(retry_summary["step_retry_lines"]),
        "step_retry_max": int(retry_summary["step_retry_max"]),
        "step_accepted_after_max": int(retry_summary["step_accepted_after_max"]),
        "step_rejected_count": int(retry_summary["step_rejected_count"]),
        "step_rejected_after_max": int(retry_summary["step_rejected_after_max"]),
        "epoch_retry_lines": int(retry_summary["epoch_retry_lines"]),
        "epoch_retry_max_attempt": int(retry_summary["epoch_retry_max_attempt"]),
        "epoch_retry_max_denom": int(retry_summary["epoch_retry_max_denom"]),
    }
    for key in (
        "initial_grid_support_source",
        "initial_grid_support",
        "initial_grid_support_x1_min",
        "initial_grid_support_x1_max",
        "initial_grid_support_x2_min",
        "initial_grid_support_x2_max",
        "initial_grid_support_x3_min",
        "initial_grid_support_x3_max",
        "rs_x3_normalizer_valid",
        "rs_x3_normalizer_source",
        "rs_x3_pred_mean",
        "rs_x3_pred_scale",
        "rs_x3_pred_min",
        "rs_x3_pred_max",
        "rs_x3_norm_support_min",
        "rs_x3_norm_support_max",
        "x3_agu_support_current",
    ):
        if key in warmstart:
            record[key] = warmstart[key]
        elif isinstance(previous_record, dict) and key in previous_record:
            record[key] = previous_record[key]
    x3_refit_meta = payload.get("x3_refit_meta")
    if isinstance(x3_refit_meta, dict):
        record["x3_refit_meta"] = x3_refit_meta
        try:
            record["rs_x3_normalizer_valid"] = True
            record["rs_x3_normalizer_source"] = str(x3_refit_meta.get("source", "prestage2_stage2light_preopt_refit"))
            record["rs_x3_pred_mean"] = float(x3_refit_meta["refit_x3_mean"])
            record["rs_x3_pred_scale"] = float(x3_refit_meta["refit_x3_scale"])
            record["rs_x3_pred_min"] = float(x3_refit_meta.get("x3_pred_min", float("nan")))
            record["rs_x3_pred_max"] = float(x3_refit_meta.get("x3_pred_max", float("nan")))
            record["rs_x3_norm_support_min"] = float(x3_refit_meta["x3_norm_support_min"])
            record["rs_x3_norm_support_max"] = float(x3_refit_meta["x3_norm_support_max"])
        except Exception:
            record["rs_x3_normalizer_valid"] = False
    initial_grid_support_meta = payload.get("initial_grid_support_meta")
    if isinstance(initial_grid_support_meta, dict):
        record["initial_grid_support_meta"] = initial_grid_support_meta
    if "initial_grid_support" in payload:
        record["initial_grid_support"] = payload.get("initial_grid_support")
    if "x3_agu_support_current" in payload:
        record["x3_agu_support_current"] = payload.get("x3_agu_support_current")
    if isinstance(previous_record, dict):
        record["source_layer_a_final_val_loss"] = float(previous_record.get("final_val_loss", float("inf")))
        record["source_layer_a_final_val_epoch"] = int(previous_record.get("final_val_epoch", -1))
    record["candidate_metric"] = str(cfg.candidate_metric)
    record["candidate_loss"] = _candidate_loss(record, cfg.candidate_metric)
    return record


def _candidate_record_from_skipped_record(
    *,
    layer_name: str,
    skipped_record: dict[str, Any],
    cfg: Prestage2Config,
    layer_a_lookup: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Promote a candidate-level skipped item into an explicit candidate record.

    A candidate that reaches Layer B and then stops with a hard zero-step is
    still part of the Layer B candidate set.  The stopped stage2light payload
    usually contains a usable history up to the stop point; use that state as
    the candidate result and mark it as skipped instead of silently dropping it.
    """

    canonical_layer = _canonical_layer_name(layer_name)
    candidate_root_raw = skipped_record.get("candidate_root", "")
    if not candidate_root_raw:
        return None
    candidate_root = Path(candidate_root_raw).resolve()
    source_mech_winner = int(skipped_record.get("source_mech_winner", 0))
    source_top_mech_winner_a = int(skipped_record.get("source_top_mech_winner_a") or 0)
    previous_record = None
    if canonical_layer == LAYER_B and layer_a_lookup is not None and source_top_mech_winner_a > 0:
        previous_record = layer_a_lookup.get(source_top_mech_winner_a)

    record: dict[str, Any] | None = None
    result_path = candidate_root / "results" / "stage2light_result_p1.pt"
    if result_path.is_file():
        try:
            payload = torch.load(result_path, map_location="cpu", weights_only=False)
        except Exception:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("history", []), list) and payload.get("history"):
            try:
                record = _extract_candidate_record(
                    layer_name=canonical_layer,
                    source_mech_winner=source_mech_winner,
                    candidate_root=candidate_root,
                    payload=payload,
                    cfg=cfg,
                    source_top_mech_winner_a=(
                        source_top_mech_winner_a if canonical_layer == LAYER_B else None
                    ),
                    assignment_item=skipped_record,
                    previous_record=previous_record,
                )
            except Exception:
                record = None

    if record is None and isinstance(previous_record, dict):
        record = dict(previous_record)
        record["layer_name"] = canonical_layer
        record["source_top_mech_winner_a"] = int(source_top_mech_winner_a)
        record["source_newrank_a"] = int(source_top_mech_winner_a)
        record["current_layer_epochs"] = 0
        record["candidate_loss"] = _candidate_loss(record, cfg.candidate_metric)
        record["candidate_metric"] = str(cfg.candidate_metric)
        record["candidate_root"] = str(candidate_root)
        record["result_path"] = str(result_path.resolve())
        record["log_path"] = str(skipped_record.get("stage2light_log_path", ""))

    if record is None:
        return None

    record["prest2_candidate_status"] = "skipped"
    record["candidate_completed"] = False
    record["candidate_skipped"] = True
    record["candidate_included_from_skipped_record"] = True
    record["skip_reason"] = str(skipped_record.get("skip_reason", "unknown"))
    if "stop_kind" in skipped_record:
        record["stop_kind"] = str(skipped_record.get("stop_kind", ""))
    if "stop_epoch" in skipped_record:
        try:
            record["stop_epoch"] = int(skipped_record.get("stop_epoch", -1))
        except Exception:
            record["stop_epoch"] = -1
    if "exit_code" in skipped_record:
        try:
            record["exit_code"] = int(skipped_record.get("exit_code", -1))
        except Exception:
            record["exit_code"] = -1
    if isinstance(skipped_record.get("retry_summary"), dict):
        record["retry_summary"] = skipped_record["retry_summary"]
    record["skipped_record"] = dict(skipped_record)
    record["candidate_loss"] = _candidate_loss(record, cfg.candidate_metric)
    return record


def run_prestage2_shard(
    cfg: Prestage2Config | None = None,
    *,
    shard_index: int | None = None,
    layer_name: str = LAYER_A,
    assignment_path: str | Path | None = None,
    layer_a_result_path: str | Path | None = None,
) -> dict[str, Any]:
    cfg = default_config() if cfg is None else cfg
    canonical_layer = _canonical_layer_name(layer_name)
    shard_index = int(cfg.shard_index if shard_index is None else shard_index)
    shard_log_path = make_shard_log_path(cfg.shard_log_dir, canonical_layer, shard_index)
    shard_log_path.parent.mkdir(parents=True, exist_ok=True)

    if assignment_path is None:
        if canonical_layer != LAYER_A:
            raise ValueError("layer_b shard execution requires an explicit assignment_path")
        stage1_payload = load_stage1_payload(cfg.stage1_input_path)
        all_items = _build_mech_winner_items(stage1_payload, cfg)
        items = [all_items[pos - 1] for pos in _assigned_positions(len(all_items), cfg.shard_count, shard_index)]
    else:
        items = _load_assignment_items(Path(assignment_path))

    shard_result_path = _layer_shard_result_path(cfg, canonical_layer, shard_index)
    candidate_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    layer_a_lookup: dict[int, dict[str, Any]] = {}
    resolved_layer_a_result_path: Path | None = None
    expected_stage2_window_meta: dict[str, Any] | None = None
    if canonical_layer == LAYER_B:
        if layer_a_result_path is None:
            raise ValueError("layer_b shard execution requires layer_a_result_path")
        resolved_layer_a_result_path = Path(layer_a_result_path).resolve()
        layer_a_lookup = _layer_a_record_lookup(resolved_layer_a_result_path)
        expected_stage2_window_meta = _expected_stage2_window_meta(cfg)

    log_mode = "a" if shard_log_path.is_file() else "w"
    with shard_log_path.open(log_mode, encoding="utf-8") as log:
        if log_mode == "a":
            _log_line(log, f"AFM05 prestage2 shard resume log append | layer={canonical_layer} | shard={shard_index}/{cfg.shard_count}")
        existing_records, existing_skipped = _load_existing_shard_records(
            shard_result_path,
            layer_name=canonical_layer,
            log=log,
        )
        candidate_records = _dedupe_records_by_key(canonical_layer, existing_records)
        skipped_records = _dedupe_records_by_key(canonical_layer, existing_skipped)
        completed_records_by_key = {
            _layer_record_key(canonical_layer, rec): rec
            for rec in candidate_records
            if _layer_record_key(canonical_layer, rec) > 0
        }
        skipped_records_by_key = {
            _layer_record_key(canonical_layer, rec): rec
            for rec in skipped_records
            if _layer_record_key(canonical_layer, rec) > 0
        }
        item_summary = [
            (
                int(item.get("source_top_mech_winner_a", item.get("source_newrank_a", 0))),
                int(item.get("source_mech_winner", 0)),
                int(item.get("source_stage1_trial_id", 0)),
            )
            for item in items
        ]
        _log_line(
            log,
            "AFM05 prestage2 shard start | "
            f"layer={canonical_layer} | shard={shard_index}/{cfg.shard_count} | items={item_summary}",
        )
        if candidate_records or skipped_records:
            _log_line(
                log,
                "resume state loaded | "
                f"completed_records={len(candidate_records)} | skipped_records={len(skipped_records)} | "
                f"shard_result={shard_result_path}",
            )
        for loop_idx, item in enumerate(items, start=1):
            source_mech_winner = int(item.get("source_mech_winner", 0))
            source_stage1_rank = int(item.get("source_stage1_rank", 0))
            source_stage1_trial_id = int(item.get("source_stage1_trial_id", item.get("best_trial_id", 0)))
            source_top_mech_winner_a = int(item.get("source_top_mech_winner_a", item.get("source_newrank_a", 0)))
            previous_record = layer_a_lookup.get(source_top_mech_winner_a) if canonical_layer == LAYER_B else None
            if canonical_layer == LAYER_B:
                if not isinstance(previous_record, dict):
                    raise KeyError(f"Missing layer A record for top mech winner A {source_top_mech_winner_a}")
                source_mech_winner = int(previous_record.get("source_mech_winner", source_mech_winner))
                source_stage1_rank = int(previous_record.get("source_stage1_rank", source_stage1_rank))
                source_stage1_trial_id = int(previous_record.get("source_stage1_trial_id", source_stage1_trial_id))
                source_stage1_warmstart_record_path = str(
                    item.get(
                        "source_stage1_warmstart_record_path",
                        previous_record.get("source_stage1_warmstart_record_path", ""),
                    )
                )
                candidate_root = _candidate_root_from_result_path(previous_record.get("result_path", ""))
            else:
                source_stage1_warmstart_record_path = str(item.get("source_stage1_warmstart_record_path", ""))
                candidate_root = _stage1_origin_trial_root(
                    cfg,
                    mech_winner=source_mech_winner,
                )

            if canonical_layer == LAYER_A:
                target_total_epochs = int(cfg.layer_a_epochs)
            else:
                target_total_epochs = int(previous_record.get("epochs_completed", cfg.layer_a_epochs)) + int(
                    cfg.layer_b_epochs
                )
            item_key = _layer_item_key(canonical_layer, item)
            existing_skipped = skipped_records_by_key.get(item_key)
            if isinstance(existing_skipped, dict):
                _log_line(
                    log,
                    "item resume-skip: already skipped in shard progress | "
                    f"layer={canonical_layer} | key={item_key} | "
                    f"reason={existing_skipped.get('skip_reason', 'unknown')} | root={candidate_root}",
                )
                continue

            existing_record = completed_records_by_key.get(item_key)
            if isinstance(existing_record, dict) and _epochs_completed(existing_record) >= target_total_epochs:
                _log_line(
                    log,
                    "item resume-skip: shard progress already complete | "
                    f"layer={canonical_layer} | key={item_key} | "
                    f"epochs={_epochs_completed(existing_record)}/{target_total_epochs} | root={candidate_root}",
                )
                continue

            recovered_record = _completed_result_record_if_available(
                layer_name=canonical_layer,
                source_mech_winner=source_mech_winner,
                source_top_mech_winner_a=(source_top_mech_winner_a if canonical_layer == LAYER_B else None),
                candidate_root=candidate_root,
                target_total_epochs=target_total_epochs,
                cfg=cfg,
                assignment_item=item,
                previous_record=previous_record,
            )
            if isinstance(recovered_record, dict):
                recovered_key = _layer_record_key(canonical_layer, recovered_record)
                if recovered_key > 0 and recovered_key not in completed_records_by_key:
                    candidate_records.append(recovered_record)
                    completed_records_by_key[recovered_key] = recovered_record
                    _save_shard_progress(
                        shard_result_path,
                        layer_name=canonical_layer,
                        shard_index=shard_index,
                        candidate_records=candidate_records,
                        skipped_records=skipped_records,
                        cfg=cfg,
                    )
                _log_line(
                    log,
                    "item resume-skip: recovered completed candidate result | "
                    f"layer={canonical_layer} | key={recovered_key} | "
                    f"epochs={_epochs_completed(recovered_record)}/{target_total_epochs} | root={candidate_root}",
                )
                continue

            if canonical_layer == LAYER_B:
                running_checkpoint = candidate_root / "checkpoints" / "stage2light_checkpoint_p1.pt"
                if not running_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Layer B seamless resume requires Layer A running checkpoint: {running_checkpoint}"
                    )
                _validate_layer_b_checkpoint_window(running_checkpoint, expected_stage2_window_meta or {})
            else:
                candidate_root.mkdir(parents=True, exist_ok=True)

            prefix_adam_epochs, prefix_lbfgs_epochs = _stage2light_prefix_schedule(
                stage2_adam_epochs=int(cfg.stage2_adam_epochs),
                target_total_epochs=target_total_epochs,
            )

            env = dict(os.environ)
            for key in (
                "HNODECB_AFM05_STAGE2LIGHT_INPUT_RANK",
                "HNODECB_AFM05_STAGE2LIGHT_INPUT_TRIAL_ID",
                "HNODECB_AFM05_STAGE2LIGHT_INPUT_MECH_WINNER",
                "HNODECB_AFM05_STAGE2LIGHT_INPUT_CANDIDATE",
                "HNODECB_AFM05_STAGE2LIGHT_PREST2_INPUT_PATH",
                "HNODECB_AFM05_STAGE2LIGHT_STAGE1_WARMSTART_RECORD_PATH",
                "HNODECB_AFM05_STAGE2LIGHT_WARMSTART_FROM_STAGE1",
            ):
                env.pop(key, None)
            env["HNODECB_AFM05_STAGE2LIGHT_OUTPUT_ROOT"] = str(candidate_root)
            env["HNODECB_AFM05_STAGE2LIGHT_SHARD_COUNT"] = "1"
            env["HNODECB_AFM05_STAGE2LIGHT_AUTOGEN_DATASET"] = "1" if cfg.auto_generate_dataset else "0"
            stage2_cfg = default_stage2light_config(cfg.repo_root)
            env["HNODECB_AFM05_STAGE2LIGHT_WINDOW_MODE"] = str(stage2_cfg.window_mode)
            env["HNODECB_AFM05_STAGE2LIGHT_PIXEL_TAG"] = str(stage2_cfg.pixel_tag)
            env["HNODECB_AFM05_STAGE2LIGHT_WINDOW_US"] = f"{float(stage2_cfg.arch_window_us):.12e}"
            env["HNODECB_AFM05_STAGE2LIGHT_WINDOW_SAMPLE_STRIDE"] = str(int(stage2_cfg.window_sample_stride))
            env["HNODECB_AFM05_STAGE2LIGHT_VAL_STRIDE"] = str(int(stage2_cfg.val_stride))
            env["HNODECB_AFM05_STAGE2LIGHT_VAL_OFFSET"] = str(int(stage2_cfg.val_offset))
            env["HNODECB_AFM05_STAGE2LIGHT_EPOCHS"] = str(target_total_epochs)
            env["HNODECB_AFM05_STAGE2LIGHT_ADAM_EPOCHS"] = str(prefix_adam_epochs)
            env["HNODECB_AFM05_STAGE2LIGHT_LBFGS_EPOCHS"] = str(prefix_lbfgs_epochs)
            env["HNODECB_AFM05_STAGE2LIGHT_CHECKPOINT_EVERY"] = str(cfg.checkpoint_every)
            env["HNODECB_AFM05_STAGE2LIGHT_LOG_EVERY"] = "1"
            # prest2 is a prefix projection of the formal st2l process.  Do not
            # silently switch validation semantics here; even validation rollouts
            # must follow st2l so the prefix remains auditable.
            env["HNODECB_AFM05_STAGE2LIGHT_VAL_EVAL_MODE"] = str(stage2_cfg.val_eval_mode)
            env["HNODECB_AFM05_STAGE2LIGHT_TRAIN_WINDOW_INDEX"] = "0"
            env["HNODECB_AFM05_STAGE2LIGHT_SOFT_MASK"] = "1"
            env["HNODECB_AFM05_STAGE2LIGHT_SOFT_MASK_TRAINABLE"] = "1"
            env["HNODECB_AFM05_STAGE2LIGHT_SOFT_MASK_S0_A0"] = "20.0"
            env["HNODECB_AFM05_STAGE2LIGHT_SOFT_MASK_S0_MIN_A0"] = "1.0"
            env["HNODECB_AFM05_STAGE2LIGHT_SOFT_MASK_S0_MAX_A0"] = "100.0"
            env["HNODECB_AFM05_STAGE2LIGHT_SOFT_MASK_ALPHA_A0"] = "0.25"
            env["HNODECB_AFM05_STAGE2LIGHT_SOFT_MASK_ALPHA_MIN_A0"] = "0.02"
            env["HNODECB_AFM05_STAGE2LIGHT_SOFT_MASK_ALPHA_MAX_A0"] = "5.0"
            env["HNODECB_AFM05_STAGE2LIGHT_LR"] = "1e-3"
            env["HNODECB_AFM05_STAGE2LIGHT_LR_ADAPT"] = "0"
            env["HNODECB_AFM05_STAGE2LIGHT_LR_ADAPT_UP_ONLY"] = "0"
            env["HNODECB_AFM05_STAGE2LIGHT_MECH_PARAMETERIZATION"] = "log_relative"
            env["HNODECB_AFM05_STAGE2LIGHT_LR_MIN"] = "1e-6"
            env["HNODECB_AFM05_STAGE2LIGHT_LR_MAX"] = "1e-2"
            env["HNODECB_AFM05_STAGE2LIGHT_LR_ETA"] = "0.05"
            env["HNODECB_AFM05_STAGE2LIGHT_LR_EMA"] = "0.97"
            env["HNODECB_AFM05_STAGE2LIGHT_EPOCH_RETRIES"] = "8"
            env["HNODECB_AFM05_STAGE2LIGHT_RETRY_LR_FACTOR"] = "0.3"
            env["HNODECB_AFM05_STAGE2LIGHT_RETRY_LR_FLOOR"] = "1e-6"
            env["HNODECB_AFM05_STAGE2LIGHT_STEP_CONTROLLER"] = "off"
            env["HNODECB_AFM05_STAGE2LIGHT_STEP_GUARD"] = "0"
            env["HNODECB_AFM05_STAGE2LIGHT_STEP_RETRIES"] = "6"
            env["HNODECB_AFM05_STAGE2LIGHT_STEP_RETRY_LR_FACTOR"] = "0.3"
            env["HNODECB_AFM05_STAGE2LIGHT_STEP_RETRY_RESET_LR_EACH_EPOCH"] = "0"
            env["HNODECB_AFM05_STAGE2LIGHT_STEP_GUARD_VALIDATE_VAL"] = "0"
            env["HNODECB_AFM05_STAGE2LIGHT_STEP_MAX_LOSS_INCREASE_FRAC"] = "0.05"
            env["HNODECB_AFM05_STAGE2LIGHT_ARMIJO_C1"] = "1e-4"
            env["HNODECB_AFM05_STAGE2LIGHT_BACKTRACK_SHRINK"] = "0.5"
            env["HNODECB_AFM05_STAGE2LIGHT_BACKTRACK_MAX"] = "10"
            env["HNODECB_AFM05_STAGE2LIGHT_BACKTRACK_MIN_ALPHA"] = "1e-8"
            env["HNODECB_AFM05_STAGE2LIGHT_LBFGS_ENABLED"] = "1" if prefix_lbfgs_epochs > 0 else "0"
            env["HNODECB_AFM05_STAGE2LIGHT_LBFGS_STRONG_WOLFE"] = "1"
            env["HNODECB_AFM05_STAGE2LIGHT_LBFGS_LR"] = "1.0"
            env["HNODECB_AFM05_STAGE2LIGHT_LBFGS_MAX_ITER"] = "1"
            env["HNODECB_AFM05_STAGE2LIGHT_LBFGS_MAX_EVAL"] = "10"
            env["HNODECB_AFM05_STAGE2LIGHT_LBFGS_HISTORY_SIZE"] = "100"
            env["HNODECB_AFM05_STAGE2LIGHT_LBFGS_TOLERANCE_GRAD"] = "1e-12"
            env["HNODECB_AFM05_STAGE2LIGHT_LBFGS_TOLERANCE_CHANGE"] = "1e-15"
            env["HNODECB_AFM05_STAGE2LIGHT_LBFGS_YS_THRESHOLD"] = "1e-15"
            env["HNODECB_AFM05_STAGE2LIGHT_PLATEAU_EARLY_STOP"] = "0"
            env["HNODECB_AFM05_STAGE2LIGHT_RECENT_VAL_EARLY_STOP"] = "0"
            env["HNODECB_AFM05_STAGE2LIGHT_RESUME"] = "1"

            if canonical_layer == LAYER_A:
                env["HNODECB_AFM05_STAGE2LIGHT_WARMSTART_SOURCE"] = "stage1"
                env["HNODECB_AFM05_STAGE2LIGHT_WARMSTART_FROM_STAGE1"] = "1"
                env["HNODECB_AFM05_STAGE2LIGHT_INPUT_PATH"] = str(cfg.stage1_input_path)
                env["HNODECB_AFM05_STAGE2LIGHT_INPUT_TRIAL_ID"] = str(source_stage1_trial_id)
                env["HNODECB_AFM05_STAGE2LIGHT_INPUT_MECH_WINNER"] = str(source_mech_winner)
                if source_stage1_warmstart_record_path.strip():
                    env["HNODECB_AFM05_STAGE2LIGHT_STAGE1_WARMSTART_RECORD_PATH"] = source_stage1_warmstart_record_path
                launch_msg = (
                    f"launch layer A item {loop_idx}/{len(items)} | "
                    f"{_mw_seed_rank_label(item)} | "
                    f"trial={source_stage1_trial_id} | root={candidate_root}"
                )
            else:
                env["HNODECB_AFM05_STAGE2LIGHT_WARMSTART_SOURCE"] = "stage1"
                env["HNODECB_AFM05_STAGE2LIGHT_WARMSTART_FROM_STAGE1"] = "1"
                env["HNODECB_AFM05_STAGE2LIGHT_INPUT_PATH"] = str(cfg.stage1_input_path)
                env["HNODECB_AFM05_STAGE2LIGHT_INPUT_TRIAL_ID"] = str(source_stage1_trial_id)
                env["HNODECB_AFM05_STAGE2LIGHT_INPUT_MECH_WINNER"] = str(source_mech_winner)
                if source_stage1_warmstart_record_path.strip():
                    env["HNODECB_AFM05_STAGE2LIGHT_STAGE1_WARMSTART_RECORD_PATH"] = source_stage1_warmstart_record_path
                launch_msg = (
                    f"launch layer B item {loop_idx}/{len(items)} | "
                    f"top mech winner A={source_top_mech_winner_a} | "
                    f"{_mw_seed_rank_label(previous_record)} | "
                    f"trial={source_stage1_trial_id} | "
                    f"resume_root={candidate_root} | epochs_total_target={target_total_epochs}"
                )

            child_log_path = _stage2light_child_log_path(candidate_root)
            child_stdout_path = child_log_path.with_suffix(child_log_path.suffix + ".stdout.txt")
            child_stderr_path = child_log_path.with_suffix(child_log_path.suffix + ".stderr.txt")
            _log_line(
                log,
                f"{launch_msg} | "
                f"st2l_prefix_target={target_total_epochs} | "
                f"optimizer_prefix=adam:{prefix_adam_epochs},lbfgs:{prefix_lbfgs_epochs} | "
                f"stage2light_log={child_log_path} | "
                f"warmstart_record={source_stage1_warmstart_record_path or 'full_stage1_fallback'} | "
                f"stdout={child_stdout_path} | stderr={child_stderr_path}",
            )
            child_stdout_path.parent.mkdir(parents=True, exist_ok=True)
            with child_stdout_path.open("w", encoding="utf-8") as stdout_fh, child_stderr_path.open(
                "w",
                encoding="utf-8",
            ) as stderr_fh:
                proc = subprocess.run(
                    [sys.executable, "-m", "AFM05.stage2light.main", "--shard-index", "1"],
                    cwd=str(cfg.repo_root),
                    env=env,
                    stdout=stdout_fh,
                    stderr=stderr_fh,
                    text=True,
                    check=False,
                )
            if proc.returncode != 0:
                retry_summary = _summarize_stage2light_retry_log(child_log_path)
                _log_line(
                    log,
                    f"item failed | {_mw_seed_rank_label(item)} | "
                    f"trial={source_stage1_trial_id} | "
                    f"exit={proc.returncode} | "
                    f"stdout={child_stdout_path} | stderr={child_stderr_path} | "
                    f"{_retry_summary_line(retry_summary)}",
                )
                hard_zero_stop = _stage2light_log_has_hard_zero_step_stop(child_log_path)
                x3_refit_stop = _stage2light_log_has_x3_refit_stop(child_log_path)
                if hard_zero_stop or x3_refit_stop:
                    if hard_zero_stop:
                        skip_reason = "lbfgs_hard_zero_step_stop"
                    else:
                        skip_reason = "x3_normalizer_refit_failed"
                    skipped_records.append(
                        {
                            "layer_name": canonical_layer,
                            "source_mech_winner": int(source_mech_winner),
                            "source_top_mech_winner_a": (
                                int(source_top_mech_winner_a)
                                if source_top_mech_winner_a is not None
                                else None
                            ),
                            "source_stage1_rank": int(source_stage1_rank),
                            "source_stage1_trial_id": int(source_stage1_trial_id),
                            "source_stage1_warmstart_record_path": str(source_stage1_warmstart_record_path),
                            "candidate_root": str(candidate_root.resolve()),
                            "stage2light_log_path": str(child_log_path.resolve()),
                            "skip_reason": skip_reason,
                            "exit_code": int(proc.returncode),
                            "retry_summary": retry_summary,
                        }
                    )
                    if item_key > 0:
                        skipped_records_by_key[item_key] = skipped_records[-1]
                    _save_pickle(
                        shard_result_path,
                        {
                            "layer_name": canonical_layer,
                            "candidate_records": candidate_records,
                            "skipped_records": skipped_records,
                            "summary": {
                                "layer_name": canonical_layer,
                                "shard_index": shard_index,
                                "completed": len(candidate_records),
                                "skipped": len(skipped_records),
                                "candidate_metric": cfg.candidate_metric,
                            },
                        },
                    )
                    _log_line(
                        log,
                        f"item skipped after candidate-level {skip_reason} | "
                        f"{_mw_seed_rank_label(item)} | trial={source_stage1_trial_id}",
                    )
                    continue
                raise RuntimeError(
                    f"prest2 {canonical_layer} run failed for mech winner {source_mech_winner}: exit={proc.returncode}"
                )

            result_path = candidate_root / "results" / "stage2light_result_p1.pt"
            if not result_path.is_file():
                raise FileNotFoundError(f"prest2 candidate result missing: {result_path}")
            payload = torch.load(result_path, map_location="cpu", weights_only=False)
            payload_hard_zero_stop = _stage2light_payload_hard_zero_step_stop(payload)
            if payload_hard_zero_stop:
                retry_summary = _summarize_stage2light_retry_log(child_log_path)
                skipped_records.append(
                    {
                        "layer_name": canonical_layer,
                        "source_mech_winner": int(source_mech_winner),
                        "source_top_mech_winner_a": (
                            int(source_top_mech_winner_a)
                            if source_top_mech_winner_a is not None
                            else None
                        ),
                        "source_stage1_rank": int(source_stage1_rank),
                        "source_stage1_trial_id": int(source_stage1_trial_id),
                        "source_stage1_warmstart_record_path": str(source_stage1_warmstart_record_path),
                        "candidate_root": str(candidate_root.resolve()),
                        "stage2light_log_path": str(child_log_path.resolve()),
                        "skip_reason": str(payload.get("stop_reason", "lbfgs_hard_zero_step_stop")),
                        "stop_kind": str(payload.get("stop_kind", "")),
                        "stop_epoch": int(payload.get("stop_epoch", -1)),
                        "retry_summary": retry_summary,
                    }
                )
                if item_key > 0:
                    skipped_records_by_key[item_key] = skipped_records[-1]
                _save_pickle(
                    shard_result_path,
                    {
                        "layer_name": canonical_layer,
                        "candidate_records": candidate_records,
                        "skipped_records": skipped_records,
                        "summary": {
                            "layer_name": canonical_layer,
                            "shard_index": shard_index,
                            "completed": len(candidate_records),
                            "skipped": len(skipped_records),
                            "candidate_metric": cfg.candidate_metric,
                        },
                    },
                )
                _log_line(
                    log,
                    "item skipped after candidate-level lbfgs_hard_zero_step_stop | "
                    f"{_mw_seed_rank_label(item)} | trial={source_stage1_trial_id} | "
                    f"stop_epoch={int(payload.get('stop_epoch', -1))} | "
                    f"{_retry_summary_line(retry_summary)}",
                )
                continue
            record = _extract_candidate_record(
                layer_name=canonical_layer,
                source_mech_winner=source_mech_winner,
                candidate_root=candidate_root,
                payload=payload,
                cfg=cfg,
                source_top_mech_winner_a=(source_top_mech_winner_a if canonical_layer == LAYER_B else None),
                assignment_item=item,
                previous_record=previous_record,
            )
            candidate_records.append(record)
            record_key = _layer_record_key(canonical_layer, record)
            if record_key > 0:
                completed_records_by_key[record_key] = record
            retry_summary = record.get("retry_summary", {})
            _save_pickle(
                shard_result_path,
                {
                    "layer_name": canonical_layer,
                    "candidate_records": candidate_records,
                    "skipped_records": skipped_records,
                    "summary": {
                        "layer_name": canonical_layer,
                        "shard_index": shard_index,
                        "completed": len(candidate_records),
                        "skipped": len(skipped_records),
                        "candidate_metric": cfg.candidate_metric,
                    },
                },
            )
            _log_line(
                log,
                "item done | "
                f"{_mw_seed_rank_label(record)} | "
                f"top_mech_winner_a={source_top_mech_winner_a} | "
                f"trial={source_stage1_trial_id} | "
                f"epochs_total={record['epochs_completed']} | "
                f"epochs_layer={record['current_layer_epochs']} | "
                f"final_train={record['final_train_loss']:.6e} | "
                f"final_val={record['final_val_loss']:.6e} @ e{record['final_val_epoch']} | "
                f"ks={record['ks_hat']:.6e} | "
                f"cs={record['cs_hat']:.6e} | "
                f"val_x1={record['val_x1_rec']:.2f}% | "
                f"val_x2={record['val_x2_rec']:.2f}% | "
                f"val_x2dot={record['val_x2dot_rec']:.2f}% | "
                f"{_retry_summary_line(retry_summary)}",
            )

        _log_line(
            log,
            f"AFM05 prestage2 shard done | layer={canonical_layer} | "
            f"completed={len(candidate_records)} | skipped={len(skipped_records)}",
        )

    return {
        "layer_name": canonical_layer,
        "shard_index": shard_index,
        "log_path": str(shard_log_path),
        "result_path": str(shard_result_path),
        "completed": len(candidate_records),
        "skipped": len(skipped_records),
    }


def merge_prestage2_results(cfg: Prestage2Config | None = None, *, layer_name: str = LAYER_B) -> dict[str, Any]:
    cfg = default_config() if cfg is None else cfg
    canonical_layer = _canonical_layer_name(layer_name)
    all_records: list[dict[str, Any]] = []
    all_skipped_records: list[dict[str, Any]] = []
    source_shards: list[str] = []
    for shard_index in range(1, cfg.shard_count + 1):
        shard_path = _layer_shard_result_path(cfg, canonical_layer, shard_index)
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing prest2 shard export: {shard_path}")
        payload = _read_pickle(shard_path)
        source_shards.append(str(shard_path.resolve()))
        records = payload.get("candidate_records", [])
        all_records.extend([rec for rec in records if isinstance(rec, dict)])
        skipped_records = payload.get("skipped_records", [])
        all_skipped_records.extend([rec for rec in skipped_records if isinstance(rec, dict)])

    for record in all_records:
        record.setdefault("prest2_candidate_status", "completed")
        record.setdefault("candidate_completed", True)
        record.setdefault("candidate_skipped", False)

    completed_keys = {
        _layer_record_key(canonical_layer, record)
        for record in all_records
        if _layer_record_key(canonical_layer, record) > 0
    }
    included_skipped_records: list[dict[str, Any]] = []
    dropped_skipped_records: list[dict[str, Any]] = []
    layer_a_lookup: dict[int, dict[str, Any]] = {}
    if canonical_layer == LAYER_B and all_skipped_records:
        layer_a_path = _layer_merged_result_path(cfg, LAYER_A)
        if layer_a_path.is_file():
            layer_a_lookup = _layer_a_record_lookup(layer_a_path)
    for skipped in all_skipped_records:
        skipped_key = _layer_record_key(canonical_layer, skipped)
        if skipped_key <= 0 or skipped_key in completed_keys:
            dropped_skipped_records.append(skipped)
            continue
        skipped_record = _candidate_record_from_skipped_record(
            layer_name=canonical_layer,
            skipped_record=skipped,
            cfg=cfg,
            layer_a_lookup=layer_a_lookup,
        )
        if skipped_record is None:
            dropped_skipped_records.append(skipped)
            continue
        all_records.append(skipped_record)
        included_skipped_records.append(skipped_record)
        completed_keys.add(skipped_key)

    ranked = _rank_candidates(all_records, cfg.candidate_metric)
    ranking_label = _rank_label_name(canonical_layer)
    for idx, record in enumerate(ranked, start=1):
        if canonical_layer == LAYER_A:
            record["top_mech_winner_a"] = int(idx)
            record["top_mech_winner_a_label"] = f"top mech winner A {idx}"
            record["newrank_a"] = int(idx)  # compatibility alias
            record["newrank_a_label"] = f"top mech winner A {idx}"
        else:
            record["candidate_b"] = int(idx)
            record["candidate_b_label"] = f"candidate B {idx}"
            record["candidate"] = int(idx)  # compatibility alias
            record["candidate_label"] = f"candidate B {idx}"

    summary = {
        "layer_name": canonical_layer,
        "ranking_label": ranking_label,
        "total_candidates": len(ranked),
        "completed_candidates": sum(1 for rec in ranked if not bool(rec.get("candidate_skipped", False))),
        "skipped_candidates": sum(1 for rec in ranked if bool(rec.get("candidate_skipped", False))),
        "raw_completed_records": len(all_records) - len(included_skipped_records),
        "raw_skipped_records": len(all_skipped_records),
        "included_skipped_records": len(included_skipped_records),
        "dropped_skipped_records": len(dropped_skipped_records),
        "candidate_metric": cfg.candidate_metric,
        "best_candidate_loss": float(_candidate_loss(ranked[0], cfg.candidate_metric)) if ranked else float("inf"),
        "source_shards": source_shards,
    }
    payload = {
        "layer_name": canonical_layer,
        "ranking_label": ranking_label,
        "candidate_records": ranked,
        "skipped_records": all_skipped_records,
        "included_skipped_candidate_records": included_skipped_records,
        "dropped_skipped_records": dropped_skipped_records,
        "summary": summary,
        "source_shards": source_shards,
    }
    if canonical_layer == LAYER_A:
        payload["top_mech_winner_a_records"] = ranked
        payload["newrank_a_records"] = ranked
    else:
        payload["candidate_b_records"] = ranked

    merged_path = _layer_merged_result_path(cfg, canonical_layer)
    _save_pickle(merged_path, payload)
    legacy_path = _legacy_layer_merged_result_path(cfg, canonical_layer)
    if legacy_path != merged_path:
        _save_pickle(legacy_path, payload)
    summary_path = merged_path.with_suffix(merged_path.suffix + ".summary.json")
    topk_path = merged_path.with_suffix(merged_path.suffix + ".topk.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)
    with topk_path.open("w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False, default=_json_default)
    return payload


def _run_layer_processes(
    cfg: Prestage2Config,
    *,
    layer_name: str,
    driver_log,
    assignment_paths: list[Path],
    layer_a_result_path: Path | None = None,
) -> None:
    canonical_layer = _canonical_layer_name(layer_name)
    procs: list[tuple[int, subprocess.Popen[str], Path]] = []
    base_env = dict(os.environ)
    base_env["HNODECB_AFM05_PREST2_SHARD_COUNT"] = str(cfg.shard_count)
    base_env["HNODECB_AFM05_PREST2_CHECKPOINT_EVERY"] = str(cfg.checkpoint_every)
    base_env["HNODECB_AFM05_PREST2_INPUT_PATH"] = str(cfg.stage1_input_path)
    base_env["HNODECB_AFM05_PREST2_OUTPUT_ROOT"] = str(cfg.output_root)
    if cfg.auto_generate_dataset:
        base_env["HNODECB_AFM05_PREST2_AUTOGEN_DATASET"] = "1"

    for shard_index in range(1, cfg.shard_count + 1):
        env = dict(base_env)
        env["HNODECB_AFM05_PREST2_SHARD_INDEX"] = str(shard_index)
        args = [
            sys.executable,
            "-m",
            "AFM05.prestage2.main",
            "--shard-index",
            str(shard_index),
            "--layer",
            canonical_layer,
            "--assignment-path",
            str(assignment_paths[shard_index - 1]),
        ]
        if layer_a_result_path is not None:
            args.extend(["--layer-a-result-path", str(layer_a_result_path)])
        shard_log = make_shard_log_path(cfg.shard_log_dir, canonical_layer, shard_index)
        shard_stdout_path = shard_log.with_suffix(shard_log.suffix + ".stdout.txt")
        shard_stderr_path = shard_log.with_suffix(shard_log.suffix + ".stderr.txt")
        shard_stdout_path.parent.mkdir(parents=True, exist_ok=True)
        shard_stdout_fh = shard_stdout_path.open("w", encoding="utf-8")
        shard_stderr_fh = shard_stderr_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            args,
            cwd=str(cfg.repo_root),
            env=env,
            stdout=shard_stdout_fh,
            stderr=shard_stderr_fh,
            text=True,
        )
        proc._afm05_stdout_fh = shard_stdout_fh  # type: ignore[attr-defined]
        proc._afm05_stderr_fh = shard_stderr_fh  # type: ignore[attr-defined]
        procs.append((shard_index, proc, shard_log))
        _log_line(
            driver_log,
            f"launched {canonical_layer} shard {shard_index}/{cfg.shard_count} | "
            f"pid={proc.pid} | log={shard_log} | stdout={shard_stdout_path} | stderr={shard_stderr_path}",
            echo=True,
        )

    exit_codes: dict[int, int] = {}
    try:
        for shard_index, proc, shard_log in procs:
            code = proc.wait()
            exit_codes[shard_index] = int(code)
            _log_line(
                driver_log,
                f"{canonical_layer} shard {shard_index}/{cfg.shard_count} exited | code={code} | log={shard_log}",
                echo=True,
            )
    finally:
        for _shard_index, proc, _shard_log in procs:
            if proc.poll() is None:
                proc.kill()
            stdout_fh = getattr(proc, "_afm05_stdout_fh", None)
            stderr_fh = getattr(proc, "_afm05_stderr_fh", None)
            if stdout_fh is not None:
                stdout_fh.close()
            if stderr_fh is not None:
                stderr_fh.close()

    failed = {idx: code for idx, code in exit_codes.items() if code != 0}
    if failed:
        raise RuntimeError(f"AFM05 prestage2 {canonical_layer} failed: {failed}")


def _print_layer_ranking(log, *, layer_name: str, records: list[dict[str, Any]]) -> None:
    canonical_layer = _canonical_layer_name(layer_name)
    if canonical_layer == LAYER_A:
        _log_line(log, "prest2 layer A top mech winners A summary:", echo=True)
        for record in records:
            _log_line(
                log,
                f"{_mw_seed_rank_label(record)} -> "
                f"top mech winner A {int(record.get('top_mech_winner_a', record.get('newrank_a', 0)))} | "
                f"trial={int(record.get('trial_id', 0))} | "
                f"epochs_total={int(record.get('epochs_completed', 0))} | "
                f"epochs_layer={int(record.get('current_layer_epochs', 0))} | "
                f"final_train={float(record.get('final_train_loss', float('inf'))):.6e} | "
                f"final_val={float(record.get('final_val_loss', float('inf'))):.6e} @ e{int(record.get('final_val_epoch', -1))} | "
                f"ks={float(record.get('ks_hat', float('nan'))):.6e} | "
                f"cs={float(record.get('cs_hat', float('nan'))):.6e} | "
                f"val_x1={float(record.get('val_x1_rec', float('nan'))):.2f}% | "
                f"val_x2={float(record.get('val_x2_rec', float('nan'))):.2f}% | "
                f"val_x2dot={float(record.get('val_x2dot_rec', float('nan'))):.2f}%",
                echo=True,
            )
        return

    _log_line(log, "prest2 candidates B summary:", echo=True)
    for record in records:
        _log_line(
            log,
            f"{_mw_seed_rank_label(record)} -> "
            f"top mech winner A {int(record.get('source_top_mech_winner_a', record.get('source_newrank_a', 0)))} -> "
            f"candidate B {int(record.get('candidate_b', record.get('candidate', 0)))} | "
            f"trial={int(record.get('trial_id', 0))} | "
            f"epochs_total={int(record.get('epochs_completed', 0))} | "
            f"epochs_layer={int(record.get('current_layer_epochs', 0))} | "
            f"final_train={float(record.get('final_train_loss', float('inf'))):.6e} | "
            f"final_val={float(record.get('final_val_loss', float('inf'))):.6e} @ e{int(record.get('final_val_epoch', -1))} | "
            f"ks={float(record.get('ks_hat', float('nan'))):.6e} | "
            f"cs={float(record.get('cs_hat', float('nan'))):.6e} | "
            f"val_x1={float(record.get('val_x1_rec', float('nan'))):.2f}% | "
            f"val_x2={float(record.get('val_x2_rec', float('nan'))):.2f}% | "
            f"val_x2dot={float(record.get('val_x2dot_rec', float('nan'))):.2f}%",
            echo=True,
        )


def run_driver(cfg: Prestage2Config | None = None) -> dict[str, Any]:
    cfg = default_config() if cfg is None else cfg
    driver_log_path = make_driver_log_path(cfg.log_dir)
    driver_log_path.parent.mkdir(parents=True, exist_ok=True)
    stage1_payload = load_stage1_payload(cfg.stage1_input_path)
    stage1_trial_count = validate_complete_stage1plus_payload(stage1_payload, path=cfg.stage1_input_path)

    layer_a_items = _build_mech_winner_items(stage1_payload, cfg)
    layer_a_assignments = _write_assignments(cfg, LAYER_A, layer_a_items)

    with driver_log_path.open("w", encoding="utf-8") as log:
        _log_line(
            log,
            "AFM05 prestage2 driver start | "
            f"shards={cfg.shard_count} | "
            f"layerA_mech_winners={len(layer_a_items)} | layerA_epochs={cfg.layer_a_epochs} | "
            f"layerB_top_mech_winners_A={cfg.layer_b_top_candidates} | layerB_epochs={cfg.layer_b_epochs} | "
            f"st2l_schedule=total:{cfg.stage2_total_epochs},adam:{cfg.stage2_adam_epochs},"
            f"lbfgs:{cfg.stage2_lbfgs_epochs} | "
            "prest2_layers_select_prefix_length_only | "
            f"metric={cfg.candidate_metric} | "
            f"stage1_input={cfg.stage1_input_path} | stage1_trials={stage1_trial_count}",
            echo=True,
        )

        _run_layer_processes(cfg, layer_name=LAYER_A, driver_log=log, assignment_paths=layer_a_assignments)
        merged_a = merge_prestage2_results(cfg, layer_name=LAYER_A)
        records_a = [rec for rec in merged_a.get("candidate_records", []) if isinstance(rec, dict)]
        _print_layer_ranking(log, layer_name=LAYER_A, records=records_a)

        layer_a_result_path = _layer_merged_result_path(cfg, LAYER_A)
        layer_a_checkpoint_path = cfg.output_root / "checkpoints" / "afm_prest2_05_after_layer_a.checkpoint.pkl"
        _save_pickle(
            layer_a_checkpoint_path,
            {
                "completed_layer": LAYER_A,
                "layer_a_result_path": str(layer_a_result_path.resolve()),
                "layer_a_top_candidates": int(cfg.layer_a_top_candidates),
                "layer_a_zone_filter": str(cfg.layer_a_zone_filter),
                "layer_a_epochs": int(cfg.layer_a_epochs),
                "layer_b_pending": True,
            },
        )
        _log_line(log, f"Layer A checkpoint saved -> {layer_a_checkpoint_path}", echo=True)

        if cfg.stop_after_layer_a:
            _log_line(log, "Stop-after-Layer-A switch is enabled; Layer B was not started.", echo=True)
            return {
                "driver_log": str(driver_log_path),
                "completed_layer": LAYER_A,
                "layer_a_result_path": str(layer_a_result_path.resolve()),
                "layer_a_checkpoint_path": str(layer_a_checkpoint_path.resolve()),
                "layer_b_started": False,
            }

        promoted = records_a[: min(cfg.layer_b_top_candidates, len(records_a))]
        layer_b_items = [
            {
                "source_top_mech_winner_a": int(record.get("top_mech_winner_a", record.get("newrank_a", 0))),
                "source_mech_winner": int(record.get("source_mech_winner", 0)),
                "source_stage1_rank": int(record.get("source_stage1_rank", 0)),
                "source_stage1_trial_id": int(record.get("source_stage1_trial_id", record.get("trial_id", 0))),
                "node_label": str(record.get("source_mech_node_label", "")),
                "ks_node_idx": int(record.get("source_ks_node_idx", 0)),
                "cs_node_idx": int(record.get("source_cs_node_idx", 0)),
                "mean_loss_viable": float(record.get("source_mech_mean_loss_viable", float("inf"))),
                "viable_count": int(record.get("source_mech_viable_count", 0)),
                "n_total": int(record.get("source_mech_total_count", 0)),
                "source_stage1_best_seedbank": int(record.get("source_stage1_best_seedbank", 0)),
                "source_stage1_best_initseed": int(record.get("source_stage1_best_initseed", 0)),
                "best_loss": float(record.get("source_stage1_best_loss", record.get("source_stage1_loss", float("inf")))),
                "source_stage1_warmstart_record_path": str(record.get("source_stage1_warmstart_record_path", "")),
            }
            for record in promoted
        ]
        layer_b_assignments = _write_assignments(cfg, LAYER_B, layer_b_items)

        _run_layer_processes(
            cfg,
            layer_name=LAYER_B,
            driver_log=log,
            assignment_paths=layer_b_assignments,
            layer_a_result_path=layer_a_result_path,
        )
        merged_b = merge_prestage2_results(cfg, layer_name=LAYER_B)
        records_b = [rec for rec in merged_b.get("candidate_records", []) if isinstance(rec, dict)]
        _print_layer_ranking(log, layer_name=LAYER_B, records=records_b)
        _log_line(
            log,
            f"driver merge complete | best_candidate_loss={float(merged_b['summary']['best_candidate_loss']):.6e}",
            echo=True,
        )

    return {
        "driver_log": str(driver_log_path),
        "layer_a_result_path": str(_layer_merged_result_path(cfg, LAYER_A).resolve()),
        "merged_result_path": str(_layer_merged_result_path(cfg, LAYER_B).resolve()),
    }


__all__ = [
    "make_driver_log_path",
    "make_shard_log_path",
    "merge_prestage2_results",
    "run_driver",
    "run_prestage2_shard",
]
