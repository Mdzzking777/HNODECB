"""
AFM05 stage2light archive entry helper for step3 identifiability.

The original HNODECB step3 scripts read Julia .jld local-optimum payloads.
AFM05 stage2light archives are PyTorch payloads, so this helper defines a
small manifest format and a loader/validator around the existing archive
layout. It does not run identifiability by itself; it makes the step2 local
point explicit and reproducible for the AFM05 step3 implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np


SCHEMA_VERSION = "afm05_stage2light_step3_entry_v1"


RANK_CANDIDATE_PATTERNS = (
    re.compile(r"B(?P<candidate>\d+)[_-]?rank(?P<rank>\d+)", re.IGNORECASE),
    re.compile(r"rank(?P<rank>\d+)(?:[_-]?B(?P<candidate>\d+))?", re.IGNORECASE),
)


@dataclass(frozen=True)
class EntryPaths:
    rank_dir: Path
    result_payload: Path
    result_viz_payload: Path
    ode_data: Path
    pert_df: Path

    @classmethod
    def from_rank_dir(cls, rank_dir: Path) -> "EntryPaths":
        return cls(
            rank_dir=rank_dir,
            result_payload=rank_dir / "result" / "stage2light_result_p1.pt",
            result_viz_payload=rank_dir / "result" / "stage2light_result_p1.viz.pkl",
            ode_data=rank_dir / "data" / "ode_data_afm_dmt_kv.npz",
            pert_df=rank_dir / "data" / "pert_df_afm_dmt_kv.npz",
        )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def as_repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_torch_payload(path: Path) -> dict[str, Any]:
    # PyTorch 2.6 defaults weights_only=True; AFM05 payloads contain metadata.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict payload in {path}, got {type(payload)!r}")
    return payload


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return as_repo_rel(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def compact_state_dict(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {"available": False}
    return {
        "available": True,
        "keys": sorted(str(k) for k in state.keys()),
        "values": jsonable(state),
    }


def parse_rank_candidate_label(label: str) -> tuple[int | None, int | None]:
    for pattern in RANK_CANDIDATE_PATTERNS:
        match = pattern.search(str(label))
        if match:
            rank = int(match.group("rank")) if match.group("rank") is not None else None
            candidate = int(match.group("candidate")) if match.group("candidate") is not None else None
            return rank, candidate
    return None, None


def payload_candidate_meta(payload: dict[str, Any]) -> dict[str, Any]:
    warmstart = payload.get("prestage2_warmstart")
    if not isinstance(warmstart, dict):
        warmstart = {}
    resume_identity = payload.get("resume_identity")
    if not isinstance(resume_identity, dict):
        resume_identity = {}
    stage1 = payload.get("stage1_warmstart")
    if not isinstance(stage1, dict):
        stage1 = {}
    return {
        "candidate_b": jsonable(warmstart.get("candidate_b", warmstart.get("candidate", resume_identity.get("prestage2_candidate")))),
        "original_rank": jsonable(warmstart.get("original_rank", warmstart.get("source_stage1_rank", stage1.get("rank")))),
        "source_stage1_rank": jsonable(warmstart.get("source_stage1_rank", stage1.get("rank"))),
        "source_mech_winner": jsonable(warmstart.get("source_mech_winner", stage1.get("mech_winner"))),
        "trial_id": jsonable(warmstart.get("trial_id", resume_identity.get("prestage2_trial_id", stage1.get("trial_id")))),
        "seed": jsonable(warmstart.get("seed", stage1.get("seed"))),
        "label": jsonable(warmstart.get("label", stage1.get("label"))),
        "st2l_start_policy": jsonable(warmstart.get("st2l_start_policy")),
        "prest2_final_state_used": jsonable(warmstart.get("prest2_final_state_used")),
    }


def compact_known_pars(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    out: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (np.ndarray, torch.Tensor, list, tuple)):
            arr = np.asarray(item.detach().cpu() if isinstance(item, torch.Tensor) else item)
            raw = np.ascontiguousarray(arr).view(np.uint8)
            out[str(key)] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        else:
            out[str(key)] = jsonable(item)
    return out


def payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    keys_of_interest = [
        "epoch",
        "final_val_epoch",
        "final_val_loss",
        "stop_kind",
        "stop_epoch",
        "stop_reason",
        "failure_epoch",
        "failure_reason",
        "warmstart_source",
        "optimizer_phase",
        "validation_eval_mode",
    ]
    summary = {k: jsonable(payload.get(k)) for k in keys_of_interest if k in payload}
    summary["window_meta"] = jsonable(payload.get("window_meta", {}))
    summary["x3_refit_meta"] = jsonable(payload.get("x3_refit_meta", {}))
    summary["initial_grid_support_meta"] = jsonable(payload.get("initial_grid_support_meta", {}))
    summary["known_pars"] = compact_known_pars(payload.get("known_pars", {}))
    summary["has_state_dict"] = "state_dict" in payload or "final_state_dict" in payload
    summary["has_mech_state_dict"] = "mech_state_dict" in payload or "final_mech_state_dict" in payload
    summary["has_normalizer"] = "state_mean" in payload and "state_scale" in payload
    summary["history_length"] = len(payload.get("history", [])) if isinstance(payload.get("history"), list) else None
    return summary


def build_manifest(rank_dir: Path, entry_payload_kind: str = "result") -> dict[str, Any]:
    paths = EntryPaths.from_rank_dir(rank_dir)
    if entry_payload_kind != "result":
        raise ValueError("AFM05 Step 3 analyzes the final-epoch result payload only")
    result_payload = load_torch_payload(paths.result_payload)

    rank_name = rank_dir.name
    rank_value, candidate_from_label = parse_rank_candidate_label(rank_name)
    candidate_meta = payload_candidate_meta(result_payload)
    candidate_value = candidate_meta.get("candidate_b")
    if candidate_value is None:
        candidate_value = candidate_from_label

    final_mech = result_payload.get("final_mech_state_dict") or result_payload.get("mech_state_dict")

    return {
        "schema_version": SCHEMA_VERSION,
        "source_project": "AFM05",
        "source_stage": "stage2light",
        "source_rank_dir": as_repo_rel(rank_dir),
        "rank_label": rank_name,
        "rank": rank_value,
        "candidate_b": candidate_value,
        "candidate_meta": candidate_meta,
        "entry_payload_kind": entry_payload_kind,
        "entry_payload": as_repo_rel(paths.result_payload),
        "paths": {
            "result_payload": as_repo_rel(paths.result_payload),
            "result_viz_payload": as_repo_rel(paths.result_viz_payload),
            "ode_data": as_repo_rel(paths.ode_data),
            "pert_df": as_repo_rel(paths.pert_df),
        },
        "result_summary": payload_summary(result_payload),
        "mechanistic_parameters": {
            "final": compact_state_dict(final_mech),
        },
        "normalizer": {
            "state_mean": jsonable(result_payload.get("state_mean")),
            "state_scale": jsonable(result_payload.get("state_scale")),
        },
        "step3_notes": [
            "Use the result payload and final_state_dict/final_mech_state_dict for the final-epoch local point.",
            "AFM05 Step 3 uses x1, x2, and x2dot only as observable trajectories.",
            "x3_compat is used only for the rollout initial condition and is never treated as observed truth.",
            "This manifest is generated from existing st2l artifacts; no st2l rerun is required.",
        ],
    }


def manifest_path(rank_dir: Path) -> Path:
    return rank_dir / "step3_entry_manifest.json"


def write_manifest(rank_dir: Path, entry_payload_kind: str = "result") -> Path:
    manifest = build_manifest(rank_dir, entry_payload_kind=entry_payload_kind)
    out = manifest_path(rank_dir)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unexpected schema_version in {path}: {manifest.get('schema_version')!r}")
    return manifest


def resolve_manifest_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root() / path


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = load_manifest(path)
    required_path_keys = ["entry_payload", "ode_data", "pert_df"]
    resolved_paths: dict[str, Path] = {
        "entry_payload": resolve_manifest_path(manifest["entry_payload"]),
        "ode_data": resolve_manifest_path(manifest["paths"]["ode_data"]),
        "pert_df": resolve_manifest_path(manifest["paths"]["pert_df"]),
    }
    missing = [name for name in required_path_keys if not resolved_paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing required step3 files for {path}: {missing}")

    payload = load_torch_payload(resolved_paths["entry_payload"])
    required_keys = [
        "state_mean",
        "state_scale",
        "known_pars",
        "window_meta",
        "final_state_dict",
        "final_mech_state_dict",
        "final_snapshot",
    ]
    absent = [key for key in required_keys if key not in payload]
    if absent:
        raise KeyError(f"Entry payload {resolved_paths['entry_payload']} is missing keys: {absent}")
    return {
        "manifest": str(path),
        "entry_payload": str(resolved_paths["entry_payload"]),
        "entry_payload_kind": manifest.get("entry_payload_kind"),
        "rank": manifest.get("rank"),
        "required_files_ok": True,
        "required_payload_keys_ok": True,
    }


def iter_rank_dirs(root: Path) -> list[Path]:
    if (root / "result" / "stage2light_result_p1.pt").is_file():
        return [root]
    return sorted(
        p.parent.parent
        for p in root.rglob("result/stage2light_result_p1.pt")
        if p.parent.parent.is_dir()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        default=r"AFM05/Archive/st2l/valley",
        help="A st2l archive parent directory or one concrete rank directory.",
    )
    parser.add_argument(
        "--entry-payload-kind",
        choices=["result"],
        default="result",
        help="Which payload step3 should treat as the local-point entry.",
    )
    parser.add_argument("--write", action="store_true", help="Write step3_entry_manifest.json into each rank directory.")
    parser.add_argument("--check", action="store_true", help="Validate existing or newly written manifests.")
    args = parser.parse_args()

    root = resolve_manifest_path(args.archive_root)
    rank_dirs = iter_rank_dirs(root)
    if not rank_dirs:
        raise FileNotFoundError(f"No rank directories found under {root}")

    for rank_dir in rank_dirs:
        if args.write:
            path = write_manifest(rank_dir, entry_payload_kind=args.entry_payload_kind)
            print(f"wrote {as_repo_rel(path)}")
        else:
            path = manifest_path(rank_dir)
            print(f"manifest {as_repo_rel(path)}")
        if args.check:
            result = validate_manifest(path)
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
