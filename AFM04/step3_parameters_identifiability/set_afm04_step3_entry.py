"""
Select the current AFM04 stage2light local point for step3.

This script is the switchboard for AFM04 step3. It points step3 at one
completed st2l rank/candidate archive, writes/refreshes that rank's
step3_entry_manifest.json, validates the payload, and writes
current_step3_entry.json in this directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM04.step3_parameters_identifiability.afm04_stage2light_step3_entry import (  # noqa: E402
    as_repo_rel,
    build_manifest,
    jsonable,
    load_torch_payload,
    manifest_path,
    parse_rank_candidate_label,
    resolve_manifest_path,
    validate_manifest,
    write_manifest,
)


CURRENT_ENTRY_SCHEMA = "afm04_step3_current_entry_v1"
DEFAULT_ARCHIVE_ROOT = "AFM04/archive/st2l/3_10x10x200_W0 new"
DEFAULT_OUTPUT = "AFM04/step3_parameters_identifiability/results/current_step3_entry.json"


def parse_candidate(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower().startswith("candidate"):
        text = text[len("candidate") :]
    if text.lower().startswith("b"):
        text = text[1:]
    return int(text)


def maybe_number(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def list_rank_dirs(root: Path) -> list[Path]:
    if root.name.lower().startswith("rank"):
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Archive root does not exist: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("rank"))


def entry_payload_path(rank_dir: Path, entry_payload_kind: str) -> Path:
    if entry_payload_kind == "result":
        return rank_dir / "result" / "stage2light_result_p1.pt"
    if entry_payload_kind == "best":
        return rank_dir / "checkpoint" / "stage2light_best_p1.pt"
    if entry_payload_kind == "checkpoint":
        return rank_dir / "checkpoint" / "stage2light_checkpoint_p1.pt"
    raise ValueError(f"Unsupported entry_payload_kind: {entry_payload_kind}")


def summarize_candidate(rank_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    rank_from_label, candidate_from_label = parse_rank_candidate_label(rank_dir.name)
    warmstart = payload.get("prestage2_warmstart")
    if not isinstance(warmstart, dict):
        warmstart = {}
    resume_identity = payload.get("resume_identity")
    if not isinstance(resume_identity, dict):
        resume_identity = {}

    candidate_b = maybe_number(warmstart.get("candidate_b"))
    if candidate_b is None:
        candidate_b = maybe_number(warmstart.get("candidate"))
    if candidate_b is None:
        candidate_b = maybe_number(resume_identity.get("prestage2_candidate"))
    if candidate_b is None:
        candidate_b = candidate_from_label

    rank = maybe_number(warmstart.get("original_rank"))
    if rank is None:
        rank = maybe_number(warmstart.get("source_stage1_rank"))
    if rank is None:
        rank = rank_from_label

    return {
        "rank_dir": rank_dir,
        "rank_label": rank_dir.name,
        "rank": rank,
        "candidate_b": candidate_b,
        "source_mech_winner": maybe_number(warmstart.get("source_mech_winner")),
        "trial_id": maybe_number(warmstart.get("trial_id")) or maybe_number(resume_identity.get("prestage2_trial_id")),
        "seed": maybe_number(warmstart.get("seed")),
        "warmstart_source": payload.get("warmstart_source"),
        "st2l_start_policy": warmstart.get("st2l_start_policy"),
        "final_val_epoch": payload.get("final_val_epoch"),
        "final_val_loss": payload.get("final_val_loss"),
        "best_epoch": payload.get("best_epoch"),
        "best_val_loss": payload.get("best_val_loss"),
        "stop_kind": payload.get("stop_kind"),
        "stop_epoch": payload.get("stop_epoch"),
        "stop_reason": payload.get("stop_reason"),
    }


def completed_result_status(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for key in ("final_state_dict", "final_mech_state_dict", "state_mean", "state_scale", "known_pars", "window_meta"):
        if key not in payload:
            missing.append(key)
    final_epoch = payload.get("final_val_epoch")
    if final_epoch is None:
        missing.append("final_val_epoch")
    else:
        try:
            if int(final_epoch) < 0:
                missing.append("final_val_epoch>=0")
        except (TypeError, ValueError):
            missing.append("final_val_epoch:int")
    return len(missing) == 0, missing


def scan_entries(root: Path, entry_payload_kind: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for rank_dir in list_rank_dirs(root):
        payload_path = entry_payload_path(rank_dir, entry_payload_kind)
        result_path = entry_payload_path(rank_dir, "result")
        if not payload_path.exists() or not result_path.exists():
            continue
        result_payload = load_torch_payload(result_path)
        complete, missing = completed_result_status(result_payload)
        entry_payload = result_payload if entry_payload_kind == "result" else load_torch_payload(payload_path)
        info = summarize_candidate(rank_dir, result_payload)
        info.update(
            {
                "entry_payload_kind": entry_payload_kind,
                "entry_payload": payload_path,
                "result_payload": result_path,
                "complete_st2l_result": complete,
                "missing_for_complete_result": missing,
                "entry_payload_has_state": ("final_state_dict" in entry_payload or "state_dict" in entry_payload),
            }
        )
        entries.append(info)
    return entries


def select_entry(
    entries: list[dict[str, Any]],
    *,
    rank: int | None,
    candidate: int | None,
    rank_dir: Path | None,
) -> dict[str, Any]:
    if rank_dir is not None:
        rank_dir_resolved = rank_dir.resolve()
        matches = [e for e in entries if Path(e["rank_dir"]).resolve() == rank_dir_resolved]
    elif rank is not None and candidate is not None:
        matches = [e for e in entries if e.get("rank") == rank and e.get("candidate_b") == candidate]
    elif rank is not None:
        matches = [e for e in entries if e.get("rank") == rank]
    elif candidate is not None:
        matches = [e for e in entries if e.get("candidate_b") == candidate]
    else:
        raise ValueError("Specify one of --rank-dir, --rank, or --candidate.")

    if not matches:
        available = ", ".join(
            f"{e['rank_label']}(rank={e.get('rank')},B={e.get('candidate_b')})" for e in entries
        )
        raise LookupError(f"No matching completed st2l entry found. Available: {available}")
    if len(matches) > 1:
        available = ", ".join(
            f"{e['rank_label']}(rank={e.get('rank')},B={e.get('candidate_b')})" for e in matches
        )
        raise LookupError(f"Selection is ambiguous; use --rank-dir. Matches: {available}")
    selected = matches[0]
    if not selected.get("complete_st2l_result"):
        raise RuntimeError(
            "Selected object does not have a complete st2l result payload; "
            f"missing={selected.get('missing_for_complete_result')}"
        )
    return selected


def write_current_entry(selected: dict[str, Any], manifest_file: Path, output: Path) -> Path:
    payload = load_torch_payload(Path(selected["result_payload"]))
    warmstart = payload.get("prestage2_warmstart")
    if not isinstance(warmstart, dict):
        warmstart = {}
    current = {
        "schema_version": CURRENT_ENTRY_SCHEMA,
        "purpose": "Current AFM04 step3 entry selection.",
        "source_stage": "stage2light",
        "rank_label": selected.get("rank_label"),
        "rank": selected.get("rank"),
        "candidate_b": selected.get("candidate_b"),
        "source_mech_winner": selected.get("source_mech_winner"),
        "trial_id": selected.get("trial_id"),
        "seed": selected.get("seed"),
        "entry_payload_kind": selected.get("entry_payload_kind"),
        "rank_dir": as_repo_rel(Path(selected["rank_dir"])),
        "manifest": as_repo_rel(manifest_file),
        "entry_payload": as_repo_rel(Path(selected["entry_payload"])),
        "result_payload": as_repo_rel(Path(selected["result_payload"])),
        "data_files": {
            "ode_data": build_manifest(Path(selected["rank_dir"]), selected["entry_payload_kind"])["paths"]["ode_data"],
            "pert_df": build_manifest(Path(selected["rank_dir"]), selected["entry_payload_kind"])["paths"]["pert_df"],
        },
        "st2l_result_status": {
            "complete": selected.get("complete_st2l_result"),
            "final_val_epoch": jsonable(selected.get("final_val_epoch")),
            "final_val_loss": jsonable(selected.get("final_val_loss")),
            "best_epoch": jsonable(selected.get("best_epoch")),
            "best_val_loss": jsonable(selected.get("best_val_loss")),
            "stop_kind": jsonable(selected.get("stop_kind")),
            "stop_epoch": jsonable(selected.get("stop_epoch")),
            "stop_reason": jsonable(selected.get("stop_reason")),
        },
        "warmstart_summary": {
            "source": warmstart.get("source"),
            "label": warmstart.get("label"),
            "st2l_start_policy": warmstart.get("st2l_start_policy"),
            "prest2_final_state_used": warmstart.get("prest2_final_state_used"),
            "ks0": jsonable(warmstart.get("ks0")),
            "cs0": jsonable(warmstart.get("cs0")),
        },
        "note": "Replace the step3 object by rerunning set_afm04_step3_entry.py with another --rank/--candidate/--rank-dir.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def print_entries(entries: list[dict[str, Any]]) -> None:
    for e in entries:
        status = "complete" if e.get("complete_st2l_result") else "incomplete"
        print(
            f"{e['rank_label']}: rank={e.get('rank')} B={e.get('candidate_b')} "
            f"final_epoch={e.get('final_val_epoch')} best_epoch={e.get('best_epoch')} "
            f"stop={e.get('stop_kind') or 'finished'} status={status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--rank-dir", default=None, help="Concrete completed st2l rank/candidate directory.")
    parser.add_argument("--rank", type=int, default=None, help="Select by original/stage1 rank.")
    parser.add_argument("--candidate", default=None, help="Select by candidate B index, e.g. B11 or 11.")
    parser.add_argument(
        "--entry-payload-kind",
        choices=["result", "best", "checkpoint"],
        default="result",
        help="Which payload future step3 code should analyze.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true", help="List available completed st2l entries and exit.")
    parser.add_argument("--no-write-rank-manifest", action="store_true")
    args = parser.parse_args()

    archive_root = resolve_manifest_path(args.archive_root)
    entries = scan_entries(archive_root, args.entry_payload_kind)
    if args.list:
        print_entries(entries)
        return

    selected = select_entry(
        entries,
        rank=args.rank,
        candidate=parse_candidate(args.candidate),
        rank_dir=resolve_manifest_path(args.rank_dir) if args.rank_dir else None,
    )
    selected_rank_dir = Path(selected["rank_dir"])
    if args.no_write_rank_manifest:
        manifest_file = manifest_path(selected_rank_dir)
    else:
        manifest_file = write_manifest(selected_rank_dir, entry_payload_kind=args.entry_payload_kind)
    check = validate_manifest(manifest_file)
    output = resolve_manifest_path(args.output)
    current_file = write_current_entry(selected, manifest_file, output)
    print(f"selected rank={selected.get('rank')} candidate B={selected.get('candidate_b')} dir={as_repo_rel(selected_rank_dir)}")
    print(f"manifest ok: {check['required_files_ok']} payload keys ok: {check['required_payload_keys_ok']}")
    print(f"wrote {as_repo_rel(current_file)}")


if __name__ == "__main__":
    main()
