from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ARCHIVE_DIRS = (
    "candidate_runs",
    "checkpoints",
    "logs",
    "results",
    "runner",
    "visualization",
    "visualization_runner",
)
TOP_LEVEL_FILES = ("config.py", "main.py", "runner.py", "__init__.py")
PAYLOAD_SUFFIXES = {".pt", ".pkl"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_nbytes(value: Any) -> int | None:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    return None


def _shared_ref(
    *,
    source_path: Path,
    repo_relative_path: Path,
    source_sha256: str,
    removed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "externalized": True,
        "source_path": str(source_path),
        "repo_relative_path": str(repo_relative_path),
        "sha256": source_sha256,
        "npz_keys": {"actuation_times": "t", "F_actuation": "F_actuation"},
        "reason": (
            "Shared AFM05 actuation trajectory is stored once in the dataset "
            "and must not be duplicated per candidate."
        ),
        "removed_arrays": removed,
    }


def _externalize_payload(
    payload: Any,
    *,
    source_path: Path,
    repo_relative_path: Path,
    source_sha256: str,
) -> tuple[int, int]:
    removed_count = 0
    removed_bytes = 0

    def visit(value: Any) -> None:
        nonlocal removed_count, removed_bytes
        if isinstance(value, dict):
            removed: dict[str, dict[str, Any]] = {}
            for key, array_key in (
                ("actuation_times", "t"),
                ("F_actuation", "F_actuation"),
            ):
                if key not in value:
                    continue
                nbytes = _array_nbytes(value[key])
                if nbytes is None:
                    continue
                value.pop(key)
                removed[key] = {"bytes": int(nbytes), "array_key": array_key}
                removed_count += 1
                removed_bytes += int(nbytes)
            if removed:
                value["shared_actuation_ref"] = _shared_ref(
                    source_path=source_path,
                    repo_relative_path=repo_relative_path,
                    source_sha256=source_sha256,
                    removed=removed,
                )
            for child in list(value.values()):
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(payload)
    return removed_count, removed_bytes


def _load_payload(path: Path) -> Any:
    if path.suffix.lower() == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    with path.open("rb") as stream:
        return pickle.load(stream)


def _save_payload(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pt":
        torch.save(payload, path)
    else:
        with path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _copy_tree(source: Path, destination: Path) -> int:
    count = 0
    destination.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return count
    for path in sorted(source.rglob("*"), key=lambda item: str(item).lower()):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix.lower() == ".pyc":
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def _copy_candidate_runs_lightweight(
    source: Path,
    destination: Path,
    *,
    actuation_source: Path,
    actuation_relative: Path,
    actuation_sha256: str,
) -> dict[str, int]:
    stats = {
        "files": 0,
        "payload_files": 0,
        "changed_payload_files": 0,
        "removed_arrays": 0,
        "removed_array_bytes": 0,
        "errors": 0,
    }
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*"), key=lambda item: str(item).lower()):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix.lower() == ".pyc":
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.suffix.lower() in PAYLOAD_SUFFIXES:
                stats["payload_files"] += 1
                payload = _load_payload(path)
                removed_count, removed_bytes = _externalize_payload(
                    payload,
                    source_path=actuation_source,
                    repo_relative_path=actuation_relative,
                    source_sha256=actuation_sha256,
                )
                _save_payload(target, payload)
                if removed_count:
                    stats["changed_payload_files"] += 1
                    stats["removed_arrays"] += int(removed_count)
                    stats["removed_array_bytes"] += int(removed_bytes)
                shutil.copystat(path, target)
            else:
                shutil.copy2(path, target)
            stats["files"] += 1
        except Exception:
            stats["errors"] += 1
            raise
    return stats


def _directory_inventory(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _write_report(
    destination: Path,
    *,
    candidate_stats: dict[str, int],
    actuation_source: Path,
    actuation_relative: Path,
    actuation_sha256: str,
) -> None:
    removed_mb = candidate_stats["removed_array_bytes"] / (1024.0**2)
    text = f"""AFM05 prest2 archive lightweight externalization report
========================================================
archive_root: {destination}
candidate_root: {destination / 'candidate_runs'}
actuation_source_path: {actuation_source}
actuation_repo_relative_path: {actuation_relative}
actuation_sha256: {actuation_sha256}
scanned_payload_files: {candidate_stats['payload_files']}
changed_payload_files: {candidate_stats['changed_payload_files']}
removed_arrays: {candidate_stats['removed_arrays']}
approx_removed_array_bytes_before_reserialization: {candidate_stats['removed_array_bytes']}
approx_removed_array_MB_before_reserialization: {removed_mb:.2f}
errors: {candidate_stats['errors']}

Policy:
Per-candidate payloads no longer duplicate known_pars.actuation_times or known_pars.F_actuation.
The arrays are referenced through known_pars.shared_actuation_ref with source path, repo-relative path, sha256, and npz keys.
"""
    (destination / "lightweight_externalized_shared_actuation_report.txt").write_text(text, encoding="utf-8")


def _write_manifest(
    source: Path,
    destination: Path,
    *,
    inventory: dict[str, tuple[int, int]],
    candidate_count: int,
) -> None:
    lines = [
        "AFM05 prest2 archive manifest",
        "=============================",
        "",
        f"Archive path: {destination}",
        f"Source copied from: {source}",
        f"Archive date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Purpose:",
        "Completed AFM05 prest2 valley-screening package for restart, post-processing, visualization, and downstream stages.",
        "",
        "Included content follows AFM05/Archive/prest2/overall:",
        "candidate_runs/, checkpoints/, logs/, results/, runner/, visualization/, visualization_runner/",
        "config.py, main.py, runner.py, __init__.py",
        "",
        f"Layer A entrant directories: {candidate_count}",
        "",
        "Inventory:",
    ]
    for name in ARCHIVE_DIRS:
        count, nbytes = inventory[name]
        lines.append(f"  {name:24s} {count:6d} files {nbytes / (1024.0**2):10.2f} MB")
    lines.extend(
        [
            "",
            "Lightweight policy:",
            "Per-candidate actuation_times and F_actuation arrays are externalized to the shared AFM05 dataset.",
            "See lightweight_externalized_shared_actuation_report.txt for the source path and SHA-256.",
            "",
        ]
    )
    (destination / "archive_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def archive(source: Path, destination: Path, repo_root: Path) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if any(destination.iterdir()) if destination.exists() else False:
        raise RuntimeError(f"Destination must be empty before archival: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    actuation_source = (repo_root / "AFM05" / "datasets" / "e0.0" / "data" / "afm05_F_actuation.npz").resolve()
    if not actuation_source.is_file():
        raise FileNotFoundError(actuation_source)
    actuation_relative = actuation_source.relative_to(repo_root)
    actuation_sha256 = _sha256(actuation_source)

    for name in ARCHIVE_DIRS:
        (destination / name).mkdir(parents=True, exist_ok=True)
    for name in TOP_LEVEL_FILES:
        shutil.copy2(source / name, destination / name)

    candidate_stats = _copy_candidate_runs_lightweight(
        source / "candidate_runs",
        destination / "candidate_runs",
        actuation_source=actuation_source,
        actuation_relative=actuation_relative,
        actuation_sha256=actuation_sha256,
    )
    for name in ARCHIVE_DIRS:
        if name == "candidate_runs":
            continue
        _copy_tree(source / name, destination / name)

    _write_report(
        destination,
        candidate_stats=candidate_stats,
        actuation_source=actuation_source,
        actuation_relative=actuation_relative,
        actuation_sha256=actuation_sha256,
    )
    inventory = {name: _directory_inventory(destination / name) for name in ARCHIVE_DIRS}
    candidate_origin = destination / "candidate_runs" / "stage1_origin"
    candidate_count = sum(path.is_dir() for path in candidate_origin.iterdir()) if candidate_origin.exists() else 0
    _write_manifest(
        source,
        destination,
        inventory=inventory,
        candidate_count=candidate_count,
    )
    summary = {
        "source": str(source),
        "destination": str(destination),
        "candidate_stats": candidate_stats,
        "inventory": {name: {"files": values[0], "bytes": values[1]} for name, values in inventory.items()},
        "candidate_count": candidate_count,
        "actuation_sha256": actuation_sha256,
    }
    return summary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=repo_root / "AFM05" / "prestage2")
    parser.add_argument("--destination", type=Path, default=repo_root / "AFM05" / "Archive" / "prest2" / "valley")
    args = parser.parse_args()
    summary = archive(args.source, args.destination, repo_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
