"""Archive one completed AFM05 step3 run without latest/shard duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]


def resolve_path(text: str) -> Path:
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def copy_file(src: Path, dst: Path, archive_root: Path, records: list[dict[str, Any]]) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    records.append(
        {
            "archive_path": dst.relative_to(archive_root).as_posix(),
            "source": str(src),
            "bytes": int(dst.stat().st_size),
            "sha256": sha256(dst),
        }
    )


def newest_since(directory: Path, pattern: str, started_at: float) -> Path:
    matches = [
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.stat().st_mtime >= started_at - 2.0
    ]
    if not matches:
        raise FileNotFoundError(f"No newly generated file matching {pattern!r} under {directory}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-dir", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--run-name", required=True, help="Canonical BXX_rankXXX name.")
    parser.add_argument("--parameter-set", default="trained", choices=("trained", "mech"))
    parser.add_argument("--parameter-limit", type=int, default=0)
    parser.add_argument("--started-at", type=float, required=True, help="Step3 start time as Unix seconds.")
    args = parser.parse_args()

    rank_dir = resolve_path(args.rank_dir)
    archive_dir = resolve_path(args.archive_dir)
    run_name = args.run_name.strip()
    if rank_dir.name.lower() != run_name.lower():
        raise ValueError(f"Rank directory name {rank_dir.name!r} does not match run name {run_name!r}")

    step3_root = THIS_DIR
    results_dir = step3_root / "results"
    post_dir = results_dir / "post_analysis"
    logs_dir = step3_root / "logs"
    visualization_dir = step3_root / "visualization"
    entry_manifest = rank_dir / "step3_entry_manifest.json"

    parameter_tag = args.parameter_set
    if args.parameter_limit > 0:
        parameter_tag = f"{parameter_tag}_limit{args.parameter_limit}"
    ident_latest = results_dir / f"afm05_step3_identifiability_{parameter_tag}_latest.json"
    ident_meta = load_json(ident_latest)
    if str(ident_meta.get("rank_label", "")).lower() != run_name.lower():
        raise RuntimeError(
            f"Latest identifiability result belongs to {ident_meta.get('rank_label')!r}, expected {run_name!r}"
        )

    post_latest = post_dir / "afm05_step3_post_analysis_latest.json"
    post_meta = load_json(post_latest)
    if str(post_meta.get("rank_label", "")).lower() != run_name.lower():
        raise RuntimeError(
            f"Latest post-analysis result belongs to {post_meta.get('rank_label')!r}, expected {run_name!r}"
        )

    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []

    copy_file(entry_manifest, archive_dir / "conditional_dependency" / "step3_entry_manifest.json", archive_dir, records)
    for suffix in (".csv", ".json", ".npz", ".txt"):
        src = ident_latest.with_suffix(suffix)
        copy_file(src, archive_dir / "results" / "identifiability" / f"identifiability{suffix}", archive_dir, records)
    for suffix in (".csv", ".json", ".txt"):
        src = post_latest.with_suffix(suffix)
        copy_file(src, archive_dir / "results" / "post_analysis" / f"post_analysis{suffix}", archive_dir, records)

    plot_suffixes = (
        "eigen_spectrum.png",
        "null_projection_groups.png",
        "threshold_sweep.png",
    )
    for suffix in plot_suffixes:
        src = newest_since(visualization_dir, f"afm05_step3_post_analysis_*_{suffix}", args.started_at)
        copy_file(src, archive_dir / "visualization" / f"afm05_step3_{run_name}_{suffix}", archive_dir, records)

    log_sources = sorted(
        path
        for path in logs_dir.glob("afm05_step3_*.txt")
        if path.is_file() and path.stat().st_mtime >= args.started_at - 2.0
    )
    if not log_sources:
        raise FileNotFoundError(f"No step3 logs generated after {args.started_at} under {logs_dir}")
    for src in log_sources:
        copy_file(src, archive_dir / "logs" / src.name, archive_dir, records)

    manifest = {
        "schema": "afm05_step3_compact_archive_v1",
        "complete": True,
        "created_at_unix": time.time(),
        "run_name": run_name,
        "candidate_b": ident_meta.get("candidate_b"),
        "rank": ident_meta.get("rank"),
        "parameter_set": ident_meta.get("parameter_set"),
        "source_stage2light_archive": str(rank_dir),
        "intermediate_parameter_shards_archived": False,
        "intermediate_parameter_shards_note": (
            "Omitted because the merged identifiability JSON/NPZ is the complete downstream result."
        ),
        "latest_alias_duplicates_archived": False,
        "files": records,
    }
    (archive_dir / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"AFM05 step3 compact archive completed -> {archive_dir}")
    print(f"Archived files -> {len(records)}")


if __name__ == "__main__":
    main()
