from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


_RESULT_PT_RE = re.compile(r"stage2light_result_p(\d+)\.pt$")

_ARCHIVE_SUBDIRS = (
    "result",
    "checkpoint",
    "logs",
    "data",
    "conditional dependency",
    "visualization",
)


def _timestamp_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log_line(log, message: str) -> None:
    if log is None:
        print(message, flush=True)
        return
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.write(f"[{stamp}] {message}\n")
    log.flush()


def _sorted_paths(paths: list[Path], pattern: re.Pattern[str] | None = None) -> list[Path]:
    if pattern is None:
        return sorted(paths, key=lambda p: p.name.lower())

    ranked: list[tuple[int, str, Path]] = []
    for path in paths:
        match = pattern.match(path.name)
        idx = int(match.group(1)) if match is not None else 10**9
        ranked.append((idx, path.name.lower(), path))
    return [path for _, _, path in sorted(ranked)]


def _discover_result_pt_paths(result_dir: Path) -> list[Path]:
    return _sorted_paths(list(result_dir.glob("stage2light_result_p*.pt")), _RESULT_PT_RE)


def _load_primary_result_payload(cfg) -> dict[str, Any]:
    result_paths = _discover_result_pt_paths(cfg.result_dir)
    if not result_paths:
        raise FileNotFoundError(f"No stage2light_result_p*.pt files found under: {cfg.result_dir}")
    payload = torch.load(result_paths[0], map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected result payload type: {type(payload)!r}")
    return payload


def _infer_stage1_rank(payload: dict[str, Any], cfg) -> int:
    for meta_key in ("stage1_warmstart", "prestage2_warmstart", "warmstart"):
        meta = payload.get(meta_key)
        if not isinstance(meta, dict):
            continue
        for key in ("rank", "original_rank", "source_stage1_rank", "stage1_rank"):
            if meta.get(key) is not None:
                return int(meta[key])

    warmstart_source = str(payload.get("warmstart_source", cfg.warmstart_source)).strip().lower()
    if warmstart_source == "stage1":
        return int(cfg.stage1_input_rank)
    return int(cfg.prestage2_input_candidate)


def _resolve_archive_family_dir(repo_root: Path) -> Path:
    archive_root = repo_root / "AFM05" / "archive" / "st2l"
    archive_root.mkdir(parents=True, exist_ok=True)

    preferred = archive_root / "2_10x10x500_W0"
    if preferred.exists():
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred

    family_pattern = re.compile(r"^\d+_.+")
    family_dirs = [
        path
        for path in archive_root.iterdir()
        if path.is_dir() and family_pattern.match(path.name)
    ]
    if len(family_dirs) == 1:
        return family_dirs[0]
    return archive_root


def _resolve_archive_run_dir(family_dir: Path, *, stage1_rank: int) -> Path:
    base = family_dir / f"rank{stage1_rank}"
    if not base.exists():
        return base

    if not any((base / subdir).exists() for subdir in _ARCHIVE_SUBDIRS) and not any(base.iterdir()):
        return base

    return family_dir / f"rank{stage1_rank}_{_timestamp_compact()}"


def _resolve_explicit_archive_run_dir(cfg) -> Path | None:
    raw = os.environ.get("HNODECB_AFM05_STAGE2LIGHT_ARCHIVE_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = cfg.repo_root / path
    return path.resolve()


def _ensure_archive_layout(archive_dir: Path, *, compact: bool = False) -> dict[str, Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {"root": archive_dir}
    subdirs = tuple(item for item in _ARCHIVE_SUBDIRS if not (compact and item == "conditional dependency"))
    for subdir in subdirs:
        path = archive_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        out[subdir] = path
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_shared_data_file(src: Path, *, archive_dir: Path, data_dir: Path) -> dict[str, Any] | None:
    if not src.is_file():
        return None
    digest = _sha256(src)
    shared_dir = archive_dir.parent / "_shared" / "data"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_name = f"{src.stem}_{digest[:16]}{src.suffix}"
    shared_path = shared_dir / shared_name
    if not shared_path.is_file():
        shutil.copy2(src, shared_path)

    destination = data_dir / src.name
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    storage = "hardlink"
    try:
        os.link(shared_path, destination)
    except OSError:
        shutil.copy2(shared_path, destination)
        storage = "copy_fallback"
    return {
        "name": src.name,
        "sha256": digest,
        "bytes": int(src.stat().st_size),
        "shared_path": str(shared_path),
        "storage": storage,
    }


def _copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> int:
    count = 0
    for path in sorted(src_dir.glob(pattern), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        shutil.copy2(path, dst_dir / path.name)
        count += 1
    return count


def _copy_file(src: Path, dst_dir: Path) -> int:
    if not src.is_file():
        return 0
    shutil.copy2(src, dst_dir / src.name)
    return 1


def _run_all_visualizations(cfg, log) -> None:
    script_path = cfg.repo_root / "AFM05" / "stage2light" / "runner" / "visualization" / "run_afm_stage2light_all_visualizations_05.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing visualization runner: {script_path}")

    _log_line(log, f"archive: running visualizations | script={script_path}")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(cfg.repo_root),
        env=dict(os.environ),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        if proc.stdout.strip():
            for line in proc.stdout.splitlines():
                _log_line(log, f"archive viz stdout: {line}")
        if proc.stderr.strip():
            for line in proc.stderr.splitlines():
                _log_line(log, f"archive viz stderr: {line}")
        raise RuntimeError(
            "stage2light visualization runner failed "
            f"(exit={proc.returncode})"
        )
    if proc.stdout.strip():
        tail = proc.stdout.splitlines()[-1].strip()
        if tail:
            _log_line(log, f"archive viz: {tail}")


def archive_completed_stage2light_run(cfg, *, log=None) -> dict[str, Any]:
    payload = _load_primary_result_payload(cfg)
    stage1_rank = _infer_stage1_rank(payload, cfg)
    compact = os.environ.get("HNODECB_AFM05_STAGE2LIGHT_ARCHIVE_COMPACT", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    archive_dir = _resolve_explicit_archive_run_dir(cfg)
    if archive_dir is None:
        family_dir = _resolve_archive_family_dir(cfg.repo_root)
        archive_dir = _resolve_archive_run_dir(family_dir, stage1_rank=stage1_rank)
    else:
        _log_line(log, f"archive: using explicit candidate directory | dir={archive_dir}")
    layout = _ensure_archive_layout(archive_dir, compact=compact)
    if compact:
        stale_conditional = archive_dir / "conditional dependency"
        if stale_conditional.is_dir():
            shutil.rmtree(stale_conditional)
        for pattern in ("stage2light_best_p*.viz.pkl", "stage2light_final_p*.pt", "stage2light_final_p*.viz.pkl"):
            for stale in layout["checkpoint"].glob(pattern):
                if stale.is_file():
                    stale.unlink()

    _run_all_visualizations(cfg, log)

    result_count = 0
    result_count += _copy_glob(cfg.result_dir, "stage2light_result_p*.viz.pkl", layout["result"])
    result_count += _copy_glob(cfg.result_dir, "stage2light_result_p*.pt", layout["result"])
    result_count += _copy_glob(cfg.checkpoint_dir, "stage2light_best_p*.viz.pkl", layout["result"])

    checkpoint_count = 0
    checkpoint_count += _copy_glob(cfg.checkpoint_dir, "stage2light_checkpoint_p*.pt", layout["checkpoint"])
    checkpoint_count += _copy_glob(cfg.checkpoint_dir, "stage2light_best_p*.pt", layout["checkpoint"])
    if not compact:
        checkpoint_count += _copy_glob(cfg.checkpoint_dir, "stage2light_best_p*.viz.pkl", layout["checkpoint"])
        checkpoint_count += _copy_glob(cfg.checkpoint_dir, "stage2light_final_p*.pt", layout["checkpoint"])
        checkpoint_count += _copy_glob(cfg.checkpoint_dir, "stage2light_final_p*.viz.pkl", layout["checkpoint"])

    log_count = _copy_glob(cfg.shard_log_dir, "log2_05_step2a_stage2light_local_p*.txt", layout["logs"])

    data_dir = cfg.dataset_root / cfg.error_level / "data"
    data_count = 0
    shared_data: list[dict[str, Any]] = []
    if compact:
        for name in ("ode_data_afm_dmt_kv.npz", "pert_df_afm_dmt_kv.npz"):
            record = _archive_shared_data_file(data_dir / name, archive_dir=archive_dir, data_dir=layout["data"])
            if record is not None:
                shared_data.append(record)
        data_count = len(shared_data)
        (layout["data"] / "shared_data_manifest.json").write_text(
            json.dumps({"schema": "afm05_shared_archive_data_v1", "files": shared_data}, indent=2) + "\n",
            encoding="utf-8",
        )
        cond_count = 0
    else:
        data_count += _copy_file(data_dir / "ode_data_afm_dmt_kv.npz", layout["data"])
        data_count += _copy_file(data_dir / "pert_df_afm_dmt_kv.npz", layout["data"])
        cond_count = _copy_glob(cfg.result_dir, "stage2light_result_p*.pt", layout["conditional dependency"])
    viz_count = _copy_glob(cfg.visualization_dir, "afm_param_stage2light_05_*", layout["visualization"])

    summary = {
        "complete": True,
        "archive_dir": str(archive_dir),
        "stage1_rank": int(stage1_rank),
        "result_files": int(result_count),
        "checkpoint_files": int(checkpoint_count),
        "log_files": int(log_count),
        "data_files": int(data_count),
        "conditional_dependency_files": int(cond_count),
        "visualization_files": int(viz_count),
        "compact": bool(compact),
        "shared_data": shared_data,
    }
    if compact:
        (archive_dir / "archive_manifest.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    _log_line(
        log,
        "archive complete | "
        f"dir={archive_dir} | "
        f"rank={stage1_rank} | "
        f"result={result_count} checkpoint={checkpoint_count} logs={log_count} data={data_count} "
        f"conditional={cond_count} viz={viz_count}",
    )
    return summary


__all__ = ["archive_completed_stage2light_run"]
