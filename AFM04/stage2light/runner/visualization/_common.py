from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
RESULT_DIR = REPO_ROOT / "AFM04" / "stage2light" / "results"
OUT_DIR = REPO_ROOT / "AFM04" / "stage2light" / "logs" / "visualization"
LOG_DIR = REPO_ROOT / "AFM04" / "stage2light" / "logs" / "window_per_shard"

_RESULT_VIZ_RE = re.compile(r"stage2light_result_p(\d+)\.viz\.pkl$")
_RESULT_PT_RE = re.compile(r"stage2light_result_p(\d+)\.pt$")
_LOG_RE = re.compile(r"log2_04_step2a_stage2light_local_p(\d+)\.txt$")


def _match_shard_index(path: Path, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def discover_log_paths(log_dir: Path = LOG_DIR) -> list[Path]:
    paths: list[tuple[int, Path]] = []
    for path in log_dir.glob("log2_04_step2a_stage2light_local_p*.txt"):
        shard_index = _match_shard_index(path, _LOG_RE)
        if shard_index is not None:
            paths.append((shard_index, path))
    return [path for _, path in sorted(paths, key=lambda item: item[0])]


def _discover_result_paths(result_dir: Path) -> list[Path]:
    selected: dict[int, Path] = {}
    for path in result_dir.glob("stage2light_result_p*.viz.pkl"):
        shard_index = _match_shard_index(path, _RESULT_VIZ_RE)
        if shard_index is not None:
            selected[shard_index] = path
    for path in result_dir.glob("stage2light_result_p*.pt"):
        shard_index = _match_shard_index(path, _RESULT_PT_RE)
        if shard_index is not None and shard_index not in selected:
            selected[shard_index] = path
    return [selected[idx] for idx in sorted(selected)]


def load_result_payloads(result_dir: Path = RESULT_DIR) -> list[dict]:
    payloads: list[dict] = []
    for result_path in _discover_result_paths(result_dir):
        if result_path.name.endswith(".viz.pkl"):
            with result_path.open("rb") as f:
                payload = pickle.load(f)
        else:
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Visualization needs either a '.viz.pkl' sidecar or a Python env with 'torch' "
                    f"to load legacy result file: {result_path}"
                ) from exc
            payload = torch.load(result_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected payload type for result file: {result_path}")
        payloads.append(payload)
    if not payloads:
        raise FileNotFoundError(f"No shard results found under: {result_dir}")
    return payloads


def window_title(payload: dict) -> str:
    meta = payload.get("window_meta", {})
    return str(meta.get("title", meta.get("role", "window")))


def stage_title(payload: dict) -> str:
    meta = payload.get("window_meta", {})
    role = str(meta.get("role", ""))
    if role == "first_contact":
        return "W1"
    if role == "max_x1_pp_change":
        return "W2"
    if role == "tail_stable":
        return "W3"
    return role or "W?"


def out_path(filename: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def finalize_and_save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {path}")
    return path
