from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM05" / "stage2light").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
_OUTPUT_ROOT_RAW = os.environ.get("HNODECB_AFM05_STAGE2LIGHT_OUTPUT_ROOT", "").strip()
OUTPUT_ROOT = (
    Path(_OUTPUT_ROOT_RAW).expanduser().resolve()
    if _OUTPUT_ROOT_RAW
    else REPO_ROOT / "AFM05" / "stage2light"
)
RESULT_DIR = OUTPUT_ROOT / "results"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
OUT_DIR = OUTPUT_ROOT / "logs" / "visualization"
LOG_DIR = OUTPUT_ROOT / "logs" / "window_per_shard"

_RESULT_VIZ_RE = re.compile(r"stage2light_result_p(\d+)\.viz\.pkl$")
_RESULT_PT_RE = re.compile(r"stage2light_result_p(\d+)\.pt$")
_CHECKPOINT_PT_RE = re.compile(r"stage2light_checkpoint_p(\d+)\.pt$")
_BEST_OR_FINAL_PT_RE = re.compile(r"stage2light_(?:best|final)_p(\d+)\.pt$")
_LOG_RE = re.compile(r"log2_05_step2a_stage2light_local_p(\d+)\.txt$")


def _match_shard_index(path: Path, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def discover_log_paths(log_dir: Path = LOG_DIR) -> list[Path]:
    paths: list[tuple[int, Path]] = []
    for path in log_dir.glob("log2_05_step2a_stage2light_local_p*.txt"):
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


def _discover_checkpoint_paths(checkpoint_dir: Path) -> list[Path]:
    selected: dict[int, Path] = {}
    for path in checkpoint_dir.glob("stage2light_checkpoint_p*.pt"):
        shard_index = _match_shard_index(path, _CHECKPOINT_PT_RE)
        if shard_index is not None:
            selected[shard_index] = path
    for path in checkpoint_dir.glob("stage2light_best_p*.pt"):
        shard_index = _match_shard_index(path, _BEST_OR_FINAL_PT_RE)
        if shard_index is not None:
            current = selected.get(shard_index)
            if current is None or path.stat().st_mtime > current.stat().st_mtime:
                selected[shard_index] = path
    for path in checkpoint_dir.glob("stage2light_final_p*.pt"):
        shard_index = _match_shard_index(path, _BEST_OR_FINAL_PT_RE)
        if shard_index is not None:
            current = selected.get(shard_index)
            if current is None or path.stat().st_mtime > current.stat().st_mtime:
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
        payload = dict(payload)
        payload["_source_path"] = str(result_path)
        payload["_source_mtime"] = float(result_path.stat().st_mtime)
        payloads.append(payload)
    if not payloads:
        raise FileNotFoundError(f"No shard results found under: {result_dir}")
    return payloads


def load_checkpoint_payloads(checkpoint_dir: Path = CHECKPOINT_DIR) -> list[dict]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Visualization needs a Python env with 'torch' to load running stage2light checkpoints."
        ) from exc

    payloads: list[dict] = []
    for checkpoint_path in _discover_checkpoint_paths(checkpoint_dir):
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Could not load running checkpoint: {checkpoint_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected payload type for checkpoint file: {checkpoint_path}")
        payload = dict(payload)
        payload["_source_path"] = str(checkpoint_path)
        payload["_source_mtime"] = float(checkpoint_path.stat().st_mtime)
        payloads.append(payload)
    if not payloads:
        raise FileNotFoundError(f"No running stage2light checkpoints found under: {checkpoint_dir}")
    return payloads


def _payload_key(payload: dict) -> str:
    meta = payload.get("window_meta", {})
    role = str(meta.get("role", "")).strip()
    label = str(meta.get("label", "")).strip()
    if role:
        return role
    if label:
        return label
    return str(payload.get("shard_index", "unknown"))


def load_available_payloads(
    *,
    result_dir: Path = RESULT_DIR,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> list[dict]:
    """Load completed results, or current running checkpoints if results are absent.

    This is for in-progress visualization: result payloads remain preferred for
    finished runs, while running checkpoints supply the latest history during a
    live stage2light process.
    """
    merged: dict[str, dict] = {}

    try:
        checkpoint_payloads = load_checkpoint_payloads(checkpoint_dir)
    except FileNotFoundError:
        checkpoint_payloads = []
    for payload in checkpoint_payloads:
        merged[_payload_key(payload)] = payload

    try:
        result_payloads = load_result_payloads(result_dir)
    except FileNotFoundError:
        result_payloads = []
    for payload in result_payloads:
        key = _payload_key(payload)
        current = merged.get(key)
        if current is None or float(payload.get("_source_mtime", 0.0)) >= float(current.get("_source_mtime", 0.0)):
            merged[key] = payload

    if not merged:
        raise FileNotFoundError(
            f"No stage2light completed results under {result_dir} or running checkpoints under {checkpoint_dir}"
        )
    return [merged[key] for key in sorted(merged)]


def window_title(payload: dict) -> str:
    meta = payload.get("window_meta", {})
    role = str(meta.get("role", "")).strip()
    if role == "first_contact":
        return "W0: first-contact window"
    if role == "middle":
        return "W1: middle window"
    return str(meta.get("title", meta.get("role", "window")))


def stage_title(payload: dict) -> str:
    meta = payload.get("window_meta", {})
    role = str(meta.get("role", ""))
    if role == "first_contact":
        return "W0"
    if role == "middle":
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


def phase_color(base_color: str, *, phase: str) -> str:
    if phase != "lbfgs":
        return base_color
    light = {
        "steelblue": "lightskyblue",
        "firebrick": "lightcoral",
        "seagreen": "mediumaquamarine",
        "purple": "plum",
        "black": "0.55",
        "royalblue": "cornflowerblue",
        "darkorange": "moccasin",
        "crimson": "lightcoral",
    }
    return light.get(base_color, base_color)


def plot_phase_series(
    ax,
    x_values: list[int] | list[float],
    y_values: list[float],
    phases: list[str] | None,
    *,
    label: str,
    color: str,
    linewidth: float = 2.0,
    linestyle: str = "-",
    marker: str | None = None,
    markersize: float = 3.0,
) -> None:
    if not x_values:
        return
    if phases is None or len(phases) != len(x_values):
        phases = ["adam"] * len(x_values)

    has_lbfgs = any(phase == "lbfgs" for phase in phases)
    start = 0
    first_label_for_phase: set[str] = set()
    for idx in range(1, len(x_values) + 1):
        if idx < len(x_values) and phases[idx] == phases[start]:
            continue
        phase = phases[start]
        seg_start = start - 1 if start > 0 else start
        xs = list(x_values[seg_start:idx])
        ys = list(y_values[seg_start:idx])
        plot_label = label if not has_lbfgs else f"{label} {'LBFGS' if phase == 'lbfgs' else 'Adam'}"
        if plot_label in first_label_for_phase:
            plot_label = "_nolegend_"
        else:
            first_label_for_phase.add(plot_label)
        ax.plot(
            xs,
            ys,
            label=plot_label,
            color=phase_color(color, phase=phase),
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
        )
        start = idx
