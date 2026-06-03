from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.stage2light.runner.visualization._common import (
    REPO_ROOT,
    finalize_and_save,
    load_result_payloads,
    out_path,
    stage_title,
    window_title,
)


DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "AFM04" / "stage2light" / "checkpoints"
_BEST_VIZ_RE = re.compile(r"stage2light_best_p(\d+)\.viz\.pkl$")


def _true_based_ylim(f_true: np.ndarray) -> tuple[float, float]:
    lo = float(np.nanmin(f_true))
    hi = float(np.nanmax(f_true))
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        scale = max(abs(lo), abs(hi), 1.0e-12)
        pad = 0.1 * scale
    else:
        pad = 0.12 * span
    return lo - pad, hi + pad


def _role_sort_key(role: str) -> tuple[int, str]:
    role_norm = role.strip().lower()
    if role_norm == "first_contact":
        return (0, role_norm)
    if role_norm == "middle":
        return (1, role_norm)
    if role_norm == "max_x1_pp_change":
        return (2, role_norm)
    if role_norm == "tail_stable":
        return (3, role_norm)
    return (99, role_norm)


def _payload_key(payload: dict) -> str:
    meta = payload.get("window_meta", {})
    role = str(meta.get("role", "")).strip()
    label = str(meta.get("label", "")).strip()
    return role if role else label


def _load_checkpoint_viz_payloads(checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> list[dict]:
    selected: dict[int, Path] = {}
    for path in checkpoint_dir.glob("stage2light_best_p*.viz.pkl"):
        match = _BEST_VIZ_RE.match(path.name)
        if match is not None:
            selected[int(match.group(1))] = path

    payloads: list[dict] = []
    for shard_index in sorted(selected):
        with selected[shard_index].open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def load_available_payloads() -> list[dict]:
    merged: dict[str, dict] = {}

    for payload in _load_checkpoint_viz_payloads():
        merged[_payload_key(payload)] = payload

    try:
        result_payloads = load_result_payloads()
    except FileNotFoundError:
        result_payloads = []
    for payload in result_payloads:
        merged[_payload_key(payload)] = payload

    if not merged:
        raise FileNotFoundError("No stage2light completed results or checkpoint viz payloads found.")

    return sorted(
        merged.values(),
        key=lambda payload: _role_sort_key(str(payload.get("window_meta", {}).get("role", ""))),
    )


def _select_snapshot(payload: dict) -> dict:
    final_snapshot = payload.get("final_snapshot")
    if isinstance(final_snapshot, dict):
        snap = final_snapshot.get("full") or final_snapshot.get("val")
        if isinstance(snap, dict):
            return snap

    best_snapshot = payload["best"]["best_snapshot"]
    snap = best_snapshot.get("full") or best_snapshot.get("val")
    if snap is None:
        raise KeyError("Neither final nor best snapshot contains 'full'/'val' data")
    return snap


def main() -> None:
    payloads = load_available_payloads()
    fig, axes = plt.subplots(1, len(payloads), figsize=(6 * len(payloads), 5.2), sharex=False, squeeze=False)
    axes = axes[0]
    for ax, payload in zip(axes, payloads):
        snap = _select_snapshot(payload)
        times_us = 1.0e6 * np.asarray(snap["times"], dtype=float)
        f_true = np.asarray(snap["fts_teacher_true"], dtype=float)
        f_pred = np.asarray(snap["fts_rollout_pred"], dtype=float)
        title = f"{stage_title(payload)}: {window_title(payload)}"
        ax.plot(times_us, f_true, color="black", linewidth=2.4, label="Fts teacher true", zorder=2)
        ax.plot(times_us, f_pred, color="crimson", linewidth=2, linestyle="--", label="Fts rollout pred", zorder=3)
        ax.set_ylim(*_true_based_ylim(f_true))
        ax.set_title(title)
        ax.set_xlabel("time (μs)")
        ax.set_ylabel("force (N)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    finalize_and_save(fig, out_path("afm_param_stage2light_04_fcontact_grid.png"))


if __name__ == "__main__":
    main()
