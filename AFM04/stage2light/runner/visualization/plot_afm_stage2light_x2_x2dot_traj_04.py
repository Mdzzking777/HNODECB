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
    fig, axes = plt.subplots(2, len(payloads), figsize=(6 * len(payloads), 9), sharex=False, squeeze=False)
    for col, payload in enumerate(payloads):
        snap = _select_snapshot(payload)
        times_us = 1.0e6 * np.asarray(snap["times"], dtype=float)
        ode_true = np.asarray(snap["ode_true"], dtype=float)
        traj_pred = np.asarray(snap["traj_pred"], dtype=float)
        x2dot_true = np.asarray(snap["x2dot_true"], dtype=float)
        x2dot_pred = np.asarray(snap["x2dot_pred"], dtype=float)
        title = f"{stage_title(payload)}: {window_title(payload)}"

        ax1 = axes[0, col]
        ax1.plot(times_us, ode_true[1, :], color="black", linewidth=2, label="x2 true")
        ax1.plot(times_us, traj_pred[1, :], color="crimson", linewidth=2, linestyle="--", label="x2 pred")
        ax1.set_title(title)
        ax1.set_xlabel("time (μs)")
        ax1.set_ylabel("x2")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        ax2.plot(times_us, x2dot_true, color="black", linewidth=2, label="x2dot true")
        ax2.plot(times_us, x2dot_pred, color="crimson", linewidth=2, linestyle="--", label="x2dot pred")
        ax2.set_title(title)
        ax2.set_xlabel("time (μs)")
        ax2.set_ylabel("x2dot")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

    finalize_and_save(fig, out_path("afm_param_stage2light_04_x2_x2dot_traj_grid.png"))


if __name__ == "__main__":
    main()
