from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.runner.visualization._common import (
    CHECKPOINT_DIR,
    RESULT_DIR,
    _checkpoint_path_for_tag,
    _config_from_payload,
    _latest_existing_path,
    _result_paths_for_tag,
    finalize_and_save,
    load_replayed_handoff_history,
    out_path,
    stage_title,
)


INITIAL_Q_RE = re.compile(r"Plan Z x3dot input: .*?\bq_init=([0-9eE+\-.]+|NaN|Inf|-Inf)\b")
Q_EPOCH_RE = re.compile(r"\bPlanZ q: .*?\bq_init=([0-9eE+\-.]+|NaN|Inf|-Inf)\b")
EPOCH_RE = re.compile(r"KAN epoch (\d+) train=")
LBFGS_EPOCH_RE = re.compile(r"KAN LBFGS step \d+ epoch (\d+) train=")
ROLE_RE = re.compile(r"role=([^|]+)")
LABEL_RE = re.compile(r"label=([^|]+)")


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
DEFAULT_LOG_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "window_per_shard"
DEFAULT_OUT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "visualization"
DEFAULT_OUT_FILE = "afm_param_kan_full_test_04_q_init_grid.png"


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _load_pickle(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_torch_payload(path: Path) -> dict[str, Any] | None:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_light_payload() -> dict[str, Any]:
    candidates: list[Path] = []
    for tag in ("modified_w0", "w0"):
        checkpoint = _checkpoint_path_for_tag(tag, CHECKPOINT_DIR)
        if checkpoint.is_file():
            candidates.append(checkpoint)
        candidates.extend(path for path in _result_paths_for_tag(tag, RESULT_DIR) if path.is_file())

    latest = _latest_existing_path(candidates)
    if latest is None:
        return {}
    if latest.suffix.lower() == ".pkl":
        return _load_pickle(latest) or {}
    return _load_torch_payload(latest) or {}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos = x >= 0.0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _true_x3dot(ode: np.ndarray, known_pars: tuple[float, ...], eta_star: float, mech_true: np.ndarray) -> np.ndarray:
    _, _, _, _, _, r_tip, dist, estar, hamaker, a0, beta = known_pars
    x1 = np.asarray(ode[0, :], dtype=float)
    x2 = np.asarray(ode[1, :], dtype=float)
    x3 = np.asarray(ode[2, :], dtype=float)
    ks = float(np.asarray(mech_true, dtype=float)[0])
    cs = float(np.asarray(mech_true, dtype=float)[1])

    s = dist + x1 - x3
    g = _sigmoid(beta * (s - a0))
    denom = np.maximum(g * (s - a0) + a0, 1.0e-15)
    adhesion = -(hamaker * r_tip) / (6.0 * np.square(denom))
    delta = np.maximum(a0 - s, 0.0)
    hertz = (4.0 / 3.0) * estar * np.sqrt(r_tip) * np.power(delta, 1.5)
    kv_coeff = float(eta_star) * np.sqrt(r_tip) * np.sqrt(delta)
    f_static = adhesion + (1.0 - g) * hertz
    return (-f_static + kv_coeff * x2 - ks * x3) / np.maximum(cs + kv_coeff, 1.0e-15)


def _select_split(prepared, payload: dict[str, Any], cfg):
    meta = payload.get("window_meta", {})
    role = str(meta.get("role", "")).strip().lower() if isinstance(meta, dict) else ""
    if role:
        for split in prepared.splits:
            if str(split.role).strip().lower() == role:
                return split

    index = int(getattr(cfg, "train_window_index", 1))
    if index < 1 or index > len(prepared.splits):
        index = 1
    return prepared.splits[index - 1]


def q_reference_values(payload: dict[str, Any]) -> dict[str, float | str]:
    from AFM04.KAN_full_test.config import default_config
    from AFM04.KAN_full_test.data import prepare_data
    from AFM04.stage1pluslight.data import load_dataset
    from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import DEFAULT_SETTINGS

    cfg = _config_from_payload(payload) if payload else default_config(REPO_ROOT)
    prepared = prepare_data(cfg)
    split = _select_split(prepared, payload, cfg)

    ode_raw, pert_raw = load_dataset(cfg.dataset_root, cfg.error_level, auto_generate=cfg.auto_generate_dataset)
    contact = np.asarray(pert_raw["contact"], dtype=int)
    contact_idx = np.flatnonzero(contact == 1)
    if contact_idx.size == 0:
        raise RuntimeError("No first-contact point found in AFM04 dataset.")
    raw_first_contact = int(contact_idx[0])
    raw_window_start = raw_first_contact + int(split.start_idx)

    settings = DEFAULT_SETTINGS
    known_pars = (
        settings.k,
        settings.wd,
        settings.m,
        settings.c,
        settings.Fd,
        settings.R,
        settings.dist,
        settings.Estar,
        settings.A,
        settings.a0,
        settings.beta,
    )
    mech_true = np.array([settings.ks, settings.cs], dtype=float)
    x3dot_raw = _true_x3dot(np.asarray(ode_raw, dtype=float), known_pars, float(settings.eta_star), mech_true)

    if raw_window_start < 0 or raw_window_start >= x3dot_raw.shape[0]:
        raise RuntimeError(f"Window start index outside raw dataset: {raw_window_start}")

    lag_idx = raw_window_start - 1
    lagged_true = float(x3dot_raw[lag_idx]) if lag_idx >= 0 else float("nan")
    initial_true = float(x3dot_raw[raw_window_start])

    ref_payload = {"window_meta": {"role": split.role, "label": split.label}}
    return {
        "lagged_true": lagged_true,
        "initial_true": initial_true,
        "role": str(split.role),
        "label": str(split.label),
        "title": stage_title(ref_payload),
        "raw_window_start": float(raw_window_start),
        "raw_lag_index": float(lag_idx),
    }


def parse_q_init_series(log_path: Path) -> dict[str, Any]:
    q_by_epoch: dict[int, float] = {}
    role = ""
    label = log_path.name
    current_epoch: int | None = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not role:
            match = ROLE_RE.search(line)
            if match is not None:
                role = match.group(1).strip()
        if label == log_path.name:
            match = LABEL_RE.search(line)
            if match is not None:
                label = match.group(1).strip()

        match = INITIAL_Q_RE.search(line)
        if match is not None:
            q_by_epoch.setdefault(0, parse_float(match.group(1)))
            continue

        match = EPOCH_RE.search(line)
        if match is not None:
            current_epoch = int(match.group(1))
            continue

        match = LBFGS_EPOCH_RE.search(line)
        if match is not None:
            current_epoch = int(match.group(1))
            continue

        match = Q_EPOCH_RE.search(line)
        if match is not None and current_epoch is not None:
            q_by_epoch[current_epoch] = parse_float(match.group(1))
            current_epoch = None

    for row in load_replayed_handoff_history(log_path):
        try:
            epoch = int(float(row.get("epoch", -1)))
            q_init = float(row.get("x3dot_q_init", float("nan")))
        except (TypeError, ValueError):
            continue
        if epoch >= 0:
            q_by_epoch.setdefault(epoch, q_init)

    epochs = sorted(q_by_epoch)
    return {
        "path": str(log_path),
        "label": label,
        "role": role if role else label,
        "epochs": epochs,
        "q_init": [q_by_epoch[e] for e in epochs],
    }


def build_subplot(ax, rec: dict[str, Any], refs: dict[str, float | str]) -> None:
    epochs = rec["epochs"]
    q_init = rec["q_init"]
    ax.set_title(f"{refs['title']} | {rec['label']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("x3dot / q (um/s)")
    ax.grid(True, alpha=0.25)

    lagged_um_s = float(refs["lagged_true"]) * 1.0e6
    initial_um_s = float(refs["initial_true"]) * 1.0e6

    if epochs:
        ax.plot(
            epochs,
            [value * 1.0e6 for value in q_init],
            color="darkorange",
            linewidth=2.2,
            marker="o",
            markersize=3,
            label="q_init trainable",
        )
    else:
        ax.text(0.5, 0.5, "no q_init history", transform=ax.transAxes, ha="center", va="center", color="darkred")

    ax.axhline(
        lagged_um_s,
        color="royalblue",
        linestyle="--",
        linewidth=2,
        label="x3dot initial lagged true",
    )
    ax.axhline(
        initial_um_s,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="x3dot initial true",
    )

    if epochs:
        ax.set_xlim(min(epochs), max(epochs) + 1)
    ax.legend(loc="best")


def run_one(log_dir: Path, out_dir: Path) -> Path:
    log_paths = sorted(log_dir.glob("log2_04_step2a_kan_full_test_local_*.txt"))
    if not log_paths:
        raise FileNotFoundError(f"No KAN full-test shard logs found under: {log_dir}")

    payload = _latest_light_payload()
    refs = q_reference_values(payload)
    recs = [parse_q_init_series(path) for path in log_paths]

    ncols = max(1, len(recs))
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 5.2), sharey=True, squeeze=False)
    for ax, rec in zip(axes[0], recs):
        build_subplot(ax, rec, refs)

    out_file = out_path(DEFAULT_OUT_FILE, out_dir)
    finalize_and_save(fig, out_file)

    print(
        "Reference x3dot values: "
        f"lagged_true={float(refs['lagged_true']):.6e} m/s, "
        f"initial_true={float(refs['initial_true']):.6e} m/s"
    )
    for rec in recs:
        if not rec["epochs"]:
            print(f"{rec['label']}: no q_init data found")
            continue
        print(
            f"{rec['label']}: epochs={rec['epochs'][0]}-{rec['epochs'][-1]} | "
            f"q_init_points={len(rec['epochs'])}"
        )
    return out_file


def main() -> None:
    log_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_LOG_DIR
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    run_one(log_dir, out_dir)


if __name__ == "__main__":
    main()
