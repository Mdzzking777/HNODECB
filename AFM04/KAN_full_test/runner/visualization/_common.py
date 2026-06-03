from __future__ import annotations

import json
import math
import pickle
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


def find_repo_root(start_dir: str | Path) -> Path:
    here = Path(start_dir).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start_dir}")


REPO_ROOT = find_repo_root(__file__)
RESULT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "results"
CHECKPOINT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "checkpoints"
OUT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "visualization"


def _is_path_field(name: str) -> bool:
    return name.endswith("_root") or name.endswith("_dir")


def _config_from_payload(payload: dict[str, Any]):
    from AFM04.KAN_full_test.config import default_config

    cfg = default_config(REPO_ROOT)
    raw = payload.get("config", {})
    if not isinstance(raw, dict):
        return cfg

    known_fields = {field.name for field in fields(cfg)}
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known_fields:
            continue
        if key == "width":
            updates[key] = tuple(int(v) for v in value)
        elif _is_path_field(key):
            updates[key] = Path(value)
        elif key == "device":
            # Visualization should be passive; keep reconstruction off the active training device.
            updates[key] = "cpu"
        else:
            updates[key] = value
    return replace(cfg, **updates)


def _checkpoint_path_for_tag(tag: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> Path:
    return checkpoint_dir / f"kan_full_test_checkpoint_{tag}.pt"


def _result_paths_for_tag(tag: str, result_dir: Path = RESULT_DIR) -> tuple[Path, Path]:
    return result_dir / f"kan_full_test_result_{tag}.viz.pkl", result_dir / f"kan_full_test_result_{tag}.pt"


def _latest_existing_path(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _should_use_latest_checkpoint(tag: str, result_dir: Path = RESULT_DIR) -> tuple[bool, Path | None, Path | None]:
    checkpoint_path = _checkpoint_path_for_tag(tag)
    result_path = _latest_existing_path(list(_result_paths_for_tag(tag, result_dir)))
    if not checkpoint_path.is_file():
        return False, result_path, None
    if result_path is None:
        return True, result_path, checkpoint_path
    return checkpoint_path.stat().st_mtime > result_path.stat().st_mtime + 0.5, result_path, checkpoint_path


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


def _build_model_for_checkpoint(cfg, prepared, dtype):
    from AFM04.KAN_full_test.data import x3dot_scale_from_split
    from AFM04.KAN_full_test.kan_backend import KANForceModule

    train_wpred_enabled = bool(getattr(cfg, "train_wpred_enabled", getattr(cfg, "wpred_enabled", False)))
    split = _select_split(prepared, {}, cfg)
    return KANForceModule(
        pykan_root=cfg.pykan_root,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        seed=cfg.seed,
        width=cfg.width,
        grid=cfg.grid,
        spline_k=cfg.spline_k,
        base_fun=cfg.base_fun,
        symbolic_enabled=cfg.symbolic_enabled,
        auto_save=cfg.auto_save,
        noise_scale=cfg.noise_scale,
        affine_trainable=cfg.affine_trainable,
        grid_eps=cfg.grid_eps,
        grid_range=(cfg.grid_range_lo, cfg.grid_range_hi),
        dist=float(prepared.known_pars[6]),
        a0=float(prepared.known_pars[9]),
        wpred_enabled=train_wpred_enabled,
        wpred_eps=cfg.wpred_eps,
        gnn_learnable=cfg.gnn_learnable,
        soft_mask_enabled=cfg.soft_mask_enabled,
        soft_mask_trainable=cfg.soft_mask_trainable,
        soft_mask_s0_a0=cfg.soft_mask_s0_a0,
        soft_mask_s0_min_a0=cfg.soft_mask_s0_min_a0,
        soft_mask_s0_max_a0=cfg.soft_mask_s0_max_a0,
        soft_mask_alpha_a0=cfg.soft_mask_alpha_a0,
        soft_mask_alpha_min_a0=cfg.soft_mask_alpha_min_a0,
        soft_mask_alpha_max_a0=cfg.soft_mask_alpha_max_a0,
        x3dot_input_enabled=getattr(cfg, "x3dot_input_enabled", False),
        x3dot_init_trainable=getattr(cfg, "x3dot_init_trainable", False),
        x3dot_init_value=getattr(cfg, "x3dot_init_value", 0.0),
        x3dot_scale=x3dot_scale_from_split(
            split,
            configured_scale=getattr(cfg, "x3dot_scale", 0.0),
            scale_mode=getattr(cfg, "x3dot_scale_mode", "x2_scale_tenth"),
            a0=float(prepared.known_pars[9]),
            window_span=float(getattr(cfg, "arch_window_us", 6.288e-6)),
        ),
        x3dot_lag_detach=getattr(cfg, "x3dot_lag_detach", False),
        device=cfg.device,
        dtype=dtype,
    ).to(cfg.device)


def _load_latest_checkpoint_payload(tag: str, checkpoint_path: Path) -> dict:
    try:
        import torch

        from AFM04.KAN_full_test.data import prepare_data
        from AFM04.KAN_full_test.train import _snapshot_split, _torch_dtype, _window_to_torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Checkpoint-based live visualization needs a Python env with torch and KFT modules."
        ) from exc

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"checkpoint temporarily unavailable: {checkpoint_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected checkpoint payload type: {checkpoint_path}")
    if "state_dict" not in payload:
        raise RuntimeError(f"Checkpoint missing state_dict: {checkpoint_path}")

    cfg = _config_from_payload(payload)
    prepared = prepare_data(cfg)
    split = _select_split(prepared, payload, cfg)
    dtype = _torch_dtype(cfg.dtype)
    model = _build_model_for_checkpoint(cfg, prepared, dtype)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    tensors = _window_to_torch(split, dtype=dtype, device=cfg.device)
    mech_true = torch.as_tensor(prepared.mech_true, dtype=dtype, device=cfg.device)
    snapshot = _snapshot_split(
        force_module=model,
        known_pars=prepared.known_pars,
        eta_star_true=prepared.eta_star_true,
        mech_true=mech_true,
        split=split,
        tensors=tensors,
        dtype=dtype,
        device=cfg.device,
        ode_method=cfg.ode_method,
        ode_rtol=cfg.ode_rtol,
        ode_atol=cfg.ode_atol,
    )
    window_meta = dict(payload.get("window_meta") or {})
    if not window_meta:
        window_meta = {
            "role": split.role,
            "label": split.label,
            "title": snapshot.get("title", split.role),
            "start_idx": split.start_idx,
            "stop_idx": split.stop_idx,
            "t_start": split.t_start,
            "t_stop": split.t_stop,
        }
    return {
        "best": {"best_snapshot": snapshot},
        "final_snapshot": snapshot,
        "latest_snapshot": snapshot,
        "history": payload.get("history", []),
        "window_meta": window_meta,
        "config": payload.get("config", {}),
        "source": "latest_checkpoint",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_tag": tag,
    }


def time_snapshot(payload: dict) -> dict:
    if "latest_snapshot" in payload:
        return payload["latest_snapshot"]
    if "final_snapshot" in payload:
        return payload["final_snapshot"]
    best = payload.get("best", {})
    if isinstance(best, dict) and "best_snapshot" in best:
        return best["best_snapshot"]
    raise KeyError("No latest_snapshot, final_snapshot, or best.best_snapshot found in payload")


def progress_note(payload: dict) -> str:
    if payload.get("source") != "latest_checkpoint":
        return ""

    epoch = int(payload.get("checkpoint_epoch", -1))
    phase = ""
    history = payload.get("history", [])
    if isinstance(history, list):
        for row in reversed(history):
            if not isinstance(row, dict):
                continue
            try:
                row_epoch = int(float(row.get("epoch", -1)))
            except (TypeError, ValueError):
                continue
            if row_epoch != epoch:
                continue
            phase_raw = row.get("phase", row.get("optimizer", ""))
            phase = "LBFGS" if is_lbfgs_phase(phase_raw) else "Adam"
            break

    epoch_text = f"epoch {epoch}" if epoch >= 0 else "unknown epoch"
    phase_text = f" ({phase})" if phase else ""
    return f"latest checkpoint: {epoch_text}{phase_text}"


def title_with_progress(payload: dict, title: str) -> str:
    note = progress_note(payload)
    return f"{title}\n{note}" if note else title


def load_result_payloads(result_dir: Path = RESULT_DIR) -> list[dict]:
    payloads: list[dict] = []
    for tag in ("modified_w0", "w0"):
        use_checkpoint, result_path, checkpoint_path = _should_use_latest_checkpoint(tag, result_dir)
        if use_checkpoint and checkpoint_path is not None:
            payloads.append(_load_latest_checkpoint_payload(tag, checkpoint_path))
            break

        viz_path, pt_path = _result_paths_for_tag(tag, result_dir)
        if result_path == viz_path and viz_path.is_file():
            with viz_path.open("rb") as f:
                payload = pickle.load(f)
        elif result_path == pt_path and pt_path.is_file():
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Visualization needs either a '.viz.pkl' sidecar or a Python env with 'torch' "
                    f"to load legacy result file: {pt_path}"
                ) from exc
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        else:
            continue
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected payload type for result tag {tag}")
        payloads.append(payload)
        break
    if not payloads:
        raise FileNotFoundError(f"No shard results found under: {result_dir}")
    return payloads


def load_replayed_handoff_history(log_path: Path) -> list[dict]:
    """Return Adam-side history when a live LBFGS-only log replays a handoff.

    The separate LBFGS runner intentionally writes into the normal KFT-GBO log
    location, so the current shard log starts at the LBFGS phase.  During a
    live run the final result history has not been rewritten yet.  For epoch
    plots, recover epochs 1..AdamEpochs from the handoff checkpoint referenced
    by the log.
    """
    text = log_path.read_text(encoding="utf-8")
    if "handoff replay loaded" not in text and "replay=ON" not in text:
        return []

    handoff_path: Path | None = None
    for line in text.splitlines():
        if "handoff:" not in line or "replay=ON" not in line or "path=" not in line:
            continue
        raw_path = line.split("path=", 1)[1].strip()
        if raw_path:
            handoff_path = Path(raw_path)
            break
    if handoff_path is None or not handoff_path.is_file():
        return []

    try:
        import torch
    except ModuleNotFoundError:
        return []

    payload = torch.load(handoff_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return []
    history = payload.get("history", [])
    if not isinstance(history, list):
        return []
    return [dict(row) for row in history if isinstance(row, dict)]


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
    if role == "modified_w0":
        return "modified W0"
    if role == "first_contact":
        return "W0"
    if role == "middle":
        return "W1"
    if role == "max_x1_pp_change":
        return "W2"
    if role == "tail_stable":
        return "W3"
    return role or "W?"


def is_lbfgs_phase(phase: object) -> bool:
    return "lbfgs" in str(phase or "").strip().lower().replace("-", "")


def history_phases(history: list[dict]) -> list[str]:
    phases: list[str] = []
    for row in history:
        phase = row.get("phase", row.get("optimizer", ""))
        phase_text = str(phase or "").strip().lower()
        if "warmstart" in phase_text or phase_text in ("epoch0", "epoch_0", "rs"):
            phases.append("warmstart")
        else:
            phases.append("lbfgs" if is_lbfgs_phase(phase) else "adam")
    return phases


DEFAULT_EPOCH_SAMPLE_EVERY = 1


def sample_epoch_indices(
    epochs: list[int],
    phases: list[str] | None = None,
    *,
    sample_every: int = DEFAULT_EPOCH_SAMPLE_EVERY,
) -> list[int]:
    if not epochs:
        return []
    if phases is None or len(phases) != len(epochs):
        phases = ["adam"] * len(epochs)
    sample_every = max(1, int(sample_every))
    indices: list[int] = []
    for idx, (epoch, phase) in enumerate(zip(epochs, phases)):
        phase_changed = idx > 0 and phase != phases[idx - 1]
        phase_will_change = idx + 1 < len(phases) and phase != phases[idx + 1]
        if idx == 0 or (int(epoch) - 1) % sample_every == 0 or phase_changed or phase_will_change:
            indices.append(idx)
    if indices[-1] != len(epochs) - 1:
        indices.append(len(epochs) - 1)
    return indices


def sample_epoch_series(
    epochs: list[int],
    values: list[float],
    phases: list[str] | None = None,
    *,
    sample_every: int = DEFAULT_EPOCH_SAMPLE_EVERY,
) -> tuple[list[int], list[float], list[str]]:
    if phases is None or len(phases) != len(epochs):
        phases = ["adam"] * len(epochs)
    indices = sample_epoch_indices(epochs, phases, sample_every=sample_every)
    return (
        [epochs[idx] for idx in indices],
        [values[idx] for idx in indices],
        [phases[idx] for idx in indices],
    )


def phase_color(base_color: str, *, phase: str) -> str:
    if phase == "warmstart":
        return base_color
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
        seg_start = start
        if start > 0:
            seg_start = start - 1
        xs = list(x_values[seg_start:idx])
        ys = list(y_values[seg_start:idx])
        phase_name = "LBFGS" if phase == "lbfgs" else ("warmstart" if phase == "warmstart" else "Adam")
        plot_label = label if not has_lbfgs else f"{label} {phase_name}"
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


def set_integer_epoch_axis(ax) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))


def out_path(filename: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def finalize_and_save(fig, path: Path) -> Path:
    for ax in fig.axes:
        if ax.get_xlabel().strip().lower() == "epoch":
            set_integer_epoch_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {path}")
    return path


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _tag_from_role(role: str) -> str:
    role_key = str(role or "").strip().lower()
    if role_key in ("modified_w0", "modified-w0", "shifted_w0", "shifted-w0"):
        return "modified_w0"
    if role_key in ("first_contact", "w0", ""):
        return "w0"
    return role_key


def load_epoch0_warmstart_record(role: str = "", result_dir: Path = RESULT_DIR) -> dict[str, Any] | None:
    """Return the RS/pre-GBO warmstart point for epoch-axis plots.

    KFT histories start at epoch 1 after the first optimizer update.  For
    visualization, epoch 0 is the selected warmstart model immediately after
    RS/prerun selection and before GBO optimizer steps.  At present RS summary
    stores total loss and reconstruction/force errors, but not every loss-term
    decomposition; unavailable fields are intentionally left as NaN.
    """

    tag = _tag_from_role(role)
    summary_path = result_dir / "random_search" / f"kan_full_test_random_search_summary_{tag}.json"
    if not summary_path.is_file() and tag != "w0":
        summary_path = result_dir / "random_search" / "kan_full_test_random_search_summary_w0.json"
    if not summary_path.is_file():
        return None

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    best = payload.get("best_trial") or payload.get("window_best_trial")
    if not isinstance(best, dict):
        return None

    train_loss = _finite_or_nan(best.get("train_loss", best.get("ranking_loss")))
    val_loss = _finite_or_nan(best.get("val_loss"))
    train_x1 = _finite_or_nan(best.get("train_x1_rec", best.get("x1_rec")))
    train_x3 = _finite_or_nan(best.get("train_x3_rec", best.get("x3_rec")))
    train_nn = _finite_or_nan(best.get("train_nn_err", best.get("nn_err")))
    val_x1 = _finite_or_nan(best.get("val_x1_rec"))
    val_x3 = _finite_or_nan(best.get("val_x3_rec"))
    val_nn = _finite_or_nan(best.get("val_nn_err"))

    # RS in the current KFT setup is train-only.  For single-series plots that
    # prefer validation metrics when later epochs have them, expose the same
    # warmstart point on the val_* keys while marking its source explicitly.
    val_fallback_used = not (math.isfinite(val_x1) and math.isfinite(val_x3) and math.isfinite(val_nn))
    if val_fallback_used:
        val_x1, val_x3, val_nn = train_x1, train_x3, train_nn
    if not math.isfinite(val_loss):
        val_loss = float("nan")

    return {
        "epoch": 0,
        "phase": "warmstart",
        "optimizer": "warmstart",
        "epoch0_source": "random_search_summary",
        "epoch0_metric_split": "train" if val_fallback_used else "val",
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_x1_rec": train_x1,
        "train_x3_rec": train_x3,
        "train_fts_rollout_rec": train_nn,
        "val_x1_rec": val_x1,
        "val_x3_rec": val_x3,
        "val_fts_rollout_rec": val_nn,
        "g_nn": _finite_or_nan(best.get("g_nn", best.get("init_gnn"))),
        "soft_mask_enabled": bool(best.get("soft_mask_enabled", False)),
        "soft_mask_trainable": bool(best.get("soft_mask_trainable", False)),
        "soft_mask_s0_a0": _finite_or_nan(best.get("soft_mask_s0_a0")),
        "soft_mask_alpha_a0": _finite_or_nan(best.get("soft_mask_alpha_a0")),
        "soft_mask_m_min": _finite_or_nan(best.get("soft_mask_m_min")),
        "trial": best.get("trial"),
        "seed": best.get("seed"),
    }


def prepend_epoch0_history(history: list[dict], role: str = "", result_dir: Path = RESULT_DIR) -> list[dict]:
    rows = [dict(row) for row in history if isinstance(row, dict)]
    for row in rows:
        try:
            if int(float(row.get("epoch", -1))) == 0:
                return rows
        except (TypeError, ValueError):
            continue
    epoch0 = load_epoch0_warmstart_record(role, result_dir=result_dir)
    if epoch0 is None:
        return rows
    return [epoch0, *rows]
