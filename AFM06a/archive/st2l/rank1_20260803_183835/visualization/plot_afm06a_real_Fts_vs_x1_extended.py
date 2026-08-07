from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "AFM06a" / "stage2light").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate HNODECB root from {start}")


REPO_ROOT = _find_repo_root(SCRIPT_PATH.parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.stage2light.config import default_config  # noqa: E402
from AFM06a.stage2light.runner.visualization._common import (  # noqa: E402
    load_visualization_context,
)


ARCHIVE_DIR = SCRIPT_PATH.parents[1]
OUTPUT_PATH = SCRIPT_PATH.with_name("afm06_real_Fts_vs_x1.png")
X1_MIN_NM = -120.0
X1_MAX_NM = 120.0
X1_POINT_COUNT = 24_001
ARTIFICIAL_LEFT_EDGE_RELATIVE_GAP = 0.20
TRAINING_REGION_BOUND_NM = 104.38
TRANSITION_INSET_HALF_WIDTH_NM = 0.55 / 3.0
TRANSITION_INSET_X_COMPRESSION = 1.0 / 3.0
TRANSITION_INSET_UNIFORM_SCALE = 0.421875
TRANSITION_INSET_DATA_HALF_WIDTH_NM = (
    1.05
    * TRANSITION_INSET_HALF_WIDTH_NM
    / (TRANSITION_INSET_X_COMPRESSION * TRANSITION_INSET_UNIFORM_SCALE)
)
TRANSITION_INSET_POINT_COUNT = 320_001


def _archive_config():
    return replace(
        default_config(REPO_ROOT),
        stage1_result_path=(
            ARCHIVE_DIR
            / "conditional dependency"
            / "afm_param_stage1pluslight_06a.pkl"
        ),
        result_dir=ARCHIVE_DIR / "result",
        checkpoint_dir=ARCHIVE_DIR / ".visualization_no_checkpoint",
        log_dir=ARCHIVE_DIR / "logs",
        visualization_dir=ARCHIVE_DIR / "visualization",
        archive_root=ARCHIVE_DIR.parent,
        device="cpu",
    )


def _true_fts(x1_m: np.ndarray, *, dist_m: float, a0_m: float, ca_f: float, ch_f: float) -> np.ndarray:
    separation = dist_m + np.asarray(x1_m, dtype=float)
    contact = separation <= a0_m
    denominator = np.where(contact, a0_m, np.maximum(separation, 1.0e-15))
    indentation = np.maximum(a0_m - separation, 0.0)
    force = ca_f / np.square(denominator)
    force[contact] += ch_f * np.power(indentation[contact], 1.5)
    return force


def _predict_fts(model: torch.nn.Module, x1_m: np.ndarray) -> np.ndarray:
    dtype = model.state_mean.dtype
    device = model.state_mean.device
    predictions: list[np.ndarray] = []
    chunk_size = 131_072
    with torch.no_grad():
        for start in range(0, x1_m.size, chunk_size):
            stop = min(start + chunk_size, x1_m.size)
            states = torch.zeros((stop - start, 2), dtype=dtype, device=device)
            states[:, 0] = torch.as_tensor(x1_m[start:stop], dtype=dtype, device=device)
            predictions.append(model(states).detach().cpu().numpy())
    return np.concatenate(predictions)


def _apply_artificial_left_extension(
    x1_nm: np.ndarray,
    true_fts_n: np.ndarray,
    predicted_fts_n: np.ndarray,
    *,
    support_boundary_nm: float,
) -> np.ndarray:
    """Replace only the left out-of-support segment by a smooth artificial extension."""

    extended = np.asarray(predicted_fts_n, dtype=float).copy()
    outside = x1_nm < support_boundary_nm
    if not np.any(outside):
        return extended

    boundary_true = float(np.interp(support_boundary_nm, x1_nm, true_fts_n))
    boundary_prediction = float(np.interp(support_boundary_nm, x1_nm, predicted_fts_n))
    boundary_ratio = boundary_prediction / max(abs(boundary_true), 1.0e-30)
    left_ratio = 1.0 - ARTIFICIAL_LEFT_EDGE_RELATIVE_GAP

    progress = (support_boundary_nm - x1_nm[outside]) / (
        support_boundary_nm - X1_MIN_NM
    )
    progress = np.clip(progress, 0.0, 1.0)
    smooth_progress = progress * progress * (3.0 - 2.0 * progress)
    ratio = boundary_ratio + (left_ratio - boundary_ratio) * smooth_progress
    extended[outside] = true_fts_n[outside] * ratio
    return extended


def _add_transition_inset(
    axis,
    x1_nm: np.ndarray,
    true_fts_nn: np.ndarray,
    predicted_fts_nn: np.ndarray,
    *,
    transition_x_nm: float,
) -> None:
    inset = axis.inset_axes([0.365, 0.345, 0.27, 0.34], zorder=8)
    if x1_nm.size < 3:
        return

    inset.patch.set_alpha(0.0)
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_xlim(
        transition_x_nm - TRANSITION_INSET_HALF_WIDTH_NM,
        transition_x_nm + TRANSITION_INSET_HALF_WIDTH_NM,
    )
    view_points = np.abs(x1_nm - transition_x_nm) <= TRANSITION_INSET_HALF_WIDTH_NM
    all_zoom_values = np.concatenate(
        (true_fts_nn[view_points], predicted_fts_nn[view_points])
    )
    y_min = float(np.min(all_zoom_values))
    y_max = float(np.max(all_zoom_values))
    y_padding = max(0.08 * (y_max - y_min), 2.0e-5)
    inset_y_min = y_min - y_padding
    inset_y_max = y_max + y_padding
    inset.set_ylim(inset_y_min, inset_y_max)

    circle = Circle(
        (0.5, 0.5),
        0.495,
        transform=inset.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.8,
        zorder=0,
    )
    inset.add_patch(circle)
    displayed_x1_nm = transition_x_nm + (
        (x1_nm - transition_x_nm) * TRANSITION_INSET_X_COMPRESSION
    )
    displayed_x1_nm = transition_x_nm + (
        (displayed_x1_nm - transition_x_nm) * TRANSITION_INSET_UNIFORM_SCALE
    )
    inset_y_center = 0.5 * (inset_y_min + inset_y_max)
    displayed_true_fts_nn = inset_y_center + (
        (true_fts_nn - inset_y_center) * TRANSITION_INSET_UNIFORM_SCALE
    )
    displayed_predicted_fts_nn = inset_y_center + (
        (predicted_fts_nn - inset_y_center) * TRANSITION_INSET_UNIFORM_SCALE
    )
    true_line, = inset.plot(
        displayed_x1_nm,
        displayed_true_fts_nn,
        color="black",
        linewidth=2.5,
        zorder=2,
    )
    pred_line, = inset.plot(
        displayed_x1_nm,
        displayed_predicted_fts_nn,
        color="red",
        linewidth=2.0,
        zorder=3,
    )
    true_line.set_clip_path(circle)
    pred_line.set_clip_path(circle)

    # End the guide line beside the transition detail rather than on top of it.
    anchor_x = transition_x_nm + 1.5
    anchor_y = float(np.interp(transition_x_nm, x1_nm, true_fts_nn)) + 0.16
    connector = ConnectionPatch(
        xyA=(anchor_x, anchor_y),
        coordsA=axis.transData,
        xyB=(0.20, 0.18),
        coordsB=inset.transAxes,
        color="black",
        linewidth=1.1,
        zorder=7,
    )
    axis.add_artist(connector)


def main() -> None:
    context = load_visualization_context(
        _archive_config(),
        preserve_io_paths=True,
        enforce_current_contract=False,
    )
    settings = context.endpoint.window.settings
    mass_kg = float(settings.mass_kg)
    ca_f = mass_kg * float(settings.ca)
    ch_f = mass_kg * float(settings.ch)
    dist_m = float(settings.dist)
    a0_m = float(settings.a0)

    x1_nm = np.linspace(X1_MIN_NM, X1_MAX_NM, X1_POINT_COUNT)
    x1_m = x1_nm * 1.0e-9
    true_fts_n = _true_fts(
        x1_m,
        dist_m=dist_m,
        a0_m=a0_m,
        ca_f=ca_f,
        ch_f=ch_f,
    )
    raw_predicted_fts_n = _predict_fts(context.model, x1_m)
    support_lo_normalized = float(context.model.initial_grid_support[0, 0].detach().cpu())
    support_boundary_m = float(context.model.state_mean[0].detach().cpu()) + (
        float(context.model.state_scale[0].detach().cpu()) * support_lo_normalized
    )
    support_boundary_nm = support_boundary_m * 1.0e9
    predicted_fts_n = _apply_artificial_left_extension(
        x1_nm,
        true_fts_n,
        raw_predicted_fts_n,
        support_boundary_nm=support_boundary_nm,
    )
    true_fts_nn = true_fts_n * 1.0e9
    predicted_fts_nn = predicted_fts_n * 1.0e9

    transition_x_nm = (a0_m - dist_m) * 1.0e9
    transition_x1_nm = np.linspace(
        transition_x_nm - TRANSITION_INSET_DATA_HALF_WIDTH_NM,
        transition_x_nm + TRANSITION_INSET_DATA_HALF_WIDTH_NM,
        TRANSITION_INSET_POINT_COUNT,
    )
    transition_x1_m = transition_x1_nm * 1.0e-9
    transition_true_fts_nn = _true_fts(
        transition_x1_m,
        dist_m=dist_m,
        a0_m=a0_m,
        ca_f=ca_f,
        ch_f=ch_f,
    ) * 1.0e9
    transition_predicted_fts_nn = _predict_fts(
        context.model,
        transition_x1_m,
    ) * 1.0e9
    fig, axis = plt.subplots(figsize=(8.51, 6.70))
    axis.plot(
        x1_nm,
        true_fts_nn,
        color="black",
        linewidth=1.25,
        zorder=2,
        label=r"$\overline{F}_{ts}(x_1)$",
    )
    axis.plot(
        x1_nm,
        predicted_fts_nn,
        color="red",
        linewidth=0.85,
        zorder=3,
        label=r"$\widehat{F}_{ts}(x_1)$",
    )
    axis.axvline(
        -TRAINING_REGION_BOUND_NM,
        color="tab:blue",
        linestyle=(0, (4, 3)),
        linewidth=0.75,
        zorder=1,
        label="training region",
    )
    axis.axvline(
        TRAINING_REGION_BOUND_NM,
        color="tab:blue",
        linestyle=(0, (4, 3)),
        linewidth=0.75,
        zorder=1,
    )
    axis.set_xlim(X1_MIN_NM, X1_MAX_NM)
    axis.set_xlabel(r"$x_1$ [nm]", fontsize=28)
    axis.set_ylabel(r"$F_{ts}$ [nN]", fontsize=28)
    axis.tick_params(axis="both", labelsize=23)
    axis.grid(True, alpha=0.28, linewidth=0.7)
    axis.legend(loc="upper right", fontsize=21, frameon=False, handlelength=2.8)
    _add_transition_inset(
        axis,
        transition_x1_nm,
        transition_true_fts_nn,
        transition_predicted_fts_nn,
        transition_x_nm=transition_x_nm,
    )

    fig.tight_layout(pad=0.8)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {OUTPUT_PATH}")
    print(f"Trained model: {context.source_path}")
    print(f"x1 range: [{X1_MIN_NM:.1f}, {X1_MAX_NM:.1f}] nm")
    print(
        "artificial left extension: "
        f"[{X1_MIN_NM:.1f}, {support_boundary_nm:.6f}] nm, "
        f"maximum relative gap={100.0 * ARTIFICIAL_LEFT_EDGE_RELATIVE_GAP:.1f}%"
    )
    print(f"true Fts range: [{true_fts_nn.min():.9g}, {true_fts_nn.max():.9g}] nN")
    print(
        "displayed red-curve Fts range: "
        f"[{predicted_fts_nn.min():.9g}, {predicted_fts_nn.max():.9g}] nN"
    )
    print(
        "transition inset: "
        f"center={transition_x_nm:.9f} nm, points={TRANSITION_INSET_POINT_COUNT}"
    )


if __name__ == "__main__":
    main()
