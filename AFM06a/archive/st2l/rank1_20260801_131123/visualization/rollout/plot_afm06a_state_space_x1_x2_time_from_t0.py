"""Plot AFM06a archived t=0 rollout in x1-x2-time state space."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

ADJUST_LEGEND = "--adjust-legend" in sys.argv[1:]
if ADJUST_LEGEND:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = SCRIPT_DIR.parents[1]
DATA_PATH = ARCHIVE_ROOT / "data" / "pert_df_afm_dmt_hard.npz"
CACHE_PATH = SCRIPT_DIR / "afm06a_stage2light_full_rollout_from_t0_cache_rank1.pkl"
OUTPUT_PATH = SCRIPT_DIR / "afm_param_stage2light_06a_state_space_x1_x2_time_full_rollout_from_t0.png"
MATCH_DYNAMICS_OUTPUT_PATH = (
    SCRIPT_DIR
    / "afm_param_stage2light_06a_state_space_x1_x2_time_full_rollout_from_t0_same_aspect.png"
)
MAX_RENDER_POINTS = 25_000
AXIS_FONT_SIZE = 18
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 18
DEFAULT_FIGSIZE = (10.5, 8.2)
MATCH_DYNAMICS_FIGSIZE = (14.3, 8.2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adjust-legend",
        action="store_true",
        help="Open the figure, drag the legend, and save the final position.",
    )
    parser.add_argument(
        "--match-dynamics-aspect",
        action="store_true",
        help="Save a separate version with the dynamics-residual figure aspect ratio.",
    )
    return parser.parse_args()


def _render_indices(point_count: int) -> np.ndarray:
    stride = max(1, int(np.ceil(point_count / MAX_RENDER_POINTS)))
    indices = np.arange(0, point_count, stride, dtype=int)
    if indices[-1] != point_count - 1:
        indices = np.append(indices, point_count - 1)
    return indices


def _lock_non_legend_interaction(axis) -> None:
    """Lock the 3D axes so only the legend remains draggable."""

    if hasattr(axis, "disable_mouse_rotation"):
        axis.disable_mouse_rotation()
    if hasattr(axis, "mouse_init"):
        axis.mouse_init(rotate_btn=[], zoom_btn=[])
    axis.set_navigate(False)
    axis.can_pan = lambda: False
    axis.can_zoom = lambda: False


def _build_figure(*, figsize: tuple[float, float] = DEFAULT_FIGSIZE) -> tuple[plt.Figure, str]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"archived data not found: {DATA_PATH}")
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(
            "t=0 rollout cache not found. Run "
            "plot_afm06a_x1_full_rollout_from_t0_grid.py first: "
            f"{CACHE_PATH}"
        )

    with np.load(DATA_PATH, allow_pickle=False) as data:
        true_t = np.asarray(data["t"], dtype=float)
        true_x1 = np.asarray(data["x1"], dtype=float)
        true_x2 = np.asarray(data["x2"], dtype=float)

    with CACHE_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    pred_t = np.asarray(payload["times"], dtype=float)
    pred_states = np.asarray(payload["predicted_states"], dtype=float)
    if pred_states.shape != (2, pred_t.size):
        raise ValueError(
            "expected predicted_states with shape (2, len(times)), "
            f"got {pred_states.shape} for times {pred_t.shape}"
        )

    true_idx = _render_indices(true_t.size)
    pred_idx = _render_indices(pred_t.size)

    true_t_ms = true_t[true_idx] * 1.0e3
    true_x1_nm = true_x1[true_idx] * 1.0e9
    true_x2_um_s = true_x2[true_idx] * 1.0e6

    pred_t_ms = pred_t[pred_idx] * 1.0e3
    pred_x1_nm = pred_states[0, pred_idx] * 1.0e9
    pred_x2_um_s = pred_states[1, pred_idx] * 1.0e6

    fig = plt.figure(figsize=figsize)
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(
        true_x1_nm,
        true_x2_um_s,
        true_t_ms,
        color="royalblue",
        linewidth=0.9,
        alpha=0.9,
        label="simulated true trajectory",
        zorder=2,
    )
    axis.plot(
        pred_x1_nm,
        pred_x2_um_s,
        pred_t_ms,
        color="darkorange",
        linestyle="--",
        linewidth=1.35,
        alpha=0.95,
        label="prediction",
        zorder=3,
    )
    axis.set_xlabel(r"Tip displacement, $x_1$ [nm]", labelpad=11, fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel(
        r"Tip velocity, $x_2$ [$\mu$m s$^{-1}$]",
        labelpad=13,
        fontsize=AXIS_FONT_SIZE,
    )
    axis.set_zlabel("")
    axis.set_zlim(float(true_t_ms[0]), float(true_t_ms[-1]))
    axis.view_init(elev=23, azim=-56)
    axis.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    axis.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    axis.tick_params(axis="z", labelsize=TICK_FONT_SIZE)
    axis.grid(True, alpha=0.22)
    legend = axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.90),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    legend.set_draggable(True, use_blit=True)
    if ADJUST_LEGEND:
        _lock_non_legend_interaction(axis)
    fig.text(
        0.885,
        0.52,
        "Time [ms]",
        rotation=90,
        ha="center",
        va="center",
        fontsize=AXIS_FONT_SIZE,
    )
    fig.subplots_adjust(left=0.01, right=0.94, bottom=0.03, top=0.99)
    summary = (
        f"true domain=[{true_t[0] * 1e3:.9g}, {true_t[-1] * 1e3:.9g}] ms | "
        f"prediction domain=[{pred_t[0] * 1e3:.9g}, {pred_t[-1] * 1e3:.9g}] ms | "
        f"rendered true={true_idx.size}, rendered prediction={pred_idx.size}"
    )
    return fig, summary


def _save(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    print(f"Saved: {output_path}")


def main() -> int:
    args = _parse_args()
    output_path = MATCH_DYNAMICS_OUTPUT_PATH if args.match_dynamics_aspect else OUTPUT_PATH
    figsize = MATCH_DYNAMICS_FIGSIZE if args.match_dynamics_aspect else DEFAULT_FIGSIZE
    fig, summary = _build_figure(figsize=figsize)
    if args.adjust_legend:
        saved = {"done": False}

        def save_once() -> None:
            if saved["done"]:
                return
            _save(fig, output_path)
            saved["done"] = True

        def on_key(event) -> None:
            if event.key == "s":
                saved["done"] = False
                save_once()

        def on_close(_event) -> None:
            save_once()

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", on_close)
        print("Drag the legend. Press 's' to save, or close the window to save.")
        plt.show()
    else:
        _save(fig, output_path)
        plt.close(fig)

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
