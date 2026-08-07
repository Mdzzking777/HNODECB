"""Interactively adjust annotations for AFM06a archive figures.

Usage
-----
Rollout force figure:
    python adjust_archive_figure_annotations.py --figure rollout

State-space figure:
    python adjust_archive_figure_annotations.py --figure state-space

Drag the legend/text with the mouse. Press ``s`` to save, or close the
window to save automatically.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import numpy as np


ARCHIVE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARCHIVE_DIR.parents[1]

ROLLOUT_OUTPUT = ARCHIVE_DIR / "rollout.png"
STATE_SPACE_OUTPUT = ARCHIVE_DIR / "afm06_state_space_x1_x2_time_full.png"


class DraggableAxesText:
    """Drag a Text artist while storing its position in axes coordinates."""

    def __init__(self, text_artist):
        self.text = text_artist
        self.canvas = text_artist.figure.canvas
        self.press_offset: tuple[float, float] | None = None
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def on_press(self, event) -> None:
        if event.button != 1 or event.x is None or event.y is None:
            return
        contains, _ = self.text.contains(event)
        if not contains:
            return
        text_display = self.text.get_transform().transform(self.text.get_position())
        self.press_offset = (text_display[0] - event.x, text_display[1] - event.y)

    def on_motion(self, event) -> None:
        if self.press_offset is None or event.x is None or event.y is None:
            return
        display_position = (
            event.x + self.press_offset[0],
            event.y + self.press_offset[1],
        )
        axes_position = self.text.axes.transAxes.inverted().transform(display_position)
        self.text.set_position(axes_position)
        self.canvas.draw_idle()

    def on_release(self, event) -> None:
        self.press_offset = None


def load_rollout_series():
    helper_path = (
        ARCHIVE_DIR
        / "st2l"
        / "rank1_20260728_213028"
        / "visualization"
        / "adjust_rollout_state_space_legends.py"
    )
    spec = importlib.util.spec_from_file_location("adjust_rollout_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper script: {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.left_panel_series()


def build_rollout_figure():
    import matplotlib.pyplot as plt

    times_ms, truth, pred_times_ms, pred, rel_rmse = load_rollout_series()
    blue = "#4C78FF"
    orange = "#FF8C00"

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 7.1), dpi=160)
    ax.plot(times_ms, truth, color=blue, linewidth=1.55, label="simulated true trajectory")
    ax.plot(pred_times_ms, pred, color=orange, linestyle="--", linewidth=1.55, label="prediction")
    rmse_text = ax.text(
        0.025,
        0.91,
        f"relative RMSE = {rel_rmse:.3f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
    )
    rmse_text.set_picker(True)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$\bar{F}_{ts}$ (m s$^{-2}$)")
    ax.grid(True, alpha=0.25)
    legend = ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.88), frameon=True)
    legend.set_draggable(True, use_blit=True)
    ax.text(-0.085, 1.035, "(a)", transform=ax.transAxes, ha="left", va="bottom", fontsize=22)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.93)
    fig._draggable_rmse_text = DraggableAxesText(rmse_text)
    fig._archive_save_dpi = 300
    fig._archive_save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
    return fig, ROLLOUT_OUTPUT


def render_indices(point_count: int, max_points: int = 25_000) -> np.ndarray:
    stride = max(1, int(np.ceil(point_count / max_points)))
    indices = np.arange(0, point_count, stride, dtype=int)
    if indices[-1] != point_count - 1:
        indices = np.append(indices, point_count - 1)
    return indices


def build_state_space_figure():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from PIL import Image

    image = Image.open(STATE_SPACE_OUTPUT).convert("RGB")
    dpi = 160
    fig, ax = plt.subplots(figsize=(image.width / dpi, image.height / dpi), dpi=dpi)
    ax.imshow(image, extent=(0, image.width, image.height, 0))
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.axis("off")

    # Cover only the baked-in legend area from the static PNG. The plotted data
    # are preserved as the background image; only the legend is replaced with a
    # draggable Matplotlib artist.
    ax.add_patch(
        Rectangle(
            (0, 0),
            860,
            240,
            facecolor="white",
            edgecolor="none",
            zorder=2,
        )
    )
    handles = [
        Line2D([0], [0], color="#4C78FF", linewidth=2.5),
        Line2D([0], [0], color="#FF8C00", linestyle="--", linewidth=2.5),
    ]
    legend = ax.legend(
        handles,
        ["simulated true trajectory", "prediction"],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        fontsize=21,
    )
    legend.set_draggable(True, use_blit=True)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig._archive_save_dpi = dpi
    fig._archive_save_kwargs = {"pad_inches": 0}
    return fig, STATE_SPACE_OUTPUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        choices=("rollout", "state-space"),
        required=True,
        help="Which archive figure to adjust.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Regenerate and save without opening the interactive window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.export_only:
        import matplotlib

        matplotlib.use("Agg")
    else:
        import matplotlib

        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass

    import matplotlib.pyplot as plt

    if args.figure == "rollout":
        fig, output = build_rollout_figure()
        print("Drag the rollout legend and/or relative RMSE text.")
    else:
        fig, output = build_state_space_figure()
        print("Drag the state-space legend.")

    def save() -> None:
        save_dpi = getattr(fig, "_archive_save_dpi", 300)
        save_kwargs = getattr(
            fig,
            "_archive_save_kwargs",
            {"bbox_inches": "tight", "pad_inches": 0.04},
        )
        fig.savefig(output, dpi=save_dpi, **save_kwargs)
        print(f"Saved: {output}")

    if args.export_only:
        save()
        plt.close(fig)
        return 0

    def on_key(event) -> None:
        if event.key and event.key.lower() == "s":
            save()

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Press 's' to save, or close the window to save automatically.")
    plt.show()
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
