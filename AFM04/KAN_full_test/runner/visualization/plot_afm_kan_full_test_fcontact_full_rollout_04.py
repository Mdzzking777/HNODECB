from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.runner.visualization._common import OUT_DIR, finalize_and_save
from AFM04.KAN_full_test.runner.visualization._full_rollout_common import (
    compute_rollout_fts,
    figure_title,
    panel_title,
    prepare_full_rollout_context,
    true_based_ylim,
)


def main() -> None:
    ctx = prepare_full_rollout_context()
    fts_true, fts_pred = compute_rollout_fts(ctx)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=False, squeeze=False)
    flat_axes = list(axes.ravel())
    for ax, (name, start, stop) in zip(flat_axes, ctx.panels, strict=False):
        panel_plot_pos = ctx.plot_pos[(ctx.plot_idx >= start) & (ctx.plot_idx <= stop)]
        ax.plot(
            ctx.times_us[panel_plot_pos],
            fts_true[panel_plot_pos],
            color="black",
            linewidth=2.0,
            label="Fts true",
            zorder=1,
        )
        ax.plot(
            ctx.times_us[panel_plot_pos],
            fts_pred[panel_plot_pos],
            color="crimson",
            linewidth=1.5,
            linestyle="-",
            label="Fts pred",
            zorder=3,
        )
        ax.set_title(panel_title(name, start, stop, ctx.times_rollout_full))
        ax.set_xlabel("time (us)")
        ax.set_ylabel("force (N)")
        ax.set_ylim(*true_based_ylim(fts_true[panel_plot_pos]))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    for ax in flat_axes[len(ctx.panels):]:
        ax.axis("off")

    fig.suptitle(figure_title(ctx, "Fts"), fontsize=14, y=0.98)
    finalize_and_save(fig, OUT_DIR / "afm_param_kan_full_test_04_fcontact_full_rollout_grid.png")


if __name__ == "__main__":
    main()
