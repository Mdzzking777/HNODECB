"""Render the two AFM06a thesis figures as horizontal 1-by-2 panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, ConnectionPatch


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR.parent
ROLLOUT_STEM = "external_index109_full_rollout_landscape_best_Fd_phase_delta_c_zero_06a"
FORCE_STEM = "external_index109_raw_x1_trained_kan_fts_prediction_full"
OMEGA_D_RAD_S = 1033709.6467371855
PERIOD_S = 2.0 * np.pi / OMEGA_D_RAD_S
FONT_SCALE = 1.875
CONTACT_THRESHOLD_M = (0.165 - 100.0) * 1.0e-9


RC_PARAMS = {
    "font.size": 10.0 * FONT_SCALE,
    "axes.titlesize": 12.0 * FONT_SCALE,
    "axes.labelsize": 10.0 * FONT_SCALE,
    "xtick.labelsize": 10.0 * FONT_SCALE,
    "ytick.labelsize": 10.0 * FONT_SCALE,
    "legend.fontsize": 10.0 * FONT_SCALE,
}


def _middle_indices(time_s: np.ndarray) -> np.ndarray:
    start_s = 495.0e-6
    stop_s = start_s + 2.0 * PERIOD_S
    return np.flatnonzero((time_s >= start_s) & (time_s <= stop_s))


def _first_complete_contact_pair(
    time_s: np.ndarray,
    x1_m: np.ndarray,
) -> tuple[int, int]:
    contact = x1_m <= CONTACT_THRESHOLD_M
    transitions = np.flatnonzero(np.diff(contact.astype(np.int8)) != 0) + 1
    middle = _middle_indices(time_s)
    transitions = transitions[
        (transitions >= middle[0]) & (transitions <= middle[-1])
    ]
    for position, transition_in in enumerate(transitions[:-1]):
        if not contact[transition_in]:
            continue
        transition_out = transitions[position + 1]
        if not contact[transition_out]:
            return int(transition_in), int(transition_out)
    raise RuntimeError("No complete contact pulse was found in the middle window")


def _add_force_transition_inset(
    axis,
    time_us: np.ndarray,
    force_nN: np.ndarray,
    *,
    transition_us: float,
    bounds: tuple[float, float, float, float],
) -> None:
    half_width_us = 0.096
    selected = np.abs(time_us - transition_us) <= half_width_us
    if np.count_nonzero(selected) < 3:
        raise RuntimeError(f"Insufficient points near {transition_us:.6f} us")

    local_force = force_nN[selected]
    local_minimum = float(np.min(local_force))
    negative_scale = max(abs(min(local_minimum, 0.0)), 5.0e-4)
    y_min = local_minimum - 0.35 * negative_scale
    y_max = max(3.0e-3, 4.0 * negative_scale)

    inset = axis.inset_axes(bounds, zorder=8)
    inset.patch.set_alpha(0.0)
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_xlim(transition_us - half_width_us, transition_us + half_width_us)
    inset.set_ylim(y_min, y_max)

    circle = Circle(
        (0.5, 0.5),
        0.495,
        transform=inset.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=0,
    )
    inset.add_patch(circle)
    line, = inset.plot(
        time_us,
        force_nN,
        color="#d62728",
        linewidth=1.0,
        zorder=2,
    )
    line.set_clip_path(circle)

    connector = ConnectionPatch(
        xyA=(transition_us, -0.015),
        coordsA=axis.transData,
        xyB=(0.5, 0.01),
        coordsB=inset.transAxes,
        color="black",
        linewidth=0.8,
        zorder=7,
    )
    axis.add_artist(connector)


def _plot_rollout() -> None:
    with np.load(DATA_DIR / f"{ROLLOUT_STEM}.npz", allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        true_x1_m = np.asarray(data["external_true_x1_m"], dtype=float)
        predicted_x1_m = np.asarray(data["predicted_x1_m"], dtype=float)

    full_indices = np.arange(time_s.size, dtype=int)
    full_draw = full_indices
    if full_indices.size > 20_000:
        full_draw = full_indices[
            np.linspace(0, full_indices.size - 1, 20_000, dtype=int)
        ]
    panels = (
        ("Full time-span observation window", full_draw),
        ("Middle observation window", _middle_indices(time_s)),
    )

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14.160, 5.6505),
            gridspec_kw={"wspace": 0.16},
        )
        time_us = time_s * 1.0e6
        for panel_index, (axis, (title, indices)) in enumerate(zip(axes, panels, strict=True)):
            axis.plot(
                time_us[indices],
                true_x1_m[indices] * 1.0e9,
                color="black",
                linewidth=1.15,
                label=r"$x_1^{raw}$ experimental data",
                zorder=1,
            )
            axis.plot(
                time_us[indices],
                predicted_x1_m[indices] * 1.0e9,
                color="#d62728",
                linewidth=0.75,
                label="rollout prediction",
                zorder=2,
            )
            axis.set_title(title)
            axis.set_xlabel(r"Time [$\mu$s]")
            if panel_index == 0:
                axis.set_ylabel(r"$x_1$ [nm]")
                axis.legend(loc="upper right")
            axis.grid(True, alpha=0.25)
            axis.margins(x=0.0)
        fig.savefig(OUTPUT_DIR / f"{ROLLOUT_STEM}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_force() -> None:
    with np.load(DATA_DIR / f"{FORCE_STEM}.npz", allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        x1_raw_m = np.asarray(data["x1_raw_m"], dtype=float)
        predicted_fts_nN = np.asarray(data["fts_pred_nN"], dtype=float)

    panels = (
        ("Full time-span observation window", np.arange(time_s.size, dtype=int)),
        ("Middle observation window", _middle_indices(time_s)),
    )

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14.406, 5.6505),
            gridspec_kw={"wspace": 0.16},
        )
        time_us = time_s * 1.0e6
        for panel_index, (axis, (title, indices)) in enumerate(zip(axes, panels, strict=True)):
            axis.plot(
                time_us[indices],
                predicted_fts_nN[indices],
                color="#d62728",
                linewidth=0.70 if panel_index == 0 else 1.0,
                label="trained KAN prediction",
            )
            axis.set_title(title)
            axis.set_xlabel(r"Time [$\mu$s]")
            if panel_index == 0:
                axis.set_ylabel(r"$F_{ts}^{pred}$ [nN]")
                axis.legend(loc="upper right")
            axis.grid(True, alpha=0.25)
            axis.margins(x=0.0)

        transition_in, transition_out = _first_complete_contact_pair(time_s, x1_raw_m)
        _add_force_transition_inset(
            axes[1],
            time_us,
            predicted_fts_nN,
            transition_us=float(time_us[transition_in]),
            bounds=(0.44, 0.52, 0.18, 0.245),
        )
        _add_force_transition_inset(
            axes[1],
            time_us,
            predicted_fts_nN,
            transition_us=float(time_us[transition_out]),
            bounds=(0.68, 0.52, 0.18, 0.245),
        )
        fig.savefig(OUTPUT_DIR / f"{FORCE_STEM}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    _plot_rollout()
    _plot_force()
    print(f"Saved: {OUTPUT_DIR / f'{ROLLOUT_STEM}.png'}")
    print(f"Saved: {OUTPUT_DIR / f'{FORCE_STEM}.png'}")


if __name__ == "__main__":
    main()
