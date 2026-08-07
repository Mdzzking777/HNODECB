"""Full-resolution rollout at the best point of the Fd-phase landscape."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

full_rollout = importlib.import_module(
    "run_external_index109_full_rollout_06a_framework"
)
fit_module = importlib.import_module(
    "optimize_external_index109_Fd_phase_contact_damping_06a"
)

LANDSCAPE_JSON = (
    SCRIPT_DIR / "external_index109_Fd_phase_loss_landscape_delta_c_zero_06a.json"
)
OUTPUT_STEM = (
    "external_index109_full_rollout_landscape_best_Fd_phase_delta_c_zero_06a"
)


def _plot(
    time_s: np.ndarray,
    true_x1_m: np.ndarray,
    predicted_x1_m: np.ndarray,
    fd_nN: float,
    phase_rad: float,
    output_path: Path,
) -> None:
    period_s = 2.0 * np.pi / 1033709.6467371855
    centers = (0.10 * time_s[-1], 0.50 * time_s[-1], 0.90 * time_s[-1])
    panels: list[tuple[str, np.ndarray]] = [
        ("Full time-span observation window", np.arange(time_s.size, dtype=int))
    ]
    for label, center in zip(
        (
            "Early observation window",
            "Middle observation window",
            "Tail observation window",
        ),
        centers,
        strict=True,
    ):
        panels.append(
            (
                label,
                np.flatnonzero(
                    (time_s >= center - period_s) & (time_s <= center + period_s)
                ),
            )
        )

    font_scale = 1.875
    with plt.rc_context(
        {
            "font.size": 10.0 * font_scale,
            "axes.titlesize": 12.0 * font_scale,
            "axes.labelsize": 10.0 * font_scale,
            "xtick.labelsize": 10.0 * font_scale,
            "ytick.labelsize": 10.0 * font_scale,
            "legend.fontsize": 10.0 * font_scale,
        }
    ):
        fig, axes_grid = plt.subplots(
            2,
            2,
            figsize=(14.160, 11.301),
            gridspec_kw={"hspace": 0.38, "wspace": 0.16},
        )
        axes = tuple(axes_grid.ravel())
        time_us = time_s * 1.0e6
        for panel_index, (axis, (label, indices)) in enumerate(
            zip(axes, panels, strict=True)
        ):
            draw = indices
            if label.startswith("Full") and indices.size > 20_000:
                draw = indices[np.linspace(0, indices.size - 1, 20_000, dtype=int)]
            axis.plot(
                time_us[draw],
                true_x1_m[draw] * 1.0e9,
                color="black",
                linewidth=1.15,
                label=r"$x_1^{raw}$ experimental data",
                zorder=1,
            )
            axis.plot(
                time_us[draw],
                predicted_x1_m[draw] * 1.0e9,
                color="#d62728",
                linewidth=0.75,
                label="rollout prediction",
                zorder=2,
            )
            axis.set_title(label)
            axis.set_xlabel(r"Time [$\mu$s]")
            if panel_index in (0, 2):
                axis.set_ylabel(r"$x_1$ [nm]")
            axis.grid(True, alpha=0.25)
            axis.margins(x=0.0)
            if panel_index == 0:
                axis.legend(loc="upper right")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    landscape = json.loads(LANDSCAPE_JSON.read_text(encoding="utf-8"))
    fd_nN = float(landscape["grid_minimum"]["Fd_nN"])
    phase_rad = float(landscape["grid_minimum"]["phase_rad"])

    model, _, _, _, _ = full_rollout.pointwise._restore_stage2_model(
        full_rollout.ARCHIVE_DIR
    )
    time_s, true_x1_m, mat_source = full_rollout.pointwise._load_external_x1_raw(
        None
    )
    true_x2_m_s = full_rollout.one_step._load_external_x2_raw(mat_source)
    parameters = dict(full_rollout.one_step._load_rhs_parameters())
    fitted_x2_t0, initial_fit = full_rollout._fit_initial_velocity_from_x1(
        time_s, true_x1_m, float(parameters["omega0"])
    )
    initial_state = np.asarray((true_x1_m[0], fitted_x2_t0), dtype=np.float64)
    force_x1_grid_m, force_grid_N = full_rollout._build_force_lookup(
        model, true_x1_m
    )
    trajectory, predicted_force_N, contact = fit_module._rollout(
        time_s,
        initial_state,
        force_x1_grid_m,
        force_grid_N,
        parameters,
        fd_nN * 1.0e-9,
        phase_rad,
        0.0,
    )

    output_npz = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
    np.savez_compressed(
        output_npz,
        time_s=time_s,
        external_true_x1_m=true_x1_m,
        external_true_x2_m_s=true_x2_m_s,
        predicted_x1_m=trajectory[0],
        predicted_x2_m_s=trajectory[1],
        predicted_fts_N=predicted_force_N,
        predicted_contact_mask=contact,
        landscape_best_Fd_N=np.asarray(fd_nN * 1.0e-9),
        landscape_best_phase_rad=np.asarray(phase_rad),
        fixed_delta_c_contact_N_s_m=np.asarray(0.0),
    )
    _plot(time_s, true_x1_m, trajectory[0], fd_nN, phase_rad, output_png)

    period_s = 2.0 * np.pi / float(parameters["omega0"])
    centers = (0.10 * time_s[-1], 0.50 * time_s[-1], 0.90 * time_s[-1])
    amplitudes: dict[str, dict[str, float]] = {}
    for label, center in zip(("early", "middle", "tail"), centers, strict=True):
        mask = np.abs(time_s - center) <= period_s
        amplitudes[label] = {
            "external_true_half_amplitude_nm": (
                fit_module._half_amplitude(true_x1_m[mask]) * 1.0e9
            ),
            "predicted_half_amplitude_nm": (
                fit_module._half_amplitude(trajectory[0, mask]) * 1.0e9
            ),
        }
    summary = {
        "objective": "full-span normalized mean squared x1 trajectory error",
        "parameter_source": str(LANDSCAPE_JSON),
        "free_parameters_in_landscape": ["Fd_N", "drive_phase_rad"],
        "fixed_delta_c_contact_N_s_m": 0.0,
        "initial_state_SI": {
            "x1_m": float(initial_state[0]),
            "x2_m_s": float(initial_state[1]),
            "x2_policy": "periodic fit to full external x1 raw",
        },
        "landscape_grid_point": {
            "Fd_N": fd_nN * 1.0e-9,
            "Fd_nN": fd_nN,
            "phase_rad": phase_rad,
            "phase_deg": float(np.degrees(phase_rad)),
            "delta_c_contact_N_s_m": 0.0,
            "delta_c_contact_over_c_air": 0.0,
            "c_contact_total_N_s_m": float(parameters["c"]),
        },
        "full_resolution_metrics": {
            "x1_relative_rmse_pct": fit_module._relative_rmse_pct(
                trajectory[0], true_x1_m
            ),
            "x2_relative_rmse_pct": fit_module._relative_rmse_pct(
                trajectory[1], true_x2_m_s
            ),
            "predicted_contact_fraction": float(np.mean(contact)),
            "window_half_amplitudes": amplitudes,
        },
        "initial_velocity_fit": initial_fit,
        "outputs": {"npz": str(output_npz), "png": str(output_png)},
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {output_npz}")
    print(f"Saved: {output_png}")
    print(f"Saved: {output_json}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
