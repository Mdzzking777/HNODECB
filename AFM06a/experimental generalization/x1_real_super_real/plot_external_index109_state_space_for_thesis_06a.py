from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "external_index109_full_rollout_landscape_best_Fd_phase_delta_c_zero_06a.npz"
EXPERIMENTAL_DATA_PATH = (
    ROOT.parent.parent.parent
    / "external data"
    / "Silicon exp data"
    / "acceleration jump judge"
    / "index109_400_800us_transition_slope_jumps_data.npz"
)
TILDE_DATA_PATH = (
    ROOT
    / "for thesis"
    / "external_index109_x2_tilde_x2dot_tilde_400_450us.npz"
)
OUTPUT_PATH = (
    ROOT
    / "for thesis"
    / "external_index109_full_rollout_state_space_velocity_acceleration_time_06a.png"
)

WINDOW_MS = (0.4, 0.45)


def _load_or_build_tilde_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if TILDE_DATA_PATH.exists():
        with np.load(TILDE_DATA_PATH, allow_pickle=False) as data:
            return (
                np.asarray(data["time_s"], dtype=float),
                np.asarray(data["x2_tilde_m_s"], dtype=float),
                np.asarray(data["x2dot_tilde_m_s2"], dtype=float),
            )

    with np.load(EXPERIMENTAL_DATA_PATH, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        x2_raw_m_s = np.asarray(data["x2_raw_m_s"], dtype=float)
    time_ms = time_s * 1.0e3
    selected = (time_ms >= WINDOW_MS[0]) & (time_ms <= WINDOW_MS[1])
    time_s = time_s[selected]
    x2_raw_m_s = x2_raw_m_s[selected]

    time_us = time_s * 1.0e6
    time_mean = float(np.mean(time_us))
    time_scale = float(np.std(time_us))
    x2_mean = float(np.mean(x2_raw_m_s))
    x2_scale = float(np.std(x2_raw_m_s))
    normalized_time = ((time_us - time_mean) / time_scale).reshape(-1, 1)
    normalized_x2 = (x2_raw_m_s - x2_mean) / x2_scale
    fit_indices = np.arange(0, time_s.size, 4, dtype=int)
    if fit_indices[-1] != time_s.size - 1:
        fit_indices = np.append(fit_indices, time_s.size - 1)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=0.22, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=0.04, noise_level_bounds="fixed")
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=False,
        optimizer=None,
        n_restarts_optimizer=0,
        random_state=0,
    )
    gp.fit(normalized_time[fit_indices], normalized_x2[fit_indices])
    x2_tilde_m_s = gp.predict(normalized_time) * x2_scale + x2_mean
    x2dot_tilde_m_s2 = np.gradient(x2_tilde_m_s, time_s, edge_order=2)

    TILDE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TILDE_DATA_PATH,
        time_s=time_s,
        x2_tilde_m_s=x2_tilde_m_s,
        x2dot_tilde_m_s2=x2dot_tilde_m_s2,
    )
    return time_s, x2_tilde_m_s, x2dot_tilde_m_s2


def main() -> None:
    with np.load(DATA_PATH) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        velocity_m_s = np.asarray(data["predicted_x2_m_s"], dtype=float)

    (
        experimental_time_s,
        experimental_velocity_m_s,
        experimental_acceleration_m_s2,
    ) = _load_or_build_tilde_data()

    acceleration_m_s2 = np.gradient(velocity_m_s, time_s)
    time_ms = time_s * 1.0e3
    velocity_um_s = velocity_m_s * 1.0e6
    selected = (time_ms >= WINDOW_MS[0]) & (time_ms <= WINDOW_MS[1])
    experimental_time_ms = experimental_time_s * 1.0e3
    experimental_velocity_um_s = experimental_velocity_m_s * 1.0e6
    experimental_selected = (
        (experimental_time_ms >= WINDOW_MS[0])
        & (experimental_time_ms <= WINDOW_MS[1])
    )

    fig = plt.figure(figsize=(9.0, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        experimental_velocity_um_s[experimental_selected],
        experimental_acceleration_m_s2[experimental_selected],
        experimental_time_ms[experimental_selected],
        color="black",
        linewidth=0.55,
        label="smoothed experimental data",
        zorder=1,
    )
    ax.plot(
        velocity_um_s[selected],
        acceleration_m_s2[selected],
        time_ms[selected],
        color="#ff8c00",
        linestyle="--",
        linewidth=1.35,
        label="prediction",
        zorder=2,
    )

    ax.set_xlabel(r"Tip velocity, $\dot{x}_1$ [$\mu$m s$^{-1}$]", fontsize=16, labelpad=12)
    ax.set_ylabel(r"Tip acceleration, $\ddot{x}_1$ [m s$^{-2}$]", fontsize=16, labelpad=14)
    ax.set_zlabel("Time [ms]", fontsize=16, labelpad=8)
    ax.set_zlim(*WINDOW_MS)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.view_init(elev=24, azim=-56)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), frameon=False, fontsize=17)
    fig.subplots_adjust(left=0.01, right=0.86, bottom=0.02, top=0.98)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
