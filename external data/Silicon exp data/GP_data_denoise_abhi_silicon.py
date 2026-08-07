from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAT_PATH = (
    Path(r"E:\External data SI_SI hard sample raw")
    / "OneDrive_2_2026-7-26"
    / "dataAFM_Si"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)

INDEX = 109
EXPERIMENTAL_DT_S = 1.0 / 250.0e6
DEFAULT_START_INDEX = 0
DEFAULT_SAMPLE_COUNT = 10000

GP_LENGTH_SCALE_REFERENCE = 164.52e3
DRIVE_FREQUENCY_HZ = 164526.16455353564
N_WINDOW_PERIODS = 2

Z_STATIC_M = 100.0e-9
A0_M = 0.165e-9

CANTILEVER_STIFFNESS_N_M = 22.68
CANTILEVER_RESONANCE_HZ = 164.52e3
CANTILEVER_QUALITY_FACTOR = 428.0

SAMPLED_POINT_COUNT = 394
VAL_STRIDE = 5
VAL_OFFSET = 2
NONCONTACT_WEIGHT = 1.0
CONTACT_WEIGHT = 5.0
TRANSITION_WEIGHT = 10.0

AFM06A_PERIOD_S = 2.0 * np.pi / 11.804e3
AFM06A_TRANSITION_HALF_WIDTH_S = 10.688e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Denoise index109 raw x2 with GP, compute x2dot, and build the "
            "fixed two-period training window."
        )
    )
    parser.add_argument("--mat-path", type=Path, default=DEFAULT_MAT_PATH)
    parser.add_argument("--index", type=int, default=INDEX)
    parser.add_argument("--start-index", type=int, default=DEFAULT_START_INDEX)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument(
        "--skip-window",
        action="store_true",
        help="Only save the GP x2/x2dot outputs; do not build the two-period window.",
    )
    return parser.parse_args()


def load_index_signals(
    mat_path: Path,
    *,
    index: int,
    start_index: int,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = sio.loadmat(
        str(mat_path),
        variable_names=["displacement_fwd_sweep", "velocity_fwd_sweep"],
        struct_as_record=False,
        squeeze_me=True,
    )
    x1_all = np.asarray(data["displacement_fwd_sweep"][index], dtype=np.float64).ravel()
    x2_all = np.asarray(data["velocity_fwd_sweep"][index], dtype=np.float64).ravel()
    if x1_all.shape != x2_all.shape:
        raise ValueError("x1 and x2 arrays have different lengths")
    if start_index < 0 or sample_count <= 1:
        raise ValueError("start_index must be nonnegative and sample_count must exceed 1")

    end_index = min(start_index + sample_count, x2_all.size)
    if start_index >= end_index:
        raise ValueError("requested segment lies outside the trajectory")
    source_idx = np.arange(start_index, end_index, dtype=np.int64)
    time_s = source_idx.astype(np.float64) * EXPERIMENTAL_DT_S
    return x1_all[source_idx], x2_all[source_idx], time_s, source_idx


def fit_gp_x2(x2_raw: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    kernel = RBF(
        length_scale=1.0 / GP_LENGTH_SCALE_REFERENCE,
        length_scale_bounds=(1e-1 / GP_LENGTH_SCALE_REFERENCE, 1e3 / GP_LENGTH_SCALE_REFERENCE),
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(time_s.reshape(-1, 1), x2_raw)
    x2_hat, _sigma = gp.predict(time_s.reshape(-1, 1), return_std=True)
    return np.asarray(x2_hat, dtype=np.float64)


def save_gp_outputs(
    *,
    time_s: np.ndarray,
    source_idx: np.ndarray,
    x2_raw: np.ndarray,
    x2_hat: np.ndarray,
) -> tuple[Path, Path, np.ndarray]:
    x2dot_hat = np.gradient(x2_hat, time_s, edge_order=2)
    output_npz = SCRIPT_DIR / "silicon_gp_index109_x2_x2dot.npz"
    output_png = SCRIPT_DIR / "silicon_gp_index109_x2_x2dot.png"

    np.savez_compressed(
        output_npz,
        time_s=time_s,
        source_idx=source_idx,
        segment_start_index=np.array(int(source_idx[0]), dtype=np.int64),
        segment_end_index_exclusive=np.array(int(source_idx[-1]) + 1, dtype=np.int64),
        x2_raw_m_s=x2_raw,
        x2_hat_m_s=x2_hat,
        x2dot_hat_m_s2=x2dot_hat,
    )

    time_us = time_s * 1.0e6
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    axes[0].plot(time_us, x2_raw, color="black", linewidth=0.8)
    axes[1].plot(time_us, x2_hat, color="blue", linewidth=0.8)
    axes[2].plot(time_us, x2dot_hat, color="red", linewidth=0.8)
    axes[0].set_ylabel(r"$x_2^{\mathrm{raw}}$ [m s$^{-1}$]")
    axes[1].set_ylabel(r"$\hat{x}_2$ [m s$^{-1}$]")
    axes[2].set_ylabel(r"$\widehat{\dot{x}}_2$ [m s$^{-2}$]")
    axes[2].set_xlabel(r"Time [$\mu$s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_npz, output_png, x2dot_hat


def second_zero_crossing_index(x1_m: np.ndarray, time_s: np.ndarray) -> tuple[int, float]:
    crossings = np.flatnonzero(np.signbit(x1_m[:-1]) != np.signbit(x1_m[1:]))
    if crossings.size < 2:
        raise RuntimeError("fewer than two x1 zero crossings were found")
    left = int(crossings[1])
    right = left + 1
    x_left = float(x1_m[left])
    x_right = float(x1_m[right])
    t_left = float(time_s[left])
    t_right = float(time_s[right])
    zero_time_s = t_left - x_left * (t_right - t_left) / (x_right - x_left)
    nearest = left if abs(x_left) <= abs(x_right) else right
    return nearest, zero_time_s


def make_train_val_masks(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=int)
    val_idx = idx[((idx + 1 - VAL_OFFSET) % VAL_STRIDE) == 0]
    train_mask = np.ones(n, dtype=bool)
    train_mask[val_idx] = False
    return idx[train_mask], val_idx


def regime_sampling_weights(
    time_s: np.ndarray,
    contact: np.ndarray,
    transition_half_width_s: float,
) -> np.ndarray:
    weights = np.where(contact, CONTACT_WEIGHT, NONCONTACT_WEIGHT).astype(np.float64)
    switch_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1
    for index in switch_idx:
        boundary_time = 0.5 * (time_s[index - 1] + time_s[index])
        near_transition = np.abs(time_s - boundary_time) <= transition_half_width_s
        weights[near_transition] = TRANSITION_WEIGHT
    return weights


def weighted_fixed_count_indices(
    time_s: np.ndarray,
    contact: np.ndarray,
    transition_half_width_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if SAMPLED_POINT_COUNT > time_s.size:
        raise ValueError("not enough points to apply the fixed sampling count")
    density = regime_sampling_weights(time_s, contact, transition_half_width_s)
    cumulative = np.cumsum(density)
    targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), SAMPLED_POINT_COUNT)
    positions = np.searchsorted(cumulative, targets, side="left").astype(int)
    positions[0] = 0
    positions[-1] = time_s.size - 1
    for i in range(1, positions.size):
        positions[i] = max(positions[i], positions[i - 1] + 1)
    for i in range(positions.size - 2, -1, -1):
        positions[i] = min(positions[i], positions[i + 1] - 1)
    if np.any(np.diff(positions) <= 0):
        raise RuntimeError("failed to construct strictly increasing sample indices")
    return positions, density


def force_residual(
    *,
    x1_m: np.ndarray,
    x2_hat_m_s: np.ndarray,
    x2dot_hat_m_s2: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    omega0 = 2.0 * np.pi * CANTILEVER_RESONANCE_HZ
    mass = CANTILEVER_STIFFNESS_N_M / omega0**2
    damping = mass * omega0 / CANTILEVER_QUALITY_FACTOR
    residual = mass * x2dot_hat_m_s2 + damping * x2_hat_m_s + CANTILEVER_STIFFNESS_N_M * x1_m
    return residual, omega0, mass, damping


def save_training_window(
    *,
    x1_m: np.ndarray,
    x2_raw_m_s: np.ndarray,
    x2_hat_m_s: np.ndarray,
    x2dot_hat_m_s2: np.ndarray,
    time_s: np.ndarray,
    source_idx: np.ndarray,
) -> tuple[Path, Path, Path, Path]:
    start_idx, zero_time_s = second_zero_crossing_index(x1_m, time_s)
    period_s = 1.0 / DRIVE_FREQUENCY_HZ
    end_time_s = time_s[start_idx] + N_WINDOW_PERIODS * period_s
    end_idx = min(int(np.searchsorted(time_s, end_time_s, side="right")), time_s.size)
    window = slice(start_idx, end_idx)

    time_w = time_s[window]
    source_w = source_idx[window]
    x1_w = x1_m[window]
    x2_raw_w = x2_raw_m_s[window]
    x2_hat_w = x2_hat_m_s[window]
    x2dot_w = x2dot_hat_m_s2[window]
    separation_w = Z_STATIC_M + x1_w
    contact_w = separation_w <= A0_M

    transition_half_width_s = AFM06A_TRANSITION_HALF_WIDTH_S * period_s / AFM06A_PERIOD_S
    selected_idx, sampling_density = weighted_fixed_count_indices(
        time_w,
        contact_w,
        transition_half_width_s,
    )
    train_idx, val_idx = make_train_val_masks(selected_idx.size)
    train_window_idx = selected_idx[train_idx]
    val_window_idx = selected_idx[val_idx]

    f_residual_n, omega0, mass, damping = force_residual(
        x1_m=x1_w,
        x2_hat_m_s=x2_hat_w,
        x2dot_hat_m_s2=x2dot_w,
    )

    output_npz = SCRIPT_DIR / "silicon_gp_index109_training_window_2cycles.npz"
    output_json = SCRIPT_DIR / "silicon_gp_index109_training_window_2cycles.json"
    output_txt = SCRIPT_DIR / "silicon_gp_index109_training_window_2cycles_summary.txt"
    output_png = SCRIPT_DIR / "silicon_gp_index109_training_window_2cycles.png"

    np.savez_compressed(
        output_npz,
        time_s=time_w,
        source_idx=source_w,
        x1_raw_m=x1_w,
        x2_raw_m_s=x2_raw_w,
        x2_hat_m_s=x2_hat_w,
        x2dot_hat_m_s2=x2dot_w,
        f_residual_N=f_residual_n,
        separation_m=separation_w,
        contact=contact_w,
        sampling_density=sampling_density,
        selected_window_idx=selected_idx,
        train_idx=train_idx,
        val_idx=val_idx,
        train_window_idx=train_window_idx,
        val_window_idx=val_window_idx,
        selected_source_idx=source_w[selected_idx],
        train_source_idx=source_w[train_window_idx],
        val_source_idx=source_w[val_window_idx],
    )

    metadata = {
        "dataset": "silicon forward sweep",
        "index": INDEX,
        "window_start_time_s": float(time_w[0]),
        "window_end_time_s": float(time_w[-1]),
        "x1_second_zero_crossing_time_s": float(zero_time_s),
        "sample_count": int(time_w.size),
        "sampled_point_count": int(selected_idx.size),
        "train_point_count": int(train_idx.size),
        "validation_point_count": int(val_idx.size),
        "split_policy": "periodic 4:1 split, val_stride=5, val_offset=2",
        "sampling_policy": "regime_weighted_fixed_count",
        "sampling_weights": {
            "transition": TRANSITION_WEIGHT,
            "contact": CONTACT_WEIGHT,
            "noncontact": NONCONTACT_WEIGHT,
        },
        "contact_rule": "Z_static + x1 <= a0",
        "Z_static_m": Z_STATIC_M,
        "a0_m": A0_M,
        "transition_half_width_s": float(transition_half_width_s),
        "force_residual_formula": "m*x2dot_hat + c*x2_hat + k*x1_raw",
        "cantilever": {
            "k_N_m": CANTILEVER_STIFFNESS_N_M,
            "f0_Hz": CANTILEVER_RESONANCE_HZ,
            "omega0_rad_s": float(omega0),
            "Q": CANTILEVER_QUALITY_FACTOR,
            "m_kg": float(mass),
            "c_N_s_m": float(damping),
        },
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output_txt.write_text(
        "\n".join(
            [
                "Silicon index109 fixed training window",
                "======================================",
                f"Start time: {time_w[0] * 1e6:.9f} us",
                f"End time: {time_w[-1] * 1e6:.9f} us",
                f"Samples: {time_w.size}",
                f"Sampled points: {selected_idx.size}",
                f"Training points: {train_idx.size}",
                f"Validation points: {val_idx.size}",
                "Contact rule: Z_static + x1 <= a0",
                f"Z_static = {Z_STATIC_M:.12e} m",
                f"a0 = {A0_M:.12e} m",
                (
                    "Sampling weights, transition:contact:noncontact = "
                    f"{TRANSITION_WEIGHT:g}:{CONTACT_WEIGHT:g}:{NONCONTACT_WEIGHT:g}"
                ),
                "Force residual: m*x2dot_hat + c*x2_hat + k*x1_raw",
                f"m = {mass:.12e} kg",
                f"c = {damping:.12e} N*s/m",
                f"k = {CANTILEVER_STIFFNESS_N_M:.12e} N/m",
                f"F_residual range = [{np.min(f_residual_n):.12e}, {np.max(f_residual_n):.12e}] N",
                f"F_residual rms = {np.sqrt(np.mean(f_residual_n**2)):.12e} N",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    time_us = time_w * 1.0e6
    train_time_us = time_us[train_window_idx]
    val_time_us = time_us[val_window_idx]
    figure, axes = plt.subplots(
        5,
        1,
        figsize=(10, 11),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    series = [
        (x1_w * 1e9, r"$x_1^{\mathrm{raw}}$ [nm]"),
        (x2_raw_w, r"$x_2^{\mathrm{raw}}$ [m s$^{-1}$]"),
        (x2_hat_w, r"$\hat{x}_2$ [m s$^{-1}$]"),
        (x2dot_w, r"$\widehat{\dot{x}}_2$ [m s$^{-2}$]"),
        (f_residual_n * 1e9, r"$F_{\mathrm{residual}}$ [nN]"),
    ]
    for axis, (values, ylabel) in zip(axes, series):
        axis.plot(time_us, values, color="black", linewidth=0.7, alpha=0.65)
        axis.scatter(
            train_time_us,
            values[train_window_idx],
            s=16,
            color="#1f77b4",
            edgecolors="none",
            label="training data",
            zorder=4,
        )
        axis.scatter(
            val_time_us,
            values[val_window_idx],
            s=18,
            color="#d62728",
            edgecolors="none",
            label="validation data",
            zorder=5,
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[-1].set_xlabel(r"Time [$\mu$s]")
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_npz, output_json, output_txt, output_png


def main() -> None:
    args = parse_args()
    start_wall = time.time()
    x1, x2_raw, time_s, source_idx = load_index_signals(
        args.mat_path,
        index=args.index,
        start_index=args.start_index,
        sample_count=args.sample_count,
    )
    print(f"index={args.index}")
    print(f"samples=[{int(source_idx[0])}, {int(source_idx[-1]) + 1})")

    x2_hat = fit_gp_x2(x2_raw, time_s)
    gp_npz, gp_png, x2dot_hat = save_gp_outputs(
        time_s=time_s,
        source_idx=source_idx,
        x2_raw=x2_raw,
        x2_hat=x2_hat,
    )
    print(f"Saved: {gp_npz}")
    print(f"Saved: {gp_png}")

    if not args.skip_window:
        for output in save_training_window(
            x1_m=x1,
            x2_raw_m_s=x2_raw,
            x2_hat_m_s=x2_hat,
            x2dot_hat_m_s2=x2dot_hat,
            time_s=time_s,
            source_idx=source_idx,
        ):
            print(f"Saved: {output}")

    minutes, seconds = divmod(time.time() - start_wall, 60.0)
    print(f"Wall time: {int(minutes)} min {seconds:.1f} s")


if __name__ == "__main__":
    main()
