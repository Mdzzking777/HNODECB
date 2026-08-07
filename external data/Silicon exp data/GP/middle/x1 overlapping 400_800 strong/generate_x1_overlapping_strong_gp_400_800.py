from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


SCRIPT_DIR = Path(__file__).resolve().parent
SILICON_DIR = SCRIPT_DIR.parents[2]
LOCAL_MAT = SILICON_DIR / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
RAW_MAT = (
    Path(r"E:\External data SI_SI hard sample raw")
    / "OneDrive_2_2026-7-26"
    / "dataAFM_Si"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)

INDEX = 109
DT_S = 1.0 / 250.0e6
START_US = 400.0
END_US = 800.0
WINDOW_POINTS = 7000
STEP_POINTS = 5000
OVERLAP_POINTS = WINDOW_POINTS - STEP_POINTS

GP_RBF_LENGTH_SCALE = 0.22
GP_WHITE_NOISE_LEVEL = 0.04

CANTILEVER_STIFFNESS_N_M = 22.68
CANTILEVER_RESONANCE_HZ = 164.52e3
CANTILEVER_QUALITY_FACTOR = 428.0

OUTPUT_STEM = "silicon_index109_x1_overlapping_strong_gp_400_800"


def _load_raw_x1(source_idx: np.ndarray) -> tuple[np.ndarray, Path]:
    errors: list[str] = []
    for mat_path in (LOCAL_MAT, RAW_MAT):
        if not mat_path.is_file():
            errors.append(f"{mat_path} does not exist")
            continue
        try:
            data = sio.loadmat(
                str(mat_path),
                variable_names=["displacement_fwd_sweep"],
                squeeze_me=True,
                struct_as_record=False,
            )
            x1_all = np.asarray(data["displacement_fwd_sweep"][INDEX], dtype=float).ravel()
            if int(source_idx[0]) < 0 or int(source_idx[-1]) >= x1_all.size:
                raise RuntimeError(
                    f"requested source_idx [{source_idx[0]}, {source_idx[-1]}] "
                    f"is outside x1 length {x1_all.size}"
                )
            return x1_all[source_idx], mat_path
        except Exception as exc:
            errors.append(f"{mat_path}: {exc}")
    raise RuntimeError("Could not read raw x1 from MAT file:\n" + "\n".join(errors))


def _window_starts(n: int, window: int, step: int) -> np.ndarray:
    if n < window:
        raise ValueError("n must be at least window")
    starts = list(range(0, n - window + 1, step))
    final_start = n - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return np.asarray(starts, dtype=int)


def _stitch_weights(start: int, n: int, window: int, overlap: int) -> np.ndarray:
    weights = np.ones(window, dtype=float)
    if overlap <= 0:
        return weights
    phase = np.arange(overlap, dtype=float) / float(overlap - 1)
    fade_in = 0.5 - 0.5 * np.cos(np.pi * phase)
    fade_out = 0.5 + 0.5 * np.cos(np.pi * phase)
    if start > 0:
        weights[:overlap] = np.minimum(weights[:overlap], fade_in)
    if start + window < n:
        weights[-overlap:] = np.minimum(weights[-overlap:], fade_out)
    return weights


def _fit_fixed_strong_gp(time_s: np.ndarray, x1_raw_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    time_us = time_s * 1.0e6
    t_mean = float(np.mean(time_us))
    t_scale = float(np.std(time_us))
    y_mean = float(np.mean(x1_raw_m))
    y_scale = float(np.std(x1_raw_m))
    if t_scale <= 0.0 or y_scale <= 0.0:
        raise RuntimeError("invalid time/x1 scale")

    x = ((time_us - t_mean) / t_scale).reshape(-1, 1)
    y = (x1_raw_m - y_mean) / y_scale
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=GP_RBF_LENGTH_SCALE, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=GP_WHITE_NOISE_LEVEL, noise_level_bounds="fixed")
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=False,
        optimizer=None,
        n_restarts_optimizer=0,
        random_state=0,
    )
    gp.fit(x, y)
    y_tilde, y_std = gp.predict(x, return_std=True)
    x1_tilde = y_tilde * y_scale + y_mean
    x1_std = y_std * y_scale
    metadata: dict[str, float | str] = {
        "kernel_initial": str(kernel),
        "kernel_optimized": str(gp.kernel_),
        "time_us_mean": t_mean,
        "time_us_scale": t_scale,
        "x1_raw_mean_m": y_mean,
        "x1_raw_scale_m": y_scale,
    }
    return x1_tilde, x1_std, metadata


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> int:
    start_wall = time.time()
    start_index = int(round((START_US * 1.0e-6) / DT_S))
    sample_count = int(round(((END_US - START_US) * 1.0e-6) / DT_S))
    source_idx = np.arange(start_index, start_index + sample_count, dtype=int)
    time_s = source_idx.astype(float) * DT_S
    x1_raw, mat_path = _load_raw_x1(source_idx)

    n = int(time_s.size)
    starts = _window_starts(n, WINDOW_POINTS, STEP_POINTS)
    weighted_sum = np.zeros(n, dtype=float)
    weighted_std_sum = np.zeros(n, dtype=float)
    weight_sum = np.zeros(n, dtype=float)
    window_seconds: list[float] = []
    window_metadata: list[dict[str, float | int | str]] = []

    for i, start in enumerate(starts, start=1):
        stop = int(start + WINDOW_POINTS)
        wtime0 = time.time()
        x1_window, x1_std_window, gp_meta = _fit_fixed_strong_gp(
            time_s=time_s[start:stop],
            x1_raw_m=x1_raw[start:stop],
        )
        elapsed = time.time() - wtime0
        weights = _stitch_weights(start, n, WINDOW_POINTS, OVERLAP_POINTS)
        weighted_sum[start:stop] += weights * x1_window
        weighted_std_sum[start:stop] += weights * x1_std_window
        weight_sum[start:stop] += weights
        window_seconds.append(elapsed)
        window_metadata.append(
            {
                "window_number": i,
                "start_local_index": int(start),
                "stop_local_index_exclusive": int(stop),
                "start_time_us": float(time_s[start] * 1.0e6),
                "stop_time_us": float(time_s[stop - 1] * 1.0e6),
                "wall_time_s": float(elapsed),
                **gp_meta,
            }
        )
        print(
            f"[{i:02d}/{starts.size}] "
            f"{time_s[start] * 1.0e6:.3f}-{time_s[stop - 1] * 1.0e6:.3f} us "
            f"done in {elapsed:.1f} s",
            flush=True,
        )

    if np.any(weight_sum <= 0.0):
        raise RuntimeError("some samples were not covered by overlapping GP windows")
    x1_tilde = weighted_sum / weight_sum
    x1_std = weighted_std_sum / weight_sum
    x1dot_tilde = np.gradient(x1_tilde, time_s, edge_order=2)
    x1ddot_tilde = np.gradient(x1dot_tilde, time_s, edge_order=2)

    omega0 = 2.0 * np.pi * CANTILEVER_RESONANCE_HZ
    mass_kg = CANTILEVER_STIFFNESS_N_M / (omega0 * omega0)
    damping_n_s_m = mass_kg * omega0 / CANTILEVER_QUALITY_FACTOR
    inertial_force = mass_kg * x1ddot_tilde
    damping_force = damping_n_s_m * x1dot_tilde
    stiffness_force = CANTILEVER_STIFFNESS_N_M * x1_tilde
    f_residual = inertial_force + damping_force + stiffness_force

    output_npz = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"

    np.savez_compressed(
        output_npz,
        time_s=time_s,
        time_us=time_s * 1.0e6,
        source_idx=source_idx,
        x1_raw_m=x1_raw,
        x1_tilde_overlapping_strong_m=x1_tilde,
        x1_tilde_overlapping_strong_std_m=x1_std,
        x1_tilde_overlapping_strong_dot_m_s=x1dot_tilde,
        x1_tilde_overlapping_strong_ddot_m_s2=x1ddot_tilde,
        f_residual_N=f_residual,
        inertial_force_N=inertial_force,
        damping_force_N=damping_force,
        stiffness_force_N=stiffness_force,
        weight_sum=weight_sum,
        window_starts=starts,
        window_points=np.array(WINDOW_POINTS, dtype=int),
        step_points=np.array(STEP_POINTS, dtype=int),
        overlap_points=np.array(OVERLAP_POINTS, dtype=int),
        k_N_m=np.array(CANTILEVER_STIFFNESS_N_M, dtype=float),
        f0_Hz=np.array(CANTILEVER_RESONANCE_HZ, dtype=float),
        omega0_rad_s=np.array(omega0, dtype=float),
        Q=np.array(CANTILEVER_QUALITY_FACTOR, dtype=float),
        m_kg=np.array(mass_kg, dtype=float),
        c_N_s_m=np.array(damping_n_s_m, dtype=float),
    )

    total_elapsed = time.time() - start_wall
    summary = {
        "method": "overlapping sliding-window fixed strong GP for x1 raw over 400-800 us",
        "raw_x1_source_mat": str(mat_path),
        "raw_x1_source_variable": f"displacement_fwd_sweep[{INDEX}]",
        "time_range_us": [float(time_s[0] * 1.0e6), float(time_s[-1] * 1.0e6)],
        "sample_count": n,
        "dt_s": DT_S,
        "window_points": WINDOW_POINTS,
        "step_points": STEP_POINTS,
        "overlap_points": OVERLAP_POINTS,
        "window_count": int(starts.size),
        "stitching": "cosine fade-in/fade-out weighted average in overlap regions",
        "gp": {
            "kernel": f"Constant(1)*RBF(length_scale={GP_RBF_LENGTH_SCALE})+WhiteKernel(noise_level={GP_WHITE_NOISE_LEVEL})",
            "optimizer": "None",
            "standardization": "time and x1 are standardized independently inside each window",
        },
        "wall_time_s": total_elapsed,
        "window_wall_time_s": window_seconds,
        "window_wall_time_mean_s": float(np.mean(window_seconds)),
        "window_wall_time_median_s": float(np.median(window_seconds)),
        "window_metadata": window_metadata,
        "cantilever": {
            "k_N_m": CANTILEVER_STIFFNESS_N_M,
            "f0_Hz": CANTILEVER_RESONANCE_HZ,
            "omega0_rad_s": omega0,
            "Q": CANTILEVER_QUALITY_FACTOR,
            "m_kg": mass_kg,
            "c_N_s_m": damping_n_s_m,
            "source_note": "k, f0, and Q follow the external silicon notebook/script constants; m and c are derived.",
        },
        "x1_raw_stats_nm": _stats(x1_raw * 1.0e9),
        "x1_tilde_stats_nm": _stats(x1_tilde * 1.0e9),
        "x1dot_tilde_stats_mm_s": _stats(x1dot_tilde * 1.0e3),
        "x1ddot_tilde_stats_m_s2": _stats(x1ddot_tilde),
        "f_residual_stats_nN": _stats(f_residual * 1.0e9),
        "outputs": {
            "npz": str(output_npz),
            "json": str(output_json),
            "png": str(output_png),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13.5, 10.5),
        sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    axes[0].plot(time_us, x1_raw * 1.0e9, color="0.65", linewidth=0.45, label=r"$x_1^{raw}$")
    axes[0].plot(time_us, x1_tilde * 1.0e9, color="black", linewidth=0.85, label=r"$\tilde{x}_{1,\mathrm{overlap}}$")
    axes[0].set_ylabel(r"$x_1$ [nm]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_us, x1dot_tilde * 1.0e3, color="black", linewidth=0.85)
    axes[1].set_ylabel(r"$\dot{\tilde{x}}_1$ [mm s$^{-1}$]")

    axes[2].plot(time_us, x1ddot_tilde, color="black", linewidth=0.85)
    axes[2].set_ylabel(r"$\ddot{\tilde{x}}_1$ [m s$^{-2}$]")

    axes[3].plot(time_us, f_residual * 1.0e9, color="black", linewidth=0.85)
    axes[3].set_ylabel(r"$F_{\mathrm{residual}}$ [nN]")
    axes[3].set_xlabel(r"Time [$\mu$s]")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_npz}")
    print(f"Saved: {output_json}")
    print(f"Saved: {output_png}")
    print(f"Total wall time: {total_elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
