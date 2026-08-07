from __future__ import annotations

import json
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
CONTACT_JUDGE_NPZ = (
    SILICON_DIR
    / "GP"
    / "middle"
    / "x1 for contact judge"
    / "silicon_gp_index109_mid5000_490_510us_x1_contact_judge.npz"
)
LOCAL_MAT = SILICON_DIR / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
RAW_MAT = (
    Path(r"E:\External data SI_SI hard sample raw")
    / "OneDrive_2_2026-7-26"
    / "dataAFM_Si"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)

INDEX = 109
MARGIN_FRACTION = 0.20
OUTPUT_STEM = "silicon_middle_x1_strong_gp_margin20"

# Same fixed strong-smoothing GP policy as the x2 pure-GP background.
GP_RBF_LENGTH_SCALE = 0.22
GP_WHITE_NOISE_LEVEL = 0.04

# External silicon constants used by the local notebook/script.
CANTILEVER_STIFFNESS_N_M = 22.68
CANTILEVER_RESONANCE_HZ = 164.52e3
CANTILEVER_QUALITY_FACTOR = 428.0


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


def _fit_fixed_strong_gp(
    *,
    time_s: np.ndarray,
    x1_raw_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
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
        "rbf_length_scale_standardized_time": GP_RBF_LENGTH_SCALE,
        "white_noise_level_standardized_x1": GP_WHITE_NOISE_LEVEL,
        "smoothing_policy": (
            "fixed strong GP on extended x1_raw; standardized time and x1; "
            "same Constant(1)*RBF(0.22)+White(0.04), optimizer=None policy as x2 pure GP"
        ),
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
    with np.load(CONTACT_JUDGE_NPZ, allow_pickle=False) as data:
        middle_time_s = np.asarray(data["time_s"], dtype=float)
        middle_source_idx = np.asarray(data["source_idx"], dtype=int)
        z_static_m = float(np.asarray(data["z_static_m"], dtype=float))
        a0_m = float(np.asarray(data["a0_m"], dtype=float))
        contact_judge_transition_idx = np.asarray(data["transition_idx"], dtype=int)

    middle_n = int(middle_source_idx.size)
    margin_n = int(round(MARGIN_FRACTION * middle_n))
    extended_start = int(middle_source_idx[0]) - margin_n
    extended_stop_exclusive = int(middle_source_idx[-1]) + margin_n + 1
    extended_source_idx = np.arange(extended_start, extended_stop_exclusive, dtype=int)
    middle_start_in_extended = margin_n
    middle_stop_in_extended = margin_n + middle_n
    middle_slice = slice(middle_start_in_extended, middle_stop_in_extended)

    x1_raw_extended, mat_path = _load_raw_x1(extended_source_idx)
    dt_s = float(np.median(np.diff(middle_time_s)))
    extended_time_s = extended_source_idx.astype(float) * dt_s
    if not np.allclose(extended_time_s[middle_slice], middle_time_s, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("extended time grid does not align with middle time grid")

    x1_tilde_extended, x1_std_extended, gp_metadata = _fit_fixed_strong_gp(
        time_s=extended_time_s,
        x1_raw_m=x1_raw_extended,
    )
    x1dot_tilde_extended = np.gradient(x1_tilde_extended, extended_time_s, edge_order=2)
    x1ddot_tilde_extended = np.gradient(x1dot_tilde_extended, extended_time_s, edge_order=2)

    x1_tilde_middle = x1_tilde_extended[middle_slice]
    x1dot_tilde_middle = x1dot_tilde_extended[middle_slice]
    x1ddot_tilde_middle = x1ddot_tilde_extended[middle_slice]

    omega0 = 2.0 * np.pi * CANTILEVER_RESONANCE_HZ
    mass_kg = CANTILEVER_STIFFNESS_N_M / (omega0 * omega0)
    damping_n_s_m = mass_kg * omega0 / CANTILEVER_QUALITY_FACTOR
    inertial_force = mass_kg * x1ddot_tilde_middle
    damping_force = damping_n_s_m * x1dot_tilde_middle
    stiffness_force = CANTILEVER_STIFFNESS_N_M * x1_tilde_middle
    f_residual = inertial_force + damping_force + stiffness_force

    contact_middle = (z_static_m + x1_tilde_middle) <= a0_m
    transition_idx_middle = contact_judge_transition_idx

    output_npz = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    output_json = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"

    np.savez_compressed(
        output_npz,
        extended_time_s=extended_time_s,
        extended_time_us=extended_time_s * 1.0e6,
        extended_source_idx=extended_source_idx,
        extended_x1_raw_m=x1_raw_extended,
        extended_x1_tilde_strong_m=x1_tilde_extended,
        extended_x1_tilde_strong_std_m=x1_std_extended,
        extended_x1_tilde_strong_dot_m_s=x1dot_tilde_extended,
        extended_x1_tilde_strong_ddot_m_s2=x1ddot_tilde_extended,
        middle_time_s=middle_time_s,
        middle_time_us=middle_time_s * 1.0e6,
        middle_source_idx=middle_source_idx,
        middle_x1_tilde_strong_m=x1_tilde_middle,
        middle_x1_tilde_strong_dot_m_s=x1dot_tilde_middle,
        middle_x1_tilde_strong_ddot_m_s2=x1ddot_tilde_middle,
        middle_contact=contact_middle,
        middle_transition_idx=transition_idx_middle,
        middle_inertial_force_N=inertial_force,
        middle_damping_force_N=damping_force,
        middle_stiffness_force_N=stiffness_force,
        middle_f_residual_N=f_residual,
        margin_fraction=np.array(MARGIN_FRACTION, dtype=float),
        margin_points_per_side=np.array(margin_n, dtype=int),
        z_static_m=np.array(z_static_m, dtype=float),
        a0_m=np.array(a0_m, dtype=float),
        k_N_m=np.array(CANTILEVER_STIFFNESS_N_M, dtype=float),
        f0_Hz=np.array(CANTILEVER_RESONANCE_HZ, dtype=float),
        omega0_rad_s=np.array(omega0, dtype=float),
        Q=np.array(CANTILEVER_QUALITY_FACTOR, dtype=float),
        m_kg=np.array(mass_kg, dtype=float),
        c_N_s_m=np.array(damping_n_s_m, dtype=float),
    )

    summary = {
        "method": "x1 fixed strong GP with 20% margin on both sides; derivatives computed on extended window, then cropped to middle",
        "raw_x1_source_mat": str(mat_path),
        "raw_x1_source_variable": f"displacement_fwd_sweep[{INDEX}]",
        "contact_judge_source_npz": str(CONTACT_JUDGE_NPZ),
        "middle_time_range_us": [float(middle_time_s[0] * 1.0e6), float(middle_time_s[-1] * 1.0e6)],
        "extended_time_range_us": [float(extended_time_s[0] * 1.0e6), float(extended_time_s[-1] * 1.0e6)],
        "middle_sample_count": middle_n,
        "margin_fraction_each_side": MARGIN_FRACTION,
        "margin_points_each_side": margin_n,
        "extended_sample_count": int(extended_source_idx.size),
        "x1_derivative_method": "np.gradient on extended x1_tilde_strong, edge_order=2; crop to middle afterward",
        "f_residual_definition": "middle_F_residual = m*x1ddot_tilde_strong + c*x1dot_tilde_strong + k*x1_tilde_strong",
        "gp": gp_metadata,
        "cantilever": {
            "k_N_m": CANTILEVER_STIFFNESS_N_M,
            "f0_Hz": CANTILEVER_RESONANCE_HZ,
            "omega0_rad_s": omega0,
            "Q": CANTILEVER_QUALITY_FACTOR,
            "m_kg": mass_kg,
            "c_N_s_m": damping_n_s_m,
            "source_note": "k, f0, and Q follow the external silicon notebook/script constants; m and c are derived.",
        },
        "middle_contact_transition_count": int(transition_idx_middle.size),
        "middle_contact_transition_source": str(CONTACT_JUDGE_NPZ),
        "middle_contact_transition_policy": "transition points are read from the x1 contact-judge result; they are not recomputed from x1 strong GP",
        "middle_contact_transition_times_us": [
            float(middle_time_s[int(idx)] * 1.0e6) for idx in transition_idx_middle
        ],
        "middle_f_residual_stats_nN": _stats(f_residual * 1.0e9),
        "middle_inertial_force_stats_nN": _stats(inertial_force * 1.0e9),
        "middle_damping_force_stats_nN": _stats(damping_force * 1.0e9),
        "middle_stiffness_force_stats_nN": _stats(stiffness_force * 1.0e9),
        "outputs": {
            "npz": str(output_npz),
            "json": str(output_json),
            "png": str(output_png),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    time_us = middle_time_s * 1.0e6
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12.0, 10.0),
        sharex=True,
        gridspec_kw={"hspace": 0.16},
    )
    panels = (
        (x1_tilde_middle * 1.0e9, r"$\tilde{x}_{1,\mathrm{strong}}$ [nm]"),
        (x1dot_tilde_middle * 1.0e3, r"$\dot{\tilde{x}}_{1,\mathrm{strong}}$ [mm s$^{-1}$]"),
        (x1ddot_tilde_middle, r"$\ddot{\tilde{x}}_{1,\mathrm{strong}}$ [m s$^{-2}$]"),
        (f_residual * 1.0e9, r"$\tilde{F}_{\mathrm{residual}}$ [nN]"),
    )
    for ax, (values, ylabel) in zip(axes, panels):
        ax.plot(time_us, values, color="black", linewidth=0.95)
        if transition_idx_middle.size:
            ax.scatter(
                time_us[transition_idx_middle],
                values[transition_idx_middle],
                s=16,
                color="black",
                edgecolors="white",
                linewidths=0.35,
                zorder=5,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)
    axes[-1].set_xlabel(r"Time [$\mu$s]")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_npz}")
    print(f"Saved: {output_json}")
    print(f"Saved: {output_png}")
    print(
        f"Middle: {summary['middle_time_range_us'][0]:.3f}-{summary['middle_time_range_us'][1]:.3f} us; "
        f"extended: {summary['extended_time_range_us'][0]:.3f}-{summary['extended_time_range_us'][1]:.3f} us"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
