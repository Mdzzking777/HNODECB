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
SILICON_DIR = SCRIPT_DIR.parents[1]
CONTACT_NPZ = (
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
OUTPUT_STEM = "silicon_middle_x2_raw_global_gp_smooth"


def _load_contact_time() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(CONTACT_NPZ, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        source_idx = np.asarray(data["source_idx"], dtype=int)
        transition_idx = np.asarray(data["transition_idx"], dtype=int)
    return time_s, source_idx, transition_idx


def _load_raw_x2(source_idx: np.ndarray) -> tuple[np.ndarray, Path]:
    errors: list[str] = []
    for mat_path in (LOCAL_MAT, RAW_MAT):
        if not mat_path.is_file():
            errors.append(f"{mat_path} does not exist")
            continue
        try:
            data = sio.loadmat(
                str(mat_path),
                variable_names=["velocity_fwd_sweep"],
                squeeze_me=True,
                struct_as_record=False,
            )
            x2_all = np.asarray(data["velocity_fwd_sweep"][INDEX], dtype=float).ravel()
            return x2_all[source_idx], mat_path
        except Exception as exc:
            errors.append(f"{mat_path}: {exc}")
    raise RuntimeError("Could not read raw x2 from MAT file:\n" + "\n".join(errors))


def _fit_global_gp(time_s: np.ndarray, x2_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    time_us = time_s * 1.0e6
    t_mean = float(np.mean(time_us))
    t_scale = float(np.std(time_us))
    y_mean = float(np.mean(x2_raw))
    y_scale = float(np.std(x2_raw))
    if t_scale <= 0.0 or y_scale <= 0.0:
        raise RuntimeError("invalid time/x2 scale")

    x = ((time_us - t_mean) / t_scale).reshape(-1, 1)
    y = (x2_raw - y_mean) / y_scale

    # Deliberately strong smoothing: use a fixed long RBF length-scale and a
    # fixed white-noise term so the optimizer cannot shrink the kernel to track
    # local raw-data wiggles.
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
    gp.fit(x, y)
    y_tilde, y_std = gp.predict(x, return_std=True)
    x2_tilde = y_tilde * y_scale + y_mean
    x2_std = y_std * y_scale

    metadata = {
        "kernel_initial": str(kernel),
        "kernel_optimized": str(gp.kernel_),
        "time_us_mean": t_mean,
        "time_us_scale": t_scale,
        "x2_mean_m_s": y_mean,
        "x2_scale_m_s": y_scale,
        "smoothing_policy": "single global GP over entire middle x2_raw; no regime distinctions",
    }
    return x2_tilde, x2_std, metadata


def _plot(
    *,
    time_s: np.ndarray,
    x2_raw: np.ndarray,
    x2_tilde: np.ndarray,
    x2dot_tilde: np.ndarray,
    transition_idx: np.ndarray,
    png_path: Path,
) -> None:
    time_us = time_s * 1.0e6
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.0, 8.4),
        sharex=False,
        gridspec_kw={"hspace": 0.24},
    )
    axes[0].plot(time_us, x2_raw, color="0.35", linewidth=0.75, label=r"$x_2^{raw}$")
    axes[0].set_ylabel(r"$x_2^{raw}$ [m s$^{-1}$]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_us, x2_tilde, color="#1f77b4", linewidth=1.15, label=r"$\tilde{x}_2$ (GP)")
    axes[1].set_ylabel(r"$\tilde{x}_2$ [m s$^{-1}$]")
    axes[1].legend(loc="upper right")

    axes[2].plot(time_us, x2dot_tilde, color="#d62728", linewidth=0.95, label=r"$\dot{\tilde{x}}_2$ (GP)")
    axes[2].set_ylabel(r"$\dot{\tilde{x}}_2$ [m s$^{-2}$]")
    axes[2].set_xlabel(r"Time [$\mu$s]")
    axes[2].legend(loc="upper right")

    for idx in transition_idx:
        for ax in axes:
            ax.axvline(time_us[int(idx)], color="black", linewidth=0.45, alpha=0.18)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    time_s, source_idx, transition_idx = _load_contact_time()
    x2_raw, mat_path = _load_raw_x2(source_idx)
    x2_tilde, x2_std, gp_metadata = _fit_global_gp(time_s, x2_raw)
    x2dot_tilde = np.gradient(x2_tilde, time_s, edge_order=2)

    npz_path = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    json_path = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
    txt_path = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.txt"
    png_path = SCRIPT_DIR / f"{OUTPUT_STEM}.png"

    np.savez_compressed(
        npz_path,
        time_s=time_s,
        source_idx=source_idx,
        x2_raw_m_s=x2_raw,
        x2_tilde_m_s=x2_tilde,
        x2_tilde_std_m_s=x2_std,
        x2dot_tilde_m_s2=x2dot_tilde,
        transition_idx=transition_idx,
    )

    residual = x2_raw - x2_tilde
    summary = {
        "method": "global Gaussian-process smoothing of x2_raw",
        "raw_x2_source_mat": str(mat_path),
        "raw_x2_source_variable": f"velocity_fwd_sweep[{INDEX}]",
        "contact_source_npz": str(CONTACT_NPZ),
        "sample_count": int(time_s.size),
        "time_range_us": [float(time_s[0] * 1.0e6), float(time_s[-1] * 1.0e6)],
        "transition_count": int(transition_idx.size),
        "fit": {
            "rmse_x2_raw_minus_x2_tilde_m_s": float(np.sqrt(np.mean(residual**2))),
            "std_x2_raw_minus_x2_tilde_m_s": float(np.std(residual)),
        },
        "gp": gp_metadata,
        "outputs": {
            "npz": str(npz_path),
            "json": str(json_path),
            "txt": str(txt_path),
            "png": str(png_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "Global GP smoothing for middle-slice x2 raw",
        "==========================================",
        "",
        "A single GP is applied to all x2_raw points without contact/noncontact distinctions.",
        f"Time range: {summary['time_range_us'][0]:.3f}-{summary['time_range_us'][1]:.3f} us",
        f"Samples: {summary['sample_count']}",
        f"Optimized kernel: {gp_metadata['kernel_optimized']}",
        f"RMSE(x2_raw - x2_tilde): {summary['fit']['rmse_x2_raw_minus_x2_tilde_m_s']:.6e} m s^-1",
        "",
        f"Saved NPZ: {npz_path}",
        f"Saved PNG: {png_path}",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    _plot(
        time_s=time_s,
        x2_raw=x2_raw,
        x2_tilde=x2_tilde,
        x2dot_tilde=x2dot_tilde,
        transition_idx=transition_idx,
        png_path=png_path,
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
