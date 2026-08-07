"""Plot raw index109 x1/x2 and numerical derivatives in a fixed 3x2 layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MAT = SCRIPT_DIR / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
RAW_MAT = (
    Path(r"E:\External data SI_SI hard sample raw")
    / "OneDrive_2_2026-7-26"
    / "dataAFM_Si"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)

INDEX = 109
DT_S = 1.0 / 250.0e6
OUTPUT = SCRIPT_DIR / "silicon_raw_forward_index109_x1_x2_numerical_derivatives_grid.png"
F_RESIDUAL_NPZ = (
    SCRIPT_DIR
    / "F_residual"
    / "silicon_middle_f_residual_x1strong_x2constraints.npz"
)


def load_index109() -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    errors: list[str] = []
    for mat_path in (LOCAL_MAT, RAW_MAT):
        if not mat_path.is_file():
            errors.append(f"{mat_path} does not exist")
            continue
        try:
            data = sio.loadmat(
                str(mat_path),
                variable_names=["displacement_fwd_sweep", "velocity_fwd_sweep"],
                squeeze_me=True,
                struct_as_record=False,
            )
            x1 = np.asarray(data["displacement_fwd_sweep"][INDEX], dtype=float).ravel()
            x2 = np.asarray(data["velocity_fwd_sweep"][INDEX], dtype=float).ravel()
            if x1.shape != x2.shape or x1.size < 3:
                raise ValueError(f"invalid shapes: x1={x1.shape}, x2={x2.shape}")
            time_s = np.arange(x1.size, dtype=float) * DT_S
            return time_s, x1, x2, mat_path
        except Exception as exc:
            errors.append(f"{mat_path}: {exc}")
    raise RuntimeError("Could not read silicon MAT file:\n" + "\n".join(errors))


def main() -> int:
    time_s, x1_raw, x2_raw, mat_path = load_index109()
    x1dot_raw = np.gradient(x1_raw, time_s, edge_order=2)
    x1ddot_raw = np.gradient(x1dot_raw, time_s, edge_order=2)
    x2dot_raw = np.gradient(x2_raw, time_s, edge_order=2)
    time_us = time_s * 1.0e6

    with np.load(F_RESIDUAL_NPZ, allow_pickle=False) as residual_data:
        window_time_s = np.asarray(residual_data["time_s"], dtype=float)
        display_mask = np.asarray(residual_data["display_mask"], dtype=bool)
    if window_time_s.ndim != 1 or display_mask.shape != window_time_s.shape:
        raise RuntimeError(f"invalid residual window arrays in {F_RESIDUAL_NPZ}")
    window_start_s = float(window_time_s[display_mask][0])
    window_stop_s = float(window_time_s[display_mask][-1])
    window_mask = (time_s >= window_start_s) & (time_s <= window_stop_s)
    if not np.any(window_mask):
        raise RuntimeError("residual display window does not overlap raw index109 time")

    fig = plt.figure(figsize=(13.2, 11.0))
    gs = fig.add_gridspec(
        4,
        2,
        width_ratios=(1.0, 1.0),
        height_ratios=(1.0, 1.0, 1.0, 1.0),
        hspace=0.22,
        wspace=0.18,
    )
    axes = {
        "x1": fig.add_subplot(gs[0, 0]),
        "x1dot": fig.add_subplot(gs[1, 0]),
        "x1ddot": fig.add_subplot(gs[2, 0]),
        "x2": fig.add_subplot(gs[1, 1]),
        "x2dot": fig.add_subplot(gs[2, 1]),
        "x1ddot_window": fig.add_subplot(gs[3, 0]),
        "x2dot_window": fig.add_subplot(gs[3, 1]),
    }
    blank = fig.add_subplot(gs[0, 1])
    blank.axis("off")

    line = {"color": "black", "linewidth": 0.65}
    axes["x1"].plot(time_us, x1_raw * 1.0e9, **line)
    axes["x1dot"].plot(time_us, x1dot_raw, **line)
    axes["x1ddot"].plot(time_us, x1ddot_raw, **line)
    axes["x2"].plot(time_us, x2_raw, **line)
    axes["x2dot"].plot(time_us, x2dot_raw, **line)
    axes["x1ddot_window"].plot(time_us[window_mask], x1ddot_raw[window_mask], **line)
    axes["x2dot_window"].plot(time_us[window_mask], x2dot_raw[window_mask], **line)

    axes["x1"].set_ylabel(r"$x_1^{\mathrm{raw}}$ [nm]")
    axes["x1dot"].set_ylabel(r"$d x_1^{\mathrm{raw}}/dt$ [m s$^{-1}$]")
    axes["x1ddot"].set_ylabel(r"$d^2 x_1^{\mathrm{raw}}/dt^2$ [m s$^{-2}$]")
    axes["x2"].set_ylabel(r"$x_2^{\mathrm{raw}}$ [m s$^{-1}$]")
    axes["x2dot"].set_ylabel(r"$d x_2^{\mathrm{raw}}/dt$ [m s$^{-2}$]")
    axes["x1ddot_window"].set_ylabel(r"$d^2 x_1^{\mathrm{raw}}/dt^2$ [m s$^{-2}$]")
    axes["x2dot_window"].set_ylabel(r"$d x_2^{\mathrm{raw}}/dt$ [m s$^{-2}$]")

    for key, ax in axes.items():
        ax.grid(True, alpha=0.25)
        if key.endswith("_window"):
            ax.set_xlim(float(window_start_s * 1.0e6), float(window_stop_s * 1.0e6))
            ax.set_xlabel(r"time [$\mu$s]")
        else:
            ax.set_xlim(float(time_us[0]), float(time_us[-1]))
        if key in {"x1ddot", "x2dot"}:
            ax.set_xlabel(r"time [$\mu$s]")
        elif not key.endswith("_window"):
            ax.tick_params(labelbottom=False)

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Read MAT: {mat_path}")
    print(f"Window: {window_start_s * 1.0e6:.3f}-{window_stop_s * 1.0e6:.3f} us")
    print(f"Saved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
