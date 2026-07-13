"""Plot x3dot over W0, W1, and W3 windows from trajectory.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_F_ts_time_domain import build_training_style_windows, load_trajectory


WINDOW_LABELS = {
    "modified_w0": "W0",
    "W1": "W1",
    "W3": "W3",
}


def selected_windows(windows: list[dict]) -> list[dict]:
    selected: list[dict] = []
    for target in ("modified_w0", "W1", "W3"):
        match = next((win for win in windows if str(win.get("rank")) == target), None)
        if match is None:
            raise RuntimeError(f"Could not find required window: {target}")
        selected.append(match)
    return selected


def save_combined_plot(
    *,
    t_us: np.ndarray,
    x3dot_um_s: np.ndarray,
    contact: np.ndarray,
    windows: list[dict],
    out_png: Path,
    out_pdf: Path,
    quantity_label: str = "x3dot",
    title: str = "AFM DMT-KV x3dot in W0 / W1 / W3",
) -> None:
    fig, axes = plt.subplots(len(windows), 1, figsize=(12, 8.2), sharey=True)
    for ax, win in zip(axes, windows):
        sl = slice(int(win["full_start_idx"]), int(win["full_stop_idx"]) + 1)
        label = WINDOW_LABELS[str(win["rank"])]
        ax.plot(t_us[sl], x3dot_um_s[sl], color="tab:blue", linewidth=1.25, label=quantity_label)
        ax.fill_between(
            t_us[sl],
            np.nanmin(x3dot_um_s[sl]),
            np.nanmax(x3dot_um_s[sl]),
            where=contact[sl],
            color="tab:red",
            alpha=0.10,
            step="mid",
            label="contact",
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(
            f"{label}: {float(t_us[sl][0]):.3f}-{float(t_us[sl][-1]):.3f} us"
        )
        ax.set_ylabel(f"{quantity_label} [um/s]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [us]")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    plots_dir = root / "plots"
    data = load_trajectory(root / "trajectory.csv")

    t_s = np.asarray(data["time_s"], dtype=float)
    t_us = t_s * 1.0e6
    x1 = np.asarray(data["x_tip_m"], dtype=float)
    contact = np.asarray(data["contact_status"], dtype=float) > 0.5
    x3dot_um_s = np.asarray(data["x3dot_sample_velocity"], dtype=float) * 1.0e6

    windows = selected_windows(build_training_style_windows(t_s, contact, x1))
    save_combined_plot(
        t_us=t_us,
        x3dot_um_s=x3dot_um_s,
        contact=contact,
        windows=windows,
        out_png=plots_dir / "x3dot_time_domain_w0_w1_w3.png",
        out_pdf=plots_dir / "x3dot_time_domain_w0_w1_w3.pdf",
    )
    save_combined_plot(
        t_us=t_us,
        x3dot_um_s=-x3dot_um_s,
        contact=contact,
        windows=windows,
        out_png=plots_dir / "minus_x3dot_time_domain_w0_w1_w3.png",
        out_pdf=plots_dir / "minus_x3dot_time_domain_w0_w1_w3.pdf",
        quantity_label="-x3dot",
        title="AFM DMT-KV -x3dot in W0 / W1 / W3",
    )

    print("Saved:")
    print(f"  {plots_dir / 'x3dot_time_domain_w0_w1_w3.png'}")
    print(f"  {plots_dir / 'x3dot_time_domain_w0_w1_w3.pdf'}")
    print(f"  {plots_dir / 'minus_x3dot_time_domain_w0_w1_w3.png'}")
    print(f"  {plots_dir / 'minus_x3dot_time_domain_w0_w1_w3.pdf'}")
    for win in windows:
        label = WINDOW_LABELS[str(win["rank"])]
        sl = slice(int(win["full_start_idx"]), int(win["full_stop_idx"]) + 1)
        print(
            f"  {label}: {float(t_us[sl][0]):.3f}-{float(t_us[sl][-1]):.3f} us | "
            f"x3dot_um_s=[{float(np.nanmin(x3dot_um_s[sl])):.6e}, {float(np.nanmax(x3dot_um_s[sl])):.6e}]"
        )


if __name__ == "__main__":
    main()
