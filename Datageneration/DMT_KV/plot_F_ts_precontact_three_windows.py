"""Plot three consecutive two-cycle Fts windows before first contact."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


F0_HZ = 313.57e3
TWO_CYCLE_SPAN_S = 2.0 / F0_HZ


def main() -> int:
    root = Path(__file__).resolve().parent
    data = np.genfromtxt(root / "trajectory.csv", delimiter=",", names=True)
    t_s = np.asarray(data["time_s"], dtype=float)
    fts_nN = np.asarray(data["fts_interaction_force"], dtype=float) * 1.0e9
    contact = np.asarray(data["contact_status"], dtype=float) > 0.5

    contact_indices = np.flatnonzero(contact)
    if contact_indices.size == 0:
        raise RuntimeError("No contact point exists in trajectory.csv")
    first_contact_index = int(contact_indices[0])
    first_contact_time_s = float(t_s[first_contact_index])

    windows: list[tuple[int, int]] = []
    for position in range(3):
        stop_s = first_contact_time_s - (2 - position) * TWO_CYCLE_SPAN_S
        start_s = stop_s - TWO_CYCLE_SPAN_S
        start = int(np.searchsorted(t_s, start_s, side="left"))
        stop = int(np.searchsorted(t_s, stop_s, side="left")) - 1
        if start < 0 or stop < start or stop >= first_contact_index:
            raise RuntimeError("Unable to construct a complete pre-contact window")
        if np.any(contact[start : stop + 1]):
            raise RuntimeError("A pre-contact window unexpectedly contains contact samples")
        windows.append((start, stop))

    selected_force = np.concatenate([fts_nN[start : stop + 1] for start, stop in windows])
    force_span = float(np.ptp(selected_force))
    padding = 0.08 * max(force_span, 1.0e-6)
    y_limits = (float(np.min(selected_force) - padding), float(np.max(selected_force) + padding))

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharey=True)
    for number, (ax, (start, stop)) in enumerate(zip(axes, windows, strict=True), start=1):
        time_us = t_s[start : stop + 1] * 1.0e6
        force = fts_nN[start : stop + 1]
        ax.plot(time_us, force, color="tab:purple", linewidth=1.1)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(
            f"Pre-contact window {number}: "
            f"{time_us[0]:.3f}-{time_us[-1]:.3f} us"
        )
        ax.set_ylabel(r"$F_{ts}$ [nN]")
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(r"Time [$\mu$s]")
    fig.suptitle(
        "AFM DMT-KV $F_{ts}$ before first contact "
        rf"($t_{{contact}}={first_contact_time_s * 1.0e6:.3f}$ $\mu$s)"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output = root / "plots" / "F_ts_time_domain_precontact_three_windows.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)

    print(f"First contact: {first_contact_time_s * 1.0e6:.6f} us")
    for number, (start, stop) in enumerate(windows, start=1):
        print(
            f"Window {number}: {t_s[start] * 1.0e6:.6f}-"
            f"{t_s[stop] * 1.0e6:.6f} us, "
            f"Fts=[{np.min(fts_nN[start:stop+1]):.6f}, "
            f"{np.max(fts_nN[start:stop+1]):.6f}] nN"
        )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
