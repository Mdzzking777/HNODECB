"""
Plot DMT-KV F_ts time-domain signal from trajectory.csv.

This script uses the same unified DMT-like force law as
generate_trajectory_DMT_KV.py.
"""

from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ============================================================================
# Fixed parameters (must match generate_trajectory_DMT_KV.py)
# ============================================================================

Estar = 15e6
eta_star = 1.3
R = 10e-9
A = 6.0e-20
a0 = 3.0e-10
beta = 5.0e11

WINDOW_OVERRIDES_US = {
    "first_contact": ("start", 1034.0),
    "max_x1_pp_change": ("stop", 1057.0),
    "tail_stable": ("stop", 1998.0),
}
MODIFIED_W0_START_US = 236.0


def load_trajectory(csv_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data loaded from {csv_path}")
    return data


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_f_ts_from_distance(
    s_m: np.ndarray,
    *,
    t_s: np.ndarray | None = None,
) -> np.ndarray:
    """
    Unified DMT-like force.

    For legacy CSVs without the exact saved F_ts column, use a fallback
    approximation for the Kelvin-Voigt term based on delta_dot computed from s(t).

        F_ts(s) =
            -A*R / (6 * (g(s) * (s - a0) + a0)^2)
            + (1 - g(s)) * (4/3) * Estar * sqrt(R) * max(a0 - s, 0)^(3/2)
            + eta_star * sqrt(R) * max(a0 - s, 0)^(1/2) * (a0 - s)_dot
    """
    g = sigmoid(beta * (s_m - a0))
    denom = g * (s_m - a0) + a0
    denom = np.maximum(denom, 1.0e-15)
    adhesion = -(A * R) / (6.0 * np.square(denom))
    indentation = np.maximum(a0 - s_m, 0.0)
    f_hertz = (4.0 / 3.0) * Estar * np.sqrt(R) * np.power(indentation, 1.5)
    if t_s is None:
        delta_dot = np.zeros_like(indentation)
    else:
        delta = indentation
        delta_dot = np.gradient(delta, t_s)
        delta_dot = np.where(delta > 0.0, delta_dot, 0.0)
    f_kv = eta_star * np.sqrt(R) * np.sqrt(indentation) * delta_dot
    f_ts = adhesion + (1.0 - g) * f_hertz + f_kv
    return f_ts


def contact_onsets(contact_mask: np.ndarray) -> np.ndarray:
    contact_mask = np.asarray(contact_mask, dtype=bool)
    if contact_mask.size == 0:
        return np.array([], dtype=int)
    starts = contact_mask & np.concatenate(([True], ~contact_mask[:-1]))
    return np.flatnonzero(starts)


def peak_to_peak(values: np.ndarray) -> float:
    return float(np.nanmax(values) - np.nanmin(values))


def relocate_window_with_anchor(
    t_post_us: np.ndarray, start_idx: int, stop_idx: int, anchor: str, target_us: float
):
    """
    Relocate a window in post-contact coordinates while preserving its length.

    Indices are 0-based and inclusive.
    Returns (new_start_idx, new_stop_idx, applied_override).
    If the target is outside the feasible range for the current trajectory,
    keep the original window unchanged.
    """
    n = len(t_post_us)
    length = stop_idx - start_idx + 1
    if length <= 0 or n < length:
        return start_idx, stop_idx, False

    if anchor == "start":
        lo = t_post_us[0]
        hi = t_post_us[n - length]
        if not (lo <= target_us <= hi):
            return start_idx, stop_idx, False
        new_start = int(np.searchsorted(t_post_us, target_us, side="left"))
        new_start = min(max(new_start, 0), n - length)
        new_stop = new_start + length - 1
    elif anchor == "stop":
        lo = t_post_us[length - 1]
        hi = t_post_us[-1]
        if not (lo <= target_us <= hi):
            return start_idx, stop_idx, False
        new_stop = int(np.argmin(np.abs(t_post_us - target_us)))
        new_stop = min(max(new_stop, length - 1), n - 1)
        new_start = new_stop - length + 1
    else:
        raise ValueError(f"Unsupported anchor: {anchor}")

    return new_start, new_stop, True


def extend_stop_to_contact_end(contact_mask: np.ndarray, stop_idx: int) -> int:
    """
    If the window currently ends inside a contact segment, extend the stop index
    until that segment fully ends.
    """
    n = len(contact_mask)
    stop_idx = min(max(int(stop_idx), 0), n - 1)
    if not bool(contact_mask[stop_idx]):
        return stop_idx

    new_stop = stop_idx
    while new_stop + 1 < n and bool(contact_mask[new_stop + 1]):
        new_stop += 1
    return new_stop


def build_training_style_windows(t_s: np.ndarray, contact_mask: np.ndarray, x1_signal: np.ndarray):
    first_contact_idx = np.flatnonzero(contact_mask)
    if first_contact_idx.size == 0:
        raise RuntimeError("No contact point found in trajectory.csv")
    first_contact_idx = int(first_contact_idx[0])

    t_post = t_s[first_contact_idx:]
    t_post_us = t_post * 1e6
    contact_post = contact_mask[first_contact_idx:]
    x1_post = x1_signal[first_contact_idx:]

    onset_post = contact_onsets(contact_post)
    if onset_post.size < 3:
        raise RuntimeError("Need at least 3 contact onsets after first contact to build W1/W2/W3")

    cycle_pp = []
    for k in range(len(onset_post) - 1):
        lo = onset_post[k]
        hi = onset_post[k + 1] - 1
        if hi < lo:
            raise RuntimeError("Invalid cycle bounds while building training-style windows")
        cycle_pp.append(peak_to_peak(x1_post[lo : hi + 1]))

    candidates = []
    for k in range(len(onset_post) - 2):
        start_idx = int(onset_post[k])
        stop_idx = int(onset_post[k + 2] - 1)
        candidates.append(
            {
                "candidate_index": k + 1,
                "start_idx": start_idx,
                "stop_idx": stop_idx,
                "len": stop_idx - start_idx + 1,
                "t_start_us": float(t_post_us[start_idx]),
                "t_stop_us": float(t_post_us[stop_idx]),
                "cycle1_index": k + 1,
                "cycle2_index": k + 2,
                "x1_pp_cycle1": float(cycle_pp[k]),
                "x1_pp_cycle2": float(cycle_pp[k + 1]),
                "x1_pp_delta": float(abs(cycle_pp[k + 1] - cycle_pp[k])),
            }
        )

    if not candidates:
        raise RuntimeError("No valid two-cycle windows were built from the post-contact trajectory")

    first_idx = 0
    change_order = list(np.argsort([cand["x1_pp_delta"] for cand in candidates])[::-1])
    tail_order = list(range(len(candidates) - 1, -1, -1))

    selected = []
    seen = set()

    def add_window(role: str, candidate_idx: int) -> bool:
        cand = candidates[candidate_idx].copy()
        key = (cand["start_idx"], cand["stop_idx"])
        if key in seen:
            return False
        seen.add(key)
        cand["role"] = role
        cand["label"] = f"{role}_window"
        selected.append(cand)
        return True

    def add_first_unique(role: str, order):
        for candidate_idx in order:
            if add_window(role, int(candidate_idx)):
                return

    add_window("first_contact", first_idx)
    add_first_unique("max_x1_pp_change", change_order)
    add_first_unique("tail_stable", tail_order)

    if len(selected) < 3:
        raise RuntimeError(f"Could not find 3 unique windows (found {len(selected)})")

    role_to_rank = {
        "first_contact": "W1",
        "max_x1_pp_change": "W2",
        "tail_stable": "W3",
    }

    adjusted = []
    for win in selected:
        anchor, target_us = WINDOW_OVERRIDES_US[win["role"]]
        new_start, new_stop, applied = relocate_window_with_anchor(
            t_post_us, win["start_idx"], win["stop_idx"], anchor, target_us
        )
        if win["role"] == "first_contact":
            new_stop = extend_stop_to_contact_end(contact_post, new_stop)
        win["start_idx"] = new_start
        win["stop_idx"] = new_stop
        win["len"] = new_stop - new_start + 1
        win["t_start_us"] = float(t_post_us[new_start])
        win["t_stop_us"] = float(t_post_us[new_stop])
        win["override_anchor"] = anchor
        win["override_target_us"] = float(target_us)
        win["override_applied"] = bool(applied)
        win["rank"] = role_to_rank[win["role"]]
        win["full_start_idx"] = first_contact_idx + new_start
        win["full_stop_idx"] = first_contact_idx + new_stop
        adjusted.append(win)

    w1_ref = next((win for win in adjusted if win["rank"] == "W1"), None)
    if w1_ref is None:
        raise RuntimeError("Could not find W1 after building training windows")

    w0_len = int(w1_ref["len"])
    w0_start = int(np.searchsorted(t_post_us, MODIFIED_W0_START_US, side="left"))
    w0_start = min(max(w0_start, 0), max(len(t_post_us) - w0_len, 0))
    w0_stop = min(w0_start + w0_len - 1, len(t_post_us) - 1)
    w0 = {
        "candidate_index": 0,
        "start_idx": w0_start,
        "stop_idx": w0_stop,
        "len": w0_stop - w0_start + 1,
        "t_start_us": float(t_post_us[w0_start]),
        "t_stop_us": float(t_post_us[w0_stop]),
        "cycle1_index": 0,
        "cycle2_index": 0,
        "x1_pp_cycle1": float("nan"),
        "x1_pp_cycle2": float("nan"),
        "x1_pp_delta": float("nan"),
        "role": "modified_w0",
        "label": "modified_w0_window",
        "override_anchor": "start",
        "override_target_us": float(MODIFIED_W0_START_US),
        "override_applied": True,
        "rank": "modified_w0",
        "full_start_idx": first_contact_idx + w0_start,
        "full_stop_idx": first_contact_idx + w0_stop,
    }

    return [w0, *adjusted]


def save_plot(x, y, contact_mask, out_png: Path, out_pdf: Path, title: str, *, figsize=(12, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color="tab:purple", linewidth=0.9, label="F_ts")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.7)

    ax.set_xlabel("Time [μs]")
    ax.set_ylabel("F_ts [nN]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


def save_training_window_plot(
    t_us: np.ndarray,
    f_ts_nN: np.ndarray,
    contact_mask: np.ndarray,
    windows,
    out_png: Path,
    out_pdf: Path,
):
    fig, axes = plt.subplots(len(windows), 1, figsize=(12, 2.8 * len(windows) + 1.1), sharey=True)
    axes = np.atleast_1d(axes)

    y_vals = []
    for win in windows:
        sl = slice(win["full_start_idx"], win["full_stop_idx"] + 1)
        y_vals.append(f_ts_nN[sl])
    y_all = np.concatenate(y_vals)
    y_pad = 0.08 * max(1e-6, float(y_all.max() - y_all.min()))
    y_lo = float(y_all.min() - y_pad)
    y_hi = float(y_all.max() + y_pad)

    role_titles = {
        "modified_w0": "modified_w0",
        "first_contact": "first_contact",
        "max_x1_pp_change": "max_x1_pp_change",
        "tail_stable": "tail_stable",
    }

    for ax, win in zip(axes, windows):
        sl = slice(win["full_start_idx"], win["full_stop_idx"] + 1)
        x = t_us[sl]
        y = f_ts_nN[sl]
        ax.plot(x, y, color="tab:purple", linewidth=1.0)
        valley_indices = build_contact_valley_markers(t_us, f_ts_nN, contact_mask, win)
        if valley_indices:
            global_idx = np.asarray(valley_indices, dtype=int)
            marker_x = np.asarray(t_us[global_idx], dtype=float).copy()
            marker_y = np.asarray(f_ts_nN[global_idx], dtype=float)
            dt_us = float(np.median(np.diff(x))) if len(x) > 1 else 0.0
            dup_groups: dict[int, list[int]] = {}
            for pos, idx in enumerate(global_idx.tolist()):
                dup_groups.setdefault(int(idx), []).append(pos)
            for positions in dup_groups.values():
                if len(positions) <= 1 or not np.isfinite(dt_us) or dt_us <= 0.0:
                    continue
                offsets = np.linspace(-0.18 * dt_us, 0.18 * dt_us, len(positions))
                for pos, dx in zip(positions, offsets):
                    marker_x[pos] += float(dx)
            ax.scatter(
                marker_x,
                marker_y,
                color="red",
                s=28,
                zorder=4,
            )
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.7)
        override_tag = "override" if win["override_applied"] else "auto"
        ax.set_title(
            f"{win['rank']} [{role_titles[win['role']]}]  "
            f"{win['t_start_us']:.3f}-{win['t_stop_us']:.3f} μs  "
            f"({override_tag})"
        )
        ax.set_ylabel("F_ts [nN]")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_lo, y_hi)

    axes[-1].set_xlabel("Time [μs]")
    fig.suptitle("AFM DMT-KV F_ts Time-Domain Signal (modified_w0 / W1 / W2 / W3 windows)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


def build_contact_valley_markers(
    t_us_full: np.ndarray,
    f_ts_nN_full: np.ndarray,
    contact_mask_full: np.ndarray,
    window,
) -> list[int]:
    y = np.asarray(f_ts_nN_full, dtype=float)
    contact = np.asarray(contact_mask_full, dtype=bool)
    n = len(y)
    if n < 4 or contact.size != n or not np.any(contact):
        return []

    win_start = int(window["full_start_idx"])
    win_stop = int(window["full_stop_idx"])
    starts = np.flatnonzero(contact & np.concatenate(([True], ~contact[:-1])))
    stops = np.flatnonzero(contact & np.concatenate((~contact[1:], [True])))
    if starts.size == 0 or stops.size == 0:
        return []

    overlapping_runs = [
        run_idx
        for run_idx, (run_start, run_stop) in enumerate(zip(starts, stops))
        if not (int(run_stop) < win_start or int(run_start) > win_stop)
    ][:2]
    if not overlapping_runs:
        return []

    markers: list[int] = []
    for run_idx in overlapping_runs:
        run_start = int(starts[run_idx])
        run_stop = int(stops[run_idx])
        peak_idx = int(run_start + np.nanargmax(y[run_start : run_stop + 1]))

        prev_stop = -1 if run_idx == 0 else int(stops[run_idx - 1])
        next_start = n if run_idx + 1 >= len(starts) else int(starts[run_idx + 1])

        left_lo = max(win_start, prev_stop + 1)
        left_hi = int(peak_idx)
        if left_hi >= left_lo:
            left_idx = int(left_lo + np.nanargmin(y[left_lo : left_hi + 1]))
            markers.append(left_idx)

        right_lo = int(peak_idx)
        right_hi = min(win_stop, int(next_start) - 1)
        if right_hi >= right_lo:
            right_idx = int(right_lo + np.nanargmin(y[right_lo : right_hi + 1]))
            markers.append(right_idx)

    markers = [int(idx) for idx in markers if 0 <= idx < n]
    return markers


def build_long_window_from_w1(windows, total_len: int):
    w1 = next((win for win in windows if win["rank"] == "W1"), None)
    if w1 is None:
        raise RuntimeError("Could not find W1 to build the long window")

    start_idx = int(w1["full_start_idx"])
    base_len = int(w1["len"])
    long_len = max(1, 3 * base_len)
    stop_idx = min(start_idx + long_len - 1, total_len - 1)

    long_window = {
        "rank": "LW",
        "role": "long_window_from_w1",
        "full_start_idx": start_idx,
        "full_stop_idx": stop_idx,
        "len": stop_idx - start_idx + 1,
        "base_len": base_len,
    }
    return long_window


def save_long_window_plot(
    t_us: np.ndarray,
    f_ts_nN: np.ndarray,
    contact_mask: np.ndarray,
    window,
    out_png: Path,
    out_pdf: Path,
):
    sl = slice(window["full_start_idx"], window["full_stop_idx"] + 1)
    x = t_us[sl]
    y = f_ts_nN[sl]
    c = contact_mask[sl]

    title = (
        "AFM DMT-KV F_ts Time-Domain Signal "
        f"(long window from W1 start, {window['len']} samples)"
    )
    save_plot(x, y, c, out_png, out_pdf, title, figsize=(16, 5.5))


def build_first_nonlinear_annotations(
    t_us: np.ndarray,
    f_ts_nN: np.ndarray,
    contact_mask: np.ndarray,
    window,
):
    sl = slice(window["full_start_idx"], window["full_stop_idx"] + 1)
    x = t_us[sl]
    y = f_ts_nN[sl]
    max_abs = float(np.nanmax(np.abs(y)))
    if len(x) < 8 or not np.isfinite(max_abs) or max_abs <= 0.0:
        return []

    eps = max(1e-4, 0.02 * max_abs)
    sign = np.where(y > eps, 1, np.where(y < -eps, -1, 0))
    nz_idx = np.flatnonzero(sign != 0)
    if nz_idx.size == 0:
        return []

    run_signs = []
    run_starts = []
    run_ends = []
    last_sign = None
    for idx in nz_idx:
        sgn = int(sign[idx])
        if sgn != last_sign:
            if last_sign is not None:
                run_ends.append(int(idx) - 1)
            run_signs.append(sgn)
            run_starts.append(int(idx))
            last_sign = sgn
    run_ends.append(int(nz_idx[-1]))

    first_neg_run = next((i for i, sgn in enumerate(run_signs) if sgn == -1), None)
    if first_neg_run is None:
        first_neg = int(np.nanargmin(y))
    else:
        lo = run_starts[first_neg_run]
        hi = run_ends[first_neg_run]
        first_neg = lo + int(np.nanargmin(y[lo : hi + 1]))

    first_pos = None
    second_neg = None
    second_neg_run = None
    if first_neg_run is not None:
        for run_idx in range(first_neg_run + 1, len(run_signs)):
            if first_pos is None and run_signs[run_idx] == 1:
                first_pos = run_starts[run_idx]
                continue
            if first_pos is not None and run_signs[run_idx] == -1:
                second_neg_run = run_idx
                break

    if first_pos is None:
        first_pos = int(np.nanargmax(y))
    if second_neg_run is not None:
        lo = run_starts[second_neg_run]
        hi = run_ends[second_neg_run]
        second_neg = lo + int(np.nanargmin(y[lo : hi + 1]))
    else:
        later = y[first_pos + 1 :] if first_pos + 1 < len(y) else np.array([])
        if later.size:
            second_neg = first_pos + 1 + int(np.nanargmin(later))
        else:
            second_neg = min(first_pos, len(y) - 1)

    annotations = [
        {
            "idx": first_neg,
            "label": "close-contact:\nadhesion occurs",
            "offset": (-95, -38),
        },
        {
            "idx": first_pos,
            "label": "contact:\nHertz dominant",
            "offset": (18, 28),
        },
        {
            "idx": second_neg,
            "label": "asymmetry:\nviscous dissipation",
            "offset": (30, -42),
        },
    ]

    return annotations


def save_long_window_gif(
    t_us: np.ndarray,
    f_ts_nN: np.ndarray,
    contact_mask: np.ndarray,
    window,
    out_gif: Path,
):
    sl = slice(window["full_start_idx"], window["full_stop_idx"] + 1)
    x = t_us[sl]
    y = f_ts_nN[sl]
    c = contact_mask[sl]

    n = len(x)
    if n < 2:
        raise RuntimeError("Long window is too short to animate")

    n_frames = min(120, max(45, n // 4))
    frame_indices = np.linspace(1, n, n_frames, dtype=int)
    annotations = build_first_nonlinear_annotations(t_us, f_ts_nN, contact_mask, window)

    y_pad = 0.08 * max(1e-6, float(np.nanmax(y) - np.nanmin(y)))
    y_lo = float(np.nanmin(y) - y_pad)
    y_hi = float(np.nanmax(y) + y_pad)

    frames = []
    for stop in frame_indices:
        fig, ax = plt.subplots(figsize=(16, 5.5))
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.7)
        ax.plot(x[:stop], y[:stop], color="tab:purple", linewidth=1.2, label="F_ts")
        ax.scatter([x[stop - 1]], [y[stop - 1]], color="tab:orange", s=28, zorder=3)
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Time [μs]")
        ax.set_ylabel("F_ts [nN]")
        ax.set_title("tip sample interaction response of polymer")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        ax.text(
            0.015,
            0.96,
            f"t = {x[stop - 1]:.3f} μs",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )
        for ann in annotations:
            idx = int(ann["idx"])
            if stop <= idx:
                continue
            ax.annotate(
                ann["label"],
                xy=(x[idx], y[idx]),
                xytext=ann["offset"],
                textcoords="offset points",
                fontsize=9,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25"),
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.88, edgecolor="0.8"),
            )
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
        disposal=2,
    )


def main():
    root = Path(__file__).resolve().parent
    csv_path = root / "trajectory.csv"
    plots_dir = root / "plots"
    plots_dir.mkdir(exist_ok=True)

    data = load_trajectory(csv_path)

    t_s = np.asarray(data["time_s"], dtype=float)
    x_tip_m = np.asarray(data["x_tip_m"], dtype=float)
    s_m = np.asarray(data["s_tip_sample_distance_m"], dtype=float)

    if "fts_interaction_force" in data.dtype.names:
        f_ts_n = np.asarray(data["fts_interaction_force"], dtype=float)
    else:
        f_ts_n = compute_f_ts_from_distance(s_m, t_s=t_s)
    contact = s_m <= a0

    t_us = t_s * 1e6
    f_ts_nN = f_ts_n * 1e9

    save_plot(
        t_us,
        f_ts_nN,
        contact,
        plots_dir / "F_ts_time_domain_full.png",
        plots_dir / "F_ts_time_domain_full.pdf",
        "AFM DMT-KV F_ts Time-Domain Signal (sigmoid-gated unified DMT-like force in s)",
    )

    training_windows = build_training_style_windows(t_s, contact, x_tip_m)
    save_training_window_plot(
        t_us,
        f_ts_nN,
        contact,
        training_windows,
        plots_dir / "F_ts_time_domain_zoomed.png",
        plots_dir / "F_ts_time_domain_zoomed.pdf",
    )
    long_window = build_long_window_from_w1(training_windows, len(t_us))
    save_long_window_plot(
        t_us,
        f_ts_nN,
        contact,
        long_window,
        plots_dir / "F_ts_time_domain_long_window.png",
        plots_dir / "F_ts_time_domain_long_window.pdf",
    )
    save_long_window_gif(
        t_us,
        f_ts_nN,
        contact,
        long_window,
        plots_dir / "F_ts_time_domain_long_window.gif",
    )

    print("Saved:")
    print(f"  {plots_dir / 'F_ts_time_domain_full.png'}")
    print(f"  {plots_dir / 'F_ts_time_domain_full.pdf'}")
    print(f"  {plots_dir / 'F_ts_time_domain_zoomed.png'}")
    print(f"  {plots_dir / 'F_ts_time_domain_zoomed.pdf'}")
    print(f"  {plots_dir / 'F_ts_time_domain_long_window.png'}")
    print(f"  {plots_dir / 'F_ts_time_domain_long_window.pdf'}")
    print(f"  {plots_dir / 'F_ts_time_domain_long_window.gif'}")
    for win in training_windows:
        print(
            f"{win['rank']} [{win['role']}] -> "
            f"{win['t_start_us']:.3f}-{win['t_stop_us']:.3f} us "
            f"(override={'yes' if win['override_applied'] else 'no'})"
        )
    print(
        "LW [long_window_from_w1] -> "
        f"{t_us[long_window['full_start_idx']]:.3f}-{t_us[long_window['full_stop_idx']]:.3f} us"
    )
    print(f"F_ts min/max [nN]: {f_ts_nN.min():.6f}, {f_ts_nN.max():.6f}")
    print(f"Contact fraction [%]: {100.0 * np.mean(contact):.2f}")


if __name__ == "__main__":
    main()
