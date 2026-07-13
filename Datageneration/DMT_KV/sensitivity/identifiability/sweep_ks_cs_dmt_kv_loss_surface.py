"""Sweep ks/cs for the fully known AFM DMT-KV state-space model.

The script assumes the DMT-KV force law and every AFM parameter are known
except the Kelvin-Voigt surface parameters ks and cs.  For each (ks, cs) pair it
rolls out the state equation from the true window initial condition, compares
the result with trajectory.csv, writes a CSV loss table, and plots a 2-D loss
surface over ks/cs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


DMT_KV_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = DMT_KV_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM04.stage1pluslight.grid import CS_BOUNDS, CS_TRUE, KS_BOUNDS, KS_TRUE  # noqa: E402
from AFM04.stage1pluslight.windows import window_manifests  # noqa: E402
from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import (  # noqa: E402
    A,
    A0,
    BETA,
    DIST,
    ESTAR,
    ETA_STAR,
    FD,
    R,
    c,
    k,
    m,
    wd,
)


DEFAULT_WINDOW_MODE = "stage2_w0"
DEFAULT_WINDOW_US = 6.288e-6

RAINBOW_PURPLE_LOW_RED_HIGH = LinearSegmentedColormap.from_list(
    "rainbow_purple_low_red_high",
    ["#4b0082", "#0000ff", "#00ffff", "#00aa00", "#ffff00", "#ff7f00", "#ff0000"],
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def load_trajectory(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing trajectory CSV: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data loaded from {path}")
    return data


def select_window(data: np.ndarray, window_mode: str, window_us: float, window_index: int) -> tuple[np.ndarray, object]:
    times = np.asarray(data["time_s"], dtype=float)
    normalized_mode = str(window_mode).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_mode in {"full", "full_time", "full_time_span", "full_timespan"}:
        idxs = np.arange(times.size, dtype=int)
        win = SimpleNamespace(
            idxs=idxs,
            label="full_time_span",
            role="full_time_span",
            start_idx=0,
            stop_idx=int(times.size - 1),
            t_start=float(times[0]),
            t_stop=float(times[-1]),
        )
        return idxs, win

    contact = np.asarray(data["contact_status"], dtype=float) > 0.5
    x1 = np.asarray(data["x_tip_m"], dtype=float)
    manifests = window_manifests(times, contact, window_mode, window_us, x1_signal=x1)
    if not manifests:
        raise RuntimeError("No window manifest was produced")
    idx = int(window_index)
    if idx < 0 or idx >= len(manifests):
        raise IndexError(f"window_index={idx} out of range for {len(manifests)} manifest(s)")
    win = manifests[idx]
    return np.asarray(win.idxs, dtype=int), win


def f_ts_ydot_vec(x: np.ndarray, v: np.ndarray, y: np.ndarray, ks: np.ndarray, cs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized unified DMT-KV force closure."""

    s = DIST + x - y
    g = sigmoid(BETA * (s - A0))
    denom = g * (s - A0) + A0
    denom = np.maximum(denom, 1.0e-15)
    adhesion = -(A * R) / (6.0 * np.square(denom))
    delta = np.maximum(A0 - s, 0.0)
    f_hertz = (4.0 / 3.0) * ESTAR * np.sqrt(R) * np.power(delta, 1.5)
    kv_coeff = ETA_STAR * np.sqrt(R) * np.sqrt(delta)

    ydot = (-adhesion - (1.0 - g) * f_hertz + kv_coeff * v - ks * y) / (cs + kv_coeff)
    delta_dot = np.where(delta > 0.0, ydot - v, 0.0)
    f_ts = adhesion + (1.0 - g) * f_hertz + kv_coeff * delta_dot
    return f_ts, ydot, delta_dot


def rhs_vec(t: float, x: np.ndarray, v: np.ndarray, y: np.ndarray, ks: np.ndarray, cs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_ts, ydot, _ = f_ts_ydot_vec(x, v, y, ks, cs)
    dvdt = (FD * np.cos(wd * t) - k * x - c * v + f_ts) / m
    return v, dvdt, ydot


def rk4_step_vec(
    t: float,
    dt: float,
    x: np.ndarray,
    v: np.ndarray,
    y: np.ndarray,
    ks: np.ndarray,
    cs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k1x, k1v, k1y = rhs_vec(t, x, v, y, ks, cs)
    k2x, k2v, k2y = rhs_vec(t + 0.5 * dt, x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, y + 0.5 * dt * k1y, ks, cs)
    k3x, k3v, k3y = rhs_vec(t + 0.5 * dt, x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, y + 0.5 * dt * k2y, ks, cs)
    k4x, k4v, k4y = rhs_vec(t + dt, x + dt * k3x, v + dt * k3v, y + dt * k3y, ks, cs)
    x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    y_next = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    return x_next, v_next, y_next


def signal_scale(values: np.ndarray, eps: float = 1.0e-30) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return max(float(np.ptp(finite)), float(np.max(np.abs(finite))), eps)


def simulate_chunk_losses(
    *,
    data: np.ndarray,
    idxs: np.ndarray,
    ks_flat: np.ndarray,
    cs_flat: np.ndarray,
    scales: dict[str, float],
) -> dict[str, np.ndarray]:
    times = np.asarray(data["time_s"], dtype=float)
    x_true = np.asarray(data["x_tip_m"], dtype=float)
    v_true = np.asarray(data["x1dot_velocity"], dtype=float)
    y_true = np.asarray(data["y_sample_m"], dtype=float)
    x2dot_true = np.asarray(data["x2dot_acceleration"], dtype=float)
    fts_true = np.asarray(data["fts_interaction_force"], dtype=float)

    n = ks_flat.size
    start = int(idxs[0])
    x = np.full(n, x_true[start], dtype=float)
    v = np.full(n, v_true[start], dtype=float)
    y = np.full(n, y_true[start], dtype=float)
    alive = np.ones(n, dtype=bool)

    sums = {name: np.zeros(n, dtype=float) for name in ("x1", "x2", "x3", "x2dot", "fts")}
    count = 0

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for pos, global_idx in enumerate(idxs):
            t = float(times[int(global_idx)])
            fts, _ydot, _delta_dot = f_ts_ydot_vec(x, v, y, ks_flat, cs_flat)
            x2dot = (FD * np.cos(wd * t) - k * x - c * v + fts) / m

            preds = {
                "x1": x,
                "x2": v,
                "x3": y,
                "x2dot": x2dot,
                "fts": fts,
            }
            truths = {
                "x1": float(x_true[int(global_idx)]),
                "x2": float(v_true[int(global_idx)]),
                "x3": float(y_true[int(global_idx)]),
                "x2dot": float(x2dot_true[int(global_idx)]),
                "fts": float(fts_true[int(global_idx)]),
            }
            for name, pred in preds.items():
                err = (pred - truths[name]) / scales[name]
                finite_err = np.isfinite(err) & alive
                sums[name] += np.where(finite_err, np.square(err), 0.0)
                alive &= np.isfinite(pred)

            count += 1
            if pos + 1 >= idxs.size:
                break

            next_idx = int(idxs[pos + 1])
            dt = float(times[next_idx] - times[int(global_idx)])
            x, v, y = rk4_step_vec(t, dt, x, v, y, ks_flat, cs_flat)
            finite_state = np.isfinite(x) & np.isfinite(v) & np.isfinite(y)
            alive &= finite_state
            x = np.where(alive, x, 0.0)
            v = np.where(alive, v, 0.0)
            y = np.where(alive, y, 0.0)

    losses = {name: sums[name] / max(count, 1) for name in sums}
    losses["observed"] = (losses["x1"] + losses["x2"] + losses["x2dot"]) / 3.0
    losses["state"] = (losses["x1"] + losses["x2"] + losses["x3"]) / 3.0
    losses["full"] = (losses["x1"] + losses["x2"] + losses["x3"] + losses["x2dot"] + losses["fts"]) / 5.0
    for value in losses.values():
        value[~alive] = np.inf
    return losses


def sweep_losses(
    *,
    data: np.ndarray,
    idxs: np.ndarray,
    ks_values: np.ndarray,
    cs_values: np.ndarray,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    kk, cc = np.meshgrid(ks_values, cs_values, indexing="ij")
    ks_flat = kk.ravel()
    cs_flat = cc.ravel()

    scales = {
        "x1": signal_scale(np.asarray(data["x_tip_m"], dtype=float)[idxs]),
        "x2": signal_scale(np.asarray(data["x1dot_velocity"], dtype=float)[idxs]),
        "x3": signal_scale(np.asarray(data["y_sample_m"], dtype=float)[idxs]),
        "x2dot": signal_scale(np.asarray(data["x2dot_acceleration"], dtype=float)[idxs]),
        "fts": signal_scale(np.asarray(data["fts_interaction_force"], dtype=float)[idxs]),
    }

    out = {name: np.empty_like(ks_flat, dtype=float) for name in ("x1", "x2", "x3", "x2dot", "fts", "observed", "state", "full")}
    total = ks_flat.size
    step = max(1, int(chunk_size))
    for start in range(0, total, step):
        stop = min(start + step, total)
        losses = simulate_chunk_losses(
            data=data,
            idxs=idxs,
            ks_flat=ks_flat[start:stop],
            cs_flat=cs_flat[start:stop],
            scales=scales,
        )
        for name, values in losses.items():
            out[name][start:stop] = values
        print(f"chunk {stop}/{total} ({100.0 * stop / total:.2f}%)", flush=True)

    return {name: values.reshape(len(ks_values), len(cs_values)) for name, values in out.items()}


def finite_min(values: np.ndarray) -> tuple[int, int, float]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0, 0, float("inf")
    masked = np.where(finite, values, np.inf)
    flat_idx = int(np.argmin(masked))
    i, j = np.unravel_index(flat_idx, values.shape)
    return int(i), int(j), float(values[i, j])


def log_values_with_required(lo: float, hi: float, count: int, required: float) -> np.ndarray:
    values = np.exp(np.linspace(np.log(float(lo)), np.log(float(hi)), int(count)))
    req = float(required)
    if float(lo) <= req <= float(hi) and not np.any(np.isclose(values, req, rtol=1.0e-13, atol=0.0)):
        values = np.sort(np.concatenate([values, np.asarray([req], dtype=float)]))
    return values


def write_csv(path: Path, ks_values: np.ndarray, cs_values: np.ndarray, losses: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ks", "cs", "ks_err_pct", "cs_err_pct", "loss_observed", "loss_state", "loss_full", "loss_x1", "loss_x2", "loss_x3", "loss_x2dot", "loss_fts"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, ks_val in enumerate(ks_values):
            for j, cs_val in enumerate(cs_values):
                writer.writerow(
                    {
                        "ks": float(ks_val),
                        "cs": float(cs_val),
                        "ks_err_pct": 100.0 * (float(ks_val) - float(KS_TRUE)) / float(KS_TRUE),
                        "cs_err_pct": 100.0 * (float(cs_val) - float(CS_TRUE)) / float(CS_TRUE),
                        "loss_observed": float(losses["observed"][i, j]),
                        "loss_state": float(losses["state"][i, j]),
                        "loss_full": float(losses["full"][i, j]),
                        "loss_x1": float(losses["x1"][i, j]),
                        "loss_x2": float(losses["x2"][i, j]),
                        "loss_x3": float(losses["x3"][i, j]),
                        "loss_x2dot": float(losses["x2dot"][i, j]),
                        "loss_fts": float(losses["fts"][i, j]),
                    }
                )


def plot_loss_surface(
    *,
    path_png: Path,
    path_pdf: Path,
    ks_values: np.ndarray,
    cs_values: np.ndarray,
    loss_grid: np.ndarray,
    metric: str,
    win: object,
    clip_percentile: float,
    dpi: int,
) -> tuple[float, float, float]:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    best_i, best_j, best_loss = finite_min(loss_grid)
    best_ks = float(ks_values[best_i])
    best_cs = float(cs_values[best_j])
    plot_values = np.log10(np.where(loss_grid > 0.0, loss_grid, np.nan))
    finite = plot_values[np.isfinite(plot_values)]
    if finite.size == 0:
        raise RuntimeError("No finite loss values available for plotting")
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanpercentile(finite, float(clip_percentile)))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))

    kk, cc = np.meshgrid(ks_values, cs_values, indexing="ij")
    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    mesh = ax.pcolormesh(kk, cc, plot_values, shading="auto", cmap=RAINBOW_PURPLE_LOW_RED_HIGH, vmin=vmin, vmax=vmax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ks [N/m]")
    ax.set_ylabel("cs [N s/m]")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(f"log10({metric} loss)")
    ax.set_title(
        "DMT-KV ks/cs loss surface\n"
        f"window={getattr(win, 'label', 'unknown')} "
        f"t=[{1.0e6 * float(getattr(win, 't_start', math.nan)):.3f}, "
        f"{1.0e6 * float(getattr(win, 't_stop', math.nan)):.3f}] us | metric={metric}"
    )
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path_png, dpi=int(dpi))
    fig.savefig(path_pdf)
    plt.close(fig)
    return best_ks, best_cs, best_loss


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    win: object,
    best_ks: float,
    best_cs: float,
    best_loss: float,
    true_loss: float,
    png_path: Path,
    csv_path: Path,
) -> None:
    lines = [
        "DMT-KV ks/cs sweep summary",
        f"trajectory={args.csv}",
        f"window_mode={args.window_mode}",
        f"window_index={args.window_index}",
        f"window_label={getattr(win, 'label', 'unknown')}",
        f"window_indices=[{getattr(win, 'start_idx', 'nan')}, {getattr(win, 'stop_idx', 'nan')}]",
        f"window_time_us=[{1.0e6 * float(getattr(win, 't_start', math.nan)):.9f}, {1.0e6 * float(getattr(win, 't_stop', math.nan)):.9f}]",
        f"ks_nodes={args.ks_nodes}",
        f"cs_nodes={args.cs_nodes}",
        f"ks_bounds=[{args.ks_lo:.12e}, {args.ks_hi:.12e}]",
        f"cs_bounds=[{args.cs_lo:.12e}, {args.cs_hi:.12e}]",
        "range_source=AFM04.stage1pluslight.grid.KS_BOUNDS/CS_BOUNDS unless CLI overrides are used",
        f"metric={args.metric}",
        f"true_ks={float(KS_TRUE):.12e}",
        f"true_cs={float(CS_TRUE):.12e}",
        f"true_grid_loss={float(true_loss):.12e}",
        f"best_ks={best_ks:.12e}",
        f"best_cs={best_cs:.12e}",
        f"best_loss={best_loss:.12e}",
        f"best_ks_err_pct={100.0 * (best_ks - float(KS_TRUE)) / float(KS_TRUE):.6f}",
        f"best_cs_err_pct={100.0 * (best_cs - float(CS_TRUE)) / float(CS_TRUE):.6f}",
        f"csv={csv_path}",
        f"png={png_path}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    out_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DMT_KV_ROOT / "trajectory.csv")
    parser.add_argument("--out-dir", type=Path, default=out_root)
    parser.add_argument("--window-mode", default=DEFAULT_WINDOW_MODE)
    parser.add_argument("--window-us", type=float, default=DEFAULT_WINDOW_US)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--ks-nodes", type=int, default=600)
    parser.add_argument("--cs-nodes", type=int, default=600)
    parser.add_argument("--ks-lo", type=float, default=float(KS_BOUNDS[0]))
    parser.add_argument("--ks-hi", type=float, default=float(KS_BOUNDS[1]))
    parser.add_argument("--cs-lo", type=float, default=float(CS_BOUNDS[0]))
    parser.add_argument("--cs-hi", type=float, default=float(CS_BOUNDS[1]))
    parser.add_argument("--metric", choices=("observed", "state", "full", "x1", "x2", "x3", "x2dot", "fts"), default="observed")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--stem", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_trajectory(args.csv)
    idxs, win = select_window(data, args.window_mode, args.window_us, args.window_index)
    if idxs.size < 2:
        raise RuntimeError("Selected window has fewer than two samples")

    ks_values = log_values_with_required(args.ks_lo, args.ks_hi, args.ks_nodes, float(KS_TRUE))
    cs_values = log_values_with_required(args.cs_lo, args.cs_hi, args.cs_nodes, float(CS_TRUE))
    losses = sweep_losses(
        data=data,
        idxs=idxs,
        ks_values=ks_values,
        cs_values=cs_values,
        chunk_size=args.chunk_size,
    )

    stem = args.stem.strip()
    if not stem:
        role = str(getattr(win, "role", args.window_mode)).replace(" ", "_")
        stem = f"dmt_kv_ks_cs_loss_surface_{role}_{args.metric}_{args.ks_nodes}x{args.cs_nodes}"
    csv_path = args.out_dir / f"{stem}.csv"
    png_path = args.out_dir / f"{stem}.png"
    pdf_path = args.out_dir / f"{stem}.pdf"
    summary_path = args.out_dir / f"{stem}.txt"

    write_csv(csv_path, ks_values, cs_values, losses)
    best_ks, best_cs, best_loss = plot_loss_surface(
        path_png=png_path,
        path_pdf=pdf_path,
        ks_values=ks_values,
        cs_values=cs_values,
        loss_grid=losses[args.metric],
        metric=args.metric,
        win=win,
        clip_percentile=args.clip_percentile,
        dpi=args.dpi,
    )

    true_ks_idx = int(np.argmin(np.abs(ks_values - float(KS_TRUE))))
    true_cs_idx = int(np.argmin(np.abs(cs_values - float(CS_TRUE))))
    true_loss = float(losses[args.metric][true_ks_idx, true_cs_idx])
    write_summary(
        summary_path,
        args=args,
        win=win,
        best_ks=best_ks,
        best_cs=best_cs,
        best_loss=best_loss,
        true_loss=true_loss,
        png_path=png_path,
        csv_path=csv_path,
    )

    print(f"Window: {getattr(win, 'label', 'unknown')} [{getattr(win, 'start_idx', '?')}, {getattr(win, 'stop_idx', '?')}]")
    print(f"Metric: {args.metric}")
    print(f"Best: ks={best_ks:.9e}, cs={best_cs:.9e}, loss={best_loss:.9e}")
    print(f"True: ks={float(KS_TRUE):.9e}, cs={float(CS_TRUE):.9e}, nearest-grid loss={true_loss:.9e}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
