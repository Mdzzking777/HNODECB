from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.config import default_config
from AFM04.stage1pluslight.data import load_dataset, truncate_to_first_contact
from AFM04.stage1pluslight.windows import window_manifests


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "AFM04" / "KAN_full_test" / "logs" / "analysis"
TEMPORAL_EXCLUSION_POINTS = 25
DIST_EPS = 1.0e-12
METHOD3_FOLDS = 5
METHOD3_KNN_K = 8


def _double_window_bounds(start_idx: int, stop_idx: int, total_len: int) -> tuple[int, int]:
    length = int(stop_idx) - int(start_idx) + 1
    target = min(int(total_len), max(length, 2 * length))
    center = 0.5 * (int(start_idx) + int(stop_idx))
    new_start = int(round(center - 0.5 * (target - 1)))
    new_stop = new_start + target - 1
    if new_start < 0:
        new_stop += -new_start
        new_start = 0
    if new_stop >= total_len:
        shift = new_stop - total_len + 1
        new_start = max(0, new_start - shift)
        new_stop = total_len - 1
    return int(new_start), int(new_stop)


def _window_short(role: str) -> str:
    role_norm = str(role).strip().lower()
    if role_norm == "first_contact":
        return "W0"
    if role_norm == "middle":
        return "W1"
    if role_norm == "max_x1_pp_change":
        return "W2"
    if role_norm == "tail_stable":
        return "W3"
    return role or "W?"


def _robust_normalize(reference: np.ndarray, query: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    ref = np.asarray(reference, dtype=float)
    q = None if query is None else np.asarray(query, dtype=float)
    center = np.median(ref, axis=0)
    q75 = np.quantile(ref, 0.75, axis=0)
    q25 = np.quantile(ref, 0.25, axis=0)
    scale = np.maximum(q75 - q25, 1.0e-30)
    ref_n = (ref - center[None, :]) / scale[None, :]
    q_n = None if q is None else (q - center[None, :]) / scale[None, :]
    return ref_n, q_n


def _nearest_with_temporal_exclusion(x_norm: np.ndarray, global_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(x_norm.shape[0])
    dist2 = np.sum((x_norm[:, None, :] - x_norm[None, :, :]) ** 2, axis=2)
    temporal_gap = np.abs(global_idx[:, None] - global_idx[None, :])
    dist2[temporal_gap <= TEMPORAL_EXCLUSION_POINTS] = np.inf
    nearest_local = np.argmin(dist2, axis=1)
    nearest_dist = np.sqrt(dist2[np.arange(n), nearest_local])
    finite = np.isfinite(nearest_dist)
    return nearest_local[finite], nearest_dist[finite]


def _nearest_from_reference(query_norm: np.ndarray, ref_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist2 = np.sum((query_norm[:, None, :] - ref_norm[None, :, :]) ** 2, axis=2)
    nearest_local = np.argmin(dist2, axis=1)
    nearest_dist = np.sqrt(dist2[np.arange(query_norm.shape[0]), nearest_local])
    return nearest_local, nearest_dist


def _rel_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    den = float(np.sqrt(np.mean(np.square(truth_arr))))
    if den <= 0.0 or not np.isfinite(den):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pred_arr - truth_arr))) / den)


def _r2_score(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    den = float(np.sum(np.square(truth_arr - np.mean(truth_arr))))
    if den <= 0.0 or not np.isfinite(den):
        return float("nan")
    return float(1.0 - np.sum(np.square(pred_arr - truth_arr)) / den)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {key: float("nan") for key in ("p50", "p90", "p95", "p99", "max")}
    return {
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _neighbor_ambiguity_metrics(
    *,
    states: np.ndarray,
    global_idx: np.ndarray,
    fts: np.ndarray,
    x3dot: np.ndarray,
) -> dict[str, float]:
    x_norm, _ = _robust_normalize(states)
    nearest_local, nearest_dist = _nearest_with_temporal_exclusion(x_norm, global_idx)
    valid_source = np.arange(len(global_idx), dtype=int)
    valid_source = valid_source[np.isfinite(np.sum((x_norm[:, None, :] - x_norm[None, :, :]) ** 2, axis=2)).any(axis=1)]
    if len(valid_source) != len(nearest_local):
        # Recompute source mask cheaply and explicitly; readability matters more than clever indexing here.
        dist2 = np.sum((x_norm[:, None, :] - x_norm[None, :, :]) ** 2, axis=2)
        temporal_gap = np.abs(global_idx[:, None] - global_idx[None, :])
        dist2[temporal_gap <= TEMPORAL_EXCLUSION_POINTS] = np.inf
        finite = np.isfinite(np.sqrt(np.min(dist2, axis=1)))
        valid_source = np.arange(len(global_idx), dtype=int)[finite]

    source = valid_source
    target = nearest_local
    f_rms = max(float(np.sqrt(np.mean(np.square(fts)))), 1.0e-300)
    x3dot_rms = max(float(np.sqrt(np.mean(np.square(x3dot)))), 1.0e-300)
    d_fts_rel = np.abs(fts[source] - fts[target]) / f_rms
    d_x3dot_rel = np.abs(x3dot[source] - x3dot[target]) / x3dot_rms
    dist = np.maximum(nearest_dist, DIST_EPS)
    f_fragility = d_fts_rel / dist
    x3dot_fragility = d_x3dot_rel / dist

    out = {
        "neighbor_pairs": float(len(dist)),
        "neighbor_dist_p50": _quantiles(dist)["p50"],
        "neighbor_dist_p95": _quantiles(dist)["p95"],
        "fts_rel_delta_p95": _quantiles(d_fts_rel)["p95"],
        "x3dot_rel_delta_p95": _quantiles(d_x3dot_rel)["p95"],
    }
    for prefix, values in (("fts_fragility", f_fragility), ("x3dot_fragility", x3dot_fragility)):
        for key, value in _quantiles(values).items():
            out[f"{prefix}_{key}"] = value
    return out


def _knn_predict(
    train_states: np.ndarray,
    train_values: np.ndarray,
    test_states: np.ndarray,
    *,
    k: int = METHOD3_KNN_K,
) -> tuple[np.ndarray, np.ndarray]:
    train_norm, test_norm = _robust_normalize(train_states, test_states)
    assert test_norm is not None
    dist2 = np.sum((test_norm[:, None, :] - train_norm[None, :, :]) ** 2, axis=2)
    dist = np.sqrt(dist2)
    k_eff = max(1, min(int(k), train_norm.shape[0]))
    kth = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    kth_dist = np.take_along_axis(dist, kth, axis=1)
    kth_values = np.asarray(train_values, dtype=float)[kth]
    weights = 1.0 / np.maximum(kth_dist, DIST_EPS)
    pred = np.sum(weights * kth_values, axis=1) / np.sum(weights, axis=1)
    nearest_dist = np.min(kth_dist, axis=1)
    return pred, nearest_dist


def _state_only_x3dot_approximability_metrics(
    *,
    states: np.ndarray,
    x3dot: np.ndarray,
    folds: int = METHOD3_FOLDS,
    k: int = METHOD3_KNN_K,
) -> dict[str, float]:
    """Method 3: can x3dot be approximated from only (x1, x2, x3)?

    This is intentionally a held-out, state-only postprocess test.  It does not
    use rollout states, KFT predictions, 2x extrapolation labels, or any force
    model internals.  Contiguous blocked folds avoid the trivial answer from
    adjacent time samples on the same trajectory.
    """

    x = np.asarray(states, dtype=float)
    y = np.asarray(x3dot, dtype=float)
    n = int(x.shape[0])
    fold_count = max(2, min(int(folds), n))
    fold_ids = np.array_split(np.arange(n, dtype=int), fold_count)

    pred_all = np.full(n, np.nan, dtype=float)
    nearest_all = np.full(n, np.nan, dtype=float)
    for test_local in fold_ids:
        if test_local.size == 0:
            continue
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_local] = False
        train_local = np.flatnonzero(train_mask)
        if train_local.size == 0:
            continue
        pred, nearest_dist = _knn_predict(x[train_local, :], y[train_local], x[test_local, :], k=k)
        pred_all[test_local] = pred
        nearest_all[test_local] = nearest_dist

    finite = np.isfinite(pred_all)
    if not np.any(finite):
        return {
            "method3_points": 0.0,
            "method3_knn_k": float(k),
            "method3_folds": float(fold_count),
            "method3_x3dot_rel_rmse": float("nan"),
            "method3_x3dot_r2": float("nan"),
            "method3_neighbor_dist_p50": float("nan"),
            "method3_neighbor_dist_p95": float("nan"),
        }

    err_rel = np.abs(pred_all[finite] - y[finite]) / max(float(np.sqrt(np.mean(y[finite] ** 2))), 1.0e-300)
    out = {
        "method3_points": float(np.sum(finite)),
        "method3_knn_k": float(k),
        "method3_folds": float(fold_count),
        "method3_x3dot_rel_rmse": _rel_rmse(pred_all[finite], y[finite]),
        "method3_x3dot_r2": _r2_score(pred_all[finite], y[finite]),
        "method3_neighbor_dist_p50": _quantiles(nearest_all[finite])["p50"],
        "method3_neighbor_dist_p95": _quantiles(nearest_all[finite])["p95"],
    }
    for key, value in _quantiles(err_rel).items():
        out[f"method3_x3dot_abs_rel_error_{key}"] = value
    return out


def _inferability_metrics(
    *,
    train_states: np.ndarray,
    train_x3dot: np.ndarray,
    test_states: np.ndarray,
    test_x3dot: np.ndarray,
    test_fts: np.ndarray,
    train_fts: np.ndarray,
) -> dict[str, float]:
    train_norm, test_norm = _robust_normalize(train_states, test_states)
    assert test_norm is not None
    nearest_local, nearest_dist = _nearest_from_reference(test_norm, train_norm)
    x3dot_nn = train_x3dot[nearest_local]
    fts_nn = train_fts[nearest_local]
    x3dot_err_rel = np.abs(x3dot_nn - test_x3dot) / max(float(np.sqrt(np.mean(test_x3dot**2))), 1.0e-300)
    fts_err_rel = np.abs(fts_nn - test_fts) / max(float(np.sqrt(np.mean(test_fts**2))), 1.0e-300)
    out = {
        "extrap_points": float(len(test_x3dot)),
        "extrap_neighbor_dist_p50": _quantiles(nearest_dist)["p50"],
        "extrap_neighbor_dist_p95": _quantiles(nearest_dist)["p95"],
        "x3dot_nn_rel_rmse": _rel_rmse(x3dot_nn, test_x3dot),
        "fts_nn_rel_rmse": _rel_rmse(fts_nn, test_fts),
    }
    for prefix, values in (("x3dot_nn_abs_rel_error", x3dot_err_rel), ("fts_nn_abs_rel_error", fts_err_rel)):
        for key, value in _quantiles(values).items():
            out[f"{prefix}_{key}"] = value
    return out


def main() -> None:
    cfg = default_config(REPO_ROOT)
    ode_data, pert_df = load_dataset(cfg.dataset_root, cfg.error_level, auto_generate=False)
    ode_data, pert_df = truncate_to_first_contact(ode_data, pert_df)

    times = np.asarray(pert_df["t"], dtype=float)
    contact = np.asarray(pert_df["contact"], dtype=bool)
    states_all = np.asarray(ode_data.T, dtype=float)
    x1_signal = np.asarray(ode_data[0, :], dtype=float)
    x3dot_all = np.asarray(pert_df["x3dot"], dtype=float)
    fts_all = np.asarray(pert_df["fts"], dtype=float)

    manifests = [
        *window_manifests(times, contact, "w0", float(cfg.arch_window_us), x1_signal=x1_signal),
        *window_manifests(times, contact, "stage2_w123", float(cfg.arch_window_us), x1_signal=x1_signal),
    ]

    rows: list[dict[str, float | str]] = []
    for man in manifests:
        train_idx = np.asarray(man.idxs, dtype=int)
        d_start, d_stop = _double_window_bounds(int(man.start_idx), int(man.stop_idx), len(times))
        double_idx = np.arange(d_start, d_stop + 1, dtype=int)
        extrap_idx = np.setdiff1d(double_idx, train_idx, assume_unique=False)

        row: dict[str, float | str] = {
            "window": _window_short(str(man.role)),
            "role": str(man.role),
            "train_t_us": f"{man.t_start * 1.0e6:.3f}-{man.t_stop * 1.0e6:.3f}",
            "double_t_us": f"{times[d_start] * 1.0e6:.3f}-{times[d_stop] * 1.0e6:.3f}",
            "train_points": float(len(train_idx)),
            "contact_frac": float(np.mean(contact[train_idx])),
        }
        row.update(
            {
                f"train_{key}": value
                for key, value in _neighbor_ambiguity_metrics(
                    states=states_all[train_idx, :],
                    global_idx=train_idx,
                    fts=fts_all[train_idx],
                    x3dot=x3dot_all[train_idx],
                ).items()
            }
        )
        row.update(
            _state_only_x3dot_approximability_metrics(
                states=states_all[train_idx, :],
                x3dot=x3dot_all[train_idx],
            )
        )
        if len(extrap_idx) > 0:
            row.update(
                {
                    f"extrap_{key}": value
                    for key, value in _inferability_metrics(
                        train_states=states_all[train_idx, :],
                        train_x3dot=x3dot_all[train_idx],
                        test_states=states_all[extrap_idx, :],
                        test_x3dot=x3dot_all[extrap_idx],
                        test_fts=fts_all[extrap_idx],
                        train_fts=fts_all[train_idx],
                    ).items()
                }
            )
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "kft_x3dot_omission_fragility_analysis.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    txt_path = OUT_DIR / "kft_x3dot_omission_fragility_analysis.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("KFT x3dot omission fragility analysis\n")
        f.write("Methods:\n")
        f.write("  Method 2: neighbor ambiguity inside each training window.\n")
        f.write(
            "  Method 3: state-only held-out x3dot approximability, "
            "i.e. can (x1,x2,x3) predict x3dot without using time, rollout, or KFT outputs?\n"
        )
        f.write("  Extra diagnostic: nearest-state 2x extrapolation check, not the Method 3 conclusion.\n")
        f.write("No training, no optimizer, no rollout.\n\n")
        for row in rows:
            f.write(
                f"{row['window']} ({row['role']}) | train={row['train_t_us']} | "
                f"2x={row['double_t_us']} | contact_frac={row['contact_frac']:.4f}\n"
            )
            f.write(
                "  neighbor ambiguity: "
                f"fts_fragility_p95={row['train_fts_fragility_p95']:.6e}, "
                f"x3dot_fragility_p95={row['train_x3dot_fragility_p95']:.6e}, "
                f"neighbor_dist_p50={row['train_neighbor_dist_p50']:.6e}\n"
            )
            f.write(
                "  Method 3 state-only x3dot approximation: "
                f"blocked_cv_knn_rel_rmse={row['method3_x3dot_rel_rmse']:.6e}, "
                f"r2={row['method3_x3dot_r2']:.6e}, "
                f"nearest_state_dist_p50={row['method3_neighbor_dist_p50']:.6e}\n"
            )
            f.write(
                "  extra diagnostic, nearest-state 2x extrap: "
                f"x3dot_nn_rel_rmse={row['extrap_x3dot_nn_rel_rmse']:.6e}, "
                f"fts_nn_rel_rmse={row['extrap_fts_nn_rel_rmse']:.6e}, "
                f"neighbor_dist_p50={row['extrap_extrap_neighbor_dist_p50']:.6e}\n\n"
            )
    print(f"Saved analysis CSV to: {csv_path}")
    print(f"Saved analysis summary to: {txt_path}")


if __name__ == "__main__":
    main()
