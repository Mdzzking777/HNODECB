from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter


SCRIPT_DIR = Path(__file__).resolve().parent
SILICON_DIR = SCRIPT_DIR.parent

LOCAL_MAT = SILICON_DIR / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
RAW_MAT = (
    Path(r"E:\External data SI_SI hard sample raw")
    / "OneDrive_2_2026-7-26"
    / "dataAFM_Si"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)
MIDDLE_GP_NPZ = (
    SILICON_DIR
    / "smooth"
    / "x2 raw"
    / "silicon_middle_x2_raw_global_gp_smooth.npz"
)
MIDDLE_CONTACT_NPZ = (
    SILICON_DIR
    / "GP"
    / "middle"
    / "x1 for contact judge"
    / "silicon_gp_index109_mid5000_490_510us_x1_contact_judge.npz"
)

INDEX = 109
DT_S = 1.0 / 250.0e6
START_US = 400.0
END_US = 800.0

Z_STATIC_M = 100.0e-9
A0_M = 0.165e-9

X1_SG_WINDOW = 201
X1_SG_POLYORDER = 3

NONCONTACT_GUARD_NS = 64.0
MAX_PHASE_SHIFT_US = 0.16
CONTACT_BUMP_PHASE_POINTS = 201

OUTPUT_STEM = "middle_gp_template_to_400_800_alignment"


def _load_index_signals() -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
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
            x1_all = np.asarray(data["displacement_fwd_sweep"][INDEX], dtype=float).ravel()
            x2_all = np.asarray(data["velocity_fwd_sweep"][INDEX], dtype=float).ravel()
            if x1_all.shape != x2_all.shape:
                raise ValueError(f"x1/x2 shapes differ: {x1_all.shape} vs {x2_all.shape}")
            time_all = np.arange(x1_all.size, dtype=float) * DT_S
            return time_all, x1_all, x2_all, mat_path
        except Exception as exc:
            errors.append(f"{mat_path}: {exc}")
    raise RuntimeError("Could not read MAT file:\n" + "\n".join(errors))


def _odd_window_at_most(length: int, preferred: int, polyorder: int) -> int:
    window = min(length, preferred)
    if window % 2 == 0:
        window -= 1
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    if window < minimum:
        raise ValueError("segment is too short for Savitzky-Golay smoothing")
    return window


def _contact_from_x1(time_s: np.ndarray, x1_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    window = _odd_window_at_most(time_s.size, X1_SG_WINDOW, X1_SG_POLYORDER)
    x1_tilde = savgol_filter(
        x1_raw,
        window_length=window,
        polyorder=X1_SG_POLYORDER,
        mode="interp",
    )
    contact = (Z_STATIC_M + x1_tilde) <= A0_M
    transition_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1
    return x1_tilde, contact, transition_idx, window


def _transition_kind(contact: np.ndarray, idx: int) -> str:
    before = bool(contact[idx - 1]) if idx > 0 else False
    after = bool(contact[idx]) if idx < contact.size else False
    if (not before) and after:
        return "N_to_C"
    if before and (not after):
        return "C_to_N"
    return "unknown"


def _build_middle_template() -> dict[str, np.ndarray | float | int]:
    with np.load(MIDDLE_GP_NPZ, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=float)
        x2_tilde = np.asarray(data["x2_tilde_m_s"], dtype=float)
    with np.load(MIDDLE_CONTACT_NPZ, allow_pickle=False) as data:
        contact = np.asarray(data["contact_tilde"], dtype=bool)
        transition_idx = np.asarray(data["transition_idx"], dtype=int)

    n_to_c = [int(idx) for idx in transition_idx if _transition_kind(contact, int(idx)) == "N_to_C"]
    if len(n_to_c) < 2:
        raise RuntimeError("middle slice does not contain enough N_to_C transitions for a cycle template")

    cycle_periods_s = np.diff(time_s[n_to_c])
    period_s = float(np.median(cycle_periods_s))
    n_template = int(round(period_s / DT_S))
    phase_s = np.arange(n_template, dtype=float) * DT_S

    cycles: list[np.ndarray] = []
    for start_idx, next_start_idx in zip(n_to_c[:-1], n_to_c[1:]):
        absolute_t = time_s[start_idx] + phase_s
        valid = absolute_t <= time_s[next_start_idx - 1]
        y = np.full(n_template, np.nan, dtype=float)
        y[valid] = np.interp(absolute_t[valid], time_s[start_idx:next_start_idx], x2_tilde[start_idx:next_start_idx])
        cycles.append(y)
    stack = np.vstack(cycles)
    template = np.nanmean(stack, axis=0)
    if np.any(~np.isfinite(template)):
        finite = np.flatnonzero(np.isfinite(template))
        template = np.interp(np.arange(template.size), finite, template[finite])

    return {
        "period_s": period_s,
        "phase_s": phase_s,
        "x2_template_m_s": template,
        "middle_n_to_c_indices": np.asarray(n_to_c, dtype=int),
        "middle_cycle_count_used": int(len(cycles)),
    }


def _eval_periodic_template(phase_s: np.ndarray, template_phase_s: np.ndarray, template: np.ndarray, period_s: float) -> np.ndarray:
    phase = np.mod(phase_s, period_s)
    phase_ext = np.concatenate([template_phase_s, [period_s]])
    template_ext = np.concatenate([template, [template[0]]])
    return np.interp(phase, phase_ext, template_ext)


def _fit_amplitude_offset_for_shift(template_values: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    design = np.column_stack([template_values, np.ones_like(template_values)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    fitted = a * template_values + b
    return a, b, fitted


def _fit_cycle_alignment(
    *,
    rel_time_s: np.ndarray,
    x2_raw: np.ndarray,
    fit_mask: np.ndarray,
    template_phase_s: np.ndarray,
    template: np.ndarray,
    period_s: float,
) -> tuple[float, float, float, float, float]:
    if np.count_nonzero(fit_mask) < 20:
        raise RuntimeError("not enough noncontact points for alignment")
    rel_fit = rel_time_s[fit_mask]
    y_fit = x2_raw[fit_mask]

    def objective(shift_s: float) -> float:
        x_template = _eval_periodic_template(rel_fit + shift_s, template_phase_s, template, period_s)
        _, _, fitted = _fit_amplitude_offset_for_shift(x_template, y_fit)
        return float(np.sqrt(np.mean((y_fit - fitted) ** 2)))

    result = minimize_scalar(
        objective,
        bounds=(-MAX_PHASE_SHIFT_US * 1.0e-6, MAX_PHASE_SHIFT_US * 1.0e-6),
        method="bounded",
        options={"xatol": 0.05e-9},
    )
    shift_s = float(result.x)
    x_template = _eval_periodic_template(rel_fit + shift_s, template_phase_s, template, period_s)
    a, b, fitted = _fit_amplitude_offset_for_shift(x_template, y_fit)
    residual = y_fit - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))
    median_abs = float(np.median(np.abs(residual)))
    return a, b, shift_s, rmse, median_abs


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    time_all, x1_all, x2_all, mat_path = _load_index_signals()
    mask = (time_all >= START_US * 1.0e-6) & (time_all <= END_US * 1.0e-6)
    time_s = time_all[mask]
    source_idx = np.flatnonzero(mask)
    x1_raw = x1_all[mask]
    x2_raw = x2_all[mask]
    x1_tilde, contact, transition_idx, x1_sg_window = _contact_from_x1(time_s, x1_raw)

    template_info = _build_middle_template()
    period_s = float(template_info["period_s"])
    template_phase_s = np.asarray(template_info["phase_s"], dtype=float)
    template = np.asarray(template_info["x2_template_m_s"], dtype=float)

    n_to_c = [int(idx) for idx in transition_idx if _transition_kind(contact, int(idx)) == "N_to_C"]
    c_to_n = [int(idx) for idx in transition_idx if _transition_kind(contact, int(idx)) == "C_to_N"]
    c_to_n_set = set(c_to_n)
    guard_points = int(round(NONCONTACT_GUARD_NS * 1.0e-9 / DT_S))

    cycle_id = np.full(time_s.size, -1, dtype=int)
    aligned_background = np.full(time_s.size, np.nan, dtype=float)
    residual_all = np.full(time_s.size, np.nan, dtype=float)
    residual_contact = np.full(time_s.size, np.nan, dtype=float)

    rows: list[dict[str, float | int | str]] = []
    contact_phase_grid = np.linspace(0.0, 1.0, CONTACT_BUMP_PHASE_POINTS)
    contact_residual_stack: list[np.ndarray] = []
    contact_raw_stack: list[np.ndarray] = []
    contact_bg_stack: list[np.ndarray] = []

    cycle_index = 0
    for start_idx, next_start_idx in zip(n_to_c[:-1], n_to_c[1:]):
        contact_end_candidates = [idx for idx in c_to_n_set if start_idx < idx < next_start_idx]
        if len(contact_end_candidates) != 1:
            continue
        contact_end_idx = int(contact_end_candidates[0])

        local_slice = slice(start_idx, next_start_idx)
        rel_time_s = time_s[local_slice] - time_s[start_idx]
        y_cycle = x2_raw[local_slice]
        local_contact = contact[local_slice]

        fit_mask = np.zeros_like(local_contact, dtype=bool)
        local_contact_end = contact_end_idx - start_idx
        nc_start = min(local_contact.size, local_contact_end + guard_points)
        nc_stop = max(nc_start, local_contact.size - guard_points)
        fit_mask[nc_start:nc_stop] = ~local_contact[nc_start:nc_stop]
        if np.count_nonzero(fit_mask) < 20:
            fit_mask = ~local_contact

        try:
            a, b, shift_s, non_rmse, non_median_abs = _fit_cycle_alignment(
                rel_time_s=rel_time_s,
                x2_raw=y_cycle,
                fit_mask=fit_mask,
                template_phase_s=template_phase_s,
                template=template,
                period_s=period_s,
            )
        except RuntimeError:
            continue

        template_values = _eval_periodic_template(rel_time_s + shift_s, template_phase_s, template, period_s)
        bg = a * template_values + b
        resid = y_cycle - bg

        cycle_id[local_slice] = cycle_index
        aligned_background[local_slice] = bg
        residual_all[local_slice] = resid
        contact_local = np.arange(start_idx, contact_end_idx)
        residual_contact[contact_local] = residual_all[contact_local]

        contact_slice = slice(start_idx, contact_end_idx)
        contact_resid = residual_all[contact_slice]
        contact_raw = x2_raw[contact_slice]
        contact_bg = aligned_background[contact_slice]
        contact_phase = np.linspace(0.0, 1.0, contact_resid.size)
        contact_residual_stack.append(np.interp(contact_phase_grid, contact_phase, contact_resid))
        contact_raw_stack.append(np.interp(contact_phase_grid, contact_phase, contact_raw))
        contact_bg_stack.append(np.interp(contact_phase_grid, contact_phase, contact_bg))

        rows.append(
            {
                "cycle_index": cycle_index,
                "cycle_start_time_us": float(time_s[start_idx] * 1.0e6),
                "contact_end_time_us": float(time_s[contact_end_idx] * 1.0e6),
                "cycle_end_time_us": float(time_s[next_start_idx] * 1.0e6),
                "cycle_duration_us": float((time_s[next_start_idx] - time_s[start_idx]) * 1.0e6),
                "contact_duration_us": float((time_s[contact_end_idx] - time_s[start_idx]) * 1.0e6),
                "cycle_sample_count": int(next_start_idx - start_idx),
                "contact_sample_count": int(contact_end_idx - start_idx),
                "noncontact_fit_sample_count": int(np.count_nonzero(fit_mask)),
                "amplitude_scale_a": a,
                "offset_b_m_s": b,
                "phase_shift_us": float(shift_s * 1.0e6),
                "noncontact_fit_rmse_m_s": non_rmse,
                "noncontact_fit_median_abs_m_s": non_median_abs,
                "contact_bump_mean_m_s": float(np.mean(contact_resid)),
                "contact_bump_rms_m_s": float(np.sqrt(np.mean(contact_resid**2))),
                "contact_bump_median_abs_m_s": float(np.median(np.abs(contact_resid))),
                "contact_bump_min_m_s": float(np.min(contact_resid)),
                "contact_bump_max_m_s": float(np.max(contact_resid)),
                "contact_bump_ptp_m_s": float(np.ptp(contact_resid)),
                "contact_bump_start_m_s": float(contact_resid[0]),
                "contact_bump_end_m_s": float(contact_resid[-1]),
            }
        )
        cycle_index += 1

    if not rows:
        raise RuntimeError("no cycles were aligned")

    residual_contact_by_cycle = np.vstack(contact_residual_stack)
    raw_contact_by_cycle = np.vstack(contact_raw_stack)
    bg_contact_by_cycle = np.vstack(contact_bg_stack)

    csv_path = SCRIPT_DIR / f"{OUTPUT_STEM}_cycles.csv"
    npz_path = SCRIPT_DIR / f"{OUTPUT_STEM}.npz"
    json_path = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
    txt_path = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.txt"

    _write_csv(csv_path, rows)

    a_values = np.asarray([float(row["amplitude_scale_a"]) for row in rows], dtype=float)
    b_values = np.asarray([float(row["offset_b_m_s"]) for row in rows], dtype=float)
    shift_values_us = np.asarray([float(row["phase_shift_us"]) for row in rows], dtype=float)
    non_rmse_values = np.asarray([float(row["noncontact_fit_rmse_m_s"]) for row in rows], dtype=float)
    contact_rms_values = np.asarray([float(row["contact_bump_rms_m_s"]) for row in rows], dtype=float)
    contact_ptp_values = np.asarray([float(row["contact_bump_ptp_m_s"]) for row in rows], dtype=float)

    summary = {
        "purpose": "align middle-slice strong-GP x2 background template to 400-800 us raw x2 cycles using only noncontact interior points",
        "mat_path": str(mat_path),
        "index": INDEX,
        "time_range_us": [START_US, END_US],
        "sample_count": int(time_s.size),
        "dt_s": DT_S,
        "source_start_index": int(source_idx[0]),
        "source_end_index_inclusive": int(source_idx[-1]),
        "contact_judge": {
            "x1_smoothing": "Savitzky-Golay only for contact judgment",
            "x1_sg_window_points": int(x1_sg_window),
            "x1_sg_window_us": float(x1_sg_window * DT_S * 1.0e6),
            "x1_sg_polyorder": X1_SG_POLYORDER,
            "contact_rule": "contact if z_static + x1_tilde <= a0",
            "z_static_m": Z_STATIC_M,
            "a0_m": A0_M,
        },
        "middle_template": {
            "source_npz": str(MIDDLE_GP_NPZ),
            "contact_source_npz": str(MIDDLE_CONTACT_NPZ),
            "period_s": period_s,
            "period_us": float(period_s * 1.0e6),
            "template_sample_count": int(template.size),
            "middle_cycle_count_used": int(template_info["middle_cycle_count_used"]),
        },
        "alignment_policy": {
            "cycle_definition": "N_to_C transition to next N_to_C transition",
            "fit_region": "noncontact interior only",
            "noncontact_guard_ns": NONCONTACT_GUARD_NS,
            "parameters": "amplitude scale a, offset b, phase shift dt",
            "max_phase_shift_us": MAX_PHASE_SHIFT_US,
        },
        "aligned_cycle_count": int(len(rows)),
        "transition_count": int(transition_idx.size),
        "n_to_c_transition_count": int(len(n_to_c)),
        "c_to_n_transition_count": int(len(c_to_n)),
        "fit_summary": {
            "amplitude_scale_a_mean": float(np.mean(a_values)),
            "amplitude_scale_a_std": float(np.std(a_values, ddof=1)),
            "offset_b_m_s_mean": float(np.mean(b_values)),
            "offset_b_m_s_std": float(np.std(b_values, ddof=1)),
            "phase_shift_us_mean": float(np.mean(shift_values_us)),
            "phase_shift_us_std": float(np.std(shift_values_us, ddof=1)),
            "noncontact_fit_rmse_m_s_mean": float(np.mean(non_rmse_values)),
            "noncontact_fit_rmse_m_s_median": float(np.median(non_rmse_values)),
            "contact_bump_rms_m_s_mean": float(np.mean(contact_rms_values)),
            "contact_bump_rms_m_s_median": float(np.median(contact_rms_values)),
            "contact_bump_ptp_m_s_mean": float(np.mean(contact_ptp_values)),
            "contact_bump_ptp_m_s_median": float(np.median(contact_ptp_values)),
        },
        "outputs": {
            "csv": str(csv_path),
            "npz": str(npz_path),
            "json": str(json_path),
            "txt": str(txt_path),
        },
    }

    np.savez_compressed(
        npz_path,
        time_s=time_s,
        source_idx=source_idx,
        x1_raw_m=x1_raw,
        x1_tilde_m=x1_tilde,
        x2_raw_m_s=x2_raw,
        contact=contact,
        transition_idx=transition_idx,
        cycle_id=cycle_id,
        x2_background_aligned_m_s=aligned_background,
        x2_residual_all_m_s=residual_all,
        x2_bump_residual_contact_only_m_s=residual_contact,
        template_period_s=np.asarray(period_s),
        template_phase_s=template_phase_s,
        template_x2_background_m_s=template,
        contact_bump_phase_grid=contact_phase_grid,
        contact_bump_residual_by_cycle_m_s=residual_contact_by_cycle,
        contact_bump_residual_mean_m_s=np.mean(residual_contact_by_cycle, axis=0),
        contact_bump_residual_median_m_s=np.median(residual_contact_by_cycle, axis=0),
        contact_x2_raw_by_cycle_m_s=raw_contact_by_cycle,
        contact_x2_background_by_cycle_m_s=bg_contact_by_cycle,
        cycle_amplitude_scale_a=a_values,
        cycle_offset_b_m_s=b_values,
        cycle_phase_shift_us=shift_values_us,
        cycle_noncontact_fit_rmse_m_s=non_rmse_values,
        cycle_contact_bump_rms_m_s=contact_rms_values,
    )

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "Middle-GP template alignment to 400-800 us raw x2",
        "=================================================",
        "",
        "No visualization was generated by this script.",
        "",
        f"Time range: {START_US:.1f}-{END_US:.1f} us",
        f"Samples: {time_s.size}",
        f"Aligned complete cycles: {len(rows)}",
        f"Template period: {period_s * 1.0e6:.6f} us",
        f"Template middle cycles used: {int(template_info['middle_cycle_count_used'])}",
        "",
        "Alignment policy:",
        "  cycle = N_to_C to next N_to_C",
        "  fit region = noncontact interior only",
        f"  noncontact guard = {NONCONTACT_GUARD_NS:.1f} ns",
        "  fitted parameters = amplitude scale a, offset b, phase shift dt",
        "",
        "Summary:",
        f"  a mean/std = {summary['fit_summary']['amplitude_scale_a_mean']:.6g} / {summary['fit_summary']['amplitude_scale_a_std']:.6g}",
        f"  b mean/std = {summary['fit_summary']['offset_b_m_s_mean']:.6e} / {summary['fit_summary']['offset_b_m_s_std']:.6e} m s^-1",
        f"  phase shift mean/std = {summary['fit_summary']['phase_shift_us_mean']:.6g} / {summary['fit_summary']['phase_shift_us_std']:.6g} us",
        f"  noncontact RMSE mean/median = {summary['fit_summary']['noncontact_fit_rmse_m_s_mean']:.6e} / {summary['fit_summary']['noncontact_fit_rmse_m_s_median']:.6e} m s^-1",
        f"  contact bump RMS mean/median = {summary['fit_summary']['contact_bump_rms_m_s_mean']:.6e} / {summary['fit_summary']['contact_bump_rms_m_s_median']:.6e} m s^-1",
        f"  contact bump peak-to-peak mean/median = {summary['fit_summary']['contact_bump_ptp_m_s_mean']:.6e} / {summary['fit_summary']['contact_bump_ptp_m_s_median']:.6e} m s^-1",
        "",
        f"Saved CSV: {csv_path}",
        f"Saved NPZ: {npz_path}",
        f"Saved JSON: {json_path}",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
