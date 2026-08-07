"""Generate AFM06a e0.0_real data from the settings manifest.

This generator is intentionally separate from the legacy AFM06a generator:
the interaction force is stored as the physical force Fts, and the RHS uses
(1/m) * Fts rather than a pre-divided force column.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.datasets.non_perturbed_dataset_generator import (  # noqa: E402
    generate_non_perturbed_training_set,
    save_table_npz,
    table_to_matrix,
)
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (  # noqa: E402
    B1,
    C1,
    C2,
    ETA_STAR_LENGTH,
    OMEGA_BAR,
    Q,
    REFERENCE_FORCE_OMEGA0,
    Y_BAR,
)


ERROR_LEVEL = "e0.0_real"
DATA_COLUMNS = ("t", "x1", "x2", "x2dot", "contact", "s", "delta", "Fts")
STATE_COLUMNS = ("t", "x1", "x2")
END_TIME_S = 15.0e-3
DT_S = 16.0e-9


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "afm06a_generation_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"generation manifest must be a JSON object: {path}")
    return payload


def _refresh_fixed_force_coefficients(manifest: dict[str, Any]) -> None:
    names = [str(name) for name in manifest.get("parameter_names", [])]
    values = [float(value) for value in manifest.get("parameter_vector", [])]
    if len(names) != len(values):
        raise ValueError("manifest parameter_names and parameter_vector lengths differ")
    p = dict(zip(names, values, strict=True))
    required = ("omega0", "m", "c", "Fd", "CA_F", "CH_F")
    missing = [name for name in required if name not in p]
    if missing:
        raise ValueError("manifest is missing physical e0.0_real parameters: " + ", ".join(missing))

    eta = float(ETA_STAR_LENGTH)
    omega_ref = float(REFERENCE_FORCE_OMEGA0)
    ca_bar = float(C1) * (eta**3) * (omega_ref**2)
    ch_bar = float(C2) * (omega_ref**2) / math.sqrt(eta)
    ca_f = float(p["m"]) * ca_bar
    ch_f = float(p["m"]) * ch_bar
    p["CA_F"] = ca_f
    p["CH_F"] = ch_f
    manifest["parameter_vector"] = [p[name] for name in names]

    fixed = manifest.setdefault("fixed_dmt_coefficients", {})
    fixed.update(
        {
            "policy": "infer the mass-normalized CA_bar and CH_bar once from the AFM06a equivalent coefficients, then convert to physical force coefficients CA_F=m*CA_bar and CH_F=m*CH_bar for Fts",
            "CA_bar_definition": "CA_bar = -A*R/(6*m) = C1 * eta_star_length^3 * omega0_ref^2",
            "CH_bar_definition": "CH_bar = 4*E_star*sqrt(R)/(3*m) = C2 * omega0_ref^2 / sqrt(eta_star_length)",
            "CA_F_definition": "CA_F = m * CA_bar = -A*R/6",
            "CH_F_definition": "CH_F = m * CH_bar = 4*E_star*sqrt(R)/3",
            "C1": float(C1),
            "C2": float(C2),
            "eta_star_length_m": eta,
            "omega0_ref_rad_s": omega_ref,
            "CA_bar_value": ca_bar,
            "CH_bar_value": ch_bar,
            "mass_for_conversion_kg": float(p["m"]),
            "CA_F_value": ca_f,
            "CH_F_value": ch_f,
            "invariance": "CA_F and CH_F are fixed physical tip-sample force coefficients; changing omega0 later does not rescale them",
        }
    )

    physical = manifest.setdefault("physical_constants", {})
    physical.update(
        {
            "Q": float(Q),
            "Fd_N": float(p["Fd"]),
            "Fd_over_m_m_s2": float(p["Fd"] / p["m"]),
            "CA_F": ca_f,
            "CH_F": ch_f,
            "CA_bar": ca_bar,
            "CH_bar": ch_bar,
        }
    )
    rhs_policy = manifest.setdefault("rhs_coefficient_policy", {})
    rhs_policy.update(
        {
            "Fd_over_m_m_s2": float(p["Fd"] / p["m"]),
            "damping_coefficient_value_s_minus1": float(p["c"] / p["m"]),
            "x1_coefficient_value_s_minus2": float(p.get("k", 0.0) / p["m"]) if "k" in p else rhs_policy.get("x1_coefficient_value_s_minus2"),
        }
    )

    legacy_names = [str(name) for name in manifest.get("legacy_parameter_names", [])]
    if legacy_names:
        omega0 = float(p["omega0"])
        legacy_values: dict[str, float] = {
            "omega0": omega0,
            "c1": ca_bar / ((omega0**2) * (eta**3)),
            "c2": ch_bar * math.sqrt(eta) / (omega0**2),
            "b1": float(p["Fd"] / p["m"]) / ((omega0**2) * eta * (float(OMEGA_BAR) ** 2) * float(Y_BAR)),
            "d1": float(p["c"] / (p["m"] * omega0)),
            "d2": float(p["c"] / (p["m"] * omega0)),
            "eta_star_length": eta,
            "y_bar": float(Y_BAR),
            "omega_bar": float(OMEGA_BAR),
            "dist": float(p.get("dist", 0.0)),
            "a0": float(p.get("a0", 0.0)),
            "beta": float(p.get("beta", 0.0)),
        }
        manifest["legacy_parameter_vector"] = [legacy_values[name] for name in legacy_names]


def _parameter_dict(manifest: dict[str, Any]) -> dict[str, float]:
    names = manifest.get("parameter_names")
    values = manifest.get("parameter_vector")
    if not isinstance(names, list) or not isinstance(values, list):
        raise ValueError("manifest must define parameter_names and parameter_vector")
    out = {str(name): float(value) for name, value in zip(names, values, strict=True)}
    required = ("k", "omega0", "m", "c", "Fd", "CA_F", "CH_F", "dist", "a0", "beta")
    missing = [name for name in required if name not in out]
    if missing:
        raise ValueError("manifest is missing physical e0.0_real parameters: " + ", ".join(missing))
    return out


def _fts_force_from_x1(x1: float, p: dict[str, float]) -> tuple[float, float, float, bool]:
    s = float(p["dist"]) + float(x1)
    a0 = float(p["a0"])
    delta = max(a0 - s, 0.0)
    contact = s <= a0
    denom = a0 if contact else max(s, 1.0e-15)
    fts = float(p["CA_F"]) / (denom**2)
    if contact:
        fts += float(p["CH_F"]) * (delta**1.5)
    return fts, s, delta, contact


def _rhs(t: float, u: list[float], p: dict[str, float]) -> list[float]:
    x1, x2 = float(u[0]), float(u[1])
    fts, _s, _delta, _contact = _fts_force_from_x1(x1, p)
    x2dot = (
        float(p["Fd"]) * math.sin(float(p["omega0"]) * float(t))
        - float(p["c"]) * x2
        - float(p["k"]) * x1
        + fts
    ) / float(p["m"])
    return [x2, x2dot]


def _build_data_table(solution_table: dict[str, np.ndarray], p: dict[str, float]) -> dict[str, np.ndarray]:
    t = np.asarray(solution_table["t"], dtype=float)
    x1 = np.asarray(solution_table["x1"], dtype=float)
    x2 = np.asarray(solution_table["x2"], dtype=float)
    s = np.empty_like(x1)
    delta = np.empty_like(x1)
    contact = np.empty(x1.shape, dtype=int)
    fts = np.empty_like(x1)
    x2dot = np.empty_like(x1)
    for i, (ti, x1i, x2i) in enumerate(zip(t, x1, x2, strict=True)):
        ftsi, si, deltai, contacti = _fts_force_from_x1(float(x1i), p)
        s[i] = si
        delta[i] = deltai
        contact[i] = int(contacti)
        fts[i] = ftsi
        x2dot[i] = (
            float(p["Fd"]) * math.sin(float(p["omega0"]) * float(ti))
            - float(p["c"]) * float(x2i)
            - float(p["k"]) * float(x1i)
            + ftsi
        ) / float(p["m"])
    return {
        "t": t.copy(),
        "x1": x1.copy(),
        "x2": x2.copy(),
        "x2dot": x2dot,
        "contact": contact,
        "s": s,
        "delta": delta,
        "Fts": fts,
    }


def _zeros_table_like(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in table.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.integer):
            out[key] = np.zeros_like(arr, dtype=int)
        else:
            out[key] = np.zeros_like(arr, dtype=float)
    return out


def _update_manifest(
    manifest: dict[str, Any],
    *,
    data_dir: Path,
    p: dict[str, float],
    table: dict[str, np.ndarray],
    nsteps: int,
) -> None:
    times = np.asarray(table["t"], dtype=float)
    x1 = np.asarray(table["x1"], dtype=float)
    fts = np.asarray(table["Fts"], dtype=float)
    contact = np.asarray(table["contact"], dtype=bool)
    contact_idx = np.flatnonzero(contact)
    manifest["schema_version"] = max(int(manifest.get("schema_version", 0)), 3)
    manifest["data_status"] = "generated"
    manifest["generated_data_policy"] = {
        "generator": "AFM06a/datasets/generate_real_dataset_06a.py",
        "time_span_s": [float(times[0]), float(times[-1])],
        "dt_s": float(times[1] - times[0]),
        "nsteps": int(nsteps),
        "num_points": int(times.size),
        "force_column": "Fts",
        "force_unit": "N",
        "rhs_force_usage": "(1/m)*Fts",
    }
    manifest["initial_time_s"] = float(times[0])
    manifest["end_time_s"] = float(times[-1])
    manifest["nsteps"] = int(nsteps)
    manifest["num_points"] = int(times.size)
    manifest["dt_s"] = float(times[1] - times[0])
    manifest["data_columns"] = {
        "t": "time [s]",
        "x1": "tip displacement [m]",
        "x2": "tip velocity [m/s]",
        "x2dot": "tip acceleration [m/s^2]",
        "contact": "1 if s <= a0 else 0",
        "s": "tip-sample separation [m]",
        "delta": "indentation max(a0-s,0) [m]",
        "Fts": "physical tip-sample interaction force [N]",
    }
    manifest["physical_constants"].update(
        {
            "k_over_m_s_minus2": float(p["k"] / p["m"]),
            "c_over_m_s_minus1": float(p["c"] / p["m"]),
            "Fd_over_m_m_s2": float(p["Fd"] / p["m"]),
        }
    )
    manifest["generated_contact_and_amplitude_summary"] = {
        "contact_points_full": int(contact_idx.size),
        "contact_fraction_full": float(np.mean(contact)),
        "first_contact_s": float(times[contact_idx[0]]) if contact_idx.size else None,
        "first_contact_ms": float(times[contact_idx[0]] * 1.0e3) if contact_idx.size else None,
        "x1_min_nm_full": float(np.min(x1) * 1.0e9),
        "x1_max_nm_full": float(np.max(x1) * 1.0e9),
        "x1_half_range_nm_full": float(0.5 * (np.max(x1) - np.min(x1)) * 1.0e9),
        "Fts_min_N": float(np.min(fts)),
        "Fts_max_N": float(np.max(fts)),
        "Fts_over_m_min_m_s2": float(np.min(fts / p["m"])),
        "Fts_over_m_max_m_s2": float(np.max(fts / p["m"])),
        "s_min_nm": float(np.min(table["s"]) * 1.0e9),
        "s_max_nm": float(np.max(table["s"]) * 1.0e9),
    }
    _write_json(data_dir / "afm06a_generation_manifest.json", manifest)


def _write_windows_placeholder(data_dir: Path, *, nsteps: int, num_points: int) -> None:
    _write_json(
        data_dir / "windows_manifest.json",
        {
            "schema_version": 3,
            "dataset": "AFM06a/datasets/e0.0_real/data",
            "status": "full time-span data generated; training/validation windows not selected yet",
            "full_time_span_s": [0.0, END_TIME_S],
            "dt_s": DT_S,
            "nsteps": int(nsteps),
            "num_points": int(num_points),
            "windows": [],
        },
    )


def main() -> None:
    data_dir = REPO_ROOT / "AFM06a" / "datasets" / ERROR_LEVEL / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(data_dir)
    _refresh_fixed_force_coefficients(manifest)
    p = _parameter_dict(manifest)
    nsteps = int(round(END_TIME_S / DT_S))
    tsteps = np.linspace(0.0, END_TIME_S, nsteps + 1, dtype=float)
    initial_state = manifest.get("initial_state", [0.0, 0.0])
    if not isinstance(initial_state, list) or len(initial_state) < 2:
        initial_state = [0.0, 0.0]
    u0 = [float(initial_state[0]), float(initial_state[1])]
    solution_table = generate_non_perturbed_training_set(
        lambda t, u, _unused: _rhs(t, u, p),
        u0,
        [],
        (0.0, END_TIME_S),
        tsteps,
        column_names=STATE_COLUMNS,
    )
    solution_matrix = table_to_matrix(solution_table, STATE_COLUMNS)
    ode_data = solution_matrix[:, 1:].T
    ode_data_std = np.zeros_like(ode_data)
    afm_table = _build_data_table(solution_table, p)
    afm_table_sd = _zeros_table_like(afm_table)

    np.savez(data_dir / "ode_data_afm_dmt_hard.npz", data=ode_data)
    np.savez(data_dir / "ode_data_std_afm_dmt_hard.npz", data=ode_data_std)
    save_table_npz(str(data_dir / "pert_df_afm_dmt_hard.npz"), afm_table, DATA_COLUMNS)
    save_table_npz(str(data_dir / "pert_df_sd_afm_dmt_hard.npz"), afm_table_sd, DATA_COLUMNS)
    np.savetxt(
        data_dir / "pert_df_afm_dmt_hard.csv",
        np.column_stack([afm_table[name] for name in DATA_COLUMNS]),
        delimiter=",",
        header=",".join(DATA_COLUMNS),
        comments="",
    )
    np.savetxt(
        data_dir / "pert_df_sd_afm_dmt_hard.csv",
        np.column_stack([afm_table_sd[name] for name in DATA_COLUMNS]),
        delimiter=",",
        header=",".join(DATA_COLUMNS),
        comments="",
    )
    _update_manifest(manifest, data_dir=data_dir, p=p, table=afm_table, nsteps=nsteps)
    _write_windows_placeholder(data_dir, nsteps=nsteps, num_points=tsteps.size)

    summary = manifest["generated_contact_and_amplitude_summary"]
    print(f"Generated e0.0_real data: {data_dir}")
    print(f"time=[0, {END_TIME_S:.6g}] s, dt={DT_S:.3g} s, points={tsteps.size}")
    print(f"contact_fraction={summary['contact_fraction_full']:.6g}, first_contact_ms={summary['first_contact_ms']}")
    print(f"Fts_N=[{summary['Fts_min_N']:.6e}, {summary['Fts_max_N']:.6e}]")


if __name__ == "__main__":
    main()
