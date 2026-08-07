"""Generate the AFM06a fixed hard-silicon DMT dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from AFM06a.datasets.non_perturbed_dataset_generator import (
    generate_non_perturbed_training_set,
    save_table_npz,
    table_to_matrix,
)
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_functions import (
    interaction_from_tip_state,
    ground_truth_rhs,
)
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (
    CONTACT_SWITCH_POLICY,
    AFM06aHardSampleInputs,
    LEGACY_PARAMETER_NAMES,
    PARAMETER_NAMES,
)


AFM_SOLUTION_COLUMNS = ("t", "x1", "x2")
AFM_DATA_COLUMNS = (
    "t",
    "x1",
    "x2",
    "x2dot",
    "contact",
    "s",
    "delta",
    "bar_fts",
    "fts_N",
)


def _zeros_table_like(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in table.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.integer):
            out[key] = np.zeros_like(arr, dtype=int)
        else:
            out[key] = np.zeros_like(arr, dtype=float)
    return out


def generate_afm_dmt_hard_dataset(
    *,
    settings: AFM06aHardSampleInputs,
    error_level: str = "e0.0",
    nsteps: int | None = None,
    output_root: str | Path | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Generate the AFM06a fixed hard-sample trajectory and optionally save it."""

    if not isinstance(settings, AFM06aHardSampleInputs):
        raise TypeError("settings must be an AFM06aHardSampleInputs instance")
    settings.validate()

    steps = int(settings.data_nsteps) if nsteps is None else int(nsteps)
    if isinstance(nsteps, bool) or (nsteps is not None and steps != nsteps):
        raise ValueError("nsteps must be an integer")
    if steps < 1:
        raise ValueError("nsteps must be at least 1")

    tspan = (float(settings.initial_time), float(settings.end_time))
    tsteps = np.linspace(tspan[0], tspan[1], steps + 1, dtype=float)
    parameter_vector = settings.parameter_vector
    initial_state = settings.initial_state
    solution_table = generate_non_perturbed_training_set(
        ground_truth_rhs,
        initial_state,
        parameter_vector,
        tspan,
        tsteps,
        column_names=AFM_SOLUTION_COLUMNS,
    )

    solution_matrix = table_to_matrix(solution_table, AFM_SOLUTION_COLUMNS)
    ode_data = solution_matrix[:, 1:].T
    ode_data_std = np.zeros_like(ode_data)

    tvals = solution_table["t"]
    x1 = solution_table["x1"]
    x2 = solution_table["x2"]
    s = float(settings.dist) + x1
    delta = np.maximum(float(settings.a0) - s, 0.0)
    contact = (s <= float(settings.a0)).astype(int)

    k_n_m, omega0, mass_kg, c_n_s_m, fd_n, ca, ch, dist, a0, beta = parameter_vector
    x2dot = np.empty_like(x1)
    bar_fts = np.empty_like(x1)
    fts_n = np.empty_like(x1)
    actuation_acceleration = fd_n / mass_kg
    for i, t in enumerate(tvals):
        bar_fts_i, s_i, _ = interaction_from_tip_state(
            float(x1[i]),
            dist=float(dist),
            ca=float(ca),
            ch=float(ch),
            a0=float(a0),
            beta=float(beta),
        )
        x2dot[i] = (
            (fd_n * np.sin(omega0 * float(t)) - c_n_s_m * x2[i] - k_n_m * x1[i])
            / mass_kg
            + bar_fts_i
        )
        bar_fts[i] = bar_fts_i
        fts_n[i] = mass_kg * bar_fts_i

    afm_table = {
        "t": tvals.copy(),
        "x1": x1.copy(),
        "x2": x2.copy(),
        "x2dot": x2dot,
        "contact": contact,
        "s": s,
        "delta": delta,
        "bar_fts": bar_fts,
        "fts_N": fts_n,
    }
    afm_table_sd = _zeros_table_like(afm_table)

    result: dict[str, object] = {
        "tspan": tspan,
        "tsteps": tsteps,
        "solution_table": solution_table,
        "ode_data": ode_data,
        "ode_data_std": ode_data_std,
        "afm_table": afm_table,
        "afm_table_sd": afm_table_sd,
    }
    if not save_outputs:
        return result

    output_root = Path(__file__).resolve().parent if output_root is None else Path(output_root)
    data_dir = output_root / error_level / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez(data_dir / "ode_data_afm_dmt_hard.npz", data=ode_data)
    np.savez(data_dir / "ode_data_std_afm_dmt_hard.npz", data=ode_data_std)
    save_table_npz(str(data_dir / "pert_df_afm_dmt_hard.npz"), afm_table, AFM_DATA_COLUMNS)
    save_table_npz(str(data_dir / "pert_df_sd_afm_dmt_hard.npz"), afm_table_sd, AFM_DATA_COLUMNS)
    np.savetxt(
        data_dir / "pert_df_afm_dmt_hard.csv",
        np.column_stack([afm_table[name] for name in AFM_DATA_COLUMNS]),
        delimiter=",",
        header=",".join(AFM_DATA_COLUMNS),
        comments="",
    )
    np.savetxt(
        data_dir / "pert_df_sd_afm_dmt_hard.csv",
        np.column_stack([afm_table_sd[name] for name in AFM_DATA_COLUMNS]),
        delimiter=",",
        header=",".join(AFM_DATA_COLUMNS),
        comments="",
    )
    generation_manifest = {
        "schema_version": 1,
        "system": "AFM06a_hard_sample",
        "governing_equation": (
            "m*x2dot = Fd*sin(omega0*t) - c*x2 - k*x1 + Fts; "
            "bar_fts = Fts/m"
        ),
        "contact_switch_policy": CONTACT_SWITCH_POLICY,
        "contact_criterion": "s <= a0",
        "noncontact_branch": {
            "damping": "c",
            "adhesion": "C_A / s^2",
            "hertz": "0",
        },
        "contact_branch": {
            "damping": "c",
            "adhesion": "C_A / a0^2",
            "hertz": "C_H * (a0 - s)^(3/2)",
        },
        "initial_time_s": float(settings.initial_time),
        "end_time_s": float(settings.end_time),
        "nsteps": steps,
        "num_points": steps + 1,
        "dt_s": float((settings.end_time - settings.initial_time) / steps),
        "initial_state": [float(value) for value in settings.initial_state],
        "parameter_names": list(PARAMETER_NAMES),
        "parameter_vector": [float(value) for value in parameter_vector],
        "legacy_parameter_names": list(LEGACY_PARAMETER_NAMES),
        "legacy_parameter_vector": [float(value) for value in settings.legacy_parameter_vector],
        "physical_constants": {
            "k_N_m": float(k_n_m),
            "omega0_rad_s": float(omega0),
            "m_kg": float(mass_kg),
            "c_N_s_m": float(c_n_s_m),
            "Fd_N": float(fd_n),
            "Fd_over_m_m_s2": float(actuation_acceleration),
            "CA_m3_s2": float(ca),
            "CH_m_minus_half_s2": float(ch),
            "Q": float(mass_kg * omega0 / c_n_s_m) if c_n_s_m > 0.0 else float("inf"),
            "dist_m": float(dist),
            "a0_m": float(a0),
        },
    }
    (data_dir / "afm06a_generation_manifest.json").write_text(
        json.dumps(generation_manifest, indent=2),
        encoding="utf-8",
    )
    result["data_dir"] = data_dir
    result["generation_manifest"] = generation_manifest
    return result


def main() -> None:
    settings = AFM06aHardSampleInputs()
    result = generate_afm_dmt_hard_dataset(
        settings=settings,
        error_level="e0.0",
        save_outputs=True,
    )
    data_dir = Path(result["data_dir"])
    table = result["afm_table"]
    contact_count = int(np.count_nonzero(table["contact"]))
    print(f"AFM06a dataset generated: {data_dir}")
    print(
        f"policy={CONTACT_SWITCH_POLICY} points={table['t'].size} "
        f"contact_points={contact_count} dt={(table['t'][1] - table['t'][0]):.12g} s"
    )


if __name__ == "__main__":
    main()
