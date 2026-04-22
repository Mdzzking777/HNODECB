"""AFM04 AFM DMT-KV dataset generator.

This module is the Python counterpart of `datasets/afm_dataset_generator.jl`,
adapted to the AFM04 shared physics definition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from AFM04.datasets.non_perturbed_dataset_generator import (
    generate_non_perturbed_training_set,
    save_table_npz,
    table_to_matrix,
)
from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_functions import ground_truth_rhs, f_ts_from_state
from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import (
    A0,
    DATA_NSTEPS,
    DIST,
    end_time_training,
    initial_time_training,
    original_parameters,
    original_u0,
)


AFM_SOLUTION_COLUMNS = ("t", "x1", "x2", "x3")
AFM_DATA_COLUMNS = ("t", "x1", "x2", "x3", "x2dot", "contact", "s", "x3dot", "delta_dot", "fts")


def _zeros_table_like(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in table.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.integer):
            out[key] = np.zeros_like(arr, dtype=int)
        else:
            out[key] = np.zeros_like(arr, dtype=float)
    return out


def generate_afm_dmt_kv_dataset(
    *,
    error_level: str = "e0.0",
    nsteps: int = DATA_NSTEPS,
    output_root: str | Path | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Generate the AFM04 AFM DMT-KV dataset and optionally save it."""

    tspan = (float(initial_time_training), float(end_time_training))
    tsteps = np.linspace(tspan[0], tspan[1], int(nsteps) + 1, dtype=float)

    solution_table = generate_non_perturbed_training_set(
        ground_truth_rhs,
        original_u0,
        original_parameters,
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
    x3 = solution_table["x3"]
    s = DIST + x1 - x3
    contact = (s <= A0).astype(int)

    k, wd, m, c, Fd, R, dist, Estar, eta_star, A, a0, beta, ks, cs = original_parameters
    x2dot = np.empty_like(x1)
    x3dot = np.empty_like(x1)
    delta_dot = np.empty_like(x1)
    fts = np.empty_like(x1)
    for i, t in enumerate(tvals):
        fts_i, x3dot_i, delta_dot_i, _ = f_ts_from_state(
            float(x1[i]),
            float(x2[i]),
            float(x3[i]),
            dist=float(dist),
            Estar=float(Estar),
            eta_star=float(eta_star),
            R=float(R),
            A=float(A),
            a0=float(a0),
            beta=float(beta),
            ks=float(ks),
            cs=float(cs),
        )
        x2dot[i] = (Fd * np.cos(wd * float(t)) - k * x1[i] - c * x2[i] + fts_i) / m
        x3dot[i] = x3dot_i
        delta_dot[i] = delta_dot_i
        fts[i] = fts_i

    afm_table = {
        "t": tvals.copy(),
        "x1": x1.copy(),
        "x2": x2.copy(),
        "x3": x3.copy(),
        "x2dot": x2dot,
        "contact": contact,
        "s": s,
        "x3dot": x3dot,
        "delta_dot": delta_dot,
        "fts": fts,
    }
    afm_table_sd = _zeros_table_like(afm_table)

    result = {
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

    if output_root is None:
        output_root = Path(__file__).resolve().parent
    else:
        output_root = Path(output_root)

    data_dir = output_root / error_level / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez(data_dir / "ode_data_afm_dmt_kv.npz", data=ode_data)
    np.savez(data_dir / "ode_data_std_afm_dmt_kv.npz", data=ode_data_std)
    save_table_npz(str(data_dir / "pert_df_afm_dmt_kv.npz"), afm_table, AFM_DATA_COLUMNS)
    save_table_npz(str(data_dir / "pert_df_sd_afm_dmt_kv.npz"), afm_table_sd, AFM_DATA_COLUMNS)

    np.savetxt(
        data_dir / "pert_df_afm_dmt_kv.csv",
        np.column_stack([afm_table[name] for name in AFM_DATA_COLUMNS]),
        delimiter=",",
        header=",".join(AFM_DATA_COLUMNS),
        comments="",
    )
    np.savetxt(
        data_dir / "pert_df_sd_afm_dmt_kv.csv",
        np.column_stack([afm_table_sd[name] for name in AFM_DATA_COLUMNS]),
        delimiter=",",
        header=",".join(AFM_DATA_COLUMNS),
        comments="",
    )

    return result


def main() -> None:
    result = generate_afm_dmt_kv_dataset()
    afm_table = result["afm_table"]
    contact_frac = 100.0 * float(np.mean(afm_table["contact"]))
    tsteps = result["tsteps"]
    dt = float(tsteps[1] - tsteps[0])
    print("Generated AFM04 dataset:")
    print(f"  points            : {len(afm_table['t'])}")
    print(f"  nsteps            : {len(afm_table['t']) - 1}")
    print(f"  dt [ns]           : {dt * 1e9:.3f}")
    print(f"  contact fraction  : {contact_frac:.2f}%")
    print(f"  s range [nm]      : [{afm_table['s'].min()*1e9:.2f}, {afm_table['s'].max()*1e9:.2f}]")
    print(f"  fts range [nN]    : [{afm_table['fts'].min()*1e9:.4f}, {afm_table['fts'].max()*1e9:.4f}]")


if __name__ == "__main__":
    main()
