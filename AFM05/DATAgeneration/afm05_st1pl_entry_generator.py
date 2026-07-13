"""AFM05 experimental st1pl entry generator.

This file is adapted from ``AFM04/datasets/afm_dataset_generator.py``.

AFM04 generated a fully synthetic truth table.  AFM05 does not have true
sample-side quantities such as x3(t), x3dot(t), delta_dot(t), Fts(t), ks, or
cs.  The purpose of this script is only to build an AFM04-compatible *data
interface* for AFM05 stage1pluslight:

    ode_data: shape (3, N), rows [x1, x2, x3]
    pert_df : t, x1, x2, x3, x2dot, contact, s

x3 is a constructed initialization-side state, not an observed sample trajectory.
AFM05 does not emit x3dot, delta_dot, or fts truth-side fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM04.datasets.non_perturbed_dataset_generator import save_table_npz


AFM05_SOLUTION_COLUMNS = ("t", "x1", "x2", "x3")
AFM05_DATA_COLUMNS = ("t", "x1", "x2", "x3", "x2dot", "contact", "s")

DEFAULT_TRACE_NPZ = Path(__file__).with_name(
    "PS_cantilever_disp_vel_time_3_pixels_Z_63.77_a0_0.07108_file_scan04143.imp_.npz"
)
DEFAULT_IN_AIR_NPZ = Path(__file__).with_name("PS_cantilever_disp_in_air_file_scan04143.imp_.npz")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "datasets"

DEFAULT_INITIAL_INDEX = 114943


def pixel_tag_to_key(pixel_tag: str) -> str:
    tag = str(pixel_tag).strip()
    if tag == "":
        raise ValueError("pixel_tag must be nonempty")
    return tag if tag.startswith("pixel_") else f"pixel_{tag}"


def pixel_key_to_tag(pixel_key: str) -> str:
    key = str(pixel_key).strip()
    return key.removeprefix("pixel_")


def _load_npz_dict(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.files}


def _load_add_data_dict(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=True) as data:
        if "add_data_dict" not in data.files:
            return {}
        raw = data["add_data_dict"].item()
    if not isinstance(raw, Mapping):
        raise TypeError(f"add_data_dict in {path} must be a mapping")
    return dict(raw)


def _first_pixel_key(npz_payload: Mapping[str, Any]) -> str:
    keys = sorted(key for key in npz_payload if str(key).startswith("pixel_"))
    if not keys:
        raise KeyError("No pixel_* array was found in the AFM05 trace npz")
    return keys[0]


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _meta_float(meta: Mapping[str, Any], *keys: str, default: float | None = None) -> float:
    for key in keys:
        if key in meta:
            return float(np.asarray(meta[key]).reshape(-1)[0])
    if default is not None:
        return float(default)
    raise KeyError(f"Missing required metadata key; tried {keys}")


def _meta_int(meta: Mapping[str, Any], *keys: str, default: int | None = None) -> int:
    for key in keys:
        if key in meta:
            return int(np.asarray(meta[key]).reshape(-1)[0])
    if default is not None:
        return int(default)
    raise KeyError(f"Missing required metadata key; tried {keys}")


def _initial_index_for_pixel(meta: Mapping[str, Any], *, pixel_key: str, pixel_tag: str) -> int:
    tagged_keys = (
        f"{pixel_key}_AFM05_initial_condition_index",
        f"{pixel_key}_initial_condition_index",
        f"{pixel_key}_first_contact_index",
        f"AFM05_initial_condition_index_{pixel_tag}",
        f"initial_condition_index_{pixel_tag}",
        f"first_contact_index_{pixel_tag}",
    )
    for key in tagged_keys:
        if key in meta:
            return _meta_int(meta, key)

    generic_pixel = str(meta.get("AFM05_first_contact_pixel_key", "")).strip()
    if generic_pixel and generic_pixel != pixel_key:
        raise ValueError(
            "AFM05 generic initial-condition index belongs to a different pixel: "
            f"{generic_pixel!r} != {pixel_key!r}. Add a tagged initial index for {pixel_key}."
        )

    return _meta_int(
        meta,
        "AFM05_initial_condition_index",
        "AFM05_first_contact_index",
        "initial_condition_index",
        "first_contact_index",
        "first_contact_point",
        default=DEFAULT_INITIAL_INDEX,
    )


def _infer_unit_scale(values: np.ndarray, *, target: str) -> float:
    """Infer a conservative conversion to SI units.

    Current AFM05 file convention says the pixel columns are x1 [nm], x2 [nm/s],
    and t [s].  This helper keeps the generator robust if the source file has
    already been converted.
    """

    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise ValueError(f"Cannot infer units for empty/nonfinite {target}")
    amp = float(np.nanmax(np.abs(finite)))
    if target == "x1":
        # Nanometer-scale displacement stored as nm is O(10-100); in meters it
        # is O(1e-8).  Anything above micron scale is treated as nm.
        return 1.0e-9 if amp > 1.0e-5 else 1.0
    if target == "x2":
        # Stored nm/s can be O(1e6-1e7), while SI m/s is O(1e-3-1e-2).
        return 1.0e-9 if amp > 1.0e3 else 1.0
    raise ValueError(f"unsupported target: {target}")


def _zeros_table_like(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in table.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.integer):
            out[key] = np.zeros_like(arr, dtype=int)
        else:
            out[key] = np.zeros_like(arr, dtype=float)
    return out


def _observed_x2dot(t: np.ndarray, x2: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if t.ndim != 1 or x2.ndim != 1 or t.size != x2.size:
        raise ValueError("t and x2 must be one-dimensional arrays with equal length")
    if t.size < 3:
        raise ValueError("Need at least three samples to compute x2dot")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("AFM05 time vector must be strictly increasing")
    return np.gradient(x2, t, edge_order=2)


def _infer_x2dot_unit_scale(values: np.ndarray) -> float:
    """Infer conversion for x2dot to SI m/s^2.

    Current pixel_184_152 stores x2dot already in m/s^2.  If a future source
    stores nm/s^2, its amplitude will be about 1e9 times larger.
    """

    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise ValueError("Cannot infer units for empty/nonfinite x2dot")
    amp = float(np.nanmax(np.abs(finite)))
    return 1.0e-9 if amp > 1.0e8 else 1.0


def _parse_pixel_trace(
    pixel: np.ndarray,
    *,
    pixel_key: str,
    meta: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Parse AFM05 pixel trace into SI x1, x2, x2dot, and t arrays.

    Supported layouts:
    - current AFM05 layout: [x1_m, x2_m_per_s, x2dot_m_per_s2, t_s]
    - legacy layout: [x1, x2, t], where x1/x2 units are inferred and x2dot is
      computed from x2(t).
    """

    arr = np.asarray(pixel, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{pixel_key} must have at least three columns")

    column_order = str(meta.get(f"{pixel_key}_column_order", "")).strip()
    unit_system = str(meta.get(f"{pixel_key}_unit_system", "")).strip()

    if arr.shape[1] >= 4:
        x1_raw = np.asarray(arr[:, 0], dtype=float)
        x2_raw = np.asarray(arr[:, 1], dtype=float)
        x2dot_raw = np.asarray(arr[:, 2], dtype=float)
        t = np.asarray(arr[:, 3], dtype=float)

        x1_scale = _infer_unit_scale(x1_raw, target="x1")
        x2_scale = _infer_unit_scale(x2_raw, target="x2")
        x2dot_scale = _infer_x2dot_unit_scale(x2dot_raw)
        x1 = x1_raw * x1_scale
        x2 = x2_raw * x2_scale
        x2dot = x2dot_raw * x2dot_scale
        x2dot_source = "pixel_column_2"
        layout = "four_column_x1_x2_x2dot_t"
    else:
        x1_raw = np.asarray(arr[:, 0], dtype=float)
        x2_raw = np.asarray(arr[:, 1], dtype=float)
        t = np.asarray(arr[:, 2], dtype=float)

        x1_scale = _infer_unit_scale(x1_raw, target="x1")
        x2_scale = _infer_unit_scale(x2_raw, target="x2")
        x2dot_scale = float("nan")
        x1 = x1_raw * x1_scale
        x2 = x2_raw * x2_scale
        x2dot = _observed_x2dot(t, x2)
        x2dot_source = "np.gradient(x2, t, edge_order=2)"
        layout = "legacy_three_column_x1_x2_t"

    if t.ndim != 1 or x1.ndim != 1 or x2.ndim != 1 or x2dot.ndim != 1:
        raise ValueError("Parsed pixel arrays must be one-dimensional")
    if not (t.size == x1.size == x2.size == x2dot.size):
        raise ValueError("Parsed pixel arrays must have equal length")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("AFM05 time vector must be strictly increasing")
    if not np.all(np.isfinite(x1)) or not np.all(np.isfinite(x2)) or not np.all(np.isfinite(x2dot)) or not np.all(np.isfinite(t)):
        raise ValueError("Parsed pixel arrays must be finite")

    info = {
        "pixel_layout": layout,
        "source_column_order": column_order,
        "source_unit_system": unit_system,
        "x1_unit_scale_to_m": float(x1_scale),
        "x2_unit_scale_to_m_per_s": float(x2_scale),
        "x2dot_unit_scale_to_m_per_s2": float(x2dot_scale) if np.isfinite(x2dot_scale) else None,
        "x2dot_source": x2dot_source,
        "dt_median_s": float(np.median(np.diff(t))) if t.size > 1 else float("nan"),
    }
    return x1, x2, x2dot, t, info


def _candidate_force_keys(payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    force_names = (
        "eta1_in_air_F_actuation_t",
        "F_actuation_t",
    )
    time_names = (
        "F_actuation_time_s",
        "F_act_time_s",
        "eta1_in_air_time_s",
        "time_s",
        "t",
    )
    force_key = next((key for key in force_names if key in payload), None)
    time_key = next((key for key in time_names if key in payload), None)
    return force_key, time_key


def _extract_force_from_source(
    *,
    source_path: Path,
    target_t_full: np.ndarray,
    start_idx: int,
    target_len: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Try to load F_actuation(t) from one source file."""

    if not source_path.is_file():
        return np.full(target_len, np.nan, dtype=float), {
            "source": str(source_path),
            "status": "missing_source_file",
        }

    payload = _load_npz_dict(source_path)
    meta = _load_add_data_dict(source_path)
    combined: dict[str, Any] = {**payload, **meta}
    force_key, time_key = _candidate_force_keys(combined)
    if force_key is None:
        return np.full(target_len, np.nan, dtype=float), {
            "source": str(source_path),
            "status": "missing_force_key",
            "available_keys": sorted(str(k) for k in combined.keys()),
        }

    force = _as_float_array(combined[force_key])
    if force.size == target_t_full.size:
        return force[start_idx : start_idx + target_len].copy(), {
            "source": str(source_path),
            "status": "full_length_sliced",
            "force_key": force_key,
            "time_key": "",
        }
    if force.size == target_len:
        return force.copy(), {
            "source": str(source_path),
            "status": "already_window_length",
            "force_key": force_key,
            "time_key": "",
        }
    if time_key is not None:
        force_t = _as_float_array(combined[time_key])
        if force_t.size == force.size and force_t.size >= 2:
            target_t = target_t_full[start_idx : start_idx + target_len]
            return np.interp(target_t, force_t, force), {
                "source": str(source_path),
                "status": "interpolated_to_target_time",
                "force_key": force_key,
                "time_key": time_key,
            }

    return np.full(target_len, np.nan, dtype=float), {
        "source": str(source_path),
        "status": "length_mismatch_no_usable_time_key",
        "force_key": force_key,
        "time_key": time_key or "",
        "force_len": int(force.size),
        "target_full_len": int(target_t_full.size),
        "target_len": int(target_len),
    }


def _load_force_actuation(
    *,
    trace_npz: Path,
    in_air_npz: Path,
    target_t_full: np.ndarray,
    start_idx: int,
    target_len: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    for source in (trace_npz, in_air_npz):
        force, info = _extract_force_from_source(
            source_path=Path(source),
            target_t_full=target_t_full,
            start_idx=start_idx,
            target_len=target_len,
        )
        if np.any(np.isfinite(force)):
            return force, info
    info = {
        "source": "",
        "status": "F_actuation_not_found_in_trace_or_in_air_npz",
    }
    raise FileNotFoundError(
        "AFM05 requires F_actuation(t) for stage1pluslight entry generation; "
        f"status={info['status']}"
    )


def generate_afm05_st1pl_entry_dataset(
    *,
    trace_npz: str | Path = DEFAULT_TRACE_NPZ,
    in_air_npz: str | Path = DEFAULT_IN_AIR_NPZ,
    pixel_tag: str = "184_152",
    pixel_key: str | None = None,
    error_level: str = "e0.0",
    output_root: str | Path | None = None,
    save_outputs: bool = True,
    start_at_initial_condition: bool = True,
) -> dict[str, object]:
    """Build AFM04-compatible AFM05 st1pl entry data.

    The output filenames intentionally match AFM04's dataset names so AFM05 can
    later use a lightly adapted AFM04 data loader.
    """

    trace_path = Path(trace_npz)
    in_air_path = Path(in_air_npz)
    payload = _load_npz_dict(trace_path)
    meta = _load_add_data_dict(trace_path)

    if pixel_key is None:
        pixel_key = pixel_tag_to_key(pixel_tag)
    pixel_tag = pixel_key_to_tag(pixel_key)
    if pixel_key not in payload:
        available = sorted(key for key in payload if str(key).startswith("pixel_"))
        raise KeyError(f"Requested AFM05 pixel {pixel_key!r} is missing. Available pixel keys: {available}")
    pixel = np.asarray(payload[pixel_key], dtype=float)
    x1_full, x2_full, x2dot_full, t_full, pixel_info = _parse_pixel_trace(
        pixel,
        pixel_key=str(pixel_key),
        meta=meta,
    )

    start_idx = _initial_index_for_pixel(meta, pixel_key=str(pixel_key), pixel_tag=str(pixel_tag))
    start_idx = min(max(start_idx, 0), t_full.size - 1)
    if not start_at_initial_condition:
        start_idx = 0

    t = t_full[start_idx:].copy()
    x1 = x1_full[start_idx:].copy()
    x2 = x2_full[start_idx:].copy()
    x2dot = x2dot_full[start_idx:].copy()
    gain_force_key = f"F_ts_{pixel_tag}"
    gain_force_reference_full = None
    if gain_force_key in payload:
        gain_force_reference_raw = np.asarray(payload[gain_force_key], dtype=float).reshape(-1)
        if gain_force_reference_raw.size != t_full.size:
            raise ValueError(
                f"{gain_force_key} length must match the pixel trace length: "
                f"{gain_force_reference_raw.size} != {t_full.size}"
            )
        if not np.all(np.isfinite(gain_force_reference_raw)):
            raise ValueError(f"{gain_force_key} contains non-finite values")
        gain_force_reference_full = gain_force_reference_raw[start_idx:].copy()
    else:
        raise KeyError(f"AFM05 requires gain force reference key {gain_force_key!r} in {trace_path}")

    dist = _meta_float(meta, "Z", "dist")
    a0 = _meta_float(meta, "a0")
    x3_init = _meta_float(meta, "x3_init", default=a0)
    x3 = np.full_like(x1, x3_init, dtype=float)
    s = dist + x1 - x3
    contact = (s <= a0).astype(int)

    ode_data = np.vstack((x1, x2, x3))
    ode_data_std = np.zeros_like(ode_data)
    afm_table = {
        "t": t,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x2dot": x2dot,
        "contact": contact,
        "s": s,
    }
    afm_table_sd = _zeros_table_like(afm_table)

    f_actuation, f_actuation_info = _load_force_actuation(
        trace_npz=trace_path,
        in_air_npz=in_air_path,
        target_t_full=t_full,
        start_idx=start_idx,
        target_len=t.size,
    )

    k_eff = _meta_float(meta, "k", "k_eff")
    omega0 = _meta_float(meta, "omega0")
    q_factor = _meta_float(meta, "Q_factor", "Q")
    m_eff = k_eff / (omega0**2)
    c_eff = k_eff / (q_factor * omega0)
    metadata = {
        "dataset": "AFM05 experimental AFM04-compatible st1pl entry",
        "source_trace_npz": str(trace_path),
        "source_in_air_npz": str(in_air_path),
        "pixel_tag": str(pixel_tag),
        "pixel_key": str(pixel_key),
        "start_idx_source": int(start_idx),
        "start_at_initial_condition": bool(start_at_initial_condition),
        "time_start_s": float(t[0]),
        "time_stop_s": float(t[-1]),
        "points": int(t.size),
        **pixel_info,
        "dist_Z_m": float(dist),
        "a0_m": float(a0),
        "x3_init_m": float(x3_init),
        "AFM05_initial_condition": [float(x1[0]), float(x2[0]), float(x3_init)],
        "k_eff": float(k_eff),
        "omega0": float(omega0),
        "m_eff": float(m_eff),
        "c_eff": float(c_eff),
        "Q_factor": float(q_factor),
        "F_actuation_info": f_actuation_info,
        "F_actuation_role": (
            "Global experimental actuation setting reconstructed from the in-air trajectory; "
            "not a pixel-tagged quantity."
        ),
        "F_actuation_alignment": (
            "The global F_actuation(t) trajectory is sliced or interpolated onto the selected "
            "stage1pluslight time grid so rollout inputs share the same timestamps."
        ),
        "gain_force_reference_key": gain_force_key,
        "gain_force_reference_available": bool(gain_force_reference_full is not None),
        "gain_force_reference_role": (
            "AFM05 reconstructed tip-sample force for the selected pixel, used for KAN gain amplitude initialization; "
            "not synthetic oracle truth."
        ),
        "oracle_truth_available": False,
        "constructed_state_fields_not_observed_truth": ["x3"],
        "strict_rule": (
            "AFM05 has no true x3(t), x3dot(t), delta_dot(t), Fts(t), ks, or cs. "
            "Only x1, x2, x2dot, t, initial condition, and derived force references are available."
        ),
    }

    result = {
        "metadata": metadata,
        "tspan": (float(t[0]), float(t[-1])),
        "tsteps": t,
        "solution_table": {"t": t, "x1": x1, "x2": x2, "x3": x3},
        "ode_data": ode_data,
        "ode_data_std": ode_data_std,
        "afm_table": afm_table,
        "afm_table_sd": afm_table_sd,
        "F_actuation": f_actuation,
        "gain_force_reference": gain_force_reference_full,
    }

    if not save_outputs:
        return result

    root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    data_dir = root / error_level / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez(data_dir / "ode_data_afm_dmt_kv.npz", data=ode_data)
    np.savez(data_dir / "ode_data_std_afm_dmt_kv.npz", data=ode_data_std)
    save_table_npz(str(data_dir / "pert_df_afm_dmt_kv.npz"), afm_table, AFM05_DATA_COLUMNS)
    save_table_npz(str(data_dir / "pert_df_sd_afm_dmt_kv.npz"), afm_table_sd, AFM05_DATA_COLUMNS)
    np.savez(data_dir / "afm05_F_actuation.npz", t=t, F_actuation=f_actuation)
    np.savez(data_dir / f"afm05_F_ts_{pixel_tag}.npz", t=t, **{gain_force_key: gain_force_reference_full})

    np.savetxt(
        data_dir / "pert_df_afm_dmt_kv.csv",
        np.column_stack([afm_table[name] for name in AFM05_DATA_COLUMNS]),
        delimiter=",",
        header=",".join(AFM05_DATA_COLUMNS),
        comments="",
    )
    np.savetxt(
        data_dir / "pert_df_sd_afm_dmt_kv.csv",
        np.column_stack([afm_table_sd[name] for name in AFM05_DATA_COLUMNS]),
        delimiter=",",
        header=",".join(AFM05_DATA_COLUMNS),
        comments="",
    )
    with (data_dir / "afm05_st1pl_entry_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
    with (data_dir / "README_AFM05_st1pl_entry.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            "AFM05 st1pl entry data\n"
            "======================\n\n"
            "This directory uses AFM04-compatible filenames with AFM05-specific fields.\n"
            "x3 is a constructed initial-condition-side state, not observed x3(t).\n"
            "AFM05 does not write x3dot, delta_dot, or fts truth-side columns.\n"
            "See afm05_st1pl_entry_metadata.json.\n"
        )

    return result


def main() -> None:
    result = generate_afm05_st1pl_entry_dataset()
    table = result["afm_table"]
    meta = result["metadata"]
    contact_frac = 100.0 * float(np.mean(table["contact"]))
    tsteps = np.asarray(result["tsteps"], dtype=float)
    dt = float(np.median(np.diff(tsteps))) if tsteps.size > 1 else float("nan")
    print("Generated AFM05 st1pl entry dataset:")
    print(f"  points                 : {len(table['t'])}")
    print(f"  pixel tag              : {meta['pixel_tag']}")
    print(f"  pixel key              : {meta['pixel_key']}")
    print(f"  source start idx        : {meta['start_idx_source']}")
    print(f"  dt median [ns]          : {dt * 1e9:.3f}")
    print(f"  contact fraction        : {contact_frac:.2f}%")
    print(f"  x1 range [nm]           : [{np.nanmin(table['x1'])*1e9:.6f}, {np.nanmax(table['x1'])*1e9:.6f}]")
    print(f"  x2 range [m/s]          : [{np.nanmin(table['x2']):.6e}, {np.nanmax(table['x2']):.6e}]")
    print(f"  x2dot range [m/s^2]     : [{np.nanmin(table['x2dot']):.6e}, {np.nanmax(table['x2dot']):.6e}]")
    print(f"  x3 constructed init [nm]: {float(meta['x3_init_m']) * 1e9:.6f}")
    print(f"  F_actuation status      : {meta['F_actuation_info']['status']}")


if __name__ == "__main__":
    main()
