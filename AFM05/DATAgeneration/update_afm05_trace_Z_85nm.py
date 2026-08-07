"""Update the AFM05 experimental trace metadata from Z=63.771 nm to 85.0 nm."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
OLD_TRACE = DATA_DIR / (
    "PS_cantilever_disp_vel_time_3_pixels_Z_63.77_a0_0.07108_file_scan04143.imp_.npz"
)
NEW_TRACE = DATA_DIR / (
    "PS_cantilever_disp_vel_time_3_pixels_Z_85.0_a0_0.07108_file_scan04143.imp_.npz"
)
NEW_Z_M = 85.0e-9
INITIAL_INDEX = 151372
TRAINING_WINDOW_SPAN_S = 25.152e-6
WINDOW_SAMPLE_STRIDE = 16
VAL_STRIDE = 5
VAL_OFFSET = 2
DATASET_SIGNATURE = "afm05_z85_t0_605488us_downward_negative_v1"
PIXEL_KEY = "pixel_184_152"
PIXEL_TAG = "184_152"


def main() -> None:
    source = NEW_TRACE if NEW_TRACE.is_file() else OLD_TRACE
    if not source.is_file():
        raise FileNotFoundError(f"AFM05 source trace is missing: {OLD_TRACE} or {NEW_TRACE}")

    with np.load(source, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}

    metadata = dict(payload["add_data_dict"].item())
    trace = np.asarray(payload[PIXEL_KEY], dtype=np.float64)
    x1_m = trace[:, 0]
    time_s = trace[:, 3]
    a0_m = float(metadata["a0"])
    initial_index = int(INITIAL_INDEX)
    x1_init_m = float(x1_m[initial_index])
    x2_init_m_s = float(trace[initial_index, 1])
    x3_init_m = float(NEW_Z_M + x1_init_m - a0_m)
    initial_time_s = float(time_s[initial_index])

    metadata["Z"] = float(NEW_Z_M)
    metadata["AFM05_dataset_signature"] = DATASET_SIGNATURE
    metadata["AFM05_coordinate_convention"] = "tip displacement toward the sample is negative"
    metadata["AFM05_initial_condition_index"] = int(initial_index)
    metadata["AFM05_initial_time_s"] = initial_time_s
    metadata["AFM05_first_contact_index"] = int(initial_index)
    metadata["AFM05_first_contact_time_s"] = initial_time_s
    metadata["AFM05_first_contact_time_unit"] = "s"
    metadata["AFM05_first_contact_pixel_key"] = PIXEL_KEY
    metadata["AFM05_x1_init"] = x1_init_m
    metadata["AFM05_x2_init"] = x2_init_m_s
    metadata["AFM05_x3_init"] = x3_init_m
    metadata["x3_init"] = x3_init_m
    metadata["AFM05_x3_init_definition"] = "x3(t0) = Z + x1(t0) - a0"
    metadata["x3_init_definition"] = "x3(t0) = Z + x1(t0) - a0"
    metadata["AFM05_initial_condition"] = np.asarray(
        [x1_init_m, x2_init_m_s, x3_init_m], dtype=np.float64
    )
    metadata["AFM05_initial_condition_order"] = (
        "[x1_init, x2_init, x3_init] in SI units; "
        "x3_init = Z + x1_init - a0"
    )
    metadata["AFM05_contact_definition"] = "s = Z + x1 - x3; contact iff s <= a0"
    metadata["AFM05_first_contact_point_definition"] = (
        "Operational AFM05 initial/contact-side point; downward displacement is negative; "
        f"{PIXEL_KEY} index {initial_index} at t={initial_time_s:.9f} s."
    )
    metadata["AFM05_Z_update_note"] = (
        "Z corrected to 85.0 nm. The operational AFM05 initial time/index is retained; "
        "x3_init is recomputed so that s(t0) = a0."
    )
    for key in tuple(metadata):
        if key.startswith(f"pixel_{PIXEL_TAG}_x1_equals_"):
            del metadata[key]

    force_key = f"F_ts_{PIXEL_TAG}"
    force = np.asarray(payload[force_key], dtype=np.float64).reshape(-1)
    stop_time_s = initial_time_s + TRAINING_WINDOW_SPAN_S
    stop_index = int(np.searchsorted(time_s, stop_time_s, side="right") - 1)
    sampled_source_indices = np.arange(initial_index, stop_index + 1, WINDOW_SAMPLE_STRIDE, dtype=int)
    local_indices = np.arange(sampled_source_indices.size, dtype=int)
    val_mask = ((local_indices + 1 - VAL_OFFSET) % VAL_STRIDE) == 0
    train_values = force[sampled_source_indices[~val_mask]]
    val_values = force[sampled_source_indices[val_mask]]
    full_values = force[sampled_source_indices]
    prefix = f"{force_key}_gain_amp"
    metadata[f"{prefix}_window_mode"] = "window_05_initial"
    metadata[f"{prefix}_arch_window_s"] = float(TRAINING_WINDOW_SPAN_S)
    metadata[f"{prefix}_val_stride"] = int(VAL_STRIDE)
    metadata[f"{prefix}_val_offset"] = int(VAL_OFFSET)
    metadata[f"{prefix}_source_start_idx"] = int(initial_index)
    metadata[f"{prefix}_window_source_start_idx"] = int(sampled_source_indices[0])
    metadata[f"{prefix}_window_source_stop_idx"] = int(sampled_source_indices[-1])
    metadata[f"{prefix}_window_sample_stride"] = int(WINDOW_SAMPLE_STRIDE)
    metadata[f"{prefix}_window_length"] = int(sampled_source_indices.size)
    metadata[f"{prefix}_train_count"] = int(train_values.size)
    metadata[f"{prefix}_val_count"] = int(val_values.size)
    metadata[f"{force_key}_q95_abs_window_05_initial_full"] = float(np.quantile(np.abs(full_values), 0.95))
    metadata[f"{force_key}_q95_abs_window_05_initial_train"] = float(np.quantile(np.abs(train_values), 0.95))
    metadata[f"{force_key}_q95_abs_window_05_initial_val"] = float(np.quantile(np.abs(val_values), 0.95))

    payload["add_data_dict"] = np.asarray(metadata, dtype=object)
    temporary = NEW_TRACE.with_suffix(NEW_TRACE.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, NEW_TRACE)
    if OLD_TRACE != NEW_TRACE and OLD_TRACE.is_file():
        OLD_TRACE.unlink()

    print(f"Updated trace: {NEW_TRACE}")
    print(f"Z [nm]: {float(metadata['Z']) * 1.0e9:.6f}")
    print(f"t0 [us]: {float(time_s[initial_index]) * 1.0e6:.6f}")
    print(f"x3_init [nm]: {x3_init_m * 1.0e9:.9f}")
    print(f"s(t0) - a0 [m]: {(NEW_Z_M + x1_init_m - x3_init_m - a0_m):.3e}")


if __name__ == "__main__":
    main()
