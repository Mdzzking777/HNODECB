"""Dataset loading and preprocessing for AFM04 stage1pluslight."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from AFM04.datasets.afm_dataset_generator import generate_afm_dmt_kv_dataset
from AFM04.datasets.non_perturbed_dataset_generator import load_table_npz


def make_train_val_masks(n: int, val_stride: int, val_offset: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=int)
    val = idx[((idx + 1 - val_offset) % val_stride) == 0]
    train_mask = np.ones(n, dtype=bool)
    train_mask[val] = False
    train = idx[train_mask]
    return train, val


def _dataset_paths(dataset_root: Path, error_level: str) -> dict[str, Path]:
    data_dir = dataset_root / error_level / "data"
    return {
        "data_dir": data_dir,
        "ode_data": data_dir / "ode_data_afm_dmt_kv.npz",
        "pert_df": data_dir / "pert_df_afm_dmt_kv.npz",
    }


def ensure_dataset(dataset_root: Path, error_level: str, *, auto_generate: bool = False) -> dict[str, Path]:
    paths = _dataset_paths(dataset_root, error_level)
    if paths["ode_data"].is_file() and paths["pert_df"].is_file():
        return paths
    if not auto_generate:
        raise FileNotFoundError(
            f"AFM04 dataset files are missing under {paths['data_dir']}. "
            "Generate them first with AFM04.datasets.afm_dataset_generator."
        )
    generate_afm_dmt_kv_dataset(error_level=error_level, output_root=dataset_root, save_outputs=True)
    return paths


def load_dataset(dataset_root: Path, error_level: str, *, auto_generate: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    paths = ensure_dataset(dataset_root, error_level, auto_generate=auto_generate)
    with np.load(paths["ode_data"]) as data:
        ode_data = np.asarray(data["data"], dtype=float)
    pert_df = load_table_npz(str(paths["pert_df"]))
    return ode_data, pert_df


def truncate_to_first_contact(ode_data: np.ndarray, pert_df: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    contact = np.asarray(pert_df["contact"], dtype=int)
    idx = np.flatnonzero(contact == 1)
    if idx.size == 0:
        raise RuntimeError("No contact point found in AFM04 dataset.")
    start = int(idx[0])
    out_df = {name: np.asarray(values)[start:].copy() for name, values in pert_df.items()}
    out_ode = np.asarray(ode_data, dtype=float)[:, start:].copy()
    return out_ode, out_df


__all__ = [
    "ensure_dataset",
    "load_dataset",
    "make_train_val_masks",
    "truncate_to_first_contact",
]
