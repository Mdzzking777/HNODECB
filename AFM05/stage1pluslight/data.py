"""Dataset loading and preprocessing for AFM05 stage1pluslight."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from AFM05.DATAgeneration.afm05_st1pl_entry_generator import generate_afm05_st1pl_entry_dataset


def make_train_val_masks(n: int, val_stride: int, val_offset: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=int)
    val = idx[((idx + 1 - val_offset) % val_stride) == 0]
    train_mask = np.ones(n, dtype=bool)
    train_mask[val] = False
    train = idx[train_mask]
    return train, val


def load_table_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a dict-of-columns table saved by the AFM05 entry generator."""

    with np.load(path, allow_pickle=True) as data:
        columns = [str(x) for x in data["__columns__"].tolist()]
        return {name: np.asarray(data[name]) for name in columns}


def _dataset_paths(dataset_root: Path, error_level: str, *, pixel_tag: str = "184_152") -> dict[str, Path]:
    data_dir = dataset_root / error_level / "data"
    tag = str(pixel_tag).strip()
    return {
        "data_dir": data_dir,
        "ode_data": data_dir / "ode_data_afm_dmt_kv.npz",
        "pert_df": data_dir / "pert_df_afm_dmt_kv.npz",
        "f_actuation": data_dir / "afm05_F_actuation.npz",
        "gain_force_reference": data_dir / f"afm05_F_ts_{tag}.npz",
        "metadata": data_dir / "afm05_st1pl_entry_metadata.json",
    }


def _metadata_matches_pixel_tag(path: Path, *, pixel_tag: str) -> bool:
    if not path.is_file():
        return False
    import json

    with path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return str(metadata.get("pixel_tag", "")).strip() == str(pixel_tag).strip()


def ensure_dataset(
    dataset_root: Path,
    error_level: str,
    *,
    pixel_tag: str = "184_152",
    auto_generate: bool = False,
) -> dict[str, Path]:
    paths = _dataset_paths(dataset_root, error_level, pixel_tag=pixel_tag)
    if (
        paths["ode_data"].is_file()
        and paths["pert_df"].is_file()
        and paths["f_actuation"].is_file()
        and paths["gain_force_reference"].is_file()
        and _metadata_matches_pixel_tag(paths["metadata"], pixel_tag=pixel_tag)
    ):
        return paths
    if not auto_generate:
        raise FileNotFoundError(
            f"AFM05 st1pl entry dataset files are missing under {paths['data_dir']}. "
            f"Required pixel_tag={pixel_tag!r}. Generate them first with "
            "AFM05.DATAgeneration.afm05_st1pl_entry_generator."
        )
    generate_afm05_st1pl_entry_dataset(
        pixel_tag=pixel_tag,
        error_level=error_level,
        output_root=dataset_root,
        save_outputs=True,
    )
    return paths


def _require_finite(name: str, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
        raise ValueError(f"AFM05 required observable field {name!r} contains {bad} non-finite values.")


def validate_observable_fields(ode_data: np.ndarray, pert_df: dict[str, np.ndarray]) -> None:
    """Validate AFM05 observable fields."""

    if np.asarray(ode_data).ndim != 2 or np.asarray(ode_data).shape[0] != 3:
        raise ValueError("AFM05 ode_data must have shape (3, N), ordered as [x1, x2, x3_compat].")
    _require_finite("ode_data[0]=x1", np.asarray(ode_data)[0])
    _require_finite("ode_data[1]=x2", np.asarray(ode_data)[1])
    _require_finite("ode_data[2]=x3_compat", np.asarray(ode_data)[2])
    for name in ("t", "x1", "x2", "x2dot"):
        if name not in pert_df:
            raise KeyError(f"AFM05 pert_df is missing required observable field {name!r}.")
        _require_finite(name, pert_df[name])
    t = np.asarray(pert_df["t"], dtype=float)
    if t.ndim != 1 or t.size != np.asarray(ode_data).shape[1]:
        raise ValueError("AFM05 pert_df['t'] must be one-dimensional and match ode_data length.")
    if t.size > 1 and not np.all(np.diff(t) > 0.0):
        raise ValueError("AFM05 pert_df['t'] must be strictly increasing.")


def load_dataset(
    dataset_root: Path,
    error_level: str,
    *,
    pixel_tag: str = "184_152",
    auto_generate: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    paths = ensure_dataset(dataset_root, error_level, pixel_tag=pixel_tag, auto_generate=auto_generate)
    with np.load(paths["ode_data"]) as data:
        ode_data = np.asarray(data["data"], dtype=float)
    pert_df = load_table_npz(str(paths["pert_df"]))
    validate_observable_fields(ode_data, pert_df)
    return ode_data, pert_df


def initial_condition_aligned_dataset(
    ode_data: np.ndarray,
    pert_df: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return AFM05 data unchanged.

    The AFM05 entry generator already slices the experimental trajectory at the
    selected initial-condition index. This function intentionally does not search
    for synthetic first contact.
    """

    return np.asarray(ode_data, dtype=float), {name: np.asarray(values).copy() for name, values in pert_df.items()}


__all__ = [
    "ensure_dataset",
    "initial_condition_aligned_dataset",
    "load_dataset",
    "make_train_val_masks",
    "validate_observable_fields",
]
