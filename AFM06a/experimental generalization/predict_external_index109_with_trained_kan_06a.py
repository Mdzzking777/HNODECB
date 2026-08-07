"""Pointwise AFM06a trained-KAN force prediction on external silicon x1 raw data.

This script does not run a rollout and does not train anything. It feeds the
external raw tip displacement x1 into one archived AFM06a stage2light KAN using
the selected input-normalizer policy, then saves the full-time-span force
prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM06a.stage1pluslight.checkpoint import load_checkpoint  # noqa: E402
from AFM06a.stage2light.config import default_config  # noqa: E402
from AFM06a.stage2light.data import load_stage1_endpoint  # noqa: E402
from AFM06a.stage2light.kan_backend import restore_exact_stage1_model  # noqa: E402


DEFAULT_ARCHIVE_DIR = (
    REPO_ROOT / "AFM06a" / "archive" / "st2l" / "rank1_20260801_131123"
)
DEFAULT_LOCAL_MAT = (
    REPO_ROOT
    / "external data"
    / "Silicon exp data"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)
DEFAULT_RAW_MAT = (
    Path(r"E:\External data SI_SI hard sample raw")
    / "OneDrive_2_2026-7-26"
    / "dataAFM_Si"
    / "rawtimedata_sweep_06_11_18_si_si_100nm_0p016V.mat"
)
DEFAULT_OUTPUT_STEM = "external_index109_raw_x1_trained_kan_fts_prediction_full"

INDEX = 109
DT_S = 1.0 / 250.0e6
K_N_PER_M = 22.68
Q_FACTOR = 428.0
DRIVE_FREQUENCY_HZ = 164520.0
OMEGA0_RAD_S = 2.0 * np.pi * DRIVE_FREQUENCY_HZ
M_KG = K_N_PER_M / (OMEGA0_RAD_S**2)
PERIOD_S = 1.0 / DRIVE_FREQUENCY_HZ


def _required_dict(path: Path) -> dict[str, Any]:
    payload = load_checkpoint(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a dictionary checkpoint payload: {path}")
    return payload


def _resolve_archive_files(archive_dir: Path) -> tuple[Path, Path]:
    archive = archive_dir.expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"AFM06a archive is missing: {archive}")
    result_files = sorted((archive / "result").glob("afm_param_stage2light_06a_rank*.pkl"))
    stage1_files = sorted(
        (archive / "conditional dependency").glob("afm_param_stage1pluslight_06a.pkl")
    )
    if len(result_files) != 1:
        raise RuntimeError(
            f"Expected one stage2light result in {archive / 'result'}, found {len(result_files)}"
        )
    if len(stage1_files) != 1:
        raise RuntimeError(
            "Expected one st1pl dependency in "
            f"{archive / 'conditional dependency'}, found {len(stage1_files)}"
        )
    return result_files[0], stage1_files[0]


def _restore_stage2_model(archive_dir: Path):
    stage2_result, stage1_result = _resolve_archive_files(archive_dir)
    payload = _required_dict(stage2_result)
    if not bool(payload.get("complete")):
        raise RuntimeError(f"Stage2light result is not complete: {stage2_result}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Stage2light result has no model_state_dict: {stage2_result}")

    base = default_config(REPO_ROOT)
    saved = payload.get("config")
    updates: dict[str, Any] = {
        "input_rank": int(payload.get("input_rank", base.input_rank)),
        "device": "cpu",
        "stage1_result_path": stage1_result,
    }
    if isinstance(saved, dict):
        if saved.get("pykan_root") is not None:
            updates["pykan_root"] = Path(saved["pykan_root"])
        for key in (
            "gain_enabled",
            "gain_learnable",
            "soft_mask_enabled",
            "soft_mask_trainable",
        ):
            if saved.get(key) is not None:
                updates[key] = bool(saved[key])
    config = replace(base, **updates)
    endpoint = load_stage1_endpoint(config, enforce_current_contract=False)
    model = restore_exact_stage1_model(config, endpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, endpoint, payload, stage2_result, stage1_result


def _load_external_x1_raw(mat_path: Path | None) -> tuple[np.ndarray, np.ndarray, Path]:
    candidates = []
    if mat_path is not None:
        candidates.append(mat_path.expanduser())
    candidates.extend([DEFAULT_LOCAL_MAT, DEFAULT_RAW_MAT])
    errors: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            errors.append(f"{candidate} does not exist")
            continue
        try:
            data = sio.loadmat(
                str(candidate),
                variable_names=["displacement_fwd_sweep"],
                squeeze_me=True,
                struct_as_record=False,
            )
            x1 = np.asarray(data["displacement_fwd_sweep"][INDEX], dtype=np.float64).ravel()
            if x1.size < 2 or not np.all(np.isfinite(x1)):
                raise ValueError(f"invalid x1 array, shape={x1.shape}")
            time_s = np.arange(x1.size, dtype=np.float64) * DT_S
            return time_s, x1, candidate
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("Could not load external x1 raw:\n" + "\n".join(errors))


def _record_training_normalizer(
    model: torch.nn.Module,
) -> dict[str, list[float] | str]:
    trained_mean = np.asarray(model.state_mean.detach().cpu(), dtype=float).reshape(-1).copy()
    trained_scale = np.asarray(model.state_scale.detach().cpu(), dtype=float).reshape(-1).copy()
    if (
        trained_mean.size != 1
        or trained_scale.size != 1
        or not np.all(np.isfinite(trained_mean))
        or not np.all(np.isfinite(trained_scale))
        or np.any(trained_scale <= 0.0)
    ):
        raise RuntimeError(
            "invalid archived training normalizer: "
            f"mean={trained_mean.tolist()}, scale={trained_scale.tolist()}"
        )

    return {
        "normalizer_source": "archived_trained_kan_training_mean_std",
        "trained_state_mean": trained_mean.tolist(),
        "trained_state_scale": trained_scale.tolist(),
        "applied_state_mean": trained_mean.tolist(),
        "applied_state_scale": trained_scale.tolist(),
    }


def _apply_external_x1_normalizer(
    model: torch.nn.Module,
    x1_raw_m: np.ndarray,
) -> dict[str, list[float] | str]:
    x1 = np.asarray(x1_raw_m, dtype=np.float64).reshape(-1)
    mean = float(np.mean(x1))
    scale = float(np.std(x1))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"invalid external x1 normalizer: mean={mean}, scale={scale}")

    trained_mean = np.asarray(model.state_mean.detach().cpu(), dtype=float).reshape(-1).copy()
    trained_scale = np.asarray(model.state_scale.detach().cpu(), dtype=float).reshape(-1).copy()
    with torch.no_grad():
        model.state_mean.copy_(
            torch.as_tensor([mean], dtype=model.state_mean.dtype, device=model.state_mean.device)
        )
        model.state_scale.copy_(
            torch.as_tensor([scale], dtype=model.state_scale.dtype, device=model.state_scale.device)
        )

    return {
        "normalizer_source": "external_x1_raw_self_mean_std",
        "trained_state_mean": trained_mean.tolist(),
        "trained_state_scale": trained_scale.tolist(),
        "applied_state_mean": [mean],
        "applied_state_scale": [scale],
    }


@torch.no_grad()
def _predict_fts_N(model: torch.nn.Module, x1_raw_m: np.ndarray, chunk_size: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    zeros = np.zeros_like(x1_raw_m)
    for start in range(0, x1_raw_m.size, chunk_size):
        stop = min(x1_raw_m.size, start + chunk_size)
        states = np.column_stack((x1_raw_m[start:stop], zeros[start:stop]))
        tensor = torch.as_tensor(states, dtype=torch.float64, device="cpu")
        pred = model(tensor).detach().cpu().numpy().reshape(-1)
        predictions.append(np.asarray(pred, dtype=np.float64))
    return np.concatenate(predictions)


def _stats(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "rms": float(np.sqrt(np.mean(data**2))),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "p01": float(np.percentile(data, 1.0)),
        "p99": float(np.percentile(data, 99.0)),
    }


def _plot(
    *,
    time_s: np.ndarray,
    x1_raw_m: np.ndarray,
    fts_pred_nN: np.ndarray,
    output_png: Path,
) -> None:
    time_us = time_s * 1.0e6
    window_centers_s = (
        max(0.5 * PERIOD_S, 0.10 * float(time_s[-1])),
        0.50 * float(time_s[-1]),
        min(float(time_s[-1]) - 0.5 * PERIOD_S, 0.90 * float(time_s[-1])),
    )
    font_scale = 1.875
    with plt.rc_context(
        {
            "font.size": 10.0 * font_scale,
            "axes.titlesize": 12.0 * font_scale,
            "axes.labelsize": 10.0 * font_scale,
            "xtick.labelsize": 10.0 * font_scale,
            "ytick.labelsize": 10.0 * font_scale,
            "legend.fontsize": 10.0 * font_scale,
        }
    ):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14.406, 11.301),
            gridspec_kw={"hspace": 0.38, "wspace": 0.16},
        )
        fts_axes = tuple(axes.ravel())

        half_width_s = PERIOD_S
        panels: tuple[tuple[str, np.ndarray], ...] = (
            ("Full time-span observation window", np.arange(time_s.size, dtype=int)),
            *(
                (
                    label,
                    np.flatnonzero(
                        (time_s >= max(float(time_s[0]), float(center_s) - half_width_s))
                        & (time_s <= min(float(time_s[-1]), float(center_s) + half_width_s))
                    ),
                )
                for label, center_s in zip(
                    (
                        "Early observation window",
                        "Middle observation window",
                        "Tail observation window",
                    ),
                    window_centers_s,
                    strict=True,
                )
            ),
        )

        for panel_index, (axis, (label, indices)) in enumerate(
            zip(fts_axes, panels, strict=True)
        ):
            panel_time_us = time_us[indices]
            panel_force = fts_pred_nN[indices]
            axis.plot(
                panel_time_us,
                panel_force,
                color="#d62728",
                linewidth=0.70 if label.startswith("Full") else 1.0,
                label=r"trained KAN prediction",
            )
            axis.set_title(label)
            if panel_index in (0, 2):
                axis.set_ylabel(r"$F_{ts}^{pred}$ [nN]")
            axis.set_xlabel(r"Time [$\mu$s]")
            axis.grid(True, alpha=0.25)
            if panel_index == 0:
                axis.legend(loc="upper right")

        for axis in fts_axes:
            axis.margins(x=0.0)
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--mat-path", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--normalizer-policy",
        choices=("trained_archive_x1_mean_std", "external_x1_raw_self_mean_std"),
        default="trained_archive_x1_mean_std",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / f"{DEFAULT_OUTPUT_STEM}.npz"
    output_json = output_dir / f"{DEFAULT_OUTPUT_STEM}_summary.json"
    output_png = output_dir / f"{DEFAULT_OUTPUT_STEM}.png"

    torch.set_num_threads(1)
    model, endpoint, payload, stage2_result, stage1_result = _restore_stage2_model(args.archive_dir)
    time_s, x1_raw_m, mat_source = _load_external_x1_raw(args.mat_path)
    if args.normalizer_policy == "external_x1_raw_self_mean_std":
        normalizer_record = _apply_external_x1_normalizer(model, x1_raw_m)
    else:
        normalizer_record = _record_training_normalizer(model)
    fts_pred_N = _predict_fts_N(model, x1_raw_m, int(args.chunk_size))
    fts_eff_pred_m_s2 = fts_pred_N / M_KG
    fts_pred_nN = fts_pred_N * 1.0e9

    np.savez_compressed(
        output_npz,
        time_s=time_s,
        time_us=time_s * 1.0e6,
        x1_raw_m=x1_raw_m,
        fts_pred_N=fts_pred_N,
        fts_eff_pred_m_s2=fts_eff_pred_m_s2,
        fts_pred_nN=fts_pred_nN,
        index=np.asarray(INDEX, dtype=np.int64),
        dt_s=np.asarray(DT_S, dtype=np.float64),
        m_kg=np.asarray(M_KG, dtype=np.float64),
        k_N_per_m=np.asarray(K_N_PER_M, dtype=np.float64),
        q_factor=np.asarray(Q_FACTOR, dtype=np.float64),
        drive_frequency_Hz=np.asarray(DRIVE_FREQUENCY_HZ, dtype=np.float64),
        normalizer_source=np.asarray(normalizer_record["normalizer_source"]),
        trained_state_mean_m=np.asarray(normalizer_record["trained_state_mean"], dtype=np.float64),
        trained_state_scale_m=np.asarray(normalizer_record["trained_state_scale"], dtype=np.float64),
        applied_state_mean_m=np.asarray(normalizer_record["applied_state_mean"], dtype=np.float64),
        applied_state_scale_m=np.asarray(normalizer_record["applied_state_scale"], dtype=np.float64),
    )
    _plot(time_s=time_s, x1_raw_m=x1_raw_m, fts_pred_nN=fts_pred_nN, output_png=output_png)

    digest = str(payload.get("model_state_dict_sha256", "")).strip()
    if not digest:
        digest = hashlib.sha256(stage2_result.read_bytes()).hexdigest()
    metadata = model.metadata()
    summary = {
        "definition": (
            "Pointwise prediction only: external x1 raw is fed to the archived "
            "AFM06a trained KAN at every time sample using the selected "
            "normalizer policy. No autonomous rollout or training is performed."
        ),
        "archive_dir": str(args.archive_dir.expanduser().resolve()),
        "stage2_result": str(stage2_result.resolve()),
        "stage1_dependency": str(stage1_result.resolve()),
        "external_mat_source": str(mat_source.resolve()),
        "index": INDEX,
        "sample_count": int(time_s.size),
        "time_range_us": [float(time_s[0] * 1.0e6), float(time_s[-1] * 1.0e6)],
        "dt_s": DT_S,
        "detail_windows_us": [
            [
                float(max(float(time_s[0]), float(center) - PERIOD_S) * 1.0e6),
                float(min(float(time_s[-1]), float(center) + PERIOD_S) * 1.0e6),
            ]
            for center in (
                max(0.5 * PERIOD_S, 0.10 * float(time_s[-1])),
                0.50 * float(time_s[-1]),
                min(float(time_s[-1]) - 0.5 * PERIOD_S, 0.90 * float(time_s[-1])),
            )
        ],
        "model_rank": int(payload.get("input_rank", endpoint.rank)),
        "model_state_dict_sha256": digest,
        "input_policy": str(metadata.get("input_policy")),
        "normalizer_application": normalizer_record["normalizer_source"],
        "trained_state_mean": normalizer_record["trained_state_mean"],
        "trained_state_scale": normalizer_record["trained_state_scale"],
        "applied_state_mean": normalizer_record["applied_state_mean"],
        "applied_state_scale": normalizer_record["applied_state_scale"],
        "state_mean": np.asarray(metadata.get("state_mean"), dtype=float).tolist(),
        "state_scale": np.asarray(metadata.get("state_scale"), dtype=float).tolist(),
        "force_output_policy": str(metadata.get("force_output_policy")),
        "force_mean": float(metadata.get("force_mean", 0.0)),
        "force_scale": float(metadata.get("force_scale", 1.0)),
        "gain_enabled": bool(metadata.get("gain_enabled")),
        "soft_mask": metadata.get("soft_mask"),
        "cantilever_for_conversion": {
            "m_kg": float(M_KG),
            "k_N_per_m": K_N_PER_M,
            "Q": Q_FACTOR,
            "drive_frequency_Hz": DRIVE_FREQUENCY_HZ,
            "note": "The trained KAN directly outputs physical Fts_pred_N; Fts_eff_pred_m_s2 = Fts_pred_N / m_kg is stored only as a derived diagnostic.",
        },
        "x1_raw_nm_stats": _stats(x1_raw_m * 1.0e9),
        "fts_eff_pred_m_s2_stats": _stats(fts_eff_pred_m_s2),
        "fts_pred_N_stats": _stats(fts_pred_N),
        "fts_pred_nN_stats": _stats(fts_pred_nN),
        "output_npz": str(output_npz),
        "output_png": str(output_png),
    }
    _write_json(output_json, summary)

    print(f"Saved: {output_npz}")
    print(f"Saved: {output_json}")
    print(f"Saved: {output_png}")
    print(
        "Fts_pred_nN range: "
        f"{float(np.min(fts_pred_nN)):.6g} to {float(np.max(fts_pred_nN)):.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
