"""Evaluate a trained AFM04 stage2light result after swapping ks/cs to truth.

This is a post-process diagnostic.  It does not train, does not backpropagate,
and does not create figures.  For one archived stage2light rank/candidate, it
reloads the final epoch model, evaluates the current final ks/cs, then evaluates
the same NN with ks/cs replaced by the known synthetic truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def find_repo_root(start: str | Path) -> Path:
    here = Path(start).resolve()
    for path in [here, *here.parents]:
        if (path / "AFM04").is_dir() and (path / "user requirements").is_dir():
            return path
    raise RuntimeError(f"Could not locate repo root from {start}")


REPO_ROOT = find_repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AFM04.stage1pluslight.data import make_train_val_masks, truncate_to_first_contact  # noqa: E402
from AFM04.stage2light.kan_backend import KANForceModule  # noqa: E402
from AFM04.stage2light.losses import TorchLossParts, evaluate_split  # noqa: E402
from AFM04.stage2light.rollout import LearnableMechModule  # noqa: E402


ARCHIVE_ROOT = REPO_ROOT / "AFM04" / "archive" / "st2l" / "3_10x10x1000_W0 new"
DEFAULT_OUTPUT_NAME = "afm04_stage2light_true_mech_swap_loss_04.txt"


def _torch_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("float64", "double"):
        return torch.float64
    if key in ("float32", "single"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def _load_table_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: np.asarray(data[k]) for k in data.files if k != "__columns__"}


def _resolve_rank_dir(args: argparse.Namespace) -> Path:
    if args.rank_dir:
        rank_dir = Path(args.rank_dir)
        if not rank_dir.is_absolute():
            rank_dir = (REPO_ROOT / rank_dir).resolve()
        return rank_dir

    archive_root = Path(args.archive_root)
    if not archive_root.is_absolute():
        archive_root = (REPO_ROOT / archive_root).resolve()
    if args.rank is None and args.candidate is None:
        raise ValueError("provide --rank-dir, or at least --rank / --candidate")

    matches: list[Path] = []
    for path in archive_root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        ok = True
        if args.rank is not None:
            ok = ok and name.startswith(f"rank{int(args.rank)}_")
        if args.candidate is not None:
            cand = str(args.candidate).strip().upper().removeprefix("B")
            suffixes = (f"_B{int(cand):02d}", f"_B{int(cand)}")
            ok = ok and name.upper().endswith(suffixes)
        if ok:
            matches.append(path)
    if len(matches) != 1:
        listed = ", ".join(p.name for p in matches[:10])
        raise RuntimeError(f"expected exactly one rank/candidate match, found {len(matches)}: {listed}")
    return matches[0]


def _load_result_payload(rank_dir: Path) -> tuple[Path, dict[str, Any]]:
    result_path = rank_dir / "result" / "stage2light_result_p1.pt"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing stage2light result payload: {result_path}")
    payload = torch.load(result_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected payload type in {result_path}: {type(payload)!r}")
    return result_path, payload


def _build_model_and_mech(payload: dict[str, Any]) -> tuple[KANForceModule, LearnableMechModule, tuple[float, ...], torch.dtype]:
    cfg = payload["config"]
    dtype = _torch_dtype(cfg["dtype"])
    device = "cpu"
    known_pars = tuple(float(v) for v in payload["known_pars"])

    model = KANForceModule(
        pykan_root=Path(cfg["pykan_root"]),
        state_mean=np.asarray(payload["state_mean"], dtype=float),
        state_scale=np.asarray(payload["state_scale"], dtype=float),
        seed=int(cfg["seed"]),
        width=tuple(int(v) for v in cfg["width"]),
        grid=int(cfg["grid"]),
        spline_k=int(cfg["spline_k"]),
        base_fun=str(cfg["base_fun"]),
        symbolic_enabled=bool(cfg["symbolic_enabled"]),
        auto_save=bool(cfg["auto_save"]),
        noise_scale=float(cfg["noise_scale"]),
        affine_trainable=bool(cfg["affine_trainable"]),
        grid_eps=float(cfg["grid_eps"]),
        grid_range=(float(cfg["grid_range_lo"]), float(cfg["grid_range_hi"])),
        gnn_learnable=bool(cfg["gnn_learnable"]),
        dist=float(known_pars[6]),
        a0=float(known_pars[9]),
        soft_mask_enabled=bool(cfg.get("soft_mask_enabled", True)),
        soft_mask_trainable=bool(cfg.get("soft_mask_trainable", True)),
        soft_mask_s0_a0=float(cfg.get("soft_mask_s0_a0", 20.0)),
        soft_mask_s0_min_a0=float(cfg.get("soft_mask_s0_min_a0", 1.0)),
        soft_mask_s0_max_a0=float(cfg.get("soft_mask_s0_max_a0", 100.0)),
        soft_mask_alpha_a0=float(cfg.get("soft_mask_alpha_a0", 0.25)),
        soft_mask_alpha_min_a0=float(cfg.get("soft_mask_alpha_min_a0", 0.02)),
        soft_mask_alpha_max_a0=float(cfg.get("soft_mask_alpha_max_a0", 5.0)),
        device=device,
        dtype=dtype,
    ).to(device)

    mech_module = LearnableMechModule(
        ks_init=float(payload["mech_true"][0]),
        cs_init=float(payload["mech_true"][1]),
        ks_bounds=(float(cfg["ks_lo"]), float(cfg["ks_hi"])),
        cs_bounds=(float(cfg["cs_lo"]), float(cfg["cs_hi"])),
        dtype=dtype,
        device=device,
        parameterization=str(cfg.get("mech_parameterization", "sigmoid_bounded")),
    ).to(device)

    state_dict = payload.get("final_state_dict")
    mech_state_dict = payload.get("final_mech_state_dict")
    if not isinstance(state_dict, dict) or not isinstance(mech_state_dict, dict):
        state_dict = payload.get("state_dict")
        mech_state_dict = payload.get("mech_state_dict")
    if not isinstance(state_dict, dict) or not isinstance(mech_state_dict, dict):
        raise RuntimeError("payload is missing final_state_dict/final_mech_state_dict")

    model.load_state_dict(state_dict)
    mech_module.load_state_dict(mech_state_dict, strict=False)
    model.eval()
    mech_module.eval()
    return model, mech_module, known_pars, dtype


def _build_true_mech_module(payload: dict[str, Any], dtype: torch.dtype) -> LearnableMechModule:
    cfg = payload["config"]
    mech_true = np.asarray(payload["mech_true"], dtype=float).reshape(-1)
    if mech_true.size < 2:
        raise RuntimeError("payload mech_true must contain ks and cs")
    module = LearnableMechModule(
        ks_init=float(mech_true[0]),
        cs_init=float(mech_true[1]),
        ks_bounds=(float(cfg["ks_lo"]), float(cfg["ks_hi"])),
        cs_bounds=(float(cfg["cs_lo"]), float(cfg["cs_hi"])),
        dtype=dtype,
        device="cpu",
        parameterization=str(cfg.get("mech_parameterization", "sigmoid_bounded")),
    ).to("cpu")
    module.eval()
    return module


def _load_window_tensors(rank_dir: Path, payload: dict[str, Any], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    data_dir = rank_dir / "data"
    ode_path = data_dir / "ode_data_afm_dmt_kv.npz"
    pert_path = data_dir / "pert_df_afm_dmt_kv.npz"
    if not ode_path.is_file() or not pert_path.is_file():
        raise FileNotFoundError(f"missing archived data files under {data_dir}")

    with np.load(ode_path) as data:
        ode_data = np.asarray(data["data"], dtype=float)
    pert_df = _load_table_npz(pert_path)
    ode_data, pert_df = truncate_to_first_contact(ode_data, pert_df)

    meta = payload.get("window_meta", {})
    start = int(meta.get("start_idx", 0))
    stop = int(meta.get("stop_idx", ode_data.shape[1] - 1))
    if start < 0 or stop < start or stop >= ode_data.shape[1]:
        raise RuntimeError(f"invalid window_meta indices after first-contact truncation: start={start} stop={stop}")
    idx = np.arange(start, stop + 1, dtype=int)

    ode_full = np.asarray(ode_data[:, idx], dtype=float)
    times_full = np.asarray(pert_df["t"], dtype=float)[idx]
    x2dot_full = np.asarray(pert_df["x2dot"], dtype=float)[idx]
    contact_full = np.asarray(pert_df["contact"], dtype=float)[idx]

    cfg = payload["config"]
    train_idx, val_idx = make_train_val_masks(len(times_full), int(cfg["val_stride"]), int(cfg["val_offset"]))
    return {
        "ode_full": torch.as_tensor(ode_full, dtype=dtype),
        "times_full": torch.as_tensor(times_full, dtype=dtype),
        "x2dot_full": torch.as_tensor(x2dot_full, dtype=dtype),
        "contact_full": torch.as_tensor(contact_full, dtype=dtype),
        "train_idx": torch.as_tensor(train_idx, dtype=torch.long),
        "val_idx": torch.as_tensor(val_idx, dtype=torch.long),
    }


def _loss_indices(tensors: dict[str, torch.Tensor], split: str) -> torch.Tensor | None:
    split = split.strip().lower()
    if split == "full":
        return None
    if split == "train":
        return tensors["train_idx"]
    if split == "val":
        return tensors["val_idx"]
    raise ValueError(f"unsupported split: {split!r}")


def _metrics_dict(total: torch.Tensor, parts: TorchLossParts) -> dict[str, float]:
    return {
        "loss": float(total.detach()),
        "state": float(parts.state),
        "x1_state": float(parts.x1_state),
        "x2_state": float(parts.x2_state),
        "x2dot": float(parts.x2dot),
        "x3_range": float(parts.x3_range),
        "fts_range": float(parts.fts_range),
        "cont": float(parts.cont),
        "x1_rec_pct": float(parts.x1_rec),
        "x3_rec_pct": float(parts.x3_rec),
        "fts_rollout_rec_pct": float(parts.fts_rollout_rec),
    }


def _reference_metrics_from_payload(payload: dict[str, Any], split: str) -> dict[str, float] | None:
    snapshot = payload.get("final_snapshot")
    if not isinstance(snapshot, dict):
        return None
    section = snapshot.get(split)
    if not isinstance(section, dict):
        return None
    metrics = section.get("metrics")
    if not isinstance(metrics, dict):
        return None

    aliases = {
        "x1_rec_pct": "x1_rec",
        "x3_rec_pct": "x3_rec",
        "fts_rollout_rec_pct": "fts_rollout_rec",
    }
    out: dict[str, float] = {}
    for key in _METRIC_KEYS:
        source_key = aliases.get(key, key)
        if source_key not in metrics:
            return None
        out[key] = float(metrics[source_key])
    return out


_METRIC_KEYS = (
    "loss",
    "state",
    "x1_state",
    "x2_state",
    "x2dot",
    "x3_range",
    "fts_range",
    "cont",
    "x1_rec_pct",
    "x3_rec_pct",
    "fts_rollout_rec_pct",
)


def _evaluate(
    *,
    force_module: KANForceModule,
    mech_module: LearnableMechModule,
    payload: dict[str, Any],
    known_pars: tuple[float, ...],
    tensors: dict[str, torch.Tensor],
    split: str,
) -> dict[str, float]:
    cfg = payload["config"]
    with torch.no_grad():
        total, parts, _traj = evaluate_split(
            force_module=force_module,
            known_pars=known_pars,
            mech_module=mech_module,
            ode_true=tensors["ode_full"],
            x2dot_true=tensors["x2dot_full"],
            contact_mask=tensors["contact_full"],
            times=tensors["times_full"],
            ode_method=str(cfg["ode_method"]),
            ode_rtol=float(cfg["ode_rtol"]),
            ode_atol=float(cfg["ode_atol"]),
            mech_true=torch.as_tensor(payload["mech_true"], dtype=tensors["ode_full"].dtype),
            eta_star_true=float(payload["eta_star_true"]),
            loss_indices=_loss_indices(tensors, split),
        )
    return _metrics_dict(total, parts)


def _fmt(value: float) -> str:
    if not np.isfinite(float(value)):
        return str(value)
    return f"{float(value):.16e}"


def _write_report(
    *,
    out_path: Path,
    rank_dir: Path,
    result_path: Path,
    payload: dict[str, Any],
    split: str,
    before_mech: np.ndarray,
    true_mech: np.ndarray,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    reference_metrics: dict[str, float] | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("AFM04 stage2light true-mech swap loss diagnostic")
    lines.append("=" * 55)
    lines.append("")
    lines.append(f"rank_dir: {rank_dir}")
    lines.append(f"result_payload: {result_path}")
    lines.append(f"split: {split}")
    lines.append(f"window_role: {payload.get('window_meta', {}).get('role', '')}")
    lines.append(f"window_label: {payload.get('window_meta', {}).get('label', '')}")
    lines.append(f"window_indices_after_first_contact: {payload.get('window_meta', {}).get('start_idx', '')}..{payload.get('window_meta', {}).get('stop_idx', '')}")
    lines.append("")
    lines.append("Mechanistic parameters")
    lines.append("----------------------")
    lines.append(f"ks_pred_final: {_fmt(float(before_mech[0]))}")
    lines.append(f"cs_pred_final: {_fmt(float(before_mech[1]))}")
    lines.append(f"ks_true:       {_fmt(float(true_mech[0]))}")
    lines.append(f"cs_true:       {_fmt(float(true_mech[1]))}")
    lines.append("")
    lines.append("Original final epoch mech")
    lines.append("-------------------------")
    for key, value in before_metrics.items():
        lines.append(f"{key}: {_fmt(value)}")
    if reference_metrics is not None:
        lines.append("")
        lines.append("Consistency check against st2l final_snapshot")
        lines.append("---------------------------------------------")
        max_abs_diff = 0.0
        for key in _METRIC_KEYS:
            diff = before_metrics[key] - reference_metrics[key]
            max_abs_diff = max(max_abs_diff, abs(diff))
            lines.append(f"{key}_diff_recomputed_minus_saved: {_fmt(diff)}")
        lines.append(f"max_abs_diff: {_fmt(max_abs_diff)}")
    lines.append("")
    lines.append("True ks/cs swapped mech")
    lines.append("------------------------")
    for key, value in after_metrics.items():
        lines.append(f"{key}: {_fmt(value)}")
    lines.append("")
    lines.append("Delta: swapped - original")
    lines.append("-------------------------")
    for key in before_metrics:
        lines.append(f"{key}: {_fmt(after_metrics[key] - before_metrics[key])}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", default=str(ARCHIVE_ROOT), help="Archive root containing rank*_B* directories.")
    parser.add_argument("--rank-dir", default="", help="Specific archived stage2light rank directory.")
    parser.add_argument("--rank", type=int, default=None, help="Rank number to discover under --archive-root.")
    parser.add_argument("--candidate", default=None, help="Candidate B number to discover under --archive-root.")
    parser.add_argument("--split", default="train", choices=("train", "val", "full"), help="Loss split to evaluate. Default: train.")
    parser.add_argument("--output", default="", help="Output txt path. Default: rank_dir/result/afm04_stage2light_true_mech_swap_loss_04.txt")
    args = parser.parse_args()

    rank_dir = _resolve_rank_dir(args).resolve()
    result_path, payload = _load_result_payload(rank_dir)
    force_module, mech_module, known_pars, dtype = _build_model_and_mech(payload)
    true_mech_module = _build_true_mech_module(payload, dtype)
    tensors = _load_window_tensors(rank_dir, payload, dtype)

    before_mech = mech_module().detach().cpu().numpy().reshape(-1)
    true_mech = np.asarray(payload["mech_true"], dtype=float).reshape(-1)
    before_metrics = _evaluate(
        force_module=force_module,
        mech_module=mech_module,
        payload=payload,
        known_pars=known_pars,
        tensors=tensors,
        split=args.split,
    )
    reference_metrics = _reference_metrics_from_payload(payload, args.split)
    # Use the same NN/normalizer/grid state, only swap the mechanistic module.
    after_metrics = _evaluate(
        force_module=force_module,
        mech_module=true_mech_module,
        payload=payload,
        known_pars=known_pars,
        tensors=tensors,
        split=args.split,
    )

    out_path = Path(args.output) if args.output else rank_dir / "result" / DEFAULT_OUTPUT_NAME
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    _write_report(
        out_path=out_path,
        rank_dir=rank_dir,
        result_path=result_path,
        payload=payload,
        split=args.split,
        before_mech=before_mech,
        true_mech=true_mech,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        reference_metrics=reference_metrics,
    )
    print(f"wrote {out_path}")
    print(f"original_loss={before_metrics['loss']:.16e}")
    print(f"swapped_loss={after_metrics['loss']:.16e}")
    if reference_metrics is not None:
        max_abs_diff = max(abs(before_metrics[k] - reference_metrics[k]) for k in _METRIC_KEYS)
        print(f"original_vs_saved_final_snapshot_max_abs_diff={max_abs_diff:.16e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
