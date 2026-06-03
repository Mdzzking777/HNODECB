from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from AFM04.KAN_full_test.runner.visualization._common import (
    CHECKPOINT_DIR,
    _build_model_for_checkpoint,
    _config_from_payload,
    _select_split,
    finalize_and_save,
    history_phases,
    load_result_payloads,
    out_path,
    plot_phase_series,
    prepend_epoch0_history,
    sample_epoch_series,
    stage_title,
    window_title,
)


def _rs_tag_from_role(role: str) -> str:
    role_key = str(role or "").strip().lower()
    if role_key in ("modified_w0", "modified-w0", "shifted_w0", "shifted-w0"):
        return "modified_w0"
    return "w0"


def _epoch0_loss_terms_from_rs(payload: dict) -> dict[str, float] | None:
    role = str((payload.get("window_meta") or {}).get("role", ""))
    rs_path = CHECKPOINT_DIR / "random_search" / f"kan_full_test_random_search_best_{_rs_tag_from_role(role)}.pt"
    if not rs_path.is_file():
        return None
    try:
        import torch

        from AFM04.KAN_full_test.data import prepare_data
        from AFM04.KAN_full_test.losses import evaluate_split
        from AFM04.KAN_full_test.train import _split_loss_indices, _torch_dtype, _window_to_torch
    except ModuleNotFoundError:
        return None

    try:
        rs_payload = torch.load(rs_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(rs_payload, dict) or "best_state_dict" not in rs_payload:
        return None

    cfg = _config_from_payload(rs_payload)
    prepared = prepare_data(cfg)
    split = _select_split(prepared, rs_payload, cfg)
    dtype = _torch_dtype(cfg.dtype)
    model = _build_model_for_checkpoint(cfg, prepared, dtype)
    model.load_state_dict(rs_payload["best_state_dict"])
    model.eval()

    tensors = _window_to_torch(split, dtype=dtype, device=cfg.device)
    mech_true = torch.as_tensor(prepared.mech_true, dtype=dtype, device=cfg.device)
    out: dict[str, float] = {}
    with torch.no_grad():
        for name in ("train", "val"):
            total, parts, _ = evaluate_split(
                force_module=model,
                known_pars=prepared.known_pars,
                mech_true=mech_true,
                ode_true=tensors["ode_full"],
                x2dot_true=tensors["x2dot_full"],
                contact_mask=tensors["contact_full"],
                times=tensors["times_full"],
                ode_method=cfg.ode_method,
                ode_rtol=cfg.ode_rtol,
                ode_atol=cfg.ode_atol,
                eta_star_true=prepared.eta_star_true,
                loss_indices=_split_loss_indices(tensors, name),
            )
            out[f"{name}_loss"] = float(total.detach())
            out[f"{name}_x1_state"] = float(parts.x1_state)
            out[f"{name}_x2_state"] = float(parts.x2_state)
            out[f"{name}_x2dot"] = float(parts.x2dot)
    return out


def main() -> None:
    payloads = load_result_payloads()
    ncols = max(1, len(payloads))
    fig, axes = plt.subplots(3, ncols, figsize=(6 * ncols, 13), sharex=False, squeeze=False)
    for col, payload in enumerate(payloads):
        hist = payload.get("history", [])
        hist = prepend_epoch0_history(hist, payload.get("window_meta", {}).get("role", ""))
        if hist and int(float(hist[0].get("epoch", -1))) == 0:
            epoch0_terms = _epoch0_loss_terms_from_rs(payload)
            if epoch0_terms:
                hist[0].update(epoch0_terms)
        epochs = [int(row["epoch"]) for row in hist]
        phases = history_phases(hist)
        x1_state = [float(row.get("val_x1_state", float("nan"))) for row in hist]
        x2_state = [float(row.get("val_x2_state", float("nan"))) for row in hist]
        x2dot = [float(row.get("val_x2dot", float("nan"))) for row in hist]
        title = f"{stage_title(payload)}: {window_title(payload)}"
        epochs_x1, x1_state_sampled, phases_x1 = sample_epoch_series(epochs, x1_state, phases)
        epochs_x2, x2_state_sampled, phases_x2 = sample_epoch_series(epochs, x2_state, phases)
        epochs_x2dot, x2dot_sampled, phases_x2dot = sample_epoch_series(epochs, x2dot, phases)

        ax1 = axes[0, col]
        plot_phase_series(ax1, epochs_x1, x1_state_sampled, phases_x1, color="seagreen", linewidth=2, label="x1_state")
        ax1.set_title(title)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("x1 state loss")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        ax2 = axes[1, col]
        plot_phase_series(ax2, epochs_x2, x2_state_sampled, phases_x2, color="royalblue", linewidth=2, label="x2_state")
        ax2.set_title(title)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("x2 state loss")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

        ax3 = axes[2, col]
        plot_phase_series(ax3, epochs_x2dot, x2dot_sampled, phases_x2dot, color="darkorange", linewidth=2, label="x2dot")
        ax3.set_title(title)
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("x2dot loss")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="best")

    finalize_and_save(fig, out_path("afm_param_kan_full_test_04_x1x2state_x2dot_grid.png"))


if __name__ == "__main__":
    main()
