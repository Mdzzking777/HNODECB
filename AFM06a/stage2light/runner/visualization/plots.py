"""AFM06a stage2light plot implementations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from AFM06a.stage1pluslight.data import load_dataset
from AFM06a.stage2light.runner.visualization._common import (
    VisualizationContext,
    finalize_and_save,
    finite_history_series,
    full_rollout_panels,
    full_rollout_panels_from_w0,
    full_rollout_payload,
    full_rollout_payload_from_w0,
    load_visualization_context,
    nested_history_series,
    output_path,
    pointwise_full_resolution_payload,
    pointwise_full_resolution_payload_from_w0,
    relative_rmse_pct,
)
from AFM06a.stage2light.train import _forward_evaluation, _loss_term_gradient_norms


def _context(context: VisualizationContext | None) -> VisualizationContext:
    return load_visualization_context() if context is None else context


def _finish(fig, context: VisualizationContext, filename: str):
    return finalize_and_save(fig, output_path(context, filename))


def _mark_lbfgs_start(
    ax: plt.Axes,
    context: VisualizationContext,
    *,
    label: bool = False,
) -> None:
    first_lbfgs_epoch = context.config.adam_epochs + 1
    ax.axvline(
        first_lbfgs_epoch,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        label="L-BFGS starts" if label else None,
        zorder=5,
    )


def plot_losses(context: VisualizationContext | None = None):
    ctx = _context(context)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for key, label, color, style in (
        ("train_loss", "training loss", "royalblue", "-"),
        ("val_loss", "validation loss", "crimson", "--"),
    ):
        epochs, values = finite_history_series(ctx.history, key)
        if values.size:
            ax.semilogy(epochs, values, color=color, linestyle=style, linewidth=2.0, label=label)
    _mark_lbfgs_start(ax, ctx, label=True)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\mathcal{L}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    return _finish(fig, ctx, "afm_param_stage2light_06a_loss.png")


def plot_grad_norm(context: VisualizationContext | None = None):
    ctx = _context(context)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharex=True)
    series = (
        ("grad_norm_total", r"total loss", "black"),
        (
            "grad_norm_loss_dynamics_residual",
            r"dynamics-residual loss",
            "seagreen",
        ),
        (
            "grad_norm_loss_smoothness",
            r"smoothness loss",
            "mediumpurple",
        ),
    )
    component_keys = {key for key, _, _ in series[1:]}
    has_component_history = any(
        finite_history_series(ctx.history, key)[1].size for key in component_keys
    )
    final_component_norms: dict[str, float] = {}
    if not has_component_history:
        final_evaluation = _forward_evaluation(ctx.model, ctx.endpoint)
        values = _loss_term_gradient_norms(ctx.model, final_evaluation.train_term_tensors)
        final_component_norms = {
            "grad_norm_loss_dynamics_residual": values["dynamics_residual"],
            "grad_norm_loss_smoothness": values["smoothness"],
        }
    final_epoch = max((int(row.get("epoch", 0)) for row in ctx.history), default=0)
    for panel_index, (ax, (key, label, color)) in enumerate(
        zip(axes.ravel(), series, strict=True)
    ):
        epochs, values = finite_history_series(ctx.history, key)
        final_only = False
        if not values.size and key in final_component_norms:
            epochs = np.asarray([final_epoch], dtype=int)
            values = np.asarray([final_component_norms[key]], dtype=float)
            final_only = True
        if values.size:
            positive = np.maximum(values, np.finfo(float).tiny)
            ax.semilogy(epochs, positive, color=color, linewidth=1.8, marker="o", markersize=4)
            if final_only:
                ax.text(
                    0.5,
                    0.08,
                    "final model only; legacy history has no per-term gradients",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    color="dimgray",
                    fontsize=9,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "not recorded in legacy epochs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="dimgray",
            )
        _mark_lbfgs_start(ax, ctx, label=panel_index == 0)
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("gradient norm")
        ax.grid(True, which="both", alpha=0.25)
        if panel_index == 0:
            ax.legend(loc="best")
    return _finish(fig, ctx, "afm_param_stage2light_06a_grad_norm_grid.png")


def plot_state_bar_fts_losses(context: VisualizationContext | None = None):
    ctx = _context(context)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
    items = (
        ("dynamics_residual", r"$\mathcal{L}_{r_{\mathrm{dyn}}}$"),
        ("smoothness", r"$\lambda_{\mathrm{sm}}\mathcal{L}_{\mathrm{sm}}$"),
    )
    for panel_index, (ax, (key, ylabel)) in enumerate(
        zip(axes, items, strict=True)
    ):
        for group, label, color, style in (
            ("loss_parts", "training", "royalblue", "-"),
            ("val_loss_parts", "validation", "crimson", "--"),
        ):
            epochs, values = nested_history_series(ctx.history, group, key)
            if values.size:
                ax.semilogy(
                    epochs,
                    np.maximum(values, np.finfo(float).tiny),
                    color=color,
                    linestyle=style,
                    linewidth=1.8,
                    label=label,
                )
        _mark_lbfgs_start(ax, ctx, label=panel_index == 0)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("epoch")
    return _finish(
        fig,
        ctx,
        "afm_param_stage2light_06a_dynamics_residual_grid.png",
    )


def plot_reconstruction(context: VisualizationContext | None = None):
    ctx = _context(context)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False)
    items = (
        ("x1_rec", r"$x_1$ reconstruction error"),
        ("x2_rec", r"$x_2$ reconstruction error"),
    )
    for panel_index, (ax, (key, title)) in enumerate(zip(axes[:2], items, strict=True)):
        for group, label, color, style in (
            ("loss_parts", "training", "royalblue", "-"),
            ("val_loss_parts", "validation", "crimson", "--"),
        ):
            epochs, values = nested_history_series(ctx.history, group, key)
            if values.size:
                ax.plot(epochs, values, color=color, linestyle=style, linewidth=1.8, label=label)
        _mark_lbfgs_start(ax, ctx, label=panel_index == 0)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("relative RMSE (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    force_ax = axes[2]
    for key, label, color, style in (
        ("bar_fts_rec_train_pct", "training", "royalblue", "-"),
        ("bar_fts_rec_val_pct", "validation", "crimson", "--"),
    ):
        epochs, values = finite_history_series(ctx.history, key)
        if values.size:
            force_ax.plot(epochs, values, color=color, linestyle=style, linewidth=1.8, label=label)
    _mark_lbfgs_start(force_ax, ctx)
    force_ax.set_title(r"$\bar{F}_{ts}$ diagnostic error")
    force_ax.set_xlabel("epoch")
    force_ax.set_ylabel("relative RMSE (%)")
    force_ax.grid(True, alpha=0.25)
    force_ax.legend(loc="best")
    return _finish(fig, ctx, "afm_param_stage2light_06a_recon_nn_grid.png")


def plot_final_bar_fts(context: VisualizationContext | None = None):
    ctx = _context(context)
    times_ms = 1.0e3 * ctx.endpoint.window.times
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.6))
    ax.plot(
        times_ms,
        ctx.endpoint.true_fts,
        color="black",
        linewidth=2.0,
        label=r"$F_{ts}$",
        zorder=2,
    )
    ax.plot(
        times_ms,
        ctx.predicted_fts,
        color="crimson",
        linestyle="-",
        linewidth=1.0,
        label=r"$\hat{F}_{ts}$",
        zorder=3,
    )
    error = relative_rmse_pct(ctx.predicted_fts, ctx.endpoint.true_fts)
    ax.text(
        0.98,
        0.95,
        f"relative RMSE = {error:.4f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.set_ylabel(r"$F_{ts}$ (N)")
    ax.set_xlabel("time (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    return _finish(fig, ctx, "afm_param_stage2light_06a_bar_fts_traj.png")


def plot_final_observables(context: VisualizationContext | None = None):
    ctx = _context(context)
    times_ms = 1.0e3 * ctx.endpoint.window.times
    truth = (ctx.endpoint.window.states[0] * 1.0e9, ctx.endpoint.window.states[1] * 1.0e3, ctx.endpoint.window.x2dot)
    prediction = (ctx.predicted_states[0] * 1.0e9, ctx.predicted_states[1] * 1.0e3, ctx.predicted_x2dot)
    labels = ((r"$x_1$", "nm"), (r"$x_2$", r"mm s$^{-1}$"), (r"$\dot{x}_2$", r"m s$^{-2}$"))
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), sharex=True)
    for ax, ref, pred, (symbol, unit) in zip(axes, truth, prediction, labels, strict=True):
        ax.plot(times_ms, ref, color="black", linewidth=2.0, label=fr"${symbol.strip('$')}$ observed")
        ax.plot(times_ms, pred, color="crimson", linestyle="--", linewidth=1.7, label=fr"${symbol.strip('$')}$ predicted")
        train = ctx.endpoint.window.train_idx
        val = ctx.endpoint.window.val_idx
        ax.scatter(times_ms[train], ref[train], s=8, color="royalblue", alpha=0.55, label="training samples")
        if val.size:
            ax.scatter(times_ms[val], ref[val], s=10, color="red", alpha=0.75, label="validation samples")
        ax.text(0.02, 0.95, f"relative RMSE = {relative_rmse_pct(pred, ref):.4f}%", transform=ax.transAxes, ha="left", va="top")
        ax.set_ylabel(f"{symbol} ({unit})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", ncol=2)
    axes[-1].set_xlabel("time (ms)")
    return _finish(fig, ctx, "afm_param_stage2light_06a_x1_x2_x2dot_traj_grid.png")


def plot_preopt_rank(context: VisualizationContext | None = None):
    ctx = _context(context)
    record = ctx.endpoint.record
    times_ms = 1.0e3 * ctx.endpoint.window.times
    if "predicted_states" in record and "predicted_x2dot" in record:
        pred_states = np.asarray(record["predicted_states"], dtype=float)
        pred_x2dot = np.asarray(record["predicted_x2dot"], dtype=float)
    else:
        pred_states = np.full_like(ctx.endpoint.window.states, np.nan, dtype=float)
        pred_x2dot = np.full_like(ctx.endpoint.window.x2dot, np.nan, dtype=float)
    pred_force = np.asarray(record["predicted_bar_fts"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    panels = (
        (ctx.endpoint.true_bar_fts, pred_force, r"$F_{ts,\mathrm{eff}}$", r"m s$^{-2}$"),
        (ctx.endpoint.window.states[0] * 1.0e9, pred_states[0] * 1.0e9, r"$x_1$", "nm"),
        (ctx.endpoint.window.states[1] * 1.0e3, pred_states[1] * 1.0e3, r"$x_2$", r"mm s$^{-1}$"),
        (ctx.endpoint.window.x2dot, pred_x2dot, r"$\dot{x}_2$", r"m s$^{-2}$"),
    )
    for ax, (truth, pred, symbol, unit) in zip(axes.ravel(), panels, strict=True):
        ax.plot(times_ms, truth, color="black", linewidth=2.0, label="observed/reference")
        ax.plot(times_ms, pred, color="steelblue", linestyle="--", linewidth=1.6, label="st1pl prediction")
        ax.set_title(f"{symbol} at st2l entry")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(f"{symbol} ({unit})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    return _finish(fig, ctx, "afm_param_stage2light_06a_preopt_rank_grid.png")


def _training_style_panel_subset(
    context: VisualizationContext,
    payload: dict,
    indices: np.ndarray,
    *,
    target_count: int,
) -> np.ndarray:
    """Subsample a dense panel with the formal 10:5:1 W1 density policy."""

    indices = np.asarray(indices, dtype=int)
    if indices.size <= target_count:
        return indices

    times = np.asarray(payload["times"], dtype=float)[indices]
    full_contact = np.asarray(payload["true_contact"], dtype=bool)
    if full_contact.shape != np.asarray(payload["times"]).shape:
        raise RuntimeError("full-rollout contact labels do not match the time grid")
    contact = full_contact[indices]

    saved_config = context.endpoint.payload["config"]
    weights = np.where(
        contact,
        float(saved_config["contact_sampling_weight"]),
        float(saved_config["noncontact_sampling_weight"]),
    )
    half_width = float(saved_config["transition_half_width_s"])
    switch_idx = np.flatnonzero(contact[1:] != contact[:-1]) + 1
    for switch in switch_idx:
        boundary = 0.5 * (times[switch - 1] + times[switch])
        weights[np.abs(times - boundary) <= half_width] = float(
            saved_config["transition_sampling_weight"]
        )

    cell_width = np.empty_like(times)
    cell_width[1:-1] = 0.5 * (times[2:] - times[:-2])
    cell_width[0] = 0.5 * (times[1] - times[0])
    cell_width[-1] = 0.5 * (times[-1] - times[-2])
    cumulative = np.cumsum(weights * cell_width)
    targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), target_count)
    positions: list[int] = []
    used = np.zeros(indices.size, dtype=bool)
    for target in targets:
        insertion = int(np.searchsorted(cumulative, target, side="left"))
        insertion = min(max(insertion, 0), indices.size - 1)
        left = insertion
        right = insertion + 1
        while True:
            candidates = []
            if left >= 0 and not used[left]:
                candidates.append(left)
            if right < indices.size and not used[right]:
                candidates.append(right)
            if candidates:
                selected = min(
                    candidates,
                    key=lambda position: abs(float(cumulative[position]) - float(target)),
                )
                used[selected] = True
                positions.append(selected)
                break
            left -= 1
            right += 1
    positions = np.asarray(sorted(positions), dtype=int)
    return indices[positions]


def _plot_full_quantity(
    context: VisualizationContext,
    *,
    truth_key: str,
    prediction_key: str,
    component: int | None,
    scale: float,
    ylabel: str,
    filename: str,
    use_training_style_samples: bool = False,
    use_full_rate_reference: bool = False,
    payload_builder=full_rollout_payload,
    panels_builder=full_rollout_panels,
    reference_label_override: str | None = None,
    prediction_label: str = "Rollout prediction",
):
    payload = payload_builder(context)
    panels = panels_builder(context, payload)
    times_ms = 1.0e3 * np.asarray(payload["times"], dtype=float)
    truth = np.asarray(payload[truth_key], dtype=float)
    prediction = np.asarray(payload[prediction_key], dtype=float)
    if component is not None:
        truth = truth[component]
        prediction = prediction[component]
    truth = scale * truth
    prediction = scale * prediction

    panel_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, indices in panels:
        panel_indices = np.asarray(indices, dtype=int)
        if use_training_style_samples:
            window_duration = float(
                context.endpoint.window.times[-1]
                - context.endpoint.window.times[0]
            )
            if name.startswith("Observation window"):
                target_count = int(context.endpoint.window.times.size)
            else:
                panel_duration = float(
                    payload["times"][panel_indices[-1]]
                    - payload["times"][panel_indices[0]]
                )
                target_count = max(
                    2,
                    int(
                        round(
                            context.endpoint.window.times.size
                            * panel_duration
                            / window_duration
                        )
                    ),
                )
            panel_indices = _training_style_panel_subset(
                context,
                payload,
                panel_indices,
                target_count=target_count,
            )
        panel_times = times_ms[panel_indices]
        panel_truth = truth[panel_indices]
        panel_prediction = prediction[panel_indices]
        panel_series[name] = (panel_times, panel_truth, panel_prediction)

    if use_training_style_samples:
        full_name = "Full time-span"
        full_times, full_truth, full_prediction = panel_series[full_name]
        for detail_name, _ in panels[1:]:
            detail_times, detail_truth, detail_prediction = panel_series[detail_name]
            outside_detail = (full_times < detail_times[0]) | (
                full_times > detail_times[-1]
            )
            full_times = np.concatenate((full_times[outside_detail], detail_times))
            full_truth = np.concatenate((full_truth[outside_detail], detail_truth))
            full_prediction = np.concatenate(
                (full_prediction[outside_detail], detail_prediction)
            )
            order = np.argsort(full_times)
            full_times = full_times[order]
            full_truth = full_truth[order]
            full_prediction = full_prediction[order]
        panel_series[full_name] = (full_times, full_truth, full_prediction)

    full_rate_times_ms = None
    full_rate_truth = None
    if use_full_rate_reference:
        if truth_key == "true_bar_fts" and component is None:
            full_rate_values = "bar_fts"
        elif truth_key == "true_fts" and component is None:
            full_rate_values = "Fts"
        elif truth_key == "true_states" and component is not None:
            full_rate_values = int(component)
        else:
            raise ValueError(
                "full-rate reference is not defined for this plotted quantity"
            )
        saved_config = context.endpoint.payload["config"]
        loaded = load_dataset(
            saved_config["dataset_root"],
            str(saved_config["error_level"]),
        )
        raw_times_ms = 1.0e3 * np.asarray(loaded.table["t"], dtype=float)
        if isinstance(full_rate_values, str):
            raw_truth = scale * np.asarray(
                loaded.table[full_rate_values],
                dtype=float,
            )
        else:
            raw_truth = scale * np.asarray(
                loaded.states[full_rate_values],
                dtype=float,
            )
        full_bounds = panel_series["Full time-span"][0]
        in_full_span = (raw_times_ms >= full_bounds[0]) & (
            raw_times_ms <= full_bounds[-1]
        )
        full_rate_times_ms = raw_times_ms[in_full_span]
        full_rate_truth = raw_truth[in_full_span]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for ax, (name, _) in zip(axes.ravel(), panels, strict=True):
        panel_times, panel_truth, panel_prediction = panel_series[name]
        error = relative_rmse_pct(panel_prediction, panel_truth)
        if use_full_rate_reference:
            if full_rate_times_ms is None or full_rate_truth is None:
                raise RuntimeError("full-rate reference was not prepared")
            in_panel = (full_rate_times_ms >= panel_times[0]) & (
                full_rate_times_ms <= panel_times[-1]
            )
            reference_times = full_rate_times_ms[in_panel]
            reference_values = full_rate_truth[in_panel]
        else:
            reference_times = panel_times
            reference_values = panel_truth
        reference_label = reference_label_override or (
            "Original full-resolution data"
            if use_full_rate_reference
            else "Sampled true data"
        )
        ax.plot(
            reference_times,
            reference_values,
            color="black",
            linewidth=1.8,
            label=reference_label,
        )
        ax.plot(
            panel_times,
            panel_prediction,
            color="crimson",
            linestyle="-",
            linewidth=1.0,
            label=prediction_label,
        )
        ax.text(
            0.02,
            0.95,
            (
                "relative RMSE = "
                f"{error:.3f}%"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        ax.set_title(name)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    return _finish(fig, context, filename)


def plot_bar_fts_full_rollout(context: VisualizationContext | None = None):
    ctx = _context(context)
    return _plot_full_quantity(
        ctx,
        truth_key="true_fts",
        prediction_key="predicted_fts",
        component=None,
        scale=1.0e9,
        ylabel=r"$F_{ts}$ (nN)",
        filename="afm_param_stage2light_06a_bar_fts_full_rollout_grid.png",
        use_training_style_samples=True,
        use_full_rate_reference=True,
    )


def plot_bar_fts_full_rollout_from_w0(
    context: VisualizationContext | None = None,
):
    ctx = _context(context)
    return _plot_full_quantity(
        ctx,
        truth_key="true_fts",
        prediction_key="predicted_fts",
        component=None,
        scale=1.0e9,
        ylabel=r"$F_{ts}$ (nN)",
        filename="afm_param_stage2light_06a_bar_fts_full_rollout_from_W0_grid.png",
        use_training_style_samples=True,
        use_full_rate_reference=True,
        payload_builder=full_rollout_payload_from_w0,
        panels_builder=full_rollout_panels_from_w0,
    )


def plot_bar_fts_pointwise_full_resolution_from_w0(
    context: VisualizationContext | None = None,
):
    ctx = _context(context)
    return _plot_full_quantity(
        ctx,
        truth_key="true_fts",
        prediction_key="predicted_fts",
        component=None,
        scale=1.0e9,
        ylabel=r"$F_{ts}$ (nN)",
        filename=(
            "afm_param_stage2light_06a_bar_fts_pointwise_"
            "original_full_resolution_from_W0_grid.png"
        ),
        payload_builder=pointwise_full_resolution_payload_from_w0,
        panels_builder=full_rollout_panels_from_w0,
        reference_label_override="Original full-resolution data",
        prediction_label="Pointwise trained KAN output",
    )


def plot_bar_fts_pointwise_full_resolution(
    context: VisualizationContext | None = None,
):
    ctx = _context(context)
    return _plot_full_quantity(
        ctx,
        truth_key="true_fts",
        prediction_key="predicted_fts",
        component=None,
        scale=1.0e9,
        ylabel=r"$F_{ts}$ (nN)",
        filename=(
            "afm_param_stage2light_06a_bar_fts_pointwise_"
            "original_full_resolution_grid.png"
        ),
        payload_builder=pointwise_full_resolution_payload,
        panels_builder=full_rollout_panels,
        reference_label_override="Original full-resolution data",
        prediction_label="Pointwise trained KAN output",
    )


def plot_x1_full_rollout(context: VisualizationContext | None = None):
    ctx = _context(context)
    return _plot_full_quantity(
        ctx,
        truth_key="true_states",
        prediction_key="predicted_states",
        component=0,
        scale=1.0e9,
        ylabel=r"$x_1$ (nm)",
        filename="afm_param_stage2light_06a_x1_full_rollout_grid.png",
        use_training_style_samples=True,
        use_full_rate_reference=True,
        payload_builder=full_rollout_payload,
        panels_builder=full_rollout_panels,
    )


__all__ = [
    "plot_bar_fts_full_rollout",
    "plot_bar_fts_full_rollout_from_w0",
    "plot_bar_fts_pointwise_full_resolution",
    "plot_bar_fts_pointwise_full_resolution_from_w0",
    "plot_final_bar_fts",
    "plot_final_observables",
    "plot_grad_norm",
    "plot_losses",
    "plot_preopt_rank",
    "plot_reconstruction",
    "plot_state_bar_fts_losses",
    "plot_x1_full_rollout",
]
