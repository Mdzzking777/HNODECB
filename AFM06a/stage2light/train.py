"""AFM06a stage2light direct-warmstart training and entry diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np
import torch

from AFM06a.stage1pluslight.checkpoint import load_checkpoint, write_checkpoint_atomic
from AFM06a.stage1pluslight.kan_backend import state_dict_digest, training_force_normalizer
from AFM06a.stage2light.config import Stage2LightConfig
from AFM06a.stage2light.data import Stage1Endpoint, load_stage1_endpoint
from AFM06a.stage2light.kan_backend import restore_exact_stage1_model, trainable_parameter_manifest
from AFM06a.stage2light.losses import (
    evaluate_dynamics_residual_loss,
    evaluate_dynamics_residual_loss_with_terms,
    nan_loss_parts,
    parts_as_dict,
)
from AFM06a.stage2light.optim import LBFGS
_STAGE2_RESUME_CONTRACT_FIELDS = (
    "device",
    "dtype",
    "adam_lr",
    "adam_amsgrad",
    "lbfgs_lr",
    "lbfgs_max_iter",
    "lbfgs_max_eval",
    "lbfgs_history_size",
    "lbfgs_tolerance_grad",
    "lbfgs_tolerance_change",
    "lbfgs_ys_threshold",
    "lbfgs_sparsification_enabled",
    "lbfgs_sparsification_lambda",
    "lbfgs_sparsification_entropy_weight",
    "lbfgs_sparsification_epsilon",
    "prune_on_first_zero_step",
    "pruning_max_val_loss_increase_fraction",
    "pruning_max_val_loss_increase_absolute",
    "pruning_min_hidden_nodes",
    "lbfgs_zero_step_max_events",
    "gain_enabled",
    "gain_learnable",
    "soft_mask_enabled",
    "soft_mask_trainable",
    "adaptive_grid_enabled",
    "agu_update_every_adam_epoch",
    "entry_policy",
    "epoch0_preparation",
)


@dataclass(frozen=True)
class ForwardEvaluation:
    train_loss: torch.Tensor
    val_loss: torch.Tensor
    train_parts: dict[str, float]
    val_parts: dict[str, float]
    train_term_tensors: dict[str, torch.Tensor]
    trajectory: torch.Tensor | None
    predicted_x2dot: torch.Tensor | None
    predicted_fts: torch.Tensor
    predicted_bar_fts: torch.Tensor


def source_endpoint_digest(endpoint: Stage1Endpoint) -> str:
    """Fingerprint every saved st1pl condition that defines the st2l entry."""

    digest = hashlib.sha256()

    def update_array(name: str, values: Any) -> None:
        array = np.ascontiguousarray(np.asarray(values))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())

    for name, values in (
        ("window_source_indices", endpoint.window.source_idx),
        ("window_times", endpoint.window.times),
        ("train_idx", endpoint.window.train_idx),
        ("val_idx", endpoint.window.val_idx),
        ("state_mean", endpoint.warmstart["state_mean"]),
        ("state_scale", endpoint.warmstart["state_scale"]),
        ("initial_grid_support", endpoint.warmstart["initial_grid_support"]),
        ("initial_state", endpoint.warmstart["initial_state"]),
    ):
        update_array(name, values)

    saved_config = endpoint.payload.get("config", {})
    metadata = {
        "rank": int(endpoint.rank),
        "trial_id": int(endpoint.record["params"]["trial_id"]),
        "kan_state_dict_sha256": str(endpoint.warmstart["kan_state_dict_sha256"]),
        "input_policy": str(endpoint.warmstart.get("input_policy", "")),
        "force_output_policy": str(endpoint.warmstart["force_output_policy"]),
        "force_output_quantity": str(endpoint.warmstart.get("force_output_quantity", "")),
        "force_mean": float(endpoint.warmstart["force_mean"]),
        "force_scale": float(endpoint.warmstart["force_scale"]),
        "window_start_s": saved_config.get("window_start_s"),
        "window_stop_s": saved_config.get("window_stop_s"),
        "sample_stride": saved_config.get("sample_stride"),
        "sample_count": saved_config.get("sample_count"),
        "sampling_policy": saved_config.get("sampling_policy"),
        "transition_sampling_weight": saved_config.get("transition_sampling_weight"),
        "contact_sampling_weight": saved_config.get("contact_sampling_weight"),
        "noncontact_sampling_weight": saved_config.get("noncontact_sampling_weight"),
        "transition_half_width_s": saved_config.get("transition_half_width_s"),
        "smoothness_loss_weight": saved_config.get("smoothness_loss_weight"),
        "normalizer_policy": saved_config.get("normalizer_policy"),
        "loss_policy": saved_config.get("loss_policy"),
        "data_manifest": endpoint.payload.get("data_manifest"),
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _relative_rmse_pct(prediction: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(truth, dtype=float)
    if pred.size == 0 or ref.size == 0:
        return float("nan")
    valid = np.isfinite(pred) & np.isfinite(ref)
    if not np.any(valid):
        return float("nan")
    pred = pred[valid]
    ref = ref[valid]
    numerator = float(np.sqrt(np.mean(np.square(pred - ref))))
    denominator = max(float(np.sqrt(np.mean(np.square(ref)))), 1.0e-30)
    return 100.0 * numerator / denominator


def _bar_fts_reconstruction_metrics(
    evaluation: ForwardEvaluation,
    endpoint: Stage1Endpoint,
) -> tuple[float, float]:
    pred = evaluation.predicted_bar_fts.detach().cpu().numpy()
    truth = endpoint.true_bar_fts
    return (
        _relative_rmse_pct(pred[endpoint.window.train_idx], truth[endpoint.window.train_idx]),
        _relative_rmse_pct(pred[endpoint.window.val_idx], truth[endpoint.window.val_idx]),
    )


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    sums = {"total": 0.0, "kan": 0.0, "gain": 0.0, "soft_mask": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        square_sum = float(torch.sum(torch.square(parameter.grad.detach())).cpu())
        sums["total"] += square_sum
        if name.startswith("kan."):
            sums["kan"] += square_sum
        elif name == "log_gnn":
            sums["gain"] += square_sum
        elif name.startswith("soft_mask_"):
            sums["soft_mask"] += square_sum
    return {name: float(np.sqrt(max(value, 0.0))) for name, value in sums.items()}


def _make_lbfgs(
    config: Stage2LightConfig,
    parameters: list[torch.nn.Parameter],
) -> LBFGS:
    return LBFGS(
        parameters,
        lr=config.lbfgs_lr,
        max_iter=config.lbfgs_max_iter,
        max_eval=config.lbfgs_max_eval,
        tolerance_grad=config.lbfgs_tolerance_grad,
        tolerance_change=config.lbfgs_tolerance_change,
        history_size=config.lbfgs_history_size,
        line_search_fn="strong_wolfe",
        ys_threshold=config.lbfgs_ys_threshold,
    )


def _model_state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def _restore_model_state(
    model: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> None:
    model.load_state_dict(snapshot, strict=True)
    for parameter in model.parameters():
        parameter.grad = None


def _default_lbfgs_zero_step_state() -> dict[str, Any]:
    return {
        "event_count": 0,
        "pruning_attempted": False,
        "pruning_performed": False,
        "pruning_epoch": 0,
        "pruned_hidden_nodes": [],
        "pruning_node_scores": [],
        "pruning_baseline_val_loss": float("nan"),
        "pruning_candidate_val_loss": float("nan"),
        "pruning_rejection_reason": "",
    }


def _normalized_lbfgs_zero_step_state(value: Any) -> dict[str, Any]:
    state = _default_lbfgs_zero_step_state()
    if isinstance(value, dict):
        for name in state:
            if name in value:
                state[name] = value[name]
    state["event_count"] = max(0, int(state["event_count"]))
    state["pruning_attempted"] = bool(state["pruning_attempted"])
    state["pruning_performed"] = bool(state["pruning_performed"])
    state["pruning_epoch"] = max(0, int(state["pruning_epoch"]))
    state["pruned_hidden_nodes"] = [
        int(index) for index in state["pruned_hidden_nodes"]
    ]
    state["pruning_node_scores"] = [
        float(score) for score in state["pruning_node_scores"]
    ]
    state["pruning_baseline_val_loss"] = float(state["pruning_baseline_val_loss"])
    state["pruning_candidate_val_loss"] = float(state["pruning_candidate_val_loss"])
    state["pruning_rejection_reason"] = str(state["pruning_rejection_reason"])
    return state


def _lbfgs_zero_step_action(
    event_count: int,
    max_events: int = 3,
    *,
    pruning_enabled: bool = True,
) -> str:
    if event_count <= 0:
        raise ValueError("event_count must be positive")
    if max_events != 3:
        raise ValueError("AFM06a stage2light requires exactly three hard-zero-step events")
    if event_count == 1:
        return "prune_and_restart" if pruning_enabled else "restart_only"
    if event_count == 2:
        return "restart_only"
    return "early_stop"


def _kan_hidden_layers(model: torch.nn.Module) -> tuple[Any, Any]:
    kan = getattr(model, "kan", None)
    layers = getattr(kan, "act_fun", None)
    if layers is None or len(layers) != 2:
        raise RuntimeError(
            "AFM06a automatic sparsification/pruning currently requires one hidden KAN layer"
        )
    incoming, outgoing = layers
    if int(incoming.out_dim) != int(outgoing.in_dim):
        raise RuntimeError("AFM06a KAN hidden-layer dimensions are inconsistent")
    return incoming, outgoing


def _kan_active_hidden_nodes(model: torch.nn.Module) -> list[int]:
    incoming, outgoing = _kan_hidden_layers(model)
    incoming_active = torch.any(incoming.mask.detach() != 0.0, dim=0)
    outgoing_active = torch.any(outgoing.mask.detach() != 0.0, dim=1)
    active = torch.nonzero(incoming_active & outgoing_active, as_tuple=False).reshape(-1)
    return [int(index) for index in active.cpu().tolist()]


def _kan_hidden_node_groups(model: torch.nn.Module) -> list[list[int]]:
    incoming, _ = _kan_hidden_layers(model)
    return [list(range(int(incoming.out_dim)))]


def _kan_edge_strengths(layer: Any, epsilon: float) -> torch.Tensor:
    coefficient_rms = torch.sqrt(
        torch.mean(torch.square(layer.coef), dim=-1) + float(epsilon) ** 2
    )
    spline_strength = layer.scale_sp * coefficient_rms
    strength = torch.sqrt(
        torch.square(layer.scale_base)
        + torch.square(spline_strength)
        + float(epsilon) ** 2
    )
    return strength * layer.mask


def _kan_hidden_node_strengths(
    model: torch.nn.Module,
    *,
    epsilon: float,
) -> torch.Tensor:
    incoming, outgoing = _kan_hidden_layers(model)
    incoming_strength = _kan_edge_strengths(incoming, epsilon)
    outgoing_strength = _kan_edge_strengths(outgoing, epsilon)
    return torch.sqrt(
        torch.sum(torch.square(incoming_strength), dim=0)
        + torch.sum(torch.square(outgoing_strength), dim=1)
        + float(epsilon) ** 2
    )


def _kan_node_sparsification_penalty(
    model: torch.nn.Module,
    *,
    entropy_weight: float,
    epsilon: float,
) -> torch.Tensor:
    strengths = _kan_hidden_node_strengths(model, epsilon=epsilon)
    active_nodes = _kan_active_hidden_nodes(model)
    if not active_nodes:
        return torch.zeros((), dtype=strengths.dtype, device=strengths.device)
    active_set = set(active_nodes)
    group_penalties: list[torch.Tensor] = []
    for group in _kan_hidden_node_groups(model):
        group_active = [index for index in group if index in active_set]
        if not group_active:
            continue
        active_index = torch.as_tensor(
            group_active,
            dtype=torch.long,
            device=strengths.device,
        )
        active_strengths = torch.index_select(strengths, 0, active_index)
        l1_term = torch.sum(active_strengths)
        probabilities = active_strengths / (
            torch.sum(active_strengths) + float(epsilon)
        )
        entropy = -torch.sum(
            probabilities * torch.log(probabilities + float(epsilon))
        )
        group_penalties.append(l1_term + float(entropy_weight) * entropy)
    if not group_penalties:
        return torch.zeros((), dtype=strengths.dtype, device=strengths.device)
    return torch.mean(torch.stack(group_penalties))


def _select_prunable_hidden_node(
    node_scores: list[float],
    active_nodes: list[int],
    *,
    min_hidden_nodes: int,
    groups: list[list[int]] | None = None,
) -> int | None:
    if groups is None:
        groups = [list(active_nodes)]
    active_set = set(active_nodes)
    candidates: list[tuple[float, int]] = []
    for group in groups:
        group_active = [index for index in group if index in active_set]
        if len(group_active) <= min_hidden_nodes:
            continue
        finite_scores = [
            float(node_scores[index])
            for index in group_active
            if index < len(node_scores) and np.isfinite(node_scores[index])
        ]
        scale = sum(abs(score) for score in finite_scores)
        if scale <= 0.0:
            scale = 1.0
        for index in group_active:
            if index < len(node_scores) and np.isfinite(node_scores[index]):
                candidates.append((float(node_scores[index]) / scale, int(index)))
    return None if not candidates else min(candidates)[1]


@torch.no_grad()
def _mask_hidden_node(model: torch.nn.Module, hidden_index: int) -> None:
    incoming, outgoing = _kan_hidden_layers(model)
    if hidden_index < 0 or hidden_index >= int(incoming.out_dim):
        raise IndexError(f"hidden node index is out of range: {hidden_index}")
    incoming.mask[:, hidden_index] = 0.0
    outgoing.mask[hidden_index, :] = 0.0
    symbolic_layers = getattr(model.kan, "symbolic_fun", None)
    if symbolic_layers is not None and len(symbolic_layers) == 2:
        symbolic_layers[0].mask[hidden_index, :] = 0.0
        symbolic_layers[1].mask[:, hidden_index] = 0.0


def _trainable_parameter_delta_stats(
    model: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> tuple[float, float]:
    square_sum = 0.0
    max_abs = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        before = snapshot[name].to(device=parameter.device, dtype=parameter.dtype)
        delta = parameter.detach() - before
        square_sum += float(torch.sum(torch.square(delta)).cpu())
        if delta.numel() > 0:
            max_abs = max(max_abs, float(torch.max(torch.abs(delta)).cpu()))
    return float(np.sqrt(max(square_sum, 0.0))), float(max_abs)


def _loss_term_gradient_norms(
    model: torch.nn.Module,
    term_tensors: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Return ||d L_term / d theta||_2 over every trainable parameter."""

    names = ("dynamics_residual", "smoothness")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    terms = torch.stack(tuple(term_tensors[name] for name in names))
    gradients = torch.autograd.grad(
        terms,
        parameters,
        grad_outputs=torch.eye(len(names), dtype=terms.dtype, device=terms.device),
        retain_graph=True,
        allow_unused=True,
        is_grads_batched=True,
    )
    square_sums = torch.zeros(len(names), dtype=terms.dtype, device=terms.device)
    for gradient in gradients:
        if gradient is not None:
            square_sums = square_sums + torch.sum(
                torch.square(gradient.detach().reshape(len(names), -1)),
                dim=1,
            )
    values = torch.sqrt(torch.clamp(square_sums, min=0.0)).cpu().tolist()
    return dict(zip(names, (float(value) for value in values), strict=True))


def _log_line(log_handle, message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    log_handle.write(line + "\n")
    log_handle.flush()


def _saved_rollout_settings(endpoint: Stage1Endpoint) -> tuple[str, float, float, float, int]:
    saved = endpoint.payload.get("config")
    if not isinstance(saved, dict):
        raise RuntimeError("st1pl payload is missing its rollout configuration")
    return (
        str(saved["ode_method"]),
        float(saved["ode_rtol"]),
        float(saved["ode_atol"]),
        float(saved["state_guard_multiplier"]),
        int(saved["ode_step_budget"]),
    )


def _saved_smoothness_weight(endpoint: Stage1Endpoint) -> float:
    saved = endpoint.payload.get("config")
    if not isinstance(saved, dict):
        raise RuntimeError("st1pl payload is missing its loss configuration")
    return float(saved["smoothness_loss_weight"])


def _forward_evaluation(
    model: torch.nn.Module,
    endpoint: Stage1Endpoint,
    *,
    include_rollout: bool = False,
) -> ForwardEvaluation:
    if include_rollout:
        raise RuntimeError(
            "AFM06a st1pl/st2l no longer perform internal rollout; "
            "use explicit rollout visualization scripts outside training."
        )
    device = next(model.parameters()).device
    dtype = torch.float64
    window = endpoint.window
    true_states = torch.as_tensor(window.states, dtype=dtype, device=device)
    times = torch.as_tensor(window.times, dtype=dtype, device=device)
    true_x2dot = torch.as_tensor(window.x2dot, dtype=dtype, device=device)
    train_idx = torch.as_tensor(window.train_idx, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(window.val_idx, dtype=torch.long, device=device)
    settings = window.settings
    force_states = torch.stack(
        (true_states[0], torch.zeros_like(true_states[0])),
        dim=1,
    )
    predicted_fts = model(force_states)
    _k, _omega0, mass_kg, _c, _fd, _ca, _ch, _dist, _a0, _beta = settings.parameter_vector
    predicted_bar_fts = predicted_fts / float(mass_kg)
    smoothness_weight = _saved_smoothness_weight(endpoint)
    _residual_mean, residual_scale_value = training_force_normalizer(
        window.bar_fts_rhs_residual[window.train_idx]
    )
    residual_scale = torch.as_tensor(residual_scale_value, dtype=dtype, device=device)
    trajectory: torch.Tensor | None = None
    predicted_x2dot: torch.Tensor | None = None
    train_loss, train_parts, train_term_tensors = (
        evaluate_dynamics_residual_loss_with_terms(
            predicted_fts=predicted_fts,
            true_states=true_states,
            true_x2dot=true_x2dot,
            times=times,
            indices=train_idx,
            residual_scale=residual_scale,
            predicted_states=trajectory,
            settings=settings,
            smoothness_weight=smoothness_weight,
        )
    )
    if val_idx.numel() > 0:
        val_loss, val_parts = evaluate_dynamics_residual_loss(
            predicted_fts=predicted_fts,
            true_states=true_states,
            true_x2dot=true_x2dot,
            times=times,
            indices=val_idx,
            residual_scale=residual_scale,
            predicted_states=trajectory,
            settings=settings,
            smoothness_weight=smoothness_weight,
        )
    else:
        val_loss = torch.as_tensor(float("nan"), dtype=dtype, device=device)
        val_parts = nan_loss_parts()
    return ForwardEvaluation(
        train_loss=train_loss,
        val_loss=val_loss,
        train_parts=parts_as_dict(train_parts),
        val_parts=parts_as_dict(val_parts),
        train_term_tensors=train_term_tensors,
        trajectory=trajectory,
        predicted_x2dot=predicted_x2dot,
        predicted_fts=predicted_fts,
        predicted_bar_fts=predicted_bar_fts,
    )


def _hidden_node_attribution_scores(
    model: torch.nn.Module,
    observed_states: torch.Tensor,
) -> list[float]:
    _kan_hidden_layers(model)
    kan = model.kan
    previous_save_act = bool(kan.save_act)
    try:
        kan.save_act = True
        with torch.no_grad():
            model.raw_output(observed_states.transpose(0, 1))
        kan.attribute(plot=False)
        scores = kan.node_scores[1].detach().cpu().reshape(-1)
        return [float(value) for value in scores.tolist()]
    finally:
        kan.save_act = previous_save_act


def _attempt_one_time_automatic_pruning(
    *,
    config: Stage2LightConfig,
    model: torch.nn.Module,
    endpoint: Stage1Endpoint,
    epoch: int,
) -> dict[str, Any]:
    report = _default_lbfgs_zero_step_state()
    report["pruning_attempted"] = True
    report["pruning_epoch"] = int(epoch)
    if not config.prune_on_first_zero_step:
        report["pruning_rejection_reason"] = "disabled_by_configuration"
        return report

    snapshot = _model_state_snapshot(model)
    was_training = bool(model.training)
    try:
        model.eval()
        with torch.no_grad():
            baseline = _forward_evaluation(model, endpoint, include_rollout=False)
        baseline_val = float(baseline.val_loss.detach().cpu())
        report["pruning_baseline_val_loss"] = baseline_val
        node_scores = _hidden_node_attribution_scores(
            model,
            torch.as_tensor(
                endpoint.window.states,
                dtype=torch.float64,
                device=next(model.parameters()).device,
            ),
        )
        report["pruning_node_scores"] = node_scores
        active_nodes = _kan_active_hidden_nodes(model)
        selected = _select_prunable_hidden_node(
            node_scores,
            active_nodes,
            min_hidden_nodes=config.pruning_min_hidden_nodes,
            groups=_kan_hidden_node_groups(model),
        )
        if selected is None:
            report["pruning_rejection_reason"] = (
                "no_finite_prunable_hidden_node_or_minimum_width_reached"
            )
            return report

        _mask_hidden_node(model, selected)
        with torch.no_grad():
            candidate = _forward_evaluation(model, endpoint, include_rollout=False)
        candidate_val = float(candidate.val_loss.detach().cpu())
        report["pruning_candidate_val_loss"] = candidate_val
        accepted_limit = (
            baseline_val * (1.0 + config.pruning_max_val_loss_increase_fraction)
            + config.pruning_max_val_loss_increase_absolute
        )
        if not np.isfinite(candidate_val):
            _restore_model_state(model, snapshot)
            report["pruning_rejection_reason"] = "nonfinite_validation_loss"
            return report
        if candidate_val > accepted_limit:
            _restore_model_state(model, snapshot)
            report["pruning_rejection_reason"] = (
                "validation_loss_increase_exceeded_limit"
            )
            return report

        report["pruning_performed"] = True
        report["pruned_hidden_nodes"] = [int(selected) + 1]
        return report
    except Exception as error:
        _restore_model_state(model, snapshot)
        report["pruning_rejection_reason"] = (
            f"automatic_pruning_error:{type(error).__name__}:{error}"
        )
        return report
    finally:
        model.train(was_training)


def inherited_epoch0_record(endpoint: Stage1Endpoint) -> dict[str, Any]:
    """Return metadata only; this function performs no model evaluation."""

    force_train = _relative_rmse_pct(
        np.asarray(endpoint.record["predicted_bar_fts"])[endpoint.window.train_idx],
        endpoint.true_bar_fts[endpoint.window.train_idx],
    )
    force_val = _relative_rmse_pct(
        np.asarray(endpoint.record["predicted_bar_fts"])[endpoint.window.val_idx],
        endpoint.true_bar_fts[endpoint.window.val_idx],
    )
    return {
        "epoch": 0,
        "phase": "inherited_st1pl_endpoint",
        "train_loss": float(endpoint.record["train_loss"]),
        "val_loss": float(endpoint.record["val_loss"]),
        "loss_parts": dict(endpoint.record.get("loss_parts", {})),
        "val_loss_parts": dict(endpoint.record.get("val_loss_parts", {})),
        "entry_policy": "direct_complete_st1pl_warmstart",
        "epoch0_preparation_performed": False,
        "bar_fts_rec_train_pct": force_train,
        "bar_fts_rec_val_pct": force_val,
        "grad_norm_total": float("nan"),
        "grad_norm_kan": float("nan"),
        "grad_norm_gain": float("nan"),
        "grad_norm_soft_mask": float("nan"),
        "grad_norm_loss_dynamics_residual": float("nan"),
        "grad_norm_loss_smoothness": float("nan"),
    }


def _update_agu_from_observed_training_states(
    model: torch.nn.Module,
    endpoint: Stage1Endpoint,
) -> float:
    """Update AGU from fixed observed x1/x2 samples inside an Adam epoch."""

    device = next(model.parameters()).device
    states = endpoint.window.states[:, endpoint.window.train_idx].T
    states_t = torch.as_tensor(states, dtype=torch.float64, device=device)
    started = perf_counter()
    with torch.no_grad():
        model.update_grid_from_states(states_t)
    return float(perf_counter() - started)


def replay_stage1_endpoint(config: Stage2LightConfig) -> dict[str, Any]:
    """Probe A: restore and replay st1pl without changing normalizer/grid/gain."""

    config.validate()
    endpoint = load_stage1_endpoint(config)
    model = restore_exact_stage1_model(config, endpoint)
    digest_before = state_dict_digest(model.frozen_state_dict())
    model.eval()
    with torch.no_grad():
        evaluation = _forward_evaluation(model, endpoint, include_rollout=False)
    digest_after = state_dict_digest(model.frozen_state_dict())
    saved_force = np.asarray(endpoint.record["predicted_bar_fts"], dtype=float)
    current_force = evaluation.predicted_bar_fts.detach().cpu().numpy()
    report = {
        "input_rank": endpoint.rank,
        "trial_id": int(endpoint.record["params"]["trial_id"]),
        "nn_init_seed": int(endpoint.record["params"]["nn_init_seed"]),
        "saved_train_loss": float(endpoint.record["train_loss"]),
        "replayed_train_loss": float(evaluation.train_loss.detach().cpu()),
        "train_loss_abs_delta": abs(float(evaluation.train_loss.detach().cpu()) - float(endpoint.record["train_loss"])),
        "saved_val_loss": float(endpoint.record["val_loss"]),
        "replayed_val_loss": float(evaluation.val_loss.detach().cpu()),
        "val_loss_abs_delta": abs(float(evaluation.val_loss.detach().cpu()) - float(endpoint.record["val_loss"])),
        "bar_fts_max_abs_delta": float(np.max(np.abs(current_force - saved_force))),
        "state_dict_sha256_before": digest_before,
        "state_dict_sha256_after": digest_after,
        "state_dict_unchanged": digest_before == digest_after,
        "entry_policy": config.entry_policy,
        "epoch0_preparation_performed": False,
        "rollout_performed": False,
        "trainable_parameters": trainable_parameter_manifest(model),
    }
    if not report["state_dict_unchanged"]:
        raise RuntimeError("entry replay mutated the AFM06a st1pl model state")
    return report


def backward_smoke(config: Stage2LightConfig) -> dict[str, Any]:
    """Run one forward/backward pass without taking an optimizer step."""

    config.validate()
    endpoint = load_stage1_endpoint(config)
    model = restore_exact_stage1_model(config, endpoint)
    digest_before = state_dict_digest(model.frozen_state_dict())
    model.zero_grad(set_to_none=True)
    evaluation = _forward_evaluation(model, endpoint, include_rollout=False)
    evaluation.train_loss.backward()
    grad_norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            grad_norms[name] = 0.0
        else:
            if not bool(torch.all(torch.isfinite(parameter.grad))):
                raise RuntimeError(f"nonfinite gradient in {name}")
            grad_norms[name] = float(torch.linalg.vector_norm(parameter.grad.detach()).cpu())
    digest_after = state_dict_digest(model.frozen_state_dict())
    if digest_before != digest_after:
        raise RuntimeError("backward-only smoke unexpectedly changed model parameters")
    return {
        "input_rank": endpoint.rank,
        "train_loss": float(evaluation.train_loss.detach().cpu()),
        "val_loss": float(evaluation.val_loss.detach().cpu()),
        "gradient_norms": grad_norms,
        "nonzero_gradient_parameter_count": sum(value > 0.0 for value in grad_norms.values()),
        "state_dict_unchanged": True,
        "optimizer_step_performed": False,
        "epoch0_preparation_performed": False,
    }


def _checkpoint_payload(
    *,
    config: Stage2LightConfig,
    endpoint: Stage1Endpoint,
    model: torch.nn.Module,
    adam: torch.optim.Adam,
    lbfgs: LBFGS,
    epoch: int,
    history: list[dict[str, Any]],
    lbfgs_zero_step_state: dict[str, Any],
    stop_kind: str = "",
    stop_epoch: int = 0,
    stop_reason: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "stage": "AFM06a_stage2light",
        "input_rank": endpoint.rank,
        "source_trial_id": int(endpoint.record["params"]["trial_id"]),
        "source_state_dict_sha256": str(endpoint.warmstart["kan_state_dict_sha256"]),
        "source_endpoint_sha256": source_endpoint_digest(endpoint),
        "entry_policy": config.entry_policy,
        "epoch0_preparation_performed": False,
        "epoch": int(epoch),
        "history": history,
        "stop_kind": str(stop_kind),
        "stop_epoch": int(stop_epoch),
        "stop_reason": str(stop_reason),
        "lbfgs_zero_step_state": _normalized_lbfgs_zero_step_state(
            lbfgs_zero_step_state
        ),
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "adam_state_dict": adam.state_dict(),
        "lbfgs_state_dict": lbfgs.state_dict(),
        "config": asdict(config),
    }


def _restore_training_checkpoint(
    *,
    config: Stage2LightConfig,
    endpoint: Stage1Endpoint,
    model: torch.nn.Module,
    adam: torch.optim.Adam,
    lbfgs: LBFGS,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    if not config.resume_enabled:
        return 0, [inherited_epoch0_record(endpoint)], {
            "lbfgs_zero_step_state": _default_lbfgs_zero_step_state(),
        }
    payload = load_checkpoint(config.checkpoint_path)
    if payload is None:
        return 0, [inherited_epoch0_record(endpoint)], {
            "lbfgs_zero_step_state": _default_lbfgs_zero_step_state(),
        }
    if int(payload.get("input_rank", -1)) != endpoint.rank:
        raise RuntimeError("stage2light checkpoint belongs to a different st1pl rank")
    if payload.get("source_state_dict_sha256") != endpoint.warmstart["kan_state_dict_sha256"]:
        raise RuntimeError("stage2light checkpoint source no longer matches the selected st1pl endpoint")
    saved_endpoint_digest = payload.get("source_endpoint_sha256")
    if not isinstance(saved_endpoint_digest, str):
        raise RuntimeError(
            "stage2light checkpoint lacks the complete source-endpoint fingerprint; "
            "archive or clear this legacy checkpoint before resuming"
        )
    if saved_endpoint_digest != source_endpoint_digest(endpoint):
        raise RuntimeError(
            "stage2light checkpoint belongs to a different st1pl window, split, "
            "normalizer, support, force transform, or source dataset"
        )
    if payload.get("entry_policy") != config.entry_policy or bool(payload.get("epoch0_preparation_performed")):
        raise RuntimeError("stage2light checkpoint violates the AFM06a direct-entry contract")
    saved_config = payload.get("config")
    if not isinstance(saved_config, dict):
        raise RuntimeError("stage2light checkpoint is missing its optimizer schedule")
    current_config = asdict(config)
    for name in _STAGE2_RESUME_CONTRACT_FIELDS:
        if saved_config.get(name) != current_config.get(name):
            raise RuntimeError(
                f"stage2light checkpoint contract mismatch for {name}; "
                "archive or clear the checkpoint before changing this setting"
            )
    saved_epochs = int(saved_config.get("epochs", -1))
    saved_adam_epochs = int(saved_config.get("adam_epochs", -1))
    saved_lbfgs_epochs = int(saved_config.get("lbfgs_epochs", -1))
    saved_schedule = (saved_epochs, saved_adam_epochs, saved_lbfgs_epochs)
    current_schedule = (config.epochs, config.adam_epochs, config.lbfgs_epochs)
    schedule_is_valid = (
        saved_epochs > 0
        and saved_adam_epochs >= 0
        and saved_lbfgs_epochs >= 0
        and saved_adam_epochs + saved_lbfgs_epochs == saved_epochs
    )
    schedule_keeps_phase_boundary = (
        schedule_is_valid
        and config.adam_epochs == saved_adam_epochs
        and config.lbfgs_epochs == config.epochs - config.adam_epochs
    )
    checkpoint_epoch = int(payload.get("epoch", -1))
    checkpoint_fits_current_schedule = (
        0 <= checkpoint_epoch <= saved_epochs
        and checkpoint_epoch <= config.epochs
    )
    if not schedule_keeps_phase_boundary or not checkpoint_fits_current_schedule:
        raise RuntimeError(
            "stage2light checkpoint optimizer phases are incompatible with the current schedule: "
            f"saved={saved_schedule}, current={current_schedule}. "
            "The total/L-BFGS tail may be shortened or extended only when the Adam "
            "phase boundary is unchanged and the new total is not below the "
            "checkpoint epoch."
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    adam.load_state_dict(payload["adam_state_dict"])
    lbfgs.load_state_dict(payload["lbfgs_state_dict"])
    stop_state = {
        "stop_kind": str(payload.get("stop_kind", "")),
        "stop_epoch": int(payload.get("stop_epoch", 0)),
        "stop_reason": str(payload.get("stop_reason", "")),
        "lbfgs_zero_step_state": _normalized_lbfgs_zero_step_state(
            payload.get("lbfgs_zero_step_state")
        ),
    }
    return int(payload["epoch"]), list(payload["history"]), stop_state


def run_stage2light(config: Stage2LightConfig) -> dict[str, Any]:
    config.validate()
    config.result_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.visualization_dir.mkdir(parents=True, exist_ok=True)
    endpoint = load_stage1_endpoint(config)
    model = restore_exact_stage1_model(config, endpoint)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("AFM06a stage2light has no trainable parameters")
    adam = torch.optim.Adam(parameters, lr=config.adam_lr, amsgrad=config.adam_amsgrad)
    lbfgs = _make_lbfgs(config, parameters)
    start_epoch, history, restored_stop = _restore_training_checkpoint(
        config=config,
        endpoint=endpoint,
        model=model,
        adam=adam,
        lbfgs=lbfgs,
    )
    stop_kind = str(restored_stop.get("stop_kind", ""))
    stop_epoch = int(restored_stop.get("stop_epoch", 0))
    stop_reason = str(restored_stop.get("stop_reason", ""))
    lbfgs_zero_step_state = _normalized_lbfgs_zero_step_state(
        restored_stop.get("lbfgs_zero_step_state")
    )
    started = perf_counter()
    log_mode = "a" if start_epoch > 0 and config.run_log_path.is_file() else "w"
    with config.run_log_path.open(log_mode, encoding="utf-8") as log_handle:
        _log_line(
            log_handle,
            "AFM06a stage2light start | "
            f"rank={endpoint.rank} trial={endpoint.record['params']['trial_id']} "
            f"resume_epoch={start_epoch} epochs={config.epochs} "
            "epoch0_preparation=OFF AGU=every_Adam_epoch "
            "loss=force_scale_normalized_dynamics_plus_residual_smoothness "
            "KAN_input=observed_x1_at_every_time "
            f"lbfgs_sparsification={'ON' if config.lbfgs_sparsification_enabled else 'OFF'} "
            f"zero_step_events={lbfgs_zero_step_state['event_count']}/"
            f"{config.lbfgs_zero_step_max_events}",
        )
        if stop_kind:
            _log_line(
                log_handle,
                "AFM06a stage2light terminal checkpoint restored | "
                f"stop_epoch={stop_epoch} stop_kind={stop_kind} stop_reason={stop_reason}",
            )
        for epoch in (range(start_epoch + 1, config.epochs + 1) if not stop_kind else ()):
            phase = "adam" if epoch <= config.adam_epochs else "lbfgs_strong_wolfe"
            record_loss_term_gradients = (
                epoch % config.loss_term_grad_every == 0 or epoch == config.epochs
            )
            agu_updated = False
            agu_update_sec = 0.0
            grad_norms = {"total": float("nan"), "kan": float("nan"), "gain": float("nan"), "soft_mask": float("nan")}
            loss_term_grad_norms = {
                "dynamics_residual": float("nan"),
                "smoothness": float("nan"),
            }
            lbfgs_fresh_restart_used = False
            lbfgs_pruning_attempted_this_epoch = False
            lbfgs_pruning_performed_this_epoch = False
            lbfgs_sparsification_penalty = float("nan")
            lbfgs_param_delta_l2 = float("nan")
            lbfgs_param_delta_max_abs = float("nan")
            if phase == "adam":
                if config.adaptive_grid_enabled and config.agu_update_every_adam_epoch:
                    agu_update_sec = _update_agu_from_observed_training_states(model, endpoint)
                    agu_updated = True
                adam.zero_grad(set_to_none=True)
                evaluation = _forward_evaluation(
                    model,
                    endpoint,
                    include_rollout=False,
                )
                if record_loss_term_gradients:
                    loss_term_grad_norms = _loss_term_gradient_norms(model, evaluation.train_term_tensors)
                evaluation.train_loss.backward()
                grad_norms = _gradient_norms(model)
                adam.step()
            else:
                lbfgs_start_model_state = _model_state_snapshot(model)
                while True:
                    def closure() -> torch.Tensor:
                        lbfgs.zero_grad(set_to_none=True)
                        closure_evaluation = _forward_evaluation(
                            model,
                            endpoint,
                            include_rollout=False,
                        )
                        objective = closure_evaluation.train_loss
                        if config.lbfgs_sparsification_enabled:
                            objective = objective + (
                                config.lbfgs_sparsification_lambda
                                * _kan_node_sparsification_penalty(
                                    model,
                                    entropy_weight=(
                                        config.lbfgs_sparsification_entropy_weight
                                    ),
                                    epsilon=config.lbfgs_sparsification_epsilon,
                                )
                            )
                        objective.backward()
                        return objective

                    lbfgs.step(closure)
                    lbfgs_param_delta_l2, lbfgs_param_delta_max_abs = _trainable_parameter_delta_stats(
                        model,
                        lbfgs_start_model_state,
                    )
                    hard_zero_step = (
                        np.isfinite(lbfgs_param_delta_max_abs)
                        and lbfgs_param_delta_max_abs <= config.lbfgs_tolerance_change
                    )
                    if not hard_zero_step:
                        break

                    _restore_model_state(model, lbfgs_start_model_state)
                    event_count = int(lbfgs_zero_step_state["event_count"]) + 1
                    lbfgs_zero_step_state["event_count"] = event_count
                    action = _lbfgs_zero_step_action(
                        event_count,
                        max_events=config.lbfgs_zero_step_max_events,
                        pruning_enabled=config.prune_on_first_zero_step,
                    )
                    if action == "prune_and_restart":
                        pruning_report = _attempt_one_time_automatic_pruning(
                            config=config,
                            model=model,
                            endpoint=endpoint,
                            epoch=epoch,
                        )
                        for name, value in pruning_report.items():
                            if name != "event_count":
                                lbfgs_zero_step_state[name] = value
                        lbfgs_pruning_attempted_this_epoch = True
                        lbfgs_pruning_performed_this_epoch = bool(
                            pruning_report["pruning_performed"]
                        )
                        lbfgs = _make_lbfgs(config, parameters)
                        lbfgs_fresh_restart_used = True
                        lbfgs_start_model_state = _model_state_snapshot(model)
                        _log_line(
                            log_handle,
                            "LBFGS hard zero-step event 1/3 -- "
                            f"epoch={epoch} param_delta_max_abs={lbfgs_param_delta_max_abs:.3e} "
                            f"<= tol_change={config.lbfgs_tolerance_change:.1e}; "
                            f"automatic_pruning={'ACCEPTED' if pruning_report['pruning_performed'] else 'NOT_APPLIED'} "
                            f"pruned_hidden_nodes={pruning_report['pruned_hidden_nodes']} "
                            f"reason={pruning_report['pruning_rejection_reason'] or 'validation_accepted'}; "
                            "clear history and retry the same epoch",
                        )
                        continue
                    if action == "restart_only":
                        lbfgs = _make_lbfgs(config, parameters)
                        lbfgs_fresh_restart_used = True
                        lbfgs_start_model_state = _model_state_snapshot(model)
                        _log_line(
                            log_handle,
                            f"LBFGS hard zero-step event {event_count}/3 -- "
                            f"epoch={epoch} param_delta_max_abs={lbfgs_param_delta_max_abs:.3e} "
                            f"<= tol_change={config.lbfgs_tolerance_change:.1e}; "
                            "clear history without pruning and retry the same epoch",
                        )
                        continue

                    stop_epoch = int(epoch)
                    stop_kind = "lbfgs_hard_zero_step_stop"
                    stop_reason = "lbfgs_hard_zero_step_third_event"
                    _log_line(
                        log_handle,
                        "LBFGS hard zero-step event 3/3 -- early stop "
                        f"| epoch={epoch} "
                        f"| param_delta_max_abs={lbfgs_param_delta_max_abs:.3e} "
                        f"<= tol_change={config.lbfgs_tolerance_change:.1e} "
                        f"| reason={stop_reason}; rollback parameters and stop",
                    )
                    break

                if stop_kind:
                    completed_epoch = int(history[-1].get("epoch", 0)) if history else 0
                    write_checkpoint_atomic(
                        config.checkpoint_path,
                        _checkpoint_payload(
                            config=config,
                            endpoint=endpoint,
                            model=model,
                            adam=adam,
                            lbfgs=lbfgs,
                            epoch=completed_epoch,
                            history=history,
                            lbfgs_zero_step_state=lbfgs_zero_step_state,
                            stop_kind=stop_kind,
                            stop_epoch=stop_epoch,
                            stop_reason=stop_reason,
                        ),
                    )
                    break
                grad_norms = _gradient_norms(model)
                if record_loss_term_gradients:
                    term_evaluation = _forward_evaluation(
                        model,
                        endpoint,
                        include_rollout=False,
                    )
                    loss_term_grad_norms = _loss_term_gradient_norms(model, term_evaluation.train_term_tensors)
                if config.lbfgs_sparsification_enabled:
                    with torch.no_grad():
                        lbfgs_sparsification_penalty = float(
                            _kan_node_sparsification_penalty(
                                model,
                                entropy_weight=(
                                    config.lbfgs_sparsification_entropy_weight
                                ),
                                epsilon=config.lbfgs_sparsification_epsilon,
                            ).cpu()
                        )
            model.eval()
            with torch.no_grad():
                evaluation = _forward_evaluation(
                    model,
                    endpoint,
                    include_rollout=False,
                )
            model.train()
            force_rec_train, force_rec_val = _bar_fts_reconstruction_metrics(evaluation, endpoint)
            row = {
                "epoch": epoch,
                "phase": phase,
                "train_loss": float(evaluation.train_loss.detach().cpu()),
                "val_loss": float(evaluation.val_loss.detach().cpu()),
                "loss_parts": evaluation.train_parts,
                "val_loss_parts": evaluation.val_parts,
                "bar_fts_rec_train_pct": force_rec_train,
                "bar_fts_rec_val_pct": force_rec_val,
                "grad_norm_total": grad_norms["total"],
                "grad_norm_kan": grad_norms["kan"],
                "grad_norm_gain": grad_norms["gain"],
                "grad_norm_soft_mask": grad_norms["soft_mask"],
                "grad_norm_loss_dynamics_residual": loss_term_grad_norms[
                    "dynamics_residual"
                ],
                "grad_norm_loss_smoothness": loss_term_grad_norms["smoothness"],
                "gain": float(model.gain().detach().cpu()),
                "soft_mask_s0": float(model.soft_mask_s0().detach().cpu()),
                "soft_mask_alpha": float(model.soft_mask_alpha().detach().cpu()),
                "agu_updated": bool(agu_updated),
                "agu_update_source": "observed_training_x1_x2" if agu_updated else "none",
                "agu_update_sec": float(agu_update_sec),
                "lbfgs_fresh_restart_used": bool(lbfgs_fresh_restart_used),
                "lbfgs_zero_step_event_count": int(
                    lbfgs_zero_step_state["event_count"]
                ),
                "lbfgs_pruning_attempted_this_epoch": bool(
                    lbfgs_pruning_attempted_this_epoch
                ),
                "lbfgs_pruning_performed_this_epoch": bool(
                    lbfgs_pruning_performed_this_epoch
                ),
                "lbfgs_sparsification_penalty": float(
                    lbfgs_sparsification_penalty
                ),
                "lbfgs_param_delta_l2": float(lbfgs_param_delta_l2),
                "lbfgs_param_delta_max_abs": float(lbfgs_param_delta_max_abs),
                "rollout_performed": False,
                "elapsed_s": float(perf_counter() - started),
            }
            history.append(row)
            _log_line(
                log_handle,
                f"epoch={epoch}/{config.epochs} phase={phase} "
                f"train={row['train_loss']:.12e} val={row['val_loss']:.12e} "
                f"grad={row['grad_norm_total']:.6e} gain={row['gain']:.6e} "
                f"AGU={'ON' if agu_updated else 'OFF'} agu_s={agu_update_sec:.4f}",
            )
            if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
                write_checkpoint_atomic(
                    config.checkpoint_path,
                    _checkpoint_payload(
                        config=config,
                        endpoint=endpoint,
                        model=model,
                        adam=adam,
                        lbfgs=lbfgs,
                        epoch=epoch,
                        history=history,
                        lbfgs_zero_step_state=lbfgs_zero_step_state,
                        stop_kind=stop_kind,
                        stop_epoch=stop_epoch,
                        stop_reason=stop_reason,
                    ),
                )
    model.eval()
    with torch.no_grad():
        final = _forward_evaluation(model, endpoint, include_rollout=False)
    result = {
        "schema_version": 4,
        "stage": "AFM06a_stage2light",
        "complete": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_rank": endpoint.rank,
        "source_trial": dict(endpoint.record["params"]),
        "source_train_loss": float(endpoint.record["train_loss"]),
        "source_val_loss": float(endpoint.record["val_loss"]),
        "source_state_dict_sha256": str(endpoint.warmstart["kan_state_dict_sha256"]),
        "source_endpoint_sha256": source_endpoint_digest(endpoint),
        "entry_policy": config.entry_policy,
        "force_input_source": "observed_x1_at_every_time",
        "loss_policy": "force_scaled_dynamics_residual_true_x1_residual_smooth",
        "loss_target": 0.0,
        "diagnostic_rollout_policy": "none_in_st1pl_st2l_process",
        "rollout_performed": False,
        "epoch0_preparation_performed": False,
        "stop_kind": stop_kind,
        "stop_epoch": int(stop_epoch),
        "stop_reason": stop_reason,
        "lbfgs_zero_step_state": _normalized_lbfgs_zero_step_state(
            lbfgs_zero_step_state
        ),
        "stopped_early": bool(stop_kind),
        "completed_epoch": int(history[-1].get("epoch", 0)) if history else 0,
        "history": history,
        "final_train_loss": float(final.train_loss.detach().cpu()),
        "final_val_loss": float(final.val_loss.detach().cpu()),
        "final_loss_parts": final.train_parts,
        "final_val_loss_parts": final.val_parts,
        "predicted_fts": final.predicted_fts.detach().cpu().numpy(),
        "predicted_bar_fts": final.predicted_bar_fts.detach().cpu().numpy(),
        "times": endpoint.window.times.copy(),
        "train_idx": endpoint.window.train_idx.copy(),
        "val_idx": endpoint.window.val_idx.copy(),
        "window_source_indices": endpoint.window.source_idx.copy(),
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "model_state_dict_sha256": state_dict_digest(model.frozen_state_dict()),
        "trainable_parameters": trainable_parameter_manifest(model),
        "config": asdict(config),
    }
    write_checkpoint_atomic(config.result_path, result)
    return {
        "result_path": str(config.result_path),
        "checkpoint_path": str(config.checkpoint_path),
        "input_rank": endpoint.rank,
        "epochs": config.epochs,
        "completed_epoch": result["completed_epoch"],
        "stop_kind": result["stop_kind"],
        "stop_epoch": result["stop_epoch"],
        "stop_reason": result["stop_reason"],
        "final_train_loss": result["final_train_loss"],
        "final_val_loss": result["final_val_loss"],
    }


__all__ = [
    "ForwardEvaluation",
    "backward_smoke",
    "inherited_epoch0_record",
    "replay_stage1_endpoint",
    "run_stage2light",
    "source_endpoint_digest",
]
