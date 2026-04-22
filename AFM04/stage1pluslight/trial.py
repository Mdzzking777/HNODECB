"""Single-trial execution shell for AFM04 stage1pluslight."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from AFM04.stage1pluslight.checkpoint import trial_id_or_zero
from AFM04.stage1pluslight.losses import LossParts, loss_single_or_ms, parts_as_dict
from AFM04.stage1pluslight.mathutils import (
    bound_param,
    join_nonempty_unique,
    mean_finite,
    raw_from_value,
    rel_err_pct,
    stage1pluslight_nn_bank_seed,
)
from AFM04.stage1pluslight.mlp import build_mlp_bundle
from AFM04.test_case_settings.afm_dmt_kv_settings.afm_dmt_kv_model_settings import KS, CS, original_parameters


KnownPars = tuple[float, ...]


@dataclass(frozen=True)
class TrialModelBundle:
    model: Callable[[np.ndarray, Any], float] | None
    model_params: Any
    model_param_vec: np.ndarray
    unflatten: Callable[[np.ndarray], Any] | None
    force_mode: str
    backend_label: str


def build_zero_contact_model(seed: int, num_hidden_layers: int, num_hidden_nodes: int) -> TrialModelBundle:
    _ = (seed, num_hidden_layers, num_hidden_nodes)

    def model(u: np.ndarray, _params: Any) -> float:
        _ = u
        return 0.0

    return TrialModelBundle(
        model=model,
        model_params=np.zeros(0, dtype=float),
        model_param_vec=np.zeros(0, dtype=float),
        unflatten=lambda flat: np.asarray(flat, dtype=float).reshape(-1),
        force_mode="direct_contact_model",
        backend_label="zero_contact",
    )


def build_mlp_model(seed: int, num_hidden_layers: int, num_hidden_nodes: int) -> TrialModelBundle:
    bundle = build_mlp_bundle(seed, num_hidden_layers, num_hidden_nodes)
    return TrialModelBundle(
        model=bundle["model"],
        model_params=bundle["model_params"],
        model_param_vec=np.asarray(bundle["model_param_vec"], dtype=float),
        unflatten=bundle.get("unflatten"),
        force_mode=str(bundle["force_mode"]),
        backend_label=str(bundle["backend_label"]),
    )


def _known_pars_prefix(parameters: tuple[float, ...] = original_parameters) -> KnownPars:
    return tuple(float(x) for x in parameters[:12])


def _val_diag_or_empty(parts: LossParts) -> dict[str, float]:
    return parts_as_dict(parts)


def _theta_from_vectors(model_param_vec: np.ndarray, mech_raw: np.ndarray, bundle: TrialModelBundle) -> dict[str, Any]:
    params = bundle.model_params if bundle.unflatten is None else bundle.unflatten(np.asarray(model_param_vec, dtype=float))
    return {
        "model_params": params,
        "mech_raw": np.asarray(mech_raw, dtype=float).copy(),
    }


def _fd_gradient(
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    model_param_vec: np.ndarray,
    mech_raw: np.ndarray,
    *,
    rel_eps: float,
    abs_eps: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    base_model = np.asarray(model_param_vec, dtype=float).copy()
    base_mech = np.asarray(mech_raw, dtype=float).copy()
    base_loss = float(loss_fn(base_model, base_mech))
    grad_model = np.zeros_like(base_model)
    grad_mech = np.zeros_like(base_mech)

    for i in range(base_model.size):
        step = max(abs_eps, rel_eps * max(1.0, abs(base_model[i])))
        trial = base_model.copy()
        trial[i] += step
        loss_hi = float(loss_fn(trial, base_mech))
        grad_model[i] = (loss_hi - base_loss) / step

    for i in range(base_mech.size):
        step = max(abs_eps, rel_eps * max(1.0, abs(base_mech[i])))
        trial = base_mech.copy()
        trial[i] += step
        loss_hi = float(loss_fn(base_model, trial))
        grad_mech[i] = (loss_hi - base_loss) / step

    return base_loss, grad_model, grad_mech


def make_stage1pluslight_failed_record(
    global_trial_id: int,
    ks_fixed: float,
    cs_fixed: float,
    ks_node_idx: int,
    cs_node_idx: int,
    nn_seed_bank_idx: int,
    num_hidden_layers: int,
    num_hidden_nodes: int,
    *,
    window_role: str = "unknown",
    window_roles: list[str] | None = None,
    failure_reason: str = "",
    failure_epoch: int = 0,
    completed_epochs: int = 0,
    retry_count_total: int = 0,
    early_stopped: bool = False,
    early_stop_reason: str = "",
    lr_last: float = float("nan"),
) -> dict[str, Any]:
    params = {
        "trial_id": int(global_trial_id),
        "ks0": float(ks_fixed),
        "cs0": float(cs_fixed),
        "ks_node_idx": int(ks_node_idx),
        "cs_node_idx": int(cs_node_idx),
        "node_label": f"node_{ks_node_idx}*{cs_node_idx}",
        "nn_seed_bank_idx": int(nn_seed_bank_idx),
        "nn_force_mode": "direct_contact_model",
        "num_hidden_layers": int(num_hidden_layers),
        "num_hidden_nodes": int(num_hidden_nodes),
        "stage1plus_standalone": True,
        "stage1plus_search_mode": "loggrid_seedbank",
        "window_role": str(window_role),
    }
    if window_roles:
        params["window_roles"] = list(window_roles)
    return {
        "loss": float("inf"),
        "train_loss": float("inf"),
        "val_loss": float("inf"),
        "val_loss_start": float("inf"),
        "params": params,
        "val_parts": {
            "state": float("nan"),
            "x2dot": float("nan"),
            "x3_range": float("nan"),
            "fts_range": float("nan"),
            "cont": float("nan"),
            "x1_rec": float("nan"),
            "x3_rec": float("nan"),
        },
        "ks_hat": float(ks_fixed),
        "cs_hat": float(cs_fixed),
        "ks_err_pct": rel_err_pct(float(ks_fixed), KS, 1.0e-9),
        "cs_err_pct": rel_err_pct(float(cs_fixed), CS, 1.0e-9),
        "val_nn_err_start": float("nan"),
        "val_nn_err": float("nan"),
        "is_viable": False,
        "train_reason": str(failure_reason).strip(),
        "val_reason": str(failure_reason).strip(),
        "p_net_vec": np.zeros(0, dtype=float),
        "nn_gain": float("nan"),
        "nn_force_mode": "direct_contact_model",
        "completed_epochs": int(completed_epochs),
        "retry_count_total": int(retry_count_total),
        "early_stopped": bool(early_stopped),
        "early_stop_reason": str(early_stop_reason).strip(),
        "trial_failed": True,
        "failure_reason": str(failure_reason).strip(),
        "failure_epoch": int(failure_epoch),
        "lr_last": float(lr_last),
        "time_per_epoch": float("inf"),
    }


def stage1pluslight_joint_trial(
    global_trial_id: int,
    ks_fixed: float,
    cs_fixed: float,
    ks_node_idx: int,
    cs_node_idx: int,
    nn_seed_bank_idx: int,
    num_hidden_layers: int,
    num_hidden_nodes: int,
    ode_train: np.ndarray,
    x2dot_train: np.ndarray,
    contact_train: np.ndarray,
    times_train: np.ndarray,
    ode_val: np.ndarray,
    x2dot_val: np.ndarray,
    contact_val: np.ndarray,
    times_val: np.ndarray,
    state12_scale: np.ndarray,
    x2dot_scale: float,
    x3_t0_val: float,
    ms_group_size: int,
    ms_continuity_term: float,
    zero_contact_override: bool,
    trial_opt_enabled: bool,
    trial_epochs: int,
    trial_lr: float,
    *,
    window_role: str = "single",
    run_seed: int = 20260406,
    model_factory: Callable[[int, int, int], TrialModelBundle] | None = None,
    known_pars: KnownPars | None = None,
    use_multiple_shooting: bool = False,
    fd_rel_eps: float = 1.0e-4,
    fd_abs_eps: float = 1.0e-8,
    step_guard_enabled: bool = True,
    step_retry_max: int = 6,
    step_retry_lr_factor: float = 0.1,
    step_max_loss_frac: float = 0.1,
    early_stop_epoch: int = 5,
    early_stop_min_drop_frac: float = 0.01,
    ode_solver: str = "Radau",
    ode_fallback_solver: str = "BDF",
    ode_rtol: float = 1.0e-8,
    ode_atol: float = 1.0e-8,
    ode_max_step: float = 0.0,
) -> dict[str, Any]:
    if known_pars is None:
        known_pars = _known_pars_prefix()
    if model_factory is None:
        model_factory = build_mlp_model

    seed = stage1pluslight_nn_bank_seed(run_seed, num_hidden_layers, num_hidden_nodes, nn_seed_bank_idx)
    bundle = model_factory(seed, num_hidden_layers, num_hidden_nodes)
    model_param_vec = np.asarray(bundle.model_param_vec, dtype=float).copy()
    mech_raw = np.array(
        [
            raw_from_value(ks_fixed, 0.005, 5.0),
            raw_from_value(cs_fixed, 5.0e-8, 5.0e-6),
        ],
        dtype=float,
    )
    theta0 = _theta_from_vectors(model_param_vec, mech_raw, bundle)

    val_loss_start, _ = loss_single_or_ms(
        theta0,
        ode_val,
        x2dot_val,
        contact_val,
        times_val,
        state12_scale,
        x2dot_scale,
        100.0e-9,
        use_multiple_shooting,
        ms_group_size,
        ms_continuity_term,
        bundle.model,
        known_pars,
        x3_t0_val,
        zero_contact_override=zero_contact_override,
        ode_solver=ode_solver,
        ode_fallback_solver=ode_fallback_solver,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        ode_max_step=ode_max_step,
    )

    def train_loss_from_vectors(model_vec: np.ndarray, mech_vec: np.ndarray) -> float:
        theta = _theta_from_vectors(model_vec, mech_vec, bundle)
        return float(
            loss_single_or_ms(
                theta,
                ode_train,
                x2dot_train,
                contact_train,
                times_train,
                state12_scale,
                x2dot_scale,
                100.0e-9,
                use_multiple_shooting,
                ms_group_size,
                ms_continuity_term,
                bundle.model,
                known_pars,
                x3_t0_val,
                zero_contact_override=zero_contact_override,
                ode_solver=ode_solver,
                ode_fallback_solver=ode_fallback_solver,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                ode_max_step=ode_max_step,
            )[0]
        )

    def val_loss_from_vectors(model_vec: np.ndarray, mech_vec: np.ndarray) -> tuple[float, dict[str, float]]:
        theta = _theta_from_vectors(model_vec, mech_vec, bundle)
        loss, parts = loss_single_or_ms(
            theta,
            ode_val,
            x2dot_val,
            contact_val,
            times_val,
            state12_scale,
            x2dot_scale,
            100.0e-9,
            use_multiple_shooting,
            ms_group_size,
            ms_continuity_term,
            bundle.model,
            known_pars,
            x3_t0_val,
            zero_contact_override=zero_contact_override,
            ode_solver=ode_solver,
            ode_fallback_solver=ode_fallback_solver,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_max_step=ode_max_step,
        )
        return float(loss), _val_diag_or_empty(parts)

    trial_failed = False
    trial_fail_reason = ""
    trial_fail_epoch = 0
    early_stopped = False
    early_stop_reason = ""
    retry_count_total = 0
    completed_epochs = 0
    lr = float(trial_lr)
    model_vec_cur = model_param_vec.copy()
    mech_raw_cur = mech_raw.copy()
    best_val_loss = float(val_loss_start)
    epoch_durations: list[float] = []
    train_loss_last = float("inf")

    if trial_opt_enabled:
        for epoch in range(1, max(0, int(trial_epochs)) + 1):
            epoch_t0 = time.time()
            try:
                train_loss_before, grad_model, grad_mech = _fd_gradient(
                    train_loss_from_vectors,
                    model_vec_cur,
                    mech_raw_cur,
                    rel_eps=fd_rel_eps,
                    abs_eps=fd_abs_eps,
                )
            except Exception as err:
                trial_failed = True
                trial_fail_reason = f"adjoint_exception:{err}"
                trial_fail_epoch = epoch
                break

            if not np.isfinite(train_loss_before) or not np.all(np.isfinite(grad_model)) or not np.all(np.isfinite(grad_mech)):
                trial_failed = True
                trial_fail_reason = "nonfinite_train_or_grad"
                trial_fail_epoch = epoch
                break

            model_base = model_vec_cur.copy()
            mech_base = mech_raw_cur.copy()
            train_loss_after = float("inf")
            val_loss_epoch = float("inf")
            val_diag_epoch = None
            step_fail_reason = ""
            step_accepted = False
            prev_epoch_loss_ref = train_loss_last if (completed_epochs > 0 and np.isfinite(train_loss_last)) else float("nan")

            attempts_total = 1 if not step_guard_enabled else max(1, step_retry_max + 1)
            lr_local = lr
            for attempt_idx in range(1, attempts_total + 1):
                model_trial = model_base - lr_local * grad_model
                mech_trial = mech_base - lr_local * grad_mech
                try:
                    train_loss_trial = train_loss_from_vectors(model_trial, mech_trial)
                except Exception as err:
                    trial_failed = True
                    trial_fail_reason = f"step_trial_exception:{err}"
                    trial_fail_epoch = epoch
                    break

                step_max_loss_increase = abs(train_loss_before) * step_max_loss_frac if np.isfinite(train_loss_before) else float("inf")
                prev_epoch_max_loss = prev_epoch_loss_ref * (1.0 + step_max_loss_frac) if np.isfinite(prev_epoch_loss_ref) else float("inf")
                if not np.isfinite(train_loss_trial):
                    step_fail_reason = "step_trial_loss_nonfinite"
                elif np.isfinite(prev_epoch_max_loss) and train_loss_trial > prev_epoch_max_loss:
                    step_fail_reason = "step_prev_epoch_loss_jump"
                elif np.isfinite(step_max_loss_increase) and train_loss_trial > train_loss_before + step_max_loss_increase:
                    step_fail_reason = "step_trial_loss_jump"
                else:
                    step_fail_reason = ""

                if step_fail_reason == "":
                    model_vec_cur = model_trial
                    mech_raw_cur = mech_trial
                    train_loss_after = float(train_loss_trial)
                    try:
                        val_loss_epoch, val_diag_epoch = val_loss_from_vectors(model_vec_cur, mech_raw_cur)
                    except Exception as err:
                        trial_failed = True
                        trial_fail_reason = f"step_val_exception:{err}"
                        trial_fail_epoch = epoch
                        break
                    step_accepted = True
                    retry_count_total += attempt_idx - 1
                    lr = lr_local
                    break

                if trial_failed:
                    break

                if attempt_idx >= attempts_total:
                    retry_count_total += max(0, attempts_total - 1)
                    if step_fail_reason == "step_trial_loss_nonfinite":
                        trial_failed = True
                        trial_fail_reason = step_fail_reason
                        trial_fail_epoch = epoch
                        break
                    model_vec_cur = model_base
                    mech_raw_cur = mech_base
                    train_loss_after = float(train_loss_before)
                    try:
                        val_loss_epoch, val_diag_epoch = val_loss_from_vectors(model_vec_cur, mech_raw_cur)
                    except Exception as err:
                        trial_failed = True
                        trial_fail_reason = f"fallback_val_exception:{err}"
                        trial_fail_epoch = epoch
                    break

                lr_local = max(lr_local * step_retry_lr_factor, 1.0e-12)

            if trial_failed:
                break

            epoch_dt = float(time.time() - epoch_t0)
            epoch_durations.append(epoch_dt)
            train_loss_last = float(train_loss_after)
            if np.isfinite(val_loss_epoch):
                best_val_loss = min(best_val_loss, float(val_loss_epoch))
            completed_epochs += 1

            if completed_epochs >= max(1, int(early_stop_epoch)) and np.isfinite(val_loss_start) and np.isfinite(best_val_loss):
                drop_frac = (val_loss_start - best_val_loss) / max(abs(val_loss_start), 1.0e-30)
                if drop_frac < float(early_stop_min_drop_frac):
                    early_stopped = True
                    early_stop_reason = f"epoch_{completed_epochs}_drop_lt_{early_stop_min_drop_frac:.4f}"
                    break

    theta_final = _theta_from_vectors(model_vec_cur, mech_raw_cur, bundle)
    try:
        train_loss_end, _ = loss_single_or_ms(
            theta_final,
            ode_train,
            x2dot_train,
            contact_train,
            times_train,
            state12_scale,
            x2dot_scale,
            100.0e-9,
            use_multiple_shooting,
            ms_group_size,
            ms_continuity_term,
            bundle.model,
            known_pars,
            x3_t0_val,
            zero_contact_override=zero_contact_override,
            ode_solver=ode_solver,
            ode_fallback_solver=ode_fallback_solver,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_max_step=ode_max_step,
        )
        val_loss_end, val_diag = loss_single_or_ms(
            theta_final,
            ode_val,
            x2dot_val,
            contact_val,
            times_val,
            state12_scale,
            x2dot_scale,
            100.0e-9,
            use_multiple_shooting,
            ms_group_size,
            ms_continuity_term,
            bundle.model,
            known_pars,
            x3_t0_val,
            zero_contact_override=zero_contact_override,
            ode_solver=ode_solver,
            ode_fallback_solver=ode_fallback_solver,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_max_step=ode_max_step,
        )
        val_diag = _val_diag_or_empty(val_diag)
    except Exception as err:
        trial_failed = True
        if trial_fail_reason == "":
            trial_fail_reason = f"final_eval_exception:{err}"
        if trial_fail_epoch == 0:
            trial_fail_epoch = completed_epochs
        train_loss_end = float("inf")
        val_loss_end = float("inf")
        val_diag = {
            "state": float("nan"),
            "x2dot": float("nan"),
            "x3_range": float("nan"),
            "fts_range": float("nan"),
            "cont": float("nan"),
            "x1_rec": float("nan"),
            "x3_rec": float("nan"),
        }

    ks_hat = float(bound_param(float(mech_raw_cur[0]), 0.005, 5.0))
    cs_hat = float(bound_param(float(mech_raw_cur[1]), 5.0e-8, 5.0e-6))
    ks_err_pct = rel_err_pct(ks_hat, KS, 1.0e-9)
    cs_err_pct = rel_err_pct(cs_hat, CS, 1.0e-9)
    is_viable = (not trial_failed) and (not early_stopped) and np.isfinite(train_loss_end) and np.isfinite(val_loss_end)
    params = {
        "trial_id": int(global_trial_id),
        "ks0": float(ks_fixed),
        "cs0": float(cs_fixed),
        "ks_node_idx": int(ks_node_idx),
        "cs_node_idx": int(cs_node_idx),
        "node_label": f"node_{ks_node_idx}*{cs_node_idx}",
        "nn_seed_bank_idx": int(nn_seed_bank_idx),
        "nn_init_seed": int(seed),
        "nn_force_mode": bundle.force_mode,
        "nn_backend": bundle.backend_label,
        "num_hidden_layers": int(num_hidden_layers),
        "num_hidden_nodes": int(num_hidden_nodes),
        "stage1plus_standalone": True,
        "stage1plus_search_mode": "loggrid_seedbank",
        "window_role": str(window_role),
    }

    return {
        "loss": float(val_loss_end),
        "train_loss": float(train_loss_end),
        "val_loss": float(val_loss_end),
        "val_loss_start": float(val_loss_start),
        "params": params,
        "val_parts": val_diag,
        "ks_hat": ks_hat,
        "cs_hat": cs_hat,
        "ks_err_pct": ks_err_pct,
        "cs_err_pct": cs_err_pct,
        "val_nn_err_start": float("nan"),
        "val_nn_err": float("nan"),
        "is_viable": bool(is_viable),
        "train_reason": trial_fail_reason if trial_failed else "",
        "val_reason": trial_fail_reason if trial_failed else "",
        "p_net_vec": np.asarray(model_vec_cur, dtype=float).copy(),
        "nn_gain": float("nan"),
        "nn_force_mode": bundle.force_mode,
        "completed_epochs": int(completed_epochs if trial_opt_enabled else max(trial_epochs, 0)),
        "retry_count_total": int(retry_count_total),
        "early_stopped": bool(early_stopped),
        "early_stop_reason": str(early_stop_reason),
        "trial_failed": bool(trial_failed),
        "failure_reason": str(trial_fail_reason),
        "failure_epoch": int(trial_fail_epoch),
        "lr_last": float(lr),
        "time_per_epoch": float(mean_finite(epoch_durations) if epoch_durations else float("inf")),
    }


def stage1pluslight_joint_trial_multiwindow(
    global_trial_id: int,
    ks_fixed: float,
    cs_fixed: float,
    ks_node_idx: int,
    cs_node_idx: int,
    nn_seed_bank_idx: int,
    num_hidden_layers: int,
    num_hidden_nodes: int,
    window_bundles: list[dict[str, Any]],
    state12_scale: np.ndarray,
    x2dot_scale: float,
    ms_group_size: int,
    ms_continuity_term: float,
    zero_contact_override: bool,
    trial_opt_enabled: bool,
    trial_epochs: int,
    trial_lr: float,
    *,
    run_seed: int = 20260406,
    model_factory: Callable[[int, int, int], TrialModelBundle] | None = None,
    known_pars: KnownPars | None = None,
    use_multiple_shooting: bool = False,
    fd_rel_eps: float = 1.0e-4,
    fd_abs_eps: float = 1.0e-8,
    step_guard_enabled: bool = True,
    step_retry_max: int = 6,
    step_retry_lr_factor: float = 0.1,
    step_max_loss_frac: float = 0.1,
    early_stop_epoch: int = 5,
    early_stop_min_drop_frac: float = 0.01,
    ode_solver: str = "Radau",
    ode_fallback_solver: str = "BDF",
    ode_rtol: float = 1.0e-8,
    ode_atol: float = 1.0e-8,
    ode_max_step: float = 0.0,
) -> dict[str, Any]:
    window_recs = [
        stage1pluslight_joint_trial(
            global_trial_id,
            ks_fixed,
            cs_fixed,
            ks_node_idx,
            cs_node_idx,
            nn_seed_bank_idx,
            num_hidden_layers,
            num_hidden_nodes,
            bundle["ode_train"],
            bundle["x2dot_train"],
            bundle["contact_train"],
            bundle["times_train"],
            bundle["ode_val"],
            bundle["x2dot_val"],
            bundle["contact_val"],
            bundle["times_val"],
            state12_scale,
            x2dot_scale,
            bundle["x3_t0_val"],
            ms_group_size,
            ms_continuity_term,
            zero_contact_override,
            trial_opt_enabled,
            trial_epochs,
            trial_lr,
            window_role=str(bundle["role"]),
            run_seed=run_seed,
            model_factory=model_factory,
            known_pars=known_pars,
            use_multiple_shooting=use_multiple_shooting,
            fd_rel_eps=fd_rel_eps,
            fd_abs_eps=fd_abs_eps,
            step_guard_enabled=step_guard_enabled,
            step_retry_max=step_retry_max,
            step_retry_lr_factor=step_retry_lr_factor,
            step_max_loss_frac=step_max_loss_frac,
            early_stop_epoch=early_stop_epoch,
            early_stop_min_drop_frac=early_stop_min_drop_frac,
            ode_solver=ode_solver,
            ode_fallback_solver=ode_fallback_solver,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_max_step=ode_max_step,
        )
        for bundle in window_bundles
    ]
    if not window_recs:
        return make_stage1pluslight_failed_record(
            global_trial_id,
            ks_fixed,
            cs_fixed,
            ks_node_idx,
            cs_node_idx,
            nn_seed_bank_idx,
            num_hidden_layers,
            num_hidden_nodes,
            window_role="stage2_w123",
            failure_reason="empty_window_bundle",
        )

    ref = window_recs[0]
    params = dict(ref["params"])
    params["stage1plus_window_mode"] = "stage2_w123"
    params["window_roles"] = [str(bundle["role"]) for bundle in window_bundles]
    params["window_train_losses"] = [float(rec["train_loss"]) for rec in window_recs]
    params["window_val_losses"] = [float(rec["val_loss"]) for rec in window_recs]
    params["window_val_loss_starts"] = [float(rec["val_loss_start"]) for rec in window_recs]
    return {
        "loss": mean_finite(rec["loss"] for rec in window_recs),
        "train_loss": mean_finite(rec["train_loss"] for rec in window_recs),
        "val_loss": mean_finite(rec["val_loss"] for rec in window_recs),
        "val_loss_start": mean_finite(rec["val_loss_start"] for rec in window_recs),
        "params": params,
        "val_parts": {
            key: mean_finite(rec["val_parts"][key] for rec in window_recs)
            for key in ref["val_parts"].keys()
        },
        "ks_hat": ref["ks_hat"],
        "cs_hat": ref["cs_hat"],
        "ks_err_pct": ref["ks_err_pct"],
        "cs_err_pct": ref["cs_err_pct"],
        "val_nn_err_start": float("nan"),
        "val_nn_err": float("nan"),
        "is_viable": all(bool(rec["is_viable"]) for rec in window_recs),
        "train_reason": join_nonempty_unique(str(rec["train_reason"]) for rec in window_recs),
        "val_reason": join_nonempty_unique(str(rec["val_reason"]) for rec in window_recs),
        "p_net_vec": np.asarray(ref["p_net_vec"], dtype=float).copy(),
        "nn_gain": float("nan"),
        "nn_force_mode": ref["nn_force_mode"],
        "completed_epochs": max(int(rec["completed_epochs"]) for rec in window_recs),
        "retry_count_total": sum(int(rec["retry_count_total"]) for rec in window_recs),
        "early_stopped": any(bool(rec["early_stopped"]) for rec in window_recs),
        "early_stop_reason": join_nonempty_unique(str(rec["early_stop_reason"]) for rec in window_recs),
        "trial_failed": any(bool(rec["trial_failed"]) for rec in window_recs),
        "failure_reason": join_nonempty_unique(str(rec["failure_reason"]) for rec in window_recs),
        "failure_epoch": max(int(rec["failure_epoch"]) for rec in window_recs),
        "lr_last": ref["lr_last"],
        "time_per_epoch": mean_finite(rec["time_per_epoch"] for rec in window_recs),
        "window_train_losses": [float(rec["train_loss"]) for rec in window_recs],
        "window_val_losses": [float(rec["val_loss"]) for rec in window_recs],
    }


__all__ = [
    "TrialModelBundle",
    "build_mlp_model",
    "build_zero_contact_model",
    "make_stage1pluslight_failed_record",
    "stage1pluslight_joint_trial",
    "stage1pluslight_joint_trial_multiwindow",
    "trial_id_or_zero",
]
