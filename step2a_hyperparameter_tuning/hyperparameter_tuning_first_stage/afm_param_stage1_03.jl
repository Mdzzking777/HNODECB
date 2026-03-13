#=
Stage 1 (global TPE) hyperparameter tuning for AFM DMT-KV.
Scenario 03: x3 unobserved, Hertz force replaced by a neural network.
Single shooting or multiple shooting (toggle), no x3 observations (only range prior).
TPE explores NN architecture, mechanistic initial values, MS hyperparams, learning rate, and train/val split.
=#

cd(@__DIR__)

using Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Statistics, Dates
using ComponentArrays, SciMLSensitivity, SciMLBase, StableRNGs
using Zygote
using Optimization, OptimizationOptimisers, Optimisers
using DiffEqFlux, Flux

using PyCall
optuna = pyimport("optuna")

# Suppress noisy Lux overwrite warnings by default (set ENV["HNODECB_SUPPRESS_LUX_WARN"]="0" to disable)
if get(ENV, "HNODECB_SUPPRESS_LUX_WARN", "1") == "1"
  redirect_stderr(devnull) do
    @eval using Lux
  end
else
  using Lux
end

# weight init helper (Lux passes rng to init functions)
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

# global RNG for reproducible trial seeds
rng_global = Random.default_rng()
Random.seed!(rng_global, 0)

include("../../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_functions.jl")
include("../../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")

result_folder = "results_afm"
if !isdir(result_folder)
  mkdir(result_folder)
end
run_tag = get(ENV, "HNODECB_STAGE1_RUN_TAG", "")
stage1_variant = get(ENV, "HNODECB_STAGE1_VARIANT", "stage1")
use_stage1pluslight = stage1_variant == "stage1pluslight" || get(ENV, "HNODECB_STAGE1PLUSLIGHT_MODE", "0") == "1"
use_stage1plus = stage1_variant == "stage1plus" || stage1_variant == "stage1pluslight" ||
  get(ENV, "HNODECB_STAGE1PLUS_MODE", "0") == "1" || use_stage1pluslight
default_result_stem = use_stage1pluslight ? "afm_param_stage1pluslight_03" :
  (use_stage1plus ? "afm_param_stage1plus_03" : "afm_param_stage1_03")
result_stem = get(ENV, "HNODECB_STAGE1_RESULT_STEM", default_result_stem)
result_name_string = run_tag == "" ? result_stem * ".jld" : result_stem * "_" * run_tag * ".jld"

error_level = "e0.0"

ks_true = ks
cs_true = cs

# Load data
ode_data_full = deserialize("../../datasets/e0.0/data/ode_data_afm_dmt_kv.jld")
solution_dataframe_full = deserialize("../../datasets/e0.0/data/pert_df_afm_dmt_kv.jld")

# Keep only data after first contact
contact_idx = findfirst(solution_dataframe_full.contact .== 1)
if contact_idx === nothing
  error("No contact point found in the AFM dataset.")
end
solution_dataframe_full = solution_dataframe_full[contact_idx:end, :]
ode_data_full = ode_data_full[:, contact_idx:end]

# Times and signals
all_times = solution_dataframe_full.t
x2dot_all = solution_dataframe_full.x2dot
contact_all = solution_dataframe_full.contact .== 1
true_s_all = Float64.(solution_dataframe_full.s)
true_contact_weight_all = contact_weight.(true_s_all, adhesion_transition)
true_contact_weight_at_time = make_contact_weight_lookup(all_times, true_s_all)
monitor_idx_full = collect(1:length(all_times))

# Physical prior for x3 (no observations)
x3_range_center = 0.0
x3_range_amp = 20e-9
x3_range_eps = 1e-9
x3_range_weight = 1.0

# Contact weighting
contact_loss_weight = 4.27
noncontact_loss_weight = 1.0

# Solver settings
integrator = Rosenbrock23(autodiff=false)
abstol = 1e-8
reltol = 1e-8
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))
ode_maxiters = parse(Int, get(ENV, "HNODECB_ODE_MAXITERS", "1000000"))

# Parameter bounds (mechanistic unknowns)
ks_bounds = (0.005, 5.0)   # true ks = 0.1
cs_bounds = (5e-8, 5e-6)   # true cs = 2.4e-7

# Known parameters (fixed)
known_pars = original_parameters[1:8]

# Normalization scales (from full data)
scale_eps = 1e-9
state12_scale_full = vec(maximum(ode_data_full[1:2, :], dims=2) - minimum(ode_data_full[1:2, :], dims=2))
state12_scale_full = max.(state12_scale_full, scale_eps)
x2dot_scale_full = max(maximum(x2dot_all) - minimum(x2dot_all), scale_eps)
x3_scale = max(x3_range_amp, scale_eps)

# Helper functions
sigmoid(x) = 1 / (1 + exp(-x))
function logit(x)
  return log(x / (1 - x))
end
function bound_param(raw, lo, hi)
  return lo + (hi - lo) * sigmoid(raw)
end
function raw_from_value(val, lo, hi)
  z = (val - lo) / (hi - lo)
  z = clamp(z, 1e-6, 1 - 1e-6)
  return logit(z)
end
function inv_softplus(y, eps)
  if y <= 0.0
    return -50.0 * eps
  end
  z = y / eps
  if z > 50.0
    return y
  end
  return eps * log(expm1(z))
end

function percent_error_pct(est, truth)
  return 100.0 * abs(log10(est / truth))
end

function relative_rmse_pct(err_sum, truth_sum, count, eps)
  denom = sqrt(truth_sum / count) + eps
  return 100.0 * sqrt(err_sum / count) / denom
end
function rel_err_pct(est, truth, eps)
  return 100.0 * abs(est - truth) / (abs(truth) + eps)
end

# Timestamped println for easier runtime tracking
function tprintln(args...)
  # Avoid Zygote tracing time calls
  ts = Zygote.ignore() do
    Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
  end
  println("[", ts, "] ", args...)
end

function fmt_e(x; sigdigits=4)
  if x isa Number
    return isfinite(x) ? string(round(x, sigdigits=sigdigits)) : string(x)
  end
  return string(x)
end

function fmt_f(x; digits=2)
  if x isa Number
    return isfinite(x) ? string(round(x, digits=digits)) : string(x)
  end
  return string(x)
end

function env_float(key, default)
  v = get(ENV, key, "")
  if v == ""
    return default
  end
  try
    return parse(Float64, v)
  catch
    return default
  end
end

stage1pluslight_gnn_enabled = use_stage1pluslight && get(ENV, "HNODECB_STAGE1PLUS_GNN_ENABLE", "1") == "1"
stage1pluslight_gnn_target_spec = strip(get(ENV, "HNODECB_STAGE1PLUS_GNN_TARGET", ""))
stage1pluslight_gnn_target_mode = lowercase(strip(get(ENV, "HNODECB_STAGE1PLUS_GNN_TARGET_MODE",
  stage1pluslight_gnn_target_spec == "" ? "auto" : "manual")))
if !(stage1pluslight_gnn_target_mode in ("auto", "manual"))
  stage1pluslight_gnn_target_mode = stage1pluslight_gnn_target_spec == "" ? "auto" : "manual"
end
stage1pluslight_gnn_target_manual = stage1pluslight_gnn_target_spec == "" ? 5e-10 :
  try
    parse(Float64, stage1pluslight_gnn_target_spec)
  catch
    5e-10
  end
stage1pluslight_gnn_quantile = clamp(env_float("HNODECB_STAGE1PLUS_GNN_QUANTILE", 0.95), 0.0, 1.0)
stage1pluslight_gnn_eps = env_float("HNODECB_STAGE1PLUS_GNN_EPS", 1e-30)
stage1pluslight_gnn_contact_floor = clamp(env_float("HNODECB_STAGE1PLUS_GNN_CONTACT_FLOOR", 0.5), 0.0, 1.0)
stage1pluslight_gnn_target_quantile = clamp(
  env_float("HNODECB_STAGE1PLUS_GNN_TARGET_QUANTILE", stage1pluslight_gnn_quantile), 0.0, 1.0)
stage1pluslight_gnn_target_min = env_float("HNODECB_STAGE1PLUS_GNN_TARGET_MIN", 1e-12)
stage1pluslight_gnn_target_max = env_float("HNODECB_STAGE1PLUS_GNN_TARGET_MAX", Inf)

function grad_norm_safe(g)
  try
    return norm(vec(g))
  catch
    return NaN
  end
end

function scale_stage1plus_grad(grad_raw, p_net_scale::Float64, mech_scale::Float64)
  if grad_raw === nothing
    return nothing
  end
  p_scale = (isfinite(p_net_scale) && p_net_scale >= 0.0) ? p_net_scale : 1.0
  m_scale = (isfinite(mech_scale) && mech_scale >= 0.0) ? mech_scale : 1.0
  try
    return ComponentVector(
      p_net = p_scale .* grad_raw.p_net,
      mech_raw = m_scale .* grad_raw.mech_raw
    )
  catch
    return grad_raw
  end
end

function grad_group_norms(grad_raw)
  if grad_raw === nothing
    return NaN, NaN
  end
  try
    return sqrt(sum(abs2, grad_raw.p_net)), sqrt(sum(abs2, grad_raw.mech_raw))
  catch
    return NaN, NaN
  end
end

# "God-view" Hertz force (truth) for monitoring only (never used in training).
function fhertz_true_from_states(u_mat, idxs)
  out = Vector{Float64}(undef, length(idxs))
  @inbounds for (k, j) in enumerate(idxs)
    s = dist + u_mat[1, j] - u_mat[3, j]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    out[k] = (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
  end
  return out
end

# NN-predicted Hertz force for monitoring only (never used in training).
function fhertz_pred_from_states(u_mat, idxs, p_net_struct, appr, st; nn_gain::Float64=1.0)
  out = Vector{Float64}(undef, length(idxs))
  @inbounds for (k, j) in enumerate(idxs)
    s = dist + u_mat[1, j] - u_mat[3, j]
    w_pred = contact_weight(s, adhesion_transition)
    nn_in = nn_input_from_values(u_mat[1, j], u_mat[2, j], u_mat[3, j])
    uhat = appr(nn_in, p_net_struct, st)[1]
    out[k] = nn_gain * uhat[1] * w_pred
  end
  return out
end

function fhertz_pred_and_raw_from_states(u_mat, idxs, p_net_struct, appr, st; nn_gain::Float64=1.0)
  out = Vector{Float64}(undef, length(idxs))
  raw = Vector{Float64}(undef, length(idxs))
  @inbounds for (k, j) in enumerate(idxs)
    s = dist + u_mat[1, j] - u_mat[3, j]
    w_pred = contact_weight(s, adhesion_transition)
    nn_in = nn_input_from_values(u_mat[1, j], u_mat[2, j], u_mat[3, j])
    uhat = appr(nn_in, p_net_struct, st)[1]
    raw[k] = uhat[1]
    out[k] = nn_gain * uhat[1] * w_pred
  end
  return out, raw
end

function effective_contact_positions(idxs, fh_true)
  contact_pos = [k for (k, j) in enumerate(idxs) if contact_all[j]]
  if isempty(contact_pos)
    return Int[]
  end
  max_true = maximum(fh_true[contact_pos])
  if !isfinite(max_true) || max_true <= 0.0
    return contact_pos
  end
  floor_val = nn_monitor_effective_contact_frac * max_true
  eff_pos = [k for k in contact_pos if fh_true[k] > floor_val]
  return isempty(eff_pos) ? contact_pos : eff_pos
end

function effective_contact_positions_by_wtrue(idxs; floor=0.5)
  pos = [k for (k, j) in enumerate(idxs) if true_contact_weight_all[j] > floor]
  if isempty(pos)
    pos = [k for (k, j) in enumerate(idxs) if true_contact_weight_all[j] > 0.0]
  end
  return isempty(pos) ? collect(1:length(idxs)) : pos
end

function estimate_stage1plus_gnn_target(u_mat, idxs, eff_pos)
  k, wd, m, c, Fd, _, _, Fad = known_pars
  n = length(eff_pos)
  inert_vals = Vector{Float64}(undef, n)
  spring_vals = Vector{Float64}(undef, n)
  damp_vals = Vector{Float64}(undef, n)
  drive_vals = Vector{Float64}(undef, n)
  fadh_vals = Vector{Float64}(undef, n)
  fh_est_vals = Vector{Float64}(undef, n)
  @inbounds for (m_idx, k_idx) in enumerate(eff_pos)
    j = idxs[k_idx]
    t = all_times[j]
    x1 = u_mat[1, j]
    x2 = u_mat[2, j]
    w_true = true_contact_weight_all[j]
    inert = m * x2dot_all[j]
    spring = k * x1
    damp = c * x2
    drive = Fd * cos(wd * t)
    fadh_eff = Fad * w_true
    inert_vals[m_idx] = abs(inert)
    spring_vals[m_idx] = abs(spring)
    damp_vals[m_idx] = abs(damp)
    drive_vals[m_idx] = abs(drive)
    fadh_vals[m_idx] = abs(fadh_eff)
    fh_est_vals[m_idx] = abs(drive - spring - damp + fadh_eff - inert)
  end
  q = stage1pluslight_gnn_target_quantile
  p95_inert = quantile(inert_vals, q)
  p95_spring = quantile(spring_vals, q)
  p95_damp = quantile(damp_vals, q)
  p95_drive = quantile(drive_vals, q)
  p95_fadh = quantile(fadh_vals, q)
  target_est = quantile(fh_est_vals, q)
  if !isfinite(target_est) || target_est <= 0.0
    target_est = maximum((p95_inert, p95_spring, p95_damp, p95_drive, p95_fadh, stage1pluslight_gnn_target_min))
  end
  target_est = clamp(max(target_est, stage1pluslight_gnn_target_min),
    stage1pluslight_gnn_target_min, stage1pluslight_gnn_target_max)
  return (
    target_est=target_est,
    p95_mx2dot=p95_inert,
    p95_kx1=p95_spring,
    p95_cx2=p95_damp,
    p95_fd=p95_drive,
    p95_fadh=p95_fadh
  )
end

function compute_stage1plus_gnn_scale(u_mat, idxs, p_net_struct, appr, st)
  if !stage1pluslight_gnn_enabled || appr === nothing
    return (g_nn=1.0, amp_ref=NaN, eff_count=0,
      target=NaN, target_mode=stage1pluslight_gnn_target_mode, target_est=NaN,
      p95_mx2dot=NaN, p95_kx1=NaN, p95_cx2=NaN, p95_fd=NaN, p95_fadh=NaN)
  end
  eff_pos = effective_contact_positions_by_wtrue(idxs; floor=stage1pluslight_gnn_contact_floor)
  if isempty(eff_pos)
    return (g_nn=1.0, amp_ref=NaN, eff_count=0,
      target=NaN, target_mode=stage1pluslight_gnn_target_mode, target_est=NaN,
      p95_mx2dot=NaN, p95_kx1=NaN, p95_cx2=NaN, p95_fd=NaN, p95_fadh=NaN)
  end
  amp_vals = Vector{Float64}(undef, length(eff_pos))
  @inbounds for (m, k) in enumerate(eff_pos)
    j = idxs[k]
    s = dist + u_mat[1, j] - u_mat[3, j]
    w_pred = contact_weight(s, adhesion_transition)
    nn_in = nn_input_from_values(u_mat[1, j], u_mat[2, j], u_mat[3, j])
    uhat = appr(nn_in, p_net_struct, st)[1]
    amp_vals[m] = abs(uhat[1] * w_pred)
  end
  amp_ref = quantile(amp_vals, stage1pluslight_gnn_quantile)
  if !isfinite(amp_ref) || amp_ref < 0.0
    target_info = estimate_stage1plus_gnn_target(u_mat, idxs, eff_pos)
    target_val = stage1pluslight_gnn_target_mode == "manual" ?
      stage1pluslight_gnn_target_manual : target_info.target_est
    return (g_nn=1.0, amp_ref=amp_ref, eff_count=length(eff_pos),
      target=target_val, target_mode=stage1pluslight_gnn_target_mode,
      target_est=target_info.target_est,
      p95_mx2dot=target_info.p95_mx2dot, p95_kx1=target_info.p95_kx1,
      p95_cx2=target_info.p95_cx2, p95_fd=target_info.p95_fd, p95_fadh=target_info.p95_fadh)
  end
  target_info = estimate_stage1plus_gnn_target(u_mat, idxs, eff_pos)
  target_val = stage1pluslight_gnn_target_mode == "manual" ?
    stage1pluslight_gnn_target_manual : target_info.target_est
  g_nn = target_val / (amp_ref + stage1pluslight_gnn_eps)
  if !isfinite(g_nn) || g_nn <= 0.0
    g_nn = 1.0
  end
  return (g_nn=g_nn, amp_ref=amp_ref, eff_count=length(eff_pos),
    target=target_val, target_mode=stage1pluslight_gnn_target_mode,
    target_est=target_info.target_est,
    p95_mx2dot=target_info.p95_mx2dot, p95_kx1=target_info.p95_kx1,
    p95_cx2=target_info.p95_cx2, p95_fd=target_info.p95_fd, p95_fadh=target_info.p95_fadh)
end

function fhertz_monitor_metrics(u_mat, idxs, p_net_struct, appr, st; fh_true_ref=nothing, eff_pos_ref=nothing, nn_gain::Float64=1.0)
  fh_true = fh_true_ref === nothing ? fhertz_true_from_states(u_mat, idxs) : fh_true_ref
  fh_pred, raw = fhertz_pred_and_raw_from_states(u_mat, idxs, p_net_struct, appr, st; nn_gain=nn_gain)
  eff_pos = eff_pos_ref === nothing ? effective_contact_positions(idxs, fh_true) : eff_pos_ref
  if isempty(eff_pos)
    return (
      fhertz_err=NaN,
      raw_min=NaN,
      raw_max=NaN,
      raw_mean=NaN,
      raw_neg_frac=NaN,
      fh_min=NaN,
      fh_max=NaN,
      fh_mean=NaN,
      fh_abs_p95=NaN
    )
  end
  raw_eff = raw[eff_pos]
  fh_eff = fh_pred[eff_pos]
  fh_abs = abs.(fh_eff)
  err_sum = sum(abs2, fh_pred[eff_pos] .- fh_true[eff_pos])
  truth_sum = sum(abs2, fh_true[eff_pos])
  fhertz_err = truth_sum <= 0.0 ? NaN : relative_rmse_pct(err_sum, truth_sum, length(eff_pos), 0.0)
  raw_neg_frac = count(<(0.0), raw_eff) / length(raw_eff)
  return (
    fhertz_err=fhertz_err,
    raw_min=minimum(raw_eff),
    raw_max=maximum(raw_eff),
    raw_mean=mean(raw_eff),
    raw_neg_frac=raw_neg_frac,
    fh_min=minimum(fh_eff),
    fh_max=maximum(fh_eff),
    fh_mean=mean(fh_eff),
    fh_abs_p95=quantile(fh_abs, 0.95)
  )
end

monitor_fhertz_true = fhertz_true_from_states(ode_data_full, monitor_idx_full)
monitor_eff_positions = effective_contact_positions(monitor_idx_full, monitor_fhertz_true)

# Inf logger to diagnose bad trials (set HNODECB_INF_LOG=0 to silence)
inf_log_remaining = Ref(20)
last_inf_reason = Ref{String}("")
inf_context = Ref{String}("unknown")
function log_inf(reason)
  Zygote.ignore() do
    if get(ENV, "HNODECB_INF_LOG", "1") != "1"
      return
    end
    ctx = inf_context[]
    last_inf_reason[] = ctx == "" ? string(reason) : string(ctx, ":", reason)
    if inf_log_remaining[] > 0
      tprintln("INF_REASON: ", last_inf_reason[])
      inf_log_remaining[] -= 1
    end
  end
end

# Optional trace to locate hangs during ODE solves (enable with HNODECB_TRACE_SOLVE=1)
trace_solve_enabled = get(ENV, "HNODECB_TRACE_SOLVE", "1") == "1"
trace_solve_limit = parse(Int, get(ENV, "HNODECB_TRACE_SOLVE_LIMIT", "20"))
trace_solve_count = Ref(0)
function trace_solve(msg)
  Zygote.ignore() do
    if !trace_solve_enabled
      return
    end
    if trace_solve_count[] < trace_solve_limit
      trace_solve_count[] += 1
      tprintln("TRACE_SOLVE: ", msg isa Function ? msg() : msg)
    end
  end
end

# Failure diagnostics (default ON): print why a trial turned Inf.
fail_diag_enabled = get(ENV, "HNODECB_FAIL_DIAG", "1") == "1"
function safe_prop(x, sym, default="na")
  return hasproperty(x, sym) ? getproperty(x, sym) : default
end
function summarize_destats(sol)
  ds = sol.destats
  return "nsteps=" * string(safe_prop(ds, :nsteps)) *
    " naccept=" * string(safe_prop(ds, :naccept)) *
    " nreject=" * string(safe_prop(ds, :nreject)) *
    " nf=" * string(safe_prop(ds, :nf)) *
    " njacs=" * string(safe_prop(ds, :njacs)) *
    " nsolve=" * string(safe_prop(ds, :nsolve))
end
function first_nonfinite_matrix(A)
  idx = findfirst(x -> !isfinite(x), A)
  if idx === nothing
    return "none"
  end
  rc = ind2sub(size(A), idx)
  return "row=" * string(rc[1]) * " col=" * string(rc[2]) * " val=" * string(A[idx])
end
function first_nonfinite_vector(v)
  idx = findfirst(x -> !isfinite(x), v)
  if idx === nothing
    return "none"
  end
  return "idx=" * string(idx) * " val=" * string(v[idx])
end
function log_failure_diag(tag, sol, p, appr, st, known_pars, expected_n; zero_nn_override::Bool=false, nn_gain::Float64=1.0)
  Zygote.ignore() do
    if !fail_diag_enabled
      return
    end
    t_last = isempty(sol.t) ? NaN : sol.t[end]
    tprintln("FAIL_DIAG[", tag, "]: retcode=", sol.retcode,
      " n=", length(sol.t), "/", expected_n,
      " t_last=", fmt_e(t_last, sigdigits=4))
    tprintln("FAIL_DIAG[", tag, "] destats: ", summarize_destats(sol))

    uhat = nothing
    try
      uhat = Array(sol)
    catch ex
      tprintln("FAIL_DIAG[", tag, "]: Array(sol) failed: ", sprint(showerror, ex))
      return
    end
    if isempty(uhat)
      tprintln("FAIL_DIAG[", tag, "]: empty trajectory")
      return
    end

    maxabs = maximum(abs, uhat)
    tprintln("FAIL_DIAG[", tag, "]: uhat size=", size(uhat),
      " max|u|=", fmt_e(maxabs, sigdigits=4),
      " first_nonfinite=", first_nonfinite_matrix(uhat))

    u_last = view(uhat, :, size(uhat, 2))
    k, wd, m, c, Fd, R, dist, Fad = known_pars
    ks = p.mech[1]
    cs = p.mech[2]
    s = dist + u_last[1] - u_last[3]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    w_pred = contact_weight(s, adhesion_transition)
    w_true = true_contact_weight_at_time(t_last)
    nn_out = if appr === nothing
      NaN
    else
      nn_in = nn_input_from_values(u_last[1], u_last[2], u_last[3])
      appr(nn_in, p.p_net, st)[1][1]
    end
    F_hertz = if zero_nn_override
      0.0
    elseif appr === nothing
      (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
    else
      nn_gain * nn_out * w_pred
    end
    Fad_eff = Fad * w_pred
    rhs_x2dot = (Fd * cos(wd * t_last) - k * u_last[1] - c * u_last[2] + Fad_eff - F_hertz) / m
    rhs_x3dot = (Fad_eff - F_hertz - ks * u_last[3]) / cs
    tprintln("FAIL_DIAG[", tag, "] phys: s=", fmt_e(s, sigdigits=4),
      " delta=", fmt_e(delta, sigdigits=4),
      " w_pred=", fmt_e(w_pred, sigdigits=4),
      " w_true=", fmt_e(w_true, sigdigits=4),
      " nn_out=", fmt_e(nn_out, sigdigits=4),
      " nn_gain=", fmt_e(nn_gain, sigdigits=4),
      " F_hertz=", fmt_e(F_hertz, sigdigits=4),
      " Fad_eff=", fmt_e(Fad_eff, sigdigits=4),
      " zero_override=", zero_nn_override)
    tprintln("FAIL_DIAG[", tag, "] rhs: x2dot=", fmt_e(rhs_x2dot, sigdigits=4),
      " x3dot=", fmt_e(rhs_x3dot, sigdigits=4),
      " ks=", fmt_e(ks, sigdigits=4),
      " cs=", fmt_e(cs, sigdigits=4))
  end
end

function make_train_val_masks(n, val_stride, val_offset)
  val_idx = [i for i in 1:n if (i - val_offset) % val_stride == 0]
  train_idx = [i for i in 1:n if !(i in val_idx)]
  return sort(train_idx), sort(val_idx)
end

function parse_rank_filter(spec::AbstractString, nmax::Int, env_key::AbstractString)
  spec = strip(spec)
  if spec == ""
    return nothing
  end
  idx = Int[]
  for token in split(spec, ",")
    t = strip(token)
    if t == ""
      continue
    end
    v = tryparse(Int, t)
    if v === nothing
      error(env_key * " contains a non-integer token: " * t)
    end
    push!(idx, v)
  end
  if isempty(idx)
    error(env_key * " is set but no valid indices were parsed")
  end
  idx = sort(unique(idx))
  for v in idx
    if v < 1 || v > nmax
      error(env_key * " index " * string(v) * " is outside 1.." * string(nmax))
    end
  end
  return idx
end

nn_hidden_layers_range = 0:2
nn_hidden_nodes_range = 1:3

nn_fixed_num_hidden_layers = 2
nn_fixed_num_hidden_nodes = 3

if !(nn_fixed_num_hidden_layers in nn_hidden_layers_range)
  error("nn_fixed_num_hidden_layers must lie in nn_hidden_layers_range")
end
if !(nn_fixed_num_hidden_nodes in nn_hidden_nodes_range)
  error("nn_fixed_num_hidden_nodes must lie in nn_hidden_nodes_range")
end

function build_nn(num_hidden_layers::Int, num_hidden_nodes::Int)
  if !(num_hidden_layers in nn_hidden_layers_range)
    error("num_hidden_layers=$(num_hidden_layers) is outside nn_hidden_layers_range=$(collect(nn_hidden_layers_range))")
  end
  if !(num_hidden_nodes in nn_hidden_nodes_range)
    error("num_hidden_nodes=$(num_hidden_nodes) is outside nn_hidden_nodes_range=$(collect(nn_hidden_nodes_range))")
  end
  hidden = 2^num_hidden_nodes
  layers = Any[]
  push!(layers, Lux.Dense(3, hidden, gelu; init_weight=my_glorot_uniform, use_bias=false))
  for _ in 1:num_hidden_layers
    push!(layers, Lux.Dense(hidden, hidden, gelu; init_weight=my_glorot_uniform, use_bias=false))
  end
  push!(layers, Lux.Dense(hidden, 1; init_weight=my_glorot_uniform, use_bias=false))
  return Lux.Chain(layers...)
end

nn_input_from_state(u) = u[1:3]
nn_input_from_values(x1, x2, x3) = [x1, x2, x3]

function make_uode_func(appr, st, known_pars; zero_nn_override::Bool=false, nn_gain::Float64=1.0)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  f(du, u, p, t) =
    let appr = appr, st = st, k = k, wd = wd, m = m, c = c, Fd = Fd, R = R, dist = dist, Fad = Fad
      ks = p.mech[1]
      cs = p.mech[2]

      s = dist + u[1] - u[3]
      delta = softplus(-s, adhesion_transition)
      delta = ifelse(delta > 0.0, delta, 0.0)
      w_pred = contact_weight(s, adhesion_transition)

      F_hertz = if zero_nn_override
        0.0
      elseif appr === nothing
        (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
      else
        nn_in = nn_input_from_state(u)
        uhat = appr(nn_in, p.p_net, st)[1]
        nn_gain * uhat[1] * w_pred
      end
      Fad_eff = Fad * w_pred

      @inbounds du[1] = u[2]
      @inbounds du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
      @inbounds du[3] = (Fad_eff - F_hertz - ks * u[3]) / cs
    end
  return f
end

function x2dot_rhs(u, mech, p_net, appr, st, known_pars, t; zero_nn_override::Bool=false, nn_gain::Float64=1.0)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  ks, cs = mech
  s = dist + u[1] - u[3]
  delta = softplus(-s, adhesion_transition)
  delta = ifelse(delta > 0.0, delta, 0.0)
  w_pred = contact_weight(s, adhesion_transition)
  F_hertz = if zero_nn_override
    0.0
  elseif appr === nothing
    (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
  else
    nn_in = nn_input_from_state(u)
    uhat = appr(nn_in, p_net, st)[1]
    nn_gain * uhat[1] * w_pred
  end
  Fad_eff = Fad * w_pred
  return (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
end

function loss_single_or_ms(θ, ode_data, x2dot_data, contact_mask, times,
  state12_scale, x2dot_scale, x3_scale,
  use_multiple_shooting, ms_group_size, ms_continuity_term,
  appr, st, known_pars, x3_t0_val,
  l2_weight, re_pnet; zero_nn_override::Bool=false, nn_gain::Float64=1.0)

  # map raw -> bounded mech params
  mech = [
    bound_param(θ.mech_raw[1], ks_bounds[1], ks_bounds[2]),
    bound_param(θ.mech_raw[2], cs_bounds[1], cs_bounds[2])
  ]
  p_net_struct = re_pnet(θ.p_net)
  p = ComponentVector(p_net=p_net_struct, mech=mech)

  weights_val = ifelse.(contact_mask, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_val)
  inf_diag = (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
  if !isfinite(weights_sum) || weights_sum <= 0
    log_inf("weights_sum")
    return Inf, inf_diag
  end

  total_state = 0.0
  total_x2dot = 0.0
  total_x3range = 0.0
  total_cont = 0.0
  x1_rec_ref = Ref(NaN)
  x3_rec_ref = Ref(NaN)

  if use_multiple_shooting
    ranges = DiffEqFlux.group_ranges(length(times), ms_group_size)
    preds = Vector{Matrix{Float64}}(undef, length(ranges))
    for (i, rg) in enumerate(ranges)
      u0 = [ode_data[1, first(rg)], ode_data[2, first(rg)], x3_t0_val]
      prob = ODEProblem{true}(make_uode_func(appr, st, known_pars; zero_nn_override=zero_nn_override, nn_gain=nn_gain), u0, (times[first(rg)], times[last(rg)]), p)
      trace_solve(() -> "ms solve start seg=" * string(i) * " len=" * string(length(rg)) *
        " t=" * string(round(times[first(rg)], sigdigits=3)) * "→" * string(round(times[last(rg)], sigdigits=3)))
        t_start = Zygote.ignore() do
          time()
        end
      sol = solve(prob, integrator; saveat=times[rg], abstol=abstol, reltol=reltol, sensealg=sensealg, maxiters=ode_maxiters)
      trace_solve(() -> "ms solve done seg=" * string(i) * " dt=" * string(round(time() - t_start, digits=2)) * "s" *
        " retcode=" * string(sol.retcode) * " size=" * string(size(sol, 2)))
      if !SciMLBase.successful_retcode(sol)
        log_inf("retcode=" * string(sol.retcode))
        log_failure_diag("ms_retcode_seg" * string(i), sol, p, appr, st, known_pars, length(rg); zero_nn_override=zero_nn_override, nn_gain=nn_gain)
        return Inf, inf_diag
      end
      if size(sol, 2) != length(rg)
        log_inf("sol_size_mismatch_ms")
        log_failure_diag("ms_size_seg" * string(i), sol, p, appr, st, known_pars, length(rg); zero_nn_override=zero_nn_override, nn_gain=nn_gain)
        return Inf, inf_diag
      end
      preds[i] = Array(sol)

      uhat = preds[i]
      if any(x -> !isfinite(x), uhat)
        log_inf("uhat_nonfinite_ms")
        log_failure_diag("ms_uhat_nonfinite_seg" * string(i), sol, p, appr, st, known_pars, length(rg); zero_nn_override=zero_nn_override, nn_gain=nn_gain)
        return Inf, inf_diag
      end
      seg_weights = weights_val[rg]
      seg_weights_sum = sum(seg_weights)
      if !isfinite(seg_weights_sum) || seg_weights_sum <= 0
        log_inf("seg_weights_sum")
        return Inf, inf_diag
      end

      state_err = vec(sum(abs2.((ode_data[1:2, rg] .- uhat[1:2, :]) ./ state12_scale), dims=1))
      total_state += sum(seg_weights .* state_err) / seg_weights_sum

      contact_idx = findall(contact_mask[rg])
      if !isempty(contact_idx)
        local_idx = rg[contact_idx]
        uhat_const = Zygote.dropgrad(uhat)
        x2dot_pred = [x2dot_rhs(view(uhat_const, :, j), mech, p_net_struct, appr, st, known_pars, times[rg[j]]; zero_nn_override=zero_nn_override, nn_gain=nn_gain) for j in 1:length(rg)]
        if any(x -> !isfinite(x), x2dot_pred)
          log_inf("x2dot_pred_nonfinite_ms")
          Zygote.ignore() do
            tprintln("FAIL_DIAG[ms_x2dot_nonfinite_seg", i, "]: first_nonfinite=", first_nonfinite_vector(x2dot_pred))
          end
          log_failure_diag("ms_x2dot_nonfinite_seg" * string(i), sol, p, appr, st, known_pars, length(rg); zero_nn_override=zero_nn_override, nn_gain=nn_gain)
          return Inf, inf_diag
        end
        x2_err = abs2.((x2dot_data[local_idx] .- x2dot_pred[contact_idx]) ./ x2dot_scale)
        x2_w = seg_weights[contact_idx]
        total_x2dot += sum(x2_w .* x2_err) / sum(x2_w)
      end

      exceed = abs.(uhat[3, :]) .- x3_range_amp
      range_pen = abs2.(softplus.(exceed, x3_range_eps) ./ x3_scale)
      total_x3range += x3_range_weight * (sum(seg_weights .* range_pen) / seg_weights_sum)
    end

    for i in 2:length(preds)
      u0 = preds[i-1][:, end]
      u1 = preds[i][:, 1]
      total_cont += ms_continuity_term * sum(abs2, u0 - u1)
    end

    Zygote.ignore() do
      x1_err_sum = 0.0
      x1_truth_sum = 0.0
      x3_err_sum = 0.0
      x3_truth_sum = 0.0
      count = 0
      for (i, rg) in enumerate(ranges)
        uhat_seg = preds[i]
        x1_err_sum += sum(abs2.(ode_data[1, rg] .- uhat_seg[1, :]))
        x1_truth_sum += sum(abs2.(ode_data[1, rg]))
        x3_err_sum += sum(abs2.(ode_data[3, rg] .- uhat_seg[3, :]))
        x3_truth_sum += sum(abs2.(ode_data[3, rg]))
        count += length(rg)
      end
      x1_rec_ref[] = relative_rmse_pct(x1_err_sum, x1_truth_sum, count, scale_eps)
      x3_rec_ref[] = relative_rmse_pct(x3_err_sum, x3_truth_sum, count, scale_eps)
    end
  else
    u0 = [ode_data[1, 1], ode_data[2, 1], x3_t0_val]
    prob = ODEProblem{true}(make_uode_func(appr, st, known_pars; zero_nn_override=zero_nn_override, nn_gain=nn_gain), u0, (times[1], times[end]), p)
    trace_solve(() -> "ss solve start n=" * string(length(times)) *
      " t=" * string(round(times[1], sigdigits=3)) * "→" * string(round(times[end], sigdigits=3)))
    t_start = Zygote.ignore() do
      time()
    end
    sol = solve(prob, integrator; saveat=times, abstol=abstol, reltol=reltol, sensealg=sensealg, maxiters=ode_maxiters)
    trace_solve(() -> "ss solve done dt=" * string(round(time() - t_start, digits=2)) * "s" *
      " retcode=" * string(sol.retcode) * " size=" * string(size(sol, 2)))
    if !SciMLBase.successful_retcode(sol)
      log_inf("retcode=" * string(sol.retcode))
      log_failure_diag("ss_retcode", sol, p, appr, st, known_pars, length(times); zero_nn_override=zero_nn_override, nn_gain=nn_gain)
      return Inf, inf_diag
    end
    if size(sol, 2) != length(times)
      log_inf("sol_size_mismatch_ss")
      log_failure_diag("ss_size", sol, p, appr, st, known_pars, length(times); zero_nn_override=zero_nn_override, nn_gain=nn_gain)
      return Inf, inf_diag
    end
    uhat = Array(sol)
    if any(x -> !isfinite(x), uhat)
      log_inf("uhat_nonfinite_ss")
      log_failure_diag("ss_uhat_nonfinite", sol, p, appr, st, known_pars, length(times); zero_nn_override=zero_nn_override, nn_gain=nn_gain)
      return Inf, inf_diag
    end

    state_err = vec(sum(abs2.((ode_data[1:2, :] .- uhat[1:2, :]) ./ state12_scale), dims=1))
    total_state = sum(weights_val .* state_err) / weights_sum

    contact_idx = findall(contact_mask)
    if !isempty(contact_idx)
      uhat_const = Zygote.dropgrad(uhat)
      x2dot_pred = [x2dot_rhs(view(uhat_const, :, j), mech, p_net_struct, appr, st, known_pars, times[j]; zero_nn_override=zero_nn_override, nn_gain=nn_gain) for j in 1:length(times)]
      if any(x -> !isfinite(x), x2dot_pred)
        log_inf("x2dot_pred_nonfinite_ss")
        Zygote.ignore() do
          tprintln("FAIL_DIAG[ss_x2dot_nonfinite]: first_nonfinite=", first_nonfinite_vector(x2dot_pred))
        end
        log_failure_diag("ss_x2dot_nonfinite", sol, p, appr, st, known_pars, length(times); zero_nn_override=zero_nn_override, nn_gain=nn_gain)
        return Inf, inf_diag
      end
      x2_err = abs2.((x2dot_data[contact_idx] .- x2dot_pred[contact_idx]) ./ x2dot_scale)
      x2_w = weights_val[contact_idx]
      total_x2dot = sum(x2_w .* x2_err) / sum(x2_w)
    end

    exceed = abs.(uhat[3, :]) .- x3_range_amp
    range_pen = abs2.(softplus.(exceed, x3_range_eps) ./ x3_scale)
    total_x3range = x3_range_weight * (sum(weights_val .* range_pen) / weights_sum)

    Zygote.ignore() do
      x1_err_sum = sum(abs2.(ode_data[1, :] .- uhat[1, :]))
      x1_truth_sum = sum(abs2.(ode_data[1, :]))
      x3_err_sum = sum(abs2.(ode_data[3, :] .- uhat[3, :]))
      x3_truth_sum = sum(abs2.(ode_data[3, :]))
      count = length(times)
      x1_rec_ref[] = relative_rmse_pct(x1_err_sum, x1_truth_sum, count, scale_eps)
      x3_rec_ref[] = relative_rmse_pct(x3_err_sum, x3_truth_sum, count, scale_eps)
    end
  end

  l2_penalty = l2_weight * sum(abs2, θ.p_net)
  total = total_state + total_x2dot + total_x3range + total_cont + l2_penalty
  return total, (state=total_state, x2dot=total_x2dot, x3_range=total_x3range, cont=total_cont,
    x1_rec=x1_rec_ref[], x3_rec=x3_rec_ref[])
end

function stage1plus_window_indices(times::AbstractVector, window_span::Float64)
  n = length(times)
  if n == 0
    error("stage1plus window selection received an empty time vector")
  end
  if !isfinite(window_span) || window_span <= 0
    return collect(1:n)
  end
  t0 = times[1]
  stop_idx = findlast(t -> (t - t0) <= window_span, times)
  if stop_idx === nothing
    stop_idx = 1
  end
  return collect(1:max(1, stop_idx))
end

function normalize_min_obj_metric(values::Vector{Float64})
  finite_vals = [v for v in values if isfinite(v)]
  if isempty(finite_vals)
    return fill(1.0, length(values))
  end
  vmin = minimum(finite_vals)
  vmax = maximum(finite_vals)
  span = vmax - vmin
  if !isfinite(span) || span <= 0
    return [isfinite(v) ? 0.0 : 1.0 for v in values]
  end
  return [isfinite(v) ? (v - vmin) / span : 1.0 for v in values]
end

function score_architecture_trials_running(trial_recs, obj_a::Float64, obj_b::Float64, obj_c::Float64)
  isempty(trial_recs) && return Any[]
  obj1_vals = Float64[rec.obj1 for rec in trial_recs]
  obj2_vals = Float64[rec.obj2 for rec in trial_recs]
  obj3_vals = Float64[rec.obj3 for rec in trial_recs]
  obj1_norm = normalize_min_obj_metric(obj1_vals)
  obj2_norm = normalize_min_obj_metric(obj2_vals)
  obj3_norm = normalize_min_obj_metric(obj3_vals)
  scored = Any[]
  for i in eachindex(trial_recs)
    rec = trial_recs[i]
    min_obj = obj_a * obj1_norm[i] + obj_b * obj2_norm[i] + obj_c * obj3_norm[i]
    push!(scored, merge(rec, (
      norm_obj1=obj1_norm[i],
      norm_obj2=obj2_norm[i],
      norm_obj3=obj3_norm[i],
      weighted_obj1=obj_a * obj1_norm[i],
      weighted_obj2=obj_b * obj2_norm[i],
      weighted_obj3=obj_c * obj3_norm[i],
      min_obj=min_obj
    )))
  end
  return scored
end

function stage1plus_architecture_trial(base_rec, base_rank, rep,
  num_hidden_layers::Int, num_hidden_nodes::Int,
  ode_train, x2dot_train, contact_train, times_train,
  ode_val, x2dot_val, contact_val, times_val,
  state12_scale, x2dot_scale, x3_t0_val,
  ms_group_size, ms_continuity_term,
  zero_nn_override::Bool, arch_epochs::Int, arch_lr::Float64,
  obj_a::Float64, obj_b::Float64, obj_c::Float64)

  base_params = hasproperty(base_rec, :params) ? base_rec.params : Dict{Any, Any}()
  ks_fixed = hasproperty(base_rec, :ks_hat) ? base_rec.ks_hat :
    (haskey(base_params, "ks0") ? base_params["ks0"] : NaN)
  cs_fixed = hasproperty(base_rec, :cs_hat) ? base_rec.cs_hat :
    (haskey(base_params, "cs0") ? base_params["cs0"] : NaN)

  rng_trial = StableRNG(10_000_000 * num_hidden_layers + 1_000_000 * num_hidden_nodes + 10_000 * base_rank + rep)
  seed = abs(rand(rng_trial, Int))
  rng = StableRNG(seed)
  approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)
  p_net_init, st = Lux.setup(rng, approximating_neural_network)
  p_net_init = Flux.f64(p_net_init)
  p_net_vec0, re_pnet = Optimisers.destructure(p_net_init)
  gnn_info = compute_stage1plus_gnn_scale(
    ode_data_full, monitor_idx_full, re_pnet(p_net_vec0),
    approximating_neural_network, st
  )
  g_nn = gnn_info.g_nn

  raw_init = [
    raw_from_value(ks_fixed, ks_bounds[1], ks_bounds[2]),
    raw_from_value(cs_fixed, cs_bounds[1], cs_bounds[2])
  ]

  theta0 = ComponentVector(p_net=p_net_vec0, mech_raw=raw_init)

  function train_loss_fn(theta)
    loss_single_or_ms(theta, ode_train, x2dot_train, contact_train, times_train,
      state12_scale, x2dot_scale, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override, nn_gain=g_nn)[1]
  end

  function val_loss_fn(theta)
    loss_single_or_ms(theta, ode_val, x2dot_val, contact_val, times_val,
      state12_scale, x2dot_scale, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override, nn_gain=g_nn)[1]
  end

  lr_adapt = get(ENV, "HNODECB_STAGE1PLUS_ARCH_LR_ADAPT", get(ENV, "HNODECB_LR_ADAPT", "1")) == "1"
  lr = arch_lr
  lr_min = env_float("HNODECB_STAGE1PLUS_ARCH_LR_MIN", env_float("HNODECB_LR_MIN", 1e-30))
  lr_max = env_float("HNODECB_STAGE1PLUS_ARCH_LR_MAX", env_float("HNODECB_LR_MAX", 5e1))
  lr_eta = env_float("HNODECB_STAGE1PLUS_ARCH_LR_ETA", env_float("HNODECB_LR_ETA", 0.5))
  lr_ema_alpha = env_float("HNODECB_STAGE1PLUS_ARCH_LR_EMA", env_float("HNODECB_LR_EMA", 0.85))
  lr_eps = env_float("HNODECB_STAGE1PLUS_ARCH_LR_EPS", env_float("HNODECB_LR_EPS", 1e-30))
  lr_target_init = env_float("HNODECB_STAGE1PLUS_ARCH_LR_TARGET", NaN)
  grad_ema = Ref(isfinite(lr_target_init) && lr_target_init > 0 ? lr_target_init : NaN)
  grad_target = Ref(isfinite(lr_target_init) && lr_target_init > 0 ? lr_target_init : NaN)

  grad_scale_p_net = env_float("HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_P_NET", 0.5)
  grad_scale_mech = env_float("HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_MECH", 1.0)
  group_adapt = get(ENV, "HNODECB_STAGE1PLUS_ARCH_GROUP_ADAPT", "1") == "1"
  group_ema_alpha = env_float("HNODECB_STAGE1PLUS_ARCH_GROUP_EMA", 0.90)
  group_eta = env_float("HNODECB_STAGE1PLUS_ARCH_GROUP_ETA", 0.05)
  group_eps = env_float("HNODECB_STAGE1PLUS_ARCH_GROUP_EPS", 1e-30)
  grad_scale_p_min = env_float("HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_P_NET_MIN", 0.1)
  grad_scale_p_max = env_float("HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_P_NET_MAX", 1.0)
  grad_scale_m_min = env_float("HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_MECH_MIN", 0.5)
  grad_scale_m_max = env_float("HNODECB_STAGE1PLUS_ARCH_GRAD_SCALE_MECH_MAX", 1.5)
  grad_scale_p_net = clamp(grad_scale_p_net, grad_scale_p_min, grad_scale_p_max)
  grad_scale_mech = clamp(grad_scale_mech, grad_scale_m_min, grad_scale_m_max)
  p_grad_ema = Ref(NaN)
  m_grad_ema = Ref(NaN)

  step_guard = get(ENV, "HNODECB_STAGE1PLUS_ARCH_STEP_GUARD", "1") == "1"
  step_retry_max = max(0, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_STEP_RETRIES", "6")))
  step_retry_lr_factor = env_float("HNODECB_STAGE1PLUS_ARCH_STEP_RETRY_LR_FACTOR", 0.1)
  step_max_loss_frac = env_float("HNODECB_STAGE1PLUS_ARCH_STEP_MAX_LOSS_FRAC", 0.1)

  val_loss_start = val_loss_fn(theta0)
  theta = deepcopy(theta0)
  opt_state = Optimisers.setup(Optimisers.Adam(lr), theta)
  epoch_durations = Float64[]
  t2_gt_abs_t1_count = 0
  retry_count_total = 0
  completed_epochs = 0
  val_loss_last = val_loss_start
  train_loss_last = Inf
  trial_tag = "layers=" * string(num_hidden_layers) *
    " nodes=" * string(num_hidden_nodes) *
    " base=" * string(base_rank) *
    " rep=" * string(rep)

  for epoch in 1:arch_epochs
    epoch_t0 = Zygote.ignore() do
      time()
    end
    train_loss_before, back = Zygote.pullback(train_loss_fn, theta)
    grad_raw = first(back(1.0))
    if grad_raw === nothing
      tprintln("    arch epoch ", epoch, "/", arch_epochs,
        " -- ", trial_tag, " | grad=None -> stop")
      break
    end
    grad_norm_raw = grad_norm_safe(grad_raw)
    if !isfinite(train_loss_before) || !isfinite(grad_norm_raw)
      tprintln("    arch epoch ", epoch, "/", arch_epochs,
        " -- ", trial_tag,
        " | nonfinite train/grad -> stop",
        " | train=", fmt_e(train_loss_before, sigdigits=4),
        " grad_norm=", fmt_e(grad_norm_raw, sigdigits=4))
      break
    end

    gnorm_p_raw, gnorm_m_raw = grad_group_norms(grad_raw)
    if group_adapt && isfinite(gnorm_p_raw) && isfinite(gnorm_m_raw) &&
       gnorm_p_raw > 0.0 && gnorm_m_raw > 0.0
      if !isfinite(p_grad_ema[])
        p_grad_ema[] = gnorm_p_raw
      else
        p_grad_ema[] = group_ema_alpha * p_grad_ema[] + (1 - group_ema_alpha) * gnorm_p_raw
      end
      if !isfinite(m_grad_ema[])
        m_grad_ema[] = gnorm_m_raw
      else
        m_grad_ema[] = group_ema_alpha * m_grad_ema[] + (1 - group_ema_alpha) * gnorm_m_raw
      end
      eff_p = grad_scale_p_net * p_grad_ema[]
      eff_m = grad_scale_mech * m_grad_ema[]
      if isfinite(eff_p) && isfinite(eff_m) && eff_p > 0.0 && eff_m > 0.0
        r = log((eff_p + group_eps) / (eff_m + group_eps))
        mult = exp(-group_eta * r)
        grad_scale_p_net = clamp(grad_scale_p_net * mult, grad_scale_p_min, grad_scale_p_max)
        grad_scale_mech = clamp(grad_scale_mech / mult, grad_scale_m_min, grad_scale_m_max)
      end
    end
    grad_update = scale_stage1plus_grad(grad_raw, grad_scale_p_net, grad_scale_mech)
    grad_norm_scaled = grad_norm_safe(grad_update)

    if lr_adapt && isfinite(grad_norm_scaled) && grad_norm_scaled > 0
      if !isfinite(grad_ema[])
        grad_ema[] = grad_norm_scaled
      else
        grad_ema[] = lr_ema_alpha * grad_ema[] + (1 - lr_ema_alpha) * grad_norm_scaled
      end
      if !isfinite(grad_target[])
        grad_target[] = grad_ema[]
      end
      ratio = grad_target[] / (grad_ema[] + lr_eps)
      lr_new = clamp(lr * ratio^lr_eta, lr_min, lr_max)
      if lr_new != lr
        Optimisers.adjust!(opt_state, lr_new)
        lr = lr_new
      end
    end

    theta_base = deepcopy(theta)
    opt_state_base = deepcopy(opt_state)
    train_loss_after = Inf
    val_loss_epoch = Inf
    t1 = NaN
    t2 = NaN
    recovered_after = 0
    epoch_bad_t2 = false
    step_fail_reason = ""
    step_accepted = false

    if !step_guard
      opt_state, theta = Optimisers.update(opt_state, theta, grad_update)
      train_loss_after = train_loss_fn(theta)
      val_loss_epoch = val_loss_fn(theta)
      step_vec = theta .- theta_base
      t1 = sum(grad_raw .* step_vec)
      t2 = (train_loss_after - train_loss_before) - t1
      epoch_bad_t2 = !(isfinite(t2) && isfinite(t1)) || t2 > abs(t1)
    else
      step_attempt = 0
      while step_attempt <= step_retry_max
        step_attempt += 1
        opt_input = deepcopy(opt_state_base)
        opt_trial, theta_trial = Optimisers.update(opt_input, theta_base, grad_update)
        train_loss_trial = train_loss_fn(theta_trial)
        step_vec = theta_trial .- theta_base
        t1_trial = sum(grad_raw .* step_vec)
        t2_trial = (train_loss_trial - train_loss_before) - t1_trial
        if !(isfinite(t2_trial) && isfinite(t1_trial)) || t2_trial > abs(t1_trial)
          epoch_bad_t2 = true
        end
        step_max_loss_increase = isfinite(train_loss_before) ? abs(train_loss_before) * step_max_loss_frac : Inf
        if !isfinite(train_loss_trial)
          step_fail_reason = "step_trial_loss_nonfinite"
        elseif isfinite(step_max_loss_increase) && train_loss_trial > train_loss_before + step_max_loss_increase
          step_fail_reason = "step_trial_loss_jump"
        else
          step_fail_reason = ""
        end

        if step_fail_reason == ""
          opt_state = opt_trial
          theta = theta_trial
          train_loss_after = train_loss_trial
          val_loss_epoch = val_loss_fn(theta)
          t1 = t1_trial
          t2 = t2_trial
          step_accepted = true
          recovered_after = step_attempt - 1
          retry_count_total += recovered_after
          if recovered_after > 0
            tprintln("      step-guard: accepted after ", recovered_after,
              " retry/reduction(s) | lr=", fmt_e(lr, sigdigits=3),
              " | train=", fmt_e(train_loss_after, sigdigits=4))
          end
          break
        end

        if step_attempt > step_retry_max
          retry_count_total += step_retry_max
          tprintln("      step-guard: rejected update after ", step_retry_max,
            " retries | reason=", step_fail_reason,
            " | lr=", fmt_e(lr, sigdigits=3))
          theta = theta_base
          opt_state = opt_state_base
          train_loss_after = train_loss_before
          val_loss_epoch = val_loss_fn(theta)
          t1 = t1_trial
          t2 = t2_trial
          break
        end

        lr_new = max(lr * step_retry_lr_factor, lr_min)
        if lr_new < lr
          lr = lr_new
        end
        Optimisers.adjust!(opt_state_base, lr)
        tprintln("      step-guard retry ", step_attempt, "/", step_retry_max,
          " -- reason=", step_fail_reason,
          " | lr=", fmt_e(lr, sigdigits=3),
          " | scale[p_net=", fmt_e(grad_scale_p_net, sigdigits=3),
          ", mech_raw=", fmt_e(grad_scale_mech, sigdigits=3), "]")
      end
      if !step_accepted && train_loss_after === Inf
        train_loss_after = train_loss_before
        val_loss_epoch = val_loss_fn(theta)
      end
    end

    if epoch_bad_t2
      t2_gt_abs_t1_count += 1
    end
    epoch_dt = Zygote.ignore() do
      time() - epoch_t0
    end
    push!(epoch_durations, epoch_dt)
    tprintln("    arch epoch ", epoch, "/", arch_epochs,
      " -- ", trial_tag,
      " | train=", fmt_e(train_loss_after, sigdigits=4),
      " val=", fmt_e(val_loss_epoch, sigdigits=4),
      " | dt=", fmt_f(epoch_dt, digits=2), "s",
      " | t1=", fmt_e(t1, sigdigits=4),
      " t2=", fmt_e(t2, sigdigits=4),
      " | lr=", fmt_e(lr, sigdigits=3),
      " | scale[p_net=", fmt_e(grad_scale_p_net, sigdigits=3),
      ", mech_raw=", fmt_e(grad_scale_mech, sigdigits=3), "]")
    train_loss_last = train_loss_after
    val_loss_last = val_loss_epoch
    completed_epochs += 1
  end

  val_loss_end = completed_epochs > 0 ? val_loss_last : val_loss_fn(theta)
  trial_complete = completed_epochs == arch_epochs && isfinite(val_loss_start) && isfinite(val_loss_end)
  time_per_epoch = trial_complete ? (sum(epoch_durations) / arch_epochs) : Inf
  obj1 = trial_complete ? (val_loss_end - val_loss_start) / arch_epochs : Inf
  obj2 = time_per_epoch
  obj3 = trial_complete ? Float64(t2_gt_abs_t1_count) : Float64(arch_epochs)

  return (
    base_rank=base_rank,
    rep=rep,
    num_hidden_layers=num_hidden_layers,
    num_hidden_nodes=num_hidden_nodes,
    hidden=2^num_hidden_nodes,
    param_count=length(p_net_vec0),
    trial_complete=trial_complete,
    completed_epochs=completed_epochs,
    train_loss_end=train_loss_last,
    val_loss_start=val_loss_start,
    val_loss_end=val_loss_end,
    time_per_epoch=time_per_epoch,
    t2_gt_abs_t1_count=t2_gt_abs_t1_count,
    retry_count_total=retry_count_total,
    obj1=obj1,
    obj2=obj2,
    obj3=obj3,
    g_nn=g_nn,
    g_target=gnn_info.target,
    g_amp_ref=gnn_info.amp_ref
  )
end

function summarize_architecture_trials(arch_trials, obj_a::Float64, obj_b::Float64, obj_c::Float64)
  grouped = Dict{Tuple{Int, Int}, Vector{Any}}()
  for rec in arch_trials
    key = (rec.num_hidden_layers, rec.num_hidden_nodes)
    if !haskey(grouped, key)
      grouped[key] = Any[]
    end
    push!(grouped[key], rec)
  end

  summaries = Any[]
  for (key, recs) in grouped
    push!(summaries, (
      num_hidden_layers=key[1],
      num_hidden_nodes=key[2],
      hidden=2^key[2],
      param_count=Int(round(mean([rec.param_count for rec in recs]))),
      n_trials=length(recs),
      mean_obj1=mean([rec.obj1 for rec in recs]),
      mean_obj2=mean([rec.obj2 for rec in recs]),
      mean_obj3=mean([rec.obj3 for rec in recs]),
      mean_norm_obj1=mean([rec.norm_obj1 for rec in recs]),
      mean_norm_obj2=mean([rec.norm_obj2 for rec in recs]),
      mean_norm_obj3=mean([rec.norm_obj3 for rec in recs]),
      mean_weighted_obj1=mean([rec.weighted_obj1 for rec in recs]),
      mean_weighted_obj2=mean([rec.weighted_obj2 for rec in recs]),
      mean_weighted_obj3=mean([rec.weighted_obj3 for rec in recs]),
      mean_min_obj=mean([rec.min_obj for rec in recs]),
      mean_retry_count=mean([rec.retry_count_total for rec in recs]),
      mean_completed_epochs=mean([rec.completed_epochs for rec in recs]),
      mean_val_loss_start=mean([rec.val_loss_start for rec in recs]),
      mean_val_loss_end=mean([rec.val_loss_end for rec in recs]),
      mean_g_nn=mean([rec.g_nn for rec in recs]),
      mean_g_target=mean([rec.g_target for rec in recs]),
      mean_g_amp_ref=mean([rec.g_amp_ref for rec in recs])
    ))
  end

  sort!(summaries, by = rec -> (rec.num_hidden_layers, rec.num_hidden_nodes))
  scored = Any[
    merge(rec, (
      norm_obj1=rec.mean_norm_obj1,
      norm_obj2=rec.mean_norm_obj2,
      norm_obj3=rec.mean_norm_obj3,
      weighted_obj1=rec.mean_weighted_obj1,
      weighted_obj2=rec.mean_weighted_obj2,
      weighted_obj3=rec.mean_weighted_obj3,
      min_obj=rec.mean_min_obj
    )) for rec in summaries
  ]

  sort!(scored, by = rec -> rec.min_obj)
  return scored
end

use_multiple_shooting = false
use_l2_regularization = false
l2_weight = 0.0
# Fixed train/validation split (not part of hyperparameter search)
val_stride = 5
val_offset = 2
# Stage1: default to NO-ADAM flow unless explicitly enabled
use_adam = get(ENV, "HNODECB_STAGE1_USE_ADAM", "0") == "1"

selftest = get(ENV, "HNODECB_SELFTEST", "0") == "1"
if selftest
  tprintln(use_stage1plus ? "=== AFM Stage1plus (03) SELFTEST ===" : "=== AFM Stage1 (03) SELFTEST ===")
else
  tprintln(use_stage1plus ? "=== AFM Stage1plus (03) ===" : "=== AFM Stage1 (03) ===")
end
tprintln("Switches: MS=", use_multiple_shooting, " L2=", use_l2_regularization, " (λ=", l2_weight, ")")
tprintln("Stage1 ADAM: ", use_adam ? "ON" : "OFF")
tprintln("Stage1plus mode: ", use_stage1plus ? "ON" : "OFF")
if use_adam && !use_stage1plus
  tprintln("Stage1 warm-start export: OFF (ADAM branch stores no NN states)")
else
  tprintln("Stage1 warm-start export: ON (top candidates will save NN states for Stage2)")
end

if selftest
  # Minimal runtime check: build NN, solve ODE once, compute one loss
  n = min(50, length(all_times))
  idx = 1:n
  ode_train = ode_data_full[:, idx]
  times_train = all_times[idx]
  x2dot_train = x2dot_all[idx]
  contact_train = contact_all[idx]

  num_hidden_layers = nn_fixed_num_hidden_layers
  num_hidden_nodes = nn_fixed_num_hidden_nodes
  ms_group_size = min(10, n)
  ms_continuity_term = 1e-3

  tprintln("  selftest: building NN + init params")
  rng = StableRNG(0)
  approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)
  p_net, st = Lux.setup(rng, approximating_neural_network)
  p_net_vec, re_pnet = Optimisers.destructure(p_net)

  # Use log-mid parameters (geometric mean, matches log-uniform search) for a neutral selftest run
  inf_context[] = "selftest"
  last_inf_reason[] = ""
  ks_mid = sqrt(ks_bounds[1] * ks_bounds[2])
  cs_mid = sqrt(cs_bounds[1] * cs_bounds[2])
  raw_init = [
    raw_from_value(ks_mid, ks_bounds[1], ks_bounds[2]),
    raw_from_value(cs_mid, cs_bounds[1], cs_bounds[2])
  ]
  θ0 = ComponentVector(p_net=p_net_vec, mech_raw=raw_init)
  x3_t0_val = ode_train[3, 1]
  if get(ENV, "HNODECB_SELFTEST_PHYS", "1") == "1"
    # Print physical scale diagnostics at initial time
    p_net_struct = re_pnet(θ0.p_net)
    mech = [
      bound_param(θ0.mech_raw[1], ks_bounds[1], ks_bounds[2]),
      bound_param(θ0.mech_raw[2], cs_bounds[1], cs_bounds[2])
    ]
    u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
    t0 = times_train[1]
    k, wd, m, c, Fd, R, dist, Fad = known_pars
    s = dist + u0[1] - u0[3]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    w_pred = contact_weight(s, adhesion_transition)
    w_true = true_contact_weight_at_time(t0)
    nn_in = nn_input_from_values(u0[1], u0[2], u0[3])
    uhat = approximating_neural_network(nn_in, p_net_struct, st)[1]
    F_hertz = uhat[1] * w_pred
    Fad_eff = Fad * w_pred
    x2dot0 = (Fd * cos(wd * t0) - k * u0[1] - c * u0[2] + Fad_eff - F_hertz) / m
    x3dot0 = (Fad_eff - F_hertz - mech[1] * u0[3]) / mech[2]
    tprintln("  selftest phys @t0=", fmt_e(t0, sigdigits=3))
    tprintln("    u0: x1=", fmt_e(u0[1], sigdigits=3),
      " x2=", fmt_e(u0[2], sigdigits=3),
      " x3=", fmt_e(u0[3], sigdigits=3))
    tprintln("    contact: s=", fmt_e(s, sigdigits=3),
      " delta=", fmt_e(delta, sigdigits=3),
      " w_pred=", fmt_e(w_pred, sigdigits=3),
      " w_true=", fmt_e(w_true, sigdigits=3))
    tprintln("    nn_out=", fmt_e(uhat[1], sigdigits=3),
      " F_hertz=", fmt_e(F_hertz, sigdigits=3),
      " Fad_eff=", fmt_e(Fad_eff, sigdigits=3))
    tprintln("    x2dot0=", fmt_e(x2dot0, sigdigits=3),
      " x3dot0=", fmt_e(x3dot0, sigdigits=3))
  end
  loss, diag = loss_single_or_ms(θ0, ode_train, x2dot_train, contact_train, times_train,
    state12_scale_full, x2dot_scale_full, x3_scale,
    use_multiple_shooting, ms_group_size, ms_continuity_term,
    approximating_neural_network, st, known_pars, x3_t0_val,
    l2_weight, re_pnet)

  tprintln("  selftest loss=", fmt_e(loss, sigdigits=4))
  if diag !== nothing
    tprintln("  selftest parts: state=", fmt_e(diag.state, sigdigits=3),
      " x2dot=", fmt_e(diag.x2dot, sigdigits=3),
      " x3r=", fmt_e(diag.x3_range, sigdigits=3),
      " cont=", fmt_e(diag.cont, sigdigits=3))
  end
  if !isfinite(loss)
    reason = last_inf_reason[] == "" ? "unknown" : last_inf_reason[]
    error("SELFTEST failed: non-finite loss (reason=" * reason * ")")
  end
  tprintln("=== SELFTEST OK ===")
else
  last_trial_metrics = Ref((train_loss=Inf, val_loss=Inf))
  last_trial_reasons = Ref((train="", val=""))
  did_preflight = Ref(false)
  last_diag = Ref{Any}(nothing)
  log_every = parse(Int, get(ENV, "HNODECB_STAGE1_LOG_EVERY", "1"))

  # Sanity check: perfect mech params + oracle NN (for monitoring only)
  if get(ENV, "HNODECB_STAGE1_SANITY", "1") == "1"
    tprintln("=== Stage1 SANITY (oracle F_hertz + true mech) ===")
    train_idx, _ = make_train_val_masks(length(all_times), val_stride, val_offset)
    ode_train = ode_data_full[:, train_idx]
    times_train = all_times[train_idx]
    x2dot_train = x2dot_all[train_idx]
    contact_train = contact_all[train_idx]
    contact_idx_train = findall(contact_train)

    # dummy NN params (oracle ignores them)
    rng = StableRNG(0)
    sanity_nn = build_nn(1, 3)
    p_net, st = Lux.setup(rng, sanity_nn)
    p_net_vec, re_pnet = Optimisers.destructure(p_net)

    raw_init = [
      raw_from_value(ks_true, ks_bounds[1], ks_bounds[2]),
      raw_from_value(cs_true, cs_bounds[1], cs_bounds[2])
    ]
    θ_sanity = ComponentVector(p_net=p_net_vec, mech_raw=raw_init)
    x3_t0_val = ode_train[3, 1]

    loss, diag = loss_single_or_ms(θ_sanity, ode_train, x2dot_train, contact_train, times_train,
      state12_scale_full, x2dot_scale_full, x3_scale,
      use_multiple_shooting, 10, 1e-3,
      nothing, st, known_pars, x3_t0_val,
      0.0, re_pnet)

    tprintln("  sanity loss=", fmt_e(loss, sigdigits=4))
    if diag !== nothing
      tprintln("  sanity parts: state=", fmt_e(diag.state, sigdigits=3),
        " x2dot=", fmt_e(diag.x2dot, sigdigits=3),
        " x3r=", fmt_e(diag.x3_range, sigdigits=3),
        " cont=", fmt_e(diag.cont, sigdigits=3))
      tprintln("  sanity rec: x1=", fmt_f(diag.x1_rec, digits=2),
        "% x3=", fmt_f(diag.x3_rec, digits=2), "%")
    end

    if use_adam && get(ENV, "HNODECB_LOG_FHERTZ_ERR", "1") == "1"
      tprintln("  sanity nn: F_hertz err=0.0%")
    end
  end

function objective(trial)
  try
    last_inf_reason[] = ""
    inf_context[] = "objective"
    inf_log_remaining[] = 20
    # hyperparameters
    ks0 = trial.suggest_float("ks0", ks_bounds[1], ks_bounds[2], log=true)
    cs0 = trial.suggest_float("cs0", cs_bounds[1], cs_bounds[2], log=true)
    learning_rate_adam = trial.suggest_float("learning_rate_adam", 1e-5, 1e-2, log=true)
    num_hidden_layers = nn_fixed_num_hidden_layers
    num_hidden_nodes = nn_fixed_num_hidden_nodes
    ms_group_size = trial.suggest_int("ms_group_size", 10, 200)
    ms_continuity_term = trial.suggest_float("ms_continuity_term", 1e-6, 10.0, log=true)
    # train/val split
    train_idx, val_idx = make_train_val_masks(length(all_times), val_stride, val_offset)

    ode_train = ode_data_full[:, train_idx]
    ode_val = ode_data_full[:, val_idx]
    times_train = all_times[train_idx]
    times_val = all_times[val_idx]
    x2dot_train = x2dot_all[train_idx]
    x2dot_val = x2dot_all[val_idx]
    contact_train = contact_all[train_idx]
    contact_val = contact_all[val_idx]
    contact_idx_train = findall(contact_train)

    state12_scale_train = state12_scale_full
    x2dot_scale_train = x2dot_scale_full

    fhertz_monitor_enabled = get(ENV, "HNODECB_LOG_FHERTZ_ERR", "1") == "1" && !isempty(monitor_eff_positions)

    # NN architecture
    tprintln("  building NN + init params")
    flush(stdout)
    seed = abs(rand(rng_global, Int))
    rng = StableRNG(seed)
    approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)
    p_net, st = Lux.setup(rng, approximating_neural_network)
    p_net_vec, re_pnet = Optimisers.destructure(p_net)

    # initial parameters (bounded raw)
    raw_init = [
      raw_from_value(ks0, ks_bounds[1], ks_bounds[2]),
      raw_from_value(cs0, cs_bounds[1], cs_bounds[2])
    ]
    θ0 = ComponentVector(p_net=p_net_vec, mech_raw=raw_init)

    # x3 initial value (allowed: first contact)
    x3_t0_val = ode_train[3, 1]

    # loss function
    function loss_fn(θ)
      inf_context[] = "train"
      loss, diag = loss_single_or_ms(θ, ode_train, x2dot_train, contact_train, times_train,
        state12_scale_train, x2dot_scale_train, x3_scale,
        use_multiple_shooting, ms_group_size, ms_continuity_term,
        approximating_neural_network, st, known_pars, x3_t0_val,
        l2_weight, re_pnet)
      last_diag[] = diag
      return loss, diag
    end

    # Optional preflight to locate hangs (only runs once)
    if !did_preflight[] && get(ENV, "HNODECB_STAGE1_PREFLIGHT", "0") == "1"
      tprintln("  preflight loss")
      @time loss_fn(θ0)
      if get(ENV, "HNODECB_STAGE1_PREFLIGHT_GRAD", "0") == "1"
        tprintln("  preflight grad")
        @time Zygote.gradient(x -> loss_fn(x)[1], θ0)
      end
      did_preflight[] = true
    end

    # ADAM optimization (manual loop to access grad_norm + adaptive lr)
    tprintln("  starting ADAM")
    flush(stdout)

    maxiters = 300
    training_costs = fill(Inf, maxiters)
    stuck = Ref(false)

    lr_adapt = get(ENV, "HNODECB_LR_ADAPT", "0") == "1"
    lr_min = env_float("HNODECB_LR_MIN", 1e-6)
    lr_max = env_float("HNODECB_LR_MAX", 1e-2)
    lr_eta = env_float("HNODECB_LR_ETA", 0.05)
    lr_ema_alpha = env_float("HNODECB_LR_EMA", 0.97)
    lr_eps = env_float("HNODECB_LR_EPS", 1e-30)
    lr_target_init = env_float("HNODECB_LR_TARGET", NaN)

    lr = learning_rate_adam
    opt_state = Optimisers.setup(Optimisers.Adam(lr), θ0)
    θ = θ0
    grad_ema = Ref(lr_target_init)
    grad_target = Ref(lr_target_init)

    for epoch in 1:maxiters
      inf_context[] = "train"
      last_inf_reason[] = ""

      # loss + grad
      loss = Inf
      grad = nothing
      try
        loss, back = Zygote.pullback(x -> loss_fn(x)[1], θ)
        grad = first(back(1.0))
      catch ex
        last_inf_reason[] = "exception: " * sprint(showerror, ex)
        bt = catch_backtrace()
        tprintln("EXCEPTION: ", sprint(showerror, ex, bt))
        stuck[] = true
        break
      end

      training_costs[epoch] = loss
      # Print every epoch to monitor runtime
      tprintln("Stage1 trial epoch ", epoch, " train=", fmt_e(loss, sigdigits=4))

      gnorm = grad_norm_safe(grad)
      lr_changed = false
      if lr_adapt && isfinite(gnorm) && gnorm > 0
        if !isfinite(grad_ema[])
          grad_ema[] = gnorm
        else
          grad_ema[] = lr_ema_alpha * grad_ema[] + (1 - lr_ema_alpha) * gnorm
        end
        if !isfinite(grad_target[])
          grad_target[] = grad_ema[]
        end
        ratio = grad_target[] / (grad_ema[] + lr_eps)
        lr_new = clamp(lr * ratio^lr_eta, lr_min, lr_max)
        if lr_new != lr
          Optimisers.adjust!(opt_state, lr_new)
          lr = lr_new
          lr_changed = true
        end
      end

      if get(ENV, "HNODECB_LOG_GRADNORM", "1") == "1"
        lr_note = lr_changed ? " (adapt)" : ""
        tprintln("  grad_norm=", fmt_e(gnorm, sigdigits=3),
          " lr=", fmt_e(lr, sigdigits=3), lr_note)
      end

      if log_every > 0 && (epoch % log_every == 0)
        diag = last_diag[]
        if diag !== nothing
          tprintln("  parts: state=", fmt_e(diag.state, sigdigits=3),
            " x2dot=", fmt_e(diag.x2dot, sigdigits=3),
            " x3r=", fmt_e(diag.x3_range, sigdigits=3),
            " cont=", fmt_e(diag.cont, sigdigits=3))
        end
        ks_hat = bound_param(θ.mech_raw[1], ks_bounds[1], ks_bounds[2])
        cs_hat = bound_param(θ.mech_raw[2], cs_bounds[1], cs_bounds[2])
        tprintln("  mech: ks=", fmt_e(ks_hat, sigdigits=3),
          " (err=", fmt_f(rel_err_pct(ks_hat, ks_true, scale_eps), digits=2), "%)",
          " cs=", fmt_e(cs_hat, sigdigits=3),
          " (err=", fmt_f(rel_err_pct(cs_hat, cs_true, scale_eps), digits=2), "%)")
        if diag !== nothing
          tprintln("  rec: x1=", fmt_f(diag.x1_rec, digits=2), "% x3=",
            fmt_f(diag.x3_rec, digits=2), "%")
        end
        if fhertz_monitor_enabled
          Zygote.ignore() do
            p_net_struct = re_pnet(θ.p_net)
            nn_metrics = fhertz_monitor_metrics(
              ode_data_full, monitor_idx_full, p_net_struct,
              approximating_neural_network, st;
              fh_true_ref=monitor_fhertz_true, eff_pos_ref=monitor_eff_positions
            )
            tprintln("  nn: F_hertz err=", fmt_f(nn_metrics.fhertz_err, digits=2), "%")
            tprintln("  nn raw: min=", fmt_e(nn_metrics.raw_min, sigdigits=3),
              " max=", fmt_e(nn_metrics.raw_max, sigdigits=3),
              " mean=", fmt_e(nn_metrics.raw_mean, sigdigits=3),
              " neg=", fmt_f(100 * nn_metrics.raw_neg_frac, digits=2), "%")
          end
        end
      end

      if !isfinite(loss)
        log_inf("train_loss_nonfinite")
        stuck[] = true
        break
      end

      # update params
      opt_state, θ = Optimisers.update(opt_state, θ, grad)

      if epoch > 10 && minimum(training_costs[max(1, epoch-5):epoch]) > 1e6
        stuck[] = true
        break
      end
    end

    θ_best = θ

    if stuck[]
      last_inf_reason[] = "train:stuck"
      last_trial_metrics[] = (train_loss=Inf, val_loss=Inf)
      last_trial_reasons[] = (train=last_inf_reason[], val=last_inf_reason[])
      return Inf
    end

    # training loss (with L2)
    inf_context[] = "train_eval"
    last_inf_reason[] = ""
    train_loss, _ = loss_single_or_ms(θ_best, ode_train, x2dot_train, contact_train, times_train,
      state12_scale_full, x2dot_scale_full, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      l2_weight, re_pnet)
    train_reason = last_inf_reason[]

    # validation loss (no L2)
    inf_context[] = "val_eval"
    last_inf_reason[] = ""
    val_loss, _ = loss_single_or_ms(θ_best, ode_val, x2dot_val, contact_val, times_val,
      state12_scale_full, x2dot_scale_full, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet)
    val_reason = last_inf_reason[]

    last_trial_metrics[] = (train_loss=train_loss, val_loss=val_loss)
    last_trial_reasons[] = (train=train_reason, val=val_reason)
    return val_loss
  catch ex
    last_inf_reason[] = "exception: " * sprint(showerror, ex)
    last_trial_metrics[] = (train_loss=Inf, val_loss=Inf)
    last_trial_reasons[] = (train=last_inf_reason[], val=last_inf_reason[])
    bt = catch_backtrace()
    tprintln("EXCEPTION: ", sprint(showerror, ex, bt))
    tprintln("trial params = ", Dict(trial.params))
    return Inf
  end
end

if use_adam
  # TPE optimization (ADAM on)
  num_trials = 200
  study = optuna.create_study(sampler=optuna.samplers.TPESampler(consider_prior=false, multivariate=true, seed=0))
  trial_parameters = []

  # Optional sharding to avoid duplicate trials across processes
  shard_idx = parse(Int, get(ENV, "HNODECB_STAGE1_SHARD_INDEX", "1"))
  shard_cnt = parse(Int, get(ENV, "HNODECB_STAGE1_SHARD_COUNT", "1"))
  if shard_idx < 1 || shard_idx > shard_cnt
    error("HNODECB_STAGE1_SHARD_INDEX must be in 1..HNODECB_STAGE1_SHARD_COUNT")
  end
  trial_indices = [i for i in 1:num_trials if ((i - shard_idx) % shard_cnt) == 0]

  for optuna_iteration in trial_indices
    tprintln("Stage1 trial ", optuna_iteration, " start")
    flush(stdout)
    cost = Inf
    params = Dict{Any, Any}()
    metrics = (train_loss=Inf, val_loss=Inf)
    reasons = (train="unknown", val="unknown")
    try
      trial = study.ask()
      cost = objective(trial)
      # always record the trial, even if cost is Inf/NaN
      study.tell(trial, cost)
      params = Dict(trial.params)
      params["trial_id"] = optuna_iteration
      metrics = last_trial_metrics[]
      reasons = last_trial_reasons[]
    catch ex
      bt = catch_backtrace()
      tprintln("TRIAL_FATAL: ", sprint(showerror, ex, bt))
      last_inf_reason[] = "trial_fatal: " * sprint(showerror, ex)
    end

    push!(trial_parameters, (loss=cost, train_loss=metrics.train_loss, val_loss=metrics.val_loss, params=params))
    tprintln("Stage1 trial ", optuna_iteration,
      " -- train=", fmt_e(metrics.train_loss, sigdigits=4),
      " val=", fmt_e(metrics.val_loss, sigdigits=4))
    if !isfinite(metrics.train_loss) || !isfinite(metrics.val_loss)
      train_reason = reasons.train == "" ? "unknown" : reasons.train
      val_reason = reasons.val == "" ? "unknown" : reasons.val
      tprintln("  Inf reason: train=", train_reason, " | val=", val_reason)
    end
  end

  # summarize
  sorted = sort(trial_parameters, by = r -> r.loss)
  best = sorted[1]
  tprintln("Stage1 done. Best val loss=", fmt_e(best.val_loss, sigdigits=4),
    " | train=", fmt_e(best.train_loss, sigdigits=4))

  tprintln("Stage1 top-10 summary (train/val):")
  for (i, rec) in enumerate(sorted[1:min(10, length(sorted))])
    ks0 = rec.params["ks0"]
    cs0 = rec.params["cs0"]
    ks_hat = hasproperty(rec, :ks_hat) ? rec.ks_hat : ks0
    cs_hat = hasproperty(rec, :cs_hat) ? rec.cs_hat : cs0
    ks_err_pct = hasproperty(rec, :ks_err_pct) ? rec.ks_err_pct : NaN
    cs_err_pct = hasproperty(rec, :cs_err_pct) ? rec.cs_err_pct : NaN
    tprintln("  Rank ", i,
      " -- train=", fmt_e(rec.train_loss, sigdigits=4),
      " val=", fmt_e(rec.val_loss, sigdigits=4),
      " | ks0=", fmt_e(ks0, sigdigits=3),
      " cs0=", fmt_e(cs0, sigdigits=3))
    tprintln("     val parts: state=", fmt_e(rec.val_parts.state, sigdigits=3),
      " x2dot=", fmt_e(rec.val_parts.x2dot, sigdigits=3),
      " x3r=", fmt_e(rec.val_parts.x3_range, sigdigits=3),
      " cont=", fmt_e(rec.val_parts.cont, sigdigits=3))
    tprintln("     rec: x1=", fmt_f(rec.val_parts.x1_rec, digits=2),
      "% x3=", fmt_f(rec.val_parts.x3_rec, digits=2), "%")
    tprintln("     mech: ks=", fmt_e(ks_hat, sigdigits=3),
      " (err=", fmt_f(ks_err_pct, digits=2), "%)",
      " cs=", fmt_e(cs_hat, sigdigits=3),
      " (err=", fmt_f(cs_err_pct, digits=2), "%)")
  end

  serialize(result_folder * "/" * result_name_string, (
    study=nothing,
    trial_parameters=trial_parameters,
    warm_start_top=Any[],
    best=best,
    bounds=(ks=ks_bounds, cs=cs_bounds),
    true_values=(ks=ks_true, cs=cs_true),
    use_multiple_shooting=use_multiple_shooting,
    use_l2_regularization=use_l2_regularization,
    val_stride=val_stride,
    val_offset=val_offset,
    x3_obs_fraction=0.0,
    error_level=error_level
  ))
elseif use_stage1plus
  let
  function rand_loguniform(rng, lo, hi)
    return exp(rand(rng) * (log(hi) - log(lo)) + log(lo))
  end

  stage1_input_basename = get(ENV, "HNODECB_STAGE1PLUS_INPUT_BASENAME", "afm_param_stage1_03.jld")
  stage1_input_file = normpath(joinpath(result_folder, stage1_input_basename))
  if !isfile(stage1_input_file)
    error("Stage1plus: missing Stage1 input file: " * stage1_input_file)
  end
  stage1_base = deserialize(stage1_input_file)
  base_trials = haskey(stage1_base, :trial_parameters) ? stage1_base.trial_parameters : Any[]
  if isempty(base_trials)
    error("Stage1plus: no candidates found in Stage1 input file.")
  end

  stage1plus_input_topk = min(parse(Int, get(ENV, "HNODECB_STAGE1PLUS_INPUT_TOPK", "10")), length(base_trials))
  searches_per_candidate = parse(Int, get(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE", "600"))
  final_topk = parse(Int, get(ENV, "HNODECB_STAGE1PLUS_FINAL_TOPK", "9"))
  base_filter_spec = get(ENV, "HNODECB_STAGE1PLUS_BASE_INDICES", "")
  zero_nn_override_stage1plus = get(ENV, "HNODECB_STAGE1PLUS_ZERO_NN", "1") == "1"
  shard_idx = parse(Int, get(ENV, "HNODECB_STAGE1_SHARD_INDEX", "1"))
  shard_cnt = parse(Int, get(ENV, "HNODECB_STAGE1_SHARD_COUNT", "1"))
  if shard_idx < 1 || shard_idx > shard_cnt
    error("HNODECB_STAGE1_SHARD_INDEX must be in 1..HNODECB_STAGE1_SHARD_COUNT")
  end

  fixed_ms_group_size = 10
  fixed_ms_continuity_term = 1e-3
  arch_selected_layers = begin
    raw = strip(get(ENV, "HNODECB_STAGE1PLUS_SELECTED_LAYERS", string(nn_fixed_num_hidden_layers)))
    v = tryparse(Int, raw)
    (v === nothing) ? nn_fixed_num_hidden_layers : v
  end
  arch_selected_nodes = begin
    raw = strip(get(ENV, "HNODECB_STAGE1PLUS_SELECTED_NODES", string(nn_fixed_num_hidden_nodes)))
    v = tryparse(Int, raw)
    (v === nothing) ? nn_fixed_num_hidden_nodes : v
  end
  if !(arch_selected_layers in nn_hidden_layers_range)
    error("HNODECB_STAGE1PLUS_SELECTED_LAYERS=$(arch_selected_layers) is outside nn_hidden_layers_range=$(collect(nn_hidden_layers_range))")
  end
  if !(arch_selected_nodes in nn_hidden_nodes_range)
    error("HNODECB_STAGE1PLUS_SELECTED_NODES=$(arch_selected_nodes) is outside nn_hidden_nodes_range=$(collect(nn_hidden_nodes_range))")
  end
  arch_screen_only = get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_ONLY", "0") == "1"
  arch_screen_trials_per_arch = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH", "10")))
  arch_screen_epochs = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_EPOCHS", "20")))
  arch_screen_warmup = get(ENV, "HNODECB_STAGE1PLUS_ARCH_WARMUP", "1") == "1"
  arch_screen_warmup_epochs = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_WARMUP_EPOCHS", "1")))
  arch_screen_window_us = env_float("HNODECB_STAGE1PLUS_ARCH_WINDOW_US", 5e-6)
  arch_screen_lr = env_float("HNODECB_STAGE1PLUS_ARCH_LR", 1e-3)
  arch_obj_a = env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_A", 0.35)
  arch_obj_b = env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_B", 0.45)
  arch_obj_c = env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_C", 0.20)
  fhertz_monitor_enabled = !isempty(monitor_eff_positions)
  sorted_base = sort(base_trials, by = r -> r.loss)
  base_candidates = sorted_base[1:stage1plus_input_topk]
  base_rank_indices = collect(1:length(base_candidates))
  base_filter_indices = parse_rank_filter(base_filter_spec, length(base_candidates), "HNODECB_STAGE1PLUS_BASE_INDICES")
  if base_filter_indices !== nothing
    allowed = Set(base_filter_indices)
    base_rank_indices = [i for i in base_rank_indices if i in allowed]
    tprintln("Stage1plus base candidate filter: [", join(base_filter_indices, ", "), "]")
  end
  if isempty(base_rank_indices)
    error("Stage1plus: no base candidates assigned after HNODECB_STAGE1PLUS_BASE_INDICES filter.")
  end

  if arch_screen_only
    arch_window_idx = stage1plus_window_indices(all_times, arch_screen_window_us)
    ode_window = ode_data_full[:, arch_window_idx]
    times_window = all_times[arch_window_idx]
    x2dot_window = x2dot_all[arch_window_idx]
    contact_window = contact_all[arch_window_idx]
    train_idx_screen, val_idx_screen = make_train_val_masks(length(times_window), val_stride, val_offset)
    if isempty(train_idx_screen) || isempty(val_idx_screen)
      error("Stage1plus architecture screen window is too small to form train/val splits")
    end
    ode_train_screen = ode_window[:, train_idx_screen]
    ode_val_screen = ode_window[:, val_idx_screen]
    x2dot_train_screen = x2dot_window[train_idx_screen]
    x2dot_val_screen = x2dot_window[val_idx_screen]
    contact_train_screen = contact_window[train_idx_screen]
    contact_val_screen = contact_window[val_idx_screen]
    times_train_screen = times_window[train_idx_screen]
    times_val_screen = times_window[val_idx_screen]
    x3_t0_screen = ode_train_screen[3, 1]
    arch_trials = Any[]

    tprintln("Stage1plus architecture screen: ON")
    tprintln("  window_us=", fmt_e(arch_screen_window_us, sigdigits=4),
      " | points=", length(arch_window_idx),
      " | tspan=[", fmt_e(times_window[1], sigdigits=4), ", ", fmt_e(times_window[end], sigdigits=4), "]")
    tprintln("  trials_per_arch=", arch_screen_trials_per_arch,
      " | epochs=", arch_screen_epochs,
      " | lr=", fmt_e(arch_screen_lr, sigdigits=3),
      " | obj_weights[a=", fmt_f(arch_obj_a, digits=2),
      " b=", fmt_f(arch_obj_b, digits=2),
      " c=", fmt_f(arch_obj_c, digits=2), "]")
    tprintln("  dummy_warmup=", arch_screen_warmup ? "ON" : "OFF",
      arch_screen_warmup ? " | epochs=" * string(arch_screen_warmup_epochs) : "")

    if arch_screen_warmup
      warmup_layers = first(nn_hidden_layers_range)
      warmup_nodes = first(nn_hidden_nodes_range)
      warmup_base_rank = first(base_rank_indices)
      warmup_base_rec = base_candidates[warmup_base_rank]
      tprintln("  architecture screen warm-up: layers=", warmup_layers,
        " nodes=", warmup_nodes,
        " hidden=", 2^warmup_nodes,
        " | base=", warmup_base_rank,
        " | epochs=", arch_screen_warmup_epochs)
      stage1plus_architecture_trial(
        warmup_base_rec, warmup_base_rank, 0,
        warmup_layers, warmup_nodes,
        ode_train_screen, x2dot_train_screen, contact_train_screen, times_train_screen,
        ode_val_screen, x2dot_val_screen, contact_val_screen, times_val_screen,
        state12_scale_full, x2dot_scale_full, x3_t0_screen,
        fixed_ms_group_size, fixed_ms_continuity_term,
        zero_nn_override_stage1plus, arch_screen_warmup_epochs, arch_screen_lr,
        arch_obj_a, arch_obj_b, arch_obj_c
      )
    end

    for num_hidden_layers in nn_hidden_layers_range
      for num_hidden_nodes in nn_hidden_nodes_range
        tprintln("  screening arch: layers=", num_hidden_layers,
          " nodes=", num_hidden_nodes,
          " hidden=", 2^num_hidden_nodes)
        for base_rank in base_rank_indices
          base_rec = base_candidates[base_rank]
          for rep in 1:arch_screen_trials_per_arch
            rec = stage1plus_architecture_trial(
              base_rec, base_rank, rep,
              num_hidden_layers, num_hidden_nodes,
              ode_train_screen, x2dot_train_screen, contact_train_screen, times_train_screen,
              ode_val_screen, x2dot_val_screen, contact_val_screen, times_val_screen,
              state12_scale_full, x2dot_scale_full, x3_t0_screen,
              fixed_ms_group_size, fixed_ms_continuity_term,
              zero_nn_override_stage1plus, arch_screen_epochs, arch_screen_lr,
              arch_obj_a, arch_obj_b, arch_obj_c
            )
            arch_trials_same = Any[
              r for r in arch_trials
              if r.num_hidden_layers == num_hidden_layers &&
                 r.num_hidden_nodes == num_hidden_nodes
            ]
            arch_progress_scored = score_architecture_trials_running(
              vcat(arch_trials_same, Any[rec]), arch_obj_a, arch_obj_b, arch_obj_c
            )
            current_trial = only([r for r in arch_progress_scored if
              r.num_hidden_layers == num_hidden_layers &&
              r.num_hidden_nodes == num_hidden_nodes &&
              r.base_rank == rec.base_rank && r.rep == rec.rep])
            push!(arch_trials, current_trial)
            tprintln("      trial min_obj after ", arch_screen_epochs,
              " epochs -- layers=", num_hidden_layers,
              " nodes=", num_hidden_nodes,
              " base=", base_rank,
              " rep=", rep,
              " | min_obj=", fmt_e(current_trial.min_obj, sigdigits=4))
            tprintln("         detail: (", fmt_f(arch_obj_a, digits=2), ")*part_a=",
              fmt_e(current_trial.weighted_obj1, sigdigits=4),
              ", raw_a=", fmt_e(current_trial.obj1, sigdigits=4),
              ", norm_a=", fmt_e(current_trial.norm_obj1, sigdigits=4),
              " ; (", fmt_f(arch_obj_b, digits=2), ")*part_b=",
              fmt_e(current_trial.weighted_obj2, sigdigits=4),
              ", raw_b=", fmt_e(current_trial.obj2, sigdigits=4),
              ", norm_b=", fmt_e(current_trial.norm_obj2, sigdigits=4),
              " ; (", fmt_f(arch_obj_c, digits=2), ")*part_c=",
              fmt_e(current_trial.weighted_obj3, sigdigits=4),
              ", raw_c=", fmt_e(current_trial.obj3, sigdigits=4),
              ", norm_c=", fmt_e(current_trial.norm_obj3, sigdigits=4))
          end
        end
      end
    end

    arch_ranked = summarize_architecture_trials(arch_trials, arch_obj_a, arch_obj_b, arch_obj_c)
    selected_arch = arch_ranked[1]
    tprintln("Stage1plus architecture ranking (9 architectures):")
    for (rank, rec) in enumerate(arch_ranked)
      tprintln("  Rank ", rank,
        " -- layers=", rec.num_hidden_layers,
        " nodes=", rec.num_hidden_nodes,
        " hidden=", rec.hidden,
        " params=", rec.param_count,
        " | min_obj=", fmt_e(rec.min_obj, sigdigits=4))
      tprintln("     min_obj detail: (", fmt_f(arch_obj_a, digits=2), ")*part_a=",
        fmt_e(rec.weighted_obj1, sigdigits=4),
        ", raw_a=", fmt_e(rec.mean_obj1, sigdigits=4),
        ", norm_a=", fmt_e(rec.norm_obj1, sigdigits=4),
        " ; (", fmt_f(arch_obj_b, digits=2), ")*part_b=",
        fmt_e(rec.weighted_obj2, sigdigits=4),
        ", raw_b=", fmt_e(rec.mean_obj2, sigdigits=4),
        ", norm_b=", fmt_e(rec.norm_obj2, sigdigits=4),
        " ; (", fmt_f(arch_obj_c, digits=2), ")*part_c=",
        fmt_e(rec.weighted_obj3, sigdigits=4),
        ", raw_c=", fmt_e(rec.mean_obj3, sigdigits=4),
        ", norm_c=", fmt_e(rec.norm_obj3, sigdigits=4))
    end
    tprintln("Stage1plus architecture winner: layers=", selected_arch.num_hidden_layers,
      " nodes=", selected_arch.num_hidden_nodes,
      " hidden=", selected_arch.hidden,
      " | min_obj=", fmt_e(selected_arch.min_obj, sigdigits=4))

    serialize(result_folder * "/" * result_name_string, (
      study=nothing,
      trial_parameters=arch_trials,
      warm_start_top=Any[],
      selected=arch_ranked,
      best=selected_arch,
      architecture_ranked=arch_ranked,
      selected_architecture=(
        num_hidden_layers=selected_arch.num_hidden_layers,
        num_hidden_nodes=selected_arch.num_hidden_nodes
      ),
      bounds=(ks=ks_bounds, cs=cs_bounds),
      true_values=(ks=ks_true, cs=cs_true),
      use_multiple_shooting=use_multiple_shooting,
      use_l2_regularization=use_l2_regularization,
      val_stride=val_stride,
      val_offset=val_offset,
      x3_obs_fraction=0.0,
      error_level=error_level,
      stage1_input_file=stage1_input_file,
      stage1_input_topk=stage1plus_input_topk,
      arch_trials_per_arch=arch_screen_trials_per_arch,
      arch_epochs=arch_screen_epochs,
      arch_window_us=arch_screen_window_us,
      arch_lr=arch_screen_lr,
      arch_obj_weights=(a=arch_obj_a, b=arch_obj_b, c=arch_obj_c)
    ))
  else
    fixed_num_hidden_layers = arch_selected_layers
    fixed_num_hidden_nodes = arch_selected_nodes
  assignments = Tuple{Int, Int, Int}[]
  global_trial_counter = 0
  for base_rank in base_rank_indices
    for rep in 1:searches_per_candidate
      global_trial_counter += 1
      if ((global_trial_counter - shard_idx) % shard_cnt) == 0
        push!(assignments, (global_trial_counter, base_rank, rep))
      end
    end
  end

  train_idx, val_idx = make_train_val_masks(length(all_times), val_stride, val_offset)
  ode_train = ode_data_full[:, train_idx]
  ode_val = ode_data_full[:, val_idx]
  times_train = all_times[train_idx]
  times_val = all_times[val_idx]
  x2dot_train = x2dot_all[train_idx]
  x2dot_val = x2dot_all[val_idx]
  contact_train = contact_all[train_idx]
  contact_val = contact_all[val_idx]

  state12_scale_train = state12_scale_full
  x2dot_scale_train = x2dot_scale_full
  x3_t0_val = ode_train[3, 1]

  trial_parameters = []
  tprintln("Stage1plus input: ", stage1_input_file)
  tprintln("Stage1plus shard ", shard_idx, "/", shard_cnt,
    " | base_candidates=", length(base_rank_indices),
    " | searches_per_candidate=", searches_per_candidate,
    " | local_trials=", length(assignments),
    " | total_trials=", length(base_rank_indices) * searches_per_candidate)
  tprintln("Stage1plus mode: NN-only random search | NN monitor=",
    fhertz_monitor_enabled ? "ON" : "OFF",
    " | zero_nn_override=", zero_nn_override_stage1plus ? "ON" : "OFF",
    " | g_nn=", stage1pluslight_gnn_enabled ? "ON" : "OFF",
    " | g_target_mode=", stage1pluslight_gnn_target_mode,
    " | fixed NN: layers=", fixed_num_hidden_layers,
    " nodes=", fixed_num_hidden_nodes)

  for (global_trial_id, base_rank, rep) in assignments
    base_rec = base_candidates[base_rank]
    base_params = hasproperty(base_rec, :params) ? base_rec.params : Dict{Any, Any}()
    ks_fixed = hasproperty(base_rec, :ks_hat) ? base_rec.ks_hat :
      (haskey(base_params, "ks0") ? base_params["ks0"] : NaN)
    cs_fixed = hasproperty(base_rec, :cs_hat) ? base_rec.cs_hat :
      (haskey(base_params, "cs0") ? base_params["cs0"] : NaN)
    base_trial_id = haskey(base_params, "trial_id") ? base_params["trial_id"] : missing

    tprintln("Stage1plus trial ", global_trial_id,
      " start (base_rank=", base_rank, ", rep=", rep, ")")
    flush(stdout)

    rng_trial = StableRNG(1_000_000 * base_rank + rep)
    num_hidden_layers = fixed_num_hidden_layers
    num_hidden_nodes = fixed_num_hidden_nodes
    seed = abs(rand(rng_trial, Int))
    rng = StableRNG(seed)
    approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)
    p_net, st = Lux.setup(rng, approximating_neural_network)
    p_net = Flux.f64(p_net)
    p_net_vec, re_pnet = Optimisers.destructure(p_net)
    p_net_struct = re_pnet(p_net_vec)
    gnn_info = compute_stage1plus_gnn_scale(
      ode_data_full, monitor_idx_full, p_net_struct,
      approximating_neural_network, st
    )
    g_nn = gnn_info.g_nn
    raw_init = [
      raw_from_value(ks_fixed, ks_bounds[1], ks_bounds[2]),
      raw_from_value(cs_fixed, cs_bounds[1], cs_bounds[2])
    ]
    θ0 = ComponentVector(p_net=p_net_vec, mech_raw=raw_init)

    inf_context[] = "train_eval"
    last_inf_reason[] = ""
    train_loss, diag = loss_single_or_ms(θ0, ode_train, x2dot_train, contact_train, times_train,
      state12_scale_train, x2dot_scale_train, x3_scale,
      use_multiple_shooting, fixed_ms_group_size, fixed_ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override_stage1plus, nn_gain=g_nn)
    train_reason = last_inf_reason[]

    inf_context[] = "val_eval"
    last_inf_reason[] = ""
    val_loss, val_diag = loss_single_or_ms(θ0, ode_val, x2dot_val, contact_val, times_val,
      state12_scale_full, x2dot_scale_full, x3_scale,
      use_multiple_shooting, fixed_ms_group_size, fixed_ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override_stage1plus, nn_gain=g_nn)
    val_reason = last_inf_reason[]

    ks_hat = bound_param(θ0.mech_raw[1], ks_bounds[1], ks_bounds[2])
    cs_hat = bound_param(θ0.mech_raw[2], cs_bounds[1], cs_bounds[2])
    ks_err_pct = rel_err_pct(ks_hat, ks_true, scale_eps)
    cs_err_pct = rel_err_pct(cs_hat, cs_true, scale_eps)
    val_nn_metrics = Ref((
      fhertz_err=NaN,
      raw_min=NaN,
      raw_max=NaN,
      raw_mean=NaN,
      raw_neg_frac=NaN,
      fh_min=NaN,
      fh_max=NaN,
      fh_mean=NaN,
      fh_abs_p95=NaN
    ))
    if fhertz_monitor_enabled
      Zygote.ignore() do
        p_net_struct = re_pnet(θ0.p_net)
        val_nn_metrics[] = fhertz_monitor_metrics(
          ode_data_full, monitor_idx_full, p_net_struct,
          approximating_neural_network, st;
          fh_true_ref=monitor_fhertz_true, eff_pos_ref=monitor_eff_positions, nn_gain=g_nn
        )
      end
    end
    is_viable = isfinite(train_loss) && isfinite(val_loss) &&
      (!fhertz_monitor_enabled || isfinite(val_nn_metrics[].fhertz_err))

    params = Dict{Any, Any}(
      "trial_id" => global_trial_id,
      "base_rank" => base_rank,
      "base_trial_id" => base_trial_id,
      "ks0" => ks_fixed,
      "cs0" => cs_fixed,
      "g_nn" => g_nn,
      "g_target" => gnn_info.target,
      "g_target_mode" => gnn_info.target_mode,
      "g_target_est" => gnn_info.target_est,
      "g_amp_ref" => gnn_info.amp_ref,
      "g_eff_count" => gnn_info.eff_count,
      "g_p95_mx2dot" => gnn_info.p95_mx2dot,
      "g_p95_kx1" => gnn_info.p95_kx1,
      "g_p95_cx2" => gnn_info.p95_cx2,
      "g_p95_fd" => gnn_info.p95_fd,
      "g_p95_fadh" => gnn_info.p95_fadh,
      "num_hidden_layers" => num_hidden_layers,
      "num_hidden_nodes" => num_hidden_nodes
    )
    push!(trial_parameters, (
      loss=val_loss,
      train_loss=train_loss,
      val_loss=val_loss,
      params=params,
      val_parts=val_diag,
      ks_hat=ks_hat,
      cs_hat=cs_hat,
      ks_err_pct=ks_err_pct,
      cs_err_pct=cs_err_pct,
      val_nn_err=val_nn_metrics[].fhertz_err,
      val_raw_min=val_nn_metrics[].raw_min,
      val_raw_max=val_nn_metrics[].raw_max,
      val_raw_mean=val_nn_metrics[].raw_mean,
      val_raw_neg_frac=val_nn_metrics[].raw_neg_frac,
      val_fh_min=val_nn_metrics[].fh_min,
      val_fh_max=val_nn_metrics[].fh_max,
      val_fh_mean=val_nn_metrics[].fh_mean,
      val_fh_abs_p95=val_nn_metrics[].fh_abs_p95,
      is_viable=is_viable,
      train_reason=train_reason,
      val_reason=val_reason,
      p_net_vec=copy(p_net_vec),
      g_nn=g_nn,
      g_target=gnn_info.target,
      g_target_mode=gnn_info.target_mode,
      g_target_est=gnn_info.target_est,
      g_amp_ref=gnn_info.amp_ref,
      g_eff_count=gnn_info.eff_count,
      g_p95_mx2dot=gnn_info.p95_mx2dot,
      g_p95_kx1=gnn_info.p95_kx1,
      g_p95_cx2=gnn_info.p95_cx2,
      g_p95_fd=gnn_info.p95_fd,
      g_p95_fadh=gnn_info.p95_fadh
    ))

    tprintln("Stage1plus trial ", global_trial_id,
      " -- train=", fmt_e(train_loss, sigdigits=4),
      " val=", fmt_e(val_loss, sigdigits=4))
    if !is_viable
      train_reason = train_reason == "" ? "unknown" : train_reason
      val_reason = val_reason == "" ? "unknown" : val_reason
      tprintln("  filtered: viable=false | train=", train_reason, " | val=", val_reason)
    end
    if diag !== nothing
      tprintln("  parts: state=", fmt_e(diag.state, sigdigits=3),
        " x2dot=", fmt_e(diag.x2dot, sigdigits=3),
        " x3r=", fmt_e(diag.x3_range, sigdigits=3),
        " cont=", fmt_e(diag.cont, sigdigits=3))
    end
    tprintln("  base: rank=", base_rank,
      " ks=", fmt_e(ks_fixed, sigdigits=3),
      " cs=", fmt_e(cs_fixed, sigdigits=3))
    tprintln("  mech: ks=", fmt_e(ks_hat, sigdigits=3),
      " (err=", fmt_f(ks_err_pct, digits=2), "%)",
      " cs=", fmt_e(cs_hat, sigdigits=3),
      " (err=", fmt_f(cs_err_pct, digits=2), "%)")
    if diag !== nothing
      tprintln("  rec: x1=", fmt_f(diag.x1_rec, digits=2), "% x3=",
        fmt_f(diag.x3_rec, digits=2), "%")
    end
    if fhertz_monitor_enabled
      nn_metrics = val_nn_metrics[]
      tprintln("  nn: F_hertz err=", fmt_f(nn_metrics.fhertz_err, digits=2), "%")
      tprintln("  nn gain: g_nn=", fmt_e(g_nn, sigdigits=3),
        " target=", fmt_e(gnn_info.target, sigdigits=3),
        " amp_ref=", fmt_e(gnn_info.amp_ref, sigdigits=3),
        " eff_n=", gnn_info.eff_count,
        " mode=", gnn_info.target_mode)
      tprintln("  nn target est: |m*x2dot|=", fmt_e(gnn_info.p95_mx2dot, sigdigits=3),
        " |k*x1|=", fmt_e(gnn_info.p95_kx1, sigdigits=3),
        " |c*x2|=", fmt_e(gnn_info.p95_cx2, sigdigits=3),
        " |Fd|=", fmt_e(gnn_info.p95_fd, sigdigits=3),
        " |Fad_eff|=", fmt_e(gnn_info.p95_fadh, sigdigits=3))
      tprintln("  nn raw: min=", fmt_e(nn_metrics.raw_min, sigdigits=3),
        " max=", fmt_e(nn_metrics.raw_max, sigdigits=3),
        " mean=", fmt_e(nn_metrics.raw_mean, sigdigits=3),
        " neg=", fmt_f(100 * nn_metrics.raw_neg_frac, digits=2), "%")
      tprintln("  nn force: min=", fmt_e(nn_metrics.fh_min, sigdigits=3),
        " max=", fmt_e(nn_metrics.fh_max, sigdigits=3),
        " mean=", fmt_e(nn_metrics.fh_mean, sigdigits=3),
        " p95|F|=", fmt_e(nn_metrics.fh_abs_p95, sigdigits=3))
    end
  end

  viable_trials = [rec for rec in trial_parameters if hasproperty(rec, :is_viable) && rec.is_viable]
  selection_pool = isempty(viable_trials) ? trial_parameters : viable_trials
  sorted = sort(selection_pool, by = r -> r.loss)
  selected = sorted[1:min(final_topk, length(sorted))]
  warm_start_top = [(
      trial_id=rec.params["trial_id"],
      params=rec.params,
      p_net=copy(rec.p_net_vec),
      train_loss=rec.train_loss,
      val_loss=rec.val_loss,
      g_nn=(haskey(rec.params, "g_nn") ? rec.params["g_nn"] : 1.0)
    ) for rec in selected]
  local_warm_has_pnet = !isempty(warm_start_top) && hasproperty(warm_start_top[1], :p_net)

  tprintln("Stage1plus done. Viable=", length(viable_trials), "/", length(trial_parameters),
    " | selected=", length(selected))
  if !isempty(selected)
    tprintln("Stage1plus best val loss=", fmt_e(selected[1].val_loss, sigdigits=4),
      " | train=", fmt_e(selected[1].train_loss, sigdigits=4))
  end
  tprintln("Stage1plus top-", length(selected), " summary (finite/stable filtered):")
  for (i, rec) in enumerate(selected)
    tprintln("  Rank ", i,
      " -- train=", fmt_e(rec.train_loss, sigdigits=4),
      " val=", fmt_e(rec.val_loss, sigdigits=4),
      " | base=", rec.params["base_rank"],
      " | ks0=", fmt_e(rec.params["ks0"], sigdigits=3),
      " cs0=", fmt_e(rec.params["cs0"], sigdigits=3))
    tprintln("     val parts: state=", fmt_e(rec.val_parts.state, sigdigits=3),
      " x2dot=", fmt_e(rec.val_parts.x2dot, sigdigits=3),
      " x3r=", fmt_e(rec.val_parts.x3_range, sigdigits=3),
      " cont=", fmt_e(rec.val_parts.cont, sigdigits=3))
    tprintln("     rec: x1=", fmt_f(rec.val_parts.x1_rec, digits=2),
      "% x3=", fmt_f(rec.val_parts.x3_rec, digits=2), "%")
    tprintln("     mech: ks=", fmt_e(rec.ks_hat, sigdigits=3),
      " (err=", fmt_f(rec.ks_err_pct, digits=2), "%)",
      " cs=", fmt_e(rec.cs_hat, sigdigits=3),
      " (err=", fmt_f(rec.cs_err_pct, digits=2), "%)")
    tprintln("     nn: F_hertz err=", fmt_f(rec.val_nn_err, digits=2), "%")
    tprintln("     nn gain: g_nn=", fmt_e(hasproperty(rec, :g_nn) ? rec.g_nn :
      (haskey(rec.params, "g_nn") ? rec.params["g_nn"] : NaN), sigdigits=3),
      " target=", fmt_e(hasproperty(rec, :g_target) ? rec.g_target :
      (haskey(rec.params, "g_target") ? rec.params["g_target"] : NaN), sigdigits=3),
      " amp_ref=", fmt_e(hasproperty(rec, :g_amp_ref) ? rec.g_amp_ref :
      (haskey(rec.params, "g_amp_ref") ? rec.params["g_amp_ref"] : NaN), sigdigits=3))
    tprintln("     nn raw: min=", fmt_e(rec.val_raw_min, sigdigits=3),
      " max=", fmt_e(rec.val_raw_max, sigdigits=3),
      " mean=", fmt_e(rec.val_raw_mean, sigdigits=3),
      " neg=", fmt_f(100 * rec.val_raw_neg_frac, digits=2), "%")
    tprintln("     nn force: min=", fmt_e(hasproperty(rec, :val_fh_min) ? rec.val_fh_min : NaN, sigdigits=3),
      " max=", fmt_e(hasproperty(rec, :val_fh_max) ? rec.val_fh_max : NaN, sigdigits=3),
      " mean=", fmt_e(hasproperty(rec, :val_fh_mean) ? rec.val_fh_mean : NaN, sigdigits=3),
      " p95|F|=", fmt_e(hasproperty(rec, :val_fh_abs_p95) ? rec.val_fh_abs_p95 : NaN, sigdigits=3))
  end
  tprintln("Stage1plus warm-start export (shard): ON | saved=", length(warm_start_top),
    " | first_has_p_net=", local_warm_has_pnet)

  serialize(result_folder * "/" * result_name_string, (
    study=nothing,
    trial_parameters=trial_parameters,
    warm_start_top=warm_start_top,
    selected=selected,
    best=isempty(selected) ? nothing : selected[1],
    bounds=(ks=ks_bounds, cs=cs_bounds),
    true_values=(ks=ks_true, cs=cs_true),
    use_multiple_shooting=use_multiple_shooting,
    use_l2_regularization=use_l2_regularization,
    val_stride=val_stride,
    val_offset=val_offset,
    x3_obs_fraction=0.0,
    error_level=error_level,
    stage1_input_file=stage1_input_file,
    stage1_input_topk=stage1plus_input_topk,
    searches_per_candidate=searches_per_candidate,
    final_topk=final_topk,
    gnn_target_mode=stage1pluslight_gnn_target_mode,
    gnn_target_manual=stage1pluslight_gnn_target_manual,
    gnn_target_quantile=stage1pluslight_gnn_target_quantile
  ))
  end
  end
else
  let
  # ADAM off: Stage1-A random search (mechanistic-only; NN frozen to zero output)
  function rand_loguniform(rng, lo, hi)
    return exp(rand(rng) * (log(hi) - log(lo)) + log(lo))
  end

  # Defaults live in code; env can override if needed
  num_trials = parse(Int, get(ENV, "HNODECB_STAGE1_NOADAM_TRIALS", "3000"))
  top_k = parse(Int, get(ENV, "HNODECB_STAGE1_NOADAM_TOPK", "10"))
  refine_trials = parse(Int, get(ENV, "HNODECB_STAGE1_NOADAM_REFINE", "200"))
  shard_idx = parse(Int, get(ENV, "HNODECB_STAGE1_SHARD_INDEX", "1"))
  shard_cnt = parse(Int, get(ENV, "HNODECB_STAGE1_SHARD_COUNT", "1"))
  if shard_idx < 1 || shard_idx > shard_cnt
    error("HNODECB_STAGE1_SHARD_INDEX must be in 1..HNODECB_STAGE1_SHARD_COUNT")
  end
  # Split a fixed global trial budget across shards (e.g. 3000 -> 4x750).
  trial_indices = [i for i in 1:num_trials if ((i - shard_idx) % shard_cnt) == 0]

  trial_parameters = []
  top_candidates = Vector{Any}()

  fixed_num_hidden_layers = nn_fixed_num_hidden_layers
  fixed_num_hidden_nodes = nn_fixed_num_hidden_nodes
  fixed_ms_group_size = 10
  fixed_ms_continuity_term = 1e-3
  fhertz_monitor_enabled = false

  fixed_nn_rng = StableRNG(0)
  fixed_approximating_neural_network = build_nn(fixed_num_hidden_layers, fixed_num_hidden_nodes)
  fixed_p_net, fixed_st = Lux.setup(fixed_nn_rng, fixed_approximating_neural_network)
  fixed_p_net = Flux.f64(fixed_p_net)
  fixed_p_net_vec, fixed_re_pnet = Optimisers.destructure(fixed_p_net)
  fixed_p_net_vec .= 0.0

  train_idx, val_idx = make_train_val_masks(length(all_times), val_stride, val_offset)
  ode_train = ode_data_full[:, train_idx]
  ode_val = ode_data_full[:, val_idx]
  times_train = all_times[train_idx]
  times_val = all_times[val_idx]
  x2dot_train = x2dot_all[train_idx]
  x2dot_val = x2dot_all[val_idx]
  contact_train = contact_all[train_idx]
  contact_val = contact_all[val_idx]
  contact_idx_train = findall(contact_train)
  contact_idx_val = findall(contact_val)

  state12_scale_train = state12_scale_full
  x2dot_scale_train = x2dot_scale_full
  x3_t0_val = ode_train[3, 1]

  tprintln("Stage1 no-ADAM shard ", shard_idx, "/", shard_cnt,
    " local_trials=", length(trial_indices), " total_trials=", num_trials)
  tprintln("Stage1 no-ADAM mode: frozen zero NN (layers=", fixed_num_hidden_layers,
    ", nodes=", fixed_num_hidden_nodes, ") | NN monitor=OFF | zero-Hertz override=ON")
  tprintln("Stage1 warm-start capture (shard): ENABLED | local_top_k=", top_k)
  for i in trial_indices
    tprintln("Stage1 trial ", i, " start (no-ADAM)")
    flush(stdout)
    # Per-trial RNG keeps mechanistic sampling reproducible and avoids cross-shard duplicates.
    rng_trial = StableRNG(i)
    ks0 = rand_loguniform(rng_trial, ks_bounds[1], ks_bounds[2])
    cs0 = rand_loguniform(rng_trial, cs_bounds[1], cs_bounds[2])
    raw_init = [
      raw_from_value(ks0, ks_bounds[1], ks_bounds[2]),
      raw_from_value(cs0, cs_bounds[1], cs_bounds[2])
    ]
    p_net_vec = copy(fixed_p_net_vec)
    re_pnet = fixed_re_pnet
    st = fixed_st
    approximating_neural_network = fixed_approximating_neural_network
    θ0 = ComponentVector(p_net=p_net_vec, mech_raw=raw_init)

    inf_context[] = "train_eval"
    last_inf_reason[] = ""
    train_loss, diag = loss_single_or_ms(θ0, ode_train, x2dot_train, contact_train, times_train,
      state12_scale_train, x2dot_scale_train, x3_scale,
      use_multiple_shooting, fixed_ms_group_size, fixed_ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=true)
    train_reason = last_inf_reason[]

    inf_context[] = "val_eval"
    last_inf_reason[] = ""
    val_loss, val_diag = loss_single_or_ms(θ0, ode_val, x2dot_val, contact_val, times_val,
      state12_scale_full, x2dot_scale_full, x3_scale,
      use_multiple_shooting, fixed_ms_group_size, fixed_ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=true)
    val_reason = last_inf_reason[]
    ks_hat = bound_param(θ0.mech_raw[1], ks_bounds[1], ks_bounds[2])
    cs_hat = bound_param(θ0.mech_raw[2], cs_bounds[1], cs_bounds[2])
    ks_err_pct = rel_err_pct(ks_hat, ks_true, scale_eps)
    cs_err_pct = rel_err_pct(cs_hat, cs_true, scale_eps)

    params = Dict{Any, Any}(
      "trial_id" => i,
      "ks0" => ks0, "cs0" => cs0,
      "num_hidden_layers" => fixed_num_hidden_layers,
      "num_hidden_nodes" => fixed_num_hidden_nodes
    )
    push!(trial_parameters, (
      loss=val_loss,
      train_loss=train_loss,
      val_loss=val_loss,
      params=params,
      val_parts=val_diag,
      ks_hat=ks_hat,
      cs_hat=cs_hat,
      ks_err_pct=ks_err_pct,
      cs_err_pct=cs_err_pct
    ))

    tprintln("Stage1 trial ", i,
      " -- train=", fmt_e(train_loss, sigdigits=4),
      " val=", fmt_e(val_loss, sigdigits=4))
    if !isfinite(train_loss) || !isfinite(val_loss)
      train_reason = train_reason == "" ? "unknown" : train_reason
      val_reason = val_reason == "" ? "unknown" : val_reason
      tprintln("  Inf reason: train=", train_reason, " | val=", val_reason)
    end
    if diag !== nothing
      tprintln("  parts: state=", fmt_e(diag.state, sigdigits=3),
        " x2dot=", fmt_e(diag.x2dot, sigdigits=3),
        " x3r=", fmt_e(diag.x3_range, sigdigits=3),
        " cont=", fmt_e(diag.cont, sigdigits=3))
    end
    tprintln("  mech: ks=", fmt_e(ks_hat, sigdigits=3),
      " (err=", fmt_f(ks_err_pct, digits=2), "%)",
      " cs=", fmt_e(cs_hat, sigdigits=3),
      " (err=", fmt_f(cs_err_pct, digits=2), "%)")
    if diag !== nothing
      tprintln("  rec: x1=", fmt_f(diag.x1_rec, digits=2), "% x3=",
        fmt_f(diag.x3_rec, digits=2), "%")
    end
    if fhertz_monitor_enabled
      Zygote.ignore() do
        p_net_struct = re_pnet(θ0.p_net)
        nn_metrics = fhertz_monitor_metrics(
          ode_data_full, monitor_idx_full, p_net_struct,
          approximating_neural_network, st;
          fh_true_ref=monitor_fhertz_true, eff_pos_ref=monitor_eff_positions
        )
        tprintln("  nn: F_hertz err=", fmt_f(nn_metrics.fhertz_err, digits=2), "%")
        tprintln("  nn raw: min=", fmt_e(nn_metrics.raw_min, sigdigits=3),
          " max=", fmt_e(nn_metrics.raw_max, sigdigits=3),
          " mean=", fmt_e(nn_metrics.raw_mean, sigdigits=3),
          " neg=", fmt_f(100 * nn_metrics.raw_neg_frac, digits=2), "%")
      end
    end
    # maintain top-k for refinement
    push!(top_candidates, (
      val_loss=val_loss,
      train_loss=train_loss,
      θ=θ0,
      re_pnet=re_pnet,
      st=st,
      appr=approximating_neural_network,
      ks0=ks0,
      cs0=cs0,
      trial_id=i,
      params=params,
      p_net_vec=copy(p_net_vec),
      num_hidden_layers=fixed_num_hidden_layers,
      num_hidden_nodes=fixed_num_hidden_nodes
    ))
    top_candidates = sort(top_candidates, by = c -> c.val_loss)
    if length(top_candidates) > top_k
      pop!(top_candidates)
    end
  end

  refined = []
  if use_multiple_shooting
    # Stage1-B: for each top candidate, randomize hyperparams (no ADAM)
    tprintln("Stage1-B refine (no-ADAM): ", top_k, " candidates x ", refine_trials, " trials")
    for (ci, cand) in enumerate(top_candidates)
      for r in 1:refine_trials
        ms_group_size = rand(rng_global, 10:200)
        ms_continuity_term = rand_loguniform(rng_global, 1e-6, 10.0)

        inf_context[] = "val_eval"
        last_inf_reason[] = ""
        val_loss, _ = loss_single_or_ms(cand.θ, ode_val, x2dot_val, contact_val, times_val,
          state12_scale_full, x2dot_scale_full, x3_scale,
          use_multiple_shooting, ms_group_size, ms_continuity_term,
          cand.appr, cand.st, known_pars, x3_t0_val,
          0.0, cand.re_pnet; zero_nn_override=true)
        val_reason = last_inf_reason[]

        params = Dict{Any, Any}(
          "stage" => "B",
          "cand_rank" => ci,
          "ks0" => cand.ks0,
          "cs0" => cand.cs0,
          "num_hidden_layers" => cand.num_hidden_layers,
          "num_hidden_nodes" => cand.num_hidden_nodes,
          "ms_group_size" => ms_group_size,
          "ms_continuity_term" => ms_continuity_term
        )
        push!(refined, (loss=val_loss, train_loss=cand.train_loss, val_loss=val_loss, params=params))
        if !isfinite(val_loss)
          val_reason = val_reason == "" ? "unknown" : val_reason
          tprintln("  Inf reason: val=", val_reason)
        end
      end
    end
  else
    tprintln("Stage1-B skipped (MS disabled)")
  end

  # combine and pick best from refined set (if any)
  all_trials = isempty(refined) ? trial_parameters : vcat(trial_parameters, refined)
  sorted = sort(all_trials, by = r -> r.loss)
  best = sorted[1]
  warm_start_top = [(
      trial_id=c.trial_id,
      params=c.params,
      p_net=copy(c.p_net_vec),
      train_loss=c.train_loss,
      val_loss=c.val_loss
    ) for c in sort(top_candidates, by = c -> c.val_loss)]
  local_warm_has_pnet = !isempty(warm_start_top) && hasproperty(warm_start_top[1], :p_net)
  tprintln("Stage1 done. Best val loss=", fmt_e(best.val_loss, sigdigits=4),
    " | train=", fmt_e(best.train_loss, sigdigits=4))
  tprintln("Stage1 warm-start export (shard): ON | saved=", length(warm_start_top),
    " | first_has_p_net=", local_warm_has_pnet)

  tprintln("Stage1 top-10 summary (train/val):")
  for (i, rec) in enumerate(sorted[1:min(10, length(sorted))])
    ks0 = rec.params["ks0"]
    cs0 = rec.params["cs0"]
    ks_hat = hasproperty(rec, :ks_hat) ? rec.ks_hat : ks0
    cs_hat = hasproperty(rec, :cs_hat) ? rec.cs_hat : cs0
    ks_err_pct = hasproperty(rec, :ks_err_pct) ? rec.ks_err_pct : NaN
    cs_err_pct = hasproperty(rec, :cs_err_pct) ? rec.cs_err_pct : NaN
    tprintln("  Rank ", i,
      " -- train=", fmt_e(rec.train_loss, sigdigits=4),
      " val=", fmt_e(rec.val_loss, sigdigits=4),
      " | ks0=", fmt_e(ks0, sigdigits=3),
      " cs0=", fmt_e(cs0, sigdigits=3))
    tprintln("     val parts: state=", fmt_e(rec.val_parts.state, sigdigits=3),
      " x2dot=", fmt_e(rec.val_parts.x2dot, sigdigits=3),
      " x3r=", fmt_e(rec.val_parts.x3_range, sigdigits=3),
      " cont=", fmt_e(rec.val_parts.cont, sigdigits=3))
    tprintln("     rec: x1=", fmt_f(rec.val_parts.x1_rec, digits=2),
      "% x3=", fmt_f(rec.val_parts.x3_rec, digits=2), "%")
    tprintln("     mech: ks=", fmt_e(ks_hat, sigdigits=3),
      " (err=", fmt_f(ks_err_pct, digits=2), "%)",
      " cs=", fmt_e(cs_hat, sigdigits=3),
      " (err=", fmt_f(cs_err_pct, digits=2), "%)")
  end

  serialize(result_folder * "/" * result_name_string, (
    study=nothing,
    trial_parameters=all_trials,
    warm_start_top=warm_start_top,
    best=best,
    bounds=(ks=ks_bounds, cs=cs_bounds),
    true_values=(ks=ks_true, cs=cs_true),
    use_multiple_shooting=use_multiple_shooting,
    use_l2_regularization=use_l2_regularization,
    val_stride=val_stride,
    val_offset=val_offset,
    x3_obs_fraction=0.0,
    error_level=error_level
  ))
  end
end
end
