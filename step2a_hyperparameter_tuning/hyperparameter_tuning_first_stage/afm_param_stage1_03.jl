#=
Stage 1 (global TPE) hyperparameter tuning for AFM DMT-KV.
Scenario 03: sparse x3 observations enabled by default, net contact-force term replaced by a neural network.
Single shooting or multiple shooting (toggle), x3 observations + range prior.
TPE explores NN architecture, mechanistic initial values, MS hyperparams, learning rate, and train/val split.
=#

cd(@__DIR__)

using Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Statistics, Dates
using ComponentArrays, SciMLSensitivity, SciMLBase, StableRNGs
using Zygote
using Optimization, OptimizationOptimisers, Optimisers
using DiffEqFlux, Flux

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

const DEFAULT_STAGE1PLUSLIGHT_RUN_SEED = 314159265
const STAGE1_OPTUNA_REF = Ref{Any}(nothing)

function stage1_get_optuna()
  if STAGE1_OPTUNA_REF[] === nothing
    pycall_mod = Base.require(Main, :PyCall)
    STAGE1_OPTUNA_REF[] = pycall_mod.pyimport("optuna")
  end
  return STAGE1_OPTUNA_REF[]
end

function resolve_stage1_run_seed()
  seed_str = strip(get(ENV, "HNODECB_STAGE1_RUN_SEED", ""))
  if !isempty(seed_str)
    seed = try
      parse(Int, seed_str)
    catch
      error("HNODECB_STAGE1_RUN_SEED must parse as Int, got: " * seed_str)
    end
    seed == 0 && (seed = 1)
    ENV["HNODECB_STAGE1_RUN_SEED"] = string(seed)
    return seed
  end
  stage1_variant_env = get(ENV, "HNODECB_STAGE1_VARIANT", "stage1")
  use_stage1pluslight_seed =
    stage1_variant_env == "stage1pluslight" || get(ENV, "HNODECB_STAGE1PLUSLIGHT_MODE", "0") == "1"
  seed = use_stage1pluslight_seed ? DEFAULT_STAGE1PLUSLIGHT_RUN_SEED : abs(rand(RandomDevice(), Int))
  seed == 0 && (seed = 1)
  ENV["HNODECB_STAGE1_RUN_SEED"] = string(seed)
  return seed
end

mix_stage1_trial_seed(run_seed::Int, trial_id::Int) = begin
  seed = mod(run_seed + 1_000_003 * trial_id, typemax(Int))
  seed = abs(seed)
  seed == 0 && (seed = 1)
  seed
end

const stage1_run_seed = resolve_stage1_run_seed()

# global RNG for reproducible per-run randomness
rng_global = Random.default_rng()
Random.seed!(rng_global, stage1_run_seed)

stage1plus_mix64(x::UInt64) = begin
  z = x + 0x9e3779b97f4a7c15
  z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
  z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
  z ⊻ (z >> 31)
end

stage1plus_hash_u01(run_seed::Int, item_id::Int, stream_id::Int) = begin
  base = UInt64(abs(run_seed % typemax(Int)))
  item = UInt64(abs(item_id % typemax(Int)))
  stream = UInt64(abs(stream_id % typemax(Int)))
  key = base ⊻ (item * 0x9e3779b97f4a7c15) ⊻ (stream * 0xbf58476d1ce4e5b9)
  mixed = stage1plus_mix64(key)
  (Float64(mixed >> 11) + 0.5) * 0x1.0p-53
end

stage1plus_hash_loguniform(run_seed::Int, item_id::Int, stream_id::Int, lo::Float64, hi::Float64) =
  exp(stage1plus_hash_u01(run_seed, item_id, stream_id) * (log(hi) - log(lo)) + log(lo))

stage1plus_hash_seed(run_seed::Int, item_id::Int, stream_id::Int) = begin
  base = UInt64(abs(run_seed % typemax(Int)))
  item = UInt64(abs(item_id % typemax(Int)))
  stream = UInt64(abs(stream_id % typemax(Int)))
  mixed = stage1plus_mix64(base ⊻ (item * 0x94d049bb133111eb) ⊻ (stream * 0x369dea0f31a53f85))
  seed = Int(rem(mixed, UInt64(typemax(Int))))
  seed == 0 && (seed = 1)
  seed
end

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
x3_range_amp = 100e-9
x3_range_eps = 1e-9
x3_range_weight = 1.0
fts_range_amp = 1e-8
fts_range_eps = 1e-10
fts_range_weight = let
  v = get(ENV, "HNODECB_STAGE1_FTS_RANGE_WEIGHT", get(ENV, "HNODECB_FTS_RANGE_WEIGHT", "1.0"))
  p = tryparse(Float64, v)
  p === nothing ? 1.0 : p
end

# Contact weighting
contact_loss_weight = 4.27
noncontact_loss_weight = 1.0

# Solver settings
integrator = Rosenbrock23(autodiff=false)
abstol = 1e-8
reltol = 1e-8
sensealg = GaussAdjoint(autojacvec=ZygoteVJP())
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
fts_scale = max(fts_range_amp, scale_eps)

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
  flush(stdout)
end

function fmt_e(x; sigdigits=4)
  y = x
  for _ in 1:8
    if y isa Number && hasproperty(y, :value)
      y_next = getproperty(y, :value)
      y_next === y && break
      y = y_next
    else
      break
    end
  end
  if y isa Number
    try
      return isfinite(y) ? string(round(y, sigdigits=sigdigits)) : string(y)
    catch
      return string(y)
    end
  end
  return string(y)
end

function fmt_f(x; digits=2)
  y = x
  for _ in 1:8
    if y isa Number && hasproperty(y, :value)
      y_next = getproperty(y, :value)
      y_next === y && break
      y = y_next
    else
      break
    end
  end
  if y isa Number
    try
      return isfinite(y) ? string(round(y, digits=digits)) : string(y)
    catch
      return string(y)
    end
  end
  return string(y)
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

function env_int(key, default::Int)
  v = get(ENV, key, "")
  if v == ""
    return default
  end
  try
    return parse(Int, v)
  catch
    return default
  end
end

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

# "God-view" net contact force (truth) for monitoring only (never used in training).
function contact_net_true_from_states(u_mat, idxs)
  out = Vector{Float64}(undef, length(idxs))
  @inbounds for (k, j) in enumerate(idxs)
    s = dist + u_mat[1, j] - u_mat[3, j]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    w_true = contact_weight(s, adhesion_transition)
    f_hertz = (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
    out[k] = f_hertz - Fad * w_true
  end
  return out
end

# NN-predicted net contact force for monitoring only (never used in training).
function contact_net_pred_from_states(u_mat, idxs, p_net_struct, appr, st; nn_gain::Float64=1.0)
  out = Vector{Float64}(undef, length(idxs))
  if appr === nothing
    @inbounds for (k, j) in enumerate(idxs)
      s = dist + u_mat[1, j] - u_mat[3, j]
      delta = softplus(-s, adhesion_transition)
      delta = ifelse(delta > 0.0, delta, 0.0)
      w_pred = contact_weight(s, adhesion_transition)
      out[k] = (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad * w_pred
    end
    return out
  end
  @inbounds for (k, j) in enumerate(idxs)
    s = dist + u_mat[1, j] - u_mat[3, j]
    w_pred = contact_weight(s, adhesion_transition)
    nn_in = nn_input_from_values(u_mat[1, j], u_mat[2, j], u_mat[3, j])
    uhat = appr(nn_in, p_net_struct, st)[1]
    out[k] = nn_gain * uhat[1] * w_pred
  end
  return out
end

function contact_net_pred_and_raw_from_states(u_mat, idxs, p_net_struct, appr, st; nn_gain::Float64=1.0)
  out = Vector{Float64}(undef, length(idxs))
  raw = Vector{Float64}(undef, length(idxs))
  if appr === nothing
    @inbounds for (k, j) in enumerate(idxs)
      s = dist + u_mat[1, j] - u_mat[3, j]
      delta = softplus(-s, adhesion_transition)
      delta = ifelse(delta > 0.0, delta, 0.0)
      w_pred = contact_weight(s, adhesion_transition)
      raw[k] = NaN
      out[k] = (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad * w_pred
    end
    return out, raw
  end
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

function effective_contact_positions(idxs, contact_true)
  contact_pos = [k for (k, j) in enumerate(idxs) if contact_all[j]]
  if isempty(contact_pos)
    return Int[]
  end
  max_true = maximum(abs.(contact_true[contact_pos]))
  if !isfinite(max_true) || max_true <= 0.0
    return contact_pos
  end
  floor_val = nn_monitor_effective_contact_frac * max_true
  eff_pos = [k for k in contact_pos if abs(contact_true[k]) > floor_val]
  return isempty(eff_pos) ? contact_pos : eff_pos
end

function effective_contact_positions_from_truth(contact_true)
  if isempty(contact_true)
    return Int[]
  end
  abs_true = abs.(contact_true)
  max_true = maximum(abs_true)
  if !isfinite(max_true) || max_true <= 0.0
    return collect(eachindex(contact_true))
  end
  floor_val = nn_monitor_effective_contact_frac * max_true
  eff_pos = findall(v -> isfinite(v) && v > floor_val, abs_true)
  return isempty(eff_pos) ? collect(eachindex(contact_true)) : eff_pos
end

rounded_log10_decade(v::Real) = (isfinite(v) && v > 0.0) ? round(Int, log10(v)) : nothing

function decade_nn_gain_from_states(u_mat, idxs, p_net_struct, appr, st;
  contact_true_ref=nothing, eff_pos_ref=nothing,
  q::Float64=0.95, gain_min::Float64=1e-12, gain_max::Float64=1e12,
  current_gain=nothing, log10_update_min::Float64=1.0)
  fallback_gain = (current_gain isa Real && isfinite(current_gain) && current_gain > 0.0) ? Float64(current_gain) : 1.0
  if appr === nothing
    return (
      gain=fallback_gain,
      candidate_gain=fallback_gain,
      true_amp=NaN,
      pred_amp=NaN,
      true_decade=nothing,
      pred_decade=nothing,
      decade_delta=nothing,
      log10_diff=NaN,
      updated=false,
      reason="no_nn"
    )
  end
  contact_true = contact_true_ref === nothing ? contact_net_true_from_states(u_mat, idxs) : contact_true_ref
  contact_pred = contact_net_pred_from_states(u_mat, idxs, p_net_struct, appr, st; nn_gain=1.0)
  eff_pos = eff_pos_ref === nothing ? effective_contact_positions_from_truth(contact_true) : eff_pos_ref
  if isempty(eff_pos)
    return (
      gain=fallback_gain,
      candidate_gain=fallback_gain,
      true_amp=NaN,
      pred_amp=NaN,
      true_decade=nothing,
      pred_decade=nothing,
      decade_delta=nothing,
      log10_diff=NaN,
      updated=false,
      reason="empty_eff_pos"
    )
  end
  true_abs = abs.(contact_true[eff_pos])
  pred_abs = abs.(contact_pred[eff_pos])
  true_amp = quantile(true_abs, q)
  pred_amp = quantile(pred_abs, q)
  if !(isfinite(true_amp) && true_amp > 0.0 && isfinite(pred_amp) && pred_amp > 0.0)
    return (
      gain=fallback_gain,
      candidate_gain=fallback_gain,
      true_amp=true_amp,
      pred_amp=pred_amp,
      true_decade=rounded_log10_decade(true_amp),
      pred_decade=rounded_log10_decade(pred_amp),
      decade_delta=nothing,
      log10_diff=NaN,
      updated=false,
      reason="nonfinite_amp"
    )
  end
  true_decade = rounded_log10_decade(true_amp)
  pred_decade = rounded_log10_decade(pred_amp)
  if true_decade === nothing || pred_decade === nothing
    return (
      gain=fallback_gain,
      candidate_gain=fallback_gain,
      true_amp=true_amp,
      pred_amp=pred_amp,
      true_decade=true_decade,
      pred_decade=pred_decade,
      decade_delta=nothing,
      log10_diff=NaN,
      updated=false,
      reason="invalid_decade"
    )
  end
  decade_delta = true_decade - pred_decade
  log10_diff = log10(true_amp) - log10(pred_amp)
  candidate_gain = clamp(10.0 ^ decade_delta, gain_min, gain_max)
  if !(isfinite(candidate_gain) && candidate_gain > 0.0)
    return (
      gain=fallback_gain,
      candidate_gain=fallback_gain,
      true_amp=true_amp,
      pred_amp=pred_amp,
      true_decade=true_decade,
      pred_decade=pred_decade,
      decade_delta=decade_delta,
      log10_diff=log10_diff,
      updated=false,
      reason="invalid_candidate"
    )
  end
  gain = candidate_gain
  updated = true
  reason = current_gain === nothing ? "init_quantized" : "updated_quantized"
  if current_gain isa Real && isfinite(current_gain) && current_gain > 0.0 && abs(log10_diff) < log10_update_min
    gain = Float64(current_gain)
    updated = false
    reason = "kept_prev_within_log10_band"
  end
  return (
    gain=gain,
    candidate_gain=candidate_gain,
    true_amp=true_amp,
    pred_amp=pred_amp,
    true_decade=true_decade,
    pred_decade=pred_decade,
    decade_delta=decade_delta,
    log10_diff=log10_diff,
    updated=updated,
    reason=reason
  )
end

function contact_net_monitor_metrics(u_mat, idxs, p_net_struct, appr, st; contact_true_ref=nothing, eff_pos_ref=nothing, nn_gain::Float64=1.0)
  contact_true = contact_true_ref === nothing ? contact_net_true_from_states(u_mat, idxs) : contact_true_ref
  contact_pred, raw = contact_net_pred_and_raw_from_states(u_mat, idxs, p_net_struct, appr, st; nn_gain=nn_gain)
  eff_pos = eff_pos_ref === nothing ? effective_contact_positions(idxs, contact_true) : eff_pos_ref
  if isempty(eff_pos)
    return (
      fcontact_err=NaN,
      raw_min=NaN,
      raw_max=NaN,
      raw_mean=NaN,
      raw_neg_frac=NaN,
      fnet_min=NaN,
      fnet_max=NaN,
      fnet_mean=NaN,
      fnet_abs_p95=NaN
    )
  end
  raw_eff = raw[eff_pos]
  fnet_eff = contact_pred[eff_pos]
  fnet_abs = abs.(fnet_eff)
  err_sum = sum(abs2, contact_pred[eff_pos] .- contact_true[eff_pos])
  truth_sum = sum(abs2, contact_true[eff_pos])
  fcontact_err = truth_sum <= 0.0 ? NaN : relative_rmse_pct(err_sum, truth_sum, length(eff_pos), 0.0)
  raw_neg_frac = count(<(0.0), raw_eff) / length(raw_eff)
  return (
    fcontact_err=fcontact_err,
    raw_min=minimum(raw_eff),
    raw_max=maximum(raw_eff),
    raw_mean=mean(raw_eff),
    raw_neg_frac=raw_neg_frac,
    fnet_min=minimum(fnet_eff),
    fnet_max=maximum(fnet_eff),
    fnet_mean=mean(fnet_eff),
    fnet_abs_p95=quantile(fnet_abs, 0.95)
  )
end

monitor_contact_true = contact_net_true_from_states(ode_data_full, monitor_idx_full)
monitor_eff_positions = effective_contact_positions(monitor_idx_full, monitor_contact_true)

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
trace_epoch_ref = Ref(0)
trace_first_epoch_pending_ref = Ref(true)
trace_epoch_latched_ref = Ref(false)
trace_scope_enabled() = begin
  shard_ok = get(ENV, "HNODECB_STAGE1_SHARD_INDEX", "1") == "1"
  arch_ok = get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_INDEX", "1") == "1"
  shard_ok && arch_ok
end
function trace_begin_epoch(epoch::Int)
  trace_epoch_ref[] = epoch
  trace_epoch_latched_ref[] = epoch == 1 && trace_scope_enabled() && trace_first_epoch_pending_ref[]
end
function trace_end_epoch()
  if trace_epoch_latched_ref[]
    trace_first_epoch_pending_ref[] = false
  end
  trace_epoch_latched_ref[] = false
  trace_epoch_ref[] = 0
end
trace_epoch_active() = trace_epoch_ref[] == 1 && trace_epoch_latched_ref[]
function trace_solve(msg)
  Zygote.ignore() do
    if !trace_solve_enabled || !trace_epoch_active()
      return
    end
    if trace_solve_count[] < trace_solve_limit
      trace_solve_count[] += 1
      tprintln("TRACE_SOLVE: ", msg isa Function ? msg() : msg)
    end
  end
end

trace_loss_enabled = get(ENV, "HNODECB_TRACE_LOSS", "1") == "1"
trace_loss_limit = parse(Int, get(ENV, "HNODECB_TRACE_LOSS_LIMIT", "200"))
trace_loss_count = Ref(0)
function trace_loss(msg)
  Zygote.ignore() do
    if !trace_loss_enabled || !trace_epoch_active()
      return
    end
    if trace_loss_count[] < trace_loss_limit
      trace_loss_count[] += 1
      tprintln("TRACE_LOSS: ", msg isa Function ? msg() : msg)
    end
  end
end

trace_point_enabled = get(ENV, "HNODECB_TRACE_POINT", "1") == "1"
trace_point_every = max(1, parse(Int, get(ENV, "HNODECB_TRACE_POINT_EVERY", "25")))
trace_point_limit = parse(Int, get(ENV, "HNODECB_TRACE_POINT_LIMIT", "400"))
trace_point_count = Ref(0)
function trace_point(msg)
  Zygote.ignore() do
    if !trace_point_enabled || !trace_epoch_active()
      return
    end
    if trace_point_count[] < trace_point_limit
      trace_point_count[] += 1
      tprintln("TRACE_POINT: ", msg isa Function ? msg() : msg)
    end
  end
end

trace_rhs_enabled = get(ENV, "HNODECB_TRACE_RHS", "0") == "1"
trace_rhs_every = max(1, parse(Int, get(ENV, "HNODECB_TRACE_RHS_EVERY", "5000")))
trace_rhs_limit = parse(Int, get(ENV, "HNODECB_TRACE_RHS_LIMIT", "100"))
trace_rhs_count = Ref(0)
function trace_rhs(msg)
  Zygote.ignore() do
    if !trace_rhs_enabled || !trace_epoch_active()
      return
    end
    if trace_rhs_count[] < trace_rhs_limit
      trace_rhs_count[] += 1
      tprintln("TRACE_RHS: ", msg isa Function ? msg() : msg)
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
  rc = idx isa CartesianIndex ? Tuple(idx) : Tuple(CartesianIndices(A)[idx])
  return "row=" * string(rc[1]) * " col=" * string(rc[2]) * " val=" * string(A[idx])
end
function first_nonfinite_vector(v)
  idx = findfirst(x -> !isfinite(x), v)
  if idx === nothing
    return "none"
  end
  return "idx=" * string(idx) * " val=" * string(v[idx])
end
function log_failure_diag(tag, sol, p, appr, st, known_pars, expected_n; zero_nn_override::Bool=false)
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
    Fad_eff = Fad * w_pred
    F_contact = if zero_nn_override
      0.0
    elseif appr === nothing
      (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad_eff
    else
      nn_out * w_pred
    end
    rhs_x2dot = (Fd * cos(wd * t_last) - k * u_last[1] - c * u_last[2] + F_contact) / m
    rhs_x3dot = (-F_contact - ks * u_last[3]) / cs
    tprintln("FAIL_DIAG[", tag, "] phys: s=", fmt_e(s, sigdigits=4),
      " delta=", fmt_e(delta, sigdigits=4),
      " w_pred=", fmt_e(w_pred, sigdigits=4),
      " w_true=", fmt_e(w_true, sigdigits=4),
      " nn_out=", fmt_e(nn_out, sigdigits=4),
      " F_contact=", fmt_e(F_contact, sigdigits=4),
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
nn_hidden_nodes_range = 1:max(3, env_int("HNODECB_STAGE1PLUS_MANUAL_NODES", 3))

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
  push!(layers, Lux.Dense(3, hidden, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform, use_bias=true))
  for _ in 1:num_hidden_layers
    push!(layers, Lux.Dense(hidden, hidden, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform, use_bias=true))
  end
  push!(layers, Lux.Dense(hidden, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform, use_bias=true))
  return Lux.Chain(layers...)
end

nn_input_from_state(u) = u[1:3]
nn_input_from_values(x1, x2, x3) = [x1, x2, x3]

function make_uode_func(appr, st, known_pars; zero_nn_override::Bool=false, nn_gain::Float64=1.0)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  rhs_call_counter = Ref(0)
  rhs_vals(u, p, t) =
    let appr = appr, st = st, k = k, wd = wd, m = m, c = c, Fd = Fd, R = R, dist = dist, Fad = Fad
      if trace_rhs_enabled
        rhs_call_counter[] += 1
        cidx = rhs_call_counter[]
        if cidx == 1 || cidx % trace_rhs_every == 0
          trace_rhs(() -> "rhs call=" * string(cidx) *
            " t=" * fmt_e(t, sigdigits=4) *
            " u1=" * fmt_e(u[1], sigdigits=4) *
            " u2=" * fmt_e(u[2], sigdigits=4) *
            " u3=" * fmt_e(u[3], sigdigits=4))
        end
      end
      ks = p.mech[1]
      cs = p.mech[2]

      s = dist + u[1] - u[3]
      delta = softplus(-s, adhesion_transition)
      delta = ifelse(delta > 0.0, delta, 0.0)
      w_pred = contact_weight(s, adhesion_transition)

      Fad_eff = Fad * w_pred
      F_contact = if zero_nn_override
        0.0
      elseif appr === nothing
        (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad_eff
      else
        nn_in = nn_input_from_state(u)
        uhat = appr(nn_in, p.p_net, st)[1]
        nn_gain * uhat[1] * w_pred
      end

      du1 = u[2]
      du2 = (Fd * cos(wd * t) - k * u[1] - c * u[2] + F_contact) / m
      du3 = (-F_contact - ks * u[3]) / cs
      return du1, du2, du3
    end

  f(u, p, t) = begin
    du1, du2, du3 = rhs_vals(u, p, t)
    [du1, du2, du3]
  end

  f(du, u, p, t) = begin
    du1, du2, du3 = rhs_vals(u, p, t)
    @inbounds du[1] = du1
    @inbounds du[2] = du2
    @inbounds du[3] = du3
    nothing
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
  Fad_eff = Fad * w_pred
  F_contact = if zero_nn_override
    0.0
  elseif appr === nothing
    (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad_eff
  else
    nn_in = nn_input_from_state(u)
    uhat = appr(nn_in, p_net, st)[1]
    nn_gain * uhat[1] * w_pred
  end
  return (Fd * cos(wd * t) - k * u[1] - c * u[2] + F_contact) / m
end

function fts_pred_from_state(u, p_net, appr, st, known_pars; zero_nn_override::Bool=false, nn_gain::Float64=1.0)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  s = dist + u[1] - u[3]
  delta = softplus(-s, adhesion_transition)
  delta = ifelse(delta > 0.0, delta, 0.0)
  w_pred = contact_weight(s, adhesion_transition)
  Fad_eff = Fad * w_pred
  return if zero_nn_override
    0.0
  elseif appr === nothing
    (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad_eff
  else
    nn_in = nn_input_from_state(u)
    uhat = appr(nn_in, p_net, st)[1]
    nn_gain * uhat[1] * w_pred
  end
end

function fts_from_x2dot_signal(uhat, x2dot_pred, times, known_pars)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  x1 = vec(uhat[1, :])
  x2 = vec(uhat[2, :])
  return m .* x2dot_pred .- Fd .* cos.(wd .* times) .+ k .* x1 .+ c .* x2
end

function loss_single_or_ms(θ, ode_data, x2dot_data, contact_mask, times,
  state12_scale, x2dot_scale, x3_scale,
  use_multiple_shooting, ms_group_size, ms_continuity_term,
  appr, st, known_pars, x3_t0_val,
  l2_weight, re_pnet; zero_nn_override::Bool=false,
  pred_traj_ref=nothing, nn_gain::Float64=1.0)

  # map raw -> bounded mech params
  mech = [
    bound_param(θ.mech_raw[1], ks_bounds[1], ks_bounds[2]),
    bound_param(θ.mech_raw[2], cs_bounds[1], cs_bounds[2])
  ]
  p_net_struct = re_pnet(θ.p_net)
  p = ComponentVector(p_net=p_net_struct, mech=mech)

  weights_val = ifelse.(contact_mask, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_val)
  inf_diag = (state=Inf, x2dot=Inf, x3_range=Inf, fts_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
  if !isfinite(weights_sum) || weights_sum <= 0
    log_inf("weights_sum")
    return Inf, inf_diag
  end

  total_state = 0.0
  total_x2dot = 0.0
  total_x3range = 0.0
  total_ftsrange = 0.0
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
        log_failure_diag("ms_retcode_seg" * string(i), sol, p, appr, st, known_pars, length(rg); zero_nn_override=zero_nn_override)
        return Inf, inf_diag
      end
      if size(sol, 2) != length(rg)
        log_inf("sol_size_mismatch_ms")
        log_failure_diag("ms_size_seg" * string(i), sol, p, appr, st, known_pars, length(rg); zero_nn_override=zero_nn_override)
        return Inf, inf_diag
      end
      preds[i] = Array(sol)

      uhat = preds[i]
      if any(x -> !isfinite(x), uhat)
        log_inf("uhat_nonfinite_ms")
        log_failure_diag("ms_uhat_nonfinite_seg" * string(i), sol, p, appr, st, known_pars, length(rg); zero_nn_override=zero_nn_override)
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
      x2dot_pred = nothing
      if !isempty(contact_idx) || fts_range_weight != 0.0
        local_idx = rg[contact_idx]
        uhat_const = Zygote.dropgrad(uhat)
        x2dot_pred = map(1:length(rg)) do j
          if trace_point_enabled && (j == 1 || j == length(rg) || j % trace_point_every == 0)
            trace_point(() -> "ms x2dot seg=" * string(i) * " j=" * string(j) * "/" * string(length(rg)))
          end
          x2dot_rhs(view(uhat_const, :, j), mech, p_net_struct, appr, st, known_pars, times[rg[j]]; zero_nn_override=zero_nn_override, nn_gain=nn_gain)
        end
        if any(x -> !isfinite(x), x2dot_pred)
          log_inf("x2dot_pred_nonfinite_ms")
          Zygote.ignore() do
            tprintln("FAIL_DIAG[ms_x2dot_nonfinite_seg", i, "]: first_nonfinite=", first_nonfinite_vector(x2dot_pred))
          end
          log_failure_diag("ms_x2dot_nonfinite_seg" * string(i), sol, p, appr, st, known_pars, length(rg); zero_nn_override=zero_nn_override)
          return Inf, inf_diag
        end
        if !isempty(contact_idx)
          x2_err = abs2.((x2dot_data[local_idx] .- x2dot_pred[contact_idx]) ./ x2dot_scale)
          x2_w = seg_weights[contact_idx]
          total_x2dot += sum(x2_w .* x2_err) / sum(x2_w)
        end
      end

      if fts_range_weight != 0.0
        fts_pred = fts_from_x2dot_signal(uhat, x2dot_pred, times[rg], known_pars)
        fts_exceed = abs.(fts_pred) .- fts_range_amp
        fts_pen = abs2.(softplus.(fts_exceed, fts_range_eps) ./ fts_scale)
        total_ftsrange += fts_range_weight * (sum(seg_weights .* fts_pen) / seg_weights_sum)
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

    if pred_traj_ref !== nothing
      Zygote.ignore() do
        pred_full = Matrix{Float64}(undef, size(ode_data, 1), length(times))
        for (i, rg) in enumerate(ranges)
          pred_full[:, rg] = preds[i]
        end
        pred_traj_ref[] = pred_full
      end
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
    trace_loss(() -> "ss postsolve enter n=" * string(length(times)))
    if !SciMLBase.successful_retcode(sol)
      log_inf("retcode=" * string(sol.retcode))
      log_failure_diag("ss_retcode", sol, p, appr, st, known_pars, length(times); zero_nn_override=zero_nn_override)
      return Inf, inf_diag
    end
    if size(sol, 2) != length(times)
      log_inf("sol_size_mismatch_ss")
      log_failure_diag("ss_size", sol, p, appr, st, known_pars, length(times); zero_nn_override=zero_nn_override)
      return Inf, inf_diag
    end
    trace_loss("ss postsolve array begin")
    uhat = Array(sol)
    trace_loss("ss postsolve array done")
    if any(x -> !isfinite(x), uhat)
      log_inf("uhat_nonfinite_ss")
      log_failure_diag("ss_uhat_nonfinite", sol, p, appr, st, known_pars, length(times); zero_nn_override=zero_nn_override)
      return Inf, inf_diag
    end

    trace_loss("ss postsolve state begin")
    state_err = vec(sum(abs2.((ode_data[1:2, :] .- uhat[1:2, :]) ./ state12_scale), dims=1))
    total_state = sum(weights_val .* state_err) / weights_sum
    trace_loss(() -> "ss postsolve state done state=" * fmt_e(total_state, sigdigits=4))

    contact_idx = findall(contact_mask)
    x2dot_pred = nothing
    if !isempty(contact_idx) || fts_range_weight != 0.0
      trace_loss(() -> "ss postsolve x2dot begin n=" * string(length(contact_idx)))
      uhat_const = Zygote.dropgrad(uhat)
      x2dot_pred = map(1:length(times)) do j
        if trace_point_enabled && (j == 1 || j == length(times) || j % trace_point_every == 0)
          trace_point(() -> "ss x2dot j=" * string(j) * "/" * string(length(times)))
        end
        x2dot_rhs(view(uhat_const, :, j), mech, p_net_struct, appr, st, known_pars, times[j]; zero_nn_override=zero_nn_override, nn_gain=nn_gain)
      end
      if any(x -> !isfinite(x), x2dot_pred)
        log_inf("x2dot_pred_nonfinite_ss")
        Zygote.ignore() do
          tprintln("FAIL_DIAG[ss_x2dot_nonfinite]: first_nonfinite=", first_nonfinite_vector(x2dot_pred))
        end
        log_failure_diag("ss_x2dot_nonfinite", sol, p, appr, st, known_pars, length(times); zero_nn_override=zero_nn_override)
        return Inf, inf_diag
      end
      if !isempty(contact_idx)
        x2_err = abs2.((x2dot_data[contact_idx] .- x2dot_pred[contact_idx]) ./ x2dot_scale)
        x2_w = weights_val[contact_idx]
        total_x2dot = sum(x2_w .* x2_err) / sum(x2_w)
      end
      trace_loss(() -> "ss postsolve x2dot done x2dot=" * fmt_e(total_x2dot, sigdigits=4))
    end

    trace_loss("ss postsolve fts begin")
    if fts_range_weight != 0.0
      fts_pred = fts_from_x2dot_signal(uhat, x2dot_pred, times, known_pars)
      fts_exceed = abs.(fts_pred) .- fts_range_amp
      fts_pen = abs2.(softplus.(fts_exceed, fts_range_eps) ./ fts_scale)
      total_ftsrange = fts_range_weight * (sum(weights_val .* fts_pen) / weights_sum)
    end
    trace_loss(() -> "ss postsolve fts done ftsr=" * fmt_e(total_ftsrange, sigdigits=4))

    trace_loss("ss postsolve x3range begin")
    exceed = abs.(uhat[3, :]) .- x3_range_amp
    range_pen = abs2.(softplus.(exceed, x3_range_eps) ./ x3_scale)
    total_x3range = x3_range_weight * (sum(weights_val .* range_pen) / weights_sum)
    trace_loss(() -> "ss postsolve x3range done x3r=" * fmt_e(total_x3range, sigdigits=4))

    if pred_traj_ref !== nothing
      Zygote.ignore() do
        pred_traj_ref[] = copy(uhat)
      end
    end

    Zygote.ignore() do
      x1_err_sum = sum(abs2.(ode_data[1, :] .- uhat[1, :]))
      x1_truth_sum = sum(abs2.(ode_data[1, :]))
      x3_err_sum = sum(abs2.(ode_data[3, :] .- uhat[3, :]))
      x3_truth_sum = sum(abs2.(ode_data[3, :]))
      count = length(times)
      x1_rec_ref[] = relative_rmse_pct(x1_err_sum, x1_truth_sum, count, scale_eps)
      x3_rec_ref[] = relative_rmse_pct(x3_err_sum, x3_truth_sum, count, scale_eps)
    end
    trace_loss("ss postsolve rec done")
  end

  l2_penalty = l2_weight * sum(abs2, θ.p_net)
  total = total_state + total_x2dot + total_x3range + total_ftsrange + total_cont + l2_penalty
  trace_loss(() -> "loss_single_or_ms done total=" * fmt_e(total, sigdigits=4))
  return total, (state=total_state, x2dot=total_x2dot, x3_range=total_x3range, fts_range=total_ftsrange, cont=total_cont,
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

peak_to_peak(v) = isempty(v) ? NaN : (maximum(v) - minimum(v))

function contact_onsets(contact_mask::AbstractVector{Bool})
  out = Int[]
  for i in eachindex(contact_mask)
    if contact_mask[i] && (i == firstindex(contact_mask) || !contact_mask[i - 1])
      push!(out, i)
    end
  end
  return out
end

function nearest_time_index(t_us::AbstractVector{<:Real}, target_us::Real)
  idx = findmin(abs.(Float64.(t_us) .- Float64(target_us)))[2]
  return Int(idx)
end

function start_anchor_index(t_us::AbstractVector{<:Real}, target_us::Real)
  idx = searchsortedfirst(Float64.(t_us), Float64(target_us))
  return clamp(Int(idx), 1, length(t_us))
end

function extend_stop_to_contact_end(contact_mask::AbstractVector{Bool}, stop_idx::Int)
  stop_idx = clamp(stop_idx, 1, length(contact_mask))
  contact_mask[stop_idx] || return stop_idx
  new_stop = stop_idx
  while new_stop < length(contact_mask) && contact_mask[new_stop + 1]
    new_stop += 1
  end
  return new_stop
end

function relocate_window_with_anchor(t_us::AbstractVector{<:Real}, start_idx::Int, stop_idx::Int; anchor::Symbol, target_us::Real)
  len = stop_idx - start_idx + 1
  n = length(t_us)
  len <= n || error("Window length exceeds available trajectory length.")

  if anchor == :start
    new_start = start_anchor_index(t_us, target_us)
    new_start = clamp(new_start, 1, n - len + 1)
    new_stop = new_start + len - 1
  elseif anchor == :stop
    new_stop = nearest_time_index(t_us, target_us)
    new_stop = clamp(new_stop, len, n)
    new_start = new_stop - len + 1
  else
    error("Unsupported anchor: " * string(anchor))
  end
  return new_start, new_stop
end

function stage1plus_stage2_w1_window_indices(times::AbstractVector, contact_mask, x1_signal::AbstractVector)
  isempty(times) && error("Stage1plus stage2_w1 window: empty time vector")
  length(contact_mask) == length(times) || error("Stage1plus stage2_w1 window: contact mask length does not match times")
  length(x1_signal) == length(times) || error("Stage1plus stage2_w1 window: x1 signal length does not match times")

  onset_post = contact_onsets(Vector{Bool}(contact_mask))
  length(onset_post) >= 3 || error("Stage1plus stage2_w1 window needs at least 3 contact onsets to build the first two-cycle window.")

  start_idx = onset_post[1]
  stop_idx = onset_post[3] - 1
  stop_idx >= start_idx || error("Stage1plus stage2_w1 window produced invalid bounds.")

  t_us = Float64.(times) .* 1e6
  new_start, new_stop = relocate_window_with_anchor(t_us, start_idx, stop_idx; anchor=:start, target_us=1034.0)
  new_stop = extend_stop_to_contact_end(Vector{Bool}(contact_mask), new_stop)
  return collect(new_start:new_stop)
end

function stage1plus_apply_stage2_window_time_overrides(selected::Vector{<:NamedTuple}, times::AbstractVector, contact_mask::AbstractVector)
  overrides = Dict(
    "first_contact" => (anchor=:start, target_us=1034.0),
    "max_x1_pp_change" => (anchor=:stop, target_us=1057.0),
    "tail_stable" => (anchor=:stop, target_us=1998.0),
  )

  t_us = Float64.(times) .* 1e6
  adjusted = NamedTuple[]
  for win in selected
    spec = overrides[win.role]
    new_start, new_stop = relocate_window_with_anchor(t_us, win.start_idx, win.stop_idx; anchor=spec.anchor, target_us=spec.target_us)
    if win.role == "first_contact"
      new_stop = extend_stop_to_contact_end(Vector{Bool}(contact_mask), new_stop)
    end
    push!(adjusted, merge(win, (
      start_idx=new_start,
      stop_idx=new_stop,
      len=new_stop - new_start + 1,
      t_start=times[new_start],
      t_stop=times[new_stop],
      idxs=collect(new_start:new_stop)
    )))
  end
  return adjusted
end

function stage1plus_stage2_window_manifest(times::AbstractVector, contact_mask, x1_signal::AbstractVector)
  isempty(times) && error("Stage1plus stage2 window manifest: empty time vector")
  length(contact_mask) == length(times) || error("Stage1plus stage2 window manifest: contact mask length does not match times")
  length(x1_signal) == length(times) || error("Stage1plus stage2 window manifest: x1 signal length does not match times")

  onset_post = contact_onsets(Vector{Bool}(contact_mask))
  length(onset_post) >= 3 || error("Stage1plus stage2 window manifest needs at least 3 contact onsets.")

  cycle_pp = Float64[]
  for k in 1:(length(onset_post) - 1)
    lo = onset_post[k]
    hi = onset_post[k + 1] - 1
    hi >= lo || error("Invalid cycle bounds while building Stage1plus stage2 window manifest.")
    push!(cycle_pp, peak_to_peak(@view x1_signal[lo:hi]))
  end

  candidates = NamedTuple[]
  for k in 1:(length(onset_post) - 2)
    start_idx = onset_post[k]
    stop_idx = onset_post[k + 2] - 1
    len = stop_idx - start_idx + 1
    pp1 = cycle_pp[k]
    pp2 = cycle_pp[k + 1]
    delta_pp = abs(pp2 - pp1)
    push!(candidates, (
      candidate_index=k,
      start_idx=start_idx,
      stop_idx=stop_idx,
      len=len,
      t_start=times[start_idx],
      t_stop=times[stop_idx],
      cycle1_index=k,
      cycle2_index=k + 1,
      x1_pp_cycle1=pp1,
      x1_pp_cycle2=pp2,
      x1_pp_delta=delta_pp,
      role="",
      label="",
      idxs=collect(start_idx:stop_idx)
    ))
  end
  isempty(candidates) && error("No valid Stage1plus stage2-style windows were built.")

  first_idx = 1
  deltas = [cand.x1_pp_delta for cand in candidates]
  change_order = sortperm(deltas; rev=true)
  tail_order = collect(length(candidates):-1:1)

  selected = NamedTuple[]
  seen = Set{Tuple{Int, Int}}()
  function add_window(role::String, candidate_idx::Int)
    cand = candidates[candidate_idx]
    key = (cand.start_idx, cand.stop_idx)
    key in seen && return false
    push!(seen, key)
    push!(selected, merge(cand, (role=role, label=role * "_window")))
    return true
  end
  function add_first_unique(role::String, candidate_order)
    for candidate_idx in candidate_order
      add_window(role, candidate_idx) && return candidate_idx
    end
    return nothing
  end

  add_window("first_contact", first_idx)
  add_first_unique("max_x1_pp_change", change_order)
  add_first_unique("tail_stable", tail_order)
  length(selected) >= 3 || error("Stage1plus stage2 window manifest could not find 3 unique windows.")
  return stage1plus_apply_stage2_window_time_overrides(selected, times, contact_mask)
end

function stage1plus_window_manifests(times::AbstractVector, contact_mask, window_mode::AbstractString, window_us::Real; x1_signal=nothing)
  mode = lowercase(strip(window_mode))
  if mode == "stage2_w123" || mode == "stage2-w123"
    x1_signal === nothing && error("stage2_w123 window mode requires x1_signal")
    return stage1plus_stage2_window_manifest(times, contact_mask, x1_signal)
  end
  idxs = collect(stage1plus_select_window_indices(times, contact_mask, window_mode, window_us; x1_signal=x1_signal))
  role = mode in ("full", "full_horizon", "full-horizon") ? "full_horizon" :
    (mode in ("stage2_w1", "stage2-w1") ? "first_contact" : "first_contact")
  return [(
    role=role,
    label=role * "_window",
    start_idx=first(idxs),
    stop_idx=last(idxs),
    len=length(idxs),
    t_start=times[first(idxs)],
    t_stop=times[last(idxs)],
    idxs=idxs
  )]
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

stage1plus_rand_loguniform(rng, lo, hi) = exp(rand(rng) * (log(hi) - log(lo)) + log(lo))

function stage1plus_shared_mech_draw(draw_id::Int)
  draw_key = 50_000_000 + draw_id
  ks0 = stage1plus_hash_loguniform(stage1_run_seed, draw_key, 1, ks_bounds[1], ks_bounds[2])
  cs0 = stage1plus_hash_loguniform(stage1_run_seed, draw_key, 2, cs_bounds[1], cs_bounds[2])
  return (draw_id=draw_id, ks0=ks0, cs0=cs0)
end

function stage1pluslight_loggrid_value(node_idx::Int, node_count::Int, lo::Float64, hi::Float64)
  node_count >= 1 || error("stage1pluslight log-grid node count must be >= 1")
  1 <= node_idx <= node_count || error("stage1pluslight node index out of range")
  if node_count == 1
    return sqrt(lo * hi)
  end
  α = (node_idx - 1) / (node_count - 1)
  return exp(log(lo) + α * (log(hi) - log(lo)))
end

stage1pluslight_total_trials(ks_node_count::Int, cs_node_count::Int, nn_seed_bank_size::Int) =
  ks_node_count * cs_node_count * nn_seed_bank_size

function stage1pluslight_decode_grid_trial(global_trial_id::Int, ks_node_count::Int, cs_node_count::Int, nn_seed_bank_size::Int)
  total_trials = stage1pluslight_total_trials(ks_node_count, cs_node_count, nn_seed_bank_size)
  1 <= global_trial_id <= total_trials || error("stage1pluslight global_trial_id out of range")
  trial0 = global_trial_id - 1
  nn_seed_bank_idx = mod(trial0, nn_seed_bank_size) + 1
  node_flat_idx = fld(trial0, nn_seed_bank_size) + 1
  ks_node_idx = fld(node_flat_idx - 1, cs_node_count) + 1
  cs_node_idx = mod(node_flat_idx - 1, cs_node_count) + 1
  ks0 = stage1pluslight_loggrid_value(ks_node_idx, ks_node_count, ks_bounds[1], ks_bounds[2])
  cs0 = stage1pluslight_loggrid_value(cs_node_idx, cs_node_count, cs_bounds[1], cs_bounds[2])
  node_label = "node_" * string(ks_node_idx) * "*" * string(cs_node_idx)
  return (
    global_trial_id=global_trial_id,
    node_flat_idx=node_flat_idx,
    ks_node_idx=ks_node_idx,
    cs_node_idx=cs_node_idx,
    nn_seed_bank_idx=nn_seed_bank_idx,
    ks0=ks0,
    cs0=cs0,
    node_label=node_label
  )
end

function stage1pluslight_nn_bank_seed(num_hidden_layers::Int, num_hidden_nodes::Int, nn_seed_bank_idx::Int)
  item_id = 200_000_000 + 10_000_000 * (1 + num_hidden_layers) + 1_000_000 * (1 + num_hidden_nodes) + nn_seed_bank_idx
  return stage1plus_hash_seed(stage1_run_seed, item_id, 23)
end

function stage1plus_architecture_trial(mech_draw_id::Int, ks_fixed::Float64, cs_fixed::Float64,
  num_hidden_layers::Int, num_hidden_nodes::Int,
  ode_train, x2dot_train, contact_train, times_train,
  ode_val, x2dot_val, contact_val, times_val,
  state12_scale, x2dot_scale, x3_t0_val,
  ms_group_size, ms_continuity_term,
  zero_nn_override::Bool, arch_epochs::Int, arch_lr::Float64)

  arch_item_id =
    100_000_000 + 10_000_000 * (1 + num_hidden_layers) + 1_000_000 * (1 + num_hidden_nodes) + mech_draw_id
  seed = stage1plus_hash_seed(stage1_run_seed, arch_item_id, 17)
  rng = StableRNG(seed)
  approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)
  p_net_init, st = Lux.setup(rng, approximating_neural_network)
  p_net_init = Flux.f64(p_net_init)
  p_net_vec0, re_pnet = Optimisers.destructure(p_net_init)
  p_net_struct0 = re_pnet(p_net_vec0)
  train_pred_ref = Ref{Any}(nothing)
  gnn_pred_ref = Ref{Any}(nothing)
  arch_use_dynamic_gnn = get(ENV, "HNODECB_STAGE1PLUS_ARCH_DYNAMIC_GNN", "0") == "1"
  arch_use_fixed_gnn = get(ENV, "HNODECB_STAGE1PLUS_ARCH_FIXED_GNN", "0") == "1"
  arch_fixed_gnn_q = env_float("HNODECB_STAGE1PLUS_ARCH_FIXED_GNN_Q", 0.95)
  arch_fixed_gnn_min = env_float("HNODECB_STAGE1PLUS_ARCH_FIXED_GNN_MIN", 1e-12)
  arch_fixed_gnn_max = env_float("HNODECB_STAGE1PLUS_ARCH_FIXED_GNN_MAX", 1e12)
  arch_gnn_log10_update_min = env_float("HNODECB_STAGE1PLUS_ARCH_GNN_LOG10_UPDATE_MIN", 1.0)
  train_idxs = collect(1:size(ode_train, 2))
  train_contact_true = contact_net_true_from_states(ode_train, train_idxs)
  train_eff_pos = effective_contact_positions_from_truth(train_contact_true)
  nn_gain_init_info = if arch_use_fixed_gnn || arch_use_dynamic_gnn
    decade_nn_gain_from_states(
      ode_train, train_idxs, p_net_struct0, approximating_neural_network, st;
      contact_true_ref=train_contact_true, eff_pos_ref=train_eff_pos,
      q=arch_fixed_gnn_q, gain_min=arch_fixed_gnn_min, gain_max=arch_fixed_gnn_max,
      log10_update_min=arch_gnn_log10_update_min
    )
  else
    (
      gain=1.0,
      candidate_gain=1.0,
      true_amp=NaN,
      pred_amp=NaN,
      true_decade=nothing,
      pred_decade=nothing,
      decade_delta=nothing,
      log10_diff=NaN,
      updated=false,
      reason="off"
    )
  end
  nn_gain_current = nn_gain_init_info.gain

  raw_init = [
    raw_from_value(ks_fixed, ks_bounds[1], ks_bounds[2]),
    raw_from_value(cs_fixed, cs_bounds[1], cs_bounds[2])
  ]

  theta0 = ComponentVector(p_net=p_net_vec0, mech_raw=raw_init)

  function train_loss_fn(theta; pred_traj_ref=nothing)
    loss_single_or_ms(theta, ode_train, x2dot_train, contact_train, times_train,
      state12_scale, x2dot_scale, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override,
      pred_traj_ref=pred_traj_ref, nn_gain=nn_gain_current)[1]
  end

  function val_loss_fn(theta)
    loss_single_or_ms(theta, ode_val, x2dot_val, contact_val, times_val,
      state12_scale, x2dot_scale, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override, nn_gain=nn_gain_current)[1]
  end

  lr_adapt = get(ENV, "HNODECB_STAGE1PLUS_ARCH_LR_ADAPT", get(ENV, "HNODECB_LR_ADAPT", "1")) == "1"
  lr_up_only = get(ENV, "HNODECB_STAGE1PLUS_ARCH_LR_UP_ONLY", "0") == "1"
  lr = arch_lr
  lr_min = env_float("HNODECB_STAGE1PLUS_ARCH_LR_MIN", env_float("HNODECB_LR_MIN", 1e-30))
  lr_max = env_float("HNODECB_STAGE1PLUS_ARCH_LR_MAX", env_float("HNODECB_LR_MAX", 5e1))
  lr_eta = env_float("HNODECB_STAGE1PLUS_ARCH_LR_ETA", env_float("HNODECB_LR_ETA", 0.5))
  lr_up_eta = env_float("HNODECB_STAGE1PLUS_ARCH_LR_UP_ETA", 1.0)
  lr_up_trigger_ratio = env_float("HNODECB_STAGE1PLUS_ARCH_LR_UP_TRIGGER_RATIO", 1.10)
  lr_up_min_factor = env_float("HNODECB_STAGE1PLUS_ARCH_LR_UP_MIN_FACTOR", 1.10)
  lr_up_max_factor = env_float("HNODECB_STAGE1PLUS_ARCH_LR_UP_MAX_FACTOR", 2.0)
  lr_ema_alpha = env_float("HNODECB_STAGE1PLUS_ARCH_LR_EMA", env_float("HNODECB_LR_EMA", 0.85))
  lr_eps = env_float("HNODECB_STAGE1PLUS_ARCH_LR_EPS", env_float("HNODECB_LR_EPS", 1e-30))
  lr_target_init = env_float("HNODECB_STAGE1PLUS_ARCH_LR_TARGET", NaN)
  lr_target_mult = env_float("HNODECB_STAGE1PLUS_ARCH_LR_TARGET_MULT", 1.0)
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
  trial_failed = false
  trial_fail_reason = ""
  trial_fail_epoch = 0
  trial_tag = "layers=" * string(num_hidden_layers) *
    " nodes=" * string(num_hidden_nodes) *
    " draw=" * string(mech_draw_id)
  gnn_mode = arch_use_dynamic_gnn ? "DYNAMIC" : (arch_use_fixed_gnn ? "FIXED" : "OFF")
  tprintln("      g_nn init=", fmt_e(nn_gain_current, sigdigits=4),
    " | mode=", gnn_mode,
    " | A_true=", fmt_e(nn_gain_init_info.true_amp, sigdigits=4),
    " A_pred=", fmt_e(nn_gain_init_info.pred_amp, sigdigits=4),
    " | dec_true=", (nn_gain_init_info.true_decade === nothing ? "NA" : string(nn_gain_init_info.true_decade)),
    " dec_pred=", (nn_gain_init_info.pred_decade === nothing ? "NA" : string(nn_gain_init_info.pred_decade)),
    " delta_dec=", (nn_gain_init_info.decade_delta === nothing ? "NA" : string(nn_gain_init_info.decade_delta)),
    " | candidate=", fmt_e(nn_gain_init_info.candidate_gain, sigdigits=4),
    " log10diff=", fmt_f(nn_gain_init_info.log10_diff, digits=3),
    " | reason=", nn_gain_init_info.reason)

  for epoch in 1:arch_epochs
    trace_begin_epoch(epoch)
    epoch_t0 = Zygote.ignore() do
      time()
    end
    nn_gain_epoch = nn_gain_current
    trace_loss(() -> "arch pullback start epoch=" * string(epoch) * " | " * trial_tag)
    train_loss_before, back = Zygote.pullback(train_loss_fn, theta)
    trace_loss(() -> "arch pullback done epoch=" * string(epoch) * " train=" * fmt_e(train_loss_before, sigdigits=4))
    trace_loss(() -> "arch back start epoch=" * string(epoch))
    grad_raw = first(back(1.0))
    trace_loss(() -> "arch back done epoch=" * string(epoch))
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
        grad_target[] = grad_ema[] * max(lr_target_mult, 1.0)
      end
      ratio = grad_target[] / (grad_ema[] + lr_eps)
      lr_new = lr
      if lr_up_only
        if ratio > lr_up_trigger_ratio
          up_factor = clamp(ratio^lr_up_eta, lr_up_min_factor, lr_up_max_factor)
          lr_new = clamp(lr * up_factor, lr_min, lr_max)
        end
      else
        lr_new = clamp(lr * ratio^lr_eta, lr_min, lr_max)
      end
      if lr_new != lr
        Optimisers.adjust!(opt_state, lr_new)
        lr = lr_new
      end
    end

    theta_base = deepcopy(theta)
    opt_state_base = deepcopy(opt_state)
    train_loss_after = Inf
    val_loss_epoch = Inf
    train_loss_trial_log = NaN
    train_loss_final_log = NaN
    t1 = NaN
    t2 = NaN
    recovered_after = 0
    epoch_bad_t2 = false
    step_fail_reason = ""
    step_accepted = false
    prev_epoch_loss_ref = (completed_epochs > 0 && isfinite(train_loss_last)) ? train_loss_last : NaN

    if !step_guard
      opt_state, theta = Optimisers.update(opt_state, theta, grad_update)
      train_pred_ref[] = nothing
      train_loss_after = train_loss_fn(theta; pred_traj_ref=train_pred_ref)
      train_loss_trial_log = train_loss_after
      val_loss_epoch = val_loss_fn(theta)
      step_vec = theta .- theta_base
      t1 = sum(grad_raw .* step_vec)
      t2 = (train_loss_after - train_loss_before) - t1
      epoch_bad_t2 = !(isfinite(t2) && isfinite(t1)) || t2 > abs(t1)
      step_accepted = true
    else
      step_attempt = 0
      while step_attempt <= step_retry_max
        step_attempt += 1
        opt_input = deepcopy(opt_state_base)
        opt_trial, theta_trial = Optimisers.update(opt_input, theta_base, grad_update)
        train_pred_ref[] = nothing
        train_loss_trial = train_loss_fn(theta_trial; pred_traj_ref=train_pred_ref)
        train_loss_trial_log = train_loss_trial
        step_vec = theta_trial .- theta_base
        t1_trial = sum(grad_raw .* step_vec)
        t2_trial = (train_loss_trial - train_loss_before) - t1_trial
        if !(isfinite(t2_trial) && isfinite(t1_trial)) || t2_trial > abs(t1_trial)
          epoch_bad_t2 = true
        end
        step_max_loss_increase = isfinite(train_loss_before) ? abs(train_loss_before) * step_max_loss_frac : Inf
        prev_epoch_max_loss = isfinite(prev_epoch_loss_ref) ? prev_epoch_loss_ref * (1 + step_max_loss_frac) : Inf
        if !isfinite(train_loss_trial)
          step_fail_reason = "step_trial_loss_nonfinite"
        elseif isfinite(prev_epoch_max_loss) && train_loss_trial > prev_epoch_max_loss
          step_fail_reason = "step_prev_epoch_loss_jump"
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
          if step_fail_reason == "step_trial_loss_nonfinite"
            tprintln("      trial fails -- ", trial_tag,
              " | epoch=", epoch,
              " | reason=", step_fail_reason,
              " | lr=", fmt_e(lr, sigdigits=3))
            trial_failed = true
            trial_fail_reason = step_fail_reason
            trial_fail_epoch = epoch
            train_loss_after = Inf
            val_loss_epoch = Inf
            t1 = t1_trial
            t2 = t2_trial
            break
          end
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

    if trial_failed
      train_loss_last = Inf
      val_loss_last = Inf
      break
    end

    gnn_pred_ref[] = nothing
    theta_loss_eval = train_loss_fn(theta; pred_traj_ref=gnn_pred_ref)
    train_loss_after = theta_loss_eval
    train_loss_final_log = train_loss_after
    val_loss_epoch = val_loss_fn(theta)

    gnn_loss_before = train_loss_after
    gnn_val_before = val_loss_epoch
    gnn_loss_after = gnn_loss_before
    gnn_val_after = gnn_val_before
    gnn_next = nn_gain_epoch
    gnn_update_reason = "unchanged"
    gnn_true_amp = NaN
    gnn_pred_amp = NaN
    gnn_true_decade = nothing
    gnn_pred_decade = nothing
    gnn_decade_delta = nothing
    gnn_log10_diff = NaN
    gnn_candidate = nn_gain_epoch
    if arch_use_dynamic_gnn
      gnn_info = decade_nn_gain_from_states(
        ode_train, train_idxs, re_pnet(theta.p_net), approximating_neural_network, st;
        contact_true_ref=train_contact_true, eff_pos_ref=train_eff_pos,
        q=arch_fixed_gnn_q, gain_min=arch_fixed_gnn_min, gain_max=arch_fixed_gnn_max,
        current_gain=nn_gain_epoch, log10_update_min=arch_gnn_log10_update_min
      )
      gnn_next = gnn_info.gain
      gnn_true_amp = gnn_info.true_amp
      gnn_pred_amp = gnn_info.pred_amp
      gnn_true_decade = gnn_info.true_decade
      gnn_pred_decade = gnn_info.pred_decade
      gnn_decade_delta = gnn_info.decade_delta
      gnn_log10_diff = gnn_info.log10_diff
      gnn_candidate = gnn_info.candidate_gain
      nn_gain_current = gnn_next
      gnn_loss_after = train_loss_fn(theta)
      gnn_val_after = val_loss_fn(theta)
      train_loss_after = gnn_loss_after
      val_loss_epoch = gnn_val_after
      gnn_update_reason = gnn_info.reason
    end

    gnn_changed = !(isfinite(nn_gain_epoch) && isfinite(gnn_next) &&
      isapprox(nn_gain_epoch, gnn_next; rtol=1e-12, atol=0.0))
    if gnn_changed
      tprintln("      gnn-loss-flow",
        " | gnn_old=", fmt_e(nn_gain_epoch, sigdigits=4),
        " theta_fixed=", fmt_e(gnn_loss_before, sigdigits=4),
        " -> gnn_new=", fmt_e(gnn_next, sigdigits=4),
        " after_gnn=", fmt_e(gnn_loss_after, sigdigits=4),
        " | A_true=", fmt_e(gnn_true_amp, sigdigits=4),
        " A_pred=", fmt_e(gnn_pred_amp, sigdigits=4),
        " | dec_true=", (gnn_true_decade === nothing ? "NA" : string(gnn_true_decade)),
        " dec_pred=", (gnn_pred_decade === nothing ? "NA" : string(gnn_pred_decade)),
        " delta_dec=", (gnn_decade_delta === nothing ? "NA" : string(gnn_decade_delta)),
        " | candidate=", fmt_e(gnn_candidate, sigdigits=4),
        " log10diff=", fmt_f(gnn_log10_diff, digits=3),
        " | val_before=", fmt_e(gnn_val_before, sigdigits=4),
        " val_after=", fmt_e(gnn_val_after, sigdigits=4),
        " | reason=", gnn_update_reason)
    end

    if trial_failed
      train_loss_last = Inf
      val_loss_last = Inf
      break
    end

    if epoch_bad_t2
      t2_gt_abs_t1_count += 1
    end
    ks_hat_epoch = bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2])
    cs_hat_epoch = bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])
    ks_err_epoch = percent_error_pct(ks_hat_epoch, ks_true)
    cs_err_epoch = percent_error_pct(cs_hat_epoch, cs_true)
    _, train_parts_epoch = loss_single_or_ms(theta, ode_train, x2dot_train, contact_train, times_train,
      state12_scale, x2dot_scale, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override, nn_gain=nn_gain_current)
    _, val_parts_epoch = loss_single_or_ms(theta, ode_val, x2dot_val, contact_val, times_val,
      state12_scale, x2dot_scale, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override, nn_gain=nn_gain_current)
    train_nn_metrics_epoch = contact_net_monitor_metrics(
      ode_train, train_idxs, re_pnet(theta.p_net), approximating_neural_network, st;
      contact_true_ref=train_contact_true, eff_pos_ref=train_eff_pos, nn_gain=nn_gain_current
    )
    epoch_dt = Zygote.ignore() do
      time() - epoch_t0
    end
    push!(epoch_durations, epoch_dt)
    tprintln("    arch epoch ", epoch, "/", arch_epochs,
      " -- ", trial_tag,
      " | train_final=", fmt_e(train_loss_after, sigdigits=4),
      " val=", fmt_e(val_loss_epoch, sigdigits=4),
      " | dt=", fmt_f(epoch_dt, digits=2), "s",
      " | t1=", fmt_e(t1, sigdigits=4),
      " t2=", fmt_e(t2, sigdigits=4),
      " | gnn=", fmt_e(nn_gain_current, sigdigits=4),
      " | lr=", fmt_e(lr, sigdigits=3),
      " | scale[p_net=", fmt_e(grad_scale_p_net, sigdigits=3),
      ", mech_raw=", fmt_e(grad_scale_mech, sigdigits=3), "]")
    tprintln("      mech/nn",
      " | ks=", fmt_e(ks_hat_epoch, sigdigits=4),
      " (true=", fmt_e(ks_true, sigdigits=4),
      ", err=", fmt_f(ks_err_epoch, digits=2), "%)",
      " | cs=", fmt_e(cs_hat_epoch, sigdigits=4),
      " (true=", fmt_e(cs_true, sigdigits=4),
      ", err=", fmt_f(cs_err_epoch, digits=2), "%)",
      " | nnErr(train)=", fmt_f(train_nn_metrics_epoch.fcontact_err, digits=2), "%")
    tprintln("      loss parts(train)",
      " | state=", fmt_e(train_parts_epoch.state, sigdigits=3),
      " x2dot=", fmt_e(train_parts_epoch.x2dot, sigdigits=3),
      " x3r=", fmt_e(train_parts_epoch.x3_range, sigdigits=3),
      " ftsr=", fmt_e(train_parts_epoch.fts_range, sigdigits=3),
      " cont=", fmt_e(train_parts_epoch.cont, sigdigits=3))
    tprintln("      loss parts(val)",
      " | state=", fmt_e(val_parts_epoch.state, sigdigits=3),
      " x2dot=", fmt_e(val_parts_epoch.x2dot, sigdigits=3),
      " x3r=", fmt_e(val_parts_epoch.x3_range, sigdigits=3),
      " ftsr=", fmt_e(val_parts_epoch.fts_range, sigdigits=3),
      " cont=", fmt_e(val_parts_epoch.cont, sigdigits=3))
    train_loss_last = train_loss_after
    val_loss_last = val_loss_epoch
    completed_epochs += 1
  end
  trace_end_epoch()

  val_loss_end = trial_failed ? Inf : (completed_epochs > 0 ? val_loss_last : val_loss_fn(theta))
  trial_complete = !trial_failed && completed_epochs == arch_epochs && isfinite(val_loss_start) && isfinite(val_loss_end)
  time_per_epoch = trial_complete ? (sum(epoch_durations) / arch_epochs) : Inf
  obj1 = trial_complete ? (val_loss_end - val_loss_start) / arch_epochs : Inf
  obj2 = time_per_epoch
  obj3 = trial_complete ? Float64(t2_gt_abs_t1_count) : Float64(arch_epochs)

  return (
    mech_draw_id=mech_draw_id,
    ks0=ks_fixed,
    cs0=cs_fixed,
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
    nn_gain=nn_gain_current,
    obj1=obj1,
    obj2=obj2,
    obj3=obj3,
    failure_epoch=trial_fail_epoch,
    failure_reason=trial_fail_reason
  )
end

function stage1pluslight_joint_trial(global_trial_id::Int, ks_fixed::Float64, cs_fixed::Float64,
  ks_node_idx::Int, cs_node_idx::Int, nn_seed_bank_idx::Int,
  num_hidden_layers::Int, num_hidden_nodes::Int,
  ode_train, x2dot_train, contact_train, times_train,
  ode_val, x2dot_val, contact_val, times_val,
  state12_scale, x2dot_scale, x3_t0_val,
  ms_group_size, ms_continuity_term,
  zero_nn_override::Bool, trial_opt_enabled::Bool, trial_epochs::Int, trial_lr::Float64;
  window_role::AbstractString="single")

  seed = stage1pluslight_nn_bank_seed(num_hidden_layers, num_hidden_nodes, nn_seed_bank_idx)
  rng = StableRNG(seed)
  approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)
  p_net_init, st = Lux.setup(rng, approximating_neural_network)
  p_net_init = Flux.f64(p_net_init)
  p_net_vec0, re_pnet = Optimisers.destructure(p_net_init)
  train_pred_ref = Ref{Any}(nothing)
  gnn_pred_ref = Ref{Any}(nothing)
  p_net_struct0 = re_pnet(p_net_vec0)
  trial_use_dynamic_gnn = get(ENV, "HNODECB_STAGE1PLUS_TRIAL_DYNAMIC_GNN", "0") == "1"
  trial_use_fixed_gnn = get(ENV, "HNODECB_STAGE1PLUS_TRIAL_FIXED_GNN", "0") == "1"
  trial_fixed_gnn_q = env_float("HNODECB_STAGE1PLUS_TRIAL_FIXED_GNN_Q", 0.95)
  trial_fixed_gnn_min = env_float("HNODECB_STAGE1PLUS_TRIAL_FIXED_GNN_MIN", 1e-12)
  trial_fixed_gnn_max = env_float("HNODECB_STAGE1PLUS_TRIAL_FIXED_GNN_MAX", 1e12)
  trial_gnn_log10_update_min = env_float("HNODECB_STAGE1PLUS_TRIAL_GNN_LOG10_UPDATE_MIN", 1.0)
  train_idxs = collect(1:size(ode_train, 2))
  train_contact_true = contact_net_true_from_states(ode_train, train_idxs)
  train_eff_pos = effective_contact_positions_from_truth(train_contact_true)
  nn_gain_init_info = if trial_use_fixed_gnn || trial_use_dynamic_gnn
    decade_nn_gain_from_states(
      ode_train, train_idxs, p_net_struct0, approximating_neural_network, st;
      contact_true_ref=train_contact_true, eff_pos_ref=train_eff_pos,
      q=trial_fixed_gnn_q, gain_min=trial_fixed_gnn_min, gain_max=trial_fixed_gnn_max,
      log10_update_min=trial_gnn_log10_update_min
    )
  else
    (
      gain=1.0,
      candidate_gain=1.0,
      true_amp=NaN,
      pred_amp=NaN,
      true_decade=nothing,
      pred_decade=nothing,
      decade_delta=nothing,
      log10_diff=NaN,
      updated=false,
      reason="off"
    )
  end
  nn_gain_current = nn_gain_init_info.gain

  raw_init = [
    raw_from_value(ks_fixed, ks_bounds[1], ks_bounds[2]),
    raw_from_value(cs_fixed, cs_bounds[1], cs_bounds[2])
  ]
  theta0 = ComponentVector(p_net=p_net_vec0, mech_raw=raw_init)

  function train_loss_fn(theta; pred_traj_ref=nothing)
    loss_single_or_ms(theta, ode_train, x2dot_train, contact_train, times_train,
      state12_scale, x2dot_scale, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override,
      pred_traj_ref=pred_traj_ref, nn_gain=nn_gain_current)[1]
  end

  function val_loss_fn(theta)
    loss_single_or_ms(theta, ode_val, x2dot_val, contact_val, times_val,
      state12_scale, x2dot_scale, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override, nn_gain=nn_gain_current)[1]
  end

  lr_adapt = get(ENV, "HNODECB_STAGE1PLUS_TRIAL_LR_ADAPT", "1") == "1"
  lr_up_only = get(ENV, "HNODECB_STAGE1PLUS_TRIAL_LR_UP_ONLY", "1") == "1"
  lr = trial_lr
  lr_min = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_MIN", 1e-30)
  lr_max = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_MAX", 5e1)
  lr_eta = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_ETA", 0.5)
  lr_up_eta = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_UP_ETA", 1.0)
  lr_up_trigger_ratio = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_UP_TRIGGER_RATIO", 1.10)
  lr_up_min_factor = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_UP_MIN_FACTOR", 1.10)
  lr_up_max_factor = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_UP_MAX_FACTOR", 2.0)
  lr_ema_alpha = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_EMA", 0.85)
  lr_eps = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_EPS", 1e-30)
  lr_target_init = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_TARGET", NaN)
  lr_target_mult = env_float("HNODECB_STAGE1PLUS_TRIAL_LR_TARGET_MULT", 3.0)
  grad_ema = Ref(isfinite(lr_target_init) && lr_target_init > 0 ? lr_target_init : NaN)
  grad_target = Ref(isfinite(lr_target_init) && lr_target_init > 0 ? lr_target_init : NaN)

  grad_scale_p_net = env_float("HNODECB_STAGE1PLUS_TRIAL_GRAD_SCALE_P_NET", 1.0)
  grad_scale_mech = env_float("HNODECB_STAGE1PLUS_TRIAL_GRAD_SCALE_MECH", 1.0)

  step_guard = get(ENV, "HNODECB_STAGE1PLUS_TRIAL_STEP_GUARD", "1") == "1"
  step_retry_max = max(0, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_TRIAL_STEP_RETRIES", "6")))
  step_retry_lr_factor = env_float("HNODECB_STAGE1PLUS_TRIAL_STEP_RETRY_LR_FACTOR", 0.1)
  step_max_loss_frac = env_float("HNODECB_STAGE1PLUS_TRIAL_STEP_MAX_LOSS_FRAC", 0.1)

  early_stop_epoch = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_TRIAL_EARLY_STOP_EPOCHS", "5")))
  early_stop_min_drop_frac = env_float("HNODECB_STAGE1PLUS_TRIAL_EARLY_STOP_MIN_DROP_FRAC", 0.01)

  trial_failed = false
  trial_fail_reason = ""
  trial_fail_epoch = 0
  early_stopped = false
  early_stop_reason = ""
  monitor_enabled = !isempty(monitor_eff_positions)
  empty_nn_metrics = (
    fcontact_err=NaN,
    raw_min=NaN,
    raw_max=NaN,
    raw_mean=NaN,
    raw_neg_frac=NaN,
    fnet_min=NaN,
    fnet_max=NaN,
    fnet_mean=NaN,
    fnet_abs_p95=NaN
  )
  node_label = "node_" * string(ks_node_idx) * "*" * string(cs_node_idx)
  trial_tag = "trial=" * string(global_trial_id) *
    " " * node_label *
    " window=" * window_role *
    " seed_bank=" * string(nn_seed_bank_idx) *
    " layers=" * string(num_hidden_layers) *
    " nodes=" * string(num_hidden_nodes)
  trial_gnn_mode = trial_use_dynamic_gnn ? "DYNAMIC" : (trial_use_fixed_gnn ? "FIXED" : "OFF")
  tprintln("      g_nn init=", fmt_e(nn_gain_current, sigdigits=4),
    " | mode=", trial_gnn_mode,
    " | A_true=", fmt_e(nn_gain_init_info.true_amp, sigdigits=4),
    " A_pred=", fmt_e(nn_gain_init_info.pred_amp, sigdigits=4),
    " | dec_true=", (nn_gain_init_info.true_decade === nothing ? "NA" : string(nn_gain_init_info.true_decade)),
    " dec_pred=", (nn_gain_init_info.pred_decade === nothing ? "NA" : string(nn_gain_init_info.pred_decade)),
    " delta_dec=", (nn_gain_init_info.decade_delta === nothing ? "NA" : string(nn_gain_init_info.decade_delta)),
    " | candidate=", fmt_e(nn_gain_init_info.candidate_gain, sigdigits=4),
    " log10diff=", fmt_f(nn_gain_init_info.log10_diff, digits=3),
    " | reason=", nn_gain_init_info.reason)

  function trial_exception_reason(stage::AbstractString, err)
    err isa InterruptException && rethrow(err)
    msg = sprint(showerror, err)
    msg = replace(msg, '\n' => ' ')
    msg = replace(msg, '\r' => ' ')
    return stage * ":" * msg
  end

  val_loss_start = try
    val_loss_fn(theta0)
  catch err
    trial_failed = true
    trial_fail_reason = trial_exception_reason("initial_val_exception", err)
    trial_fail_epoch = 0
    tprintln("Stage1pluslight ", trial_tag,
      " -- trial fails before epoch 1 | reason=", trial_fail_reason)
    Inf
  end

  val_nn_metrics_start = empty_nn_metrics
  if monitor_enabled
    try
      p_net_struct_start = re_pnet(theta0.p_net)
      val_nn_metrics_start = contact_net_monitor_metrics(
        ode_data_full, monitor_idx_full, p_net_struct_start,
        approximating_neural_network, st;
        contact_true_ref=monitor_contact_true, eff_pos_ref=monitor_eff_positions, nn_gain=nn_gain_current
      )
    catch err
      tprintln("Stage1pluslight ", trial_tag,
        " -- pre-gradient nn monitor failed | reason=",
        trial_exception_reason("nn_monitor_start_exception", err))
      val_nn_metrics_start = empty_nn_metrics
    end
  end

  theta = deepcopy(theta0)
  opt_state = Optimisers.setup(Optimisers.Adam(lr), theta)
  epoch_durations = Float64[]
  retry_count_total = 0
  completed_epochs = 0
  val_loss_last = val_loss_start
  train_loss_last = Inf
  best_val_loss = val_loss_start

  if trial_opt_enabled
    for epoch in 1:trial_epochs
      trace_begin_epoch(epoch)
      epoch_t0 = Zygote.ignore() do
        time()
      end
      train_loss_before = Inf
      grad_raw = nothing
      try
        train_loss_before, back = Zygote.pullback(train_loss_fn, theta)
        grad_raw = first(back(1.0))
      catch err
        trial_failed = true
        trial_fail_reason = trial_exception_reason("adjoint_exception", err)
        trial_fail_epoch = epoch
        tprintln("      trial fails -- ", trial_tag,
          " | epoch=", epoch,
          " | reason=", trial_fail_reason)
        break
      end
      if grad_raw === nothing
        tprintln("    joint epoch ", epoch, "/", trial_epochs,
          " -- ", trial_tag, " | grad=None -> stop")
        trial_failed = true
        trial_fail_reason = "grad_none"
        trial_fail_epoch = epoch
        break
      end
      grad_norm_raw = grad_norm_safe(grad_raw)
      if !isfinite(train_loss_before) || !isfinite(grad_norm_raw)
        tprintln("    joint epoch ", epoch, "/", trial_epochs,
          " -- ", trial_tag,
          " | nonfinite train/grad -> stop",
          " | train=", fmt_e(train_loss_before, sigdigits=4),
          " grad_norm=", fmt_e(grad_norm_raw, sigdigits=4))
        trial_failed = true
        trial_fail_reason = "nonfinite_train_or_grad"
        trial_fail_epoch = epoch
        break
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
          grad_target[] = grad_ema[] * max(lr_target_mult, 1.0)
        end
        ratio = grad_target[] / (grad_ema[] + lr_eps)
        lr_new = lr
        if lr_up_only
          if ratio > lr_up_trigger_ratio
            up_factor = clamp(ratio^lr_up_eta, lr_up_min_factor, lr_up_max_factor)
            lr_new = clamp(lr * up_factor, lr_min, lr_max)
          end
        else
          lr_new = clamp(lr * ratio^lr_eta, lr_min, lr_max)
        end
        if lr_new != lr
          Optimisers.adjust!(opt_state, lr_new)
          lr = lr_new
        end
      end

      theta_base = deepcopy(theta)
      opt_state_base = deepcopy(opt_state)
      train_loss_after = Inf
      val_loss_epoch = Inf
      train_loss_trial_log = NaN
      train_loss_final_log = NaN
      step_fail_reason = ""
      step_accepted = false
      prev_epoch_loss_ref = (completed_epochs > 0 && isfinite(train_loss_last)) ? train_loss_last : NaN

      if !step_guard
        try
          opt_state, theta = Optimisers.update(opt_state, theta, grad_update)
          train_pred_ref[] = nothing
          train_loss_after = train_loss_fn(theta; pred_traj_ref=train_pred_ref)
          train_loss_trial_log = train_loss_after
          val_loss_epoch = val_loss_fn(theta)
          step_accepted = true
        catch err
          trial_failed = true
          trial_fail_reason = trial_exception_reason("step_exception", err)
          trial_fail_epoch = epoch
          tprintln("      trial fails -- ", trial_tag,
            " | epoch=", epoch,
            " | reason=", trial_fail_reason)
        end
      else
        step_attempt = 0
        while step_attempt <= step_retry_max
          step_attempt += 1
          opt_input = deepcopy(opt_state_base)
          opt_trial, theta_trial = Optimisers.update(opt_input, theta_base, grad_update)
          train_pred_ref[] = nothing
          train_loss_trial = Inf
          try
            train_loss_trial = train_loss_fn(theta_trial; pred_traj_ref=train_pred_ref)
            train_loss_trial_log = train_loss_trial
          catch err
            trial_failed = true
            trial_fail_reason = trial_exception_reason("step_trial_exception", err)
            trial_fail_epoch = epoch
            tprintln("      trial fails -- ", trial_tag,
              " | epoch=", epoch,
              " | reason=", trial_fail_reason)
            break
          end
          step_max_loss_increase = isfinite(train_loss_before) ? abs(train_loss_before) * step_max_loss_frac : Inf
          prev_epoch_max_loss = isfinite(prev_epoch_loss_ref) ? prev_epoch_loss_ref * (1 + step_max_loss_frac) : Inf
          if !isfinite(train_loss_trial)
            step_fail_reason = "step_trial_loss_nonfinite"
          elseif isfinite(prev_epoch_max_loss) && train_loss_trial > prev_epoch_max_loss
            step_fail_reason = "step_prev_epoch_loss_jump"
          elseif isfinite(step_max_loss_increase) && train_loss_trial > train_loss_before + step_max_loss_increase
            step_fail_reason = "step_trial_loss_jump"
          else
            step_fail_reason = ""
          end

          tprintln("      step-guard metrics ", step_attempt, "/", step_retry_max,
            " | train_prev_epoch=", fmt_e(prev_epoch_loss_ref, sigdigits=4),
            " train_epoch_start=", fmt_e(train_loss_before, sigdigits=4),
            " train_trial=", fmt_e(train_loss_trial_log, sigdigits=4),
            " | prev_epoch_max=", fmt_e(prev_epoch_max_loss, sigdigits=4),
            " current_epoch_max=", fmt_e(train_loss_before + step_max_loss_increase, sigdigits=4))

          if step_fail_reason == ""
            opt_state = opt_trial
            theta = theta_trial
            train_loss_after = train_loss_trial
            try
              val_loss_epoch = val_loss_fn(theta)
            catch err
              trial_failed = true
              trial_fail_reason = trial_exception_reason("step_val_exception", err)
              trial_fail_epoch = epoch
              tprintln("      trial fails -- ", trial_tag,
                " | epoch=", epoch,
                " | reason=", trial_fail_reason)
              break
            end
            step_accepted = true
            retry_count_total += (step_attempt - 1)
            break
          end

          if trial_failed
            break
          end

          if step_attempt > step_retry_max
            retry_count_total += step_retry_max
            if step_fail_reason == "step_trial_loss_nonfinite"
              tprintln("      trial fails -- ", trial_tag,
                " | epoch=", epoch,
                " | reason=", step_fail_reason,
                " | lr=", fmt_e(lr, sigdigits=3))
              trial_failed = true
              trial_fail_reason = step_fail_reason
              trial_fail_epoch = epoch
              break
            end
            tprintln("      step-guard: rejected update after ", step_retry_max,
              " retries | reason=", step_fail_reason,
              " | lr=", fmt_e(lr, sigdigits=3))
            theta = theta_base
            opt_state = opt_state_base
            train_loss_after = train_loss_before
            val_loss_epoch = val_loss_fn(theta)
            break
          end

          lr_new = max(lr * step_retry_lr_factor, lr_min)
          if lr_new < lr
            lr = lr_new
          end
          Optimisers.adjust!(opt_state_base, lr)
          tprintln("      step-guard retry ", step_attempt, "/", step_retry_max,
            " -- reason=", step_fail_reason,
            " | lr=", fmt_e(lr, sigdigits=3))
        end
        if !step_accepted && train_loss_after === Inf && !trial_failed
          train_loss_after = train_loss_before
          try
            val_loss_epoch = val_loss_fn(theta)
          catch err
            trial_failed = true
            trial_fail_reason = trial_exception_reason("fallback_val_exception", err)
            trial_fail_epoch = epoch
            tprintln("      trial fails -- ", trial_tag,
              " | epoch=", epoch,
              " | reason=", trial_fail_reason)
          end
        end
      end

      if trial_failed
        train_loss_last = Inf
        val_loss_last = Inf
        break
      end

      train_loss_final_log = train_loss_after
      gnn_loss_before = train_loss_after
      gnn_val_before = val_loss_epoch
      gnn_loss_after = gnn_loss_before
      gnn_val_after = gnn_val_before
      gnn_epoch = nn_gain_current
      gnn_next = gnn_epoch
      gnn_update_reason = "unchanged"
      if trial_use_dynamic_gnn
        gnn_info = decade_nn_gain_from_states(
          ode_train, train_idxs, re_pnet(theta.p_net), approximating_neural_network, st;
          contact_true_ref=train_contact_true, eff_pos_ref=train_eff_pos,
          q=trial_fixed_gnn_q, gain_min=trial_fixed_gnn_min, gain_max=trial_fixed_gnn_max,
          current_gain=gnn_epoch, log10_update_min=trial_gnn_log10_update_min
        )
        gnn_next = gnn_info.gain
        gnn_update_reason = gnn_info.reason
        nn_gain_current = gnn_next
        gnn_pred_ref[] = nothing
        gnn_loss_after = train_loss_fn(theta; pred_traj_ref=gnn_pred_ref)
        gnn_val_after = val_loss_fn(theta)
        train_loss_after = gnn_loss_after
        train_loss_final_log = gnn_loss_after
        val_loss_epoch = gnn_val_after
      elseif trial_use_fixed_gnn
        nn_gain_current = nn_gain_init_info.gain
      end

      if trial_failed
        train_loss_last = Inf
        val_loss_last = Inf
        break
      end
      gnn_changed = !(isfinite(gnn_epoch) && isfinite(gnn_next) &&
        isapprox(gnn_epoch, gnn_next; rtol=1e-12, atol=0.0))
      if gnn_changed
        tprintln("      gnn-loss-flow",
          " | gnn_old=", fmt_e(gnn_epoch, sigdigits=4),
          " gnn_new=", fmt_e(gnn_next, sigdigits=4),
          " | train_before=", fmt_e(gnn_loss_before, sigdigits=4),
          " after_gnn=", fmt_e(gnn_loss_after, sigdigits=4),
          " | val_before=", fmt_e(gnn_val_before, sigdigits=4),
          " val_after=", fmt_e(gnn_val_after, sigdigits=4),
          " | reason=", gnn_update_reason)
      end

      epoch_dt = Zygote.ignore() do
        time() - epoch_t0
      end
      push!(epoch_durations, epoch_dt)
      tprintln("    joint epoch ", epoch, "/", trial_epochs,
        " -- ", trial_tag,
        " | train_final=", fmt_e(train_loss_after, sigdigits=4),
        " val=", fmt_e(val_loss_epoch, sigdigits=4),
        " | dt=", fmt_f(epoch_dt, digits=2), "s",
        " | gnn=", fmt_e(nn_gain_current, sigdigits=4),
        " | lr=", fmt_e(lr, sigdigits=3))
      tprintln("      epoch-train-flow",
        " | prev_epoch=", fmt_e(prev_epoch_loss_ref, sigdigits=4),
        " start=", fmt_e(train_loss_before, sigdigits=4),
        " trial=", fmt_e(train_loss_trial_log, sigdigits=4),
        " final=", fmt_e(train_loss_final_log, sigdigits=4),
        " | guard_reason=", (step_fail_reason == "" ? "accepted_or_kept" : step_fail_reason))

      train_loss_last = train_loss_after
      val_loss_last = val_loss_epoch
      if isfinite(val_loss_epoch)
        best_val_loss = min(best_val_loss, val_loss_epoch)
      end
      completed_epochs += 1

      if completed_epochs >= early_stop_epoch &&
         isfinite(val_loss_start) && isfinite(best_val_loss)
        drop_frac = (val_loss_start - best_val_loss) / max(abs(val_loss_start), 1e-30)
        if drop_frac < early_stop_min_drop_frac
          early_stopped = true
          early_stop_reason = "epoch_" * string(completed_epochs) *
            "_drop_lt_" * fmt_f(early_stop_min_drop_frac, digits=4)
          tprintln("      early-stop -- ", trial_tag,
            " | epoch=", completed_epochs,
            " | best_drop=", fmt_f(100 * drop_frac, digits=2), "%")
          break
        end
      end
    end
  end
  trace_end_epoch()

  empty_parts = (state=NaN, x2dot=NaN, x3_range=NaN, fts_range=NaN, cont=NaN, x1_rec=NaN, x3_rec=NaN)
  train_reason = ""
  val_reason = ""
  train_loss_end = Inf
  val_loss_end = Inf
  val_diag = empty_parts

  if !trial_failed
    try
      inf_context[] = "train_eval"
      last_inf_reason[] = ""
      train_loss_end, _ = loss_single_or_ms(theta, ode_train, x2dot_train, contact_train, times_train,
        state12_scale, x2dot_scale, x3_scale,
        use_multiple_shooting, ms_group_size, ms_continuity_term,
        approximating_neural_network, st, known_pars, x3_t0_val,
        0.0, re_pnet; zero_nn_override=zero_nn_override, nn_gain=nn_gain_current)
      train_reason = last_inf_reason[]

      inf_context[] = "val_eval"
      last_inf_reason[] = ""
      val_loss_end, val_diag = loss_single_or_ms(theta, ode_val, x2dot_val, contact_val, times_val,
        state12_scale, x2dot_scale, x3_scale,
        use_multiple_shooting, ms_group_size, ms_continuity_term,
        approximating_neural_network, st, known_pars, x3_t0_val,
        0.0, re_pnet; zero_nn_override=zero_nn_override, nn_gain=nn_gain_current)
      val_reason = last_inf_reason[]
    catch err
      trial_failed = true
      trial_fail_reason = trial_exception_reason("final_eval_exception", err)
      if trial_fail_epoch == 0
        trial_fail_epoch = completed_epochs
      end
      train_reason = trial_fail_reason
      val_reason = trial_fail_reason
      train_loss_end = Inf
      val_loss_end = Inf
      val_diag = empty_parts
      tprintln("Stage1pluslight ", trial_tag,
        " -- trial fails during final eval | reason=", trial_fail_reason)
    end
  else
    train_reason = trial_fail_reason
    val_reason = trial_fail_reason
  end

  ks_hat = bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2])
  cs_hat = bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])
  ks_err_pct = rel_err_pct(ks_hat, ks_true, scale_eps)
  cs_err_pct = rel_err_pct(cs_hat, cs_true, scale_eps)

  val_nn_metrics = empty_nn_metrics
  if monitor_enabled && !trial_failed
    p_net_struct_final = re_pnet(theta.p_net)
    val_nn_metrics = contact_net_monitor_metrics(
      ode_data_full, monitor_idx_full, p_net_struct_final,
      approximating_neural_network, st;
      contact_true_ref=monitor_contact_true, eff_pos_ref=monitor_eff_positions, nn_gain=nn_gain_current
    )
  end

  is_viable = !trial_failed && !early_stopped && isfinite(train_loss_end) && isfinite(val_loss_end) &&
    (!monitor_enabled || isfinite(val_nn_metrics.fcontact_err))

  params = Dict{Any, Any}(
    "trial_id" => global_trial_id,
    "ks0" => ks_fixed,
    "cs0" => cs_fixed,
    "ks_node_idx" => ks_node_idx,
    "cs_node_idx" => cs_node_idx,
    "node_label" => node_label,
    "nn_seed_bank_idx" => nn_seed_bank_idx,
    "nn_init_seed" => seed,
    "nn_force_mode" => "contact_net",
    "num_hidden_layers" => num_hidden_layers,
    "num_hidden_nodes" => num_hidden_nodes,
    "stage1plus_standalone" => true,
    "stage1plus_search_mode" => "loggrid_seedbank"
  )

  return (
    loss=val_loss_end,
    train_loss=train_loss_end,
    val_loss=val_loss_end,
    val_loss_start=val_loss_start,
    params=params,
    val_parts=val_diag,
    ks_hat=ks_hat,
    cs_hat=cs_hat,
    ks_err_pct=ks_err_pct,
    cs_err_pct=cs_err_pct,
    val_nn_err_start=val_nn_metrics_start.fcontact_err,
    val_raw_min_start=val_nn_metrics_start.raw_min,
    val_raw_max_start=val_nn_metrics_start.raw_max,
    val_raw_mean_start=val_nn_metrics_start.raw_mean,
    val_raw_neg_frac_start=val_nn_metrics_start.raw_neg_frac,
    val_fnet_min_start=val_nn_metrics_start.fnet_min,
    val_fnet_max_start=val_nn_metrics_start.fnet_max,
    val_fnet_mean_start=val_nn_metrics_start.fnet_mean,
    val_fnet_abs_p95_start=val_nn_metrics_start.fnet_abs_p95,
    val_nn_err=val_nn_metrics.fcontact_err,
    val_raw_min=val_nn_metrics.raw_min,
    val_raw_max=val_nn_metrics.raw_max,
    val_raw_mean=val_nn_metrics.raw_mean,
    val_raw_neg_frac=val_nn_metrics.raw_neg_frac,
    val_fnet_min=val_nn_metrics.fnet_min,
    val_fnet_max=val_nn_metrics.fnet_max,
    val_fnet_mean=val_nn_metrics.fnet_mean,
    val_fnet_abs_p95=val_nn_metrics.fnet_abs_p95,
    is_viable=is_viable,
    train_reason=train_reason == "" ? (trial_failed ? trial_fail_reason : "") : train_reason,
    val_reason=val_reason == "" ? (trial_failed ? trial_fail_reason : "") : val_reason,
    p_net_vec=copy(theta.p_net),
    nn_gain=nn_gain_current,
    nn_force_mode="contact_net",
    completed_epochs=completed_epochs,
    retry_count_total=retry_count_total,
    early_stopped=early_stopped,
    early_stop_reason=early_stop_reason,
    trial_failed=trial_failed,
    failure_reason=trial_fail_reason,
    failure_epoch=trial_fail_epoch,
    lr_last=lr,
    time_per_epoch=(completed_epochs > 0 ? sum(epoch_durations) / completed_epochs : Inf)
  )
end

mean_finite(vals) = begin
  finite = [Float64(v) for v in vals if v isa Number && isfinite(v)]
  isempty(finite) ? NaN : mean(finite)
end

trial_id_or_zero(rec) = begin
  if hasproperty(rec, :params) && rec.params isa AbstractDict && haskey(rec.params, "trial_id")
    v = rec.params["trial_id"]
    return v isa Integer ? Int(v) : 0
  end
  return 0
end

function stage1pluslight_load_resume_trials(path::AbstractString;
  searches_total::Int,
  shard_idx::Int,
  shard_cnt::Int,
  ks_nodes::Int,
  cs_nodes::Int,
  nn_seeds_per_node::Int,
  window_mode::AbstractString)

  checkpoint_path = String(path)
  backup_path = checkpoint_path * ".bak"
  candidate_paths = String[]
  if isfile(checkpoint_path)
    try
      filesize(checkpoint_path) > 0 && push!(candidate_paths, checkpoint_path)
    catch
    end
  end
  if isfile(backup_path)
    try
      filesize(backup_path) > 0 && push!(candidate_paths, backup_path)
    catch
    end
  end
  isempty(candidate_paths) && return Any[], ""

  data = nothing
  loaded_from = ""
  last_reason = ""
  for candidate_path in candidate_paths
    local candidate_data
    try
      candidate_data = deserialize(candidate_path)
    catch err
      last_reason = "deserialize_failed:" * sprint(showerror, err)
      continue
    end
    data = candidate_data
    loaded_from = candidate_path == backup_path ? "backup" : "primary"
    break
  end
  data === nothing && return Any[], (last_reason == "" ? "missing_checkpoint_payload" : last_reason)
  haskey(data, :trial_parameters) || return Any[], "missing_trial_parameters"
  if haskey(data, :searches_per_candidate) && data.searches_per_candidate != searches_total
    return Any[], "searches_total_mismatch"
  end
  if haskey(data, :stage1plus_shard_index) && data.stage1plus_shard_index != shard_idx
    return Any[], "shard_index_mismatch"
  end
  if haskey(data, :stage1plus_shard_count) && data.stage1plus_shard_count != shard_cnt
    return Any[], "shard_count_mismatch"
  end
  if haskey(data, :stage1plus_grid_ks_nodes) && data.stage1plus_grid_ks_nodes != ks_nodes
    return Any[], "ks_nodes_mismatch"
  end
  if haskey(data, :stage1plus_grid_cs_nodes) && data.stage1plus_grid_cs_nodes != cs_nodes
    return Any[], "cs_nodes_mismatch"
  end
  if haskey(data, :stage1plus_grid_nn_seeds_per_node) && data.stage1plus_grid_nn_seeds_per_node != nn_seeds_per_node
    return Any[], "nn_seeds_mismatch"
  end
  if haskey(data, :stage1plus_window_mode) && String(data.stage1plus_window_mode) != String(window_mode)
    return Any[], "window_mode_mismatch"
  end
  return copy(data.trial_parameters), (loaded_from == "backup" ? "loaded_from_backup" : "")
end

function stage1pluslight_write_checkpoint_atomic(path::AbstractString, payload)
  checkpoint_path = String(path)
  tmp_path = checkpoint_path * ".tmp"
  backup_path = checkpoint_path * ".bak"

  rm(tmp_path; force=true)
  open(tmp_path, "w") do io
    serialize(io, payload)
    flush(io)
  end

  if isfile(checkpoint_path)
    rm(backup_path; force=true)
    mv(checkpoint_path, backup_path; force=true)
  end

  mv(tmp_path, checkpoint_path; force=true)
  return nothing
end

join_nonempty_unique(vals) = begin
  kept = String[]
  for v in vals
    s = strip(String(v))
    isempty(s) && continue
    s in kept || push!(kept, s)
  end
  isempty(kept) ? "" : join(kept, "; ")
end

function average_val_parts(parts_list)
  return (
    state=mean_finite([p.state for p in parts_list]),
    x2dot=mean_finite([p.x2dot for p in parts_list]),
    x3_range=mean_finite([p.x3_range for p in parts_list]),
    fts_range=mean_finite([p.fts_range for p in parts_list]),
    cont=mean_finite([p.cont for p in parts_list]),
    x1_rec=mean_finite([p.x1_rec for p in parts_list]),
    x3_rec=mean_finite([p.x3_rec for p in parts_list])
  )
end

function stage1pluslight_joint_trial_multiwindow(global_trial_id::Int, ks_fixed::Float64, cs_fixed::Float64,
  ks_node_idx::Int, cs_node_idx::Int, nn_seed_bank_idx::Int,
  num_hidden_layers::Int, num_hidden_nodes::Int,
  window_bundles,
  state12_scale, x2dot_scale,
  ms_group_size, ms_continuity_term,
  zero_nn_override::Bool, trial_opt_enabled::Bool, trial_epochs::Int, trial_lr::Float64)

  trial_opt_enabled && error("stage2_w123 window aggregation currently supports only trial_opt=OFF")
  window_recs = Any[]
  for bundle in window_bundles
    rec = stage1pluslight_joint_trial(
      global_trial_id, ks_fixed, cs_fixed, ks_node_idx, cs_node_idx, nn_seed_bank_idx,
      num_hidden_layers, num_hidden_nodes,
      bundle.ode_train, bundle.x2dot_train, bundle.contact_train, bundle.times_train,
      bundle.ode_val, bundle.x2dot_val, bundle.contact_val, bundle.times_val,
      state12_scale, x2dot_scale, bundle.x3_t0_val,
      ms_group_size, ms_continuity_term,
      zero_nn_override, trial_opt_enabled, trial_epochs, trial_lr;
      window_role=bundle.role
    )
    push!(window_recs, rec)
  end

  ref = first(window_recs)
  window_roles = [bundle.role for bundle in window_bundles]
  window_train_losses = [rec.train_loss for rec in window_recs]
  window_val_losses = [rec.val_loss for rec in window_recs]
  window_val_starts = [rec.val_loss_start for rec in window_recs]
  window_nn_gains = [rec.nn_gain for rec in window_recs]
  params = copy(ref.params)
  params["stage1plus_window_mode"] = "stage2_w123"
  params["window_roles"] = window_roles
  params["window_train_losses"] = window_train_losses
  params["window_val_losses"] = window_val_losses
  params["window_val_loss_starts"] = window_val_starts
  params["window_nn_gains"] = window_nn_gains

  return (
    loss=mean_finite([rec.loss for rec in window_recs]),
    train_loss=mean_finite(window_train_losses),
    val_loss=mean_finite(window_val_losses),
    val_loss_start=mean_finite(window_val_starts),
    params=params,
    val_parts=average_val_parts([rec.val_parts for rec in window_recs]),
    ks_hat=ref.ks_hat,
    cs_hat=ref.cs_hat,
    ks_err_pct=ref.ks_err_pct,
    cs_err_pct=ref.cs_err_pct,
    val_nn_err_start=mean_finite([rec.val_nn_err_start for rec in window_recs]),
    val_raw_min_start=mean_finite([rec.val_raw_min_start for rec in window_recs]),
    val_raw_max_start=mean_finite([rec.val_raw_max_start for rec in window_recs]),
    val_raw_mean_start=mean_finite([rec.val_raw_mean_start for rec in window_recs]),
    val_raw_neg_frac_start=mean_finite([rec.val_raw_neg_frac_start for rec in window_recs]),
    val_fnet_min_start=mean_finite([rec.val_fnet_min_start for rec in window_recs]),
    val_fnet_max_start=mean_finite([rec.val_fnet_max_start for rec in window_recs]),
    val_fnet_mean_start=mean_finite([rec.val_fnet_mean_start for rec in window_recs]),
    val_fnet_abs_p95_start=mean_finite([rec.val_fnet_abs_p95_start for rec in window_recs]),
    val_nn_err=mean_finite([rec.val_nn_err for rec in window_recs]),
    val_raw_min=mean_finite([rec.val_raw_min for rec in window_recs]),
    val_raw_max=mean_finite([rec.val_raw_max for rec in window_recs]),
    val_raw_mean=mean_finite([rec.val_raw_mean for rec in window_recs]),
    val_raw_neg_frac=mean_finite([rec.val_raw_neg_frac for rec in window_recs]),
    val_fnet_min=mean_finite([rec.val_fnet_min for rec in window_recs]),
    val_fnet_max=mean_finite([rec.val_fnet_max for rec in window_recs]),
    val_fnet_mean=mean_finite([rec.val_fnet_mean for rec in window_recs]),
    val_fnet_abs_p95=mean_finite([rec.val_fnet_abs_p95 for rec in window_recs]),
    is_viable=all(rec.is_viable for rec in window_recs),
    train_reason=join_nonempty_unique([rec.train_reason for rec in window_recs]),
    val_reason=join_nonempty_unique([rec.val_reason for rec in window_recs]),
    p_net_vec=copy(ref.p_net_vec),
    nn_gain=mean_finite(window_nn_gains),
    nn_force_mode=ref.nn_force_mode,
    completed_epochs=ref.completed_epochs,
    retry_count_total=sum(rec.retry_count_total for rec in window_recs),
    early_stopped=any(rec.early_stopped for rec in window_recs),
    early_stop_reason=join_nonempty_unique([rec.early_stop_reason for rec in window_recs]),
    trial_failed=any(rec.trial_failed for rec in window_recs),
    failure_reason=join_nonempty_unique([rec.failure_reason for rec in window_recs]),
    failure_epoch=maximum([rec.failure_epoch for rec in window_recs]),
    lr_last=ref.lr_last,
    time_per_epoch=mean_finite([rec.time_per_epoch for rec in window_recs]),
    window_train_losses=window_train_losses,
    window_val_losses=window_val_losses
  )
end

sanitize_stage1plus_reason(reason::AbstractString) = replace(replace(String(reason), '\n' => ' '), '\r' => ' ')

function make_stage1plus_architecture_failed_record(mech_draw_id::Int, ks_fixed::Float64, cs_fixed::Float64,
  num_hidden_layers::Int, num_hidden_nodes::Int, arch_epochs::Int;
  param_count::Int=0, nn_gain::Float64=NaN, val_loss_start::Float64=Inf,
  completed_epochs::Int=0, retry_count_total::Int=0, failure_epoch::Int=0,
  failure_reason::AbstractString="")

  reason = sanitize_stage1plus_reason(failure_reason)
  return (
    mech_draw_id=mech_draw_id,
    ks0=ks_fixed,
    cs0=cs_fixed,
    num_hidden_layers=num_hidden_layers,
    num_hidden_nodes=num_hidden_nodes,
    hidden=2^num_hidden_nodes,
    param_count=param_count,
    trial_complete=false,
    completed_epochs=completed_epochs,
    train_loss_end=Inf,
    val_loss_start=val_loss_start,
    val_loss_end=Inf,
    time_per_epoch=Inf,
    t2_gt_abs_t1_count=arch_epochs,
    retry_count_total=retry_count_total,
    nn_gain=nn_gain,
    obj1=Inf,
    obj2=Inf,
    obj3=Float64(arch_epochs),
    failure_epoch=failure_epoch,
    failure_reason=reason
  )
end

function make_stage1pluslight_failed_record(global_trial_id::Int, ks_fixed::Float64, cs_fixed::Float64,
  ks_node_idx::Int, cs_node_idx::Int, nn_seed_bank_idx::Int,
  num_hidden_layers::Int, num_hidden_nodes::Int;
  window_role::AbstractString="unknown", window_roles::Vector{String}=String[],
  failure_reason::AbstractString="", failure_epoch::Int=0,
  completed_epochs::Int=0, retry_count_total::Int=0,
  early_stopped::Bool=false, early_stop_reason::AbstractString="", lr_last::Float64=NaN)

  reason = sanitize_stage1plus_reason(failure_reason)
  node_label = "node_" * string(ks_node_idx) * "*" * string(cs_node_idx)
  empty_parts = (state=NaN, x2dot=NaN, x3_range=NaN, fts_range=NaN, cont=NaN, x1_rec=NaN, x3_rec=NaN)
  params = Dict{Any, Any}(
    "trial_id" => global_trial_id,
    "ks0" => ks_fixed,
    "cs0" => cs_fixed,
    "ks_node_idx" => ks_node_idx,
    "cs_node_idx" => cs_node_idx,
    "node_label" => node_label,
    "nn_seed_bank_idx" => nn_seed_bank_idx,
    "nn_force_mode" => "contact_net",
    "num_hidden_layers" => num_hidden_layers,
    "num_hidden_nodes" => num_hidden_nodes,
    "stage1plus_standalone" => true,
    "stage1plus_search_mode" => "loggrid_seedbank"
  )
  if window_role == "stage2_w123"
    params["stage1plus_window_mode"] = "stage2_w123"
    params["window_roles"] = window_roles
    params["window_train_losses"] = fill(Inf, length(window_roles))
    params["window_val_losses"] = fill(Inf, length(window_roles))
    params["window_val_loss_starts"] = fill(Inf, length(window_roles))
    params["window_nn_gains"] = fill(NaN, length(window_roles))
  end
  return (
    loss=Inf,
    train_loss=Inf,
    val_loss=Inf,
    val_loss_start=Inf,
    params=params,
    val_parts=empty_parts,
    ks_hat=ks_fixed,
    cs_hat=cs_fixed,
    ks_err_pct=rel_err_pct(ks_fixed, ks_true, scale_eps),
    cs_err_pct=rel_err_pct(cs_fixed, cs_true, scale_eps),
    val_nn_err_start=NaN,
    val_raw_min_start=NaN,
    val_raw_max_start=NaN,
    val_raw_mean_start=NaN,
    val_raw_neg_frac_start=NaN,
    val_fnet_min_start=NaN,
    val_fnet_max_start=NaN,
    val_fnet_mean_start=NaN,
    val_fnet_abs_p95_start=NaN,
    val_nn_err=NaN,
    val_raw_min=NaN,
    val_raw_max=NaN,
    val_raw_mean=NaN,
    val_raw_neg_frac=NaN,
    val_fnet_min=NaN,
    val_fnet_max=NaN,
    val_fnet_mean=NaN,
    val_fnet_abs_p95=NaN,
    is_viable=false,
    train_reason=reason,
    val_reason=reason,
    p_net_vec=Float64[],
    nn_gain=NaN,
    nn_force_mode="contact_net",
    completed_epochs=completed_epochs,
    retry_count_total=retry_count_total,
    early_stopped=early_stopped,
    early_stop_reason=early_stop_reason,
    trial_failed=true,
    failure_reason=reason,
    failure_epoch=failure_epoch,
    lr_last=lr_last,
    time_per_epoch=Inf
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
      mean_retry_count=mean([rec.retry_count_total for rec in recs]),
      mean_completed_epochs=mean([rec.completed_epochs for rec in recs]),
      mean_val_loss_start=mean([rec.val_loss_start for rec in recs]),
      mean_val_loss_end=mean([rec.val_loss_end for rec in recs])
    ))
  end

  sort!(summaries, by = rec -> (rec.num_hidden_layers, rec.num_hidden_nodes))
  obj1_norm = normalize_min_obj_metric(Float64[rec.mean_obj1 for rec in summaries])
  obj2_norm = normalize_min_obj_metric(Float64[rec.mean_obj2 for rec in summaries])
  obj3_norm = normalize_min_obj_metric(Float64[rec.mean_obj3 for rec in summaries])
  scored = Any[]
  for i in eachindex(summaries)
    rec = summaries[i]
    weighted_obj1 = obj_a * obj1_norm[i]
    weighted_obj2 = obj_b * obj2_norm[i]
    weighted_obj3 = obj_c * obj3_norm[i]
    min_obj = sqrt(weighted_obj1^2 + weighted_obj2^2 + weighted_obj3^2)
    push!(scored, merge(rec, (
      norm_obj1=obj1_norm[i],
      norm_obj2=obj2_norm[i],
      norm_obj3=obj3_norm[i],
      weighted_obj1=weighted_obj1,
      weighted_obj2=weighted_obj2,
      weighted_obj3=weighted_obj3,
      min_obj=min_obj
    )))
  end

  sort!(scored, by = rec -> rec.min_obj)
  return scored
end

function stage1plus_first_contact_window_indices(times::AbstractVector, contact_mask, window_us::Real)
  isempty(times) && error("Stage1plus first-contact window: empty time vector")
  if length(contact_mask) != length(times)
    error("Stage1plus first-contact window: contact mask length does not match times length")
  end
  first_contact_idx = findfirst(identity, contact_mask)
  if first_contact_idx === nothing
    return stage1plus_window_indices(times, window_us)
  end
  t_start = times[first_contact_idx]
  t_stop = t_start + window_us
  stop_idx = searchsortedlast(times, t_stop)
  stop_idx = clamp(stop_idx, first_contact_idx, length(times))
  return first_contact_idx:stop_idx
end

function stage1plus_select_window_indices(times::AbstractVector, contact_mask, window_mode::AbstractString, window_us::Real; x1_signal=nothing)
  mode = lowercase(strip(window_mode))
  if mode == "full" || mode == "full_horizon" || mode == "full-horizon"
    return eachindex(times)
  elseif mode == "stage2_w123" || mode == "stage2-w123"
    x1_signal === nothing && error("stage2_w123 window mode requires x1_signal")
    # Single-window helper paths fall back to W1 when the main search is
    # configured to aggregate W1&W2&W3.
    return stage1plus_stage2_w1_window_indices(times, contact_mask, x1_signal)
  elseif mode == "stage2_w1" || mode == "stage2-w1"
    x1_signal === nothing && error("stage2_w1 window mode requires x1_signal")
    return stage1plus_stage2_w1_window_indices(times, contact_mask, x1_signal)
  elseif mode == "first_contact" || mode == "first-contact"
    return stage1plus_first_contact_window_indices(times, contact_mask, window_us)
  else
    error("Unsupported HNODECB_STAGE1PLUS_WINDOW_MODE=$(repr(window_mode)); expected 'full', 'stage2_w123', 'stage2_w1', or 'first_contact'")
  end
end

use_multiple_shooting = false
use_l2_regularization = false
l2_weight = 0.0
# Fixed train/validation split (not part of hyperparameter search)
val_stride = 5
val_offset = 2
# Stage1: default to NO-ADAM flow unless explicitly enabled
use_adam = get(ENV, "HNODECB_STAGE1_USE_ADAM", "0") == "1"
defer_main = get(ENV, "HNODECB_STAGE1_DEFER_MAIN", "0") == "1"

if !defer_main
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
    F_contact = uhat[1] * w_pred
    x2dot0 = (Fd * cos(wd * t0) - k * u0[1] - c * u0[2] + F_contact) / m
    x3dot0 = (-F_contact - mech[1] * u0[3]) / mech[2]
    tprintln("  selftest phys @t0=", fmt_e(t0, sigdigits=3))
    tprintln("    u0: x1=", fmt_e(u0[1], sigdigits=3),
      " x2=", fmt_e(u0[2], sigdigits=3),
      " x3=", fmt_e(u0[3], sigdigits=3))
    tprintln("    contact: s=", fmt_e(s, sigdigits=3),
      " delta=", fmt_e(delta, sigdigits=3),
      " w_pred=", fmt_e(w_pred, sigdigits=3),
      " w_true=", fmt_e(w_true, sigdigits=3))
    tprintln("    nn_out=", fmt_e(uhat[1], sigdigits=3),
      " F_contact=", fmt_e(F_contact, sigdigits=3))
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
      " ftsr=", fmt_e(diag.fts_range, sigdigits=3),
      " cont=", fmt_e(diag.cont, sigdigits=3))
  end
  if get(ENV, "HNODECB_LOG_NN_ERR", "1") == "1"
    nn_metrics = contact_net_monitor_metrics(
      ode_data_full, monitor_idx_full, p_net_struct, approximating_neural_network, st;
      contact_true_ref=monitor_contact_true, eff_pos_ref=monitor_eff_positions)
    tprintln("  selftest nn: F_contact err=", fmt_f(nn_metrics.fcontact_err, digits=2), "%")
    tprintln("  selftest nn raw: min=", fmt_e(nn_metrics.raw_min, sigdigits=3),
      " max=", fmt_e(nn_metrics.raw_max, sigdigits=3),
      " mean=", fmt_e(nn_metrics.raw_mean, sigdigits=3),
      " neg=", fmt_f(100 * nn_metrics.raw_neg_frac, digits=2), "%")
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
    tprintln("=== Stage1 SANITY (oracle F_contact + true mech) ===")
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
        " ftsr=", fmt_e(diag.fts_range, sigdigits=3),
        " cont=", fmt_e(diag.cont, sigdigits=3))
      tprintln("  sanity rec: x1=", fmt_f(diag.x1_rec, digits=2),
        "% x3=", fmt_f(diag.x3_rec, digits=2), "%")
    end

    if get(ENV, "HNODECB_LOG_NN_ERR", "1") == "1"
      nn_metrics = contact_net_monitor_metrics(
        ode_data_full, monitor_idx_full, p_net, nothing, st;
        contact_true_ref=monitor_contact_true, eff_pos_ref=monitor_eff_positions)
      tprintln("  sanity nn: F_contact err=", fmt_f(nn_metrics.fcontact_err, digits=2), "%")
      tprintln("  sanity nn raw: min=", fmt_e(nn_metrics.raw_min, sigdigits=3),
        " max=", fmt_e(nn_metrics.raw_max, sigdigits=3),
        " mean=", fmt_e(nn_metrics.raw_mean, sigdigits=3),
        " neg=", fmt_f(100 * nn_metrics.raw_neg_frac, digits=2), "%")
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

    fcontact_monitor_enabled = get(ENV, "HNODECB_LOG_FHERTZ_ERR", "1") == "1" && !isempty(monitor_eff_positions)

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
            " ftsr=", fmt_e(diag.fts_range, sigdigits=3),
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
        if fcontact_monitor_enabled
          Zygote.ignore() do
            p_net_struct = re_pnet(θ.p_net)
            nn_metrics = contact_net_monitor_metrics(
              ode_data_full, monitor_idx_full, p_net_struct,
              approximating_neural_network, st;
              contact_true_ref=monitor_contact_true, eff_pos_ref=monitor_eff_positions
            )
            tprintln("  nn: F_contact err=", fmt_f(nn_metrics.fcontact_err, digits=2), "%")
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
  optuna = stage1_get_optuna()
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
      " ftsr=", fmt_e(rec.val_parts.fts_range, sigdigits=3),
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
    error_level=error_level
  ))
elseif use_stage1pluslight
  let
  searches_total = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE", "250000")))
  final_topk = max(0, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_FINAL_TOPK", "0")))
  zero_nn_override_stage1plus = get(ENV, "HNODECB_STAGE1PLUS_ZERO_NN", "0") == "1"
  shard_idx = parse(Int, get(ENV, "HNODECB_STAGE1_SHARD_INDEX", "1"))
  shard_cnt = parse(Int, get(ENV, "HNODECB_STAGE1_SHARD_COUNT", "1"))
  stage1plus_emit_local_selection = get(ENV, "HNODECB_STAGE1PLUS_EMIT_LOCAL_SELECTION", "1") == "1"
  if shard_idx < 1 || shard_idx > shard_cnt
    error("HNODECB_STAGE1_SHARD_INDEX must be in 1..HNODECB_STAGE1_SHARD_COUNT")
  end

  fixed_ms_group_size = 10
  fixed_ms_continuity_term = 1e-3
  inherit_stage1plus_arch = get(ENV, "HNODECB_STAGE1PLUS_INHERIT_STAGE1_ARCH", "1") != "0"
  arch_selected_layers = begin
    default_layers = inherit_stage1plus_arch ? nn_fixed_num_hidden_layers :
      begin
        raw_manual = strip(get(ENV, "HNODECB_STAGE1PLUS_MANUAL_LAYERS", string(nn_fixed_num_hidden_layers)))
        v_manual = tryparse(Int, raw_manual)
        (v_manual === nothing) ? nn_fixed_num_hidden_layers : v_manual
      end
    raw = strip(get(ENV, "HNODECB_STAGE1PLUS_SELECTED_LAYERS", string(default_layers)))
    v = tryparse(Int, raw)
    (v === nothing) ? default_layers : v
  end
  arch_selected_nodes = begin
    default_nodes = inherit_stage1plus_arch ? nn_fixed_num_hidden_nodes :
      begin
        raw_manual = strip(get(ENV, "HNODECB_STAGE1PLUS_MANUAL_NODES", string(nn_fixed_num_hidden_nodes)))
        v_manual = tryparse(Int, raw_manual)
        (v_manual === nothing) ? nn_fixed_num_hidden_nodes : v_manual
      end
    raw = strip(get(ENV, "HNODECB_STAGE1PLUS_SELECTED_NODES", string(default_nodes)))
    v = tryparse(Int, raw)
    (v === nothing) ? default_nodes : v
  end
  if !(arch_selected_layers in nn_hidden_layers_range)
    error("HNODECB_STAGE1PLUS_SELECTED_LAYERS=$(arch_selected_layers) is outside nn_hidden_layers_range=$(collect(nn_hidden_layers_range))")
  end
  if !(arch_selected_nodes in nn_hidden_nodes_range)
    error("HNODECB_STAGE1PLUS_SELECTED_NODES=$(arch_selected_nodes) is outside nn_hidden_nodes_range=$(collect(nn_hidden_nodes_range))")
  end

  arch_screen_only = get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_ONLY", "0") == "1"
  arch_screen_trials_per_arch = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH", "60")))
  arch_screen_epochs = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_EPOCHS", "50")))
  arch_screen_warmup = get(ENV, "HNODECB_STAGE1PLUS_ARCH_WARMUP", "1") == "1"
  arch_screen_warmup_epochs = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_WARMUP_EPOCHS", "1")))
  arch_screen_shard_idx = parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_INDEX", "1"))
  arch_screen_shard_cnt = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_COUNT", "1")))
  arch_screen_emit_local_ranking = get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_EMIT_LOCAL_RANKING", "1") == "1"
  stage1plus_window_mode = get(ENV, "HNODECB_STAGE1PLUS_WINDOW_MODE", "full")
  arch_screen_window_us = env_float("HNODECB_STAGE1PLUS_ARCH_WINDOW_US", 5e-6)
  arch_screen_lr = env_float("HNODECB_STAGE1PLUS_ARCH_LR", 1e-3)
  joint_trial_opt_enabled = get(ENV, "HNODECB_STAGE1PLUS_TRIAL_OPT_ENABLE", "1") == "1"
  joint_trial_epochs = max(0, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_TRIAL_EPOCHS", "10")))
  joint_trial_lr = env_float("HNODECB_STAGE1PLUS_TRIAL_LR", 1e-3)
  arch_obj_a = env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_A", 0.35)
  arch_obj_b = env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_B", 0.45)
  arch_obj_c = env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_C", 0.20)
  if arch_screen_shard_idx < 1 || arch_screen_shard_idx > arch_screen_shard_cnt
    error("HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_INDEX must be in 1..HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_COUNT")
  end

  fcontact_monitor_enabled = !isempty(monitor_eff_positions)

  if arch_screen_only
    arch_window_idx = stage1plus_select_window_indices(all_times, contact_all, stage1plus_window_mode, arch_screen_window_us; x1_signal=vec(ode_data_full[1, :]))
    ode_window = ode_data_full[:, arch_window_idx]
    times_window = all_times[arch_window_idx]
    x2dot_window = x2dot_all[arch_window_idx]
    contact_window = contact_all[arch_window_idx]
    train_idx_screen, val_idx_screen = make_train_val_masks(length(times_window), val_stride, val_offset)
    if isempty(train_idx_screen) || isempty(val_idx_screen)
      error("Stage1pluslight architecture screen window is too small to form train/val splits")
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
    arch_pairs_all = Tuple{Int, Int}[]
    for num_hidden_layers in nn_hidden_layers_range
      for num_hidden_nodes in nn_hidden_nodes_range
        push!(arch_pairs_all, (num_hidden_layers, num_hidden_nodes))
      end
    end
    chunk_size = cld(length(arch_pairs_all), arch_screen_shard_cnt)
    chunk_start = (arch_screen_shard_idx - 1) * chunk_size + 1
    chunk_stop = min(length(arch_pairs_all), arch_screen_shard_idx * chunk_size)
    assigned_arch_pairs = arch_pairs_all[chunk_start:chunk_stop]
    isempty(assigned_arch_pairs) && error("Architecture screen shard assignment is empty")
    shared_mech_draws = [stage1plus_shared_mech_draw(rep) for rep in 1:arch_screen_trials_per_arch]

    tprintln("Stage1pluslight architecture screen: ON (standalone)")
    if lowercase(strip(stage1plus_window_mode)) in ("full", "full_horizon", "full-horizon")
      tprintln("  run_seed=", stage1_run_seed,
        " | window_mode=full-horizon",
        " | points=", length(arch_window_idx),
        " | tspan=[", fmt_e(times_window[1], sigdigits=4), ", ", fmt_e(times_window[end], sigdigits=4), "]")
    elseif lowercase(strip(stage1plus_window_mode)) in ("stage2_w1", "stage2-w1")
      tprintln("  run_seed=", stage1_run_seed,
        " | window_mode=stage2lw-W1",
        " | points=", length(arch_window_idx),
        " | tspan=[", fmt_e(times_window[1], sigdigits=4), ", ", fmt_e(times_window[end], sigdigits=4), "]")
    else
      tprintln("  run_seed=", stage1_run_seed,
        " | window_mode=first-contact",
        " | window_us=", fmt_e(arch_screen_window_us, sigdigits=4),
        " | points=", length(arch_window_idx),
        " | tspan=[", fmt_e(times_window[1], sigdigits=4), ", ", fmt_e(times_window[end], sigdigits=4), "]")
    end
    tprintln("  shared_mech_draws=", arch_screen_trials_per_arch,
      " | epochs=", arch_screen_epochs,
      " | lr=", fmt_e(arch_screen_lr, sigdigits=3),
      " | obj_weights[a=", fmt_f(arch_obj_a, digits=2),
      " b=", fmt_f(arch_obj_b, digits=2),
      " c=", fmt_f(arch_obj_c, digits=2), "]")
    tprintln("  arch_screen_shard=", arch_screen_shard_idx, "/", arch_screen_shard_cnt,
      " | assigned_architectures=",
      join(["(layers=$(p[1]), nodes=$(p[2]))" for p in assigned_arch_pairs], ", "))
    tprintln("  mech sharing: identical ks/cs draw pool reused across all architectures")
    tprintln("  dummy_warmup=", arch_screen_warmup ? "ON" : "OFF",
      arch_screen_warmup ? " | epochs=" * string(arch_screen_warmup_epochs) : "")

    if arch_screen_warmup
      warmup_layers = first(nn_hidden_layers_range)
      warmup_nodes = first(nn_hidden_nodes_range)
      warmup_draw = stage1plus_shared_mech_draw(0)
      tprintln("  architecture screen warm-up: layers=", warmup_layers,
        " nodes=", warmup_nodes,
        " hidden=", 2^warmup_nodes,
        " | draw=", warmup_draw.draw_id,
        " | epochs=", arch_screen_warmup_epochs)
      try
        stage1plus_architecture_trial(
          warmup_draw.draw_id, warmup_draw.ks0, warmup_draw.cs0,
          warmup_layers, warmup_nodes,
          ode_train_screen, x2dot_train_screen, contact_train_screen, times_train_screen,
          ode_val_screen, x2dot_val_screen, contact_val_screen, times_val_screen,
          state12_scale_full, x2dot_scale_full, x3_t0_screen,
          fixed_ms_group_size, fixed_ms_continuity_term,
          zero_nn_override_stage1plus, arch_screen_warmup_epochs, arch_screen_lr
        )
      catch err
        err isa InterruptException && rethrow(err)
        tprintln("  architecture screen warm-up failed -- reason=",
          sanitize_stage1plus_reason(sprint(showerror, err)))
      end
    end

    for (num_hidden_layers, num_hidden_nodes) in assigned_arch_pairs
      tprintln("  screening arch: layers=", num_hidden_layers,
        " nodes=", num_hidden_nodes,
        " hidden=", 2^num_hidden_nodes)
      for mech_draw in shared_mech_draws
        rec = try
          stage1plus_architecture_trial(
            mech_draw.draw_id, mech_draw.ks0, mech_draw.cs0,
            num_hidden_layers, num_hidden_nodes,
            ode_train_screen, x2dot_train_screen, contact_train_screen, times_train_screen,
            ode_val_screen, x2dot_val_screen, contact_val_screen, times_val_screen,
            state12_scale_full, x2dot_scale_full, x3_t0_screen,
            fixed_ms_group_size, fixed_ms_continuity_term,
            zero_nn_override_stage1plus, arch_screen_epochs, arch_screen_lr
          )
        catch err
          err isa InterruptException && rethrow(err)
          reason = "arch_trial_exception:" * sanitize_stage1plus_reason(sprint(showerror, err))
          tprintln("      architecture trial failed -- layers=", num_hidden_layers,
            " nodes=", num_hidden_nodes,
            " draw=", mech_draw.draw_id,
            " | reason=", reason)
          make_stage1plus_architecture_failed_record(
            mech_draw.draw_id, mech_draw.ks0, mech_draw.cs0,
            num_hidden_layers, num_hidden_nodes, arch_screen_epochs;
            failure_reason=reason
          )
        end
        push!(arch_trials, rec)
        tprintln("      trial raw objectives after ", arch_screen_epochs,
          " epochs -- layers=", num_hidden_layers,
          " nodes=", num_hidden_nodes,
          " draw=", mech_draw.draw_id,
          " ks0=", fmt_e(mech_draw.ks0, sigdigits=3),
          " cs0=", fmt_e(mech_draw.cs0, sigdigits=3))
        tprintln("         detail: raw_a=", fmt_e(rec.obj1, sigdigits=4),
          " ; raw_b=", fmt_e(rec.obj2, sigdigits=4),
          " ; raw_c=", fmt_e(rec.obj3, sigdigits=4))
      end
    end

    if arch_screen_shard_cnt > 1 || !arch_screen_emit_local_ranking
      tprintln("Stage1pluslight architecture raw-result export: shard=", arch_screen_shard_idx,
        "/", arch_screen_shard_cnt,
        " | architectures=", length(assigned_arch_pairs),
        " | trials=", length(arch_trials))
      serialize(result_folder * "/" * result_name_string, (
        study=nothing,
        trial_parameters=arch_trials,
        warm_start_top=Any[],
        bounds=(ks=ks_bounds, cs=cs_bounds),
        true_values=(ks=ks_true, cs=cs_true),
        use_multiple_shooting=use_multiple_shooting,
        use_l2_regularization=use_l2_regularization,
        val_stride=val_stride,
        val_offset=val_offset,
        error_level=error_level,
        stage1_input_file="",
        stage1_input_topk=0,
        stage1plus_standalone=true,
        stage1plus_joint_random_search=true,
        arch_trials_per_arch=arch_screen_trials_per_arch,
        arch_epochs=arch_screen_epochs,
        arch_window_mode=stage1plus_window_mode,
        arch_window_us=arch_screen_window_us,
        arch_lr=arch_screen_lr,
        arch_obj_weights=(a=arch_obj_a, b=arch_obj_b, c=arch_obj_c),
        arch_norm_scope="global_across_architecture_means",
        arch_trial_logs="raw_only",
        arch_selection_metric="weighted_l2_min_obj",
        arch_pareto_axes="unweighted_norm_obj1_obj2_obj3",
        arch_mech_draw_policy="shared_across_architectures",
        arch_screen_partial=true,
        arch_screen_shard_index=arch_screen_shard_idx,
        arch_screen_shard_count=arch_screen_shard_cnt,
        run_seed=stage1_run_seed
      ))
    else
      arch_ranked = summarize_architecture_trials(arch_trials, arch_obj_a, arch_obj_b, arch_obj_c)
      selected_arch = arch_ranked[1]
      tprintln("Stage1pluslight architecture ranking (9 architectures; unified normalization across architecture means, weighted L2 min_obj selection):")
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
      tprintln("Stage1pluslight architecture winner: layers=", selected_arch.num_hidden_layers,
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
        error_level=error_level,
        stage1_input_file="",
        stage1_input_topk=0,
        stage1plus_standalone=true,
        stage1plus_joint_random_search=true,
        arch_trials_per_arch=arch_screen_trials_per_arch,
        arch_epochs=arch_screen_epochs,
        arch_window_mode=stage1plus_window_mode,
        arch_window_us=arch_screen_window_us,
        arch_lr=arch_screen_lr,
        arch_obj_weights=(a=arch_obj_a, b=arch_obj_b, c=arch_obj_c),
        arch_norm_scope="global_across_architecture_means",
        arch_trial_logs="raw_only",
        arch_selection_metric="weighted_l2_min_obj",
        arch_pareto_axes="unweighted_norm_obj1_obj2_obj3",
        arch_mech_draw_policy="shared_across_architectures",
        run_seed=stage1_run_seed
      ))
    end
  else
    fixed_num_hidden_layers = arch_selected_layers
    fixed_num_hidden_nodes = arch_selected_nodes
    stage1pluslight_ks_node_count = max(1, env_int("HNODECB_STAGE1PLUS_GRID_KS_NODES", 50))
    stage1pluslight_cs_node_count = max(1, env_int("HNODECB_STAGE1PLUS_GRID_CS_NODES", 50))
    stage1pluslight_nn_seed_bank_size = max(1, env_int("HNODECB_STAGE1PLUS_GRID_NN_SEEDS_PER_NODE", 100))
    searches_total = stage1pluslight_total_trials(
      stage1pluslight_ks_node_count,
      stage1pluslight_cs_node_count,
      stage1pluslight_nn_seed_bank_size
    )
    assignments = [i for i in 1:searches_total if ((i - shard_idx) % shard_cnt) == 0]

    main_windows = stage1plus_window_manifests(all_times, contact_all, stage1plus_window_mode, arch_screen_window_us; x1_signal=vec(ode_data_full[1, :]))
    window_bundles = map(main_windows) do win
      idxs = win.idxs
      ode_window = ode_data_full[:, idxs]
      times_window = all_times[idxs]
      x2dot_window = x2dot_all[idxs]
      contact_window = contact_all[idxs]
      train_idx, val_idx = make_train_val_masks(length(times_window), val_stride, val_offset)
      ode_train = ode_window[:, train_idx]
      ode_val = ode_window[:, val_idx]
      times_train = times_window[train_idx]
      times_val = times_window[val_idx]
      x2dot_train = x2dot_window[train_idx]
      x2dot_val = x2dot_window[val_idx]
      contact_train = contact_window[train_idx]
      contact_val = contact_window[val_idx]
      (
        role=win.role,
        ode_train=ode_train,
        ode_val=ode_val,
        times_train=times_train,
        times_val=times_val,
        x2dot_train=x2dot_train,
        x2dot_val=x2dot_val,
        contact_train=contact_train,
        contact_val=contact_val,
        x3_t0_val=ode_train[3, 1],
        len=win.len,
        t_start=win.t_start,
        t_stop=win.t_stop
      )
    end

    state12_scale_train = state12_scale_full
    x2dot_scale_train = x2dot_scale_full

    checkpoint_every = max(1, env_int("HNODECB_STAGE1PLUS_CHECKPOINT_EVERY", 100))
    resume_enabled = get(ENV, "HNODECB_STAGE1PLUS_RESUME", "1") != "0"
    shard_result_path = result_folder * "/" * result_name_string

    function save_stage1pluslight_partial(trial_parameters_local)
      stage1pluslight_write_checkpoint_atomic(shard_result_path, (
        study=nothing,
        trial_parameters=trial_parameters_local,
        warm_start_top=Any[],
        bounds=(ks=ks_bounds, cs=cs_bounds),
        true_values=(ks=ks_true, cs=cs_true),
        use_multiple_shooting=use_multiple_shooting,
        use_l2_regularization=use_l2_regularization,
        val_stride=val_stride,
        val_offset=val_offset,
        error_level=error_level,
        stage1_input_file="",
        stage1_input_topk=0,
        stage1plus_standalone=true,
        stage1plus_joint_random_search=false,
        stage1plus_joint_grid_search=true,
        stage1plus_grid_ks_nodes=stage1pluslight_ks_node_count,
        stage1plus_grid_cs_nodes=stage1pluslight_cs_node_count,
        stage1plus_grid_nn_seeds_per_node=stage1pluslight_nn_seed_bank_size,
        stage1plus_window_mode=stage1plus_window_mode,
        searches_per_candidate=searches_total,
        final_topk=final_topk,
        stage1plus_partial=true,
        stage1plus_shard_index=shard_idx,
        stage1plus_shard_count=shard_cnt,
        run_seed=stage1_run_seed
      ))
    end

    trial_parameters = Any[]
    resume_reason = ""
    if resume_enabled
      trial_parameters, resume_reason = stage1pluslight_load_resume_trials(
        shard_result_path;
        searches_total=searches_total,
        shard_idx=shard_idx,
        shard_cnt=shard_cnt,
        ks_nodes=stage1pluslight_ks_node_count,
        cs_nodes=stage1pluslight_cs_node_count,
        nn_seeds_per_node=stage1pluslight_nn_seed_bank_size,
        window_mode=stage1plus_window_mode
      )
    end
    completed_trial_ids = Set{Int}(filter!(>(0), [trial_id_or_zero(rec) for rec in trial_parameters]))
    pending_assignments = [i for i in assignments if !(i in completed_trial_ids)]
    tprintln("Stage1pluslight input: standalone joint search (no Stage1 dependency)")
    tprintln("Stage1pluslight shard ", shard_idx, "/", shard_cnt,
      " | run_seed=", stage1_run_seed,
      " | total_trials=", searches_total,
      " | local_trials=", length(assignments),
      " | pending_trials=", length(pending_assignments),
      " | final_topk=", final_topk)
    if resume_enabled
      if !isempty(trial_parameters)
        tprintln("Stage1pluslight resume: loaded ", length(trial_parameters),
          " completed trials from checkpoint",
          resume_reason == "loaded_from_backup" ? " (.bak fallback)" : "",
          " | file=", shard_result_path)
      elseif resume_reason != ""
        tprintln("Stage1pluslight resume: ignored existing checkpoint | reason=", resume_reason)
      end
    end
    if lowercase(strip(stage1plus_window_mode)) in ("full", "full_horizon", "full-horizon")
      only_window = first(main_windows)
      tprintln("Stage1pluslight window: full-horizon",
        " | points=", only_window.len,
        " | tspan=[", fmt_e(only_window.t_start, sigdigits=4), ", ", fmt_e(only_window.t_stop, sigdigits=4), "]")
    elseif lowercase(strip(stage1plus_window_mode)) in ("stage2_w123", "stage2-w123")
      tprintln("Stage1pluslight windows: stage2lw-W1&W2&W3")
      for (widx, win) in enumerate(main_windows)
        tprintln("  W", widx, " [", win.role, "]",
          " | points=", win.len,
          " | tspan=[", fmt_e(win.t_start, sigdigits=4), ", ", fmt_e(win.t_stop, sigdigits=4), "]")
      end
    elseif lowercase(strip(stage1plus_window_mode)) in ("stage2_w1", "stage2-w1")
      only_window = first(main_windows)
      tprintln("Stage1pluslight window: stage2lw-W1",
        " | points=", only_window.len,
        " | tspan=[", fmt_e(only_window.t_start, sigdigits=4), ", ", fmt_e(only_window.t_stop, sigdigits=4), "]")
    else
      only_window = first(main_windows)
      tprintln("Stage1pluslight window: first-contact + ", fmt_e(arch_screen_window_us, sigdigits=4),
        "s | points=", only_window.len,
        " | tspan=[", fmt_e(only_window.t_start, sigdigits=4), ", ", fmt_e(only_window.t_stop, sigdigits=4), "]")
    end
    tprintln("Stage1pluslight mode: joint log-grid nodes over ks/cs + fixed NN seed bank | NN monitor=",
      fcontact_monitor_enabled ? "ON" : "OFF",
      " | zero_nn_override=", zero_nn_override_stage1plus ? "ON" : "OFF",
      " | trial_opt=", joint_trial_opt_enabled ? "ON" : "OFF",
      " | fixed NN: layers=", fixed_num_hidden_layers,
      " nodes=", fixed_num_hidden_nodes,
      " | ks_nodes=", stage1pluslight_ks_node_count,
      " cs_nodes=", stage1pluslight_cs_node_count,
      " nn_seeds_per_node=", stage1pluslight_nn_seed_bank_size,
      " | trial_epochs=", joint_trial_opt_enabled ? joint_trial_epochs : 0,
      " | trial_lr=", fmt_e(joint_trial_lr, sigdigits=3))

    for global_trial_id in pending_assignments
      grid_trial = stage1pluslight_decode_grid_trial(
        global_trial_id,
        stage1pluslight_ks_node_count,
        stage1pluslight_cs_node_count,
        stage1pluslight_nn_seed_bank_size
      )
      tprintln("Stage1pluslight trial ", global_trial_id, " start | ",
        grid_trial.node_label, " | seed_bank=", grid_trial.nn_seed_bank_idx)
      flush(stdout)

      ks_fixed = grid_trial.ks0
      cs_fixed = grid_trial.cs0
      num_hidden_layers = fixed_num_hidden_layers
      num_hidden_nodes = fixed_num_hidden_nodes
      rec = try
        if lowercase(strip(stage1plus_window_mode)) in ("stage2_w123", "stage2-w123")
          stage1pluslight_joint_trial_multiwindow(
            global_trial_id, ks_fixed, cs_fixed, grid_trial.ks_node_idx, grid_trial.cs_node_idx, grid_trial.nn_seed_bank_idx,
            num_hidden_layers, num_hidden_nodes,
            window_bundles,
            state12_scale_train, x2dot_scale_train,
            fixed_ms_group_size, fixed_ms_continuity_term,
            zero_nn_override_stage1plus, joint_trial_opt_enabled, joint_trial_epochs, joint_trial_lr
          )
        else
          bundle = first(window_bundles)
          stage1pluslight_joint_trial(
            global_trial_id, ks_fixed, cs_fixed, grid_trial.ks_node_idx, grid_trial.cs_node_idx, grid_trial.nn_seed_bank_idx,
            num_hidden_layers, num_hidden_nodes,
            bundle.ode_train, bundle.x2dot_train, bundle.contact_train, bundle.times_train,
            bundle.ode_val, bundle.x2dot_val, bundle.contact_val, bundle.times_val,
            state12_scale_train, x2dot_scale_train, bundle.x3_t0_val,
            fixed_ms_group_size, fixed_ms_continuity_term,
            zero_nn_override_stage1plus, joint_trial_opt_enabled, joint_trial_epochs, joint_trial_lr;
            window_role=bundle.role
          )
        end
      catch err
        err isa InterruptException && rethrow(err)
        reason = "trial_exception:" * sanitize_stage1plus_reason(sprint(showerror, err))
        tprintln("Stage1pluslight trial ", global_trial_id,
          " -- failed and skipped | ", grid_trial.node_label,
          " | seed_bank=", grid_trial.nn_seed_bank_idx,
          " | reason=", reason)
        make_stage1pluslight_failed_record(
          global_trial_id, ks_fixed, cs_fixed,
          grid_trial.ks_node_idx, grid_trial.cs_node_idx, grid_trial.nn_seed_bank_idx,
          num_hidden_layers, num_hidden_nodes;
          window_role=(lowercase(strip(stage1plus_window_mode)) in ("stage2_w123", "stage2-w123") ? "stage2_w123" : first(window_bundles).role),
          window_roles=[bundle.role for bundle in window_bundles],
          failure_reason=reason
        )
      end
      push!(trial_parameters, rec)

      tprintln("Stage1pluslight trial ", global_trial_id,
        " -- train=", fmt_e(rec.train_loss, sigdigits=4),
        " val=", fmt_e(rec.val_loss, sigdigits=4),
        " | epochs=", rec.completed_epochs,
        rec.early_stopped ? " | early_stop=" * rec.early_stop_reason : "")
      if !rec.is_viable
        local_train_reason = rec.train_reason == "" ? "unknown" : rec.train_reason
        local_val_reason = rec.val_reason == "" ? "unknown" : rec.val_reason
        tprintln("  filtered: viable=false | train=", local_train_reason, " | val=", local_val_reason)
      end
      if rec.val_parts !== nothing
        tprintln("  parts: state=", fmt_e(rec.val_parts.state, sigdigits=3),
          " x2dot=", fmt_e(rec.val_parts.x2dot, sigdigits=3),
          " x3r=", fmt_e(rec.val_parts.x3_range, sigdigits=3),
          " ftsr=", fmt_e(rec.val_parts.fts_range, sigdigits=3),
          " cont=", fmt_e(rec.val_parts.cont, sigdigits=3))
      end
      tprintln("  mech init: ks0=", fmt_e(ks_fixed, sigdigits=3),
        " cs0=", fmt_e(cs_fixed, sigdigits=3),
        " | node=", grid_trial.node_label,
        " | seed_bank=", grid_trial.nn_seed_bank_idx)
      tprintln("  mech: ks=", fmt_e(rec.ks_hat, sigdigits=3),
        " (err=", fmt_f(rec.ks_err_pct, digits=2), "%)",
        " cs=", fmt_e(rec.cs_hat, sigdigits=3),
        " (err=", fmt_f(rec.cs_err_pct, digits=2), "%)")
      if rec.val_parts !== nothing
        tprintln("  rec: x1=", fmt_f(rec.val_parts.x1_rec, digits=2), "% x3=",
          fmt_f(rec.val_parts.x3_rec, digits=2), "%")
      end
      if fcontact_monitor_enabled
        tprintln("  nn: F_contact err=", fmt_f(rec.val_nn_err, digits=2), "%")
        tprintln("  nn raw: min=", fmt_e(rec.val_raw_min, sigdigits=3),
          " max=", fmt_e(rec.val_raw_max, sigdigits=3),
          " mean=", fmt_e(rec.val_raw_mean, sigdigits=3),
          " neg=", fmt_f(100 * rec.val_raw_neg_frac, digits=2), "%")
        tprintln("  nn contact-force: min=", fmt_e(rec.val_fnet_min, sigdigits=3),
          " max=", fmt_e(rec.val_fnet_max, sigdigits=3),
          " mean=", fmt_e(rec.val_fnet_mean, sigdigits=3),
          " p95|F|=", fmt_e(rec.val_fnet_abs_p95, sigdigits=3))
      end
      if checkpoint_every > 0 && (length(trial_parameters) % checkpoint_every == 0)
        save_stage1pluslight_partial(trial_parameters)
        tprintln("Stage1pluslight checkpoint saved: shard ", shard_idx, "/", shard_cnt,
          " | completed=", length(trial_parameters), "/", length(assignments),
          " | file=", shard_result_path)
      end
    end

    if shard_cnt > 1 || !stage1plus_emit_local_selection
      viable_trials = [rec for rec in trial_parameters if hasproperty(rec, :is_viable) && rec.is_viable]
      tprintln("Stage1pluslight raw-result export: shard=", shard_idx, "/", shard_cnt,
        " | trials=", length(trial_parameters),
        " | viable=", length(viable_trials))
      save_stage1pluslight_partial(trial_parameters)
    else
      viable_trials = [rec for rec in trial_parameters if hasproperty(rec, :is_viable) && rec.is_viable]
      selection_pool = viable_trials
      sorted = sort(selection_pool, by = r -> r.loss)
      selected = final_topk > 0 ? sorted[1:min(final_topk, length(sorted))] : Any[]
      warm_start_top = final_topk > 0 ? [(
          trial_id=rec.params["trial_id"],
          params=rec.params,
          p_net=copy(rec.p_net_vec),
          train_loss=rec.train_loss,
          val_loss=rec.val_loss
        ) for rec in selected] : Any[]
      local_warm_has_pnet = !isempty(warm_start_top) && hasproperty(warm_start_top[1], :p_net)

      tprintln("Stage1pluslight done. Viable=", length(viable_trials), "/", length(trial_parameters),
        " | selected=", length(selected))
      if final_topk > 0
        if !isempty(selected)
          tprintln("Stage1pluslight best val loss=", fmt_e(selected[1].val_loss, sigdigits=4),
            " | train=", fmt_e(selected[1].train_loss, sigdigits=4))
        end
        tprintln("Stage1pluslight top-", length(selected), " summary (finite/stable filtered):")
        for (i, rec) in enumerate(selected)
          tprintln("  Rank ", i,
            " -- train=", fmt_e(rec.train_loss, sigdigits=4),
            " val=", fmt_e(rec.val_loss, sigdigits=4),
            " | trial=", rec.params["trial_id"],
            " | ks0=", fmt_e(rec.params["ks0"], sigdigits=3),
            " cs0=", fmt_e(rec.params["cs0"], sigdigits=3))
          tprintln("     val parts: state=", fmt_e(rec.val_parts.state, sigdigits=3),
            " x2dot=", fmt_e(rec.val_parts.x2dot, sigdigits=3),
            " x3r=", fmt_e(rec.val_parts.x3_range, sigdigits=3),
            " ftsr=", fmt_e(rec.val_parts.fts_range, sigdigits=3),
            " cont=", fmt_e(rec.val_parts.cont, sigdigits=3))
          tprintln("     rec: x1=", fmt_f(rec.val_parts.x1_rec, digits=2),
            "% x3=", fmt_f(rec.val_parts.x3_rec, digits=2), "%")
          tprintln("     mech: ks=", fmt_e(rec.ks_hat, sigdigits=3),
            " (err=", fmt_f(rec.ks_err_pct, digits=2), "%)",
            " cs=", fmt_e(rec.cs_hat, sigdigits=3),
            " (err=", fmt_f(rec.cs_err_pct, digits=2), "%)")
          tprintln("     nn: F_contact err=", fmt_f(rec.val_nn_err, digits=2), "%")
          tprintln("     nn raw: min=", fmt_e(rec.val_raw_min, sigdigits=3),
            " max=", fmt_e(rec.val_raw_max, sigdigits=3),
            " mean=", fmt_e(rec.val_raw_mean, sigdigits=3),
            " neg=", fmt_f(100 * rec.val_raw_neg_frac, digits=2), "%")
          tprintln("     nn contact-force: min=", fmt_e(hasproperty(rec, :val_fnet_min) ? rec.val_fnet_min : NaN, sigdigits=3),
            " max=", fmt_e(hasproperty(rec, :val_fnet_max) ? rec.val_fnet_max : NaN, sigdigits=3),
            " mean=", fmt_e(hasproperty(rec, :val_fnet_mean) ? rec.val_fnet_mean : NaN, sigdigits=3),
            " p95|F|=", fmt_e(hasproperty(rec, :val_fnet_abs_p95) ? rec.val_fnet_abs_p95 : NaN, sigdigits=3))
        end
      else
        tprintln("Stage1pluslight final Top-K selection: DISABLED")
      end
      tprintln("Stage1pluslight warm-start export (shard): ",
        final_topk > 0 ? "ON" : "OFF",
        " | saved=", length(warm_start_top),
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
        error_level=error_level,
        stage1_input_file="",
        stage1_input_topk=0,
        stage1plus_standalone=true,
        stage1plus_joint_random_search=false,
        stage1plus_joint_grid_search=true,
        stage1plus_grid_ks_nodes=stage1pluslight_ks_node_count,
        stage1plus_grid_cs_nodes=stage1pluslight_cs_node_count,
        stage1plus_grid_nn_seeds_per_node=stage1pluslight_nn_seed_bank_size,
        searches_per_candidate=searches_total,
        final_topk=final_topk,
        run_seed=stage1_run_seed
      ))
    end
  end
  end
elseif use_stage1plus && !use_stage1pluslight
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
  stage1plus_emit_local_selection = get(ENV, "HNODECB_STAGE1PLUS_EMIT_LOCAL_SELECTION", "1") == "1"
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
  arch_screen_trials_per_arch = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH", "20")))
  arch_screen_epochs = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_EPOCHS", "50")))
  arch_screen_warmup = get(ENV, "HNODECB_STAGE1PLUS_ARCH_WARMUP", "1") == "1"
  arch_screen_warmup_epochs = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_WARMUP_EPOCHS", "1")))
  arch_screen_shard_idx = parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_INDEX", "1"))
  arch_screen_shard_cnt = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_COUNT", "1")))
  arch_screen_emit_local_ranking = get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_EMIT_LOCAL_RANKING", "1") == "1"
  arch_screen_window_us = env_float("HNODECB_STAGE1PLUS_ARCH_WINDOW_US", 5e-6)
  arch_screen_lr = env_float("HNODECB_STAGE1PLUS_ARCH_LR", 1e-3)
  arch_obj_a = env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_A", 0.35)
  arch_obj_b = env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_B", 0.45)
  arch_obj_c = env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_C", 0.20)
  if arch_screen_shard_idx < 1 || arch_screen_shard_idx > arch_screen_shard_cnt
    error("HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_INDEX must be in 1..HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_COUNT")
  end
  fcontact_monitor_enabled = !isempty(monitor_eff_positions)
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
    arch_window_idx = stage1plus_select_window_indices(all_times, contact_all, stage1plus_window_mode, arch_screen_window_us; x1_signal=vec(ode_data_full[1, :]))
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
    arch_pairs_all = Tuple{Int, Int}[]
    for num_hidden_layers in nn_hidden_layers_range
      for num_hidden_nodes in nn_hidden_nodes_range
        push!(arch_pairs_all, (num_hidden_layers, num_hidden_nodes))
      end
    end
    chunk_size = cld(length(arch_pairs_all), arch_screen_shard_cnt)
    chunk_start = (arch_screen_shard_idx - 1) * chunk_size + 1
    chunk_stop = min(length(arch_pairs_all), arch_screen_shard_idx * chunk_size)
    assigned_arch_pairs = arch_pairs_all[chunk_start:chunk_stop]
    isempty(assigned_arch_pairs) && error("Architecture screen shard assignment is empty")

    tprintln("Stage1plus architecture screen: ON")
    if lowercase(strip(stage1plus_window_mode)) in ("full", "full_horizon", "full-horizon")
      tprintln("  window_mode=full-horizon",
        " | points=", length(arch_window_idx),
        " | tspan=[", fmt_e(times_window[1], sigdigits=4), ", ", fmt_e(times_window[end], sigdigits=4), "]")
    elseif lowercase(strip(stage1plus_window_mode)) in ("stage2_w1", "stage2-w1")
      tprintln("  window_mode=stage2lw-W1",
        " | points=", length(arch_window_idx),
        " | tspan=[", fmt_e(times_window[1], sigdigits=4), ", ", fmt_e(times_window[end], sigdigits=4), "]")
    else
      tprintln("  window_mode=first-contact",
        " | window_us=", fmt_e(arch_screen_window_us, sigdigits=4),
        " | points=", length(arch_window_idx),
        " | tspan=[", fmt_e(times_window[1], sigdigits=4), ", ", fmt_e(times_window[end], sigdigits=4), "]")
    end
    tprintln("  trials_per_arch=", arch_screen_trials_per_arch,
      " | epochs=", arch_screen_epochs,
      " | lr=", fmt_e(arch_screen_lr, sigdigits=3),
      " | obj_weights[a=", fmt_f(arch_obj_a, digits=2),
      " b=", fmt_f(arch_obj_b, digits=2),
      " c=", fmt_f(arch_obj_c, digits=2), "]")
    tprintln("  arch_screen_shard=", arch_screen_shard_idx, "/", arch_screen_shard_cnt,
      " | assigned_architectures=",
      join(["(layers=$(p[1]), nodes=$(p[2]))" for p in assigned_arch_pairs], ", "))
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
      try
        stage1plus_architecture_trial(
          warmup_base_rec, warmup_base_rank, 0,
          warmup_layers, warmup_nodes,
          ode_train_screen, x2dot_train_screen, contact_train_screen, times_train_screen,
          ode_val_screen, x2dot_val_screen, contact_val_screen, times_val_screen,
          state12_scale_full, x2dot_scale_full, x3_t0_screen,
          fixed_ms_group_size, fixed_ms_continuity_term,
          zero_nn_override_stage1plus, arch_screen_warmup_epochs, arch_screen_lr
        )
      catch err
        err isa InterruptException && rethrow(err)
        tprintln("  architecture screen warm-up failed -- base=", warmup_base_rank,
          " | reason=", sanitize_stage1plus_reason(sprint(showerror, err)))
      end
    end

    for (num_hidden_layers, num_hidden_nodes) in assigned_arch_pairs
      tprintln("  screening arch: layers=", num_hidden_layers,
        " nodes=", num_hidden_nodes,
        " hidden=", 2^num_hidden_nodes)
      for base_rank in base_rank_indices
        base_rec = base_candidates[base_rank]
        base_params = hasproperty(base_rec, :params) ? base_rec.params : Dict{Any, Any}()
        base_ks = hasproperty(base_rec, :ks_hat) ? base_rec.ks_hat :
          (haskey(base_params, "ks0") ? base_params["ks0"] : NaN)
        base_cs = hasproperty(base_rec, :cs_hat) ? base_rec.cs_hat :
          (haskey(base_params, "cs0") ? base_params["cs0"] : NaN)
        for rep in 1:arch_screen_trials_per_arch
          rec = try
            stage1plus_architecture_trial(
              base_rec, base_rank, rep,
              num_hidden_layers, num_hidden_nodes,
              ode_train_screen, x2dot_train_screen, contact_train_screen, times_train_screen,
              ode_val_screen, x2dot_val_screen, contact_val_screen, times_val_screen,
              state12_scale_full, x2dot_scale_full, x3_t0_screen,
              fixed_ms_group_size, fixed_ms_continuity_term,
              zero_nn_override_stage1plus, arch_screen_epochs, arch_screen_lr
            )
          catch err
            err isa InterruptException && rethrow(err)
            reason = "arch_trial_exception:" * sanitize_stage1plus_reason(sprint(showerror, err))
            tprintln("      architecture trial failed -- layers=", num_hidden_layers,
              " nodes=", num_hidden_nodes,
              " base=", base_rank,
              " rep=", rep,
              " | reason=", reason)
            make_stage1plus_architecture_failed_record(
              1_000_000 * base_rank + rep, base_ks, base_cs,
              num_hidden_layers, num_hidden_nodes, arch_screen_epochs;
              failure_reason=reason
            )
          end
          push!(arch_trials, rec)
          tprintln("      trial raw objectives after ", arch_screen_epochs,
            " epochs -- layers=", num_hidden_layers,
            " nodes=", num_hidden_nodes,
            " base=", base_rank,
            " rep=", rep)
          tprintln("         detail: raw_a=", fmt_e(rec.obj1, sigdigits=4),
            " ; raw_b=", fmt_e(rec.obj2, sigdigits=4),
            " ; raw_c=", fmt_e(rec.obj3, sigdigits=4))
        end
      end
    end

    if arch_screen_shard_cnt > 1 || !arch_screen_emit_local_ranking
      tprintln("Stage1plus architecture raw-result export: shard=", arch_screen_shard_idx,
        "/", arch_screen_shard_cnt,
        " | architectures=", length(assigned_arch_pairs),
        " | trials=", length(arch_trials))
      serialize(result_folder * "/" * result_name_string, (
        study=nothing,
        trial_parameters=arch_trials,
        warm_start_top=Any[],
        bounds=(ks=ks_bounds, cs=cs_bounds),
        true_values=(ks=ks_true, cs=cs_true),
        use_multiple_shooting=use_multiple_shooting,
        use_l2_regularization=use_l2_regularization,
        val_stride=val_stride,
        val_offset=val_offset,
        error_level=error_level,
        stage1_input_file=stage1_input_file,
        stage1_input_topk=stage1plus_input_topk,
        arch_trials_per_arch=arch_screen_trials_per_arch,
        arch_epochs=arch_screen_epochs,
        arch_window_mode=stage1plus_window_mode,
        arch_window_us=arch_screen_window_us,
        arch_lr=arch_screen_lr,
        arch_obj_weights=(a=arch_obj_a, b=arch_obj_b, c=arch_obj_c),
        arch_norm_scope="global_across_architecture_means",
        arch_trial_logs="raw_only",
        arch_selection_metric="weighted_l2_min_obj",
        arch_pareto_axes="unweighted_norm_obj1_obj2_obj3",
        arch_screen_partial=true,
        arch_screen_shard_index=arch_screen_shard_idx,
        arch_screen_shard_count=arch_screen_shard_cnt
      ))
    else
      arch_ranked = summarize_architecture_trials(arch_trials, arch_obj_a, arch_obj_b, arch_obj_c)
      selected_arch = arch_ranked[1]
      tprintln("Stage1plus architecture ranking (9 architectures; unified normalization across architecture means, weighted L2 min_obj selection):")
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
        error_level=error_level,
        stage1_input_file=stage1_input_file,
        stage1_input_topk=stage1plus_input_topk,
        arch_trials_per_arch=arch_screen_trials_per_arch,
        arch_epochs=arch_screen_epochs,
        arch_window_mode=stage1plus_window_mode,
        arch_window_us=arch_screen_window_us,
        arch_lr=arch_screen_lr,
        arch_obj_weights=(a=arch_obj_a, b=arch_obj_b, c=arch_obj_c),
        arch_norm_scope="global_across_architecture_means",
        arch_trial_logs="raw_only",
        arch_selection_metric="weighted_l2_min_obj",
        arch_pareto_axes="unweighted_norm_obj1_obj2_obj3"
      ))
    end
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
    fcontact_monitor_enabled ? "ON" : "OFF",
    " | zero_nn_override=", zero_nn_override_stage1plus ? "ON" : "OFF",
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
      0.0, re_pnet; zero_nn_override=zero_nn_override_stage1plus)
    train_reason = last_inf_reason[]

    inf_context[] = "val_eval"
    last_inf_reason[] = ""
    val_loss, val_diag = loss_single_or_ms(θ0, ode_val, x2dot_val, contact_val, times_val,
      state12_scale_full, x2dot_scale_full, x3_scale,
      use_multiple_shooting, fixed_ms_group_size, fixed_ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val,
      0.0, re_pnet; zero_nn_override=zero_nn_override_stage1plus)
    val_reason = last_inf_reason[]

    ks_hat = bound_param(θ0.mech_raw[1], ks_bounds[1], ks_bounds[2])
    cs_hat = bound_param(θ0.mech_raw[2], cs_bounds[1], cs_bounds[2])
    ks_err_pct = rel_err_pct(ks_hat, ks_true, scale_eps)
    cs_err_pct = rel_err_pct(cs_hat, cs_true, scale_eps)
    val_nn_metrics = Ref((
      fcontact_err=NaN,
      raw_min=NaN,
      raw_max=NaN,
      raw_mean=NaN,
      raw_neg_frac=NaN,
      fnet_min=NaN,
      fnet_max=NaN,
      fnet_mean=NaN,
      fnet_abs_p95=NaN
    ))
    if fcontact_monitor_enabled
      Zygote.ignore() do
        p_net_struct = re_pnet(θ0.p_net)
        val_nn_metrics[] = contact_net_monitor_metrics(
          ode_data_full, monitor_idx_full, p_net_struct,
          approximating_neural_network, st;
          contact_true_ref=monitor_contact_true, eff_pos_ref=monitor_eff_positions
        )
      end
    end
    is_viable = isfinite(train_loss) && isfinite(val_loss) &&
      (!fcontact_monitor_enabled || isfinite(val_nn_metrics[].fcontact_err))

    params = Dict{Any, Any}(
      "trial_id" => global_trial_id,
      "base_rank" => base_rank,
      "base_trial_id" => base_trial_id,
      "ks0" => ks_fixed,
      "cs0" => cs_fixed,
      "nn_force_mode" => "contact_net",
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
      val_nn_err=val_nn_metrics[].fcontact_err,
      val_raw_min=val_nn_metrics[].raw_min,
      val_raw_max=val_nn_metrics[].raw_max,
      val_raw_mean=val_nn_metrics[].raw_mean,
      val_raw_neg_frac=val_nn_metrics[].raw_neg_frac,
      val_fnet_min=val_nn_metrics[].fnet_min,
      val_fnet_max=val_nn_metrics[].fnet_max,
      val_fnet_mean=val_nn_metrics[].fnet_mean,
      val_fnet_abs_p95=val_nn_metrics[].fnet_abs_p95,
      is_viable=is_viable,
      train_reason=train_reason,
      val_reason=val_reason,
      p_net_vec=copy(p_net_vec),
      nn_force_mode="contact_net",
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
        " ftsr=", fmt_e(diag.fts_range, sigdigits=3),
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
    if fcontact_monitor_enabled
      nn_metrics = val_nn_metrics[]
      tprintln("  nn: F_contact err=", fmt_f(nn_metrics.fcontact_err, digits=2), "%")
      tprintln("  nn raw: min=", fmt_e(nn_metrics.raw_min, sigdigits=3),
        " max=", fmt_e(nn_metrics.raw_max, sigdigits=3),
        " mean=", fmt_e(nn_metrics.raw_mean, sigdigits=3),
        " neg=", fmt_f(100 * nn_metrics.raw_neg_frac, digits=2), "%")
      tprintln("  nn contact-force: min=", fmt_e(nn_metrics.fnet_min, sigdigits=3),
        " max=", fmt_e(nn_metrics.fnet_max, sigdigits=3),
        " mean=", fmt_e(nn_metrics.fnet_mean, sigdigits=3),
        " p95|F|=", fmt_e(nn_metrics.fnet_abs_p95, sigdigits=3))
    end
  end

  if shard_cnt > 1 || !stage1plus_emit_local_selection
    viable_trials = [rec for rec in trial_parameters if hasproperty(rec, :is_viable) && rec.is_viable]
    tprintln("Stage1plus raw-result export: shard=", shard_idx, "/", shard_cnt,
      " | trials=", length(trial_parameters),
      " | viable=", length(viable_trials))
    serialize(result_folder * "/" * result_name_string, (
      study=nothing,
      trial_parameters=trial_parameters,
      warm_start_top=Any[],
      bounds=(ks=ks_bounds, cs=cs_bounds),
      true_values=(ks=ks_true, cs=cs_true),
      use_multiple_shooting=use_multiple_shooting,
      use_l2_regularization=use_l2_regularization,
      val_stride=val_stride,
      val_offset=val_offset,
      error_level=error_level,
      stage1_input_file=stage1_input_file,
      stage1_input_topk=stage1plus_input_topk,
      searches_per_candidate=searches_per_candidate,
      final_topk=final_topk,
      stage1plus_partial=true,
      stage1plus_shard_index=shard_idx,
      stage1plus_shard_count=shard_cnt
    ))
  else
    viable_trials = [rec for rec in trial_parameters if hasproperty(rec, :is_viable) && rec.is_viable]
    selection_pool = isempty(viable_trials) ? trial_parameters : viable_trials
    sorted = sort(selection_pool, by = r -> r.loss)
    selected = sorted[1:min(final_topk, length(sorted))]
    warm_start_top = [(
        trial_id=rec.params["trial_id"],
        params=rec.params,
        p_net=copy(rec.p_net_vec),
        train_loss=rec.train_loss,
        val_loss=rec.val_loss
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
        " ftsr=", fmt_e(rec.val_parts.fts_range, sigdigits=3),
        " cont=", fmt_e(rec.val_parts.cont, sigdigits=3))
      tprintln("     rec: x1=", fmt_f(rec.val_parts.x1_rec, digits=2),
        "% x3=", fmt_f(rec.val_parts.x3_rec, digits=2), "%")
      tprintln("     mech: ks=", fmt_e(rec.ks_hat, sigdigits=3),
        " (err=", fmt_f(rec.ks_err_pct, digits=2), "%)",
        " cs=", fmt_e(rec.cs_hat, sigdigits=3),
        " (err=", fmt_f(rec.cs_err_pct, digits=2), "%)")
      tprintln("     nn: F_contact err=", fmt_f(rec.val_nn_err, digits=2), "%")
      tprintln("     nn raw: min=", fmt_e(rec.val_raw_min, sigdigits=3),
        " max=", fmt_e(rec.val_raw_max, sigdigits=3),
        " mean=", fmt_e(rec.val_raw_mean, sigdigits=3),
        " neg=", fmt_f(100 * rec.val_raw_neg_frac, digits=2), "%")
      tprintln("     nn contact-force: min=", fmt_e(hasproperty(rec, :val_fnet_min) ? rec.val_fnet_min : NaN, sigdigits=3),
        " max=", fmt_e(hasproperty(rec, :val_fnet_max) ? rec.val_fnet_max : NaN, sigdigits=3),
        " mean=", fmt_e(hasproperty(rec, :val_fnet_mean) ? rec.val_fnet_mean : NaN, sigdigits=3),
        " p95|F|=", fmt_e(hasproperty(rec, :val_fnet_abs_p95) ? rec.val_fnet_abs_p95 : NaN, sigdigits=3))
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
      error_level=error_level,
      stage1_input_file=stage1_input_file,
      stage1_input_topk=stage1plus_input_topk,
      searches_per_candidate=searches_per_candidate,
      final_topk=final_topk
    ))
  end
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
  fcontact_monitor_enabled = false

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
  tprintln("Stage1 no-ADAM run_seed=", stage1_run_seed, " | sampling=per_run_randomized")
  tprintln("Stage1 no-ADAM mode: frozen zero NN (layers=", fixed_num_hidden_layers,
    ", nodes=", fixed_num_hidden_nodes, ") | NN monitor=OFF | zero-Hertz override=ON")
  tprintln("Stage1 warm-start capture (shard): ENABLED | local_top_k=", top_k)
  for i in trial_indices
    tprintln("Stage1 trial ", i, " start (no-ADAM)")
    flush(stdout)
    # Per-trial RNG keeps mechanistic sampling reproducible and avoids cross-shard duplicates.
    rng_trial = StableRNG(mix_stage1_trial_seed(stage1_run_seed, i))
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
        " ftsr=", fmt_e(diag.fts_range, sigdigits=3),
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
    if fcontact_monitor_enabled
      Zygote.ignore() do
        p_net_struct = re_pnet(θ0.p_net)
        nn_metrics = contact_net_monitor_metrics(
          ode_data_full, monitor_idx_full, p_net_struct,
          approximating_neural_network, st;
          contact_true_ref=monitor_contact_true, eff_pos_ref=monitor_eff_positions
        )
        tprintln("  nn: F_contact err=", fmt_f(nn_metrics.fcontact_err, digits=2), "%")
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
      " ftsr=", fmt_e(rec.val_parts.fts_range, sigdigits=3),
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
    error_level=error_level
  ))
  end
end
end
end
