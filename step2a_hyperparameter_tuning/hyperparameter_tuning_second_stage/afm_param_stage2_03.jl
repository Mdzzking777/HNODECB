#=
Stage 2 (grid) hyperparameter tuning for AFM DMT-KV.
Scenario 03: x3 unobserved, Hertz force replaced by a neural network.
Grid search on L2 regularization weight using best hyperparameters from Stage 1.
=#

cd(@__DIR__)

using Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Statistics, Printf, Dates
using ComponentArrays, SciMLSensitivity, StableRNGs
using Zygote
using Optimisers
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

include("../../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_functions.jl")
include("../../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")

result_folder = "results_afm"
if !isdir(result_folder)
  mkdir(result_folder)
end
stage2_result_basename = get(ENV, "HNODECB_STAGE2_RESULT_BASENAME", "afm_param_stage2_03.jld")
stage2_result_root, stage2_result_ext = splitext(stage2_result_basename)
if stage2_result_ext == ""
  stage2_result_ext = ".jld"
  stage2_result_basename *= stage2_result_ext
  stage2_result_root = splitext(stage2_result_basename)[1]
end
result_name_string = stage2_result_basename
stage2_shard_index = parse(Int, get(ENV, "HNODECB_STAGE2_SHARD_INDEX", "1"))
stage2_shard_count = parse(Int, get(ENV, "HNODECB_STAGE2_SHARD_COUNT", "1"))
stage2_input_topk = max(1, parse(Int, get(ENV, "HNODECB_STAGE2_INPUT_TOPK", "9")))
stage2_final_topk = max(1, parse(Int, get(ENV, "HNODECB_STAGE2_FINAL_TOPK", "3")))
stage2_candidate_filter_spec = strip(get(ENV, "HNODECB_STAGE2_CANDIDATE_INDICES", ""))
stage2_nn_warm_enabled = get(ENV, "HNODECB_STAGE2_NN_WARM_ENABLED", "0") == "1"
stage2_nn_warm_basename = strip(get(ENV, "HNODECB_STAGE2_NN_WARM_INPUT_BASENAME", ""))
if stage2_nn_warm_basename == ""
  stage2_nn_warm_basename = "afm_param_stage1pluslight_03.jld"
end
stage2_nn_warm_rank = begin
  v = tryparse(Int, strip(get(ENV, "HNODECB_STAGE2_NN_WARM_RANK", "1")))
  (v === nothing || v < 1) ? 1 : v
end
stage2_use_gnn = get(ENV, "HNODECB_STAGE2_USE_GNN", "0") == "1"
if stage2_shard_index < 1 || stage2_shard_index > stage2_shard_count
  error("HNODECB_STAGE2_SHARD_INDEX must be in 1..HNODECB_STAGE2_SHARD_COUNT")
end
if stage2_shard_count > 1
  result_name_string = stage2_result_root * "_p" * string(stage2_shard_index) * stage2_result_ext
end

error_level = "e0.0"

ks_true = ks
cs_true = cs

function tprintln(args...)
  ts = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
  println("[", ts, "] ", args...)
  flush(stdout)
end

function env_int(name, default)
  raw = strip(get(ENV, name, ""))
  parsed = tryparse(Int, raw)
  return parsed === nothing ? default : parsed
end

# Load stage1 results
stage2_input_basename = get(ENV, "HNODECB_STAGE2_INPUT_BASENAME", "afm_param_stage1_03.jld")
stage1 = deserialize("../hyperparameter_tuning_first_stage/results_afm/" * stage2_input_basename)

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

stage2_window_start = env_int("HNODECB_STAGE2_WINDOW_START", 1)
stage2_window_len = env_int("HNODECB_STAGE2_WINDOW_LEN", 0)
stage2_window_active = stage2_window_len > 0
stage2_window_full_count = nrow(solution_dataframe_full)
stage2_window_start_used = 1
stage2_window_stop_used = stage2_window_full_count
if stage2_window_active
  stage2_window_start_used = clamp(stage2_window_start, 1, stage2_window_full_count)
  stage2_window_len_used = clamp(stage2_window_len, 1, stage2_window_full_count - stage2_window_start_used + 1)
  stage2_window_stop_used = stage2_window_start_used + stage2_window_len_used - 1
  solution_dataframe_full = solution_dataframe_full[stage2_window_start_used:stage2_window_stop_used, :]
  ode_data_full = ode_data_full[:, stage2_window_start_used:stage2_window_stop_used]
end

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
if stage2_window_active
  tprintln("Stage2 windowed horizon: full_points=", stage2_window_full_count,
    " | window=[", stage2_window_start_used, ", ", stage2_window_stop_used, "]",
    " len=", length(all_times),
    " | tspan=[", @sprintf("%.6e", all_times[1]), ", ", @sprintf("%.6e", all_times[end]), "]")
end

function nn_input_from_state(u)
  return u[1:3]
end

function nn_input_from_values(x1, x2, x3)
  return [x1, x2, x3]
end

# Multiple shooting toggle (keep on to match baseline)
use_multiple_shooting = false
selftest = get(ENV, "HNODECB_SELFTEST", "0") == "1"
stage2_sanity = get(ENV, "HNODECB_STAGE2_SANITY", "1") == "1"

if selftest
  tprintln("=== AFM Stage2 (03) SELFTEST ===")
else
  tprintln("=== AFM Stage2 (03) ===")
end
tprintln("Switches: MS=", use_multiple_shooting, " | Sanity=", stage2_sanity, " | L2-grid pending...")

# Solver settings
solver_internal_ad = get(ENV, "HNODECB_SOLVER_INTERNAL_AD", "0") == "1"
integrator = Rosenbrock23(autodiff=solver_internal_ad)
abstol = 1e-8
reltol = 1e-8
sensealg = GaussAdjoint(autojacvec=ZygoteVJP())
ode_maxiters = parse(Int, get(ENV, "HNODECB_ODE_MAXITERS", "1000000"))
tprintln("Solver: Rosenbrock23(autodiff=", solver_internal_ad, ") | rhs=out-of-place | sensealg=GaussAdjoint(ZygoteVJP())")

# Parameter bounds (mechanistic unknowns)
ks_bounds = stage1.bounds.ks
cs_bounds = stage1.bounds.cs

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

env_float(name, default) = tryparse(Float64, get(ENV, name, "")) !== nothing ?
  parse(Float64, get(ENV, name, "")) : default

# Optional override for x3-range prior weight (used by stage2light to disable x3r).
x3_range_weight = env_float("HNODECB_STAGE2_X3R_WEIGHT", x3_range_weight)
stage2_gnn_default = env_float("HNODECB_STAGE2_GNN_DEFAULT", 1.0)
tprintln("Stage2 g_nn mode: ", stage2_use_gnn ? "ON" : "OFF", " | default=", stage2_gnn_default)

function relative_rmse_pct(err_sum, truth_sum, count, eps)
  denom = max(sqrt(truth_sum / max(count, 1)), eps)
  return 100 * sqrt(err_sum / max(count, 1)) / denom
end

function rel_err_pct(est, truth, eps)
  return 100 * abs(est - truth) / max(abs(truth), eps)
end

function fmt_e(x; sigdigits=4)
  if x isa Number
    return isfinite(x) ? string(round(x, sigdigits=sigdigits)) : string(x)
  end
  return string(x)
end

fmt_hp(x) = (x isa Number && isfinite(x)) ? @sprintf("%.16e", x) : string(x)

function fmt_f(x; digits=2)
  if x isa Number
    return isfinite(x) ? string(round(x, digits=digits)) : string(x)
  end
  return string(x)
end

function fmt_loss_part(x, enabled; sigdigits=3)
  return enabled ? fmt_e(x, sigdigits=sigdigits) : "None"
end

function grad_norm_safe(g)
  if g === nothing
    return NaN
  end
  # ComponentVector / NamedTuple / Array
  try
    return sqrt(sum(abs2, g))
  catch
    try
      return sqrt(sum(abs2, values(g)))
    catch
      return NaN
    end
  end
end

function make_train_val_masks(n, val_stride, val_offset)
  val_idx = [i for i in 1:n if (i - val_offset) % val_stride == 0]
  train_idx = [i for i in 1:n if !(i in val_idx)]
  return sort(train_idx), sort(val_idx)
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

function fhertz_monitor_metrics(u_mat, idxs, p_net_struct, appr, st; fh_true_ref=nothing, eff_pos_ref=nothing, nn_gain::Float64=1.0)
  fh_true = fh_true_ref === nothing ? fhertz_true_from_states(u_mat, idxs) : fh_true_ref
  fh_pred, raw = fhertz_pred_and_raw_from_states(u_mat, idxs, p_net_struct, appr, st; nn_gain=nn_gain)
  eff_pos = eff_pos_ref === nothing ? effective_contact_positions(idxs, fh_true) : eff_pos_ref
  if isempty(eff_pos)
    return (fhertz_err=NaN, raw_min=NaN, raw_max=NaN, raw_span=NaN, raw_mean=NaN, raw_neg_frac=NaN)
  end
  raw_eff = raw[eff_pos]
  raw_min = minimum(raw_eff)
  raw_max = maximum(raw_eff)
  err_sum = sum(abs2, fh_pred[eff_pos] .- fh_true[eff_pos])
  truth_sum = sum(abs2, fh_true[eff_pos])
  fhertz_err = truth_sum <= 0.0 ? NaN : relative_rmse_pct(err_sum, truth_sum, length(eff_pos), 0.0)
  raw_neg_frac = count(<(0.0), raw_eff) / length(raw_eff)
  return (
    fhertz_err=fhertz_err,
    raw_min=raw_min,
    raw_max=raw_max,
    raw_span=raw_max - raw_min,
    raw_mean=mean(raw_eff),
    raw_neg_frac=raw_neg_frac
  )
end

monitor_fhertz_true = fhertz_true_from_states(ode_data_full, monitor_idx_full)
monitor_eff_positions = effective_contact_positions(monitor_idx_full, monitor_fhertz_true)

function make_uode_func(appr, st, known_pars; nn_gain::Float64=1.0)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  f(du, u, p, t) =
    let appr = appr, st = st, k = k, wd = wd, m = m, c = c, Fd = Fd, R = R, dist = dist, Fad = Fad
      ks = p.mech[1]
      cs = p.mech[2]

      s = dist + u[1] - u[3]
      delta = softplus(-s, adhesion_transition)
      delta = ifelse(delta > 0.0, delta, 0.0)
      w_pred = contact_weight(s, adhesion_transition)

      F_hertz = if appr === nothing
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

function make_uode_func_oop(appr, st, known_pars; nn_gain::Float64=1.0)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  function f(u, p, t)
    ks = p.mech[1]
    cs = p.mech[2]

    s = dist + u[1] - u[3]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    w_pred = contact_weight(s, adhesion_transition)

    F_hertz = if appr === nothing
      (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
    else
      nn_in = nn_input_from_state(u)
      uhat = appr(nn_in, p.p_net, st)[1]
      nn_gain * uhat[1] * w_pred
    end
    Fad_eff = Fad * w_pred

    du1 = u[2]
    du2 = (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
    du3 = (Fad_eff - F_hertz - ks * u[3]) / cs
    return [du1, du2, du3]
  end
  return f
end

function x2dot_rhs(u, mech, p_net, appr, st, known_pars, t; nn_gain::Float64=1.0)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  ks, cs = mech
  s = dist + u[1] - u[3]
  delta = softplus(-s, adhesion_transition)
  delta = ifelse(delta > 0.0, delta, 0.0)
  w_pred = contact_weight(s, adhesion_transition)
  F_hertz = if appr === nothing
    (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
  else
    nn_in = nn_input_from_state(u)
    uhat = appr(nn_in, p_net, st)[1]
    nn_gain * uhat[1] * w_pred
  end
  Fad_eff = Fad * w_pred
  return (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
end

function x2dot_rhs_batch(uhat, contact_idx, mech, p_net, appr, st, known_pars, times; nn_gain::Float64=1.0)
  isempty(contact_idx) && return Float64[]
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  @views x1 = vec(uhat[1, contact_idx])
  @views x2 = vec(uhat[2, contact_idx])
  @views x3 = vec(uhat[3, contact_idx])
  t_contact = times[contact_idx]
  s = dist .+ x1 .- x3
  w_pred = contact_weight.(s, adhesion_transition)
  F_hertz = if appr === nothing
    delta = softplus.(-s, adhesion_transition)
    delta = ifelse.(delta .> 0.0, delta, 0.0)
    (4.0 / 3.0) .* Estar .* sqrt(R) .* (delta .^ 1.5)
  else
    nn_in = @views uhat[1:2, contact_idx]
    nn_out = appr(nn_in, p_net, st)[1]
    nn_gain .* vec(nn_out) .* w_pred
  end
  Fad_eff = Fad .* w_pred
  return (Fd .* cos.(wd .* t_contact) .- k .* x1 .- c .* x2 .+ Fad_eff .- F_hertz) ./ m
end

function loss_single_or_ms(θ, ode_data, x2dot_data, contact_mask, times,
  state12_scale, x2dot_scale, x3_scale,
  use_multiple_shooting, ms_group_size, ms_continuity_term,
  appr, st, known_pars, x3_t0_val,
  l2_weight, re_pnet, monitor_ref=nothing; nn_gain::Float64=1.0)

  eval_id = Zygote.ignore() do
    global stage2_loss_eval_counter
    if !@isdefined(stage2_loss_eval_counter)
      stage2_loss_eval_counter = 0
    end
    stage2_loss_eval_counter += 1
    stage2_loss_eval_counter
  end

  mech = [
    bound_param(θ.mech_raw[1], ks_bounds[1], ks_bounds[2]),
    bound_param(θ.mech_raw[2], cs_bounds[1], cs_bounds[2])
  ]
  p_net_struct = re_pnet(θ.p_net)
  p = ComponentVector(p_net=p_net_struct, mech=mech)

  total_state = 0.0
  total_x2dot = 0.0
  total_x3range = 0.0
  total_cont = 0.0
  x2dot_enabled = false
  x3_range_enabled = x3_range_weight != 0.0
  cont_enabled = false
  l2_enabled = l2_weight != 0.0
  x1_rec_ref = Ref(NaN)
  x3_rec_ref = Ref(NaN)
  fhertz_err_ref = Ref(NaN)
  raw_min_ref = Ref(NaN)
  raw_max_ref = Ref(NaN)
  raw_span_ref = Ref(NaN)
  raw_mean_ref = Ref(NaN)
  raw_neg_frac_ref = Ref(NaN)

  function set_monitor_status(status, reason, loss_value)
    if monitor_ref === nothing
      return
    end
    Zygote.ignore() do
      monitor_ref[] = (
        eval_id=eval_id,
        status=status,
        reason=reason,
        loss=loss_value,
        state=total_state, x2dot=total_x2dot, x3_range=total_x3range, cont=total_cont,
        l2=NaN,
        x2dot_enabled=x2dot_enabled,
        x3_range_enabled=x3_range_enabled,
        cont_enabled=cont_enabled,
        l2_enabled=l2_enabled,
        x1_rec=x1_rec_ref[], x3_rec=x3_rec_ref[], fhertz_err=fhertz_err_ref[],
        raw_min=raw_min_ref[], raw_max=raw_max_ref[], raw_span=raw_span_ref[], raw_mean=raw_mean_ref[],
        raw_neg_frac=raw_neg_frac_ref[]
      )
    end
  end

  weights_val = ifelse.(contact_mask, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_val)
  if !isfinite(weights_sum) || weights_sum <= 0
    set_monitor_status("failed", "weights_sum_nonfinite_or_nonpositive", Inf)
    return Inf
  end

  if use_multiple_shooting
    ranges = DiffEqFlux.group_ranges(length(times), ms_group_size)
    preds = Vector{Matrix{Float64}}(undef, length(ranges))
    for (i, rg) in enumerate(ranges)
      u0 = [ode_data[1, first(rg)], ode_data[2, first(rg)], x3_t0_val]
      prob = ODEProblem{false}(make_uode_func_oop(appr, st, known_pars; nn_gain=nn_gain), u0, (times[first(rg)], times[last(rg)]), p)
      sol = solve(prob, integrator; saveat=times[rg], abstol=abstol, reltol=reltol, sensealg=sensealg, maxiters=ode_maxiters)
      if string(sol.retcode) != "Success"
        set_monitor_status("failed", "ms_solver_retcode_" * string(sol.retcode), Inf)
        return Inf
      end
      if size(sol, 2) != length(rg)
        set_monitor_status("failed", "ms_sol_size_mismatch", Inf)
        return Inf
      end
      preds[i] = Array(sol)

      uhat = preds[i]
      seg_weights = weights_val[rg]
      seg_weights_sum = sum(seg_weights)
      if !isfinite(seg_weights_sum) || seg_weights_sum <= 0
        set_monitor_status("failed", "ms_seg_weights_sum_nonfinite_or_nonpositive", Inf)
        return Inf
      end

      state_err = vec(sum(abs2.((ode_data[1:2, rg] .- uhat[1:2, :]) ./ state12_scale), dims=1))
      total_state += sum(seg_weights .* state_err) / seg_weights_sum

      contact_idx = findall(contact_mask[rg])
      if !isempty(contact_idx)
        x2dot_enabled = true
        local_idx = rg[contact_idx]
        x2dot_pred_contact = x2dot_rhs_batch(uhat, contact_idx, mech, p_net_struct, appr, st, known_pars, times[rg]; nn_gain=nn_gain)
        if any(x -> !isfinite(x), x2dot_pred_contact)
          set_monitor_status("failed", "ms_x2dot_pred_nonfinite", Inf)
          return Inf
        end
        x2_err = abs2.((x2dot_data[local_idx] .- x2dot_pred_contact) ./ x2dot_scale)
        x2_w = seg_weights[contact_idx]
        total_x2dot += sum(x2_w .* x2_err) / sum(x2_w)
      end

      exceed = abs.(uhat[3, :]) .- x3_range_amp
      range_pen = abs2.(softplus.(exceed, x3_range_eps) ./ x3_scale)
      total_x3range += x3_range_weight * (sum(seg_weights .* range_pen) / seg_weights_sum)
    end

    for i in 2:length(preds)
      cont_enabled = ms_continuity_term != 0.0
      u0 = preds[i-1][:, end]
      u1 = preds[i][:, 1]
      total_cont += ms_continuity_term * sum(abs2, u0 - u1)
    end
  else
    u0 = [ode_data[1, 1], ode_data[2, 1], x3_t0_val]
    prob = ODEProblem{false}(make_uode_func_oop(appr, st, known_pars; nn_gain=nn_gain), u0, (times[1], times[end]), p)
    sol = solve(prob, integrator; saveat=times, abstol=abstol, reltol=reltol, sensealg=sensealg, maxiters=ode_maxiters)
    if string(sol.retcode) != "Success"
      set_monitor_status("failed", "ss_solver_retcode_" * string(sol.retcode), Inf)
      return Inf
    end
    if size(sol, 2) != length(times)
      set_monitor_status("failed", "ss_sol_size_mismatch", Inf)
      return Inf
    end
    uhat = Array(sol)

    state_err = vec(sum(abs2.((ode_data[1:2, :] .- uhat[1:2, :]) ./ state12_scale), dims=1))
    total_state = sum(weights_val .* state_err) / weights_sum

    contact_idx = findall(contact_mask)
    if !isempty(contact_idx)
      x2dot_enabled = true
      x2dot_pred_contact = x2dot_rhs_batch(uhat, contact_idx, mech, p_net_struct, appr, st, known_pars, times; nn_gain=nn_gain)
      if any(x -> !isfinite(x), x2dot_pred_contact)
        set_monitor_status("failed", "ss_x2dot_pred_nonfinite", Inf)
        return Inf
      end
      x2_err = abs2.((x2dot_data[contact_idx] .- x2dot_pred_contact) ./ x2dot_scale)
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

      if !isempty(monitor_idx_full)
        if appr === nothing
          fhertz_err_ref[] = 0.0
        else
          nn_metrics = fhertz_monitor_metrics(
            ode_data_full, monitor_idx_full, p_net_struct, appr, st;
            fh_true_ref=monitor_fhertz_true, eff_pos_ref=monitor_eff_positions, nn_gain=nn_gain
          )
          fhertz_err_ref[] = nn_metrics.fhertz_err
          raw_min_ref[] = nn_metrics.raw_min
          raw_max_ref[] = nn_metrics.raw_max
          raw_span_ref[] = nn_metrics.raw_span
          raw_mean_ref[] = nn_metrics.raw_mean
          raw_neg_frac_ref[] = nn_metrics.raw_neg_frac
        end
      end
    end
  end

  l2_penalty = l2_weight * sum(abs2, θ.p_net)
  total = total_state + total_x2dot + total_x3range + total_cont + l2_penalty
  if monitor_ref !== nothing
    Zygote.ignore() do
      monitor_ref[] = (
        eval_id=eval_id,
        status="ok",
        reason="",
        loss=total,
        state=total_state, x2dot=total_x2dot, x3_range=total_x3range, cont=total_cont,
        l2=l2_penalty,
        x2dot_enabled=x2dot_enabled,
        x3_range_enabled=x3_range_enabled,
        cont_enabled=cont_enabled,
        l2_enabled=l2_enabled,
        x1_rec=x1_rec_ref[], x3_rec=x3_rec_ref[], fhertz_err=fhertz_err_ref[],
        raw_min=raw_min_ref[], raw_max=raw_max_ref[], raw_span=raw_span_ref[], raw_mean=raw_mean_ref[],
        raw_neg_frac=raw_neg_frac_ref[]
      )
    end
  end
  return total
end

# Select top-10 candidates from Stage1 (by val_loss, fallback to loss)
function trial_val(r)
  if hasproperty(r, :val_loss)
    return r.val_loss
  elseif hasproperty(r, :loss)
    return r.loss
  else
    return Inf
  end
end

stage1_trials = stage1.trial_parameters
sorted_stage1 = sort(stage1_trials, by = trial_val)
top_candidates = sorted_stage1[1:min(stage2_input_topk, length(sorted_stage1))]
if isempty(top_candidates)
  error("Stage2: no candidates from Stage1")
end

function parse_candidate_filter(spec::AbstractString, nmax::Int)
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
      error("HNODECB_STAGE2_CANDIDATE_INDICES contains a non-integer token: " * t)
    end
    push!(idx, v)
  end
  if isempty(idx)
    error("HNODECB_STAGE2_CANDIDATE_INDICES is set but no valid indices were parsed")
  end
  idx = sort(unique(idx))
  for v in idx
    if v < 1 || v > nmax
      error("HNODECB_STAGE2_CANDIDATE_INDICES index " * string(v) *
            " is outside 1.." * string(nmax))
    end
  end
  return idx
end

candidate_indices = [i for i in 1:length(top_candidates) if ((i - stage2_shard_index) % stage2_shard_count) == 0]
candidate_filter_indices = parse_candidate_filter(stage2_candidate_filter_spec, length(top_candidates))
if candidate_filter_indices !== nothing
  allowed = Set(candidate_filter_indices)
  candidate_indices = [i for i in candidate_indices if i in allowed]
  tprintln("Stage2 candidate filter: [", join(candidate_filter_indices, ", "), "]")
end
if isempty(candidate_indices)
  tprintln("Stage2 shard ", stage2_shard_index, "/", stage2_shard_count, ": no assigned candidates")
end
warm_start_by_trial = Dict{Any, Any}()
if haskey(stage1, :warm_start_top)
  for rec in stage1.warm_start_top
    if hasproperty(rec, :trial_id)
      warm_start_by_trial[rec.trial_id] = rec
    end
  end
end
stage2_nn_warm_by_base_rank = Dict{Int, Vector{Any}}()
stage2_nn_warm_file = normpath(joinpath(@__DIR__, "..", "hyperparameter_tuning_first_stage",
  "results_afm", stage2_nn_warm_basename))
if stage2_nn_warm_enabled
  if isfile(stage2_nn_warm_file)
    stage2_nn_warm_data = deserialize(stage2_nn_warm_file)
    warm_pool = haskey(stage2_nn_warm_data, :selected) ? stage2_nn_warm_data.selected :
      (haskey(stage2_nn_warm_data, :trial_parameters) ? stage2_nn_warm_data.trial_parameters : Any[])
    for rec in warm_pool
      params = hasproperty(rec, :params) ? rec.params : Dict{Any, Any}()
      if !(params isa AbstractDict) || !haskey(params, "base_rank")
        continue
      end
      base_rank_raw = params["base_rank"]
      base_rank = try
        Int(base_rank_raw)
      catch
        continue
      end
      p_vec = if hasproperty(rec, :p_net_vec)
        rec.p_net_vec
      elseif hasproperty(rec, :p_net)
        rec.p_net
      else
        nothing
      end
      if p_vec === nothing
        continue
      end
      train_loss = hasproperty(rec, :train_loss) ? rec.train_loss : Inf
      val_loss = hasproperty(rec, :val_loss) ? rec.val_loss : Inf
      g_nn = 1.0
      if stage2_use_gnn
        g_nn = if hasproperty(rec, :g_nn)
          try
            Float64(rec.g_nn)
          catch
            stage2_gnn_default
          end
        elseif haskey(params, "g_nn")
          try
            Float64(params["g_nn"])
          catch
            stage2_gnn_default
          end
        else
          stage2_gnn_default
        end
        if !isfinite(g_nn) || g_nn <= 0.0
          g_nn = stage2_gnn_default
        end
      end
      if !haskey(stage2_nn_warm_by_base_rank, base_rank)
        stage2_nn_warm_by_base_rank[base_rank] = Any[]
      end
      push!(stage2_nn_warm_by_base_rank[base_rank], (
        p_net_vec=copy(p_vec),
        g_nn=g_nn,
        train_loss=train_loss,
        val_loss=val_loss,
        num_hidden_layers=(haskey(params, "num_hidden_layers") ? try Int(params["num_hidden_layers"]) catch nn_fixed_num_hidden_layers end : nn_fixed_num_hidden_layers),
        num_hidden_nodes=(haskey(params, "num_hidden_nodes") ? try Int(params["num_hidden_nodes"]) catch nn_fixed_num_hidden_nodes end : nn_fixed_num_hidden_nodes)
      ))
    end
    for recs in values(stage2_nn_warm_by_base_rank)
      sort!(recs, by = r -> (isfinite(r.val_loss) ? r.val_loss : Inf))
    end
    warm_base_ranks = sort!(collect(keys(stage2_nn_warm_by_base_rank)))
    tprintln("Stage2 NN warm-start: loaded ", length(warm_pool), " entries from ",
      stage2_nn_warm_file)
    tprintln("Stage2 NN warm-start: base_ranks=[", join(warm_base_ranks, ", "),
      "] | rank_pick=", stage2_nn_warm_rank)
  else
    tprintln("Stage2 NN warm-start: file not found -> ", stage2_nn_warm_file,
      " | fallback=random init")
  end
end
tprintln("Stage2 using top-", length(top_candidates), " candidates from Stage1")
tprintln("Stage2 shard assignment: ", stage2_shard_index, "/", stage2_shard_count,
  " | local_candidates=", length(candidate_indices))
if isempty(candidate_indices)
  tprintln("Stage2 shard candidates: []")
else
  tprintln("Stage2 shard candidates: [", join(candidate_indices, ", "), "]")
end
tprintln("Stage2 warm-start NN states available: ", length(warm_start_by_trial))
tprintln("Stage2 NN config: fixed layers=", nn_fixed_num_hidden_layers,
  " nodes=", nn_fixed_num_hidden_nodes, " | bias=OFF | p_net warm-start=",
  stage2_nn_warm_enabled ? "ON" : "OFF")
tprintln("Stage2 input top-", length(top_candidates), " from Stage1 (train/val):")
for (rank, rec) in enumerate(top_candidates)
  train_loss = hasproperty(rec, :train_loss) ? rec.train_loss : trial_val(rec)
  val_loss = trial_val(rec)
  ks0 = hasproperty(rec, :params) && haskey(rec.params, "ks0") ? rec.params["ks0"] : NaN
  cs0 = hasproperty(rec, :params) && haskey(rec.params, "cs0") ? rec.params["cs0"] : NaN
  tprintln("  Rank ", rank,
    " -- train=", @sprintf("%.4e", train_loss),
    " val=", @sprintf("%.4e", val_loss),
    " | ks0=", @sprintf("%.3e", ks0),
    " cs0=", @sprintf("%.3e", cs0))
end

val_stride = haskey(stage1, :val_stride) ? stage1.val_stride : 5
val_offset = haskey(stage1, :val_offset) ? stage1.val_offset : 2

# Build train/val split
train_idx, val_idx = make_train_val_masks(length(all_times), val_stride, val_offset)
ode_train = ode_data_full[:, train_idx]
ode_val = ode_data_full[:, val_idx]
times_train = all_times[train_idx]
times_val = all_times[val_idx]
x2dot_train = x2dot_all[train_idx]
x2dot_val = x2dot_all[val_idx]
contact_train = contact_all[train_idx]
contact_val = contact_all[val_idx]

function rand_loguniform(rng, lo, hi)
  return exp(rand(rng) * (log(hi) - log(lo)) + log(lo))
end

function scale_stage2_grad(grad_raw, p_net_scale::Float64, mech_scale::Float64)
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

function run_adam(theta0, loss_fn; lr_init, maxiters=500, log_every=100, monitor_ref=nothing,
  init_source="none", init_attempt=1, init_attempt_max=1)
  lr_adapt = get(ENV, "HNODECB_LR_ADAPT", "1") == "1"
  optimizer_name = lowercase(strip(get(ENV, "HNODECB_STAGE2_OPTIMIZER", "amsgrad")))
  trace_epoch_first_attempt = get(ENV, "HNODECB_STAGE2_TRACE_EPOCH_FIRST_ATTEMPT",
    get(ENV, "HNODECB_STAGE2_TRACE_FIRST_ATTEMPT", "1")) == "1"
  mech_grad_diag = get(ENV, "HNODECB_STAGE2_MECH_GRAD_DIAG", "1") == "1"
  group_step_probe = get(ENV, "HNODECB_STAGE2_GROUP_STEP_PROBE", "1") == "1"
  mech_fd_eps = env_float("HNODECB_STAGE2_MECH_FD_EPS", 1e-6)
  if !isfinite(mech_fd_eps) || mech_fd_eps <= 0.0
    mech_fd_eps = 1e-6
  end
  grad_scale_p_net = env_float("HNODECB_STAGE2_GRAD_SCALE_P_NET", 0.2)
  grad_scale_mech = env_float("HNODECB_STAGE2_GRAD_SCALE_MECH", 1.0)
  if !isfinite(grad_scale_p_net) || grad_scale_p_net < 0.0
    grad_scale_p_net = 1.0
  end
  if !isfinite(grad_scale_mech) || grad_scale_mech < 0.0
    grad_scale_mech = 1.0
  end
  group_adapt = get(ENV, "HNODECB_STAGE2_GROUP_ADAPT", "1") == "1"
  group_ema_alpha = env_float("HNODECB_STAGE2_GROUP_EMA", 0.90)
  group_eta = env_float("HNODECB_STAGE2_GROUP_ETA", 0.15)
  group_eps = env_float("HNODECB_STAGE2_GROUP_EPS", 1e-30)
  grad_scale_p_min = env_float("HNODECB_STAGE2_GRAD_SCALE_P_NET_MIN", 0.02)
  grad_scale_p_max = env_float("HNODECB_STAGE2_GRAD_SCALE_P_NET_MAX", 1.0)
  grad_scale_m_min = env_float("HNODECB_STAGE2_GRAD_SCALE_MECH_MIN", 0.2)
  grad_scale_m_max = env_float("HNODECB_STAGE2_GRAD_SCALE_MECH_MAX", 2.0)
  if !isfinite(group_ema_alpha) || group_ema_alpha < 0.0 || group_ema_alpha >= 1.0
    group_ema_alpha = 0.90
  end
  if !isfinite(group_eta) || group_eta <= 0.0
    group_eta = 0.15
  end
  if !isfinite(group_eps) || group_eps <= 0.0
    group_eps = 1e-30
  end
  if !isfinite(grad_scale_p_min) || grad_scale_p_min <= 0.0
    grad_scale_p_min = 0.02
  end
  if !isfinite(grad_scale_p_max) || grad_scale_p_max < grad_scale_p_min
    grad_scale_p_max = max(grad_scale_p_min, 1.0)
  end
  if !isfinite(grad_scale_m_min) || grad_scale_m_min <= 0.0
    grad_scale_m_min = 0.2
  end
  if !isfinite(grad_scale_m_max) || grad_scale_m_max < grad_scale_m_min
    grad_scale_m_max = max(grad_scale_m_min, 2.0)
  end
  grad_scale_p_net = clamp(grad_scale_p_net, grad_scale_p_min, grad_scale_p_max)
  grad_scale_mech = clamp(grad_scale_mech, grad_scale_m_min, grad_scale_m_max)
  lr_min = env_float("HNODECB_LR_MIN", 1e-6)
  lr_max = env_float("HNODECB_LR_MAX", 1e-2)
  lr_eta = env_float("HNODECB_LR_ETA", 0.05)
  lr_ema_alpha = env_float("HNODECB_LR_EMA", 0.97)
  lr_eps = env_float("HNODECB_LR_EPS", 1e-30)
  lr_target_init = env_float("HNODECB_LR_TARGET", NaN)
  plateau_early_stop = get(ENV, "HNODECB_STAGE2_PLATEAU_EARLY_STOP", "0") == "1"
  plateau_window = 5
  plateau_tol = 1e-6
  good_enough_loss = 1e-10
  epoch_retry_max = max(0, parse(Int, get(ENV, "HNODECB_STAGE2_EPOCH_RETRIES", "8")))
  epoch_retry_lr_factor = env_float("HNODECB_STAGE2_RETRY_LR_FACTOR", 0.2)
  if !isfinite(epoch_retry_lr_factor) || epoch_retry_lr_factor <= 0.0 || epoch_retry_lr_factor >= 1.0
    epoch_retry_lr_factor = 0.2
  end
  epoch_retry_lr_floor = env_float("HNODECB_STAGE2_RETRY_LR_FLOOR", lr_min)
  if !isfinite(epoch_retry_lr_floor) || epoch_retry_lr_floor <= 0.0
    epoch_retry_lr_floor = lr_min
  end
  step_guard = get(ENV, "HNODECB_STAGE2_STEP_GUARD", "1") == "1"
  step_retry_max = max(0, parse(Int, get(ENV, "HNODECB_STAGE2_STEP_RETRIES", "6")))
  step_retry_lr_factor = env_float("HNODECB_STAGE2_STEP_RETRY_LR_FACTOR", 0.5)
  if !isfinite(step_retry_lr_factor) || step_retry_lr_factor <= 0.0 || step_retry_lr_factor >= 1.0
    step_retry_lr_factor = 0.5
  end
  step_max_loss_increase = env_float("HNODECB_STAGE2_STEP_MAX_LOSS_INCREASE", 1e-2)
  verify_adam_dir = get(ENV, "HNODECB_STAGE2_VERIFY_ADAM_DIR", "1") == "1"
  verify_adam_eps = env_float("HNODECB_STAGE2_VERIFY_ADAM_EPS", 1e-30)

  lr = lr_init

  trace_dt_str(t_ns) = fmt_e((time_ns() - t_ns) / 1e9, sigdigits=4)
  opt_rule = if optimizer_name == "adam"
    Optimisers.Adam(lr)
  elseif optimizer_name == "amsgrad"
    Optimisers.AMSGrad(lr)
  else
    tprintln("  warn: unknown HNODECB_STAGE2_OPTIMIZER=", optimizer_name,
      " -> fallback=amsgrad")
    optimizer_name = "amsgrad"
    Optimisers.AMSGrad(lr)
  end
  opt_state = Optimisers.setup(opt_rule, theta0)
  theta = theta0
  grad_ema = Ref(lr_target_init)
  grad_target = Ref(lr_target_init)
  recent_losses = Float64[]
  p_grad_ema = Ref(NaN)
  m_grad_ema = Ref(NaN)
  epoch1_loss = Inf
  epoch1_gnorm_raw = NaN
  epoch1_accept_logged = false
  failure_epoch = 0
  failure_reason = ""
  tprintln("  optimizer=", uppercase(optimizer_name),
    " | grad_scales: p_net=", fmt_e(grad_scale_p_net, sigdigits=3),
    " mech_raw=", fmt_e(grad_scale_mech, sigdigits=3))
  tprintln("  group_adapt=", group_adapt ? "ON" : "OFF",
    " | group_eta=", fmt_e(group_eta, sigdigits=3),
    " | scale_bounds[p_net=", fmt_e(grad_scale_p_min, sigdigits=3), "..", fmt_e(grad_scale_p_max, sigdigits=3),
    ", mech_raw=", fmt_e(grad_scale_m_min, sigdigits=3), "..", fmt_e(grad_scale_m_max, sigdigits=3), "]")

  function log_nonfinite_debug(epoch, loss, gnorm, theta_now)
    tprintln("  debug: nonfinite training signal at epoch ", epoch,
      " loss=", fmt_e(loss, sigdigits=4),
      " grad_norm=", fmt_e(gnorm, sigdigits=3))
    if monitor_ref !== nothing && monitor_ref[] !== nothing
      diag = monitor_ref[]
      diag_eval_id = hasproperty(diag, :eval_id) ? string(diag.eval_id) : "unknown"
      diag_status = hasproperty(diag, :status) ? string(diag.status) : "unknown"
      diag_reason = hasproperty(diag, :reason) ? string(diag.reason) : "unknown"
      diag_loss = hasproperty(diag, :loss) ? fmt_e(diag.loss, sigdigits=4) : "unknown"
      tprintln("  debug monitor: eval_id=", diag_eval_id,
        " status=", diag_status,
        " reason=", diag_reason,
        " loss=", diag_loss)
      tprintln("  debug parts: state=", fmt_loss_part(diag.state, true, sigdigits=3),
        " x2dot=", fmt_loss_part(diag.x2dot, diag.x2dot_enabled, sigdigits=3),
        " x3r=", fmt_loss_part(diag.x3_range, diag.x3_range_enabled, sigdigits=3),
        " cont=", fmt_loss_part(diag.cont, diag.cont_enabled, sigdigits=3),
        " l2=", fmt_loss_part(diag.l2, diag.l2_enabled, sigdigits=3))
      tprintln("  debug rec: x1=", fmt_f(diag.x1_rec, digits=2),
        "% x3=", fmt_f(diag.x3_rec, digits=2), "%")
      tprintln("  debug nn: F_hertz err=", fmt_f(diag.fhertz_err, digits=2), "%")
      tprintln("  debug nn raw: min=", fmt_e(diag.raw_min, sigdigits=3),
        " max=", fmt_e(diag.raw_max, sigdigits=3),
        " span=", fmt_e(diag.raw_span, sigdigits=3),
        " mean=", fmt_e(diag.raw_mean, sigdigits=3),
        " neg=", fmt_f(100 * diag.raw_neg_frac, digits=2), "%")
    end
    ks_dbg = bound_param(theta_now.mech_raw[1], ks_bounds[1], ks_bounds[2])
    cs_dbg = bound_param(theta_now.mech_raw[2], cs_bounds[1], cs_bounds[2])
    tprintln("  debug mech: ks=", fmt_e(ks_dbg, sigdigits=3),
      " (err=", fmt_f(rel_err_pct(ks_dbg, ks_true, scale_eps), digits=2), "%)",
      " cs=", fmt_e(cs_dbg, sigdigits=3),
      " (err=", fmt_f(rel_err_pct(cs_dbg, cs_true, scale_eps), digits=2), "%)")
  end

  function monitor_failure_reason()
    if monitor_ref === nothing || monitor_ref[] === nothing
      return ""
    end
    diag = monitor_ref[]
    if hasproperty(diag, :status) && string(diag.status) == "failed"
      return hasproperty(diag, :reason) ? string(diag.reason) : "monitor_failed_unknown_reason"
    end
    return ""
  end

  function eval_trial_loss(theta_trial)
    if monitor_ref !== nothing
      monitor_ref[] = nothing
    end
    trial_loss = Inf
    reason = ""
    try
      trial_loss = loss_fn(theta_trial)
    catch ex
      reason = "exception: " * sprint(showerror, ex)
    end
    if reason == ""
      monitor_reason = monitor_failure_reason()
      if monitor_reason != ""
        reason = monitor_reason
      elseif !isfinite(trial_loss)
        reason = "loss_nonfinite"
      end
    end
    return trial_loss, reason
  end

  function mech_bound_slopes(theta_now)
    slopes = fill(NaN, length(theta_now.mech_raw))
    for i in eachindex(theta_now.mech_raw)
      if i == 1
        lo, hi = ks_bounds[1], ks_bounds[2]
      else
        lo, hi = cs_bounds[1], cs_bounds[2]
      end
      sig = sigmoid(theta_now.mech_raw[i])
      slopes[i] = (hi - lo) * sig * (1 - sig)
    end
    return slopes
  end

  function mech_fd_grad(theta_now)
    fd = fill(NaN, length(theta_now.mech_raw))
    steps = fill(NaN, length(theta_now.mech_raw))
    reasons = String[]
    for i in eachindex(theta_now.mech_raw)
      h = mech_fd_eps * max(1.0, abs(theta_now.mech_raw[i]))
      steps[i] = h

      theta_plus = ComponentVector(p_net=copy(theta_now.p_net), mech_raw=copy(theta_now.mech_raw))
      theta_minus = ComponentVector(p_net=copy(theta_now.p_net), mech_raw=copy(theta_now.mech_raw))
      theta_plus.mech_raw[i] += h
      theta_minus.mech_raw[i] -= h

      loss_plus, reason_plus = eval_trial_loss(theta_plus)
      loss_minus, reason_minus = eval_trial_loss(theta_minus)
      if reason_plus == "" && reason_minus == ""
        fd[i] = (loss_plus - loss_minus) / (2h)
      else
        push!(reasons, "mech[" * string(i) * "] += " *
          (reason_plus == "" ? "ok" : reason_plus) * ", -= " *
          (reason_minus == "" ? "ok" : reason_minus))
      end
    end
    return fd, steps, reasons
  end

  function group_probe_dloss(theta_base, p_delta, m_delta, loss_base, dL_full_known)
    p_norm = sqrt(sum(abs2, p_delta))
    m_norm = sqrt(sum(abs2, m_delta))
    full_norm = sqrt(p_norm^2 + m_norm^2)

    dL_full = dL_full_known
    dL_net = NaN
    dL_mech = NaN
    interaction = NaN
    reason_full = isfinite(dL_full_known) ? "" : "full_step_unknown"
    reason_net = ""
    reason_mech = ""

    if p_norm == 0.0
      dL_net = 0.0
    else
      theta_net = ComponentVector(
        p_net = theta_base.p_net .+ p_delta,
        mech_raw = copy(theta_base.mech_raw)
      )
      loss_net, reason_net = eval_trial_loss(theta_net)
      if reason_net == ""
        dL_net = loss_net - loss_base
      end
    end

    if m_norm == 0.0
      dL_mech = 0.0
    else
      theta_mech = ComponentVector(
        p_net = copy(theta_base.p_net),
        mech_raw = theta_base.mech_raw .+ m_delta
      )
      loss_mech, reason_mech = eval_trial_loss(theta_mech)
      if reason_mech == ""
        dL_mech = loss_mech - loss_base
      end
    end

    if reason_full == "" && reason_net == "" && reason_mech == ""
      interaction = dL_full - dL_net - dL_mech
    end

    return (
      dL_full=dL_full,
      dL_net=dL_net,
      dL_mech=dL_mech,
      interaction=interaction,
      full_norm=full_norm,
      net_norm=p_norm,
      mech_norm=m_norm,
      reason_full=reason_full,
      reason_net=reason_net,
      reason_mech=reason_mech
    )
  end

  for epoch in 1:maxiters
    epoch_start_theta = deepcopy(theta)
    epoch_start_opt_state = deepcopy(opt_state)
    max_attempts = epoch_retry_max + 1
    attempt = 0
    recovered = false
    loss = Inf
    grad_raw = nothing
    grad_update = nothing
    gnorm = NaN
    gnorm_raw = NaN
    gnorm_p_raw = NaN
    gnorm_m_raw = NaN
    fail_reason = ""

    while attempt < max_attempts
      attempt += 1
      trace_this_attempt = trace_epoch_first_attempt && epoch == 1 && attempt == 1
      loss = Inf
      grad_raw = nothing
      grad_update = nothing
      gnorm = NaN
      gnorm_raw = NaN
      gnorm_p_raw = NaN
      gnorm_m_raw = NaN
      fail_reason = ""
      if monitor_ref !== nothing
        monitor_ref[] = nothing
      end

      try
        pullback_t0 = trace_this_attempt ? time_ns() : 0
        if trace_this_attempt
          tprintln("  trace[epoch1/attempt1]: init_source=", init_source,
            " init_attempt=", init_attempt, "/", init_attempt_max)
          tprintln("  trace[epoch1/attempt1]: pullback_begin")
        end
        loss, back = Zygote.pullback(loss_fn, theta)
        if epoch == 1
          epoch1_loss = loss
        end
        if trace_this_attempt
          tprintln("  trace[epoch1/attempt1]: pullback_done dt=", trace_dt_str(pullback_t0),
            " loss=", fmt_e(loss, sigdigits=4))
        end
        backward_t0 = trace_this_attempt ? time_ns() : 0
        if trace_this_attempt
          tprintln("  trace[epoch1/attempt1]: backward_begin")
        end
        grad_raw = first(back(1.0))
        if trace_this_attempt
          tprintln("  trace[epoch1/attempt1]: backward_done dt=", trace_dt_str(backward_t0))
        end
        gradprep_t0 = trace_this_attempt ? time_ns() : 0
        gnorm_raw = grad_norm_safe(grad_raw)
        if epoch == 1
          epoch1_gnorm_raw = gnorm_raw
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
        grad_update = scale_stage2_grad(grad_raw, grad_scale_p_net, grad_scale_mech)
        gnorm = grad_norm_safe(grad_update)
        if trace_this_attempt
          tprintln("  trace[epoch1/attempt1]: gradprep_done dt=", trace_dt_str(gradprep_t0),
            " | grad_norm_raw=", fmt_e(gnorm_raw, sigdigits=3),
            " grad_norm_scaled=", fmt_e(gnorm, sigdigits=3))
        end
      catch ex
        fail_reason = "exception: " * sprint(showerror, ex)
      end

      if fail_reason == ""
        monitor_reason = monitor_failure_reason()
        if monitor_reason != ""
          fail_reason = monitor_reason
        elseif !isfinite(loss)
          fail_reason = "train_loss_nonfinite"
        elseif !isfinite(gnorm_raw)
          fail_reason = "grad_norm_nonfinite"
        elseif !isfinite(gnorm)
          fail_reason = "grad_scaled_norm_nonfinite"
        end
      end

      if fail_reason == ""
        recovered = attempt > 1
        break
      end

      if attempt < max_attempts
        theta = deepcopy(epoch_start_theta)
        opt_state = deepcopy(epoch_start_opt_state)
        lr_new = max(lr * epoch_retry_lr_factor, epoch_retry_lr_floor)
        if lr_new < lr
          Optimisers.adjust!(opt_state, lr_new)
          lr = lr_new
        end
        tprintln("  retry epoch ", epoch, " attempt ", attempt, "/", max_attempts,
          " -- reason=", fail_reason, " | lr=", fmt_e(lr, sigdigits=3))
      end
    end

    if fail_reason != ""
      failure_epoch = epoch
      failure_reason = fail_reason
      log_nonfinite_debug(epoch, loss, gnorm, theta)
      return theta, Inf, "epoch_retry_exhausted: " * fail_reason, (
        epoch1_loss=epoch1_loss,
        epoch1_gnorm_raw=epoch1_gnorm_raw,
        failure_epoch=failure_epoch,
        failure_reason=failure_reason
      )
    end

    if epoch == 1 && !epoch1_accept_logged
      tprintln("  init accepted on attempt ", init_attempt, "/", init_attempt_max,
        " -- epoch1 loss=", fmt_e(epoch1_loss, sigdigits=4),
        " grad_norm=", fmt_e(epoch1_gnorm_raw, sigdigits=3),
        " | source=", init_source)
      if monitor_ref !== nothing && monitor_ref[] !== nothing
        diag = monitor_ref[]
        tprintln("  epoch1 nn raw: min=", fmt_e(diag.raw_min, sigdigits=3),
          " max=", fmt_e(diag.raw_max, sigdigits=3),
          " span=", fmt_e(diag.raw_span, sigdigits=3),
          " mean=", fmt_e(diag.raw_mean, sigdigits=3),
          " neg=", fmt_f(100 * diag.raw_neg_frac, digits=2), "%")
      end
      epoch1_accept_logged = true
    end

    if epoch % log_every == 0
      tprintln("Stage2 epoch ", epoch, " train=", fmt_e(loss, sigdigits=4))
      tprintln("  grad_norm=", fmt_e(gnorm, sigdigits=3),
        " lr=", fmt_e(lr, sigdigits=3))
      tprintln("  lr_groups(eqv): p_net=", fmt_e(lr * grad_scale_p_net, sigdigits=3),
        " mech_raw=", fmt_e(lr * grad_scale_mech, sigdigits=3),
        " | base=", fmt_e(lr, sigdigits=3))
      if isfinite(gnorm_p_raw) && isfinite(gnorm_m_raw)
        tprintln("  grad_groups(raw): p_net=", fmt_e(gnorm_p_raw, sigdigits=3),
          " mech_raw=", fmt_e(gnorm_m_raw, sigdigits=3))
      end
      if grad_scale_p_net != 1.0 || grad_scale_mech != 1.0
        tprintln("  grad_norm_raw=", fmt_e(gnorm_raw, sigdigits=3),
          " grad_norm_scaled=", fmt_e(gnorm, sigdigits=3),
          " | scale[p_net=", fmt_e(grad_scale_p_net, sigdigits=3),
          ", mech_raw=", fmt_e(grad_scale_mech, sigdigits=3), "]")
      end
      if recovered
        tprintln("  retry: recovered_after=", attempt - 1, " rollback(s)")
      end
      if monitor_ref !== nothing && monitor_ref[] !== nothing
        diag = monitor_ref[]
        tprintln("  parts: state=", fmt_loss_part(diag.state, true, sigdigits=3),
          " x2dot=", fmt_loss_part(diag.x2dot, diag.x2dot_enabled, sigdigits=3),
          " x3r=", fmt_loss_part(diag.x3_range, diag.x3_range_enabled, sigdigits=3),
          " cont=", fmt_loss_part(diag.cont, diag.cont_enabled, sigdigits=3),
          " l2=", fmt_loss_part(diag.l2, diag.l2_enabled, sigdigits=3))
        tprintln("  rec: x1=", fmt_f(diag.x1_rec, digits=2),
          "% x3=", fmt_f(diag.x3_rec, digits=2), "%")
        tprintln("  nn: F_hertz err=", fmt_f(diag.fhertz_err, digits=2), "%")
        tprintln("  nn raw: min=", fmt_e(diag.raw_min, sigdigits=3),
          " max=", fmt_e(diag.raw_max, sigdigits=3),
          " span=", fmt_e(diag.raw_span, sigdigits=3),
          " mean=", fmt_e(diag.raw_mean, sigdigits=3),
          " neg=", fmt_f(100 * diag.raw_neg_frac, digits=2), "%")
      end
      ks_hat = bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2])
      cs_hat = bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])
      tprintln("  mech: ks=", fmt_e(ks_hat, sigdigits=3),
        " (err=", fmt_f(rel_err_pct(ks_hat, ks_true, scale_eps), digits=2), "%)",
        " cs=", fmt_e(cs_hat, sigdigits=3),
        " (err=", fmt_f(rel_err_pct(cs_hat, cs_true, scale_eps), digits=2), "%)")

      if verify_adam_dir && grad_raw !== nothing && grad_update !== nothing && isfinite(loss)
        theta_base = deepcopy(theta)
        opt_base = deepcopy(opt_state)

        adam_trial_loss = Inf
        adam_reason = ""
        theta_adam = nothing
        dL_adam = NaN
        t1_adam = NaN
        t2_adam = NaN
        try
          opt_probe = deepcopy(opt_base)
          _, theta_adam = Optimisers.update(opt_probe, theta_base, grad_update)
          adam_trial_loss, adam_reason = eval_trial_loss(theta_adam)
        catch ex
          adam_reason = "update_exception: " * sprint(showerror, ex)
        end

        raw_trial_loss = Inf
        raw_reason = ""
        step_adam_norm = NaN
        step_raw_norm = NaN
        dL_raw = NaN
        t1_raw = NaN
        t2_raw = NaN
        if adam_reason == "" && theta_adam !== nothing
          adam_step = theta_adam .- theta_base
          step_adam_norm = sqrt(sum(abs2, adam_step))
          dL_adam = adam_trial_loss - loss
          t1_adam = sum(grad_raw .* adam_step)
          t2_adam = dL_adam - t1_adam

          grad_desc = map(x -> -x, grad_raw)
          grad_desc_norm = sqrt(sum(abs2, grad_desc))
          if isfinite(step_adam_norm) && step_adam_norm > 0.0 &&
             isfinite(grad_desc_norm) && grad_desc_norm > 0.0
            raw_scale = step_adam_norm / (grad_desc_norm + verify_adam_eps)
            theta_raw = theta_base .+ (raw_scale .* grad_desc)
            raw_step = theta_raw .- theta_base
            step_raw_norm = sqrt(sum(abs2, raw_step))
            raw_trial_loss, raw_reason = eval_trial_loss(theta_raw)
            if raw_reason == ""
              dL_raw = raw_trial_loss - loss
              t1_raw = sum(grad_raw .* raw_step)
              t2_raw = dL_raw - t1_raw
            end
          else
            raw_reason = "degenerate_step_norm"
          end
        else
          raw_reason = "skip_due_to_adam_fail"
        end

        if adam_reason == "" && raw_reason == ""
          tprintln("  adam-vs-raw: dL_adam=", fmt_e(dL_adam, sigdigits=3),
            " dL_raw=", fmt_e(dL_raw, sigdigits=3),
            " | step_norm_adam=", fmt_e(step_adam_norm, sigdigits=3),
            " step_norm_raw=", fmt_e(step_raw_norm, sigdigits=3))
          tprintln("  adam-vs-raw terms: adam[t1=", fmt_e(t1_adam, sigdigits=3),
            ", t2=", fmt_e(t2_adam, sigdigits=3),
            "] raw[t1=", fmt_e(t1_raw, sigdigits=3),
            ", t2=", fmt_e(t2_raw, sigdigits=3), "]")
        else
          tprintln("  adam-vs-raw: adam=",
            (adam_reason == "" ? "ok" : adam_reason),
            " raw=",
            (raw_reason == "" ? "ok" : raw_reason))
        end

        if mech_grad_diag && hasproperty(grad_raw, :mech_raw) && length(grad_raw.mech_raw) >= 2
          slopes = mech_bound_slopes(theta_base)
          fd_mech, fd_steps, fd_reasons = mech_fd_grad(theta_base)
          ad_mech = grad_raw.mech_raw
          tprintln("  mech-grad AD(raw): [1]=", fmt_hp(ad_mech[1]),
            " [2]=", fmt_hp(ad_mech[2]))
          tprintln("  mech-grad FD(raw): [1]=", fmt_hp(fd_mech[1]),
            " [2]=", fmt_hp(fd_mech[2]),
            " | h=[", fmt_hp(fd_steps[1]), ", ", fmt_hp(fd_steps[2]), "]")
          tprintln("  mech-bound slope dp/draw: ks=", fmt_hp(slopes[1]),
            " cs=", fmt_hp(slopes[2]))
          if isempty(fd_reasons)
            diff1 = ad_mech[1] - fd_mech[1]
            diff2 = ad_mech[2] - fd_mech[2]
            tprintln("  mech-grad AD-FD diff(raw): [1]=", fmt_hp(diff1),
              " [2]=", fmt_hp(diff2))
          else
            tprintln("  mech-grad FD status: ", join(fd_reasons, " | "))
          end
        end

        if group_step_probe && adam_reason == "" && theta_adam !== nothing
          adam_p_delta = theta_adam.p_net .- theta_base.p_net
          adam_m_delta = theta_adam.mech_raw .- theta_base.mech_raw
          adam_group_probe = group_probe_dloss(theta_base, adam_p_delta, adam_m_delta, loss, dL_adam)
          if adam_group_probe.reason_full == "" &&
             adam_group_probe.reason_net == "" &&
             adam_group_probe.reason_mech == ""
            tprintln("  adam-group-probe: dL_full=", fmt_e(adam_group_probe.dL_full, sigdigits=3),
              " dL_net=", fmt_e(adam_group_probe.dL_net, sigdigits=3),
              " dL_mech=", fmt_e(adam_group_probe.dL_mech, sigdigits=3),
              " cross=", fmt_e(adam_group_probe.interaction, sigdigits=3))
            tprintln("  adam-group-step-norms: full=", fmt_e(adam_group_probe.full_norm, sigdigits=3),
              " net=", fmt_e(adam_group_probe.net_norm, sigdigits=3),
              " mech=", fmt_e(adam_group_probe.mech_norm, sigdigits=3))
          else
            tprintln("  adam-group-probe: full=",
              (adam_group_probe.reason_full == "" ? "ok" : adam_group_probe.reason_full),
              " net=",
              (adam_group_probe.reason_net == "" ? "ok" : adam_group_probe.reason_net),
              " mech=",
              (adam_group_probe.reason_mech == "" ? "ok" : adam_group_probe.reason_mech))
          end
        end
      end

    end

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
      end
    end

    if plateau_early_stop
      push!(recent_losses, loss)
      if length(recent_losses) > plateau_window
        popfirst!(recent_losses)
      end
    end
    if loss < good_enough_loss
      tprintln("  early-stop: good_enough (loss < 1e-10)")
      return theta, loss, "good_enough", (
        epoch1_loss=epoch1_loss,
        epoch1_gnorm_raw=epoch1_gnorm_raw,
        failure_epoch=failure_epoch,
        failure_reason=failure_reason
      )
    end
    if plateau_early_stop && length(recent_losses) == plateau_window &&
       abs(recent_losses[end] - recent_losses[1]) < plateau_tol
      tprintln("  early-stop: plateau_5ep (|Δloss| < 1e-6 over 5 epochs)")
      return theta, loss, "plateau_5ep", (
        epoch1_loss=epoch1_loss,
        epoch1_gnorm_raw=epoch1_gnorm_raw,
        failure_epoch=failure_epoch,
        failure_reason=failure_reason
      )
    end

    if !step_guard
      if trace_epoch_first_attempt && epoch == 1 && attempt == 1
        update_t0 = time_ns()
        tprintln("  trace[epoch1/attempt1]: direct_update_begin")
      end
      opt_state, theta = Optimisers.update(opt_state, theta, grad_update)
      if trace_epoch_first_attempt && epoch == 1 && attempt == 1
        tprintln("  trace[epoch1/attempt1]: direct_update_done dt=", trace_dt_str(update_t0))
      end
    else
      theta_base = deepcopy(theta)
      opt_state_base = deepcopy(opt_state)
      step_attempt = 0
      step_accepted = false

      while step_attempt <= step_retry_max
        step_attempt += 1
        trace_this_step = trace_epoch_first_attempt && epoch == 1 && attempt == 1 && step_attempt == 1
        step_fail_reason = ""
        trial_loss = Inf
        opt_input = deepcopy(opt_state_base)
        update_t0 = trace_this_step ? time_ns() : 0
        if trace_this_step
          tprintln("  trace[epoch1/attempt1]: step_update_begin")
        end
        opt_trial, theta_trial = Optimisers.update(opt_input, theta_base, grad_update)
        if trace_this_step
          tprintln("  trace[epoch1/attempt1]: step_update_done dt=", trace_dt_str(update_t0))
        end

        if monitor_ref !== nothing
          monitor_ref[] = nothing
        end
        try
          trial_t0 = trace_this_step ? time_ns() : 0
          if trace_this_step
            tprintln("  trace[epoch1/attempt1]: trial_loss_begin")
          end
          trial_loss = loss_fn(theta_trial)
          if trace_this_step
            tprintln("  trace[epoch1/attempt1]: trial_loss_done dt=", trace_dt_str(trial_t0),
              " trial_loss=", fmt_e(trial_loss, sigdigits=4))
          end
        catch ex
          step_fail_reason = "step_trial_exception: " * sprint(showerror, ex)
        end

        if step_fail_reason == ""
          trial_monitor_reason = monitor_failure_reason()
          if trial_monitor_reason != ""
            step_fail_reason = "step_trial_" * trial_monitor_reason
          elseif !isfinite(trial_loss)
            step_fail_reason = "step_trial_loss_nonfinite"
          elseif isfinite(step_max_loss_increase) && trial_loss > loss + step_max_loss_increase
            step_fail_reason = "step_trial_loss_jump"
          end
        end

        if step_fail_reason == ""
          opt_state = opt_trial
          theta = theta_trial
          step_accepted = true
          if step_attempt > 1
            tprintln("  step-guard: accepted after ", step_attempt - 1, " retry/reduction(s)",
              " | lr=", fmt_e(lr, sigdigits=3),
              " | trial_loss=", fmt_e(trial_loss, sigdigits=4))
          end
          break
        end

        if step_attempt > step_retry_max
          tprintln("  step-guard: rejected update after ", step_retry_max,
            " retries; keep previous parameters | reason=", step_fail_reason,
            " | lr=", fmt_e(lr, sigdigits=3))
          theta = theta_base
          opt_state = opt_state_base
          break
        end

        lr_new = max(lr * step_retry_lr_factor, epoch_retry_lr_floor)
        if lr_new < lr
          lr = lr_new
        end
        Optimisers.adjust!(opt_state_base, lr)
        tprintln("  step-guard retry ", step_attempt, "/", step_retry_max,
          " -- reason=", step_fail_reason,
          " | lr=", fmt_e(lr, sigdigits=3),
          " | scale[p_net=", fmt_e(grad_scale_p_net, sigdigits=3),
          ", mech_raw=", fmt_e(grad_scale_mech, sigdigits=3), "]")
      end

      if !step_accepted
        # Keep training with unchanged parameters; next epoch may recover with reduced lr.
        continue
      end
    end
  end

  return theta, loss_fn(theta), "", (
    epoch1_loss=epoch1_loss,
    epoch1_gnorm_raw=epoch1_gnorm_raw,
    failure_epoch=failure_epoch,
    failure_reason=failure_reason
  )
end

# x3 initial value (allowed: first contact)
x3_t0_val = ode_train[3, 1]

function run_stage2_sanity(; strict::Bool)
  tprintln("=== Stage2 SANITY (oracle F_hertz + true mech) ===")

  # Dummy NN params; the oracle override ignores them.
  rng = StableRNG(0)
  approximating_neural_network = build_nn(1, 3)
  p_net, st = Lux.setup(rng, approximating_neural_network)
  p_net = Flux.f64(p_net)
  p_net_vec, re_pnet = Optimisers.destructure(p_net)

  raw_init = [
    raw_from_value(ks_true, ks_bounds[1], ks_bounds[2]),
    raw_from_value(cs_true, cs_bounds[1], cs_bounds[2])
  ]
  theta0 = ComponentVector(p_net=p_net_vec, mech_raw=raw_init)

  l2_weight = 0.0
  sanity_monitor = Ref{Any}(nothing)

  loss = loss_single_or_ms(theta0, ode_train, x2dot_train, contact_train, times_train,
    state12_scale_full, x2dot_scale_full, x3_scale,
    use_multiple_shooting, 10, 1e-3,
    nothing, st, known_pars, x3_t0_val,
    l2_weight, re_pnet, sanity_monitor)

  tprintln("  sanity loss=", @sprintf("%.4e", loss))
  if sanity_monitor[] !== nothing
    diag = sanity_monitor[]
    tprintln("  sanity parts: state=", fmt_loss_part(diag.state, true, sigdigits=3),
      " x2dot=", fmt_loss_part(diag.x2dot, diag.x2dot_enabled, sigdigits=3),
      " x3r=", fmt_loss_part(diag.x3_range, diag.x3_range_enabled, sigdigits=3),
      " cont=", fmt_loss_part(diag.cont, diag.cont_enabled, sigdigits=3),
      " l2=", fmt_loss_part(diag.l2, diag.l2_enabled, sigdigits=3))
    tprintln("  sanity rec: x1=", fmt_f(diag.x1_rec, digits=2),
      "% x3=", fmt_f(diag.x3_rec, digits=2), "%")
    tprintln("  sanity nn: F_hertz err=", fmt_f(diag.fhertz_err, digits=2), "%")
    tprintln("  sanity nn raw: min=", fmt_e(diag.raw_min, sigdigits=3),
      " max=", fmt_e(diag.raw_max, sigdigits=3),
      " span=", fmt_e(diag.raw_span, sigdigits=3),
      " mean=", fmt_e(diag.raw_mean, sigdigits=3),
      " neg=", fmt_f(100 * diag.raw_neg_frac, digits=2), "%")
  end
  if !isfinite(loss)
    msg = "Stage2 sanity check failed: non-finite loss"
    if strict
      error(msg)
    else
      tprintln("  WARNING: ", msg)
    end
  else
    tprintln("=== Stage2 SANITY OK ===")
  end
  return loss
end

if selftest
  run_stage2_sanity(; strict=true)
  tprintln("=== SELFTEST OK ===")
else
  if stage2_sanity
    run_stage2_sanity(; strict=false)
  else
    tprintln("Stage2 sanity check skipped")
  end

  l2_grid = [0.0]
  stage2_log_every = max(1, parse(Int, get(ENV, "HNODECB_STAGE2_LOG_EVERY", "10")))
  stage2_init_retries = max(1, parse(Int, get(ENV, "HNODECB_STAGE2_INIT_RETRIES", "200")))
  stage2_init_retry_log_every = max(1, parse(Int, get(ENV, "HNODECB_STAGE2_INIT_RETRY_LOG_EVERY", "25")))
  tprintln("L2 grid: ", l2_grid)

  trial_parameters = []
  rng_global = StableRNG(0)

  for (local_idx, ci) in enumerate(candidate_indices)
    cand = top_candidates[ci]
    params = cand.params
    ks0 = params["ks0"]
    cs0 = params["cs0"]
    num_hidden_layers = get(params, "num_hidden_layers", nn_fixed_num_hidden_layers)
    num_hidden_nodes = get(params, "num_hidden_nodes", nn_fixed_num_hidden_nodes)
    ms_group_size = get(params, "ms_group_size", 50)
    ms_continuity_term = get(params, "ms_continuity_term", 1e-3)

    for l2_weight in l2_grid
      warm_init_vec = nothing
      warm_rank_used = 0
      g_nn_current = stage2_use_gnn ? stage2_gnn_default : 1.0
      if stage2_nn_warm_enabled && haskey(stage2_nn_warm_by_base_rank, ci)
        warm_recs = stage2_nn_warm_by_base_rank[ci]
        if !isempty(warm_recs)
          warm_rank_used = min(stage2_nn_warm_rank, length(warm_recs))
          warm_pick = warm_recs[warm_rank_used]
          warm_init_vec = copy(warm_pick.p_net_vec)
          num_hidden_layers = warm_pick.num_hidden_layers
          num_hidden_nodes = warm_pick.num_hidden_nodes
          if stage2_use_gnn && hasproperty(warm_pick, :g_nn)
            g_nn_current = warm_pick.g_nn
          end
        end
      end
      warm_used = warm_init_vec !== nothing
      approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)
      p_net_init_raw, st = Lux.setup(StableRNG(0), approximating_neural_network)
      p_net_init = Flux.f64(p_net_init_raw)
      p_net_template_vec, re_pnet = Optimisers.destructure(p_net_init)

      raw_init = [
        raw_from_value(ks0, ks_bounds[1], ks_bounds[2]),
        raw_from_value(cs0, cs_bounds[1], cs_bounds[2])
      ]
      train_monitor = Ref{Any}(nothing)

      function loss_fn(theta)
        loss_single_or_ms(theta, ode_train, x2dot_train, contact_train, times_train,
          state12_scale_full, x2dot_scale_full, x3_scale,
          use_multiple_shooting, ms_group_size, ms_continuity_term,
          approximating_neural_network, st, known_pars, x3_t0_val,
          l2_weight, re_pnet, train_monitor; nn_gain=g_nn_current)
      end

      lr_force = env_float("HNODECB_STAGE2_LR_FORCE", NaN)
      if isfinite(lr_force)
        lr_init = lr_force
      else
        lr_init = get(params, "learning_rate_adam", NaN)
        if !isfinite(lr_init)
          lr_init = env_float("HNODECB_STAGE2_LR_INIT", NaN)
        end
        if !isfinite(lr_init)
          lr_init = rand_loguniform(rng_global, 1e-5, 1e-2)
        end
      end

      tprintln("Stage2 candidate ", ci, "/", length(top_candidates),
        " (local ", local_idx, "/", length(candidate_indices), ")",
        " start -- l2=", @sprintf("%.2e", l2_weight),
        " lr=", @sprintf("%.2e", lr_init),
        " layers=", num_hidden_layers,
        " nodes=", num_hidden_nodes,
        " g_nn=", @sprintf("%.2e", g_nn_current),
        " warm=", warm_used,
        (warm_used ? " (rank=" * string(warm_rank_used) * ")" : ""))

      theta_best = nothing
      train_loss = Inf
      reason = "init_not_run"
      init_attempts = 0
      epoch1_loss = Inf
      epoch1_gnorm = NaN
      init_reason = "uninitialized"
      init_source = "none"
      while init_attempts < stage2_init_retries
        init_attempts += 1
        p_net_try_vec = nothing
        init_source = "random"
        if init_attempts == 1 && warm_init_vec !== nothing
          if length(warm_init_vec) == length(p_net_template_vec)
            p_net_try_vec = copy(warm_init_vec)
            init_source = "warm_rank_" * string(warm_rank_used)
          else
            init_source = "random_after_warm_mismatch"
          end
        end
        if p_net_try_vec === nothing
          rng = StableRNG(abs(rand(rng_global, Int)))
          p_net_rand, _ = Lux.setup(rng, approximating_neural_network)
          p_net_rand = Flux.f64(p_net_rand)
          p_net_try_vec, _ = Optimisers.destructure(p_net_rand)
        end
        theta0 = ComponentVector(p_net=copy(p_net_try_vec), mech_raw=raw_init)
        train_monitor[] = nothing
        theta_try_best, train_loss_try, reason_try, run_meta = run_adam(theta0, loss_fn;
          lr_init=lr_init, maxiters=500, log_every=stage2_log_every, monitor_ref=train_monitor,
          init_source=init_source, init_attempt=init_attempts, init_attempt_max=stage2_init_retries)
        epoch1_loss = run_meta.epoch1_loss
        epoch1_gnorm = run_meta.epoch1_gnorm_raw

        if run_meta.failure_epoch == 1
          init_reason = run_meta.failure_reason
          if init_attempts <= 3 || (init_attempts % stage2_init_retry_log_every == 0) || init_attempts == stage2_init_retries
            tprintln("  init retry ", init_attempts, "/", stage2_init_retries,
              " rejected -- epoch1 loss=", fmt_e(epoch1_loss, sigdigits=4),
              " grad_norm=", fmt_e(epoch1_gnorm, sigdigits=3),
              " | reason=", init_reason,
              " | source=", init_source)
            if train_monitor[] !== nothing
              diag = train_monitor[]
              tprintln("    epoch1 nn raw: min=", fmt_e(diag.raw_min, sigdigits=3),
                " max=", fmt_e(diag.raw_max, sigdigits=3),
                " span=", fmt_e(diag.raw_span, sigdigits=3),
                " mean=", fmt_e(diag.raw_mean, sigdigits=3),
                " neg=", fmt_f(100 * diag.raw_neg_frac, digits=2), "%")
            end
          end
          continue
        end

        theta_best = theta_try_best
        train_loss = train_loss_try
        reason = reason_try
        break
      end

      if theta_best === nothing
        tprintln("  init failed after ", stage2_init_retries,
          " attempts -- skipping candidate (last reason=", init_reason, ")")
        ks_hat = bound_param(raw_init[1], ks_bounds[1], ks_bounds[2])
        cs_hat = bound_param(raw_init[2], cs_bounds[1], cs_bounds[2])
        ks_err_pct = rel_err_pct(ks_hat, ks_true, scale_eps)
        cs_err_pct = rel_err_pct(cs_hat, cs_true, scale_eps)
        val_nn_err = (train_monitor[] !== nothing && hasproperty(train_monitor[], :fhertz_err)) ? train_monitor[].fhertz_err : NaN
        params_out = Dict{Any, Any}(
          "cand_rank" => ci,
          "trial_id" => length(trial_parameters) + 1,
          "ks0" => ks0,
          "cs0" => cs0,
          "num_hidden_layers" => num_hidden_layers,
          "num_hidden_nodes" => num_hidden_nodes,
          "ms_group_size" => ms_group_size,
          "ms_continuity_term" => ms_continuity_term,
          "l2_regularization" => l2_weight,
          "learning_rate_adam" => lr_init,
          "g_nn" => g_nn_current,
          "init_attempts" => init_attempts,
          "init_source" => init_source,
          "nn_warm_enabled" => stage2_nn_warm_enabled,
          "nn_warm_rank" => warm_rank_used,
          "nn_warm_used" => warm_used,
          "nn_warm_file" => stage2_nn_warm_file,
          "init_failed" => true,
          "init_failure_reason" => init_reason
        )
        push!(trial_parameters, (
          loss=Inf,
          train_loss=Inf,
          val_loss=Inf,
          params=params_out,
          val_parts=train_monitor[],
          ks_hat=ks_hat,
          cs_hat=cs_hat,
          ks_err_pct=ks_err_pct,
          cs_err_pct=cs_err_pct,
          val_nn_err=val_nn_err
        ))
        continue
      end

      if startswith(reason, "epoch_retry_exhausted:") ||
         reason == "train_loss_nonfinite" ||
         reason == "grad_norm_nonfinite" ||
         startswith(reason, "exception:")
        train_loss = Inf
      end

      val_monitor = Ref{Any}(nothing)
      val_loss = loss_single_or_ms(theta_best, ode_val, x2dot_val, contact_val, times_val,
        state12_scale_full, x2dot_scale_full, x3_scale,
        use_multiple_shooting, ms_group_size, ms_continuity_term,
        approximating_neural_network, st, known_pars, x3_t0_val,
        0.0, re_pnet, val_monitor; nn_gain=g_nn_current)

      ks_hat = bound_param(theta_best.mech_raw[1], ks_bounds[1], ks_bounds[2])
      cs_hat = bound_param(theta_best.mech_raw[2], cs_bounds[1], cs_bounds[2])
      ks_err_pct = rel_err_pct(ks_hat, ks_true, scale_eps)
      cs_err_pct = rel_err_pct(cs_hat, cs_true, scale_eps)
      val_nn_err = (val_monitor[] !== nothing && hasproperty(val_monitor[], :fhertz_err)) ? val_monitor[].fhertz_err : NaN
      trial_id = length(trial_parameters) + 1

      params_out = Dict{Any, Any}(
        "cand_rank" => ci,
        "trial_id" => trial_id,
        "ks0" => ks0,
        "cs0" => cs0,
        "num_hidden_layers" => num_hidden_layers,
        "num_hidden_nodes" => num_hidden_nodes,
        "ms_group_size" => ms_group_size,
        "ms_continuity_term" => ms_continuity_term,
        "l2_regularization" => l2_weight,
        "learning_rate_adam" => lr_init,
        "g_nn" => g_nn_current,
        "init_attempts" => init_attempts,
        "init_source" => init_source,
        "nn_warm_enabled" => stage2_nn_warm_enabled,
        "nn_warm_rank" => warm_rank_used,
        "nn_warm_used" => warm_used,
        "nn_warm_file" => stage2_nn_warm_file
      )
      push!(trial_parameters, (
        loss=val_loss,
        train_loss=train_loss,
        val_loss=val_loss,
        params=params_out,
        val_parts=val_monitor[],
        ks_hat=ks_hat,
        cs_hat=cs_hat,
        ks_err_pct=ks_err_pct,
        cs_err_pct=cs_err_pct,
        val_nn_err=val_nn_err
      ))

      tprintln("Stage2 candidate ", ci, "/", length(top_candidates),
        " (local ", local_idx, "/", length(candidate_indices), ")",
        " done -- train=", @sprintf("%.4e", train_loss),
        " val=", @sprintf("%.4e", val_loss),
        (reason == "" ? "" : " | reason=" * reason))
      if train_monitor[] !== nothing
        diag = train_monitor[]
        tprintln("  final parts: state=", fmt_loss_part(diag.state, true, sigdigits=3),
          " x2dot=", fmt_loss_part(diag.x2dot, diag.x2dot_enabled, sigdigits=3),
          " x3r=", fmt_loss_part(diag.x3_range, diag.x3_range_enabled, sigdigits=3),
          " cont=", fmt_loss_part(diag.cont, diag.cont_enabled, sigdigits=3),
          " l2=", fmt_loss_part(diag.l2, diag.l2_enabled, sigdigits=3))
        tprintln("  final rec: x1=", fmt_f(diag.x1_rec, digits=2),
          "% x3=", fmt_f(diag.x3_rec, digits=2), "%")
        tprintln("  final nn: F_hertz err=", fmt_f(diag.fhertz_err, digits=2), "%")
        tprintln("  final nn raw: min=", fmt_e(diag.raw_min, sigdigits=3),
          " max=", fmt_e(diag.raw_max, sigdigits=3),
          " span=", fmt_e(diag.raw_span, sigdigits=3),
          " mean=", fmt_e(diag.raw_mean, sigdigits=3),
          " neg=", fmt_f(100 * diag.raw_neg_frac, digits=2), "%")
      end
    end
  end

  sorted = sort(trial_parameters, by = r -> r.loss)
  selected = sorted[1:min(stage2_final_topk, length(sorted))]
  tprintln("Stage2 final selection: top-", length(selected), " of ", length(sorted))
  tprintln("Stage2 top-", length(selected), " summary (train/val):")
  for (i, rec) in enumerate(selected)
    tprintln("  Rank ", i,
      " -- train=", @sprintf("%.4e", rec.train_loss),
      " val=", @sprintf("%.4e", rec.val_loss),
      " | l2=", @sprintf("%.2e", rec.params["l2_regularization"]),
      " cand=", rec.params["cand_rank"])
  end

  best_rec = isempty(selected) ? nothing : selected[1]
  serialize(result_folder * "/" * result_name_string, (
    study=nothing,
    trial_parameters=trial_parameters,
    results=sorted,
    selected=selected,
    best=best_rec,
    bounds=(ks=ks_bounds, cs=cs_bounds),
    true_values=(ks=ks_true, cs=cs_true),
    use_multiple_shooting=use_multiple_shooting,
    l2_grid=l2_grid,
    shard_index=stage2_shard_index,
    shard_count=stage2_shard_count,
    candidate_filter=(candidate_filter_indices === nothing ? Int[] : candidate_filter_indices),
    result_basename=stage2_result_basename,
    x3_obs_fraction=0.0,
    error_level=error_level
  ))

  tprintln("Stage2 done. Best loss=", best_rec === nothing ? "Inf" : @sprintf("%.4e", best_rec.loss))
end
