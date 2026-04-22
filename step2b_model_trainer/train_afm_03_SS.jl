#= 
Script to train AFM DMT-KV model parameters on the e0.0 dataset.
Scenario 03: sparse x3 observations enabled by default, single shooting.
Net contact-force term is replaced by a neural network.
Physical prior only: |x3| <= 20 nm.
=#

cd(@__DIR__)

using ComponentArrays, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Statistics, Printf, Dates
using Flux
using Lux
using Optimization, OptimizationOptimisers
using SciMLSensitivity, DiffEqFlux
using Zygote

include("_local_lbfgs.jl")

timestamp_now() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
function tprintln(args...)
  println(args...)
  flush(stdout)
end

parse_env_int(name) = begin
  raw = strip(get(ENV, name, ""))
  isempty(raw) && return nothing
  val = tryparse(Int, raw)
  val === nothing && error("Environment variable " * name * " must be an integer, got: " * raw)
  return val
end

parse_env_float(name, default) = begin
  raw = strip(get(ENV, name, ""))
  isempty(raw) && return default
  val = tryparse(Float64, raw)
  val === nothing && return default
  return val
end

function field_or(x, key::Symbol, default)
  if x isa NamedTuple
    return hasproperty(x, key) ? getproperty(x, key) : default
  elseif x isa AbstractDict
    return haskey(x, key) ? x[key] : default
  else
    return hasproperty(x, key) ? getproperty(x, key) : default
  end
end

result_name_string = strip(get(ENV, "HNODECB_STEP2B_RESULT_NAME", "afm_03.jld"))
isempty(result_name_string) && error("HNODECB_STEP2B_RESULT_NAME must not be empty.")
step2b_sanity = get(ENV, "HNODECB_STEP2B_SANITY", "1") == "1"
step2b_resume_enabled = get(ENV, "HNODECB_STEP2B_RESUME", "1") != "0"
step2b_checkpoint_every = max(0, tryparse(Int, get(ENV, "HNODECB_STEP2B_CHECKPOINT_EVERY", "5")) === nothing ? 5 : parse(Int, get(ENV, "HNODECB_STEP2B_CHECKPOINT_EVERY", "5")))

folder_name = strip(get(ENV, "HNODECB_STEP2B_RESULT_DIR", "res_afm_03"))
isempty(folder_name) && error("HNODECB_STEP2B_RESULT_DIR must not be empty.")
if !isdir(folder_name)
  mkpath(folder_name)
end
step2b_result_path = normpath(joinpath(folder_name, result_name_string))

window_start_idx = parse_env_int("HNODECB_STEP2B_WINDOW_START")
window_len = parse_env_int("HNODECB_STEP2B_WINDOW_LEN")
window_stop_idx = parse_env_int("HNODECB_STEP2B_WINDOW_STOP")
window_label = strip(get(ENV, "HNODECB_STEP2B_WINDOW_LABEL", ""))
window_role = strip(get(ENV, "HNODECB_STEP2B_WINDOW_ROLE", ""))
stage2light_filter_label = strip(get(ENV, "HNODECB_STEP2B_STAGE2LIGHT_FILTER_LABEL", window_label))
stage2light_filter_role = strip(get(ENV, "HNODECB_STEP2B_STAGE2LIGHT_FILTER_ROLE", window_role))

if (window_start_idx === nothing) != (window_len === nothing)
  error("HNODECB_STEP2B_WINDOW_START and HNODECB_STEP2B_WINDOW_LEN must be set together.")
end
if window_start_idx !== nothing && window_start_idx < 1
  error("HNODECB_STEP2B_WINDOW_START must be >= 1.")
end
if window_len !== nothing && window_len < 1
  error("HNODECB_STEP2B_WINDOW_LEN must be >= 1.")
end

error_level = "e0.0"

include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_functions.jl")
include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")

stage2_input_basename = strip(get(ENV, "HNODECB_STEP2B_STAGE2_INPUT_BASENAME", "afm_param_stage2light_windowed_03.jld"))
isempty(stage2_input_basename) && error("HNODECB_STEP2B_STAGE2_INPUT_BASENAME must not be empty.")
stage2_path = "../step2a_hyperparameter_tuning/hyperparameter_tuning_second_stage/results_afm/" * stage2_input_basename

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

function build_nn(num_hidden_layers::Int, num_hidden_nodes::Int)
  hidden = 2 ^ num_hidden_nodes
  layers = Any[]
  push!(layers, Lux.Dense(3, hidden, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform, use_bias=true))
  for _ in 1:num_hidden_layers
    push!(layers, Lux.Dense(hidden, hidden, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform, use_bias=true))
  end
  push!(layers, Lux.Dense(hidden, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform, use_bias=true))
  return Lux.Chain(layers...)
end

function setup_nn_bundle(num_hidden_layers::Int, num_hidden_nodes::Int)
  rng_net = Random.default_rng()
  Random.seed!(rng_net, 0)
  net = build_nn(num_hidden_layers, num_hidden_nodes)
  p_net_init_raw, st_local = Lux.setup(rng_net, net)
  return net, st_local, Flux.f64(ComponentArray(p_net_init_raw))
end

function stage2_window_matches(wr)
  if !isempty(stage2light_filter_label)
    return haskey(wr, :window) && haskey(wr.window, :label) &&
      string(wr.window.label) == stage2light_filter_label
  elseif !isempty(stage2light_filter_role)
    return haskey(wr, :window) && haskey(wr.window, :role) &&
      string(wr.window.role) == stage2light_filter_role
  end
  return true
end

function params_get_int(params, key, default)
  if params isa AbstractDict && haskey(params, key)
    v = tryparse(Int, string(params[key]))
    return v === nothing ? default : v
  end
  return default
end

function params_get_float(params, key, default)
  if params isa AbstractDict && haskey(params, key)
    v = tryparse(Float64, string(params[key]))
    return v === nothing ? default : v
  end
  return default
end

function result_get_float(result_like, key::Symbol, default)
  if result_like isa AbstractDict
    return haskey(result_like, key) ? Float64(result_like[key]) : default
  elseif result_like isa NamedTuple
    return hasproperty(result_like, key) ? Float64(getproperty(result_like, key)) : default
  end
  return hasproperty(result_like, key) ? Float64(getproperty(result_like, key)) : default
end

function read_primary_stage2_nn_config()
  default_cfg = (num_hidden_layers=0, num_hidden_nodes=2, p_net_vec=nothing, nn_gain=1.0)
  if !isfile(stage2_path)
    return default_cfg
  end
  stage2 = deserialize(stage2_path)
  haskey(stage2, :window_results) || return default_cfg
  candidates = NamedTuple[]
  for wr in stage2.window_results
    stage2_window_matches(wr) || continue
    haskey(wr, :best) || continue
    b = wr.best
    params = haskey(b, :params) ? b.params : Dict{Any, Any}()
    push!(candidates, (
      loss=haskey(b, :loss) ? Float64(b.loss) : Inf,
      num_hidden_layers=params_get_int(params, "num_hidden_layers", 0),
      num_hidden_nodes=params_get_int(params, "num_hidden_nodes", 2),
      p_net_vec=haskey(b, :p_net_vec) ? copy(Vector{Float64}(b.p_net_vec)) : nothing,
      nn_gain=result_get_float(b, :nn_gain, params_get_float(params, "nn_gain", 1.0)),
    ))
  end
  isempty(candidates) && return default_cfg
  best = sort(candidates, by = c -> c.loss)[1]
  return (
    num_hidden_layers=best.num_hidden_layers,
    num_hidden_nodes=best.num_hidden_nodes,
    p_net_vec=best.p_net_vec,
    nn_gain=best.nn_gain,
  )
end

primary_stage2_nn = read_primary_stage2_nn_config()
approximating_neural_network, st, p_net_init = setup_nn_bundle(
  primary_stage2_nn.num_hidden_layers,
  primary_stage2_nn.num_hidden_nodes,
)
if primary_stage2_nn.p_net_vec !== nothing && length(primary_stage2_nn.p_net_vec) == length(p_net_init)
  p_net_init .= primary_stage2_nn.p_net_vec
end

step2b_dynamic_gnn = get(ENV, "HNODECB_STEP2B_DYNAMIC_GNN", get(ENV, "HNODECB_STAGE2_DYNAMIC_GNN", "1")) == "1"
step2b_fixed_gnn = get(ENV, "HNODECB_STEP2B_FIXED_GNN", get(ENV, "HNODECB_STAGE2_FIXED_GNN", "0")) == "1"
step2b_gnn_q = let
  p = parse_env_float("HNODECB_STEP2B_GNN_Q", parse_env_float("HNODECB_STAGE2_GNN_Q", 0.95))
  (!isfinite(p) || p <= 0.0 || p >= 1.0) ? 0.95 : p
end
step2b_gnn_min = let
  p = parse_env_float("HNODECB_STEP2B_GNN_MIN", parse_env_float("HNODECB_STAGE2_GNN_MIN", 1e-12))
  (!isfinite(p) || p <= 0.0) ? 1e-12 : p
end
step2b_gnn_max = let
  p = parse_env_float("HNODECB_STEP2B_GNN_MAX", parse_env_float("HNODECB_STAGE2_GNN_MAX", 1e12))
  (!isfinite(p) || p <= 0.0) ? 1e12 : p
end
step2b_gnn_log10_update_min = let
  p = parse_env_float("HNODECB_STEP2B_GNN_LOG10_UPDATE_MIN", parse_env_float("HNODECB_STAGE2_GNN_LOG10_UPDATE_MIN", 1.0))
  (!isfinite(p) || p < 0.0) ? 1.0 : p
end
current_nn_gain_ref = Ref(Float64(primary_stage2_nn.nn_gain))

# Load data
ode_data = deserialize("../datasets/e0.0/data/ode_data_afm_dmt_kv.jld")
solution_dataframe = deserialize("../datasets/e0.0/data/pert_df_afm_dmt_kv.jld")

# Keep only data after first contact
contact_idx = findfirst(solution_dataframe.contact .== 1)
if contact_idx === nothing
  error("No contact point found in the AFM dataset.")
end

solution_dataframe = solution_dataframe[contact_idx:end, :]
ode_data = ode_data[:, contact_idx:end]

window_metadata = (
  enabled=false,
  start_idx=1,
  stop_idx=size(ode_data, 2),
  len=size(ode_data, 2),
  label=window_label,
  role=window_role
)

if window_start_idx !== nothing
  start_idx = window_start_idx
  stop_idx = start_idx + window_len - 1
  if window_stop_idx !== nothing && window_stop_idx != stop_idx
    error("HNODECB_STEP2B_WINDOW_STOP=" * string(window_stop_idx) *
      " does not match start/len derived stop=" * string(stop_idx))
  end
  n_post = size(ode_data, 2)
  if stop_idx > n_post
    error("Requested step2b window [" * string(start_idx) * ", " * string(stop_idx) *
      "] exceeds post-contact trajectory length " * string(n_post))
  end
  solution_dataframe = solution_dataframe[start_idx:stop_idx, :]
  ode_data = ode_data[:, start_idx:stop_idx]
  window_metadata = (
    enabled=true,
    start_idx=start_idx,
    stop_idx=stop_idx,
    len=window_len,
    label=window_label,
    role=window_role
  )
  tprintln("=== Step2b (03): windowed slice enabled ===")
  tprintln("Window label=", isempty(window_label) ? "None" : window_label,
    " | role=", isempty(window_role) ? "None" : window_role,
    " | post_window=[", start_idx, ", ", stop_idx, "] len=", window_len)
end

tmp_steps = solution_dataframe.t
x2dot_data = solution_dataframe.x2dot
true_contact_weight_at_time = make_contact_weight_lookup(tmp_steps, solution_dataframe.s)

function nn_input_from_state(u)
  return u[1:3]
end

function nn_input_from_values(x1, x2, x3)
  return [x1, x2, x3]
end

function make_uode_derivative_function_contact_net_nn_local(appr_neural_network, state)
  f(u, p, t) =
    let appr_neural_network = appr_neural_network, st_local = state
      ode_par = p.ode_par
      k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs = ode_par

      s = dist + u[1] - u[3]
      delta = softplus(-s, adhesion_transition)
      delta = ifelse(delta > 0.0, delta, 0.0)
      w_pred = contact_weight(s, adhesion_transition)

      Fad_eff = Fad * w_pred
      F_contact = if appr_neural_network === nothing
        (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad_eff
      else
        nn_in = nn_input_from_state(u)
        uhat = appr_neural_network(nn_in, p.p_net, st_local)[1]
        current_nn_gain_ref[] * uhat[1] * w_pred
      end

      x1dot = u[2]
      x2dot = (Fd * cos(wd * t) - k * u[1] - c * u[2] + F_contact) / m
      x3dot = (-F_contact - ks * u[3]) / cs
      return [x1dot, x2dot, x3dot]
    end
end

uode_derivative_function = make_uode_derivative_function_contact_net_nn_local(
  approximating_neural_network,
  st,
)

# Multiple shooting settings
small_group_size = 100
large_group_size = 800
boundary_window = 100
min_group_size = 10
continuity_term = 0.001
use_multiple_shooting = false
use_u0_x3_param = false
# Initial x3 is known at first contact and is penalized explicitly.
# x3(t0) penalty disabled: x3 at first contact is treated as fixed known initial condition.
use_continuity_loss = false

# L2 regularization on NN weights (toggle)
use_l2_regularization = get(ENV, "HNODECB_STEP2B_USE_L2", "0") == "1"
l2_weight = use_l2_regularization ? parse(Float64, get(ENV, "HNODECB_STEP2B_L2_WEIGHT", "1e-6")) : 0.0

# Contact weighting
contact_loss_weight = 4.27
noncontact_loss_weight = 1.0

# x3 range prior (physical prior, no x3 data used)
x3_range_center = 0.0
x3_range_amp = 100e-9
x3_range_eps = 1e-9
x3_range_weight = 1.0
fts_range_amp = 1e-8
fts_range_eps = 1e-10
fts_range_weight = parse(Float64, get(ENV, "HNODECB_STEP2B_FTS_RANGE_WEIGHT", get(ENV, "HNODECB_FTS_RANGE_WEIGHT", "1.0")))
# If u0_x3 is ever enabled, keep its bounds consistent with the prior.
x3_u0_center = x3_range_center
x3_u0_amp = x3_range_amp
x3_u0_weight = 50.0


integrator = Rosenbrock23(autodiff=false)
abstol = 1e-8
reltol = 1e-8
sensealg = GaussAdjoint(autojacvec=ZygoteVJP())
tprintln("Solver: Rosenbrock23(autodiff=false) | rhs=out-of-place | sensealg=GaussAdjoint(ZygoteVJP())")

datasize = size(ode_data, 2)
tspan = (tmp_steps[1], tmp_steps[end])

# Normalization scales
# Use only observable signals for scaling. For x3, use the physical prior amplitude.
scale_eps = 1e-9
state12_scale = vec(maximum(ode_data[1:2, :], dims=2) - minimum(ode_data[1:2, :], dims=2))
state12_scale = max.(state12_scale, scale_eps)
x2dot_scale = max(maximum(x2dot_data) - minimum(x2dot_data), scale_eps)
x3_scale = max(x3_range_amp, scale_eps)
fts_scale = max(fts_range_amp, scale_eps)

last_loss_components = Ref((state=0.0, x3=0.0, x3_u0=0.0, x2dot=0.0, x3_range=0.0, fts_range=0.0, continuity=0.0))
last_recon_metrics = Ref((x1=0.0, x3=0.0))
last_rollout_states = Ref{Any}(nothing)
last_nn_metrics = Ref((fcontact_err=NaN, raw_min=NaN, raw_max=NaN, raw_span=NaN, raw_mean=NaN, raw_neg_frac=NaN))

ranges_ref = Ref(UnitRange{Int}[])
range_is_contact_ref = Ref(Bool[])
x3_u0_target_ref = Ref(0.0)
segment_times_ref = Ref(Float64[])
control_segments_ref = Ref(Int[])
control_times_ref = Ref(Float64[])

function retcode_success(sol)
  return string(sol.retcode) == "Success"
end

function percent_error_pct(est, truth)
  return 100.0 * abs(log10(est / truth))
end

function relative_rmse_pct(pred, truth, eps)
  denom = sqrt(mean(abs2, truth)) + eps
  return 100.0 * sqrt(mean(abs2.(pred .- truth))) / denom
end

logmsg(msg) = tprintln("[", timestamp_now(), "] ", msg)

function fmt_num(x; sigdigits=3)
  return (x isa Number && isfinite(x)) ? @sprintf("%.*e", sigdigits, x) : "None"
end

function fmt_pct(x; digits=2)
  return (x isa Number && isfinite(x)) ? @sprintf("%.*f", digits, x) : "None"
end

function _sum_structured_terms(values_iter)
  total = nothing
  for v in values_iter
    term = structured_l2_penalty(v)
    total = total === nothing ? term : total + term
  end
  return total === nothing ? 0.0 : total
end

function structured_l2_penalty(x)
  if x isa Number
    return abs2(x)
  elseif x isa ComponentVector
    names = propertynames(x)
    return isempty(names) ? sum(abs2, x) : _sum_structured_terms((getproperty(x, name) for name in names))
  elseif x isa NamedTuple
    return _sum_structured_terms(values(x))
  elseif x isa AbstractArray
    return sum(abs2, x)
  else
    names = propertynames(x)
    return isempty(names) ? 0.0 : _sum_structured_terms((getproperty(x, name) for name in names))
  end
end

function contact_net_true_from_states_local(u_mat)
  out = Vector{Float64}(undef, size(u_mat, 2))
  @inbounds for j in axes(u_mat, 2)
    s = dist + u_mat[1, j] - u_mat[3, j]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    w_true = contact_weight(s, adhesion_transition)
    f_hertz = (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
    out[j] = f_hertz - Fad * w_true
  end
  return out
end

function contact_net_pred_and_raw_from_states_local(u_mat, times, p_net_struct)
  fnet_pred = Vector{Float64}(undef, size(u_mat, 2))
  raw = Vector{Float64}(undef, size(u_mat, 2))
  @inbounds for j in axes(u_mat, 2)
    s = dist + u_mat[1, j] - u_mat[3, j]
    w_pred = contact_weight(s, adhesion_transition)
    nn_in = nn_input_from_values(u_mat[1, j], u_mat[2, j], u_mat[3, j])
    uhat = approximating_neural_network(nn_in, p_net_struct, st)[1]
    raw[j] = uhat[1]
    fnet_pred[j] = current_nn_gain_ref[] * uhat[1] * w_pred
  end
  return fnet_pred, raw
end

rounded_log10_decade(v::Real) = (isfinite(v) && v > 0.0) ? round(Int, log10(v)) : nothing

function effective_contact_positions_from_truth_local(contact_true)
  if isempty(contact_true)
    return Int[]
  end
  abs_true = abs.(contact_true)
  max_true = maximum(abs_true)
  if !isfinite(max_true) || max_true <= 0.0
    return collect(eachindex(contact_true))
  end
  floor_val = 0.01 * max_true
  eff_pos = findall(v -> isfinite(v) && v > floor_val, abs_true)
  return isempty(eff_pos) ? collect(eachindex(contact_true)) : eff_pos
end

function effective_contact_positions_local(contact_mask, contact_true)
  contact_pos = findall(contact_mask)
  if isempty(contact_pos)
    return Int[]
  end
  max_true = maximum(abs.(contact_true[contact_pos]))
  if !isfinite(max_true) || max_true <= 0.0
    return contact_pos
  end
  floor_val = 0.01 * max_true
  eff_pos = [k for k in contact_pos if abs(contact_true[k]) > floor_val]
  return isempty(eff_pos) ? contact_pos : eff_pos
end

function contact_net_monitor_metrics_local(u_mat, times, contact_mask, p_net_struct)
  contact_true = contact_net_true_from_states_local(u_mat)
  contact_pred, raw = contact_net_pred_and_raw_from_states_local(u_mat, times, p_net_struct)
  eff_pos = effective_contact_positions_local(contact_mask, contact_true)
  if isempty(eff_pos)
    return (fcontact_err=NaN, raw_min=NaN, raw_max=NaN, raw_span=NaN, raw_mean=NaN, raw_neg_frac=NaN)
  end
  raw_eff = raw[eff_pos]
  raw_min = minimum(raw_eff)
  raw_max = maximum(raw_eff)
  fcontact_err = relative_rmse_pct(contact_pred[eff_pos], contact_true[eff_pos], 0.0)
  raw_neg_frac = count(<(0.0), raw_eff) / length(raw_eff)
  return (
    fcontact_err=fcontact_err,
    raw_min=raw_min,
    raw_max=raw_max,
    raw_span=raw_max - raw_min,
    raw_mean=mean(raw_eff),
    raw_neg_frac=raw_neg_frac
  )
end

function decade_nn_gain_from_states_local(u_mat, times, p_net_struct;
  contact_true_ref=nothing, eff_pos_ref=nothing,
  q::Float64=0.95, gain_min::Float64=1e-12, gain_max::Float64=1e12,
  current_gain=nothing, log10_update_min::Float64=1.0)
  fallback_gain = (current_gain isa Real && isfinite(current_gain) && current_gain > 0.0) ? Float64(current_gain) : 1.0
  contact_true = contact_true_ref === nothing ? contact_net_true_from_states_local(u_mat) : contact_true_ref
  contact_pred, _ = contact_net_pred_and_raw_from_states_local(u_mat, times, p_net_struct)
  eff_pos = eff_pos_ref === nothing ? effective_contact_positions_from_truth_local(contact_true) : eff_pos_ref
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

function push_range!(ranges, labels, current, seg_start, seg_end, min_size)
  if seg_start > seg_end
    return
  end
  seg_len = seg_end - seg_start + 1
  if seg_len < min_size && !isempty(ranges) && labels[end] == current
    ranges[end] = first(ranges[end]):seg_end
  else
    push!(ranges, seg_start:seg_end)
    push!(labels, current)
  end
end

function split_and_push!(ranges, labels, current, seg_start, seg_end, chunk_size, min_size)
  idx = seg_start
  while idx <= seg_end
    seg_last = min(idx + chunk_size - 1, seg_end)
    push_range!(ranges, labels, current, idx, seg_last, min_size)
    idx = seg_last + 1
  end
end

function build_contact_ranges(contact::AbstractVector{Bool}, small_size::Int, large_size::Int, boundary_len::Int, min_size::Int)
  ranges = UnitRange{Int}[]
  labels = Bool[]
  n = length(contact)
  start = 1
  while start <= n
    current = contact[start]
    run_end = start
    while run_end < n && contact[run_end + 1] == current
      run_end += 1
    end

    run_start = start
    run_len = run_end - run_start + 1
    if current
      run_len = run_end - run_start + 1
      mid = run_start + (run_len ÷ 2) - 1
      if mid < run_start
        mid = run_start
      end
      if mid >= run_end
        push_range!(ranges, labels, current, run_start, run_end, min_size)
      else
        push_range!(ranges, labels, current, run_start, mid, min_size)
        push_range!(ranges, labels, current, mid + 1, run_end, min_size)
      end
    elseif boundary_len <= 0 || run_len <= 2 * boundary_len
      split_and_push!(ranges, labels, current, run_start, run_end, small_size, min_size)
    else
      left_end = run_start + boundary_len - 1
      right_start = run_end - boundary_len + 1
      split_and_push!(ranges, labels, current, run_start, left_end, small_size, min_size)
      split_and_push!(ranges, labels, current, left_end + 1, right_start - 1, large_size, min_size)
      split_and_push!(ranges, labels, current, right_start, run_end, small_size, min_size)
    end

    start = run_end + 1
  end
  return ranges, labels
end

function update_ranges!(contact::AbstractVector{Bool})
  ranges, labels = build_contact_ranges(contact, small_group_size, large_group_size, boundary_window, min_group_size)
  ranges_ref[] = ranges
  range_is_contact_ref[] = labels
end

function bound_u0_x3(u0_raw::AbstractVector)
  return x3_u0_center .+ x3_u0_amp .* tanh.(u0_raw)
end

function unbound_u0_x3(u0_bounded::AbstractVector)
  z = (u0_bounded .- x3_u0_center) ./ x3_u0_amp
  z = clamp.(z, -1 + 1e-6, 1 - 1e-6)
  return atanh.(z)
end

function update_x3_targets!()
  # Scenario 03: use the first-contact x3 value as the explicit x3 init target
  x3_u0_target_ref[] = ode_data[3, 1]
end

control_point_count = 20
control_weight_contact = 5.0
control_weight_boundary = 2.0
control_weight_noncontact = 1.0
use_interpolation = false

function weighted_sample_indices(weights::Vector{Float64}, k::Int, rng)
  if k <= 0
    return Int[]
  end
  keys = similar(weights)
  for i in eachindex(weights)
    w = weights[i]
    if w <= 0
      keys[i] = -Inf
    else
      keys[i] = rand(rng)^(1 / w)
    end
  end
  return partialsortperm(keys, 1:k; rev=true)
end

function update_control_points!()
  ranges = ranges_ref[]
  labels = range_is_contact_ref[]
  if isempty(ranges)
    segment_times_ref[] = Float64[]
    control_segments_ref[] = Int[]
    control_times_ref[] = Float64[]
    return
  end
  if !use_multiple_shooting
    segment_times_ref[] = [tmp_steps[1]]
    control_segments_ref[] = [1]
    control_times_ref[] = [tmp_steps[1]]
    return
  end
  segment_times = [tmp_steps[first(rg)] for rg in ranges]
  segment_times_ref[] = segment_times

  n = length(ranges)
  if !use_interpolation
    control_segments_ref[] = collect(1:n)
    control_times_ref[] = segment_times
    return
  end
  weights = zeros(Float64, n)
  for i in 1:n
    if labels[i]
      weights[i] = control_weight_contact
    else
      boundary = (i > 1 && labels[i - 1]) || (i < n && labels[i + 1])
      weights[i] = boundary ? control_weight_boundary : control_weight_noncontact
    end
  end

  k = min(control_point_count, n)
  ctrl_idx = sort(weighted_sample_indices(weights, k, Random.default_rng()))
  control_segments_ref[] = ctrl_idx
  control_times_ref[] = segment_times[ctrl_idx]
end

function interp_value(t, ctrl_times, ctrl_values)
  if isempty(ctrl_times)
    return 0.0
  end
  if t <= ctrl_times[1]
    return ctrl_values[1]
  elseif t >= ctrl_times[end]
    return ctrl_values[end]
  end
  idx = searchsortedlast(ctrl_times, t)
  if idx >= length(ctrl_times)
    return ctrl_values[end]
  end
  t0 = ctrl_times[idx]
  t1 = ctrl_times[idx + 1]
  v0 = ctrl_values[idx]
  v1 = ctrl_values[idx + 1]
  w = (t - t0) / (t1 - t0)
  return v0 + (v1 - v0) * w
end

function interp_u0_x3(times, ctrl_times, ctrl_values)
  return [interp_value(t, ctrl_times, ctrl_values) for t in times]
end

# Parameter bounds
ks_bounds = (0.005, 5.0) #true ks = 0.1
cs_bounds = (5e-8, 5e-6) #true cs = 2.4e-7

function sample_log_uniform(rng, lower, upper)
  return exp(rand(rng) * (log(upper) - log(lower)) + log(lower))
end

function sample_log_uniform_narrow(rng, lower, upper, log_span)
  log_lower = log(lower)
  log_upper = log(upper)
  log_mid = 0.5 * (log_lower + log_upper)
  log_min = max(log_lower, log_mid - log_span)
  log_max = min(log_upper, log_mid + log_span)
  return exp(rand(rng) * (log_max - log_min) + log_min)
end

function sample_log_band(rng, lower, upper, band::Symbol)
  log_lower = log(lower)
  log_upper = log(upper)
  log_mid = 0.5 * (log_lower + log_upper)
  log_range = log_upper - log_lower
  half = 0.5 * log_range
  if band == :center
    span = 0.3 * half
    return exp(rand(rng) * (2 * span) + (log_mid - span))
  elseif band == :mid
    inner = 0.3 * half
    outer = 0.6 * half
    if rand(rng) < 0.5
      return exp(rand(rng) * (outer - inner) + (log_mid - outer))
    end
    return exp(rand(rng) * (outer - inner) + (log_mid + inner))
  elseif band == :outer
    inner = 0.6 * half
    outer = 1.0 * half
    if rand(rng) < 0.5
      return exp(rand(rng) * (outer - inner) + (log_mid - outer))
    end
    return exp(rand(rng) * (outer - inner) + (log_mid + inner))
  end
  return sample_log_uniform(rng, lower, upper)
end

function band_for_run(run_id)
  if run_id == 1
    return :center
  elseif run_id == 2
    return :mid
  end
  return :outer
end

function initial_theta_from_guess(ks0, cs0)
  return log.([ks0, cs0])
end

function build_parameter_vector(theta)
  ks = exp(theta[1])
  cs = exp(theta[2])
  return [original_parameters[1:8]...; Estar; ks; cs]
end

function x2dot_rhs(u, p, t, p_net)
  k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs = p
  s = dist + u[1] - u[3]
  delta = softplus(-s, adhesion_transition)
  delta = ifelse(delta > 0.0, delta, 0.0)
  w_pred = contact_weight(s, adhesion_transition)
  Fad_eff = Fad * w_pred
  F_contact = if p_net === nothing
    (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad_eff
  else
    nn_in = nn_input_from_state(u)
    û = approximating_neural_network(nn_in, p_net, st)[1]
    current_nn_gain_ref[] * û[1] * w_pred
  end
  return (Fd * cos(wd * t) - k * u[1] - c * u[2] + F_contact) / m
end

function fts_pred_from_state(u, p, p_net)
  k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs = p
  s = dist + u[1] - u[3]
  delta = softplus(-s, adhesion_transition)
  delta = ifelse(delta > 0.0, delta, 0.0)
  w_pred = contact_weight(s, adhesion_transition)
  Fad_eff = Fad * w_pred
  return if p_net === nothing
    (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad_eff
  else
    nn_in = nn_input_from_state(u)
    û = approximating_neural_network(nn_in, p_net, st)[1]
    current_nn_gain_ref[] * û[1] * w_pred
  end
end

prob_pred = ODEProblem{false}(uode_derivative_function, original_u0, tspan, (p_net=p_net_init, ode_par=original_parameters))

function loss_multiple_shooting_with_u0(p_est, u0_x3, p_net; debug::Bool=false)
  ranges = ranges_ref[]
  range_is_contact = range_is_contact_ref[]
  ks_est = p_est[10]
  cs_est = p_est[11]
  if any(x -> !isfinite(x), p_est) ||
     ks_est < ks_bounds[1] || ks_est > ks_bounds[2] ||
     cs_est < cs_bounds[1] || cs_est > cs_bounds[2]
    if debug
      println("MS sanity fail: param guard. ks=", ks_est, " cs=", cs_est)
    end
    return Inf
  end
  if isempty(ranges)
    if debug
      println("MS sanity fail: no ranges built")
    end
    return Inf
  end
  if length(u0_x3) != length(ranges)
    if debug
      println("MS sanity fail: u0_x3 length mismatch")
    end
    return Inf
  end

  function unstable_check(dt, u, p, t)
    if any(x -> !isfinite(x), u) || any(abs.(u) .> 1e7)
      return true
    end
    return false
  end

  sols = Vector{Any}(undef, length(ranges))
  for (i, rg) in enumerate(ranges)
    u0 = [ode_data[1, first(rg)], ode_data[2, first(rg)], u0_x3[i]]
    sol = solve(
      remake(
        prob_pred;
        p=(p_net=p_net, ode_par=p_est),
        tspan=(tmp_steps[first(rg)], tmp_steps[last(rg)]),
        u0=u0
      ),
      integrator;
      saveat=tmp_steps[rg],
      reltol=reltol,
      abstol=abstol,
      sensealg=sensealg,
      unstable_check=unstable_check,
      verbose=false
    )
    if !retcode_success(sol)
      if debug
        println("MS sanity fail: retcode ", sol.retcode, " at range ", i)
      end
      return Inf
    end
    sols[i] = sol
  end

  for i in 1:length(sols)
    if size(Array(sols[i]))[2] != length(ranges[i])
      if debug
        println("MS sanity fail: size mismatch at range ", i)
      end
      return Inf
    end
  end

  group_predictions = Array.(sols)
  for (i, preds) in enumerate(group_predictions)
    if any(x -> !isfinite(x), preds)
      if debug
        println("MS sanity fail: nonfinite state at range ", i)
      end
      return Inf
    end
  end

  loss_zero = zero(eltype(p_est))
  state_loss = loss_zero
  x3_range_loss = loss_zero
  fts_range_loss = loss_zero
  x2dot_loss = loss_zero
  x2dot_weight_sum = 0.0
  compute_recon = eltype(p_est) == Float64
  x1_err_sum = 0.0
  x1_truth_sum = 0.0
  x1_count = 0
  x3_err_sum = 0.0
  x3_truth_sum = 0.0
  x3_count = 0
  weight_sum = 0.0
  x3_u0_target = x3_u0_target_ref[]
  for (i, rg) in enumerate(ranges)
    weight = range_is_contact[i] ? contact_loss_weight : noncontact_loss_weight
    weight_sum += weight
    u = ode_data[:, rg]
    û = group_predictions[i]
    state_loss += weight * (1 / size(u, 2) * sum(abs2.((u[1:2, :] .- û[1:2, :]) ./ state12_scale)))

    range_pen = abs2.(softplus.(abs.(û[3, :]) .- x3_range_amp, x3_range_eps) ./ x3_scale)
    x3_range_loss += weight * x3_range_weight * (sum(range_pen) / size(û, 2))

    fts_pred = [fts_pred_from_state(view(û, :, j), p_est, p_net) for j in 1:size(û, 2)]
    fts_pen = abs2.(softplus.(abs.(fts_pred) .- fts_range_amp, fts_range_eps) ./ fts_scale)
    fts_range_loss += weight * fts_range_weight * (sum(fts_pen) / size(û, 2))

    if range_is_contact[i]
      x2dot_pred = [x2dot_rhs(û[:, j], p_est, tmp_steps[idx], p_net) for (j, idx) in enumerate(rg)]
      if any(x -> !isfinite(x), x2dot_pred)
        if debug
          println("MS sanity fail: nonfinite x2dot at range ", i)
        end
        return Inf
      end
      x2dot_loss += weight * (1 / length(rg) * sum(abs2.((x2dot_data[rg] .- x2dot_pred) ./ x2dot_scale)))
      x2dot_weight_sum += weight
    end

    if compute_recon
      x1_err_sum += sum(abs2, u[1, :] .- û[1, :])
      x1_truth_sum += sum(abs2, u[1, :])
      x1_count += size(u, 2)
      x3_err_sum += sum(abs2, u[3, :] .- û[3, :])
      x3_truth_sum += sum(abs2, u[3, :])
      x3_count += size(u, 2)
    end
  end
  if weight_sum == 0.0
    if debug
      println("MS sanity fail: weight_sum is zero")
    end
    return Inf
  end
  state_loss /= weight_sum
  x3_range_loss /= weight_sum
  fts_range_loss /= weight_sum
  if x2dot_weight_sum > 0.0
    x2dot_loss /= x2dot_weight_sum
  else
    x2dot_loss = 0.0
  end

  continuity_loss = loss_zero
  if use_continuity_loss
    for (i, rg) in enumerate(ranges)
      if i == 1
        continue
      end
      u0 = group_predictions[i-1][:, end]
      u1 = group_predictions[i][:, 1]
      continuity_loss += continuity_term * sum(abs2, u0 - u1)
    end
  end

  x3_u0_loss = 0.0
  x3_loss = x3_u0_loss

  if eltype(p_est) == Float64
    Zygote.ignore() do
      last_loss_components[] = (state=state_loss, x3=x3_loss, x3_u0=x3_u0_loss, x2dot=x2dot_loss, x3_range=x3_range_loss, fts_range=fts_range_loss, continuity=continuity_loss)
      if x1_count > 0
        x1_rmse = sqrt(x1_err_sum / x1_count)
        x1_denom = sqrt(x1_truth_sum / x1_count) + scale_eps
        x1_recon = 100.0 * x1_rmse / x1_denom
        if x3_count > 0
          x3_rmse = sqrt(x3_err_sum / x3_count)
          x3_denom = sqrt(x3_truth_sum / x3_count) + scale_eps
          x3_recon = 100.0 * x3_rmse / x3_denom
          last_recon_metrics[] = (x1=x1_recon, x3=x3_recon)
        else
          last_recon_metrics[] = (x1=x1_recon, x3=NaN)
        end
      end
    end
  end
  l2_penalty = use_l2_regularization ? (l2_weight * structured_l2_penalty(p_net)) : 0.0
  return state_loss + x3_loss + x2dot_loss + x3_range_loss + fts_range_loss + continuity_loss + l2_penalty
end

function loss_multiple_shooting(θ; debug::Bool=false)
  p_est = build_parameter_vector(θ.p)
  u0_ctrl = bound_u0_x3(θ.u0_x3_ctrl)
  ctrl_times = control_times_ref[]
  if use_interpolation
    if length(u0_ctrl) != length(ctrl_times)
      if debug
        println("MS sanity fail: u0_x3_ctrl length mismatch")
      end
      return Inf
    end
    u0_x3 = interp_u0_x3(segment_times_ref[], ctrl_times, u0_ctrl)
  else
    if length(u0_ctrl) != length(ranges_ref[])
      if debug
        println("MS sanity fail: u0_x3_ctrl length mismatch")
      end
      return Inf
    end
    u0_x3 = u0_ctrl
  end
  return loss_multiple_shooting_with_u0(p_est, u0_x3, θ.p_net; debug=debug)
end

function loss_single_shooting_with_u0(p_est, u0_x3, p_net; debug::Bool=false)
  ks_est = p_est[10]
  cs_est = p_est[11]
  if any(x -> !isfinite(x), p_est) ||
     ks_est < ks_bounds[1] || ks_est > ks_bounds[2] ||
     cs_est < cs_bounds[1] || cs_est > cs_bounds[2]
    if debug
      println("SS sanity fail: param guard. ks=", ks_est, " cs=", cs_est)
    end
    return Inf
  end

  function unstable_check(dt, u, p, t)
    if any(x -> !isfinite(x), u) || any(abs.(u) .> 1e7)
      return true
    end
    return false
  end

  prob = remake(
    prob_pred;
    p=(p_net=p_net, ode_par=p_est),
    tspan=(tmp_steps[1], tmp_steps[end]),
    u0=[ode_data[1, 1], ode_data[2, 1], u0_x3]
  )

  sol = solve(prob, integrator; saveat=tmp_steps, reltol=reltol, abstol=abstol,
    sensealg=sensealg, unstable_check=unstable_check, verbose=false)
  if !retcode_success(sol)
    if debug
      println("SS sanity fail: retcode ", sol.retcode)
    end
    return Inf
  end

  x = Array(sol)
  if size(x)[2] != length(tmp_steps)
    if debug
      println("SS sanity fail: size mismatch")
    end
    return Inf
  end
  if any(x -> !isfinite(x), x)
    if debug
      println("SS sanity fail: nonfinite state")
    end
    return Inf
  end

  contact_val = solution_dataframe.contact .== 1
  weights_val = ifelse.(contact_val, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_val)
  if weights_sum == 0.0
    if debug
      println("SS sanity fail: weight_sum is zero")
    end
    return Inf
  end

  state_err = sum(abs2.((ode_data[1:2, :] .- x[1:2, :]) ./ state12_scale); dims=1)
  state_loss = 1 / weights_sum * sum(weights_val .* vec(state_err))

  x3_u0_loss = 0.0
  x3_loss = x3_u0_loss

  range_pen_per_t = abs2.(softplus.(abs.(x[3, :]) .- x3_range_amp, x3_range_eps) ./ x3_scale)
  x3_range_loss = x3_range_weight * (sum(weights_val .* range_pen_per_t) / weights_sum)
  fts_pred = [fts_pred_from_state(view(x, :, i), p_est, p_net) for i in eachindex(tmp_steps)]
  fts_pen_per_t = abs2.(softplus.(abs.(fts_pred) .- fts_range_amp, fts_range_eps) ./ fts_scale)
  fts_range_loss = fts_range_weight * (sum(weights_val .* fts_pen_per_t) / weights_sum)

  weights_x2dot = ifelse.(contact_val, contact_loss_weight, 0.0)
  weights_x2dot_sum = sum(weights_x2dot)
  if weights_x2dot_sum > 0.0
    x2dot_pred = [x2dot_rhs(x[:, i], p_est, tmp_steps[i], p_net) for i in eachindex(tmp_steps)]
    if any(x -> !isfinite(x), x2dot_pred)
      if debug
        println("SS sanity fail: nonfinite x2dot")
      end
      return Inf
    end
    x2dot_loss = 1 / weights_x2dot_sum * sum(weights_x2dot .* abs2.((x2dot_data .- x2dot_pred) ./ x2dot_scale))
  else
    x2dot_loss = 0.0
  end

  if eltype(p_est) == Float64
    Zygote.ignore() do
      last_loss_components[] = (state=state_loss, x3=x3_loss, x3_u0=x3_u0_loss, x2dot=x2dot_loss, x3_range=x3_range_loss, fts_range=fts_range_loss, continuity=0.0)
      x1_recon = relative_rmse_pct(x[1, :], ode_data[1, :], scale_eps)
      x3_recon = relative_rmse_pct(x[3, :], ode_data[3, :], scale_eps)
      last_recon_metrics[] = (x1=x1_recon, x3=x3_recon)
      last_rollout_states[] = copy(x)
      last_nn_metrics[] = contact_net_monitor_metrics_local(x, tmp_steps, contact_val, p_net)
    end
  end

  l2_penalty = use_l2_regularization ? (l2_weight * structured_l2_penalty(p_net)) : 0.0
  return state_loss + x3_loss + x2dot_loss + x3_range_loss + fts_range_loss + l2_penalty
end

function loss_single_shooting(θ; debug::Bool=false)
  p_est = build_parameter_vector(θ.p)
  if use_u0_x3_param
    u0_x3 = bound_u0_x3(θ.u0_x3_ctrl)[1]
  else
    u0_x3 = ode_data[3, 1]
  end
  return loss_single_shooting_with_u0(p_est, u0_x3, θ.p_net; debug=debug)
end

function loss_function(θ; debug::Bool=false)
  return use_multiple_shooting ? loss_multiple_shooting(θ; debug=debug) : loss_single_shooting(θ; debug=debug)
end

function validation_loss_multiple_shooting(θ, validation_df)
  p_est = build_parameter_vector(θ.p)
  tsteps_val = validation_df.t
  contact_val = validation_df.contact .== 1
  weights_val = ifelse.(contact_val, contact_loss_weight, noncontact_loss_weight)
  ks_est = p_est[10]
  cs_est = p_est[11]
  u0_ctrl = bound_u0_x3(θ.u0_x3_ctrl)
  ctrl_times = control_times_ref[]
  if length(u0_ctrl) != length(ctrl_times)
    return Inf
  end
  if any(x -> !isfinite(x), p_est) ||
     ks_est < ks_bounds[1] || ks_est > ks_bounds[2] ||
     cs_est < cs_bounds[1] || cs_est > cs_bounds[2]
    return Inf
  end

  function unstable_check(dt, u, p, t)
    if any(x -> !isfinite(x), u) || any(abs.(u) .> 1e7)
      return true
    end
    return false
  end

  if isempty(ctrl_times)
    return Inf
  end
  u0_x3_val = interp_value(tsteps_val[1], ctrl_times, u0_ctrl)
  u0_val = [validation_df.x1[1], validation_df.x2[1], u0_x3_val]
  prob = remake(
    prob_pred;
    p=(p_net=θ.p_net, ode_par=p_est),
    tspan=(tsteps_val[1], tsteps_val[end]),
    u0=u0_val
  )

  local solutions
  try
  solutions = solve(prob, integrator; saveat=tsteps_val, reltol=reltol, abstol=abstol,
    sensealg=sensealg, unstable_check=unstable_check, verbose=false)
  catch
    return Inf
  end
  if !retcode_success(solutions)
    return Inf
  end
  x = Array(solutions)

  if size(x)[2] != size(validation_df, 1)
    return Inf
  end

  x2dot_pred = [x2dot_rhs(x[:, i], p_est, tsteps_val[i], θ.p_net) for i in eachindex(tsteps_val)]

  state_err = sum(abs2.((Array(validation_df[:, [:x1, :x2]])' .- x[1:2, :]) ./ state12_scale); dims=1)
  weights_sum = sum(weights_val)
  if weights_sum == 0.0
    return Inf
  end
  loss = 1 / weights_sum * sum(weights_val .* vec(state_err))

  x3_u0_loss = 0.0
  range_pen_per_t = abs2.(softplus.(abs.(x[3, :]) .- x3_range_amp, x3_range_eps) ./ x3_scale)
  x3_range_loss = x3_range_weight * (sum(weights_val .* range_pen_per_t) / sum(weights_val))
  fts_pred = [fts_pred_from_state(view(x, :, i), p_est, θ.p_net) for i in eachindex(tsteps_val)]
  fts_pen_per_t = abs2.(softplus.(abs.(fts_pred) .- fts_range_amp, fts_range_eps) ./ fts_scale)
  fts_range_loss = fts_range_weight * (sum(weights_val .* fts_pen_per_t) / sum(weights_val))
  loss += x3_u0_loss + x3_range_loss + fts_range_loss

  weights_x2dot = ifelse.(contact_val, contact_loss_weight, 0.0)
  weights_x2dot_sum = sum(weights_x2dot)
  if weights_x2dot_sum > 0.0
    loss += 1 / weights_x2dot_sum * sum(weights_x2dot .* abs2.((validation_df.x2dot .- x2dot_pred) ./ x2dot_scale))
  end

  return loss
end

function validation_loss_single_shooting(θ, validation_df)
  p_est = build_parameter_vector(θ.p)
  tsteps_val = validation_df.t
  contact_val = validation_df.contact .== 1
  weights_val = ifelse.(contact_val, contact_loss_weight, noncontact_loss_weight)
  ks_est = p_est[10]
  cs_est = p_est[11]
  if any(x -> !isfinite(x), p_est) ||
     ks_est < ks_bounds[1] || ks_est > ks_bounds[2] ||
     cs_est < cs_bounds[1] || cs_est > cs_bounds[2]
    return Inf
  end

  function unstable_check(dt, u, p, t)
    if any(x -> !isfinite(x), u) || any(abs.(u) .> 1e7)
      return true
    end
    return false
  end

  u0_x3_val = use_u0_x3_param ? bound_u0_x3(θ.u0_x3_ctrl)[1] : ode_data[3, 1]
  u0_val = [validation_df.x1[1], validation_df.x2[1], u0_x3_val]
  prob = remake(
    prob_pred;
    p=(p_net=θ.p_net, ode_par=p_est),
    tspan=(tsteps_val[1], tsteps_val[end]),
    u0=u0_val
  )

  local solutions
  try
    solutions = solve(prob, integrator; saveat=tsteps_val, reltol=reltol, abstol=abstol,
      sensealg=sensealg, unstable_check=unstable_check, verbose=false)
  catch
    return Inf
  end
  if !retcode_success(solutions)
    return Inf
  end
  x = Array(solutions)
  if size(x)[2] != size(validation_df, 1)
    return Inf
  end

  x2dot_pred = similar(validation_df.x2dot, eltype(p_est))
  for i in eachindex(tsteps_val)
    x2dot_pred[i] = x2dot_rhs(x[:, i], p_est, tsteps_val[i], θ.p_net)
  end

  state_err = sum(abs2.((Array(validation_df[:, [:x1, :x2]])' .- x[1:2, :]) ./ state12_scale); dims=1)
  weights_sum = sum(weights_val)
  if weights_sum == 0.0
    return Inf
  end
  loss = 1 / weights_sum * sum(weights_val .* vec(state_err))

  x3_u0_loss = 0.0
  range_pen_per_t = abs2.(softplus.(abs.(x[3, :]) .- x3_range_amp, x3_range_eps) ./ x3_scale)
  x3_range_loss = x3_range_weight * (sum(weights_val .* range_pen_per_t) / sum(weights_val))
  fts_pred = [fts_pred_from_state(view(x, :, i), p_est, θ.p_net) for i in eachindex(tsteps_val)]
  fts_pen_per_t = abs2.(softplus.(abs.(fts_pred) .- fts_range_amp, fts_range_eps) ./ fts_scale)
  fts_range_loss = fts_range_weight * (sum(weights_val .* fts_pen_per_t) / sum(weights_val))
  loss += x3_u0_loss + x3_range_loss + fts_range_loss

  weights_x2dot = ifelse.(contact_val, contact_loss_weight, 0.0)
  weights_x2dot_sum = sum(weights_x2dot)
  if weights_x2dot_sum > 0.0
    loss += 1 / weights_x2dot_sum * sum(weights_x2dot .* abs2.((validation_df.x2dot .- x2dot_pred) ./ x2dot_scale))
  end

  return loss
end

function validation_loss_function(θ, validation_df)
  return use_multiple_shooting ? validation_loss_multiple_shooting(θ, validation_df) : validation_loss_single_shooting(θ, validation_df)
end

function run_step2b_sanity(; strict::Bool=false)
  tprintln("=== Step2b SANITY (oracle F_contact + true mech) ===")

  contact_mask = solution_dataframe.contact .== 1
  weights_val = ifelse.(contact_mask, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_val)
  if !isfinite(weights_sum) || weights_sum <= 0.0
    msg = "Step2b sanity check failed: non-finite or non-positive weights"
    strict ? error(msg) : tprintln("  WARNING: ", msg)
    return Inf, nothing
  end

  u0 = [ode_data[1, 1], ode_data[2, 1], ode_data[3, 1]]
  oracle_uode = make_uode_derivative_function_contact_net_nn_local(nothing, nothing)
  oracle_prob = ODEProblem{false}(oracle_uode, u0, (tmp_steps[1], tmp_steps[end]), (p_net=nothing, ode_par=original_parameters))
  sol = solve(oracle_prob, integrator; saveat=tmp_steps, abstol=abstol, reltol=reltol, sensealg=sensealg, verbose=false)

  if !retcode_success(sol) || size(sol, 2) != length(tmp_steps)
    msg = "Step2b sanity check failed: solver retcode=" * string(sol.retcode)
    strict ? error(msg) : tprintln("  WARNING: ", msg)
    return Inf, nothing
  end

  x = Array(sol)
  x1_scale, x2_scale = state12_scale
  x1_state_err = vec(abs2.((ode_data[1, :] .- x[1, :]) ./ x1_scale))
  x2_state_err = vec(abs2.((ode_data[2, :] .- x[2, :]) ./ x2_scale))
  x1_state_loss = sum(weights_val .* x1_state_err) / weights_sum
  x2_state_loss = sum(weights_val .* x2_state_err) / weights_sum
  state_loss = x1_state_loss + x2_state_loss

  weights_x2dot = ifelse.(contact_mask, contact_loss_weight, 0.0)
  weights_x2dot_sum = sum(weights_x2dot)
  if weights_x2dot_sum > 0.0
    x2dot_pred = [x2dot_rhs(x[:, i], original_parameters, tmp_steps[i], nothing) for i in eachindex(tmp_steps)]
    x2dot_loss = sum(weights_x2dot .* abs2.((x2dot_data .- x2dot_pred) ./ x2dot_scale)) / weights_x2dot_sum
  else
    x2dot_loss = 0.0
  end

  range_pen = abs2.(softplus.(abs.(x[3, :]) .- x3_range_amp, x3_range_eps) ./ x3_scale)
  x3_range_loss = x3_range_weight * (sum(weights_val .* range_pen) / weights_sum)
  fts_pred = [fts_pred_from_state(view(x, :, i), original_parameters, nothing) for i in eachindex(tmp_steps)]
  fts_pen = abs2.(softplus.(abs.(fts_pred) .- fts_range_amp, fts_range_eps) ./ fts_scale)
  fts_range_loss = fts_range_weight * (sum(weights_val .* fts_pen) / weights_sum)

  x1_rec = relative_rmse_pct(x[1, :], ode_data[1, :], scale_eps)
  x3_rec = relative_rmse_pct(x[3, :], ode_data[3, :], scale_eps)
  total = state_loss + x2dot_loss + x3_range_loss + fts_range_loss
  diag = (
    state=state_loss,
    x1_state=x1_state_loss,
    x2_state=x2_state_loss,
    x2dot=x2dot_loss,
    x3_range=x3_range_loss,
    fts_range=fts_range_loss,
    cont=0.0,
    l2=0.0,
    x1_rec=x1_rec,
    x3_rec=x3_rec,
    fcontact_err=0.0,
    raw_min=NaN,
    raw_max=NaN,
    raw_span=NaN,
    raw_mean=NaN,
    raw_neg_frac=NaN,
  )

  tprintln("  sanity loss=", @sprintf("%.4e", total))
  tprintln("  sanity parts: state=", fmt_num(diag.state),
    " x1_state=", fmt_num(diag.x1_state),
    " x2_state=", fmt_num(diag.x2_state),
    " x2dot=", fmt_num(diag.x2dot),
    " x3r=", fmt_num(diag.x3_range),
    " ftsr=", fmt_num(diag.fts_range),
    " cont=", fmt_num(diag.cont),
    " l2=", fmt_num(diag.l2))
  tprintln("  sanity rec: x1=", fmt_pct(diag.x1_rec), "% x3=", fmt_pct(diag.x3_rec), "%")
  tprintln("  sanity nn: F_contact err=", fmt_pct(diag.fcontact_err), "%")
  tprintln("  sanity nn raw: min=", fmt_num(diag.raw_min),
    " max=", fmt_num(diag.raw_max),
    " span=", fmt_num(diag.raw_span),
    " mean=", fmt_num(diag.raw_mean),
    " neg=", fmt_pct(100 * diag.raw_neg_frac), "%")

  if !isfinite(total)
    msg = "Step2b sanity check failed: non-finite loss"
    strict ? error(msg) : tprintln("  WARNING: ", msg)
  else
    tprintln("=== Step2b SANITY OK ===")
  end

  return total, diag
end

update_ranges!(solution_dataframe.contact .== 1)
update_x3_targets!()
update_control_points!()

sanity_loss = NaN
sanity_diag = nothing
if step2b_sanity
  sanity_loss, sanity_diag = run_step2b_sanity(; strict=false)
else
  tprintln("Step2b sanity check skipped")
end
sanity_payload = (
  oracle=(loss=sanity_loss, comps=sanity_diag),
)

# Training/validation split (fixed stride, 25% validation)
# Keep the very first post-contact point in training so the initial condition remains valid.
n_total = size(solution_dataframe, 1)
val_stride = 4
val_offset = val_stride
validation_mask = collect(val_offset:val_stride:n_total)
training_mask = setdiff(1:n_total, validation_mask)

training_mask = sort(training_mask)
validation_mask = sort(validation_mask)

original_ode_data = deepcopy(ode_data)
original_solution_dataframe = deepcopy(solution_dataframe)

ode_data = original_ode_data[:, training_mask]
solution_dataframe = original_solution_dataframe[training_mask, :]
x2dot_data = solution_dataframe.x2dot

validation_solution_dataframe = original_solution_dataframe[validation_mask, :]

rng = Random.default_rng()

val_contact = validation_solution_dataframe.contact .== 1
val_ranges, val_labels = build_contact_ranges(val_contact, small_group_size, large_group_size, boundary_window, min_group_size)
tmp_steps = solution_dataframe.t
datasize = size(ode_data, 2)
tspan = (tmp_steps[1], tmp_steps[end])
update_ranges!(solution_dataframe.contact .== 1)
update_x3_targets!()
update_control_points!()

learning_rate_adam = 1e-3
max_adam_iters = parse(Int, get(ENV, "HNODECB_STEP2B_MAX_ADAM_ITERS", "1000"))
max_lbfgs_iters = parse(Int, get(ENV, "HNODECB_STEP2B_MAX_LBFGS_ITERS", "1000"))
print_interval_adam = 1
print_interval_lbfgs = 1
val_log_every = 1
use_adam_stage = false
plateau_early_stop = get(ENV, "HNODECB_STAGE2_PLATEAU_EARLY_STOP", "0") == "1"
plateau_window = 5
plateau_rel_tol = 0.01
good_enough_loss = try
  parse(Float64, get(ENV, "HNODECB_STEP2B_GOOD_ENOUGH_LOSS", "1e-10"))
catch
  1e-10
end
step_max_loss_increase_frac = try
  parse(Float64, get(ENV, "HNODECB_STAGE2_STEP_MAX_LOSS_INCREASE_FRAC", "0.05"))
catch
  0.05
end
if !isfinite(step_max_loss_increase_frac) || step_max_loss_increase_frac < 0.0
  step_max_loss_increase_frac = 0.05
end


num_random_initial_guesses = 1
use_stage2_init = true
# Disable multi-start for now:
# - when stage2 windowed results exist, inherit only the single best stage2 candidate for that window
# - otherwise, fall back to a single random initial guess
stage2_mode = :best # :best or :topk
stage2_topk = 1
append_random_after_stage2 = false

function step2b_result_signature()
  return (
    result_name=result_name_string,
    stage2_input_basename=stage2_input_basename,
    window_start=window_metadata.start_idx,
    window_stop=window_metadata.stop_idx,
    window_len=window_metadata.len,
    window_label=window_metadata.label,
    window_role=window_metadata.role,
    use_multiple_shooting=use_multiple_shooting,
    use_u0_x3_param=use_u0_x3_param,
    use_adam_stage=use_adam_stage,
    max_adam_iters=max_adam_iters,
    max_lbfgs_iters=max_lbfgs_iters,
    dynamic_gnn=step2b_dynamic_gnn,
    fixed_gnn=step2b_fixed_gnn,
    gnn_q=step2b_gnn_q
  )
end

function step2b_load_resume_state(path::AbstractString; signature)
  if !isfile(path)
    return Any[], Any[], nothing, "missing"
  end
  data = try
    deserialize(path)
  catch ex
    return Any[], Any[], nothing, "deserialize_error:" * string(typeof(ex))
  end
  saved_signature = field_or(data, :step2b_resume_signature, nothing)
  if saved_signature != signature
    return Any[], Any[], nothing, "signature_mismatch"
  end
  results_loaded = copy(field_or(data, :results, Any[]))
  run_summaries_loaded = copy(field_or(data, :run_summaries, Any[]))
  active_checkpoint = field_or(data, :active_checkpoint, nothing)
  return results_loaded, run_summaries_loaded, active_checkpoint, ""
end

results = Any[]

function stage2_p_net_from_vec(p_net_vec, num_hidden_layers::Int, num_hidden_nodes::Int)
  _, _, template = setup_nn_bundle(num_hidden_layers, num_hidden_nodes)
  if length(p_net_vec) != length(template)
    tprintln("Stage2 init: p_net length mismatch (got ", length(p_net_vec),
      ", expected ", length(template), "); fallback to template init.")
    return deepcopy(template)
  end
  p_net = deepcopy(template)
  p_net .= p_net_vec
  return p_net
end

function make_stage2_guess(ks0, cs0, source;
  p_net=deepcopy(p_net_init),
  num_hidden_layers=primary_stage2_nn.num_hidden_layers,
  num_hidden_nodes=primary_stage2_nn.num_hidden_nodes,
  nn_gain=primary_stage2_nn.nn_gain)
  return (
    ks=Float64(ks0),
    cs=Float64(cs0),
    p_net=deepcopy(p_net),
    num_hidden_layers=Int(num_hidden_layers),
    num_hidden_nodes=Int(num_hidden_nodes),
    nn_gain=Float64(nn_gain),
    source=source
  )
end

function load_stage2_initial_guesses()
  if !use_stage2_init
    return NamedTuple[]
  end
  tprintln("Stage2 init: looking for file at ", stage2_path)
  if !isfile(stage2_path)
    tprintln("Stage2 init enabled but file not found: ", stage2_path)
    return NamedTuple[]
  end
  stage2 = deserialize(stage2_path)
  if haskey(stage2, :use_multiple_shooting)
    if stage2.use_multiple_shooting != use_multiple_shooting
      tprintln("目标函数不一致: stage2 use_multiple_shooting=",
        stage2.use_multiple_shooting, " vs step2b use_multiple_shooting=", use_multiple_shooting)
      exit(1)
    end
  else
    tprintln("目标函数不一致: stage2 file missing use_multiple_shooting metadata. Please rerun stage2.")
    exit(1)
  end
  if haskey(stage2, :window_results)
    window_candidates = NamedTuple[]
    for (i, wr) in enumerate(stage2.window_results)
      if !isempty(stage2light_filter_label)
        if !(haskey(wr, :window) && haskey(wr.window, :label) && string(wr.window.label) == stage2light_filter_label)
          continue
        end
      elseif !isempty(stage2light_filter_role)
        if !(haskey(wr, :window) && haskey(wr.window, :role) && string(wr.window.role) == stage2light_filter_role)
          continue
        end
      end
      if !haskey(wr, :best)
        tprintln("Stage2 init: window result ", i, " missing :best; skipping.")
        continue
      end
      b = wr.best
      if !(haskey(b, :ks_hat) && haskey(b, :cs_hat) && haskey(b, :loss) && haskey(b, :p_net_vec))
        tprintln("Stage2 init: window result ", i, " missing ks_hat/cs_hat/loss/p_net_vec; skipping.")
        continue
      end
      params = haskey(b, :params) ? b.params : Dict{Any, Any}()
      num_hidden_layers = params_get_int(params, "num_hidden_layers", primary_stage2_nn.num_hidden_layers)
      num_hidden_nodes = params_get_int(params, "num_hidden_nodes", primary_stage2_nn.num_hidden_nodes)
      win_label = (haskey(wr, :window) && haskey(wr.window, :label)) ? string(wr.window.label) : "w$(i)"
      push!(window_candidates, (
        loss=Float64(b.loss),
        ks=Float64(b.ks_hat),
        cs=Float64(b.cs_hat),
        p_net=stage2_p_net_from_vec(b.p_net_vec, num_hidden_layers, num_hidden_nodes),
        num_hidden_layers=num_hidden_layers,
        num_hidden_nodes=num_hidden_nodes,
        nn_gain=result_get_float(b, :nn_gain, params_get_float(params, "nn_gain", primary_stage2_nn.nn_gain)),
        source="stage2light_$(win_label)"
      ))
    end
    tprintln("Stage2 init: found ", length(window_candidates), " windowed candidates in file.")
    if isempty(window_candidates)
      return NamedTuple[]
    end
    res_sorted = sort(window_candidates, by = r -> r.loss)
    if stage2_mode == :best
      b = res_sorted[1]
      return [make_stage2_guess(b.ks, b.cs, b.source;
        p_net=b.p_net,
        num_hidden_layers=b.num_hidden_layers,
        num_hidden_nodes=b.num_hidden_nodes,
        nn_gain=b.nn_gain)]
    elseif stage2_mode == :topk
      k = min(stage2_topk, length(res_sorted))
      return [make_stage2_guess(res_sorted[i].ks, res_sorted[i].cs,
        "$(res_sorted[i].source)_rank$(i)";
        p_net=res_sorted[i].p_net,
        num_hidden_layers=res_sorted[i].num_hidden_layers,
        num_hidden_nodes=res_sorted[i].num_hidden_nodes,
        nn_gain=res_sorted[i].nn_gain) for i in 1:k]
    else
      tprintln("Unknown stage2_mode=", stage2_mode, " (use :best or :topk).")
      return NamedTuple[]
    end
  elseif haskey(stage2, :results)
    tprintln("Stage2 init: found ", length(stage2.results), " results in file.")
    if stage2_mode == :best
      b = stage2.best
      return [make_stage2_guess(b.ks, b.cs, "stage2_best"; nn_gain=result_get_float(b, :nn_gain, primary_stage2_nn.nn_gain))]
    elseif stage2_mode == :topk
      res_sorted = sort(stage2.results, by = r -> r.loss)
      k = min(stage2_topk, length(res_sorted))
      if k == 0
        tprintln("Stage2 init: results list is empty.")
        return NamedTuple[]
      end
      return [make_stage2_guess(res_sorted[i].ks, res_sorted[i].cs,
        "stage2_top$(k)_$(i)"; nn_gain=result_get_float(res_sorted[i], :nn_gain, primary_stage2_nn.nn_gain)) for i in 1:k]
    else
      tprintln("Unknown stage2_mode=", stage2_mode, " (use :best or :topk).")
      return NamedTuple[]
    end
  else
    tprintln("Stage2 init enabled but file does not contain :results or :window_results. Keys=", collect(keys(stage2)))
    return NamedTuple[]
  end
end
range_starts = [first(rg) for rg in ranges_ref[]]
if use_u0_x3_param
  if use_interpolation
    ctrl_idx = control_segments_ref[]
    initial_u0_x3_ctrl = isempty(ctrl_idx) ? Float64[] : unbound_u0_x3(fill(ode_data[3, 1], length(ctrl_idx)))
  else
    initial_u0_x3_ctrl = isempty(range_starts) ? Float64[] : unbound_u0_x3(fill(ode_data[3, 1], length(range_starts)))
  end
else
  initial_u0_x3_ctrl = Float64[]
end
run_summaries = Any[]

stage_ref = Ref(use_adam_stage ? "adam" : "lbfgs")

stage2_inits = load_stage2_initial_guesses()
initial_guesses = NamedTuple[]
if isempty(stage2_inits)
  tprintln("无法找到可继承结果，将使用随机起点！")
  for run_id in 1:num_random_initial_guesses
    band = band_for_run(run_id)
    ks0 = sample_log_band(rng, ks_bounds[1], ks_bounds[2], band)
    cs0 = sample_log_band(rng, cs_bounds[1], cs_bounds[2], band)
    push!(initial_guesses, (
      ks=ks0, cs=cs0,
      p_net=deepcopy(p_net_init),
      num_hidden_layers=primary_stage2_nn.num_hidden_layers,
      num_hidden_nodes=primary_stage2_nn.num_hidden_nodes,
      nn_gain=1.0,
      source=string(band)
    ))
  end
else
  append!(initial_guesses, stage2_inits)
  if append_random_after_stage2
    for run_id in 1:num_random_initial_guesses
      band = band_for_run(run_id)
      ks0 = sample_log_band(rng, ks_bounds[1], ks_bounds[2], band)
      cs0 = sample_log_band(rng, cs_bounds[1], cs_bounds[2], band)
      push!(initial_guesses, (
        ks=ks0, cs=cs0,
        p_net=deepcopy(p_net_init),
        num_hidden_layers=primary_stage2_nn.num_hidden_layers,
        num_hidden_nodes=primary_stage2_nn.num_hidden_nodes,
        nn_gain=1.0,
        source=string(band)
      ))
    end
  end
end

resume_signature = step2b_result_signature()
active_checkpoint_ref = Ref{Any}(nothing)
resume_reason = ""
if step2b_resume_enabled
  results_loaded, summaries_loaded, active_checkpoint_loaded, resume_reason = step2b_load_resume_state(
    step2b_result_path; signature=resume_signature)
  append!(results, results_loaded)
  append!(run_summaries, summaries_loaded)
  active_checkpoint_ref[] = active_checkpoint_loaded
  if !isempty(results) || !isempty(run_summaries)
    tprintln("Step2b resume: loaded ", length(results), " result(s), ",
      length(run_summaries), " run summary record(s) from checkpoint.")
  elseif resume_reason != "" && resume_reason != "missing"
    tprintln("Step2b resume: ignored existing checkpoint | reason=", resume_reason)
  end
  if active_checkpoint_ref[] !== nothing
    tprintln("Step2b resume: active checkpoint found | run=",
      field_or(active_checkpoint_ref[], :run_id, -1),
      " | stage=", string(field_or(active_checkpoint_ref[], :stage, "unknown")),
      " | epoch=", field_or(field_or(active_checkpoint_ref[], :run_state, nothing), :epoch, 0))
  end
end

function save_step2b_partial(results_local, run_summaries_local, active_checkpoint_local)
  serialize(step2b_result_path, (
    results=results_local,
    run_summaries=run_summaries_local,
    sanity=sanity_payload,
    window=window_metadata,
    step2b_resume_signature=resume_signature,
    active_checkpoint=active_checkpoint_local
  ))
end

completed_run_ids = Set{Int}()
for s in run_summaries
  run_id_val = field_or(s, :run_id, nothing)
  run_id_val isa Integer || continue
  push!(completed_run_ids, Int(run_id_val))
end

for run_id in 1:length(initial_guesses)
  if run_id in completed_run_ids
    tprintln("Run ", run_id, " skip -- already completed from checkpoint.")
    continue
  end
  guess = initial_guesses[run_id]
  guess_num_hidden_layers = Int(guess.num_hidden_layers)
  guess_num_hidden_nodes = Int(guess.num_hidden_nodes)
  global approximating_neural_network, st, p_net_init, uode_derivative_function, prob_pred
  approximating_neural_network, st, p_net_init = setup_nn_bundle(
    guess_num_hidden_layers,
    guess_num_hidden_nodes,
  )
  uode_derivative_function = make_uode_derivative_function_contact_net_nn_local(
    approximating_neural_network,
    st,
  )
  prob_pred = ODEProblem{false}(
    uode_derivative_function,
    original_u0,
    tspan,
    (p_net=p_net_init, ode_par=original_parameters)
  )
  ks0 = guess.ks
  cs0 = guess.cs
  p_net0 = guess.p_net
  inherited_nn_gain = (haskey(guess, :nn_gain) && isfinite(guess.nn_gain) && guess.nn_gain > 0.0) ? Float64(guess.nn_gain) : 1.0
  if length(p_net0) != length(p_net_init)
    tprintln("Run ", run_id, " inherited p_net length mismatch for architecture ",
      "(layers=", guess_num_hidden_layers, ", nodes=", guess_num_hidden_nodes, "); fallback to template init.")
    p_net0 = deepcopy(p_net_init)
  end
  theta0 = initial_theta_from_guess(ks0, cs0)
  nn_gain_init_info = (
    gain=inherited_nn_gain,
    candidate_gain=inherited_nn_gain,
    true_amp=NaN,
    pred_amp=NaN,
    true_decade=nothing,
    pred_decade=nothing,
    decade_delta=nothing,
    log10_diff=NaN,
    updated=false,
    reason="inherited_stage2"
  )
  current_nn_gain_ref[] = inherited_nn_gain
  if step2b_fixed_gnn || step2b_dynamic_gnn
    if !(startswith(String(guess.source), "stage2light_") && isfinite(inherited_nn_gain) && inherited_nn_gain > 0.0)
      train_contact_true_init = contact_net_true_from_states_local(ode_data)
      train_eff_pos_init = effective_contact_positions_from_truth_local(train_contact_true_init)
      nn_gain_init_info = decade_nn_gain_from_states_local(
        ode_data, tmp_steps, p_net0;
        contact_true_ref=train_contact_true_init, eff_pos_ref=train_eff_pos_init,
        q=step2b_gnn_q, gain_min=step2b_gnn_min, gain_max=step2b_gnn_max,
        current_gain=nothing, log10_update_min=step2b_gnn_log10_update_min
      )
      current_nn_gain_ref[] = nn_gain_init_info.gain
    end
  else
    current_nn_gain_ref[] = 1.0
    nn_gain_init_info = (
      gain=1.0,
      candidate_gain=1.0,
      true_amp=NaN,
      pred_amp=NaN,
      true_decade=nothing,
      pred_decade=nothing,
      decade_delta=nothing,
      log10_diff=NaN,
      updated=false,
      reason="disabled"
    )
  end

  ks0_err = percent_error_pct(ks0, ks)
  cs0_err = percent_error_pct(cs0, cs)
  tprintln("Run ", run_id, " initial guess (", guess.source, "): ks0=", @sprintf("%.3e", ks0),
    " (true=", @sprintf("%.3e", ks), ", ", @sprintf("%.2f", ks0_err), "%)",
    " cs0=", @sprintf("%.3e", cs0),
    " (true=", @sprintf("%.3e", cs), ", ", @sprintf("%.2f", cs0_err), "%)",
    " | nn(layers=", guess_num_hidden_layers, ", nodes=", guess_num_hidden_nodes, ")")
  tprintln("  g_nn init=", fmt_num(current_nn_gain_ref[], sigdigits=4),
    " | mode=", step2b_dynamic_gnn ? "DYNAMIC" : (step2b_fixed_gnn ? "FIXED" : "OFF"),
    " | A_true=", fmt_num(nn_gain_init_info.true_amp, sigdigits=4),
    " A_pred=", fmt_num(nn_gain_init_info.pred_amp, sigdigits=4),
    " | dec_true=", (nn_gain_init_info.true_decade === nothing ? "NA" : string(nn_gain_init_info.true_decade)),
    " dec_pred=", (nn_gain_init_info.pred_decade === nothing ? "NA" : string(nn_gain_init_info.pred_decade)),
    " delta_dec=", (nn_gain_init_info.decade_delta === nothing ? "NA" : string(nn_gain_init_info.decade_delta)),
    " | candidate=", fmt_num(nn_gain_init_info.candidate_gain, sigdigits=4),
    " log10diff=", fmt_pct(nn_gain_init_info.log10_diff, digits=3),
    " | reason=", nn_gain_init_info.reason)

  p0 = ComponentArray(theta0)
  if use_u0_x3_param
    starting_point_in = ComponentVector(p=p0, u0_x3_ctrl=initial_u0_x3_ctrl, p_net=p_net0)
  else
    starting_point_in = ComponentVector(p=p0, p_net=p_net0)
  end
  resume_checkpoint = nothing
  if active_checkpoint_ref[] !== nothing && field_or(active_checkpoint_ref[], :run_id, -1) == run_id
    resume_checkpoint = active_checkpoint_ref[]
    tprintln("Step2b resume: continuing run ", run_id,
      " | stage=", string(field_or(resume_checkpoint, :stage, "lbfgs")),
      " | resume_epoch=", field_or(field_or(resume_checkpoint, :run_state, nothing), :epoch, 0))
  end

  if resume_checkpoint === nothing
    init_loss = try
      loss_function(starting_point_in; debug=true)
    catch ex
      tprintln("Run ", run_id, " init diagnostics exception: ", typeof(ex), " - ", ex)
      Inf
    end
    init_comps = last_loss_components[]
    init_recon = last_recon_metrics[]
    if !isfinite(init_loss)
      init_comps = (state=NaN, x3=NaN, x3_u0=NaN, x2dot=NaN, x3_range=NaN, fts_range=NaN, continuity=NaN)
      init_recon = (x1=NaN, x3=NaN)
      tprintln("Run ", run_id, " init diagnostics: no guard fired (loss=Inf)")
    end
    tprintln("Run ", run_id, " init diagnostics (", guess.source, "): loss=", @sprintf("%.4e", init_loss),
      " | x1_rec=", @sprintf("%.2f", init_recon.x1), "% x3_rec=", @sprintf("%.2f", init_recon.x3), "%",
      " | state=", @sprintf("%.4e", init_comps.state),
      " x3=", @sprintf("%.4e", init_comps.x3), " (u0=None)",
      " x2dot=", @sprintf("%.4e", init_comps.x2dot),
      " x3_range=", @sprintf("%.4e", init_comps.x3_range),
      " fts_range=", @sprintf("%.4e", init_comps.fts_range),
      " cont=", @sprintf("%.4e", init_comps.continuity))
  end

  best_training_parameters = [starting_point_in]
  best_training_gnn = Ref(current_nn_gain_ref[])
  training_epochs = zeros(Int, max_adam_iters)
  training_costs = fill(Inf, max_adam_iters)
  adam_epoch = Ref(0)
  lbfgs_epoch = Ref(0)
  best_loss_ref = Ref(Inf)
  lbfgs_recent = Float64[]
  adam_recent = Float64[]
  prev_theta_vec_ref = Ref(Vector{Float64}())
  prev_grad_vec_ref = Ref(Vector{Float64}())
  prev_loss_ref = Ref(NaN)
  prev_epoch_loss_ref = Ref(NaN)
  last_epoch_wall_ref = Ref(time())
  if resume_checkpoint !== nothing
    best_theta_resume = field_or(resume_checkpoint, :best_theta, nothing)
    if best_theta_resume !== nothing && length(Vector(best_theta_resume)) == length(Vector(starting_point_in))
      best_training_parameters[1] = deepcopy(best_theta_resume)
    end
    best_gnn_resume = field_or(resume_checkpoint, :best_gnn, current_nn_gain_ref[])
    if isfinite(best_gnn_resume) && best_gnn_resume > 0.0
      best_training_gnn[] = Float64(best_gnn_resume)
    end
    best_loss_resume = field_or(resume_checkpoint, :best_loss, Inf)
    best_loss_ref[] = (isfinite(best_loss_resume) ? Float64(best_loss_resume) : Inf)
    current_gnn_resume = field_or(resume_checkpoint, :current_nn_gain, current_nn_gain_ref[])
    if isfinite(current_gnn_resume) && current_gnn_resume > 0.0
      current_nn_gain_ref[] = Float64(current_gnn_resume)
    end
    lbfgs_recent = copy(field_or(resume_checkpoint, :lbfgs_recent, Float64[]))
    prev_loss_ref[] = field_or(resume_checkpoint, :prev_loss, NaN)
    prev_epoch_loss_ref[] = field_or(resume_checkpoint, :prev_epoch_loss, NaN)
    if string(field_or(resume_checkpoint, :stage, "lbfgs")) == "lbfgs"
      lbfgs_epoch[] = field_or(field_or(resume_checkpoint, :run_state, nothing), :epoch, 0)
    end
  end
  train_contact_true = contact_net_true_from_states_local(ode_data)
  train_eff_pos = effective_contact_positions_from_truth_local(train_contact_true)
  function maybe_update_best!(θ, l)
    if l < best_loss_ref[]
      best_loss_ref[] = l
      best_training_parameters[1] = deepcopy(θ)
      best_training_gnn[] = current_nn_gain_ref[]
    end
  end
  function gnn_update_fn(theta, current_gain)
    return decade_nn_gain_from_states_local(
      ode_data, tmp_steps, theta.p_net;
      contact_true_ref=train_contact_true, eff_pos_ref=train_eff_pos,
      q=step2b_gnn_q, gain_min=step2b_gnn_min, gain_max=step2b_gnn_max,
      current_gain=current_gain, log10_update_min=step2b_gnn_log10_update_min
    )
  end
  function callback(θ, l, stats=nothing)
    epoch_wall = time() - last_epoch_wall_ref[]
    last_epoch_wall_ref[] = time()
    if stage_ref[] == "adam"
      adam_epoch[] += 1
      epoch = adam_epoch[]
      if epoch <= length(training_epochs)
        training_epochs[epoch] = epoch
        training_costs[epoch] = l
      end
    else
      if stats !== nothing && hasproperty(stats, :iter)
        lbfgs_epoch[] = Int(stats.iter)
      else
        lbfgs_epoch[] += 1
      end
      epoch = lbfgs_epoch[]
    end
    maybe_update_best!(θ, l)
    if (stage_ref[] == "adam" && epoch % print_interval_adam == 0) ||
       (stage_ref[] == "lbfgs" && epoch % print_interval_lbfgs == 0)
      comps = last_loss_components[]
      recon = last_recon_metrics[]
      nn_diag = last_nn_metrics[]
      p_est = build_parameter_vector(θ.p)
      ks_est = p_est[10]
      cs_est = p_est[11]
      ks_err = percent_error_pct(ks_est, ks)
      cs_err = percent_error_pct(cs_est, cs)
      grad_vec = stats === nothing ? Vector{Float64}() : copy(stats.grad)
      grad_norm = isempty(grad_vec) ? NaN : norm(grad_vec)
      step_vec = stats === nothing ? Vector{Float64}() : copy(stats.step)
      step_norm = stats === nothing ? NaN : stats.step_norm
      loss_delta = stats === nothing ? (isnan(prev_loss_ref[]) ? NaN : (l - prev_loss_ref[])) : stats.dloss
      secant_sy = stats === nothing ? NaN : stats.sTy
      secant_curv = stats === nothing ? NaN : stats.curv
      grad_dot_step = stats === nothing ? NaN : stats.g_dot_step
      val_loss_epoch = if epoch % val_log_every == 0
        try
          validation_loss_function(θ, validation_solution_dataframe)
        catch
          NaN
        end
      else
        NaN
      end
      loss_rebound = isfinite(prev_epoch_loss_ref[]) && l > prev_epoch_loss_ref[]
      loss_jump = isfinite(prev_epoch_loss_ref[]) && isfinite(step_max_loss_increase_frac) &&
        l > prev_epoch_loss_ref[] * (1.0 + step_max_loss_increase_frac)

      logmsg("Step2b " * stage_ref[] * " epoch " * string(epoch) * " train=" * @sprintf("%.4e", l))
      logmsg("  grad_norm=" * fmt_num(grad_norm) * " val=" * fmt_num(val_loss_epoch) *
        " | iter_s=" * @sprintf("%.1f", epoch_wall))
      logmsg("  parts: state=" * fmt_num(comps.state) *
        " x3=" * fmt_num(comps.x3) * " (u0=None) x2dot=" * fmt_num(comps.x2dot) *
        " x3_range=" * fmt_num(comps.x3_range) *
        " fts_range=" * fmt_num(comps.fts_range) *
        " cont=" * fmt_num(comps.continuity))
      logmsg("  rec: x1=" * fmt_pct(recon.x1) * "% x3=" * fmt_pct(recon.x3) * "%")
      logmsg("  nn: F_contact err=" * fmt_pct(nn_diag.fcontact_err) * "%")
      logmsg("  nn raw: min=" * fmt_num(nn_diag.raw_min) *
        " max=" * fmt_num(nn_diag.raw_max) *
        " span=" * fmt_num(nn_diag.raw_span) *
        " mean=" * fmt_num(nn_diag.raw_mean) *
        " neg=" * fmt_pct(100 * nn_diag.raw_neg_frac) * "%")
      logmsg("  g_nn=" * fmt_num(current_nn_gain_ref[], sigdigits=4))
      logmsg("  mech: ks=" * fmt_num(ks_est) * " (err=" * fmt_pct(ks_err) * "%)" *
        " cs=" * fmt_num(cs_est) * " (err=" * fmt_pct(cs_err) * "%)")
      if stage_ref[] == "lbfgs"
        logmsg("  lbfgs: step_norm=" * fmt_num(step_norm) *
          " dloss=" * fmt_num(loss_delta) *
          " sTy=" * fmt_num(secant_sy) *
          " curv=" * fmt_num(secant_curv) *
          " g_dot_step=" * fmt_num(grad_dot_step))
      else
        logmsg("  adam: step_norm=" * fmt_num(step_norm) *
          " dloss=" * fmt_num(loss_delta))
      end
      if loss_rebound
        if loss_jump
          logmsg("  loss-rebound: observed (jump > " * @sprintf("%.1f", 100 * step_max_loss_increase_frac) *
            "%) | no lr/p_net change")
        else
          logmsg("  loss-rebound: observed | no lr/p_net change")
        end
      end

      prev_theta_vec_ref[] = copy(Vector(θ))
      prev_grad_vec_ref[] = grad_vec
    end
    effective_l = l
    if step2b_dynamic_gnn
      gnn_epoch = current_nn_gain_ref[]
      gnn_info = gnn_update_fn(θ, gnn_epoch)
      gnn_next = gnn_info.gain
      gnn_changed = !(isfinite(gnn_epoch) && isfinite(gnn_next) &&
        isapprox(gnn_epoch, gnn_next; rtol=1e-12, atol=0.0))
      if gnn_changed
        gnn_loss_before = effective_l
        gnn_val_before = try
          validation_loss_function(θ, validation_solution_dataframe)
        catch
          NaN
        end
        current_nn_gain_ref[] = gnn_next
        effective_l = loss_function(θ)
        maybe_update_best!(θ, effective_l)
        gnn_val_after = try
          validation_loss_function(θ, validation_solution_dataframe)
        catch
          NaN
        end
        logmsg("  gnn-loss-flow" *
          " | gnn_old=" * fmt_num(gnn_epoch, sigdigits=4) *
          " gnn_new=" * fmt_num(gnn_next, sigdigits=4) *
          " | train_before=" * fmt_num(gnn_loss_before, sigdigits=4) *
          " after_gnn=" * fmt_num(effective_l, sigdigits=4) *
          " | val_before=" * fmt_num(gnn_val_before, sigdigits=4) *
          " val_after=" * fmt_num(gnn_val_after, sigdigits=4) *
          " | reason=" * string(gnn_info.reason))
      end
    end
    prev_epoch_loss_ref[] = effective_l
    prev_loss_ref[] = effective_l
    if stage_ref[] == "lbfgs"
      push!(lbfgs_recent, effective_l)
      if length(lbfgs_recent) > plateau_window
        deleteat!(lbfgs_recent, 1)
      end
      if plateau_early_stop && length(lbfgs_recent) == plateau_window
        window_improve = abs(lbfgs_recent[end] - lbfgs_recent[1])
        plateau_thresh = plateau_rel_tol * max(abs(effective_l), eps(Float64))
        if window_improve < plateau_thresh
          tprintln("Run ", run_id, " LBFGS early-stop: plateau_5ep (|Δloss| < 1% of current loss over 5 epochs), Δ=", window_improve,
            " thresh=", plateau_thresh)
          return true
        end
      end
    else
      push!(adam_recent, effective_l)
      if length(adam_recent) > plateau_window
        deleteat!(adam_recent, 1)
      end
      if plateau_early_stop && length(adam_recent) == plateau_window
        window_improve = abs(adam_recent[end] - adam_recent[1])
        plateau_thresh = plateau_rel_tol * max(abs(effective_l), eps(Float64))
        if window_improve < plateau_thresh
          tprintln("Run ", run_id, " Adam early-stop: plateau_5ep (|Δloss| < 1% of current loss over 5 epochs), Δ=", window_improve,
            " thresh=", plateau_thresh)
          return true
        end
      end
    end
    if good_enough_loss > 0.0 && effective_l < good_enough_loss
      tprintln("Run ", run_id, " ", uppercase(stage_ref[]), " early-stop: good_enough (loss < ", good_enough_loss, ") at epoch ", epoch,
        " (loss=", effective_l, ")")
      return true
    end
    return false
  end

  checkpoint_holder = Ref{Any}(resume_checkpoint)
  function save_run_checkpoint(run_state)
    checkpoint_holder[] = (
      run_id=run_id,
      stage=String(stage_ref[]),
      source=guess.source,
      ks0=Float64(ks0),
      cs0=Float64(cs0),
      num_hidden_layers=guess_num_hidden_layers,
      num_hidden_nodes=guess_num_hidden_nodes,
      best_theta=deepcopy(best_training_parameters[1]),
      best_loss=best_loss_ref[],
      best_gnn=best_training_gnn[],
      current_nn_gain=current_nn_gain_ref[],
      lbfgs_recent=copy(lbfgs_recent),
      prev_loss=prev_loss_ref[],
      prev_epoch_loss=prev_epoch_loss_ref[],
      run_state=run_state
    )
    active_checkpoint_ref[] = checkpoint_holder[]
    save_step2b_partial(results, run_summaries, checkpoint_holder[])
    tprintln("Step2b checkpoint saved: run=", run_id,
      " | stage=", stage_ref[],
      " | epoch=", field_or(run_state, :epoch, 0))
  end

  adtype = Optimization.AutoZygote()
  optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, starting_point_in)
  if use_adam_stage && max_adam_iters > 0
    opt = OptimizationOptimisers.Adam(learning_rate_adam)
    stage_ref[] = "adam"
    Optimization.solve(optprob, opt, callback=callback, maxiters=max_adam_iters)
  else
    tprintln("Run ", run_id, " skipping Adam stage; inherited initialization enters LBFGS directly.")
  end

  stage_ref[] = "lbfgs"
  if resume_checkpoint === nothing || string(field_or(resume_checkpoint, :stage, "lbfgs")) != "lbfgs"
    empty!(lbfgs_recent)
  end
  prev_theta_vec_ref[] = Vector{Float64}()
  prev_grad_vec_ref[] = Vector{Float64}()
  if resume_checkpoint === nothing || string(field_or(resume_checkpoint, :stage, "lbfgs")) != "lbfgs"
    prev_loss_ref[] = NaN
    prev_epoch_loss_ref[] = NaN
  end
  lbfgs_resume_state = (resume_checkpoint !== nothing &&
    string(field_or(resume_checkpoint, :stage, "lbfgs")) == "lbfgs") ?
    field_or(resume_checkpoint, :run_state, nothing) : nothing
  run_local_lbfgs(loss_function, best_training_parameters[1];
    callback=callback,
    maxiters=max_lbfgs_iters,
    allow_f_increases=true,
    resume_state=lbfgs_resume_state,
    checkpoint_every=step2b_checkpoint_every,
    checkpoint_fn=save_run_checkpoint)

  best_parameterization = best_training_parameters[1]
  best_gnn = best_training_gnn[]
  nn_gain_prev = current_nn_gain_ref[]
  current_nn_gain_ref[] = best_gnn
  validation_resulting_cost = validation_loss_function(best_parameterization, validation_solution_dataframe)

  p_est = build_parameter_vector(best_parameterization.p)
  best_loss = loss_function(best_parameterization)
  recon = last_recon_metrics[]
  current_nn_gain_ref[] = nn_gain_prev
  ks_est = p_est[10]
  cs_est = p_est[11]
  ks_err = percent_error_pct(ks_est, ks)
  cs_err = percent_error_pct(cs_est, cs)
  tprintln("Run ", run_id, " summary: train_loss=", best_loss, " val_loss=", validation_resulting_cost,
    " | ks=", @sprintf("%.3e", ks_est), " (", @sprintf("%.2f", ks_err), "%)",
    " cs=", @sprintf("%.3e", cs_est), " (", @sprintf("%.2f", cs_err), "%)",
    " | x1_rec=", @sprintf("%.2f", recon.x1), "% x3_rec=", @sprintf("%.2f", recon.x3), "%",
    " | initial guess ks0=", @sprintf("%.3e", ks0),
    " cs0=", @sprintf("%.3e", cs0),
    " | nn(layers=", guess_num_hidden_layers, ", nodes=", guess_num_hidden_nodes, ")",
    " | g_nn=", @sprintf("%.3e", best_gnn))
  push!(run_summaries, (
    run_id=run_id,
    best_loss=best_loss,
    validation_loss=validation_resulting_cost,
    ks=ks_est,
    cs=cs_est,
    ks_err=ks_err,
    cs_err=cs_err,
    x1_rec=recon.x1,
    x3_rec=recon.x3,
    ks0=ks0,
    cs0=cs0,
    nn_gain=best_gnn,
    num_hidden_layers=guess_num_hidden_layers,
    num_hidden_nodes=guess_num_hidden_nodes,
    source=guess.source
  ))
  if use_u0_x3_param
    u0_x3_best = bound_u0_x3(best_parameterization.u0_x3_ctrl)
    u0_x3_t0 = use_multiple_shooting ? interp_value(segment_times_ref[][1], control_times_ref[], u0_x3_best) : u0_x3_best[1]
  else
    u0_x3_best = [ode_data[3, 1]]
    u0_x3_t0 = u0_x3_best[1]
  end
  result = (
    run_id=run_id,
    parameters_training=p_est,
    initial_state_training=[ode_data[1, 1], ode_data[2, 1], u0_x3_t0],
    u0_x3=u0_x3_best,
    p_net=best_parameterization.p_net,
    nn_gain=best_gnn,
    num_hidden_layers=guess_num_hidden_layers,
    num_hidden_nodes=guess_num_hidden_nodes,
    nn_force_mode="contact_net",
    validation_resulting_cost=validation_resulting_cost,
    initial_guess=(ks0=ks0, cs0=cs0),
    best_loss=best_loss,
    parameter_errors=(ks_err=ks_err, cs_err=cs_err),
    source=guess.source,
    status="success"
  )

  push!(results, result)
  push!(completed_run_ids, run_id)
  active_checkpoint_ref[] = nothing
  save_step2b_partial(results, run_summaries, active_checkpoint_ref[])
end

save_payload = (
  results=results,
  run_summaries=run_summaries,
  sanity=sanity_payload,
  window=window_metadata,
  step2b_resume_signature=resume_signature,
  active_checkpoint=nothing
)
serialize(step2b_result_path, save_payload)

  if !isempty(results)
    tprintln("Run summaries:")
  for s in run_summaries
    tprintln("Run ", s.run_id, " train_loss=", s.best_loss, " val_loss=", s.validation_loss,
      " | ks=", @sprintf("%.3e", s.ks), " (", @sprintf("%.2f", s.ks_err), "%)",
      " cs=", @sprintf("%.3e", s.cs), " (", @sprintf("%.2f", s.cs_err), "%)",
      " | x1_rec=", @sprintf("%.2f", s.x1_rec), "% x3_rec=", @sprintf("%.2f", s.x3_rec), "%",
      " | initial guess ks0=", @sprintf("%.3e", s.ks0),
      " cs0=", @sprintf("%.3e", s.cs0),
      " | g_nn=", @sprintf("%.3e", s.nn_gain))
  end
  validation_costs = [r.validation_resulting_cost for r in results]
  best_idx = argmin(validation_costs)
  best = results[best_idx]
  best_train_loss = best.best_loss
  best_val_loss = best.validation_resulting_cost
  p_best = best.parameters_training
  ks_best = p_best[10]
  cs_best = p_best[11]
  ks_err = percent_error_pct(ks_best, ks)
  cs_err = percent_error_pct(cs_best, cs)

  full_tsteps = original_solution_dataframe.t
  if use_multiple_shooting && use_u0_x3_param
    u0_x3_full = interp_value(full_tsteps[1], control_times_ref[], best.u0_x3)
  else
    u0_x3_full = best.u0_x3[1]
  end
  prob_full = remake(
    prob_pred;
    p=(p_net=best.p_net, ode_par=p_best),
    tspan=(full_tsteps[1], full_tsteps[end]),
    u0=[original_ode_data[1, 1], original_ode_data[2, 1], u0_x3_full]
  )
  local full_sol
  try
    full_sol = solve(prob_full, integrator; saveat=full_tsteps, reltol=reltol, abstol=abstol,
      sensealg=sensealg, unstable_check=(dt, u, p, t) -> any(x -> !isfinite(x), u) || any(abs.(u) .> 1e7),
      verbose=false)
  catch
    full_sol = nothing
  end

  tprintln("Final best params (run ", best_idx, "): train_loss=", best_train_loss, " val_loss=", best_val_loss,
    " | ks=", @sprintf("%.3e", ks_best),
    " (", @sprintf("%.2f", ks_err), "%) cs=", @sprintf("%.3e", cs_best),
    " (", @sprintf("%.2f", cs_err), "%)")

  if full_sol !== nothing && retcode_success(full_sol)
    full_pred = Array(full_sol)
    x_true = original_ode_data[1, :]
    y_true = original_ode_data[3, :]
    x_pred = full_pred[1, :]
    y_pred = full_pred[3, :]
    x_err = relative_rmse_pct(x_pred, x_true, scale_eps)
    y_err = relative_rmse_pct(y_pred, y_true, scale_eps)
    tprintln("Reconstruction error: x=", @sprintf("%.2f", x_err), "% y=", @sprintf("%.2f", y_err), "%")
  else
    tprintln("Reconstruction error: solve failed for best parameters.")
  end
end
