#= 
Script to train AFM DMT-KV model parameters on the e0.0 dataset.
Partial x3 observation with multiple shooting and continuity penalty.
=#

cd(@__DIR__)

using ComponentArrays, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Statistics, Printf
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, DiffEqFlux

result_name_string = "afm_01.jld"

folder_name = "res_afm_01"
if !isdir(folder_name)
  mkdir(folder_name)
end

error_level = "e0.0"

include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_functions.jl")
include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")

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

tmp_steps = solution_dataframe.t
x2dot_data = solution_dataframe.x2dot

# Multiple shooting settings
small_group_size = 100
large_group_size = 800
boundary_window = 100
min_group_size = 10
continuity_term = 0.001

# Contact weighting
contact_loss_weight = 4.27
noncontact_loss_weight = 1.0

# Sparse x3 observation settings
x3_obs_fraction = 0.01
x3_contact_weight = contact_loss_weight
x3_noncontact_weight = 1.0
x3_u0_weight = 50.0
x3_u0_center = 0.0
x3_u0_amp = 20e-9


integrator = Rosenbrock23(autodiff=false)
abstol = 1e-8
reltol = 1e-8
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

datasize = size(ode_data, 2)
tspan = (tmp_steps[1], tmp_steps[end])

# Normalization scales (max-min)
scale_eps = 1e-9
state_scale = vec(maximum(ode_data, dims=2) - minimum(ode_data, dims=2))
state_scale = max.(state_scale, scale_eps)
x2dot_scale = max(maximum(x2dot_data) - minimum(x2dot_data), scale_eps)
state12_scale = state_scale[1:2]
x3_scale = max(state_scale[3], scale_eps)

# Sanity check thresholds (adjust if needed)
sanity_tol_data = 1e-6
sanity_tol_integrated = 1e-4

last_loss_components = Ref((state=0.0, x3=0.0, x3_obs=0.0, x3_u0=0.0, x2dot=0.0, continuity=0.0))
last_recon_metrics = Ref((x1=0.0, x3=0.0))

ranges_ref = Ref(UnitRange{Int}[])
range_is_contact_ref = Ref(Bool[])
x3_obs_mask_ref = Ref(BitVector())
x3_obs_weights_ref = Ref(Float64[])
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

function build_x3_observation_mask(contact::AbstractVector{Bool}, ranges::Vector{UnitRange{Int}}, range_is_contact::Vector{Bool}, fraction::Float64, rng)
  n = length(contact)
  mask = falses(n)
  if n == 0
    return mask
  end

  n_obs = max(1, round(Int, n * fraction))
  contact_idx = findall(contact)
  noncontact_idx = findall(.!contact)
  contact_ratio = isempty(contact_idx) ? 0.0 : length(contact_idx) / n
  n_contact = round(Int, n_obs * contact_ratio)
  if !isempty(contact_idx)
    n_contact = max(n_contact, 1)
  end
  n_contact = min(n_contact, n_obs)
  n_noncontact = n_obs - n_contact
  if n_noncontact == 0 && !isempty(noncontact_idx) && n_contact > 1
    n_noncontact = 1
    n_contact -= 1
  end

  contact_ranges = [rg for (i, rg) in enumerate(ranges) if range_is_contact[i]]
  noncontact_ranges = [rg for (i, rg) in enumerate(ranges) if !range_is_contact[i]]

  function pick_from_ranges!(ranges_list, n_target)
    for rg in shuffle(rng, ranges_list)
      if n_target <= 0
        break
      end
      idx = rand(rng, rg)
      if !mask[idx]
        mask[idx] = true
        n_target -= 1
      end
    end
    return n_target
  end

  n_contact = pick_from_ranges!(contact_ranges, n_contact)
  n_noncontact = pick_from_ranges!(noncontact_ranges, n_noncontact)

  if n_contact > 0
    cand = shuffle(rng, findall(i -> contact[i] && !mask[i], 1:n))
    for idx in cand[1:min(n_contact, length(cand))]
      mask[idx] = true
    end
  end
  if n_noncontact > 0
    cand = shuffle(rng, findall(i -> !contact[i] && !mask[i], 1:n))
    for idx in cand[1:min(n_noncontact, length(cand))]
      mask[idx] = true
    end
  end

  mask[1] = true
  return mask
end

function update_x3_observations!(contact::AbstractVector{Bool}, rng)
  ranges = ranges_ref[]
  labels = range_is_contact_ref[]
  x3_obs_mask_ref[] = build_x3_observation_mask(contact, ranges, labels, x3_obs_fraction, rng)
  x3_obs_weights_ref[] = ifelse.(contact, x3_contact_weight, x3_noncontact_weight)
  if !isempty(ranges)
    x3_u0_target_ref[] = ode_data[3, first(ranges[1])]
  end
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
Estar_bounds = (5e5, 5e8) #true Estar = 1.5e7

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

function initial_theta_from_guess(ks0, cs0, Estar0)
  return log.([ks0, cs0, Estar0])
end

function build_parameter_vector(theta)
  ks = exp(theta[1])
  cs = exp(theta[2])
  Estar = exp(theta[3])
  return [original_parameters[1:8]...; Estar; ks; cs]
end

function x2dot_rhs(u, p, t)
  k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs = p
  s = dist + u[1] - u[3]
  delta = softplus(-s, adhesion_transition)
  delta = ifelse(delta > 0.0, delta, 0.0)
  w = contact_weight(s, adhesion_transition)
  F_hertz = (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
  Fad_eff = Fad * w
  return (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
end

prob_pred = ODEProblem{true}(ground_truth_function, original_u0, tspan, original_parameters)

function loss_multiple_shooting_with_u0(p_est, u0_x3; debug::Bool=false)
  ranges = ranges_ref[]
  range_is_contact = range_is_contact_ref[]
  ks_est = p_est[10]
  cs_est = p_est[11]
  Estar_est = p_est[9]
  if any(x -> !isfinite(x), p_est) ||
     ks_est < ks_bounds[1] || ks_est > ks_bounds[2] ||
     cs_est < cs_bounds[1] || cs_est > cs_bounds[2] ||
     Estar_est < Estar_bounds[1] || Estar_est > Estar_bounds[2]
    if debug
      println("MS sanity fail: param guard. ks=", ks_est, " cs=", cs_est, " Estar=", Estar_est)
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
    local sol
    u0 = [ode_data[1, first(rg)], ode_data[2, first(rg)], u0_x3[i]]
    try
      sol = solve(
      remake(
        prob_pred;
        p=p_est,
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
    catch ex
      if debug
        println("MS sanity fail: solve exception at range ", i, " ex=", typeof(ex))
      end
      return Inf
    end
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
  x3_obs_sum = loss_zero
  x3_weight_sum = loss_zero
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
  x3_obs_mask = x3_obs_mask_ref[]
  x3_obs_weights = x3_obs_weights_ref[]
  x3_u0_target = x3_u0_target_ref[]
  for (i, rg) in enumerate(ranges)
    weight = range_is_contact[i] ? contact_loss_weight : noncontact_loss_weight
    weight_sum += weight
    u = ode_data[:, rg]
    û = group_predictions[i]
    state_loss += weight * (1 / size(u, 2) * sum(abs2.((u[1:2, :] .- û[1:2, :]) ./ state12_scale)))

    local_idx = findall(x3_obs_mask[rg])
    if !isempty(local_idx)
      obs_weights = x3_obs_weights[rg][local_idx]
      x3_err = (u[3, local_idx] .- û[3, local_idx]) ./ x3_scale
      x3_obs_sum += sum(obs_weights .* abs2.(x3_err))
      x3_weight_sum += sum(obs_weights)
    end

    if range_is_contact[i]
      x2dot_pred = [x2dot_rhs(û[:, j], p_est, tmp_steps[idx]) for (j, idx) in enumerate(rg)]
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
  if x3_weight_sum == 0.0
    if debug
      println("MS sanity fail: x3_weight_sum is zero")
    end
    return Inf
  end
  state_loss /= weight_sum
  x3_obs_loss = x3_obs_sum / x3_weight_sum
  if x2dot_weight_sum > 0.0
    x2dot_loss /= x2dot_weight_sum
  else
    x2dot_loss = 0.0
  end

  continuity_loss = loss_zero
  for (i, rg) in enumerate(ranges)
    if i == 1
      continue
    end
    u0 = group_predictions[i-1][:, end]
    u1 = group_predictions[i][:, 1]
    continuity_loss += continuity_term * sum(abs2, u0 - u1)
  end

  x3_u0_loss = x3_u0_weight * abs2((u0_x3[1] - x3_u0_target) / x3_scale)
  x3_loss = x3_obs_loss + x3_u0_loss

  if eltype(p_est) == Float64
    last_loss_components[] = (state=state_loss, x3=x3_loss, x3_obs=x3_obs_loss, x3_u0=x3_u0_loss, x2dot=x2dot_loss, continuity=continuity_loss)
    if x1_count > 0
      x1_rmse = sqrt(x1_err_sum / x1_count)
      x1_denom = sqrt(x1_truth_sum / x1_count) + scale_eps
      x1_recon = 100.0 * x1_rmse / x1_denom
      x3_rmse = sqrt(x3_err_sum / x3_count)
      x3_denom = sqrt(x3_truth_sum / x3_count) + scale_eps
      x3_recon = 100.0 * x3_rmse / x3_denom
      last_recon_metrics[] = (x1=x1_recon, x3=x3_recon)
    end
  end
  return state_loss + x3_loss + x2dot_loss + continuity_loss
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
  return loss_multiple_shooting_with_u0(p_est, u0_x3; debug=debug)
end

function validation_loss_multiple_shooting(θ, validation_df)
  p_est = build_parameter_vector(θ.p)
  tsteps_val = validation_df.t
  contact_val = validation_df.contact .== 1
  weights_val = ifelse.(contact_val, contact_loss_weight, noncontact_loss_weight)
  ks_est = p_est[10]
  cs_est = p_est[11]
  Estar_est = p_est[9]
  u0_ctrl = bound_u0_x3(θ.u0_x3_ctrl)
  ctrl_times = control_times_ref[]
  if length(u0_ctrl) != length(ctrl_times)
    return Inf
  end
  if any(x -> !isfinite(x), p_est) ||
     ks_est < ks_bounds[1] || ks_est > ks_bounds[2] ||
     cs_est < cs_bounds[1] || cs_est > cs_bounds[2] ||
     Estar_est < Estar_bounds[1] || Estar_est > Estar_bounds[2]
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
    p=p_est,
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
    u = x[:, i]
    x2dot_pred[i] = x2dot_rhs(u, p_est, tsteps_val[i])
  end

  state_err = sum(abs2.((Array(validation_df[:, [:x1, :x2]])' .- x[1:2, :]) ./ state12_scale); dims=1)
  weights_sum = sum(weights_val)
  if weights_sum == 0.0
    return Inf
  end
  loss = 1 / weights_sum * sum(weights_val .* vec(state_err))

  obs_idx = findall(validation_x3_obs_mask)
  if isempty(obs_idx)
    return Inf
  end
  x3_err = (validation_df.x3[obs_idx] .- x[3, obs_idx]) ./ x3_scale
  x3_obs_loss = sum(validation_x3_obs_weights[obs_idx] .* abs2.(x3_err)) / sum(validation_x3_obs_weights[obs_idx])
  x3_u0_loss = x3_u0_weight * abs2((u0_x3_val - validation_df.x3[1]) / x3_scale)
  loss += x3_obs_loss + x3_u0_loss

  weights_x2dot = ifelse.(contact_val, contact_loss_weight, 0.0)
  weights_x2dot_sum = sum(weights_x2dot)
  if weights_x2dot_sum > 0.0
    loss += 1 / weights_x2dot_sum * sum(weights_x2dot .* abs2.((validation_df.x2dot .- x2dot_pred) ./ x2dot_scale))
  end

  return loss
end

function validation_loss_function(θ, validation_df)
  return validation_loss_multiple_shooting(θ, validation_df)
end

# Sanity check with true parameters (data-based and integrated)
function sanity_check_data()
  ranges = ranges_ref[]
  range_is_contact = range_is_contact_ref[]
  x3_obs_mask = x3_obs_mask_ref[]
  x3_obs_weights = x3_obs_weights_ref[]
  x3_u0_target = x3_u0_target_ref[]
  state_loss = 0.0
  x3_obs_sum = 0.0
  x3_weight_sum = 0.0
  x2dot_loss = 0.0
  x2dot_weight_sum = 0.0
  weight_sum = 0.0
  for (i, rg) in enumerate(ranges)
    weight = range_is_contact[i] ? contact_loss_weight : noncontact_loss_weight
    weight_sum += weight
    u = ode_data[:, rg]
    u_hat = u
    state_loss += weight * (1 / size(u, 2) * sum(abs2.((u[1:2, :] .- u_hat[1:2, :]) ./ state12_scale)))

    local_idx = findall(x3_obs_mask[rg])
    if !isempty(local_idx)
      obs_weights = x3_obs_weights[rg][local_idx]
      x3_err = (u[3, local_idx] .- u_hat[3, local_idx]) ./ x3_scale
      x3_obs_sum += sum(obs_weights .* abs2.(x3_err))
      x3_weight_sum += sum(obs_weights)
    end

    if range_is_contact[i]
      x2dot_pred = similar(x2dot_data[rg])
      for (j, idx) in enumerate(rg)
        x2dot_pred[j] = x2dot_rhs(u[:, j], original_parameters, tmp_steps[idx])
      end
      x2dot_loss += weight * (1 / length(rg) * sum(abs2.((x2dot_data[rg] .- x2dot_pred) ./ x2dot_scale)))
      x2dot_weight_sum += weight
    end
  end
  if weight_sum == 0.0
    return Inf
  end
  state_loss /= weight_sum
  x3_obs_loss = x3_weight_sum == 0.0 ? Inf : (x3_obs_sum / x3_weight_sum)
  if x2dot_weight_sum > 0.0
    x2dot_loss /= x2dot_weight_sum
  else
    x2dot_loss = 0.0
  end

  x3_u0_loss = x3_u0_weight * abs2((ode_data[3, first(ranges[1])] - x3_u0_target) / x3_scale)
  x3_loss = x3_obs_loss + x3_u0_loss

  total = state_loss + x3_loss + x2dot_loss
  return total, (state=state_loss, x3=x3_loss, x3_obs=x3_obs_loss, x3_u0=x3_u0_loss, x2dot=x2dot_loss, continuity=0.0)
end

function sanity_check_integrated()
  prob = remake(
    prob_pred;
    p=original_parameters,
    tspan=(tmp_steps[1], tmp_steps[end]),
    u0=ode_data[:, 1]
  )

  solutions = solve(prob, integrator; saveat=tmp_steps, reltol=reltol, abstol=abstol,
    sensealg=sensealg, verbose=false)
  x = Array(solutions)

  if size(x)[2] != length(tmp_steps)
    return Inf
  end

  x2dot_pred = similar(x2dot_data)
  for i in eachindex(tmp_steps)
    u = x[:, i]
    x2dot_pred[i] = x2dot_rhs(u, original_parameters, tmp_steps[i])
  end

  contact_val = solution_dataframe.contact .== 1
  weights_val = ifelse.(contact_val, contact_loss_weight, noncontact_loss_weight)
  weights_x2dot = ifelse.(contact_val, contact_loss_weight, 0.0)
  state_err = sum(abs2.((ode_data[1:2, :] .- x[1:2, :]) ./ state12_scale); dims=1)
  weights_sum = sum(weights_val)
  if weights_sum == 0.0
    return Inf
  end
  state_loss = 1 / weights_sum * sum(weights_val .* vec(state_err))

  x3_obs_mask = x3_obs_mask_ref[]
  x3_obs_weights = x3_obs_weights_ref[]
  obs_idx = findall(x3_obs_mask)
  if isempty(obs_idx)
    return Inf
  end
  x3_err = (ode_data[3, obs_idx] .- x[3, obs_idx]) ./ x3_scale
  x3_obs_loss = sum(x3_obs_weights[obs_idx] .* abs2.(x3_err)) / sum(x3_obs_weights[obs_idx])
  x3_u0_target = x3_u0_target_ref[]
  x3_u0_loss = x3_u0_weight * abs2((ode_data[3, 1] - x3_u0_target) / x3_scale)
  x3_loss = x3_obs_loss + x3_u0_loss

  weights_x2dot_sum = sum(weights_x2dot)
  if weights_x2dot_sum > 0.0
    x2dot_loss = 1 / weights_x2dot_sum * sum(weights_x2dot .* abs2.((x2dot_data .- x2dot_pred) ./ x2dot_scale))
  else
    x2dot_loss = 0.0
  end
  total = state_loss + x3_loss + x2dot_loss
  return total, (state=state_loss, x3=x3_loss, x3_obs=x3_obs_loss, x3_u0=x3_u0_loss, x2dot=x2dot_loss, continuity=0.0)
end

function sanity_check_integrated_ms_oracle()
  ranges = ranges_ref[]
  u0_x3 = isempty(ranges) ? Float64[] : ode_data[3, first.(ranges)]
  true_theta = initial_theta_from_guess(ks, cs, Estar)
  p_est = build_parameter_vector(true_theta)
  total = loss_multiple_shooting_with_u0(p_est, u0_x3; debug=true)
  comps = last_loss_components[]
  return total, comps
end

function sanity_check_integrated_ms_model()
  true_theta = initial_theta_from_guess(ks, cs, Estar)
  ranges = ranges_ref[]
  if use_interpolation
    ctrl_idx = control_segments_ref[]
    u0_x3 = isempty(ctrl_idx) ? Float64[] : ode_data[3, first.(ranges[ctrl_idx])]
  else
    u0_x3 = isempty(ranges) ? Float64[] : ode_data[3, first.(ranges)]
  end
  u0_x3_raw = isempty(u0_x3) ? Float64[] : unbound_u0_x3(u0_x3)
  θ_true = ComponentVector(p=true_theta, u0_x3_ctrl=u0_x3_raw)
  total = loss_multiple_shooting(θ_true; debug=true)
  comps = last_loss_components[]
  return total, comps
end

update_ranges!(solution_dataframe.contact .== 1)
update_x3_observations!(solution_dataframe.contact .== 1, Random.default_rng())
update_control_points!()

sanity_data, sanity_data_comps = sanity_check_data()
sanity_integrated, sanity_integrated_comps = sanity_check_integrated()
println("Sanity check loss (data-based, post-contact): ", sanity_data, " | state=", sanity_data_comps.state, " x3=", sanity_data_comps.x3, " (obs=", sanity_data_comps.x3_obs, " u0=", sanity_data_comps.x3_u0, ") x2dot=", sanity_data_comps.x2dot, " cont=", sanity_data_comps.continuity)
println("Sanity check loss (integrated, post-contact): ", sanity_integrated, " | state=", sanity_integrated_comps.state, " x3=", sanity_integrated_comps.x3, " (obs=", sanity_integrated_comps.x3_obs, " u0=", sanity_integrated_comps.x3_u0, ") x2dot=", sanity_integrated_comps.x2dot, " cont=", sanity_integrated_comps.continuity)
sanity_integrated_ms_oracle, sanity_integrated_ms_oracle_comps = sanity_check_integrated_ms_oracle()
sanity_integrated_ms_model, sanity_integrated_ms_model_comps = sanity_check_integrated_ms_model()
println("Sanity check loss (integrated, multiple shooting: oracle): ", sanity_integrated_ms_oracle, " | state=", sanity_integrated_ms_oracle_comps.state, " x3=", sanity_integrated_ms_oracle_comps.x3, " (obs=", sanity_integrated_ms_oracle_comps.x3_obs, " u0=", sanity_integrated_ms_oracle_comps.x3_u0, ") x2dot=", sanity_integrated_ms_oracle_comps.x2dot, " cont=", sanity_integrated_ms_oracle_comps.continuity)
println("Sanity check loss (integrated, multiple shooting: model): ", sanity_integrated_ms_model, " | state=", sanity_integrated_ms_model_comps.state, " x3=", sanity_integrated_ms_model_comps.x3, " (obs=", sanity_integrated_ms_model_comps.x3_obs, " u0=", sanity_integrated_ms_model_comps.x3_u0, ") x2dot=", sanity_integrated_ms_model_comps.x2dot, " cont=", sanity_integrated_ms_model_comps.continuity)
sanity_data_gate = sanity_data_comps.state + sanity_data_comps.x3 + sanity_data_comps.x2dot
sanity_integrated_gate = sanity_integrated_comps.state + sanity_integrated_comps.x3 + sanity_integrated_comps.x2dot
if sanity_data_gate > sanity_tol_data || sanity_integrated_gate > sanity_tol_integrated
  error("Sanity check failed. data=" * string(sanity_data_gate) * ", integrated=" * string(sanity_integrated_gate))
end

# Training/validation split
shuffled_positions = shuffle(2:size(solution_dataframe, 1))
first_validation = rand(2:5)
validation_mask = [(first_validation + k * 5) for k in 0:3]
training_mask = [j for j in 1:size(solution_dataframe, 1) if !(j in validation_mask)]

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
validation_x3_obs_mask = build_x3_observation_mask(val_contact, val_ranges, val_labels, x3_obs_fraction, rng)
validation_x3_obs_weights = ifelse.(val_contact, x3_contact_weight, x3_noncontact_weight)

tmp_steps = solution_dataframe.t
datasize = size(ode_data, 2)
tspan = (tmp_steps[1], tmp_steps[end])
update_ranges!(solution_dataframe.contact .== 1)
update_x3_observations!(solution_dataframe.contact .== 1, rng)
update_control_points!()

learning_rate_adam = 1e-3
max_adam_iters = 1000
max_lbfgs_iters = 1000
print_interval_adam = 1
print_interval_lbfgs = 1


num_random_initial_guesses = 3
use_stage2_init = true
stage2_mode = :topk # :best or :topk
stage2_topk = 3
append_random_after_stage2 = false
stage2_path = "../step2a_hyperparameter_tuning/hyperparameter_tuning_second_stage/results_afm/afm_param_stage2_01.jld"

results = []

function load_stage2_initial_guesses()
  if !use_stage2_init
    return NamedTuple[]
  end
  if !isfile(stage2_path)
    println("Stage2 init enabled but file not found: ", stage2_path)
    return NamedTuple[]
  end
  stage2 = deserialize(stage2_path)
  if stage2_mode == :best
    b = stage2.best
    return [(ks=b.ks, cs=b.cs, Estar=b.Estar, source="stage2_best")]
  elseif stage2_mode == :topk
    res_sorted = sort(stage2.results, by = r -> r.loss)
    k = min(stage2_topk, length(res_sorted))
    return [(ks=res_sorted[i].ks, cs=res_sorted[i].cs, Estar=res_sorted[i].Estar,
      source="stage2_top$(k)_$(i)") for i in 1:k]
  else
    println("Unknown stage2_mode=", stage2_mode, " (use :best or :topk).")
    return NamedTuple[]
  end
end
range_starts = [first(rg) for rg in ranges_ref[]]
if use_interpolation
  ctrl_idx = control_segments_ref[]
  initial_u0_x3_ctrl = isempty(ctrl_idx) ? Float64[] : unbound_u0_x3(ode_data[3, range_starts[ctrl_idx]])
else
  initial_u0_x3_ctrl = isempty(range_starts) ? Float64[] : unbound_u0_x3(ode_data[3, range_starts])
end
run_summaries = []

stage_ref = Ref("adam")
adam_loss_target = 1e-4
lbfgs_loss_target = 1e-6

stage2_inits = load_stage2_initial_guesses()
initial_guesses = NamedTuple[]
if isempty(stage2_inits)
  for run_id in 1:num_random_initial_guesses
    band = band_for_run(run_id)
    ks0 = sample_log_band(rng, ks_bounds[1], ks_bounds[2], band)
    cs0 = sample_log_band(rng, cs_bounds[1], cs_bounds[2], band)
    Estar0 = sample_log_band(rng, Estar_bounds[1], Estar_bounds[2], band)
    push!(initial_guesses, (ks=ks0, cs=cs0, Estar=Estar0, source=string(band)))
  end
else
  append!(initial_guesses, stage2_inits)
  if append_random_after_stage2
    for run_id in 1:num_random_initial_guesses
      band = band_for_run(run_id)
      ks0 = sample_log_band(rng, ks_bounds[1], ks_bounds[2], band)
      cs0 = sample_log_band(rng, cs_bounds[1], cs_bounds[2], band)
      Estar0 = sample_log_band(rng, Estar_bounds[1], Estar_bounds[2], band)
      push!(initial_guesses, (ks=ks0, cs=cs0, Estar=Estar0, source=string(band)))
    end
  end
end

for run_id in 1:length(initial_guesses)
  guess = initial_guesses[run_id]
  ks0 = guess.ks
  cs0 = guess.cs
  Estar0 = guess.Estar
  theta0 = initial_theta_from_guess(ks0, cs0, Estar0)

  ks0_err = percent_error_pct(ks0, ks)
  cs0_err = percent_error_pct(cs0, cs)
  Estar0_err = percent_error_pct(Estar0, Estar)
  println("Run ", run_id, " initial guess (", guess.source, "): ks0=", @sprintf("%.3e", ks0),
    " (true=", @sprintf("%.3e", ks), ", ", @sprintf("%.2f", ks0_err), "%)",
    " cs0=", @sprintf("%.3e", cs0),
    " (true=", @sprintf("%.3e", cs), ", ", @sprintf("%.2f", cs0_err), "%)",
    " Estar0=", @sprintf("%.3e", Estar0),
    " (true=", @sprintf("%.3e", Estar), ", ", @sprintf("%.2f", Estar0_err), "%)")

  p0 = ComponentArray(theta0)
  starting_point_in = ComponentVector(p=p0, u0_x3_ctrl=initial_u0_x3_ctrl)

  init_loss = try
    loss_multiple_shooting(starting_point_in; debug=true)
  catch ex
    println("Run ", run_id, " init diagnostics exception: ", typeof(ex), " - ", ex)
    Inf
  end
  init_comps = last_loss_components[]
  init_recon = last_recon_metrics[]
  if !isfinite(init_loss)
    init_comps = (state=NaN, x3=NaN, x3_obs=NaN, x3_u0=NaN, x2dot=NaN, continuity=NaN)
    init_recon = (x1=NaN, x3=NaN)
    println("Run ", run_id, " init diagnostics: no guard fired (loss=Inf)")
  end
  println("Run ", run_id, " init diagnostics (", guess.source, "): loss=", @sprintf("%.4e", init_loss),
    " | x1_rec=", @sprintf("%.2f", init_recon.x1), "% x3_rec=", @sprintf("%.2f", init_recon.x3), "%",
    " | state=", @sprintf("%.4e", init_comps.state),
    " x3=", @sprintf("%.4e", init_comps.x3), " (obs=", @sprintf("%.4e", init_comps.x3_obs),
    " u0=", @sprintf("%.4e", init_comps.x3_u0), ")",
    " x2dot=", @sprintf("%.4e", init_comps.x2dot),
    " cont=", @sprintf("%.4e", init_comps.continuity))

  best_training_parameters = [starting_point_in]
  training_epochs = zeros(Int, max_adam_iters)
  training_costs = fill(Inf, max_adam_iters)
  adam_epoch = Ref(0)
  lbfgs_epoch = Ref(0)
  best_loss_ref = Ref(Inf)
  lbfgs_recent = Float64[]
  function callback(θ, l)
    t_start = time()
    if stage_ref[] == "adam"
      adam_epoch[] += 1
      epoch = adam_epoch[]
      if epoch <= length(training_epochs)
        training_epochs[epoch] = epoch
        training_costs[epoch] = l
      end
    else
      lbfgs_epoch[] += 1
      epoch = lbfgs_epoch[]
    end
    if l < best_loss_ref[]
      best_loss_ref[] = l
      best_training_parameters[1] = deepcopy(θ)
    end
    if (stage_ref[] == "adam" && epoch % print_interval_adam == 0) ||
       (stage_ref[] == "lbfgs" && epoch % print_interval_lbfgs == 0)
      comps = last_loss_components[]
      recon = last_recon_metrics[]
      p_est = build_parameter_vector(θ.p)
      ks_est = p_est[10]
      cs_est = p_est[11]
      Estar_est = p_est[9]
      ks_err = percent_error_pct(ks_est, ks)
      cs_err = percent_error_pct(cs_est, cs)
      Estar_err = percent_error_pct(Estar_est, Estar)
      iter_seconds = time() - t_start
      println("Run ", run_id, " ", stage_ref[], " Epoch ", epoch, " -- cost: ", l,
        " | state=", comps.state, " x3=", comps.x3, " (obs=", comps.x3_obs, " u0=", comps.x3_u0, ") x2dot=", comps.x2dot, " cont=", comps.continuity,
        " | iter_s=", @sprintf("%.1f", iter_seconds),
        " | x1_rec=", @sprintf("%.2f", recon.x1), "% x3_rec=", @sprintf("%.2f", recon.x3), "%",
        " | ks=", @sprintf("%.3e", ks_est), " (", @sprintf("%.2f", ks_err), "%)",
        " cs=", @sprintf("%.3e", cs_est), " (", @sprintf("%.2f", cs_err), "%)",
        " Estar=", @sprintf("%.3e", Estar_est), " (", @sprintf("%.2f", Estar_err), "%)")
    end
    if stage_ref[] == "lbfgs"
      push!(lbfgs_recent, l)
      if length(lbfgs_recent) > 10
        deleteat!(lbfgs_recent, 1)
      end
      if length(lbfgs_recent) == 10
        window_improve = maximum(lbfgs_recent) - minimum(lbfgs_recent)
        if window_improve < 1e-8
          println("Run ", run_id, " LBFGS plateau stop: last10 Δ=", window_improve)
          return true
        end
      end
    end
    if stage_ref[] == "adam" && l < adam_loss_target
      println("Run ", run_id, " Adam early stop at epoch ", epoch, " (loss=", l, ")")
      return true
    end
    if stage_ref[] == "lbfgs" && l < lbfgs_loss_target
      println("Run ", run_id, " LBFGS early stop at epoch ", epoch, " (loss=", l, ")")
      return true
    end
    return false
  end

  adtype = Optimization.AutoForwardDiff()
  optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, starting_point_in)
  opt = OptimizationOptimisers.Adam(learning_rate_adam)

  stage_ref[] = "adam"
  res = Optimization.solve(optprob, opt, callback=callback, maxiters=max_adam_iters)

  stage_ref[] = "lbfgs"
  optprob2 = remake(optprob, u0=best_training_parameters[1])
  res = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=max_lbfgs_iters, allow_f_increases=true)

  best_parameterization = best_training_parameters[1]
  validation_resulting_cost = validation_loss_function(best_parameterization, validation_solution_dataframe)

  p_est = build_parameter_vector(best_parameterization.p)
  best_loss = loss_multiple_shooting(best_parameterization)
  recon = last_recon_metrics[]
  ks_est = p_est[10]
  cs_est = p_est[11]
  Estar_est = p_est[9]
  ks_err = percent_error_pct(ks_est, ks)
  cs_err = percent_error_pct(cs_est, cs)
  Estar_err = percent_error_pct(Estar_est, Estar)
  println("Run ", run_id, " summary: train_loss=", best_loss, " val_loss=", validation_resulting_cost,
    " | ks=", @sprintf("%.3e", ks_est), " (", @sprintf("%.2f", ks_err), "%)",
    " cs=", @sprintf("%.3e", cs_est), " (", @sprintf("%.2f", cs_err), "%)",
    " Estar=", @sprintf("%.3e", Estar_est), " (", @sprintf("%.2f", Estar_err), "%)",
    " | x1_rec=", @sprintf("%.2f", recon.x1), "% x3_rec=", @sprintf("%.2f", recon.x3), "%",
    " | initial guess ks0=", @sprintf("%.3e", ks0),
    " cs0=", @sprintf("%.3e", cs0), " Estar0=", @sprintf("%.3e", Estar0))
  push!(run_summaries, (
    run_id=run_id,
    best_loss=best_loss,
    validation_loss=validation_resulting_cost,
    ks=ks_est,
    cs=cs_est,
    Estar=Estar_est,
    ks_err=ks_err,
    cs_err=cs_err,
    Estar_err=Estar_err,
    x1_rec=recon.x1,
    x3_rec=recon.x3,
    ks0=ks0,
    cs0=cs0,
    Estar0=Estar0
  ))
  u0_x3_best = bound_u0_x3(best_parameterization.u0_x3_ctrl)
  u0_x3_t0 = interp_value(segment_times_ref[][1], control_times_ref[], u0_x3_best)
  result = (
    parameters_training=p_est,
    initial_state_training=[ode_data[1, 1], ode_data[2, 1], u0_x3_t0],
    u0_x3=u0_x3_best,
    validation_resulting_cost=validation_resulting_cost,
    initial_guess=(ks0=ks0, cs0=cs0, Estar0=Estar0),
    best_loss=best_loss,
    parameter_errors=(ks_err=ks_err, cs_err=cs_err, Estar_err=Estar_err),
    status="success"
  )

  push!(results, result)
end

serialize(folder_name * "/" * result_name_string, results)

  if !isempty(results)
    println("Run summaries:")
  for s in run_summaries
    println("Run ", s.run_id, " train_loss=", s.best_loss, " val_loss=", s.validation_loss,
      " | ks=", @sprintf("%.3e", s.ks), " (", @sprintf("%.2f", s.ks_err), "%)",
      " cs=", @sprintf("%.3e", s.cs), " (", @sprintf("%.2f", s.cs_err), "%)",
      " Estar=", @sprintf("%.3e", s.Estar), " (", @sprintf("%.2f", s.Estar_err), "%)",
      " | x1_rec=", @sprintf("%.2f", s.x1_rec), "% x3_rec=", @sprintf("%.2f", s.x3_rec), "%",
      " | initial guess ks0=", @sprintf("%.3e", s.ks0),
      " cs0=", @sprintf("%.3e", s.cs0), " Estar0=", @sprintf("%.3e", s.Estar0))
  end
  validation_costs = [r.validation_resulting_cost for r in results]
  best_idx = argmin(validation_costs)
  best = results[best_idx]
  best_train_loss = best.best_loss
  best_val_loss = best.validation_resulting_cost
  p_best = best.parameters_training
  ks_best = p_best[10]
  cs_best = p_best[11]
  Estar_best = p_best[9]
  ks_err = percent_error_pct(ks_best, ks)
  cs_err = percent_error_pct(cs_best, cs)
  Estar_err = percent_error_pct(Estar_best, Estar)

  full_tsteps = original_solution_dataframe.t
  u0_x3_full = interp_value(full_tsteps[1], control_times_ref[], best.u0_x3)
  prob_full = remake(
    prob_pred;
    p=p_best,
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

  println("Final best params (run ", best_idx, "): train_loss=", best_train_loss, " val_loss=", best_val_loss,
    " | ks=", @sprintf("%.3e", ks_best),
    " (", @sprintf("%.2f", ks_err), "%) cs=", @sprintf("%.3e", cs_best),
    " (", @sprintf("%.2f", cs_err), "%) Estar=", @sprintf("%.3e", Estar_best),
    " (", @sprintf("%.2f", Estar_err), "%)")

  if full_sol !== nothing && retcode_success(full_sol)
    full_pred = Array(full_sol)
    x_true = original_ode_data[1, :]
    y_true = original_ode_data[3, :]
    x_pred = full_pred[1, :]
    y_pred = full_pred[3, :]
    x_err = relative_rmse_pct(x_pred, x_true, scale_eps)
    y_err = relative_rmse_pct(y_pred, y_true, scale_eps)
    println("Reconstruction error: x=", @sprintf("%.2f", x_err), "% y=", @sprintf("%.2f", y_err), "%")
  else
    println("Reconstruction error: solve failed for best parameters.")
  end
end
