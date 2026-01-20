#= 
Script to train AFM DMT-KV model parameters on the e0.0 dataset.
Uses multiple shooting with continuity penalty and bounded parameters.
=#

cd(@__DIR__)

using ComponentArrays, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Statistics, Printf
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, DiffEqFlux

result_name_string = "afm_00.jld"

folder_name = "res_afm"
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
large_group_size = 400
boundary_window = 200
min_group_size = 10
continuity_term = 0.001

# Contact weighting
contact_loss_weight = 4.27
noncontact_loss_weight = 1.0

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

# Sanity check thresholds (adjust if needed)
sanity_tol_data = 1e-6
sanity_tol_integrated = 1e-4

last_loss_components = Ref((state=0.0, x2dot=0.0, continuity=0.0))

ranges_ref = Ref(UnitRange{Int}[])
range_is_contact_ref = Ref(Bool[])

function retcode_success(sol)
  return string(sol.retcode) == "Success"
end

function percent_error_pct(est, truth)
  return 100.0 * abs((est - truth) / truth)
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
    if boundary_len <= 0 || run_len <= 2 * boundary_len
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

# Parameter bounds
ks_bounds = (0.01, 1.0)
cs_bounds = (1e-7, 1e-5)
Estar_bounds = (1e5, 1e8)

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

function loss_multiple_shooting(θ; debug::Bool=false)
  p_est = build_parameter_vector(θ.p)

  ranges = ranges_ref[]
  range_is_contact = range_is_contact_ref[]
  initial_points = ode_data
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

  function unstable_check(dt, u, p, t)
    if any(x -> !isfinite(x), u) || any(abs.(u) .> 1e7)
      return true
    end
    return false
  end

  sols = Vector{Any}(undef, length(ranges))
  for (i, rg) in enumerate(ranges)
    local sol
    try
      sol = solve(
      remake(
        prob_pred;
        p=p_est,
        tspan=(tmp_steps[first(rg)], tmp_steps[last(rg)]),
        u0=initial_points[:, first(rg)]
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
  x2dot_loss = loss_zero
  weight_sum = 0.0
  for (i, rg) in enumerate(ranges)
    weight = range_is_contact[i] ? contact_loss_weight : noncontact_loss_weight
    weight_sum += weight
    u = ode_data[:, rg]
    û = group_predictions[i]
    state_loss += weight * (1 / size(u, 2) * sum(abs2.((u .- û) ./ state_scale)))

    x2dot_pred = [x2dot_rhs(û[:, j], p_est, tmp_steps[idx]) for (j, idx) in enumerate(rg)]
    if any(x -> !isfinite(x), x2dot_pred)
      if debug
        println("MS sanity fail: nonfinite x2dot at range ", i)
      end
      return Inf
    end
    x2dot_loss += weight * (1 / length(rg) * sum(abs2.((x2dot_data[rg] .- x2dot_pred) ./ x2dot_scale)))
  end
  if weight_sum == 0.0
    if debug
      println("MS sanity fail: weight_sum is zero")
    end
    return Inf
  end
  state_loss /= weight_sum
  x2dot_loss /= weight_sum

  # Continuity term
  continuity_loss = loss_zero
  for (i, rg) in enumerate(ranges)
    if i == 1
      continue
    end
    u0 = group_predictions[i-1][:, end]
    u1 = group_predictions[i][:, 1]
    continuity_loss += continuity_term * sum(abs2, u0 - u1)
  end

  if eltype(p_est) == Float64
    last_loss_components[] = (state=state_loss, x2dot=x2dot_loss, continuity=continuity_loss)
  end
  return state_loss + x2dot_loss + continuity_loss
end

function validation_loss_function(θ, validation_df)
  p_est = build_parameter_vector(θ.p)
  tsteps_val = validation_df.t
  contact_val = validation_df.contact .== 1
  weights_val = ifelse.(contact_val, contact_loss_weight, noncontact_loss_weight)
  ks_est = p_est[10]
  cs_est = p_est[11]
  Estar_est = p_est[9]
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

  u0_val = [validation_df.x1[1], validation_df.x2[1], validation_df.x3[1]]
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

  x2dot_pred = similar(validation_df.x2dot)
  for i in eachindex(tsteps_val)
    u = x[:, i]
    x2dot_pred[i] = x2dot_rhs(u, p_est, tsteps_val[i])
  end

  state_err = sum(abs2.((Array(validation_df[:, [:x1, :x2, :x3]])' .- x) ./ state_scale); dims=1)
  weights_sum = sum(weights_val)
  if weights_sum == 0.0
    return Inf
  end
  loss = 1 / weights_sum * sum(weights_val .* vec(state_err))
  loss += 1 / weights_sum * sum(weights_val .* abs2.((validation_df.x2dot .- x2dot_pred) ./ x2dot_scale))

  return loss
end

# Sanity check with true parameters (data-based and integrated)
function sanity_check_data()
  ranges = ranges_ref[]
  range_is_contact = range_is_contact_ref[]
  state_loss = 0.0
  x2dot_loss = 0.0
  weight_sum = 0.0
  for (i, rg) in enumerate(ranges)
    weight = range_is_contact[i] ? contact_loss_weight : noncontact_loss_weight
    weight_sum += weight
    u = ode_data[:, rg]
    u_hat = u
    state_loss += weight * (1 / size(u, 2) * sum(abs2.((u .- u_hat) ./ state_scale)))

    x2dot_pred = similar(x2dot_data[rg])
    for (j, idx) in enumerate(rg)
      x2dot_pred[j] = x2dot_rhs(u[:, j], original_parameters, tmp_steps[idx])
    end
    x2dot_loss += weight * (1 / length(rg) * sum(abs2.((x2dot_data[rg] .- x2dot_pred) ./ x2dot_scale)))
  end
  if weight_sum == 0.0
    return Inf
  end
  state_loss /= weight_sum
  x2dot_loss /= weight_sum

  continuity_loss = 0.0
  for i in 2:length(ranges)
    u0 = ode_data[:, last(ranges[i-1])]
    u1 = ode_data[:, first(ranges[i])]
    continuity_loss += continuity_term * sum(abs2, u0 - u1)
  end

  total = state_loss + x2dot_loss + continuity_loss
  return total, (state=state_loss, x2dot=x2dot_loss, continuity=continuity_loss)
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
  state_err = sum(abs2.((ode_data .- x) ./ state_scale); dims=1)
  weights_sum = sum(weights_val)
  if weights_sum == 0.0
    return Inf
  end
  state_loss = 1 / weights_sum * sum(weights_val .* vec(state_err))
  x2dot_loss = 1 / weights_sum * sum(weights_val .* abs2.((x2dot_data .- x2dot_pred) ./ x2dot_scale))
  total = state_loss + x2dot_loss
  return total, (state=state_loss, x2dot=x2dot_loss, continuity=0.0)
end

function sanity_check_integrated_ms()
  true_theta = initial_theta_from_guess(ks, cs, Estar)
  θ_true = ComponentVector(p=true_theta)
  total = loss_multiple_shooting(θ_true; debug=true)
  comps = last_loss_components[]
  return total, comps
end

update_ranges!(solution_dataframe.contact .== 1)

sanity_data, sanity_data_comps = sanity_check_data()
sanity_integrated, sanity_integrated_comps = sanity_check_integrated()
sanity_integrated_ms, sanity_integrated_ms_comps = sanity_check_integrated_ms()
println("Sanity check loss (data-based, post-contact): ", sanity_data, " | state=", sanity_data_comps.state, " x2dot=", sanity_data_comps.x2dot, " cont=", sanity_data_comps.continuity)
println("Sanity check loss (integrated, post-contact): ", sanity_integrated, " | state=", sanity_integrated_comps.state, " x2dot=", sanity_integrated_comps.x2dot, " cont=", sanity_integrated_comps.continuity)
println("Sanity check loss (integrated, multiple shooting): ", sanity_integrated_ms, " | state=", sanity_integrated_ms_comps.state, " x2dot=", sanity_integrated_ms_comps.x2dot, " cont=", sanity_integrated_ms_comps.continuity)
sanity_data_gate = sanity_data_comps.state + sanity_data_comps.x2dot
sanity_integrated_gate = sanity_integrated
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

tmp_steps = solution_dataframe.t
datasize = size(ode_data, 2)
tspan = (tmp_steps[1], tmp_steps[end])
update_ranges!(solution_dataframe.contact .== 1)

results = []

learning_rate_adam = 1e-3 
max_adam_iters = 500
max_lbfgs_iters = 1000

rng = Random.default_rng()
num_initial_guesses = 3
initial_guess_log_span = 0.5

stage_ref = Ref("adam")
adam_loss_target = 1e-4
lbfgs_loss_target = 1e-4

for run_id in 1:num_initial_guesses
  ks0 = sample_log_uniform_narrow(rng, ks_bounds[1], ks_bounds[2], initial_guess_log_span)
  cs0 = sample_log_uniform_narrow(rng, cs_bounds[1], cs_bounds[2], initial_guess_log_span)
  Estar0 = sample_log_uniform_narrow(rng, Estar_bounds[1], Estar_bounds[2], initial_guess_log_span)
  theta0 = initial_theta_from_guess(ks0, cs0, Estar0)

  println("Run ", run_id, " initial guess: ks0=", @sprintf("%.3e", ks0),
    " cs0=", @sprintf("%.3e", cs0), " Estar0=", @sprintf("%.3e", Estar0))

  p0 = ComponentArray(theta0)
  starting_point_in = ComponentVector(p=p0)

  best_training_parameters = [starting_point_in]
  training_epochs = zeros(Int, max_adam_iters)
  training_costs = fill(Inf, max_adam_iters)

  function callback(θ, l)
    epoch = extrema(training_epochs)[2] + 1
    if epoch <= length(training_epochs)
      training_epochs[epoch] = epoch
      training_costs[epoch] = l
    end
    if epoch == 1 || l < minimum(training_costs[1:(epoch-1)])
      best_training_parameters[1] = deepcopy(θ)
    end
    if epoch % 100 == 0
      comps = last_loss_components[]
      p_est = build_parameter_vector(θ.p)
      ks_est = p_est[10]
      cs_est = p_est[11]
      Estar_est = p_est[9]
      ks_err = percent_error_pct(ks_est, ks)
      cs_err = percent_error_pct(cs_est, cs)
      Estar_err = percent_error_pct(Estar_est, Estar)
      println("Run ", run_id, " Epoch ", epoch, " -- cost: ", l,
        " | state=", comps.state, " x2dot=", comps.x2dot, " cont=", comps.continuity,
        " | ks=", @sprintf("%.3e", ks_est), " (", @sprintf("%.2f", ks_err), "%)",
        " cs=", @sprintf("%.3e", cs_est), " (", @sprintf("%.2f", cs_err), "%)",
        " Estar=", @sprintf("%.3e", Estar_est), " (", @sprintf("%.2f", Estar_err), "%)")
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
  result = (
    parameters_training=p_est,
    initial_state_training=ode_data[:, 1],
    validation_resulting_cost=validation_resulting_cost,
    initial_guess=(ks0=ks0, cs0=cs0, Estar0=Estar0),
    status="success"
  )

  push!(results, result)
end

serialize(folder_name * "/" * result_name_string, results)

if !isempty(results)
  validation_costs = [r.validation_resulting_cost for r in results]
  best_idx = argmin(validation_costs)
  best = results[best_idx]
  p_best = best.parameters_training
  ks_best = p_best[10]
  cs_best = p_best[11]
  Estar_best = p_best[9]
  ks_err = percent_error_pct(ks_best, ks)
  cs_err = percent_error_pct(cs_best, cs)
  Estar_err = percent_error_pct(Estar_best, Estar)

  full_tsteps = original_solution_dataframe.t
  prob_full = remake(
    prob_pred;
    p=p_best,
    tspan=(full_tsteps[1], full_tsteps[end]),
    u0=original_ode_data[:, 1]
  )
  local full_sol
  try
    full_sol = solve(prob_full, integrator; saveat=full_tsteps, reltol=reltol, abstol=abstol,
      sensealg=sensealg, unstable_check=(dt, u, p, t) -> any(x -> !isfinite(x), u) || any(abs.(u) .> 1e7),
      verbose=false)
  catch
    full_sol = nothing
  end

  println("Final best params (run ", best_idx, "): ks=", @sprintf("%.3e", ks_best),
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
