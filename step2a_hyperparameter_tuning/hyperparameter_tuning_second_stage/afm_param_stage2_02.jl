#=
Stage 2 (local refinement) for AFM DMT-KV mechanistic parameters.
Single shooting over the full post-contact trajectory.
Simplified loss: state (x1,x2) + contact x2dot + x3 range penalty (±20nm).
No x3 observations (unobserved x3). Does not use oracle x3.
=#

cd(@__DIR__)

using Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Statistics, Printf

include("../../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_functions.jl")
include("../../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")

result_folder = "results_afm"
if !isdir(result_folder)
  mkdir(result_folder)
end
result_name_string = "afm_param_stage2_02.jld"

error_level = "e0.0"

ks_true = ks
cs_true = cs
Estar_true = Estar

# Load stage1 results
stage1 = deserialize("../hyperparameter_tuning_first_stage/results_afm/afm_param_stage1_02.jld")
stage1_results = stage1.results

# Load data
ode_data = deserialize("../../datasets/e0.0/data/ode_data_afm_dmt_kv.jld")
solution_dataframe = deserialize("../../datasets/e0.0/data/pert_df_afm_dmt_kv.jld")

# Keep only data after first contact
contact_idx = findfirst(solution_dataframe.contact .== 1)
if contact_idx === nothing
  error("No contact point found in the AFM dataset.")
end
solution_dataframe = solution_dataframe[contact_idx:end, :]
ode_data = ode_data[:, contact_idx:end]

tmp_steps = solution_dataframe.t
x2dot_data = solution_dataframe.x2dot
contact_mask = solution_dataframe.contact .== 1

# Multiple shooting segmentation (same scheme as training)
small_group_size = 100
large_group_size = 800
boundary_window = 100
min_group_size = 10

# Contact weighting
contact_loss_weight = 4.27
noncontact_loss_weight = 1.0

# x3 range prior (no x3 observations)
x3_range_center = 0.0
x3_range_amp = 20e-9
x3_range_eps = 1e-9
x3_range_weight = 1.0

integrator = Rosenbrock23(autodiff=false)
abstol = 1e-8
reltol = 1e-8

datasize = size(ode_data, 2)
tspan = (tmp_steps[1], tmp_steps[end])

# Normalization scales
# Use only observable signals for scaling. For x3, use the physical prior amplitude.
scale_eps = 1e-9
state12_scale = vec(maximum(ode_data[1:2, :], dims=2) - minimum(ode_data[1:2, :], dims=2))
state12_scale = max.(state12_scale, scale_eps)
x2dot_scale = max(maximum(x2dot_data) - minimum(x2dot_data), scale_eps)
x3_scale = max(x3_range_amp, scale_eps)

# Parameter bounds
ks_bounds = stage1.bounds.ks
cs_bounds = stage1.bounds.cs
Estar_bounds = stage1.bounds.Estar

function retcode_success(sol)
  return string(sol.retcode) == "Success"
end

function percent_error_pct(est, truth)
  return 100.0 * abs(log10(est / truth))
end

function relative_rmse_pct(err_sum, truth_sum, count, eps)
  denom = sqrt(truth_sum / count) + eps
  return 100.0 * sqrt(err_sum / count) / denom
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

function make_parameter_vector(ks, cs, Estar)
  return [original_parameters[1:8]...; Estar; ks; cs]
end

function evaluate_loss(p_est, ranges, range_is_contact)
  # Single shooting: ranges are ignored; kept only for interface compatibility.
  if any(x -> !isfinite(x), p_est)
    return Inf, (state=Inf, x2dot=Inf, x3_obs=Inf, x3_range=Inf, x1_rec=Inf, x3_rec=Inf)
  end
  ks_est = p_est[10]
  cs_est = p_est[11]
  Estar_est = p_est[9]
  if ks_est < ks_bounds[1] || ks_est > ks_bounds[2] ||
     cs_est < cs_bounds[1] || cs_est > cs_bounds[2] ||
     Estar_est < Estar_bounds[1] || Estar_est > Estar_bounds[2]
    return Inf, (state=Inf, x2dot=Inf, x3_obs=Inf, x3_range=Inf, x1_rec=Inf, x3_rec=Inf)
  end

  u0 = [ode_data[1, 1], ode_data[2, 1], ode_data[3, 1]]
  prob = ODEProblem{true}(ground_truth_function, u0, tspan, p_est)
  sol = solve(prob, integrator; saveat=tmp_steps, abstol=abstol, reltol=reltol)
  if !retcode_success(sol)
    return Inf, (state=Inf, x2dot=Inf, x3_obs=Inf, x3_range=Inf, x1_rec=Inf, x3_rec=Inf)
  end
  uhat = Array(sol)
  if size(uhat, 2) != length(tmp_steps)
    return Inf, (state=Inf, x2dot=Inf, x3_obs=Inf, x3_range=Inf, x1_rec=Inf, x3_rec=Inf)
  end

  weights_val = ifelse.(contact_mask, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_val)
  if !isfinite(weights_sum) || weights_sum <= 0
    return Inf, (state=Inf, x2dot=Inf, x3_obs=Inf, x3_range=Inf, x1_rec=Inf, x3_rec=Inf)
  end

  state_err_per_t = vec(sum(abs2.((ode_data[1:2, :] .- uhat[1:2, :]) ./ state12_scale), dims=1))
  state_loss = sum(weights_val .* state_err_per_t) / weights_sum

  contact_idx = findall(contact_mask)
  if isempty(contact_idx)
    x2dot_loss = 0.0
  else
    x2dot_pred = [x2dot_rhs(uhat[:, j], p_est, tmp_steps[j]) for j in eachindex(tmp_steps)]
    if any(x -> !isfinite(x), x2dot_pred)
      return Inf, (state=Inf, x2dot=Inf, x3_obs=Inf, x3_range=Inf, x1_rec=Inf, x3_rec=Inf)
    end
    x2dot_err = abs2.((x2dot_data[contact_idx] .- x2dot_pred[contact_idx]) ./ x2dot_scale)
    x2dot_w = weights_val[contact_idx]
    x2dot_loss = sum(x2dot_w .* x2dot_err) / sum(x2dot_w)
  end

  exceed = abs.(uhat[3, :]) .- x3_range_amp
  range_pen_per_t = abs2.(softplus.(exceed, x3_range_eps) ./ x3_scale)
  x3_range_loss = x3_range_weight * (sum(weights_val .* range_pen_per_t) / weights_sum)

  x1_err_sum = sum(abs2.(ode_data[1, :] .- uhat[1, :]))
  x1_truth_sum = sum(abs2.(ode_data[1, :]))
  x3_err_sum = sum(abs2.(ode_data[3, :] .- uhat[3, :]))
  x3_truth_sum = sum(abs2.(ode_data[3, :]))
  count_sum = length(tmp_steps)
  x1_rec = relative_rmse_pct(x1_err_sum, x1_truth_sum, count_sum, scale_eps)
  x3_rec = relative_rmse_pct(x3_err_sum, x3_truth_sum, count_sum, scale_eps)

  total = state_loss + x2dot_loss + x3_range_loss
  return total, (state=state_loss, x2dot=x2dot_loss, x3_obs=0.0, x3_range=x3_range_loss,
    x1_rec=x1_rec, x3_rec=x3_rec)
end

# Build ranges once
ranges, range_is_contact = build_contact_ranges(contact_mask, small_group_size, large_group_size, boundary_window, min_group_size)
rng = MersenneTwister(0)

# Stage 2: local grid around best K from stage1
top_k = 10
sorted_stage1 = sort(stage1_results, by = r -> r.loss)
seed_candidates = sorted_stage1[1:min(top_k, length(sorted_stage1))]

log_span = 0.5 # half-decade around each seed
grid_offsets = [-log_span, 0.0, log_span]

stage2_results = NamedTuple[]
best_ref = Ref((loss=Inf, ks=NaN, cs=NaN, Estar=NaN, state=Inf, x2dot=Inf, x3_obs=Inf, x3_range=Inf,
  x1_rec=Inf, x3_rec=Inf))

eval_id = Ref(0)
for (seed_idx, seed) in enumerate(seed_candidates)
  println("Stage2 seed ", seed_idx, "/", length(seed_candidates),
    " ks=", @sprintf("%.3e", seed.ks), " cs=", @sprintf("%.3e", seed.cs),
    " Estar=", @sprintf("%.3e", seed.Estar))

  for dks in grid_offsets, dcs in grid_offsets, dE in grid_offsets
    local ks = 10.0^(log10(seed.ks) + dks)
    local cs = 10.0^(log10(seed.cs) + dcs)
    local Estar = 10.0^(log10(seed.Estar) + dE)
    ks = clamp(ks, ks_bounds[1], ks_bounds[2])
    cs = clamp(cs, cs_bounds[1], cs_bounds[2])
    Estar = clamp(Estar, Estar_bounds[1], Estar_bounds[2])
    p_est = make_parameter_vector(ks, cs, Estar)
    loss, comps = evaluate_loss(p_est, ranges, range_is_contact)
    eval_id[] += 1
    ks_err = percent_error_pct(ks, ks_true)
    cs_err = percent_error_pct(cs, cs_true)
    Estar_err = percent_error_pct(Estar, Estar_true)
    println("Stage2 eval ", eval_id[], " -- loss: ", @sprintf("%.4e", loss),
      " | x1_rec=", @sprintf("%.2f", comps.x1_rec), "% x3_rec=", @sprintf("%.2f", comps.x3_rec), "%",
      " | state=", @sprintf("%.4e", comps.state),
      " x2dot=", @sprintf("%.4e", comps.x2dot),
      " x3_obs=None",
      " x3_range=", @sprintf("%.4e", comps.x3_range),
      " x3_u0=None cont=None",
      " | ks=", @sprintf("%.3e", ks), " (true=", @sprintf("%.3e", ks_true), ", ", @sprintf("%.2f", ks_err), "%)",
      " cs=", @sprintf("%.3e", cs), " (true=", @sprintf("%.3e", cs_true), ", ", @sprintf("%.2f", cs_err), "%)",
      " Estar=", @sprintf("%.3e", Estar), " (true=", @sprintf("%.3e", Estar_true), ", ", @sprintf("%.2f", Estar_err), "%)")
    push!(stage2_results, (loss=loss, state=comps.state, x2dot=comps.x2dot, x3_obs=comps.x3_obs, x3_range=comps.x3_range,
      x1_rec=comps.x1_rec, x3_rec=comps.x3_rec, ks=ks, cs=cs, Estar=Estar, seed_idx=seed_idx))
    if loss < best_ref[].loss
      best_ref[] = (loss=loss, ks=ks, cs=cs, Estar=Estar,
        state=comps.state, x2dot=comps.x2dot, x3_obs=comps.x3_obs, x3_range=comps.x3_range,
        x1_rec=comps.x1_rec, x3_rec=comps.x3_rec)
    end
  end
end

serialize(result_folder * "/" * result_name_string, (
  results=stage2_results,
  best=best_ref[],
  bounds=(ks=ks_bounds, cs=cs_bounds, Estar=Estar_bounds),
  x3_obs_fraction=0.0,
  error_level=error_level
))

println("Stage2 done. Best loss=", @sprintf("%.4e", best_ref[].loss),
  " ks=", @sprintf("%.3e", best_ref[].ks),
  " cs=", @sprintf("%.3e", best_ref[].cs),
  " Estar=", @sprintf("%.3e", best_ref[].Estar))

# Oracle sanity check under the same loss definition
p_oracle = make_parameter_vector(ks_true, cs_true, Estar_true)
oracle_loss, oracle_comps = evaluate_loss(p_oracle, ranges, range_is_contact)
println("Stage2 oracle sanity (02) -- loss: ", @sprintf("%.4e", oracle_loss),
  " | state=", @sprintf("%.4e", oracle_comps.state),
  " x2dot=", @sprintf("%.4e", oracle_comps.x2dot),
  " x3_range=", @sprintf("%.4e", oracle_comps.x3_range))
if isfinite(oracle_loss) && oracle_loss > 0
  println("Stage2 best/oracle ratio: ", @sprintf("%.3e", best_ref[].loss / oracle_loss))
else
  println("Stage2 best/oracle ratio: Inf")
end
sorted2 = sort(stage2_results, by = r -> r.loss)
println("Stage2 top-10 summary:")
topn = min(10, length(sorted2))
for i in 1:topn
  r = sorted2[i]
  ks_err = percent_error_pct(r.ks, ks_true)
  cs_err = percent_error_pct(r.cs, cs_true)
  Estar_err = percent_error_pct(r.Estar, Estar_true)
  println("  Rank ", i, " -- loss: ", @sprintf("%.4e", r.loss),
    " | x1_rec=", @sprintf("%.2f", r.x1_rec), "% x3_rec=", @sprintf("%.2f", r.x3_rec), "%",
    " | state=", @sprintf("%.4e", r.state),
    " x2dot=", @sprintf("%.4e", r.x2dot),
    " x3_obs=None",
    " x3_range=", @sprintf("%.4e", r.x3_range),
    " x3_u0=None cont=None",
    " | ks=", @sprintf("%.3e", r.ks), " (true=", @sprintf("%.3e", ks_true), ", ", @sprintf("%.2f", ks_err), "%)",
    " cs=", @sprintf("%.3e", r.cs), " (true=", @sprintf("%.3e", cs_true), ", ", @sprintf("%.2f", cs_err), "%)",
    " Estar=", @sprintf("%.3e", r.Estar), " (true=", @sprintf("%.3e", Estar_true), ", ", @sprintf("%.2f", Estar_err), "%)")
end

