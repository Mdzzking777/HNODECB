#=
Stage 2 (grid) hyperparameter tuning for AFM DMT-KV.
Scenario 03: x3 unobserved, Hertz force replaced by a neural network.
Grid search on L2 regularization weight using best hyperparameters from Stage 1.
=#

cd(@__DIR__)

using Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Statistics, Printf
using ComponentArrays, SciMLSensitivity, StableRNGs
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

include("../../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_functions.jl")
include("../../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")

result_folder = "results_afm"
if !isdir(result_folder)
  mkdir(result_folder)
end
result_name_string = "afm_param_stage2_03.jld"

error_level = "e0.0"

ks_true = ks
cs_true = cs

# Load stage1 results
stage1 = deserialize("../hyperparameter_tuning_first_stage/results_afm/afm_param_stage1_03.jld")
best = stage1.best
best_params = best.params

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

# Physical prior for x3 (no observations)
x3_range_center = 0.0
x3_range_amp = 20e-9
x3_range_eps = 1e-9
x3_range_weight = 1.0

# Contact weighting
contact_loss_weight = 4.27
noncontact_loss_weight = 1.0

# Multiple shooting toggle (keep on to match baseline)
use_multiple_shooting = false
selftest = get(ENV, "HNODECB_SELFTEST", "0") == "1"

if selftest
  println("=== AFM Stage2 (03) SELFTEST ===")
else
  println("=== AFM Stage2 (03) ===")
end
println("Switches: MS=", use_multiple_shooting, " | L2-grid pending...")

# Solver settings
integrator = Rosenbrock23(autodiff=false)
abstol = 1e-8
reltol = 1e-8
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

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

function make_train_val_masks(n, val_stride, val_offset)
  val_idx = [i for i in 1:n if (i - val_offset) % val_stride == 0]
  train_idx = [i for i in 1:n if !(i in val_idx)]
  return sort(train_idx), sort(val_idx)
end

function build_nn(num_hidden_layers::Int, num_hidden_nodes::Int)
  hidden = 2^num_hidden_nodes
  layers = Any[]
  push!(layers, Lux.Dense(4, hidden, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform))
  for _ in 1:num_hidden_layers
    push!(layers, Lux.Dense(hidden, hidden, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform))
  end
  push!(layers, Lux.Dense(hidden, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform))
  return Lux.Chain(layers...)
end

function make_uode_func(appr, st, known_pars)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  f(du, u, p, t) =
    let appr = appr, st = st, k = k, wd = wd, m = m, c = c, Fd = Fd, R = R, dist = dist, Fad = Fad
      ks = p.mech[1]
      cs = p.mech[2]

      s = dist + u[1] - u[3]
      delta = softplus(-s, adhesion_transition)
      delta = ifelse(delta > 0.0, delta, 0.0)
      w = contact_weight(s, adhesion_transition)

      nn_in = [u[1], u[2], u[3], delta]
      uhat = appr(nn_in, p.p_net, st)[1]
      F_hertz = softplus(uhat[1], adhesion_transition) * w
      Fad_eff = Fad * w

      @inbounds du[1] = u[2]
      @inbounds du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
      @inbounds du[3] = (Fad_eff - F_hertz - ks * u[3]) / cs
    end
  return f
end

function x2dot_rhs(u, mech, p_net, appr, st, known_pars, t)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  ks, cs = mech
  s = dist + u[1] - u[3]
  delta = softplus(-s, adhesion_transition)
  delta = ifelse(delta > 0.0, delta, 0.0)
  w = contact_weight(s, adhesion_transition)
  nn_in = [u[1], u[2], u[3], delta]
  uhat = appr(nn_in, p_net, st)[1]
  F_hertz = softplus(uhat[1], adhesion_transition) * w
  Fad_eff = Fad * w
  return (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
end

function loss_single_or_ms(θ, ode_data, x2dot_data, contact_mask, times,
  state12_scale, x2dot_scale, x3_scale,
  use_multiple_shooting, ms_group_size, ms_continuity_term,
  appr, st, known_pars, x3_init_val,
  l2_weight, re_pnet)

  mech = [
    bound_param(θ.mech_raw[1], ks_bounds[1], ks_bounds[2]),
    bound_param(θ.mech_raw[2], cs_bounds[1], cs_bounds[2])
  ]
  p_net_struct = re_pnet(θ.p_net)
  p = ComponentVector(p_net=p_net_struct, mech=mech)

  weights_val = ifelse.(contact_mask, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_val)
  if !isfinite(weights_sum) || weights_sum <= 0
    return Inf
  end

  total_state = 0.0
  total_x2dot = 0.0
  total_x3range = 0.0
  total_cont = 0.0

  if use_multiple_shooting
    ranges = DiffEqFlux.group_ranges(length(times), ms_group_size)
    preds = Vector{Matrix{Float64}}(undef, length(ranges))
    for (i, rg) in enumerate(ranges)
      u0 = [ode_data[1, first(rg)], ode_data[2, first(rg)], x3_init_val]
      prob = ODEProblem{true}(make_uode_func(appr, st, known_pars), u0, (times[first(rg)], times[last(rg)]), p)
      sol = solve(prob, integrator; saveat=times[rg], abstol=abstol, reltol=reltol, sensealg=sensealg)
      if size(sol, 2) != length(rg)
        return Inf
      end
      preds[i] = Array(sol)

      uhat = preds[i]
      seg_weights = weights_val[rg]
      seg_weights_sum = sum(seg_weights)
      if !isfinite(seg_weights_sum) || seg_weights_sum <= 0
        return Inf
      end

      state_err = vec(sum(abs2.((ode_data[1:2, rg] .- uhat[1:2, :]) ./ state12_scale), dims=1))
      total_state += sum(seg_weights .* state_err) / seg_weights_sum

      contact_idx = findall(contact_mask[rg])
      if !isempty(contact_idx)
        local_idx = rg[contact_idx]
        uhat_const = Zygote.dropgrad(uhat)
        x2dot_pred = [x2dot_rhs(view(uhat_const, :, j), mech, p_net_struct, appr, st, known_pars, times[rg[j]]) for j in 1:length(rg)]
        if any(x -> !isfinite(x), x2dot_pred)
          return Inf
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
  else
    u0 = [ode_data[1, 1], ode_data[2, 1], x3_init_val]
    prob = ODEProblem{true}(make_uode_func(appr, st, known_pars), u0, (times[1], times[end]), p)
    sol = solve(prob, integrator; saveat=times, abstol=abstol, reltol=reltol, sensealg=sensealg)
    if size(sol, 2) != length(times)
      return Inf
    end
    uhat = Array(sol)

    state_err = vec(sum(abs2.((ode_data[1:2, :] .- uhat[1:2, :]) ./ state12_scale), dims=1))
    total_state = sum(weights_val .* state_err) / weights_sum

    contact_idx = findall(contact_mask)
    if !isempty(contact_idx)
      uhat_const = Zygote.dropgrad(uhat)
      x2dot_pred = [x2dot_rhs(view(uhat_const, :, j), mech, p_net_struct, appr, st, known_pars, times[j]) for j in 1:length(times)]
      if any(x -> !isfinite(x), x2dot_pred)
        return Inf
      end
      x2_err = abs2.((x2dot_data[contact_idx] .- x2dot_pred[contact_idx]) ./ x2dot_scale)
      x2_w = weights_val[contact_idx]
      total_x2dot = sum(x2_w .* x2_err) / sum(x2_w)
    end

    exceed = abs.(uhat[3, :]) .- x3_range_amp
    range_pen = abs2.(softplus.(exceed, x3_range_eps) ./ x3_scale)
    total_x3range = x3_range_weight * (sum(weights_val .* range_pen) / weights_sum)
  end

  l2_penalty = l2_weight * sum(abs2, θ.p_net)
  total = total_state + total_x2dot + total_x3range + total_cont + l2_penalty
  return total
end

# Stage1 best hyperparameters
ks0 = best_params["ks0"]
cs0 = best_params["cs0"]
learning_rate_adam = best_params["learning_rate_adam"]
num_hidden_layers = best_params["num_hidden_layers"]
num_hidden_nodes = best_params["num_hidden_nodes"]
ms_group_size = best_params["ms_group_size"]
ms_continuity_term = best_params["ms_continuity_term"]
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

# NN architecture
rng = StableRNG(0)
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
x3_init_val = ode_train[3, 1]

if selftest
  n = min(50, length(times_train))
  idx = 1:n
  ode_train_s = ode_train[:, idx]
  times_train_s = times_train[idx]
  x2dot_train_s = x2dot_train[idx]
  contact_train_s = contact_train[idx]
  l2_weight = 0.0

  loss = loss_single_or_ms(θ0, ode_train_s, x2dot_train_s, contact_train_s, times_train_s,
    state12_scale_full, x2dot_scale_full, x3_scale,
    use_multiple_shooting, ms_group_size, ms_continuity_term,
    approximating_neural_network, st, known_pars, x3_init_val,
    l2_weight, re_pnet)

  println("  selftest loss=", @sprintf("%.4e", loss))
  if !isfinite(loss)
    error("SELFTEST failed: non-finite loss")
  end
  println("=== SELFTEST OK ===")
else
function internal_objective(l2_weight)
  function loss_fn(θ)
    loss_single_or_ms(θ, ode_train, x2dot_train, contact_train, times_train,
    state12_scale_full, x2dot_scale_full, x3_scale,
    use_multiple_shooting, ms_group_size, ms_continuity_term,
    approximating_neural_network, st, known_pars, x3_init_val,
    l2_weight, re_pnet)
  end

  adtype = Optimization.AutoZygote()
  optf = Optimization.OptimizationFunction((x, p) -> loss_fn(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, θ0)
  opt = OptimizationOptimisers.Adam(learning_rate_adam)

  maxiters = 500
  epoch_ref = Ref(1)
  start_time = time()
  stuck = Ref(false)

  function callback(θ, l)
    epoch = epoch_ref[]
    if epoch % 100 == 0
      println("Stage2 epoch ", epoch, " loss=", @sprintf("%.4e", l))
    end
    if time() - start_time > 5 * 60
      stuck[] = true
      return true
    end
    if epoch > 10 && l > 1e6
      stuck[] = true
      return true
    end
    epoch_ref[] = epoch + 1
    return epoch >= maxiters
  end

  res = Optimization.solve(optprob, opt; callback=callback, maxiters=maxiters)
  θ_best = res.u

  if stuck[]
    last_trial_metrics[] = (train_loss=Inf, val_loss=Inf)
    return Inf
  end

  # training loss (with L2)
  train_loss = loss_single_or_ms(θ_best, ode_train, x2dot_train, contact_train, times_train,
    state12_scale_full, x2dot_scale_full, x3_scale,
    use_multiple_shooting, ms_group_size, ms_continuity_term,
    approximating_neural_network, st, known_pars, x3_init_val,
    l2_weight, re_pnet)

  # validation loss (no L2)
  val_loss = loss_single_or_ms(θ_best, ode_val, x2dot_val, contact_val, times_val,
    state12_scale_full, x2dot_scale_full, x3_scale,
    use_multiple_shooting, ms_group_size, ms_continuity_term,
    approximating_neural_network, st, known_pars, x3_init_val,
    0.0, re_pnet)

  last_trial_metrics[] = (train_loss=train_loss, val_loss=val_loss)
  return val_loss
end

# Grid search for L2 regularization
search_space = Dict(
  # L2 regularization disabled for this scenario
  "l2_regularization" => [0.0]
)

println("L2 grid: ", search_space["l2_regularization"])

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space, seed=0))
trial_parameters = []
last_trial_metrics = Ref((train_loss=Inf, val_loss=Inf))

function objective(trial)
  l2_reg = trial.suggest_float("l2_regularization", 0.0, 1.0)
  val_loss = internal_objective(l2_reg)
  metrics = last_trial_metrics[]
  trial.set_user_attr("train_loss", metrics.train_loss)
  trial.set_user_attr("val_loss", metrics.val_loss)
  return val_loss
end

study.optimize(objective)

# collect results
for t in study.trials
  params = Dict(t.params)
  train_loss = haskey(t.user_attrs, "train_loss") ? t.user_attrs["train_loss"] : Inf
  val_loss = haskey(t.user_attrs, "val_loss") ? t.user_attrs["val_loss"] : t.value
  push!(trial_parameters, (loss=t.value, train_loss=train_loss, val_loss=val_loss, params=params))
end

sorted = sort(trial_parameters, by = r -> r.loss)
println("Stage2 top-10 summary (train/val):")
for (i, rec) in enumerate(sorted[1:min(10, length(sorted))])
  println("  Rank ", i,
    " -- train=", @sprintf("%.4e", rec.train_loss),
    " val=", @sprintf("%.4e", rec.val_loss),
    " | l2=", @sprintf("%.2e", rec.params["l2_regularization"]))
end

serialize(result_folder * "/" * result_name_string, (
  study=study,
  trial_parameters=trial_parameters,
  best=study.best_trial === nothing ? nothing : (loss=study.best_trial.value, params=Dict(study.best_trial.params)),
  bounds=(ks=ks_bounds, cs=cs_bounds),
  use_multiple_shooting=use_multiple_shooting,
  l2_grid=search_space["l2_regularization"],
  x3_obs_fraction=0.0,
  error_level=error_level
))

println("Stage2 done. Best loss=", study.best_trial === nothing ? "Inf" : @sprintf("%.4e", study.best_trial.value))
end
