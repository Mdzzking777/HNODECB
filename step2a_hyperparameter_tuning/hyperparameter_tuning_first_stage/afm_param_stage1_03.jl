#=
Stage 1 (global TPE) hyperparameter tuning for AFM DMT-KV.
Scenario 03: x3 unobserved, Hertz force replaced by a neural network.
Single shooting or multiple shooting (toggle), no x3 observations (only range prior).
TPE explores NN architecture, mechanistic initial values, MS hyperparams, learning rate, and train/val split.
=#

cd(@__DIR__)

using Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Statistics, Printf
using ComponentArrays, SciMLSensitivity, StableRNGs
using Optimization, OptimizationOptimisers
using DiffEqFlux, .Flux

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
result_name_string = "afm_param_stage1_03.jld"

error_level = "e0.0"

ks_true = ks
cs_true = cs
Estar_true = Estar

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

# Solver settings
integrator = Rosenbrock23(autodiff=false)
abstol = 1e-8
reltol = 1e-8
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

# Parameter bounds (mechanistic unknowns)
ks_bounds = (0.005, 5.0)   # true ks = 0.1
cs_bounds = (5e-8, 5e-6)   # true cs = 2.4e-7
Estar_bounds = (5e5, 5e8)  # true Estar = 1.5e7

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

function percent_error_pct(est, truth)
  return 100.0 * abs(log10(est / truth))
end

function relative_rmse_pct(err_sum, truth_sum, count, eps)
  denom = sqrt(truth_sum / count) + eps
  return 100.0 * sqrt(err_sum / count) / denom
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
      Estar = p.mech[1]
      ks = p.mech[2]
      cs = p.mech[3]

      s = dist + u[1] - u[3]
      delta = softplus(-s, adhesion_transition)
      delta = ifelse(delta > 0.0, delta, 0.0)
      w = contact_weight(s, adhesion_transition)

      nn_in = [u[1], u[2], u[3], delta]
      uhat = appr(nn_in, p.p_net, st)[1]
      F_hertz = Estar * softplus(uhat[1], adhesion_transition) * w
      Fad_eff = Fad * w

      @inbounds du[1] = u[2]
      @inbounds du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
      @inbounds du[3] = (Fad_eff - F_hertz - ks * u[3]) / cs
    end
  return f
end

function x2dot_rhs(u, mech, p_net, appr, st, known_pars, t)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  Estar, ks, cs = mech
  s = dist + u[1] - u[3]
  delta = softplus(-s, adhesion_transition)
  delta = ifelse(delta > 0.0, delta, 0.0)
  w = contact_weight(s, adhesion_transition)
  nn_in = [u[1], u[2], u[3], delta]
  uhat = appr(nn_in, p_net, st)[1]
  F_hertz = Estar * softplus(uhat[1], adhesion_transition) * w
  Fad_eff = Fad * w
  return (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
end

function loss_single_or_ms(θ, ode_data, x2dot_data, contact_mask, times,
  state12_scale, x2dot_scale, x3_scale,
  use_multiple_shooting, ms_group_size, ms_continuity_term,
  appr, st, known_pars, x3_init_val,
  l2_weight, re_pnet)

  # map raw -> bounded mech params
  mech = [
    bound_param(θ.mech_raw[1], Estar_bounds[1], Estar_bounds[2]),
    bound_param(θ.mech_raw[2], ks_bounds[1], ks_bounds[2]),
    bound_param(θ.mech_raw[3], cs_bounds[1], cs_bounds[2])
  ]
  p_net_struct = re_pnet(θ.p_net)
  p = ComponentVector(p_net=p_net_struct, mech=mech)

  weights_val = ifelse.(contact_mask, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_val)
  if !isfinite(weights_sum) || weights_sum <= 0
    return Inf, (state=Inf, x2dot=Inf, x3_range=Inf)
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
        return Inf, (state=Inf, x2dot=Inf, x3_range=Inf)
      end
      preds[i] = Array(sol)

      uhat = preds[i]
      seg_weights = weights_val[rg]
      seg_weights_sum = sum(seg_weights)
      if !isfinite(seg_weights_sum) || seg_weights_sum <= 0
        return Inf, (state=Inf, x2dot=Inf, x3_range=Inf)
      end

      state_err = vec(sum(abs2.((ode_data[1:2, rg] .- uhat[1:2, :]) ./ state12_scale), dims=1))
      total_state += sum(seg_weights .* state_err) / seg_weights_sum

      contact_idx = findall(contact_mask[rg])
      if !isempty(contact_idx)
        local_idx = rg[contact_idx]
        x2dot_pred = [x2dot_rhs(uhat[:, j], mech, p_net_struct, appr, st, known_pars, times[rg[j]]) for j in 1:length(rg)]
        if any(x -> !isfinite(x), x2dot_pred)
          return Inf, (state=Inf, x2dot=Inf, x3_range=Inf)
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
      return Inf, (state=Inf, x2dot=Inf, x3_range=Inf)
    end
    uhat = Array(sol)

    state_err = vec(sum(abs2.((ode_data[1:2, :] .- uhat[1:2, :]) ./ state12_scale), dims=1))
    total_state = sum(weights_val .* state_err) / weights_sum

    contact_idx = findall(contact_mask)
    if !isempty(contact_idx)
      x2dot_pred = [x2dot_rhs(uhat[:, j], mech, p_net_struct, appr, st, known_pars, times[j]) for j in 1:length(times)]
      if any(x -> !isfinite(x), x2dot_pred)
        return Inf, (state=Inf, x2dot=Inf, x3_range=Inf)
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
  return total, (state=total_state, x2dot=total_x2dot, x3_range=total_x3range, cont=total_cont)
end

use_multiple_shooting = false
use_l2_regularization = false
l2_weight = 0.0

println("=== AFM Stage1 (03) ===")
println("Switches: MS=", use_multiple_shooting, " L2=", use_l2_regularization, " (λ=", l2_weight, ")")

last_trial_metrics = Ref((train_loss=Inf, val_loss=Inf))

function objective(trial)
  try
    # hyperparameters
    ks0 = trial.suggest_float("ks0", ks_bounds[1], ks_bounds[2], log=true)
    cs0 = trial.suggest_float("cs0", cs_bounds[1], cs_bounds[2], log=true)
    Estar0 = trial.suggest_float("Estar0", Estar_bounds[1], Estar_bounds[2], log=true)
    learning_rate_adam = trial.suggest_float("learning_rate_adam", 1e-5, 1e-2, log=true)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)
    num_hidden_nodes = trial.suggest_int("num_hidden_nodes", 3, 5)
    ms_group_size = trial.suggest_int("ms_group_size", 10, 200)
    ms_continuity_term = trial.suggest_float("ms_continuity_term", 1e-6, 10.0, log=true)
    # Fixed train/validation split (not part of hyperparameter search)
    val_stride = 5
    val_offset = 2

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

    state12_scale_train = state12_scale_full
    x2dot_scale_train = x2dot_scale_full

    # NN architecture
    println("  building NN + init params")
    flush(stdout)
    seed = abs(rand(rng_global, Int))
    rng = StableRNG(seed)
    approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)
    p_net, st = Lux.setup(rng, approximating_neural_network)
    p_net_vec, re_pnet = Lux.destructure(p_net)

    # initial parameters (bounded raw)
    raw_init = [
      raw_from_value(Estar0, Estar_bounds[1], Estar_bounds[2]),
      raw_from_value(ks0, ks_bounds[1], ks_bounds[2]),
      raw_from_value(cs0, cs_bounds[1], cs_bounds[2])
    ]
    θ0 = ComponentVector(p_net=p_net_vec, mech_raw=raw_init)

    # x3 initial value (allowed: first contact)
    x3_init_val = ode_train[3, 1]

    # loss function
    function loss_fn(θ)
      loss_single_or_ms(θ, ode_train, x2dot_train, contact_train, times_train,
        state12_scale_train, x2dot_scale_train, x3_scale,
        use_multiple_shooting, ms_group_size, ms_continuity_term,
        approximating_neural_network, st, known_pars, x3_init_val,
        l2_weight, re_pnet)
    end

    # ADAM optimization
    println("  starting ADAM")
    flush(stdout)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_fn(x)[1], adtype)
    optprob = Optimization.OptimizationProblem(optf, θ0)
    opt = OptimizationOptimisers.Adam(learning_rate_adam)

    maxiters = 300
    training_costs = fill(Inf, maxiters)
    epoch_ref = Ref(1)
    start_time = time()
    stuck = Ref(false)

    function callback(θ, l)
      epoch = epoch_ref[]
      training_costs[epoch] = l
      # Print every epoch to monitor runtime
      println("Stage1 trial epoch ", epoch, " loss=", @sprintf("%.4e", l))
      if time() - start_time > 4 * 60
        stuck[] = true
        return true
      end
      if epoch > 10 && minimum(training_costs[max(1, epoch-5):epoch]) > 1e6
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
    train_loss, _ = loss_single_or_ms(θ_best, ode_train, x2dot_train, contact_train, times_train,
      state12_scale_full, x2dot_scale_full, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_init_val,
      l2_weight, re_pnet)

    # validation loss (no L2)
    val_loss, _ = loss_single_or_ms(θ_best, ode_val, x2dot_val, contact_val, times_val,
      state12_scale_full, x2dot_scale_full, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_init_val,
      0.0, re_pnet)

    last_trial_metrics[] = (train_loss=train_loss, val_loss=val_loss)
    return val_loss
  catch ex
    last_trial_metrics[] = (train_loss=Inf, val_loss=Inf)
    @error "Stage1 trial failed" exception=(ex, catch_backtrace())
    println("trial params = ", Dict(trial.params))
    return Inf
  end
end

# TPE optimization
num_trials = 200
study = optuna.create_study(sampler=optuna.samplers.TPESampler(consider_prior=false, multivariate=true, seed=0))
trial_parameters = []

for optuna_iteration in 1:num_trials
  println("Stage1 trial ", optuna_iteration, " start")
  flush(stdout)
  trial = study.ask()
  cost = objective(trial)
  study.tell(trial, cost)
  # store a plain Julia copy of params
  params = Dict(trial.params)
  metrics = last_trial_metrics[]
  push!(trial_parameters, (loss=cost, train_loss=metrics.train_loss, val_loss=metrics.val_loss, params=params))
  println("Stage1 trial ", optuna_iteration,
    " -- train=", @sprintf("%.4e", metrics.train_loss),
    " val=", @sprintf("%.4e", metrics.val_loss))
end

# summarize
sorted = sort(trial_parameters, by = r -> r.loss)
best = sorted[1]
println("Stage1 done. Best val loss=", @sprintf("%.4e", best.val_loss),
  " | train=", @sprintf("%.4e", best.train_loss))

println("Stage1 top-10 summary (train/val):")
for (i, rec) in enumerate(sorted[1:min(10, length(sorted))])
  ks0 = rec.params["ks0"]
  cs0 = rec.params["cs0"]
  Estar0 = rec.params["Estar0"]
  println("  Rank ", i,
    " -- train=", @sprintf("%.4e", rec.train_loss),
    " val=", @sprintf("%.4e", rec.val_loss),
    " | ks0=", @sprintf("%.3e", ks0),
    " cs0=", @sprintf("%.3e", cs0),
    " Estar0=", @sprintf("%.3e", Estar0))
end

serialize(result_folder * "/" * result_name_string, (
  study=study,
  trial_parameters=trial_parameters,
  best=best,
  bounds=(ks=ks_bounds, cs=cs_bounds, Estar=Estar_bounds),
  use_multiple_shooting=use_multiple_shooting,
  use_l2_regularization=use_l2_regularization,
  val_stride=val_stride,
  val_offset=val_offset,
  x3_obs_fraction=0.0,
  error_level=error_level
))
