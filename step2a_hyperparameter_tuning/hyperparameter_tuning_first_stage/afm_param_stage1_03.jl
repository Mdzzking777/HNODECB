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
result_name_string = run_tag == "" ? "afm_param_stage1_03.jld" : "afm_param_stage1_03_" * run_tag * ".jld"

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

function grad_norm_safe(g)
  try
    return norm(vec(g))
  catch
    return NaN
  end
end

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
trace_solve_enabled = get(ENV, "HNODECB_TRACE_SOLVE", "0") == "1"
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

  # map raw -> bounded mech params
  mech = [
    bound_param(θ.mech_raw[1], ks_bounds[1], ks_bounds[2]),
    bound_param(θ.mech_raw[2], cs_bounds[1], cs_bounds[2])
  ]
  p_net_struct = re_pnet(θ.p_net)
  p = ComponentVector(p_net=p_net_struct, mech=mech)

  weights_val = ifelse.(contact_mask, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_val)
  if !isfinite(weights_sum) || weights_sum <= 0
    log_inf("weights_sum")
    return Inf, (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
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
      u0 = [ode_data[1, first(rg)], ode_data[2, first(rg)], x3_init_val]
      prob = ODEProblem{true}(make_uode_func(appr, st, known_pars), u0, (times[first(rg)], times[last(rg)]), p)
      trace_solve(() -> "ms solve start seg=" * string(i) * " len=" * string(length(rg)) *
        " t=" * string(round(times[first(rg)], sigdigits=3)) * "→" * string(round(times[last(rg)], sigdigits=3)))
        t_start = Zygote.ignore() do
          time()
        end
      sol = solve(prob, integrator; saveat=times[rg], abstol=abstol, reltol=reltol, sensealg=sensealg)
      trace_solve(() -> "ms solve done seg=" * string(i) * " dt=" * string(round(time() - t_start, digits=2)) * "s" *
        " retcode=" * string(sol.retcode) * " size=" * string(size(sol, 2)))
      if !SciMLBase.successful_retcode(sol)
        log_inf("retcode=" * string(sol.retcode))
        return Inf, (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
      end
      if size(sol, 2) != length(rg)
        log_inf("sol_size_mismatch_ms")
        return Inf, (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
      end
      preds[i] = Array(sol)

      uhat = preds[i]
      if any(x -> !isfinite(x), uhat)
        log_inf("uhat_nonfinite_ms")
        return Inf, (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
      end
      seg_weights = weights_val[rg]
      seg_weights_sum = sum(seg_weights)
      if !isfinite(seg_weights_sum) || seg_weights_sum <= 0
        log_inf("seg_weights_sum")
        return Inf, (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
      end

      state_err = vec(sum(abs2.((ode_data[1:2, rg] .- uhat[1:2, :]) ./ state12_scale), dims=1))
      total_state += sum(seg_weights .* state_err) / seg_weights_sum

      contact_idx = findall(contact_mask[rg])
      if !isempty(contact_idx)
        local_idx = rg[contact_idx]
        uhat_const = Zygote.dropgrad(uhat)
        x2dot_pred = [x2dot_rhs(view(uhat_const, :, j), mech, p_net_struct, appr, st, known_pars, times[rg[j]]) for j in 1:length(rg)]
        if any(x -> !isfinite(x), x2dot_pred)
          log_inf("x2dot_pred_nonfinite_ms")
          return Inf, (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
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
    u0 = [ode_data[1, 1], ode_data[2, 1], x3_init_val]
    prob = ODEProblem{true}(make_uode_func(appr, st, known_pars), u0, (times[1], times[end]), p)
    trace_solve(() -> "ss solve start n=" * string(length(times)) *
      " t=" * string(round(times[1], sigdigits=3)) * "→" * string(round(times[end], sigdigits=3)))
    t_start = Zygote.ignore() do
      time()
    end
    sol = solve(prob, integrator; saveat=times, abstol=abstol, reltol=reltol, sensealg=sensealg)
    trace_solve(() -> "ss solve done dt=" * string(round(time() - t_start, digits=2)) * "s" *
      " retcode=" * string(sol.retcode) * " size=" * string(size(sol, 2)))
    if !SciMLBase.successful_retcode(sol)
      log_inf("retcode=" * string(sol.retcode))
      return Inf, (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
    end
    if size(sol, 2) != length(times)
      log_inf("sol_size_mismatch_ss")
      return Inf, (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
    end
    uhat = Array(sol)
    if any(x -> !isfinite(x), uhat)
      log_inf("uhat_nonfinite_ss")
      return Inf, (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
    end

    state_err = vec(sum(abs2.((ode_data[1:2, :] .- uhat[1:2, :]) ./ state12_scale), dims=1))
    total_state = sum(weights_val .* state_err) / weights_sum

    contact_idx = findall(contact_mask)
    if !isempty(contact_idx)
      uhat_const = Zygote.dropgrad(uhat)
      x2dot_pred = [x2dot_rhs(view(uhat_const, :, j), mech, p_net_struct, appr, st, known_pars, times[j]) for j in 1:length(times)]
      if any(x -> !isfinite(x), x2dot_pred)
        log_inf("x2dot_pred_nonfinite_ss")
        return Inf, (state=Inf, x2dot=Inf, x3_range=Inf, cont=Inf, x1_rec=Inf, x3_rec=Inf)
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

use_multiple_shooting = false
use_l2_regularization = false
l2_weight = 0.0
# Fixed train/validation split (not part of hyperparameter search)
val_stride = 5
val_offset = 2

selftest = get(ENV, "HNODECB_SELFTEST", "0") == "1"
if selftest
  tprintln("=== AFM Stage1 (03) SELFTEST ===")
else
  tprintln("=== AFM Stage1 (03) ===")
end
tprintln("Switches: MS=", use_multiple_shooting, " L2=", use_l2_regularization, " (λ=", l2_weight, ")")

if selftest
  # Minimal runtime check: build NN, solve ODE once, compute one loss
  n = min(50, length(all_times))
  idx = 1:n
  ode_train = ode_data_full[:, idx]
  times_train = all_times[idx]
  x2dot_train = x2dot_all[idx]
  contact_train = contact_all[idx]

  num_hidden_layers = 1
  num_hidden_nodes = 3
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
  x3_init_val = ode_train[3, 1]
  if get(ENV, "HNODECB_SELFTEST_PHYS", "1") == "1"
    # Print physical scale diagnostics at initial time
    p_net_struct = re_pnet(θ0.p_net)
    mech = [
      bound_param(θ0.mech_raw[1], ks_bounds[1], ks_bounds[2]),
      bound_param(θ0.mech_raw[2], cs_bounds[1], cs_bounds[2])
    ]
    u0 = [ode_train[1, 1], ode_train[2, 1], x3_init_val]
    t0 = times_train[1]
    k, wd, m, c, Fd, R, dist, Fad = known_pars
    s = dist + u0[1] - u0[3]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    w = contact_weight(s, adhesion_transition)
    nn_in = [u0[1], u0[2], u0[3], delta]
    uhat = approximating_neural_network(nn_in, p_net_struct, st)[1]
    F_hertz = softplus(uhat[1], adhesion_transition) * w
    Fad_eff = Fad * w
    x2dot0 = (Fd * cos(wd * t0) - k * u0[1] - c * u0[2] + Fad_eff - F_hertz) / m
    x3dot0 = (Fad_eff - F_hertz - mech[1] * u0[3]) / mech[2]
    tprintln("  selftest phys @t0=", fmt_e(t0, sigdigits=3))
    tprintln("    u0: x1=", fmt_e(u0[1], sigdigits=3),
      " x2=", fmt_e(u0[2], sigdigits=3),
      " x3=", fmt_e(u0[3], sigdigits=3))
    tprintln("    contact: s=", fmt_e(s, sigdigits=3),
      " delta=", fmt_e(delta, sigdigits=3),
      " w=", fmt_e(w, sigdigits=3))
    tprintln("    nn_out=", fmt_e(uhat[1], sigdigits=3),
      " F_hertz=", fmt_e(F_hertz, sigdigits=3),
      " Fad_eff=", fmt_e(Fad_eff, sigdigits=3))
    tprintln("    x2dot0=", fmt_e(x2dot0, sigdigits=3),
      " x3dot0=", fmt_e(x3dot0, sigdigits=3))
  end
  loss, diag = loss_single_or_ms(θ0, ode_train, x2dot_train, contact_train, times_train,
    state12_scale_full, x2dot_scale_full, x3_scale,
    use_multiple_shooting, ms_group_size, ms_continuity_term,
    approximating_neural_network, st, known_pars, x3_init_val,
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

function objective(trial)
  try
    last_inf_reason[] = ""
    inf_context[] = "objective"
    inf_log_remaining[] = 20
    # hyperparameters
    ks0 = trial.suggest_float("ks0", ks_bounds[1], ks_bounds[2], log=true)
    cs0 = trial.suggest_float("cs0", cs_bounds[1], cs_bounds[2], log=true)
    learning_rate_adam = trial.suggest_float("learning_rate_adam", 1e-5, 1e-2, log=true)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)
    num_hidden_nodes = trial.suggest_int("num_hidden_nodes", 3, 5)
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

    state12_scale_train = state12_scale_full
    x2dot_scale_train = x2dot_scale_full

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
    x3_init_val = ode_train[3, 1]

    # loss function
    function loss_fn(θ)
      inf_context[] = "train"
      loss, diag = loss_single_or_ms(θ, ode_train, x2dot_train, contact_train, times_train,
        state12_scale_train, x2dot_scale_train, x3_scale,
        use_multiple_shooting, ms_group_size, ms_continuity_term,
        approximating_neural_network, st, known_pars, x3_init_val,
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
    lr_eps = env_float("HNODECB_LR_EPS", 1e-12)
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
      approximating_neural_network, st, known_pars, x3_init_val,
      l2_weight, re_pnet)
    train_reason = last_inf_reason[]

    # validation loss (no L2)
    inf_context[] = "val_eval"
    last_inf_reason[] = ""
    val_loss, _ = loss_single_or_ms(θ_best, ode_val, x2dot_val, contact_val, times_val,
      state12_scale_full, x2dot_scale_full, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_init_val,
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

# TPE optimization
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
  tprintln("  Rank ", i,
    " -- train=", fmt_e(rec.train_loss, sigdigits=4),
    " val=", fmt_e(rec.val_loss, sigdigits=4),
    " | ks0=", fmt_e(ks0, sigdigits=3),
    " cs0=", fmt_e(cs0, sigdigits=3))
end

serialize(result_folder * "/" * result_name_string, (
  study=nothing,
  trial_parameters=trial_parameters,
  best=best,
  bounds=(ks=ks_bounds, cs=cs_bounds),
  use_multiple_shooting=use_multiple_shooting,
  use_l2_regularization=use_l2_regularization,
  val_stride=val_stride,
  val_offset=val_offset,
  x3_obs_fraction=0.0,
  error_level=error_level
))
end
