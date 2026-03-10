cd(@__DIR__)

using Serialization, DifferentialEquations, Statistics, Printf, DataFrames, Profile, Dates
using ComponentArrays, SciMLSensitivity, StableRNGs
using Zygote
using Optimisers
using DiffEqFlux, Flux, Lux

const HAVE_FORWARDDIFF = let ok = false
  try
    @eval using ForwardDiff
    ok = true
  catch
    ok = false
  end
  ok
end

include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_functions.jl")
include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

sigmoid(x) = 1 / (1 + exp(-x))
logit(x) = log(x / (1 - x))

function bound_param(raw, lo, hi)
  return lo + (hi - lo) * sigmoid(raw)
end

function raw_from_value(val, lo, hi)
  z = (val - lo) / (hi - lo)
  z = clamp(z, 1e-6, 1 - 1e-6)
  return logit(z)
end

nn_input_from_state(u) = u[1:2]
nn_input_from_values(x1, x2) = [x1, x2]

function build_nn()
  hidden = 2^5
  layers = Any[]
  push!(layers, Lux.Dense(2, hidden, gelu; init_weight=Lux.glorot_uniform, use_bias=false))
  for _ in 1:3
    push!(layers, Lux.Dense(hidden, hidden, gelu; init_weight=Lux.glorot_uniform, use_bias=false))
  end
  push!(layers, Lux.Dense(hidden, 1; init_weight=Lux.glorot_uniform, use_bias=false))
  return Lux.Chain(layers...)
end

function relative_rmse_pct(err_sum, truth_sum, count, eps)
  denom = max(sqrt(truth_sum / max(count, 1)), eps)
  return 100 * sqrt(err_sum / max(count, 1)) / denom
end

function make_uode_func_current(appr, st, known_pars, true_contact_weight_at_time; nn_gain::Float64)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  function f(du, u, p, t)
    ks = p.mech[1]
    cs = p.mech[2]
    s = dist + u[1] - u[3]
    w_pred = contact_weight(s, adhesion_transition)
    nn_in = nn_input_from_state(u)
    uhat = appr(nn_in, p.p_net, st)[1]
    F_hertz = nn_gain * uhat[1] * true_contact_weight_at_time(t)
    Fad_eff = Fad * w_pred
    @inbounds du[1] = u[2]
    @inbounds du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
    @inbounds du[3] = (Fad_eff - F_hertz - ks * u[3]) / cs
  end
  return f
end

function make_uode_func_flat(appr, st, known_pars, true_contact_weight_at_time; nn_gain::Float64)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  function f(du, u, p, t)
    ks = p.ks
    cs = p.cs
    s = dist + u[1] - u[3]
    w_pred = contact_weight(s, adhesion_transition)
    nn_in = nn_input_from_state(u)
    uhat = appr(nn_in, p.p_net, st)[1]
    F_hertz = nn_gain * uhat[1] * true_contact_weight_at_time(t)
    Fad_eff = Fad * w_pred
    @inbounds du[1] = u[2]
    @inbounds du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
    @inbounds du[3] = (Fad_eff - F_hertz - ks * u[3]) / cs
  end
  return f
end

function make_uode_func_current_oop(appr, st, known_pars, true_contact_weight_at_time; nn_gain::Float64)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  function f(u, p, t)
    ks = p.mech[1]
    cs = p.mech[2]
    s = dist + u[1] - u[3]
    w_pred = contact_weight(s, adhesion_transition)
    nn_in = nn_input_from_state(u)
    uhat = appr(nn_in, p.p_net, st)[1]
    F_hertz = nn_gain * uhat[1] * true_contact_weight_at_time(t)
    Fad_eff = Fad * w_pred
    du1 = u[2]
    du2 = (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
    du3 = (Fad_eff - F_hertz - ks * u[3]) / cs
    return [du1, du2, du3]
  end
  return f
end

function make_uode_func_flat_oop(appr, st, known_pars, true_contact_weight_at_time; nn_gain::Float64)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  function f(u, p, t)
    ks = p.ks
    cs = p.cs
    s = dist + u[1] - u[3]
    w_pred = contact_weight(s, adhesion_transition)
    nn_in = nn_input_from_state(u)
    uhat = appr(nn_in, p.p_net, st)[1]
    F_hertz = nn_gain * uhat[1] * true_contact_weight_at_time(t)
    Fad_eff = Fad * w_pred
    du1 = u[2]
    du2 = (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
    du3 = (Fad_eff - F_hertz - ks * u[3]) / cs
    return [du1, du2, du3]
  end
  return f
end

function fhertz_true_from_states_diag(u_mat, idxs)
  out = Vector{Float64}(undef, length(idxs))
  @inbounds for (k, j) in enumerate(idxs)
    s = dist + u_mat[1, j] - u_mat[3, j]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    out[k] = (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
  end
  return out
end

function fhertz_pred_and_raw_from_states_diag(u_mat, idxs, p_net_struct, appr, st; nn_gain::Float64=1.0)
  out = Vector{Float64}(undef, length(idxs))
  raw = Vector{Float64}(undef, length(idxs))
  @inbounds for (k, j) in enumerate(idxs)
    w_true = true_contact_weight_all[j]
    nn_in = nn_input_from_values(u_mat[1, j], u_mat[2, j])
    uhat = appr(nn_in, p_net_struct, st)[1]
    raw[k] = uhat[1]
    out[k] = nn_gain * uhat[1] * w_true
  end
  return out, raw
end

function effective_contact_positions_diag(idxs, fh_true)
  contact_pos = [k for (k, j) in enumerate(idxs) if contact_all[j]]
  if isempty(contact_pos)
    return Int[]
  end
  max_true = maximum(fh_true[contact_pos])
  if !isfinite(max_true) || max_true <= 0.0
    return contact_pos
  end
  floor_val = 0.05 * max_true
  eff_pos = [k for k in contact_pos if fh_true[k] > floor_val]
  return isempty(eff_pos) ? contact_pos : eff_pos
end

function fhertz_monitor_metrics_diag(u_mat, idxs, p_net_struct, appr, st; fh_true_ref=nothing, eff_pos_ref=nothing, nn_gain::Float64=1.0)
  fh_true = fh_true_ref === nothing ? fhertz_true_from_states_diag(u_mat, idxs) : fh_true_ref
  fh_pred, raw = fhertz_pred_and_raw_from_states_diag(u_mat, idxs, p_net_struct, appr, st; nn_gain=nn_gain)
  eff_pos = eff_pos_ref === nothing ? effective_contact_positions_diag(idxs, fh_true) : eff_pos_ref
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

function x2dot_rhs_local(u, p_net, appr, st, known_pars, true_contact_weight_at_time, t; nn_gain::Float64)
  k, wd, m, c, Fd, R, dist, Fad = known_pars
  s = dist + u[1] - u[3]
  w_pred = contact_weight(s, adhesion_transition)
  nn_in = nn_input_from_state(u)
  uhat = appr(nn_in, p_net, st)[1]
  F_hertz = nn_gain * uhat[1] * true_contact_weight_at_time(t)
  Fad_eff = Fad * w_pred
  return (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
end

function finite_diff_mech(loss_fn, theta; eps=1e-6)
  fd = similar(theta.mech_raw)
  hs = similar(theta.mech_raw)
  for i in eachindex(theta.mech_raw)
    h = eps * max(1.0, abs(theta.mech_raw[i]))
    hs[i] = h
    θp = ComponentVector(p_net=copy(theta.p_net), mech_raw=copy(theta.mech_raw))
    θm = ComponentVector(p_net=copy(theta.p_net), mech_raw=copy(theta.mech_raw))
    θp.mech_raw[i] += h
    θm.mech_raw[i] -= h
    lp = loss_fn(θp)
    lm = loss_fn(θm)
    fd[i] = (lp - lm) / (2h)
  end
  return fd, hs
end

function ad_mech(loss_fn, theta)
  loss, back = Zygote.pullback(loss_fn, theta)
  grad = first(back(1.0))
  return loss, grad.mech_raw
end

function finite_diff_vec2(loss_fn, raw; eps=1e-6)
  fd = similar(raw)
  hs = similar(raw)
  for i in eachindex(raw)
    h = eps * max(1.0, abs(raw[i]))
    hs[i] = h
    rp = copy(raw)
    rm = copy(raw)
    rp[i] += h
    rm[i] -= h
    fd[i] = (loss_fn(rp) - loss_fn(rm)) / (2h)
  end
  return fd, hs
end

function ad_vec2(loss_fn, raw)
  loss, back = Zygote.pullback(loss_fn, raw)
  grad = first(back(1.0))
  return loss, grad
end

function forwarddiff_vec2(loss_fn, raw)
  HAVE_FORWARDDIFF || error("ForwardDiff unavailable in current project.")
  loss = loss_fn(raw)
  grad = ForwardDiff.gradient(loss_fn, raw)
  return loss, grad
end

function report_diag(name, loss_fn, theta)
  println("=== ", name, " ===")
  try
    loss_ad, ad = ad_mech(loss_fn, theta)
    fd, hs = finite_diff_mech(loss_fn, theta)
    println("loss = ", @sprintf("%.8e", loss_ad))
    println("AD(mech_raw) = [", @sprintf("%.16e", ad[1]), ", ", @sprintf("%.16e", ad[2]), "]")
    println("FD(mech_raw) = [", @sprintf("%.16e", fd[1]), ", ", @sprintf("%.16e", fd[2]), "]")
    println("diff        = [", @sprintf("%.16e", ad[1] - fd[1]), ", ", @sprintf("%.16e", ad[2] - fd[2]), "]")
    println("h           = [", @sprintf("%.16e", hs[1]), ", ", @sprintf("%.16e", hs[2]), "]")
  catch ex
    println("ERROR: ", sprint(showerror, ex))
  end
  println()
end

function report_diag_vec2(name, loss_fn, raw)
  println("=== ", name, " ===")
  try
    loss_ad, ad = ad_vec2(loss_fn, raw)
    fd, hs = finite_diff_vec2(loss_fn, raw)
    println("loss = ", @sprintf("%.8e", loss_ad))
    println("AD(raw) = [", @sprintf("%.16e", ad[1]), ", ", @sprintf("%.16e", ad[2]), "]")
    println("FD(raw) = [", @sprintf("%.16e", fd[1]), ", ", @sprintf("%.16e", fd[2]), "]")
    println("diff    = [", @sprintf("%.16e", ad[1] - fd[1]), ", ", @sprintf("%.16e", ad[2] - fd[2]), "]")
    println("h       = [", @sprintf("%.16e", hs[1]), ", ", @sprintf("%.16e", hs[2]), "]")
  catch ex
    println("ERROR: ", sprint(showerror, ex))
  end
  println()
end

function report_diag_vec2_fwd(name, loss_fn, raw)
  println("=== ", name, " ===")
  try
    loss_z, ad = ad_vec2(loss_fn, raw)
    fd, hs = finite_diff_vec2(loss_fn, raw)
    println("loss(z) = ", @sprintf("%.8e", loss_z))
    println("AD(raw)  = [", @sprintf("%.16e", ad[1]), ", ", @sprintf("%.16e", ad[2]), "]")
    if HAVE_FORWARDDIFF
      loss_f, fwd = forwarddiff_vec2(loss_fn, raw)
      println("loss(f) = ", @sprintf("%.8e", loss_f))
      println("FWD(raw) = [", @sprintf("%.16e", fwd[1]), ", ", @sprintf("%.16e", fwd[2]), "]")
      println("FWD-FD   = [", @sprintf("%.16e", fwd[1] - fd[1]), ", ", @sprintf("%.16e", fwd[2] - fd[2]), "]")
    else
      println("FWD(raw) = unavailable")
    end
    println("FD(raw)  = [", @sprintf("%.16e", fd[1]), ", ", @sprintf("%.16e", fd[2]), "]")
    println("AD-FD    = [", @sprintf("%.16e", ad[1] - fd[1]), ", ", @sprintf("%.16e", ad[2] - fd[2]), "]")
    println("h        = [", @sprintf("%.16e", hs[1]), ", ", @sprintf("%.16e", hs[2]), "]")
  catch ex
    println("ERROR: ", sprint(showerror, ex))
  end
  println()
end

function finite_diff_pnet_samples(loss_fn, theta, sample_idx; eps=1e-6)
  fd = Dict{Int, Float64}()
  hs = Dict{Int, Float64}()
  base_p = copy(theta.p_net)
  base_m = copy(theta.mech_raw)
  for idx in sample_idx
    h = eps * max(1.0, abs(base_p[idx]))
    hs[idx] = h
    θp = ComponentVector(p_net=copy(base_p), mech_raw=copy(base_m))
    θm = ComponentVector(p_net=copy(base_p), mech_raw=copy(base_m))
    θp.p_net[idx] += h
    θm.p_net[idx] -= h
    fd[idx] = (loss_fn(θp) - loss_fn(θm)) / (2h)
  end
  return fd, hs
end

function ad_pnet(loss_fn, theta)
  loss, back = Zygote.pullback(loss_fn, theta)
  grad = first(back(1.0))
  return loss, grad.p_net
end

function default_pnet_sample_idx(pnet_len, nsamples)
  ns = min(max(nsamples, 1), pnet_len)
  idx = unique(round.(Int, collect(range(1, pnet_len, length=ns))))
  return sort!(idx)
end

function report_diag_pnet_samples(name, loss_fn, theta, sample_idx; eps=1e-6)
  println("=== ", name, " ===")
  try
    loss_ad, ad = ad_pnet(loss_fn, theta)
    fd, hs = finite_diff_pnet_samples(loss_fn, theta, sample_idx; eps=eps)
    println("loss = ", @sprintf("%.8e", loss_ad))
    println("samples = [", join(sample_idx, ", "), "]")
    n_ok = 0
    for idx in sample_idx
      adv = ad[idx]
      fdv = fd[idx]
      diff = adv - fdv
      denom = max(abs(fdv), 1e-14)
      rel = abs(diff) / denom
      same_sign = signbit(adv) == signbit(fdv) || adv == 0.0 || fdv == 0.0
      n_ok += same_sign ? 1 : 0
      println(
        "idx=", idx,
        " AD=", @sprintf("%.16e", adv),
        " FD=", @sprintf("%.16e", fdv),
        " diff=", @sprintf("%.16e", diff),
        " rel=", @sprintf("%.6e", rel),
        " h=", @sprintf("%.16e", hs[idx]),
        " sign_match=", same_sign
      )
    end
    println("sign_match_count = ", n_ok, " / ", length(sample_idx))
  catch ex
    println("ERROR: ", sprint(showerror, ex))
  end
  println()
end

function theta_with_pnet(pnet)
  return ComponentVector(p_net=copy(pnet), mech_raw=copy(theta0.mech_raw))
end

function theta_with_mechraw(raw)
  return ComponentVector(p_net=copy(theta0.p_net), mech_raw=copy(raw))
end

function loss_state_current_oop_pnet_only(pnet; sensealg_local=sensealg)
  return loss_state_current_oop(theta_with_pnet(pnet); sensealg_local=sensealg_local)
end

function loss_x2dot_current_oop_pnet_only(pnet; drop_uhat::Bool, sensealg_local=sensealg)
  return loss_x2dot_current_oop(theta_with_pnet(pnet); drop_uhat=drop_uhat, sensealg_local=sensealg_local)
end

function loss_train_active_current_oop_pnet_only(pnet; sensealg_local=sensealg)
  return loss_train_active_current_oop(theta_with_pnet(pnet); sensealg_local=sensealg_local)
end

function time_pullback_triplet(loss_fn, x)
  t_loss = @timed loss_fn(x)
  t_pull = @timed Zygote.pullback(loss_fn, x)
  loss2, back = t_pull.value
  t_back = @timed first(back(1.0))
  return (
    loss=t_loss.value,
    loss_dt=t_loss.time,
    loss_bytes=t_loss.bytes,
    pull_loss=loss2,
    pull_dt=t_pull.time,
    pull_bytes=t_pull.bytes,
    grad=t_back.value,
    back_dt=t_back.time,
    back_bytes=t_back.bytes,
  )
end

function grad_l2(grad)
  try
    return sqrt(sum(abs2, grad))
  catch
    return NaN
  end
end

function report_timing_triplet(name, loss_fn, x)
  println("=== ", name, " ===")
  try
    r = time_pullback_triplet(loss_fn, x)
    println("loss        = ", @sprintf("%.8e", r.loss))
    println("loss_dt_s   = ", @sprintf("%.3f", r.loss_dt), " bytes=", r.loss_bytes)
    println("pull_dt_s   = ", @sprintf("%.3f", r.pull_dt), " bytes=", r.pull_bytes, " loss2=", @sprintf("%.8e", r.pull_loss))
    println("back_dt_s   = ", @sprintf("%.3f", r.back_dt), " bytes=", r.back_bytes, " grad_l2=", @sprintf("%.8e", grad_l2(r.grad)))
  catch ex
    println("ERROR: ", sprint(showerror, ex))
  end
  println()
end

function capture_profile_print(; maxdepth=24)
  io = IOBuffer()
  Profile.print(io; format=:flat, sortedby=:count, maxdepth=maxdepth)
  return String(take!(io))
end

const DIAG_LOG_IO = Ref{Union{Nothing, IO}}(nothing)
const DIAG_LOG_PATH = Ref{Union{Nothing, String}}(nothing)

function emitln(args...)
  msg = join(string.(args), "")
  println(msg)
  if DIAG_LOG_IO[] !== nothing
    println(DIAG_LOG_IO[], msg)
    flush(DIAG_LOG_IO[])
  end
end

function dprintln(args...)
  emitln("[", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "] ", join(string.(args), ""))
end

function init_diag_log(; prefix::String)
  log_dir = get(ENV, "HNODECB_DIAG_LOG_DIR", joinpath(REPO_ROOT, "logs", "stage2_step2a", "local"))
  mkpath(log_dir)
  stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
  log_path = joinpath(log_dir, prefix * "_" * stamp * ".txt")
  DIAG_LOG_IO[] = open(log_path, "w")
  DIAG_LOG_PATH[] = log_path
  emitln("diag_log_path = ", log_path)
end

function close_diag_log()
  if DIAG_LOG_IO[] !== nothing
    close(DIAG_LOG_IO[])
    DIAG_LOG_IO[] = nothing
  end
end

function report_formal_precheck_profile(name, loss_fn, x)
  emitln("=== ", name, " ===")
  try
    loss_t0 = time_ns()
    dprintln("trace[profile]: loss_begin")
    loss = loss_fn(x)
    dprintln("trace[profile]: loss_done dt_s=", @sprintf("%.6f", (time_ns() - loss_t0) / 1e9),
      " loss=", @sprintf("%.8e", loss))

    pull_t0 = time_ns()
    dprintln("trace[profile]: pullback_begin")
    pull_timed = @timed Zygote.pullback(loss_fn, x)
    loss2, back = pull_timed.value
    dprintln("trace[profile]: pullback_done dt_s=", @sprintf("%.6f", pull_timed.time),
      " bytes=", pull_timed.bytes,
      " loss2=", @sprintf("%.8e", loss2))

    Profile.clear()
    dprintln("trace[profile]: backward_begin")
    back_t0 = time_ns()
    grad = nothing
    Profile.clear()
    @profile begin
      grad = first(back(1.0))
    end
    back_dt = (time_ns() - back_t0) / 1e9
    dprintln("trace[profile]: backward_done dt_s=", @sprintf("%.6f", back_dt),
      " grad_l2=", @sprintf("%.8e", grad_l2(grad)))
    dprintln("profile_hotspots_begin")
    hotspots = capture_profile_print()
    print(hotspots)
    if DIAG_LOG_IO[] !== nothing
      print(DIAG_LOG_IO[], hotspots)
      flush(DIAG_LOG_IO[])
    end
    dprintln("profile_hotspots_end")
  catch ex
    emitln("ERROR: ", sprint(showerror, ex))
    bt = catch_backtrace()
    dprintln("stacktrace_begin")
    for line in stacktrace(bt)
      emitln(line)
    end
    dprintln("stacktrace_end")
  end
  emitln()
end

stage1 = deserialize(joinpath(REPO_ROOT, "step2a_hyperparameter_tuning", "hyperparameter_tuning_first_stage", "results_afm", "afm_param_stage1_03.jld"))
stage1pluslight = deserialize(joinpath(REPO_ROOT, "step2a_hyperparameter_tuning", "hyperparameter_tuning_first_stage", "results_afm", "afm_param_stage1pluslight_03.jld"))
ode_data_full = deserialize(joinpath(REPO_ROOT, "datasets", "e0.0", "data", "ode_data_afm_dmt_kv.jld"))
solution_dataframe_full = deserialize(joinpath(REPO_ROOT, "datasets", "e0.0", "data", "pert_df_afm_dmt_kv.jld"))

contact_idx = findfirst(solution_dataframe_full.contact .== 1)
solution_dataframe_full = solution_dataframe_full[contact_idx:end, :]
ode_data_full = ode_data_full[:, contact_idx:end]

all_times = solution_dataframe_full.t
x2dot_all = solution_dataframe_full.x2dot
contact_all = solution_dataframe_full.contact .== 1
true_s_all = Float64.(solution_dataframe_full.s)
true_contact_weight_at_time = make_contact_weight_lookup(all_times, true_s_all)
true_contact_weight_all = contact_weight.(true_s_all, adhesion_transition)
monitor_idx_full = collect(1:length(all_times))

scale_eps = 1e-9
state12_scale_full = vec(maximum(ode_data_full[1:2, :], dims=2) - minimum(ode_data_full[1:2, :], dims=2))
state12_scale_full = max.(state12_scale_full, scale_eps)
x2dot_scale_full = max(maximum(x2dot_all) - minimum(x2dot_all), scale_eps)

val_stride = stage1.val_stride
val_offset = stage1.val_offset
train_idx = [i for i in 1:length(all_times) if (i - val_offset) % val_stride != 0]

npts = let v = tryparse(Int, get(ENV, "HNODECB_MECH_DIAG_NPTS", "128"))
  (v === nothing || v < 16) ? 128 : v
end
npts = min(npts, length(train_idx))
train_idx = train_idx[1:npts]
diag_mode = strip(get(ENV, "HNODECB_MECH_DIAG_MODE", "all"))
pnet_diag_nsamples = let v = tryparse(Int, get(ENV, "HNODECB_PNET_DIAG_NSAMPLES", "6"))
  (v === nothing || v < 1) ? 6 : v
end
formal_profile_branch = lowercase(strip(get(ENV, "HNODECB_PRECHECK_PROFILE_BRANCH", "active")))
formal_profile_with_monitor = get(ENV, "HNODECB_PRECHECK_PROFILE_WITH_MONITOR", "1") == "1"

ode_train = ode_data_full[:, train_idx]
times_train = all_times[train_idx]
x2dot_train = x2dot_all[train_idx]
contact_train = contact_all[train_idx]
weights_val = ifelse.(contact_train, 4.27, 1.0)
weights_sum = sum(weights_val)
contact_positions = findall(contact_train)

known_pars = original_parameters[1:8]
ks_bounds = stage1.bounds.ks
cs_bounds = stage1.bounds.cs

sorted_stage1 = sort(stage1.trial_parameters, by = x -> hasproperty(x, :val_loss) ? x.val_loss : x.loss)
candidate6 = sorted_stage1[6]
ks0 = candidate6.params["ks0"]
cs0 = candidate6.params["cs0"]

warm_rec = first(filter(r -> r.params["base_rank"] == 6, stage1pluslight.selected))
nn_gain = 4.89e-5

approximating_neural_network = build_nn()
p_net0, st = Lux.setup(StableRNG(0), approximating_neural_network)
p_net0 = Flux.f64(p_net0)
p_template_vec, re_pnet = Optimisers.destructure(p_net0)
p_net_vec = copy(warm_rec.p_net_vec)
length(p_net_vec) == length(p_template_vec) || error("Warm-start p_net_vec length mismatch.")

raw_init = [
  raw_from_value(ks0, ks_bounds[1], ks_bounds[2]),
  raw_from_value(cs0, cs_bounds[1], cs_bounds[2])
]
theta0 = ComponentVector(p_net=p_net_vec, mech_raw=raw_init)
pnet_sample_idx = default_pnet_sample_idx(length(theta0.p_net), pnet_diag_nsamples)

integrator = Rosenbrock23(autodiff=false)
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))
abstol = 1e-8
reltol = 1e-8
ode_maxiters = 1_000_000
x3_t0_val = ode_train[3, 1]

function local_rhs_x3dot_current(theta)
  p_net_struct = re_pnet(theta.p_net)
  ks = bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2])
  cs = bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])
  u = ode_train[:, 1]
  t = times_train[1]
  s = known_pars[7] + u[1] - u[3]
  w_pred = contact_weight(s, adhesion_transition)
  nn_in = nn_input_from_state(u)
  uhat = approximating_neural_network(nn_in, p_net_struct, st)[1]
  F_hertz = nn_gain * uhat[1] * true_contact_weight_at_time(t)
  Fad_eff = known_pars[8] * w_pred
  return (Fad_eff - F_hertz - ks * u[3]) / cs
end

function local_rhs_x3dot_packed_current(theta)
  p_net_struct = re_pnet(theta.p_net)
  mech = [bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u = ode_train[:, 1]
  t = times_train[1]
  du = zeros(3)
  f = make_uode_func_current(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain)
  f(du, u, p, t)
  return du[3]
end

function local_rhs_x3dot_packed_component_fields(theta)
  p_net_struct = re_pnet(theta.p_net)
  ks = bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2])
  cs = bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])
  p = ComponentVector(p_net=p_net_struct, ks=ks, cs=cs)
  u = ode_train[:, 1]
  t = times_train[1]
  du = zeros(3)
  f = make_uode_func_flat(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain)
  f(du, u, p, t)
  return du[3]
end

function loss_state_current(theta; sensealg_local=sensealg)
  p_net_struct = re_pnet(theta.p_net)
  mech = [bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{true}(make_uode_func_current(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg_local, maxiters=ode_maxiters)
  uhat = Array(sol)
  state_err = vec(sum(abs2.((ode_train[1:2, :] .- uhat[1:2, :]) ./ state12_scale_full), dims=1))
  return sum(weights_val .* state_err) / weights_sum
end

function loss_state_component_fields(theta; sensealg_local=sensealg)
  p_net_struct = re_pnet(theta.p_net)
  ks = bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2])
  cs = bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])
  p = ComponentVector(p_net=p_net_struct, ks=ks, cs=cs)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{true}(make_uode_func_flat(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg_local, maxiters=ode_maxiters)
  uhat = Array(sol)
  state_err = vec(sum(abs2.((ode_train[1:2, :] .- uhat[1:2, :]) ./ state12_scale_full), dims=1))
  return sum(weights_val .* state_err) / weights_sum
end

function loss_state_current_raw_only(raw; sensealg_local=sensealg)
  p_net_struct = re_pnet(theta0.p_net)
  mech = [bound_param(raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{true}(make_uode_func_current(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg_local, maxiters=ode_maxiters)
  uhat = Array(sol)
  state_err = vec(sum(abs2.((ode_train[1:2, :] .- uhat[1:2, :]) ./ state12_scale_full), dims=1))
  return sum(weights_val .* state_err) / weights_sum
end

function loss_state_current_oop(theta; sensealg_local=sensealg)
  p_net_struct = re_pnet(theta.p_net)
  mech = [bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{false}(make_uode_func_current_oop(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg_local, maxiters=ode_maxiters)
  uhat = Array(sol)
  state_err = vec(sum(abs2.((ode_train[1:2, :] .- uhat[1:2, :]) ./ state12_scale_full), dims=1))
  return sum(weights_val .* state_err) / weights_sum
end

function loss_state_current_oop_gain(theta; sensealg_local=sensealg, nn_gain_local::Float64=nn_gain, with_monitor::Bool=false)
  p_net_struct = re_pnet(theta.p_net)
  mech = [bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{false}(make_uode_func_current_oop(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain_local),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg_local, maxiters=ode_maxiters)
  uhat = Array(sol)
  state_err = vec(sum(abs2.((ode_train[1:2, :] .- uhat[1:2, :]) ./ state12_scale_full), dims=1))
  if with_monitor
    Zygote.ignore() do
      fh_true = fhertz_true_from_states_diag(ode_data_full, monitor_idx_full)
      eff_pos = effective_contact_positions_diag(monitor_idx_full, fh_true)
      fhertz_monitor_metrics_diag(ode_data_full, monitor_idx_full, p_net_struct, approximating_neural_network, st;
        fh_true_ref=fh_true, eff_pos_ref=eff_pos, nn_gain=nn_gain_local)
    end
  end
  return sum(weights_val .* state_err) / weights_sum
end

function loss_state_current_oop_raw_only(raw; sensealg_local=sensealg)
  p_net_struct = re_pnet(theta0.p_net)
  mech = [bound_param(raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{false}(make_uode_func_current_oop(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg_local, maxiters=ode_maxiters)
  uhat = Array(sol)
  state_err = vec(sum(abs2.((ode_train[1:2, :] .- uhat[1:2, :]) ./ state12_scale_full), dims=1))
  return sum(weights_val .* state_err) / weights_sum
end

function loss_x2dot_current(theta; drop_uhat::Bool)
  p_net_struct = re_pnet(theta.p_net)
  mech = [bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{true}(make_uode_func_current(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg, maxiters=ode_maxiters)
  uhat = Array(sol)
  uhat_eval = drop_uhat ? Zygote.dropgrad(uhat) : uhat
  x2dot_pred = [
    x2dot_rhs_local(view(uhat_eval, :, j), p_net_struct, approximating_neural_network, st, known_pars, true_contact_weight_at_time, times_train[j]; nn_gain=nn_gain)
    for j in contact_positions
  ]
  x2_err = abs2.((x2dot_train[contact_positions] .- x2dot_pred) ./ x2dot_scale_full)
  x2_w = weights_val[contact_positions]
  return sum(x2_w .* x2_err) / sum(x2_w)
end

function loss_x2dot_current_oop(theta; drop_uhat::Bool, sensealg_local=sensealg)
  p_net_struct = re_pnet(theta.p_net)
  mech = [bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{false}(make_uode_func_current_oop(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg_local, maxiters=ode_maxiters)
  uhat = Array(sol)
  uhat_eval = drop_uhat ? Zygote.dropgrad(uhat) : uhat
  x2dot_pred = [
    x2dot_rhs_local(view(uhat_eval, :, j), p_net_struct, approximating_neural_network, st, known_pars, true_contact_weight_at_time, times_train[j]; nn_gain=nn_gain)
    for j in contact_positions
  ]
  x2_err = abs2.((x2dot_train[contact_positions] .- x2dot_pred) ./ x2dot_scale_full)
  x2_w = weights_val[contact_positions]
  return sum(x2_w .* x2_err) / sum(x2_w)
end

function loss_x2dot_current_oop_gain(theta; drop_uhat::Bool, sensealg_local=sensealg, nn_gain_local::Float64=nn_gain, with_monitor::Bool=false)
  p_net_struct = re_pnet(theta.p_net)
  mech = [bound_param(theta.mech_raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(theta.mech_raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{false}(make_uode_func_current_oop(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain_local),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg_local, maxiters=ode_maxiters)
  uhat = Array(sol)
  uhat_eval = drop_uhat ? Zygote.dropgrad(uhat) : uhat
  x2dot_pred = [
    x2dot_rhs_local(view(uhat_eval, :, j), p_net_struct, approximating_neural_network, st, known_pars, true_contact_weight_at_time, times_train[j]; nn_gain=nn_gain_local)
    for j in contact_positions
  ]
  x2_err = abs2.((x2dot_train[contact_positions] .- x2dot_pred) ./ x2dot_scale_full)
  x2_w = weights_val[contact_positions]
  if with_monitor
    Zygote.ignore() do
      fh_true = fhertz_true_from_states_diag(ode_data_full, monitor_idx_full)
      eff_pos = effective_contact_positions_diag(monitor_idx_full, fh_true)
      fhertz_monitor_metrics_diag(ode_data_full, monitor_idx_full, p_net_struct, approximating_neural_network, st;
        fh_true_ref=fh_true, eff_pos_ref=eff_pos, nn_gain=nn_gain_local)
    end
  end
  return sum(x2_w .* x2_err) / sum(x2_w)
end

function loss_x2dot_current_oop_raw_only(raw; drop_uhat::Bool, sensealg_local=sensealg)
  p_net_struct = re_pnet(theta0.p_net)
  mech = [bound_param(raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{false}(make_uode_func_current_oop(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg_local, maxiters=ode_maxiters)
  uhat = Array(sol)
  uhat_eval = drop_uhat ? Zygote.dropgrad(uhat) : uhat
  x2dot_pred = [
    x2dot_rhs_local(view(uhat_eval, :, j), p_net_struct, approximating_neural_network, st, known_pars, true_contact_weight_at_time, times_train[j]; nn_gain=nn_gain)
    for j in contact_positions
  ]
  x2_err = abs2.((x2dot_train[contact_positions] .- x2dot_pred) ./ x2dot_scale_full)
  x2_w = weights_val[contact_positions]
  return sum(x2_w .* x2_err) / sum(x2_w)
end

function loss_x2dot_current_raw_only(raw; drop_uhat::Bool, sensealg_local=sensealg)
  p_net_struct = re_pnet(theta0.p_net)
  mech = [bound_param(raw[1], ks_bounds[1], ks_bounds[2]),
          bound_param(raw[2], cs_bounds[1], cs_bounds[2])]
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  u0 = [ode_train[1, 1], ode_train[2, 1], x3_t0_val]
  prob = ODEProblem{true}(make_uode_func_current(approximating_neural_network, st, known_pars, true_contact_weight_at_time; nn_gain=nn_gain),
    u0, (times_train[1], times_train[end]), p)
  sol = solve(prob, integrator; saveat=times_train, abstol=abstol, reltol=reltol, sensealg=sensealg_local, maxiters=ode_maxiters)
  uhat = Array(sol)
  uhat_eval = drop_uhat ? Zygote.dropgrad(uhat) : uhat
  x2dot_pred = [
    x2dot_rhs_local(view(uhat_eval, :, j), p_net_struct, approximating_neural_network, st, known_pars, true_contact_weight_at_time, times_train[j]; nn_gain=nn_gain)
    for j in contact_positions
  ]
  x2_err = abs2.((x2dot_train[contact_positions] .- x2dot_pred) ./ x2dot_scale_full)
  x2_w = weights_val[contact_positions]
  return sum(x2_w .* x2_err) / sum(x2_w)
end

function loss_train_active_current(theta; sensealg_local=sensealg)
  return loss_state_current(theta; sensealg_local=sensealg_local) +
         loss_x2dot_current(theta; drop_uhat=false)
end

function loss_train_active_current_oop(theta; sensealg_local=sensealg)
  return loss_state_current_oop(theta; sensealg_local=sensealg_local) +
         loss_x2dot_current_oop(theta; drop_uhat=false, sensealg_local=sensealg_local)
end

function loss_train_active_current_oop_gain(theta; sensealg_local=sensealg, nn_gain_local::Float64=nn_gain, with_monitor::Bool=false)
  return loss_state_current_oop_gain(theta; sensealg_local=sensealg_local, nn_gain_local=nn_gain_local, with_monitor=with_monitor) +
         loss_x2dot_current_oop_gain(theta; drop_uhat=false, sensealg_local=sensealg_local, nn_gain_local=nn_gain_local, with_monitor=false)
end

println("mode = ", diag_mode)
println("npts = ", npts)
println("candidate rank 6 ks0 = ", @sprintf("%.16e", ks0), " cs0 = ", @sprintf("%.16e", cs0))
println("nn_gain = ", @sprintf("%.16e", nn_gain))
println("pnet_diag_nsamples = ", pnet_diag_nsamples)
println("formal_profile_branch = ", formal_profile_branch)
println("formal_profile_with_monitor = ", formal_profile_with_monitor)
println()

if diag_mode == "all"
  report_diag("RHS-local x3dot wrt mech_raw", local_rhs_x3dot_current, theta0)
  report_diag("RHS-local x3dot via current packed p.mech", local_rhs_x3dot_packed_current, theta0)
  report_diag("RHS-local x3dot via flat packed p.ks/p.cs", local_rhs_x3dot_packed_component_fields, theta0)
  report_diag("x2dot-loss current (dropgrad uhat)", θ -> loss_x2dot_current(θ; drop_uhat=true), theta0)
  report_diag("x2dot-loss current (no dropgrad)", θ -> loss_x2dot_current(θ; drop_uhat=false), theta0)
  report_diag("state-loss current mech Vector", loss_state_current, theta0)
  report_diag("state-loss ComponentVector flat ks/cs", loss_state_component_fields, theta0)
  report_diag("state-loss current mech Vector + QuadratureAdjoint(ZygoteVJP)", θ -> loss_state_current(θ; sensealg_local=QuadratureAdjoint(autojacvec=ZygoteVJP())), theta0)
  report_diag("state-loss current mech Vector + InterpolatingAdjoint(ZygoteVJP)", θ -> loss_state_current(θ; sensealg_local=InterpolatingAdjoint(autojacvec=ZygoteVJP())), theta0)
elseif diag_mode == "backend"
  report_diag("state-loss current mech Vector", loss_state_current, theta0)
  report_diag_vec2("state-loss raw-only current + QuadratureAdjoint(ReverseDiffVJP)", raw -> loss_state_current_raw_only(raw; sensealg_local=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))), copy(theta0.mech_raw))
  report_diag_vec2("state-loss raw-only current + QuadratureAdjoint(ZygoteVJP)", raw -> loss_state_current_raw_only(raw; sensealg_local=QuadratureAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
  report_diag_vec2("state-loss raw-only current + InterpolatingAdjoint(ZygoteVJP)", raw -> loss_state_current_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
elseif diag_mode == "oop"
  report_diag("state-loss current mech Vector", loss_state_current, theta0)
  report_diag("state-loss current mech Vector (out-of-place)", loss_state_current_oop, theta0)
  report_diag("x2dot-loss current (no dropgrad)", θ -> loss_x2dot_current(θ; drop_uhat=false), theta0)
  report_diag("x2dot-loss current (no dropgrad, out-of-place)", θ -> loss_x2dot_current_oop(θ; drop_uhat=false), theta0)
  report_diag_vec2_fwd("state-loss raw-only current OOP + QuadratureAdjoint(ReverseDiffVJP)", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("state-loss raw-only current OOP + InterpolatingAdjoint(ZygoteVJP)", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
  report_diag_vec2_fwd("state-loss raw-only current OOP + ForwardSensitivity()", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=ForwardSensitivity()), copy(theta0.mech_raw))
  report_diag_vec2_fwd("state-loss raw-only current OOP + ForwardDiffSensitivity()", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=ForwardDiffSensitivity()), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current OOP (no dropgrad) + QuadratureAdjoint(ReverseDiffVJP)", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current OOP (no dropgrad) + InterpolatingAdjoint(ZygoteVJP)", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=InterpolatingAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current OOP (no dropgrad) + ForwardSensitivity()", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=ForwardSensitivity()), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current OOP (no dropgrad) + ForwardDiffSensitivity()", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=ForwardDiffSensitivity()), copy(theta0.mech_raw))
elseif diag_mode == "backend_qzygote"
  report_diag_vec2("state-loss raw-only current + QuadratureAdjoint(ZygoteVJP)", raw -> loss_state_current_raw_only(raw; sensealg_local=QuadratureAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
elseif diag_mode == "backend_interp"
  report_diag_vec2("state-loss raw-only current + InterpolatingAdjoint(ZygoteVJP)", raw -> loss_state_current_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
elseif diag_mode == "oop_zygote"
  report_diag_vec2("state-loss raw-only current OOP + QuadratureAdjoint(ZygoteVJP)", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=QuadratureAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
  report_diag_vec2("x2dot-loss raw-only current OOP (no dropgrad) + QuadratureAdjoint(ZygoteVJP)", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=QuadratureAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
  report_diag_vec2("state-loss raw-only current OOP + InterpolatingAdjoint(ZygoteVJP)", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
  report_diag_vec2("x2dot-loss raw-only current OOP (no dropgrad) + InterpolatingAdjoint(ZygoteVJP)", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=InterpolatingAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
elseif diag_mode == "oop_qzygote"
  report_diag_vec2("state-loss raw-only current OOP + QuadratureAdjoint(ZygoteVJP)", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=QuadratureAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
  report_diag_vec2("x2dot-loss raw-only current OOP (no dropgrad) + QuadratureAdjoint(ZygoteVJP)", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=QuadratureAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
elseif diag_mode == "oop_interpzygote"
  report_diag_vec2("state-loss raw-only current OOP + InterpolatingAdjoint(ZygoteVJP)", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
  report_diag_vec2("x2dot-loss raw-only current OOP (no dropgrad) + InterpolatingAdjoint(ZygoteVJP)", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=InterpolatingAdjoint(autojacvec=ZygoteVJP())), copy(theta0.mech_raw))
elseif diag_mode == "oop_forward"
  report_diag_vec2_fwd("state-loss raw-only current OOP + ForwardSensitivity()", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=ForwardSensitivity()), copy(theta0.mech_raw))
  report_diag_vec2_fwd("state-loss raw-only current OOP + ForwardDiffSensitivity()", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=ForwardDiffSensitivity()), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current OOP (no dropgrad) + ForwardSensitivity()", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=ForwardSensitivity()), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current OOP (no dropgrad) + ForwardDiffSensitivity()", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=ForwardDiffSensitivity()), copy(theta0.mech_raw))
elseif diag_mode == "inplace_forward"
  report_diag_vec2_fwd("state-loss raw-only current inplace + ForwardSensitivity()", raw -> loss_state_current_raw_only(raw; sensealg_local=ForwardSensitivity()), copy(theta0.mech_raw))
  report_diag_vec2_fwd("state-loss raw-only current inplace + ForwardDiffSensitivity()", raw -> loss_state_current_raw_only(raw; sensealg_local=ForwardDiffSensitivity()), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current inplace (no dropgrad) + ForwardSensitivity()", raw -> loss_x2dot_current_raw_only(raw; drop_uhat=false, sensealg_local=ForwardSensitivity()), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current inplace (no dropgrad) + ForwardDiffSensitivity()", raw -> loss_x2dot_current_raw_only(raw; drop_uhat=false, sensealg_local=ForwardDiffSensitivity()), copy(theta0.mech_raw))
elseif diag_mode == "inplace_fdiff"
  report_diag_vec2_fwd("state-loss raw-only current inplace + ForwardDiffSensitivity()", raw -> loss_state_current_raw_only(raw; sensealg_local=ForwardDiffSensitivity()), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current inplace (no dropgrad) + ForwardDiffSensitivity()", raw -> loss_x2dot_current_raw_only(raw; drop_uhat=false, sensealg_local=ForwardDiffSensitivity()), copy(theta0.mech_raw))
elseif diag_mode == "adjoint_compare"
  report_diag_vec2_fwd("state-loss raw-only current inplace + QuadratureAdjoint(ReverseDiffVJP(true))", raw -> loss_state_current_raw_only(raw; sensealg_local=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("state-loss raw-only current inplace + QuadratureAdjoint(ReverseDiffVJP(false))", raw -> loss_state_current_raw_only(raw; sensealg_local=QuadratureAdjoint(autojacvec=ReverseDiffVJP(false))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("state-loss raw-only current inplace + InterpolatingAdjoint(ReverseDiffVJP(true))", raw -> loss_state_current_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("state-loss raw-only current inplace + InterpolatingAdjoint(ReverseDiffVJP(false))", raw -> loss_state_current_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current inplace (no dropgrad) + QuadratureAdjoint(ReverseDiffVJP(true))", raw -> loss_x2dot_current_raw_only(raw; drop_uhat=false, sensealg_local=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current inplace (no dropgrad) + QuadratureAdjoint(ReverseDiffVJP(false))", raw -> loss_x2dot_current_raw_only(raw; drop_uhat=false, sensealg_local=QuadratureAdjoint(autojacvec=ReverseDiffVJP(false))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current inplace (no dropgrad) + InterpolatingAdjoint(ReverseDiffVJP(true))", raw -> loss_x2dot_current_raw_only(raw; drop_uhat=false, sensealg_local=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current inplace (no dropgrad) + InterpolatingAdjoint(ReverseDiffVJP(false))", raw -> loss_x2dot_current_raw_only(raw; drop_uhat=false, sensealg_local=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))), copy(theta0.mech_raw))
elseif diag_mode == "interp_compare"
  report_diag_vec2_fwd("state-loss raw-only current inplace + InterpolatingAdjoint(ReverseDiffVJP(true))", raw -> loss_state_current_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("state-loss raw-only current inplace + InterpolatingAdjoint(ReverseDiffVJP(false))", raw -> loss_state_current_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current inplace (no dropgrad) + InterpolatingAdjoint(ReverseDiffVJP(true))", raw -> loss_x2dot_current_raw_only(raw; drop_uhat=false, sensealg_local=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("x2dot-loss raw-only current inplace (no dropgrad) + InterpolatingAdjoint(ReverseDiffVJP(false))", raw -> loss_x2dot_current_raw_only(raw; drop_uhat=false, sensealg_local=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))), copy(theta0.mech_raw))
elseif diag_mode == "interp_state"
  report_diag_vec2_fwd("state-loss raw-only current inplace + InterpolatingAdjoint(ReverseDiffVJP(true))", raw -> loss_state_current_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))), copy(theta0.mech_raw))
  report_diag_vec2_fwd("state-loss raw-only current inplace + InterpolatingAdjoint(ReverseDiffVJP(false))", raw -> loss_state_current_raw_only(raw; sensealg_local=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))), copy(theta0.mech_raw))
elseif diag_mode == "pnet_sample"
  report_diag_pnet_samples("active-loss current inplace + QuadratureAdjoint(ReverseDiffVJP(true))", loss_train_active_current, theta0, pnet_sample_idx)
  report_diag_pnet_samples("state-loss current inplace + QuadratureAdjoint(ReverseDiffVJP(true))", loss_state_current, theta0, pnet_sample_idx)
  report_diag_pnet_samples("x2dot-loss current inplace (no dropgrad) + QuadratureAdjoint(ReverseDiffVJP(true))", θ -> loss_x2dot_current(θ; drop_uhat=false), theta0, pnet_sample_idx)
elseif diag_mode == "pnet_sample_oop_qzygote"
  qzygote = QuadratureAdjoint(autojacvec=ZygoteVJP())
  report_diag_pnet_samples("active-loss current OOP + QuadratureAdjoint(ZygoteVJP)", θ -> loss_train_active_current_oop(θ; sensealg_local=qzygote), theta0, pnet_sample_idx)
  report_diag_pnet_samples("state-loss current OOP + QuadratureAdjoint(ZygoteVJP)", θ -> loss_state_current_oop(θ; sensealg_local=qzygote), theta0, pnet_sample_idx)
  report_diag_pnet_samples("x2dot-loss current OOP (no dropgrad) + QuadratureAdjoint(ZygoteVJP)", θ -> loss_x2dot_current_oop(θ; drop_uhat=false, sensealg_local=qzygote), theta0, pnet_sample_idx)
elseif diag_mode == "timing_matrix_oop_qzygote"
  qzygote = QuadratureAdjoint(autojacvec=ZygoteVJP())
  report_timing_triplet("state-loss full theta OOP + QuadratureAdjoint(ZygoteVJP)", θ -> loss_state_current_oop(θ; sensealg_local=qzygote), theta0)
  report_timing_triplet("state-loss pnet-only OOP + QuadratureAdjoint(ZygoteVJP)", p -> loss_state_current_oop_pnet_only(p; sensealg_local=qzygote), copy(theta0.p_net))
  report_timing_triplet("state-loss mech-only OOP + QuadratureAdjoint(ZygoteVJP)", raw -> loss_state_current_oop_raw_only(raw; sensealg_local=qzygote), copy(theta0.mech_raw))
  report_timing_triplet("x2dot-loss full theta OOP + QuadratureAdjoint(ZygoteVJP)", θ -> loss_x2dot_current_oop(θ; drop_uhat=false, sensealg_local=qzygote), theta0)
  report_timing_triplet("x2dot-loss pnet-only OOP + QuadratureAdjoint(ZygoteVJP)", p -> loss_x2dot_current_oop_pnet_only(p; drop_uhat=false, sensealg_local=qzygote), copy(theta0.p_net))
  report_timing_triplet("x2dot-loss mech-only OOP + QuadratureAdjoint(ZygoteVJP)", raw -> loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=qzygote), copy(theta0.mech_raw))
  report_timing_triplet("active-loss full theta OOP + QuadratureAdjoint(ZygoteVJP)", θ -> loss_train_active_current_oop(θ; sensealg_local=qzygote), theta0)
  report_timing_triplet("active-loss pnet-only OOP + QuadratureAdjoint(ZygoteVJP)", p -> loss_train_active_current_oop_pnet_only(p; sensealg_local=qzygote), copy(theta0.p_net))
  report_timing_triplet("active-loss mech-only OOP + QuadratureAdjoint(ZygoteVJP)", raw -> (loss_state_current_oop_raw_only(raw; sensealg_local=qzygote) + loss_x2dot_current_oop_raw_only(raw; drop_uhat=false, sensealg_local=qzygote)), copy(theta0.mech_raw))
elseif diag_mode == "timing_full_oop_qzygote"
  qzygote = QuadratureAdjoint(autojacvec=ZygoteVJP())
  report_timing_triplet("state-loss full theta OOP + QuadratureAdjoint(ZygoteVJP)", θ -> loss_state_current_oop(θ; sensealg_local=qzygote), theta0)
  report_timing_triplet("x2dot-loss full theta OOP + QuadratureAdjoint(ZygoteVJP)", θ -> loss_x2dot_current_oop(θ; drop_uhat=false, sensealg_local=qzygote), theta0)
  report_timing_triplet("active-loss full theta OOP + QuadratureAdjoint(ZygoteVJP)", θ -> loss_train_active_current_oop(θ; sensealg_local=qzygote), theta0)
elseif diag_mode == "formal_vs_debug_oop_qzygote"
  qzygote = QuadratureAdjoint(autojacvec=ZygoteVJP())
  report_timing_triplet("debug-active OOP + QuadratureAdjoint(ZygoteVJP) g_nn=4.89e-5", θ -> loss_train_active_current_oop_gain(θ; sensealg_local=qzygote, nn_gain_local=nn_gain, with_monitor=false), theta0)
  report_timing_triplet("formal-active OOP + QuadratureAdjoint(ZygoteVJP) g_nn=1.0", θ -> loss_train_active_current_oop_gain(θ; sensealg_local=qzygote, nn_gain_local=1.0, with_monitor=false), theta0)
  report_timing_triplet("formal-active+monitor OOP + QuadratureAdjoint(ZygoteVJP) g_nn=1.0", θ -> loss_train_active_current_oop_gain(θ; sensealg_local=qzygote, nn_gain_local=1.0, with_monitor=true), theta0)
  report_timing_triplet("formal-state OOP + QuadratureAdjoint(ZygoteVJP) g_nn=1.0", θ -> loss_state_current_oop_gain(θ; sensealg_local=qzygote, nn_gain_local=1.0, with_monitor=false), theta0)
  report_timing_triplet("formal-x2dot OOP + QuadratureAdjoint(ZygoteVJP) g_nn=1.0", θ -> loss_x2dot_current_oop_gain(θ; drop_uhat=false, sensealg_local=qzygote, nn_gain_local=1.0, with_monitor=false), theta0)
elseif diag_mode == "formal_precheck_profile"
  init_diag_log(; prefix=get(ENV, "HNODECB_DIAG_LOG_PREFIX", "log2_03_step2a_stage2light_precheck_profile"))
  emitln("mode = ", diag_mode)
  emitln("npts = ", npts)
  emitln("candidate rank 6 ks0 = ", @sprintf("%.16e", ks0), " cs0 = ", @sprintf("%.16e", cs0))
  emitln("nn_gain = ", @sprintf("%.16e", nn_gain))
  emitln("pnet_diag_nsamples = ", pnet_diag_nsamples)
  emitln("formal_profile_branch = ", formal_profile_branch)
  emitln("formal_profile_with_monitor = ", formal_profile_with_monitor)
  emitln()
  qzygote = QuadratureAdjoint(autojacvec=ZygoteVJP())
  if formal_profile_branch == "state"
    report_formal_precheck_profile(
      "formal-precheck state OOP + QuadratureAdjoint(ZygoteVJP) g_nn=1.0",
      θ -> loss_state_current_oop_gain(θ; sensealg_local=qzygote, nn_gain_local=1.0, with_monitor=formal_profile_with_monitor),
      theta0
    )
  elseif formal_profile_branch == "x2dot"
    report_formal_precheck_profile(
      "formal-precheck x2dot OOP + QuadratureAdjoint(ZygoteVJP) g_nn=1.0",
      θ -> loss_x2dot_current_oop_gain(θ; drop_uhat=false, sensealg_local=qzygote, nn_gain_local=1.0, with_monitor=formal_profile_with_monitor),
      theta0
    )
  elseif formal_profile_branch == "active"
    report_formal_precheck_profile(
      "formal-precheck active OOP + QuadratureAdjoint(ZygoteVJP) g_nn=1.0",
      θ -> loss_train_active_current_oop_gain(θ; sensealg_local=qzygote, nn_gain_local=1.0, with_monitor=formal_profile_with_monitor),
      theta0
    )
  else
    error("Unknown HNODECB_PRECHECK_PROFILE_BRANCH=$(formal_profile_branch). Expected state, x2dot, or active.")
  end
  close_diag_log()
end
