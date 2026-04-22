cd(@__DIR__)

using Serialization, Statistics, Printf, Random, DataFrames, LinearAlgebra
using DifferentialEquations, ComponentArrays, SciMLSensitivity
using Zygote
using Flux
using StableRNGs
using Lux

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))

include(normpath(joinpath(repo_root, "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_functions.jl")))
include(normpath(joinpath(repo_root, "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_settings.jl")))

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

function build_nn(num_hidden_layers::Int, num_hidden_nodes::Int)
  hidden = 2^num_hidden_nodes
  layers = Any[]
  push!(layers, Lux.Dense(3, hidden, gelu; init_weight=my_glorot_uniform, use_bias=false))
  for _ in 1:num_hidden_layers
    push!(layers, Lux.Dense(hidden, hidden, gelu; init_weight=my_glorot_uniform, use_bias=false))
  end
  push!(layers, Lux.Dense(hidden, 1; init_weight=my_glorot_uniform, use_bias=false))
  return Lux.Chain(layers...)
end

peak_to_peak(x) = maximum(x) - minimum(x)

function contact_onsets(contact_mask::AbstractVector{Bool})
  idx = Int[]
  prev = false
  for (i, c) in pairs(contact_mask)
    if c && !prev
      push!(idx, i)
    end
    prev = c
  end
  return idx
end

function nearest_time_index(t_us::AbstractVector{<:Real}, target_us::Real)
  return argmin(abs.(t_us .- target_us))
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
  n = length(t_us)
  len = stop_idx - start_idx + 1
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

function apply_window_time_overrides(selected::Vector{<:NamedTuple}, t_post::AbstractVector{<:Real})
  overrides = Dict(
    "first_contact" => (anchor=:start, target_us=1034.0),
    "max_x1_pp_change" => (anchor=:stop, target_us=1057.0),
    "tail_stable" => (anchor=:stop, target_us=1998.0),
  )

  adjusted = NamedTuple[]
  for win in selected
    if haskey(overrides, win.role)
      spec = overrides[win.role]
      new_start, new_stop = relocate_window_with_anchor(t_post .* 1e6, win.start_idx, win.stop_idx; anchor=spec.anchor, target_us=spec.target_us)
      if win.role == "first_contact" && hasproperty(win, :contact_post)
        new_stop = extend_stop_to_contact_end(win.contact_post, new_stop)
      end
      push!(adjusted, merge(win, (
        start_idx=new_start,
        stop_idx=new_stop,
        len=new_stop - new_start + 1,
        t_start=t_post[new_start],
        t_stop=t_post[new_stop],
      )))
    else
      push!(adjusted, win)
    end
  end
  return adjusted
end

function build_window_manifest(repo_root::String)
  pert_file = normpath(joinpath(repo_root, "datasets", "e0.0", "data", "pert_df_afm_dmt_kv.jld"))
  ode_file = normpath(joinpath(repo_root, "datasets", "e0.0", "data", "ode_data_afm_dmt_kv.jld"))
  solution_dataframe_full = deserialize(pert_file)
  ode_data_full = deserialize(ode_file)
  contact_full = Vector{Bool}(solution_dataframe_full.contact .== 1)
  first_contact_idx = findfirst(contact_full)
  first_contact_idx === nothing && error("No contact point found in the AFM dataset.")

  solution_post = solution_dataframe_full[first_contact_idx:end, :]
  ode_post = Float64.(ode_data_full[:, first_contact_idx:end])
  t_post = Float64.(solution_post.t)
  contact_post = Vector{Bool}(solution_post.contact .== 1)
  x1_post = vec(Float64.(ode_post[1, :]))
  onset_post = contact_onsets(contact_post)
  length(onset_post) >= 3 || error("Need at least 3 contact onsets after first contact.")

  candidates = NamedTuple[]
  cycle_pp = Float64[]
  for k in 1:(length(onset_post) - 1)
    lo = onset_post[k]
    hi = onset_post[k + 1] - 1
    push!(cycle_pp, peak_to_peak(@view x1_post[lo:hi]))
  end
  for k in 1:(length(onset_post) - 2)
    start_idx = onset_post[k]
    stop_idx = onset_post[k + 2] - 1
    pp1 = cycle_pp[k]
    pp2 = cycle_pp[k + 1]
    push!(candidates, (
      candidate_index=k,
      start_idx=start_idx,
      stop_idx=stop_idx,
      len=stop_idx - start_idx + 1,
      t_start=t_post[start_idx],
      t_stop=t_post[stop_idx],
      cycle1_index=k,
      cycle2_index=k + 1,
      x1_pp_cycle1=pp1,
      x1_pp_cycle2=pp2,
      x1_pp_delta=abs(pp2 - pp1)
    ))
  end

  first_idx = 1
  last_idx = length(candidates)
  max_change_idx = findmax([cand.x1_pp_delta for cand in candidates])[2]

  selected = NamedTuple[]
  push!(selected, merge(candidates[first_idx], (role="first_contact", label="first_contact_window", contact_post=contact_post)))
  push!(selected, merge(candidates[max_change_idx], (role="max_x1_pp_change", label="max_x1_pp_change_window")))
  push!(selected, merge(candidates[last_idx], (role="tail_stable", label="tail_stable_window")))

  return apply_window_time_overrides(selected, t_post), solution_post, ode_post
end

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

function make_train_val_masks(n, val_stride, val_offset)
  val_idx = [i for i in 1:n if (i - val_offset) % val_stride == 0]
  train_idx = [i for i in 1:n if !(i in val_idx)]
  return sort(train_idx), sort(val_idx)
end

function nn_input_from_state(u)
  return u[1:3]
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

    Fad_eff = Fad * w_pred
    F_contact = if appr === nothing
      (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad_eff
    else
      nn_in = nn_input_from_state(u)
      uhat = appr(nn_in, p.p_net, st)[1]
      nn_gain * uhat[1] * w_pred
    end

    du1 = u[2]
    du2 = (Fd * cos(wd * t) - k * u[1] - c * u[2] - F_contact) / m
    du3 = (-F_contact - ks * u[3]) / cs
    return [du1, du2, du3]
  end
  return f
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
  Fad_eff = Fad .* w_pred
  F_contact = if appr === nothing
    delta = softplus.(-s, adhesion_transition)
    delta = ifelse.(delta .> 0.0, delta, 0.0)
    (4.0 / 3.0) .* Estar .* sqrt(R) .* (delta .^ 1.5) .- Fad_eff
  else
    nn_in = @views uhat[1:3, contact_idx]
    nn_out = appr(nn_in, p_net, st)[1]
    nn_gain .* vec(nn_out) .* w_pred
  end
  return (Fd .* cos.(wd .* t_contact) .- k .* x1 .- c .* x2 .- F_contact) ./ m
end

function state_scale_pair(state12_scale)
  x1_scale = state12_scale[1]
  x2_scale = state12_scale[2]
  return x1_scale, x2_scale
end

function state_component_errors(x1_true, x1_pred, x2_true, x2_pred, x1_scale::Float64, x2_scale::Float64)
  x1_err = vec(abs2.((x1_true .- x1_pred) ./ x1_scale))
  x2_err = vec(abs2.((x2_true .- x2_pred) ./ x2_scale))
  return x1_err, x2_err
end

function loss_single_ss_nomonitor(θ, ode_data, x2dot_data, contact_mask, times,
  state12_scale, x2dot_scale, appr, st, known_pars, x3_t0_val,
  l2_weight, re_pnet, ks_bounds, cs_bounds;
  nn_gain::Float64=1.0, x3_range_weight_local::Float64=0.0)

  mech = [
    bound_param(θ.mech_raw[1], ks_bounds[1], ks_bounds[2]),
    bound_param(θ.mech_raw[2], cs_bounds[1], cs_bounds[2])
  ]
  p_net_struct = re_pnet(θ.p_net)
  p = ComponentVector(p_net=p_net_struct, mech=mech)
  x1_state_scale, x2_state_scale = state_scale_pair(state12_scale)

  weights_val = ifelse.(contact_mask, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_val)

  u0 = [ode_data[1, 1], ode_data[2, 1], x3_t0_val]
  prob = ODEProblem{false}(make_uode_func_oop(appr, st, known_pars; nn_gain=nn_gain), u0, (times[1], times[end]), p)
  sol = solve(prob, Rosenbrock23(autodiff=false); saveat=times, abstol=1e-8, reltol=1e-8,
    sensealg=GaussAdjoint(autojacvec=ZygoteVJP()), maxiters=1_000_000)
  string(sol.retcode) == "Success" || return Inf
  size(sol, 2) == length(times) || return Inf
  uhat = Array(sol)

  x1_state_err, x2_state_err = state_component_errors(
    ode_data[1, :], uhat[1, :], ode_data[2, :], uhat[2, :], x1_state_scale, x2_state_scale
  )
  total_state = sum(weights_val .* x1_state_err) / weights_sum + sum(weights_val .* x2_state_err) / weights_sum

  total_x2dot = 0.0
  contact_idx = findall(contact_mask)
  if !isempty(contact_idx)
    x2dot_pred_contact = x2dot_rhs_batch(uhat, contact_idx, mech, p_net_struct, appr, st, known_pars, times; nn_gain=nn_gain)
    x2_err = abs2.((x2dot_data[contact_idx] .- x2dot_pred_contact) ./ x2dot_scale)
    x2_w = weights_val[contact_idx]
    total_x2dot = sum(x2_w .* x2_err) / sum(x2_w)
  end

  total_x3range = 0.0
  if x3_range_weight_local != 0.0
    x3_scale = max(x3_range_amp, 1e-9)
    exceed = abs.(uhat[3, :]) .- x3_range_amp
    range_pen = abs2.(softplus.(exceed, x3_range_eps) ./ x3_scale)
    total_x3range = x3_range_weight_local * (sum(weights_val .* range_pen) / weights_sum)
  end

  l2_penalty = l2_weight * sum(abs2, θ.p_net)
  total = total_state + total_x2dot + total_x3range + l2_penalty
  return isfinite(total) ? total : Inf
end

trial_val(rec) = hasproperty(rec, :val_loss) ? rec.val_loss : (hasproperty(rec, :loss) ? rec.loss : Inf)
fmt_num(x; sigdigits=4) = (x isa Number && isfinite(x)) ? @sprintf("%.*e", sigdigits - 1, x) : "NaN"

stage1_file = normpath(joinpath(repo_root, "step2a_hyperparameter_tuning", "hyperparameter_tuning_first_stage", "results_afm", "afm_param_stage1pluslight_03.jld"))
stage1 = deserialize(stage1_file)
trials = haskey(stage1, :trial_parameters) ? stage1.trial_parameters : Any[]
sorted_trials = sort(trials, by=trial_val)
selected_ranks = [1, 8]

windows, solution_post, ode_post = build_window_manifest(repo_root)

val_stride = haskey(stage1, :val_stride) ? stage1.val_stride : 5
val_offset = haskey(stage1, :val_offset) ? stage1.val_offset : 2
ks_bounds = stage1.bounds.ks
cs_bounds = stage1.bounds.cs
known_pars = original_parameters[1:8]
scale_eps = 1e-9
l2_weight = 0.0
x3_range_weight_local = 0.0
contact_loss_weight = 4.27
noncontact_loss_weight = 1.0

lines = String[]
push!(lines, "AFM03 Check-2: p_net AD-vs-FD directional derivative on stage2 train loss")
push!(lines, "Date: 2026-03-25")
push!(lines, "Input: afm_param_stage1pluslight_03.jld")
push!(lines, "Config: single-shooting train loss, x3_range_weight=0, l2=0, Rosenbrock23(ad=false), GaussAdjoint(ZygoteVJP())")
push!(lines, "")

for rank in selected_ranks
  rec = sorted_trials[rank]
  params = rec.params
  num_hidden_layers = Int(get(params, "num_hidden_layers", 0))
  num_hidden_nodes = Int(get(params, "num_hidden_nodes", 2))
  g_nn = hasproperty(rec, :g_nn) ? Float64(rec.g_nn) :
    (haskey(params, "g_nn") ? Float64(params["g_nn"]) : 1.0)
  p_net_vec = hasproperty(rec, :p_net_vec) ? Vector{Float64}(rec.p_net_vec) : Vector{Float64}(rec.p_net)
  ks0 = Float64(get(params, "ks0", NaN))
  cs0 = Float64(get(params, "cs0", NaN))

  nn = build_nn(num_hidden_layers, num_hidden_nodes)
  p_net_init_raw, st = Lux.setup(StableRNG(0), nn)
  p_net_init = Flux.f64(p_net_init_raw)
  p_net_template_vec, re_pnet = Flux.destructure(p_net_init)
  length(p_net_template_vec) == length(p_net_vec) || error("Rank $rank p_net length mismatch")

  raw_init = [
    raw_from_value(ks0, ks_bounds[1], ks_bounds[2]),
    raw_from_value(cs0, cs_bounds[1], cs_bounds[2])
  ]
  θ0 = ComponentVector(p_net=copy(p_net_vec), mech_raw=raw_init)

  push!(lines, "Rank $rank")
  push!(lines, "  arch: layers=$(num_hidden_layers) nodes=$(num_hidden_nodes) | p_count=$(length(p_net_vec))")
  push!(lines, "  init: ks0=$(fmt_num(ks0)) cs0=$(fmt_num(cs0)) g_nn=$(fmt_num(g_nn))")

  for win in windows
    idxs = win.start_idx:win.stop_idx
    ode_window = ode_post[:, idxs]
    x2dot_window = Float64.(solution_post.x2dot[idxs])
    contact_window = Vector{Bool}(solution_post.contact[idxs] .== 1)
    times_window = Float64.(solution_post.t[idxs])

    state12_scale_full = vec(maximum(ode_window[1:2, :], dims=2) - minimum(ode_window[1:2, :], dims=2))
    state12_scale_full = max.(state12_scale_full, scale_eps)
    x2dot_scale_full = max(maximum(x2dot_window) - minimum(x2dot_window), scale_eps)

    train_idx, val_idx = make_train_val_masks(length(times_window), val_stride, val_offset)
    ode_train = ode_window[:, train_idx]
    times_train = times_window[train_idx]
    x2dot_train = x2dot_window[train_idx]
    contact_train = contact_window[train_idx]
    x3_t0_val = ode_train[3, 1]

    function loss_fn(theta)
      loss_single_ss_nomonitor(theta, ode_train, x2dot_train, contact_train, times_train,
        state12_scale_full, x2dot_scale_full, nn, st, known_pars, x3_t0_val,
        l2_weight, re_pnet, ks_bounds, cs_bounds; nn_gain=g_nn, x3_range_weight_local=x3_range_weight_local)
    end

    loss0 = loss_fn(θ0)
    grad = Zygote.gradient(loss_fn, θ0)[1]
    gp = Vector{Float64}(grad.p_net)
    gm = Vector{Float64}(grad.mech_raw)
    gp_norm = norm(gp)
    gm_norm = norm(gm)

    push!(lines, "  Window $(win.role) [$(win.start_idx), $(win.stop_idx)] len=$(win.len)")
    push!(lines, "    train_loss(theta0)=$(fmt_num(loss0)) | ||grad_p_net||=$(fmt_num(gp_norm)) ||grad_mech_raw||=$(fmt_num(gm_norm))")

    if !isfinite(gp_norm) || gp_norm == 0.0
      push!(lines, "    p_net gradient is zero/nonfinite; skip FD check")
      continue
    end

    v = gp ./ gp_norm
    ad_dir = dot(gp, v)
    push!(lines, "    AD directional derivative along +grad_p/||grad_p|| = $(fmt_num(ad_dir))")

    for h in (1e-6, 1e-5, 1e-4)
      θplus = ComponentVector(p_net=θ0.p_net .+ h .* v, mech_raw=θ0.mech_raw)
      θminus = ComponentVector(p_net=θ0.p_net .- h .* v, mech_raw=θ0.mech_raw)
      lplus = loss_fn(θplus)
      lminus = loss_fn(θminus)
      fd = (lplus - lminus) / (2h)
      rel_gap = (isfinite(fd) && abs(fd) > 0.0) ? abs(fd - ad_dir) / abs(fd) : NaN
      push!(lines, "    FD h=$(fmt_num(h, sigdigits=3)) : l+= $(fmt_num(lplus)) l-= $(fmt_num(lminus)) dir=$(fmt_num(fd)) rel_gap=$(fmt_num(rel_gap))")
    end
  end
  push!(lines, "")
end

out_dir = normpath(joinpath(repo_root, "logs", "stage2_step2a", "local", "windowed_dualrank", "diagnostics"))
isdir(out_dir) || mkpath(out_dir)
out_file = joinpath(out_dir, "afm03_check2_pnet_ad_vs_fd_train_loss.txt")
write(out_file, join(lines, "\n") * "\n")

println(join(lines, "\n"))
println()
println("Saved -> ", out_file)
