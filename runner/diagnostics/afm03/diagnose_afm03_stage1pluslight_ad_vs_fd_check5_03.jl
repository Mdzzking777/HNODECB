cd(@__DIR__)

using Serialization
using Printf
using LinearAlgebra
using Statistics
using Zygote

env_int(name, default) = begin
  raw = strip(get(ENV, name, ""))
  parsed = tryparse(Int, raw)
  parsed === nothing ? default : parsed
end

env_float(name, default) = begin
  raw = strip(get(ENV, name, ""))
  parsed = tryparse(Float64, raw)
  parsed === nothing ? default : parsed
end

env_string(name, default) = begin
  raw = strip(get(ENV, name, ""))
  raw == "" ? default : raw
end

function parse_trial_ids(spec::AbstractString)
  ids = Int[]
  for token in split(spec, ",")
    s = strip(token)
    isempty(s) && continue
    parsed = tryparse(Int, s)
    parsed === nothing && error("Invalid trial id: " * s)
    push!(ids, parsed)
  end
  isempty(ids) && error("No trial ids provided.")
  return ids
end

function rel_gap(ad::Float64, fd::Float64)
  abs(ad - fd) / max(abs(fd), 1e-12)
end

function fmt_hp(x)
  isfinite(x) ? @sprintf("%.12e", x) : string(x)
end

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
stage1_dir = normpath(joinpath(repo_root, "step2a_hyperparameter_tuning", "hyperparameter_tuning_first_stage"))
stage1_file = normpath(joinpath(stage1_dir, "afm_param_stage1_03.jl"))
result_file = normpath(joinpath(stage1_dir, "results_afm", "afm_param_stage1pluslight_03.jld"))

ENV["HNODECB_STAGE1_VARIANT"] = "stage1pluslight"
ENV["HNODECB_STAGE1PLUS_MODE"] = "1"
ENV["HNODECB_STAGE1PLUSLIGHT_MODE"] = "1"
ENV["HNODECB_STAGE1_DEFER_MAIN"] = "1"
ENV["HNODECB_SELFTEST"] = "0"
ENV["HNODECB_STAGE1_RESULT_STEM"] = "afm_param_stage1pluslight_03"

include(stage1_file)

trial_ids = parse_trial_ids(env_string("HNODECB_STAGE1PLUS_GRADCHECK_TRIALS", "269,3218,748"))
fd_eps_mech = [1e-6, 1e-5, 1e-4]
fd_eps_dir = [1e-6, 1e-5, 1e-4]
pnet_samples = env_int("HNODECB_STAGE1PLUS_GRADCHECK_PNET_SAMPLES", 4)

result_data = deserialize(result_file)
records = result_data[:trial_parameters]

function find_trial_record(records, trial_id::Int)
  matches = [rec for rec in records if get(rec.params, "trial_id", -1) == trial_id]
  isempty(matches) && error("Trial id not found in result file: " * string(trial_id))
  return matches[1]
end

function build_stage1_case(rec)
  trial_id = rec.params["trial_id"]
  num_hidden_layers = rec.params["num_hidden_layers"]
  num_hidden_nodes = rec.params["num_hidden_nodes"]
  ks0 = rec.params["ks0"]
  cs0 = rec.params["cs0"]

  window_us = env_float("HNODECB_STAGE1PLUS_ARCH_WINDOW_US", 5e-6)
  main_window_idx = stage1plus_first_contact_window_indices(all_times, contact_all, window_us)
  times_window = all_times[main_window_idx]
  ode_window = ode_data_full[:, main_window_idx]
  x2dot_window = x2dot_all[main_window_idx]
  contact_window = contact_all[main_window_idx]

  train_idx, val_idx = make_train_val_masks(length(times_window), val_stride, val_offset)
  ode_train = ode_window[:, train_idx]
  ode_val = ode_window[:, val_idx]
  x2dot_train = x2dot_window[train_idx]
  x2dot_val = x2dot_window[val_idx]
  contact_train = contact_window[train_idx]
  contact_val = contact_window[val_idx]
  times_train = times_window[train_idx]
  times_val = times_window[val_idx]
  x3_t0_val_local = ode_train[3, 1]

  seed = stage1plus_hash_seed(stage1_run_seed, trial_id, 3)
  rng = StableRNG(seed)
  approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)
  p_net_init_raw, st = Lux.setup(rng, approximating_neural_network)
  p_net_init = Flux.f64(p_net_init_raw)
  p_net_vec0, re_pnet = Optimisers.destructure(p_net_init)

  gnn_info = compute_stage1plus_gnn_scale(
    ode_data_full, monitor_idx_full, re_pnet(p_net_vec0),
    approximating_neural_network, st
  )

  raw_init = [
    raw_from_value(ks0, ks_bounds[1], ks_bounds[2]),
    raw_from_value(cs0, cs_bounds[1], cs_bounds[2])
  ]
  theta0 = ComponentVector(p_net=copy(p_net_vec0), mech_raw=raw_init)

  ms_group_size = 10
  ms_continuity_term = 1e-3

  function train_loss_fn(theta)
    inf_context[] = "gradcheck_stage1_train"
    last_inf_reason[] = ""
    loss_single_or_ms(theta, ode_train, x2dot_train, contact_train, times_train,
      state12_scale_full, x2dot_scale_full, x3_scale,
      use_multiple_shooting, ms_group_size, ms_continuity_term,
      approximating_neural_network, st, known_pars, x3_t0_val_local,
      0.0, re_pnet; zero_nn_override=false, nn_gain=gnn_info.g_nn)[1]
  end

  return (
    trial_id=trial_id,
    ks0=ks0,
    cs0=cs0,
    num_hidden_layers=num_hidden_layers,
    num_hidden_nodes=num_hidden_nodes,
    theta0=theta0,
    train_loss_fn=train_loss_fn,
    pnet_len=length(theta0.p_net),
    g_nn=gnn_info.g_nn,
    g_target=gnn_info.target,
    amp_ref=gnn_info.amp_ref,
    train_points=length(times_train),
    val_points=length(times_val),
    timespan=(times_window[1], times_window[end]),
    record=rec
  )
end

function sampled_pnet_indices(pnet_len::Int, sample_count::Int)
  count = clamp(sample_count, 1, pnet_len)
  idx = unique(round.(Int, range(1, pnet_len, length=count)))
  sort!(idx)
  return idx
end

function fd_mech(loss_fn, theta0, idx::Int, eps_rel::Float64)
  base = theta0.mech_raw[idx]
  h = eps_rel * max(1.0, abs(base))
  theta_plus = ComponentVector(p_net=copy(theta0.p_net), mech_raw=copy(theta0.mech_raw))
  theta_minus = ComponentVector(p_net=copy(theta0.p_net), mech_raw=copy(theta0.mech_raw))
  theta_plus.mech_raw[idx] += h
  theta_minus.mech_raw[idx] -= h
  l_plus = loss_fn(theta_plus)
  l_minus = loss_fn(theta_minus)
  return (l_plus - l_minus) / (2h), h, l_plus, l_minus
end

function fd_pnet_entry(loss_fn, theta0, idx::Int, eps_rel::Float64)
  base = theta0.p_net[idx]
  h = eps_rel * max(1.0, abs(base))
  theta_plus = ComponentVector(p_net=copy(theta0.p_net), mech_raw=copy(theta0.mech_raw))
  theta_minus = ComponentVector(p_net=copy(theta0.p_net), mech_raw=copy(theta0.mech_raw))
  theta_plus.p_net[idx] += h
  theta_minus.p_net[idx] -= h
  l_plus = loss_fn(theta_plus)
  l_minus = loss_fn(theta_minus)
  return (l_plus - l_minus) / (2h), h, l_plus, l_minus
end

function fd_directional(loss_fn, theta0, dir_p::AbstractVector{<:Real}, hs::Vector{Float64})
  out = NamedTuple[]
  for h in hs
    theta_plus = ComponentVector(p_net=copy(theta0.p_net .+ h .* dir_p), mech_raw=copy(theta0.mech_raw))
    theta_minus = ComponentVector(p_net=copy(theta0.p_net .- h .* dir_p), mech_raw=copy(theta0.mech_raw))
    l_plus = loss_fn(theta_plus)
    l_minus = loss_fn(theta_minus)
    push!(out, (h=h, loss_plus=l_plus, loss_minus=l_minus, dir=(l_plus - l_minus) / (2h)))
  end
  return out
end

function appendln!(lines, s="")
  push!(lines, s)
end

lines = String[]
appendln!(lines, "AFM03 Check-5: stage1pluslight AD-vs-FD on joint-trial train loss")
appendln!(lines, "Date: " * string(Dates.now()))
appendln!(lines, "Input: afm_param_stage1pluslight_03.jld")
appendln!(lines, "Config: first-contact + 5us window, val_stride=5, val_offset=2, fixed ms_group_size=10, fixed ms_continuity_term=1e-3")
appendln!(lines, "Route: Rosenbrock23(ad=false), QuadratureAdjoint(ReverseDiffVJP(true))")
appendln!(lines)

for trial_id in trial_ids
  rec = find_trial_record(records, trial_id)
  case = build_stage1_case(rec)
  loss_fn = case.train_loss_fn

  GC.gc()
  ad_loss, back = Zygote.pullback(loss_fn, case.theta0)
  ad_grad = first(back(1.0))
  p_grad_norm = sqrt(sum(abs2, ad_grad.p_net))
  m_grad_norm = sqrt(sum(abs2, ad_grad.mech_raw))

  appendln!(lines, "Trial $(case.trial_id)")
  appendln!(lines, "  arch: layers=$(case.num_hidden_layers) nodes=$(case.num_hidden_nodes) | p_count=$(case.pnet_len)")
  appendln!(lines, "  init: ks0=$(fmt_e(case.ks0, sigdigits=4)) cs0=$(fmt_e(case.cs0, sigdigits=4)) g_nn=$(fmt_e(case.g_nn, sigdigits=4))")
  appendln!(lines, "  train loss(theta0)=$(fmt_e(ad_loss, sigdigits=6)) | ||grad_p_net||=$(fmt_e(p_grad_norm, sigdigits=6)) ||grad_mech_raw||=$(fmt_e(m_grad_norm, sigdigits=6))")
  appendln!(lines, "  train points=$(case.train_points) val points=$(case.val_points) | window tspan=[$(fmt_e(case.timespan[1], sigdigits=6)), $(fmt_e(case.timespan[2], sigdigits=6))]")

  if isfinite(p_grad_norm) && p_grad_norm > 0.0
    dir_p = ad_grad.p_net ./ p_grad_norm
    ad_dir = sum(ad_grad.p_net .* dir_p)
    appendln!(lines, "  AD directional derivative along +grad_p/||grad_p|| = $(fmt_hp(ad_dir))")
    for fd in fd_directional(loss_fn, case.theta0, dir_p, fd_eps_dir)
      appendln!(lines,
        "    FD h=" * @sprintf("%.2e", fd.h) *
        " : l+= " * fmt_e(fd.loss_plus, sigdigits=6) *
        " l-= " * fmt_e(fd.loss_minus, sigdigits=6) *
        " dir=" * fmt_hp(fd.dir) *
        " rel_gap=" * fmt_e(rel_gap(ad_dir, fd.dir), sigdigits=4))
    end
  else
    appendln!(lines, "  AD directional derivative along +grad_p/||grad_p|| = unavailable (zero/NaN p-net grad)")
  end

  for i in eachindex(case.theta0.mech_raw)
    appendln!(lines, "  mech_raw[$i]: AD=$(fmt_hp(ad_grad.mech_raw[i]))")
    for eps_rel in fd_eps_mech
      fd_val, h, l_plus, l_minus = fd_mech(loss_fn, case.theta0, i, eps_rel)
      appendln!(lines,
        "    FD eps=" * @sprintf("%.2e", eps_rel) *
        " h=" * fmt_hp(h) *
        " : l+= " * fmt_e(l_plus, sigdigits=6) *
        " l-= " * fmt_e(l_minus, sigdigits=6) *
        " dir=" * fmt_hp(fd_val) *
        " rel_gap=" * fmt_e(rel_gap(ad_grad.mech_raw[i], fd_val), sigdigits=4))
    end
  end

  for idx in sampled_pnet_indices(case.pnet_len, pnet_samples)
    appendln!(lines, "  p_net[$idx]: AD=$(fmt_hp(ad_grad.p_net[idx]))")
    for eps_rel in fd_eps_mech
      fd_val, h, l_plus, l_minus = fd_pnet_entry(loss_fn, case.theta0, idx, eps_rel)
      appendln!(lines,
        "    FD eps=" * @sprintf("%.2e", eps_rel) *
        " h=" * fmt_hp(h) *
        " : l+= " * fmt_e(l_plus, sigdigits=6) *
        " l-= " * fmt_e(l_minus, sigdigits=6) *
        " dir=" * fmt_hp(fd_val) *
        " rel_gap=" * fmt_e(rel_gap(ad_grad.p_net[idx], fd_val), sigdigits=4))
    end
  end

  appendln!(lines)
end

out_dir = normpath(joinpath(repo_root, "logs", "stage1_step2a", "1pluslight", "local", "diagnostics"))
mkpath(out_dir)
out_file = joinpath(out_dir, "afm03_check5_stage1pluslight_ad_vs_fd_train_loss.txt")
open(out_file, "w") do io
  for line in lines
    println(io, line)
  end
end

for line in lines
  println(line)
end
println()
println("Saved diagnostics -> ", out_file)
