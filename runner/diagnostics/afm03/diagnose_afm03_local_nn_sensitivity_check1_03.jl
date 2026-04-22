cd(@__DIR__)

using Serialization, Statistics, Printf, Random, DataFrames, LinearAlgebra
using Zygote
using Optimisers
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

trial_val(rec) = hasproperty(rec, :val_loss) ? rec.val_loss : (hasproperty(rec, :loss) ? rec.loss : Inf)

function safe_quantile(v::AbstractVector{<:Real}, q::Real)
  isempty(v) && return NaN
  return quantile(collect(v), q)
end

function summarize_vec(v::AbstractVector{<:Real})
  if isempty(v)
    return (min=NaN, p50=NaN, p95=NaN, mean=NaN, max=NaN)
  end
  vv = collect(v)
  return (
    min=minimum(vv),
    p50=safe_quantile(vv, 0.50),
    p95=safe_quantile(vv, 0.95),
    mean=mean(vv),
    max=maximum(vv)
  )
end

function fmt_num(x; sigdigits=4)
  return (x isa Number && isfinite(x)) ? @sprintf("%.*e", sigdigits - 1, x) : "NaN"
end

stage1_file = normpath(joinpath(repo_root, "step2a_hyperparameter_tuning", "hyperparameter_tuning_first_stage", "results_afm", "afm_param_stage1pluslight_03.jld"))
stage1 = deserialize(stage1_file)
trials = haskey(stage1, :trial_parameters) ? stage1.trial_parameters : Any[]
sorted_trials = sort(trials, by=trial_val)
selected_ranks = [1, 8]

windows, solution_post, ode_post = build_window_manifest(repo_root)
contact_post = Vector{Bool}(solution_post.contact .== 1)

lines = String[]
push!(lines, "AFM03 Check-1: local p_net -> F_contact sensitivity on fixed true window states")
push!(lines, "Date: 2026-03-25")
push!(lines, "Input: afm_param_stage1pluslight_03.jld")
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
  ks_hat = hasproperty(rec, :ks_hat) ? Float64(rec.ks_hat) : ks0
  cs_hat = hasproperty(rec, :cs_hat) ? Float64(rec.cs_hat) : cs0

  nn = build_nn(num_hidden_layers, num_hidden_nodes)
  p_net_template_raw, st = Lux.setup(StableRNG(0), nn)
  p_net_template = Flux.f64(p_net_template_raw)
  p_vec_template, re_pnet = Optimisers.destructure(p_net_template)
  length(p_vec_template) == length(p_net_vec) || error("Rank $rank p_net length mismatch")

  push!(lines, "Rank $rank")
  push!(lines, "  arch: layers=$(num_hidden_layers) nodes=$(num_hidden_nodes) | p_count=$(length(p_net_vec))")
  push!(lines, "  mech: ks0=$(fmt_num(ks0)) cs0=$(fmt_num(cs0)) | ks_hat=$(fmt_num(ks_hat)) cs_hat=$(fmt_num(cs_hat))")
  push!(lines, "  g_nn=$(fmt_num(g_nn))")

  for win in windows
    idxs = win.start_idx:win.stop_idx
    u_mat = @view ode_post[:, idxs]
    local_contact_mask = contact_post[idxs]

    raw_vals = Float64[]
    w_vals = Float64[]
    gated_vals = Float64[]
    jraw_norms = Float64[]
    jgated_norms = Float64[]
    jraw_norms_contact = Float64[]
    jgated_norms_contact = Float64[]

    for j in axes(u_mat, 2)
      u = Float64.(u_mat[:, j])
      s = dist + u[1] - u[3]
      w_pred = contact_weight(s, adhesion_transition)
      raw_from_vec = function(v)
        p_struct = re_pnet(v)
        nn_in = u[1:3]
        uhat = nn(nn_in, p_struct, st)[1]
        return uhat[1]
      end
      raw = raw_from_vec(p_net_vec)
      grad_raw = Zygote.gradient(raw_from_vec, p_net_vec)[1]
      jraw = norm(grad_raw)
      jgated = abs(g_nn * w_pred) * jraw
      gated = g_nn * raw * w_pred

      push!(raw_vals, raw)
      push!(w_vals, w_pred)
      push!(gated_vals, gated)
      push!(jraw_norms, jraw)
      push!(jgated_norms, jgated)
      if local_contact_mask[j]
        push!(jraw_norms_contact, jraw)
        push!(jgated_norms_contact, jgated)
      end
    end

    w_stats = summarize_vec(w_vals)
    raw_stats = summarize_vec(raw_vals)
    gated_stats = summarize_vec(gated_vals)
    jraw_stats = summarize_vec(jraw_norms)
    jgated_stats = summarize_vec(jgated_norms)
    jraw_contact_stats = summarize_vec(jraw_norms_contact)
    jgated_contact_stats = summarize_vec(jgated_norms_contact)
    contact_count = count(local_contact_mask)

    push!(lines, "  Window $(win.role) [$(win.start_idx), $(win.stop_idx)] len=$(win.len) contact_count=$(contact_count)")
    push!(lines, "    w_pred     : mean=$(fmt_num(w_stats.mean)) p50=$(fmt_num(w_stats.p50)) p95=$(fmt_num(w_stats.p95)) max=$(fmt_num(w_stats.max))")
    push!(lines, "    raw(nn)    : mean=$(fmt_num(raw_stats.mean)) p50=$(fmt_num(raw_stats.p50)) p95=$(fmt_num(raw_stats.p95)) max=$(fmt_num(raw_stats.max))")
    push!(lines, "    F_contact  : mean=$(fmt_num(gated_stats.mean)) p50=$(fmt_num(gated_stats.p50)) p95=$(fmt_num(gated_stats.p95)) max=$(fmt_num(gated_stats.max))")
    push!(lines, "    ||draw/dp|| all     : mean=$(fmt_num(jraw_stats.mean)) p50=$(fmt_num(jraw_stats.p50)) p95=$(fmt_num(jraw_stats.p95)) max=$(fmt_num(jraw_stats.max))")
    push!(lines, "    ||dF/dp||   all     : mean=$(fmt_num(jgated_stats.mean)) p50=$(fmt_num(jgated_stats.p50)) p95=$(fmt_num(jgated_stats.p95)) max=$(fmt_num(jgated_stats.max))")
    push!(lines, "    ||draw/dp|| contact : mean=$(fmt_num(jraw_contact_stats.mean)) p50=$(fmt_num(jraw_contact_stats.p50)) p95=$(fmt_num(jraw_contact_stats.p95)) max=$(fmt_num(jraw_contact_stats.max))")
    push!(lines, "    ||dF/dp||   contact : mean=$(fmt_num(jgated_contact_stats.mean)) p50=$(fmt_num(jgated_contact_stats.p50)) p95=$(fmt_num(jgated_contact_stats.p95)) max=$(fmt_num(jgated_contact_stats.max))")
  end
  push!(lines, "")
end

out_dir = normpath(joinpath(repo_root, "logs", "stage2_step2a", "local", "windowed_dualrank", "diagnostics"))
isdir(out_dir) || mkpath(out_dir)
out_file = joinpath(out_dir, "afm03_check1_local_nn_sensitivity_true_states.txt")
write(out_file, join(lines, "\n") * "\n")

println(join(lines, "\n"))
println()
println("Saved -> ", out_file)
