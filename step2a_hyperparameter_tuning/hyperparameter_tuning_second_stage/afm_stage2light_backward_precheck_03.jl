#=
Formal backward precheck for AFM Stage2light (scenario 03).

This script is intentionally strict:
- it reuses the official Stage2 loss path
- it does not monkey-patch RHS / x2dot helpers
- it performs one exact loss -> pullback -> backward call
- optional warmup uses the same exact path

Key env vars:
- HNODECB_STAGE2_DIAG_CANDIDATE
- HNODECB_STAGE2_DIAG_WARM_RANK
- HNODECB_STAGE2_DIAG_NPTS
- HNODECB_STAGE2_DIAG_COMPILE_NPTS
- HNODECB_STAGE2_DIAG_SENSEALG      (gausszygote only)
- HNODECB_STAGE2_DIAG_PROFILE       (0/1)
=#

cd(@__DIR__)

using Profile
using Printf

env_bool(name, default) = lowercase(strip(get(ENV, name, default ? "1" : "0"))) in ("1", "true", "yes", "on")

function env_int(name, default)
  raw = strip(get(ENV, name, ""))
  parsed = tryparse(Int, raw)
  return parsed === nothing ? default : parsed
end

function env_string(name, default)
  raw = strip(get(ENV, name, ""))
  return raw == "" ? default : raw
end

function default_candidate()
  raw = strip(get(ENV, "HNODECB_STAGE2LIGHT_CANDIDATE", "6"))
  parsed = tryparse(Int, raw)
  return parsed === nothing || parsed < 1 ? 6 : parsed
end

if !haskey(ENV, "HNODECB_SELFTEST")
  ENV["HNODECB_SELFTEST"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_INPUT_TOPK")
  ENV["HNODECB_STAGE2_INPUT_TOPK"] = "9"
end
if !haskey(ENV, "HNODECB_STAGE2_NN_WARM_ENABLED")
  ENV["HNODECB_STAGE2_NN_WARM_ENABLED"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_NN_WARM_INPUT_BASENAME")
  ENV["HNODECB_STAGE2_NN_WARM_INPUT_BASENAME"] = "afm_param_stage1pluslight_03.jld"
end

include("afm_param_stage2_03.jl")

function resolve_sensealg(spec::AbstractString)
  mode = lowercase(strip(spec))
  if mode in ("gausszygote", "gauss")
    return GaussAdjoint(autojacvec=ZygoteVJP()), "GaussAdjoint(ZygoteVJP())"
  end
  error("Unsupported HNODECB_STAGE2_DIAG_SENSEALG: " * spec * " (only gausszygote is kept)")
end

function select_diag_indices(n::Int, npts::Int)
  if npts <= 0 || npts >= n
    return collect(1:n)
  end
  idx = unique(round.(Int, range(1, n, length=npts)))
  sort!(idx)
  if first(idx) != 1
    pushfirst!(idx, 1)
  end
  if last(idx) != n
    push!(idx, n)
  end
  return unique(idx)
end

function build_diag_case(candidate_idx::Int, warm_rank::Int, npts::Int)
  1 <= candidate_idx <= length(top_candidates) || error("candidate_idx out of range 1..$(length(top_candidates))")
  candidate = top_candidates[candidate_idx]
  params = candidate.params
  ks0 = params["ks0"]
  cs0 = params["cs0"]
  ms_group_size = get(params, "ms_group_size", 50)
  ms_continuity_term = get(params, "ms_continuity_term", 1e-3)

  approximating_neural_network = build_nn(nn_fixed_num_hidden_layers, nn_fixed_num_hidden_nodes)
  p_net_init_raw, st = Lux.setup(StableRNG(0), approximating_neural_network)
  p_net_init = Flux.f64(p_net_init_raw)
  p_net_template_vec, re_pnet = Optimisers.destructure(p_net_init)

  warm_rank_used = 0
  warm_source = "random_template"
  p_net_vec = copy(p_net_template_vec)

  if stage2_nn_warm_enabled && haskey(stage2_nn_warm_by_base_rank, candidate_idx)
    warm_recs = stage2_nn_warm_by_base_rank[candidate_idx]
    if !isempty(warm_recs)
      warm_rank_used = min(max(warm_rank, 1), length(warm_recs))
      warm_pick = warm_recs[warm_rank_used]
      length(warm_pick.p_net_vec) == length(p_net_template_vec) ||
        error("Warm-start p_net_vec length mismatch: warm=$(length(warm_pick.p_net_vec)) template=$(length(p_net_template_vec))")
      p_net_vec .= warm_pick.p_net_vec
      warm_source = "stage1pluslight_rank_" * string(warm_rank_used)
    end
  end

  raw_init = [
    raw_from_value(ks0, ks_bounds[1], ks_bounds[2]),
    raw_from_value(cs0, cs_bounds[1], cs_bounds[2])
  ]
  theta0 = ComponentVector(p_net=p_net_vec, mech_raw=raw_init)

  idx = select_diag_indices(length(times_train), npts)
  ode_used = ode_train[:, idx]
  x2dot_used = x2dot_train[idx]
  contact_used = contact_train[idx]
  times_used = times_train[idx]

  loss_fn = theta -> loss_single_or_ms(theta, ode_used, x2dot_used, contact_used, times_used,
    state12_scale_full, x2dot_scale_full, x3_scale,
    use_multiple_shooting, ms_group_size, ms_continuity_term,
    approximating_neural_network, st, known_pars, ode_used[3, 1],
    0.0, re_pnet, nothing)

  return (
    theta0=theta0,
    loss_fn=loss_fn,
    pnet_len=length(p_net_template_vec),
    warm_rank_used=warm_rank_used,
    warm_source=warm_source,
    npts=length(idx),
    full_npts=length(times_train),
    tspan=(times_used[1], times_used[end])
  )
end

fmt_secs(x) = @sprintf("%.3f", x)

function run_exact_pass(case; do_profile::Bool, profile_mincount::Int, profile_maxdepth::Int)
  GC.gc()
  tprintln("precheck: loss_begin")
  loss_val = NaN
  loss_dt = @elapsed begin
    loss_val = case.loss_fn(case.theta0)
  end
  tprintln("precheck: loss_done dt_s=", fmt_secs(loss_dt),
    " loss=", fmt_e(loss_val, sigdigits=6))

  GC.gc()
  tprintln("precheck: pullback_begin")
  pull_loss = NaN
  back = nothing
  pullback_dt = @elapsed begin
    pull_loss, back = Zygote.pullback(case.loss_fn, case.theta0)
  end
  tprintln("precheck: pullback_done dt_s=", fmt_secs(pullback_dt),
    " loss=", fmt_e(pull_loss, sigdigits=6))

  GC.gc()
  tprintln("precheck: backward_begin")
  grad_raw = nothing
  backward_dt = @elapsed begin
    grad_raw = first(back(1.0))
  end
  grad_norm = grad_norm_safe(grad_raw)
  tprintln("precheck: backward_done dt_s=", fmt_secs(backward_dt),
    " grad_norm=", fmt_e(grad_norm, sigdigits=6))

  if do_profile
    GC.gc()
    tprintln("precheck: profile_begin")
    _, back_profile = Zygote.pullback(case.loss_fn, case.theta0)
    Profile.clear()
    @profile first(back_profile(1.0))
    tprintln("precheck: profile_flat")
    Profile.print(stdout; format=:flat, sortedby=:count, mincount=profile_mincount, maxdepth=profile_maxdepth)
    flush(stdout)
  end
end

diag_candidate = env_int("HNODECB_STAGE2_DIAG_CANDIDATE", default_candidate())
diag_warm_rank = env_int("HNODECB_STAGE2_DIAG_WARM_RANK", env_int("HNODECB_STAGE2_NN_WARM_RANK", 1))
diag_npts = env_int("HNODECB_STAGE2_DIAG_NPTS", 0)
diag_compile_npts = env_int("HNODECB_STAGE2_DIAG_COMPILE_NPTS", 8)
diag_profile = env_bool("HNODECB_STAGE2_DIAG_PROFILE", false)
diag_profile_mincount = env_int("HNODECB_STAGE2_DIAG_PROFILE_MINCOUNT", 20)
diag_profile_maxdepth = env_int("HNODECB_STAGE2_DIAG_PROFILE_MAXDEPTH", 18)
diag_sensealg_spec = env_string("HNODECB_STAGE2_DIAG_SENSEALG", "gausszygote")

global sensealg
diag_sensealg, diag_sensealg_label = resolve_sensealg(diag_sensealg_spec)
sensealg = diag_sensealg
tprintln("=== Stage2 FORMAL BACKWARD PRECHECK (03) ===")
tprintln("diag config: candidate=", diag_candidate,
  " warm_rank=", diag_warm_rank,
  " npts=", diag_npts <= 0 ? "full" : string(diag_npts),
  " sensealg=", diag_sensealg_label)

if diag_compile_npts > 0
  warmup_case = build_diag_case(diag_candidate, diag_warm_rank, diag_compile_npts)
  tprintln("compile warmup: npts=", warmup_case.npts, "/", warmup_case.full_npts)
  tprintln("compile warmup: loss_begin")
  warmup_loss = NaN
  warmup_loss_dt = @elapsed begin
    warmup_loss = warmup_case.loss_fn(warmup_case.theta0)
  end
  tprintln("compile warmup: loss_done dt_s=", fmt_secs(warmup_loss_dt),
    " loss=", fmt_e(warmup_loss, sigdigits=6))
  tprintln("compile warmup: pullback_begin")
  warmup_pull_loss = NaN
  warmup_back = nothing
  warmup_pullback_dt = @elapsed begin
    warmup_pull_loss, warmup_back = Zygote.pullback(warmup_case.loss_fn, warmup_case.theta0)
  end
  tprintln("compile warmup: pullback_done dt_s=", fmt_secs(warmup_pullback_dt),
    " loss=", fmt_e(warmup_pull_loss, sigdigits=6))
  tprintln("compile warmup: backward_begin")
  warmup_grad = nothing
  warmup_backward_dt = @elapsed begin
    warmup_grad = first(warmup_back(1.0))
  end
  tprintln("compile warmup: backward_done dt_s=", fmt_secs(warmup_backward_dt),
    " grad_norm=", fmt_e(grad_norm_safe(warmup_grad), sigdigits=6))
  GC.gc()
end

case = build_diag_case(diag_candidate, diag_warm_rank, diag_npts)
tprintln("diag init: p_net_len=", case.pnet_len,
  " warm_source=", case.warm_source,
  " used_npts=", case.npts, "/", case.full_npts,
  " tspan=[", fmt_e(case.tspan[1], sigdigits=6), ", ", fmt_e(case.tspan[2], sigdigits=6), "]")

run_exact_pass(case;
  do_profile=diag_profile,
  profile_mincount=diag_profile_mincount,
  profile_maxdepth=diag_profile_maxdepth)

tprintln("=== Stage2 FORMAL BACKWARD PRECHECK DONE ===")
