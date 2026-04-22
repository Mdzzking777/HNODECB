#=
Reduced AD-vs-FD correctness check for AFM Stage2light (scenario 03).

This script reuses the official Stage2 loss ingredients and route, but on a reduced set
of saveat/loss points. It checks:
- p_mech_raw (all entries)
- sampled p_net entries

Default branches:
- active
- state
- x2dot
=#

cd(@__DIR__)

using Printf
using Zygote

env_bool(name, default) = lowercase(strip(get(ENV, name, default ? "1" : "0"))) in ("1", "true", "yes", "on")

function env_int(name, default)
  raw = strip(get(ENV, name, ""))
  parsed = tryparse(Int, raw)
  return parsed === nothing ? default : parsed
end

function env_float(name, default)
  raw = strip(get(ENV, name, ""))
  parsed = tryparse(Float64, raw)
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
if !haskey(ENV, "HNODECB_STAGE2_X3R_WEIGHT")
  ENV["HNODECB_STAGE2_X3R_WEIGHT"] = "0.0"
end

include("afm_param_stage2_03.jl")

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

function parse_branches(spec::AbstractString)
  raw = [lowercase(strip(s)) for s in split(spec, ",")]
  branches = [s for s in raw if s != ""]
  isempty(branches) && return ["active", "state", "x2dot"]
  allowed = Set(["active", "state", "x2dot"])
  for branch in branches
    branch in allowed || error("Unsupported branch: " * branch)
  end
  return branches
end

function sampled_pnet_indices(pnet_len::Int, sample_count::Int)
  count = clamp(sample_count, 1, pnet_len)
  idx = unique(round.(Int, range(1, pnet_len, length=count)))
  sort!(idx)
  return idx
end

function build_diag_case(candidate_idx::Int, warm_rank::Int, npts::Int)
  1 <= candidate_idx <= length(top_candidates) || error("candidate_idx out of range 1..$(length(top_candidates))")
  candidate = top_candidates[candidate_idx]
  params = candidate.params
  ks0 = params["ks0"]
  cs0 = params["cs0"]

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
  weights_used = ifelse.(contact_used, contact_loss_weight, noncontact_loss_weight)
  weights_sum = sum(weights_used)

  return (
    theta0=theta0,
    appr=approximating_neural_network,
    st=st,
    re_pnet=re_pnet,
    pnet_len=length(p_net_template_vec),
    warm_rank_used=warm_rank_used,
    warm_source=warm_source,
    ode_used=ode_used,
    x2dot_used=x2dot_used,
    contact_used=contact_used,
    times_used=times_used,
    weights_used=weights_used,
    weights_sum=weights_sum,
    npts=length(idx),
    full_npts=length(times_train),
    tspan=(times_used[1], times_used[end]),
    x3_t0_val=ode_used[3, 1]
  )
end

function branch_loss(case, θ, branch::AbstractString)
  mech = [
    bound_param(θ.mech_raw[1], ks_bounds[1], ks_bounds[2]),
    bound_param(θ.mech_raw[2], cs_bounds[1], cs_bounds[2])
  ]
  p_net_struct = case.re_pnet(θ.p_net)
  p = ComponentVector(p_net=p_net_struct, mech=mech)

  u0 = [case.ode_used[1, 1], case.ode_used[2, 1], case.x3_t0_val]
  prob = ODEProblem{false}(make_uode_func_oop(case.appr, case.st, known_pars), u0, case.tspan, p)
  sol = solve(prob, integrator; saveat=case.times_used, abstol=abstol, reltol=reltol, sensealg=sensealg, maxiters=ode_maxiters)
  string(sol.retcode) == "Success" || return Inf
  size(sol, 2) == length(case.times_used) || return Inf
  uhat = Array(sol)

  state_err = vec(sum(abs2.((case.ode_used[1:2, :] .- uhat[1:2, :]) ./ state12_scale_full), dims=1))
  state_loss = sum(case.weights_used .* state_err) / case.weights_sum

  x2dot_loss = 0.0
  contact_idx = findall(case.contact_used)
  if !isempty(contact_idx)
    x2dot_pred_contact = x2dot_rhs_batch(uhat, contact_idx, mech, p_net_struct, case.appr, case.st, known_pars, case.times_used)
    any(x -> !isfinite(x), x2dot_pred_contact) && return Inf
    x2_err = abs2.((case.x2dot_used[contact_idx] .- x2dot_pred_contact) ./ x2dot_scale_full)
    x2_w = case.weights_used[contact_idx]
    x2dot_loss = sum(x2_w .* x2_err) / sum(x2_w)
  end

  if branch == "active"
    return state_loss + x2dot_loss
  elseif branch == "state"
    return state_loss
  elseif branch == "x2dot"
    return x2dot_loss
  end
  error("Unsupported branch: " * branch)
end

function fd_entry(loss_fn, theta0, group::Symbol, idx::Int, eps_rel::Float64)
  base = group === :p_net ? theta0.p_net[idx] : theta0.mech_raw[idx]
  h = eps_rel * max(1.0, abs(base))
  theta_plus = ComponentVector(p_net=copy(theta0.p_net), mech_raw=copy(theta0.mech_raw))
  theta_minus = ComponentVector(p_net=copy(theta0.p_net), mech_raw=copy(theta0.mech_raw))
  if group === :p_net
    theta_plus.p_net[idx] += h
    theta_minus.p_net[idx] -= h
  else
    theta_plus.mech_raw[idx] += h
    theta_minus.mech_raw[idx] -= h
  end
  loss_plus = loss_fn(theta_plus)
  loss_minus = loss_fn(theta_minus)
  return (loss_plus - loss_minus) / (2h), h, loss_plus, loss_minus
end

function rel_err(ad, fd)
  return abs(ad - fd) / max(abs(fd), 1e-12)
end

fmt_secs(x) = @sprintf("%.3f", x)

diag_candidate = env_int("HNODECB_STAGE2_DIAG_CANDIDATE", default_candidate())
diag_warm_rank = env_int("HNODECB_STAGE2_DIAG_WARM_RANK", env_int("HNODECB_STAGE2_NN_WARM_RANK", 1))
diag_npts = env_int("HNODECB_STAGE2_DIAG_NPTS", 8)
diag_branches = parse_branches(env_string("HNODECB_STAGE2_GRADCHECK_BRANCHES", "active,state,x2dot"))
diag_pnet_samples = env_int("HNODECB_STAGE2_GRADCHECK_PNET_SAMPLES", 4)
diag_fd_eps = env_float("HNODECB_STAGE2_GRADCHECK_FD_EPS", 1e-6)

tprintln("=== Stage2 REDUCED AD-vs-FD GRADCHECK (03) ===")
tprintln("diag config: candidate=", diag_candidate,
  " warm_rank=", diag_warm_rank,
  " npts=", diag_npts,
  " branches=", join(diag_branches, ","),
  " fd_eps=", fmt_e(diag_fd_eps, sigdigits=3))
tprintln("route: rhs=out-of-place | sensealg=GaussAdjoint(ZygoteVJP())")

case = build_diag_case(diag_candidate, diag_warm_rank, diag_npts)
pnet_idx = sampled_pnet_indices(case.pnet_len, diag_pnet_samples)
tprintln("diag init: p_net_len=", case.pnet_len,
  " warm_source=", case.warm_source,
  " used_npts=", case.npts, "/", case.full_npts,
  " p_net_samples=", join(pnet_idx, ","))

for branch in diag_branches
  tprintln("--- branch=", branch, " ---")
  loss_fn = θ -> branch_loss(case, θ, branch)

  GC.gc()
  tprintln("ad: pullback_begin")
  ad_loss = NaN
  ad_back = nothing
  ad_pull_dt = @elapsed begin
    ad_loss, ad_back = Zygote.pullback(loss_fn, case.theta0)
  end
  tprintln("ad: pullback_done dt_s=", fmt_secs(ad_pull_dt), " loss=", fmt_e(ad_loss, sigdigits=6))

  GC.gc()
  tprintln("ad: backward_begin")
  ad_grad = nothing
  ad_back_dt = @elapsed begin
    ad_grad = first(ad_back(1.0))
  end
  tprintln("ad: backward_done dt_s=", fmt_secs(ad_back_dt),
    " p_net_norm=", fmt_e(sqrt(sum(abs2, ad_grad.p_net)), sigdigits=6),
    " mech_norm=", fmt_e(sqrt(sum(abs2, ad_grad.mech_raw)), sigdigits=6))

  mech_rel_max = 0.0
  for i in eachindex(case.theta0.mech_raw)
    fd_val, h, loss_plus, loss_minus = fd_entry(loss_fn, case.theta0, :mech_raw, i, diag_fd_eps)
    ad_val = ad_grad.mech_raw[i]
    err = rel_err(ad_val, fd_val)
    mech_rel_max = max(mech_rel_max, err)
    tprintln("mech[", i, "]: AD=", fmt_hp(ad_val),
      " FD=", fmt_hp(fd_val),
      " relerr=", fmt_e(err, sigdigits=4),
      " h=", fmt_hp(h),
      " loss+=", fmt_e(loss_plus, sigdigits=6),
      " loss-=", fmt_e(loss_minus, sigdigits=6))
  end

  pnet_rel_max = 0.0
  for idx in pnet_idx
    fd_val, h, loss_plus, loss_minus = fd_entry(loss_fn, case.theta0, :p_net, idx, diag_fd_eps)
    ad_val = ad_grad.p_net[idx]
    err = rel_err(ad_val, fd_val)
    pnet_rel_max = max(pnet_rel_max, err)
    tprintln("p_net[", idx, "]: AD=", fmt_hp(ad_val),
      " FD=", fmt_hp(fd_val),
      " relerr=", fmt_e(err, sigdigits=4),
      " h=", fmt_hp(h),
      " loss+=", fmt_e(loss_plus, sigdigits=6),
      " loss-=", fmt_e(loss_minus, sigdigits=6))
  end

  tprintln("branch summary: ", branch,
    " | mech_relerr_max=", fmt_e(mech_rel_max, sigdigits=4),
    " | p_net_relerr_max=", fmt_e(pnet_rel_max, sigdigits=4))
end

tprintln("=== Stage2 REDUCED AD-vs-FD GRADCHECK DONE ===")
