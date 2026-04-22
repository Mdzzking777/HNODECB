ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Serialization
using DataFrames
using DifferentialEquations
using ComponentArrays
using Printf
using Statistics

if !isdefined(ComponentArrays, :Shaped1DAxis) && isdefined(ComponentArrays, :ShapedAxis)
    @eval ComponentArrays const Shaped1DAxis = ShapedAxis
end

function step2b_find_repo_root(start_dir::String)
    dir = normpath(abspath(start_dir))
    while true
        if isfile(joinpath(dir, "run_step2b_windowed_03_local.ps1")) || isdir(joinpath(dir, "logs"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not locate repo root from: " * start_dir)
        dir = parent
    end
end

const STEP2B_VIS_REPO_ROOT = step2b_find_repo_root(@__DIR__)
const STEP2B_RESULT_DIR = normpath(joinpath(STEP2B_VIS_REPO_ROOT, "step2b_model_trainer", "res_afm_03_windowed"))
const STEP2B_LOG_DIR = normpath(joinpath(STEP2B_VIS_REPO_ROOT, "logs", "step2b_windowed", "print"))
const STEP2B_OUT_DIR = normpath(joinpath(STEP2B_VIS_REPO_ROOT, "logs", "step2b_windowed", "visualization"))
const STEP2B_CACHE_DIR = normpath(joinpath(STEP2B_OUT_DIR, ".cache"))
const STEP2B_EXPORTER_PATH = normpath(joinpath(@__DIR__, "_export_step2b_portable_03.jl"))
const STEP2B_HPC_ENV = normpath(joinpath(STEP2B_VIS_REPO_ROOT, "HPC_env"))
const STEP2B_DATA_DIR = normpath(joinpath(STEP2B_VIS_REPO_ROOT, "datasets", "e0.0", "data"))
const STEP2B_ODE_DATA_PATH = joinpath(STEP2B_DATA_DIR, "ode_data_afm_dmt_kv.jld")
const STEP2B_TRAJ_PATH = joinpath(STEP2B_DATA_DIR, "pert_df_afm_dmt_kv.jld")
const STEP2B_T_US_SCALE = 1.0e6

include(joinpath(STEP2B_VIS_REPO_ROOT, "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_settings.jl"))
include(joinpath(STEP2B_VIS_REPO_ROOT, "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_functions.jl"))

step2b_gelu(x::Real) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

function step2b_layer_syms(p_net)
    syms = Symbol[s for s in propertynames(p_net) if startswith(String(s), "layer_")]
    sort!(syms; by = s -> something(tryparse(Int, split(String(s), "_")[end]), typemax(Int)))
    return syms
end

step2b_fmt_num(x; sigdigits=3) = (x isa Number && isfinite(x)) ? @sprintf("%.*e", sigdigits, x) : "None"
step2b_fmt_pct(x; digits=2) = (x isa Number && isfinite(x)) ? @sprintf("%.*f", digits, x) : "None"

function step2b_window_title(role::AbstractString, label::AbstractString = "")
    role_norm = lowercase(strip(role))
    if role_norm == "first_contact"
        return "window1: right after first contact"
    elseif role_norm == "max_x1_pp_change"
        return "window2: the most drastic region"
    elseif role_norm == "tail_stable"
        return "window3: stable region at the end"
    end
    return isempty(strip(label)) ? role : label
end

function step2b_default_result_paths(result_dir::String = STEP2B_RESULT_DIR)
    return [
        joinpath(result_dir, "afm_03_windowed_p1.jld"),
        joinpath(result_dir, "afm_03_windowed_p2.jld"),
        joinpath(result_dir, "afm_03_windowed_p3.jld"),
    ]
end

function step2b_default_log_paths(log_dir::String = STEP2B_LOG_DIR)
    return [
        joinpath(log_dir, "log2_03_step2b_windowed_local_p1.txt"),
        joinpath(log_dir, "log2_03_step2b_windowed_local_p2.txt"),
        joinpath(log_dir, "log2_03_step2b_windowed_local_p3.txt"),
    ]
end

function step2b_shard_log_series(log_dir::String, shard::Integer)
    paths = String[]
    if isdir(log_dir)
        base_paths = String[]
        resume_paths = Pair{Int, String}[]
        resume_pattern = Regex("_resume(\\d+)_p$(shard)\\.txt\$")
        base_pattern = Regex("_p$(shard)\\.txt\$")
        for path in readdir(log_dir; join=true)
            name = basename(path)
            m_resume = match(resume_pattern, name)
            if m_resume !== nothing
                idx = tryparse(Int, m_resume.captures[1])
                idx === nothing || push!(resume_paths, idx => path)
            elseif match(base_pattern, name) !== nothing
                push!(base_paths, path)
            end
        end
        sort!(base_paths)
        sort!(resume_paths; by=first)
        append!(paths, base_paths)
        append!(paths, last.(resume_paths))
    end
    if isempty(paths)
        push!(paths, joinpath(log_dir, "log2_03_step2b_windowed_local_p$(shard).txt"))
    end
    return paths
end

function step2b_default_log_series(log_dir::String = STEP2B_LOG_DIR; shard_count::Int=3)
    return [step2b_shard_log_series(log_dir, shard) for shard in 1:shard_count]
end

function step2b_existing_log_series(log_dir::String = STEP2B_LOG_DIR; shard_count::Int=3)
    series = step2b_default_log_series(log_dir; shard_count=shard_count)
    return [filter(isfile, paths) for paths in series]
end

function step2b_merge_epoch_rows(rows::Vector{<:NamedTuple})
    by_epoch = Dict{Int, NamedTuple}()
    for row in rows
        by_epoch[Int(row.epoch)] = row
    end
    epochs = sort!(collect(keys(by_epoch)))
    return [by_epoch[e] for e in epochs]
end

function step2b_post_contact_data()
    ode_data_full = deserialize(STEP2B_ODE_DATA_PATH)
    traj_df_full = deserialize(STEP2B_TRAJ_PATH)
    first_contact = findfirst(traj_df_full.contact .== 1)
    first_contact === nothing && error("No contact point found in AFM dataset.")
    return (
        ode = Float64.(ode_data_full[:, first_contact:end]),
        traj = traj_df_full[first_contact:end, :],
    )
end

function step2b_best_result(payload)
    haskey(payload, :results) || error("Step2b result file lacks :results.")
    isempty(payload.results) && error("Step2b result file has no finalized :results yet. Wait until step2b finishes before generating trajectory/reconstruction plots.")
    vals = [r.validation_resulting_cost for r in payload.results]
    return payload.results[argmin(vals)]
end

function step2b_portable_cache_path(path::String)
    stem = replace(basename(path), r"\.jld$" => ".portable.bin")
    return joinpath(STEP2B_CACHE_DIR, stem)
end

function step2b_ensure_portable_cache(path::String)
    cache_path = step2b_portable_cache_path(path)
    cache_stale = !isfile(cache_path) || mtime(cache_path) < mtime(path)
    if cache_stale
        mkpath(dirname(cache_path))
        cmd = `$(Base.julia_cmd()) --project=$(STEP2B_HPC_ENV) $(STEP2B_EXPORTER_PATH) $(path) $(cache_path)`
        run(cmd)
    end
    return cache_path
end

function step2b_read_portable_payload(path::String)
    cache_path = step2b_ensure_portable_cache(path)
    return deserialize(cache_path)
end

function step2b_nn_raw(u::AbstractVector, p_net)
    x = Float64[u[1], u[2], u[3]]
    layer_syms = step2b_layer_syms(p_net)
    isempty(layer_syms) && error("Step2b p_net contains no layer_* entries.")
    for (i, lname) in enumerate(layer_syms)
        layer = getproperty(p_net, lname)
        z = layer.weight * x .+ layer.bias
        if i == length(layer_syms)
            return Float64(z[1])
        end
        x = step2b_gelu.(z)
    end
    error("Unreachable step2b_nn_raw.")
end

function step2b_true_fhertz_from_state(u::AbstractVector)
    s = dist + u[1] - u[3]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    return (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
end

function step2b_true_x2dot_from_state(u::AbstractVector, t::Float64)
    F_contact = step2b_true_fcontact_from_state(u)
    return (Fd * cos(wd * t) - k * u[1] - c * u[2] + F_contact) / m
end

function step2b_true_x3dot_from_state(u::AbstractVector)
    F_contact = step2b_true_fcontact_from_state(u)
    return (-F_contact - ks * u[3]) / cs
end

function step2b_true_fcontact_from_state(u::AbstractVector)
    s = dist + u[1] - u[3]
    w_true = contact_weight(s, adhesion_transition)
    return step2b_true_fhertz_from_state(u) - Fad * w_true
end

function step2b_trapz_integral(times::AbstractVector, values::AbstractVector)
    n = length(times)
    n == length(values) || error("times/values length mismatch.")
    n < 2 && return 0.0
    total = 0.0
    @inbounds for i in 1:(n - 1)
        dt = times[i + 1] - times[i]
        total += 0.5 * dt * (values[i + 1] + values[i])
    end
    return total
end

function step2b_overlap_ratio(times::AbstractVector, a::AbstractVector, b::AbstractVector)
    mins = min.(a, b)
    maxs = max.(a, b)
    denom = step2b_trapz_integral(times, maxs)
    denom == 0.0 && return NaN
    return step2b_trapz_integral(times, mins) / denom
end

function step2b_make_deriv(; nn_gain::Float64=1.0)
    function f(du, u, p, t)
        ode_par = p.ode_par
        k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs = ode_par
        s = dist + u[1] - u[3]
        w_pred = contact_weight(s, adhesion_transition)
        nn_raw = step2b_nn_raw(u, p.p_net)
        Fad_eff = Fad * w_pred
        F_contact = nn_gain * nn_raw * w_pred
        @inbounds du[1] = u[2]
        @inbounds du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] + F_contact) / m
        @inbounds du[3] = (-F_contact - ks * u[3]) / cs
    end
    return f
end

function step2b_window_record(path::String, ode_post, traj_post)
    portable = step2b_read_portable_payload(path)
    best = portable.best
    win = portable.window

    start_idx = Int(win.start_idx)
    stop_idx = Int(win.stop_idx)
    role = hasproperty(win, :role) ? String(win.role) : ""
    label = hasproperty(win, :label) ? String(win.label) : basename(path)
    title = step2b_window_title(role, label)

    ode_win = Float64.(ode_post[:, start_idx:stop_idx])
    traj_win = traj_post[start_idx:stop_idx, :]
    times = Float64.(traj_win.t)
    times_us = times .* STEP2B_T_US_SCALE

    p_est = Float64.(best.parameters_training)
    p_net = best.p_net
    nn_gain = hasproperty(best, :nn_gain) ? Float64(best.nn_gain) : 1.0
    num_hidden_layers = hasproperty(best, :num_hidden_layers) ? Int(best.num_hidden_layers) : 0
    num_hidden_nodes = hasproperty(best, :num_hidden_nodes) ? Int(best.num_hidden_nodes) : 2
    u0 = vec(Float64.(ode_win[:, 1]))
    deriv = step2b_make_deriv(; nn_gain=nn_gain)
    prob = ODEProblem{true}(deriv, u0, (times[1], times[end]), ComponentVector(p_net=p_net, ode_par=p_est))
    sol = solve(prob, Rosenbrock23(autodiff=false); saveat=times, abstol=1e-8, reltol=1e-8, maxiters=1_000_000)
    string(sol.retcode) == "Success" || error("ODE solve failed for $(basename(path)): retcode=$(sol.retcode)")
    uhat = Array(sol)

    x2dot_true = Float64[step2b_true_x2dot_from_state(view(ode_win, :, j), times[j]) for j in axes(ode_win, 2)]
    x2dot_pred = Float64[step2b_predicted_x2dot_from_state(view(uhat, :, j), times[j], p_est, p_net; nn_gain=nn_gain) for j in axes(uhat, 2)]
    x3dot_true = Float64[step2b_true_x3dot_from_state(view(ode_win, :, j)) for j in axes(ode_win, 2)]
    x3dot_pred = Float64[step2b_predicted_x3dot_from_state(view(uhat, :, j), times[j], p_est, p_net; nn_gain=nn_gain) for j in axes(uhat, 2)]
    fcontact_true = Float64[step2b_true_fcontact_from_state(view(ode_win, :, j)) for j in axes(ode_win, 2)]
    fcontact_pred = Float64[step2b_predicted_fcontact_from_state(view(uhat, :, j), times[j], p_net; nn_gain=nn_gain) for j in axes(uhat, 2)]
    nn_raw = Float64[step2b_nn_raw(view(uhat, :, j), p_net) for j in axes(uhat, 2)]
    w_pred = Float64[contact_weight(dist + uhat[1, j] - uhat[3, j], adhesion_transition) for j in axes(uhat, 2)]
    w_true = Float64[contact_weight(traj_win.s[j], adhesion_transition) for j in axes(traj_win.s, 1)]

    return (
        payload = portable,
        best = best,
        window = win,
        title = title,
        role = role,
        label = label,
        start_idx = start_idx,
        stop_idx = stop_idx,
        times = times,
        times_us = times_us,
        ode_win = ode_win,
        traj_win = traj_win,
        uhat = uhat,
        p_est = p_est,
        p_net = p_net,
        nn_gain = nn_gain,
        num_hidden_layers = num_hidden_layers,
        num_hidden_nodes = num_hidden_nodes,
        x2dot_true = x2dot_true,
        x2dot_pred = x2dot_pred,
        x3dot_true = x3dot_true,
        x3dot_pred = x3dot_pred,
        fcontact_true = fcontact_true,
        fcontact_pred = fcontact_pred,
        nn_raw = nn_raw,
        w_pred = w_pred,
        w_true = w_true,
    )
end

function step2b_predicted_fcontact_from_state(u::AbstractVector, t::Float64, p_net; nn_gain::Float64=1.0)
    s = dist + u[1] - u[3]
    w_pred = contact_weight(s, adhesion_transition)
    return nn_gain * step2b_nn_raw(u, p_net) * w_pred
end

function step2b_predicted_x2dot_from_state(u::AbstractVector, t::Float64, p_est::AbstractVector, p_net; nn_gain::Float64=1.0)
    F_contact = step2b_predicted_fcontact_from_state(u, t, p_net; nn_gain=nn_gain)
    return (Fd * cos(wd * t) - k * u[1] - c * u[2] + F_contact) / m
end

function step2b_predicted_x3dot_from_state(u::AbstractVector, t::Float64, p_est::AbstractVector, p_net; nn_gain::Float64=1.0)
    F_contact = step2b_predicted_fcontact_from_state(u, t, p_net; nn_gain=nn_gain)
    ks_hat = p_est[10]
    cs_hat = p_est[11]
    return (-F_contact - ks_hat * u[3]) / cs_hat
end

function step2b_log_role_label(log_path::String)
    role = ""
    label = basename(log_path)
    role_re = r"role=([^|]+)"
    label_re = r"Window label=([^|]+)"
    for line in eachline(log_path)
        if label == basename(log_path)
            m = match(label_re, line)
            if m !== nothing
                label = strip(m.captures[1])
            end
        end
        if isempty(role)
            m = match(role_re, line)
            if m !== nothing
                role = strip(m.captures[1])
            end
        end
        (!isempty(role) && label != basename(log_path)) && break
    end
    return role, label
end

function step2b_log_role_label(log_paths::AbstractVector{<:AbstractString})
    for path in log_paths
        !isfile(path) && continue
        role, label = step2b_log_role_label(path)
        if !isempty(role) || label != basename(path)
            return role, label
        end
    end
    fallback = isempty(log_paths) ? "" : basename(first(log_paths))
    return "", fallback
end

function step2b_epoch_xlim(records)
    last_epochs = Int[]
    for rec in records
        if hasproperty(rec, :epochs) && !isempty(rec.epochs)
            push!(last_epochs, last(rec.epochs))
        end
    end
    hi = isempty(last_epochs) ? 1000 : maximum(last_epochs)
    return (0, max(hi, 500))
end
