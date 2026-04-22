ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Serialization
using DataFrames
using Plots

function find_repo_root(start_dir::String)
    dir = normpath(abspath(start_dir))
    while true
        if isfile(joinpath(dir, "run_stage2light_windowed_03_local.ps1")) || isdir(joinpath(dir, "test_case_settings"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not locate repo root from: " * start_dir)
        dir = parent
    end
end

const REPO_ROOT = find_repo_root(@__DIR__)

include(joinpath(REPO_ROOT, "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_settings.jl"))
include(joinpath(REPO_ROOT, "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_functions.jl"))

const DATA_DIR = joinpath(REPO_ROOT, "datasets", "e0.0", "data")
const ODE_DATA_PATH = joinpath(DATA_DIR, "ode_data_afm_dmt_kv.jld")
const TRAJ_PATH = joinpath(DATA_DIR, "pert_df_afm_dmt_kv.jld")
const T_US_SCALE = 1.0e6

peak_to_peak(v) = isempty(v) ? NaN : (maximum(v) - minimum(v))

function contact_onsets(contact_mask::AbstractVector{Bool})
    out = Int[]
    for i in eachindex(contact_mask)
        if contact_mask[i] && (i == firstindex(contact_mask) || !contact_mask[i - 1])
            push!(out, i)
        end
    end
    return out
end

function nearest_time_index(t_us::AbstractVector{<:Real}, target_us::Real)
    idx = findmin(abs.(Float64.(t_us) .- Float64(target_us)))[2]
    return Int(idx)
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
    len = stop_idx - start_idx + 1
    n = length(t_us)
    len <= n || error("Window length exceeds available trajectory length.")

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
            new_start, new_stop = relocate_window_with_anchor(t_post .* T_US_SCALE, win.start_idx, win.stop_idx; anchor=spec.anchor, target_us=spec.target_us)
            if win.role == "first_contact" && hasproperty(win, :contact_post)
                new_stop = extend_stop_to_contact_end(win.contact_post, new_stop)
            end
            push!(adjusted, merge(win, (
                start_idx = new_start,
                stop_idx = new_stop,
                t_start = t_post[new_start],
                t_stop = t_post[new_stop],
            )))
        else
            push!(adjusted, win)
        end
    end
    return adjusted
end

function window_title(role::AbstractString)
    role_norm = lowercase(strip(role))
    if role_norm == "first_contact"
        return "window1: right after first contact"
    elseif role_norm == "max_x1_pp_change"
        return "window2: the most drastic region"
    elseif role_norm == "tail_stable"
        return "window3: stable region at the end"
    end
    return String(role)
end

function build_window_manifest(traj_df, ode_data)
    contact_full = Vector{Bool}(traj_df.contact .== 1)
    first_contact_idx = findfirst(contact_full)
    first_contact_idx === nothing && error("No contact point found in the AFM dataset.")

    solution_post = traj_df[first_contact_idx:end, :]
    t_post = Float64.(solution_post.t)
    contact_post = Vector{Bool}(solution_post.contact .== 1)
    x1_post = vec(Float64.(ode_data[1, first_contact_idx:end]))
    onset_post = contact_onsets(contact_post)
    length(onset_post) >= 3 || error("Need at least 3 contact onsets after first contact to build two-cycle windows.")

    cycle_pp = Float64[]
    for k in 1:(length(onset_post) - 1)
        lo = onset_post[k]
        hi = onset_post[k + 1] - 1
        push!(cycle_pp, peak_to_peak(@view x1_post[lo:hi]))
    end

    candidates = NamedTuple[]
    for k in 1:(length(onset_post) - 2)
        start_idx = onset_post[k]
        stop_idx = onset_post[k + 2] - 1
        push!(candidates, (
            candidate_index = k,
            start_idx = start_idx,
            stop_idx = stop_idx,
            t_start = t_post[start_idx],
            t_stop = t_post[stop_idx],
            x1_pp_delta = abs(cycle_pp[k + 1] - cycle_pp[k])
        ))
    end
    isempty(candidates) && error("No valid two-cycle windows were built from the post-contact trajectory.")

    first_idx = 1
    change_order = sortperm([cand.x1_pp_delta for cand in candidates]; rev=true)
    tail_order = collect(length(candidates):-1:1)

    selected = NamedTuple[]
    seen = Set{Tuple{Int, Int}}()
    function add_window(role::String, idx::Int)
        cand = candidates[idx]
        key = (cand.start_idx, cand.stop_idx)
        key in seen && return false
        push!(seen, key)
        push!(selected, merge(cand, (role = role, title = window_title(role), contact_post = contact_post)))
        return true
    end

    function add_first_unique(role::String, order)
        for idx in order
            add_window(role, idx) && return idx
        end
        return nothing
    end

    add_window("first_contact", first_idx)
    add_first_unique("max_x1_pp_change", change_order)
    add_first_unique("tail_stable", tail_order)
    return apply_window_time_overrides(selected, t_post)
end

ode_data = deserialize(ODE_DATA_PATH)
traj_df = deserialize(TRAJ_PATH)

times = collect(traj_df.t)
times_us = times .* T_US_SCALE
x1 = vec(ode_data[1, :])
x3 = vec(ode_data[3, :])

raw_delta = @. -dist - x1 + x3
delta = similar(raw_delta)
raw_f_hertz = zeros(length(raw_delta))

coeff = (4.0 / 3.0) * Estar * sqrt(R)

for i in eachindex(raw_delta)
    delta[i] = softplus(raw_delta[i], adhesion_transition)
    if raw_delta[i] >= 0.0
        raw_f_hertz[i] = coeff * (raw_delta[i]^1.5)
    end
end

f_hertz_generated = coeff .* (delta .^ 1.5)
windows = build_window_manifest(traj_df, ode_data)
length(windows) == 3 || error("Expected exactly 3 windows, got $(length(windows)).")

plot_dir = joinpath(REPO_ROOT, "logs", "pics")
mkpath(plot_dir)
plot_path = joinpath(plot_dir, "afm_hertz_reference_signals_03.png")

panels = Plots.Plot[]

for win in windows
    x_window = (win.t_start * T_US_SCALE, win.t_stop * T_US_SCALE)
    p = plot(
        times_us,
        raw_f_hertz;
        label = "raw F_hertz true",
        xlabel = "time (μs)",
        ylabel = "force",
        title = win.title,
        linewidth = 2,
        xlims = x_window,
        color = :black,
        legend = :topright,
    )
    push!(panels, p)
end

for win in windows
    x_window = (win.t_start * T_US_SCALE, win.t_stop * T_US_SCALE)
    p = plot(
        times_us,
        f_hertz_generated;
        label = "softplus F_hertz true",
        xlabel = "time (μs)",
        ylabel = "force",
        title = win.title,
        linewidth = 2,
        xlims = x_window,
        color = :firebrick,
        legend = :topright,
    )
    push!(panels, p)
end

for win in windows
    x_window = (win.t_start * T_US_SCALE, win.t_stop * T_US_SCALE)
    p = plot(
        times_us,
        delta;
        label = "delta",
        xlabel = "time (μs)",
        ylabel = "indentation",
        title = win.title,
        linewidth = 2,
        xlims = x_window,
        color = :steelblue,
        legend = :topright,
    )
    push!(panels, p)
end

fig = plot(panels...; layout = (3, 3), size = (1800, 1200))
savefig(fig, plot_path)

println("Saved plot to: ", normpath(plot_path))
for (i, win) in enumerate(windows)
    println("W", i, " [", win.role, "] t=[", round(win.t_start * T_US_SCALE; digits = 3), ", ", round(win.t_stop * T_US_SCALE; digits = 3), "] us")
end
