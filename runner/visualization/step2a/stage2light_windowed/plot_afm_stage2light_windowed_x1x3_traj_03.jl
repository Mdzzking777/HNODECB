ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Serialization
using DataFrames
using Printf
using Plots
using DifferentialEquations
using ComponentArrays

function find_repo_root(start_dir::String)
    dir = normpath(abspath(start_dir))
    while true
        if isfile(joinpath(dir, "run_stage2light_windowed_03_local.ps1")) || isdir(joinpath(dir, "logs"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not locate repo root from: " * start_dir)
        dir = parent
    end
end

const REPO_ROOT = find_repo_root(@__DIR__)
const RESULT_DIR = normpath(joinpath(REPO_ROOT, "step2a_hyperparameter_tuning", "hyperparameter_tuning_second_stage", "results_afm"))
const DEFAULT_OUT_DIR = normpath(joinpath(REPO_ROOT, "logs", "stage2_step2a", "local", "windowed", "visualization"))
const DEFAULT_OUT_FILE = "afm_param_stage2light_windowed_03_x1x3_traj_grid.png"
const DATA_DIR = normpath(joinpath(REPO_ROOT, "datasets", "e0.0", "data"))
const ODE_DATA_PATH = joinpath(DATA_DIR, "ode_data_afm_dmt_kv.jld")
const TRAJ_PATH = joinpath(DATA_DIR, "pert_df_afm_dmt_kv.jld")
const T_US_SCALE = 1.0e6

include(joinpath(REPO_ROOT, "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_settings.jl"))
include(joinpath(REPO_ROOT, "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_functions.jl"))
include(joinpath(@__DIR__, "_dualrank_utils_03.jl"))
include(joinpath(@__DIR__, "_checkpoint_fallback_utils_03.jl"))

function nn_input_from_state(u)
    return u[1:3]
end

function gelu_scalar(x::Float64)
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
end

function unpack_nn_weights(p_vec::AbstractVector{<:Real}, num_hidden_layers::Int, num_hidden_nodes::Int)
    hidden = 2^num_hidden_nodes
    dims = Tuple{Int, Int}[(hidden, 3)]
    for _ in 1:num_hidden_layers
        push!(dims, (hidden, hidden))
    end
    push!(dims, (1, hidden))

    layers = NamedTuple{(:W, :b), Tuple{Matrix{Float64}, Vector{Float64}}}[]
    offset = 1
    for (rows, cols) in dims
        w_len = rows * cols
        w_stop = offset + w_len - 1
        w_stop <= length(p_vec) || error("p_net_vec length mismatch while unpacking NN weights.")
        W = reshape(Float64.(p_vec[offset:w_stop]), rows, cols)
        offset = w_stop + 1

        b_stop = offset + rows - 1
        b_stop <= length(p_vec) || error("p_net_vec bias length mismatch while unpacking NN weights.")
        b = Float64.(p_vec[offset:b_stop])
        offset = b_stop + 1
        push!(layers, (W=W, b=b))
    end
    offset == length(p_vec) + 1 || error("Unused entries remain in p_net_vec after unpacking NN weights.")
    return layers
end

function nn_forward(weights, x::AbstractVector{<:Real})
    y = Float64.(x)
    for i in 1:length(weights)
        y = weights[i].W * y .+ weights[i].b
        i < length(weights) && (y = gelu_scalar.(y))
    end
    return y
end

function true_fcontact_from_state(u)
    s = dist + u[1] - u[3]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    w_true = contact_weight(s, adhesion_transition)
    return (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5) - Fad * w_true
end

function make_uode_func_oop(nn_weights, known_pars; nn_gain::Float64=1.0)
    k, wd, m, c, Fd, R, dist, Fad = known_pars
    function f(u, p, t)
        ks = p.mech[1]
        cs = p.mech[2]

        s = dist + u[1] - u[3]
        delta = softplus(-s, adhesion_transition)
        delta = ifelse(delta > 0.0, delta, 0.0)
        w_pred = contact_weight(s, adhesion_transition)

        nn_in = nn_input_from_state(u)
        nn_out = nn_forward(nn_weights, nn_in)
        F_contact = nn_gain * nn_out[1] * w_pred

        du1 = u[2]
        du2 = (Fd * cos(wd * t) - k * u[1] - c * u[2] + F_contact) / m
        du3 = (-F_contact - ks * u[3]) / cs
        return [du1, du2, du3]
    end
    return f
end

function true_x3dot_from_state(u)
    F_contact = true_fcontact_from_state(u)
    return (-F_contact - ks * u[3]) / cs
end

function predicted_x3dot_from_state(u, nn_weights, mech; nn_gain::Float64=1.0)
    s = dist + u[1] - u[3]
    w_pred = contact_weight(s, adhesion_transition)
    nn_out = nn_forward(nn_weights, nn_input_from_state(u))
    F_contact = nn_gain * nn_out[1] * w_pred
    return (-F_contact - mech[1] * u[3]) / mech[2]
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

function load_result_paths(result_dir::String=RESULT_DIR)
    fixed = [
        joinpath(result_dir, "afm_param_stage2light_windowed_03_p1.jld"),
        joinpath(result_dir, "afm_param_stage2light_windowed_03_p2.jld"),
        joinpath(result_dir, "afm_param_stage2light_windowed_03_p3.jld"),
    ]
    if all(isfile, fixed)
        return fixed
    end
    paths = String[]
    for shard in 1:3
        matches = sort(filter(path -> occursin(Regex("_p$(shard)\\.jld\$"), basename(path)), readdir(result_dir; join=true)))
        isempty(matches) && error("Missing result shard _p$(shard) in: " * result_dir)
        push!(paths, matches[end])
    end
    return paths
end

function post_contact_data()
    ode_data_full = deserialize(ODE_DATA_PATH)
    traj_df_full = deserialize(TRAJ_PATH)
    first_contact = findfirst(traj_df_full.contact .== 1)
    first_contact === nothing && error("No contact point found in AFM dataset.")
    return (
        ode = ode_data_full[:, first_contact:end],
        traj = traj_df_full[first_contact:end, :],
    )
end

function reconstruct_window_prediction(path::String, ode_post, traj_post)
    state = load_stage2_result_state(path)
    data = state.data
    p_vec = state.p_net_vec

    role = String(data.window_meta.role)
    title = window_title(role)
    start_idx = Int(data.window_meta.start)
    stop_idx = Int(data.window_meta.stop)

    ode_win = Float64.(ode_post[:, start_idx:stop_idx])
    times = Float64.(traj_post.t[start_idx:stop_idx])
    times_us = times .* T_US_SCALE

    num_hidden_layers = state.num_hidden_layers
    num_hidden_nodes = state.num_hidden_nodes
    nn_weights = unpack_nn_weights(p_vec, num_hidden_layers, num_hidden_nodes)
    nn_gain = state.nn_gain

    mech = [state.ks_hat, state.cs_hat]
    p = ComponentVector(mech=mech)

    u0 = vec(Float64.(ode_win[:, 1]))
    known_pars = original_parameters[1:8]
    prob = ODEProblem{false}(make_uode_func_oop(nn_weights, known_pars; nn_gain=nn_gain), u0, (times[1], times[end]), p)
    sol = solve(prob, Rosenbrock23(autodiff=false); saveat=times, abstol=1e-8, reltol=1e-8, maxiters=1_000_000)
    string(sol.retcode) == "Success" || error("ODE solve failed for $(basename(path)): retcode=$(sol.retcode)")
    uhat = Array(sol)

    return (
        source = state.source,
        checkpoint_epoch = state.checkpoint_epoch,
        title = title,
        role = role,
        times_us = times_us,
        x1_true = vec(ode_win[1, :]),
        x1_pred = vec(uhat[1, :]),
        x3_true = vec(ode_win[3, :]),
        x3_pred = vec(uhat[3, :]),
        x3dot_true = Float64.([true_x3dot_from_state(view(ode_win, :, j)) for j in axes(ode_win, 2)]),
        x3dot_pred = Float64.([predicted_x3dot_from_state(view(uhat, :, j), nn_weights, mech; nn_gain=nn_gain) for j in axes(uhat, 2)]),
    )
end

function build_panel(times_us, y_true, y_pred, title, ylabel)
    p = plot(
        times_us,
        y_true;
        linewidth = 2,
        color = :black,
        label = "true",
        title = title,
        xlabel = "time (μs)",
        ylabel = ylabel,
        legend = :topright,
    )
    plot!(p, times_us, y_pred; linewidth = 2, color = :crimson, linestyle = :dash, label = "predicted")
    return p
end

function run_one(result_dir::String, out_dir::String)
    paths = load_result_paths(result_dir)
    for p in paths
        isfile(p) || error("Missing result file: " * p)
    end

    post = post_contact_data()
    recs = [reconstruct_window_prediction(path, post.ode, post.traj) for path in paths]

    panels = Plots.Plot[]
    for rec in recs
        push!(panels, build_panel(rec.times_us, rec.x1_true, rec.x1_pred, rec.title, "x1"))
    end
    for rec in recs
        push!(panels, build_panel(rec.times_us, rec.x3_true, rec.x3_pred, rec.title, "x3"))
    end
    for rec in recs
        push!(panels, build_panel(rec.times_us, rec.x3dot_true, rec.x3dot_pred, rec.title, "x3dot"))
    end

    mkpath(out_dir)
    out_path = stage2light_checkpoint_plot_path(out_dir, DEFAULT_OUT_FILE, recs)
    fig = plot(panels...; layout = (3, 3), size = (1800, 1300), margin = 8 * Plots.mm)
    savefig(fig, out_path)
    println("Saved plot to: ", out_path)
    return out_path
end

function main()
    if should_use_dualrank_defaults(REPO_ROOT, ARGS)
        for role in dualrank_available_roles(REPO_ROOT)
            print_dualrank_banner(REPO_ROOT, role)
            run_one(dualrank_results_dir(REPO_ROOT, role), dualrank_visualization_dir(REPO_ROOT, role))
        end
        return
    end

    result_dir = RESULT_DIR
    out_dir = DEFAULT_OUT_DIR
    if length(ARGS) >= 2
        result_dir = normpath(abspath(ARGS[1]))
        out_dir = normpath(abspath(ARGS[2]))
    elseif length(ARGS) == 1
        arg1 = normpath(abspath(ARGS[1]))
        shard_hits = isdir(arg1) ? filter(path -> occursin(r"_p\d+\.jld$", basename(path)), readdir(arg1; join=true)) : String[]
        if !isempty(shard_hits)
            result_dir = arg1
        else
            out_dir = arg1
        end
    end
    run_one(result_dir, out_dir)
end

main()
