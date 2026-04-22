ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Printf
using Plots

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
const DEFAULT_LOG_DIR = normpath(joinpath(REPO_ROOT, "logs", "stage2_step2a", "local", "windowed", "window_per_shard"))
const DEFAULT_OUT_DIR = normpath(joinpath(REPO_ROOT, "logs", "stage2_step2a", "local", "windowed", "visualization"))
const DEFAULT_OUT_FILE = "afm_param_stage2light_windowed_03_recon_nn_grid.png"
if !isdefined(@__MODULE__, :EPOCH_FORMATTER)
    const EPOCH_FORMATTER = x -> @sprintf("%d", round(Int, x))
end

include(joinpath(@__DIR__, "_dualrank_utils_03.jl"))

function nn_axis_spec(values::Vector{Float64})
    max_val = isempty(values) ? 100.0 : maximum(values)
    top = max(50.0, ceil(max_val / 25.0) * 25.0)
    step =
        top <= 150 ? 25.0 :
        top <= 300 ? 25.0 :
        top <= 600 ? 50.0 :
        100.0
    ticks = collect(0.0:step:top)
    return (0.0, top, ticks)
end

function shard_title(role::AbstractString, label::AbstractString)
    role_norm = lowercase(strip(role))
    if role_norm == "first_contact"
        return "window1: right after first contact"
    elseif role_norm == "max_x1_pp_change"
        return "window2: the most drastic region"
    elseif role_norm == "tail_stable"
        return "window3: stable region at the end"
    end
    return isempty(role) ? label : role
end

function parse_monitor_history(log_paths::Vector{String})
    epoch_re = r"Stage2 epoch (\d+) train="
    rec_re = r"rec: x1=([0-9eE+\-\.]+)%\s+x3=([0-9eE+\-\.]+)%"
    nn_re = r"nn: F_contact err=([0-9eE+\-\.]+)%"
    role_re = r"role=([^|]+)"
    label_re = r"label=([^|]+)"

    role = ""
    label = isempty(log_paths) ? "" : basename(first(log_paths))
    current_epoch = nothing

    x1_by_epoch = Dict{Int, Float64}()
    x3_by_epoch = Dict{Int, Float64}()
    nn_by_epoch = Dict{Int, Float64}()

    pending_x1 = nothing
    pending_x3 = nothing

    for log_path in log_paths
        isfile(log_path) || continue
        for line in eachline(log_path)
            if isempty(role)
                m = match(role_re, line)
                if m !== nothing
                    role = strip(m.captures[1])
                end
            end
            if label == basename(first(log_paths))
                m = match(label_re, line)
                if m !== nothing
                    label = strip(m.captures[1])
                end
            end

            m_epoch = match(epoch_re, line)
            if m_epoch !== nothing
                current_epoch = parse(Int, m_epoch.captures[1])
                pending_x1 = nothing
                pending_x3 = nothing
                continue
            end

            m_rec = match(rec_re, line)
            if m_rec !== nothing && current_epoch !== nothing
                pending_x1 = parse(Float64, m_rec.captures[1])
                pending_x3 = parse(Float64, m_rec.captures[2])
                continue
            end

            m_nn = match(nn_re, line)
            if m_nn !== nothing && current_epoch !== nothing && pending_x1 !== nothing && pending_x3 !== nothing
                x1_by_epoch[current_epoch] = pending_x1
                x3_by_epoch[current_epoch] = pending_x3
                nn_by_epoch[current_epoch] = parse(Float64, m_nn.captures[1])
                pending_x1 = nothing
                pending_x3 = nothing
            end
        end
    end

    epochs = sort(collect(keys(nn_by_epoch)))
    x1_rec = [x1_by_epoch[e] for e in epochs]
    x3_rec = [x3_by_epoch[e] for e in epochs]
    nn_err = [nn_by_epoch[e] for e in epochs]

    return (
        paths = log_paths,
        role = role,
        label = label,
        title = shard_title(role, label),
        epochs = epochs,
        x1_rec = x1_rec,
        x3_rec = x3_rec,
        nn_err = nn_err,
    )
end

function build_recon_subplot(rec)
    if isempty(rec.epochs)
        return plot(
            title = rec.title,
            xlabel = "epoch",
            ylabel = "reconstruction error (%)",
            xlims = (0, 500),
            legend = false,
            annotations = (250, 0.5, text("no epoch data", 10, :darkred, :center)),
        )
    end

    p = plot(
        rec.epochs,
        rec.x1_rec;
        xlabel = "epoch",
        ylabel = "reconstruction error (%)",
        title = rec.title,
        linewidth = 2,
        color = :green,
        label = "x1",
        xlims = (0, 500),
        legend = :topright,
        xformatter = EPOCH_FORMATTER,
    )
    plot!(
        p,
        rec.epochs,
        rec.x3_rec;
        linewidth = 2,
        color = :purple,
        label = "x3",
    )
    return p
end

function build_nn_subplot(rec)
    if isempty(rec.epochs)
        return plot(
            title = rec.title,
            xlabel = "epoch",
            ylabel = "NN F_contact error (%)",
            xlims = (0, 500),
            legend = false,
            annotations = (250, 0.5, text("no epoch data", 10, :darkred, :center)),
        )
    end

    y_lo, y_hi, y_ticks = nn_axis_spec(rec.nn_err)
    return plot(
        rec.epochs,
        rec.nn_err;
        xlabel = "epoch",
        ylabel = "NN F_contact error (%)",
        title = rec.title,
        linewidth = 2,
        color = :black,
        label = "NN F_contact error",
        xlims = (0, 500),
        ylims = (y_lo, y_hi),
        yticks = y_ticks,
        legend = :topright,
        xformatter = EPOCH_FORMATTER,
        minorgrid = true,
    )
end

function run_one(log_dir::String, out_dir::String)
    log_series = windowed_default_log_series(log_dir; shard_count=3)
    for paths in log_series
        any(isfile, paths) || error("Missing shard log: " * join(paths, ", "))
    end

    recs = [parse_monitor_history(paths) for paths in log_series]
    panels = Plots.Plot[]
    for rec in recs
        push!(panels, build_recon_subplot(rec))
    end
    for rec in recs
        push!(panels, build_nn_subplot(rec))
    end

    mkpath(out_dir)
    out_path = joinpath(out_dir, DEFAULT_OUT_FILE)
    fig = plot(panels...; layout=(2, 3), size=(1800, 900), margin=8 * Plots.mm)
    savefig(fig, out_path)

    println("Saved plot to: ", out_path)
    for (i, rec) in enumerate(recs)
        println(
            "W", i, " [", rec.title, "]: epochs=",
            isempty(rec.epochs) ? "none" : string(first(rec.epochs), "-", last(rec.epochs)),
            " | points=", length(rec.epochs),
        )
    end
    return out_path
end

function main()
    if should_use_dualrank_defaults(REPO_ROOT, ARGS)
        for role in dualrank_available_roles(REPO_ROOT)
            print_dualrank_banner(REPO_ROOT, role)
            run_one(dualrank_log_dir(REPO_ROOT, role), dualrank_visualization_dir(REPO_ROOT, role))
        end
        return
    end

    log_dir = length(ARGS) >= 1 ? normpath(abspath(ARGS[1])) : DEFAULT_LOG_DIR
    out_dir = length(ARGS) >= 2 ? normpath(abspath(ARGS[2])) : DEFAULT_OUT_DIR
    run_one(log_dir, out_dir)
end

main()
