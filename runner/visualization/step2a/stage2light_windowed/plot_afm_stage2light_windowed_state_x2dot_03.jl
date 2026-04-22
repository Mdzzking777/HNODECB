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
const DEFAULT_OUT_FILE = "afm_param_stage2light_windowed_03_x1x2state_x2dot_grid.png"
if !isdefined(@__MODULE__, :EPOCH_FORMATTER)
    const EPOCH_FORMATTER = x -> @sprintf("%d", round(Int, x))
end

include(joinpath(@__DIR__, "_dualrank_utils_03.jl"))

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

function parse_parts_history(log_paths::Vector{String})
    epoch_re = r"Stage2 epoch (\d+) train="
    parts_re_new = r"parts: state=([0-9eE+\-\.]+)\s+x1_state=([0-9eE+\-\.]+)\s+x2_state=([0-9eE+\-\.]+)\s+x2dot=([0-9eE+\-\.]+)"
    parts_re_old = r"parts: state=([0-9eE+\-\.]+)\s+x2dot=([0-9eE+\-\.]+)"
    role_re = r"role=([^|]+)"
    label_re = r"label=([^|]+)"

    role = ""
    label = isempty(log_paths) ? "" : basename(first(log_paths))
    current_epoch = nothing

    state_by_epoch = Dict{Int, Float64}()
    x1_by_epoch = Dict{Int, Float64}()
    x2_by_epoch = Dict{Int, Float64}()
    x2dot_by_epoch = Dict{Int, Float64}()

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
                continue
            end

            m_parts = match(parts_re_new, line)
            if m_parts !== nothing && current_epoch !== nothing
                state_by_epoch[current_epoch] = parse(Float64, m_parts.captures[1])
                x1_by_epoch[current_epoch] = parse(Float64, m_parts.captures[2])
                x2_by_epoch[current_epoch] = parse(Float64, m_parts.captures[3])
                x2dot_by_epoch[current_epoch] = parse(Float64, m_parts.captures[4])
                continue
            end

            m_parts_old = match(parts_re_old, line)
            if m_parts_old !== nothing && current_epoch !== nothing
                state_by_epoch[current_epoch] = parse(Float64, m_parts_old.captures[1])
                x2dot_by_epoch[current_epoch] = parse(Float64, m_parts_old.captures[2])
            end
        end
    end

    epochs = sort!(collect(union(keys(state_by_epoch), keys(x1_by_epoch), keys(x2_by_epoch), keys(x2dot_by_epoch))))
    state_loss = [get(state_by_epoch, e, NaN) for e in epochs]
    x1_state_loss = [get(x1_by_epoch, e, NaN) for e in epochs]
    x2_state_loss = [get(x2_by_epoch, e, NaN) for e in epochs]
    x2dot_loss = [get(x2dot_by_epoch, e, NaN) for e in epochs]

    return (
        paths = log_paths,
        role = role,
        label = label,
        title = shard_title(role, label),
        epochs = epochs,
        state_loss = state_loss,
        x1_state_loss = x1_state_loss,
        x2_state_loss = x2_state_loss,
        x2dot_loss = x2dot_loss,
    )
end

function positive_log_limits(values::Vector{Float64})
    if isempty(values)
        return (1e-12, 1.0)
    end
    vals = filter(x -> isfinite(x) && x > 0.0, values)
    isempty(vals) && return (1e-12, 1.0)
    ylo = 10.0^floor(log10(minimum(vals)))
    yhi = 10.0^ceil(log10(maximum(vals)))
    ylo == yhi && (yhi *= 10.0)
    return (ylo, yhi)
end

function log_ticks(values::Vector{Float64})
    ylo, yhi = positive_log_limits(values)
    p_lo = floor(Int, log10(ylo))
    p_hi = ceil(Int, log10(yhi))
    ticks = Float64[10.0^p for p in p_lo:p_hi]
    labels = ["1e$(Int(round(log10(t))))" for t in ticks]
    return (ticks, labels)
end

function finite_epoch_series(epochs::Vector{Int}, values::Vector{Float64})
    mask = isfinite.(values)
    return epochs[mask], values[mask]
end

function empty_panel(title::String, ylabel::String, note::String)
    return plot(
        title = title,
        xlabel = "epoch",
        ylabel = ylabel,
        xlims = (0, 500),
        legend = false,
        xformatter = EPOCH_FORMATTER,
        annotations = (250, 0.5, text(note, 10, :darkred, :center)),
    )
end

function build_x1state_subplot(rec)
    epochs, values = finite_epoch_series(rec.epochs, rec.x1_state_loss)
    isempty(values) && return empty_panel(rec.title, "x1 state loss", "no x1/x2 split in log")
    yt = log_ticks(values)
    ylo, yhi = positive_log_limits(values)
    return plot(
        epochs,
        values;
        xlabel = "epoch",
        ylabel = "x1 state loss",
        title = rec.title,
        linewidth = 2,
        color = :seagreen,
        label = "x1 state",
        xlims = (0, 500),
        yscale = :log10,
        ylims = (ylo, yhi),
        yticks = yt,
        legend = :topright,
        xformatter = EPOCH_FORMATTER,
        minorgrid = true,
    )
end

function build_x2state_subplot(rec)
    epochs, values = finite_epoch_series(rec.epochs, rec.x2_state_loss)
    isempty(values) && return empty_panel(rec.title, "x2 state loss", "no x1/x2 split in log")
    yt = log_ticks(values)
    ylo, yhi = positive_log_limits(values)
    return plot(
        epochs,
        values;
        xlabel = "epoch",
        ylabel = "x2 state loss",
        title = rec.title,
        linewidth = 2,
        color = :royalblue,
        label = "x2 state",
        xlims = (0, 500),
        yscale = :log10,
        ylims = (ylo, yhi),
        yticks = yt,
        legend = :topright,
        xformatter = EPOCH_FORMATTER,
        minorgrid = true,
    )
end

function build_x2dot_subplot(rec)
    epochs, values = finite_epoch_series(rec.epochs, rec.x2dot_loss)
    isempty(values) && return empty_panel(rec.title, "x2dot loss", "no epoch data")
    yt = log_ticks(values)
    ylo, yhi = positive_log_limits(values)
    return plot(
        epochs,
        values;
        xlabel = "epoch",
        ylabel = "x2dot loss",
        title = rec.title,
        linewidth = 2,
        color = :darkorange,
        label = "x2dot",
        xlims = (0, 500),
        yscale = :log10,
        ylims = (ylo, yhi),
        yticks = yt,
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

    recs = [parse_parts_history(paths) for paths in log_series]
    panels = Plots.Plot[]
    for rec in recs
        push!(panels, build_x1state_subplot(rec))
    end
    for rec in recs
        push!(panels, build_x2state_subplot(rec))
    end
    for rec in recs
        push!(panels, build_x2dot_subplot(rec))
    end

    mkpath(out_dir)
    out_path = joinpath(out_dir, DEFAULT_OUT_FILE)
    fig = plot(panels...; layout=(3, 3), size=(1800, 1320), margin=8 * Plots.mm)
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
