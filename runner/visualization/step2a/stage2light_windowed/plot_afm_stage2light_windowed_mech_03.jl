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
const DEFAULT_OUT_FILE = "afm_param_stage2light_windowed_03_mech_grid.png"
const KS_TRUE = 1.0e-1
const CS_TRUE = 2.4e-7
const N_ANNOTATION_POINTS = 5
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

function parse_mech_history(log_paths::Vector{String})
    epoch_re = r"Stage2 epoch (\d+) train="
    mech_re = r"mech: ks=([0-9eE+\-\.]+) \(err=([0-9eE+\-\.]+)%\)\s+cs=([0-9eE+\-\.]+) \(err=([0-9eE+\-\.]+)%\)"
    role_re = r"role=([^|]+)"
    label_re = r"label=([^|]+)"

    role = ""
    label = isempty(log_paths) ? "" : basename(first(log_paths))
    current_epoch = nothing

    ks_vals_by_epoch = Dict{Int, Float64}()
    ks_errs_by_epoch = Dict{Int, Float64}()
    cs_vals_by_epoch = Dict{Int, Float64}()
    cs_errs_by_epoch = Dict{Int, Float64}()

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

            m_mech = match(mech_re, line)
            if m_mech !== nothing && current_epoch !== nothing
                ks_vals_by_epoch[current_epoch] = parse(Float64, m_mech.captures[1])
                ks_errs_by_epoch[current_epoch] = parse(Float64, m_mech.captures[2])
                cs_vals_by_epoch[current_epoch] = parse(Float64, m_mech.captures[3])
                cs_errs_by_epoch[current_epoch] = parse(Float64, m_mech.captures[4])
            end
        end
    end

    epochs = sort(collect(keys(ks_vals_by_epoch)))
    ks_vals = [ks_vals_by_epoch[e] for e in epochs]
    ks_errs = [ks_errs_by_epoch[e] for e in epochs]
    cs_vals = [cs_vals_by_epoch[e] for e in epochs]
    cs_errs = [cs_errs_by_epoch[e] for e in epochs]

    return (
        paths = log_paths,
        role = role,
        label = label,
        title = shard_title(role, label),
        epochs = epochs,
        ks_vals = ks_vals,
        ks_errs = ks_errs,
        cs_vals = cs_vals,
        cs_errs = cs_errs,
    )
end

function annotation_indices(epochs::Vector{Int})
    isempty(epochs) && return Int[]
    target_stop = last(epochs) >= 500 ? 500 : last(epochs)
    targets = round.(Int, range(first(epochs), stop=target_stop, length=N_ANNOTATION_POINTS))
    idxs = Int[]
    for target in targets
        idx = argmin(abs.(epochs .- target))
        if !(idx in idxs)
            push!(idxs, idx)
        end
    end
    if length(idxs) < min(N_ANNOTATION_POINTS, length(epochs))
        fallback = round.(Int, range(1, stop=length(epochs), length=min(N_ANNOTATION_POINTS, length(epochs))))
        for idx in fallback
            if !(idx in idxs)
                push!(idxs, idx)
            end
        end
    end
    return sort(unique(idxs))
end

function annotate_error_points!(p, epochs, values, errs; color=:black)
    isempty(values) && return
    x_span = max(last(epochs) - first(epochs), 1)
    y_span = max(maximum(values) - minimum(values), maximum(abs.(values)) * 0.08, eps(Float64))
    x_pad = 0.04 * x_span
    y_pad = 0.08 * y_span

    for (j, idx) in enumerate(annotation_indices(epochs))
        x = epochs[idx]
        y = values[idx]
        err = errs[idx]
        scatter!(p, [x], [y]; color=color, marker=:circle, markersize=4, label=false)

        x_text = isodd(j) ? x + x_pad : x - x_pad
        y_text = isodd(j) ? y + y_pad : y - y_pad
        align = isodd(j) ? :left : :right
        annotate!(p, x_text, y_text, text(@sprintf("%.2f%%", err), 8, color, align))
    end
end

function build_subplot(rec, values, errs, truth; ylabel::String)
    if isempty(rec.epochs)
        return plot(
            title = rec.title * " | " * ylabel,
            xlabel = "epoch",
            ylabel = ylabel,
            xlims = (0, 500),
            legend = false,
            annotations = (250, 0.5, text("no epoch data", 10, :darkred, :center)),
        )
    end

    p = plot(
        rec.epochs,
        values;
        xlabel = "epoch",
        ylabel = ylabel,
        title = rec.title * " | " * ylabel,
        linewidth = 2,
        color = :black,
        label = ylabel,
        xlims = (0, 500),
        legend = :topright,
        xformatter = EPOCH_FORMATTER,
        yformatter = :scientific,
    )
    hline!(p, [truth]; color=:blue, linestyle=:dash, linewidth=2, label="$(ylabel)_true")
    annotate_error_points!(p, rec.epochs, values, errs; color=:darkred)
    return p
end

function run_one(log_dir::String, out_dir::String)
    log_series = windowed_default_log_series(log_dir; shard_count=3)
    for paths in log_series
        any(isfile, paths) || error("Missing shard log: " * join(paths, ", "))
    end

    recs = [parse_mech_history(paths) for paths in log_series]
    panels = Plots.Plot[]
    for rec in recs
        push!(panels, build_subplot(rec, rec.ks_vals, rec.ks_errs, KS_TRUE; ylabel="ks"))
    end
    for rec in recs
        push!(panels, build_subplot(rec, rec.cs_vals, rec.cs_errs, CS_TRUE; ylabel="cs"))
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
            " | ks_points=", length(rec.ks_vals),
            " | cs_points=", length(rec.cs_vals),
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
