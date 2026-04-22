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
const DEFAULT_OUT_FILE = "afm_param_stage2light_windowed_03_loss_triptych.png"
const DEFAULT_SAMPLE_EVERY = 25
if !isdefined(@__MODULE__, :EPOCH_FORMATTER)
    const EPOCH_FORMATTER = x -> @sprintf("%d", round(Int, x))
end

include(joinpath(@__DIR__, "_dualrank_utils_03.jl"))

function log_tick_spec(y_min::Float64, y_max::Float64)
    lo = floor(Int, log10(y_min))
    hi = ceil(Int, log10(y_max))
    vals = Float64[]
    labels = String[]
    for k in lo:hi
        v = 10.0^k
        if y_min <= v <= y_max
            push!(vals, v)
            push!(labels, "1e" * string(k))
        end
    end
    return (vals, labels)
end

function parse_epoch_losses(log_paths::Vector{String})
    epoch_re = r"Stage2 epoch (\d+) train=([0-9eE+\-\.]+)"
    val_epoch_re = r"Stage2 val epoch (\d+) val=([0-9eE+\-\.]+|NaN|Inf|-Inf)"
    sanity_re = r"sanity loss=([0-9eE+\-\.]+)"
    role_re = r"role=([^|]+)"
    label_re = r"label=([^|]+)"
    done_re = r"done -- train=([0-9eE+\-\.]+)\s+val=([0-9eE+\-\.]+)"
    best_re = r"best: train=([0-9eE+\-\.]+)\s+val=([0-9eE+\-\.]+)"

    train_losses_by_epoch = Dict{Int, Float64}()
    val_losses_by_epoch = Dict{Int, Float64}()
    role = ""
    label = isempty(log_paths) ? "" : basename(first(log_paths))
    final_val = NaN
    sanity_loss = NaN

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
                train_losses_by_epoch[parse(Int, m_epoch.captures[1])] = parse(Float64, m_epoch.captures[2])
                continue
            end

            m_val_epoch = match(val_epoch_re, line)
            if m_val_epoch !== nothing
                val_epoch = parse(Int, m_val_epoch.captures[1])
                val_loss = tryparse(Float64, m_val_epoch.captures[2])
                if val_loss !== nothing && isfinite(val_loss)
                    val_losses_by_epoch[val_epoch] = val_loss
                end
                continue
            end

            m_sanity = match(sanity_re, line)
            if m_sanity !== nothing
                sanity_loss = parse(Float64, m_sanity.captures[1])
                continue
            end

            m_done = match(done_re, line)
            if m_done !== nothing
                final_val = parse(Float64, m_done.captures[2])
                continue
            end

            m_best = match(best_re, line)
            if m_best !== nothing
                final_val = parse(Float64, m_best.captures[2])
            end
        end
    end

    epochs = sort(collect(keys(train_losses_by_epoch)))
    train_losses = [train_losses_by_epoch[e] for e in epochs]
    val_epochs = sort(collect(keys(val_losses_by_epoch)))
    val_losses = [val_losses_by_epoch[e] for e in val_epochs]

    return (
        paths = log_paths,
        label = label,
        role = isempty(role) ? label : role,
        epochs = epochs,
        train_losses = train_losses,
        val_epochs = val_epochs,
        val_losses = val_losses,
        final_val = final_val,
        sanity_loss = sanity_loss,
    )
end

function sample_points(epochs::Vector{Int}, losses::Vector{Float64}, sample_every::Int)
    sample_epoch = Int[]
    sample_loss = Float64[]
    for (e, l) in zip(epochs, losses)
        if (e - 1) % sample_every == 0
            push!(sample_epoch, e)
            push!(sample_loss, l)
        end
    end
    if isempty(sample_epoch) && !isempty(epochs)
        push!(sample_epoch, epochs[end])
        push!(sample_loss, losses[end])
    elseif !isempty(epochs) && sample_epoch[end] != epochs[end]
        push!(sample_epoch, epochs[end])
        push!(sample_loss, losses[end])
    end
    return sample_epoch, sample_loss
end

function shard_title(rec)
    role = lowercase(strip(rec.role))
    if role == "first_contact"
        return "window1: right after first contact"
    elseif role == "max_x1_pp_change"
        return "window2: the most drastic region"
    elseif role == "tail_stable"
        return "window3: stable region at the end"
    end
    base = rec.role
    return isempty(base) ? rec.label : base
end

function build_subplot(rec; sample_every::Int)
    epochs = rec.epochs
    losses = rec.train_losses
    if isempty(epochs)
        return plot(
            title = shard_title(rec),
            xlabel = "epoch",
            ylabel = "loss",
            xlims = (0, 500),
            legend = false,
            annotations = (250, 0.5, text("no epoch data", 10, :darkred, :center)),
        )
    end
    sample_epoch, sample_loss = sample_points(epochs, losses, sample_every)
    sample_val_epoch, sample_val_loss = sample_points(rec.val_epochs, rec.val_losses, sample_every)
    y_values = Float64[]
    append!(y_values, filter(isfinite, sample_loss))
    append!(y_values, filter(isfinite, sample_val_loss))
    if isfinite(rec.final_val)
        push!(y_values, rec.final_val)
    end
    if isfinite(rec.sanity_loss)
        push!(y_values, rec.sanity_loss)
    end
    y_values = filter(x -> x > 0, y_values)
    y_min = isempty(y_values) ? 1e-12 : minimum(y_values) / 1.8
    y_max = isempty(y_values) ? 1e-6 : maximum(y_values) * 1.8
    y_ticks = log_tick_spec(y_min, y_max)

    p = plot(
        sample_epoch,
        sample_loss;
        xlabel = "epoch",
        ylabel = "loss",
        title = shard_title(rec),
        label = "train",
        linewidth = 2,
        marker = :circle,
        markersize = 4,
        color = :steelblue,
        xlims = (0, 500),
        ylims = (y_min, y_max),
        yticks = y_ticks,
        yscale = :log10,
        legend = :topright,
        xformatter = EPOCH_FORMATTER,
    )

    if isfinite(rec.sanity_loss)
        hline!(
            p,
            [rec.sanity_loss];
            label = "sanity",
            color = :blue,
            linestyle = :dash,
            linewidth = 2,
        )
        sanity_text_y = min(rec.sanity_loss * 1.6, y_max / 1.1)
        annotate!(
            p,
            490,
            sanity_text_y,
            text("sanity=" * @sprintf("%.2e", rec.sanity_loss), 8, :blue, :right),
        )
    end

    if !isempty(sample_val_epoch)
        plot!(
            p,
            sample_val_epoch,
            sample_val_loss;
            label = "validation",
            linewidth = 2,
            marker = :diamond,
            markersize = 4,
            color = :firebrick,
        )
    elseif isfinite(rec.final_val)
        scatter!(
            p,
            [maximum(epochs)],
            [rec.final_val];
            label = "final val",
            color = :firebrick,
            marker = :diamond,
            markersize = 6,
        )
    else
        annotate!(
            p,
            250,
            maximum(sample_loss) * 0.92,
            text("val history unavailable", 8, :darkred, :center),
        )
    end

    return p
end

function run_one(log_dir::String, out_dir::String, sample_every::Int)
    log_series = windowed_default_log_series(log_dir; shard_count=3)
    for paths in log_series
        any(isfile, paths) || error("Missing shard log: " * join(paths, ", "))
    end

    recs = [parse_epoch_losses(paths) for paths in log_series]
    subplots = [build_subplot(rec; sample_every=sample_every) for rec in recs]

    mkpath(out_dir)
    out_path = joinpath(out_dir, DEFAULT_OUT_FILE)
    fig = plot(subplots...; layout = (1, 3), size = (1800, 520), margin = 8 * Plots.mm)
    savefig(fig, out_path)

    println("Saved plot to: ", out_path)
    println("Sampled every ", sample_every, " epochs.")
    for (i, rec) in enumerate(recs)
        if isempty(rec.epochs)
            println("W", i, " [", shard_title(rec), "]: no epoch data found")
            continue
        end
        println(
            "W", i, " [", shard_title(rec), "]: epochs=",
            first(rec.epochs), "-", last(rec.epochs),
            " | sampled_points=", length(sample_points(rec.epochs, rec.train_losses, sample_every)[1]),
            " | val_points=", length(rec.val_epochs),
            " | sanity=", isfinite(rec.sanity_loss) ? @sprintf("%.4e", rec.sanity_loss) : "missing",
            " | final_val=", isfinite(rec.final_val) ? @sprintf("%.4e", rec.final_val) : "missing",
        )
    end
    return out_path
end

function main()
    sample_every = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : DEFAULT_SAMPLE_EVERY
    if should_use_dualrank_defaults(REPO_ROOT, ARGS)
        for role in dualrank_available_roles(REPO_ROOT)
            print_dualrank_banner(REPO_ROOT, role)
            run_one(dualrank_log_dir(REPO_ROOT, role), dualrank_visualization_dir(REPO_ROOT, role), sample_every)
        end
        return
    end

    log_dir = length(ARGS) >= 1 ? normpath(abspath(ARGS[1])) : DEFAULT_LOG_DIR
    out_dir = length(ARGS) >= 2 ? normpath(abspath(ARGS[2])) : DEFAULT_OUT_DIR
    run_one(log_dir, out_dir, sample_every)
end

main()
