ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Printf
using Plots

include(joinpath(@__DIR__, "_step2b_windowed_common_03.jl"))

const DEFAULT_OUT_FILE = "afm_step2b_windowed_03_loss_triptych.png"
const DEFAULT_SAMPLE_EVERY = 25
const EPOCH_FORMATTER = x -> @sprintf("%d", round(Int, x))

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

function parse_epoch_losses(log_paths)
    epoch_re = r"Step2b (?:adam|lbfgs) epoch (\d+) train=([0-9eE+\-\.]+)"
    val_re = r"grad_norm=.* val=([0-9eE+\-\.]+|NaN|Inf|-Inf)"

    series = log_paths isa AbstractString ? [String(log_paths)] : collect(String.(log_paths))
    role, label = step2b_log_role_label(series)
    train_rows = NamedTuple[]
    val_rows = NamedTuple[]

    for log_path in series
        current_epoch = nothing
        isfile(log_path) || continue
        for line in eachline(log_path)
            m_epoch = match(epoch_re, line)
            if m_epoch !== nothing
                current_epoch = parse(Int, m_epoch.captures[1])
                push!(train_rows, (epoch=current_epoch, loss=parse(Float64, m_epoch.captures[2])))
                continue
            end

            m_val = match(val_re, line)
            if m_val !== nothing && current_epoch !== nothing
                val = tryparse(Float64, m_val.captures[1])
                if val !== nothing && isfinite(val)
                    push!(val_rows, (epoch=current_epoch, loss=val))
                end
            end
        end
    end

    train_merged = step2b_merge_epoch_rows(train_rows)
    val_merged = step2b_merge_epoch_rows(val_rows)

    return (
        paths = series,
        label = label,
        role = isempty(role) ? label : role,
        epochs = Int[row.epoch for row in train_merged],
        train_losses = Float64[row.loss for row in train_merged],
        val_epochs = Int[row.epoch for row in val_merged],
        val_losses = Float64[row.loss for row in val_merged],
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

function build_subplot(rec; sample_every::Int, xlims)
    isempty(rec.epochs) && return plot(
        title = step2b_window_title(rec.role, rec.label),
        xlabel = "epoch",
        ylabel = "loss",
        xlims = xlims,
        legend = false,
        annotations = ((xlims[2] / 2), 0.5, text("no epoch data", 10, :darkred, :center)),
    )

    sample_epoch, sample_loss = sample_points(rec.epochs, rec.train_losses, sample_every)
    sample_val_epoch, sample_val_loss = sample_points(rec.val_epochs, rec.val_losses, sample_every)

    y_values = filter(x -> isfinite(x) && x > 0, vcat(sample_loss, sample_val_loss))
    y_min = isempty(y_values) ? 1e-12 : minimum(y_values) / 1.8
    y_max = isempty(y_values) ? 1e-6 : maximum(y_values) * 1.8
    p = plot(
        sample_epoch,
        sample_loss;
        xlabel = "epoch",
        ylabel = "loss",
        title = step2b_window_title(rec.role, rec.label),
        label = "train",
        linewidth = 2,
        marker = :circle,
        markersize = 4,
        color = :steelblue,
        xlims = xlims,
        ylims = (y_min, y_max),
        yticks = log_tick_spec(y_min, y_max),
        yscale = :log10,
        legend = :topright,
        xformatter = EPOCH_FORMATTER,
    )
    if !isempty(sample_val_epoch)
        plot!(p, sample_val_epoch, sample_val_loss; label = "validation", linewidth = 2, marker = :diamond, markersize = 4, color = :firebrick)
    else
        annotate!(p, xlims[2] * 0.55, y_max / 2, text("val history unavailable", 8, :darkred, :center))
    end
    return p
end

function main()
    log_dir = length(ARGS) >= 1 ? normpath(abspath(ARGS[1])) : STEP2B_LOG_DIR
    out_dir = length(ARGS) >= 2 ? normpath(abspath(ARGS[2])) : STEP2B_OUT_DIR
    log_series = step2b_existing_log_series(log_dir)
    foreach(paths -> isempty(paths) && error("Missing shard log series in: " * log_dir), log_series)

    recs = [parse_epoch_losses(paths) for paths in log_series]
    xlims = step2b_epoch_xlim(recs)
    panels = [build_subplot(rec; sample_every = DEFAULT_SAMPLE_EVERY, xlims = xlims) for rec in recs]

    mkpath(out_dir)
    out_path = joinpath(out_dir, DEFAULT_OUT_FILE)
    fig = plot(panels...; layout = (1, 3), size = (1800, 520), margin = 8 * Plots.mm)
    savefig(fig, out_path)
    println("Saved plot to: ", out_path)
end

main()
