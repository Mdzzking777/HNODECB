ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Printf
using Plots

include(joinpath(@__DIR__, "_step2b_windowed_common_03.jl"))

const DEFAULT_OUT_FILE = "afm_step2b_windowed_03_state_x3_x2dot_grid.png"
const EPOCH_FORMATTER = x -> @sprintf("%d", round(Int, x))

function parse_parts_history(log_paths)
    epoch_re = r"Step2b (?:adam|lbfgs) epoch (\d+) train="
    parts_re = r"parts: state=([0-9eE+\-\.]+)\s+x3=([0-9eE+\-\.]+)\s+\(u0=[^\)]*\)\s+x2dot=([0-9eE+\-\.]+)\s+x3_range=([0-9eE+\-\.]+)\s+fts_range=([0-9eE+\-\.]+)\s+cont=([0-9eE+\-\.]+)"
    series = log_paths isa AbstractString ? [String(log_paths)] : collect(String.(log_paths))
    role, label = step2b_log_role_label(series)

    rows = NamedTuple[]
    for log_path in series
        current_epoch = nothing
        isfile(log_path) || continue
        for line in eachline(log_path)
            m_epoch = match(epoch_re, line)
            if m_epoch !== nothing
                current_epoch = parse(Int, m_epoch.captures[1])
                continue
            end
            m_parts = match(parts_re, line)
            if m_parts !== nothing && current_epoch !== nothing
                push!(rows, (
                    epoch=current_epoch,
                    state_loss=parse(Float64, m_parts.captures[1]),
                    x3_loss=parse(Float64, m_parts.captures[2]),
                    x2dot_loss=parse(Float64, m_parts.captures[3]),
                ))
            end
        end
    end

    merged = step2b_merge_epoch_rows(rows)

    return (
        paths = series,
        role = role,
        label = label,
        epochs = Int[row.epoch for row in merged],
        state_loss = Float64[row.state_loss for row in merged],
        x3_loss = Float64[row.x3_loss for row in merged],
        x2dot_loss = Float64[row.x2dot_loss for row in merged],
    )
end

function positive_log_limits(values::Vector{Float64})
    vals = filter(>(0.0), values)
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

function empty_panel(title::String, ylabel::String, note::String, xlims)
    return plot(title = title, xlabel = "epoch", ylabel = ylabel, xlims = xlims, legend = false, xformatter = EPOCH_FORMATTER, annotations = ((xlims[2] / 2), 0.5, text(note, 10, :darkred, :center)))
end

function build_subplot(rec, values, ylabel::String, color, xlims)
    isempty(rec.epochs) && return empty_panel(step2b_window_title(rec.role, rec.label), ylabel, "no epoch data", xlims)
    yt = log_ticks(values)
    ylo, yhi = positive_log_limits(values)
    return plot(rec.epochs, values; xlabel = "epoch", ylabel = ylabel, title = step2b_window_title(rec.role, rec.label), linewidth = 2, color = color, label = ylabel, xlims = xlims, yscale = :log10, ylims = (ylo, yhi), yticks = yt, legend = :topright, xformatter = EPOCH_FORMATTER, minorgrid = true)
end

function main()
    log_dir = length(ARGS) >= 1 ? normpath(abspath(ARGS[1])) : STEP2B_LOG_DIR
    out_dir = length(ARGS) >= 2 ? normpath(abspath(ARGS[2])) : STEP2B_OUT_DIR
    log_series = step2b_existing_log_series(log_dir)
    foreach(paths -> isempty(paths) && error("Missing shard log series in: " * log_dir), log_series)

    recs = [parse_parts_history(paths) for paths in log_series]
    xlims = step2b_epoch_xlim(recs)
    panels = Plots.Plot[]
    for rec in recs
        push!(panels, build_subplot(rec, rec.state_loss, "state loss", :royalblue, xlims))
    end
    for rec in recs
        push!(panels, build_subplot(rec, rec.x3_loss, "x3 loss", :seagreen, xlims))
    end
    for rec in recs
        push!(panels, build_subplot(rec, rec.x2dot_loss, "x2dot loss", :darkorange, xlims))
    end

    mkpath(out_dir)
    out_path = joinpath(out_dir, DEFAULT_OUT_FILE)
    fig = plot(panels...; layout = (3, 3), size = (1800, 1320), margin = 8 * Plots.mm)
    savefig(fig, out_path)
    println("Saved plot to: ", out_path)
end

main()
