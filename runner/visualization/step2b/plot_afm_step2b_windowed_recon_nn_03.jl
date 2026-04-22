ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Printf
using Plots

include(joinpath(@__DIR__, "_step2b_windowed_common_03.jl"))

const DEFAULT_OUT_FILE = "afm_step2b_windowed_03_recon_nn_grid.png"
const EPOCH_FORMATTER = x -> @sprintf("%d", round(Int, x))

function nn_axis_spec(values::Vector{Float64})
    max_val = isempty(values) ? 100.0 : maximum(values)
    top = max(50.0, ceil(max_val / 25.0) * 25.0)
    step = top <= 300 ? 25.0 : (top <= 600 ? 50.0 : 100.0)
    ticks = collect(0.0:step:top)
    return (0.0, top, ticks)
end

function parse_monitor_history(log_paths)
    epoch_re = r"Step2b (?:adam|lbfgs) epoch (\d+) train="
    rec_re = r"rec: x1=([0-9eE+\-\.]+)%\s+x3=([0-9eE+\-\.]+)%"
    nn_re = r"nn: F_contact err=([0-9eE+\-\.]+)%"
    series = log_paths isa AbstractString ? [String(log_paths)] : collect(String.(log_paths))
    role, label = step2b_log_role_label(series)

    rows = NamedTuple[]
    for log_path in series
        current_epoch = nothing
        pending_x1 = nothing
        pending_x3 = nothing
        isfile(log_path) || continue
        for line in eachline(log_path)
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
                push!(rows, (
                    epoch=current_epoch,
                    x1_rec=pending_x1,
                    x3_rec=pending_x3,
                    nn_err=parse(Float64, m_nn.captures[1]),
                ))
                pending_x1 = nothing
                pending_x3 = nothing
            end
        end
    end

    merged = step2b_merge_epoch_rows(rows)

    return (
        paths = series,
        role = role,
        label = label,
        epochs = Int[row.epoch for row in merged],
        x1_rec = Float64[row.x1_rec for row in merged],
        x3_rec = Float64[row.x3_rec for row in merged],
        nn_err = Float64[row.nn_err for row in merged],
    )
end

function build_recon_subplot(rec, xlims)
    isempty(rec.epochs) && return plot(
        title = step2b_window_title(rec.role, rec.label),
        xlabel = "epoch",
        ylabel = "reconstruction error (%)",
        xlims = xlims,
        legend = false,
        annotations = ((xlims[2] / 2), 0.5, text("no epoch data", 10, :darkred, :center)),
    )
    p = plot(rec.epochs, rec.x1_rec; xlabel = "epoch", ylabel = "reconstruction error (%)", title = step2b_window_title(rec.role, rec.label), linewidth = 2, color = :green, label = "x1", xlims = xlims, legend = :topright, xformatter = EPOCH_FORMATTER)
    plot!(p, rec.epochs, rec.x3_rec; linewidth = 2, color = :purple, label = "x3")
    return p
end

function build_nn_subplot(rec, xlims)
    isempty(rec.epochs) && return plot(
        title = step2b_window_title(rec.role, rec.label),
        xlabel = "epoch",
        ylabel = "NN F_contact error (%)",
        xlims = xlims,
        legend = false,
        annotations = ((xlims[2] / 2), 0.5, text("no epoch data", 10, :darkred, :center)),
    )
    y_lo, y_hi, y_ticks = nn_axis_spec(rec.nn_err)
    return plot(rec.epochs, rec.nn_err; xlabel = "epoch", ylabel = "NN F_contact error (%)", title = step2b_window_title(rec.role, rec.label), linewidth = 2, color = :black, label = "NN F_contact error", xlims = xlims, ylims = (y_lo, y_hi), yticks = y_ticks, legend = :topright, xformatter = EPOCH_FORMATTER, minorgrid = true)
end

function main()
    log_dir = length(ARGS) >= 1 ? normpath(abspath(ARGS[1])) : STEP2B_LOG_DIR
    out_dir = length(ARGS) >= 2 ? normpath(abspath(ARGS[2])) : STEP2B_OUT_DIR
    log_series = step2b_existing_log_series(log_dir)
    foreach(paths -> isempty(paths) && error("Missing shard log series in: " * log_dir), log_series)

    recs = [parse_monitor_history(paths) for paths in log_series]
    xlims = step2b_epoch_xlim(recs)
    panels = Plots.Plot[]
    for rec in recs
        push!(panels, build_recon_subplot(rec, xlims))
    end
    for rec in recs
        push!(panels, build_nn_subplot(rec, xlims))
    end

    mkpath(out_dir)
    out_path = joinpath(out_dir, DEFAULT_OUT_FILE)
    fig = plot(panels...; layout = (2, 3), size = (1800, 900), margin = 8 * Plots.mm)
    savefig(fig, out_path)
    println("Saved plot to: ", out_path)
end

main()
