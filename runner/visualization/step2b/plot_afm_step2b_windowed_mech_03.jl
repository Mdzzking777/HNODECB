ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Printf
using Plots

include(joinpath(@__DIR__, "_step2b_windowed_common_03.jl"))

const DEFAULT_OUT_FILE = "afm_step2b_windowed_03_mech_grid.png"
const KS_TRUE = 1.0e-1
const CS_TRUE = 2.4e-7
const N_ANNOTATION_POINTS = 5
const EPOCH_FORMATTER = x -> @sprintf("%d", round(Int, x))

function parse_mech_history(log_paths)
    epoch_re = r"Step2b (?:adam|lbfgs) epoch (\d+) train="
    mech_re = r"mech: ks=([0-9eE+\-\.]+) \(err=([0-9eE+\-\.]+)%\)\s+cs=([0-9eE+\-\.]+) \(err=([0-9eE+\-\.]+)%\)"
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
            m_mech = match(mech_re, line)
            if m_mech !== nothing && current_epoch !== nothing
                push!(rows, (
                    epoch=current_epoch,
                    ks=parse(Float64, m_mech.captures[1]),
                    ks_err=parse(Float64, m_mech.captures[2]),
                    cs=parse(Float64, m_mech.captures[3]),
                    cs_err=parse(Float64, m_mech.captures[4]),
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
        ks_vals = Float64[row.ks for row in merged],
        ks_errs = Float64[row.ks_err for row in merged],
        cs_vals = Float64[row.cs for row in merged],
        cs_errs = Float64[row.cs_err for row in merged],
    )
end

function annotation_indices(epochs::Vector{Int})
    isempty(epochs) && return Int[]
    targets = round.(Int, range(first(epochs), stop = last(epochs), length = min(N_ANNOTATION_POINTS, length(epochs))))
    idxs = Int[argmin(abs.(epochs .- t)) for t in targets]
    return sort(unique(idxs))
end

function annotate_error_points!(p, epochs, values, errs; color = :black)
    isempty(values) && return
    x_span = max(last(epochs) - first(epochs), 1)
    y_span = max(maximum(values) - minimum(values), maximum(abs.(values)) * 0.08, eps(Float64))
    x_pad = 0.04 * x_span
    y_pad = 0.08 * y_span
    for (j, idx) in enumerate(annotation_indices(epochs))
        x = epochs[idx]
        y = values[idx]
        err = errs[idx]
        scatter!(p, [x], [y]; color = color, marker = :circle, markersize = 4, label = false)
        x_text = isodd(j) ? x + x_pad : x - x_pad
        y_text = isodd(j) ? y + y_pad : y - y_pad
        align = isodd(j) ? :left : :right
        annotate!(p, x_text, y_text, text(@sprintf("%.2f%%", err), 8, color, align))
    end
end

function build_subplot(rec, values, errs, truth; ylabel::String, xlims)
    isempty(rec.epochs) && return plot(
        title = step2b_window_title(rec.role, rec.label) * " | " * ylabel,
        xlabel = "epoch",
        ylabel = ylabel,
        xlims = xlims,
        legend = false,
        annotations = ((xlims[2] / 2), 0.5, text("no epoch data", 10, :darkred, :center)),
    )
    p = plot(
        rec.epochs,
        values;
        xlabel = "epoch",
        ylabel = ylabel,
        title = step2b_window_title(rec.role, rec.label) * " | " * ylabel,
        linewidth = 2,
        color = :black,
        label = ylabel,
        xlims = xlims,
        legend = :topright,
        xformatter = EPOCH_FORMATTER,
        yformatter = :scientific,
    )
    hline!(p, [truth]; color = :blue, linestyle = :dash, linewidth = 2, label = "$(ylabel)_true")
    annotate_error_points!(p, rec.epochs, values, errs; color = :darkred)
    return p
end

function main()
    log_dir = length(ARGS) >= 1 ? normpath(abspath(ARGS[1])) : STEP2B_LOG_DIR
    out_dir = length(ARGS) >= 2 ? normpath(abspath(ARGS[2])) : STEP2B_OUT_DIR
    log_series = step2b_existing_log_series(log_dir)
    foreach(paths -> isempty(paths) && error("Missing shard log series in: " * log_dir), log_series)

    recs = [parse_mech_history(paths) for paths in log_series]
    xlims = step2b_epoch_xlim(recs)
    panels = Plots.Plot[]
    for rec in recs
        push!(panels, build_subplot(rec, rec.ks_vals, rec.ks_errs, KS_TRUE; ylabel = "ks", xlims = xlims))
    end
    for rec in recs
        push!(panels, build_subplot(rec, rec.cs_vals, rec.cs_errs, CS_TRUE; ylabel = "cs", xlims = xlims))
    end

    mkpath(out_dir)
    out_path = joinpath(out_dir, DEFAULT_OUT_FILE)
    fig = plot(panels...; layout = (2, 3), size = (1800, 900), margin = 8 * Plots.mm)
    savefig(fig, out_path)
    println("Saved plot to: ", out_path)
end

main()
