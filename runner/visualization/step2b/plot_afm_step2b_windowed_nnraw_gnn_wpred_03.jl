ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Printf
using Plots

include(joinpath(@__DIR__, "_step2b_windowed_common_03.jl"))

const DEFAULT_OUT_FILE = "afm_step2b_windowed_03_nnraw_wpred_grid.png"

function build_scalar_panel(times_us, values, title, ylabel; color = :black, style = :solid, label = ylabel, ylim_override = nothing)
    p = plot(times_us, values; linewidth = 2, color = color, linestyle = style, label = label, title = title, xlabel = "time (μs)", ylabel = ylabel, legend = :topright)
    ylim_override !== nothing && plot!(p; ylims = ylim_override)
    return p
end

function build_gate_panel(rec)
    p = plot(rec.times_us, rec.w_true; linewidth = 2, color = :royalblue, label = "gate_true", title = rec.title, xlabel = "time (μs)", ylabel = "gate", legend = :topright)
    plot!(p, rec.times_us, rec.w_pred; linewidth = 2, color = :seagreen, linestyle = :dash, label = "w_pred")
    return p
end

function main()
    out_dir = length(ARGS) >= 1 ? normpath(abspath(ARGS[1])) : STEP2B_OUT_DIR
    paths = step2b_default_result_paths()
    foreach(path -> isfile(path) || error("Missing result file: " * path), paths)

    post = step2b_post_contact_data()
    recs = [step2b_window_record(path, post.ode, post.traj) for path in paths]

    panels = Plots.Plot[]
    for rec in recs
        push!(panels, build_scalar_panel(rec.times_us, rec.nn_raw, rec.title, "NN_raw"; color = :crimson, label = "NN_raw"))
    end
    for rec in recs
        push!(panels, build_scalar_panel(rec.times_us, rec.fcontact_true, rec.title, "F_contact true"; color = :black, label = "F_contact"))
    end
    for rec in recs
        push!(panels, build_gate_panel(rec))
    end

    mkpath(out_dir)
    out_path = joinpath(out_dir, DEFAULT_OUT_FILE)
    fig = plot(panels...; layout = (3, 3), size = (1800, 1240), margin = 8 * Plots.mm)
    savefig(fig, out_path)
    println("Saved plot to: ", out_path)
end

main()
