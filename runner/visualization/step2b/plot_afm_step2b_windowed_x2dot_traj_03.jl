ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Printf
using Plots

include(joinpath(@__DIR__, "_step2b_windowed_common_03.jl"))

const DEFAULT_OUT_FILE = "afm_step2b_windowed_03_x2_x2dot_traj_grid.png"

function build_panel(times_us, y_true, y_pred, title, ylabel)
    p = plot(times_us, y_true; linewidth = 2, color = :black, label = "true", title = title, xlabel = "time (μs)", ylabel = ylabel, legend = :topright)
    plot!(p, times_us, y_pred; linewidth = 2, color = :crimson, linestyle = :dash, label = "predicted")
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
        push!(panels, build_panel(rec.times_us, rec.x2dot_true, rec.x2dot_pred, rec.title, "x2dot"))
    end
    for rec in recs
        push!(panels, build_panel(rec.times_us, vec(rec.ode_win[2, :]), vec(rec.uhat[2, :]), rec.title, "x2"))
    end

    mkpath(out_dir)
    out_path = joinpath(out_dir, DEFAULT_OUT_FILE)
    fig = plot(panels...; layout = (2, 3), size = (1800, 900), margin = 8 * Plots.mm)
    savefig(fig, out_path)
    println("Saved plot to: ", out_path)
end

main()
