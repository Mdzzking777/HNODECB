ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Printf
using Plots

include(joinpath(@__DIR__, "_step2b_windowed_common_03.jl"))

const DEFAULT_OUT_FILE = "afm_step2b_windowed_03_fcontact_grid.png"

function build_panel(rec)
    area_true = step2b_trapz_integral(rec.times, rec.fcontact_true)
    area_pred = step2b_trapz_integral(rec.times, rec.fcontact_pred)
    area_ratio = area_true == 0.0 ? NaN : area_pred / area_true
    overlap = step2b_overlap_ratio(rec.times, rec.fcontact_true, rec.fcontact_pred)

    p = plot(rec.times_us, rec.fcontact_true; linewidth = 2, color = :black, label = "F_contact true", title = rec.title, xlabel = "time (μs)", ylabel = "force (N)", legend = :topright)
    plot!(p, rec.times_us, rec.fcontact_pred; linewidth = 2, color = :crimson, linestyle = :dash, label = "NN raw * gate_pred")
    x_annot = rec.times_us[1] + 0.04 * (rec.times_us[end] - rec.times_us[1])
    y_max = max(maximum(rec.fcontact_true), maximum(rec.fcontact_pred))
    annotate!(p, x_annot, 0.90 * y_max, text(@sprintf("area ratio = %.3f", area_ratio), 10, :black, :left))
    annotate!(p, x_annot, 0.80 * y_max, text(@sprintf("overlap = %.3f", overlap), 10, :black, :left))
    return p
end

function main()
    out_dir = length(ARGS) >= 1 ? normpath(abspath(ARGS[1])) : STEP2B_OUT_DIR
    paths = step2b_default_result_paths()
    foreach(path -> isfile(path) || error("Missing result file: " * path), paths)

    post = step2b_post_contact_data()
    recs = [step2b_window_record(path, post.ode, post.traj) for path in paths]
    panels = [build_panel(rec) for rec in recs]

    mkpath(out_dir)
    out_path = joinpath(out_dir, DEFAULT_OUT_FILE)
    fig = plot(panels...; layout = (1, 3), size = (1800, 520), margin = 8 * Plots.mm)
    savefig(fig, out_path)
    println("Saved plot to: ", out_path)
end

main()
