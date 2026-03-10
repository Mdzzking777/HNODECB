ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Serialization
using DataFrames
using Plots

include(joinpath(@__DIR__, "..", "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_settings.jl"))
include(joinpath(@__DIR__, "..", "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_functions.jl"))

const DATA_DIR = joinpath(@__DIR__, "..", "datasets", "e0.0", "data")
const ODE_DATA_PATH = joinpath(DATA_DIR, "ode_data_afm_dmt_kv.jld")
const TRAJ_PATH = joinpath(DATA_DIR, "pert_df_afm_dmt_kv.jld")
const T_US_SCALE = 1.0e6
const CONTACT_EDGE_NUMBER = 1
const ZOOM_HALF_WIDTH_US = 5.0

ode_data = deserialize(ODE_DATA_PATH)
traj_df = deserialize(TRAJ_PATH)

times = collect(traj_df.t)
times_us = times .* T_US_SCALE
x1 = vec(ode_data[1, :])
x3 = vec(ode_data[3, :])

raw_delta = @. -dist - x1 + x3
delta = similar(raw_delta)
raw_f_hertz = fill(NaN, length(raw_delta))

coeff = (4.0 / 3.0) * Estar * sqrt(R)

for i in eachindex(raw_delta)
    delta[i] = softplus(raw_delta[i], adhesion_transition)
    if raw_delta[i] >= 0.0
        raw_f_hertz[i] = coeff * (raw_delta[i]^1.5)
    end
end

f_hertz_generated = coeff .* (delta .^ 1.5)

rising_edges = findall(i -> raw_delta[i - 1] < 0.0 && raw_delta[i] >= 0.0, 2:length(raw_delta))
if isempty(rising_edges)
    error("No non-contact to contact transition found in raw_delta.")
end

edge_idx = rising_edges[clamp(CONTACT_EDGE_NUMBER, 1, length(rising_edges))]
edge_time_us = times_us[edge_idx]
x_window = (edge_time_us - ZOOM_HALF_WIDTH_US, edge_time_us + ZOOM_HALF_WIDTH_US)

plot_dir = joinpath(@__DIR__, "..", "logs", "pics")
mkpath(plot_dir)
plot_path = joinpath(plot_dir, "afm_hertz_reference_signals_03.png")

p1 = plot(
    times_us,
    raw_f_hertz;
    label = "raw F_hertz",
    xlabel = "time (μs)",
    ylabel = "force",
    title = "Raw F_hertz around contact edge",
    linewidth = 2,
    xlims = x_window,
)
vline!(p1, [edge_time_us]; label = "contact edge", linestyle = :dash, color = :black)

p2 = plot(
    times_us,
    f_hertz_generated;
    label = "generated F_hertz",
    xlabel = "time (μs)",
    ylabel = "force",
    title = "Generated F_hertz around contact edge",
    linewidth = 2,
    xlims = x_window,
)
vline!(p2, [edge_time_us]; label = "contact edge", linestyle = :dash, color = :black)

p3 = plot(
    times_us,
    delta;
    label = "delta",
    xlabel = "time (μs)",
    ylabel = "indentation",
    title = "Smooth delta around contact edge",
    linewidth = 2,
    xlims = x_window,
)
vline!(p3, [edge_time_us]; label = "contact edge", linestyle = :dash, color = :black)

fig = plot(p1, p2, p3; layout = (3, 1), size = (1200, 900))
savefig(fig, plot_path)

println("Saved plot to: ", normpath(plot_path))
println("Contact edge #", CONTACT_EDGE_NUMBER, " at ", round(edge_time_us; digits = 3), " μs")
