#= Plot AFM DMT-KV time-domain signals (x, y, s). =#

cd(@__DIR__)

using Serialization, DataFrames, Plots

df = deserialize("e0.0/data/pert_df_afm_dmt_kv.jld")

t = df.t
x = df.x1
y = df.x3
s = df.s

# Convert units for readability
t_us = t .* 1e6
x_nm = x .* 1e9
y_nm = y .* 1e9
s_nm = s .* 1e9

plots_dir = joinpath("e0.0", "plots")
if !isdir(plots_dir)
    mkdir(plots_dir)
end

plt = plot(layout=(3, 1), size=(1000, 900), legend=false)
plot!(plt[1], t_us, x_nm, color=:blue, linewidth=0.8, title="AFM DMT-KV time series", ylabel="x (nm)")
plot!(plt[2], t_us, y_nm, color=:red, linewidth=0.8, ylabel="y (nm)")
plot!(plt[3], t_us, s_nm, color=:green, linewidth=0.8, ylabel="s (nm)", xlabel="time (us)")
hline!(plt[3], [0.0], color=:black, linestyle=:dash, linewidth=0.6)

savefig(plt, joinpath(plots_dir, "afm_time_series.png"))
savefig(plt, joinpath(plots_dir, "afm_time_series.pdf"))
