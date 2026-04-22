#=
Quick inspector for afm_param_stage2_02.jld
Prints best result and top-10 summary.
=#

cd(@__DIR__)

using Serialization, Printf

include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")

function percent_error_pct(est, truth)
  return 100.0 * abs(log10(est / truth))
end

path = "../step2a_hyperparameter_tuning/hyperparameter_tuning_second_stage/results_afm/afm_param_stage2_02.jld"
if !isfile(path)
  error("Stage2 file not found: " * path)
end

stage2 = deserialize(path)

println("=== Stage2 best ===")
println(stage2.best)

println("=== Stage2 top-10 by loss ===")
sorted = sort(stage2.results, by = r -> r.loss)
topn = min(10, length(sorted))
for i in 1:topn
  r = sorted[i]
  ks_err = percent_error_pct(r.ks, ks)
  cs_err = percent_error_pct(r.cs, cs)
  Estar_err = percent_error_pct(r.Estar, Estar)
  println("Rank ", i, " -- loss=", @sprintf("%.4e", r.loss),
    " | x1_rec=", @sprintf("%.2f", get(r, :x1_rec, NaN)), "% x3_rec=", @sprintf("%.2f", get(r, :x3_rec, NaN)), "%",
    " | state=", @sprintf("%.4e", get(r, :state, NaN)),
    " x2dot=", @sprintf("%.4e", get(r, :x2dot, NaN)),
    " x3_range=", @sprintf("%.4e", get(r, :x3_range, NaN)),
    " | ks=", @sprintf("%.3e", r.ks), " (true=", @sprintf("%.3e", ks), ", ", @sprintf("%.2f", ks_err), "%)",
    " cs=", @sprintf("%.3e", r.cs), " (true=", @sprintf("%.3e", cs), ", ", @sprintf("%.2f", cs_err), "%)",
    " Estar=", @sprintf("%.3e", r.Estar), " (true=", @sprintf("%.3e", Estar), ", ", @sprintf("%.2f", Estar_err), "%)")
end
