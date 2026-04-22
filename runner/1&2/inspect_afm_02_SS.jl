#=
Quick inspector for res_afm_02/afm_02.jld
Prints best result and top-5 summary.
=#

cd(@__DIR__)

using Serialization, Printf

include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")

function percent_error_pct(est, truth)
  return 100.0 * abs(log10(est / truth))
end

path = "../step2b_model_trainer/res_afm_02/afm_02.jld"
if !isfile(path)
  error("Result file not found: " * path)
end

payload = deserialize(path)

results = haskey(payload, :results) ? payload.results : payload

println("=== AFM 02 best ===")
println(results)

if haskey(payload, :sanity)
  sanity = payload.sanity
  println("=== Sanity check ===")
  println("data: loss=", sanity.data.loss, " comps=", sanity.data.comps)
  println("integrated: loss=", sanity.integrated.loss, " comps=", sanity.integrated.comps)
  if haskey(sanity, :ms_oracle)
    println("ms_oracle: loss=", sanity.ms_oracle.loss, " comps=", sanity.ms_oracle.comps)
  end
  if haskey(sanity, :ms_model)
    println("ms_model: loss=", sanity.ms_model.loss, " comps=", sanity.ms_model.comps)
  end
end

if haskey(payload, :run_summaries)
  println("=== Run summaries ===")
  for s in payload.run_summaries
    println("Run ", s.run_id, " train_loss=", s.best_loss, " val_loss=", s.validation_loss,
      " | ks=", @sprintf("%.3e", s.ks), " (", @sprintf("%.2f", s.ks_err), "%)",
      " cs=", @sprintf("%.3e", s.cs), " (", @sprintf("%.2f", s.cs_err), "%)",
      " Estar=", @sprintf("%.3e", s.Estar), " (", @sprintf("%.2f", s.Estar_err), "%)",
      " | x1_rec=", @sprintf("%.2f", s.x1_rec), "% x3_rec=", @sprintf("%.2f", s.x3_rec), "%",
      " | initial guess ks0=", @sprintf("%.3e", s.ks0),
      " cs0=", @sprintf("%.3e", s.cs0), " Estar0=", @sprintf("%.3e", s.Estar0))
  end
end

if !isempty(results) && results isa AbstractVector && haskey(results[1], :loss)
  println("=== AFM 02 top-5 by loss ===")
  sorted = sort(results, by = r -> r.loss)
  topn = min(5, length(sorted))
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
end
