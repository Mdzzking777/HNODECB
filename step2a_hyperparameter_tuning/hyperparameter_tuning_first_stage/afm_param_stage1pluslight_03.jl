#=
Stage1pluslight wrapper for AFM scenario 03.
Standalone joint search:
- architecture screening uses shared ks/cs draws across architectures
- main search jointly randomizes ks/cs and NN initialization
- exports top-k standalone candidates for Stage2
=#

cd(@__DIR__)

function stage1pluslight_candidate_index()
  raw = strip(get(ENV, "HNODECB_STAGE1PLUSLIGHT_CANDIDATE", "1"))
  idx = tryparse(Int, raw)
  if idx === nothing || idx < 1
    error("HNODECB_STAGE1PLUSLIGHT_CANDIDATE must be a positive integer, got: " * raw)
  end
  return idx
end

if !haskey(ENV, "HNODECB_STAGE1_VARIANT")
  ENV["HNODECB_STAGE1_VARIANT"] = "stage1pluslight"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_MODE")
  ENV["HNODECB_STAGE1PLUS_MODE"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUSLIGHT_MODE")
  ENV["HNODECB_STAGE1PLUSLIGHT_MODE"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1_RESULT_STEM")
  ENV["HNODECB_STAGE1_RESULT_STEM"] = "afm_param_stage1pluslight_03"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE")
  ENV["HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE"] = "8000"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_FINAL_TOPK")
  ENV["HNODECB_STAGE1PLUS_FINAL_TOPK"] = "10"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ZERO_NN")
  ENV["HNODECB_STAGE1PLUS_ZERO_NN"] = "0"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_GNN_ENABLE")
  ENV["HNODECB_STAGE1PLUS_GNN_ENABLE"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_GNN_QUANTILE")
  ENV["HNODECB_STAGE1PLUS_GNN_QUANTILE"] = "0.95"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_GNN_EPS")
  ENV["HNODECB_STAGE1PLUS_GNN_EPS"] = "1e-30"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_GNN_CONTACT_FLOOR")
  ENV["HNODECB_STAGE1PLUS_GNN_CONTACT_FLOOR"] = "0.5"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH")
  ENV["HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH"] = "60"
end

include("afm_param_stage1_03.jl")
