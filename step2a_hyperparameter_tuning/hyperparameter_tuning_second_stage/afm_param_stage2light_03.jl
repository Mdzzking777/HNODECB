#=
Stage2light wrapper for AFM scenario 03.
Same Stage2 pipeline, but optimizes only one candidate rank (default 6) with a single shard.
=#

cd(@__DIR__)

function stage2light_candidate_index()
  raw = strip(get(ENV, "HNODECB_STAGE2LIGHT_CANDIDATE", "6"))
  idx = tryparse(Int, raw)
  if idx === nothing || idx < 1
    error("HNODECB_STAGE2LIGHT_CANDIDATE must be a positive integer, got: " * raw)
  end
  return idx
end

if !haskey(ENV, "HNODECB_STAGE2_SHARD_COUNT")
  ENV["HNODECB_STAGE2_SHARD_COUNT"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_INPUT_TOPK")
  ENV["HNODECB_STAGE2_INPUT_TOPK"] = "9"
end
if !haskey(ENV, "HNODECB_STAGE2_CANDIDATE_INDICES")
  ENV["HNODECB_STAGE2_CANDIDATE_INDICES"] = string(stage2light_candidate_index())
end
if !haskey(ENV, "HNODECB_STAGE2_FINAL_TOPK")
  ENV["HNODECB_STAGE2_FINAL_TOPK"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_RESULT_BASENAME")
  ENV["HNODECB_STAGE2_RESULT_BASENAME"] = "afm_param_stage2light_03.jld"
end
if !haskey(ENV, "HNODECB_STAGE2_X3R_WEIGHT")
  ENV["HNODECB_STAGE2_X3R_WEIGHT"] = "0.0"
end
if !haskey(ENV, "HNODECB_STAGE2_USE_GNN")
  ENV["HNODECB_STAGE2_USE_GNN"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_GNN_DEFAULT")
  ENV["HNODECB_STAGE2_GNN_DEFAULT"] = "1.0"
end

include("afm_param_stage2_03.jl")
