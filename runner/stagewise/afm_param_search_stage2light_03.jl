#=
Standalone Stage2light runner for AFM parameter search (case 03).
Same as Stage2 runner, but fixed to one candidate rank (default 6) and shard count 1.
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

if !haskey(ENV, "HNODECB_STAGE2_SCRIPT")
  ENV["HNODECB_STAGE2_SCRIPT"] = "afm_param_stage2light_03.jl"
end
if !haskey(ENV, "HNODECB_STAGE2_SHARD_COUNT")
  ENV["HNODECB_STAGE2_SHARD_COUNT"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_INPUT_TOPK")
  ENV["HNODECB_STAGE2_INPUT_TOPK"] = "9"
end
if !haskey(ENV, "HNODECB_STAGE2_LOG_EVERY")
  ENV["HNODECB_STAGE2_LOG_EVERY"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_OPTIMIZER")
  ENV["HNODECB_STAGE2_OPTIMIZER"] = "amsgrad"
end
if !haskey(ENV, "HNODECB_STAGE2_GRAD_SCALE_P_NET")
  ENV["HNODECB_STAGE2_GRAD_SCALE_P_NET"] = "0.2"
end
if !haskey(ENV, "HNODECB_STAGE2_GRAD_SCALE_MECH")
  ENV["HNODECB_STAGE2_GRAD_SCALE_MECH"] = "1.0"
end
if !haskey(ENV, "HNODECB_STAGE2_GROUP_ADAPT")
  ENV["HNODECB_STAGE2_GROUP_ADAPT"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_GROUP_ETA")
  ENV["HNODECB_STAGE2_GROUP_ETA"] = "0.15"
end
if !haskey(ENV, "HNODECB_STAGE2_VERIFY_ADAM_DIR")
  ENV["HNODECB_STAGE2_VERIFY_ADAM_DIR"] = "1"
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
if !haskey(ENV, "HNODECB_STAGE2_NN_WARM_ENABLED")
  ENV["HNODECB_STAGE2_NN_WARM_ENABLED"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_NN_WARM_INPUT_BASENAME")
  ENV["HNODECB_STAGE2_NN_WARM_INPUT_BASENAME"] = "afm_param_stage1pluslight_03.jld"
end
if !haskey(ENV, "HNODECB_STAGE2_NN_WARM_RANK")
  ENV["HNODECB_STAGE2_NN_WARM_RANK"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_LOG_PREFIX")
  ENV["HNODECB_STAGE2_LOG_PREFIX"] = "log2_03_step2a_stage2light"
end
if !haskey(ENV, "HNODECB_STAGE2_USE_GNN")
  ENV["HNODECB_STAGE2_USE_GNN"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_GNN_DEFAULT")
  ENV["HNODECB_STAGE2_GNN_DEFAULT"] = "1.0"
end

include("afm_param_search_stage2_03.jl")
