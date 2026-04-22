#=
Stage1 wrapper for AFM parameter search (case 03).
Stage1 is now the architecture screening stage only, implemented by reusing
the standalone stage1pluslight architecture-screening path with Stage1 naming.
=#

cd(@__DIR__)

if !haskey(ENV, "HNODECB_STAGE1_RUN_LABEL")
  ENV["HNODECB_STAGE1_RUN_LABEL"] = "Stage1"
end
if !haskey(ENV, "HNODECB_STAGE1_RESULT_STEM")
  ENV["HNODECB_STAGE1_RESULT_STEM"] = "afm_param_stage1_03"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_MODE")
  ENV["HNODECB_STAGE1PLUS_MODE"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUSLIGHT_MODE")
  ENV["HNODECB_STAGE1PLUSLIGHT_MODE"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_ONLY")
  ENV["HNODECB_STAGE1PLUS_ARCH_SCREEN_ONLY"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH")
  ENV["HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH"] = "10"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_EPOCHS")
  ENV["HNODECB_STAGE1PLUS_ARCH_EPOCHS"] = "10"
end
if !haskey(ENV, "HNODECB_STAGE1PLUSLIGHT_LOG_SUBDIR")
  ENV["HNODECB_STAGE1PLUSLIGHT_LOG_SUBDIR"] = "stage1_step2a/local"
end
if !haskey(ENV, "HNODECB_STAGE1PLUSLIGHT_LOG_PREFIX")
  ENV["HNODECB_STAGE1PLUSLIGHT_LOG_PREFIX"] = "log2_03_step2a_stage1_archscreen"
end
if !haskey(ENV, "HNODECB_STAGE1_ARCH_PARETO_SCRIPT")
  ENV["HNODECB_STAGE1_ARCH_PARETO_SCRIPT"] = normpath(joinpath(@__DIR__, "..", "visualization", "step2a",
    "stage1", "plot_afm_stage1_archscreen_pareto_03.jl"))
end

include("afm_param_search_stage1pluslight_03.jl")
