#=
Pipeline runner for AFM parameter search.
Runs Stage1, then Stage2 (which loads Stage1 results).
=#

cd(@__DIR__)

println("=== AFM Stage1: global coarse search ===")
include("../step2a_hyperparameter_tuning/hyperparameter_tuning_first_stage/afm_param_stage1_01.jl")

println("=== AFM Stage2: local refinement ===")
include("../step2a_hyperparameter_tuning/hyperparameter_tuning_second_stage/afm_param_stage2_01.jl")

println("=== AFM Stage1+Stage2 complete ===")
