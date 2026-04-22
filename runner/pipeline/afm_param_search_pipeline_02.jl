#=
Pipeline runner for AFM parameter search (case 02: x3 unobserved).
Runs Stage1, then Stage2 (which loads Stage1 results).
=#

cd(@__DIR__)

println("=== AFM Stage1 (02): global coarse search ===")
include("../step2a_hyperparameter_tuning/hyperparameter_tuning_first_stage/afm_param_stage1_02.jl")

println("=== AFM Stage2 (02): local refinement ===")
include("../step2a_hyperparameter_tuning/hyperparameter_tuning_second_stage/afm_param_stage2_02.jl")

println("=== AFM Stage1+Stage2 (02) complete ===")
