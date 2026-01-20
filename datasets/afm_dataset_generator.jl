#= Script to generate the AFM DMT-KV dataset with high-frequency sampling and no noise. =#

cd(@__DIR__)

using Serialization, DifferentialEquations, LinearAlgebra, DataFrames, Statistics

include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_functions.jl")
include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")
include("non_perturbed_dataset_generator.jl")

error_level = "e0.0"
column_names = ["t", "x1", "x2", "x3"]

# High-frequency sampling (matches Python generator length)
tspan = (initial_time_training, end_time_training)
tsteps = range(tspan[1], tspan[2], length=125000 + 1)

# Generate noise-free trajectories
solution_dataframe = generate_non_perturbed_training_set(
  ground_truth_function,
  original_u0,
  original_parameters,
  tspan,
  tsteps;
  column_names=column_names,
  integrator=Rosenbrock23(autodiff=false)
)

solution_matrix = Array(solution_dataframe[:, :])
ode_data = transpose(solution_matrix[:, 2:end])
ode_data_std = zeros(size(ode_data))

# Derived quantities for AFM analysis
tvals = solution_dataframe.t
x1 = ode_data[1, :]
x2 = ode_data[2, :]
x3 = ode_data[3, :]
s = dist .+ x1 .- x3
contact = Int.(s .<= 0.0)

x2dot = similar(x1)
du = zeros(3)
for i in eachindex(tvals)
  u = [x1[i], x2[i], x3[i]]
  ground_truth_function(du, u, original_parameters, tvals[i])
  x2dot[i] = du[2]
end

afm_dataframe = DataFrame(
  t=tvals,
  x1=x1,
  x2=x2,
  x3=x3,
  x2dot=x2dot,
  contact=contact,
  s=s
)

afm_dataframe_sd = DataFrame(
  t=tvals,
  x1=zeros(length(tvals)),
  x2=zeros(length(tvals)),
  x3=zeros(length(tvals)),
  x2dot=zeros(length(tvals)),
  contact=zeros(Int, length(tvals)),
  s=zeros(length(tvals))
)

folder_name = error_level
data_folder_name = joinpath(folder_name, "data")
if !isdir(folder_name)
  mkdir(folder_name)
end
if !isdir(data_folder_name)
  mkdir(data_folder_name)
end

serialize(joinpath(data_folder_name, "ode_data_afm_dmt_kv.jld"), ode_data)
serialize(joinpath(data_folder_name, "ode_data_std_afm_dmt_kv.jld"), ode_data_std)
serialize(joinpath(data_folder_name, "pert_df_afm_dmt_kv.jld"), afm_dataframe)
serialize(joinpath(data_folder_name, "pert_df_sd_afm_dmt_kv.jld"), afm_dataframe_sd)
