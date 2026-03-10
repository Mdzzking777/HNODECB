#=
Offline oracle diagnostics for AFM 03 (NN Hertz replacement).
This does NOT affect training. It only compares NN Hertz outputs to the
ground-truth Hertz force using full (oracle) state trajectories.
=#

cd(@__DIR__)

using Serialization, Printf, Statistics, Random
using ComponentArrays, Lux

include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_functions.jl")
include("../test_case_settings/afm_dmt_kv_settings/afm_dmt_kv_model_settings.jl")

path = "../step2b_model_trainer/res_afm_03/afm_03.jld"
if !isfile(path)
  error("Result file not found: " * path)
end

payload = deserialize(path)
results = haskey(payload, :results) ? payload.results : payload

# Rebuild the Hertz-NN architecture (same as train_afm_03_SS.jl)
approximating_neural_network = Lux.Chain(
  Lux.Dense(2, 16, tanh),
  Lux.Dense(16, 16, tanh),
  Lux.Dense(16, 1)
)
_, st = Lux.setup(Random.default_rng(), approximating_neural_network)

# Load data (oracle trajectory)
ode_data = deserialize("../datasets/e0.0/data/ode_data_afm_dmt_kv.jld")
solution_dataframe = deserialize("../datasets/e0.0/data/pert_df_afm_dmt_kv.jld")
contact_idx = findfirst(solution_dataframe.contact .== 1)
if contact_idx === nothing
  error("No contact point found in the AFM dataset.")
end
solution_dataframe = solution_dataframe[contact_idx:end, :]
ode_data = ode_data[:, contact_idx:end]
contact_mask = solution_dataframe.contact .== 1

function hertz_true(u1, u3)
  s = dist + u1 - u3
  delta = softplus(-s, adhesion_transition)
  delta = ifelse(delta > 0.0, delta, 0.0)
  return (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
end

function hertz_pred(u1, u2, u3, p_net)
  s = dist + u1 - u3
  w = contact_weight(s, adhesion_transition)
  nn_in = [u1, u2]
  û = approximating_neural_network(nn_in, p_net, st)[1]
  return û[1] * w
end

function mape_pct(pred, truth; eps=1e-12)
  return 100.0 * mean(abs.(pred .- truth) ./ (abs.(truth) .+ eps))
end

function rel_rmse_pct(pred, truth; eps=1e-12)
  denom = sqrt(mean(abs2, truth)) + eps
  return 100.0 * sqrt(mean(abs2.(pred .- truth))) / denom
end

println("=== ORACLE NN Hertz diagnostics (offline only) ===")
println("NOTE: oracle uses true x1/x2/x3 and true parameters for Hertz.")

if results isa AbstractVector
  for (i, r) in enumerate(results)
    if !haskey(r, :p_net) || !haskey(r, :parameters_training)
      println("Run ", i, " [ORACLE] skipped (missing p_net or parameters_training).")
      continue
    end
    p_net = ComponentArray(r.p_net)

    n = size(ode_data, 2)
    F_true = Vector{Float64}(undef, n)
    F_pred = Vector{Float64}(undef, n)
    for j in 1:n
      u1 = ode_data[1, j]
      u2 = ode_data[2, j]
      u3 = ode_data[3, j]
      F_true[j] = hertz_true(u1, u3)
      F_pred[j] = hertz_pred(u1, u2, u3, p_net)
    end

    idx_contact = findall(contact_mask)
    F_true_c = F_true[idx_contact]
    F_pred_c = F_pred[idx_contact]

    mape_all = mape_pct(F_pred, F_true)
    rmse_all = rel_rmse_pct(F_pred, F_true)
    mape_c = isempty(idx_contact) ? NaN : mape_pct(F_pred_c, F_true_c)
    rmse_c = isempty(idx_contact) ? NaN : rel_rmse_pct(F_pred_c, F_true_c)

    println("Run ", i, " [ORACLE] Hertz MAPE(all)=", @sprintf("%.3f", mape_all),
      "% RMSE(all)=", @sprintf("%.3f", rmse_all), "%",
      " | MAPE(contact)=", @sprintf("%.3f", mape_c),
      "% RMSE(contact)=", @sprintf("%.3f", rmse_c), "%")
  end
else
  println("Results format not recognized; no oracle diagnostics computed.")
end
