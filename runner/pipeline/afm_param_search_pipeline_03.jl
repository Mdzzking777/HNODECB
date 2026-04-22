#=
Pipeline runner for AFM parameter search (case 03: x3 unobserved, Hertz NN).
Runs Stage1, then Stage2 (which loads Stage1 results).
=#

cd(@__DIR__)
using Serialization
using Printf

if get(ENV, "HNODECB_PIPELINE03_AUTOTUNE", "1") == "1"
  if !haskey(ENV, "JULIA_NUM_THREADS")
    ENV["JULIA_NUM_THREADS"] = "8"
  end
  if !haskey(ENV, "HNODECB_STAGE1_PREFLIGHT")
    ENV["HNODECB_STAGE1_PREFLIGHT"] = "0"
  end
  if !haskey(ENV, "HNODECB_STAGE1_LOG_EVERY")
    ENV["HNODECB_STAGE1_LOG_EVERY"] = "1"
  end
  if !haskey(ENV, "HNODECB_INF_LOG")
    ENV["HNODECB_INF_LOG"] = "1"
  end
  if !haskey(ENV, "HNODECB_LOG_NN_ERR")
    ENV["HNODECB_LOG_NN_ERR"] = "1"
  end
  if !haskey(ENV, "HNODECB_STAGE1_SHARD_COUNT")
    ENV["HNODECB_STAGE1_SHARD_COUNT"] = "1"
  end
end

let
  threads = get(ENV, "JULIA_NUM_THREADS", "1")
  shard_idx = get(ENV, "HNODECB_STAGE1_SHARD_INDEX", "1")
  shard_cnt = get(ENV, "HNODECB_STAGE1_SHARD_COUNT", "1")
  preflight = get(ENV, "HNODECB_STAGE1_PREFLIGHT", "0")
  log_every = get(ENV, "HNODECB_STAGE1_LOG_EVERY", "1")
  inf_log = get(ENV, "HNODECB_INF_LOG", "1")
  nn_err = get(ENV, "HNODECB_LOG_NN_ERR", "1")
  println("=== Pipeline03 runtime config ===")
  println("Threads: ", threads, " | Shard: ", shard_idx, "/", shard_cnt)
  println("Preflight: ", preflight, " | LogEvery: ", log_every,
          " | InfLog: ", inf_log, " | NNErr: ", nn_err)
end

# flag to skip Stage1 include if we already spawned shards
did_autospawn = false

# Optional: auto-spawn multiple Stage1 shards in parallel and merge results
if get(ENV, "HNODECB_PIPELINE03_AUTOSPAWN", "1") == "1" &&
   get(ENV, "HNODECB_STAGE1_SHARD_COUNT", "1") != "1" &&
   !haskey(ENV, "HNODECB_STAGE1_SHARD_INDEX")

  shard_cnt = parse(Int, ENV["HNODECB_STAGE1_SHARD_COUNT"])
  repo_root = normpath(joinpath(@__DIR__, ".."))
  project = get(ENV, "HNODECB_JULIA_PROJECT", repo_root)
  stage1_path = normpath(joinpath(@__DIR__, "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_first_stage", "afm_param_stage1_03.jl"))
  log_dir = normpath(joinpath(@__DIR__, "..", "logs"))
  isdir(log_dir) || mkpath(log_dir)

  println("=== Pipeline03: spawning ", shard_cnt, " Stage1 shards ===")
  procs = []
  ios = IO[]
  for i in 1:shard_cnt
    env = copy(ENV)
    env["HNODECB_STAGE1_SHARD_INDEX"] = string(i)
    env["HNODECB_STAGE1_SHARD_COUNT"] = string(shard_cnt)
    env["HNODECB_STAGE1_RUN_TAG"] = "p" * string(i)
    logfile = joinpath(log_dir, "log2_03_step2a_stage1_p" * string(i) * ".txt")
    io = open(logfile, "w")
    cmd = `$(Base.julia_cmd()) --project=$project $stage1_path`
    p = run(pipeline(setenv(cmd, env), stdout=io, stderr=io); wait=false)
    push!(procs, p)
    push!(ios, io)
    println("  shard ", i, "/", shard_cnt, " -> ", logfile)
  end
  for p in procs
    wait(p)
  end
  for io in ios
    close(io)
  end

  # Merge shard results into the default Stage1 result file for Stage2
  result_dir = normpath(joinpath(@__DIR__, "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_first_stage", "results_afm"))
  merged_trials = []
  local first_meta = nothing
  for i in 1:shard_cnt
    shard_file = joinpath(result_dir, "afm_param_stage1_03_p" * string(i) * ".jld")
    if !isfile(shard_file)
      error("Missing shard result file: " * shard_file)
    end
    data = deserialize(shard_file)
    if first_meta === nothing
      first_meta = data
    end
    append!(merged_trials, data.trial_parameters)
  end
  sorted = sort(merged_trials, by = r -> r.loss)
  best = sorted[1]
  println("=== Pipeline03: Stage1 global top-10 (merged) ===")
  for (rank, rec) in enumerate(sorted[1:min(10, length(sorted))])
    train_loss = hasproperty(rec, :train_loss) ? rec.train_loss : rec.loss
    val_loss = hasproperty(rec, :val_loss) ? rec.val_loss : rec.loss
    ks0 = hasproperty(rec, :params) && haskey(rec.params, "ks0") ? rec.params["ks0"] : NaN
    cs0 = hasproperty(rec, :params) && haskey(rec.params, "cs0") ? rec.params["cs0"] : NaN
    println("  Rank ", rank,
      " -- train=", @sprintf("%.4e", train_loss),
      " val=", @sprintf("%.4e", val_loss),
      " | ks0=", @sprintf("%.3e", ks0),
      " cs0=", @sprintf("%.3e", cs0))
  end
  merged_file = joinpath(result_dir, "afm_param_stage1_03.jld")
  serialize(merged_file, (
    study=nothing,
    trial_parameters=merged_trials,
    best=best,
    bounds=first_meta.bounds,
    use_multiple_shooting=first_meta.use_multiple_shooting,
    use_l2_regularization=first_meta.use_l2_regularization,
    val_stride=first_meta.val_stride,
    val_offset=first_meta.val_offset,
    error_level=first_meta.error_level
  ))
  println("=== Pipeline03: merged Stage1 results -> ", merged_file, " ===")
  did_autospawn = true
end

if !did_autospawn
  println("=== AFM Stage1 (03): global coarse search ===")
  include("../step2a_hyperparameter_tuning/hyperparameter_tuning_first_stage/afm_param_stage1_03.jl")
end

println("=== AFM Stage2 (03): local refinement ===")
include("../step2a_hyperparameter_tuning/hyperparameter_tuning_second_stage/afm_param_stage2_03.jl")

println("=== AFM Stage1+Stage2 (03) complete ===")
