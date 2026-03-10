#=
Standalone Stage1 runner for AFM parameter search (case 03).
Runs Stage1 only. If sharding is enabled, it spawns shards, merges results,
prints the global top-10, and writes the merged Stage1 `.jld` for Stage2.
=#

cd(@__DIR__)
using Serialization
using Printf

fmt_e(x; sigdigits=4) = (isfinite(x) ? @sprintf("%.*e", max(sigdigits - 1, 0), x) : "None")
fmt_f(x; digits=2) = (isfinite(x) ? @sprintf("%.*f", digits, x) : "None")
fmt_pct(x; digits=2) = (isfinite(x) ? @sprintf("%.*f%%", digits, x) : "None")
fmt_part(rec, key) = (
  hasproperty(rec, :val_parts) && rec.val_parts !== nothing && hasproperty(rec.val_parts, key)
) ? fmt_e(getproperty(rec.val_parts, key), sigdigits=3) : "None"

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
  use_adam = get(ENV, "HNODECB_STAGE1_USE_ADAM", "0") == "1"
  nn_err = use_adam ? get(ENV, "HNODECB_LOG_NN_ERR", "1") : "0 (forced off in no-ADAM)"
  println("=== Stage1 (03) runtime config ===")
  println("Threads: ", threads, " | Shard: ", shard_idx, "/", shard_cnt)
  println("Preflight: ", preflight, " | LogEvery: ", log_every,
          " | InfLog: ", inf_log, " | NNErr: ", nn_err)
  flush(stdout)
end

did_autospawn = false

if get(ENV, "HNODECB_STAGE1_AUTOSPAWN", get(ENV, "HNODECB_PIPELINE03_AUTOSPAWN", "1")) == "1" &&
   get(ENV, "HNODECB_STAGE1_SHARD_COUNT", "1") != "1" &&
   !haskey(ENV, "HNODECB_STAGE1_SHARD_INDEX")

  shard_cnt = parse(Int, ENV["HNODECB_STAGE1_SHARD_COUNT"])
  repo_root = normpath(joinpath(@__DIR__, "..", ".."))
  project = get(ENV, "HNODECB_JULIA_PROJECT", repo_root)
  stage1_path = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_first_stage", "afm_param_stage1_03.jl"))
  log_dir = normpath(joinpath(@__DIR__, "..", "..", "logs"))
  isdir(log_dir) || mkpath(log_dir)

  println("=== Stage1 (03): spawning ", shard_cnt, " shards ===")
  flush(stdout)
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
    flush(stdout)
  end

  for p in procs
    wait(p)
  end
  for io in ios
    close(io)
  end

  result_dir = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_first_stage", "results_afm"))
  merged_trials = []
  merged_warm_candidates = []
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
    if haskey(data, :warm_start_top)
      append!(merged_warm_candidates, data.warm_start_top)
    end
  end

  sorted = sort(merged_trials, by = r -> r.loss)
  best = sorted[1]
  ks_true = (hasproperty(first_meta, :true_values) && hasproperty(first_meta.true_values, :ks)) ? first_meta.true_values.ks : NaN
  cs_true = (hasproperty(first_meta, :true_values) && hasproperty(first_meta.true_values, :cs)) ? first_meta.true_values.cs : NaN
  warm_by_trial = Dict{Any, Any}()
  for rec in merged_warm_candidates
    if hasproperty(rec, :trial_id)
      warm_by_trial[rec.trial_id] = rec
    end
  end
  merged_warm_top = Any[]
  println("=== Stage1 (03): global top-10 (merged) ===")
  for (rank, rec) in enumerate(sorted[1:min(10, length(sorted))])
    train_loss = hasproperty(rec, :train_loss) ? rec.train_loss : rec.loss
    val_loss = hasproperty(rec, :val_loss) ? rec.val_loss : rec.loss
    ks0 = hasproperty(rec, :params) && haskey(rec.params, "ks0") ? rec.params["ks0"] : NaN
    cs0 = hasproperty(rec, :params) && haskey(rec.params, "cs0") ? rec.params["cs0"] : NaN
    if hasproperty(rec, :params) && haskey(rec.params, "trial_id")
      trial_id = rec.params["trial_id"]
      if haskey(warm_by_trial, trial_id)
        push!(merged_warm_top, warm_by_trial[trial_id])
      end
    end
    ks_hat = hasproperty(rec, :ks_hat) ? rec.ks_hat : ks0
    cs_hat = hasproperty(rec, :cs_hat) ? rec.cs_hat : cs0
    ks_err_pct = hasproperty(rec, :ks_err_pct) ? rec.ks_err_pct : NaN
    cs_err_pct = hasproperty(rec, :cs_err_pct) ? rec.cs_err_pct : NaN
    nn_err = hasproperty(rec, :val_nn_err) ? rec.val_nn_err : NaN
    raw_min = hasproperty(rec, :val_raw_min) ? rec.val_raw_min : NaN
    raw_max = hasproperty(rec, :val_raw_max) ? rec.val_raw_max : NaN
    raw_mean = hasproperty(rec, :val_raw_mean) ? rec.val_raw_mean : NaN
    raw_neg_frac = hasproperty(rec, :val_raw_neg_frac) ? rec.val_raw_neg_frac : NaN
    println("  Rank ", rank,
      " -- train=", fmt_e(train_loss, sigdigits=4),
      " val=", fmt_e(val_loss, sigdigits=4),
      " | ks0=", fmt_e(ks0, sigdigits=3),
      " cs0=", fmt_e(cs0, sigdigits=3))
    println("     val parts: state=", fmt_part(rec, :state),
      " x2dot=", fmt_part(rec, :x2dot),
      " x3r=", fmt_part(rec, :x3_range),
      " cont=", fmt_part(rec, :cont))
    println("     rec: x1=", fmt_part(rec, :x1_rec),
      " x3=", fmt_part(rec, :x3_rec))
    println("     mech: ks=", fmt_e(ks_hat, sigdigits=3),
      " (true=", fmt_e(ks_true, sigdigits=3), " err=", fmt_pct(ks_err_pct, digits=2), ")",
      " cs=", fmt_e(cs_hat, sigdigits=3),
      " (true=", fmt_e(cs_true, sigdigits=3), " err=", fmt_pct(cs_err_pct, digits=2), ")")
    if hasproperty(rec, :val_nn_err)
      println("     nn: F_hertz err=", fmt_pct(nn_err, digits=2))
      println("     nn raw: min=", fmt_e(raw_min, sigdigits=3),
        " max=", fmt_e(raw_max, sigdigits=3),
        " mean=", fmt_e(raw_mean, sigdigits=3),
        " neg=", fmt_pct(100 * raw_neg_frac, digits=2))
    end
  end
  merged_warm_has_pnet = !isempty(merged_warm_top) && hasproperty(merged_warm_top[1], :p_net)
  println("Stage1 warm-start export (merged): ",
    isempty(merged_warm_top) ? "OFF" : "ON",
    " | saved=", length(merged_warm_top),
    " | first_has_p_net=", merged_warm_has_pnet)
  println("Warm-start NN states saved for ", length(merged_warm_top), " merged top candidates")
  flush(stdout)

  merged_file = joinpath(result_dir, "afm_param_stage1_03.jld")
  serialize(merged_file, (
    study=nothing,
    trial_parameters=merged_trials,
    warm_start_top=merged_warm_top,
    best=best,
    bounds=first_meta.bounds,
    true_values=hasproperty(first_meta, :true_values) ? first_meta.true_values : (ks=NaN, cs=NaN),
    use_multiple_shooting=first_meta.use_multiple_shooting,
    use_l2_regularization=first_meta.use_l2_regularization,
    val_stride=first_meta.val_stride,
    val_offset=first_meta.val_offset,
    x3_obs_fraction=first_meta.x3_obs_fraction,
    error_level=first_meta.error_level
  ))
  println("=== Stage1 (03): merged results -> ", merged_file, " ===")
  println("=== AFM Stage1 (03) complete ===")
  flush(stdout)
  did_autospawn = true
end

if !did_autospawn
  println("=== AFM Stage1 (03): global coarse search ===")
  flush(stdout)
  include("../../step2a_hyperparameter_tuning/hyperparameter_tuning_first_stage/afm_param_stage1_03.jl")
  println("=== AFM Stage1 (03) complete ===")
  flush(stdout)
end
