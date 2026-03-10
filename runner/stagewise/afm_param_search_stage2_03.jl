#=
Standalone Stage2 runner for AFM parameter search (case 03).
Requires the merged Stage1 `.jld` result file to exist.
=#

cd(@__DIR__)
using Serialization
using Printf

fmt_part(rec, key; sigdigits=3) = begin
  if hasproperty(rec, :val_parts) && rec.val_parts !== nothing && hasproperty(rec.val_parts, key)
    v = getproperty(rec.val_parts, key)
    v isa Number && isfinite(v) ? @sprintf("%.*e", sigdigits, v) : string(v)
  else
    "None"
  end
end

fmt_pct(x; digits=2) = (x isa Number && isfinite(x)) ? @sprintf("%.*f", digits, x) : "None"

fmt_num(x; sigdigits=3) = (x isa Number && isfinite(x)) ? @sprintf("%.*e", sigdigits, x) : "None"

if get(ENV, "HNODECB_STAGE2_AUTOTUNE", "1") == "1"
  if !haskey(ENV, "HNODECB_LR_ADAPT")
    ENV["HNODECB_LR_ADAPT"] = "1"
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
  if !haskey(ENV, "HNODECB_STAGE2_LOG_EVERY")
    ENV["HNODECB_STAGE2_LOG_EVERY"] = "10"
  end
  if !haskey(ENV, "HNODECB_STAGE2_SHARD_COUNT")
    ENV["HNODECB_STAGE2_SHARD_COUNT"] = "1"
  end
  if !haskey(ENV, "HNODECB_STAGE2_INPUT_TOPK")
    ENV["HNODECB_STAGE2_INPUT_TOPK"] = "9"
  end
  if !haskey(ENV, "HNODECB_STAGE2_FINAL_TOPK")
    ENV["HNODECB_STAGE2_FINAL_TOPK"] = "3"
  end
end

let
  threads = get(ENV, "JULIA_NUM_THREADS", "1")
  shard_idx = get(ENV, "HNODECB_STAGE2_SHARD_INDEX", "1")
  shard_cnt = get(ENV, "HNODECB_STAGE2_SHARD_COUNT", "1")
  input_topk = get(ENV, "HNODECB_STAGE2_INPUT_TOPK", "9")
  log_every = get(ENV, "HNODECB_STAGE2_LOG_EVERY", "10")
  final_topk = get(ENV, "HNODECB_STAGE2_FINAL_TOPK", "3")
  input_file = get(ENV, "HNODECB_STAGE2_INPUT_BASENAME", "afm_param_stage1_03.jld")
  result_file = get(ENV, "HNODECB_STAGE2_RESULT_BASENAME", "afm_param_stage2_03.jld")
  optimizer = get(ENV, "HNODECB_STAGE2_OPTIMIZER", "amsgrad")
  grad_scale_p_net = get(ENV, "HNODECB_STAGE2_GRAD_SCALE_P_NET", "0.2")
  grad_scale_mech = get(ENV, "HNODECB_STAGE2_GRAD_SCALE_MECH", "1.0")
  group_adapt = get(ENV, "HNODECB_STAGE2_GROUP_ADAPT", "1")
  group_eta = get(ENV, "HNODECB_STAGE2_GROUP_ETA", "0.15")
  candidate_filter = get(ENV, "HNODECB_STAGE2_CANDIDATE_INDICES", "")
  stage2_script = get(ENV, "HNODECB_STAGE2_SCRIPT", "afm_param_stage2_03.jl")
  println("=== Stage2 (03) runtime config ===")
  println("Threads: ", threads, " | Shard: ", shard_idx, "/", shard_cnt)
  println("InputFile: ", input_file, " | InputTopK: ", input_topk,
    " | LogEvery: ", log_every, " | FinalTopK: ", final_topk,
    " | ResultFile: ", result_file)
  println("Optimizer: ", optimizer, " | GradScale(p_net/mech)=", grad_scale_p_net, "/", grad_scale_mech)
  println("GroupAdapt: ", group_adapt, " | GroupEta: ", group_eta)
  if strip(candidate_filter) != ""
    println("CandidateFilter: ", candidate_filter)
  end
  println("Stage2Script: ", stage2_script)
  flush(stdout)
end

stage2_input_basename = get(ENV, "HNODECB_STAGE2_INPUT_BASENAME", "afm_param_stage1_03.jld")
stage2_script_name = get(ENV, "HNODECB_STAGE2_SCRIPT", "afm_param_stage2_03.jl")
stage2_result_basename = get(ENV, "HNODECB_STAGE2_RESULT_BASENAME", "afm_param_stage2_03.jld")
stage2_result_root, stage2_result_ext = splitext(stage2_result_basename)
if stage2_result_ext == ""
  stage2_result_ext = ".jld"
  stage2_result_basename *= stage2_result_ext
  stage2_result_root = splitext(stage2_result_basename)[1]
end
stage2_shard_file(i) = stage2_result_root * "_p" * string(i) * stage2_result_ext
stage1_file = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
  "hyperparameter_tuning_first_stage", "results_afm", stage2_input_basename))

if !isfile(stage1_file)
  error("Missing Stage1 result file required by Stage2: " * stage1_file)
end

did_autospawn = false

if get(ENV, "HNODECB_STAGE2_AUTOSPAWN", "1") == "1" &&
   get(ENV, "HNODECB_STAGE2_SHARD_COUNT", "1") != "1" &&
   !haskey(ENV, "HNODECB_STAGE2_SHARD_INDEX")

  shard_cnt = parse(Int, ENV["HNODECB_STAGE2_SHARD_COUNT"])
  repo_root = normpath(joinpath(@__DIR__, "..", ".."))
  project = get(ENV, "HNODECB_JULIA_PROJECT", repo_root)
  stage2_path = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_second_stage", stage2_script_name))
  log_subdir = get(ENV, "HNODECB_STAGE2_LOG_SUBDIR", "")
  log_prefix = get(ENV, "HNODECB_STAGE2_LOG_PREFIX", "log2_03_step2a_stage2")
  log_dir = log_subdir == "" ?
    normpath(joinpath(@__DIR__, "..", "..", "logs")) :
    normpath(joinpath(@__DIR__, "..", "..", "logs", log_subdir))
  isdir(log_dir) || mkpath(log_dir)
  result_dir = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_second_stage", "results_afm"))
  isdir(result_dir) || mkpath(result_dir)
  merged_file = joinpath(result_dir, stage2_result_basename)
  if isfile(merged_file)
    rm(merged_file; force=true)
  end
  for i in 1:shard_cnt
    shard_file = joinpath(result_dir, stage2_shard_file(i))
    if isfile(shard_file)
      rm(shard_file; force=true)
    end
  end

  println("=== Stage2 (03): spawning ", shard_cnt, " shards ===")
  flush(stdout)
  procs = []
  ios = IO[]
  for i in 1:shard_cnt
    env = copy(ENV)
    env["HNODECB_STAGE2_SHARD_INDEX"] = string(i)
    env["HNODECB_STAGE2_SHARD_COUNT"] = string(shard_cnt)
    logfile = joinpath(log_dir, log_prefix * "_p" * string(i) * ".txt")
    io = open(logfile, "w")
    cmd = `$(Base.julia_cmd()) --project=$project $stage2_path`
    p = run(pipeline(setenv(cmd, env), stdout=io, stderr=io); wait=false)
    push!(procs, p)
    push!(ios, io)
    println("  shard ", i, "/", shard_cnt, " -> ", logfile)
    flush(stdout)
  end

  for (i, p) in enumerate(procs)
    wait(p)
    if !success(p)
      error("Stage2 shard " * string(i) * " exited with failure. Inspect shard logs before merge.")
    end
  end
  for io in ios
    close(io)
  end

  merged_trials = []
  local first_meta = nothing
  for i in 1:shard_cnt
    shard_file = joinpath(result_dir, stage2_shard_file(i))
    if !isfile(shard_file)
      error("Missing shard result file: " * shard_file)
    end
    if filesize(shard_file) == 0
      error("Empty shard result file: " * shard_file)
    end
    data = deserialize(shard_file)
    if first_meta === nothing
      first_meta = data
    end
    append!(merged_trials, data.trial_parameters)
  end

  sorted = sort(merged_trials, by = r -> r.loss)
  best = isempty(sorted) ? nothing : sorted[1]
  final_topk = max(1, parse(Int, get(ENV, "HNODECB_STAGE2_FINAL_TOPK", "3")))
  selected = sorted[1:min(final_topk, length(sorted))]
  println("=== Stage2 (03): global ranked results (merged) ===")
  for (rank, rec) in enumerate(sorted)
    println("  Rank ", rank,
      " -- train=", @sprintf("%.4e", rec.train_loss),
      " val=", @sprintf("%.4e", rec.val_loss),
      " | l2=", @sprintf("%.2e", rec.params["l2_regularization"]),
      " cand=", rec.params["cand_rank"])
    println("     val parts: state=", fmt_part(rec, :state),
      " x2dot=", fmt_part(rec, :x2dot),
      " x3r=", fmt_part(rec, :x3_range),
      " cont=", fmt_part(rec, :cont),
      " l2=", fmt_part(rec, :l2))
    println("     mech: ks=", fmt_num(hasproperty(rec, :ks_hat) ? rec.ks_hat : NaN),
      " (true=", fmt_num(first_meta.true_values.ks), " err=", fmt_pct(hasproperty(rec, :ks_err_pct) ? rec.ks_err_pct : NaN), "%)",
      " cs=", fmt_num(hasproperty(rec, :cs_hat) ? rec.cs_hat : NaN),
      " (true=", fmt_num(first_meta.true_values.cs), " err=", fmt_pct(hasproperty(rec, :cs_err_pct) ? rec.cs_err_pct : NaN), "%)")
    println("     nn: F_hertz err=", fmt_pct(hasproperty(rec, :val_nn_err) ? rec.val_nn_err : NaN), "%")
  end

  println("=== Stage2 (03): final selection top-", length(selected), " ===")
  for (rank, rec) in enumerate(selected)
    println("  Rank ", rank,
      " -- train=", @sprintf("%.4e", rec.train_loss),
      " val=", @sprintf("%.4e", rec.val_loss),
      " | l2=", @sprintf("%.2e", rec.params["l2_regularization"]),
      " cand=", rec.params["cand_rank"])
  end
  flush(stdout)

  serialize(merged_file, (
    study=nothing,
    trial_parameters=merged_trials,
    results=sorted,
    selected=selected,
    best=isempty(selected) ? nothing : selected[1],
    bounds=first_meta.bounds,
    true_values=first_meta.true_values,
    use_multiple_shooting=first_meta.use_multiple_shooting,
    l2_grid=first_meta.l2_grid,
    x3_obs_fraction=first_meta.x3_obs_fraction,
    error_level=first_meta.error_level
  ))
  println("=== Stage2 (03): merged results -> ", merged_file, " ===")
  println("=== AFM Stage2-only (03) complete ===")
  flush(stdout)
  did_autospawn = true
end

if !did_autospawn
  println("=== AFM Stage2-only (03) ===")
  println("Using Stage1 result: ", stage1_file)
  flush(stdout)
  include("../../step2a_hyperparameter_tuning/hyperparameter_tuning_second_stage/" * stage2_script_name)

  println("=== AFM Stage2-only (03) complete ===")
  flush(stdout)
end
