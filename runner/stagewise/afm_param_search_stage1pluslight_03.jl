#=
Standalone Stage1pluslight runner for AFM parameter search (case 03).
Supports shard autospawn + merge, then emits a merged hall-of-fame.
=#

cd(@__DIR__)
using Serialization
using Printf

function stage1pluslight_candidate_index()
  raw = strip(get(ENV, "HNODECB_STAGE1PLUSLIGHT_CANDIDATE", "6"))
  idx = tryparse(Int, raw)
  if idx === nothing || idx < 1
    error("HNODECB_STAGE1PLUSLIGHT_CANDIDATE must be a positive integer, got: " * raw)
  end
  return idx
end

if !haskey(ENV, "HNODECB_STAGE1_VARIANT")
  ENV["HNODECB_STAGE1_VARIANT"] = "stage1pluslight"
end
if !haskey(ENV, "HNODECB_STAGE1_RESULT_STEM")
  ENV["HNODECB_STAGE1_RESULT_STEM"] = "afm_param_stage1pluslight_03"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_INPUT_BASENAME")
  ENV["HNODECB_STAGE1PLUS_INPUT_BASENAME"] = "afm_param_stage1_03.jld"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_INPUT_TOPK")
  ENV["HNODECB_STAGE1PLUS_INPUT_TOPK"] = "9"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_BASE_INDICES")
  ENV["HNODECB_STAGE1PLUS_BASE_INDICES"] = string(stage1pluslight_candidate_index())
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE")
  ENV["HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE"] = "1000"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_FINAL_TOPK")
  ENV["HNODECB_STAGE1PLUS_FINAL_TOPK"] = "10"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ZERO_NN")
  ENV["HNODECB_STAGE1PLUS_ZERO_NN"] = "0"
end
if !haskey(ENV, "HNODECB_STAGE1_SHARD_COUNT")
  ENV["HNODECB_STAGE1_SHARD_COUNT"] = "3"
end
if !haskey(ENV, "HNODECB_STAGE1_AUTOSPAWN")
  ENV["HNODECB_STAGE1_AUTOSPAWN"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_ENABLE")
  ENV["HNODECB_STAGE1PLUS_ARCH_SCREEN_ENABLE"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH")
  ENV["HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH"] = "10"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_EPOCHS")
  ENV["HNODECB_STAGE1PLUS_ARCH_EPOCHS"] = "20"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_WINDOW_US")
  ENV["HNODECB_STAGE1PLUS_ARCH_WINDOW_US"] = "5e-6"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_OBJ_A")
  ENV["HNODECB_STAGE1PLUS_ARCH_OBJ_A"] = "0.35"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_OBJ_B")
  ENV["HNODECB_STAGE1PLUS_ARCH_OBJ_B"] = "0.45"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_OBJ_C")
  ENV["HNODECB_STAGE1PLUS_ARCH_OBJ_C"] = "0.20"
end

fmt_e(x; sigdigits=4) = (x isa Number && isfinite(x)) ? @sprintf("%.*e", max(sigdigits - 1, 0), x) : "None"
fmt_f(x; digits=2) = (x isa Number && isfinite(x)) ? @sprintf("%.*f", digits, x) : "None"

did_autospawn = false

if get(ENV, "HNODECB_STAGE1_AUTOSPAWN", "1") == "1" &&
   get(ENV, "HNODECB_STAGE1_SHARD_COUNT", "1") != "1" &&
   !haskey(ENV, "HNODECB_STAGE1_SHARD_INDEX")

  shard_cnt = parse(Int, ENV["HNODECB_STAGE1_SHARD_COUNT"])
  repo_root = normpath(joinpath(@__DIR__, "..", ".."))
  project = get(ENV, "HNODECB_JULIA_PROJECT", repo_root)
  stage1plus_path = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_first_stage", "afm_param_stage1pluslight_03.jl"))
  log_subdir = get(ENV, "HNODECB_STAGE1PLUSLIGHT_LOG_SUBDIR", "stage1_step2a")
  log_prefix = get(ENV, "HNODECB_STAGE1PLUSLIGHT_LOG_PREFIX", "log2_03_step2a_stage1pluslight")
  log_dir = normpath(joinpath(@__DIR__, "..", "..", "logs", log_subdir))
  isdir(log_dir) || mkpath(log_dir)
  result_dir = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_first_stage", "results_afm"))
  isdir(result_dir) || mkpath(result_dir)

  result_stem = get(ENV, "HNODECB_STAGE1_RESULT_STEM", "afm_param_stage1pluslight_03")
  merged_file = joinpath(result_dir, result_stem * ".jld")
  archscreen_file = joinpath(result_dir, result_stem * "_archscreen.jld")
  if isfile(merged_file)
    rm(merged_file; force=true)
  end
  if isfile(archscreen_file)
    rm(archscreen_file; force=true)
  end
  for i in 1:shard_cnt
    shard_file = joinpath(result_dir, result_stem * "_p" * string(i) * ".jld")
    if isfile(shard_file)
      rm(shard_file; force=true)
    end
  end

  arch_screen_enable = get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_ENABLE", "1") == "1"
  arch_screen_only = get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_ONLY", "0") == "1"
  have_selected_arch = haskey(ENV, "HNODECB_STAGE1PLUS_SELECTED_LAYERS") && haskey(ENV, "HNODECB_STAGE1PLUS_SELECTED_NODES")
  archscreen_data = nothing
  if arch_screen_enable && !have_selected_arch
    println("=== Stage1pluslight (03): architecture screening ===")
    flush(stdout)
    arch_env = copy(ENV)
    arch_env["HNODECB_STAGE1_AUTOSPAWN"] = "0"
    arch_env["HNODECB_STAGE1PLUS_ARCH_SCREEN_ONLY"] = "1"
    arch_env["HNODECB_STAGE1_RESULT_STEM"] = result_stem * "_archscreen"
    arch_cmd = `$(Base.julia_cmd()) --project=$project $stage1plus_path`
    run(setenv(arch_cmd, arch_env))
    if !isfile(archscreen_file)
      error("Missing architecture screening result file: " * archscreen_file)
    end
    archscreen_data = deserialize(archscreen_file)
    if !haskey(archscreen_data, :selected_architecture)
      error("Architecture screening result file does not contain selected_architecture: " * archscreen_file)
    end
    selected_arch = archscreen_data.selected_architecture
    ENV["HNODECB_STAGE1PLUS_SELECTED_LAYERS"] = string(selected_arch.num_hidden_layers)
    ENV["HNODECB_STAGE1PLUS_SELECTED_NODES"] = string(selected_arch.num_hidden_nodes)
    println("=== Stage1pluslight (03): architecture screen complete ===")
    println("Selected architecture -> layers=", selected_arch.num_hidden_layers,
      " nodes=", selected_arch.num_hidden_nodes)
    flush(stdout)
    if arch_screen_only
      did_autospawn = true
    end
  end

  if !did_autospawn
  println("=== Stage1pluslight (03): spawning ", shard_cnt, " shards ===")
  println("Candidate=", get(ENV, "HNODECB_STAGE1PLUS_BASE_INDICES", "6"),
    " | TrialsPerCandidate=", get(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE", "1000"))
  if haskey(ENV, "HNODECB_STAGE1PLUS_SELECTED_LAYERS") && haskey(ENV, "HNODECB_STAGE1PLUS_SELECTED_NODES")
    println("Selected architecture: layers=", ENV["HNODECB_STAGE1PLUS_SELECTED_LAYERS"],
      " nodes=", ENV["HNODECB_STAGE1PLUS_SELECTED_NODES"])
  end
  flush(stdout)

  procs = []
  ios = IO[]
  for i in 1:shard_cnt
    env = copy(ENV)
    env["HNODECB_STAGE1_SHARD_INDEX"] = string(i)
    env["HNODECB_STAGE1_SHARD_COUNT"] = string(shard_cnt)
    env["HNODECB_STAGE1_RUN_TAG"] = "p" * string(i)
    env["HNODECB_STAGE1_AUTOSPAWN"] = "0"
    env["HNODECB_STAGE1PLUS_ARCH_SCREEN_ONLY"] = "0"
    logfile = joinpath(log_dir, log_prefix * "_p" * string(i) * ".txt")
    io = open(logfile, "w")
    cmd = `$(Base.julia_cmd()) --project=$project $stage1plus_path`
    p = run(pipeline(setenv(cmd, env), stdout=io, stderr=io); wait=false)
    push!(procs, p)
    push!(ios, io)
    println("  shard ", i, "/", shard_cnt, " -> ", logfile)
    flush(stdout)
  end

  for (i, p) in enumerate(procs)
    wait(p)
    if !success(p)
      error("Stage1pluslight shard " * string(i) * " exited with failure. Inspect shard logs before merge.")
    end
  end
  for io in ios
    close(io)
  end

  merged_trials = Any[]
  local first_meta = nothing
  for i in 1:shard_cnt
    shard_file = joinpath(result_dir, result_stem * "_p" * string(i) * ".jld")
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

  viable_trials = [rec for rec in merged_trials if hasproperty(rec, :is_viable) && rec.is_viable]
  selection_pool = isempty(viable_trials) ? merged_trials : viable_trials
  sorted = sort(selection_pool, by = r -> r.loss)
  final_topk = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_FINAL_TOPK", "10")))
  selected = sorted[1:min(final_topk, length(sorted))]
  warm_start_top = [(
      trial_id=rec.params["trial_id"],
      params=rec.params,
      p_net=copy(rec.p_net_vec),
      train_loss=rec.train_loss,
      val_loss=rec.val_loss,
      g_nn=(haskey(rec.params, "g_nn") ? rec.params["g_nn"] : 1.0)
    ) for rec in selected if hasproperty(rec, :p_net_vec)]

  println("=== Stage1pluslight (03): hall of fame (merged top-", length(selected), ") ===")
  for (rank, rec) in enumerate(selected)
    println("  Rank ", rank,
      " -- train=", fmt_e(rec.train_loss, sigdigits=4),
      " val=", fmt_e(rec.val_loss, sigdigits=4),
      " | base=", rec.params["base_rank"],
      " | ks0=", fmt_e(rec.params["ks0"], sigdigits=3),
      " cs0=", fmt_e(rec.params["cs0"], sigdigits=3))
    println("     mech: ks=", fmt_e(hasproperty(rec, :ks_hat) ? rec.ks_hat : NaN, sigdigits=3),
      " (err=", fmt_f(hasproperty(rec, :ks_err_pct) ? rec.ks_err_pct : NaN, digits=2), "%)",
      " cs=", fmt_e(hasproperty(rec, :cs_hat) ? rec.cs_hat : NaN, sigdigits=3),
      " (err=", fmt_f(hasproperty(rec, :cs_err_pct) ? rec.cs_err_pct : NaN, digits=2), "%)")
    println("     nn gain: g_nn=", fmt_e(haskey(rec.params, "g_nn") ? rec.params["g_nn"] : NaN, sigdigits=3),
      " amp_ref=", fmt_e(haskey(rec.params, "g_amp_ref") ? rec.params["g_amp_ref"] : NaN, sigdigits=3))
    println("     nn: F_hertz err=", fmt_f(hasproperty(rec, :val_nn_err) ? rec.val_nn_err : NaN, digits=2), "%")
  end
  flush(stdout)

  serialize(merged_file, (
    study=nothing,
    trial_parameters=merged_trials,
    warm_start_top=warm_start_top,
    selected=selected,
    best=isempty(selected) ? nothing : selected[1],
    bounds=first_meta.bounds,
    true_values=first_meta.true_values,
    use_multiple_shooting=first_meta.use_multiple_shooting,
    use_l2_regularization=first_meta.use_l2_regularization,
    val_stride=first_meta.val_stride,
    val_offset=first_meta.val_offset,
    x3_obs_fraction=first_meta.x3_obs_fraction,
    error_level=first_meta.error_level,
    stage1_input_file=(haskey(first_meta, :stage1_input_file) ? first_meta.stage1_input_file : ""),
    stage1_input_topk=(haskey(first_meta, :stage1_input_topk) ? first_meta.stage1_input_topk : 0),
    searches_per_candidate=parse(Int, get(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE", "1000")),
    final_topk=final_topk,
    shard_count=shard_cnt,
    architecture_screen=archscreen_data
  ))
  println("=== Stage1pluslight (03): merged results -> ", merged_file, " ===")
  flush(stdout)
  did_autospawn = true
  end
end

if !did_autospawn
  let
    threads = get(ENV, "JULIA_NUM_THREADS", "1")
    candidate = get(ENV, "HNODECB_STAGE1PLUS_BASE_INDICES", string(stage1pluslight_candidate_index()))
    trials = get(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE", "1000")
    input_file = get(ENV, "HNODECB_STAGE1PLUS_INPUT_BASENAME", "afm_param_stage1_03.jld")
    result_stem = get(ENV, "HNODECB_STAGE1_RESULT_STEM", "afm_param_stage1pluslight_03")
    println("=== Stage1pluslight (03) runtime config ===")
    println("Threads: ", threads, " | Candidate: ", candidate, " | TrialsPerCandidate: ", trials)
    println("InputFile: ", input_file, " | ResultStem: ", result_stem)
    flush(stdout)
  end

  println("=== AFM Stage1pluslight (03) ===")
  flush(stdout)
  include("../../step2a_hyperparameter_tuning/hyperparameter_tuning_first_stage/afm_param_stage1pluslight_03.jl")
  println("=== AFM Stage1pluslight (03) complete ===")
  flush(stdout)
end
