#=
Windowed Step2b runner for AFM 03.
Reuses the exact W1/W2/W3 window manifest produced by stage2light_windowed,
spawns 3 step2b shards, then merges their outputs.
=#

cd(@__DIR__)

using Serialization
using Printf
using ComponentArrays

if !haskey(ENV, "HNODECB_STEP2B_RESUME")
  ENV["HNODECB_STEP2B_RESUME"] = "1"
end
if !haskey(ENV, "HNODECB_STEP2B_CHECKPOINT_EVERY")
  ENV["HNODECB_STEP2B_CHECKPOINT_EVERY"] = "5"
end

fmt_num(x; sigdigits=3) = (x isa Number && isfinite(x)) ? @sprintf("%.*e", sigdigits, x) : "None"
fmt_pct(x; digits=2) = (x isa Number && isfinite(x)) ? @sprintf("%.*f", digits, x) : "None"
fmt_window_us(x; digits=3) = (x isa Number && isfinite(x)) ? @sprintf("%.*f", digits, 1e6 * x) : "None"

function shard_file_name(root::AbstractString, ext::AbstractString, i::Integer)
  return root * "_p" * string(i) * ext
end

env_truthy(name; default="0") = lowercase(strip(get(ENV, name, default))) in ("1", "true", "yes", "on")

function regex_escape(raw::AbstractString)
  return replace(String(raw), r"([\\.^$|?*+()\[\]{}])" => s"\\\1")
end

function has_existing_step2b_shard_results(result_dir::String, root::String, ext::String, count::Int)
  for i in 1:count
    shard_file = joinpath(result_dir, shard_file_name(root, ext, i))
    if isfile(shard_file)
      try
        filesize(shard_file) > 0 && return true
      catch
      end
    end
  end
  return false
end

function next_resume_log_suffix(log_dir::String, log_prefix::String)
  max_idx = 0
  pattern = Regex("^" * regex_escape(log_prefix) * "_resume(\\d+)_p\\d+\\.txt\$")
  if isdir(log_dir)
    for path in readdir(log_dir; join=true)
      m = match(pattern, basename(path))
      m === nothing && continue
      idx = tryparse(Int, m.captures[1])
      idx === nothing && continue
      max_idx = max(max_idx, idx)
    end
  end
  return @sprintf("_resume%02d", max_idx + 1)
end

shard_logfile(log_dir::String, log_prefix::String, shard_idx::Int, resume_suffix::AbstractString="") =
  joinpath(log_dir, log_prefix * String(resume_suffix) * "_p" * string(shard_idx) * ".txt")

function best_from_payload(payload)
  if !haskey(payload, :results) || isempty(payload.results)
    return nothing
  end
  vals = [r.validation_resulting_cost for r in payload.results]
  idx = argmin(vals)
  return payload.results[idx]
end

let
  repo_root = normpath(joinpath(@__DIR__, "..", ".."))
  project = get(ENV, "HNODECB_JULIA_PROJECT", repo_root)
  train_script = normpath(joinpath(repo_root, "step2b_model_trainer", "train_afm_03_SS.jl"))
  stage2_input_basename = get(ENV, "HNODECB_STEP2B_STAGE2_INPUT_BASENAME", "afm_param_stage2light_windowed_03.jld")
  stage2_input_file = normpath(joinpath(repo_root, "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_second_stage", "results_afm", stage2_input_basename))
  result_dir_name = get(ENV, "HNODECB_STEP2B_RESULT_DIR", "res_afm_03_windowed")
  result_basename = get(ENV, "HNODECB_STEP2B_RESULT_NAME", "afm_03_windowed.jld")
  result_root, result_ext = splitext(result_basename)
  if result_ext == ""
    result_ext = ".jld"
    result_basename *= result_ext
    result_root = splitext(result_basename)[1]
  end
  result_dir = isabspath(result_dir_name) ? result_dir_name : normpath(joinpath(repo_root, "step2b_model_trainer", result_dir_name))
  log_subdir = get(ENV, "HNODECB_STEP2B_LOG_SUBDIR", "step2b_windowed/print")
  shard_log_subdir = get(ENV, "HNODECB_STEP2B_SHARD_LOG_SUBDIR", log_subdir)
  log_prefix = get(ENV, "HNODECB_STEP2B_LOG_PREFIX", "log2_03_step2b_windowed_local")
  log_dir = normpath(joinpath(repo_root, "logs", shard_log_subdir))
  driver_log_dir = normpath(joinpath(repo_root, "logs", log_subdir))

  isfile(stage2_input_file) || error("Missing stage2light windowed result file: " * stage2_input_file)
  isfile(train_script) || error("Missing step2b trainer script: " * train_script)
  isdir(result_dir) || mkpath(result_dir)
  isdir(log_dir) || mkpath(log_dir)
  isdir(driver_log_dir) || mkpath(driver_log_dir)
  step2b_resume_enabled = env_truthy("HNODECB_STEP2B_RESUME"; default="1")

  stage2 = deserialize(stage2_input_file)
  haskey(stage2, :window_manifest) || error("Stage2light file does not contain :window_manifest: " * stage2_input_file)
  window_manifest = stage2.window_manifest
  isempty(window_manifest) && error("Stage2light file has empty window_manifest: " * stage2_input_file)

  merged_file = joinpath(result_dir, result_basename)
  if isfile(merged_file)
    rm(merged_file; force=true)
  end
  if !step2b_resume_enabled
    for i in 1:length(window_manifest)
      shard_file = joinpath(result_dir, shard_file_name(result_root, result_ext, i))
      if isfile(shard_file)
        rm(shard_file; force=true)
      end
    end
  end
  resume_log_suffix = (step2b_resume_enabled &&
    has_existing_step2b_shard_results(result_dir, result_root, result_ext, length(window_manifest))) ?
    next_resume_log_suffix(log_dir, log_prefix) : ""

  println("=== Step2b Windowed (03): auto window manifest ===")
  println("Project: ", project)
  println("Stage2light input: ", stage2_input_file)
  println("Train script: ", train_script)
  println("Result dir: ", result_dir)
  println("Merged result: ", merged_file)
  println("Resume: ", step2b_resume_enabled ? "ON" : "OFF",
    " | checkpoint_every=", get(ENV, "HNODECB_STEP2B_CHECKPOINT_EVERY", "5"))
  if resume_log_suffix != ""
    println("ResumeLogGroup: ", resume_log_suffix[2:end])
  end
  println("Windows: ", length(window_manifest))
  for (i, win) in enumerate(window_manifest)
    println("  W", i, " [", win.role, "]",
      " post_window=[", win.start_idx, ", ", win.stop_idx, "]",
      " len=", win.len,
      " | t=[", fmt_window_us(win.t_start), ", ", fmt_window_us(win.t_stop), "] us")
  end
  flush(stdout)

  procs = []
  ios = IO[]
  for (i, win) in enumerate(window_manifest)
    env = copy(ENV)
    env["HNODECB_STEP2B_RESULT_DIR"] = result_dir_name
    env["HNODECB_STEP2B_RESULT_NAME"] = shard_file_name(result_root, result_ext, i)
    env["HNODECB_STEP2B_WINDOW_START"] = string(win.start_idx)
    env["HNODECB_STEP2B_WINDOW_LEN"] = string(win.len)
    env["HNODECB_STEP2B_WINDOW_STOP"] = string(win.stop_idx)
    env["HNODECB_STEP2B_WINDOW_LABEL"] = win.label
    env["HNODECB_STEP2B_WINDOW_ROLE"] = win.role
    env["HNODECB_STEP2B_STAGE2LIGHT_FILTER_LABEL"] = win.label
    env["HNODECB_STEP2B_STAGE2LIGHT_FILTER_ROLE"] = win.role
    env["HNODECB_STEP2B_STAGE2_INPUT_BASENAME"] = stage2_input_basename

    logfile = shard_logfile(log_dir, log_prefix, i, resume_log_suffix)
    io = open(logfile, "w")
    cmd = `$(Base.julia_cmd()) --project=$project $train_script`
    p = run(pipeline(setenv(cmd, env), stdout=io, stderr=io); wait=false)
    push!(procs, p)
    push!(ios, io)
    println("  window shard ", i, "/", length(window_manifest), " -> ", logfile)
    flush(stdout)
  end

  for (i, p) in enumerate(procs)
    wait(p)
    if !success(p)
      error("Step2b window shard " * string(i) * " exited with failure. Inspect shard logs before merge.")
    end
  end
  for io in ios
    close(io)
  end

  window_results = NamedTuple[]
  for (i, win) in enumerate(window_manifest)
    shard_file = joinpath(result_dir, shard_file_name(result_root, result_ext, i))
    if !isfile(shard_file)
      error("Missing step2b window shard result file: " * shard_file)
    end
    if filesize(shard_file) == 0
      error("Empty step2b window shard result file: " * shard_file)
    end
    payload = deserialize(shard_file)
    push!(window_results, (
      window=win,
      payload=payload,
      results=(haskey(payload, :results) ? payload.results : Any[]),
      run_summaries=(haskey(payload, :run_summaries) ? payload.run_summaries : Any[]),
      best=best_from_payload(payload)
    ))
  end

  println("=== Step2b Windowed (03): merged summary ===")
  for (i, rec) in enumerate(window_results)
    if rec.best === nothing
      println("Window ", i, " [", rec.window.role, "] -- no successful result")
      continue
    end
    best = rec.best
    p_est = best.parameters_training
    ks_hat = p_est[10]
    cs_hat = p_est[11]
    println("Window ", i, " [", rec.window.role, "]",
      " | train=", @sprintf("%.4e", best.best_loss),
      " val=", @sprintf("%.4e", best.validation_resulting_cost),
      " | ks=", fmt_num(ks_hat),
      " cs=", fmt_num(cs_hat))
  end
  flush(stdout)

  serialize(merged_file, (
    mode="window_manifest",
    stage2light_input_basename=stage2_input_basename,
    window_manifest=window_manifest,
    window_results=window_results
  ))
  println("=== Step2b Windowed (03): merged per-window results -> ", merged_file, " ===")
  flush(stdout)
end
