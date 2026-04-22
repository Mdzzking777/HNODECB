#=
Standalone Stage2 runner for AFM parameter search (case 03).
Requires the merged Stage1 `.jld` result file to exist.
=#

cd(@__DIR__)
using Serialization
using Printf
using DataFrames

if !haskey(ENV, "HNODECB_STAGE2_RESUME")
  ENV["HNODECB_STAGE2_RESUME"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE2_CHECKPOINT_EVERY")
  ENV["HNODECB_STAGE2_CHECKPOINT_EVERY"] = "5"
end

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

function nn_force_label(rec)
  if hasproperty(rec, :nn_force_mode)
    return rec.nn_force_mode == "contact_net" ? "F_contact" : "F_hertz"
  end
  if hasproperty(rec, :params) && rec.params isa AbstractDict && haskey(rec.params, "nn_force_mode")
    return rec.params["nn_force_mode"] == "contact_net" ? "F_contact" : "F_hertz"
  end
  return "F_contact"
end

env_truthy(name; default="0") = lowercase(strip(get(ENV, name, default))) in ("1", "true", "yes", "on")

peak_to_peak(v) = isempty(v) ? NaN : (maximum(v) - minimum(v))

function contact_onsets(contact_mask::AbstractVector{Bool})
  out = Int[]
  for i in eachindex(contact_mask)
    if contact_mask[i] && (i == firstindex(contact_mask) || !contact_mask[i - 1])
      push!(out, i)
    end
  end
  return out
end

function nearest_time_index(t_us::AbstractVector{<:Real}, target_us::Real)
  idx = findmin(abs.(Float64.(t_us) .- Float64(target_us)))[2]
  return Int(idx)
end

function start_anchor_index(t_us::AbstractVector{<:Real}, target_us::Real)
  idx = searchsortedfirst(Float64.(t_us), Float64(target_us))
  return clamp(Int(idx), 1, length(t_us))
end

function extend_stop_to_contact_end(contact_mask::AbstractVector{Bool}, stop_idx::Int)
  stop_idx = clamp(stop_idx, 1, length(contact_mask))
  contact_mask[stop_idx] || return stop_idx
  new_stop = stop_idx
  while new_stop < length(contact_mask) && contact_mask[new_stop + 1]
    new_stop += 1
  end
  return new_stop
end

function relocate_window_with_anchor(t_us::AbstractVector{<:Real}, start_idx::Int, stop_idx::Int; anchor::Symbol, target_us::Real)
  len = stop_idx - start_idx + 1
  n = length(t_us)
  len <= n || error("Window length exceeds available trajectory length.")

  if anchor == :start
    new_start = start_anchor_index(t_us, target_us)
    new_start = clamp(new_start, 1, n - len + 1)
    new_stop = new_start + len - 1
  elseif anchor == :stop
    new_stop = nearest_time_index(t_us, target_us)
    new_stop = clamp(new_stop, len, n)
    new_start = new_stop - len + 1
  else
    error("Unsupported anchor: " * string(anchor))
  end
  return new_start, new_stop
end

function apply_window_time_overrides(selected::Vector{<:NamedTuple}, t_post::AbstractVector{<:Real})
  overrides = Dict(
    "first_contact" => (anchor=:start, target_us=1034.0),
    "max_x1_pp_change" => (anchor=:stop, target_us=1057.0),
    "tail_stable" => (anchor=:stop, target_us=1998.0),
  )

  adjusted = NamedTuple[]
  for win in selected
    if haskey(overrides, win.role)
      spec = overrides[win.role]
      new_start, new_stop = relocate_window_with_anchor(t_post .* 1e6, win.start_idx, win.stop_idx; anchor=spec.anchor, target_us=spec.target_us)
      if win.role == "first_contact" && hasproperty(win, :contact_post)
        new_stop = extend_stop_to_contact_end(win.contact_post, new_stop)
      end
      push!(adjusted, merge(win, (
        start_idx=new_start,
        stop_idx=new_stop,
        len=new_stop - new_start + 1,
        t_start=t_post[new_start],
        t_stop=t_post[new_stop],
        manual_override=true,
        override_anchor=String(spec.anchor),
        override_target_us=Float64(spec.target_us)
      )))
    else
      push!(adjusted, merge(win, (
        manual_override=false,
        override_anchor="",
        override_target_us=NaN
      )))
    end
  end
  return adjusted
end

function build_window_manifest(repo_root::String)
  pert_file = normpath(joinpath(repo_root, "datasets", "e0.0", "data", "pert_df_afm_dmt_kv.jld"))
  ode_file = normpath(joinpath(repo_root, "datasets", "e0.0", "data", "ode_data_afm_dmt_kv.jld"))
  isfile(pert_file) || error("Missing perturbation data file: " * pert_file)
  isfile(ode_file) || error("Missing ODE data file: " * ode_file)

  solution_dataframe_full = deserialize(pert_file)
  ode_data_full = deserialize(ode_file)
  contact_full = Vector{Bool}(solution_dataframe_full.contact .== 1)
  first_contact_idx = findfirst(contact_full)
  first_contact_idx === nothing && error("No contact point found in the AFM dataset.")

  solution_post = solution_dataframe_full[first_contact_idx:end, :]
  t_post = Float64.(solution_post.t)
  contact_post = Vector{Bool}(solution_post.contact .== 1)
  x1_post = vec(Float64.(ode_data_full[1, first_contact_idx:end]))
  onset_post = contact_onsets(contact_post)
  length(onset_post) >= 3 || error("Need at least 3 contact onsets after first contact to build two-cycle windows.")

  cycle_pp = Float64[]
  for k in 1:(length(onset_post) - 1)
    lo = onset_post[k]
    hi = onset_post[k + 1] - 1
    hi >= lo || error("Invalid cycle bounds detected while building window manifest.")
    push!(cycle_pp, peak_to_peak(@view x1_post[lo:hi]))
  end

  candidates = NamedTuple[]
  for k in 1:(length(onset_post) - 2)
    start_idx = onset_post[k]
    stop_idx = onset_post[k + 2] - 1
    len = stop_idx - start_idx + 1
    pp1 = cycle_pp[k]
    pp2 = cycle_pp[k + 1]
    delta_pp = abs(pp2 - pp1)
    push!(candidates, (
      candidate_index=k,
      start_idx=start_idx,
      stop_idx=stop_idx,
      len=len,
      t_start=t_post[start_idx],
      t_stop=t_post[stop_idx],
      cycle1_index=k,
      cycle2_index=k + 1,
      x1_pp_cycle1=pp1,
      x1_pp_cycle2=pp2,
      x1_pp_delta=delta_pp
    ))
  end

  isempty(candidates) && error("No valid two-cycle windows were built from the post-contact trajectory.")

  first_idx = 1
  deltas = [cand.x1_pp_delta for cand in candidates]
  change_order = sortperm(deltas; rev=true)
  tail_order = collect(length(candidates):-1:1)

  selected = NamedTuple[]
  seen = Set{Tuple{Int, Int}}()
  function add_window(role::String, candidate_idx::Int)
    cand = candidates[candidate_idx]
    key = (cand.start_idx, cand.stop_idx)
    key in seen && return false
    push!(seen, key)
    push!(selected, merge(cand, (
      role=role,
      label=role * "_window",
      first_contact_idx=first_contact_idx,
      contact_post=contact_post
    )))
    return true
  end

  function add_first_unique(role::String, candidate_order)
    for candidate_idx in candidate_order
      add_window(role, candidate_idx) && return candidate_idx
    end
    return nothing
  end

  add_window("first_contact", first_idx)
  max_change_idx = add_first_unique("max_x1_pp_change", change_order)
  tail_idx = add_first_unique("tail_stable", tail_order)
  length(selected) >= 3 || error("Auto window manifest could not find 3 unique windows (found " * string(length(selected)) * ").")
  return apply_window_time_overrides(selected, t_post)
end

fmt_window_us(x; digits=3) = (x isa Number && isfinite(x)) ? @sprintf("%.*f", digits, 1e6 * x) : "None"

function print_window_summary(io::IO, rank::Int, win, rec, true_values)
  println(io, "Window ", rank, " [", win.role, "]",
    " -- post_window=[", win.start_idx, ", ", win.stop_idx, "]",
    " len=", win.len,
    " | t=[", fmt_window_us(win.t_start), ", ", fmt_window_us(win.t_stop), "] us")
  println(io, "  x1 pp cycles: c", win.cycle1_index, "=", fmt_num(win.x1_pp_cycle1),
    " c", win.cycle2_index, "=", fmt_num(win.x1_pp_cycle2),
    " | delta=", fmt_num(win.x1_pp_delta))
  println(io, "  best: train=", @sprintf("%.4e", rec.train_loss),
    " val=", @sprintf("%.4e", rec.val_loss))
  println(io, "  val parts: state=", fmt_part(rec, :state),
    " x2dot=", fmt_part(rec, :x2dot),
    " x3r=", fmt_part(rec, :x3_range),
    " cont=", fmt_part(rec, :cont),
    " l2=", fmt_part(rec, :l2))
  println(io, "  mech: ks=", fmt_num(hasproperty(rec, :ks_hat) ? rec.ks_hat : NaN),
    " (true=", fmt_num(true_values.ks), " err=", fmt_pct(hasproperty(rec, :ks_err_pct) ? rec.ks_err_pct : NaN), "%)",
    " cs=", fmt_num(hasproperty(rec, :cs_hat) ? rec.cs_hat : NaN),
    " (true=", fmt_num(true_values.cs), " err=", fmt_pct(hasproperty(rec, :cs_err_pct) ? rec.cs_err_pct : NaN), "%)")
  println(io, "  rec: x1=", fmt_pct((hasproperty(rec, :val_parts) && rec.val_parts !== nothing && hasproperty(rec.val_parts, :x1_rec)) ? rec.val_parts.x1_rec : NaN),
    "% x3=", fmt_pct((hasproperty(rec, :val_parts) && rec.val_parts !== nothing && hasproperty(rec.val_parts, :x3_rec)) ? rec.val_parts.x3_rec : NaN), "%")
end

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
  println("Resume: ", env_truthy("HNODECB_STAGE2_RESUME"; default="1") ? "ON" : "OFF",
    " | CheckpointEvery=", get(ENV, "HNODECB_STAGE2_CHECKPOINT_EVERY", "5"))
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
stage2_resume_enabled = env_truthy("HNODECB_STAGE2_RESUME"; default="1")

function regex_escape(raw::AbstractString)
  return replace(String(raw), r"([\\.^$|?*+()\[\]{}])" => s"\\\1")
end

function stage2_has_existing_shard_results(result_dir::String, shard_cnt::Int)
  for i in 1:shard_cnt
    shard_file = joinpath(result_dir, stage2_shard_file(i))
    if isfile(shard_file)
      try
        filesize(shard_file) > 0 && return true
      catch
      end
    end
  end
  return false
end

function stage2_next_resume_log_suffix(log_dir::String, log_prefix::String)
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

stage2_shard_logfile(log_dir::String, log_prefix::String, shard_idx::Int, resume_suffix::AbstractString="") =
  joinpath(log_dir, log_prefix * String(resume_suffix) * "_p" * string(shard_idx) * ".txt")

function stage2_resume_log_suffix(result_dir::String, log_dir::String, log_prefix::String, shard_cnt::Int, resume_enabled::Bool)
  if resume_enabled && stage2_has_existing_shard_results(result_dir, shard_cnt)
    return stage2_next_resume_log_suffix(log_dir, log_prefix)
  end
  return ""
end

stage1_file = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
  "hyperparameter_tuning_first_stage", "results_afm", stage2_input_basename))

if !isfile(stage1_file)
  error("Missing Stage1 result file required by Stage2: " * stage1_file)
end

did_autospawn = false
selftest = get(ENV, "HNODECB_SELFTEST", "0") == "1"

window_automanifest = env_truthy("HNODECB_STAGE2_WINDOW_AUTOMANIFEST")

if get(ENV, "HNODECB_STAGE2_AUTOSPAWN", "1") == "1" &&
   window_automanifest &&
   !haskey(ENV, "HNODECB_STAGE2_SHARD_INDEX")

  repo_root = normpath(joinpath(@__DIR__, "..", ".."))
  project = get(ENV, "HNODECB_JULIA_PROJECT", repo_root)
  stage2_path = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_second_stage", stage2_script_name))
  log_subdir = get(ENV, "HNODECB_STAGE2_LOG_SUBDIR", "")
  shard_log_subdir = get(ENV, "HNODECB_STAGE2_SHARD_LOG_SUBDIR", log_subdir)
  log_prefix = get(ENV, "HNODECB_STAGE2_LOG_PREFIX", "log2_03_step2a_stage2")
  log_dir = shard_log_subdir == "" ?
    normpath(joinpath(@__DIR__, "..", "..", "logs")) :
    normpath(joinpath(@__DIR__, "..", "..", "logs", shard_log_subdir))
  isdir(log_dir) || mkpath(log_dir)
  result_dir = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_second_stage", "results_afm"))
  isdir(result_dir) || mkpath(result_dir)
  merged_file = joinpath(result_dir, stage2_result_basename)
  if isfile(merged_file)
    rm(merged_file; force=true)
  end

  window_manifest = build_window_manifest(repo_root)
  window_cnt = length(window_manifest)
  if !stage2_resume_enabled
    for i in 1:window_cnt
      shard_file = joinpath(result_dir, stage2_shard_file(i))
      if isfile(shard_file)
        rm(shard_file; force=true)
      end
    end
  end
  resume_log_suffix = stage2_resume_log_suffix(result_dir, log_dir, log_prefix, window_cnt, stage2_resume_enabled)

  println("=== Stage2 (03): auto window manifest ===")
  println("CandidateFilter: ", get(ENV, "HNODECB_STAGE2_CANDIDATE_INDICES", ""))
  println("Resume: ", stage2_resume_enabled ? "ON" : "OFF",
    " | checkpoint_every=", get(ENV, "HNODECB_STAGE2_CHECKPOINT_EVERY", "5"))
  if resume_log_suffix != ""
    println("ResumeLogGroup: ", resume_log_suffix[2:end])
  end
  println("Windows: ", window_cnt)
  for (i, win) in enumerate(window_manifest)
    println("  W", i, " [", win.role, "]",
      " post_window=[", win.start_idx, ", ", win.stop_idx, "]",
      " len=", win.len,
      " | t=[", fmt_window_us(win.t_start), ", ", fmt_window_us(win.t_stop), "] us",
      " | x1_pp_delta=", fmt_num(win.x1_pp_delta))
  end
  flush(stdout)

  procs = []
  ios = IO[]
  for (i, win) in enumerate(window_manifest)
    env = copy(ENV)
    env["HNODECB_STAGE2_WINDOW_AUTOMANIFEST"] = "0"
    env["HNODECB_STAGE2_SHARD_COUNT"] = "1"
    delete!(env, "HNODECB_STAGE2_SHARD_INDEX")
    env["HNODECB_STAGE2_RESULT_BASENAME"] = stage2_shard_file(i)
    env["HNODECB_STAGE2_WINDOW_START"] = string(win.start_idx)
    env["HNODECB_STAGE2_WINDOW_LEN"] = string(win.len)
    env["HNODECB_STAGE2_WINDOW_LABEL"] = win.label
    env["HNODECB_STAGE2_WINDOW_ROLE"] = win.role
    env["HNODECB_STAGE2_WINDOW_STOP"] = string(win.stop_idx)
    env["HNODECB_STAGE2_WINDOW_TSTART_US"] = @sprintf("%.9f", 1e6 * win.t_start)
    env["HNODECB_STAGE2_WINDOW_TSTOP_US"] = @sprintf("%.9f", 1e6 * win.t_stop)
    env["HNODECB_STAGE2_WINDOW_CYCLE1_INDEX"] = string(win.cycle1_index)
    env["HNODECB_STAGE2_WINDOW_CYCLE2_INDEX"] = string(win.cycle2_index)
    env["HNODECB_STAGE2_WINDOW_X1_PP_CYCLE1"] = @sprintf("%.9e", win.x1_pp_cycle1)
    env["HNODECB_STAGE2_WINDOW_X1_PP_CYCLE2"] = @sprintf("%.9e", win.x1_pp_cycle2)
    env["HNODECB_STAGE2_WINDOW_X1_PP_DELTA"] = @sprintf("%.9e", win.x1_pp_delta)
    logfile = stage2_shard_logfile(log_dir, log_prefix, i, resume_log_suffix)
    io = open(logfile, "w")
    cmd = `$(Base.julia_cmd()) --project=$project $stage2_path`
    p = run(pipeline(setenv(cmd, env), stdout=io, stderr=io); wait=false)
    push!(procs, p)
    push!(ios, io)
    println("  window shard ", i, "/", window_cnt, " -> ", logfile)
    flush(stdout)
  end

  for (i, p) in enumerate(procs)
    wait(p)
    if !success(p)
      error("Stage2 window shard " * string(i) * " exited with failure. Inspect shard logs before merge.")
    end
  end
  for io in ios
    close(io)
  end

  if selftest
    println("=== Stage2 (03): window shard selftest OK ===")
    flush(stdout)
    did_autospawn = true
    return nothing
  end

  window_results = NamedTuple[]
  merged_trials = []
  local first_meta = nothing
  for (i, win) in enumerate(window_manifest)
    shard_file = joinpath(result_dir, stage2_shard_file(i))
    if !isfile(shard_file)
      error("Missing window shard result file: " * shard_file)
    end
    if filesize(shard_file) == 0
      error("Empty window shard result file: " * shard_file)
    end
    data = deserialize(shard_file)
    if first_meta === nothing
      first_meta = data
    end
    append!(merged_trials, data.trial_parameters)
    best_rec = if hasproperty(data, :selected) && !isempty(data.selected)
      data.selected[1]
    elseif hasproperty(data, :results) && !isempty(data.results)
      data.results[1]
    else
      nothing
    end
    push!(window_results, (
      window=win,
      trial_parameters=data.trial_parameters,
      results=(haskey(data, :results) ? data.results : Any[]),
      selected=(haskey(data, :selected) ? data.selected : Any[]),
      best=best_rec
    ))
  end

  println("=== Stage2 (03): per-window merged summary ===")
  for (i, winrec) in enumerate(window_results)
    if winrec.best === nothing
      println("Window ", i, " [", winrec.window.role, "] -- no successful result")
    else
      print_window_summary(stdout, i, winrec.window, winrec.best, first_meta.true_values)
    end
  end
  flush(stdout)

  serialize(merged_file, (
    mode="window_manifest",
    window_manifest=window_manifest,
    window_results=window_results,
    trial_parameters=merged_trials,
    bounds=first_meta.bounds,
    true_values=first_meta.true_values,
    use_multiple_shooting=first_meta.use_multiple_shooting,
    l2_grid=first_meta.l2_grid,
    error_level=first_meta.error_level
  ))
  println("=== Stage2 (03): merged per-window results -> ", merged_file, " ===")
  println("=== AFM Stage2-only (03) complete ===")
  flush(stdout)
  did_autospawn = true

elseif get(ENV, "HNODECB_STAGE2_AUTOSPAWN", "1") == "1" &&
   get(ENV, "HNODECB_STAGE2_SHARD_COUNT", "1") != "1" &&
   !haskey(ENV, "HNODECB_STAGE2_SHARD_INDEX")

  shard_cnt = parse(Int, ENV["HNODECB_STAGE2_SHARD_COUNT"])
  repo_root = normpath(joinpath(@__DIR__, "..", ".."))
  project = get(ENV, "HNODECB_JULIA_PROJECT", repo_root)
  stage2_path = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_second_stage", stage2_script_name))
  log_subdir = get(ENV, "HNODECB_STAGE2_LOG_SUBDIR", "")
  shard_log_subdir = get(ENV, "HNODECB_STAGE2_SHARD_LOG_SUBDIR", log_subdir)
  log_prefix = get(ENV, "HNODECB_STAGE2_LOG_PREFIX", "log2_03_step2a_stage2")
  log_dir = shard_log_subdir == "" ?
    normpath(joinpath(@__DIR__, "..", "..", "logs")) :
    normpath(joinpath(@__DIR__, "..", "..", "logs", shard_log_subdir))
  isdir(log_dir) || mkpath(log_dir)
  result_dir = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_second_stage", "results_afm"))
  isdir(result_dir) || mkpath(result_dir)
  merged_file = joinpath(result_dir, stage2_result_basename)
  if isfile(merged_file)
    rm(merged_file; force=true)
  end
  if !stage2_resume_enabled
    for i in 1:shard_cnt
      shard_file = joinpath(result_dir, stage2_shard_file(i))
      if isfile(shard_file)
        rm(shard_file; force=true)
      end
    end
  end
  resume_log_suffix = stage2_resume_log_suffix(result_dir, log_dir, log_prefix, shard_cnt, stage2_resume_enabled)

  println("=== Stage2 (03): spawning ", shard_cnt, " shards ===")
  println("Resume: ", stage2_resume_enabled ? "ON" : "OFF",
    " | checkpoint_every=", get(ENV, "HNODECB_STAGE2_CHECKPOINT_EVERY", "5"))
  if resume_log_suffix != ""
    println("ResumeLogGroup: ", resume_log_suffix[2:end])
  end
  flush(stdout)
  procs = []
  ios = IO[]
  for i in 1:shard_cnt
    env = copy(ENV)
    env["HNODECB_STAGE2_SHARD_INDEX"] = string(i)
    env["HNODECB_STAGE2_SHARD_COUNT"] = string(shard_cnt)
    logfile = stage2_shard_logfile(log_dir, log_prefix, i, resume_log_suffix)
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
    println("     nn: ", nn_force_label(rec), " err=", fmt_pct(hasproperty(rec, :val_nn_err) ? rec.val_nn_err : NaN), "%")
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
