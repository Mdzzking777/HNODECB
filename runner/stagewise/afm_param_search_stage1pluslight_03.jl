#=
Standalone Stage1pluslight runner for AFM parameter search (case 03).
Supports shard autospawn + merge, then emits a merged hall-of-fame.
=#

cd(@__DIR__)
using Serialization
using Printf
using Statistics

if !haskey(ENV, "HNODECB_STAGE1_VARIANT")
  ENV["HNODECB_STAGE1_VARIANT"] = "stage1pluslight"
end
if !haskey(ENV, "HNODECB_STAGE1_RESULT_STEM")
  ENV["HNODECB_STAGE1_RESULT_STEM"] = "afm_param_stage1pluslight_03"
end
if !haskey(ENV, "HNODECB_STAGE1PLUSLIGHT_MODE")
  ENV["HNODECB_STAGE1PLUSLIGHT_MODE"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_GRID_KS_NODES")
  ENV["HNODECB_STAGE1PLUS_GRID_KS_NODES"] = "50"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_GRID_CS_NODES")
  ENV["HNODECB_STAGE1PLUS_GRID_CS_NODES"] = "50"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_GRID_NN_SEEDS_PER_NODE")
  ENV["HNODECB_STAGE1PLUS_GRID_NN_SEEDS_PER_NODE"] = "50"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE")
  grid_ks = parse(Int, get(ENV, "HNODECB_STAGE1PLUS_GRID_KS_NODES", "50"))
  grid_cs = parse(Int, get(ENV, "HNODECB_STAGE1PLUS_GRID_CS_NODES", "50"))
  grid_nn = parse(Int, get(ENV, "HNODECB_STAGE1PLUS_GRID_NN_SEEDS_PER_NODE", "50"))
  ENV["HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE"] = string(grid_ks * grid_cs * grid_nn)
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_FINAL_TOPK")
  ENV["HNODECB_STAGE1PLUS_FINAL_TOPK"] = "0"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ZERO_NN")
  ENV["HNODECB_STAGE1PLUS_ZERO_NN"] = "0"
end
if !haskey(ENV, "HNODECB_STAGE1_SHARD_COUNT")
  ENV["HNODECB_STAGE1_SHARD_COUNT"] = "4"
end
if !haskey(ENV, "HNODECB_STAGE1_AUTOSPAWN")
  ENV["HNODECB_STAGE1_AUTOSPAWN"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_ENABLE")
  ENV["HNODECB_STAGE1PLUS_ARCH_SCREEN_ENABLE"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH")
  ENV["HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH"] = "60"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_EPOCHS")
  ENV["HNODECB_STAGE1PLUS_ARCH_EPOCHS"] = "50"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_WINDOW_MODE")
  ENV["HNODECB_STAGE1PLUS_WINDOW_MODE"] = "stage2_w123"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_RESUME")
  ENV["HNODECB_STAGE1PLUS_RESUME"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_CHECKPOINT_EVERY")
  ENV["HNODECB_STAGE1PLUS_CHECKPOINT_EVERY"] = "100"
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
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_PARETO_PLOT")
  ENV["HNODECB_STAGE1PLUS_ARCH_PARETO_PLOT"] = "1"
end
if !haskey(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_COUNT")
  ENV["HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_COUNT"] = "3"
end
if !haskey(ENV, "HNODECB_STAGE1_RUN_SEED") || isempty(strip(get(ENV, "HNODECB_STAGE1_RUN_SEED", "")))
  ENV["HNODECB_STAGE1_RUN_SEED"] = "314159265"
end

run_label = get(ENV, "HNODECB_STAGE1_RUN_LABEL", "Stage1pluslight")

fmt_e(x; sigdigits=4) = (x isa Number && isfinite(x)) ? @sprintf("%.*e", max(sigdigits - 1, 0), x) : "None"
fmt_f(x; digits=2) = (x isa Number && isfinite(x)) ? @sprintf("%.*f", digits, x) : "None"

function regex_escape(raw::AbstractString)
  return replace(String(raw), r"([\\.^$|?*+()\[\]{}])" => s"\\\1")
end

function stage1plus_has_existing_shard_results(result_dir::String, result_stem::String, shard_cnt::Int)
  for i in 1:shard_cnt
    shard_file = joinpath(result_dir, result_stem * "_p" * string(i) * ".jld")
    if isfile(shard_file)
      try
        filesize(shard_file) > 0 && return true
      catch
      end
    end
  end
  return false
end

function stage1plus_next_resume_log_suffix(log_dir::String, log_prefix::String)
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

stage1plus_shard_logfile(log_dir::String, log_prefix::String, shard_idx::Int, resume_suffix::AbstractString="") =
  joinpath(log_dir, log_prefix * String(resume_suffix) * "_p" * string(shard_idx) * ".txt")

function nn_force_label(rec)
  if hasproperty(rec, :nn_force_mode)
    return rec.nn_force_mode == "contact_net" ? "F_contact" : "F_hertz"
  end
  if hasproperty(rec, :params) && rec.params isa AbstractDict && haskey(rec.params, "nn_force_mode")
    return rec.params["nn_force_mode"] == "contact_net" ? "F_contact" : "F_hertz"
  end
  return "F_contact"
end

function env_float(key, default)
  raw = get(ENV, key, "")
  if isempty(raw)
    return default
  end
  parsed = tryparse(Float64, raw)
  return parsed === nothing ? default : parsed
end

function env_int(key, default)
  raw = strip(get(ENV, key, string(default)))
  parsed = tryparse(Int, raw)
  return parsed === nothing ? default : parsed
end

function normalize_min_obj_metric(values::Vector{Float64})
  finite_vals = [v for v in values if isfinite(v)]
  if isempty(finite_vals)
    return fill(1.0, length(values))
  end
  vmin = minimum(finite_vals)
  vmax = maximum(finite_vals)
  span = vmax - vmin
  if !isfinite(span) || span <= 0
    return [isfinite(v) ? 0.0 : 1.0 for v in values]
  end
  return [isfinite(v) ? (v - vmin) / span : 1.0 for v in values]
end

function summarize_architecture_trials_runner(arch_trials, obj_a::Float64, obj_b::Float64, obj_c::Float64)
  grouped = Dict{Tuple{Int, Int}, Vector{Any}}()
  for rec in arch_trials
    key = (rec.num_hidden_layers, rec.num_hidden_nodes)
    if !haskey(grouped, key)
      grouped[key] = Any[]
    end
    push!(grouped[key], rec)
  end

  summaries = Any[]
  for (key, recs) in grouped
    push!(summaries, (
      num_hidden_layers=key[1],
      num_hidden_nodes=key[2],
      hidden=2^key[2],
      param_count=Int(round(mean([rec.param_count for rec in recs]))),
      n_trials=length(recs),
      mean_obj1=mean([rec.obj1 for rec in recs]),
      mean_obj2=mean([rec.obj2 for rec in recs]),
      mean_obj3=mean([rec.obj3 for rec in recs]),
      mean_retry_count=mean([rec.retry_count_total for rec in recs]),
      mean_completed_epochs=mean([rec.completed_epochs for rec in recs]),
      mean_val_loss_start=mean([rec.val_loss_start for rec in recs]),
      mean_val_loss_end=mean([rec.val_loss_end for rec in recs])
    ))
  end

  sort!(summaries, by = rec -> (rec.num_hidden_layers, rec.num_hidden_nodes))
  obj1_norm = normalize_min_obj_metric(Float64[rec.mean_obj1 for rec in summaries])
  obj2_norm = normalize_min_obj_metric(Float64[rec.mean_obj2 for rec in summaries])
  obj3_norm = normalize_min_obj_metric(Float64[rec.mean_obj3 for rec in summaries])
  scored = Any[]
  for i in eachindex(summaries)
    rec = summaries[i]
    weighted_obj1 = obj_a * obj1_norm[i]
    weighted_obj2 = obj_b * obj2_norm[i]
    weighted_obj3 = obj_c * obj3_norm[i]
    min_obj = sqrt(weighted_obj1^2 + weighted_obj2^2 + weighted_obj3^2)
    push!(scored, merge(rec, (
      norm_obj1=obj1_norm[i],
      norm_obj2=obj2_norm[i],
      norm_obj3=obj3_norm[i],
      weighted_obj1=weighted_obj1,
      weighted_obj2=weighted_obj2,
      weighted_obj3=weighted_obj3,
      min_obj=min_obj
    )))
  end
  sort!(scored, by = rec -> rec.min_obj)
  return scored
end

function architecture_ranking_lines(arch_ranked, arch_weights)
  lines = String[]
  push!(lines, run_label * " architecture ranking (9 architectures; unified normalization across architecture means, weighted L2 min_obj selection):")
  for (rank, rec) in enumerate(arch_ranked)
    push!(lines,
      "  Rank " * string(rank) *
      " -- layers=" * string(rec.num_hidden_layers) *
      " nodes=" * string(rec.num_hidden_nodes) *
      " hidden=" * string(rec.hidden) *
      " params=" * string(rec.param_count) *
      " | min_obj=" * fmt_e(rec.min_obj, sigdigits=4))
    push!(lines,
      "     min_obj detail: (" * fmt_f(arch_weights.a, digits=2) * ")*part_a=" *
      fmt_e(rec.weighted_obj1, sigdigits=4) *
      ", raw_a=" * fmt_e(rec.mean_obj1, sigdigits=4) *
      ", norm_a=" * fmt_e(rec.norm_obj1, sigdigits=4) *
      " ; (" * fmt_f(arch_weights.b, digits=2) * ")*part_b=" *
      fmt_e(rec.weighted_obj2, sigdigits=4) *
      ", raw_b=" * fmt_e(rec.mean_obj2, sigdigits=4) *
      ", norm_b=" * fmt_e(rec.norm_obj2, sigdigits=4) *
      " ; (" * fmt_f(arch_weights.c, digits=2) * ")*part_c=" *
      fmt_e(rec.weighted_obj3, sigdigits=4) *
      ", raw_c=" * fmt_e(rec.mean_obj3, sigdigits=4) *
      ", norm_c=" * fmt_e(rec.norm_obj3, sigdigits=4))
  end
  return lines
end

function write_architecture_ranking_txt(path::String, arch_ranked, arch_weights, selected_arch; source_path::String="")
  open(path, "w") do io
    println(io, "AFM03 ", run_label, " architecture global ranking")
    if source_path != ""
      println(io, "source = ", source_path)
    end
    println(io, "")
    for line in architecture_ranking_lines(arch_ranked, arch_weights)
      println(io, line)
    end
    println(io, "")
    println(io, "Selected architecture -> layers=", selected_arch.num_hidden_layers,
      " nodes=", selected_arch.num_hidden_nodes,
      " hidden=", selected_arch.hidden,
      " | min_obj=", fmt_e(selected_arch.min_obj, sigdigits=4))
  end
end

did_autospawn = false

if get(ENV, "HNODECB_STAGE1_AUTOSPAWN", "1") == "1" &&
   get(ENV, "HNODECB_STAGE1_SHARD_COUNT", "1") != "1" &&
   !haskey(ENV, "HNODECB_STAGE1_SHARD_INDEX")

  shard_cnt = parse(Int, ENV["HNODECB_STAGE1_SHARD_COUNT"])
  repo_root = normpath(joinpath(@__DIR__, "..", ".."))
  project = get(ENV, "HNODECB_JULIA_PROJECT", repo_root)
  stage1plus_path = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_first_stage", "afm_param_stage1pluslight_03.jl"))
  pareto_plot_path = get(ENV, "HNODECB_STAGE1_ARCH_PARETO_SCRIPT",
    normpath(joinpath(@__DIR__, "..", "visualization", "step2a", "stage1pluslight", "plot_afm_stage1plus_archscreen_pareto_03.jl")))
  log_subdir = get(ENV, "HNODECB_STAGE1PLUSLIGHT_LOG_SUBDIR", "stage1_step2a")
  log_prefix = get(ENV, "HNODECB_STAGE1PLUSLIGHT_LOG_PREFIX", "log2_03_step2a_stage1pluslight")
  arch_shard_log_subdir = get(ENV, "HNODECB_STAGE1_ARCH_SHARD_LOG_SUBDIR", "archi_screen_shard_p")
  log_dir = normpath(joinpath(@__DIR__, "..", "..", "logs", log_subdir))
  isdir(log_dir) || mkpath(log_dir)
  visualization_dir = joinpath(log_dir, "visualization")
  isdir(visualization_dir) || mkpath(visualization_dir)
  ranking_subdir = strip(get(ENV, "HNODECB_STAGE1_ARCH_RANKING_SUBDIR", ""))
  ranking_dir = isempty(ranking_subdir) ? log_dir : joinpath(log_dir, ranking_subdir)
  isdir(ranking_dir) || mkpath(ranking_dir)
  result_dir = normpath(joinpath(@__DIR__, "..", "..", "step2a_hyperparameter_tuning",
    "hyperparameter_tuning_first_stage", "results_afm"))
  isdir(result_dir) || mkpath(result_dir)

  result_stem = get(ENV, "HNODECB_STAGE1_RESULT_STEM", "afm_param_stage1pluslight_03")
  resume_enabled = get(ENV, "HNODECB_STAGE1PLUS_RESUME", "1") != "0"
  merged_file = joinpath(result_dir, result_stem * ".jld")
  archscreen_file = joinpath(result_dir, result_stem * "_archscreen.jld")
  if isfile(merged_file)
    rm(merged_file; force=true)
  end
  if isfile(archscreen_file)
    rm(archscreen_file; force=true)
  end
  if !resume_enabled
    for i in 1:shard_cnt
      shard_file = joinpath(result_dir, result_stem * "_p" * string(i) * ".jld")
      if isfile(shard_file)
        rm(shard_file; force=true)
      end
    end
  end
  weight_shard_log_dir = joinpath(log_dir, "weight_search_shard_p")
  isdir(weight_shard_log_dir) || mkpath(weight_shard_log_dir)
  resume_log_suffix = if resume_enabled &&
      stage1plus_has_existing_shard_results(result_dir, result_stem, shard_cnt)
    stage1plus_next_resume_log_suffix(weight_shard_log_dir, log_prefix)
  else
    ""
  end

  arch_screen_only = get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_ONLY", "0") == "1"
  arch_screen_shard_cnt = max(1, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_COUNT", "1")))
  have_selected_arch = haskey(ENV, "HNODECB_STAGE1PLUS_SELECTED_LAYERS") && haskey(ENV, "HNODECB_STAGE1PLUS_SELECTED_NODES")
  inherit_stage1_arch = get(ENV, "HNODECB_STAGE1PLUS_INHERIT_STAGE1_ARCH", "1") != "0"
  manual_layers = env_int("HNODECB_STAGE1PLUS_MANUAL_LAYERS", 0)
  manual_nodes = env_int("HNODECB_STAGE1PLUS_MANUAL_NODES", 1)
  local archscreen_data = nothing
  stage1_input_spec = strip(get(ENV, "HNODECB_STAGE1_INPUT_BASENAME", "afm_param_stage1_03.jld"))
  stage1_input_file = isabspath(stage1_input_spec) ? normpath(stage1_input_spec) : joinpath(result_dir, stage1_input_spec)
  if arch_screen_only
    arch_trials = Any[]
    local first_arch_meta = nothing
    for i in 1:arch_screen_shard_cnt
      shard_arch_file = joinpath(result_dir, result_stem * "_p" * string(i) * ".jld")
      if isfile(shard_arch_file)
        rm(shard_arch_file; force=true)
      end
    end
    if arch_screen_shard_cnt == 1
      println("=== ", run_label, " (03): architecture screening ===")
      flush(stdout)
      arch_env = copy(ENV)
      arch_env["HNODECB_STAGE1_AUTOSPAWN"] = "0"
      arch_env["HNODECB_STAGE1PLUS_ARCH_SCREEN_ONLY"] = "1"
      arch_env["HNODECB_STAGE1PLUS_ARCH_SCREEN_EMIT_LOCAL_RANKING"] = "0"
      arch_cmd = `$(Base.julia_cmd()) --project=$project $stage1plus_path`
      run(setenv(arch_cmd, arch_env))
      if !isfile(merged_file)
        error("Missing architecture screening result file: " * merged_file)
      end
      archscreen_raw = deserialize(merged_file)
      first_arch_meta = archscreen_raw
      append!(arch_trials, archscreen_raw.trial_parameters)
    else
      println("=== ", run_label, " (03): architecture screening with ", arch_screen_shard_cnt, " shards ===")
      flush(stdout)
      arch_procs = []
      arch_ios = IO[]
      for i in 1:arch_screen_shard_cnt
        arch_env_i = copy(ENV)
        arch_env_i["HNODECB_STAGE1_AUTOSPAWN"] = "0"
        arch_env_i["HNODECB_STAGE1PLUS_ARCH_SCREEN_ONLY"] = "1"
        arch_env_i["HNODECB_STAGE1PLUS_ARCH_SCREEN_EMIT_LOCAL_RANKING"] = "0"
        arch_env_i["HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_INDEX"] = string(i)
        arch_env_i["HNODECB_STAGE1PLUS_ARCH_SCREEN_SHARD_COUNT"] = string(arch_screen_shard_cnt)
        arch_env_i["HNODECB_STAGE1_RESULT_STEM"] = result_stem * "_p" * string(i)
        shard_log_dir = joinpath(log_dir, arch_shard_log_subdir)
        isdir(shard_log_dir) || mkpath(shard_log_dir)
        logfile = joinpath(shard_log_dir, log_prefix * "_p" * string(i) * ".txt")
        io = open(logfile, "w")
        arch_cmd_i = `$(Base.julia_cmd()) --project=$project $stage1plus_path`
        proc = run(pipeline(setenv(arch_cmd_i, arch_env_i), stdout=io, stderr=io); wait=false)
        push!(arch_procs, proc)
        push!(arch_ios, io)
        println("  arch shard ", i, "/", arch_screen_shard_cnt, " -> ", logfile)
        flush(stdout)
      end
      for (i, proc) in enumerate(arch_procs)
        wait(proc)
        if !success(proc)
          error("Architecture screening shard " * string(i) * " exited with failure. Inspect architecture shard logs before merge.")
        end
      end
      for io in arch_ios
        close(io)
      end

      for i in 1:arch_screen_shard_cnt
        shard_arch_file = joinpath(result_dir, result_stem * "_p" * string(i) * ".jld")
        if !isfile(shard_arch_file)
          error("Missing architecture screening shard result file: " * shard_arch_file)
        end
        if filesize(shard_arch_file) == 0
          error("Empty architecture screening shard result file: " * shard_arch_file)
        end
        archscreen_shard_data = deserialize(shard_arch_file)
        if first_arch_meta === nothing
          first_arch_meta = archscreen_shard_data
        end
        append!(arch_trials, archscreen_shard_data.trial_parameters)
      end
    end
    if first_arch_meta === nothing
      error("Architecture screening merge failed: no shard metadata found")
    end

    arch_weights = haskey(first_arch_meta, :arch_obj_weights) ? first_arch_meta.arch_obj_weights : (
      a=env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_A", 0.35),
      b=env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_B", 0.45),
      c=env_float("HNODECB_STAGE1PLUS_ARCH_OBJ_C", 0.20)
    )
    arch_ranked = summarize_architecture_trials_runner(
      arch_trials, arch_weights.a, arch_weights.b, arch_weights.c)
    isempty(arch_ranked) && error("Architecture screening merge produced no ranked architectures")
    selected_arch = arch_ranked[1]
    ranking_txt_path = joinpath(ranking_dir, result_stem * "_archscreen_ranking.txt")
    write_architecture_ranking_txt(ranking_txt_path, arch_ranked, arch_weights, selected_arch; source_path=merged_file)
    println("=== ", run_label, " (03): merged architecture ranking (9 architectures) ===")
    for line in architecture_ranking_lines(arch_ranked, arch_weights)
      println(line)
    end
    println("Saved architecture ranking TXT to: ", ranking_txt_path)

    archscreen_data = (
      study=nothing,
      trial_parameters=arch_trials,
      warm_start_top=Any[],
      selected=arch_ranked,
      best=selected_arch,
      architecture_ranked=arch_ranked,
      selected_architecture=(
        num_hidden_layers=selected_arch.num_hidden_layers,
        num_hidden_nodes=selected_arch.num_hidden_nodes
      ),
      bounds=first_arch_meta.bounds,
      true_values=first_arch_meta.true_values,
      use_multiple_shooting=first_arch_meta.use_multiple_shooting,
      use_l2_regularization=first_arch_meta.use_l2_regularization,
      val_stride=first_arch_meta.val_stride,
      val_offset=first_arch_meta.val_offset,
      error_level=first_arch_meta.error_level,
      stage1_input_file=(haskey(first_arch_meta, :stage1_input_file) ? first_arch_meta.stage1_input_file : ""),
      stage1_input_topk=(haskey(first_arch_meta, :stage1_input_topk) ? first_arch_meta.stage1_input_topk : 0),
      arch_trials_per_arch=(haskey(first_arch_meta, :arch_trials_per_arch) ? first_arch_meta.arch_trials_per_arch :
        parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_TRIALS_PER_ARCH", "20"))),
      arch_epochs=(haskey(first_arch_meta, :arch_epochs) ? first_arch_meta.arch_epochs :
        parse(Int, get(ENV, "HNODECB_STAGE1PLUS_ARCH_EPOCHS", "50"))),
      arch_window_us=(haskey(first_arch_meta, :arch_window_us) ? first_arch_meta.arch_window_us :
        env_float("HNODECB_STAGE1PLUS_ARCH_WINDOW_US", 5e-6)),
      arch_lr=(haskey(first_arch_meta, :arch_lr) ? first_arch_meta.arch_lr :
        env_float("HNODECB_STAGE1PLUS_ARCH_LR", 1e-5)),
      arch_obj_weights=arch_weights,
      arch_norm_scope=(haskey(first_arch_meta, :arch_norm_scope) ? first_arch_meta.arch_norm_scope :
        "global_across_architecture_means"),
      arch_trial_logs=(haskey(first_arch_meta, :arch_trial_logs) ? first_arch_meta.arch_trial_logs : "raw_only"),
      arch_selection_metric=(haskey(first_arch_meta, :arch_selection_metric) ?
        first_arch_meta.arch_selection_metric : "weighted_l2_min_obj"),
      arch_pareto_axes=(haskey(first_arch_meta, :arch_pareto_axes) ?
        first_arch_meta.arch_pareto_axes : "unweighted_norm_obj1_obj2_obj3")
    )
    serialize(merged_file, archscreen_data)
    selected_arch = archscreen_data.selected_architecture
    ENV["HNODECB_STAGE1PLUS_SELECTED_LAYERS"] = string(selected_arch.num_hidden_layers)
    ENV["HNODECB_STAGE1PLUS_SELECTED_NODES"] = string(selected_arch.num_hidden_nodes)
    println("=== ", run_label, " (03): architecture screen complete ===")
    println("Selected architecture -> layers=", selected_arch.num_hidden_layers,
      " nodes=", selected_arch.num_hidden_nodes)
    if get(ENV, "HNODECB_STAGE1PLUS_ARCH_PARETO_PLOT", "1") == "1"
      try
        println("=== ", run_label, " (03): rendering archi screening Pareto plot ===")
        flush(stdout)
        plot_env = copy(ENV)
        plot_env["GKSwstype"] = get(plot_env, "GKSwstype", "100")
        plot_cmd = `$(Base.julia_cmd()) --project=$repo_root $pareto_plot_path $merged_file $visualization_dir`
        run(setenv(plot_cmd, plot_env))
      catch ex
        println("WARN: architecture Pareto plot generation failed: ", sprint(showerror, ex))
      end
    end
    flush(stdout)
    did_autospawn = true
  elseif !have_selected_arch && inherit_stage1_arch
    isfile(stage1_input_file) || error("Missing Stage1 architecture screening result file: " * stage1_input_file)
    archscreen_data = deserialize(stage1_input_file)
    selected_arch = if haskey(archscreen_data, :selected_architecture)
      archscreen_data.selected_architecture
    elseif haskey(archscreen_data, :architecture_screen) && haskey(archscreen_data.architecture_screen, :selected_architecture)
      archscreen_data.architecture_screen.selected_architecture
    else
      error("Stage1 file does not contain selected_architecture: " * stage1_input_file)
    end
    ENV["HNODECB_STAGE1PLUS_SELECTED_LAYERS"] = string(selected_arch.num_hidden_layers)
    ENV["HNODECB_STAGE1PLUS_SELECTED_NODES"] = string(selected_arch.num_hidden_nodes)
    println("=== ", run_label, " (03): inheriting Stage1 archi screening ===")
    println("Stage1 input -> ", stage1_input_file)
    println("Selected architecture -> layers=", selected_arch.num_hidden_layers,
      " nodes=", selected_arch.num_hidden_nodes)
    flush(stdout)
  elseif !have_selected_arch && !inherit_stage1_arch
    ENV["HNODECB_STAGE1PLUS_SELECTED_LAYERS"] = string(manual_layers)
    ENV["HNODECB_STAGE1PLUS_SELECTED_NODES"] = string(manual_nodes)
    println("=== ", run_label, " (03): manual NN architecture override ===")
    println("Manual architecture -> layers=", manual_layers,
      " nodes=", manual_nodes)
    flush(stdout)
  elseif isfile(stage1_input_file)
    try
      archscreen_data = deserialize(stage1_input_file)
    catch
      archscreen_data = nothing
    end
  end

  if !did_autospawn
  println("=== ", run_label, " (03): spawning ", shard_cnt, " shards ===")
  println("Standalone joint search: trials=", get(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE", "250000"),
    " | ks_nodes=", get(ENV, "HNODECB_STAGE1PLUS_GRID_KS_NODES", "50"),
    " cs_nodes=", get(ENV, "HNODECB_STAGE1PLUS_GRID_CS_NODES", "50"),
    " nn_seeds_per_node=", get(ENV, "HNODECB_STAGE1PLUS_GRID_NN_SEEDS_PER_NODE", "50"),
    " | final_topk=", get(ENV, "HNODECB_STAGE1PLUS_FINAL_TOPK", "0"),
    " | stage1_input=", stage1_input_file,
    " | resume=", resume_enabled ? "ON" : "OFF",
    " | checkpoint_every=", get(ENV, "HNODECB_STAGE1PLUS_CHECKPOINT_EVERY", "100"))
  if resume_log_suffix != ""
    println("ResumeLogGroup: ", resume_log_suffix[2:end])
  end
  if haskey(ENV, "HNODECB_STAGE1PLUS_SELECTED_LAYERS") && haskey(ENV, "HNODECB_STAGE1PLUS_SELECTED_NODES")
    arch_mode = inherit_stage1_arch ? "stage1-selected" : "manual-override"
    println("Selected architecture (", arch_mode, "): layers=", ENV["HNODECB_STAGE1PLUS_SELECTED_LAYERS"],
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
    env["HNODECB_STAGE1PLUS_EMIT_LOCAL_SELECTION"] = "0"
    logfile = stage1plus_shard_logfile(weight_shard_log_dir, log_prefix, i, resume_log_suffix)
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
      error(run_label * " shard " * string(i) * " exited with failure. Inspect shard logs before merge.")
    end
  end
  for io in ios
    close(io)
  end

  merged_trials_all = Any[]
  local first_meta = nothing
  for i in 1:shard_cnt
    shard_file = joinpath(result_dir, result_stem * "_p" * string(i) * ".jld")
    if !isfile(shard_file)
      error("Missing shard result file: " * shard_file)
    end
    if filesize(shard_file) == 0
      error("Empty shard result file: " * shard_file)
    end
    shard_data = deserialize(shard_file)
    if first_meta === nothing
      first_meta = shard_data
    end
    append!(merged_trials_all, shard_data.trial_parameters)
  end

  viable_trials = [rec for rec in merged_trials_all if hasproperty(rec, :is_viable) && rec.is_viable]
  early_stopped_trials = [rec for rec in merged_trials_all if hasproperty(rec, :early_stopped) && rec.early_stopped]
  failed_trials = [rec for rec in merged_trials_all if hasproperty(rec, :trial_failed) && rec.trial_failed]
  selection_pool = viable_trials
  sorted = sort(selection_pool, by = r -> r.loss)
  final_topk = max(0, parse(Int, get(ENV, "HNODECB_STAGE1PLUS_FINAL_TOPK", "0")))
  selected = final_topk > 0 ? sorted[1:min(final_topk, length(sorted))] : Any[]
  warm_start_top = final_topk > 0 ? [(
      trial_id=rec.params["trial_id"],
      params=rec.params,
      p_net=copy(rec.p_net_vec),
      train_loss=rec.train_loss,
      val_loss=rec.val_loss
    ) for rec in selected if hasproperty(rec, :p_net_vec)] : Any[]

  ranking_txt_path = joinpath(log_dir, result_stem * "_ranking.txt")
  if final_topk > 0
    println("=== ", run_label, " (03): hall of fame (merged top-", length(selected), ") ===")
    for (rank, rec) in enumerate(selected)
      println("  Rank ", rank,
        " -- train=", fmt_e(rec.train_loss, sigdigits=4),
        " val=", fmt_e(rec.val_loss, sigdigits=4),
        " | trial=", rec.params["trial_id"],
        " | ks0=", fmt_e(rec.params["ks0"], sigdigits=3),
        " cs0=", fmt_e(rec.params["cs0"], sigdigits=3))
      if hasproperty(rec, :val_parts)
        println("     val parts: state=", fmt_e(rec.val_parts.state, sigdigits=3),
          " x2dot=", fmt_e(rec.val_parts.x2dot, sigdigits=3),
          " x3r=", fmt_e(rec.val_parts.x3_range, sigdigits=3),
          " cont=", fmt_e(rec.val_parts.cont, sigdigits=3))
        println("     rec: x1=", fmt_f(rec.val_parts.x1_rec, digits=2),
          "% x3=", fmt_f(rec.val_parts.x3_rec, digits=2), "%")
      end
      println("     mech: ks=", fmt_e(hasproperty(rec, :ks_hat) ? rec.ks_hat : NaN, sigdigits=3),
        " (err=", fmt_f(hasproperty(rec, :ks_err_pct) ? rec.ks_err_pct : NaN, digits=2), "%)",
        " cs=", fmt_e(hasproperty(rec, :cs_hat) ? rec.cs_hat : NaN, sigdigits=3),
        " (err=", fmt_f(hasproperty(rec, :cs_err_pct) ? rec.cs_err_pct : NaN, digits=2), "%)")
      println("     nn: ", nn_force_label(rec), " err=", fmt_f(hasproperty(rec, :val_nn_err) ? rec.val_nn_err : NaN, digits=2), "%")
    end
    open(ranking_txt_path, "w") do io
      println(io, "AFM03 ", run_label, " standalone top-", length(selected), " ranking")
      println(io, "source = ", merged_file)
      println(io)
      for (rank, rec) in enumerate(selected)
        println(io, "Rank ", rank,
          " -- train=", fmt_e(rec.train_loss, sigdigits=4),
          " val=", fmt_e(rec.val_loss, sigdigits=4),
          " | trial=", rec.params["trial_id"],
          " | ks0=", fmt_e(rec.params["ks0"], sigdigits=3),
          " cs0=", fmt_e(rec.params["cs0"], sigdigits=3))
        if hasproperty(rec, :val_parts)
          println(io, "  val parts: state=", fmt_e(rec.val_parts.state, sigdigits=3),
            " x2dot=", fmt_e(rec.val_parts.x2dot, sigdigits=3),
            " x3r=", fmt_e(rec.val_parts.x3_range, sigdigits=3),
            " cont=", fmt_e(rec.val_parts.cont, sigdigits=3))
          println(io, "  rec: x1=", fmt_f(rec.val_parts.x1_rec, digits=2),
            "% x3=", fmt_f(rec.val_parts.x3_rec, digits=2), "%")
        end
        println(io, "  mech: ks=", fmt_e(hasproperty(rec, :ks_hat) ? rec.ks_hat : NaN, sigdigits=3),
          " (err=", fmt_f(hasproperty(rec, :ks_err_pct) ? rec.ks_err_pct : NaN, digits=2), "%)",
          " cs=", fmt_e(hasproperty(rec, :cs_hat) ? rec.cs_hat : NaN, sigdigits=3),
          " (err=", fmt_f(hasproperty(rec, :cs_err_pct) ? rec.cs_err_pct : NaN, digits=2), "%)")
        println(io, "  nn: ", nn_force_label(rec), " err=",
          fmt_f(hasproperty(rec, :val_nn_err) ? rec.val_nn_err : NaN, digits=2), "%")
      end
    end
    println("Saved ", run_label, " ranking TXT to: ", ranking_txt_path)
    flush(stdout)
  else
    isfile(ranking_txt_path) && rm(ranking_txt_path; force=true)
    println("=== ", run_label, " (03): final Top-K selection DISABLED; merged viable trial set only ===")
  end

  println(run_label, " merge filter: total=", length(merged_trials_all),
    " | viable=", length(viable_trials),
    " | early_stop_excluded=", length(early_stopped_trials),
    " | failed_excluded=", length(failed_trials))

  serialize(merged_file, (
    study=nothing,
    trial_parameters=viable_trials,
    trial_parameters_all=merged_trials_all,
    warm_start_top=warm_start_top,
    selected=selected,
    best=isempty(selected) ? nothing : selected[1],
    bounds=first_meta.bounds,
    true_values=first_meta.true_values,
    use_multiple_shooting=first_meta.use_multiple_shooting,
    use_l2_regularization=first_meta.use_l2_regularization,
    val_stride=first_meta.val_stride,
    val_offset=first_meta.val_offset,
    error_level=first_meta.error_level,
    stage1_input_file=stage1_input_file,
    stage1_input_topk=(haskey(first_meta, :stage1_input_topk) ? first_meta.stage1_input_topk : 0),
    searches_per_candidate=parse(Int, get(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE", "250000")),
    final_topk=final_topk,
    shard_count=shard_cnt,
    stage1plus_standalone=(haskey(first_meta, :stage1plus_standalone) ? first_meta.stage1plus_standalone : false),
    architecture_screen=archscreen_data
  ))
  println("=== ", run_label, " (03): merged results -> ", merged_file, " ===")
  flush(stdout)
  did_autospawn = true
  end
end

if !did_autospawn
  let
    threads = get(ENV, "JULIA_NUM_THREADS", "1")
    trials = get(ENV, "HNODECB_STAGE1PLUS_TRIALS_PER_CANDIDATE", "250000")
    result_stem = get(ENV, "HNODECB_STAGE1_RESULT_STEM", "afm_param_stage1pluslight_03")
    stage1_input_spec = strip(get(ENV, "HNODECB_STAGE1_INPUT_BASENAME", "afm_param_stage1_03.jld"))
    println("=== ", run_label, " (03) runtime config ===")
    println("Threads: ", threads, " | StandaloneJointSearch: ON | Trials: ", trials,
      " | ks_nodes=", get(ENV, "HNODECB_STAGE1PLUS_GRID_KS_NODES", "50"),
      " cs_nodes=", get(ENV, "HNODECB_STAGE1PLUS_GRID_CS_NODES", "50"),
      " nn_seeds_per_node=", get(ENV, "HNODECB_STAGE1PLUS_GRID_NN_SEEDS_PER_NODE", "50"))
    println("Stage1Input: ", stage1_input_spec, " | ResultStem: ", result_stem)
    flush(stdout)
  end

  println("=== AFM ", run_label, " (03) ===")
  flush(stdout)
  include("../../step2a_hyperparameter_tuning/hyperparameter_tuning_first_stage/afm_param_stage1pluslight_03.jl")
  println("=== AFM ", run_label, " (03) complete ===")
  flush(stdout)
end
