const STAGE2LIGHT_CACHE_DIR = normpath(joinpath(REPO_ROOT, "logs", "stage2_step2a", "local", "windowed", "visualization", ".cache"))
const STAGE2LIGHT_EXPORTER_PATH = normpath(joinpath(@__DIR__, "_export_stage2light_windowed_portable_03.jl"))
const STAGE2LIGHT_HPC_ENV = normpath(joinpath(REPO_ROOT, "HPC_env"))

function stage2light_portable_cache_path(path::String)
    stem = replace(basename(path), r"\.jld$" => ".portable.bin")
    return joinpath(STAGE2LIGHT_CACHE_DIR, stem)
end

function stage2light_ensure_portable_cache(path::String)
    cache_path = stage2light_portable_cache_path(path)
    cache_stale = !isfile(cache_path) || mtime(cache_path) < mtime(path)
    if cache_stale
        mkpath(dirname(cache_path))
        cmd = `$(Base.julia_cmd()) --project=$(STAGE2LIGHT_HPC_ENV) $(STAGE2LIGHT_EXPORTER_PATH) $(path) $(cache_path)`
        run(cmd)
    end
    return cache_path
end

function load_stage2_result_state(path::String)
    portable = deserialize(stage2light_ensure_portable_cache(path))
    state = portable.state
    win = portable.window_meta
    data = (window_meta = win,)
    return (
        source = state.source,
        data = data,
        p_net_vec = state.p_net_vec,
        nn_gain = state.nn_gain,
        ks_hat = state.ks_hat,
        cs_hat = state.cs_hat,
        num_hidden_layers = state.num_hidden_layers,
        num_hidden_nodes = state.num_hidden_nodes,
        checkpoint_epoch = state.checkpoint_epoch,
    )
end

function stage2light_checkpoint_plot_path(out_dir::String, default_file::String, recs)
    if any(rec -> get(rec, :source, :final) == :checkpoint, recs)
        max_epoch = maximum([Int(get(rec, :checkpoint_epoch, 0)) for rec in recs if get(rec, :source, :final) == :checkpoint]; init=0)
        stem = replace(default_file, r"\.png$" => "")
        return joinpath(out_dir, string(stem, "_checkpoint_epoch", max_epoch, ".png"))
    end
    return joinpath(out_dir, default_file)
end
