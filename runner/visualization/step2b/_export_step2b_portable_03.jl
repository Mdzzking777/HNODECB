using Serialization
using ComponentArrays

function find_repo_root(start_dir::String)
    dir = normpath(abspath(start_dir))
    while true
        if isfile(joinpath(dir, "run_step2b_windowed_03_local.ps1")) || isdir(joinpath(dir, "logs"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not locate repo root from: " * start_dir)
        dir = parent
    end
end

const REPO_ROOT = find_repo_root(@__DIR__)

include(joinpath(REPO_ROOT, "test_case_settings", "afm_dmt_kv_settings", "afm_dmt_kv_model_settings.jl"))

function layer_syms(p_net)
    syms = Symbol[s for s in propertynames(p_net) if startswith(String(s), "layer_")]
    sort!(syms; by = s -> something(tryparse(Int, split(String(s), "_")[end]), typemax(Int)))
    return syms
end

function plainify_p_net(p_net)
    pairs = Pair{Symbol, NamedTuple}[]
    for lname in layer_syms(p_net)
        layer = getproperty(p_net, lname)
        push!(pairs, lname => (
            weight = Matrix{Float64}(layer.weight),
            bias = Vector{Float64}(layer.bias),
        ))
    end
    return (; pairs...)
end

function portable_best(payload)
    haskey(payload, :results) || error("Step2b result file lacks :results.")
    isempty(payload.results) && error("Step2b result file has no finalized :results yet. Wait until step2b finishes before exporting portable visualization cache.")
    vals = [r.validation_resulting_cost for r in payload.results]
    best = payload.results[argmin(vals)]
    return (
        parameters_training = Float64.(best.parameters_training),
        p_net = plainify_p_net(best.p_net),
        nn_gain = hasproperty(best, :nn_gain) ? Float64(best.nn_gain) : 1.0,
        num_hidden_layers = hasproperty(best, :num_hidden_layers) ? Int(best.num_hidden_layers) : 0,
        num_hidden_nodes = hasproperty(best, :num_hidden_nodes) ? Int(best.num_hidden_nodes) : 2,
        training_resulting_cost = Float64(best.training_resulting_cost),
        validation_resulting_cost = Float64(best.validation_resulting_cost),
    )
end

function plainify_window(win)
    return (
        enabled = Bool(win.enabled),
        start_idx = Int(win.start_idx),
        stop_idx = Int(win.stop_idx),
        len = Int(win.len),
        label = String(win.label),
        role = String(win.role),
    )
end

function main()
    length(ARGS) == 2 || error("usage: julia _export_step2b_portable_03.jl <src_jld> <dst_cache>")
    src = normpath(abspath(ARGS[1]))
    dst = normpath(abspath(ARGS[2]))
    payload = deserialize(src)
    portable = (
        window = plainify_window(payload.window),
        best = portable_best(payload),
        source_mtime = isfile(src) ? mtime(src) : 0.0,
        source_basename = basename(src),
    )
    mkpath(dirname(dst))
    serialize(dst, portable)
    println("Exported portable step2b cache to: ", dst)
end

main()
