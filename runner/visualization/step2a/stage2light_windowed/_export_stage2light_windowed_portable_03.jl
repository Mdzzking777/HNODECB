using Serialization
using ComponentArrays
using Optimisers

if !isdefined(ComponentArrays, :Shaped1DAxis) && isdefined(ComponentArrays, :ShapedAxis)
    @eval ComponentArrays const Shaped1DAxis = ShapedAxis
end

field_or(x, name::Symbol, default=nothing) = (x !== nothing && hasproperty(x, name)) ? getproperty(x, name) : default

sigmoid(x) = 1 / (1 + exp(-x))

function bound_param(raw, lo, hi)
    return lo + (hi - lo) * sigmoid(raw)
end

function portable_state(payload)
    selected = field_or(payload, :selected, Any[])
    if selected !== nothing && !isempty(selected)
        rec = selected[1]
        params = field_or(rec, :params, nothing)
        params === nothing && error("Stage2light result file lacks final params metadata.")
        p_vec = field_or(rec, :p_net_vec, nothing)
        p_vec === nothing && error("Stage2light result file lacks final p_net_vec.")
        return (
            source = :final,
            p_net_vec = Float64.(p_vec),
            nn_gain = Float64(field_or(rec, :nn_gain, get(params, "nn_gain", 1.0))),
            ks_hat = Float64(field_or(rec, :ks_hat, NaN)),
            cs_hat = Float64(field_or(rec, :cs_hat, NaN)),
            num_hidden_layers = Int(params["num_hidden_layers"]),
            num_hidden_nodes = Int(params["num_hidden_nodes"]),
            checkpoint_epoch = nothing,
        )
    end

    active_checkpoint = field_or(payload, :active_checkpoint, nothing)
    run_state = field_or(active_checkpoint, :run_state, nothing)
    theta = field_or(run_state, :theta, nothing)
    theta === nothing && error("Stage2light result file has neither finalized :selected result nor usable :active_checkpoint.run_state.theta.")

    bounds = field_or(payload, :bounds, nothing)
    ks_bounds_local = field_or(bounds, :ks, nothing)
    cs_bounds_local = field_or(bounds, :cs, nothing)
    (ks_bounds_local === nothing || cs_bounds_local === nothing) && error("Stage2light result file lacks mech bounds.")

    num_hidden_layers = Int(field_or(active_checkpoint, :num_hidden_layers, 0))
    num_hidden_nodes = Int(field_or(active_checkpoint, :num_hidden_nodes, 0))
    (num_hidden_layers < 0 || num_hidden_nodes <= 0) && error("Stage2light active checkpoint lacks usable NN shape metadata.")

    return (
        source = :checkpoint,
        p_net_vec = Float64.(copy(theta.p_net)),
        nn_gain = Float64(field_or(run_state, :nn_gain, 1.0)),
        ks_hat = Float64(bound_param(theta.mech_raw[1], ks_bounds_local[1], ks_bounds_local[2])),
        cs_hat = Float64(bound_param(theta.mech_raw[2], cs_bounds_local[1], cs_bounds_local[2])),
        num_hidden_layers = num_hidden_layers,
        num_hidden_nodes = num_hidden_nodes,
        checkpoint_epoch = Int(field_or(run_state, :epoch, 0)),
    )
end

function plain_window_meta(payload)
    win = field_or(payload, :window_meta, nothing)
    win === nothing && error("Stage2light result file lacks window_meta.")
    return (
        label = String(field_or(win, :label, "")),
        role = String(field_or(win, :role, "")),
        start = Int(field_or(win, :start, 0)),
        stop = Int(field_or(win, :stop, 0)),
        len = Int(field_or(win, :len, 0)),
    )
end

function main()
    length(ARGS) == 2 || error("usage: julia _export_stage2light_windowed_portable_03.jl <src_jld> <dst_cache>")
    src = normpath(abspath(ARGS[1]))
    dst = normpath(abspath(ARGS[2]))
    payload = deserialize(src)
    portable = (
        window_meta = plain_window_meta(payload),
        state = portable_state(payload),
        source_mtime = isfile(src) ? mtime(src) : 0.0,
        source_basename = basename(src),
    )
    mkpath(dirname(dst))
    serialize(dst, portable)
    println("Exported portable stage2light cache to: ", dst)
end

main()
