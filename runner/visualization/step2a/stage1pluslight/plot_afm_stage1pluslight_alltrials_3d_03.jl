ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Serialization
using Printf

field_or(rec, name::Symbol, default) = hasproperty(rec, name) ? getproperty(rec, name) : default
js_array(x) = repr(x)
param_or(params, key, default) = (params isa AbstractDict && haskey(params, key)) ? params[key] : default

function sci_tick_spec(lo::Float64, hi::Float64)
    lo_exp = floor(Int, log10(lo))
    hi_exp = ceil(Int, log10(hi))
    vals = collect(lo_exp:hi_exp)
    texts = ["1e" * string(e) for e in vals]
    return vals, texts
end

function empirical_quantile(sorted_vals::AbstractVector{<:Real}, p::Float64)
    n = length(sorted_vals)
    n == 0 && error("Cannot take quantile of empty vector.")
    idx = clamp(ceil(Int, p * n), 1, n)
    return Float64(sorted_vals[idx])
end

fmt_e_html(x) = (x isa Number && isfinite(x)) ? @sprintf("%.6e", x) : "None"
fmt_f_html(x; digits=2) = (x isa Number && isfinite(x)) ? @sprintf("%.*f", digits, x) : "None"

function window_roles(records)
    isempty(records) && return String[]
    params = field_or(first(records), :params, Dict{Any, Any}())
    roles = param_or(params, "window_roles", String[])
    roles isa AbstractVector || return String[]
    return [String(role) for role in roles]
end

function metric_specs(records)
    roles = window_roles(records)
    if isempty(roles)
        return [(
            slug="val_loss",
            label="val_loss",
            title="val_loss",
            idx=0,
            role="single"
        )]
    elseif length(roles) == 1
        return [(
            slug="w1_val_loss",
            label="W1 val_loss",
            title="W1 val_loss (" * roles[1] * ")",
            idx=1,
            role=roles[1]
        )]
    else
        specs = Any[(
            slug="mean_val_loss",
            label="mean val_loss",
            title="mean val_loss",
            idx=0,
            role="mean"
        )]
        for (i, role) in enumerate(roles)
            push!(specs, (
                slug="w" * string(i) * "_val_loss",
                label="W" * string(i) * " val_loss",
                title="W" * string(i) * " val_loss (" * role * ")",
                idx=i,
                role=role
            ))
        end
        return specs
    end
end

function metric_value(rec, spec)
    if spec.idx > 0
        vals = param_or(field_or(rec, :params, Dict{Any, Any}()), "window_val_losses", nothing)
        if vals isa AbstractVector && length(vals) >= spec.idx
            v = vals[spec.idx]
            return v isa Number ? Float64(v) : NaN
        end
    end
    v = field_or(rec, :val_loss, field_or(rec, :val_loss_start, NaN))
    return v isa Number ? Float64(v) : NaN
end

function metric_train_value(rec, spec)
    if spec.idx > 0
        vals = param_or(field_or(rec, :params, Dict{Any, Any}()), "window_train_losses", nothing)
        if vals isa AbstractVector && length(vals) >= spec.idx
            v = vals[spec.idx]
            return v isa Number ? Float64(v) : NaN
        end
    end
    v = field_or(rec, :train_loss, NaN)
    return v isa Number ? Float64(v) : NaN
end

function nn_err_value(rec)
    v = field_or(rec, :val_nn_err, field_or(rec, :val_nn_err_start, NaN))
    return v isa Number ? Float64(v) : NaN
end

function all_window_lines(rec)
    params = field_or(rec, :params, Dict{Any, Any}())
    roles = param_or(params, "window_roles", String[])
    vals = param_or(params, "window_val_losses", Float64[])
    trains = param_or(params, "window_train_losses", Float64[])
    (!(roles isa AbstractVector) || !(vals isa AbstractVector)) && return ""
    n = min(length(roles), length(vals))
    n == 0 && return ""
    io = IOBuffer()
    for i in 1:n
        role = String(roles[i])
        train_text = (trains isa AbstractVector && length(trains) >= i) ? fmt_e_html(trains[i]) : "None"
        print(io,
            "<br>W", i, " (", role, ") val_loss=", fmt_e_html(vals[i]),
            "<br>W", i, " (", role, ") train_loss=", train_text)
    end
    return String(take!(io))
end

function hover_text(rec, spec)
    parts = field_or(rec, :val_parts, nothing)
    params = field_or(rec, :params, Dict{Any, Any}())
    trial_id = get(params, "trial_id", -1)
    ks0 = param_or(params, "ks0", field_or(rec, :ks_hat, NaN))
    cs0 = param_or(params, "cs0", field_or(rec, :cs_hat, NaN))
    z_val = metric_value(rec, spec)
    z_train = metric_train_value(rec, spec)
    mean_val_loss = field_or(rec, :val_loss, field_or(rec, :val_loss_start, NaN))
    mean_train_loss = field_or(rec, :train_loss, NaN)
    nn_err = nn_err_value(rec)

    io = IOBuffer()
    print(io,
        "trial=", trial_id,
        "<br>", spec.label, "=", fmt_e_html(z_val),
        "<br>", spec.label, " train=", fmt_e_html(z_train),
        "<br>mean val_loss=", fmt_e_html(mean_val_loss),
        "<br>mean train_loss=", fmt_e_html(mean_train_loss),
        "<br>ks0=", fmt_e_html(ks0),
        "<br>cs0=", fmt_e_html(cs0),
        "<br>ks_end=", fmt_e_html(field_or(rec, :ks_hat, NaN)),
        "<br>cs_end=", fmt_e_html(field_or(rec, :cs_hat, NaN)),
        "<br>log10(ks0)=", fmt_f_html(log10(ks0); digits=6),
        "<br>log10(cs0)=", fmt_f_html(log10(cs0); digits=6),
        "<br>ks err=", fmt_f_html(field_or(rec, :ks_err_pct, NaN); digits=2), "%",
        "<br>cs err=", fmt_f_html(field_or(rec, :cs_err_pct, NaN); digits=2), "%",
        "<br>F_contact err=", fmt_f_html(nn_err; digits=2), "%")
    if parts !== nothing
        print(io,
            "<br>state=", fmt_e_html(parts.state),
            "<br>x2dot=", fmt_e_html(parts.x2dot),
            "<br>x1 rec err=", fmt_f_html(parts.x1_rec; digits=2), "%",
            "<br>x3 rec err=", fmt_f_html(parts.x3_rec; digits=2), "%")
    end
    print(io, all_window_lines(rec))
    return String(take!(io))
end

function write_plot_html(
    html_path::String,
    result_path::String,
    spec,
    kept_records,
    true_ks::Float64,
    true_cs::Float64,
    excluded_count::Int,
    z_display_max::Float64
)
    xs_raw = Float64[param_or(field_or(rec, :params, Dict{Any,Any}()), "ks0", field_or(rec, :ks_hat, NaN)) for rec in kept_records]
    ys_raw = Float64[param_or(field_or(rec, :params, Dict{Any,Any}()), "cs0", field_or(rec, :cs_hat, NaN)) for rec in kept_records]
    zs_raw = Float64[metric_value(rec, spec) for rec in kept_records]
    ferrs = Float64[nn_err_value(rec) for rec in kept_records]
    finite_ferrs = filter(isfinite, ferrs)
    isempty(finite_ferrs) && error("No finite val_nn_err values found in: " * result_path)
    hovertexts = [hover_text(rec, spec) for rec in kept_records]

    ferr_cap = 200.0
    ferr_overflow = ferr_cap + 1.0
    ferrs_color = Float64[
        !isfinite(f) ? NaN : (f > ferr_cap ? ferr_overflow : max(f, 0.0))
        for f in ferrs
    ]

    xs = log10.(xs_raw)
    ys = log10.(ys_raw)
    zs = log10.(zs_raw)
    highlight_mask = falses(length(kept_records))
    if !isempty(kept_records)
        n_highlight = max(1, ceil(Int, 0.001 * length(kept_records)))
        highlight_order = sortperm(zs_raw)
        highlight_mask[highlight_order[1:n_highlight]] .= true
    end
    base_mask = .!highlight_mask
    xs_base = xs[base_mask]
    ys_base = ys[base_mask]
    zs_base = zs[base_mask]
    ferrs_base = ferrs_color[base_mask]
    hovertexts_base = hovertexts[base_mask]
    xs_high = xs[highlight_mask]
    ys_high = ys[highlight_mask]
    zs_high = zs[highlight_mask]
    ferrs_high = ferrs_color[highlight_mask]
    hovertexts_high = hovertexts[highlight_mask]

    sorted_zs_raw = sort(zs_raw)
    zfocus_lo_raw = empirical_quantile(sorted_zs_raw, 0.01)
    zfocus_hi_raw = empirical_quantile(sorted_zs_raw, 0.995)
    zfocus_lo = log10(zfocus_lo_raw)
    zfocus_hi = log10(zfocus_hi_raw)
    zpad = max((zfocus_hi - zfocus_lo) * 0.12, 0.015)
    highlight_lo_raw = isempty(zs_high) ? zfocus_lo_raw : 10.0 ^ minimum(zs_high)
    highlight_lo = log10(highlight_lo_raw)
    highlight_pad = max((zfocus_lo - highlight_lo) * 0.08, 0.01)
    zfloor = min(zfocus_lo - zpad, highlight_lo - highlight_pad)
    ztop = zfocus_hi + zpad
    z_hidden_low = count(<(zfocus_lo_raw), zs_raw)
    z_hidden_high = count(>(zfocus_hi_raw), zs_raw)

    truth_x = isfinite(true_ks) ? log10(true_ks) : NaN
    truth_y = isfinite(true_cs) ? log10(true_cs) : NaN
    truth_z = zfloor

    xlo, xhi = minimum(xs), maximum(xs)
    ylo, yhi = minimum(ys), maximum(ys)

    xtickvals, xticktext = sci_tick_spec(minimum(xs_raw), maximum(xs_raw))
    ytickvals, yticktext = sci_tick_spec(minimum(ys_raw), maximum(ys_raw))
    ztickvals, zticktext = sci_tick_spec(zfocus_lo_raw, zfocus_hi_raw)
    ferr_tickvals = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, ferr_overflow]
    ferr_ticktext = ["0%", "25%", "50%", "75%", "100%", "125%", "150%", "175%", "200%", ">200%"]

    annotation_text =
        "source: " * replace(result_path, "\\" => "\\\\") *
        "<br>x=ks0 (log10), y=cs0 (log10), z=" * spec.label * " (log10)" *
        "<br>color uses stored F_contact err; excluded " * string(excluded_count) *
        " trials with " * spec.label * " > " * @sprintf("%.1e", z_display_max) *
        " or <= 0; z-focus uses q01..q99.5 and hides " * string(z_hidden_low) *
        " low / " * string(z_hidden_high) * " high outliers from the visible z span"

    open(html_path, "w") do io
        println(io, "<!DOCTYPE html>")
        println(io, "<html lang=\"en\">")
        println(io, "<head>")
        println(io, "  <meta charset=\"utf-8\">")
        println(io, "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
        println(io, "  <title>AFM03 Stage1pluslight all-trials log(ks0)-log(cs0)-" * spec.label * " 3D</title>")
        println(io, "  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>")
        println(io, "  <style>")
        println(io, "    body { margin: 0; font-family: sans-serif; background: #ffffff; }")
        println(io, "    #plot { width: 100vw; height: 100vh; }")
        println(io, "  </style>")
        println(io, "</head>")
        println(io, "<body>")
        println(io, "  <div id=\"plot\"></div>")
        println(io, "  <script>")
        println(io, "    const traceTrials = {")
        println(io, "      type: 'scatter3d',")
        println(io, "      mode: 'markers',")
        println(io, "      name: 'trials with " * spec.label * " <= " * @sprintf("%.1e", z_display_max) * "',")
        println(io, "      x: ", js_array(xs_base), ",")
        println(io, "      y: ", js_array(ys_base), ",")
        println(io, "      z: ", js_array(zs_base), ",")
        println(io, "      hovertext: ", js_array(hovertexts_base), ",")
        println(io, "      hovertemplate: '%{hovertext}<extra></extra>',")
        println(io, "      marker: {")
        println(io, "        size: 3,")
        println(io, "        opacity: 0.82,")
        println(io, "        color: ", js_array(ferrs_base), ",")
        println(io, "        cmin: 0.0,")
        println(io, "        cmax: ", repr(ferr_overflow), ",")
        println(io, "        colorscale: [")
        println(io, "          [0.0000, '#4b00ff'],")
        println(io, "          [0.1110, '#0057ff'],")
        println(io, "          [0.2220, '#00b7ff'],")
        println(io, "          [0.3330, '#00ffd0'],")
        println(io, "          [0.4440, '#2cff5c'],")
        println(io, "          [0.5550, '#b7ff00'],")
        println(io, "          [0.6660, '#ffe600'],")
        println(io, "          [0.7770, '#ff9a00'],")
        println(io, "          [0.8880, '#ff3b00'],")
        println(io, "          [0.9950, '#c40000'],")
        println(io, "          [0.9951, '#8c8c8c'],")
        println(io, "          [1.0000, '#8c8c8c']")
        println(io, "        ],")
        println(io, "        colorbar: {")
        println(io, "          title: 'F_contact err %',")
        println(io, "          tickmode: 'array',")
        println(io, "          tickvals: ", js_array(ferr_tickvals), ",")
        println(io, "          ticktext: ", js_array(ferr_ticktext))
        println(io, "        },")
        println(io, "        line: { width: 0 }")
        println(io, "      }")
        println(io, "    };")
        println(io, "    const traceHighlights = {")
        println(io, "      type: 'scatter3d',")
        println(io, "      mode: 'markers',")
        println(io, "      name: 'lowest 0.1% " * spec.label * " (black outline)',")
        println(io, "      x: ", js_array(xs_high), ",")
        println(io, "      y: ", js_array(ys_high), ",")
        println(io, "      z: ", js_array(zs_high), ",")
        println(io, "      hovertext: ", js_array(hovertexts_high), ",")
        println(io, "      hovertemplate: '%{hovertext}<extra></extra>',")
        println(io, "      marker: {")
        println(io, "        size: 5,")
        println(io, "        opacity: 1.0,")
        println(io, "        color: ", js_array(ferrs_high), ",")
        println(io, "        cmin: 0.0,")
        println(io, "        cmax: ", repr(ferr_overflow), ",")
        println(io, "        colorscale: [")
        println(io, "          [0.0000, '#4b00ff'],")
        println(io, "          [0.1110, '#0057ff'],")
        println(io, "          [0.2220, '#00b7ff'],")
        println(io, "          [0.3330, '#00ffd0'],")
        println(io, "          [0.4440, '#2cff5c'],")
        println(io, "          [0.5550, '#b7ff00'],")
        println(io, "          [0.6660, '#ffe600'],")
        println(io, "          [0.7770, '#ff9a00'],")
        println(io, "          [0.8880, '#ff3b00'],")
        println(io, "          [0.9950, '#c40000'],")
        println(io, "          [0.9951, '#8c8c8c'],")
        println(io, "          [1.0000, '#8c8c8c']")
        println(io, "        ],")
        println(io, "        showscale: false,")
        println(io, "        line: { color: '#000000', width: 4 }")
        println(io, "      }")
        println(io, "    };")
        if isfinite(true_ks) && isfinite(true_cs)
            println(io, "    const traceKsTrueLine = {")
            println(io, "      type: 'scatter3d',")
            println(io, "      mode: 'lines',")
            println(io, "      name: 'ks_true',")
            println(io, "      x: [", repr(truth_x), ", ", repr(truth_x), "],")
            println(io, "      y: [", repr(ylo), ", ", repr(yhi), "],")
            println(io, "      z: [", repr(zfloor), ", ", repr(zfloor), "],")
            println(io, "      hovertemplate: 'ks_true=", @sprintf("%.6e", true_ks), "<extra></extra>',")
            println(io, "      line: { color: '#111111', width: 2 }")
            println(io, "    };")
            println(io, "    const traceKsTrueLineTop = {")
            println(io, "      type: 'scatter3d',")
            println(io, "      mode: 'lines',")
            println(io, "      name: 'ks_true (top)',")
            println(io, "      showlegend: false,")
            println(io, "      x: [", repr(truth_x), ", ", repr(truth_x), "],")
            println(io, "      y: [", repr(ylo), ", ", repr(yhi), "],")
            println(io, "      z: [", repr(ztop), ", ", repr(ztop), "],")
            println(io, "      hovertemplate: 'ks_true=", @sprintf("%.6e", true_ks), "<extra></extra>',")
            println(io, "      line: { color: '#111111', width: 2 }")
            println(io, "    };")
            println(io, "    const traceCsTrueLine = {")
            println(io, "      type: 'scatter3d',")
            println(io, "      mode: 'lines',")
            println(io, "      name: 'cs_true',")
            println(io, "      x: [", repr(xlo), ", ", repr(xhi), "],")
            println(io, "      y: [", repr(truth_y), ", ", repr(truth_y), "],")
            println(io, "      z: [", repr(zfloor), ", ", repr(zfloor), "],")
            println(io, "      hovertemplate: 'cs_true=", @sprintf("%.6e", true_cs), "<extra></extra>',")
            println(io, "      line: { color: '#111111', width: 2 }")
            println(io, "    };")
            println(io, "    const traceCsTrueLineTop = {")
            println(io, "      type: 'scatter3d',")
            println(io, "      mode: 'lines',")
            println(io, "      name: 'cs_true (top)',")
            println(io, "      showlegend: false,")
            println(io, "      x: [", repr(xlo), ", ", repr(xhi), "],")
            println(io, "      y: [", repr(truth_y), ", ", repr(truth_y), "],")
            println(io, "      z: [", repr(ztop), ", ", repr(ztop), "],")
            println(io, "      hovertemplate: 'cs_true=", @sprintf("%.6e", true_cs), "<extra></extra>',")
            println(io, "      line: { color: '#111111', width: 2 }")
            println(io, "    };")
            println(io, "    const traceTruth = {")
            println(io, "      type: 'scatter3d',")
            println(io, "      mode: 'markers+text',")
            println(io, "      name: 'true (ks, cs)',")
            println(io, "      x: [", repr(truth_x), "],")
            println(io, "      y: [", repr(truth_y), "],")
            println(io, "      z: [", repr(truth_z), "],")
            println(io, "      text: ['truth'],")
            println(io, "      textposition: 'bottom center',")
            println(io, "      hovertemplate: 'true ks=", @sprintf("%.6e", true_ks), "<br>true cs=", @sprintf("%.6e", true_cs), "<extra></extra>',")
            println(io, "      marker: { size: 7, symbol: 'diamond', color: '#111111' }")
            println(io, "    };")
            traces_expr = "[traceTrials, traceHighlights, traceKsTrueLine, traceCsTrueLine, traceKsTrueLineTop, traceCsTrueLineTop, traceTruth]"
        else
            traces_expr = "[traceTrials, traceHighlights]"
        end
        println(io, "    const layout = {")
        println(io, "      title: 'AFM03 Stage1pluslight trials: ks0 vs cs0 vs " * spec.label * "',")
        println(io, "      paper_bgcolor: '#ffffff',")
        println(io, "      plot_bgcolor: '#ffffff',")
        println(io, "      legend: { orientation: 'h', x: 0, y: 1.02 },")
        println(io, "      margin: { l: 0, r: 0, t: 70, b: 0 },")
        println(io, "      annotations: [{")
        println(io, "        text: '", annotation_text, "',")
        println(io, "        x: 0, y: 1.08, xref: 'paper', yref: 'paper',")
        println(io, "        xanchor: 'left', yanchor: 'bottom',")
        println(io, "        showarrow: false, font: { size: 12, color: '#444444' }")
        println(io, "      }],")
        println(io, "      scene: {")
        println(io, "        xaxis: { title: 'ks', tickmode: 'array', tickvals: ", js_array(xtickvals), ", ticktext: ", js_array(xticktext), ", range: [", repr(xlo), ", ", repr(xhi), "] },")
        println(io, "        yaxis: { title: 'cs', tickmode: 'array', tickvals: ", js_array(ytickvals), ", ticktext: ", js_array(yticktext), ", range: [", repr(ylo), ", ", repr(yhi), "] },")
        println(io, "        zaxis: { title: '" * spec.label * "', tickmode: 'array', tickvals: ", js_array(ztickvals), ", ticktext: ", js_array(zticktext), ", range: [", repr(zfloor), ", ", repr(ztop), "] },")
        println(io, "        camera: { eye: { x: 1.5, y: 1.32, z: 0.92 } }")
        println(io, "      }")
        println(io, "    };")
        println(io, "    Plotly.newPlot('plot', ", traces_expr, ", layout, {responsive: true});")
        println(io, "  </script>")
        println(io, "</body>")
        println(io, "</html>")
    end
end

function main()
    result_path = isempty(ARGS) ? joinpath(
        @__DIR__, "..", "..", "..", "..",
        "step2a_hyperparameter_tuning", "hyperparameter_tuning_first_stage",
        "results_afm", "afm_param_stage1pluslight_03.jld"
    ) : ARGS[1]
    result_path = normpath(abspath(result_path))
    isfile(result_path) || error("Missing Stage1pluslight result file: " * result_path)

    default_out_dir = normpath(joinpath(
        @__DIR__, "..", "..", "..", "..",
        "logs", "stage1_step2a", "1pluslight", "local", "visualization"
    ))
    out_dir = length(ARGS) >= 2 ? normpath(abspath(ARGS[2])) : default_out_dir
    isdir(out_dir) || mkpath(out_dir)

    data = deserialize(result_path)
    haskey(data, :trial_parameters) || error("Result file does not contain :trial_parameters.")
    records = [
        rec for rec in data.trial_parameters
        if !field_or(rec, :early_stopped, false) &&
           !field_or(rec, :trial_failed, false) &&
           field_or(rec, :is_viable, true)
    ]
    isempty(records) && error("No trial records found in: " * result_path)

    z_display_max = 1e-1
    true_ks = haskey(data, :true_values) ? field_or(data.true_values, :ks, NaN) : NaN
    true_cs = haskey(data, :true_values) ? field_or(data.true_values, :cs, NaN) : NaN
    specs = metric_specs(records)
    stem = splitext(basename(result_path))[1]

    saved = String[]
    for spec in specs
        kept_records = [
            rec for rec in records
            if isfinite(metric_value(rec, spec)) &&
               metric_value(rec, spec) > 0.0 &&
               metric_value(rec, spec) <= z_display_max
        ]
        excluded_count = length(records) - length(kept_records)
        isempty(kept_records) && error("No trial records remain after z cutoff for " * spec.label * ".")

        html_path = joinpath(out_dir, stem * "_alltrials_ks_logcs_" * spec.slug * "_3d.html")
        write_plot_html(html_path, result_path, spec, kept_records, true_ks, true_cs, excluded_count, z_display_max)
        push!(saved, html_path)
        println("Saved 3D HTML to: ", html_path)
        println("Trials plotted for ", spec.label, ": ", length(kept_records))
        println("Trials excluded by z cutoff for ", spec.label, ": ", excluded_count)
    end

    println("Saved ", length(saved), " Stage1pluslight 3D plots.")
end

main()
