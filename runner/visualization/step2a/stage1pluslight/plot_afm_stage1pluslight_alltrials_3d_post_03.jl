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

function linear_tick_spec(lo::Float64, hi::Float64; max_ticks::Int=8)
    if !isfinite(lo) || !isfinite(hi)
        return Float64[], String[]
    end
    hi < lo && ((lo, hi) = (hi, lo))
    if hi == lo
        vals = [lo]
        texts = [abs(lo) >= 1e4 ? @sprintf("%.1e", lo) : @sprintf("%.0f%%", lo)]
        return vals, texts
    end

    span = hi - lo
    raw_step = span / max(max_ticks - 1, 1)
    mag = 10.0 ^ floor(log10(raw_step))
    unit = raw_step / mag
    nice_unit =
        unit <= 1.0 ? 1.0 :
        unit <= 2.0 ? 2.0 :
        unit <= 5.0 ? 5.0 : 10.0
    step = nice_unit * mag

    start = floor(lo / step) * step
    stop = ceil(hi / step) * step
    vals = collect(start:step:stop)
    texts = [abs(v) >= 1e4 || step >= 1e3 ? @sprintf("%.1e", v) : @sprintf("%.0f%%", v) for v in vals]
    return vals, texts
end

function sci_linear_tick_spec(lo::Float64, hi::Float64; max_ticks::Int=6)
    vals, _ = linear_tick_spec(lo, hi; max_ticks=max_ticks)
    texts = [@sprintf("%.3e", v) for v in vals]
    return vals, texts
end

function empirical_quantile(v::Vector{Float64}, p::Float64)
    isempty(v) && return NaN
    s = sort(v)
    idx = clamp(Int(ceil(p * length(s))), 1, length(s))
    return s[idx]
end

function hover_text(rec)
    parts = rec.val_parts
    trial_id = get(rec.params, "trial_id", -1)
    ks0 = param_or(rec.params, "ks0", rec.ks_hat)
    cs0 = param_or(rec.params, "cs0", rec.cs_hat)
    val_loss_start = field_or(rec, :val_loss_start, rec.val_loss)
    val_drop_pct = 100 * (val_loss_start - rec.val_loss) / max(abs(val_loss_start), 1e-30)
    return @sprintf(
        "trial=%d<br>completed_epochs=%d<br>val_loss_start=%.6e<br>val_loss_end=%.6e<br>val_drop=%.2f%%<br>train_loss_end=%.6e<br>ks0=%.6e<br>cs0=%.6e<br>ks_end=%.6e<br>cs_end=%.6e<br>log10(ks_end)=%.6f<br>log10(cs_end)=%.6f<br>ks err=%.2f%%<br>cs err=%.2f%%<br>F_contact err start=%.2f%%<br>F_contact err end=%.2f%%<br>state=%.6e<br>x2dot=%.6e<br>x1 rec err=%.2f%%<br>x3 rec err=%.2f%%",
        trial_id,
        field_or(rec, :completed_epochs, 0),
        val_loss_start,
        rec.val_loss,
        val_drop_pct,
        rec.train_loss,
        ks0,
        cs0,
        rec.ks_hat,
        rec.cs_hat,
        log10(rec.ks_hat),
        log10(rec.cs_hat),
        rec.ks_err_pct,
        rec.cs_err_pct,
        field_or(rec, :val_nn_err_start, NaN),
        rec.val_nn_err,
        parts.state,
        parts.x2dot,
        parts.x1_rec,
        parts.x3_rec
    )
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

    viable_records = [
        rec for rec in data[:trial_parameters]
        if field_or(rec, :is_viable, false) &&
           !field_or(rec, :early_stopped, true) &&
           !field_or(rec, :trial_failed, true) &&
           isfinite(field_or(rec, :val_loss, NaN)) &&
           isfinite(field_or(rec, :train_loss, NaN)) &&
           isfinite(field_or(rec, :val_nn_err, NaN)) &&
           isfinite(field_or(rec, :ks_hat, NaN)) &&
           isfinite(field_or(rec, :cs_hat, NaN))
    ]
    isempty(viable_records) && error("No viable trial records found in: " * result_path)

    default_epochs = maximum(field_or(rec, :completed_epochs, 0) for rec in viable_records)
    target_epochs = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : default_epochs

    records = [
        rec for rec in viable_records
        if field_or(rec, :completed_epochs, 0) == target_epochs
    ]
    isempty(records) && error("No viable trial records with completed_epochs == $(target_epochs).")

    z_display_max = 1e-1
    kept_records = [
        rec for rec in records
        if field_or(rec, :val_loss, NaN) <= z_display_max
    ]
    excluded_count = length(records) - length(kept_records)
    isempty(kept_records) && error("No viable post-epoch records remain after z cutoff <= 1e-1.")

    xs_raw = Float64[field_or(rec, :ks_hat, NaN) for rec in kept_records]
    ys_raw = Float64[field_or(rec, :cs_hat, NaN) for rec in kept_records]
    zs_raw = Float64[field_or(rec, :val_loss, NaN) for rec in kept_records]
    ferrs = Float64[field_or(rec, :val_nn_err, NaN) for rec in kept_records]
    hovertexts = [hover_text(rec) for rec in kept_records]

    true_ks = haskey(data, :true_values) ? field_or(data.true_values, :ks, NaN) : NaN
    true_cs = haskey(data, :true_values) ? field_or(data.true_values, :cs, NaN) : NaN

    xs = log10.(xs_raw)
    ys = log10.(ys_raw)
    zmin_raw = minimum(filter(isfinite, zs_raw))
    zq01_raw = empirical_quantile(filter(isfinite, zs_raw), 0.01)
    zq99_raw = empirical_quantile(filter(isfinite, zs_raw), 0.99)
    zpad = max((zq99_raw - zq01_raw) * 0.08, max(abs(zq01_raw), 1e-30) * 1e-4)
    zs = zs_raw
    zfloor = min(zmin_raw, zq01_raw) - zpad
    ztop = zq99_raw + zpad
    truth_x = isfinite(true_ks) ? log10(true_ks) : NaN
    truth_y = isfinite(true_cs) ? log10(true_cs) : NaN
    truth_z = zfloor

    xlo, xhi = minimum(xs), maximum(xs)
    ylo, yhi = minimum(ys), maximum(ys)

    xtickvals, xticktext = sci_tick_spec(minimum(xs_raw), maximum(xs_raw))
    ytickvals, yticktext = sci_tick_spec(minimum(ys_raw), maximum(ys_raw))
    ztickvals, zticktext = sci_linear_tick_spec(zfloor, ztop)

    nn_color_cap = 150.0
    color_records = [i for (i, ferr) in enumerate(ferrs) if isfinite(ferr) && ferr <= nn_color_cap]
    gray_records = [i for (i, ferr) in enumerate(ferrs) if !isfinite(ferr) || ferr > nn_color_cap]
    color_x = [xs[i] for i in color_records]
    color_y = [ys[i] for i in color_records]
    color_z = [zs[i] for i in color_records]
    color_f = [ferrs[i] for i in color_records]
    color_h = [hovertexts[i] for i in color_records]
    gray_x = [xs[i] for i in gray_records]
    gray_y = [ys[i] for i in gray_records]
    gray_z = [zs[i] for i in gray_records]
    gray_h = [hovertexts[i] for i in gray_records]
    ferr_tickvals, ferr_ticktext = linear_tick_spec(0.0, nn_color_cap)

    stem = splitext(basename(result_path))[1]
    html_path = joinpath(out_dir, stem * "_alltrials_ks_logcs_loss_3d_post_epoch" * string(target_epochs) * ".html")

    open(html_path, "w") do io
        println(io, "<!DOCTYPE html>")
        println(io, "<html lang=\"en\">")
        println(io, "<head>")
        println(io, "  <meta charset=\"utf-8\">")
        println(io, "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
        println(io, "  <title>AFM03 Stage1pluslight all-trials post-gradient ks-cs-loss 3D</title>")
        println(io, "  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>")
        println(io, "  <style>")
        println(io, "    body { margin: 0; font-family: sans-serif; background: #ffffff; }")
        println(io, "    #plot { width: 100vw; height: 100vh; }")
        println(io, "  </style>")
        println(io, "</head>")
        println(io, "<body>")
        println(io, "  <div id=\"plot\"></div>")
        println(io, "  <script>")
        println(io, "    const traceTrialsGray = {")
        println(io, "      type: 'scatter3d',")
        println(io, "      mode: 'markers',")
        println(io, "      name: 'NN err > 150%',")
        println(io, "      x: ", js_array(gray_x), ",")
        println(io, "      y: ", js_array(gray_y), ",")
        println(io, "      z: ", js_array(gray_z), ",")
        println(io, "      hovertext: ", js_array(gray_h), ",")
        println(io, "      hovertemplate: '%{hovertext}<extra></extra>',")
        println(io, "      marker: { size: 3, opacity: 0.55, color: '#B8B8B8', line: { width: 0 } }")
        println(io, "    };")
        println(io, "    const traceTrialsColor = {")
        println(io, "      type: 'scatter3d',")
        println(io, "      mode: 'markers',")
        println(io, "      name: '0% <= NN err <= 150%',")
        println(io, "      x: ", js_array(color_x), ",")
        println(io, "      y: ", js_array(color_y), ",")
        println(io, "      z: ", js_array(color_z), ",")
        println(io, "      hovertext: ", js_array(color_h), ",")
        println(io, "      hovertemplate: '%{hovertext}<extra></extra>',")
        println(io, "      marker: {")
        println(io, "        size: 3,")
        println(io, "        opacity: 0.82,")
        println(io, "        color: ", js_array(color_f), ",")
        println(io, "        cmin: 0.0,")
        println(io, "        cmax: ", repr(nn_color_cap), ",")
        println(io, "        colorscale: [")
        println(io, "          [0.00, '#2c1e7f'],")
        println(io, "          [0.08, '#2146b7'],")
        println(io, "          [0.16, '#1e88e5'],")
        println(io, "          [0.24, '#22c1c3'],")
        println(io, "          [0.32, '#2ec27e'],")
        println(io, "          [0.40, '#7ad151'],")
        println(io, "          [0.48, '#bddf26'],")
        println(io, "          [0.56, '#fde725'],")
        println(io, "          [0.64, '#f9c74f'],")
        println(io, "          [0.72, '#f8961e'],")
        println(io, "          [0.80, '#f3722c'],")
        println(io, "          [0.88, '#e85d04'],")
        println(io, "          [0.94, '#d00000'],")
        println(io, "          [1.00, '#6a040f']")
        println(io, "        ],")
        println(io, "        colorbar: {")
        println(io, "          title: 'F_contact err end %',")
        println(io, "          tickmode: 'array',")
        println(io, "          tickvals: ", js_array(ferr_tickvals), ",")
        println(io, "          ticktext: ", js_array(ferr_ticktext))
        println(io, "        },")
        println(io, "        line: { width: 0 }")
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
            traces_expr = "[traceTrialsGray, traceTrialsColor, traceKsTrueLine, traceCsTrueLine, traceKsTrueLineTop, traceCsTrueLineTop, traceTruth]"
        else
            traces_expr = "[traceTrialsGray, traceTrialsColor]"
        end
        println(io, "    const layout = {")
        println(io, "      title: 'AFM03 Stage1pluslight trials: ks_end vs cs_end vs val_loss (epoch ", target_epochs, ")',")
        println(io, "      paper_bgcolor: '#ffffff',")
        println(io, "      plot_bgcolor: '#ffffff',")
        println(io, "      legend: { orientation: 'h', x: 0, y: 1.02 },")
        println(io, "      margin: { l: 0, r: 0, t: 60, b: 0 },")
        println(io, "      annotations: [{")
        println(io, "        text: 'source: " * replace(result_path, "\\" => "\\\\") * "<br>post-gradient view: x=ks_hat, y=cs_hat, z=val_loss, color uses 0-150% NN err only; NN err > 150% shown in gray; hard-filtered to is_viable && !early_stopped && !trial_failed && completed_epochs==" * string(target_epochs) * "; z-axis tightened to q01-q99 of val_loss among plotted points; excluded " * string(excluded_count) * " trials with val_loss > 1e-1',")
        println(io, "        x: 0, y: 1.08, xref: 'paper', yref: 'paper',")
        println(io, "        xanchor: 'left', yanchor: 'bottom',")
        println(io, "        showarrow: false, font: { size: 12, color: '#444444' }")
        println(io, "      }],")
        println(io, "      scene: {")
        println(io, "        xaxis: { title: 'ks', tickmode: 'array', tickvals: ", js_array(xtickvals), ", ticktext: ", js_array(xticktext), ", range: [", repr(xlo), ", ", repr(xhi), "] },")
        println(io, "        yaxis: { title: 'cs', tickmode: 'array', tickvals: ", js_array(ytickvals), ", ticktext: ", js_array(yticktext), ", range: [", repr(ylo), ", ", repr(yhi), "] },")
        println(io, "        zaxis: { title: 'val_loss', tickmode: 'array', tickvals: ", js_array(ztickvals), ", ticktext: ", js_array(zticktext), ", range: [", repr(zfloor), ", ", repr(ztop), "] },")
        println(io, "        camera: { eye: { x: 1.5, y: 1.32, z: 0.92 } }")
        println(io, "      }")
        println(io, "    };")
        println(io, "    Plotly.newPlot('plot', ", traces_expr, ", layout, {responsive: true});")
        println(io, "  </script>")
        println(io, "</body>")
        println(io, "</html>")
    end

    println("Saved 3D HTML to: ", html_path)
    println("Trials plotted: ", length(kept_records))
    println("Trials excluded by z cutoff (>1e-1): ", excluded_count)
    println("Target completed_epochs: ", target_epochs)
    println("NN err color split: <=150% => ", length(color_records), " | >150% => ", length(gray_records))
    println("val_loss visible range: q01=", @sprintf("%.6e", zq01_raw),
        " | q99=", @sprintf("%.6e", zq99_raw),
        " | axis=[", @sprintf("%.6e", zfloor), ", ", @sprintf("%.6e", ztop), "]")
end

main()
