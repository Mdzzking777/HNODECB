ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Serialization
using Printf

field_or(rec, name::Symbol, default) = hasproperty(rec, name) ? getproperty(rec, name) : default
js_array(x) = repr(x)

function hover_text(rank::Int, rec)
    parts = rec.val_parts
    trial_id = get(rec.params, "trial_id", -1)
    return @sprintf(
        "Rank %d<br>trial=%d<br>val_loss=%.6e<br>train_loss=%.6e<br>ks=%.6e<br>cs=%.6e<br>ks err=%.2f%%<br>cs err=%.2f%%<br>F_contact err=%.2f%%<br>state=%.6e<br>x2dot=%.6e<br>x1 rec err=%.2f%%<br>x3 rec err=%.2f%%",
        rank,
        trial_id,
        rec.val_loss,
        rec.train_loss,
        rec.ks_hat,
        rec.cs_hat,
        rec.ks_err_pct,
        rec.cs_err_pct,
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
    haskey(data, :selected) || error("Result file does not contain :selected top-k records.")
    records = data.selected
    isempty(records) && error("No selected records found in: " * result_path)

    xs = Float64[field_or(rec, :ks_hat, NaN) for rec in records]
    ys = Float64[field_or(rec, :cs_hat, NaN) for rec in records]
    zs = Float64[field_or(rec, :val_loss, NaN) for rec in records]
    ferrs = Float64[field_or(rec, :val_nn_err, NaN) for rec in records]
    labels = ["R" * string(i) for i in eachindex(records)]
    hovertexts = [hover_text(i, records[i]) for i in eachindex(records)]

    true_ks = haskey(data, :true_values) ? field_or(data.true_values, :ks, NaN) : NaN
    true_cs = haskey(data, :true_values) ? field_or(data.true_values, :cs, NaN) : NaN

    zmin = minimum(filter(isfinite, zs))
    zmax = maximum(filter(isfinite, zs))
    zpad = max((zmax - zmin) * 0.12, max(abs(zmin), 1.0) * 1e-6)
    truth_z = zmin - zpad

    stem = splitext(basename(result_path))[1]
    html_path = joinpath(out_dir, stem * "_top10_ks_cs_loss_3d.html")

    open(html_path, "w") do io
        println(io, "<!DOCTYPE html>")
        println(io, "<html lang=\"en\">")
        println(io, "<head>")
        println(io, "  <meta charset=\"utf-8\">")
        println(io, "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
        println(io, "  <title>AFM03 Stage1pluslight top-10 ks-cs-loss 3D</title>")
        println(io, "  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>")
        println(io, "  <style>")
        println(io, "    body { margin: 0; font-family: sans-serif; background: #ffffff; }")
        println(io, "    #plot { width: 100vw; height: 100vh; }")
        println(io, "  </style>")
        println(io, "</head>")
        println(io, "<body>")
        println(io, "  <div id=\"plot\"></div>")
        println(io, "  <script>")
        println(io, "    const traceTop10 = {")
        println(io, "      type: 'scatter3d',")
        println(io, "      mode: 'markers+text',")
        println(io, "      name: 'top-10',")
        println(io, "      x: ", js_array(xs), ",")
        println(io, "      y: ", js_array(ys), ",")
        println(io, "      z: ", js_array(zs), ",")
        println(io, "      text: ", js_array(labels), ",")
        println(io, "      textposition: 'top center',")
        println(io, "      hovertext: ", js_array(hovertexts), ",")
        println(io, "      hovertemplate: '%{hovertext}<extra></extra>',")
        println(io, "      marker: {")
        println(io, "        size: 8,")
        println(io, "        color: ", js_array(ferrs), ",")
        println(io, "        colorscale: 'Turbo',")
        println(io, "        colorbar: { title: 'F_contact err %' },")
        println(io, "        line: { color: '#111111', width: 1 }")
        println(io, "      }")
        println(io, "    };")
        if isfinite(true_ks) && isfinite(true_cs)
            println(io, "    const traceTruth = {")
            println(io, "      type: 'scatter3d',")
            println(io, "      mode: 'markers+text',")
            println(io, "      name: 'true (ks, cs)',")
            println(io, "      x: [", repr(true_ks), "],")
            println(io, "      y: [", repr(true_cs), "],")
            println(io, "      z: [", repr(truth_z), "],")
            println(io, "      text: ['truth'],")
            println(io, "      textposition: 'bottom center',")
            println(io, "      hovertemplate: 'true ks=", @sprintf("%.6e", true_ks), "<br>true cs=", @sprintf("%.6e", true_cs), "<extra></extra>',")
            println(io, "      marker: { size: 7, symbol: 'diamond', color: '#111111' }")
            println(io, "    };")
            traces_expr = "[traceTop10, traceTruth]"
        else
            traces_expr = "[traceTop10]"
        end
        println(io, "    const layout = {")
        println(io, "      title: 'AFM03 Stage1pluslight top-10: ks vs cs vs val_loss',")
        println(io, "      paper_bgcolor: '#ffffff',")
        println(io, "      plot_bgcolor: '#ffffff',")
        println(io, "      legend: { orientation: 'h', x: 0, y: 1.02 },")
        println(io, "      margin: { l: 0, r: 0, t: 60, b: 0 },")
        println(io, "      annotations: [{")
        println(io, "        text: 'source: " * replace(result_path, "\\" => "\\\\") * "',")
        println(io, "        x: 0, y: 1.08, xref: 'paper', yref: 'paper',")
        println(io, "        xanchor: 'left', yanchor: 'bottom',")
        println(io, "        showarrow: false, font: { size: 12, color: '#444444' }")
        println(io, "      }],")
        println(io, "      scene: {")
        println(io, "        xaxis: { title: 'ks' },")
        println(io, "        yaxis: { title: 'cs' },")
        println(io, "        zaxis: { title: 'val_loss' },")
        println(io, "        camera: { eye: { x: 1.55, y: 1.35, z: 0.9 } }")
        println(io, "      }")
        println(io, "    };")
        println(io, "    Plotly.newPlot('plot', ", traces_expr, ", layout, {responsive: true});")
        println(io, "  </script>")
        println(io, "</body>")
        println(io, "</html>")
    end

    println("Saved 3D HTML to: ", html_path)
    println("Top-10 points plotted: ", length(records))
end

main()
