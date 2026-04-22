ENV["HNODECB_STAGE1_ARCHSCREEN_DEFAULT_PATH"] = get(ENV, "HNODECB_STAGE1_ARCHSCREEN_DEFAULT_PATH",
  joinpath(@__DIR__, "..", "..", "..", "..",
    "step2a_hyperparameter_tuning", "hyperparameter_tuning_first_stage",
    "results_afm", "afm_param_stage1_03.jld"))
ENV["HNODECB_STAGE1_ARCHSCREEN_DEFAULT_OUTDIR"] = get(ENV, "HNODECB_STAGE1_ARCHSCREEN_DEFAULT_OUTDIR",
  joinpath(@__DIR__, "..", "..", "..", "..", "logs", "stage1_step2a", "local", "visualization"))
ENV["HNODECB_STAGE1_ARCHSCREEN_HTML_TITLE"] = get(ENV, "HNODECB_STAGE1_ARCHSCREEN_HTML_TITLE",
  "AFM03 Stage1 archi screening Pareto front")
ENV["HNODECB_STAGE1_ARCHSCREEN_PLOT_TITLE"] = get(ENV, "HNODECB_STAGE1_ARCHSCREEN_PLOT_TITLE",
  "AFM03 Stage1 archi screening Pareto front (unweighted axes)")

include(normpath(joinpath(@__DIR__, "..", "stage1pluslight", "plot_afm_stage1plus_archscreen_pareto_03.jl")))
