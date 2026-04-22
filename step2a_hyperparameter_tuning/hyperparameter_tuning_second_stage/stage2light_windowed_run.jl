#=
Windowed Stage2light training entry for AFM03.

This keeps the official Stage2light training code path and warm-start logic,
but restricts the post-contact horizon to a contiguous window.

Preferred mode:
- set HNODECB_STAGE2_WINDOW_AUTOMANIFEST=1 in the runner
- auto-build event-anchored two-cycle windows:
  first-contact, max-x1-peak-to-peak-change, tail-stable

Manual fallback defaults:
- post-contact point index start = 1
- window length = 399 points (~two drive periods)

Override with:
- HNODECB_STAGE2_WINDOW_START
- HNODECB_STAGE2_WINDOW_LEN
- HNODECB_STAGE2_RESULT_BASENAME
=#

cd(@__DIR__)

function env_missing(name)
  !haskey(ENV, name) || strip(get(ENV, name, "")) == ""
end

if env_missing("HNODECB_STAGE2_WINDOW_START")
  ENV["HNODECB_STAGE2_WINDOW_START"] = "1"
end

if env_missing("HNODECB_STAGE2_WINDOW_LEN")
  ENV["HNODECB_STAGE2_WINDOW_LEN"] = "399"
end

if env_missing("HNODECB_STAGE2_RESULT_BASENAME")
  ENV["HNODECB_STAGE2_RESULT_BASENAME"] = "afm_param_stage2light_windowed_03.jld"
end

include("afm_param_stage2_03.jl")
