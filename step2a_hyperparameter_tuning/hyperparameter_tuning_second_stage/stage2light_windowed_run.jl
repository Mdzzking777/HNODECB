#=
Windowed Stage2light training entry for AFM03.

This keeps the official Stage2light training code path and warm-start logic,
but restricts the post-contact horizon to a contiguous stable-region window.

Defaults:
- stable-region window starting at a contact onset
- full post-contact point index start = 17124
- window length = 939 points (~15 us)

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
  ENV["HNODECB_STAGE2_WINDOW_START"] = "17124"
end

if env_missing("HNODECB_STAGE2_WINDOW_LEN")
  ENV["HNODECB_STAGE2_WINDOW_LEN"] = "939"
end

if env_missing("HNODECB_STAGE2_RESULT_BASENAME")
  ENV["HNODECB_STAGE2_RESULT_BASENAME"] = "afm_param_stage2light_windowed_03.jld"
end

include("afm_param_stage2_03.jl")
