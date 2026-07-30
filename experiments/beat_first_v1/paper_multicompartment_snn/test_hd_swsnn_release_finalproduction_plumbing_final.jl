# Windows ACLs prevented updating the newly added v2 file in place.  Evaluate
# its exact source after the single parser-only range formatting correction.
const _PLUMBING_V2_PATH = joinpath(
    @__DIR__,
    "test_hd_swsnn_release_finalproduction_plumbing_v2.jl",
)
const _PLUMBING_V2_SOURCE = read(_PLUMBING_V2_PATH, String)
const _PLUMBING_V2_FIXED = replace(
    _PLUMBING_V2_SOURCE,
    "firstindex(_PLUMBING_SOURCE):\n        " *
    "prevind(_PLUMBING_SOURCE, first(_PLUMBING_BOUNDARY))" =>
    "firstindex(_PLUMBING_SOURCE):prevind(" *
    "_PLUMBING_SOURCE, first(_PLUMBING_BOUNDARY))",
)
include_string(
    Main,
    _PLUMBING_V2_FIXED,
    _PLUMBING_V2_PATH * ":final",
)
